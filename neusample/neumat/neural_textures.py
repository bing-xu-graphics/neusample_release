from math import sqrt
import torch
import numpy as np
import utils

def fmod(x):
    x = torch.remainder(x, 1.0)
    return x
def fmod_minus1_1(x):
    x = x * 0.5 + 0.5
    x = fmod(x)
    x = x * 2 - 1
    return x
def pad_for_tiling(x):
    x = torch.cat([x[:,:,:,-1:], x, x[:,:,:,:1]], dim=-1)
    x = torch.cat([x[:,:,-1:,:], x, x[:,:,:1,:]], dim=-2)
    return x

# grid: [b, h, w, 2]
# Assumes the textures are tiled and adds rows and columns at the border accordingly.
def grid_sample_tiled(input, grid, mode='bilinear'):

    assert(input.shape[0] == grid.shape[0])
    assert(grid.shape[-1] == 2)

    grid = fmod_minus1_1(grid)
    pm = 'border'

    grid_x = grid[:,:,:,0:1]
    grid_y = grid[:,:,:,1:2]

    ac = True

    input = pad_for_tiling(input)
    height, width = input.shape[2:] # modified
    grid_x = grid_x*(width-2)/(width-1)
    grid_y = grid_y*(height-2)/(height-1)

    grid = torch.cat([grid_x, grid_y], dim=-1)

    result = torch.nn.functional.grid_sample(input, grid, mode=mode, padding_mode=pm, align_corners=ac)

    return result
def fetch2D(texture,uv,sigma):
    """ fetch neumip feature givne uv and sigma"""
    B,_ = uv.shape
    uv[...,1] = 1-uv[...,1]
    uv = uv*2-1
    ret = grid_sample_tiled(texture, uv.reshape(1,B,1,2)).reshape(-1,B).T
    #ret = NF.grid_sample(texture,uv.reshape(1,B,1,2),mode='bilinear',align_corners=True).reshape(-1,B).T
    return ret

class NeuralTextureSingle(torch.nn.Module):
    """
    Neural texture with a fixed resolution.
    """
    def __init__(self, num_channels, resolution, min_sigma, offset_texture=False, device="cuda:0") -> None:
        super().__init__()

        self.offset_texture = offset_texture

        if self.offset_texture:
            self.nt_outfile = "offset_texture.exr"
        else:
            self.nt_outfile = "neural_texture.exr"

        self.num_channels = num_channels
        self.min_sigma = min_sigma
        self.last_sigma = min_sigma

        res = np.array(list(resolution))
        res = [num_channels] + list(res.astype(int))

        self.texture = torch.randn(res, requires_grad=True)
        self.texture = self.texture * 0.01

        self.iteration = 0
        self.texture = torch.nn.Parameter(self.texture)

    def get_params(self):
        return [self.texture]

    def forward(self, location, sigma):
        neural_texture = self.texture.unsqueeze(0) # add dim for batch

        sigma = max(sigma, self.min_sigma) # min_sigma is non-zero for neural offset
        # print("~~~~~~~~~~~~~~~~~~~~~ hello there self.min_sigma =  sigma = ", self.min_sigma,sigma)

        # if self.training:
        #     self.last_sigma = sigma
        # else:
        sigma = self.last_sigma #just doing inference here

        neural_texture = utils.tensor.blur(sigma, neural_texture)
        # num_locations = location.shape[0]
        # loc_w = loc_h = int(sqrt(num_locations))
        # # loc = location.squeeze().clone()
        # loc = location.squeeze(1).squeeze(1).clone()
        # loc[:,1] = 1.0 - loc[:,1] # Y-up
        # loc = loc * 2.0 - 1.0
        # loc = loc.reshape(neural_texture.shape[0], loc_h, loc_w, loc.shape[-1])
        # result = utils.tensor.grid_sample_tiled(neural_texture, loc)
        # result = result.squeeze(0)
        # result = result.permute(1, 2, 0)
        # result = result.reshape(num_locations, -1)
        result = fetch2D(neural_texture, location.squeeze(1).squeeze(1), sigma)

        # Write out neural texture every few steps
        # TODO: Move to a callback
        # if True:
        #     if self.iteration % 100 == 0:
        #         with torch.no_grad():
        #             outres = self.texture[:3,:,:].permute(1, 2, 0)
        #             utils.exr.write16(np.array(outres.cpu(), dtype=np.float32), self.nt_outfile)

        return result

def calculate_neural_offset(ray_dir, neural_depth):
    def compute_z_from_xy(x):
        if x is None:
            return None
        x_a = x[:, 0:1]
        x_b = x[:, 1:2]
        result = torch.sqrt((1 - (x_a*x_a + x_b * x_b)).clamp(min=0))
        return result

    dir_z = compute_z_from_xy(ray_dir)
    dir_z = torch.clamp(dir_z, min=0.6) # 0.6 is from the paper (eq. 7)!

    neural_offset = neural_depth * ray_dir / dir_z
    return neural_offset
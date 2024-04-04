import torch
import torch.nn as nn
import torch.nn.functional as NF
import numpy as np

OFFSET_NW_INPUT_CHANNELS = 2 # UV
RGB_NW_INPUT_CHANNELS = 4 # cam + light direction
RGB_NW_OUT_CHANNELS = 3
OFFSET_NW_OUT_CHANNELS = 1
MLP_HIDDEN_LAYER_NEURONS = 32


def blur1d(sigma, x, hor=False, wrap=True, sm=1.5):
    num_ch = x.shape[-3]

    radius = int(np.ceil(sm*sigma))

    if sigma < 0.2:
        return x

    xx = np.linspace(-radius, radius, 2*radius+1)

    kernel = np.exp(- (xx*xx) / (sigma*sigma*2))

    if hor:
        exp_dim = -2
        padding = (0, radius)
    else:
        exp_dim = -1
        padding = (radius, 0)

    if wrap:
        padding = 0

    kernel = kernel/kernel.sum()
    kernel = torch.tensor(kernel, device=x.device,dtype=x.dtype).unsqueeze(0).unsqueeze(0).unsqueeze(exp_dim)

    big_kernel = torch.zeros([num_ch,num_ch, kernel.shape[-2], kernel.shape[-1]], device=x.device,dtype=x.dtype)
    for idx in range(num_ch):
        big_kernel[idx:idx+1, idx:idx+1, :, :] = kernel

    if wrap:
        if hor:
            size = x.shape[-1]
        else:
            size = x.shape[-2]

        repeat = int(radius/size)
        rr = radius%size
        repeat = repeat*2 + 1
        items = [x] * repeat

        if hor:
            items.append(x[:,:,:,:rr])
            if rr >0:
                items = [x[:,:,:,-rr:]] + items
            x = torch.cat(items, -1)
        else:
            items.append(x[:,:,:rr,:])
            if rr >0:
                items = [x[:,:,-rr:,:]] + items

            x = torch.cat(items, -2)

        x = torch.nn.functional.conv2d(x, big_kernel, padding=padding)

    return x

def blur(sigma, x, wrap=True, sm=3):

    x = blur1d(sigma, x, False, wrap, sm)
    x = blur1d(sigma, x, True, wrap, sm)

    return x

def fetch2D(texture,uv,sigma):
    """ fetch neumip feature givne uv and sigma"""
    B,_ = uv.shape
    texture = blur(sigma, texture[None])
    uv[...,1] = 1-uv[...,1]
    uv = uv*2-1
    ret = grid_sample_tiled(texture, uv.reshape(1,B,1,2)).reshape(-1,B).T
    #ret = NF.grid_sample(texture,uv.reshape(1,B,1,2),mode='bilinear',align_corners=True).reshape(-1,B).T
    return ret


class ThreeLayerMLP(nn.Module):
    """Classic MLP with three hidden layers"""
    def __init__(self, num_inputs, num_outputs) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, MLP_HIDDEN_LAYER_NEURONS),
            nn.LeakyReLU(),
            nn.Linear(MLP_HIDDEN_LAYER_NEURONS, MLP_HIDDEN_LAYER_NEURONS),
            nn.LeakyReLU(),
            nn.Linear(MLP_HIDDEN_LAYER_NEURONS, MLP_HIDDEN_LAYER_NEURONS),
            nn.LeakyReLU(),
            nn.Linear(MLP_HIDDEN_LAYER_NEURONS, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)

class NeuMIP(nn.Module):
    def __init__(self,res=(512,512)):
        super(NeuMIP,self).__init__()
        self.offset_network = ThreeLayerMLP(8+2,1)
        self.rgb_network = ThreeLayerMLP(8+4,3)
        self.register_parameter('offset_texture',nn.Parameter(torch.randn(8,*res)*0.01))
        self.register_parameter('rgb_texture',nn.Parameter(torch.randn(8,*res)*0.01))
    
    def get_rgb_texture(self,uv,wo=None,sigma_rgb=0.5,sigma_offset=2.0):
        """ get rgb texture for sampler
        Args:
            uv: Bx2 uv location in [0,1]
            wo: Bx3 viewing direction, if None, the texture will not be offset
        Return:
            Bx8 rgb feature vector
        """
        B = uv.shape[0]

        if wo is not None:
            offset = fetch2D(self.offset_texture,uv.clone(),sigma_offset)
            r = self.offset_network(torch.cat([offset,wo[...,:2]],1))
            uv_offset = (r/wo[...,2:].clamp_min(0.6))*wo[...,:2]
            uv = (uv_offset + uv)
            uv = uv % 1

        texture = fetch2D(self.rgb_texture,uv,sigma_rgb)
        return texture
    
    
    def get_brdf(self,uv,wo,wi,offset=True,sigma_rgb=0.5,sigma_offset=2.0):
        """  btf query """
        B = uv.shape[0]
        if offset:
            offset = fetch2D(self.offset_texture,uv.clone(),sigma_offset)
            r = self.offset_network(torch.cat([offset,wo[...,:2]],1))
            uv_offset = (r/wo[...,2:].clamp_min(0.6))*wo[...,:2]
            uv = (uv_offset + uv)
            uv = uv % 1

        texture = fetch2D(self.rgb_texture,uv,sigma_rgb)
        brdf = self.rgb_network(torch.cat([
            wi[...,:2],wo[...,:2],texture
        ],-1)).relu()
        return brdf

    def get_lobe(self,uv,wo,res,offset=True,sigma_rgb=0.5,sigma_offset=2.0):
        """ get lobe get given uv,wo
        Args:
            uv: Bx2
            wo: Bx3
            res: lobe image resolution
            offset: whether use offset network
        Return:
            Bxresxresx3 lobe image
        """
        B = uv.shape[0]
        if offset:
            offset = fetch2D(self.offset_texture,uv.clone(),sigma_offset)
            r = self.offset_network(torch.cat([offset,wo[...,:2]],1))
            uv_offset = (r/wo[...,2:].clamp_min(0.6))*wo[...,:2]
            uv = (uv_offset + uv)
            uv = uv % 1
        texture = fetch2D(self.rgb_texture,uv,sigma_rgb)
        y,x = torch.meshgrid(
            torch.linspace(-1,1,res,device=uv.device),
            torch.linspace(-1,1,res,device=uv.device)
        )
        wi = torch.stack([x,y],-1) #x,y
  
        z = (1-wi.pow(2).sum(-1))
        valid = (z>=0)
        wi = wi.reshape(-1,2)
        
        brdf = self.rgb_network(torch.cat([
            wi[None].expand(B,res*res,2),
            wo[:,None,:2].expand(B,res*res,2),
            texture[:,None].expand(B,res*res,texture.shape[-1])
        ],-1).reshape(B*res*res,-1)).relu().reshape(B,res,res,3)
        brdf = brdf * valid[None,:,:,None]
        return brdf
    
    
    def forward(self,uv,wo,wi,sigma_rgb=0.5,sigma_offset=2.0,tiled=True):
        """ query the neumip btf
        Args:
            uv: Bx2
            wo: Bx2
            wi: Bx2
            sigma: blur kernel size
            tiled: whether the uv is warped to 0,1
        """
        B,_ = uv.shape
        #if tiled:
        #    uv = uv%1
        offset = fetch2D(self.offset_texture,uv.clone(),sigma_offset)
        r = self.offset_network(torch.cat([offset,wo[...,:2]],1))
        uv_offset = (r/wo[...,2:].clamp_min(0.6))*wo[...,:2]
        uv_new = (uv_offset + uv)
        if tiled:
            uv_new = uv_new%1
        texture = fetch2D(self.rgb_texture,uv_new.clone(),sigma_rgb)
        brdf = self.rgb_network(torch.cat([wi[...,:2],wo[...,:2],texture],1)).relu()
        return {
            'r': r,
            'brdf': brdf
        }
        
        
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
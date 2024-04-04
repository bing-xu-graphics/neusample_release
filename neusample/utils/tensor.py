import torch
import numpy as np

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

class Type:
    def __init__(self, x):
        self.shape = x.shape
        self.device = x.device
        self.dtype = x.dtype

    def same_type(self):
        return {"device": self.device,
                "dtype": self.dtype
                }

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
    kernel = torch.tensor(kernel, **Type(x).same_type()).unsqueeze(0).unsqueeze(0).unsqueeze(exp_dim)

    big_kernel = torch.zeros([num_ch,num_ch, kernel.shape[-2], kernel.shape[-1]], **Type(x).same_type())
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

def get_masks(probabilities, shape):
    probabilities = np.array(probabilities)
    cum_prob = np.cumsum(probabilities)
    cum_prob = cum_prob/cum_prob[-1]
    cum_prob = np.insert(cum_prob, 0, 0., axis=0)

    rand = torch.rand(shape) #, **type.same_type())
    masks = []

    for i in range(len(cum_prob)-1):
        mask = torch.logical_and(cum_prob[i] < rand, rand <= cum_prob[i+1])
        masks.append(mask)

    return masks
import numpy as np
import torch
from torch import nn

from ..base import Flow
from ..reshape import Split, Merge
from normflows.nets import MLP
from normflows.utils import encoding
ENCODING_WI = False
    
class AffineCouplingNTCond(Flow):
    """
    Affine Coupling layer as introduced RealNVP paper, see arXiv: 1605.08803
    """

    def __init__(self,param_map, scale=True, scale_map="exp"):
        """
        Constructorgitfsdfv
        :param param_map: Maps features to shift and scale parameter (if applicable)
        :param scale: Flag whether scale shall be applied
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow, 'sigmoid_inv' uses multiplicative sigmoid
        scale when sampling from the model
        """
        super().__init__()
        
        self.scale = scale
        self.scale_map = scale_map
        # param_map = MLP([1+vector_dim, 32, 32, 2], init_zeros=True)
        self.add_module("param_map", param_map)
        # self.param_map2 = MLP([2, 8, 8, 2], leaky=0.01) ###TODO

    def affine_encode_wi(self, wi, conditional_vec):
        # param = self.param_map2(conditional_vec[:,:2])
        # wi = wi*param[:,0::2] +param[:,1::2]
        wi = encoding.positional_encoding_1(wi)
        return wi

    def forward(self, z, cond_vector):
        """
        z is a list of z1 and z2; z = [z1, z2]
        z1 is left constant and affine map is applied to z2 with parameters depending
        on z1
        """
        z1, z2 = z
        if ENCODING_WI:
            # encoded_z1 = positional_encoding(z1, POSITIONAL_ENCODING_BASIS_NUM)
            encoded_z1 = self.affine_encode_wi(z1, cond_vector)
            z1_cond = torch.cat([z1,encoded_z1, cond_vector], dim=1)
        else:
            z1_cond = torch.cat([z1, cond_vector], dim=1)
        param = self.param_map(z1_cond)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            # scale_ = self.param_map.scalenet(scale_) ##added
            if self.scale_map == "exp":
                z2 = z2 * torch.exp(scale_) + shift
                log_det = torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 / scale + shift
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 * scale + shift
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 += param
            log_det = 0
        return [z1, z2], log_det

    def inverse(self, z, cond_vector):
        z1, z2 = z
        if ENCODING_WI:
            # encoded_z1 = positional_encoding(z1, POSITIONAL_ENCODING_BASIS_NUM)
            # with torch.no_grad():
            encoded_z1 = self.affine_encode_wi(z1, cond_vector)
            z1_cond = torch.cat([z1,encoded_z1, cond_vector], dim=1)
        else:
            z1_cond = torch.cat([z1, cond_vector], dim=1)

        param = self.param_map(z1_cond)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            # scale_ = self.param_map.scalenet(scale_) ##added
            if self.scale_map == "exp":
                z2 = (z2 - shift) * torch.exp(-scale_)
                log_det = -torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) * scale
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) / scale
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 -= param
            log_det = 0
        return [z1, z2], log_det


class SplitCond(Flow):
    """
    Split features into two sets
    """

    def __init__(self, mode="channel"):
        """
        Constructor
        :param mode: Splitting mode, can be
            channel: Splits first feature dimension, usually channels, into two halfs
            channel_inv: Same as channel, but with z1 and z2 flipped
            checkerboard: Splits features using a checkerboard pattern (last feature dimension must be even)
            checkerboard_inv: Same as checkerboard, but with inverted coloring
        """
        super().__init__()
        self.mode = mode

    def forward(self, z, y=None):
        if self.mode == "channel":
            z1, z2 = z.chunk(2, dim=1)
        elif self.mode == "channel_inv":
            z2, z1 = z.chunk(2, dim=1)
        elif "checkerboard" in self.mode:
            n_dims = z.dim()
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z.size(n_dims - i))]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z.size(n_dims - i))]
            cb = cb1 if "inv" in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(len(z), *((n_dims - 1) * [1]))
            cb = cb.to(z.device)
            z_size = z.size()
            z1 = z.reshape(-1)[torch.nonzero(cb.view(-1), as_tuple=False)].view(
                *z_size[:-1], -1
            )
            z2 = z.reshape(-1)[torch.nonzero((1 - cb).view(-1), as_tuple=False)].view(
                *z_size[:-1], -1
            )
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = 0
        return [z1, z2], log_det

    def inverse(self, z, y=None):
        z1, z2 = z
        if self.mode == "channel":
            z = torch.cat([z1, z2], 1)
        elif self.mode == "channel_inv":
            z = torch.cat([z2, z1], 1)
        elif "checkerboard" in self.mode:
            n_dims = z1.dim()
            z_size = list(z1.size())
            z_size[-1] *= 2
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z_size[n_dims - i])]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z_size[n_dims - i])]
            cb = cb1 if "inv" in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(z_size[0], *((n_dims - 1) * [1]))
            cb = cb.to(z1.device)
            z1 = z1[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z2 = z2[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z = cb * z1 + (1 - cb) * z2
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = 0
        return z, log_det


class MergeCond(SplitCond):
    """
    Same as Split but with forward and backward pass interchanged
    """

    def __init__(self, mode="channel"):
        super().__init__(mode)

    def forward(self, z, y=None):
        return super().inverse(z, y)

    def inverse(self, z, y=None):
        return super().forward(z,y)


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)



class AffineCouplingNTCondBlock(Flow):
    """
    Affine Coupling with Neural texture condition layer including split and merge operation
    """

    def __init__(self,param_map, scale=True, scale_map="exp", split_mode="channel"):
        super().__init__()

        #build the normalizing flows
        self.flows = nn.ModuleList([])
        # Split layer
        self.flows += [SplitCond(split_mode)]
        # Affine coupling layer
        self.flows += [AffineCouplingNTCond(param_map, scale, scale_map)]
        # Merge layer
        self.flows += [MergeCond(split_mode)]

    def forward(self, z, conditional_vector):

        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z, conditional_vector)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z, conditional_vector):
        
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, conditional_vector)
            log_det_tot += log_det
        return z, log_det_tot


import torch
import torch.nn as nn
import torch.nn.functional as NF
import math

class PositionalEncoding(nn.Module):
    def __init__(self, L):
        """ L: number of frequency bands """
        super(PositionalEncoding, self).__init__()
        self.L= L
        
    def forward(self, inputs):
        L = self.L
        encoded = [inputs]
        for l in range(L):
            encoded.append(torch.sin((2 ** l * math.pi) * inputs))
            encoded.append(torch.cos((2 ** l * math.pi) * inputs))
        return torch.cat(encoded, -1)


def mlp(dim_in, dims, dim_out):
    """ create an MLP in format: dim_in->dims[0]->...->dims[-1]->dim_out"""
    lists = []
    dims = [dim_in] + dims
    
    for i in range(len(dims)-1):
        lists.append(nn.Linear(dims[i],dims[i+1]))
        lists.append(nn.ReLU(inplace=True))
    lists.append(nn.Linear(dims[-1], dim_out))
    
    return nn.Sequential(*lists)


class ImplicitMLP(nn.Module):
    """ implicit voxel feature MLP"""
    def __init__(self, D, C, S, dim_out, dim_enc,dim_in=3):
        """
        Args:
            D: depth of the MLP
            C: intermediate feature dim
            S: skip layers
            dim_out: out feature dimension
            dim_enc: frequency band of positional encdoing for the vertex location
        """
        super(ImplicitMLP, self).__init__()
        self.input_ch = dim_enc * 2 * dim_in + dim_in
       
        self.D = D
        self.C = C
        self.skips = S
        self.dim_out = dim_out
        
        self.point_encode = PositionalEncoding(dim_enc)
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.C)] + 
            [
                nn.Linear(self.C, self.C) 
                if i not in self.skips 
                else nn.Linear(self.C + self.input_ch, self.C) 
                for i in range(1, self.D) 
            ]
        )
        self.feature_linear = nn.Linear(self.C, self.dim_out)
        
    def forward(self, x):
        """ 
        Args:
            x: Bx3 3D vertex locations
        Return:
            Bx3 corresponding feature vectors
        """
        points = self.point_encode(x)
        p = points
        for i, l in enumerate(self.pts_linears):
            if i in self.skips:
                p = torch.cat([points, p], 1)
            p = l(p)
            p = NF.relu(p)
        feature = self.feature_linear(p)
        return feature

    
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]
    
def sh(dirs, degree=4):
    N = dirs.shape[:-1]
    basis_dim = (degree+1)*(degree+1)
    result = torch.empty(*N,basis_dim,device=dirs.device)
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y ;
        result[..., 2] = SH_C1 * z;
        result[..., 3] = -SH_C1 * x;
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy;
            result[..., 5] = SH_C2[1] * yz;
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = SH_C2[3] * xz;
            result[..., 8] = SH_C2[4] * (xx - yy);

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy) ;
                result[..., 10] = SH_C3[1] * xy * z ;
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy) ;
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) ;
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy) ;
                result[..., 14] = SH_C3[5] * z * (xx - yy) ;
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy) ;

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy) ;
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy) ;
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1) ;
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3) ;
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3) ;
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3) ;
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1) ;
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy) ;
                    result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) ;
    return result
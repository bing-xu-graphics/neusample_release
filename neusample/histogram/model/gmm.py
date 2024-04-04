import torch
import torch.nn as nn
import torch.nn.functional as NF

import math
from scipy.spatial.transform import Rotation

import sys
sys.path.append('..')
from model.mlps import mlp



def positional_encoding_1(
    tensor, num_encoding_functions=6, log_sampling=True
):
    encoding = [tensor]
    
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
                0.0,
                num_encoding_functions - 1,
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        
    for freq in frequency_bands:
        encoding.append(torch.sin(tensor * freq))
        encoding.append(torch.cos(tensor * freq))

    return torch.cat(encoding, dim=-1)

class GMM(nn.Module):
    def __init__(self,K=2):
        super(GMM,self).__init__()
        self.K = K
        self.mlp = mlp(30,[32],self.K*2*2+self.K+1)
    
    def forward(self,wi,cond,inverse=False):
        cond =  torch.cat([positional_encoding_1(cond[...,-2:],5,log_sampling=True),
                           cond[...,:-2]],-1)
        
        B = cond.shape[0]
        ret = self.mlp(cond)
        mu = ret[...,:self.K*2].reshape(-1,self.K,2)
        log_sigma = ret[...,self.K*2:self.K*4].reshape(-1,self.K,2)
        sigma = torch.exp(log_sigma)
        weight = ret[...,self.K*4:].reshape(-1,self.K+1).abs()
        weight = weight/weight.sum(-1,keepdim=True).clamp_min(1e-12)

        if inverse:
            lobe_idx = torch.searchsorted(
                weight.cumsum(-1).contiguous(),wi[...,2:].clamp_min(1e-12))
            wi = wi[...,:2]
            #mask = torch.zeros_like(lobe_idx).bool().squeeze(-1)
            mask = (lobe_idx >= self.K).squeeze(-1)
            # box muller
            lobe_idx = lobe_idx.reshape(B,1,1).expand(B,1,2).clamp(0,self.K-1)
            u1,u2 = wi[...,0],wi[...,1]
            z0 = (-2*torch.log(u1.clamp_min(1e-12))).sqrt()
            z1 = torch.sin(2*math.pi*u2)*z0
            z0 = torch.cos(2*math.pi*u2)*z0
            z = torch.stack([z0,z1],-1)

            wi_out = z*torch.gather(sigma,1,lobe_idx).squeeze(1)\
              + torch.gather(mu,1,lobe_idx).squeeze(1)
            
            wi = wi[mask]
            theta = torch.asin(wi[...,0].sqrt())
            phi = 2*math.pi*wi[...,1]
            sin_theta = torch.sin(theta)
            wi_out[mask] = torch.stack([sin_theta*torch.cos(phi),sin_theta*torch.sin(phi)],-1)
            wi = wi_out
        
        
        rr = (wi[:,None]-mu)/sigma
        rr = (rr*rr).sum(2)
        fac = torch.exp(-log_sigma.sum(2))/(2*math.pi)

        pdf = (weight[...,:-1]*torch.exp(-0.5*rr)*fac).sum(1)
        pdf = pdf + weight[...,-1]*(wi.pow(2).sum(-1)<=1.0).float()/math.pi
        
        return wi,pdf
    
    
    
class BaseLine(nn.Module):
    def __init__(self,D=3):
        super(BaseLine,self).__init__()
        self.mlp = mlp(30,[32]*D,5)
    
    def forward(self,wi,cond,inverse=False):
        cond =  torch.cat([positional_encoding_1(cond[...,-2:],5,log_sampling=True),
                           cond[...,:-2]],-1)
        
        B = cond.shape[0]
        ret = self.mlp(cond)
        mu = ret[...,:2]
        log_sigma = ret[...,2:3]
        sigma = torch.exp(log_sigma)
        weight = ret[...,3:].abs()
        weight = weight/weight.sum(-1,keepdim=True).clamp_min(1e-12)

        if inverse:
            mask = (wi[...,2] > weight.cumsum(-1)[...,0])
            # box muller
            u1,u2 = wi[...,0],wi[...,1]
            z0 = (-2*torch.log(u1.clamp_min(1e-12))).sqrt()
            z1 = torch.sin(2*math.pi*u2)*z0
            z0 = torch.cos(2*math.pi*u2)*z0
            z = torch.stack([z0,z1],-1)
            wi_out = sigma*z+mu
            
            
            wi = wi[mask]
            theta = torch.asin(wi[...,0].sqrt())
            phi = 2*math.pi*wi[...,1]
            sin_theta = torch.sin(theta)
            wi_out[mask] = torch.stack([sin_theta*torch.cos(phi),sin_theta*torch.sin(phi)],-1)
            wi = wi_out
        
        
        rr = (wi-mu)/sigma
        rr = (rr*rr).sum(-1)
        fac = torch.exp(-log_sigma.squeeze(-1)*2)/(2*math.pi)
        pdf = (weight[...,0]*torch.exp(-0.5*rr)*fac)
        
        pdf = pdf + weight[...,-1]*(wi.pow(2).sum(-1)<=1.0).float()/math.pi
        
        return wi,pdf
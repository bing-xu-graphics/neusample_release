import torch
import torch.nn as nn
import torch.nn.functional as NF

import math

from .mlps import mlp,PositionalEncoding
import sys
sys.path.append('..')
from utils.cuda import sample_idx_texture_2d,fetch_idx_texture_2d

class Base2D(nn.Module):
    def __init__(self,res,D):
        super(Base2D,self).__init__()
        self.encode = PositionalEncoding(4)
        self.C = 64
        self.features0 = mlp(3*(4*2+1),[self.C]*4,self.C)
        self.features1 = nn.Linear(self.C,D)
        
        self.mlp = mlp(8+2,[32]*3,D*4)
        self.D = D
    
    def get_lobe(self,wi,cos_theta):
        B = len(cos_theta)
        x = torch.cat([wi,cos_theta[...,None]],-1)
        
        # BxDxC
        intermediate = self.features0(self.encode(x.reshape(B*self.D,3))).reshape(B,self.D,self.C).relu()
        
        # 3xDxC
        linear_weight = self.features1.weight.reshape(self.D,self.C)
        linear_bias = self.features1.bias.reshape(self.D)
        lobes = torch.einsum('bdc,dc->bd',intermediate,linear_weight) + linear_bias
        lobes = NF.softplus(lobes) # softplus
        return lobes
    
    def forward(self,wi,cond,vq=False):
        B = cond.shape[0]
        feat = self.mlp(cond)
        weight,v,cos_theta = feat[...,:self.D],feat[...,self.D:self.D*3].reshape(B,self.D,2),feat[...,self.D*3:]
        
        weight = weight.relu()
        cos_theta = cos_theta.sigmoid().reshape(B,self.D)
        
        v = NF.normalize(v,dim=-1)
        R = torch.stack([v[...,0],-v[...,1],
                        v[...,1],v[...,0]],-1).reshape(B,self.D,2,2)
        # BxDx2
        wi = torch.einsum('bdij,bj->bdi',R,wi)
        
        lobes = self.get_lobe(wi,cos_theta)
        ret = (lobes*weight).sum(-1)
        
        if vq:
            vq_theta = (cos_theta.detach()*100).round().clamp(0,99)/100.0
            vq_lobes = self.get_lobe(wi,vq_theta)
            vq_ret = (vq_lobes*weight).sum(-1)
            
            return {
                'cos_theta': cos_theta,
                'vq_theta': vq_theta,
                'pdf': ret,
                'vq_pdf': vq_ret
            }
            
            
        else:
        
            return {
                'cos_theta': cos_theta,
                'pdf': ret
            }  

class Base2DInference(nn.Module):
    def __init__(self,res,angular,D):
        super(Base2DInference,self).__init__()
        self.register_parameter('pdf',
            nn.Parameter(torch.randn(angular,D,res,res)*1e-2,requires_grad=True))
        self.register_parameter('fac',nn.Parameter(torch.randn(angular,D)*1e-2,requires_grad=True))
        self.mlp = mlp(8+2,[32]*3,D*4)
        self.res = res
        self.angular = angular
        self.D = D
        
    def init_pdf(self,):
        pdf_y = self.pdf.data.sum(-1)
        cdf_y = pdf_y.cumsum(-1).contiguous()
        pdf_x = self.pdf.data/pdf_y[:,:,:,None].clamp_min(1e-12)
        cdf_x = pdf_x.cumsum(-1).contiguous()
        self.register_buffer('cdf_y',cdf_y)
        self.register_buffer('cdf_x',cdf_x)
    
    def forward(self,wi,cond,inverse=False):
        if not inverse:
            B = len(cond)
            weight = self.mlp(cond)
            weight,v,z = weight[...,:self.D],weight[...,self.D:self.D*3].reshape(B,self.D,2),weight[...,self.D*3:]
            
            v = NF.normalize(v,dim=-1).contiguous()
            z = z.sigmoid().reshape(B,self.D)
            z_idx = (z*self.angular).round().clamp(0,self.angular-1).int().contiguous()
            weight = weight.relu()*torch.gather(self.fac,0,z_idx.long())
            weight = NF.normalize(weight,dim=-1,p=1).contiguous()

            pdf_out = torch.full((B,),-1.0,device=wi.device).contiguous()
            fetch_idx_texture_2d(z_idx,v,weight,self.pdf,wi.contiguous(),pdf_out)
            return None,pdf_out
            
        else:
            B = len(cond)
            weight = self.mlp(cond)
            weight,v,z = weight[...,:self.D],weight[...,self.D:self.D*3].reshape(B,self.D,2),weight[...,self.D*3:]
            
            v = NF.normalize(v,dim=-1).contiguous()
            
            z = z.sigmoid().reshape(B,self.D)
            z_idx = (z*self.angular).round().clamp(0,self.angular-1).int().contiguous()
            weight = weight.relu()*torch.gather(self.fac,0,z_idx.long())
            weight = NF.normalize(weight,dim=-1,p=1).contiguous()
            cdf_z = weight.cumsum(-1).contiguous()
            
            wi_out = torch.zeros(B,2,device=wi.device).contiguous()
            pdf_out = torch.zeros(B,device=wi.device).contiguous()
            
            sample_idx_texture_2d(
                z_idx,v,cdf_z,self.cdf_y,self.cdf_x,
                wi.contiguous(),wi_out.contiguous(),pdf_out)
            fetch_idx_texture_2d(z_idx,v,weight,self.pdf,wi_out,pdf_out)
            
            return wi_out,pdf_out
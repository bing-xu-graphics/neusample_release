import torch
import torch.nn as nn
import torch.nn.functional as NF

import math

from .mlps import mlp,PositionalEncoding
import sys
sys.path.append('..')
from utils.cuda import sample_batch_texture_2d

class Histogram2D(nn.Module):
    def __init__(self,res=16,mode='bilinear'):
        super(Histogram2D,self).__init__()
        self.mlp = mlp(8+2,[32]*3,res*res)
        self.res = res
        self.mode =mode
    
    def forward(self,wi,cond):
        B = wi.shape[0]
        hist = NF.softplus(self.mlp(cond))
        #hist = self.mlp(cond)
        #weight = hist[...,-1].sigmoid()
        #hist = NF.softplus(hist[...,:-1])
        hist = hist.reshape(B,1,self.res,self.res)
        
        if self.mode == 'bilinear':
            pdf = NF.grid_sample(hist,wi.reshape(B,1,1,2),mode='bilinear',align_corners=True)
        else:
            pdf = NF.grid_sample(hist,wi.reshape(B,1,1,2),mode='nearest',align_corners=False)
        pdf = pdf.reshape(B)
        #pdf = pdf*weight + (1-weight)*(wi[...,:2].pow(2).sum(-1)<=1).float()
        return pdf
        
        
class Histogram2DInference(nn.Module):
    def __init__(self,res=16,mode='nearest'):
        super(Histogram2DInference,self).__init__()
        self.mlp = mlp(8+2,[32]*3,res*res)
        self.res = res
        self.mode =mode
    
    def forward(self,wi,cond,inverse=False):

        B = wi.shape[0]
        hist = NF.softplus(self.mlp(cond))

        hist = hist.reshape(B,1,self.res,self.res)
        hist = hist / hist.sum([-1,-2],keepdim=True).clamp_min(1e-12)
        
        if not inverse:
            #if self.mode == 'bilinear':
            #    pdf = NF.grid_sample(hist,wi.reshape(B,1,1,2),mode='bilinear',align_corners=True)
            #else:
            pdf = NF.grid_sample(hist,wi.reshape(B,1,1,2),mode='nearest',align_corners=False)
            pdf = pdf.reshape(B)
            return None,pdf*self.res*self.res/4.0
        else:
            pdf_y = hist.sum(-1)
            cdf_y = pdf_y.cumsum(-1).contiguous()
            
            pdf_x = hist/pdf_y[...,None].clamp_min(1e-12)
            cdf_x = pdf_x.cumsum(-1).contiguous()
            
            wi_out = torch.zeros(B,2,device=wi.device).contiguous()
            pdf_out = torch.zeros(B,device=wi.device).contiguous()
            
            sample_batch_texture_2d(cdf_y,cdf_x,wi.contiguous(),wi_out,pdf_out)
            
            return wi_out,pdf_out 
        
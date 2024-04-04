import torch
import torch.nn.functional as NF
from torch.utils.data import Dataset
import json
import numpy as np
import os

class NeuMIPDataset(Dataset):
    def __init__(self, spatial, angular, split='train'):
        self.spatial = spatial
        self.angular = angular
        self.length = self.spatial*self.spatial*self.angular*self.angular
        self.spatial_size = 1.0/self.spatial
        self.angular_size = 1.0/self.angular
        self.split = split
        self.dump = torch.zeros(self.length)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ stratified sampling uv"""
        dump = self.dump[idx]
        angular_len = self.angular*self.angular
        
        sij = idx % angular_len
        ij = idx // angular_len
        si,sj = sij%self.angular,sij//self.angular
        i,j = ij%self.spatial,ij//self.spatial

        suv = torch.rand(2)*self.angular_size
        suv[0] += si * self.angular_size
        suv[1] += sj * self.angular_size
        suv = suv*2 - 1.0
        
        uv = torch.rand(2)*self.spatial_size
        uv[0] += i*self.spatial_size
        uv[1] += j*self.spatial_size
        
        # map square to sphere
        r1 = suv.abs().max(-1)[0]
        r2 = suv.norm(dim=-1,p=2)
        suv = suv * (r1/r2.clamp_min(1e-12))
        z = (1-suv.pow(2).sum(-1,keepdim=True).relu()).sqrt()
        wo = torch.cat([suv,z],dim=-1)
        wo = NF.normalize(wo,dim=-1)
        
        return {
            'uv': uv.reshape(2),
            'wo': wo.reshape(3)
        }
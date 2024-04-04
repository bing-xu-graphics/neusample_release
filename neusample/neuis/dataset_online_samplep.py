
import numpy as np
import torch
import torch.optim as optim
from sys import exit
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import struct
from pybind import samplewi
from utils import exr, ops, la2
from .sampling import Distribution2D

M_PI = 3.14159265359
PI_over_4 = 0.78539816339
PI_over_2 = 1.57079632679

torch.set_default_dtype(torch.float64)
NUM_UV_DIM = 512
NUM_WO_VARY = 512
NUM_WI_SAMPLES = 1024
WI_RES = 256
LOADING_SIZE = 18*18
# LOADING_SIZE = 12*12

from scripts import vis_helper

# torch.autograd.set_detect_anomaly(True)
class NTOnlineSamplingPDataset(Dataset):
    def __init__(self, vis_helper, uv_jitter = False):
        if not uv_jitter:
            u = torch.linspace(-1, 1, steps = NUM_UV_DIM)
            v = torch.linspace(-1, 1, steps = NUM_UV_DIM)
            grid_u, grid_v = torch.meshgrid(u, v)
            grid = torch.stack([grid_u, grid_v], dim = -1)
            self.uv_grid = grid.reshape((-1, 2))
        else:
            u = torch.arange(0, NUM_UV_DIM)/NUM_UV_DIM
            v = torch.arange(0, NUM_UV_DIM)/NUM_UV_DIM
            grid_u, grid_v = torch.meshgrid(u, v)
            grid = torch.stack([grid_u, grid_v], dim = -1).reshape((-1, 2))
            self.grid = grid + torch.rand((NUM_UV_DIM*NUM_UV_DIM, 2))/NUM_UV_DIM

        ### wi grids
        wiX = torch.linspace(-1.0,1.0, steps = WI_RES)
        wiY = torch.linspace(-1.0,1.0, steps = WI_RES)
        grid_z1, grid_z2 = torch.meshgrid(wiX, wiY)
        gridwi = torch.stack([grid_z1, grid_z2], dim = -1)
        light_dir = gridwi.reshape((-1, 2))
        invalid_dirs = torch.square(light_dir[...,0]) + torch.square(light_dir[...,1]) > 0.995 #TODO CHANGED here
        self.light_dir = light_dir
        self.invalid_dirs = invalid_dirs
        self.vis_helper = vis_helper
        self.A = 4/(WI_RES * WI_RES)

    def __len__(self):
        return NUM_UV_DIM*NUM_UV_DIM //LOADING_SIZE

    def sample_wo(self, num_samples):
        wo = torch.rand(num_samples, 2) * 2.0 - 1.0
        woSample = torch.zeros(wo.shape)
        zero_positions = torch.logical_and(wo[:, 0] == 0 , wo[:, 1]==0)
        nonzero_positions = ~zero_positions
        condition1 = torch.logical_and(torch.abs(wo[:,0]) > torch.abs(wo[:,1]),nonzero_positions)
        condition2 = torch.logical_and(~condition1 ,nonzero_positions)

        woSample[condition1,0] = wo[condition1,0] * torch.cos(PI_over_4 * wo[condition1,1]/wo[condition1,0])
        woSample[condition1,1] = wo[condition1,0] * torch.sin(PI_over_4 * wo[condition1,1]/wo[condition1,0])

        woSample[condition2,0] = wo[condition2,1] * torch.cos(PI_over_2 - PI_over_4 * wo[condition2,0]/wo[condition2,1])
        woSample[condition2,1] = wo[condition2,1] * torch.sin(PI_over_2 - PI_over_4 * wo[condition2,0]/wo[condition2,1])
        return woSample

    def stratified_sampling_2d(self, spp_):
        #round spp to square number
        torch.manual_seed(42)
        side = 1
        while (side*side< spp_):
            side += 1
        # spp_square = side*side
        # side = int(math.sqrt(spp_square))
        us = torch.arange(0, side)/side 
        vs = torch.arange(0, side)/side
        u, v = torch.meshgrid(us, vs)
        uv = torch.stack([u,v], dim = -1)

        uv = uv.reshape(-1,2)
        # uv = uv[:spp_,...]
        uv = uv[torch.randperm(uv.shape[0]), ...]  #TODO CHECK
        jitter = torch.rand((spp_, 2))/side
        # print("~~~~~~~~~~~~~~~~~~~~~~ stratified samples: ", uv.shape)
        return uv + jitter

    
    def stratified_sample_wo(self, num_samples):
        wo = self.stratified_sampling_2d(num_samples) * 2.0 - 1.0

        woSample = torch.zeros(wo.shape)
        zero_positions = torch.logical_and(wo[:, 0] == 0 , wo[:, 1]==0)
        nonzero_positions = ~zero_positions
        condition1 = torch.logical_and(torch.abs(wo[:,0]) > torch.abs(wo[:,1]),nonzero_positions)
        condition2 = torch.logical_and(~condition1 ,nonzero_positions)

        woSample[condition1,0] = wo[condition1,0] * torch.cos(PI_over_4 * wo[condition1,1]/wo[condition1,0])
        woSample[condition1,1] = wo[condition1,0] * torch.sin(PI_over_4 * wo[condition1,1]/wo[condition1,0])

        woSample[condition2,0] = wo[condition2,1] * torch.cos(PI_over_2 - PI_over_4 * wo[condition2,0]/wo[condition2,1])
        woSample[condition2,1] = wo[condition2,1] * torch.sin(PI_over_2 - PI_over_4 * wo[condition2,0]/wo[condition2,1])
        
        return woSample

    def one_sample_wo(self):
        wo = torch.rand(2)*2.0 - 1.0
        if ( wo[0] == 0 and wo[1] == 0):
            woX = 0
            woY = 0
        else:
            if(torch.abs(wo[0]) > torch.abs(wo[1])):
                r = wo[0]
                theta = PI_over_4 * (wo[1]/ wo[0])
            else:
                r = wo[1]
                theta = PI_over_2 - PI_over_4 * (wo[0]/wo[1])
            woX = r * torch.cos(theta)
            woY = r * torch.sin(theta)
        wo[0] = woX 
        wo[1] = woY
        return wo

    def importance_sample_wi(self, uv_wo, img,  num_samples):
        samples = list()
        resolution = len(img)
        distribution = Distribution2D(img, resolution, resolution)
        for i in range(num_samples):
            sampleWi, pdf = distribution.SampleContinuous(torch.rand(2))
            sampleWi[0] = sampleWi[0]*2.0 - 1.0
            sampleWi[1] = sampleWi[1]*2.0 -1.0
            if sampleWi[0]* sampleWi[0] + sampleWi[1] * sampleWi[1] > 1.0:
                print("one invalid")
            samples.append([*uv_wo, *sampleWi])
        return samples

    def __getitem__(self, idx):
        uv = self.uv_grid[idx*LOADING_SIZE: (idx+1)*LOADING_SIZE]
        # camera_dir = self.sample_wo(LOADING_SIZE)
        camera_dir = self.stratified_sample_wo(LOADING_SIZE)
        uv_wo_records = torch.cat((uv, camera_dir), dim = -1)
        uv_tensor = torch.tile(uv, (self.light_dir.shape[0], )).reshape(-1,2).cuda()
        camera_dir_tensor = torch.tile(camera_dir, (self.light_dir.shape[0], )).reshape(-1,2).cuda()
        light_dir_tensor = torch.tile(self.light_dir, (uv.shape[0], 1)).reshape(-1,2).cuda()
        with torch.no_grad():
            rgb_pred = self.vis_helper.neumip.model( camera_dir_tensor, light_dir_tensor, uv_tensor) #uv_tensor
            rgb_pred[rgb_pred < 0] = 0
        with torch.no_grad():
            rgb_pred = rgb_pred.reshape(LOADING_SIZE, WI_RES* WI_RES, -1)
            rgb_pred[:, self.invalid_dirs] = 0
            lumi_pred = self.vis_helper.rgb2lum_batch(rgb_pred)
            # print("self.invalid_dirs.shape ", self.invalid_dirs.shape)
            lumi_pred[:, self.invalid_dirs] = 0
            lumi_pred = lumi_pred.reshape(LOADING_SIZE, WI_RES, WI_RES)#.cpu()
            gt_pdf = lumi_pred.flatten().detach().cpu()
            wi_samples = torch.Tensor(samplewi.samplewi(gt_pdf, LOADING_SIZE, NUM_WI_SAMPLES)).reshape(-1, 2)

            uv_wo_tensor = torch.tile(uv_wo_records, (NUM_WI_SAMPLES,)).reshape(-1,4)
            wi_samples_in = torch.cat((uv_wo_tensor, wi_samples), dim=-1)
            
        wi_samples_in = torch.Tensor(wi_samples_in).to(torch.device("cuda"))
        idx = torch.randperm(wi_samples_in.shape[0])
        return wi_samples_in[idx]




import scipy
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
import numpy as np

import numpy as np
import torch
import torch.optim as optim
from sys import exit
import sys
import os
import math
# Insert root dir in path
p = os.path.abspath('.')
sys.path.insert(1, p)

from utils import exr
# import larsflow as lf
import cv2


M_PI = torch.pi 
PI_over_4 = torch.pi *4
PI_over_2 = torch.pi *2
INV_PI =    1/torch.pi

device_ = torch.device(0)

class LambertianLobe(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, num_samples=1, randseed=None): #sample
        wo = torch.rand(num_samples, 2, device=device_, dtype=torch.float) * 2.0 - 1.0  #[-1,1]
        
        woSample = torch.zeros(wo.shape, device=device_, dtype=torch.float)
        zero_positions = torch.logical_and(wo[:, 0] == 0 , wo[:, 1]==0) 
        nonzero_positions = ~zero_positions
        condition1 = torch.logical_and(torch.abs(wo[:,0]) > torch.abs(wo[:,1]),nonzero_positions)
        condition2 = torch.logical_and(~condition1 ,nonzero_positions)

        woSample[condition1,0] = wo[condition1,0] * torch.cos(PI_over_4 * wo[condition1,1]/wo[condition1,0])
        woSample[condition1,1] = wo[condition1,0] * torch.sin(PI_over_4 * wo[condition1,1]/wo[condition1,0])
        woSample[condition2,0] = wo[condition2,1] * torch.cos(PI_over_2 - PI_over_4 * wo[condition2,0]/wo[condition2,1])
        woSample[condition2,1] = wo[condition2,1] * torch.sin(PI_over_2 - PI_over_4 * wo[condition2,0]/wo[condition2,1])

        return woSample, torch.full((num_samples,), INV_PI).to(woSample.device)

    def prob(self, z, rm_invalid = True):
        invalid_mask = torch.square(z[:,0]) + torch.square(z[:,1]) >=1
        pdf = torch.full((z.shape[0],), INV_PI).to(z.device)
        if rm_invalid:
            pdf[invalid_mask] = 1e-10 ##TODO
        return pdf


class GMMIso(nn.Module):
    def __init__(
        self, n_modes, dim, loc=None, scale=None, weights=None, trainable=True
    ):
        super().__init__()
       
        self.lambertianLobe = LambertianLobe()
        for param in self.lambertianLobe.parameters():
            param.requires_grad = False

        self.n_modes = n_modes
        self.dim = dim

        if weights is None:
            weights = np.ones(self.n_modes+1)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)
        print(weights)

        # self.register_buffer("loc", torch.tensor(0.0 * loc))
        # self.register_buffer("log_scale", torch.tensor(np.log(1.0 * scale)))
        self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights)))

        self.loc = torch.zeros(2,device=device_, dtype=torch.float)
        self.log_scale = torch.tensor([[0.1,0],[0,0.1]],device=device_, dtype=torch.float)
        self.gaussian = multivariate_normal(self.loc.cpu().numpy(), self.log_scale.cpu().numpy())


    def sample(self, num_samples=1):
        return self.forward( num_samples)

    def forward(self, num_samples=1):
       
        weights = torch.softmax(self.weight_scores, 1)
        weights = weights.repeat(num_samples,1)

        rdn = torch.rand(num_samples, device=device_)
        lambert_mask = rdn < weights[...,-1]
        gauss_mask = ~lambert_mask

        num_gauss = torch.sum(gauss_mask)
        num_lambert = num_samples - num_gauss
        
        z_lambert, logp_lambert = self.lambertianLobe(num_lambert)
        z_guass = torch.tensor(np.random.multivariate_normal(self.loc.cpu().numpy(), self.log_scale.cpu().numpy(),size=num_gauss.item()),device = device_,dtype=torch.float)
        z = torch.zeros(num_samples, 2, device=device_, dtype=torch.float)
        z[gauss_mask] = z_guass
        z[lambert_mask] = z_lambert
        return z, None

    def log_prob(self, z):
        # Get weights
        weights = torch.softmax(self.weight_scores, 1)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",torch.sum(self.lambertianLobe.prob(z)==0))
        log_p = torch.log(torch.tensor(self.gaussian.pdf(z.cpu()))).cuda() + torch.log(weights[...,0])
        lambert_log_p = torch.log(self.lambertianLobe.prob(z)) + torch.log(weights[...,-1]) ####TODO?? check
        log_p = torch.cat((log_p.unsqueeze(1), lambert_log_p.unsqueeze(1)), dim=-1)
        log_p = torch.logsumexp(log_p, 1)
        return log_p



class Gaussian(nn.Module):
    def __init__(
        self, n_modes, dim, loc=None, scale=None, weights=None, trainable=True
    ):
        super().__init__()
       
        self.lambertianLobe = LambertianLobe()
        for param in self.lambertianLobe.parameters():
            param.requires_grad = False

        self.n_modes = n_modes
        self.dim = dim

        if weights is None:
            weights = np.ones(self.n_modes+1)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)
        print(weights)

        self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights)))

        self.loc = torch.zeros(2,device=device_, dtype=torch.float)
        self.log_scale = torch.tensor([[0.1,0],[0,0.1]],device=device_, dtype=torch.float)
        self.gaussian = multivariate_normal(self.loc.cpu().numpy(), self.log_scale.cpu().numpy())

    def sample(self, num_samples=1):
        return self.forward( num_samples)

    def forward(self, num_samples=1):
       
        z_guass = torch.tensor(np.random.multivariate_normal(self.loc.cpu().numpy(), self.log_scale.cpu().numpy(),size=num_samples.item()),device = device_,dtype=torch.float)
        return z_guass, None

    def log_prob(self, z):
        # Get weights
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",torch.sum(self.lambertianLobe.prob(z)==0))
        log_p = torch.log(torch.tensor(self.gaussian.pdf(z.cpu()))).cuda()
        return log_p


def draw_pdf(out_path, model, resolution=128):
    # woX = torch.linspace(-1.0,1.0, steps = resolution, device=device_)
    # woY = torch.linspace(-1.0,1.0, steps = resolution, device=device_)
    woX = torch.linspace(-5.0,5.0, steps = resolution, device=device_)
    woY = torch.linspace(-5.0,5.0, steps = resolution, device=device_)
    grid_z1, grid_z2 = torch.meshgrid(woX, woY)
    grid = torch.stack([grid_z1, grid_z2], dim = -1)
    light_dir = grid.reshape((-1, 2))
    logp = model.log_prob(light_dir)###TODO            
    pdfs = torch.exp(logp).cpu()

    A = 4/(resolution * resolution)
    pdfs *= 1/(A*torch.sum(pdfs))
    pdfs = pdfs.reshape(resolution, resolution)
    pdfs_3c = torch.stack([pdfs, pdfs, pdfs], dim=2)
    exr.write32(np.array(pdfs_3c, dtype=np.float32), out_path) 
    return pdfs
    
def draw_samples(out_path, model, resolution=128, num_samples=128*128*64*64): 
    with torch.no_grad():
        x, _ = model(num_samples)
    x = x.cpu().detach().numpy()
    # H, xedges, yedges = np.histogram2d(x=x[:,0], y=x[:,1], range=[[-1,1],[-1,1]], bins=(resolution,resolution),density=True)
    H, xedges, yedges = np.histogram2d(x=x[:,0], y=x[:,1], range=[[-5,5],[-5,5]], bins=(resolution,resolution),density=True)

    invalid_dirs = np.square(x[:,0]) + np.square(x[:,1]) > 1
    print("invalid percentage : ", (np.sum(invalid_dirs)/num_samples*100))
    A = 4/(resolution * resolution)
    # H *= 1.0/(A*num_samples)
    H_3c = np.stack([H, H, H], axis=2)
    exr.write32(np.array(H_3c, dtype=np.float32), out_path)
    return H


def plot_the_comparison():
    model = Gaussian(1,2,trainable=False).to(device=device_)
    draw_pdf("/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/renderer/output/debug_bias/test_pdf_withlambert.exr", model)
    draw_samples("/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/renderer/output/debug_bias/test_samples_withlambert.exr", model)


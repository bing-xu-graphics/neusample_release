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

ROOT_DIR = "./"
# from weighted_sum_dist import *
class BaseDistribution(nn.Module):
    """
    Base distribution of a flow-based model
    Parameters do not depend of target variable (as is the case for a VAE encoder)
    """

    def __init__(self):
        super().__init__()

    def forward(self, num_samples=1):
        """
        Samples from base distribution and calculates log probability
        :param num_samples: Number of samples to draw from the distriubtion
        :return: Samples drawn from the distribution, log probability
        """
        raise NotImplementedError

    def log_prob(self, z):
        """
        Calculate log probability of batch of samples
        :param z: Batch of random variables to determine log probability for
        :return: log probability for each batch element
        """
        raise NotImplementedError

M_PI = torch.pi 
PI_over_4 = torch.pi *4
PI_over_2 = torch.pi *2
INV_PI =    1/torch.pi

class LambertianLobe(nn.Module):
    def __init__(self):
        super().__init__()

    # def forward(self, num_samples=1):
    #     sample2 = torch.rand(num_samples, 2, device=device_, dtype=torch.double) * 2.0 - 1.0  #[-1,1]
    #     theta = sample2[0] * torch.pi
    #     phi = sample2[1] * torch.pi * 2.0 - torch.pi
    #     x = torch.sin(theta) * torch.cos(phi)
    #     y = torch.sin(theta) * torch.sin(phi)
    #     return torch.stack([x,y], axis=-1),torch.full((num_samples,), INV_PI).to(sample2.device)
   
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
        # cosTheta = torch.sqrt(torch.clamp(1 - torch.square(z[:,0]) - torch.square(z[:,1]), 0.0, 1.0))
        # pdf = INV_PI * cosTheta
        invalid_mask = torch.square(z[:,0]) + torch.square(z[:,1]) >=1
        # pdf = torch.full((z.shape[0],), 1/4).to(z.device) ### TODO debug
        pdf = torch.full((z.shape[0],), INV_PI).to(z.device) ### TODO debug

        if rm_invalid:
            pdf[invalid_mask] = 1e-5 ##TODO
        return pdf
    # def log_prob(self, z, rm_invalid = True):
    #     # cosTheta = torch.sqrt(torch.clamp(1 - torch.square(z[:,0]) - torch.square(z[:,1]), 0.0, 1.0))
    #     # pdf = INV_PI * cosTheta
    #     invalid_mask = torch.square(z[:,0]) + torch.square(z[:,1]) >1
    #     pdf = torch.full((z.shape[0],), 1/4).to(z.device) ### TODO debug
    #     if rm_invalid:
    #         pdf[invalid_mask] = 0.0 ##TODO
    #     return torch.log(pdf)

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

        if loc is None:
            loc = np.random.randn(self.n_modes, self.dim)
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((self.n_modes, self.dim))
        scale = np.array(scale)[None, ...]
        if weights is None:
            weights = np.ones(self.n_modes+1)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)
        print(weights)

        if trainable:
            self.loc = nn.Parameter(torch.tensor(1.0 * loc))
            self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 * scale)))
            self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights)))
        else:
            # self.register_buffer("loc", torch.tensor(1.0 * loc))
            # self.register_buffer("log_scale", torch.tensor(np.log(1.0 * scale)))
            self.register_buffer("loc", torch.tensor(0.0 * loc, dtype=torch.float))
            self.register_buffer("log_scale", torch.tensor(np.log(0.05 * scale),dtype=torch.float))
            self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights),dtype=torch.float))

    def sample(self, num_samples=1):
        return self.forward( num_samples)

    def forward(self, num_samples=1):
       
        weights = torch.softmax(self.weight_scores, 1)
        weights = weights.repeat(num_samples,1)
        loc = self.loc.repeat(num_samples,1,1) 
        log_scale = self.log_scale.repeat(num_samples,1,1)

        rdn = torch.rand(num_samples, device=device_)
        # print("hello there.....", weights)
        lambert_mask = rdn < weights[...,-1]
        gauss_mask = ~lambert_mask

        num_gauss = torch.sum(gauss_mask)
        num_lambert = num_samples - num_gauss
        print("++++++++++++++++++++++++++ num_gauss = ,", num_gauss)
        print("++++++++++++++++++++++++++ num_lambert = ,", num_lambert)
        print("~~~~~~~~~~~~~~~~log_scale.shape", log_scale.shape)
        eps_ = torch.randn(
            num_gauss, self.dim, dtype=loc.dtype, device=loc.device
        )
        z_lambert, logp_lambert = self.lambertianLobe(num_lambert)

        # print("eps_.shape", eps_.shape)
        scale_sample = torch.sum(torch.exp(log_scale[gauss_mask]), 1)
        loc_sample = torch.sum(loc[gauss_mask], 1)
        z_guass = eps_ * scale_sample + loc_sample
        z = torch.zeros(num_samples, 2, device=device_, dtype = self.loc.dtype)
        z[gauss_mask] = z_guass
        z[lambert_mask] = z_lambert

        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights[...,:-1]+1e-5) ###TODO check??? this weights
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )

        lambert_log_p = torch.log(self.lambertianLobe.prob(z)+1e-5) + torch.log(weights[...,-1]) ####TODO?? check
        log_p = torch.cat((log_p, lambert_log_p.unsqueeze(1)), dim=-1)
        log_p = torch.logsumexp(log_p, 1)

        return z, log_p

    def log_prob(self, z):
        # Get weights
        weights = torch.softmax(self.weight_scores, 1)
        loc = self.loc 
        log_scale = self.log_scale

        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(torch.clamp(weights[...,:-1],1e-5)) ###TODO check??? this weights
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )

        lambert_log_p = torch.log(torch.clamp(self.lambertianLobe.prob(z), 1e-5)) + torch.log(weights[...,-1]) ####TODO?? check
        log_p = torch.cat((log_p, lambert_log_p.unsqueeze(1)), dim=-1)
        log_p = torch.logsumexp(log_p, 1)
        return log_p

from torch.distributions.multivariate_normal import MultivariateNormal
# class GMMIso(nn.Module):
#     def __init__(
#         self, n_modes, dim, loc=None, scale=None, weights=None, trainable=True
#     ):
#         super().__init__()
#         self.lambertianLobe = LambertianLobe()
#         for param in self.lambertianLobe.parameters():
#             param.requires_grad = False
#         self.n_modes = n_modes
#         self.dim = dim
#         if loc is None:
#             loc = np.random.randn(self.n_modes, self.dim)
#         loc = np.array(loc)[None, ...]
#         if scale is None:
#             scale = np.ones((self.n_modes, self.dim))
#         scale = np.array(scale)[None, ...]
#         if weights is None:
#             weights = np.ones(self.n_modes+1)
#         weights = np.array(weights)[None, ...]
#         weights /= weights.sum(1)
#         print(weights)


#         self.register_buffer("loc", torch.tensor(0.0 * loc))
#         self.register_buffer("log_scale", torch.tensor(np.log(0.05 * scale)))
#         self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights)))

#         loc = torch.zeros(2,device=device_, dtype=torch.float)
#         scale = torch.eye(2,device=device_, dtype=torch.float)

#         # self.gaussian =MultivariateNormal(self.loc, scale_tril=torch.diag(self.log_scale))
#         self.gaussian =MultivariateNormal(loc, scale)



#     def sample(self, num_samples=1):
#         return self.forward( num_samples)

#     def forward(self, num_samples=1):
       
#         weights = torch.softmax(self.weight_scores, 1)
#         weights = weights.repeat(num_samples,1)
#         loc = self.loc.repeat(num_samples,1,1) 
#         log_scale = self.log_scale.repeat(num_samples,1,1)

#         rdn = torch.rand(num_samples, device=device_)
#         # print("hello there.....", weights)
#         lambert_mask = rdn < weights[...,-1]
#         gauss_mask = ~lambert_mask

#         num_gauss = torch.sum(gauss_mask)
#         num_lambert = num_samples - num_gauss
#         print("++++++++++++++++++++++++++ num_gauss = ,", num_gauss)
#         print("++++++++++++++++++++++++++ num_lambert = ,", num_lambert)
#         print("~~~~~~~~~~~~~~~~log_scale.shape", log_scale.shape)
        
#         z_lambert, logp_lambert = self.lambertianLobe(num_lambert)

#         # eps_ = torch.randn(
#         #     num_gauss, self.dim, dtype=loc.dtype, device=loc.device
#         # )
#         # scale_sample = torch.sum(torch.exp(log_scale[gauss_mask]), 1)
#         # loc_sample = torch.sum(loc[gauss_mask], 1)
#         z_guass = self.gaussian.sample(sample_shape=(num_gauss,))
#         print("++++++++++++++++++ z_gauss.shape", z_guass.shape)
#         z = torch.zeros(num_samples, 2, device=device_, dtype=torch.float)
#         z[gauss_mask] = z_guass
#         z[lambert_mask] = z_lambert

#         log_p = self.gaussian.log_prob(z)

#         lambert_log_p = torch.log(self.lambertianLobe.prob(z)+1e-5) + torch.log(weights[...,-1]) ####TODO?? check
#         log_p = torch.cat((log_p.unsqueeze(1), lambert_log_p.unsqueeze(1)), dim=-1)
#         log_p = torch.logsumexp(log_p, 1)

#         return z, log_p

#     def log_prob(self, z):
#         # Get weights
#         weights = torch.softmax(self.weight_scores, 1)
#         log_p = self.gaussian.log_prob(z)
#         lambert_log_p = torch.log(torch.clamp(self.lambertianLobe.prob(z), 1e-5)) + torch.log(weights[...,-1]) ####TODO?? check
#         print("~~~~~~~~~~~~` log_p.shape", log_p.shape, lambert_log_p.shape)
#         log_p = torch.cat((log_p.unsqueeze(1), lambert_log_p.unsqueeze(1)), dim=-1)
#         log_p = torch.logsumexp(log_p, 1)
#         return log_p



import scipy
from scipy.stats import multivariate_normal

class GMMComeon(nn.Module):
    """
    Mixture of Gaussians with diagonal covariance matrix, weighted sum with a lambertian lobe, 
    condition on (uv, wo)
    """
    def __init__(
        self, n_modes, dim, loc=None, scale=None, weights=None, trainable=True
    ):

        super().__init__()
       
        self.lambertianLobe = LambertianLobe()
        for param in self.lambertianLobe.parameters():
            param.requires_grad = False


        self.n_modes = n_modes
        self.dim = dim

        if loc is None:
            loc = np.random.randn(self.n_modes, self.dim)
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((self.n_modes, self.dim))
        scale = np.array(scale)[None, ...]
        if weights is None:
            weights = np.ones(self.n_modes+1)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)

        if trainable:
            self.loc = nn.Parameter(torch.tensor(1.0 * loc))
            self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 * scale)))
            self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights)))
        else:
            self.register_buffer("loc", torch.tensor(1.0 * loc))
            self.register_buffer("log_scale", torch.tensor(np.log(1.0 * scale)))
            self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights)))




    def sample(self, num_samples=1):
        return self.forward( num_samples)

    def forward(self, num_samples=1):
        # Get weights
        assert(self.n_modes == 2)
        weights = torch.softmax(self.weight_scores, 1)
        loc = self.loc 
        log_scale = self.log_scale
        weights_gauss = weights[...,:-1].clone()
        # weights_gauss /= torch.sum(weights_gauss, dim=1).unsqueeze(-1)
        
        rdn = torch.rand(num_samples).cuda()
       

        # lambert_mask = rdn < weights[...,-1]
        # gauss_mask = ~lambert_mask
        weights_cumsum =torch.cumsum(weights, dim=-1)
        gauss1_mask = rdn < weights_cumsum[...,0]
        gauss2_mask = (~gauss1_mask) & (rdn < weights_cumsum[...,1])
        gauss_mask = gauss1_mask | gauss2_mask
        lambert_mask = ~gauss_mask
        num_gauss = torch.sum(gauss_mask)
        num_gauss1 = torch.sum(gauss1_mask)
        num_gauss2 = num_gauss - num_gauss1
        num_lambert = num_samples - num_gauss

        mode = torch.zeros((num_samples,  1), dtype=torch.long).cuda()
        mode[gauss1_mask] = torch.full((num_gauss1,1), 0 ).cuda()
        mode[gauss2_mask] = torch.full((num_gauss2,1), 1).cuda()

        mode_1h = nn.functional.one_hot(mode[gauss_mask], self.n_modes).squeeze().unsqueeze(-1) #CHEKED CORRECTLY!
        # Get samples
        eps_ = torch.randn(
            num_gauss, self.dim, dtype=loc.dtype, device=loc.device
        )
        z_lambert, logp_lambert = self.lambertianLobe(num_lambert)
        

        scale_sample = torch.sum(torch.exp(log_scale[gauss_mask]) * mode_1h, 1)
        loc_sample = torch.sum(loc[gauss_mask] * mode_1h, 1)
        z_guass = eps_ * scale_sample + loc_sample
        
        z = torch.zeros(num_samples, 2, device=device_)
        z[lambert_mask] = z_lambert
        z[gauss_mask] = z_guass

        ## compute the pdfs
        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_gauss_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights[...,:-1]+1e-5)
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )
        lambert_log_p = torch.log(self.lambertianLobe.prob(z)+1e-5) + torch.log(weights[...,-1]) 
        log_p = torch.cat((log_gauss_p, lambert_log_p.unsqueeze(1)), dim=-1)
        log_p = torch.logsumexp(log_p, 1)

        return z, log_p


    def log_prob(self, z):
            # Get weights
            # Compute log probability
        weights = torch.softmax(self.weight_scores, 1)
        loc = self.loc 
        log_scale = self.log_scale
        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights[...,:-1]+1e-5) ###TODO check??? this weights
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )

        lambert_log_p = torch.log(self.lambertianLobe.prob(z)+1e-5) + torch.log(weights[...,-1]) ####TODO?? check
        log_p = torch.cat((log_p, lambert_log_p.unsqueeze(1)), dim=-1)
        log_p = torch.logsumexp(log_p, 1)
        return log_p

class GaussianMixture(BaseDistribution):
    """
    Mixture of Gaussians with diagonal covariance matrix
    """

    def __init__(
        self, n_modes, dim, loc=None, scale=None, weights=None, trainable=True
    ):

        super().__init__()

        self.n_modes = n_modes
        self.dim = dim

        if loc is None:
            loc = np.random.randn(self.n_modes, self.dim)
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((self.n_modes, self.dim))
        scale = np.array(scale)[None, ...]
        if weights is None:
            weights = np.ones(self.n_modes)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)

        if trainable:
            self.loc = nn.Parameter(torch.tensor(1.0 * loc))
            self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 * scale)))
            self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights)))
        else:
            self.register_buffer("loc", torch.tensor(1.0 * loc))
            self.register_buffer("log_scale", torch.tensor(np.log(1.0 * scale)))
            self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights)))

    def forward(self, num_samples=1):
        # Get weights
        weights = torch.softmax(self.weight_scores, 1)

        # Sample mode indices
        mode = torch.multinomial(weights[0, :], num_samples, replacement=True)
        mode_1h = nn.functional.one_hot(mode, self.n_modes)
        mode_1h = mode_1h[..., None]

        # Get samples
        eps_ = torch.randn(
            num_samples, self.dim, dtype=self.loc.dtype, device=self.loc.device
        )
        scale_sample = torch.sum(torch.exp(self.log_scale) * mode_1h, 1)
        loc_sample = torch.sum(self.loc * mode_1h, 1)
        z = eps_ * scale_sample + loc_sample

        # Compute log probability
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights)
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(self.log_scale, 2)
        )
        log_p = torch.logsumexp(log_p, 1)

        return z, log_p

    def log_prob(self, z):
        # Get weights
        weights = torch.softmax(self.weight_scores, 1)

        # Compute log probability
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights)
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(self.log_scale, 2)
        )
        log_p = torch.logsumexp(log_p, 1)

        return log_p

device_ = torch.device(0)
# def draw_pdf(out_path, model, resolution=128):
#     woX = torch.linspace(-1.0,1.0, steps = resolution, device=device_)
#     woY = torch.linspace(-1.0,1.0, steps = resolution, device=device_)
#     grid_z1, grid_z2 = torch.meshgrid(woX, woY)
#     grid = torch.stack([grid_z1, grid_z2], dim = -1)
#     light_dir = grid.reshape((-1, 2))
#     logp = model.log_prob(light_dir)###TODO            
#     pdfs = torch.exp(logp).cpu()
#     # pdfs = model.prob(light_dir).cpu()

#     A = 4/(resolution * resolution)
#     pdfs *= 1/(A*torch.sum(pdfs))
#     pdfs = pdfs.reshape(resolution, resolution)
#     pdfs_3c = torch.stack([pdfs, pdfs, pdfs], dim=2)
#     exr.write32(np.array(pdfs_3c, dtype=np.float32), out_path) 
#     # if(png):
#     #     cv2.imwrite(out_path+".png",self.tonemap(np.array(pdfs_3c, dtype=np.float32))*255.0)
#     return pdfs
    
def draw_samples(out_path, model, resolution=128, num_samples=128*128*64*64): 
    with torch.no_grad():
        x, _ = model(num_samples)
    x = x.cpu().detach().numpy()
    H, xedges, yedges = np.histogram2d(x=x[:,0], y=x[:,1], range=[[-1,1],[-1,1]], bins=(resolution,resolution),density=True)
    invalid_dirs = np.square(x[:,0]) + np.square(x[:,1]) > 1
    print("invalid percentage : ", (np.sum(invalid_dirs)/num_samples*100))
    A = 4/(resolution * resolution)
    # H *= 1.0/(A*num_samples)
    H_3c = np.stack([H, H, H], axis=2)
    exr.write32(np.array(H_3c, dtype=np.float32), out_path)
    # cv2.imwrite(out_path+".png", self.tonemap(np.array(H_3c, dtype=np.float32))*255.0)
    return H
def draw_pdf(out_path, model, resolution=128):
    #woX = torch.linspace(-1.0,1.0, steps = resolution, device=device_)
    #woY = torch.linspace(-1.0,1.0, steps = resolution, device=device_)
    #grid_z1, grid_z2 = torch.meshgrid(woX, woY)
    #grid = torch.stack([grid_z1, grid_z2], dim = -1)
    wi_base = torch.meshgrid(*[torch.linspace(0,resolution-1,resolution,device=device_)]*2)
    wi_base = torch.stack(wi_base,-1).reshape(resolution*resolution,1,2)

    S = 1000
    wi_base = (wi_base+torch.rand(resolution*resolution,S,2,device=device_)).div(resolution)*2-1
    light_dir = wi_base.reshape(-1,2)

    #light_dir = grid.reshape((-1, 2))
    logp = model.log_prob(light_dir)###TODO            
    pdfs = torch.exp(logp).cpu().reshape(resolution*resolution,S).mean(-1)

    A = 4/(resolution * resolution)
    pdfs *= 1/(A*torch.sum(pdfs))
    pdfs = pdfs.reshape(resolution, resolution)
    pdfs_3c = torch.stack([pdfs, pdfs, pdfs], dim=2)
    exr.write32(np.array(pdfs_3c, dtype=np.float32), out_path) 
    return pdfs

# def draw_samples(out_path, model, resolution=128, num_samples=128*128*64*64): 
#     with torch.no_grad():
#         x, _ = model(num_samples)
#     #x = x.cpu().detach().numpy()
#     #H, xedges, yedges = np.histogram2d(x=x[:,0], y=x[:,1], range=[[-1,1],[-1,1]], bins=(resolution,resolution),density=True)
#     # invalid_dirs = np.square(x[:,0]) + np.square(x[:,1]) > 1
#     x = x.detach().cpu()
#     x_ind = (x*0.5+0.5).mul(resolution).long().clamp(0,resolution-1)
#     x_ind = x_ind[...,0] + x_ind[...,1]*resolution
#     H = torch.zeros(resolution*resolution)
#     H.scatter_add_(0,x_ind,torch.ones_like(x_ind).float())
#     H = H/H.sum()
#     H = H.reshape(resolution,resolution)
    
#     # print("invalid percentage : ", (np.sum(invalid_dirs)/num_samples*100))
#     A = 4/(resolution * resolution)
#     # H *= 1.0/(A*num_samples)
#     H_3c = np.stack([H, H, H], axis=2)
#     exr.write32(np.array(H_3c, dtype=np.float32), out_path)
#     return H

def plot_the_comparison():
    # model = GaussianMixture(2,2,trainable=False).to(device=device_)
    model = GMMIso(1,2,trainable=False).to(device=device_)
    # model = LambertianLobe().to(device_)
    draw_pdf("/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/renderer/output/debug_bias/test_pdf_withlambert.exr", model)
    draw_samples("/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/renderer/output/debug_bias/test_samples_withlambert.exr", model)






### test gaussian lobes
if __name__ == "__main__":
    plot_the_comparison()    
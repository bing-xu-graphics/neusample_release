

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
import matplotlib.pyplot as plt
import argparse

from utils import exr, ops, la2
from nsf_light_bk import *
from xie_model import *
from neumat import interfaces
import normflows as nf
# import larsflow as lf
import cv2

ROOT_DIR = "./"
from weighted_sum_dist import *
import bnaf_cond

class neumip_inferencer:
    def __init__(self, ckpt_path="../tortoise_shell.ckpt", resolution=[512, 512], device="cuda:0", verbose=False):
        device = "cpu"
        if torch.cuda.is_available():
            device = f"cuda:0"
        else:
            raise ValueError("CPU not supported")

        ckpt = os.path.abspath(ckpt_path)
        if not os.path.exists(ckpt):
            raise ValueError("Checkpoint doesn't exist")

        model = interfaces.NeuMIPv1Module.load_from_checkpoint(ckpt, strict=False)
        model.model.initial_sigma = 1
        model.model.iterations_to_sigma_1 = 1
        if verbose:
            print("Neumip Model parameters:")
            for n, p in model.named_parameters():
                print(f"  {n}, {p.mean().item():.4f} {p.std().item():.4f}")

        model.to(device)
        model.eval()
        model.freeze()
        for param in model.parameters():
            param.requires_grad=False
        self.model = model

    
    def render_query(self, camera_dir, light_dir, uv):
        # with torch.no_grad():
        #     # clamping the vectors
            
        #     invalid_condition1 = torch.square(camera_dir[:,0]) + torch.square(camera_dir[:,1]) >1
        #     invalid_condition2 = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) >1

            
        #     # grazing_select_cam = torch.square(camera_dir[:,0]) + torch.square(camera_dir[:,1]) > 0.989074 # 3 degree
        #     # camera_dir[grazing_select_cam and camera_dir[:,0]<0, 0] = camera_dir[grazing_select_cam and camera_dir[:,0]<0, 0] + 0.0739127852
        #     # camera_dir[grazing_select_cam and camera_dir[:,0]>0, 0] = camera_dir[grazing_select_cam and camera_dir[:,0]>0, 0] - 0.0739127852
        #     # grazing_select_light = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) > 0.989074 # 3 degree
        #     # camera_dir[grazing_select_light and camera_dir[:,0]<0, 0] = camera_dir[grazing_select_light and camera_dir[:,0]<0, 0] + 0.0739127852
        #     # camera_dir[grazing_select_light and camera_dir[:,0]>0, 0] = camera_dir[grazing_select_light and camera_dir[:,0]>0, 0] - 0.0739127852
            
        #     rgb_prediction = self.model(camera_dir, light_dir, uv)
        #     rgb_prediction[invalid_condition1] = 0
        #     rgb_prediction[invalid_condition2] = 0
        #     rgb_prediction[rgb_prediction<0] = 0
            
        #     return rgb_prediction
        with torch.no_grad():
            rgb_prediction = self.model(camera_dir, light_dir, uv)
            invalid_condition1 = torch.square(camera_dir[:,0]) + torch.square(camera_dir[:,1]) >1
            invalid_condition2 = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) >1
            rgb_prediction[invalid_condition1] = 0
            rgb_prediction[invalid_condition2] = 0
            rgb_prediction[rgb_prediction<0] = 0
            return rgb_prediction


device_ = torch.device(0)

class vis_helper:
    def __init__(self, exp_name, ckpt_path) -> None:
        self.exp_name = exp_name
        self.neumip = neumip_inferencer(ckpt_path)

    def plot_results_group(self, model, condEncoder=None, root_dir = "./", base=None):
        # uv_input = torch.Tensor([0.3, 0.5])
        uv_test_list =  [torch.Tensor([0.3, 0.5]),torch.Tensor([-1.0,-0.3]), torch.Tensor([-0.7,-0.7]),torch.Tensor([-0.5,-0.3]), torch.Tensor([-0.3,-0.6]),
        torch.Tensor([0.0,0.3]),torch.Tensor([0.2,0.5]),torch.Tensor([0.5,0.5]),torch.Tensor([0.7,0.3]), torch.Tensor([0.3,0.5]), torch.Tensor([0.3,0.5]), torch.Tensor([0.7,0.7])] #,torch.Tensor([0.7,0.7])
        wo_test_list = [torch.Tensor([-0.8,-0.1]), torch.Tensor([-0.7,-0.7]),torch.Tensor([-0.5,-0.3]),torch.Tensor([-0.3,-0.6]), torch.Tensor([0.0,0.3]),
        torch.Tensor([0.2,0.5]),torch.Tensor([0.5,0.5]),torch.Tensor([0.7,0.3]),torch.Tensor([0.7,0.7]), torch.Tensor([-0.7,-0.7]), torch.Tensor([0.5,0.5]), torch.Tensor([-0.5,-0.3])]
        model.eval()
        output_dir = os.path.join(root_dir, "output/train_test/%s"%(self.exp_name))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i in range(len(wo_test_list)):
            print("inference %d........................................"%(i))
            uv_input = uv_test_list[i]
            wo_input = wo_test_list[i]
            with torch.no_grad():
                self.render_2d_func_pdf_nt(os.path.join(output_dir, "test_%d_2d_fitted_pdf_uv_wo_%f_%f_%f_%f.exr"%(i,uv_input[0],uv_input[1], wo_input[0],wo_input[1])), model, wo_input, uv_input, base, 128,None, plot_z=False, png=True, encoder=condEncoder)
                self.render_2d_func_sample_nt(os.path.join(output_dir, "test_%d_2d_fitted_sample_uv_wo_%f_%f_%f_%f.exr"%(i,uv_input[0],uv_input[1], wo_input[0],wo_input[1])), model, wo_input, uv_input, base, 128, 128*128*64, encoder=condEncoder)
                # self.render_2d_func_sample_nt(os.path.join(output_dir, "test_%d_2d_fitted_sample_uv_wo_%f_%f_%f_%f.exr"%(i,uv_input[0],uv_input[1], wo_input[0],wo_input[1])), model, wo_input, uv_input, base, 128, 64*64, encoder=condEncoder)
                self.render_2d_slice_gt(os.path.join(output_dir, "test_%d_2d_gt_pdf_uv_wo_%f_%f_%f_%f.exr"%(i,uv_input[0],uv_input[1], wo_input[0],wo_input[1])), wo_input, uv_input, 128, False, False,png=True, encoder=condEncoder)
                # self.render_2d_f_over_p(os.path.join(output_dir, "test_%d_2d_f_over_p_uv_wo_%f_%f_%f_%f"%(i,uv_input[0],uv_input[1], wo_input[0],wo_input[1])), model, wo_input, uv_input,512, False, False,png=True, encoder=condEncoder)
                # for spp in [4, 16, 4*25, 4*36, 4* 49, 4* 64, 4* 81, 4* 100]:
                #     self.render_2d_func_sample_nt(os.path.join(output_dir, "test_%d_2d_fitted_sample_uv_wo_%f_%f_%f_%f_%d_stratify.exr"%(i,uv_input[0],uv_input[1], wo_input[0],wo_input[1], spp)), model, wo_input, uv_input, base, 128, spp*32*32, encoder=condEncoder)

            # if i>2:break
        model.train()


    def plot_results(self, model, iter, condEncoder=None, root_dir = ROOT_DIR,base=None):
        uv_input = torch.Tensor([0.3, 0.5])
        wo_test_list = [torch.Tensor([-0.7,-0.7]), torch.Tensor([0.5,0.5])]
        # wo_test_list = [torch.Tensor([0.5,0.5])] #for 2d
        output_dir = os.path.join(root_dir, "output/train_test/%s"%(self.exp_name))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if(isinstance(model, nn.Module)):
            model.eval()
        else:
            model.flow.eval()
        for i in range(len(wo_test_list)):
            wo_input = wo_test_list[i]
            with torch.no_grad():
                resampled_base_path = os.path.join(output_dir, "test_%d_2d_resampled_z_iter_%d.exr"%(i, iter))
                fitted_pdf_path = os.path.join(output_dir, "test_%d_2d_fitted_pdf_iter_%d.exr"%(i, iter))
                self.render_2d_func_pdf_nt(fitted_pdf_path, model, wo_input, uv_input, base, 128, resampled_base_path, plot_z=False, png=False, encoder=condEncoder)
                # self.render_2d_func_sample_nt(os.path.join(output_dir, "test_%d_2d_fitted_sample_iter_%d.exr"%(i, iter)), model, wo_input, uv_input, base, 128, 128*128*64, condEncoder)
                if iter<2:
                    self.render_2d_slice_gt(os.path.join(output_dir, "test_%d_2d_gt_pdf_iter_%d.exr"%(i, iter)), wo_input, uv_input, 128, False,False,png=False, encoder=condEncoder)
        if(isinstance(model, nn.Module)):
            model.train()
        else:
            model.flow.train()

    def rgb2lum(self,rgb):
        # lum = 0.299*rgb[:,0] + 0.587*rgb[:,1] + 0.114*rgb[:,2]
        lum = 0.2126*rgb[:,0] + 0.7152*rgb[:,1] + 0.0722*rgb[:,2]
        return torch.stack([lum, lum, lum], dim=1)

    def rgb2lum_batch(self,rgb):
        # lum = 0.299*rgb[:,0] + 0.587*rgb[:,1] + 0.114*rgb[:,2]
        lum = 0.2126*rgb[:,:,0] + 0.7152*rgb[:,:,1] + 0.0722*rgb[:,:,2]
        return lum
    def rgb2lum_batch_(self,rgb):
        # lum = 0.299*rgb[:,0] + 0.587*rgb[:,1] + 0.114*rgb[:,2]
        lum = 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]
        return lum
    # def tonemap(self, im):
    #     tonemapper = cv2.createTonemapReinhard(1.5, 0,0,0)
    #     ldr = tonemapper.process(im)
    #     ldr = np.clip(ldr, 0.0,1.0)
    #     return ldr
    def tonemap(self, img, gamma=2.2):
        if type(img) == torch.Tensor:
            img = torch.clamp(img, min=0.0, max=None)
            img = (img / (1 + img)) ** (1.0 / gamma)
        elif type(img) == np.ndarray:
            img = np.clip(img, 0.0, None)
            img = (img / (1 + img)) ** (1.0 / gamma)
        else:
            assert(False)

        ## TO check
        # print(img.shape)
        # ret = np.zeros(img.shape)
        # ret[...,0] = img[...,2]
        # ret[...,1] = img[...,1]
        # ret[...,2] = img[...,0]

        return img

    def eval_neumip_unnomailzed_bsdf(self, camera_dir, uv, light_dir):
        with torch.no_grad():
            rgb_pred = self.neumip.model(camera_dir, light_dir, uv)
            rgb_pred[rgb_pred<0] = 0
        invalid_dirs = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) > 1
        rgb_pred[invalid_dirs] = 0.0
        lumi_pred = self.rgb2lum_batch_(rgb_pred)
        return lumi_pred

    def render_2d_slice_gt(self, out_path, camera_dir, uv, resolution, h5file = False, WithCosine=False,png=False, encoder=None):#changing one direction
        woX = torch.linspace(-1.0,1.0, steps = resolution)
        woY = torch.linspace(-1.0,1.0, steps = resolution)
        grid_z1, grid_z2 = torch.meshgrid(woX, woY)
        grid = torch.stack([grid_z1, grid_z2], dim = -1)
        light_dir = grid.reshape((-1, 2)).cuda()
        camera_dir_tensor = torch.tile(camera_dir, (light_dir.shape[0],)).reshape(-1,2).cuda()
        uv_tensor = torch.tile(uv, (light_dir.shape[0],)).reshape(-1,2).cuda()
        with torch.no_grad():
            rgb_pred = self.neumip.model(camera_dir_tensor, light_dir, uv_tensor)
            rgb_pred[rgb_pred<0] = 0
        invalid_dirs = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) > 1
        # valid_dirs = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) <= 1.0
        rgb_pred[invalid_dirs] = 0.0
        lumi_pred = self.rgb2lum(rgb_pred)
        if WithCosine:
            print(light_dir.shape)
            cos_theta = torch.sqrt(1.0- torch.square(light_dir[:,0]) - torch.square(light_dir[:,1])).unsqueeze(1)
            cos_theta[invalid_dirs] = 0.0
            lumi_pred *= cos_theta ### added cos_theta part onto the bsdf
        A = 4/(resolution * resolution)
        lumi_pred *= 1/(A * torch.sum(lumi_pred[:,0]))
        lumi_pred = lumi_pred.reshape(resolution, resolution, -1).cpu()
        if(not h5file):
            exr.write32(np.array(lumi_pred, dtype=np.float32), out_path)
        if(png):
            cv2.imwrite(out_path+".png", self.tonemap(np.array(lumi_pred, dtype=np.float32))*255.0)
        return lumi_pred
    
    def render_2d_f_over_p(self, out_path, model, camera_dir, uv, resolution, h5file = False, WithCosine=False,png=False, encoder=None):#changing one direction
        woX = torch.linspace(-1.0,1.0, steps = resolution)
        woY = torch.linspace(-1.0,1.0, steps = resolution)
        grid_z1, grid_z2 = torch.meshgrid(woX, woY)
        grid = torch.stack([grid_z1, grid_z2], dim = -1)
        light_dir = grid.reshape((-1, 2)).cuda()
        camera_dir_tensor = torch.tile(camera_dir, (light_dir.shape[0],)).reshape(-1,2).cuda()
        uv_tensor = torch.tile(uv, (light_dir.shape[0],)).reshape(-1,2).cuda()
        with torch.no_grad():
            rgb_pred = self.neumip.model(camera_dir_tensor, light_dir, uv_tensor)
            rgb_pred[rgb_pred<0] = 0
        invalid_dirs = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) > 1
        valid_mask = ~invalid_dirs
        rgb_pred[invalid_dirs] = 0.0
        # lumi_pred = self.rgb2lum(rgb_pred)
        
        with torch.no_grad():
            
            if (isinstance(model, GMMWeightedCond) or isinstance(model, GMMWeightedCondLarge) ):
                cond_vec = encoder.compute_cond(camera_dir_tensor, uv_tensor)
                logp = model.log_prob(light_dir, cond_vec)
                # logp = model.log_prob_topweight_gauss(light_dir, cond_vec)
            elif (isinstance(model, GMMWeightedCondIsoLarge)):
                cond_vec = encoder.compute_cond(camera_dir_tensor, uv_tensor)
                logp = model.log_prob(light_dir, cond_vec, camera_dir_tensor)

            else:
                raise NotImplementedError("Not supported model type")

        pdfs = torch.exp(logp).cpu()
        # pdfs[invalid_dirs] = 0.0
        A = 4/(resolution * resolution)
        pdfs *= 1/(A*torch.sum(pdfs))
        rgb_pred = rgb_pred.reshape(resolution, resolution, -1).cpu()
        pdfs = pdfs.reshape(resolution, resolution)
        pdfs_3c = torch.stack([pdfs, pdfs, pdfs], dim=-1)
        f_over_p = rgb_pred / pdfs_3c
        triple = torch.cat((rgb_pred, pdfs_3c, f_over_p), dim=1)

        exr.write32(np.array(triple, dtype=np.float32), out_path+".exr")
        if(png):
            cv2.imwrite(out_path+".png", self.tonemap(np.array(triple, dtype=np.float32))*255.0)
        return f_over_p
        

    def render_2d_func_pdf_nt(self, out_path, model, wi, loc, prior=None, resolution=128, plot_z_outpath=None, plot_z=False, png=False, encoder=None):
        # woX = torch.linspace(-1.0,1.0, steps = resolution) + 
        # woY = torch.linspace(-1.0,1.0, steps = resolution)
        # grid_z1, grid_z2 = torch.meshgrid(woX, woY)
        # grid = torch.stack([grid_z1, grid_z2], dim = -1)
        # light_dir = grid.reshape((-1, 2)).cuda()
        
        wi_base = torch.meshgrid(*[torch.linspace(0,resolution-1,resolution,device=device_)]*2)
        wi_base = torch.stack(wi_base,-1).reshape(resolution*resolution,1,2)

        S = 121
        wi_base = (wi_base+torch.rand(resolution*resolution,S,2,device=device_)).div(resolution)*2-1
        light_dir = wi_base.reshape(-1,2)

        uv_input = torch.tile(loc, (light_dir.shape[0],)).reshape(-1,2).cuda()
        wi_input = torch.tile(wi, (light_dir.shape[0],)).reshape(-1,2).cuda()
        invalid_dirs = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) > 1
        valid_mask = ~invalid_dirs
        pdfs = torch.zeros(uv_input.shape[0])
        with torch.no_grad():
            if(isinstance(model, nf.NTCondNormalizingFlow)):
                logp = model.log_prob(light_dir, wi_input, uv_input)
            
            elif (isinstance(model, WeightedSumDistribution) or isinstance(model, GMMCond)):
                cond_vec = encoder.compute_cond(wi_input, uv_input)
                # logp = model.log_prob(light_dir, cond_vec)
                logp = model.log_prob(light_dir, cond_vec)
            elif (isinstance(model, WeightedSumDistributionNoCond)):
                logp = model.log_prob(light_dir)
            elif (isinstance(model, GMMWeightedCond) or isinstance(model, GMMWeightedCondLarge)):
                cond_vec = encoder.compute_cond(wi_input, uv_input)
                logp = model.log_prob(light_dir, cond_vec)
                # logp = model.log_prob_topweight_gauss(light_dir, cond_vec)
            elif (isinstance(model, GMMWeightedCondIsoLarge)):
                cond_vec = encoder.compute_cond(wi_input, uv_input)
                logp = model.log_prob(light_dir, cond_vec, wi_input)

            elif (isinstance(model, bnaf_cond.Sequential)):
                cond_vec = encoder.compute_cond(wi_input, uv_input)
                inputs = torch.cat((cond_vec, light_dir), dim=1)
                logp = bnaf_cond.compute_log_p_x(model, inputs)
            else:
                assert(False)
                y_mb, log_diag_j_mb = model(light_dir)
                log_p_y_mb = (
                    torch.distributions.Normal(torch.zeros_like(y_mb), torch.ones_like(y_mb))
                    .log_prob(y_mb)
                    .sum(-1)
                    )
                logp = log_p_y_mb + log_diag_j_mb

        pdfs = torch.exp(logp).cpu()
        A = 4/(resolution * resolution)
        pdfs[invalid_dirs] = 0.0
        pdfs = pdfs.reshape(resolution, resolution, S).mean(-1)
        # pdfs = pdfs.reshape(resolution, resolution)
        pdfs *= 1/(A*torch.sum(pdfs))
        # pdfs[invalid_dirs] = 0.0
        
        
        pdfs_3c = torch.stack([pdfs, pdfs, pdfs], dim=2)
        exr.write32(np.array(pdfs_3c, dtype=np.float32), out_path) 
        if(png):
            cv2.imwrite(out_path+".png",self.tonemap(np.array(pdfs_3c, dtype=np.float32))*255.0)
        if(plot_z):
            assert(cond_vec!=None)
            # cond_vec = model.compute_cond(wi_input, uv_input)
            if model.base == "condLars":
                log_prob = model.q0.log_prob(light_dir, cond_vec).to('cpu').view(*pdfs.shape)
            elif model.base == "condLars_uni":
                x = (light_dir + 1.0 )*0.5 ##TODO remove
                log_prob = model.q0.log_prob(x, cond_vec).to('cpu').view(*pdfs.shape)
            elif model.base == "condGauss":
                _,_,log_prob= model.q0.forward(light_dir, cond_vec,eps = 0, reverse = False)
                log_prob = log_prob.to('cpu').view(*pdfs.shape)
            else:
                x = (light_dir + 1.0 )*0.5 ##TODO remove
                log_prob = model.q0.log_prob(x).to('cpu').view(*pdfs.shape)
                
            prob = torch.exp(log_prob)
            prob[torch.isnan(prob)] = 0
            prob_base = torch.stack([prob, prob, prob], dim=2).numpy()
            exr.write32(np.array(prob_base, dtype=np.float32), plot_z_outpath) 
        return pdfs
        
    def render_2d_func_sample_nt(self, out_path, model, wi, loc, prior=None, resolution=128, num_samples=128*128*64, encoder=None): 
        uv_input = torch.tile(loc, (num_samples,)).reshape(-1,2).cuda()
        wi_input = torch.tile(wi, (num_samples,)).reshape(-1,2).cuda()
        with torch.no_grad():
            if(isinstance(model, nf.NTCondNormalizingFlow)):
                x, log_q = model.sample( wi_input, uv_input,num_samples)
            elif (isinstance(model, GMMWeightedCond) or isinstance(model, GMMWeightedCondLarge)):
                cond_vec = encoder.compute_cond(wi_input, uv_input)
                x, _ = model(cond_vec, num_samples)
            elif (isinstance(model, GMMWeightedCondIsoLarge)):
                cond_vec = encoder.compute_cond(wi_input, uv_input)
                x, _ = model(cond_vec, num_samples, wi_input)
            elif (isinstance(model, WeightedSumDistribution)or isinstance(model, GMMCond)):
                cond_vec = encoder.compute_cond(wi_input, uv_input)
                x,_ = model(cond_vec, num_samples)
            elif (isinstance(model, WeightedSumDistributionNoCond) ):
                x,_ = model(num_samples)
            else:
                assert(False)
        x = x.cpu().detach().numpy()
        H, xedges, yedges = np.histogram2d(x=x[:,0], y=x[:,1], range=[[-1,1],[-1,1]], bins=(resolution,resolution),density=False)
        invalid_dirs = np.square(x[:,0]) + np.square(x[:,1]) > 1
        print("invalid percentage : ", (np.sum(invalid_dirs)/num_samples*100))
        A = 4/(resolution * resolution)
        H *= 1.0/(A*num_samples)
        H_3c = np.stack([H, H, H], axis=2)
        exr.write32(np.array(H_3c, dtype=np.float32), out_path)
        # cv2.imwrite(out_path+".png", self.tonemap(np.array(H_3c, dtype=np.float32))*255.0)
        return H
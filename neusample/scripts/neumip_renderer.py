
import argparse, os, sys

from xml.etree.ElementTree import VERSION
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
from sys import exit
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
import os
# Insert root dir in path
p = os.path.abspath('.')
sys.path.insert(1, p)
from torch.utils.data import Dataset, DataLoader
# from dataset_samplep import BSDFSamplingPDataset
from torch import distributions
# from ipypb import ipb
from tqdm import tqdm
import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader
from neumat import interfaces, lights, datasets
from utils import exr, ops, la2
from normflows import utils
from normflows import distributions

# ROOT_DIR = "/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy"
ROOT_DIR = "./"


class NeuMipSampler(distributions.BaseDistribution):
    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer

    def forward(self, num_samples=1):
        """
        Samples from base distribution and calculates log probability
        :param num_samples: Number of samples to draw from the distriubtion
        :return: Samples drawn from the distribution, log probability
        """
        
        raise NotImplementedError

    def log_prob(self, z, wo, uv):
        with torch.no_grad():
            rgb_pred = self.renderer.model(wo, z, uv) #uv_tensor
            rgb_pred[rgb_pred < 0] = 1e-5
            
            # rgb_pred[:, self.invalid_dirs] = 0.0
            
            
            rgb_pred = rgb_pred.reshape(wo.shape[0], -1, 3)
            lumi_pred = self.renderer.rgb2lum_batch(rgb_pred)
            return torch.log(lumi_pred)

class NeuMipRenderer:
    def __init__(self, resolution, neumat, dataset=None, device="cuda:0"):
        self.width = resolution[0]
        self.height = resolution[1]
        self.device = device

        self.model = neumat
        
        self.dataset = dataset

    def render_query(self, camera_dir, light_dir, uv):
        with torch.no_grad():
            rgb_prediction = self.model(camera_dir, light_dir, uv)
            invalid_condition1 = torch.square(camera_dir[:,0]) + torch.square(camera_dir[:,1]) >1
            invalid_condition2 = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) >1
            rgb_prediction[invalid_condition1 ] = 0.0
            rgb_prediction[invalid_condition2 ] = 0.0
            return self.rgb2lum_1c(rgb_prediction)


    # def render_btf_sampling(self, frames, outdir, realnvp):
    #     if self.dataset is None or not os.path.exists(self.dataset):
    #         raise ValueError("Valid input dataset required to visualize btf queries")

    #     for frame in range(10):
    #         #generate random camera_dir and uv 
    #         fig, ax = plt.subplots(1,2)
    #         fig.set_size_inches(10,5)
    #         THETA_BIN = 50
    #         PHI_BIN = 100

    #         camera_dir = torch.rand()*2.0 - 1.0
                
    #         uv = torch.rand()*2.0 - 1.0
                
    #         #using realnvp to sample
    #         z = torch.rand(VISUALIZE_NUM_SAMPLES, 2).cuda()
    #         camera_dir = (torch.rand(1,2)*2.0 - 1.0).repeat(VISUALIZE_NUM_SAMPLES).cuda()
    #         uv = (torch.rand(1,2)*2.0 - 1.0).repeat(VISUALIZE_NUM_SAMPLES).cuda()
    #         x, _ = realNVP(z, camera_dir, uv)
    #         x = (x+1.0)/2.0

    #         H, xedges, yedges = np.histogram2d(x=x[:,0], y=x[:,1], range=[[0,1],[0,1]], bins=(THETA_BIN,PHI_BIN),density=True)
    #         H = H/np.sum(H)
    #         ax[0].imshow(H)

    #         theta = torch.linspace(-1, 1, steps=THETA_BIN)
    #         phi = torch.linspace(-1,1, steps=PHI_BIN)
    #         grid_z1, grid_z2 = torch.meshgrid(theta, phi)
    #         grid = torch.stack([grid_z1, grid_z2], dim = -1)
    #         light_dir = grid.reshape((-1, 2)).cuda()

    #         # camera_dir_tile = torch.tile(camera_dir, (light_dir.shape[0],)).reshape((-1,2)).cuda() 
    #         # uv_tile = torch.tile(uv, (light_dir.shape[0],)).reshape((-1,2)).cuda()   
            
    #         rgb_prediction = self.model(camera_dir, light_dir, uv)
    #         neumip_prediction = rgb2gray(rgb_prediction)
    #         neumip_prediction = neumip_prediction.reshape(THETA_BIN, PHI_BIN)
    #         neumip_prediction = neumip_prediction.cpu().numpy()
    #         neumip_prediction = neumip_prediction/np.sum(neumip_prediction)
    #         ax[1].imshow(neumip_prediction)
            
    #         # visualize_bsdf_side_by_side(H, bsdf, os.path.join(ROOT_DIR, "/output/neumip_%s_polar_frame_%d.png"%(exp_name,frame)))
            

    def render_btf_queries(self, frames, outdir):
        if self.dataset is None or not os.path.exists(self.dataset):
            raise ValueError("Valid input dataset required to visualize btf queries")

        neumipdata = datasets.NeuMIPv1Dataset(self.dataset)

        # Visualize inputs
        neumipdata.dump_h5(outdir, frames)

        dataloader = DataLoader(neumipdata, batch_size=1, num_workers=0, shuffle=False)

        curr_frame = 0
        for batch_idx, batch in enumerate(dataloader):
            camera_dir, light_dir, color, uv, cam_qr = batch

            camera_dir = camera_dir.reshape(-1, camera_dir.shape[-1])
            light_dir = light_dir.reshape(-1, light_dir.shape[-1])
            uv = uv.reshape(-1, uv.shape[-1])
            color = color.reshape(-1, color.shape[-1])
            with torch.no_grad():
                rgb_prediction = self.model(camera_dir, light_dir, uv)

            rgb_prediction = rgb_prediction.reshape(self.width, self.height, -1).cpu()
            outfile = os.path.join(outdir, "pred_" + str(curr_frame) + ".exr")
            exr.write16(np.array(rgb_prediction, dtype=np.float32), outfile)

            curr_frame += 1

            if (curr_frame == frames):
                break
    def rgb2lum(self,rgb):
        # lum = 0.299*rgb[:,0] + 0.587*rgb[:,1] + 0.114*rgb[:,2]
        lum = 0.2126*rgb[:,0] + 0.7152*rgb[:,1] + 0.0722*rgb[:,2]
        return torch.stack([lum, lum, lum], dim=1)

    def rgb2lum_batch(self,rgb):
        # lum = 0.299*rgb[:,0] + 0.587*rgb[:,1] + 0.114*rgb[:,2]
        lum = 0.2126*rgb[:,:,0] + 0.7152*rgb[:,:,1] + 0.0722*rgb[:,:,2]
        return lum

    def rgb2lum_1c(self,rgb):
        # lum = torch.zeros(rgb.shape[0], 1)
        lum = 0.2126*rgb[:,0] + 0.7152*rgb[:,1] + 0.0722*rgb[:,2]
        return lum

    def batch_unhalf(self, wo, wh):
        wo_world = torch.stack((wo[:,0], wo[:,1], torch.sqrt(1.0 - torch.square(wo[:,0]) - torch.square(wo[:,1]))), dim=1)
        wh_world = torch.stack((wh[:,0], wh[:,1], torch.sqrt(1.0 - torch.square(wh[:,0]) - torch.square(wh[:,1]))), dim=1)
        wi_world = 2*wh_world - wo_world
        wi_world = torch.nn.functional.normalize(wi_world,dim=1)
        return wi_world

    def batch_relection(self, wo, wh):
        wo_world = torch.stack((wo[:,0], wo[:,1], torch.sqrt(1.0 - torch.square(wo[:,0]) - torch.square(wo[:,1]))), dim=1)
        wh_world = torch.stack((wh[:,0], wh[:,1], torch.sqrt(1.0 - torch.square(wh[:,0]) - torch.square(wh[:,1]))), dim=1)
        wi_world = wo_world - 2.0*torch.sum(wo_world*wh_world, dim=-1).unsqueeze(1).expand(-1,3) * wh_world 
        # wi_world = torch.nn.functional.normalize(wi_world,dim=1)
        return wi_world

    def dot_product_in_3d(self, wi, wh):
        wi_world = torch.stack((wi[:,0], wi[:,1], torch.sqrt(1.0 - torch.square(wi[:,0]) - torch.square(wi[:,1]))), dim=1)
        wh_world = torch.stack((wh[:,0], wh[:,1], torch.sqrt(1.0 - torch.square(wh[:,0]) - torch.square(wh[:,1]))), dim=1)
        return torch.sum(wi_world*wh_world, dim=-1)

    def render_2d_slice_gt_halfvector(self, out_path, camera_dir, uv, resolution, png = False):
        whX = torch.linspace(-1.0,1.0, steps = resolution)
        whY = torch.linspace(-1.0,1.0, steps = resolution)
        grid_z1, grid_z2 = torch.meshgrid(whX, whY)
        grid = torch.stack([grid_z1, grid_z2], dim = -1)
        half_dir = grid.reshape((-1, 2)).cuda() 
        invalid_dirs_half = torch.square(half_dir[:,0]) + torch.square(half_dir[:,1]) > 1
        half_dir[invalid_dirs_half] = 0.0
        camera_dir_tensor = torch.tile(camera_dir, (half_dir.shape[0],)).reshape(-1,2).cuda()
        light_dir = self.batch_unhalf(camera_dir_tensor, half_dir)
        below_the_plane = light_dir[:,2]<0
        light_dir = light_dir[:,:2]
    
        assert(torch.all((torch.square(light_dir[:,0]) + torch.square(light_dir[:,1])) <=1))
        uv_tensor = torch.tile(uv, (light_dir.shape[0],)).reshape(-1,2).cuda()
        with torch.no_grad():
            rgb_pred = self.model(camera_dir_tensor, light_dir, uv_tensor)
            rgb_pred[rgb_pred<0] = 0
        invalid_dirs = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) > 1
        rgb_pred[invalid_dirs] = 0.0
        rgb_pred[invalid_dirs_half] = 0.0
        rgb_pred[below_the_plane] = 0.0
        
        lumi_pred = self.rgb2lum(rgb_pred)

        #times cosine
        # cos_theta = torch.sqrt(1.0- torch.square(light_dir[:,0]) - torch.square(light_dir[:,1])).unsqueeze(1)
        # cos_theta[invalid_dirs] = 0.0
        # lumi_pred *= cos_theta ### added cos_theta part onto the bsdf

        #account for change of variable
        widotwh = self.dot_product_in_3d(light_dir,half_dir).unsqueeze(1)
        lumi_pred *= 4*widotwh

        A = 4/(resolution * resolution)
        lumi_pred *= 1/(A * torch.sum(lumi_pred[:,0]))
        lumi_pred = lumi_pred.reshape(resolution, resolution, -1).cpu()
        # out_img = np.swapaxes(np.array(lumi_pred, dtype=np.float32),0, 1)
        out_img = np.array(lumi_pred, dtype=np.float32)
        # exr.write32(np.array(lumi_pred, dtype=np.float32), out_path)
        exr.write32(out_img, out_path)
        if(png):
            cv2.imwrite(out_path+".png", self.tonemap(out_img)*255.0)
        return lumi_pred

        
    def render_2d_slice_gt(self, out_path, camera_dir, uv, resolution, h5file = False, WithCosine=False,png=False):#changing one direction
        woX = torch.linspace(-1.0,1.0, steps = resolution)
        woY = torch.linspace(-1.0,1.0, steps = resolution)
        grid_z1, grid_z2 = torch.meshgrid(woX, woY)
        grid = torch.stack([grid_z1, grid_z2], dim = -1)
        light_dir = grid.reshape((-1, 2)).cuda()
        camera_dir_tensor = torch.tile(camera_dir, (light_dir.shape[0],)).reshape(-1,2).cuda()
        uv_tensor = torch.tile(uv, (light_dir.shape[0],)).reshape(-1,2).cuda()
        with torch.no_grad():
            rgb_pred = self.model(camera_dir_tensor, light_dir, uv_tensor)
            rgb_pred[rgb_pred<0] = 0
        invalid_dirs = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) > 1
        valid_dirs = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) <= 1.0
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
            # exr.write32(np.swapaxes(np.array(lumi_pred, dtype=np.float32),0, 1), out_path)
        if(png):
            # cv2.imwrite(out_path+".png", self.tonemap(np.swapaxes(np.array(lumi_pred, dtype=np.float32),0, 1))*255.0)
            cv2.imwrite(out_path+".png", self.tonemap(np.array(lumi_pred, dtype=np.float32))*255.0)
        return lumi_pred

    def render_2d_func_pdf_autoregressive(self, out_path, realNVP, prior, resolution, plot_z_outpath, plot_z):
        realNVP.eval()
        woX = torch.linspace(-1.0,1.0, steps = resolution)
        woY = torch.linspace(-1.0,1.0, steps = resolution)
        grid_z1, grid_z2 = torch.meshgrid(woX, woY)
        grid = torch.stack([grid_z1, grid_z2], dim = -1)
        light_dir = grid.reshape((-1, 2)).cuda()
        with torch.no_grad():
            z, logdet = realNVP.inverse(light_dir)
            logp = prior.log_prob(z) + logdet 
        
        pdfs = torch.exp(logp)
        A = 4/(resolution * resolution)
        pdfs *= 1/(A*torch.sum(pdfs))
        pdfs = pdfs.reshape(resolution, resolution)
        pdfs_3c = torch.stack([pdfs, pdfs, pdfs], dim=2).cpu()
        exr.write32(np.array(pdfs_3c, dtype=np.float32), out_path)
        return pdfs


    def tonemap(self, im):
        tonemapper = cv2.createTonemapReinhard(1.5, 0,0,0)
        ldr = tonemapper.process(im)
        ldr = np.clip(ldr, 0.0,1.0)
        return ldr
        
    def render_2d_func_sample_autoregressive(self, out_path, realNVP, prior, resolution, num_samples): #by histogram
            realNVP.eval()
            z = torch.normal(0, 1, size = (num_samples, 2)).cuda()
            with torch.no_grad():
                x, _ = realNVP(z)
            x = x.cpu().detach().numpy()
            H, xedges, yedges = np.histogram2d(x=x[:,0], y=x[:,1], range=[[-1,1],[-1,1]], bins=(resolution,resolution),density=False)
            A = 4/(resolution * resolution)
            H *= 1.0/(A*num_samples)
            H_3c = np.stack([H, H, H], axis=2)
            exr.write32(np.array(H_3c, dtype=np.float32), out_path)
            return H

    def render_2d_func_pdf(self, out_path, realNVP, prior, resolution, plot_z_outpath, plot_z):
        realNVP.eval()
        woX = torch.linspace(-1.0,1.0, steps = resolution)
        woY = torch.linspace(-1.0,1.0, steps = resolution)
        grid_z1, grid_z2 = torch.meshgrid(woX, woY)
        grid = torch.stack([grid_z1, grid_z2], dim = -1)
        light_dir = grid.reshape((-1, 2)).cuda()
        with torch.no_grad():
            # z, logdet = realNVP.inverse(light_dir)
            # logp = prior.log_prob(z) + logdet 
            logp = realNVP.log_prob(light_dir)
        invalid_dirs = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) > 1
        logp[invalid_dirs] = 0.0
        pdfs = torch.exp(logp)
        pdfs[invalid_dirs] = 0.0
        A = 4/(resolution * resolution)
        pdfs *= 1/(A*torch.sum(pdfs))
        pdfs = pdfs.reshape(resolution, resolution)
        pdfs_3c = torch.stack([pdfs, pdfs, pdfs], dim=2).cpu()
        
        exr.write32(np.array(pdfs_3c, dtype=np.float32), out_path)
        if(plot_z):
            ### also print out the z!! see where it falls into
            # if realNVP.base == "condLars":
            log_prob = realNVP.q0.log_prob(light_dir).view(*pdfs.shape)
            log_prob = log_prob.to('cpu').view(*pdfs.shape)
            prob = torch.exp(log_prob)
            prob[torch.isnan(prob)] = 0
            # prob_base = prob.data.numpy()
            prob_base = torch.stack([prob, prob, prob], dim=2).detach().numpy()
            exr.write32(np.array(prob_base, dtype=np.float32), plot_z_outpath) 
            
        realNVP.train()
        return pdfs

    def render_2d_func_sample(self, out_path, realNVP, prior, resolution, num_samples): #by histogram
            realNVP.eval()
            z = torch.normal(0, 1, size = (num_samples, 2)).cuda()
            with torch.no_grad():
                # x, _ = realNVP(z)
                x, log_q = realNVP.sample(num_samples)
            x = x.cpu().detach().numpy()
            H, xedges, yedges = np.histogram2d(x=x[:,0], y=x[:,1], range=[[-1,1],[-1,1]], bins=(resolution,resolution),density=False)
            A = 4/(resolution * resolution)
            H *= 1.0/(A*num_samples)
            H_3c = np.stack([H, H, H], axis=2)
            exr.write32(np.array(H_3c, dtype=np.float32), out_path)
            realNVP.train()
            return H

####### for xyz and new model            
    def render_3d_func_pdf(self, out_path, realNVP, prior, resolution, plot_z_outpath, plot_z = False):
        realNVP.eval()
        woX = torch.linspace(-1.0,1.0, steps = resolution)
        woY = torch.linspace(-1.0,1.0, steps = resolution)
        grid_z1, grid_z2 = torch.meshgrid(woX, woY)
        grid = torch.stack([grid_z1, grid_z2], dim = -1)
        light_dir = grid.reshape((-1, 2)).cuda()


        light_dir = torch.cat((light_dir, torch.sqrt(1 - torch.square(light_dir[:, 0]) - torch.square(light_dir[:,1])).unsqueeze(1)), dim=1) 
        with torch.no_grad():
            logp = realNVP.log_prob(light_dir)
        invalid_condition1 = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) >1
        logp[invalid_condition1 ] = 0.0
        pdfs = torch.exp(logp)
        pdfs[invalid_condition1] = 0.0
        A = 4/(resolution * resolution)
        pdfs *= 1/(A*torch.sum(pdfs))
        pdfs = pdfs.reshape(resolution, resolution)
        pdfs_3c = torch.stack([pdfs, pdfs, pdfs], dim=2).cpu()
        exr.write32(np.array(pdfs_3c, dtype=np.float32), out_path)
        realNVP.train()
        return pdfs
        
    def render_3d_func_sample(self, out_path, realNVP, prior, resolution, num_samples): #by histogram
        realNVP.eval()
        with torch.no_grad():
            x, log_q = realNVP.sample(num_samples)
        #normalize x and y 
        norm = torch.sqrt(torch.square(x[:,0]) + torch.square(x[:,1]) + torch.square(x[:,2]))
        x[:, 0] /= norm 
        x[:, 1] /= norm
        x = x.cpu().detach().numpy()
        H, xedges, yedges = np.histogram2d(x=x[:,0], y=x[:,1], range=[[-1,1],[-1,1]], bins=(resolution,resolution),density=False)
        A = 4/(resolution * resolution)
        H *= 1.0/(A*num_samples)
        H_3c = np.stack([H, H, H], axis=2)
        exr.write32(np.array(H_3c, dtype=np.float32), out_path)
        realNVP.train()
        return H

    #for neural textures
    def render_2d_func_pdf_nt(self, out_path, realNVP, wi, loc, prior=None, resolution=128, plot_z_outpath=None, plot_z=False, png=False):
        woX = torch.linspace(-1.0,1.0, steps = resolution)
        woY = torch.linspace(-1.0,1.0, steps = resolution)
        grid_z1, grid_z2 = torch.meshgrid(woX, woY)
        grid = torch.stack([grid_z1, grid_z2], dim = -1)
        light_dir = grid.reshape((-1, 2)).cuda()
        uv_input = torch.tile(loc, (light_dir.shape[0],)).reshape(-1,2).cuda()
        wi_input = torch.tile(wi, (light_dir.shape[0],)).reshape(-1,2).cuda()
        with torch.no_grad():
            # z, logdet = realNVP.inverse(light_dir, wi_input, uv_input)
            # logp = prior.log_prob(z) + logdet 
            logp = realNVP.log_prob(light_dir, wi_input, uv_input)
        invalid_dirs = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) > 1
        logp[invalid_dirs] = 0.0
        pdfs = torch.exp(logp)
        pdfs[invalid_dirs] = 0.0
        A = 4/(resolution * resolution)
        pdfs *= 1/(A*torch.sum(pdfs))
        pdfs = pdfs.reshape(resolution, resolution)
        pdfs_3c = torch.stack([pdfs, pdfs, pdfs], dim=2).cpu()
        exr.write32(np.array(pdfs_3c, dtype=np.float32), out_path) 
        if(png):
            cv2.imwrite(out_path+".png",self.tonemap(np.array(pdfs_3c, dtype=np.float32))*255.0)
        if(plot_z):
            # log_prob = realNVP.log_prob(wi_input, None)
            conditional_vector = realNVP.compute_cond(wi_input, uv_input)
            # conditional_vector = utils.encoding.positional_encoding_1(wi_input, 5)
            if realNVP.base == "condLars":
                log_prob = realNVP.q0.log_prob(light_dir, conditional_vector).to('cpu').view(*pdfs.shape)
            elif realNVP.base == "condLars_uni":
                x = (light_dir + 1.0 )*0.5 ##TODO remove
                log_prob = realNVP.q0.log_prob(x, conditional_vector).to('cpu').view(*pdfs.shape)
            elif realNVP.base == "condGauss":
                _,_,log_prob= realNVP.q0.forward(light_dir, conditional_vector,eps = 0, reverse = False)
                log_prob = log_prob.to('cpu').view(*pdfs.shape)
            else:
                x = (light_dir + 1.0 )*0.5 ##TODO remove
                log_prob = realNVP.q0.log_prob(x).to('cpu').view(*pdfs.shape)
                
            prob = torch.exp(log_prob)
            prob[torch.isnan(prob)] = 0
            # prob_base = prob.data.numpy()
            prob_base = torch.stack([prob, prob, prob], dim=2).numpy()
            exr.write32(np.array(prob_base, dtype=np.float32), plot_z_outpath) 
        return pdfs
        

    

    def render_2d_func_sample_nt(self, out_path, realNVP,wi, loc, prior=None, resolution=128, num_samples=128*128*100): #by histogram
        uv_input = torch.tile(loc, (num_samples,)).reshape(-1,2).cuda()
        wi_input = torch.tile(wi, (num_samples,)).reshape(-1,2).cuda()
        with torch.no_grad():
            # x, _ = realNVP(z, wi_input, uv_input)
            x, log_q = realNVP.sample( wi_input, uv_input,num_samples)
        x = x.cpu().detach().numpy()
        H, xedges, yedges = np.histogram2d(x=x[:,0], y=x[:,1], range=[[-1,1],[-1,1]], bins=(resolution,resolution),density=False)
        invalid_dirs = np.square(x[:,0]) + np.square(x[:,1]) > 1
        print("invalid percentage : ", (np.sum(invalid_dirs)/num_samples*100))
        A = 4/(resolution * resolution)
        H *= 1.0/(A*num_samples)
        H_3c = np.stack([H, H, H], axis=2)
        exr.write32(np.array(H_3c, dtype=np.float32), out_path)
        return H

def neumip_renderer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=["pointlight", "btf"], default="btf")
    # parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="../tortoise_shell.ckpt")
    # parser.add_argument("--checkpoint", type=str, default="../../stylized_wool.ckpt")

    parser.add_argument("--dataset", type=str, default="/home/bingxu/projects/NeuMIP/res/datasets/shell.hdf5")
    parser.add_argument("--output", default=os.path.join("renders/fabric_moving_light"), help="dir to render outputs")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--resolution", default=[512, 512]) #type=list[int], 
    parser.add_argument("--frames", type=int, default=1)
    args = parser.parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:0"
    else:
        raise ValueError("CPU not supported")

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.exists(ckpt):
        raise ValueError("Checkpoint doesn't exist")

    model = interfaces.NeuMIPv1Module.load_from_checkpoint(ckpt, strict=False)

    model.model.initial_sigma = 1
    model.model.iterations_to_sigma_1 = 1
    conf = model.conf

    if args.verbose:
        print("Model parameters:")
        for n, p in model.named_parameters():
            print(f"  {n}, {p.mean().item():.4f} {p.std().item():.4f}")

    model.to(device)
    model.eval()
    model.freeze()
    for param in model.parameters():
        param.requires_grad=False
    
    renderer = NeuMipRenderer(args.resolution, model, dataset=args.dataset, device=device)
    return renderer
    # renderer.render_btf_queries(args.frames, args.output)
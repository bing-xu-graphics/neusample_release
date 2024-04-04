
import argparse, os, sys
from logging import root

from xml.etree.ElementTree import VERSION
import numpy as np
from render import Renderer
import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
import pickle
import math
from sys import exit
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
import os
# Insert root dir in path
p = os.path.abspath('.')
sys.path.insert(1, p)
from constatns import *
from torch.utils.data import Dataset, DataLoader
# from dataset_samplep import BSDFSamplingPDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch import distributions
# from ipypb import ipb
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)

from fast_histogram import histogram1d, histogram2d
import seaborn as sns


from bsdf_visualizer import *

from torch.utils.data import DataLoader

from neuis import realnvp_conditional 
from neumat import interfaces, lights, datasets
from utils import exr, ops, la2
from neumip_renderer import *
def rgb2gray(rgb):
    gray = torch.zeros(rgb.shape[0], 1)
    gray[:,0] = (rgb[:,0] + rgb[:,1] + rgb[:,2])/3.0
    return gray


# class NeuMipRenderer:
#     def __init__(self, resolution, neumat, dataset=None, device="cuda:0"):
#         self.width = resolution[0]
#         self.height = resolution[1]
#         self.device = device

#         self.model = neumat
        
#         self.dataset = dataset

#     def render_query(self, camera_dir, light_dir, uv):
#         with torch.no_grad():
#             rgb_prediction = self.model(camera_dir, light_dir, uv)
#             return rgb2gray(rgb_prediction)


#     def render_btf_sampling(self, frames, outdir, realnvp):
#         if self.dataset is None or not os.path.exists(self.dataset):
#             raise ValueError("Valid input dataset required to visualize btf queries")

#         for frame in range(10):
#             #generate random camera_dir and uv 
#             fig, ax = plt.subplots(1,2)
#             fig.set_size_inches(10,5)
#             THETA_BIN = 50
#             PHI_BIN = 100

#             camera_dir = torch.rand()*2.0 - 1.0
                
#             uv = torch.rand()*2.0 - 1.0
                
#             #using realnvp to sample
#             z = torch.rand(VISUALIZE_NUM_SAMPLES, 2).cuda()
#             camera_dir = (torch.rand(1,2)*2.0 - 1.0).repeat(VISUALIZE_NUM_SAMPLES).cuda()
#             uv = (torch.rand(1,2)*2.0 - 1.0).repeat(VISUALIZE_NUM_SAMPLES).cuda()
#             x, _ = realNVP(z, camera_dir, uv)
#             x = (x+1.0)/2.0

#             H, xedges, yedges = np.histogram2d(x=x[:,0], y=x[:,1], range=[[0,1],[0,1]], bins=(THETA_BIN,PHI_BIN),density=True)
#             H = H/np.sum(H)
#             ax[0].imshow(H)

#             theta = torch.linspace(-1, 1, steps=THETA_BIN)
#             phi = torch.linspace(-1,1, steps=PHI_BIN)
#             grid_z1, grid_z2 = torch.meshgrid(theta, phi)
#             grid = torch.stack([grid_z1, grid_z2], dim = -1)
#             light_dir = grid.reshape((-1, 2)).cuda()

#             # camera_dir_tile = torch.tile(camera_dir, (light_dir.shape[0],)).reshape((-1,2)).cuda() 
#             # uv_tile = torch.tile(uv, (light_dir.shape[0],)).reshape((-1,2)).cuda()   
            
#             rgb_prediction = self.model(camera_dir, light_dir, uv)
#             neumip_prediction = rgb2gray(rgb_prediction)
#             neumip_prediction = neumip_prediction.reshape(THETA_BIN, PHI_BIN)
#             neumip_prediction = neumip_prediction.cpu().numpy()
#             neumip_prediction = neumip_prediction/np.sum(neumip_prediction)
#             ax[1].imshow(neumip_prediction)
            
#             visualize_bsdf_side_by_side(H, bsdf, os.path.join(ROOT_DIR, "/output/neumip_%s_polar_frame_%d.png"%(exp_name,frame)))
            

#     def render_btf_queries(self, frames, outdir):
#         if self.dataset is None or not os.path.exists(self.dataset):
#             raise ValueError("Valid input dataset required to visualize btf queries")

#         neumipdata = datasets.NeuMIPv1Dataset(self.dataset)

#         # Visualize inputs
#         neumipdata.dump_h5(outdir, frames)

#         dataloader = DataLoader(neumipdata, batch_size=1, num_workers=0, shuffle=False)

#         curr_frame = 0
#         for batch_idx, batch in enumerate(dataloader):
#             camera_dir, light_dir, color, uv, cam_qr = batch

#             camera_dir = camera_dir.reshape(-1, camera_dir.shape[-1])
#             light_dir = light_dir.reshape(-1, light_dir.shape[-1])
#             uv = uv.reshape(-1, uv.shape[-1])
#             color = color.reshape(-1, color.shape[-1])

#             rgb_prediction = self.model(camera_dir, light_dir, uv)

#             rgb_prediction = rgb_prediction.reshape(self.width, self.height, -1).cpu()
#             outfile = os.path.join(outdir, "pred_" + str(curr_frame) + ".exr")
#             exr.write16(np.array(rgb_prediction, dtype=np.float32), outfile)

#             curr_frame += 1

#             if (curr_frame == frames):
#                 break


#     def render_2d_slice_gt(self, out_path, camera_dir, uv, resolution):#changing one direction
#         woX = torch.linspace(-1.0,1.0, steps = resolution)
#         woY = torch.linspace(-1.0,1.0, steps = resolution)
#         grid_z1, grid_z2 = torch.meshgrid(woX, woY)
#         grid = torch.stack([grid_z1, grid_z2], dim = -1)
#         light_dir = grid.reshape((-1, 2)).cuda()
#         camera_dir_tensor = torch.tile(camera_dir, (light_dir.shape[0],)).reshape(-1,2).cuda()
#         uv_tensor = torch.tile(uv, (light_dir.shape[0],)).reshape(-1,2).cuda()
#         rgb_pred = self.model(camera_dir_tensor, light_dir, uv_tensor)
#         lumi_pred = self.rgb2lum(rgb_pred)
#         A = 4/(resolution * resolution)
#         lumi_pred *= 1/(A * torch.sum(lumi_pred[:,0]))
#         lumi_pred = lumi_pred.reshape(resolution, resolution, -1).cpu()
#         exr.write32(np.array(lumi_pred, dtype=np.float32), out_path)
#         return lumi_pred

#     def render_2d_func_pdf(self, out_path, realNVP, prior, resolution):
#         realNVP.eval()
#         woX = torch.linspace(-1.0,1.0, steps = resolution)
#         woY = torch.linspace(-1.0,1.0, steps = resolution)
#         grid_z1, grid_z2 = torch.meshgrid(woX, woY)
#         grid = torch.stack([grid_z1, grid_z2], dim = -1)
#         light_dir = grid.reshape((-1, 2)).cuda()
        
#         with torch.no_grad():
#             z, logdet = realNVP.inverse(light_dir)
#             logp = prior.log_prob(z) + logdet 
#         pdfs = torch.exp(logp)
#         A = 4/(resolution * resolution)
#         pdfs *= (1/A*torch.sum(pdfs))
#         pdfs = pdfs.reshape(resolution, resolution)
#         pdfs_3c = torch.stack((pdfs, pdfs, pdfs), dim=1).cpu()
#         exr.write32(np.array(pdfs_3c, dtype=np.float32), out_path)
        
#         return pdfs
            
        

#     def render_2d_func_sample(self, out_path, realNVP, prior, resolution, num_samples): #by histogram
#         realNVP.eval()
#         z = torch.normal(0, 1, size = (num_samples, 2)).cuda()
#         with torch.no_grad():
#             x, _ = realNVP(z)
#         x = x.cpu().detach().numpy()
#         H, xedges, yedges = np.histogram2d(x=x[:,0], y=x[:,1], range=[[-1,1],[-1,1]], bins=(self.THETA_BIN,self.PHI_BIN),density=False)
#         A = 4/(resolution * resolution)
#         H *= (1.0/(A*num_samples))
#         H_3c = torch.stack((H,H, H), dim=1).cpu()
#         exr.write32(np.array(H_3c, dtype=np.float32), out_path)
#         return H



# def neumip_renderer():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mode', type=str, choices=["pointlight", "btf"], default="btf")
#     # parser.add_argument("--checkpoint", type=str, required=True)
#     parser.add_argument("--checkpoint", type=str, default="/home/bingxu/projects/neumip_adobe/tortoise_shell.ckpt")
#     parser.add_argument("--dataset", type=str, default="/home/bingxu/projects/NeuMIP/res/datasets/shell.hdf5")
#     parser.add_argument("--output", default=os.path.join("renders/fabric_moving_light"), help="dir to render outputs")
#     parser.add_argument("--verbose", default=False, action="store_true")
#     parser.add_argument("--resolution", default=[512, 512]) #type=list[int], 
#     parser.add_argument("--frames", type=int, default=1)
#     args = parser.parse_args()

#     device = "cpu"
#     if torch.cuda.is_available():
#         device = f"cuda:0"
#     else:
#         raise ValueError("CPU not supported")

#     ckpt = os.path.abspath(args.checkpoint)
#     if not os.path.exists(ckpt):
#         raise ValueError("Checkpoint doesn't exist")

#     model = interfaces.NeuMIPv1Module.load_from_checkpoint(ckpt, strict=False)

#     model.model.initial_sigma = 1
#     model.model.iterations_to_sigma_1 = 1
#     conf = model.conf

#     if args.verbose:
#         print("Model parameters:")
#         for n, p in model.named_parameters():
#             print(f"  {n}, {p.mean().item():.4f} {p.std().item():.4f}")

#     model.to(device)
#     model.eval()
#     model.freeze()
#     for param in model.parameters():
#         param.requires_grad=False
    
#     renderer = NeuMipRenderer(args.resolution, model, dataset=args.dataset, device=device)
#     return renderer
#     # renderer.render_btf_queries(args.frames, args.output)

# def range_query(x_range, y_range, U_sat, sigma, V_sat):
#     [x1,x2] = x_range
#     [y1,y2] = y_range
#     x1 = int(x1)
#     x2 = int(x2)
#     y1 = int(y1)
#     y2 = int(y2)
#     # print("(%d,%d) - (%d,%d)"%(x1,y1,x2,y2))
#     RANK = len(sigma)
#     value = 0.
#     for i in range(RANK):
#         U_i = U_sat[:, i]
#         V_i = V_sat[i, :]
#         umin = 0.
#         if(x1>0):
#             umin = U_i[x1-1,0]
#         umax = U_i[x2,0]
#         uavg = (umax - umin)/(x2 - x1 +1)
#         vmin = 0.
#         if(y1>0):
#             vmin = V_i[0,y1-1]
#         vmax = V_i[0,y2]
#         vavg = (vmax - vmin)/(y2 - y1 +1)
#         value = value + uavg * sigma[i,i] * vavg 
#     # print(x1,x2,y1,y2,value)
#     return value

# def sample_2d_tensor(U_sat, sigma, V_sat,img_width=128, img_height=128):

#     #hierachical sampling
#     level = int(math.log2(img_width))
#     # print("level", level)
#     lefttop = [0,0]
#     q = 1.
#     for i in range(1,level + 1):
#         prob00 = range_query(lefttop[0]+np.asarray([0, img_height/pow(2, i)-1]), lefttop[1]+np.asarray([0, img_width/pow(2, i)-1]), U_sat, sigma, V_sat)
#         prob01 = range_query(lefttop[0]+np.asarray([0, img_height/pow(2, i)-1]), lefttop[1]+np.asarray([img_width/pow(2, i), img_width/pow(2, i-1)-1]), U_sat, sigma, V_sat )
#         prob10 = range_query(lefttop[0]+np.asarray([img_height/pow(2, i), img_height/pow(2, i-1) -1]),  lefttop[1]+np.asarray([0, img_width/pow(2, i)-1]) ,U_sat, sigma, V_sat)
#         prob11 = range_query(lefttop[0]+np.asarray([img_height/pow(2, i), img_height/pow(2, i-1) -1]),  lefttop[1]+np.asarray([img_width/pow(2, i), img_width/pow(2, i-1)-1]) ,U_sat, sigma, V_sat)
#         pdfs = [prob00, prob01, prob10, prob11] / (prob00 + prob01 + prob10 + prob11)
#         # print("pdfs", pdfs)
#         cdfs = pdfs.cumsum(axis=0)
#         # print("cdfs", cdfs)
#         sample = random.uniform(0, 1)
#         k = 0 
#         while(sample > cdfs[k]):
#             k +=1
#         assert(k<4)
#         if (k==0):
#             pass #keep the leftmost
#         elif (k==1):
#             lefttop[1] += img_width/pow(2, i)
#         elif (k==2):
#             lefttop[0] += img_height/pow(2, i)
#         else:
#             lefttop[1] += img_width/pow(2, i)
#             lefttop[0] += img_height/pow(2, i)
#         # print("level = %d, k = %d, lefttop = (%d,%d)"%(i, k, lefttop[0], lefttop[1]))
#         # print("new lefttop ", lefttop)
#         q *= pdfs[k] #accumulate the pdf

#     # offset = offset_tent()
#     # lefttop[0] = min(max(0,offset[0]+lefttop[0]), img_height-1)
#     # lefttop[1] = min(max(0,offset[1]+lefttop[1]), img_width-1)
#     assert(lefttop[0]>=0 and lefttop[0]< img_height)
#     assert(lefttop[1]>=0 and lefttop[1]< img_width)
#     # print(lefttop)

#     return lefttop, q

    
            
def test():
    #render a 2d slice for testing
    renderer = neumip_renderer()
    # uv = torch.Tensor([-0.5,-0.7])
    loc = [0.3, 0.5] #[-0.5,-0.7]#
    uv = torch.Tensor(loc)
    camera_dir_x = np.linspace(-1.0, 1.0, 5)
    camera_dir_y = np.linspace(-1.0, 1.0, 5)
    camera_dirs = []
    for x in camera_dir_x:
        for y in camera_dir_y:
            print(x,y)
            camera_dirs.append(torch.Tensor([x,y]))

    out_dir = "/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/output/2d_slice/loc_%f_%f"%(loc[0], loc[1])
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cnt = 0
    for camera_dir in camera_dirs:
        pred = renderer.render_2d_slice_gt(os.path.join(out_dir, "tortoise_2d_%d.exr"%(cnt)), uv, camera_dir, 512)
        cnt += 1


from bsdf_visualizer import ResultVisualizer
def verify_sampling(inpath, exrpath,resolution, num_samples):
    visualizer = ResultVisualizer(None, None, None)
    # "/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/data/samplep/2d/uv_0.300000_0.500000_wo_0.500000_0.500000_wi_samples_10k.npy"
    pdf_plot = visualizer.target_pdf(inpath, resolution, resolution, num_samples)
    # pdf_plot = np.flipud(pdf_plot)
    exr.write32(visualizer.gray2rgb(pdf_plot.astype("float32")) ,exrpath)

def verify_sampling_6d():
    visualizer = ResultVisualizer(None, None, None)
    pdf_plot = visualizer.target_pdf("/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/data/samplep/6d_neumip/neumip_samples_6d.npy",\
        50,50, 1024)
    # pdf_plot = np.flipud(pdf_plot)
    exr.write32(visualizer.gray2rgb(pdf_plot.astype("float32")) ,os.path.join("/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/data/samplep/6d_neumip/neumip_6d_samples_verify.exr"))

def generating_trainig_data_2d():
    renderer = neumip_renderer()
    root_dir = "/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/data/samplep/2d"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    resolution = 256
    loc = [0.996086, -0.972603] #[-0.5,-0.7]#7_plot_uv_0.996086_-0.972603_wo_-0.093781_-0.782236_wi_table
    uv = torch.Tensor(loc)
    wo = torch.Tensor([-0.093781, -0.782236])
    outpath = os.path.join(root_dir, "uv_%f_%f_wo_%f_%f_wi_table.npy"%(uv[0], uv[1], wo[0], wo[1]))
    exrpath = os.path.join(root_dir, "uv_%f_%f_wo_%f_%f_wi_table.exr"%(uv[0], uv[1], wo[0], wo[1]))
    
    gt_pdf = renderer.render_2d_slice_gt(exrpath, wo, uv, resolution, False) 
    # gt_pdf = gt_pdf[:,0].reshape(resolution, resolution).cpu()
    save_to_npy = gt_pdf[:,:,0]
    # save_to_npy *=4
    print(save_to_npy.shape)
    np.save(outpath, save_to_npy)
    return outpath
    print(torch.sum(save_to_npy))

from neuis import sampling
def importance_sample_wi(uv_wo, img,  num_samples):
    samples = list()
    resolution = len(img)
    print("resolution of img: ", resolution)
    distribution = sampling.Distribution2D(img, resolution, resolution)
    for i in range(num_samples):
        sampleWi, pdf = distribution.SampleContinuous(torch.rand(2))
        sampleWi[0] = sampleWi[0]*2.0 - 1.0
        sampleWi[1] = sampleWi[1]*2.0 -1.0
        if sampleWi[0]* sampleWi[0] + sampleWi[1] * sampleWi[1] > 1.0:
            continue
        samples.append(sampleWi)
    
    return samples

def test_importance_sample_wi():
    tic = time.perf_counter()
    root_dir = "/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/data/samplep/2d"
    loc = [0.3, 0.5] #[-0.5,-0.7]#
    uv = torch.Tensor(loc)
    wo = torch.Tensor([0.5,0.5])
    inpath = os.path.join(root_dir, "uv_%f_%f_wo_%f_%f_wi_table.npy"%(uv[0], uv[1], wo[0], wo[1]))
    data = np.load(inpath)
    samples = importance_sample_wi(None, data ,1024)
    outpath = os.path.join(root_dir, "uv_%f_%f_wo_%f_%f_wi_samples_1k.npy"%(uv[0], uv[1], wo[0], wo[1]))
    np.save(outpath, samples)
    verify_sampling(outpath, 50,1024)
    print(f"profile importance_sample_wi  in {time.perf_counter() - tic:0.4f} seconds")

from pybind import samplewi

def test_pybind_samplewi():
    renderer = neumip_renderer()
    tic = time.perf_counter()
    root_dir = "/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/data/samplep/2d"
    loc = [0.996086, -0.972603] #[-0.5,-0.7]#7_plot_uv_0.996086_-0.972603_wo_-0.093781_-0.782236_wi_table
    uv = torch.Tensor(loc)
    wo = torch.Tensor([-0.093781, -0.782236])
    #1)generate img
    imgpath = os.path.join(root_dir, "uv_%f_%f_wo_%f_%f_wi_table.npy"%(uv[0], uv[1], wo[0], wo[1]))
    exrpath = os.path.join(root_dir, "uv_%f_%f_wo_%f_%f_wi_table.exr"%(uv[0], uv[1], wo[0], wo[1]))
    
    gt_pdf = renderer.render_2d_slice_gt(exrpath, wo, uv, 256, False) 
    # gt_pdf = gt_pdf[:,0].reshape(resolution, resolution).cpu()
    save_to_npy = gt_pdf[:,:,0]
    np.save(imgpath, save_to_npy)

    data = np.load(imgpath)
    data = data.flatten()
    samples = samplewi.samplewi(data,1,1024*256) #2014
    samples = samples.reshape(-1,2)

    outpath = os.path.join(root_dir, "uv_%f_%f_wo_%f_%f_wi_samples_verify.npy"%(uv[0], uv[1], wo[0], wo[1]))
    exrpath = os.path.join(root_dir, "uv_%f_%f_wo_%f_%f_wi_samples_verify.exr"%(uv[0], uv[1], wo[0], wo[1]))
    np.save(outpath, samples)
    verify_sampling(outpath,exrpath, 256, 1024*256)
    print(f"profile test_pybind_samplewi  in {time.perf_counter() - tic:0.4f} seconds")

import multiprocessing as mp
from joblib import Parallel, delayed
#used to sample from 2d distributions
def generate_samples_for_one_location(renderer, root_dir,resolution, uv):
    #sample the projected hemisphere. union disk
    for i in range(64):
        idx += 1
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
        ##tabulating for wi 
        outpath = os.path.join(root_dir, "sample_%d_uv_%f_%f_wo_%f_%f_wi_table.npy"%(idx, uv[0], uv[1], woX, woY))
        exrpath = os.path.join(root_dir, "gt/sample_%d_uv_%f_%f_wo_%f_%f_wi_table.exr"%(idx, uv[0], uv[1], woX, woY))
        gt_pdf = renderer.render_2d_slice_gt(exrpath, wo, uv, resolution, True, False) 
        gt_pdf = gt_pdf[:,:,0]
        np.save(outpath, gt_pdf)


def generating_training_data_6d_in_parallel(): ###it doesn't work this way
    renderer = neumip_renderer()
    NUM_UV_VARY = 64*64 
    NUM_WO_VARY = 64
    NUM_WI_VARY = 256*256
    NUM_SAMPLING = 1024 #per (uv, wo) pair 
    n_workers = 2 * mp.cpu_count()
    print(f"{n_workers} workers are available")
    pool = mp.Pool(n_workers)
    root_dir = "/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/data/samplep/6d"
    gt_dir = os.path.join(root_dir, "gt")
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    idx = 0
    resolution = 256
    uv_wo_records = list()
    # u = torch.linspace(-1, 1, steps = 64)
    # v = torch.linspace(-1, 1, steps = 64)
    u = np.linspace(-1, 1, num = 64)
    v = np.linspace(-1, 1, num = 64)
    grid_u, grid_v = np.meshgrid(u, v)
    # grid = torch.stack([grid_u, grid_v], dim = -1)
    grid = np.stack([grid_u, grid_v], axis = -1)
    uv_list = grid.reshape((-1, 2))
    for uv in uv_list:
        proc = pool.apply_async(generate_samples_for_one_location, args=[renderer,root_dir,resolution, uv])
        # break
    pool.close()
    pool.join()
    # np.save(os.path.join("/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/data/samplep/","sample_6d_uv_wo_idx_nocosine.npy"), np.array(uv_wo_records))
            
def generating_training_data_6d():
    renderer = neumip_renderer()
    UV_DIM = 64
    NUM_WO_VARY = 64
    WI_RES = 256
    # UV_DIM = 2
    # NUM_WO_VARY = 4
    # WI_RES = 2

    root_dir = "/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/data/samplep/6d"
    gt_dir = os.path.join(root_dir, "gt")
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    
    #common variables for each loop
    wiX = torch.linspace(-1.0,1.0, steps = WI_RES)
    wiY = torch.linspace(-1.0,1.0, steps = WI_RES)
    grid_z1, grid_z2 = torch.meshgrid(wiX, wiY)
    gridwi = torch.stack([grid_z1, grid_z2], dim = -1)
    light_dir = gridwi.reshape((-1, 2)).cuda()
    invalid_dirs = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) > 1
    A = 4/(WI_RES * WI_RES)
    ##########################
    u = torch.linspace(-1, 1, steps = UV_DIM)
    v = torch.linspace(-1, 1, steps = UV_DIM)
    grid_u, grid_v = torch.meshgrid(u, v)
    grid = torch.stack([grid_u, grid_v], dim = -1)
    uv_list = grid.reshape((-1, 2))
    uv_wo_records = list()
    num_uv_record = 0
    total_idx = 0
    for uv in uv_list:
        num_uv_record +=1
        if(num_uv_record<1946):
            total_idx+=64
            continue
        uv_wo_records.clear()
        for i in range(NUM_WO_VARY):
            total_idx += 1
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
            # a pair of (uv, wo)
            uv_wo_records.append([*uv, *wo])
           
        assert(len(uv_wo_records) == NUM_WO_VARY)      
        uv_wo = torch.Tensor(uv_wo_records)
        # print("uv_wo.shape ",uv_wo.shape)
        camera_dir = uv_wo[:,2:]
        uv = uv_wo[:,:2]
        ##tabulating for wi 
        
        camera_dir_tensor = torch.tile(camera_dir, (light_dir.shape[0],)).reshape(-1,2).cuda()
        uv_tensor = torch.tile(uv, (light_dir.shape[0],)).reshape(-1,2).cuda()
        light_dir_tensor = torch.tile(light_dir, (camera_dir.shape[0],1)).cuda()
        # print("input shapes ", camera_dir_tensor.shape, light_dir_tensor.shape, uv_tensor.shape)
        with torch.no_grad():
            rgb_pred = renderer.model(camera_dir_tensor, light_dir_tensor, uv_tensor)
        
        #decouple the images
        wi_list = rgb_pred.chunk(chunks = len(uv_wo_records), dim=0)
        
        
        for idx in range(len(uv_wo_records)):
            uv_wo = uv_wo_records[idx]
            wi_img = wi_list[idx]
            # valid_dirs = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) <= 1.0
            wi_img[invalid_dirs] = 0.0
            lumi_pred = renderer.rgb2lum(wi_img)
            lumi_pred *= 1/(A * torch.sum(lumi_pred[:,0])) #no need for normalization for sampling
            lumi_pred = lumi_pred.reshape(WI_RES, WI_RES, -1).cpu()
            outpath = os.path.join(root_dir, "sample_%d_uv_%f_%f_wo_%f_%f_wi_table.npy"%(total_idx-NUM_WO_VARY+idx, uv_wo[0], uv_wo[1], uv_wo[2], uv_wo[3]))
            # exrpath = os.path.join(root_dir, "gt/sample_%d_uv_%f_%f_wo_%f_%f_wi_table.exr"%(idx, uv_wo[0], uv_wo[1], uv_wo[2], uv_wo[3]))
            gt_pdf = lumi_pred[:,:,0]
            # exr.write32(np.array(lumi_pred, dtype=np.float32), exrpath)
            np.save(outpath, gt_pdf)
        print("total_idx: ", total_idx)


def handle_a_chunk(renderer,uv_wo_records, light_dir, invalid_dirs,A, root_dir, WI_RES, total_idx):
    # tic = time.perf_counter()
    
    uv_wo = torch.Tensor(uv_wo_records)
    uv = uv_wo[:,:2]
    camera_dir = uv_wo[:, 2:]
    ##tabulating for wi 
    camera_dir_tensor = torch.tile(camera_dir, (light_dir.shape[0],)).reshape(-1,2).cuda()
    uv_tensor = torch.tile(uv, (light_dir.shape[0],)).reshape(-1,2).cuda()
    light_dir_tensor = torch.tile(light_dir, (camera_dir.shape[0],1)).cuda()
    # print("input shapes ", camera_dir_tensor.shape, light_dir_tensor.shape, uv_tensor.shape)
    # print(f"profile 0  in {time.perf_counter() - tic:0.4f} seconds")
    # tic = time.perf_counter()
    with torch.no_grad():
        rgb_pred = renderer.model(camera_dir_tensor, light_dir_tensor, uv_tensor)
        wi_list = rgb_pred.chunk(chunks =len(uv_wo_records), dim=0)
    # print(f"profile 1  in {time.perf_counter() - tic:0.4f} seconds")
    # tic = time.perf_counter()
    #decouple the images
    
    with torch.no_grad():
        for idx in range(len(uv_wo_records)):
            total_idx+=1
            uv_wo = uv_wo_records[idx]
            wi_img = wi_list[idx].cpu()
            # valid_dirs = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) <= 1.0
            wi_img[invalid_dirs] = 0.0
            lumi_pred = renderer.rgb2lum(wi_img)
            lumi_pred *= 1/(A * torch.sum(lumi_pred[:,0])) #no need for normalization for sampling
            lumi_pred = lumi_pred.reshape(WI_RES, WI_RES, -1)#.cpu()
            outpath = os.path.join(root_dir, "sample_%d_uv_%f_%f_wo_%f_%f_wi_table.npy"%(total_idx, uv_wo[0], uv_wo[1], uv_wo[2], uv_wo[3]))
            # exrpath = os.path.join(root_dir, "gt/sample_%d_uv_%f_%f_wo_%f_%f_wi_table.exr"%(idx, uv_wo[0], uv_wo[1], uv_wo[2], uv_wo[3]))
            gt_pdf = lumi_pred[:,:,0]
            # exr.write32(np.array(lumi_pred, dtype=np.float32), exrpath)
            np.save(outpath, gt_pdf)
    print("total_idx: ", total_idx)
    # print(f"profile 2  in {time.perf_counter() - tic:0.4f} seconds")
    # tic = time.perf_counter()
    return total_idx

import gc
import time


def sample_wo(NUM_WO_VARY):
    wo = torch.rand(NUM_WO_VARY, 2) * 2.0 - 1.0

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

def generate_training_data_4d():
    renderer = neumip_renderer()
    UV_DIM = 1
    NUM_WO_VARY = 1024
    WI_RES = 256
    root_dir = "/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/data/samplep/4d"
    gt_dir = os.path.join(root_dir, "gt")
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    
    #common variables for each loop
    wiX = torch.linspace(-1.0,1.0, steps = WI_RES)
    wiY = torch.linspace(-1.0,1.0, steps = WI_RES)
    grid_z1, grid_z2 = torch.meshgrid(wiX, wiY)
    gridwi = torch.stack([grid_z1, grid_z2], dim = -1)
    light_dir = gridwi.reshape((-1, 2)).cuda()
    invalid_dirs = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) > 1
    A = 4/(WI_RES * WI_RES)
    ##########################
    camera_dir = sample_wo(NUM_WO_VARY)
    uv = torch.Tensor([0.3,0.5])
    uv_tensor = torch.tile(uv, (camera_dir.shape[0],)).reshape(-1,2)
    uv_wo_records = torch.cat((uv_tensor, camera_dir), dim=-1)
    assert(len(uv_wo_records) ==  NUM_WO_VARY*UV_DIM*UV_DIM)   
    uv_wo_pair_size = len(uv_wo_records)
    BATCH_SIZE = 64*4#100000

    chunk_start = 0
    total_idx = 0
    # chunk_start = 450048
    # total_idx = 450048
    print("start batch processing")
    gc.disable()
    while chunk_start < uv_wo_pair_size:
        tic = time.perf_counter()
        chunk_end = min(uv_wo_pair_size, chunk_start + BATCH_SIZE)
        total_idx = handle_a_chunk(renderer, uv_wo_records[chunk_start:chunk_end], light_dir, invalid_dirs,A, root_dir, WI_RES, total_idx)
        if chunk_start % (64*16) == 0:
            gc.collect()
        chunk_start = chunk_end
        toc = time.perf_counter()
        print(f"one batch in {toc - tic:0.4f} seconds")
    gc.enable()


def generating_training_data_6d_faster():
    renderer = neumip_renderer()
    UV_DIM = 512
    NUM_WO_VARY = 2
    WI_RES = 256
    # UV_DIM = 2
    # NUM_WO_VARY = 4
    # WI_RES = 256

    root_dir = "/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/data/samplep/6d_large"
    gt_dir = os.path.join(root_dir, "gt")
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    
    #common variables for each loop
    wiX = torch.linspace(-1.0,1.0, steps = WI_RES)
    wiY = torch.linspace(-1.0,1.0, steps = WI_RES)
    grid_z1, grid_z2 = torch.meshgrid(wiX, wiY)
    gridwi = torch.stack([grid_z1, grid_z2], dim = -1)
    light_dir = gridwi.reshape((-1, 2)).cuda()
    invalid_dirs = torch.square(light_dir[:,0]) + torch.square(light_dir[:,1]) > 1
    A = 4/(WI_RES * WI_RES)
    ##########################
    u = torch.linspace(-1, 1, steps = UV_DIM)
    v = torch.linspace(-1, 1, steps = UV_DIM)
    grid_u, grid_v = torch.meshgrid(u, v)
    grid = torch.stack([grid_u, grid_v], dim = -1)
    uv_grid = grid.reshape((-1, 2))

    uv_wo_records = list()
    camera_dir = sample_wo(NUM_WO_VARY)
    uv_tensor = torch.tile(uv_grid, (camera_dir.shape[0],)).reshape(-1,2)
    camera_dir_tensor = torch.tile(camera_dir, (uv_grid.shape[0],1))
    uv_wo_records = torch.cat((uv_tensor, camera_dir_tensor), dim=-1)

    # for uv in uv_grid:
    #     for i in range(NUM_WO_VARY):
    #         wo = torch.rand(2)*2.0 - 1.0
    #         if ( wo[0] == 0 and wo[1] == 0):
    #             woX = 0
    #             woY = 0
    #         else:
    #             if(torch.abs(wo[0]) > torch.abs(wo[1])):
    #                 r = wo[0]
    #                 theta = PI_over_4 * (wo[1]/ wo[0])
    #             else:
    #                 r = wo[1]
    #                 theta = PI_over_2 - PI_over_4 * (wo[0]/wo[1])
    #             woX = r * torch.cos(theta)
    #             woY = r * torch.sin(theta)
    #         wo[0] = woX 
    #         wo[1] = woY
    #         # a pair of (uv, wo)
    #         uv_wo_records.append([*uv, *wo])
           
    assert(len(uv_wo_records) ==  NUM_WO_VARY*UV_DIM*UV_DIM)   
    uv_wo_pair_size = len(uv_wo_records)

    
    #handle in batch
    BATCH_SIZE = 64*4#100000
    # chunk_size = uv_wo_pair_size // BATCH_SIZE# + (uv_wo_pair_size % BATCH_SIZE >0)
    
    # chunk_start = 0
    # total_idx = 0
    chunk_start = 450048
    total_idx = 450048
    print("start batch processing")
    gc.disable()
    while chunk_start < uv_wo_pair_size:
        tic = time.perf_counter()
        # if chunk_start % (64*64) == 0:
        #     gc.enable()
        chunk_end = min(uv_wo_pair_size, chunk_start + BATCH_SIZE)
        # uv_wo_records_chunk = 
        total_idx = handle_a_chunk(renderer, uv_wo_records[chunk_start:chunk_end], light_dir, invalid_dirs,A, root_dir, WI_RES, total_idx)
        if chunk_start % (64*16) == 0:
            gc.collect()
        chunk_start = chunk_end
        toc = time.perf_counter()
        print(f"one batch in {toc - tic:0.4f} seconds")
    gc.enable()
        
    
from neuis import dataset_online_samplep_4d
if __name__ == "__main__":
    # test()
    # outpath = generating_trainig_data_2d()
    # verify_sampling(outpath, 50,256*1024)
    # generating_training_data_6d()
    # generating_training_data_6d_in_parallel()
    # generating_training_data_6d_faster()
    # verify_sampling_6d()
    # test_importance_sample_wi()
    # test_pybind_samplewi()
    ds = dataset_online_samplep_4d.NTOnlineSamplingPDataset4D()
    ds.__test_data_generation__()

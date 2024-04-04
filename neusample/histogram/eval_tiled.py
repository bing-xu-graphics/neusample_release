import torch
import torch.nn as nn
import torch.nn.functional as NF

import numpy as np
import math
from scipy.spatial.transform import Rotation

from model.mlps import mlp,PositionalEncoding
from model.neumip import NeuMIP
from model.base2d import Base2DInference
from model.nsf import NSF_prior
from model.histogram2d import Histogram2DInference
from model.gmm import GMM,BaseLine
from utils.mitsuba_helper import *


import gc
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ['MI_DEFAULT_VARIANT'] = 'cuda_ad_rgb'
import cv2


import drjit as dr
import mitsuba as mi
#mitsuba.set_variant('scalar_rgb')
mi.set_variant('cuda_ad_rgb')

import matplotlib.pyplot as plt

import time
from tqdm import tqdm
from argparse import ArgumentParser


def diffuse_sampler(samples,cond_feature):
    theta = torch.asin(torch.sqrt(samples[...,0]))
    phi = 2*math.pi*samples[...,1]
    sin_theta = torch.sin(theta)
    x = sin_theta*torch.cos(phi)
    y = sin_theta*torch.sin(phi)
    wi = torch.stack([x,y],dim=-1)
    pdf = torch.ones_like(y)/math.pi
    return wi,pdf

def diffuse_pdf(wi,cond_feature):
    return torch.ones_like(wi[...,0])/math.pi

def batched_render(scene,spp,seed):
    sensor = scene.sensors()[0]
    sensor_param = mi.traverse(sensor)
    W,H = sensor_param['film.size']
    img = torch.zeros((H,W,3),device=device)
    MAX_LOAD = 20*20*512
    crop_num = math.ceil(math.sqrt((W*H*spp*1.0)/MAX_LOAD))
    w,h = math.floor(W*1.0/crop_num),math.floor(H*1.0/crop_num)
    H_,W_ = math.ceil(H*1.0/h),math.ceil(W*1.0/w)
    print(w,h)
    for inds in tqdm(range(H_*W_)):
        i = inds // W_
        j = inds % W_
        w0,w1 = w*j,min(w*j+w,W)
        h0,h1 = h*i,min(h*i+h,H)
        sensor_param['film.crop_size'] = [w1-w0,h1-h0]
        sensor_param['film.crop_offset'] = [w0,h0]
        sensor_param.update()
        img_ = mi.render(scene,spp=spp,seed=seed)
        seed += 1
        img[h0:h1,w0:w1] = img_.torch()
    img = img.cpu()
    return img.numpy(),seed

if __name__ == '__main__':
    dr.set_flag(dr.JitFlag.VCallRecord, False)
    parser = ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('--neumip_path', type=str, required=True)
    parser.add_argument('--sampler_path', type=str,default=None)
    parser.add_argument('--threshold', type=str,default='0.98')
    parser.add_argument('--sampler', type=str, default='diffuse')
    parser.add_argument('--image_path', type=str, default='outputs/lowdis')
    parser.add_argument('--scene_file', type=str,default='notebooks/scene1/distributed3.xml')
    
    args = parser.parse_args()
    
    # gpu device
    device = torch.device(0)
    
    # model path
    NEUMIP_PATH = args.neumip_path
    SAMPLER_PATH = args.sampler_path
    SAMPLER = args.sampler
    
    # metric file
    IMAGE_PATH = args.image_path
    # material name
    MAT_NAME = NEUMIP_PATH.split('/')[-1].split('.')[0]
    
    # scene path, configureation
    SCENE_FILE = args.scene_file
    
    UV_SCALE = 2
    THRESHOLD = float(args.threshold)
    
    
    SPP = [4,16,64,256]
    
    
    neumip = NeuMIP()
    neumip.load_state_dict(torch.load(NEUMIP_PATH,map_location='cpu'))
    for p in neumip.parameters():
        p.requires_grad=False
    neumip.to(device)



    # define sampling model
    if SAMPLER == 'mixture histogram':
        # load sampler
        state_dict = torch.load(SAMPLER_PATH,map_location='cpu')
        model = Base2DInference(64,100,10)
        model.load_state_dict(state_dict,strict=False)
        model.init_pdf()

    elif SAMPLER == 'histogram':
        state_dict = torch.load(SAMPLER_PATH,map_location='cpu')['state_dict']
        weight = {}
        for k,v in state_dict.items():
            if 'base2d.' in k:
                weight[k.replace('base2d.','')] = v
        model = Histogram2DInference(32)
        model.load_state_dict(weight)

    elif SAMPLER == 'gmm':
        state_dict = torch.load(SAMPLER_PATH,map_location='cpu')['net']

        weight = {}
        for k,v in state_dict.items():
            if 'net.net.' in k:
                weight[k.replace('net.net.','mlp.')] = v
        model = GMM(2)
        model.load_state_dict(weight)

    elif SAMPLER == 'baseline':
        state_dict = torch.load(SAMPLER_PATH,map_location='cpu')['net']

        weight = {}
        for k,v in state_dict.items():
            if 'net.net.' in k:
                weight[k.replace('net.net.','mlp.')] = v
        model = BaseLine()
        model.load_state_dict(weight)
        
    elif SAMPLER == 'normflow2':
        state_dict = torch.load(SAMPLER_PATH,map_location='cpu')['net']    
        weight = {}
        for k,v in state_dict.items():
            if 'q0.net.net.' in k:
                weight[k.replace('q0.net.net.','mlp0.')] = v
            if 'flows.0.pw_rq_coupling.transform_net.net.' in k:
                weight[k.replace('flows.0.pw_rq_coupling.transform_net.net.','mlps.0.')] = v
            if 'flows.2.pw_rq_coupling.transform_net.net.' in k:
                weight[k.replace('flows.2.pw_rq_coupling.transform_net.net.','mlps.1.')] = v  

        model = NSF_prior()
        model.load_state_dict(weight)
        

    if SAMPLER == 'diffuse':
        model_sampler = diffuse_sampler
        sampler_pdf = diffuse_pdf
    else:
        for p in model.parameters():
            p.requires_grad=False
        model.to(device)
        def model_sampler(samples,cond_feature):
            wi,pdf = model(samples,cond_feature,inverse=True)
            return wi,pdf

        def sampler_pdf(wi,cond_feature):
            pdf = model(wi,cond_feature,inverse=False)[1]
            return pdf
        
        
    # define NeuMIPBSDF
    class NeuMIPBSDF(mi.BSDF):
        def __init__(self, props):
            mi.BSDF.__init__(self, props)
            reflection_flags   = mi.BSDFFlags.SpatiallyVarying|mi.BSDFFlags.DiffuseReflection|mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
            self.m_components  = [reflection_flags]
            self.m_flags = reflection_flags

        def sample(self, ctx, si, sample1, sample2, active):
            wi = si.wi.torch()
            
            s_dpdu = dr.dot(si.sh_frame.s[0], si.dp_du[0])[0]
            t_dpdv = dr.dot(si.sh_frame.t[0], si.dp_dv[0])[0]    
            if(s_dpdu<0):
                wi[:,0] *= -1
            if(t_dpdv<0):
                wi[:,1] *= -1

            flip = wi[...,-1:].sign()
            flip[flip==0] = 1
            wi *= flip

            uv = si.uv.torch()*UV_SCALE

            rgb_feature = neumip.get_rgb_texture(uv,wo=wi)
            cond_feature = torch.cat([rgb_feature,wi[...,:2]],-1)

            wo,pdf = model_sampler(
                torch.cat([sample2.torch(),sample1.torch().unsqueeze(-1)],-1),
                cond_feature)

            rr = wo[...,:2].pow(2).sum(-1)
            wo = torch.cat([wo,(1-rr).relu().sqrt().unsqueeze(-1)],-1)
            btf = neumip(uv,wi,wo)['brdf']
            btf[rr>THRESHOLD] = 0

            wo = NF.normalize(wo,dim=-1)*flip

            if(s_dpdu<0):
                wo[:,0] *=-1
            if(t_dpdv<0):
                wo[:,1] *=-1
        
            
            
            value = btf / pdf.unsqueeze(-1).clamp_min(1e-4)
            value[value.isnan()] = 0.0
            #value[pdf<1e-4] = 0.0

            pdf = pdf*wo[...,-1].relu()

            pdf_mi = dr.select(active,mi.Float(pdf),0.0)
            wo_mi = mi.Vector3f(wo[...,0],wo[...,1],wo[...,2])
            value_mi = mi.Vector3f(value[...,0],value[...,1],value[...,2])

            del wo,pdf,wi,flip,btf,rr,value,rgb_feature,cond_feature
            gc.collect()

            torch.cuda.empty_cache()

            bs = mi.BSDFSample3f()
            bs.pdf = pdf_mi
            bs.sampled_component = mi.UInt32(0)
            bs.sampled_type = mi.UInt32(+self.m_flags)
            bs.wo = wo_mi
            bs.eta = 1.0

            return (bs,value_mi)

        def eval(self, ctx, si, wo, active):
            wo = wo.torch()
            wi = si.wi.torch()
            if(dr.dot(si.sh_frame.s[0], si.dp_du[0])[0]<0):
                wi[:,0] *=-1
            if(dr.dot(si.sh_frame.t[0], si.dp_dv[0])[0]<0):
                wi[:,1] *=-1

            flip = wi[...,-1:].sign()
            flip[flip==0] = 1
            wi *= flip
            wo *= flip

            btf = neumip(si.uv.torch()*UV_SCALE,wi,wo)['brdf']
            btf[wo[...,:2].pow(2).sum(-1)>THRESHOLD] = 0
            btf *= wo[...,2:].relu()

            btf_mi = mi.Vector3f(btf[...,0],btf[...,1],btf[...,2])

            del wo,wi,flip,btf
            gc.collect()

            torch.cuda.empty_cache()

            return btf_mi


        def pdf(self, ctx, si, wo,active):
            wi = si.wi.torch()
            if(dr.dot(si.sh_frame.s[0], si.dp_du[0])[0]<0):
                wi[:,0] *=-1
            if(dr.dot(si.sh_frame.t[0], si.dp_dv[0])[0]<0):
                wi[:,1] *=-1

            flip = wi[...,-1:].sign()
            flip[flip==0] = 1
            wi *=flip
            wo = wo.torch()*flip

            rgb_feature = neumip.get_rgb_texture(si.uv.torch()*UV_SCALE,wo=wi)
            cond_feature = torch.cat([rgb_feature,wi[...,:2]],-1)
            pdf = sampler_pdf(wo[...,:2],cond_feature)
            pdf *= wo[...,-1].relu()

            pdf_mi = dr.select(active,mi.Float(pdf),0.0)


            del wi,wo,flip,rgb_feature,cond_feature,pdf
            gc.collect()

            torch.cuda.empty_cache()

            return pdf_mi


        def eval_pdf(self, ctx, si, wo, active=True):
            f = self.eval(ctx,si,wo,active)
            pdf = self.pdf(ctx,si,wo,active)
            return f,pdf
        def to_string(self,):
            return 'NeuMIPBSDF'

    mi.register_bsdf("neumip", lambda props: NeuMIPBSDF(props))
    
    os.makedirs(os.path.join(IMAGE_PATH,MAT_NAME),exist_ok=True)
    scene = mi.load_file(SCENE_FILE)
    
    seed = 0
    for spp in SPP:
        rgb,seed = batched_render(scene,spp,seed)
        cv2.imwrite(
            os.path.join(IMAGE_PATH,MAT_NAME,'{}_{:04d}.exr'.format(SAMPLER,spp)),
                    rgb[:,:,[2,1,0]])
    
    for spp in SPP:
        rgb,seed = batched_render(scene,spp,seed)
        cv2.imwrite(
            os.path.join(IMAGE_PATH,MAT_NAME,'{}_{:04d}_1.exr'.format(SAMPLER,spp)),
                    rgb[:,:,[2,1,0]])
    
    # reference rgb
    if not os.path.exists(os.path.join(IMAGE_PATH,MAT_NAME,'reference.exr')):     
        rgb_reference,seed = batched_render(scene,SPP[-1],seed)
        cv2.imwrite(
            os.path.join(IMAGE_PATH,MAT_NAME,'reference.exr'),
            rgb_reference[:,:,[2,1,0]])
    
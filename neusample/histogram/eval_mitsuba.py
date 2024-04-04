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

def mitsuba_render(scene,spp,seed):
    sensor = scene.sensors()[0]
    integrator = scene.integrator()
    sampler,_ = prepare(sensor,seed,1)
    bsdf_ctx = mi.BSDFContext()
    W,H=mi.traverse(sensor)['film.size']
    
    
    L = mi.Spectrum(torch.zeros(H*W,3,device=device))
    torch.cuda.synchronize()
    start_time = time.time()
    for idx in tqdm(range(spp)):
        
        ray,pos = sample_rays(scene,sensor,sampler)
        pi = scene.ray_intersect_preliminary(ray)
        si = pi.compute_surface_interaction(ray)
        active=mi.Bool(np.ones((H*W),dtype=bool))


        Le = si.emitter(scene).eval(si)
        L += Le

        active_next = si.is_valid()

        bsdf = si.bsdf()
        
        
        bsdf_sample,bsdf_weight = bsdf.sample(
            bsdf_ctx,si,sampler.next_1d(active_next),sampler.next_2d(active_next),active_next)

        ray_bsdf = si.spawn_ray(si.to_world(bsdf_sample.wo))
        active_bsdf = active_next & dr.any(dr.neq(bsdf_weight,0.0))

        bsdf_weight = dr.select(dr.neq(bsdf_sample.pdf,0.0),bsdf_weight,0.0)
        
        si_bsdf = scene.ray_intersect(ray_bsdf,active_bsdf)
        L_bsdf = si_bsdf.emitter(scene).eval(si_bsdf,active_bsdf)
        L += L_bsdf*bsdf_weight
    torch.cuda.synchronize()
    render_time = (time.time()-start_time)/spp 
    L = L/spp
    L = L.numpy().reshape(H,W,3)
    return L, render_time


if __name__ == '__main__':
    dr.set_flag(dr.JitFlag.VCallRecord, False)
    parser = ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('--neumip_path', type=str, required=True)
    parser.add_argument('--sampler_path', type=str,default=None)
    parser.add_argument('--threshold', type=str,default='0.98')
    parser.add_argument('--sampler', type=str, default='diffuse')
    parser.add_argument('--image_path', type=str, default='outputs/teaser')
    parser.add_argument('--scene_file', type=str,default='notebooks/teaser1/teaser1.xml')
    
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
    
    
    SPP = [4,8,16,32,64,128,256,512]
    
    
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
        
    elif SAMPLER == 'normflow':
        state_dict = torch.load(SAMPLER_PATH,map_location='cpu')
        weight = {}
        for k,v in state_dict['net'].items():
            if 'flows.0.pw_qua_coupling.transform_net.net.' in k:
                weight[k.replace('flows.0.pw_qua_coupling.transform_net.net.',
                                 'mlps.0.')] = v
            elif 'flows.2.pw_qua_coupling.transform_net.net.' in k:
                weight[k.replace('flows.2.pw_qua_coupling.transform_net.net.',
                                 'mlps.1.')] = v

        D = 0
        for k in weight.keys():
            if 'mlps.0' in k:
                D += 1
        D = (D//2) - 1
        model = NIS(32,D=D)
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
    sensor = scene.sensors()[0]
    W,H=mi.traverse(sensor)['film.size']
    
    
    for spp in SPP:
        rgb,_ = mitsuba_render(scene,spp,0)
        cv2.imwrite(
            os.path.join(IMAGE_PATH,MAT_NAME,'{}_{:04d}.exr'.format(SAMPLER,spp)),
                    rgb.reshape(H,W,3)[:,:,[2,1,0]])
    
    render_times = []
    for spp in SPP:
        rgb,render_time =mitsuba_render(scene,spp,3)
        cv2.imwrite(
            os.path.join(IMAGE_PATH,MAT_NAME,'{}_{:04d}_1.exr'.format(SAMPLER,spp)),
                    rgb.reshape(H,W,3)[:,:,[2,1,0]])
        render_times.append(render_time)
    
    # reference rgb
    if not os.path.exists(os.path.join(IMAGE_PATH,MAT_NAME,'reference.exr')):     
        rgb_reference = mitsuba_render(scene,512,7)[0]
        cv2.imwrite(
            os.path.join(IMAGE_PATH,MAT_NAME,'reference.exr'),
            rgb_reference.reshape(H,W,3)[:,:,[2,1,0]])
    
    
    if os.path.exists(os.path.join(IMAGE_PATH,'timing.pth')):
        metric = torch.load(os.path.join(IMAGE_PATH,'timing.pth'))
    else:
        metric = {}
    
    if metric.get(MAT_NAME) is None: 
        metric[MAT_NAME] = {}
        
    metric[MAT_NAME][SAMPLER] = render_times
    
    torch.save(metric,os.path.join(IMAGE_PATH,'timing.pth'))
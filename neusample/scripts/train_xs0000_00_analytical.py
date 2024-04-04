
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
import torch.nn as nn

# Insert root dir in path
p = os.path.abspath('.')
sys.path.insert(1, p)

# from dataset_samplep import BSDFSamplingPDataset
from torch.utils.data.sampler import SubsetRandomSampler
# from ipypb import ipb
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from neumip_renderer import *
from neuis import  dataset_online_samplep
import normflows as nf
import math
from weighted_sum_dist import *
torch.autograd.set_detect_anomaly(True)
from condition_encoder import ConditionEncoder

"""
sampling from the GT distribution and train the normalizaing flow at the same time
"""
ROOT_DIR = "./"


exp_name = "xs0000_02_shell_analytical"

# target_ckpt_path = "../giallo_antico_marble.ckpt"
# target_ckpt_path = "../stylized_light_bulb_screw_base.ckpt"
# target_ckpt_path = "../elephant_leather_chainmail_emboss.ckpt"
# target_ckpt_path = "../scifi_center_quincunx_vent_border_layered_metal.ckpt" #xs0003
# target_ckpt_path = "../cowhide_leather_chainmail_emboss.ckpt" #"

target_ckpt_path = "../tortoise_shell.ckpt"

ENCODING = True 
NUM_LAYERS = 2
UNIFORM_DISTRIBUTION = True
RESOLUTION = 512
VERBOSE = True
from vis_helper import *

visHelper = vis_helper(exp_name, target_ckpt_path)

BASE = "uniform" ##"condLars"

def main():
    random_seed = 42

    BATCH_SIZE = 1
    SHUFFLE_DATASET = True
    dataset = dataset_online_samplep.NTOnlineSamplingPDataset(visHelper)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = 0
    if SHUFFLE_DATASET :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices = indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, sampler = train_sampler)

    model = GMMWeightedCond(n_modes=2, dim=2, cond_C=30, trainable=True)
    condEncoder = ConditionEncoder(ckpt_path=target_ckpt_path, encode_wo = True)
    
    if torch.cuda.device_count():
        model = model.cuda()
        condEncoder = condEncoder.cuda()

    ####################3
    RESUME = False
    TRAIN = True
    
    if not os.path.exists(os.path.join(ROOT_DIR, 'ckpts/')):
        os.makedirs(os.path.join(ROOT_DIR, 'ckpts/'))
    ckpt_path = os.path.join(ROOT_DIR, 'ckpts/',exp_name+"/")
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    best_ckpt_path = os.path.join(ROOT_DIR, 'ckpts/',exp_name, '%s_best.pth.tar'%(exp_name))
    newest_ckpt_path = os.path.join(ROOT_DIR, 'ckpts/',exp_name, '%s_%d.pth.tar'%(exp_name, 65)) 
    
    if(RESUME):
        checkpoint = torch.load(best_ckpt_path)
        model.load_state_dict(checkpoint['net'])


    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    num_steps = 500

    best_loss = 100
    writer = SummaryWriter(os.path.join(ROOT_DIR, "runs/neumip_%s"%(exp_name)))
    mdist = torch.distributions.MultivariateNormal(torch.zeros(2).cuda(), torch.eye(2).cuda()) ##Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    if TRAIN:
        render_step = 0
        batch = 0
        for idx_step in range(num_steps):
            if VERBOSE:
                it_dataset = tqdm(train_dataloader)
            for iter, sampleBatch in enumerate(it_dataset):
                optimizer.zero_grad()
                batch += 1
                sample = sampleBatch[0]

                uv = sample[:, :2]
                wo = sample[:, 2:4]
                x = sample[:, 4:6]

                cond_vec = condEncoder.compute_cond(wo, uv)
                ll= model.log_prob(x, cond_vec)
                loss = - ll.mean()

                # ll, loc, log_scale = model.log_prob(x, cond_vec)
                # # loc_nll = -(mdist.log_prob(loc[:,:2]), mdist.log_prob(loc[:,2:4])).mean()
                # # loc_loss = torch.square(loc).mean()
                # logscale_loss = torch.square(log_scale[:,0,:]).mean() + torch.square((log_scale[:,1,:].squeeze()- torch.Tensor([[-2.0,-2.0]]).cuda())).mean()
                # loss = logscale_loss
                # loss += - ll.mean()

                loss.backward()
                optimizer.step()
                state = {
                    'net': model.state_dict(),
                    'train_loss': loss,
                    'epoch': idx_step,
                }
                os.makedirs('ckpts', exist_ok=True)
                if loss < best_loss:
                    torch.save(state, best_ckpt_path)
                    best_loss = loss
                if (iter + 1) % 400 == 0:
                    print(f"idx_steps: {iter:}, loss: {loss.item():.5f}")
                    writer.add_scalar("Loss/train", loss.item(), batch)

                if (iter + 1) % 400 ==0 :
                    render_step += 1
                    print('Saving...')
                   
                    torch.save(state, os.path.join(ROOT_DIR, 'ckpts/',exp_name, '%s_%d.pth.tar'%(exp_name, render_step)) )
                   
                    with torch.no_grad():
                        # plot_results(renderer, model, base, wi_to_test.detach(), loc_to_test.detach(), render_step)
                        visHelper.plot_results(model, render_step,condEncoder)
                        # model.set_iteration_index(batch)
                        # model.dump_feature_texture(exp_name)
    writer.flush()   
    #for testing
    
    visHelper.plot_results_group(model, condEncoder, None)
    # plot_results(renderer, model, -1)



if __name__ == "__main__":
    main()
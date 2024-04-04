

from vis_helper import *
from condition_encoder import ConditionEncoder
from torch.utils.data import DataLoader
from weighted_sum_dist import *
import normflows as nf
from neuis import  dataset_online_samplep
from neumip_renderer import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import distributions
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import torch.nn.functional as F
import argparse
from sys import exit
import torch.optim as optim
import argparse
import os
import sys

from nsf_light_bk import *

from xml.etree.ElementTree import VERSION
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
# Insert root dir in path
p = os.path.abspath('.')
sys.path.insert(1, p)
# from dataset_samplep import BSDFSamplingPDataset
# from ipypb import ipb
torch.autograd.set_detect_anomaly(True)
from aaa_list_materials import *

ROOT_DIR = "./"


idx = "xs0027"
material = get_material_by_idx()[idx]
exp_name = idx+"_"+material + "_04_nsf_prior"



ENCODING = True
NUM_LAYERS = 2
UNIFORM_DISTRIBUTION = True
RESOLUTION = 512
VERBOSE = True

target_ckpt_path = get_neumip_model_dict()[material]

visHelper = vis_helper(exp_name, target_ckpt_path)


def main():
    random_seed = 42

    BATCH_SIZE = 1
    SHUFFLE_DATASET = True
    dataset = dataset_online_samplep.NTOnlineSamplingPDataset(visHelper)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = 0
    if SHUFFLE_DATASET:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices = indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=train_sampler)

    
    # base = nf.distributions.base.MultiUniform(
    #     2, torch.Tensor([0.0], device=DEVICE_), torch.Tensor([1.0], device=DEVICE_))
    # base = nf.distributions.DiagGaussian(2, trainable=False)
    base = ConditionalDiaGaussian(DISTRIBUTION_DIM, feat_dim = 30)
    # model = NSFLight_CL(base, feat_dim=30, num_layers=2,num_bins=32)
    model = NSFLight_CL(base, feat_dim=30, num_layers=2,num_bins=20, mlp_size=2)
    
    condEncoder = ConditionEncoder(ckpt_path=target_ckpt_path, encode_wo=True)
    
    if torch.cuda.device_count():
        model = model.cuda()
        base = base.cuda()
        condEncoder = condEncoder.cuda()  # TODO train the parameters??

    
    # 3
    RESUME = False
    TRAIN = True

    if not os.path.exists(os.path.join(ROOT_DIR, 'ckpts/')):
        os.makedirs(os.path.join(ROOT_DIR, 'ckpts/'))
    ckpt_path = os.path.join(ROOT_DIR, 'ckpts/', exp_name+"/")
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    best_ckpt_path = os.path.join(
        ROOT_DIR, 'ckpts/', exp_name, '%s_best.pth.tar' % (exp_name))
    newest_ckpt_path = os.path.join(
        ROOT_DIR, 'ckpts/', exp_name, '%s_%d.pth.tar' % (exp_name, 220))
    
    if (RESUME):
        checkpoint = torch.load(best_ckpt_path)
        model.load_state_dict(checkpoint['net'])

    optimizer = torch.optim.Adam(model.parameters(), 5e-4, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(list(model.parameters())+list(base.parameters()), lr=1e-3, weight_decay=1e-5)

    num_steps = 500

    best_loss = 100
    writer = SummaryWriter(os.path.join(
        ROOT_DIR, "runs/neumip_%s" % (exp_name)))

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
                ll = model.log_prob(x, cond_vec)
                loss = -torch.mean(ll)
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

                if (iter + 1) % 400 == 0:
                    render_step += 1
                    print('Saving...')
                    
                    torch.save(state, os.path.join(
                        ROOT_DIR, 'ckpts/', exp_name, '%s_%d.pth.tar' % (exp_name, render_step)))
                    if loss < best_loss:
                        torch.save(state, best_ckpt_path)
                        best_loss = loss

                    with torch.no_grad():
                        # plot_results(renderer, model, base, wi_to_test.detach(), loc_to_test.detach(), render_step)
                        visHelper.plot_results(
                            model, render_step, condEncoder, base=base)
                        # model.set_iteration_index(batch)
                        # model.dump_feature_texture(exp_name)
    writer.flush()
    # for testing

    visHelper.plot_results_group(model, condEncoder, base=base)
    # plot_results(renderer, model, -1)


if __name__ == "__main__":
    main()

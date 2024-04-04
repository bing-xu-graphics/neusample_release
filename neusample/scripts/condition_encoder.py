
import argparse, os, sys

from xml.etree.ElementTree import VERSION
import numpy as np
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
import torch.nn as nn
from utils import exr
from normflows.utils import encoding
# from normflows.utils import encoding_more

# Insert root dir in path
p = os.path.abspath('.')
sys.path.insert(1, p)
from neumat import neural_textures as nt


OFFSET_NW_INPUT_CHANNELS = 2 # UV
RGB_NW_INPUT_CHANNELS = 2 # only cam direction
OFFSET_NW_OUT_CHANNELS = 1
MLP_HIDDEN_LAYER_NEURONS = 32
POSITIONAL_ENCODING_BASIS_NUM = 5

class ThreeLayerMLP(nn.Module):
    """Classic MLP with three hidden layers"""
    def __init__(self, num_inputs, num_outputs) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, MLP_HIDDEN_LAYER_NEURONS),
            nn.LeakyReLU(),
            nn.Linear(MLP_HIDDEN_LAYER_NEURONS, MLP_HIDDEN_LAYER_NEURONS),
            nn.LeakyReLU(),
            nn.Linear(MLP_HIDDEN_LAYER_NEURONS, MLP_HIDDEN_LAYER_NEURONS),
            nn.LeakyReLU(),
            nn.Linear(MLP_HIDDEN_LAYER_NEURONS, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)



import time
class ConditionEncoder(nn.Module):
    """
    to encode (wo, uv, ..) condition
    """

    def __init__(self, resolution: int = [512,512],
                use_offset: bool = True,
                rgb_texture_num_channels: int = 8,
                offset_texture_num_channels: int = 8,
                initial_sigma: float = 8.0,
                iterations_to_sigma_1: int = 100, 
                use_pretrain_texture: bool = True, 
                ckpt_path ="../tortoise_shell.ckpt",
                encode_wo: bool = True,
                finetune: bool = False):

        super().__init__()
        self.encode_wo = encode_wo
        self.iteration_idx = 0
        self.use_offset = use_offset
        self.resolution = resolution

        self.initial_sigma = initial_sigma
        self.sigma_exp_decay = iterations_to_sigma_1 / np.log(initial_sigma)

        if use_offset:
            offset_texture_min_sigma = 2.0
            self.offset_texture = nt.NeuralTextureSingle(offset_texture_num_channels, resolution, offset_texture_min_sigma, offset_texture=True)
            offset_network_total_channels = offset_texture_num_channels + OFFSET_NW_INPUT_CHANNELS
            self.offset_network = ThreeLayerMLP(offset_network_total_channels, OFFSET_NW_OUT_CHANNELS)

        rgb_texture_min_sigma = 0.5
        self.rgb_texture = nt.NeuralTextureSingle(rgb_texture_num_channels, resolution, rgb_texture_min_sigma)
        self.rgb_network_total_channels = rgb_texture_num_channels + RGB_NW_INPUT_CHANNELS
        
        if use_pretrain_texture:
            #load in the weights for offset_network and rgb_textures and freeze those
            
            self.rgb_texture.requires_grad_(finetune)
            
            checkpoint = torch.load(ckpt_path)
            state_dict = checkpoint["state_dict"]
            # print(state_dict.keys())
            if use_offset:
                self.offset_network.requires_grad_(False)
                self.offset_texture.requires_grad_(False)
                #reformulating the state_dict
                offset_texture_state_dict = dict()
                offset_texture_state_dict["texture"] = state_dict["model.offset_texture.texture"]
                offset_network_state_dict = dict()

                offset_network_state_dict["layers.0.weight"] = state_dict["model.offset_network.layers.0.weight"]
                offset_network_state_dict["layers.0.bias"] = state_dict["model.offset_network.layers.0.bias"]
                offset_network_state_dict["layers.2.weight"] = state_dict["model.offset_network.layers.2.weight"]
                offset_network_state_dict["layers.2.bias"] = state_dict["model.offset_network.layers.2.bias"]
                offset_network_state_dict["layers.4.weight"] = state_dict["model.offset_network.layers.4.weight"]
                offset_network_state_dict["layers.4.bias"] = state_dict["model.offset_network.layers.4.bias"]
                offset_network_state_dict["layers.6.weight"] = state_dict["model.offset_network.layers.6.weight"]
                offset_network_state_dict["layers.6.bias"] = state_dict["model.offset_network.layers.6.bias"]
                
                self.offset_texture.load_state_dict(offset_texture_state_dict,strict=True)
                self.offset_network.load_state_dict(offset_network_state_dict,strict=True)
                self.offset_network.eval()
                self.offset_texture.eval()

            rgb_texture_state_dict = dict()
            rgb_texture_state_dict["texture"] = state_dict["model.rgb_texture.texture"]
            self.rgb_texture.load_state_dict(rgb_texture_state_dict,strict=True)
            self.rgb_texture.eval()
           
            # self.offset_network.freeze()
            # self.offset_texture.freeze()
            # self.rgb_texture.freeze()

    def dump_feature_texture(self, exp_name):
        output_dir = "./output/train_test/%s"%(exp_name)

        rgb_tex = self.rgb_texture.texture[3:6, :, :].cpu()
        rgb_tex = rgb_tex.permute(1, 2, 0)
        exr.write32(np.array(rgb_tex, dtype=np.float32), os.path.join(output_dir, "rgb_texture_%d.exr"%(self.iteration_idx)))
    
    def set_iteration_index(self, idx):
        self.iteration_idx = idx    

    def _initialize_weights(self):
        def initGaussian(m):
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight, std=0.01)

                if m.bias != None:
                    torch.nn.init.zeros_(m.bias)

        self.rgb_network.apply(initGaussian)

    def compute_sigma(self, iteration_idx):
        sigma = self.initial_sigma * np.exp(-iteration_idx / self.sigma_exp_decay)
        sigma = np.clip(sigma, 0.1, float("inf"))
        return sigma


    def compute_cond(self, wo, uv = None):
        ## convert latent space variables into observed variables
        assert self.iteration_idx >= 0
        
        self.rgb_texture.iteration = self.iteration_idx
        if(uv!=None):
            
            sigma = self.compute_sigma(self.iteration_idx)
            offset_intersection = uv.unsqueeze(1).unsqueeze(1) # Add dims for height, width
            if self.use_offset:
                camera_dir = wo
                self.offset_texture.iteration = self.iteration_idx
                neural_offset_latent_vector = self.offset_texture(offset_intersection, sigma)
                neural_offset = self.offset_network(torch.cat([neural_offset_latent_vector, camera_dir], dim=-1))
                neural_offset = nt.calculate_neural_offset(camera_dir, neural_offset)

                neural_offset = neural_offset.unsqueeze(1).unsqueeze(1)
                offset_intersection = offset_intersection + neural_offset
            
            rgb_latent_vector = self.rgb_texture(offset_intersection, sigma)
            if self.encode_wo:
                # encoded_wo = encoding_more.sh_encoding(wo)
                encoded_wo = encoding.positional_encoding_1(wo, POSITIONAL_ENCODING_BASIS_NUM)
                conditional_vector = torch.cat([encoded_wo, rgb_latent_vector ], dim=1)
            else:
                conditional_vector = torch.cat([wo, rgb_latent_vector ], dim=1)
        else:
            if self.encode_wo:
                conditional_vector = encoding.positional_encoding_1(wo, POSITIONAL_ENCODING_BASIS_NUM)
            else:
                conditional_vector = wo
        
        return conditional_vector
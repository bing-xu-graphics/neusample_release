"""Various Neural Material models"""
import torch
import torch.nn as nn
import numpy as np

from . import neural_textures as nt
from utils import exr

OFFSET_NW_INPUT_CHANNELS = 2 # UV
RGB_NW_INPUT_CHANNELS = 4 # cam + light direction
RGB_NW_OUT_CHANNELS = 3
OFFSET_NW_OUT_CHANNELS = 1
MLP_HIDDEN_LAYER_NEURONS = 32

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


class NeuMIPv1SingleRes(nn.Module):
    """
    Model for NeuMIP: Multi-Resolution Neural Materials
    https://cseweb.ucsd.edu/~viscomp/projects/NeuMIP/

    Note: No MIP support in this model
    """

    def __init__(self,
                 resolution,
                 use_offset: bool = True,
                 rgb_texture_num_channels: int = 8,
                 offset_texture_num_channels: int = 8,
                 initial_sigma: float = 8.0,
                 iterations_to_sigma_1: int = 3333):
        super().__init__()

        self.iteration_idx = 0
        self.use_offset = use_offset
        self.resolution = resolution

        self.initial_sigma = initial_sigma
        self.sigma_exp_decay = iterations_to_sigma_1 / np.log(initial_sigma)

        # TODO: Add controls for min_sigma

        if use_offset:
            print("there's offset")
            offset_texture_min_sigma = 2.0
            self.offset_texture = nt.NeuralTextureSingle(offset_texture_num_channels, resolution, offset_texture_min_sigma, offset_texture=True)
            offset_network_total_channels = offset_texture_num_channels + OFFSET_NW_INPUT_CHANNELS
            self.offset_network = ThreeLayerMLP(offset_network_total_channels, OFFSET_NW_OUT_CHANNELS)

        rgb_texture_min_sigma = 0.5
        self.rgb_texture = nt.NeuralTextureSingle(rgb_texture_num_channels, resolution, rgb_texture_min_sigma)
        self.rgb_network_total_channels = rgb_texture_num_channels + RGB_NW_INPUT_CHANNELS
        self.rgb_network = ThreeLayerMLP(self.rgb_network_total_channels, RGB_NW_OUT_CHANNELS)

        self._initialize_weights()

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

    def forward(self, camera_dir, light_dir, uv):
        assert self.iteration_idx >= 0

        self.rgb_texture.iteration = self.iteration_idx
        sigma = self.compute_sigma(self.iteration_idx)

        uv = uv.unsqueeze(1).unsqueeze(1) # Add dims for height, width
        offset_intersection = uv
        # default_offset_sigma = 2.0
        # default_rgb_sigma = 0.5
        if self.use_offset:
            # print("using offset netowork........................ sigma = ", sigma)
            self.offset_texture.iteration = self.iteration_idx
            neural_offset_latent_vector = self.offset_texture(offset_intersection, sigma)
            neural_offset = self.offset_network(torch.cat([neural_offset_latent_vector, camera_dir], dim=-1))
            neural_offset = nt.calculate_neural_offset(camera_dir, neural_offset)

            # if True and not self.training:
            #     with torch.no_grad():
            #         noff_out = neural_offset.detach()
            #         noff_out = noff_out.reshape(self.resolution[0], self.resolution[1], -1)
            #         noff_out = np.dstack([noff_out.cpu(), np.zeros(self.resolution)])
            #         exr.write16(np.array(noff_out, dtype=np.float32), "offset_value.exr")
            neural_offset = neural_offset.unsqueeze(1).unsqueeze(1)
            offset_intersection = offset_intersection + neural_offset
        else:
            print("turning offset netowork off....................")
        rgb_latent_vector = self.rgb_texture(offset_intersection, sigma) #TODO Check

        rgb_network_inputs = torch.cat([light_dir, camera_dir], dim=1)
        rgb_network_inputs = [rgb_network_inputs, rgb_latent_vector]
        rgb_network_inputs = torch.cat(rgb_network_inputs, dim=1)

        rgb_output = self.rgb_network(rgb_network_inputs)

        return rgb_output

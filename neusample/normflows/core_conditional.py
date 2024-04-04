import torch
import torch.nn as nn
import numpy as np

from . import distributions
from . import utils
from utils import exr, ops, la2
import os
# import larsflow as lf
from .utils import encoding
from .utils import prior
from .nets import mlp
# import nets
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
INV_PI = 0.31830988618
class NTCondNormalizingFlow(nn.Module):
    """
    Normalizing Flow model to approximate target distribution
    """

    def __init__(self, q0, flows, p=None, 
                resolution: int = [512,512],
                use_offset: bool = True,
                rgb_texture_num_channels: int = 8,
                offset_texture_num_channels: int = 8,
                initial_sigma: float = 8.0,
                iterations_to_sigma_1: int = 100, 
                use_pretrain_texture: bool = True, 
                ckpt_path ="../tortoise_shell.ckpt",
                # ckpt_path ="../stylized_wool.ckpt",
                encode_wo: bool = False,
                base = "condLars"):

        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p
        self.encode_wo = encode_wo
        #neural material related variables
        self.iteration_idx = 0
        self.use_offset = use_offset
        self.resolution = resolution

        self.initial_sigma = initial_sigma
        self.sigma_exp_decay = iterations_to_sigma_1 / np.log(initial_sigma)
        self.base = base 

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
            self.offset_network.requires_grad_(False)
            self.offset_texture.requires_grad_(False)
            self.rgb_texture.requires_grad_(False)
            
            checkpoint = torch.load(ckpt_path)
            state_dict = checkpoint["state_dict"]
            # print(state_dict.keys())

            #reformulating the state_dict
            offset_texture_state_dict = dict()
            offset_texture_state_dict["texture"] = state_dict["model.offset_texture.texture"]
            rgb_texture_state_dict = dict()
            rgb_texture_state_dict["texture"] = state_dict["model.rgb_texture.texture"]
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
            self.rgb_texture.load_state_dict(rgb_texture_state_dict,strict=True)
            self.offset_network.load_state_dict(offset_network_state_dict,strict=True)
            self.offset_network.eval()
            self.offset_texture.eval()
            self.rgb_texture.eval()
           
            # self.offset_network.freeze()
            # self.offset_texture.freeze()
            # self.rgb_texture.freeze()

    def dump_feature_texture(self, exp_name):
        output_dir = "./output/train_test/%s"%(exp_name)

        rgb_tex = self.rgb_texture.texture[3:6, :, :].cpu()
        rgb_tex = rgb_tex.permute(1, 2, 0)
        exr.write32(np.array(rgb_tex, dtype=np.float32), os.path.join(output_dir, "rgb_texture_%d.exr"%(self.iteration_idx)))
    def preload_neumat(self, ckpt_path):
        pass
    
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

    def compute_cond(self, wi, uv = None):
        ## convert latent space variables into observed variables
        assert self.iteration_idx >= 0
        
        self.rgb_texture.iteration = self.iteration_idx
        if(uv!=None):
            sigma = self.compute_sigma(self.iteration_idx)
            offset_intersection = uv.unsqueeze(1).unsqueeze(1) # Add dims for height, width
            if self.use_offset:
                camera_dir = wi
                self.offset_texture.iteration = self.iteration_idx
                neural_offset_latent_vector = self.offset_texture(offset_intersection, sigma)
                neural_offset = self.offset_network(torch.cat([neural_offset_latent_vector, camera_dir], dim=-1))
                neural_offset = nt.calculate_neural_offset(camera_dir, neural_offset)

                neural_offset = neural_offset.unsqueeze(1).unsqueeze(1)
                offset_intersection = offset_intersection + neural_offset
            
            rgb_latent_vector = self.rgb_texture(offset_intersection, sigma)
            # rgb_latent_vector = encoding.positional_encoding_1(uv, POSITIONAL_ENCODING_BASIS_NUM)
            if self.encode_wo:
                encoded_wi = encoding.positional_encoding_1(wi, POSITIONAL_ENCODING_BASIS_NUM)
                # print("encoded_wi.shape ", encoded_wi.shape)
                conditional_vector = torch.cat([encoded_wi, rgb_latent_vector ], dim=1)
                # print("conditional_vector.shape ", conditional_vector.shape)
            else:
                conditional_vector = torch.cat([wi, rgb_latent_vector ], dim=1)
        else:
            if self.encode_wo:
                conditional_vector = encoding.positional_encoding_1(wi, POSITIONAL_ENCODING_BASIS_NUM)
            else:
                conditional_vector = wi
        
        return conditional_vector

    def forward_kld(self, x, wo, uv):
        x = (x + 1.0 )*0.5 ##TODO remove
        conditional_vector = self.compute_cond(wo, uv)

        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, conditional_vector)
            log_q += log_det
        if self.base == "condLars_uni":
            log_q += self.q0.log_prob(z, conditional_vector)
        else:
            log_q += self.q0.log_prob(z)
        return -torch.mean(log_q)

    def sample(self, wo, uv,num_samples=1):
        """
        Samples from flow-based approximate distribution
        :param num_samples: Number of samples to draw
        :return: Samples, log probability
        """
        conditional_vector = self.compute_cond(wo, uv)
        if self.base == "condLars_uni":
            z, log_q = self.q0(conditional_vector, num_samples)
        else:
            z, log_q = self.q0(None, num_samples)
        # log_q = torch.zeros(log_q.shape).cuda()
        # log_q = torch.log(torch.full(log_q.shape, INV_PI).cuda())
        for flow in self.flows:
            z, log_det = flow(z, conditional_vector)
            log_q -= log_det
        z = z * 2.0 - 1.0 ##TODO remove
        return z, log_q

    def sample(self,conditional_vector,num_samples=1):
        """
        Samples from flow-based approximate distribution
        :param num_samples: Number of samples to draw
        :return: Samples, log probability
        """
        if self.base == "condLars_uni":
            z, log_q = self.q0(conditional_vector, num_samples)
        else:
            z, log_q = self.q0(None, num_samples)
        # log_q = torch.zeros(log_q.shape).cuda()
        # log_q = torch.log(torch.full(log_q.shape, INV_PI).cuda())
        for flow in self.flows:
            z, log_det = flow(z, conditional_vector)
            log_q -= log_det
        z = z * 2.0 - 1.0 ##TODO remove
        return z, log_q

    def log_prob(self, x, wo, uv):
        """
        Get log probability for batch
        :param x: Batch
        :return: log probability
        """
        x = (x + 1.0 )*0.5 ##TODO remove
        # uv = None
        conditional_vector = self.compute_cond(wo, uv)

        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z,conditional_vector)
            log_q += log_det
        # log_q += self.q0.log_prob(z)
        if self.base == "condLars_uni":
            log_q += self.q0.log_prob(z, conditional_vector)
        else:
            log_q += self.q0.log_prob(z)
        return log_q

    def save(self, path):
        """
        Save state dict of model
        :param path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load model from state dict
        :param path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))


class NTCondNormalizingFlow_prior(nn.Module):
    """
    Normalizing Flow model to approximate target distribution
    """

    def __init__(self, flows, p=None, 
                resolution: int = [512,512],
                use_offset: bool = True,
                rgb_texture_num_channels: int = 8,
                offset_texture_num_channels: int = 8,
                initial_sigma: float = 8.0,
                iterations_to_sigma_1: int = 100, 
                use_pretrain_texture: bool = True, 
                ckpt_path ="../tortoise_shell.ckpt",
                encode_wo: bool = False,
                input_dim: int = 2,
                cond_channels: int = 0,
                base = "condGauss"):

        super().__init__()
        self.base = base
        ## the trainable prior
        if base == "condGauss":
            self.q0 = prior.GaussianPrior(input_dim, cond_channels)
        elif base == "condLars":
            # a = mlp.CondMLP([input_dim + cond_channels, 256, 256, 1], output_fn="sigmoid")
            a = mlp.CondMLP([input_dim + cond_channels, 64, 64, 64, 1], output_fn="sigmoid")
            # self.q0 = lf.distributions.CondResampledGaussian(input_dim, a, 100, 0.1, trainable=False)
            self.q0 = prior.GaussianPrior(input_dim, cond_channels)
        else:
            print("Unsupported base type!! set to condGauss")
            self.base = "condGauss"
            self.q0 = prior.GaussianPrior(input_dim, cond_channels)

        self.conditional_encoder = mlp.MLP([cond_channels, 64, 64, cond_channels], output_fn="tanh") #todo

        self.flows = nn.ModuleList(flows)
        self.p = p
        self.encode_wo = encode_wo
        #neural material related variables
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
            self.offset_network.requires_grad_(False)
            self.offset_texture.requires_grad_(False)
            self.rgb_texture.requires_grad_(False)
            
            checkpoint = torch.load(ckpt_path)
            state_dict = checkpoint["state_dict"]
            # print(state_dict.keys())

            #reformulating the state_dict
            offset_texture_state_dict = dict()
            offset_texture_state_dict["texture"] = state_dict["model.offset_texture.texture"]
            rgb_texture_state_dict = dict()
            rgb_texture_state_dict["texture"] = state_dict["model.rgb_texture.texture"]
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
            self.rgb_texture.load_state_dict(rgb_texture_state_dict,strict=True)
            self.offset_network.load_state_dict(offset_network_state_dict,strict=True)
            self.offset_network.eval()
            self.offset_texture.eval()
            self.rgb_texture.eval()
           
            # self.offset_network.freeze()
            # self.offset_texture.freeze()
            # self.rgb_texture.freeze()

    def dump_feature_texture(self, exp_name):
        output_dir = "/home/bingxu/projects/neumip_adobe/blender-neumip-mulliala-deploy/output/train_test/%s"%(exp_name)

        rgb_tex = self.rgb_texture.texture[3:6, :, :].cpu()
        rgb_tex = rgb_tex.permute(1, 2, 0)
        exr.write32(np.array(rgb_tex, dtype=np.float32), os.path.join(output_dir, "rgb_texture_%d.exr"%(self.iteration_idx)))
    def preload_neumat(self, ckpt_path):
        pass
    
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
        # if(uv):
        #     sigma = self.compute_sigma(self.iteration_idx)
        #     uv = uv.unsqueeze(1).unsqueeze(1) # Add dims for height, width
        #     offset_intersection = uv
        #     if self.use_offset:
        #         camera_dir = wi
        #         self.offset_texture.iteration = self.iteration_idx
        #         neural_offset_latent_vector = self.offset_texture(offset_intersection, sigma)
        #         neural_offset = self.offset_network(torch.cat([neural_offset_latent_vector, camera_dir], dim=-1))
        #         neural_offset = nt.calculate_neural_offset(camera_dir, neural_offset)

        #         neural_offset = neural_offset.unsqueeze(1).unsqueeze(1)
        #         offset_intersection = offset_intersection + neural_offset
            
        #     rgb_latent_vector = self.rgb_texture(offset_intersection, sigma)
        #     if ENCODING:
        #         encoded_wi = positional_encoding(wi, POSITIONAL_ENCODING_BASIS_NUM)
        #         # print("encoded_wi.shape ", encoded_wi.shape)
        #         conditional_vector = torch.cat([encoded_wi, rgb_latent_vector ], dim=1)
        #         # print("conditional_vector.shape ", conditional_vector.shape)
        #     else:
        #         conditional_vector = torch.cat([wi, rgb_latent_vector ], dim=1)
        # else:
        if self.encode_wo:
            conditional_vector = encoding.positional_encoding_1(wo, POSITIONAL_ENCODING_BASIS_NUM)
        else:
            conditional_vector = wo
        
        return conditional_vector, self.conditional_encoder(conditional_vector)#TODO remove 

    def forward_kld(self, x, wi, uv):
        conditional_vector, conditional_vector_mlp = self.compute_cond(wi, uv)

        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, conditional_vector_mlp)
            log_q += log_det

        if(self.base == "condGauss"):
            _,_,logp= self.q0.forward(z, conditional_vector,eps = 0, reverse = False)

        elif(self.base == "condLars"):
            logp = self.q0.log_prob(z, conditional_vector)
            
        
        log_q += logp
        return -torch.mean(log_q), -torch.mean(logp)


    def reverse_kld(self,  wo, uv, num_samples=1, beta=1.0, score_fn=True):
        
        conditional_vector = self.compute_cond(wo, None) ### TODO change uv here

        z, log_q_ = self.q0.forward(conditional_vector, num_samples)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z, conditional_vector)
            log_q -= log_det
        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_, conditional_vector)
                log_q += log_det
            log_q += self.q0.log_prob(z_, conditional_vector)
            utils.set_requires_grad(self, True)
        log_p = self.p.log_prob(z, wo, uv)
        return torch.mean(log_q) - beta * torch.mean(log_p)
  
    def sample(self, wi, uv, num_samples=1):
        """
        Samples from flow-based approximate distribution
        :param num_samples: Number of samples to draw
        :return: Samples, log probability
        """
        conditional_vector, conditional_vector_mlp = self.compute_cond(wi, uv)
        # z, log_q = self.q0(num_samples)
        if(self.base == "condGauss"):
            z, _, log_q = self.q0.forward(None, conditional_vector,eps = 0, reverse = True)
        elif (self.base == "condLars"):
            z, log_q = self.q0(conditional_vector, num_samples)
        
        for flow in self.flows:
            z, log_det = flow(z, conditional_vector_mlp)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x, wi, uv):
        """
        Get log probability for batch
        :param x: Batch
        :return: log probability
        """
        conditional_vector, conditional_vector_mlp = self.compute_cond(wi, uv)

        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z,conditional_vector_mlp)
            log_q += log_det
        # log_q += self.q0.log_prob(z)
        if(self.base == "condGauss"):
            _,_,logp= self.q0.forward(z, conditional_vector,eps = 0, reverse = False)

        elif(self.base == "condLars"):
            logp = self.q0.log_prob(z, conditional_vector)
        
        log_q += logp
        return log_q

    def save(self, path):
        """
        Save state dict of model
        :param path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load model from state dict
        :param path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))

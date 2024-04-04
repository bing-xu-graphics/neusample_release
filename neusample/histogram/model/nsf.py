import torch
import torch.nn as nn
import torch.nn.functional as NF

import numpy as np
import math

import sys
sys.path.append('..')
from model.mlps import mlp

import time

def leaky_mlp(dim_in, dims, dim_out):
    """ create an MLP in format: dim_in->dims[0]->...->dims[-1]->dim_out"""
    lists = []
    dims = [dim_in] + dims
    
    for i in range(len(dims)-1):
        lists.append(nn.Linear(dims[i],dims[i+1]))
        lists.append(nn.LeakyReLU(0.01))
    lists.append(nn.Linear(dims[-1], dim_out))
    return nn.Sequential(*lists)

def positional_encoding_1(
    tensor, num_encoding_functions=6, log_sampling=True
):
    encoding = [tensor]
    
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
                0.0,
                num_encoding_functions - 1,
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        
    for freq in frequency_bands:
        encoding.append(torch.sin(tensor * freq))
        encoding.append(torch.cos(tensor * freq))

    return torch.cat(encoding, dim=-1)
    
class NSF_prior(nn.Module):
    def __init__(self,K=20,D=2):
        super(NSF_prior,self).__init__()
        self.K = K
        self.mlp0 = leaky_mlp(30,[32],4)
        self.mlps = nn.ModuleList([
            leaky_mlp(39,[32]*D,K*3-1),
            leaky_mlp(39,[32]*D,K*3-1)
        ])

    def q0(self,wi,cond,inverse):
        ret=  self.mlp0(cond)
        mu = ret[...,:2]
        log_sigma = ret[...,2:]
        sigma = torch.exp(log_sigma)
        
        if inverse:
            u1,u2 = wi[...,0],wi[...,1]
            z0 = (-2*torch.log(u1.clamp_min(1e-12))).sqrt()
            z1 = torch.sin(2*math.pi*u2)*z0
            z0 = torch.cos(2*math.pi*u2)*z0
            z = torch.stack([z0,z1],-1)
            z = mu + z*sigma
            wi = z
        
        rr = (wi-mu)/sigma
        rr = (rr*rr).sum(1)
        
        log_pdf = -log_sigma.sum(-1)-math.log(2*math.pi)-0.5*rr
        
        return wi, log_pdf.unsqueeze(-1)
        
    def forward(self,wi,cond,inverse=False):
        id1,id2 = 1,0
        cond =  torch.cat([
                positional_encoding_1(cond[...,-2:],5,log_sampling=True),cond[...,:-2]],-1)
        if not inverse:
            
            xin = torch.cat([
                positional_encoding_1(wi[...,id1].unsqueeze(-1),4,log_sampling=False),
                cond],-1)
            transform_params = self.mlps[1](xin)
            width = transform_params[...,:self.K].unsqueeze(-2)
            height = transform_params[...,self.K:self.K*2].unsqueeze(-2)
            derivative = transform_params[...,self.K*2:].unsqueeze(-2)
            
            
            wi_out,log_pdf = unconstrained_rational_quadratic_spline(wi[...,id2].unsqueeze(-1),width,height,derivative,inverse=inverse)
            wi[...,id2] = wi_out.squeeze(-1)
            
            xin = torch.cat([positional_encoding_1(wi[...,id2].unsqueeze(-1),4,log_sampling=False),cond],-1)
            transform_params = self.mlps[0](xin)
            width = transform_params[...,:self.K].unsqueeze(-2)
            height = transform_params[...,self.K:self.K*2].unsqueeze(-2)
            derivative = transform_params[...,self.K*2:].unsqueeze(-2)
        
            wi_out,log_pdf_ = unconstrained_rational_quadratic_spline(wi[...,id1].unsqueeze(-1),width,height,derivative,inverse=inverse)
            wi[...,id1] = wi_out.squeeze(-1)
            log_pdf += log_pdf_
            
            
            wi,log_pdf_ = self.q0(wi,cond,inverse=inverse)
            log_pdf += log_pdf_
            
            return wi,(log_pdf).exp().squeeze(-1)
        else:
            wi = wi[...,:2]
            
            wi,log_pdf = self.q0(wi,cond,inverse)
            
            xin = torch.cat([positional_encoding_1(wi[...,id2].unsqueeze(-1),4,log_sampling=False),cond],-1)
            transform_params = self.mlps[0](xin)
            width = transform_params[...,:self.K].unsqueeze(-2)
            height = transform_params[...,self.K:self.K*2].unsqueeze(-2)
            derivative = transform_params[...,self.K*2:].unsqueeze(-2)
            
            wi_out,log_pdf_ = unconstrained_rational_quadratic_spline(wi[...,id1].unsqueeze(-1),width,height,derivative,inverse=inverse)
            wi[...,id1] = wi_out.squeeze(-1)
            log_pdf -= log_pdf_
            
            xin = torch.cat([positional_encoding_1(wi[...,id1].unsqueeze(-1),4,log_sampling=False),cond],-1)
            transform_params = self.mlps[1](xin)
            width = transform_params[...,:self.K].unsqueeze(-2)
            height = transform_params[...,self.K:self.K*2].unsqueeze(-2)
            derivative = transform_params[...,self.K*2:].unsqueeze(-2)
            
            wi_out,log_pdf_ = unconstrained_rational_quadratic_spline(wi[...,id2].unsqueeze(-1),width,height,derivative,inverse=inverse)
            wi[...,id2] = wi_out.squeeze(-1)

            log_pdf -= log_pdf_

            return wi,log_pdf.exp().squeeze(-1)
        
        
        

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives_ = NF.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives_[..., 0] = constant
        unnormalized_derivatives_[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    elif tails == "circular":
        unnormalized_derivatives_ = NF.pad(unnormalized_derivatives, pad=(0, 1))
        unnormalized_derivatives_[..., -1] = unnormalized_derivatives_[..., 0]

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    elif isinstance(tails, list) or isinstance(tails, tuple):
        unnormalized_derivatives_ = unnormalized_derivatives.clone()
        ind_lin = [t == "linear" for t in tails]
        ind_circ = [t == "circular" for t in tails]
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives_[..., ind_lin, 0] = constant
        unnormalized_derivatives_[..., ind_lin, -1] = constant
        unnormalized_derivatives_[..., ind_circ, -1] = unnormalized_derivatives_[
            ..., ind_circ, 0
        ]
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if torch.is_tensor(tail_bound):
        tail_bound_ = torch.broadcast_to(tail_bound, inputs.shape)
        left = -tail_bound_[inside_interval_mask]
        right = tail_bound_[inside_interval_mask]
        bottom = -tail_bound_[inside_interval_mask]
        top = tail_bound_[inside_interval_mask]
    else:
        left = -tail_bound
        right = tail_bound
        bottom = -tail_bound
        top = tail_bound

    (
        outputs[inside_interval_mask],
        logabsdet[inside_interval_mask],
    ) = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives_[inside_interval_mask, :],
        inverse=inverse,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    num_bins = unnormalized_widths.shape[-1]

    if torch.is_tensor(left):
        lim_tensor = True
    else:
        lim_tensor = False

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = NF.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = NF.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    if lim_tensor:
        cumwidths = (right[..., None] - left[..., None]) * cumwidths + left[..., None]
    else:
        cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + NF.softplus(unnormalized_derivatives)

    heights = NF.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = NF.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    if lim_tensor:
        cumheights = (top[..., None] - bottom[..., None]) * cumheights + bottom[
            ..., None
        ]
    else:
        cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet

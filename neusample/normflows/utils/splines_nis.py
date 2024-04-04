import torch
from torch.nn import functional as F

import numpy as np

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

"""
pdf is piecewise constant; 
K == |B| == unnormalized_widths.shape[-1]
"""
def linear_spline( ###assure that unnormalized_heights is the same as unnomalized_derivatives
    inputs,
    unnormalized_widths,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    num_bins = unnormalized_widths.shape[-1]
    assert unnormalized_widths.shape[-1] == unnormalized_derivatives.shape[-1] #K 

    if torch.is_tensor(left):
        lim_tensor = True
    else:
        lim_tensor = False

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
   
    
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    
    if lim_tensor:
        cumwidths = (right[..., None] - left[..., None]) * cumwidths + left[..., None]
    else:
        cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)
    derivatives = F.softmax(derivatives, dim=-1) #normalize the constant slopes, TODO check

    cum_derivatives = torch.cumsum(derivatives, dim=-1)
    cum_derivatives = F.pad(cum_derivatives, pad=(1, 0), mode="constant", value=0.0)

    if inverse:
        bin_idx = searchsorted(cum_derivatives, inputs)[..., None] #TODO change to cum_derivatives? and leave heights alone
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]
    

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]    
    input_cum_derivatives = cum_derivatives.gather(-1, bin_idx)[..., 0]

    if inverse:
        w = 1/num_bins
        outputs = (inputs - input_cum_derivatives)/input_derivatives + input_derivatives * w #calculate x by innvert CDF 
        eps = torch.finfo(outputs.dtype).eps
        outputs = outputs.clamp(
            min=eps,
            max=1. - eps
        )

        logabsdet = torch.log(input_derivatives)
        return outputs, -logabsdet
    else:       
        alpha = (inputs - input_cumwidths) / input_bin_widths
        outputs = alpha * input_derivatives + input_cum_derivatives
        ## make sure the outputs are within range[0,1]
        eps = torch.finfo(outputs.dtype).eps
        outputs = outputs.clamp(min=eps, max=1-eps)
        
        logabsdet = torch.log(input_derivatives/input_bin_widths) #bin_widths are the same; == 1/num_bins
        return outputs, logabsdet


def quadratic_softmax(v,w):
    v=torch.exp(v)
    vnorms=torch.cumsum(torch.mul((v[:,:,:-1]+v[:,:,1:])/2,w),dim=-1)
    vnorms_tot=vnorms[:, :, -1].clone() 
    return torch.div(v,torch.unsqueeze(vnorms_tot,dim=-1)) 
import time
def quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    num_bins = unnormalized_widths.shape[-1]
    assert((unnormalized_widths.shape[-1] + 1) == unnormalized_derivatives.shape[-1]) #K+1 vertices for derivative values 

    if torch.is_tensor(left):
        lim_tensor = True
    else:
        lim_tensor = False

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    if lim_tensor:
        cumwidths = (right[..., None] - left[..., None]) * cumwidths + left[..., None]
    else:
        cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)
    derivatives = quadratic_softmax(derivatives, widths) #TODO check normalize the vertices

    cum_derivatives = torch.cumsum(derivatives, dim=-1)
    cum_derivatives = F.pad(cum_derivatives, pad=(1, 0), mode="constant", value=0.0)
    
    if inverse:
        # to find the bin ##TODO
        tmp = (derivatives[:,:,:-1] + derivatives[:,:,1:] ) * widths *0.5
        cum_tmp = torch.cumsum(tmp, dim=-1)
        cum_tmp = F.pad(cum_tmp, pad=(1, 0), mode="constant", value=0.0)  # [N, D, B] 
        # bin_idx = searchsorted(cum_tmp, inputs)[..., None] 
        finder=torch.where(cum_tmp>torch.unsqueeze(inputs,dim=-1),torch.zeros_like(cum_tmp),torch.ones_like(cum_tmp))        
        eps = torch.finfo(inputs.dtype).eps
        # to_cat = torch.cat((torch.ones([cum_tmp.shape[0],cum_tmp.shape[1],1]).to(cum_tmp.device, cum_tmp.dtype)*eps,finder*(cum_tmp+1)),axis=-1)
        to_cat = torch.cat((torch.ones([cum_tmp.shape[0],cum_tmp.shape[1],1], device= cum_tmp.device, dtype=cum_tmp.dtype)*eps,finder*(cum_tmp+1)),axis=-1)
        mx=torch.unsqueeze(
        torch.argmax(to_cat, dim=-1),dim=-1)-1 
        bin_idx = torch.clamp(mx, 0, num_bins - 1)#.to(torch.long)
        

    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]
    

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    if inverse:
        # spline_start = time.time()
        a =  (input_derivatives_plus_one - input_derivatives) * input_bin_widths *0.5
        b =  input_derivatives * input_bin_widths 
        c = cum_tmp.gather(-1, bin_idx)[..., 0] - inputs #[batch, |B|]  #constant_term - c
        
        ### ref from github repo to handle division by zero
        eps = torch.finfo(a.dtype).eps
        
        a=torch.where(torch.abs(a)<eps,eps*torch.ones_like(a),a)

        discriminant = b.pow(2) - 4 * a * c
        # assert (discriminant >= 0).all()
        root1 = (-b - torch.sqrt(discriminant))/ (2 * a)  
        root2 = (-b + torch.sqrt(discriminant))/ (2 * a) 
        root=torch.where((root1>=0)&(root1<1), root1, root2) #choose the valid root
        outputs = root * input_bin_widths + input_cumwidths
        
        eps = torch.finfo(outputs.dtype).eps
        outputs = outputs.clamp(min=eps, max=1-eps)

        inv_q = torch.lerp(input_derivatives, input_derivatives_plus_one, root)
        logabsdet = torch.log(inv_q)
        return outputs, -logabsdet
    else:
        alpha = (inputs - input_cumwidths) / input_bin_widths
        q = torch.lerp(input_derivatives, input_derivatives_plus_one, alpha)

        # tmp_ib = (v_ib + v_(i+1)b) * w_ib /2 , the constant term of the quadratic equation
        tmp = (derivatives[:,:,:-1] + derivatives[:,:,1:] ) * widths * 0.5
        cum_tmp = torch.cumsum(tmp, dim=-1)
        cum_tmp = F.pad(cum_tmp, pad=(1, 0), mode="constant", value=0.0)
        input_tmp = cum_tmp.gather(-1, bin_idx)[..., 0]   #[batch, |B|]

        outputs = alpha.pow(2)/2 * (input_derivatives_plus_one - input_derivatives) * input_bin_widths \
            + alpha * input_derivatives * input_bin_widths \
            + input_tmp #in [batch, |B|]

        eps = torch.finfo(outputs.dtype).eps
        outputs = outputs.clamp(min=eps, max=1-eps)
        logabsdet = torch.log(q) #multiplication along |B|, would be summed outside
        return outputs, logabsdet

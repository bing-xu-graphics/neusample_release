import torch
from torch import nn
import numpy as np
import os, sys

from torch.distributions.multivariate_normal import MultivariateNormal
# Insert root dir in path
p = os.path.abspath('.')
sys.path.insert(1, p)

class DiagGaussian(nn.Module):
    def __init__(self, shape, loc, scale, trainable=True):

        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        if trainable:
            if(not loc):
                self.loc = nn.Parameter(torch.tensor([[loc[0], loc[1]]]))
            else:
                self.loc = nn.Parameter(torch.zeros(1, *self.shape))
            
            if(not scale):
                self.log_scale = nn.Parameter(torch.tensor([[scale[0], scale[1]]]))
            else:
                self.log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        else:
            self.register_buffer("loc", torch.zeros(1, *self.shape))
            self.register_buffer("log_scale", torch.zeros(1, *self.shape))
    
    def forward(self, num_samples=1):
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device
        )
        log_scale = self.log_scale
        z = self.loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p

    def log_prob(self, z):
        log_scale = self.log_scale
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        # log_scale = self.log_scale
        # log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
        #     log_scale.unsqueeze(0) + 0.5 * torch.pow((z - self.loc.unsqueeze(0)) / torch.exp(log_scale.unsqueeze(0)), 2),
        #     list(range(1, self.n_dim + 1)),
        # )
        return log_p


class DiagGaussianCond(nn.Module):
    def __init__(self, shape, trainable=True):

        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)  #2
        self.temperature = None


    def forward(self, mean, sigma, num_samples=1):
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=mean.dtype, device=mean.device
        )
        if self.temperature is None:
            log_scale = sigma
        else:
            log_scale = sigma + np.log(self.temperature)
        z = mean + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p

    def prob(self, z, mean, sigma):
        if self.temperature is None:
            log_scale = sigma
        else:
            log_scale = sigma + np.log(self.temperature)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - mean) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return torch.exp(log_p)

from normflows import mlp

M_PI = 3.14159265359
PI_over_4 = 0.78539816339
PI_over_2 = 1.57079632679
INV_PI =    0.31830988618

class LambertianLobe(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, num_samples=1, randseed=None): #sample
        if randseed==None:
            wo = torch.rand(num_samples, 2,  device=DEVICE_, dtype = DTYPE_) * 2.0 - 1.0  #[-1,1]
        else:
            wo = randseed* 2.0 - 1.0
        woSample = torch.zeros(wo.shape,  device=DEVICE_, dtype = DTYPE_)
        zero_positions = torch.logical_and(wo[:, 0] == 0 , wo[:, 1]==0) 
        nonzero_positions = ~zero_positions
        condition1 = torch.logical_and(torch.abs(wo[:,0]) > torch.abs(wo[:,1]),nonzero_positions)
        condition2 = torch.logical_and(~condition1 ,nonzero_positions)

        woSample[condition1,0] = wo[condition1,0] * torch.cos(PI_over_4 * wo[condition1,1]/wo[condition1,0])
        woSample[condition1,1] = wo[condition1,0] * torch.sin(PI_over_4 * wo[condition1,1]/wo[condition1,0])
        woSample[condition2,0] = wo[condition2,1] * torch.cos(PI_over_2 - PI_over_4 * wo[condition2,0]/wo[condition2,1])
        woSample[condition2,1] = wo[condition2,1] * torch.sin(PI_over_2 - PI_over_4 * wo[condition2,0]/wo[condition2,1])

        return woSample, torch.full((num_samples,), INV_PI,  device=DEVICE_, dtype = DTYPE_)

    def prob(self, z, rm_invalid = True):
        # cosTheta = torch.sqrt(torch.clamp(1 - torch.square(z[:,0]) - torch.square(z[:,1]), 0.0, 1.0))
        # pdf = INV_PI * cosTheta
        invalid_mask = torch.square(z[:,0]) + torch.square(z[:,1]) >1
        pdf = torch.full((z.shape[0],), INV_PI).to(z.device) ### TODO debug
        # pdf = torch.full((z.shape[0],), 1/4.0).to(z.device) ### TODO debug

        if rm_invalid:
            pdf[invalid_mask] = 0.0 ##TODO
        return pdf
        
    # def prob(self, z):
    #     pdf = torch.full((z.shape[0],), 0.25).to(z.device) ### TODO debug
    #     return pdf

from torch.autograd import Variable

class WeightedSumDistribution(nn.Module):
    def __init__(self, cond_C, verbose=True) -> None:
        super().__init__()
        self.cond_C = cond_C
        # self.weight = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        # self.weight.requires_grad=True
        self.net = mlp.MLP([self.cond_C, 32, 32, 5], leaky=0.01)
        # self.net = mlp.MLP([self.cond_C, 128, 512, 128, 32, 5]) #using ReLU
        self.gaussianLobe = DiagGaussianCond(2)
        self.lambertianLobe = LambertianLobe()
        for param in self.lambertianLobe.parameters():
            param.requires_grad = False

    def forward(self, cond_vec, num_samples=1):
        num_gauss = int(num_samples * weight) 
        mean, sigma, weight = self.learn_prior(cond_vec)
        # weight = nn.Sigmoid()(weight)
        z_guass,_ = self.gaussianLobe(mean, sigma, num_gauss)
        z_lambert,_ = self.lambertianLobe(num_samples - num_gauss)
        samples =  torch.stack((z_guass, z_lambert), dim=0)
        return samples[torch.randperm(num_samples), :] #shuffle

    def learn_prior(self, cond_vec):
        h = self.net(cond_vec)
        mean, sigma = h[:, 0:2], nn.functional.softplus(h[:, 2::4])
        weight = nn.Sigmoid()(h[:,4])
        return mean, sigma, weight

    def log_prob(self, z, cond_vec):

        mean, sigma, weight = self.learn_prior(cond_vec)
        guass_log_p = self.gaussianLobe.prob(z, mean, sigma)        
        lambert_log_p = self.lambertianLobe.prob(z)
        p = guass_log_p * weight + lambert_log_p * (1 - weight)+ 1e-5
        # print((guass_log_p<=0).any(), (weight <=0).any(), (lambert_log_p<=0).any())
        log_p =  torch.log(p)
        return log_p


def softmax(x, dim):
    f = torch.exp(x - torch.max(x))  # shift values
    return f / torch.sum(f, dim).unsqueeze(-1)

DEVICE_ = torch.device(0) 
DTYPE_ = torch.double

class WeightedSumDistributionNoCond(nn.Module):
    def __init__(self, dim=2, verbose=True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.gaussianLobe = DiagGaussian(2)
        self.dim = dim
        self.n_modes = 1
        loc = np.random.randn(1, self.dim)
        loc = np.array(loc)[None, ...]
        scale = np.ones((1, self.dim))
        scale = np.array(scale)[None, ...]

        weights = np.ones(self.n_modes+1)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)

        self.loc = nn.Parameter(torch.tensor(1.0 * loc))
        self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 * scale)))
        self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights)))

        self.lambertianLobe = LambertianLobe()
        for param in self.lambertianLobe.parameters():
            param.requires_grad = False

    def forward(self, num_samples=1):
        weights = torch.softmax(self.weight_scores, 1)
        print("num_samples ", num_samples)
        rdn = torch.rand(num_samples)
        num_lambert = torch.sum(rdn<weights[...,-1].item())
        # num_lambert = int(weights[...,-1].item()) * num_samples
        print("num_lambert",num_lambert)

        num_gmm = num_samples - num_lambert
        # Sample mode indices
        mode = torch.multinomial(weights[0, :-1], num_samples, replacement=True)
        mode_1h = nn.functional.one_hot(mode, self.n_modes)
        mode_1h = mode_1h[..., None]
        print("mode_1h.shape", mode_1h.shape)
        print("self.log_scale.shape ", self.log_scale.shape)
        # Get samples
        eps_ = torch.randn(
            num_samples, self.dim, dtype=self.loc.dtype, device=self.loc.device
        )
        scale_sample = torch.sum(torch.exp(self.log_scale) * mode_1h, 1)
        loc_sample = torch.sum(self.loc * mode_1h, 1)
        z_guass = eps_ * scale_sample + loc_sample

        z_lambert,_ = self.lambertianLobe(num_lambert)
        # z_lambert = z_lambert
        samples =  torch.cat((z_guass[:num_gmm,...], z_lambert), dim=0)
        z =  samples[torch.randperm(num_samples), :] #shuffle

        return z, self.log_prob(z)


    def log_prob(self, z):
        weights = torch.softmax(self.weight_scores, 1)
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights[:,:1])
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(self.log_scale, 2)
        )
        guass_log_p =log_p# torch.logsumexp(log_p, 1)
        lambert_log_p = torch.log(self.lambertianLobe.prob(z)) + torch.log(weights[...,1])
        log_p = torch.cat((guass_log_p, lambert_log_p.unsqueeze(1)), dim=1)
        log_p = torch.logsumexp(log_p, 1)
        return log_p


def test():
    gaussianLobe = DiagGaussian(2, trainable=False)
    t = torch.rand(2,)
    logp1 = gaussianLobe.log_prob(t)
    m = MultivariateNormal(torch.zeros(2), torch.eye(2))
    logp2 = m.log_prob(t)
    print("logp1, logp2", logp1, logp2)



class ConditionalDiaGaussian(nn.Module):
    def __init__(self, dim, feat_dim):
        super().__init__()
        self.shape = (dim,)
        self.n_dim = dim
        self.d = np.prod(self.shape)
        self.net= mlp.MLP([feat_dim, 32, 2*dim], leaky=0.01)

    def learn_prior(self, cond_vec):
        h = self.net(cond_vec)
        loc = h[:, :self.n_dim].unsqueeze(1)
        log_scale = h[:,self.n_dim:].unsqueeze(1)

        return loc, log_scale

    def forward(self, cond_vector, num_samples=1):
        loc, log_scale = self.learn_prior(cond_vector)
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device
        )
        z = loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p

    def log_prob(self, z, cond_vector):
        loc, log_scale = self.learn_prior(cond_vector)
        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p

class GMMWeightedCond(nn.Module):
    """
    Mixture of Gaussians with diagonal covariance matrix, weighted sum with a lambertian lobe, 
    condition on (uv, wo)
    """

    def __init__(
        self, n_modes, dim, cond_C, trainable=True
    ):
     
        super().__init__()
        self.n_modes = n_modes  
        self.dim = dim
        self.cond_C = cond_C
        num_out_features = self.n_modes * (self.dim) * 2 + self.n_modes + 1 #loc scale, weights

        self.net = mlp.MLP([self.cond_C, 32, num_out_features])
        self.lambertianLobe = LambertianLobe()
        for param in self.lambertianLobe.parameters():
            param.requires_grad = False
        self._initialize_weights()
    
    def _initialize_weights(self):
        def initGaussian(m):
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight, std=0.01)
                if m.bias != None:
                    torch.nn.init.zeros_(m.bias)

        self.net.apply(initGaussian)

    def learn_prior(self, cond_vec):
        h = self.net(cond_vec)
        loc = h[:, :self.n_modes * self.dim] 

        loc = loc.reshape(-1,self.n_modes, self.dim)
        
        scale = h[:, self.n_modes * self.dim  : self.n_modes * self.dim * 2]
        scale = scale.reshape(-1,self.n_modes, self.dim)
        
        weights = h[:, self.n_modes * self.dim * 2:] #nn.functional.softplus(
        assert(weights.shape[-1] == (self.n_modes+1))
        # weights /= torch.sum(weights, dim=1).unsqueeze(-1)
        # weights = torch.softmax(weights, 1)
        weights = torch.abs(weights)
        weights /= torch.sum(weights, dim=1).unsqueeze(-1)
        return loc, scale, weights

    def sample(self, cond_vec, num_samples=1, randseed=None):
        return self.forward(cond_vec, num_samples, randseed)

    def forward(self, cond_vec, num_samples=1, randseed=None, verbose=False):
        # Get weights
        # randseed = stratified_sampling_2d(num_samples).cuda()

        assert(self.n_modes == 2)
        loc, log_scale, weights = self.learn_prior(cond_vec)
        # weights_gauss = weights[...,:-1].clone()
        # weights_gauss /= torch.sum(weights_gauss, dim=1).unsqueeze(-1)
        
        if randseed==None:
            rdn = torch.rand(num_samples, device=DEVICE_, dtype = DTYPE_)
        else:
            rdn = randseed[:,0]

        weights_cumsum =torch.cumsum(weights, dim=-1)
        gauss1_mask = rdn < weights_cumsum[...,0]
        gauss2_mask = (~gauss1_mask) & (rdn < weights_cumsum[...,1])
        gauss_mask = gauss1_mask | gauss2_mask
        lambert_mask = ~gauss_mask
        num_gauss = torch.sum(gauss_mask)
        num_gauss1 = torch.sum(gauss1_mask)
        num_gauss2 = num_gauss - num_gauss1
        num_lambert = num_samples - num_gauss

        mode = torch.zeros((num_samples,  1), dtype=torch.long, device=DEVICE_)
        mode[gauss1_mask] = torch.full((num_gauss1,1), 0 , device=DEVICE_)
        mode[gauss2_mask] = torch.full((num_gauss2,1), 1,  device=DEVICE_)

        if randseed!=None:
            randseed[gauss1_mask, 0] = randseed[gauss1_mask, 0] / weights_cumsum[gauss1_mask,0]
            randseed[gauss2_mask, 0] = (randseed[gauss2_mask, 0] - weights_cumsum[gauss2_mask,0])/weights[gauss2_mask,1]
            randseed[lambert_mask, 0] = (randseed[lambert_mask, 0] - weights_cumsum[lambert_mask,1])/weights[lambert_mask,-1]

        if(verbose):
            print("num_samples = %d, num_lambert = %d, num_gauss =%d, gauss percentage = %f", num_samples,num_lambert.item(), num_gauss.item(), num_gauss.item()/num_samples)
        # Sample mode indices
        # mode = torch.multinomial(weights_gauss[gauss_mask], 1, replacement=True)

        mode_1h = nn.functional.one_hot(mode[gauss_mask], self.n_modes).squeeze().unsqueeze(-1) #CHEKED CORRECTLY!
        # Get samples
        if randseed==None:
            eps_ = torch.randn(
                num_gauss, self.dim, dtype=loc.dtype, device=loc.device
            )
            z_lambert, logp_lambert = self.lambertianLobe(num_lambert)
        else:
            U1 = randseed[gauss_mask,0]
            U2 = randseed[gauss_mask,1]
            R = torch.sqrt(-2 * torch.log(U1))
            theta = 2* torch.pi * U2 
            X = R * torch.cos(theta)
            Y = R * torch.sin(theta)
            eps_ = torch.stack([X, Y], dim=-1)

            z_lambert, logp_lambert = self.lambertianLobe(num_lambert, randseed[lambert_mask,...])
        
        scale_sample = torch.sum(torch.exp(log_scale[gauss_mask]) * mode_1h, 1)
        loc_sample = torch.sum(loc[gauss_mask] * mode_1h, 1)
        z_guass = eps_ * scale_sample + loc_sample
        
        z = torch.zeros(cond_vec.shape[0], 2,  device=DEVICE_, dtype = DTYPE_)
        z[lambert_mask] = z_lambert
        z[gauss_mask] = z_guass

        ## compute the pdfs
        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_gauss_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights[...,:-1]+1e-5)
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )
        lambert_log_p = torch.log(self.lambertianLobe.prob(z)+1e-5) + torch.log(weights[...,-1]) 
        log_p = torch.cat((log_gauss_p, lambert_log_p.unsqueeze(1)), dim=-1)
        log_p = torch.logsumexp(log_p, 1)

        return z, log_p

    def log_prob(self, z, cond_vec):
        # return self.log_prob_topweight_gauss(z, cond_vec)
        return self.log_prob_mixture(z, cond_vec)

    def log_prob_mixture(self, z, cond_vec):
            # Get weights
            loc, log_scale, weights = self.learn_prior(cond_vec)
            # Compute log probability
            eps = (z[:, None, :] - loc) / torch.exp(log_scale)
            log_p = (
                -0.5 * self.dim * np.log(2 * np.pi)
                + torch.log(weights[...,:-1]+1e-5) ###TODO check??? this weights
                - 0.5 * torch.sum(torch.pow(eps, 2), 2)
                - torch.sum(log_scale, 2)
            )

            lambert_log_p = torch.log(self.lambertianLobe.prob(z)+1e-5) + torch.log(weights[...,-1]) ####TODO?? check
            log_p = torch.cat((log_p, lambert_log_p.unsqueeze(1)), dim=-1)
            log_p = torch.logsumexp(log_p, 1)
            return log_p

    def log_prob_lambert(self, z, cond_vec):
        log_p = torch.log(self.lambertianLobe.prob(z)+1e-5)
        return log_p

    def log_prob_gmm(self, z, cond_vec):
        loc, log_scale, weights = self.learn_prior(cond_vec)
        weights = weights[...,:-1]
        weights /= torch.sum(weights, dim=1).unsqueeze(-1) #renormalize
        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights+1e-5) ###TODO check
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )
        return log_p
    
    

    def log_prob_topsharp_gauss(self, z, cond_vec):
        ## get the narrowest gauss
        loc, log_scale, weight_scores = self.learn_prior(cond_vec)
        mean_xy_log_scale = torch.sum(log_scale, dim=-1)
        print(mean_xy_log_scale.shape)
        min_log_scale = torch.min(mean_xy_log_scale, dim=1)
        idx_mode = min_log_scale.indices        
        # sharp_log_scale0 = log_scale[torch.arange(log_scale.shape[0]), idx_mode*self.dim]
        sharp_log_scale = log_scale[torch.arange(log_scale.shape[0]), idx_mode].squeeze()
        print("sharp_log_scale.shape", sharp_log_scale.shape)

        sharp_loc = loc[torch.arange(log_scale.shape[0]),idx_mode].squeeze() 
        print("sharp_loc.shape", sharp_loc.shape)

        sharp_log_p = -0.5 * self.dim * np.log(2 * np.pi) - torch.sum(sharp_log_scale +
            0.5 * torch.pow((z - sharp_loc) / torch.exp(sharp_log_scale), 2),
            list(range(1, 2)),
        )
        return sharp_log_p

    def log_prob_topweight_gauss(self, z, cond_vec):
        ## get the narrowest gauss
        loc, log_scale, weight_scores = self.learn_prior(cond_vec)
        
        min_log_scale = torch.max(weight_scores, dim=1)
        idx_mode = min_log_scale.indices        
        # sharp_log_scale0 = log_scale[torch.arange(log_scale.shape[0]), idx_mode*self.dim]
        sharp_log_scale = log_scale[torch.arange(log_scale.shape[0]), idx_mode].squeeze()
        # print("sharp_log_scale.shape", sharp_log_scale.shape)

        sharp_loc = loc[torch.arange(log_scale.shape[0]),idx_mode].squeeze() 
        # print("sharp_loc.shape", sharp_loc.shape)

        sharp_log_p = -0.5 * self.dim * np.log(2 * np.pi) - torch.sum(sharp_log_scale +
            0.5 * torch.pow((z - sharp_loc) / torch.exp(sharp_log_scale), 2),
            list(range(1, 2)),
        )
        return sharp_log_p


class GMMWeightedCondLarge(nn.Module):

    def __init__(
        self, n_modes, dim, cond_C, trainable=True
    ):
     
        super().__init__()
        self.n_modes = n_modes  
        self.dim = dim
        self.cond_C = cond_C
        num_out_features = self.n_modes * (self.dim) * 2 + self.n_modes + 1 #loc scale, weights

        # self.net = mlp.MLP([self.cond_C, 128, 128, num_out_features])
        # self.net = mlp.MLP([self.cond_C, 32, num_out_features])
        self.net = mlp.MLP([self.cond_C, 128, 512, 128, 32, num_out_features]) #using ReLU
        # self.net = mlp.MLP([self.cond_C, 32, 128, 64, 32, num_out_features]) #using ReLU

        self.lambertianLobe = LambertianLobe()
        for param in self.lambertianLobe.parameters():
            param.requires_grad = False
        self._initialize_weights()
    
    def _initialize_weights(self):
        def initGaussian(m):
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight, std=0.01)

                if m.bias != None:
                    torch.nn.init.zeros_(m.bias)

        self.net.apply(initGaussian)

    def learn_prior(self, cond_vec):
        h = self.net(cond_vec)
        loc = h[:, :self.n_modes * self.dim] 

        loc = loc.reshape(-1,self.n_modes, self.dim)
        
        scale = h[:, self.n_modes * self.dim  : self.n_modes * self.dim * 2]
        scale = scale.reshape(-1,self.n_modes, self.dim)
        
        weights = h[:, self.n_modes * self.dim * 2:] #nn.functional.softplus(
        assert(weights.shape[-1] == (self.n_modes+1))
        # weights /= torch.sum(weights, dim=1).unsqueeze(-1)
        # weights = torch.softmax(weights, 1)
        weights = torch.abs(weights)
        weights /= torch.sum(weights, dim=1).unsqueeze(-1)
        return loc, scale, weights

    def sample(self, cond_vec, num_samples=1, randseed=None):
        return self.forward(cond_vec, num_samples, randseed)

    def forward(self, cond_vec, num_samples=1, randseed=None, verbose=False):
        # Get weights
        loc, log_scale, weights = self.learn_prior(cond_vec)
        weights_gauss = weights[...,:-1].clone()
        weights_gauss /= torch.sum(weights_gauss, dim=1).unsqueeze(-1)
        if randseed==None:
            rdn = torch.rand(num_samples, device=DEVICE_, dtype = DTYPE_)
        else:
            rdn = randseed[:,0] #get the first dimension of the stratified samples
            # print("what the hell happened!!!")
            # rdn = torch.rand(num_samples).cuda()

        lambert_mask = rdn < weights[...,-1]
        gauss_mask = ~lambert_mask
        if randseed!=None:
            randseed[lambert_mask, 0] = randseed[lambert_mask, 0] / weights[lambert_mask,-1]
            randseed[gauss_mask, 0] = (randseed[gauss_mask, 0] - weights[gauss_mask,-1])/(1 - weights[gauss_mask,-1])

        num_gauss = torch.sum(gauss_mask)
        num_lambert = num_samples - num_gauss
        if(verbose):
            print("num_samples = %d, num_lambert = %d, num_gauss =%d, gauss percentage = %f", num_samples,num_lambert.item(), num_gauss.item(), num_gauss.item()/num_samples)
        # Get samples
        if randseed==None:
            eps_ = torch.randn(
                num_gauss, self.dim, dtype=loc.dtype, device=loc.device
            )
            z_lambert, logp_lambert = self.lambertianLobe(num_lambert)

        else:
            U1 = randseed[gauss_mask,0]
            U2 = randseed[gauss_mask,1]
            R = torch.sqrt(-2 * torch.log(U1))
            theta = 2* torch.pi * U2 
            X = R * torch.cos(theta)
            Y = R * torch.sin(theta)
            eps_ = torch.stack([X, Y], dim=-1)
            z_lambert, logp_lambert = self.lambertianLobe(num_lambert, randseed[lambert_mask,...])

        # print("eps_.shape", eps_.shape)
        scale_sample = torch.sum(torch.exp(log_scale[gauss_mask]), 1)
        loc_sample = torch.sum(loc[gauss_mask], 1)
        z_guass = eps_ * scale_sample + loc_sample
        # z_lambert = z_lambert.cuda()
        # logp_lambert = logp_lambert.cuda()
        z = torch.zeros(cond_vec.shape[0], 2,  device=DEVICE_, dtype = DTYPE_)
        z[lambert_mask] = z_lambert
        z[gauss_mask] = z_guass

        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights[...,:-1]+1e-5) ###TODO check??? this weights
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )

        lambert_log_p = torch.log(self.lambertianLobe.prob(z)+1e-5) + torch.log(weights[...,-1]) ####TODO?? check
        log_p = torch.cat((log_p, lambert_log_p.unsqueeze(1)), dim=-1)
        log_p = torch.logsumexp(log_p, 1)
        return z, log_p

    def log_prob(self, z, cond_vec):
        # return self.log_prob_topweight_gauss(z, cond_vec)
        return self.log_prob_mixture(z, cond_vec)

    def log_prob_mixture(self, z, cond_vec):
        # Get weights
        loc, log_scale, weights = self.learn_prior(cond_vec)
        # Compute log probability
        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights[...,:-1]+1e-5) ###TODO check??? this weights
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )

        lambert_log_p = torch.log(self.lambertianLobe.prob(z)+1e-5) + torch.log(weights[...,-1]) ####TODO?? check
        log_p = torch.cat((log_p, lambert_log_p.unsqueeze(1)), dim=-1)
        log_p = torch.logsumexp(log_p, 1)
        return log_p


def stratified_sampling_2d(wavefront_size):
    #round spp to square number
    side = 1
    while (side*side< wavefront_size):
        side += 1
    us = torch.arange(0, side)/side 
    vs = torch.arange(0, side)/side
    u, v = torch.meshgrid(us, vs)
    uv = torch.stack([u,v], dim = -1)
    uv = uv.reshape(-1,2)[:wavefront_size,...]
    uv = uv[torch.randperm(uv.shape[0]), ...]
    # uv_all_pixels = uv.unsqueeze(0).repeat(n_pixels, 1,1).reshape(-1,2)
    
    jitter = torch.rand((wavefront_size, 2))/side
    return uv + jitter 

class GMMWeightedCondIsoLarge(nn.Module):

    def __init__(
        self, n_modes, dim, cond_C, trainable=True
    ):
     
        super().__init__()
        self.n_modes = n_modes  
        self.dim = dim
        self.cond_C = cond_C
        num_out_features = self.n_modes * (self.dim+1) + self.n_modes + 1 #loc scale, weights

        # self.net = mlp.MLP([self.cond_C, 128, 512, 128, 32, num_out_features]) #using ReLU
        self.net = mlp.MLP([self.cond_C, 32, 32, 32, num_out_features]) #using ReLU


        self.lambertianLobe = LambertianLobe()
        for param in self.lambertianLobe.parameters():
            param.requires_grad = False
        self._initialize_weights()
    
    def _initialize_weights(self):
        def initGaussian(m):
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight, std=0.01)

                if m.bias != None:
                    torch.nn.init.zeros_(m.bias)

        self.net.apply(initGaussian)

    def learn_prior(self, cond_vec):
        h = self.net(cond_vec)
        loc = h[:, :self.n_modes * self.dim] 
        loc = loc.reshape(-1,self.n_modes, self.dim)
        scale = h[:, self.n_modes * self.dim  : self.n_modes * (self.dim +1)]

        scale = scale.reshape(-1, self.n_modes, 1)
        scale = scale.repeat(1, 1, self.dim)
        
        weights = h[:, self.n_modes * (self.dim + 1):] #nn.functional.softplus(
        assert(weights.shape[-1] == (self.n_modes+1))
        # weights /= torch.sum(weights, dim=1).unsqueeze(-1)
        # weights = torch.softmax(weights, 1)
        weights = torch.abs(weights)

        weights /= torch.sum(weights, dim=1).unsqueeze(-1)
        return loc, scale, weights

    def sample(self, cond_vec, num_samples=1, randseed=None):
        return self.forward(cond_vec, num_samples, randseed)

    def forward(self, cond_vec, num_samples=1, randseed=None, verbose=False):
        # Get weights
        # randseed = stratified_sampling_2d(num_samples).cuda()
        loc, log_scale, weights = self.learn_prior(cond_vec)
        # weights_gauss = weights[...,:-1].clone()
        # weights_gauss /= torch.sum(weights_gauss, dim=1).unsqueeze(-1)

        if randseed==None:
            rdn = torch.rand(num_samples,  device=DEVICE_, dtype = DTYPE_)
        else:
            rdn = randseed[...,0]#get the first dimension of the stratified samples

        lambert_mask = rdn < weights[...,-1]
        gauss_mask = ~lambert_mask

        if randseed!=None:
            randseed[lambert_mask, 0] = randseed[lambert_mask, 0] / weights[lambert_mask,-1]
            randseed[gauss_mask, 0] = (randseed[gauss_mask, 0] - weights[gauss_mask,-1])/(1 - weights[gauss_mask,-1])

        num_gauss = torch.sum(gauss_mask)
        num_lambert = num_samples - num_gauss
        if(verbose):
            print("num_samples = %d, num_lambert = %d, num_gauss =%d, gauss percentage = %f", num_samples,num_lambert.item(), num_gauss.item(), num_gauss.item()/num_samples)
        # Get samples
        if randseed==None:
            eps_ = torch.randn(
                num_gauss, self.dim, dtype=loc.dtype, device=loc.device
            )
            z_lambert, logp_lambert = self.lambertianLobe(num_lambert)

        else:
            # eps_ = randseed[gauss_mask,...].cuda()
            U1 = randseed[gauss_mask,0]
            U2 = randseed[gauss_mask,1]
            R = torch.sqrt(-2 * torch.log(U1))
            theta = 2* torch.pi * U2 
            X = R * torch.cos(theta)
            Y = R * torch.sin(theta)
            eps_ = torch.stack([X, Y], dim=-1)

            z_lambert, logp_lambert = self.lambertianLobe(num_lambert, randseed[lambert_mask,...])

        # print("eps_.shape", eps_.shape)
        scale_sample = torch.sum(torch.exp(log_scale[gauss_mask]), 1)
        loc_sample = torch.sum(loc[gauss_mask], 1)
        z_guass = eps_ * scale_sample + loc_sample
        # z_lambert = z_lambert.cuda()
        # logp_lambert = logp_lambert.cuda()
        z = torch.zeros(cond_vec.shape[0], 2, device=DEVICE_, dtype = DTYPE_)
        z[lambert_mask] = z_lambert
        z[gauss_mask] = z_guass

        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights[...,:-1]+1e-5) ###TODO check??? this weights
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )

        lambert_log_p = torch.log(self.lambertianLobe.prob(z)+1e-5) + torch.log(weights[...,-1]) ####TODO?? check
        log_p = torch.cat((log_p, lambert_log_p.unsqueeze(1)), dim=-1)
        log_p = torch.logsumexp(log_p, 1)

        return z, log_p

    def log_prob(self, z, cond_vec, wo):
        # return self.log_prob_topweight_gauss(z, cond_vec)
        return self.log_prob_mixture(z, cond_vec)

    def log_prob_mixture(self, z, cond_vec):
        # Get weights
        loc, log_scale, weights = self.learn_prior(cond_vec)
        # weights_gauss = weights[...,:-1].clone()
        # weights_gauss /= torch.sum(weights_gauss, dim=1).unsqueeze(-1)
        # Compute log probability
        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(torch.clamp(weights[...,:-1],1e-5)) ###TODO check??? this weights
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )

        lambert_log_p = torch.log(torch.clamp(self.lambertianLobe.prob(z), 1e-5)) + torch.log(weights[...,-1]) ####TODO?? check
        log_p = torch.cat((log_p, lambert_log_p.unsqueeze(1)), dim=-1)
        log_p = torch.logsumexp(log_p, 1)
        return log_p




# class GMMWeightedCondIsoLarge(nn.Module):
#     """
#     no normal-mapped isotropic gaussian version
#     """
#     def __init__(
#         self, n_modes, dim, cond_C, trainable=True
#     ):
     
#         super().__init__()
#         self.n_modes = n_modes  
#         self.dim = dim
#         self.cond_C = cond_C
#         num_out_features = self.n_modes + self.n_modes + 1 #loc scale, weights
#         self.net = mlp.MLP([self.cond_C, 32, 32, 32, num_out_features]) #using ReLU
#         self.lambertianLobe = LambertianLobe()
#         for param in self.lambertianLobe.parameters():
#             param.requires_grad = False
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         def initGaussian(m):
#             if type(m) == nn.Linear:
#                 torch.nn.init.normal_(m.weight, std=0.01)

#                 if m.bias != None:
#                     torch.nn.init.zeros_(m.bias)

#         self.net.apply(initGaussian)

#     def learn_prior(self, cond_vec, wo):
#         h = self.net(cond_vec)
#         loc = -wo #the reflection direction
#         loc = loc.reshape(-1,self.n_modes, self.dim)
#         scale = h[:, :self.n_modes] 
#         scale = scale.reshape(-1, self.n_modes, 1)
#         scale = scale.repeat(1, 1, self.dim)
        
#         weights = h[:, self.n_modes:] #nn.functional.softplus(
#         weights[...,0] = 0.5
#         weights[...,1] = 0.5
#         assert(weights.shape[-1] == (self.n_modes+1))
#         weights = torch.abs(weights)
#         weights /= torch.sum(weights, dim=1).unsqueeze(-1)
#         return loc, scale, weights

#     def sample(self, cond_vec, num_samples=1, randseed=None):
#         return self.forward(cond_vec, num_samples, randseed)

#     def forward(self, cond_vec, wo, num_samples=1, randseed=None, verbose=False):
#         # Get weights
#         # randseed = stratified_sampling_2d(num_samples).cuda()
#         loc, log_scale, weights = self.learn_prior(cond_vec, wo)
#         if randseed==None:
#             rdn = torch.rand(num_samples,  device=DEVICE_, dtype = DTYPE_)
#         else:
#             rdn = randseed[...,0]#get the first dimension of the stratified samples

#         lambert_mask = rdn < weights[...,-1]
#         gauss_mask = ~lambert_mask

#         if randseed!=None:
#             randseed[lambert_mask, 0] = randseed[lambert_mask, 0] / weights[lambert_mask,-1]
#             randseed[gauss_mask, 0] = (randseed[gauss_mask, 0] - weights[gauss_mask,-1])/(1 - weights[gauss_mask,-1])

#         num_gauss = torch.sum(gauss_mask)
#         num_lambert = num_samples - num_gauss
#         if(verbose):
#             print("num_samples = %d, num_lambert = %d, num_gauss =%d, gauss percentage = %f", num_samples,num_lambert.item(), num_gauss.item(), num_gauss.item()/num_samples)
#         # Get samples
#         if randseed==None:
#             eps_ = torch.randn(
#                 num_gauss, self.dim, dtype=loc.dtype, device=loc.device
#             )
#             z_lambert, logp_lambert = self.lambertianLobe(num_lambert)

#         else:
#             # eps_ = randseed[gauss_mask,...].cuda()
#             U1 = randseed[gauss_mask,0]
#             U2 = randseed[gauss_mask,1]
#             R = torch.sqrt(-2 * torch.log(U1))
#             theta = 2* torch.pi * U2 
#             X = R * torch.cos(theta)
#             Y = R * torch.sin(theta)
#             eps_ = torch.stack([X, Y], dim=-1)

#             z_lambert, logp_lambert = self.lambertianLobe(num_lambert, randseed[lambert_mask,...])

#         # print("eps_.shape", eps_.shape)
#         scale_sample = torch.sum(torch.exp(log_scale[gauss_mask]), 1)
#         loc_sample = torch.sum(loc[gauss_mask], 1)
#         z_guass = eps_ * scale_sample + loc_sample
#         z = torch.zeros(cond_vec.shape[0], 2, device=DEVICE_, dtype = DTYPE_)
#         z[lambert_mask] = z_lambert
#         z[gauss_mask] = z_guass

#         eps = (z[:, None, :] - loc) / torch.exp(log_scale)
#         log_p = (
#             -0.5 * self.dim * np.log(2 * np.pi)
#             + torch.log(weights[...,:-1]+1e-5) ###TODO check??? this weights
#             - 0.5 * torch.sum(torch.pow(eps, 2), 2)
#             - torch.sum(log_scale, 2)
#         )

#         lambert_log_p = torch.log(self.lambertianLobe.prob(z)+1e-5) + torch.log(weights[...,-1]) ####TODO?? check
#         log_p = torch.cat((log_p, lambert_log_p.unsqueeze(1)), dim=-1)
#         log_p = torch.logsumexp(log_p, 1)

#         return z, log_p

#     def log_prob(self, z, cond_vec, wo):
#         loc, log_scale, weights = self.learn_prior(cond_vec, wo)
#         eps = (z[:, None, :] - loc) / torch.exp(log_scale)
#         log_p = (
#             -0.5 * self.dim * np.log(2 * np.pi)
#             + torch.log(torch.clamp(weights[...,:-1],1e-5)) ###TODO check??? this weights
#             - 0.5 * torch.sum(torch.pow(eps, 2), 2)
#             - torch.sum(log_scale, 2)
#         )
#         lambert_log_p = torch.log(torch.clamp(self.lambertianLobe.prob(z), 1e-5)) + torch.log(weights[...,-1]) ####TODO?? check
#         log_p = torch.cat((log_p, lambert_log_p.unsqueeze(1)), dim=-1)
#         log_p = torch.logsumexp(log_p, 1)
#         return log_p


class BoundedMLP(nn.Module):
    """Classic MLP with three hidden layers"""
    def __init__(self, num_inputs, num_outputs, num_hidden, gmm_modes=2, dim=2) -> None:
        super().__init__()
        # self.bound_weight0 = nn.Parameter(torch.Tensor([0.01,0.01]))
        self.bound_offset0 = nn.Parameter(torch.Tensor([-0.1,-0.1]))
        # self.bound_weight1 = nn.Parameter(torch.Tensor([0.01,0.01]))
        self.bound_offset1 = nn.Parameter(torch.Tensor([-2.0,-2.0]))
        self.gmm_modes = gmm_modes
        self.dim = dim
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.LeakyReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.LeakyReLU(),
            nn.Linear(num_hidden, num_outputs),
        )

    def forward(self, x):
        x =  self.layers(x)
        # value = torch.tanh(x)
        value = x.clone()
        idx_logscale = self.gmm_modes * self.dim 
        # value[:,4:6] = nn.functional.linear(value[:,4:6], weight = self.bound_weight0[0], bias=self.bound_offset0[0]) #TODO only for 2 guass for now
        # value[:,6:8] = nn.functional.linear(value[:,6:8], weight = self.bound_weight1, bias=self.bound_offset1) #TODO
        # value[:,:self.dim] = value[:,:self.dim]
        # value[:,idx_logscale:idx_logscale+self.dim] = self.bound_offset0.unsqueeze(0)
        value[:,idx_logscale+self.dim:idx_logscale+self.dim * 2] = self.bound_offset1.unsqueeze(0)
        
        return value



class GMMCond(nn.Module): #only for two gaussians right now
    """
    conditional gaussian mixture models
    """
    def __init__(self, n_modes, dim, cond_C, trainable=True) -> None:
        super().__init__()
        self.n_modes = n_modes
        # assert(self.n_modes == 2)
        self.dim = dim
        self.cond_C = cond_C
        num_out_features = self.n_modes * (self.dim) * 2 + self.n_modes #loc scale, weights
        # self.net = mlp.MLP([self.cond_C, 32, 32, num_out_features])
        self.net = BoundedMLP(num_inputs=self.cond_C,num_outputs= num_out_features, num_hidden= 64, gmm_modes=n_modes)

        
    def learn_prior(self, cond_vec):
        h = self.net(cond_vec)
        loc = h[:, :self.n_modes * self.dim] 
        loc = loc.reshape(-1,self.n_modes, self.dim)
        scale = h[:, self.n_modes * self.dim  : self.n_modes * self.dim * 2]
        scale = scale.reshape(-1,self.n_modes, self.dim)
        weights = h[:, self.n_modes * self.dim * 2:] 
        assert(weights.shape[-1] == self.n_modes)
        weights /= torch.sum(weights, dim=1).unsqueeze(-1)
        return loc, scale, weights

    def forward(self, cond_vec, num_samples=1):
        # Get weights
        loc, log_scale, weight_scores = self.learn_prior(cond_vec)
        weights = torch.softmax(weight_scores, 1)
        # Sample mode indices
        mode = torch.multinomial(weights[0, :], num_samples, replacement=True)
        mode_1h = nn.functional.one_hot(mode, self.n_modes)
        mode_1h = mode_1h[..., None]
        
        # Get samples
        eps_ = torch.randn(
            num_samples, self.dim, dtype=loc.dtype, device=loc.device
        )
        scale_sample = torch.sum(torch.exp(log_scale) * mode_1h, 1)
        loc_sample = torch.sum(loc * mode_1h, 1)
        z = eps_ * scale_sample + loc_sample
        return z, self.log_prob(z, cond_vec)

    
    def log_prob(self, z, cond_vec):
        # Get weights
        loc, log_scale, weight_scores = self.learn_prior(cond_vec)
        weights = torch.softmax(weight_scores, 1)
        # weights[:,0] *= 0.25 * M_PI ###TODO!!! remove
        # weights[:,1] *= 4 * INV_PI ###TODO!! remove
        # Compute log probability
        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights+1e-5) ###TODO check
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )
        log_p = torch.logsumexp(log_p, 1)
        return log_p


    

### test gaussian lobes
if __name__ == "__main__":
    test()
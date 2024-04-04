import torch
from torch import nn
import numpy as np
from normflows.nets import mlp

class Squeeze(nn.Module):
    def __init__(self, factor=2):
        super(Squeeze, self).__init__()
        assert factor > 1 and isinstance(
            factor, int
        ), "no point of using this if factor <= 1"
        self.factor = factor

    def squeeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert h % self.factor == 0 and w % self.factor == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c, h // self.factor, self.factor, w // self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(
            bs, c * self.factor * self.factor, h // self.factor, w // self.factor
        )

        return x

    def unsqueeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert c >= 4 and c % 4 == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c // self.factor ** 2, self.factor, self.factor, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(bs, c // self.factor ** 2, h * self.factor, w * self.factor)
        return x

    def forward(self, x, reverse=False):
        if len(x.size()) != 4:
            raise NotImplementedError
        if not reverse:
            return self.squeeze_bchw(x)
        else:
            return self.unsqueeze_bchw(x)



class conv2d_zeros(nn.Conv2d):
    def __init__(
        self,
        channels_in,
        channels_out,
        filter_size=3,
        stride=1,
        padding=0,
        logscale=3.0,
    ):
        super().__init__(
            channels_in, channels_out, filter_size, stride=stride, padding=padding
        )
        self.register_parameter("logs", nn.Parameter(torch.zeros(channels_out, 1, 1)))
        self.logscale_factor = logscale

    def reset_parameters(self):
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        out = super().forward(input)
        return out * torch.exp(self.logs * self.logscale_factor)


class Gaussian_Diag(object):
    def __init__(self):
        super().__init__()
        pass

    def logp(self, x, mean, sigma):
        ones = torch.ones_like(x)
        ll = -0.5 * (x - mean) ** 2 / (sigma ** 2) - 0.5 * torch.log(
            2 * np.pi * (sigma ** 2) * ones
        )
        # print("ll.shape ", ll.shape)
        # return torch.sum(ll, [1, 2, 3])
        
        logprob =  torch.sum(ll, [1]) ###TODO change when the dimension changes!!
        return logprob
    def sample(self, mean, sigma, eps=0):
        noise = torch.randn_like(mean)
        return mean + eps * sigma * noise


#

class DiagGaussianCond(nn.Module):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """

    def __init__(self, shape, trainable=True):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)  #2
        # if trainable:
        #     self.loc = nn.Parameter(torch.zeros(1, *self.shape))
        #     self.log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        # else:
        #     self.register_buffer("loc", torch.zeros(1, *self.shape))
        #     self.register_buffer("log_scale", torch.zeros(1, *self.shape))
        self.temperature = None  # Temperature parameter for annealed sampling

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

    def log_prob(self, z, mean, sigma):
        if self.temperature is None:
            log_scale = sigma
        else:
            log_scale = sigma + np.log(self.temperature)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - mean) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p



class GaussianPrior(nn.Module):
    def __init__(self, C, cond_channels, final=True): ### C should be the input dimensions
        super(GaussianPrior, self).__init__()
        # self.flow_var_shape = flow_var_shape
        self.cond_channels = cond_channels
        self.final = final
        self.squeezer = Squeeze()
        self.prior = Gaussian_Diag()
        assert(C == 2)
        # self.prior = DiagGaussianCond(C) #2

        # if final:
        #     self.conv = conv2d_zeros(self.cond_channels, 2 * C, padding=1)
        # else:
        #     self.conv = conv2d_zeros(self.cond_channels + C // 2, C, padding=1)

        if final:
            self.conv = mlp.MLP([self.cond_channels, 32, 32, 2*C], leaky=0.01)
        else:
            assert(False)

    def split2d_prior(self, z, lr_feat_map):
        x = torch.cat((z, lr_feat_map), 1)
        h = self.conv(x)
        mean, sigma = h[:, 0::2], nn.functional.softplus(h[:, 1::2])
        return mean, sigma

    def final_prior(self, lr_feat_map):
        h = self.conv(lr_feat_map)
        mean, sigma = h[:, 0::2], nn.functional.softplus(h[:, 1::2])
        return mean, sigma

    def forward(
        self, x, lr_feat_map, eps, reverse, logpz=0, logdet=0, use_stored=False
    ):

        if not reverse: #learning density estimation
            # final prior computation
            mean, sigma = self.final_prior(lr_feat_map)
            # print("dim of x mean and sigma ", x.shape, mean.shape, sigma.shape)
            # logpz += self.prior.log_prob(x, mean, sigma)
            logpz += self.prior.logp(x, mean, sigma)
            z = x
        else:#sampling
            mean, sigma = self.final_prior(lr_feat_map)
            # print(mean, sigma)
            z = self.prior.sample(mean, sigma, eps=eps)
            # z,_ = self.prior.forward(mean, sigma)

        return z, logdet, logpz
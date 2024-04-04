import torch
from torch import nn
from .. import utils


class MLP(nn.Module):
    """
    A multilayer perceptron with Leaky ReLU nonlinearities
    """

    def __init__(
        self,
        layers,
        leaky=0.0,
        score_scale=None,
        output_fn=None,
        output_scale=None,
        init_zeros=False,
        dropout=None,
    ):
        """
        :param layers: list of layer sizes from start to end
        :param leaky: slope of the leaky part of the ReLU,
        if 0.0, standard ReLU is used
        :param score_scale: Factor to apply to the scores, i.e. output before
        output_fn.
        :param output_fn: String, function to be applied to the output, either
        None, "sigmoid", "relu", "tanh", or "clampexp"
        :param output_scale: Rescale outputs if output_fn is specified, i.e.
        scale * output_fn(out / scale)
        :param init_zeros: Flag, if true, weights and biases of last layer
        are initialized with zeros (helpful for deep models, see arXiv 1807.03039)
        :param dropout: Float, if specified, dropout is done before last layer;
        if None, no dropout is done
        """
        super().__init__()
        net = nn.ModuleList([])
        for k in range(len(layers) - 2):
            net.append(nn.Linear(layers[k], layers[k + 1]))
            net.append(nn.LeakyReLU(leaky))
        if dropout is not None:
            net.append(nn.Dropout(p=dropout))
        net.append(nn.Linear(layers[-2], layers[-1]))
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        if output_fn is not None:
            if score_scale is not None:
                net.append(utils.ConstScaleLayer(score_scale))
            if output_fn == "sigmoid":
                net.append(nn.Sigmoid())
            elif output_fn == "relu":
                net.append(nn.ReLU())
            elif output_fn == "tanh":
                net.append(nn.Tanh())
            # elif output_fn == "scale":
            #     net.append(nn.Tanh())
            #     net.append(nn.utils.weight_norm(Rescale(layers[-1])))
            elif output_fn == "clampexp":
                net.append(utils.ClampExp())
            else:
                NotImplementedError("This output function is not implemented.")
            if output_scale is not None:
                net.append(utils.ConstScaleLayer(output_scale))
        self.net = nn.Sequential(*net)
        # scalenet = nn.ModuleList([])
        # scalenet.append(nn.Tanh())
        # scalenet.append(nn.utils.weight_norm(Rescale(layers[-1]//2))) ##added check
        # self.scalenet = nn.Sequential(*scalenet)

    def forward(self, x):
        return self.net(x)

class CondMLP(nn.Module):
    """
    just a conditional converion of MLP; for convenience
    """

    def __init__(
        self,
        layers,
        leaky=0.0,
        score_scale=None,
        output_fn=None,
        output_scale=None,
        init_zeros=False,
        dropout=None,
    ):
       
        super().__init__()
        net = nn.ModuleList([])
        for k in range(len(layers) - 2):
            net.append(nn.Linear(layers[k], layers[k + 1]))
            net.append(nn.LeakyReLU(leaky))
        if dropout is not None:
            net.append(nn.Dropout(p=dropout))
        net.append(nn.Linear(layers[-2], layers[-1]))
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        if output_fn is not None:
            if score_scale is not None:
                net.append(utils.ConstScaleLayer(score_scale))
            if output_fn == "sigmoid":
                net.append(nn.Sigmoid())
            elif output_fn == "relu":
                net.append(nn.ReLU())
            elif output_fn == "tanh":
                net.append(nn.Tanh())
         
            elif output_fn == "clampexp":
                net.append(utils.ClampExp())
            else:
                NotImplementedError("This output function is not implemented.")
            if output_scale is not None:
                net.append(utils.ConstScaleLayer(output_scale))
        self.net = nn.Sequential(*net)
        # scalenet = nn.ModuleList([])
        # scalenet.append(nn.Tanh())
        # scalenet.append(nn.utils.weight_norm(Rescale(layers[-1]//2))) ##added check
        # self.scalenet = nn.Sequential(*scalenet)

    def forward(self, x, cond_features):
        xx = torch.cat((x, cond_features), dim=-1)
        return self.net(xx)




class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.

    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        x = self.weight * x
        return x

MLP_HIDDEN_LAYER_NEURONS = 8

class ThreeLayerMLP_s(nn.Module):
    """Classic MLP with three hidden layers"""
    def __init__(self, num_inputs, num_outputs) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, MLP_HIDDEN_LAYER_NEURONS),
            nn.LeakyReLU(),
            nn.Linear(MLP_HIDDEN_LAYER_NEURONS, MLP_HIDDEN_LAYER_NEURONS),
            nn.LeakyReLU(),
            nn.Linear(MLP_HIDDEN_LAYER_NEURONS, num_outputs),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)

class ThreeLayerMLP_t(nn.Module):
    """Classic MLP with three hidden layers"""
    def __init__(self, num_inputs, num_outputs) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, MLP_HIDDEN_LAYER_NEURONS),
            nn.LeakyReLU(),
            nn.Linear(MLP_HIDDEN_LAYER_NEURONS, MLP_HIDDEN_LAYER_NEURONS),
            nn.LeakyReLU(),
            nn.Linear(MLP_HIDDEN_LAYER_NEURONS, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)

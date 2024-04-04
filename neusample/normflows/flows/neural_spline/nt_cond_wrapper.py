import torch
from torch import nn
import numpy as np

from ..base import Flow
from .coupling import PiecewiseRationalQuadraticCoupling
from .autoregressive import MaskedPiecewiseRationalQuadraticAutoregressive
from ...nets.resnet import ResidualNet
from ...utils.masks import create_alternating_binary_mask
from ...utils.nn import PeriodicFeatures
from ...utils.splines import DEFAULT_MIN_DERIVATIVE


from torch.nn import functional as F

from ... import utils
# from ..affine.autoregressive import Autoregressive
from normflows.nets import made as made_module
from normflows.utils import splines
from normflows.utils.nn import PeriodicFeatures


class Autoregressive(Flow):
    """Transforms each input variable with an invertible elementwise transformation.

    The parameters of each invertible elementwise transformation can be functions of previous input
    variables, but they must not depend on the current or any following input variables.

    NOTE: Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.
    """

    def __init__(self, autoregressive_net):
        super(Autoregressive, self).__init__()
        self.autoregressive_net = autoregressive_net

    def forward(self, inputs, context=None):
        autoregressive_params = self.autoregressive_net(inputs, context)
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        num_inputs = np.prod(inputs.shape[1:]) # 2
        outputs = torch.zeros_like(inputs) 
        logabsdet = None
        for _ in range(num_inputs):
            autoregressive_params = self.autoregressive_net(outputs, context)
            outputs, logabsdet = self._elementwise_inverse(
                inputs, autoregressive_params
            )###??TODO check
        return outputs, logabsdet

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, inputs, autoregressive_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, inputs, autoregressive_params):
        raise NotImplementedError()

class NTCondMaskedPiecewiseRationalQuadraticAutoregressive(Autoregressive):
    def __init__(
        self,
        features, #2
        hidden_features, #32
        context_features=None,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        permute_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        init_identity=True,
        min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=splines.DEFAULT_MIN_DERIVATIVE,
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        assert(context_features)
        if isinstance(self.tails, list) or isinstance(self.tails, tuple):
            ind_circ = []
            for i in range(features):
                if self.tails[i] == "circular":
                    ind_circ += [i]
            if torch.is_tensor(tail_bound):
                scale_pf = np.pi / tail_bound[ind_circ]
            else:
                scale_pf = np.pi / tail_bound
            preprocessing = PeriodicFeatures(features, ind_circ, scale_pf)
        else:
            preprocessing = None

        autoregressive_net = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            permute_mask=permute_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
            preprocessing=preprocessing,
        )

        if init_identity:
            torch.nn.init.constant_(autoregressive_net.final_layer.weight, 0.0)
            torch.nn.init.constant_(
                autoregressive_net.final_layer.bias,
                np.log(np.exp(1 - min_derivative) - 1),
            )

        super().__init__(autoregressive_net)

        if torch.is_tensor(tail_bound):
            self.register_buffer("tail_bound", tail_bound)
        else:
            self.tail_bound = tail_bound

    def _output_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails == "circular":
            return self.num_bins * 3
        else:
            return self.num_bins * 3 + 1

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.autoregressive_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, utils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)


class NTCondAutoregressiveRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see https://github.com/bayesiains/nsf
    """

    def __init__(
        self,
        num_input_channels,#2
        num_blocks, #2 hidden layers 
        num_hidden_channels, #32 hiddle_units
        context_features,
        num_bins=8,
        tail_bound=3,
        activation=nn.ReLU,
        dropout_probability=0.0,
        permute_mask=False,
        init_identity=True,
    ):
        """
        Constructor
        :param num_input_channels: Flow dimension
        :type num_input_channels: Int
        :param num_blocks: Number of residual blocks of the parameter NN
        :type num_blocks: Int
        :param num_hidden_channels: Number of hidden units of the NN
        :type num_hidden_channels: Int
        :param num_bins: Number of bins
        :type num_bins: Int
        :param tail_bound: Bound of the spline tails
        :type tail_bound: Int
        :param activation: Activation function
        :type activation: torch module
        :param dropout_probability: Dropout probability of the NN
        :type dropout_probability: Float
        :param permute_mask: Flag, permutes the mask of the NN
        :type permute_mask: Boolean
        :param init_identity: Flag, initialize transform as identity
        :type init_identity: Boolean
        """
        super().__init__()
        assert(context_features)

        self.mprqat = NTCondMaskedPiecewiseRationalQuadraticAutoregressive(
            features=num_input_channels, #2
            hidden_features=num_hidden_channels, #32
            context_features=context_features, 
            num_bins=num_bins, #8
            tails="linear",
            tail_bound=tail_bound,
            num_blocks=num_blocks, #2 #hidden layers
            use_residual_blocks=True,
            random_mask=False,
            permute_mask=permute_mask,
            activation=activation(),
            dropout_probability=dropout_probability,
            use_batch_norm=False,
            init_identity=init_identity,
        )

    def forward(self, z, cond_vector):
        z, log_det = self.mprqat.inverse(z, cond_vector)
        return z, log_det.view(-1)

    def inverse(self, z, cond_vector):
        z, log_det = self.mprqat(z, cond_vector)
        return z, log_det.view(-1)

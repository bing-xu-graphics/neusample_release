import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

from ..base import Flow
from .coupling import PiecewiseCoupling
from .autoregressive import MaskedPiecewiseRationalQuadraticAutoregressive
from ...nets.resnet import ResidualNet
from ...utils.masks import create_alternating_binary_mask
from ...utils.nn import PeriodicFeatures
from ...utils.splines import DEFAULT_MIN_DERIVATIVE
import warnings

from ... import utils
from ...utils import splines
from ...utils import splines_nis


class PiecewiseLinearCDF(Flow):
    def __init__(
        self,
        shape,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        identity_init=True,
        bin_width = splines.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.DEFAULT_MIN_BIN_HEIGHT,
    ):
        super().__init__()

        
        self.min_bin_height = min_bin_height

        self.bin_width = bin_width
        self.unnormalized_widths = nn.Parameter(torch.full((*shape, num_bins), bin_width))


        if torch.is_tensor(tail_bound):
            self.register_buffer("tail_bound", tail_bound)
        else:
            self.tail_bound = tail_bound
        self.tails = tails

        if self.tails == "linear":
            num_derivatives = num_bins - 1
        elif self.tails == "circular":
            num_derivatives = num_bins
        else:
            num_derivatives = num_bins #TODO careful

        if identity_init:
            self.unnormalized_derivatives = nn.Parameter(torch.zeros(*shape, num_bins))
        else:
            self.unnormalized_derivatives = nn.Parameter(
                torch.rand(*shape, num_derivatives)
            )

    @staticmethod
    def _share_across_batch(params, batch_size):
        return params[None, ...].expand(batch_size, *params.shape)

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths = self._share_across_batch(
            self.unnormalized_widths, batch_size
        )
        unnormalized_derivatives = self._share_across_batch(
            self.unnormalized_derivatives, batch_size
        )

        # if self.tails is None:
        spline_fn = splines_nis.linear_spline
        spline_kwargs = {}
        # else:
        #     spline_fn = splines.unconstrained_rational_quadratic_spline
        #     spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_height=self.min_bin_height,
            **spline_kwargs
        )

        return outputs, utils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)


class PiecewiseLinearCoupling(PiecewiseCoupling):
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        apply_unconditional_transform=False,
        min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
        min_derivative=splines.DEFAULT_MIN_DERIVATIVE,
    ):
        
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        # self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        assert(self.num_bins>1)
        self.bin_width = 1/self.num_bins

        # Split tails parameter if needed
        features_vector = torch.arange(len(mask))
        identity_features = features_vector.masked_select(mask <= 0)
        transform_features = features_vector.masked_select(mask > 0)
        if isinstance(tails, list) or isinstance(tails, tuple):
            self.tails = [tails[i] for i in transform_features]
            tails_ = [tails[i] for i in identity_features]
        else:
            self.tails = tails
            tails_ = tails

        if torch.is_tensor(tail_bound):
            tail_bound_ = tail_bound[identity_features]
        else:
            self.tail_bound = tail_bound
            tail_bound_ = tail_bound

        if apply_unconditional_transform:
            unconditional_transform = lambda features: PiecewiseLinearCDF(
                shape=[features],
                num_bins=num_bins,
                tails=tails_,
                tail_bound=tail_bound_,
                min_bin_width=min_bin_width,
                min_derivative=min_derivative,
            )
        else:
            unconditional_transform = None

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

        if torch.is_tensor(tail_bound):
            self.register_buffer("tail_bound", tail_bound[transform_features]) 


    def _transform_dim_multiplier(self):
        # if self.tails == "linear":
        #     return self.num_bins * 3 - 1
        # elif self.tails == "circular":
        #     return self.num_bins * 3
        # else:
        return self.num_bins

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_derivatives = transform_params[..., : self.num_bins]
        unnormalized_widths = torch.full(unnormalized_derivatives.shape, 1/unnormalized_derivatives.shape[-1]).cuda()

        # if hasattr(self.transform_net, "hidden_features"):
        #     unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
        #     unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)
        # elif hasattr(self.transform_net, "hidden_channels"):
        #     unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
        #     unnormalized_heights /= np.sqrt(self.transform_net.hidden_channels)
        # else:
        #     warnings.warn(
        #         "Inputs to the softmax are not scaled down: initialization might be bad."
        #     )

        # if self.tails is None:
        spline_fn = splines_nis.linear_spline
        spline_kwargs = {}
        # else:
        #     spline_fn = splines.unconstrained_rational_quadratic_spline
        #     spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            # unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            # min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )
        



class PiecewiseQuadraticCDF(Flow):
    def __init__(
        self,
        shape,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        identity_init=True,
        min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
        min_derivative=splines.DEFAULT_MIN_DERIVATIVE,
    ):
        super().__init__()

        self.min_bin_width = min_bin_width
        self.min_derivative = min_derivative

        if torch.is_tensor(tail_bound):
            self.register_buffer("tail_bound", tail_bound)
        else:
            self.tail_bound = tail_bound
        self.tails = tails

        if self.tails == "linear":
            num_derivatives = num_bins - 1
        elif self.tails == "circular":
            num_derivatives = num_bins
        else:
            num_derivatives = num_bins + 1

        if identity_init:
            self.unnormalized_widths = nn.Parameter(torch.zeros(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.zeros(*shape, num_bins))

            constant = np.log(np.exp(1 - min_derivative) - 1)
            self.unnormalized_derivatives = nn.Parameter(
                constant * torch.ones(*shape, num_derivatives)
            )
        else:
            self.unnormalized_widths = nn.Parameter(torch.rand(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.rand(*shape, num_bins))

            self.unnormalized_derivatives = nn.Parameter(
                torch.rand(*shape, num_derivatives)
            )

    @staticmethod
    def _share_across_batch(params, batch_size):
        return params[None, ...].expand(batch_size, *params.shape)

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths = self._share_across_batch(
            self.unnormalized_widths, batch_size
        )
        unnormalized_heights = self._share_across_batch(
            self.unnormalized_heights, batch_size
        )
        unnormalized_derivatives = self._share_across_batch(
            self.unnormalized_derivatives, batch_size
        )

        # if self.tails is None:
        spline_fn = splines_nis.quadratic_spline
        spline_kwargs = {}
        # else:
            # spline_fn = splines.unconstrained_rational_quadratic_spline
            # spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, utils.sum_except_batch(logabsdet)

    def _transform_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins  - 1
        elif self.tails == "circular":
            return self.num_bins
        else:
            return self.num_bins + 1

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)


class PiecewiseQuadraticCoupling(PiecewiseCoupling):
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        apply_unconditional_transform=False,
        img_shape=None,
        min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
        min_derivative=splines.DEFAULT_MIN_DERIVATIVE,
        extra_feature_dim = None
    ):

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_derivative = min_derivative

        # Split tails parameter if needed
        features_vector = torch.arange(len(mask))
        identity_features = features_vector.masked_select(mask <= 0)
        transform_features = features_vector.masked_select(mask > 0)
        if isinstance(tails, list) or isinstance(tails, tuple):
            self.tails = [tails[i] for i in transform_features]
            tails_ = [tails[i] for i in identity_features]
        else:
            self.tails = tails
            tails_ = tails

        if torch.is_tensor(tail_bound):
            tail_bound_ = tail_bound[identity_features]
        else:
            self.tail_bound = tail_bound
            tail_bound_ = tail_bound

        if apply_unconditional_transform:
            unconditional_transform = lambda features: PiecewiseQuadraticCDF(
                shape=[features] + (img_shape if img_shape else []),
                num_bins=num_bins,
                tails=tails_,
                tail_bound=tail_bound_,
                min_bin_width=min_bin_width,
                min_derivative=min_derivative,
            )
        else:
            unconditional_transform = None

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
            extra_input_features=extra_feature_dim
        )

        if torch.is_tensor(tail_bound):
            self.register_buffer("tail_bound", tail_bound[transform_features])

    def _transform_dim_multiplier(self):
        # if self.tails == "linear":
        #     return self.num_bins * 3 - 1
        # elif self.tails == "circular":
        #     return self.num_bins * 3
        # else:
        return self.num_bins * 2 + 1

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_derivatives = transform_params[..., self.num_bins : ]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn(
                "Inputs to the softmax are not scaled down: initialization might be bad."
            )


        spline_fn = splines_nis.quadratic_spline
        spline_kwargs = {}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )


class CoupledLinearSpline(Flow):
    """
    Implementation for NIS. piecewise-linear
    @bingx modified from neural spline flows
    """

    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        num_bins=8,
        tails="linear",
        tail_bound=3.0,
        activation=nn.ReLU,
        dropout_probability=0.0,
        reverse_mask=False,
        context_features = None,
        encode = None,
    ):
        super().__init__()
        self.encode = encode

        def transform_net_create_fn(in_features, out_features):
            return ResidualNet(
                in_features=in_features,
                out_features=out_features,
                context_features=context_features,
                hidden_features=num_hidden_channels,
                num_blocks=num_blocks,
                activation=activation(),
                dropout_probability=dropout_probability,
                use_batch_norm=False,
            )

        self.pw_ln_coupling = PiecewiseLinearCoupling(
            mask=create_alternating_binary_mask(num_input_channels, even=reverse_mask),
            transform_net_create_fn=transform_net_create_fn,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            # Setting True corresponds to equations (4), (5), (6) in the NSF paper:
            apply_unconditional_transform=False, #TODO set to True and finish
        )

    def forward(self, z):
        z, log_det = self.pw_ln_coupling.inverse(z, self.encode)
        return z, log_det.view(-1)

    def inverse(self, z):
        z, log_det = self.pw_ln_coupling(z, self.encode)
        return z, log_det.view(-1)

class CoupledQuadraticSpline(Flow):
    """
    Implementation for a modified version of NIS. piecewise-quadratic
    @bingx modified from neural spline flows
    """

    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        num_bins=8,
        tails="linear",
        tail_bound=3.0,
        activation=nn.ReLU,
        dropout_probability=0.0,
        reverse_mask=False,
        context_features = None,
        encode = False,
    ):
        super().__init__()
        self.encode = encode

        def transform_net_create_fn(in_features, out_features):
            return ResidualNet(
                in_features=in_features,
                out_features=out_features,
                context_features=context_features,
                hidden_features=num_hidden_channels,
                num_blocks=num_blocks,
                activation=activation(),
                dropout_probability=dropout_probability,
                use_batch_norm=False,
            )

        self.pw_qua_coupling = PiecewiseQuadraticCoupling(
            mask=create_alternating_binary_mask(num_input_channels, even=reverse_mask),
            transform_net_create_fn=transform_net_create_fn,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            # Setting True corresponds to equations (4), (5), (6) in the NSF paper:
            apply_unconditional_transform=False, #TODO set to True and finish
            extra_feature_dim = self.encode*2 if self.encode else 0   #only for uv
        )


    def forward(self, z, context = None):
        z, log_det = self.pw_qua_coupling.inverse(z, context, self.encode)
        return z, log_det.view(-1)

    def inverse(self, z, context = None):
        z, log_det = self.pw_qua_coupling(z, context,self.encode)
        return z, log_det.view(-1)


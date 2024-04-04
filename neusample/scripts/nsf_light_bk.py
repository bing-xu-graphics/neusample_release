
import torch
import torch.nn as nn
from normflows import mlp
from normflows import utils
from normflows.utils import encoding
from torch.nn import functional as F, init
from normflows import mlp

from normflows.flows import mixing


DISTRIBUTION_DIM = 2

class Flow(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        raise NotImplementedError("Forward pass has not been implemented.")

    def inverse(self, z):
        raise NotImplementedError("This flow has no algebraic inverse.")


class Coupling(Flow):
    """A base class for coupling layers. Supports 2D inputs (NxD), as well as 4D inputs for
    images (NxCxHxW). For images the splitting is done on the channel dimension, using the
    provided 1D mask."""

    def __init__(self, mask, transform_net_create_fn,unconditional_transform=None,  extra_input_features = None):       
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError("Mask must be a 1-dim tensor.")
        if mask.numel() <= 0:
            raise ValueError("Mask can't be empty.")

        super().__init__()
        self.features = len(mask)
        features_vector = torch.arange(self.features)

        self.register_buffer(
            "identity_features", features_vector.masked_select(mask <= 0)
        )
        self.register_buffer(
            "transform_features", features_vector.masked_select(mask > 0)
        )

        assert self.num_identity_features + self.num_transform_features == self.features

        input_feat_dim = extra_input_features+self.num_identity_features if extra_input_features else self.num_identity_features
        self.transform_net = transform_net_create_fn(
            input_feat_dim,
            self.num_transform_features * self._transform_dim_multiplier(),
        )

        
    @property
    def num_identity_features(self):
        return len(self.identity_features)

    @property
    def num_transform_features(self):
        return len(self.transform_features)

    def forward(self, inputs, context=None, encode = None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Inputs must be a 2D or a 4D tensor.")

        if inputs.shape[1] != self.features:
            raise ValueError(
                "Expected features = {}, got {}.".format(self.features, inputs.shape[1])
            )

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        if(encode):
            mlp_input = torch.cat((encoding.positional_encoding_1(identity_split, encode, log_sampling=False), context), dim=1)
            transform_params = self.transform_net(mlp_input)
        else:
            assert(False)
            mlp_input = torch.cat((identity_split, context), dim=1)
            transform_params = self.transform_net(mlp_input)
        transform_split, logabsdet = self._coupling_transform_forward(
            inputs=transform_split, transform_params=transform_params
        )

        # if self.unconditional_transform is not None:
        #     identity_split, logabsdet_identity = self.unconditional_transform(
        #         identity_split, context
        #     )
        #     logabsdet += logabsdet_identity

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features, ...] = identity_split
        outputs[:, self.transform_features, ...] = transform_split

        return outputs, logabsdet

    def inverse(self, inputs, context=None,encode = None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Inputs must be a 2D or a 4D tensor.")

        if inputs.shape[1] != self.features:
            raise ValueError(
                "Expected features = {}, got {}.".format(self.features, inputs.shape[1])
            )

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        logabsdet = 0.0
        
        if(encode):
            # print(identity_split.shape)
            # encoded_half_input = encoding.positional_encoding_0(identity_split, encode)
            mlp_input = torch.cat((encoding.positional_encoding_1(identity_split, encode, log_sampling=False), context), dim=1)
            # print(encoded_half_input.shape)
            transform_params = self.transform_net(mlp_input, context)
        else:
            assert(False)
            transform_params = self.transform_net(identity_split, context)

        transform_split, logabsdet_split = self._coupling_transform_inverse(
            inputs=transform_split, transform_params=transform_params
        )
        logabsdet += logabsdet_split

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features] = identity_split
        outputs[:, self.transform_features] = transform_split

        return outputs, logabsdet

    def _transform_dim_multiplier(self):
        """Number of features to output for each transform dimension."""
        raise NotImplementedError()

    def _coupling_transform_forward(self, inputs, transform_params):
        """Forward pass of the coupling transform."""
        raise NotImplementedError()

    def _coupling_transform_inverse(self, inputs, transform_params):
        """Inverse of the coupling transform."""
        raise NotImplementedError()


def create_alternating_binary_mask(features, even=True):
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


def _get_input_degrees(in_features):
    """Returns the degrees an input to MADE should have."""
    return torch.arange(1, in_features + 1)


def tile(x, n):
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_

class MaskedLinear(nn.Linear):
    """A linear module with a masked weight matrix."""

    def __init__(
        self,
        in_degrees,
        out_features,
        autoregressive_features,
        random_mask,
        is_output,
        bias=True,
        out_degrees_=None,
    ):
        super().__init__(
            in_features=len(in_degrees), out_features=out_features, bias=bias
        )
        mask, degrees = self._get_mask_and_degrees(
            in_degrees=in_degrees,
            out_features=out_features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=is_output,
            out_degrees_=out_degrees_,
        )
        self.register_buffer("mask", mask)
        self.register_buffer("degrees", degrees)

    @classmethod
    def _get_mask_and_degrees(
        cls,
        in_degrees,
        out_features,
        autoregressive_features,
        random_mask,
        is_output,
        out_degrees_=None,
    ):
        if is_output:
            if out_degrees_ is None:
                out_degrees_ = _get_input_degrees(autoregressive_features)
            out_degrees = tile(out_degrees_, out_features // autoregressive_features)
            mask = (out_degrees[..., None] > in_degrees).float()

        else:
            if random_mask:
                min_in_degree = torch.min(in_degrees).item()
                min_in_degree = min(min_in_degree, autoregressive_features - 1)
                out_degrees = torch.randint(
                    low=min_in_degree,
                    high=autoregressive_features,
                    size=[out_features],
                    dtype=torch.long,
                )
            else:
                max_ = max(1, autoregressive_features - 1)
                min_ = min(1, autoregressive_features - 1)
                out_degrees = torch.arange(out_features) % max_ + min_
            mask = (out_degrees[..., None] >= in_degrees).float()

        return mask, out_degrees

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedFeedforwardBlock(nn.Module):
    """A feedforward block based on a masked linear module.
    NOTE: In this implementation, the number of output features is taken to be equal to
    the number of input features.
    """

    def __init__(
        self,
        in_degrees,
        autoregressive_features,
        context_features=None,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        features = len(in_degrees)

        # Batch norm.
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features, eps=1e-3)
        else:
            self.batch_norm = None

        if context_features is not None:
            raise NotImplementedError()

        # Masked linear.
        self.linear = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=False,
        )
        self.degrees = self.linear.degrees

        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs, context=None):
        if context is not None:
            raise NotImplementedError()

        if self.batch_norm:
            outputs = self.batch_norm(inputs)
        else:
            outputs = inputs
        outputs = self.linear(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class MaskedResidualBlock(nn.Module):
    """A residual block containing masked linear modules."""

    def __init__(
        self,
        in_degrees,
        autoregressive_features,
        context_features=None,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        if random_mask:
            raise ValueError("Masked residual block can't be used with random masks.")
        super().__init__()
        features = len(in_degrees)

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)            
            # self.context_layer = MLP([context_features+features, 64, 64, features], leaky=0.01, init_zeros=True)

        # Batch norm.
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )

        # Masked linear.
        linear_0 = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        linear_1 = MaskedLinear(
            in_degrees=linear_0.degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        self.linear_layers = nn.ModuleList([linear_0, linear_1])
        self.degrees = linear_1.degrees
        if torch.all(self.degrees >= in_degrees).item() != 1:
            raise RuntimeError(
                "In a masked residual block, the output degrees can't be"
                " less than the corresponding input degrees."
            )

        # Activation and dropout
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

        # Initialization.
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, a=-1e-3, b=1e-3)
            init.uniform_(self.linear_layers[-1].bias, a=-1e-3, b=1e-3)

    def forward(self, inputs, context=None):
        assert(context!=None)
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
            # temps = self.context_layer(torch.cat((temps,context), dim=1))
        return inputs + temps


class ConditionalMADE(nn.Module):
    """Implementation of MADE.
    It can use either feedforward blocks or residual blocks (default is residual).
    Optionally, it can use batch norm or dropout within blocks (default is no).
    """

    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        output_multiplier=1,
        use_residual_blocks=False,
        random_mask=False,
        permute_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random masks.")
        super().__init__()

        # Initial layer.
        input_degrees_ = _get_input_degrees(features) # 1,2
        if permute_mask:
            input_degrees_ = input_degrees_[torch.randperm(features)]
        self.initial_layer = MaskedLinear(
            in_degrees=input_degrees_,
            out_features=hidden_features,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=False,
        )

        if context_features is not None:
            # self.context_layer = nn.Linear(context_features, hidden_features)
            self.context_layer = mlp.MLP([context_features, 32, hidden_features])


        # Residual blocks.
        blocks = []
        if use_residual_blocks:
            block_constructor = MaskedResidualBlock
        else:
            block_constructor = MaskedFeedforwardBlock
        prev_out_degrees = self.initial_layer.degrees
        for _ in range(num_blocks):
            blocks.append(
                block_constructor(
                    in_degrees=prev_out_degrees,
                    autoregressive_features=features,
                    context_features=context_features,
                    random_mask=random_mask,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)

        # Final layer.
        self.final_layer = MaskedLinear(
            in_degrees=prev_out_degrees,
            out_features=features * output_multiplier,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=True,
            out_degrees_=input_degrees_,
        )

    def forward(self, inputs, context=None):
        outputs = inputs
        outputs = self.initial_layer(outputs)
        if context is not None:
            outputs += self.context_layer(context)
        for block in self.blocks:
            outputs = block(outputs, context)
        outputs = self.final_layer(outputs)
        return outputs


class MaskedPiecewiseRationalQuadraticAutoregressive(Flow):

    def __init__(
        self,
        features, #2
        hidden_features,
        context_features=None,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        permute_mask=False,
        activation = F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        init_identity=True
    ):
        self.num_bins = num_bins
        self.min_bin_width = utils.splines.DEFAULT_MIN_BIN_WIDTH
        self.min_bin_height = utils.splines.DEFAULT_MIN_BIN_HEIGHT
        self.min_derivative = utils.splines.DEFAULT_MIN_DERIVATIVE
        self.tails = tails
        assert(context_features)

        autoregressive_net = ConditionalMADE(
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
        )

        self.autoregressive_net = autoregressive_net
      

        if init_identity:
            torch.nn.init.constant_(autoregressive_net.final_layer.weight, 0.0)
            torch.nn.init.constant_(
                autoregressive_net.final_layer.bias,
                np.log(np.exp(1 - self.min_derivative) - 1),
            )

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
            spline_fn = utils.splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = utils.splines.unconstrained_rational_quadratic_spline
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
                inputs, autoregressive_params)###??TODO check
        return outputs, logabsdet

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)


class AutoregressiveRQSpline(Flow):
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
        num_bins=32,
        tail_bound=3,
        activation=nn.ReLU,
        dropout_probability=0.0,
        permute_mask=False,
        init_identity=True,
    ):
        
        super().__init__()
        assert(context_features)

        self.mprqat = MaskedPiecewiseRationalQuadraticAutoregressive(
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

    def forward(self, z, cond_vector): # sample
        z, log_det = self.mprqat.inverse(z, cond_vector)
        return z, log_det.view(-1)

    def inverse(self, z, cond_vector): #calculate pdf
        z, log_det = self.mprqat(z, cond_vector)
        return z, log_det.view(-1)


class PiecewiseRQCoupling(Coupling):
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        extra_feature_dim = None
    ):

        self.num_bins = num_bins
        self.min_bin_width = utils.splines.DEFAULT_MIN_BIN_WIDTH
        self.min_bin_height = utils.splines.DEFAULT_MIN_BIN_HEIGHT
        self.min_derivative = utils.splines.DEFAULT_MIN_DERIVATIVE
        self.tails = tails
        assert(extra_feature_dim)

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
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails == "circular":
            return self.num_bins * 3
        else:
            return self.num_bins * 3 + 1

    def _coupling_transform_forward(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=False)

    def _coupling_transform_inverse(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=True)

    def _coupling_transform(self, inputs, transform_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = transform_params.view(
            batch_size, features, self._transform_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]
        
        if self.tails is None:
            spline_fn = utils.splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = utils.splines.unconstrained_rational_quadratic_spline
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


class CoupledRQSpline(Flow):
    def __init__(
        self,
        num_input_channels,
        num_bins=8,
        tails="linear",
        tail_bound=1.0,
        activation=nn.ReLU,
        dropout_probability=0.0,
        reverse_mask=False,
        context_features = None,
        encode = False,
        transform_net_create_fn = None,
    ):
        super().__init__()
        assert(context_features)
        self.encode = encode

        # def transform_net_create_fn(in_features, out_features):
        #     # return mlp.MLP([in_features + context_features, 32, 32, out_features], leaky=0.01)# 2layer for mixture
        #     return mlp.MLP([in_features + context_features, 32, 32, out_features], leaky=0.01)

        self.pw_rq_coupling = PiecewiseRQCoupling(
            mask=create_alternating_binary_mask(num_input_channels, even=reverse_mask),
            transform_net_create_fn=transform_net_create_fn,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            # Setting True corresponds to equations (4), (5), (6) in the NSF paper:
            extra_feature_dim = self.encode*2 if self.encode else 0   #only for uv
        )


    def forward(self, z, context = None):
        z, log_det = self.pw_rq_coupling.inverse(z, context, self.encode)
        return z, log_det.view(-1)

    def inverse(self, z, context = None):
        z, log_det = self.pw_rq_coupling(z, context,self.encode)
        return z, log_det.view(-1)

class Permute(Flow):
    """
    Permutation features along the channel dimension
    """

    def __init__(self, num_channels, mode="shuffle"):
        """
        Constructor
        :param num_channel: Number of channels
        :param mode: Mode of permuting features, can be shuffle for
        random permutation or swap for interchanging upper and lower part
        """
        super().__init__()
        self.mode = mode
        self.num_channels = num_channels
        if self.mode == "shuffle":
            perm = torch.randperm(self.num_channels)
            inv_perm = torch.empty_like(perm).scatter_(
                dim=0, index=perm, src=torch.arange(self.num_channels)
            )
            self.register_buffer("perm", perm)
            self.register_buffer("inv_perm", inv_perm)

    def forward(self, z, ph0=None, ph1=None):
        if self.mode == "shuffle":
            z = z[:, self.perm, ...]
        elif self.mode == "swap":
            z1 = z[:, : self.num_channels // 2, ...]
            z2 = z[:, self.num_channels // 2 :, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError("The mode " + self.mode + " is not implemented.")
        log_det = 0
        return z, log_det

    def inverse(self, z, ph0=None, ph1=None): #add two placeholders
        if self.mode == "shuffle":
            z = z[:, self.inv_perm, ...]
        elif self.mode == "swap":
            z1 = z[:, : (self.num_channels + 1) // 2, ...]
            z2 = z[:, (self.num_channels + 1) // 2 :, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError("The mode " + self.mode + " is not implemented.")
        log_det = 0
        return z, log_det
DEVICE_ = torch.device(0)
DTYPE_ = torch.double
from weighted_sum_dist import *

class NSFLight_CL(nn.Module):
    def __init__(self, q0, feat_dim, num_layers=2,num_bins = 8, mlp_size = 2):  #let's have just num_layers = 1

        super().__init__()
        self.q0 = q0
        flows = []
        def transform_net_create_fn(in_features, out_features):
            if mlp_size == 2:
                return mlp.MLP([in_features + feat_dim, 32, 32, out_features], leaky=0.01)# 2layer for mixture
            elif mlp_size == 3:
                return mlp.MLP([in_features + feat_dim, 32, 32, 32, out_features], leaky=0.01)
            else:
                assert(False)

        for i in range(num_layers):
            flows.append(CoupledRQSpline(\
                DISTRIBUTION_DIM, 
                num_bins=num_bins,
                tails="linear",
                tail_bound=1.0,
                activation=torch.nn.ReLU,
                dropout_probability=0.0,
                reverse_mask=False,
                context_features=feat_dim,
                encode = 4,
                transform_net_create_fn = transform_net_create_fn,

                )) 
            # flows += [mixing.LULinearPermute(DISTRIBUTION_DIM)]
            flows.append(Permute(DISTRIBUTION_DIM, mode='swap'))

        self.flows = nn.ModuleList(flows)
    
    def forward(self, cond_vec, num_samples=1):
        return self.sample(cond_vec, num_samples)
    
    def sample(self, cond_vec, num_samples=1):
        z, log_q = self.q0(cond_vec, num_samples)
        if(isinstance(self.q0, GMMWeightedCond)): 
            z = (z + 1.0) * 0.5
        for flow in self.flows:
            z, log_det = flow(z, cond_vec)
            log_q -= log_det
        # z = z * 2.0 - 1.0 ##change to [-1,1]
        return z, log_q

    def log_prob(self, x, cond_vec):
        # x = (x + 1.0 )*0.5 ##change to [0,1]
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z,cond_vec)
            log_q += log_det
        log_q += self.q0.log_prob(z, cond_vec)
        return log_q


# BNN layers partially borrowed from BayesianTorch: https://github.com/IntelLabs/bayesian-torch
# ======================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_variational_layer import _BaseVariationalLayer, get_kernel_size
from torch.quantization.qconfig import QConfig
from torch.quantization.observer import HistogramObserver, PerChannelMinMaxObserver, MinMaxObserver

from collections import namedtuple
mean_var_tuple = namedtuple('mean_var_tuple', ['mean', 'var'])  # Define a named tuple


class LinearFlipout(_BaseVariationalLayer):
    """
    Alternative implementation of Bayesian Linear layer.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        super(LinearFlipout, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_sigma',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features, in_features),
                             persistent=False)

        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer('prior_bias_mu', torch.Tensor(out_features), persistent=False)
            self.register_buffer('prior_bias_sigma', torch.Tensor(out_features), persistent=False)
            self.register_buffer('eps_bias', torch.Tensor(out_features), persistent=False)

        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            self.register_buffer('eps_bias', None, persistent=False)

        self.init_parameters()

    def init_parameters(self):
        # init prior mu
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        # init weight and base perturbation weights
        self.weight_mu.data.normal_(mean=self.posterior_mu_init, std=0.1)
        self.weight_rho.data.normal_(mean=self.posterior_rho_init, std=0.1)

        if self.bias_mu is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.bias_mu.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.bias_rho.data.normal_(mean=self.posterior_rho_init, std=0.1)

    def kl_loss(self):
        weight_sigma = F.softplus(self.weight_rho)
        kl = self.kl_div(self.weight_mu, weight_sigma, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias_mu is not None:
            bias_sigma = F.softplus(self.bias_rho)
            kl += self.kl_div(self.bias_mu, bias_sigma, self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, x, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        # deterministic path
        outputs = F.linear(x, self.weight_mu, self.bias_mu)

        # stochastic path
        # eps ~ N(0,1) for weights (one per weight element; shared across batch)
        eps_weight = self.eps_weight.data.normal_()
        weight_sigma = F.softplus(self.weight_rho)  # weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        delta_weight = weight_sigma * eps_weight

        sign_input = x.clone().uniform_(-1, 1).sign()  # (B*, in)
        sign_output = outputs.clone().uniform_(-1, 1).sign()  # (B*, out)

        delta_bias = None
        if self.bias_rho is not None:
            bias_sigma = F.softplus(self.bias_rho)  # bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            delta_bias = bias_sigma * self.eps_bias.data.normal_()  # (out,)

        perturbed_outputs = F.linear(x * sign_input, delta_weight, delta_bias)  # (B*, out)
        perturbed_outputs = perturbed_outputs * sign_output  # (B*, out)
        out = outputs + perturbed_outputs  # returning outputs + perturbations

        # get kl divergence
        if return_kl:
            kl = self.kl_div(self.weight_mu, weight_sigma, self.prior_weight_mu, self.prior_weight_sigma)
            if self.bias_rho is not None:
                kl = kl + self.kl_div(self.bias_mu, bias_sigma, self.prior_bias_mu, self.prior_bias_sigma)
            return out, kl
        return out


class SparseLinearFlipout(_BaseVariationalLayer):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True,):
        super(SparseLinearFlipout, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('prior_weight_mu', torch.Tensor(out_features, in_features), persistent=False)
        self.register_buffer('prior_weight_sigma', torch.Tensor(out_features, in_features), persistent=False)

        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer('prior_bias_mu', torch.Tensor(out_features), persistent=False)
            self.register_buffer('prior_bias_sigma', torch.Tensor(out_features), persistent=False)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()

    def init_parameters(self):
        # init prior mu
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        # init weight and base perturbation weights
        self.weight_mu.data.normal_(mean=self.posterior_mu_init, std=0.1)
        self.weight_rho.data.normal_(mean=self.posterior_rho_init, std=0.1)

        if self.bias_mu is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.bias_mu.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.bias_rho.data.normal_(mean=self.posterior_rho_init, std=0.1)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.weight_rho))
        kl = self.kl_div(self.weight_mu, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias_mu is not None:
            sigma_bias = torch.log1p(torch.exp(self.bias_rho))
            kl += self.kl_div(self.bias_mu, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)
        return kl
    
    def bnn_linear(self, x, weight_mu, weight_sigma, bias_sigma=None):
        # deterministic path
        output = torch.matmul(weight_mu, x.unsqueeze(-1)).squeeze(-1)
        if self.bias_mu is not None:
            output += self.bias_mu
        
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = output.clone().uniform_(-1, 1).sign()
        x_tmp = x * sign_input

        eps_weight = weight_sigma.new_empty(weight_sigma.shape[-2:]).normal_()
        delta_weight = weight_sigma * eps_weight
        perturbed_output = torch.matmul(delta_weight, x_tmp.unsqueeze(-1)).squeeze(-1)
        perturbed_output = perturbed_output * sign_output

        if bias_sigma is not None:
            eps_b = torch.randn_like(self.bias_mu)
            delta_bias = bias_sigma * eps_b
            perturbed_output = perturbed_output + sign_output * delta_bias

        res = output + perturbed_output
        return res
    
    def sparse_mu_sigma(self, x, idx=None, return_kl=True, sparse=True):
        bsize, d_in = x.shape[0], x.shape[1]
        d_out = self.out_features

        weight_sigma = F.softplus(self.weight_rho)
        kl = None
        if return_kl:
            kl = self.kl_div(self.weight_mu, weight_sigma, self.prior_weight_mu, self.prior_weight_sigma)

        bias_sigma = None
        if self.bias_mu is not None:
            bias_sigma = F.softplus(self.bias_rho)
            if return_kl:
                kl = kl + self.kl_div(self.bias_mu, bias_sigma, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        if sparse:
            if idx is None:
                raise ValueError("idx must be provided for sparse forward pass.")
            idx = idx.unsqueeze(1).expand(-1, d_out, -1, -1)  # (bsize, fsize, L) --> (bsize, osize, fsize, L)
            weight_mu = self.weight_mu.unsqueeze(0).expand(bsize, -1, -1).view(bsize, d_out, d_in, -1)
            weight_sigma = weight_sigma.unsqueeze(0).expand(bsize, -1, -1).view(bsize, d_out, d_in, -1)

            weight_mu_nonzero = torch.gather(weight_mu, dim=-1, index=idx).view(bsize, d_out, -1)  # (bsize, osize, fsize*L)
            weight_sigma_nonzero = torch.gather(weight_sigma, dim=-1, index=idx).view(bsize, d_out, -1)  # (bsize, osize, fsize*L)
            return weight_mu_nonzero, weight_sigma_nonzero, bias_sigma, kl if return_kl else None

        weight_mu = self.weight_mu.unsqueeze(0).expand(bsize, -1, -1) #(bsize, osize, fsize)
        weight_sigma = weight_sigma.unsqueeze(0).expand(bsize, -1, -1) #(bsize, osize, fsize)
        return weight_mu, weight_sigma, bias_sigma, kl if return_kl else None
    
    def forward(self, x, idx=None, return_kl=True, sparse=True):
        if self.dnn_to_bnn_flag:
            return_kl = False
        mu_weight, sigma_weight, sigma_bias, kl = self.sparse_mu_sigma(x, idx, return_kl=return_kl, sparse=sparse)
        x = x.flatten(start_dim=1)
        res = self.bnn_linear(x, mu_weight, sigma_weight, sigma_bias if self.bias_mu is not None else None)
        if return_kl:
            return res, kl
        else:
            return res

    def mc_forward(self, x, idx=None, num_mc=1, return_kl=True, sparse=True):
        output_ = []
        mu_weight, sigma_weight, sigma_bias, kl = self.sparse_mu_sigma(x, idx, return_kl=return_kl, sparse=sparse)
        x = x.flatten(start_dim=1)
        for mc_run in range(num_mc):
            output = self.bnn_linear(x, mu_weight, sigma_weight, sigma_bias if self.bias_mu is not None else None)
            output_.append(output)
        res = torch.stack(output_)
        if return_kl: 
            return res, kl
        else:
            return res
        

class Conv1dFlipout(_BaseVariationalLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements Conv1d layer with Flipout reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.bias = bias

        self.kl = 0

        self.mu_kernel = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size))
        self.rho_kernel = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size))

        self.register_buffer(
            'eps_kernel',
            torch.Tensor(out_channels, in_channels // groups, kernel_size),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(out_channels, in_channels // groups, kernel_size),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size),
            persistent=False)

        if self.bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.rho_bias = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()
        self.quant_prepare=False
    
    def prepare(self):
        self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(4)])
        self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(8)])
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare=True

    def init_parameters(self):
        # prior values
        self.prior_weight_mu.data.fill_(self.prior_mean)
        self.prior_weight_sigma.data.fill_(self.prior_variance)

        # init our weights for the deterministic and perturbated weights
        self.mu_kernel.data.normal_(mean=self.posterior_mu_init, std=.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init, std=.1)

        if self.bias:
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)
            self.prior_bias_mu.data.fill_(self.prior_mean)
            self.prior_bias_sigma.data.fill_(self.prior_variance)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
           sigma_bias = torch.log1p(torch.exp(self.rho_bias))
           kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, x, return_kl=True):

        if self.dnn_to_bnn_flag:
            return_kl = False

        # linear outputs
        outputs = F.conv1d(x,
                           weight=self.mu_kernel,
                           bias=self.mu_bias,
                           stride=self.stride,
                           padding=self.padding,
                           dilation=self.dilation,
                           groups=self.groups)

        # sampling perturbation signs
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        # gettin perturbation weights
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()

        delta_kernel = (sigma_weight * eps_kernel)

        if return_kl:
            kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu,
                             self.prior_weight_sigma)

        bias = None
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = (sigma_bias * eps_bias)
            if return_kl:
                kl = kl + self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        # perturbed feedforward
        x_tmp = x * sign_input
        perturbed_outputs_tmp = F.conv1d(x * sign_input,
                                     weight=delta_kernel,
                                     bias=bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     groups=self.groups)
        perturbed_outputs = perturbed_outputs_tmp * sign_output
        out = outputs + perturbed_outputs

        if self.quant_prepare:
            # quint8 quantstub
            x = self.quint_quant[0](x) # input
            outputs = self.quint_quant[1](outputs) # output
            sign_input = self.quint_quant[2](sign_input)
            sign_output = self.quint_quant[3](sign_output)
            x_tmp = self.quint_quant[4](x_tmp)
            perturbed_outputs_tmp = self.quint_quant[5](perturbed_outputs_tmp) # output
            perturbed_outputs = self.quint_quant[6](perturbed_outputs) # output
            out = self.quint_quant[7](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_kernel = self.qint_quant[1](self.mu_kernel) # weight
            eps_kernel = self.qint_quant[2](eps_kernel) # random variable
            delta_kernel =self.qint_quant[3](delta_kernel) # multiply activation

        # returning outputs + perturbations
        if return_kl:
            return outputs + perturbed_outputs, kl
        return outputs + perturbed_outputs


class Conv2dFlipout(_BaseVariationalLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements Conv2d layer with Flipout reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.bias = bias

        self.kl = 0
        kernel_size = get_kernel_size(kernel_size, 2)
        self.mu_kernel = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]))
        self.rho_kernel = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]))

        self.register_buffer(
            'eps_kernel',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)

        if self.bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.rho_bias = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()
        self.quant_prepare=False
    
    def prepare(self):
        self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(4)])
        self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(8)])
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare=True

    def init_parameters(self):
        # prior values
        self.prior_weight_mu.data.fill_(self.prior_mean)
        self.prior_weight_sigma.data.fill_(self.prior_variance)

        # init our weights for the deterministic and perturbated weights
        self.mu_kernel.data.normal_(mean=self.posterior_mu_init, std=.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init, std=.1)

        if self.bias:
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)
            self.prior_bias_mu.data.fill_(self.prior_mean)
            self.prior_bias_sigma.data.fill_(self.prior_variance)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
           sigma_bias = torch.log1p(torch.exp(self.rho_bias))
           kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, x, return_kl=True):

        if self.dnn_to_bnn_flag:
            return_kl = False

        # linear outputs
        outputs = F.conv2d(x,
                           weight=self.mu_kernel,
                           bias=self.mu_bias,
                           stride=self.stride,
                           padding=self.padding,
                           dilation=self.dilation,
                           groups=self.groups)

        # sampling perturbation signs
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        # gettin perturbation weights
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()

        delta_kernel = (sigma_weight * eps_kernel)

        if return_kl:
            kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu,
                             self.prior_weight_sigma)

        bias = None
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = (sigma_bias * eps_bias)
            if return_kl:
                kl = kl + self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        # perturbed feedforward
        x_tmp = x * sign_input
        perturbed_outputs_tmp = F.conv2d(x * sign_input,
                                     weight=delta_kernel,
                                     bias=bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     groups=self.groups)
        perturbed_outputs = perturbed_outputs_tmp * sign_output
        out = outputs + perturbed_outputs

        if self.quant_prepare:
            # quint8 quantstub
            x = self.quint_quant[0](x) # input
            outputs = self.quint_quant[1](outputs) # output
            sign_input = self.quint_quant[2](sign_input)
            sign_output = self.quint_quant[3](sign_output)
            x_tmp = self.quint_quant[4](x_tmp)
            perturbed_outputs_tmp = self.quint_quant[5](perturbed_outputs_tmp) # output
            perturbed_outputs = self.quint_quant[6](perturbed_outputs) # output
            out = self.quint_quant[7](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_kernel = self.qint_quant[1](self.mu_kernel) # weight
            eps_kernel = self.qint_quant[2](eps_kernel) # random variable
            delta_kernel =self.qint_quant[3](delta_kernel) # multiply activation

        # returning outputs + perturbations
        if return_kl:
            return out, kl
        return out

# BNN layers partially borrowed from BayesianTorch: https://github.com/IntelLabs/bayesian-torch
# ===============================================================================================


import torch
import torch.nn as nn
from itertools import repeat
import collections


def get_kernel_size(x, n):
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


class _BaseVariationalLayer(nn.Module):
    """
    The base variational layer is implemented as a :class:`torch.nn.Module` that, when called on two distributions 
    :math:`Q` and :math:`P` returns a :obj:`torch.Tensor` that represents the KL divergence between two gaussians.
    """

    def __init__(self):
        super().__init__()
        self._dnn_to_bnn_flag = False

    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        """
        KL( N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) ), elementwise then sum.
        Shapes broadcastable.
        """
        # avoid log(0)
        eps = 1e-8
        sigma_q = torch.clamp(sigma_q, min=eps)
        sigma_p = torch.clamp(sigma_p, min=eps)

        kl = torch.log(sigma_p) - torch.log(sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2.0 * sigma_p**2) - 0.5
        return kl.mean()  # kl.mean()

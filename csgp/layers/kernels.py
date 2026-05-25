import math
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn


class LaplaceL1Kernel(nn.Module):
    """
    Computes the covariance based on the Laplace L1 kernel.

    :param lengthscale: Set this if you want a customized lengthscale. (Default: 1.0.)
    :type lengthscale: float, optional
    """

    def __init__(self, lengthscale=1.0):
        super().__init__()
        assert lengthscale > 0, "lengthscale should be positive"
        self.register_buffer("lengthscale", torch.Tensor(1,))
        self.lengthscale.data.fill_(lengthscale)

    def forward(self, x1: Tensor, x2: Optional[Tensor] = None, 
                diag: bool = False, **params) -> Tensor:
        """
        :param x1: First set of data of shape :math:`(n,d)`.
        :type x1: torch.Tensor.float
        :param x2: Second set of data of shape :math:`(m,d)`.
        :type x2: torch.Tensor.float
        :param diag: Compute diagonal covariance matrix if `True`. It must be the case that `x1 == x2`.
        :type diag: bool, optional
        
        :return: The kernel matrix or vector. The shape depends on the kernel's mode:
            * 'full_cov`: `n x m`
            * `diag`: `n`
        """
        # Size checking
        if x1.ndimension() == 1:
            x1 = x1.unsqueeze(1)    # Add a last dimension, if necessary
        if x2 is not None:
            if x2.ndimension() == 1:
                x2 = x2.unsqueeze(1)
            if not x1.size(-1) == x2.size(-1):
                raise RuntimeError("x1 and x2 must have the same number of dimensions!")
        else:
            x2 = x1

        adjustment = x1.mean(dim=-2, keepdim=True)  # [d,] size tensor
        x1_ = (x1 - adjustment) / self.lengthscale
        x2_ = (x2 - adjustment) / self.lengthscale
        x1_eq_x2 = torch.equal(x1_, x2_)

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                distance = torch.zeros(*x1_.shape[:-2], x1_.shape[-2], dtype=x1_.dtype, device=x1.device)
            else:
                distance = torch.sum(torch.abs(x1_-x2_), dim=-1)
        else:
            distance = torch.cdist(x1_, x2_, p=1)

        res = torch.exp(-distance).clamp_min(1e-15)
        return res
    
    
class RBFKernel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ard: bool = True,
        init_lengthscale: float = 1.0,
        init_variance: float = 1.0,
    ):
        super().__init__()

        ls_shape = (input_dim,) if ard else (1,)
        self.log_lengthscale = nn.Parameter(
            torch.full(ls_shape, math.log(init_lengthscale), dtype=torch.get_default_dtype())
        )
        self.log_variance = nn.Parameter(
            torch.tensor(math.log(init_variance), dtype=torch.get_default_dtype())
        )

    @property
    def lengthscale(self):
        return self.log_lengthscale.exp()

    @property
    def variance(self):
        return self.log_variance.exp()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: (N, d)
        x2: (M, d)
        return: (N, M)
        """
        if x1.dim() != 2 or x2.dim() != 2:
            raise ValueError("RBFKernel 这里假设输入是 2D tensor: (N, d) 和 (M, d)")

        x1_scaled = x1 / self.lengthscale
        x2_scaled = x2 / self.lengthscale

        x1_sq = (x1_scaled ** 2).sum(dim=-1, keepdim=True)            # (N, 1)
        x2_sq = (x2_scaled ** 2).sum(dim=-1).unsqueeze(0)             # (1, M)
        sqdist = (x1_sq + x2_sq - 2.0 * x1_scaled @ x2_scaled.T).clamp_min(0.0)

        return self.variance * torch.exp(-0.5 * sqdist)
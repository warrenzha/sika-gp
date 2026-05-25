import numpy as np
import torch
import torch.nn as nn
import gpytorch
from ..design_class import HyperbolicCrossDesign
from .kernels import LaplaceL1Kernel
from ..chol_inv import mk_chol_inv


def dyadic_nonzero_indices(x: torch.Tensor, L: int, return_anchor: bool = False):
    if x.ndim == 0:
        x = x.unsqueeze(0)  # promote scalar to shape (1,)

    # 2^s for s=1..L
    pow2 = torch.pow(2, torch.arange(1, L + 1, device=x.device, dtype=torch.int64))  # (L,)

    # k_s = ceil(2^s * x) clamped to [1, 2^s - 1]
    ks = torch.ceil(x[..., None] * pow2.to(x.dtype)).to(torch.int64)  # (..., L)
    ks = torch.clamp(ks, min=1)
    ks_max = (pow2 - 1)  # (L,)
    ks = torch.minimum(ks, ks_max)  # (..., L)

    # r_s^(odd): force to be odd (right endpoint index made odd)
    # if ks even -> ks-1, else ks
    rs = ks - ((ks & 1) == 0).to(torch.int64)  # (..., L), odd in {1,3,...,2^s-1}

    # position within level s: t_s in {1,...,2^{s-1}}
    ts = (rs + 1) // 2  # (..., L)

    # offsets: number of columns before level s (0-based indexing)
    offsets = (pow2 // 2) - 1  # (L,)

    # global 0-based indices: J_s = offset(s) + (t_s - 1)
    idx = offsets + (ts - 1)  # (..., L)

    if return_anchor:
        anchor = torch.tensor([2 ** L - 1, 2 ** L], device=x.device, dtype=torch.int64).expand(
            (*idx.shape[:-1], 2))  # (..., 2)
        idx = torch.cat([idx, anchor], dim=-1)  # (..., L+2)

    return idx


def dyadic_to_dense(vals, idx, L, return_anchor: bool = False):
    """
    Convert dyadic sparse representation to dense.

    Args
    ----
    vals : (..., L) tensor
        Values at the dyadic non-zero indices.
    idx : (..., L) long tensor
        0-based global column indices in dyadic order for each level (DC is level 1).
        The leading shape of idx must match that of vals.

    Returns
    -------
    dense : (..., 2^L +- 1) tensor
        Dense representation.
    """
    if return_anchor:
        m = 2 ** L + 1
    else:
        m = 2 ** L - 1

    dense = torch.zeros(*vals.shape[:-1], m, device=vals.device, dtype=vals.dtype)
    dense.scatter_(-1, idx, vals)
    return dense


class LapL1Feature(nn.Module):
    def __init__(self, dyadic_level: int = 3):
        super(LapL1Feature, self).__init__()

        self.dyadic_level = dyadic_level
        pow2 = 2 ** torch.arange(1, dyadic_level + 1, dtype=torch.int64)  # (L,)
        self.register_buffer('pow2', pow2)
        # design_points = HyperbolicCrossDesign(dyadic_sort=True, return_neighbors=True)(deg=dyadic_level).points  # (2^L-1,)
        # self.register_buffer('design_points', design_points)

    def psi_anchor(self, x: torch.Tensor, ell_c: float = 1.0):
        x = torch.exp(- (x / ell_c)) + torch.exp(- ((1 - x) / ell_c))
        coeff = torch.tensor(
            [1.0 / np.sqrt(2.0 * (1 + np.exp(- 1.0 / ell_c))), 1.0 / np.sqrt(2.0 * (1 - np.exp(- 1.0 / ell_c)))],
            device=x.device, dtype=x.dtype
        )
        res = x.unsqueeze(-1) @ coeff.unsqueeze(0)  # (..., N, 1)
        return res  # (..., N, 2)

    def dyadic_psi(self, x: torch.Tensor, ell_c: float = 1.0, return_idx: bool = True, return_anchor: bool = False):
        """
        Batched dyadic nonzero indices.

        Args
        ----
        x : (...,) tensor with values in [0, 1].
            Works with any number of leading batch dims.
        ell_c: length scale of the Laplace kernel.
        return_idx: return the non-zero indices.

        Returns
        -------
        idx : (..., L) long tensor
            0-based global column indices in dyadic order for each level (DC is level 1).
            The returned shape matches the leading shape of x, with an extra trailing dim of size L.
        """
        device, dtype = x.device, x.dtype

        if x.ndim == 0:
            x = x.unsqueeze(0)  # promote scalar to shape (1,)

        # idx = dyadic_nonzero_indices(x, self.dyadic_level)
        # u = self.design_points
        # view_shape = (1,) * x.dim() + (u.shape[0],)  # (1,1,...,1, 2^L-1)
        # u_selected = torch.gather(u.view(view_shape).expand(*x.shape, -1), dim=-1, index=idx)  # (..., L)
        # delta = torch.abs(x.unsqueeze(-1) - u_selected)  # |x - m2^{-l}|

        pow2 = self.pow2

        # k_s = ceil(2^s * x) clamped to [1, 2^s - 1]
        ks = torch.ceil(x[..., None] * pow2.to(dtype)).to(torch.int64)  # (..., L)
        ks = torch.clamp(ks, min=1)
        ks_max = (pow2 - 1)  # (L,)
        ks = torch.minimum(ks, ks_max)  # (..., L)

        # r_s^(odd): force to be odd (right endpoint index made odd)
        # if ks even -> ks-1, else ks
        rs = ks - ((ks & 1) == 0).to(torch.int64)  # (..., L), odd in {1,3,...,2^s-1}

        # position within level s: t_s in {1,...,2^{s-1}}
        ts = (rs + 1) // 2  # (..., L)

        # offsets: number of columns before level s (0-based indexing)
        offsets = (pow2 // 2) - 1  # (L,)

        # global 0-based indices: J_s = offset(s) + (t_s - 1)
        idx = offsets + (ts - 1)  # (..., L)

        delta = torch.abs(x.unsqueeze(-1) - (rs / pow2).to(dtype))  # (..., L)
        pow2_f = (1.0 / pow2).to(dtype)
        # pow2_f = torch.pow(0.5, torch.arange(1, self.dyadic_level + 1, device=x.device, dtype=x.dtype))  # (L,)

        psi = torch.sqrt(2 / torch.sinh(pow2_f * 2 / ell_c)) * torch.sinh((pow2_f - delta) / ell_c)

        if return_anchor:
            anchor_idx = torch.tensor([2 ** self.dyadic_level - 1, 2 ** self.dyadic_level], device=x.device,
                                      dtype=torch.int64).expand((*idx.shape[:-1], 2))  # (..., 2)
            idx = torch.cat([idx, anchor_idx], dim=-1)  # (..., L+2)
            psi_anchor = self.psi_anchor(x, ell_c=ell_c)  # (..., 2)
            psi = torch.cat([psi, psi_anchor], dim=-1)  # (..., L+2)

        if return_idx:
            return psi, idx
        else:
            return psi

    def forward(self, x, ell_c: float = 1.0, return_sparse: bool = True, return_anchor: bool = False):
        psi, idx = self.dyadic_psi(x, ell_c, return_idx=True, return_anchor=return_anchor)
        if return_sparse:
            return psi, idx
        else:
            psi = dyadic_to_dense(psi, idx, self.dyadic_level, return_anchor)
            return psi


class LapL1Cholesky(nn.Module):
    def __init__(self,
                 dyadic_level,
                 lengthscale=1.0,
                 grid_bounds=(-1., 1.),
                 ):
        super().__init__()

        self.dyadic_level = dyadic_level  # Dyadic level: L, M = 2^L - 1
        self.kernel = LaplaceL1Kernel(lengthscale=lengthscale)  # stationary kernel k(x,y) = exp(-||x-y||_1/ell)
        dyadic_design = HyperbolicCrossDesign(dyadic_sort=True, return_neighbors=True)(deg=dyadic_level,
                                                                                       input_lb=grid_bounds[0],
                                                                                       input_ub=grid_bounds[1])

        chol_inv = mk_chol_inv(dyadic_design=dyadic_design, markov_kernel=self.kernel, upper=True)  # [M, M] size tensor
        design_points = dyadic_design.points.reshape(-1, 1)  # [M, 1] size tensor

        self.register_buffer('design_points', design_points)  # [M, D] size tensor
        self.register_buffer('chol_inv', chol_inv)  # [M, M] size tensor, inverse of Cholesky

    def forward(self, x, return_sparse=False):
        x = x.unsqueeze(dim=-1)  # reshape x of size [B, D] --> size [B, D, 1]
        out = self.kernel(x, self.design_points)  # [B, D, M] size tensor
        out = torch.matmul(out, self.chol_inv).to_dense().contiguous()  # [B, D, M] size tensor

        if return_sparse:
            idx = dyadic_nonzero_indices(x.squeeze(-1), self.dyadic_level)  # [B, L] size tensor
            out = torch.gather(out, dim=-1, index=idx)  # [B, D, L] size tensor
            return out, idx
        else:
            return out
 

class KernelInducingFeatures(nn.Module):
    r"""
    phi(X) = K(X, U) L^{-T}, where K(U, U) = L L^T.

    Shapes:
        X:   (..., N, d)
        U:   (M, d)
        Kxu: (..., N, M)
        L:   (M, M)
        phi: (..., N, M)
    """

    def __init__(
        self,
        kernel: nn.Module,
        inducing_points: torch.Tensor,
        learn_inducing_locations: bool = True,
        learn_kernel_hyperparams: bool = True,
        jitter: float = 1e-6,
    ):
        super().__init__()
        self.kernel = kernel
        self.jitter = jitter

        inducing_points = inducing_points.clone()
        if learn_inducing_locations:
            self.register_parameter(
                "inducing_points",
                torch.nn.Parameter(inducing_points),
            )
        else:
            self.register_buffer("inducing_points", inducing_points)

        if not learn_kernel_hyperparams:
            for p in self.kernel.parameters():
                p.requires_grad_(False)

    @staticmethod
    def _to_dense(obj: torch.Tensor) -> torch.Tensor:
        # GPyTorch kernels may return a LinearOperator instead of a plain Tensor.
        return obj.to_dense() if hasattr(obj, "to_dense") else obj

    def _compute_cholesky(self) -> torch.Tensor:
        U = self.inducing_points  # (M, d)
        Kuu = self._to_dense(self.kernel(U, U))  # (M, M)

        M = U.size(-2)
        eye = torch.eye(M, dtype=Kuu.dtype, device=Kuu.device)
        Kuu = Kuu + self.jitter * eye

        # Lower-triangular L such that Kuu = L L^T
        L = torch.linalg.cholesky(Kuu)
        return L

    def whitening_matrix(self) -> torch.Tensor:
        """
        Returns L^{-T} with shape (M, M).
        """
        L = self._compute_cholesky()
        I = torch.eye(L.size(-1), dtype=L.dtype, device=L.device)

        # L^{-1}
        L_inv = torch.linalg.solve_triangular(L, I, upper=False)
        # L^{-T}
        return L_inv.transpose(-1, -2)

    def forward(self, X: torch.Tensor, return_cholesky: bool = False):
        """
        Args:
            X: (..., N, d)
            return_cholesky: if True, also return L

        Returns:
            phi: (..., N, M)
            optionally L: (M, M)
        """
        if X.size(-1) != self.inducing_points.size(-1):
            raise ValueError(
                f"Input dim mismatch: X has d={X.size(-1)}, "
                f"but inducing_points have d={self.inducing_points.size(-1)}."
            )

        U = self.inducing_points  # (M, d)

        # K(X, U): (..., N, M)
        Kxu = self._to_dense(self.kernel(X, U))

        # L: (M, M)
        L = self._compute_cholesky()

        # phi = Kxu @ L^{-T}
        # computed stably as:
        # phi^T = L^{-1} @ Kxu^T
        phi = torch.linalg.solve_triangular(
            L,                       # (M, M), lower triangular
            Kxu.transpose(-1, -2),   # (..., M, N)
            upper=False,
        ).transpose(-1, -2)         # (..., N, M)

        if return_cholesky:
            return phi, L
        return phi
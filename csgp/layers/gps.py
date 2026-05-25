import torch
import torch.nn as nn
from .activation import LapL1Feature, LapL1Cholesky
from .linear import SparseLinearFlipout, LinearFlipout
from ..utils import ScaleToBounds


class CSGP(nn.Module):
    def __init__(self, in_features, out_features, dyadic_level, ell_c=1.0, grid_bounds=(0., 1.),
                 prior_mean=0, prior_variance=1, posterior_mu_init=0, posterior_rho_init=-3.0,
                 bias=True, anchor=False):
        super(CSGP, self).__init__()

        self.kernel_act = LapL1Feature(dyadic_level=dyadic_level)
        self.ell_c = ell_c

        if anchor:
            self.anchor = True
            m = (2 ** dyadic_level) + 1  # number of features
        else:
            self.anchor = False
            m = (2 ** dyadic_level) - 1  # number of features

        self.linear = SparseLinearFlipout(
            in_features = in_features * m,
            out_features=out_features,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=bias
        )

        # This module will scale the features so that they're nice values
        self.scale_to_bounds = ScaleToBounds(grid_bounds[0], grid_bounds[1])

    def get_sparse_val_idx(self, x):
        x = self.scale_to_bounds(x)
        val, idx = self.kernel_act(x, ell_c=self.ell_c, return_sparse=True, return_anchor=self.anchor)
        return val, idx

    def forward(self, x, return_kl=False, sparse=True):
        x = self.scale_to_bounds(x)
        if sparse:
            psi, idx = self.kernel_act(x, ell_c=self.ell_c, return_sparse=True, return_anchor=self.anchor)
            if return_kl:
                res, kl = self.linear(psi, idx=idx, return_kl=return_kl, sparse=True)
                return res, kl
            else:
                res = self.linear(psi, idx=idx, return_kl=return_kl, sparse=True)
                return res
        else:
            psi = self.kernel_act(x, ell_c=self.ell_c, return_sparse=False, return_anchor=self.anchor)
            if return_kl:
                res, kl = self.linear(psi, return_kl=return_kl, sparse=False)
                return res, kl
            else:
                res = self.linear(psi, return_kl=return_kl, sparse=False)
                return res

    def mc_forward(self, x, num_mc=1, return_kl=False, sparse=True):
        x = self.scale_to_bounds(x)
        idx = None
        if sparse:
            psi, idx = self.kernel_act(x, ell_c=self.ell_c, return_sparse=sparse, return_anchor=self.anchor)
        else:
            psi = self.kernel_act(x, ell_c=self.ell_c, return_sparse=sparse, return_anchor=self.anchor)
        res, kl = self.linear.mc_forward(psi, idx, num_mc=num_mc, return_kl=return_kl, sparse=sparse)
        if return_kl:
            return res, kl
        else:
            return res


class CholesGP(nn.Module):
    def __init__(self, in_features, out_features, dyadic_level, ell_c=1.0, grid_bounds=(0., 1.),
                 prior_mean=0, prior_variance=1, posterior_mu_init=0, posterior_rho_init=-3.0,
                 bias=True):
        super(CholesGP, self).__init__()

        self.kernel_act = LapL1Cholesky(dyadic_level=dyadic_level, lengthscale=ell_c, grid_bounds=grid_bounds)
        self.ell_c = ell_c

        m = (2 ** dyadic_level) - 1  # number of features
        fsize = in_features * m

        self.linear = SparseLinearFlipout(
            in_features=fsize,
            out_features=out_features,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=bias
        )

        # This module will scale the features so that they're nice values
        self.scale_to_bounds = ScaleToBounds(grid_bounds[0], grid_bounds[1])

    def get_sparse_val_idx(self, x):
        x = self.scale_to_bounds(x)
        val, idx = self.kernel_act(x, return_sparse=True)
        return val, idx

    def forward(self, x, return_kl=False, sparse=True):
        x = self.scale_to_bounds(x)
        if sparse:
            psi, idx = self.kernel_act(x, return_sparse=True)
            # psi = torch.gather(psi, dim=-1, index=idx)  # (..., L)
            if return_kl:
                res, kl = self.linear(psi, idx=idx, return_kl=return_kl, sparse=True)
                return res, kl
            else:
                res = self.linear(psi, idx=idx, return_kl=return_kl, sparse=True)
                return res
        else:
            psi = self.kernel_act(x, return_sparse=False)
            if return_kl:
                res, kl = self.linear(psi, return_kl=return_kl, sparse=False)
                return res, kl
            else:
                res = self.linear(psi, return_kl=return_kl, sparse=False)
                return res

    def mc_forward(self, x, num_mc=1, return_kl=False, sparse=True):
        x = self.scale_to_bounds(x)
        idx = None
        if sparse:
            psi, idx = self.kernel_act(x, return_sparse=sparse)
        else:
            psi = self.kernel_act(x, return_sparse=sparse)
        res, kl = self.linear.mc_forward(psi, idx, num_mc=num_mc, return_kl=return_kl, sparse=sparse)
        if return_kl:
            return res, kl
        else:
            return res


class CompactGPwEmbeding(nn.Module):
    def __init__(self, in_features, out_features, dyadic_level, ell_c=1.0, grid_bounds=(0., 1.),
                 prior_mean=0, prior_variance=1, posterior_mu_init=0, posterior_rho_init=-3.0,
                 bias=True, anchor=False, sparse_linear=True):
        super(CompactGPwEmbeding, self).__init__()

        self.kernel_act = LapL1Feature(dyadic_level=dyadic_level)
        self.ell_c = ell_c
        self.use_sparse = sparse_linear

        if anchor:
            self.anchor = True
            m = (2 ** dyadic_level) + 1  # number of features
            L = dyadic_level + 2
        else:
            self.anchor = False
            m = (2 ** dyadic_level) - 1  # number of features
            L = dyadic_level

        if sparse_linear:
            in_dim = in_features * L
            self.idx_embed = nn.Embedding(m, 1)
        else:
            in_dim = in_features * m

        self.linear = LinearFlipout(
            in_features=in_dim,
            out_features=out_features,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=bias
        )

        # This module will scale the features so that they're nice values
        self.scale_to_bounds = ScaleToBounds(grid_bounds[0], grid_bounds[1])

    def forward(self, x, return_kl=False, sparse=True):
        x = self.scale_to_bounds(x)
        if self.use_sparse:
            psi, idx = self.kernel_act(x, ell_c=self.ell_c, return_sparse=True, return_anchor=self.anchor)
            # Get gamma and beta
            idx_embed = self.idx_embed(idx)  # (..., L, 1)
            psi = psi + idx_embed.squeeze(-1)  # (..., L)
        else:
            psi = self.kernel_act(x, ell_c=self.ell_c, return_sparse=False, return_anchor=self.anchor)
        psi = torch.flatten(psi, start_dim=-2)
        if return_kl:
            res, kl = self.linear(psi, return_kl=return_kl)
            return res, kl
        else:
            res = self.linear(psi, return_kl=return_kl)
            return res

    def mc_forward(self, x, num_mc=1, return_kl=False, sparse=True):
        x = self.scale_to_bounds(x)
        if self.use_sparse:
            psi, idx = self.kernel_act(x, ell_c=self.ell_c, return_sparse=True, return_anchor=self.anchor)
            idx_embed = self.idx_embed(idx)  # (..., L, 1)
            psi = psi + idx_embed.squeeze(-1)  # (..., L)
        else:
            psi = self.kernel_act(x, ell_c=self.ell_c, return_sparse=False, return_anchor=self.anchor)
        psi = torch.flatten(psi, start_dim=-2)

        res_ = []
        kl_ = []
        for mc_run in range(num_mc):
            res, kl = self.linear(psi, return_kl=True)
            res_.append(res)
            kl_.append(kl)
        res = torch.stack(res_)  # (num_mc, batch_size, out_features)
        kl = torch.mean(torch.stack(kl_), dim=0)
        if return_kl:
            return res, kl
        else:
            return res
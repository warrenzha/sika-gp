import torch.nn as nn
from ..layers.gps import CSGP, CholesGP, CompactGPwEmbeding as CompactGP


class DeepCSGP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, dyadic_levels, ell_c=1.0,
                 prior_mean=0, prior_variance=1, posterior_mu_init=0, posterior_rho_init=-3.0,
                 bias=True, anchor=False):
        super(DeepCSGP, self).__init__()

        layers = []
        all_features = [in_features] + hidden_features + [out_features]
        num_layers = len(all_features) - 1

        assert len(dyadic_levels) == num_layers, "Length of dyadic_levels must match number of layers"

        for i in range(num_layers):
            layers.append(
                CSGP(
                    in_features=all_features[i],
                    out_features=all_features[i + 1],
                    dyadic_level=dyadic_levels[i],
                    ell_c=ell_c,
                    prior_mean=prior_mean,
                    prior_variance=prior_variance,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init,
                    bias=bias,
                    anchor=anchor,
                )
            )
            # if i < num_layers - 1:
            #     layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x, return_kl=False, sparse=True):
        if return_kl:
            total_kl = 0
            for layer in self.model:
                if isinstance(layer, CSGP):
                    x, kl = layer(x, return_kl=return_kl, sparse=sparse)
                    total_kl += kl
                else:
                    x = layer(x)
            return x, total_kl
        else:
            for layer in self.model:
                if isinstance(layer, CSGP):
                    x = layer(x, return_kl=return_kl, sparse=sparse)
                else:
                    x = layer(x)
            return x


class DeepCholesGP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, dyadic_levels, ell_c=1.0, grid_bounds=(0., 1.),
                 prior_mean=0, prior_variance=1, posterior_mu_init=0, posterior_rho_init=-3.0,
                 bias=True):
        super(DeepCholesGP, self).__init__()

        self.embedding = nn.LazyLinear(in_features)

        layers = []
        all_features = [in_features] + hidden_features + [out_features]
        num_layers = len(all_features) - 1

        assert len(dyadic_levels) == num_layers, "Length of dyadic_levels must match number of layers"

        for i in range(num_layers):
            layers.append(
                CholesGP(
                    in_features=all_features[i],
                    out_features=all_features[i + 1],
                    dyadic_level=dyadic_levels[i],
                    ell_c=ell_c,
                    grid_bounds=grid_bounds,
                    prior_mean=prior_mean,
                    prior_variance=prior_variance,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init,
                    bias=bias,
                )
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x, return_kl=False, sparse=True):
        x = self.embedding(x)
        if return_kl:
            total_kl = 0
            for layer in self.model:
                x, kl = layer(x, return_kl=return_kl, sparse=sparse)
                total_kl += kl
            return x, total_kl
        else:
            for layer in self.model:
                x = layer(x, return_kl=return_kl, sparse=sparse)
            return x


class DeepCompactGP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, dyadic_levels, ell_c=1.0,
                 prior_mean=0, prior_variance=1, posterior_mu_init=0, posterior_rho_init=-3.0,
                 bias=True, anchor=False, sparse_linear=True):
        super(DeepCompactGP, self).__init__()
        layers = []
        all_features = [in_features] + hidden_features + [out_features]
        num_layers = len(all_features) - 1

        assert len(dyadic_levels) == num_layers, "Length of dyadic_levels must match number of layers"

        for i in range(num_layers):
            layers.append(
                CompactGP(
                    in_features=all_features[i],
                    out_features=all_features[i + 1],
                    dyadic_level=dyadic_levels[i],
                    ell_c=ell_c,
                    prior_mean=prior_mean,
                    prior_variance=prior_variance,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init,
                    bias=bias,
                    anchor=anchor,
                    sparse_linear=sparse_linear,
                )
            )
            # if i < num_layers - 1:
            #     layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x, return_kl=False, sparse=True):
        if return_kl:
            total_kl = 0
            for layer in self.model:
                if isinstance(layer, CompactGP):
                    x, kl = layer(x, return_kl=return_kl, sparse=sparse)
                    total_kl += kl
                else:
                    x = layer(x)
            return x, total_kl
        else:
            for layer in self.model:
                if isinstance(layer, CompactGP):
                    x = layer(x, return_kl=return_kl, sparse=sparse)
                else:
                    x = layer(x)
            return x

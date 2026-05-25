import torch
import gpytorch
import torch.nn as nn

from csgp.layers.gps import CSGP, CholesGP


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

class DAK(nn.Module):
    """DAK with Monte Carlo Sampling"""

    def __init__(self, feature_extractor,
                 num_features=64, num_tasks=10,
                 dyadic_level=3, ell_c=1.0,
                 grid_bounds=(0., 1.),
                 embedding=True,
                 ):
        super(DAK, self).__init__()
        self.feature_extractor = feature_extractor
        self.embedding = nn.Linear(feature_extractor.out_features, num_features, bias=True)
        num_dim = num_features
        if not embedding:
            self.embedding = None
            num_dim = feature_extractor.out_features

        self.gp = CSGP(
            in_features=num_dim,
            out_features=num_tasks,
            dyadic_level=dyadic_level,
            ell_c=ell_c,
            grid_bounds=grid_bounds,
        )
        # self._init_params()

    def _init_params(self):
        self.apply(_weights_init)
        
    def forward(self, x, return_kl=True, sparse=True):
        res = self.feature_extractor(x)
        res = self.embedding(res) if self.embedding is not None else res
        
        res, kl = self.gp(res, return_kl=True, sparse=sparse)

        return res, kl if return_kl else res

    def forward_with_MC(self, x, num_mc=1, return_kl=True, sparse=True):
        x = self.feature_extractor(x)
        if self.embedding is not None:
            x = self.embedding(x)
            
        output_mc, kl = self.gp.mc_forward(x, num_mc=num_mc, return_kl=return_kl, sparse=sparse)
        res = torch.mean(output_mc, dim=0)
        
        if return_kl:
            return res, kl
        else:
            return res
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from linear_operator.utils.cholesky import psd_safe_cholesky  # from gpytorch's linear_operator
from csgp.design_class import HyperbolicCrossDesign
from csgp.layers.kernels import LaplaceL1Kernel


@torch.no_grad()  # drop this if you want gradients through everything
def inv_cholesky_transpose(K: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    """
    Returns U = L^{-T}, where K = L L^T (L lower-triangular), so K^{-1} = U U^T.
    Works with batched SPD matrices (..., n, n).
    """
    n = K.size(-1)
    I = torch.eye(n, dtype=K.dtype, device=K.device)
    # Stable Cholesky of K (lower-triangular)
    L = psd_safe_cholesky(K, jitter=jitter)  # (..., n, n), lower-triangular
    # Solve L^T U = I  ->  U = L^{-T} (upper-triangular)
    U = torch.linalg.solve_triangular(L.transpose(-1, -2), I, upper=True)
    return U  # U is upper-triangular and K^{-1} = U @ U.transpose(-1, -2)

def phi(x, L, ell_c=1.0):
    device = x.device
    dyadic_design = HyperbolicCrossDesign(dyadic_sort=True, return_neighbors=True)(deg=L, input_lb=0, input_ub=1)
    design_points = dyadic_design.points.reshape(-1, 1).to(device)  # [m, 1] size tensor

    covar = LaplaceL1Kernel(lengthscale=ell_c).to(device)(design_points)
    L_inv_T = inv_cholesky_transpose(covar)
    h = LaplaceL1Kernel(lengthscale=ell_c).to(device)(x.unsqueeze(-1), design_points)
    phi = torch.matmul(h, L_inv_T) # [B, D, M] size tensor
    return phi

def dyadic_psi(x: torch.Tensor, L: int, ell_c: float = 1.0):
    """
    Batched dyadic nonzero indices.

    Args
    ----
    x : (...,) tensor with values in [0, 1].
        Works with any number of leading batch dims.
    L : int, number of dyadic levels (total columns m = 2^L - 1).

    Returns
    -------
    idx : (..., L) long tensor
        0-based global column indices in dyadic order for each level (DC is level 1).
        The returned shape matches the leading shape of x, with an extra trailing dim of size L.
    """
    if x.ndim == 0:
        x = x.unsqueeze(0)  # promote scalar to shape (1,)
    device, dtype = x.device, x.dtype

    # 2^s for s=1..L
    pow2 = torch.pow(2, torch.arange(1, L+1, device=device, dtype=torch.int64))  # (L,)

    # k_s = ceil(2^s * x) clamped to [1, 2^s - 1]
    ks = torch.ceil(x[..., None] * pow2.to(x.dtype)).to(torch.int64)  # (..., L)
    ks = torch.clamp(ks, min=1)
    ks_max = (pow2 - 1)  # (L,)
    ks = torch.minimum(ks, ks_max)  # (..., L)

    # r_s^(odd): force to be odd (right endpoint index made odd)
    # if ks even -> ks-1, else ks
    rs = ks - ((ks & 1) == 0).to(torch.int64) # (..., L), odd in {1,3,...,2^s-1}

    # position within level s: t_s in {1,...,2^{s-1}}
    ts = (rs + 1) // 2  # (..., L)

    # offsets: number of columns before level s (0-based indexing)
    offsets = (pow2 // 2) - 1  # (L,)

    # global 0-based indices: J_s = offset(s) + (t_s - 1)
    idx = offsets + (ts - 1)  # (..., L)

    u = HyperbolicCrossDesign(dyadic_sort=True, return_neighbors=True)(deg=L, input_lb=0, input_ub=1).points.to(device) # (2^L-1,)
    view_shape = (1,) * x.dim() + (u.shape[0],)      # (1,1,...,1, 2^L-1)
    u_selected = torch.gather(u.view(view_shape).expand(*x.shape, -1), dim=-1, index=idx) # (..., L)

    delta = torch.abs(x.unsqueeze(-1) - u_selected) # |x - m2^{-l}|
    # pow2_f = (1.0 / pow2).view(view_shape).expand(*x.shape, -1)  # (..., L), 2^{-l} for l=1..L
    pow2_f = (1.0 / pow2).to(x.dtype)

    psi =  torch.sqrt(2 / torch.sinh(pow2_f * 2 * ell_c)) * torch.sinh(ell_c * (pow2_f - delta))

    return psi, idx  # (N, L), long

# Dense multiplication
@torch.inference_mode()
def dense_csgp(x, weight, L, num_samples=1):
    bsize = x.shape[0]
    x_ = phi(x, L, ell_c=1.0).view(bsize, -1) # [B, N * M] size tensor
    weight = weight.unsqueeze(0).expand(bsize, -1, -1) # [B, out_features, N * M] size tensor
    output_ = []
    for _ in range(num_samples):
        noise = torch.randn_like(weight, device=weight.device)
        output = torch.matmul(weight * noise, x_.unsqueeze(-1)).squeeze(-1)
        output_.append(output)
    res = torch.mean(torch.stack(output_), dim=0)
    return res

# Sparse inference based on compact support
@torch.inference_mode()
def sparse_csgp(x, weight, L, num_samples=1):
    bsize = x.shape[0]
    fsize = x.shape[1]
    osize = weight.shape[0]

    x_, idx = dyadic_psi(x, L, ell_c=1.0)  # [B, N * L] size tensor

    w_batch = weight.unsqueeze(0).expand(bsize, -1, -1).view(bsize, fsize, -1, osize)
    idx_batch = idx.unsqueeze(-1).expand(*idx.shape, osize)
    w_selected = torch.gather(w_batch, dim=-2, index=idx_batch).view(bsize, osize, -1)

    x_ = x_.view(bsize, -1)
    output_ = []
    for _ in range(num_samples):
        noise = torch.randn_like(w_selected, device=weight.device)
        output = torch.matmul(w_selected * noise, x_.unsqueeze(-1)).squeeze(-1)
        output_.append(output)
    res = torch.mean(torch.stack(output_), dim=0)
    return res

# --------- Quick timing ----------
def time_once(fn, device):
    # warmup
    for _ in range(5): _ = fn()
    if device == "cuda":
        start = torch.cuda.Event(True); end = torch.cuda.Event(True)
        start.record(); _ = fn(); end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)  # ms
    else:
        t0 = time.perf_counter(); _ = fn(); t1 = time.perf_counter()
        return (t1 - t0) * 1000.0


parser = argparse.ArgumentParser(description='Time Analysis')
parser.add_argument('--log-dir', type=str, default='./logs/time_analysis')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--in-features', type=int, default=128)
parser.add_argument('--out-features', type=int, default=10)
parser.add_argument('--samples', type=int, default=10)


if __name__ == '__main__':
    global args
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    log_f_name = args.log_dir + '/batch' + str(args.batch_size) + '_in' + str(args.in_features) + '_out' + str(args.out_features) + '_samples' + str(args.samples) + '.csv'
    os.makedirs(args.log_dir, exist_ok=True)
    log_f = open(log_f_name, 'w+')
    log_f.write('L,dense_mean,dense_std,sparse_mean,sparse_std\n')
    
    # set 3 seeds for each L
    seeds = [0, 2024, 42]
    for L in range(1, 13):  # L=1..12
        m = 2**L - 1
        print(f'L={L}, m={m}')
        dense_times, sparse_times = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            x = torch.rand(args.batch_size, args.in_features, device=device)  # [B, N]
            weight = torch.randn(args.out_features, m * args.in_features, device=device)  # [M, out_features]

            dense_fn = lambda: dense_csgp(x, weight, L, num_samples=args.samples)
            sparse_fn = lambda: sparse_csgp(x, weight, L, num_samples=args.samples)

            dense_time = time_once(dense_fn, device)
            sparse_time = time_once(sparse_fn, device)
            print(f'  seed={seed}, dense_time={dense_time:.2f} ms, sparse_time={sparse_time:.2f} ms')
            dense_times.append(dense_time)
            sparse_times.append(sparse_time)
        
        avg_dense_time = np.mean(dense_times, axis=0)
        std_dense_time = np.std(dense_times, axis=0)
        avg_sparse_time = np.mean(sparse_times, axis=0)
        std_sparse_time = np.std(sparse_times, axis=0)
        log_f.write(f'{L},{avg_dense_time:.2f},{std_dense_time:.2f},{avg_sparse_time:.2f},{std_sparse_time:.2f}\n')
        log_f.flush()
    
    log_f.close()
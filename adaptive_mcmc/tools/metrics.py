from typing import Optional

import numpy as np
import ot
from sklearn.neighbors import KernelDensity
import torch

from KDEpy import FFTKDE


def compute_ess(
    samples: torch.Tensor,
    stride: int = 1,
    max_lag: int = 150,
    threshold: float = 0.05,
) -> (float, float):
    """
    Effective Sample Size (ESS) per chain ÷ dimension.

    Args:
      samples: Tensor of shape (T, C, D)
      stride: subsample stride along time
      max_lag: how many lags to include
      threshold: ignore autocorr below this

    Returns:
      (ess_mean, ess_std) across all C×D estimates
    """
    arr = samples[::stride]
    T1, C, D = arr.shape

    mean = arr.mean(dim=0, keepdim=True)          # [1, C, D]
    var0 = ((arr - mean) * (arr - mean)).sum(dim=0) / T1  # [C, D]

    rho_sum = torch.zeros_like(var0)
    for lag in range(1, min(max_lag, T1)):
        cov = ((arr[:-lag] - mean) * (arr[lag:] - mean)).sum(dim=0) / (T1 - lag)
        rho = cov / (var0 + 1e-12)
        rho_sum += torch.clamp(rho - threshold, min=0.0)

    ess = 1.0 / (1.0 + 2.0 * rho_sum)  # [C, D]
    ess = ess.mean(dim=-1)
    return ess.mean().item(), ess.std().item()


def compute_tv(
    xs_true: torch.Tensor,
    xs_pred: torch.Tensor,
    density_probe_count: Optional[int] = None,
    projection_count: int = 25,
    method: str = 'fastkde',
    device: torch.device = None,
) -> (float, float):
    """
    Approximate average projected total‐variation distance.

    Args:
      xs_true:  Tensor [N, D]
      xs_pred:  Tensor [N, C, D]
      density_probe_count: # of bins (histogram) or grid points (KDE)
      projection_count: # random 1‑D projections
      method: one of 'histogram' or 'kde'
    Returns:
      (tv_mean, tv_std) across all projections × chains
    """
    if device is None:
        device = xs_true.device

    N, D = xs_true.shape
    _, C, _ = xs_pred.shape
    x0 = xs_true.mean(dim=0, keepdim=True)  # [1, D]

    all_tvs = []
    for _ in range(projection_count):
        v = torch.randn(D, device=device)
        v = v / (v.norm() + 1e-12)

        t_proj = (xs_true - x0) @ v           # [N]
        p_proj = (xs_pred - x0) @ v           # [N, C]

        if method == 'histogram':
            mn = torch.min(t_proj.min(), p_proj.min())
            mx = torch.max(t_proj.max(), p_proj.max())

            if density_probe_count is None:
                density_probe_count = 2 * int(max(t_proj.shape[0], p_proj.shape[0])**0.5)

            hist_t = torch.histogram(
                t_proj, bins=density_probe_count, range=(mn.item(), mx.item())
            )[0].float()
            hist_t /= hist_t.sum()

            hists_p = []
            for c in range(C):
                h = torch.histogram(
                    p_proj[:, c], bins=density_probe_count, range=(mn.item(), mx.item())
                )[0].float()
                hists_p.append(h / h.sum())
            hist_p = torch.stack(hists_p, dim=1)   # [bins, C]

            tv = 0.5 * torch.sum(torch.abs(hist_p - hist_t.unsqueeze(1)), dim=0)  # [C]

        elif method == 'kde':
            t_np = t_proj.cpu().numpy()
            p_np = p_proj.cpu().numpy()
            n, C = p_np.shape

            bandwidth = 1.0

            kde_t = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(t_np[:, None])

            x_min = min(t_np.min(), p_np.min())
            x_max = max(t_np.max(), p_np.max())

            grid = np.linspace(x_min, x_max, density_probe_count)[:, None]

            log_t = kde_t.score_samples(grid)
            t_pdf = np.exp(log_t)

            tv_vals = []
            for c in range(C):
                kde_p = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(p_np[:, c][:, None])
                log_p = kde_p.score_samples(grid)
                p_pdf = np.exp(log_p)
                tv_c = 0.5 * np.trapz(np.abs(t_pdf - p_pdf), x=grid.ravel())
                tv_vals.append(tv_c)

            tv = torch.tensor(tv_vals, device=device, dtype=torch.float32)
        elif method == "fastkde":
            t_np = t_proj.cpu().numpy()           # (n,)
            p_np = p_proj.cpu().numpy()           # (n, C)
            _, C = p_np.shape

            bandwidth = 1.0

            kde_t = FFTKDE(kernel='gaussian', bw=bandwidth).fit(t_np)
            x_t, t_pdf = kde_t.evaluate(density_probe_count)

            tv_vals = []
            for c in range(C):
                kde_p = FFTKDE(kernel='gaussian', bw=bandwidth).fit(p_np[:, c])
                x_p, p_pdf = kde_p.evaluate(density_probe_count)

                x_min = min(x_t.min(), x_p.min())
                x_max = max(x_t.max(), x_p.max())
                grid = np.linspace(x_min, x_max, density_probe_count)

                t_on_union = np.interp(grid, x_t, t_pdf)
                p_on_union = np.interp(grid, x_p, p_pdf)

                tv_c = 0.5 * np.trapz(np.abs(t_on_union - p_on_union), x=grid)
                tv_vals.append(tv_c)

            tv = torch.tensor(tv_vals, device=device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown method {method!r}, choose 'histogram', 'kde', or 'fastkde'")

        all_tvs.append(tv)

    tvs = torch.stack(all_tvs, dim=0).reshape(-1)

    return tvs.mean().item(), tvs.std().item()


def compute_emd(samples_true, samples_pred, max_iter_ot=1_000_000):
    M = np.array(ot.dist(samples_true, samples_pred))
    return ot.lp.emd2([], [], M, numItermax=max_iter_ot)


def compute_metrics(
    xs_true: torch.Tensor,
    xs_pred: torch.Tensor,
    name: Optional[str] = None,
    stride: int = 1,
    max_lag: int = 150,
    threshold: float = 0.05,
    method: str = "fastkde",
    density_probe_count: int = 1000,
    projection_count: int = 25,
    max_iter_ot: int = 1_000_000,
    scale: float = 1.0,
):
    """
    xs_true -> (sample_count, dimension)
    xs_pred -> (sample_count, chain_count, dimension)
    """
    metrics = dict()

    metrics["ess_mean"], metrics["ess_conf_sigma"] = compute_ess(
        samples=xs_pred,
        stride=stride,
        max_lag=max_lag,
        threshold=threshold,
    )
    metrics["ess_conf_sigma"] /= xs_pred.shape[1] ** 0.5

    metrics["tv_mean"], metrics["tv_conf_sigma"] = compute_tv(
        xs_true,
        xs_pred,
        projection_count=projection_count,
        density_probe_count=density_probe_count,
        method=method,
    )
    metrics["tv_conf_sigma"] /= xs_pred.shape[1] ** 0.5

    # metrics["wasserstein"] = 0
    # for b in range(xs_pred.shape[1]):
    #     metrics["wasserstein"] += compute_emd(
    #         xs_true / scale, xs_pred[:, b] / scale, max_iter_ot=max_iter_ot
    #     )
    # metrics["wasserstein"] /= xs_pred.shape[1]

    return metrics

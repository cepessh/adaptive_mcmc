import jax
import jax.numpy as jnp
import numpy as np
import ot

from ex2mcmc.metrics.total_variation import average_total_variation
from ex2mcmc.metrics.chain import ESS, acl_spectrum


def compute_tv(xs_true, xs_pred, density_probe_count, projection_count=25, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)

    tracker = average_total_variation(
        key,
        xs_true,
        xs_pred,
        projection_count=projection_count,
        density_probe_count=density_probe_count,
    )

    return tracker.mean(), tracker.std()


def compute_ess(samples, ess_rar=1):
    return ESS(acl_spectrum(samples[::ess_rar] - samples[::ess_rar].mean(0)[None, ...])).mean()


def compute_emd(samples_true, samples_pred, max_iter_ot=1_000_000):
    M = np.array(ot.dist(samples_true, samples_pred))
    return ot.lp.emd2([], [], M, numItermax=max_iter_ot)


def compute_metrics(
    xs_true,
    xs_pred,
    name=None,
    density_probe_count=1000,
    scale=1.0,
    trunc_chain_len: int = 0,
    ess_rar=1,
    max_iter_ot=1_000_000,
    projection_count=25,
):
    """
    xs_true -> (sample_count, dimension)
    xs_pred -> (sample_count, chain_count, dimension)
    """
    if not isinstance(xs_true, jnp.ndarray):
        xs_true = jnp.array(xs_true)
    if not isinstance(xs_pred, jnp.ndarray):
        xs_pred = jnp.array(xs_pred)

    metrics = dict()
    key = jax.random.PRNGKey(0)

    metrics["ess"] = compute_ess(xs_pred, ess_rar)

    xs_pred = xs_pred[-trunc_chain_len:]

    metrics["tv_mean"], metrics["tv_conf_sigma"] = compute_tv(
        xs_true,
        xs_pred,
        projection_count=projection_count,
        density_probe_count=density_probe_count,
        key=key,
    )
    metrics["tv_conf_sigma"] /= xs_pred.shape[1] ** 0.5

    # metrics["wasserstein"] = 0

    # for b in range(xs_pred.shape[1]):
    #     metrics["wasserstein"] += compute_emd(
    #         xs_true / scale, xs_pred[:, b] / scale, max_iter_ot=max_iter_ot
    #     )
    # metrics["wasserstein"] /= xs_pred.shape[1]

    return metrics

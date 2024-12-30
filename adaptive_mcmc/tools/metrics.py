import jax
import jax.numpy as jnp
import numpy as np
import ot

from ex2mcmc.metrics.total_variation import average_total_variation
from ex2mcmc.metrics.chain import ESS, acl_spectrum, autocovariance


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


def compute_metrics(
    xs_true,
    xs_pred,
    name=None,
    density_probe_count=1000,
    scale=1.0,
    trunc_chain_len: int = 0,
    ess_rar=1,
    max_iter_ot=1_000_000,
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
    projection_count = 25

    ess = ESS(
        acl_spectrum(
            xs_pred[::ess_rar] - xs_pred[::ess_rar].mean(0)[None, ...],
        ),
    ).mean()
    metrics["ess"] = ess

    xs_pred = xs_pred[-trunc_chain_len:]

    metrics["tv_mean"], metrics["tv_conf_sigma"] = compute_tv(
        xs_true,
        xs_pred,
        projection_count=projection_count,
        density_probe_count=density_probe_count,
        key=key,
    )

    metrics["wasserstein"] = 0

    for b in range(xs_pred.shape[1]):
        M = np.array(ot.dist(xs_true / scale, xs_pred[:, b] / scale))
        emd = ot.lp.emd2([], [], M, numItermax=max_iter_ot)
        metrics["wasserstein"] += emd / xs_pred.shape[1]

    return metrics

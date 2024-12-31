from dataclasses import dataclass, field

import jax.numpy as jnp

from adaptive_mcmc.samplers.base_sampler import Cache
from adaptive_mcmc.tools.metrics import compute_tv


@dataclass
class StopStatus:
    is_stop: bool
    meta: dict = field(default_factory=dict)


@dataclass
class TVStop:
    threshold: float = 0.1
    projection_count: int = 25
    density_probe_count: int = 1000
    tail_count_cap: int = 0

    def __call__(self, cache: Cache) -> StopStatus:
        tv_mean, tv_std = compute_tv(
            jnp.array(cache.true_samples),
            jnp.array(cache.samples),
            density_probe_count=self.density_probe_count,
            projection_count=self.projection_count
        )

        return StopStatus(
            is_stop=tv_mean < self.threshold,
            meta={
                "tv_mean": tv_mean,
                "tv_std": tv_std,
            },
        )


@dataclass
class NoStop:
    projection_count: int = 25
    density_probe_count: int = 1000
    tail_count_cap: int = 0

    def __call__(self, cache: Cache) -> StopStatus:
        tv_mean, tv_std = compute_tv(
            jnp.array(cache.true_samples),
            jnp.array(cache.samples),
            density_probe_count=self.density_probe_count,
            projection_count=self.projection_count
        )

        return StopStatus(
            is_stop=False,
            meta={
                "tv_mean": tv_mean,
                "tv_std": tv_std,
            },
        )

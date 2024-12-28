from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from adaptive_mcmc.distributions.distribution import (
    SamplableDistribution, GaussianMixture, Distribution
)
from adaptive_mcmc.samplers.base_sampler import Cache
from adaptive_mcmc.tools.metrics import tv_threshold


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
        tv_mean, tv_std = tv_threshold(
            jnp.array(cache.true_samples), jnp.array(cache.samples[-self.tail_count_cap:]),
            self.density_probe_count, self.projection_count
        )

        return StopStatus(
            is_stop=tv_mean < self.threshold,
            meta={
                "tv_mean": tv_mean,
                "tv_std": tv_std,
            },
        )

from dataclasses import dataclass
import time
from typing import Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
from torch import Tensor
import torch

from adaptive_mcmc.distributions.distribution import Distribution, SamplableDistribution
import adaptive_mcmc.tools.metrics as metrics


@dataclass
class BenchmarkUtils:

    @staticmethod
    def generate_starting_points(chain_count: int, dimension: int,
                                 mass_points: Tensor, distance: float) -> Tensor:
        starting_points = Tensor(chain_count, dimension).uniform_(-1, 1)
        starting_points /= torch.norm(starting_points, dim=1).reshape(-1, 1)

        length = Tensor(chain_count).uniform_(0, distance).reshape(-1, 1)
        return mass_points[torch.randint(0, len(mass_points), (chain_count,))] + \
                           starting_points * length

    @staticmethod
    def sample_mcmc(sampling_algorithm: Callable, starting_points: Tensor,
                    target_dist: Distribution, sample_count: int, **kwargs) -> Tensor:
        return sampling_algorithm(starting_points=starting_points,
                                  target_dist=target_dist,
                                  sample_count=sample_count, **kwargs)

    @staticmethod
    def plot_samples(ax: plt.Axes, samples: Tensor,
                     title: Union[None, str] = None) -> None:
        ax.scatter(samples[:, 0], samples[:, 1])
        if title is not None:
            ax.set_title(title)

    @staticmethod
    def create_plot(mcmc_samples: Tensor, true_samples: Union[None, Tensor] = None,
                    target_title: Union[None, str] = None, fig_side: int = 5):

        chain_count = mcmc_samples.shape[1]
        mcmc_first_ax = 0

        if true_samples is not None:
            fig, axes = plt.subplots(nrows=chain_count+1, ncols=1,
                                     figsize=(fig_side, fig_side*(chain_count+1)))
            BenchmarkUtils.plot_samples(axes[0], true_samples, target_title)
            mcmc_first_ax = 1
        else:
            fig, axes = plt.subplots(nrows=chain_count, ncols=1,
                                     figsize=(fig_side, fig_side*chain_count))

        for chain_index, ax in enumerate(axes[mcmc_first_ax:]):
            BenchmarkUtils.plot_samples(ax, mcmc_samples[:, chain_index],
                                        f"chain_{chain_index+1}")

    @staticmethod
    def compute_metrics(mcmc_samples: Tensor, true_samples: Tensor,
                        **kwargs) -> dict:
        return metrics.compute_metrics(jnp.array(true_samples),
                                       jnp.array(mcmc_samples), **kwargs)

@dataclass
class Benchmark:
    target_dist: SamplableDistribution
    target_dist_mass_points: Tensor
    target_dist_title: str
    dimension: int
    sampling_algorithm: Callable
    sample_count: int
    chain_count: int
    distance_to_mass_points: float

    def run(self, plot=False, **kwargs) -> dict:
        start_time = time.perf_counter()

        starting_points = BenchmarkUtils.generate_starting_points(
            self.chain_count, self.dimension,
            self.target_dist_mass_points,
            self.distance_to_mass_points
        )

        true_samples = self.target_dist.sample(self.sample_count)

        mcmc_samples = BenchmarkUtils.sample_mcmc(
            self.sampling_algorithm, starting_points, self.target_dist,
            sample_count=self.sample_count, **kwargs)[0].detach()

        if plot:
            BenchmarkUtils.create_plot(mcmc_samples, true_samples, "true dist")

        res_metrics = BenchmarkUtils.compute_metrics(mcmc_samples, true_samples)
        time_elapsed = time.perf_counter() - start_time
        res_metrics["time_elapsed"] = time_elapsed

        return res_metrics


@dataclass
class BenchmarkStoppingRule:
    pass

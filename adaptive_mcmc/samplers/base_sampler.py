from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
import time
import tqdm
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from adaptive_mcmc.distributions.distribution import Distribution


@dataclass
class Params:
    target_dist: Optional[Union[torchDist, Distribution]] = None
    starting_point: Optional[Tensor] = None
    proposal_dist: Optional[Union[torchDist, Distribution]] = None

    meta: dict = field(default_factory=dict)

    def copy(self) -> "Params":
        return Params(
            target_dist=self.target_dist,
            starting_point=self.starting_point,
            proposal_dist=self.proposal_dist,
            meta=self.meta.copy()
        )

    def update_meta(self, new_meta) -> "Params":
        self.meta.update(new_meta)
        return self.copy()


@dataclass
class Cache:
    params: Params = field(default_factory=Params)

    true_samples: Optional[Tensor] = None
    samples: Optional[Tensor] = None

    point: Optional[Tensor] = None
    logp: Optional[Tensor] = None
    grad: Optional[Tensor] = None


def update_params(params: Params, cache: Cache) -> None:
    if cache is None:
        return

    params.starting_point = cache.point
    params.update_meta(cache.params.meta)


@dataclass
class Iteration(ABC):
    cache: Cache

    def init(self) -> None:
        self.cache.point = self.cache.params.starting_point.requires_grad_()
        self.cache.logp = self.cache.params.target_dist.log_prob(self.cache.point)
        self.cache.grad = torch.autograd.grad(
            self.cache.logp.sum(),
            self.cache.point,
        )[0].detach()

    @abstractmethod
    def run(self) -> Cache:
        raise NotImplementedError


@dataclass
class SampleBlock:
    iteration: Iteration
    iteration_count: int
    stopping_rule: Optional[Callable] = None
    probe_period: Optional[int] = None
    stop_data_hist: list = field(default_factory=list)

    def run(self, cache: Cache = None, progress: bool = True) -> Cache:
        update_params(self.iteration.cache.params, cache)
        self.iteration.init()

        bar = tqdm.notebook.trange(self.iteration_count) if progress else range(self.iteration_count)

        for iter_step in bar:
            self.iteration.run()

            if self.stopping_rule and (iter_step + 1) % self.probe_period == 0:
                stop_status = self.stopping_rule(self.iteration.cache)
                self.stop_data_hist.append(stop_status.meta)

                if stop_status.is_stop:
                    break

        return self.iteration.cache


def track_runtime(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f"Runtime: {runtime:.2f}s")

        if hasattr(self, 'runtime'):
            self.runtime = runtime

        return result

    return wrapper


@dataclass
class Pipeline:
    sample_blocks: list[SampleBlock]
    runtime: float = 0

    @track_runtime
    def run(self, cache: Optional[Cache] = None) -> None:
        print("number of blocks:", len(self.sample_blocks))
        for block_index, block in enumerate(self.sample_blocks):
            print("processing block:", block_index + 1)
            cache = block.run(cache)


@dataclass
class Algorithm(ABC):
    pipeline: Optional[Pipeline]

    @abstractmethod
    def load_params(self, params: Params):
        pass


@dataclass
class AlgorithmStoppingRule(Algorithm):
    def load_true_samples(self, true_samples: Tensor, node_index: int = -1) -> None:
        self.pipeline.sample_blocks[node_index].iteration.cache.true_samples = true_samples

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
import time
import tqdm
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from adaptive_mcmc.distributions.distribution import Distribution


@dataclass
class Params(ABC):
    target_dist: Optional[Union[torchDist, Distribution]] = None
    starting_point: Optional[Tensor] = None
    proposal_dist: Optional[Union[torchDist, Distribution]] = None
    device: str = "cpu"

    def copy(self) -> "Params":
        return deepcopy(self)

    def update(self, new_params: "Params") -> None:
        for field_name in new_params.__dict__:
            new_value = getattr(new_params, field_name)
            if new_value is not None:
                setattr(self, field_name, new_value)

    def copy_update(self, new_params: "Params") -> None:
        updated = self.copy()
        updated.update(new_params)

        return updated


@dataclass
class Cache:
    true_samples: Optional[Tensor] = None
    samples: Optional[Tensor] = None

    point: Optional[Tensor] = None
    logp: Optional[Tensor] = None
    grad: Optional[Tensor] = None


def update_params(params: Params, cache: Cache, new_params: Params) -> None:
    if cache is None:
        return

    params.starting_point = cache.point
    params.update(new_params)


@dataclass
class Iteration(ABC):
    cache: Cache = field(default_factory=Cache)
    params: Params = field(default_factory=Params)

    def init(self) -> None:
        self.cache.point = self.params.starting_point.requires_grad_()
        self.cache.logp = self.params.target_dist.log_prob(self.cache.point)
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

    def run(self, cache: Optional[Cache] = None, params: Optional[Params] = None, progress: bool = True) -> Tuple[Cache, Params]:
        update_params(self.iteration.params, cache, params)
        self.iteration.init()

        bar = tqdm.notebook.trange(self.iteration_count) if progress else range(self.iteration_count)

        for iter_step in bar:
            self.iteration.run()

            if self.stopping_rule and (iter_step + 1) % self.probe_period == 0:
                stop_status = self.stopping_rule(self.iteration.cache)
                self.stop_data_hist.append(stop_status.meta)

                if stop_status.is_stop:
                    break

        return self.iteration.cache, self.iteration.params


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
    def run(self, cache: Optional[Cache] = None, params: Optional[Params] = None) -> None:
        print("number of blocks:", len(self.sample_blocks))
        for block_index, block in enumerate(self.sample_blocks):
            print("processing block:", block_index + 1)
            cache, params = block.run(cache, params)


@dataclass
class Algorithm(ABC):
    pipeline: Optional[Pipeline]
    name: str

    @abstractmethod
    def load_params(self, params: Params):
        pass

    def run(self):
        print("Running", self.name)
        self.pipeline.run()


@dataclass
class AlgorithmStoppingRule(Algorithm):
    def load_true_samples(self, true_samples: Tensor, node_index: int = -1) -> None:
        self.pipeline.sample_blocks[node_index].iteration.cache.true_samples = true_samples

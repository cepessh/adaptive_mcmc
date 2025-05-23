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
class CommonParams(ABC):
    target_dist: Optional[Union[torchDist, Distribution]] = None
    starting_point: Optional[Tensor] = None
    proposal_dist: Optional[Union[torchDist, Distribution]] = None

    def copy(self) -> "CommonParams":
        return deepcopy(self)

    def update(self, new_params: "CommonParams") -> None:
        for field_name in new_params.__dict__:
            new_value = getattr(new_params, field_name)
            if new_value is not None:
                setattr(self, field_name, new_value)

    def copy_update(self, new_params: "CommonParams") -> None:
        updated = self.copy()
        updated.update(new_params)

        return updated


@dataclass
class FixedParams(ABC):
    device: str = "cpu"


@dataclass
class Cache:
    """
    samples: (sample_count, chain_count, dimension)
    """
    true_samples: Optional[Tensor] = None
    samples: Optional[Tensor] = None

    point: Optional[Tensor] = None
    logp: Optional[Tensor] = None
    grad: Optional[Tensor] = None
    accept_prob_hist: list[float] = field(default_factory=list[float])
    broken_trajectory_count: int = 0


def update_params(params: CommonParams, cache: Cache, new_params: CommonParams) -> None:
    if new_params is not None:
        params.update(new_params)

    if cache is not None:
        params.starting_point = cache.point


@dataclass
class Iteration(ABC):
    cache: Cache = field(default_factory=Cache)
    common_params: CommonParams = field(default_factory=CommonParams)
    fixed_params: FixedParams = field(default_factory=FixedParams)
    collect_required: bool = False
    index: int = 0

    def init(self, cache=None) -> None:
        self.cache.point = self.common_params.starting_point.clone().requires_grad_()
        self.cache.logp = self.common_params.target_dist.log_prob(self.cache.point)
        self.cache.grad = torch.autograd.grad(
            self.cache.logp.sum(),
            self.cache.point,
            retain_graph=False,
        )[0].detach()
        self.cache.point = self.cache.point.detach()

    @abstractmethod
    def run(self) -> Cache:
        raise NotImplementedError

    def collect_sample(self, sample: Tensor):
        assert self.cache.samples is not None
        self.cache.samples[self.index] = sample.unsqueeze(0)
        self.index += 1


@dataclass
class MHIteration(Iteration):
    def MHStep(self, point_new: Tensor, logp_new: Tensor, grad_new: Tensor, accept_prob: Tensor) -> None:
        with torch.no_grad():
            mask = torch.rand_like(accept_prob, device=self.fixed_params.device) < accept_prob

            self.cache.point[mask] = point_new[mask]
            self.cache.logp[mask] = logp_new[mask]
            self.cache.grad[mask] = grad_new[mask]

            self.cache.accept_prob_hist.append(accept_prob.mean().item())

        self.cache.point = self.cache.point.detach()    # .requires_grad_(self.cache.point.requires_grad)
        self.cache.logp = self.cache.logp.detach()  # .requires_grad_(self.cache.logp.requires_grad)
        self.cache.grad = self.cache.grad.detach()  # .requires_grad_(self.cache.grad.requires_grad)


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
class SampleBlock:
    iteration: Iteration
    iteration_count: int
    stopping_rule: Optional[Callable] = None
    probe_period: Optional[int] = None
    stop_data_hist: list = field(default_factory=list)
    pure_runtime: float = 0.
    callback: Optional[Callable] = None

    def run(self, cache: Optional[Cache] = None, common_params: Optional[CommonParams] = None,
            progress: bool = True) -> Tuple[Cache, CommonParams]:
        update_params(self.iteration.common_params, cache, common_params)

        if cache is not None:
            if isinstance(cache, type(self.iteration.cache)):
                self.iteration.cache = cache
            else:
                for name, value in vars(cache).items():
                    setattr(self.iteration.cache, name, value)

        self.iteration.init(cache)
        self.pure_runtime = 0.

        bar = tqdm.notebook.trange(self.iteration_count) if progress else range(self.iteration_count)

        for iter_step in bar:
            start_time = time.perf_counter()
            self.iteration.run()
            self.pure_runtime += time.perf_counter() - start_time

            if self.stopping_rule and (iter_step + 1) % self.probe_period == 0:
                stop_status = self.stopping_rule(self.iteration.cache)
                self.stop_data_hist.append(stop_status.meta)

                if stop_status.is_stop:
                    break

        return self.iteration.cache, self.iteration.common_params


@dataclass
class Pipeline:
    sample_blocks: list[SampleBlock] = field(default_factory=list[SampleBlock])
    runtime: float = 0.
    pure_runtime: float = 0.
    broken_trajectory_ratio: float = 0.

    def append(self, block: SampleBlock):
        self.sample_blocks.append(block)

    @track_runtime
    def run(self, cache: Optional[Cache] = None, params: Optional[CommonParams] = None) -> None:
        print("number of blocks:", len(self.sample_blocks))

        sample_count = 0
        for block in self.sample_blocks:
            shape = block.iteration.common_params.starting_point.shape
            device = block.iteration.fixed_params.device
            if block.iteration.collect_required:
                block.iteration.index = sample_count
                sample_count += block.iteration_count

        samples = None
        if sample_count:
            samples = torch.empty(sample_count, *shape, device=device)
        self.sample_blocks[0].iteration.cache.samples = samples

        for block_index, block in enumerate(self.sample_blocks):
            print("processing block:", block_index + 1)

            cache, params = block.run(cache, params)

            if block.callback is not None:
                block.callback()

            self.pure_runtime += block.pure_runtime


@dataclass
class Algorithm(ABC):
    pipeline: Optional[Pipeline]
    name: str

    @abstractmethod
    def load_params(self, common_params: CommonParams):
        pass

    def run(self):
        print("Running", self.name)
        self.pipeline.run()


@dataclass
class AlgorithmStoppingRule(Algorithm):
    def load_true_samples(self, true_samples: Tensor, node_index: int = 0) -> None:
        self.pipeline.sample_blocks[node_index].iteration.cache.true_samples = true_samples

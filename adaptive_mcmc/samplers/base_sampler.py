from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from distributions import SamplableDistribution, GaussianMixture, Distribution


@dataclass
class Params:
    target_dist: Optional[Union[torchDist, Distribution]] = None
    starting_point: Optional[Tensor] = None
    proposal_dist: Optional[Union[torchDist, Distribution]] = None

    meta: dict = field(default_factory=dict)

    def copy(self):
        return Params(
            target_dist=self.target_dist,
            starting_point=self.starting_point,
            proposal_dist=self.proposal_dist,
            meta=self.meta.copy())

    def update_meta(self, new_meta):
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


def update_params(params: Params, cache: Cache):
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

    def run(self, cache: Cache = None) -> Cache:
        update_params(self.iteration.cache.params, cache)
        self.iteration.init()

        for iter_step in range(self.iteration_count):
            self.iteration.run()

            if self.stopping_rule and (iter_step + 1) % self.probe_period == 0:
                stop_data = self.stopping_rule(self.iteration.cache)
                self.stop_data_hist.append(stop_data.meta)

                if stop_data.is_stop:
                    break

        return self.iteration.cache


@dataclass
class Pipeline:
    sample_blocks: list[SampleBlock]

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

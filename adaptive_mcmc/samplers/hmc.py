from dataclasses import dataclass
from functools import partial
from typing import Callable, Union

import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from adaptive_mcmc.distributions.distribution import Distribution
from adaptive_mcmc.samplers import base_sampler


class HMCIter(base_sampler.Iteration):
    def init(self):
        pass

    def run(self):
        pass


class HMCVanilla(base_sampler.AlgorithmStoppingRule):
    prec_burn_in_params: dict
    prec_burn_in_iter_count: int
    sample_iter_count: int
    probe_period: int
    stopping_rule: Callable

    def load_params(self, params: base_sampler.Params):
        self.pipeline = base_sampler.Pipeline(
            [
                base_sampler.SampleBlock(
                    iteration=MALAIter(base_sampler.Cache(
                        params.update_meta(self.sigma_burn_in_params))),
                    iteration_count=self.sigma_burn_in_iter_count,
                ),
                base_sampler.SampleBlock(
                    iteration=FisherMALAIter(base_sampler.Cache(
                        params.update_meta(self.prec_burn_in_params))),
                    iteration_count=self.prec_burn_in_iter_count,
                ),
                base_sampler.SampleBlock(
                    iteration=FisherMALAIter(base_sampler.Cache(
                        params.update_meta(self.prec_burn_in_params))),
                    iteration_count=self.sample_iter_count,
                    stopping_rule=self.stopping_rule,
                    probe_period=self.probe_period,
                ),
            ]
        )

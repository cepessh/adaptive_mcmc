from dataclasses import dataclass, field
from typing import Callable, Union

import torch
from torch import Tensor

from adaptive_mcmc.samplers import base_sampler


@dataclass
class MALAParams(base_sampler.Params):
    sigma: Union[Tensor, float] = 1.
    target_acceptance: float = 0.574
    sigma_lr: float = 0.015


@dataclass
class MALAIter(base_sampler.MHIteration):
    params: MALAParams = field(default_factory=MALAParams)

    def init(self, cache=None):
        super().init(cache)

        if isinstance(self.params.sigma, float):
            self.params.sigma = Tensor([self.params.sigma]).repeat(
                *self.cache.point.shape[:-1], 1)
        else:
            self.params.sigma = self.params.sigma.reshape(
                *self.cache.point.shape[:-1], 1)

    def run(self):
        params = self.params

        noise = params.proposal_dist.sample(self.cache.point.shape[:-1])

        proposal_point = (
            self.cache.point
            + 0.5 * self.cache.grad * params.sigma ** 2
            + noise * params.sigma
        ).detach().requires_grad_()

        logp_y = params.target_dist.log_prob(proposal_point)
        grad_y = torch.autograd.grad(logp_y.sum(), proposal_point)[0].detach()

        with torch.no_grad():
            log_qyx = params.proposal_dist.log_prob(noise)
            log_qxy = params.proposal_dist.log_prob(
                (self.cache.point - proposal_point
                 - 0.5 * params.sigma ** 2 * grad_y) / params.sigma
            )
            accept_prob = torch.clamp((logp_y + log_qxy - self.cache.logp - log_qyx).exp(), max=1.).detach()

            params.sigma = params.sigma * (
                1 + params.sigma_lr * (
                    accept_prob.unsqueeze(-1) - params.target_acceptance
                )
            ) ** 0.5

        self.MHStep(
            point_new=proposal_point,
            logp_new=logp_y,
            grad_new=grad_y,
            accept_prob=accept_prob,
        )

        if self.params.collect_sample:
            self.collect_sample(self.cache.point.detach().clone())


@dataclass
class MALAVanilla(base_sampler.AlgorithmStoppingRule):
    sigma_burn_in_params: MALAParams
    sigma_burn_in_iter_count: int
    sample_iter_count: int
    probe_period: int
    stopping_rule: Callable

    def load_params(self, params: base_sampler.Params):
        self.pipeline = base_sampler.Pipeline([
            base_sampler.SampleBlock(
                iteration=MALAIter(params=params.copy_update(self.sigma_burn_in_params)),
                iteration_count=self.sigma_burn_in_iter_count,
            ),
            base_sampler.SampleBlock(
                iteration=MALAIter(params=params.copy_update(self.sigma_burn_in_params)),
                iteration_count=self.sample_iter_count,
                stopping_rule=self.stopping_rule,
                probe_period=self.probe_period,
            ),
        ])

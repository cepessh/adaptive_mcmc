from dataclasses import dataclass, field
from typing import Callable, Union

import torch
from torch import Tensor

from adaptive_mcmc.samplers import base_sampler


@dataclass
class MALACommonParams(base_sampler.CommonParams):
    sigma: Union[Tensor, float] = 1e-2
    sigma_lr: float = 1e-2


@dataclass
class MALAFixedParams(base_sampler.FixedParams):
    target_acceptance: float = 0.574


@dataclass
class MALAIter(base_sampler.MHIteration):
    common_params: MALACommonParams = field(default_factory=MALACommonParams)
    fixed_params: MALAFixedParams = field(default_factory=MALAFixedParams)

    def init(self, cache=None):
        super().init(cache=cache)

        if isinstance(self.common_params.sigma, float):
            self.common_params.sigma = torch.full(
                (self.cache.point.shape[0], 1),
                self.common_params.sigma,
                device=self.fixed_params.device,
            )

    def run(self):
        params = self.common_params

        noise = params.proposal_dist.sample(self.cache.point.shape[:-1])

        while True:
            proposal_point = (
                self.cache.point
                + 0.5 * self.cache.grad * params.sigma ** 2
                + noise * params.sigma
            ).detach().requires_grad_()

            logp_y = params.target_dist.log_prob(proposal_point)
            grad_y = torch.autograd.grad(logp_y.sum(), proposal_point)[0].detach()

            mask = (~torch.isfinite(logp_y))
            mask |= (~torch.isfinite(grad_y)).any(dim=-1)

            if mask.any():
                params.sigma = params.sigma * (1 - params.sigma_lr * mask.unsqueeze(-1))
            else:
                break

        with torch.no_grad():
            while True:
                try:
                    log_qyx = params.proposal_dist.log_prob(noise)
                    log_qxy = params.proposal_dist.log_prob(
                        (self.cache.point - proposal_point
                         - 0.5 * params.sigma ** 2 * grad_y) / params.sigma
                    )
                    accept_prob = torch.clamp((logp_y + log_qxy - self.cache.logp - log_qyx).exp(), max=1.).detach()
                    break

                except ValueError:
                    params.sigma = params.sigma * (1 - params.sigma_lr * mask)

            params.sigma = params.sigma * (
                1 + params.sigma_lr * (
                    accept_prob.unsqueeze(-1) - self.fixed_params.target_acceptance
                )
            )

        self.MHStep(
            point_new=proposal_point,
            logp_new=logp_y,
            grad_new=grad_y,
            accept_prob=accept_prob,
        )

        if self.collect_required:
            self.collect_sample(self.cache.point.detach().clone())


@dataclass
class MALAVanilla(base_sampler.AlgorithmStoppingRule):
    sigma_burn_in_params: MALACommonParams
    sigma_burn_in_iter_count: int
    sample_iter_count: int
    probe_period: int
    stopping_rule: Callable

    def load_params(self, params: base_sampler.CommonParams):
        self.pipeline = base_sampler.Pipeline([
            base_sampler.SampleBlock(
                iteration=MALAIter(common_params=params),
                iteration_count=self.sigma_burn_in_iter_count,
            ),
            base_sampler.SampleBlock(
                iteration=MALAIter(
                    common_params=params,
                    collect_required=True,
                ),
                iteration_count=self.sample_iter_count,
                stopping_rule=self.stopping_rule,
                probe_period=self.probe_period,
            ),
        ])

        def make_callback(block: base_sampler.SampleBlock):
            def callback():
                print(f"step_size={block.iteration.common_params.sigma.mean()}")
            return callback

        for block in self.pipeline.sample_blocks:
            block.callback = make_callback(block)

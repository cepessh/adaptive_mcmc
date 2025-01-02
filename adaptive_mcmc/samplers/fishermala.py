from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from adaptive_mcmc.distributions.distribution import Distribution
from adaptive_mcmc.samplers import base_sampler
from adaptive_mcmc.samplers.mala import MALAIter, MALAParams


def h(z: Tensor, v: Tensor, sigma: Tensor, prec_factors: list[Tensor],
      target_dist: Union[Distribution, torchDist]) -> Tensor:
    """
    z, v (sample_count, n_dim)
    sigma (sample_count)
    prec_factors List[(sample_count, n_dim, n_dim)]
    """

    logp_v = target_dist.log_prob(v)
    grad_v = torch.autograd.grad(logp_v.sum(), v)[0].detach()

    grad_v_img = prec_factors[-1] @ grad_v[..., None]
    for factor in reversed(prec_factors[:-1]):
        grad_v_img = factor @ grad_v_img

    grad_v_img = grad_v_img.squeeze()

    return 0.5 * (
        grad_v[:, None, :] @ (z - v - 0.25 * grad_v_img * sigma[..., None] ** 2)[..., None]
    ).squeeze()


@dataclass
class FisherMALAParams(base_sampler.Params):
    prec: Optional[Tensor] = None
    sigma: Union[Tensor, float] = 1.
    sigma_prec: Optional[Tensor] = None
    target_acceptance: float = 0.574
    sigma_lr: float = 0.015
    dampening: float = 10.


@dataclass
class FisherMALAIter(base_sampler.Iteration):
    params: FisherMALAParams = field(default_factory=FisherMALAParams)

    def init(self):
        super().init()
        self.step_id = 0

        if self.params.prec is None:
            self.params.prec = torch.eye(self.cache.point.shape[-1]).repeat(*self.cache.point.shape[:-1], 1, 1)

        """
        sigma_prec: (chain_count, 1, 1)
        """
        if isinstance(self.params.sigma, float):
            self.params.sigma_prec = Tensor([self.params.sigma]).repeat(*self.cache.point.shape[:-1], 1, 1)
            self.params.sigma = Tensor([self.params.sigma]).repeat(*self.cache.point.shape[:-1], 1)
        else:
            self.params.sigma = self.params.sigma.reshape(*self.cache.point.shape[:-1], 1)
            self.params.sigma_prec = self.params.sigma
            while len(self.params.sigma_prec.shape) < 3:
                self.params.sigma_prec = self.params.sigma_prec[..., None]

    def run(self):
        params = self.params

        h_ = partial(h, prec_factors=[params.prec, params.prec.permute(0, 2, 1)],
                     target_dist=params.target_dist, sigma=params.sigma_prec.squeeze())

        noise = params.proposal_dist.sample(self.cache.point.shape[:-1])

        grad_x_img = self.cache.grad[..., None]
        grad_x_img = params.prec @ (params.prec.permute(0, 2, 1) @ grad_x_img)

        proposal_point = (
            self.cache.point + (
                0.5 * grad_x_img * params.sigma_prec ** 2
                + params.prec @ noise[..., None] * params.sigma_prec
            ).squeeze()
        )
        proposal_point = proposal_point.detach().requires_grad_()

        logp_y = params.target_dist.log_prob(proposal_point)
        grad_y = torch.autograd.grad(logp_y.sum(), proposal_point)[0].detach()

        accept_prob = torch.clamp(
            torch.exp(
                logp_y + h_(self.cache.point, proposal_point)
                - self.cache.logp - h_(proposal_point, self.cache.point)
            ),
            max=1
        ).detach()

        with torch.no_grad():

            signal_adaptation = torch.sqrt(accept_prob)[..., None] * (grad_y - self.cache.grad)

            phi_n = params.prec.permute(0, 2, 1) @ signal_adaptation[..., None]

            gramm_diag = phi_n.permute(0, 2, 1) @ phi_n

            if self.step_id == 0:
                r_1 = 1. / (1 + torch.sqrt(params.dampening / (params.dampening + gramm_diag)))
                shift = phi_n @ phi_n.permute(0, 2, 1)
                params.prec = 1. / params.dampening ** 0.5 * (
                    params.prec - shift * r_1 / (params.dampening + gramm_diag)
                )
            else:
                r_n = 1. / (1 + torch.sqrt(1 / (1 + gramm_diag)))
                shift = (params.prec @ phi_n) @ phi_n.permute(0, 2, 1)
                params.prec = params.prec - shift * r_n / (1 + gramm_diag)

            params.sigma = params.sigma * (
                1 + params.sigma_lr * (accept_prob[..., None] - params.target_acceptance)
            ) ** 0.5

            trace_prec = (params.prec[..., None, :] @ params.prec[..., None]).sum(dim=1)
            normalizer = (1. / self.cache.point.shape[-1]) * trace_prec
            params.sigma_prec = params.sigma[..., None] / normalizer ** 0.5

            mask = torch.rand_like(accept_prob) < accept_prob

            self.cache.point[mask] = proposal_point[mask]
            self.cache.logp[mask] = logp_y[mask]
            self.cache.grad[mask] = grad_y[mask]

        self.cache.point = self.cache.point.detach().requires_grad_()

        if self.cache.samples is None:
            self.cache.samples = self.cache.point.detach().clone()[None, ...]
        else:
            self.cache.samples = torch.cat([self.cache.samples, self.cache.point.detach().clone()[None, ...]], 0)

        self.step_id += 1


@dataclass
class FisherMALAVanilla(base_sampler.AlgorithmStoppingRule):
    sigma_burn_in_params: MALAParams
    sigma_burn_in_iter_count: int
    prec_burn_in_params: FisherMALAParams
    prec_burn_in_iter_count: int
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
                iteration=FisherMALAIter(params=params.copy_update(self.prec_burn_in_params)),
                iteration_count=self.prec_burn_in_iter_count,
            ),
            base_sampler.SampleBlock(
                iteration=FisherMALAIter(params=params.copy_update(self.prec_burn_in_params)),
                iteration_count=self.sample_iter_count,
                stopping_rule=self.stopping_rule,
                probe_period=self.probe_period,
            ),
        ])

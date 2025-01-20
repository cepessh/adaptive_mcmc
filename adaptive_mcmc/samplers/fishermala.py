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

    grad_v_img = torch.einsum("...ij,...j->...i", prec_factors[-1], grad_v)
    for factor in reversed(prec_factors[:-1]):
        grad_v_img = torch.einsum("...ij,...j->...i", factor, grad_v_img)

    return 0.5 * torch.einsum(
        "...i,...i->...",
        grad_v,
        z - v - 0.25 * torch.einsum("...i,...->...i", grad_v_img, sigma ** 2)
    )


@dataclass
class FisherMALAParams(base_sampler.Params):
    prec: Optional[Tensor] = None
    sigma: Union[Tensor, float] = 1.
    sigma_prec: Optional[Tensor] = None
    target_acceptance: float = 0.574
    sigma_lr: float = 0.015
    dampening: float = 10.


@dataclass
class AdaptiveCache(base_sampler.Cache):
    prec: Optional[Tensor] = None


@dataclass
class FisherMALAIter(base_sampler.MHIteration):
    params: FisherMALAParams = field(default_factory=FisherMALAParams)
    cache: AdaptiveCache = field(default_factory=AdaptiveCache)

    def init(self):
        super().init()
        self.step_id = 0

        # TODO: move adaptive parameters like prec from params to cache
        if self.params.prec is None:
            self.params.prec = torch.eye(self.cache.point.shape[-1]).repeat(*self.cache.point.shape[:-1], 1, 1)

        if isinstance(self.params.sigma, float):
            self.params.sigma_prec = Tensor([self.params.sigma]).repeat(*self.cache.point.shape[:-1], 1, 1)
            self.params.sigma = Tensor([self.params.sigma]).repeat(*self.cache.point.shape[:-1], 1)
        else:
            self.params.sigma = self.params.sigma.reshape(*self.cache.point.shape[:-1], 1)
            self.params.sigma_prec = self.params.sigma.clone()

    def _adapt(self, accept_prob: Tensor, grad_new: Tensor, ):
        with torch.no_grad():
            signal_adaptation = torch.sqrt(accept_prob).unsqueeze(-1) * (grad_new - self.cache.grad)

            phi_n = torch.einsum("...ji,...j->...i", self.params.prec, signal_adaptation)
            gramm_diag = torch.square(phi_n).sum(dim=-1, keepdim=True).unsqueeze(-1)

            if self.step_id == 0:
                r_1 = 1. / (1 + torch.sqrt(self.params.dampening / (self.params.dampening + gramm_diag)))
                shift = torch.einsum("...i,...j->...ij", phi_n, phi_n)
                self.params.prec = 1. / self.params.dampening ** 0.5 * (
                    self.params.prec - shift * r_1 / (self.params.dampening + gramm_diag)
                )
            else:
                r_n = 1. / (1 + torch.sqrt(1 / (1 + gramm_diag)))
                shift = torch.einsum(
                    "...i,...j->...ij",
                    torch.einsum("...ij,...j->...i", self.params.prec, phi_n),
                    phi_n,
                )
                self.params.prec = self.params.prec - shift * r_n / (1 + gramm_diag)

            self.params.sigma = self.params.sigma * (
                1 + self.params.sigma_lr * (accept_prob.unsqueeze(-1) - self.params.target_acceptance)
            ) ** 0.5

            trace_prec = torch.square(self.params.prec).sum(dim=[-2, -1]).unsqueeze(-1)
            normalizer = (1. / self.cache.point.shape[-1]) * trace_prec
            self.params.sigma_prec = self.params.sigma / normalizer ** 0.5

        self.cache.point = self.cache.point.detach().requires_grad_()

    def run(self):
        h_ = partial(h, prec_factors=[self.params.prec, self.params.prec.permute(0, 2, 1)],
                     target_dist=self.params.target_dist, sigma=self.params.sigma_prec.squeeze(-1))

        noise = self.params.proposal_dist.sample(self.cache.point.shape[:-1])

        grad_x_img = torch.einsum(
            "...ij,...j->...i",
            self.params.prec,
            torch.einsum("...ji,...j->...i", self.params.prec, self.cache.grad)
        )

        point_new = (
            self.cache.point + (
                0.5 * grad_x_img * self.params.sigma_prec ** 2
                + torch.einsum("...ij,...j->...i", self.params.prec, noise) * self.params.sigma_prec
            ).squeeze()
        )

        point_new = point_new.detach().requires_grad_()

        logp_new = self.params.target_dist.log_prob(point_new)
        grad_new = torch.autograd.grad(logp_new.sum(), point_new)[0].detach()

        accept_prob = torch.clamp(
            torch.exp(
                logp_new + h_(self.cache.point, point_new)
                - self.cache.logp - h_(point_new, self.cache.point)
            ),
            max=1.
        ).detach()

        self.MHStep(
            point_new=point_new,
            logp_new=logp_new,
            grad_new=grad_new,
            accept_prob=accept_prob,
        )
        self.collect_sample(self.cache.point.detach().clone())
        self.step_id += 1

        self._adapt(accept_prob=accept_prob, grad_new=grad_new)


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

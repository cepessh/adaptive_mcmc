from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from adaptive_mcmc.distributions.distribution import Distribution
from adaptive_mcmc.samplers import base_sampler
from adaptive_mcmc.samplers.mala import MALAIter
# , MALACommonParams


def h(z: Tensor, v: Tensor, sigma: Tensor, prec_factors: list[Tensor],
      target_dist: Union[Distribution, torchDist]) -> Tensor:
    """
    z, v (sample_count, n_dim)
    sigma (sample_count)
    prec_factors List[(sample_count, n_dim, n_dim)]
    """

    v = v.requires_grad_(True)
    logp_v = target_dist.log_prob(v)
    grad_v = torch.autograd.grad(logp_v.sum(), v)[0].detach()
    v = v.requires_grad_(False)

    with torch.no_grad():
        grad_v_img = torch.einsum("...ij,...j->...i", prec_factors[-1], grad_v)
        for factor in reversed(prec_factors[:-1]):
            grad_v_img = torch.einsum("...ij,...j->...i", factor, grad_v_img)

        return 0.5 * torch.einsum(
            "...i,...i->...",
            grad_v,
            z - v - 0.25 * torch.einsum("...i,...->...i", grad_v_img, sigma ** 2)
        )


@dataclass
class FisherMALACommonParams(base_sampler.CommonParams):
    sigma: Union[Tensor, float] = 1e0
    sigma_prec: Optional[Tensor] = None
    sigma_lr: float = 1.5e-2
    dampening: float = 10.


@dataclass
class FisherMALAFixedParams(base_sampler.FixedParams):
    target_acceptance: float = 0.574
    adapt_prec_required: bool = True


@dataclass
class AdaptiveCache(base_sampler.Cache):
    prec: Optional[Tensor] = None


@dataclass
class FisherMALAIter(base_sampler.MHIteration):
    common_params: FisherMALACommonParams = field(default_factory=FisherMALACommonParams)
    fixed_params: FisherMALAFixedParams = field(default_factory=FisherMALAFixedParams)
    cache: AdaptiveCache = field(default_factory=AdaptiveCache)

    def init(self, cache=None):
        super().init(cache)
        self.step_id = 0

        if self.cache.prec is None:
            self.cache.prec = torch.eye(
                self.cache.point.shape[-1],
                dtype=self.fixed_params.dtype,
                device=self.fixed_params.device,
            ).repeat(*self.cache.point.shape[:-1], 1, 1)

        if isinstance(self.common_params.sigma, float):
            self.common_params.sigma_prec = Tensor([self.common_params.sigma]).repeat(*self.cache.point.shape[:-1], 1, 1)
            self.common_params.sigma = Tensor([self.common_params.sigma]).repeat(*self.cache.point.shape[:-1], 1)
        else:
            self.common_params.sigma = self.common_params.sigma.reshape(*self.cache.point.shape[:-1], 1)
            self.common_params.sigma_prec = self.common_params.sigma.clone()

    def _adapt(self, accept_prob: Tensor, grad_new: Tensor, ):
        with torch.no_grad():
            signal_adaptation = torch.sqrt(accept_prob).unsqueeze(-1) * (grad_new - self.cache.grad)

            phi_n = torch.einsum("...ji,...j->...i", self.cache.prec, signal_adaptation)
            gramm_diag = torch.square(phi_n).sum(dim=-1, keepdim=True).unsqueeze(-1)

            if self.step_id == 0:
                r_1 = 1. / (1 + torch.sqrt(self.common_params.dampening / (self.common_params.dampening + gramm_diag)))
                shift = torch.einsum("...i,...j->...ij", phi_n, phi_n)
                self.cache.prec = 1. / self.common_params.dampening ** 0.5 * (
                    self.cache.prec - shift * r_1 / (self.common_params.dampening + gramm_diag)
                )
                print(self.cache.prec.abs().max())
            else:
                r_n = 1. / (1 + torch.sqrt(1 / (1 + gramm_diag)))
                shift = torch.einsum(
                    "...i,...j->...ij",
                    torch.einsum("...ij,...j->...i", self.cache.prec, phi_n),
                    phi_n,
                )
                self.cache.prec = self.cache.prec - shift * r_n / (1 + gramm_diag)
                print(self.cache.prec.abs().max())

            self.common_params.sigma = self.common_params.sigma * (
                1 + self.common_params.sigma_lr * (
                    accept_prob.unsqueeze(-1) - self.fixed_params.target_acceptance
                )
            )

            trace_prec = torch.square(self.cache.prec).sum(dim=[-2, -1]).unsqueeze(-1)
            normalizer = (1. / self.cache.point.shape[-1]) * trace_prec
            self.common_params.sigma_prec = self.common_params.sigma / normalizer ** 0.5

        # self.cache.point = self.cache.point.detach().requires_grad_()

    def run(self):
        h_ = partial(h, prec_factors=[self.cache.prec, self.cache.prec.permute(0, 2, 1)],
                     target_dist=self.common_params.target_dist, sigma=self.common_params.sigma_prec.squeeze(-1))

        with torch.no_grad():
            noise = self.common_params.proposal_dist.sample(self.cache.point.shape[:-1])

            grad_x_img = torch.einsum(
                "...ij,...j->...i",
                self.cache.prec,
                torch.einsum("...ji,...j->...i", self.cache.prec, self.cache.grad)
            )

            point_new = (
                self.cache.point + (
                    0.5 * grad_x_img * self.common_params.sigma_prec ** 2
                    + torch.einsum("...ij,...j->...i", self.cache.prec, noise) * self.common_params.sigma_prec
                ).squeeze()
            )

        point_new = point_new.detach().requires_grad_()
        logp_new = self.common_params.target_dist.log_prob(point_new)
        grad_new = torch.autograd.grad(logp_new.sum(), point_new)[0].detach()
        point_new = point_new.requires_grad_(False)

        h_val = h_(self.cache.point, point_new) - h_(point_new, self.cache.point)

        with torch.no_grad():
            accept_prob = torch.clamp(
                torch.exp(
                    logp_new + h_val - self.cache.logp
                ),
                max=1.
            )

        self.MHStep(
            point_new=point_new,
            logp_new=logp_new,
            grad_new=grad_new,
            accept_prob=accept_prob,
        )

        if self.collect_required:
            self.collect_sample(self.cache.point.detach().clone())

        self.step_id += 1

        if self.fixed_params.adapt_prec_required:
            self._adapt(accept_prob=accept_prob, grad_new=grad_new)


@dataclass
class FisherMALAVanilla(base_sampler.AlgorithmStoppingRule):
    sigma_burn_in_iter_count: int
    prec_burn_in_iter_count: int
    sample_iter_count: int
    probe_period: int
    stopping_rule: Callable
    common_params: Optional[FisherMALACommonParams] = None
    fixed_params: FisherMALAFixedParams = field(default_factory=FisherMALAFixedParams)

    def load_params(self, common_params: base_sampler.CommonParams):
        self.pipeline = base_sampler.Pipeline([
            base_sampler.SampleBlock(
                iteration=MALAIter(
                    common_params=common_params,
                ),
                iteration_count=self.sigma_burn_in_iter_count,
            ),
            base_sampler.SampleBlock(
                iteration=FisherMALAIter(
                    common_params=common_params,
                ),
                iteration_count=self.prec_burn_in_iter_count,
            ),
            base_sampler.SampleBlock(
                iteration=FisherMALAIter(
                    common_params=common_params,
                    collect_required=True,
                ),
                iteration_count=self.sample_iter_count,
                stopping_rule=self.stopping_rule,
                probe_period=self.probe_period,
            ),
        ])

        self.pipeline.sample_blocks[-1].iteration.fixed_params.adapt_prec_required = False

        def make_callback(block: base_sampler.SampleBlock):
            def callback():
                print(f"step_size={block.iteration.common_params.sigma.mean()}")
            return callback

        for block in self.pipeline.sample_blocks:
            block.callback = make_callback(block)

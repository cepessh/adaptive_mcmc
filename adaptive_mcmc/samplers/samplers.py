from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from distributions import SamplableDistribution, GaussianMixture, Distribution
from samplers import base_sampler


@dataclass
class MALAIter(base_sampler.Iteration):
    def init(self):
        super().init()

        if isinstance(self.cache.params.meta["sigma"], float):
            self.cache.params.meta["sigma"] = Tensor([self.cache.params.meta["sigma"]]).repeat(
                *self.cache.point.shape[:-1], 1)
        else:
            self.cache.params.meta["sigma"] = self.cache.params.meta["sigma"].reshape(
                *self.cache.point.shape[:-1], 1)

    def run(self):
        params = self.cache.params

        noise = params.proposal_dist.sample(self.cache.point.shape[:-1])

        proposal_point = (
            self.cache.point + 
            0.5 * self.cache.grad * params.meta["sigma"] ** 2 + 
            noise * params.meta["sigma"]
        ).detach().requires_grad_()

        logp_y = params.target_dist.log_prob(proposal_point)
        grad_y = torch.autograd.grad(logp_y.sum(), proposal_point)[0].detach()

        with torch.no_grad():
            log_qyx = params.proposal_dist.log_prob(noise)
            log_qxy = params.proposal_dist.log_prob(
                (self.cache.point - proposal_point - 
                0.5 * params.meta["sigma"] ** 2 * grad_y) / params.meta["sigma"]
            )
            
            accept_prob = torch.clamp((logp_y + log_qxy - self.cache.logp - log_qyx).exp(), max=1).detach()
            mask = torch.rand_like(accept_prob) < accept_prob

            self.cache.point[mask] = proposal_point[mask]
            self.cache.logp[mask] = logp_y[mask]
            self.cache.grad[mask] = grad_y[mask]

            params.meta["sigma"] = params.meta["sigma"] * (
                1 + params.meta["sigma_lr"] * (
                    accept_prob[..., None] - params.meta["target_acceptance"]
                )
            ) ** 0.5
            
        if self.cache.samples is None:
            self.cache.samples = self.cache.point.detach().clone()[None, ...]
        else:
            self.cache.samples = torch.cat([self.cache.samples, self.cache.point.detach().clone()[None, ...]], 0)
        # print(self.cache.params.meta["sigma"])


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

    return 0.5 * (grad_v[:, None, :] @ 
                  (z - v - 0.25 * grad_v_img * sigma[..., None] ** 2)[..., None]
                 ).squeeze()


@dataclass
class FisherMALAIter(base_sampler.Iteration):

    def init(self):
        super().init()
        self.step_id = 0

        if "prec" not in self.cache.params.meta:
            self.cache.params.meta["prec"] = torch.eye(
                self.cache.point.shape[-1]).repeat(*self.cache.point.shape[:-1], 1, 1)

        """
        sigma_prec: (chain_count, 1, 1)
        """
        if isinstance(self.cache.params.meta["sigma"], float):
            self.cache.params.meta["sigma_prec"] = Tensor([self.cache.params.meta["sigma"]]).repeat(
                *self.cache.point.shape[:-1], 1, 1)
            self.cache.params.meta["sigma"] = Tensor([self.cache.params.meta["sigma"]]).repeat(
                *self.cache.point.shape[:-1], 1)
        else:
            self.cache.params.meta["sigma"] = self.cache.params.meta["sigma"].reshape(*self.cache.point.shape[:-1], 1)
            self.cache.params.meta["sigma_prec"] = self.cache.params.meta["sigma"]
            while len(self.cache.params.meta["sigma_prec"].shape) < 3:
                self.cache.params.meta["sigma_prec"] = self.cache.params.meta["sigma_prec"][..., None]

    def run(self):        
        params = self.cache.params

        h_ = partial(h, prec_factors=[params.meta["prec"], params.meta["prec"].permute(0, 2, 1)],
                     target_dist=params.target_dist, sigma=params.meta["sigma_prec"].squeeze())

        noise = params.proposal_dist.sample(self.cache.point.shape[:-1])

        grad_x_img = self.cache.grad[..., None]
        grad_x_img = params.meta["prec"] @ (params.meta["prec"].permute(0, 2, 1) @ grad_x_img)

        proposal_point = self.cache.point + (
            0.5 * grad_x_img * params.meta["sigma_prec"] ** 2 + 
            params.meta["prec"] @ noise[..., None] * params.meta["sigma_prec"]
        ).squeeze()
        proposal_point = proposal_point.detach().requires_grad_()

        logp_y = params.target_dist.log_prob(proposal_point)
        grad_y = torch.autograd.grad(logp_y.sum(), proposal_point)[0].detach()

        accept_prob = torch.clamp(
            torch.exp(
                logp_y + h_(self.cache.point, proposal_point) - 
                self.cache.logp - h_(proposal_point, self.cache.point)
            ),
            max=1
        ).detach()

        with torch.no_grad():

            signal_adaptation = torch.sqrt(accept_prob)[..., None] * (grad_y - self.cache.grad)

            phi_n = params.meta["prec"].permute(0, 2, 1) @ signal_adaptation[..., None]

            gramm_diag = phi_n.permute(0, 2, 1) @ phi_n

            if self.step_id == 0:
                r_1 = 1. / (1 + torch.sqrt(params.meta["damping"] / (params.meta["damping"] + gramm_diag)))
                shift = phi_n @ phi_n.permute(0, 2, 1)
                params.meta["prec"] = 1. / params.meta["damping"] ** 0.5 * (
                    params.meta["prec"] - shift * r_1 / (params.meta["damping"] + gramm_diag)
                )
            else:
                r_n = 1. / (1 + torch.sqrt(1 / (1 + gramm_diag)))
                shift = (params.meta["prec"] @ phi_n) @ phi_n.permute(0, 2, 1)
                params.meta["prec"] = params.meta["prec"] - shift * r_n / (1 + gramm_diag)

            params.meta["sigma"] = params.meta["sigma"] * (
                1 + params.meta["sigma_lr"] * (accept_prob[..., None] - params.meta["target_acceptance"])
            ) ** 0.5

            trace_prec = (params.meta["prec"][..., None, :] @ params.meta["prec"][..., None]).sum(dim=1)
            normalizer = (1. / self.cache.point.shape[-1]) * trace_prec
            params.meta["sigma_prec"] = params.meta["sigma"][..., None] / normalizer ** 0.5

            mask = torch.rand_like(accept_prob) < accept_prob
            mask = mask

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
    sigma_burn_in_params: dict
    sigma_burn_in_iter_count: int
    prec_burn_in_params: dict
    prec_burn_in_iter_count: int
    sample_iter_count: int
    probe_period: int
    stopping_rule: Callable
    name = "FisherMALA"

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

    def run(self):
        print("Running", self.name)
        self.pipeline.run()


@dataclass
class MALAVanilla(base_sampler.AlgorithmStoppingRule):
    sigma_burn_in_params: dict
    sigma_burn_in_iter_count: int
    sample_iter_count: int
    probe_period: int
    stopping_rule: Callable
    name = "MALA"

    def load_params(self, params: base_sampler.Params):
        self.pipeline = base_sampler.Pipeline(
            [
                base_sampler.SampleBlock(
                    iteration=MALAIter(base_sampler.Cache(
                        params.update_meta(self.sigma_burn_in_params))),
                    iteration_count=self.sigma_burn_in_iter_count,
                ),
                base_sampler.SampleBlock(
                    iteration=MALAIter(base_sampler.Cache(
                        params.update_meta(self.sigma_burn_in_params))),
                    iteration_count=self.sample_iter_count,
                    stopping_rule=self.stopping_rule,
                    probe_period=self.probe_period,
                ),
            ]
        )

    def run(self):
        print("Running", self.name)
        self.pipeline.run()
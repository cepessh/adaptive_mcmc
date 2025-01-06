from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from adaptive_mcmc.distributions.distribution import Distribution
from adaptive_mcmc.samplers import base_sampler


@dataclass
class HMCParams(base_sampler.Params):
    prec: Optional[Tensor] = None
    lf_step_size: float = 1e-3
    lf_step_count: int = 5
    target_acceptance: float = 0.65


@dataclass
class Leapfrog():
    def step(self, q_prev: Tensor, p_prev: Tensor, gradq_prev: Tensor,
             Minv: Tensor, target_dist: Union[Distribution, torchDist],
             step_size: float = 1e-3) -> dict:

        p_half = p_prev + 0.5 * step_size * gradq_prev

        q_next = q_prev + step_size * torch.bmm(Minv, p_half.unsqueeze(-1)).squeeze(-1)
        q_next = q_next.detach().requires_grad_()

        logq_next = target_dist.log_prob(q_next)
        gradq_next = torch.autograd.grad(logq_next.sum(), q_next)[0].detach()

        p_next = p_half + 0.5 * step_size * gradq_next

        return {
            "q": q_next,
            "p": p_next,
            "gradq": gradq_next,
        }

    def run(self, q: Tensor, p: Tensor, gradq: Tensor, Minv: Tensor,
            target_dist: Union[Distribution, torchDist],
            step_count: int = 5, step_size: float = 1e-3) -> List[dict]:

        ret = {"q": q, "p": p, "gradq": gradq}
        trajectory = [ret]

        for step in range(step_count):
            ret = self.step(q_prev=ret["q"], p_prev=ret["p"], gradq_prev=ret["gradq"],
                            Minv=Minv, target_dist=target_dist, step_size=step_size)
            trajectory.append(ret)

        return trajectory


@dataclass
class HMCIter(base_sampler.Iteration):
    params: HMCParams = field(default_factory=HMCParams)
    lf_intergrator: Leapfrog = field(default_factory=Leapfrog)

    def init(self):
        super().init()
        self.step_id = 0

        if self.params.prec is None:
            self.params.prec = torch.eye(self.cache.point.shape[-1]).repeat(*self.cache.point.shape[:-1], 1, 1)

    def run(self):
        params = self.params

        noise = params.proposal_dist.sample(self.cache.point.shape[:-1])
        p = torch.linalg.solve(self.params.prec, noise)

        Minv = torch.bmm(self.params.prec, self.params.prec.permute(0, 2, 1))

        trajectory = self.lf_intergrator.run(
            q=self.cache.point,
            p=p,
            gradq=self.cache.grad,
            Minv=Minv,
            target_dist=params.target_dist,
            step_count=params.lf_step_count,
            step_size=params.lf_step_size,
        )

        point_new = trajectory[-1]["q"]
        logp_new = params.target_dist.log_prob(point_new)
        grad_new = trajectory[-1]["gradq"]
        p_new = trajectory[-1]["p"]

        accept_prob = torch.clamp(
            torch.exp(
                logp_new - self.cache.logp
                - 0.5 * (
                    torch.bmm(torch.bmm(p_new.unsqueeze(1), Minv), p_new.unsqueeze(-1)).squeeze()
                    - torch.bmm(torch.bmm(p.unsqueeze(1), Minv), p.unsqueeze(-1)).squeeze()
                )
            ),
            max=1
        ).detach()

        with torch.no_grad():
            mask = torch.rand_like(accept_prob) < accept_prob

            self.cache.point[mask] = point_new[mask]
            self.cache.logp[mask] = logp_new[mask]
            self.cache.grad[mask] = grad_new[mask]

        self.cache.point = self.cache.point.detach().requires_grad_()

        if self.cache.samples is None:
            self.cache.samples = self.cache.point.detach().clone()[None, ...]
        else:
            self.cache.samples = torch.cat([self.cache.samples, self.cache.point.detach().clone()[None, ...]], 0)

        self.step_id += 1


@dataclass
class HMCAdaptiveParams(HMCParams):
    max_truncation_level: int = 100
    spectral_normalization_decay: float = 0.9
    learning_rate: float = 5e-3

    entropy_weight: float = 1.
    entropy_weight_min: float = 1e-2
    entropy_weight_max: float = 1e2
    entropy_weight_adaptive_rate: float = 1e-2

    penalty_func: Callable = None
    penalty_weight: float = 1.
    penalty_weight_min: float = 1e-3
    penalty_weight_max: float = 1e5
    penalty_weight_adaptive_rate: float = 1.


@dataclass
class HMCAdaptiveIter(base_sampler.Iteration):
    params: HMCAdaptiveParams = field(default_factory=HMCAdaptiveParams)
    lf_intergrator: Leapfrog = field(default_factory=Leapfrog)

    def init(self):
        super().init()
        self.step_id = 0

        if self.params.prec is None:
            self.params.prec = torch.eye(self.cache.point.shape[-1]).repeat(*self.cache.point.shape[:-1], 1, 1)

    def run(self):
        params = self.params

        noise = params.proposal_dist.sample(self.cache.point.shape[:-1])
        p = torch.linalg.solve(self.params.prec, noise)

        Minv = torch.bmm(self.params.prec, self.params.prec.permute(0, 2, 1))

        trajectory = self.lf_intergrator.run(
            q=self.cache.point,
            p=p,
            gradq=self.cache.grad,
            Minv=Minv,
            target_dist=params.target_dist,
            step_count=params.lf_step_count,
            step_size=params.lf_step_size,
        )

        point_new = trajectory[-1]["q"]
        logp_new = params.target_dist.log_prob(point_new)
        grad_new = trajectory[-1]["gradq"]
        p_new = trajectory[-1]["p"]

        accept_prob = torch.clamp(
            torch.exp(
                logp_new - self.cache.logp
                - 0.5 * (
                    torch.bmm(torch.bmm(p_new.unsqueeze(1), Minv), p_new.unsqueeze(-1)).squeeze()
                    - torch.bmm(torch.bmm(p.unsqueeze(1), Minv), p.unsqueeze(-1)).squeeze()
                )
            ),
            max=1
        ).detach()

        with torch.no_grad():
            mask = torch.rand_like(accept_prob) < accept_prob

            self.cache.point[mask] = point_new[mask]
            self.cache.logp[mask] = logp_new[mask]
            self.cache.grad[mask] = grad_new[mask]

        self.cache.point = self.cache.point.detach().requires_grad_()

        if self.cache.samples is None:
            self.cache.samples = self.cache.point.detach().clone()[None, ...]
        else:
            self.cache.samples = torch.cat([self.cache.samples, self.cache.point.detach().clone()[None, ...]], 0)

        self.step_id += 1


@dataclass
class HMCVanilla(base_sampler.AlgorithmStoppingRule):
    params: HMCParams
    burn_in_iter_count: int
    sample_iter_count: int
    probe_period: int
    stopping_rule: Callable

    def load_params(self, params: base_sampler.Params):
        self.pipeline = base_sampler.Pipeline([
            base_sampler.SampleBlock(
                iteration=HMCIter(params=params.copy_update(params)),
                iteration_count=self.burn_in_iter_count,
            ),
            base_sampler.SampleBlock(
                iteration=HMCIter(params=params.copy_update(params)),
                iteration_count=self.sample_iter_count,
                stopping_rule=self.stopping_rule,
                probe_period=self.probe_period,
            ),
        ])

from contextlib import nullcontext
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
    Minv: Optional[Tensor] = None
    lf_step_size: float = 1e-2
    lf_step_size_adaptive_rate: float = 1e-2
    lf_step_count: int = 5
    target_acceptance: float = 0.65
    stop_grad: bool = False
    no_grad: bool = True


class Leapfrog():
    def step(self, q_prev: Tensor, p_prev: Tensor, gradq_prev: Tensor,
             Minv: Tensor, target_dist: Union[Distribution, torchDist],
             step_size: float = 1e-3, stop_grad: bool = True, no_grad: bool = False) -> dict:

        context = torch.no_grad() if no_grad else nullcontext()

        with context:
            p_half = p_prev + 0.5 * step_size * gradq_prev
            q_next = q_prev + step_size * torch.einsum("...ij,...j->...i", Minv, p_half)

        if stop_grad:
            q_next = q_next.detach().requires_grad_()
        else:
            q_next = q_next.requires_grad_()

        logq_next = target_dist.log_prob(q_next)
        gradq_next = torch.autograd.grad(logq_next.sum(), q_next, retain_graph=not no_grad)[0]

        if stop_grad:
            gradq_next = gradq_next.detach().requires_grad_()

        with context:
            p_next = p_half + 0.5 * step_size * gradq_next

        return {
            "q": q_next,
            "p": p_next,
            "gradq": gradq_next,
        }

    def run(self, q: Tensor, p: Tensor, gradq: Tensor, Minv: Tensor,
            target_dist: Union[Distribution, torchDist],
            step_count: int = 5, step_size: float = 1e-3,
            stop_grad: bool = True, no_grad: bool = False, keep_trajectory=False) -> Optional[List[dict]]:

        ret = {"q": q, "p": p, "gradq": gradq}
        trajectory = [ret]

        for step in range(step_count):
            ret = self.step(
                q_prev=ret["q"], p_prev=ret["p"], gradq_prev=ret["gradq"],
                Minv=Minv, target_dist=target_dist, step_size=step_size,
                stop_grad=stop_grad, no_grad=no_grad,
            )
            if any(torch.isnan(tensor).any().item() for tensor in [ret["q"], ret["p"], ret["gradq"]]):
                return None

            trajectory.append(ret)

        return trajectory


@dataclass
class HMCIter(base_sampler.MHIteration):
    params: HMCParams = field(default_factory=HMCParams)
    lf_intergrator: Leapfrog = field(default_factory=Leapfrog)

    def init(self):
        super().init()
        self.step_id = 0

        if hasattr(self.cache, "prec"):
            self.params.prec = self.cache.prec
            self.params.Minv = torch.einsum("...ij,...kj->...ik", self.cache.prec, self.cache.prec)
        elif self.params.prec is None:
            self.params.prec = torch.eye(self.cache.point.shape[-1], device=self.params.device).repeat(*self.cache.point.shape[:-1], 1, 1)
            self.params.Minv = self.params.prec

    def _run_iter(self, prec: Tensor, Minv: Tensor) -> dict:
        trajectory = None
        while trajectory is None:
            noise = self.params.proposal_dist.sample(self.cache.point.shape[:-1])
            p = torch.linalg.solve_triangular(
                torch.einsum("...ij->...ji", prec),
                noise.unsqueeze(-1),
                upper=True,
            ).squeeze(-1)

            trajectory = self.lf_intergrator.run(
                q=self.cache.point,
                p=p,
                gradq=self.cache.grad,
                Minv=Minv,
                target_dist=self.params.target_dist,
                step_count=self.params.lf_step_count,
                step_size=self.params.lf_step_size,
                stop_grad=self.params.stop_grad,
                no_grad=self.params.no_grad,
            )

            if trajectory is None:
                self.params.lf_step_size *= 1 - self.params.lf_step_size_adaptive_rate

        context = torch.no_grad() if self.params.no_grad else nullcontext()

        with context:
            self.trajectory = trajectory

            point_new = trajectory[-1]["q"]
            logp_new = self.params.target_dist.log_prob(point_new)
            grad_new = trajectory[-1]["gradq"]
            p_new = trajectory[-1]["p"]

            energy_error = (
                logp_new - self.cache.logp.detach()
                - 0.5 * (
                    torch.einsum("...i,...ij,...j->...", p_new, Minv, p_new)
                    - torch.einsum("...i,...ij,...j->...", p, Minv, p)
                )
            )

            accept_prob = torch.clamp(torch.exp(energy_error), max=1)

            self.MHStep(
                point_new=point_new,
                logp_new=logp_new,
                grad_new=grad_new,
                accept_prob=accept_prob,
            )
            self.collect_sample(self.cache.point.detach().clone())
            self.step_id += 1

        return {
            "energy_error": energy_error,
            "accept_prob": accept_prob,
            "trajectory": trajectory,
            "grad_new": grad_new,
        }

    def run(self):
        self._run_iter(prec=self.params.prec, Minv=self.params.Minv)


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

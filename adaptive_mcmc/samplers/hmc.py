from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from adaptive_mcmc.distributions.distribution import Distribution
from adaptive_mcmc.samplers import base_sampler
from adaptive_mcmc.linalg import fastmv


class PrecType(str, Enum):
    Dense = "dense"
    TRIDIAG = "tridiag"
    NONE = "none"


@dataclass
class HMCCommonParams(base_sampler.CommonParams):
    prec: Optional[Tensor] = None
    Minv: Optional[Tensor] = None
    lf_step_size: Union[float, Tensor] = 1e-2
    lf_step_size_max: Union[float, Tensor] = 1e-2
    lf_step_size_min: float = 1e-2
    lf_step_size_adaptive_rate: float = 1e-2
    lf_step_size_adaptive_bad_traj_period: int = 5
    lf_step_count: int = 3


@dataclass
class HMCFixedParams(base_sampler.FixedParams):
    target_acceptance: float = 0.65
    stop_grad: bool = False
    no_grad: bool = True
    prec_type: PrecType = PrecType.NONE


@dataclass
class HMCCache(base_sampler.Cache):
    pass


class Leapfrog():
    def step(self, q_prev: Tensor, p_prev: Tensor, gradq_prev: Tensor,
             matvec_minv: Callable, target_dist: Union[Distribution, torchDist],
             step_size: float = 1e-3, stop_grad: bool = True, no_grad: bool = False) -> dict:

        context = torch.no_grad() if no_grad else nullcontext()

        with context:
            p_half = p_prev + 0.5 * step_size * gradq_prev
            q_next = q_prev + step_size * matvec_minv(p_half)

        q_next = q_next.requires_grad_(True)

        logq_next = target_dist.log_prob(q_next)
        gradq_next = torch.autograd.grad(logq_next.sum(), q_next, retain_graph=not no_grad)[0]

        if no_grad:
            q_next = q_next.requires_grad_(False)
            gradq_next = gradq_next.detach()
            q_next = q_next.detach()

        with context:
            p_next = p_half + 0.5 * step_size * gradq_next

        return {
            "q": q_next,
            "p": p_next,
            "gradq": gradq_next,
        }

    def run(self, q: Tensor, p: Tensor, gradq: Tensor, matvec_minv: Callable,
            target_dist: Union[Distribution, torchDist],
            step_count: int = 5, step_size: float = 1e-3,
            stop_grad: bool = True, no_grad: bool = False, keep_trajectory=False) -> Optional[List[dict]]:

        ret = {"q": q, "p": p, "gradq": gradq}
        trajectory = [ret]

        for step in range(step_count):
            try:
                ret = self.step(
                    q_prev=ret["q"], p_prev=ret["p"], gradq_prev=ret["gradq"],
                    matvec_minv=matvec_minv, target_dist=target_dist, step_size=step_size,
                    stop_grad=stop_grad, no_grad=no_grad,
                )

                with torch.no_grad():
                    mask = (~torch.isfinite(ret["q"])).any(dim=-1)
                    mask |= (~torch.isfinite(ret["p"])).any(dim=-1)
                    mask |= (~torch.isfinite(ret["gradq"])).any(dim=-1)

                    if mask.any():
                        return None, mask

            # happens when NaNs are in ret self.step computation
            except ValueError:
                # print("val err")
                return None, None

            trajectory.append(ret)

        return trajectory, None


@dataclass
class HMCIter(base_sampler.MHIteration):
    cache: HMCCache = field(default_factory=HMCCache)
    common_params: HMCCommonParams = field(default_factory=HMCCommonParams)
    fixed_params: HMCFixedParams = field(default_factory=HMCFixedParams)
    lf_intergrator: Leapfrog = field(default_factory=Leapfrog)
    adapt_step_size_required: bool = False

    def init(self, cache=None):
        super().init(cache=cache)
        self.step_id = 0

        if not hasattr(self.cache, 'prec'):
            self.cache.prec = None

        if self.common_params.prec is not None:
            self.cache.prec = self.common_params.prec
        else:
            if (
                self.fixed_params.prec_type == PrecType.TRIDIAG
                and hasattr(self.cache, 'prec_params')
                and self.cache.prec_params is not None
            ):
                self.cache.prec = self.cache.prec_params.params
            else:
                pass

        if self.cache.prec is not None or self.fixed_params.prec_type == PrecType.NONE:
            self.make_matvecs(self.cache.prec)

        if isinstance(self.common_params.lf_step_size, float):
            self.common_params.lf_step_size = torch.full(
                (self.cache.point.shape[0], 1),
                self.common_params.lf_step_size,
            )

        if isinstance(self.common_params.lf_step_size_max, float):
            self.common_params.lf_step_size_max = torch.full(
                (self.cache.point.shape[0], 1),
                self.common_params.lf_step_size_max,
            )

    def make_matvecs(self, prec: Tensor):
        if self.fixed_params.prec_type == PrecType.TRIDIAG:
            d = (prec.shape[1] + 1) // 2
            diag = torch.exp(prec[:, :d])
            lower = prec[:, d:]

        def matvec_minv(v: Tensor) -> Tensor:
            return matvec_c(matvec_ct(v))

        def matvec_ct(v: Tensor) -> Tensor:
            if self.fixed_params.prec_type == PrecType.NONE:
                return v
            elif self.fixed_params.prec_type == PrecType.TRIDIAG:
                return fastmv.tridiag_matmul(v, diag=diag, upper=lower)
            else:
                if v.dim() == 2:
                    # (b, d) -> (b, d, 1) -> (b, d, 1) -> (b, d)
                    return torch.bmm(self.prec.transpose(-2, -1), v.unsqueeze(-1)).squeeze(-1)
                elif v.dim() == 3:
                    return torch.matmul(v, self.prec)
                # return torch.einsum("...ji,...j->...i", prec, v)

        def matvec_c(v: Tensor) -> Tensor:
            if self.fixed_params.prec_type == PrecType.NONE:
                return v
            elif self.fixed_params.prec_type == PrecType.TRIDIAG:
                return fastmv.tridiag_matmul(v, diag=diag, lower=lower)
            else:
                if v.dim() == 2:
                    # (b, d) -> (b, d, 1) -> (b, d, 1) -> (b, d)
                    return torch.bmm(self.prec, v.unsqueeze(-1)).squeeze(-1)
                elif v.dim() == 3:
                    return torch.matmul(v, self.prec.transpose(-2, -1))
                # return torch.einsum("...ij,...j->...i", prec, v)

        self.matvec_minv = matvec_minv
        self.matvec_ct = matvec_ct
        self.matvec_c = matvec_c

    def _eval_trajectory(self, noise: Tensor, prec: Tensor, context):
        trajectory = None
        d = noise.shape[-1]

        with context:
            if self.fixed_params.prec_type == PrecType.NONE:
                p = noise
            elif self.fixed_params.prec_type == PrecType.TRIDIAG:
                p = fastmv.bidiag_solve_jit(noise, torch.exp(prec[:, :d]), prec[:, d:])
            else:
                p = torch.linalg.solve_triangular(
                    prec.transpose(-2, -1),
                    noise.unsqueeze(-1),
                    upper=True,
                ).squeeze(-1)

        while trajectory is None:
            trajectory, mask = self.lf_intergrator.run(
                q=self.cache.point,
                p=p,
                gradq=self.cache.grad,
                matvec_minv=self.matvec_minv,
                target_dist=self.common_params.target_dist,
                step_count=self.common_params.lf_step_count,
                step_size=self.common_params.lf_step_size,
                stop_grad=self.fixed_params.stop_grad,
                no_grad=self.fixed_params.no_grad,
            )

            if trajectory is None:
                self.cache.broken_trajectory_count += 1
                with torch.no_grad():
                    self.common_params.lf_step_size = torch.clamp(
                        self.common_params.lf_step_size * (
                            1 - self.common_params.lf_step_size_adaptive_rate * (1 if mask is None else mask.unsqueeze(-1))
                        ),
                        min=self.common_params.lf_step_size_min,
                    )

                    if self.cache.broken_trajectory_count % self.common_params.lf_step_size_adaptive_bad_traj_period == 0:
                        self.common_params.lf_step_size_max = torch.clamp(
                            self.common_params.lf_step_size_max * (1 - self.common_params.lf_step_size_adaptive_rate * (
                                1 if mask is None else mask.unsqueeze(-1)
                            )),
                            min=self.common_params.lf_step_size_min,
                        )

        return trajectory[-1]["q"], trajectory

    def _run_iter(self, prec: Tensor, noise_grad: bool = False) -> Dict[str, Tensor]:
        context = torch.no_grad() if self.fixed_params.no_grad else nullcontext()

        noise = self.common_params.proposal_dist.sample(self.cache.point.shape[:-1]).requires_grad_(noise_grad)

        _, trajectory = self._eval_trajectory(noise, prec, context)

        with context:
            trajectory[0]["noise"] = noise

            p = trajectory[0]["p"]

            point_new = trajectory[-1]["q"]
            logp_new = self.common_params.target_dist.log_prob(point_new)
            grad_new = trajectory[-1]["gradq"]
            p_new = trajectory[-1]["p"]

            new_norm = self.matvec_ct(p_new).pow(2).sum(dim=-1)
            old_norm = self.matvec_ct(p).pow(2).sum(dim=-1)

            energy_error = (
                -logp_new + self.cache.logp.detach()
                + 0.5 * (new_norm - old_norm)
            )

            accept_prob = torch.exp(torch.clamp(-energy_error, max=0))

            if self.adapt_step_size_required:
                with torch.no_grad():
                    self.common_params.lf_step_size = torch.min(
                        torch.clamp(
                            self.common_params.lf_step_size * (
                                1 + self.common_params.lf_step_size_adaptive_rate * (
                                    accept_prob.unsqueeze(-1) - self.fixed_params.target_acceptance
                                )
                            ),
                            min=self.common_params.lf_step_size_min,
                        ),
                        self.common_params.lf_step_size_max,
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

        return {
            "energy_error": energy_error,
            "accept_prob": accept_prob,
            "trajectory": trajectory,
        }

    def run(self):
        self._run_iter(prec=self.cache.prec)


@dataclass
class HMCVanilla(base_sampler.AlgorithmStoppingRule):
    step_size_burn_in_iter_count: int
    burn_in_iter_count: int
    sample_iter_count: int
    probe_period: int
    stopping_rule: Callable
    common_params: HMCCommonParams = field(default_factory=HMCCommonParams)
    fixed_params: HMCFixedParams = field(default_factory=HMCFixedParams)

    def load_params(
        self,
        common_params: base_sampler.CommonParams,
    ):
        self.pipeline = base_sampler.Pipeline([
            base_sampler.SampleBlock(
                iteration=HMCIter(
                    common_params=common_params,
                    fixed_params=self.fixed_params,
                    adapt_step_size_required=True,
                ),
                iteration_count=self.step_size_burn_in_iter_count,
            ),
            base_sampler.SampleBlock(
                iteration=HMCIter(
                    common_params=common_params,
                    fixed_params=self.fixed_params,
                    adapt_step_size_required=False,
                ),
                iteration_count=self.burn_in_iter_count,
            ),
            base_sampler.SampleBlock(
                iteration=HMCIter(
                    common_params=common_params,
                    fixed_params=self.fixed_params,
                    adapt_step_size_required=True,
                    collect_required=True,
                ),
                iteration_count=self.sample_iter_count,
                stopping_rule=self.stopping_rule,
                probe_period=self.probe_period,
            ),
        ])

        self.adjust_step_exploration()
        self.add_callbacks()

    def adjust_step_exploration(self):
        self.pipeline.sample_blocks[0].iteration.common_params = self.pipeline.sample_blocks[0].iteration.common_params.copy()
        self.pipeline.sample_blocks[0].iteration.common_params.lf_step_size_adaptive_rate *= 5

    def add_callbacks(self):
        def make_callback(block: base_sampler.SampleBlock):
            def callback():
                print(f"step_size={block.iteration.common_params.lf_step_size.mean()}")
                print(f"max_step_size={block.iteration.common_params.lf_step_size_max.mean()}")
                print(f"broken_traj={block.iteration.cache.broken_trajectory_count}")
            return callback

        for block in self.pipeline.sample_blocks:
            block.callback = make_callback(block)

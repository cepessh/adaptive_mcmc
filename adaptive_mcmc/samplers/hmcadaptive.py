from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Type

import numpy as np
import torch
from torch import Tensor

from adaptive_mcmc.samplers import base_sampler
from adaptive_mcmc.samplers.hmc import HMCParams, HMCIter, Leapfrog


def default_penalty_fn(x: Tensor) -> Tensor:
    delta_1 = 0.75
    delta_2 = 1.75

    absx = torch.abs(x)
    penalty = torch.zeros_like(x)

    condition_2 = (absx >= delta_1) & (absx < delta_2)
    penalty[condition_2] = (x[condition_2] - delta_1) ** 2

    condition_3 = absx >= delta_2
    penalty[condition_3] = (delta_2 - delta_1) ** 2 + (delta_2 - delta_1) * (absx[condition_3] - delta_2)

    return penalty


@dataclass
class HMCAdaptiveParams(HMCParams):
    truncation_level_prob: float = 0.5
    min_truncation_level: int = 2
    spectral_normalization_decay: float = 0.99
    learning_rate: float = 5e-3

    entropy_weight: float = 1.
    entropy_weight_min: float = 1e-3
    entropy_weight_max: float = 1e2
    entropy_weight_adaptive_rate: float = 1e-2

    penalty_func: Callable = default_penalty_fn
    penalty_weight: float = 1.
    penalty_weight_min: float = 1.
    penalty_weight_max: float = 1e5
    penalty_weight_adaptive_rate: float = 1e-2

    stop_grad: bool = False
    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam


@dataclass
class AdaptiveCache(base_sampler.Cache):
    prec: Optional[Tensor] = None
    optimizer: Optional[torch.optim.Optimizer] = None


@dataclass
class HMCAdaptiveIter(HMCIter):
    cache: AdaptiveCache = field(default_factory=AdaptiveCache)
    params: HMCAdaptiveParams = field(default_factory=HMCAdaptiveParams)
    lf_intergrator: Leapfrog = field(default_factory=Leapfrog)

    def init(self):
        super().init()
        self.step_id = 0

        if self.cache.prec is None:
            self.cache.prec = torch.eye(
                self.cache.point.shape[-1],
                device=self.params.device,
            ).repeat(*self.cache.point.shape[:-1], 1, 1).detach().requires_grad_()

            # pos_def = torch.randn(self.cache.point.shape[0], self.cache.point.shape[-1], self.cache.point.shape[-1],
            #                       device=self.params.device)
            # pos_def = torch.bmm(pos_def, pos_def.permute(0, 2, 1))

            # self.cache.prec = torch.linalg.cholesky(pos_def).detach().requires_grad_()

        if isinstance(self.params.entropy_weight, float):
            self.params.entropy_weight = Tensor([self.params.entropy_weight]).repeat(*self.cache.point.shape[:-1], 1)
        else:
            while len(self.params.entropy_weight.shape) < 3:
                self.params.entropy_weight = self.params.entropy_weight[..., None]

        if isinstance(self.params.penalty_weight, float):
            self.params.penalty_weight = Tensor([self.params.penalty_weight]).repeat(*self.cache.point.shape[:-1], 1)
        else:
            while len(self.params.penalty_weight.shape) < 3:
                self.params.penalty_weight = self.params.penalty_weight[..., None]

        if self.cache.optimizer is None:
            self.cache.optimizer = self.params.optimizer_cls(
                [self.cache.prec],
                lr=self.params.learning_rate
            )

        def grad_logp(v: Tensor) -> Tensor:
            return torch.autograd.grad(
                self.params.target_dist.log_prob(v).sum(),
                v,
                retain_graph=True,
                create_graph=True
            )[0]

        def DL(x: Tensor, v: Tensor) -> Tensor:
            z = torch.autograd.functional.jvp(
                grad_logp,
                x,
                torch.bmm(self.cache.prec, v.unsqueeze(-1)).squeeze(-1),
                create_graph=True,
            )[1]
            return (
                -self.params.lf_step_size ** 2 * (self.params.lf_step_count ** 2 - 1) / 6
                * torch.bmm(self.cache.prec.permute(0, 2, 1), z.unsqueeze(-1)).squeeze(-1)
            ).requires_grad_()

        self.DL = DL

        def geometric_cdf(k):
            if k < self.params.min_truncation_level:
                return 0
            return 1 - (1 - self.params.min_truncation_level) ** (k - self.params.min_truncation_level + 1)
        self.geometric_cdf = geometric_cdf

    def run(self):
        noise = self.params.proposal_dist.sample(self.cache.point.shape[:-1])
        p = torch.linalg.solve(self.cache.prec, noise)

        Minv = torch.bmm(self.cache.prec, self.cache.prec.permute(0, 2, 1))

        trajectory = self.lf_intergrator.run(
            q=self.cache.point,
            p=p,
            gradq=self.cache.grad,
            Minv=Minv,
            target_dist=self.params.target_dist,
            step_count=self.params.lf_step_count,
            step_size=self.params.lf_step_size,
            stop_grad=self.params.stop_grad,
        )

        self.trajectory = trajectory

        point_new = trajectory[-1]["q"]
        logp_new = self.params.target_dist.log_prob(point_new)
        grad_new = trajectory[-1]["gradq"]
        p_new = trajectory[-1]["p"]

        # ??\Xi_L
        # grad_trajectory_sum = torch.zeros_like(trajectory[0]["gradq"])
        # for i in range(1, self.params.lf_step_count):
        #     grad_trajectory_sum += (self.params.lf_step_count - i) * trajectory[i]["gradq"]

        # ??\Delta(q_0, v)
        energy_error = (
            logp_new - self.cache.logp
            - 0.5 * (
                torch.bmm(torch.bmm(p_new.unsqueeze(1), Minv), p_new.unsqueeze(-1)).squeeze()
                - torch.bmm(torch.bmm(p.unsqueeze(1), Minv), p.unsqueeze(-1)).squeeze()
            )
        )

        accept_prob = torch.clamp(torch.exp(energy_error), max=1)

        self._adapt(
            energy_error=energy_error,
            accept_prob=accept_prob,
            trajectory=trajectory,
            grad_new=grad_new,
        )

        self.MHStep(
            point_new=point_new,
            logp_new=logp_new,
            grad_new=grad_new,
            accept_prob=accept_prob,
        )
        self.collect_sample(self.cache.point.detach().clone())
        self.step_id += 1

    def _normalized_trace_estimator(self, mid_traj: Tensor) -> Tuple[Tensor, Tensor]:
        truncation_level = self.params.min_truncation_level
        truncation_level += int(torch.distributions.Geometric(self.params.truncation_level_prob).sample((1,)).item())

        eps = torch.randint_like(mid_traj, low=0, high=2, device=self.params.device) * 2 - 1
        cur_vec = eps

        trace = torch.zeros_like(eps, device=self.params.device)
        sign = -1

        for i in range(1, truncation_level + 1):
            cur_res = self.DL(mid_traj, cur_vec)

            spectral_normalization = torch.clamp(
                self.params.spectral_normalization_decay
                * torch.norm(cur_vec, dim=-1, p=2) / torch.norm(cur_res, dim=-1, p=2),
                max=1,
            ).unsqueeze(-1)
            cur_vec = spectral_normalization * cur_res
            trace += sign / (1 - self.geometric_cdf(i)) * cur_vec
            sign *= -1

        return torch.bmm(trace.unsqueeze(1), self.DL(mid_traj, eps).unsqueeze(-1)).squeeze(), cur_vec

    def _adapt(self, energy_error: Tensor, accept_prob: Tensor, trajectory: Tensor, grad_new: Tensor):
        mid_traj = trajectory[1 + self.params.lf_step_count // 2]["q"]
        dimension = grad_new.shape[-1]

        ltheta, eta = self._normalized_trace_estimator(mid_traj)

        b_n = eta / torch.norm(eta, p=2, dim=-1, keepdim=True)
        mu_n = torch.bmm(b_n.unsqueeze(1), self.DL(mid_traj, b_n).unsqueeze(-1)).squeeze(-1)
        penalty = self.params.penalty_func(torch.abs(mu_n))

        loss = (
            -torch.clamp(-energy_error, max=0)
            - self.params.entropy_weight * (
                dimension * np.log(self.params.lf_step_size)
                + torch.log(self.cache.prec.diagonal(dim1=-2, dim2=-1).prod(dim=-1))
                + ltheta - self.params.penalty_weight * penalty
            )
        ).mean()

        # TODO: try batch update, i.e. accumulate the gradient and step once every kth iteration
        self.cache.optimizer.zero_grad()
        loss.backward()
        self.cache.optimizer.step()

        with torch.no_grad():
            self.params.entropy_weight = torch.clamp(
                self.params.entropy_weight * (
                    1 + self.params.entropy_weight_adaptive_rate * (accept_prob[..., None] - self.params.target_acceptance)
                ),
                min=self.params.entropy_weight_min,
                max=self.params.entropy_weight_max,
            )

            self.params.penalty_weight = torch.clamp(
                self.params.penalty_weight * (
                    1 + self.params.penalty_weight_adaptive_rate * penalty[..., None]
                ),
                min=self.params.penalty_weight_min,
                max=self.params.penalty_weight_max,
            )


@dataclass
class HMCAdaptiveVanilla(base_sampler.AlgorithmStoppingRule):
    params: HMCAdaptiveParams
    burn_in_iter_count: int
    sample_iter_count: int
    probe_period: int
    stopping_rule: Callable

    def load_params(self, params: base_sampler.Params):
        self.pipeline = base_sampler.Pipeline([
            base_sampler.SampleBlock(
                iteration=HMCAdaptiveIter(params=params.copy_update(params)),
                iteration_count=self.burn_in_iter_count,
            ),
            base_sampler.SampleBlock(
                iteration=HMCAdaptiveIter(params=params.copy_update(params)),
                iteration_count=self.sample_iter_count,
                stopping_rule=self.stopping_rule,
                probe_period=self.probe_period,
            ),
        ])

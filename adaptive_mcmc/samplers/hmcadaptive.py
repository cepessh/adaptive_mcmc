from dataclasses import dataclass, field
import math
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


class WarmupCosineAnnealingScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_factor for base_lr in self.base_lrs]


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
    clip_grad_value: float = 1e3

    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam
    scheduler_cls: Optional[Type[WarmupCosineAnnealingScheduler]] = None

    iter_count: Optional[int] = None
    warm_up_ratio: float = 0.05

    def __post__init__(self):
        self.no_grad = False


@dataclass
class CholeskyParametrization:
    def __init__(self, batch_size, dimension, device, scale=1e-2):
        self.batch_size = batch_size
        self.dimension = dimension
        self.device = device
        self.params = scale * torch.randn(batch_size * dimension * (dimension + 1) // 2, device=device)
        self.params = self.params.requires_grad_()

    def make_prec(self) -> Tensor:
        self.prec = torch.zeros(self.batch_size, self.dimension, self.dimension, device=self.device)

        tril_ind = torch.tril_indices(row=self.dimension, col=self.dimension)
        for i in range(self.batch_size):
            start_idx = i * self.dimension * (self.dimension + 1) // 2
            end_idx = start_idx + self.dimension * (self.dimension + 1) // 2
            raw_params = self.params[start_idx:end_idx]

            L = torch.zeros(self.dimension, self.dimension, device=self.device)
            L[tril_ind[0], tril_ind[1]] = raw_params
            diag_ind = torch.arange(self.dimension)
            L[diag_ind, diag_ind] = torch.exp(L[diag_ind, diag_ind])
            self.prec[i] = L

        return self.prec


@dataclass
class AdaptiveCache(base_sampler.Cache):
    prec_params: Optional[CholeskyParametrization] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[torch.torch.optim.lr_scheduler._LRScheduler] = None


@dataclass
class HMCAdaptiveIter(HMCIter):
    cache: AdaptiveCache = field(default_factory=AdaptiveCache)
    params: HMCAdaptiveParams = field(default_factory=HMCAdaptiveParams)
    lf_intergrator: Leapfrog = field(default_factory=Leapfrog)

    def init(self):
        super().init()
        self.step_id = 0

        if self.cache.prec_params is None:
            self.cache.prec_params = CholeskyParametrization(
                batch_size=self.cache.point.shape[0],
                dimension=self.cache.point.shape[-1],
                device=self.params.device,
            )

        def ensure_tensor(param, shape):
            if isinstance(param, float):
                param = Tensor([param]).repeat(*shape, 1)
            else:
                while len(param.shape) < 2:
                    param = param.unsqueeze(-1)
            return param

        self.params.entropy_weight = ensure_tensor(self.params.entropy_weight, self.cache.point.shape[:-1])
        self.params.penalty_weight = ensure_tensor(self.params.penalty_weight, self.cache.point.shape[:-1])

        if self.cache.optimizer is None:
            self.cache.optimizer = self.params.optimizer_cls(
                [self.cache.prec_params.params],
                lr=self.params.learning_rate
            )

        if self.cache.scheduler is None and self.params.scheduler_cls is not None:
            self.cache.scheduler = self.params.scheduler_cls(
                self.cache.optimizer,
                warmup_epochs=int(self.params.warm_up_ratio * self.params.iter_count),
                total_epochs=self.params.iter_count,
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
                * torch.einsum("...ij,...i->...j", self.cache.prec, z)
            ).requires_grad_()

        self.DL = DL

        def geometric_cdf(k):
            if k < self.params.min_truncation_level:
                return 0
            return 1 - (1 - self.params.min_truncation_level) ** (k - self.params.min_truncation_level + 1)
        self.geometric_cdf = geometric_cdf
        self.geom_dist = torch.distributions.Geometric(self.params.truncation_level_prob)

    def run(self):
        self.cache.prec = self.cache.prec_params.make_prec()
        self.params.prec = self.cache.prec
        Minv = torch.einsum("...ij,...kj->...ik", self.cache.prec, self.cache.prec)

        ret = self._run_iter(prec=self.cache.prec, Minv=Minv)
        self._adapt(
            energy_error=ret["energy_error"],
            accept_prob=ret["accept_prob"],
            trajectory=ret["trajectory"],
            grad_new=ret["grad_new"],
        )

    def _normalized_trace_estimator(self, mid_traj: Tensor) -> Tuple[Tensor, Tensor]:
        truncation_level = self.params.min_truncation_level
        truncation_level += int(self.geom_dist.sample((1,)).item())

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

        return torch.einsum("...i,...i->...", trace, self.DL(mid_traj, eps)), cur_vec

    def _adapt(self, energy_error: Tensor, accept_prob: Tensor, trajectory: Tensor, grad_new: Tensor):
        mid_traj = trajectory[1 + self.params.lf_step_count // 2]["q"]
        dimension = grad_new.shape[-1]

        ltheta, eta = self._normalized_trace_estimator(mid_traj)

        b_n = eta / torch.norm(eta, p=2, dim=-1, keepdim=True)
        mu_n = torch.einsum("...i,...i->...", b_n, self.DL(mid_traj, b_n)).unsqueeze(-1)

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

        torch.nn.utils.clip_grad_value_(
            self.cache.optimizer.param_groups[0]["params"],
            clip_value=self.params.clip_grad_value,
        )

        self.cache.optimizer.step()
        if self.cache.scheduler is not None:
            self.cache.scheduler.step()

        with torch.no_grad():
            self.params.entropy_weight = torch.clamp(
                self.params.entropy_weight * (
                    1 + self.params.entropy_weight_adaptive_rate * (accept_prob.unsqueeze(-1) - self.params.target_acceptance)
                ),
                min=self.params.entropy_weight_min,
                max=self.params.entropy_weight_max,
            )

            self.params.penalty_weight = torch.clamp(
                self.params.penalty_weight * (
                    1 + self.params.penalty_weight_adaptive_rate * penalty
                ),
                min=self.params.penalty_weight_min,
                max=self.params.penalty_weight_max,
            )


@dataclass
class HMCAdaptive(base_sampler.AlgorithmStoppingRule):
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
                iteration=HMCIter(params=params.copy_update(params)),
                iteration_count=self.sample_iter_count,
                stopping_rule=self.stopping_rule,
                probe_period=self.probe_period,
            ),
        ])

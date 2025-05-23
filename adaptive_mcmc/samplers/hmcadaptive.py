from dataclasses import dataclass, field, replace
from enum import Enum
import math
from typing import Callable, Dict, List, Optional, Tuple, Type

import torch
from torch import Tensor

from adaptive_mcmc.samplers import base_sampler
from adaptive_mcmc.samplers.hmc import (
    HMCCommonParams, HMCFixedParams, HMCIter, Leapfrog, PrecType, HMCCache, HMCVanilla
)
from adaptive_mcmc.linalg.logdet import lanczos_trace_estimator, taylor_trace_estimator
# from adaptive_mcmc.tools.computational_graph import print_graph_with_tensors, count_graph_nodes


class TraceMethod(str, Enum):
    HUTCH_TAYLOR = "hutch_taylor_roulette"
    HUTCH_LANCZOS = "hutch_lanczos"


class EntropyMethod(str, Enum):
    NONE = "none"
    FULL = "full"
    LOCGAUS = "locgaus"


class BackpropMethod(str, Enum):
    FULL = "full"
    APPROX = "approx"


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
class HMCAdaptiveCommonParams(HMCCommonParams):
    truncation_level_prob: float = 0.5
    min_truncation_level: int = 2
    spectral_normalization_decay: float = 0.99

    lanczos_steps: int = 5
    krylov_probe_vectors: int = 5

    learning_rate: float = 5e-3

    entropy_weight: float = 1.
    entropy_weight_min: float = 1e-3
    entropy_weight_max: float = 1e2
    entropy_weight_adaptive_rate: float = 1e-2

    penalty_weight: float = 1.
    penalty_weight_min: float = 1e-3
    penalty_weight_max: float = 1e5
    penalty_weight_adaptive_rate: float = 1e-2

    clip_grad_value: float = 1e3


@dataclass
class HMCAdaptiveFixedParams(HMCFixedParams):
    iter_count: Optional[int] = None
    warm_up_ratio: float = 0.05

    prec_init_scale: float = 1e-2

    no_grad: bool = False
    stop_grad: bool = False

    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam
    scheduler_cls: Optional[Type[WarmupCosineAnnealingScheduler]] = None

    trace_method: TraceMethod = TraceMethod.HUTCH_TAYLOR
    entropy_method: EntropyMethod = EntropyMethod.LOCGAUS
    backprop_method: BackpropMethod = BackpropMethod.FULL
    prec_type: PrecType = PrecType.Dense

    penalty_func: Callable = default_penalty_fn


@dataclass
class CholeskyParametrization:
    def __init__(self, batch_size, dimension, device, prec_type, scale=1e-2):
        self.batch_size = batch_size
        self.dimension = dimension
        self.device = device
        self.prec_type = prec_type

        if prec_type == PrecType.Dense:
            params_count = dimension * (dimension + 1) // 2
        elif prec_type == PrecType.TRIDIAG:
            params_count = 2 * dimension - 1

        self.params = (
            scale * torch.randn(batch_size, params_count, device=device)
        ).requires_grad_()

    def make_prec(self) -> Tensor:
        b, d, dev = self.batch_size, self.dimension, self.device

        if self.prec_type == PrecType.Dense:
            raw = self.params

            L = torch.zeros(b, d, d, device=dev)

            tri_i, tri_j = torch.tril_indices(d, d, device=dev)
            L[:, tri_i, tri_j] = raw

            diag = torch.arange(d, device=dev)
            L[:, diag, diag] = torch.exp(L[:, diag, diag])

        elif self.prec_type == PrecType.TRIDIAG:
            raw = self.params

            main = raw[:, :d]       # (batch, d)
            lower = raw[:, d:]      # (batch, d-1)

            L = torch.zeros(b, d, d, device=dev)

            diag = torch.arange(d, device=dev)
            L[:, diag, diag] = torch.exp(main)
            L[:, diag[1:], diag[:-1]] = lower

        self.prec = L
        return L


@dataclass
class AdaptiveCache(HMCCache):
    prec_params: Optional[CholeskyParametrization] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[torch.torch.optim.lr_scheduler._LRScheduler] = None
    prec: Optional[Tensor] = None
    grad_norm: List[Tensor] = field(default_factory=list)


@dataclass
class HMCAdaptiveIter(HMCIter):
    cache: AdaptiveCache = field(default_factory=AdaptiveCache)
    common_params: HMCAdaptiveCommonParams = field(default_factory=HMCAdaptiveCommonParams)
    fixed_params: HMCAdaptiveFixedParams = field(default_factory=HMCAdaptiveFixedParams)
    lf_intergrator: Leapfrog = field(default_factory=Leapfrog)

    def init(self, cache: Optional[base_sampler.Cache] = None):
        HMCIter.init(self)
        self.step_id = 0

        # if isinstance(self.common_params.lf_step_size, float):
        #     self.common_params.lf_step_size = torch.full(
        #         (self.cache.point.shape[0], 1),
        #         self.common_params.lf_step_size,
        #     )

        if self.cache.prec_params is None:
            self.cache.prec_params = CholeskyParametrization(
                batch_size=self.cache.point.shape[0],
                dimension=self.cache.point.shape[-1],
                device=self.fixed_params.device,
                scale=self.fixed_params.prec_init_scale,
                prec_type=self.fixed_params.prec_type,
            )
            if hasattr(self.cache, "prec"):
                self.cache.prec = None

        def ensure_tensor(param, shape):
            if isinstance(param, float):
                param = Tensor([param]).repeat(*shape, 1)
            else:
                while len(param.shape) < 2:
                    param = param.unsqueeze(-1)
            return param

        self.common_params.entropy_weight = ensure_tensor(self.common_params.entropy_weight, self.cache.point.shape[:-1])
        self.common_params.penalty_weight = ensure_tensor(self.common_params.penalty_weight, self.cache.point.shape[:-1])

        if self.cache.optimizer is None:
            self.cache.optimizer = self.fixed_params.optimizer_cls(
                [self.cache.prec_params.params],
                lr=self.common_params.learning_rate
            )

        if self.cache.scheduler is None and self.fixed_params.scheduler_cls is not None:
            self.cache.scheduler = self.fixed_params.scheduler_cls(
                self.cache.optimizer,
                warmup_epochs=int(self.fixed_params.warm_up_ratio * self.fixed_params.iter_count),
                total_epochs=self.fixed_params.iter_count,
            )

        def grad_logp(v: Tensor) -> Tensor:
            return torch.autograd.grad(
                self.common_params.target_dist.log_prob(v).sum(),
                v,
                retain_graph=False,
                create_graph=False,
            )[0]

        def grad_logp_(v: Tensor) -> Tensor:
            return torch.autograd.grad(
                self.common_params.target_dist.log_prob(v).sum(),
                v,
                retain_graph=True,
                create_graph=True,
            )[0]

        def DL(x: Tensor, v: Tensor) -> Tensor:
            """
            x: (b, d)
            v: (b, d) or (b, n, d), in second case x -> (b, n, d)
            """
            if v.dim() == 3:
                x = x.unsqueeze(1).repeat(1, v.shape[1], 1)

            z = torch.autograd.functional.vhp(
                lambda y: self.common_params.target_dist.log_prob(y).sum(),
                x,
                self.matvec_c(v),
                create_graph=True,
            )[1]

            coef = -self.common_params.lf_step_size ** 2 * (self.common_params.lf_step_count ** 2 - 1) / 6

            if v.dim() == 3:
                coef = coef.view(-1, 1, 1)

            return coef * self.matvec_ct(z)

        self.DL = DL
        self.grad_logp = grad_logp

        def dist_cdf(k):
            if k <= self.common_params.min_truncation_level:
                return 0
            return 1 - (1 - self.common_params.truncation_level_prob) ** (k - self.common_params.min_truncation_level)

        self.truncation_dist_cdf = dist_cdf
        self.truncation_dist = torch.distributions.Geometric(self.common_params.truncation_level_prob)

    def run(self):
        if self.fixed_params.prec_type == PrecType.Dense:
            self.cache.prec = self.cache.prec_params.make_prec()
            self.make_matvecs(self.cache.prec)
            prec = self.cache.prec
        elif self.fixed_params.prec_type == PrecType.TRIDIAG:
            prec = self.cache.prec_params.params
            self.make_matvecs(prec)

        ret = self._run_iter(
            prec=prec,
            noise_grad=(self.fixed_params.entropy_method == EntropyMethod.FULL),
        )

        self._adapt(
            prec=prec,
            energy_error=ret["energy_error"],
            accept_prob=ret["accept_prob"],
            trajectory=ret["trajectory"],
        )

    def _locgaus_entropy_estimator(self, prec, trajectory) -> Tensor:
        mid_traj = trajectory[len(trajectory) // 2]["q"]
        dimension = trajectory[0]["q"].shape[-1]

        log_diag_sum = 0.
        if self.fixed_params.prec_type == PrecType.Dense:
            log_diag_sum = torch.log(prec.diagonal(dim1=-2, dim2=-1)).sum(dim=-1)
        elif self.fixed_params.prec_type == PrecType.TRIDIAG:
            log_diag_sum = prec[:, :dimension].sum(dim=-1)

        entropy = self.common_params.entropy_weight * log_diag_sum

        if self.fixed_params.trace_method == TraceMethod.HUTCH_TAYLOR:
            def matvec(v: Tensor) -> Tensor:
                return self.DL(mid_traj, v)

            ltheta, eta = taylor_trace_estimator(
                matvec=matvec,
                dimension=dimension,
                batch_size=mid_traj.shape[0],
                min_truncation_level=self.common_params.min_truncation_level,
                dist=self.truncation_dist,
                dist_cdf=self.truncation_dist_cdf,
                spectral_normalization_decay=self.common_params.spectral_normalization_decay,
                device=self.fixed_params.device,
            )

            b_n = eta / torch.norm(eta, p=2, dim=-1, keepdim=True)
            mu_n = torch.einsum("...i,...i->...", b_n, self.DL(mid_traj, b_n)).unsqueeze(-1)

            penalty = self.fixed_params.penalty_func(torch.abs(mu_n))

            entropy = entropy + self.common_params.entropy_weight * ltheta - self.common_params.penalty_weight * penalty

            with torch.no_grad():
                if self.fixed_params.trace_method == TraceMethod.HUTCH_TAYLOR:
                    self.common_params.penalty_weight = torch.clamp(
                        self.common_params.penalty_weight * (
                            1 + self.common_params.penalty_weight_adaptive_rate * penalty
                        ),
                        min=self.common_params.penalty_weight_min,
                        max=self.common_params.penalty_weight_max,
                    )

        elif self.fixed_params.trace_method == TraceMethod.HUTCH_LANCZOS:
            def matvec(v: Tensor) -> Tensor:
                return v + self.DL(mid_traj, v)

            ltheta = lanczos_trace_estimator(
                matvec=matvec,
                dimension=dimension,
                batch_size=mid_traj.shape[0],
                probe_vector_count=self.common_params.krylov_probe_vectors,
                lanczos_steps=self.common_params.lanczos_steps,
                device=self.fixed_params.device,
            )

            entropy = entropy + self.common_params.entropy_weight * ltheta

        return entropy

    def _full_entropy_estimator(self, trajectory: List[Dict[str, Tensor]]):
        noise = trajectory[0]["noise"]
        logprob_noise = self.common_params.proposal_dist.log_prob(noise)

        def matvec(v: Tensor) -> Tensor:
            return torch.autograd.grad(
                trajectory[-1]["q"],
                noise,
                grad_outputs=v,
                create_graph=True,
                retain_graph=True,
            )[0]

        logdetT = lanczos_trace_estimator(
            matvec=matvec,
            dimension=noise.shape[-1],
            batch_size=noise.shape[0],
            probe_vector_count=self.common_params.krylov_probe_vectors,
            lanczos_steps=self.common_params.lanczos_steps,
            device=self.common_params.device,
        )

        return -self.common_params.entropy_weight * (logprob_noise - logdetT)

    def _point_potential_grad(self, q: Tensor) -> Tensor:
        q = q.detach().requires_grad_(True)
        neg_pot = self.grad_logp(q).detach()
        q = q.requires_grad_(False)

        return -neg_pot

    def _trajectory_potential_grad(self, trajectory: List[Dict[str, Tensor]]) -> Tuple[Tensor, Tensor]:
        potential = torch.zeros_like(trajectory[0]["q"], device=self.fixed_params.device)
        weighed_potential = torch.zeros_like(trajectory[0]["q"], device=self.fixed_params.device)

        L = len(trajectory)
        for i, entry in enumerate(trajectory[:-1]):
            cur_pot = self._point_potential_grad(entry["q"])
            with torch.no_grad():
                potential = potential + cur_pot
                weighed_potential = weighed_potential + (L - i - 1) * cur_pot

        return potential, weighed_potential

    def _energy_error(
        self,
        trajectory: List[Dict[str, Tensor]],
        traj_pot: Tensor,
        weighted_traj_pot: Tensor
    ) -> Tensor:
        L = len(trajectory)
        h = self.common_params.lf_step_size
        v = trajectory[0]["noise"]

        with torch.no_grad():
            q0 = trajectory[0]["q"]
            q0_pot = -self.common_params.proposal_dist.log_prob(q0)
            q0_kin = 0.5 * v.pow(2).sum(dim=-1)

        q0_pot_grad = self._point_potential_grad(q0)

        qL = (
            q0 + L * h * self.matvec_c(v) - h.pow(2) * self.matvec_minv(weighted_traj_pot)
            - 0.5 * L * h.pow(2) * self.matvec_minv(q0_pot_grad)
        )
        qL_pot = -self.common_params.proposal_dist.log_prob(qL)

        qL_pot_grad = self._point_potential_grad(qL)
        vL = v - 0.5 * h * self.matvec_c(q0_pot_grad + qL_pot_grad) - h * self.matvec_c(traj_pot)
        qL_kin = 0.5 * vL.pow(2).sum(dim=-1)

        return qL_pot + qL_kin - q0_pot - q0_kin

    def _adapt(self, prec: Tensor, energy_error: Tensor, accept_prob: Tensor, trajectory: List[Dict[str, Tensor]]) -> None:
        if self.fixed_params.backprop_method == BackpropMethod.APPROX:
            traj_pot, weighted_traj_pot = self._trajectory_potential_grad(trajectory)
            energy_error = self._energy_error(trajectory, traj_pot, weighted_traj_pot)

        loss = -torch.clamp(-energy_error, max=0)

        entropy = 0

        if self.fixed_params.entropy_method == EntropyMethod.LOCGAUS:
            entropy = self._locgaus_entropy_estimator(
                prec=prec,
                trajectory=trajectory,
            )
        elif self.fixed_params.entropy_method == EntropyMethod.FULL:
            entropy = self._full_entropy_estimator(trajectory)

        loss = (loss - entropy).mean()
        # print(count_graph_nodes(loss))
        # print_graph_with_tensors(loss)
        # print(f"Loss={loss:.4}")

        self.cache.optimizer.zero_grad()

        # torch.autograd.set_detect_anomaly(True)

        loss.backward()

        for p in self.cache.optimizer.param_groups[0]["params"]:
            if p.grad is None:
                continue
            torch.nan_to_num_(p.grad, nan=0.0)

        with torch.no_grad():
            torch.nn.utils.clip_grad_value_(
                self.cache.optimizer.param_groups[0]["params"],
                self.common_params.clip_grad_value,
            )

            grads = self.cache.prec_params.params.grad
            norms = grads.norm(p=2, dim=-1, keepdim=True)
            # scale = torch.clamp(self.common_params.clip_grad_value / (norms + 1e-6), max=1.)
            # grads.mul_(scale)

            # grad_norm = (norms * scale).mean()
            # self.cache.grad_norm.append(grad_norm)
            self.cache.grad_norm.append(norms.mean())

        self.cache.optimizer.step()
        if self.cache.scheduler is not None:
            self.cache.scheduler.step()

        with torch.no_grad():
            self.common_params.entropy_weight = torch.clamp(
                self.common_params.entropy_weight * (
                    1 + self.common_params.entropy_weight_adaptive_rate * (
                        accept_prob.unsqueeze(-1) - self.fixed_params.target_acceptance
                    )
                ),
                min=self.common_params.entropy_weight_min,
                max=self.common_params.entropy_weight_max,
            )


@dataclass
class HMCAdaptive(HMCVanilla):
    step_size_burn_in_iter_count: int
    burn_in_iter_count: int
    sample_iter_count: int
    probe_period: int
    stopping_rule: Callable
    common_params: HMCAdaptiveCommonParams = field(default_factory=HMCAdaptiveCommonParams)
    fixed_params: HMCAdaptiveFixedParams = field(default_factory=HMCAdaptiveFixedParams)

    def load_params(self, common_params: base_sampler.CommonParams):
        self.pipeline = base_sampler.Pipeline([
            base_sampler.SampleBlock(
                iteration=HMCIter(
                    common_params=common_params.copy(),
                    fixed_params=replace(self.fixed_params),
                    collect_required=False,
                ),
                iteration_count=self.step_size_burn_in_iter_count,
            ),
            base_sampler.SampleBlock(
                iteration=HMCAdaptiveIter(
                    common_params=common_params,
                    fixed_params=replace(self.fixed_params),
                ),
                iteration_count=self.burn_in_iter_count,
            ),
            # base_sampler.SampleBlock(
            #     iteration=HMCAdaptiveIter(
            #         common_params=common_params,
            #         fixed_params=self.fixed_params,
            #         adapt_step_size_required=False,
            #         collect_required=True
            #     ),
            #     iteration_count=self.sample_iter_count,
            #     stopping_rule=self.stopping_rule,
            #     probe_period=self.probe_period,
            # ),
            base_sampler.SampleBlock(
                iteration=HMCIter(
                    common_params=common_params,
                    fixed_params=replace(self.fixed_params),
                    collect_required=True,
                ),
                iteration_count=self.sample_iter_count,
                stopping_rule=self.stopping_rule,
                probe_period=self.probe_period,
            ),
        ])

        self.pipeline.sample_blocks[0].iteration.fixed_params.prec_type = PrecType.NONE
        self.adjust_step_exploration()
        self.add_callbacks()
        self.apply_adapt_step_size_mask()

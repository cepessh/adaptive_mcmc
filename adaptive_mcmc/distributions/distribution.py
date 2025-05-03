from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.distributions import Normal, MultivariateNormal, Categorical


class Distribution(ABC):
    """Abstract class for distributions."""

    @abstractmethod
    def log_prob(self, z: Tensor) -> Tensor:
        """Computes the log probability of input z."""
        raise NotImplementedError


class SamplableDistribution(Distribution):
    """Abstract class for distributions that can be sampled."""

    @abstractmethod
    def sample(self, sample_count: int) -> Tensor:
        """Generates samples from the distribution."""
        raise NotImplementedError


class Gaussian(SamplableDistribution):
    def __init__(self, mean: Tensor, cov: Tensor):
        self.mean = mean
        self.cov = cov
        self.dist = MultivariateNormal(loc=mean, covariance_matrix=cov)

    def log_prob(self, z: Tensor) -> Tensor:
        return self.dist.log_prob(z)

    def sample(self, sample_count: int) -> Tensor:
        return self.dist.sample((sample_count, ))


class GaussianMixture(SamplableDistribution):
    """Gaussian Mixture Model distribution."""

    def __init__(self, means: Tensor, covs: Tensor, weights: Tensor, sample_batch_size=256):
        """
        Args:
            means (Tensor): Means of the Gaussian components.
            covs (Tensor): Covariance matrices of the Gaussian components.
            weights (Tensor): Weights of the Gaussian components.
        """
        self.weights = weights.to(dtype=torch.float32)
        self.means = means.to(dtype=torch.float32)
        self.covs = covs.to(dtype=torch.float32)
        self.sample_batch_size = sample_batch_size

        self.category = torch.distributions.Categorical(self.weights)
        self.gaussians = torch.distributions.MultivariateNormal(
            loc=self.means,
            covariance_matrix=self.covs
        )

    def _sample(self, sample_count: int) -> Tensor:
        """
        Generates samples from the Gaussian mixture distribution.

        Args:
            sample_count (int): Number of samples to generate.

        Returns:
            Tensor: Generated samples.
        """
        which_gaussian = self.category.sample((sample_count,))
        all_samples = self.gaussians.sample((sample_count,))

        return all_samples[range(sample_count), which_gaussian, :]

    def sample(self, sample_count: int) -> Tensor:
        """
        Generates samples from the Gaussian mixture distribution.

        Args:
            sample_count (int): Number of samples to generate.

        Returns:
            Tensor: Generated samples.
        """
        all_samples = []
        for i in range(0, sample_count, self.sample_batch_size):
            all_samples.append(self._sample(min(self.sample_batch_size, sample_count - i)))
        return torch.cat(all_samples, dim=0)

    def log_prob(self, z: Tensor) -> Tensor:
        """
        Computes the log probability of data under the Gaussian mixture.

        Args:
            z (Tensor): Input data.

        Returns:
            Tensor: Log probabilities of the input data.
        """
        logs = self.gaussians.log_prob(z.unsqueeze(-2)) + torch.log(self.weights)

        return torch.logsumexp(logs, dim=1)


class FunnelDistribution(SamplableDistribution):
    """A hierarchical Funnel Distribution."""

    def __init__(self, dimension: int, scale: float = 3.0):
        """
        Args:
            dimension (int): Total dimensionality of the funnel distribution.
            scale (float): Controls the spread of the first dimension.
        """
        self.dimension = dimension
        self.scale = scale

        self.base_normal = Normal(0, 1)

    def sample(self, sample_count: int) -> Tensor:
        """
        Generates samples from the Funnel distribution.
        Args:

            sample_count (int): Number of samples to generate.

        Returns:

            Tensor: Samples of shape (sample_count, dimension).
        """
        first_dim = self.base_normal.sample((sample_count, 1)) * self.scale
        rest_dims = self.base_normal.sample((sample_count, self.dimension - 1)) * torch.exp(first_dim / 2)

        return torch.cat([first_dim, rest_dims], dim=1)

    def log_prob(self, z: Tensor) -> Tensor:
        """
        Computes the log probability of data under the Funnel distribution.

        Args:
            z (Tensor): Input data of shape (N, dimension).

        Returns:
            Tensor: Log probabilities of the input data.
        """
        first_dim = z[:, 0]
        rest_dims = z[:, 1:]

        ret = torch.square(rest_dims).sum(dim=-1) / (2 * torch.exp(first_dim))
        ret += 0.5 * (self.dimension - 1) * first_dim
        ret += 0.5 * torch.square(first_dim) / (2 * self.scale)

        return -ret


# class ManyWellDistribution(SamplableDistribution):
#     """
#     A “many-well” multimodal distribution in R^d, defined as a uniform mixture
#     of isotropic Gaussians (the “wells”) whose centers are scattered in space.
#     """
#     def __init__(self,
#                  dimension: int,
#                  num_wells: int = 10,
#                  sigma: float = 0.3,
#                  domain: float = 4.0,
#                  device: str = 'cpu',
#                  dtype=torch.float32):
#         """
#         Args:
#             dimension (int): Dimensionality of the ambient space.
#             num_wells (int): Number of Gaussian wells (modes).
#             sigma (float): Standard deviation of each well.
#             domain (float): Half‐width of the cube in which well centers lie,
#                             i.e. centers ~ Uniform([-domain,domain]^d).
#             device (str): Torch device for storage.
#             dtype: Torch dtype.
#         """
#         self.dimension = dimension
#         self.num_wells = num_wells
#         self.sigma = sigma
#         self.device = device
#         self.dtype = dtype
# 
#         self.log_weight = -torch.log(torch.tensor(float(num_wells),
#                                                   device=device, dtype=dtype))
# 
#         self.means = (
#             torch.rand(num_wells, dimension, device=device, dtype=dtype) * 2 * domain
#             - domain
#         )
# 
#         self.log_norm_const = -0.5 * dimension * torch.log(2 * torch.pi * sigma**2)
# 
#         self.cat = Categorical(logits=torch.full((num_wells,),
#                                                  self.log_weight,
#                                                  device=device,
#                                                  dtype=dtype))
# 
#     def sample(self, sample_count: int) -> Tensor:
#         """
#         Draw samples from the many‐well mixture by first selecting a well
#         uniformly, then sampling from N(mean, σ² I).
# 
#         Returns:
#             Tensor shape (sample_count, dimension)
#         """
#         idx = self.cat.sample((sample_count,))
#         chosen_means = self.means[idx]
#         eps = torch.randn(sample_count, self.dimension,
#                           device=self.device,
#                           dtype=self.dtype) * self.sigma
#         return chosen_means + eps
# 
#     def log_prob(self, z: Tensor) -> Tensor:
#         """
#         Compute log p(z) = log [ (1/K) * Σ_i N(z | μ_i, σ² I) ] via log-sum-exp.
# 
#         Args:
#             z: Tensor of shape (N, dimension)
#         Returns:
#             Tensor of shape (N,) giving log-probabilities
#         """
#         # z: (N, d) → expand to (N, num_wells, d)
#         z_expanded = z.unsqueeze(1)                      # (N, 1, d)
#         # compute squared distances to each well center: (N, num_wells)
#         diff2 = torch.sum((z_expanded - self.means.unsqueeze(0))**2,
#                           dim=-1)                        # (N, num_wells)
#         # Gaussian log densities (N_i): (N, num_wells)
#         log_gauss = self.log_norm_const - 0.5 * diff2 / (self.sigma**2)
#         # add log mixture weight, then log-sum-exp
#         return torch.logsumexp(log_gauss + self.log_weight, dim=1)

from abc import ABC, abstractmethod

import torch
from torch import Tensor


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

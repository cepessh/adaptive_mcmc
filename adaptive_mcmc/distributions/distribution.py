from abc import ABC, abstractmethod
from dataclasses import dataclass
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

    def __init__(self, means: Tensor, covs: Tensor, weights: Tensor):
        """
        Args:
            means (Tensor): Means of the Gaussian components.
            covs (Tensor): Covariance matrices of the Gaussian components.
            weights (Tensor): Weights of the Gaussian components.
        """
        self.weights = weights
        self.category = torch.distributions.Categorical(self.weights)
        self.means = means
        self.covs = covs

    def sample(self, sample_count: int) -> Tensor:
        """
        Generates samples from the Gaussian mixture distribution.
        
        Args:
            sample_count (int): Number of samples to generate.

        Returns:
            Tensor: Generated samples.
        """
        which_gaussian = self.category.sample(torch.Size((sample_count,)))
        chosen_means = self.means[which_gaussian]
        chosen_covs = self.covs[which_gaussian]
        
        multivariate_normal = torch.distributions.MultivariateNormal(
            loc=chosen_means, covariance_matrix=chosen_covs
        )
        return multivariate_normal.sample()

    def log_prob(self, z: Tensor) -> Tensor:
        """
        Computes the log probability of data under the Gaussian mixture.

        Args:
            z (Tensor): Input data.

        Returns:
            Tensor: Log probabilities of the input data.
        """
        logs = torch.distributions.MultivariateNormal(
            loc=self.means, covariance_matrix=self.covs
        ).log_prob(z[:, None, :])
        logs += torch.log(self.weights)

        return torch.logsumexp(logs, dim=1)

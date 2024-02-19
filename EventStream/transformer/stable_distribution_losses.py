"""This module implements the TTE and regression generative emission layers used in the model."""
import torch
from pytorch_lognormal_mixture import LogNormalMixtureDistribution


class NLLSafeExponentialDist(torch.distributions.exponential.Exponential):
    """A class that implements the exponential distribution with a numerically stable negative log-likelihood.

    This class is used to represent an Exponential distribution with a numerically stable negative
    log-likelihood function. It takes as input the ``log_rate`` parameter, not the ``rate`` parameter.
    """

    def __init__(self, log_rate: torch.FloatTensor):
        rate = torch.exp(log_rate)
        super().__init__(rate=rate)
        self.log_rate = log_rate

    def NLL(self, T: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of the exponential distribution.

        Args:
            T: The input tensor.

        Returns:
            The negative log-likelihood of the exponential distribution with rate parameter specified by
            `self.log_rate.exp()`.
        """
        torch._assert(T >= 0, "T must be non-negative.")

        return -self.log_rate + self.rate * T


class NLLSafeGaussianDist(torch.distributions.normal.Normal):
    """A class that implements the Gaussian distribution with a numerically stable negative log-likelihood.

    This class is used to represent a Gaussian distribution with a numerically stable negative
    log-likelihood function. It returns the NLL **only up to a constant factor**. This is because the
    constant factor is not needed when comparing NLLs for the purposes of training as it does not affect
    the gradient.

    It takes as input the ``loc`` and ``scale`` parameters, not the ``mean`` and
    ``stddev`` parameters.
    """

    def __init__(self, loc: torch.FloatTensor, log_scale: torch.FloatTensor):
        scale = torch.exp(log_scale)
        self.log_scale = log_scale
        super().__init__(loc=loc, scale=scale)

    def NLL(self, T: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Negative log-likelihood of the Gaussian distribution **up to a constant factor**.

        Args:
            T: The input tensor.

        Returns:
            The negative log-likelihood of the Gaussian distribution with parameters specified by `self.loc`
            and `self.scale`.
        """
        var = torch.clamp(self.scale**2, min=eps)
        log_var = torch.clamp(2 * self.log_scale, min=eps.log())
        return 0.5 * (log_var + ((T - self.loc) ** 2) / var)


class NLLSafeLogNormalDist(torch.distributions.log_normal.LogNormal):
    """A class that implements the lognormal distribution with a numerically stable negative log-likelihood.

    This class is used to represent a LogNormal distribution with a numerically stable negative
    log-likelihood function. It returns the NLL **only up to a constant factor**. This is because the
    constant factor is not needed when comparing NLLs for the purposes of training as it does not affect
    the gradient.

    It takes as input the ``loc`` and ``scale`` parameters, not the ``mean`` and
    ``stddev`` parameters.
    """

    def __init__(self, loc: torch.FloatTensor, log_scale: torch.FloatTensor):
        scale = torch.exp(log_scale)
        self.log_scale = log_scale
        super().__init__(loc=loc, scale=scale)

    def NLL(self, T: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Negative log-likelihood of the lognormal distribution **up to a constant factor**.

        Args:
            T: The input tensor.

        Returns:
            The negative log-likelihood of the lognormal distribution with parameters specified by `self.loc`
            and `self.scale`.
        """
        var = torch.clamp(self.scale**2, min=eps)
        log_var = torch.clamp(2 * self.log_scale, min=eps.log())
        return 0.5 * (log_var + ((torch.log(T) - self.loc) ** 2) / var)


class NLLSafeLogNormalMixtureDist(LogNormalMixtureDistribution):
    """A class that implements the lognormal mixture distribution with a numerically stable negative log-
    likelihood."""

    def __init__(
        self,
        locs: torch.FloatTensor,
        log_scales: torch.FloatTensor,
        log_weights: torch.FloatTensor,
        mean_log_inter_time: float,
        std_log_inter_time: float,
    ):
        self.log_weights = log_weights
        self.log_scales = log_scales
        self.locs = locs
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        super().__init__(locs, log_scales, log_weights, mean_log_inter_time, std_log_inter_time)

    def NLL(self, T: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of the lognormal mixture distribution.

        Args:
            T: The input tensor.

        Returns:
            The negative log-likelihood of the lognormal mixture distribution with parameters specified by
            `self.locs`, `self.log_scales`, and `self.log_weights`.
        """

        # Compute the NLL of each component
        nlls = NLLSafeLogNormalDist(self.locs, self.log_scales).NLL(T)

        # Compute the NLL of the mixture
        nll = torch.logsumexp(nlls + self.log_weights, dim=-1)

        return nll

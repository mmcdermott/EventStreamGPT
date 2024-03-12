"""This module implements the TTE and regression generative emission layers used in the model."""
import math

import torch
from pytorch_lognormal_mixture import LogNormalMixtureDistribution


class NLLSafeExponentialDist(torch.distributions.exponential.Exponential):
    """A class that implements the exponential distribution with a numerically stable negative log-likelihood.

    This class is used to represent an Exponential distribution with a numerically stable negative
    log-likelihood function. It takes as input the ``log_rate`` parameter, not the ``rate`` parameter.
    """

    def __init__(
        self,
        log_rate: torch.FloatTensor,
        log_mean: float | torch.FloatTensor = 0,
    ):
        # E[self] = 1/exp(self.log_rate)
        #         = 1/exp(-log_mean + log_rate/log_var)
        #         = mean * (1/exp(log_rate/log_var))
        self.log_rate = -log_mean + log_rate

        rate = torch.exp(self.log_rate)
        super().__init__(rate=rate)

    def NLL(self, T: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of ``T`` w.r.t. the exponential distribution.

        Args:
            T: The input tensor.

        Returns:
            The negative log-likelihood of the exponential distribution with rate parameter specified by
            `self.log_rate.exp()`.

        Examples:
            >>> dist = NLLSafeExponentialDist(log_rate=torch.Tensor([0.0, -1.0, 1.0]), log_mean=-4.0)
            >>> T = torch.Tensor([0.2, 1.3, -3.4])
            >>> dist.NLL(T)
            Traceback (most recent call last):
                ...
            AssertionError: T must be non-negative; has min value -3.40e+00
            >>> T = torch.Tensor([0.2, 1.3, 3.4])
            >>> dist.NLL(T)
            tensor([  6.9196,  23.1112, 499.6048])
            >>> -dist.log_prob(T)
            tensor([  6.9196,  23.1112, 499.6048])
            >>> dist = NLLSafeExponentialDist(log_rate=torch.Tensor([0.0]), log_mean=-2.0)
            >>> T = torch.Tensor([0.2])
            >>> dist.NLL(T)
            tensor([-0.5222])
            >>> -dist.log_prob(T)
            tensor([-0.5222])
        """
        torch._assert((T >= 0).all(), f"T must be non-negative; has min value {T.min():.2e}")

        return -self.log_rate + self.rate * T


class NLLSafeGaussianDist(torch.distributions.normal.Normal):
    """A class that implements the Gaussian distribution with a numerically stable negative log-likelihood.

    This class is used to represent a Gaussian distribution with a numerically stable negative
    log-likelihood function. It returns the NLL **only up to an additive scalar**. This is because the
    constant factor is not needed when comparing NLLs for the purposes of training as it does not affect
    the gradient.

    It takes as input the ``loc`` and ``log_scale`` parameters, not the ``mean`` and ``stddev`` parameters.
    """

    def __init__(self, loc: torch.FloatTensor, log_scale: torch.FloatTensor):
        scale = torch.exp(log_scale)
        self.log_scale = log_scale
        super().__init__(loc=loc, scale=scale)

    def NLL(self, T: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Negative log-likelihood of ``T`` w.r.t. the Gaussian distribution **up to an additive scalar**.

        Args:
            T: The input tensor.

        Returns:
            The negative log-likelihood of the Gaussian distribution with parameters specified by `self.loc`
            and `self.log_scale`, in particular up to the additive scalar of ``(1/2)*math.log(2*math.pi)``

        Examples:
            >>> loc = torch.Tensor([0.0, -1.0, 1.0])
            >>> log_scale = torch.Tensor([0.0, 1.0, -1.0])
            >>> dist = NLLSafeGaussianDist(loc, log_scale)
            >>> T = torch.Tensor([-0.2, 0.8, 0.4])
            >>> -dist.log_prob(T)
            tensor([0.9389, 2.1382, 1.2490])
            >>> dist.NLL(T) + (1/2)*math.log(2*math.pi)
            tensor([0.9389, 2.1382, 1.2490])
            >>> dist = NLLSafeGaussianDist(torch.Tensor([0.0]), torch.Tensor([-80.0]))
            >>> T = torch.Tensor([-0.2])
            >>> dist.NLL(T)
            tensor([19993.0938])
            >>> -dist.log_prob(T)
            tensor([inf])
            >>> dist = NLLSafeGaussianDist(torch.Tensor([0.0]), torch.Tensor([-8.0]))
            >>> dist.NLL(T)
            tensor([19993.0938])
            >>> -dist.log_prob(T)
            tensor([177715.1562])
            >>> dist.NLL(T, eps=1e-16) + (1/2)*math.log(2*math.pi)
            tensor([177715.1562])
        """
        var = torch.clamp(self.scale**2, min=eps)
        log_var = torch.clamp(2 * self.log_scale, min=math.log(eps))
        return 0.5 * (log_var + ((T - self.loc) ** 2) / var)


class NLLSafeLogNormalDist(torch.distributions.log_normal.LogNormal):
    """A class that implements the lognormal distribution with a numerically stable negative log-likelihood.

    This class is used to represent a LogNormal distribution with a numerically stable negative log-likelihood
    function. It returns the NLL **only up to an additive scalar**. This is because the constant factor is not
    needed when comparing NLLs for the purposes of training as it does not affect the gradient.

    It takes as input the ``loc`` and ``log_scale`` parameters, not the ``mean`` and ``stddev`` parameters.
    """

    def __init__(self, loc: torch.FloatTensor, log_scale: torch.FloatTensor):
        self.log_scale = log_scale
        scale = torch.exp(self.log_scale)
        super().__init__(loc=loc, scale=scale)

    def NLL(self, T: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Negative log-likelihood of ``T`` w.r.t. the distribution **up to model independent addition**.

        Args:
            T: The input tensor.

        Returns:
            The negative log-likelihood up to an additive term that does not depend on the model parameters of
            the lognormal distribution with parameters specified by `self.loc` and `self.log_scale`. The
            additive term in particular is ``T.log() + (1/2)*math.log(2*math.pi)``

        Examples:
            >>> loc = torch.Tensor([0.0])
            >>> log_scale = torch.Tensor([0.0])
            >>> dist = NLLSafeLogNormalDist(loc, log_scale)
            >>> T = torch.Tensor([0.2])
            >>> -dist.log_prob(T)
            tensor([0.6046])
            >>> dist.NLL(T) + T.log() + (1/2)*math.log(2*math.pi)
            tensor([0.6046])
            >>> T = torch.Tensor([0.3])
            >>> -dist.log_prob(T)
            tensor([0.4397])
            >>> dist.NLL(T) + T.log() + (1/2)*math.log(2*math.pi)
            tensor([0.4397])
        """
        torch._assert((T >= 0).all(), f"T must be non-negative; has min value {T.min():.2e}")
        var = torch.clamp(self.scale**2, min=eps)
        log_var = torch.clamp(2 * self.log_scale, min=math.log(eps))
        return 0.5 * (log_var + ((torch.log(T) - self.loc) ** 2) / var)


class NLLSafeLogNormalMixtureDist(LogNormalMixtureDistribution):
    """A class that implements the lognormal mixture distribution with a numerically stable negative log-
    likelihood.

    THIS DOES NOT WORK -- this actually can't be disentangled nicely because you can't wrap the sum through
    the log. All terms end up being relevant.
    """

    def __init__(
        self,
        locs: torch.FloatTensor,
        log_scales: torch.FloatTensor,
        log_weights: torch.FloatTensor,
        mean_log_inter_time: float,
        std_log_inter_time: float,
    ):
        self.log_weights = log_weights

        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time

        self.locs = locs
        self.log_scales = log_scales

        super().__init__(locs, log_scales, log_weights, mean_log_inter_time, std_log_inter_time)

    def NLL(self, T: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of ``T`` w.r.t. the distribution **up to model independent addition**.

        Args:
            T: The input tensor.

        Returns:
            The negative log-likelihood up to an additive term that does not depend on the model parameters of
            the lognormal distribution with parameters specified by `self.loc` and `self.log_scale`. The
            additive term in particular is ``T.log() + (1/2)*math.log(2*math.pi)``

            with mean 0, std 1
            #>>> mean = -3.2
            #>>> std = 0.1
            tensor([0.3557, 1.3180])

            with mean -3.2, std 0.1
            tensor([124.1945,  52.2644])


        Examples:
            >>> locs = torch.Tensor([[0.0, -1.0], [2.0, -0.4]])
            >>> log_scales = torch.Tensor([[0.0, -1.2], [1.0, 0.2]])
            >>> mean = 0
            >>> std = 1
            >>> log_weights = torch.Tensor([0.2, 0.2])
            >>> dist = NLLSafeLogNormalMixtureDist(locs, log_scales, log_weights, mean, std)
            >>> T = torch.Tensor([-0.2, 0.8])
            >>> -dist.log_prob(T)
            Traceback (most recent call last):
                ...
            ValueError: Expected value argument (Tensor of shape ...) to be within the support...
            >>> dist.NLL(T)
            Traceback (most recent call last):
                ...
            AssertionError: T must be non-negative; has min value -2.00e-01
            >>> T = torch.Tensor([0.2, 0.8])
            >>> -dist.log_prob(T)
            tensor([0.3557, 1.3180])
            >>> dist.NLL(T) + T.log() + (1/2)*math.log(2*math.pi)
            tensor([0.3557, 1.3180])
        """

        locs = (self.locs + self.mean_log_inter_time) * self.std_log_inter_time
        log_scales = self.log_scales * self.std_log_inter_time

        # Compute the NLL of each component
        nlls = NLLSafeLogNormalDist(locs, log_scales).NLL(T)

        # Compute the NLL of the mixture
        nll = torch.logsumexp(nlls + self.log_weights, dim=-1)

        return nll

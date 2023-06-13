"""This module implements the TTE and regression generative emission layers used in the model."""
import torch
from pytorch_lognormal_mixture import LogNormalMixtureDistribution


class LogNormalMixtureTTELayer(torch.nn.Module):
    """A class that outputs a mixture-of-lognormal distribution for time-to-event.

    This class is used to initialize a module and project the input tensor to get a specific
    LogNormal Mixture distribution.

    Args:
        in_dim: The dimension of the input tensor.
        num_components: The number of lognormal components in the mixture distribution.
        mean_log_inter_time: The mean of the log of the inter-event times. Used to initialize the mean
                             of the log of the output distribution. Defaults to 0.0.
        std_log_inter_time: The standard deviation of the log of the inter-event times. Used to initialize
                            the standard deviation of the logs of the output distributions. Defaults to 1.0.
    """

    def __init__(
        self,
        in_dim: int,
        num_components: int,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
    ):
        super().__init__()

        # We multiply by 3 in the projections as we need to get the locs, log_scales, and weights for each
        # component.
        self.proj = torch.nn.Linear(in_dim, 3 * num_components)

        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time

    def forward(self, T: torch.Tensor) -> LogNormalMixtureDistribution:
        """Forward pass.

        Args:
            T: The input tensor.

        Returns:
            A `LogNormalMixtureDistribution` with parameters specified by `self.proj(T)` which has output
            shape `(batch_size, sequence_length, 1)`.
        """
        params = self.proj(T)

        locs = params[..., 0::3]
        log_scales = params[..., 1::3]
        log_weights = params[..., 2::3]

        return LogNormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_log_inter_time=self.mean_log_inter_time,
            std_log_inter_time=self.std_log_inter_time,
        )


class ExponentialTTELayer(torch.nn.Module):
    """A class that outputs an exponential distribution for time-to-event.

    This class is used to initialize the ExponentialTTELayer and project the input tensor to get the
    implied exponential distribution.

    Args:
        in_dim: The dimensionality of the input.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, 1)

    def forward(self, T: torch.Tensor) -> torch.distributions.exponential.Exponential:
        """Forward pass.

        Args:
            T: The input tensor.

        Returns:
            An `Exponential` distribution with parameters specified by `self.proj(T)` which has output shape
            `(batch_size, sequence_length, 1)`.
        """
        # torch.nn.functional.elu has Image (-1, 1), but we need our rate parameter to be > 0. So we need to
        # add 1 to the output here. To ensure validity given numerical imprecision, we also add a buffer given
        # by the smallest possible positive value permissible given the type of `T`.
        rate = torch.nn.functional.elu(self.proj(T)) + 1 + torch.finfo(T.dtype).tiny

        # The rate currently has shape (batch_size, sequence_length, 1). We want to squeeze that last
        # dimension.
        rate = rate.squeeze(dim=-1)

        return torch.distributions.exponential.Exponential(rate=rate)


class GaussianIndexedRegressionLayer(torch.nn.Module):
    """This module implements an indexed, probabilistic regression layer.

    This module outputs `(proj @ X).gather(2, idx)` after projecting the input tensor and subselecting those
    down to just the set of regression targets `idx` that are needed.

    Args:
        n_regression_targets: How many regression targets there are.
        in_dim: The input dimensionality.
    """

    def __init__(self, n_regression_targets: int, in_dim: int):
        super().__init__()

        # We multiply `n_regression_targets` by 2 because we need both mean and standard deviation outputs.
        self.proj = torch.nn.Linear(in_dim, n_regression_targets * 2)

    def forward(
        self, X: torch.Tensor, idx: torch.LongTensor | None = None
    ) -> torch.distributions.normal.Normal:
        """Forward pass.

        Args:
            X: The input tensor.
            idx: The indices of the regression targets to output. If None, then all regression targets are
                predicted.

        Returns:
            The `torch.distributions.normal.Normal` distribution with parameters `self.proj(X)` on indices
            specified by `idx`, which will have output shape `(batch_size, sequence_length, num_predictions)`,
            unless `idx` is None in which case it will have predictions for all indices and have shape
            `(batch_size, sequence_length, n_regression_targets)`.
        """
        Z = self.proj(X)

        Z_mean = Z[..., 0::2]

        # torch.nn.functional.elu has idxmage (-1, 1), but we need our std parameter to be > 0. So we need to
        # add 1 to the output here. To ensure validity given numerical imprecision, we also add a buffer given
        # by the smallest possible positive value permissible given the type of `T`.
        Z_std = torch.nn.functional.elu(Z[..., 1::2]) + 1 + torch.finfo(X.dtype).tiny

        if idx is None:
            return torch.distributions.normal.Normal(loc=Z_mean, scale=Z_std)

        mean = Z_mean.gather(-1, idx)
        std = Z_std.gather(-1, idx)

        return torch.distributions.normal.Normal(loc=mean, scale=std)


class GaussianRegressionLayer(torch.nn.Module):
    """This module implements a probabilistic regression layer.

    Given an input `X`, this module predicts probabilistic regression outputs for each input in `X`
    for one regression target.

    Args:
        in_dim: The input dimensionality.
    """

    def __init__(self, in_dim: int):
        super().__init__()

        # We multiply `n_regression_targets` by 2 because we need both mean and standard deviation outputs.
        self.proj = torch.nn.Linear(in_dim, 2)

    def forward(self, X: torch.Tensor) -> torch.distributions.normal.Normal:
        """Forward pass.

        Args:
            X: The input tensor of shape `(batch_size, sequence_length, in_dim)`.

        Returns:
            The `torch.distributions.normal.Normal` distribution with parameters `self.proj(X)`,
            which will have output shape `(batch_size, sequence_length, 1)`.
        """
        Z = self.proj(X)

        Z_mean = Z[..., 0::2]

        # torch.nn.functional.elu has idxmage (-1, 1), but we need our std parameter to be > 0. So we need to
        # add 1 to the output here. To ensure validity given numerical imprecision, we also add a buffer given
        # by the smallest possible positive value permissible given the type of `T`.
        Z_std = torch.nn.functional.elu(Z[..., 1::2]) + 1 + torch.finfo(X.dtype).tiny

        return torch.distributions.normal.Normal(loc=Z_mean, scale=Z_std)

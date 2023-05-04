from typing import Optional

import torch
from pytorch_lognormal_mixture import LogNormalMixtureDistribution


class LogNormalMixtureTTELayer(torch.nn.Module):
    """Outputs a mixture-of-lognormal distribution for time-to-event."""

    def __init__(
        self,
        in_dim: int,
        num_components: int,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
    ):
        """Initializes the module.

        Args:
            `in_dim` (`int`): The dimension of the input tensor.
            `num_components` (`int`): The number of lognormal components in the mixture distribution.
            `mean_log_inter_time`: (`float`, defaults to 0.0):
                The mean of the log of the inter-event times. Used to initialize the mean of the log of the
                output distribution.
            `std_log_inter_time`: (`float`, defaults to 1.0):
                The standard deviation of the log of the inter-event times. Used to initialize the standard
                deviation of the logs of the output distributions.
        """
        super().__init__()

        # We multiply by 3 in the projections as we need to get the locs, log_scales, and weights for each
        # component.
        self.proj = torch.nn.Linear(in_dim, 3 * num_components)

        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time

    def forward(self, T: torch.Tensor) -> LogNormalMixtureDistribution:
        """Returns the specific LogNormal Mixture distribution given by the input tensor `T`.

        Args:
            `T` is a float Tensor of shape `(batch_size, sequence_length, in_dim)`.
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
    """Outputs an exponential distribution for time-to-event."""

    def __init__(self, in_dim: int):
        """Initializes the ExponentialTTELayer.

        `in_dim` is the dimensionality of the input.
        """
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, 1)

    def forward(self, T: torch.Tensor) -> torch.distributions.exponential.Exponential:
        """Returns the implied exponential distribution.

        Args:
            `T` is a float Tensor of shape `(batch_size, sequence_length, in_dim)`.
        Returns:
            An `Exponential` distribution with parameters specified by `self.proj(T)` which has output
            shape `(batch_size, sequence_length, 1)`.
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
    """
    This module implements an indexed, probabilistic regression layer. Given an input `X`, this module
    predicts probabilistic regression outputs for each input in `X` for many regression targets, then
    subselects those down to just the set of regression targets `idx` that are actually needed in practice,
    and returns those. We can view this as a bilinear matrix multiplication: namely, given the following
    inputs:
        * `X`, a dense, batched, per-event input tensor of shape `(batch_size, sequence_len, in_dim)`
        * `proj`, a dense transformation matrix from the input space to the regression target space of shape
        `(in_dim, regression_dim)`.
        * `idx`, a dense, batched, per-event index tensor indicating which regression target each element of
        `X` corresponds to, of shape `(batch_size, sequence_len, num_predictions)`. Elements of idx are the
        indices in `[0, regression_dim)`.
    then this module outputs `(proj @ X).gather(2, idx)`. Note that this requires us to fully compute
    `Z = proj @ X`, where `Z` is thus of size `(batch_size, sequence_len, regression_dim)`, which may be
    large, when in reality we actually likely only need a small subset of this (based on the indices in idx).
    """

    def __init__(self, n_regression_targets: int, in_dim: int):
        """Initializes the layer.

        Args:
            `n_regression_targets` (`int`): How many regression targets there are.
            `in_dim` (`int`): The input dimensionality.
        """
        super().__init__()

        # We multiply `n_regression_targets` by 2 because we need both mean and standard deviation outputs.
        self.proj = torch.nn.Linear(in_dim, n_regression_targets * 2)

    def forward(
        self, X: torch.Tensor, idx: torch.LongTensor | None = None
    ) -> torch.distributions.normal.Normal:
        """Returns the `Normal` distribution according to the indexed regression task on `X` for
        indices `idx`.

        Args:
            `X` is a float Tensor of shape `(batch_size, sequence_length, in_dim)`.
            `idx` is an optional long Tensor of shape `(batch_size, sequence_length, num_predictions)`

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

        # TODO(mmd): validate args
        return torch.distributions.normal.Normal(loc=mean, scale=std)


class GaussianRegressionLayer(torch.nn.Module):
    """This module implements a probabilistic regression layer.

    Given an input `X`, this module predicts probabilistic regression outputs for each input in `X`
    for one regression target.
    """

    def __init__(self, in_dim: int):
        """Initializes the layer.

        Args:
            `n_regression_targets` (`int`): How many regression targets there are.
            `in_dim` (`int`): The input dimensionality.
        """
        super().__init__()

        # We multiply `n_regression_targets` by 2 because we need both mean and standard deviation outputs.
        self.proj = torch.nn.Linear(in_dim, 2)

    def forward(self, X: torch.Tensor) -> torch.distributions.normal.Normal:
        """Returns the `Normal` distribution according to the indexed regression task on `X` for
        indices `idx`.

        Args:
            `X` is a float Tensor of shape `(batch_size, sequence_length, in_dim)`.
            `idx` is an optional long Tensor of shape `(batch_size, sequence_length, num_predictions)`

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

        return torch.distributions.normal.Normal(loc=Z_mean, scale=Z_std)

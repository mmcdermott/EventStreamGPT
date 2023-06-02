import inspect
from collections.abc import Sequence
from typing import Union

import torch

VALID_INDEX_T = Union[int, slice, type(Ellipsis)]
INDEX_SELECT_T = Union[VALID_INDEX_T, Sequence[VALID_INDEX_T]]


def str_summary(T: torch.Tensor):
    """Returns a string summary of a tensor for debugging purposes.

    Args:
        T: The tensor to summarize.

    Returns:
        A string summary of the tensor, documenting the tensor's shape, dtype, and the range of values it
        contains.

    Examples:
        >>> import torch
        >>> T = torch.FloatTensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]])
        >>> str_summary(T)
        'shape: (1, 2, 5), type: torch.float32, range: 1-10'
        >>> T = torch.LongTensor([[2, 3, -4, 5], [6, 7, 9, -10]])
        >>> str_summary(T)
        'shape: (2, 4), type: torch.int64, range: -10-9'
    """
    return f"shape: {tuple(T.shape)}, type: {T.dtype}, range: {T.min():n}-{T.max():n}"


def expand_indexed_regression(X: torch.Tensor, idx: torch.Tensor, vocab_size: int):
    """Expands sparse values `X` with indices `idx` into a dense representation.

    Args:
        X: A tensor of shape [..., # of observed values] containing observed values. Shape must match that of
            `idx`.
        idx: A tensor of shape [..., # of observed values] containing indices of observed values. Each index
            must be in the range [0, `vocab_size`). Shape must match that of `X`.
        vocab_size: The size of the vocabulary to expand into. Indices in `idx` are indexes into this
            vocabulary.

    Returns:
        A dense tensor of shape [..., `vocab_size`], such that the value at index `idx[i]` in the last
        dimension is `X[i]` for all `i` and the value at all other indices is 0.

    Examples:
        >>> import torch
        >>> X = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
        >>> idx = torch.LongTensor([[0, 1, 2], [1, 3, 0]])
        >>> vocab_size = 5
        >>> expand_indexed_regression(X, idx, vocab_size)
        tensor([[1., 2., 3., 0., 0.],
                [6., 4., 0., 5., 0.]])
    """
    expanded = torch.zeros(*idx.shape[:-1], vocab_size, device=X.device, dtype=X.dtype)
    return expanded.scatter(-1, idx, X)


def safe_masked_max(X: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    """Returns a safe max over the last dimension of `X` respecting the mask `mask`.

    This function takes the max over all elements of the last dimension of `X` where `mask` is True. `mask`
    can take one of two forms:
        * An element-wise mask, in which case it must have the same shape as `X`.
        * A column-wise mask, in which case it must have the same shape as `X` _excluding the second to last
            dimension, which should be omitted_, This case is used when you wish to, for example, take the
            maximum of the hidden states of a network over the sequence length, while respecting an event
            mask.
    If `mask` is uniformly False for a row, the output is zero.

    Args:
        X: A tensor of shape [..., # of rows, # of columns] containing elements to take the max over.
        mask: A Boolean tensor either of shape [..., # of rows, # of columns] or [..., # of columns]
            containing a mask indicating which elements can be considered for the max.

    Returns:
        A tensor of shape [...] containing the max over the last dimension of `X` respecting the mask `mask`.
        If `mask` is uniformly False for a row, the output is zero.

    Raises:
        AssertionError: If `mask` is not the correct shape for either mode.

    Examples:
        >>> import torch
        >>> # An element-wise mask
        >>> X = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
        >>> mask = torch.BoolTensor([[True, True, False], [False, False, False]])
        >>> safe_masked_max(X, mask)
        tensor([2., 0.])
        >>> # A column-wise mask, with a batch dimension.
        >>> X = torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        >>> mask = torch.BoolTensor([[False, True, False], [True, False, True]])
        >>> safe_masked_max(X, mask)
        tensor([[ 2.,  5.],
                [ 9., 12.]])
        >>> X = torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        >>> mask = torch.BoolTensor([[[False, True], [True, True]]])
        >>> safe_masked_max(X, mask)
        Traceback (most recent call last):
            ...
        AssertionError: mask torch.Size([1, 2, 2]) must be the same shape as X torch.Size([2, 2, 3])\
 or the same shape as X excluding the second to last dimension
        >>> X = torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        >>> mask = torch.BoolTensor([[False, True], [True, True]])
        >>> safe_masked_max(X, mask)
        Traceback (most recent call last):
            ...
        AssertionError: mask torch.Size([2, 2]) must be the same shape as X torch.Size([2, 2, 3])\
 or the same shape as X excluding the second to last dimension
    """

    shape_err_string = (
        f"mask {mask.shape} must be the same shape as X {X.shape} "
        "or the same shape as X excluding the second to last dimension"
    )

    if len(mask.shape) < len(X.shape):
        try:
            mask = mask.unsqueeze(-2).expand_as(X)
        except RuntimeError as e:
            raise AssertionError(shape_err_string) from e
    else:
        torch._assert(mask.shape == X.shape, shape_err_string)

    masked_X = torch.where(mask, X, -float("inf"))
    maxes = masked_X.max(-1)[0]
    return torch.nan_to_num(maxes, nan=None, posinf=None, neginf=0)


def safe_weighted_avg(X: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Produces a weighted average of the last dimension of `X`, weighted by the weights in
    `weights` (which must be the same shape as `X`), except in the case where the sum of the weights
    is 0, in which case the output returned is zero.

    Also returns the sum of the weights. `weights` must be >= 0.
    """
    torch._assert(
        (weights >= 0).all(),
        f"`weights` should be >= 0! Got {weights} with minimum {weights.min()}.",
    )

    if len(weights.shape) < len(X.shape):
        weights = weights.unsqueeze(-2).expand_as(X)

    torch._assert(
        weights.shape == X.shape, f"weights, {weights.shape} must be the same shape as X {X.shape}"
    )

    denom = weights.float().sum(dim=-1)
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    return (
        torch.where(
            denom > 0, (X * weights.float()).sum(dim=-1) / safe_denom, torch.zeros_like(denom)
        ),
        denom,
    )


def weighted_loss(loss_per_event: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
    """Given a tensor `loss_per_event` of shape [# subjects, # events] containing loss values per
    event per subject and a tensor `event_mask` containing binary indicators of whether any given
    event is present or not, returns the average per-subject of the average per-event loss for each
    subject, excluding subjects who have no events."""
    loss_per_subject, events_per_subject = safe_weighted_avg(loss_per_event, event_mask)
    return safe_weighted_avg(loss_per_subject, (events_per_subject > 0))[0]


_PROBS_LOGITS_NOT_BOTH_DISTRIBUTIONS = (
    torch.distributions.Bernoulli,
    torch.distributions.Binomial,
    torch.distributions.Categorical,
    torch.distributions.ContinuousBernoulli,
    torch.distributions.Multinomial,
    torch.distributions.RelaxedBernoulli,
)


def idx_distribution(
    D: torch.distributions.Distribution,
    index: INDEX_SELECT_T,
) -> torch.distributions.Distribution:
    """Slices a torch Distribution so its outputs are of the appropriate shape. Sourced from:
    https://github.com/pytorch/pytorch/issues/52625 on 2-16-22 and 12:40 ET.

    Args:
        `D` (`torch.distributions.Distribution`): The distribution to slice.
        `index` (`Union[INDEX_SELECT_T, Sequence[INDEX_SELECT_T]]`):
            The index or slice to apply to the parameters.
    """
    if not isinstance(index, tuple):
        index = (index,)

    # For custom distributions
    if hasattr(D, "__getitem__"):
        return D[index]

    # We need to handle mixture and transformed distributions separately.
    if isinstance(D, torch.distributions.MixtureSameFamily):
        mixture_dist = D.mixture_distribution
        component_dist = D.component_distribution
        result = torch.distributions.MixtureSameFamily(
            mixture_distribution=idx_distribution(mixture_dist, index),
            component_distribution=idx_distribution(component_dist, index),
            validate_args=False,
        )
    elif isinstance(D, torch.distributions.TransformedDistribution):
        transforms = D.transforms
        for transform in transforms:
            assert transform.sign in (-1, 1)  # Asserts transforms are univariate bij

        base_dist = D.base_dist
        result = torch.distributions.TransformedDistribution(
            transforms=transforms,
            base_distribution=idx_distribution(base_dist, index),
            validate_args=False,
        )
    else:
        params = {}
        colon = (slice(None),)
        for name, constraint in D.arg_constraints.items():
            try:
                params[name] = getattr(D, name)[index + colon * constraint.event_dim]
            except IndexError:
                print(
                    f"Failed to slice {name} of shape {getattr(D, name).shape} with "
                    f"{index} + (:,) * {constraint.event_dim} = {index + colon * constraint.event_dim}"
                )
                raise

        cls = type(D)
        if "validate_args" in inspect.signature(cls).parameters.keys():
            params["validate_args"] = False

        if (
            isinstance(D, _PROBS_LOGITS_NOT_BOTH_DISTRIBUTIONS)
            and "probs" in params
            and "logits" in params
        ):
            params.pop("probs")

        result = cls(**params)

    if hasattr(D, "_validate_args"):
        result._validate_args = getattr(D, "_validate_args")

    return result

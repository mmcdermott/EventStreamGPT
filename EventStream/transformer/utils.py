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
        * A column-wise mask, in which case it must have the same shape as `X` excluding the second to last
          dimension, which should be omitted, This case is used when you wish to, for example, take the
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
    """Returns the weighted average of the last dimension of `X`, weighted by `weights`.

    Args:
        X: A tensor containing elements to take the weighted average over.
        weights: The weights for the weighted average. Must be >= 0, and can take one of two forms:

            * Element-wise, in which case it must have the same shape as `X`.
            * Column-wise, in which case it must have the same shape as `X` excluding the second to last
              dimension, which should be omitted, This case is used when you wish to, for example, take the
              average of the hidden states of a network over the sequence length, while respecting an event
              mask.


    Returns:
        For each index in the last dimension of `X`, returns a tuple containing:

            * The weighted average of the last dimension of `X` weighted by `weights` for that index, unless
              the weights for that index sum to 0, in which case the output returned is zero.
            * The sum of the weights for that index (the denominator of the weighted average).

    Raises:
        AssertionError: If `weights` contains negative elements or has an invalid shape.

    Examples:
        >>> import torch
        >>> X = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
        >>> weights = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
        >>> safe_weighted_avg(X, weights)
        (tensor([2.3333, 5.1333]), tensor([ 6., 15.]))
        >>> X = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
        >>> weights = torch.FloatTensor([[0, 0, 0], [1, 0, 0]])
        >>> safe_weighted_avg(X, weights)
        (tensor([0., 4.]), tensor([0., 1.]))
        >>> X = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
        >>> weights = torch.FloatTensor([[0, 0, 0], [-1, 0, 0]])
        >>> safe_weighted_avg(X, weights)
        Traceback (most recent call last):
            ...
        AssertionError: weights should be >= 0
        >>> X = torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        >>> weights = torch.FloatTensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
        >>> safe_weighted_avg(X, weights)
        Traceback (most recent call last):
            ...
        AssertionError: weights torch.Size([2, 3, 2]) must be the same shape as X torch.Size([2, 2, 3])\
 or the same shape as X excluding the second to last dimension
    """

    torch._assert(
        (weights >= 0).all(),
        "weights should be >= 0",
    )

    shape_err_string = (
        f"weights {weights.shape} must be the same shape as X {X.shape} "
        "or the same shape as X excluding the second to last dimension"
    )

    if len(weights.shape) < len(X.shape):
        try:
            weights = weights.unsqueeze(-2).expand_as(X)
        except RuntimeError as e:
            raise AssertionError(shape_err_string) from e
    else:
        torch._assert(weights.shape == X.shape, shape_err_string)

    denom = weights.float().sum(dim=-1)
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    return (
        torch.where(denom > 0, (X * weights.float()).sum(dim=-1) / safe_denom, torch.zeros_like(denom)),
        denom,
    )


def weighted_loss(loss_per_event: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
    """Returns the weighted average of the average per-event loss for each subject.

    Given a tensor `loss_per_event` of shape [# subjects, # events] containing loss values per
    event per subject and a tensor `event_mask` containing binary indicators of whether any given
    event is present or not, returns the average per-subject of the average per-event loss for each
    subject, excluding subjects who have no events.

    Args:
        loss_per_event: A tensor of shape [# subjects, # events] containing loss values per event
        event_mask: A tensor of shape [# subjects, # events] containing binary indicators of whether an event
            was present or not.

    Returns:
        A tensor of shape [] containing the weighted average of the average per-event loss for each subject,
        excluding subjects who have no events. If no subjects have any events, returns 0.

    Examples:
        >>> import torch
        >>> loss_per_event = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
        >>> event_mask = torch.FloatTensor([[1, 1, 1], [1, 0, 0]])
        >>> weighted_loss(loss_per_event, event_mask)
        tensor(3.)
    """
    loss_per_subject, events_per_subject = safe_weighted_avg(loss_per_event, event_mask)
    return safe_weighted_avg(loss_per_subject, (events_per_subject > 0))[0]


_PROBS_LOGITS_NOT_BOTH_DISTRIBUTIONS: tuple[torch.distributions.Distribution] = (
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
    """Slices a torch Distribution so its outputs are of the appropriate shape.

    Torch distributions output tensors of consistent shape upon sample(). This method slices their internal
    parameters so as to yield a new, transformed distribution whose outputs are sliced into a desired shape.
    In many cases, but not all, if you use the slice/index you would use on an output sample() as the `index`
    input to this method, the output distribution will have the desired shape.
    Only works with select distributions.
    Sourced from: https://github.com/pytorch/pytorch/issues/52625 on 2-16-22 at 12:40 ET.

    Args:
        D: The distribution to slice.
        index: The index or slice to apply to the parameters.

    Returns:
        The sliced distribution.

    Raises:
        IndexError: If the index is invalid for the distribution.

    Examples:
        >>> import torch
        >>> logits_tensor = torch.Tensor([[1, 2, -3], [4, 1, 0]])
        >>> D = torch.distributions.Bernoulli(logits=logits_tensor)
        >>> D.sample().shape
        torch.Size([2, 3])
        >>> D2 = idx_distribution(D, (slice(None), slice(None, 1)))
        >>> D2.sample().shape
        torch.Size([2, 1])
        >>> D2.logits
        tensor([[1.],
                [4.]])
        >>> probs_tensor = torch.FloatTensor([[0.1, 0.2, 0.7], [0.2, 0.8, 0.0]])
        >>> D = torch.distributions.Categorical(probs=probs_tensor)
        >>> D.sample().shape
        torch.Size([2])
        >>> D2 = idx_distribution(D, 1)
        >>> D2.sample().shape
        torch.Size([])
        >>> # We have to round because distributions modify their probs params which yields precision errors
        >>> D2.probs.round(decimals=1)
        tensor([0.2000, 0.8000, 0.0000])
        >>> D2 = idx_distribution(D, (slice(None), 2))
        Traceback (most recent call last):
            ...
        IndexError: Failed to slice probs of shape torch.Size([2, 3]) with\
 (slice(None, None, None), 2) + (:,) * 1 = (slice(None, None, None), 2, slice(None, None, None))
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
            except IndexError as e:
                raise IndexError(
                    f"Failed to slice {name} of shape {getattr(D, name).shape} with "
                    f"{index} + (:,) * {constraint.event_dim} = {index + colon * constraint.event_dim}"
                ) from e

        cls = type(D)
        if "validate_args" in inspect.signature(cls).parameters.keys():
            params["validate_args"] = False

        if isinstance(D, _PROBS_LOGITS_NOT_BOTH_DISTRIBUTIONS) and "probs" in params and "logits" in params:
            params.pop("probs")

        result = cls(**params)

    if hasattr(D, "_validate_args"):
        result._validate_args = getattr(D, "_validate_args")

    return result

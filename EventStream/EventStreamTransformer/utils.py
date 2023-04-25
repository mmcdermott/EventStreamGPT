import inspect, torch

from typing import Sequence, Tuple, Union

VALID_INDEX_T = Union[int, slice, type(Ellipsis)]
INDEX_SELECT_T = Union[VALID_INDEX_T, Sequence[VALID_INDEX_T]]

def expand_indexed_regression(X: torch.Tensor, I: torch.Tensor, vocab_size: int):
    """Expands values `X` with indices `I` into a dense representation."""
    expanded = torch.zeros(*I.shape[:-1], vocab_size, device=X.device, dtype=X.dtype)
    return expanded.scatter(-1, I, X)

def safe_masked_max(X: torch.Tensor, mask: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the max of the last dimension of `X` considering only positions where `mask` is `True`, except in
    the case where `mask` is uniformly `False`, in which case the output returned is zero.
    `mask` must have the same shape as `X` up to the last dimension, which should be omitted.
    E.g., if `X` has shape `[41, 8, 23]`, `mask` can either have shape `[41, 8]`.
    """

    if len(mask.shape) < len(X.shape):
        mask = mask.unsqueeze(-2).expand_as(X)

    torch._assert(mask.shape == X.shape, f"mask {mask.shape} must be the same shape as X {X.shape}")

    masked_X = torch.where(mask, X, -float('inf'))
    maxes = masked_X.max(-1)[0]
    return torch.nan_to_num(maxes, nan=None, posinf=None, neginf=0)

def safe_weighted_avg(X: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Produces a weighted average of the last dimension of `X`, weighted by the weights in `weights` (which must
    be the same shape as `X`), except in the case where the sum of the weights is 0, in which case the output
    returned is zero. Also returns the sum of the weights. `weights` must be >= 0.
    """
    torch._assert(
        (weights >= 0).all(),
        f"`weights` should be >= 0! Got {weights} with minimum {weights.min()}."
    )

    if len(weights.shape) < len(X.shape):
        weights = weights.unsqueeze(-2).expand_as(X)

    torch._assert(weights.shape == X.shape, f"weights, {weights.shape} must be the same shape as X {X.shape}")

    denom      = weights.float().sum(dim=-1)
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    return torch.where(
        denom > 0,
        (X * weights.float()).sum(dim=-1) / safe_denom,
        torch.zeros_like(denom)
    ), denom

def weighted_loss(loss_per_event: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor `loss_per_event` of shape [# subjects, # events] containing loss values per event per
    subject and a tensor `event_mask` containing binary indicators of whether any given event is present or
    not, returns the average per-subject of the average per-event loss for each subject, excluding subjects
    who have no events.
    """
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
    """
    Slices a torch Distribution so its outputs are of the appropriate shape. Sourced from:
    https://github.com/pytorch/pytorch/issues/52625 on 2-16-22 and 12:40 ET

    Args:
        `D` (`torch.distributions.Distribution`): The distribution to slice.
        `index` (`Union[INDEX_SELECT_T, Sequence[INDEX_SELECT_T]]`): The index or slice to apply to the parameters.
    """
    if not isinstance(index, tuple): index = (index,)

    # For custom distributions
    if hasattr(D, '__getitem__'): return D[index]

    # We need to handle mixture and transformed distributions separately.
    if isinstance(D, torch.distributions.MixtureSameFamily):
        mixture_dist = D.mixture_distribution
        component_dist = D.component_distribution
        result = torch.distributions.MixtureSameFamily(
            mixture_distribution = idx_distribution(mixture_dist, index),
            component_distribution = idx_distribution(component_dist, index),
            validate_args = False,
        )
    elif isinstance(D, torch.distributions.TransformedDistribution):
        transforms = D.transforms
        for transform in transforms: assert transform.sign in (-1, 1) # Asserts transforms are univariate bij

        base_dist = D.base_dist
        result = torch.distributions.TransformedDistribution(
            transforms = transforms,
            base_distribution = idx_distribution(base_dist, index),
            validate_args = False,
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
        if 'validate_args' in inspect.signature(cls).parameters.keys(): params['validate_args'] = False

        if isinstance(D, _PROBS_LOGITS_NOT_BOTH_DISTRIBUTIONS) and 'probs' in params and 'logits' in params:
            params.pop('probs')

        result = cls(**params)

    if hasattr(D, '_validate_args'): result._validate_args = getattr(D, "_validate_args")

    return result

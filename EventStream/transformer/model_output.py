"""Classes and utilities for model output layers.

Attributes:
    BERNOULLI_DIST_T: The type of a bernoulli distribution.
    CATEGORICAL_DIST_T: The type of a categorical distribution.
    REGRESSION_DIST_T: The type of a regression distribution.
"""
from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers.utils import ModelOutput

from ..data.data_embedding_layer import MeasIndexGroupOptions
from ..data.types import DataModality, PytorchBatch, TemporalityType
from .config import (
    MEAS_INDEX_GROUP_T,
    StructuredTransformerConfig,
    TimeToEventGenerationHeadType,
)
from .generative_layers import (
    ExponentialTTELayer,
    GaussianIndexedRegressionLayer,
    GaussianRegressionLayer,
    LogNormalMixtureTTELayer,
)
from .utils import (
    INDEX_SELECT_T,
    expand_indexed_regression,
    idx_distribution,
    safe_weighted_avg,
    str_summary,
    weighted_loss,
)

BERNOULLI_DIST_T = torch.distributions.Bernoulli
CATEGORICAL_DIST_T = torch.distributions.Categorical
REGRESSION_DIST_T = torch.distributions.Normal


def get_event_types(
    dynamic_measurement_indices: torch.LongTensor,
    dynamic_indices: torch.LongTensor,
    event_type_measurement_idx: int,
    event_type_vocab_offset: int,
) -> torch.LongTensor:
    """Identifies the event types from given dynamic measurements and indices.

    Args:
        dynamic_measurement_indices: Measurement indices to evaluate.
        dynamic_indices: Dynamic indices related to the measurements.
        event_type_measurement_idx: Index to determine the event type.
        event_type_vocab_offset: Offset value applied to dynamic indices.

    Returns:
        The identified event types.

    Raises:
        AssertionError: If there is more than one event type per event.

    Examples:
        >>> import torch
        >>> dynamic_measurement_indices = torch.LongTensor([
        ...     [[1, 2, 2, 2], [1, 2, 2, 0], [2, 2, 1, 0], [2, 1, 0, 0]],
        ...     [[1, 0, 0, 0], [3, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ... ])
        >>> dynamic_indices = torch.LongTensor([
        ...     [[1, 11, 14, 18], [3, 11, 12, 0], [11, 10, 2, 0], [15, 8, 0, 0]],
        ...     [[3, 0, 0, 0], [31, 9, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ... ])
        >>> event_type_measurement_idx = 1
        >>> event_type_vocab_offset = 1
        >>> print(get_event_types(
        ...     dynamic_measurement_indices=dynamic_measurement_indices,
        ...     dynamic_indices=dynamic_indices,
        ...     event_type_measurement_idx=event_type_measurement_idx,
        ...     event_type_vocab_offset=event_type_vocab_offset,
        ... ))
        tensor([[0, 2, 1, 7],
                [2, 8, 0, 0]])
        >>> dynamic_measurement_indices = torch.LongTensor([
        ...     [[1, 1, 2, 2], [1, 2, 2, 0], [2, 2, 1, 0], [2, 1, 0, 0]],
        ...     [[1, 0, 0, 0], [3, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ... ])
        >>> dynamic_indices = torch.LongTensor([
        ...     [[1, 4, 14, 18], [3, 11, 12, 0], [11, 10, 2, 0], [15, 8, 0, 0]],
        ...     [[3, 0, 0, 0], [31, 9, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ... ])
        >>> get_event_types(
        ...     dynamic_measurement_indices=dynamic_measurement_indices,
        ...     dynamic_indices=dynamic_indices,
        ...     event_type_measurement_idx=event_type_measurement_idx,
        ...     event_type_vocab_offset=event_type_vocab_offset,
        ... )
        Traceback (most recent call last):
            ...
        AssertionError: Got 2 event types per event!
    """

    event_type_mask = dynamic_measurement_indices == event_type_measurement_idx

    num_event_types = event_type_mask.sum(-1)
    torch._assert((num_event_types <= 1).all().all(), f"Got {num_event_types.max()} event types per event!")

    return torch.where(event_type_mask, dynamic_indices - event_type_vocab_offset, 0).sum(-1)


def strip_unused_indices(dynamic_indices, *other_tensors):
    """Rearranges `dynamic_indices` and other passed tensors to minimize the number of padding (0) indices.

    For each slice of `dynamic_indices` in the last dimension, this function re-arranges the elements of that
    slice (in `dynamic_indices` and all other passed tensors) such that the maximum number of zero-indices are
    removed and all non-zero indices are at the front of the tensor. This is used during generation, when
    newly generated elements may fill up the end of the tensor and may have zeros in them which we want to
    remove to minimize the size of the output tensors.

    Args:
        dynamic_indices: The indices to be evaluated. This is not the dynamic indices as input to the model,
            but rather that output during generation for a new event, so it is of shape
            (batch, num_dynamic_measurements)
        *other_tensors: Additional tensors to be re-arranged identically to `dynamic_indices`. All such
            tensors must have the same shape as `dynamic_indices`.

    Returns:
        The processed indices or a tuple of processed tensors.

    Examples:
        >>> import torch
        >>> dynamic_indices = torch.LongTensor([
        ...     [1, 11, 0, 18], [3, 0, 12, 0], [0, 0, 2, 0], [15, 8, 0, 0],
        ... ])
        >>> dynamic_measurement_indices = torch.LongTensor([
        ...     [1, 2, 3, 4], [1, 2, 3, 0], [2, 2, 1, 0], [2, 1, 0, 0],
        ... ])
        >>> for T in strip_unused_indices(dynamic_indices, dynamic_measurement_indices):
        ...     print(T)
        tensor([[ 1, 11, 18],
                [ 3, 12,  0],
                [ 2,  0,  0],
                [15,  8,  0]])
        tensor([[1, 2, 4],
                [1, 3, 0],
                [1, 0, 0],
                [2, 1, 0]])
    """

    is_present = dynamic_indices != 0

    present_indices = torch.argwhere(is_present)
    present_rows = present_indices[:, 0]
    col_counts = torch.ones_like(present_rows).cumsum(0)
    present_row_change = torch.cat([torch.ones_like(present_rows[:1]), (present_rows.diff() != 0).long()], 0)

    present_cols = col_counts - (col_counts * present_row_change).cummax(0)[0]

    device = dynamic_indices.device
    index = torch.zeros(dynamic_indices.shape[0], is_present.sum(-1).max(), device=device).long()
    mask = torch.zeros(dynamic_indices.shape[0], is_present.sum(-1).max(), device=device).bool()

    index.index_put_((present_rows, present_cols), present_indices[:, 1])
    mask.index_put_((present_rows, present_cols), torch.ones_like(present_indices[:, 1]).bool())

    def idx_fn(T: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, torch.gather(T, -1, index=index), torch.zeros_like(index, dtype=T.dtype))

    if not other_tensors:
        return idx_fn(dynamic_indices)
    else:
        return tuple([idx_fn(dynamic_indices), *[idx_fn(T) for T in other_tensors]])


class NestedIndexableMixin:
    """Mixin for indexable nested elements.

    Provides a way to slice through nested indexable elements, using a static method and an instance method
    for slicing. This will index through dictionaries, tuples, torch distributions, and naturally indexable
    objects. Inputs of `None` will likewise return `None`. This assumes that inhereting classes can be mapped
    to plain dictionaries via `dataclasses.asdict`.
    """

    @staticmethod
    def _recursive_slice(val: Any, idx: INDEX_SELECT_T):
        match val:
            case None:
                return None
            case dict():
                return {k: NestedIndexableMixin._recursive_slice(v, idx) for k, v in val.items()}
            case tuple():
                return tuple(NestedIndexableMixin._recursive_slice(v, idx) for v in val)
            case torch.distributions.Distribution():
                return idx_distribution(val, idx)
            case _:
                return val[idx]

    def slice(self, idx: INDEX_SELECT_T):
        """Performs joint index selection on the nested elements.

        Args:
            idx: The indices to be selected.

        Returns:
            An instance of the class indexed to the appropriate parameters.
        """

        return self.__class__(**self._recursive_slice(asdict(self), idx))


@dataclass
class TransformerOutputWithPast(ModelOutput):
    """Holds output data from a transformer model.

    This class is designed to manage output data from a transformer model,
    which may include last hidden state, past key values, hidden states, and attentions.

    Args:
        last_hidden_state: The last hidden state from the model.
        past_key_values: The past key values from the model.
        hidden_states: The hidden states from the model.
        attentions: The attentions from the model.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: tuple[tuple[torch.FloatTensor]] | dict[str, tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@dataclass
class GenerativeSequenceModelLosses(ModelOutput):
    """Holds losses data for a Generative Sequence Model.

    This class is designed to manage losses from a Generative Sequence Model,
    which can include classification, regression and time to event losses.

    Args:
        classification: Losses for the classification task.
        regression: Losses for the regression task.
        time_to_event: Loss for the time-to-event task.
    """

    classification: dict[str, torch.FloatTensor] | None = None
    regression: dict[str, torch.FloatTensor] | None = None
    time_to_event: torch.FloatTensor | None = None


@dataclass
class GenerativeSequenceModelSamples(ModelOutput):
    """A single sample (event) of a generative sequence model.

    Args:
        event_mask: A boolean tensor of shape [batch_size,] indicating whether events exist.
        time_to_event: A float tensor of shape [batch_size,]. Is 0 if the event does not exist, otherwise
            quantifies the time between the prior event and this event in the series.
        classification: A dictionary with keys as measurements and values as tensors. Shape of value tensor is
            [batch_size,] if measurement is single label classification or [batch_size, vocab_size] if
            measurement is multi label classification. The tensor contains either the class index (starting at
            0, not the global offset) for the prediction for that data type for this event or per-label binary
            labels for multi label data types for the prediction for that data type. If the event is not
            present, all predictions will be zero.
        regression: A dictionary with keys as measurements and values as tensors.
            Shape of value tensor is [batch_size,] if measurement is univariate or
            [batch_size, n_regression_targets] if measurement is multivariate.
            The tensor contains the floating-point predictions for that measurement. If an event is not
            present, predictions will be zero. Predictions are ordered in accordance with the index-labels
            (starting at zero) for the data-type vocabulary contained in regression_indices. If
            regression_indices is None, predictions span the entire vocabulary in vocabulary order.
        regression_indices: A dictionary with keys as measurements and values as tensors. Shape of value
            tensor is [batch_size, n_regression_targets] Contains the indices for which regression contains
            predictions for each data type. If None, regression predictions correspond to the entire
            vocabulary in vocabulary order.
    """

    event_mask: torch.BoolTensor | None = None
    time_to_event: torch.FloatTensor | None = None
    classification: dict[str, torch.LongTensor] | None = None
    regression: dict[str, torch.FloatTensor] | None = None
    regression_indices: dict[str, torch.LongTensor] | None = None

    def _build_new_batch_element(
        self,
        batch: PytorchBatch,
        config: StructuredTransformerConfig,
    ) -> tuple[
        torch.FloatTensor,
        torch.BoolTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.FloatTensor,
        torch.BoolTensor,
    ]:
        """Builds a new batch element from the prediction sample in this object.

        Args:
            batch: The current batch.
            config: The transformer configuration.

        Returns: A tuple containing the time to event, event mask, dynamic indices, dynamic measurement
            indices, dynamic values, and dynamic values mask. The dynamic attributes are updated using the
            function provided in the configuration.

        Raises:
            ValueError: If the input dimensions do not match the expected dimensions.
        """

        # Add data elements (indices, values, types, values_mask)
        dynamic_measurement_indices = []
        dynamic_indices = []
        dynamic_values = []
        dynamic_values_mask = []

        # Add event_mask
        event_mask = self.event_mask

        duration_since_start = torch.where(batch.event_mask[:, :-1], batch.time_delta[:, :-1], 0).sum(-1)
        new_time = torch.where(event_mask, batch.start_time + duration_since_start + self.time_to_event, 0)

        # Add time-dependent values if present.
        for m, cfg in config.measurement_configs.items():
            if cfg.temporality != TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                continue
            if cfg.modality == DataModality.DROPPED:
                continue

            # Produce the functional, time-dependent outputs.

            # TODO(mmd): This may be wrong in some cases! Don't know the actual start time as it is
            # initialized to zero!
            fn = cfg.functor

            is_meas_map = batch.dynamic_measurement_indices[:, -1, :] == config.measurements_idxmap[m]
            indices = batch.dynamic_indices[:, -1, :]
            values = batch.dynamic_values[:, -1, :]
            values_mask = batch.dynamic_values_mask[:, -1, :]

            # We sum(-1) here as there must be exactly one time-dependent-event observation of a given type
            # per event, by definition.
            indices = torch.where(is_meas_map, indices, torch.zeros_like(indices)).sum(-1)
            vals = torch.where(is_meas_map & values_mask, values, torch.zeros_like(values)).sum(-1)

            offset = config.vocab_offsets_by_measurement[m]

            new_indices, new_values = fn.update_from_prior_timepoint(
                prior_indices=indices - offset,
                prior_values=vals,
                new_delta=self.time_to_event,
                new_time=new_time,
                vocab=cfg.vocabulary,
                measurement_metadata=cfg.measurement_metadata,
            )

            new_indices = (new_indices + offset).unsqueeze(-1)
            new_values = new_values.unsqueeze(-1)
            new_measurement_indices = config.measurements_idxmap[m] * torch.ones_like(new_indices)

            dynamic_indices.append(new_indices)
            dynamic_values_mask.append(~torch.isnan(new_values))
            dynamic_values.append(torch.nan_to_num(new_values, nan=0, posinf=0, neginf=0))
            dynamic_measurement_indices.append(new_measurement_indices)

        if dynamic_indices:
            dynamic_indices = torch.cat(dynamic_indices, 1)
            dynamic_measurement_indices = torch.cat(dynamic_measurement_indices, 1)
            dynamic_values = torch.cat(dynamic_values, 1)
            dynamic_values_mask = torch.cat(dynamic_values_mask, 1)
        else:
            if config.measurements_per_dep_graph_level is None:
                dynamic_indices = torch.zeros(batch.batch_size, 1, 0, dtype=torch.long, device=batch.device)
            else:
                dynamic_indices = torch.zeros(
                    batch.batch_size,
                    1,
                    len(config.measurements_per_dep_graph_level),
                    0,
                    dtype=torch.long,
                    device=batch.device,
                )
            dynamic_measurement_indices = torch.zeros_like(dynamic_indices)
            dynamic_values = torch.zeros_like(dynamic_indices).float()
            dynamic_values_mask = torch.zeros_like(dynamic_indices).bool()

        return (
            self.time_to_event,
            event_mask,
            *strip_unused_indices(
                dynamic_indices,
                dynamic_measurement_indices,
                dynamic_values,
                dynamic_values_mask,
            ),
        )

    def format_updates_to_last_batch_event(
        self,
        batch: PytorchBatch,
        config: StructuredTransformerConfig,
        measurements_to_build: set[MEAS_INDEX_GROUP_T] | None = None,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.BoolTensor]:
        """Generate a new batch element from the prediction sample in the object.

        This function is used for generation. It dynamically builds various elements such as indices,
        values, types, and values_mask based on the given configuration and measurements.

        Args:
            batch: The Pytorch batch object.
            config: The structured transformer configuration object.
            measurements_to_build: The set of measurements indices group to be built. If None, all are built.

        Returns:
            A tuple containing four tensors: the new dynamic indices, the new dynamic measurement indices,
            the new dynamic values, and the new dynamic values mask.

        Raises:
            ValueError: If measurement is missing in the config's vocab_offsets_by_measurement, or the shape
            of the prediction does not match the expected shape, or the prediction is greater than or equal to
            the vocab size.
            RuntimeError: If the indices cannot be gathered due to mismatch in shape or values.
        """

        # Add data elements (indices, values, types, values_mask)
        dynamic_measurement_indices = []
        dynamic_indices = []
        dynamic_values = []
        dynamic_values_mask = []

        def add_single_label_classification(measurement: str):
            if measurement not in config.vocab_offsets_by_measurement:
                raise ValueError(f"Missing {measurement}")

            vocab_offset = config.vocab_offsets_by_measurement[measurement]
            vocab_size = config.vocab_sizes_by_measurement[measurement]

            if measurement not in self.classification:
                print(
                    f"WARNING: Attempting to generate improper measurement {measurement}! "
                    f"Acceptable targets: {', '.join(self.classification.keys())}"
                )
                return

            preds = self.classification[measurement]

            if len(preds.shape) != 1:
                raise ValueError(f"For {measurement}, expect 1D preds, got {preds.shape}!")
            if (preds >= vocab_size).any():
                raise ValueError("For {measurement}, need preds < vocab_size!")
            indices = vocab_offset + preds

            measurement_indices = config.measurements_idxmap[measurement] * torch.ones_like(indices)

            dynamic_indices.append(indices.unsqueeze(-1))
            dynamic_measurement_indices.append(measurement_indices.unsqueeze(-1))

        def add_multi_label_classification(measurement: str):
            if measurement not in config.vocab_offsets_by_measurement:
                raise ValueError(f"Missing {measurement}")

            vocab_offset = config.vocab_offsets_by_measurement[measurement]
            vocab_size = config.vocab_sizes_by_measurement[measurement]

            if measurement not in self.classification:
                print(f"WARNING: Attempting to generate improper measurement {measurement}!")
                return

            preds = self.classification[measurement]
            if len(preds.shape) != 2:
                raise ValueError(f"For {measurement}, expect 2D preds, got {preds.shape}!")
            if preds.shape[-1] != vocab_size:
                raise ValueError(
                    f"For {measurement}, expect preds.shape[-1] == vocab_size, got {preds.shape[-1]}!"
                )

            indices = torch.arange(vocab_size, device=preds.device).long() + vocab_offset
            indices = indices.unsqueeze(0).expand_as(preds)
            indices = torch.where(preds == 1, indices, 0)

            indices = strip_unused_indices(indices)

            measurement_indices = config.measurements_idxmap[measurement] * (
                torch.where(indices != 0, torch.ones_like(indices), torch.zeros_like(indices))
            )

            dynamic_indices.append(indices)
            dynamic_measurement_indices.append(measurement_indices)

        def add_univariate_regression(measurement: str):
            if measurement not in self.regression:
                raise ValueError(f"Attempting to generate improper measurement {measurement}!")

            preds = self.regression[measurement].squeeze(-1)
            if len(preds.squeeze(-1).shape) != 1:
                raise ValueError(f"For {measurement}, expect 1D preds, got {preds.shape}!")

            dynamic_values_mask.append(~torch.isnan(preds.unsqueeze(-1)))
            dynamic_values.append(torch.nan_to_num(preds.unsqueeze(-1), nan=0))

        def add_multivariate_regression(measurement: str, indices: torch.LongTensor):
            if measurement not in self.regression:
                raise ValueError(f"Attempting to generate improper measurement {measurement}!")

            regressed_values = self.regression[measurement]
            regressed_values_mask = torch.ones_like(regressed_values).bool()
            vocab_size = config.vocab_sizes_by_measurement[measurement]

            # Now we need to align the regressed_indices to the classification indices, as indices we
            # regressed over but don't think were actually observed in the event wouldn't have
            # values. To do this, we'll first expand out over all possible values/targets, if
            # necessary, then sub-select down.
            if (
                (self.regression_indices is not None)
                and (measurement in self.regression_indices)
                and (self.regression_indices[measurement] is not None)
            ):
                regressed_indices = self.regression_indices[measurement]

                # TODO(mmd): this is inefficient -- don't need to expand fully to dense then back to a
                # different spares...
                regressed_values = expand_indexed_regression(regressed_values, regressed_indices, vocab_size)
                regressed_values_mask = expand_indexed_regression(
                    regressed_values_mask, regressed_indices, vocab_size
                )

            vocab_offset = config.vocab_offsets_by_measurement[measurement]
            try:
                mask = indices >= vocab_offset
                idx_gather_T = torch.where(mask, indices - vocab_offset, 0).long()

                values = regressed_values.gather(-1, idx_gather_T)
                values_mask = regressed_values_mask.gather(-1, idx_gather_T)
            except RuntimeError:
                print(f"Failed on {measurement} with {indices.shape} indices")
                print(f"Vocab offset: {vocab_offset}")
                print(f"Indices:\n{indices}")
                raise

            values = torch.where(mask, values, 0)
            values_mask = torch.where(mask, values_mask, False)

            dynamic_values.append(values)
            dynamic_values_mask.append(values_mask)

        if "event_type" in measurements_to_build:
            add_single_label_classification("event_type")

            # Event type has no value associated with it.
            dynamic_values.append((0 * dynamic_indices[-1]).float())
            dynamic_values_mask.append((0 * dynamic_indices[-1]).bool())

        for m in measurements_to_build:
            if type(m) in (list, tuple):
                assert len(m) == 2
                m, group_mode = m
            else:
                group_mode = None

            if m == "event_type":
                continue
            else:
                cfg = config.measurement_configs[m]
                modality = cfg.modality

            match (modality, group_mode):
                case (DataModality.SINGLE_LABEL_CLASSIFICATION, None):
                    add_single_label_classification(m)
                    dynamic_values.append((0 * dynamic_indices[-1]).float())
                    dynamic_values_mask.append((0 * dynamic_indices[-1]).bool())
                case (DataModality.MULTI_LABEL_CLASSIFICATION, None):
                    assert group_mode is None

                    add_multi_label_classification(m)

                    dynamic_values.append((0 * dynamic_indices[-1]).float())
                    dynamic_values_mask.append((0 * dynamic_indices[-1]).bool())
                case (DataModality.UNIVARIATE_REGRESSION, None):
                    add_univariate_regression(m)

                    indices = config.vocab_offsets_by_measurement[m] * dynamic_values_mask[-1].long()
                    measurement_indices = config.measurements_idxmap[m] * dynamic_values_mask[-1].long()

                    dynamic_indices.append(indices)
                    dynamic_measurement_indices.append(measurement_indices)
                case (
                    DataModality.MULTIVARIATE_REGRESSION,
                    None | MeasIndexGroupOptions.CATEGORICAL_AND_NUMERICAL,
                ):
                    add_multi_label_classification(m)
                    add_multivariate_regression(m, indices=dynamic_indices[-1])
                case (
                    DataModality.MULTIVARIATE_REGRESSION,
                    MeasIndexGroupOptions.CATEGORICAL_ONLY,
                ):
                    add_multi_label_classification(m)
                    dynamic_values.append((0 * dynamic_indices[-1]).float())
                    dynamic_values_mask.append((0 * dynamic_indices[-1]).bool())
                case (DataModality.MULTIVARIATE_REGRESSION, MeasIndexGroupOptions.NUMERICAL_ONLY):
                    meas_index = config.measurements_idxmap[m]
                    existing_mask = batch.dynamic_measurement_indices[:, -1] == meas_index

                    indices = torch.where(existing_mask, batch.dynamic_indices[:, -1], 0)

                    indices = strip_unused_indices(indices)
                    measurement_indices = meas_index * torch.ones_like(indices)

                    dynamic_indices.append(indices)
                    dynamic_measurement_indices.append(measurement_indices)

                    add_multivariate_regression(m, indices=indices)
                case _:
                    raise ValueError(f"{modality}, {group_mode} invalid!")

        dynamic_indices = torch.cat(dynamic_indices, 1)
        dynamic_measurement_indices = torch.cat(dynamic_measurement_indices, 1)
        dynamic_values = torch.cat(dynamic_values, 1)
        dynamic_values_mask = torch.cat(dynamic_values_mask, 1)

        return strip_unused_indices(
            dynamic_indices, dynamic_measurement_indices, dynamic_values, dynamic_values_mask
        )

    @staticmethod
    def pad_data_elements(
        batch: PytorchBatch,
        new_dynamic_indices: torch.LongTensor,
        new_dynamic_measurement_indices: torch.LongTensor,
        new_dynamic_values: torch.FloatTensor,
        new_dynamic_values_mask: torch.BoolTensor,
    ):
        """Pads the dimensions of the new batch elements to match the old ones.

        This static method adjusts the shape of the given new dynamic data elements (indices, measurement
        indices, values, and values mask) to match the shape of those in the given batch. It achieves this
        by padding the shorter one of the new and old data elements with zeros (for LongTensors and
        FloatTensors) or with False (for BoolTensors).

        Args:
            batch: A PytorchBatch object whose data element dimensions are to be matched.
            new_dynamic_indices: The indices tensor to be resized. This just
            new_dynamic_measurement_indices: The measurement indices tensor to be resized.
            new_dynamic_values: The values tensor to be resized.
            new_dynamic_values_mask: The values mask tensor to be resized.

        Returns:
            A tuple of two tuples. The first inner tuple contains the possibly-padded dynamic data elements
            of the given batch. The second inner tuple contains the possibly-padded new dynamic data elements.

        Examples:
            >>> import torch
            >>> batch = PytorchBatch(
            ...     dynamic_indices=torch.tensor([
            ...         [[1, 2, 3], [4, 5, 6]],
            ...         [[7, 8, 9], [10, 11, 12]]
            ...     ]),
            ...     dynamic_measurement_indices=torch.tensor([
            ...         [[1, 2, 3], [4, 5, 6]],
            ...         [[7, 8, 9], [10, 11, 12]]
            ...     ]),
            ...     dynamic_values=torch.tensor([
            ...         [[1., 2., 3.], [4., 5., 6.]],
            ...         [[7., 8., 9.], [10., 11., 12.]]
            ...     ]),
            ...     dynamic_values_mask=torch.tensor([
            ...         [[True, True, True], [True, True, True]],
            ...         [[True, True, True], [True, True, True]]
            ...     ])
            ... )
            >>> new_dynamic_indices = torch.tensor([
            ...     [[1, 2], [3, 4]],
            ...     [[5, 6], [7, 8]]
            ... ])
            >>> new_dynamic_measurement_indices = torch.tensor([
            ...     [[1, 2], [3, 4]],
            ...     [[5, 6], [7, 8]]
            ... ])
            >>> new_dynamic_values = torch.tensor([
            ...     [[1., 2.], [3., 4.]],
            ...     [[5., 6.], [7., 8.]]
            ... ])
            >>> new_dynamic_values_mask = torch.tensor([
            ...     [[True, True], [True, True]],
            ...     [[True, True], [True, True]]
            ... ])
            >>> out = GenerativeSequenceModelSamples.pad_data_elements(
            ...     batch,
            ...     new_dynamic_indices,
            ...     new_dynamic_measurement_indices,
            ...     new_dynamic_values,
            ...     new_dynamic_values_mask
            ... )
            >>> len(out)
            2
            >>> for tensor_tuple in out:
            ...     print(len(tensor_tuple))
            ...     for tensor in tensor_tuple:
            ...         print(tensor)
            4
            tensor([[[ 1,  2,  3],
                     [ 4,  5,  6]],
            <BLANKLINE>
                    [[ 7,  8,  9],
                     [10, 11, 12]]])
            tensor([[[ 1,  2,  3],
                     [ 4,  5,  6]],
            <BLANKLINE>
                    [[ 7,  8,  9],
                     [10, 11, 12]]])
            tensor([[[ 1.,  2.,  3.],
                     [ 4.,  5.,  6.]],
            <BLANKLINE>
                    [[ 7.,  8.,  9.],
                     [10., 11., 12.]]])
            tensor([[[True, True, True],
                     [True, True, True]],
            <BLANKLINE>
                    [[True, True, True],
                     [True, True, True]]])
            4
            tensor([[[1, 2, 0],
                     [3, 4, 0]],
            <BLANKLINE>
                    [[5, 6, 0],
                     [7, 8, 0]]])
            tensor([[[1, 2, 0],
                     [3, 4, 0]],
            <BLANKLINE>
                    [[5, 6, 0],
                     [7, 8, 0]]])
            tensor([[[1., 2., 0.],
                     [3., 4., 0.]],
            <BLANKLINE>
                    [[5., 6., 0.],
                     [7., 8., 0.]]])
            tensor([[[ True,  True, False],
                     [ True,  True, False]],
            <BLANKLINE>
                    [[ True,  True, False],
                     [ True,  True, False]]])
            >>> batch = PytorchBatch(
            ...     dynamic_indices=torch.tensor([
            ...         [[1], [4]],
            ...         [[7], [10]]
            ...     ]),
            ...     dynamic_measurement_indices=torch.tensor([
            ...         [[1], [4]],
            ...         [[7], [10]]
            ...     ]),
            ...     dynamic_values=torch.tensor([
            ...         [[1.], [4.]],
            ...         [[7.], [10.]]
            ...     ]),
            ...     dynamic_values_mask=torch.tensor([
            ...         [[True], [True]],
            ...         [[True], [True]]
            ...     ])
            ... )
            >>> new_dynamic_indices = torch.tensor([
            ...     [[1, 2], [3, 4]],
            ...     [[5, 6], [7, 8]]
            ... ])
            >>> new_dynamic_measurement_indices = torch.tensor([
            ...     [[1, 2], [3, 4]],
            ...     [[5, 6], [7, 8]]
            ... ])
            >>> new_dynamic_values = torch.tensor([
            ...     [[1., 2.], [3., 4.]],
            ...     [[5., 6.], [7., 8.]]
            ... ])
            >>> new_dynamic_values_mask = torch.tensor([
            ...     [[True, True], [True, True]],
            ...     [[True, True], [True, True]]
            ... ])
            >>> out = GenerativeSequenceModelSamples.pad_data_elements(
            ...     batch,
            ...     new_dynamic_indices,
            ...     new_dynamic_measurement_indices,
            ...     new_dynamic_values,
            ...     new_dynamic_values_mask
            ... )
            >>> len(out)
            2
            >>> for tensor_tuple in out:
            ...     print(len(tensor_tuple))
            ...     for tensor in tensor_tuple:
            ...         print(tensor)
            4
            tensor([[[ 1,  0],
                     [ 4,  0]],
            <BLANKLINE>
                    [[ 7,  0],
                     [10,  0]]])
            tensor([[[ 1,  0],
                     [ 4,  0]],
            <BLANKLINE>
                    [[ 7,  0],
                     [10,  0]]])
            tensor([[[ 1.,  0.],
                     [ 4.,  0.]],
            <BLANKLINE>
                    [[ 7.,  0.],
                     [10.,  0.]]])
            tensor([[[ True, False],
                     [ True, False]],
            <BLANKLINE>
                    [[ True, False],
                     [ True, False]]])
            4
            tensor([[[1, 2],
                     [3, 4]],
            <BLANKLINE>
                    [[5, 6],
                     [7, 8]]])
            tensor([[[1, 2],
                     [3, 4]],
            <BLANKLINE>
                    [[5, 6],
                     [7, 8]]])
            tensor([[[1., 2.],
                     [3., 4.]],
            <BLANKLINE>
                    [[5., 6.],
                     [7., 8.]]])
            tensor([[[True, True],
                     [True, True]],
            <BLANKLINE>
                    [[True, True],
                     [True, True]]])
        """
        n_data_elements_old = batch.dynamic_indices.shape[-1]
        n_data_elements_new = new_dynamic_indices.shape[-1]

        dynamic_indices = batch.dynamic_indices
        dynamic_measurement_indices = batch.dynamic_measurement_indices
        dynamic_values = batch.dynamic_values
        dynamic_values_mask = batch.dynamic_values_mask

        if n_data_elements_new < n_data_elements_old:
            data_delta = n_data_elements_old - n_data_elements_new
            new_dynamic_indices = torch.nn.functional.pad(new_dynamic_indices, (0, data_delta), value=0)
            new_dynamic_measurement_indices = torch.nn.functional.pad(
                new_dynamic_measurement_indices, (0, data_delta), value=0
            )
            new_dynamic_values = torch.nn.functional.pad(new_dynamic_values, (0, data_delta), value=0)
            new_dynamic_values_mask = torch.nn.functional.pad(
                new_dynamic_values_mask, (0, data_delta), value=False
            )
        elif n_data_elements_new > n_data_elements_old:
            data_delta = n_data_elements_new - n_data_elements_old
            dynamic_indices = torch.nn.functional.pad(dynamic_indices, (0, data_delta), value=0)
            dynamic_measurement_indices = torch.nn.functional.pad(
                dynamic_measurement_indices, (0, data_delta), value=0
            )
            dynamic_values = torch.nn.functional.pad(dynamic_values, (0, data_delta), value=0)
            dynamic_values_mask = torch.nn.functional.pad(dynamic_values_mask, (0, data_delta), value=False)

        return (
            (dynamic_indices, dynamic_measurement_indices, dynamic_values, dynamic_values_mask),
            (
                new_dynamic_indices,
                new_dynamic_measurement_indices,
                new_dynamic_values,
                new_dynamic_values_mask,
            ),
        )

    def append_to_batch(
        self,
        batch: PytorchBatch,
        config: StructuredTransformerConfig,
    ) -> PytorchBatch:
        """Appends a new batch element to the input batch.

        This function first constructs a new batch element from the current object, and then appends it to the
        given batch. It adjusts the time delta and event mask of the batch accordingly, and ensures that the
        dynamic data elements of the batch and the new element are of the same dimensions by applying padding
        as needed.

        Args:
            batch: The PytorchBatch object to which the new element will be added.
            config: A StructuredTransformerConfig object containing configuration data.

        Returns:
            A new PytorchBatch object, which includes the original data plus the appended new batch element.
        """

        (
            new_event_time_delta,
            new_event_mask,
            new_dynamic_indices,
            new_dynamic_measurement_indices,
            new_dynamic_values,
            new_dynamic_values_mask,
        ) = self._build_new_batch_element(batch, config)

        # Combine everything
        seq_dim = 1

        time_delta = batch.time_delta.clone()
        time_delta[:, -1] = new_event_time_delta
        time_delta = torch.cat(
            (time_delta, torch.ones_like(new_event_time_delta).unsqueeze(seq_dim)), seq_dim
        )
        event_mask = torch.cat((batch.event_mask, new_event_mask.unsqueeze(seq_dim)), seq_dim)

        # Re-pad data elements.
        (
            (dynamic_indices, dynamic_measurement_indices, dynamic_values, dynamic_values_mask),
            (
                new_dynamic_indices,
                new_dynamic_measurement_indices,
                new_dynamic_values,
                new_dynamic_values_mask,
            ),
        ) = self.pad_data_elements(
            batch,
            new_dynamic_indices,
            new_dynamic_measurement_indices,
            new_dynamic_values,
            new_dynamic_values_mask,
        )

        dynamic_indices = torch.cat((dynamic_indices, new_dynamic_indices.unsqueeze(seq_dim)), seq_dim)
        dynamic_measurement_indices = torch.cat(
            (dynamic_measurement_indices, new_dynamic_measurement_indices.unsqueeze(seq_dim)),
            seq_dim,
        )
        dynamic_values = torch.cat((dynamic_values, new_dynamic_values.unsqueeze(seq_dim)), seq_dim)
        dynamic_values_mask = torch.cat(
            (dynamic_values_mask, new_dynamic_values_mask.unsqueeze(seq_dim)), seq_dim
        )

        return PytorchBatch(
            time_delta=time_delta,
            event_mask=event_mask,
            dynamic_indices=dynamic_indices,
            dynamic_measurement_indices=dynamic_measurement_indices,
            dynamic_values=dynamic_values,
            dynamic_values_mask=dynamic_values_mask,
            static_indices=batch.static_indices,
            static_measurement_indices=batch.static_measurement_indices,
            start_time=batch.start_time,
            stream_labels=batch.stream_labels,
        )

    def update_last_event_data(
        self,
        batch: PytorchBatch,
        config: StructuredTransformerConfig,
        measurements_to_fill: set[MEAS_INDEX_GROUP_T] | None = None,
    ) -> PytorchBatch:
        """Updates the last batch element with data from the current object.

        This method modifies the last batch element in the given PytorchBatch object, based on the data
        available in the current object. The measurements that will be filled in the batch element are
        determined by the configuration and the 'measurements_to_fill' argument.

        Args:
            batch: The PytorchBatch object containing the batch element to be updated.
            config: A StructuredTransformerConfig object containing configuration data.
            measurements_to_fill: A set of MEAS_INDEX_GROUP_T that specifies which measurements to fill. If
                not specified, all dynamic measurements from the config that are not dropped will be filled.

        Raises:
            ValueError: If 'time' is included in the 'measurements_to_fill' set.

        Returns:
            A new PytorchBatch object that includes the updated batch element.
        """

        if measurements_to_fill is None:
            measurements_to_fill = []
            for m, cfg in config.measurement_configs.items():
                if not cfg.is_dropped and cfg.temporality == TemporalityType.DYNAMIC:
                    measurements_to_fill.append(m)
            measurements_to_fill = set(measurements_to_fill)

        if not measurements_to_fill:
            return batch

        if "time" in measurements_to_fill:
            raise ValueError("You shouldn't ever be trying to fill the 'time' aspect of a batch!")

        prev_dynamic_indices = batch.dynamic_indices[:, -1]
        prev_dynamic_measurement_indices = batch.dynamic_measurement_indices[:, -1]
        prev_dynamic_values = batch.dynamic_values[:, -1]
        prev_dynamic_values_mask = batch.dynamic_values_mask[:, -1]

        (
            new_dynamic_indices,
            new_dynamic_measurement_indices,
            new_dynamic_values,
            new_dynamic_values_mask,
        ) = self.format_updates_to_last_batch_event(batch, config, measurements_to_build=measurements_to_fill)

        # The `format_updates_to_last_batch_event` function takes care of only building the relevant metrics,
        # including building either just categorical elements or categorical and numerical or numerical only.
        # However, in the case where we build numerical only, we end up appending the categorical value
        # indices again, which we want to remove.
        prev_measurements_to_drop_idx = torch.zeros_like(prev_dynamic_indices, dtype=torch.bool)

        for m in measurements_to_fill:
            if (type(m) is not tuple) or (m[1] != MeasIndexGroupOptions.NUMERICAL_ONLY):
                continue

            m = m[0]
            prev_measurements_to_drop_idx |= prev_dynamic_measurement_indices == config.measurements_idxmap[m]

        data_tensors = []
        for dt in (
            prev_dynamic_indices,
            prev_dynamic_measurement_indices,
            prev_dynamic_values,
            prev_dynamic_values_mask,
        ):
            data_tensors.append(torch.where(prev_measurements_to_drop_idx, 0, dt))

        (
            prev_dynamic_indices,
            prev_dynamic_measurement_indices,
            prev_dynamic_values,
            prev_dynamic_values_mask,
        ) = strip_unused_indices(*data_tensors)

        try:
            new_dynamic_indices = torch.cat((prev_dynamic_indices, new_dynamic_indices), 1)
        except BaseException:
            print(prev_dynamic_indices.shape)
            print(new_dynamic_indices.shape)
        new_dynamic_measurement_indices = torch.cat(
            (prev_dynamic_measurement_indices, new_dynamic_measurement_indices), 1
        )
        new_dynamic_values = torch.cat((prev_dynamic_values, new_dynamic_values), 1)
        new_dynamic_values_mask = torch.cat((prev_dynamic_values_mask, new_dynamic_values_mask), 1)

        # Re-pad data elements.
        (
            (dynamic_indices, dynamic_measurement_indices, dynamic_values, dynamic_values_mask),
            (
                new_dynamic_indices,
                new_dynamic_measurement_indices,
                new_dynamic_values,
                new_dynamic_values_mask,
            ),
        ) = self.pad_data_elements(
            batch,
            new_dynamic_indices,
            new_dynamic_measurement_indices,
            new_dynamic_values,
            new_dynamic_values_mask,
        )

        dynamic_indices[:, -1] = new_dynamic_indices
        dynamic_measurement_indices[:, -1] = new_dynamic_measurement_indices
        dynamic_values[:, -1] = new_dynamic_values
        dynamic_values_mask[:, -1] = new_dynamic_values_mask

        return PytorchBatch(
            time_delta=batch.time_delta,
            event_mask=batch.event_mask,
            dynamic_indices=dynamic_indices,
            dynamic_measurement_indices=dynamic_measurement_indices,
            dynamic_values=dynamic_values,
            dynamic_values_mask=dynamic_values_mask,
            static_indices=batch.static_indices,
            static_measurement_indices=batch.static_measurement_indices,
            start_time=batch.start_time,
            stream_labels=batch.stream_labels,
        )


@dataclass
class GenerativeSequenceModelPredictions(ModelOutput, NestedIndexableMixin):
    """Contains the predictions for the GenerativeSequenceModel head.

    Args:
        classification: The predicted classification task results.
        regression: The predicted regression task results.
        regression_indices: The predicted indices for the regression task.
        time_to_event: The predicted time-to-event results.
    """

    classification: dict[
        str, tuple[None, BERNOULLI_DIST_T] | tuple[BERNOULLI_DIST_T, CATEGORICAL_DIST_T]
    ] | None = None
    regression: dict[str, torch.distributions.Distribution] | None = None
    regression_indices: dict[str, torch.LongTensor] | None = None
    time_to_event: torch.distributions.Distribution | None = None

    def sample(
        self,
        event_mask: torch.BoolTensor,
    ) -> GenerativeSequenceModelSamples:
        """Generates a sample from the contained predictions.

        Args:
            event_mask: A boolean tensor representing the event mask. This is used only to provide a source
                for the sampled event's mask (which is copied from the last sequence dimension of this input).

        Returns:
            A sample from the GenerativeSequenceModel.

        Raises:
            ValueError: If the classification or regression distributions are malformed or unrecognized.
        """

        match self.classification:
            case None:
                sampled_classification = None
            case dict():
                sampled_classification = {}
                for k, v in self.classification.items():
                    match v:
                        case [None, BERNOULLI_DIST_T() as joint_dist]:
                            sampled_classification[k] = joint_dist.sample()
                        case [
                            BERNOULLI_DIST_T() as is_obs_dist,
                            CATEGORICAL_DIST_T() as samp_dist,
                        ]:
                            is_obs = is_obs_dist.sample() == 1
                            samp = samp_dist.sample()
                            sampled_classification[k] = torch.where(is_obs, samp, torch.zeros_like(samp))
                        case _:
                            raise ValueError(f"Don't know how to sample classification dist {v}!")
            case _:
                raise ValueError(f"self.classification is malformed! Got\n{self.classification}")

        match self.regression:
            case None:
                sampled_regression = None
            case dict():
                sampled_regression = {}
                for k, v in self.regression.items():
                    match v:
                        case [None, REGRESSION_DIST_T() as joint_dist]:
                            sampled_regression[k] = joint_dist.sample()
                        case [BERNOULLI_DIST_T() as is_obs_dist, REGRESSION_DIST_T() as samp_dist]:
                            is_obs = is_obs_dist.sample() == 1
                            samp = samp_dist.sample()
                            is_obs = is_obs.unsqueeze(-1).expand_as(samp)
                            nans = float("nan") * torch.ones_like(samp)
                            sampled_regression[k] = torch.where(is_obs, samp, nans)
                        case _:
                            raise ValueError(f"Don't know how to sample regression dist {v}!")
            case _:
                raise ValueError(f"self.regression is malformed! Got\n{self.regression}")

        if self.time_to_event is not None:
            time_to_event = self.time_to_event.sample()
            # This is wrong!
            # TODO(mmd): Make this correct.
            time_to_event = torch.nan_to_num(time_to_event, nan=None, posinf=1000)
        else:
            time_to_event = None

        return GenerativeSequenceModelSamples(
            event_mask=event_mask[:, -1].detach(),
            classification=sampled_classification,
            regression=sampled_regression,
            regression_indices=self.regression_indices,
            time_to_event=time_to_event,
        )


@dataclass
class GenerativeSequenceModelLabels(ModelOutput):
    """Contains the labels for the GenerativeSequenceModel head.

    The labels are split by task type. Single-label classification task labels will have
    shape batch X seq and have raw integer labels, whereas multi-label classification task labels
    will have shape batch X seq X vocab size and have binary indicators for each label.

    Args:
        classification: The classification task labels.
        regression: The regression task labels.
        regression_indices: The indices for the regression task.
        time_to_event: The time-to-event task labels.
    """

    classification: dict[str, torch.LongTensor] | None = None
    regression: dict[str, torch.FloatTensor] | None = None
    regression_indices: dict[str, torch.LongTensor] | None = None
    time_to_event: torch.FloatTensor | None = None


@dataclass
class GenerativeSequenceModelOutput(ModelOutput):
    """Contains all GenerativeSequenceModel outputs.

    The outputs include losses, predictions, labels, and masks, among others.

    Args:
        loss: The overall model loss.
        losses: The specific model losses by task type.
        preds: The model predictions.
        labels: The model labels.
        event_mask: A boolean tensor representing the event mask.
        dynamic_values_mask: A boolean tensor representing the dynamic values mask.
        past_key_values: The past key values from the model.
        hidden_states: The hidden states from the model.
        attentions: The attentions from the model.
    """

    loss: torch.FloatTensor
    losses: GenerativeSequenceModelLosses | None = None
    preds: GenerativeSequenceModelPredictions | None = None
    labels: GenerativeSequenceModelLabels | None = None
    event_mask: torch.BoolTensor | None = None
    dynamic_values_mask: torch.BoolTensor | None = None

    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@dataclass
class StreamClassificationModelOutput(ModelOutput):
    """Contains all outputs for the Stream Classification Model.

    Args:
        loss: The overall model loss.
        preds: The model predictions.
        labels: The model labels.
    """

    loss: torch.FloatTensor
    preds: torch.FloatTensor = None
    labels: torch.LongTensor | torch.FloatTensor = None


class GenerativeOutputLayerBase(torch.nn.Module):
    """A base class for the output layer of a generative model.

    This class is responsible for constructing the time-to-event (TTE) layer based on the
    TTE_generation_layer_type in the given config, along with observation and classification layers. It also
    establishes the criteria for observation and classification. It does not contain a forward method which
    actually calls these helper methods, as those are implemented by subclass specific methods depending on
    how the encoded state is structured.

    This class should not be instantiated directly. Instead, use one of the derived classes.

    Args:
        config: A configuration object of type StructuredTransformerConfig.

    Raises:
        ValueError: If the TTE_generation_layer_type in the config is not valid.
        ValueError: If any measurements are duplicated in the regression layers.
    """

    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__()

        self.config = config

        match self.config.TTE_generation_layer_type:
            case TimeToEventGenerationHeadType.LOG_NORMAL_MIXTURE:
                self.TTE_layer = LogNormalMixtureTTELayer(
                    in_dim=config.hidden_size,
                    num_components=config.TTE_lognormal_generation_num_components,
                    mean_log_inter_time=config.mean_log_inter_event_time_min,
                    std_log_inter_time=config.std_log_inter_event_time_min,
                )
            case TimeToEventGenerationHeadType.EXPONENTIAL:
                self.TTE_layer = ExponentialTTELayer(in_dim=config.hidden_size)
            case _:
                raise ValueError(
                    f"Invalid option for `config.TTE_generation_layer_type`. Must be "
                    f"a member of the `TimeToEventGenerationHeadType` enum: "
                    f"({TimeToEventGenerationHeadType.values()}). got {config.TTE_generation_layer_type}."
                )

        self.IsObservedLayer = torch.nn.Linear(config.hidden_size, len(config.measurements_idxmap))
        self.ClassificationLayer = torch.nn.Linear(config.hidden_size, config.vocab_size)

        self.is_observed_criteria = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.classification_criteria = {}
        for measurement in config.measurements_for(DataModality.SINGLE_LABEL_CLASSIFICATION):
            self.classification_criteria[measurement] = torch.nn.CrossEntropyLoss(reduction="none")
        for measurement in config.measurements_for(DataModality.MULTI_LABEL_CLASSIFICATION):
            self.classification_criteria[measurement] = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.regression_layers = torch.nn.ModuleDict({})
        for measurement in config.measurements_for(DataModality.MULTIVARIATE_REGRESSION):
            self.regression_layers[measurement] = GaussianIndexedRegressionLayer(
                n_regression_targets=config.vocab_sizes_by_measurement[measurement],
                in_dim=config.hidden_size,
            )
        for measurement in config.measurements_for(DataModality.UNIVARIATE_REGRESSION):
            if measurement in self.regression_layers:
                raise ValueError(f"{measurement} duplicated!")
            self.regression_layers[measurement] = GaussianRegressionLayer(in_dim=config.hidden_size)

        self.classification_mode_per_measurement = {}
        for generative_mode, measurements in config.measurements_per_generative_mode.items():
            if generative_mode not in (
                DataModality.SINGLE_LABEL_CLASSIFICATION,
                DataModality.MULTI_LABEL_CLASSIFICATION,
            ):
                continue
            for measurement in measurements:
                assert measurement not in self.classification_mode_per_measurement
                self.classification_mode_per_measurement[measurement] = generative_mode

    def get_TTE_outputs(
        self, batch: PytorchBatch, encoded: torch.FloatTensor, is_generation: bool = False
    ) -> tuple[torch.FloatTensor, torch.distributions.Distribution, torch.FloatTensor,]:
        """Produces time-to-event predictions and log likelihoods (**not NLLs!**) for the model.

        Args:
            batch: The batch of data for which the classification predictions are desired.
            encoded: The final encodings used to predict the time from the event at a position to the
                subsequent event. This tensor is of shape (batch size X sequence length X hidden dim).
            is_generation: A boolean to indicate if the function is used for generation. Defaults to False. If
                true, then the model will only return the predicted distribution (as that is all that is used
                in generative use-cases).

        Returns:
            A tuple containing the following items:
                TTE_LL: A torch scalar containing the average log-likelihood of observed time-to-events given
                the predicted distribution.
                TTE_dist: The predicted torch Distribution for modelling time-to-event.
                TTE_true: A tensor containing the observed time between events for each batch element.

        Raises:
            ValueError: If NaNs are found in TTE_obs_mask_exp, TTE_true_exp or TTE_LL or if there is no
            observed time-to-event for >= 1 patient in the batch.
        """
        TTE_dist = self.TTE_layer(encoded)

        if is_generation:
            return None, TTE_dist, None

        # TTE_dist is a distribution with random variables of shape (batch size, sequence length)
        TTE_obs_mask = batch["event_mask"][:, 1:] & batch["event_mask"][:, :-1]
        TTE_delta = batch["time_delta"][:, :-1]
        TTE_true = torch.where(TTE_obs_mask, TTE_delta, torch.ones_like(TTE_delta))

        # As TTE_dist contains a predicted distribution for the last sequence element, which we want to return
        # for generative purposes, we add a fake observation to the last element.
        TTE_true_exp = torch.cat((TTE_true, torch.ones_like(TTE_true[:, -1]).unsqueeze(-1)), dim=-1)
        TTE_obs_mask_exp = torch.cat(
            (TTE_obs_mask, torch.zeros_like(TTE_obs_mask[:, -1]).unsqueeze(-1)), dim=-1
        )

        # We skip the last event as we have no true time to event for that event.
        # TODO(mmd): Use NLL-\beta?
        try:
            TTE_LL = TTE_dist.log_prob(TTE_true_exp)
        except ValueError as e:
            print(f"Failed to compute TTE log prob on input {str_summary(TTE_true_exp)}: {e}")
            raise

        if TTE_obs_mask_exp.isnan().any():
            raise ValueError(f"NaNs in TTE_obs_mask_exp: {batch}")
        elif TTE_true_exp.isnan().any():
            raise ValueError(f"NaNs in TTE_true_exp: {batch}")
        elif TTE_LL.isnan().any():
            raise ValueError(f"NaNs in TTE_LL: {batch}")
        elif (TTE_obs_mask_exp.float().sum(-1) == 0).any():
            raise ValueError(f"No observed time-to-event for >= 1 patient in batch: {batch}")

        TTE_LL_per_patient = (TTE_LL * TTE_obs_mask_exp.float()).sum(-1) / TTE_obs_mask_exp.float().sum(-1)
        TTE_LL_overall = TTE_LL_per_patient.mean()

        return TTE_LL_overall, TTE_dist, TTE_true

    def get_classification_outputs(
        self,
        batch: PytorchBatch,
        encoded: torch.FloatTensor,
        valid_measurements: set[str],
    ) -> tuple[
        dict[str, torch.FloatTensor],
        dict[str, tuple[None, BERNOULLI_DIST_T] | tuple[BERNOULLI_DIST_T, CATEGORICAL_DIST_T]],
        dict[str, torch.LongTensor | torch.FloatTensor],
    ]:
        """Produces classification predictions and losses for the model.

        Args:
            batch: The batch of data for which the classification predictions are desired.
            encoded: The final encodings *to be used to predict for each position in the sequence*. For
                example, the vector ``encoded[i][j]`` (which is of size ``hidden_dim``) is *not* the summary
                encoding of the batch element at batch index ``i`` and sequence index ``j``, but rather is the
                input to be used to form classification predictions corresponding to batch element ``i`` at
                sequence position ``j``.
            valid_measurements: The classification measurements in the batch that should be predicted from
                this input ``encoded``.

        Returns:
            The following three dictionaries

            1. ``classification_losses_by_measurement``:
               A dictionary from `measurement` to scalar tensors consisting of the average NLL of the data
               given the classiciation model. Averaging happens via the following procedure:

               * For multi-label measurements:

                 1. NLL is averaged over labels per sequence event, for all unmasked sequence events (as in
                    theory any event could have observed labels for binary multi-lable predictions).
                    TODO(mmd): this should likely be specific to events with certain event types.
                 2. NLL is macro-averaged across unmasked sequence events per batch element.
                 3. NLL is macro-averaged across batch elements.

               * For single-task measurements:

                 1. NLL is computed on any event that has a label for that task.
                    TODO(mmd): Maybe should be conditioned on specific event types too?
                 2. NLL is macro-averaged across events which had a label for that task per sequence.
                    Sequences without any events with that label receive a loss of zero.
                 3. NLL is macro-averaged across batch elements.

            2. ``classification_dists_by_measurement``:
               A dictionary from `measurement` to classification distributions of shape
               `[batch_size X sequence_length X vocabulary_size]` or `[batch_size X sequence_length]`
               reflecting the probabilities for each event for that measurement. Returns scores for all
               events, even those that are masked, including the final event.
            3. ``classification_labels_by_measurement``:
               A dictionary from `measurement` to tensors of one of two types:

               * For multi-label measurements, returns FloatTensors of shape
                 `[batch_size X sequence_length X vocabulary_size]` containing binary labels for each
                 vocabulary element for each event.
               * For single-label measurements, returns LongTensors of shape
                 `[batch_size, sequence_length]` containing label indices for each event with that task
                 observed, otherwise contains zeros.
        """

        if not valid_measurements:
            return {}, {}, {}

        torch._assert(~torch.isnan(encoded).any(), f"{torch.isnan(encoded).sum()} NaNs in encoded")

        # Classification of what elements are going to occur:
        is_observed_score = self.IsObservedLayer(encoded)
        torch._assert(
            ~torch.isnan(is_observed_score).any(),
            f"{torch.isnan(is_observed_score).sum()} NaNs in is_observed_score",
        )

        classification_scores = self.ClassificationLayer(encoded)

        classification_losses_by_measurement = {}
        classification_dists_by_measurement = {}
        classification_labels_by_measurement = {}

        for measurement, classification_mode in self.classification_mode_per_measurement.items():
            if measurement not in valid_measurements:
                continue

            event_mask = batch["event_mask"]

            measurement_idx = self.config.measurements_idxmap[measurement]
            vocab_start = self.config.vocab_offsets_by_measurement[measurement]
            vocab_end = min(
                o
                for o in list(self.config.vocab_offsets_by_measurement.values()) + [self.config.vocab_size]
                if o > vocab_start
            )

            scores = classification_scores[:, :, vocab_start:vocab_end]
            # scores is of shape [batch X seq X vocab_end-vocab_start]
            # We subtract 1 here as the measurement_idx of 0 is withheld for missing data.
            is_obs_score = is_observed_score[:, :, measurement_idx - 1]
            # is_obs_score is of shape [batch X seq]

            # We don't need to shift here, as given this is a structured model, we'll always rely on elements
            # of the dependency graph that don't include these inputs to predict them (e.g., predict the
            # contents of the event given the time at which the event occurred).
            dynamic_indices = batch["dynamic_indices"]
            tensor_idx = batch["dynamic_measurement_indices"] == measurement_idx

            if classification_mode == DataModality.SINGLE_LABEL_CLASSIFICATION:
                # As there is only one index of this type for this setting,
                # we can directly multiply by the mask and sum
                events_with_label = tensor_idx.any(dim=-1)
                is_obs_loss = self.is_observed_criteria(is_obs_score, events_with_label.float())

                labels = (
                    (dynamic_indices.long() * tensor_idx.long()).sum(dim=-1) - vocab_start
                ) * events_with_label.long()
                # labels is of shape [batch X seq]

                try:
                    loss_per_event = self.classification_criteria[measurement](scores.transpose(1, 2), labels)
                except IndexError as e:
                    print(f"Failed to get loss for {measurement}: {e}!")
                    print(f"vocab_start: {vocab_start}, vocab_end: {vocab_end}")
                    print(f"max(labels): {labels.max()}, min(labels): {labels.min()}")
                    print(
                        f"max(dynamic_indices*tensor_idx): {((dynamic_indices*tensor_idx).max())}, "
                        f"min(dynamic_indices*tensor_idx): {((dynamic_indices*tensor_idx).min())}"
                    )
                    print(f"max(tensor_idx.sum(-1)): {tensor_idx.sum(-1).max()}")
                    print(f"scores.shape: {scores.shape}")
                    raise

                event_mask = event_mask & events_with_label

                is_obs_dist = torch.distributions.Bernoulli(logits=is_obs_score)
                dists = torch.distributions.Categorical(logits=scores)

            elif classification_mode == DataModality.MULTI_LABEL_CLASSIFICATION:
                data_labels_or_zero = torch.where(
                    tensor_idx,
                    dynamic_indices - vocab_start + 1,
                    torch.zeros_like(dynamic_indices),
                ).long()

                labels = torch.zeros(
                    scores.shape[0], scores.shape[1], 1 + scores.shape[2], device=scores.device
                ).scatter(
                    dim=2,
                    index=data_labels_or_zero,
                    value=1,
                )

                labels = labels[:, :, 1:]  # Drop the omitted labels...

                loss_per_label = self.classification_criteria[measurement](scores, labels)
                loss_per_event = loss_per_label.mean(dim=-1)

                # Multi-label doesn't use a separate is-observed path, as it handles that natively.
                is_obs_loss = None
                is_obs_dist = None

                dists = torch.distributions.Bernoulli(logits=scores)

            else:
                raise ValueError(f"Classification mode {classification_mode} Invalid!")

            if is_obs_loss is not None:
                loss_per_event = loss_per_event + is_obs_loss
            loss_overall = weighted_loss(loss_per_event, event_mask)

            classification_losses_by_measurement[measurement] = loss_overall
            classification_dists_by_measurement[measurement] = (is_obs_dist, dists)
            classification_labels_by_measurement[measurement] = labels
        return (
            classification_losses_by_measurement,
            classification_dists_by_measurement,
            classification_labels_by_measurement,
        )

    def get_regression_outputs(
        self,
        batch: PytorchBatch,
        encoded: torch.FloatTensor,
        valid_measurements: set[str],
        is_generation: bool = False,
    ) -> tuple[
        dict[str, torch.FloatTensor],
        dict[str, torch.distributions.Distribution],
        dict[str, torch.FloatTensor],
        dict[str, torch.LongTensor],
    ]:
        """Produces regression predictions and losses for the model.

        Args:
            batch: The batch of data for which the regression predictions are desired.
            encoded: The final encodings (of shape batch_size X sequence_length X hidden_dim) **to be used to
                predict for each position in the sequence**. For example, the vector `encoded[i][j]` (which is
                of size `hidden_dim`) is _not_ the summary encoding of the batch element at batch index `i`
                and sequence index `j`, but rather is the input to be used to form regression predictions
                corresponding to batch element `i` at sequence position `j`.
            valid_measurements: The regression measurements in the batch that should be predicted from this
                input `encoded`.

        Returns:
            Four dictionaries:

            * regression_loss_values: A dictionary from `measurement` to scalar tensors consisting of the
              average NLL of the data given the regression model. Averaging happens via the following
              procedure:

              1. NLL is averaged over data elements of the correct measurement per event.
                 TODO(mmd): This is likely a bit wrong; if a regression task has no observed value, that
                 should be taken into account here but I don't think it is currently.
              2. Per-event NLLs are averaged over unmasked events with labels per batch element.
              3. NLL is macro-averaged over the batch.
            * regression_dists: A dictionary from `measurement` to torch distributions modelling the
              regression targets for each data element in each event. In particular, samples from these
              distributions will have shape `[batch_size, sequence_length, num_data_elements_per_event]`, such
              that `sample[i][j][k]` will correspond to a prediction for the regression target indexed
              by `batch['dynamic_indices'][i][j][k]`.
            * regression_labels: A dictionary from `measurement` to tensors of shape
              `[batch_size, sequence_length, num_data_elements_per_event]` containing regression targets for
              each data element, or 0 if that regression target is unobserved.
            * regression_indices: A dictionary from `measurement` to tensors of shape
              `[batch_size, sequence_length, num_data_elements_per_event]` containing the integer index of
              the regression component observed in that position, or 0 if that regression target is
              unobserved. E.g., if we have 200 laboratory tests that we are regressing over, these indices
              state to which laboratory test results the values in `regression_labels` correspond.
        """
        if not valid_measurements:
            return {}, {}, {}, {}

        is_observed_score = self.IsObservedLayer(encoded)

        regression_loss_values = {}
        regression_dists = {}
        regression_labels = {}
        regression_indices = {}
        for measurement in self.config.measurements_for(DataModality.MULTIVARIATE_REGRESSION):
            if measurement not in valid_measurements:
                continue

            event_mask = batch["event_mask"]

            measurement_idx = self.config.measurements_idxmap[measurement]
            vocab_start = self.config.vocab_offsets_by_measurement[measurement]

            # TODO(mmd): If we wanted, we could have `indices_measured_or_zero` reflect just the former part
            # of this `&`, and thus have predictions on all indices, even for those we don't observe values
            # for, but for now this functionality is not required, so we standardize them.
            tensor_idx = (batch["dynamic_measurement_indices"] == measurement_idx) & batch[
                "dynamic_values_mask"
            ]

            indices_measured_or_zero = torch.where(
                tensor_idx,
                batch["dynamic_indices"] - vocab_start,
                torch.zeros_like(batch["dynamic_indices"]),
            ).long()

            regr_dist = self.regression_layers[measurement](
                X=encoded, idx=(None if is_generation else indices_measured_or_zero)
            )

            values_observed_or_zero = torch.where(
                tensor_idx,
                batch["dynamic_values"],
                torch.zeros_like(batch["dynamic_values"]),
            ).float()

            # We don't need to shift here, as given this is a structured model, we'll always rely on elements
            # of the dependency graph that don't include these inputs to predict them (e.g., predict the
            # contents of the event given the time at which the event occurred).

            # TODO(mmd): Use NLL-\beta?
            if is_generation:
                loss_overall = None
            else:
                loss_per_label = -regr_dist.log_prob(values_observed_or_zero)
                loss_per_event, _ = safe_weighted_avg(loss_per_label, tensor_idx)

                events_with_label = event_mask & tensor_idx.any(dim=-1)
                loss_overall = weighted_loss(loss_per_event, events_with_label)

            regression_loss_values[measurement] = loss_overall
            regression_dists[measurement] = (None, regr_dist)
            regression_labels[measurement] = values_observed_or_zero
            regression_indices[measurement] = indices_measured_or_zero

        for measurement in self.config.measurements_for(DataModality.UNIVARIATE_REGRESSION):
            if measurement not in valid_measurements:
                continue

            event_mask = batch["event_mask"]

            measurement_idx = self.config.measurements_idxmap[measurement]

            # We subtract 1 here as the measurement_idx of 0 is withheld for missing data.
            is_obs_score = is_observed_score[:, :, measurement_idx - 1]
            # is_obs_score is of shape [batch X seq]

            tensor_idx = batch["dynamic_measurement_indices"] == measurement_idx
            is_obs_loss = self.is_observed_criteria(is_obs_score, tensor_idx.any(dim=-1).float())

            # As there is only one index of this type for this setting,
            # we can directly multiply by the mask and sum
            tensor_with_labels_idx = tensor_idx & batch["dynamic_values_mask"]
            events_with_label = tensor_with_labels_idx.any(dim=-1)

            event_mask = event_mask & events_with_label

            is_obs_dist = torch.distributions.Bernoulli(logits=is_obs_score)
            regr_dist = self.regression_layers[measurement](X=encoded)

            values_observed_or_zero = (
                torch.where(
                    tensor_with_labels_idx,
                    batch["dynamic_values"],
                    torch.zeros_like(batch["dynamic_values"]),
                )
                .float()
                .sum(dim=-1)
                * events_with_label.float()
            )
            values_observed_or_zero = values_observed_or_zero.unsqueeze(-1)

            # We don't need to shift here, as given this is a structured model, we'll always rely on elements
            # of the dependency graph that don't include these inputs to predict them (e.g., predict the
            # contents of the event given the time at which the event occurred).

            # TODO(mmd): Use NLL-\beta?
            if is_generation:
                loss_overall = None
            else:
                loss_per_event = -regr_dist.log_prob(values_observed_or_zero).squeeze(-1)

                events_with_label = event_mask
                loss_overall = weighted_loss(loss_per_event + is_obs_loss, events_with_label)

            regression_loss_values[measurement] = loss_overall
            regression_dists[measurement] = (is_obs_dist, regr_dist)
            regression_labels[measurement] = values_observed_or_zero
            regression_indices[measurement] = None

        return (
            regression_loss_values,
            regression_dists,
            None if is_generation else regression_labels,
            None if is_generation else regression_indices,
        )

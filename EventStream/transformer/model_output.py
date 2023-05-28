from dataclasses import asdict, dataclass
from typing import Any, Union

import lightning as L
import torch
from mixins import SeedableMixin
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

CATEGORICAL_DIST_T = Union[torch.distributions.Bernoulli, torch.distributions.Categorical]


def get_event_types(
    dynamic_measurement_indices: torch.LongTensor,
    dynamic_indices: torch.LongTensor,
    event_type_measurment_idx: int,
    event_type_vocab_offset: int,
) -> torch.LongTensor:
    event_type_mask = dynamic_measurement_indices == event_type_measurment_idx

    num_event_types = event_type_mask.sum(-1)
    torch._assert(
        (num_event_types <= 1).all().all(), f"Got {num_event_types.max()} event types per event!"
    )

    return torch.where(event_type_mask, dynamic_indices - event_type_vocab_offset, 0).sum(-1)


# TODO(mmd): Move to batch class?
def get_event_type_mask_per_measurement(
    dynamic_measurement_indices: torch.LongTensor,
    dynamic_indices: torch.LongTensor,
    config: StructuredTransformerConfig,
) -> dict[str, torch.BoolTensor | None]:
    if config.event_types_per_measurement is None:
        return None

    event_type_indices = get_event_types(
        dynamic_measurement_indices,
        dynamic_indices,
        config.measurements_idxmap["event_type"],
        config.vocab_offsets_by_measurement["event_type"],
    )

    out_masks = {}
    for measurement, valid_event_types in config.event_types_per_measurement.items():
        valid_event_types = config.event_types_per_measurement[measurement]
        valid_event_type_indices = {config.event_types_idxmap[et] for et in valid_event_types}

        # We only want to predict for events that are of the correct type.
        out_masks[measurement] = torch.any(
            torch.stack([(event_type_indices == i) for i in valid_event_type_indices], 0), dim=0
        )
    return out_masks


def strip_unused_indices(dynamic_indices, *other_tensors):
    is_present = dynamic_indices != 0

    present_indices = torch.argwhere(is_present)
    present_rows = present_indices[:, 0]
    col_counts = torch.ones_like(present_rows).cumsum(0)
    present_row_change = torch.cat(
        [torch.ones_like(present_rows[:1]), (present_rows.diff() != 0).long()], 0
    )

    present_cols = col_counts - (col_counts * present_row_change).cummax(0)[0]

    device = dynamic_indices.device
    index = torch.zeros(dynamic_indices.shape[0], is_present.sum(-1).max(), device=device).long()
    mask = torch.zeros(dynamic_indices.shape[0], is_present.sum(-1).max(), device=device).bool()

    try:
        index.index_put_((present_rows, present_cols), present_indices[:, 1])
        mask.index_put_(
            (present_rows, present_cols), torch.ones_like(present_indices[:, 1]).bool()
        )
    except IndexError as e:
        print(dynamic_indices)
        print(index)
        print(present_indices)
        print(present_rows)
        print(present_row_change)
        print(col_counts)
        print(col_counts * present_row_change)
        print((col_counts * present_row_change).cummax(0)[0])
        print(present_cols)
        raise e

    def idx_fn(T: torch.Tensor) -> torch.Tensor:
        return torch.where(
            mask, torch.gather(T, -1, index=index), torch.zeros_like(index, dtype=T.dtype)
        )

    if not other_tensors:
        return idx_fn(dynamic_indices)
    else:
        return tuple([idx_fn(dynamic_indices), *[idx_fn(T) for T in other_tensors]])


class NestedIndexableMixin:
    @staticmethod
    def _recursive_slice(val: Any, idx: INDEX_SELECT_T):
        if val is None:
            return val
        elif isinstance(val, dict):
            return {k: NestedIndexableMixin._recursive_slice(v, idx) for k, v in val.items()}
        elif isinstance(val, torch.distributions.Distribution):
            return idx_distribution(val, idx)
        else:
            return val[idx]

    def slice(self, idx: INDEX_SELECT_T):
        """Allows for performing joint index selection option on the nested elements."""

        return self.__class__(**self._recursive_slice(asdict(self), idx))


@dataclass
class TransformerOutputWithPast(ModelOutput):
    """Transformer Model Outputs, with optional past key values and hidden states."""

    last_hidden_state: torch.FloatTensor = None
    past_key_values: tuple[tuple[torch.FloatTensor]] | dict[
        str, tuple[torch.FloatTensor]
    ] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@dataclass
class GenerativeSequenceModelLosses(ModelOutput):
    """Losses for the GenerativeSequenceModel head, split by task type."""

    classification: dict[str, torch.FloatTensor] | None = None
    regression: dict[str, torch.FloatTensor] | None = None
    time_to_event: torch.FloatTensor | None = None


@dataclass
class GenerativeSequenceModelSamples(ModelOutput):
    """A single sample (event) of a generative sequence model.

    Args:
        `event_mask` (`torch.BoolTensor`, default: None):
            Shape: [batch_size,]
            Is `True` if and only if the event at that batch position truly exists. Is unset (and has the
            value `None` if all events exist.
        `time_to_event` (`torch.FloatTensor`, default: None):
            Shape: [batch_size,]
            Is 0 if the event does not exist, otherwise quantifies the time between the prior event and this
            event in the series. Should not be None in practice.
        `classification` (`Dict[str, torch.LongTensor]`, default: None):
            Shape: {
                measurement:
                    [batch_size,] if measurement is single label classification or [batch_size, vocab_size]
            }
            If a prediction for measurement is present, then at that key, the tensor contains either the class
            index (starting at 0, not the global offset) for the prediction for that data type for this event
            or per-label binary labels for multi label data types for the prediction for that data type. If
            the event is not present, all predictions will be zero.
        `regression` (`Dict[str, torch.FloatTensor]`, default: None):
            Shape: {
                measurement:
                    [batch_size,] if measurement is univariate or [batch_size, n_regression_targets]
            }
            If a prediction for measurement is present, then at that key, the tensor contains the
            floating-point predictions for that measurement. If an event is not present, predictions will be
            zero. Predictions are ordered in accordance with the index-labels (starting at zero) for the
            data-type vocabulary contained in regression_indices. If regression_indices is `None`, predictions
            span the entire vocabulary in vocabulary order.
        `regression_indices` (`dict[str, torch.LongTensor | None] | None`, default: None):
            Shape: {measurement: [batch_size, n_regression_targets]}
            Contains the indices for which `self.regression` contains predictions for each data type. If
            `None`, self.regression predictions correspond to the entire vocabulary in vocabulary order.
    """

    event_mask: torch.BoolTensor | None = None
    time_to_event: torch.FloatTensor | None = None
    classification: dict[str, torch.LongTensor] | None = None
    regression: dict[str, torch.FloatTensor] | None = None
    regression_indices: dict[str, torch.LongTensor] | None = None

    def build_new_batch_element(
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
        """This function is used for generation, and builds a new batch element from the prediction
        sample in this object."""

        # Add data elements (indices, values, types, values_mask)
        dynamic_measurement_indices = []
        dynamic_indices = []
        dynamic_values = []
        dynamic_values_mask = []

        # Add event_mask
        event_mask = self.event_mask

        duration_since_start = torch.where(
            batch.event_mask[:, :-1], batch.time_delta[:, :-1], 0
        ).sum(-1)
        new_time = torch.where(
            event_mask, batch.start_time + duration_since_start + self.time_to_event, 0
        )

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

            is_meas_map = (
                batch.dynamic_measurement_indices[:, -1, :] == config.measurements_idxmap[m]
            )
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
            dynamic_values.append(torch.nan_to_num(new_values, 0))
            dynamic_measurement_indices.append(new_measurement_indices)

        if dynamic_indices:
            dynamic_indices = torch.cat(dynamic_indices, 1)
            dynamic_measurement_indices = torch.cat(dynamic_measurement_indices, 1)
            dynamic_values = torch.cat(dynamic_values, 1)
            dynamic_values_mask = torch.cat(dynamic_values_mask, 1)
        else:
            if config.measurements_per_dep_graph_level is None:
                dynamic_indices = torch.zeros(
                    batch.batch_size, 1, 0, dtype=torch.long, device=batch.device
                )
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
        """This function is used for generation, and builds a new batch element from the prediction
        sample in this object."""

        # Add data elements (indices, values, types, values_mask)
        dynamic_measurement_indices = []
        dynamic_indices = []
        dynamic_values = []
        dynamic_values_mask = []

        def add_single_label_classification(
            measurement: str, mask: torch.BoolTensor | None = None
        ):
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

            measurement_indices = config.measurements_idxmap[measurement] * torch.ones_like(
                indices
            )

            if mask is not None:
                try:
                    indices = torch.where(mask, indices, 0)
                    measurement_indices = torch.where(mask, measurement_indices, 0)
                except RuntimeError:
                    print(measurement)
                    print(indices.shape)
                    print(indices)
                    print(mask.shape)
                    print(mask)
                    raise

            dynamic_indices.append(indices.unsqueeze(-1))
            dynamic_measurement_indices.append(measurement_indices.unsqueeze(-1))

        def add_multi_label_classification(measurement: str, mask: torch.BoolTensor | None = None):
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

            if mask is not None:
                try:
                    mask = mask.unsqueeze(-1).expand_as(indices)
                    indices = torch.where(mask, indices, 0)
                    measurement_indices = torch.where(mask, measurement_indices, 0)
                except RuntimeError:
                    print(measurement)
                    print(indices.shape)
                    print(indices)
                    print(mask.shape)
                    print(mask)
                    raise

            dynamic_indices.append(indices)
            dynamic_measurement_indices.append(measurement_indices)

        def add_univariate_regression(measurement: str, mask: torch.BoolTensor | None = None):
            if measurement not in self.regression:
                raise ValueError(f"Attempting to generate improper measurement {measurement}!")

            preds = self.regression[measurement].squeeze(-1)
            if len(preds.squeeze(-1).shape) != 1:
                raise ValueError(f"For {measurement}, expect 1D preds, got {preds.shape}!")

            if mask is not None:
                try:
                    preds = torch.where(mask, preds, 0)
                except RuntimeError:
                    print(measurement)
                    print(preds.shape)
                    print(preds)
                    print(mask.shape)
                    print(mask)
                    raise

            dynamic_values_mask.append(~torch.isnan(preds.unsqueeze(-1)))
            dynamic_values.append(torch.nan_to_num(preds.unsqueeze(-1), nan=0))

        def add_multivariate_regression(
            measurement: str, indices: torch.LongTensor, mask: torch.BoolTensor | None = None
        ):
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
                regressed_values = expand_indexed_regression(
                    regressed_values, regressed_indices, vocab_size
                )
                regressed_values_mask = expand_indexed_regression(
                    regressed_values_mask, regressed_indices, vocab_size
                )

            vocab_offset = config.vocab_offsets_by_measurement[measurement]
            try:
                if mask is None:
                    mask = indices >= vocab_offset
                else:
                    mask = mask & (indices >= vocab_offset)
                idx_gather_T = torch.where(mask, indices - vocab_offset, 0).long()

                values = regressed_values.gather(-1, idx_gather_T)
                values_mask = regressed_values_mask.gather(-1, idx_gather_T)
            except RuntimeError:
                print(f"Failed on {measurement} with {indices.shape} indices")
                print(f"Vocab offset: {vocab_offset}")
                print(f"Indices:\n{indices}")
                raise

            if mask is not None:
                values = torch.where(mask, values, 0)
                values_mask = torch.where(mask, values_mask, False)

            dynamic_values.append(values)
            dynamic_values_mask.append(values_mask)

        event_type_mask_kwargs = {
            "dynamic_measurement_indices": batch.dynamic_measurement_indices[:, -1],
            "dynamic_indices": batch.dynamic_indices[:, -1],
            "config": config,
        }

        if "event_type" in measurements_to_build:
            add_single_label_classification("event_type")

            # Event type has no value associated with it.
            dynamic_values.append((0 * dynamic_indices[-1]).float())
            dynamic_values_mask.append((0 * dynamic_indices[-1]).bool())

            event_type_mask_kwargs["dynamic_measurement_indices"] = dynamic_measurement_indices[-1]
            event_type_mask_kwargs["dynamic_indices"] = dynamic_indices[-1]

        event_type_mask_per_measurement = get_event_type_mask_per_measurement(
            **event_type_mask_kwargs
        )

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

            event_type_mask = event_type_mask_per_measurement.get(m, None)

            match (modality, group_mode):
                case (DataModality.SINGLE_LABEL_CLASSIFICATION, None):
                    add_single_label_classification(m, mask=event_type_mask)
                    dynamic_values.append((0 * dynamic_indices[-1]).float())
                    dynamic_values_mask.append((0 * dynamic_indices[-1]).bool())
                case (DataModality.MULTI_LABEL_CLASSIFICATION, None):
                    assert group_mode is None

                    add_multi_label_classification(m, mask=event_type_mask)

                    dynamic_values.append((0 * dynamic_indices[-1]).float())
                    dynamic_values_mask.append((0 * dynamic_indices[-1]).bool())
                case (DataModality.UNIVARIATE_REGRESSION, None):
                    add_univariate_regression(m, mask=event_type_mask)
                    indices = config.vocab_offsets_by_measurement[m] + torch.zeros_like(
                        dynamic_values[-1]
                    )
                    measurement_indices = config.measurements_idxmap[m] * torch.ones_like(indices)

                    dynamic_indices.append(indices)
                    dynamic_measurement_indices.append(measurement_indices)
                case (
                    DataModality.MULTIVARIATE_REGRESSION,
                    None | MeasIndexGroupOptions.CATEGORICAL_AND_NUMERICAL,
                ):
                    add_multi_label_classification(m, mask=event_type_mask)
                    if event_type_mask is not None:
                        etm = event_type_mask.unsqueeze(-1).expand_as(dynamic_indices[-1])
                    else:
                        etm = None
                    add_multivariate_regression(m, indices=dynamic_indices[-1], mask=etm)
                case (
                    DataModality.MULTIVARIATE_REGRESSION,
                    MeasIndexGroupOptions.CATEGORICAL_ONLY,
                ):
                    add_multi_label_classification(m, mask=event_type_mask)
                    dynamic_values.append((0 * dynamic_indices[-1]).float())
                    dynamic_values_mask.append((0 * dynamic_indices[-1]).bool())
                case (DataModality.MULTIVARIATE_REGRESSION, MeasIndexGroupOptions.NUMERICAL_ONLY):
                    meas_index = config.measurements_idxmap[m]
                    existing_mask = batch.dynamic_measurement_indices[:, -1] == meas_index

                    indices = torch.where(existing_mask, batch.dynamic_indices[:, -1], 0)

                    indices = strip_unused_indices(indices)
                    measurement_indices = meas_index * torch.ones_like(indices)

                    if event_type_mask is not None:
                        event_type_mask = event_type_mask.unsqueeze(-1).expand_as(indices)
                        try:
                            indices = torch.where(event_type_mask, indices, 0)
                            measurement_indices = torch.where(
                                event_type_mask, measurement_indices, 0
                            )
                        except RuntimeError:
                            print(indices.shape)
                            print(event_type_mask.shape)
                            raise

                    dynamic_indices.append(indices)
                    dynamic_measurement_indices.append(measurement_indices)

                    add_multivariate_regression(m, indices=indices, mask=event_type_mask)
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
        n_data_elements_old = batch.dynamic_indices.shape[-1]
        n_data_elements_new = new_dynamic_indices.shape[-1]

        dynamic_indices = batch.dynamic_indices
        dynamic_measurement_indices = batch.dynamic_measurement_indices
        dynamic_values = batch.dynamic_values
        dynamic_values_mask = batch.dynamic_values_mask

        if n_data_elements_new < n_data_elements_old:
            data_delta = n_data_elements_old - n_data_elements_new
            new_dynamic_indices = torch.nn.functional.pad(
                new_dynamic_indices, (0, data_delta), value=0
            )
            new_dynamic_measurement_indices = torch.nn.functional.pad(
                new_dynamic_measurement_indices, (0, data_delta), value=0
            )
            new_dynamic_values = torch.nn.functional.pad(
                new_dynamic_values, (0, data_delta), value=0
            )
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
            dynamic_values_mask = torch.nn.functional.pad(
                dynamic_values_mask, (0, data_delta), value=False
            )

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
        """This function builds a new batch element from self, then appends it to the end of the
        input batch.

        TODO(mmd): should this function only append the new event time, every time?
        """

        (
            new_event_time_delta,
            new_event_mask,
            new_dynamic_indices,
            new_dynamic_measurement_indices,
            new_dynamic_values,
            new_dynamic_values_mask,
        ) = self.build_new_batch_element(batch, config)

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

        dynamic_indices = torch.cat(
            (dynamic_indices, new_dynamic_indices.unsqueeze(seq_dim)), seq_dim
        )
        dynamic_measurement_indices = torch.cat(
            (dynamic_measurement_indices, new_dynamic_measurement_indices.unsqueeze(seq_dim)),
            seq_dim,
        )
        dynamic_values = torch.cat(
            (dynamic_values, new_dynamic_values.unsqueeze(seq_dim)), seq_dim
        )
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
        """This function updates the last batch element from self."""

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
        ) = self.format_updates_to_last_batch_event(
            batch, config, measurements_to_build=measurements_to_fill
        )

        # The `format_updates_to_last_batch_event` function takes care of only building the relevant metrics,
        # including building either just categorical elements or categorical and numerical or numerical only.
        # However, in the case where we build numerical only, we end up appending the categorical value
        # indices again, which we want to remove.
        prev_measurements_to_drop_idx = torch.zeros_like(prev_dynamic_indices, dtype=torch.bool)

        for m in measurements_to_fill:
            if (type(m) is not tuple) or (m[1] != MeasIndexGroupOptions.NUMERICAL_ONLY):
                continue

            m = m[0]
            prev_measurements_to_drop_idx |= (
                prev_dynamic_measurement_indices == config.measurements_idxmap[m]
            )

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
class GenerativeSequenceModelPredictions(ModelOutput, NestedIndexableMixin, SeedableMixin):
    """Predictions for the GenerativeSequenceModel head, split by task type."""

    classification: dict[str, CATEGORICAL_DIST_T] | None = None
    regression: dict[str, torch.distributions.Distribution] | None = None
    regression_indices: dict[str, torch.LongTensor] | None = None
    time_to_event: torch.distributions.Distribution | None = None

    def mode(self, event_mask: torch.BoolTensor) -> GenerativeSequenceModelSamples:
        """Returns a mode (not guaranteed to be unique or maximal) of each of the contained
        distributions."""

        return GenerativeSequenceModelSamples(
            event_mask=event_mask[:, -1].detach(),
            classification={k: v.mode for k, v in self.classification.items()},
            regression={k: v.mode for k, v in self.regression.items()},
            regression_indices=self.regression_indices,
            time_to_event=self.time_to_event.mode,
        )

    @SeedableMixin.WithSeed
    def sample(
        self,
        event_mask: torch.BoolTensor,
        seed: int | None = None,
    ) -> GenerativeSequenceModelSamples:
        """Returns a sample from the nested distributions."""

        if seed is not None:
            L.seed_everything(seed)

        return GenerativeSequenceModelSamples(
            event_mask=event_mask[:, -1].detach(),
            classification={k: v.sample() for k, v in self.classification.items()},
            regression={k: v.sample() for k, v in self.regression.items()},
            regression_indices=self.regression_indices,
            time_to_event=self.time_to_event.sample() if self.time_to_event is not None else None,
        )


@dataclass
class GenerativeSequenceModelLabels(ModelOutput):
    """Labels for the GenerativeSequenceModel head, split by task type."""

    # Single-label classification task labels will have shape batch X seq and have raw integer labels in
    # it, whereas multi-label classification task labels will have shape batch X seq X vocab size and have
    # binary indicators for each label.
    classification: dict[str, torch.LongTensor] | None = None
    regression: dict[str, torch.FloatTensor] | None = None
    regression_indices: dict[str, torch.LongTensor] | None = None
    time_to_event: torch.FloatTensor | None = None


@dataclass
class GenerativeSequenceModelOutput(ModelOutput):
    """All GenerativeSequenceModel outputs, including losses, predictions, labels, and masks."""

    loss: torch.FloatTensor
    losses: GenerativeSequenceModelLosses | None = None
    preds: GenerativeSequenceModelPredictions | None = None
    labels: GenerativeSequenceModelLabels | None = None
    event_type_mask_per_measurement: dict[str, torch.BoolTensor] | None = None
    event_mask: torch.BoolTensor | None = None
    dynamic_values_mask: torch.BoolTensor | None = None

    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@dataclass
class StreamClassificationModelOutput(ModelOutput):
    """All GenerativeSequenceModel outputs, including losses, predictions, labels, and masks."""

    loss: torch.FloatTensor
    preds: torch.FloatTensor = None
    labels: torch.LongTensor | torch.FloatTensor = None


class GenerativeOutputLayerBase(torch.nn.Module):
    # TODO(mmd): Allow for use of NLL-beta throughout?
    # TODO(mmd): Per-subject, NLL should be averaged over total duration, not # of events?
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

        self.ClassificationLayer = torch.nn.Linear(config.hidden_size, config.vocab_size)

        self.classification_criteria = {}
        for measurement in config.measurements_for(DataModality.SINGLE_LABEL_CLASSIFICATION):
            self.classification_criteria[measurement] = torch.nn.CrossEntropyLoss(reduction="none")
        for measurement in config.measurements_for(DataModality.MULTI_LABEL_CLASSIFICATION):
            self.classification_criteria[measurement] = torch.nn.BCEWithLogitsLoss(
                reduction="none"
            )

        self.regression_layers = torch.nn.ModuleDict({})
        for measurement in config.measurements_for(DataModality.MULTIVARIATE_REGRESSION):
            self.regression_layers[measurement] = GaussianIndexedRegressionLayer(
                n_regression_targets=config.vocab_sizes_by_measurement[measurement],
                in_dim=config.hidden_size,
            )
        for measurement in config.measurements_for(DataModality.UNIVARIATE_REGRESSION):
            if measurement in self.regression_layers:
                raise ValueError(f"{measurement} duplicated!")
            self.regression_layers[measurement] = GaussianRegressionLayer(
                in_dim=config.hidden_size
            )

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
        """Produces time-to-event predictions and log likelihoods (_not NLLs!_) for the model.

        Args:
            `batch` (`PytorchBatch`):
                The batch of data for which the classification predictions are desired.
            `encoded` (`torch.FloatTensor`, shape is batch_size X sequence_length X hidden_dim):
                The final encodings _to be used to predict the time from the event at that position to the
                subsequent event_. For example, the vector `encoded[i][j]` (which is of size `hidden_dim` and
                corresponds to event `j` for batch element `i`) is
                _not_ used to predict the time from event `j-1` to event `j`, but rather is used to predict
                the time from event `j` to event `j+1` (all for batch index `i`, of course). _Note that this
                is shifted from how `encoded` is used in other functions in this class._

        Returns:
            `TTE_LL` (`torch.FloatTensor`):
                A torch scalar containing the average log-likelihood of observed time-to-events given the
                predicted distribution. Averaging is done over all unmasked events per batch element first,
                then in a macro manner over all batch elements.
                TODO(mmd): Should probably be NLL, not LL.
            `TTE_dist` (`torch.distributions.Distribution`):
                The predicted torch Distribution for modelling time-to-event. The distribution's shape is such
                that samples drawn from the distribution will have shape `[batch_size, sequence_length]` and
                `sample[i][j]` will be a prediction for the time between events `j` and `j+1` for batch
                element `i` (note that this includes a prediction for the time until the event after the end
                of the sequence, though such an event is naturally not observed).
            `TTE_true` (`torch.FloatTensor`):
                A tensor of shape `[batch_size, sequence_length - 1]` such that `TTE_true[i][j]` contains the
                observed time between events `j` and `j+1` for batch element `i`.
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
        TTE_true_exp = torch.cat(
            (TTE_true, torch.ones_like(TTE_true[:, -1]).unsqueeze(-1)), dim=-1
        )
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

        TTE_LL_per_patient = (TTE_LL * TTE_obs_mask_exp.float()).sum(
            -1
        ) / TTE_obs_mask_exp.float().sum(-1)
        TTE_LL_overall = TTE_LL_per_patient.mean()

        return TTE_LL_overall, TTE_dist, TTE_true

    def get_classification_outputs(
        self,
        batch: PytorchBatch,
        encoded: torch.FloatTensor,
        valid_measurements: set[str],
        event_type_mask_per_measurement: dict[str, torch.BoolTensor] | None = None,
    ) -> tuple[
        dict[str, torch.FloatTensor],
        dict[str, torch.FloatTensor],
        dict[str, torch.LongTensor | torch.FloatTensor],
    ]:
        """Produces classification predictions and losses for the model.

        Args:
            `batch` (`PytorchBatch`):
                The batch of data for which the classification predictions are desired.
            `encoded` (`torch.FloatTensor`, shape is batch_size X sequence_length X hidden_dim):
                The final encodings _to be used to predict for each position in the sequence_. For example,
                the vector `encoded[i][j]` (which is of size `hidden_dim`) is _not_ the summary encoding of
                the batch element at batch index `i` and sequence index `j`, but rather is the input to be
                used to form classification predictions corresponding to batch element `i` at sequence
                position `j`.
            `valid_measurements` (`Set[str]`):
                The classification measurements in the batch that should be predicted from this input
                `encoded`.
            `event_type_mask_per_measurement` (`Optional[Dict[str, torch.BoolTensor]]`, defaults to None):
                A dictionary from measurement to a tensor of shape `[batch_size, sequence_length]` such that
                `event_type_mask_per_measurement[measurement][i][j]` is `True` if the event at batch index `i`
                and sequence index `j` is of a type that should be used to form predictions for the
                measurement `measurement`. If `None`, then all events are used to form predictions for all
                measurements.

        Returns:
            `classification_losses_by_measurement` (`Dict[str, torch.FloatTensor]`):
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
            `classification_dists_by_measurement` (`Dict[str, torch.FloatTensor]`):
                A dictionary from `measurement` to classification distributions of shape
                `[batch_size X sequence_length X vocabulary_size]` or `[batch_size X sequence_length]`
                reflecting the probabilities for each event for that measurement. Returns scores for all
                events, even those that are masked, including the final event.
            `classification_labels_by_measurement` (`Dict[str, Union[torch.LongTensor, torch.FloatTensor]]`):
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

        # Classification of what elements are going to occur:
        classification_scores = self.ClassificationLayer(encoded)

        classification_losses_by_measurement = {}
        classification_dists_by_measurement = {}
        classification_labels_by_measurement = {}

        for measurement, classification_mode in self.classification_mode_per_measurement.items():
            if measurement not in valid_measurements:
                continue

            if (
                event_type_mask_per_measurement is not None
                and measurement in event_type_mask_per_measurement
            ):
                event_mask = event_type_mask_per_measurement[measurement] & batch["event_mask"]
            else:
                event_mask = batch["event_mask"]

            measurement_idx = self.config.measurements_idxmap[measurement]
            vocab_start = self.config.vocab_offsets_by_measurement[measurement]
            vocab_end = min(
                o
                for o in list(self.config.vocab_offsets_by_measurement.values())
                + [self.config.vocab_size]
                if o > vocab_start
            )

            scores = classification_scores[:, :, vocab_start:vocab_end]
            # scores is of shape [batch X seq X vocab_end-vocab_start]

            # We don't need to shift here, as given this is a structured model, we'll always rely on elements
            # of the dependency graph that don't include these inputs to predict them (e.g., predict the
            # contents of the event given the time at which the event occurred).
            dynamic_indices = batch["dynamic_indices"]
            tensor_idx = batch["dynamic_measurement_indices"] == measurement_idx

            if classification_mode == DataModality.SINGLE_LABEL_CLASSIFICATION:
                # As there is only one index of this type for this setting,
                # we can directly multiply by the mask and sum
                events_with_label = tensor_idx.any(dim=-1)
                labels = (
                    (dynamic_indices.long() * tensor_idx.long()).sum(dim=-1) - vocab_start
                ) * events_with_label.long()
                # labels is of shape [batch X seq]

                try:
                    loss_per_event = self.classification_criteria[measurement](
                        scores.transpose(1, 2), labels
                    )
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

                dists = torch.distributions.Bernoulli(logits=scores)

            else:
                raise ValueError(f"Classification mode {classification_mode} Invalid!")

            loss_overall = weighted_loss(loss_per_event, event_mask)

            classification_losses_by_measurement[measurement] = loss_overall
            classification_dists_by_measurement[measurement] = dists
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
        event_type_mask_per_measurement: dict[str, torch.BoolTensor] | None = None,
    ) -> tuple[
        dict[str, torch.FloatTensor],
        dict[str, torch.distributions.Distribution],
        dict[str, torch.FloatTensor],
        dict[str, torch.LongTensor],
    ]:
        """Produces regression predictions and losses for the model.

        Args:
            `batch` (`PytorchBatch`):
                The batch of data for which the regression predictions are desired.
            `encoded` (`torch.FloatTensor`, shape is batch_size X sequence_length X hidden_dim):
                The final encodings _to be used to predict for each position in the sequence_. For example,
                the vector `encoded[i][j]` (which is of size `hidden_dim`) is _not_ the summary encoding of
                the batch element at batch index `i` and sequence index `j`, but rather is the input to be
                used to form regression predictions corresponding to batch element `i` at sequence
                position `j`.
            `valid_measurements` (`Set[str]`):
                The regression measurements in the batch that should be predicted from this input `encoded`.
            `event_type_mask_per_measurement` (`Optional[Dict[str, torch.BoolTensor]]`, defaults to None):
                A dictionary from measurement to a tensor of shape `[batch_size, sequence_length]` such that
                `event_type_mask_per_measurement[measurement][i][j]` is `True` if the event at batch index `i`
                and sequence index `j` is of a type that should be used to form predictions for the
                measurement `measurement`. If `None`, then all events are used to form predictions for all
                measurements.

        Returns:
            `regression_loss_values` (`Dict[str, torch.FloatTensor]`):
                A dictionary from `measurement` to scalar tensors consisting of the average NLL of the data
                given the regression model. Averaging happens via the following procedure:
                  1. NLL is averaged over data elements of the correct measurement per event.
                     TODO(mmd): This is likely a bit wrong; if a regression task has no observed value, that
                     should be taken into account here but I don't think it is currently.
                  2. Per-event NLLs are averaged over unmasked events with labels per batch element.
                  3. NLL is macro-averaged over the batch.
            `regression_dists` (`Dict[str, torch.distributions.Distribution]`):
                A dictionary from `measurement` to torch distributions modelling the regression targets for
                each data element in each event. In particular, samples from these distributions will have
                shape
                `[batch_size, sequence_length, num_data_elements_per_event]`, such that `sample[i][j][k]` will
                correspond to a prediction for the regression target indexed by
                `batch['dynamic_indices'][i][j][k]`.
            `regression_labels` (`Dict[str, torch.FloatTensor]`):
                A dictionary from `measurement` to tensors of shape
                `[batch_size, sequence_length, num_data_elements_per_event]` containing regression targets for
                each data element, or 0 if that regression target is unobserved.
            `regression_indices` (`Dict[str, torch.LongTensor]`):
                A dictionary from `measurement` to tensors of shape
                `[batch_size, sequence_length, num_data_elements_per_event]` containing the integer index of
                the regression component observed in that position, or 0 if that regression target is
                unobserved. E.g., if we have 200 laboratory tests that we are regressing over, these indices
                state to which laboratory test results the values in `regression_labels` correspond.
        """
        if not valid_measurements:
            return {}, {}, {}, {}

        regression_loss_values = {}
        regression_dists = {}
        regression_labels = {}
        regression_indices = {}
        for measurement in self.config.measurements_for(DataModality.MULTIVARIATE_REGRESSION):
            if measurement not in valid_measurements:
                continue

            if (
                event_type_mask_per_measurement is not None
                and measurement in event_type_mask_per_measurement
            ):
                event_mask = event_type_mask_per_measurement[measurement] & batch["event_mask"]
            else:
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
            regression_dists[measurement] = regr_dist
            regression_labels[measurement] = values_observed_or_zero
            regression_indices[measurement] = indices_measured_or_zero

        for measurement in self.config.measurements_for(DataModality.UNIVARIATE_REGRESSION):
            if measurement not in valid_measurements:
                continue

            if (
                event_type_mask_per_measurement is not None
                and measurement in event_type_mask_per_measurement
            ):
                event_mask = event_type_mask_per_measurement[measurement] & batch["event_mask"]
            else:
                event_mask = batch["event_mask"]

            measurement_idx = self.config.measurements_idxmap[measurement]

            # TODO(mmd): If we wanted, we could have `indices_measured_or_zero` reflect just the former part
            # of this `&`, and thus have predictions on all indices, even for those we don't observe values
            # for, but for now this functionality is not required, so we standardize them.
            tensor_idx = (batch["dynamic_measurement_indices"] == measurement_idx) & batch[
                "dynamic_values_mask"
            ]

            # As there is only one index of this type for this setting,
            # we can directly multiply by the mask and sum
            events_with_label = tensor_idx.any(dim=-1)
            event_mask = event_mask & events_with_label

            regr_dist = self.regression_layers[measurement](X=encoded)

            values_observed_or_zero = (
                torch.where(
                    tensor_idx,
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

                events_with_label = event_mask & tensor_idx.any(dim=-1)
                loss_overall = weighted_loss(loss_per_event, events_with_label)

            regression_loss_values[measurement] = loss_overall
            regression_dists[measurement] = regr_dist
            regression_labels[measurement] = values_observed_or_zero
            regression_indices[measurement] = None

        return (
            regression_loss_values,
            regression_dists,
            None if is_generation else regression_labels,
            None if is_generation else regression_indices,
        )

    def get_event_type_mask_per_measurement(
        self, batch: PytorchBatch
    ) -> dict[str, torch.BoolTensor | None]:
        return get_event_type_mask_per_measurement(
            batch.dynamic_measurement_indices, batch.dynamic_indices, self.config
        )

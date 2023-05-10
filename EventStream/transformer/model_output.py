from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Union

import torch
from mixins import SeedableMixin
from transformers.utils import ModelOutput

from ..data.data_embedding_layer import MeasIndexGroupOptions
from ..data.types import DataModality, PytorchBatch, TemporalityType
from .config import MEAS_INDEX_GROUP_T, StructuredTransformerConfig
from .utils import INDEX_SELECT_T, expand_indexed_regression, idx_distribution

CATEGORICAL_DIST_T = Union[torch.distributions.Bernoulli, torch.distributions.Categorical]


# TODO(mmd): Move to batch class?
def get_event_type_mask_per_measurement(
    dynamic_measurement_indices: torch.LongTensor,
    dynamic_indices: torch.LongTensor,
    config: StructuredTransformerConfig,
) -> dict[str, torch.BoolTensor | None]:
    datetime.now()
    if config.event_types_per_measurement is None:
        return None

    event_type_mask = dynamic_measurement_indices == config.measurements_idxmap["event_type"]

    num_event_types = event_type_mask.sum(-1)
    torch._assert(
        (num_event_types <= 1).all().all(), f"Got {num_event_types.max()} event types per event!"
    )

    event_type_indices = torch.where(
        event_type_mask, dynamic_indices - config.vocab_offsets_by_measurement["event_type"], 0
    ).sum(-1)

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
    present_row_change = torch.cat([torch.ones_like(present_rows[:1]), present_rows.diff()], 0)

    present_cols = col_counts - (col_counts * present_row_change).cummax(0)[0]

    device = dynamic_indices.device
    index = torch.zeros(dynamic_indices.shape[0], is_present.sum(-1).max(), device=device).long()
    mask = torch.zeros(dynamic_indices.shape[0], is_present.sum(-1).max(), device=device).bool()
    index.index_put_((present_rows, present_cols), present_indices[:, 1])
    mask.index_put_((present_rows, present_cols), torch.ones_like(present_indices[:, 1]).bool())

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
    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None
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

        dynamic_indices = torch.cat(dynamic_indices, 1)
        dynamic_measurement_indices = torch.cat(dynamic_measurement_indices, 1)
        dynamic_values = torch.cat(dynamic_values, 1)
        dynamic_values_mask = torch.cat(dynamic_values_mask, 1)

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
                print(f"WARNING: Attempting to generate improper measurement {measurement}!")
                return

            preds = self.classification[measurement]

            if len(preds.shape) != 1:
                raise ValueError(f"For {measurement}, expect 1D preds, got {preds.shape}!")
            if (preds >= vocab_size).any():
                raise ValueError("For {measurement}, need preds < vocab_size!")
            indices = (vocab_offset + preds)

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

            indices = torch.arange(vocab_size).long() + vocab_offset
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
                    add_multivariate_regression(
                        m, indices=dynamic_indices[-1], mask=event_type_mask
                    )
                case (DataModality.MULTIVARIATE_REGRESSION, MeasIndexGroupOptions.CATEGORICAL_ONLY):
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
        measurements_to_fill: set[MEAS_INDEX_GROUP_T],
    ) -> PytorchBatch:
        """This function updates the last batch element from self."""
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
    def sample(self, event_mask: torch.BoolTensor) -> GenerativeSequenceModelSamples:
        """Returns a sample from the nested distributions."""

        return GenerativeSequenceModelSamples(
            event_mask=event_mask[:, -1].detach(),
            classification={k: v.sample() for k, v in self.classification.items()},
            regression={k: v.sample() for k, v in self.regression.items()},
            regression_indices=self.regression_indices,
            time_to_event=self.time_to_event.sample(),
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

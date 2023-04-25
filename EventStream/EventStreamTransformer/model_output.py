import torch, numpy as np, pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime
from mixins import SeedableMixin
from transformers.utils import ModelOutput
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Set

from .utils import INDEX_SELECT_T, idx_distribution, expand_indexed_regression
from .config import StructuredEventStreamTransformerConfig, MEAS_INDEX_GROUP_T
from ..EventStreamData.types import TemporalityType, DataModality, EventStreamPytorchBatch
from ..EventStreamData.event_stream_dataset_base import EventStreamDatasetBase
from ..EventStreamData.data_embedding_layer import MeasIndexGroupOptions
from ..EventStreamData.config import MeasurementConfig

CATEGORICAL_DIST_T = Union[torch.distributions.Bernoulli, torch.distributions.Categorical]

# TODO(mmd): Move to batch class?
def get_event_type_mask_per_measurement(
    dynamic_measurement_indices: torch.LongTensor,
    dynamic_indices: torch.LongTensor,
    config: StructuredEventStreamTransformerConfig,
) -> Dict[str, Optional[torch.BoolTensor]]:
    if config.event_types_per_measurement is None: return None

    event_type_mask = (
        dynamic_measurement_indices == config.measurements_idxmap['event_type']
    )

    event_type_indices = torch.where(
        event_type_mask, dynamic_indices - config.vocab_offsets_by_measurement['event_type'], -1
    )

    out_masks = {}
    for measurement, valid_event_types in config.event_types_per_measurement.items():
        valid_event_types = config.event_types_per_measurement[measurement]
        valid_event_type_indices = {config.event_types_idxmap[et] for et in valid_event_types}

        # We only want to predict for events that are of the correct type.
        out_masks[measurement] = torch.any(
            torch.stack([(event_type_indices == i) for i in valid_event_type_indices], 0),
            dim=0
        ).any(-1)
    return out_masks

class NestedIndexableMixin:
    @staticmethod
    def _recursive_slice(val: Any, idx: INDEX_SELECT_T):
        if val is None: return val
        elif isinstance(val, dict):
            return {k: NestedIndexableMixin._recursive_slice(v, idx) for k, v in val.items()}
        elif isinstance(val, torch.distributions.Distribution): return idx_distribution(val, idx)
        else: return val[idx]

    def slice(self, idx: INDEX_SELECT_T):
        """Allows for performing joint index selection option on the nested elements."""

        return self.__class__(**self._recursive_slice(asdict(self), idx))

@dataclass
class EventStreamTransformerOutputWithPast(ModelOutput):
    """ EventStreamTransformer Model Outputs, with optional past key values and hidden states.  """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class GenerativeSequenceModelLosses(ModelOutput):
    """Losses for the GenerativeSequenceModel head, split by task type."""
    classification: Optional[Dict[str, torch.FloatTensor]] = None
    regression: Optional[Dict[str, torch.FloatTensor]] = None
    time_to_event: Optional[torch.FloatTensor] = None

@dataclass
class GenerativeSequenceModelSamples(ModelOutput):
    """
    A single sample (event) of a generative sequence model.

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
            Shape: {measurement: [batch_size, n_regression_targets]}
            If a prediction for measurement is present, then at that key, the tensor contains the floating-point
            predictions for that measurement. If an event is not present, predictions will be zero. Predictions
            are ordered in accordance with the index-labels (starting at zero) for the data-type vocabulary
            contained in regression_indices. If regression_indices is `None`, predictions span the entire
            vocabulary in vocabulary order.
        `regression_indices` (`Dict[str, torch.LongTensor]`, defaul: None):
            Shape: {measurement: [batch_size, n_regression_targets]}
            Contains the indices for which `self.regression` contains predictions for each data type. If
            `None`, self.regression predictions correspond to the entire vocabulary in vocabulary order.
    """

    event_mask: Optional[torch.BoolTensor] = None
    time_to_event: Optional[torch.FloatTensor] = None
    classification: Optional[Dict[str, torch.LongTensor]] = None
    regression: Optional[Dict[str, torch.FloatTensor]] = None
    regression_indices: Optional[Dict[str, torch.LongTensor]] = None

    def set_event_mask(self, event_mask: torch.BoolTensor):
        self.event_mask = event_mask
        event_mask_exp = event_mask.unsqueeze(-1)

        self.time_to_event = torch.where(self.event_mask, self.time_to_event, 0)

        new_classification = {}
        for k, v in self.classification.items():
            if len(v.shape) == 1: new_classification[k] = torch.where(self.event_mask, v, 0)
            else: new_classification[k] = torch.where(event_mask_exp.expand_as(v), v, 0)
        self.classification = new_classification

        new_regression = {}
        for k, v in self.regression.items():
            new_regression[k] = torch.where(event_mask_exp.expand_as(v), v, 0)
        self.regression = new_regression

    def build_new_batch_element(
        self,
        batch: EventStreamPytorchBatch,
        config: StructuredEventStreamTransformerConfig,
        base_dataset: Optional[EventStreamDatasetBase] = None,
        batch_schema: Optional[List[Tuple[int, datetime, datetime]]] = None,
        static_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[
        torch.FloatTensor, torch.BoolTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor,
        torch.BoolTensor
    ]:
        """
        This function is used for generation, and builds a new batch element from the prediction sample in
        this object.
        """

        # Add data elements (indices, values, types, values_mask)
        dynamic_measurement_indices = []
        dynamic_indices = []
        dynamic_values = []
        dynamic_values_mask = []

        # Add time
        time = batch.time[:, -1] + self.time_to_event

        # Add event_mask
        event_mask = self.event_mask

        # Add time-dependent values if present.
        for m, cfg in base_dataset.measurement_configs.items():
            if cfg.temporality != TemporalityType.FUNCTIONAL_TIME_DEPENDENT: continue
            if cfg.modality == DataModality.DROPPED: continue

            # Produce the functional, time-dependent outputs.

            # TODO(mmd): This may be wrong in some cases! Don't know the actual start time as it is
            # initialized to zero!
            fn = cfg.functor
            subjects = [subj for subj, _, _ in batch_schema]
            start_time = [st for _, st, _ in batch_schema]
            time_inputs = pd.DataFrame(
                {'delta_time_min': time.detach().cpu().numpy(), 'start_time': start_time},
                index = pd.Index(subjects, name='subject_id'),
            )
            time_vals = time_inputs.start_time + pd.to_timedelta(time_inputs.delta_time_min, unit='minutes')

            vals = fn(time_vals, static_data)

            vals_df = pd.DataFrame({'vals': vals})
            vals_df['key'] = config.vocab_offsets_by_measurement[m]

            # Post-process the raw values
            match cfg.modality:
                case DataModality.DROPPED: continue
                case DataModality.SINGLE_LABEL_CLASSIFICATION:
                    assert cfg.vocabulary is not None
                    vals_df['key'] += vals_df['vals'].apply(lambda e: cfg.vocabulary.idxmap.get(e, 0))
                    vals_df['vals'] = np.NaN
                case DataModality.UNIVARIATE_REGRESSION:
                    vals_df['is_inlier'] = pd.Series([None] * len(vals_df), dtype='boolean')
                    new_keys = EventStreamDatasetBase.transform_categorical_values_series(
                        measurement_metadata=cfg.measurement_metadata, vals=vals_df.vals
                    )

                    vals_df = EventStreamDatasetBase._transform_numerical_metadata_column_vals(
                        vals_df, cfg.measurement_metadata, val_col='vals', inlier_col='is_inlier'
                    )

                    if new_keys is not None:
                        assert cfg.vocabulary is not None
                        vals_df['key'] += new_keys.apply(lambda e: cfg.vocabulary.idxmap.get(e, 0))
                case _: raise ValueError(f"Unsupported modality {cfg.modality} for {m}")

            # Convert to indices and values.
            new_indices = torch.LongTensor(vals_df.key.values, device=time.device).unsqueeze(-1)
            new_values = torch.FloatTensor(vals_df.vals.values, device=time.device).unsqueeze(-1)
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
            time, event_mask, dynamic_indices, dynamic_measurement_indices, dynamic_values,
            dynamic_values_mask
        )

    def format_updates_to_last_batch_event(
        self,
        batch: EventStreamPytorchBatch,
        config: StructuredEventStreamTransformerConfig,
        measurements_to_build: Optional[Set[MEAS_INDEX_GROUP_T]] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.BoolTensor]:
        """
        This function is used for generation, and builds a new batch element from the prediction sample in
        this object.
        """

        # Add data elements (indices, values, types, values_mask)
        dynamic_measurement_indices = []
        dynamic_indices = []
        dynamic_values = []
        dynamic_values_mask = []

        def add_classification_measurement(measurement: str, mask: Optional[torch.BoolTensor] = None):
            # Add the data index.
            if measurement not in self.classification: return

            vocab_offset = config.vocab_offsets_by_measurement[measurement]
            vocab_size = config.vocab_sizes_by_measurement[measurement]
            preds = self.classification[measurement]

            classification_generative_mode = None
            for classification_mode in (
                DataModality.SINGLE_LABEL_CLASSIFICATION,
                DataModality.MULTI_LABEL_CLASSIFICATION,
            ):
                if measurement in config.measurements_per_generative_mode[classification_mode]:
                    assert classification_generative_mode is None
                    classification_generative_mode = classification_mode

            assert classification_generative_mode is not None

            match classification_generative_mode:
                case DataModality.SINGLE_LABEL_CLASSIFICATION:
                    # In this case, preds should be of shape [batch_size,] and contain the index (offset from
                    # zero) of the label that was classified.
                    assert len(preds.shape) == 1
                    assert (preds < vocab_size).all()
                    indices = (vocab_offset + preds).unsqueeze(-1)
                case DataModality.MULTI_LABEL_CLASSIFICATION:
                    # In this case, preds should be of shape [batch_size, vocab_size] and contain binary
                    # predictions of all predicted elements.
                    assert len(preds.shape) == 2
                    assert vocab_size == preds.shape[-1]

                    indices = torch.arange(vocab_size).long() + vocab_offset
                    indices = indices.unsqueeze(0).expand_as(preds)
                    indices = torch.where(preds == 1, indices, 0)

                    present_mask = (indices != 0).any(dim=0)
                    indices = indices[:, present_mask]
                case _: raise NotImplementedError(
                    f"Classification mode {classification_generative_mode}) not recognized."
                )

            measurement_indices = config.measurements_idxmap[measurement] * torch.ones_like(indices)

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

            dynamic_indices.append(indices)
            dynamic_measurement_indices.append(measurement_indices)

        def add_regression_measurement(
            measurement: str, indices: torch.LongTensor, mask: Optional[torch.BoolTensor] = None
        ):
            regression_mode = DataModality.MULTIVARIATE_REGRESSION
            if (
                (measurement not in config.measurements_per_generative_mode[regression_mode]) or
                (measurement not in self.regression)
            ):
                values = torch.zeros_like(indices).float()
                values_mask = torch.zeros_like(indices).bool()
            else:
                regressed_values = self.regression[measurement]
                regressed_values_mask = torch.ones_like(regressed_values).bool()
                vocab_size = config.vocab_sizes_by_measurement[measurement]

                # Now we need to align the regressed_indices to the classification indices, as indices we
                # regressed over but don't think were actually observed in the event wouldn't have
                # values. To do this, we'll first expand out over all possible values/targets, if
                # necessary, then sub-select down.
                if (
                    (self.regression_indices is not None) and
                    (measurement in self.regression_indices) and
                    (self.regression_indices[measurement] is not None)
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
                values = regressed_values.gather(-1, indices - vocab_offset)
                values_mask = regressed_values_mask.gather(-1, indices - vocab_offset)

            if mask is not None:
                values = torch.where(mask, values, 0)
                values_mask = torch.where(mask, values, False)

            dynamic_values.append(values)
            dynamic_values_mask.append(values_mask)

        event_type_mask_kwargs = {
            'dynamic_measurement_indices': batch.dynamic_measurement_indices[:, -1],
            'dynamic_indices': batch.dynamic_indices[:, -1],
            'config': config
        }
        if 'event_type' in measurements_to_build:
            add_classification_measurement('event_type')

            # Event type has no value associated with it.
            dynamic_values.append((0 * dynamic_indices[-1]).float())
            dynamic_values_mask.append((0 * dynamic_indices[-1]).bool())

            event_type_mask_kwargs['dynamic_measurement_indices'] = dynamic_measurement_indices[-1]
            event_type_mask_kwargs['dynamic_indices'] = dynamic_indices[-1]

        event_type_mask_per_measurement = get_event_type_mask_per_measurement(**event_type_mask_kwargs)

        for m in measurements_to_build:
            if m == 'event_type': continue

            if type(m) in (list, tuple):
                assert len(m) == 2
                m, group_mode = m
            else:
                group_mode = MeasIndexGroupOptions.CATEGORICAL_AND_NUMERICAL

            event_type_mask = event_type_mask_per_measurement.get(m, None)

            match group_mode:
                case MeasIndexGroupOptions.CATEGORICAL_AND_NUMERICAL:
                    add_classification_measurement(m, mask=event_type_mask)
                    add_regression_measurement(m, indices=dynamic_indices[-1], mask=event_type_mask)

                case MeasIndexGroupOptions.CATEGORICAL_ONLY:
                    add_classification_measurement(m, mask=event_type_mask)
                    dynamic_values.append((0 * dynamic_indices[-1]).float())
                    dynamic_values_mask.append((0 * dynamic_indices[-1]).bool())

                case MeasIndexGroupOptions.NUMERICAL_ONLY:
                    # Get existing dynamic indices
                    meas_index = config.measurements_idxmap[m]
                    existing_mask = (batch.dynamic_measurement_indices[:, -1] == meas_index)

                    indices = torch.where(existing_mask, batch.dynamic_indices[:, -1], 0)
                    present_mask = (indices != 0).any(dim=0)
                    if not present_mask.any(): continue

                    indices = indices[:, present_mask]
                    measurement_indices = meas_index * torch.ones_like(indices)

                    if event_type_mask is not None:
                        try:
                            indices = torch.where(event_type_mask, indices, 0)
                            measurement_indices = torch.where(event_type_mask, measurement_indices, 0)
                        except RuntimeError:
                            print(indices.shape)
                            print(event_type_mask.shape)
                            raise

                    dynamic_indices.append(indices)
                    dynamic_measurement_indices.append(measurement_indices)

                    add_regression_measurement(m, indices=indices, mask=event_type_mask)

                case _: raise ValueError(f"Invalid group mode: {group_mode}")

        dynamic_indices = torch.cat(dynamic_indices, 1)
        dynamic_measurement_indices = torch.cat(dynamic_measurement_indices, 1)
        dynamic_values = torch.cat(dynamic_values, 1)
        dynamic_values_mask = torch.cat(dynamic_values_mask, 1)

        present_mask = (dynamic_indices != 0).any(dim=0)

        dynamic_indices = dynamic_indices[:, present_mask]
        dynamic_measurement_indices = dynamic_measurement_indices[:, present_mask]
        dynamic_values = dynamic_values[:, present_mask]
        dynamic_values_mask = dynamic_values_mask[:, present_mask]

        return dynamic_indices, dynamic_measurement_indices, dynamic_values, dynamic_values_mask

    @staticmethod
    def pad_data_elements(
        batch: EventStreamPytorchBatch,
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
            new_dynamic_indices = torch.nn.functional.pad(new_dynamic_indices, (0, data_delta), value=0)
            new_dynamic_measurement_indices = torch.nn.functional.pad(new_dynamic_measurement_indices, (0, data_delta), value=0)
            new_dynamic_values = torch.nn.functional.pad(new_dynamic_values, (0, data_delta), value=0)
            new_dynamic_values_mask = torch.nn.functional.pad(new_dynamic_values_mask, (0, data_delta), value=False)
        elif n_data_elements_new > n_data_elements_old:
            data_delta = n_data_elements_new - n_data_elements_old
            dynamic_indices = torch.nn.functional.pad(dynamic_indices, (0, data_delta), value=0)
            dynamic_measurement_indices = torch.nn.functional.pad(dynamic_measurement_indices, (0, data_delta), value=0)
            dynamic_values = torch.nn.functional.pad(dynamic_values, (0, data_delta), value=0)
            dynamic_values_mask = torch.nn.functional.pad(dynamic_values_mask, (0, data_delta), value=False)

        return (
            (dynamic_indices, dynamic_measurement_indices, dynamic_values, dynamic_values_mask),
            (new_dynamic_indices, new_dynamic_measurement_indices, new_dynamic_values, new_dynamic_values_mask),
        )

    def append_to_batch(
        self, batch: EventStreamPytorchBatch, config: StructuredEventStreamTransformerConfig,
        base_dataset: Optional[EventStreamDatasetBase] = None,
        batch_schema: Optional[List[Tuple[int, datetime, datetime]]] = None,
        static_data: Optional[pd.DataFrame] = None,
    ) -> EventStreamPytorchBatch:
        """
        This function builds a new batch element from self, then appends it to the end of the input batch.
        TODO(mmd): should this function only append the new event time, every time?
        """

        assert base_dataset is not None, "base_dataset must be provided for structured event generation."
        assert batch_schema is not None, \
            "batch_schema must be provided for structured event generation."
        assert static_data is not None, "static_data must be provided for structured event generation."

        (
            new_event_time, new_event_mask, new_dynamic_indices, new_dynamic_measurement_indices,
            new_dynamic_values, new_dynamic_values_mask,
        ) = self.build_new_batch_element(batch, config, base_dataset, batch_schema, static_data)

        # Combine everything
        seq_dim = 1

        time = torch.cat((batch.time, new_event_time.unsqueeze(seq_dim)), seq_dim)
        event_mask = torch.cat((batch.event_mask, new_event_mask.unsqueeze(seq_dim)), seq_dim)

        # Re-pad data elements.
        (
            (dynamic_indices, dynamic_measurement_indices, dynamic_values, dynamic_values_mask),
            (new_dynamic_indices, new_dynamic_measurement_indices, new_dynamic_values, new_dynamic_values_mask),
        ) = self.pad_data_elements(
            batch, new_dynamic_indices, new_dynamic_measurement_indices, new_dynamic_values,
            new_dynamic_values_mask
        )

        dynamic_indices = torch.cat((dynamic_indices, new_dynamic_indices.unsqueeze(seq_dim)), seq_dim)
        dynamic_measurement_indices = torch.cat(
            (dynamic_measurement_indices, new_dynamic_measurement_indices.unsqueeze(seq_dim)), seq_dim
        )
        dynamic_values = torch.cat((dynamic_values, new_dynamic_values.unsqueeze(seq_dim)), seq_dim)
        dynamic_values_mask = torch.cat(
            (dynamic_values_mask, new_dynamic_values_mask.unsqueeze(seq_dim)), seq_dim
        )

        return EventStreamPytorchBatch(
            time=time,
            event_mask=event_mask,
            dynamic_indices=dynamic_indices,
            dynamic_measurement_indices=dynamic_measurement_indices,
            dynamic_values=dynamic_values,
            dynamic_values_mask=dynamic_values_mask,
        )

    def update_last_event_data(
        self, batch: EventStreamPytorchBatch, config: StructuredEventStreamTransformerConfig,
        measurements_to_fill: Set[MEAS_INDEX_GROUP_T],
    ) -> EventStreamPytorchBatch:
        """This function updates the last batch element from self."""
        if 'time' in measurements_to_fill:
            raise ValueError("You shouldn't ever be trying to fill the 'time' aspect of a batch!")

        prev_dynamic_indices = batch.dynamic_indices[:, -1]
        prev_dynamic_measurement_indices = batch.dynamic_measurement_indices[:, -1]
        prev_dynamic_values = batch.dynamic_values[:, -1]
        prev_dynamic_values_mask = batch.dynamic_values_mask[:, -1]

        (
            new_dynamic_indices, new_dynamic_measurement_indices, new_dynamic_values, new_dynamic_values_mask
        ) = self.format_updates_to_last_batch_event(
            batch, config, measurements_to_build=measurements_to_fill
        )

        # The `format_updates_to_last_batch_event` function takes care of only building the relevant metrics,
        # including building either just categorical elements or categorical and numerical or numerical only.
        # However, in the case where we build numerical only, we end up appending the categorical value
        # indices again, which we want to remove.
        prev_measurements_to_drop_idx = torch.zeros_like(prev_dynamic_indices, dtype=torch.bool)
        for m in measurements_to_fill:
            if (type(m) is not tuple) or (m[1] != MeasIndexGroupOptions.NUMERICAL_ONLY): continue

            m = m[0]
            prev_measurements_to_drop_idx |= (prev_dynamic_measurement_indices == config.measurements_idxmap[m])

        kept_cols_mask = ~(prev_measurements_to_drop_idx.all(dim=0))

        data_tensors = []
        for dt in (
            prev_dynamic_indices, prev_dynamic_measurement_indices, prev_dynamic_values,
            prev_dynamic_values_mask
        ):
            data_tensors.append(torch.where(prev_measurements_to_drop_idx, 0, dt)[:, kept_cols_mask])

        (
            prev_dynamic_indices, prev_dynamic_measurement_indices, prev_dynamic_values,
            prev_dynamic_values_mask
        ) = data_tensors

        new_dynamic_indices = torch.cat((prev_dynamic_indices, new_dynamic_indices), 1)
        new_dynamic_measurement_indices = torch.cat(
            (prev_dynamic_measurement_indices, new_dynamic_measurement_indices), 1
        )
        new_dynamic_values = torch.cat((prev_dynamic_values, new_dynamic_values), 1)
        new_dynamic_values_mask = torch.cat((prev_dynamic_values_mask, new_dynamic_values_mask), 1)

        # Re-pad data elements.
        (
            (dynamic_indices, dynamic_measurement_indices, dynamic_values, dynamic_values_mask),
            (new_dynamic_indices, new_dynamic_measurement_indices, new_dynamic_values, new_dynamic_values_mask),
        ) = self.pad_data_elements(
            batch, new_dynamic_indices, new_dynamic_measurement_indices, new_dynamic_values, new_dynamic_values_mask
        )

        dynamic_indices[:, -1] = new_dynamic_indices
        dynamic_measurement_indices[:, -1] = new_dynamic_measurement_indices
        dynamic_values[:, -1] = new_dynamic_values
        dynamic_values_mask[:, -1] = new_dynamic_values_mask

        return EventStreamPytorchBatch(
            time=batch.time,
            event_mask=batch.event_mask,
            dynamic_indices=dynamic_indices,
            dynamic_measurement_indices=dynamic_measurement_indices,
            dynamic_values=dynamic_values,
            dynamic_values_mask=dynamic_values_mask,
        )

@dataclass
class GenerativeSequenceModelPredictions(ModelOutput, NestedIndexableMixin, SeedableMixin):
    """Predictions for the GenerativeSequenceModel head, split by task type."""
    classification: Optional[Dict[str, CATEGORICAL_DIST_T]] = None
    regression: Optional[Dict[str, torch.distributions.Distribution]] = None
    regression_indices: Optional[Dict[str, torch.LongTensor]] = None
    time_to_event: Optional[torch.distributions.Distribution] = None

    def mode(self, event_mask: torch.BoolTensor) -> GenerativeSequenceModelSamples:
        """Returns a mode (not guaranteed to be unique or maximal) of each of the contained distributions."""

        return GenerativeSequenceModelSamples(
            event_mask = event_mask[:, -1].detach(),
            classification = {k: v.mode for k, v in self.classification.items()},
            regression = {k: v.mode for k, v in self.regression.items()},
            regression_indices = self.regression_indices,
            time_to_event = self.time_to_event.mode,
        )

    @SeedableMixin.WithSeed
    def sample(self, event_mask: torch.BoolTensor) -> GenerativeSequenceModelSamples:
        """Returns a sample from the nested distributions."""

        return GenerativeSequenceModelSamples(
            event_mask = event_mask[:, -1].detach(),
            classification = {k: v.sample() for k, v in self.classification.items()},
            regression = {k: v.sample() for k, v in self.regression.items()},
            regression_indices = self.regression_indices,
            time_to_event = self.time_to_event.sample(),
        )

@dataclass
class GenerativeSequenceModelLabels(ModelOutput):
    """Labels for the GenerativeSequenceModel head, split by task type."""
    # Single-label classification task labels will have shape batch X seq and have raw integer labels in
    # it, whereas multi-label classification task labels will have shape batch X seq X vocab size and have
    # binary indicators for each label.
    classification: Optional[Dict[str, torch.LongTensor]] = None
    regression: Optional[Dict[str, torch.FloatTensor]] = None
    regression_indices: Optional[Dict[str, torch.LongTensor]] = None
    time_to_event: Optional[torch.FloatTensor] = None

@dataclass
class EventStreamTransformerForGenerativeSequenceModelOutput(ModelOutput):
    """ All GenerativeSequenceModel outputs, including losses, predictions, labels, and masks."""

    loss: torch.FloatTensor
    losses: Optional[GenerativeSequenceModelLosses] = None
    preds: Optional[GenerativeSequenceModelPredictions] = None
    labels: Optional[GenerativeSequenceModelLabels] = None
    event_type_mask_per_measurement: Optional[Dict[str, torch.BoolTensor]] = None
    event_mask: Optional[torch.BoolTensor] = None
    dynamic_values_mask: Optional[torch.BoolTensor] = None

@dataclass
class EventStreamTransformerForStreamClassificationModelOutput(ModelOutput):
    """ All GenerativeSequenceModel outputs, including losses, predictions, labels, and masks."""

    loss: torch.FloatTensor
    preds: torch.FloatTensor = None
    labels: Union[torch.LongTensor, torch.FloatTensor] = None

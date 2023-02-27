import torch
from dataclasses import dataclass, asdict
from mixins import SeedableMixin
from transformers.utils import ModelOutput
from typing import Any, Dict, Optional, Tuple, Union, Sequence, Set

from .utils import INDEX_SELECT_T, idx_distribution, expand_indexed_regression
from .config import StructuredEventStreamTransformerConfig
from ..EventStreamData.types import DataModality, EventStreamPytorchBatch

CATEGORICAL_DIST_T = Union[torch.distributions.Bernoulli, torch.distributions.Categorical]

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
        self, batch: EventStreamPytorchBatch, config: StructuredEventStreamTransformerConfig
    ) -> Tuple[torch.Tensor]:
        """
        This function is used for generation, and builds a new batch element from the prediction sample in
        this object.
        """

        # Add time
        time = batch.time[:, -1] + self.time_to_event

        # Add event_mask
        event_mask = self.event_mask

        # Add data elements (indices, values, types, values_mask)
        dynamic_measurement_indices = []
        dynamic_indices = []
        dynamic_values = []
        dynamic_values_mask = []

        for measurement in self.regression.keys(): assert measurement in self.classification

        for measurement, preds in self.classification.items():
            # Add the data index.
            vocab_offset = config.vocab_offsets_by_measurement[measurement]
            vocab_size = config.vocab_sizes_by_measurement[measurement]

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

            dynamic_indices.append(indices)

            # Add the data type.
            dynamic_measurement_indices.append(config.measurements_idxmap[measurement] * torch.ones_like(indices))

            # Add data values, if present
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

                values = regressed_values.gather(-1, indices - vocab_offset)
                values_mask = regressed_values_mask.gather(-1, indices - vocab_offset)

            dynamic_values.append(values)
            dynamic_values_mask.append(values_mask)

        dynamic_indices = torch.cat(dynamic_indices, 1)
        dynamic_measurement_indices = torch.cat(dynamic_measurement_indices, 1)
        dynamic_values = torch.cat(dynamic_values, 1)
        dynamic_values_mask = torch.cat(dynamic_values_mask, 1)

        return time, event_mask, dynamic_indices, dynamic_measurement_indices, dynamic_values, dynamic_values_mask

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
        self, batch: EventStreamPytorchBatch, config: StructuredEventStreamTransformerConfig
    ) -> EventStreamPytorchBatch:
        """
        This function builds a new batch element from self, then appends it to the end of the input batch.
        TODO(mmd): should this function only append the new event time, every time?
        """

        (
            new_event_time, new_event_mask, new_dynamic_indices, new_dynamic_measurement_indices, new_dynamic_values,
            new_dynamic_values_mask,
        ) = self.build_new_batch_element(batch, config)

        # Combine everything
        seq_dim = 1

        time = torch.cat((batch.time, new_event_time.unsqueeze(seq_dim)), seq_dim)
        event_mask = torch.cat((batch.event_mask, new_event_mask.unsqueeze(seq_dim)), seq_dim)

        # Re-pad data elements.
        (
            (dynamic_indices, dynamic_measurement_indices, dynamic_values, dynamic_values_mask),
            (new_dynamic_indices, new_dynamic_measurement_indices, new_dynamic_values, new_dynamic_values_mask),
        ) = self.pad_data_elements(
            batch, new_dynamic_indices, new_dynamic_measurement_indices, new_dynamic_values, new_dynamic_values_mask
        )

        dynamic_indices = torch.cat((dynamic_indices, new_dynamic_indices.unsqueeze(seq_dim)), seq_dim)
        dynamic_measurement_indices = torch.cat((dynamic_measurement_indices, new_dynamic_measurement_indices.unsqueeze(seq_dim)), seq_dim)
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
        measurements_to_fill: Set[str],
    ) -> EventStreamPytorchBatch:
        """This function updates the last batch element from self."""

        (
            _, _, new_dynamic_indices, new_dynamic_measurement_indices, new_dynamic_values, new_dynamic_values_mask
        ) = self.build_new_batch_element(batch, config)

        if 'time' in measurements_to_fill:
            raise ValueError(f"You shouldn't ever be trying to fill the 'time' aspect of a batch!")

        to_fill_indices = [
            (new_dynamic_measurement_indices == config.measurements_idxmap[measurement]) for measurement in measurements_to_fill
        ]
        to_fill_idx = torch.stack(to_fill_indices, dim=0).any(dim=0)

        present_mask = to_fill_idx.any(dim=0)

        data_tensors = []
        for dt in (new_dynamic_indices, new_dynamic_measurement_indices, new_dynamic_values, new_dynamic_values_mask):
            dt = torch.where(to_fill_idx, dt, torch.zeros_like(dt))
            dt = dt[:, present_mask]
            data_tensors.append(dt)

        (new_dynamic_indices, new_dynamic_measurement_indices, new_dynamic_values, new_dynamic_values_mask) = data_tensors

        prev_dynamic_indices = batch.dynamic_indices[:, -1]
        prev_dynamic_measurement_indices = batch.dynamic_measurement_indices[:, -1]
        prev_dynamic_values = batch.dynamic_values[:, -1]
        prev_dynamic_values_mask = batch.dynamic_values_mask[:, -1]

        will_be_replaced_indices = [
            (prev_dynamic_measurement_indices == config.measurements_idxmap[measurement]) for measurement in measurements_to_fill
        ]
        will_be_replaced_idx = torch.stack(will_be_replaced_indices, dim=0).any(dim=0)

        any_kept_idx = ~(will_be_replaced_idx.all(dim=0))
        data_tensors = []
        for dt in (prev_dynamic_indices, prev_dynamic_measurement_indices, prev_dynamic_values, prev_dynamic_values_mask):
            dt = torch.where(will_be_replaced_idx, torch.zeros_like(dt), dt)
            dt = dt[:, any_kept_idx]
            data_tensors.append(dt)

        (prev_dynamic_indices, prev_dynamic_measurement_indices, prev_dynamic_values, prev_dynamic_values_mask) = data_tensors

        new_dynamic_indices = torch.cat((prev_dynamic_indices, new_dynamic_indices), 1)
        new_dynamic_measurement_indices = torch.cat((prev_dynamic_measurement_indices, new_dynamic_measurement_indices), 1)
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

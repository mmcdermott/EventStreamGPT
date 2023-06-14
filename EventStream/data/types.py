"""A collection of objects and enumerations for better type support in data applications."""

import dataclasses
import enum
from typing import Any, Union

import torch

from ..utils import StrEnum


class InputDFType(StrEnum):
    """The kinds of input dataframes that can be used to construct a dataset."""

    STATIC = enum.auto()
    """A dataframe such that each row contains static (non-time-varying) data for each subject."""

    EVENT = enum.auto()
    """A dataframe containing event-level data about a subject.

    Each row will contain a timestamp, associated measurements, and subject ID. Timestamps may be duplicated
    in these input dataframes, but will be deduplicated in the resulting dataset.
    """

    RANGE = enum.auto()
    """A dataframe containing range-level data about a subject.

    Each row contains a start and end timestamp, associated measurements, and subject ID. RANGE dataframes are
    converted into start, end, and equal (start time = end time) event-level dataframes. Timestamps may be
    duplicated in these input dataframes, but will be deduplicated in the resulting dataset.
    """


class InputDataType(StrEnum):
    """The kinds of data that can be contained in an input dataframe column."""

    CATEGORICAL = enum.auto()
    """A categorical variable."""

    FLOAT = enum.auto()
    """A floating-point variable."""

    TIMESTAMP = enum.auto()
    """A timestamp variable.

    This may also be associated with a separate string for timestamp format, if the timestamp is originally
    presented as a string.
    """

    BOOLEAN = enum.auto()
    """A boolean variable."""


@dataclasses.dataclass
class PytorchBatch:
    """A dataclass representing a batch of event flow data for a Pytorch model.

    This class defines the data-output interface for deep learning models built off Event Flow GPT datasets.
    It stores the underlying data in the batch in a set of tensors, and also exposes some helpful methods and
    properties to simplify interacting with data.

    Attributes:
        event_mask: A boolean tensor of shape (batch_size, sequence_length) indicating which events in the
            batch are valid (i.e., which are not padding).
        time_delta: A float tensor of shape (batch_size, sequence_length) indicating the time delta in minutes
            between each event and the subsequent event in that subject's sequence in the batch.
        time: A float tensor of shape (batch_size, sequence_length) indicating the time in minutes since the
            start of the subject's sequence of each event in the batch. This is often left unset, as it is
            generally redundant with `time_delta`. However, it is used in generation, when the batch is
            truncated to use efficient caching so the raw time point can't be recovered from the time delta.
        static_indices: A long tensor of shape (batch_size, n_static_data_elements) indicating the indices of
            the static data elements observed for each subject in the batch. These are *unordered*; meaning
            that the second dimension position of a given element in this tensor is not necessarily
            meaningful. This is because the static data elements are sparsely encoded, so the indices are
            sufficient to recover the original data even in an unordered form.
        static_measurement_indices: A long tensor of shape (batch_size, n_static_data_elements) indicating
            which measurements the indices in `static_indices` correspond to. E.g., if there is a static data
            element corresponding to race, then the value in `static_measurement_indices` at the associated
            position would be an integer index corresponding to the race measurement overall, whereas the
            index at the identical position in `static_indices` would be an integer index corresponding to the
            specific race observed for the subject (e.g., "White", "Black", etc.).
        dynamic_indices: A long tensor of shape (batch_size, sequence_length, n_data_elements) indicating the
            indices of the dynamic data elements observed for each subject in the batch. These are
            *unordered* in the last dimension, meaning that the third dimension position of a given element in
            this tensor is not necessarily meaningful. This is because the dynamic data elements are sparsely
            encoded, so the indices and values are sufficient to recover the original data even in an
            unordered form.
        dynamic_measurement_indices: A long tensor of shape (batch_size, sequence_length, n_data_elements)
            indicating which measurements the indices in `dynamic_indices` correspond to, similar to the
            `static_measurement_indices` attribute.
        dynamic_values: A float tensor of shape (batch_size, sequence_length, n_data_elements) indicating the
            numeric values associated with each dynamic data element in the `dynamic_indices` tensor. If no
            value was recorded for a given dynamic data element, the value in this tensor will be zero.
        dynamic_values_mask: A boolean tensor of shape (batch_size, sequence_length, n_data_elements)
            indicating which values in the `dynamic_values` tensor were actually observed.
        start_time: A float tensor of shape (batch_size,) indicating the start time in minutes since the epoch
            of each subject's sequence in the batch. This is often unset, as it is only used in generation
            when we may need to know the actual time of day of any generated event.
        stream_labels: A dictionary mapping task names to label LongTensors of shape (batch_size,) providing
            labels for the associated tasks for the sequences in the batch. Is only used during fine-tuning or
            zero-shot evaluation runs.
    """

    event_mask: torch.BoolTensor | None = None

    # We track this instead of raw times as it is less likely to suffer from underflow errors.
    time_delta: torch.FloatTensor | None = None

    # We don't often use this, but it is used in generation.
    time: torch.FloatTensor | None = None

    static_indices: torch.LongTensor | None = None
    static_measurement_indices: torch.LongTensor | None = None

    dynamic_indices: torch.LongTensor | None = None
    dynamic_measurement_indices: torch.LongTensor | None = None
    dynamic_values: torch.FloatTensor | None = None
    dynamic_values_mask: torch.BoolTensor | None = None

    start_time: torch.FloatTensor | None = None

    stream_labels: dict[str, torch.FloatTensor | torch.LongTensor] | None = None

    @property
    def device(self) -> torch.device:
        """Returns the device storing the tensors in this batch.

        Assumes all elements of the batch are on the same device.
        """
        return self.event_mask.device

    @property
    def batch_size(self) -> int:
        """Returns the batch size of this batch.

        Assumes the batch has not been sliced from its initial configuration.
        """
        return self.event_mask.shape[0]

    @property
    def sequence_length(self) -> int:
        """Returns the maximum sequence length of the sequences in this batch.

        Assumes the batch has not been sliced from its initial configuration.
        """
        return self.event_mask.shape[1]

    @property
    def n_data_elements(self) -> int:
        """Returns the maximum number of dynamic data elements of the events in this batch.

        Assumes the batch has not been sliced from its initial configuration.
        """
        return self.dynamic_indices.shape[2]

    @property
    def n_static_data_elements(self) -> int:
        """Returns the maximum number of static data elements of the subjects in this batch.

        Assumes the batch has not been sliced from its initial configuration.
        """
        return self.static_indices.shape[1]

    def get(self, item: str, default: Any) -> Any:
        """A dictionary like get method for this batch, by attribute name."""
        return getattr(self, item) if item in self.keys() else default

    def _slice(self, index: tuple[int | slice] | int | slice) -> "PytorchBatch":
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) == 0 or len(index) > 3:
            raise ValueError(f"Invalid index {index} for PytorchBatch! Must be of length 1, 2, or 3.")
        if any(not isinstance(i, (int, slice)) for i in index):
            raise ValueError(f"Invalid index {index} for PytorchBatch! Can only consist of ints and slices.")

        batch_index = index[0]
        seq_index = slice(None)
        meas_index = slice(None)

        if len(index) > 1:
            seq_index = index[1]
        if len(index) > 2:
            meas_index = index[2]

        return PytorchBatch(
            event_mask=self.event_mask[batch_index, seq_index],
            time_delta=self.time_delta[batch_index, seq_index],
            static_indices=None if self.static_indices is None else self.static_indices[batch_index],
            static_measurement_indices=(
                None
                if self.static_measurement_indices is None
                else self.static_measurement_indices[batch_index]
            ),
            dynamic_indices=self.dynamic_indices[batch_index, seq_index, meas_index],
            dynamic_measurement_indices=self.dynamic_measurement_indices[batch_index, seq_index, meas_index],
            dynamic_values=self.dynamic_values[batch_index, seq_index, meas_index],
            dynamic_values_mask=self.dynamic_values_mask[batch_index, seq_index, meas_index],
            start_time=None if self.start_time is None else self.start_time[batch_index],
            stream_labels=(
                None
                if self.stream_labels is None
                else {k: v[batch_index] for k, v in self.stream_labels.items()}
            ),
            time=None if self.time is None else self.time[batch_index, seq_index],
        )

    def __getitem__(self, item: str | tuple[int | slice]) -> Union[torch.Tensor, "PytorchBatch"]:
        match item:
            case str():
                return dataclasses.asdict(self)[item]
            case tuple() | int() | slice():
                return self._slice(item)
            case _:
                raise TypeError(f"Invalid type {type(item)} for {item} for indexing!")

    def __setitem__(self, item: str, val: torch.Tensor):
        if not hasattr(self, item):
            raise KeyError(f"Key {item} not found")
        setattr(self, item, val)

    def items(self):
        """A dictionary like items` method for the elements of this batch, by attribute."""
        return dataclasses.asdict(self).items()

    def keys(self):
        """A dictionary like keys method for the elements of this batch, by attribute."""
        return dataclasses.asdict(self).keys()

    def values(self):
        """A dictionary like values method for the elements of this batch, by attribute."""
        return dataclasses.asdict(self).values()

    def last_sequence_element_unsqueezed(self) -> "PytorchBatch":
        """Filters the batch down to just the last event, while retaining the same # of dims."""
        return self[:, -1:]


class TemporalityType(StrEnum):
    """The ways a measurement can vary in time."""

    STATIC = enum.auto()
    """This measure is static per-subject.

    Currently only supported with classificaton data modalities.
    """

    DYNAMIC = enum.auto()
    """This measure is dynamic with respect to time in a general manner.

    It will be recorded potentially many times per-event, and can take on either categorical or partially
    observed regression data modalities.
    """

    FUNCTIONAL_TIME_DEPENDENT = enum.auto()
    """This measure varies predictably with respect to time and the static measures of a subject.

    The "observations" of this measure will be computed on the basis of that functional form and added to the
    observed events. Currently only supported with categorical or fully observed regression variables.
    """


class DataModality(StrEnum):
    """The modality of a data element.

    Measurement modality dictates pre-processing, embedding, and possible generation of said element.
    """

    DROPPED = enum.auto()
    """This column was dropped due to occurring too infrequently for use."""

    SINGLE_LABEL_CLASSIFICATION = enum.auto()
    """This data modality must take on a single label in all possible instances where it is observed.

    This will never have an associated data value measured. Element will be generated via consecutive
    prediction of whether or not the event will be observed at all, followed by single- label, multi-class
    classification of what label will be observed.
    """

    MULTI_LABEL_CLASSIFICATION = enum.auto()
    """This data modality can occur zero or more times with different labels.

    This will never have an associated data value measured (see MULTIVARIATE_REGRESSION). Element will be
    generated via multi-label, binary classification.
    """

    MULTIVARIATE_REGRESSION = enum.auto()
    """A column which can occur 0+ times per event with different labels and values.

    All multivariate regression measures are assumed to be partially observed at present. Element keys will be
    generated via multi-label, binary classification. Values will be generated via probabilistic regression.
    """

    UNIVARIATE_REGRESSION = enum.auto()
    """This column is a continuous-valued, one-dimensional numerical measure which is partially observed.

    The model first predicts whether or not this measurement will be observed, then what value it would take
    on.
    """


class NumericDataModalitySubtype(StrEnum):
    """Numeric value types.

    These are used to characterize both entire measures (e.g., 'age' takes on integer values) or sub-measures
    (e.g., within the measure of "vitals signs", observations for the key "heart rate" take on float values).
    """

    DROPPED = enum.auto()
    """The values of this measure (or sub-measure) were dropped."""

    INTEGER = enum.auto()
    """This measure (or sub-measure) takes on integer values."""

    FLOAT = enum.auto()
    """This measure (or sub-measure) takes on floating point values."""

    CATEGORICAL_INTEGER = enum.auto()
    """This formerly integer measure/sub-measure has been converted to take on categorical values.

    Options can be found in the global vocabulary, with the syntax ``f"{key_col}__EQ_{orig_val}"``.
    """

    CATEGORICAL_FLOAT = enum.auto()
    """This formerly floating point measure/sub-measure has been converted to take on categorical values.

    Options can be found in the global vocabulary, with the syntax ``f"{key_col}__EQ_{orig_val}"``.
    """

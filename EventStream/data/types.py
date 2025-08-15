"""A collection of objects and enumerations for better type support in data applications."""

import dataclasses
import enum
from collections import defaultdict
from typing import Any, Union

import polars as pl
import torch

from ..utils import StrEnum


def de_pad(L: list[int], *other_L) -> list[int] | tuple[list[int]]:
    """Filters down all passed lists to only the indices where the first arg is non-zero.

    Args:
        L: The list whose entries denote padding (0) or non-padding (non-zero).
        *other_L: Any other lists that should be de-padded in the same way as L.

    Examples:
        >>> de_pad([1, 3, 0, 4, 0, 0], [10, 0, 5, 8, 1, 0])
        ([1, 3, 4], [10, 0, 8])
        >>> de_pad([1, 3, 0, 4, 0, 0])
        [1, 3, 4]
    """

    out_L = []
    out_other = [None if x is None else [] for x in other_L]

    for i, v in enumerate(L):
        if v != 0:
            out_L.append(v)
            for j, LL in enumerate(other_L):
                if LL is not None:
                    out_other[j].append(LL[i])

    if other_L:
        return tuple([out_L] + out_other)
    else:
        return out_L


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
            sufficient to recover the original data even in an unordered form. Here, by "indices" we mean that
            these are integer values indicating the index of the associated categorical vocabulary element
            corresponding to this observation; e.g., if the static measurement records that the subject's eye
            color is brown, then if the categorical measurement of ``eye_color/BROWN``` in the unified
            vocabulary is at position 32, then the index for that observation would be 32.
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
        start_idx: A long tensor of shape (batch_size,) indicating the start index of the sampled sub-sequence
            for each subject in the batch relative to their raw data.
        end_idx: A long tensor of shape (batch_size,) indicating the end index of the sampled sub-sequence
            for each subject in the batch relative to their raw data.
        subject_id: A long tensor of shape (batch_size,) indicating the subject ID of each member of the
            batch.
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
    start_idx: torch.LongTensor | None = None
    end_idx: torch.LongTensor | None = None
    subject_id: torch.LongTensor | None = None

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
            start_idx=None if self.start_idx is None else self.start_idx[batch_index],
            end_idx=None if self.end_idx is None else self.end_idx[batch_index],
            subject_id=None if self.subject_id is None else self.subject_id[batch_index],
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

    def __eq__(self, other: "PytorchBatch") -> bool:
        """Checks for equality between self and other."""
        if self.keys() != other.keys():
            return False

        for k in self.keys():
            self_v = self[k]
            other_v = other[k]

            if type(self_v) is not type(other_v):
                return False

            match self_v:
                case dict() if k == "stream_labels":
                    if self_v.keys() != other_v.keys():
                        return False
                    for kk in self_v.keys():
                        self_vv = self_v[kk]
                        other_vv = other_v[kk]

                        if self_vv.shape != other_vv.shape:
                            return False
                        if (self_vv != other_vv).any():
                            return False

                case torch.Tensor():
                    if self_v.shape != other_v.shape:
                        return False
                    if (self_v != other_v).any():
                        return False
                case None if k in ("time", "stream_labels", "start_idx", "end_idx", "subject_id"):
                    if other_v is not None:
                        return False
                case _:
                    raise ValueError(f"{k}: {type(self_v)} not supported in batch!")
        return True

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

    def repeat_batch_elements(self, expand_size: int) -> "PytorchBatch":
        """Repeats each batch element expand_size times in order. Used for generation.

        Args:
            expand_size: The number of times each batch elements data should be repeated.

        Returns: A new PytorchBatch object with each batch element's data repeated expand_size times.

        Examples:
            >>> import torch
            >>> batch = PytorchBatch(
            ...     event_mask=torch.tensor([[True, True, True], [True, True, False]]),
            ...     time_delta=torch.tensor([[1.0, 2.0, 3.0], [1.0, 5.0, 0.0]]),
            ...     static_indices=torch.tensor([[0, 1], [1, 2]]),
            ...     static_measurement_indices=torch.tensor([[0, 1], [1, 1]]),
            ...     dynamic_indices=torch.tensor([[[0, 1], [1, 2], [2, 3]], [[0, 1], [1, 5], [0, 0]]]),
            ...     dynamic_measurement_indices=torch.tensor(
            ...         [[[0, 1], [1, 2], [2, 3]], [[0, 1], [1, 2], [0, 0]]]
            ...     ),
            ...     dynamic_values=torch.tensor(
            ...         [[[0.0, 1.0], [1.0, 2.0], [0, 0]], [[0.0, 1.0], [1.0, 0.0], [0, 0]]]
            ...     ),
            ...     dynamic_values_mask=torch.tensor([
            ...         [[False, True], [True, True], [False, False]],
            ...         [[False, True], [True, False], [False, False]]
            ...     ]),
            ...     start_time=torch.tensor([0.0, 10.0]),
            ...     stream_labels={"a": torch.tensor([0, 1]), "b": torch.tensor([1, 2])},
            ...     time=None,
            ... )
            >>> repeated_batch = batch.repeat_batch_elements(2)
            >>> for k, v in repeated_batch.items():
            ...     print(k)
            ...     print(v)
            event_mask
            tensor([[ True,  True,  True],
                    [ True,  True,  True],
                    [ True,  True, False],
                    [ True,  True, False]])
            time_delta
            tensor([[1., 2., 3.],
                    [1., 2., 3.],
                    [1., 5., 0.],
                    [1., 5., 0.]])
            time
            None
            static_indices
            tensor([[0, 1],
                    [0, 1],
                    [1, 2],
                    [1, 2]])
            static_measurement_indices
            tensor([[0, 1],
                    [0, 1],
                    [1, 1],
                    [1, 1]])
            dynamic_indices
            tensor([[[0, 1],
                     [1, 2],
                     [2, 3]],
            <BLANKLINE>
                    [[0, 1],
                     [1, 2],
                     [2, 3]],
            <BLANKLINE>
                    [[0, 1],
                     [1, 5],
                     [0, 0]],
            <BLANKLINE>
                    [[0, 1],
                     [1, 5],
                     [0, 0]]])
            dynamic_measurement_indices
            tensor([[[0, 1],
                     [1, 2],
                     [2, 3]],
            <BLANKLINE>
                    [[0, 1],
                     [1, 2],
                     [2, 3]],
            <BLANKLINE>
                    [[0, 1],
                     [1, 2],
                     [0, 0]],
            <BLANKLINE>
                    [[0, 1],
                     [1, 2],
                     [0, 0]]])
            dynamic_values
            tensor([[[0., 1.],
                     [1., 2.],
                     [0., 0.]],
            <BLANKLINE>
                    [[0., 1.],
                     [1., 2.],
                     [0., 0.]],
            <BLANKLINE>
                    [[0., 1.],
                     [1., 0.],
                     [0., 0.]],
            <BLANKLINE>
                    [[0., 1.],
                     [1., 0.],
                     [0., 0.]]])
            dynamic_values_mask
            tensor([[[False,  True],
                     [ True,  True],
                     [False, False]],
            <BLANKLINE>
                    [[False,  True],
                     [ True,  True],
                     [False, False]],
            <BLANKLINE>
                    [[False,  True],
                     [ True, False],
                     [False, False]],
            <BLANKLINE>
                    [[False,  True],
                     [ True, False],
                     [False, False]]])
            start_time
            tensor([ 0.,  0., 10., 10.])
            start_idx
            None
            end_idx
            None
            subject_id
            None
            stream_labels
            {'a': tensor([0, 0, 1, 1]), 'b': tensor([1, 1, 2, 2])}
        """

        expanded_return_idx = (
            torch.arange(self.batch_size).view(-1, 1).repeat(1, expand_size).view(-1).to(self.device)
        )

        out_batch = {}

        for k, v in self.items():
            match v:
                case dict():
                    out_batch[k] = {kk: vv.index_select(0, expanded_return_idx) for kk, vv in v.items()}
                case torch.Tensor():
                    out_batch[k] = v.index_select(0, expanded_return_idx)
                case None if k in ("time", "stream_labels", "start_idx", "end_idx", "subject_id"):
                    out_batch[k] = None
                case _:
                    raise TypeError(f"{k}: {type(v)} not supported in batch for generation!")

        return PytorchBatch(**out_batch)

    def split_repeated_batch(self, n_splits: int) -> list["PytorchBatch"]:
        """Split a batch into a list of batches by chunking batch elements into groups.

        This is the inverse of `PytorchBatch.repeat_batch_elements`. It is used for taking a generated batch
        that has been expanded and splitting it into separate list elements with independent generations for
        each batch element in the original batch.

        Args:
            n_splits: The number of splits to make.

        Returns: A list of length `n_splits` of PytorchBatch objects, such that the list element i contains
            batch elements [i, i+self.batch_size/n_splits).

        Raises:
            ValueError: if `n_splits` is not a positive integer divisor of `self.batch_size`.

        Examples:
            >>> import torch
            >>> batch = PytorchBatch(
            ...     event_mask=torch.tensor([
            ...         [True, True, True],
            ...         [True, True, False],
            ...         [True, False, False],
            ...         [False, False, False]
            ...     ]),
            ...     time_delta=torch.tensor([
            ...         [1.0, 2.0, 3.0],
            ...         [1.0, 5.0, 0.0],
            ...         [2.3, 0.0, 0.0],
            ...         [0.0, 0.0, 0.0],
            ...     ]),
            ...     static_indices=torch.tensor([[0, 1], [1, 2], [1, 3], [0, 5]]),
            ...     static_measurement_indices=torch.tensor([[0, 1], [1, 1], [1, 1], [0, 2]]),
            ...     dynamic_indices=torch.tensor([
            ...         [[0, 1], [1, 2], [2, 3]],
            ...         [[0, 1], [1, 5], [0, 0]],
            ...         [[0, 2], [0, 0], [0, 0]],
            ...         [[0, 0], [0, 0], [0, 0]],
            ...     ]),
            ...     dynamic_measurement_indices=torch.tensor([
            ...         [[0, 1], [1, 2], [2, 3]],
            ...         [[0, 1], [1, 2], [0, 0]],
            ...         [[0, 2], [0, 0], [0, 0]],
            ...         [[0, 0], [0, 0], [0, 0]],
            ...     ]),
            ...     dynamic_values=torch.tensor([
            ...         [[0.0, 1.0], [1.0, 2.0], [0.0, 0.0]],
            ...         [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            ...         [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            ...         [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ...     ]),
            ...     dynamic_values_mask=torch.tensor([
            ...         [[False, True], [True, True], [False, False]],
            ...         [[False, True], [True, False], [False, False]],
            ...         [[False, True], [False, False], [False, False]],
            ...         [[False, False], [False, False], [False, False]],
            ...     ]),
            ...     start_time=torch.tensor([0.0, 10.0, 3.0, 2.2]),
            ...     stream_labels={"a": torch.tensor([0, 1, 0, 1]), "b": torch.tensor([1, 2, 4, 3])},
            ...     time=None,
            ... )
            >>> batch.split_repeated_batch(3)
            Traceback (most recent call last):
                ...
            ValueError: n_splits (3) must be a positive integer divisor of batch_size (4)
            >>> for i, T in enumerate(batch.split_repeated_batch(2)):
            ...     print(f"Returned batch {i}:")
            ...     for k, v in T.items():
            ...         print(k)
            ...         print(v)
            Returned batch 0:
            event_mask
            tensor([[ True,  True,  True],
                    [ True, False, False]])
            time_delta
            tensor([[1.0000, 2.0000, 3.0000],
                    [2.3000, 0.0000, 0.0000]])
            time
            None
            static_indices
            tensor([[0, 1],
                    [1, 3]])
            static_measurement_indices
            tensor([[0, 1],
                    [1, 1]])
            dynamic_indices
            tensor([[[0, 1],
                     [1, 2],
                     [2, 3]],
            <BLANKLINE>
                    [[0, 2],
                     [0, 0],
                     [0, 0]]])
            dynamic_measurement_indices
            tensor([[[0, 1],
                     [1, 2],
                     [2, 3]],
            <BLANKLINE>
                    [[0, 2],
                     [0, 0],
                     [0, 0]]])
            dynamic_values
            tensor([[[0., 1.],
                     [1., 2.],
                     [0., 0.]],
            <BLANKLINE>
                    [[0., 1.],
                     [0., 0.],
                     [0., 0.]]])
            dynamic_values_mask
            tensor([[[False,  True],
                     [ True,  True],
                     [False, False]],
            <BLANKLINE>
                    [[False,  True],
                     [False, False],
                     [False, False]]])
            start_time
            tensor([0., 3.])
            start_idx
            None
            end_idx
            None
            subject_id
            None
            stream_labels
            {'a': tensor([0, 0]), 'b': tensor([1, 4])}
            Returned batch 1:
            event_mask
            tensor([[ True,  True, False],
                    [False, False, False]])
            time_delta
            tensor([[1., 5., 0.],
                    [0., 0., 0.]])
            time
            None
            static_indices
            tensor([[1, 2],
                    [0, 5]])
            static_measurement_indices
            tensor([[1, 1],
                    [0, 2]])
            dynamic_indices
            tensor([[[0, 1],
                     [1, 5],
                     [0, 0]],
            <BLANKLINE>
                    [[0, 0],
                     [0, 0],
                     [0, 0]]])
            dynamic_measurement_indices
            tensor([[[0, 1],
                     [1, 2],
                     [0, 0]],
            <BLANKLINE>
                    [[0, 0],
                     [0, 0],
                     [0, 0]]])
            dynamic_values
            tensor([[[0., 1.],
                     [1., 0.],
                     [0., 0.]],
            <BLANKLINE>
                    [[0., 0.],
                     [0., 0.],
                     [0., 0.]]])
            dynamic_values_mask
            tensor([[[False,  True],
                     [ True, False],
                     [False, False]],
            <BLANKLINE>
                    [[False, False],
                     [False, False],
                     [False, False]]])
            start_time
            tensor([10.0000,  2.2000])
            start_idx
            None
            end_idx
            None
            subject_id
            None
            stream_labels
            {'a': tensor([1, 1]), 'b': tensor([2, 3])}
            >>> repeat_batch = batch.repeat_batch_elements(5)
            >>> split_batches = repeat_batch.split_repeated_batch(5)
            >>> for i, v in enumerate(split_batches):
            ...     assert v == batch, f"Batch {i} ({v}) not equal to original batch {batch}!"
        """

        if not isinstance(n_splits, int) or n_splits <= 0 or self.batch_size % n_splits != 0:
            raise ValueError(
                f"n_splits ({n_splits}) must be a positive integer divisor of batch_size ({self.batch_size})"
            )

        self.batch_size // n_splits
        out_batches = [defaultdict(dict) for _ in range(n_splits)]
        for k, v in self.items():
            match v:
                case dict():
                    for kk, vv in v.items():
                        reshaped = vv.reshape(vv.shape[0] // n_splits, n_splits, *vv.shape[1:])
                        for i in range(n_splits):
                            out_batches[i][k][kk] = reshaped[:, i, ...]
                case torch.Tensor():
                    reshaped = v.reshape(v.shape[0] // n_splits, n_splits, *v.shape[1:])
                    for i in range(n_splits):
                        out_batches[i][k] = reshaped[:, i, ...]
                case None if k in ("time", "stream_labels", "start_idx", "end_idx", "subject_id"):
                    pass
                case _:
                    raise TypeError(f"{k}: {type(v)} not supported in batch for generation!")

        return [PytorchBatch(**B) for B in out_batches]

    def convert_to_DL_DF(self) -> pl.DataFrame:
        """Converts the batch data into a sparse DataFrame representation.

        Examples:
            >>> import torch
            >>> batch = PytorchBatch(
            ...     event_mask=torch.tensor([
            ...         [True, True, True],
            ...         [True, True, False],
            ...         [True, False, False],
            ...         [False, False, False]
            ...     ]),
            ...     time_delta=torch.tensor([
            ...         [1.0, 2.0, 3.0],
            ...         [1.0, 5.0, 0.0],
            ...         [2.3, 0.0, 0.0],
            ...         [0.0, 0.0, 0.0],
            ...     ]),
            ...     static_indices=torch.tensor([[0, 1], [1, 2], [1, 3], [0, 5]]),
            ...     static_measurement_indices=torch.tensor([[0, 1], [1, 1], [1, 1], [0, 2]]),
            ...     dynamic_indices=torch.tensor([
            ...         [[0, 1], [1, 2], [2, 3]],
            ...         [[0, 1], [1, 5], [0, 0]],
            ...         [[0, 2], [0, 0], [0, 0]],
            ...         [[0, 0], [0, 0], [0, 0]],
            ...     ]),
            ...     dynamic_measurement_indices=torch.tensor([
            ...         [[0, 1], [1, 2], [2, 3]],
            ...         [[0, 1], [1, 2], [0, 0]],
            ...         [[0, 2], [0, 0], [0, 0]],
            ...         [[0, 0], [0, 0], [0, 0]],
            ...     ]),
            ...     dynamic_values=torch.tensor([
            ...         [[0.0, 1.0], [1.0, 2.0], [0.0, 0.0]],
            ...         [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            ...         [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            ...         [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ...     ]),
            ...     dynamic_values_mask=torch.tensor([
            ...         [[False, True], [True, True], [False, False]],
            ...         [[False, True], [True, False], [False, False]],
            ...         [[False, True], [False, False], [False, False]],
            ...         [[False, False], [False, False], [False, False]],
            ...     ]),
            ...     start_time=torch.tensor([0.0, 10.0, 3.0, 2.2]),
            ...     stream_labels={"a": torch.tensor([0, 1, 0, 1]), "b": torch.tensor([1, 2, 4, 3])},
            ...     time=None,
            ... )
            >>> pl.Config.set_tbl_width_chars(80)
            <class 'polars.config.Config'>
            >>> batch.convert_to_DL_DF()
            shape: (4, 7)
            ┌───────────┬───────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
            │ time_delt ┆ static_in ┆ static_m ┆ dynamic_ ┆ dynamic_ ┆ dynamic_ ┆ start_ti │
            │ a         ┆ dices     ┆ easureme ┆ indices  ┆ measurem ┆ values   ┆ me       │
            │ ---       ┆ ---       ┆ nt_indic ┆ ---      ┆ ent_indi ┆ ---      ┆ ---      │
            │ list[f64] ┆ list[f64] ┆ es       ┆ list[lis ┆ ces      ┆ list[lis ┆ f64      │
            │           ┆           ┆ ---      ┆ t[f64]]  ┆ ---      ┆ t[f64]]  ┆          │
            │           ┆           ┆ list[f64 ┆          ┆ list[lis ┆          ┆          │
            │           ┆           ┆ ]        ┆          ┆ t[f64]]  ┆          ┆          │
            ╞═══════════╪═══════════╪══════════╪══════════╪══════════╪══════════╪══════════╡
            │ [1.0,     ┆ [1.0]     ┆ [1.0]    ┆ [[1.0],  ┆ [[1.0],  ┆ [[1.0],  ┆ 0.0      │
            │ 2.0, 3.0] ┆           ┆          ┆ [1.0,    ┆ [1.0,    ┆ [1.0,    ┆          │
            │           ┆           ┆          ┆ 2.0],    ┆ 2.0],    ┆ 2.0],    ┆          │
            │           ┆           ┆          ┆ [2.0,    ┆ [2.0,    ┆ [null,   ┆          │
            │           ┆           ┆          ┆ 3.0]…    ┆ 3.0]…    ┆ nul…     ┆          │
            │ [1.0,     ┆ [1.0,     ┆ [1.0,    ┆ [[1.0],  ┆ [[1.0],  ┆ [[1.0],  ┆ 10.0     │
            │ 5.0]      ┆ 2.0]      ┆ 1.0]     ┆ [1.0,    ┆ [1.0,    ┆ [1.0,    ┆          │
            │           ┆           ┆          ┆ 5.0]]    ┆ 2.0]]    ┆ null]]   ┆          │
            │ [2.3]     ┆ [1.0,     ┆ [1.0,    ┆ [[2.0]]  ┆ [[2.0]]  ┆ [[1.0]]  ┆ 3.0      │
            │           ┆ 3.0]      ┆ 1.0]     ┆          ┆          ┆          ┆          │
            │ []        ┆ [5.0]     ┆ [2.0]    ┆ []       ┆ []       ┆ []       ┆ 2.2      │
            └───────────┴───────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
        """

        df = {
            k: []
            for k, v in self.items()
            if k not in ("stream_labels", "event_mask", "dynamic_values_mask") and v is not None
        }

        for k in ("start_time", "subject_id", "start_idx", "end_idx"):
            if self[k] is not None:
                df[k] = list(self[k])

        for i in range(self.batch_size):
            idx, measurement_idx = de_pad(self.static_indices[i], self.static_measurement_indices[i])
            df["static_indices"].append(idx)
            df["static_measurement_indices"].append(measurement_idx)

            _, time_delta, time, idx, measurement_idx, vals, vals_mask = de_pad(
                self.event_mask[i],
                None if self.time_delta is None else self.time_delta[i],
                None if self.time is None else self.time[i],
                self.dynamic_indices[i],
                self.dynamic_measurement_indices[i],
                self.dynamic_values[i],
                self.dynamic_values_mask[i],
            )

            if time_delta is not None:
                df["time_delta"].append(time_delta)
            if time is not None:
                df["time"].append(time)

            names = ("dynamic_indices", "dynamic_measurement_indices", "dynamic_values")
            for n in names:
                df[n].append([])

            for j in range(len(idx)):
                de_padded_vals = de_pad(idx[j], measurement_idx[j], vals[j], vals_mask[j])
                # Now we add the indices and measurement indices
                for n, v in zip(names[:-1], de_padded_vals[:-2]):
                    df[n][i].append(v)

                df["dynamic_values"][i].append([None if not m else v for v, m in zip(*de_padded_vals[-2:])])

        return pl.DataFrame(df)


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

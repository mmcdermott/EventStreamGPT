"""Various configuration classes for EventStream data objects."""

from __future__ import annotations

import dataclasses
import enum
import random
from collections import OrderedDict, defaultdict
from collections.abc import Hashable, Sequence
from io import StringIO, TextIOBase
from pathlib import Path
from textwrap import shorten, wrap
from typing import Any, Union

import omegaconf
import pandas as pd

from ..utils import (
    COUNT_OR_PROPORTION,
    PROPORTION,
    JSONableMixin,
    StrEnum,
    hydra_dataclass,
    num_initial_spaces,
)
from .time_dependent_functor import AgeFunctor, TimeDependentFunctor, TimeOfDayFunctor
from .types import DataModality, InputDataType, InputDFType, TemporalityType
from .vocabulary import Vocabulary

# Represents the type for a column name in a dataframe.
DF_COL = Union[str, Sequence[str]]

# Represents the type of an input column during pre-processing.
INPUT_COL_T = Union[InputDataType, tuple[InputDataType, str]]

# A unified type for a schema of an input dataframe.
DF_SCHEMA = Union[
    # For cases where you specify a list of columns of a constant type.
    tuple[list[DF_COL], INPUT_COL_T],
    # For specifying a single column and type.
    tuple[DF_COL, INPUT_COL_T],
    # For specifying a dictionary of columns to types.
    dict[DF_COL, INPUT_COL_T],
    # For specifying a dictionary of column in names to column out names and types.
    dict[DF_COL, tuple[str, INPUT_COL_T]],
    # For specifying a dictionary of column in names to out names, all of a constant type.
    tuple[dict[DF_COL, str], INPUT_COL_T],
]


@dataclasses.dataclass
class DatasetSchema(JSONableMixin):
    """Represents the schema of an input dataset, including static and dynamic data sources.

    Contains the information necessary for extracting and pulling input dataset elements during a
    pre-processing pipeline. Inputs can be represented in either structured (typed) or plain (dictionary)
    form. There can only be one static schema currently, but arbitrarily many dynamic measurement schemas.
    During pre-processing the model will read all these dynamic input datasets and combine their outputs into
    the appropriate format. This can be written to or read from JSON files via the `JSONableMixin` base class
    methods.

    Attributes:
        static: The schema for the input dataset containing static (per-subject) information, in either object
            or dict form.
        dynamic: A list of schemas for all dynamic dataset schemas, each in either object or dict form.

    Raises:
        ValueError: If the static schema is `None`, if there is not a subject ID column specified in the
            static schema, if the passed "static" schema is not typed as a static schema, or if any dynamic
            schema is typed as a static schema.

    Examples:
        >>> DatasetSchema(dynamic=[])
        Traceback (most recent call last):
            ...
        ValueError: Must specify a static schema!
        >>> DatasetSchema(
        ...     static=dict(type="event", event_type="foo", input_df="/path/to/df.csv", ts_col="col"),
        ...     dynamic=[]
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Must pass a static schema config for static.
        >>> DatasetSchema(
        ...     static=dict(type="static", input_df="/path/to/df.csv", subject_id_col="col"),
        ...     dynamic=[dict(type="static", input_df="/path/to/df.csv", subject_id_col="col")]
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Must pass dynamic schemas in self.dynamic!
        >>> DS = DatasetSchema(
        ...     static=dict(type="static", input_df="/path/to/df.csv", subject_id_col="col"),
        ...     dynamic=[
        ...         dict(type="event", event_type="foo", input_df="/path/to/foo.csv", ts_col="col"),
        ...         dict(type="event", event_type="bar", input_df="/path/to/bar.csv", ts_col="col"),
        ...         dict(type="event", event_type="bar2", input_df="/path/to/bar.csv", ts_col="col2"),
        ...     ],
        ... )
        >>> DS.dynamic_by_df
        {'/path/to/foo.csv': [InputDFSchema(input_df='/path/to/foo.csv', type='event', event_type='foo',\
 subject_id_col='col', ts_col='col')], '/path/to/bar.csv': [InputDFSchema(input_df='/path/to/bar.csv',\
 type='event', event_type='bar', subject_id_col='col', ts_col='col'),\
 InputDFSchema(input_df='/path/to/bar.csv', type='event', event_type='bar2', subject_id_col='col',\
 ts_col='col2')]}
    """

    static: dict[str, Any] | InputDFSchema | None = None
    dynamic: list[InputDFSchema | dict[str, Any]] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.static is None:
            raise ValueError("Must specify a static schema!")

        if type(self.static) is dict:
            self.static = InputDFSchema(**self.static)
            if not self.static.is_static:
                raise ValueError("Must pass a static schema config for static.")

        if self.dynamic is not None:
            new_dynamic = []
            for v in self.dynamic:
                if type(v) is dict:
                    v = InputDFSchema(**v)
                v.subject_id_col = self.static.subject_id_col

                new_dynamic.append(v)

                if v.is_static:
                    raise ValueError("Must pass dynamic schemas in self.dynamic!")
            self.dynamic = new_dynamic

        self.dynamic_by_df = defaultdict(list)
        for v in self.dynamic:
            self.dynamic_by_df[v.input_df].append(v)
        self.dynamic_by_df = {k: v for k, v in self.dynamic_by_df.items()}


@dataclasses.dataclass
class InputDFSchema(JSONableMixin):
    """The schema for one input DataFrame.

    Dataclass that defines the schema for an input DataFrame. It verifies the provided attributes
    during the post-initialization stage, and raises exceptions if mandatory attributes are missing or
    if any inconsistencies are found. It stores sufficient data to extract subject IDs;
    produce event or range timestamps; extract, rename, and convert columns; and filter data.

    Attributes:
        input_df: DataFrame input. This can take on many types, including an actual dataframe, a query to a
            database, or a path to a dataframe stored on disk. Mandatory attribute.
        type: Type of the input data. Possible values are InputDFType.STATIC, InputDFType.EVENT,
            or InputDFType.RANGE. Mandatory attribute.
        event_type: What categorical event_type should be assigned to events sourced from this input
            dataframe? For events, must be only a single string, or for ranges can either be a single string
            or a tuple of strings indicating event type names for start, start == stop, and stop events. If
            the string starts with "COL:" then the remaining portion of the string will be interpreted as a
            column name in the input from which the event type should be read. Otherwise it will be
            intrepreted as a literal event_type category name.
        subject_id_col: The name of the column containing the subject ID.
        ts_col: Column name containing timestamp for events.
        start_ts_col: Column name containing start timestamp for ranges.
        end_ts_col: Column name containing end timestamp for ranges.
        ts_format: String format of the timestamp in ts_col.
        start_ts_format: String format of the timestamp in start_ts_col.
        end_ts_format: String format of the timestamp in end_ts_col.
        data_schema: Schema of the input data.
        start_data_schema: Schema of the start data in a range. If unspecified for a range, will fall back on
            data_schema.
        end_data_schema: Schema of the end data in a range. If unspecified for a range, will fall back on
            data_schema.
        must_have: List of mandatory columns or filters to apply, as a mapping from column name to filter to
            apply. The filter can either be `True`, in which case the column simply must have a non-null
            value, or a list of options, in which case the column must take on one of those values for the row
            to be included.

    Raises:
        ValueError: If mandatory attributes (input_df, type) are not provided, or if inconsistencies
            are found in the attributes based on the input data type.
        TypeError: If attributes are of the wrong type.

    Examples:
        >>> S = InputDFSchema(
        ...     input_df="/path/to/df.csv",
        ...     type='static',
        ...     subject_id_col='subj_id',
        ...     must_have=['subj_id', ['foo', ['opt1', 'opt2']]],
        ... )
        >>> S.filter_on
        {'subj_id': True, 'foo': ['opt1', 'opt2']}
        >>> S.is_static
        True
        >>> S = InputDFSchema(
        ...     input_df="/path/to_df.parquet",
        ...     type='event',
        ...     ts_col='col',
        ...     event_type='bar',
        ... )
        >>> S.is_static
        False
        >>> S
        InputDFSchema(input_df='/path/to_df.parquet', type='event', event_type='bar', ts_col='col')
        >>> InputDFSchema()
        Traceback (most recent call last):
            ...
        ValueError: Missing mandatory parameter input_df!
        >>> S = InputDFSchema(input_df="/path/to/df.csv")
        Traceback (most recent call last):
            ...
        ValueError: Missing mandatory parameter type!
        >>> S = InputDFSchema(
        ...     input_df="/path/to/df.csv",
        ...     type='static',
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Must set subject_id_col for static source!
        >>> S = InputDFSchema(
        ...     input_df="/path/to/df.csv",
        ...     type='static',
        ...     subject_id_col='subj_id',
        ...     must_have=[34]
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Malformed filter: 34
        >>> S = InputDFSchema(
        ...     input_df="/path/to/df.parquet",
        ...     type=InputDFType.RANGE,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Missing mandatory range parameter event_type!
        >>>
    """

    input_df: Any | None = None

    type: InputDFType | None = None
    event_type: str | tuple[str, str, str] | None = None

    subject_id_col: str | None = None
    ts_col: DF_COL | None = None
    start_ts_col: DF_COL | None = None
    end_ts_col: DF_COL | None = None
    ts_format: str | None = None
    start_ts_format: str | None = None
    end_ts_format: str | None = None

    data_schema: DF_SCHEMA | list[DF_SCHEMA] | None = None
    start_data_schema: DF_SCHEMA | list[DF_SCHEMA] | None = None
    end_data_schema: DF_SCHEMA | list[DF_SCHEMA] | None = None

    must_have: list[str | tuple[str, list[Any]]] = dataclasses.field(default_factory=list)

    @property
    def is_static(self):
        """Returns True if and only if the input data type is static."""
        return self.type == InputDFType.STATIC

    def __post_init__(self):
        if self.input_df is None:
            raise ValueError("Missing mandatory parameter input_df!")
        if self.type is None:
            raise ValueError("Missing mandatory parameter type!")
        if type(self.data_schema) is not list and self.data_schema is not None:
            self.data_schema = [self.data_schema]
        if type(self.start_data_schema) is not list and self.start_data_schema is not None:
            self.start_data_schema = [self.start_data_schema]
        if type(self.end_data_schema) is not list and self.end_data_schema is not None:
            self.end_data_schema = [self.end_data_schema]

        self.filter_on = {}
        for filter_col in self.must_have:
            match filter_col:
                case str():
                    self.filter_on[filter_col] = True
                case (str() as filter_col, list() as vals):
                    self.filter_on[filter_col] = vals
                case _:
                    raise ValueError(f"Malformed filter: {filter_col}")

        match self.type:
            case InputDFType.STATIC:
                if self.subject_id_col is None:
                    raise ValueError("Must set subject_id_col for static source!")
                if self.event_type is not None:
                    raise ValueError("Event_type can't be set if type == 'static'!")

                for param in ("ts_col", "start_ts_col", "end_ts_col"):
                    if getattr(self, param) is not None:
                        raise ValueError(f"Set invalid param {param} for static source!")

            case InputDFType.EVENT:
                if self.ts_col is None:
                    raise ValueError("Missing mandatory event parameter ts_col!")
                match self.event_type:
                    case None:
                        raise ValueError("Missing mandatory range parameter event_type!")
                    case str():
                        pass
                    case _:
                        raise TypeError(f"event_type must be a string for events. Got {self.event_type}")
                if self.subject_id_col is not None:
                    raise ValueError("subject_id_col should be None for non-static types!")
                for param in (
                    "start_ts_col",
                    "end_ts_col",
                    "start_ts_format",
                    "end_ts_format",
                    "start_data_schema",
                    "end_data_schema",
                ):
                    val = getattr(self, param)
                    if val is not None:
                        raise ValueError(f"{param} should be None for {self.type} schema: Got {val}")

            case InputDFType.RANGE:
                match self.event_type:
                    case None:
                        raise ValueError("Missing mandatory range parameter event_type!")
                    case (str(), str(), str()):
                        pass
                    case str():
                        self.event_type = (
                            self.event_type,
                            f"{self.event_type}_START",
                            f"{self.event_type}_END",
                        )
                    case _:
                        raise TypeError(
                            "event_type must be a string or a 3-element tuple (eq_type, st_type, end_type) "
                            f"for ranges. Got {self.event_type}."
                        )

                if self.data_schema is not None:
                    for param in ("start_data_schema", "end_data_schema"):
                        val = getattr(self, param)
                        if val is not None:
                            raise ValueError(
                                f"{param} can't be simultaneously set with `self.data_schema`! Got {val}"
                            )

                    self.start_data_schema = self.data_schema
                    self.end_data_schema = self.data_schema

                if self.start_ts_col is None:
                    raise ValueError("Missing mandatory range parameter start_ts_col!")
                if self.end_ts_col is None:
                    raise ValueError("Missing mandatory range parameter end_ts_col!")
                if self.ts_col is not None:
                    raise ValueError(f"ts_col should be `None` for {self.type} schemas! Got: {self.ts_col}.")
                if self.subject_id_col is not None:
                    raise ValueError("subject_id_col should be None for non-static types!")
                if self.start_ts_format is not None:
                    if self.end_ts_format is None:
                        raise ValueError(
                            "If start_ts_format is specified, end_ts_format must also be specified!"
                        )
                    if self.ts_format is not None:
                        raise ValueError("If start_ts_format is specified, ts_format must be `None`!")
                else:
                    if self.end_ts_format is not None:
                        raise ValueError(
                            "If end_ts_format is specified, start_ts_format must also be specified!"
                        )

                    self.start_ts_format = self.ts_format
                    self.end_ts_format = self.ts_format
                    self.ts_format = None

        # This checks validity.
        self.columns_to_load

    def __repr__(self) -> str:
        kwargs = {k: v for k, v in self.to_dict().items() if v}
        kwargs_str = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        return f"InputDFSchema({', '.join(kwargs_str)})"

    @property
    def columns_to_load(self) -> list[tuple[str, InputDataType]]:
        """Computes the columns to be loaded based on the input data type and schema.

        Returns:
            A list of tuples of column names and desired types for the columns to be loaded from the input
            dataframe.

        Raises:
            ValueError: If any of the column definitions are invalid or repeated.
        """
        columns_to_load = {}

        match self.type:
            case InputDFType.EVENT | InputDFType.STATIC:
                for in_col, (out_col, dt) in self.unified_schema.items():
                    if in_col in columns_to_load:
                        raise ValueError(f"Duplicate column {in_col}!")
                    columns_to_load[in_col] = dt
            case InputDFType.RANGE:
                for unified_schema in self.unified_schema:
                    for in_col, (out_col, dt) in unified_schema.items():
                        if in_col in columns_to_load:
                            if dt != columns_to_load[in_col]:
                                raise ValueError(f"Duplicate column {in_col} with differing dts!")
                        else:
                            columns_to_load[in_col] = dt
            case _:
                raise ValueError(f"Unrecognized type {self.type}!")

        columns_to_load = list(columns_to_load.items())

        for param, fmt_param in [
            ("start_ts_col", "start_ts_format"),
            ("end_ts_col", "end_ts_format"),
            ("ts_col", "ts_format"),
        ]:
            val = getattr(self, param)
            fmt_param = getattr(self, fmt_param)
            if fmt_param is None:
                fmt = InputDataType.TIMESTAMP
            else:
                fmt = (InputDataType.TIMESTAMP, fmt_param)

            match val:
                case list():
                    columns_to_load.extend([(c, fmt) for c in val])
                case str():
                    columns_to_load.append((val, fmt))
                case None:
                    pass
                case _:
                    raise ValueError(f"Can't parse timestamp {param}, {fmt_param}, {val}")

        return columns_to_load

    @property
    def unified_schema(self) -> dict[str, tuple[str, InputDataType]]:
        """Computes the unified schema based on the input data type and data schema.

        Returns:
            A unified schema mapping from output column names to input column names and types.

        Raises:
            ValueError: If the type attribute of the calling object is invalid.
        """
        match self.type:
            case InputDFType.EVENT | InputDFType.STATIC:
                return self.unified_event_schema
            case InputDFType.RANGE:
                return [self.unified_eq_schema, self.unified_start_schema, self.unified_end_schema]
            case _:
                raise ValueError(f"Unrecognized type {self.type}!")

    @property
    def unified_event_schema(self) -> dict[str, tuple[str, InputDataType]]:
        return self._unify_schema(self.data_schema)

    @property
    def unified_start_schema(self) -> dict[str, tuple[str, InputDataType]]:
        if self.type != InputDFType.RANGE:
            raise ValueError(f"Start schema is invalid for {self.type}")

        if self.start_data_schema is None:
            return self._unify_schema(self.data_schema)
        return self._unify_schema(self.start_data_schema)

    @property
    def unified_end_schema(self) -> dict[str, tuple[str, InputDataType]]:
        if self.type != InputDFType.RANGE:
            raise ValueError(f"End schema is invalid for {self.type}")

        if self.end_data_schema is None:
            return self._unify_schema(self.data_schema)
        return self._unify_schema(self.end_data_schema)

    @property
    def unified_eq_schema(self) -> dict[str, tuple[str, InputDataType]]:
        if self.type != InputDFType.RANGE:
            raise ValueError(f"Start=End schema is invalid for {self.type}")

        if self.start_data_schema is None and self.end_data_schema is None:
            return self._unify_schema(self.data_schema)

        ds = []
        if self.start_data_schema is not None:
            if type(self.start_data_schema) is list:
                ds.extend(self.start_data_schema)
            else:
                ds.append(self.start_data_schema)

        if self.end_data_schema is not None:
            if type(self.end_data_schema) is list:
                ds.extend(self.end_data_schema)
            else:
                ds.append(self.end_data_schema)

        return self._unify_schema(ds)

    @classmethod
    def __add_to_schema(
        cls,
        container: dict[str, tuple[str, InputDataType]],
        in_col: str,
        dt: INPUT_COL_T,
        out_col: str | None = None,
    ):
        if out_col is None:
            out_col = in_col

        if type(in_col) is not str or type(out_col) is not str:
            raise ValueError(f"Column names must be strings! Got {in_col}, {out_col}")
        elif in_col in container and container[in_col] != (out_col, dt):
            raise ValueError(
                f"Column {in_col} is repeated in schema with different value!\n"
                f"Existing: {container[in_col]}\n"
                f"New: ({out_col}, {dt})"
            )
        container[in_col] = (out_col, dt)

    @classmethod
    def _unify_schema(
        cls, data_schema: DF_SCHEMA | list[DF_SCHEMA] | None
    ) -> dict[str, tuple[str, InputDataType]]:
        if data_schema is None:
            return {}

        unified_schema = {}
        for schema in data_schema:
            match schema:
                case str() as col, (InputDataType() | [InputDataType.TIMESTAMP, str()]) as dt:
                    cls.__add_to_schema(unified_schema, in_col=col, dt=dt)
                case list() as cols, (InputDataType() | [InputDataType.TIMESTAMP, str()]) as dt:
                    for c in cols:
                        cls.__add_to_schema(unified_schema, in_col=c, dt=dt)
                case dict():
                    for in_col, schema_info in schema.items():
                        match schema_info:
                            case str() as out_col, str() as dt if dt in InputDataType.values():
                                cls.__add_to_schema(unified_schema, in_col=in_col, dt=dt, out_col=out_col)
                            case str() as out_col, InputDataType() as dt:
                                cls.__add_to_schema(unified_schema, in_col=in_col, dt=dt, out_col=out_col)
                            case str() as out_col, [InputDataType.TIMESTAMP, str()] as dt:
                                cls.__add_to_schema(unified_schema, in_col=in_col, dt=dt, out_col=out_col)
                            case [InputDataType.TIMESTAMP, str()] as dt:
                                cls.__add_to_schema(unified_schema, in_col=in_col, dt=dt)
                            case str() | InputDataType() as dt if dt in InputDataType.values():
                                cls.__add_to_schema(unified_schema, in_col=in_col, dt=dt)
                            case _:
                                raise ValueError(f"Schema Unprocessable!\n{schema_info}")
                case dict() as col_names_map, (InputDataType() | [InputDataType.TIMESTAMP, str()]) as dt:
                    for in_col, out_col in col_names_map.items():
                        cls.__add_to_schema(unified_schema, in_col=in_col, dt=dt, out_col=out_col)
                case _:
                    raise ValueError(f"Schema Unprocessable!\n{schema}")

        return unified_schema


@dataclasses.dataclass
class VocabularyConfig(JSONableMixin):
    """Dataclass that describes the vocabulary of a dataset, for initializing model parameters.

    This does not configure a vocabulary, but rather describes the vocabulary learned during dataset
    pre-processing for an entire dataset. This description includes the sizes of all per-measurement
    vocabularies (where measurements without a vocabulary, such as univariate regression measurements) are
    omitted as their vocabularies have size 1, vocabulary offsets per measurement, which detail how the
    various vocabularies are stuck together to form a unified vocabulary, the indices of each global
    measurement type, the generative modes used by each measurement, and the event type indices.

    Attributes:
        vocab_sizes_by_measurement: A dictionary mapping measurements to their respective vocabulary sizes.
        vocab_offsets_by_measurement: A dictionary mapping measurements to their respective vocabulary
            offsets.
        measurements_idxmap: A dictionary mapping measurements to their integer indices.
        measurements_per_generative_mode: A dictionary mapping data modality to a list of measurements.
        event_types_idxmap: A dictionary mapping event types to their respective indices.
    """

    vocab_sizes_by_measurement: dict[str, int] | None = None
    vocab_offsets_by_measurement: dict[str, int] | None = None
    measurements_idxmap: dict[str, dict[Hashable, int]] | None = None
    measurements_per_generative_mode: dict[DataModality, list[str]] | None = None
    event_types_idxmap: dict[str, int] | None = None

    @property
    def total_vocab_size(self) -> int:
        """Returns the total vocab size of the vocabulary described here.

        The total vocabulary size is the sum of (1) all the individual measurement vocabularies' sizes, (2)
        any offset the global vocabulary has from 0, to account for padding indices, and (3) any measurements
        who have length-1 vocabularies (which are not included in `vocab_sizes_by_measurement`) as is
        reflected by elements in the vocab offsets dictionary that aren't in the vocab sizes dictionary.

        Examples:
            >>> config = VocabularyConfig(
            ...     vocab_sizes_by_measurement={"measurement1": 10, "measurement2": 3},
            ...     vocab_offsets_by_measurement={"measurement1": 5, "measurement2": 15, "measurement3": 18}
            ... )
            >>> config.total_vocab_size
            19
        """
        return (
            sum(self.vocab_sizes_by_measurement.values())
            + min(self.vocab_offsets_by_measurement.values())
            + (len(self.vocab_offsets_by_measurement) - len(self.vocab_sizes_by_measurement))
        )


class SeqPaddingSide(StrEnum):
    """Enumeration for the side of sequence padding during PyTorch Batch construction."""

    RIGHT = enum.auto()
    """Pad on the right side (at the end of the sequence).

    This is the default during normal training.
    """

    LEFT = enum.auto()
    """Pad on the left side (at the beginning of the sequence).

    This is the default during generation.
    """


class SubsequenceSamplingStrategy(StrEnum):
    """Enumeration for subsequence sampling strategies.

    When the maximum allowed sequence length for a PyTorchDataset is shorter than the sequence length of a
    subject's data, this enumeration dictates how we sample a subsequence to include.
    """

    TO_END = enum.auto()
    """Sample subsequences of the maximum length up to the end of the permitted window.

    This is the default during fine-tuning and with task dataframes.
    """

    FROM_START = enum.auto()
    """Sample subsequences of the maximum length from the start of the permitted window."""

    RANDOM = enum.auto()
    """Sample subsequences of the maximum length randomly within the permitted window.

    This is the default during pre-training.
    """


@hydra_dataclass
class PytorchDatasetConfig(JSONableMixin):
    """Configuration options for building a PyTorch dataset from a `Dataset`.

    This is the main configuration object for a `PytorchDataset`. The `PytorchDataset` class specializes the
    representation of the data in a base `Dataset` class for sequential deep learning. This dataclass is also
    an acceptable `Hydra Structured Config`_ object with the name "pytorch_dataset_config".

    .. _Hydra Structured Config: https://hydra.cc/docs/tutorials/structured_config/intro/

    Attributes:
        save_dir: Directory where the base dataset, including the deep learning representation outputs, is
            saved.
        max_seq_len: Maximum sequence length the dataset should output in any individual item.
        min_seq_len: Minimum sequence length required to include a subject in the dataset.
        seq_padding_side: Whether to pad smaller sequences on the right or the left.
        subsequence_sampling_strategy: Strategy for sampling subsequences when an individual item's total
            sequence length in the raw data exceeds the maximum allowed sequence length.
        train_subset_size: If the training data should be subsampled randomly, this specifies the size of the
            training subset. If `None` or "FULL", then the full training data is used.
        train_subset_seed: If the training data should be subsampled randomly, this specifies the seed for
            that random subsampling. Should be None if train_subset_size is None or "FULL".
        task_df_name: If the raw dataset should be limited to a task dataframe view, this specifies the name
            of the task dataframe, and indirectly the path on disk from where that task dataframe will be
            read (save_dir / "task_dfs" / f"{task_df_name}.parquet").
        do_include_subject_id: Whether or not to include the subject ID of the individual for this batch.
        do_include_subsequence_indices: Whether or not to include the start and end indices of the sampled
            subsequence for the individual from their full dataset for this batch. This is sometimes used
            during generative-based evaluation.
        do_include_start_time_min: Whether or not to include the start time of the individual's sequence in
            minutes since the epoch (1/1/1970) in the output data. This is necessary during generation, and
            not used anywhere else currently.

    Raises:
        ValueError: If 'seq_padding_side' is not a valid value; If 'min_seq_len' is not a non-negative
            integer; If 'max_seq_len' is not an integer greater or equal to 'min_seq_len'; If
            'train_subset_seed' is not None when 'train_subset_size' is None or 'FULL'; If 'train_subset_size'
            is negative when it's an integer; If 'train_subset_size' is not within (0, 1) when it's a float.
        TypeError: If 'train_subset_size' is of unrecognized type.

    Examples:
        >>> config = PytorchDatasetConfig(
        ...     save_dir='./dataset',
        ...     max_seq_len=256,
        ...     min_seq_len=2,
        ...     seq_padding_side=SeqPaddingSide.RIGHT,
        ...     subsequence_sampling_strategy=SubsequenceSamplingStrategy.RANDOM,
        ...     train_subset_size="FULL",
        ...     train_subset_seed=None,
        ...     task_df_name=None,
        ...     do_include_start_time_min=False
        ... )
        >>> config_dict = config.to_dict()
        >>> new_config = PytorchDatasetConfig.from_dict(config_dict)
        >>> config == new_config
        True
        >>> config = PytorchDatasetConfig(train_subset_size=-1)
        Traceback (most recent call last):
            ...
        ValueError: If integral, train_subset_size must be positive! Got -1
        >>> config = PytorchDatasetConfig(train_subset_size=1.2)
        Traceback (most recent call last):
            ...
        ValueError: If float, train_subset_size must be in (0, 1)! Got 1.2
        >>> config = PytorchDatasetConfig(train_subset_size='200')
        Traceback (most recent call last):
            ...
        TypeError: train_subset_size is of unrecognized type <class 'str'>.
        >>> config = PytorchDatasetConfig(train_subset_size="FULL", train_subset_seed=10)
        Traceback (most recent call last):
            ...
        ValueError: train_subset_seed 10 should be None if train_subset_size is FULL.
        >>> config = PytorchDatasetConfig(
        ...     save_dir='./dataset',
        ...     max_seq_len=256,
        ...     min_seq_len=2,
        ...     seq_padding_side='left',
        ...     subsequence_sampling_strategy=SubsequenceSamplingStrategy.RANDOM,
        ...     train_subset_size=100,
        ...     train_subset_seed=None,
        ...     task_df_name=None,
        ...     do_include_start_time_min=False
        ... )
        WARNING! train_subset_size is set, but train_subset_seed is not. Setting to...
        >>> assert config.train_subset_seed is not None
    """

    save_dir: Path = omegaconf.MISSING

    max_seq_len: int = 256
    min_seq_len: int = 2
    seq_padding_side: SeqPaddingSide = SeqPaddingSide.RIGHT
    subsequence_sampling_strategy: SubsequenceSamplingStrategy = SubsequenceSamplingStrategy.RANDOM

    train_subset_size: int | float | str = "FULL"
    train_subset_seed: int | None = None

    task_df_name: str | None = None

    do_include_subsequence_indices: bool = False
    do_include_subject_id: bool = False
    do_include_start_time_min: bool = False

    def __post_init__(self):
        if self.seq_padding_side not in SeqPaddingSide.values():
            raise ValueError(f"seq_padding_side invalid; must be in {', '.join(SeqPaddingSide.values())}")
        if type(self.min_seq_len) is not int or self.min_seq_len < 0:
            raise ValueError(f"min_seq_len must be a non-negative integer; got {self.min_seq_len}")
        if type(self.max_seq_len) is not int or self.max_seq_len < self.min_seq_len:
            raise ValueError(
                f"max_seq_len must be an integer at least equal to min_seq_len; got {self.max_seq_len} "
                f"(min_seq_len = {self.min_seq_len})"
            )

        if type(self.save_dir) is str and self.save_dir != omegaconf.MISSING:
            self.save_dir = Path(self.save_dir)

        match self.train_subset_size:
            case (None | "FULL") if self.train_subset_seed is not None:
                raise ValueError(
                    f"train_subset_seed {self.train_subset_seed} should be None "
                    f"if train_subset_size is {self.train_subset_size}."
                )
            case int() as n if n < 0:
                raise ValueError(f"If integral, train_subset_size must be positive! Got {n}")
            case float() as frac if frac <= 0 or frac >= 1:
                raise ValueError(f"If float, train_subset_size must be in (0, 1)! Got {frac}")
            case int() | float() if (self.train_subset_seed is None):
                seed = int(random.randint(1, int(1e6)))
                print(f"WARNING! train_subset_size is set, but train_subset_seed is not. Setting to {seed}")
                self.train_subset_seed = seed
            case None | "FULL" | int() | float():
                pass
            case _:
                raise TypeError(f"train_subset_size is of unrecognized type {type(self.train_subset_size)}.")

    def to_dict(self) -> dict:
        """Represents this configuration object as a plain dictionary."""
        as_dict = dataclasses.asdict(self)
        as_dict["save_dir"] = str(as_dict["save_dir"])
        return as_dict

    @classmethod
    def from_dict(cls, as_dict: dict) -> PytorchDatasetConfig:
        """Creates a new instance of this class from a plain dictionary."""
        as_dict["save_dir"] = Path(as_dict["save_dir"])
        return cls(**as_dict)


@dataclasses.dataclass
class MeasurementConfig(JSONableMixin):
    """The Configuration class for a measurement in the Dataset.

    A measurement is any observation in the dataset; be it static or dynamic, categorical or continuous. This
    class contains configuration options to define a measurement and dictate how it should be pre-processed,
    embedded, and generated in generative models.

    Attributes:
        name:
            Stores the name of this measurement; also the column in the appropriate internal dataframe
            (`subjects_df`, `events_df`, or `dynamic_measurements_df`) that will contain this measurement. All
            measurements will have this set.

            The 'column' linkage has slightly different meanings depending on `self.modality`:

            * If `modality == DataModality.UNIVARIATE_REGRESSION`, then this column stores the values
              associated with this continuous-valued measure.
            * If `modality == DataModality.MULTIVARIATE_REGRESSION`, then this column stores the keys that
              dictate the dimensions for which the associated `values_column` has the values.
            * Otherwise, this column stores the categorical values of this measure.

            Similarly, it has slightly different meanings depending on `self.temporality`:

            * If `temporality == TemporalityType.STATIC`, this is an existent column in the `subjects_df`
              dataframe.
            * If `temporality == TemporalityType.DYNAMIC`, this is an existent column in the
              `dynamic_measurements_df` dataframe.
            * Otherwise, (when `temporality == TemporalityType.FUNCTIONAL_TIME_DEPENDENT`), then this is
              the name the *output-to-be-created* column will take in the `events_df` dataframe.

        modality: The modality of this measurement. If `DataModality.UNIVARIATE_REGRESSION`, then this
            measurement takes on single-variate continuous values. If `DataModality.MULTIVARIATE_REGRESSION`,
            then this measurement consists of key-value pairs of categorical covariate identifiers and
            continuous values. Keys are stored in the column reflected in `self.name` and values in
            `self.values_column`.
        temporality: How this measure varies in time. If `TemporalityType.STATIC`, this is a static
            measurement. If `TemporalityType.FUNCTIONAL_TIME_DEPENDENT`, then this measurement is a
            time-dependent measure that varies with time and static data in an analytically computable manner
            (e.g., age). If `TemporalityType.DYNAMIC`, then this is a measurement that varies in time in a
            non-a-priori computable manner.
        observation_frequency: The fraction of valid instances in which this measure is observed. Is set
            dynamically during pre-procesisng, and not specified at construction.
        functor: If `temporality == TemporalityType.FUNCTIONAL_TIME_DEPENDENT`, then this will be set to the
            functor used to compute the value of a known-time-depedency measure. In this case, `functor` must
            be a subclass of `data.time_dependent_functor.TimeDependentFunctor`. If `temporality` is anything
            else, then this will be `None`.
        vocabulary: The vocabulary for this column, realized as a `Vocabulary` object. Begins with `'UNK'`.
            Not set on `modality==UNIVARIATE_REGRESSION` measurements.
        values_column: For `modality==MULTIVARIATE_REGRESSION` measurements, this will store the name of the
            column which will contain the numerical values corresponding to this measurement. Otherwise will
            be `None`.
        measurement_metadata: Stores metadata about the numerical values corresponding to this measurement.
            This can take one of two forms, depending on the measurement modality. If
            `modality==UNIVARIATE_REGRESSION`, then this will be a `pd.Series` whose index will contain the
            set of possible column headers listed below. If `modality==MULTIVARIATE_REGRESSION`, then this
            will be a `pd.DataFrame`, whose index will contain the possible regression covariate identifier
            keys and whose columns will contain the set of possible columns listed below.

            Metadata Columns:

            * drop_lower_bound: A lower bound such that values either below or at or below this level will
              be dropped (key presence will be retained for multivariate regression measures). Optional.
            * drop_lower_bound_inclusive: This must be set if `drop_lower_bound` is set. If this is true,
              then values will be dropped if they are $<=$ `drop_lower_bound`. If it is false, then values
              will be dropped if they are $<$ `drop_lower_bound`.
            * censor_lower_bound: A lower bound such that values either below or at or below this level,
              but above the level of `drop_lower_bound`, will be replaced with the value
              `censor_lower_bound`. Optional.
            * drop_upper_bound An upper bound such that values either above or at or above this level will
              be dropped (key presence will be retained for multivariate regression measures). Optional.
            * drop_upper_bound_inclusive: This must be set if `drop_upper_bound` is set. If this is true,
              then values will be dropped if they are $>=$ `drop_upper_bound`. If it is false, then values
              will be dropped if they are $>$ `drop_upper_bound`.
            * censor_upper_bound: An upper bound such that values either above or at or above this level,
              but below the level of `drop_upper_bound`, will be replaced with the value
              `censor_upper_bound`. Optional.
            * value_type: To which kind of value (e.g., integer, categorical, float) this key corresponds.
              Must be an element of the enum `NumericMetadataValueType`. Optional. If not pre-specified,
              will be inferred from the data.
            * outlier_model: The parameters (in dictionary form) for the fit outlier model. Optional. If
              not pre-specified, will be inferred from the data.
            * normalizer: The parameters (in dictionary form) for the fit normalizer model. Optional. If
              not pre-specified, will be inferred from the data.

    Raises:
        ValueError: If the configuration is not self consistent (e.g., a functor specified on a
            non-functional_time_dependent measure).
        NotImplementedError: If the configuration relies on a measurement configuration that is not yet
            supported, such as numeric, static measurements.


    Examples:
        >>> cfg = MeasurementConfig(
        ...     name='key',
        ...     modality='multi_label_classification',
        ...     temporality='dynamic',
        ...     vocabulary=Vocabulary(['foo', 'bar', 'baz'], [0.3, 0.4, 0.3]),
        ... )
        >>> cfg.is_numeric
        False
        >>> cfg.is_dropped
        False
        >>> cfg = MeasurementConfig(
        ...     name='key',
        ...     modality='univariate_regression',
        ...     temporality='dynamic',
        ...     _measurement_metadata=pd.Series([1, 0.2], index=['censor_upper_bound', 'censor_lower_bound']),
        ... )
        >>> cfg.is_numeric
        True
        >>> cfg.is_dropped
        False
        >>> cfg = MeasurementConfig(
        ...     name='key',
        ...     modality='multivariate_regression',
        ...     temporality='dynamic',
        ...     values_column='vals',
        ...     _measurement_metadata=pd.DataFrame(
        ...         {'censor_lower_bound': [1, 0.2, 0.1]},
        ...         index=pd.Index(['foo', 'bar', 'baz'], name='key'),
        ...     ),
        ...     vocabulary=Vocabulary(['foo', 'bar', 'baz'], [0.3, 0.4, 0.3]),
        ... )
        >>> cfg.is_numeric
        True
        >>> cfg.is_dropped
        False
        >>> MeasurementConfig()
        Traceback (most recent call last):
            ...
        ValueError: `self.temporality = None` Invalid! Must be in static, dynamic, functional_time_dependent
        >>> MeasurementConfig(
        ...     temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
        ...     functor=None,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: functor must be set for functional_time_dependent measurements!
        >>> MeasurementConfig(
        ...     temporality=TemporalityType.STATIC,
        ...     functor=AgeFunctor(dob_col="date_of_birth"),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: functor should be None for static measurements! Got ...
        >>> MeasurementConfig(
        ...     temporality=TemporalityType.DYNAMIC,
        ...     modality=DataModality.MULTIVARIATE_REGRESSION,
        ...     _measurement_metadata=pd.Series([1, 10], index=['censor_lower_bound', 'censor_upper_bound']),
        ...     values_column='vals',
        ... )
        Traceback (most recent call last):
            ...
        ValueError: If set, measurement_metadata must be a DataFrame on a multivariate_regression\
 MeasurementConfig. Got <class 'pandas.core.series.Series'>
        censor_lower_bound     1
        censor_upper_bound    10
        dtype: int64
    """

    FUNCTORS = {
        "AgeFunctor": AgeFunctor,
        "TimeOfDayFunctor": TimeOfDayFunctor,
    }

    PREPROCESSING_METADATA_COLUMNS = OrderedDict(
        {
            "value_type": str,
            "outlier_model": object,
            "normalizer": object,
        }
    )

    # Present in all measures
    name: str | None = None
    temporality: TemporalityType | None = None
    modality: DataModality | None = None
    observation_frequency: float | None = None

    # Specific to time-dependent measures
    functor: TimeDependentFunctor | None = None

    # Specific to categorical or partially observed multivariate regression measures.
    vocabulary: Vocabulary | None = None

    # Specific to numeric measures
    values_column: str | None = None
    _measurement_metadata: pd.DataFrame | pd.Series | str | Path | None = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Checks the internal state of `self` and ensures internal consistency and validity."""
        match self.temporality:
            case TemporalityType.STATIC:
                if self.functor is not None:
                    raise ValueError(
                        f"functor should be None for {self.temporality} measurements! Got {self.functor}"
                    )

                if self.is_numeric:
                    raise NotImplementedError(
                        f"Numeric data modalities like {self.modality} not yet supported on static measures."
                    )
            case TemporalityType.DYNAMIC:
                if self.functor is not None:
                    raise ValueError(
                        f"functor should be None for {self.temporality} measurements! Got {self.functor}"
                    )
                if self.modality == DataModality.SINGLE_LABEL_CLASSIFICATION:
                    raise ValueError(
                        f"{self.modality} on {self.temporality} measurements is not currently supported, as "
                        "event aggregation can turn single-label tasks into multi-label tasks in a manner "
                        "that is not currently automatically detected or compensated for."
                    )

            case TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                if self.functor is None:
                    raise ValueError(f"functor must be set for {self.temporality} measurements!")

                if self.modality is None:
                    self.modality = self.functor.OUTPUT_MODALITY
                elif self.modality not in (DataModality.DROPPED, self.functor.OUTPUT_MODALITY):
                    raise ValueError(
                        "self.modality must either be DataModality.DROPPED or "
                        f"{self.functor.OUTPUT_MODALITY} for {self.temporality} measures; got {self.modality}"
                    )
            case _:
                raise ValueError(
                    f"`self.temporality = {self.temporality}` Invalid! Must be in "
                    f"{', '.join(TemporalityType.values())}"
                )

        err_strings = []
        match self.modality:
            case DataModality.MULTIVARIATE_REGRESSION:
                if self.values_column is None:
                    err_strings.append(f"values_column must be set on a {self.modality} MeasurementConfig")
                if (self.measurement_metadata is not None) and not isinstance(
                    self.measurement_metadata, pd.DataFrame
                ):
                    err_strings.append(
                        f"If set, measurement_metadata must be a DataFrame on a {self.modality} "
                        f"MeasurementConfig. Got {type(self.measurement_metadata)}\n"
                        f"{self.measurement_metadata}"
                    )
            case DataModality.UNIVARIATE_REGRESSION:
                if self.values_column is not None:
                    err_strings.append(
                        f"values_column must be None on a {self.modality} MeasurementConfig. "
                        f"Got {self.values_column}"
                    )
                if (self.measurement_metadata is not None) and not isinstance(
                    self.measurement_metadata, pd.Series
                ):
                    err_strings.append(
                        f"If set, measurement_metadata must be a Series on a {self.modality} "
                        f"MeasurementConfig. Got {type(self.measurement_metadata)}\n"
                        f"{self.measurement_metadata}"
                    )
            case DataModality.SINGLE_LABEL_CLASSIFICATION | DataModality.MULTI_LABEL_CLASSIFICATION:
                if self.values_column is not None:
                    err_strings.append(
                        f"values_column must be None on a {self.modality} MeasurementConfig. "
                        f"Got {self.values_column}"
                    )
                if self._measurement_metadata is not None:
                    err_strings.append(
                        f"measurement_metadata must be None on a {self.modality} MeasurementConfig. "
                        f"Got {type(self.measurement_metadata)}\n{self.measurement_metadata}"
                    )
            case DataModality.DROPPED:
                if self.vocabulary is not None:
                    err_strings.append(
                        f"vocabulary must be None on a {self.modality} MeasurementConfig. "
                        f"Got {self.vocabulary}"
                    )
                if self._measurement_metadata is not None:
                    err_strings.append(
                        f"measurement_metadata must be None on a {self.modality} MeasurementConfig. "
                        f"Got {type(self.measurement_metadata)}\n{self.measurement_metadata}"
                    )
            case _:
                raise ValueError(f"`self.modality = {self.modality}` Invalid!")
        if err_strings:
            raise ValueError("\n".join(err_strings))

    def drop(self):
        """Sets the modality to DROPPED and does associated post-processing to ensure validity.

        Examples:
            >>> cfg = MeasurementConfig(
            ...     name='key',
            ...     modality='multivariate_regression',
            ...     temporality='dynamic',
            ...     values_column='vals',
            ...     _measurement_metadata=pd.DataFrame(
            ...         {'censor_lower_bound': [1, 0.2, 0.1]},
            ...         index=pd.Index(['foo', 'bar', 'baz'], name='key'),
            ...     ),
            ...     vocabulary=Vocabulary(['foo', 'bar', 'baz'], [0.3, 0.4, 0.3]),
            ... )
            >>> cfg.drop()
            >>> cfg.modality
            <DataModality.DROPPED: 'dropped'>
            >>> assert cfg._measurement_metadata is None
            >>> assert cfg.vocabulary is None
            >>> assert cfg.is_dropped
        """
        self.modality = DataModality.DROPPED
        self._measurement_metadata = None
        self.vocabulary = None

    @property
    def is_dropped(self) -> bool:
        return self.modality == DataModality.DROPPED

    @property
    def is_numeric(self) -> bool:
        return self.modality in (
            DataModality.MULTIVARIATE_REGRESSION,
            DataModality.UNIVARIATE_REGRESSION,
        )

    @property
    def measurement_metadata(self) -> pd.DataFrame | pd.Series | None:
        match self._measurement_metadata:
            case None | pd.DataFrame() | pd.Series():
                return self._measurement_metadata
            case (Path() | str()) as fp:
                out = pd.read_csv(fp, index_col=0)

                if self.modality == DataModality.UNIVARIATE_REGRESSION:
                    if out.shape[1] != 1:
                        raise ValueError(
                            f"For {self.modality}, measurement metadata at {fp} should be a series, but "
                            f"it has shape {out.shape} (expecting out.shape[1] == 1)!"
                        )
                    out = out.iloc[:, 0]
                    for col in ("outlier_model", "normalizer"):
                        if col in out:
                            out[col] = eval(out[col])
                elif self.modality != DataModality.MULTIVARIATE_REGRESSION:
                    raise ValueError(
                        "Only DataModality.UNIVARIATE_REGRESSION and DataModality.MULTIVARIATE_REGRESSION "
                        f"measurements should have measurement metadata paths stored. Got {fp} on "
                        f"{self.modality} measurement!"
                    )
                else:
                    for col in ("outlier_model", "normalizer"):
                        if col in out:
                            out[col] = out[col].apply(eval)
                return out
            case _:
                raise ValueError(f"_measurement_metadata is invalid! Got {type(self.measurement_metadata)}!")

    @measurement_metadata.setter
    def measurement_metadata(self, new_metadata: pd.DataFrame | pd.Series | None):
        if new_metadata is None:
            self._measurement_metadata = None
            return

        if isinstance(self._measurement_metadata, (str, Path)):
            new_metadata.to_csv(self._measurement_metadata)
        else:
            self._measurement_metadata = new_metadata

    def cache_measurement_metadata(self, fp: Path):
        if isinstance(self._measurement_metadata, (str, Path)):
            if str(fp) != str(self._measurement_metadata):
                raise ValueError(f"Caching is already enabled at {self._measurement_metadata} != {fp}")
            return
        if self.measurement_metadata is None:
            return

        fp.parent.mkdir(exist_ok=True, parents=True)
        self.measurement_metadata.to_csv(fp)
        self._measurement_metadata = str(fp.resolve())

    def uncache_measurement_metadata(self):
        if self._measurement_metadata is None:
            return

        if not isinstance(self._measurement_metadata, (str, Path)):
            raise ValueError("Caching is not enabled, can't uncache!")

        self._measurement_metadata = self.measurement_metadata

    def add_empty_metadata(self):
        """Adds an empty `measurement_metadata` dataframe or series."""
        if self.measurement_metadata is not None:
            raise ValueError(f"Can't add empty metadata; already set to {self.measurement_metadata}")

        match self.modality:
            case DataModality.UNIVARIATE_REGRESSION:
                self._measurement_metadata = pd.Series(
                    [None] * len(self.PREPROCESSING_METADATA_COLUMNS),
                    index=self.PREPROCESSING_METADATA_COLUMNS,
                    dtype=object,
                )
            case DataModality.MULTIVARIATE_REGRESSION:
                self._measurement_metadata = pd.DataFrame(
                    {c: pd.Series([], dtype=t) for c, t in self.PREPROCESSING_METADATA_COLUMNS.items()},
                    index=pd.Index([], name=self.name),
                )
            case _:
                raise ValueError(f"Can't add metadata to a {self.modality} measure!")

    def add_missing_mandatory_metadata_cols(self):
        if not self.is_numeric:
            raise ValueError("Only numeric measurements can have measurement metadata")
        match self.measurement_metadata:
            case None:
                self.add_empty_metadata()

            case pd.DataFrame():
                for col, dtype in self.PREPROCESSING_METADATA_COLUMNS.items():
                    if col not in self.measurement_metadata.columns:
                        self.measurement_metadata[col] = pd.Series(
                            [None] * len(self.measurement_metadata), dtype=dtype
                        )
                if self.measurement_metadata.index.names == [None]:
                    self.measurement_metadata.index.names = [self.name]
            case pd.Series():
                for col, dtype in self.PREPROCESSING_METADATA_COLUMNS.items():
                    if col not in self.measurement_metadata.index:
                        self.measurement_metadata[col] = None

    def to_dict(self) -> dict:
        """Represents this configuration object as a plain dictionary."""
        as_dict = dataclasses.asdict(self)
        match self._measurement_metadata:
            case pd.DataFrame():
                as_dict["_measurement_metadata"] = self.measurement_metadata.to_dict(orient="tight")
            case pd.Series():
                as_dict["_measurement_metadata"] = self.measurement_metadata.to_dict(into=OrderedDict)
            case Path():
                as_dict["_measurement_metadata"] = str(self._measurement_metadata)
        if self.temporality == TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
            as_dict["functor"] = self.functor.to_dict()
        return as_dict

    @classmethod
    def from_dict(cls, as_dict: dict) -> MeasurementConfig:
        """Build a configuration object from a plain dictionary representation."""
        if as_dict["vocabulary"] is not None:
            as_dict["vocabulary"] = Vocabulary(**as_dict["vocabulary"])

        match as_dict["_measurement_metadata"], as_dict["modality"]:
            case str() | None, _:
                pass
            case dict(), DataModality.MULTIVARIATE_REGRESSION:
                as_dict["_measurement_metadata"] = pd.DataFrame.from_dict(
                    as_dict["_measurement_metadata"], orient="tight"
                )
            case dict(), DataModality.UNIVARIATE_REGRESSION:
                as_dict["_measurement_metadata"] = pd.Series(as_dict["_measurement_metadata"])
            case _:
                raise ValueError(f"{as_dict['measurement_metadata']} and {as_dict['modality']} incompatible!")

        if as_dict["functor"] is not None:
            if as_dict["temporality"] != TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                raise ValueError(
                    "Only TemporalityType.FUNCTIONAL_TIME_DEPENDENT measures can have functors. Got "
                    f"{as_dict['temporality']}"
                )
            as_dict["functor"] = cls.FUNCTORS[as_dict["functor"]["class"]].from_dict(as_dict["functor"])

        return cls(**as_dict)

    def __eq__(self, other: MeasurementConfig) -> bool:
        return self.to_dict() == other.to_dict()

    def describe(
        self, line_width: int = 60, wrap_lines: bool = False, stream: TextIOBase | None = None
    ) -> int | None:
        """Provides a plain-text description of the measurement.

        Prints the following information about the MeasurementConfig object:

        1. The measurement's name, temporality, modality, and observation frequency.
        2. What value types (e.g., integral, float, etc.) it's values take on, if the measurement is a
           numerical modality whose values may take on distinct value types.
        3. Details about its internal `self.vocabulary` object, via `Vocabulary.describe`.

        Args:
            line_width: The maximum width of each line in the description.
            wrap_lines: Whether to wrap lines that exceed the `line_width`.
            stream: The stream to write the description to. If `None`, the description is printed to stdout.

        Returns:
            The number of characters written to the stream if a stream was provided, otherwise `None`.

        Raises:
            ValueError: if the calling object is misconfigured.

        Examples:
            >>> vocab = Vocabulary(
            ...     vocabulary=['apple', 'banana', 'pear', 'UNK'],
            ...     obs_frequencies=[3, 4, 1, 2],
            ... )
            >>> cfg = MeasurementConfig(
            ...     name="MVR",
            ...     values_column='bar',
            ...     temporality='dynamic',
            ...     modality='multivariate_regression',
            ...     observation_frequency=0.6816,
            ...     _measurement_metadata=pd.DataFrame(
            ...         {'value_type': ['float', 'categorical', 'categorical']},
            ...         index=pd.Index(['apple', 'pear', 'banana'], name='MVR'),
            ...     ),
            ...     vocabulary=vocab,
            ... )
            >>> cfg.describe()
            MVR: dynamic, multivariate_regression observed 68.2%
            Value Types:
              2 categorical
              1 float
            Vocabulary:
              4 elements, 20.0% UNKs
              Frequencies: █▆▁
              Elements:
                (40.0%) banana
                (30.0%) apple
                (10.0%) pear
            >>> cfg.modality = 'wrong'
            >>> cfg.describe()
            Traceback (most recent call last):
                ...
            ValueError: Can't describe wrong measure MVR!
        """
        lines = []
        lines.append(
            f"{self.name}: {self.temporality}, {self.modality} observed {100*self.observation_frequency:.1f}%"
        )

        match self.modality:
            case DataModality.UNIVARIATE_REGRESSION:
                lines.append(f"Value is a {self.measurement_metadata.value_type}")
            case DataModality.MULTIVARIATE_REGRESSION:
                lines.append("Value Types:")
                for t, cnt in self.measurement_metadata.value_type.value_counts().items():
                    lines.append(f"  {cnt} {t}")
            case DataModality.MULTI_LABEL_CLASSIFICATION:
                pass
            case DataModality.SINGLE_LABEL_CLASSIFICATION:
                pass
            case _:
                raise ValueError(f"Can't describe {self.modality} measure {self.name}!")

        if self.vocabulary is not None:
            SIO = StringIO()
            self.vocabulary.describe(line_width=line_width - 2, stream=SIO, wrap_lines=wrap_lines)
            lines.append("Vocabulary:")
            lines.extend(f"  {line}" for line in SIO.getvalue().split("\n"))

        line_indents = [num_initial_spaces(line) for line in lines]
        if wrap_lines:
            lines = [
                wrap(line, width=line_width, initial_indent="", subsequent_indent=(" " * ind))
                for line, ind in zip(lines, line_indents)
            ]
        else:
            lines = [
                shorten(line, width=line_width, initial_indent=(" " * ind))
                for line, ind in zip(lines, line_indents)
            ]

        desc = "\n".join(lines)
        if stream is None:
            print(desc)
            return
        return stream.write(desc)


@dataclasses.dataclass
class DatasetConfig(JSONableMixin):
    """Configuration options for a Dataset class.

    This is the core configuration object for Dataset objects. Contains configuration options for
    pre-processing a dataset already in the "Subject-Events-Measurements" data model or interpreting an
    existing dataset. This configures details such as

    1. Which measurements should be extracted and included in the raw dataset, via the `measurement_configs`
       arg.
    2. What filtering parameters should be applied to eliminate infrequently observed variables or columns.
    3. How/whether or not numerical values should be re-cast as categorical or integral types.
    4. Configuration options for outlier detector or normalization models.
    5. Time aggregation controls.
    6. The output save directory.

    These configuration options do not include options to extract the raw dataset from source. For options for
    raw dataset extraction, see `DatasetSchema` and `InputDFSchema`, and for options for the raw script
    builder, see `configs/dataset_base.yml`.

    Attributes:
        measurement_configs: The dataset configuration for this `Dataset`. Keys are measurement names, and
            values are `MeasurementConfig` objects detailing configuration parameters for that measure.
            Measurement names / dictionary keys are also used as source columns for the data of that measure,
            though in the case of `DataModality.MULTIVARIATE_REGRESSION` measures, this name will reference
            the categorical regression target index column and the config will also contain a reference to a
            values column name which points to the column containing the associated numerical values.
            Columns not referenced in any configs are not pre-processed. Measurement configs are checked for
            validity upon creation. Dictionary keys must match measurement config object names if such are
            specified; if measurement config object names are not specified, they will be set to their
            associated dictionary keys.

        min_valid_column_observations: The minimum number of column observations or proportion of possible
            events that contain a column that must be observed for the column to be included in the training
            set. If fewer than this many observations are observed, the entire column will be dropped. Can be
            either an integer count or a proportion (of total vocabulary size) in (0, 1). If `None`, no
            constraint is applied.

        min_valid_vocab_element_observations: The minimum number or proportion of observations of a particular
            metadata vocabulary element that must be observed for the element to be included in the training
            set vocabulary. If fewer than this many observations are observed, observed elements will be
            dropped. Can be either an integer count or a proportion (of total vocabulary size) in (0, 1). If
            `None`, no constraint is applied.

        min_true_float_frequency: The minimum proportion of true float values that must be observed in order
            for observations to be treated as true floating point numbers, not integers.

        min_unique_numerical_observations: The minimum number of unique values a numerical column must have in
            the training set to be treated as a numerical type (rather than an implied categorical or ordinal
            type). Numerical entries with fewer than this many observations will be converted to categorical
            or ordinal types. Can be either an integer count or a proportion (of total numerical observations)
            in (0, 1). If `None`, no constraint is applied.

        outlier_detector_config: Configuration options for outlier detection. If not `None`, must contain the
            key `'cls'`, which points to the class used outlier detection. All other keys and values are
            keyword arguments to be passed to the specified class. The API of these objects is expected to
            mirror scikit-learn outlier detection model APIs. If `None`, numerical outlier values are not
            removed.

        normalizer_config: Configuration options for normalization. If not `None`, must contain the key
            `'cls'`, which points to the class used normalization. All other keys and values are keyword
            arguments to be passed to the specified class. The API of these objects is expected to mirror
            scikit-learn normalization system APIs. If `None`, numerical values are not normalized.

        save_dir: The output save directory for this dataset. Will be converted to a `pathlib.Path` upon
            creation if it is not already one.

        agg_by_time_scale: Aggregate events into temporal buckets at this frequency. Uses the string language
            described here:
            https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.groupby_dynamic.html

    Raises:
        ValueError: If configuration parameters are invalid (e.g., proportion parameters being > 1, etc.).
        TypeError: If configuration parameters are of invalid types.

    Examples:
        >>> cfg = DatasetConfig(
        ...     measurement_configs={
        ...         "meas1": MeasurementConfig(
        ...             temporality=TemporalityType.DYNAMIC,
        ...             modality=DataModality.MULTI_LABEL_CLASSIFICATION,
        ...         ),
        ...     },
        ...     min_valid_column_observations=0.5,
        ...     save_dir="/path/to/save/dir",
        ... )
        >>> cfg.save_dir
        PosixPath('/path/to/save/dir')
        >>> cfg.to_dict()
        {'measurement_configs': {'meas1': {'name': 'meas1', 'temporality': <TemporalityType.DYNAMIC:\
 'dynamic'>, 'modality': <DataModality.MULTI_LABEL_CLASSIFICATION: 'multi_label_classification'>,\
 'observation_frequency': None, 'functor': None, 'vocabulary': None, 'values_column': None,\
 '_measurement_metadata': None}}, 'min_events_per_subject': None, 'agg_by_time_scale': '1h',\
 'min_valid_column_observations': 0.5, 'min_valid_vocab_element_observations': None,\
 'min_true_float_frequency': None, 'min_unique_numerical_observations': None,\
 'outlier_detector_config': None, 'normalizer_config': None, 'save_dir': '/path/to/save/dir'}
        >>> cfg2 = DatasetConfig.from_dict(cfg.to_dict())
        >>> assert cfg == cfg2
        >>> DatasetConfig(
        ...     measurement_configs={
        ...         "meas1": MeasurementConfig(
        ...             name="invalid_name",
        ...             temporality=TemporalityType.DYNAMIC,
        ...             modality=DataModality.MULTI_LABEL_CLASSIFICATION,
        ...         ),
        ...     },
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Measurement config meas1 has name invalid_name which differs from dict key!
        >>> DatasetConfig(
        ...     min_valid_column_observations="invalid type"
        ... )
        Traceback (most recent call last):
            ...
        TypeError: min_valid_column_observations must either be a fraction (float between 0 and 1) or count\
 (int > 1). Got <class 'str'> of invalid type
        >>> measurement_configs = {
        ...     "meas1": MeasurementConfig(
        ...         temporality=TemporalityType.DYNAMIC,
        ...         modality=DataModality.MULTI_LABEL_CLASSIFICATION,
        ...     ),
        ... }
        >>> # Make one of the measurements invalid to show that validitiy is re-checked...
        >>> measurement_configs["meas1"].temporality = None
        >>> DatasetConfig(
        ...     measurement_configs=measurement_configs,
        ...     min_valid_column_observations=0.5,
        ...     save_dir="/path/to/save/dir",
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Measurement config meas1 invalid!
    """

    measurement_configs: dict[str, MeasurementConfig] = dataclasses.field(default_factory=lambda: {})

    min_events_per_subject: int | None = None

    agg_by_time_scale: str | None = "1h"

    min_valid_column_observations: COUNT_OR_PROPORTION | None = None
    min_valid_vocab_element_observations: COUNT_OR_PROPORTION | None = None
    min_true_float_frequency: PROPORTION | None = None
    min_unique_numerical_observations: COUNT_OR_PROPORTION | None = None

    outlier_detector_config: dict[str, Any] | None = None
    normalizer_config: dict[str, Any] | None = None

    save_dir: Path | None = None

    def __post_init__(self):
        """Validates that parameters take on valid values."""
        for name, cfg in self.measurement_configs.items():
            if cfg.name is None:
                cfg.name = name
            elif cfg.name != name:
                raise ValueError(
                    f"Measurement config {name} has name {cfg.name} which differs from dict key!"
                )

        for var in (
            "min_valid_column_observations",
            "min_valid_vocab_element_observations",
            "min_unique_numerical_observations",
        ):
            val = getattr(self, var)
            match val:
                case None:
                    pass
                case float() if (0 < val) and (val < 1):
                    pass
                case int() if val > 1:
                    pass
                case float():
                    raise ValueError(f"{var} must be in (0, 1) if float; got {val}!")
                case int():
                    raise ValueError(f"{var} must be > 1 if integral; got {val}!")
                case _:
                    raise TypeError(
                        f"{var} must either be a fraction (float between 0 and 1) or count (int > 1). Got "
                        f"{type(val)} of {val}"
                    )

        for var in ("min_true_float_frequency",):
            val = getattr(self, var)
            match val:
                case None:
                    pass
                case float() if (0 < val) and (val < 1):
                    pass
                case float():
                    raise ValueError(f"{var} must be in (0, 1) if float; got {val}!")
                case _:
                    raise TypeError(
                        f"{var} must be a fraction (float between 0 and 1). Got {type(val)} of {val}"
                    )

        for var in ("outlier_detector_config", "normalizer_config"):
            val = getattr(self, var)
            if val is not None and (type(val) is not dict or "cls" not in val):
                raise ValueError(f"{var} must be either None or a dictionary with 'cls' as a key! Got {val}")

        for k, v in self.measurement_configs.items():
            try:
                v._validate()
            except Exception as e:
                raise ValueError(f"Measurement config {k} invalid!") from e

        if type(self.save_dir) is str:
            self.save_dir = Path(self.save_dir)

    def to_dict(self) -> dict:
        """Represents this configuration object as a plain dictionary.

        Returns:
            A plain dictionary representation of self (nested through measurement configs as well).
        """
        as_dict = dataclasses.asdict(self)
        if self.save_dir is not None:
            as_dict["save_dir"] = str(self.save_dir.absolute())
        as_dict["measurement_configs"] = {k: v.to_dict() for k, v in self.measurement_configs.items()}
        return as_dict

    @classmethod
    def from_dict(cls, as_dict: dict) -> DatasetConfig:
        """Build a configuration object from a plain dictionary representation.

        Args:
            as_dict: The plain dictionary representation to be converted.

        Returns: A DatasetConfig instance containing the same data as `as_dict`.
        """
        as_dict["measurement_configs"] = {
            k: MeasurementConfig.from_dict(v) for k, v in as_dict["measurement_configs"].items()
        }
        if type(as_dict["save_dir"]) is str:
            as_dict["save_dir"] = Path(as_dict["save_dir"])

        return cls(**as_dict)

    def __eq__(self, other: DatasetConfig) -> bool:
        """Returns true if self and other are equal."""
        return self.to_dict() == other.to_dict()

from __future__ import annotations

import dataclasses
import random
from collections import OrderedDict, defaultdict
from collections.abc import Hashable, Sequence
from io import StringIO, TextIOBase
from pathlib import Path
from textwrap import shorten, wrap
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import omegaconf
import pandas as pd

from ..utils import (
    COUNT_OR_PROPORTION,
    PROPORTION,
    JSONableMixin,
    hydra_dataclass,
    num_initial_spaces,
)
from .time_dependent_functor import AgeFunctor, TimeDependentFunctor, TimeOfDayFunctor
from .types import DataModality, InputDataType, InputDFType, TemporalityType
from .vocabulary import Vocabulary

DF_COL = Union[str, Sequence[str]]

INPUT_COL_T = Union[InputDataType, tuple[InputDataType, str]]

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
    static: dict[str, Any] | InputDFSchema | None = None
    dynamic: list[InputDFSchema | dict[str, Any]] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.static is None:
            raise ValueError("Must specify a static schema!")

        if type(self.static) is dict:
            self.static = InputDFSchema(**self.static)
            if not self.static.is_static:
                raise TypeError("Must pass a static schema config for static.")

        if self.static.subject_id_col is None:
            raise ValueError("Must specify a subject_id_col source for the static dataframe!")

        if self.dynamic is not None:
            new_dynamic = []
            for v in self.dynamic:
                if type(v) is dict:
                    v = InputDFSchema(**v)
                if v.subject_id_col is None:
                    v.subject_id_col = self.static.subject_id_col

                if v.subject_id_col != self.static.subject_id_col:
                    print(
                        f"WARNING: {v.input_df} subject ID col name ({v.subject_id_col}) differs from static "
                        f"({self.static.subject_id_col})."
                    )

                new_dynamic.append(v)

                if v.is_static:
                    raise TypeError("Must pass dynamic schemas in `self.dynamic`!")
            self.dynamic = new_dynamic

        self.dynamic_by_df = defaultdict(list)
        for v in self.dynamic:
            self.dynamic_by_df[v.input_df].append(v)
        self.dynamic_by_df = {k: v for k, v in self.dynamic_by_df.items()}


@dataclasses.dataclass
class InputDFSchema(JSONableMixin):
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
                        raise TypeError(
                            f"event_type must be a string for events. Got {self.event_type}"
                        )
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
                        raise ValueError(
                            f"{param} should be None for {self.type} schema: Got {val}"
                        )

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
                    raise ValueError(
                        f"ts_col should be `None` for {self.type} schemas! Got: {self.ts_col}."
                    )
                if self.subject_id_col is not None:
                    raise ValueError("subject_id_col should be None for non-static types!")
                if self.start_ts_format is not None:
                    if self.end_ts_format is None:
                        raise ValueError(
                            "If start_ts_format is specified, end_ts_format must also be specified!"
                        )
                    if self.ts_format is not None:
                        raise ValueError(
                            "If start_ts_format is specified, ts_format must be `None`!"
                        )
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

    @property
    def columns_to_load(self) -> list[tuple[str, InputDataType]]:
        columns_to_load = {}

        for in_col, (out_col, dt) in self.unified_schema.items():
            if in_col in columns_to_load:
                raise ValueError(f"Duplicate column {in_col}!")
            columns_to_load[in_col] = dt

        if self.type == InputDFType.RANGE:
            for in_col, (out_col, dt) in self.unified_start_schema.items():
                if in_col in columns_to_load:
                    if dt != columns_to_load[in_col]:
                        raise ValueError(f"Duplicate column {in_col} with differing dts!")
                else:
                    columns_to_load[in_col] = dt
            for in_col, (out_col, dt) in self.unified_end_schema.items():
                if in_col in columns_to_load:
                    if dt != columns_to_load[in_col]:
                        raise ValueError(f"Duplicate column {in_col} with differing dts!")
                else:
                    columns_to_load[in_col] = dt

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
        elif in_col in container:
            raise ValueError(
                f"Column {in_col} is repeated in schema!\n"
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
                case (str() as col, (InputDataType() | (InputDataType(), str())) as dt):
                    cls.__add_to_schema(unified_schema, in_col=col, dt=dt)
                case (list() as cols, (InputDataType() | (InputDataType(), str())) as dt):
                    for c in cols:
                        cls.__add_to_schema(unified_schema, in_col=c, dt=dt)
                case dict():
                    for in_col, schema_info in schema.items():
                        match schema_info:
                            case (out_col, (InputDataType() | (InputDataType(), str())) as dt):
                                cls.__add_to_schema(
                                    unified_schema, in_col=in_col, dt=dt, out_col=out_col
                                )
                            case (InputDataType() | (InputDataType(), str())) as dt:
                                cls.__add_to_schema(unified_schema, in_col=in_col, dt=dt)
                            case _:
                                raise ValueError(f"Schema Unprocessable!\n{schema_info}")
                case (dict() as col_names_map, (InputDataType() | (InputDataType(), str())) as dt):
                    for in_col, out_col in col_names_map.items():
                        cls.__add_to_schema(unified_schema, in_col=in_col, dt=dt, out_col=out_col)
                case _:
                    raise ValueError(f"Schema Unprocessable!\n{schema}")

        return unified_schema


@dataclasses.dataclass
class VocabularyConfig(JSONableMixin):
    vocab_sizes_by_measurement: dict[str, int] | None = None
    vocab_offsets_by_measurement: dict[str, int] | None = None
    measurements_idxmap: dict[str, dict[Hashable, int]] | None = None
    measurements_per_generative_mode: dict[DataModality, list[str]] | None = None
    event_types_per_measurement: dict[str, list[str]] | None = None
    event_types_idxmap: dict[str, int] | None = None

    @property
    def total_vocab_size(self) -> int:
        return (
            sum(self.vocab_sizes_by_measurement.values())
            + min(self.vocab_offsets_by_measurement.values())
            + (len(self.vocab_offsets_by_measurement) - len(self.vocab_sizes_by_measurement))
        )


@hydra_dataclass
class PytorchDatasetConfig(JSONableMixin):
    """Configuration options for building a PyTorch dataset from an `Dataset`.

    Args:
        `max_seq_len` (`int`):
            Captures the maximum sequence length the pytorch dataset should output in any individual item.
            Note that batche are _not_ universally normalized to have this sequence length --- it is a
            maximum, so individual batches can have shorter sequence lengths in practice.
        `min_seq_len` (`int`):
            Only include subjects with at least this many events in the raw data.
        `seq_padding_side` (`str`, defaults to `'right'`):
            Whether to pad smaller sequences on the right (default) or the left (used for generation).
        `do_produce_static_data` (`bool`):
            Whether or not to produce static data when processing the dataset.
    """

    save_dir: Path = omegaconf.MISSING

    max_seq_len: int = 256
    min_seq_len: int = 2
    seq_padding_side: str = "right"

    train_subset_size: int | str = "FULL"
    train_subset_seed: int | None = None

    task_df_name: str | None = None

    def __post_init__(self):
        assert self.seq_padding_side in ("left", "right")
        assert self.min_seq_len >= 0
        assert self.max_seq_len >= 1
        assert self.max_seq_len >= self.min_seq_len

        if type(self.save_dir) is str and self.save_dir != omegaconf.MISSING:
            self.save_dir = Path(self.save_dir)

        match self.train_subset_size:
            case (None | "FULL") if self.train_subset_seed is not None:
                raise ValueError(
                    f"train_subset_seed {self.train_subset_seed} should be None "
                    "if train_subset_size is {self.train_subset_size}."
                )
            case int() as n if n < 0:
                raise ValueError(f"If integral, train_subset_size must be positive! Got {n}")
            case float() as frac if frac <= 0 or frac >= 1:
                raise ValueError(f"If float, train_subset_size must be in (0, 1)! Got {frac}")
            case int() | float() if (self.train_subset_seed is None):
                seed = int(random.randint(1, int(1e6)))
                print(
                    f"WARNING! train_subset_size is set, but train_subset_seed is not. Setting to {seed}"
                )
                self.train_subset_seed = seed
            case None | "FULL" | int() | float():
                pass
            case _:
                raise TypeError(
                    f"train_subset_size is of unrecognized type {type(self.train_subset_size)}."
                )

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

    """
    Base configuration class for a measurement in the Dataset.
    A measurement is any observation in the dataset; be it static or dynamic, categorical or continuous. This
    class contains configuration options to define a measurement and dictate how it should be pre-processed,
    embedded, and generated in generative models.

    Args:
        # Present in all measures
        `name` (`Optional[str]`, defaults to `None`):
            Stores the name of this measurement; also the column that contains this measurement.
            The 'column' linkage has slightly different meanings depending on `self.modality`:
                * In the case that `modality == DataModality.UNIVARIATE_REGRESSION`, then this column
                  stores the values associated with this continuous-valued measure.
                * In the case that `modality == DataModality.MULTIVARIATE_REGRESSION`, then this column stores
                  the keys that dictate the dimensions for which the associated `values_column` has the
                  values.
                * In the case that `modality` is neither of the above two options, then this column stores the
                  categorical values of this measure.
            Similarly, it has slightly different meanings depending on `self.temporality`:
                * In the case that `temporality == TemporalityType.STATIC`, this is an existent column in the
                  `subjects_df` dataframe.
                * In the case that `temporality == TemporalityType.DYNAMIC`, this is an existent column in the
                  `joint_metadata_df` dataframe.
                * In the case that `temporality == TemporalityType.FUNCTIONAL_TIME_DEPENDENT`, then this is
                  the name the _output, to-be-created_ column will take in the `events_df` dataframe.
        `modality` (`Optional[DataModality]`, defaults to `None`):
            What modality the values of this measure are.
        `temporality` (`Optional[TemporalityType]`, defaults to `None`):
            How this measure varies in time.
        `observation_frequency` (`Optional[float]`, defaults to `None`):
            The fraction of valid instances in which this measure is observed. Is set dynamically during
            pre-procesisng, and not specified at construction.

        # Specific to dynamic measures
        `present_in_event_types` (`Optional[Set[str]]`, defaults to `None`):
            Within which event types this column can be present.
            If `None`, this column can be present in *all* event types.

        # Specific to time-dependent measures
        `functor` (`Optional[TimeDependentFunctor]`, defaults to `None`):
            The functor used to compute the value of a known-time-depedency measure (e.g., Age). Must be None
            if measure is not a known-time-dependency measure.
            The vocabulary for this column. Begins with 'UNK'.

        # Specific to categorical or partially observed multivariate regression measures.
        `vocabulary` (`Optional[Vocabulary]`, defaults to `None`):

        # Specific to numeric measures
        `values_column` (`Optional[str]`, defaults to `None`):
            Which column stores the numerical values corresponding to this measurement. If `None`, then
            measurement values are stored in the same column.

        `measurement_metadata` (`Optional[pd.DataFrame]`, *optional*, defaults to `None`):
            Stores metadata (dataframe columns) about the numerical values corresponding to each
            key (dataframe index). Metadata columns must include the following, which are sentinel values in
            preprocessing steps:
                * `unit`: The unit of measure of this key.
                * `drop_lower_bound`:
                    A lower bound such that values either below or at or below this level will be dropped
                    (key presence will be retained).
                * `drop_lower_bound_inclusive`:
                    Is the drop lower bound inclusive or exclusive?
                * `censor_lower_bound`:
                    A lower bound such that values either below or at or below this level, but above the
                    level of `drop_lower_bound`, will be replaced with the value of `censor_lower_bound`.
                * `drop_upper_bound`:
                    An upper bound such that values either above or at or above this level will be dropped
                    (key presence will be retained).
                * `drop_upper_bound_inclusive`:
                    Is the drop upper bound inclusive or exclusive?
                * `censor_upper_bound`:
                    An upper bound such that values either above or at or above this level, but below the
                    level of `drop_upper_bound`, will be replaced with the value of `censor_upper_bound`.
                * `value_type`:
                    To which kind of value this key corresponds. Must be an element of the enum
                    `NumericMetadataValueType`
                * `outlier_model`: The fit outlier model associated with this key.
                * `normalizer`: The fit normalizer model associated with this key.
    """

    # Present in all measures
    name: str | None = None
    temporality: TemporalityType | None = None
    modality: DataModality | None = None
    observation_frequency: float | None = None

    # Specific to dynamic measures
    present_in_event_types: set[str] | None = None

    # Specific to time-dependent measures
    functor: TimeDependentFunctor | None = None

    # Specific to categorical or partially observed multivariate regression measures.
    vocabulary: Vocabulary | None = None

    # Specific to numeric measures
    values_column: str | None = None
    measurement_metadata: pd.DataFrame | pd.Series | None = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Checks the internal state of `self` and ensures internal consistency and validity."""
        match self.temporality:
            case TemporalityType.STATIC:
                assert self.present_in_event_types is None
                assert self.functor is None

                if self.is_numeric:
                    raise NotImplementedError(
                        f"Numeric data modalities like {self.modality} not yet supported on static measures."
                    )
            case TemporalityType.DYNAMIC:
                assert self.functor is None

            case TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                assert self.functor is not None
                assert self.present_in_event_types is None

                if self.modality is None:
                    self.modality = self.functor.OUTPUT_MODALITY
                else:
                    assert self.modality in (DataModality.DROPPED, self.functor.OUTPUT_MODALITY)

            case _:
                raise ValueError(f"`self.temporality = {self.temporality}` Invalid!")

        match self.modality:
            case DataModality.MULTIVARIATE_REGRESSION:
                assert self.values_column is not None
                if self.measurement_metadata is not None:
                    assert type(self.measurement_metadata) is pd.DataFrame
            case DataModality.UNIVARIATE_REGRESSION:
                assert self.values_column is None
                if self.measurement_metadata is not None:
                    assert type(self.measurement_metadata) is pd.Series
            case DataModality.SINGLE_LABEL_CLASSIFICATION | DataModality.MULTI_LABEL_CLASSIFICATION:
                assert self.measurement_metadata is None
                assert self.values_column is None
            case DataModality.DROPPED:
                assert self.measurement_metadata is None
                assert self.vocabulary is None
            case _:
                raise ValueError(f"`self.modality = {self.modality}` Invalid!")

    def drop(self):
        """Sets the modality to DROPPED and does associated post-processing to ensure validity."""
        self.modality = DataModality.DROPPED
        self.measurement_metadata = None
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

    def add_empty_metadata(self):
        """Adds an empty `measurement_metadata` dataframe or series."""
        assert self.measurement_metadata is None

        match self.modality:
            case DataModality.UNIVARIATE_REGRESSION:
                self.measurement_metadata = pd.Series(
                    [None] * len(self.PREPROCESSING_METADATA_COLUMNS),
                    index=self.PREPROCESSING_METADATA_COLUMNS,
                    dtype=object,
                )
            case DataModality.MULTIVARIATE_REGRESSION:
                self.measurement_metadata = pd.DataFrame(
                    {
                        c: pd.Series([], dtype=t)
                        for c, t in self.PREPROCESSING_METADATA_COLUMNS.items()
                    },
                    index=pd.Index([], name=self.name),
                )
            case _:
                raise ValueError(f"Can't add metadata to a {self.modality} measure!")

    def add_missing_mandatory_metadata_cols(self):
        assert self.is_numeric
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
        match self.measurement_metadata:
            case pd.DataFrame():
                as_dict["measurement_metadata"] = self.measurement_metadata.to_dict(orient="tight")
            case pd.Series():
                as_dict["measurement_metadata"] = self.measurement_metadata.to_dict(
                    into=OrderedDict
                )
        if self.temporality == TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
            as_dict["functor"] = self.functor.to_dict()
        if self.present_in_event_types is not None:
            as_dict["present_in_event_types"] = list(self.present_in_event_types)
        return as_dict

    @classmethod
    def from_dict(cls, as_dict: dict) -> MeasurementConfig:
        """Build a configuration object from a plain dictionary representation."""
        if as_dict["vocabulary"] is not None:
            as_dict["vocabulary"] = Vocabulary(**as_dict["vocabulary"])

        if as_dict["measurement_metadata"] is not None:
            match as_dict["modality"]:
                case DataModality.MULTIVARIATE_REGRESSION:
                    as_dict["measurement_metadata"] = pd.DataFrame.from_dict(
                        as_dict["measurement_metadata"], orient="tight"
                    )
                case DataModality.UNIVARIATE_REGRESSION:
                    as_dict["measurement_metadata"] = pd.Series(as_dict["measurement_metadata"])
                case _:
                    raise ValueError(
                        "Config is non-numeric but has a measurement_metadata value observed."
                    )

        if as_dict["functor"] is not None:
            assert as_dict["temporality"] == TemporalityType.FUNCTIONAL_TIME_DEPENDENT
            as_dict["functor"] = cls.FUNCTORS[as_dict["functor"]["class"]].from_dict(
                as_dict["functor"]
            )

        if as_dict["present_in_event_types"] is not None:
            as_dict["present_in_event_types"] = set(as_dict["present_in_event_types"])

        return cls(**as_dict)

    def __eq__(self, other: MeasurementConfig) -> bool:
        return self.to_dict() == other.to_dict()

    def describe(
        self, line_width: int = 60, wrap_lines: bool = False, stream: TextIOBase | None = None
    ) -> int | None:
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
    """Configuration options for parsing an `Dataset`.

    Args:
        `measurement_configs` (`Dict[str, MeasurementConfig]`, defaults to `{}`):
            The dataset configuration for this `Dataset`. Keys are measurement names, and
            values are `MeasurementConfig` objects detailing configuration parameters for that measure.
            Measurement configs may point to other columns as well, as in the case of key-value processed
            multivariate, partially observed regression values.
            Columns not referenced in any configs are not pre-processed.

        `min_valid_column_observations` (`Optional[COUNT_OR_PROPORTION]`, defaults to `None`):
            The minimum number of column observations or proportion of possible events that contain a column
            that must be observed for the column to be included in the training set. If fewer than this
            many observations are observed, the entire column will be dropped.
            Can be either an integer count or a proportion (of total vocabulary size) in (0, 1).
            If `None`, no constraint is applied.

        `min_valid_vocab_element_observations` (`Optional[COUNT_OR_PROPORTION]`, defaults to `None`):
            The minimum number or proportion of observations of a particular metadata vocabulary element that
            must be observed for the element to be included in the training set vocabulary. If fewer than this
            many observations are observed, observed elements will be dropped.
            Can be either an integer count or a proportion (of total vocabulary size) in (0, 1).
            If `None`, no constraint is applied.

        `min_true_float_frequency` (`Optional[PROPORTION]`, defaults to `None`):
            The minimum proportion of true float values that must be observed in order for observations to be
            treated as true floating point numbers, not integers.

        `min_unique_numerical_observations` (`Optional[COUNT_OR_PROPORTION]`, defaults to `None`):
            The minimum number of unique values a numerical column must have in the training set to
            be treated as a numerical type (rather than an implied categorical or ordinal type). Numerical
            entries with fewer than this many observations will be converted to categorical or ordinal types.
            Can be either an integer count or a proportion (of total numerical observations) in (0, 1).
            If `None`, no constraint is applied.

        `outlier_detector_config` (`Optional[Dict[str, Any]]`, defaults to `None`):
            Configuration options for outlier detection. If not `None`, must contain the key `'cls'`, which
            points to the class used outlier detection. All other keys and values are keyword arguments to be
            passed to the specified class. The API of these objects is expected to mirror scikit-learn outlier
            detection model APIs.
            If `None`, numerical outlier values are not removed.

        `normalizer_config` (`Optional[Dict[str, Any]]`, defaults to `None`):
            Configuration options for normalization. If not `None`, must contain the key `'cls'`, which points
            to the class used normalization. All other keys and values are keyword arguments to be passed to
            the specified class. The API of these objects is expected to mirror scikit-learn normalization
            system APIs.
            If `None`, numerical values are not normalized.

        `save_dir`

        `agg_by_time_scale` (`Optional[str]`...
            Uses the string language described here:
            https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.groupby_dynamic.html
    """

    measurement_configs: dict[str, MeasurementConfig] = dataclasses.field(
        default_factory=lambda: {}
    )

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
            else:
                assert cfg.name == name

        for var in (
            "min_valid_column_observations",
            "min_valid_vocab_element_observations",
            "min_unique_numerical_observations",
        ):
            val = getattr(self, var)
            if val is not None:
                assert ((type(val) is float) and (0 < val) and (val < 1)) or (
                    (type(val) is int) and (val > 1)
                )

        for var in ("min_true_float_frequency",):
            val = getattr(self, var)
            if val is not None:
                assert type(val) is float and (0 < val) and (val < 1)

        for var in ("outlier_detector_config", "normalizer_config"):
            val = getattr(self, var)
            if val is not None:
                assert type(val) is dict and "cls" in val

        for k, v in self.measurement_configs.items():
            try:
                v._validate()
            except Exception as e:
                raise ValueError(f"Measurement config {k} invalid!") from e

        if type(self.save_dir) is str:
            self.save_dir = Path(self.save_dir)

    def to_dict(self) -> dict:
        """Represents this configuration object as a plain dictionary."""
        as_dict = dataclasses.asdict(self)
        as_dict["measurement_configs"] = {
            k: v.to_dict() for k, v in self.measurement_configs.items()
        }
        return as_dict

    @classmethod
    def from_dict(cls, as_dict: dict) -> DatasetConfig:
        """Build a configuration object from a plain dictionary representation."""
        as_dict["measurement_configs"] = {
            k: MeasurementConfig.from_dict(v) for k, v in as_dict["measurement_configs"].items()
        }

        return cls(**as_dict)

    @classmethod
    def from_simple_args(
        cls,
        dynamic_measurement_columns: Sequence[str | tuple[str, str]] | None = None,
        static_measurement_columns: Sequence[str] | None = None,
        time_dependent_measurement_columns: None
        | (Sequence[tuple[str, TimeDependentFunctor]]) = None,
        **kwargs,
    ) -> DatasetConfig:
        """Builds an appropriate configuration object given a simple list of columns:

        Args:
            `dynamic_measurement_columns`
            (`Sequence[Union[str, Tuple[str, str]]]`, *optional*, defaults to `None`):
                A list of either multi_label_classification columns (if only one str) or
                multivariate_regression columns (if given a pair of strings, in which case the former is
                considered the main measure / key column name and the latter the values column). All produced
                measures are dynamic.

            `static_measurement_columns` (`Sequence[str]`, *optional*, defaults to `None`):
                A list of columns that will be interpreted as static, single_label_classification tasks.
                If None, no such columns will be added.

            `time_dependent_measurement_columns`
            (`Sequence[Tuple[str, TimeDependentFunctor]]`, *optional, defaults to `None`):
                A list of tuples of column names and computation functions that will define time-dependent
                functional columns. If None, no such columns will be added.

            `**kwargs`: Other keyword arguments will be passed to the `DatasetConfig` constructor.

        Returns:
            An `DatasetConfig` object with an appropriate `measurement_configs` variable set based
            on the above args, and all other passed keyword args passed as well.
        """
        measurement_configs = {}

        if dynamic_measurement_columns is not None:
            for measurement in dynamic_measurement_columns:
                col_kwargs = {"temporality": TemporalityType.DYNAMIC}
                col_name = None
                match measurement:
                    case (None, str() as col_name):
                        col_kwargs["modality"] = DataModality.UNIVARIATE_REGRESSION
                    case (str() as col_name, str() as val_col):
                        col_kwargs["modality"] = DataModality.MULTIVARIATE_REGRESSION
                        col_kwargs["values_column"] = val_col
                    case str() as col_name:
                        col_kwargs["modality"] = DataModality.MULTI_LABEL_CLASSIFICATION
                    case _:
                        raise TypeError(f"{measurement} is of incorrect type!")

                measurement_configs[col_name] = MeasurementConfig(**col_kwargs)

        if static_measurement_columns is not None:
            for measurement in static_measurement_columns:
                measurement_configs[measurement] = MeasurementConfig(
                    modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
                    temporality=TemporalityType.STATIC,
                )

        if time_dependent_measurement_columns is not None:
            for measurement, functor in time_dependent_measurement_columns:
                measurement_configs[measurement] = MeasurementConfig(
                    temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT, functor=functor
                )

        return cls(measurement_configs=measurement_configs, **kwargs)

    def __eq__(self, other: DatasetConfig) -> bool:
        return self.to_dict() == other.to_dict()

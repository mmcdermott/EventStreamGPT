from __future__ import annotations

import dataclasses, pandas as pd

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Set, Sequence, Tuple, Union

from ..utils import COUNT_OR_PROPORTION, PROPORTION, JSONableMixin
from .time_dependent_functor import AgeFunctor, TimeOfDayFunctor, TimeDependentFunctor
from .types import TemporalityType, DataModality, InputDFType, InputDataType
from .vocabulary import Vocabulary

DF_COL = Union[str, Sequence[str]]

DF_SCHEMA = Union[
    # For cases where you specify a list of columns of a constant type.
    Tuple[List[DF_COL], DataModality],
    # For specifying a dictionary of columns to types.
    Dict[DF_COL, DataModality],
    # For specifying a dictionary of column in names to column out names and types.
    Dict[DF_COL, Tuple[str, DataModality]],
    # For specifying a dictionary of column in names to out names, all of a constant type.
    Tuple[Dict[DF_COL, str], DataModality],
]

@dataclasses.dataclass
class InputDFSchema(JSONableMixin):
    type: Optional[InputDFType] = None
    event_type: Optional[str] = None

    ts_col: Optional[DF_COL] = None
    start_ts_col: Optional[DF_COL] = None
    end_ts_col: Optional[DF_COL] = None
    ts_format: Optional[str] = "%Y-%m-%d %H:%M:%S"
    start_ts_format: Optional[str] = None
    end_ts_format: Optional[str] = None

    do_make_unique: Optional[bool] = None
    data_schema: Optional[Union[DF_SCHEMA, List[DF_SCHEMA]]] = None

    def __post_init__(self):
        if self.type is None: raise ValueError("Missing mandatory parameter type!")
        if self.event_type is None: raise ValueError("Missing mandatory parameter event_type!")
        if type(self.data_schema) is not list and self.data_schema is not None:
            self.data_schema = [self.data_schema]

        self.columns_to_load = []

        match self.type:
            case InputDFType.EVENT:
                if self.ts_col is None: raise ValueError("Missing mandatory event parameter ts_col!")
                if (
                    (self.start_ts_col is not None) or (self.end_ts_col is not None) or
                    (self.start_ts_format is not None) or (self.end_ts_format is not None)
                ):
                    raise ValueError(
                        "start_ts_col, end_ts_col, start_ts_format, and end_ts_format should be `None` "
                        f"for {self.type} schemas! Got:\n"
                        f"  start_ts_col: {self.start_ts_col}\n"
                        f"  end_ts_col: {self.end_ts_col}\n"
                        f"  start_ts_format: {self.start_ts_format}\n"
                        f"  end_ts_format: {self.end_ts_format}"
                    )
                if type(self.ts_col) is list:
                    for c in self.ts_col:
                        self.columns_to_load.append((c, (InputDataType.TIMESTAMP, self.ts_format)))
                else: self.columns_to_load.append((self.ts_col, (InputDataType.TIMESTAMP, self.ts_format)))

            case InputDFType.RANGE:
                if self.start_ts_col is None:
                    raise ValueError("Missing mandatory range parameter start_ts_col!")
                if self.end_ts_col is None:
                    raise ValueError("Missing mandatory range parameter end_ts_col!")
                if self.ts_col is not None:
                    raise ValueError(f"ts_col should be `None` for {self.type} schemas! Got: {self.ts_col}.")
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
                    if self.ts_format is None:
                        raise ValueError(
                            "If start_ts_format is not specified, ts_format must be specified!"
                        )
                    self.start_ts_format = self.ts_format
                    self.end_ts_format = self.ts_format
                    self.ts_format = None

                if self.start_ts_col is list:
                    for c in self.start_ts_col:
                        self.columns_to_load.append((c, (InputDataType.TIMESTAMP, self.start_ts_format)))
                else:
                    self.columns_to_load.append((
                        self.start_ts_col, (InputDataType.TIMESTAMP, self.start_ts_format)
                    ))
                if self.end_ts_col is list:
                    for c in self.end_ts_col:
                        self.columns_to_load.append((c, (InputDataType.TIMESTAMP, self.end_ts_format)))
                else:
                    self.columns_to_load.append((
                        self.end_ts_col, (InputDataType.TIMESTAMP, self.end_ts_format)
                    ))

        self._set_unified_schema()

    def __add_to_schema_local(self, in_col: str, out_col: str, data_type: InputDataType):
        if in_col in self.unified_schema:
            raise ValueError(
                f"Column {in_col} is repeated in schema!\n"
                f"Existing: {self.unified_schema[in_col]}\n"
                f"New: ({out_col}, {data_type})"
            )
        elif type(in_col) is not str or type(out_col) is not str:
            raise ValueError(f"Column names must be strings! Got {in_col}, {out_col}")
        self.unified_schema[in_col] = (out_col, data_type)
        self.columns_to_load.append((in_col, data_type))

    def __add_to_schema(
        self, in_col: DF_COL, dt: DataModality, out_col: Optional[DF_COL] = None
    ):
        if out_col is None: out_col = in_col

        match dt:
            case DataModality.DROPPED: raise ValueError(f"Cannot specify columns for {dt}!")
            case DataModality.MULTIVARIATE_REGRESSION:
                for c in (in_col, out_col):
                    if type(c) is not tuple or len(c) != 2:
                        raise ValueError(
                            f"For {dt} columns, you must specify both a key column and a value column "
                            f"in a tuple: (key, value). Got {c}!"
                        )
                self.__add_to_schema_local(in_col[0], out_col[0], InputDataType.CATEGORICAL)
                self.__add_to_schema_local(in_col[1], out_col[1], InputDataType.FLOAT)
            case DataModality.UNIVARIATE_REGRESSION:
                self.__add_to_schema_local(in_col, out_col, InputDataType.FLOAT)
            case DataModality.SINGLE_LABEL_CLASSIFICATION | DataModality.MULTI_LABEL_CLASSIFICATION:
                self.__add_to_schema_local(in_col, out_col, InputDataType.CATEGORICAL)
            case _: raise ValueError(f"DataModality invalid! {dt}")

    def _set_unified_schema(self):
        self.unified_schema = {}
        if self.data_schema is None: return

        for schema in self.data_schema:
            match schema:
                case (list() as cols, DataModality() as dt):
                    for c in cols: self.__add_to_schema(in_col=c, dt=dt)
                case dict():
                    for in_col, schema_info in schema.items():
                        match schema_info:
                            case (out_col, DataModality() as dt):
                                self.__add_to_schema(in_col=in_col, dt=dt, out_col=out_col)
                            case DataModality() as dt:
                                self.__add_to_schema(in_col=in_col, dt=dt)
                            case _: raise ValueError(f"Schema Unprocessable!\n{schema_info}")
                case (dict() as col_names_map, DataModality() as dt):
                    for in_col, out_col in col_names_map.items():
                        self.__add_to_schema(in_col=in_col, dt=dt, out_col=out_col)
                case _:
                    raise ValueError(f"Schema Unprocessable!\n{schema}")

@dataclasses.dataclass
class VocabularyConfig(JSONableMixin):
    vocab_sizes_by_measurement: Optional[Dict[str, int]] = None
    vocab_offsets_by_measurement: Optional[Dict[str, int]] = None
    measurements_idxmap: Optional[Dict[str, Dict[Hashable, int]]] = None
    measurements_per_generative_mode: Optional[Dict[DataModality, List[str]]] = None
    event_types_per_measurement: Optional[Dict[str, List[str]]] = None
    event_types_idxmap: Optional[Dict[str, int]] = None

    @property
    def total_vocab_size(self) -> int:
        return (
            sum(self.vocab_sizes_by_measurement.values()) +
            min(self.vocab_offsets_by_measurement.values()) +
            (len(self.vocab_offsets_by_measurement) - len(self.vocab_sizes_by_measurement))
        )

@dataclasses.dataclass
class EventStreamPytorchDatasetConfig(JSONableMixin):
    """
    Configuration options for building a PyTorch dataset from an `EventStreamDataset`.

    Args:
        `do_normalize_log_inter_event_times` (`bool`):
            Captures whether or not the presented times in the batch should be transformed such that the log
            of the times between events have mean 0 and standard deviation 1.
        `max_seq_len` (`int`):
            Captures the maximum sequence length the pytorch dataset should output in any individual item.
            Note that batche are _not_ universally normalized to have this sequence length --- it is a
            maximum, so individual batches can have shorter sequence lengths in practice.
        `min_seq_len` (`int`):
            Only include subjects with at least this many events in the raw data.
        `seq_padding_side` (`str`, defaults to `'right'`):
            Whether to pad smaller sequences on the right (default) or the left (used for generation).
    """
    max_seq_len: int = 256
    min_seq_len: int = 2
    seq_padding_side: str = 'right'

    do_produce_static_data: bool = True

    save_dir: Optional[Path] = None

    def __post_init__(self):
        assert self.seq_padding_side in ('left', 'right')
        assert self.min_seq_len >= 0
        assert self.max_seq_len >= 1
        assert self.max_seq_len >= self.min_seq_len

        if type(self.save_dir) is str: self.save_dir = Path(self.save_dir)

    def to_dict(self) -> dict:
        """Represents this configuation object as a plain dictionary."""
        as_dict = dataclasses.asdict(self)
        as_dict['save_dir'] = str(as_dict['save_dir'])
        return as_dict

    @classmethod
    def from_dict(cls, as_dict: dict) -> 'EventStreamPytorchDatasetConfig':
        """Creates a new instance of this class from a plain dictionary."""
        as_dict['save_dir'] = Path(as_dict['save_dir'])
        return cls(**as_dict)

@dataclasses.dataclass
class MeasurementConfig(JSONableMixin):
    FUNCTORS = {
        'AgeFunctor': AgeFunctor,
        'TimeOfDayFunctor': TimeOfDayFunctor,
    }

    PREPROCESSING_METADATA_COLUMNS = OrderedDict({
        'value_type': str,
        'outlier_model': object,
        'normalizer': object,
    })

    """
    Base configuration class for a measurement in the EventStreamDataset.
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

        `measurement_metadata` (`Optional[pd.DataFrame]`, *optional*, defauls to `None`):
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
    name: Optional[str] = None
    temporality: Optional[TemporalityType] = None
    modality: Optional[DataModality] = None
    observation_frequency: Optional[float] = None

    # Specific to dynamic measures
    present_in_event_types: Optional[Set[str]] = None

    # Specific to time-dependent measures
    functor: Optional[TimeDependentFunctor] = None

    # Specific to categorical or partially observed multivariate regression measures.
    vocabulary: Optional[Vocabulary] = None

    # Specific to numeric measures
    values_column: Optional[str] = None
    measurement_metadata: Optional[Union[pd.DataFrame, pd.Series]] = None

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

                if self.modality is None: self.modality = self.functor.OUTPUT_MODALITY
                else: assert self.modality in (DataModality.DROPPED, self.functor.OUTPUT_MODALITY)

            case _: raise ValueError(f"`self.temporality = {self.temporality}` Invalid!")

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
            case _: raise ValueError(f"`self.modality = {self.modality}` Invalid!")

    def drop(self):
        """Sets the modality to DROPPED and does associated post-processing to ensure validity."""
        self.modality = DataModality.DROPPED
        self.measurement_metadata = None
        self.vocabulary = None

    @property
    def is_dropped(self) -> bool: return self.modality == DataModality.DROPPED
    @property
    def is_numeric(self) -> bool:
        return self.modality in (
            DataModality.MULTIVARIATE_REGRESSION, DataModality.UNIVARIATE_REGRESSION
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
                    {c: pd.Series([], dtype=t) for c, t in self.PREPROCESSING_METADATA_COLUMNS.items()},
                    index=pd.Index([], name=self.name),
                )
            case _: raise ValueError(f"Can't add metadata to a {self.modality} measure!")

    def add_missing_mandatory_metadata_cols(self):
        assert self.is_numeric
        match self.measurement_metadata:
            case None: self.add_empty_metadata()

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
        """Represents this configuation object as a plain dictionary."""
        as_dict = dataclasses.asdict(self)
        match self.measurement_metadata:
            case pd.DataFrame():
                as_dict['measurement_metadata'] = self.measurement_metadata.to_dict(orient='tight')
            case pd.Series():
                as_dict['measurement_metadata'] = self.measurement_metadata.to_dict(into=OrderedDict)
        if self.temporality == TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
            as_dict['functor'] = self.functor.to_dict()
        if self.present_in_event_types is not None:
            self.present_in_event_types = list(self.present_in_event_types)
        return as_dict

    @classmethod
    def from_dict(cls, as_dict: dict) -> 'MeasurementConfig':
        """Build a configuration object from a plain dictionary representation."""
        if as_dict['vocabulary'] is not None:
            as_dict['vocabulary'] = Vocabulary(**as_dict['vocabulary'])

        if as_dict['measurement_metadata'] is not None:
            match as_dict['modality']:
                case DataModality.MULTIVARIATE_REGRESSION:
                    as_dict['measurement_metadata'] = pd.DataFrame.from_dict(
                        as_dict['measurement_metadata'], orient='tight'
                    )
                case DataModality.UNIVARIATE_REGRESSION:
                    as_dict['measurement_metadata'] = pd.Series(as_dict['measurement_metadata'])
                case _:
                    raise ValueError("Config is non-numeric but has a measurement_metadata value observed.")

        if as_dict['functor'] is not None:
            assert as_dict['temporality'] == TemporalityType.FUNCTIONAL_TIME_DEPENDENT
            as_dict['functor'] = cls.FUNCTORS[as_dict['functor']['class']].from_dict(as_dict['functor'])

        if as_dict['present_in_event_types'] is not None:
            as_dict['present_in_event_types'] = set(as_dict['present_in_event_types'])

        return cls(**as_dict)

    def __eq__(self, other: EventStreamDatasetConfig) -> bool:
        return self.to_dict() == other.to_dict()

@dataclasses.dataclass
class EventStreamDatasetConfig(JSONableMixin):
    """
    Configuration options for parsing an `EventStreamDataset`.

    Args:
        `measurement_configs` (`Dict[str, MeasurementConfig]`, defaults to `{}`):
            The dataset configuration for this `EventStreamDataset`. Keys are measurement names, and
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

        `outlier_detector_config` (`Optional[Dict[str, Any]]`, defauls to `None`):
            Configuation options for outlier detection. If not `None`, must contain the key `'cls'`, which
            points to the class used outlier detection. All other keys and values are keyword arguments to be
            passed to the specified class. The API of these objects is expected to mirror scikit-learn outlier
            detection model APIs.
            If `None`, numerical outlier values are not removed.

        `normalizer_config` (`Optional[Dict[str, Any]]`, defauls to `None`):
            Configuation options for normalization. If not `None`, must contain the key `'cls'`, which points
            to the class used normalization. All other keys and values are keyword arguments to be passed to
            the specified class. The API of these objects is expected to mirror scikit-learn normalization
            system APIs.
            If `None`, numerical values are not normalized.
    """

    measurement_configs: Dict[str, MeasurementConfig] = dataclasses.field(default_factory = lambda: {})

    min_valid_column_observations: Optional[COUNT_OR_PROPORTION] = None
    min_valid_vocab_element_observations: Optional[COUNT_OR_PROPORTION] = None
    min_true_float_frequency: Optional[PROPORTION] = None
    min_unique_numerical_observations: Optional[COUNT_OR_PROPORTION] = None
    min_events_per_subject: Optional[int] = None

    outlier_detector_config: Optional[Dict[str, Any]] = None
    normalizer_config: Optional[Dict[str, Any]] = None

    save_dir: Optional[Path] = None

    def __post_init__(self):
        """Validates that parameters take on valid values."""
        for name, cfg in self.measurement_configs.items():
            if cfg.name is None: cfg.name = name
            else: assert cfg.name == name

        for var in (
            'min_valid_column_observations',
            'min_valid_vocab_element_observations',
            'min_unique_numerical_observations',
        ):
            val = getattr(self, var)
            if val is not None:
                assert (
                    ((type(val) is float) and (0 < val) and (val < 1)) or
                    ((type(val) is int) and (val > 1))
                )

        for var in ('min_true_float_frequency',):
            val = getattr(self, var)
            if val is not None: assert type(val) is float and (0 < val) and (val < 1)

        for var in ('outlier_detector_config', 'normalizer_config'):
            val = getattr(self, var)
            if val is not None: assert type(val) is dict and 'cls' in val

        for k, v in self.measurement_configs.items():
            try: v._validate()
            except Exception as e:
                raise ValueError(f"Measurement config {k} invalid!") from e

        if type(self.save_dir) is str: self.save_dir = Path(self.save_dir)

    def to_dict(self) -> dict:
        """Represents this configuation object as a plain dictionary."""
        as_dict = dataclasses.asdict(self)
        as_dict['measurement_configs'] = {
            k: v.to_dict() for k, v in self.measurement_configs.items()
        }
        return as_dict

    @classmethod
    def from_dict(cls, as_dict: dict) -> 'EventStreamDatasetConfig':
        """Build a configuration object from a plain dictionary representation."""
        as_dict['measurement_configs'] = {
            k: MeasurementConfig.from_dict(v) for k, v in as_dict['measurement_configs'].items()
        }

        return cls(**as_dict)

    @classmethod
    def from_simple_args(
        cls,
        dynamic_measurement_columns: Optional[Sequence[Union[str, Tuple[str, str]]]] = None,
        static_measurement_columns: Optional[Sequence[str]] = None,
        time_dependent_measurement_columns: Optional[Sequence[Tuple[str, TimeDependentFunctor]]] = None,
        **kwargs
    ) -> 'EventStreamDatasetConfig':
        """
        Builds an appropriate configuration object given a simple list of columns:

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

            `**kwargs`: Other keyword arguments will be passed to the `EventStreamDatasetConfig` constructor.

        Returns:
            An `EventStreamDatasetConfig` object with an appropriate `measurement_configs` variable set based
            on the above args, and all other passed keyword args passed as well.
        """
        measurement_configs = {}

        if dynamic_measurement_columns is not None:
            for measurement in dynamic_measurement_columns:
                if type(measurement) is tuple:
                    measurement, val_col = measurement
                    col_cfg = MeasurementConfig(
                        modality = DataModality.MULTIVARIATE_REGRESSION,
                        temporality = TemporalityType.DYNAMIC,
                        values_column = val_col
                    )
                else:
                    col_cfg = MeasurementConfig(
                        modality = DataModality.MULTI_LABEL_CLASSIFICATION,
                        temporality = TemporalityType.DYNAMIC,
                    )

                measurement_configs[measurement] = col_cfg

        if static_measurement_columns is not None:
            for measurement in static_measurement_columns:
                measurement_configs[measurement] = MeasurementConfig(
                    modality = DataModality.SINGLE_LABEL_CLASSIFICATION,
                    temporality = TemporalityType.STATIC,
                )

        if time_dependent_measurement_columns is not None:
            for measurement, functor in time_dependent_measurement_columns:
                measurement_configs[measurement] = MeasurementConfig(
                    temporality = TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                    functor = functor
                )

        return cls(measurement_configs=measurement_configs, **kwargs)

    def __eq__(self, other: EventStreamDatasetConfig) -> bool:
        return self.to_dict() == other.to_dict()

"""The base class for core dataset processing logic.

Attributes:
    INPUT_DF_T: This defines the type of the allowable input dataframes -- e.g., databases, filepaths,
        dataframes, etc.
    DF_T: This defines the type of internal dataframes -- e.g. polars DataFrames.
"""

import abc
import copy
import itertools
import json
from collections import defaultdict
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any, Generic, TypeVar

import humanize
import numpy as np
import pandas as pd
from mixins import SaveableMixin, SeedableMixin, TimeableMixin, TQDMableMixin
from plotly.graph_objs._figure import Figure

from ..utils import lt_count_or_proportion
from .config import (
    DatasetConfig,
    DatasetSchema,
    InputDFSchema,
    MeasurementConfig,
    VocabularyConfig,
)
from .types import DataModality, InputDataType, InputDFType, TemporalityType
from .visualize import Visualizer
from .vocabulary import Vocabulary

INPUT_DF_T = TypeVar("INPUT_DF_T")

DF_T = TypeVar("DF_T")


class DatasetBase(
    abc.ABC, Generic[DF_T, INPUT_DF_T], SeedableMixin, SaveableMixin, TimeableMixin, TQDMableMixin
):
    """A unified base class for dataset objects using different processing libraries.

    Args:
        config: Configuration object for this dataset.
        subjects_df: The dataframe containing all static, subject-level data. If this is specified,
            `events_df` and `dynamic_measurements_df` should also be specified. Otherwise, this will be built
            from source via the extraction pipeline defined in `input_schema`.
        events_df:  The dataframe containing all event timestamps, types, and subject IDs. If this is
            specified, `subjects_df` and `dynamic_measurements_df` should also be specified. Otherwise, this
            will be built from source via the extraction pipeline defined in `input_schema`.
        dynamic_measurements_df: The dataframe containing all time-varying measurement observations. If this
            is specified, `subjects_df` and `events_df` should also be specified. Otherwise, this will be
            built from source via the extraction pipeline defined in `input_schema`.
        input_schema: The schema configuration object to define the extraction pipeline for pulling raw data
            from source and produce the `subjects_df`, `events_df`, `dynamic_measurements_df` input view.
    """

    _PICKLER: str = "dill"
    """Dictates via which pickler the `_save` and `_load` methods will save/load objects of this class, as
    defined in `SaveableMixin`."""

    _DEL_BEFORE_SAVING_ATTRS: list[str] = [
        "_subjects_df",
        "_events_df",
        "_dynamic_measurements_df",
        "config",
        "inferred_measurement_configs",
    ]
    """Attributes that are saved via separate files, and will be deleted before pickling."""

    DF_SAVE_FORMAT: str = "parquet"
    """The save format for internal dataframes in this dataset."""

    SUBJECTS_FN: str = "subjects_df"
    """The name for the ``subjects_df`` save file."""

    EVENTS_FN: str = "events_df"
    """The name for the ``events_df`` save file."""

    DYNAMIC_MEASUREMENTS_FN: str = "dynamic_measurements_df"
    """The name for the ``dynamic_measurements_df`` save file."""

    @classmethod
    def subjects_fp(cls, save_dir: Path) -> Path:
        """Returns the filepath for the ``subjects_df`` given `save_dir` and class parameters."""
        return save_dir / f"{cls.SUBJECTS_FN}.{cls.DF_SAVE_FORMAT}"

    @classmethod
    def events_fp(cls, save_dir: Path) -> Path:
        """Returns the filepath for the ``events_df`` given `save_dir` and class parameters."""
        return save_dir / f"{cls.EVENTS_FN}.{cls.DF_SAVE_FORMAT}"

    @classmethod
    def dynamic_measurements_fp(cls, save_dir: Path) -> Path:
        """Returns the filepath for the ``dynamic_measurements_df`` given `save_dir` and class parameters."""
        return save_dir / f"{cls.DYNAMIC_MEASUREMENTS_FN}.{cls.DF_SAVE_FORMAT}"

    @classmethod
    @abc.abstractmethod
    def _load_input_df(
        cls,
        df: INPUT_DF_T,
        columns: list[tuple[str, InputDataType | tuple[InputDataType, str]]],
        subject_id_col: str | None = None,
        subject_ids_map: dict[Any, int] | None = None,
        subject_id_dtype: Any | None = None,
        filter_on: dict[str, bool | list[Any]] | None = None,
    ) -> DF_T:
        """Loads an input dataframe into the format expected by the processing library."""
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    @abc.abstractmethod
    def _process_events_and_measurements_df(
        cls,
        df: DF_T,
        event_type: str,
        columns_schema: dict[str, tuple[str, InputDataType]],
        ts_col: str | list[str],
    ) -> tuple[DF_T, DF_T | None]:
        """Performs the following steps on an input events and measurements dataframe:

        1. Produces a unified timestamp column representing the minimum of passed timestamps, with the name,
           ``'timestamp'``.
        2. Adds a categorical event type column either from column (if `event_type` begins with ``'COL:'``) or
           with value `event_type`.
        3. Extracts and renames the columns present in `columns_schema`.
        4. Adds an integer `event_id` column.
        4. Splits the dataframe into an events dataframe, storing `event_id`, `subject_id`, `event_type`,
           and `timestamp`, and a `measurements` dataframe, storing `event_id` and all other data columns.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    @abc.abstractmethod
    def _split_range_events_df(
        cls, df: DF_T, start_ts_col: str | list[str], end_ts_col: str | list[str]
    ) -> tuple[DF_T, DF_T, DF_T]:
        """Performs the following steps:

        1. Produces unified start and end timestamp columns representing the minimum of the passed start and
           end timestamps, respectively.
        2. Filters out records where the end timestamp is earlier than the start timestamp.
        3. Splits the dataframe into 3 events dataframes, all with only a single timestamp column, named
           `'timestamp'`:
           (a) An "EQ" dataframe, where start_ts_col == end_ts_col,
           (b) A "start" dataframe, with start events, and
           (c) An "end" dataframe, with end events.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    @abc.abstractmethod
    def _inc_df_col(cls, df: DF_T, col: str, inc_by: int) -> tuple[DF_T, int]:
        """Increments the values in `col` by a given amount and returns the resulting df."""
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    @abc.abstractmethod
    def _concat_dfs(cls, dfs: list[DF_T]) -> DF_T:
        """Concatenates a list of dataframes into a single dataframe."""
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    @abc.abstractmethod
    def _resolve_ts_col(cls, df: DF_T, ts_col: str | list[str], out_name: str = "timestamp") -> DF_T:
        """Adds the minimum of the columns `ts_col` as a `datetime` column with name `out_name`"""
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    @abc.abstractmethod
    def _rename_cols(cls, df: DF_T, to_rename: dict[str, str]) -> DF_T:
        """Renames the columns in df according to the {in_name: out_name}s specified in to_rename."""
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    def build_subjects_dfs(cls, schema: InputDFSchema) -> tuple[DF_T, dict[Hashable, int]]:
        """Builds and returns the subjects dataframe from `schema`.

        Args:
            schema: The input schema defining the subjects dataframe. This will include a definition of the
                input dataframe, the subject ID column, the static measurements columns to load, etc.

        Returns:
            Both the built `subjects_df` as well as a dictionary from the raw subject ID column values to the
            inferred numeric subject IDs.
        """
        subjects_df, ID_map = cls._load_input_df(
            schema.input_df,
            [(schema.subject_id_col, InputDataType.CATEGORICAL)] + schema.columns_to_load,
            filter_on=schema.filter_on,
            subject_id_source_col=schema.subject_id_col,
        )

        subjects_df = cls._rename_cols(subjects_df, {i: o for i, (o, _) in schema.unified_schmea.items()})

        return subjects_df, ID_map

    @classmethod
    def build_event_and_measurement_dfs(
        cls,
        subject_ids_map: dict[Any, int],
        subject_id_col: str,
        subject_id_dtype: Any,
        schemas_by_df: dict[INPUT_DF_T, list[InputDFSchema]],
    ) -> tuple[DF_T, DF_T]:
        """Builds and returns events and measurements dataframes from the input schema map.

        Args:
            subject_ids_map: A mapping from the input subject ID space to the inferred, output ID space. This
                is also used to filter dynamic input dataframes down to only valid subjects.
            subject_id_col: The name of the column containing (input) subject IDs.
            subject_id_dtype: The dtype of the output subject ID column.
            schemas_by_df: A mapping from input dataframe to associated event/measurement schemas.

        Returns:
            Both the built `events_df` and `dynamic_measurements_df`.
        """
        all_events_and_measurements = []
        event_types = []

        for df, schemas in schemas_by_df.items():
            all_columns = []

            all_columns.extend(itertools.chain.from_iterable(s.columns_to_load for s in schemas))

            try:
                df = cls._load_input_df(df, all_columns, subject_id_col, subject_ids_map, subject_id_dtype)
            except Exception as e:
                raise ValueError(f"Errored while loading {df}") from e

            for schema in schemas:
                if schema.filter_on:
                    df = cls._filter_col_inclusion(schema.filter_on)
                match schema.type:
                    case InputDFType.EVENT:
                        df = cls._resolve_ts_col(df, schema.ts_col, "timestamp")
                        all_events_and_measurements.append(
                            cls._process_events_and_measurements_df(
                                df=df,
                                event_type=schema.event_type,
                                columns_schema=schema.unified_schema,
                            )
                        )
                        event_types.append(schema.event_type)
                    case InputDFType.RANGE:
                        df = cls._resolve_ts_col(df, schema.start_ts_col, "start_time")
                        df = cls._resolve_ts_col(df, schema.end_ts_col, "end_time")
                        for et, unified_schema, sp_df in zip(
                            schema.event_type,
                            schema.unified_schema,
                            cls._split_range_events_df(df=df),
                        ):
                            all_events_and_measurements.append(
                                cls._process_events_and_measurements_df(
                                    sp_df, columns_schema=unified_schema, event_type=et
                                )
                            )
                        event_types.extend(schema.event_type)
                    case _:
                        raise ValueError(f"Invalid schema type {schema.type}.")

        all_events, all_measurements = [], []
        running_event_id_max = 0
        for event_type, (events, measurements) in zip(event_types, all_events_and_measurements):
            try:
                new_events = cls._inc_df_col(events, "event_id", running_event_id_max)
            except Exception as e:
                raise ValueError(f"Failed to increment event_id on {event_type}") from e

            if len(new_events) == 0:
                print(f"Empty new events dataframe of type {event_type}!")
                continue

            all_events.append(new_events)
            if measurements is not None:
                all_measurements.append(cls._inc_df_col(measurements, "event_id", running_event_id_max))

            running_event_id_max = all_events[-1]["event_id"].max() + 1

        return cls._concat_dfs(all_events), cls._concat_dfs(all_measurements)

    @classmethod
    def _get_preprocessing_model(
        cls,
        model_config: dict[str, Any],
        for_fit: bool = False,
    ) -> Any:
        """Returns the appropriate model class or instance given the config for pre-processing.

        This fetches the appropriate pre-processing model class (stored in ``model_config['cls']``) and either
        returns it directly (if not `for_fit`) or instantiates it with the non-``'cls'`` config parameters and
        returns the instance.

        Args:
            model_config: The configuration for the particular pre-processing model in question.
            for_fit: Whether the retrieved model will be used for fitting (in which case it must be
                instantiated with the passed configuration) or for transforming/predicting (in which case the
                fit parameters will be stored with the data and so only the class is needed).

        Returns:
            Either the model class (as indicated via ``model_config['cls']``) or an instance of the class as
            defined by non-``'cls'`` keyword parameters in `model_config`.

        Raises:
            KeyError: if ``'cls'`` is not in `model_config` or ``model_config['cls']`` is not in
                `PREPROCESSORS`.

        Examples:
            >>> class MockPreprocessor:
            ...     def __init__(self, name: str):
            ...         self.name = name
            ...     def __repr__(self) -> str:
            ...         return f"MockPreprocessor(name={repr(self.name)})"
            >>> DatasetBase.PREPROCESSORS = {'mock': MockPreprocessor}
            >>> DatasetBase._get_preprocessing_model({'cls': 'mock', 'name': 'test'}, True)
            MockPreprocessor(name='test')
            >>> DatasetBase._get_preprocessing_model({'cls': 'mock', 'name': 'test'}, False)
            <class '...MockPreprocessor'>
            >>> DatasetBase._get_preprocessing_model({'name': 'test'}, True)
            Traceback (most recent call last):
                ...
            KeyError: "Missing mandatory preprocessor class configuration parameter `'cls'`."
            >>> DatasetBase._get_preprocessing_model({'cls': 'invalid', 'name': 'test'}, True)
            Traceback (most recent call last):
                ...
            KeyError: 'Invalid preprocessor model class invalid! DatasetBase options are mock'
        """
        model_config = copy.deepcopy(model_config)
        if "cls" not in model_config:
            raise KeyError("Missing mandatory preprocessor class configuration parameter `'cls'`.")
        if model_config["cls"] not in cls.PREPROCESSORS:
            raise KeyError(
                f"Invalid preprocessor model class {model_config['cls']}! {cls.__name__} options are "
                f"{', '.join(cls.PREPROCESSORS.keys())}"
            )

        model_cls = cls.PREPROCESSORS[model_config.pop("cls")]
        return model_cls(**model_config) if for_fit else model_cls

    @classmethod
    @abc.abstractmethod
    def _read_df(cls, fp: Path, **kwargs) -> DF_T:
        """Reads a dataframe from `fp`."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _write_df(cls, df: DF_T, fp: Path, **kwargs):
        """Writes `df` to `fp`."""
        raise NotImplementedError

    @property
    def subjects_df(self) -> DF_T:
        """Lazily loads and/or returns the subjects dataframe from the implicit filepath.

        This will return the `_subjects_df` attribute, if defined and not `None`; otherwise, it will attempt
        to load the subjects dataframe from the implicit filepath defined by `config.save_dir` and
        `SUBJECTS_FN`.
        """
        if (not hasattr(self, "_subjects_df")) or self._subjects_df is None:
            subjects_fp = self.subjects_fp(self.config.save_dir)
            print(f"Loading subjects from {subjects_fp}...")
            self._subjects_df = self._read_df(subjects_fp)

        return self._subjects_df

    @subjects_df.setter
    def subjects_df(self, subjects_df: DF_T):
        self._subjects_df = subjects_df

    @property
    def events_df(self) -> DF_T:
        """Lazily loads and/or returns the events dataframe from the implicit filepath.

        This will return the `_events_df` attribute, if defined and not `None`; otherwise, it will attempt to
        load the events dataframe from the implicit filepath defined by `config.save_dir` and `EVENTS_FN`.
        """
        if (not hasattr(self, "_events_df")) or self._events_df is None:
            events_fp = self.events_fp(self.config.save_dir)
            print(f"Loading events from {events_fp}...")
            self._events_df = self._read_df(events_fp)

        return self._events_df

    @events_df.setter
    def events_df(self, events_df: DF_T):
        self._events_df = events_df

    @property
    def dynamic_measurements_df(self) -> DF_T:
        """Lazily loads and/or returns the measurements dataframe from the implicit filepath.

        This will return the `_dynamic_measurements_df` attribute, if defined and not `None`; otherwise, it
        will attempt to load the dynamic measurements dataframe from the implicit filepath defined by
        `config.save_dir` and `DYNAMIC_MEASUREMENTS_FN`.
        """
        if (not hasattr(self, "_dynamic_measurements_df")) or self._dynamic_measurements_df is None:
            dynamic_measurements_fp = self.dynamic_measurements_fp(self.config.save_dir)
            print(f"Loading dynamic_measurements from {dynamic_measurements_fp}...")
            self._dynamic_measurements_df = self._read_df(dynamic_measurements_fp)

        return self._dynamic_measurements_df

    @dynamic_measurements_df.setter
    def dynamic_measurements_df(self, dynamic_measurements_df: DF_T):
        self._dynamic_measurements_df = dynamic_measurements_df

    @classmethod
    def load(cls, load_dir: Path) -> "DatasetBase":
        """Loads and returns a dataset from disk.

        This function re-loads an instance of the calling class from disk. This function assumes that files
        are stored on disk in the following, distributed format:

        * The base configuration object is stored in the file ``'config.json'``, in JSON format.
        * If the saved dataset has already been fit, then the pre-processed measurement configs with inferred
          parameters are stroed in ``'inferred_measurement_configs.json'``, in JSON format. Note that these
          configs may in turn store their own attributes in further files, such as their
          `measurement_metadata` dataframes, which are stored on disk in separate files to facilitate lazy
          loading.
        * The raw or fully pre-processed subjects, events, and measurements dataframes are stored in their
          respective filenames (`SUBJECTS_FN`, `EVENTS_FN`, `DYNAMIC_MEASUREMENTS_FN`).
        * Remaining attributes are stored in pickle format at ``'E.pkl'``.

        Args:
            load_dir: The path to the directory on disk from which the dataset should be loaded.

        Raises:
            FileNotFoundError: If either the attributes file or config file do not exist.
        """

        attrs_fp = load_dir / "E.pkl"

        attrs_to_add = {
            "config": DatasetConfig.from_json_file(load_dir / "config.json"),
        }
        inferred_measurement_configs_fp = load_dir / "inferred_measurement_configs.json"
        if inferred_measurement_configs_fp.is_file():
            with open(inferred_measurement_configs_fp) as f:
                attrs_to_add["inferred_measurement_configs"] = {
                    k: MeasurementConfig.from_dict(v) for k, v in json.load(f).items()
                }

        return super()._load(attrs_fp, **attrs_to_add)

    def save(self, **kwargs):
        """Saves the calling object to disk, in the directory `self.config.save_dir`.

        This function stores to disk the internal parameters of the calling object, in the following format:

        * The base configuration object is stored in the file ``'config.json'``, in JSON format.
        * If the saved dataset has already been fit, then the pre-processed measurement configs with inferred
          parameters are stroed in ``'inferred_measurement_configs.json'``, in JSON format. Note that these
          configs may in turn store their own attributes in further files, such as their
          `measurement_metadata` dataframes, which are stored on disk in separate files to facilitate lazy
          loading.
        * The raw or fully pre-processed subjects, events, and measurements dataframes are stored in their
          respective filenames (`SUBJECTS_FN`, `EVENTS_FN`, `DYNAMIC_MEASUREMENTS_FN`).
        * Remaining attributes are stored in pickle format at ``'E.pkl'``.

        Args:
            do_overwrite: Keyword only; if passed with a value evaluating to `True`, then the system will
                overwrite any files that exist, rather than erroring.

        Raises:
            FileExistsError: If any of the desired filepaths already exist and `do_overwrite` is False.
        """

        self.config.save_dir.mkdir(parents=True, exist_ok=True)

        do_overwrite = kwargs.get("do_overwrite", False)

        config_fp = self.config.save_dir / "config.json"
        self.config.to_json_file(config_fp, do_overwrite=do_overwrite)

        if self._is_fit:
            inferred_measurement_metadata_dir = self.config.save_dir / "inferred_measurement_metadata"
            for k, v in self.inferred_measurement_configs.items():
                fp = inferred_measurement_metadata_dir / f"{k}.csv"
                v.cache_measurement_metadata(fp)

            inferred_measurement_configs_fp = self.config.save_dir / "inferred_measurement_configs.json"
            inferred_measurement_configs = {
                k: v.to_dict() for k, v in self.inferred_measurement_configs.items()
            }

            with open(inferred_measurement_configs_fp, mode="w") as f:
                json.dump(inferred_measurement_configs, f)

        super()._save(self.config.save_dir / "E.pkl", **kwargs)

        vocab_config_fp = self.config.save_dir / "vocabulary_config.json"

        self.vocabulary_config.to_json_file(vocab_config_fp, do_overwrite=do_overwrite)

        subjects_fp = self.subjects_fp(self.config.save_dir)
        events_fp = self.events_fp(self.config.save_dir)
        dynamic_measurements_fp = self.dynamic_measurements_fp(self.config.save_dir)

        self._write_df(self.subjects_df, subjects_fp, do_overwrite=do_overwrite)
        self._write_df(self.events_df, events_fp, do_overwrite=do_overwrite)
        self._write_df(self.dynamic_measurements_df, dynamic_measurements_fp, do_overwrite=do_overwrite)

    def __init__(
        self,
        config: DatasetConfig,
        subjects_df: DF_T | None = None,
        events_df: DF_T | None = None,
        dynamic_measurements_df: DF_T | None = None,
        input_schema: DatasetSchema | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if "do_overwrite" in kwargs:
            self.do_overwrite = kwargs["do_overwrite"]

        if (
            subjects_df is None or events_df is None or dynamic_measurements_df is None
        ) and input_schema is None:
            raise ValueError(
                "Must set input_schema if subjects_df, events_df, or dynamic_measurements_df are None!"
            )

        if input_schema is None:
            if subjects_df is None:
                raise ValueError("Must set subjects_df if input_schema is None!")
            if events_df is None:
                raise ValueError("Must set events_df if input_schema is None!")
            if dynamic_measurements_df is None:
                raise ValueError("Must set dynamic_measurements_df if input_schema is None!")
        else:
            if subjects_df is not None:
                raise ValueError("Can't set subjects_df if input_schema is not None!")
            if events_df is not None:
                raise ValueError("Can't set events_df if input_schema is not None!")
            if dynamic_measurements_df is not None:
                raise ValueError("Can't set dynamic_measurements_df if input_schema is not None!")

            subjects_df, ID_map = self.build_subjects_dfs(input_schema.static)
            subject_id_dtype = subjects_df["subject_id"].dtype

            events_df, dynamic_measurements_df = self.build_event_and_measurement_dfs(
                ID_map,
                input_schema.static.subject_id_col,
                subject_id_dtype,
                input_schema.dynamic_by_df,
            )

        self.config = config
        self._is_fit = False

        # After pre-processing, we may infer new types or otherwise change measurement configuration, so
        # we store a separage configuration object for post-processing. It is initialized as empty as we have
        # not yet pre-processed anything.
        self.inferred_measurement_configs = {}

        self._validate_and_set_initial_properties(subjects_df, events_df, dynamic_measurements_df)

        self.split_subjects = {}

    def _validate_and_set_initial_properties(self, subjects_df, events_df, dynamic_measurements_df):
        """Validates the input dataframes and sets initial properties of the calling object.

        This validates that the initial dataframes are appropriately configured, re-sets certain types to
        minimal-memory ``dtypes`` (e.g., ensuring ID columns are set to the smallest valid ``uint`` type), and
        sets non-DF parameters such as `subject_ids`, `event_types`, and `n_events_per_subject`.

        Args:
            subjects_df: The subjects dataframe.
            events_df: The events dataframe.
            dynamic_measurements_df: The dynamic measurements dataframe.
        """

        self.subject_ids = []
        self.event_types = []
        self.n_events_per_subject = {}

        (
            self.subjects_df,
            self.events_df,
            self.dynamic_measurements_df,
        ) = self._validate_initial_dfs(subjects_df, events_df, dynamic_measurements_df)

        if self.events_df is not None:
            self._agg_by_time()
            self._sort_events()
        self._update_subject_event_properties()

    @abc.abstractmethod
    def _validate_initial_dfs(
        self, subjects_df: DF_T, events_df: DF_T, dynamic_measurements_df: DF_T
    ) -> tuple[DF_T, DF_T, DF_T]:
        """Validates input dataframes and massages their internal types to minimize memory requirements."""
        raise NotImplementedError("This method must be implemented by a subclass.")

    @abc.abstractmethod
    def _update_subject_event_properties(self):
        """Updates the `subject_ids`, `event_types`, and `n_events_per_subject` internal properties."""
        raise NotImplementedError("This method must be implemented by a subclass.")

    @TimeableMixin.TimeAs
    def _filter_subjects(self):
        """Filters the internal subjects dataframe to only those who have a minimum number of events."""
        if self.config.min_events_per_subject is None:
            return

        subjects_to_keep = [
            s for s, n in self.n_events_per_subject.items() if n >= self.config.min_events_per_subject
        ]
        self.subjects_df = self._filter_col_inclusion(self.subjects_df, {"subject_id": subjects_to_keep})
        self.events_df = self._filter_col_inclusion(self.events_df, {"subject_id": subjects_to_keep})
        self.dynamic_measurements_df = self._filter_col_inclusion(
            self.dynamic_measurements_df, {"event_id": list(self.events_df["event_id"])}
        )

    @TimeableMixin.TimeAs
    @abc.abstractmethod
    def _agg_by_time(self):
        """Aggregates events into temporal buckets governed by `self.config.agg_by_time_scale`.

        Aggregates the events_df by subject_id and timestamp (into buckets of size
        `self.config.agg_by_time_scale`), combining event_types into grouped categories with names
        concatenated with a separator of '&', then re-aligns measurements into the new event IDs in
        `dynamic_measurements_df`. Note that no numerical aggregation (e.g., mean, etc.) happens here; all
        data is retained, and only dynamic measurement event IDs are updated.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    @TimeableMixin.TimeAs
    @abc.abstractmethod
    def _sort_events(self):
        """Sorts events by subject ID and timestamp in ascending order."""
        raise NotImplementedError("This method must be implemented by a subclass.")

    @SeedableMixin.WithSeed
    @TimeableMixin.TimeAs
    def split(
        self,
        split_fracs: Sequence[float],
        split_names: Sequence[str] | None = None,
    ):
        """Splits the underlying dataset into random sets by `subject_id`.

        Args:
            split_fracs: The fractional sizes of the desired splits. If it sums to < 1, the remainder will be
                tracked **in an extra split** at the end of the list. All split fractions must be positive
                floating point numbers less than 1.
            split_names: If specified, assigns the passed names to each split. Must be of the same size as
                `split_fracs` (after it is expanded to sum to 1 if necessary). If unset, and there are two
                splits, it defaults to [`train`, `held_out`]. If there are three, it defaults to `['train',
                'tuning', 'held_out']. If more than 3, it defaults to `['split_0', 'split_1', ...]`. Split
                names of `train`, `tuning`, and `held_out` have special significance and are used elsewhere in
                the model, so if `split_names` does not reflect those other things may not work down the line.

        Raises:
            ValueError: if `split_fracs` contains anything outside the range of (0, 1], sums to something > 1,
                or is not of the same length as `split_names`.
        """
        split_fracs = list(split_fracs)

        if min(split_fracs) <= 0 or max(split_fracs) > 1 or sum(split_fracs) > 1:
            raise ValueError(
                "split_fracs invalid! Want a list of numbers in (0, 1] that sums to no more than 1; got "
                f"{repr(split_fracs)}"
            )

        if sum(split_fracs) < 1:
            split_fracs.append(1 - sum(split_fracs))

        if split_names is None:
            if len(split_fracs) == 2:
                split_names = ["train", "held_out"]
            elif len(split_fracs) == 3:
                split_names = ["train", "tuning", "held_out"]
            else:
                split_names = [f"split_{i}" for i in range(len(split_fracs))]
        elif len(split_names) != len(split_fracs):
            raise ValueError(
                f"split_names and split_fracs must be the same length; got {len(split_names)} and "
                f"{len(split_fracs)}"
            )

        # As split fractions may not result in integer split sizes, we shuffle the split names and fractions
        # so that the splits that exceed the desired size are not always the last ones in the original passed
        # order.
        split_names_idx = np.random.permutation(len(split_names))
        split_names = [split_names[i] for i in split_names_idx]
        split_fracs = [split_fracs[i] for i in split_names_idx]

        subjects = np.random.permutation(list(self.subject_ids))
        split_lens = (np.array(split_fracs[:-1]) * len(subjects)).round().astype(int)
        split_lens = np.append(split_lens, len(subjects) - split_lens.sum())

        subjects_per_split = np.split(subjects, split_lens.cumsum())

        self.split_subjects = {k: set(v) for k, v in zip(split_names, subjects_per_split)}

    @classmethod
    @abc.abstractmethod
    def _filter_col_inclusion(cls, df: DF_T, col_inclusion_targets: dict[str, bool | Sequence[Any]]) -> DF_T:
        """Filters `df` via the mapping of column names to allowed values in `col_inclusion_targets`."""
        raise NotImplementedError("This method must be implemented by a subclass.")

    @property
    def train_subjects_df(self) -> DF_T:
        """Returns the train set split of subjects_df."""
        return self._filter_col_inclusion(self.subjects_df, {"subject_id": self.split_subjects["train"]})

    @property
    def tuning_subjects_df(self) -> DF_T:
        """Returns the tuning set split of subjects_df."""
        return self._filter_col_inclusion(self.subjects_df, {"subject_id": self.split_subjects["tuning"]})

    @property
    def held_out_subjects_df(self) -> DF_T:
        """Returns the held-out set split of subjects_df."""
        return self._filter_col_inclusion(self.subjects_df, {"subject_id": self.split_subjects["held_out"]})

    @property
    def train_events_df(self) -> DF_T:
        """Returns the train set split of events_df."""
        return self._filter_col_inclusion(self.events_df, {"subject_id": self.split_subjects["train"]})

    @property
    def tuning_events_df(self) -> DF_T:
        """Returns the tuning set split of events_df."""
        return self._filter_col_inclusion(self.events_df, {"subject_id": self.split_subjects["tuning"]})

    @property
    def held_out_events_df(self) -> DF_T:
        """Returns the held-out set split of events_df."""
        return self._filter_col_inclusion(self.events_df, {"subject_id": self.split_subjects["held_out"]})

    @property
    def train_dynamic_measurements_df(self) -> DF_T:
        """Returns the train set split of dynamic_measurements_df."""
        event_ids = self.train_events_df["event_id"]
        return self._filter_col_inclusion(self.dynamic_measurements_df, {"event_id": list(event_ids)})

    @property
    def tuning_dynamic_measurements_df(self) -> DF_T:
        """Returns the tuning set split of dynamic_measurements_df."""
        event_ids = self.tuning_events_df["event_id"]
        return self._filter_col_inclusion(self.dynamic_measurements_df, {"event_id": list(event_ids)})

    @property
    def held_out_dynamic_measurements_df(self) -> DF_T:
        """Returns the held-out set split of dynamic_measurements_df."""
        event_ids = self.held_out_events_df["event_id"]
        return self._filter_col_inclusion(self.dynamic_measurements_df, {"event_id": list(event_ids)})

    @TimeableMixin.TimeAs
    def preprocess(self):
        """Fits all pre-processing parameters over the train set, then transforms all observations.

        This entails the following steps:

        1. First, filter out subjects that have too few events.
        2. Next, pre-compute the `FUNCTIONAL_TIME_DEPENDENT` temporality measurements and store their values
           in the events dataframe.
        3. Next, fit all pre-processing parameters over the observed measurements.
        4. Finally, transform all data via the fit pre-processing parameters.
        """
        self._filter_subjects()
        self._add_time_dependent_measurements()
        self.fit_measurements()
        self.transform_measurements()

    @TimeableMixin.TimeAs
    @abc.abstractmethod
    def _add_time_dependent_measurements(self):
        """Adds `FUNCTIONAL_TIME_DEPENDENT` temporality measurement values to events_df."""
        raise NotImplementedError("This method must be implemented by a subclass.")

    @TimeableMixin.TimeAs
    def _get_source_df(self, config: MeasurementConfig, do_only_train: bool = True) -> tuple[str, str, DF_T]:
        """Returns the name of the source attribute, its id column, and that dataframe for `config`.

        Measurements with different configs are stored in different internal dataframes (e.g., `STATIC`
        measurements in `subjects_df`, `DYNAMIC` measurements in `dynamic_measurements_df`), and are goverend
        by different natural ID columns. This function gets and returns the appropriate attribute name, ID
        column name for that attribute, and the associated dataframe.

        Args:
            config: The measurement config for which we should retrieve the source dataframe.
            do_only_train: Whether or not we should also return only these data on the train set or not.

        Raises:
            ValueError: If the passed measurement config has an invalid temporality type.
        """
        match config.temporality:
            case TemporalityType.DYNAMIC:
                source_attr = "dynamic_measurements_df"
                source_id = "measurement_id"
                if do_only_train:
                    source_df = self.train_dynamic_measurements_df
                else:
                    source_df = self.dynamic_measurements_df

            case TemporalityType.STATIC:
                source_attr = "subjects_df"
                source_id = "subject_id"
                source_df = self.train_subjects_df if do_only_train else self.subjects_df

            case TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                source_attr = "events_df"
                source_id = "event_id"
                source_df = self.train_events_df if do_only_train else self.events_df

            case _:
                raise ValueError(f"Called get_source_df on temporality type {config.temporality}!")
        return source_attr, source_id, source_df

    @TimeableMixin.TimeAs
    def fit_measurements(self):
        """Fits all preprocessing parameters over the training dataset, according to `self.config`.

        Raises:
            ValueError: if fitting preprocessing parameters fails for a given measurement.
        """
        self._is_fit = False

        for measure, config in self.config.measurement_configs.items():
            if config.is_dropped:
                continue

            self.inferred_measurement_configs[measure] = copy.deepcopy(config)
            config = self.inferred_measurement_configs[measure]

            _, _, source_df = self._get_source_df(config, do_only_train=True)

            if measure not in source_df:
                print(f"WARNING: Measure {measure} not found! Dropping...")
                config.drop()
                continue

            total_possible, total_observed = self._total_possible_and_observed(measure, config, source_df)
            source_df = self._filter_col_inclusion(source_df, {measure: True})

            if total_possible == 0:
                print(f"Found no possible events for {measure}!")
                config.drop()
                continue

            config.observation_frequency = total_observed / total_possible

            # 2. Drop the column if observations occur too rarely.
            if lt_count_or_proportion(
                total_observed, self.config.min_valid_column_observations, total_possible
            ):
                config.drop()
                continue

            if config.is_numeric:
                config.add_missing_mandatory_metadata_cols()
                try:
                    config.measurement_metadata = self._fit_measurement_metadata(measure, config, source_df)
                except BaseException as e:
                    raise ValueError(f"Fitting measurement metadata failed for measure {measure}!") from e

            if config.vocabulary is None:
                config.vocabulary = self._fit_vocabulary(measure, config, source_df)

                # 4. Eliminate observations that occur too rarely.
                if config.vocabulary is not None:
                    if self.config.min_valid_vocab_element_observations is not None:
                        config.vocabulary.filter(
                            len(source_df), self.config.min_valid_vocab_element_observations
                        )

                    # 5. If all observations were eliminated, drop the column.
                    if config.vocabulary.vocabulary == ["UNK"]:
                        config.drop()

        self._is_fit = True

    @abc.abstractmethod
    def _total_possible_and_observed(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> tuple[int, int]:
        """Returns the total possible/actual instances where `measure` could be/was observed.

        Possible means number of subjects (for static measurements) or number of unique events (for dynamic or
        functional time dependent measurements). Actual means where the given measurement column takes on a
        non-null value. For a multivariate regression measurement, the column that must be non-null is the key
        column, not the value column.

        Args:
            measure: The name of the measurement.
            config: The measurement config for the given measurement.
            source_df: The dataframe from which to compute the total possible/actual instances.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    @abc.abstractmethod
    def _fit_measurement_metadata(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> pd.DataFrame:
        """Fits & returns the metadata df for a numeric measurement over the source df.

        The measurement metadata structure stores pre-processing parameters for numerical variables like
        value type, outlier model parameters, normalizer parameters, etc.

        Args:
            measure: The name of the measurement.
            config: The measurement config for the given measurement.
            source_df: The dataframe from which to compute the measurement metadata columns.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    @TimeableMixin.TimeAs
    @abc.abstractmethod
    def _fit_vocabulary(self, measure: str, config: MeasurementConfig, source_df: DF_T) -> Vocabulary:
        """Fits and returns the vocabulary for a categorical measurement over the source dataframe.

        Args:
            measure: The name of the measurement.
            config: The measurement config for the given measurement.
            source_df: The dataframe from which to compute the measurement metadata columns.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    @TimeableMixin.TimeAs
    def transform_measurements(self):
        """Transforms the entire dataset given the fit preprocessing parameters.

        Raises:
            ValueError: If transforming fails for a given measurement.
        """
        for measure, config in self.measurement_configs.items():
            source_attr, id_col, source_df = self._get_source_df(config, do_only_train=False)

            source_df = self._filter_col_inclusion(source_df, {measure: True})
            updated_cols = [measure]

            try:
                if config.is_numeric:
                    source_df = self._transform_numerical_measurement(measure, config, source_df)

                    if config.modality == DataModality.MULTIVARIATE_REGRESSION:
                        updated_cols.append(config.values_column)

                    if self.config.outlier_detector_config is not None:
                        updated_cols.append(f"{measure}_is_inlier")

                if config.vocabulary is not None:
                    source_df = self._transform_categorical_measurement(measure, config, source_df)

            except BaseException as e:
                raise ValueError(f"Transforming measurement failed for measure {measure}!") from e

            self._update_attr_df(source_attr, id_col, source_df, updated_cols)

    @TimeableMixin.TimeAs
    @abc.abstractmethod
    def _update_attr_df(self, attr: str, id_col: str, df: DF_T, cols_to_update: list[str]):
        """Replaces the columns in `cols_to_update` in self's df stored @ `attr` with the vals in `df`.

        Replaces all values in the currently stored dataframe at the columns in cols_to_update with
        None, then further updates the dataframe by ID with the values for those columns in `df`.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    @TimeableMixin.TimeAs
    @abc.abstractmethod
    def _transform_numerical_measurement(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> DF_T:
        """Transforms the numerical measurement `measure` according to config `config`.

        Performs the following steps:

        1. Transforms keys to categorical representations for categorical keys.
        2. Eliminates any values associated with dropped or categorical keys.
        3. Eliminates hard outliers and performs censoring via specified config.
        4. Converts values to desired types.
        5. Adds inlier/outlier indices and remove learned outliers.
        6. Normalizes values.

        Args:
            measure: The column name of the governing measurement to transform.
            config: The configuration object governing this measure.
            source_df: The dataframe object containing the measure to be transformed.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    @TimeableMixin.TimeAs
    @abc.abstractmethod
    def _transform_categorical_measurement(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> DF_T:
        """Converts the elements to categorical column types according to the learned vocabularies.

        Args:
            measure: The column name of the governing measurement to transform.
            config: The configuration object governing this measure.
            source_df: The dataframe object containing the measure to be transformed.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    @property
    def has_static_measurements(self):
        """Returns `True` if the dataset has any static measurements."""
        return (self.subjects_df is not None) and any(
            cfg.temporality == TemporalityType.STATIC for cfg in self.measurement_configs.values()
        )

    @property
    def measurement_configs(self):
        """Errors if not fit; otherwise returns all fit, non-dropped measurement configs.

        Raises:
            ValueError: if is not fit.
        """

        if not self._is_fit:
            raise ValueError("Can't call measurement_configs if not yet fit!")
        return {m: c for m, c in self.inferred_measurement_configs.items() if not c.is_dropped}

    @property
    def dynamic_numerical_columns(self):
        """Returns all numerical metadata column key-column, value-column pairs."""
        return [
            (k, cfg.values_column)
            for k, cfg in self.measurement_configs.items()
            if (cfg.is_numeric and cfg.temporality == TemporalityType.DYNAMIC)
        ]

    @property
    def time_dependent_numerical_columns(self):
        """Returns all numerical metadata column key-column, value-column pairs."""
        return [
            k
            for k, cfg in self.measurement_configs.items()
            if (cfg.is_numeric and cfg.temporality == TemporalityType.FUNCTIONAL_TIME_DEPENDENT)
        ]

    @property
    def measurement_idxmaps(self):
        """Accesses the fit vocabularies vocabulary idxmap objects, per measurement column."""
        idxmaps = {"event_type": {et: i for i, et in enumerate(self.event_types)}}
        for m, config in self.measurement_configs.items():
            if config.vocabulary is not None:
                idxmaps[m] = config.vocabulary.idxmap
        return idxmaps

    @property
    def measurement_vocabs(self):
        """Accesses the fit vocabularies vocabulary objects, per measurement column."""

        vocabs = {"event_type": self.event_types}
        for m, config in self.measurement_configs.items():
            if config.vocabulary is not None:
                vocabs[m] = config.vocabulary.vocabulary
        return vocabs

    @TimeableMixin.TimeAs
    def cache_deep_learning_representation(
        self, subjects_per_output_file: int | None = None, do_overwrite: bool = False
    ):
        """Writes a deep-learning friendly representation of the dataset to disk.

        The deep learning format produced will have one row per subject, with the following columns:

        * ``subject_id``: This column will be an unsigned integer type, and will have the ID of the subject
          for each row.
        * ``start_time``: This column will be a `datetime` type, and will contain the start time of the
          subject's record.
        * ``static_indices``: This column is a ragged, sparse representation of the categorical static
          measurements observed for this subject. Each element of this column will itself be a list of
          unsigned integers corresponding to indices into the unified vocabulary for the static measurements
          observed for that subject.
        * ``static_measurement_indices``: This column corresponds in shape to ``static_indices``, but contains
          unsigned integer indices into the unified measurement vocabulary, defining to which measurement each
          observation corresponds. It is of the same shape and of a consistent order as ``static_indices.``
        * ``time``: This column is a ragged array of the time in minutes from the start time at which each
          event takes place. For a given row, the length of the array within this column corresponds to the
          number of events that subject has.
        * ``dynamic_indices``: This column is a doubly ragged array containing the indices of the observed
          values within the unified vocabulary per event per subject. Each subject's data for this column
          consists of an array of arrays, each containing only the indices observed at each event.
        * ``dynamic_measurement_indices`` This column is a doubly ragged array containing the indices of the
          observed measurements per event per subject. Each subject's data for this column consists of an
          array of arrays, each containing only the indices of measurements observed at each event. It is of
          the same shape and of a consistent order as ``dynamic_indices``.
        * ``dynamic_values`` This column is a doubly ragged array containing the indices of the
          observed measurements per event per subject. Each subject's data for this column consists of an
          array of arrays, each containing only the indices of measurements observed at each event. It is of
          the same shape and of a consistent order as ``dynamic_indices``.

        Args:
            subjects_per_output_file: How big to chunk the dataset down for writing to disk; larger values
                will make fewer chunks but increase the memory cost.
            do_overwrite: Whether or not to overwrite any existing file on disk.
        """

        DL_dir = self.config.save_dir / "DL_reps"
        DL_dir.mkdir(exist_ok=True, parents=True)

        if subjects_per_output_file is None:
            subject_chunks = [None]
        else:
            subjects = np.random.permutation(list(self.subject_ids))
            subject_chunks = np.array_split(
                subjects,
                np.arange(subjects_per_output_file, len(subjects), subjects_per_output_file),
            )
            subject_chunks = [list(c) for c in subject_chunks]

        for chunk_idx, subjects_list in self._tqdm(list(enumerate(subject_chunks))):
            cached_df = self.build_DL_cached_representation(subject_ids=subjects_list)

            for split, subjects in self.split_subjects.items():
                fp = DL_dir / f"{split}_{chunk_idx}.{self.DF_SAVE_FORMAT}"

                split_cached_df = self._filter_col_inclusion(cached_df, {"subject_id": subjects})
                self._write_df(split_cached_df, fp, do_overwrite=do_overwrite)

    @property
    def vocabulary_config(self) -> VocabularyConfig:
        """Returns the implied `VocabularyConfig` object corresponding to this (fit) dataset.

        This property collates vocabulary information across all measurements into a format that is concise,
        but complete for downstream DL applications.
        """
        measurements_per_generative_mode = defaultdict(list)
        measurements_per_generative_mode[DataModality.SINGLE_LABEL_CLASSIFICATION].append("event_type")
        for m, cfg in self.measurement_configs.items():
            if cfg.temporality != TemporalityType.DYNAMIC:
                continue

            measurements_per_generative_mode[cfg.modality].append(m)
            if cfg.modality == DataModality.MULTIVARIATE_REGRESSION:
                measurements_per_generative_mode[DataModality.MULTI_LABEL_CLASSIFICATION].append(m)

        return VocabularyConfig(
            vocab_sizes_by_measurement={m: len(idxmap) for m, idxmap in self.measurement_idxmaps.items()},
            vocab_offsets_by_measurement=self.unified_vocabulary_offsets,
            measurements_idxmap=self.unified_measurements_idxmap,
            event_types_idxmap=self.unified_vocabulary_idxmap["event_type"],
            measurements_per_generative_mode=dict(measurements_per_generative_mode),
        )

    @property
    def unified_measurements_vocab(self) -> list[str]:
        """Returns a unified vocabulary of observed measurements."""
        return ["event_type"] + list(sorted(self.measurement_configs.keys()))

    @property
    def unified_measurements_idxmap(self) -> dict[str, int]:
        """Returns a unified idxmap of observed measurements."""
        return {m: i + 1 for i, m in enumerate(self.unified_measurements_vocab)}

    @property
    def unified_vocabulary_offsets(self) -> dict[str, int]:
        """Returns a set of offsets detailing at what position each measurement's vocab starts."""
        offsets, curr_offset = {}, 1
        for m in self.unified_measurements_vocab:
            offsets[m] = curr_offset
            if m in self.measurement_vocabs:
                curr_offset += len(self.measurement_vocabs[m])
            else:
                curr_offset += 1
        return offsets

    @property
    def unified_vocabulary_idxmap(self) -> dict[str, dict[str, int]]:
        """Provides a unified idxmap spanning all measurements' vocabularies (concatenated via offsets)."""
        idxmaps = {}
        for m, offset in self.unified_vocabulary_offsets.items():
            if m in self.measurement_idxmaps:
                idxmaps[m] = {v: i + offset for v, i in self.measurement_idxmaps[m].items()}
            else:
                idxmaps[m] = {m: offset}
        return idxmaps

    @abc.abstractmethod
    def build_DL_cached_representation(
        self, subject_ids: list[int] | None = None, do_sort_outputs: bool = False
    ) -> DF_T:
        """Produces the deep learning format dataframe described previously for the passed
        subjects:"""

        raise NotImplementedError("This method must be implemented by a subclass.")

    @abc.abstractmethod
    def _denormalize(self, events_df: DF_T, col: str) -> DF_T:
        """Un-normalizes the column `col` in df `events_df`."""
        raise NotImplementedError("This method must be implemented by a subclass.")

    def describe(
        self,
        do_print_measurement_summaries: bool = True,
        viz_config: Visualizer | None = None,
    ) -> list[Figure] | None:
        """Describes the dataset, both in language and in figures."""
        print(
            f"Dataset has {humanize.intword(len(self.subjects_df))} subjects, "
            f"with {humanize.intword(len(self.events_df))} events and "
            f"{humanize.intword(len(self.dynamic_measurements_df))} measurements."
        )
        if do_print_measurement_summaries:
            print(f"Dataset has {len(self.measurement_configs)} measurements:")
            for meas, cfg in self.measurement_configs.items():
                if cfg.name is None:
                    cfg.name = meas
                cfg.describe(line_width=60)
                print()

        if viz_config is not None:
            return self.visualize(viz_config)

    def visualize(
        self,
        viz_config: Visualizer,
    ) -> list[Figure]:
        """Visualizes the dataset, along several axes."""

        if viz_config.subset_size is not None:
            viz_config.subset_random_seed = self._seed(seed=viz_config.subset_random_seed, key="visualize")

        if viz_config.subset_size is not None:
            subject_ids = list(np.random.choice(list(self.subject_ids), viz_config.subset_size))

            subjects_df = self._filter_col_inclusion(self.subjects_df, {"subject_id": subject_ids})
            events_df = self._filter_col_inclusion(self.events_df, {"subject_id": subject_ids})
            dynamic_measurements_df = self._filter_col_inclusion(
                self.dynamic_measurements_df, {"event_id": list(events_df["event_id"])}
            )
        else:
            subjects_df = self.subjects_df
            events_df = self.events_df
            dynamic_measurements_df = self.dynamic_measurements_df

        if viz_config.age_col is not None:
            events_df = self._denormalize(events_df, viz_config.age_col)

        figs = viz_config.plot(subjects_df, events_df, dynamic_measurements_df)
        return figs

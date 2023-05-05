import abc
import copy
import itertools
from collections import defaultdict
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

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

# This defines the type of the allowable input dataframes -- e.g., databases, filepaths, dataframes, etc.
INPUT_DF_T = TypeVar("INPUT_DF_T")

# This defines the type of internal dataframes -- e.g. polars DataFrames.
DF_T = TypeVar("DF_T")


class DatasetBase(
    abc.ABC, Generic[DF_T, INPUT_DF_T], SeedableMixin, SaveableMixin, TimeableMixin, TQDMableMixin
):
    """A unified base class for dataset objects using different processing libraries."""

    # Dictates in which format the `_save` and `_load` methods will save/load objects of this class, as
    # defined in `SaveableMixin`.
    _PICKLER = "dill"

    # Attributes that are saved via separate, explicit filetypes.
    _DEL_BEFORE_SAVING_ATTRS = ["_subjects_df", "_events_df", "_dynamic_measurements_df"]

    # Dictates how dataframes are saved and loaded in this class.
    DF_SAVE_FORMAT = "parquet"
    SUBJECTS_FN = "subjects_df"
    EVENTS_FN = "events_df"
    DYNAMIC_MEASUREMENTS_FN = "dynamic_measurements_df"

    @classmethod
    def subjects_fp(cls, save_dir: Path) -> Path:
        return save_dir / f"{cls.SUBJECTS_FN}.{cls.DF_SAVE_FORMAT}"

    @classmethod
    def events_fp(cls, save_dir: Path) -> Path:
        return save_dir / f"{cls.EVENTS_FN}.{cls.DF_SAVE_FORMAT}"

    @classmethod
    def dynamic_measurements_fp(cls, save_dir: Path) -> Path:
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
    def process_events_and_measurements_df(
        cls,
        df: DF_T,
        event_type: str,
        columns_schema: dict[str, tuple[str, InputDataType]],
        ts_col: str | list[str],
    ) -> tuple[DF_T, DF_T | None]:
        """Performs the following pre-processing steps on an input events and measurements
        dataframe:

        1. Produces a unified timestamp column representing the minimum of passed timestamps, with the name,
           `'timestamp'`.
        2. Adds a categorical event type column with value `event_type`.
        3. Extracts and renames the columns present in `columns_schema`.
        4. Adds an integer `event_id` column.
        4. Splits the dataframe into an events dataframe, storing `event_id`, `subject_id`, `event_type`,
           and `timestamp`, and a `measurements` dataframe, storing `event_id` and all other data columns.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    @abc.abstractmethod
    def split_range_events_df(
        cls, df: DF_T, start_ts_col: str | list[str], end_ts_col: str | list[str]
    ) -> tuple[DF_T, DF_T, DF_T]:
        """Performs the following steps:

        1. Produces unified start and end timestamp columns representing the minimum of the passed start and end
           timestamps, respectively.
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
    def _inc_df_col(cls, df: DF_T, col: str, inc_by: int) -> DF_T:
        """Increments the values in a column by a given amount and returns a dataframe with the
        incremented column."""
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    @abc.abstractmethod
    def _concat_dfs(cls, dfs: list[DF_T]) -> DF_T:
        """Concatenates a list of dataframes into a single dataframe."""
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    @abc.abstractmethod
    def resolve_ts_col(
        cls, df: DF_T, ts_col: str | list[str], out_name: str = "timestamp"
    ) -> DF_T:
        """Produces an output column of type datetime that contains the minimum of the passed
        columns in `ts_col`"""
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    def build_subjects_dfs(cls, schema: InputDFSchema) -> tuple[DF_T, dict[Hashable, int]]:
        return cls._load_input_df(
            schema.input_df,
            [(schema.subject_id_col, InputDataType.CATEGORICAL)] + schema.columns_to_load,
            filter_on=schema.filter_on,
            subject_id_source_col=schema.subject_id_col,
        )

    def build_event_and_measurement_dfs(
        self,
        subject_ids_map: dict[Any, int],
        subject_id_col: str,
        subject_id_dtype: Any,
        schemas_by_df: dict[INPUT_DF_T, list[InputDFSchema]],
    ) -> tuple[DF_T, DF_T]:
        all_events_and_measurements = []
        event_types = []

        for df, schemas in self._tqdm(list(schemas_by_df.items()), desc="Input DataFrames"):
            all_columns = []

            all_columns.extend(itertools.chain.from_iterable(s.columns_to_load for s in schemas))

            try:
                df = self._load_input_df(
                    df, all_columns, subject_id_col, subject_ids_map, subject_id_dtype
                )
            except:
                print(f"Errored out reading\n{df}")
                raise

            for schema in schemas:
                if schema.filter_on:
                    df = self._filter_col_inclusion(schema.filter_on)
                match schema.type:
                    case InputDFType.EVENT:
                        df = self.resolve_ts_col(df, schema.ts_col, "timestamp")
                        all_events_and_measurements.append(
                            self.process_events_and_measurements_df(
                                df=df,
                                event_type=schema.event_type,
                                columns_schema=schema.unified_schema,
                            )
                        )
                        event_types.append(schema.event_type)
                    case InputDFType.RANGE:
                        df = self.resolve_ts_col(df, schema.start_ts_col, "start_time")
                        df = self.resolve_ts_col(df, schema.end_ts_col, "end_time")
                        for et, sp_df in zip(schema.event_type, self.split_range_events_df(df=df)):
                            all_events_and_measurements.append(
                                self.process_events_and_measurements_df(
                                    sp_df, columns_schema=schema.unified_schema, event_type=et
                                )
                            )
                        event_types.extend(schema.event_type)
                    case _:
                        raise ValueError(f"Invalid schema type {schema.type}.")

        all_events, all_measurements = [], []
        running_event_id_max = 0
        for event_type, (events, measurements) in zip(event_types, all_events_and_measurements):
            try:
                new_events = self._inc_df_col(events, "event_id", running_event_id_max)
            except:
                print(f"Failed to increment event_id on {event_type}")
                raise

            if len(new_events) == 0:
                print(f"Empty new events dataframe of type {event_type}!")
                continue

            all_events.append(new_events)
            if measurements is not None:
                all_measurements.append(
                    self._inc_df_col(measurements, "event_id", running_event_id_max)
                )

            running_event_id_max = all_events[-1]["event_id"].max() + 1

        return self._concat_dfs(all_events), self._concat_dfs(all_measurements)

    @classmethod
    def _get_metadata_model(
        cls,
        model_config: dict[str, Any],
        for_fit: bool = False,
    ) -> Any | tuple[dict[str, Any], Any]:
        """Fits a model as specified in `model_config` on the values in `vals`."""
        model_config = copy.deepcopy(model_config)
        if "cls" not in model_config:
            raise KeyError("Missing mandatory preprocessor class configuration parameter `'cls'`.")
        if model_config["cls"] not in cls.PREPROCESSORS:
            raise KeyError(
                f"Invalid preprocessor model class {model_config['cls']}! {cls} Options are "
                f"{', '.join(cls.PREPROCESSORS.keys())}"
            )

        model_cls = cls.PREPROCESSORS[model_config.pop("cls")]

        if not for_fit:
            return model_cls

        fit_config = copy.deepcopy(model_config)
        return fit_config, model_cls(**fit_config)

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
    def events_df(self) -> DF_T:
        if (not hasattr(self, "_events_df")) or self._events_df is None:
            events_fp = self.events_fp(self.config.save_dir)
            print(f"Loading events from {events_fp}...")
            self._events_df = self._read_df(events_fp)

        return self._events_df

    @events_df.setter
    def events_df(self, events_df: DF_T):
        self._events_df = events_df

    @property
    def subjects_df(self) -> DF_T:
        if (not hasattr(self, "_subjects_df")) or self._subjects_df is None:
            subjects_fp = self.subjects_fp(self.config.save_dir)
            print(f"Loading subjects from {subjects_fp}...")
            self._subjects_df = self._read_df(subjects_fp)

        return self._subjects_df

    @subjects_df.setter
    def subjects_df(self, subjects_df: DF_T):
        self._subjects_df = subjects_df

    @property
    def dynamic_measurements_df(self) -> DF_T:
        if (
            not hasattr(self, "_dynamic_measurements_df")
        ) or self._dynamic_measurements_df is None:
            dynamic_measurements_fp = self.dynamic_measurements_fp(self.config.save_dir)
            print(f"Loading dynamic_measurements from {dynamic_measurements_fp}...")
            self._dynamic_measurements_df = self._read_df(dynamic_measurements_fp)

        return self._dynamic_measurements_df

    @dynamic_measurements_df.setter
    def dynamic_measurements_df(self, dynamic_measurements_df: DF_T):
        self._dynamic_measurements_df = dynamic_measurements_df

    @classmethod
    def _load(cls, load_dir: Path, do_load_dfs: bool = False) -> "DatasetBase":
        # We need to load the base configuration file, the inferred metadata configuration objects,
        # the other base properties, and the actual dataframes.

        attrs_fp = load_dir / "E.pkl"

        if do_load_dfs:
            subjects_fp = cls.subjects_fp(load_dir)
            events_fp = cls.events_fp(load_dir)
            dynamic_measurements_fp = cls.dynamic_measurements_fp(load_dir)

            subjects_df = cls._read_df(subjects_fp)
            events_df = cls._read_df(events_fp)
            dynamic_measurements_df = cls._read_df(dynamic_measurements_fp)

            attrs_to_add = {
                "subjects_df": subjects_df,
                "events_df": events_df,
                "dynamic_measurements_df": dynamic_measurements_df,
            }
        else:
            attrs_to_add = {}

        return super()._load(attrs_fp, **attrs_to_add)

    def _save(self, **kwargs):
        # We need to save the base configuration file, the inferred metadata configuration objects,
        # the other base properties, and the actual dataframes.

        self.config.save_dir.mkdir(parents=True, exist_ok=True)

        super()._save(self.config.save_dir / "E.pkl", **kwargs)

        vocab_config_fp = self.config.save_dir / "vocabulary_config.json"

        if "do_overwrite" in kwargs:
            self.vocabulary_config.to_json_file(
                vocab_config_fp, do_overwrite=kwargs["do_overwrite"]
            )
        else:
            self.vocabulary_config.to_json_file(vocab_config_fp)

        subjects_fp = self.subjects_fp(self.config.save_dir)
        events_fp = self.events_fp(self.config.save_dir)
        dynamic_measurements_fp = self.dynamic_measurements_fp(self.config.save_dir)

        self._write_df(self.subjects_df, subjects_fp)
        self._write_df(self.events_df, events_fp)
        self._write_df(self.dynamic_measurements_df, dynamic_measurements_fp)

    def __init__(
        self,
        config: DatasetConfig,
        subjects_df: DF_T | None = None,
        events_df: DF_T | None = None,
        dynamic_measurements_df: DF_T | None = None,
        input_schema: DatasetSchema | None = None,
        **kwargs,
    ):
        """Builds the `Dataset` object.

        Args:
            `subjects_df` (`Optiona[DF_T]`, defaults to `None`):
                Per-subject data. The following columns are mandatory:
                    * `subject_id`, which must be unique.
            `events_df` (`Optional[DF_T]`, defaults to `None`):
                Per-event data. The following columns are mandatory:
                    * `subject_id`: The ID of the subject of the row.
                    * `event_type`: The type of the row's event.
                    * `timestamp`: The timestamp of the row's event.
            `dynamic_measurements_df` (`Optional[DF_T]`, defaults to `None`):
                Dynamic measurements data. The following columns are mandatory:
                    * `event_id`: The ID of the event to which this measurement is associated.
            `config` (`DatasetConfig`):
                Configuration objects for this dataset. Largely details how metadata should be processed.
        """
        super().__init__(**kwargs)

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

    def _validate_and_set_initial_properties(
        self, subjects_df, events_df, dynamic_measurements_df
    ):
        self.subject_ids = []
        self.event_types = []
        self.n_events_per_subject = {}

        (
            self.subjects_df,
            self.events_df,
            self.dynamic_measurements_df,
        ) = self._validate_initial_dfs(subjects_df, events_df, dynamic_measurements_df)

        if self.events_df is not None:
            self.agg_by_time()
            self.sort_events()
        self._update_subject_event_properties()

    @abc.abstractmethod
    def _validate_initial_dfs(
        self, subjects_df: DF_T, events_df: DF_T, dynamic_measurements_df: DF_T
    ) -> tuple[DF_T, DF_T, DF_T]:
        raise NotImplementedError("This method must be implemented by a subclass.")

    @abc.abstractmethod
    def _update_subject_event_properties(self):
        """Must update:

        self.event_types = [e for e, _ in Counter(self.events_df.event_type).most_common()]
        self.subject_ids = set(self.events_df.subject_id)
        self.n_events_per_subject = self.events_df.groupby('subject_id').timestamp.count().to_dict()
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    @TimeableMixin.TimeAs
    def filter_subjects(self):
        if self.config.min_events_per_subject is None:
            return

        subjects_to_keep = [
            s
            for s, n in self.n_events_per_subject.items()
            if n >= self.config.min_events_per_subject
        ]
        self.subjects_df = self._filter_col_inclusion(
            self.subjects_df, {"subject_id": subjects_to_keep}
        )
        self.events_df = self._filter_col_inclusion(
            self.events_df, {"subject_id": subjects_to_keep}
        )
        self.dynamic_measurements_df = self._filter_col_inclusion(
            self.dynamic_measurements_df, {"event_id": list(self.events_df["event_id"])}
        )

    @TimeableMixin.TimeAs
    @abc.abstractmethod
    def agg_by_time(self):
        """Aggregates the events_df by subject_id, timestamp, combining event_types into grouped
        categories, tracking all associated metadata.

        Note that no numerical aggregation (e.g., mean, etc.) happens here; all data is retained,
        and only dynamic measurement event IDs are updated.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    @TimeableMixin.TimeAs
    @abc.abstractmethod
    def sort_events(self):
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
            `split_fracs` (`Sequence[float]`, each split_frac must be >= 0 and the sum must be <= 1):
                The fractional sizes of the desired splits. If it sums to < 1, the remainder will be tracked
                in an extra split at the end of the list.
            `split_names` (`Sequence[str]`, *optional*, defaults to None):
                If specified, assigns the passed names to each split. Must be of the same size as
                `split_fracs` (after it is expanded to sum to 1 if necessary). If unset, and there are two
                splits, it defaults to [`train`, `held_out`]. If there are three, it defaults to `['train',
                'tuning', 'held_out']. If more than 3, it defaults to `['split_0', 'split_1', ...]`. Split
                names of `train`, `tuning`, and `held_out` have special significance and are used elsewhere in
                the model, so if `split_names` does not reflect those other things may not work down the line.
        """
        split_fracs = list(split_fracs)

        assert min(split_fracs) >= 0 and max(split_fracs) <= 1
        assert sum(split_fracs) <= 1
        if sum(split_fracs) < 1:
            split_fracs.append(1 - sum(split_fracs))

        if split_names is None:
            if len(split_fracs) == 2:
                split_names = ["train", "held_out"]
            elif len(split_fracs) == 3:
                split_names = ["train", "tuning", "held_out"]
            else:
                split_names = [f"split_{i}" for i in range(len(split_fracs))]
        else:
            assert len(split_names) == len(split_fracs)

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
    def _filter_col_inclusion(
        cls, df: DF_T, col_inclusion_targets: dict[str, bool | Sequence[Any]]
    ) -> DF_T:
        """Filters the given dataframe to only the rows such that the column `col` is in
        incl_target."""
        raise NotImplementedError("This method must be implemented by a subclass.")

    # Special accessors for train, tuning, and held-out splits.
    @property
    def train_subjects_df(self) -> DF_T:
        return self._filter_col_inclusion(
            self.subjects_df, {"subject_id": self.split_subjects["train"]}
        )

    @property
    def tuning_subjects_df(self) -> DF_T:
        return self._filter_col_inclusion(
            self.subjects_df, {"subject_id": self.split_subjects["tuning"]}
        )

    @property
    def held_out_subjects_df(self) -> DF_T:
        return self._filter_col_inclusion(
            self.subjects_df, {"subject_id": self.split_subjects["held_out"]}
        )

    @property
    def train_events_df(self) -> DF_T:
        return self._filter_col_inclusion(
            self.events_df, {"subject_id": self.split_subjects["train"]}
        )

    @property
    def tuning_events_df(self) -> DF_T:
        return self._filter_col_inclusion(
            self.events_df, {"subject_id": self.split_subjects["tuning"]}
        )

    @property
    def held_out_events_df(self) -> DF_T:
        return self._filter_col_inclusion(
            self.events_df, {"subject_id": self.split_subjects["held_out"]}
        )

    @TimeableMixin.TimeAs
    def _filter_measurements_df(
        self,
        event_types: Sequence[str] | None = None,
        event_type: str | None = None,
        splits: Sequence[str] | None = None,
        split: str | None = None,
        subject_ids: Sequence[Hashable] | None = None,
        subject_id: Hashable | None = None,
    ) -> DF_T:
        """Returns a subframe of `self.dynamic_measurements_df` corresponding to events following
        input constraints. The index returned is in the same order as
        `self.dynamic_measurements_df` as of the time of the function call.

        Args:
            * `event_types` (`Optional[Sequence[str]]`), *optional*, defaults to `None`:
                If specified, the returned index will only contain metadata events corresponding to events of
                a type in `event_types`.
                Cannot be simultanesouly set with `event_type`.
            * `event_type` (`Optional[str]`), *optional*, defaults to `None`:
                If specified, the returned index will only contain metadata events corresponding to events of
                type `event_type`.
                Cannot be simultanesouly set with `event_types`.
            * `splits` (`Optional[Sequence[str]]`), *optional*, defaults to `None`:
                If specified, the returned index will only contain metadata events corresponding to events of
                a subject in a split in `splits`.
                Cannot be simultanesouly set with `split`.
            * `split` (`Optional[str]`), *optional*, defaults to `None`:
                If specified, the returned index will only contain metadata events corresponding to events of
                a subject in split `split`.
                Cannot be simultanesouly set with `split`.
            * `subject_ids` (`Optional[Sequence[Hashable]]`), *optional*, defaults to `None`:
                If specified, the returned index will only contain metadata events corresponding to events of
                a subject in `subject_ids`.
                Cannot be simultanesouly set with `subject_id`.
            * `subject_id` (`Optional[Hashable]`), *optional*, defaults to `None`:
                If specified, the returned index will only contain metadata events corresponding to events of
                subject `subject_id`.
                Cannot be simultanesouly set with `subject_ids`.
        Returns:
        """
        assert not ((subject_id is not None) and (subject_ids is not None))
        assert not ((event_type is not None) and (event_types is not None))
        assert not (
            ((subject_id is not None) or (subject_ids is not None))
            and ((split is not None) or (splits is not None))
        )

        if split is not None:
            subject_ids = self.split_subjects[split]
        elif splits is not None:
            subject_ids = list(
                set(itertools.chain.from_iterable(self.split_subjects[sp] for sp in splits))
            )

        filter_cols = {}
        if event_type is not None:
            filter_cols["event_type"] = [event_type]
        elif event_types is not None:
            filter_cols["event_type"] = event_types
        if subject_id is not None:
            filter_cols["subject_id"] = [subject_id]
        elif subject_ids is not None:
            filter_cols["subject_id"] = subject_ids

        event_ids = self._filter_col_inclusion(self.events_df, filter_cols)["event_id"]
        return self._filter_col_inclusion(
            self.dynamic_measurements_df, {"event_id": list(event_ids)}
        )

    @TimeableMixin.TimeAs
    def preprocess_measurements(self):
        """Fits all metadata over the train set, then transforms all metadata."""
        self.filter_subjects()
        self.add_time_dependent_measurements()
        self.fit_measurements()
        self.transform_measurements()

    @TimeableMixin.TimeAs
    @abc.abstractmethod
    def add_time_dependent_measurements(self):
        """Adds time-dependent columns to events_df."""
        raise NotImplementedError("This method must be implemented by a subclass.")

    @TimeableMixin.TimeAs
    def _get_source_df(
        self, config: MeasurementConfig, do_only_train: bool = True
    ) -> tuple[str, DF_T]:
        match config.temporality:
            case TemporalityType.DYNAMIC:
                source_attr = "dynamic_measurements_df"
                source_id = "measurement_id"
                if do_only_train:
                    source_df = self._filter_measurements_df(
                        event_types=config.present_in_event_types, split="train"
                    )
                else:
                    source_df = self._filter_measurements_df(
                        event_types=config.present_in_event_types
                    )

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
    @abc.abstractmethod
    def _get_valid_event_types(self) -> dict[str, list[str]]:
        raise NotImplementedError("This method must be implemented by a subclass.")

    @TimeableMixin.TimeAs
    def fit_measurements(self):
        """Fits preprocessing models, variables, and vocabularies over all metadata, including both
        numerical and categorical columns, over the training split.

        Details of pre-processing are dictated by `self.config`.
        """
        self._is_fit = False

        # Get valid event types per measure
        event_types_obs_per_measure = self._get_valid_event_types()

        for measure, config in self.config.measurement_configs.items():
            if config.is_dropped:
                continue

            self.inferred_measurement_configs[measure] = copy.deepcopy(config)
            config = self.inferred_measurement_configs[measure]

            # Add inferred event type limitations:
            if (config.temporality == TemporalityType.DYNAMIC) and (
                config.present_in_event_types is None
            ):
                config.present_in_event_types = event_types_obs_per_measure.get(measure, None)

            _, _, source_df = self._get_source_df(config, do_only_train=True)

            if measure not in source_df:
                config.drop()
                continue

            total_possible, total_observed = self._total_possible_and_observed(
                measure, config, source_df
            )
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
                    config.measurement_metadata = self._fit_measurement_metadata(
                        measure, config, source_df
                    )
                except BaseException as e:
                    raise ValueError(
                        f"Fitting measurement metadata failed for measure {measure}!"
                    ) from e

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
        raise NotImplementedError("This method must be implemented by a subclass.")

    @abc.abstractmethod
    def _fit_measurement_metadata(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> pd.DataFrame:
        raise NotImplementedError("This method must be implemented by a subclass.")

    @TimeableMixin.TimeAs
    @abc.abstractmethod
    def _fit_vocabulary(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> Vocabulary:
        raise NotImplementedError("This method must be implemented by a subclass.")

    @TimeableMixin.TimeAs
    def transform_measurements(self):
        """Transforms the entire dataset metadata given the fit pre-processors."""
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

            self.update_attr_df(source_attr, id_col, source_df, updated_cols)

    @TimeableMixin.TimeAs
    @abc.abstractmethod
    def update_attr_df(self, attr: str, id_col: str, df: DF_T, cols_to_update: list[str]):
        """Updates the attribute `attr` with the dataframe `df`."""
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
            `measure` (`str`): The column name of the governing measurement to transform.
            `config` (`MeasurementConfig`): The configuration object governing this measure.
            `source_df` (`DF_T`): The dataframe object containing the measure to be transformed.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    @TimeableMixin.TimeAs
    @abc.abstractmethod
    def _transform_categorical_measurement(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> DF_T:
        """Transforms the categorical measurement `measure` according to config `config`.

        Performs the following steps:
            1. Converts the elements to categorical column types according to the learned vocabularies.

        Args:
            `measure` (`str`): The column name of the governing measurement to transform.
            `config` (`MeasurementConfig`): The configuration object governing this measure.
            `source_df` (`DF_T`): The dataframe object containing the measure to be transformed.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    @property
    def has_static_measurements(self):
        return (self.subjects_df is not None) and any(
            cfg.temporality == TemporalityType.STATIC for cfg in self.measurement_configs.values()
        )

    @property
    def measurement_configs(self):
        """Returns the inferred configuration objects if the metadata preprocessors have been fit,
        otherwise the passed configuration objects."""
        assert self._is_fit
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
    def cache_deep_learning_representation(self, subjects_per_output_file: int | None = None):
        """Produces a cached, batched representation of the dataset suitable for deep learning
        applications and writes it to cache_fp in the specified format."""

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
                self._write_df(split_cached_df, fp)

    @property
    def vocabulary_config(self) -> VocabularyConfig:
        event_types_per_measurement = {}
        measurements_per_generative_mode = defaultdict(list)
        measurements_per_generative_mode[DataModality.SINGLE_LABEL_CLASSIFICATION].append(
            "event_type"
        )
        for m, cfg in self.measurement_configs.items():
            if cfg.temporality != TemporalityType.DYNAMIC:
                continue

            measurements_per_generative_mode[cfg.modality].append(m)
            if cfg.modality == DataModality.MULTIVARIATE_REGRESSION:
                measurements_per_generative_mode[DataModality.MULTI_LABEL_CLASSIFICATION].append(m)

            if cfg.present_in_event_types is None:
                event_types_per_measurement[m] = self.event_types
            else:
                event_types_per_measurement[m] = list(cfg.present_in_event_types)

        return VocabularyConfig(
            vocab_sizes_by_measurement={
                m: len(idxmap) for m, idxmap in self.measurement_idxmaps.items()
            },
            vocab_offsets_by_measurement=self.unified_vocabulary_offsets,
            measurements_idxmap=self.unified_measurements_idxmap,
            event_types_idxmap=self.unified_vocabulary_idxmap["event_type"],
            measurements_per_generative_mode=dict(measurements_per_generative_mode),
            event_types_per_measurement=event_types_per_measurement,
        )

    @property
    def unified_measurements_vocab(self) -> list[str]:
        return ["event_type"] + list(sorted(self.measurement_configs.keys()))

    @property
    def unified_measurements_idxmap(self) -> dict[str, int]:
        return {m: i + 1 for i, m in enumerate(self.unified_measurements_vocab)}

    @property
    def unified_vocabulary_offsets(self) -> dict[str, int]:
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
        """
        Produces a format with the below syntax:

        ```
        subject_id | start_time | batched_representation
        1          | 2019-01-01 | batch_1,
        ...

        Batch Representation:
          N = number of time points
          M = maximum number of dynamic measurements at any time point
          K = number of static measurements
        batch_1 = {
          'time': [...] float, (N,), minutes since start_time of event. No missing values.
          'dynamic_indices': [[...]] int, (N, M), indices of dynamic measurements. 0 Iff missing.
          'dynamic_values': [[...]] float, (N, M), values of dynamic measurements. 0 If missing.
          'dynamic_measurement_indices': [[...]] int, (N, M), indices of dynamic measurements. 0 Iff missing.
          'static_indices': [...] int, (K,), indices of static measurements. No missing values.
          'static_measurement_indices': [...] int, (K,), indices of static measurements. No missing values.
        ```
        """

        raise NotImplementedError("This method must be implemented by a subclass.")

    @abc.abstractmethod
    def denormalize(self, events_df: DF_T, col: str) -> DF_T:
        """Un-normalizes the column `col` in df `events_df`."""
        raise NotImplementedError("This method must be implemented by a subclass.")

    def describe(
        self,
        do_print_measurement_summaries: bool = True,
        viz_config: Visualizer | None = None,
    ) -> list[Figure] | None:
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
        """Visualizes the dataset, along the following axes:

        1. By time
        2. By subject age at event
        3. Overall histograms
        """

        if viz_config.subset_size is not None:
            viz_config.subset_random_seed = self._seed(
                seed=viz_config.subset_random_seed, key="visualize"
            )

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
            events_df = self.denormalize(events_df, viz_config.age_col)

        figs = viz_config.plot(subjects_df, events_df, dynamic_measurements_df)
        return figs

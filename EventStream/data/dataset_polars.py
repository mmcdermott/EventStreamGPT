"""The polars implementation of the Dataset class.

Attributes:
    INPUT_DF_T: The types of supported input dataframes, which includes paths, pandas dataframes, polars
        dataframes, or queries.
    DF_T: The types of supported dataframes, which include polars lazyframes, dataframes, expressions, or
        series.
"""

import dataclasses
import multiprocessing
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import polars as pl
from mixins import TimeableMixin

from ..utils import lt_count_or_proportion
from .config import MeasurementConfig
from .dataset_base import DatasetBase
from .preprocessing import Preprocessor, StandardScaler, StddevCutoffOutlierDetector
from .types import (
    DataModality,
    InputDataType,
    NumericDataModalitySubtype,
    TemporalityType,
)
from .vocabulary import Vocabulary

# We need to do this so that categorical columns can be reliably used via category names.
pl.enable_string_cache(True)


@dataclasses.dataclass(frozen=True)
class Query:
    """A structure for database query based input dataframes.

    Args:
        connection_uri: The connection URI for the database. This is in the `connectorx`_ format.
        query: The query to be run over the database. It can be specified either as a direct string, a path to
            a file on disk containing the query in txt format, or a list of said options.
        partition_on: If the query should be partitioned, on what column should it be partitioned? See the
            `polars documentation`_ for more details.
        partition_num: If the query should be partitioned, into how many partitions should it be divided? See
            the `polars documentation`_ for more details.
        protocol: The `connectorx`_ backend protocol.

    .. connectorx_: https://github.com/sfu-db/connector-x
    .. polars documentation_: https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.read_database.html
    """  # noqa E501

    connection_uri: str
    query: str | Path | list[str | Path]
    partition_on: str | None = None
    partition_num: int | None = None
    protocol: str = "binary"

    def __str__(self):
        return f'Query("{self.query}")'


DF_T = Union[pl.LazyFrame, pl.DataFrame, pl.Expr, pl.Series]
INPUT_DF_T = Union[Path, pd.DataFrame, pl.DataFrame, Query]


class Dataset(DatasetBase[DF_T, INPUT_DF_T]):
    """The polars specific implementation of the dataset.

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

    # Dictates what models can be fit on numerical metadata columns, for both outlier detection and
    # normalization.
    PREPROCESSORS: dict[str, Preprocessor] = {
        # Outlier Detectors
        "stddev_cutoff": StddevCutoffOutlierDetector,
        # Normalizers
        "standard_scaler": StandardScaler,
    }
    """A dictionary containing the valid pre-processors that can be used by this model class."""

    METADATA_SCHEMA = {
        "drop_upper_bound": pl.Float64,
        "drop_upper_bound_inclusive": pl.Boolean,
        "drop_lower_bound": pl.Float64,
        "drop_lower_bound_inclusive": pl.Boolean,
        "censor_upper_bound": pl.Float64,
        "censor_lower_bound": pl.Float64,
        "outlier_model": lambda outlier_params_schema: pl.Struct(outlier_params_schema),
        "normalizer": lambda normalizer_params_schema: pl.Struct(normalizer_params_schema),
        "value_type": pl.Categorical,
    }
    """The Polars schema of the numerical measurement metadata dataframes which track fit parameters."""

    @staticmethod
    def get_smallest_valid_int_type(num: int | float | pl.Expr) -> pl.DataType:
        """Returns the smallest valid unsigned integral type for an ID variable with `num` unique options.

        Args:
            num: The number of IDs that must be uniquely expressed.

        Raises:
            ValueError: If there is no unsigned int type big enough to express the passed number of ID
                variables.

        Examples:
            >>> import polars as pl
            >>> Dataset.get_smallest_valid_int_type(num=1)
            UInt8
            >>> Dataset.get_smallest_valid_int_type(num=2**8-1)
            UInt16
            >>> Dataset.get_smallest_valid_int_type(num=2**16-1)
            UInt32
            >>> Dataset.get_smallest_valid_int_type(num=2**32-1)
            UInt64
            >>> Dataset.get_smallest_valid_int_type(num=2**64-1)
            Traceback (most recent call last):
                ...
            ValueError: Value is too large to be expressed as an int!
        """
        if num >= (2**64) - 1:
            raise ValueError("Value is too large to be expressed as an int!")
        if num >= (2**32) - 1:
            return pl.UInt64
        elif num >= (2**16) - 1:
            return pl.UInt32
        elif num >= (2**8) - 1:
            return pl.UInt16
        else:
            return pl.UInt8

    @classmethod
    def _load_input_df(
        cls,
        df: INPUT_DF_T,
        columns: list[tuple[str, InputDataType | tuple[InputDataType, str]]],
        subject_id_col: str | None = None,
        subject_ids_map: dict[Any, int] | None = None,
        subject_id_dtype: Any | None = None,
        filter_on: dict[str, bool | list[Any]] | None = None,
        subject_id_source_col: str | None = None,
    ) -> DF_T | tuple[DF_T, str]:
        """Loads an input dataframe into the format expected by the processing library."""
        if subject_id_col is None:
            if subject_ids_map is not None:
                raise ValueError("Must not set subject_ids_map if subject_id_col is not set")
            if subject_id_dtype is not None:
                raise ValueError("Must not set subject_id_dtype if subject_id_col is not set")
        else:
            if subject_ids_map is None:
                raise ValueError("Must set subject_ids_map if subject_id_col is set")
            if subject_id_dtype is None:
                raise ValueError("Must set subject_id_dtype if subject_id_col is set")

        match df:
            case (str() | Path()) as fp:
                if not isinstance(fp, Path):
                    fp = Path(fp)

                if fp.suffix == ".csv":
                    df = pl.scan_csv(df, null_values="")
                elif fp.suffix == ".parquet":
                    df = pl.scan_parquet(df)
                else:
                    raise ValueError(f"Can't read dataframe from file of suffix {fp.suffix}")
            case pd.DataFrame():
                df = pl.from_pandas(df, include_index=True).lazy()
            case pl.DataFrame():
                df = df.lazy()
            case pl.LazyFrame():
                pass
            case Query() as q:
                query = q.query
                if not isinstance(query, (list, tuple)):
                    query = [query]

                out_query = []
                for qq in query:
                    if type(qq) is Path:
                        with open(qq) as f:
                            qq = f.read()
                    elif type(qq) is not str:
                        raise ValueError(f"{type(qq)} is an invalid query.")
                    out_query.append(qq)

                if len(out_query) == 1:
                    partition_kwargs = {
                        "partition_on": subject_id_col if q.partition_on is None else q.partition_on,
                        "partition_num": (
                            multiprocessing.cpu_count() if q.partition_num is None else q.partition_num
                        ),
                    }
                elif q.partition_on is not None or q.partition_num is not None:
                    raise ValueError(
                        "Partitioning ({q.partition_on}, {q.partition_num}) not supported when "
                        "passing multiple queries ({out_query})"
                    )
                else:
                    partition_kwargs = {}

                df = pl.read_database(
                    query=out_query,
                    connection_uri=q.connection_uri,
                    protocol=q.protocol,
                    **partition_kwargs,
                ).lazy()
            case _:
                raise TypeError(f"Input dataframe `df` is of invalid type {type(df)}!")

        col_exprs = []

        df = df.select(pl.all().shrink_dtype())

        if filter_on:
            df = cls._filter_col_inclusion(df, filter_on)

        if subject_id_source_col is not None:
            internal_subj_key = "subject_id"
            while internal_subj_key in df.columns:
                internal_subj_key = f"_{internal_subj_key}"
            df = df.with_row_count(internal_subj_key)
            col_exprs.append(internal_subj_key)
        else:
            assert subject_id_col is not None
            df = df.with_columns(pl.col(subject_id_col).cast(pl.Utf8).cast(pl.Categorical))
            df = cls._filter_col_inclusion(df, {subject_id_col: list(subject_ids_map.keys())})
            col_exprs.append(
                pl.col(subject_id_col).map_dict(subject_ids_map).cast(subject_id_dtype).alias("subject_id")
            )

        for in_col, out_dt in columns:
            match out_dt:
                case InputDataType.FLOAT:
                    col_exprs.append(pl.col(in_col).cast(pl.Float32, strict=False))
                case InputDataType.CATEGORICAL:
                    col_exprs.append(pl.col(in_col).cast(pl.Utf8).cast(pl.Categorical))
                case InputDataType.BOOLEAN:
                    col_exprs.append(pl.col(in_col).cast(pl.Boolean, strict=False))
                case InputDataType.TIMESTAMP:
                    col_exprs.append(pl.col(in_col).cast(pl.Datetime, strict=True))
                case (InputDataType.TIMESTAMP, str() as ts_format):
                    col_exprs.append(pl.col(in_col).str.strptime(pl.Datetime, ts_format, strict=False))
                case _:
                    raise ValueError(f"Invalid out data type {out_dt}!")

        if subject_id_source_col is not None:
            df = df.select(col_exprs).collect()

            ID_map = {o: n for o, n in zip(df[subject_id_source_col], df[internal_subj_key])}
            df = df.with_columns(pl.col(internal_subj_key).alias("subject_id"))
            return df, ID_map
        else:
            return df.select(col_exprs)

    @classmethod
    def _rename_cols(cls, df: DF_T, to_rename: dict[str, str]) -> DF_T:
        """Renames the columns in df according to the {in_name: out_name}s specified in to_rename.

        Args:
            df: The dataframe whose columns should be renamed.
            to_rename: A mapping of in column names to out column names.

        Returns: The dataframe with columns renamed.

        Examples:
            >>> import polars as pl
            >>> df = pl.DataFrame({'a': [1, 2, 3], 'b': ['foo', None, 'bar'], 'c': [1., 2.0, float('inf')]})
            >>> Dataset._rename_cols(df, {'a': 'a', 'b': 'biz'})
            shape: (3, 3)
            ┌─────┬──────┬─────┐
            │ a   ┆ biz  ┆ c   │
            │ --- ┆ ---  ┆ --- │
            │ i64 ┆ str  ┆ f64 │
            ╞═════╪══════╪═════╡
            │ 1   ┆ foo  ┆ 1.0 │
            │ 2   ┆ null ┆ 2.0 │
            │ 3   ┆ bar  ┆ inf │
            └─────┴──────┴─────┘
        """

        return df.rename(to_rename)

    @classmethod
    def _resolve_ts_col(cls, df: DF_T, ts_col: str | list[str], out_name: str = "timestamp") -> DF_T:
        match ts_col:
            case list():
                ts_expr = pl.min(ts_col)
                ts_to_drop = [c for c in ts_col if c != out_name]
            case str():
                ts_expr = pl.col(ts_col)
                ts_to_drop = [ts_col] if ts_col != out_name else []

        return df.with_columns(ts_expr.alias(out_name)).drop(ts_to_drop)

    @classmethod
    def _process_events_and_measurements_df(
        cls,
        df: DF_T,
        event_type: str,
        columns_schema: dict[str, tuple[str, InputDataType]],
    ) -> tuple[DF_T, DF_T | None]:
        """Performs the following pre-processing steps on an input events and measurements
        dataframe:

        1. Adds a categorical event type column with value `event_type`.
        2. Extracts and renames the columns present in `columns_schema`.
        3. Adds an integer `event_id` column.
        4. Splits the dataframe into an events dataframe, storing `event_id`, `subject_id`, `event_type`,
           and `timestamp`, and a `measurements` dataframe, storing `event_id` and all other data columns.
        """

        cols_select_exprs = [
            "timestamp",
            "subject_id",
        ]
        if event_type.startswith("COL:"):
            event_type_col = event_type[len("COL:") :]
            cols_select_exprs.append(pl.col(event_type_col).cast(pl.Categorical).alias("event_type"))
        else:
            cols_select_exprs.append(pl.lit(event_type).cast(pl.Categorical).alias("event_type"))

        for in_col, (out_col, _) in columns_schema.items():
            cols_select_exprs.append(pl.col(in_col).alias(out_col))

        df = (
            df.filter(pl.col("timestamp").is_not_null() & pl.col("subject_id").is_not_null())
            .select(cols_select_exprs)
            .unique()
            .with_row_count("event_id")
        )

        events_df = df.select("event_id", "subject_id", "timestamp", "event_type")

        if len(df.columns) > 4:
            dynamic_measurements_df = df.drop("subject_id", "timestamp", "event_type")
        else:
            dynamic_measurements_df = None

        return events_df, dynamic_measurements_df

    @classmethod
    def _split_range_events_df(cls, df: DF_T) -> tuple[DF_T, DF_T, DF_T]:
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

        df = df.filter(pl.col("start_time") <= pl.col("end_time"))

        eq_df = df.filter(pl.col("start_time") == pl.col("end_time"))
        ne_df = df.filter(pl.col("start_time") != pl.col("end_time"))

        st_col, end_col = pl.col("start_time").alias("timestamp"), pl.col("end_time").alias("timestamp")
        drop_cols = ["start_time", "end_time"]
        return (
            eq_df.with_columns(st_col).drop(drop_cols),
            ne_df.with_columns(st_col).drop(drop_cols),
            ne_df.with_columns(end_col).drop(drop_cols),
        )

    @classmethod
    def _inc_df_col(cls, df: DF_T, col: str, inc_by: int) -> DF_T:
        """Increments the values in a column by a given amount and returns a dataframe with the incremented
        column."""
        return df.with_columns(pl.col(col) + inc_by).collect()

    @classmethod
    def _concat_dfs(cls, dfs: list[DF_T]) -> DF_T:
        """Concatenates a list of dataframes into a single dataframe."""
        return pl.concat(dfs, how="diagonal")

    @classmethod
    def _read_df(cls, fp: Path, **kwargs) -> DF_T:
        return pl.read_parquet(fp)

    @classmethod
    def _write_df(cls, df: DF_T, fp: Path, **kwargs):
        do_overwrite = kwargs.get("do_overwrite", False)

        if not do_overwrite and fp.is_file():
            raise FileExistsError(f"{fp} exists and do_overwrite is {do_overwrite}!")

        df.write_parquet(fp)

    def get_metadata_schema(self, config: MeasurementConfig) -> dict[str, pl.DataType]:
        schema = {
            "value_type": self.METADATA_SCHEMA["value_type"],
        }

        if self.config.outlier_detector_config is not None:
            M = self._get_preprocessing_model(self.config.outlier_detector_config, for_fit=False)
            schema["outlier_model"] = self.METADATA_SCHEMA["outlier_model"](M.params_schema())
        if self.config.normalizer_config is not None:
            M = self._get_preprocessing_model(self.config.normalizer_config, for_fit=False)
            schema["normalizer"] = self.METADATA_SCHEMA["normalizer"](M.params_schema())

        metadata = config.measurement_metadata
        if metadata is None:
            return schema

        for col in (
            "drop_upper_bound",
            "drop_lower_bound",
            "censor_upper_bound",
            "censor_lower_bound",
            "drop_upper_bound_inclusive",
            "drop_lower_bound_inclusive",
        ):
            if col in metadata:
                schema[col] = self.METADATA_SCHEMA[col]

        return schema

    @staticmethod
    def drop_or_censor(
        col: pl.Expr,
        drop_lower_bound: pl.Expr | None = None,
        drop_lower_bound_inclusive: pl.Expr | None = None,
        drop_upper_bound: pl.Expr | None = None,
        drop_upper_bound_inclusive: pl.Expr | None = None,
        censor_lower_bound: pl.Expr | None = None,
        censor_upper_bound: pl.Expr | None = None,
        **ignored_kwargs,
    ) -> pl.Expr:
        """Appropriately either drops (returns np.NaN) or censors (returns the censor value) the value `val`
        based on the bounds in `row`.

        TODO(mmd): could move this code to an outlier model in Preprocessing and have it be one that is
        pre-set in metadata.

        Args:
            val: The value to drop, censor, or return unchanged.
            drop_lower_bound: A lower bound such that if `val` is either below or at or below this level,
                `np.NaN` will be returned. If `None` or `np.NaN`, no bound will be applied.
            drop_lower_bound_inclusive: If `True`, returns `np.NaN` if ``val <= row['drop_lower_bound']``.
                Else, returns `np.NaN` if ``val < row['drop_lower_bound']``.
            drop_upper_bound: An upper bound such that if `val` is either above or at or above this level,
                `np.NaN` will be returned. If `None` or `np.NaN`, no bound will be applied.
            drop_upper_bound_inclusive: If `True`, returns `np.NaN` if ``val >= row['drop_upper_bound']``.
                Else, returns `np.NaN` if ``val > row['drop_upper_bound']``.
            censor_lower_bound: A lower bound such that if `val` is below this level but above
                `drop_lower_bound`, `censor_lower_bound` will be returned. If `None` or `np.NaN`, no bound
                will be applied.
            censor_upper_bound: An upper bound such that if `val` is above this level but below
                `drop_upper_bound`, `censor_upper_bound` will be returned. If `None` or `np.NaN`, no bound
                will be applied.
        """

        conditions = []

        if drop_lower_bound is not None:
            conditions.append(
                (
                    (col < drop_lower_bound) | ((col == drop_lower_bound) & drop_lower_bound_inclusive),
                    np.NaN,
                )
            )

        if drop_upper_bound is not None:
            conditions.append(
                (
                    (col > drop_upper_bound) | ((col == drop_upper_bound) & drop_upper_bound_inclusive),
                    np.NaN,
                )
            )

        if censor_lower_bound is not None:
            conditions.append((col < censor_lower_bound, censor_lower_bound))
        if censor_upper_bound is not None:
            conditions.append((col > censor_upper_bound, censor_upper_bound))

        if not conditions:
            return col

        expr = pl.when(conditions[0][0]).then(conditions[0][1])
        for cond, val in conditions[1:]:
            expr = expr.when(cond).then(val)
        return expr.otherwise(col)

    @staticmethod
    def _validate_id_col(id_col: pl.Series) -> tuple[pl.Series, pl.datatypes.DataTypeClass]:
        """Validate the given ID column.

        This validates that the ID column is unique, integral, strictly positive, and returns it converted to
        the smallest valid dtype.

        Args:
            id_col (pl.Expr): The ID column to validate.

        Returns:
            pl.Expr: The validated ID column.

        Raises:
            ValueError: If the ID column is not unique.
        """

        if not id_col.is_unique().all():
            raise ValueError(f"ID column {id_col.name} is not unique!")
        match id_col.dtype:
            case pl.Float32 | pl.Float64:
                if not (id_col == id_col.round(0)).all() and (id_col >= 0).all():
                    raise ValueError(f"ID column {id_col.name} is not a non-negative integer type!")
            case pl.Int8 | pl.Int16 | pl.Int32 | pl.Int64:
                if not (id_col >= 0).all():
                    raise ValueError(f"ID column {id_col.name} is not a non-negative integer type!")
            case pl.UInt8 | pl.UInt16 | pl.UInt32 | pl.UInt64:
                pass
            case _:
                raise ValueError(f"ID column {id_col.name} is not a non-negative integer type!")

        max_val = id_col.max()
        dt = Dataset.get_smallest_valid_int_type(max_val)

        id_col = id_col.cast(dt)

        return id_col, dt

    def _validate_initial_df(
        self,
        source_df: DF_T | None,
        id_col_name: str,
        valid_temporality_type: TemporalityType,
        linked_id_cols: dict[str, pl.datatypes.DataTypeClass] | None = None,
    ) -> tuple[DF_T | None, pl.datatypes.DataTypeClass]:
        if source_df is None:
            return None, None

        if linked_id_cols:
            for id_col, id_col_dt in linked_id_cols.items():
                if id_col not in source_df:
                    raise ValueError(f"Missing mandatory linkage col {id_col}")
                source_df = source_df.with_columns(pl.col(id_col).cast(id_col_dt))

        if id_col_name not in source_df:
            source_df = source_df.with_row_count(name=id_col_name)

        id_col, id_col_dt = self._validate_id_col(source_df.get_column(id_col_name))
        source_df = source_df.with_columns(id_col)

        for col, cfg in self.config.measurement_configs.items():
            match cfg.modality:
                case DataModality.DROPPED:
                    continue
                case DataModality.UNIVARIATE_REGRESSION:
                    cat_col, val_col = None, col
                case DataModality.MULTIVARIATE_REGRESSION:
                    cat_col, val_col = col, cfg.values_column
                case _:
                    cat_col, val_col = col, None

            if cat_col is not None and cat_col in source_df:
                if cfg.temporality != valid_temporality_type:
                    raise ValueError(f"Column {cat_col} found in dataframe of wrong temporality")

                source_df = source_df.with_columns(pl.col(cat_col).cast(pl.Utf8).cast(pl.Categorical))

            if val_col is not None and val_col in source_df:
                if cfg.temporality != valid_temporality_type:
                    raise ValueError(f"Column {val_col} found in dataframe of wrong temporality")

                source_df = source_df.with_columns(pl.col(val_col).cast(pl.Float64))

        return source_df, id_col_dt

    def _validate_initial_dfs(
        self,
        subjects_df: DF_T | None,
        events_df: DF_T | None,
        dynamic_measurements_df: DF_T | None,
    ) -> tuple[DF_T | None, DF_T | None, DF_T | None]:
        """Validate and preprocess the given subjects, events, and dynamic_measurements dataframes.

        For each dataframe, this method checks for the presence of specific columns and unique IDs.
        It also casts certain columns to appropriate data types and performs necessary joins.

        Args:
            subjects_df: A dataframe containing subjects information, with an optional 'subject_id' column.
            events_df: A dataframe containing events information, with optional 'event_id', 'event_type', and
                'subject_id' columns.
            dynamic_measurements_df: A dataframe containing dynamic measurements information, with an optional
                'dynamic_measurement_id' column and other measurement-specific columns.

        Returns:
            A tuple containing the preprocessed subjects, events, and dynamic_measurements dataframes.

        Raises:
            ValuesError: If any of the required columns are missing or invalid.
        """
        subjects_df, subjects_id_type = self._validate_initial_df(
            subjects_df, "subject_id", TemporalityType.STATIC
        )
        events_df, event_id_type = self._validate_initial_df(
            events_df,
            "event_id",
            TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
            {"subject_id": subjects_id_type} if subjects_df is not None else None,
        )
        if events_df is not None:
            if "event_type" not in events_df:
                raise ValueError("Missing event_type column!")
            events_df = events_df.with_columns(pl.col("event_type").cast(pl.Categorical))

            if "timestamp" not in events_df or events_df["timestamp"].dtype != pl.Datetime:
                raise ValueError("Malformed timestamp column!")

        if dynamic_measurements_df is not None:
            linked_ids = {}
            if events_df is not None:
                linked_ids["event_id"] = event_id_type

            dynamic_measurements_df, dynamic_measurement_id_types = self._validate_initial_df(
                dynamic_measurements_df, "measurement_id", TemporalityType.DYNAMIC, linked_ids
            )

        return subjects_df, events_df, dynamic_measurements_df

    @TimeableMixin.TimeAs
    def _sort_events(self):
        self.events_df = self.events_df.sort("subject_id", "timestamp", descending=False)

    @TimeableMixin.TimeAs
    def _agg_by_time(self):
        event_id_dt = self.events_df["event_id"].dtype

        if self.config.agg_by_time_scale is None:
            grouped = self.events_df.groupby(["subject_id", "timestamp"], maintain_order=True)
        else:
            grouped = self.events_df.sort(["subject_id", "timestamp"], descending=False).groupby_dynamic(
                "timestamp",
                every=self.config.agg_by_time_scale,
                truncate=True,
                closed="left",
                start_by="datapoint",
                by="subject_id",
            )

        grouped = (
            grouped.agg(
                pl.col("event_type").unique().sort(),
                pl.col("event_id").unique().alias("old_event_id"),
            )
            .sort("subject_id", "timestamp", descending=False)
            .with_row_count("event_id")
            .with_columns(
                pl.col("event_id").cast(event_id_dt),
                pl.col("event_type")
                .list.eval(pl.col("").cast(pl.Utf8))
                .list.join("&")
                .cast(pl.Categorical)
                .alias("event_type"),
            )
        )

        new_to_old_set = grouped[["event_id", "old_event_id"]].explode("old_event_id")

        self.events_df = grouped.drop("old_event_id")

        self.dynamic_measurements_df = (
            self.dynamic_measurements_df.rename({"event_id": "old_event_id"})
            .join(new_to_old_set, on="old_event_id", how="left")
            .drop("old_event_id")
        )

    def _update_subject_event_properties(self):
        if self.events_df is not None:
            self.event_types = (
                self.events_df.get_column("event_type")
                .value_counts(sort=True)
                .get_column("event_type")
                .to_list()
            )

            n_events_pd = self.events_df.get_column("subject_id").value_counts(sort=False).to_pandas()
            self.n_events_per_subject = n_events_pd.set_index("subject_id")["counts"].to_dict()
            self.subject_ids = set(self.n_events_per_subject.keys())

        if self.subjects_df is not None:
            subjects_with_no_events = (
                set(self.subjects_df.get_column("subject_id").to_list()) - self.subject_ids
            )
            for sid in subjects_with_no_events:
                self.n_events_per_subject[sid] = 0
            self.subject_ids.update(subjects_with_no_events)

    @classmethod
    def _filter_col_inclusion(cls, df: DF_T, col_inclusion_targets: dict[str, bool | Sequence[Any]]) -> DF_T:
        filter_exprs = []
        for col, incl_targets in col_inclusion_targets.items():
            match incl_targets:
                case True:
                    filter_exprs.append(pl.col(col).is_not_null())
                case False:
                    filter_exprs.append(pl.col(col).is_null())
                case _:
                    filter_exprs.append(pl.col(col).is_in(list(incl_targets)))

        return df.filter(pl.all(filter_exprs))

    @TimeableMixin.TimeAs
    def _add_time_dependent_measurements(self):
        exprs = []
        join_cols = set()
        for col, cfg in self.config.measurement_configs.items():
            if cfg.temporality != TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                continue
            fn = cfg.functor
            join_cols.update(fn.link_static_cols)
            exprs.append(fn.pl_expr().alias(col))

        join_cols = list(join_cols)

        if join_cols:
            self.events_df = (
                self.events_df.join(self.subjects_df.select("subject_id", *join_cols), on="subject_id")
                .with_columns(exprs)
                .drop(join_cols)
            )
        else:
            self.events_df = self.events_df.with_columns(exprs)

    @TimeableMixin.TimeAs
    def _prep_numerical_source(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> tuple[DF_T, str, str, str, pl.DataFrame]:
        metadata = config.measurement_metadata

        metadata_schema = self.get_metadata_schema(config)

        match config.modality:
            case DataModality.UNIVARIATE_REGRESSION:
                key_col = "const_key"
                val_col = measure
                metadata_as_polars = pl.DataFrame(
                    {key_col: [measure], **{c: [v] for c, v in metadata.items()}}
                )
                source_df = source_df.with_columns(pl.lit(measure).cast(pl.Categorical).alias(key_col))
            case DataModality.MULTIVARIATE_REGRESSION:
                key_col = measure
                val_col = config.values_column
                metadata_as_polars = pl.from_pandas(metadata, include_index=True)
            case _:
                raise ValueError(f"Called _pre_numerical_source on {config.modality} measure {measure}!")

        if "outlier_model" in metadata_as_polars and len(metadata_as_polars.drop_nulls("outlier_model")) == 0:
            metadata_as_polars = metadata_as_polars.with_columns(pl.lit(None).alias("outlier_model"))
        if "normalizer" in metadata_as_polars and len(metadata_as_polars.drop_nulls("normalizer")) == 0:
            metadata_as_polars = metadata_as_polars.with_columns(pl.lit(None).alias("normalizer"))

        metadata_as_polars = metadata_as_polars.with_columns(
            pl.col(key_col).cast(pl.Categorical),
            **{k: pl.col(k).cast(v) for k, v in metadata_schema.items()},
        )

        source_df = source_df.join(metadata_as_polars, on=key_col, how="left")
        return source_df, key_col, val_col, f"{measure}_is_inlier", metadata_as_polars

    def _total_possible_and_observed(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> tuple[int, int]:
        agg_by_col = pl.col("event_id") if config.temporality == TemporalityType.DYNAMIC else None

        if agg_by_col is None:
            num_possible = len(source_df)
            num_non_null = len(source_df.drop_nulls(measure))
        else:
            num_possible = source_df.select(pl.col("event_id").n_unique()).item()
            num_non_null = source_df.select(
                pl.col("event_id").filter(pl.col(measure).is_not_null()).n_unique()
            ).item()
        return num_possible, num_non_null

    @TimeableMixin.TimeAs
    def _add_inferred_val_types(
        self,
        measurement_metadata: DF_T,
        source_df: DF_T,
        vocab_keys_col: str,
        vals_col: str,
    ) -> DF_T:
        """Infers the appropriate type of the passed metadata column values. Performs the following
        steps:

        1. Determines if the column should be dropped for having too few measurements.
        2. Determines if the column actually contains integral, not floating point values.
        3. Determines if the column should be partially or fully re-categorized as a categorical column.

        Args:
            measurement_metadata: The metadata (pre-set or to-be-fit pre-processing parameters) for the
                numerical measure in question.
            source_df: The governing source dataframe for this measurement.
            vocab_keys_col: The column containing the "keys" for this measure. If it is a multivariate
                regression measure, this column will be the column that indicates to which covariate the value
                in the values column corresponds. If it is a univariate regression measure, this column will
                be an artificial column containing a constant key.
            vals_col: The column containing the numerical values to be assessed.


        Returns: The appropriate `NumericDataModalitySubtype` for the values.
        """

        vals_col = pl.col(vals_col)

        if "value_type" in measurement_metadata:
            missing_val_types = measurement_metadata.filter(pl.col("value_type").is_null())[vocab_keys_col]
            for_val_type_inference = source_df.filter(
                (~pl.col(vocab_keys_col).is_in(measurement_metadata[vocab_keys_col]))
                | pl.col(vocab_keys_col).is_in(missing_val_types)
            )
        else:
            for_val_type_inference = source_df

        # a. Convert to integeres where appropriate.
        if self.config.min_true_float_frequency is not None:
            is_int_expr = (
                ((vals_col == vals_col.round(0)).mean() > (1 - self.config.min_true_float_frequency))
                .cast(pl.Boolean)
                .alias("is_int")
            )
            int_keys = for_val_type_inference.groupby(vocab_keys_col).agg(is_int_expr)

            measurement_metadata = measurement_metadata.join(int_keys, on=vocab_keys_col, how="outer")

            key_is_int = pl.col(vocab_keys_col).is_in(int_keys.filter("is_int")[vocab_keys_col])
            for_val_type_inference = for_val_type_inference.with_columns(
                pl.when(key_is_int).then(vals_col.round(0)).otherwise(vals_col)
            )
        else:
            measurement_metadata = measurement_metadata.with_columns(pl.lit(False).alias("is_int"))

        # b. Drop if only has a single observed numerical value.
        dropped_keys = (
            for_val_type_inference.groupby(vocab_keys_col)
            .agg((vals_col.n_unique() == 1).cast(pl.Boolean).alias("should_drop"))
            .filter("should_drop")
        )
        keep_key_expr = ~pl.col(vocab_keys_col).is_in(dropped_keys[vocab_keys_col])
        measurement_metadata = measurement_metadata.with_columns(
            pl.when(keep_key_expr)
            .then(pl.col("value_type"))
            .otherwise(pl.lit(NumericDataModalitySubtype.DROPPED))
            .alias("value_type")
        )
        for_val_type_inference = for_val_type_inference.filter(keep_key_expr)

        # c. Convert to categorical if too few unique observations are seen.
        if self.config.min_unique_numerical_observations is not None:
            is_cat_expr = (
                lt_count_or_proportion(
                    vals_col.n_unique(),
                    self.config.min_unique_numerical_observations,
                    vals_col.len(),
                )
                .cast(pl.Boolean)
                .alias("is_categorical")
            )

            categorical_keys = for_val_type_inference.groupby(vocab_keys_col).agg(is_cat_expr)

            measurement_metadata = measurement_metadata.join(categorical_keys, on=vocab_keys_col, how="outer")
        else:
            measurement_metadata = measurement_metadata.with_columns(pl.lit(False).alias("is_categorical"))

        inferred_value_type = (
            pl.when(pl.col("is_int") & pl.col("is_categorical"))
            .then(pl.lit(NumericDataModalitySubtype.CATEGORICAL_INTEGER))
            .when(pl.col("is_categorical"))
            .then(pl.lit(NumericDataModalitySubtype.CATEGORICAL_FLOAT))
            .when(pl.col("is_int"))
            .then(pl.lit(NumericDataModalitySubtype.INTEGER))
            .otherwise(pl.lit(NumericDataModalitySubtype.FLOAT))
        )

        return measurement_metadata.with_columns(
            pl.coalesce(["value_type", inferred_value_type]).alias("value_type")
        ).drop(["is_int", "is_categorical"])

    @TimeableMixin.TimeAs
    def _fit_measurement_metadata(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> pd.DataFrame:
        source_df, vocab_keys_col, vals_col, _, measurement_metadata = self._prep_numerical_source(
            measure, config, source_df
        )
        # 1. Determines which vocab elements should be dropped due to insufficient occurrences.
        if self.config.min_valid_vocab_element_observations is not None:
            if config.temporality == TemporalityType.DYNAMIC:
                num_possible = source_df.select(pl.col("event_id").n_unique()).item()
                num_non_null = pl.col("event_id").filter(pl.col(vocab_keys_col).is_not_null()).n_unique()
            else:
                num_possible = len(source_df)
                num_non_null = pl.col(vocab_keys_col).drop_nulls().len()

            should_drop_expr = lt_count_or_proportion(
                num_non_null, self.config.min_valid_vocab_element_observations, num_possible
            ).cast(pl.Boolean)

            dropped_keys = (
                source_df.groupby(vocab_keys_col)
                .agg(should_drop_expr.alias("should_drop"))
                .filter("should_drop")
                .with_columns(pl.lit(NumericDataModalitySubtype.DROPPED).alias("value_type"))
                .drop("should_drop")
            )

            measurement_metadata = (
                measurement_metadata.join(
                    dropped_keys,
                    on=vocab_keys_col,
                    how="outer",
                    suffix="_right",
                )
                .with_columns(pl.coalesce(["value_type", "value_type_right"]).alias("value_type"))
                .drop("value_type_right")
            )
            source_df = source_df.filter(~pl.col(vocab_keys_col).is_in(dropped_keys[vocab_keys_col]))

            if len(source_df) == 0:
                measurement_metadata = measurement_metadata.to_pandas()
                measurement_metadata = measurement_metadata.set_index(vocab_keys_col)

                if config.modality == DataModality.UNIVARIATE_REGRESSION:
                    assert len(measurement_metadata) == 1
                    return measurement_metadata.loc[measure]
                else:
                    return measurement_metadata

        source_df = source_df.drop_nulls([vocab_keys_col, vals_col]).filter(pl.col(vals_col).is_not_nan())

        # 2. Eliminates hard outliers and performs censoring via specified config.
        bound_cols = {}
        for col in (
            "drop_upper_bound",
            "drop_upper_bound_inclusive",
            "drop_lower_bound",
            "drop_lower_bound_inclusive",
            "censor_lower_bound",
            "censor_upper_bound",
        ):
            if col in source_df:
                bound_cols[col] = pl.col(col)

        if bound_cols:
            source_df = source_df.with_columns(
                self.drop_or_censor(pl.col(vals_col), **bound_cols).alias(vals_col)
            )

        source_df = source_df.filter(pl.col(vals_col).is_not_nan())
        if len(source_df) == 0:
            return config.measurement_metadata

        # 3. Infer the value type and convert where necessary.
        measurement_metadata = self._add_inferred_val_types(
            measurement_metadata, source_df, vocab_keys_col, vals_col
        )

        source_df = (
            source_df.update(measurement_metadata.select(vocab_keys_col, "value_type"), on=vocab_keys_col)
            .with_columns(
                pl.when(pl.col("value_type") == NumericDataModalitySubtype.INTEGER)
                .then(pl.col(vals_col).round(0))
                .when(pl.col("value_type") == NumericDataModalitySubtype.FLOAT)
                .then(pl.col(vals_col))
                .otherwise(None)
                .alias(vals_col)
            )
            .drop_nulls(vals_col)
            .filter(pl.col(vals_col).is_not_nan())
        )

        # 4. Infer outlier detector and normalizer parameters.
        if self.config.outlier_detector_config is not None:
            with self._time_as("fit_outlier_detector"):
                M = self._get_preprocessing_model(self.config.outlier_detector_config, for_fit=True)
                outlier_model_params = source_df.groupby(vocab_keys_col).agg(
                    M.fit_from_polars(pl.col(vals_col)).alias("outlier_model")
                )

                measurement_metadata = measurement_metadata.with_columns(
                    pl.col("outlier_model").cast(outlier_model_params["outlier_model"].dtype)
                )
                source_df = source_df.with_columns(
                    pl.col("outlier_model").cast(outlier_model_params["outlier_model"].dtype)
                )

                measurement_metadata = measurement_metadata.update(outlier_model_params, on=vocab_keys_col)
                source_df = source_df.update(
                    measurement_metadata.select(vocab_keys_col, "outlier_model"), on=vocab_keys_col
                )

                is_inlier = ~M.predict_from_polars(pl.col(vals_col), pl.col("outlier_model"))
                source_df = source_df.filter(is_inlier)

        # 5. Fit a normalizer model.
        if self.config.normalizer_config is not None:
            with self._time_as("fit_normalizer"):
                M = self._get_preprocessing_model(self.config.normalizer_config, for_fit=True)
                normalizer_params = source_df.groupby(vocab_keys_col).agg(
                    M.fit_from_polars(pl.col(vals_col)).alias("normalizer")
                )
                measurement_metadata = measurement_metadata.with_columns(
                    pl.col("normalizer").cast(normalizer_params["normalizer"].dtype)
                )
                measurement_metadata = measurement_metadata.update(normalizer_params, on=vocab_keys_col)

        # 6. Convert to the appropriate type and return.
        measurement_metadata = measurement_metadata.to_pandas()
        measurement_metadata = measurement_metadata.set_index(vocab_keys_col)

        if config.modality == DataModality.UNIVARIATE_REGRESSION:
            assert len(measurement_metadata) == 1
            return measurement_metadata.loc[measure]
        else:
            return measurement_metadata

    @TimeableMixin.TimeAs
    def _fit_vocabulary(self, measure: str, config: MeasurementConfig, source_df: DF_T) -> Vocabulary:
        match config.modality:
            case DataModality.MULTIVARIATE_REGRESSION:
                val_types = pl.from_pandas(
                    config.measurement_metadata[["value_type"]], include_index=True
                ).with_columns(
                    pl.col("value_type").cast(pl.Categorical), pl.col(measure).cast(pl.Categorical)
                )
                observations = (
                    source_df.join(val_types, on=measure)
                    .with_columns(
                        pl.when(pl.col("value_type") == NumericDataModalitySubtype.CATEGORICAL_INTEGER)
                        .then(
                            pl.col(measure).cast(pl.Utf8)
                            + "__EQ_"
                            + pl.col(config.values_column).round(0).cast(int).cast(pl.Utf8)
                        )
                        .when(pl.col("value_type") == NumericDataModalitySubtype.CATEGORICAL_FLOAT)
                        .then(
                            pl.col(measure).cast(pl.Utf8)
                            + "__EQ_"
                            + pl.col(config.values_column).cast(pl.Utf8)
                        )
                        .otherwise(pl.col(measure))
                        .alias(measure)
                    )
                    .get_column(measure)
                )
            case DataModality.UNIVARIATE_REGRESSION:
                match config.measurement_metadata.value_type:
                    case NumericDataModalitySubtype.CATEGORICAL_INTEGER:
                        observations = source_df.with_columns(
                            (f"{measure}__EQ_" + pl.col(measure).round(0).cast(int).cast(pl.Utf8)).alias(
                                measure
                            )
                        ).get_column(measure)
                    case NumericDataModalitySubtype.CATEGORICAL_FLOAT:
                        observations = source_df.with_columns(
                            (f"{measure}__EQ_" + pl.col(measure).cast(pl.Utf8)).alias(measure)
                        ).get_column(measure)
                    case _:
                        return
            case _:
                observations = source_df.get_column(measure)

        # 1. Set the overall observation frequency for the column.
        observations = observations.drop_nulls()
        N = len(observations)
        if N == 0:
            return

        # 3. Fit metadata vocabularies on the training set.
        if config.vocabulary is None:
            try:
                value_counts = observations.value_counts()
                vocab_elements = value_counts.get_column(measure).to_list()
                el_counts = value_counts.get_column("counts")
                return Vocabulary(vocabulary=vocab_elements, obs_frequencies=el_counts)
            except AssertionError as e:
                raise AssertionError(f"Failed to build vocabulary for {measure}") from e

    @TimeableMixin.TimeAs
    def _transform_numerical_measurement(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> DF_T:
        source_df, keys_col_name, vals_col_name, inliers_col_name, _ = self._prep_numerical_source(
            measure, config, source_df
        )
        keys_col = pl.col(keys_col_name)
        vals_col = pl.col(vals_col_name)

        cols_to_drop_at_end = []
        for col in config.measurement_metadata:
            if col != measure and col in source_df:
                cols_to_drop_at_end.append(col)

        bound_cols = {}
        for col in (
            "drop_upper_bound",
            "drop_upper_bound_inclusive",
            "drop_lower_bound",
            "drop_lower_bound_inclusive",
            "censor_lower_bound",
            "censor_upper_bound",
        ):
            if col in source_df:
                bound_cols[col] = pl.col(col)

        if bound_cols:
            vals_col = self.drop_or_censor(vals_col, **bound_cols)

        value_type = pl.col("value_type")
        keys_col = (
            pl.when(value_type == NumericDataModalitySubtype.DROPPED)
            .then(keys_col)
            .when(value_type == NumericDataModalitySubtype.CATEGORICAL_INTEGER)
            .then(keys_col + "__EQ_" + vals_col.round(0).fill_nan(-1).cast(pl.Int64).cast(pl.Utf8))
            .when(value_type == NumericDataModalitySubtype.CATEGORICAL_FLOAT)
            .then(keys_col + "__EQ_" + vals_col.cast(pl.Utf8))
            .otherwise(keys_col)
            .alias(keys_col_name)
        )

        vals_col = (
            pl.when(
                value_type.is_in(
                    [
                        NumericDataModalitySubtype.DROPPED,
                        NumericDataModalitySubtype.CATEGORICAL_INTEGER,
                        NumericDataModalitySubtype.CATEGORICAL_FLOAT,
                    ]
                )
            )
            .then(np.NaN)
            .when(value_type == NumericDataModalitySubtype.INTEGER)
            .then(vals_col.round(0))
            .otherwise(vals_col)
            .alias(vals_col_name)
        )

        source_df = source_df.with_columns(keys_col, vals_col)

        null_idx = keys_col.is_null() | vals_col.is_null() | vals_col.is_nan()

        null_source = source_df.filter(null_idx)
        present_source = source_df.filter(~null_idx)

        if len(present_source) == 0:
            if self.config.outlier_detector_config is not None:
                null_source = null_source.with_columns(pl.lit(None).cast(pl.Boolean).alias(inliers_col_name))
            return null_source.drop(cols_to_drop_at_end)

        # 5. Add inlier/outlier indices and remove learned outliers.
        if self.config.outlier_detector_config is not None:
            M = self._get_preprocessing_model(self.config.outlier_detector_config, for_fit=False)

            inliers_col = ~M.predict_from_polars(vals_col, pl.col("outlier_model")).alias(inliers_col_name)
            vals_col = pl.when(inliers_col).then(vals_col).otherwise(np.NaN)

            present_source = present_source.with_columns(inliers_col, vals_col)
            null_source = null_source.with_columns(pl.lit(None).cast(pl.Boolean).alias(inliers_col_name))

            new_nulls = present_source.filter(~pl.col(inliers_col_name))
            null_source = null_source.vstack(new_nulls)
            present_source = present_source.filter(inliers_col_name)

        if len(present_source) == 0:
            return null_source.drop(cols_to_drop_at_end)

        # 6. Normalize values.
        if self.config.normalizer_config is not None:
            M = self._get_preprocessing_model(self.config.normalizer_config, for_fit=False)

            vals_col = M.predict_from_polars(vals_col, pl.col("normalizer"))
            present_source = present_source.with_columns(vals_col)

        source_df = present_source.vstack(null_source)

        return source_df.drop(cols_to_drop_at_end)

    @TimeableMixin.TimeAs
    def _transform_categorical_measurement(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> DF_T:
        if (config.modality == DataModality.UNIVARIATE_REGRESSION) and (
            config.measurement_metadata.value_type
            not in (
                NumericDataModalitySubtype.CATEGORICAL_INTEGER,
                NumericDataModalitySubtype.CATEGORICAL_FLOAT,
            )
        ):
            return source_df

        transform_expr = []
        if config.modality == DataModality.MULTIVARIATE_REGRESSION:
            transform_expr.append(
                pl.when(~pl.col(measure).is_in(config.vocabulary.vocabulary))
                .then(np.NaN)
                .otherwise(pl.col(config.values_column))
                .alias(config.values_column)
            )
            vocab_el_col = pl.col(measure)
        elif config.modality == DataModality.UNIVARIATE_REGRESSION:
            vocab_el_col = pl.col("const_key")
        else:
            vocab_el_col = pl.col(measure)

        transform_expr.append(
            pl.when(vocab_el_col.is_null())
            .then(None)
            .when(~vocab_el_col.is_in(config.vocabulary.vocabulary))
            .then("UNK")
            .otherwise(vocab_el_col)
            .cast(pl.Categorical)
            .alias(measure)
        )

        return source_df.with_columns(transform_expr)

    @TimeableMixin.TimeAs
    def _update_attr_df(self, attr: str, id_col: str, df: DF_T, cols_to_update: list[str]):
        old_df = getattr(self, attr)

        old_df = old_df.with_columns(**{c: pl.lit(None).cast(df[c].dtype) for c in cols_to_update})
        new_df = df.select(id_col, *cols_to_update)

        setattr(self, attr, old_df.update(new_df, on=id_col))

    def _melt_df(self, source_df: DF_T, id_cols: Sequence[str], measures: list[str]) -> pl.Expr:
        """Re-formats `source_df` into the desired deep-learning output format."""
        struct_exprs = []
        total_vocab_size = self.vocabulary_config.total_vocab_size
        idx_dt = self.get_smallest_valid_int_type(total_vocab_size)

        for m in measures:
            if m == "event_type":
                cfg = None
                modality = DataModality.SINGLE_LABEL_CLASSIFICATION
            else:
                cfg = self.measurement_configs[m]
                modality = cfg.modality

            if m in self.measurement_vocabs:
                idx_present_expr = pl.col(m).is_not_null() & pl.col(m).is_in(self.measurement_vocabs[m])
                idx_value_expr = pl.col(m).map_dict(self.unified_vocabulary_idxmap[m], return_dtype=idx_dt)
            else:
                idx_present_expr = pl.col(m).is_not_null()
                idx_value_expr = pl.lit(self.unified_vocabulary_idxmap[m][m]).cast(idx_dt)

            idx_present_expr = idx_present_expr.cast(pl.Boolean).alias("present")
            idx_value_expr = idx_value_expr.alias("index")

            if (modality == DataModality.UNIVARIATE_REGRESSION) and (
                cfg.measurement_metadata.value_type
                in (NumericDataModalitySubtype.FLOAT, NumericDataModalitySubtype.INTEGER)
            ):
                val_expr = pl.col(m)
            elif modality == DataModality.MULTIVARIATE_REGRESSION:
                val_expr = pl.col(cfg.values_column)
            else:
                val_expr = pl.lit(None).cast(pl.Float64)

            struct_exprs.append(
                pl.struct([idx_present_expr, idx_value_expr, val_expr.alias("value")]).alias(m)
            )

        measurements_idx_dt = self.get_smallest_valid_int_type(len(self.unified_measurements_idxmap))
        return (
            source_df.select(*id_cols, *struct_exprs)
            .melt(
                id_vars=id_cols,
                value_vars=measures,
                variable_name="measurement",
                value_name="value",
            )
            .filter(pl.col("value").struct.field("present"))
            .select(
                *id_cols,
                pl.col("measurement")
                .map_dict(self.unified_measurements_idxmap)
                .cast(measurements_idx_dt)
                .alias("measurement_index"),
                pl.col("value").struct.field("index").alias("index"),
                pl.col("value").struct.field("value").alias("value"),
            )
        )

    def build_DL_cached_representation(
        self, subject_ids: list[int] | None = None, do_sort_outputs: bool = False
    ) -> DF_T:
        # Identify the measurements sourced from each dataframe:
        subject_measures, event_measures, dynamic_measures = [], ["event_type"], []
        for m in self.unified_measurements_vocab[1:]:
            temporality = self.measurement_configs[m].temporality
            match temporality:
                case TemporalityType.STATIC:
                    subject_measures.append(m)
                case TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                    event_measures.append(m)
                case TemporalityType.DYNAMIC:
                    dynamic_measures.append(m)
                case _:
                    raise ValueError(f"Unknown temporality type {temporality} for {m}")

        # 1. Process subject data into the right format.
        if subject_ids:
            subjects_df = self._filter_col_inclusion(self.subjects_df, {"subject_id": subject_ids})
        else:
            subjects_df = self.subjects_df

        static_data = (
            self._melt_df(subjects_df, ["subject_id"], subject_measures)
            .groupby("subject_id")
            .agg(
                pl.col("measurement_index").alias("static_measurement_indices"),
                pl.col("index").alias("static_indices"),
            )
        )

        # 2. Process event data into the right format.
        if subject_ids:
            events_df = self._filter_col_inclusion(self.events_df, {"subject_id": subject_ids})
            event_ids = list(events_df["event_id"])
        else:
            events_df = self.events_df
            event_ids = None
        event_data = self._melt_df(events_df, ["subject_id", "timestamp", "event_id"], event_measures)

        # 3. Process measurement data into the right base format:
        if event_ids:
            dynamic_measurements_df = self._filter_col_inclusion(
                self.dynamic_measurements_df, {"event_id": event_ids}
            )
        else:
            dynamic_measurements_df = self.dynamic_measurements_df

        dynamic_ids = ["event_id", "measurement_id"] if do_sort_outputs else ["event_id"]
        dynamic_data = self._melt_df(dynamic_measurements_df, dynamic_ids, dynamic_measures)

        if do_sort_outputs:
            dynamic_data = dynamic_data.sort("event_id", "measurement_id")

        # 4. Join dynamic and event data.

        event_data = pl.concat([event_data, dynamic_data], how="diagonal")
        event_data = (
            event_data.groupby("event_id")
            .agg(
                pl.col("timestamp").drop_nulls().first().alias("timestamp"),
                pl.col("subject_id").drop_nulls().first().alias("subject_id"),
                pl.col("measurement_index").alias("dynamic_measurement_indices"),
                pl.col("index").alias("dynamic_indices"),
                pl.col("value").alias("dynamic_values"),
            )
            .sort("subject_id", "timestamp")
            .groupby("subject_id")
            .agg(
                pl.col("timestamp").first().alias("start_time"),
                ((pl.col("timestamp") - pl.col("timestamp").min()).dt.nanoseconds() / (1e9 * 60)).alias(
                    "time"
                ),
                pl.col("dynamic_measurement_indices"),
                pl.col("dynamic_indices"),
                pl.col("dynamic_values"),
            )
        )

        out = static_data.join(event_data, on="subject_id", how="outer")
        if do_sort_outputs:
            out = out.sort("subject_id")

        return out

    def _denormalize(self, events_df: DF_T, col: str) -> DF_T:
        if self.config.normalizer_config is None:
            return events_df
        elif self.config.normalizer_config["cls"] != "standard_scaler":
            raise ValueError(f"De-normalizing from {self.config.normalizer_config} not yet supported!")

        config = self.measurement_configs[col]
        if config.modality != DataModality.UNIVARIATE_REGRESSION:
            raise ValueError(f"De-normalizing {config.modality} is not currently supported.")

        normalizer_params = config.measurement_metadata.normalizer
        return events_df.with_columns(
            ((pl.col(col) * normalizer_params["std_"]) + normalizer_params["mean_"]).alias(col)
        )

import json
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime


import numpy as np
import polars as pl
import torch
from mixins import SaveableMixin, SeedableMixin, TimeableMixin

from .config import (
    MeasurementConfig,
    PytorchDatasetConfig,
    SeqPaddingSide,
    SubsequenceSamplingStrategy,
    VocabularyConfig,
)
from .types import PytorchBatch

DATA_ITEM_T = dict[str, list[float]]


def to_int_index(col: pl.Expr) -> pl.Expr:
    """Returns an integer index of the unique elements seen in this column.

    The returned index is into a vocabulary sorted lexographically.

    Args:
        col: The column containing the data to be converted into integer indices.

    Examples:
        >>> import polars as pl
        >>> X = pl.DataFrame({
        ...     'c': ['foo', 'bar', 'foo', 'bar', 'baz', None, 'bar', 'aba'],
        ...     'd': [1, 2, 3, 4, 5, 6, 7, 8]
        ... })
        >>> X.with_columns(to_int_index(pl.col('c')))
        shape: (8, 2)
        ┌──────┬─────┐
        │ c    ┆ d   │
        │ ---  ┆ --- │
        │ u32  ┆ i64 │
        ╞══════╪═════╡
        │ 4    ┆ 1   │
        │ 1    ┆ 2   │
        │ 4    ┆ 3   │
        │ 1    ┆ 4   │
        │ 2    ┆ 5   │
        │ null ┆ 6   │
        │ 1    ┆ 7   │
        │ 0    ┆ 8   │
        └──────┴─────┘
    """

    indices = col.unique(maintain_order=True).drop_nulls().search_sorted(col)
    return pl.when(col.is_null()).then(pl.lit(None)).otherwise(indices).alias(col.meta.output_name())


class PytorchDataset(SaveableMixin, SeedableMixin, TimeableMixin, torch.utils.data.Dataset):
    """A PyTorch Dataset class built on a pre-processed `DatasetBase` instance.

    This class enables accessing the deep-learning friendly representation produced by
    `Dataset.build_DL_cached_representation` in a PyTorch Dataset format. The `getitem` method of this class
    will return a dictionary containing a subject's data from this deep learning representation, with event
    sequences sliced to be within max sequence length according to configuration parameters, and the `collate`
    method of this class will collate those output dictionaries into a `PytorchBatch` object usable by
    downstream pipelines.

    Upon construction, this class will try to load a number of dataset files from disk. These files should be
    saved in accordance with the `Dataset.save` method; in particular,

    * There should be pre-cached deep-learning representation parquet dataframes stored in ``config.save_dir /
      'DL_reps' / f"{split}*.parquet"``
    * There should be a vocabulary config object in json form stored in ``config.save_dir /
      'vocabulary_config.json'``
    * There should be a set of inferred measurement configs stored in ``config.save_dir /
      'inferred_measurement_configs.json'``
    * If a task dataframe name is specified in the configuration object, then there should be either a
      pre-cached task-specifid DL representation dataframe in ``config.save_dir / 'DL_reps' / 'for_task' /
      config.task_df_name / f"{split}.parquet"``, or a "raw" task dataframe, containing subject IDs, start and
      end times, and labels, stored in ``config.save_dir / task_dfs / f"{config.task_df_name}.parquet"``. In
      the case that the latter is all that exists, then the former will be constructed by limiting the input
      cached dataframe down to the appropriate sequences and adding label columns. This newly constructed
      datafrmae will then be saved in the former filepath for future use. This construction process should
      happen first on the train split, so that inferred task vocabularies are shared across splits.

    Args:
        config: Configuration options for the dataset.
        split: The split of data which should be used in this dataset (e.g., ``'train'``, ``'tuning'``,
            ``'held_out'``). This will dictate where the system looks for pre-cached deep-learning
            representation files.
    """

    TYPE_CHECKERS = {
        "multi_class_classification": [
            (
                {pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Int8, pl.Int16, pl.Int32, pl.Int64},
                None,
            ),
            ({pl.Categorical}, to_int_index),
            ({pl.Utf8}, to_int_index),
        ],
        "binary_classification": [({pl.Boolean}, lambda Y: Y.cast(pl.Float32))],
        "regression": [({pl.Float32, pl.Float64}, None)],
    }
    """Type checker and conversion parameters for labeled datasets."""

    @classmethod
    def normalize_task(cls, col: pl.Expr, dtype: pl.DataType) -> tuple[str, pl.Expr]:
        """Normalizes the task labels in `col` of dtype `dtype` to a common format.

        Args:
            col: The column containing the task labels, in polars expression format.
            dtype: The polars data type of the task labels.

        Returns:
            The task type (a string key into the `TYPE_CHECKERS` dictionary) and the normalized column
            expression.

        Raises:
            TypeError: If the task labels are not of a supported type.
        """
        for task_type, checkers in cls.TYPE_CHECKERS.items():
            for valid_dtypes, normalize_fn in checkers:
                if dtype in valid_dtypes:
                    return task_type, (col if normalize_fn is None else normalize_fn(col))

        raise TypeError(f"Can't process label of {dtype} type!")

    def __init__(self, config: PytorchDatasetConfig, split: str):
        super().__init__()

        print('EveryQueryGPT dataloader')

        self.config = config
        self.task_types = {}
        self.task_vocabs = {}

        self.vocabulary_config = VocabularyConfig.from_json_file(
            self.config.save_dir / "vocabulary_config.json"
        )

        inferred_measurement_config_fp = self.config.save_dir / "inferred_measurement_configs.json"
        with open(inferred_measurement_config_fp) as f:
            inferred_measurement_configs = {
                k: MeasurementConfig.from_dict(v) for k, v in json.load(f).items()
            }
        self.measurement_configs = {k: v for k, v in inferred_measurement_configs.items() if not v.is_dropped}

        self.split = split

        if self.config.task_df_name is not None:
            task_dir = self.config.save_dir / "DL_reps" / "for_task" / config.task_df_name
            raw_task_df_fp = self.config.save_dir / "task_dfs" / f"{self.config.task_df_name}.parquet"
            task_info_fp = task_dir / "task_info.json"

            self.has_task = True

            if len(list(task_dir.glob(f"{split}*.parquet"))) > 0:
                print(
                    f"Re-loading task data for {self.config.task_df_name} from {task_dir}:\n"
                    f"{', '.join([str(fp) for fp in task_dir.glob(f'{split}*.parquet')])}"
                )
                self.cached_data = pl.scan_parquet(task_dir / f"{split}*.parquet")
                with open(task_info_fp) as f:
                    task_info = json.load(f)
                    self.tasks = sorted(task_info["tasks"])
                    self.task_vocabs = task_info["vocabs"]
                    self.task_types = task_info["types"]

            elif raw_task_df_fp.is_file():
                task_df = pl.scan_parquet(raw_task_df_fp)

                self.tasks = sorted(
                    [c for c in task_df.columns if c not in ["subject_id", "start_time", "end_time"]]
                )

                normalized_cols = []
                for t in self.tasks:
                    task_type, normalized_vals = self.normalize_task(col=pl.col(t), dtype=task_df.schema[t])
                    self.task_types[t] = task_type
                    normalized_cols.append(normalized_vals.alias(t))

                task_df = task_df.with_columns(normalized_cols)

                for t in self.tasks:
                    match self.task_types[t]:
                        case "binary_classification":
                            self.task_vocabs[t] = [False, True]
                        case "multi_class_classification":
                            self.task_vocabs[t] = list(
                                range(task_df.select(pl.col(t).max()).collect().item() + 1)
                            )

                task_info_fp = task_dir / "task_info.json"
                task_info = {
                    "tasks": sorted(self.tasks),
                    "vocabs": self.task_vocabs,
                    "types": self.task_types,
                }
                if task_info_fp.is_file():
                    with open(task_info_fp) as f:
                        loaded_task_info = json.load(f)
                    if loaded_task_info != task_info and self.split != "train":
                        raise ValueError(
                            f"Task info differs from on disk!\nDisk:\n{loaded_task_info}\n"
                            f"Local:\n{task_info}\nSplit: {self.split}"
                        )
                    print(f"Re-built existing {task_info_fp}! Not overwriting...")
                else:
                    task_info_fp.parent.mkdir(exist_ok=True, parents=True)
                    with open(task_info_fp, mode="w") as f:
                        json.dump(task_info, f)

                if self.split != "train":
                    print(f"WARNING: Constructing task-specific dataset on non-train split {self.split}!")
                for cached_data_fp in Path(self.config.save_dir / "DL_reps").glob(f"{split}*.parquet"):
                    task_df_fp = task_dir / cached_data_fp.name
                    if task_df_fp.is_file():
                        continue

                    print(f"Caching DL task dataframe for data file {cached_data_fp} at {task_df_fp}...")

                    task_cached_data = self._build_task_cached_df(task_df, pl.scan_parquet(cached_data_fp))

                    task_df_fp.parent.mkdir(exist_ok=True, parents=True)
                    task_cached_data.collect().write_parquet(task_df_fp)

                self.cached_data = pl.scan_parquet(task_dir / f"{split}*.parquet")
            else:
                raise FileNotFoundError(
                    f"Neither {task_dir}/*.parquet nor {raw_task_df_fp} exist, but config.task_df_name = "
                    f"{config.task_df_name}!"
                )
        else:
            self.cached_data = pl.scan_parquet(self.config.save_dir / "DL_reps" / f"{split}*.parquet")
            self.has_task = False
            self.tasks = None
            self.task_vocabs = None

        self.do_produce_static_data = "static_indices" in self.cached_data.columns
        self.seq_padding_side = config.seq_padding_side
        self.max_seq_len = config.max_seq_len

        length_constraint = pl.col("dynamic_indices").list.lengths() >= config.min_seq_len
        self.cached_data = self.cached_data.filter(length_constraint)

        if "time_delta" not in self.cached_data.columns:
            self.cached_data = self.cached_data.with_columns(
                (pl.col("start_time") + pl.duration(minutes=pl.col("time").list.first())).alias("start_time"),
                pl.col("time")
                .list.eval(
                    # We fill with 1 here as it will be ignored in the code anyways as the next event's
                    # event mask will be null.
                    # TODO(mmd): validate this in a test.
                    (pl.col("").shift(-1) - pl.col("")).fill_null(1)
                )
                .alias("time_delta"),
            ).drop("time")

        stats = (
            self.cached_data.select(pl.col("time_delta").explode().drop_nulls().alias("inter_event_time"))
            .select(
                pl.col("inter_event_time").min().alias("min"),
                pl.col("inter_event_time").log().mean().alias("mean_log"),
                pl.col("inter_event_time").log().std().alias("std_log"),
            )
            .collect()
        )

        if stats["min"].item() <= 0:
            bad_inter_event_times = self.cached_data.filter(pl.col("time_delta").list.min() <= 0).collect()
            bad_subject_ids = [str(x) for x in list(bad_inter_event_times["subject_id"])]
            warning_strs = [
                f"WARNING: Observed inter-event times <= 0 for {len(bad_inter_event_times)} subjects!",
                f"ESD Subject IDs: {', '.join(bad_subject_ids)}",
                f"Global min: {stats['min'].item()}",
            ]
            if self.config.save_dir is not None:
                fp = self.config.save_dir / f"malformed_data_{self.split}.parquet"
                bad_inter_event_times.write_parquet(fp)
                warning_strs.append(f"Wrote malformed data records to {fp}")
            warning_strs.append("Removing malformed subjects")

            print("\n".join(warning_strs))

            self.cached_data = self.cached_data.filter(pl.col("time_delta").list.min() > 0)

        self.mean_log_inter_event_time_min = stats["mean_log"].item()
        self.std_log_inter_event_time_min = stats["std_log"].item()

        self.cached_data = self.cached_data.collect()

        if self.config.train_subset_size not in (None, "FULL") and self.split == "train":
            match self.config.train_subset_size:
                case int() as n if n > 0:
                    kwargs = {"n": n}
                case float() as frac if 0 < frac < 1:
                    kwargs = {"fraction": frac}
                case _:
                    raise TypeError(
                        f"Can't process subset size of {type(self.config.train_subset_size)}, "
                        f"{self.config.train_subset_size}"
                    )

            self.cached_data = self.cached_data.sample(seed=self.config.train_subset_seed, **kwargs)

        with self._time_as("convert_to_rows"):
            self.subject_ids = self.cached_data["subject_id"].to_list()
            self.cached_data = self.cached_data.drop("subject_id")
            self.columns = self.cached_data.columns
            self.cached_data = self.cached_data.rows()

    @staticmethod
    def _build_task_cached_df(task_df: pl.LazyFrame, cached_data: pl.LazyFrame) -> pl.LazyFrame:
        """Restricts the data in a cached dataframe to only contain data for the passed task dataframe.

        Args:
            task_df: A polars LazyFrame, which must have columns ``subject_id``, ``start_time`` and
                ``end_time``. These three columns define the schema of the task (the inputs). The remaining
                columns in the task dataframe will be interpreted as labels.
            cached_data: A polars LazyFrame containing the data to be restricted to the task dataframe. Must
                have the columns ``subject_id``, ``start_time``, ``time`` or ``time_delta``,
                ``dynamic_indices``, ``dynamic_values``, and ``dynamic_measurement_indices``. These columns
                will all be restricted to just contain those events whose time values are in the specified
                task specific time range.

        Returns:
            The restricted cached dataframe, which will have the same columns as the input cached dataframe
            plus the task label columns, and will be limited to just those subjects and time-periods specified
            in the task dataframe.

        Examples:
            >>> import polars as pl
            >>> from datetime import datetime
            >>> cached_data = pl.DataFrame({
            ...     "subject_id": [0, 1, 2, 3],
            ...     "start_time": [
            ...         datetime(2020, 1, 1),
            ...         datetime(2020, 2, 1),
            ...         datetime(2020, 3, 1),
            ...         datetime(2020, 1, 2)
            ...     ],
            ...     "time": [
            ...         [0.0, 60*24.0, 2*60*24., 3*60*24., 4*60*24.],
            ...         [0.0, 7*60*24.0, 2*7*60*24., 3*7*60*24., 4*7*60*24.],
            ...         [0.0, 60*12.0, 2*60*12.],
            ...         [0.0, 60*24.0, 2*60*24., 3*60*24., 4*60*24.],
            ...     ],
            ...     "dynamic_measurement_indices": [
            ...         [[0, 1, 1], [0, 2], [0], [0, 3], [0]],
            ...         [[0, 1, 1], [0, 4], [0], [0, 1], [0]],
            ...         [[0, 1, 1], [0], [0, 4]],
            ...         [[0, 1, 1], [0, 4], [0], [0, 2], [0]],
            ...     ],
            ...     "dynamic_indices": [
            ...         [[6, 11, 12], [1, 40], [5], [1, 55], [5]],
            ...         [[2, 11, 13], [1, 84], [8], [1, 19], [5]],
            ...         [[1, 18, 21], [1], [5, 87]],
            ...         [[3, 20, 21], [1, 94], [8], [1, 33], [9]],
            ...     ],
            ...     "dynamic_values": [
            ...         [[None, 0.2, 1.0], [None, 0.0], [None], [None, None], [None]],
            ...         [[None, -0.1, 0.0], [None, None], [None], [None, -4.2], [None]],
            ...         [[None, 0.9, 1.2], [None], [None, None]],
            ...         [[None, 3.2, -1.0], [None, None], [None], [None, 0.5], [None]],
            ...     ],
            ... })
            >>> task_df = pl.DataFrame({
            ...     "subject_id": [0, 1, 2, 5],
            ...     "start_time": [
            ...         datetime(2020, 1, 1),
            ...         datetime(2020, 1, 11),
            ...         datetime(2020, 3, 1, 13),
            ...         datetime(2020, 1, 2)
            ...     ],
            ...     "end_time": [
            ...         datetime(2020, 1, 3),
            ...         datetime(2020, 1, 21),
            ...         datetime(2020, 3, 4),
            ...         datetime(2020, 1, 3)
            ...     ],
            ...     "label1": [0, 1, 0, 1],
            ...     "label2": [0, 1, 5, 1]
            ... })
            >>> pl.Config.set_tbl_width_chars(88)
            <class 'polars.config.Config'>
            >>> PytorchDataset._build_task_cached_df(task_df, cached_data)
            shape: (3, 8)
            ┌───────────┬───────────┬───────────┬──────────┬──────────┬──────────┬────────┬────────┐
            │ subject_i ┆ start_tim ┆ time      ┆ dynamic_ ┆ dynamic_ ┆ dynamic_ ┆ label1 ┆ label2 │
            │ d         ┆ e         ┆ ---       ┆ measurem ┆ indices  ┆ values   ┆ ---    ┆ ---    │
            │ ---       ┆ ---       ┆ list[f64] ┆ ent_indi ┆ ---      ┆ ---      ┆ i64    ┆ i64    │
            │ i64       ┆ datetime[ ┆           ┆ ces      ┆ list[lis ┆ list[lis ┆        ┆        │
            │           ┆ μs]       ┆           ┆ ---      ┆ t[i64]]  ┆ t[f64]]  ┆        ┆        │
            │           ┆           ┆           ┆ list[lis ┆          ┆          ┆        ┆        │
            │           ┆           ┆           ┆ t[i64]]  ┆          ┆          ┆        ┆        │
            ╞═══════════╪═══════════╪═══════════╪══════════╪══════════╪══════════╪════════╪════════╡
            │ 0         ┆ 2020-01-0 ┆ [0.0,     ┆ [[0, 1,  ┆ [[6, 11, ┆ [[null,  ┆ 0      ┆ 0      │
            │           ┆ 1         ┆ 1440.0]   ┆ 1], [0,  ┆ 12], [1, ┆ 0.2,     ┆        ┆        │
            │           ┆ 00:00:00  ┆           ┆ 2]]      ┆ 40]]     ┆ 1.0],    ┆        ┆        │
            │           ┆           ┆           ┆          ┆          ┆ [null,   ┆        ┆        │
            │           ┆           ┆           ┆          ┆          ┆ 0.0]]    ┆        ┆        │
            │ 1         ┆ 2020-02-0 ┆ []        ┆ []       ┆ []       ┆ []       ┆ 1      ┆ 1      │
            │           ┆ 1         ┆           ┆          ┆          ┆          ┆        ┆        │
            │           ┆ 00:00:00  ┆           ┆          ┆          ┆          ┆        ┆        │
            │ 2         ┆ 2020-03-0 ┆ [1440.0]  ┆ [[0, 4]] ┆ [[5,     ┆ [[null,  ┆ 0      ┆ 5      │
            │           ┆ 1         ┆           ┆          ┆ 87]]     ┆ null]]   ┆        ┆        │
            │           ┆ 00:00:00  ┆           ┆          ┆          ┆          ┆        ┆        │
            └───────────┴───────────┴───────────┴──────────┴──────────┴──────────┴────────┴────────┘
        """
        time_dep_cols = [c for c in ("time", "time_delta") if c in cached_data.columns]
        time_dep_cols.extend(["dynamic_indices", "dynamic_values", "dynamic_measurement_indices"])

        if "time" in cached_data.columns:
            time_col_expr = pl.col("time")
        elif "time_delta" in cached_data.columns:
            time_col_expr = pl.col("time_delta").cumsum().over("subject_id")

        start_idx_expr = (
            time_col_expr.list.explode().search_sorted(pl.col("start_time_min")).over("subject_id")
        )
        end_idx_expr = time_col_expr.list.explode().search_sorted(pl.col("end_time_min")).over("subject_id")

        return (
            cached_data.join(task_df, on="subject_id", how="inner", suffix="_task")
            .with_columns(
                start_time_min=(pl.col("start_time_task") - pl.col("start_time")) / np.timedelta64(1, "m"),
                end_time_min=(pl.col("end_time") - pl.col("start_time")) / np.timedelta64(1, "m"),
            )
            .with_columns(
                **{
                    t: pl.col(t).list.slice(start_idx_expr, end_idx_expr - start_idx_expr)
                    for t in time_dep_cols
                },
            )
            .drop("start_time_task", "end_time_min", "start_time_min", "end_time")
        )

        return cached_data

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx: int) -> dict[str, list]:
        """Returns a Returns a dictionary corresponding to a single subject's data.

        The output of this will not be tensorized as that work will need to be re-done in the collate function
        regardless. The output will have structure:
        ``
        {
            'time_delta': [seq_len],
            'dynamic_indices': [seq_len, n_data_per_event] (ragged),
            'dynamic_values': [seq_len, n_data_per_event] (ragged),
            'dynamic_measurement_indices': [seq_len, n_data_per_event] (ragged),
            'static_indices': [seq_len, n_data_per_event] (ragged),
            'static_measurement_indices': [seq_len, n_data_per_event] (ragged),
        }
        ``

        1. ``time_delta`` captures the time between each event and the subsequent event.
        2. ``dynamic_indices`` captures the categorical metadata elements listed in `self.data_cols` in a
           unified vocabulary space spanning all metadata vocabularies.
        3. ``dynamic_values`` captures the numerical metadata elements listed in `self.data_cols`. If no
           numerical elements are listed in `self.data_cols` for a given categorical column, the according
           index in this output will be `np.NaN`.
        4. ``dynamic_measurement_indices`` captures which measurement vocabulary was used to source a given
           data element.
        5. ``static_indices`` captures the categorical metadata elements listed in `self.static_cols` in a
           unified vocabulary.
        6. ``static_measurement_indices`` captures which measurement vocabulary was used to source a given
           data element.
        """
        return self._seeded_getitem(idx)

    @SeedableMixin.WithSeed
    @TimeableMixin.TimeAs
    def _seeded_getitem(self, idx: int) -> dict[str, list]:
        """Returns a Returns a dictionary corresponding to a single subject's data.

        This function is automatically seeded for robustness. See `__getitem__` for a description of the
        output format.
        """

        full_subj_data = {c: v for c, v in zip(self.columns, self.cached_data[idx])}
        for k in ["static_indices", "static_measurement_indices"]:
            if full_subj_data[k] is None:
                full_subj_data[k] = []
        if self.config.do_include_subject_id:
            full_subj_data["subject_id"] = self.subject_ids[idx]

        min_future_days = 180 #todo: add to config
        start_time = full_subj_data["start_time"].timestamp() / 60 # minutes 
        times = np.array([sum(full_subj_data["time_delta"][:i]) for i in range(1, len(full_subj_data["time_delta"])+1)])
        min_future = min(times[-1], min_future_days*24*60)
        max_end_time = start_time + times[-1] - min_future
        max_end_idx = np.max(np.argwhere((times+start_time) < max_end_time))

        print(f'there are {len(full_subj_data["time_delta"])} events')
        print('data start at',datetime.fromtimestamp(60*(start_time)))
        print('data cuts off inputs at',datetime.fromtimestamp(60*(start_time+times[max_end_idx])))
        print('data actually ends at',datetime.fromtimestamp(60*(start_time+times[-1])))

        if max_end_idx <= self.max_seq_len: 
            input_start_idx = 0
            input_end_idx = max_end_idx
        else: 
            input_start_idx = np.random.choice(max_end_idx - self.max_seq_len)
            input_end_idx = input_start_idx + self.max_seq_len

        min_query_duration_hours = 1  #todo: add to config
        min_duration = min_query_duration_hours*60

        print(f'answer to the query is calculated from {input_end_idx} to end of events {len(times)}')
        query_start_time = np.random.randint(times[input_end_idx], times[-1]-min_duration)
        query_end_time = np.random.randint(query_start_time+min_duration, times[-1])
        query_start_time += start_time
        query_end_time  += start_time
        query_start_idx = np.min(np.argwhere((times+start_time) >= query_start_time))
        query_end_idx = np.max(np.argwhere((times+start_time) <= query_end_time))

        print(f'query duration {(query_end_time-query_start_time)/(24*60)} days')
        print(f'query start time {datetime.fromtimestamp(60*(query_start_time))}')
        print(f'query start idx time {datetime.fromtimestamp(60*(start_time+times[query_start_idx]))} at idx {query_start_idx}')
        print(f'query end time {datetime.fromtimestamp(60*(query_end_time))}')
        print(f'query end idx time {datetime.fromtimestamp(60*(start_time+times[query_end_idx]))} at idx {query_end_idx}')

        # should the code already be present in the answer period or be random from the list of codes? 
        # ideally this should be sampled from all codes in the training data 
        codes_observed = Counter(sum( full_subj_data['dynamic_indices'][query_start_idx:query_end_idx], []))
        code = np.random.choice(list(codes_observed.keys())) 
        query = {
            # s, end time e, and code c with some (potentially fully open) value range (L,R)
            'start_delta_from_input_end': query_start_time-times[input_end_idx],
            'end_delta_from_input_end': query_end_time-times[input_end_idx],
            'duration': query_end_time-query_start_time,
            'code': code
            # ignorning values for now not sure how to look up the min max range for the entire patient set
        } 

        poisson_rate = codes_observed[code] / query['duration'] # frequency / minutes ?? 
        # need to normalize somehow?? 
        # is poisson right for uneven sampling? 

        print(f'inputs go from {input_start_idx} to {input_end_idx}')
        for k in (
                    "time_delta",
                    "dynamic_indices",
                    "dynamic_values",
                    "dynamic_measurement_indices",
                ):
                    full_subj_data[k] = full_subj_data[k][input_start_idx : input_end_idx]

        return full_subj_data, query, poisson_rate

    def __static_and_dynamic_collate(self, batch: list[DATA_ITEM_T]) -> PytorchBatch:
        """An internal collate function for both static and dynamic data."""
        out_batch = self.__dynamic_only_collate(batch)

        # Get the maximum number of static elements in the batch.
        max_n_static = max(len(e["static_indices"]) for e in batch)

        # Walk through the batch and pad the associated tensors in all requisite dimensions.
        self._register_start("collate_static_padding")
        out = defaultdict(list)
        for e in batch:
            if self.do_produce_static_data:
                n_static = len(e["static_indices"])
                static_delta = max_n_static - n_static
                out["static_indices"].append(
                    torch.nn.functional.pad(
                        torch.Tensor(e["static_indices"]), (0, static_delta), value=np.NaN
                    )
                )
                out["static_measurement_indices"].append(
                    torch.nn.functional.pad(
                        torch.Tensor(e["static_measurement_indices"]),
                        (0, static_delta),
                        value=np.NaN,
                    )
                )
        self._register_end("collate_static_padding")

        self._register_start("collate_static_post_padding")
        # Unsqueeze the padded tensors into the batch dimension and combine them.
        out = {k: torch.cat([T.unsqueeze(0) for T in Ts], dim=0) for k, Ts in out.items()}

        # Convert to the right types and add to the batch.
        out_batch["static_indices"] = torch.nan_to_num(out["static_indices"], nan=0).long()
        out_batch["static_measurement_indices"] = torch.nan_to_num(
            out["static_measurement_indices"], nan=0
        ).long()
        self._register_end("collate_static_post_padding")

        return out_batch

    def __dynamic_only_collate(self, batch: list[DATA_ITEM_T]) -> PytorchBatch:
        """An internal collate function for dynamic data alone."""
        # Get the local max sequence length and n_data elements for padding.
        max_seq_len = max(len(e["time_delta"]) for e in batch)

        max_n_data = 0
        for e in batch:
            for v in e["dynamic_indices"]:
                max_n_data = max(max_n_data, len(v))
        if max_n_data == 0:
            raise ValueError(f"Batch has no dynamic measurements! Got:\n{batch[0]}\n{batch[1]}\n...")

        # Walk through the batch and pad the associated tensors in all requisite dimensions.
        self._register_start("collate_dynamic_padding")
        out = defaultdict(list)
        for e in batch:
            seq_len = len(e["time_delta"])
            seq_delta = max_seq_len - seq_len

            if self.seq_padding_side == SeqPaddingSide.RIGHT:
                out["time_delta"].append(
                    torch.nn.functional.pad(torch.Tensor(e["time_delta"]), (0, seq_delta), value=np.NaN)
                )
            else:
                out["time_delta"].append(
                    torch.nn.functional.pad(torch.Tensor(e["time_delta"]), (seq_delta, 0), value=np.NaN)
                )

            data_elements = defaultdict(list)
            for k in ("dynamic_indices", "dynamic_values", "dynamic_measurement_indices"):
                for vs in e[k]:
                    if vs is None:
                        vs = [np.NaN] * max_n_data

                    data_delta = max_n_data - len(vs)
                    vs = [v if v is not None else np.NaN for v in vs]

                    # We don't worry about seq_padding_side here as this is not the sequence dimension.
                    data_elements[k].append(
                        torch.nn.functional.pad(torch.Tensor(vs), (0, data_delta), value=np.NaN)
                    )

                if len(data_elements[k]) == 0:
                    raise ValueError(f"Batch element has no {k}! Got:\n{e}.")

                if self.seq_padding_side == SeqPaddingSide.RIGHT:
                    data_elements[k] = torch.nn.functional.pad(
                        torch.cat([T.unsqueeze(0) for T in data_elements[k]]),
                        (0, 0, 0, seq_delta),
                        value=np.NaN,
                    )
                else:
                    data_elements[k] = torch.nn.functional.pad(
                        torch.cat([T.unsqueeze(0) for T in data_elements[k]]),
                        (0, 0, seq_delta, 0),
                        value=np.NaN,
                    )

                out[k].append(data_elements[k])
        self._register_end("collate_dynamic_padding")

        self._register_start("collate_post_padding_processing")
        # Unsqueeze the padded tensors into the batch dimension and combine them.
        out_batch = {k: torch.cat([T.unsqueeze(0) for T in Ts], dim=0) for k, Ts in out.items()}

        # Add event and data masks on the basis of which elements are present, then convert the tensor
        # elements to the appropriate types.
        out_batch["event_mask"] = ~out_batch["time_delta"].isnan()
        out_batch["dynamic_values_mask"] = ~out_batch["dynamic_values"].isnan()

        out_batch["time_delta"] = torch.nan_to_num(out_batch["time_delta"], nan=0)

        out_batch["dynamic_indices"] = torch.nan_to_num(out_batch["dynamic_indices"], nan=0).long()
        out_batch["dynamic_measurement_indices"] = torch.nan_to_num(
            out_batch["dynamic_measurement_indices"], nan=0
        ).long()
        out_batch["dynamic_values"] = torch.nan_to_num(out_batch["dynamic_values"], nan=0)

        if self.config.do_include_start_time_min:
            out_batch["start_time"] = torch.FloatTensor([e["start_time"] for e in batch])
        if self.config.do_include_subsequence_indices:
            out_batch["start_idx"] = torch.LongTensor([e["start_idx"] for e in batch])
            out_batch["end_idx"] = torch.LongTensor([e["end_idx"] for e in batch])
        if self.config.do_include_subject_id:
            out_batch["subject_id"] = torch.LongTensor([e["subject_id"] for e in batch])

        out_batch = PytorchBatch(**out_batch)
        self._register_end("collate_post_padding_processing")

        if not self.has_task:
            return out_batch

        self._register_start("collate_task_labels")
        out_labels = {}

        for task in self.tasks:
            task_type = self.task_types[task]

            out_labels[task] = []
            for e in batch:
                out_labels[task].append(e[task])

            match task_type:
                case "multi_class_classification":
                    out_labels[task] = torch.LongTensor(out_labels[task])
                case "binary_classification":
                    out_labels[task] = torch.FloatTensor(out_labels[task])
                case "regression":
                    out_labels[task] = torch.FloatTensor(out_labels[task])
                case _:
                    raise TypeError(f"Don't know how to tensorify task of type {task_type}!")

        out_batch.stream_labels = out_labels
        self._register_end("collate_task_labels")

        return out_batch

    @TimeableMixin.TimeAs
    def collate(self, batch: list[DATA_ITEM_T]) -> PytorchBatch:
        """Combines the ragged dictionaries produced by `__getitem__` into a tensorized batch.

        This function handles conversion of arrays to tensors and padding of elements within the batch across
        static data elements, sequence events, and dynamic data elements.

        Args:
            batch: A list of `__getitem__` format output dictionaries.

        Returns:
            A fully collated, tensorized, and padded batch.
        """
        if self.do_produce_static_data:
            return self.__static_and_dynamic_collate(batch)
        else:
            return self.__dynamic_only_collate(batch)

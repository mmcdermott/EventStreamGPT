import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger
from mixins import SeedableMixin
from nested_ragged_tensors.ragged_numpy import (
    NP_FLOAT_TYPES,
    NP_INT_TYPES,
    NP_UINT_TYPES,
    JointNestedRaggedTensorDict,
)
from tqdm.auto import tqdm

from ..utils import count_or_proportion
from .config import PytorchDatasetConfig, SeqPaddingSide, SubsequenceSamplingStrategy
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


class PytorchDataset(SeedableMixin, torch.utils.data.Dataset):
    """A PyTorch Dataset class.

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
            ({pl.Categorical(ordering="physical"), pl.Categorical(ordering="lexical")}, to_int_index),
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

    def __init__(self, config: PytorchDatasetConfig, split: str, just_cache: bool = False):
        super().__init__()

        self.config = config
        self.split = split

        logger.info("Reading vocabulary")
        self.read_vocabulary()

        logger.info("Reading splits & patient shards")
        self.read_shards()

        logger.info("Reading patient descriptors")
        self.read_patient_descriptors()

        if self.config.min_seq_len is not None and self.config.min_seq_len > 1:
            logger.info(f"Restricting to subjects with at least {config.min_seq_len} events")
            self.filter_to_min_seq_len()

        if self.config.train_subset_size not in (None, "FULL") and self.split == "train":
            logger.info(f"Filtering training subset size to {self.config.train_subset_size}")
            self.filter_to_subset()

        self.set_inter_event_time_stats()

    @property
    def static_dir(self) -> Path:
        return self.config.save_dir / "DL_reps"

    @property
    def task_dir(self) -> Path:
        return self.config.save_dir / "task_dfs"

    @property
    def NRTs_dir(self) -> Path:
        return self.config.save_dir / "NRT_reps"

    def read_vocabulary(self):
        """Reads the vocabulary either from the ESGPT or MEDS dataset."""
        self.vocabulary_config = self.config.vocabulary_config

    def read_shards(self):
        """Reads the split-specific patient shards from the ESGPT or MEDS dataset."""
        shards_fp = self.config.save_dir / "DL_shards.json"
        all_shards = json.loads(shards_fp.read_text())
        self.shards = {sp: subjs for sp, subjs in all_shards.items() if sp.startswith(f"{self.split}/")}
        self.subj_map = {subj: sp for sp, subjs in self.shards.items() for subj in subjs}

    @property
    def measurement_configs(self):
        """Grabs the measurement configs from the config."""
        return self.config.measurement_configs

    def read_patient_descriptors(self):
        """Reads the patient descriptors from the ESGPT or MEDS dataset."""
        self.static_dfs = {}
        self.subj_indices = {}
        self.subj_seq_bounds = {}

        shards = tqdm(self.shards.keys(), total=len(self.shards), desc="Reading static shards", leave=False)
        for shard in shards:
            static_fp = self.static_dir / f"{shard}.parquet"
            df = pl.read_parquet(
                static_fp,
                columns=[
                    "subject_id",
                    "start_time",
                    "static_indices",
                    "static_measurement_indices",
                    "time_delta",
                ],
                use_pyarrow=True,
            )

            self.static_dfs[shard] = df
            subject_ids = df["subject_id"]
            n_events = df.select(pl.col("time_delta").list.lengths().alias("n_events")).get_column("n_events")
            for i, (subj, n_events) in enumerate(zip(subject_ids, n_events)):
                if subj in self.subj_indices or subj in self.subj_seq_bounds:
                    raise ValueError(f"Duplicate subject {subj} in {shard}!")

                self.subj_indices[subj] = i
                self.subj_seq_bounds[subj] = (0, n_events)

        if self.config.task_df_name is None:
            self.index = [(subj, *bounds) for subj, bounds in self.subj_seq_bounds.items()]
            self.labels = {}
            self.tasks = None
            self.task_types = None
            self.task_vocabs = None
        else:
            task_df_fp = self.task_dir / f"{self.config.task_df_name}.parquet"
            task_info_fp = self.task_dir / f"{self.config.task_df_name}_info.json"

            logger.info(f"Reading task constraints for {self.config.task_df_name} from {task_df_fp}")
            task_df = pl.read_parquet(task_df_fp, use_pyarrow=True)

            task_info = self.get_task_info(task_df)

            if task_info_fp.is_file():
                loaded_task_info = json.loads(task_info_fp.read_text())
                if loaded_task_info != task_info:
                    raise ValueError(
                        f"Task info differs from on disk!\nDisk:\n{loaded_task_info}\n"
                        f"Local:\n{task_info}\nSplit: {self.split}"
                    )
                logger.info(f"Re-built existing {task_info_fp} and it matches.")
            else:
                task_info_fp.parent.mkdir(exist_ok=True, parents=True)
                task_info_fp.write_text(json.dumps(task_info))

            idx_col = "_row_index"
            while idx_col in task_df.columns:
                idx_col = f"_{idx_col}"

            task_df_joint = (
                task_df.select("subject_id", "start_time", "end_time")
                .with_row_index(idx_col)
                .group_by("subject_id")
                .agg("start_time", "end_time", idx_col)
                .join(
                    pl.concat(self.static_dfs.values()).select(
                        "subject_id", pl.col("start_time").alias("start_time_global"), "time_delta"
                    ),
                    on="subject_id",
                    how="left",
                )
                .with_columns(
                    pl.col("time_delta")
                    .list.eval(pl.element().fill_null(0).cum_sum())
                    .alias("min_since_start")
                )
            )

            min_at_task_start = (
                (pl.col("start_time") - pl.col("start_time_global")).dt.total_seconds() / 60
            ).alias("min_at_task_start")
            min_at_task_end = (
                (pl.col("end_time") - pl.col("start_time_global")).dt.total_seconds() / 60
            ).alias("min_at_task_end")

            start_idx_expr = (pl.col("min_since_start").search_sorted(pl.col("min_at_task_start"))).alias(
                "start_idx"
            )
            end_idx_expr = (pl.col("min_since_start").search_sorted(pl.col("min_at_task_end"))).alias(
                "end_idx"
            )

            task_df_joint = (
                task_df_joint.explode(idx_col, "start_time", "end_time")
                .with_columns(min_at_task_start, min_at_task_end)
                .explode("min_since_start")
                .group_by("subject_id", idx_col, "min_at_task_start", "min_at_task_end", maintain_order=True)
                .agg(start_idx_expr.first(), end_idx_expr.first())
                .sort(by=idx_col, descending=False)
            )

            subject_ids = task_df_joint["subject_id"]
            start_indices = task_df_joint["start_idx"]
            end_indices = task_df_joint["end_idx"]

            self.labels = {t: task_df.get_column(t).to_list() for t in self.tasks}
            self.index = list(zip(subject_ids, start_indices, end_indices))

    def get_task_info(self, task_df: pl.DataFrame):
        """Gets the task information from the task dataframe."""
        self.tasks = sorted([c for c in task_df.columns if c not in ["subject_id", "start_time", "end_time"]])

        self.task_types = {}
        self.task_vocabs = {}

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
                    self.task_vocabs[t] = list(range(task_df.select(pl.col(t).max()).item() + 1))
                case _:
                    raise NotImplementedError(f"Task type {self.task_types[t]} not implemented!")

        return {"tasks": sorted(self.tasks), "vocabs": self.task_vocabs, "types": self.task_types}

    def filter_to_min_seq_len(self):
        """Filters the dataset to only include subjects with at least `config.min_seq_len` events."""
        if self.config.task_df_name is not None:
            logger.warning(
                f"Filtering task {self.config.task_df_name} to min_seq_len {self.config.min_seq_len}. "
                "This may result in incomparable model results against runs with different constraints!"
            )

        orig_len = len(self)
        orig_n_subjects = len(set(self.subject_ids))
        valid_indices = [
            i for i, (subj, start, end) in enumerate(self.index) if end - start >= self.config.min_seq_len
        ]
        self.index = [self.index[i] for i in valid_indices]
        self.labels = {t: [t_labels[i] for i in valid_indices] for t, t_labels in self.labels.items()}
        new_len = len(self)
        new_n_subjects = len(set(self.subject_ids))
        logger.info(
            f"Filtered data due to sequence length constraint (>= {self.config.min_seq_len}) from "
            f"{orig_len} to {new_len} rows and {orig_n_subjects} to {new_n_subjects} subjects."
        )

    def filter_to_subset(self):
        """Filters the dataset to only include a subset of subjects."""

        orig_len = len(self)
        orig_n_subjects = len(set(self.subject_ids))
        rng = np.random.default_rng(self.config.train_subset_seed)
        subset_subjects = rng.choice(
            list(set(self.subject_ids)),
            size=count_or_proportion(orig_n_subjects, self.config.train_subset_size),
            replace=False,
        )
        valid_indices = [i for i, (subj, start, end) in enumerate(self.index) if subj in subset_subjects]
        self.index = [self.index[i] for i in valid_indices]
        self.labels = {t: [t_labels[i] for i in valid_indices] for t, t_labels in self.labels.items()}
        new_len = len(self)
        new_n_subjects = len(set(self.subject_ids))
        logger.info(
            f"Filtered data to subset of {self.config.train_subset_size} subjects from "
            f"{orig_len} to {new_len} rows and {orig_n_subjects} to {new_n_subjects} subjects."
        )

    def set_inter_event_time_stats(self):
        """Sets the inter-event time statistics for the dataset."""
        data_for_stats = pl.concat([x.lazy() for x in self.static_dfs.values()])
        stats = (
            data_for_stats.select(
                pl.col("time_delta").explode().drop_nulls().drop_nans().alias("inter_event_time")
            )
            .select(
                pl.col("inter_event_time").min().alias("min"),
                pl.col("inter_event_time").log().mean().alias("mean_log"),
                pl.col("inter_event_time").log().std().alias("std_log"),
            )
            .collect()
        )

        if stats["min"].item() <= 0:
            bad_inter_event_times = data_for_stats.filter(pl.col("time_delta").list.min() <= 0).collect()
            bad_subject_ids = set(bad_inter_event_times["subject_id"].to_list())
            warning_strs = [
                f"Observed inter-event times <= 0 for {len(bad_inter_event_times)} subjects!",
                f"Bad Subject IDs: {', '.join(str(x) for x in bad_subject_ids)}",
                f"Global min: {stats['min'].item()}",
            ]
            if self.config.save_dir is not None:
                fp = self.config.save_dir / f"malformed_data_{self.split}.parquet"
                bad_inter_event_times.write_parquet(fp)
                warning_strs.append(f"Wrote malformed data records to {fp}")
            warning_strs.append("Removing malformed subjects")

            logger.warning("\n".join(warning_strs))

            self.index = [x for x in self.index if x[0] not in bad_subject_ids]

        self.mean_log_inter_event_time_min = stats["mean_log"].item()
        self.std_log_inter_event_time_min = stats["std_log"].item()

    @property
    def subject_ids(self) -> list[int]:
        return [x[0] for x in self.index]

    def __len__(self):
        return len(self.index)

    @property
    def has_task(self) -> bool:
        return self.config.task_df_name is not None

    @property
    def seq_padding_side(self) -> SeqPaddingSide:
        return self.config.seq_padding_side

    @property
    def max_seq_len(self) -> int:
        return self.config.max_seq_len

    @property
    def is_subset_dataset(self) -> bool:
        return self.config.train_subset_size != "FULL"

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
    def _seeded_getitem(self, idx: int) -> dict[str, list[float]]:
        """Returns a Returns a dictionary corresponding to a single subject's data.

        This function is a seedable version of `__getitem__`.
        """

        subject_id, st, end = self.index[idx]

        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]
        static_row = self.static_dfs[shard][subject_idx].to_dict()

        out = {
            "static_indices": static_row["static_indices"].item().to_list(),
            "static_measurement_indices": static_row["static_measurement_indices"].item().to_list(),
        }

        if self.config.do_include_subject_id:
            out["subject_id"] = subject_id

        seq_len = end - st
        if seq_len > self.max_seq_len:
            match self.config.subsequence_sampling_strategy:
                case SubsequenceSamplingStrategy.RANDOM:
                    start_offset = np.random.choice(seq_len - self.max_seq_len)
                case SubsequenceSamplingStrategy.TO_END:
                    start_offset = seq_len - self.max_seq_len
                case SubsequenceSamplingStrategy.FROM_START:
                    start_offset = 0
                case _:
                    raise ValueError(
                        f"Invalid subsequence sampling strategy {self.config.subsequence_sampling_strategy}!"
                    )

            st += start_offset
            end = min(end, st + self.max_seq_len)

        if self.config.do_include_subsequence_indices:
            out["start_idx"] = st
            out["end_idx"] = end

        out["dynamic"] = JointNestedRaggedTensorDict.load_slice(self.NRTs_dir / f"{shard}.pt", subject_idx)[
            st:end
        ]

        if self.config.do_include_start_time_min:
            out["start_time"] = static_row["start_time"] = static_row[
                "start_time"
            ].item().timestamp() / 60.0 + sum(static_row["time_delta"].item().to_list()[:st])

        for t, t_labels in self.labels.items():
            out[t] = t_labels[idx]

        return out

    def __dynamic_only_collate(self, batch: list[dict[str, list[float]]]) -> PytorchBatch:
        """An internal collate function for only dynamic data."""
        keys = batch[0].keys()
        dense_keys = {k for k in keys if k not in ("dynamic", "static_indices", "static_measurement_indices")}

        if dense_keys:
            dense_collated = torch.utils.data.default_collate([{k: x[k] for k in dense_keys} for x in batch])
        else:
            dense_collated = {}

        dynamic = JointNestedRaggedTensorDict.vstack([x["dynamic"] for x in batch]).to_dense(
            padding_side=self.seq_padding_side
        )
        dynamic["event_mask"] = dynamic.pop("dim1/mask")
        dynamic["dynamic_values_mask"] = dynamic.pop("dim2/mask") & ~np.isnan(dynamic["dynamic_values"])

        dynamic_collated = {}
        for k, v in dynamic.items():
            if k.endswith("mask"):
                dynamic_collated[k] = torch.from_numpy(v)
            elif v.dtype in NP_UINT_TYPES + NP_INT_TYPES:
                dynamic_collated[k] = torch.from_numpy(v.astype(int)).long()
            elif v.dtype in NP_FLOAT_TYPES:
                dynamic_collated[k] = torch.from_numpy(v.astype(float)).float()
            else:
                raise TypeError(f"Don't know how to tensorify {k} of type {v.dtype}!")

        collated = {**dense_collated, **dynamic_collated}

        out_batch = {}
        out_batch["event_mask"] = collated["event_mask"]
        out_batch["dynamic_values_mask"] = collated["dynamic_values_mask"]
        out_batch["time_delta"] = torch.nan_to_num(collated["time_delta"].float(), nan=0)
        out_batch["dynamic_indices"] = collated["dynamic_indices"].long()
        out_batch["dynamic_measurement_indices"] = collated["dynamic_measurement_indices"].long()
        out_batch["dynamic_values"] = torch.nan_to_num(collated["dynamic_values"].float(), nan=0)

        if self.config.do_include_start_time_min:
            out_batch["start_time"] = collated["start_time"].float()
        if self.config.do_include_subsequence_indices:
            out_batch["start_idx"] = collated["start_idx"].long()
            out_batch["end_idx"] = collated["end_idx"].long()
        if self.config.do_include_subject_id:
            out_batch["subject_id"] = collated["subject_id"].long()

        out_batch = PytorchBatch(**out_batch)

        if not self.has_task:
            return out_batch

        out_labels = {}
        for task in self.tasks:
            match self.task_types[task]:
                case "multi_class_classification":
                    out_labels[task] = collated[task].long()
                case "binary_classification":
                    out_labels[task] = collated[task].float()
                case "regression":
                    out_labels[task] = collated[task].float()
                case _:
                    raise TypeError(f"Don't know how to tensorify task of type {self.task_types[task]}!")
        out_batch.stream_labels = out_labels

        return out_batch

    def collate(self, batch: list[DATA_ITEM_T]) -> PytorchBatch:
        """Combines the ragged dictionaries produced by `__getitem__` into a tensorized batch.

        This function handles conversion of arrays to tensors and padding of elements within the batch across
        static data elements, sequence events, and dynamic data elements.

        Args:
            batch: A list of `__getitem__` format output dictionaries.

        Returns:
            A fully collated, tensorized, and padded batch.
        """

        out_batch = self.__dynamic_only_collate(batch)

        max_n_static = max(len(x["static_indices"]) for x in batch)
        static_padded_fields = defaultdict(list)
        for e in batch:
            n_static = len(e["static_indices"])
            static_delta = max_n_static - n_static
            for k in ("static_indices", "static_measurement_indices"):
                if static_delta > 0:
                    static_padded_fields[k].append(
                        torch.nn.functional.pad(
                            torch.tensor(e[k], dtype=torch.long), (0, static_delta), value=0
                        )
                    )
                else:
                    static_padded_fields[k].append(torch.tensor(e[k], dtype=torch.long))

        for k, v in static_padded_fields.items():
            out_batch[k] = torch.cat([T.unsqueeze(0) for T in v], dim=0)

        return out_batch

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from mixins import SaveableMixin, SeedableMixin, TimeableMixin

from .config import (
    PytorchDatasetConfig,
    SeqPaddingSide,
    SubsequenceSamplingStrategy,
    VocabularyConfig,
)
from .dataset_polars import Dataset
from .types import PytorchBatch

DATA_ITEM_T = dict[str, list[float]]


def to_int_index(col: pl.Expr) -> pl.Expr:
    return col.unique(maintain_order=True).search_sorted(col)


class PytorchDataset(SaveableMixin, SeedableMixin, TimeableMixin, torch.utils.data.Dataset):
    """
    Ultimately, this class will produce a set of batches for ML that will have the following structure:
    {
        'event_mask': [batch_size X seq_len],
        'time_delta': [batch_size X seq_len],

        'static_indices': [batch_size X N_static_data],
        'static_measurement_indices': [batch_size X N_static_data],

        'dynamic_indices': [batch_size X seq_len X N_data],
        'dynamic_values': [batch_size X seq_len X N_data],
        'dynamic_measurement_indices': [batch_size X seq_len X N_data],
        'dynamic_values_mask': [batch_size X seq_len X N_data],

        'stream_labels': {'task_name': [batch_size X N_labels]},
    }

    In this batch, the tensors will have the following types/properties:
    'event_mask'
        boolean type, represents whether or not an event is present at that index in the batch.
    'time_delta'
        floating point type, represents the time_delta in minutes from that subject's prior event for the
        given event in the sequence.

    'dynamic_indices'
        integer type, represents the index of the particular data element measured for the subject's data
        at a given event in the sequence and data element within the event. This index is w.r.t. a unified
        vocabulary.
    'dynamic_values'
        float type, represents a numerical value associated with the data event at the corresponding index. If
        there is no associated numerical value with the given data event, will be `0`.
    'dynamic_measurement_indices'
        integer type, represents the index of the type of data element at the given index. Here, by `type`, we
        mean the underlying vocabulary.
    'dynamic_values_mask'
        boolean type, represents whether or not a data value is present at that position in the batch.

    'stream_labels'

    Note that the data elements, while ordered relative to one another, should be seen as an unordered
    set of data events overall, and processed accordingly.
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

    @classmethod
    def normalize_task(cls, col: pl.Expr, dtype: pl.DataType) -> tuple[str, pl.Expr]:
        for task_type, checkers in cls.TYPE_CHECKERS.items():
            for valid_dtypes, normalize_fn in checkers:
                if dtype in valid_dtypes:
                    return task_type, (col if normalize_fn is None else normalize_fn(col))

        raise TypeError(f"Can't process label of {dtype} type!")

    def __init__(
        self,
        config: PytorchDatasetConfig,
        split: str | None = None,
        vocabulary_config: VocabularyConfig | Path | None = None,
        data: pl.DataFrame | Path | None = None,
        task_df: pd.DataFrame | None = None,
    ):
        """
        Constructs the PytorchDataset).
        Args:
            `data` (): The underlying source dataset.
            `config` (`PytorchDatasetConfig`): Configuration options for the dataset.
            `task_df` (`Optional[pd.DataFrame]`, *optional*, defaults to `None`):
                If specified, this dictates how the raw data is sampled to produce batches and task labels
                (assumed to be stream-prediction level labels) are added to the output batches on the basis of
                the columns in `task_df`. `task_df` must have columns
                    * `subject_id`
                    * `start_time`
                    * `end_time`
                All other columns will be treated as labels and will be normalized then represented in the
                batch.
        """
        super().__init__()

        self.config = config
        do_cache_task_data = False
        self.task_types = {}
        self.task_vocabs = {}
        if data is None:
            assert vocabulary_config is None
            assert self.config.save_dir is not None
            assert split is not None

            data = self.config.save_dir / "DL_reps" / f"{split}*.parquet"

            if self.config.task_df_name is not None:
                task_dir = self.config.save_dir / "DL_reps" / "for_task" / config.task_df_name
                raw_task_df_fp = self.config.save_dir / "task_dfs" / f"{self.config.task_df_name}.parquet"
                task_df_fp = task_dir / f"{split}.parquet"
                task_info_fp = task_dir / "task_info.json"

                if task_df_fp.is_file():
                    print(f"Re-loading task data for {self.config.task_df_name} from {task_df_fp}...")
                    data = task_df_fp
                    with open(task_info_fp) as f:
                        task_info = json.load(f)
                        self.tasks = sorted(task_info["tasks"])
                        self.task_vocabs = task_info["vocabs"]
                        self.task_types = task_info["types"]

                    task_df = "PRE_CACHED_IN_DATA"
                elif raw_task_df_fp.is_file():
                    print(f"Loading raw task df from {raw_task_df_fp}")
                    task_df = pl.scan_parquet(raw_task_df_fp)
                    do_cache_task_data = True
                elif task_df is not None:
                    do_cache_task_data = True
                else:
                    raise FileNotFoundError(
                        f"Neither {task_df_fp} nor {raw_task_df_fp} exist and task_df was not passed!"
                    )

            vocabulary_config = self.config.save_dir / "vocabulary_config.json"

            # TODO(mmd): This is bad as it necessitates loading the dataset regardless of dataset class
            # identity!
            ESD = Dataset._load(self.config.save_dir)
            self.measurement_configs = ESD.measurement_configs
        else:
            assert vocabulary_config is not None
            assert split is None

        self.split = split

        match data:
            case pl.DataFrame():
                self.cached_data = data.lazy()
            case pl.LazyFrame():
                pass
            case Path() as fp if fp.suffix == ".parquet":
                self.cached_data = pl.scan_parquet(fp)
            case Path() as fp if fp.suffix == ".csv":
                self.cached_data = pl.scan_csv(fp)
            case Path() as fp:
                raise ValueError(f"Can't read a dataframe of suffix {fp.suffix}!")
            case _:
                raise TypeError(f"Can't process data of type {type(data)}!")

        if isinstance(vocabulary_config, VocabularyConfig):
            self.vocabulary_config = vocabulary_config
        elif isinstance(vocabulary_config, Path):
            self.vocabulary_config = VocabularyConfig.from_json_file(vocabulary_config)
        else:
            raise TypeError(f"Can't process vocabulary_config of type {type(vocabulary_config)}!")

        self.do_produce_static_data = "static_indices" in self.cached_data.columns
        self.seq_padding_side = config.seq_padding_side
        self.max_seq_len = config.max_seq_len

        if "time" in self.cached_data.columns:
            seq_lens = pl.col("time").arr.lengths()
        else:
            seq_lens = pl.col("time_delta").arr.lengths()

        self.cached_data = self.cached_data.filter(seq_lens >= config.min_seq_len)

        if task_df is None:
            self.task_df = None
            self.tasks = None
        elif type(task_df) is str and task_df == "PRE_CACHED_IN_DATA":
            self.task_df = task_df
        else:
            print("Constructing task-specific cached dataset!")
            if split != "train":
                print(f"WARNING: Constructing task-specific dataset on non-train split {split}!")

            self.task_df = task_df

            self.tasks = sorted(
                [c for c in self.task_df.columns if c not in ["subject_id", "start_time", "end_time"]]
            )

            normalized_cols = []
            for t in self.tasks:
                task_type, normalized_vals = self.normalize_task(col=pl.col(t), dtype=self.task_df.schema[t])
                self.task_types[t] = task_type
                normalized_cols.append(normalized_vals.alias(t))

                if task_type == "binary_classification":
                    self.task_vocabs[t] = [False, True]

            self.task_df = (
                self.task_df.with_columns(normalized_cols)
                .join(self.cached_data.select("subject_id"), on="subject_id", how="inner")
                .with_row_count("task_row_num")
            )

            time_dep_cols = [c for c in ("time", "time_delta") if c in self.cached_data.columns]
            time_dep_cols.extend(["dynamic_indices", "dynamic_values", "dynamic_measurement_indices"])

            if "time" in self.cached_data.columns:
                time_col_expr = pl.col("time")
            elif "time_delta" in self.cached_data.columns:
                time_col_expr = pl.col("time_delta").cumsum().over("subject_id")

            start_idx_expr = time_col_expr.arr.explode().search_sorted(pl.col("start_time_min").first())
            end_idx_expr = time_col_expr.arr.explode().search_sorted(pl.col("end_time_min").first())

            self.cached_data = (
                self.cached_data.join(self.task_df, on="subject_id", how="inner", suffix="_task")
                .with_columns(
                    start_time_min=(pl.col("start_time_task") - pl.col("start_time"))
                    / np.timedelta64(1, "m"),
                    end_time_min=(pl.col("end_time") - pl.col("start_time")) / np.timedelta64(1, "m"),
                )
                .with_row_count("__id")
                .groupby("__id")
                .agg(
                    pl.col("task_row_num").first(),
                    **{c: pl.col(c).first() for c in self.cached_data.columns if c not in time_dep_cols},
                    **{c: pl.col(c).first() for c in self.tasks},
                    **{
                        t: pl.col(t).arr.explode().slice(start_idx_expr, end_idx_expr - start_idx_expr)
                        for t in time_dep_cols
                    },
                )
                .drop("__id")
            )

            self.cached_data = (
                self.cached_data.filter(seq_lens >= config.min_seq_len)
                .sort("task_row_num")
                .drop("task_row_num")
            )

            self.task_df = self.task_df.collect()

            for t in self.tasks:
                if task_type == "multi_class_classification":
                    self.task_vocabs[t] = list(range(self.task_df.select(pl.col(t).max()).item()))

            if self.config.task_df_name is not None and do_cache_task_data:
                task_dir = self.config.save_dir / "DL_reps" / "for_task" / config.task_df_name
                task_df_fp = task_dir / f"{split}.parquet"
                task_info_fp = task_dir / "task_info.json"

                if task_df_fp.is_file():
                    print(f"Re-built existent {task_df_fp} dataset! Not overwriting...")
                else:
                    task_df_fp.parent.mkdir(exist_ok=True, parents=True)
                    self.cached_data.collect().write_parquet(task_df_fp)

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

        if "time_delta" not in self.cached_data.columns:
            self.cached_data = self.cached_data.with_columns(
                (pl.col("start_time") + pl.duration(minutes=pl.col("time").arr.first())).alias("start_time"),
                pl.col("time")
                .arr.eval(
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
            bad_inter_event_times = self.cached_data.filter(pl.col("time_delta").arr.min() <= 0).collect()
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

            self.cached_data = self.cached_data.filter(pl.col("time_delta").arr.min() > 0)

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
            self.cached_data = self.cached_data.drop("subject_id")
            self.columns = self.cached_data.columns
            self.cached_data = self.cached_data.rows()

    @property
    def has_task(self) -> bool:
        return self.task_df is not None

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx: int) -> dict[str, list]:
        return self._seeded_getitem(idx)

    @SeedableMixin.WithSeed
    @TimeableMixin.TimeAs
    def _seeded_getitem(self, idx: int) -> dict[str, list]:
        """
        Returns a dictionary corresponding to a batch element for a single patient. This will not be
        tensorized as that work will need to be re-done in the collate function regardless. The output will
        have structure:
        {
            'time': [seq_len],
            'dynamic_indices': [seq_len, n_data_per_event] (ragged),
            'dynamic_values': [seq_len, n_data_per_event] (ragged),
            'dynamic_measurement_indices': [seq_len, n_data_per_event] (ragged),
            'static_indices': [seq_len, n_data_per_event] (ragged),
            'static_measurement_indices': [seq_len, n_data_per_event] (ragged),
        }
          1. `time` captures the time of the sequence elements.
          2. `dynamic_indices` captures the categorical metadata elements listed in `self.data_cols` in a
             unified vocabulary space spanning all metadata vocabularies.
          3. `dynamic_values` captures the numerical metadata elements listed in `self.data_cols`. If no
             numerical elements are listed in `self.data_cols` for a given categorical column, the according
             index in this output will be `np.NaN`.
          4. `dynamic_measurement_indices` captures which metadata vocabulary was used to source a given data
             element.

        If `self.do_normalize_log_inter_event_times`, then `time` will be approximately modified as follows:
            1. `obs_TTE = time.diff()` Capture the observed inter_event_times.
            2. `mod_TTE = np.exp((np.log(obs_TTE + 1) - self.mean_log_inter...)/self.std_log_inter...)`:
               Modify the times between events by first padding them so none are <= 0, then ensuring that
               their log has mean 0 and standard deviation 1, then re-exponentiating them out of the log
               space.
            3. `mod_time = mod_TTE.cumsum()` Re-sum the modified inter-event times to get modified raw times.
        """

        full_subj_data = {c: v for c, v in zip(self.columns, self.cached_data[idx])}
        if self.config.do_include_start_time_min:
            full_subj_data["start_time"] = full_subj_data["start_time"].timestamp() / 60.0
        else:
            full_subj_data.pop("start_time")

        # If we need to truncate to `self.max_seq_len`, grab a random full-size span to capture that.
        # TODO(mmd): This will proportionally underweight the front and back ends of the subjects data
        # relative to the middle, as there are fewer full length sequences containing those elements.
        seq_len = len(full_subj_data["time_delta"])
        if seq_len > self.max_seq_len:
            with self._time_as("truncate_to_max_seq_len"):
                match self.config.subsequence_sampling_strategy:
                    case SubsequenceSamplingStrategy.RANDOM:
                        start_idx = np.random.choice(seq_len - self.max_seq_len)
                    case SubsequenceSamplingStrategy.TO_END:
                        start_idx = seq_len - self.max_seq_len
                    case SubsequenceSamplingStrategy.FROM_START:
                        start_idx = 0
                    case _:
                        raise ValueError(
                            f"Invalid sampling strategy: {self.config.subsequence_sampling_strategy}!"
                        )

                for k in (
                    "time_delta",
                    "dynamic_indices",
                    "dynamic_values",
                    "dynamic_measurement_indices",
                ):
                    full_subj_data[k] = full_subj_data[k][start_idx : start_idx + self.max_seq_len]

        return full_subj_data

    def __static_and_dynamic_collate(self, batch: list[DATA_ITEM_T]) -> PytorchBatch:
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
        """Combines the ragged dictionaries produced by __getitem__ into a tensorized batch."""
        if self.do_produce_static_data:
            return self.__static_and_dynamic_collate(batch)
        else:
            return self.__dynamic_only_collate(batch)

"""Utilities for collecting baseline performance of fine-tuning tasks defined over ESGPT datasets."""

import copy
import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import polars.selectors as cs
from scipy.stats import bernoulli, loguniform, randint, rv_discrete
from sklearn.decomposition import NMF, PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..data.dataset_polars import Dataset

pl.enable_string_cache(True)


def add_tasks_from(
    tasks_dir: Path, name_prefix: str = "", container: dict[str, pl.LazyFrame] | None = None
) -> dict[str, pl.LazyFrame]:
    """Collects all task dataframes (stored as parquet files) in a nested directory structure.

    Args:
        tasks_dir: The root directory for the tasks tree to be collected.
        name_prefix: What prefix should be prepended to task names when collecting tasks.
        container: Tasks will be added into this container. If `None` (the default), a new container will be
            created. This object **will** be modified.

    Returns:
        The updated (or newly created) container object mapping task names (as indexed by relative file paths)
        to polars lazy frame objects for those dataframes.

    Examples:
        >>> import tempfile, polars as pl
        >>> from datetime import datetime
        >>> from pathlib import Path
        >>> task_1_name = "task"
        >>> task_df_1 = pl.DataFrame({
        ...     'subject_id': [1, 1, 1, 2, 3],
        ...     'start_time': [datetime(2020, 1, 1), None, None, datetime(2020, 3, 2), None],
        ...     'end_time': [
        ...         datetime(2020, 1, 4), datetime(1980, 1, 2), datetime(1991, 2, 5),
        ...         datetime(2022, 1, 1), datetime(2022, 1, 1),
        ...     ],
        ...     'label': [1, 0, 0, 1, 0],
        ... })
        >>> task_2_name = "foobar/foo"
        >>> task_df_2 = pl.DataFrame({
        ...     'subject_id': [1, 2, 4, 2, 3],
        ...     'start_time': [None, None, None, None, None],
        ...     'end_time': [
        ...         datetime(2023, 1, 4), datetime(1984, 1, 2), datetime(1995, 3, 5),
        ...         datetime(2021, 1, 1), datetime(2012, 1, 1),
        ...     ],
        ...     'foo': [0, 5, 19, 2, 1],
        ... })
        >>> task_3_name = "foobar/bar"
        >>> task_df_3 = pl.DataFrame({
        ...     'subject_id': [1, 3, 3],
        ...     'start_time': [None, None, None],
        ...     'end_time': [datetime(2010, 1, 4), datetime(1985, 1, 2), datetime(1931, 2, 5)],
        ...     'bar': [3.12, 8.1, 1.0],
        ... })
        >>> tasks = {task_1_name: task_df_1, task_2_name: task_df_2, task_3_name: task_df_3}
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     base_path = Path(tmpdir)
        ...     for name, task_df in tasks.items():
        ...         task_fp = base_path / f"{name}.parquet"
        ...         task_fp.parent.mkdir(exist_ok=True, parents=True)
        ...         task_df.write_parquet(task_fp)
        ...     read_dfs = add_tasks_from(base_path)
        ...     read_dfs = {k: v.collect() for k, v in read_dfs.items()}
        >>> len(read_dfs)
        3
        >>> read_dfs["task"]
        shape: (5, 4)
        ┌────────────┬─────────────────────┬─────────────────────┬───────┐
        │ subject_id ┆ start_time          ┆ end_time            ┆ label │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---   │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64   │
        ╞════════════╪═════════════════════╪═════════════════════╪═══════╡
        │ 1          ┆ 2020-01-01 00:00:00 ┆ 2020-01-04 00:00:00 ┆ 1     │
        │ 1          ┆ null                ┆ 1980-01-02 00:00:00 ┆ 0     │
        │ 1          ┆ null                ┆ 1991-02-05 00:00:00 ┆ 0     │
        │ 2          ┆ 2020-03-02 00:00:00 ┆ 2022-01-01 00:00:00 ┆ 1     │
        │ 3          ┆ null                ┆ 2022-01-01 00:00:00 ┆ 0     │
        └────────────┴─────────────────────┴─────────────────────┴───────┘
        >>> read_dfs["foobar/foo"]
        shape: (5, 4)
        ┌────────────┬────────────┬─────────────────────┬─────┐
        │ subject_id ┆ start_time ┆ end_time            ┆ foo │
        │ ---        ┆ ---        ┆ ---                 ┆ --- │
        │ i64        ┆ f32        ┆ datetime[μs]        ┆ i64 │
        ╞════════════╪════════════╪═════════════════════╪═════╡
        │ 1          ┆ null       ┆ 2023-01-04 00:00:00 ┆ 0   │
        │ 2          ┆ null       ┆ 1984-01-02 00:00:00 ┆ 5   │
        │ 4          ┆ null       ┆ 1995-03-05 00:00:00 ┆ 19  │
        │ 2          ┆ null       ┆ 2021-01-01 00:00:00 ┆ 2   │
        │ 3          ┆ null       ┆ 2012-01-01 00:00:00 ┆ 1   │
        └────────────┴────────────┴─────────────────────┴─────┘
        >>> read_dfs["foobar/bar"]
        shape: (3, 4)
        ┌────────────┬────────────┬─────────────────────┬──────┐
        │ subject_id ┆ start_time ┆ end_time            ┆ bar  │
        │ ---        ┆ ---        ┆ ---                 ┆ ---  │
        │ i64        ┆ f32        ┆ datetime[μs]        ┆ f64  │
        ╞════════════╪════════════╪═════════════════════╪══════╡
        │ 1          ┆ null       ┆ 2010-01-04 00:00:00 ┆ 3.12 │
        │ 3          ┆ null       ┆ 1985-01-02 00:00:00 ┆ 8.1  │
        │ 3          ┆ null       ┆ 1931-02-05 00:00:00 ┆ 1.0  │
        └────────────┴────────────┴─────────────────────┴──────┘
    """

    if container is None:
        container = {}

    for sub_path in tasks_dir.iterdir():
        if sub_path.is_file() and sub_path.suffix == ".parquet":
            container[f"{name_prefix}{sub_path.stem}"] = pl.scan_parquet(sub_path)
        elif sub_path.is_dir():
            add_tasks_from(sub_path, f"{name_prefix}{sub_path.name}/", container)

    return container


KEY_COLS = ["subject_id", "start_time", "end_time"]


def summarize_binary_task(task_df: pl.LazyFrame):
    """Prints a summary dataframe describing binary tasks.

    Args:
        task_df: The task dataframe in question.

    Examples:
        >>> import polars as pl
        >>> from datetime import datetime
        >>> task_df = pl.DataFrame({
        ...     'subject_id': [1, 1, 1, 2, 3, 4, 4],
        ...     'start_time': [datetime(2020, 1, 1), None, None, datetime(2020, 3, 2), None, None, None],
        ...     'end_time': [
        ...         datetime(2020, 1, 4), datetime(1980, 1, 2), datetime(1991, 2, 5),
        ...         datetime(2022, 1, 1), datetime(2022, 1, 1), None, None,
        ...     ],
        ...     'label': [1, 0, 0, 1, 0, 0, 1],
        ... }).lazy()
        >>> pl.Config.set_tbl_width_chars(80)
        <class 'polars.config.Config'>
        >>> summarize_binary_task(task_df)
        shape: (1, 4)
        ┌───────────────────┬──────────────────────────┬────────────────────┬──────────┐
        │ n_samples_overall ┆ median_samples_per_subje ┆ label/subject Mean ┆ label    │
        │ ---               ┆ ct                       ┆ ---                ┆ ---      │
        │ u32               ┆ ---                      ┆ f64                ┆ f64      │
        │                   ┆ f64                      ┆                    ┆          │
        ╞═══════════════════╪══════════════════════════╪════════════════════╪══════════╡
        │ 7                 ┆ 1.5                      ┆ 0.458333           ┆ 0.428571 │
        └───────────────────┴──────────────────────────┴────────────────────┴──────────┘
    """
    label_cols = [c for c in task_df.columns if c not in KEY_COLS]
    print(
        task_df.groupby("subject_id")
        .agg(
            pl.count().alias("samples_per_subject"),
            *[pl.col(c).mean() for c in label_cols],
        )
        .select(
            pl.col("samples_per_subject").sum().alias("n_samples_overall"),
            pl.col("samples_per_subject").median().alias("median_samples_per_subject"),
            *[pl.col(c).mean().alias(f"{c}/subject Mean") for c in label_cols],
            *[
                (pl.col(c) * pl.col("samples_per_subject")).sum()
                / (pl.col("samples_per_subject").sum()).alias(f"{c} Mean")
                for c in label_cols
            ],
        )
        .collect()
    )


def load_flat_rep(
    ESD: Dataset,
    window_sizes: list[str],
    feature_inclusion_frequency: float | dict[str, float] | None = None,
    include_only_measurements: set[str] | None = None,
    do_update_if_missing: bool = True,
) -> dict[str, pl.LazyFrame]:
    flat_dir = ESD.config.save_dir / "flat_reps"

    feature_inclusion_frequency, include_only_measurements = ESD._resolve_flat_rep_cache_params(
        feature_inclusion_frequency, include_only_measurements
    )

    cache_kwargs = dict(
        feature_inclusion_frequency=feature_inclusion_frequency,
        window_sizes=window_sizes,
        include_only_measurements=include_only_measurements,
        do_overwrite=False,
        do_update=True,
    )

    params_fp = flat_dir / "params.json"
    if not params_fp.is_file():
        if not do_update_if_missing:
            raise FileNotFoundError("Flat representation files haven't been written!")
        else:
            ESD.cache_flat_representation(**cache_kwargs)

    with open(params_fp) as f:
        params = json.load(f)

    needs_more_measurements = not set(include_only_measurements).issubset(params["include_only_measurements"])
    needs_more_features = params["feature_inclusion_frequency"] is not None and (
        (feature_inclusion_frequency is None)
        or any(
            params["feature_inclusion_frequency"].get(m, float("inf")) > m_freq
            for m, m_freq in feature_inclusion_frequency.items()
        )
    )
    needs_more_windows = False
    for window_size in window_sizes:
        if not (flat_dir / "past" / "train" / window_size).is_dir():
            needs_more_windows = True

    if needs_more_measurements or needs_more_features or needs_more_windows:
        ESD.cache_flat_representation(**cache_kwargs)
        with open(params_fp) as f:
            params = json.load(f)

    allowed_features = []
    for meas, cfg in ESD.measurement_configs.items():
        if meas not in include_only_measurements:
            continue

        if cfg.vocabulary is None or feature_inclusion_frequency is None:
            allowed_features.append(meas)
            continue

        vocab = copy.deepcopy(cfg.vocabulary)
        vocab.filter(total_observations=None, min_valid_element_freq=feature_inclusion_frequency[meas])
        allowed_vocab = vocab.vocabulary
        for e in allowed_vocab:
            allowed_features.append(f"{meas}/{e}")

    by_split = {}
    for sp in ESD.split_subjects.keys():
        dfs = []
        for window_size in window_sizes:
            allowed_columns = cs.starts_with("static")
            for feat in allowed_features:
                allowed_columns = allowed_columns | cs.starts_with(f"{window_size}/{feat}")

            window_dir = flat_dir / "past" / sp / window_size
            window_dfs = []
            for fp in window_dir.glob("*.parquet"):
                window_dfs.append(pl.scan_parquet(fp).select("subject_id", "timestamp", allowed_columns))
            dfs.append(pl.concat(window_dfs, how="diagonal"))
        by_split[sp] = pl.concat(dfs, how="align")
    return by_split


WINDOW_OPTIONS = [
    "6h",
    "1d",
    "3d",
    "7d",
    "10d",
    "30d",
    "90d",
    "180d",
    "365d",
    "730d",
    "1825d",
    "3650d",
    "FULL",
]


class WindowSizeDist:
    def __init__(
        self,
        window_options: list[str] = WINDOW_OPTIONS,
        n_windows_dist: rv_discrete = randint(1, 5),
        window_ps: list[float] | None = None,
    ):
        self.window_options = window_options
        self.window_ps = window_ps
        self.n_windows_dist = n_windows_dist

    def rvs(self, size: int = 1, random_state: int | None = None) -> list[str] | list[list[str]]:
        np.random.seed(random_state)
        n_windows = self.n_windows_dist.rvs(size=size, random_state=random_state)
        windows = np.random.choice(self.window_options, size=n_windows, p=self.window_ps)

        if size == 1:
            return list(windows[0])
        else:
            return list([list(x) for x in windows])


class ESDFlatFeatureLoader:
    DEFAULT_HYPERPARAMETER_DIST: dict[str, Any] = {
        "window_sizes": WindowSizeDist(),
        "feature_inclusion_frequency": loguniform(-5, -2),
        "convert_to_mean_var": bernoulli(0.5),
    }

    def __init__(
        self,
        ESD: Dataset,
        window_sizes: list[str] | None = None,
        feature_inclusion_frequency: float | dict[str, float] | None = None,
        include_only_measurements: set[str] | None = None,
        convert_to_mean_var: bool = True,
    ):
        self.ESD = ESD
        self.window_sizes = window_sizes
        self.feature_inclusion_frequency = feature_inclusion_frequency
        self.include_only_measurements = include_only_measurements
        self.convert_to_mean_var = convert_to_mean_var

    def fit(self, flat_rep_df: pl.LazyFrame, _):
        if self.window_sizes is None:
            raise ValueError("Must specify window sizes!")

        self.feature_columns = self.ESD._get_flat_rep_feature_cols(
            feature_inclusion_frequency=self.feature_inclusion_frequency,
            window_sizes=self.window_sizes,
            include_only_measurements=self.include_only_measurements,
        )

    def transform(self, flat_rep_df: pl.LazyFrame) -> np.ndarray:
        out_df = flat_rep_df.select(self.feature_columns)

        if self.convert_to_mean_var:

            def last_part(s: str) -> str:
                return "/".join(s.split("/")[:-1])

            cols = {last_part(c) for c in self.final_col_schema if c.endswith("has_values_count")}
            out_df = out_df.with_columns(
                *[(pl.col(f"{c}/sum") / pl.col(f"{c}/has_values_count")).alias(f"{c}/mean") for c in cols],
                *[
                    (
                        (pl.col(f"{c}/sum_sqd") / pl.col(f"{c}/has_values_count"))
                        - (pl.col(f"{c}/mean") ** 2)
                    ).alias(f"{c}/var")
                    for c in cols
                ],
            ).drop(
                *[f"{c}/sum" for c in cols],
                *[f"{c}/sum_sqd" for c in cols],
                *[f"{c}/has_values_count" for c in cols],
            )

        out_df = out_df.with_columns(
            cs.ends_with("/count").fill_null(0),
            cs.matches(r"static/.*/.*").fill_null(False),
        )

        return out_df.collect(streaming=True).to_numpy()


# Building on
# https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html#sphx-glr-auto-examples-compose-plot-compare-reduction-py
DEFAULT_SKLEARN_PIPELINE = [
    ("scaling", "passthrough"),
    ("imputation", "passthrough"),
    ("reduce_dim", "passthrough"),
    ("model", "passthrough"),
]

SCALING_OPTIONS = [
    MinMaxScaler(),
    StandardScaler(),
]

IMPUTATION_OPTIONS = [
    (
        SimpleImputer(),
        dict(
            stratgy=["constant", "mean", "median", "most_frequent"],
            fill_value=[0],
            add_indicator=[True, False],
        ),
    ),
    (
        KNNImputer(),
        dict(n_neighbors=randint(2, 10), weights=["uniform", "distance"], add_indicator=[True, False]),
    ),
]

DIM_REDUCE_OPTIONS = [
    "passthrough",
    (
        [NMF(), PCA()],
        dict(n_components=randint(2, 256)),
    ),
    (
        SelectKBest(mutual_info_classif),
        dict(k=randint(2, 256)),
    ),
]


def construct_pipeline(
    ESD: Dataset,
    model_cls: Callable,
    model_param_distributions: dict[str, Any],
    scaling_options: list[Any] = SCALING_OPTIONS,
    impute_options: list[Any] = IMPUTATION_OPTIONS,
    dim_reduce_options: list[Any] = DIM_REDUCE_OPTIONS,
) -> tuple[Pipeline, list[dict[str, Any]]]:
    pipe = copy.deepcopy(DEFAULT_SKLEARN_PIPELINE)
    pipe = [("loading_features", ESDFlatFeatureLoader(ESD))] + pipe
    pipe[-1] = ("model", model_cls())

    model_dist = {f"model__{k}": v for k, v in model_param_distributions.items()}

    dist = []
    for scaling_option, impute_option, dim_reduce_option in itertools.product(
        scaling_options, impute_options, dim_reduce_options
    ):
        new_options = copy.deepcopy(model_dist)
        for n, option in [
            ("scaling", scaling_options),
            ("imputation", impute_options),
            ("reduce_dim", dim_reduce_options),
        ]:
            if type(option) is tuple:
                cls_options, cls_params = option
            else:
                cls_options = option
                cls_params = {}

            if type(cls_options) is not list:
                cls_options = [cls_options]
            new_options[n] = cls_options
            for cls_param, cls_param_dist in cls_params.items():
                k = f"{n}__{cls_param}"
                if k in new_options:
                    raise KeyError(f"Key {k} is already present in parameter dict!")

                new_options[k] = cls_param_dist

        dist.append(new_options)

    return Pipeline(pipe), dist


def fit_baseline_task_model(
    task_df: pl.LazyFrame,
    finetuning_task: str,
    ESD: Dataset,
    n_samples: int,
    model_cls: Callable,
    hyperparameter_search_distributions: dict[str, Any],
    window_size_options: list[str] = WINDOW_OPTIONS,
    hyperparameter_search_budget: int = 25,
    n_processes: int = -1,
    seed: int = 1,
):
    # TODO(mmd): Use https://github.com/automl/auto-sklearn

    min_window_size = task_df.select((pl.col("end_time") - pl.col("start_time")).drop_nulls().min()).collect()

    min_window_size = min_window_size.item()

    if min_window_size is not None and min_window_size < max(window_size_options):
        raise ValueError(
            f"Cannot use window sizes {window_size_options} on dataset with min end-start size of "
            f"{min_window_size}."
        )

    flat_reps = load_flat_rep(ESD, window_sizes=window_size_options)
    Xs_and_Ys = {}
    for splits in (["train", "tuning"], ["held_out"]):
        df = pl.concat([flat_reps[sp] for sp in splits], how="vertical").join(
            task_df.select("subject_id", pl.col("end_timestamp").alias("timestamp"), finetuning_task),
            on=["subject_id", "timestamp"],
            how="inner",
        )
        X = df.drop(["subject_id", "timestamp", finetuning_task])
        Y = df[finetuning_task].collect().to_numpy()
        Xs_and_Ys["&".join(splits)] = (X, Y)

    CV = RandomizedSearchCV(
        estimator=model_cls,
        param_distributions=hyperparameter_search_distributions,
        cv=n_samples,
        n_iter=hyperparameter_search_budget,
        n_jobs=n_processes,
        random_state=seed,
    )

    CV.fit(*Xs_and_Ys["train&tuning"])

    return CV

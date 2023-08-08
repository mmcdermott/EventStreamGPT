"""Utilities for collecting baseline performance of fine-tuning tasks defined over ESGPT datasets."""

import copy
import itertools
import json
from collections.abc import Callable
from datetime import datetime
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
    """Loads a set of flat representations from a passed dataset that satisfy the given constraints.

    Args:
        ESD: The dataset for which the flat representations should be loaded.
        window_size: A list of strings in polars timedelta syntax specifying the time windows over which the
            features should be summarized.
        feature_inclusion_frequency: TODO
    """
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

    allowed_features = ESD._get_flat_rep_feature_cols(
        feature_inclusion_frequency=feature_inclusion_frequency,
        window_sizes=window_sizes,
        include_only_measurements=include_only_measurements,
    )

    static_features = [f for f in allowed_features if f.startswith("static/")]
    [f for f in allowed_features if not f.startswith("static/")]

    by_split = {}
    for sp in ESD.split_subjects.keys():
        dfs = []
        for window_size in window_sizes:
            window_dir = flat_dir / "past" / sp / window_size
            window_dfs = []
            for fp in window_dir.glob("*.parquet"):
                window_dfs.append(pl.scan_parquet(fp))

            dfs.append(pl.concat(window_dfs, how="vertical"))

        joined_df = dfs[0]
        for jdf in dfs[1:]:
            joined_df = joined_df.join(jdf.drop(static_features), on=["subject_id", "timestamp"], how="inner")

        by_split[sp] = joined_df

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
    """A random variable that returns lists of window sizes chosen among the passed window options.

    Args:
        window_options: A list of possible window size options (as strings) that can be used.
        n_windows_dist: A random variable that chooses the number of windows that should be included in
            any given sample.
        window_ps: The probability that any individual sample will use a given window size.
    """

    def __init__(
        self,
        window_options: list[str] = WINDOW_OPTIONS,
        n_windows_dist: rv_discrete = randint(1, 5),
        window_ps: list[float] | None = None,
    ):
        self.window_options = window_options
        self.window_ps = window_ps
        self.n_windows_dist = n_windows_dist

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> list[str] | list[list[str]]:
        """Sample a set of `size` window sizes lists.

        Args:
            size: The number of samples to take. Defaults to 1.
            random_state: The numpy random state to use to seed this sampling process.

        Returns:
            A list of sampled lists of window sizes. The list is of length ``size``, unless ``size == 1`` in
            which case only a single sample is returned. This will also be a list (as each sample for this
            distribution is a list) but it will have variable length determined by the sampling process. No
            duplicates will be included in the output list, so if ``self.n_windows_dist`` includes support
            more than the number of total options, excess window size options per sample will be ignored.

        Examples:
            >>> from scipy.stats import randint
            >>> import numpy as np
            >>> wsd = WindowSizeDist(["1d", "3d", "7d"], randint(1, 3))
            >>> wsd.rvs(random_state=1)
            ['1d', '3d']
            >>> wsd.rvs(size=3, random_state=1)
            [['1d', '3d'], ['1d', '3d'], ['7d']]
            >>> wsd = WindowSizeDist(["1d", "2d", "3d", "4d"], randint(1, 3), [0.33, 0.33, 0.33, 0.01])
            >>> wsd.rvs(size=3, random_state=2)
            [['1d'], ['1d', '3d'], ['1d', '2d']]
            >>> wsd = WindowSizeDist(["1d", "3d", "7d"], randint(5, 6))
            >>> wsd.rvs(random_state=1)
            ['1d', '3d', '7d']
        """

        W = len(self.window_options)

        n_windows = self.n_windows_dist.rvs(size=size, random_state=random_state)

        if random_state is None:
            random_state = np.random.default_rng()
        elif type(random_state) is int:
            random_state = np.random.default_rng(random_state)

        windows = [
            list(random_state.choice(self.window_options, size=min(n, W), p=self.window_ps, replace=False))
            for n in n_windows
        ]

        if size == 1:
            return windows[0]
        else:
            return windows


class ESDFlatFeatureLoader:
    """A flat feature pre-processor in line with scikit-learn's APIs.

    This can dynamically apply window size, feature inclusion frequency, measurement restrictions, and mean
    variable conversions to flat feature sets. All window sizes indicated in this featurizer must be included
    in the passed dataframes.
    """

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

    def set_params(
        self,
        ESD: Dataset | None = None,
        window_sizes: list[str] | None = None,
        feature_inclusion_frequency: float | dict[str, float] | None = None,
        include_only_measurements: set[str] | None = None,
        convert_to_mean_var: bool | None = None,
    ):
        if ESD is not None:
            self.ESD = ESD
        if window_sizes is not None:
            self.window_sizes = window_sizes
        if feature_inclusion_frequency is not None:
            self.feature_inclusion_frequency = feature_inclusion_frequency
        if include_only_measurements is not None:
            self.include_only_measurements = include_only_measurements
        if convert_to_mean_var is not None:
            self.convert_to_mean_var = convert_to_mean_var

    def fit(self, flat_rep_df: pl.DataFrame, _) -> "ESDFlatFeatureLoader":
        if self.window_sizes is None:
            raise ValueError("Must specify window sizes!")

        self.feature_columns = self.ESD._get_flat_rep_feature_cols(
            feature_inclusion_frequency=self.feature_inclusion_frequency,
            window_sizes=self.window_sizes,
            include_only_measurements=self.include_only_measurements,
        )

        want_cols = set(self.feature_columns)
        have_cols = set(flat_rep_df.columns)
        if not want_cols.issubset(have_cols):
            missing_cols = list(want_cols - have_cols)
            raise ValueError(
                f"Missing {len(missing_cols)} required columns:\n"
                f"  {', '.join(missing_cols[:5])}{'...' if len(missing_cols) > 5 else ''}."
                f"Have columns:\n{', '.join(flat_rep_df.columns)}"
            )

        flat_rep_df = flat_rep_df.select(self.feature_columns)
        non_null_cols = [s.name for s in flat_rep_df if s.null_count() != flat_rep_df.height]

        self.feature_columns = non_null_cols

        return self

    def transform(self, flat_rep_df: pl.DataFrame) -> np.ndarray:
        out_df = flat_rep_df.lazy().select(self.feature_columns)

        if self.convert_to_mean_var:

            def last_part(s: str) -> str:
                return "/".join(s.split("/")[:-1])

            cols = {last_part(c) for c in self.feature_columns if c.endswith("has_values_count")}
            out_df = (
                out_df.with_columns(
                    *[
                        (pl.col(f"{c}/sum") / pl.col(f"{c}/has_values_count")).alias(f"{c}/mean")
                        for c in cols
                    ],
                )
                .with_columns(
                    *[
                        (
                            (pl.col(f"{c}/sum_sqd") / pl.col(f"{c}/has_values_count"))
                            - (pl.col(f"{c}/mean") ** 2)
                        ).alias(f"{c}/var")
                        for c in cols
                    ],
                )
                .drop(
                    *[f"{c}/sum" for c in cols],
                    *[f"{c}/sum_sqd" for c in cols],
                    *[f"{c}/has_values_count" for c in cols],
                )
            )

        out_df = out_df.with_columns(
            cs.ends_with("/count").fill_null(0),
            cs.matches(r"static/.*/.*").fill_null(False),
        )

        return out_df.collect().to_numpy()


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
            strategy=["constant", "mean", "median", "most_frequent"],
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
        dict(n_components=randint(2, 32)),
    ),
    (
        SelectKBest(mutual_info_classif),
        dict(k=randint(2, 32)),
    ),
]


def fit_baseline_task_model(
    task_df: pl.LazyFrame,
    finetuning_task: str,
    ESD: Dataset,
    n_samples: int,
    model_cls: Callable,
    model_param_distributions: dict[str, Any],
    scaling_options: list[Any] = SCALING_OPTIONS,
    impute_options: list[Any] = IMPUTATION_OPTIONS,
    dim_reduce_options: list[Any] = DIM_REDUCE_OPTIONS,
    window_size_options: list[str] = WINDOW_OPTIONS,
    hyperparameter_search_budget: int = 25,
    n_processes: int = -1,
    seed: int = 1,
    verbose: int = 0,
    error_score: str | float = np.NaN,
    train_subset_size: int | float | None = None,
):
    if type(error_score) is str and error_score != "raise":
        raise ValueError(f"error_score must be either 'raise' or a float; got {error_score}")
    # TODO(mmd): Use https://github.com/automl/auto-sklearn

    print("Checking for validity of window size options...")
    min_window_size = (
        task_df.select((pl.col("end_time") - pl.col("start_time")).drop_nulls().min()).collect().item()
    )

    if min_window_size is not None and min_window_size < max(window_size_options):
        raise ValueError(
            f"Cannot use window sizes {window_size_options} on dataset with min end-start size of "
            f"{min_window_size}."
        )

    print(f"Loading representations for {', '.join(window_size_options)}")
    flat_reps = load_flat_rep(ESD, window_sizes=window_size_options)
    subjects_included = {}
    Xs_and_Ys = {}
    for splits in (["train", "tuning"], ["held_out"]):
        st = datetime.now()
        print(f"Loading datasets for {', '.join(splits)}")
        df = pl.concat([flat_reps[sp] for sp in splits], how="vertical").join(
            task_df.select("subject_id", pl.col("end_time").alias("timestamp"), finetuning_task),
            on=["subject_id", "timestamp"],
            how="inner",
        )
        subject_ids = list(itertools.chain.from_iterable(ESD.split_subjects[sp] for sp in splits))
        if "train" in splits and train_subset_size is not None:
            prng = np.random.default_rng(seed)
            match train_subset_size:
                case int() as n_samples if n_samples > 1:
                    subject_ids = prng.choice(subject_ids, size=n_samples, replace=False)
                case float() as frac if 0 < frac < 1:
                    subject_ids = prng.choice(
                        subject_ids, size=int(frac*len(subject_ids)), replace=False
                    )
                case _:
                    raise ValueError(
                        f"train_subset_size must be either `None`, an int > 1, or a float between 0 and 1; "
                        f"got {train_subset_size}"
                    )
            df = df.filter(pl.col("subject_id").is_in(subject_ids))

        subjects_included["&".join(splits)] = subject_ids

        df = df.collect()

        X = df.drop(["subject_id", "timestamp", finetuning_task])
        Y = df[finetuning_task].to_numpy()
        print(
            f"Done with datasets for {', '.join(splits)} with X of shape {X.shape} "
            f"(elapsed: {datetime.now() - st})"
        )
        Xs_and_Ys["&".join(splits)] = (X, Y)

    # Constructing Pipeline
    pipe = copy.deepcopy(DEFAULT_SKLEARN_PIPELINE)
    pipe = [("loading_features", ESDFlatFeatureLoader(ESD, window_size_options))] + pipe
    pipe[-1] = ("model", model_cls())

    loading_features_dist = {
        "window_sizes": WindowSizeDist(window_options=window_size_options),
        "feature_inclusion_frequency": loguniform(a=1e-7, b=1e-3),
        "convert_to_mean_var": bernoulli(0.5),
    }

    model_dist = {f"model__{k}": v for k, v in model_param_distributions.items()}

    dist = []
    for scaling_option, impute_option, dim_reduce_option in itertools.product(
        scaling_options, impute_options, dim_reduce_options
    ):
        new_options = copy.deepcopy(model_dist)
        new_options.update({f"loading_features__{k}": v for k, v in loading_features_dist.items()})
        for n, option in [
            ("scaling", scaling_option),
            ("imputation", impute_option),
            ("reduce_dim", dim_reduce_option),
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

    pipeline = Pipeline(pipe)
    # print(f"Running with pipeline\n{pipe}\n\nAnd distribution\n{dist}")

    CV = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=dist,
        cv=n_samples,
        n_iter=hyperparameter_search_budget,
        n_jobs=n_processes,
        random_state=seed,
        verbose=verbose,
        error_score=error_score,
    )

    print("Fitting model!")
    CV.fit(*Xs_and_Ys["train&tuning"])

    return CV, subjects_included, Xs_and_Ys

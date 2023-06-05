from __future__ import annotations

import dataclasses

import pandas as pd
import plotly.express as px
import polars as pl
from plotly.graph_objs._figure import Figure

from ..utils import JSONableMixin


@dataclasses.dataclass
class Visualizer(JSONableMixin):
    """A visualization configuration and plotting class.

    This class helps visualize `Dataset` objects. It is both a configuration object and performs
    the actual data manipulations for final visualization, interfacing only with the `Dataset`
    object to obtain appropriately sampled and processed cuts of the data to visualize. It currently
    produces the following plots. All plots are broken down by `static_covariates`, which are
    covariates that are constant for each subject.

    ## Analyzing the data over time (only produced if `plot_by_time` is True)
    Given an $x$-axis of time $t$, the following plots are produced:
        - "Active Subjects": $y$ = the number of active subjects at time $x$ (i.e. the number of subjects who
          have at least one event before $t$ and have not yet had their last event at $t$).
        - "Cumulative Subjects": $y$ = the number of cumulative subjects at time $t$ (i.e., the number of
          subjects who have at least one event before $t$).
        - "Cumulative Events": $y$ = the number of events the dataset would obtain were it to be terminated
          at time $t$.
        - "Events / Subject": $y$ = the average number of events per subject as would be observed were the
          dataset to be terminated at time $t$.
        - "Events / (Subject, Time)": $y$ = the average rate of events per unit time per subject at time
          $t$
        - "Age Distribution over Time": A 2D Density Heatmap plot showing the distributions of the ages of
          active subjects in the dataset at time $t$. Only produced if `age_col` is specified. Age is binned
          into `n_age_buckets` buckets.

    ## Analyzing the data over age (only produced if `plot_by_age` is True)
    Given an $x$-axis of age bucket $a$, the following plots are produced:
        - "Cumulative Subjects": $y$ = the number of subjects in the dataset who have an event in the age
          bucket $a$.
        - "Cumulative Events": $y$ = the number of events included in the dataset that occur at an age up to
          or before $a$.
        - "Events / Subject": $y$ = the average number of events per subject that occur when the subject is at
          age bucket $a$.

    Attributes:
        subset_size: When plotting, use an IID random subsample (over subjects) of the input dataset of this
            size. This makes plotting much faster, and is statistically unbiased, though can increase
            variance.
        subset_random_seed: If subsampling the raw data, use this random seed to control that subsampling.
        static_covariates: When plotting, split plots by these static covariates.
        plot_by_time: If `True`, also plot how the dataset changes over time.
        time_unit: If `plot_by_time` is `True`, aggregate timepoints into buckets of this size.
        plot_by_age: If `True`, plot how datasret characteristics evolve with subject age.
        age_col: The column in the Dataset's `events_df` where age is stored. This should typically be the
            name of the measurement employing the `AgeFunctor` time dependent functor object, unless age is
            pre-computed in the dataset.
        dob_col: This is used to compute ages of subjects at inferred timepoints created dynamically during
            plotting. This string should point to the date of birth (in datetime format) within the subjects
            dataframe.
        n_age_buckets: If `plot_by_age` is `True`, this controls how many buckets ages are discretized into to
            limit plot granularity.
        min_sub_to_plot_age_dist: If set, do not plot sub-population distributions over age if the total
            number of patients in the sub-population is below this value. Useful for limiting variance.

    Raises:
        ValueError: If
            * `subset_size` is specified but `subset_random_seed` is not.
            * `plot_by_age` is `True`, but `age_col` or `n_age_buckets` is `None`
            * `age_col` is specified but `dob_col` is not
            * `plot_by_time` is `True`, but `time_unit` is None

    Examples:
        >>> V = Visualizer()
        >>> V = Visualizer(
        ...     subset_size=100, subset_random_seed=1,
        ...     plot_by_age=True, age_col='age', dob_col='dob', n_age_buckets=100,
        ...     plot_by_time=True, time_unit='1y',
        ... )
        >>> V = Visualizer(subset_size=100)
        Traceback (most recent call last):
            ...
        ValueError: subset_size is specified, but subset_random_seed is not!
        >>> V = Visualizer(plot_by_age=True, age_col='age', n_age_buckets=None)
        Traceback (most recent call last):
            ...
        ValueError: plot_by_age is True, but n_age_buckets is unspecified!
        >>> V = Visualizer(age_col='age')
        Traceback (most recent call last):
            ...
        ValueError: age_col is specified, but dob_col is not!
        >>> V = Visualizer(plot_by_time=True, time_unit=None)
        Traceback (most recent call last):
            ...
        ValueError: plot_by_time is True, but time_unit is unspecified!
    """

    subset_size: int | None = None
    subset_random_seed: int | None = None

    static_covariates: list[str] = dataclasses.field(default_factory=list)

    plot_by_time: bool = True
    time_unit: str | None = "1y"

    plot_by_age: bool = False
    age_col: str | None = None
    dob_col: str | None = None
    n_age_buckets: int | None = 200

    min_sub_to_plot_age_dist: int | None = 50

    def __post_init__(self):
        if self.subset_size is not None and self.subset_random_seed is None:
            raise ValueError("subset_size is specified, but subset_random_seed is not!")
        if self.plot_by_age:
            if self.age_col is None:
                raise ValueError("plot_by_age is True, but age_col is unspecified!")
            if self.n_age_buckets is None:
                raise ValueError("plot_by_age is True, but n_age_buckets is unspecified!")
        if self.age_col is not None and self.dob_col is None:
            raise ValueError("age_col is specified, but dob_col is not!")
        if self.plot_by_time and self.time_unit is None:
            raise ValueError("plot_by_time is True, but time_unit is unspecified!")

    @staticmethod
    def _normalize_to_pandas(df: pl.DataFrame, covariate: str | None = None) -> pd.DataFrame:
        df = df.to_pandas()

        if covariate is None:
            return df

        if df[covariate].isna().any():
            if "UNK" not in df[covariate].cat.categories:
                df[covariate] = df[covariate].cat.add_categories("UNK")

            df[covariate] = df[covariate].fillna("UNK")
        df[covariate] = df[covariate].cat.remove_unused_categories()

        return df

    def plot_counts_over_time(self, in_events_df: pl.DataFrame) -> list[Figure]:
        figures = []
        if not self.plot_by_time:
            return figures

        in_events_df = (
            in_events_df.sort("timestamp", descending=False)
            .with_columns(
                pl.when(
                    (pl.col("timestamp") == pl.col("start_time"))
                    & (pl.col("timestamp") == pl.col("end_time"))
                )
                .then(0)
                .when(pl.col("timestamp") == pl.col("start_time"))
                .then(1)
                .when(pl.col("timestamp") == pl.col("end_time"))
                .then(-1)
                .otherwise(0)
                .alias("active_subj_increment"),
                pl.when(pl.col("timestamp") == pl.col("start_time"))
                .then(1)
                .otherwise(0)
                .alias("cumulative_subj_increment"),
            )
            .groupby_dynamic(
                index_column="timestamp",
                every=self.time_unit,
                by=self.static_covariates,
            )
            .agg(
                pl.col("subject_id").n_unique().alias("n_subjects"),
                pl.col("event_id").n_unique().alias("n_events"),
                pl.col("active_subj_increment").sum().alias("active_subjects_delta"),
                pl.col("cumulative_subj_increment").sum().alias("cumulative_subjects_delta"),
            )
            .sort("timestamp", descending=False)
        )

        for static_covariate in self.static_covariates:
            plt_kwargs = {"x": "timestamp", "color": static_covariate}

            events_df = (
                in_events_df.groupby("timestamp", static_covariate)
                .agg(
                    pl.col("n_subjects").sum(),
                    pl.col("n_events").sum(),
                    pl.col("active_subjects_delta").sum(),
                    pl.col("cumulative_subjects_delta").sum(),
                )
                .with_columns(
                    (pl.col("n_events") / pl.col("n_subjects")).alias("events_per_subject_per_time"),
                )
                .sort("timestamp", descending=False)
            )

            # "Active Subjects": $y$ = the number of active subjects at time $x$ (i.e. the number of subjects
            # who have at least one event before $t$ and have not yet had their last event at $t$).
            # "Cumulative Subjects": $y$ = the number of cumulative subjects at time $t$ (i.e., the number of
            # subjects who have at least one event before $t$).
            subjects_as_of_time = self._normalize_to_pandas(
                events_df.select(
                    "timestamp",
                    static_covariate,
                    pl.col("active_subjects_delta").cumsum().over(static_covariate).alias("Active Subjects"),
                    pl.col("cumulative_subjects_delta")
                    .cumsum()
                    .over(static_covariate)
                    .alias("Cumulative Subjects"),
                ),
                static_covariate,
            )

            figures.extend(
                [
                    px.line(subjects_as_of_time, y="Active Subjects", **plt_kwargs),
                    px.line(subjects_as_of_time, y="Cumulative Subjects", **plt_kwargs),
                ]
            )

            # "Cumulative Events": $y$ = the number of events the dataset would obtain were it to be
            # terminated at time $t$.
            # "Events / Subject": $y$ = the average number of events per subject as would be observed were the
            # dataset to be terminated at time $t$.
            # "Events / (Subject, Time)": $y$ = the average rate of events per unit time per subject at time
            # $t$

            events_as_of_time = self._normalize_to_pandas(
                events_df.select(
                    "timestamp",
                    static_covariate,
                    pl.col("n_events").cumsum().over(static_covariate).alias("Cumulative Events"),
                    (
                        pl.col("n_events").cumsum().over(static_covariate)
                        / pl.col("cumulative_subjects_delta").cumsum().over(static_covariate)
                    ).alias("Average Events / Subject"),
                    pl.col("events_per_subject_per_time").alias("New Events / Subject / time"),
                ),
                static_covariate,
            )

            figures.extend(
                [
                    px.line(events_as_of_time, y="Cumulative Events", **plt_kwargs),
                    px.line(events_as_of_time, y="Average Events / Subject", **plt_kwargs),
                    px.line(events_as_of_time, y="New Events / Subject / time", **plt_kwargs),
                ]
            )

        return figures

    def plot_age_distribution_over_time(
        self, subjects_df: pl.DataFrame, subj_ranges: pl.DataFrame
    ) -> list[Figure]:
        figures = []
        if not self.plot_by_time:
            return figures
        if self.dob_col is None:
            return figures

        start_time = subj_ranges["start_time"].min()
        end_time = subj_ranges["end_time"].max()

        subj_ranges = subj_ranges.join(
            subjects_df.select("subject_id", self.dob_col, *self.static_covariates),
            on="subject_id",
        )

        time_points = pl.DataFrame(
            {"timestamp": pl.date_range(start_time, end_time, interval=self.time_unit)}
        )
        n_time_bins = len(time_points) + 1

        cross_df_all = (
            subj_ranges.join(time_points, how="cross")
            .filter(
                (pl.col("start_time") <= pl.col("timestamp")) & (pl.col("timestamp") <= pl.col("end_time"))
            )
            .select(
                "timestamp",
                "subject_id",
                *self.static_covariates,
                (
                    (pl.col("timestamp") - pl.col(self.dob_col)).dt.nanoseconds()
                    / (1e9 * 60 * 60 * 24 * 365.25)
                ).alias(self.age_col),
                pl.col("subject_id").n_unique().over("timestamp").alias("num_subjects"),
            )
            .filter(pl.col("num_subjects") > 20)
        )

        for static_covariate in self.static_covariates:
            cross_df = (
                cross_df_all.with_columns(
                    pl.col("subject_id").n_unique().over("timestamp", static_covariate).alias("num_subjects")
                )
                .filter(pl.col("num_subjects") > 20)
                .with_columns((1 / pl.col("num_subjects")).alias("% Subjects @ time"))
            )

            if self.min_sub_to_plot_age_dist is not None:
                val_counts = subjects_df[static_covariate].value_counts()
                valid_categories = val_counts.filter(pl.col("counts") > self.min_sub_to_plot_age_dist)[
                    static_covariate
                ].to_list()

                cross_df = cross_df.filter(pl.col(static_covariate).is_in(valid_categories))

            figures.append(
                px.density_heatmap(
                    self._normalize_to_pandas(cross_df, static_covariate),
                    x="timestamp",
                    y=self.age_col,
                    z="% Subjects @ time",
                    facet_col=static_covariate,
                    nbinsy=self.n_age_buckets,
                    nbinsx=n_time_bins,
                    histnorm=None,
                    histfunc="sum",
                )
            )

        return figures

    def plot_static_variables_breakdown(self, static_variables: pl.DataFrame) -> list[Figure]:
        figures = []
        if not self.static_covariates:
            return

        for static_covariate in self.static_covariates:
            df = static_variables.groupby(static_covariate).agg(
                pl.col("subject_id").n_unique().alias("# Subjects")
            )
            figures.append(
                px.bar(
                    self._normalize_to_pandas(df, static_covariate),
                    x=static_covariate,
                    y="# Subjects",
                )
            )
        return figures

    def plot_counts_over_age(self, events_df: pl.DataFrame) -> list[Figure]:
        figures = []
        if not self.plot_by_age:
            return figures

        min_age = events_df[self.age_col].min()
        max_age = events_df[self.age_col].max()
        age_bucket_size = (max_age - min_age) / (self.n_age_buckets)

        events_df = (
            events_df.with_columns(
                (pl.col("age") / age_bucket_size).round(0).cast(pl.Int64, strict=False).alias("age_bucket"),
                pl.col("subject_id").n_unique().over(*self.static_covariates).alias("total_n_subjects"),
            )
            .drop_nulls("age_bucket")
            .groupby("age_bucket", *self.static_covariates)
            .agg(
                pl.col(self.age_col).mean(),
                pl.col("event_id").n_unique().alias("n_events"),
                pl.col("subject_id").n_unique().alias("n_subjects_at_age"),
                pl.col("total_n_subjects").first(),
            )
            .sort(by=self.age_col, descending=False)
            .with_columns(
                pl.col("n_events").cumsum().over(*self.static_covariates).alias("Cumulative Events"),
            )
        )

        for static_covariate in self.static_covariates:
            plt_kwargs = {"x": self.age_col, "color": static_covariate}

            counts_at_age = self._normalize_to_pandas(
                events_df.groupby("age_bucket", static_covariate)
                .agg(
                    (
                        (pl.col(self.age_col) * pl.col("n_subjects_at_age")).sum()
                        / pl.col("n_subjects_at_age").sum()
                    ).alias(self.age_col),
                    pl.col("n_subjects_at_age").sum().alias("Subjects with Event @ Age"),
                    pl.col("n_events").sum().alias("Events @ Age"),
                    pl.col("Cumulative Events").sum().alias("Events <= Age"),
                    pl.col("total_n_subjects").sum().alias("Total Subjects"),
                )
                .with_columns(
                    (pl.col("Subjects with Event @ Age") / pl.col("Total Subjects")).alias(
                        "% Subjects with Event @ Age"
                    ),
                    (pl.col("Events @ Age") / pl.col("Subjects with Event @ Age")).alias(
                        "Events @ Age / (Subjects with >= 1 Event @ Age)"
                    ),
                    (pl.col("Events @ Age") / pl.col("Total Subjects")).alias("Events @ Age / Subject"),
                    (pl.col("Events <= Age") / pl.col("Total Subjects")).alias("Events <= Age / Subject"),
                )
                .sort(self.age_col, descending=False),
                static_covariate,
            )

            figures.extend(
                [
                    px.line(counts_at_age, y="% Subjects with Event @ Age", **plt_kwargs),
                    px.line(counts_at_age, y="Events @ Age / Subject", **plt_kwargs),
                    px.line(counts_at_age, y="Events <= Age / Subject", **plt_kwargs),
                    px.line(
                        counts_at_age,
                        y="Events @ Age / (Subjects with >= 1 Event @ Age)",
                        **plt_kwargs,
                    ),
                ]
            )

        return figures

    def plot_events_per_patient(self, events_df: pl.DataFrame) -> list[Figure]:
        events_per_patient = events_df.groupby("subject_id", *self.static_covariates).agg(
            pl.col("event_id").n_unique().alias("# of Events")
        )

        return [
            px.histogram(self._normalize_to_pandas(events_per_patient, c), x="# of Events", color=c)
            for c in self.static_covariates
        ]

    def plot(
        self,
        subjects_df: pl.DataFrame,
        events_df: pl.DataFrame,
        dynamic_measurements_df: pl.DataFrame,
    ) -> list[Figure]:
        subj_ranges = events_df.groupby("subject_id").agg(
            pl.col("timestamp").min().alias("start_time"),
            pl.col("timestamp").max().alias("end_time"),
        )

        static_variables = subj_ranges.join(
            subjects_df.select("subject_id", *self.static_covariates), on="subject_id"
        )

        events_df = events_df.join(static_variables, on="subject_id")

        figs = []
        figs.extend(self.plot_static_variables_breakdown(static_variables))
        figs.extend(self.plot_counts_over_time(events_df))
        figs.extend(self.plot_age_distribution_over_time(subjects_df, subj_ranges))
        figs.extend(self.plot_counts_over_age(events_df))
        figs.extend(self.plot_events_per_patient(events_df))

        return figs

"""This file contains code to aid in longitudinal, MCF-based evaluation over measurement predicates."""

import numpy as np
import polars as pl

RANGE_T = tuple[None | tuple[float, bool] | float, None | tuple[float, bool]]


def crps(samples: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Computes the Continuous Ranked Probability Score (CRPS) [1].

    Given an empirical distribution and a true observation, this computes the CRPS between the two. For a
    single sample, this reduces to absolute error. The empirical distribution should be arranged such that
    independent samples of the distribution are on the first axis, and all other axes should be equal.

    Initial Source: https://docs.pyro.ai/en/stable/_modules/pyro/ops/stats.html#crps_empirical

    [1] Tilmann Gneiting, Adrian E. Raftery (2007)
        `Strictly Proper Scoring Rules, Prediction, and Estimation`
        https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

    Args:
        samples: A numpy array of shape (n_samples, ...) containing the drawn empirical samples for the
            distribution in question. May contain NaNs, which represents missing or censored samples.
        true: A numpy array of shape (...) containing true observations. May contain NaNs, which represent
            missing or censored true observations.

    Returns:
        A numpy array of shape (...) containing the CRPS score results for the true observations and empirical
            distributions corresponding to each position. Will be NaN if either the true observation was NaN
            at that position or if all sampled observations were NaN at that position.

    Raises:
        ValueError: If the shape of ``true`` does not match the shape of ``samples`` absent the first
            dimension.

    Examples:
        >>> import numpy as np
        >>> true = np.array([0])
        >>> samples = np.array([[-2]])
        >>> crps(samples, true)
        array([2])
        >>> true = np.array([0])
        >>> samples = np.array([[-2], [np.NaN], [np.NaN], [1], [2]])
        >>> crps(samples, true)
        array([0.77777778])
        >>> true = np.array([0])
        >>> samples = np.array([[-2], [-1], [0], [1], [2]])
        >>> crps(samples, true)
        array([0.4])
        >>> true = np.array([-2, 0, -2, np.NaN])
        >>> samples = np.array([
        ...     [-1, 1,  -1,      -1],
        ...     [1, -2,   1,       1],
        ...     [2, -20,  np.NaN,  2],
        ...     [0,  10,  0,       0],
        ...     [3,  1,   3,       3],
        ...     [1,  1,   1,       1]
        ... ])
        >>> crps(samples, true)
        array([2.27777778, 1.41666667, 2.08      ,        nan])
        >>> crps(np.array([-2, -1, 0, 1, 2]), true)
        Traceback (most recent call last):
            ...
        ValueError: The shape of true (4,) must match that of samples (5,) after the 1st dimension.
    """

    if true.shape != samples.shape[1:]:
        raise ValueError(
            f"The shape of true {true.shape} must match that of samples {samples.shape} after "
            "the 1st dimension."
        )

    if samples.shape[0] == 1:
        return np.abs(samples[0] - true)

    n_samples = (~np.isnan(samples)).sum(0)

    samples = np.sort(samples, axis=0)
    diff = samples[1:] - samples[:-1]

    counting_up = np.ones_like(samples).cumsum(0)[:-1]
    lhs = counting_up - (np.isnan(samples).sum(0))
    lhs = np.where(lhs > 0, lhs, np.NaN)

    rhs = np.where(~np.isnan(lhs), np.flip(counting_up, 0), np.NaN)
    weight = np.flip(lhs * rhs, 0)

    abs_error = np.nanmean(np.abs(true - samples), 0)
    return abs_error - (np.nansum(diff * weight, axis=0) / n_samples**2)


def get_MCF(
    aligned_Ts: list[float], MCF_cols: list[str], *dfs: list[pl.DataFrame]
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the population censor mask and the cumulative predicate incidence delta function for dfs.

    Args:
        aligned_Ts: The timestamps for which the final MCF and censoring mask should be computed.
        MCF_cols: A list of `pl.List[pl.Boolean]` columns in the dataframes to compute the MCF over.
        dfs: A list of dataframes to include in the final MCF. Each must be in the same order and have columns
            ``time``, and ``MCF_cols[i]`` for all ``i``.

    Returns:
        1. A boolean numpy array of shape ``(len(dfs), dfs[0].shape[0], len(aligned_Ts))`` which contains a 1
            at index ``[n, i, j]`` if subject ``i`` has any data at or after time ``aligned_Ts[j]`` in
            ``dfs[n]``.
        2. A uint numpy array of shape ``(len(dfs), dfs[0].shape[0], len(aligned_Ts), len(MCF_cols))`` such
            that the value at index ``[n, i, j, k]`` is the count of new instances where ``MCF_cols[k]`` is
            True for subject ``i`` between time ``aligned_Ts[j-1]`` (or negative infinity if ``j == 0``) and
            ``aligned_Ts[j]`` in ``dfs[n]``.

    Examples:
        >>> df_1 = pl.DataFrame({
        ...     "subject_id": [1, 2],
        ...     "time": [
        ...         [-3.2, -2, 0, 10.2],
        ...         [0., 1.],
        ...     ],
        ...     "pred_1": [
        ...         [False, True, True, False],
        ...         [True, True],
        ...     ],
        ...     "pred_2": [
        ...         [True, False, False, True],
        ...         [False, False],
        ...     ],
        ... })
        >>> df_2 = pl.DataFrame({
        ...     "subject_id": [1, 2],
        ...     "time": [
        ...         [-1.9, 0., 0.2],
        ...         [-10., 0., 2.3],
        ...     ],
        ...     "pred_1": [
        ...         [False, True, False],
        ...         [True, True, False],
        ...     ],
        ...     "pred_2": [
        ...         [True, False, True],
        ...         [True, False, False],
        ...     ],
        ... })
        >>> aligned_Ts = [-3, 3, 6, 10]
        >>> out = get_MCF(aligned_Ts, ["pred_1", "pred_2"], df_1, df_2)
        >>> print(f"Got a {type(out)} of len {len(out)}")
        Got a <class 'tuple'> of len 2
        >>> out[0]
        array([[[ True,  True,  True,  True,  True],
                [ True,  True, False, False, False]],
        <BLANKLINE>
               [[ True,  True, False, False, False],
                [ True,  True, False, False, False]]])
        >>> out[1]
        array([[[[ 0.,  1.],
                 [ 2.,  0.],
                 [ 0.,  0.],
                 [ 0.,  0.],
                 [ 0.,  1.]],
        <BLANKLINE>
                [[nan, nan],
                 [ 2.,  0.],
                 [ 0.,  0.],
                 [ 0.,  0.],
                 [nan, nan]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[nan, nan],
                 [ 1.,  2.],
                 [ 0.,  0.],
                 [ 0.,  0.],
                 [ 0.,  0.]],
        <BLANKLINE>
                [[ 1.,  1.],
                 [ 1.,  0.],
                 [ 0.,  0.],
                 [ 0.,  0.],
                 [ 0.,  0.]]]])
    """

    time_outputs = aligned_Ts + [float("inf")]
    output_col_names = [str(i) for i in range(len(time_outputs))]

    censor_slices, MCF_slices = [], []
    for df in dfs:
        censor_slices.append(
            df.with_columns(max_time=pl.col("time").list.max())
            .sort(by=["subject_id"])
            .select(
                pl.lit(True), *[(pl.col("max_time") >= t).alias(str(i)) for i, t in enumerate(aligned_Ts)]
            )
            .to_numpy()
        )

        MCF_idx_slices = []

        exploded_MCF_df = (
            df.select("subject_id", "time", *MCF_cols)
            .explode("time", *MCF_cols)
            .with_columns(
                pl.lit(output_col_names)
                .take(pl.lit(aligned_Ts).search_sorted(pl.col("time")))
                .alias("aligned_time_bucket")
            )
        )

        for MCF_col in MCF_cols:
            MCF_df = exploded_MCF_df.pivot(
                index="subject_id",
                columns="aligned_time_bucket",
                values=MCF_col,
                aggregate_function="sum",
            ).sort(by="subject_id")

            MCF_idx_slices.append(
                MCF_df.with_columns(
                    pl.lit(False).alias(c) for c in output_col_names if c not in MCF_df.columns
                )
                .select(output_col_names)
                .to_numpy()
            )

        MCF_slices.append(np.stack(MCF_idx_slices, axis=-1))

    return np.stack(censor_slices, axis=0), np.stack(MCF_slices, axis=0)


def get_aligned_timestamps(
    control_T: pl.Series, *sample_Ts: list[pl.Series], n_timestamps: int | None = None
) -> list[float]:
    """Gets the aligned timestamps given the input raw timestamps.

    Args:
        control_T: the timestamps from the control population, as a series of lists.
        sample_Ts: any sample timestamps to also be included.
        n_timestamps: If specified, downsample the provided timestamps to no more than this many.

    Returns:
        A sorted list of time values.

    Examples:
        >>> control_T = pl.Series([
        ...     [-10., 0, 1, 2], [-105, 1, 4],
        ... ])
        >>> sample_T_1 = pl.Series([
        ...     [8, 21.1], [46, 132, 188, 200.],
        ... ])
        >>> sample_T_2 = pl.Series([
        ...     [1.1], None
        ... ])
        >>> get_aligned_timestamps(control_T, sample_T_1, sample_T_2)
        [-105.0, -10.0, 0.0, 1.0, 1.1, 2.0, 4.0, 8.0, 21.1, 46.0, 132.0, 188.0, 200.0]
        >>> get_aligned_timestamps(control_T, sample_T_1, sample_T_2, n_timestamps=40)
        [-105.0, -10.0, 0.0, 1.0, 1.1, 2.0, 4.0, 8.0, 21.1, 46.0, 132.0, 188.0, 200.0]
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> get_aligned_timestamps(control_T, sample_T_1, sample_T_2, n_timestamps=4)
        [1.1, 2.0, 4.0, 46.0]
    """

    def get_Ts(S: pl.Series) -> list:
        return S.explode().drop_nulls().to_list()

    all_Ts = list(set(get_Ts(control_T)).union(*[get_Ts(T) for T in sample_Ts]))
    if n_timestamps is not None and len(all_Ts) > n_timestamps:
        all_Ts = list(np.random.choice(all_Ts, size=n_timestamps, replace=False))

    return sorted(all_Ts)


def eval_range(
    rng: RANGE_T,
    val: pl.Expr,
) -> pl.Expr:
    """Returns true if val satisfies the range rng.

    Args:
        rng: The range in question. If it is a boolean, it is returned directly, otherwise True is returned if
            val is in the described range.
        val: The value to evaluate.
    Returns:
        True if and only if value satisfies the range.

    Examples:
        >>> pl.select(eval_range(True, pl.lit(0.1))).item()
        True
        >>> pl.select(eval_range(False, pl.lit(0.1))).item()
        False
        >>> pl.select(eval_range((1, 2), pl.lit(0.1))).item()
        False
        >>> pl.select(eval_range((None, 2), pl.lit(0.1))).item()
        True
        >>> pl.select(eval_range((1, 2), pl.lit(1))).item()
        False
        >>> pl.select(eval_range(((1, False), 2), pl.lit(1))).item()
        False
        >>> pl.select(eval_range(((1, True), 2), pl.lit(1))).item()
        True
        >>> pl.select(eval_range((1, 2), pl.lit(3))).item()
        False
        >>> pl.select(eval_range((1, None), pl.lit(3))).item()
        True
    """

    if type(rng) is bool:
        return pl.lit(rng)

    lower_bound, upper_bound = rng

    if lower_bound is None and upper_bound is None:
        return pl.lit(True)

    expr = []

    match lower_bound:
        case None:
            pass
        case float() | int() as bound, bool() as incl:
            if incl:
                expr.append(val >= bound)
            else:
                expr.append(val > bound)
        case float() | int() as bound:
            expr.append(val > bound)
        case _:
            raise ValueError(f"{lower_bound} must be either None, a number, or a (number, bool)!")

    match upper_bound:
        case None:
            pass
        case float() | int() as bound, bool() as incl:
            if incl:
                expr.append(val <= bound)
            else:
                expr.append(val < bound)
        case float() | int() as bound:
            expr.append(val < bound)
        case _:
            raise ValueError(f"{upper_bound} must be either None, a number, or a (number, bool)!")

    return pl.all(*expr)


def align_time_and_eval_predicates(
    df: pl.DataFrame,
    measurement_predicates: dict[int, bool | RANGE_T],
) -> pl.DataFrame:
    """Adjusts the input DataFrame's time column and evaluates the measurement predicates.

    Args:
        df: The dataframe to be adjusted. Must have the columns ``subject_id``, ``time``, ``dynamic_indices``,
            ``dynamic_values``, and ``align_time``.
        measurement_predicates: A dictionary from dynamic measurement index to either the boolean True, in
            which case the presence of the measurement is used alone, or a range dictating
            bounds for the measurement's value to satisfy the predicate. The range is in the format
            ``(LOWER_BOUND, UPPER_BOUND)``, where ``*_BOUND`` can be either `None` (in which case there is no
            bound on that side), a floating point value (in which case the bound is considered to be
            exclusive), or a tuple of a floating point value and a boolean value where the boolean value
            indicates an inclusive or exclusive bound.

    Returns:
        A modified dataframe such that the elements of the (nested) time column are normalized such that ``0``
        indicates a time value of ``align_time`` and such that the dynamic indices and values columns are
        replaced by a set of boolean list columns detailing whether or not the event at that index satisfies
        the given predicate.

    Examples:
        >>> df = pl.DataFrame({
        ...     'subject_id': [1, 2, 3],
        ...     'time': [
        ...         [0., 10, 20],
        ...         [0., 100],
        ...         [0., 1, 2, 3],
        ...     ],
        ...     'dynamic_indices': [
        ...         [[1, 2], [3, 3, 2], [4]],
        ...         [[1], [3]],
        ...         [[2, 3], [1], [8], [3, 1, 1]],
        ...     ],
        ...     'dynamic_values': [
        ...         [[None, 0], [-1, 4, 0.2], [None]],
        ...         [[None], [3]],
        ...         [[-0.1, 10], [None], [None], [6, None, None]],
        ...     ],
        ...     'align_time': [10, 100, 1.5],
        ... })
        >>> measurement_predicates = {
        ...     3: (3.5, None),
        ...     1: True,
        ... }
        >>> out = align_time_and_eval_predicates(df, measurement_predicates)
        >>> pl.Config.set_tbl_width_chars(80)
        <class 'polars.config.Config'>
        >>> out
        shape: (3, 4)
        ┌────────────┬─────────────────────┬─────────────────┬─────────────────────────┐
        │ subject_id ┆ time                ┆ pred_3          ┆ pred_1                  │
        │ ---        ┆ ---                 ┆ ---             ┆ ---                     │
        │ i64        ┆ list[f64]           ┆ list[bool]      ┆ list[bool]              │
        ╞════════════╪═════════════════════╪═════════════════╪═════════════════════════╡
        │ 1          ┆ [-10.0, 0.0, 10.0]  ┆ [false, true,   ┆ [true, false, false]    │
        │            ┆                     ┆ false]          ┆                         │
        │ 2          ┆ [-100.0, 0.0]       ┆ [false, false]  ┆ [true, false]           │
        │ 3          ┆ [-1.5, -0.5, … 1.5] ┆ [true, false, … ┆ [false, true, … true]   │
        │            ┆                     ┆ true]           ┆                         │
        └────────────┴─────────────────────┴─────────────────┴─────────────────────────┘
        >>> out[2]['time'].item().to_list()
        [-1.5, -0.5, 0.5, 1.5]
        >>> out[2]['pred_3'].item().to_list()
        [True, False, False, True]
        >>> out[2]['pred_1'].item().to_list()
        [False, True, False, True]
    """

    return (
        df.explode("time", "dynamic_indices", "dynamic_values")
        .with_columns((pl.col("time") - pl.col("align_time")).alias("time"))
        .drop("align_time")
        .explode("dynamic_indices", "dynamic_values")
        .with_columns(
            **{
                f"pred_{idx}": (
                    pl.when(pl.col("dynamic_indices") == idx)
                    .then(eval_range(rng, pl.col("dynamic_values")))
                    .otherwise(False)
                )
                for idx, rng in measurement_predicates.items()
            }
        )
        .groupby(["subject_id", "time"])
        .agg(*[pl.col(f"pred_{idx}").any() for idx in measurement_predicates.keys()])
        .sort(by=["subject_id", "time"])
        .groupby("subject_id", maintain_order=True)
        .agg(pl.all())
    )


def get_MCF_coordinates(
    control_df: pl.DataFrame,
    sample_dfs: list[pl.DataFrame],
    measurement_predicates: dict[int, bool | RANGE_T | list[RANGE_T]],
    n_timestamps: int | None = None,
) -> tuple[list[int], list[float], list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns aligned MCF coordinates per subject comparing the control and sample dataframes.

    Args:
        control_df: A dataframe in the "deep-learning friendly format" containing the control data for
            comparison. Must have columns ``subject_id``, ``time``, ``dynamic_indices``, and
            ``dynamic_values``.
        sample_dfs: A list of dataframes in the "deep-learning friendly format" containing the comparison
            population. Must have the same columns as the control_df, plus additional column
            ``control_align_idx``, which states what event index within the control dataframe is the temporal
            alignment point. Each entry of the list is interpreted to be an independent sample for comparison,
            and list order is presumed to be meaningless.
        measurement_predicates: A dictionary from dynamic measurement index to either the boolean True, in
            which case the presence of the measurement is used alone, or a range dictating
            bounds for the measurement's value to satisfy the predicate. The range is in the format
            ``(LOWER_BOUND, UPPER_BOUND)``, where ``*_BOUND`` can be either `None` (in which case there is no
            bound on that side), a floating point value (in which case the bound is considered to be
            exclusive), or a tuple of a floating point value and a boolean value where the boolean value
            indicates an inclusive or exclusive bound.
        n_timestamps: Downsample (without replacement) the set of possible aligned timepoints to this number
            if specified.

    Returns:
        1. The subject IDs in order of the rows of the returned coordinates.
        2. The aligned MCF time-values (aligned so that 0 is the alignment point between control and sample
        dataframes per subject).
        3. The output index of dynamic measurement indices.
        4. A boolean numpy array indicating whether or not a given subject (row) in the control population has
        data at or after a timepoint (column)
        5. A boolean numpy array containing incidence markers for measurement predicates (dimension 0) by
        subject (dimension 1) and time (dimension 3).
        4. A boolean numpy array indicating whether or not a given subject (dimension 0) in the sample
        population has data at or after a timepoint (dimension 1) across all sample populations (dimension 2)
        6. A boolean np array containing incidence markers for measurement predicates (dimension 0) by subject
        (dimension 1) and time (dimension 2) across all sample populations (dimension 3).

    Examples:
        >>> control_df = pl.DataFrame({
        ...     'subject_id': [1, 2, 3],
        ...     'control_align_idx': [1, 1, 0],
        ...     'time': [
        ...         [0., 10, 20],
        ...         [0., 100],
        ...         [0., 1, 2, 3],
        ...     ],
        ...     'dynamic_indices': [
        ...         [[1, 2], [3, 3, 2], [4]],
        ...         [[1], [3]],
        ...         [[2, 3], [1], [8], [3, 1, 1]],
        ...     ],
        ...     'dynamic_values': [
        ...         [[None, 0], [-1, 4, 0.2], [None]],
        ...         [[None], [3]],
        ...         [[-0.1, 10], [None], [None], [6, None, None]],
        ...     ],
        ... })
        >>> sample_df_1 = pl.DataFrame({
        ...     'subject_id': [2, 1, 3],
        ...     'time': [
        ...         [200, 300, 400],
        ...         [18, 24, 33],
        ...         [2.1, 3, 4.1],
        ...     ],
        ...     'dynamic_indices': [
        ...         [[1], [3], [1, 2]],
        ...         [[3], [2], [1]],
        ...         [[2, 3], [], [3, 3]],
        ...     ],
        ...     'dynamic_values': [
        ...         [[None], [3.1], [None, 0.03]],
        ...         [[0], [0.21], [None]],
        ...         [[-0.1, 10], [], [6, -1]],
        ...     ],
        ... })
        >>> sample_df_2 = pl.DataFrame({
        ...     'subject_id': [3, 1, 2],
        ...     'time': [
        ...         [5.1, 6, 7.1],
        ...         [11, 14, 23],
        ...         [110, 202, 250],
        ...     ],
        ...     'dynamic_indices': [
        ...         [[], [1, 2], [1]],
        ...         [[1, 2], [1], [1]],
        ...         [[1], [3], [3, 3]],
        ...     ],
        ...     'dynamic_values': [
        ...         [[], [None, 0.1], [None]],
        ...         [[None, -0.04], [None], [None]],
        ...         [[None], [13.1], [0.5, 0.3]],
        ...     ],
        ... })
        >>> measurement_predicates = {
        ...     3: (3.5, None),
        ...     1: True,
        ... }
        >>> out = get_MCF_coordinates(control_df, [sample_df_1, sample_df_2], measurement_predicates)
        >>> subject_ids, Ts, dynamic_indices, control_censor_mask, control_MCF, sample_mask, sample_MCF = out
        >>> subject_ids
        [1, 2, 3]
        >>> len(Ts)
        20
        >>> Ts[:10]
        [-100.0, -10.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.1, 6.0, 7.1]
        >>> Ts[10:]
        [8.0, 10.0, 13.0, 14.0, 23.0, 100.0, 102.0, 150.0, 200.0, 300.0]
        >>> dynamic_indices
        [3, 1]
        >>> control_censor_mask.shape
        (1, 3, 21)
        >>> control_MCF.shape
        (1, 3, 21, 2)
        >>> sample_mask.shape
        (2, 3, 21)
        >>> sample_MCF.shape
        (2, 3, 21, 2)
    """

    align_time_expr = pl.col("time").list.get(pl.col("control_align_idx")).alias("align_time")

    with_align_time = control_df.with_columns(align_time_expr)
    aligned_sample_dfs = []
    for df in sample_dfs:
        aligned_sample_dfs.append(
            align_time_and_eval_predicates(
                df.join(with_align_time.select("subject_id", "align_time"), on=["subject_id"], how="inner"),
                measurement_predicates,
            )
        )

    control_df = align_time_and_eval_predicates(with_align_time, measurement_predicates)

    subject_ids = control_df["subject_id"].to_list()

    aligned_timestamps = get_aligned_timestamps(
        control_df["time"], *[df["time"] for df in aligned_sample_dfs], n_timestamps=n_timestamps
    )

    dynamic_indices = list(measurement_predicates.keys())

    MCF_cols = [f"pred_{i}" for i in dynamic_indices]
    control_censor_mask, control_MCF = get_MCF(aligned_timestamps, MCF_cols, control_df)
    sample_censor_mask, sample_MCF = get_MCF(aligned_timestamps, MCF_cols, *aligned_sample_dfs)

    return (
        subject_ids,
        aligned_timestamps,
        dynamic_indices,
        control_censor_mask,
        control_MCF,
        sample_censor_mask,
        sample_MCF,
    )

from collections.abc import Callable
from typing import Dict, Union

import polars as pl

from ...utils import PROPORTION
from .preprocessor import Preprocessor


# This is a default that has proven to work reasonably well in practice, and is stored as a top level function
# so the module can be easily pickled.
def _default_std_delta_thresh(N: int) -> float:
    return 10 * (1 / N**0.6)


class VarianceImpactOutlierDetector(Preprocessor):
    """This outlier detector removes extremal elements that have an outsized impact on the datasets
    standard deviation."""

    @classmethod
    def params_schema(cls) -> dict[str, pl.DataType]:
        return {"thresh_large_": pl.Float64, "thresh_small_": pl.Float64}

    def __init__(
        self,
        subsample_frac: PROPORTION = 0.1,
        max_prob_of_exclusion: PROPORTION = 0.05,
        max_std_delta_thresh: Callable[[int], float] | float = _default_std_delta_thresh,
    ):
        assert (0 < subsample_frac) and (subsample_frac < 1)
        assert (0 < max_prob_of_exclusion) and (max_prob_of_exclusion < 1)

        if type(max_std_delta_thresh) is float:
            assert 0 < max_std_delta_thresh < 1
        else:
            assert callable(max_std_delta_thresh)

        self.subsample_frac = subsample_frac
        self.max_prob_of_exclusion = max_prob_of_exclusion
        self.max_std_delta_thresh = max_std_delta_thresh

    def _max_std_delta_thresh(self, N: int) -> float:
        """Returns either the passed threshold or evaluates the threshold on the dataset size."""
        if type(self.max_std_delta_thresh) is float:
            return self.max_std_delta_thresh
        else:
            return pl.max([pl.lit(0), pl.min([pl.lit(1), self.max_std_delta_thresh(N)])])

    def _max_L(self, N: int | pl.Expr) -> pl.Expr:
        """Returns the maximum size $L$ of a specific subset of elements from a dataset of total
        size $N$ such that the chance that none of those $L$ elements would appear in a random, iid
        subsample of the dataset of fractional size `self.subsample_frac` (denoted by r) would not
        exceed `self.max_prob_of_exclusion`.

        Let p be the probability that a randomly chosen subset of size $S$ with replacement from a collection
        of $N$ options will not contain any elements from a set of size $L$ within $N$. We can see that for
        this to be true, then each random choice of the $S$ selections must choose one of the elements in $N$
        not in the subset of size $L$, which happens with probability $(1 - L/N)$. Thus,

        p = (1 - (L/N))^(S)
        => 1 - L/N = p^(1/S)
        => L = N - N p^(1/S)

        We wish to find that the maximum number of elements $L$ such that it is possible with at least the
        chance $self.max_prob_of_exclusion$ that the elements in the size-$L$ subset would be excluded from a
        validation set of size $floor(N * self.subsample_frac)$. This implies the inequality

        L <= N - N * (self.max_prob_of_exclusion) ** (1/floor(N * self.subsample_frac))

        As $L$ must be an integer, and L cannot be greater than
        (N - N * (self.max_prob_of_exclusion) ** (1/floor(N * self.subsample_frac))),
        then we can see that
        L <= (N - N * (self.max_prob_of_exclusion) ** (1/floor(N * self.subsample_frac))).floor()
        """

        return pl.min(
            [
                N - 1,
                (
                    N - N * self.max_prob_of_exclusion ** (1 / (self.subsample_frac * N).floor())
                ).floor(),
            ]
        )

    def _max_deviation_factor(self, N: int | pl.Expr, delta: float | pl.Expr) -> pl.Expr | float:
        """
        curr_sum_X, curr_sum_X2
        delta = self._max_std_delta_thresh(N+1)

        1 - np.sqrt(curr_var/new_var) = delta
        curr_var / (1 - delta)**2 = new_var

        Let mx be the new maximum added to reach the maximum allowed deviation factor delta.

        curr_var = curr_sum_X2/N - (curr_sum_X/N)**2
        new_var = (curr_sum_X2 + mx**2)/(N+1) - ((curr_sum_X + mx)/(N+1))**2

        Let us imagine centering all our data by the curr_mean = curr_sum_X/N. Then, we can rewrite the above
        equations as:

        sum_X = 0
        sum_X2 = (x1 - curr_mean)**2 + (x2 - curr_mean)**2 + ... + (xN - curr_mean)**2
               = (x1**2 - 2*x1*curr_mean + curr_mean**2) + (x2**2 - 2*x2*curr_mean + curr_mean**2)
                 + ... + (xN**2 - 2*xN*curr_mean + curr_mean**2)
               = (x1**2 + x2**2 + ... + xN**2) - 2*curr_mean*(x1 + x2 + ... + xN) + N*curr_mean**2
               = curr_sum_X2 - 2*curr_mean*curr_sum_X + N*curr_mean**2
               = curr_sum_X2 - 2*N*curr_mean**2 + N*curr_mean**2
               = curr_sum_X2- N*curr_mean**2

        var = sum_X2/N = curr_sum_X2/N - curr_mean**2 = curr_var

        new_var = (sum_X2 + (mx-curr_mean)**2)/(N+1) - ((mx-curr_mean)/(N+1))**2

        Letting m = mx - curr_mean, we can rewrite the above as:
        new_var = (sum_X2 + m**2)/(N+1) - (m/(N+1))**2
                = m**2 * ((1/(N+1)) - (1/(N+1)**2)) + sum_X2/(N+1)
                = m**2 * (1/(N+1)) * (1 - (1/(N+1))) + (N/(N+1)) * curr_var
                = m**2 * (1/(N+1)) * (N/(N+1)) + (N/(N+1)) * curr_var

        Thus, we can then solve as follows:

        ==> m**2 * (1/(N+1)) * (N/(N+1)) + (N/(N+1)) * curr_var = curr_var / (1 - delta)**2
        ==> m**2 * (1/(N+1)) * (N/(N+1)) + curr_var * (N/(N+1)) - curr_var / (1 - delta)**2 = 0
        ==> m**2 * (1/(N+1)) * (N/(N+1)) + curr_var * (N/(N+1) - 1 / (1 - delta)**2) = 0
        ==> m**2 * (1/(N+1)) + curr_var * (1 - (N+1) / (N*(1 - delta)**2)) = 0
        ==> m**2 * (1/(N+1)) - curr_var * ((N+1) / (N*(1 - delta)**2) - 1) = 0
        ==> m**2 - curr_var * (((N+1)/(1 - delta))**2 * (1/N) - (N+1)) = 0
        ==> m**2 = curr_var * (((N+1)/(1 - delta))**2 * (1/N) - (N+1))
        ==> (m/curr_std)**2 = ((N+1)/(1 - delta))**2 * (1/N) - (N+1)
        ==> m/curr_std = sqrt(((N+1)/(1 - delta))**2 * (1/N) - (N+1))

        For this to be valid, then
        ((N+1)/(1 - delta))**2 * (1/N) - (N+1) >= 0
        ==> ((N+1)/(1 - delta))**2 * (1/N) >= N+1
        ==> (N+1)/(N*(1 - delta)**2) >= 1
        ==> N+1 >= N*(1 - delta)**2
        ==> (1-delta)**2 <= (N+1)/N
        ==> abs(1-delta) <= sqrt((N+1)/N)

        As we assume that new_var >= curr_var, then we must have that 0 <= delta < 1. Therefore, we can
        further constrain the above as follows:

        ==> 1 - delta <= sqrt((N+1)/N)
        ==> delta >= 1 - sqrt((N+1)/N)

        When valid, then m/curr_std = sqrt(((N+1)/(1 - delta))**2 * (1/N) - (N+1)), which means that
        mx = curr_mean + sqrt(((N+1)/(1 - delta))**2 * (1/N) - (N+1)) * curr_std
        """

        return (((N + 1) / (1 - delta)) ** 2 * (1 / N) - (N + 1)) ** 0.5

    @staticmethod
    def _sorted_deviations_and_cnts_expr(raw_column: pl.Expr) -> pl.Expr:
        """Returns a struct containing three fields:

        * `deviations`, containing the sorted absolute deviations from the mean.
        * `count_pos`, containing the number of times each deviation occurs in the positive direction.
        * `count_neg`, containing the number of times each deviation occurs in the negative direction.
        """

        column = (raw_column - raw_column.mean()).alias("val")
        # Now it has mean zero.

        val_counts = column.value_counts(sort=True)
        # `val_counts` contains structs with two fields:
        #   * `val`, containing the unique values within df[col].
        #   * `count` containing the number of times they occur.

        val = val_counts.struct.field("val")
        count = val_counts.struct.field("counts")

        is_gt_mean = val > 0
        is_lt_mean = ~is_gt_mean

        dev = val.abs()

        dev_gt_mean = dev.filter(is_gt_mean)
        count_gt_mean = count.filter(is_gt_mean)

        dev_lt_mean = dev.filter(is_lt_mean)
        count_lt_mean = count.filter(is_lt_mean)

        dev_gt_in_lt = dev_gt_mean.is_in(dev_lt_mean)
        dev_lt_in_gt = dev_lt_mean.is_in(dev_gt_mean)

        dev_only_gt_mean = dev_gt_mean.filter(~dev_gt_in_lt)
        count_only_gt_mean = count_gt_mean.filter(~dev_gt_in_lt)

        dev_only_lt_mean = dev_lt_mean.filter(~dev_lt_in_gt)
        count_only_lt_mean = count_lt_mean.filter(~dev_lt_in_gt)

        dev_both = dev_gt_mean.filter(dev_gt_in_lt)
        count_pos_both = count_gt_mean.filter(dev_gt_in_lt)
        count_neg_both = count_lt_mean.filter(dev_lt_in_gt).reverse()

        deviations = pl.concat([dev_both, dev_only_gt_mean, dev_only_lt_mean])
        count_pos = pl.concat([count_pos_both, count_only_gt_mean, 0 * count_only_lt_mean])
        count_neg = pl.concat([count_neg_both, 0 * count_only_gt_mean, count_only_lt_mean])

        return pl.struct(deviations=deviations, count_pos=count_pos, count_neg=count_neg).sort_by(
            deviations
        )

    def fit_from_polars(self, column: pl.Expr) -> pl.Expr:
        mean = column.mean()

        deviations = self._sorted_deviations_and_cnts_expr(column)
        # `deviation_with_count` contains structs with three fields:
        #   * `deviations`, containing the deviation of each unique value from the mean.
        #   * `count_pos` containing the number of times this deviation occurs > the mean.
        #   * `count_neg` containing the number of times this deviation occurs <= the mean.

        count = deviations.struct.field("count_pos") + deviations.struct.field("count_neg")
        val_times_count = deviations.struct.field("deviations") * (
            deviations.struct.field("count_pos").cast(pl.Int64)
            - deviations.struct.field("count_neg").cast(pl.Int64)
        )
        val_squared_times_count = deviations.struct.field("deviations") ** 2 * count

        # *_LE_val means including all values less than or equal to the extremal level of `val`.

        count_LE_val = count.cumsum()
        sum_LE_val = val_times_count.cumsum()
        mean_LE_val = sum_LE_val / count_LE_val

        sum_squared_LE_val = val_squared_times_count.cumsum()
        std_LE_val = (sum_squared_LE_val / count_LE_val - mean_LE_val**2) ** 0.5

        delta_std_after_removing_GT_val = 1 - std_LE_val / std_LE_val.shift(-1)
        max_std_delta_thresh_if_inc_val = self._max_std_delta_thresh(count_LE_val)

        max_removable_as_of_val = self._max_L(count_LE_val)
        valid_to_remove_GE_val = (
            (count <= max_removable_as_of_val).cumprod(reverse=True).cast(pl.Boolean)
        )

        can_remove_GT_val = valid_to_remove_GE_val.shift(-1).fill_null(True) & (
            delta_std_after_removing_GT_val > max_std_delta_thresh_if_inc_val
        ).fill_null(True)

        allowed_deviation_if_removed_GT_val = self._max_deviation_factor(
            count_LE_val, max_std_delta_thresh_if_inc_val
        )
        mean_if_removed_GT_val = mean_LE_val
        std_if_removed_GT_val = std_LE_val

        dev_thresh = allowed_deviation_if_removed_GT_val.filter(can_remove_GT_val).first()
        mean_at_thresh = mean_if_removed_GT_val.filter(can_remove_GT_val).first()
        std_at_thresh = std_if_removed_GT_val.filter(can_remove_GT_val).first()

        return pl.struct(
            [
                (mean_at_thresh + mean - dev_thresh * std_at_thresh).alias("thresh_small_"),
                (mean_at_thresh + mean + dev_thresh * std_at_thresh).alias("thresh_large_"),
            ]
        )

    @classmethod
    def predict_from_polars(cls, column: pl.Expr, model_column: pl.Expr) -> pl.Expr:
        """This provides a polars expression capable of producing the predictions of a fitted
        VarianceImpactOutlierDetector instance.

        It can be used either on a raw column or within a groupby expression, and will output a
        polars boolean column where True indicates an outlier.
        """

        return (
            (column > model_column.struct.field("thresh_large_"))
            | (column < model_column.struct.field("thresh_small_"))
        ).alias("is_outlier")

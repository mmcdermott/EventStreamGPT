from __future__ import annotations

import dataclasses, enum, functools, numpy as np

from typing import Callable, Union
from ..utils import StrEnum, PROPORTION

@dataclasses.dataclass(frozen=True)
class DataStats():
    """
    A simple container to store running data statistics used in the outlier detector. It tracks:
        * `N`, the number of observations
        * `sum_X`, the sum of the observations
        * `sum_X2`, the sum of the square of the observations.

    It has simple acessors for return the mean, variance, and standard deviation. These objects are frozen and
    can't be modified.
    """

    N: int
    sum_X: float
    sum_X2: float

    @functools.cached_property
    def mean(self): return self.sum_X/self.N
    @functools.cached_property
    def var(self): return self.sum_X2/self.N - self.mean**2
    @functools.cached_property
    def std(self): return self.var**0.5

    def remove(self, val_to_remove: float, copies_to_remove: int = 1) -> "DataStats":
        """Returns a new stats object after removing `copies_to_remove` copies of `val_to_remove`."""
        return DataStats(
            N = self.N - copies_to_remove,
            sum_X = self.sum_X - val_to_remove * copies_to_remove,
            sum_X2 = self.sum_X2 - (val_to_remove**2) * copies_to_remove,
        )

    @classmethod
    def from_array(cls, X: np.ndarray) -> "DataStats":
        """Returns a stats object describing the array `X`"""
        return cls(N=len(X), sum_X=X.sum(), sum_X2=(X**2).sum())

class ExtremeValSide(StrEnum):
    """A simple enum to track which side is being examined; the minimum or maximum."""
    MIN = enum.auto()
    MAX = enum.auto()

# This is a default that has proven to work reasonably well in practice, and is stored as a top level function
# so the module can be easily pickled.
def _default_std_delta_thresh(N: int) -> float: return 10*(1/N**0.6)

class VarianceImpactOutlierDetector():
    """
    This outlier detector removes extremal elements that have an outsized impact on the datasets standard
    deviation.
    """

    @staticmethod
    def validate_and_squeeze(vals: np.ndarray) -> np.ndarray:
        """Checks that `vals` is in the expected sklearn shape but is actually 1D, then squeezes it down."""
        assert len(vals.shape) == 2
        assert len(vals.squeeze(-1).shape) == 1
        return vals.squeeze(-1)

    def __init__(
        self,
        subsample_frac: PROPORTION = 0.1,
        max_prob_of_exclusion: PROPORTION = 0.05,
        max_std_delta_thresh: Union[Callable[int, float], float] = _default_std_delta_thresh,
        thresh_large_: Optional[float] = None,
        thresh_small_: Optional[float] = None,
    ):
        assert (0 < subsample_frac) and (subsample_frac < 1)
        assert (0 < max_prob_of_exclusion) and (max_prob_of_exclusion < 1)

        self.subsample_frac = subsample_frac
        self.max_prob_of_exclusion = max_prob_of_exclusion
        self.max_std_delta_thresh = max_std_delta_thresh

        if thresh_large_ is not None: self.thresh_large_ = thresh_large_
        if thresh_small_ is not None: self.thresh_small_ = thresh_small_

    def __str__(self) -> str:
        """Returns a nice str representation for use in displays on jupyter notebooks."""
        if hasattr(self, 'thresh_large_'):
            return f"{self.__class__.__name__}({self.thresh_small_:.1e} < inlier < {self.thresh_large_:.1e})"
        else: return self.__repr__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in vars(self).items())})"

    def _max_std_delta_thresh(self, N: int) -> float:
        """Returns either the passed threshold or evaluates the threshold on the dataset size."""
        if type(self.max_std_delta_thresh) in (float, int): return self.max_std_delta_thresh
        else: return self.max_std_delta_thresh(N)

    def _max_L(self, N: int) -> int:
        """
        Returns the maximum size $L$ of a specific subset of elements that could be excluded from a dataset of
        total size $N$ such that the chance that none of those $L$ elements would appear in a random, iid
        subsample of the dataset of fractional size `self.subsample_frac` would not exceed
        `self.max_prob_of_exclusion`.

        Let p be the chance that the validation set won't include a particular set of L elements. Then,

        p = (1 - (L/N))^(rN)
        => log((1 - (L/N))^(rN)) = log(p)
        => log(p)/rN = log(1 - L/N)
        => exp(log(p)/rN) = 1-L/N
        => 1 - p^(1/rN) = L/N
        => L = N - N p^(1/rN)
        """

        # So, if we want the chance that our validation set doesn't include these L elements to be less than
        # prob_thresh, then L must be >= floor(N - N p^(1/rN))
        return min(N-1, int(np.floor(N * (1 - self.max_prob_of_exclusion**(1/(self.subsample_frac*N))))))

    def _max_deviation_factor(self, N: int) -> float:
        """
        curr_sum_X, curr_sum_X2
        delta = self._max_std_delta_thresh(N+1)

        1 - np.sqrt(curr_var/new_var) = delta
        curr_var / (1 - delta)**2 = new_var


        If data has sum zero, then

        new_var = (curr_sum_X2 + m**2)/(N+1) - (m/(N+1))**2
        curr_var = curr_sum_X2/N

        => m**2 * ((1/(N+1))  - (1 / (N+1)**2)) + N/(N+1) * curr_var = curr_var / (1 - delta)**2
        => m**2 * ((1/(N+1))  - (1 / (N+1)**2)) + curr_var * (N/(N+1) - 1 / (1 - delta)**2) = 0
        => m**2 * (1  - 1 / (N+1)) + curr_var * (N - (N+1) / (1 - delta)**2) = 0
        => m**2 * (N / (N+1)) + curr_var * (N - (N+1) / (1 - delta)**2) = 0
        => (m/curr_std)**2 * (N / (N+1)) + (N - (N+1) / (1 - delta)**2) = 0
        => (m/curr_std)**2 = - (N - (N+1) / (1 - delta)**2) * (N+1) / N
        => (m/curr_std)**2 = (N+1)**2 / N*(1 - delta)**2 - (N+1)
        => m/curr_std = (N+1) * np.sqrt(1/(N*(1 - delta)**2) - 1/(N+1))

        For this to be valid, then
        1/(N*(1 - delta)**2) - 1/(N+1) > 0
        => 1/(N*(1 - delta)**2) > 1/(N+1)
        => N*(1 - delta)**2 < N+1
        => (1 - delta)**2 < (N+1)/N
        => 1 - delta < np.sqrt((N+1)/N) AND -(1 - delta) < np.sqrt((N+1)/N)
        => delta > 1 - np.sqrt((N+1)/N) AND delta < 1 + np.sqrt((N+1)/N)
        => -np.sqrt((N+1)/N) < delta - 1 < np.sqrt((N+1)/N)
        """

        delta = self._max_std_delta_thresh(N+1)

        if abs(delta - 1) >= np.sqrt((N+1)/N): return float('inf')
        return (N+1) * np.sqrt(1/(N * (1 - delta)**2) - 1/(N+1))

    def get_starting_bounds(self, X):
        """Determines the input bounds for the dataset `X`."""
        X = self.validate_and_squeeze(X)

        curr_stats = DataStats.from_array(X)

        max_dev = self._max_deviation_factor(curr_stats.N)

        self.thresh_large_ = curr_stats.mean + curr_stats.std * max_dev
        self.thresh_small_ = curr_stats.mean - curr_stats.std * max_dev

        return np.sort(X), curr_stats

    @staticmethod
    def _std_delta_for_side(
        X: np.ndarray, side: ExtremeValSide, max_L: int, curr_stats: DataStats
    ):
        """Determines the standard deviation for the removing the most extreme value on the specified size"""
        found_endpoint = False
        is_min = side == ExtremeValSide.MIN

        # If there are too many values that sit in this extreme, we don't want to remove them.
        for L in range(1, max_L):
            if (is_min  and X[L] != X[L-1]) or (X[-L] != X[-L-1]):
                found_endpoint = True
                break

        if not found_endpoint: return 0, 0, curr_stats

        extreme_val = X[L-1] if is_min else X[-L]
        new_stats = curr_stats.remove(extreme_val, copies_to_remove=L)

        std_delta = abs(curr_stats.std - new_stats.std)/curr_stats.std

        return std_delta, extreme_val, new_stats

    def fit(self, X):
        """Finds the thresholds that minimize standard deviation impacts on the training set."""
        X, curr_stats = self.get_starting_bounds(X)

        while curr_stats.N > 1:
            any_removed = False
            max_L = self._max_L(curr_stats.N)

            min_side = self._std_delta_for_side(X, ExtremeValSide.MIN, max_L, curr_stats)
            max_side = self._std_delta_for_side(X, ExtremeValSide.MAX, max_L, curr_stats)

            min_more_extreme = min_side[0] > max_side[0]

            std_delta, extreme_val, new_stats = (min_side if min_more_extreme else max_side)

            if std_delta > self._max_std_delta_thresh(curr_stats.N):
                if min_more_extreme:
                    self.thresh_small_ = extreme_val
                    X = X[-new_stats.N:]
                else:
                    self.thresh_large_ = extreme_val
                    X = X[:new_stats.N]
                curr_stats = new_stats
            else: break

    def predict(self, X):
        """Identifies inliers as points within the bounds and outliers otherwise."""
        X = self.validate_and_squeeze(X)
        inliers = np.ones_like(X)
        inliers[(X <= self.thresh_small_) | (X >= self.thresh_large_)] = -1
        return inliers

import sys

sys.path.append("../..")

import unittest

import numpy as np
import polars as pl

from EventStream.data.preprocessing.stddev_cutoff import StddevCutoffOutlierDetector

from ...utils import MLTypeEqualityCheckableMixin


class TestStddevCutoffOutlierDetector(MLTypeEqualityCheckableMixin, unittest.TestCase):
    """Tests the StddevCutoffOutlierDetector class."""

    def test_gets_correct_thresh(self):
        M = StddevCutoffOutlierDetector(2.1)

        X = np.array([-1, 0, 1, -1, 1, -1, 1, 10])
        mean = X.mean()
        std = X.std(ddof=1)

        want_inliers = np.array([-1, 0, 1, -1, 1, -1, 1])

        want = {
            "thresh_small_": mean - 2.1 * std,
            "thresh_large_": mean + 2.1 * std,
        }

        X_pl = pl.from_numpy(X)
        col = pl.col("column_0")

        expr = M.fit_from_polars(col)

        want = {k: round(v, 4) for k, v in want.items()}
        got = {k: round(v, 4) for k, v in X_pl.select(expr).item().items()}
        self.assertEqual(want, got)

        with_params = X_pl.with_columns(expr.alias("outlier_params"))

        outliers_expr = M.predict_from_polars(col, pl.col("outlier_params"))
        got_inliers = X[~with_params.select(outliers_expr)[:, 0].to_numpy()]

        self.assertEqual(got_inliers, want_inliers)

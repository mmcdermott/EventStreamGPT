import sys

sys.path.append("../..")

import unittest

import numpy as np
import polars as pl

from EventStream.data.preprocessing.standard_scaler import StandardScaler

from ...utils import MLTypeEqualityCheckableMixin


class TestStandardScaler(MLTypeEqualityCheckableMixin, unittest.TestCase):
    """Tests the StddevCutoffOutlierDetector class."""

    def test_e2e(self):
        M = StandardScaler()

        X = np.array([-1, 0, 1, -1, 1, 10])

        mean = X.mean()
        std = X.std(ddof=1)

        want_transformed = (X - mean) / std
        want_params = {"mean_": mean, "std_": std}

        X_pl = pl.from_numpy(X)
        col = pl.col("column_0")

        expr = M.fit_from_polars(col)

        want = {k: round(v, 4) for k, v in want_params.items()}
        got = {k: round(v, 4) for k, v in X_pl.select(expr).item().items()}
        self.assertEqual(want, got)

        with_params = X_pl.with_columns(expr.alias("params"))

        transformed_expr = M.predict_from_polars(col, pl.col("params"))
        got_transformed = with_params.select(transformed_expr)[:, 0].to_numpy().round(4)
        want_transformed = want_transformed.round(4)

        self.assertEqual(want_transformed, got_transformed)

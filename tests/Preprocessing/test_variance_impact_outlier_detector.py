import sys
sys.path.append('../..')

import math, unittest, numpy as np, polars as pl

from ..mixins import MLTypeEqualityCheckableMixin

from EventStream.Preprocessing.variance_impact_outlier_detector import VarianceImpactOutlierDetector

class TestVarianceImpactOutlierDetector(MLTypeEqualityCheckableMixin, unittest.TestCase):
    """Tests the VarianceImpactOutlierDetector class."""

    def test_max_L(self):
        """Tests the _max_L method."""

        n_samples = 1000
        for max_prob_of_exclusion in [0.01, 0.13]:
            for subsample_frac in [0.053, 0.21]:
                M = VarianceImpactOutlierDetector(
                    subsample_frac=subsample_frac, max_prob_of_exclusion=max_prob_of_exclusion
                )

                for N in (101, 10013):
                    np.random.seed(N)
                    n_subsample = int(math.floor(N * subsample_frac))
                    got_max_L = pl.DataFrame({}).select(M._max_L(pl.lit(N))).item()

                    with self.subTest(
                        N=N, subsample_frac=subsample_frac, max_prob_of_exclusion=max_prob_of_exclusion
                    ):
                        self.assertTrue(0 <= got_max_L <= N-1)

                        got_max_L_selected = 0
                        got_max_L_plus_one_selected = 0
                        for samp in range(n_samples):
                            rand_sel = np.random.choice(N, n_subsample, replace=True)

                            if rand_sel.min() < got_max_L: got_max_L_selected += 1
                            if rand_sel.min() < got_max_L + 1: got_max_L_plus_one_selected += 1

                        got_perc_max_L_excluded = 1 - (got_max_L_selected / n_samples)
                        got_perc_max_L_plus_one_excluded = 1 - (got_max_L_plus_one_selected / n_samples)

                        # We add a tolerance of 0.5% here to add robustness.
                        self.assertGreaterEqual(got_perc_max_L_excluded, max_prob_of_exclusion-0.005)
                        self.assertLessEqual(got_perc_max_L_plus_one_excluded, max_prob_of_exclusion+0.005)

    def test_max_deviation_factor(self):
        for N in (103, 10001):
            for delta in (0.01, 0.13):
                M = VarianceImpactOutlierDetector(max_std_delta_thresh=delta)

                got_max_deviation_factor = M._max_deviation_factor(N, delta)
                got_max_deviation_factor_pl = pl.DataFrame({}).select(
                    M._max_deviation_factor(pl.lit(N), pl.lit(delta))
                ).item()

                with self.subTest(N=N, delta=delta):
                    self.assertEqual(got_max_deviation_factor, got_max_deviation_factor_pl)

                    X = np.random.normal(size=(N,))
                    X = (X - X.mean()) / X.std()

                    X_with_delta = np.array(list(X) + [got_max_deviation_factor])

                    orig_std = 1.0
                    std_with_delta = X_with_delta.std()

                    got_delta = np.round(1 - orig_std / std_with_delta, 5)

                    self.assertEqual(delta, got_delta)

    @unittest.skip("Currently broken due to a bug in Polars.")
    def test_sorted_deviations_and_cnts_expr(self):
        want_deviations = [0, 1, 2, 3, 4]
        want_count_pos = [0, 4, 2, 1, 0]
        want_count_neg = [1, 1, 0, 2, 1]

        X = pl.DataFrame({
            'X': [-4, -3, -3, -1, 0, 1, 1, 1, 1, 2, 2, 3],
            'k': [0] * 12,
        })

        deviations_expr = VarianceImpactOutlierDetector._sorted_deviations_and_cnts_expr(pl.col('X'))

        got = X.groupby('k').agg(deviations_expr.alias('d'))

        got_deviations = got.select(pl.col('d').struct.field('deviations'))[:, 0].to_list()
        got_count_pos = got.select(pl.col('d').struct.field('count_pos'))[:, 0].to_list()
        got_count_neg = got.select(pl.col('d').struct.field('count_neg'))[:, 0].to_list()

        self.assertEqual(want_deviations, got_deviations)
        self.assertEqual(want_count_pos, got_count_pos)
        self.assertEqual(want_count_neg, got_count_neg)

    @unittest.skip("Currently broken due to a bug in Polars.")
    def test_gets_correct_thresh(self):
        M = VarianceImpactOutlierDetector(max_std_delta_thresh=0.1)
        deviation = 3.0

        want_inliers = np.array([
            -3, -3, -3, -3, -2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
        ])

        want = {
            'thresh_small_': -deviation * want_inliers.std(),
            'thresh_large_': deviation * want_inliers.std()
        }

        actual_delta_1 = M._max_deviation_factor(28, 0.1)*want_inliers.std() + 1
        actual_delta_2 = M._max_deviation_factor(30, 0.1)*want_inliers.std() + 100
        #print('actual delta 1', actual_delta_1)
        #print('actual delta 2', actual_delta_2)

        def max_L_mock(_, N): return 0*N + 3
        M._max_L = max_L_mock.__get__(M, VarianceImpactOutlierDetector)
        def max_deviation_factor_mock(_, N, delta): return (0*N + deviation)
        M._max_deviation_factor = max_deviation_factor_mock.__get__(M, VarianceImpactOutlierDetector)

        # Now, M will alwasy report that it is valid to remove up to 3 entries from the input and that the max
        # deviation from the mean is always 20.0.

        cases = [
            {
                'msg': 'No outliers',
                'X': np.array([
                    -3, -3, -3, -3, -2, -2, -2, -2, -1, -1, -1, -1,
                    0, 0, 0, 0,
                    1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                ]),
            }, {
                'msg': 'Outliers on both sides, max more extreme',
                'X': np.array([
                    -(actual_delta_1), -(actual_delta_1),
                    -3, -3, -3, -3, -2, -2, -2, -2, -1, -1, -1, -1,
                    0, 0, 0, 0,
                    1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                    (2*actual_delta_1/30 + actual_delta_2), (2*actual_delta_1/30 + actual_delta_2)
                ]),
            }, {
                'msg': 'Outliers on both sides, min more extreme',
                'X': np.array([
                    -(2*actual_delta_1/30 + actual_delta_2), -(2*actual_delta_1/30 + actual_delta_2),
                    -3, -3, -3, -3, -2, -2, -2, -2, -1, -1, -1, -1,
                    0, 0, 0, 0,
                    1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                    (actual_delta_1), (actual_delta_1),
                ]),
            }
        ]

        for case in cases:
            with self.subTest(case['msg']):
                N = len(case['X'])
                X_pl = pl.DataFrame({
                    'key': ['k1'] * N + ['k2'] * N,
                    'X': list(case['X']) + list(case['X'] + 100),
                })
                col = pl.col('X')

                expr = M.fit_from_polars(col)

                got_params = X_pl.groupby('key').agg(expr)
                got_k1 = got_params.filter(pl.col('key') == 'k1').item()
                got_k2 = got_params.filter(pl.col('key') == 'k1').item()

                got_k1 = {k: round(v, 4) for k, v in got_k1.items()}
                got_k2 = {k: round(v, 4) for k, v in got_k2.items()}

                self.assertEqual(want, got_k1)
                self.assertEqual({k: v+100 for k, v in want.items()}, got_k2)

                X_pl = X_pl.filter(pl.col('key') == 'k1')

                with_params = X_pl.with_columns(expr.alias('outlier_params'))

                outliers_expr = M.predict_from_polars(col, pl.col('outlier_params'))
                got_inliers = case['X'][~with_params.select(outliers_expr)[:, 0].to_numpy()]

                self.assertEqual(got_inliers, want_inliers)

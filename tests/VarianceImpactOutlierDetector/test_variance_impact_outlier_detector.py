import sys
sys.path.append('../..')

from ..mixins import MLTypeEqualityCheckableMixin
import unittest, numpy as np

from EventStream.VarianceImpactOutlierDetector.variance_impact_outlier_detector import (
    VarianceImpactOutlierDetector
)

class TestEventStreamDataset(MLTypeEqualityCheckableMixin, unittest.TestCase):
    def test_flags_no_normal_outliers(self):
        for N in (10, 100, 1000, 10000, 1000000):
            with self.subTest(f"Normal(0, 1).rvs({N}) should yield no outliers under default params"):
                np.random.seed(N)
                X = np.random.normal(size=(N, 1))

                M = VarianceImpactOutlierDetector()
                M.fit(X)

                self.assertFalse(np.isnan(M.thresh_small_))
                self.assertFalse(np.isnan(M.thresh_large_))

                inliers = M.predict(X)
                self.assertTrue((inliers == 1).all())

    def test_flags_no_normal_mixture_outliers(self):
        for N in (10, 100, 1000, 10000, 1000000):
            with self.subTest(f"Normal(0, 1).rvs({N}) should yield no outliers under default params"):
                np.random.seed(N)
                X = np.concatenate((
                    np.random.normal(loc=-1, size=(N//2, 1)),
                    np.random.normal(loc=1, size=(N//2, 1)),
                ))

                M = VarianceImpactOutlierDetector()
                M.fit(X)

                self.assertFalse(np.isnan(M.thresh_small_))
                self.assertFalse(np.isnan(M.thresh_large_))

                inliers = M.predict(X)
                self.assertTrue((inliers == 1).all())

    def test_flags_real_outliers(self):
        for N in (1000, 10000, 1000000):
            with self.subTest(f"Normal(0, 1).rvs({N}) should yield correct outliers under default params"):
                np.random.seed(N)
                X = np.random.normal(size=(N - 2,))
                X = np.concatenate((X, [1e2, -1e2])).reshape((-1, 1))

                M = VarianceImpactOutlierDetector()
                M.fit(X)

                self.assertFalse(np.isnan(M.thresh_small_))
                self.assertFalse(np.isnan(M.thresh_large_))

                inliers = M.predict(X)
                self.assertEqual(2, (inliers == -1).sum())

    def test_outlier_flagging_is_specific(self):
        for N in (100, 1000, 10000, 1000000):
            with self.subTest(f"Outliers should be specific to cutoff."):
                np.random.seed(N)
                X = np.random.normal(size=(N - 2,))
                X = (X - X.mean()) / X.std() # normalie to zero mean and unit variance empirically.

                std_delta = 0.1
                new_std = 1.1
                new_var = new_std**2

                new_pt = np.sqrt((N+1) * ((N+1)/N * (1/std_delta)**2 - 1))

                X = np.concatenate((X, [new_pt, -new_pt])).reshape((-1, 1))

                M = VarianceImpactOutlierDetector(max_std_delta_thresh = std_delta)
                M.fit(X)

                self.assertEqual(new_pt, M.thresh_large_)
                self.assertEqual(-new_pt, M.thresh_small_)

                inliers = M.predict(X)
                self.assertEqual(2, (inliers == -1).sum())

    def test_flags_no_lognormal_outliers(self):
        for N in (10, 100, 1000, 10000, 1000000):
            with self.subTest(f"LogNormal(0, 1).rvs({N}) should yield no outliers under default params"):
                np.random.seed(N)
                X = np.random.lognormal(size=(N, 1))

                M = VarianceImpactOutlierDetector()
                M.fit(X)

                self.assertFalse(np.isnan(M.thresh_small_))
                self.assertFalse(np.isnan(M.thresh_large_))

                inliers = M.predict(X)
                self.assertTrue((inliers == 1).all())


    def test_flags_no_exponential_outliers(self):
        for N in (10, 100, 1000, 10000, 1000000):
            with self.subTest(f"Exponential(0, 1).rvs({N}) should yield no outliers under default params"):
                np.random.seed(N)
                X = np.random.exponential(size=(N, 1))

                M = VarianceImpactOutlierDetector()
                M.fit(X)

                self.assertFalse(np.isnan(M.thresh_small_))
                self.assertFalse(np.isnan(M.thresh_large_))

                inliers = M.predict(X)
                self.assertTrue((inliers == 1).all())

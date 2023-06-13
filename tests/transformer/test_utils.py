import sys

sys.path.append("../..")

import unittest

import torch

from EventStream.transformer.utils import (
    expand_indexed_regression,
    idx_distribution,
    safe_masked_max,
    safe_weighted_avg,
    weighted_loss,
)

from ..utils import MLTypeEqualityCheckableMixin


class TestUtils(MLTypeEqualityCheckableMixin, unittest.TestCase):
    def test_expand_indexed_regression(self):
        """`expand_indexed_regression` should convert a sparse tensor of values & indices to a dense one."""
        X = torch.Tensor(
            [
                [1, 3.4],
                [2.0, 0.0],
            ]
        )
        idx = torch.LongTensor(
            [
                [0, 2],
                [3, 1],
            ]
        )

        got = expand_indexed_regression(X, idx, vocab_size=5)
        want = torch.Tensor(
            [
                [1, 0, 3.4, 0, 0],
                [0, 0, 0, 2, 0],
            ]
        )

        self.assertEqual(want, got)

    def test_safe_masked_max(self):
        X = torch.FloatTensor(
            [
                [1.3, 1.1, 2.0, 1.2],
                [1.0, -1.0, 2.0, -2.0],
                [5.0, -1.0, 3.2, -2.0],
            ]
        )
        X.requires_grad = True

        W = torch.BoolTensor(
            [
                [True, True, False, True],
                [True, False, True, False],
                [False, False, False, False],
            ]
        )

        want_max = torch.Tensor(
            [
                1.3,
                2.0,
                0.0,
            ]
        )

        want_X_grad = torch.Tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        got_max = safe_masked_max(X, W)
        torch.testing.assert_close(want_max, got_max)

        got_max.sum().backward()

        torch.testing.assert_close(want_X_grad, X.grad)

    def test_safe_weighted_avg(self):
        X = torch.Tensor(
            [
                [1.0, 1.0, 2.0, 1.0],
                [1.0, -1.0, 2.0, -2.0],
                [5.0, -1.0, 3.2, -2.0],
            ]
        )
        X.requires_grad = True

        W = torch.Tensor(
            [
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        want_avg = torch.Tensor(
            [
                1.0,
                1.75,
                0.0,
            ]
        )

        want_denom = torch.Tensor(
            [
                3.0,
                4.0,
                0.0,
            ]
        )

        want_X_grad = torch.Tensor(
            [
                [1 / 3, 1 / 3, 0.0, 1 / 3],
                [1 / 4, 0.0, 3 / 4, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        avg, denom = safe_weighted_avg(X, W)
        torch.testing.assert_close(avg, want_avg)
        torch.testing.assert_close(denom, want_denom)

        avg.sum().backward()

        torch.testing.assert_close(X.grad, want_X_grad)

    def test_safe_weighted_avg_errors_with_negative_weights(self):
        X = torch.Tensor(
            [
                [1.0, 1.0, 2.0, 1.0],
                [5.0, -1.0, 3.2, -2.0],
            ]
        )

        W = torch.Tensor(
            [
                [1.0, 1.0, 0.0, 1.0],
                [-1.0, 0.0, 1.0, 0.0],
            ]
        )

        with self.assertRaises(AssertionError):
            avg, denom = safe_weighted_avg(X, W)

    def test_weighted_loss(self):
        X = torch.Tensor(
            [
                [1.0, 1.0, 2.0, 1.0],
                [1.0, -1.0, 2.0, -2.0],
                [5.0, -1.0, 3.2, -2.0],
            ]
        )
        X.requires_grad = True

        W = torch.Tensor(
            [
                [1, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 0, 0, 0],
            ]
        )

        want_loss = torch.Tensor([1.25])[0]

        want_X_grad = torch.Tensor(
            [
                [1 / 6, 1 / 6, 0.0, 1 / 6],
                [1 / 4, 0.0, 1 / 4, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        loss = weighted_loss(X, W)
        torch.testing.assert_close(loss, want_loss)

        loss.backward()

        torch.testing.assert_close(X.grad, want_X_grad)

    def test_idx_distribution_normal(self):
        loc = torch.randn(size=(5, 4))
        scale = torch.randn(size=(5, 4)).abs()
        D = torch.distributions.Normal(loc=loc, scale=scale)

        cases = [
            {
                "msg": "Should work with a purely integer slice.",
                "slice": 2,
                "want_dist": torch.distributions.Normal(loc=loc[2], scale=scale[2], validate_args=True),
            },
            {
                "msg": "Should work with a tuple slice.",
                "slice": (3,),
                "want_dist": torch.distributions.Normal(loc=loc[3], scale=scale[3], validate_args=True),
            },
            {
                "msg": "Should work with ellipsis.",
                "slice": (
                    Ellipsis,
                    3,
                ),
                "want_dist": torch.distributions.Normal(loc=loc[:, 3], scale=scale[:, 3], validate_args=True),
            },
            {
                "msg": "Should work with colons.",
                "slice": (
                    slice(None),
                    3,
                ),
                "want_dist": torch.distributions.Normal(loc=loc[:, 3], scale=scale[:, 3], validate_args=True),
            },
            {
                "msg": "Should work with 2 slices.",
                "slice": (slice(1, 3, 1), slice(1, 2, 1)),
                "want_dist": torch.distributions.Normal(
                    loc=loc[1:3, 1:2], scale=scale[1:3, 1:2], validate_args=True
                ),
            },
        ]

        for case in cases:
            with self.subTest(case["msg"]):
                got_dist = idx_distribution(D, case["slice"])
                self.assertDistributionsEqual(case["want_dist"], got_dist)

    def test_idx_distribution_transformed_distribution(self):
        loc = torch.randn(size=(5, 4, 3))
        scale = torch.randn(size=(5, 4, 3)).abs()
        transforms = [torch.distributions.transforms.AffineTransform(loc=4, scale=5)]
        D = torch.distributions.TransformedDistribution(
            base_distribution=torch.distributions.Normal(loc, scale),
            transforms=transforms,
        )

        cases = [
            {
                "msg": "Should work with a complex slice.",
                "slice": (3, slice(2, 3, 1), slice(None, 2, 1)),
                "want_dist": torch.distributions.TransformedDistribution(
                    base_distribution=torch.distributions.Normal(
                        loc=loc[3, 2:3, :2], scale=scale[3, 2:3, :2], validate_args=True
                    ),
                    transforms=transforms,
                    validate_args=True,
                ),
            }
        ]

        for case in cases:
            with self.subTest(case["msg"]):
                got_dist = idx_distribution(D, case["slice"])
                self.assertDistributionsEqual(case["want_dist"], got_dist)

    def test_idx_distribution_mixture_same_family(self):
        loc = torch.randn(size=(5, 4, 3))
        scale = torch.randn(size=(5, 4, 3)).abs()
        logits = torch.randn(size=(5, 4, 3))
        D = torch.distributions.MixtureSameFamily(
            component_distribution=torch.distributions.Normal(loc=loc, scale=scale),
            mixture_distribution=torch.distributions.Categorical(logits=logits),
        )

        cases = [
            {
                "msg": "Should work with a complex slice.",
                "slice": (3, slice(2, 3, 1)),
                "want_dist": torch.distributions.MixtureSameFamily(
                    component_distribution=torch.distributions.Normal(
                        loc=loc[3, 2:3], scale=scale[3, 2:3], validate_args=True
                    ),
                    mixture_distribution=torch.distributions.Categorical(
                        logits=logits[3, 2:3],
                        validate_args=True,
                    ),
                    validate_args=True,
                ),
            }
        ]

        for case in cases:
            with self.subTest(case["msg"]):
                got_dist = idx_distribution(D, case["slice"])
                self.assertDistributionsEqual(case["want_dist"], got_dist)

    def test_idx_distribution_categorical(self):
        logits = torch.randn(size=(5, 4))
        D = torch.distributions.Categorical(logits=logits)

        cases = [
            {
                "msg": "Should work with a purely integer slice.",
                "slice": 2,
                "want_dist": torch.distributions.Categorical(logits=logits[2], validate_args=True),
            },
            {
                "msg": "Should work with a tuple slice.",
                "slice": (3,),
                "want_dist": torch.distributions.Categorical(logits=logits[3], validate_args=True),
            },
            {
                "msg": "Should work with ellipsis.",
                "slice": Ellipsis,
                "want_dist": torch.distributions.Categorical(logits=logits, validate_args=True),
            },
            {
                "msg": "Should work with colons.",
                "slice": (slice(None),),
                "want_dist": torch.distributions.Categorical(logits=logits, validate_args=True),
            },
            {
                "msg": "Should work with 2 slices.",
                "slice": slice(1, 3, 1),
                "want_dist": torch.distributions.Categorical(logits=logits[1:3], validate_args=True),
            },
        ]

        for case in cases:
            with self.subTest(case["msg"]):
                got_dist = idx_distribution(D, case["slice"])
                self.assertDistributionsEqual(case["want_dist"], got_dist)


if __name__ == "__main__":
    unittest.main()

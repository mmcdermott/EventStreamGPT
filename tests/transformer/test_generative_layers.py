import sys

sys.path.append("../..")

import math
import unittest

import torch

from EventStream.transformer.generative_layers import (
    ExponentialTTELayer,
    GaussianIndexedRegressionLayer,
    LogNormalMixtureTTELayer,
)

from ..utils import MLTypeEqualityCheckableMixin


class TestLogNormalMixture(MLTypeEqualityCheckableMixin, unittest.TestCase):
    def test_forward(self):
        L = LogNormalMixtureTTELayer(
            in_dim=6,
            num_components=2,
        )
        L.proj.weight = torch.nn.Parameter(
            torch.Tensor(
                [
                    [1, 0, 0, 0, 0, 0],  # Loc for Component 1
                    [0, 1, 0, 0, 0, 0],  # Log scale for Component 1
                    [0, 0, 1, 0, 0, 0],  # Log weight for Component 1
                    [0, 0, 0, 1, 0, 0],  # Loc for Component 2
                    [0, 0, 0, 0, 1, 0],  # Log scale for Component 2
                    [0, 0, 0, 0, 0, 1],  # Log weight for Component 2
                ]
            )
        )
        L.proj.bias = torch.nn.Parameter(
            torch.Tensor(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
        )

        X = torch.Tensor(
            [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    torch.finfo(torch.float32).min,
                ],  # A standard lognormal in the 1st component.
                [
                    0,
                    0,
                    torch.finfo(torch.float32).min,
                    2,
                    1,
                    0,
                ],  # A nonstandard lognormal in the 2nd component.
                [0, 0, 1, 2, 1, 1],  # An equal combo of the first two.
                [
                    0,
                    0,
                    math.log(2 / 3),
                    2,
                    1,
                    math.log(1 / 3),
                ],  # An unequal combo of the first two.
            ]
        )
        X.requires_grad = True

        # The means of this distribution should be as follows:
        mean_0 = math.exp(0 + math.exp(0) ** 2 / 2)
        mean_1 = math.exp(2 + math.exp(1) ** 2 / 2)
        want_mean = torch.Tensor(
            [
                mean_0,
                mean_1,
                0.5 * mean_0 + 0.5 * mean_1,
                2 / 3 * mean_0 + 1 / 3 * mean_1,
            ]
        )

        # Current implementation of the lognormal mixture doesn't permit measurement of variance.
        # var_0 = (math.exp(math.exp(0)**2) - 1)*math.exp(2*0 + math.exp(0)**2)
        # var_1 = (math.exp(math.exp(1)**2) - 1)*math.exp(2*2 + math.exp(1)**2)
        # want_variance = torch.Tensor([
        #     var_0,
        #     var_0,
        #     var_1,
        #     (0.5*var_0 + 0.5*var_1) + (0.5*mean_0**2 + 0.5*mean_1**2) + (0.5*mean_0 + 0.5*mean_1)**2,
        #     (2/3*var_0 + 1/3*var_1) + (2/3*mean_0**2 + 1/3*mean_1**2) + (2/3*mean_0 + 1/3*mean_1)**2,
        # ])

        out = L(X)
        self.assertEqual(out.mean, want_mean)
        # self.assertEqual(out.variance, want_variance)

        # To test NLL and grad flow, we can check two simple lognormals and the two mixtures, each with
        # different modes. We need to select things that should increase or decrease the expected
        # mean/std/weights per component in a cartesian product.
        # We'll select something that should increase the mean for the first,

        obs = torch.Tensor(
            [
                3,  # Guaranteed to be larger than the mode of the first lognormal.
                0.1,  # Guaranteed to be smaller than the mode of the second lognormal.
                # The mode of the first component.
                math.exp(0 - math.exp(0) ** 2),
                # The mode of the second component.
                math.exp(2 - math.exp(1) ** 2),
            ]
        )

        NLL = -out.log_prob(obs)
        NLL.sum().backward()

        # To maximize NLL, we should push for each observation:
        # The first component's loc and scale should both decrease, to push the component farther away the
        # observation and narrow it. The second component's loc and scale will have no contribution as that
        # component has zero weight. Both component's weights are saturated in this example at extreme values,
        # so they will have gradients of zero, despite the fact that increasing attribution to the opposite
        # component would increase NLL further.
        obs_1_grad_signs = [-1, -1, 0, 0, 0, 0]

        # The second component's loc should increase and it's scale should decrease, to push the component
        # farther away the observation. The first component's loc and scale will have no
        # contribution as that component has zero weight. Both component's weights are saturated in this
        # example at extreme values, so they will have gradients of zero, despite the fact that increasing
        # attribution to the opposite component would increase NLL further.
        obs_2_grad_signs = [0, 0, 0, 1, -1, 0]

        # At the mode of the lognormal distribution, the gradient of the NLL wrt mu is always exactly 1. The
        # gradient with respect to sigma is 1/sigma - sigma, so it depends on the value of sigma. At sigma =
        # 1, it is zero. At sigma > 1, it is negative. At sigma < 1, it is positive. Given multi-component
        # support, the gradient of NLL wrt the log weights will always favor the misaligned component.
        # These facts tell us the expected gradient signs of 4 of 6 elements of X for observations 3 and 4.
        # For the off-component mu and sigma gradient signs, we need to TODO

        obs_3_grad_signs = [1, 0, -1, 1, -1, 1]
        obs_4_grad_signs = [1, -1, 1, 1, -1, -1]

        want_X_grad_sign = torch.Tensor(
            [
                obs_1_grad_signs,
                obs_2_grad_signs,
                obs_3_grad_signs,
                obs_4_grad_signs,
            ]
        )
        self.assertEqual(X.grad.sign(), want_X_grad_sign)


class TestExponential(MLTypeEqualityCheckableMixin, unittest.TestCase):
    def test_forward(self):
        E = ExponentialTTELayer(in_dim=3)
        E.proj.weight = torch.nn.Parameter(torch.Tensor([[1, 0, 0]]))
        E.proj.bias = torch.nn.Parameter(torch.Tensor([0]))

        X = torch.Tensor(
            [
                [1.0, 3.0, 5.0],
                [3.0, 1.0, 5.0],
                [torch.finfo(torch.float32).min, 4.0, 0.0],
            ]
        )
        X.requires_grad = True

        # The rate should be approximately 1 + X, as it is an ELU(X) + 1 + epsilon. The mean should be 1/rate.
        want_rate = torch.Tensor([2.0, 4.0, torch.finfo(torch.float32).tiny])
        want_mean = torch.Tensor([1 / 2.0, 1 / 4.0, 1 / torch.finfo(torch.float32).tiny])

        out = E(X)
        self.assertEqual(out.rate, want_rate)
        self.assertEqual(out.mean, want_mean)
        out.rate.retain_grad()

        # The mean of the distribution is 1 / rate.
        # To decrease the log probability of samples that are greater than the mean, I need to increase the
        # rate for that position, so that 1/rate is smaller. This means that the derivative of the sum of the
        # NLL corresponding to that position of the rate should be positive (as increasing NLL decreases log
        # probability).
        # Conversely, for samples that are less than the mean, the gradient should be negative.
        # Finally, for samples that are equal to the mean, the gradient should be zero.

        obs = torch.Tensor(
            [
                0.75,
                0.1,
                1 / torch.finfo(torch.float32).tiny,
            ]
        )

        NLL = -out.log_prob(obs)
        NLL.sum().backward()

        want_rate_grad_sign = torch.Tensor([1, -1, 0])

        self.assertEqual(out.rate.grad.sign(), want_rate_grad_sign)

        # The projection layer of E will have a gradient dependent on the gradients of the rates and the first
        # two rows of X (as the final rate has a gradient of zero). To decrease the log probability furthest,
        # we need to increase the rate for the first position and decrease it for the second position. As the
        # first coordinate of X in the first position is smaller than that of the second, and the latter two
        # positions of the first coordinate of X are >= those of the second, this implies we will likely want
        # a negative sign for the first coordinate and positive signs for the next two coordinates.

        want_proj_grad_sign = torch.Tensor([[-1, 1, 1]])
        self.assertEqual(E.proj.weight.grad.sign(), want_proj_grad_sign)

        # Finally, X itself should have gradients with signs corresponding to rate's signs, only relevant in
        # the first position.
        want_X_grad_sign = torch.Tensor([[1, 0, 0], [-1, 0, 0], [0, 0, 0]])
        self.assertEqual(X.grad.sign(), want_X_grad_sign)


class TestGaussianIndexedRegressionLayer(MLTypeEqualityCheckableMixin, unittest.TestCase):
    def test_forward(self):
        L = GaussianIndexedRegressionLayer(in_dim=4, n_regression_targets=2)

        L.proj.weight = torch.nn.Parameter(
            torch.Tensor(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
        )
        L.proj.bias = torch.nn.Parameter(torch.Tensor([0, 0, 0, 0]))

        X = torch.Tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0, 0.0, 0.0],
                [1.0, torch.finfo(torch.float32).min, 0.0, 0.0],
            ]
        )
        X.requires_grad = True

        idx = torch.LongTensor(
            [
                [0, 1],
                [0, 1],
                [1, 0],
                [1, 0],
            ]
        )

        want_mean = torch.Tensor(
            [
                [0, 2],
                [0, 1],
                [0, 1],
                [0, 1],
            ]
        )
        # The std should be approximately 1 + X, as it is an ELU(X) + 1 + epsilon.
        want_std = torch.Tensor(
            [
                [2.0, 4.0],
                [1.0, 1.0],
                [1.0, 1],
                [1.0, torch.finfo(torch.float32).tiny],
            ]
        )

        got = L(X, idx)

        self.assertEqual(got.mean, want_mean)
        self.assertEqual(got.stddev, want_std)

        # To check gradient flow, we'll build observations that are to the left, right, or centered over the
        # mean of the distribution, and those that are close to or far from it's mean, and check that the
        # associated values in `X` (which goven the mean and standard deviation directly) have the desired
        # sign. One special case with the gradient is in the case where the standard deviation has saturated
        # the gradient -- when it is equal to our buffer of epsilon. Given saturation, we'd then expect a
        # gradient value of 0, regardless of the observed value.
        obs = torch.Tensor(
            [
                [-5, 2],  # Mean is far to the right, then right on top.
                [5, 1.05],  # Mean is far to the left, then close to the left.
                [-0.05, 5],  # Mean is close to the right, then far to the left.
                [-0.05, 5],  # Mean is close to the right, then far to the left.
            ]
        )

        NLL = -got.log_prob(obs)
        NLL.sum().backward()

        # To increase the NLL (equivalently to decrease likelihood), we want to send:
        # For X[0]
        #   * The mean of obs (0, 0) to the right, so X[0][0].grad should be +.
        #   * The std of obs (0, 0) down (to narrow the distribution), so X[0][1].grad should be -.
        #   * The mean of obs (0, 1) is stable so X[0][2].grad should be 0.
        #   * The std of obs (0, 1) up (to widen the distribution), so X[0][3].grad should be +.
        # For X[1]
        #   * The mean of obs (1, 0) to the left, so X[1][0].grad should be -.
        #   * The std of obs (1, 0) down (to narrow the distribution), so X[1][1].grad should be -.
        #   * The mean of obs (1, 1) to the left so X[1][2].grad should be -.
        #   * The std of obs (1, 1) up (to widen the distribution), so X[1][3].grad should be +.
        # For X[2]
        #   * The mean of obs (2, 1) to the left, so X[2][0].grad should be -. However, because it is *so*
        #   far to the left already (relative to the disribution's standard deviation), the distribution will
        #   have saturated, so it will have gradient 0.
        #   * The std of obs (2, 1) down (to narrow the distribution), so X[2][1].grad would be -, but this
        #     index is saturated, do it is actually 0.
        #   * The mean of obs (2, 0) to the right, so X[2][2].grad should be +.
        #   * The std of obs (2, 0) up (to widen the distribution), so X[2][3].grad should be +.

        # See https://github.com/pytorch/pytorch/issues/9688#issuecomment-435597041 for commentary on why the
        # test with finfo(float).min yields NaN gradients. TODO(mmd): Validate whether or not this is a real
        # problem.

        want_X_grad_sign = torch.Tensor(
            [
                [1, -1, 0, 1],
                [-1, -1, -1, 1],
                [-1, -1, 1, 1],
                # [0, 0, 1, 1],  # In reality, this is what the gradient in this case should be.
                [
                    0,
                    0,
                    0,
                    0,
                ],  # but in practice it is all NaNs as autograd can't recognize some stuff.
            ]
        )

        self.assertEqual(want_X_grad_sign, X.grad.sign(), msg=f"{want_X_grad_sign} vs.  {X.grad.sign()}")


if __name__ == "__main__":
    unittest.main()

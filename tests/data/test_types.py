import sys

sys.path.append("../..")

import unittest

import torch

from EventStream.data.types import PytorchBatch

from ..utils import ConfigComparisonsMixin


class TestPytorchBatch(ConfigComparisonsMixin, unittest.TestCase):
    def test_last_sequence_element_unsqueezed(self):
        batch = PytorchBatch(
            event_mask=torch.BoolTensor([[True, True, True], [True, True, False]]),
            time_delta=torch.FloatTensor([[1, 2, 3], [4, 5, 6]]),
            time=torch.FloatTensor([[0, 1, 3], [0, 4, 9]]),
            static_indices=torch.LongTensor([[1, 2, 3]]),
            static_measurement_indices=torch.LongTensor([[2, 3, 4]]),
            dynamic_indices=torch.LongTensor(
                [
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
                    [[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]],
                ]
            ),
            dynamic_measurement_indices=torch.LongTensor(
                [
                    [[[1, 1], [3, 3]], [[5, 5], [7, 7]], [[9, 9], [11, 11]]],
                    [[[13, 13], [15, 15]], [[17, 17], [19, 19]], [[21, 21], [23, 23]]],
                ]
            ),
            dynamic_values=torch.FloatTensor(
                [
                    [[[1.1, 1], [3.1, 3]], [[5.1, 5], [7.1, 7]], [[9.1, 9], [11.1, 11]]],
                    [[[13.1, 13], [15.1, 15]], [[17.1, 17], [19.1, 19]], [[21.1, 21], [23.1, 23]]],
                ]
            ),
            dynamic_values_mask=torch.BoolTensor(
                [
                    [
                        [[True, False], [True, False]],
                        [[True, False], [True, False]],
                        [[False, True], [False, False]],
                    ],
                    [
                        [[True, False], [True, True]],
                        [[True, False], [True, False]],
                        [[True, True], [False, True]],
                    ],
                ]
            ),
            start_time="start_time",
            stream_labels="stream_labels",
        )

        want = PytorchBatch(
            event_mask=torch.BoolTensor([[True], [False]]),
            time_delta=torch.FloatTensor([[3], [6]]),
            time=torch.FloatTensor([[3], [9]]),
            static_indices=torch.LongTensor([[1, 2, 3]]),
            static_measurement_indices=torch.LongTensor([[2, 3, 4]]),
            dynamic_indices=torch.LongTensor([[[[9, 10], [11, 12]]], [[[21, 22], [23, 24]]]]),
            dynamic_measurement_indices=torch.LongTensor(
                [[[[9, 9], [11, 11]]], [[[21, 21], [23, 23]]]]
            ),
            dynamic_values=torch.FloatTensor(
                [[[[9.1, 9], [11.1, 11]]], [[[21.1, 21], [23.1, 23]]]]
            ),
            dynamic_values_mask=torch.BoolTensor(
                [
                    [[[False, True], [False, False]]],
                    [[[True, True], [False, True]]],
                ]
            ),
            start_time="start_time",
            stream_labels="stream_labels",
        )

        got = batch.last_sequence_element_unsqueezed()
        self.assertEqual(want, got)

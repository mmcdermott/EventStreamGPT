import sys

sys.path.append("../..")

import unittest

import torch

from EventStream.data.types import PytorchBatch
from EventStream.transformer.generation.generation_stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)

BASE_BATCH = {
    "event_mask": torch.BoolTensor([[True, True, True, True]]),
    "time_delta": torch.FloatTensor([[0, 2, 5, 3]]),
    "start_time": torch.FloatTensor([1.0]),
    "static_indices": torch.LongTensor([[1, 2, 3]]),
    "static_measurement_indices": torch.LongTensor([[1, 2, 3]]),
    "dynamic_values_mask": torch.BoolTensor(
        [
            [
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, True, True, True],
                [False, False, False, False, True, True],
            ],
        ]
    ),
    "dynamic_measurement_indices": torch.LongTensor(
        [
            [
                [1, 0, 0, 0, 0, 0],
                [1, 2, 0, 0, 0, 0],
                [1, 2, 2, 3, 3, 3],
                [1, 2, 2, 2, 3, 3],
            ],
        ]
    ),
    "dynamic_indices": torch.LongTensor(
        [
            [
                [1, 0, 0, 0, 0, 0],
                [2, 5, 0, 0, 0, 0],
                [2, 4, 5, 7, 8, 9],
                [2, 4, 5, 5, 8, 9],
            ],
        ]
    ),
    "dynamic_values": torch.Tensor(
        [
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1.1, -1.1, 0.0],
                [0, 0, 0, 0, -3.1, 0.2],
            ],
        ]
    ),
}


class TestMaxLengthCriteria(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.batch = PytorchBatch(**BASE_BATCH)

    def test_criteria(self):
        C = MaxLengthCriteria(5)
        self.assertFalse(C(self.batch, None))

        C = MaxLengthCriteria(4)
        self.assertTrue(C(self.batch, None))


class TestStoppingCriteriaList(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.batch = PytorchBatch(**BASE_BATCH)

    def test_criteria(self):
        C = StoppingCriteriaList([MaxLengthCriteria(5), MaxLengthCriteria(6)])
        self.assertFalse(C(self.batch, None))
        C = StoppingCriteriaList([MaxLengthCriteria(5), MaxLengthCriteria(4)])
        self.assertTrue(C(self.batch, None))

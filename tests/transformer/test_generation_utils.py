import sys

sys.path.append("../..")

import unittest

import torch

from EventStream.data.types import PytorchBatch
from EventStream.transformer.generation.generation_utils import (
    StructuredGenerationMixin,
)

from ..utils import MLTypeEqualityCheckableMixin

BASE_BATCH = {
    "event_mask": torch.BoolTensor([[False, True, True], [True, True, True]]),
    "start_time": torch.FloatTensor([1, 2]),
    "time_delta": torch.FloatTensor([[1, 1 * 60, 1], [3 * 60, 5 * 60, 1]]),
    "static_indices": torch.LongTensor([[1, 8], [2, 4]]),
    "static_measurement_indices": torch.FloatTensor([[1, 2], [1, 2]]),
    "dynamic_measurement_indices": torch.LongTensor(
        [
            [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 4, 4], [1, 2, 3, 5, 4, 4]],
            [[0, 0, 0, 0, 0, 0], [1, 3, 3, 4, 4, 4], [1, 5, 3, 8, 4, 4]],
        ]
    ),
    "dynamic_indices": torch.LongTensor(
        [
            [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 7, 4], [1, 2, 3, 5, 9, 4]],
            [[0, 0, 0, 0, 7, 0], [1, 3, 3, 4, 6, 4], [1, 5, 3, 8, 4, 4]],
        ]
    ),
    "dynamic_values": torch.Tensor(
        [
            [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 4, 4], [1, 2, 3, 5, 4, 4]],
            [[0, 0, 0, 0, 0, 0], [1, 3, 3, 4, 4, 4], [1, 5, 3, 8, 4, 4]],
        ]
    ),
    "dynamic_values_mask": torch.BoolTensor(
        [
            [
                [False, False, False, False, False, False],
                [False, True, False, True, False, False],
                [False, True, False, True, False, True],
            ],
            [
                [False, True, False, False, False, False],
                [False, True, False, False, False, False],
                [False, True, False, False, True, False],
            ],
        ]
    ),
    "stream_labels": {"clf": torch.LongTensor([1, 44]), "reg": torch.FloatTensor([2.0, 1.8])},
    "time": None,
    "start_idx": None,
    "end_idx": None,
    "subject_id": None,
}

EXPANDED_BATCH_2 = {
    "event_mask": torch.BoolTensor(
        [
            [False, True, True],
            [False, True, True],
            [True, True, True],
            [True, True, True],
        ]
    ),
    "start_time": torch.FloatTensor([1, 1, 2, 2]),
    "time_delta": torch.FloatTensor(
        [
            [1, 1 * 60, 1],
            [1, 1 * 60, 1],
            [3 * 60, 5 * 60, 1],
            [3 * 60, 5 * 60, 1],
        ]
    ),
    "static_indices": torch.LongTensor(
        [
            [1, 8],
            [1, 8],
            [2, 4],
            [2, 4],
        ]
    ),
    "static_measurement_indices": torch.FloatTensor(
        [
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
        ]
    ),
    "dynamic_measurement_indices": torch.LongTensor(
        [
            [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 4, 4], [1, 2, 3, 5, 4, 4]],
            [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 4, 4], [1, 2, 3, 5, 4, 4]],
            [[0, 0, 0, 0, 0, 0], [1, 3, 3, 4, 4, 4], [1, 5, 3, 8, 4, 4]],
            [[0, 0, 0, 0, 0, 0], [1, 3, 3, 4, 4, 4], [1, 5, 3, 8, 4, 4]],
        ]
    ),
    "dynamic_indices": torch.LongTensor(
        [
            [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 7, 4], [1, 2, 3, 5, 9, 4]],
            [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 7, 4], [1, 2, 3, 5, 9, 4]],
            [[0, 0, 0, 0, 7, 0], [1, 3, 3, 4, 6, 4], [1, 5, 3, 8, 4, 4]],
            [[0, 0, 0, 0, 7, 0], [1, 3, 3, 4, 6, 4], [1, 5, 3, 8, 4, 4]],
        ]
    ),
    "dynamic_values": torch.Tensor(
        [
            [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 4, 4], [1, 2, 3, 5, 4, 4]],
            [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 4, 4], [1, 2, 3, 5, 4, 4]],
            [[0, 0, 0, 0, 0, 0], [1, 3, 3, 4, 4, 4], [1, 5, 3, 8, 4, 4]],
            [[0, 0, 0, 0, 0, 0], [1, 3, 3, 4, 4, 4], [1, 5, 3, 8, 4, 4]],
        ]
    ),
    "dynamic_values_mask": torch.BoolTensor(
        [
            [
                [False, False, False, False, False, False],
                [False, True, False, True, False, False],
                [False, True, False, True, False, True],
            ],
            [
                [False, False, False, False, False, False],
                [False, True, False, True, False, False],
                [False, True, False, True, False, True],
            ],
            [
                [False, True, False, False, False, False],
                [False, True, False, False, False, False],
                [False, True, False, False, True, False],
            ],
            [
                [False, True, False, False, False, False],
                [False, True, False, False, False, False],
                [False, True, False, False, True, False],
            ],
        ]
    ),
    "stream_labels": {
        "clf": torch.LongTensor([1, 1, 44, 44]),
        "reg": torch.FloatTensor([2, 2, 1.8, 1.8]),
    },
    "time": None,
    "start_idx": None,
    "end_idx": None,
    "subject_id": None,
}


class TestGenerativeSequenceModelSamples(MLTypeEqualityCheckableMixin, unittest.TestCase):
    """Tests Generation Batch-building Logic."""

    def setUp(self):
        super().setUp()
        self.batch = PytorchBatch(**BASE_BATCH)

    def test_expand_inputs_for_generation(self):
        out_batch = StructuredGenerationMixin._expand_inputs_for_generation(batch=self.batch, expand_size=1)
        self.assertNestedDictEqual(BASE_BATCH, {k: v for k, v in out_batch.items()})

        out_batch_2 = StructuredGenerationMixin._expand_inputs_for_generation(batch=self.batch, expand_size=2)
        self.assertNestedDictEqual(EXPANDED_BATCH_2, {k: v for k, v in out_batch_2.items()})


if __name__ == "__main__":
    unittest.main()

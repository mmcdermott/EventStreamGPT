import sys

sys.path.append("../..")

import copy
import unittest

import torch

from EventStream.data.types import PytorchBatch
from EventStream.transformer.generation.generation_utils import (
    StructuredGenerationMixin,
)

from ...utils import ConfigComparisonsMixin


class TestGenerationUtils(ConfigComparisonsMixin, unittest.TestCase):
    def test_expand_inputs_for_generation(self):
        batch = PytorchBatch(
            event_mask=torch.BoolTensor([[True, True, True, True], [False, True, True, True]]),
            time_delta=torch.FloatTensor([[0, 2, 5, 3], [0, 3, 2, 3]]),
            start_time=torch.FloatTensor([1.0, 1412.0]),
            static_indices=torch.LongTensor([[1, 2, 3], [1, 3, 0]]),
            static_measurement_indices=torch.LongTensor([[1, 2, 3], [1, 3, 0]]),
            dynamic_values_mask=torch.BoolTensor(
                [
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, True, True, True],
                        [False, False, False, False, True, True],
                    ],
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, True],
                        [False, False, False, False, True, True],
                    ],
                ]
            ),
            dynamic_measurement_indices=torch.LongTensor(
                [
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 2, 0, 0, 0, 0],
                        [1, 2, 2, 3, 3, 3],
                        [1, 2, 2, 2, 3, 3],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 2, 0, 0, 0, 0],
                        [1, 2, 2, 2, 2, 3],
                        [1, 2, 2, 2, 3, 3],
                    ],
                ]
            ),
            dynamic_indices=torch.LongTensor(
                [
                    [
                        [1, 0, 0, 0, 0, 0],
                        [2, 5, 0, 0, 0, 0],
                        [2, 4, 5, 7, 8, 9],
                        [2, 4, 5, 5, 8, 9],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [2, 5, 0, 0, 0, 0],
                        [2, 4, 5, 4, 4, 9],
                        [2, 4, 5, 5, 8, 9],
                    ],
                ]
            ),
            dynamic_values=torch.Tensor(
                [
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1.1, -1.1, 0.0],
                        [0, 0, 0, 0, -3.1, 0.2],
                    ],
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1.4],
                        [0, 0, 0, 0, -3.0, 1.2],
                    ],
                ]
            ),
        )

        want_expanded_batch = PytorchBatch(
            event_mask=torch.BoolTensor(
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [False, True, True, True],
                    [False, True, True, True],
                ]
            ),
            time_delta=torch.FloatTensor([[0, 2, 5, 3], [0, 2, 5, 3], [0, 3, 2, 3], [0, 3, 2, 3]]),
            start_time=torch.FloatTensor([1.0, 1.0, 1412.0, 1412.0]),
            static_indices=torch.LongTensor([[1, 2, 3], [1, 2, 3], [1, 3, 0], [1, 3, 0]]),
            static_measurement_indices=torch.LongTensor(
                [[1, 2, 3], [1, 2, 3], [1, 3, 0], [1, 3, 0]]
            ),
            dynamic_values_mask=torch.BoolTensor(
                [
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, True, True, True],
                        [False, False, False, False, True, True],
                    ],
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, True, True, True],
                        [False, False, False, False, True, True],
                    ],
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, True],
                        [False, False, False, False, True, True],
                    ],
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, True],
                        [False, False, False, False, True, True],
                    ],
                ]
            ),
            dynamic_measurement_indices=torch.LongTensor(
                [
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 2, 0, 0, 0, 0],
                        [1, 2, 2, 3, 3, 3],
                        [1, 2, 2, 2, 3, 3],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 2, 0, 0, 0, 0],
                        [1, 2, 2, 3, 3, 3],
                        [1, 2, 2, 2, 3, 3],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 2, 0, 0, 0, 0],
                        [1, 2, 2, 2, 2, 3],
                        [1, 2, 2, 2, 3, 3],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 2, 0, 0, 0, 0],
                        [1, 2, 2, 2, 2, 3],
                        [1, 2, 2, 2, 3, 3],
                    ],
                ]
            ),
            dynamic_indices=torch.LongTensor(
                [
                    [
                        [1, 0, 0, 0, 0, 0],
                        [2, 5, 0, 0, 0, 0],
                        [2, 4, 5, 7, 8, 9],
                        [2, 4, 5, 5, 8, 9],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [2, 5, 0, 0, 0, 0],
                        [2, 4, 5, 7, 8, 9],
                        [2, 4, 5, 5, 8, 9],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [2, 5, 0, 0, 0, 0],
                        [2, 4, 5, 4, 4, 9],
                        [2, 4, 5, 5, 8, 9],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [2, 5, 0, 0, 0, 0],
                        [2, 4, 5, 4, 4, 9],
                        [2, 4, 5, 5, 8, 9],
                    ],
                ]
            ),
            dynamic_values=torch.Tensor(
                [
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1.1, -1.1, 0.0],
                        [0, 0, 0, 0, -3.1, 0.2],
                    ],
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1.1, -1.1, 0.0],
                        [0, 0, 0, 0, -3.1, 0.2],
                    ],
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1.4],
                        [0, 0, 0, 0, -3.0, 1.2],
                    ],
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1.4],
                        [0, 0, 0, 0, -3.0, 1.2],
                    ],
                ]
            ),
        )

        got = StructuredGenerationMixin._expand_inputs_for_generation(batch, 2)
        self.assertEqual(want_expanded_batch, got)

        bad_batch = copy.deepcopy(batch)
        bad_batch.dynamic_values = None
        with self.assertRaises(TypeError):
            StructuredGenerationMixin._expand_inputs_for_generation(bad_batch, 2)

    def test_update_model_kwargs_for_generation(self):
        self.assertEqual(
            {"past": None},
            StructuredGenerationMixin._update_model_kwargs_for_generation(
                {"wrong_key": "present"}, {}
            ),
        )
        self.assertEqual(
            {"past": "present"},
            StructuredGenerationMixin._update_model_kwargs_for_generation(
                {"past_key_values": "present"}, {}
            ),
        )


if __name__ == "__main__":
    unittest.main()

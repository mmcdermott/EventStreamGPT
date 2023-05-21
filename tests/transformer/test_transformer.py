import sys

sys.path.append("../..")

import copy
import unittest

import torch

from EventStream.data.types import PytorchBatch
from EventStream.transformer.config import (
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
)
from EventStream.transformer.transformer import (
    ConditionallyIndependentPointProcessTransformer,
)

from ..mixins import ConfigComparisonsMixin

TEST_DATA_TYPES_PER_GEN_MODE = {
    "single_label_classification": ["event_type"],
    "multi_label_classification": ["multi_label_col", "regression_col"],
    "partially_observed_regression": ["regression_col"],
}
TEST_DATA_TYPES_IDXMAP = {
    "event_type": 1,
    "multi_label_col": 2,
    "regression_col": 3,
}
# These are all including the 'UNK' tokens. So, e.g., there are 2 real options for 'event_type'.
TEST_VOCAB_SIZES_BY_DATA_TYPE = {
    "event_type": 2,
    "multi_label_col": 3,
    "regression_col": 4,
}
TEST_VOCAB_OFFSETS_BY_DATA_TYPE = {
    "event_type": 1,
    "multi_label_col": 3,
    "regression_col": 6,
}
TEST_MEASUREMENTS_PER_DEP_GRAPH_LEVEL = [[], ["event_type"], ["multi_label_col", "regression_col"]]

BASE_CONFIG_KWARGS = dict(
    structured_event_processing_mode=StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT,
    dep_graph_attention_types=None,
    dep_graph_window_size=None,
    do_full_block_in_dep_graph_attention=None,
    do_full_block_in_seq_attention=None,
    do_add_temporal_position_embeddings_to_data_embeddings=None,
    measurements_per_generative_mode=TEST_DATA_TYPES_PER_GEN_MODE,
    vocab_sizes_by_measurement=TEST_VOCAB_SIZES_BY_DATA_TYPE,
    vocab_offsets_by_measurement=TEST_VOCAB_OFFSETS_BY_DATA_TYPE,
    measurements_idxmap=TEST_DATA_TYPES_IDXMAP,
    vocab_size=10,
    hidden_size=4,
    num_hidden_layers=2,
    head_dim=None,
    num_attention_heads=2,  # Needs to divide hidden_size.
    mean_log_inter_time=0,
    std_log_inter_time=1,
    use_cache=False,
    measurements_per_dep_graph_level=TEST_MEASUREMENTS_PER_DEP_GRAPH_LEVEL,
)

BASE_BATCH = {
    "event_mask": torch.BoolTensor([[True, True, True]]),
    "time_delta": torch.FloatTensor([[0, 2, 5]]),
    "static_indices": torch.LongTensor([[1, 2, 3]]),
    "static_measurement_indices": torch.LongTensor([[1, 2, 3]]),
    "dynamic_values_mask": torch.BoolTensor(
        [
            [
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, True, True, True],
            ],
        ]
    ),
    "dynamic_measurement_indices": torch.LongTensor(
        [
            [
                [1, 0, 0, 0, 0, 0],
                [1, 2, 0, 0, 0, 0],
                [1, 2, 2, 3, 3, 3],
            ],
        ]
    ),
    "dynamic_indices": torch.LongTensor(
        [
            [
                [1, 0, 0, 0, 0, 0],
                [2, 5, 0, 0, 0, 0],
                [2, 4, 5, 7, 8, 9],
            ],
        ]
    ),
    "dynamic_values": torch.Tensor(
        [
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1.1, -1.1, 0.0],
            ],
        ]
    ),
}


class TestStructuredTransformer(ConfigComparisonsMixin, unittest.TestCase):
    def test_forward_sensitive_to_event_mask_with_batch(self):
        config = StructuredTransformerConfig(**BASE_CONFIG_KWARGS)

        M = ConditionallyIndependentPointProcessTransformer(config).cpu()
        M.eval()  # So layernorm and dropout don't affect anything.

        batch = PytorchBatch(**copy.deepcopy(BASE_BATCH))
        batch2 = copy.deepcopy(batch)
        out1 = M(batch)
        out1_alt = M(batch2)

        self.assertEqual(out1, out1_alt)

        batch.event_mask = torch.BoolTensor([[False, False, True]])

        out2 = M(batch)
        with self.assertRaises(AssertionError):
            self.assertEqual(out1, out2)


if __name__ == "__main__":
    unittest.main()

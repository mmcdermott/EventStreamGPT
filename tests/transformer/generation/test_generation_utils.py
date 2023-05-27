import sys

sys.path.append("../..")

import copy
import unittest

import torch

from EventStream.data.config import MeasurementConfig
from EventStream.data.types import DataModality, PytorchBatch, TemporalityType
from EventStream.transformer.config import (
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
)
from EventStream.transformer.nested_attention_model import (
    NAPPTForGenerativeSequenceModeling,
)

from ...utils import ConfigComparisonsMixin

TEST_DATA_TYPES_PER_GEN_MODE = {
    "single_label_classification": ["event_type"],
    "multi_label_classification": ["multi_label_col", "regression_col"],
    "multivariate_regression": ["regression_col"],
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


NA_CONFIG_KWARGS = dict(
    structured_event_processing_mode=StructuredEventProcessingMode.NESTED_ATTENTION,
    dep_graph_attention_types=["global"],
    dep_graph_window_size=None,
    do_full_block_in_dep_graph_attention=True,
    do_full_block_in_seq_attention=True,
    measurements_per_generative_mode=TEST_DATA_TYPES_PER_GEN_MODE,
    vocab_sizes_by_measurement=TEST_VOCAB_SIZES_BY_DATA_TYPE,
    vocab_offsets_by_measurement=TEST_VOCAB_OFFSETS_BY_DATA_TYPE,
    measurements_idxmap=TEST_DATA_TYPES_IDXMAP,
    vocab_size=10,
    hidden_size=4,
    num_hidden_layers=5,
    head_dim=None,
    num_attention_heads=2,  # Needs to divide hidden_size.
    mean_log_inter_time=0,
    std_log_inter_time=1,
    use_cache=False,
    measurements_per_dep_graph_level=TEST_MEASUREMENTS_PER_DEP_GRAPH_LEVEL,
    measurement_configs={
        "multi_label_col": MeasurementConfig(
            modality=DataModality.MULTI_LABEL_CLASSIFICATION,
            temporality=TemporalityType.DYNAMIC,
        ),
        "regression_col": MeasurementConfig(
            modality=DataModality.MULTIVARIATE_REGRESSION,
            temporality=TemporalityType.DYNAMIC,
            values_column="regression_val",
        ),
    },
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


class TestNAPPTGeneration(ConfigComparisonsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.config = StructuredTransformerConfig(**NA_CONFIG_KWARGS)

        self.M = NAPPTForGenerativeSequenceModeling(self.config).cpu()
        self.M.eval()  # So layernorm and dropout don't affect anything.

        self.batch = PytorchBatch(**copy.deepcopy(BASE_BATCH))

    def test_expand_inputs_for_generation(self):
        want_expanded_batch = PytorchBatch(
            **{
                "event_mask": torch.BoolTensor(
                    [[True, True, True, True], [True, True, True, True]]
                ),
                "time_delta": torch.FloatTensor([[0, 2, 5, 3], [0, 2, 5, 3]]),
                "start_time": torch.FloatTensor([1.0, 1.0]),
                "static_indices": torch.LongTensor([[1, 2, 3], [1, 2, 3]]),
                "static_measurement_indices": torch.LongTensor([[1, 2, 3], [1, 2, 3]]),
                "dynamic_values_mask": torch.BoolTensor(
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
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1.1, -1.1, 0.0],
                            [0, 0, 0, 0, -3.1, 0.2],
                        ],
                    ]
                ),
            }
        )

        self.assertEqual(want_expanded_batch, self.M._expand_inputs_for_generation(self.batch, 2))

        bad_batch = copy.deepcopy(self.batch)
        bad_batch.dynamic_values = None
        with self.assertRaises(TypeError):
            self.M._expand_inputs_for_generation(bad_batch, 2)

    def test_update_model_kwargs_for_generation(self):
        self.assertEqual(
            {"past": None},
            self.M._update_model_kwargs_for_generation({"wrong_key": "present"}, {}),
        )
        self.assertEqual(
            {"past": "present"},
            self.M._update_model_kwargs_for_generation({"past_key_values": "present"}, {}),
        )

    @unittest.skip
    def test_generation_identical_with_or_without_caching(self):
        # We want to check that the output doesn't change when we do or do not use caching. To do this, we'll
        # run the model over a partial batch without caching and store the result. Then, we'll run the model
        # over various elements of that batch, iterating through in sequence, using caching to only ever run
        # the attention calculation on the last element, and we'll validate that the predictions don't change
        # in comparison to the run without caching.

        generation_kwargs = dict(
            max_new_events=5,
            num_return_sequences=2,
            do_sample=True,
            return_dict_in_generate=False,
            output_scores=False,
            output_attentions=False,
            output_hidden_states=False,
            debug_seed=1,
        )

        out_no_caching_1 = self.M.generate(self.batch, **generation_kwargs, use_cache=False)

        out_no_caching_2 = self.M.generate(self.batch, **generation_kwargs, use_cache=False)

        self.assertEqual(out_no_caching_1, out_no_caching_2)

        out_with_caching = self.M.generate(self.batch, **generation_kwargs, use_cache=True)

        self.assertEqual(out_no_caching_1, out_with_caching)


if __name__ == "__main__":
    unittest.main()

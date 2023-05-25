import sys

sys.path.append("../..")

import copy
import unittest
from typing import Any

import torch

from EventStream.data.types import PytorchBatch
from EventStream.transformer.config import (
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
)
from EventStream.transformer.model_output import TransformerOutputWithPast
from EventStream.transformer.transformer import (
    ConditionallyIndependentPointProcessTransformer,
    NestedAttentionPointProcessTransformer,
    time_from_deltas,
)

from ..mixins import ConfigComparisonsMixin


def print_debug_info(v: Any) -> str:
    match v:
        case None:
            return "None"
        case torch.Tensor():
            return str(v.shape)
        case dict() as v_dict:
            els_strs = "\n    ".join(f"{k}: {print_debug_info(v)}" for k, v in v_dict.items())
            return f"{type(v_dict)} of len {len(v_dict)}\n" f"  Elements:\n" f"    {els_strs}"
        case (list() | tuple()) as v_list:
            els_strs = {f"{print_debug_info(v)}" for v in v_list}
            return f"{type(v_list)} of len {len(v_list)} with elements: {', '.join(els_strs)}"
        case _:
            return str(v)


TEST_DATA_TYPES_PER_GEN_MODE = {
    "single_label_classification": ["event_type"],
    "multi_label_classification": ["multi_label_col"],
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

CI_CONFIG_KWARGS = dict(
    structured_event_processing_mode=StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT,
    dep_graph_attention_types=None,
    dep_graph_window_size=None,
    do_full_block_in_dep_graph_attention=None,
    do_full_block_in_seq_attention=None,
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
    measurements_per_dep_graph_level=None,
)

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
)

BASE_BATCH = {
    "event_mask": torch.BoolTensor([[True, True, True, True]]),
    "time_delta": torch.FloatTensor([[0, 2, 5, 3]]),
    "static_indices": torch.LongTensor([[1, 2, 3]]),
    "static_measurement_indices": torch.LongTensor([[1, 2, 3]]),
    "dynamic_values_mask": torch.BoolTensor(
        [
            [
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, True, True, True],
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
                [2, 4, 5, 9, 8, 9],
            ],
        ]
    ),
    "dynamic_values": torch.Tensor(
        [
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1.1, -1.1, 0.0],
                [0, 0, 0, 1.2, -3.1, 0.2],
            ],
        ]
    ),
}


class TestConditionallyIndependentTransformer(ConfigComparisonsMixin, unittest.TestCase):
    def test_constructs(self):
        config_valid = StructuredTransformerConfig(**CI_CONFIG_KWARGS)
        ConditionallyIndependentPointProcessTransformer(config_valid)
        config_invalid = StructuredTransformerConfig(**NA_CONFIG_KWARGS)
        with self.assertRaises(ValueError):
            ConditionallyIndependentPointProcessTransformer(config_invalid)

    def test_forward_sensitive_to_event_mask_with_batch(self):
        config = StructuredTransformerConfig(**CI_CONFIG_KWARGS)

        M = ConditionallyIndependentPointProcessTransformer(config).cpu()
        M.eval()  # So layernorm and dropout don't affect anything.

        batch = PytorchBatch(**copy.deepcopy(BASE_BATCH))
        batch2 = copy.deepcopy(batch)
        out1 = M(batch)
        out1_alt = M(batch2)

        self.assertEqual(out1, out1_alt)

        batch.event_mask = torch.BoolTensor([[False, False, True, True]])

        out2 = M(batch)
        with self.assertRaises(AssertionError):
            self.assertEqual(out1, out2)

    @unittest.skip("TODO: Implement caching.")
    def test_forward_identical_with_or_without_caching(self):
        raise NotImplementedError


class TestNestedAttentionTransformer(ConfigComparisonsMixin, unittest.TestCase):
    def test_constructs(self):
        config_valid = StructuredTransformerConfig(**NA_CONFIG_KWARGS)
        NestedAttentionPointProcessTransformer(config_valid)
        config_invalid = StructuredTransformerConfig(**CI_CONFIG_KWARGS)
        with self.assertRaises(ValueError):
            NestedAttentionPointProcessTransformer(config_invalid)

    def test_forward_sensitive_to_event_mask_with_batch(self):
        config = StructuredTransformerConfig(**NA_CONFIG_KWARGS)

        M = NestedAttentionPointProcessTransformer(config).cpu()
        M.eval()  # So layernorm and dropout don't affect anything.

        batch = PytorchBatch(**copy.deepcopy(BASE_BATCH))
        batch2 = copy.deepcopy(batch)
        out1 = M(batch)
        out1_alt = M(batch2)

        self.assertEqual(out1, out1_alt)

        batch.event_mask = torch.BoolTensor([[False, False, True, False]])

        out2 = M(batch)
        with self.assertRaises(AssertionError):
            self.assertEqual(out1, out2)

    def test_forward_identical_with_or_without_caching(self):
        config = StructuredTransformerConfig(**NA_CONFIG_KWARGS)

        M = NestedAttentionPointProcessTransformer(config).cpu()
        M.eval()  # So layernorm and dropout don't affect anything.

        batch = PytorchBatch(**copy.deepcopy(BASE_BATCH))

        # We want to check that the output doesn't change when we do or do not use caching. To do this, we'll
        # run the model over a partial batch without caching and store the result. Then, we'll run the model
        # over various elements of that batch, iterating through in sequence, using caching to only ever run
        # the attention calculation on the last element, and we'll validate that the predictions don't change
        # in comparison to the run without caching.

        out_no_caching = M(batch, return_dict=True, use_cache=False)

        out_with_caching_full = M(
            batch,
            return_dict=True,
            use_cache=True,
        )
        out_with_caching_full["past_key_values"] = None
        self.assertEqual(out_no_caching, out_with_caching_full)

        source_batch_for_slicing = PytorchBatch(**copy.deepcopy(BASE_BATCH))
        # We need to add time explicitly here as that will be lost when we slice the batch. This happens
        # naturally during generation.
        source_batch_for_slicing.time = time_from_deltas(source_batch_for_slicing.time_delta)

        # We can't actually start an iterative caching process from nothing -- the system isn't designed for
        # that; Instead, we'll run the first sequence element outside the iterative selection and capture it's
        # output there, then use that as the starting point for the iterative cache-based computation.

        seq_idx = 1
        dep_graph_idx = None

        sliced_batch = copy.deepcopy(source_batch_for_slicing)
        for param in (
            "time",
            "event_mask",
            "time_delta",
            "dynamic_indices",
            "dynamic_measurement_indices",
            "dynamic_values",
            "dynamic_values_mask",
        ):
            orig_val = getattr(sliced_batch, param)
            sliced_val = orig_val[:, : (seq_idx + 1)]
            setattr(sliced_batch, param, sliced_val)

        sliced_out = M(
            sliced_batch,
            return_dict=True,
            use_cache=True,
            dep_graph_past=dep_graph_idx,
            dep_graph_el_generation_target=None,
            past=None,
        )

        new_joint_past = sliced_out["past_key_values"]
        sliced_out["past_key_values"] = None

        past = new_joint_past["seq_past"]
        dep_graph_past = new_joint_past["dep_graph_past"]

        out_with_caching = [sliced_out]
        for seq_idx in range(2, batch.sequence_length):
            out_with_caching_seq = []
            for dep_graph_idx in [1, 2, 0]:
                sliced_batch = copy.deepcopy(source_batch_for_slicing)
                for param in (
                    "time",
                    "event_mask",
                    "time_delta",
                    "dynamic_indices",
                    "dynamic_measurement_indices",
                    "dynamic_values",
                    "dynamic_values_mask",
                ):
                    orig_val = getattr(sliced_batch, param)
                    sliced_val = orig_val[:, seq_idx].unsqueeze(1)
                    setattr(sliced_batch, param, sliced_val)

                sliced_out = M(
                    sliced_batch,
                    return_dict=True,
                    use_cache=True,
                    dep_graph_past=dep_graph_past,
                    dep_graph_el_generation_target=dep_graph_idx,
                    past=past,
                )

                new_joint_past = sliced_out["past_key_values"]
                sliced_out["past_key_values"] = None

                past = new_joint_past["seq_past"]
                dep_graph_past = new_joint_past["dep_graph_past"]

                out_with_caching_seq.append(sliced_out)

            joint_seq_out_with_caching = TransformerOutputWithPast(
                last_hidden_state=torch.cat(
                    [x.last_hidden_state for x in out_with_caching_seq], dim=2
                ),
            )
            out_with_caching.append(joint_seq_out_with_caching)

        out_with_caching = TransformerOutputWithPast(
            last_hidden_state=torch.cat([x.last_hidden_state for x in out_with_caching], dim=1),
        )

        self.assertEqual(out_no_caching, out_with_caching)


if __name__ == "__main__":
    unittest.main()

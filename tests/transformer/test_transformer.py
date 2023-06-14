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
from EventStream.transformer.model_output import TransformerOutputWithPast
from EventStream.transformer.transformer import (
    ConditionallyIndependentPointProcessTransformer,
    NestedAttentionPointProcessTransformer,
    expand_mask,
    time_from_deltas,
)

from ..utils import ConfigComparisonsMixin

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
    "event_mask": torch.BoolTensor([[True, True, True, True], [False, True, True, True]]),
    "time_delta": torch.FloatTensor([[0, 2, 5, 3], [0, 3, 2, 3]]),
    "static_indices": torch.LongTensor([[1, 2, 3], [1, 3, 0]]),
    "static_measurement_indices": torch.LongTensor([[1, 2, 3], [1, 2, 0]]),
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
                [False, False, False, False, False, True],
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
                [1, 2, 2, 2, 2, 3],
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
                [2, 4, 5, 4, 4, 9],
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
                [0, 0, 0, 0, 0, 1.4],
                [0, 0, 0, 0, -3.0, 1.2],
            ],
        ]
    ),
}


class TestConditionallyIndependentTransformer(ConfigComparisonsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.config = StructuredTransformerConfig(**CI_CONFIG_KWARGS)

        self.M = ConditionallyIndependentPointProcessTransformer(self.config).cpu()
        self.M.eval()  # So layernorm and dropout don't affect anything.

        self.batch = PytorchBatch(**copy.deepcopy(BASE_BATCH))

    def test_constructs(self):
        config_valid = StructuredTransformerConfig(**CI_CONFIG_KWARGS)
        ConditionallyIndependentPointProcessTransformer(config_valid)
        config_invalid = StructuredTransformerConfig(**NA_CONFIG_KWARGS)
        with self.assertRaises(ValueError):
            ConditionallyIndependentPointProcessTransformer(config_invalid)

    def test_forward_sensitive_to_event_mask_with_batch(self):
        batch2 = copy.deepcopy(self.batch)
        out1 = self.M(self.batch)
        out1_alt = self.M(batch2)

        self.assertEqual(out1, out1_alt)

        batch2.event_mask = torch.BoolTensor([[False, False, True, True], [True, True, True, True]])

        out2 = self.M(batch2)
        with self.assertRaises(AssertionError):
            self.assertEqual(out1, out2)

    def test_forward_batch_shape_respected(self):
        out = self.M(self.batch)

        batch_subj_0 = self.batch[:1]
        batch_subj_1 = self.batch[1:2]

        out_subj_0 = self.M(batch_subj_0)
        out_subj_1 = self.M(batch_subj_1)

        self.assertEqual(out_subj_0.last_hidden_state, out.last_hidden_state[:1])
        self.assertEqual(out_subj_1.last_hidden_state, out.last_hidden_state[1:2])

    def test_forward_seq_shape_respected(self):
        out = self.M(self.batch)
        out_seq_to_2 = self.M(self.batch[:, :2])
        self.assertEqual(out_seq_to_2.last_hidden_state, out.last_hidden_state[:, :2])

    def test_forward_identical_with_or_without_caching(self):
        # We want to check that the output doesn't change when we do or do not use caching. To do this, we'll
        # run the model over a partial batch without caching and store the result. Then, we'll run the model
        # over various elements of that batch, iterating through in sequence, using caching to only ever run
        # the attention calculation on the last element, and we'll validate that the predictions don't change
        # in comparison to the run without caching.

        out_no_caching = self.M(self.batch, return_dict=True, use_cache=False)

        out_full_caching = self.M(self.batch, return_dict=True, use_cache=True)
        full_seq_past_up_to_2 = tuple(
            tuple(e[:, :, :2, :] for e in ee) for ee in out_full_caching["past_key_values"]
        )

        out_full_caching["past_key_values"] = None
        self.assertEqual(out_no_caching, out_full_caching)

        source_batch_for_slicing = PytorchBatch(**copy.deepcopy(BASE_BATCH))
        # We need to add time explicitly here as that will be lost when we slice the batch. This happens
        # naturally during generation.
        source_batch_for_slicing.time = time_from_deltas(source_batch_for_slicing)

        # We can't actually start an iterative caching process from nothing -- the system isn't designed for
        # that; Instead, we'll run the first sequence element outside the iterative selection and capture it's
        # output there, then use that as the starting point for the iterative cache-based computation.

        seq_attention_mask = expand_mask(self.batch.event_mask, out_full_caching.last_hidden_state.dtype)
        M_kwargs = {"return_dict": True, "use_cache": True}

        seq_idx = 1

        sliced_batch = copy.deepcopy(source_batch_for_slicing)
        sliced_batch = sliced_batch[:, : (seq_idx + 1)]

        sliced_out = self.M(
            sliced_batch,
            past=None,
            seq_attention_mask=seq_attention_mask[:, :, :, : (seq_idx + 1)],
            **M_kwargs,
        )

        self.assertEqual(
            out_no_caching.last_hidden_state[:, :2],
            sliced_out.last_hidden_state,
            "The initial slicing shouldn't impact the last hidden state.",
        )

        past = sliced_out["past_key_values"]
        self.assertNestedEqual(full_seq_past_up_to_2, past)

        out_iterative_caching = [sliced_out]
        for seq_idx in range(2, self.batch.sequence_length):
            sliced_batch = copy.deepcopy(source_batch_for_slicing)
            sliced_batch = sliced_batch[:, seq_idx : seq_idx + 1]

            sliced_out = self.M(
                sliced_batch,
                past=past,
                seq_attention_mask=seq_attention_mask[:, :, :, : (seq_idx + 1)],
                **M_kwargs,
            )

            # sliced_batch = copy.deepcopy(source_batch_for_slicing)
            # sliced_batch = sliced_batch[:, : (seq_idx + 1)]
            # seq_attention_mask = expand_mask(
            #    sliced_batch.event_mask, out_full_caching.last_hidden_state.dtype
            # )

            # sliced_batch = sliced_batch.last_sequence_element_unsqueezed()

            # sliced_out = self.M(
            #    sliced_batch,
            #    past=past,
            #    seq_attention_mask=seq_attention_mask,
            #    **M_kwargs,
            # )

            past = sliced_out["past_key_values"]
            out_iterative_caching.append(sliced_out)

        out_iterative_caching = TransformerOutputWithPast(
            last_hidden_state=torch.cat([x.last_hidden_state for x in out_iterative_caching], dim=1),
        )

        self.assertEqual(out_no_caching, out_iterative_caching)


class TestNestedAttentionTransformer(ConfigComparisonsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.config = StructuredTransformerConfig(**NA_CONFIG_KWARGS)

        self.M = NestedAttentionPointProcessTransformer(self.config).cpu()
        self.M.eval()  # So layernorm and dropout don't affect anything.

        self.batch = PytorchBatch(**copy.deepcopy(BASE_BATCH))

    def test_constructs(self):
        config_valid = StructuredTransformerConfig(**NA_CONFIG_KWARGS)
        NestedAttentionPointProcessTransformer(config_valid)
        config_invalid = StructuredTransformerConfig(**CI_CONFIG_KWARGS)
        with self.assertRaises(ValueError):
            NestedAttentionPointProcessTransformer(config_invalid)

    def test_forward_sensitive_to_event_mask_with_batch(self):
        batch2 = copy.deepcopy(self.batch)
        out1 = self.M(self.batch)
        out1_alt = self.M(batch2)

        self.assertEqual(out1, out1_alt)

        batch2.event_mask = torch.BoolTensor([[False, False, True, True], [True, True, True, True]])

        out2 = self.M(batch2)
        with self.assertRaises(AssertionError):
            self.assertEqual(out1, out2)

    def test_forward_batch_shape_respected(self):
        out = self.M(self.batch)

        batch_subj_0 = self.batch[:1]
        batch_subj_1 = self.batch[1:2]

        out_subj_0 = self.M(batch_subj_0)
        out_subj_1 = self.M(batch_subj_1)

        self.assertEqual(out_subj_0.last_hidden_state, out.last_hidden_state[:1])
        self.assertEqual(out_subj_1.last_hidden_state, out.last_hidden_state[1:2])

    def test_forward_seq_shape_respected(self):
        out = self.M(self.batch)
        out_seq_to_2 = self.M(self.batch[:, :2])
        self.assertEqual(out_seq_to_2.last_hidden_state, out.last_hidden_state[:, :2])

    def test_forward_identical_with_or_without_caching(self):
        # We want to check that the output doesn't change when we do or do not use caching. To do this, we'll
        # run the model over a partial batch without caching and store the result. Then, we'll run the model
        # over various elements of that batch, iterating through in sequence, using caching to only ever run
        # the attention calculation on the last element, and we'll validate that the predictions don't change
        # in comparison to the run without caching.

        out_no_caching = self.M(self.batch, return_dict=True, use_cache=False)

        out_full_caching = self.M(self.batch, return_dict=True, use_cache=True)
        full_seq_past_up_to_2 = tuple(
            tuple(e[:, :, :2, :] for e in ee) for ee in out_full_caching["past_key_values"]["seq_past"]
        )

        out_full_caching["past_key_values"] = None
        self.assertEqual(out_no_caching, out_full_caching)

        source_batch_for_slicing = PytorchBatch(**copy.deepcopy(BASE_BATCH))
        # We need to add time explicitly here as that will be lost when we slice the batch. This happens
        # naturally during generation.
        source_batch_for_slicing.time = time_from_deltas(source_batch_for_slicing)

        # We can't actually start an iterative caching process from nothing -- the system isn't designed for
        # that; Instead, we'll run the first sequence element outside the iterative selection and capture it's
        # output there, then use that as the starting point for the iterative cache-based computation.

        seq_attention_mask = expand_mask(self.batch.event_mask, out_full_caching.last_hidden_state.dtype)

        seq_idx = 1
        dep_graph_idx = None

        sliced_batch = copy.deepcopy(source_batch_for_slicing)
        sliced_batch = sliced_batch[:, : (seq_idx + 1)]

        sliced_out = self.M(
            sliced_batch,
            return_dict=True,
            use_cache=True,
            dep_graph_past=dep_graph_idx,
            dep_graph_el_generation_target=None,
            seq_attention_mask=seq_attention_mask[:, :, :, : (seq_idx + 1)],
            past=None,
        )

        self.assertEqual(
            out_no_caching.last_hidden_state[:, :2],
            sliced_out.last_hidden_state,
            "The initial slicing shouldn't impact the last hidden state.",
        )

        new_joint_past = sliced_out["past_key_values"]
        sliced_out["past_key_values"] = None

        self.assertNestedEqual(full_seq_past_up_to_2, new_joint_past["seq_past"])

        past = new_joint_past["seq_past"]
        dep_graph_past = new_joint_past["dep_graph_past"]

        out_iterative_caching = [sliced_out]
        for seq_idx in range(2, self.batch.sequence_length):
            sliced_batch = copy.deepcopy(source_batch_for_slicing)
            sliced_batch = sliced_batch[:, seq_idx : seq_idx + 1]

            out_iterative_caching_seq = []
            for dep_graph_idx in [1, 2, 0]:
                sliced_out = self.M(
                    sliced_batch,
                    return_dict=True,
                    use_cache=True,
                    dep_graph_past=dep_graph_past,
                    dep_graph_el_generation_target=dep_graph_idx,
                    seq_attention_mask=seq_attention_mask[:, :, :, : (seq_idx + 1)],
                    past=past,
                )

                new_joint_past = sliced_out["past_key_values"]
                sliced_out["past_key_values"] = None

                past = new_joint_past["seq_past"]
                dep_graph_past = new_joint_past["dep_graph_past"]

                out_iterative_caching_seq.append(sliced_out)

            joint_seq_out_iterative_caching = TransformerOutputWithPast(
                last_hidden_state=torch.cat([x.last_hidden_state for x in out_iterative_caching_seq], dim=2),
            )
            out_iterative_caching.append(joint_seq_out_iterative_caching)

        out_iterative_caching = TransformerOutputWithPast(
            last_hidden_state=torch.cat([x.last_hidden_state for x in out_iterative_caching], dim=1),
        )

        self.assertEqual(out_no_caching, out_iterative_caching)


if __name__ == "__main__":
    unittest.main()

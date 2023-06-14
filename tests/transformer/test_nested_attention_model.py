import sys

sys.path.append("../..")

import copy
import unittest
from unittest.mock import MagicMock, call

import lightning as L
import torch

from EventStream.data.config import MeasurementConfig
from EventStream.data.types import DataModality, PytorchBatch, TemporalityType
from EventStream.transformer.config import (
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
)
from EventStream.transformer.model_output import (
    GenerativeSequenceModelLabels,
    GenerativeSequenceModelLosses,
    GenerativeSequenceModelOutput,
    GenerativeSequenceModelPredictions,
)
from EventStream.transformer.nested_attention_model import (
    NAPPTForGenerativeSequenceModeling,
    NestedAttentionGenerativeOutputLayer,
)
from EventStream.transformer.transformer import expand_mask

from ..utils import ConfigComparisonsMixin, MockModule

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

CI_CONFIG_KWARGS = dict(
    structured_event_processing_mode=StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT,
    dep_graph_window_size=None,
    do_full_block_in_dep_graph_attention=None,
    do_full_block_in_seq_attention=None,
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
    "event_mask": torch.BoolTensor([[True, True, True, True], [False, True, True, True]]),
    "time_delta": torch.FloatTensor([[0, 2, 5, 3], [0, 3, 2, 3]]),
    "start_time": torch.FloatTensor([1.0, 1412.0]),
    "static_indices": torch.LongTensor([[1, 2, 3], [1, 3, 0]]),
    "static_measurement_indices": torch.LongTensor([[1, 2, 3], [1, 3, 0]]),
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


class TestNestedAttentionGenerativeOutputLayer(ConfigComparisonsMixin, unittest.TestCase):
    def test_constructs(self):
        NestedAttentionGenerativeOutputLayer(StructuredTransformerConfig(**NA_CONFIG_KWARGS))

        with self.assertRaises(ValueError):
            NestedAttentionGenerativeOutputLayer(StructuredTransformerConfig(**CI_CONFIG_KWARGS))

    def test_e2e(self):
        dummy_batch = {
            "event_mask": "event_mask",
            "dynamic_values_mask": "dynamic_values_mask",
        }

        classification_measures = ["clf1", "clf2", "clf3"]
        clf_losses_by_measurement = {"clf1": 1, "clf2": 2, "clf3": 3}
        clf_dists_by_measurement = {"clf1": 4, "clf2": 5, "clf3": 6}
        clf_labels_by_measurement = {"clf1": 7, "clf2": 8, "clf3": 9}
        default_classification_out = (
            clf_losses_by_measurement,
            clf_dists_by_measurement,
            clf_labels_by_measurement,
        )
        full_clf_loss = 6

        multivariate_regression_measures = ["mr1", "mr2"]
        univariate_regression_measures = ["ur1", "ur2"]
        regression_loss_values = {"mr1": 1, "mr2": 2, "ur1": 3, "ur2": 4}
        regression_dists = {"mr1": 5, "mr2": 6, "ur1": 7, "ur2": 8}
        regression_labels = {"mr1": 9, "mr2": 10, "ur1": 11, "ur2": 12}
        regression_indices = {"mr1": 13, "mr2": 14, "ur1": 15, "ur2": 16}
        default_regression_out = (
            regression_loss_values,
            regression_dists,
            regression_labels,
            regression_indices,
        )
        full_regression_loss = 10

        TTE_LL_overall = -1
        TTE_dist = 2
        TTE_true = 3
        default_TTE_out = (TTE_LL_overall, TTE_dist, TTE_true)

        full_out = {
            "loss": full_clf_loss + full_regression_loss - TTE_LL_overall,
            "losses": GenerativeSequenceModelLosses(
                classification=clf_losses_by_measurement,
                regression=regression_loss_values,
                time_to_event=-TTE_LL_overall,
            ),
            "preds": GenerativeSequenceModelPredictions(
                classification=clf_dists_by_measurement,
                regression=regression_dists,
                regression_indices=regression_indices,
                time_to_event=TTE_dist,
            ),
            "labels": GenerativeSequenceModelLabels(
                classification=clf_labels_by_measurement,
                regression=regression_labels,
                regression_indices=regression_indices,
                time_to_event=TTE_true,
            ),
            "event_mask": "event_mask",
            "dynamic_values_mask": "dynamic_values_mask",
        }

        no_TTE_gen_out = {
            "loss": None,
            "losses": GenerativeSequenceModelLosses(
                classification=None,
                regression=None,
                time_to_event=None,
            ),
            "preds": GenerativeSequenceModelPredictions(
                classification=clf_dists_by_measurement,
                regression=regression_dists,
                regression_indices=None,
                time_to_event=None,
            ),
            "labels": GenerativeSequenceModelLabels(
                classification=None,
                regression=None,
                regression_indices=None,
                time_to_event=None,
            ),
            "event_mask": "event_mask",
            "dynamic_values_mask": "dynamic_values_mask",
        }

        TTE_gen_out = {
            "loss": None,
            "losses": GenerativeSequenceModelLosses(
                classification=None,
                regression=None,
                time_to_event=None,
            ),
            "preds": GenerativeSequenceModelPredictions(
                classification={},
                regression={},
                regression_indices=None,
                time_to_event=TTE_dist,
            ),
            "labels": GenerativeSequenceModelLabels(
                classification=None,
                regression=None,
                regression_indices=None,
                time_to_event=None,
            ),
            "event_mask": "event_mask",
            "dynamic_values_mask": "dynamic_values_mask",
        }

        # bsz, seq_len, dep_graph_len, _ = encoded.shape
        default_encoded = torch.FloatTensor(
            [
                [
                    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 0, 0, 0]],
                    [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24], [0, 0, 0, 0]],
                ],
                [
                    [[25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36], [0, 0, 0, 0]],
                    [[37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48], [0, 0, 0, 0]],
                ],
            ]
        )

        cases = [
            {
                "msg": "Should error if given a dep_graph_el_generation_target in non-generative mode.",
                "is_generation": False,
                "dep_graph_el_generation_target": 1,
                "should_raise": ValueError,
            },
            {
                "msg": "Should error if given an invalid measurement mode.",
                "is_generation": False,
                "dep_graph_el_generation_target": None,
                "should_raise": ValueError,
                "measurements_per_dep_graph_level": [
                    [],
                    ["clf1", ["mr1", "categorical_only"]],
                    ["clf2", "ur1", ["mr1", "invalid"]],
                    ["clf3", "mr2", "ur2"],
                ],
            },
            {
                "msg": "Should work in non-generative mode.",
                "is_generation": False,
                "dep_graph_el_generation_target": None,
                "want": full_out,
                "classification_calls": [
                    call(
                        dummy_batch,
                        default_encoded[:, :, 0, :],
                        {"clf1", "mr1"},
                    ),
                    call(
                        dummy_batch,
                        default_encoded[:, :, 1, :],
                        {"clf2"},
                    ),
                    call(
                        dummy_batch,
                        default_encoded[:, :, 2, :],
                        {"clf3", "mr2"},
                    ),
                ],
                "regression_calls": [
                    call(
                        dummy_batch,
                        default_encoded[:, :, 0, :],
                        set(),
                        is_generation=False,
                    ),
                    call(
                        dummy_batch,
                        default_encoded[:, :, 1, :],
                        {"mr1", "ur1"},
                        is_generation=False,
                    ),
                    call(
                        dummy_batch,
                        default_encoded[:, :, 2, :],
                        {"mr2", "ur2"},
                        is_generation=False,
                    ),
                ],
                "TTE_calls": [call(dummy_batch, default_encoded[:, :, -1, :], is_generation=False)],
            },
            {
                "msg": "Should work in generative mode with dep_graph_el_generation_target > 0.",
                "is_generation": True,
                "dep_graph_el_generation_target": 2,
                # This doesn't have to be accurate; it isn't used as we mock everything.
                "encoded": default_encoded[:, :, 0, :].unsqueeze(2),
                "want": no_TTE_gen_out,
                "classification_calls": [
                    call(
                        dummy_batch,
                        default_encoded[:, :, 0, :],
                        {"clf2"},
                    ),
                ],
                "regression_calls": [
                    call(
                        dummy_batch,
                        default_encoded[:, :, 0, :],
                        {"mr1", "ur1"},
                        is_generation=True,
                    ),
                ],
                "TTE_calls": [],
            },
            {
                "msg": "Should work in generative mode with dep_graph_el_generation_target=0.",
                "is_generation": True,
                "dep_graph_el_generation_target": 0,
                "want": TTE_gen_out,
                "classification_calls": [],
                "regression_calls": [],
                "TTE_calls": [call(dummy_batch, default_encoded[:, :, -1, :], is_generation=True)],
            },
        ]

        for case in cases:
            with self.subTest(msg=case["msg"]):
                M = NestedAttentionGenerativeOutputLayer(StructuredTransformerConfig(**NA_CONFIG_KWARGS))

                M.classification_mode_per_measurement = {
                    m: None for m in classification_measures + multivariate_regression_measures
                }
                if "measurements_per_dep_graph_level" in case:
                    M.config.measurements_per_dep_graph_level = case["measurements_per_dep_graph_level"]
                else:
                    M.config.measurements_per_dep_graph_level = [
                        [],
                        ["clf1", ["mr1", "categorical_only"]],
                        ["clf2", "ur1", ["mr1", "numerical_only"]],
                        ["clf3", "mr2", "ur2"],
                    ]
                M.config.measurements_per_generative_mode = {
                    DataModality.MULTIVARIATE_REGRESSION: multivariate_regression_measures,
                    DataModality.UNIVARIATE_REGRESSION: univariate_regression_measures,
                }

                M.get_classification_outputs = MagicMock(return_value=default_classification_out)
                M.get_regression_outputs = MagicMock(return_value=default_regression_out)
                M.get_TTE_outputs = MagicMock(return_value=default_TTE_out)

                kwargs = dict(
                    batch=dummy_batch,
                    encoded=case.get("encoded", default_encoded),
                    is_generation=case["is_generation"],
                    dep_graph_el_generation_target=case["dep_graph_el_generation_target"],
                )

                should_raise = case.get("should_raise", None)
                if should_raise is not None:
                    with self.assertRaises(should_raise):
                        M(**kwargs)
                else:
                    got = M(**kwargs)
                    want = GenerativeSequenceModelOutput(**case["want"])
                    self.assertEqual(want, got)

                    classification_calls = case.get("classification_calls", [])
                    self.assertNestedCalledWith(M.get_classification_outputs, classification_calls)

                    regression_calls = case.get("regression_calls", [])
                    self.assertNestedCalledWith(M.get_regression_outputs, regression_calls)

                    TTE_calls = case.get("TTE_calls", [])
                    self.assertNestedCalledWith(M.get_TTE_outputs, TTE_calls)


class TestNAPPTForGenerativeSequenceModeling(ConfigComparisonsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.config = StructuredTransformerConfig(**NA_CONFIG_KWARGS)

        self.M = NAPPTForGenerativeSequenceModeling(self.config).cpu()
        self.M.eval()  # So layernorm and dropout don't affect anything.

        self.batch = PytorchBatch(**copy.deepcopy(BASE_BATCH))

    def test_constructs(self):
        with self.assertRaises(ValueError):
            NAPPTForGenerativeSequenceModeling(StructuredTransformerConfig(**CI_CONFIG_KWARGS))

    def test_prepare_inputs_for_generation(self):
        default_batch = PytorchBatch(
            time_delta=torch.Tensor([[1.0, 2.0, 3.0]]),
            event_mask=torch.BoolTensor([[True, False, True]]),
            dynamic_indices=torch.LongTensor([[[[0, 1, 2]], [[3, 4, 5]], [[6, 7, 8]]]]),
            dynamic_measurement_indices=torch.LongTensor([[[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]]]),
            dynamic_values=torch.FloatTensor([[[[0, 1, 2]], [[3, 4, 5]], [[6, 7, 8]]]]),
            dynamic_values_mask=torch.BoolTensor(
                [[[[True, True, False]], [[True, False, True]], [[False, True, True]]]]
            ),
        )

        default_attention_mask = expand_mask(default_batch.event_mask, default_batch.dynamic_values.dtype)

        unsqueezed_batch_with_time = PytorchBatch(
            time_delta=torch.Tensor([[3.0]]),
            time=torch.Tensor([[1.0]]),
            event_mask=torch.BoolTensor([[True]]),
            dynamic_indices=torch.LongTensor([[[[6, 7, 8]]]]),
            dynamic_measurement_indices=torch.LongTensor([[[[7, 8, 9]]]]),
            dynamic_values=torch.FloatTensor([[[[6, 7, 8]]]]),
            dynamic_values_mask=torch.BoolTensor([[[[False, True, True]]]]),
        )

        cases = [
            {
                "msg": "Should work with use_cache=False.",
                "want": {"batch": default_batch},
            },
            {
                "msg": "Should return extra kwargs with use_cache=False.",
                "kwargs": {"test": True},
                "want": {"batch": default_batch, "test": True},
            },
            {
                "msg": "Should error if past is None and dep_graph_el_generation_target > 0.",
                "use_cache": True,
                "past": None,
                "dep_graph_el_generation_target": 1,
                "should_raise": ValueError,
            },
            {
                "msg": "Should return no past or dep_graph_past and not modify batch if past is None.",
                "use_cache": True,
                "past": None,
                "dep_graph_el_generation_target": None,
                "want": {
                    "seq_attention_mask": default_attention_mask,
                    "dep_graph_el_generation_target": None,
                    "batch": default_batch,
                    "past": None,
                    "dep_graph_past": None,
                    "use_cache": True,
                },
            },
            {
                "msg": "Should raise ValueError if past is not a dict or None.",
                "use_cache": True,
                "past": "not a dict",
                "should_raise": ValueError,
            },
            {
                "msg": "Should raise ValueError if past doesn't have the right keys",
                "use_cache": True,
                "past": {"wrong key": True},
                "should_raise": ValueError,
            },
            {
                "msg": (
                    "Should raise ValueError if dep_graph_el_generation_target is None and dep_graph_past "
                    "is not None."
                ),
                "use_cache": True,
                "past": {"seq_past": 1, "dep_graph_past": 2},
                "dep_graph_el_generation_target": None,
                "should_raise": ValueError,
            },
            {
                "msg": (
                    "Should strip batch down to last sequence element and return no dep_graph_past if "
                    "dep_graph_el_generation_target is None and dep_graph_past is None"
                ),
                "use_cache": True,
                "past": {"seq_past": 1, "dep_graph_past": None},
                "dep_graph_el_generation_target": None,
                "want": {
                    "seq_attention_mask": default_attention_mask,
                    "batch": unsqueezed_batch_with_time,
                    "past": 1,
                    "dep_graph_past": None,
                    "use_cache": True,
                    "dep_graph_el_generation_target": None,
                },
            },
            {
                "msg": (
                    "Should strip batch down to last sequence element and return dep_graph_past if "
                    "dep_graph_el_generation_target is <= 1 and dep_graph_past is not None"
                ),
                "use_cache": True,
                "past": {"seq_past": 1, "dep_graph_past": 2},
                "dep_graph_el_generation_target": 1,
                "want": {
                    "batch": unsqueezed_batch_with_time,
                    "past": 1,
                    "dep_graph_past": 2,
                    "use_cache": True,
                    "dep_graph_el_generation_target": 1,
                    "seq_attention_mask": default_attention_mask,
                },
            },
            {
                "msg": (
                    "Should strip batch down to last sequence element and return dep_graph_past if "
                    "dep_graph_el_generation_target is > 1 and dep_graph_past is not None"
                ),
                "use_cache": True,
                "past": {"seq_past": 1, "dep_graph_past": 2},
                "dep_graph_el_generation_target": 2,
                "want": {
                    "batch": unsqueezed_batch_with_time,
                    "past": 1,
                    "dep_graph_past": 2,
                    "use_cache": True,
                    "dep_graph_el_generation_target": 2,
                    "seq_attention_mask": default_attention_mask,
                },
            },
            {
                "msg": "Should error if dep_graph_el_generation_target is > 1 and dep_graph_past is None",
                "use_cache": True,
                "past": {"seq_past": 1, "dep_graph_past": None},
                "dep_graph_el_generation_target": 2,
                "should_raise": ValueError,
            },
        ]

        for case in cases:
            with self.subTest(msg=case["msg"]):
                kwargs = dict(
                    batch=default_batch,
                    **case.get("kwargs", {}),
                )

                if "past" in case:
                    kwargs["past"] = case["past"]
                if "dep_graph_el_generation_target" in case:
                    kwargs["dep_graph_el_generation_target"] = case["dep_graph_el_generation_target"]
                if "use_cache" in case:
                    kwargs["use_cache"] = case["use_cache"]

                should_raise = case.get("should_raise", None)
                if should_raise is not None:
                    with self.assertRaises(should_raise):
                        self.M.prepare_inputs_for_generation(**kwargs)
                else:
                    got = self.M.prepare_inputs_for_generation(**kwargs)
                    want = case["want"]
                    self.assertNestedDictEqual(want, got)

    def test_forward(self):
        cases = [
            {
                "msg": "Should return no extra arguments if not indicated",
                "batch": {"input": True},
                "want": {"input": True},
            },
            {
                "msg": "Should return past_key_values if use_cache is indicated",
                "batch": {"input": True},
                "want": {"input": True, "past_key_values": "past_key_values"},
                "kwargs": {"use_cache": True},
            },
            {
                "msg": "Should return attentions if indicated",
                "batch": {"input": True},
                "want": {"input": True, "attentions": "attentions"},
                "kwargs": {"output_attentions": True},
            },
            {
                "msg": "Should return hidden_states if indicated",
                "batch": {"input": True},
                "want": {"input": True, "hidden_states": "hidden_states"},
                "kwargs": {"output_hidden_states": True},
            },
        ]

        for case in cases:
            with self.subTest(msg=case["msg"]):
                encoded_mock = MagicMock()
                self.M.encoder = MockModule(return_value=encoded_mock)
                encoded_mock.last_hidden_state = "last_hidden_state"
                encoded_mock.past_key_values = "past_key_values"
                encoded_mock.attentions = "attentions"
                encoded_mock.hidden_states = "hidden_states"
                self.M.output_layer = MockModule(side_effect=lambda batch, *args, **kwargs: batch)

                batch = case["batch"]
                kwargs = case.get("kwargs", {})
                is_generation = kwargs.get("is_generation", False)

                want = case["want"]
                got = self.M(batch=batch, is_generation=is_generation, **kwargs)
                self.assertNestedDictEqual(want, got)

                self.M.encoder.assert_called_once_with(batch, **kwargs)
                self.M.output_layer.assert_called_once_with(
                    batch,
                    "last_hidden_state",
                    is_generation=is_generation,
                    dep_graph_el_generation_target=kwargs.get("dep_graph_el_generation_target", None),
                )

    def test_generation_seed_dependent(self):
        generation_kwargs = dict(
            max_new_events=5,
            num_return_sequences=2,
            do_sample=True,
            return_dict_in_generate=False,
            output_scores=False,
            output_attentions=False,
            output_hidden_states=False,
        )

        L.seed_everything(1)
        out_1_seed_1 = self.M.generate(self.batch, **generation_kwargs, use_cache=False)

        L.seed_everything(1)
        out_2_seed_1 = self.M.generate(self.batch, **generation_kwargs, use_cache=False)
        self.assertEqual(out_1_seed_1, out_2_seed_1)

        L.seed_everything(2)
        out_1_seed_2 = self.M.generate(self.batch, **generation_kwargs, use_cache=False)

        L.seed_everything(2)
        out_2_seed_2 = self.M.generate(self.batch, **generation_kwargs, use_cache=False)
        self.assertEqual(out_1_seed_2, out_2_seed_2)

        # We use an assertRaises here as assertEqual relies on our custom type dependent assertions, and
        # assertNotEquals doesn't.
        with self.assertRaises(AssertionError):
            self.assertEqual(out_1_seed_1, out_1_seed_2)

    def test_generation_shapes(self):
        num_return_sequences = 2
        max_new_events = 5
        generation_kwargs = dict(
            max_new_events=max_new_events,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            return_dict_in_generate=False,
            output_scores=False,
            output_attentions=False,
            output_hidden_states=False,
        )

        input_seq_length = self.batch.sequence_length
        input_batch_size = self.batch.batch_size
        input_n_data_elements = self.batch.n_data_elements

        out = self.M.generate(self.batch, **generation_kwargs, use_cache=False)
        self.assertEqual(input_batch_size * num_return_sequences, out.batch_size)
        self.assertEqual(input_seq_length + max_new_events, out.sequence_length)
        self.assertTrue(out.n_data_elements >= input_n_data_elements)

        for i in range(input_batch_size):
            want_time_delta = self.batch[i].time_delta
            got_time_delta = out[i * num_return_sequences, :input_seq_length].time_delta

            self.assertEqual(want_time_delta[:-1], got_time_delta[:-1])

        for i in range(input_batch_size):
            self.batch[i].time_delta *= 0
            out.time_delta *= 0
            self.assertEqual(
                self.batch[i],
                out[i * num_return_sequences, :input_seq_length, :input_n_data_elements],
            )

    def test_generation_identical_with_or_without_caching(self):
        # We want to check that the output doesn't change when we do or do not use caching. To do this, we'll
        # run the model over a partial batch without caching and store the result. Then, we'll run the model
        # over various elements of that batch, iterating through in sequence, using caching to only ever run
        # the attention calculation on the last element, and we'll validate that the predictions don't change
        # in comparison to the run without caching.

        generation_kwargs = dict(
            max_new_events=15,
            num_return_sequences=3,
            do_sample=True,
            return_dict_in_generate=False,
            output_scores=False,
            output_attentions=False,
            output_hidden_states=True,
        )

        L.seed_everything(1)
        out_no_caching = self.M.generate(self.batch, **generation_kwargs, use_cache=False)

        L.seed_everything(1)
        out_with_caching = self.M.generate(self.batch, **generation_kwargs, use_cache=True)

        self.assertEqual(out_no_caching, out_with_caching)


if __name__ == "__main__":
    unittest.main()

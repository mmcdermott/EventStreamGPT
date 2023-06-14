import sys

sys.path.append("../..")

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from EventStream.data.types import DataModality
from EventStream.transformer.config import (
    Averaging,
    MetricCategories,
    Metrics,
    MetricsConfig,
    OptimizationConfig,
    Split,
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
    TimeToEventGenerationHeadType,
)

from ..utils import ConfigComparisonsMixin


class TestMetricsConfig(unittest.TestCase):
    def test_do_skip_all_metrics(self):
        MC = MetricsConfig(do_skip_all_metrics=True)
        self.assertEqual(MC.include_metrics, {})
        for split in Split.values():
            self.assertTrue(MC.do_log_only_loss(split))

    def test_do_log_only_loss(self):
        split = Split.TUNING

        cases = [
            {
                "msg": "Should log only loss when split is not present in include_metrics",
                "include_metrics": {Split.HELD_OUT: True},
                "want": True,
            },
            {
                "msg": "Should log only loss when split is present in include_metrics but is empty",
                "include_metrics": {split: {}},
                "want": True,
            },
            {
                "msg": (
                    "Should log only loss when split is present in include_metrics but only contains "
                    "loss_parts."
                ),
                "include_metrics": {split: {MetricCategories.LOSS_PARTS: True}},
                "want": True,
            },
            {
                "msg": (
                    "Should not log only loss when split is present in include_metrics and contains other "
                    "metrics"
                ),
                "include_metrics": {split: {MetricCategories.TTE: {Metrics.MSE: True}}},
                "want": False,
            },
        ]

        for case in cases:
            with self.subTest(msg=case["msg"]):
                MC = MetricsConfig(
                    do_skip_all_metrics=False,
                    include_metrics=case["include_metrics"],
                )
                self.assertEqual(MC.do_log_only_loss(split), case["want"])

    def test_do_log(self):
        split = Split.TUNING

        cases = [
            {
                "msg": "Should not log when do_log_only_loss(split) is True",
                "do_log_only_loss": True,
                "want": False,
                "cat": MetricCategories.CLASSIFICATION,
                "include_metrics": {},
                "metric_name": None,
            },
            {
                "msg": "Should not log when cat is not present in include_metrics",
                "cat": MetricCategories.CLASSIFICATION,
                "include_metrics": {split: {MetricCategories.REGRESSION: True}},
                "want": False,
                "metric_name": None,
            },
            {
                "msg": "Should not log when cat is present in include_metrics but is empty",
                "cat": MetricCategories.CLASSIFICATION,
                "include_metrics": {split: {MetricCategories.CLASSIFICATION: {}}},
                "want": False,
                "metric_name": None,
            },
            {
                "msg": "Should log when metric_name is None",
                "cat": MetricCategories.CLASSIFICATION,
                "include_metrics": {split: {MetricCategories.CLASSIFICATION: {Metrics.AUROC: True}}},
                "metric_name": None,
                "want": True,
            },
            {
                "msg": "Should log when include_metrics is True",
                "cat": MetricCategories.CLASSIFICATION,
                "include_metrics": {split: {MetricCategories.CLASSIFICATION: True}},
                "metric_name": f"{Averaging.WEIGHTED}_{Metrics.AUROC}",
                "want": True,
            },
            {
                "msg": "Should log when metric is included and all averagings are included.",
                "cat": MetricCategories.CLASSIFICATION,
                "include_metrics": {split: {MetricCategories.CLASSIFICATION: {Metrics.AUROC: True}}},
                "metric_name": f"{Averaging.WEIGHTED}_{Metrics.AUROC}",
                "want": True,
            },
            {
                "msg": "Should not log when metric is not included even if all averagings are included.",
                "cat": MetricCategories.CLASSIFICATION,
                "include_metrics": {split: {MetricCategories.CLASSIFICATION: {Metrics.AUROC: True}}},
                "metric_name": f"{Averaging.WEIGHTED}_{Metrics.AUPRC}",
                "want": False,
            },
            {
                "msg": "Should log when metric and averagings are included.",
                "cat": MetricCategories.CLASSIFICATION,
                "include_metrics": {
                    split: {MetricCategories.CLASSIFICATION: {Metrics.AUROC: [Averaging.WEIGHTED]}}
                },
                "metric_name": f"{Averaging.WEIGHTED}_{Metrics.AUROC}",
                "want": True,
            },
            {
                "msg": "Should not log when metric is included if averaging is not included.",
                "cat": MetricCategories.CLASSIFICATION,
                "include_metrics": {
                    split: {MetricCategories.CLASSIFICATION: {Metrics.AUROC: [Averaging.MICRO]}}
                },
                "metric_name": f"{Averaging.WEIGHTED}_{Metrics.AUROC}",
                "want": False,
            },
            {
                "msg": "Should not log when metric is not included even if it has no averaging.",
                "cat": MetricCategories.CLASSIFICATION,
                "include_metrics": {
                    split: {MetricCategories.CLASSIFICATION: {Metrics.AUROC: [Averaging.MICRO]}}
                },
                "metric_name": Metrics.AUPRC,
                "want": False,
            },
        ]

        for case in cases:
            with self.subTest(msg=case["msg"]):
                MC = MetricsConfig(do_skip_all_metrics=False, include_metrics=case["include_metrics"])
                if case.get("do_log_only_loss", False):
                    MC.do_log_only_loss = MagicMock(return_value=True)
                else:
                    MC.do_log_only_loss = MagicMock(return_value=False)
                self.assertEqual(MC.do_log(split, case["cat"], case["metric_name"]), case["want"])


DEFAULT_OPT_CONFIG_DICT = dict(
    init_lr=1e-2,
    end_lr=1e-7,
    end_lr_frac_of_init_lr=None,
    max_epochs=2,
    batch_size=32,
    lr_frac_warmup_steps=0.01,
    lr_num_warmup_steps=None,
    max_training_steps=None,
    lr_decay_power=1.0,
    weight_decay=0.01,
)


class TestOptimizationConfig(unittest.TestCase):
    def test_set_to_dataset(self):
        cfg = OptimizationConfig(
            **{
                **DEFAULT_OPT_CONFIG_DICT,
                "max_epochs": 10,
                "batch_size": 2,
                "lr_frac_warmup_steps": 6 / 30,
            }
        )

        pyd = MagicMock()
        pyd.__len__.return_value = 6

        cfg.set_to_dataset(pyd)

        self.assertEqual(10 * 3, cfg.max_training_steps)
        self.assertEqual(6, cfg.lr_num_warmup_steps)


DEFAULT_MAIN_CONFIG_DICT = dict(
    vocab_sizes_by_measurement=None,
    vocab_offsets_by_measurement=None,
    measurements_idxmap=None,
    measurements_per_generative_mode=None,
    measurements_per_dep_graph_level=None,
    max_seq_len=256,
    hidden_size=256,
    head_dim=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    seq_attention_types=None,
    seq_window_size=32,
    intermediate_size=32,
    activation_function="gelu",
    attention_dropout=0.1,
    input_dropout=0.1,
    resid_dropout=0.1,
    init_std=0.02,
    layer_norm_epsilon=1e-5,
    use_cache=True,
)

DEFAULT_NESTED_ATTENTION_DICT = dict(
    structured_event_processing_mode=StructuredEventProcessingMode.NESTED_ATTENTION,
    dep_graph_attention_types=None,
    dep_graph_window_size=2,
    do_full_block_in_dep_graph_attention=True,
    do_full_block_in_seq_attention=False,
    measurements_per_dep_graph_level=[],
)

DEFAULT_CONDITIONALLY_INDEPENDENT_DICT = dict(
    structured_event_processing_mode=StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT,
    dep_graph_attention_types=None,
    dep_graph_window_size=None,
    do_full_block_in_dep_graph_attention=None,
    do_full_block_in_seq_attention=None,
)

DEFAULT_EXPONENTIAL_DICT = dict(
    TTE_generation_layer_type=TimeToEventGenerationHeadType.EXPONENTIAL,
    TTE_lognormal_generation_num_components=None,
    mean_log_inter_event_time_min=None,
    std_log_inter_event_time_min=None,
)

DEFAULT_LOGNORMAL_MIXTURE_DICT = dict(
    TTE_generation_layer_type=TimeToEventGenerationHeadType.LOG_NORMAL_MIXTURE,
    TTE_lognormal_generation_num_components=3,
    mean_log_inter_event_time_min=None,
    std_log_inter_event_time_min=1.0,
)


class TestStructuredTransformerConfig(ConfigComparisonsMixin, unittest.TestCase):
    def test_construction(self):
        """Tests the construction and initialization logic of the `StructuredTransformerConfig` object."""

        cases = [
            {
                "msg": "Should construct with default args.",
                "kwargs": {},
            },
            {
                "msg": "Should construct with nested_attention args.",
                "kwargs": {**DEFAULT_NESTED_ATTENTION_DICT},
            },
            {
                "msg": "Should Error when head_dim and hidden_size are missing.",
                "kwargs": {"head_dim": 32, "hidden_size": 64, "num_attention_heads": 4},
                "should_raise": ValueError,
            },
            {
                "msg": "Should Error when head_dim and hidden_size are inconsistent.",
                "kwargs": {"head_dim": None, "hidden_size": None},
                "should_raise": ValueError,
            },
            {
                "msg": "Should Error when num_hidden_layers is not an int.",
                "kwargs": {**DEFAULT_NESTED_ATTENTION_DICT, "num_hidden_layers": 4.0},
                "should_raise": TypeError,
            },
            {
                "msg": "Should Error when num_hidden_layers is negative.",
                "kwargs": {"num_hidden_layers": -4},
                "should_raise": ValueError,
            },
            {
                "msg": "Should Error when nested_attention args are missing or invalid.",
                "kwargs": [
                    {
                        **DEFAULT_NESTED_ATTENTION_DICT,
                        "do_full_block_in_dep_graph_attention": None,
                    },
                    {
                        **DEFAULT_NESTED_ATTENTION_DICT,
                        "do_full_block_in_seq_attention": None,
                    },
                ],
                "should_raise": ValueError,
            },
            {
                "msg": "Should construct with conditionally_independent args.",
                "kwargs": {**DEFAULT_CONDITIONALLY_INDEPENDENT_DICT},
            },
            {
                "msg": "Should construct with Exponential TTE head args.",
                "kwargs": {**DEFAULT_EXPONENTIAL_DICT},
            },
            {
                "msg": "Should construct with Lognormal Mixture TTE head args.",
                "kwargs": {**DEFAULT_LOGNORMAL_MIXTURE_DICT},
            },
            {
                "msg": "Should error when required Lognormal Mixture args are missing or invalid.",
                "kwargs": [
                    {
                        **DEFAULT_LOGNORMAL_MIXTURE_DICT,
                        "TTE_lognormal_generation_num_components": None,
                    },
                    {
                        **DEFAULT_LOGNORMAL_MIXTURE_DICT,
                        "TTE_lognormal_generation_num_components": -1,
                    },
                ],
                "should_raise": ValueError,
            },
            {
                "msg": "Should error when required Lognormal Mixture arg is the wrong type.",
                "kwargs": {
                    **DEFAULT_LOGNORMAL_MIXTURE_DICT,
                    "TTE_lognormal_generation_num_components": 1.0,
                },
                "should_raise": TypeError,
            },
            {
                "msg": "Should error when is specified as encoder_decoder.",
                "kwargs": {"is_encoder_decoder": True},
                "should_raise": AssertionError,
            },
        ]

        for C in cases:
            with self.subTest(C["msg"]):
                kwargs = C["kwargs"] if type(C["kwargs"]) is list else [C["kwargs"]]
                for args_dict in kwargs:
                    if "should_raise" in C:
                        with self.assertRaises(C["should_raise"]):
                            StructuredTransformerConfig(**args_dict)
                    else:
                        StructuredTransformerConfig(**args_dict)

    def test_set_to_dataset(self):
        default_measurements_per_generative_mode = {
            DataModality.SINGLE_LABEL_CLASSIFICATION: ["event_type"],
            DataModality.MULTI_LABEL_CLASSIFICATION: ["B_key", "A_col"],
            DataModality.MULTIVARIATE_REGRESSION: ["B_key"],
        }
        default_measurements_idxmap = {"event_type": 1, "B_key": 2, "A_col": 3}
        default_measurement_vocab_offsets = {"event_type": 1, "B_key": 2, "A_col": 5}
        default_vocab_sizes_by_measurement = {"event_type": 2, "B_key": 3, "A_col": 4}

        default_pyd_spec = {
            "has_task": False,
            "max_seq_len": 4,
            "vocabulary_config": {
                "measurements_idxmap": default_measurements_idxmap,
                "vocab_offsets_by_measurement": default_measurement_vocab_offsets,
                "vocab_sizes_by_measurement": default_vocab_sizes_by_measurement,
                "measurements_per_generative_mode": default_measurements_per_generative_mode,
                "total_vocab_size": 9,
            },
        }

        default_want = {
            "max_seq_len": 4,
            "vocab_size": 9,
            "vocab_sizes_by_measurement": default_vocab_sizes_by_measurement,
            "measurements_idxmap": default_measurements_idxmap,
            "vocab_offsets_by_measurement": default_measurement_vocab_offsets,
            "measurements_per_generative_mode": default_measurements_per_generative_mode,
            "finetuning_task": None,
            "problem_type": None,
        }

        cases = [
            {
                "msg": "Should set appropriate with no task_df.",
                "pyd_spec": default_pyd_spec,
                "want": default_want,
            },
            {
                "msg": "Should set appropriate regression task_df.",
                "pyd_spec": {
                    **default_pyd_spec,
                    "has_task": True,
                    "task_types": {"task": "regression"},
                    "tasks": ["task"],
                },
                "want": {
                    **default_want,
                    "finetuning_task": "task",
                    "problem_type": "regression",
                    "num_labels": 1,
                },
            },
            {
                "msg": "Should set appropriate multi_class_classification task_df.",
                "pyd_spec": {
                    **default_pyd_spec,
                    "has_task": True,
                    "task_types": {"task": "multi_class_classification"},
                    "task_vocabs": {"task": ["A", "B", "C"]},
                    "tasks": ["task"],
                },
                "want": {
                    **default_want,
                    "finetuning_task": "task",
                    "problem_type": "single_label_classification",
                    "num_labels": 3,
                    "id2label": {0: "A", 1: "B", 2: "C"},
                    "label2id": {"A": 0, "B": 1, "C": 2},
                },
            },
            {
                "msg": "Should set appropriate with binary task_df.",
                "pyd_spec": {
                    **default_pyd_spec,
                    "has_task": True,
                    "task_types": {"task": "binary_classification"},
                    "task_vocabs": {"task": [False, True]},
                    "tasks": ["task"],
                },
                "want": {
                    **default_want,
                    "finetuning_task": "task",
                    "problem_type": "single_label_classification",
                    "num_labels": 2,
                    "id2label": {0: False, 1: True},
                    "label2id": {False: 0, True: 1},
                },
            },
            {
                "msg": "Should set appropriate with multi_label binary task_df.",
                "pyd_spec": {
                    **default_pyd_spec,
                    "has_task": True,
                    "task_types": {
                        "task1": "binary_classification",
                        "task2": "binary_classification",
                        "task3": "binary_classification",
                    },
                    "task_vocabs": {
                        "task1": [False, True],
                        "task2": [False, True],
                        "task3": [False, True],
                    },
                    "tasks": ["task1", "task2", "task3"],
                },
                "want": {
                    **default_want,
                    "finetuning_task": None,
                    "problem_type": "multi_label_classification",
                    "num_labels": 3,
                },
            },
            {
                "msg": "Should set appropriate with multivariate regression task_df.",
                "pyd_spec": {
                    **default_pyd_spec,
                    "has_task": True,
                    "task_types": {"task1": "regression", "task2": "regression"},
                    "tasks": ["task1", "task2"],
                },
                "want": {
                    **default_want,
                    "finetuning_task": None,
                    "problem_type": "regression",
                    "num_labels": 2,
                },
            },
            {
                "msg": "Should set appropriate with mixed task_df.",
                "pyd_spec": {
                    **default_pyd_spec,
                    "has_task": True,
                    "task_types": {"task1": "regression", "task2": "binary_classification"},
                    "tasks": ["task1", "task2"],
                },
                "want": {
                    **default_want,
                    "finetuning_task": None,
                    "problem_type": None,
                },
            },
        ]

        for C in cases:
            with self.subTest(C["msg"]):
                pyd = MagicMock()
                pyd.vocabulary_config = MagicMock()
                for k, v in C["pyd_spec"].pop("vocabulary_config").items():
                    setattr(pyd.vocabulary_config, k, v)
                for k, v in C["pyd_spec"].items():
                    setattr(pyd, k, v)

                cfg = StructuredTransformerConfig(
                    **DEFAULT_CONDITIONALLY_INDEPENDENT_DICT,
                )
                cfg.set_to_dataset(pyd)
                for k, v in C["want"].items():
                    self.assertEqual(v, getattr(cfg, k))

    def test_save_load(self):
        """Tests the saving and loading of these configs.

        While this is largely huggingface functionality, here we test it to ensure that even when set to
        various modes, with different validity requirements, saving and re-loading is still possible (e.g.,
        ensuring that post-processing doesn't invalidate validation constraints).
        """
        for params in (
            {},
            DEFAULT_CONDITIONALLY_INDEPENDENT_DICT,
            DEFAULT_NESTED_ATTENTION_DICT,
            DEFAULT_EXPONENTIAL_DICT,
        ):
            cfg = StructuredTransformerConfig(**params)

            with TemporaryDirectory() as d:
                save_path = Path(d) / "config.json"
                cfg.to_json_file(save_path)
                got_cfg = StructuredTransformerConfig.from_json_file(save_path)

                # This isn't persisted properly for some reason (dependent on Huggingface, not me!)
                got_cfg.transformers_version = None
                cfg.transformers_version = None

                self.assertEqual(cfg, got_cfg)

                with self.assertRaises(FileNotFoundError):
                    got_cfg = StructuredTransformerConfig.from_json_file(Path(d) / "not_found.json")

                with self.assertRaises(FileNotFoundError):
                    cfg.to_json_file(Path(d) / "not_found" / "config.json")

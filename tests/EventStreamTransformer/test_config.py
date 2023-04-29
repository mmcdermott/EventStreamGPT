import sys
sys.path.append('../..')

import unittest
from unittest.mock import MagicMock
from pathlib import Path
from tempfile import TemporaryDirectory

from ..mixins import ConfigComparisonsMixin

from EventStream.EventStreamData.types import DataModality
from EventStream.EventStreamTransformer.config import (
    EventStreamOptimizationConfig,
    StructuredEventProcessingMode,
    StructuredEventStreamTransformerConfig,
    TimeToEventGenerationHeadType,
)

DEFAULT_OPT_CONFIG_DICT = dict(
    init_lr=1e-2,
    end_lr=1e-7,
    max_epochs=2,
    batch_size=32,
    lr_frac_warmup_steps=0.01,
    lr_num_warmup_steps=None,
    max_training_steps=None,
    lr_decay_power=1.0,
    weight_decay=0.01,
)

class TestEventStreamOptimizationConfig(unittest.TestCase):
    def test_set_to_dataset(self):
        cfg = EventStreamOptimizationConfig(**{
            **DEFAULT_OPT_CONFIG_DICT,
            'max_epochs': 10,
            'batch_size': 2,
            'lr_frac_warmup_steps': 6/30,
        })

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
    do_add_temporal_position_embeddings_to_data_embeddings=False,
)

DEFAULT_CONDITIONALLY_INDEPENDENT_DICT = dict(
    structured_event_processing_mode=StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT,
    dep_graph_attention_types=None,
    dep_graph_window_size=None,
    do_full_block_in_dep_graph_attention=None,
    do_full_block_in_seq_attention=None,
    do_add_temporal_position_embeddings_to_data_embeddings=None,
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

class TestStructuredEventStreamTransformerConfig(ConfigComparisonsMixin, unittest.TestCase):
    def test_construction(self):
        """
        Tests the construction and initialization logic of the `StructuredEventStreamTransformerConfig`
        object.
        """

        cases = [
            {
                'msg': "Should construct with default args.",
                'kwargs': {},
            }, {
                'msg': "Should construct with nested_attention args.",
                'kwargs': {**DEFAULT_NESTED_ATTENTION_DICT},
            }, {
                'msg': "Should Error when head_dim and hidden_size are missing.",
                'kwargs': {'head_dim': 32, 'hidden_size': 64, 'num_attention_heads': 4},
                'should_raise': ValueError,
            }, {
                'msg': "Should Error when head_dim and hidden_size are inconsistent.",
                'kwargs': {'head_dim': None, 'hidden_size': None},
                'should_raise': ValueError,
            }, {
                'msg': "Should Error when num_hidden_layers is not an int.",
                'kwargs': {'num_hidden_layers': 4.0},
                'should_raise': TypeError,
            }, {
                'msg': "Should Error when num_hidden_layers is negative.",
                'kwargs': {'num_hidden_layers': -4},
                'should_raise': ValueError,
            }, {
                'msg': "Should Error when seq_attention_types is misconfigured.",
                'kwargs': {'num_hidden_layers': 4, 'seq_attention_types': [[["global"], 10]]},
                'should_raise': ValueError,
            }, {
                'msg': "Should Error when nested_attention args are missing or invalid.",
                'kwargs': [
                    {
                        **DEFAULT_NESTED_ATTENTION_DICT,
                        'do_full_block_in_dep_graph_attention': None,
                    }, {
                        **DEFAULT_NESTED_ATTENTION_DICT,
                        'do_full_block_in_seq_attention': None,
                    }, {
                        **DEFAULT_NESTED_ATTENTION_DICT,
                        'do_add_temporal_position_embeddings_to_data_embeddings': None,
                    }, {
                        **DEFAULT_NESTED_ATTENTION_DICT,
                        'num_hidden_layers': 4,
                        'dep_graph_attention_types': [[["global"], 10]],
                    }
                ],
                'should_raise': ValueError,
            }, {
                'msg': "Should construct with conditionally_independent args.",
                'kwargs': {**DEFAULT_CONDITIONALLY_INDEPENDENT_DICT},
            }, {
                'msg': "Should Error when given nested_attention args when in conditionally_independent mode.",
                'kwargs': [
                    {
                        **DEFAULT_CONDITIONALLY_INDEPENDENT_DICT,
                        'dep_graph_attention_types': [['global'], 2],
                    }, {
                        **DEFAULT_CONDITIONALLY_INDEPENDENT_DICT,
                        'dep_graph_attention_types': [['global'], 2],
                    }, {
                        **DEFAULT_CONDITIONALLY_INDEPENDENT_DICT,
                        'dep_graph_window_size': 2,
                    }, {
                        **DEFAULT_CONDITIONALLY_INDEPENDENT_DICT,
                        'do_full_block_in_dep_graph_attention': False,
                    }, {
                        **DEFAULT_CONDITIONALLY_INDEPENDENT_DICT,
                        'do_full_block_in_seq_attention': False,
                    }, {
                        **DEFAULT_CONDITIONALLY_INDEPENDENT_DICT,
                        'do_add_temporal_position_embeddings_to_data_embeddings': False,
                    },
                ],
                'should_raise': ValueError,
            }, {
                'msg': "Should construct with Exponential TTE head args.",
                'kwargs': {**DEFAULT_EXPONENTIAL_DICT},
            }, {
                'msg': "Should error when Lognormal Mixture args are passed in Exponential mode.",
                'kwargs': [
                    {**DEFAULT_EXPONENTIAL_DICT, 'TTE_lognormal_generation_num_components': 2},
                    {**DEFAULT_EXPONENTIAL_DICT, 'mean_log_inter_event_time_min': 0.},
                    {**DEFAULT_EXPONENTIAL_DICT, 'std_log_inter_event_time_min': 1.},
                ],
                'should_raise': ValueError
            }, {
                'msg': "Should construct with Lognormal Mixture TTE head args.",
                'kwargs': {**DEFAULT_LOGNORMAL_MIXTURE_DICT},
            }, {
                'msg': "Should error when required Lognormal Mixture args are missing or invalid.",
                'kwargs': [
                    {**DEFAULT_LOGNORMAL_MIXTURE_DICT, 'TTE_lognormal_generation_num_components': None},
                    {**DEFAULT_LOGNORMAL_MIXTURE_DICT, 'TTE_lognormal_generation_num_components': -1},
                ],
                'should_raise': ValueError
            }, {
                'msg': "Should error when required Lognormal Mixture arg is the wrong type.",
                'kwargs': {
                    **DEFAULT_LOGNORMAL_MIXTURE_DICT, 'TTE_lognormal_generation_num_components': 1.0,
                },
                'should_raise': TypeError
            }, {
                'msg': "Should error when is specified as encoder_decoder.",
                'kwargs': {'is_encoder_decoder': True},
                'should_raise': AssertionError
            }
        ]

        for C in cases:
            with self.subTest(C['msg']):
                kwargs = C['kwargs'] if type(C['kwargs']) is list else [C['kwargs']]
                for args_dict in kwargs:
                    if 'should_raise' in C:
                        with self.assertRaises(C['should_raise']):
                            StructuredEventStreamTransformerConfig(**args_dict)
                    else:
                        StructuredEventStreamTransformerConfig(**args_dict)

    def test_expand_attention_types_params(self):
        for C in [
            {'attention_types': [[['global'], 2]], 'want': ['global', 'global']},
            {'attention_types': [[['global', 'local'], 2]], 'want': ['global', 'local', 'global', 'local']},
            {'attention_types': [[['global'], 1], [['local'], 2]], 'want': ['global', 'local', 'local']},
        ]:
            got = StructuredEventStreamTransformerConfig.expand_attention_types_params(C['attention_types'])
            self.assertEqual(C['want'], got)

    def test_set_to_dataset(self):
        default_measurements_per_generative_mode = {
            DataModality.SINGLE_LABEL_CLASSIFICATION: ['event_type'],
            DataModality.MULTI_LABEL_CLASSIFICATION: ['B_key', 'A_col'],
            DataModality.MULTIVARIATE_REGRESSION: ['B_key'],
        }
        default_measurements_idxmap = {'event_type': 1, 'B_key': 2, 'A_col': 3}
        default_measurement_vocab_offsets = {'event_type': 1, 'B_key': 2, 'A_col': 5}
        default_vocab_sizes_by_measurement = {'event_type': 2, 'B_key': 3, 'A_col': 4}

        default_pyd_spec = {
            'has_task': False,
            'max_seq_len': 4,
            'vocabulary_config': {
                'measurements_idxmap': default_measurements_idxmap,
                'vocab_offsets_by_measurement': default_measurement_vocab_offsets,
                'vocab_sizes_by_measurement': default_vocab_sizes_by_measurement,
                'measurements_per_generative_mode': default_measurements_per_generative_mode,
                'total_vocab_size': 9,
            },
        }

        default_want = {
            'max_seq_len': 4,
            'vocab_size': 9,
            'vocab_sizes_by_measurement': default_vocab_sizes_by_measurement,
            'measurements_idxmap': default_measurements_idxmap,
            'vocab_offsets_by_measurement': default_measurement_vocab_offsets,
            'measurements_per_generative_mode': default_measurements_per_generative_mode,
            'finetuning_task': None,
            'problem_type': None,
        }

        cases = [
            {
                'msg': "Should set appropriate with no task_df.",
                'pyd_spec': default_pyd_spec, 'want': default_want,
            }, {
                'msg': "Should set appropriate regression task_df.",
                'pyd_spec': {
                    **default_pyd_spec,
                    'has_task': True,
                    'task_types': {'task': 'regression'},
                    'tasks': ['task'],
                },
                'want': {
                    **default_want,
                    'finetuning_task': 'task',
                    'problem_type': 'regression',
                    'num_labels': 1,
                },
            }, {
                'msg': "Should set appropriate multi_class_classification task_df.",
                'pyd_spec': {
                    **default_pyd_spec,
                    'has_task': True,
                    'task_types': {'task': 'multi_class_classification'},
                    'task_vocabs': {'task': ['A', 'B', 'C']},
                    'tasks': ['task'],
                },
                'want': {
                    **default_want,
                    'finetuning_task': 'task',
                    'problem_type': 'single_label_classification',
                    'num_labels': 3,
                    'id2label': {0: 'A', 1: 'B', 2: 'C'},
                    'label2id': {'A': 0, 'B': 1, 'C': 2},
                },
            }, {
                'msg': "Should set appropriate with binary task_df.",
                'pyd_spec': {
                    **default_pyd_spec,
                    'has_task': True,
                    'task_types': {'task': 'binary_classification'},
                    'task_vocabs': {'task': [False, True]},
                    'tasks': ['task'],
                },
                'want': {
                    **default_want,
                    'finetuning_task': 'task',
                    'problem_type': 'single_label_classification',
                    'num_labels': 2,
                    'id2label': {0: False, 1: True},
                    'label2id': {False: 0, True: 1},
                },
            }, {
                'msg': "Should set appropriate with multi_label binary task_df.",
                'pyd_spec': {
                    **default_pyd_spec,
                    'has_task': True,
                    'task_types': {
                        'task1': 'binary_classification',
                        'task2': 'binary_classification',
                        'task3': 'binary_classification',
                    },
                    'task_vocabs': {'task1': [False, True], 'task2': [False, True], 'task3': [False, True]},
                    'tasks': ['task1', 'task2', 'task3'],
                },
                'want': {
                    **default_want,
                    'finetuning_task': None,
                    'problem_type': 'multi_label_classification',
                    'num_labels': 3,
                },
            }, {
                'msg': "Should set appropriate with multivariate regression task_df.",
                'pyd_spec': {
                    **default_pyd_spec,
                    'has_task': True,
                    'task_types': {'task1': 'regression', 'task2': 'regression'},
                    'tasks': ['task1', 'task2'],
                },
                'want': {
                    **default_want,
                    'finetuning_task': None,
                    'problem_type': 'regression',
                    'num_labels': 2,
                },
            }, {
                'msg': "Should set appropriate with mixed task_df.",
                'pyd_spec': {
                    **default_pyd_spec,
                    'has_task': True,
                    'task_types': {'task1': 'regression', 'task2': 'binary_classification'},
                    'tasks': ['task1', 'task2'],
                },
                'want': {
                    **default_want,
                    'finetuning_task': None,
                    'problem_type': None,
                },
            }
        ]

        for C in cases:
            with self.subTest(C['msg']):
                pyd = MagicMock()
                pyd.vocabulary_config = MagicMock()
                for k, v in C['pyd_spec'].pop('vocabulary_config').items():
                    setattr(pyd.vocabulary_config, k, v)
                for k, v in C['pyd_spec'].items(): setattr(pyd, k, v)

                cfg = StructuredEventStreamTransformerConfig(
                    **DEFAULT_CONDITIONALLY_INDEPENDENT_DICT,
                )
                cfg.set_to_dataset(pyd)
                for k, v in C['want'].items():
                    self.assertEqual(v, getattr(cfg, k))

    def test_save_load(self):
        """
        Tests the saving and loading of these configs. While this is largely huggingface functionality, here
        we test it to ensure that even when set to various modes, with different validity requirements, saving
        and re-loading is still possible (e.g., ensuring that post-processing doesn't invalidate validation
        constraints).
        """
        for params in (
            {},
            DEFAULT_CONDITIONALLY_INDEPENDENT_DICT,
            DEFAULT_NESTED_ATTENTION_DICT,
            DEFAULT_EXPONENTIAL_DICT,
        ):
            cfg = StructuredEventStreamTransformerConfig(**params)

            with TemporaryDirectory() as d:
                save_path = Path(d) / 'config.json'
                cfg.to_json_file(save_path)
                got_cfg = StructuredEventStreamTransformerConfig.from_json_file(save_path)

                # This isn't persisted properly for some reason (dependent on Huggingface, not me!)
                got_cfg.transformers_version = None
                cfg.transformers_version = None

                self.assertEqual(cfg, got_cfg)

                with self.assertRaises(FileNotFoundError):
                    got_cfg = StructuredEventStreamTransformerConfig.from_json_file(Path(d)/'not_found.json')

                with self.assertRaises(FileNotFoundError):
                    cfg.to_json_file(Path(d) / 'not_found' / 'config.json')

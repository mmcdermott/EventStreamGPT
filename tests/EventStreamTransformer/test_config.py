import sys
sys.path.append('../..')

import torch, unittest, pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from ..mixins import ConfigComparisonsMixin

from EventStream.EventStreamData.config import (
    EventStreamDatasetConfig,
    EventStreamPytorchDatasetConfig,
)
from EventStream.EventStreamData.types import DataModality
from EventStream.EventStreamData.event_stream_dataset import EventStreamDataset
from EventStream.EventStreamData.event_stream_pytorch_dataset import (
    EventStreamPytorchDataset
)
from EventStream.EventStreamTransformer.config import (
    EventStreamOptimizationConfig,
    StructuredEventProcessingMode,
    StructuredEventStreamTransformerConfig,
    TimeToEventGenerationHeadType,
)

DEFAULT_OPT_CONFIG_DICT = dict(
    init_lr    = 1e-2,
    end_lr     = 1e-7,
    max_epochs = 2,
    batch_size = 32,
    lr_frac_warmup_steps = 0.01,
    lr_num_warmup_steps  = None,
    max_training_steps   = None,
    lr_decay_power       = 1.0,
    weight_decay         = 0.01,
)

class TestEventStreamOptimizationConfig(unittest.TestCase):
    def test_set_to_dataset(self):
        cfg = EventStreamOptimizationConfig( **{
            **DEFAULT_OPT_CONFIG_DICT,
            'max_epochs': 10,
            'batch_size': 2,
            'lr_frac_warmup_steps': 6/30,
        })

        n_patients = 6

        events_df = pd.DataFrame({
            'subject_id': list(range(n_patients)),
            'timestamp': ['12/1/22'] * n_patients,
            'event_type': ['A'] * n_patients,
        }, index=pd.Index(list(range(n_patients)), name='event_id'))

        metadata_df = pd.DataFrame({
            'A_col': ['foo', 'bar', 'foo', 'bar', 'bax'] * n_patients,
            'B_key': ['a', 'a', 'a', 'b', 'b'] * n_patients,
            'B_val': [1, 2, 3, 4, 5] * n_patients,
            'event_id': [i for i in range(n_patients) for _ in range(5)],
        })

        config = EventStreamDatasetConfig.from_simple_args(['A_col', ('B_key', 'B_val')])
        E = EventStreamDataset(events_df=events_df, metadata_df=metadata_df, config=config)
        E.split_subjects = {'train': set(range(n_patients))}
        E.preprocess_metadata()

        data_config = EventStreamPytorchDatasetConfig(
            do_normalize_log_inter_event_times = False,
            max_seq_len = 4,
            min_seq_len = 0,
        )

        pyd = EventStreamPytorchDataset(E, data_config, split='train')

        cfg.set_to_dataset(pyd)

        self.assertEqual(10 * 3, cfg.max_training_steps)
        self.assertEqual(6, cfg.lr_num_warmup_steps)

DEFAULT_MAIN_CONFIG_DICT = dict(
    vocab_sizes_by_measurement = None,
    vocab_offsets_by_measurement = None,
    measurements_idxmap = None,
    measurements_per_generative_mode = None,
    data_cols = None,
    measurements_per_dep_graph_level = None,
    max_seq_len = 256,
    hidden_size = 256,
    head_dim = 64,
    num_hidden_layers = 2,
    num_attention_heads = 4,
    seq_attention_types = None,
    seq_window_size = 32,
    intermediate_size = 32,
    activation_function = "gelu",
    attention_dropout = 0.1,
    input_dropout = 0.1,
    resid_dropout = 0.1,
    init_std = 0.02,
    layer_norm_epsilon = 1e-5,
    use_cache = True,
)

DEFAULT_NESTED_ATTENTION_DICT = dict(
    structured_event_processing_mode = StructuredEventProcessingMode.NESTED_ATTENTION,
    dep_graph_attention_types = None,
    dep_graph_window_size = 2,
    do_full_block_in_dep_graph_attention = True,
    do_full_block_in_seq_attention = False,
    do_add_temporal_position_embeddings_to_data_embeddings = False,
)

DEFAULT_CONDITIONALLY_INDEPENDENT_DICT = dict(
    structured_event_processing_mode = StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT,
    dep_graph_attention_types = None,
    dep_graph_window_size = None,
    do_full_block_in_dep_graph_attention = None,
    do_full_block_in_seq_attention = None,
    do_add_temporal_position_embeddings_to_data_embeddings = None,
)

DEFAULT_EXPONENTIAL_DICT = dict(
    TTE_generation_layer_type = TimeToEventGenerationHeadType.EXPONENTIAL,
    TTE_lognormal_generation_num_components = None,
    mean_log_inter_event_time_min = None,
    std_log_inter_event_time_min = None,
)

DEFAULT_LOGNORMAL_MIXTURE_DICT = dict(
    TTE_generation_layer_type = TimeToEventGenerationHeadType.LOG_NORMAL_MIXTURE,
    TTE_lognormal_generation_num_components = 3,
    mean_log_inter_event_time_min = None,
    std_log_inter_event_time_min = 1.0,
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
                            cfg = StructuredEventStreamTransformerConfig(**args_dict)
                    else:
                        cfg = StructuredEventStreamTransformerConfig(**args_dict)

    def test_expand_attention_types_params(self):
        for C in [
            {'attention_types': [[['global'], 2]], 'want': ['global', 'global']},
            {'attention_types': [[['global', 'local'], 2]], 'want': ['global', 'local', 'global', 'local']},
            {'attention_types': [[['global'], 1], [['local'], 2]], 'want': ['global', 'local', 'local']},
        ]:
            got = StructuredEventStreamTransformerConfig.expand_attention_types_params(C['attention_types'])
            self.assertEqual(C['want'], got)

    def test_set_to_dataset(self):
        events_df = pd.DataFrame({
            'subject_id': [1, 1, 1],
            'timestamp': ['12/1/22 1:00 am', '12/1/22 2:00 am', '12/2/22'],
            'event_type': ['A', 'A', 'A'],
        }, index=pd.Index([0, 1, 2], name='event_id'))
        metadata_df = pd.DataFrame({
            'A_col': ['foo', 'bar', 'foo', 'bar', 'bax'],
            'B_key': ['a', 'a', 'a', 'b', 'b'],
            'B_val': [1, 2, 3, 4, 5],
            'event_id': [0, 0, 0, 0, 0],
        })

        config = EventStreamDatasetConfig.from_simple_args(['A_col', ('B_key', 'B_val')])
        E = EventStreamDataset(events_df=events_df, metadata_df=metadata_df, config=config)
        E.split_subjects = {'train': {1}}
        E.preprocess_metadata()

        data_config = EventStreamPytorchDatasetConfig(
            do_normalize_log_inter_event_times = False,
            max_seq_len = 4,
        )
        pyd = EventStreamPytorchDataset(E, data_config, split='train')

        cfg = StructuredEventStreamTransformerConfig()
        cfg.set_to_dataset(pyd)

        self.assertEqual(4, cfg.max_seq_len)
        self.assertEqual(9, cfg.vocab_size)
        self.assertEqual([('B_key', 'B_val'), 'A_col'], cfg.data_cols)
        self.assertEqual({'event_type': 1, 'A_col': 4, 'B_key': 3}, cfg.vocab_sizes_by_measurement)
        self.assertEqual({'event_type': 1, 'B_key': 2, 'A_col': 5}, cfg.vocab_offsets_by_measurement)
        self.assertEqual({'event_type': 1, 'B_key': 2, 'A_col': 3}, cfg.measurements_idxmap)

        want_measurements_per_generative_mode = {
            DataModality.SINGLE_LABEL_CLASSIFICATION: ['event_type'],
            DataModality.MULTI_LABEL_CLASSIFICATION: ['B_key', 'A_col'],
            DataModality.MULTIVARIATE_REGRESSION: ['B_key'],
        }
        self.assertEqual(want_measurements_per_generative_mode, cfg.measurements_per_generative_mode)

        self.assertEqual(None, cfg.mean_log_inter_event_time_min)
        self.assertEqual(None, cfg.std_log_inter_event_time_min)

        # When in lognormal mode, it should also copy the inter event time stats.
        cfg = StructuredEventStreamTransformerConfig(**DEFAULT_LOGNORMAL_MIXTURE_DICT)
        data_config = EventStreamPytorchDatasetConfig(
            do_normalize_log_inter_event_times = False,
            max_seq_len = 4,
        )
        pyd = EventStreamPytorchDataset(E, data_config, split='train')
        cfg.set_to_dataset(pyd)

        self.assertEqual(E.train_mean_log_inter_event_time_min, cfg.mean_log_inter_event_time_min)
        self.assertEqual(E.train_std_log_inter_event_time_min, cfg.std_log_inter_event_time_min)

        cfg = StructuredEventStreamTransformerConfig(**DEFAULT_LOGNORMAL_MIXTURE_DICT)
        data_config = EventStreamPytorchDatasetConfig(
            do_normalize_log_inter_event_times = True,
            max_seq_len = 4,
        )
        pyd = EventStreamPytorchDataset(E, data_config, split='train')
        cfg.set_to_dataset(pyd)

        self.assertEqual(0, cfg.mean_log_inter_event_time_min)
        self.assertEqual(1, cfg.std_log_inter_event_time_min)

        # Testing fine-tuning dataset parameter setting.
        task_df = pd.DataFrame({
            'subject_id': [1, 1, 1, 1],
            'start_time': [
                pd.to_datetime('12/1/22 12:00 am'),
                pd.to_datetime('12/1/22 12:00 am'),
                pd.to_datetime('12/1/22 12:00 am'),
                pd.to_datetime('12/1/22 12:00 am'),
            ],
            'end_time': [
                pd.to_datetime('12/1/22 1:00 am'),
                pd.to_datetime('12/1/22 2:00 am'),
                pd.to_datetime('12/1/22 3:00 am'),
                pd.to_datetime('12/1/22 4:00 am'),
            ],
        })

        binary_task_df = task_df.copy()
        binary_task_df['binary'] = [True, False, True, False]

        pyd = EventStreamPytorchDataset(E, data_config, task_df=binary_task_df, split='train')

        cfg = StructuredEventStreamTransformerConfig()
        cfg.set_to_dataset(pyd)

        self.assertEqual('binary', cfg.finetuning_task)
        self.assertEqual('single_label_classification', cfg.problem_type)
        self.assertEqual(2, cfg.num_labels)
        self.assertEqual({0: False, 1: True}, cfg.id2label)
        self.assertEqual({False: 0, True: 1}, cfg.label2id)

        multi_class_task_df = task_df.copy()
        multi_class_task_df['multi_class'] = [0, 1, 2, 4]

        pyd = EventStreamPytorchDataset(E, data_config, task_df=multi_class_task_df, split='train')

        cfg = StructuredEventStreamTransformerConfig()
        cfg.set_to_dataset(pyd)

        self.assertEqual('multi_class', cfg.finetuning_task)
        self.assertEqual('single_label_classification', cfg.problem_type)
        self.assertEqual(5, cfg.num_labels)
        self.assertEqual({0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, cfg.id2label)
        self.assertEqual({0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, cfg.label2id)

        multi_class_task_df = task_df.copy()
        multi_class_task_df['multi_class'] = pd.Series(['a', 'b', 'a', 'z'], dtype='category')

        pyd = EventStreamPytorchDataset(E, data_config, task_df=multi_class_task_df, split='train')

        cfg = StructuredEventStreamTransformerConfig()
        cfg.set_to_dataset(pyd)

        self.assertEqual('multi_class', cfg.finetuning_task)
        self.assertEqual('single_label_classification', cfg.problem_type)
        self.assertEqual(3, cfg.num_labels)
        self.assertEqual({0: 'a', 1: 'b', 2: 'z'}, cfg.id2label)
        self.assertEqual({'a': 0, 'b': 1, 'z': 2}, cfg.label2id)

        regression_task_df = task_df.copy()
        regression_task_df['regression'] = [1.4, 2.3, 4.2, 1.1]

        pyd = EventStreamPytorchDataset(E, data_config, task_df=regression_task_df, split='train')

        cfg = StructuredEventStreamTransformerConfig()
        cfg.set_to_dataset(pyd)

        self.assertEqual('regression', cfg.finetuning_task)
        self.assertEqual('regression', cfg.problem_type)
        self.assertEqual(1, cfg.num_labels)

        regression_task_df = task_df.copy()
        regression_task_df['regression_1'] = [1.4, 2.3, 4.2, 1.1]
        regression_task_df['regression_2'] = [1.3, 2.2, 422, 1.9]

        pyd = EventStreamPytorchDataset(E, data_config, task_df=regression_task_df, split='train')

        cfg = StructuredEventStreamTransformerConfig()
        cfg.set_to_dataset(pyd)

        self.assertEqual(None, cfg.finetuning_task)
        self.assertEqual('regression', cfg.problem_type)
        self.assertEqual(2, cfg.num_labels)

        multi_label_task_df = task_df.copy()
        multi_label_task_df['multi_label_1'] = [True, False, False, False]
        multi_label_task_df['multi_label_2'] = [True, True, False, False]
        multi_label_task_df['multi_label_3'] = [True, True, True, False]

        pyd = EventStreamPytorchDataset(E, data_config, task_df=multi_label_task_df, split='train')

        cfg = StructuredEventStreamTransformerConfig()
        cfg.set_to_dataset(pyd)

        self.assertEqual(None, cfg.finetuning_task)
        self.assertEqual('multi_label_classification', cfg.problem_type)
        self.assertEqual(3, cfg.num_labels)

        mixed_task_df = task_df.copy()
        mixed_task_df['mixed_1'] = [True, False, False, False]
        mixed_task_df['mixed_2'] = [1, 2, 3, 4]
        mixed_task_df['mixed_3'] = [1.1, 2.3, 1.1, 3.]

        pyd = EventStreamPytorchDataset(E, data_config, task_df=mixed_task_df, split='train')

        cfg = StructuredEventStreamTransformerConfig()
        default_num_labels = cfg.num_labels
        cfg.set_to_dataset(pyd)

        self.assertEqual(None, cfg.finetuning_task)
        self.assertEqual(None, cfg.problem_type)
        self.assertEqual(default_num_labels, cfg.num_labels)

    def test_save_load(self):
        """
        Tests the saving and loading of these configs. While this is largely huggingface functionality, here
        we test it to ensure that even when set to various modes, with different validity requirements, saving
        and re-loading is still possible (e.g., ensuring that post-processing doesn't invalidate validation
        constraints).
        """
        events_df = pd.DataFrame({
            'subject_id': [1, 1, 1],
            'timestamp': ['12/1/22 1:00 am', '12/1/22 2:00 am', '12/2/22'],
            'event_type': ['A', 'A', 'A'],
        }, index=pd.Index([0, 1, 2], name='event_id'))
        metadata_df = pd.DataFrame({
            'A_col': ['foo', 'bar', 'foo', 'bar', 'bax'],
            'B_key': ['a', 'a', 'a', 'b', 'b'],
            'B_val': [1, 2, 3, 4, 5],
            'event_id': [0, 0, 0, 0, 0],
        })

        config = EventStreamDatasetConfig.from_simple_args(['A_col', ('B_key', 'B_val')])
        E = EventStreamDataset(events_df=events_df, metadata_df=metadata_df, config=config)
        E.split_subjects = {'train': {1}}
        E.preprocess_metadata()

        data_config = EventStreamPytorchDatasetConfig(
            do_normalize_log_inter_event_times = False,
            max_seq_len = 4,
        )
        pyd = EventStreamPytorchDataset(E, data_config, split='train')

        for params in (
            {},
            DEFAULT_CONDITIONALLY_INDEPENDENT_DICT,
            DEFAULT_NESTED_ATTENTION_DICT,
            DEFAULT_LOGNORMAL_MIXTURE_DICT,
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

                cfg.set_to_dataset(pyd)
                cfg.to_json_file(save_path)
                got_cfg = StructuredEventStreamTransformerConfig.from_json_file(save_path)

                # This isn't persisted properly for some reason (dependent on Huggingface, not me!)
                got_cfg.transformers_version = None
                cfg.transformers_version = None

                # Tuples are converted to lists via json...
                self.assertEqual([('B_key', 'B_val'), 'A_col'], cfg.data_cols)
                self.assertEqual([['B_key', 'B_val'], 'A_col'], got_cfg.data_cols)

                cfg.data_cols = got_cfg.data_cols

                self.assertEqual(cfg, got_cfg)

                with self.assertRaises(FileNotFoundError):
                    got_cfg = StructuredEventStreamTransformerConfig.from_json_file(Path(d)/'not_found.json')

                with self.assertRaises(FileNotFoundError):
                    cfg.to_json_file(Path(d) / 'not_found' / 'config.json')

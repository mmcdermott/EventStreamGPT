import sys
sys.path.append('../..')

import torch, unittest, numpy as np, pandas as pd
from dataclasses import asdict
from typing import Optional

from ..mixins import MLTypeEqualityCheckableMixin
from EventStream.EventStreamData.config import EventStreamDatasetConfig
from EventStream.EventStreamData.expandable_df_dict import ExpandableDfDict
from EventStream.EventStreamData.event_stream_dataset import EventStreamDataset
from EventStream.EventStreamData.event_stream_pytorch_dataset import (
    EventStreamPytorchDataset,
    EventStreamPytorchDatasetConfig,
)
from EventStream.EventStreamData.time_dependent_functor import TimeOfDayFunctor, AgeFunctor
from EventStream.EventStreamData.types import DataModality, EventStreamPytorchBatch
from EventStream.EventStreamData.vocabulary import Vocabulary

class TestEventStreamPytorchDataset(MLTypeEqualityCheckableMixin, unittest.TestCase):
    def test_normalize_task(self):
        cases = [
            {
                'msg': "Should flag Integer values as multi-class.",
                'vals': pd.Series([1, 2, 1, 4], dtype=int),
                'want_type': 'multi_class_classification',
                'want_vals': pd.Series([1, 2, 1, 4], dtype=int),
            }, {
                'msg': "Should flag Categorical values as multi-class and normalize to integers.",
                'vals': pd.Series(['a', 'b', 'a', 'z'], dtype='category'),
                'want_type': 'multi_class_classification',
                'want_vals': pd.Series([0, 1, 0, 2], dtype=int),
            }, {
                'msg': "Should flag Boolean values as binary and normalize to float.",
                'vals': pd.Series([True, False, True, False], dtype=bool),
                'want_type': 'binary_classification',
                'want_vals': pd.Series([1., 0., 1., 0.]),
            }, {
                'msg': "Should flag Float values as regression.",
                'vals': pd.Series([1.0, 2.1, 1.3, 4.1]),
                'want_type': 'regression',
                'want_vals': pd.Series([1.0, 2.1, 1.3, 4.1]),
            }, {
                'msg': "Should raise TypeError on object type.",
                'vals': pd.Series(["fuzz", 3, pd.to_datetime('12/2/22'), float('nan')]),
                'want_raise': TypeError,
            }
        ]

        for C in cases:
            with self.subTest(C['msg']):
                if C.get('want_raise', None) is not None:
                    with self.assertRaises(C['want_raise']):
                        EventStreamPytorchDataset.normalize_task(C['vals'])
                else:
                    got_type, got_vals = EventStreamPytorchDataset.normalize_task(C['vals'])
                    self.assertEqual(C['want_type'], got_type)
                    self.assertEqual(C['want_vals'], got_vals)

    def test_basic_construction(self):
        """EventStreamPytorchDataset should construct appropriately"""
        subjects_df = pd.DataFrame({
                'buzz': ['foo', 'bar', 'foo'],
                'dob': [
                    pd.to_datetime('12/1/21'),
                    pd.to_datetime('12/1/20'),
                    pd.to_datetime('12/1/90'),
                ]
            },
            index=pd.Index([1, 2, 3], name='subject_id')
        )

        events_df = pd.DataFrame({
            'subject_id': [1, 1, 1, 1, 2, 2],
            'timestamp': [
                '12/1/22 12:00 a.m.',
                '12/2/22 2:00 p.m.',
                '12/3/22 10:00 a.m.',
                '12/4/22 11:00 p.m.',
                '12/1/22 15:00',
                '12/2/22 2:00',
            ],
            'event_type': ['A', 'B', 'A', 'A', 'A', 'B'],
            'metadata': [
                ExpandableDfDict({'A_col': ['foo']}),
                ExpandableDfDict({
                    'B_key': [
                        'a', 'a', 'a',
                        'b', 'b',
                    ],
                    'B_val': [
                        1, 2, 3,
                        4, 5,
                    ],
                }),
                ExpandableDfDict({'A_col': ['bar']}),
                ExpandableDfDict({'A_col': ['foo']}),
                ExpandableDfDict({'A_col': ['foo']}),
                ExpandableDfDict({'B_key': ['a', 'b'], 'B_val': [1, 5]}),
            ],
        })

        config = EventStreamDatasetConfig.from_simple_args(
            dynamic_measurement_columns=['A_col', ('B_key', 'B_val')],
            static_measurement_columns=['buzz'],
            time_dependent_measurement_columns=[
                ('time_of_day', TimeOfDayFunctor()), ('age', AgeFunctor('dob'))
            ],
        )

        E = EventStreamDataset(events_df=events_df, subjects_df=subjects_df, config=config)
        E.split_subjects = {'train': {1, 2, 3}}
        E.preprocess_metadata()
        # Vocab is {
        #   'A_col': ['UNK', 'foo', 'bar'], 'B_key': ['UNK', 'a', 'b'],
        #   'buzz': ['UNK', 'foo', 'bar'],
        #   'time_of_day': ['UNK', 'EARLY_AM', 'PM', 'AM', 'LATE_PM'],
        # }

        data_config = EventStreamPytorchDatasetConfig(
            do_normalize_log_inter_event_times = False,
            max_seq_len = 4,
            min_seq_len = 2,
        )

        pyd = EventStreamPytorchDataset(E, data_config, split='train')

        want_subject_ids = [1, 2]
        self.assertEqual(want_subject_ids, pyd.subject_ids)
        self.assertEqual(2, len(pyd))

        want_event_types_idxmap = {'A': 0, 'B': 1}
        self.assertEqual(want_event_types_idxmap, pyd.event_types_idxmap)

        # I don't know why they have this order, but can't fix it now.
        want_dynamic_cols = [('B_key', 'B_val'), 'A_col']
        self.assertEqual(want_dynamic_cols, pyd.dynamic_cols)

        want_static_cols = ['buzz']
        self.assertEqual(want_static_cols, pyd.static_cols)

        want_event_cols = ['age', 'time_of_day']
        self.assertEqual(want_event_cols, pyd.event_cols)

        want_data_cols = want_dynamic_cols + want_event_cols + want_static_cols
        self.assertEqual(want_data_cols, pyd.data_cols)

        # It starts with event_type, then dynamic, then event, then static, in the order of the config.
        want_measurement_vocab_offsets = {
            'event_type': 1,
            'B_key': 3, 'A_col': 6,
            'age': 9, 'time_of_day': 10,
            'buzz': 15,
        }
        self.assertEqual(want_measurement_vocab_offsets, pyd.measurement_vocab_offsets)
        self.assertEqual(18, pyd.total_vocab_size)

        want_measurements_idxmap = {
            'event_type': 1,
            'B_key': 2, 'A_col': 3,
            'age': 4, 'time_of_day': 5,
            'buzz': 6,
        }
        self.assertEqual(want_measurements_idxmap, pyd.measurements_idxmap)
        self.assertEqual(6, pyd.total_n_measurements)

        # This ensures we can see the full delta if there is an error.
        self.maxDiff = None
        want_measurements_per_generative_mode = {
            DataModality.SINGLE_LABEL_CLASSIFICATION: ['event_type'],
            DataModality.MULTI_LABEL_CLASSIFICATION: ['B_key', 'A_col'],
            DataModality.MULTIVARIATE_REGRESSION: ['B_key'],
        }
        self.assertEqual(want_measurements_per_generative_mode, pyd.measurements_per_generative_mode)

        # Including empty subjects
        data_config = EventStreamPytorchDatasetConfig(
            do_normalize_log_inter_event_times = False,
            max_seq_len = 4,
            min_seq_len = 0,
        )

        pyd = EventStreamPytorchDataset(E, data_config, split='train')

        want_subject_ids = [1, 2, 3]
        self.assertEqual(want_subject_ids, pyd.subject_ids)
        self.assertEqual(3, len(pyd))

        # Including no subjects
        data_config = EventStreamPytorchDatasetConfig(
            do_normalize_log_inter_event_times = False,
            max_seq_len = 40,
            min_seq_len = 30,
        )

        pyd = EventStreamPytorchDataset(E, data_config, split='train')

        want_subject_ids = []
        self.assertEqual(want_subject_ids, pyd.subject_ids)
        self.assertEqual(0, len(pyd))

    def test_with_task_construction(self):
        events_df = pd.DataFrame({
            'subject_id': [1, 1, 1],
            'timestamp': ['12/1/22 1:00 am', '12/1/22 2:00 am', '12/2/22'],
            'event_type': ['A', 'A', 'A'],
            'metadata': [
                ExpandableDfDict({
                    'A_col': ['foo', 'bar', 'foo', 'bar', 'bax'],
                    'B_key': [
                        'a', 'a', 'a',
                        'b', 'b',
                    ],
                    'B_val': [
                        1, 2, 3,
                        4, 5,
                    ],
                }),
                ExpandableDfDict({}),
                ExpandableDfDict({}),
            ],
        })

        config = EventStreamDatasetConfig.from_simple_args(['A_col', ('B_key', 'B_val')])
        E = EventStreamDataset(events_df=events_df, config=config)
        E.split_subjects = {'train': {1}}
        E.preprocess_metadata()

        data_config = EventStreamPytorchDatasetConfig(
            do_normalize_log_inter_event_times = False,
            max_seq_len = 4,
        )

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
            'binary': [True, False, True, False],
            'multi_class_int': [0, 1, 2, 4],
            'multi_class_cat': pd.Series(['a', 'b', 'a', 'z'], dtype='category'),
            'regression': [1.2, 3.2, 1.5, 1.],
        })

        pyd = EventStreamPytorchDataset(E, data_config, task_df=task_df, split='train')

        want_tasks = ['binary', 'multi_class_int', 'multi_class_cat', 'regression']
        want_task_types = {
            'binary': 'binary_classification',
            'multi_class_int': 'multi_class_classification',
            'multi_class_cat': 'multi_class_classification',
            'regression': 'regression',
        }
        want_task_vocabs = {
            'binary': [False, True],
            'multi_class_int': [0, 1, 2, 3, 4],
            'multi_class_cat': ['a', 'b', 'z'],
        }

        want_task_df = pd.DataFrame({
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
            'binary': [1., 0., 1., 0.],
            'multi_class_int': [0, 1, 2, 4],
            'multi_class_cat': [0, 1, 0, 2],
            'regression': [1.2, 3.2, 1.5, 1.],
        })

        self.assertEqual(want_tasks, pyd.tasks)
        self.assertEqual(want_task_types, pyd.task_types)
        self.assertNestedDictEqual(want_task_vocabs, pyd.task_vocabs)
        self.assertEqual(want_task_df, pyd.task_df)

    def test_get_item(self):
        """`EventStreamPytorchDataset.__get_item__(i)` should retreve and correctly represent item `i`."""
        subjects_df = pd.DataFrame({
                'buzz': ['foo', 'bar'],
                'dob': [
                    pd.to_datetime('12/1/21'),
                    pd.to_datetime('12/1/20'),
                ]
            },
            index=pd.Index([1, 2], name='subject_id')
        )

        events_df = pd.DataFrame({
            'subject_id': [1, 1, 1, 1, 2, 2],
            'timestamp': [
                '12/1/22 12:00 a.m.',
                '12/2/22 2:00 p.m.',
                '12/3/22 10:00 a.m.',
                '12/4/22 11:00 p.m.',
                '12/1/22 15:00',
                '12/2/22 2:00',
            ],
            'event_type': ['A', 'B', 'A', 'A', 'A', 'B'],
            'metadata': [
                ExpandableDfDict({'A_col': ['foo']}),
                ExpandableDfDict({
                    'B_key': [
                        'a', 'a', 'a',
                        'b', 'b',
                    ],
                    'B_val': [
                        1, 2, 3,
                        4, 5,
                    ],
                }),
                ExpandableDfDict({'A_col': ['bar']}),
                ExpandableDfDict({'A_col': ['foo']}),
                ExpandableDfDict({'A_col': ['foo']}),
                ExpandableDfDict({'B_key': ['a', 'b'], 'B_val': [1, 5]}),
            ],
        })

        config = EventStreamDatasetConfig.from_simple_args(
            dynamic_measurement_columns=['A_col', ('B_key', 'B_val')],
            static_measurement_columns=['buzz'],
            time_dependent_measurement_columns=[
                ('time_of_day', TimeOfDayFunctor()), ('age', AgeFunctor('dob'))
            ],
        )
        E = EventStreamDataset(events_df=events_df, subjects_df=subjects_df, config=config)
        E.split_subjects = {'train': {1, 2}}
        E.preprocess_metadata()
        # Vocab is {
        #   'A_col': ['UNK', 'foo', 'bar'], 'B_key': ['UNK', 'a', 'b'],
        #   'buzz': ['UNK', 'foo', 'bar'],
        #   'time_of_day': ['UNK', 'EARLY_AM', 'PM', 'AM', 'LATE_PM'], 'age': None
        # }

        # With no truncation
        data_config = EventStreamPytorchDatasetConfig(
            do_normalize_log_inter_event_times = False,
            max_seq_len = 4,
        )

        # want_measurement_vocab_offsets = {
        #     'event_type': 1,
        #     'B_key': 3, 'A_col': 6,
        #     'age': 9, 'time_of_day': 10,
        #     'buzz': 15,
        # }
        # want_measurements_idxmap = {
        #     'event_type': 1,
        #     'B_key': 2, 'A_col': 3,
        #     'age': 4, 'time_of_day': 5,
        #     'buzz': 6,
        # }
        # Event times:
        #        '12/1/22 12:00 a.m.',
        #        '12/2/22 2:00 p.m.',
        #        '12/3/22 10:00 a.m.',
        #        '12/4/22 11:00 p.m.',
        #        '12/1/22 15:00',
        #        '12/2/22 2:00',

        pyd = EventStreamPytorchDataset(E, data_config, split='train')
        self.assertEqual([1, 2], pyd.subject_ids)

        out = pyd._seeded_getitem_from_range(subject_id=1, seed=1)
        out['time'] = list(out['time']) # comparing numpy arrays is harder...
        # Comparing full granularity floats is harder...
        rnd = lambda ragged_list: [list(map(lambda x: round(x, 7), l)) for l in ragged_list]
        fillna = lambda ragged_list: [list(map(lambda x: x if not np.isnan(x) else None, l)) for l in ragged_list]
        out['dynamic_values'] = fillna(rnd(out['dynamic_values']))

        want_subj_event_ages = [
            [1., 1 + 1/365 + 14/(24*365), 1 + 2/365 + 10/(24*365), 1 + 3/365 + 23/(24*365)],
            [2 + 15/(24*365), 2 + 1/365 + 2/(24*365)]
        ]

        want_out = {
            'time': [0., (24 + 14)*60., (2*24 + 10)*60., (3*24 + 23)*60.],
            'static_indices': [16],
            'static_measurement_indices': [6],
            'dynamic_indices': [
                [1, 7, 9, 11],
                [2, 4, 4, 4, 5, 5, 9, 12],
                [1, 8, 9, 13],
                [1, 7, 9, 14],
            ],
            'dynamic_values': [
                [np.NaN, np.NaN, want_subj_event_ages[0][0], np.NaN],
                [np.NaN, 1., 2., 3., 4., 5., want_subj_event_ages[0][1], np.NaN],
                [np.NaN, np.NaN, want_subj_event_ages[0][2], np.NaN],
                [np.NaN, np.NaN, want_subj_event_ages[0][3], np.NaN],
            ],
            'dynamic_measurement_indices': [
                [1, 3, 4, 5],
                [1, 2, 2, 2, 2, 2, 4, 5],
                [1, 3, 4, 5],
                [1, 3, 4, 5],
            ],
        }
        want_out['dynamic_values'] = fillna(rnd(want_out['dynamic_values']))

        self.maxDiff = None
        self.assertNestedDictEqual(want_out, out)

        out = pyd._seeded_getitem_from_range(subject_id=2, seed=1)
        out['time'] = list(out['time']) # comparing numpy arrays is harder...
        want_out = {
            'time': [0., 11*60.],
            'static_indices': [17],
            'static_measurement_indices': [6],
            'dynamic_indices': [
                [1, 7, 9, 12],
                [2, 4, 5, 9, 11],
            ],
            'dynamic_values': [
                [np.NaN, np.NaN, want_subj_event_ages[1][0], np.NaN],
                [np.NaN, 1., 5., want_subj_event_ages[1][1], np.NaN],
            ],
            'dynamic_measurement_indices': [
                [1, 3, 4, 5],
                [1, 2, 2, 4, 5],
            ],
        }

        out['dynamic_values'] = fillna(rnd(out['dynamic_values']))
        want_out['dynamic_values'] = fillna(rnd(want_out['dynamic_values']))

        self.assertEqual(want_out, out)

        out = pyd._seeded_getitem_from_range(
            subject_id=1, seed=1, start_time=pd.to_datetime('12/2/22 1:00 am')
        )
        out['time'] = list(out['time']) # comparing numpy arrays is harder...
        want_out = {
            'time': [0, (24 - 4)*60., (2*24 + 9)*60.],
            'static_indices': [16],
            'static_measurement_indices': [6],
            'dynamic_indices': [
                [2, 4, 4, 4, 5, 5, 9, 12],
                [1, 8, 9, 13],
                [1, 7, 9, 14],
            ],
            'dynamic_values': [
                [np.NaN, 1., 2., 3., 4., 5., want_subj_event_ages[0][1], np.NaN],
                [np.NaN, np.NaN, want_subj_event_ages[0][2], np.NaN],
                [np.NaN, np.NaN, want_subj_event_ages[0][3], np.NaN],
            ],
            'dynamic_measurement_indices': [
                [1, 2, 2, 2, 2, 2, 4, 5],
                [1, 3, 4, 5],
                [1, 3, 4, 5],
            ],
        }

        out['dynamic_values'] = fillna(rnd(out['dynamic_values']))
        want_out['dynamic_values'] = fillna(rnd(want_out['dynamic_values']))

        self.assertEqual(want_out, out)

        out = pyd._seeded_getitem_from_range(
            subject_id=1, seed=1, end_time=pd.to_datetime('12/2/22 1:00 am')
        )
        out['time'] = list(out['time']) # comparing numpy arrays is harder...
        want_out = {
            'time': [0.],
            'static_indices': [16],
            'static_measurement_indices': [6],
            'dynamic_indices': [
                [1, 7, 9, 11],
            ],
            'dynamic_values': [
                [np.NaN, np.NaN, want_subj_event_ages[0][0], np.NaN],
            ],
            'dynamic_measurement_indices': [
                [1, 3, 4, 5],
            ],
        }

        out['dynamic_values'] = fillna(rnd(out['dynamic_values']))
        want_out['dynamic_values'] = fillna(rnd(want_out['dynamic_values']))

        self.assertEqual(want_out, out)

        # With truncation
        data_config = EventStreamPytorchDatasetConfig(
            do_normalize_log_inter_event_times = False,
            max_seq_len = 2,
        )

        pyd = EventStreamPytorchDataset(E, data_config, split='train')

        seed=1
        np.random.seed(seed)
        start_idx = np.random.choice(4 - 2)
        end_idx = start_idx + 2

        self.assertEqual([1, 2], pyd.subject_ids)
        out = pyd._seeded_getitem_from_range(subject_id=pyd.subject_ids[0], seed=seed)
        out['time'] = list(out['time']) # comparing numpy arrays is harder...
        want_out = {
            'time': [0., (24 + 14)*60., (2*24 + 10)*60., (3*24 + 23)*60.][start_idx:end_idx],
            'static_indices': [16],
            'static_measurement_indices': [6],
            'dynamic_indices': [
                [1, 7, 9, 11],
                [2, 4, 4, 4, 5, 5, 9, 12],
                [1, 8, 9, 13],
                [1, 7, 9, 14],
            ][start_idx:end_idx],
            'dynamic_values': [
                [np.NaN, np.NaN, want_subj_event_ages[0][0], np.NaN],
                [np.NaN, 1., 2., 3., 4., 5., want_subj_event_ages[0][1], np.NaN],
                [np.NaN, np.NaN, want_subj_event_ages[0][2], np.NaN],
                [np.NaN, np.NaN, want_subj_event_ages[0][3], np.NaN],
            ][start_idx:end_idx],
            'dynamic_measurement_indices': [
                [1, 3, 4, 5],
                [1, 2, 2, 2, 2, 2, 4, 5],
                [1, 3, 4, 5],
                [1, 3, 4, 5],
            ][start_idx:end_idx],
        }

        out['dynamic_values'] = fillna(rnd(out['dynamic_values']))
        want_out['dynamic_values'] = fillna(rnd(want_out['dynamic_values']))

        self.assertEqual(want_out, out)

        # With TTE Normalization
        events_df = pd.DataFrame({
            'subject_id': [1, 1, 1, 1, 2, 2],
            'timestamp': [
                '12/1/22 12:00 am',
                '12/1/22 12:10 am',
                '12/1/22 12:30 am',
                '12/1/22 1:00 am',
                '12/1/22',
                '12/2/22',
            ],
            'event_type': ['A', 'B', 'A', 'A', 'A', 'B'],
            'metadata': [
                ExpandableDfDict({'A_col': ['foo']}),
                ExpandableDfDict({
                    'B_key': [
                        'a', 'a', 'a',
                        'b', 'b',
                    ],
                    'B_val': [
                        1, 2, 3,
                        4, 5,
                    ],
                }),
                ExpandableDfDict({'A_col': ['bar']}),
                ExpandableDfDict({'A_col': ['foo']}),
                ExpandableDfDict({'A_col': ['foo']}),
                ExpandableDfDict({'B_key': ['a', 'b'], 'B_val': [1, 5]}),
            ],
        })

        config = EventStreamDatasetConfig.from_simple_args(['A_col', ('B_key', 'B_val')])
        E = EventStreamDataset(events_df=events_df, config=config)
        E.split_subjects = {'train': {1}}
        E.preprocess_metadata()

        data_config = EventStreamPytorchDatasetConfig(
            do_normalize_log_inter_event_times = True,
            max_seq_len = 4,
        )

        pyd = EventStreamPytorchDataset(E, data_config, split='train')

        raw_TTE = np.array([10, 20, 30])
        mean_TTE = np.mean(np.log(raw_TTE + 1))
        std_TTE = np.std(np.log(raw_TTE + 1), ddof=1)

        normalized_time = np.exp((np.log(raw_TTE + 1) - mean_TTE)/std_TTE)
        normalized_time = np.array([0] + list(normalized_time))
        normalized_time = normalized_time.cumsum()

        self.assertEqual([1], pyd.subject_ids)
        out = pyd._seeded_getitem_from_range(subject_id=pyd.subject_ids[0], seed=1)
        out['time'] = list(out['time']) # comparing numpy arrays is harder...
        want_out = {
            'time': list(normalized_time),
            'dynamic_indices': [
                [1, 7],
                [2, 4, 4, 4, 5, 5],
                [1, 8],
                [1, 7],
            ],
            'dynamic_values': [
                [np.NaN, np.NaN],
                [np.NaN, 1, 2, 3, 4, 5],
                [np.NaN, np.NaN],
                [np.NaN, np.NaN],
            ],
            'dynamic_measurement_indices': [
                [1, 3],
                [1, 2, 2, 2, 2, 2],
                [1, 3],
                [1, 3],
            ],
        }

        self.assertEqual(want_out, out)

    def test_dynamic_collate_fn(self):
        """collate_fn should appropriately combine two batches of ragged tensors."""
        events_df = pd.DataFrame({
            'subject_id': [1, 1, 1, 1, 2, 2],
            'timestamp': ['12/1/22', '12/2/22', '12/3/22', '12/4/22', '12/1/22', '12/2/22'],
            'event_type': ['A', 'B', 'A', 'A', 'A', 'B'],
            'metadata': [
                ExpandableDfDict({'A_col': ['foo']}),
                ExpandableDfDict({
                    'B_key': [
                        'a', 'a', 'a',
                        'b', 'b',
                    ],
                    'B_val': [
                        1, 2, 3,
                        4, 5,
                    ],
                }),
                ExpandableDfDict({'A_col': ['bar']}),
                ExpandableDfDict({'A_col': ['foo']}),
                ExpandableDfDict({'A_col': ['foo']}),
                ExpandableDfDict({'B_key': ['a', 'b'], 'B_val': [1, 5]}),
            ],
        })

        config = EventStreamDatasetConfig.from_simple_args(['A_col', ('B_key', 'B_val')])
        E = EventStreamDataset(events_df=events_df, config=config)
        E.split_subjects = {'train': {1, 2}}
        E.preprocess_metadata()
        # Vocab is {'A': ['UNK', 'foo', 'bar'], 'B': ['UNK', 'a', 'b']}

        data_config = EventStreamPytorchDatasetConfig(
            do_normalize_log_inter_event_times = False,
            seq_padding_side='right',
            max_seq_len = 10,
        )

        pyd = EventStreamPytorchDataset(E, data_config, split='train')

        subj_1 = {
            'time': [0., 24*60., 2*24*60., 3*24*60.],
            'dynamic_indices': [
                [1, 4],
                [2, 7, 7, 7, 8, 8],
                [1, 5],
                [1, 4],
            ],
            'dynamic_values': [
                [np.NaN, np.NaN],
                [np.NaN, 1, 2, 3, 4, 5],
                [np.NaN, np.NaN],
                [np.NaN, np.NaN],
            ],
            'dynamic_measurement_indices': [
                [1, 2],
                [1, 3, 3, 3, 3, 3],
                [1, 2],
                [1, 2],
            ],
        }
        subj_2 = {
            'time': [0., 5, 10],
            'dynamic_indices': [
                [1, 4, 3],
                [2, 7, 7, 7],
                [1, 5],
            ],
            'dynamic_values': [
                [np.NaN, np.NaN, np.NaN],
                [np.NaN, 8, 9, 10],
                [np.NaN, np.NaN],
            ],
            'dynamic_measurement_indices': [
                [1, 2, 2],
                [1, 3, 3, 3],
                [1, 2],
            ],
        }

        batches = [subj_1, subj_2]
        out = pyd.collate(batches)

        want_out = EventStreamPytorchBatch(**{
            'event_mask': torch.BoolTensor([
                [True, True, True, True], [True, True, True, False]
            ]),
            'dynamic_values_mask': torch.BoolTensor([
                [
                    [False, False, False, False, False, False],
                    [False, True, True, True, True, True],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                ], [
                    [False, False, False, False, False, False],
                    [False, True, True, True, False, False],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                ]
            ]),
            'time': torch.Tensor([
                [0., 24*60., 2*24*60., 3*24*60.], [0, 5, 10, 0]
            ]),
            'dynamic_indices': torch.LongTensor([
                [
                    [1, 4, 0, 0, 0, 0],
                    [2, 7, 7, 7, 8, 8],
                    [1, 5, 0, 0, 0, 0],
                    [1, 4, 0, 0, 0, 0],
                ], [
                    [1, 4, 3, 0, 0, 0],
                    [2, 7, 7, 7, 0, 0],
                    [1, 5, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ]),
            'dynamic_measurement_indices': torch.LongTensor([
                [
                    [1, 2, 0, 0, 0, 0],
                    [1, 3, 3, 3, 3, 3],
                    [1, 2, 0, 0, 0, 0],
                    [1, 2, 0, 0, 0, 0],
                ], [
                    [1, 2, 2, 0, 0, 0],
                    [1, 3, 3, 3, 0, 0],
                    [1, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ]),
            'dynamic_values': torch.nan_to_num(torch.Tensor([
                [
                    [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                    [np.NaN, 1, 2, 3, 4, 5],
                    [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                    [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                ], [
                    [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                    [np.NaN, 8, 9, 10, np.NaN, np.NaN],
                    [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                    [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                ]
            ]), 0)
        })

        self.assertNestedDictEqual(asdict(want_out), asdict(out))

        data_config = EventStreamPytorchDatasetConfig(
            do_normalize_log_inter_event_times = False,
            seq_padding_side='left',
            max_seq_len = 10,
        )

        pyd = EventStreamPytorchDataset(E, data_config, split='train')

        out = pyd.collate(batches)

        want_out = EventStreamPytorchBatch(**{
            'event_mask': torch.BoolTensor([
                [True, True, True, True], [False, True, True, True]
            ]),
            'dynamic_values_mask': torch.BoolTensor([
                [
                    [False, False, False, False, False, False],
                    [False, True, True, True, True, True],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                ], [
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [False, True, True, True, False, False],
                    [False, False, False, False, False, False],
                ]
            ]),
            'time': torch.Tensor([
                [0., 24*60., 2*24*60., 3*24*60.], [0, 0, 5, 10]
            ]),
            'dynamic_indices': torch.LongTensor([
                [
                    [1, 4, 0, 0, 0, 0],
                    [2, 7, 7, 7, 8, 8],
                    [1, 5, 0, 0, 0, 0],
                    [1, 4, 0, 0, 0, 0],
                ], [
                    [0, 0, 0, 0, 0, 0],
                    [1, 4, 3, 0, 0, 0],
                    [2, 7, 7, 7, 0, 0],
                    [1, 5, 0, 0, 0, 0],
                ]
            ]),
            'dynamic_measurement_indices': torch.LongTensor([
                [
                    [1, 2, 0, 0, 0, 0],
                    [1, 3, 3, 3, 3, 3],
                    [1, 2, 0, 0, 0, 0],
                    [1, 2, 0, 0, 0, 0],
                ], [
                    [0, 0, 0, 0, 0, 0],
                    [1, 2, 2, 0, 0, 0],
                    [1, 3, 3, 3, 0, 0],
                    [1, 2, 0, 0, 0, 0],
                ]
            ]),
            'dynamic_values': torch.nan_to_num(torch.Tensor([
                [
                    [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                    [np.NaN, 1, 2, 3, 4, 5],
                    [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                    [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                ], [
                    [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                    [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                    [np.NaN, 8, 9, 10, np.NaN, np.NaN],
                    [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                ]
            ]), 0)
        })

        self.assertNestedDictEqual(asdict(want_out), asdict(out))

    def test_collate_fn(self):
        """collate_fn should appropriately combine two batches of ragged tensors."""
        subjects_df = pd.DataFrame({
                'buzz': ['foo', 'bar'],
                'dob': [
                    pd.to_datetime('12/1/21'),
                    pd.to_datetime('12/1/20'),
                ]
            },
            index=pd.Index([1, 2], name='subject_id')
        )

        events_df = pd.DataFrame({
            'subject_id': [1, 1, 1, 1, 2, 2],
            'timestamp': [
                '12/1/22 12:00 a.m.',
                '12/2/22 2:00 p.m.',
                '12/3/22 10:00 a.m.',
                '12/4/22 11:00 p.m.',
                '12/1/22 15:00',
                '12/2/22 2:00',
            ],
            'event_type': ['A', 'B', 'A', 'A', 'A', 'B'],
            'metadata': [
                ExpandableDfDict({'A_col': ['foo']}),
                ExpandableDfDict({
                    'B_key': [
                        'a', 'a', 'a',
                        'b', 'b',
                    ],
                    'B_val': [
                        1, 2, 3,
                        4, 5,
                    ],
                }),
                ExpandableDfDict({'A_col': ['bar']}),
                ExpandableDfDict({'A_col': ['foo']}),
                ExpandableDfDict({'A_col': ['foo']}),
                ExpandableDfDict({'B_key': ['a', 'b'], 'B_val': [1, 5]}),
            ],
        })

        config = EventStreamDatasetConfig.from_simple_args(
            dynamic_measurement_columns=['A_col', ('B_key', 'B_val')],
            static_measurement_columns=['buzz'],
            time_dependent_measurement_columns=[
                ('time_of_day', TimeOfDayFunctor()), ('age', AgeFunctor('dob'))
            ],
        )

        E = EventStreamDataset(events_df=events_df, subjects_df=subjects_df, config=config)
        E.split_subjects = {'train': {1, 2}}
        E.preprocess_metadata()
        # Vocab is {
        #   'A_col': ['UNK', 'foo', 'bar'], 'B_key': ['UNK', 'a', 'b'],
        #   'buzz': ['UNK', 'foo', 'bar'],
        #   'time_of_day': ['UNK', 'EARLY_AM', 'PM', 'AM', 'LATE_PM'],
        # }

        data_config = EventStreamPytorchDatasetConfig(
            do_normalize_log_inter_event_times = False,
            max_seq_len = 4,
        )

        pyd = EventStreamPytorchDataset(E, data_config, split='train')

        want_subj_event_ages = [
            [1., 1 + 1/365 + 14/(24*365), 1 + 2/365 + 10/(24*365), 1 + 3/365 + 23/(24*365)],
            [2 + 15/(24*365), 2 + 1/365 + 2/(24*365)]
        ]
        subj_1 = {
            'time': [0., (24 + 14)*60., (2*24 + 10)*60., (3*24 + 23)*60.],
            'static_indices': [16],
            'static_measurement_indices': [6],
            'dynamic_indices': [
                [1, 7, 9, 11],
                [2, 4, 4, 4, 5, 5, 9, 12],
                [1, 8, 9, 13],
                [1, 7, 9, 14],
            ],
            'dynamic_values': [
                [np.NaN, np.NaN, want_subj_event_ages[0][0], np.NaN],
                [np.NaN, 1., 2., 3., 4., 5., want_subj_event_ages[0][1], np.NaN],
                [np.NaN, np.NaN, want_subj_event_ages[0][2], np.NaN],
                [np.NaN, np.NaN, want_subj_event_ages[0][3], np.NaN],
            ],
            'dynamic_measurement_indices': [
                [1, 3, 4, 5],
                [1, 2, 2, 2, 2, 2, 4, 5],
                [1, 3, 4, 5],
                [1, 3, 4, 5],
            ],
        }
        subj_2 = {
            'time': [0., 11*60.],
            'static_indices': [17],
            'static_measurement_indices': [6],
            'dynamic_indices': [
                [1, 7, 9, 12],
                [2, 4, 5, 9, 11],
            ],
            'dynamic_values': [
                [np.NaN, np.NaN, want_subj_event_ages[1][0], np.NaN],
                [np.NaN, 1., 5., want_subj_event_ages[1][1], np.NaN],
            ],
            'dynamic_measurement_indices': [
                [1, 3, 4, 5],
                [1, 2, 2, 4, 5],
            ],
        }

        batches = [subj_1, subj_2]
        out = pyd.collate(batches)

        want_out = EventStreamPytorchBatch(**{
            'event_mask': torch.BoolTensor([
                [True, True, True, True], [True, True, False, False]
            ]),
            'dynamic_values_mask': torch.BoolTensor([
                [
                    [False, False, True, False, False, False, False, False],
                    [False, True, True, True, True, True, True, False],
                    [False, False, True, False, False, False, False, False],
                    [False, False, True, False, False, False, False, False],
                ], [
                    [False, False, True, False, False, False, False, False],
                    [False, True, True, True, False, False, False, False],
                    [False, False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False, False],
                ]
            ]),
            'time': torch.Tensor([
                [0., (24 + 14)*60., (2*24 + 10)*60., (3*24 + 23)*60.],
                [0., 11*60., 0., 0.],
            ]),
            'dynamic_indices': torch.LongTensor([
                [
                    [1, 7, 9, 11, 0, 0, 0, 0],
                    [2, 4, 4, 4, 5, 5, 9, 12],
                    [1, 8, 9, 13, 0, 0, 0, 0],
                    [1, 7, 9, 14, 0, 0, 0, 0],
                ], [
                    [1, 7, 9, 12,0, 0, 0, 0],
                    [2, 4, 5, 9, 11,0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]),
            'dynamic_measurement_indices': torch.LongTensor([
                [
                    [1, 3, 4, 5, 0, 0, 0, 0],
                    [1, 2, 2, 2, 2, 2, 4, 5],
                    [1, 3, 4, 5, 0, 0, 0, 0],
                    [1, 3, 4, 5, 0, 0, 0, 0],
                ], [
                    [1, 3, 4, 5, 0, 0, 0, 0],
                    [1, 2, 2, 4, 5, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]),
            'dynamic_values': torch.Tensor([
                [
                    [0, 0, want_subj_event_ages[0][0], 0, 0, 0, 0, 0],
                    [0, 1., 2., 3., 4., 5., want_subj_event_ages[0][1], 0],
                    [0, 0, want_subj_event_ages[0][2], 0, 0, 0, 0, 0],
                    [0, 0, want_subj_event_ages[0][3], 0, 0, 0, 0, 0],
                ], [
                    [0, 0, want_subj_event_ages[1][0], 0, 0, 0, 0, 0],
                    [0, 1., 5., want_subj_event_ages[1][1], 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]),
            'static_indices': torch.LongTensor([[16], [17]]),
            'static_measurement_indices': torch.LongTensor([[6], [6]]),
        })

        self.assertNestedDictEqual(asdict(want_out), asdict(out))

if __name__ == '__main__': unittest.main()

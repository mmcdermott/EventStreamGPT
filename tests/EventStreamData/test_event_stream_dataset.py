import sys
sys.path.append('../..')

from .test_config import ConfigComparisonsMixin

import copy, unittest, numpy as np, pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from EventStream.EventStreamData.config import EventStreamDatasetConfig, MeasurementConfig
from EventStream.EventStreamData.event_stream_dataset import EventStreamDataset
from EventStream.EventStreamData.types import (
    DataModality,
    TemporalityType,
)
from EventStream.EventStreamData.time_dependent_functor import (
    AgeFunctor,
    TimeOfDayFunctor,
)
from EventStream.EventStreamData.vocabulary import Vocabulary

class DummySklearn():
    """This is used to make fake model classes for testing outlier detection and such."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)

    def fit(self, vals: np.ndarray):
        vals = self.validate_and_squeeze(vals)
        self.mean = np.round(vals.mean(), 5) # To avoid precision issues.
        self.max = vals.max()
        self.min = vals.min()
        self.count = len(vals)

    @classmethod
    def validate_and_squeeze(cls, vals: np.ndarray) -> np.ndarray:
        assert len(vals.shape) == 2
        assert len(vals.squeeze(-1).shape) == 1
        return vals.squeeze(-1)

    def __eq__(self, other: Any) -> bool:
        return (type(self) is type(other)) and (vars(self) == vars(other))

    def __repr__(self) -> str:
        """We add a repr as it makes error messages easier to understand."""
        return f"{self.__class__.__name__}({vars(self)})"

class TestEventStreamDataset(ConfigComparisonsMixin, unittest.TestCase):
    """Tests the `EventStreamDataset` class."""
    def test_to_events_and_metadata(self):
        df = pd.DataFrame({
            'alt_subject_col': [1, 2, 3, 4],
            'alt_time_col': ['12/1/22', '12/2/22', '12/3/22', '12/1/22'],
            'metadata_col_1': ['foo', 'bar', 'foo', 'bar'],
            'metadata_col_2': ['baz', 'biz', 'buz', 'bez'],
        })

        want_events_df = pd.DataFrame({
            'subject_id': [1, 2, 3, 4],
            'timestamp': ['12/1/22', '12/2/22', '12/3/22', '12/1/22'],
            'event_type': ['A', 'A', 'A', 'A'],
        }, index = pd.Index([0, 1, 2, 3], name='event_id'))
        want_metadata_df = pd.DataFrame({
            'event_id': [0, 1, 2, 3],
            'event_type': ['A', 'A', 'A', 'A'],
            'subject_id': [1, 2, 3, 4],
            'metadata_col_1': ['foo', 'bar', 'foo', 'bar'],
            'metadata_col_2': ['baz', 'biz', 'buz', 'bez'],
        }, index = pd.Index([0, 1, 2, 3], name='metadata_id'))

        got_events_df, got_metadata_df = EventStreamDataset.to_events_and_metadata(
            df, event_type='A', subject_col='alt_subject_col',
            time_col='alt_time_col', metadata_cols=['metadata_col_1', 'metadata_col_2']
        )

        self.assertEqual(want_events_df, got_events_df)
        self.assertEqual(want_metadata_df, got_metadata_df)

    def test_infer_bounds_from_units_inplace(self):
        input_measurement_metadata = pd.DataFrame({
            'unit': ['%', 'foo', None, 'percent', 'PERCENT'],
            'drop_lower_bound': [-1, 5, None, 0.2, None],
            'drop_lower_bound_inclusive': [True, True, None, True, None],
            'censor_lower_bound': [0.1, 10, None, 0.3, None],
            'censor_lower_bound_inclusive': [False, False, None, True, None],
            'drop_upper_bound': [0.9, None, None, 1.2, np.NaN],
            'drop_upper_bound_inclusive': [False, False, None, False, None],
        }, index=['a', 'b', 'c', 'd', 'e'])

        class EventStreamDatasetDerived(EventStreamDataset):
            UNIT_BOUNDS = {
                # (unit strings): [lower, lower_inclusive, upper, upper_inclusive],
                ('%', 'percent', 'PERCENT'): [0, True, 1, True],
            }

        EventStreamDatasetDerived.infer_bounds_from_units_inplace(input_measurement_metadata)

        want_metadata = pd.DataFrame({
            'unit': ['%', 'foo', None, 'percent', 'PERCENT'],
            'drop_lower_bound': [0, 5, None, 0.2, 0],
            'drop_lower_bound_inclusive': [True, True, None, True, True],
            'censor_lower_bound': [0.1, 10, None, 0.3, None],
            'censor_lower_bound_inclusive': [False, False, None, True, None],
            'drop_upper_bound': [0.9, None, None, 1, 1],
            'drop_upper_bound_inclusive': [False, False, None, True, True],
        }, index=['a', 'b', 'c', 'd', 'e'])

        self.assertEqual(want_metadata, input_measurement_metadata)

    def test_fit_metadata_model(self):
        """Tests `EventStreamDataset._fit_metadata_model`"""
        class DumbMetadataModel():
            def __init__(self, kwarg_1 = 0, kwarg_2 = 0):
                self.kwarg_1 = kwarg_1
                self.kwarg_2 = kwarg_2

            def fit(self, vals):
                self.mean = vals.mean()
                self.max = vals.max()
                self.min = vals.min()

        class EventStreamDatasetDerived(EventStreamDataset):
            METADATA_MODELS = {
                'test_model': DumbMetadataModel,
            }

        vals = pd.Series([1, 2, 3, np.NaN, 'foo', None])
        model_config = {'cls': 'test_model', 'kwarg_1': 2}
        got_model = EventStreamDatasetDerived._fit_metadata_model(vals, model_config)

        self.assertEqual(2, got_model.kwarg_1)
        self.assertEqual(0, got_model.kwarg_2)
        self.assertEqual(1, got_model.min)
        self.assertEqual(2, got_model.mean)
        self.assertEqual(3, got_model.max)

        got_model = EventStreamDatasetDerived._fit_metadata_model(pd.Series([], dtype=float), model_config)
        self.assertTrue(got_model is None)

        got_model = EventStreamDatasetDerived._fit_metadata_model(pd.Series([None, 'foo']), model_config)
        self.assertTrue(got_model is None)

        with self.assertRaises(AssertionError): EventStreamDatasetDerived._fit_metadata_model(vals, {})
        with self.assertRaises(AssertionError):
            EventStreamDatasetDerived._fit_metadata_model(vals, {'cls': 'not found'})

    def test_basic_construction(self):
        """Upon construction, `EventStreamDataset` should add event ids and sort the events."""
        # The input can take strings and be unsorted.
        events_df = pd.DataFrame({
            'subject_id': [2, 1, 1, 2, 2],
            'timestamp': ['12/3/22', '12/2/22', '12/1/22', '12/1/22', '12/1/22'],
            'event_type': ['C', 'B', 'A', 'A', 'A'],
        }, index=pd.Index([0, 1, 2, 3, 4], name='event_id'))

        metadata_df = pd.DataFrame({
            'C_col': ['Z', None, None, None, None, None],
            'B_col': [None, 2, None, None, None, None],
            'A_col': [None, None, 1, 3, 4, 5],
            'event_id': [0, 1, 2, 3, 4, 4]
        })

        E = EventStreamDataset(
            events_df=events_df,
            metadata_df=metadata_df,
            config=EventStreamDatasetConfig()
        )

        # It should know who its subjects are.
        self.assertFalse(E.has_static_measurements)
        self.assertEqual({1, 2}, E.subject_ids)

        # Unlike the input, the parsed df should have timestamps and be sorted.
        want_events_df = pd.DataFrame({
            'subject_id': [1, 1, 2, 2, 2],
            'timestamp': [
                pd.to_datetime('12/1/22'),
                pd.to_datetime('12/2/22'),
                pd.to_datetime('12/1/22'),
                pd.to_datetime('12/1/22'),
                pd.to_datetime('12/3/22'),
            ],
            'event_type': ['A', 'B', 'A', 'A', 'C'],
        }, index=pd.Index([2, 1, 3, 4, 0], name='event_id'))

        self.assertEqual(want_events_df, E.events_df)

        # `EventStreamDataset` should be able to extracted the associated metadata_df from the raw events_df.
        want_metadata_df = pd.DataFrame({
            'event_id': [2, 1, 3, 4, 4, 0],
            'event_type': ['A', 'B', 'A', 'A', 'A', 'C'],
            'subject_id': [1, 1, 2, 2, 2, 2],
            'A_col': [1, None, 3, 4, 5, None],
            'B_col': [None, 2, None, None, None, None],
            'C_col': [None, None, None, None, None, 'Z'],
        }).iloc[[5, 1, 0, 2, 3, 4]] # sort by event_id

        want_metadata_df.index=pd.Index([0, 1, 2, 3, 4, 5], name='metadata_id')
        self.assertEqual(want_metadata_df, E.joint_metadata_df)

        self.assertEqual({1, 2}, E.subject_ids)
        self.assertEqual(['A', 'B', 'C'], E.event_types)

        # `EventStreamDataset` should be able to filter and collapse a restricted metadata_df to non-nans.
        want_type_A_metadata_df = pd.DataFrame(
            {
                'event_id': [2, 3, 4, 4], 'event_type': ['A', 'A', 'A', 'A'], 'subject_id': [1, 2, 2, 2],
                'A_col': [1., 3., 4., 5.]
            }, index=pd.Index([2, 3, 4, 5], name='metadata_id'),
        )
        self.assertEqual(want_type_A_metadata_df, E.metadata_df(event_type='A'))
        self.assertEqual(want_type_A_metadata_df, E.metadata_df(event_types=['A']))

        want_type_B_metadata_df = pd.DataFrame(
            {'event_id': [1], 'event_type': ['B'], 'subject_id': [1], 'B_col': [2.]},
            index=pd.Index([1], name='metadata_id'),
        )
        self.assertEqual(want_type_B_metadata_df, E.metadata_df(event_type='B'))

        # With subjects_df
        subjects_df = pd.DataFrame(
            {'foo': [33] * 10, 'bar': ['UNK'] * 10}, index=pd.Index(list(range(10)), name='subject_id')
        )

        events_df = pd.DataFrame({
            'subject_id': [2, 1, 1, 2, 2],
            'timestamp': ['12/3/22', '12/2/22', '12/1/22', '12/1/22', '12/1/22'],
            'event_type': ['C', 'B', 'A', 'A', 'A'],
        }, index=pd.Index([0, 1, 2, 3, 4], name='event_id'))
        metadata_df = pd.DataFrame({
            'C_col': ['Z', None, None, None, None, None],
            'B_col': [None, 2, None, None, None, None],
            'A_col': [None, None, 1, 3, 4, 5],
            'event_id': [0, 1, 2, 3, 4, 4]
        })

        E = EventStreamDataset(
            events_df=events_df,
            metadata_df=metadata_df,
            subjects_df=subjects_df,
            config=EventStreamDatasetConfig()
        )

        # It still has no static measurements, as there are non in the config.
        self.assertFalse(E.has_static_measurements)

        # Now, though, its subjects should include those in subjects_df too.
        self.assertEqual(set(range(10)), E.subject_ids)
        # However, we should know how many events each subject has.
        self.assertEqual(
            {0: 0, 1: 2, 2: 3, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
            E.n_events_per_subject
        )

        E = EventStreamDataset(
            events_df=events_df,
            subjects_df=subjects_df,
            config=EventStreamDatasetConfig.from_simple_args(static_measurement_columns=['bar']),
        )
        # Now it should recognize it has static measurements
        self.assertTrue(E.has_static_measurements)

    def test_agg_by_time_type(self):
        """
        `EventStreamDataset` should be able to aggregate the `events_df` to be unique by subject, event_type,
        and time.
        """
        events_df = pd.DataFrame({
            'subject_id': [2, 1, 1, 2, 2],
            'timestamp': ['12/3/22', '12/2/22', '12/1/22', '12/1/22', '12/1/22'],
            'event_type': ['C', 'B', 'A', 'A', 'A'],
        }, index=pd.Index([0, 1, 2, 3, 4], name='event_id'))
        metadata_df = pd.DataFrame({
            'C_col': ['Z', None, None, None, None, None],
            'B_col': [None, 2, None, None, None, None],
            'A_col': [None, None, 1, 3, 4, 5],
            'event_id': [0, 1, 2, 3, 4, 4]
        }, index=pd.Index([0, 1, 2, 3, 4, 5], name='metadata_id'))

        E = EventStreamDataset(
            events_df=events_df,
            metadata_df=metadata_df,
            config=EventStreamDatasetConfig()
        )
        E.agg_by_time_type()

        want_events_df = pd.DataFrame({
            'subject_id': [1, 1, 2, 2],
            'timestamp': [
                pd.to_datetime('12/1/22'),
                pd.to_datetime('12/2/22'),
                pd.to_datetime('12/1/22'),
                pd.to_datetime('12/3/22'),
            ],
            'event_type': ['A', 'B', 'A', 'C'],
        }, index=pd.Index([0, 1, 2, 3], name='event_id'))
        self.assertEqual(want_events_df, E.events_df)

        # `EventStreamDataset` should be able to extracted the associated metadata_df from the raw events_df.
        want_metadata_df = pd.DataFrame({
            'event_id': [0, 1, 2, 2, 2, 3],
            'event_type': ['A', 'B', 'A', 'A', 'A', 'C'],
            'subject_id': [1, 1, 2, 2, 2, 2],
            'A_col': [1, None, 3, 4, 5, None],
            'B_col': [None, 2, None, None, None, None],
            'C_col': [None, None, None, None, None, 'Z'],
        }, index=pd.Index([2, 1, 3, 4, 5, 0], name='metadata_id'))
        self.assertEqual(want_metadata_df, E.joint_metadata_df)

    def test_split(self):
        """`EventStreamDataset` should be able to split the `events_df` into splits by `subject_id`."""
        events_df = pd.DataFrame({
            'subject_id': [1, 1, 2, 2],
            'timestamp': [
                pd.to_datetime('12/1/22'),
                pd.to_datetime('12/2/22'),
                pd.to_datetime('12/1/22'),
                pd.to_datetime('12/3/22'),
            ],
            'event_type': ['A', 'B', 'A', 'C'],
        })

        E = EventStreamDataset(events_df=events_df, config=EventStreamDatasetConfig())

        E.split(split_fracs=[0.5, 0.5], split_names=['train', 'held_out'], seed=1)
        split_subjects_seed_1_draw_1 = E.split_subjects

        E.split(split_fracs=[0.5, 0.5], seed=1)
        split_subjects_seed_1_draw_2 = E.split_subjects

        E.split(split_fracs=[0.5, 0.5], split_names=['train', 'held_out'], seed=2)
        split_subjects_seed_2_draw_1 = E.split_subjects

        self.assertEqual(
            split_subjects_seed_1_draw_1, split_subjects_seed_1_draw_2,
            msg="Splits with the same seed should be equal!"
        )
        self.assertNotEqual(
            split_subjects_seed_1_draw_1, split_subjects_seed_2_draw_1,
            msg="Splits with different seeds should not be equal!"
        )

        want_subj_1_events_df = pd.DataFrame({
            'subject_id': [1, 1],
            'timestamp': [pd.to_datetime('12/1/22'), pd.to_datetime('12/2/22')],
            'event_type': ['A', 'B'],
        }, index=pd.Index([0, 1], name='event_id'))

        want_subj_2_events_df = pd.DataFrame({
            'subject_id': [2, 2],
            'timestamp': [pd.to_datetime('12/1/22'), pd.to_datetime('12/3/22')],
            'event_type': ['A', 'C'],
        }, index=pd.Index([2, 3], name='event_id'))

        if E.split_subjects['train'] == set([1]):
            self.assertEqual(set([2]), E.split_subjects['held_out'])
            self.assertEqual(want_subj_1_events_df, E.train_events_df)
            self.assertEqual(want_subj_2_events_df, E.held_out_events_df)
        elif E.split_subjects['train'] == set([2]):
            self.assertEqual(set([1]), E.split_subjects['held_out'])
            self.assertEqual(want_subj_2_events_df, E.train_events_df)
            self.assertEqual(want_subj_1_events_df, E.held_out_events_df)
        else:
            raise AssertionError(
                "Split Subjects should be either {'train': {1}, 'held_out': {2}} or "
                f"{'train': {2}, 'held_out': {1}}! Got: {E.split_subjects}"
            )

        events_df = pd.DataFrame({
            'subject_id': [1, 1, 2, 2, 3],
            'timestamp': [
                pd.to_datetime('12/1/22'),
                pd.to_datetime('12/2/22'),
                pd.to_datetime('12/1/22'),
                pd.to_datetime('12/3/22'),
                pd.to_datetime('12/4/22'),
            ],
            'event_type': ['A', 'B', 'A', 'C', 'D'],
        })

        E = EventStreamDataset(events_df=events_df, config=EventStreamDatasetConfig())

        # When passing split_fracs that don't sum to 1, `EventStreamDataset` should add an auxiliary third
        # split that captures the missing fraction.
        E.split(split_fracs=[1/3, 1/3], seed=1)
        split_subjects_seed_1_draw_1 = E.split_subjects

        split_subjects_options = [
            {'train': {1}, 'tuning': {2}, 'held_out': {3}},
            {'train': {1}, 'tuning': {3}, 'held_out': {2}},
            {'train': {2}, 'tuning': {1}, 'held_out': {3}},
            {'train': {2}, 'tuning': {3}, 'held_out': {1}},
            {'train': {3}, 'tuning': {2}, 'held_out': {1}},
            {'train': {3}, 'tuning': {1}, 'held_out': {2}},
        ]

        self.assertIn(E.split_subjects, split_subjects_options)

        want_subj_3_events_df = pd.DataFrame({
            'subject_id': [3],
            'timestamp': [pd.to_datetime('12/4/22')],
            'event_type': ['D'],
        }, index=pd.Index([4], name='event_id'))

        want_dfs = {1: want_subj_1_events_df, 2: want_subj_2_events_df, 3: want_subj_3_events_df}

        for sp, subj_set in E.split_subjects.items():
            subj = list(subj_set)[0]
            want = want_dfs[subj]
            self.assertEqual(want, E._events_for_split(sp))

            if sp == 'train': self.assertEqual(want, E.train_events_df)
            if sp == 'tuning': self.assertEqual(want, E.tuning_events_df)
            if sp == 'held_out': self.assertEqual(want, E.held_out_events_df)

    def test_TTE_functions(self):
        """`EventStreamDataset` should be able to provide the inter-event-times and associated stats."""
        events_df = pd.DataFrame({
            'subject_id': [1, 1, 1, 2, 2, 2],
            'timestamp': [
                '12/1/22 12:00:00 am',
                '12/1/22 6:00:00 am',
                '12/1/22 11:00:00 am',
                '12/2/22 1:00:00 pm',
                '12/2/22 1:20:00 pm',
                '12/2/22 1:40:00 pm',
            ],
            'event_type': ['A', 'B', 'A', 'C', 'A', 'C'],
        })

        E = EventStreamDataset(events_df=events_df, config=EventStreamDatasetConfig())
        E.split_subjects = {'train': {1}, 'held_out': {2}}

        want_train_TTEs = pd.Series([6., 5.], index=pd.Index([1, 2], name='event_id'), name='timestamp')
        got_train_TTEs = E._inter_event_times_for_split('train', unit=pd.Timedelta(1, 'hour'))
        self.assertEqual(want_train_TTEs, got_train_TTEs)

        want_held_out_TTEs = pd.Series([20., 20.], index=pd.Index([4, 5], name='event_id'), name='timestamp')
        got_held_out_TTEs = E._inter_event_times_for_split('held_out', unit=pd.Timedelta(1, 'minute'))
        self.assertEqual(want_held_out_TTEs, got_held_out_TTEs)

        want_mean_log_inter_event_time = np.mean(np.log([60*6+1, 60*5+1]))

        # Use the sample standard deviation
        want_std_log_inter_event_time = np.std(np.log([60*6+1, 60*5+1]), ddof=1)

        self.assertEqual(want_mean_log_inter_event_time, E.train_mean_log_inter_event_time_min)
        self.assertEqual(want_std_log_inter_event_time, E.train_std_log_inter_event_time_min)

    def test_backup_restore_metadata_cols(self):
        """`EventStreamDataset` should be able to backup and restore numerical metadata columns."""
        events_df = pd.DataFrame({
            'subject_id': [1, 1],
            'timestamp': ['12/1/22', '12/2/22'],
            'event_type': ['A', 'B'],
        }, index=pd.Index([0, 1], name='event_id'))
        metadata_df = pd.DataFrame({
            'event_id': [0, 1, 1, 1],
            'A_col': ['foo', None, None, None],
            'B_key': [None, 'c', 'b', 'b'],
            'B_val': [None, 1, 3, 2],
        })

        E = EventStreamDataset(
            events_df=events_df,
            metadata_df=metadata_df,
            config=EventStreamDatasetConfig(
                measurement_configs = {
                    'B_key': MeasurementConfig(
                        temporality = TemporalityType.DYNAMIC,
                        modality = DataModality.MULTIVARIATE_REGRESSION,
                        values_column = 'B_val',
                    ),
                },
            ),
        )

        # We directly set internal parameters here to test only the desired portions.
        E.split_subjects = {'train': {1}, 'held_out': {2}}

        want_metadata_df = pd.DataFrame({
            'event_id': [0, 1, 1, 1],
            'event_type': ['A', 'B', 'B', 'B'],
            'subject_id': [1, 1, 1, 1],
            'A_col': ['foo', None, None, None],
            'B_key': [None, 'c', 'b', 'b'],
            'B_val': [None, 1, 3, 2],
        }, index=pd.Index([0, 1, 2, 3], name='metadata_id'))
        self.assertEqual(want_metadata_df, E.joint_metadata_df)

        E.backup_numerical_metadata_columns()

        want_backed_up_metadata_df = pd.DataFrame({
            'event_id': [0, 1, 1, 1],
            'event_type': ['A', 'B', 'B', 'B'],
            'subject_id': [1, 1, 1, 1],
            'A_col': ['foo', None, None, None],
            'B_key': [None, 'c', 'b', 'b'],
            'B_val': [None, 1, 3, 2],
            '__backup_B_key': [None, 'c', 'b', 'b'],
            '__backup_B_val': [None, 1, 3, 2],
        }, index=pd.Index([0, 1, 2, 3], name='metadata_id'))
        self.assertEqual(want_backed_up_metadata_df, E.joint_metadata_df)

        E.joint_metadata_df['B_key'] = [1, 2, 1, 2]
        E.joint_metadata_df['B_val'] = ['a', 'b', 'c', 'd']

        E.restore_numerical_metadata_columns()
        self.assertEqual(want_metadata_df, E.joint_metadata_df)

    def test_add_time_dependent_columns(self):
        events_df = pd.DataFrame({
            'subject_id': [1, 1, 1, 2, 2],
            'timestamp': [
                '1/1/00 12:00a.m.', '1/2/00 7:00 a.m.', '1/1/01 1:00pm', '1/1/2000 10:00pm',
                '3/1/2010 12:00am'
            ],
            'event_type': ['A', 'B', 'A', 'B', 'C'],
        })

        subjects_df = pd.DataFrame({
            'dob1': ['1/1/1990', '1/1/1980'], 'dob2': ['1/1/1970', '1/1/1960'],
        }, index=pd.Index([1, 2], name='subject_id'))
        subjects_df['dob1'] = pd.to_datetime(subjects_df.dob1)
        subjects_df['dob2'] = pd.to_datetime(subjects_df.dob2)

        config = EventStreamDatasetConfig.from_simple_args(
            time_dependent_measurement_columns = [
                ('age1', AgeFunctor('dob1')), ('age2', AgeFunctor('dob2')),
                ('time_of_day', TimeOfDayFunctor()),
            ]
        )

        E = EventStreamDataset(events_df=events_df, subjects_df=subjects_df, config=config)

        want_events_df = pd.DataFrame({
            'subject_id': [1, 1, 1, 2, 2],
            'timestamp': [
                pd.to_datetime('1/1/00 12:00:00 a.m.'),
                pd.to_datetime('1/2/00 7:00:00 a.m.'),
                pd.to_datetime('1/1/01 1:00:00 p.m.'),
                pd.to_datetime('1/1/2000 10:00:00 p.m.'),
                pd.to_datetime('3/1/2010 12:00:00 a.m.'),
            ],
            'event_type': ['A', 'B', 'A', 'B', 'C'],
        }, index = pd.Index([0, 1, 2, 3, 4], name='event_id'))
        self.assertEqual(
            want_events_df, E.events_df, "Before adding time dependent columns, events_df should be normal"
        )

        # Leap years happened in 2008, 2004, 2000, 1996, 1992, 1988, 1984, 1980, 1976, 1972, 1968, 1964, 1960
        want_age1 = [
            10 + 2/365,
            10 + 3/365 + 7/(24*365),
            11 + 3/365 + 13/(24*365),
            20 + 5/365 + 22/(24*365),
            30 + 8/365 + (31 + 28)/365,
        ]
        want_age2 = [
            30 + 7/365,
            30 + 8/365 + 7/(24*365),
            31 + 8/365 + 13/(24*365),
            40 + 10/365 + 22/(24*365),
            50 + 13/365 + (31 + 28)/365
        ]

        E.add_time_dependent_columns()
        want_events_df = pd.DataFrame({
            'subject_id': [1, 1, 1, 2, 2],
            'timestamp': [
                pd.to_datetime('1/1/00 12:00 a.m.'),
                pd.to_datetime('1/2/00 7:00 a.m.'),
                pd.to_datetime('1/1/01 1:00 p.m.'),
                pd.to_datetime('1/1/2000 10:00 p.m.'),
                pd.to_datetime('3/1/2010 12:00 a.m.'),
            ],
            'event_type': ['A', 'B', 'A', 'B', 'C'],
            'age1': want_age1,
            'age2': want_age2,
            'time_of_day': ['EARLY_AM', 'AM', 'PM', 'LATE_PM', 'EARLY_AM'],
        }, index = pd.Index([0, 1, 2, 3, 4], name='event_id'))
        self.assertEqual(want_events_df, E.events_df)

    def test_measurement_configs(self):
        events_df = pd.DataFrame({
            'subject_id': [1], 'timestamp': ['12/1/22'], 'event_type': ['A'],
        })
        config = EventStreamDatasetConfig.from_simple_args(['c1', ('c2_key', 'c2_val')])
        E = EventStreamDataset(events_df=events_df, config=config)
        self.assertFalse(E.metadata_is_fit)
        self.assertEqual(E.passed_measurement_configs, E.measurement_configs)
        self.assertEqual({}, E.inferred_measurement_configs)

        E.metadata_is_fit = True
        self.assertEqual({}, E.measurement_configs)

    def test_metadata_columns(self):
        events_df = pd.DataFrame({
            'subject_id': [1], 'timestamp': ['12/1/22'], 'event_type': ['A'],
        })
        config = EventStreamDatasetConfig.from_simple_args(['c1', 'drop', ('c2_key', 'c2_val')])
        config.measurement_configs['drop'].modality = 'dropped'

        E = EventStreamDataset(events_df=events_df, config=config)
        self.assertEqual(['c1', 'c2_key'], E.measurements)

    def test_numerical_metadata_columns(self):
        events_df = pd.DataFrame({
            'subject_id': [1], 'timestamp': ['12/1/22'], 'event_type': ['A'],
        })
        config = EventStreamDatasetConfig.from_simple_args(['c1', 'drop', ('c2_key', 'c2_val')])
        config.measurement_configs['drop'].modality = 'dropped'

        E = EventStreamDataset(events_df=events_df, config=config)
        self.assertEqual([('c2_key', 'c2_val')], E.dynamic_numerical_columns)
        self.assertEqual([], E.time_dependent_numerical_columns)

    def test_fit_categorical_metadata_cutoff_params(self):
        config = EventStreamDatasetConfig.from_simple_args(
            dynamic_measurement_columns=['A_col', 'B_col', 'C_col', 'D_col'],
            static_measurement_columns=['static1', 'static2'],
            time_dependent_measurement_columns=[('time_of_day', TimeOfDayFunctor())],
            min_valid_column_observations = 5,
            min_valid_vocab_element_observations = 2/5,
        )

        # static1 will persist, static2 won't.
        subjects_df = pd.DataFrame({
            'static1': ['i', 'ii', 'iii', 'i', 'i', 'ii', 'iii', 'ii', 'i', 'ii', 'iii', 'ii', 'i', 'ii'],
            'static2': ['I', 'II', None, None, None, None, 'VII'] * 2,
        }, index=pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], name='subject_id'))

        events_df = pd.DataFrame({
            'subject_id': [1, 2, 3, 4, 5, 6, 7],
            'timestamp': [
                '12/1/22 12:00 a.m.', # EARLY_AM
                '12/2/22 2:00 a.m.',  # EARLY_AM
                '12/2/22 11:00 a.m.', # AM
                '12/4/22 12:00 p.m.', # PM
                '12/3/22 1:00 p.m.',  # PM
                '12/1/22 1:30 a.m.', # EARLY_AM
                '1/14/22 3:30 p.m.', # PM
            ],
            'event_type': ['A', 'B', 'A', 'C', 'B', 'A', 'Z'],
        }, index=pd.Index([0, 1, 2, 3, 4, 5, 6], name='event_id'))

        # A_col overall will be observed only on 4 of the 6 possible events, so will fail
        # `min_valid_column_observations`.
        # B_col will occur on 5 of the 6 possible events, and will have 1 vocab element occur once (less than
        # 2/5 of the events), one vocab element occur twice (exactly 2/5 of the events), and one occur thrice
        # (greater than 2/5 of the events.
        # C_col will occur on all 6 events and will have two vocab elements occuring 3x each.
        metadata_df = pd.DataFrame({
            'A_col': ['a', 'b', 'a', 'a', None, None, 'a', 'b', 'a', 'b', 'a'],
            'B_col': ['foo', 'bar', 'bar', 'baz' ,'baz', None, 'foo', 'bar', 'baz', 'foo', 'bar'],
            'C_col': ['1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1'],
            'D_col': ['z', 'y', 'x', 'w', 'v', 'u', None, None, None, None, None],
            'event_id': [0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6],
        })
        E = EventStreamDataset(
            events_df=events_df,
            metadata_df=metadata_df,
            subjects_df=subjects_df,
            config=config
        )

        E.split_subjects = {'train': {1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13}, 'held_out': {7, 14}}

        self.assertEqual(E.inferred_measurement_configs, {})

        E.add_time_dependent_columns()

        E.fit_metadata()

        want_inferred_measurement_configs = {
            'static1': MeasurementConfig(
                name = 'static1',
                temporality = TemporalityType.STATIC,
                modality = DataModality.SINGLE_LABEL_CLASSIFICATION,
                observation_frequency = 1,
                vocabulary = Vocabulary(
                    ['UNK', 'i', 'ii'],
                    obs_frequencies = np.array([2/12, 5/12, 5/12]),
                ),
            ),
            'static2': MeasurementConfig(
                name = 'static2',
                temporality = TemporalityType.STATIC,
                modality = DataModality.DROPPED,
                observation_frequency = 2/6,
            ),
            'time_of_day': MeasurementConfig(
                name = 'time_of_day',
                temporality = TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                functor = TimeOfDayFunctor(),
                modality = DataModality.SINGLE_LABEL_CLASSIFICATION,
                observation_frequency = 1,
                vocabulary = Vocabulary(
                    ['UNK', 'EARLY_AM'], obs_frequencies = np.array([3/6, 3/6]),
                ),
            ),
            'A_col': MeasurementConfig(
                name = 'A_col',
                temporality = TemporalityType.DYNAMIC,
                modality='dropped', observation_frequency=4/6
            ),
            'B_col': MeasurementConfig(
                temporality = TemporalityType.DYNAMIC,
                name = 'B_col',
                modality='multi_label_classification',
                observation_frequency=5/6,
                vocabulary=Vocabulary(['UNK', 'bar', 'baz'], obs_frequencies=np.array([1/5, 2/5, 2/5])),
            ),
            'C_col': MeasurementConfig(
                name = 'C_col',
                temporality = TemporalityType.DYNAMIC,
                modality='multi_label_classification',
                observation_frequency=1,
                vocabulary=Vocabulary(['UNK', '1', '2'], obs_frequencies=np.array([0, 3/6, 3/6])),
            ),
            'D_col': MeasurementConfig(
                name='D_col', modality='dropped', observation_frequency=6/6,
                temporality = TemporalityType.DYNAMIC,
            ),
        }

        self.assertNestedDictEqual(want_inferred_measurement_configs, E.inferred_measurement_configs)

    def test_fit_categorical_metadata_numerical_conversion(self):
        config = EventStreamDatasetConfig(
            measurement_configs = {
                'num_key': MeasurementConfig(
                    temporality = TemporalityType.DYNAMIC,
                    modality = 'multivariate_regression',
                    values_column = 'num_val',
                    measurement_metadata = pd.DataFrame({
                        'value_type': ['categorical_integer', 'integer', 'float', 'categorical_float'],
                    }, index = ['k1', 'k2', 'k3', 'k4']),
                )
            }
        )

        events_df = pd.DataFrame({
            'subject_id': [1],
            'timestamp': ['12/1/22'],
            'event_type': ['A'],
        }, index=pd.Index([0], name='event_id'))
        metadata_df = pd.DataFrame({
            'num_key': ['k1', 'k1', 'k1', 'k1', 'k2', 'k2', 'k3', 'k4'],
            'num_val': [1.0,  0.9,  2.1,  1.0,  2,    3,    4.5,  7],
            'event_id': [0, 0, 0, 0, 0, 0, 0, 0],
        })
        E = EventStreamDataset(events_df=events_df, metadata_df=metadata_df, config=config)
        E.split_subjects = {'train': {1}, 'held_out': set()}

        self.assertEqual(E.inferred_measurement_configs, {})

        E.fit_metadata()

        want_inferred_measurement_configs = {
            'num_key': MeasurementConfig(
                name = 'num_key',
                temporality = TemporalityType.DYNAMIC,
                modality = 'multivariate_regression',
                observation_frequency = 1,
                values_column = 'num_val',
                measurement_metadata = pd.DataFrame({
                    'value_type': ['categorical_integer', 'integer', 'float', 'categorical_float'],
                    'outlier_model': [None] * 4,
                    'normalizer': [None] * 4,
                }, index = ['k1', 'k2', 'k3', 'k4']),
                vocabulary = Vocabulary(
                    ['k1__EQ_1', 'k2', 'k1__EQ_2', 'k3', 'k4__EQ_7.0'],
                    [3/8,          2/8,  1/8,          1/8,  1/8],
                ),
            )
        }

        self.assertNestedDictEqual(want_inferred_measurement_configs, E.inferred_measurement_configs)

    def test_infer_val_type(self):
        # This function doesn't actually need to reference events_df at all.
        events_df = pd.DataFrame({'subject_id': [1], 'timestamp': ['12/1/22'], 'event_type': ['A']})

        cases = [
            {
                'msg': "Should return float without config parameters.",
                'vals': [2., 2., 2.],
                'total_col_obs': 3,
                'config_kwargs': {},
                'want_type': 'float',
            },
            {
                'msg': "Should return 'dropped' if there are too few observations (by count).",
                'vals': [1., 2., 3.],
                'total_col_obs': 3,
                'config_kwargs': {'min_valid_vocab_element_observations': 4},
                'want_type': 'dropped',
            },
            {
                'msg': "Should return 'dropped' if there are too few observations (by frequency).",
                'vals': [1., 2., 3.],
                'total_col_obs': 10,
                'config_kwargs': {'min_valid_vocab_element_observations': 4/10},
                'want_type': 'dropped',
            },
            {
                'msg': "Should not return 'dropped' if there are enough observations (counting NaNs).",
                'vals': [1., 2., 3., np.NaN],
                'total_col_obs': 10,
                'config_kwargs': {'min_valid_vocab_element_observations': 4/10},
                'want_type': 'float',
            },
            {
                'msg': "Should return 'integer' if is mostly integer.",
                'vals': [1., 2., 3., 4., 5.2],
                'total_col_obs': 5,
                'config_kwargs': {'min_true_float_frequency': 1/4},
                'want_type': 'integer',
            },
            {
                'msg': "Should return 'integer' if is mostly integer and should not count NaNs.",
                'vals': [1., 2., 3., 4., 5.2, np.NaN, np.NaN, np.NaN, np.NaN],
                'total_col_obs': 9,
                'config_kwargs': {'min_true_float_frequency': 1/4},
                'want_type': 'integer',
            },
            {
                'msg': "Should return 'float' if is mostly float.",
                'vals': [1.1, 2.2, 3.3, 4.4, 5.2],
                'total_col_obs': 5,
                'config_kwargs': {'min_true_float_frequency': 1/4},
                'want_type': 'float',
            },
            {
                'msg': (
                    "Should return 'categorical_integer' if "
                    "# integer observations < min_uniuqe_numerical_observations (by count) and if integer "
                    "conversion has triggered."
                ),
                'vals': [1., 2., 5., 5., 5.2],
                'total_col_obs': 5,
                'config_kwargs': {'min_unique_numerical_observations': 4, 'min_true_float_frequency': 1/4},
                'want_type': 'categorical_integer',
            },
            {
                'msg': (
                    "Should return 'categorical_float' if # observations < min_uniuqe_numerical_observations "
                    "(by count)."
                ),
                'vals': [1.1, 2.2, 3.3, 4.4, 5.5],
                'total_col_obs': 5,
                'config_kwargs': {'min_unique_numerical_observations': 6},
                'want_type': 'categorical_float',
            },
            {
                'msg': (
                    "Should return 'categorical_float' if # observations < min_uniuqe_numerical_observations "
                    "(by count) and should ignore NaNs."
                ),
                'vals': [1.1, 2.2, 3.3, 4.4, 5.5, np.NaN, np.NaN, np.NaN],
                'total_col_obs': 8,
                'config_kwargs': {'min_unique_numerical_observations': 6},
                'want_type': 'categorical_float',
            },
            {
                'msg': (
                    "Should return 'categorical_float' if # observations < min_uniuqe_numerical_observations "
                    "(by proportion)."
                ),
                'vals': [1.1, 1.1, 1.1, 1.1, 5.5],
                'total_col_obs': 5,
                'config_kwargs': {'min_unique_numerical_observations': 3/5},
                'want_type': 'categorical_float',
            },
            {
                'msg': (
                    "Should return 'float' if # observations > min_uniuqe_numerical_observations."
                ),
                'vals': [1.1, 2.2, 3.3, 4.4, 5.5],
                'total_col_obs': 100,
                'config_kwargs': {'min_unique_numerical_observations': 3/5},
                'want_type': 'float',
            },
            {
                'msg': "Should convert to categorical_float if a single observation is too frequent.",
                'vals': [1.1, 1.1, 1.1, 1.1, 5.5],
                'total_col_obs': 100,
                'config_kwargs': {'max_numerical_value_frequency': 3/5},
                'want_type': 'categorical_float',
            },
            {
                'msg': "Should not convert to categorical unless a single observation is too frequent.",
                'vals': [1.1, 1.1, 1.3, 1.3, 5.5],
                'total_col_obs': 5,
                'config_kwargs': {'max_numerical_value_frequency': 3/5},
                'want_type': 'float',
            },
            {
                'msg': (
                    "Should return 'dropped' if categorical checking is enabled and there is only 1 value."
                ),
                'vals': [1.1, 1.1, 1.1, 1.1, 1.1],
                'total_col_obs': 5,
                'config_kwargs': {'min_unique_numerical_observations': 6},
                'want_type': 'dropped',
            },
        ]

        for C in cases:
            with self.subTest(C['msg']):
                E = EventStreamDataset(
                    events_df=events_df, config=EventStreamDatasetConfig(**C['config_kwargs'])
                )

                vals = pd.Series(C['vals'])
                self.assertEqual(
                    C['want_type'], E._infer_val_type(vals.dropna(), C['total_col_obs'], len(vals))
                )

    def test_fit_multivariate_numerical_metadata_column_vals(self):
        # This function doesn't actually need to reference events_df at all.
        events_df = pd.DataFrame({
            'subject_id': [1], 'timestamp': ['12/1/22'], 'event_type': ['A'],
        })

        key_col = 'key_col'
        val_col = 'val_col'
        obs_key = 'obs_key'

        # DummySklearn is defined at the top of the file, and just memorizes the mean, min, max, and count of
        # the input, and asserts that it is a secretly 1D array of reshaped to a 2D array per sklearn
        # convention. validate_and_squeeze just checks that the shape is right then returns the 1D version.
        class DummyOutlier(DummySklearn):
            def predict(self, vals):
                return np.array([1 if v % 2 == 0 else -1 for v in self.validate_and_squeeze(vals)])

        class DummyNormalizer(DummySklearn):
            def transform(self, vals): return self.validate_and_squeeze(vals) - self.mean

        class Derived(EventStreamDataset):
            METADATA_MODELS = {
                'outlier': DummyOutlier,
                'normalizer': DummyNormalizer,
            }

        want_metadata_index = pd.Index([obs_key], name=key_col)
        cases = [
            {
                'msg': "Should fit an outlier model on the data if directed.",
                'vals': [1., 2., 3.],
                'total_col_obs': 3,
                'config_kwargs': {'outlier_detector_config': {'cls': 'outlier'}},
                'want_metadata_dict': {
                    'value_type': ['float'],
                    'outlier_model': [DummyOutlier(mean=2, min=1, max=3, count=3)],
                    'normalizer': [None],
                },
            },
            {
                'msg': "Should drop the column if all values are outliers.",
                'vals': [1., 3., 5.],
                'total_col_obs': 3,
                'config_kwargs': {'outlier_detector_config': {'cls': 'outlier'}},
                'want_metadata_dict': {
                    'value_type': ['dropped'],
                    'outlier_model': [DummyOutlier(mean=3, min=1, max=5, count=3)],
                    'normalizer': [None],
                },
            },
            {
                'msg': "Outlier model fit should respect conversion to integer when warranted.",
                'vals': [1., 1.9, 3.1],
                'total_col_obs': 3,
                'config_kwargs': {
                    'min_true_float_frequency': 3/4,
                    'outlier_detector_config': {'cls': 'outlier'}
                },
                'want_metadata_dict': {
                    'value_type': ['integer'],
                    'outlier_model': [DummyOutlier(mean=2, min=1, max=3, count=3)],
                    'normalizer': [None],
                },
            },
            {
                'msg': "Should fit a normalizer model on the data if directed.",
                'vals': [1., 2., 3.],
                'total_col_obs': 3,
                'config_kwargs': {'normalizer_config': {'cls': 'normalizer'}},
                'want_metadata_dict': {
                    'value_type': ['float'],
                    'outlier_model': [None],
                    'normalizer': [DummyNormalizer(mean=2, min=1, max=3, count=3)],
                },
            },
            {
                'msg': "Normalizer model fit should respect outliers and integer conversion when warranted.",
                'vals': [1., 1.9, 3.1],
                'total_col_obs': 3,
                'config_kwargs': {
                    'min_true_float_frequency': 3/4,
                    'outlier_detector_config': {'cls': 'outlier'},
                    'normalizer_config': {'cls': 'normalizer'},
                },
                'want_metadata_dict': {
                    'value_type': ['integer'],
                    'outlier_model': [DummyOutlier(mean=2, min=1, max=3, count=3)],
                    'normalizer': [DummyNormalizer(mean=2, min=2, max=2, count=1)],
                },
            },
            {
                'msg': "Normalizer model fit should respect outliers and integer conversion when pre-set.",
                'vals': [1., 1.9, 3.1],
                'total_col_obs': 3,
                'config_kwargs': {
                    'outlier_detector_config': {'cls': 'outlier'},
                    'normalizer_config': {'cls': 'normalizer'},
                },
                'preset_value_type': 'integer',
                'want_metadata_dict': {
                    'value_type': ['integer'],
                    'outlier_model': [DummyOutlier(mean=2, min=1, max=3, count=3)],
                    'normalizer': [DummyNormalizer(mean=2, min=2, max=2, count=1)],
                },
            },
            {
                'msg': "Outlier and Normalizer model should not be fit when pre-set to dropped.",
                'vals': [1., 1.9, 3.1],
                'total_col_obs': 3,
                'config_kwargs': {
                    'outlier_detector_config': {'cls': 'outlier'},
                    'normalizer_config': {'cls': 'normalizer'},
                },
                'preset_value_type': 'dropped',
                'want_metadata_dict': {
                    'value_type': ['dropped'],
                    'outlier_model': [None],
                    'normalizer': [None],
                },
            },
        ]

        for C in cases:
            with self.subTest(C['msg']):
                config = EventStreamDatasetConfig.from_simple_args([(key_col, val_col)], **C['config_kwargs'])
                config.measurement_configs[key_col].add_empty_metadata()

                # We only want to add a pre-set value type if it is specified in the case...
                if 'preset_value_type' in C:
                    config.measurement_configs[key_col].measurement_metadata.loc[obs_key, 'value_type'] = \
                        C['preset_value_type']

                E = Derived(events_df=events_df, config=config)

                # The inferred_measurement_configs is set prior to this function, so we do it manually:
                E.inferred_measurement_configs = copy.deepcopy(config.measurement_configs)

                vals = pd.Series(C['vals'], name=obs_key)
                got = E._fit_multivariate_numerical_metadata_column_vals(
                    vals, C['total_col_obs'], E.inferred_measurement_configs[key_col].measurement_metadata
                )

                self.assertTrue(got is None)

                # Models need to be compared separately.

                want_metadata = pd.DataFrame(C['want_metadata_dict'], index=want_metadata_index)

                self.assertEqual(want_metadata, E.inferred_measurement_configs[key_col].measurement_metadata)

    def test_fit_numerical_metadata(self):
        # DummySklearn is defined at the top of the file, and just memorizes the mean, min, max, and count of
        # the input, and asserts that it is a secretly 1D array of reshaped to a 2D array per sklearn
        # convention. validate_and_squeeze just checks that the shape is right then returns the 1D version.
        class DummyOutlier(DummySklearn):
            def predict(self, vals):
                return np.array([1 if v >= self.mean else -1 for v in self.validate_and_squeeze(vals)])

        class DummyNormalizer(DummySklearn):
            def transform(self, vals): return self.validate_and_squeeze(vals) - self.mean

        class Derived(EventStreamDataset):
            METADATA_MODELS = {
                'outlier': DummyOutlier,
                'normalizer': DummyNormalizer,
            }
            UNIT_BOUNDS = {
                # (unit strings): [lower, lower_inclusive, upper, upper_inclusive],
                ('%', 'percent'): [0, False, 1, False],
            }

        config = EventStreamDatasetConfig.from_simple_args(
            dynamic_measurement_columns = [
                ('num1_key', 'num1_val'), ('num2_key', 'num2_val'), ('num3_key', 'num3_val'),
                ('num4_key', 'num4_val'), ('num5_key', 'num5_val')
            ],
            time_dependent_measurement_columns = [('age', AgeFunctor('dob'))],
            min_valid_column_observations = 2,
            min_true_float_frequency = 1/2,
            min_valid_vocab_element_observations = 1/3,
            min_unique_numerical_observations = 2,
            outlier_detector_config = {'cls': 'outlier'},
            normalizer_config = {'cls': 'normalizer'},
        )
        config.measurement_configs['num5_key'].measurement_metadata = pd.DataFrame({
            'value_type': ['float', 'float'],
            'unit': [None, '%'],
            'drop_lower_bound': [-1, None],
            'drop_lower_bound_inclusive': [False, None],
            'drop_upper_bound': [1, None],
            'drop_upper_bound_inclusive': [True, None],
            'censor_lower_bound': [-0.5, None],
            'censor_upper_bound': [0.5, None],
        }, index=pd.Index(['k1', 'k2'], name='num5_key'))

        subjects_df = pd.DataFrame(
            {'dob': [pd.to_datetime('12/1/20')]}, index=pd.Index([1], name='subject_id')
        )
        events_df = pd.DataFrame({
            'subject_id': [1, 1, 1],
            'timestamp': ['12/1/21', '12/1/22', '12/1/23'],
            'event_type': ['A', 'A', 'A'],
        }, index=pd.Index([0, 1, 2], name='event_id'))
        metadata_df = pd.DataFrame({
            'event_id': [0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2],
            'num1_key': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar', None, None, None, None, None, None, None],
            'num1_val': [0.1, 0.3, 0.5, -5, -1.1, -3, None, None, None, None, None, None, None],
            'num2_key': ['biz', 'baz', 'baz', 'baz', 'baz', 'baz', None, None, None, None, None, None, None],
            'num2_val': [0.4, 0.3, 0.1, 0.2, 0.5, 0.4, None, None, None, None, None, None, None],
            'num3_key': ['one', 'two', 'one', 'two', 'one', 'two', None, None, None, None, None, None, None],
            'num3_val': [-0.1,  0.3, 0, 0.1, 0, 0.2, None, None, None, None, None, None, None],
            'num4_key': [None, None, None, None, None, None, 'one', None, None, None, None, None, None],
            'num4_val': [None, None, None, None, None, None, 0.1, None, None, None, None, None, None],
            'num5_key': ['k1', 'k1', 'k1', 'k2', 'k2', 'k2', None, 'k1', 'k1', 'k1', 'k2', 'k2', 'k2'],
            'num5_val': [-2, -1, -0.75, -1, 0, 0.5, None, 0.3, 0.7, 1, 1, 1.1, 1.2],
        })

        E = Derived(events_df=events_df, metadata_df=metadata_df, subjects_df=subjects_df, config=config)
        E.split_subjects = {'train': {1}, 'held_out': set()}

        self.assertEqual({}, E.inferred_measurement_configs)
        self.assertEqual(
            [
                ('num1_key', 'num1_val'), ('num2_key', 'num2_val'), ('num3_key', 'num3_val'),
                ('num4_key', 'num4_val'), ('num5_key', 'num5_val'),
            ],
            E.dynamic_numerical_columns
        )
        self.assertEqual(['age'], E.time_dependent_numerical_columns)

        E.add_time_dependent_columns()
        E.fit_metadata()

        want_inferred_measurement_configs = {
            'age': MeasurementConfig(
                name = 'age',
                temporality = TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                functor = AgeFunctor('dob'),
                modality = 'univariate_regression',
                measurement_metadata = pd.Series(
                    [
                        'integer',
                        # Age values are going to be 1, 2, 3
                        DummyOutlier(mean=2, min=1, max=3, count=3),
                        DummyNormalizer(mean=2.5, min=2, max=3, count=2),
                    ],
                    index = pd.Index(['value_type', 'outlier_model', 'normalizer']),
                ),
            ),
            'num1_key': MeasurementConfig(
                name = 'num1_key',
                temporality = TemporalityType.DYNAMIC,
                modality = 'multivariate_regression',
                values_column = 'num1_val',
                measurement_metadata = pd.DataFrame({
                    'value_type': ['integer', 'float'],
                    'outlier_model': [
                        DummyOutlier(mean=-3, min=-5, max=-1, count=3),
                        DummyOutlier(mean=0.3, min=0.1, max=0.5, count=3),
                    ],
                    'normalizer': [
                        DummyNormalizer(mean=-2, min=-3, max=-1, count=2),
                        DummyNormalizer(mean=0.4, min=0.3, max=0.5, count=2),
                    ],
                }, index = pd.Index(['bar', 'foo'], name='num1_key')),
                observation_frequency = 6/13,
                vocabulary = Vocabulary(['UNK', 'foo', 'bar'], [0, 1/2, 1/2]),
            ),
            'num2_key': MeasurementConfig(
                name = 'num2_key',
                temporality = TemporalityType.DYNAMIC,
                modality = 'multivariate_regression',
                values_column = 'num2_val',
                measurement_metadata = pd.DataFrame({
                    'value_type': ['float', 'dropped'],
                    'outlier_model': [
                        DummyOutlier(mean=0.3, min=0.1, max=0.5, count=5),
                        np.NaN,
                    ],
                    'normalizer': [
                        DummyNormalizer(mean=0.4, min=0.3, max=0.5, count=3),
                        np.NaN,
                    ],
                }, index = pd.Index(['baz', 'biz'], name='num2_key')),
                observation_frequency = 6/13,
                vocabulary = Vocabulary(['UNK', 'baz'], [1/6, 5/6]),
            ),
            'num3_key': MeasurementConfig(
                name = 'num3_key',
                temporality = TemporalityType.DYNAMIC,
                modality = 'multivariate_regression',
                values_column = 'num3_val',
                measurement_metadata = pd.DataFrame({
                    'value_type': ['dropped', 'float'],
                    'outlier_model': [
                        np.NaN,
                        DummyOutlier(mean=0.2, min=0.1, max=0.3, count=3),
                    ],
                    'normalizer': [
                        np.NaN,
                        DummyNormalizer(mean=0.25, min=0.2, max=0.3, count=2),
                    ],
                }, index = pd.Index(['one', 'two'], name='num3_key')),
                observation_frequency = 6/13,
                vocabulary = Vocabulary(['UNK', 'one', 'two'], [0, 3/6, 3/6]),
            ),
            # This column will be ignored as it has too few observations.
            'num4_key': MeasurementConfig(
                name = 'num4_key',
                temporality = TemporalityType.DYNAMIC,
                modality = 'dropped',
                values_column = 'num4_val',
            ),
            'num5_key': MeasurementConfig(
                name = 'num5_key',
                temporality = TemporalityType.DYNAMIC,
                modality = 'multivariate_regression',
                values_column = 'num5_val',
                measurement_metadata = pd.DataFrame({
                    'value_type': ['float', 'float'],
                    'unit': [None, '%'],
                    'drop_lower_bound': [-1., 0.],
                    'drop_lower_bound_inclusive': [False, False],
                    'drop_upper_bound': [1., 1],
                    'drop_upper_bound_inclusive': [True, False],
                    'censor_lower_bound': [-0.5, None],
                    'censor_upper_bound': [0.5, None],
                    'outlier_model': [
                        DummyOutlier(mean=-0.05, min=-0.5, max=0.5, count=4),
                        DummyOutlier(mean=0.5, min=0, max=1, count=3),
                    ],
                    'normalizer': [
                        DummyNormalizer(mean=0.4, min=0.3, max=0.5, count=2),
                        DummyNormalizer(mean=0.75, min=0.5, max=1, count=2),
                    ],
                }, index = pd.Index(['k1', 'k2'], name='num5_key')),
                observation_frequency = 12/13,
                vocabulary = Vocabulary(['UNK', 'k1', 'k2'], [0, 1/2, 1/2]),
            ),
        }

        self.assertNestedDictEqual(want_inferred_measurement_configs, E.inferred_measurement_configs)

    def test_transform_metadata(self):
        # DummySklearn is defined at the top of the file, and just memorizes the mean, min, max, and count of
        # the input, and asserts that it is a secretly 1D array of reshaped to a 2D array per sklearn
        # convention. validate_and_squeeze just checks that the shape is right then returns the 1D version.
        class DummyOutlier(DummySklearn):
            def predict(self, vals):
                return np.array([1 if v >= self.mean else -1 for v in self.validate_and_squeeze(vals)])

        class DummyNormalizer(DummySklearn):
            def transform(self, vals): return self.validate_and_squeeze(vals) - self.mean

        measurement_configs = {
            'cat': MeasurementConfig(
                modality = 'single_label_classification',
                temporality = TemporalityType.DYNAMIC,
            ),
            'cat2': MeasurementConfig(
                temporality = TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                functor = TimeOfDayFunctor(),
            ),
            'cat3': MeasurementConfig(
                modality = DataModality.SINGLE_LABEL_CLASSIFICATION,
                temporality = TemporalityType.STATIC,
            ),
            'num1_key': MeasurementConfig(
                temporality = TemporalityType.DYNAMIC,
                modality = 'multivariate_regression',
                vocabulary = Vocabulary(['bar', 'foo__EQ_0.1', 'foo__EQ_0.3'], [0.5, 0.25, 0.25]),
                values_column = 'num1_val',
                measurement_metadata = pd.DataFrame({
                    'value_type': ['integer', 'categorical_float'],
                    'outlier_model': [
                        DummyOutlier(mean=-3, min=-5, max=-1, count=3),
                        None,
                    ],
                    'normalizer': [
                        DummyNormalizer(mean=-2, min=-3, max=-1, count=2),
                        None,
                    ],
                }, index = pd.Index(['bar', 'foo'], name='num1_key')),
            ),
            'num2_key': MeasurementConfig(
                temporality = TemporalityType.DYNAMIC,
                modality = 'multivariate_regression',
                vocabulary = Vocabulary(['baz', 'biz'], [0.5, 0.5]),
                values_column = 'num2_val',
                measurement_metadata = pd.DataFrame({
                    'value_type': ['float', 'dropped'],
                    'outlier_model': [
                        DummyOutlier(mean=0.3, min=0.1, max=0.5, count=5),
                        np.NaN,
                    ],
                    'normalizer': [
                        DummyNormalizer(mean=0.4, min=0.3, max=0.5, count=3),
                        np.NaN,
                    ],
                }, index = pd.Index(['baz', 'biz'], name='num2_key')),
            ),
            'num3_key': MeasurementConfig(
                temporality = TemporalityType.DYNAMIC,
                modality = 'multivariate_regression',
                vocabulary = Vocabulary(['one__EQ_-1', 'one__EQ_1'], [0.5, 0.5]),
                values_column = 'num3_val',
                measurement_metadata = pd.DataFrame({
                    'value_type': ['categorical_integer', 'dropped'],
                    'outlier_model': [
                        np.NaN,
                        DummyOutlier(mean=0.2, min=0.1, max=0.3, count=3),
                    ],
                    'normalizer': [
                        np.NaN,
                        DummyNormalizer(mean=0.25, min=0.2, max=0.3, count=2),
                    ],
                }, index = pd.Index(['one', 'two'], name='num3_key')),
            ),
            # This column will be ignored as it has too few observations.
            'num4_key': MeasurementConfig(
                temporality = TemporalityType.DYNAMIC,
                modality = 'dropped'
            ),
            'num5_key': MeasurementConfig(
                temporality = TemporalityType.DYNAMIC,
                modality = 'multivariate_regression',
                vocabulary = Vocabulary(['k1', 'k2'], [0.5, 0.5]),
                values_column = 'num5_val',
                measurement_metadata = pd.DataFrame({
                    'value_type': ['float', 'float'],
                    'unit': [None, '%'],
                    'drop_lower_bound': [-1., 0.],
                    'drop_lower_bound_inclusive': [False, False],
                    'drop_upper_bound': [1., 1],
                    'drop_upper_bound_inclusive': [True, False],
                    'censor_lower_bound': [-0.5, None],
                    'censor_upper_bound': [0.5, None],
                    'outlier_model': [
                        DummyOutlier(mean=-0.05, min=-0.5, max=0.5, count=4),
                        DummyOutlier(mean=0.5, min=0, max=1, count=3),
                    ],
                    'normalizer': [
                        DummyNormalizer(mean=0.4, min=0.3, max=0.5, count=2),
                        DummyNormalizer(mean=0.75, min=0.5, max=1, count=2),
                    ],
                }, index = pd.Index(['k1', 'k2'], name='num5_key')),
            ),
            'TD_num_col_1': MeasurementConfig(
                temporality = TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                functor = AgeFunctor,
                vocabulary = None,
                measurement_metadata = pd.Series({
                    'value_type': 'float',
                    'unit': '%',
                    'drop_lower_bound': 0.,
                    'drop_lower_bound_inclusive': False,
                    'drop_upper_bound': 1,
                    'drop_upper_bound_inclusive': False,
                    'censor_lower_bound': None,
                    'censor_upper_bound': None,
                    'outlier_model': DummyOutlier(mean=0.5, min=0, max=1, count=3),
                    'normalizer': DummyNormalizer(mean=0.75, min=0.5, max=1, count=2),
                }),
            ),
            'TD_num_col_2': MeasurementConfig(
                temporality = TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                functor = AgeFunctor,
                vocabulary = None,
                measurement_metadata = pd.Series({
                    'value_type': 'integer',
                    'outlier_model': DummyOutlier(mean=0, min=-1, max=1, count=3),
                    'normalizer': DummyNormalizer(mean=0.5, min=0, max=1, count=2),
                }),
            ),
            'TD_num_col_3': MeasurementConfig(
                temporality = TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                modality = DataModality.DROPPED,
                functor = AgeFunctor,
            ),
        }

        config = EventStreamDatasetConfig(
            min_valid_column_observations = 2,
            min_true_float_frequency = 1/2,
            min_valid_vocab_element_observations = 1/3,
            min_unique_numerical_observations = 2,
            outlier_detector_config = {'cls': 'outlier'},
            normalizer_config = {'cls': 'normalizer'},
        )

        events_df = pd.DataFrame({
            'subject_id': [1, 1, 1],
            'timestamp': ['12/1/22', '12/2/22', '12/3/22'],
            'event_type': ['A', 'A', 'A'],
            'cat2': ['EARLY_AM', 'EARLY_AM', 'EARLY_AM'],
            'TD_num_col_1': [0.4, 0.6, 1.2],
            'TD_num_col_2': [-1, 0, 1],
            'TD_num_col_3': [None, None, 2],
        }, index=pd.Index([0, 1, 2], name='event_id'))

        metadata_df = pd.DataFrame({
            'event_id': [0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2],
            'cat': ['a', 'b', 'c', 'd', 'e', 'f', None, None, None, None, None, None, None],
            'num1_key': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar', None, None, None, None, None, None, None],
            'num1_val': [0.1, 0.3, 0.5, -5, -1.1, -3, None, None, None, None, None, None, None],
            'num2_key': ['biz', 'baz', 'baz', 'baz', 'baz', 'baz', None, None, None, None, None, None, None],
            'num2_val': [0.4, 0.3, 0.1, 0.2, 0.5, 0.4, None, None, None, None, None, None, None],
            'num3_key': ['one', 'two', 'one', 'two', 'one', 'two', None, None, None, None, None, None, None],
            'num3_val': [-0.9,  0.3, 1, 0.1, 0, 0.8, None, None, None, None, None, None, None],
            'num4_key': [None, None, None, None, None, None, 'one', None, None, None, None, None, None],
            'num4_val': [None, None, None, None, None, None, 0.1, None, None, None, None, None, None],
            'num5_key': ['k1', 'k1', 'k1', 'k2', 'k2', 'k2', None, 'k1', 'k1', 'k1', 'k2', 'k2', 'k2'],
            'num5_val': [-2, -1, -0.75, -1, 0, 0.5, None, 0.3, 0.7, 1, 1, 1.1, 1.2],
        })

        subjects_df = pd.DataFrame({'cat3': ['A']}, index=pd.Index([1], name='subject_id'))
        E = EventStreamDataset(
            events_df=events_df.copy(),
            metadata_df=metadata_df.copy(),
            subjects_df=subjects_df.copy(),
            config=config
        )

        E.metadata_is_fit = True
        E.inferred_measurement_configs = measurement_configs

        E.transform_metadata()

        want_metadata_df = pd.DataFrame({
            'event_id': [0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2],
            'event_type': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
            'subject_id': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            # We expect the categorical column to be unchanged.
            'cat': ['a', 'b', 'c', 'd', 'e', 'f', None, None, None, None, None, None, None],
            '__backup_num1_key': [
                'foo', 'foo', 'foo', 'bar', 'bar', 'bar', None, None, None, None, None, None, None,
            ],
            'num1_key': [
                'foo__EQ_0.1', 'foo__EQ_0.3', 'foo__EQ_0.5', 'bar', 'bar', 'bar', None, None, None, None,
                None, None, None
            ],
            '__backup_num1_val': [
                0.1, 0.3, 0.5, -5, -1.1, -3, None, None, None, None, None, None, None
            ],
            'num1_val': [np.NaN, np.NaN, np.NaN, np.NaN, 1, -1, None, None, None, None, None, None, None],
            'num1_val_is_inlier': [
                None, None, None, False, True, True, None, None, None, None, None, None, None
            ],
            '__backup_num2_key': [
                'biz', 'baz', 'baz', 'baz', 'baz', 'baz', None, None, None, None, None, None, None
            ],
            'num2_key': ['biz', 'baz', 'baz', 'baz', 'baz', 'baz', None, None, None, None, None, None, None],
            '__backup_num2_val': [
                0.4,   0.3,   0.1,   0.2,   0.5,   0.4, None, None, None, None, None, None, None
            ],
            'num2_val': [np.NaN, -0.1, np.NaN, np.NaN, 0.1, 0, None, None, None, None, None, None, None],
            'num2_val_is_inlier': [
                None, True, False, False, True, True, None, None, None, None, None, None, None
            ],
            '__backup_num3_key': [
                'one', 'two', 'one', 'two', 'one', 'two', None, None, None, None, None, None, None
            ],
            'num3_key': [
                'one__EQ_-1', 'two', 'one__EQ_1', 'two', 'one__EQ_0', 'two', None, None, None, None, None,
                None, None
            ],
            '__backup_num3_val': [
                -0.9,  0.3,   1,     0.1,   0,     0.8, None, None, None, None, None, None, None
            ],
            'num3_val': [
                np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, None, None, None, None, None, None, None
            ],
            'num4_key': [None, None, None, None, None, None, 'one', None, None, None, None, None, None],
            'num4_val': [None, None, None, None, None, None, 0.1, None, None, None, None, None, None],
            '__backup_num5_key': [
                'k1', 'k1', 'k1', 'k2', 'k2', 'k2', None, 'k1', 'k1', 'k1', 'k2', 'k2', 'k2'
            ],
            'num5_key': ['k1', 'k1', 'k1', 'k2', 'k2', 'k2', None, 'k1', 'k1', 'k1', 'k2', 'k2', 'k2'],
            '__backup_num5_val': [
                -2, -1, -0.75, -1, 0, 0.5, None, 0.3, 0.7, 1, 1, 1.1, 1.2,
            ],
            'num5_val': [
                np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, -0.25, None, -0.1, 0.1, np.NaN, 0.25, np.NaN, np.NaN
            ],
            'num5_val_is_inlier': [
                None, False, False, None, False, True, None, True, True, None, True, None, None
            ]
        }, index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], name='metadata_id'))

        self.assertEqual(want_metadata_df.shape, E.joint_metadata_df.shape)
        self.assertEqual(set(want_metadata_df.columns), set(E.joint_metadata_df.columns))
        want_metadata_df_reordered = want_metadata_df[E.joint_metadata_df.columns]

        self.assertEqual(want_metadata_df_reordered, E.joint_metadata_df)

        # E.events_df should also update.
        want_events_df = pd.DataFrame({
            'subject_id': [1, 1, 1],
            'timestamp': [pd.to_datetime('12/1/22'), pd.to_datetime('12/2/22'), pd.to_datetime('12/3/22')],
            'event_type': ['A', 'A', 'A'],
            'cat2': ['EARLY_AM', 'EARLY_AM', 'EARLY_AM'],
            'TD_num_col_1': [np.NaN, -0.15, np.NaN],
            'TD_num_col_2': [np.NaN, -0.5, 0.5],
            'TD_num_col_3': [np.NaN, np.NaN, 2.],
            '__backup_TD_num_col_1': [0.4, 0.6, 1.2],
            '__backup_TD_num_col_2': [-1, 0, 1],
            'TD_num_col_1_is_inlier': pd.Series([False, True, None], dtype='boolean'),
            'TD_num_col_2_is_inlier': pd.Series([False, True, True], dtype='boolean'),
        }, index=pd.Index([0, 1, 2], name='event_id'))
        self.assertEqual(want_events_df, E.events_df)

        # E.subjects_df should not update.
        self.assertEqual(subjects_df, E.subjects_df)

    def test_save_and_load(self):
        # DummySklearn is defined at the top of the file, and just memorizes the mean, min, max, and count of
        # the input, and asserts that it is a secretly 1D array of reshaped to a 2D array per sklearn
        # convention. validate_and_squeeze just checks that the shape is right then returns the 1D version.
        class DummyOutlier(DummySklearn):
            def predict(self, vals):
                return np.array([1 if v >= self.mean else -1 for v in self.validate_and_squeeze(vals)])

        class DummyNormalizer(DummySklearn):
            def transform(self, vals): return self.validate_and_squeeze(vals) - self.mean

        measurement_configs = {
            'cat': MeasurementConfig(
                temporality = TemporalityType.DYNAMIC,
                modality = 'single_label_classification'
            ),
            'num1_key': MeasurementConfig(
                temporality = TemporalityType.DYNAMIC,
                modality = 'multivariate_regression',
                vocabulary = Vocabulary(['bar', 'foo__EQ_0.1', 'foo__EQ_0.3'], [0.5, 0.25, 0.25]),
                values_column = 'num1_val',
                measurement_metadata = pd.DataFrame({
                    'value_type': ['integer', 'categorical_float'],
                    'outlier_model': [
                        DummyOutlier(mean=-3, min=-5, max=-1, count=3),
                        None,
                    ],
                    'normalizer': [
                        DummyNormalizer(mean=-2, min=-3, max=-1, count=2),
                        None,
                    ],
                }, index = pd.Index(['bar', 'foo'], name='num1_key')),
            ),
            'num2_key': MeasurementConfig(
                temporality = TemporalityType.DYNAMIC,
                modality = 'multivariate_regression',
                vocabulary = Vocabulary(['baz', 'biz'], [0.5, 0.5]),
                values_column = 'num2_val',
                measurement_metadata = pd.DataFrame({
                    'value_type': ['float', 'dropped'],
                    'outlier_model': [
                        DummyOutlier(mean=0.3, min=0.1, max=0.5, count=5),
                        np.NaN,
                    ],
                    'normalizer': [
                        DummyNormalizer(mean=0.4, min=0.3, max=0.5, count=3),
                        np.NaN,
                    ],
                }, index = pd.Index(['baz', 'biz'], name='num2_key')),
            ),
            'num3_key': MeasurementConfig(
                temporality = TemporalityType.DYNAMIC,
                modality = 'multivariate_regression',
                vocabulary = Vocabulary(['one__EQ_-1', 'one__EQ_1'], [0.5, 0.5]),
                values_column = 'num3_val',
                measurement_metadata = pd.DataFrame({
                    'value_type': ['categorical_integer', 'dropped'],
                    'outlier_model': [
                        np.NaN,
                        DummyOutlier(mean=0.2, min=0.1, max=0.3, count=3),
                    ],
                    'normalizer': [
                        np.NaN,
                        DummyNormalizer(mean=0.25, min=0.2, max=0.3, count=2),
                    ],
                }, index = pd.Index(['one', 'two'], name='num3_key')),
            ),
            # This column will be ignored as it has too few observations.
            'num4_key': MeasurementConfig(
                temporality = TemporalityType.DYNAMIC,
                modality = 'dropped'
            ),
            'num5_key': MeasurementConfig(
                temporality = TemporalityType.DYNAMIC,
                modality = 'multivariate_regression',
                vocabulary = Vocabulary(['k1', 'k2'], [0.5, 0.5]),
                values_column = 'num5_val',
                measurement_metadata = pd.DataFrame({
                    'value_type': ['float', 'float'],
                    'unit': [None, '%'],
                    'drop_lower_bound': [-1., 0.],
                    'drop_lower_bound_inclusive': [False, False],
                    'drop_upper_bound': [1., 1],
                    'drop_upper_bound_inclusive': [True, False],
                    'censor_lower_bound': [-0.5, None],
                    'censor_upper_bound': [0.5, None],
                    'outlier_model': [
                        DummyOutlier(mean=-0.05, min=-0.5, max=0.5, count=4),
                        DummyOutlier(mean=0.5, min=0, max=1, count=3),
                    ],
                    'normalizer': [
                        DummyNormalizer(mean=0.4, min=0.3, max=0.5, count=2),
                        DummyNormalizer(mean=0.75, min=0.5, max=1, count=2),
                    ],
                }, index = pd.Index(['k1', 'k2'], name='num5_key')),
            ),
        }

        def fn_arg(N: int): return N**2

        config = EventStreamDatasetConfig(
            min_valid_column_observations = 2,
            min_true_float_frequency = 1/2,
            min_valid_vocab_element_observations = 1/3,
            min_unique_numerical_observations = 2,
            outlier_detector_config = {'cls': 'outlier'},
            normalizer_config = {'cls': 'normalizer', 'fn_arg': fn_arg},
        )

        events_df = pd.DataFrame({
            'subject_id': [1, 1, 1],
            'timestamp': ['12/1/22', '12/2/22', '12/3/22'],
            'event_type': ['A', 'A', 'A'],
        }, index=pd.Index([0, 1, 2], name='event_id'))
        metadata_df = pd.DataFrame({
            'event_id': [0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2],
            'cat': ['a', 'b', 'c', 'd', 'e', 'f', None, None, None, None, None, None, None],
            'num1_key': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar', None, None, None, None, None, None, None],
            'num1_val': [0.1, 0.3, 0.5, -5, -1.1, -3, None, None, None, None, None, None, None],
            'num2_key': ['biz', 'baz', 'baz', 'baz', 'baz', 'baz', None, None, None, None, None, None, None],
            'num2_val': [0.4, 0.3, 0.1, 0.2, 0.5, 0.4, None, None, None, None, None, None, None],
            'num3_key': ['one', 'two', 'one', 'two', 'one', 'two', None, None, None, None, None, None, None],
            'num3_val': [-0.9,  0.3, 1, 0.1, 0, 0.8, None, None, None, None, None, None, None],
            'num4_key': [None, None, None, None, None, None, 'one', None, None, None, None, None, None],
            'num4_val': [None, None, None, None, None, None, 0.1, None, None, None, None, None, None],
            'num5_key': ['k1', 'k1', 'k1', 'k2', 'k2', 'k2', None, 'k1', 'k1', 'k1', 'k2', 'k2', 'k2'],
            'num5_val': [-2, -1, -0.75, -1, 0, 0.5, None, 0.3, 0.7, 1, 1, 1.1, 1.2],
        })
        E = EventStreamDataset(events_df=events_df, metadata_df=metadata_df, config=config)

        E.metadata_is_fit = True
        E.inferred_measurement_configs = measurement_configs

        E.transform_metadata()

        want_metadata_df = pd.DataFrame({
            'event_id': [0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2],
            'event_type': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
            'subject_id': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            # We expect the categorical column to be unchanged.
            'cat': ['a', 'b', 'c', 'd', 'e', 'f', None, None, None, None, None, None, None],
            '__backup_num1_key': [
                'foo', 'foo', 'foo', 'bar', 'bar', 'bar', None, None, None, None, None, None, None,
            ],
            'num1_key': [
                'foo__EQ_0.1', 'foo__EQ_0.3', 'foo__EQ_0.5', 'bar', 'bar', 'bar', None, None, None, None,
                None, None, None
            ],
            '__backup_num1_val': [
                0.1, 0.3, 0.5, -5, -1.1, -3, None, None, None, None, None, None, None
            ],
            'num1_val': [np.NaN, np.NaN, np.NaN, np.NaN, 1, -1, None, None, None, None, None, None, None],
            'num1_val_is_inlier': [
                None, None, None, False, True, True, None, None, None, None, None, None, None
            ],
            '__backup_num2_key': [
                'biz', 'baz', 'baz', 'baz', 'baz', 'baz', None, None, None, None, None, None, None
            ],
            'num2_key': ['biz', 'baz', 'baz', 'baz', 'baz', 'baz', None, None, None, None, None, None, None],
            '__backup_num2_val': [
                0.4,   0.3,   0.1,   0.2,   0.5,   0.4, None, None, None, None, None, None, None
            ],
            'num2_val': [np.NaN, -0.1, np.NaN, np.NaN, 0.1, 0, None, None, None, None, None, None, None],
            'num2_val_is_inlier': [
                None, True, False, False, True, True, None, None, None, None, None, None, None
            ],
            '__backup_num3_key': [
                'one', 'two', 'one', 'two', 'one', 'two', None, None, None, None, None, None, None
            ],
            'num3_key': [
                'one__EQ_-1', 'two', 'one__EQ_1', 'two', 'one__EQ_0', 'two', None, None, None, None, None,
                None, None
            ],
            '__backup_num3_val': [
                -0.9,  0.3,   1,     0.1,   0,     0.8, None, None, None, None, None, None, None
            ],
            'num3_val': [
                np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, None, None, None, None, None, None, None
            ],
            'num4_key': [None, None, None, None, None, None, 'one', None, None, None, None, None, None],
            'num4_val': [None, None, None, None, None, None, 0.1, None, None, None, None, None, None],
            '__backup_num5_key': [
                'k1', 'k1', 'k1', 'k2', 'k2', 'k2', None, 'k1', 'k1', 'k1', 'k2', 'k2', 'k2'
            ],
            'num5_key': ['k1', 'k1', 'k1', 'k2', 'k2', 'k2', None, 'k1', 'k1', 'k1', 'k2', 'k2', 'k2'],
            '__backup_num5_val': [
                -2, -1, -0.75, -1, 0, 0.5, None, 0.3, 0.7, 1, 1, 1.1, 1.2,
            ],
            'num5_val': [
                np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, -0.25, None, -0.1, 0.1, np.NaN, 0.25, np.NaN, np.NaN
            ],
            'num5_val_is_inlier': [
                None, False, False, None, False, True, None, True, True, None, True, None, None
            ]
        }, index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], name='metadata_id'))

        with TemporaryDirectory() as d:
            save_path = Path(d) / 'save.pkl'
            E._save(save_path)

            got_E = EventStreamDataset._load(save_path)

            self.assertEqual(set(want_metadata_df.columns), set(got_E.joint_metadata_df.columns))
            self.assertEqual(want_metadata_df.shape, got_E.joint_metadata_df.shape)
            want_metadata_df_reordered = want_metadata_df[got_E.joint_metadata_df.columns]

            self.assertEqual(want_metadata_df_reordered, got_E.joint_metadata_df)

            want_no_fn_arg = copy.deepcopy(E.config)
            self.assertTrue('fn_arg' in want_no_fn_arg.normalizer_config)
            want_no_fn_arg.normalizer_config.pop('fn_arg')

            got_no_fn_arg = copy.deepcopy(got_E.config)
            self.assertTrue('fn_arg' in got_no_fn_arg.normalizer_config)
            got_fn_arg = got_no_fn_arg.normalizer_config.pop('fn_arg')

            self.assertEqual(want_no_fn_arg, got_no_fn_arg)
            self.assertEqual([got_fn_arg(n) for n in (-2, -1, 3, 4, 0)], [4, 1, 9, 16, 0])

if __name__ == '__main__': unittest.main()

import sys

sys.path.append("../..")

import unittest

import copy
import pandas as pd
import torch
from datetime import datetime, timedelta
from pytorch_lognormal_mixture import LogNormalMixtureDistribution

from EventStream.data.types import DataModality, PytorchBatch, TemporalityType
from EventStream.data.time_dependent_functor import AgeFunctor, TimeOfDayFunctor
from EventStream.data.vocabulary import Vocabulary
from EventStream.data.config import MeasurementConfig
from EventStream.transformer.config import (
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
)
from EventStream.transformer.model_output import (
    strip_unused_indices,
    GenerativeSequenceModelSamples
)

from ..mixins import MLTypeEqualityCheckableMixin

MEASUREMENT_CONFIGS = {
    'static_clf': MeasurementConfig(
        temporality=TemporalityType.STATIC,
        modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
        vocabulary=Vocabulary(
            vocabulary=['UNK', 'static_clf_1', 'static_clf_2'],
            obs_frequencies=[0.1, 0.5, 0.4],
        ),
    ),
    'age': MeasurementConfig(
        temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
        functor=AgeFunctor('dob'),
        vocabulary=None,
        measurement_metadata=pd.Series({
            'value_type': 'float',
            'normalizer': {'mean_': 40.0, 'std_': 10.0},
            'outlier_model': {'thresh_large_': 90., 'thresh_small_': 18.},
        }),
    ),
    'tod': MeasurementConfig(
        temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
        functor=TimeOfDayFunctor(),
        vocabulary=Vocabulary(
            vocabulary=['UNK', 'EARLY_AM', 'LATE_PM', 'AM', 'PM'],
            obs_frequencies=[0.1, 0.3, 0.22, 0.2, 0.18],
        ),
    ),
    'dynamic_single_label_clf': MeasurementConfig(
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
        vocabulary=Vocabulary(
            vocabulary=['UNK', 'dynamic_single_label_1', 'dynamic_single_label_2'],
            obs_frequencies=[0.1, 0.5, 0.4],
        ),
    ),
    'dynamic_multi_label_clf': MeasurementConfig(
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.MULTI_LABEL_CLASSIFICATION,
        vocabulary=Vocabulary(
            vocabulary=['UNK', 'dynamic_multi_label_1', 'dynamic_multi_label_2', 'dynamic_multi_label_3'],
            obs_frequencies=[0.1, 0.5, 0.25, 0.15],
        ),
    ),
    'dynamic_univariate_reg': MeasurementConfig(
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.UNIVARIATE_REGRESSION,
        vocabulary=None,
        measurement_metadata=pd.Series({
            'value_type': 'float',
            'normalizer': {'mean_': 2.0, 'std_': 3.0},
            'outlier_model': {'thresh_large_': 7.0, 'thresh_small_': -3.0},
        }),
    ),
    'dynamic_multivariate_reg': MeasurementConfig(
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.MULTIVARIATE_REGRESSION,
        values_column='dynamic_multivariate_reg_values',
        vocabulary=Vocabulary(
            vocabulary=['UNK', 'dynamic_multivariate_reg_1', 'dynamic_multivariate_reg_2'],
            obs_frequencies=[0.1, 0.5, 0.4],
        ),
        measurement_metadata=pd.DataFrame(
            {
                'value_type': ['float', 'float'],
                'normalizer': [{'mean_': 2.0, 'std_': 3.0}, {'mean_': 4.0, 'std_': 5.0}],
                'outlier_model': [
                    {'thresh_large_': 7.0, 'thresh_small_': -3.0},
                    {'thresh_large_': 9.0, 'thresh_small_': -5.0}
                ],
            },
            index=pd.Index(
                ['dynamic_multivariate_reg_1', 'dynamic_multivariate_reg_2'], name='dynamic_multivariate_reg'
            ),
        ),
    ),
}

EVENT_TYPES_PER_MEASUREMENT = {
    'dynamic_single_label_clf': ['event_B'],
    'dynamic_multi_label_clf': ['event_B'],
    'dynamic_univariate_reg': ['event_B'],
    'dynamic_multivariate_reg': ['event_B'],
}

MEASUREMENTS_PER_GEN_MODE = {
    DataModality.SINGLE_LABEL_CLASSIFICATION: ["event_type", "dynamic_single_label_clf"],
    DataModality.MULTI_LABEL_CLASSIFICATION: ["dynamic_multi_label_clf", "dynamic_multivariate_reg"],
    DataModality.MULTIVARIATE_REGRESSION: ["dynamic_multivariate_reg"],
    DataModality.UNIVARIATE_REGRESSION: ["dynamic_univariate_reg"],
}
MEASUREMENTS_IDXMAP = {
    "event_type": 1,
    "static_clf": 2,
    "unused": 3,
    "age": 4,
    "tod": 5,
    "dynamic_single_label_clf": 6,
    "dynamic_multi_label_clf": 7,
    "dynamic_univariate_reg": 8,
    "dynamic_multivariate_reg": 9,
}
# These are all including the 'UNK' tokens. So, e.g., there are 2 real options for 'event_type'.
VOCAB_SIZES_BY_MEASUREMENT = {
    "event_type": 2,
    "static_clf": 3,
    "unused": 1,
    "age": 1,
    "tod": 5,
    "dynamic_single_label_clf": 3,
    "dynamic_multi_label_clf": 4,
    "dynamic_univariate_reg": 1,
    "dynamic_multivariate_reg": 3,
}
VOCAB_OFFSETS_BY_MEASUREMENT = {
    "event_type": 1,
    "static_clf": 3,
    "unused": 6,
    "age": 7,
    "tod": 8,
    "dynamic_single_label_clf": 13,
    "dynamic_multi_label_clf": 16,
    "dynamic_univariate_reg": 20,
    "dynamic_multivariate_reg": 21,
}
EVENT_TYPES_IDXMAP = {
    "event_A": 0,
    "event_B": 1,
}

UNIFIED_VOCABULARY = {
    "event_type": ["event_A", "event_B"],
    "static_clf": ["UNK", "static_clf_1", "static_clf_2"],
    "age": None,
    "tod": ['UNK', 'EARLY_AM', 'LATE_PM', 'AM', 'PM'],
    "dynamic_single_label_clf": ["UNK", "dynamic_single_label_1", "dynamic_single_label_2"],
    "dynamic_multi_label_clf": [
        "UNK", "dynamic_multi_label_1", "dynamic_multi_label_2", "dynamic_multi_label_3"
    ],
    "dynamic_univariate_reg": None,
    "dynamic_multivariate_reg": ["UNK", "dynamic_multivariate_reg_1", "dynamic_multivariate_reg_2"],
}
UNIFIED_IDXMAP = {
    "event_type": {"event_A": 1, "event_B": 2},
    "static_clf": {"UNK": 3, "static_clf_1": 4, "static_clf_2": 5},
    "age": {None: 7},
    "tod": {'UNK': 8, 'EARLY_AM': 9, 'LATE_PM': 10, 'AM': 11, 'PM': 12},
    "dynamic_single_label_clf": {"UNK": 13, "dynamic_single_label_1": 14, "dynamic_single_label_2": 15},
    "dynamic_multi_label_clf": {
        "UNK": 16,
        "dynamic_multi_label_1": 17,
        "dynamic_multi_label_2": 18,
        "dynamic_multi_label_3": 19,
    },
    "dynamic_univariate_reg": {None: 20},
    "dynamic_multivariate_reg": {
        "UNK": 21, "dynamic_multivariate_reg_1": 22, "dynamic_multivariate_reg_2": 23
    },
}

# Subject date of births
SUBJECT_DOB = [
    datetime(1990, 1, 1),
    datetime(1994, 1, 2),
]

SUBJECT_START_TIMES = [
    datetime(2020, 1, 1, 9, 30, 0),
    datetime(2020, 1, 1, 15, 0, 0)
]

SUBJECT_EVENT_TIMES = [
    [
        None,
        SUBJECT_START_TIMES[0],
        SUBJECT_START_TIMES[0] + timedelta(hours=1)
    ],
    [
        SUBJECT_START_TIMES[1],
        SUBJECT_START_TIMES[1] + timedelta(hours=3),
        SUBJECT_START_TIMES[1] + timedelta(hours=3) + timedelta(hours=5),
    ],
]

age_mean = MEASUREMENT_CONFIGS['age'].measurement_metadata['normalizer']['mean_']
age_std = MEASUREMENT_CONFIGS['age'].measurement_metadata['normalizer']['std_']
age_thresh_large = MEASUREMENT_CONFIGS['age'].measurement_metadata['outlier_model']['thresh_large_']
age_thresh_small = MEASUREMENT_CONFIGS['age'].measurement_metadata['outlier_model']['thresh_small_']

SUBJECT_AGES_AT_EVENTS = []
for dob, event_times in zip(SUBJECT_DOB, SUBJECT_EVENT_TIMES):
    ages = []
    for event_time in event_times:
        if event_time is None:
            ages.append(None)
        else:
            age = (event_time - dob) / timedelta(microseconds=1) / 1e6 / 60 / 60 / 24 / 365.25
            if age > age_thresh_large or age < age_thresh_small:
                raise NotImplementedError(f"Age {age} is outside of the range of the outlier model.")
            else:
                ages.append((age - age_mean) / age_std)
    SUBJECT_AGES_AT_EVENTS.append(ages)

BASE_BATCH = {
    "event_mask": torch.BoolTensor([
        [False, True, True],
        [True, True, True]
    ]),
    "start_time": torch.FloatTensor([
        SUBJECT_START_TIMES[0].timestamp() / 60,
        SUBJECT_START_TIMES[1].timestamp() / 60,
    ]),
    "time_delta": torch.FloatTensor([
        [1, 1*60, 1], # NA, AM, AM
        [3*60, 5*60, 1], # PM, PM, LATE_PM
    ]),
    "static_indices": torch.LongTensor([
        [UNIFIED_IDXMAP["static_clf"]["static_clf_1"]],
        [UNIFIED_IDXMAP["static_clf"]["static_clf_2"]],
    ]),
    "static_measurement_indices": torch.FloatTensor([
        [MEASUREMENTS_IDXMAP["static_clf"]],
        [MEASUREMENTS_IDXMAP["static_clf"]],
    ]),
    "dynamic_measurement_indices": torch.LongTensor([
        [
            [0, 0, 0, 0, 0, 0], # This event is padding
            [
                MEASUREMENTS_IDXMAP["event_type"], MEASUREMENTS_IDXMAP["age"], MEASUREMENTS_IDXMAP["tod"],
                MEASUREMENTS_IDXMAP["dynamic_univariate_reg"],
                MEASUREMENTS_IDXMAP["dynamic_multi_label_clf"], MEASUREMENTS_IDXMAP["dynamic_multi_label_clf"]
            ],
            [
                MEASUREMENTS_IDXMAP["event_type"], MEASUREMENTS_IDXMAP["age"], MEASUREMENTS_IDXMAP["tod"],
                MEASUREMENTS_IDXMAP["dynamic_multivariate_reg"],
                MEASUREMENTS_IDXMAP["dynamic_single_label_clf"],
                MEASUREMENTS_IDXMAP["dynamic_multivariate_reg"]
            ],
        ], [
            [
                MEASUREMENTS_IDXMAP["event_type"], MEASUREMENTS_IDXMAP["age"], MEASUREMENTS_IDXMAP["tod"],
                MEASUREMENTS_IDXMAP["dynamic_multi_label_clf"], 0, 0
            ],
            [
                MEASUREMENTS_IDXMAP["event_type"], MEASUREMENTS_IDXMAP["age"], MEASUREMENTS_IDXMAP["tod"],
                0, 0, 0
            ],
            [
                MEASUREMENTS_IDXMAP["event_type"], MEASUREMENTS_IDXMAP["age"], MEASUREMENTS_IDXMAP["tod"],
                0, 0, 0
            ],
        ]
    ]),
    "dynamic_indices": torch.LongTensor([
        [
            [0, 0, 0, 0, 0, 0], # This event is padding
            [
                UNIFIED_IDXMAP["event_type"]["event_B"],
                UNIFIED_IDXMAP["age"][None],
                UNIFIED_IDXMAP["tod"]["AM"],
                UNIFIED_IDXMAP["dynamic_univariate_reg"][None],
                UNIFIED_IDXMAP["dynamic_multi_label_clf"]["dynamic_multi_label_1"],
                UNIFIED_IDXMAP["dynamic_multi_label_clf"]["dynamic_multi_label_2"]
            ],
            [
                UNIFIED_IDXMAP["event_type"]["event_B"],
                UNIFIED_IDXMAP["age"][None],
                UNIFIED_IDXMAP["tod"]["AM"],
                UNIFIED_IDXMAP["dynamic_multivariate_reg"]["dynamic_multivariate_reg_1"],
                UNIFIED_IDXMAP["dynamic_single_label_clf"]["dynamic_single_label_1"],
                UNIFIED_IDXMAP["dynamic_multivariate_reg"]["dynamic_multivariate_reg_2"]
            ],
        ], [
            [
                UNIFIED_IDXMAP["event_type"]["event_B"],
                UNIFIED_IDXMAP["age"][None],
                UNIFIED_IDXMAP["tod"]["PM"],
                UNIFIED_IDXMAP["dynamic_multi_label_clf"]["dynamic_multi_label_1"], 0, 0
            ],
            [
                UNIFIED_IDXMAP["event_type"]["event_A"],
                UNIFIED_IDXMAP["age"][None],
                UNIFIED_IDXMAP["tod"]["PM"],
                0, 0, 0
            ],
            [
                UNIFIED_IDXMAP["event_type"]["event_A"],
                UNIFIED_IDXMAP["age"][None],
                UNIFIED_IDXMAP["tod"]["LATE_PM"],
                0, 0, 0
            ],
        ]
    ]),
    "dynamic_values": torch.Tensor([
        [
            [0, 0, 0, 0, 0, 0],
            [0, SUBJECT_AGES_AT_EVENTS[0][1], 0, 0.2, 0, 0],
            [0, SUBJECT_AGES_AT_EVENTS[0][2], 0, 0.3, 0, 0.4],
        ], [
            [0, SUBJECT_AGES_AT_EVENTS[1][0], 0, 0, 0, 0],
            [0, SUBJECT_AGES_AT_EVENTS[1][1], 0, 0, 0, 0],
            [0, SUBJECT_AGES_AT_EVENTS[1][2], 0, 0, 0, 0],
        ]
    ]),
    "dynamic_values_mask": torch.BoolTensor([
        [
            [False, False, False, False, False, False],
            [False, True, False, True, False, False],
            [False, True, False, True, False, True],
        ], [
            [False, True, False, False, False, False],
            [False, True, False, False, False, False],
            [False, True, False, False, False, False],
        ]
    ]),
}

# This makes an AM event and an EARLY_AM event.
#
# SUBJECT_START_TIMES = [
#     datetime(2020, 1, 1, 9, 59, 30),
#     datetime(2020, 1, 1, 15, 0, 0)
# ]
NEW_EVENT_DELTA_TIMES = [1*60, 2*60]
NEW_EVENT_AGES = []
for subj_dob, event_times, new_event_delta_T in zip(SUBJECT_DOB, SUBJECT_EVENT_TIMES, NEW_EVENT_DELTA_TIMES):
    age = event_times[-1] + timedelta(minutes=new_event_delta_T) - subj_dob
    age = age / timedelta(microseconds=1) / 1e6 / 60 / 60 / 24 / 365.25
    if age > age_thresh_large or age < age_thresh_small:
        raise NotImplementedError(f"Age {age} is outside of the range of the outlier model.")
    else:
        NEW_EVENT_AGES.append((age - age_mean) / age_std)

WANT_APPENDED_BATCH = {
    "event_mask": torch.BoolTensor([
        [False, True, True, True],
        [True, True, True, True]
    ]),
    "start_time": copy.deepcopy(BASE_BATCH['start_time']),
    "time_delta": torch.FloatTensor([
        [1, 1*60, NEW_EVENT_DELTA_TIMES[0], 1], # NA, AM, AM, AM
        [3*60, 5*60, NEW_EVENT_DELTA_TIMES[1], 1], # PM, PM, LATE_PM, EARLY_AM
    ]),
    "static_indices": copy.deepcopy(BASE_BATCH['static_indices']),
    "static_measurement_indices": copy.deepcopy(BASE_BATCH['static_measurement_indices']),
    "dynamic_measurement_indices": torch.LongTensor([
        [
            [0, 0, 0, 0, 0, 0], # This event is padding
            [
                MEASUREMENTS_IDXMAP["event_type"], MEASUREMENTS_IDXMAP["age"], MEASUREMENTS_IDXMAP["tod"],
                MEASUREMENTS_IDXMAP["dynamic_univariate_reg"],
                MEASUREMENTS_IDXMAP["dynamic_multi_label_clf"], MEASUREMENTS_IDXMAP["dynamic_multi_label_clf"]
            ],
            [
                MEASUREMENTS_IDXMAP["event_type"], MEASUREMENTS_IDXMAP["age"], MEASUREMENTS_IDXMAP["tod"],
                MEASUREMENTS_IDXMAP["dynamic_multivariate_reg"],
                MEASUREMENTS_IDXMAP["dynamic_single_label_clf"],
                MEASUREMENTS_IDXMAP["dynamic_multivariate_reg"]
            ],
            [
                MEASUREMENTS_IDXMAP["age"], MEASUREMENTS_IDXMAP["tod"],
                0, 0, 0, 0
            ],
        ], [
            [
                MEASUREMENTS_IDXMAP["event_type"], MEASUREMENTS_IDXMAP["age"], MEASUREMENTS_IDXMAP["tod"],
                MEASUREMENTS_IDXMAP["dynamic_multi_label_clf"], 0, 0
            ],
            [
                MEASUREMENTS_IDXMAP["event_type"], MEASUREMENTS_IDXMAP["age"], MEASUREMENTS_IDXMAP["tod"],
                0, 0, 0
            ],
            [
                MEASUREMENTS_IDXMAP["event_type"], MEASUREMENTS_IDXMAP["age"], MEASUREMENTS_IDXMAP["tod"],
                0, 0, 0
            ],
            [
                MEASUREMENTS_IDXMAP["age"], MEASUREMENTS_IDXMAP["tod"],
                0, 0, 0, 0
            ],
        ]
    ]),
    "dynamic_indices": torch.LongTensor([
        [
            [0, 0, 0, 0, 0, 0], # This event is padding
            [
                UNIFIED_IDXMAP["event_type"]["event_B"],
                UNIFIED_IDXMAP["age"][None],
                UNIFIED_IDXMAP["tod"]["AM"],
                UNIFIED_IDXMAP["dynamic_univariate_reg"][None],
                UNIFIED_IDXMAP["dynamic_multi_label_clf"]["dynamic_multi_label_1"],
                UNIFIED_IDXMAP["dynamic_multi_label_clf"]["dynamic_multi_label_2"]
            ],
            [
                UNIFIED_IDXMAP["event_type"]["event_B"],
                UNIFIED_IDXMAP["age"][None],
                UNIFIED_IDXMAP["tod"]["AM"],
                UNIFIED_IDXMAP["dynamic_multivariate_reg"]["dynamic_multivariate_reg_1"],
                UNIFIED_IDXMAP["dynamic_single_label_clf"]["dynamic_single_label_1"],
                UNIFIED_IDXMAP["dynamic_multivariate_reg"]["dynamic_multivariate_reg_2"]
            ],
            [
                UNIFIED_IDXMAP["age"][None], UNIFIED_IDXMAP["tod"]["AM"],
                0, 0, 0, 0
            ],
        ], [
            [
                UNIFIED_IDXMAP["event_type"]["event_B"],
                UNIFIED_IDXMAP["age"][None],
                UNIFIED_IDXMAP["tod"]["PM"],
                UNIFIED_IDXMAP["dynamic_multi_label_clf"]["dynamic_multi_label_1"], 0, 0
            ],
            [
                UNIFIED_IDXMAP["event_type"]["event_A"],
                UNIFIED_IDXMAP["age"][None],
                UNIFIED_IDXMAP["tod"]["PM"],
                0, 0, 0
            ],
            [
                UNIFIED_IDXMAP["event_type"]["event_A"],
                UNIFIED_IDXMAP["age"][None],
                UNIFIED_IDXMAP["tod"]["LATE_PM"],
                0, 0, 0
            ],
            [
                UNIFIED_IDXMAP["age"][None], UNIFIED_IDXMAP["tod"]["EARLY_AM"],
                0, 0, 0, 0
            ],
        ]
    ]),
    "dynamic_values": torch.Tensor([
        [
            [0, 0, 0, 0, 0, 0],
            [0, SUBJECT_AGES_AT_EVENTS[0][1], 0, 0.2, 0, 0],
            [0, SUBJECT_AGES_AT_EVENTS[0][2], 0, 0.3, 0, 0.4],
            [NEW_EVENT_AGES[0], 0, 0, 0, 0, 0],
        ], [
            [0, SUBJECT_AGES_AT_EVENTS[1][0], 0, 0, 0, 0],
            [0, SUBJECT_AGES_AT_EVENTS[1][1], 0, 0, 0, 0],
            [0, SUBJECT_AGES_AT_EVENTS[1][2], 0, 0, 0, 0],
            [NEW_EVENT_AGES[1], 0, 0, 0, 0, 0],
        ]
    ]),
    "dynamic_values_mask": torch.BoolTensor([
        [
            [False, False, False, False, False, False],
            [False, True, False, True, False, False],
            [False, True, False, True, False, True],
            [True, False, False, False, False, False],
        ], [
            [False, True, False, False, False, False],
            [False, True, False, False, False, False],
            [False, True, False, False, False, False],
            [True, False, False, False, False, False],
        ]
    ]),
    'stream_labels': None,
}

class TestGenerativeSequenceModelSamples(MLTypeEqualityCheckableMixin, unittest.TestCase):
    """Tests Generation Batch-building Logic."""

    def setUp(self):
        super().setUp()
        self.batch = PytorchBatch(**BASE_BATCH)
        self.config = StructuredTransformerConfig(
            vocab_offsets_by_measurement = VOCAB_OFFSETS_BY_MEASUREMENT,
            vocab_sizes_by_measurement = VOCAB_SIZES_BY_MEASUREMENT,
            measurements_idxmap = MEASUREMENTS_IDXMAP,
            measurement_configs = MEASUREMENT_CONFIGS,
            event_types_per_measurement = EVENT_TYPES_PER_MEASUREMENT,
            event_types_idxmap = EVENT_TYPES_IDXMAP,
        )
        self.samp = GenerativeSequenceModelSamples(
            event_mask = torch.BoolTensor([True, True]),
            time_to_event = torch.FloatTensor(NEW_EVENT_DELTA_TIMES),
            classification = {},
            regression = {},
            regression_indices = {},
        )

    def test_strip_unused_indices(self):
        T = torch.LongTensor([
            [1, 0, 4, 5, 0, 1],
            [8, 2, 0, 0, 0, 0],
            [3, 5, 0, 0, 0, 6],
        ])

        got = strip_unused_indices(T)
        want = torch.LongTensor([
            [1, 4, 5, 1],
            [8, 2, 0, 0],
            [3, 5, 6, 0],
        ])

        self.assertEqual(got, want)

        T2 = torch.FloatTensor([
            [1., 0, 0, 0, 0, 0],
            [0, 2., 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 5],
        ])

        T3 = torch.BoolTensor([
            [True, False, True, False, False, False],
            [False, True, False, False, False, False],
            [True, False, False, False, False, True],
        ])
        got_T1, got_T2, got_T3 = strip_unused_indices(T, T2, T3)

        want_T2 = torch.FloatTensor([
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [4, 0, 5, 0],
        ])
        want_T3 = torch.BoolTensor([
            [True, True, False, False],
            [False, True, False, False],
            [True, False, True, False],
        ])
        self.assertEqual(got_T1, want)
        self.assertEqual(got_T2, want_T2)
        self.assertEqual(got_T3, want_T3)

    def test_build_new_batch_element(self):
        got_batch = self.samp.append_to_batch(self.batch, self.config)

        self.assertNestedDictEqual(WANT_APPENDED_BATCH, {k: v for k, v in got_batch.items()})

        for param in ('static_indices', 'start_time', 'static_measurement_indices'):
            with self.subTest(f"{param} should not change"):
                self.assertEqual(getattr(self.batch, param), getattr(got_batch, param))

if __name__ == "__main__":
    unittest.main()

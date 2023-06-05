import sys

sys.path.append("../..")

import copy
import unittest
from datetime import datetime, timedelta

import pandas as pd
import torch
from pytorch_lognormal_mixture import LogNormalMixtureDistribution

from EventStream.data.config import MeasurementConfig
from EventStream.data.time_dependent_functor import AgeFunctor, TimeOfDayFunctor
from EventStream.data.types import DataModality, PytorchBatch, TemporalityType
from EventStream.data.vocabulary import Vocabulary
from EventStream.transformer.config import StructuredTransformerConfig
from EventStream.transformer.model_output import (
    GenerativeOutputLayerBase,
    GenerativeSequenceModelSamples,
    strip_unused_indices,
)

from ..utils import MLTypeEqualityCheckableMixin

MEASUREMENT_CONFIGS = {
    "static_clf": MeasurementConfig(
        temporality=TemporalityType.STATIC,
        modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
        vocabulary=Vocabulary(
            vocabulary=["UNK", "static_clf_1", "static_clf_2"],
            obs_frequencies=[0.1, 0.5, 0.4],
        ),
    ),
    "age": MeasurementConfig(
        temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
        functor=AgeFunctor("dob"),
        vocabulary=None,
        _measurement_metadata=pd.Series(
            {
                "value_type": "float",
                "normalizer": {"mean_": 40.0, "std_": 10.0},
                "outlier_model": {"thresh_large_": 90.0, "thresh_small_": 18.0},
            }
        ),
    ),
    "tod": MeasurementConfig(
        temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
        functor=TimeOfDayFunctor(),
        vocabulary=Vocabulary(
            vocabulary=["UNK", "EARLY_AM", "LATE_PM", "AM", "PM"],
            obs_frequencies=[0.1, 0.3, 0.22, 0.2, 0.18],
        ),
    ),
    "dynamic_multi_label_clf": MeasurementConfig(
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.MULTI_LABEL_CLASSIFICATION,
        vocabulary=Vocabulary(
            vocabulary=[
                "UNK",
                "dynamic_multi_label_1",
                "dynamic_multi_label_2",
                "dynamic_multi_label_3",
            ],
            obs_frequencies=[0.1, 0.5, 0.25, 0.15],
        ),
    ),
    "dynamic_univariate_reg": MeasurementConfig(
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.UNIVARIATE_REGRESSION,
        vocabulary=None,
        _measurement_metadata=pd.Series(
            {
                "value_type": "float",
                "normalizer": {"mean_": 2.0, "std_": 3.0},
                "outlier_model": {"thresh_large_": 7.0, "thresh_small_": -3.0},
            }
        ),
    ),
    "dynamic_multivariate_reg": MeasurementConfig(
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.MULTIVARIATE_REGRESSION,
        values_column="dynamic_multivariate_reg_values",
        vocabulary=Vocabulary(
            vocabulary=["UNK", "dynamic_multivariate_reg_1", "dynamic_multivariate_reg_2"],
            obs_frequencies=[0.1, 0.5, 0.4],
        ),
        _measurement_metadata=pd.DataFrame(
            {
                "value_type": ["float", "float"],
                "normalizer": [{"mean_": 2.0, "std_": 3.0}, {"mean_": 4.0, "std_": 5.0}],
                "outlier_model": [
                    {"thresh_large_": 7.0, "thresh_small_": -3.0},
                    {"thresh_large_": 9.0, "thresh_small_": -5.0},
                ],
            },
            index=pd.Index(
                ["dynamic_multivariate_reg_1", "dynamic_multivariate_reg_2"],
                name="dynamic_multivariate_reg",
            ),
        ),
    ),
}

MEASUREMENTS_PER_GEN_MODE = {
    DataModality.SINGLE_LABEL_CLASSIFICATION: ["event_type"],
    DataModality.MULTI_LABEL_CLASSIFICATION: [
        "dynamic_multi_label_clf",
        "dynamic_multivariate_reg",
    ],
    DataModality.MULTIVARIATE_REGRESSION: ["dynamic_multivariate_reg"],
    DataModality.UNIVARIATE_REGRESSION: ["dynamic_univariate_reg"],
}
MEASUREMENTS_IDXMAP = {
    "event_type": 1,
    "static_clf": 2,
    "unused": 3,
    "age": 4,
    "tod": 5,
    "UNUSED": 6,
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
    "tod": ["UNK", "EARLY_AM", "LATE_PM", "AM", "PM"],
    "dynamic_multi_label_clf": [
        "UNK",
        "dynamic_multi_label_1",
        "dynamic_multi_label_2",
        "dynamic_multi_label_3",
    ],
    "dynamic_univariate_reg": None,
    "dynamic_multivariate_reg": [
        "UNK",
        "dynamic_multivariate_reg_1",
        "dynamic_multivariate_reg_2",
    ],
}
UNIFIED_IDXMAP = {
    "event_type": {"event_A": 1, "event_B": 2},
    "static_clf": {"UNK": 3, "static_clf_1": 4, "static_clf_2": 5},
    "age": {None: 7},
    "tod": {"UNK": 8, "EARLY_AM": 9, "LATE_PM": 10, "AM": 11, "PM": 12},
    "dynamic_multi_label_clf": {
        "UNK": 16,
        "dynamic_multi_label_1": 17,
        "dynamic_multi_label_2": 18,
        "dynamic_multi_label_3": 19,
    },
    "dynamic_univariate_reg": {None: 20},
    "dynamic_multivariate_reg": {
        "UNK": 21,
        "dynamic_multivariate_reg_1": 22,
        "dynamic_multivariate_reg_2": 23,
    },
}

# Subject date of births
SUBJECT_DOB = [
    datetime(1990, 1, 1),
    datetime(1994, 1, 2),
]

SUBJECT_START_TIMES = [datetime(2020, 1, 1, 9, 30, 0), datetime(2020, 1, 1, 15, 0, 0)]

SUBJECT_EVENT_TIMES = [
    [None, SUBJECT_START_TIMES[0], SUBJECT_START_TIMES[0] + timedelta(hours=1)],
    [
        SUBJECT_START_TIMES[1],
        SUBJECT_START_TIMES[1] + timedelta(hours=3),
        SUBJECT_START_TIMES[1] + timedelta(hours=3) + timedelta(hours=5),
    ],
]

age_mean = MEASUREMENT_CONFIGS["age"].measurement_metadata["normalizer"]["mean_"]
age_std = MEASUREMENT_CONFIGS["age"].measurement_metadata["normalizer"]["std_"]
age_thresh_large = MEASUREMENT_CONFIGS["age"].measurement_metadata["outlier_model"]["thresh_large_"]
age_thresh_small = MEASUREMENT_CONFIGS["age"].measurement_metadata["outlier_model"]["thresh_small_"]

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
    "event_mask": torch.BoolTensor([[False, True, True], [True, True, True]]),
    "start_time": torch.FloatTensor(
        [
            SUBJECT_START_TIMES[0].timestamp() / 60,
            SUBJECT_START_TIMES[1].timestamp() / 60,
        ]
    ),
    "time_delta": torch.FloatTensor(
        [
            [1, 1 * 60, 1],  # NA, AM, AM
            [3 * 60, 5 * 60, 1],  # PM, PM, LATE_PM
        ]
    ),
    "time": None,
    "static_indices": torch.LongTensor(
        [
            [UNIFIED_IDXMAP["static_clf"]["static_clf_1"]],
            [UNIFIED_IDXMAP["static_clf"]["static_clf_2"]],
        ]
    ),
    "static_measurement_indices": torch.FloatTensor(
        [
            [MEASUREMENTS_IDXMAP["static_clf"]],
            [MEASUREMENTS_IDXMAP["static_clf"]],
        ]
    ),
    "dynamic_measurement_indices": torch.LongTensor(
        [
            [
                [0, 0, 0, 0, 0, 0],  # This event is padding
                [
                    MEASUREMENTS_IDXMAP["event_type"],
                    MEASUREMENTS_IDXMAP["age"],
                    MEASUREMENTS_IDXMAP["tod"],
                    MEASUREMENTS_IDXMAP["dynamic_univariate_reg"],
                    MEASUREMENTS_IDXMAP["dynamic_multi_label_clf"],
                    MEASUREMENTS_IDXMAP["dynamic_multi_label_clf"],
                ],
                [
                    MEASUREMENTS_IDXMAP["event_type"],
                    MEASUREMENTS_IDXMAP["age"],
                    MEASUREMENTS_IDXMAP["tod"],
                    MEASUREMENTS_IDXMAP["dynamic_multivariate_reg"],
                    MEASUREMENTS_IDXMAP["dynamic_multivariate_reg"],
                    0,
                ],
            ],
            [
                [
                    MEASUREMENTS_IDXMAP["event_type"],
                    MEASUREMENTS_IDXMAP["age"],
                    MEASUREMENTS_IDXMAP["tod"],
                    MEASUREMENTS_IDXMAP["dynamic_multi_label_clf"],
                    0,
                    0,
                ],
                [
                    MEASUREMENTS_IDXMAP["event_type"],
                    MEASUREMENTS_IDXMAP["age"],
                    MEASUREMENTS_IDXMAP["tod"],
                    0,
                    0,
                    0,
                ],
                [
                    MEASUREMENTS_IDXMAP["event_type"],
                    MEASUREMENTS_IDXMAP["age"],
                    MEASUREMENTS_IDXMAP["tod"],
                    0,
                    0,
                    0,
                ],
            ],
        ]
    ),
    "dynamic_indices": torch.LongTensor(
        [
            [
                [0, 0, 0, 0, 0, 0],  # This event is padding
                [
                    UNIFIED_IDXMAP["event_type"]["event_B"],
                    UNIFIED_IDXMAP["age"][None],
                    UNIFIED_IDXMAP["tod"]["AM"],
                    UNIFIED_IDXMAP["dynamic_univariate_reg"][None],
                    UNIFIED_IDXMAP["dynamic_multi_label_clf"]["dynamic_multi_label_1"],
                    UNIFIED_IDXMAP["dynamic_multi_label_clf"]["dynamic_multi_label_2"],
                ],
                [
                    UNIFIED_IDXMAP["event_type"]["event_B"],
                    UNIFIED_IDXMAP["age"][None],
                    UNIFIED_IDXMAP["tod"]["AM"],
                    UNIFIED_IDXMAP["dynamic_multivariate_reg"]["dynamic_multivariate_reg_1"],
                    UNIFIED_IDXMAP["dynamic_multivariate_reg"]["dynamic_multivariate_reg_2"],
                    0,
                ],
            ],
            [
                [
                    UNIFIED_IDXMAP["event_type"]["event_B"],
                    UNIFIED_IDXMAP["age"][None],
                    UNIFIED_IDXMAP["tod"]["PM"],
                    UNIFIED_IDXMAP["dynamic_multi_label_clf"]["dynamic_multi_label_1"],
                    0,
                    0,
                ],
                [
                    UNIFIED_IDXMAP["event_type"]["event_A"],
                    UNIFIED_IDXMAP["age"][None],
                    UNIFIED_IDXMAP["tod"]["PM"],
                    0,
                    0,
                    0,
                ],
                [
                    UNIFIED_IDXMAP["event_type"]["event_A"],
                    UNIFIED_IDXMAP["age"][None],
                    UNIFIED_IDXMAP["tod"]["LATE_PM"],
                    0,
                    0,
                    0,
                ],
            ],
        ]
    ),
    "dynamic_values": torch.Tensor(
        [
            [
                [0, 0, 0, 0, 0, 0],
                [0, SUBJECT_AGES_AT_EVENTS[0][1], 0, 0.2, 0, 0],
                [0, SUBJECT_AGES_AT_EVENTS[0][2], 0, 0.3, 0, 0.4],
            ],
            [
                [0, SUBJECT_AGES_AT_EVENTS[1][0], 0, 0, 0, 0],
                [0, SUBJECT_AGES_AT_EVENTS[1][1], 0, 0, 0, 0],
                [0, SUBJECT_AGES_AT_EVENTS[1][2], 0, 0, 0, 0],
            ],
        ]
    ),
    "dynamic_values_mask": torch.BoolTensor(
        [
            [
                [False, False, False, False, False, False],
                [False, True, False, True, False, False],
                [False, True, False, True, False, True],
            ],
            [
                [False, True, False, False, False, False],
                [False, True, False, False, False, False],
                [False, True, False, False, False, False],
            ],
        ]
    ),
}

# This makes an AM event and an EARLY_AM event.
#
# SUBJECT_START_TIMES = [
#     datetime(2020, 1, 1, 9, 59, 30),
#     datetime(2020, 1, 1, 15, 0, 0)
# ]
NEW_EVENT_DELTA_TIMES = [1 * 60, 2 * 60]
NEW_EVENT_AGES = []
for subj_dob, event_times, new_event_delta_T in zip(SUBJECT_DOB, SUBJECT_EVENT_TIMES, NEW_EVENT_DELTA_TIMES):
    age = event_times[-1] + timedelta(minutes=new_event_delta_T) - subj_dob
    age = age / timedelta(microseconds=1) / 1e6 / 60 / 60 / 24 / 365.25
    if age > age_thresh_large or age < age_thresh_small:
        raise NotImplementedError(f"Age {age} is outside of the range of the outlier model.")
    else:
        NEW_EVENT_AGES.append((age - age_mean) / age_std)

WANT_APPENDED_BATCH = {
    "event_mask": torch.BoolTensor([[False, True, True, True], [True, True, True, True]]),
    "start_time": copy.deepcopy(BASE_BATCH["start_time"]),
    "time_delta": torch.FloatTensor(
        [
            [1, 1 * 60, NEW_EVENT_DELTA_TIMES[0], 1],  # NA, AM, AM, AM
            [3 * 60, 5 * 60, NEW_EVENT_DELTA_TIMES[1], 1],  # PM, PM, LATE_PM, EARLY_AM
        ]
    ),
    "time": None,
    "static_indices": copy.deepcopy(BASE_BATCH["static_indices"]),
    "static_measurement_indices": copy.deepcopy(BASE_BATCH["static_measurement_indices"]),
    "dynamic_measurement_indices": torch.LongTensor(
        [
            [
                [0, 0, 0, 0, 0, 0],  # This event is padding
                [
                    MEASUREMENTS_IDXMAP["event_type"],
                    MEASUREMENTS_IDXMAP["age"],
                    MEASUREMENTS_IDXMAP["tod"],
                    MEASUREMENTS_IDXMAP["dynamic_univariate_reg"],
                    MEASUREMENTS_IDXMAP["dynamic_multi_label_clf"],
                    MEASUREMENTS_IDXMAP["dynamic_multi_label_clf"],
                ],
                [
                    MEASUREMENTS_IDXMAP["event_type"],
                    MEASUREMENTS_IDXMAP["age"],
                    MEASUREMENTS_IDXMAP["tod"],
                    MEASUREMENTS_IDXMAP["dynamic_multivariate_reg"],
                    MEASUREMENTS_IDXMAP["dynamic_multivariate_reg"],
                    0,
                ],
                [MEASUREMENTS_IDXMAP["age"], MEASUREMENTS_IDXMAP["tod"], 0, 0, 0, 0],
            ],
            [
                [
                    MEASUREMENTS_IDXMAP["event_type"],
                    MEASUREMENTS_IDXMAP["age"],
                    MEASUREMENTS_IDXMAP["tod"],
                    MEASUREMENTS_IDXMAP["dynamic_multi_label_clf"],
                    0,
                    0,
                ],
                [
                    MEASUREMENTS_IDXMAP["event_type"],
                    MEASUREMENTS_IDXMAP["age"],
                    MEASUREMENTS_IDXMAP["tod"],
                    0,
                    0,
                    0,
                ],
                [
                    MEASUREMENTS_IDXMAP["event_type"],
                    MEASUREMENTS_IDXMAP["age"],
                    MEASUREMENTS_IDXMAP["tod"],
                    0,
                    0,
                    0,
                ],
                [MEASUREMENTS_IDXMAP["age"], MEASUREMENTS_IDXMAP["tod"], 0, 0, 0, 0],
            ],
        ]
    ),
    "dynamic_indices": torch.LongTensor(
        [
            [
                [0, 0, 0, 0, 0, 0],  # This event is padding
                [
                    UNIFIED_IDXMAP["event_type"]["event_B"],
                    UNIFIED_IDXMAP["age"][None],
                    UNIFIED_IDXMAP["tod"]["AM"],
                    UNIFIED_IDXMAP["dynamic_univariate_reg"][None],
                    UNIFIED_IDXMAP["dynamic_multi_label_clf"]["dynamic_multi_label_1"],
                    UNIFIED_IDXMAP["dynamic_multi_label_clf"]["dynamic_multi_label_2"],
                ],
                [
                    UNIFIED_IDXMAP["event_type"]["event_B"],
                    UNIFIED_IDXMAP["age"][None],
                    UNIFIED_IDXMAP["tod"]["AM"],
                    UNIFIED_IDXMAP["dynamic_multivariate_reg"]["dynamic_multivariate_reg_1"],
                    UNIFIED_IDXMAP["dynamic_multivariate_reg"]["dynamic_multivariate_reg_2"],
                    0,
                ],
                [UNIFIED_IDXMAP["age"][None], UNIFIED_IDXMAP["tod"]["AM"], 0, 0, 0, 0],
            ],
            [
                [
                    UNIFIED_IDXMAP["event_type"]["event_B"],
                    UNIFIED_IDXMAP["age"][None],
                    UNIFIED_IDXMAP["tod"]["PM"],
                    UNIFIED_IDXMAP["dynamic_multi_label_clf"]["dynamic_multi_label_1"],
                    0,
                    0,
                ],
                [
                    UNIFIED_IDXMAP["event_type"]["event_A"],
                    UNIFIED_IDXMAP["age"][None],
                    UNIFIED_IDXMAP["tod"]["PM"],
                    0,
                    0,
                    0,
                ],
                [
                    UNIFIED_IDXMAP["event_type"]["event_A"],
                    UNIFIED_IDXMAP["age"][None],
                    UNIFIED_IDXMAP["tod"]["LATE_PM"],
                    0,
                    0,
                    0,
                ],
                [UNIFIED_IDXMAP["age"][None], UNIFIED_IDXMAP["tod"]["EARLY_AM"], 0, 0, 0, 0],
            ],
        ]
    ),
    "dynamic_values": torch.Tensor(
        [
            [
                [0, 0, 0, 0, 0, 0],
                [0, SUBJECT_AGES_AT_EVENTS[0][1], 0, 0.2, 0, 0],
                [0, SUBJECT_AGES_AT_EVENTS[0][2], 0, 0.3, 0, 0.4],
                [NEW_EVENT_AGES[0], 0, 0, 0, 0, 0],
            ],
            [
                [0, SUBJECT_AGES_AT_EVENTS[1][0], 0, 0, 0, 0],
                [0, SUBJECT_AGES_AT_EVENTS[1][1], 0, 0, 0, 0],
                [0, SUBJECT_AGES_AT_EVENTS[1][2], 0, 0, 0, 0],
                [NEW_EVENT_AGES[1], 0, 0, 0, 0, 0],
            ],
        ]
    ),
    "dynamic_values_mask": torch.BoolTensor(
        [
            [
                [False, False, False, False, False, False],
                [False, True, False, True, False, False],
                [False, True, False, True, False, True],
                [True, False, False, False, False, False],
            ],
            [
                [False, True, False, False, False, False],
                [False, True, False, False, False, False],
                [False, True, False, False, False, False],
                [True, False, False, False, False, False],
            ],
        ]
    ),
    "stream_labels": None,
}


# UNIFIED_VOCABULARY = {
#     "event_type": ["event_A", "event_B"],
#     "static_clf": ["UNK", "static_clf_1", "static_clf_2"],
#     "age": None,
#     "tod": ["UNK", "EARLY_AM", "LATE_PM", "AM", "PM"],
#     "dynamic_multi_label_clf": [
#         "UNK",
#         "dynamic_multi_label_1",
#         "dynamic_multi_label_2",
#         "dynamic_multi_label_3",
#     ],
#     "dynamic_univariate_reg": None,
#     "dynamic_multivariate_reg": [
#         "UNK",
#         "dynamic_multivariate_reg_1",
#         "dynamic_multivariate_reg_2",
#     ],
# }
CLASSIFICATION = {
    "event_type": torch.LongTensor([1, 1]),
    "dynamic_multi_label_clf": torch.LongTensor(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 1],
        ]
    ),
    "dynamic_multivariate_reg": torch.LongTensor(
        [
            [0, 0, 1],
            [0, 0, 0],
        ]
    ),
}
REGRESSION = {
    "dynamic_multivariate_reg": torch.FloatTensor(
        [
            [0, 0, 0.5],
            [0, 0, 0],
        ]
    ),
    "dynamic_univariate_reg": torch.FloatTensor([0.8, 0.2]),
}

WANT_UPDATED_DATA = [
    (
        [
            "event_type",
            ("dynamic_multivariate_reg", "categorical_only"),
        ],
        {
            "dynamic_measurement_indices": torch.LongTensor(
                [
                    [
                        MEASUREMENTS_IDXMAP["age"],
                        MEASUREMENTS_IDXMAP["tod"],
                        MEASUREMENTS_IDXMAP["event_type"],
                        MEASUREMENTS_IDXMAP["dynamic_multivariate_reg"],
                        0,
                        0,
                    ],
                    [
                        MEASUREMENTS_IDXMAP["age"],
                        MEASUREMENTS_IDXMAP["tod"],
                        MEASUREMENTS_IDXMAP["event_type"],
                        0,
                        0,
                        0,
                    ],
                ]
            ),
            "dynamic_indices": torch.LongTensor(
                [
                    [
                        UNIFIED_IDXMAP["age"][None],
                        UNIFIED_IDXMAP["tod"]["AM"],
                        UNIFIED_IDXMAP["event_type"]["event_B"],
                        UNIFIED_IDXMAP["dynamic_multivariate_reg"]["dynamic_multivariate_reg_2"],
                        0,
                        0,
                    ],
                    [
                        UNIFIED_IDXMAP["age"][None],
                        UNIFIED_IDXMAP["tod"]["EARLY_AM"],
                        UNIFIED_IDXMAP["event_type"]["event_B"],
                        0,
                        0,
                        0,
                    ],
                ]
            ),
            "dynamic_values": torch.FloatTensor(
                [
                    [NEW_EVENT_AGES[0], 0, 0, 0, 0, 0],
                    [NEW_EVENT_AGES[1], 0, 0, 0, 0, 0],
                ]
            ),
            "dynamic_values_mask": torch.BoolTensor(
                [
                    [True, False, False, False, False, False],
                    [True, False, False, False, False, False],
                ]
            ),
        },
    ),
    (
        [
            "dynamic_univariate_reg",
            "dynamic_multi_label_clf",
            ("dynamic_multivariate_reg", "numerical_only"),
        ],
        {
            "dynamic_measurement_indices": torch.LongTensor(
                [
                    [
                        MEASUREMENTS_IDXMAP["age"],
                        MEASUREMENTS_IDXMAP["tod"],
                        MEASUREMENTS_IDXMAP["event_type"],
                        MEASUREMENTS_IDXMAP["dynamic_univariate_reg"],
                        MEASUREMENTS_IDXMAP["dynamic_multivariate_reg"],
                        0,
                    ],
                    [
                        MEASUREMENTS_IDXMAP["age"],
                        MEASUREMENTS_IDXMAP["tod"],
                        MEASUREMENTS_IDXMAP["event_type"],
                        MEASUREMENTS_IDXMAP["dynamic_univariate_reg"],
                        MEASUREMENTS_IDXMAP["dynamic_multi_label_clf"],
                        MEASUREMENTS_IDXMAP["dynamic_multi_label_clf"],
                    ],
                ]
            ),
            "dynamic_indices": torch.LongTensor(
                [
                    [
                        UNIFIED_IDXMAP["age"][None],
                        UNIFIED_IDXMAP["tod"]["AM"],
                        UNIFIED_IDXMAP["event_type"]["event_B"],
                        UNIFIED_IDXMAP["dynamic_univariate_reg"][None],
                        UNIFIED_IDXMAP["dynamic_multivariate_reg"]["dynamic_multivariate_reg_2"],
                        0,
                    ],
                    [
                        UNIFIED_IDXMAP["age"][None],
                        UNIFIED_IDXMAP["tod"]["EARLY_AM"],
                        UNIFIED_IDXMAP["event_type"]["event_B"],
                        UNIFIED_IDXMAP["dynamic_univariate_reg"][None],
                        UNIFIED_IDXMAP["dynamic_multi_label_clf"]["dynamic_multi_label_1"],
                        UNIFIED_IDXMAP["dynamic_multi_label_clf"]["dynamic_multi_label_3"],
                    ],
                ]
            ),
            "dynamic_values": torch.FloatTensor(
                [
                    [
                        NEW_EVENT_AGES[0],
                        0,
                        0,
                        0.8,
                        0.5,
                        0,
                    ],
                    [
                        NEW_EVENT_AGES[1],
                        0,
                        0,
                        0.2,
                        0,
                        0,
                    ],
                ]
            ),
            "dynamic_values_mask": torch.BoolTensor(
                [
                    [True, False, False, True, True, False],
                    [True, False, False, True, False, False],
                ]
            ),
        },
    ),
]


class TestGenerativeSequenceModelSamples(MLTypeEqualityCheckableMixin, unittest.TestCase):
    """Tests Generation Batch-building Logic."""

    def setUp(self):
        super().setUp()
        self.batch = PytorchBatch(**BASE_BATCH)
        self.config = StructuredTransformerConfig(
            vocab_offsets_by_measurement=VOCAB_OFFSETS_BY_MEASUREMENT,
            vocab_sizes_by_measurement=VOCAB_SIZES_BY_MEASUREMENT,
            measurements_idxmap=MEASUREMENTS_IDXMAP,
            measurement_configs=MEASUREMENT_CONFIGS,
            event_types_idxmap=EVENT_TYPES_IDXMAP,
        )
        self.samp = GenerativeSequenceModelSamples(
            event_mask=torch.BoolTensor([True, True]),
            time_to_event=torch.FloatTensor(NEW_EVENT_DELTA_TIMES),
            classification=CLASSIFICATION,
            regression=REGRESSION,
            regression_indices=None,
        )

    def test_strip_unused_indices(self):
        T = torch.LongTensor(
            [
                [1, 0, 4, 5, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [3, 5, 0, 0, 0, 6],
            ]
        )

        got = strip_unused_indices(T)
        want = torch.LongTensor(
            [
                [1, 4, 5, 1],
                [0, 0, 0, 0],
                [3, 5, 6, 0],
            ]
        )

        self.assertEqual(got, want)

        T2 = torch.FloatTensor(
            [
                [1.0, 0, 0, 0, 0, 0],
                [0, 2.0, 0, 0, 0, 0],
                [4, 0, 0, 0, 0, 5],
            ]
        )

        T3 = torch.BoolTensor(
            [
                [True, False, True, False, False, False],
                [False, True, False, False, False, False],
                [True, False, False, False, False, True],
            ]
        )
        got_T1, got_T2, got_T3 = strip_unused_indices(T, T2, T3)

        want_T2 = torch.FloatTensor(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [4, 0, 5, 0],
            ]
        )
        want_T3 = torch.BoolTensor(
            [
                [True, True, False, False],
                [False, False, False, False],
                [True, False, True, False],
            ]
        )
        self.assertEqual(got_T1, want)
        self.assertEqual(got_T2, want_T2)
        self.assertEqual(got_T3, want_T3)

    def test_e2e(self):
        batch = self.samp.append_to_batch(self.batch, self.config)

        self.assertNestedDictEqual(WANT_APPENDED_BATCH, {k: v for k, v in batch.items()})

        for meas_to_fill, want_updates in WANT_UPDATED_DATA:
            batch = self.samp.update_last_event_data(
                batch=batch, config=self.config, measurements_to_fill=meas_to_fill
            )

            want_batch = copy.deepcopy(WANT_APPENDED_BATCH)
            for key, val in want_updates.items():
                want_L = val.shape[-1]
                old_L = want_batch[key].shape[-1]
                if want_L > old_L:
                    want_batch[key] = torch.nn.functional.pad(want_batch[key], (0, want_L - old_L), value=0)
                elif want_L < old_L:
                    val = torch.nn.functional.pad(val, (0, old_L - want_L), value=0)

                want_batch[key][:, -1, :] = val

            self.assertNestedDictEqual(
                want_batch, {k: v for k, v in batch.items()}, f"Batch failed for {meas_to_fill}"
            )


TEST_MEASUREMENTS_PER_GEN_MODE = {
    DataModality.SINGLE_LABEL_CLASSIFICATION: ["event_type"],
    DataModality.MULTI_LABEL_CLASSIFICATION: ["multi_label_col", "regression_col"],
    DataModality.MULTIVARIATE_REGRESSION: ["regression_col"],
}
TEST_VOCAB_SIZES_BY_MEASUREMENT = {
    "event_type": 2,
    "multi_label_col": 3,
    "regression_col": 4,
}
TEST_VOCAB_OFFSETS_BY_MEASUREMENT = {
    "event_type": 1,
    "multi_label_col": 3,
    "regression_col": 6,
}
TEST_EVENT_TYPES_IDXMAP = {
    "event_A": 0,
    "event_B": 1,
}
TEST_MEASUREMENTS_IDXMAP = {
    "event_type": 1,
    "multi_label_col": 2,
    "regression_col": 3,
}
TEST_MEASUREMENTS_PER_DEP_GRAPH_LEVEL = [[], ["event_type"], ["multi_label_col", "regression_col"]]

BASE_CONFIG_KWARGS = dict(
    measurements_per_generative_mode=TEST_MEASUREMENTS_PER_GEN_MODE,
    vocab_sizes_by_measurement=TEST_VOCAB_SIZES_BY_MEASUREMENT,
    vocab_offsets_by_measurement=TEST_VOCAB_OFFSETS_BY_MEASUREMENT,
    measurements_idxmap=TEST_MEASUREMENTS_IDXMAP,
    event_types_idxmap=TEST_EVENT_TYPES_IDXMAP,
    hidden_size=4,
    head_dim=None,
    num_attention_heads=2,  # Needs to divide hidden_size.
    measurements_per_dep_graph_level=TEST_MEASUREMENTS_PER_DEP_GRAPH_LEVEL,
)

BASE_BATCH_OUTPUT_LAYER_BASE_TEST = {
    "event_mask": torch.BoolTensor([[True, True, True]]),
    "time_delta": torch.FloatTensor([[2, 3, 1]]),
    "dynamic_values_mask": torch.BoolTensor(
        [
            [
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
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
            ],
        ]
    ),
    "dynamic_indices": torch.LongTensor(
        [
            [
                [1, 0, 0, 0, 0, 0],
                [2, 5, 0, 0, 0, 0],
                [2, 4, 5, 7, 8, 9],
            ],
        ]
    ),
    "dynamic_values": torch.Tensor(
        [
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1.1, -1.1, 0.0],
            ],
        ]
    ),
}


class TestGenerativeOutputLayerBase(MLTypeEqualityCheckableMixin, unittest.TestCase):
    """Tests the OutputLayer."""

    def test_constructs(self):
        """Tests that the Model Output Layer constructs given default configuration options."""
        config = StructuredTransformerConfig(**BASE_CONFIG_KWARGS)
        GenerativeOutputLayerBase(config)

    def test_get_classification_outputs(self):
        cases = [
            {
                "message": "Model should yield the correct outputs given inputs",
                "batch": {**BASE_BATCH_OUTPUT_LAYER_BASE_TEST},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            [-1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 0.0],
                            [-2.0, 2.0, 5.0, 8.0, 1.0, 4.0, 7.0, 0.0, 3.0, 6.0],
                        ],
                    ]
                ),
                "valid_measurements": {"event_type", "multi_label_col", "regression_col"},
                "want_dists": {
                    # All dists are of shape batch X seq X vocab size.
                    "event_type": (
                        torch.distributions.Bernoulli(logits=torch.FloatTensor([[0.0, -1.0, -2.0]])),
                        torch.distributions.Categorical(
                            logits=torch.FloatTensor(
                                [
                                    [
                                        [0.0, 1.0],
                                        [1.0, 3.0],
                                        [2.0, 5.0],
                                    ]
                                ]
                            )
                        ),
                    ),
                    "multi_label_col": (
                        None,
                        torch.distributions.Bernoulli(
                            logits=torch.FloatTensor(
                                [
                                    [
                                        [2.0, 3.0, 4.0],
                                        [5.0, 7.0, 9.0],
                                        [8.0, 1.0, 4.0],
                                    ]
                                ]
                            )
                        ),
                    ),
                    "regression_col": (
                        None,
                        torch.distributions.Bernoulli(
                            logits=torch.FloatTensor(
                                [
                                    [
                                        [5.0, 6.0, 7.0, 8.0],
                                        [2.0, 4.0, 6.0, 0.0],
                                        [7.0, 0.0, 3.0, 6.0],
                                    ]
                                ]
                            )
                        ),
                    ),
                },
                "want_labels": {
                    # The single-label classification task has shape batch X seq and has labels indices in
                    # it (in long format).
                    # Recall that event_type has no ['UNK'] currently prepending the vocab.
                    # TODO(mmd): Should likely have one.
                    "event_type": torch.LongTensor(
                        [
                            [0, 1, 1],
                        ]
                    ),
                    # The multi-label classification tasks have shape batch X seq X vocab size with
                    # binary indicators. They are also in float format, not long format. Also, note that
                    # labels are only present (non-zero) when the batch's data_type matches the target.
                    "multi_label_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0],
                                [0, 0, 1],
                                [0, 1, 1],
                            ]
                        ]
                    ),
                    "regression_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 1, 1, 1],
                            ]
                        ]
                    ),
                },
                # Losses should be given as follows.
                "want_losses": {
                    # event_type is fully observed, but has is_observed logits (0, -1, -2)
                    # We want to compute the NLL of this setting, which should then be averaged across events.
                    # So we want:
                    # 1/3 * (
                    #     -math.log(1/(1 + math.exp(0)))
                    #     -math.log(1/(1 + math.exp(1)))
                    #     -math.log(1/(1 + math.exp(2)))
                    # ) = 1.3777789597070467
                    # event_type has 3 pairs of (logit, label) across each event:
                    #   ([0.0, 1.0], 0), ([1.0, 3.0], 1), ([2.0, 5.0], 1).
                    # We want to compute the NLL of this setting, which should then be averaged across
                    # events. So we want:
                    #  1/3 * (
                    #    -math.log(math.exp(0)/(math.exp(0) + math.exp(1))) +
                    #    -math.log(math.exp(3)/(math.exp(1) + math.exp(3))) +
                    #    -math.log(math.exp(5)/(math.exp(2) + math.exp(5)))
                    #  )
                    "event_type": torch.tensor(1.8740379764186925),
                    # multi_label_col is has no positive labels for the first event (as it is actually not
                    # reported there), then has logits and labels for the last two events. Our code currently
                    # tasks the model with predicting on all events, including the first, just with all
                    # negative labels, as in theory the multi-label events that aren't observed there are
                    # valid instances of the labels not being present.
                    #
                    # (logits, labels):
                    #  ([2, 3, 4], [0, 0, 0]), ([5, 7, 9], [0, 0, 1]), ([8, 1, 4], [0, 1, 1])
                    # We want to compute the NLL of this setting, which should then be averaged first across
                    # all labels for the multi-label problem, then across only those events that are unmasked.
                    # 1/3 * (
                    #   1/3 * (
                    #     -math.log(1 - 1/(1 + math.exp(-2)))
                    #     -math.log(1 - 1/(1 + math.exp(-3)))
                    #     -math.log(1 - 1/(1 + math.exp(-4)))
                    #   ) + 1/3 * (
                    #     -math.log(1 - 1/(1 + math.exp(-5)))
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1/(1 + math.exp(-9)))
                    #   ) + 1/3 * (
                    #     -math.log(1 - 1/(1 + math.exp(-8)))
                    #     -math.log(1/(1 + math.exp(-1)))
                    #     -math.log(1/(1 + math.exp(-4)))
                    #   )
                    # ) = 3.2814624309539795
                    # If we instead only scored this on events with labels present:
                    #  ([5, 7, 9], [0, 0, 1]), ([8, 1, 4], [0, 1, 1])
                    # 1/2 * (
                    #   1/3 * (
                    #     -math.log(1 - 1/(1 + math.exp(-5)))
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1/(1 + math.exp(-9)))
                    #   ) + 1/3 * (
                    #     -math.log(1 - 1/(1 + math.exp(-8)))
                    #     -math.log(1/(1 + math.exp(-1)))
                    #     -math.log(1/(1 + math.exp(-4)))
                    #   )
                    # ) = 3.389916102091471
                    "multi_label_col": torch.tensor(3.2814624309539795),
                    # regression_col has no labels for the first two events, then has logits and labels for
                    # the last event as follows:
                    # ([5, 6, 7, 8], [0, 0, 0, 0]), ([2, 4, 6, 0], [0, 0, 0, 0]) ([7, 0, 3, 6], [0, 1, 1, 1])
                    # We want to compute the NLL of this setting, which should then be averaged first across
                    # all labels for the multi-label problem, then across only those events that are unmasked.
                    # 1/3 * (
                    #   1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-5)))
                    #     -math.log(1 - 1/(1 + math.exp(-6)))
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1 - 1/(1 + math.exp(-8)))
                    #   ) + 1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-2)))
                    #     -math.log(1 - 1/(1 + math.exp(-4)))
                    #     -math.log(1 - 1/(1 + math.exp(-6)))
                    #     -math.log(1 - 1/(1 + math.exp(-0)))
                    #   ) + 1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1/(1 + math.exp(-0)))
                    #     -math.log(1/(1 + math.exp(-3)))
                    #     -math.log(1/(1 + math.exp(-6)))
                    #   )
                    # ) = 3.883021699569808
                    # If we only wanted to do this on events with an event type for which regression_col is
                    # ever reported (event type 2, the last two events), we would have:
                    # 1/2 * (
                    #   1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-2)))
                    #     -math.log(1 - 1/(1 + math.exp(-4)))
                    #     -math.log(1 - 1/(1 + math.exp(-6)))
                    #     -math.log(1 - 1/(1 + math.exp(-0)))
                    #   ) + 1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1/(1 + math.exp(-0)))
                    #     -math.log(1/(1 + math.exp(-3)))
                    #     -math.log(1/(1 + math.exp(-6)))
                    #   )
                    # ) = 2.5732278110479676
                    # If we only did this on events with that measurement measured at all:
                    # ([7, 0, 3, 6], [0, 1, 1, 1])
                    #
                    # 1/4 * (
                    #   -math.log(1 - 1/(1 + math.exp(-7)))
                    #   -math.log(1/(1 + math.exp(-0)))
                    #   -math.log(1/(1+math.exp(-3)))
                    #   -math.log(1/(1 + math.exp(-6)))
                    # ) = 1.9362804209313096
                    "regression_col": torch.tensor(3.883021699569808),
                },
            },
            {
                "message": "Model should ignore masked events when computing losses.",
                "batch": {
                    **BASE_BATCH_OUTPUT_LAYER_BASE_TEST,
                    "event_mask": torch.BoolTensor([[True, False, True]]),
                },
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            [-1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 0.0],
                            [-2.0, 2.0, 5.0, 8.0, 1.0, 4.0, 7.0, 0.0, 3.0, 6.0],
                        ],
                    ]
                ),
                "valid_measurements": {"event_type", "multi_label_col", "regression_col"},
                "want_dists": {
                    # All dists are of shape batch X seq X vocab size.
                    "event_type": (
                        torch.distributions.Bernoulli(logits=torch.FloatTensor([[0.0, -1.0, -2.0]])),
                        torch.distributions.Categorical(
                            logits=torch.FloatTensor(
                                [
                                    [
                                        [0.0, 1.0],
                                        [1.0, 3.0],
                                        [2.0, 5.0],
                                    ]
                                ]
                            )
                        ),
                    ),
                    "multi_label_col": (
                        None,
                        torch.distributions.Bernoulli(
                            logits=torch.FloatTensor(
                                [
                                    [
                                        [2.0, 3.0, 4.0],
                                        [5.0, 7.0, 9.0],
                                        [8.0, 1.0, 4.0],
                                    ]
                                ]
                            )
                        ),
                    ),
                    "regression_col": (
                        None,
                        torch.distributions.Bernoulli(
                            logits=torch.FloatTensor(
                                [
                                    [
                                        [5.0, 6.0, 7.0, 8.0],
                                        [2.0, 4.0, 6.0, 0.0],
                                        [7.0, 0.0, 3.0, 6.0],
                                    ]
                                ]
                            )
                        ),
                    ),
                },
                "want_labels": {
                    # Labels ignore event_mask, and only respect data mask and dynamic_measurement_indices, so
                    # these are unchanged from the prior test.
                    "event_type": torch.LongTensor(
                        [
                            [0, 1, 1],
                        ]
                    ),
                    "multi_label_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0],
                                [0, 0, 1],
                                [0, 1, 1],
                            ]
                        ]
                    ),
                    "regression_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 1, 1, 1],
                            ]
                        ]
                    ),
                },
                # Losses should be modified to ignore the components of the first event.
                "want_losses": {
                    # event_type is fully observed, but has is_observed logits (0, -1, -2)
                    # We want to compute the NLL of this setting, which should then be averaged across events.
                    # So we want:
                    # 1/2 * (
                    #     -math.log(1/(1 + math.exp(0)))
                    #     MASKED
                    #     -math.log(1/(1 + math.exp(2)))
                    # ) = 1.410037595801459
                    # (logits, labels):
                    #   ([0.0, 1.0], 0), ([1.0, 3.0], 1) [MASKED], ([2.0, 5.0], 1).
                    # NLL =
                    # 1/2 * (
                    #   -math.log(math.exp(0)/(math.exp(0) + math.exp(1))) +
                    #   0*(-math.log(math.exp(3)/(math.exp(1) + math.exp(3)))) + # MASKED EVENT
                    #   -math.log(math.exp(5)/(math.exp(2) + math.exp(5)))
                    # ) = 0.6809245195459824
                    "event_type": torch.tensor(2.0909621153474416),
                    # (logits, labels):
                    #   ([2, 3, 4], [0, 0, 0]),
                    #   MASKED ([5, 7, 9], [0, 0, 1]),
                    #   ([8, 1, 4], [0, 1, 1])
                    # NLL =
                    # 1/2 * (
                    #   1/3 * (
                    #     -math.log(1 - 1/(1 + math.exp(-2)))
                    #     -math.log(1 - 1/(1 + math.exp(-3)))
                    #     -math.log(1 - 1/(1 + math.exp(-4)))
                    #   ) + 0 * ( # MASKED EVENT
                    #     -math.log(1 - 1/(1 + math.exp(-5)))
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1/(1 + math.exp(-9)))
                    #   ) + 1/3 * (
                    #     -math.log(1 - 1/(1 + math.exp(-8)))
                    #     -math.log(1/(1 + math.exp(-1)))
                    #     -math.log(1/(1 + math.exp(-4)))
                    #   )
                    # ) = 2.9209020520572953
                    "multi_label_col": torch.tensor(2.9209020520572953),
                    # (logits, labels):
                    #   ([5, 6, 7, 8], [0, 0, 0, 0]),
                    #   MASKED ([2, 4, 6, 0], [0, 0, 0, 0]),
                    #   ([7, 0, 3, 6], [0, 1, 1, 1])
                    # NLL =
                    # 1/2 * (
                    #   1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-5)))
                    #     -math.log(1 - 1/(1 + math.exp(-6)))
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1 - 1/(1 + math.exp(-8)))
                    #   ) + 0 * ( # MASKED EVENT
                    #     -math.log(1 - 1/(1 + math.exp(-2)))
                    #     -math.log(1 - 1/(1 + math.exp(-4)))
                    #     -math.log(1 - 1/(1 + math.exp(-6)))
                    #     -math.log(1 - 1/(1 + math.exp(-0)))
                    #   ) + 1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1/(1 + math.exp(-0)))
                    #     -math.log(1/(1 + math.exp(-3)))
                    #     -math.log(1/(1 + math.exp(-6)))
                    #   ) = 4.2194449487723995
                    "regression_col": torch.tensor(4.2194449487723995),
                },
            },
            {
                "message": "Model should only process selected data types.",
                "batch": {**BASE_BATCH_OUTPUT_LAYER_BASE_TEST},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 0.0],
                            [0.0, 2.0, 5.0, 8.0, 1.0, 4.0, 7.0, 0.0, 3.0, 6.0],
                        ],
                    ]
                ),
                "valid_measurements": {"multi_label_col"},
                "want_dists": {
                    "multi_label_col": (
                        None,
                        torch.distributions.Bernoulli(
                            logits=torch.FloatTensor(
                                [
                                    [
                                        [2.0, 3.0, 4.0],
                                        [5.0, 7.0, 9.0],
                                        [8.0, 1.0, 4.0],
                                    ]
                                ]
                            )
                        ),
                    ),
                },
                "want_labels": {
                    "multi_label_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0],
                                [0, 0, 1],
                                [0, 1, 1],
                            ]
                        ]
                    ),
                },
                "want_losses": {"multi_label_col": torch.tensor(3.2814624309539795)},
            },
            {
                "message": "Model should give a loss of 0 when no events have a single label task observed.",
                "batch": {
                    **BASE_BATCH_OUTPUT_LAYER_BASE_TEST,
                    "dynamic_measurement_indices": torch.LongTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                            ],
                        ]
                    ),
                    "dynamic_indices": torch.LongTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                            ],
                        ]
                    ),
                },
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 0.0],
                            [0.0, 2.0, 5.0, 8.0, 1.0, 4.0, 7.0, 0.0, 3.0, 6.0],
                        ],
                    ]
                ),
                "valid_measurements": {"event_type"},
                "want_dists": {
                    "event_type": (
                        torch.distributions.Bernoulli(logits=torch.FloatTensor([[0, 0, 0]])),
                        torch.distributions.Categorical(
                            logits=torch.FloatTensor(
                                [
                                    [
                                        [0.0, 1.0],
                                        [1.0, 3.0],
                                        [2.0, 5.0],
                                    ]
                                ]
                            )
                        ),
                    ),
                },
                "want_labels": {"event_type": torch.LongTensor([[0, 0, 0]])},
                "want_losses": {
                    # event_type has no valid observations, so should return a loss of 0.
                    "event_type": torch.tensor(0.0),
                },
            },
        ]

        for C in cases:
            with self.subTest(C["message"]):
                C["batch"] = PytorchBatch(**C["batch"])
                config = StructuredTransformerConfig(
                    **{
                        **BASE_CONFIG_KWARGS,
                        **C.get("config_kwargs", {}),
                        "hidden_size": 10,
                    }
                )

                # TODO(mmd): The config right now assumes the passed vocabulary sizes sum to the total vocab
                # size, but the model assumes there is one extra universally unused vocab element up front, so
                # we need to adjust that.
                config.vocab_size = 10
                n_measurements = 3
                assert len(config.measurements_idxmap) == n_measurements

                is_obs_weight = torch.eye(n_measurements)
                is_obs_weight = torch.nn.functional.pad(
                    is_obs_weight, (0, config.hidden_size - n_measurements, 0, 0)
                )

                layer = GenerativeOutputLayerBase(config)
                layer.IsObservedLayer.weight = torch.nn.Parameter(is_obs_weight)
                layer.IsObservedLayer.bias = torch.nn.Parameter(torch.zeros_like(layer.IsObservedLayer.bias))
                layer.ClassificationLayer.weight = torch.nn.Parameter(torch.eye(10))
                layer.ClassificationLayer.bias = torch.nn.Parameter(
                    torch.zeros_like(layer.ClassificationLayer.bias)
                )

                got_losses, got_dists, got_labels = layer.get_classification_outputs(
                    batch=C["batch"],
                    encoded=C["encoded"],
                    valid_measurements=C["valid_measurements"],
                )

                self.assertNestedDictEqual(C["want_labels"], got_labels, "Labels differ!")
                self.assertNestedDictEqual(C["want_dists"], got_dists, "Distributions differ!")
                self.assertNestedDictEqual(C["want_losses"], got_losses, "Losses differ!")

    def test_get_TTE_outputs(self):
        shared_config_kwargs = {
            **BASE_CONFIG_KWARGS,
            "hidden_size": 6,
        }
        generation_specific_config_kwargs = {
            "exponential": {"TTE_lognormal_generation_num_components": None},
            "log_normal_mixture": {"TTE_lognormal_generation_num_components": 2},
        }

        cases = [
            {
                "message": "Model should yield the correct outputs given inputs for an Exponential TTE.",
                "TTE_generation_layer_type": "exponential",
                "batch": {**BASE_BATCH_OUTPUT_LAYER_BASE_TEST},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
                            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                        ],
                    ]
                ),
                # `rate` is given by torch.nn.elu(layer.proj @ encoded):
                "want_dist": torch.distributions.Exponential(
                    rate=torch.FloatTensor([[1.0, 2.0, 3.0]]),
                ),
                "want_label": torch.FloatTensor([[2, 3]]),
                # The average log likelihood of the true observed time to event over valid pairs of unmasked
                # events (in this case the transitions between the first and second events and second and
                # third events) is given according to the PDF of the exponential distribution, shown below:
                # p(x) = rate * math.exp(-rate * x) for x greater than or equal to 0, and 0 otherwise.
                # Given our rates above, that means that we should expect a LL of
                # 1/2 * (
                #   math.log(1.0 * math.exp(-1.0 * 2.0))
                #   +math.log(2.0 * math.exp(-2.0 * 3.0))
                # ) = -3.6534264097200273
                "want_LL": torch.tensor(-3.6534264097200273),
            },
            {
                "message": "Model should yield the correct outputs given inputs for an LogNormalMixture.",
                "TTE_generation_layer_type": "log_normal_mixture",
                "batch": {**BASE_BATCH_OUTPUT_LAYER_BASE_TEST},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
                            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                        ],
                    ]
                ),
                # `the params are given by alternating positions.
                "want_dist": LogNormalMixtureDistribution(
                    locs=torch.Tensor(
                        [
                            [
                                [0, 3],
                                [1, 7],
                                [2, 8],
                            ]
                        ]
                    ),
                    log_scales=torch.Tensor(
                        [
                            [
                                [1, 4],
                                [3, 9],
                                [4, 10],
                            ]
                        ]
                    ),
                    log_weights=torch.Tensor(
                        [
                            [
                                [2, 5],
                                [5, 11],
                                [6, 12],
                            ]
                        ]
                    ),
                    mean_log_inter_time=0,
                    std_log_inter_time=1,
                ),
                "want_label": torch.FloatTensor([[2, 3]]),
                # The average log likelihood of the true observed time to event over valid pairs of unmasked
                # events (in this case the transitions between the first and second events and second and
                # third events) is given according to the weighted sum of the two component lognormal
                # distributions, given by their parameters above (recall that loc is the mu of the underlying
                # normal distribution and log_scale is the logarithm of the standard deviation of the
                # underlying normal distribution, and the two columns in the parameters above correspond to
                # the parameters for each component, with log_weights being the logits for the component
                # distributions):
                # See here for pdf: https://en.wikipedia.org/wiki/Log-normal_distribution
                # It's formula is
                # pdf(x) = (1/(x*math.exp(scale)*math.sqrt(2*math.pi))) *
                #          math.exp(-(math.log(x) - loc)**2/(2*(math.exp(scale)**2)))
                # LL = 1/2 * (
                #   math.log(
                #     math.exp(2)/(math.exp(2) + math.exp(5)) * (
                #       1/(2*math.exp(1)*math.sqrt(2*math.pi))*math.exp(
                #           -((math.log(2) - 0)**2)/(2*math.exp(1)**2)
                #          )
                #     ) + math.exp(5) / (math.exp(2) + math.exp(5)) * (
                #       1/(2*math.exp(4)*math.sqrt(2*math.pi))*math.exp(
                #            -((math.log(2) - 3)**2)/(2*math.exp(4)**2))
                #     )
                #   ) + math.log(
                #     math.exp(5)/(math.exp(11) + math.exp(5)) * (
                #       1/(3*math.exp(3)*math.sqrt(2*math.pi))*math.exp(
                #            -((math.log(3) - 1)**2)/(2*math.exp(3)**2))
                #     ) + math.exp(11) / (math.exp(11) + math.exp(5)) * (
                #       1/(3*math.exp(9)*math.sqrt(2*math.pi))*math.exp(
                #            -((math.log(3) - 7)**2)/(2*math.exp(9)**2))
                #     )
                #  )
                # ) = -7.6554941334115565
                "want_LL": torch.tensor(-7.6554941334115565),
            },
            {
                "message": "Model should respect event masking.",
                "TTE_generation_layer_type": "exponential",
                "batch": {
                    **BASE_BATCH_OUTPUT_LAYER_BASE_TEST,
                    # TODO(mmd): Were there only one valid event, the model would return a NaN Loss here, as
                    # opposed to just zeroing out that component for that patient. Is that desired?
                    "event_mask": torch.BoolTensor([[True, True, False]]),
                },
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
                            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                        ],
                    ]
                ),
                # `rate` is given by torch.nn.elu(layer.proj @ encoded):
                "want_dist": torch.distributions.Exponential(
                    rate=torch.FloatTensor([[1.0, 2.0, 3.0]]),
                ),
                # The labels are padded with 1s in locations where the event_mask is False. This is necessary
                # as multiple batch elements may have different #s of valid events.
                "want_label": torch.FloatTensor([[2, 1]]),
                # In this case, only the transition between the first and second event is valid, so the LL
                # should be:
                # math.log(1.0 * math.exp(-1.0 * 2.0)) = -2.
                "want_LL": torch.tensor(-2.0),
            },
        ]

        for C in cases:
            with self.subTest(C["message"]):
                C["batch"] = PytorchBatch(**C["batch"])
                config = StructuredTransformerConfig(
                    **shared_config_kwargs,
                    **generation_specific_config_kwargs[C["TTE_generation_layer_type"]],
                    TTE_generation_layer_type=C["TTE_generation_layer_type"],
                )

                # TODO(mmd): The config right now assumes the passed vocabulary sizes sum to the total vocab
                # size, but the model assumes there is one extra universally unused vocab element up front, so
                # we need to adjust that.
                config.vocab_size = 10

                layer = GenerativeOutputLayerBase(config)
                layer.TTE_layer.proj.bias = torch.nn.Parameter(torch.zeros_like(layer.TTE_layer.proj.bias))

                if C["TTE_generation_layer_type"] == "exponential":
                    layer.TTE_layer.proj.weight = torch.nn.Parameter(torch.Tensor([[1, 0, 0, 0, 0, 0]]))
                elif C["TTE_generation_layer_type"] == "log_normal_mixture":
                    layer.TTE_layer.proj.weight = torch.nn.Parameter(torch.eye(6))
                else:
                    raise ValueError(
                        f"TTE_generation_layer_type of {C['TTE_generation_layer_type']} unrecognized."
                    )

                got_LL, got_dist, got_label = layer.get_TTE_outputs(batch=C["batch"], encoded=C["encoded"])

                self.assertEqual(C["want_label"], got_label)
                self.assertDistributionsEqual(C["want_dist"], got_dist)
                self.assertEqual(C["want_LL"], got_LL, "Log likelihoods differ")

    def test_get_regression_outputs(self):
        cases = [
            {
                "message": "Model should yield the correct outputs given inputs.",
                "batch": {
                    **BASE_BATCH_OUTPUT_LAYER_BASE_TEST,
                    # Replicated here for clarity
                    "dynamic_measurement_indices": torch.LongTensor(
                        [
                            [
                                [1, 0, 0, 0, 0, 0],
                                [1, 2, 0, 0, 0, 0],
                                [1, 2, 2, 3, 3, 3],
                            ]
                        ]
                    ),
                    "dynamic_indices": torch.LongTensor(
                        [
                            [
                                [1, 0, 0, 0, 0, 0],
                                [2, 5, 0, 0, 0, 0],
                                [2, 4, 5, 7, 8, 9],
                            ]
                        ]
                    ),
                    "dynamic_values_mask": torch.BoolTensor(
                        [
                            [
                                [False, False, False, False, False, False],
                                [False, False, False, False, False, False],
                                [False, False, False, True, True, True],
                            ]
                        ]
                    ),
                    "dynamic_values": torch.Tensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1.1, -1.1, 0.0],
                            ]
                        ]
                    ),
                },
                "valid_measurements": {"regression_col"},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
                            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                        ],
                    ]
                ),
                # `rate` is given by torch.nn.elu(layer.proj @ encoded):
                "want_dists": {
                    # The parameters are a little weird here, because of how the layer works and because we
                    # use 0 as an index to indicate "not present" for masked data elements. This, plus the
                    # gather operation, means that the output parameters will have the parameters for the
                    # first regression target (index zero) at all masked positions, which causes the odd
                    # structure here. The only parameters that really matter are in the unmasked data
                    # positions, which are the last three of the last batch element.
                    # Further, recall that scale is elu(proj(encoded)) + 1, so there will be a plus one
                    # modifier here too.
                    "regression_col": (
                        None,
                        torch.distributions.Normal(
                            loc=torch.FloatTensor(
                                [
                                    [
                                        [0, 0, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1, 1],
                                        [2, 2, 2, 6, 10, 14],
                                    ]
                                ]
                            ),
                            scale=torch.FloatTensor(
                                [
                                    [
                                        [2, 2, 2, 2, 2, 2],
                                        [4, 4, 4, 4, 4, 4],
                                        [5, 5, 5, 9, 13, 17],
                                    ]
                                ]
                            ),
                        ),
                    ),
                },
                "want_labels": {
                    "regression_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1.1, -1.1, 0.0],
                            ]
                        ]
                    ),
                },
                "want_losses": {
                    # The average NLL over present events per patient, then over patients, is given as
                    # follows:
                    # 1/1 * (
                    #   1/3 * (
                    #     -math.log(1/(9*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((1.1 - 6)/9)**2))
                    #     -math.log(1/(13*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((-1.1 - 10)/13)**2))
                    #     -math.log(1/(17*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((0 - 14)/17)**2))
                    #   )
                    # ) = 3.734679909416965
                    "regression_col": torch.tensor(3.734679909416965),
                },
                "want_indices": {
                    "regression_col": torch.LongTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 2, 3],
                            ]
                        ]
                    ),
                },
            },
            {
                "message": "Model should appropriately average losses over events and data elements (1).",
                "batch": {
                    **BASE_BATCH_OUTPUT_LAYER_BASE_TEST,
                    "event_mask": torch.BoolTensor([[True, True, False]]),
                    "dynamic_measurement_indices": torch.LongTensor(
                        [
                            [
                                [1, 3, 3, 0, 0, 0],
                                [1, 3, 3, 3, 3, 0],
                                [1, 3, 3, 3, 3, 3],
                            ]
                        ]
                    ),
                    "dynamic_indices": torch.LongTensor(
                        [
                            [
                                [1, 7, 9, 0, 0, 0],
                                [2, 6, 7, 8, 7, 0],
                                [2, 7, 7, 7, 8, 9],
                            ]
                        ]
                    ),
                    "dynamic_values_mask": torch.BoolTensor(
                        [
                            [
                                [False, True, True, False, False, False],
                                [False, True, True, True, True, False],
                                [False, True, True, True, True, True],
                            ]
                        ]
                    ),
                    "dynamic_values": torch.Tensor(
                        [
                            [
                                [float("nan"), 2.5, 3.8, float("nan"), float("nan"), float("nan")],
                                [float("nan"), -1.2, 2.0, 4.5, -4.0, float("nan")],
                                [float("nan"), -1.2, 2.0, 4.5, -4.0, -5.0],
                            ]
                        ]
                    ),
                },
                "valid_measurements": {"regression_col"},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
                            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                        ],
                    ]
                ),
                # `rate` is given by torch.nn.elu(layer.proj @ encoded):
                "want_dists": {
                    "regression_col": (
                        None,
                        torch.distributions.Normal(
                            loc=torch.FloatTensor(
                                [
                                    [
                                        [0, 2, 6, 0, 0, 0],
                                        [1, 1, 5, 9, 5, 1],
                                        [2, 6, 6, 6, 10, 14],
                                    ]
                                ]
                            ),
                            scale=torch.FloatTensor(
                                [
                                    [
                                        [2, 4, 8, 2, 2, 2],
                                        [4, 4, 8, 12, 8, 4],
                                        [5, 9, 9, 9, 13, 17],
                                    ]
                                ]
                            ),
                        ),
                    ),
                },
                "want_labels": {
                    "regression_col": torch.FloatTensor(
                        [
                            [
                                [0, 2.5, 3.8, 0, 0, 0],
                                [0, -1.2, 2.0, 4.5, -4.0, 0],
                                [0, -1.2, 2.0, 4.5, -4.0, -5.0],
                            ]
                        ]
                    ),
                },
                "want_losses": {
                    # The average NLL over present events per patient, then over patients, is given as
                    # follows:
                    # 1/2 * (
                    #   1/2 * (
                    #     -math.log(1/(4*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((2.5 - 2)/4)**2))
                    #     -math.log(1/(8*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((3.8 - 6)/8)**2))
                    #   ) + 1/4 * (
                    #     -math.log(1/(4*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((-1.2 - 1)/4)**2))
                    #     -math.log(1/(8*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((2 - 5)/8)**2))
                    #     -math.log(1/(12*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((4.5 - 9)/12)**2))
                    #     -math.log(1/(8*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((-4 - 5)/8)**2))
                    #   )
                    # ) = 2.91612520818805
                    "regression_col": torch.tensor(2.91612520818805),
                },
                "want_indices": {
                    "regression_col": torch.LongTensor(
                        [
                            [
                                [0, 1, 3, 0, 0, 0],
                                [0, 0, 1, 2, 1, 0],
                                [0, 1, 1, 1, 2, 3],
                            ]
                        ]
                    ),
                },
            },
            {
                "message": "Model should appropriately average losses over events and data elements (2).",
                "batch": {
                    **BASE_BATCH_OUTPUT_LAYER_BASE_TEST,
                    "event_mask": torch.BoolTensor([[True, True, True]]),
                    "dynamic_measurement_indices": torch.LongTensor(
                        [
                            [
                                [1, 3, 3, 0, 0, 0],
                                [1, 3, 3, 3, 3, 0],
                                [1, 2, 2, 2, 2, 2],
                            ]
                        ]
                    ),
                    "dynamic_indices": torch.LongTensor(
                        [
                            [
                                [1, 7, 9, 0, 0, 0],
                                [2, 6, 7, 8, 7, 0],
                                [2, 4, 5, 4, 5, 4],
                            ]
                        ]
                    ),
                    "dynamic_values_mask": torch.BoolTensor(
                        [
                            [
                                [False, True, True, False, False, False],
                                [False, True, True, True, True, False],
                                [False, False, False, False, False, True],
                            ]
                        ]
                    ),
                    "dynamic_values": torch.Tensor(
                        [
                            [
                                [float("nan"), 2.5, 3.8, float("nan"), float("nan"), float("nan")],
                                [float("nan"), -1.2, 2.0, 4.5, -4.0, float("nan")],
                                [
                                    float("nan"),
                                    float("nan"),
                                    float("nan"),
                                    float("nan"),
                                    float("nan"),
                                    float("nan"),
                                ],
                            ]
                        ]
                    ),
                },
                "valid_measurements": {"regression_col"},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
                            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                        ],
                    ]
                ),
                # `rate` is given by torch.nn.elu(layer.proj @ encoded):
                "want_dists": {
                    "regression_col": (
                        None,
                        torch.distributions.Normal(
                            loc=torch.FloatTensor(
                                [
                                    [
                                        [0, 2, 6, 0, 0, 0],
                                        [1, 1, 5, 9, 5, 1],
                                        [2, 2, 2, 2, 2, 2],
                                    ]
                                ]
                            ),
                            scale=torch.FloatTensor(
                                [
                                    [
                                        [2, 4, 8, 2, 2, 2],
                                        [4, 4, 8, 12, 8, 4],
                                        [5, 5, 5, 5, 5, 5],
                                    ]
                                ]
                            ),
                        ),
                    ),
                },
                "want_labels": {
                    "regression_col": torch.FloatTensor(
                        [
                            [
                                [0, 2.5, 3.8, 0, 0, 0],
                                [0, -1.2, 2.0, 4.5, -4.0, 0],
                                [0, 0, 0, 0, 0, 0],
                            ]
                        ]
                    ),
                },
                "want_losses": {
                    # The average NLL over present events with a regression label per patient, then over
                    # patients, is given as
                    # follows:
                    # 1/2 * (
                    #   1/2 * (
                    #     -math.log(1/(4*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((2.5 - 2)/4)**2))
                    #     -math.log(1/(8*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((3.8 - 6)/8)**2))
                    #   ) + 1/4 * (
                    #     -math.log(1/(4*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((-1.2 - 1)/4)**2))
                    #     -math.log(1/(8*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((2 - 5)/8)**2))
                    #     -math.log(1/(12*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((4.5 - 9)/12)**2))
                    #     -math.log(1/(8*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((-4 - 5)/8)**2))
                    #   )
                    # ) = 2.91612520818805
                    "regression_col": torch.tensor(2.91612520818805),
                },
                "want_indices": {
                    "regression_col": torch.LongTensor(
                        [
                            [
                                [0, 1, 3, 0, 0, 0],
                                [0, 0, 1, 2, 1, 0],
                                [0, 0, 0, 0, 0, 0],
                            ]
                        ]
                    ),
                },
            },
            {
                "message": "Model should only return a loss where `dynamic_values_mask` is True.",
                "batch": {
                    **BASE_BATCH_OUTPUT_LAYER_BASE_TEST,
                    # Replicated here for clarity
                    "dynamic_measurement_indices": torch.LongTensor(
                        [
                            [
                                [1, 0, 0, 0, 0, 0],
                                [1, 2, 0, 0, 0, 0],
                                [1, 2, 2, 3, 3, 3],
                            ]
                        ]
                    ),
                    "dynamic_indices": torch.LongTensor(
                        [
                            [
                                [1, 0, 0, 0, 0, 0],
                                [2, 5, 0, 0, 0, 0],
                                [2, 4, 5, 7, 8, 9],
                            ]
                        ]
                    ),
                    "dynamic_values_mask": torch.BoolTensor(
                        [
                            [
                                [False, False, False, False, False, False],
                                [False, False, False, False, False, False],
                                [False, False, False, True, True, False],
                            ]
                        ]
                    ),
                    "dynamic_values": torch.Tensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1.1, -1.1, 0],
                            ]
                        ]
                    ),
                },
                "valid_measurements": {"regression_col"},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
                            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                        ],
                    ]
                ),
                # `rate` is given by torch.nn.elu(layer.proj @ encoded):
                "want_dists": {
                    "regression_col": (
                        None,
                        torch.distributions.Normal(
                            loc=torch.FloatTensor(
                                [
                                    [
                                        [0, 0, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1, 1],
                                        [2, 2, 2, 6, 10, 2],
                                    ]
                                ]
                            ),
                            scale=torch.FloatTensor(
                                [
                                    [
                                        [2, 2, 2, 2, 2, 2],
                                        [4, 4, 4, 4, 4, 4],
                                        [5, 5, 5, 9, 13, 5],
                                    ]
                                ]
                            ),
                        ),
                    ),
                },
                "want_labels": {
                    "regression_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1.1, -1.1, 0.0],
                            ]
                        ]
                    ),
                },
                "want_losses": {
                    # The average NLL over present events per patient, then over patients, *should be* given
                    # as follows:
                    # 1/1 * (
                    #   1/2 * (
                    #     -math.log(1/(9*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((1.1 - 6)/9)**2))
                    #     -math.log(1/(13*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((-1.1 - 10)/13)**2))
                    #   )
                    # ) = 3.556393752484623
                    "regression_col": torch.tensor(3.556393752484623),
                },
                "want_indices": {
                    "regression_col": torch.LongTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 2, 0],
                            ]
                        ]
                    ),
                },
            },
        ]

        for C in cases:
            with self.subTest(C["message"]):
                C["batch"] = PytorchBatch(**C["batch"])
                config = StructuredTransformerConfig(
                    **{
                        **BASE_CONFIG_KWARGS,
                        **C.get("config_kwargs", {}),
                        "hidden_size": 8,  # 2 * number of regression components (4)
                    }
                )

                # TODO(mmd): The config right now assumes the passed vocabulary sizes sum to the total vocab
                # size, but the model assumes there is one extra universally unused vocab element up front, so
                # we need to adjust that.
                config.vocab_size = 10

                layer = GenerativeOutputLayerBase(config)
                layer.regression_layers["regression_col"].proj.weight = torch.nn.Parameter(torch.eye(8))
                layer.regression_layers["regression_col"].proj.bias = torch.nn.Parameter(
                    torch.zeros_like(layer.regression_layers["regression_col"].proj.bias)
                )

                got_losses, got_dists, got_labels, got_indices = layer.get_regression_outputs(
                    batch=C["batch"],
                    encoded=C["encoded"],
                    valid_measurements=C["valid_measurements"],
                )

                self.assertNestedDictEqual(C["want_labels"], got_labels, "Labels differ")
                self.assertNestedDictEqual(C["want_dists"], got_dists, "Distributions differ")
                self.assertNestedDictEqual(C["want_indices"], got_indices, "Indices differ")
                self.assertNestedDictEqual(C["want_losses"], got_losses, "Losses differ")


if __name__ == "__main__":
    unittest.main()

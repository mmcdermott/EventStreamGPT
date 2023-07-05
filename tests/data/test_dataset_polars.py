import sys

sys.path.append("../..")

import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import polars as pl

from EventStream.data.config import DatasetConfig, MeasurementConfig
from EventStream.data.dataset_polars import Dataset
from EventStream.data.preprocessing import Preprocessor
from EventStream.data.time_dependent_functor import TimeDependentFunctor
from EventStream.data.types import (
    DataModality,
    NumericDataModalitySubtype,
    TemporalityType,
)
from EventStream.data.vocabulary import Vocabulary

from ..utils import ConfigComparisonsMixin


class NormalizerMock(Preprocessor):
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def params_schema(self) -> dict[str, pl.DataType]:
        return {"min": pl.Float64}

    def fit_from_polars(self, column: pl.Expr) -> pl.Expr:
        return pl.struct([column.min().alias("min")])

    @classmethod
    def predict_from_polars(cls, column: pl.Expr, model: pl.Expr) -> pl.Expr:
        return column - model.struct.field("min").round(0)


class OutlierDetectorMock(Preprocessor):
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def params_schema(self) -> dict[str, pl.DataType]:
        return {"mean": pl.Float64}

    def fit_from_polars(self, column: pl.Expr) -> pl.Expr:
        return pl.struct([column.mean().alias("mean")])

    @classmethod
    def predict_from_polars(cls, column: pl.Expr, model: pl.Expr) -> pl.Expr:
        return ((column - model.struct.field("mean")) > 10).cast(pl.Boolean)


class ESDMock(Dataset):
    PREPROCESSORS = {
        "outlier": OutlierDetectorMock,
        "normalizer": NormalizerMock,
    }


DOB_COL = "dob"


class AgeFunctorMock(TimeDependentFunctor):
    OUTPUT_MODALITY = DataModality.UNIVARIATE_REGRESSION

    def __init__(self):
        self.link_static_cols = [DOB_COL]

    def update_from_prior_timepoint(self, *args, **kwargs):
        return None

    def pl_expr(self):
        return (pl.col("timestamp") - pl.col(DOB_COL)).dt.nanoseconds() / 1e9 / 60 / 60 / 24 / 365.25


class TimeOfDayFunctorMock(TimeDependentFunctor):
    OUTPUT_MODALITY = DataModality.SINGLE_LABEL_CLASSIFICATION

    def update_from_prior_timepoint(self, *args, **kwargs):
        return None

    def pl_expr(self):
        return (
            pl.when(pl.col("timestamp").dt.hour() < 6)
            .then("EARLY_AM")
            .when(pl.col("timestamp").dt.hour() < 12)
            .then("AM")
            .when(pl.col("timestamp").dt.hour() < 21)
            .then("PM")
            .otherwise("LATE_PM")
        )


MeasurementConfig.FUNCTORS["AgeFunctorMock"] = AgeFunctorMock
MeasurementConfig.FUNCTORS["TimeOfDayFunctorMock"] = TimeOfDayFunctorMock

TEST_CONFIG = DatasetConfig(
    min_valid_column_observations=1 / 9,
    min_valid_vocab_element_observations=2,
    min_true_float_frequency=1 / 2,
    min_unique_numerical_observations=0.99,
    outlier_detector_config={"cls": "outlier"},
    normalizer_config={"cls": "normalizer"},
    agg_by_time_scale=None,
    measurement_configs={
        "pre_dropped": MeasurementConfig(temporality=TemporalityType.DYNAMIC, modality=DataModality.DROPPED),
        "not_present_dropped": MeasurementConfig(
            temporality=TemporalityType.DYNAMIC,
            modality=DataModality.MULTI_LABEL_CLASSIFICATION,
        ),
        "dynamic_preset_vocab": MeasurementConfig(
            temporality=TemporalityType.DYNAMIC,
            modality=DataModality.MULTI_LABEL_CLASSIFICATION,
            vocabulary=Vocabulary(["bar", "foo"], [1, 2]),
        ),
        "dynamic_dropped_insufficient_occurrences": MeasurementConfig(
            temporality=TemporalityType.DYNAMIC,
            modality=DataModality.MULTI_LABEL_CLASSIFICATION,
        ),
        "static": MeasurementConfig(
            temporality=TemporalityType.STATIC,
            modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
        ),
        "time_dependent_age_lt_90": MeasurementConfig(
            temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
            functor=AgeFunctorMock(),
            _measurement_metadata=pd.Series(
                [90.0, False],
                index=pd.Index(
                    ["drop_upper_bound", "drop_upper_bound_inclusive"],
                ),
                name="time_dependent_age_lt_90",
            ),
        ),
        "time_dependent_age_all": MeasurementConfig(
            temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
            functor=AgeFunctorMock(),
        ),
        "time_dependent_time_of_day": MeasurementConfig(
            temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
            functor=TimeOfDayFunctorMock(),
        ),
        "multivariate_regression_bounded_outliers": MeasurementConfig(
            temporality=TemporalityType.DYNAMIC,
            modality=DataModality.MULTIVARIATE_REGRESSION,
            values_column="mrbo_vals",
            _measurement_metadata=pd.DataFrame(
                {
                    "drop_lower_bound": [-1.1, -10.1, None],
                    "drop_lower_bound_inclusive": [True, False, None],
                    "drop_upper_bound": [1.1, None, 10.1],
                    "drop_upper_bound_inclusive": [False, None, True],
                    "censor_lower_bound": [None, -5.1, -10.1],
                    "censor_upper_bound": [0.6, 10.1, None],
                },
                index=pd.Index(["mrbo1", "mrbo2", "mrbo3"], name="multivariate_regression_bounded_outliers"),
            ),
        ),
        "multivariate_regression_preset_value_type": MeasurementConfig(
            temporality=TemporalityType.DYNAMIC,
            modality=DataModality.MULTIVARIATE_REGRESSION,
            values_column="pvt_vals",
            _measurement_metadata=pd.DataFrame(
                {
                    "value_type": [
                        NumericDataModalitySubtype.CATEGORICAL_INTEGER,
                        NumericDataModalitySubtype.CATEGORICAL_FLOAT,
                        NumericDataModalitySubtype.INTEGER,
                        NumericDataModalitySubtype.FLOAT,
                        NumericDataModalitySubtype.DROPPED,
                    ],
                },
                index=pd.Index(
                    ["pvt_cat_int", "pvt_cat_flt", "pvt_int", "pvt_flt", "pvt_drp"],
                    name="multivariate_regression_preset_value_type",
                ),
            ),
        ),
        "multivariate_regression_no_preset": MeasurementConfig(
            temporality=TemporalityType.DYNAMIC,
            modality=DataModality.MULTIVARIATE_REGRESSION,
            values_column="mrnp_vals",
        ),
    },
)

TEST_SPLIT = {"train": {1, 2, 4, 5}, "held_out": {3}}

in_event_times = {
    1: datetime(2010, 1, 1, 2),  # MVR, Subj 1, Agg 1, EARLY_AM
    2: datetime(2010, 1, 1, 2),  # MVR, Subj 1, Agg 2
    3: datetime(2010, 1, 2, 13),  # MVR, Subj 2, Agg 1, PM
    4: datetime(2010, 1, 2, 13),  # MVR, Subj 2, Agg 2,
    5: datetime(2010, 1, 3, 3),  # DDIC, Subj 1, EARLY_AM
    6: datetime(2010, 1, 4, 4),  # DDIC, Subj 2, EARLY_AM
    7: datetime(2010, 1, 5, 14),  # DPV, Subj 1, PM
    8: datetime(2010, 1, 8, 23),  # DPV, Subj 1, LATE_PM
    9: datetime(2010, 1, 9, 22, 30),  # DPV, Subj 1, LATE_PM
    10: datetime(2010, 1, 10, 3),  # DPV, Subj 2, EARLY_AM,
    11: datetime(2010, 1, 11, 15),  # DPV, Subj 2, PM
    12: datetime(2010, 1, 1, 23),  # DPV, Subj 3, LATE_PM
    13: datetime(2010, 1, 2, 23),  # DPV, Subj 3, LATE_PM
    14: datetime(2010, 1, 3, 22),  # DPV, Subj 3, LATE_PM
    15: datetime(2010, 1, 4, 11),  # DPV, Subj 3, AM
}

in_event_subjects = {
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 1,
    6: 2,
    7: 1,
    8: 1,
    9: 1,
    10: 2,
    11: 2,
    12: 3,
    13: 3,
    14: 3,
    15: 3,
}

want_event_agg_mapping = {
    1: (1, 2),
    2: (5,),
    3: (7,),
    4: (8,),
    5: (9,),
    6: (3, 4),
    7: (6,),
    8: (10,),
    9: (11,),
    10: (12,),
    11: (13,),
    12: (14,),
    13: (15,),
}

want_event_times = {want_id: in_event_times[in_ids[0]] for want_id, in_ids in want_event_agg_mapping.items()}
want_event_TODs = {
    k: "EARLY_AM" if v.hour < 6 else "UNK" if v.hour < 12 else "PM" if v.hour < 21 else "LATE_PM"
    for k, v in want_event_times.items()
}

subject_dobs = {
    1: datetime(2000, 1, 1),
    2: datetime(1900, 1, 1),
    3: datetime(1980, 1, 1),
    4: datetime(1990, 1, 1),
    5: datetime(2010, 1, 1),
}

want_event_ts_ages = {}
for want_id, in_ids in want_event_agg_mapping.items():
    want_event_ts_ages[want_id] = (
        in_event_times[in_ids[0]] - subject_dobs[in_event_subjects[in_ids[0]]]
    ) / timedelta(days=365.25)

train_ages_lt_90 = []
train_all_ages = []

for i, age in want_event_ts_ages.items():
    in_ids = want_event_agg_mapping[i]
    subj = in_event_subjects[in_ids[0]]
    if subj in TEST_SPLIT["train"]:
        if age < 90:
            train_ages_lt_90.append(age)
        train_all_ages.append(age)

train_ages_lt_90 = np.array(train_ages_lt_90)
train_all_ages = np.array(train_all_ages)

outlier_mean_lt_90 = train_ages_lt_90.mean()
outlier_mean_all = train_all_ages.mean()

inliers_lt_90 = train_ages_lt_90[train_ages_lt_90 - outlier_mean_lt_90 < 10]
inliers_all = train_all_ages[train_all_ages - outlier_mean_all < 10]

normalizer_min_lt_90 = inliers_lt_90.min()
normalizer_min_all = inliers_all.min()

want_events_ts_ages_lt_90_is_inlier = {
    k: None if (v > 90) else bool(v - outlier_mean_lt_90 < 10) for k, v in want_event_ts_ages.items()
}
want_events_ts_ages_lt_90 = {
    k: (v - normalizer_min_lt_90.round()) if (v < 90) and want_events_ts_ages_lt_90_is_inlier[k] else np.NaN
    for k, v in want_event_ts_ages.items()
}
want_events_ts_ages_all_is_inlier = {
    k: bool(v - outlier_mean_all < 10) for k, v in want_event_ts_ages.items()
}
want_events_ts_ages_all = {
    k: (v - normalizer_min_all.round()) if want_events_ts_ages_all_is_inlier[k] else np.NaN
    for k, v in want_event_ts_ages.items()
}

IN_SUBJECTS_DF = pl.DataFrame(
    data={
        "subject_id": [1, 2, 3, 4, 5],
        "static": ["foo", "foo", "bar", "bar", "bar"],
        DOB_COL: [subject_dobs[i] for i in range(1, 6)],
    },
    schema={
        "subject_id": pl.Int64,
        "static": pl.Utf8,
        DOB_COL: pl.Datetime,
    },
)

IN_EVENTS_DF = pl.DataFrame(
    data={
        "event_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "event_type": [
            "MVR",
            "MVR",
            "MVR",
            "MVR",
            "DDIC",
            "DDIC",
            "DPV",
            "DPV",
            "DPV",
            "DPV",
            "DPV",
            "DPV",
            "DPV",
            "DPV",
            "DPV",
        ],
        "subject_id": [1, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 3, 3, 3, 3],
        "timestamp": [in_event_times[i] for i in range(1, 16)],
    },
    schema={
        "event_id": pl.Float64,
        "event_type": pl.Utf8,
        "subject_id": pl.Int8,
        "timestamp": pl.Datetime,
    },
)
np.random.seed(1)
input_order = np.random.permutation(15)

IN_EVENTS_DF = IN_EVENTS_DF.sort(pl.lit(input_order))

IN_MEASUREMENTS_DF = pl.DataFrame(
    data={
        "event_id": [
            *([1] * 4 + [2] * 4 + [3] * 4 + [4] * 5),
            *([5] * 2 + [6] * 2),
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        ],
        # Has pre-set vocab ['foo', 'bar'], occurs on 'DPV' events.
        "dynamic_preset_vocab": [
            *([None] * 17),
            *([None] * 4),
            "foo",
            "foo",
            "bar",
            "bar",
            "bar",
            "baz",
            "baz",
            "foo",
            "foo",
        ],
        # Is dropped due to insufficient occurrences, occurs on 'DDIC' events.
        "dynamic_dropped_insufficient_occurrences": [
            *([None] * 17),
            "here",
            None,
            None,
            None,
            *([None] * 9),
        ],
        # Occurs on events MVR, values 'mrbo_vals'.
        # Has pre-set keys with outlier/censor bounds as follows:
        #          Outlier,  Censor
        #   mrbo1: [-1.1, 1.1),  (X, 0.6]
        #   mrbo2: (-10.1, X), [-5.1, 10.1]
        #   mrbo3: (X, 10.1],  [-10.1, X)
        "multivariate_regression_bounded_outliers": [
            "mrbo1",
            "mrbo3",
            "mrbo2",
            "mrbo1",
            "mrbo2",
            "mrbo1",
            "mrbo3",
            "mrbo2",
            "mrbo3",
            "mrbo2",
            "mrbo1",
            "mrbo3",
            None,
            None,
            None,
            None,
            None,
            *([None] * 4),
            *([None] * 9),
        ],
        "mrbo_vals": [
            -1.2,
            0.1,
            0.1,
            0.7,
            -10.1,
            -1.1,
            10.1,
            10.2,
            -11.1,
            -4.9,
            0.1,
            11.1,
            None,
            None,
            None,
            None,
            None,
            *([None] * 4),
            *([None] * 9),
        ],
        # Occurs on events MVR, values 'pvt_vals'.
        # Has pre-set keys with value types as follows:
        #          Value Type
        #   pvt_cat_int: NumericDataModalitySubtype.CATEGORICAL_INTEGER,
        #   pvt_cat_flt: NumericDataModalitySubtype.CATEGORICAL_FLOAT,
        #   pvt_int:     NumericDataModalitySubtype.INTEGER,
        #   pvt_flt:     NumericDataModalitySubtype.FLOAT,
        #   pvt_drp:     NumericDataModalitySubtype.DROPPED,
        # Also has extra key not in the pre-set of 'pvt_added'
        # Event IDs
        # *([1]*4 + [2]*4 + [3]*4 + [4]*5),
        # ... after agg
        # *([1]*8 + [2]*9),
        # *([3]*2 + [4]*2),
        "multivariate_regression_preset_value_type": [
            # Event ID 1
            "pvt_int",
            "pvt_cat_int",
            "pvt_added",
            "pvt_flt",
            "pvt_cat_int",
            "pvt_drp",
            "pvt_cat_flt",
            "pvt_cat_int",
            # Event ID 2
            "pvt_cat_flt",
            "pvt_int",
            "pvt_cat_int",
            "pvt_cat_flt",
            "pvt_drp",
            "pvt_cat_flt",
            "pvt_flt",
            "pvt_added",
            None,
            *([None] * 4),
            *([None] * 9),
        ],
        "pvt_vals": [
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            None,
            *([None] * 4),
            *([None] * 9),
        ],
        # Occurs on events MVR, values 'mrnp_vals'.
        # Keys include:
        #   'mrnp_flt', 'mrnp_int', 'mrnp_cat_int__EQ_1', 'mrnp_cat_int__EQ_2', 'mrnp_cat_int__EQ_3',
        #   'mrnp_dropped' and 'mrnp_key_dropped'
        # These should result in types float, int, categorical int, dropped, and 'mrnp_key_dropped' should be
        # dropped wholesale.
        # Event IDs
        # *([1]*4 + [2]*4 + [3]*4 + [4]*5),
        # ... after agg
        # *([1]*8 + [2]*9),
        # *([3]*2 + [4]*2),
        "multivariate_regression_no_preset": [
            # Event ID 1
            "mrnp_dropped",
            "mrnp_flt",
            "mrnp_flt",
            "mrnp_key_dropped",
            "mrnp_int",
            "mrnp_int",
            "mrnp_cat_int",
            "mrnp_cat_int",
            # Event ID 2
            "mrnp_cat_int",
            "mrnp_cat_int",
            "mrnp_cat_int",
            "mrnp_cat_int",
            "mrnp_cat_int",
            "mrnp_cat_int",
            "mrnp_flt",
            "mrnp_dropped",
            "mrnp_int",
            *([None] * 4),
            *([None] * 9),
        ],
        "mrnp_vals": [
            1.0,
            3.0,
            80.1,
            0.2,
            80.0,
            3.0,
            1.0,
            1.2,
            2.0,
            2.0,
            3.0,
            2.9,
            4.0,
            5.0,
            1.2,
            1.0,
            1.2,
            *([None] * 4),
            *([None] * 9),
        ],
    },
    schema={
        "event_id": pl.Int16,
        "dynamic_preset_vocab": pl.Utf8,
        "dynamic_dropped_insufficient_occurrences": pl.Utf8,
        "multivariate_regression_bounded_outliers": pl.Utf8,
        "mrbo_vals": pl.Float64,
        "multivariate_regression_preset_value_type": pl.Categorical,
        "pvt_vals": pl.Float32,
        "multivariate_regression_no_preset": pl.Utf8,
        "mrnp_vals": pl.Float64,
    },
)

WANT_EVENT_TYPES = ["DPV", "MVR", "DDIC"]

WANT_MEASUREMENTS_IDXMAP = {
    "event_type": 1,
    "dynamic_preset_vocab": 2,
    "multivariate_regression_bounded_outliers": 3,
    "multivariate_regression_no_preset": 4,
    "multivariate_regression_preset_value_type": 5,
    "static": 6,
    "time_dependent_age_all": 7,
    "time_dependent_age_lt_90": 8,
    "time_dependent_time_of_day": 9,
}

WANT_UNIFIED_VOCABULARY_OFFSETS = {
    "event_type": 1,
    "dynamic_preset_vocab": 4,
    "multivariate_regression_bounded_outliers": 7,
    "multivariate_regression_no_preset": 11,
    "multivariate_regression_preset_value_type": 18,
    "static": 27,
    "time_dependent_age_all": 30,
    "time_dependent_age_lt_90": 31,
    "time_dependent_time_of_day": 32,
}

WANT_INFERRED_MEASUREMENT_CONFIGS = {
    "not_present_dropped": MeasurementConfig(
        name="not_present_dropped",
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.DROPPED,
    ),
    "dynamic_preset_vocab": MeasurementConfig(
        name="dynamic_preset_vocab",
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.MULTI_LABEL_CLASSIFICATION,
        vocabulary=Vocabulary(["UNK", "foo", "bar"], [0, 2 / 3, 1 / 3]),
        observation_frequency=5 / 9,
    ),
    "dynamic_dropped_insufficient_occurrences": MeasurementConfig(
        name="dynamic_dropped_insufficient_occurrences",
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.DROPPED,
        observation_frequency=1 / 9,
    ),
    "static": MeasurementConfig(
        name="static",
        temporality=TemporalityType.STATIC,
        modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
        observation_frequency=1,
        vocabulary=Vocabulary(["UNK", "bar", "foo"], [0, 0.5, 0.5]),
    ),
    "time_dependent_age_lt_90": MeasurementConfig(
        name="time_dependent_age_lt_90",
        temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
        functor=AgeFunctorMock(),
        _measurement_metadata=pd.Series(
            [
                90.0,
                False,
                NumericDataModalitySubtype.FLOAT,
                {"mean": outlier_mean_lt_90},
                {"min": normalizer_min_lt_90},
            ],
            index=pd.Index(
                [
                    "drop_upper_bound",
                    "drop_upper_bound_inclusive",
                    "value_type",
                    "outlier_model",
                    "normalizer",
                ]
            ),
            name="time_dependent_age_lt_90",
        ),
        observation_frequency=1,
        vocabulary=None,
    ),
    "time_dependent_age_all": MeasurementConfig(
        name="time_dependent_age_all",
        temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
        functor=AgeFunctorMock(),
        observation_frequency=1,
        vocabulary=None,
        _measurement_metadata=pd.Series(
            [
                NumericDataModalitySubtype.FLOAT,
                {"mean": outlier_mean_all},
                {"min": normalizer_min_all},
            ],
            index=pd.Index(["value_type", "outlier_model", "normalizer"]),
            name="time_dependent_age_all",
        ),
    ),
    "time_dependent_time_of_day": MeasurementConfig(
        name="time_dependent_time_of_day",
        temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
        functor=TimeOfDayFunctorMock(),
        observation_frequency=1,
        vocabulary=Vocabulary(["UNK", "EARLY_AM", "PM", "LATE_PM"], [0, 4, 3, 2]),
    ),
    # Keys and Values:
    # 'mrbo1': -1.2, -1.1, 0.1, 0.7,
    # 'mrbo2': -10.1, -4.9, 0.1, 10.2,
    # 'mrbo3': -11.1, 0.1, 10.1, 11.1,
    # After dropping/censoring, becomes:
    # 'mrbo1': np.NaN, np.NaN, 0.1, 0.6,
    # 'mrbo2': -5.1, -4.9, 0.1, 10.1,
    # 'mrbo3': -10.1, 0.1, np.NaN, np.NaN,
    # Yields means / mins:
    # 'mrbo1': 0.35 / 0.1,
    # 'mrbo2': 0.05 / -5.1,
    # 'mrbo3': -5 / -10.1,
    "multivariate_regression_bounded_outliers": MeasurementConfig(
        name="multivariate_regression_bounded_outliers",
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.MULTIVARIATE_REGRESSION,
        values_column="mrbo_vals",
        _measurement_metadata=pd.DataFrame(
            {
                "drop_lower_bound": [-1.1, -10.1, None],
                "drop_lower_bound_inclusive": [True, False, None],
                "drop_upper_bound": [1.1, None, 10.1],
                "drop_upper_bound_inclusive": [False, None, True],
                "censor_lower_bound": [None, -5.1, -10.1],
                "censor_upper_bound": [0.6, 10.1, None],
                "value_type": [
                    NumericDataModalitySubtype.FLOAT,
                    NumericDataModalitySubtype.FLOAT,
                    NumericDataModalitySubtype.FLOAT,
                ],
                "outlier_model": [
                    {"mean": 0.35},
                    {"mean": 0.05},
                    {"mean": -5},
                ],
                "normalizer": [
                    {"min": 0.1},
                    {"min": -5.1},
                    {"min": -10.1},
                ],
            },
            index=pd.CategoricalIndex(
                ["mrbo1", "mrbo2", "mrbo3"], name="multivariate_regression_bounded_outliers"
            ),
        ),
        observation_frequency=2 / 9,
        vocabulary=Vocabulary(["UNK", "mrbo1", "mrbo2", "mrbo3"], [0, 1, 1, 1]),
    ),
    "multivariate_regression_preset_value_type": MeasurementConfig(
        name="multivariate_regression_preset_value_type",
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.MULTIVARIATE_REGRESSION,
        values_column="pvt_vals",
        _measurement_metadata=pd.DataFrame(
            {
                "value_type": [
                    NumericDataModalitySubtype.INTEGER,
                    NumericDataModalitySubtype.CATEGORICAL_FLOAT,
                    NumericDataModalitySubtype.CATEGORICAL_INTEGER,
                    NumericDataModalitySubtype.DROPPED,
                    NumericDataModalitySubtype.FLOAT,
                    NumericDataModalitySubtype.INTEGER,
                ],
                "outlier_model": [
                    {"mean": 1.5},
                    {"mean": None},
                    {"mean": None},
                    {"mean": None},
                    {"mean": 1.5},
                    {"mean": 1.5},
                ],
                "normalizer": [
                    {"min": 1},
                    {"min": None},
                    {"min": None},
                    {"min": None},
                    {"min": 1},
                    {"min": 1},
                ],
            },
            index=pd.CategoricalIndex(
                ["pvt_added", "pvt_cat_flt", "pvt_cat_int", "pvt_drp", "pvt_flt", "pvt_int"],
                name="multivariate_regression_preset_value_type",
            ),
        ),
        observation_frequency=2 / 9,
        vocabulary=Vocabulary(
            [
                "UNK",
                "pvt_added",
                "pvt_cat_flt__EQ_1.0",
                "pvt_cat_flt__EQ_2.0",
                "pvt_cat_int__EQ_1",
                "pvt_cat_int__EQ_2",
                "pvt_drp",
                "pvt_flt",
                "pvt_int",
            ],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
    ),
    "multivariate_regression_no_preset": MeasurementConfig(
        name="multivariate_regression_no_preset",
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.MULTIVARIATE_REGRESSION,
        values_column="mrnp_vals",
        observation_frequency=2 / 9,
        vocabulary=Vocabulary(
            [
                "UNK",
                "mrnp_flt",
                "mrnp_int",
                "mrnp_cat_int__EQ_3",
                "mrnp_cat_int__EQ_1",
                "mrnp_cat_int__EQ_2",
                "mrnp_dropped",
            ],
            [3, 3, 3, 2, 2, 2, 2],
        ),
        _measurement_metadata=pd.DataFrame(
            {
                "value_type": [
                    NumericDataModalitySubtype.FLOAT,
                    NumericDataModalitySubtype.INTEGER,
                    NumericDataModalitySubtype.CATEGORICAL_INTEGER,
                    NumericDataModalitySubtype.DROPPED,
                    NumericDataModalitySubtype.DROPPED,
                ],
                "outlier_model": [
                    {"mean": 84.3 / 3},
                    {"mean": 84 / 3},
                    {"mean": None},
                    {"mean": None},
                    {"mean": None},
                ],
                "normalizer": [
                    {"min": 1.2},
                    {"min": 1.0},
                    {"min": None},
                    {"min": None},
                    {"min": None},
                ],
            },
            index=pd.CategoricalIndex(
                ["mrnp_flt", "mrnp_int", "mrnp_cat_int", "mrnp_dropped", "mrnp_key_dropped"],
                name="multivariate_regression_no_preset",
            ),
        ),
    ),
}

WANT_UNIFIED_VOCABULARY_IDXMAP = {
    "event_type": {k: i + 1 for i, k in enumerate(WANT_EVENT_TYPES)},
    **{
        kk: {
            k: i + WANT_UNIFIED_VOCABULARY_OFFSETS[kk]
            for i, k in enumerate(WANT_INFERRED_MEASUREMENT_CONFIGS[kk].vocabulary.vocabulary)
        }
        for kk in (
            "dynamic_preset_vocab",
            "multivariate_regression_bounded_outliers",
            "multivariate_regression_no_preset",
            "multivariate_regression_preset_value_type",
            "static",
            "time_dependent_time_of_day",
        )
    },
    "time_dependent_age_all": {"time_dependent_age_all": 30},
    "time_dependent_age_lt_90": {"time_dependent_age_lt_90": 31},
}

WANT_SUBJECTS_DF = pl.DataFrame(
    data={
        "subject_id": [1, 2, 3, 4, 5],
        "static": ["foo", "foo", "bar", "bar", "bar"],
        DOB_COL: [subject_dobs[i] for i in range(1, 6)],
    },
    schema={
        "subject_id": pl.UInt8,
        "static": pl.Categorical,
        DOB_COL: pl.Datetime,
    },
)

WANT_EVENTS_DF = pl.DataFrame(
    data={
        "event_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "event_type": [
            "MVR",
            "DDIC",
            "DPV",
            "DPV",
            "DPV",
            "MVR",
            "DDIC",
            "DPV",
            "DPV",
            "DPV",
            "DPV",
            "DPV",
            "DPV",
        ],
        "subject_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        "timestamp": [want_event_times[i] for i in range(1, 14)],
        "time_dependent_age_lt_90": [want_events_ts_ages_lt_90[i] for i in range(1, 14)],
        "time_dependent_age_all": [want_events_ts_ages_all[i] for i in range(1, 14)],
        "time_dependent_age_lt_90_is_inlier": [want_events_ts_ages_lt_90_is_inlier[i] for i in range(1, 14)],
        "time_dependent_age_all_is_inlier": [want_events_ts_ages_all_is_inlier[i] for i in range(1, 14)],
        "time_dependent_time_of_day": [want_event_TODs[i] for i in range(1, 14)],
    },
    schema={
        "event_id": pl.UInt8,
        "event_type": pl.Categorical,
        "subject_id": pl.UInt8,
        "timestamp": pl.Datetime,
        "time_dependent_age_lt_90": pl.Float64,
        "time_dependent_age_all": pl.Float64,
        "time_dependent_age_lt_90_is_inlier": pl.Boolean,
        "time_dependent_age_all_is_inlier": pl.Boolean,
        "time_dependent_time_of_day": pl.Categorical,
    },
)

WANT_MEASUREMENTS_DF = pl.DataFrame(
    data={
        "measurement_id": list(range(30)),
        "event_id": [
            *([0] * 8 + [5] * 9),
            *([1] * 2 + [6] * 2),
            2,
            3,
            4,
            7,
            8,
            9,
            10,
            11,
            12,
        ],
        # Has pre-set vocab ['foo', 'bar'], occurs on 'DPV' events.
        "dynamic_preset_vocab": [
            *([None] * 17),
            *([None] * 4),
            "foo",
            "foo",
            "bar",
            "bar",
            "bar",
            "UNK",
            "UNK",
            "foo",
            "foo",
        ],
        # Is dropped due to insufficient occurrences, occurs on 'DDIC' events.
        "dynamic_dropped_insufficient_occurrences": [
            *([None] * 17),
            "here",
            None,
            None,
            None,
            *([None] * 9),
        ],
        # Occurs on events MVR, values 'mrbo_vals'.
        # Has pre-set keys with outlier/censor bounds as follows:
        #          Outlier,  Censor
        #   mrbo1: [-1.1, 1.1),  (X, 0.6]
        #   mrbo2: (-10.1, X), [-5.1, 10.1]
        #   mrbo3: (X, 10.1],  [-10.1, X)
        # Keys and Values:
        # 'mrbo1': -1.2, -1.1, 0.1, 0.7,
        # 'mrbo2': -10.1, -4.9, 0.1, 10.2,
        # 'mrbo3': -11.1, 0.1, 10.1, 11.1,
        # After dropping/censoring, becomes:
        # 'mrbo1': np.NaN, np.NaN, 0.1, 0.6,
        # 'mrbo2': -5.1, -4.9, 0.1, 10.1,
        # 'mrbo3': -10.1, 0.1, np.NaN, np.NaN,
        # Yields means / mins / mins.round(0):
        # 'mrbo1': 0.35 / 0.1 / 0,
        # 'mrbo2': 0.05 / -5.1 / -5,
        # 'mrbo3': -5 / -10.1 / -10,
        "multivariate_regression_bounded_outliers": [
            "mrbo1",
            "mrbo3",
            "mrbo2",
            "mrbo1",
            "mrbo2",
            "mrbo1",
            "mrbo3",
            "mrbo2",
            "mrbo3",
            "mrbo2",
            "mrbo1",
            "mrbo3",
            None,
            None,
            None,
            None,
            None,
            *([None] * 4),
            *([None] * 9),
        ],
        "mrbo_vals": [
            np.NaN,
            10.1,
            5.1,
            0.6,
            -0.1,
            np.NaN,
            np.NaN,
            np.NaN,
            -0.1,
            0.1,
            0.1,
            np.NaN,
            None,
            None,
            None,
            None,
            None,
            *([None] * 4),
            *([None] * 9),
        ],
        "multivariate_regression_bounded_outliers_is_inlier": [
            None,
            True,
            True,
            True,
            True,
            None,
            None,
            False,
            True,
            True,
            True,
            None,
            None,
            None,
            None,
            None,
            None,
            *([None] * 4),
            *([None] * 9),
        ],
        # Occurs on events MVR, values 'pvt_vals'.
        # Has pre-set keys with value types as follows:
        #          Value Type
        #   pvt_cat_int: NumericDataModalitySubtype.CATEGORICAL_INTEGER,
        #   pvt_cat_flt: NumericDataModalitySubtype.CATEGORICAL_FLOAT,
        #   pvt_int:     NumericDataModalitySubtype.INTEGER,
        #   pvt_flt:     NumericDataModalitySubtype.FLOAT,
        #   pvt_drp:     NumericDataModalitySubtype.DROPPED,
        # Also has extra key not in the pre-set of 'pvt_added'
        "multivariate_regression_preset_value_type": [
            "pvt_int",
            "pvt_cat_int__EQ_2",
            "pvt_added",
            "pvt_flt",
            "pvt_cat_int__EQ_1",
            "pvt_drp",
            "pvt_cat_flt__EQ_1.0",
            "pvt_cat_int__EQ_2",
            "pvt_cat_flt__EQ_1.0",
            "pvt_int",
            "pvt_cat_int__EQ_1",
            "pvt_cat_flt__EQ_2.0",
            "pvt_drp",
            "pvt_cat_flt__EQ_2.0",
            "pvt_flt",
            "pvt_added",
            None,
            *([None] * 4),
            *([None] * 9),
        ],
        "pvt_vals": [
            0,
            np.NaN,
            0,
            1.0,
            np.NaN,
            np.NaN,
            np.NaN,
            np.NaN,
            np.NaN,
            1,
            np.NaN,
            np.NaN,
            np.NaN,
            np.NaN,
            0.0,
            1,
            None,
            *([None] * 4),
            *([None] * 9),
        ],
        "multivariate_regression_preset_value_type_is_inlier": [
            True,
            None,
            True,
            True,
            None,
            None,
            None,
            None,
            None,
            True,
            None,
            None,
            None,
            None,
            True,
            True,
            None,
            *([None] * 4),
            *([None] * 9),
        ],
        # Occurs on events MVR, values 'mrnp_vals'.
        # Keys include:
        #   'mrnp_flt', 'mrnp_int', 'mrnp_cat_int__EQ_1', 'mrnp_cat_int__EQ_2', 'mrnp_cat_int__EQ_3',
        #   'mrnp_dropped' and 'mrnp_key_dropped'
        # These should result in types float, int, categorical int, dropped, and 'mrnp_key_dropped' should be
        # dropped wholesale.
        # Event IDs
        # *([1]*4 + [2]*4 + [3]*4 + [4]*5),
        # ... after agg
        # *([1]*8 + [2]*9),
        # *([3]*2 + [4]*2),
        "multivariate_regression_no_preset": [
            "mrnp_dropped",
            "mrnp_flt",
            "mrnp_flt",
            "UNK",
            "mrnp_int",
            "mrnp_int",
            "mrnp_cat_int__EQ_1",
            "mrnp_cat_int__EQ_1",
            "mrnp_cat_int__EQ_2",
            "mrnp_cat_int__EQ_2",
            "mrnp_cat_int__EQ_3",
            "mrnp_cat_int__EQ_3",
            "UNK",
            "UNK",
            "mrnp_flt",
            "mrnp_dropped",
            "mrnp_int",
            *([None] * 4),
            *([None] * 9),
        ],
        "mrnp_vals": [
            np.NaN,
            2.0,
            np.NaN,
            np.NaN,
            np.NaN,
            2.0,
            np.NaN,
            np.NaN,
            np.NaN,
            np.NaN,
            np.NaN,
            np.NaN,
            np.NaN,
            np.NaN,
            0.2,
            np.NaN,
            0.0,
            *([None] * 4),
            *([None] * 9),
        ],
        "multivariate_regression_no_preset_is_inlier": [
            None,
            True,
            False,
            None,
            False,
            True,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            True,
            None,
            True,
            *([None] * 4),
            *([None] * 9),
        ],
    },
    schema={
        "measurement_id": pl.UInt8,
        "event_id": pl.UInt8,
        "dynamic_preset_vocab": pl.Categorical,
        "dynamic_dropped_insufficient_occurrences": pl.Categorical,
        "multivariate_regression_bounded_outliers": pl.Categorical,
        "mrbo_vals": pl.Float64,
        "multivariate_regression_bounded_outliers_is_inlier": pl.Boolean,
        "multivariate_regression_preset_value_type": pl.Categorical,
        "pvt_vals": pl.Float64,
        "multivariate_regression_preset_value_type_is_inlier": pl.Boolean,
        "multivariate_regression_no_preset": pl.Categorical,
        "mrnp_vals": pl.Float64,
        "multivariate_regression_no_preset_is_inlier": pl.Boolean,
    },
)

# Events:
# 'subject_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
# Measurements Idxmap
# event_type, dynamic_preset_vocab, multivariate_regression_bounded_outliers,
# multivariate_regression_no_preset, multivariate_regression_preset_value_type, static,
# time_dependent_age_all, time_dependent_age_lt_90, time_dependent_time_of_day,

start_times = [want_event_times[1], want_event_times[6], want_event_times[10], None, None]
WANT_DL_REP_DF = pl.DataFrame(
    {
        "subject_id": [1, 2, 3, 4, 5],
        "start_time": start_times,
        "time": [
            [(want_event_times[i] - start_times[0]) / timedelta(minutes=1) for i in range(1, 6)],
            [(want_event_times[i] - start_times[1]) / timedelta(minutes=1) for i in range(6, 10)],
            [(want_event_times[i] - start_times[2]) / timedelta(minutes=1) for i in range(10, 14)],
            None,
            None,
        ],
        "static_indices": [
            [WANT_UNIFIED_VOCABULARY_IDXMAP["static"]["foo"]],
            [WANT_UNIFIED_VOCABULARY_IDXMAP["static"]["foo"]],
            [WANT_UNIFIED_VOCABULARY_IDXMAP["static"]["bar"]],
            [WANT_UNIFIED_VOCABULARY_IDXMAP["static"]["bar"]],
            [WANT_UNIFIED_VOCABULARY_IDXMAP["static"]["bar"]],
        ],
        "static_measurement_indices": [
            [WANT_MEASUREMENTS_IDXMAP["static"]],
            [WANT_MEASUREMENTS_IDXMAP["static"]],
            [WANT_MEASUREMENTS_IDXMAP["static"]],
            [WANT_MEASUREMENTS_IDXMAP["static"]],
            [WANT_MEASUREMENTS_IDXMAP["static"]],
        ],
        "dynamic_indices": [
            [
                [
                    WANT_UNIFIED_VOCABULARY_IDXMAP["event_type"]["MVR"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_all"]["time_dependent_age_all"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_lt_90"]["time_dependent_age_lt_90"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_time_of_day"][want_event_TODs[1]],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_bounded_outliers"]["mrbo1"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["mrnp_dropped"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"]["pvt_int"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_bounded_outliers"]["mrbo3"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["mrnp_flt"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"][
                        "pvt_cat_int__EQ_2"
                    ],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_bounded_outliers"]["mrbo2"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["mrnp_flt"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"]["pvt_added"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_bounded_outliers"]["mrbo1"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["UNK"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"]["pvt_flt"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_bounded_outliers"]["mrbo2"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["mrnp_int"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"][
                        "pvt_cat_int__EQ_1"
                    ],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_bounded_outliers"]["mrbo1"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["mrnp_int"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"]["pvt_drp"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_bounded_outliers"]["mrbo3"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["mrnp_cat_int__EQ_1"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"][
                        "pvt_cat_flt__EQ_1.0"
                    ],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_bounded_outliers"]["mrbo2"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["mrnp_cat_int__EQ_1"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"][
                        "pvt_cat_int__EQ_2"
                    ],
                ],
                [
                    WANT_UNIFIED_VOCABULARY_IDXMAP["event_type"]["DDIC"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_all"]["time_dependent_age_all"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_lt_90"]["time_dependent_age_lt_90"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_time_of_day"][want_event_TODs[2]],
                ],
                [
                    WANT_UNIFIED_VOCABULARY_IDXMAP["event_type"]["DPV"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_all"]["time_dependent_age_all"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_lt_90"]["time_dependent_age_lt_90"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_time_of_day"][want_event_TODs[3]],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["dynamic_preset_vocab"]["foo"],
                ],
                [
                    WANT_UNIFIED_VOCABULARY_IDXMAP["event_type"]["DPV"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_all"]["time_dependent_age_all"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_lt_90"]["time_dependent_age_lt_90"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_time_of_day"][want_event_TODs[4]],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["dynamic_preset_vocab"]["foo"],
                ],
                [
                    WANT_UNIFIED_VOCABULARY_IDXMAP["event_type"]["DPV"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_all"]["time_dependent_age_all"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_lt_90"]["time_dependent_age_lt_90"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_time_of_day"][want_event_TODs[5]],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["dynamic_preset_vocab"]["bar"],
                ],
            ],
            [
                [
                    WANT_UNIFIED_VOCABULARY_IDXMAP["event_type"]["MVR"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_all"]["time_dependent_age_all"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_lt_90"]["time_dependent_age_lt_90"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_time_of_day"][want_event_TODs[6]],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_bounded_outliers"]["mrbo3"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["mrnp_cat_int__EQ_2"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"][
                        "pvt_cat_flt__EQ_1.0"
                    ],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_bounded_outliers"]["mrbo2"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["mrnp_cat_int__EQ_2"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"]["pvt_int"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_bounded_outliers"]["mrbo1"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["mrnp_cat_int__EQ_3"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"][
                        "pvt_cat_int__EQ_1"
                    ],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_bounded_outliers"]["mrbo3"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["mrnp_cat_int__EQ_3"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"][
                        "pvt_cat_flt__EQ_2.0"
                    ],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["UNK"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"]["pvt_drp"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["UNK"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"][
                        "pvt_cat_flt__EQ_2.0"
                    ],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["mrnp_flt"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"]["pvt_flt"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["mrnp_dropped"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_preset_value_type"]["pvt_added"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["multivariate_regression_no_preset"]["mrnp_int"],
                ],
                [
                    WANT_UNIFIED_VOCABULARY_IDXMAP["event_type"]["DDIC"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_all"]["time_dependent_age_all"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_lt_90"]["time_dependent_age_lt_90"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_time_of_day"][want_event_TODs[7]],
                ],
                [
                    WANT_UNIFIED_VOCABULARY_IDXMAP["event_type"]["DPV"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_all"]["time_dependent_age_all"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_lt_90"]["time_dependent_age_lt_90"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_time_of_day"][want_event_TODs[8]],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["dynamic_preset_vocab"]["bar"],
                ],
                [
                    WANT_UNIFIED_VOCABULARY_IDXMAP["event_type"]["DPV"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_all"]["time_dependent_age_all"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_lt_90"]["time_dependent_age_lt_90"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_time_of_day"][want_event_TODs[9]],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["dynamic_preset_vocab"]["bar"],
                ],
            ],
            [
                [
                    WANT_UNIFIED_VOCABULARY_IDXMAP["event_type"]["DPV"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_all"]["time_dependent_age_all"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_lt_90"]["time_dependent_age_lt_90"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_time_of_day"][want_event_TODs[10]],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["dynamic_preset_vocab"]["UNK"],
                ],
                [
                    WANT_UNIFIED_VOCABULARY_IDXMAP["event_type"]["DPV"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_all"]["time_dependent_age_all"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_lt_90"]["time_dependent_age_lt_90"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_time_of_day"][want_event_TODs[11]],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["dynamic_preset_vocab"]["UNK"],
                ],
                [
                    WANT_UNIFIED_VOCABULARY_IDXMAP["event_type"]["DPV"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_all"]["time_dependent_age_all"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_lt_90"]["time_dependent_age_lt_90"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_time_of_day"][want_event_TODs[12]],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["dynamic_preset_vocab"]["foo"],
                ],
                [
                    WANT_UNIFIED_VOCABULARY_IDXMAP["event_type"]["DPV"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_all"]["time_dependent_age_all"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_age_lt_90"]["time_dependent_age_lt_90"],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["time_dependent_time_of_day"][want_event_TODs[13]],
                    WANT_UNIFIED_VOCABULARY_IDXMAP["dynamic_preset_vocab"]["foo"],
                ],
            ],
            None,
            None,
        ],
        "dynamic_measurement_indices": [
            [
                [
                    WANT_MEASUREMENTS_IDXMAP["event_type"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_all"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_lt_90"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_time_of_day"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_bounded_outliers"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_bounded_outliers"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_bounded_outliers"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_bounded_outliers"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_bounded_outliers"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_bounded_outliers"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_bounded_outliers"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_bounded_outliers"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                ],
                [
                    WANT_MEASUREMENTS_IDXMAP["event_type"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_all"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_lt_90"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_time_of_day"],
                ],
                [
                    WANT_MEASUREMENTS_IDXMAP["event_type"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_all"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_lt_90"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_time_of_day"],
                    WANT_MEASUREMENTS_IDXMAP["dynamic_preset_vocab"],
                ],
                [
                    WANT_MEASUREMENTS_IDXMAP["event_type"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_all"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_lt_90"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_time_of_day"],
                    WANT_MEASUREMENTS_IDXMAP["dynamic_preset_vocab"],
                ],
                [
                    WANT_MEASUREMENTS_IDXMAP["event_type"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_all"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_lt_90"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_time_of_day"],
                    WANT_MEASUREMENTS_IDXMAP["dynamic_preset_vocab"],
                ],
            ],
            [
                [
                    WANT_MEASUREMENTS_IDXMAP["event_type"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_all"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_lt_90"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_time_of_day"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_bounded_outliers"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_bounded_outliers"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_bounded_outliers"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_bounded_outliers"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_preset_value_type"],
                    WANT_MEASUREMENTS_IDXMAP["multivariate_regression_no_preset"],
                ],
                [
                    WANT_MEASUREMENTS_IDXMAP["event_type"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_all"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_lt_90"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_time_of_day"],
                ],
                [
                    WANT_MEASUREMENTS_IDXMAP["event_type"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_all"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_lt_90"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_time_of_day"],
                    WANT_MEASUREMENTS_IDXMAP["dynamic_preset_vocab"],
                ],
                [
                    WANT_MEASUREMENTS_IDXMAP["event_type"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_all"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_lt_90"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_time_of_day"],
                    WANT_MEASUREMENTS_IDXMAP["dynamic_preset_vocab"],
                ],
            ],
            [
                [
                    WANT_MEASUREMENTS_IDXMAP["event_type"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_all"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_lt_90"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_time_of_day"],
                    WANT_MEASUREMENTS_IDXMAP["dynamic_preset_vocab"],
                ],
                [
                    WANT_MEASUREMENTS_IDXMAP["event_type"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_all"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_lt_90"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_time_of_day"],
                    WANT_MEASUREMENTS_IDXMAP["dynamic_preset_vocab"],
                ],
                [
                    WANT_MEASUREMENTS_IDXMAP["event_type"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_all"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_lt_90"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_time_of_day"],
                    WANT_MEASUREMENTS_IDXMAP["dynamic_preset_vocab"],
                ],
                [
                    WANT_MEASUREMENTS_IDXMAP["event_type"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_all"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_age_lt_90"],
                    WANT_MEASUREMENTS_IDXMAP["time_dependent_time_of_day"],
                    WANT_MEASUREMENTS_IDXMAP["dynamic_preset_vocab"],
                ],
            ],
            None,
            None,
        ],
        "dynamic_values": [
            [
                [
                    None,
                    want_events_ts_ages_all[1],
                    want_events_ts_ages_lt_90[1],
                    None,
                    np.NaN,
                    np.NaN,
                    0,
                    10.1,
                    2.0,
                    np.NaN,
                    5.1,
                    np.NaN,
                    0,
                    0.6,
                    np.NaN,
                    1.0,
                    -0.1,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                    2.0,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                ],
                [
                    None,
                    want_events_ts_ages_all[2],
                    want_events_ts_ages_lt_90[2],
                    None,
                ],
                [
                    None,
                    want_events_ts_ages_all[3],
                    want_events_ts_ages_lt_90[3],
                    None,
                    None,
                ],
                [
                    None,
                    want_events_ts_ages_all[4],
                    want_events_ts_ages_lt_90[4],
                    None,
                    None,
                ],
                [
                    None,
                    want_events_ts_ages_all[5],
                    want_events_ts_ages_lt_90[5],
                    None,
                    None,
                ],
            ],
            [
                [
                    None,
                    want_events_ts_ages_all[6],
                    want_events_ts_ages_lt_90[6],
                    None,
                    -0.1,
                    np.NaN,
                    np.NaN,
                    0.1,
                    np.NaN,
                    1,
                    0.1,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                    0.2,
                    0.0,
                    np.NaN,
                    1,
                    0.0,
                ],
                [
                    None,
                    want_events_ts_ages_all[7],
                    want_events_ts_ages_lt_90[7],
                    None,
                ],
                [
                    None,
                    want_events_ts_ages_all[8],
                    want_events_ts_ages_lt_90[8],
                    None,
                    None,
                ],
                [
                    None,
                    want_events_ts_ages_all[9],
                    want_events_ts_ages_lt_90[9],
                    None,
                    None,
                ],
            ],
            [
                [
                    None,
                    want_events_ts_ages_all[10],
                    want_events_ts_ages_lt_90[10],
                    None,
                    None,
                ],
                [
                    None,
                    want_events_ts_ages_all[11],
                    want_events_ts_ages_lt_90[11],
                    None,
                    None,
                ],
                [
                    None,
                    want_events_ts_ages_all[12],
                    want_events_ts_ages_lt_90[12],
                    None,
                    None,
                ],
                [
                    None,
                    want_events_ts_ages_all[13],
                    want_events_ts_ages_lt_90[13],
                    None,
                    None,
                ],
            ],
            None,
            None,
        ],
    },
    schema={
        "subject_id": pl.UInt8,
        "start_time": pl.Datetime,
        "time": pl.List(pl.Float64),
        "static_indices": pl.List(pl.UInt8),
        "static_measurement_indices": pl.List(pl.UInt8),
        "dynamic_indices": pl.List(pl.List(pl.UInt8)),
        "dynamic_measurement_indices": pl.List(pl.List(pl.UInt8)),
        "dynamic_values": pl.List(pl.List(pl.Float64)),
    },
).with_columns(
    pl.when(pl.col("dynamic_indices").list.lengths() == 0)
    .then(pl.lit(None))
    .otherwise(pl.col("dynamic_indices"))
    .alias("dynamic_indices"),
    pl.when(pl.col("dynamic_measurement_indices").list.lengths() == 0)
    .then(pl.lit(None))
    .otherwise(pl.col("dynamic_measurement_indices"))
    .alias("dynamic_measurement_indices"),
    pl.when(pl.col("dynamic_values").list.lengths() == 0)
    .then(pl.lit(None))
    .otherwise(pl.col("dynamic_values"))
    .alias("dynamic_values"),
)


class TestDatasetEndToEnd(ConfigComparisonsMixin, unittest.TestCase):
    def test_end_to_end(self):
        E = ESDMock(
            config=TEST_CONFIG,
            subjects_df=IN_SUBJECTS_DF,
            events_df=IN_EVENTS_DF,
            dynamic_measurements_df=IN_MEASUREMENTS_DF,
        )

        E.split_subjects = TEST_SPLIT

        E.preprocess()

        self.assertNestedDictEqual(
            WANT_INFERRED_MEASUREMENT_CONFIGS, E.inferred_measurement_configs, check_like=True
        )
        self.assertEqual(WANT_SUBJECTS_DF, E.subjects_df)
        self.assertEqual(WANT_EVENTS_DF, E.events_df)
        self.assertEqual(WANT_MEASUREMENTS_DF, E.dynamic_measurements_df)

        self.assertEqual(WANT_EVENT_TYPES, E.event_types)
        self.assertEqual(WANT_MEASUREMENTS_IDXMAP, E.unified_measurements_idxmap)
        self.assertEqual(WANT_UNIFIED_VOCABULARY_OFFSETS, E.unified_vocabulary_offsets)
        self.assertNestedDictEqual(WANT_UNIFIED_VOCABULARY_IDXMAP, E.unified_vocabulary_idxmap)

        got_DL_rep = E.build_DL_cached_representation(do_sort_outputs=True)
        self.assertEqual(WANT_DL_REP_DF.drop("dynamic_values"), got_DL_rep.drop("dynamic_values"))

        exploded_expr = pl.col("dynamic_values").list.explode().list.explode().alias("dynamic_values")
        want_expl = WANT_DL_REP_DF.select(exploded_expr)
        got_expl = got_DL_rep.select(exploded_expr)

        self.assertEqual(want_expl, got_expl)

        with self.subTest("Save/load should work"):
            with TemporaryDirectory() as d:
                save_dir = Path(d) / "save_dir"
                E.config.save_dir = save_dir
                E.save()

                got_E = Dataset.load(save_dir)

                self.assertEqual(WANT_MEASUREMENTS_DF, got_E.dynamic_measurements_df)
                self.assertEqual(WANT_EVENTS_DF, got_E.events_df)
                self.assertEqual(WANT_SUBJECTS_DF, got_E.subjects_df)

                got_inferred_measurement_configs = got_E.inferred_measurement_configs
                for v in got_inferred_measurement_configs.values():
                    v.uncache_measurement_metadata()

                self.assertNestedDictEqual(
                    WANT_INFERRED_MEASUREMENT_CONFIGS, got_inferred_measurement_configs
                )

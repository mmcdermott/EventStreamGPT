import sys

sys.path.append("../..")

import dataclasses
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from EventStream.data.config import (
    DatasetConfig,
    MeasurementConfig,
    PytorchDatasetConfig,
)
from EventStream.data.time_dependent_functor import AgeFunctor
from EventStream.data.types import DataModality, TemporalityType
from EventStream.data.vocabulary import Vocabulary

from ..utils import ConfigComparisonsMixin


class TestPytorchDatasetConfig(unittest.TestCase):
    def test_validation(self):
        cases = [
            {"msg": "Should construct on default arguments", "kwargs": {}},
            {"msg": "Should construct with min_seq_len = 0", "kwargs": {"min_seq_len": 0}},
            {
                "msg": "Should construct with max_seq_len = 1",
                "kwargs": {"max_seq_len": 1, "min_seq_len": 0},
            },
            {
                "msg": "Should construct with min_seq_len = max_seq_len",
                "kwargs": {"max_seq_len": 10, "min_seq_len": 10},
            },
            {
                "msg": "Should construct with seq_padding_side='left'",
                "kwargs": {"seq_padding_side": "left"},
            },
            {
                "msg": "Should construct with seq_padding_side='right'",
                "kwargs": {"seq_padding_side": "right"},
            },
            {
                "msg": "Shouldn't construct with min_seq_len = -1",
                "kwargs": {"min_seq_len": -1},
                "should_raise": ValueError,
            },
            {
                "msg": "Shouldn't construct with max_seq_len = 0",
                "kwargs": {"max_seq_len": 0},
                "should_raise": ValueError,
            },
            {
                "msg": "Shouldn't construct with min_seq_len > max_seq_len",
                "kwargs": {"max_seq_len": 10, "min_seq_len": 11},
                "should_raise": ValueError,
            },
            {
                "msg": "Shouldn't construct with seq_padding_side=None",
                "kwargs": {"seq_padding_side": None},
                "should_raise": ValueError,
            },
        ]

        for C in cases:
            with self.subTest(C["msg"]):
                if C.get("should_raise", None) is not None:
                    with self.assertRaises(C["should_raise"]):
                        PytorchDatasetConfig(**C["kwargs"])
                else:
                    PytorchDatasetConfig(**C["kwargs"])


class TestMeasurementConfig(ConfigComparisonsMixin, unittest.TestCase):
    def test_validates_params(self):
        valid_kwargs = [
            dict(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.MULTI_LABEL_CLASSIFICATION,
            ),
            dict(
                temporality=TemporalityType.STATIC,
                modality=DataModality.MULTI_LABEL_CLASSIFICATION,
            ),
            dict(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.MULTIVARIATE_REGRESSION,
                values_column="val",
            ),
            dict(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.MULTIVARIATE_REGRESSION,
                values_column="val",
                _measurement_metadata=pd.DataFrame(
                    {"censor_lower_bound": [1, 0.2, 0.1]},
                    index=pd.Index(["foo", "bar", "baz"], name="key"),
                ),
            ),
            dict(
                temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                modality=DataModality.UNIVARIATE_REGRESSION,
                _measurement_metadata=pd.Series([None]),
                functor=AgeFunctor("dob"),
            ),
            dict(
                temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                modality="dropped",
                functor=AgeFunctor("dob"),
            ),
        ]
        for kwargs in valid_kwargs:
            MeasurementConfig(**kwargs)

        invalid_kwargs = [
            dict(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
            ),
            dict(
                temporality=TemporalityType.STATIC,
                modality=DataModality.MULTIVARIATE_REGRESSION,
                values_column="val",
            ),
            dict(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.MULTIVARIATE_REGRESSION,
                values_column="val",
                functor=AgeFunctor("dob"),
            ),
            dict(
                temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                modality=DataModality.MULTIVARIATE_REGRESSION,
                values_column="val",
                functor=AgeFunctor("dob"),
            ),
            dict(
                temporality=TemporalityType.STATIC,
                modality=DataModality.UNIVARIATE_REGRESSION,
                _measurement_metadata=pd.Series([None]),
            ),
            dict(
                modality="dropped",
            ),
            dict(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
                values_column="val",
            ),
            dict(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.MULTI_LABEL_CLASSIFICATION,
                _measurement_metadata=pd.Series([None]),
            ),
            dict(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.MULTIVARIATE_REGRESSION,
                values_column=None,
            ),
            dict(
                temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                modality=DataModality.UNIVARIATE_REGRESSION,
                _measurement_metadata=pd.DataFrame({"value_type": []}, index=pd.Index([])),
                functor=AgeFunctor("dob"),
            ),
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(str(kwargs)):
                with self.assertRaises((AssertionError, ValueError, NotImplementedError)):
                    MeasurementConfig(**kwargs)

    def test_add_missing_mandatory_metadata_cols(self):
        config = MeasurementConfig(
            temporality=TemporalityType.DYNAMIC,
            modality=DataModality.MULTIVARIATE_REGRESSION,
            _measurement_metadata=pd.DataFrame({"value_type": []}, index=pd.Index([])),
            values_column="vals",
            vocabulary=Vocabulary(vocabulary=["UNK", "A", "B"], obs_frequencies=[0, 0.5, 0.5]),
        )

        config.add_missing_mandatory_metadata_cols()
        want_measurement_metadata = pd.DataFrame(
            {
                "value_type": [],
                "mean": pd.Series([], dtype=float),
                "std": pd.Series([], dtype=float),
                "thresh_small": pd.Series([], dtype=float),
                "thresh_large": pd.Series([], dtype=float),
            },
            index=pd.Index([]),
        )
        self.assertEqual(want_measurement_metadata, config.measurement_metadata)

        config = MeasurementConfig(
            temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
            functor=AgeFunctor,
            modality=DataModality.UNIVARIATE_REGRESSION,
            _measurement_metadata=pd.Series([None], index=pd.Index(["value_type"])),
        )

        config.add_missing_mandatory_metadata_cols()
        want_measurement_metadata = pd.Series(
            [None, None, None, None, None],
            index=pd.Index(["value_type", "mean", "std", "thresh_small", "thresh_large"]),
        )
        self.assertEqual(want_measurement_metadata, config.measurement_metadata)

        config = MeasurementConfig(
            temporality=TemporalityType.DYNAMIC,
            modality=DataModality.MULTI_LABEL_CLASSIFICATION,
        )

        with self.assertRaises(ValueError):
            config.add_missing_mandatory_metadata_cols()

    def test_properties(self):
        config = MeasurementConfig(
            temporality=TemporalityType.DYNAMIC,
            modality=DataModality.MULTIVARIATE_REGRESSION,
            values_column="vals",
        )
        self.assertTrue(config.is_numeric)
        self.assertFalse(config.is_dropped)

        config = MeasurementConfig(
            temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
            functor=AgeFunctor,
            modality=DataModality.UNIVARIATE_REGRESSION,
        )
        self.assertTrue(config.is_numeric)
        self.assertFalse(config.is_dropped)

        config = MeasurementConfig(
            temporality=TemporalityType.DYNAMIC,
            modality=DataModality.DROPPED,
        )
        self.assertFalse(config.is_numeric)
        self.assertTrue(config.is_dropped)

        config = MeasurementConfig(
            temporality=TemporalityType.STATIC,
            modality=DataModality.MULTI_LABEL_CLASSIFICATION,
        )
        self.assertFalse(config.is_numeric)
        self.assertFalse(config.is_dropped)

    def test_measurement_metadata_property(self):
        cases = [
            {
                "msg": "Should work for properly formed univariate cases.",
                "config": dict(
                    modality=DataModality.UNIVARIATE_REGRESSION,
                    _measurement_metadata=pd.Series(
                        [2],
                        index=pd.Index(["mean"]),
                        name="key",
                    ),
                ),
            },
            {
                "msg": "Should work for properly formed multivariate cases.",
                "config": dict(
                    modality=DataModality.MULTIVARIATE_REGRESSION,
                    values_column="val",
                    _measurement_metadata=pd.DataFrame(
                        {
                            "censor_lower_bound": [1, 0.2, 0.1],
                        },
                        index=pd.Index(["foo", "bar", "baz"], name="key"),
                    ),
                ),
            },
        ]

        for case in cases:
            with self.subTest(case["msg"]):
                config = MeasurementConfig(temporality=TemporalityType.DYNAMIC, **case["config"])

                # Should not raise an error.
                old_meas_metadata = config.measurement_metadata

                with TemporaryDirectory() as d:
                    config.cache_measurement_metadata(Path(d), "data.csv")

                    if case.get("want_raise", None) is not None:
                        with self.assertRaises(case["want_raise"]):
                            config.measurement_metadata
                    else:
                        new_meas_metadata = config.measurement_metadata
                        self.assertEqual(old_meas_metadata, new_meas_metadata)

    def test_drop(self):
        config = MeasurementConfig(
            temporality=TemporalityType.DYNAMIC,
            modality=DataModality.MULTIVARIATE_REGRESSION,
            _measurement_metadata=pd.DataFrame({"value_type": []}, index=pd.Index([])),
            values_column="vals",
            vocabulary=Vocabulary(vocabulary=["UNK", "A", "B"], obs_frequencies=[0, 0.5, 0.5]),
        )

        config.drop()
        self.assertEqual(DataModality.DROPPED, config.modality)
        self.assertEqual("vals", config.values_column)
        self.assertEqual(None, config.vocabulary)
        self.assertEqual(None, config.measurement_metadata)

    def test_add_empty_metadata(self):
        config = MeasurementConfig(
            name="foo",
            temporality=TemporalityType.DYNAMIC,
            modality=DataModality.MULTIVARIATE_REGRESSION,
            values_column="vals",
        )

        config.add_empty_metadata()
        want_metadata = pd.DataFrame(
            {
                "value_type": pd.Series([], dtype=str),
                "mean": pd.Series([], dtype=float),
                "std": pd.Series([], dtype=float),
                "thresh_small": pd.Series([], dtype=float),
                "thresh_large": pd.Series([], dtype=float),
            },
            index=pd.Index([], name="foo"),
        )
        self.assertEqual(want_metadata, config.measurement_metadata)

        with self.assertRaises(ValueError):
            config.add_empty_metadata()

        config = MeasurementConfig(
            name="bar",
            temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
            modality=DataModality.UNIVARIATE_REGRESSION,
            functor=AgeFunctor("dob"),
        )

        config.add_empty_metadata()
        want_metadata = pd.Series(
            [None, None, None, None, None],
            index=pd.Index(["value_type", "mean", "std", "thresh_small", "thresh_large"]),
        )
        self.assertEqual(want_metadata, config.measurement_metadata)

    def test_to_and_from_dict(self):
        default_dict = {
            "name": None,
            "modality": DataModality.MULTI_LABEL_CLASSIFICATION,
            "temporality": TemporalityType.DYNAMIC,
            "vocabulary": None,
            "observation_rate_over_cases": None,
            "observation_rate_per_case": None,
            "functor": None,
            "values_column": None,
            "_measurement_metadata": None,
            "modifiers": None,
        }
        nontrivial_measurement_metadata_df = pd.DataFrame(
            {"A": [1, 2, 3], "B": ["a", "b", "c"]},
            index=pd.Index([2, 4, 6], name="col"),
        )
        nontrivial_measurement_metadata_series = pd.Series(["foo"], index=pd.Index(["value_type"]))
        nontrivial_vocabulary = Vocabulary(vocabulary=["UNK", "A", "B"], obs_frequencies=[0, 0.5, 0.5])

        cases = [
            {
                "msg": "Should work when all params are None.",
                "config": MeasurementConfig(
                    temporality=TemporalityType.DYNAMIC,
                    modality=DataModality.MULTI_LABEL_CLASSIFICATION,
                ),
                "want_dict": {**default_dict},
            },
            {
                "msg": "Should work when measurement_metadata is a dataframe.",
                "config": MeasurementConfig(
                    temporality=TemporalityType.DYNAMIC,
                    modality=DataModality.MULTIVARIATE_REGRESSION,
                    _measurement_metadata=nontrivial_measurement_metadata_df,
                    values_column="foo",
                ),
                "want_dict": {
                    **default_dict,
                    "modality": "multivariate_regression",
                    "values_column": "foo",
                    "_measurement_metadata": {
                        "index": [2, 4, 6],
                        "columns": ["A", "B"],
                        "data": [[1, "a"], [2, "b"], [3, "c"]],
                        "index_names": ["col"],
                        "column_names": [None],
                    },
                },
            },
            {
                "msg": "Should work when measurement_metadata is a series.",
                "config": MeasurementConfig(
                    temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                    modality=DataModality.UNIVARIATE_REGRESSION,
                    _measurement_metadata=nontrivial_measurement_metadata_series,
                    functor=AgeFunctor("dob"),
                ),
                "want_dict": {
                    **default_dict,
                    "temporality": "functional_time_dependent",
                    "modality": "univariate_regression",
                    "_measurement_metadata": {"value_type": "foo"},
                    "functor": {"class": "AgeFunctor", "params": {"dob_col": "dob"}},
                },
            },
            {
                "msg": "Should work when vocabulary is not None",
                "config": MeasurementConfig(
                    temporality=TemporalityType.STATIC,
                    modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
                    vocabulary=nontrivial_vocabulary,
                ),
                "want_dict": {
                    **default_dict,
                    "temporality": "static",
                    "modality": "single_label_classification",
                    "vocabulary": dataclasses.asdict(nontrivial_vocabulary),
                },
            },
        ]

        for C in cases:
            with self.subTest(C["msg"]):
                got_dict = C["config"].to_dict()
                self.assertNestedDictEqual(C["want_dict"], got_dict)

                got_config = MeasurementConfig.from_dict(C["want_dict"])
                self.assertEqual(C["config"], got_config)


class TestDatasetConfig(ConfigComparisonsMixin, unittest.TestCase):
    def test_validates_params(self):
        valid_kwargs = [
            dict(),
            dict(
                min_valid_column_observations=10,
                min_valid_vocab_element_observations=12,
                min_true_float_frequency=1e-6,
                min_unique_numerical_observations=13,
            ),
            dict(
                min_valid_column_observations=0.5,
                min_valid_vocab_element_observations=1 - 1e-6,
                min_true_float_frequency=1 - 1e-6,
                min_unique_numerical_observations=1e-6,
            ),
            dict(
                outlier_detector_config={},
            ),
        ]
        for kwargs in valid_kwargs:
            DatasetConfig(**kwargs)

        invalid_kwargs = [
            dict(
                min_valid_column_observations=1.5,
            ),
            dict(
                min_valid_vocab_element_observations=1,
            ),
            dict(
                min_true_float_frequency=1.0,
            ),
            dict(
                min_true_float_frequency=10,
            ),
            dict(
                min_unique_numerical_observations=2.0,
            ),
            dict(
                outlier_detector_config="foo",
            ),
        ]
        for kwargs in invalid_kwargs:
            with self.assertRaises((ValueError, TypeError)):
                DatasetConfig(**kwargs)

    def test_to_and_from_dict(self):
        default_dict = dict(
            measurement_configs={},
            min_valid_column_observations=None,
            min_valid_vocab_element_observations=None,
            min_true_float_frequency=None,
            min_unique_numerical_observations=None,
            outlier_detector_config=None,
            center_and_scale=True,
            save_dir=None,
            min_events_per_subject=None,
            agg_by_time_scale="1h",
        )
        nontrivial_measurement_configs = {
            "col_A": MeasurementConfig(
                modality=DataModality.MULTIVARIATE_REGRESSION,
                temporality=TemporalityType.DYNAMIC,
                values_column="foo",
                _measurement_metadata=pd.DataFrame(
                    {"A": [1, 2, 3], "B": ["a", "b", "c"]},
                    index=pd.Index([2, 4, 6], name="index_var"),
                ),
            ),
            "col_B": MeasurementConfig(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.MULTI_LABEL_CLASSIFICATION,
            ),
        }
        nontrivial_outlier_config = {"cls": "outlier", "foo": "bar"}

        cases = [
            {
                "msg": "Should work when all params are None.",
                "config": DatasetConfig(),
                "want_dict": {**default_dict},
            },
            {
                "msg": "Should work when measurement_configs is not None",
                "config": DatasetConfig(measurement_configs=nontrivial_measurement_configs),
                "want_dict": {
                    **default_dict,
                    "measurement_configs": {
                        k: cfg.to_dict() for k, cfg in nontrivial_measurement_configs.items()
                    },
                },
            },
            {
                "msg": "Should work when sub-model configs are not None",
                "config": DatasetConfig(
                    outlier_detector_config=nontrivial_outlier_config,
                ),
                "want_dict": {
                    **default_dict,
                    "outlier_detector_config": nontrivial_outlier_config,
                },
            },
        ]

        for C in cases:
            with self.subTest(C["msg"]):
                got_dict = C["config"].to_dict()
                self.assertNestedDictEqual(C["want_dict"], got_dict)

                got_config = DatasetConfig.from_dict(C["want_dict"])
                self.assertEqual(C["config"], got_config)

    def test_eq(self):
        config1 = DatasetConfig(
            measurement_configs={
                "A_key": MeasurementConfig(
                    temporality=TemporalityType.DYNAMIC,
                    modality=DataModality.MULTI_LABEL_CLASSIFICATION,
                ),
                "B_key": MeasurementConfig(
                    temporality=TemporalityType.DYNAMIC,
                    modality=DataModality.MULTIVARIATE_REGRESSION,
                    values_column="B_val",
                ),
                "C": MeasurementConfig(
                    temporality=TemporalityType.STATIC,
                    modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
                ),
                "D": MeasurementConfig(
                    temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                    functor=AgeFunctor("dob"),
                ),
            },
            min_valid_column_observations=10,
            min_valid_vocab_element_observations=0.5,
            min_true_float_frequency=0.75,
            min_unique_numerical_observations=0.25,
            outlier_detector_config={"cls": "outlier", "foo": "bar"},
        )
        config2 = DatasetConfig(
            measurement_configs={
                "A_key": MeasurementConfig(
                    temporality=TemporalityType.DYNAMIC,
                    modality=DataModality.MULTI_LABEL_CLASSIFICATION,
                ),
                "B_key": MeasurementConfig(
                    temporality=TemporalityType.DYNAMIC,
                    modality=DataModality.MULTIVARIATE_REGRESSION,
                    values_column="B_val",
                ),
                "C": MeasurementConfig(
                    temporality=TemporalityType.STATIC,
                    modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
                ),
                "D": MeasurementConfig(
                    temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                    functor=AgeFunctor("dob"),
                ),
            },
            min_valid_column_observations=10,
            min_valid_vocab_element_observations=0.5,
            min_true_float_frequency=0.75,
            min_unique_numerical_observations=0.25,
            outlier_detector_config={"cls": "outlier", "foo": "bar"},
        )

        self.assertTrue(config1 == config2)

        config3 = DatasetConfig(
            measurement_configs={
                "A_key": MeasurementConfig(
                    temporality=TemporalityType.DYNAMIC,
                    modality=DataModality.MULTI_LABEL_CLASSIFICATION,
                ),
                "B_key": MeasurementConfig(
                    temporality=TemporalityType.DYNAMIC,
                    modality=DataModality.MULTIVARIATE_REGRESSION,
                    values_column="B_val",
                ),
                "C": MeasurementConfig(
                    temporality=TemporalityType.STATIC,
                    modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
                ),
                "E": MeasurementConfig(
                    temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                    functor=AgeFunctor("dob"),
                ),
            },
            min_valid_column_observations=10,
            min_valid_vocab_element_observations=0.5,
            min_true_float_frequency=0.75,
            min_unique_numerical_observations=0.25,
            outlier_detector_config={"cls": "outlier", "foo": "bar"},
        )

        self.assertFalse(config1 == config3)

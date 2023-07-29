import sys

sys.path.append("../..")

import copy
import unittest
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from EventStream.data.config import DatasetConfig, MeasurementConfig
from EventStream.data.dataset_base import DatasetBase
from EventStream.data.time_dependent_functor import AgeFunctor
from EventStream.data.types import DataModality, InputDataType, TemporalityType
from EventStream.data.vocabulary import Vocabulary

from .test_config import ConfigComparisonsMixin


class ESDMock(DatasetBase[dict, dict]):
    FUNCTIONS_CALLED = defaultdict(list)

    def __init__(self, *args, **kwargs):
        self.functions_called = defaultdict(list)
        super().__init__(*args, **kwargs)

    def _reset_functions_called(self):
        self.functions_called = defaultdict(list)

    @classmethod
    def _rename_cols(self, df: dict, mapping: dict[str, str]) -> dict:
        self.FUNCTIONS_CALLED["_rename_cols"].append((df, mapping))
        return df

    def _validate_initial_dfs(
        self, subjects_df: dict, events_df: dict, dynamic_measurements_df: dict
    ) -> tuple[dict, dict, dict]:
        self.functions_called["_validate_initial_dfs"].append(
            (subjects_df, events_df, dynamic_measurements_df)
        )
        return subjects_df, events_df, dynamic_measurements_df

    def _update_subject_event_properties(self):
        self.functions_called["_update_subject_event_properties"].append(())

    def _agg_by_time(self):
        self.functions_called["_agg_by_time"].append(())

    def _sort_events(self):
        self.functions_called["_sort_events"].append(())

    def _add_time_dependent_measurements(self):
        self.functions_called["_add_time_dependent_measurements"].append(())
        return

    def _fit_measurement_metadata(
        self, measure: str, config: MeasurementConfig, source_df: dict
    ) -> pd.DataFrame:
        self.functions_called["_fit_measurement_metadata"].append(copy.deepcopy((measure, config, source_df)))
        return config.measurement_metadata

    def _fit_vocabulary(self, measure: str, config: MeasurementConfig, source_df: dict) -> Vocabulary:
        self.functions_called["_fit_vocabulary"].append(copy.deepcopy((measure, config, source_df)))
        return Vocabulary(["foo", "bar"], [3 / 4, 1 / 4])

    def _update_attr_df(self, attr: str, df: dict):
        self.functions_called["_update_attr_df"].append((attr, df))

    def backup_numerical_measurements(self):
        self.functions_called["backup_numerical_measurements"].append(())

    def restore_numerical_measurements(self):
        self.functions_called["restore_numerical_measurements"].append(())

    def _transform_numerical_measurement(
        self, measure: str, config: MeasurementConfig, source_df: dict
    ) -> dict:
        self.functions_called["_transform_numerical_measurement"].append(
            copy.deepcopy((measure, config, source_df))
        )
        return source_df

    def _transform_categorical_measurement(
        self, measure: str, config: MeasurementConfig, source_df: dict
    ) -> dict:
        self.functions_called["_transform_categorical_measurement"].append(
            copy.deepcopy((measure, config, source_df))
        )

    def _filter_col_inclusion(self, df: dict, col: dict[str, Sequence[Any]]) -> dict:
        self.functions_called["_filter_col_inclusion"].append((df, col))
        return df

    def _read_df(self, path: Path) -> dict:
        self.functions_called["_read_df"].append((path,))
        return {}

    def _write_df(self, df: dict, path: Path):
        self.functions_called["_write_df"].append((df, path))

    def _total_possible_and_observed(
        self, measure: str, config: MeasurementConfig, source_df: dict
    ) -> tuple[int, int]:
        self.functions_called["_total_possible_and_observed"].append(
            copy.deepcopy((measure, config, source_df))
        )
        return 3, 3

    def build_DL_cached_representation(self):
        self.functions_called["build_DL_cached_representation"].append(())

    def _denormalize(self, events_df: dict, col: str) -> dict:
        self.functions_called["_denormalize"].append((events_df, col))
        return events_df

    @classmethod
    def _load_input_df(
        cls,
        df: dict,
        columns: list[tuple[str, InputDataType | tuple[InputDataType, str]]],
        subject_id_col: str,
        subject_ids_map: dict[Any, int],
        subject_id_dtype: Any,
    ) -> dict:
        cls.FUNCTIONS_CALLED["_load_input_df"].append(
            (df, columns, subject_id_col, subject_ids_map, subject_id_dtype)
        )
        return {}

    @classmethod
    def _process_events_and_measurements_df(
        cls,
        df: dict,
        event_type: str,
        columns_schema: dict[str, tuple[str, InputDataType]],
        ts_col: str | list[str],
    ) -> tuple[dict, dict | None]:
        cls.FUNCTIONS_CALLED["_process_events_and_measurements_df"].append(
            (df, event_type, columns_schema, ts_col)
        )
        return {}, None

    @classmethod
    def _split_range_events_df(
        cls, df: dict, start_ts_col: str | list[str], end_ts_col: str | list[str]
    ) -> tuple[dict, dict, dict]:
        cls.FUNCTIONS_CALLED["_split_range_events_df"].append((df, start_ts_col, end_ts_col))
        return {}, {}, {}

    @classmethod
    def _inc_df_col(cls, df: dict, col: str, inc_by: int) -> dict:
        cls.FUNCTIONS_CALLED["_inc_df_col"].append((df, col, inc_by))
        return {}

    @classmethod
    def _concat_dfs(cls, dfs: list[dict]) -> dict:
        cls.FUNCTIONS_CALLED["_concat_dfs"].append((dfs,))
        return {}

    @classmethod
    def _resolve_ts_col(cls, df: dict, ts_col: str | list[str], out_name: str = "timestamp") -> dict:
        cls.FUNCTIONS_CALLED["_resolve_ts_col"].append((df, ts_col, out_name))
        return {}


class TestDatasetBase(ConfigComparisonsMixin, unittest.TestCase):
    """Tests the `Dataset` class."""

    def setUp(self):
        super().setUp()
        self.config = DatasetConfig()
        self.subjects_df = {"name": "subjects"}
        self.events_df = {"name": "events", "event_id": [1, 2]}
        self.dynamic_measurements_df = {"name": "dynamic_measurements"}

        self.E = ESDMock(self.config, self.subjects_df, self.events_df, self.dynamic_measurements_df)

    def test_basic_construction(self):
        self.assertEqual(self.config, self.E.config)
        self.assertEqual({}, self.E.inferred_measurement_configs)

        self.assertFalse(self.E._is_fit)

        self.assertEqual(self.subjects_df, self.E.subjects_df)
        self.assertEqual(self.events_df, self.E.events_df)
        self.assertEqual(self.dynamic_measurements_df, self.E.dynamic_measurements_df)

        self.assertEqual([], self.E.subject_ids)
        self.assertEqual([], self.E.event_types)
        self.assertEqual({}, self.E.n_events_per_subject)

        want_functions_called = {
            "_validate_initial_dfs": [(self.subjects_df, self.events_df, self.dynamic_measurements_df)],
            "_update_subject_event_properties": [()],
            "_agg_by_time": [()],
            "_sort_events": [()],
        }
        self.assertEqual(want_functions_called, self.E.functions_called)

    def test_split(self):
        self.E._reset_functions_called()

        all_subject_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        self.E.subject_ids = list(all_subject_ids)

        # When passing split_fracs that don't sum to 1, `Dataset` should add an auxiliary third
        # split that captures the missing fraction, and split names should default to 'train', 'tuning',
        # and 'held_out'.
        self.E.split(split_fracs=[1 / 3, 1 / 3], seed=1)
        split_subjects_seed_1_draw_1 = self.E.split_subjects

        self.assertEqual({"train", "tuning", "held_out"}, set(split_subjects_seed_1_draw_1.keys()))
        self.assertEqual(all_subject_ids, set().union(*split_subjects_seed_1_draw_1.values()))

        # When passing split_fracs that sum to 1, `Dataset` should not add an auxiliary third
        # split, and split names should default to 'train' and 'held_out'.
        self.E.split(split_fracs=[0.5, 0.5], seed=1)
        split_subjects_seed_1_draw_1 = self.E.split_subjects

        self.assertEqual({"train", "held_out"}, set(split_subjects_seed_1_draw_1.keys()))
        self.assertEqual(all_subject_ids, set().union(*split_subjects_seed_1_draw_1.values()))

        # Passing the same seed value should result in the same split.
        self.E.split(split_fracs=[0.5, 0.5], split_names=["a", "b"], seed=1)
        split_subjects_seed_1_draw_1 = self.E.split_subjects

        self.assertEqual(all_subject_ids, set().union(*split_subjects_seed_1_draw_1.values()))
        self.assertEqual({"a", "b"}, set(split_subjects_seed_1_draw_1.keys()))

        self.E.split(split_fracs=[0.5, 0.5], split_names=["a", "b"], seed=1)
        split_subjects_seed_1_draw_2 = self.E.split_subjects

        self.E.split(split_fracs=[0.5, 0.5], split_names=["a", "b"], seed=2)
        split_subjects_seed_2_draw_1 = self.E.split_subjects

        self.assertEqual(
            split_subjects_seed_1_draw_1,
            split_subjects_seed_1_draw_2,
            msg="Splits with the same seed should be equal!",
        )
        self.assertNotEqual(
            split_subjects_seed_1_draw_1,
            split_subjects_seed_2_draw_1,
            msg="Splits with different seeds should not be equal!",
        )

        self.assertEqual({}, self.E.functions_called)

    def test_split_accessors(self):
        self.E.split_subjects = {
            "train": [1, 2, 3],
            "tuning": [4, 5, 6],
            "held_out": [7, 8, 9, 10],
        }

        cases = [
            {
                "msg": "`train_subjects_df` should filter to the 'train' split.",
                "attr": "train_subjects_df",
                "want_out": self.subjects_df,
                "want_fn": "_filter_col_inclusion",
                "want_fn_arg": (self.subjects_df, {"subject_id": [1, 2, 3]}),
            },
            {
                "msg": "`tuning_subjects_df` should filter to the 'tuning' split.",
                "attr": "tuning_subjects_df",
                "want_out": self.subjects_df,
                "want_fn": "_filter_col_inclusion",
                "want_fn_arg": (self.subjects_df, {"subject_id": [4, 5, 6]}),
            },
            {
                "msg": "`held_out_subjects_df` should filter to the 'held_out' split.",
                "attr": "held_out_subjects_df",
                "want_out": self.subjects_df,
                "want_fn": "_filter_col_inclusion",
                "want_fn_arg": (self.subjects_df, {"subject_id": [7, 8, 9, 10]}),
            },
            {
                "msg": "`train_events_df` should filter to the 'train' split.",
                "attr": "train_events_df",
                "want_out": self.events_df,
                "want_fn": "_filter_col_inclusion",
                "want_fn_arg": (self.events_df, {"subject_id": [1, 2, 3]}),
            },
            {
                "msg": "`tuning_events_df` should filter to the 'tuning' split.",
                "attr": "tuning_events_df",
                "want_out": self.events_df,
                "want_fn": "_filter_col_inclusion",
                "want_fn_arg": (self.events_df, {"subject_id": [4, 5, 6]}),
            },
            {
                "msg": "`held_out_events_df` should filter to the 'held_out' split.",
                "attr": "held_out_events_df",
                "want_out": self.events_df,
                "want_fn": "_filter_col_inclusion",
                "want_fn_arg": (self.events_df, {"subject_id": [7, 8, 9, 10]}),
            },
        ]

        for C in cases:
            with self.subTest(msg=C["msg"]):
                self.E._reset_functions_called()
                self.assertEqual(C["want_out"], getattr(self.E, C["attr"]))
                self.assertNestedDictEqual({C["want_fn"]: [C["want_fn_arg"]]}, self.E.functions_called)

    def test_get_source_df(self):
        dynamic = MeasurementConfig(
            temporality=TemporalityType.DYNAMIC,
            modality=DataModality.MULTI_LABEL_CLASSIFICATION,
        )
        static = MeasurementConfig(temporality=TemporalityType.STATIC, modality="single_label_classification")
        time_dependent = MeasurementConfig(
            temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
            functor=AgeFunctor,
        )

        train_subject_ids = [1, 2, 3]
        self.E.split_subjects = {"train": train_subject_ids}

        cases = [
            {
                "msg": (
                    "Should filter to the appropriate split and return the measurements df "
                    "when passed a dynamic measurement."
                ),
                "config": dynamic,
                "do_only_train": True,
                "want_attr": "dynamic_measurements_df",
                "want_df": self.dynamic_measurements_df,
                "want_id": "measurement_id",
                "want_fn": "_filter_col_inclusion",
                "want_fn_arg": [
                    (self.events_df, {"subject_id": [1, 2, 3]}),
                    (self.dynamic_measurements_df, {"event_id": [1, 2]}),
                ],
            },
            {
                "msg": "Should filter to train and return subjects_df when passed a static measurement",
                "config": static,
                "do_only_train": True,
                "want_attr": "subjects_df",
                "want_df": self.subjects_df,
                "want_id": "subject_id",
                "want_fn": "_filter_col_inclusion",
                "want_fn_arg": (self.subjects_df, {"subject_id": [1, 2, 3]}),
            },
            {
                "msg": "Should return subjects_df when passed a static measurement and do_only_train = False",
                "config": static,
                "do_only_train": False,
                "want_attr": "subjects_df",
                "want_df": self.subjects_df,
                "want_id": "subject_id",
                "want_fn": None,
            },
            {
                "msg": "Should filter to train and return events_df when passed a time dependent measure",
                "config": time_dependent,
                "do_only_train": True,
                "want_attr": "events_df",
                "want_df": self.events_df,
                "want_id": "event_id",
                "want_fn": "_filter_col_inclusion",
                "want_fn_arg": (self.events_df, {"subject_id": [1, 2, 3]}),
            },
            {
                "msg": "Should return events_df when passed a time dependent measure & do_only_train = False",
                "config": time_dependent,
                "do_only_train": False,
                "want_attr": "events_df",
                "want_df": self.events_df,
                "want_id": "event_id",
                "want_fn": None,
            },
        ]

        for C in cases:
            with self.subTest(C["msg"]):
                self.E._reset_functions_called()
                got_attr, got_id, got_df = self.E._get_source_df(C["config"], C["do_only_train"])
                self.assertEqual(C["want_attr"], got_attr)
                self.assertEqual(C["want_id"], got_id)
                self.assertEqual(C["want_df"], got_df)
                if C["want_fn"] is None:
                    self.assertEqual({}, self.E.functions_called)
                elif type(C["want_fn_arg"]) is list:
                    self.assertNestedDictEqual({C["want_fn"]: C["want_fn_arg"]}, self.E.functions_called)
                else:
                    self.assertNestedDictEqual({C["want_fn"]: [C["want_fn_arg"]]}, self.E.functions_called)

    def test_preprocess(self):
        def fit_measurements(self, *args, **kwargs):
            self.functions_called["fit_measurements"].append((args, kwargs))

        def transform_measurements(self, *args, **kwargs):
            self.functions_called["transform_measurements"].append((args, kwargs))

        self.E.fit_measurements = fit_measurements.__get__(self.E)
        self.E.transform_measurements = transform_measurements.__get__(self.E)

        self.E._reset_functions_called()
        self.E.preprocess()

        want_functions_called = {
            "_add_time_dependent_measurements": [()],
            "fit_measurements": [((), {})],
            "transform_measurements": [((), {})],
        }

        self.assertNestedDictEqual(want_functions_called, self.E.functions_called)

    def test_fit_measurements(self):
        mock_source_df = {
            "name": "dynamic_measurements",
            "retained": [1],
            "numeric": [1],
            "extra_to_pad_len": [1],
        }

        def get_source_df(self, *args, **kwargs):
            self.functions_called["_get_source_df"].append(copy.deepcopy((args, kwargs)))
            return None, None, mock_source_df

        base_measurement_config_kwargs = {
            "temporality": TemporalityType.DYNAMIC,
            "modality": DataModality.MULTI_LABEL_CLASSIFICATION,
        }

        retained_config = MeasurementConfig(**base_measurement_config_kwargs)
        numeric_config = MeasurementConfig(
            **{
                **base_measurement_config_kwargs,
                "modality": DataModality.MULTIVARIATE_REGRESSION,
                "values_column": "value",
            }
        )

        self.config = DatasetConfig(
            min_valid_vocab_element_observations=2,
            min_valid_column_observations=3,
            measurement_configs={
                "retained": retained_config,
                "not_present": MeasurementConfig(**base_measurement_config_kwargs),
                "dropped": MeasurementConfig(
                    **{**base_measurement_config_kwargs, "modality": DataModality.DROPPED}
                ),
                "numeric": numeric_config,
            },
        )

        self.E = ESDMock(self.config, self.subjects_df, self.events_df, self.dynamic_measurements_df)
        self.E._get_source_df = get_source_df.__get__(self.E)

        self.assertFalse(self.E._is_fit)

        self.E._reset_functions_called()
        self.E.fit_measurements()

        self.assertTrue(self.E._is_fit)

        empty_measurement_metadata = pd.DataFrame(
            {
                "value_type": pd.Series([], dtype=object),
                "outlier_model": pd.Series([], dtype=object),
                "normalizer": pd.Series([], dtype=object),
            },
            index=pd.Index([], name="numeric"),
        )

        partial_retained_config_init = MeasurementConfig(
            **{
                **base_measurement_config_kwargs,
                "name": "retained",
            }
        )
        partial_numeric_config_init = MeasurementConfig(
            **{
                **base_measurement_config_kwargs,
                "name": "numeric",
                "values_column": "value",
                "modality": DataModality.MULTIVARIATE_REGRESSION,
            }
        )

        partial_retained_config = MeasurementConfig(
            **{
                **base_measurement_config_kwargs,
                "observation_frequency": 1.0,
                "name": "retained",
            }
        )
        partial_numeric_config = MeasurementConfig(
            **{
                **base_measurement_config_kwargs,
                "name": "numeric",
                "values_column": "value",
                "modality": DataModality.MULTIVARIATE_REGRESSION,
                "_measurement_metadata": empty_measurement_metadata,
                "observation_frequency": 1.0,
            }
        )
        want_functions_called = {
            "_get_source_df": [
                ((self.config.measurement_configs["retained"],), {"do_only_train": True}),
                ((self.config.measurement_configs["not_present"],), {"do_only_train": True}),
                ((self.config.measurement_configs["numeric"],), {"do_only_train": True}),
            ],
            "_filter_col_inclusion": [
                (mock_source_df, {"retained": True}),
                (mock_source_df, {"numeric": True}),
            ],
            "_fit_measurement_metadata": [
                (
                    "numeric",
                    MeasurementConfig(
                        **{
                            **base_measurement_config_kwargs,
                            "modality": DataModality.MULTIVARIATE_REGRESSION,
                            "values_column": "value",
                            "observation_frequency": 1.0,
                            "name": "numeric",
                            "_measurement_metadata": empty_measurement_metadata,
                        }
                    ),
                    mock_source_df,
                )
            ],
            "_fit_vocabulary": [
                ("retained", partial_retained_config, mock_source_df),
                ("numeric", partial_numeric_config, mock_source_df),
            ],
            "_total_possible_and_observed": [
                ("retained", partial_retained_config_init, mock_source_df),
                ("numeric", partial_numeric_config_init, mock_source_df),
            ],
        }
        self.assertNestedDictEqual(want_functions_called, self.E.functions_called)

        mock_vocab = Vocabulary(["UNK", "foo"], [1 / 4, 3 / 4])
        want_retained_config = MeasurementConfig(
            **{
                **base_measurement_config_kwargs,
                "vocabulary": mock_vocab,
                "observation_frequency": 1.0,
                "name": "retained",
            }
        )
        want_numeric_config = MeasurementConfig(
            **{
                **base_measurement_config_kwargs,
                "name": "numeric",
                "values_column": "value",
                "modality": DataModality.MULTIVARIATE_REGRESSION,
                "_measurement_metadata": empty_measurement_metadata,
                "vocabulary": mock_vocab,
                "observation_frequency": 1.0,
            }
        )
        want_inferred_measurement_configs = {
            "retained": want_retained_config,
            "not_present": MeasurementConfig(
                **{
                    **base_measurement_config_kwargs,
                    "name": "not_present",
                    "modality": DataModality.DROPPED,
                }
            ),
            "numeric": want_numeric_config,
        }

        self.assertNestedDictEqual(want_inferred_measurement_configs, self.E.inferred_measurement_configs)

    @unittest.skip("TODO: Implement this test!")
    def test_transform_measurements(self):
        raise NotImplementedError()


if __name__ == "__main__":
    unittest.main()

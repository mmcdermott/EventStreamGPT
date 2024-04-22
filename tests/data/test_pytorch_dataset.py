import sys

sys.path.append("../..")

import copy
import json
import unittest
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import torch

from EventStream.data.config import (
    MeasurementConfig,
    PytorchDatasetConfig,
    VocabularyConfig,
)
from EventStream.data.pytorch_dataset import ConstructorPytorchDataset as PytorchDataset
from EventStream.data.types import PytorchBatch

from ..utils import MLTypeEqualityCheckableMixin

MEASUREMENTS_IDXMAP = {
    "event_type": 1,
    "static1": 2,
    "multi_label_classification": 3,
    "single_label_classification": 4,
    "univariate_regression": 5,
    "multivariate_regression": 6,
    "static2": 7,
}

UNIFIED_VOCABULARY_IDXMAP = {
    "event_type": {"ET1": 1, "ET2": 2},
    "static1": {"UNK": 3, "foo": 4, "bar": 5},
    "multi_label_classification": {"UNK": 6, "k1": 7, "k2": 8, "k3": 9, "k4": 10},
    "single_label_classification": {"UNK": 11, "y1": 12, "y2": 13},
    "univariate_regression": {"univariate_regression": 14},
    "multivariate_regression": {"UNK": 15, "m1": 16, "m2": 17},
    "static2": {"UNK": 18, "V1": 19, "V2": 20, "V3": 21},
}

start_times = [
    datetime(1990, 1, 1),
    datetime(1992, 1, 1),
    datetime(1994, 1, 1),
    datetime(1991, 1, 1),
    datetime(1993, 1, 1),
]
subj_1_event_times = [
    (t - start_times[0]) / timedelta(minutes=1)
    for t in [
        datetime(2000, 1, 1),
        datetime(2000, 1, 2),
        datetime(2000, 1, 3),
        datetime(2000, 2, 1),
    ]
]
subj_2_event_times = [
    (t - start_times[1]) / timedelta(minutes=1)
    for t in [
        datetime(1995, 1, 1),
        datetime(2000, 1, 2),
    ]
]
subj_3_event_times = [
    (t - start_times[2]) / timedelta(minutes=1)
    for t in [
        datetime(2001, 1, 1, 12),
        datetime(2001, 1, 1, 13),
        datetime(2001, 1, 1, 14),
    ]
]

DL_REP_DF = pl.DataFrame(
    {
        "subject_id": [1, 2, 3, 4, 5],
        "start_time": start_times,
        "time": [subj_1_event_times, subj_2_event_times, subj_3_event_times, None, None],
        # 'static': ['foo', 'foo', 'bar', 'bar', 'bar'],
        "static_indices": [
            [
                UNIFIED_VOCABULARY_IDXMAP["static1"]["foo"],
                UNIFIED_VOCABULARY_IDXMAP["static2"]["V3"],
            ],
            [
                UNIFIED_VOCABULARY_IDXMAP["static2"]["V1"],
                UNIFIED_VOCABULARY_IDXMAP["static1"]["bar"],
            ],
            [],
            [],
            [UNIFIED_VOCABULARY_IDXMAP["static1"]["UNK"]],
        ],
        "static_measurement_indices": [
            [MEASUREMENTS_IDXMAP["static1"], MEASUREMENTS_IDXMAP["static2"]],
            [MEASUREMENTS_IDXMAP["static2"], MEASUREMENTS_IDXMAP["static1"]],
            [],
            [],
            [MEASUREMENTS_IDXMAP["static1"]],
        ],
        "dynamic_indices": [
            [
                [
                    UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET1"],
                    UNIFIED_VOCABULARY_IDXMAP["single_label_classification"]["UNK"],
                    UNIFIED_VOCABULARY_IDXMAP["multi_label_classification"]["k1"],
                    UNIFIED_VOCABULARY_IDXMAP["multi_label_classification"]["k4"],
                ],
                [
                    UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET2"],
                    UNIFIED_VOCABULARY_IDXMAP["univariate_regression"]["univariate_regression"],
                    UNIFIED_VOCABULARY_IDXMAP["multivariate_regression"]["m1"],
                    UNIFIED_VOCABULARY_IDXMAP["multivariate_regression"]["m2"],
                ],
                [
                    UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET2"],
                    UNIFIED_VOCABULARY_IDXMAP["multivariate_regression"]["m1"],
                ],
                [UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET2"]],
            ],
            [
                [UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET2"]],
                [
                    UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET2"],
                    UNIFIED_VOCABULARY_IDXMAP["univariate_regression"]["univariate_regression"],
                ],
            ],
            [
                [UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET1"]],
                [
                    UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET1"],
                    UNIFIED_VOCABULARY_IDXMAP["single_label_classification"]["UNK"],
                ],
                [UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET1"]],
            ],
            None,
            None,
        ],
        "dynamic_measurement_indices": [
            [
                [
                    MEASUREMENTS_IDXMAP["event_type"],
                    MEASUREMENTS_IDXMAP["single_label_classification"],
                    MEASUREMENTS_IDXMAP["multi_label_classification"],
                    MEASUREMENTS_IDXMAP["multi_label_classification"],
                ],
                [
                    MEASUREMENTS_IDXMAP["event_type"],
                    MEASUREMENTS_IDXMAP["univariate_regression"],
                    MEASUREMENTS_IDXMAP["multivariate_regression"],
                    MEASUREMENTS_IDXMAP["multivariate_regression"],
                ],
                [
                    MEASUREMENTS_IDXMAP["event_type"],
                    MEASUREMENTS_IDXMAP["multivariate_regression"],
                ],
                [MEASUREMENTS_IDXMAP["event_type"]],
            ],
            [
                [MEASUREMENTS_IDXMAP["event_type"]],
                [MEASUREMENTS_IDXMAP["event_type"], MEASUREMENTS_IDXMAP["univariate_regression"]],
            ],
            [
                [MEASUREMENTS_IDXMAP["event_type"]],
                [
                    MEASUREMENTS_IDXMAP["event_type"],
                    MEASUREMENTS_IDXMAP["single_label_classification"],
                ],
                [MEASUREMENTS_IDXMAP["event_type"]],
            ],
            None,
            None,
        ],
        "dynamic_values": [
            [[None, None, None, None], [None, 0.1, 0.3, 1.2], [None, np.NaN], [None]],
            [[None], [None, 0.2]],
            [[None], [None, None], [None]],
            None,
            None,
        ],
    },
    schema={
        "subject_id": pl.UInt8,
        "start_time": pl.Datetime,
        "time": pl.List(pl.Float64),
        "static_indices": pl.List(pl.UInt64),
        "static_measurement_indices": pl.List(pl.UInt64),
        "dynamic_indices": pl.List(pl.List(pl.UInt64)),
        "dynamic_measurement_indices": pl.List(pl.List(pl.UInt64)),
        "dynamic_values": pl.List(pl.List(pl.Float64)),
    },
)

WANT_SUBJ_1_UNCUT = {
    "time_delta": list(np.diff(subj_1_event_times)) + [1],
    "static_indices": [
        UNIFIED_VOCABULARY_IDXMAP["static1"]["foo"],
        UNIFIED_VOCABULARY_IDXMAP["static2"]["V3"],
    ],
    "static_measurement_indices": [MEASUREMENTS_IDXMAP["static1"], MEASUREMENTS_IDXMAP["static2"]],
    "dynamic_indices": [
        [
            UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET1"],
            UNIFIED_VOCABULARY_IDXMAP["single_label_classification"]["UNK"],
            UNIFIED_VOCABULARY_IDXMAP["multi_label_classification"]["k1"],
            UNIFIED_VOCABULARY_IDXMAP["multi_label_classification"]["k4"],
        ],
        [
            UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET2"],
            UNIFIED_VOCABULARY_IDXMAP["univariate_regression"]["univariate_regression"],
            UNIFIED_VOCABULARY_IDXMAP["multivariate_regression"]["m1"],
            UNIFIED_VOCABULARY_IDXMAP["multivariate_regression"]["m2"],
        ],
        [
            UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET2"],
            UNIFIED_VOCABULARY_IDXMAP["multivariate_regression"]["m1"],
        ],
        [UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET2"]],
    ],
    "dynamic_measurement_indices": [
        [
            MEASUREMENTS_IDXMAP["event_type"],
            MEASUREMENTS_IDXMAP["single_label_classification"],
            MEASUREMENTS_IDXMAP["multi_label_classification"],
            MEASUREMENTS_IDXMAP["multi_label_classification"],
        ],
        [
            MEASUREMENTS_IDXMAP["event_type"],
            MEASUREMENTS_IDXMAP["univariate_regression"],
            MEASUREMENTS_IDXMAP["multivariate_regression"],
            MEASUREMENTS_IDXMAP["multivariate_regression"],
        ],
        [MEASUREMENTS_IDXMAP["event_type"], MEASUREMENTS_IDXMAP["multivariate_regression"]],
        [MEASUREMENTS_IDXMAP["event_type"]],
    ],
    "dynamic_values": [[None, None, None, None], [None, 0.1, 0.3, 1.2], [None, np.NaN], [None]],
}

WANT_SUBJ_2_UNCUT = {
    "time_delta": list(np.diff(subj_2_event_times)) + [1],
    "static_indices": [
        UNIFIED_VOCABULARY_IDXMAP["static2"]["V1"],
        UNIFIED_VOCABULARY_IDXMAP["static1"]["bar"],
    ],
    "static_measurement_indices": [MEASUREMENTS_IDXMAP["static2"], MEASUREMENTS_IDXMAP["static1"]],
    "dynamic_indices": [
        [UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET2"]],
        [
            UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET2"],
            UNIFIED_VOCABULARY_IDXMAP["univariate_regression"]["univariate_regression"],
        ],
    ],
    "dynamic_measurement_indices": [
        [MEASUREMENTS_IDXMAP["event_type"]],
        [MEASUREMENTS_IDXMAP["event_type"], MEASUREMENTS_IDXMAP["univariate_regression"]],
    ],
    "dynamic_values": [[None], [None, 0.2]],
}

WANT_SUBJ_3_UNCUT = {
    "time_delta": list(np.diff(subj_3_event_times)) + [1],
    "static_indices": [],
    "static_measurement_indices": [],
    "dynamic_indices": [
        [UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET1"]],
        [
            UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET1"],
            UNIFIED_VOCABULARY_IDXMAP["single_label_classification"]["UNK"],
        ],
        [UNIFIED_VOCABULARY_IDXMAP["event_type"]["ET1"]],
    ],
    "dynamic_measurement_indices": [
        [MEASUREMENTS_IDXMAP["event_type"]],
        [MEASUREMENTS_IDXMAP["event_type"], MEASUREMENTS_IDXMAP["single_label_classification"]],
        [MEASUREMENTS_IDXMAP["event_type"]],
    ],
    "dynamic_values": [[None], [None, None], [None]],
}

TASK_DF = pl.DataFrame(
    {
        "subject_id": pl.Series([1, 3, 4], dtype=pl.UInt8),
        "start_time": [
            datetime(2000, 1, 1),
            datetime(2001, 1, 1, 12, 30),
            datetime(1995, 1, 1),
        ],
        "end_time": [
            datetime(2000, 1, 3),
            datetime(2001, 1, 1, 14, 30),
            datetime(2000, 1, 3),
        ],
        "binary": [True, False, True],
        "multi_class_int": [0, 1, 2],
        "multi_class_cat": pl.Series(["a", "a", "b"], dtype=pl.Categorical),
        "regression": [1.2, 3.2, 1.5],
    }
)


def get_seeded_start_index(seed, curr_len, max_seq_len):
    np.random.seed(seed)
    return np.random.choice(curr_len - max_seq_len)


class TestPytorchDataset(MLTypeEqualityCheckableMixin, unittest.TestCase):
    def get_pyd(
        self,
        split: str = "fake_split",
        task_df: pl.DataFrame | None = None,
        task_df_name: str = "fake_task",
        vocabulary_config: VocabularyConfig = VocabularyConfig(),
        measurement_configs: dict[str, MeasurementConfig] | None = None,
        **config_kwargs,
    ):
        with TemporaryDirectory() as d:
            save_dir = Path(d)

            DL_fp = save_dir / "DL_reps" / f"{split}.parquet"
            DL_fp.parent.mkdir(parents=True, exist_ok=True)
            DL_REP_DF.write_parquet(DL_fp)

            config_kwargs = {"save_dir": save_dir, **config_kwargs}
            if task_df is not None:
                config_kwargs["task_df_name"] = task_df_name

                raw_task_df_fp = save_dir / "task_dfs" / f"{task_df_name}.parquet"
                raw_task_df_fp.parent.mkdir(parents=True, exist_ok=True)
                task_df.write_parquet(raw_task_df_fp)

            vocabulary_config.to_json_file(save_dir / "vocabulary_config.json")

            if measurement_configs is None:
                measurement_configs = {}

            inferred_measurement_config_fp = save_dir / "inferred_measurement_configs.json"
            with open(inferred_measurement_config_fp, mode="w") as f:
                json.dump({k: v.to_dict() for k, v in measurement_configs.items()}, f)

            config = PytorchDatasetConfig(**config_kwargs)

            pyd = PytorchDataset(config=config, split=split)
        return config, pyd

    def test_normalize_task(self):
        cases = [
            {
                "msg": "Should flag Integer values as multi-class.",
                "vals": pl.Series([1, 2, 1, 4], dtype=pl.UInt8),
                "want_type": "multi_class_classification",
                "want_vals": pl.Series([1, 2, 1, 4], dtype=pl.UInt8),
            },
            {
                "msg": "Should flag Categorical values as multi-class and normalize to integers.",
                "vals": pl.Series(["a", "b", "a", "z"], dtype=pl.Categorical),
                "want_type": "multi_class_classification",
                "want_vals": pl.Series([0, 1, 0, 2], dtype=pl.UInt32),
            },
            {
                "msg": "Should flag Boolean values as binary and normalize to float.",
                "vals": pl.Series([True, False, True, False]),
                "want_type": "binary_classification",
                "want_vals": pl.Series([1.0, 0.0, 1.0, 0.0], dtype=pl.Float32),
            },
            {
                "msg": "Should flag Float values as regression.",
                "vals": pl.Series([1.0, 2.1, 1.3, 4.1]),
                "want_type": "regression",
                "want_vals": pl.Series([1.0, 2.1, 1.3, 4.1]),
            },
            {
                "msg": "Should raise TypeError on object type.",
                "vals": pl.Series(["fuzz", 3, float("nan")], dtype=pl.Object),
                "want_raise": TypeError,
            },
        ]

        for C in cases:
            with self.subTest(C["msg"]):
                if C.get("want_raise", None) is not None:
                    with self.assertRaises(C["want_raise"]):
                        PytorchDataset.normalize_task(pl.col("c"), C["vals"].dtype)
                else:
                    got_type, got_normalizer = PytorchDataset.normalize_task(pl.col("c"), C["vals"].dtype)
                    self.assertEqual(C["want_type"], got_type)

                    got_vals = pl.DataFrame({"c": C["vals"]}).select(got_normalizer).get_column("c")
                    want_vals = pl.DataFrame({"c": C["want_vals"]}).get_column("c")

                    self.assertEqual(want_vals.to_pandas(), got_vals.to_pandas())

    def test_get_item_should_collate(self):
        _, pyd = self.get_pyd(max_seq_len=4, min_seq_len=2)

        items = [pyd._seeded_getitem(i, seed=1) for i in range(3)]
        pyd.collate(items)

    def test_get_item(self):
        cases = [
            {
                "msg": "Should not cut sequences when not necessary.",
                "max_seq_len": 4,
                "min_seq_len": 2,
                "want_items": [WANT_SUBJ_1_UNCUT, WANT_SUBJ_2_UNCUT, WANT_SUBJ_3_UNCUT],
            },
            {
                "msg": "Should cut sequences to max sequence length.",
                "max_seq_len": 3,
                "min_seq_len": 2,
                "want_items": [WANT_SUBJ_1_UNCUT, WANT_SUBJ_2_UNCUT, WANT_SUBJ_3_UNCUT],
                "want_start_idx": [get_seeded_start_index(1, 4, 3), 0, 0],
            },
            {
                "msg": "Should drop sequences that are too short.",
                "max_seq_len": 4,
                "min_seq_len": 3,
                "want_items": [WANT_SUBJ_1_UNCUT, WANT_SUBJ_3_UNCUT],
            },
            {
                "msg": "Should re-set cached data based on task df",
                "max_seq_len": 4,
                "min_seq_len": 2,
                "task_df": TASK_DF,
                "want_items": [
                    {
                        "binary": True,
                        "multi_class_int": 0,
                        "multi_class_cat": 0,
                        "regression": 1.2,
                        **WANT_SUBJ_1_UNCUT,
                        "time_delta": [
                            t if i < (2 - 1) else 1 for i, t in enumerate(WANT_SUBJ_1_UNCUT["time_delta"])
                        ],
                    },
                    {
                        "binary": False,
                        "multi_class_int": 1,
                        "multi_class_cat": 0,
                        "regression": 3.2,
                        **WANT_SUBJ_3_UNCUT,
                        "time_delta": [
                            t if i < (3 - 1) else 1 for i, t in enumerate(WANT_SUBJ_3_UNCUT["time_delta"])
                        ],
                    },
                ],
                "want_start_idx": [0, 1],
                "want_end_idx": [2, 3],
            },
        ]
        time_dep_cols = [
            "time_delta",
            "dynamic_indices",
            "dynamic_values",
            "dynamic_measurement_indices",
        ]

        for C in cases:
            get_pyd_kwargs = {"max_seq_len": C["max_seq_len"], "min_seq_len": C["min_seq_len"]}
            if "task_df" in C:
                get_pyd_kwargs.update({"task_df": C["task_df"]})

            with self.subTest(C["msg"]):
                config, pyd = self.get_pyd(**get_pyd_kwargs)

                self.assertEqual(len(C["want_items"]), len(pyd))

                for i, it in enumerate(C["want_items"]):
                    it = copy.deepcopy(it)
                    st = C["want_start_idx"][i] if "want_start_idx" in C else 0
                    end = C["want_end_idx"][i] if "want_end_idx" in C else st + C["max_seq_len"]

                    want_it = {}
                    for k, v in it.items():
                        want_it[k] = v[st:end] if k in time_dep_cols else v

                    got_it = pyd._seeded_getitem(i, seed=1)

                    self.assertNestedDictEqual(
                        want_it, got_it, msg=f"Item {i} does not match:\n{want_it}\n{got_it}."
                    )

    def test_dynamic_collate_fn(self):
        """collate_fn should appropriately combine two batches of ragged tensors."""
        config, pyd = self.get_pyd(seq_padding_side="right", max_seq_len=10)
        pyd.do_produce_static_data = False

        subj_1 = {
            "time_delta": [0.0, 24 * 60.0, 2 * 24 * 60.0, 3 * 24 * 60.0],
            "dynamic_indices": [
                [1, 4],
                [2, 7, 7, 7, 8, 8],
                [1, 5],
                [1, 4],
            ],
            "dynamic_values": [
                [np.NaN, np.NaN],
                [np.NaN, 1, 2, 3, 4, 5],
                [np.NaN, np.NaN],
                [np.NaN, np.NaN],
            ],
            "dynamic_measurement_indices": [
                [1, 2],
                [1, 3, 3, 3, 3, 3],
                [1, 2],
                [1, 2],
            ],
        }
        subj_2 = {
            "time_delta": [0.0, 5, 10],
            "dynamic_indices": [
                [1, 4, 3],
                [2, 7, 7, 7],
                [1, 5],
            ],
            "dynamic_values": [
                [np.NaN, np.NaN, np.NaN],
                [np.NaN, 8, 9, 10],
                [np.NaN, np.NaN],
            ],
            "dynamic_measurement_indices": [
                [1, 2, 2],
                [1, 3, 3, 3],
                [1, 2],
            ],
        }

        batches = [subj_1, subj_2]
        out = pyd.collate(batches)

        want_out = PytorchBatch(
            **{
                "event_mask": torch.BoolTensor([[True, True, True, True], [True, True, True, False]]),
                "dynamic_values_mask": torch.BoolTensor(
                    [
                        [
                            [False, False, False, False, False, False],
                            [False, True, True, True, True, True],
                            [False, False, False, False, False, False],
                            [False, False, False, False, False, False],
                        ],
                        [
                            [False, False, False, False, False, False],
                            [False, True, True, True, False, False],
                            [False, False, False, False, False, False],
                            [False, False, False, False, False, False],
                        ],
                    ]
                ),
                "time_delta": torch.Tensor([[0.0, 24 * 60.0, 2 * 24 * 60.0, 3 * 24 * 60.0], [0, 5, 10, 0]]),
                "dynamic_indices": torch.LongTensor(
                    [
                        [
                            [1, 4, 0, 0, 0, 0],
                            [2, 7, 7, 7, 8, 8],
                            [1, 5, 0, 0, 0, 0],
                            [1, 4, 0, 0, 0, 0],
                        ],
                        [
                            [1, 4, 3, 0, 0, 0],
                            [2, 7, 7, 7, 0, 0],
                            [1, 5, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ],
                    ]
                ),
                "dynamic_measurement_indices": torch.LongTensor(
                    [
                        [
                            [1, 2, 0, 0, 0, 0],
                            [1, 3, 3, 3, 3, 3],
                            [1, 2, 0, 0, 0, 0],
                            [1, 2, 0, 0, 0, 0],
                        ],
                        [
                            [1, 2, 2, 0, 0, 0],
                            [1, 3, 3, 3, 0, 0],
                            [1, 2, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ],
                    ]
                ),
                "dynamic_values": torch.nan_to_num(
                    torch.Tensor(
                        [
                            [
                                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                                [np.NaN, 1, 2, 3, 4, 5],
                                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                            ],
                            [
                                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                                [np.NaN, 8, 9, 10, np.NaN, np.NaN],
                                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                            ],
                        ]
                    ),
                    0,
                ),
            }
        )

        self.assertNestedDictEqual(asdict(want_out), asdict(out))

        config, pyd = self.get_pyd(seq_padding_side="left", max_seq_len=10)
        pyd.do_produce_static_data = False

        out = pyd.collate(batches)

        want_out = PytorchBatch(
            **{
                "event_mask": torch.BoolTensor([[True, True, True, True], [False, True, True, True]]),
                "dynamic_values_mask": torch.BoolTensor(
                    [
                        [
                            [False, False, False, False, False, False],
                            [False, True, True, True, True, True],
                            [False, False, False, False, False, False],
                            [False, False, False, False, False, False],
                        ],
                        [
                            [False, False, False, False, False, False],
                            [False, False, False, False, False, False],
                            [False, True, True, True, False, False],
                            [False, False, False, False, False, False],
                        ],
                    ]
                ),
                "time_delta": torch.Tensor([[0.0, 24 * 60.0, 2 * 24 * 60.0, 3 * 24 * 60.0], [0, 0, 5, 10]]),
                "dynamic_indices": torch.LongTensor(
                    [
                        [
                            [1, 4, 0, 0, 0, 0],
                            [2, 7, 7, 7, 8, 8],
                            [1, 5, 0, 0, 0, 0],
                            [1, 4, 0, 0, 0, 0],
                        ],
                        [
                            [0, 0, 0, 0, 0, 0],
                            [1, 4, 3, 0, 0, 0],
                            [2, 7, 7, 7, 0, 0],
                            [1, 5, 0, 0, 0, 0],
                        ],
                    ]
                ),
                "dynamic_measurement_indices": torch.LongTensor(
                    [
                        [
                            [1, 2, 0, 0, 0, 0],
                            [1, 3, 3, 3, 3, 3],
                            [1, 2, 0, 0, 0, 0],
                            [1, 2, 0, 0, 0, 0],
                        ],
                        [
                            [0, 0, 0, 0, 0, 0],
                            [1, 2, 2, 0, 0, 0],
                            [1, 3, 3, 3, 0, 0],
                            [1, 2, 0, 0, 0, 0],
                        ],
                    ]
                ),
                "dynamic_values": torch.nan_to_num(
                    torch.Tensor(
                        [
                            [
                                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                                [np.NaN, 1, 2, 3, 4, 5],
                                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                            ],
                            [
                                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                                [np.NaN, 8, 9, 10, np.NaN, np.NaN],
                                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                            ],
                        ]
                    ),
                    0,
                ),
            }
        )

        self.assertNestedDictEqual(asdict(want_out), asdict(out))

    def test_collate_fn(self):
        config, pyd = self.get_pyd(max_seq_len=4)
        pyd.do_produce_static_data = True

        want_subj_event_ages = [
            [
                1.0,
                1 + 1 / 365 + 14 / (24 * 365),
                1 + 2 / 365 + 10 / (24 * 365),
                1 + 3 / 365 + 23 / (24 * 365),
            ],
            [2 + 15 / (24 * 365), 2 + 1 / 365 + 2 / (24 * 365)],
        ]
        subj_1 = {
            "time_delta": [0.0, (24 + 14) * 60.0, (2 * 24 + 10) * 60.0, (3 * 24 + 23) * 60.0],
            "static_indices": [16],
            "static_measurement_indices": [6],
            "dynamic_indices": [
                [1, 7, 9, 11],
                [2, 4, 4, 4, 5, 5, 9, 12],
                [1, 8, 9, 13],
                [1, 7, 9, 14],
            ],
            "dynamic_values": [
                [np.NaN, np.NaN, want_subj_event_ages[0][0], np.NaN],
                [np.NaN, 1.0, 2.0, 3.0, 4.0, 5.0, want_subj_event_ages[0][1], np.NaN],
                [np.NaN, np.NaN, want_subj_event_ages[0][2], np.NaN],
                [np.NaN, np.NaN, want_subj_event_ages[0][3], np.NaN],
            ],
            "dynamic_measurement_indices": [
                [1, 3, 4, 5],
                [1, 2, 2, 2, 2, 2, 4, 5],
                [1, 3, 4, 5],
                [1, 3, 4, 5],
            ],
        }
        subj_2 = {
            "time_delta": [0.0, 11 * 60.0],
            "static_indices": [17],
            "static_measurement_indices": [6],
            "dynamic_indices": [
                [1, 7, 9, 12],
                [2, 4, 5, 9, 11],
            ],
            "dynamic_values": [
                [np.NaN, np.NaN, want_subj_event_ages[1][0], np.NaN],
                [np.NaN, 1.0, 5.0, want_subj_event_ages[1][1], np.NaN],
            ],
            "dynamic_measurement_indices": [
                [1, 3, 4, 5],
                [1, 2, 2, 4, 5],
            ],
        }

        batches = [subj_1, subj_2]
        out = pyd.collate(batches)

        want_out = PytorchBatch(
            **{
                "event_mask": torch.BoolTensor([[True, True, True, True], [True, True, False, False]]),
                "dynamic_values_mask": torch.BoolTensor(
                    [
                        [
                            [False, False, True, False, False, False, False, False],
                            [False, True, True, True, True, True, True, False],
                            [False, False, True, False, False, False, False, False],
                            [False, False, True, False, False, False, False, False],
                        ],
                        [
                            [False, False, True, False, False, False, False, False],
                            [False, True, True, True, False, False, False, False],
                            [False, False, False, False, False, False, False, False],
                            [False, False, False, False, False, False, False, False],
                        ],
                    ]
                ),
                "time_delta": torch.Tensor(
                    [
                        [0.0, (24 + 14) * 60.0, (2 * 24 + 10) * 60.0, (3 * 24 + 23) * 60.0],
                        [0.0, 11 * 60.0, 0.0, 0.0],
                    ]
                ),
                "dynamic_indices": torch.LongTensor(
                    [
                        [
                            [1, 7, 9, 11, 0, 0, 0, 0],
                            [2, 4, 4, 4, 5, 5, 9, 12],
                            [1, 8, 9, 13, 0, 0, 0, 0],
                            [1, 7, 9, 14, 0, 0, 0, 0],
                        ],
                        [
                            [1, 7, 9, 12, 0, 0, 0, 0],
                            [2, 4, 5, 9, 11, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                    ]
                ),
                "dynamic_measurement_indices": torch.LongTensor(
                    [
                        [
                            [1, 3, 4, 5, 0, 0, 0, 0],
                            [1, 2, 2, 2, 2, 2, 4, 5],
                            [1, 3, 4, 5, 0, 0, 0, 0],
                            [1, 3, 4, 5, 0, 0, 0, 0],
                        ],
                        [
                            [1, 3, 4, 5, 0, 0, 0, 0],
                            [1, 2, 2, 4, 5, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                    ]
                ),
                "dynamic_values": torch.Tensor(
                    [
                        [
                            [0, 0, want_subj_event_ages[0][0], 0, 0, 0, 0, 0],
                            [0, 1.0, 2.0, 3.0, 4.0, 5.0, want_subj_event_ages[0][1], 0],
                            [0, 0, want_subj_event_ages[0][2], 0, 0, 0, 0, 0],
                            [0, 0, want_subj_event_ages[0][3], 0, 0, 0, 0, 0],
                        ],
                        [
                            [0, 0, want_subj_event_ages[1][0], 0, 0, 0, 0, 0],
                            [0, 1.0, 5.0, want_subj_event_ages[1][1], 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                    ]
                ),
                "static_indices": torch.LongTensor([[16], [17]]),
                "static_measurement_indices": torch.LongTensor([[6], [6]]),
            }
        )

        self.assertNestedDictEqual(asdict(want_out), asdict(out))


if __name__ == "__main__":
    unittest.main()

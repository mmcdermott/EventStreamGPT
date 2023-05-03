import sys

sys.path.append("../..")

import copy
import unittest
from dataclasses import asdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import torch

from EventStream.data.config import PytorchDatasetConfig, VocabularyConfig
from EventStream.data.pytorch_dataset import PytorchDataset
from EventStream.data.types import PytorchBatch

from ..mixins import MLTypeEqualityCheckableMixin

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
    "time": subj_1_event_times,
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
    "time": subj_2_event_times,
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
    "time": subj_3_event_times,
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
    "dynamic_values": [None, None, None],
}

TASK_DF = pd.DataFrame(
    {
        "subject_id": [1, 3, 4],
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
        "multi_class_cat": pd.Series(["a", "a", "b"], dtype="category"),
        "regression": [1.2, 3.2, 1.5],
    }
)


def get_seeded_start_index(seed, curr_len, max_seq_len):
    np.random.seed(seed)
    return np.random.choice(curr_len - max_seq_len)


class TestPytorchDataset(MLTypeEqualityCheckableMixin, unittest.TestCase):
    def test_normalize_task(self):
        cases = [
            {
                "msg": "Should flag Integer values as multi-class.",
                "vals": pd.Series([1, 2, 1, 4], dtype=int),
                "want_type": "multi_class_classification",
                "want_vals": pd.Series([1, 2, 1, 4], dtype=int),
            },
            {
                "msg": "Should flag Categorical values as multi-class and normalize to integers.",
                "vals": pd.Series(["a", "b", "a", "z"], dtype="category"),
                "want_type": "multi_class_classification",
                "want_vals": pd.Series([0, 1, 0, 2], dtype=int),
            },
            {
                "msg": "Should flag Boolean values as binary and normalize to float.",
                "vals": pd.Series([True, False, True, False], dtype=bool),
                "want_type": "binary_classification",
                "want_vals": pd.Series([1.0, 0.0, 1.0, 0.0]),
            },
            {
                "msg": "Should flag Float values as regression.",
                "vals": pd.Series([1.0, 2.1, 1.3, 4.1]),
                "want_type": "regression",
                "want_vals": pd.Series([1.0, 2.1, 1.3, 4.1]),
            },
            {
                "msg": "Should raise TypeError on object type.",
                "vals": pd.Series(["fuzz", 3, pd.to_datetime("12/2/22"), float("nan")]),
                "want_raise": TypeError,
            },
        ]

        for C in cases:
            with self.subTest(C["msg"]):
                if C.get("want_raise", None) is not None:
                    with self.assertRaises(C["want_raise"]):
                        PytorchDataset.normalize_task(C["vals"])
                else:
                    got_type, got_vals = PytorchDataset.normalize_task(C["vals"])
                    self.assertEqual(C["want_type"], got_type)
                    self.assertEqual(C["want_vals"], got_vals)

    def test_get_item_should_collate(self):
        config = PytorchDatasetConfig(
            max_seq_len=4,
            min_seq_len=2,
        )
        pyd_kwargs = {"data": DL_REP_DF, "config": config, "vocabulary_config": VocabularyConfig()}
        pyd = PytorchDataset(**pyd_kwargs)

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
                    },
                    {
                        "binary": False,
                        "multi_class_int": 1,
                        "multi_class_cat": 0,
                        "regression": 3.2,
                        **WANT_SUBJ_3_UNCUT,
                    },
                ],
                "want_start_idx": [0, 1],
                "want_end_idx": [2, 3],
            },
        ]
        time_dep_cols = [
            "time",
            "dynamic_indices",
            "dynamic_values",
            "dynamic_measurement_indices",
        ]

        for C in cases:
            config = PytorchDatasetConfig(
                max_seq_len=C["max_seq_len"],
                min_seq_len=C["min_seq_len"],
            )
            pyd_kwargs = {
                "data": DL_REP_DF,
                "config": config,
                "vocabulary_config": VocabularyConfig(),
            }
            if "task_df" in C:
                pyd_kwargs.update({"task_df": C["task_df"]})

            with self.subTest(C["msg"]):
                pyd = PytorchDataset(**pyd_kwargs)

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
        config = PytorchDatasetConfig(seq_padding_side="right", max_seq_len=10)
        pyd = PytorchDataset(data=DL_REP_DF, config=config, vocabulary_config=VocabularyConfig())
        pyd.do_produce_static_data = False

        subj_1 = {
            "time": [0.0, 24 * 60.0, 2 * 24 * 60.0, 3 * 24 * 60.0],
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
            "time": [0.0, 5, 10],
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
                "event_mask": torch.BoolTensor(
                    [[True, True, True, True], [True, True, True, False]]
                ),
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
                "time": torch.Tensor(
                    [[0.0, 24 * 60.0, 2 * 24 * 60.0, 3 * 24 * 60.0], [0, 5, 10, 0]]
                ),
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

        config = PytorchDatasetConfig(seq_padding_side="left", max_seq_len=10)

        pyd = PytorchDataset(data=DL_REP_DF, config=config, vocabulary_config=VocabularyConfig())
        pyd.do_produce_static_data = False

        out = pyd.collate(batches)

        want_out = PytorchBatch(
            **{
                "event_mask": torch.BoolTensor(
                    [[True, True, True, True], [False, True, True, True]]
                ),
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
                "time": torch.Tensor(
                    [[0.0, 24 * 60.0, 2 * 24 * 60.0, 3 * 24 * 60.0], [0, 0, 5, 10]]
                ),
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
        config = PytorchDatasetConfig(max_seq_len=4)

        pyd = PytorchDataset(data=DL_REP_DF, config=config, vocabulary_config=VocabularyConfig())
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
            "time": [0.0, (24 + 14) * 60.0, (2 * 24 + 10) * 60.0, (3 * 24 + 23) * 60.0],
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
            "time": [0.0, 11 * 60.0],
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
                "event_mask": torch.BoolTensor(
                    [[True, True, True, True], [True, True, False, False]]
                ),
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
                "time": torch.Tensor(
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

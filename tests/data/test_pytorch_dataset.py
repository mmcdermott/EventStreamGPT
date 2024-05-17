import sys

sys.path.append("../..")

import json
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from EventStream.data.config import PytorchDatasetConfig, VocabularyConfig
from EventStream.data.pytorch_dataset import PytorchDataset

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
subj_1_event_time_deltas = [
    subj_1_event_times[i] - subj_1_event_times[i - 1] for i in range(1, len(subj_1_event_times))
] + [float("nan")]
subj_2_event_times = [
    (t - start_times[1]) / timedelta(minutes=1)
    for t in [
        datetime(1995, 1, 1),
        datetime(2000, 1, 2),
    ]
]
subj_2_event_time_deltas = [
    subj_2_event_times[i] - subj_2_event_times[i - 1] for i in range(1, len(subj_2_event_times))
] + [float("nan")]
subj_3_event_times = [
    (t - start_times[2]) / timedelta(minutes=1)
    for t in [
        datetime(2001, 1, 1, 12),
        datetime(2001, 1, 1, 13),
        datetime(2001, 1, 1, 14),
    ]
]
subj_3_event_time_deltas = [
    subj_3_event_times[i] - subj_3_event_times[i - 1] for i in range(1, len(subj_3_event_times))
] + [float("nan")]

DL_REP_DF = pl.DataFrame(
    {
        "subject_id": [1, 2, 3, 4, 5],
        "start_time": start_times,
        "time": [subj_1_event_times, subj_2_event_times, subj_3_event_times, [], []],
        "time_delta": [subj_1_event_time_deltas, subj_2_event_time_deltas, subj_3_event_time_deltas, [], []],
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
            [],
            [],
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
            [],
            [],
        ],
        "dynamic_values": [
            [[None, None, None, None], [None, 0.1, 0.3, 1.2], [None, np.NaN], [None]],
            [[None], [None, 0.2]],
            [[None], [None, None], [None]],
            [],
            [],
        ],
    },
    schema={
        "subject_id": pl.UInt8,
        "start_time": pl.Datetime,
        "time": pl.List(pl.Float64),
        "time_delta": pl.List(pl.Float32),
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
    def setUp(self):
        self.dir_obj = TemporaryDirectory()
        self.path = Path(self.dir_obj.name)

        self.split = "fake_split"

        shards_fp = self.path / "DL_shards.json"
        shards = {
            f"{self.split}/0": list(set(DL_REP_DF["subject_id"].to_list())),
        }
        shards_fp.write_text(json.dumps(shards))

        DL_fp = self.path / "DL_reps" / f"{self.split}/0.parquet"
        DL_fp.parent.mkdir(parents=True, exist_ok=True)
        DL_REP_DF.write_parquet(DL_fp)

        NRT_fp = self.path / "NRT_reps" / f"{self.split}/0.pt"
        NRT_fp.parent.mkdir(parents=True, exist_ok=True)

        jnrt_dict = {
            k: DL_REP_DF[k].to_list()
            for k in ["time_delta", "dynamic_indices", "dynamic_measurement_indices"]
        }
        jnrt_dict["dynamic_values"] = (
            DL_REP_DF["dynamic_values"]
            .list.eval(pl.element().list.eval(pl.element().fill_null(float("nan"))))
            .to_list()
        )
        jnrt_dict = JointNestedRaggedTensorDict(jnrt_dict)
        jnrt_dict.save(NRT_fp)

        self.valid_task_name = "fake_task"

        raw_task_df_fp = self.path / "task_dfs" / f"{self.valid_task_name}.parquet"
        raw_task_df_fp.parent.mkdir(parents=True, exist_ok=True)
        TASK_DF.write_parquet(raw_task_df_fp, use_pyarrow=True)

        VocabularyConfig().to_json_file(self.path / "vocabulary_config.json")

        measurement_configs = {}

        inferred_measurement_config_fp = self.path / "inferred_measurement_configs.json"
        with open(inferred_measurement_config_fp, mode="w") as f:
            json.dump({k: v.to_dict() for k, v in measurement_configs.items()}, f)

    def tearDown(self):
        self.dir_obj.cleanup()

    def get_pyd(
        self,
        task_df_name: str | None = None,
        **config_kwargs,
    ):
        config_kwargs = {"save_dir": self.path, **config_kwargs}
        if task_df_name is not None:
            config_kwargs["task_df_name"] = task_df_name

        config = PytorchDatasetConfig(**config_kwargs)
        pyd = PytorchDataset(config=config, split=self.split)
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

                    self.assertTrue(
                        (got_vals == want_vals).all(),
                        f"want_vals:\n{want_vals.to_pandas()}\ngot_vals:\n{got_vals.to_pandas()}",
                    )

    def test_get_item_should_collate(self):
        _, pyd = self.get_pyd(max_seq_len=4, min_seq_len=2)

        items = [pyd._seeded_getitem(i, seed=1) for i in range(3)]
        pyd.collate(items)


if __name__ == "__main__":
    unittest.main()

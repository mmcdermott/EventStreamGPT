import sys

sys.path.append("../..")

import unittest
from unittest.mock import MagicMock

import torch

from EventStream.data.data_embedding_layer import (
    DataEmbeddingLayer,
    MeasIndexGroupOptions,
    StaticEmbeddingMode,
)
from EventStream.data.types import PytorchBatch

from ..utils import MLTypeEqualityCheckableMixin


class TestDataEmbeddingLayer(MLTypeEqualityCheckableMixin, unittest.TestCase):
    """This code tests the `DataEmbeddingLayer` module, which embeds batches output by the `PytorchDataset`
    object's `collate` function."""

    def test_constructs(self):
        """This tests that the DataEmbeddingLayer can be constructed properly and expects the right types."""

        valid_params = {
            "n_total_embeddings": 4,
            "out_dim": 4,
            "static_embedding_mode": StaticEmbeddingMode.DROP,
        }

        cases = [
            {
                "msg": "`DataEmbeddingLayer` should construct with valid parameters.",
                "params": valid_params,
            },
            {
                "msg": "`DataEmbeddingLayer` should error if `n_total_embeddings` is not an int.",
                "params": {**valid_params, "n_total_embeddings": 4.5},
                "should_raise": TypeError,
            },
            {
                "msg": "`DataEmbeddingLayer` should error if `n_total_embeddings` is negative.",
                "params": {**valid_params, "n_total_embeddings": -1},
                "should_raise": ValueError,
            },
            {
                "msg": "`DataEmbeddingLayer` should error if `out_dim` is not an int.",
                "params": {**valid_params, "out_dim": None},
                "should_raise": TypeError,
            },
            {
                "msg": "`DataEmbeddingLayer` should error if `out_dim` is negative.",
                "params": {**valid_params, "out_dim": -1},
                "should_raise": ValueError,
            },
            {
                "msg": (
                    "`DataEmbeddingLayer` should construct if `categorical_embedding_dim` and "
                    "`numerical_embbeding_dim` are both positive integers."
                ),
                "params": {
                    **valid_params,
                    "categorical_embedding_dim": 2,
                    "numerical_embedding_dim": 2,
                },
            },
            {
                "msg": (
                    "`DataEmbeddingLayer` should error if `categorical_embedding_dim` is specified but "
                    "`numerical_embbeding_dim` is not."
                ),
                "params": {**valid_params, "categorical_embedding_dim": 2},
                "should_raise": ValueError,
            },
            {
                "msg": (
                    "`DataEmbeddingLayer` should error if `numerical_embedding_dim` is specified but "
                    "`categorical_embbeding_dim` is not."
                ),
                "params": {**valid_params, "numerical_embedding_dim": 2},
                "should_raise": ValueError,
            },
            {
                "msg": "`DataEmbeddingLayer` should error if `categorical_embedding_dim` is not an int.",
                "params": {
                    **valid_params,
                    "numerical_embedding_dim": 2,
                    "categorical_embedding_dim": 4.5,
                },
                "should_raise": TypeError,
            },
            {
                "msg": "`DataEmbeddingLayer` should error if `categorical_embedding_dim` is negative.",
                "params": {
                    **valid_params,
                    "categorical_embedding_dim": -1,
                    "numerical_embedding_dim": 2,
                },
                "should_raise": ValueError,
            },
            {
                "msg": "`DataEmbeddingLayer` should error if `numerical_embedding_dim` is not an int.",
                "params": {
                    **valid_params,
                    "numerical_embedding_dim": 4.5,
                    "categorical_embedding_dim": 2,
                },
                "should_raise": TypeError,
            },
            {
                "msg": "`DataEmbeddingLayer` should error if `numerical_embedding_dim` is negative.",
                "params": {
                    **valid_params,
                    "numerical_embedding_dim": -1,
                    "categorical_embedding_dim": 2,
                },
                "should_raise": ValueError,
            },
            {
                "msg": (
                    "`DataEmbeddingLayer` should error if `static_embedding_mode` is not a member of the "
                    "`StaticEmbeddingMode` enum."
                ),
                "params": {**valid_params, "static_embedding_mode": "not a member of the enum"},
                "should_raise": TypeError,
            },
            {
                "msg": "`DataEmbeddingLayer` should construct if `split_by_measurement_indices` is valid.",
                "params": {
                    **valid_params,
                    "split_by_measurement_indices": [
                        [1, (2, MeasIndexGroupOptions.CATEGORICAL_ONLY)],
                        [3, (4, MeasIndexGroupOptions.NUMERICAL_ONLY)],
                        [5, 6, 7],
                        [(7, MeasIndexGroupOptions.CATEGORICAL_AND_NUMERICAL)],
                    ],
                },
            },
            {
                "msg": (
                    "`DataEmbeddingLayer` should error if `split_by_measurement_indices` contains elements "
                    "not in the `MeasIndexGroupOptions` enum."
                ),
                "params": {
                    **valid_params,
                    "split_by_measurement_indices": [
                        [1, (2, MeasIndexGroupOptions.CATEGORICAL_ONLY)],
                        [3, (4, "foo")],
                        [5, 6, 7],
                        [(7, MeasIndexGroupOptions.CATEGORICAL_AND_NUMERICAL)],
                    ],
                },
                "should_raise": TypeError,
            },
            {
                "msg": (
                    "`DataEmbeddingLayer` should error if `split_by_measurement_indices` contains elements "
                    "that are not lists"
                ),
                "params": {
                    **valid_params,
                    "split_by_measurement_indices": [
                        [1, (2, MeasIndexGroupOptions.CATEGORICAL_ONLY)],
                        "bar",
                        [5, 6, 7],
                        [(7, MeasIndexGroupOptions.CATEGORICAL_AND_NUMERICAL)],
                    ],
                },
                "should_raise": TypeError,
            },
            {
                "msg": (
                    "`DataEmbeddingLayer` should error if `split_by_measurement_indices` contains elements "
                    "that are not ints or pairs"
                ),
                "params": {
                    **valid_params,
                    "split_by_measurement_indices": [
                        [1, (2, 3, MeasIndexGroupOptions.CATEGORICAL_ONLY)],
                        [5, 6, 7],
                        [(7, MeasIndexGroupOptions.CATEGORICAL_AND_NUMERICAL)],
                    ],
                },
                "should_raise": ValueError,
            },
            {
                "msg": (
                    "`DataEmbeddingLayer` should error if `split_by_measurement_indices` is not a list or "
                    "None."
                ),
                "params": {**valid_params, "split_by_measurement_indices": "foo"},
                "should_raise": TypeError,
            },
        ]

        for case in cases:
            with self.subTest(case["msg"]):
                if "should_raise" in case:
                    with self.assertRaises(case["should_raise"]):
                        DataEmbeddingLayer(**case["params"])
                else:
                    DataEmbeddingLayer(**case["params"])

    def test_get_measurement_index_normalziation(self):
        meas_indices = torch.LongTensor(
            [
                [
                    [1, 2, 3, 2],
                    [3, 3, 3, 1],
                ],
                [
                    [2, 2, 1, 1],
                    [0, 0, 0, 0],
                ],
            ]
        )

        want_normalization_values = torch.Tensor(
            [
                [
                    [1 / 3, 1 / 6, 1 / 3, 1 / 6],
                    [1 / 6, 1 / 6, 1 / 6, 1 / 2],
                ],
                [
                    [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                    [0, 0, 0, 0],
                ],
            ]
        )

        got_normalization_values = DataEmbeddingLayer.get_measurement_index_normalziation(meas_indices)
        self.assertEqual(want_normalization_values, got_normalization_values)

        meas_indices = torch.LongTensor(
            [
                [1, 2, 3, 2],
                [2, 2, 0, 0],
                [3, 3, 3, 1],
            ]
        )

        want_normalization_values = torch.Tensor(
            [
                [1 / 3, 1 / 6, 1 / 3, 1 / 6],
                [1 / 2, 1 / 2, 0, 0],
                [1 / 6, 1 / 6, 1 / 6, 1 / 2],
            ]
        )

        got_normalization_values = DataEmbeddingLayer.get_measurement_index_normalziation(meas_indices)
        self.assertEqual(want_normalization_values, got_normalization_values)

    def test_joint_embeds(self):
        """This tests the joint embedding layer functionality."""

        valid_params = {
            "n_total_embeddings": 4,
            "out_dim": 4,
            "static_embedding_mode": StaticEmbeddingMode.DROP,
        }

        cases = [
            {
                "msg": "Joint embedding should not ignore elements where values_mask is False.",
                "params": valid_params,
                "indices": torch.LongTensor(
                    [
                        [1, 2, 3],
                        [2, 3, 0],
                    ]
                ),
                "measurement_indices": torch.LongTensor(
                    [
                        [1, 2, 1],
                        [2, 1, 0],
                    ]
                ),
                "values": torch.Tensor(
                    [
                        [0, -1.0, 0],
                        [0.5, 0, 0],
                    ]
                ),
                "values_mask": torch.BoolTensor(
                    [
                        [False, True, False],
                        [True, False, False],
                    ]
                ),
                "want": torch.Tensor(
                    [
                        [0, 1, -1, 1],
                        [0, 0, 0.5, 1],
                    ]
                ),
            },
            {
                "msg": "Joint embedding should normalize by measurement index when so directed.",
                "params": {**valid_params, "do_normalize_by_measurement_index": True},
                "indices": torch.LongTensor(
                    [
                        [1, 2, 3],
                        [2, 3, 0],
                    ]
                ),
                "measurement_indices": torch.LongTensor(
                    [
                        [1, 2, 1],
                        [2, 1, 0],
                    ]
                ),
                "values": torch.Tensor(
                    [
                        [0, -1.0, 0],
                        [0.5, 0, 0],
                    ]
                ),
                "values_mask": torch.BoolTensor(
                    [
                        [False, True, False],
                        [True, False, False],
                    ]
                ),
                "want": torch.Tensor(
                    [
                        [0, 1 / 4, -1 / 2, 1 / 4],
                        [0, 0, 0.25, 1 / 2],
                    ]
                ),
            },
        ]

        for case in cases:
            with self.subTest(case["msg"]):
                L = DataEmbeddingLayer(**case["params"])
                L.embed_layer.weight = torch.nn.Parameter(torch.eye(4))

                got = L._joint_embed(
                    indices=case["indices"],
                    measurement_indices=case["measurement_indices"],
                    values=case["values"],
                    values_mask=case["values_mask"],
                )
                self.assertEqual(case["want"], got)

    def test_split_embeds(self):
        """This tests the split embedding layer functionality."""

        valid_params = {
            "n_total_embeddings": 4,
            "out_dim": 4,
            "static_embedding_mode": StaticEmbeddingMode.DROP,
            "categorical_embedding_dim": 4,
            "numerical_embedding_dim": 4,
        }

        cases = [
            {
                "msg": "Split embedding should respect values_mask.",
                "params": valid_params,
                "indices": torch.LongTensor(
                    [
                        [1, 2, 3],
                        [2, 3, 0],
                    ]
                ),
                "measurement_indices": torch.LongTensor(
                    [
                        [1, 2, 1],
                        [2, 1, 0],
                    ]
                ),
                "values": torch.Tensor(
                    [
                        [0, -1.0, 0],
                        [0.5, 0, 0],
                    ]
                ),
                "values_mask": torch.BoolTensor(
                    [
                        [False, True, False],
                        [True, False, False],
                    ]
                ),
                "cat_mask": torch.BoolTensor(
                    [
                        [True, True, True],
                        [True, True, True],
                    ]
                ),
                # 'cat_want': torch.Tensor([
                #     [0, 1/2, 1/2, 1/2],
                #     [0, 0, 1/2, 1/2],
                # ]),
                # 'num_want': torch.Tensor([
                #     [0, 0, 2, 0],
                #     [0, 0, -1, 0],
                # ]),
                # want = 1/2*cat_want + 1/2*num_want
                "want": torch.Tensor(
                    [
                        [0, 1 / 4, 5 / 4, 1 / 4],
                        [0, 0, -1 / 4, 1 / 4],
                    ]
                ),
            },
            {
                "msg": "Split embedding should respect categorical_weight.",
                "params": {**valid_params, "categorical_weight": 1},
                "indices": torch.LongTensor(
                    [
                        [1, 2, 3],
                        [2, 3, 0],
                    ]
                ),
                "measurement_indices": torch.LongTensor(
                    [
                        [1, 2, 1],
                        [2, 1, 0],
                    ]
                ),
                "values": torch.Tensor(
                    [
                        [0, -1.0, 0],
                        [0.5, 0, 0],
                    ]
                ),
                "values_mask": torch.BoolTensor(
                    [
                        [False, True, False],
                        [True, False, False],
                    ]
                ),
                "cat_mask": torch.BoolTensor(
                    [
                        [True, True, True],
                        [True, True, True],
                    ]
                ),
                # 'cat_want': torch.Tensor([
                #     [0, 1/2, 1/2, 1/2],
                #     [0, 0, 1/2, 1/2],
                # ]),
                # 'num_want': torch.Tensor([
                #     [0, 0, 2, 0],
                #     [0, 0, -1, 0],
                # ]),
                # want = 2/3*cat_want + 1/3*num_want
                "want": torch.Tensor(
                    [
                        [0, 1 / 3, 1, 1 / 3],
                        [0, 0, 0, 1 / 3],
                    ]
                ),
            },
            {
                "msg": "Split embedding should respect do_normalize_by_measurement_index.",
                "params": {**valid_params, "do_normalize_by_measurement_index": True},
                "indices": torch.LongTensor(
                    [
                        [1, 2, 3],
                        [2, 3, 0],
                    ]
                ),
                "measurement_indices": torch.LongTensor(
                    [
                        [1, 2, 1],
                        [2, 1, 0],
                    ]
                ),
                "values": torch.Tensor(
                    [
                        [0, -1.0, 0],
                        [0.5, 0, 0],
                    ]
                ),
                "values_mask": torch.BoolTensor(
                    [
                        [False, True, False],
                        [True, False, False],
                    ]
                ),
                "cat_mask": torch.BoolTensor(
                    [
                        [True, True, True],
                        [True, True, True],
                    ]
                ),
                # 'cat_want': torch.Tensor([
                #     [0, 1/8, 1/4, 1/8],
                #     [0, 0, 1/4, 1/4],
                # ]),
                # 'num_want': torch.Tensor([
                #     [0, 0, 1, 0],
                #     [0, 0, -1/2, 0],
                # ]),
                # want = 1/2*cat_want + 1/2*num_want
                "want": torch.Tensor(
                    [
                        [0, 1 / 16, 5 / 8, 1 / 16],
                        [0, 0, -1 / 8, 1 / 8],
                    ]
                ),
            },
            {
                "msg": "Split embedding should respect cat_mask.",
                "params": valid_params,
                "indices": torch.LongTensor(
                    [
                        [1, 2, 3],
                        [2, 3, 0],
                    ]
                ),
                "measurement_indices": torch.LongTensor(
                    [
                        [1, 2, 1],
                        [2, 1, 0],
                    ]
                ),
                "values": torch.Tensor(
                    [
                        [0, -1.0, 0],
                        [0.5, 0, 0],
                    ]
                ),
                "values_mask": torch.BoolTensor(
                    [
                        [False, True, False],
                        [True, False, False],
                    ]
                ),
                "cat_mask": torch.BoolTensor(
                    [
                        [True, True, False],
                        [False, True, True],
                    ]
                ),
                # 'cat_want': torch.Tensor([
                #     [0, 1/2, 1/2, 0],
                #     [0, 0, 0, 1/2],
                # ]),
                # 'num_want': torch.Tensor([
                #     [0, 0, 2, 0],
                #     [0, 0, -1, 0],
                # ]),
                # want = 1/2*cat_want + 1/2*num_want
                "want": torch.Tensor(
                    [
                        [0, 1 / 4, 5 / 4, 0],
                        [0, 0, -1 / 2, 1 / 4],
                    ]
                ),
            },
        ]

        for case in cases:
            with self.subTest(case["msg"]):
                L = DataEmbeddingLayer(**case["params"])
                L.categorical_embed_layer.weight = torch.nn.Parameter(torch.eye(4))
                L.cat_proj.weight = torch.nn.Parameter(torch.eye(4) * 0.5)
                L.cat_proj.bias = torch.nn.Parameter(torch.zeros(4))

                L.numerical_embed_layer.weight = torch.nn.Parameter(torch.eye(4) * 2)
                L.num_proj.weight = torch.nn.Parameter(torch.eye(4) * (-1))
                L.num_proj.bias = torch.nn.Parameter(torch.zeros(4))

                got = L._split_embed(
                    indices=case["indices"],
                    measurement_indices=case["measurement_indices"],
                    values=case["values"],
                    values_mask=case["values_mask"],
                    cat_mask=case["cat_mask"],
                )
                self.assertEqual(case["want"], got)

    def test_embed(self):
        inputs = {
            "indices": torch.LongTensor([[1, 2, 3], [2, 3, 0]]),
            "measurement_indices": torch.LongTensor([[1, 2, 1], [2, 1, 0]]),
            "values": torch.Tensor([[0, -1.0, 0], [0.5, 0, 0]]),
            "values_mask": torch.BoolTensor([[False, True, False], [True, False, False]]),
            "cat_mask": torch.BoolTensor([[True, True, True], [True, True, False]]),
        }
        joint_inputs = tuple(inputs[k] for k in ["indices", "measurement_indices", "values", "values_mask"])
        split_inputs = joint_inputs + (inputs["cat_mask"],)

        cases = [
            {
                "msg": "When embedding dimensions are unset, should call joint_embed.",
                "should_call": "joint",
            },
            {
                "msg": "When embedding dimensions are set, should call split_embed.",
                "categorical_embedding_dim": 2,
                "numerical_embedding_dim": 2,
                "should_call": "split",
            },
            {
                "msg": "When indices are too large, should throw an error.",
                "n_total_embeddings": 2,
                "should_raise": AssertionError,
            },
            {
                "msg": "When embedding_mode is mangled, should throw an error.",
                "embedding_mode": "mangled",
                "should_raise": ValueError,
            },
        ]

        for i, case in enumerate(cases):
            with self.subTest(f"Subtest {i}: {case['msg']}"):
                L = DataEmbeddingLayer(
                    n_total_embeddings=case.get("n_total_embeddings", 4),
                    out_dim=4,
                    static_embedding_mode=StaticEmbeddingMode.DROP,
                    categorical_embedding_dim=case.get("categorical_embedding_dim", None),
                    numerical_embedding_dim=case.get("numerical_embedding_dim", None),
                )

                L._joint_embed = MagicMock()
                L._split_embed = MagicMock()

                if case.get("embedding_mode", None) is not None:
                    L.embedding_mode = case["embedding_mode"]

                if case.get("should_raise", None) is not None:
                    with self.assertRaises(case["should_raise"]):
                        L._embed(**inputs)
                else:
                    L._embed(**inputs)
                    if case["should_call"] == "joint":
                        L._joint_embed.assert_called_once_with(*joint_inputs)
                        L._split_embed.assert_not_called()
                    elif case["should_call"] == "split":
                        L._joint_embed.assert_not_called()
                        L._split_embed.assert_called_once_with(*split_inputs)
                    else:
                        raise ValueError(f"Case {i} has invalid should_call value ({case['should_call']}).")

    def test_split_batch_into_measurement_index_buckets(self):
        batch = PytorchBatch(
            dynamic_measurement_indices=torch.LongTensor(
                [
                    [
                        [1, 1, 2, 3, 3, 3],
                        [2, 2, 1, 1, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                ]
            ),
        )

        L = DataEmbeddingLayer(
            n_total_embeddings=4,
            out_dim=4,
            static_embedding_mode=StaticEmbeddingMode.DROP,
            split_by_measurement_indices=[
                [(1, MeasIndexGroupOptions.CATEGORICAL_AND_NUMERICAL)],
                [(2, MeasIndexGroupOptions.CATEGORICAL_ONLY), 3],
                [(2, MeasIndexGroupOptions.NUMERICAL_ONLY)],
            ],
        )

        want_cat_mask = torch.BoolTensor(
            [
                [
                    [
                        [True, True, False, False, False, False],
                        [False, False, True, True, True, True],
                        [False, False, False, False, False, False],
                    ],
                    [
                        [False, False, True, True, False, False],
                        [True, True, False, False, False, False],
                        [False, False, False, False, False, False],
                    ],
                ],
                [
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                    ],
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                    ],
                ],
            ]
        )

        want_num_mask = torch.BoolTensor(
            [
                [
                    [
                        [True, True, False, False, False, False],
                        [False, False, False, True, True, True],
                        [False, False, True, False, False, False],
                    ],
                    [
                        [False, False, True, True, False, False],
                        [False, False, False, False, False, False],
                        [True, True, False, False, False, False],
                    ],
                ],
                [
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                    ],
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                    ],
                ],
            ]
        )

        got_cat_mask, got_num_mask = L._split_batch_into_measurement_index_buckets(batch)

        self.assertEqual(want_cat_mask, got_cat_mask)
        self.assertEqual(want_num_mask, got_num_mask)

    def test_forward(self):
        """This tests data embedding layer under conditions in which it should produce an embedding."""

        valid_params = {
            "n_total_embeddings": 4,
            "out_dim": 4,
            "static_embedding_mode": StaticEmbeddingMode.DROP,
        }

        # Our default batch will have two patients, one with 2 events, and one with 1 event.
        # The first patient will have two static features, and the second will have one.
        # The events will have up to 3 dynamic features.

        default_batch = PytorchBatch(
            static_indices=torch.LongTensor(
                [
                    [1, 0],
                    [2, 3],
                ]
            ),
            static_measurement_indices=torch.LongTensor(
                [
                    [1, 0],
                    [1, 2],
                ]
            ),
            time_delta=torch.Tensor(
                [
                    [0, 2],
                    [0, 0],
                ]
            ),
            event_mask=torch.BoolTensor(
                [
                    [True, True],
                    [True, False],
                ]
            ),
            dynamic_indices=torch.LongTensor(
                [
                    [
                        [1, 2, 3],
                        [2, 3, 0],
                    ],
                    [
                        [3, 1, 0],
                        [0, 0, 0],
                    ],
                ]
            ),
            dynamic_values=torch.Tensor(
                [
                    [
                        [0, -1.0, 0],
                        [0.5, 0, 0],
                    ],
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                ]
            ),
            dynamic_values_mask=torch.BoolTensor(
                [
                    [
                        [False, True, False],
                        [True, False, False],
                    ],
                    [
                        [False, False, False],
                        [False, False, False],
                    ],
                ]
            ),
            dynamic_measurement_indices=torch.LongTensor(
                [
                    [
                        [1, 2, 1],
                        [2, 1, 0],
                    ],
                    [
                        [1, 1, 0],
                        [0, 0, 0],
                    ],
                ]
            ),
        )

        # Dynamic embeddings will be:
        # [
        #     [0, 1, -1, 1],
        #     [0, 0, 0.5, 1],
        # ],
        # [
        #     [0, 1, 0, 1],
        #     [0, 0, 0, 0],
        # ],
        #
        # Static embeddings will be:
        # [
        #     [0, 1, 0, 0],
        #     [0, 0, 1, 1],
        # ]

        cases = [
            {
                "msg": (
                    "`DataEmbeddingLayer` should produce the correct embedding for a batch of data when "
                    "static_embedding_mode = StaticEmbeddingMode.DROP."
                ),
                "params": valid_params,
                "batch": default_batch,
                "want": torch.Tensor(
                    [
                        [
                            [0, 1, -1, 1],
                            [0, 0, 0.5, 1],
                        ],
                        [
                            [0, 1, 0, 1],
                            [0, 0, 0, 0],
                        ],
                    ]
                ),
            },
            {
                "msg": (
                    "`DataEmbeddingLayer` should produce the correct embedding for a batch of data when "
                    "static_embedding_mode = StaticEmbeddingMode.SUM_ALL and weights are default."
                ),
                "params": {
                    **valid_params,
                    "static_embedding_mode": StaticEmbeddingMode.SUM_ALL,
                },
                "batch": default_batch,
                "want": torch.Tensor(
                    [
                        [
                            [0, 1, -1 / 2, 1 / 2],
                            [0, 1 / 2, 0.25, 1 / 2],
                        ],
                        [
                            [0, 1 / 2, 1 / 2, 1],
                            [0, 0, 0, 0],
                        ],
                    ]
                ),
            },
            {
                "msg": (
                    "`DataEmbeddingLayer` should produce the correct embedding for a batch of data when "
                    "static_embedding_mode = StaticEmbeddingMode.SUM_ALL and weights aren't 1/2, 1/2"
                ),
                "params": {
                    **valid_params,
                    "static_embedding_mode": StaticEmbeddingMode.SUM_ALL,
                    "static_weight": 1 / 2,
                    "dynamic_weight": 1,
                },
                "batch": default_batch,
                "want": torch.Tensor(
                    [
                        [
                            [0, 1, -2 / 3, 2 / 3],
                            [0, 1 / 3, 1 / 3, 2 / 3],
                        ],
                        [
                            [0, 2 / 3, 1 / 3, 1],
                            [0, 0, 0, 0],
                        ],
                    ]
                ),
            },
        ]

        for case in cases:
            with self.subTest(case["msg"]):
                L = DataEmbeddingLayer(**case["params"])
                L.embed_layer.weight = torch.nn.Parameter(torch.eye(4))

                got = L(case["batch"])
                self.assertEqual(case["want"], got)

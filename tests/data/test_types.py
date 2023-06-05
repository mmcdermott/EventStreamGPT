import sys

sys.path.append("../..")

import unittest

import torch

from EventStream.data.types import PytorchBatch

from ..utils import ConfigComparisonsMixin

BATCH_WITHOUT_STATIC = dict(
    event_mask=torch.BoolTensor([[True, True, True], [True, True, False]]),
    time_delta=torch.FloatTensor([[1, 2, 3], [4, 5, 6]]),
    dynamic_indices=torch.LongTensor(
        [
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
            [[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]],
        ]
    ),
    dynamic_measurement_indices=torch.LongTensor(
        [
            [[[1, 1], [3, 3]], [[5, 5], [7, 7]], [[9, 9], [11, 11]]],
            [[[13, 13], [15, 15]], [[17, 17], [19, 19]], [[21, 21], [23, 23]]],
        ]
    ),
    dynamic_values=torch.FloatTensor(
        [
            [[[1.1, 1], [3.1, 3]], [[5.1, 5], [7.1, 7]], [[9.1, 9], [11.1, 11]]],
            [[[13.1, 13], [15.1, 15]], [[17.1, 17], [19.1, 19]], [[21.1, 21], [23.1, 23]]],
        ]
    ),
    dynamic_values_mask=torch.BoolTensor(
        [
            [
                [[True, False], [True, False]],
                [[True, False], [True, False]],
                [[False, True], [False, False]],
            ],
            [
                [[True, False], [True, True]],
                [[True, False], [True, False]],
                [[True, True], [False, True]],
            ],
        ]
    ),
)

BATCH_WITH_STATIC = dict(
    **BATCH_WITHOUT_STATIC,
    static_indices=torch.LongTensor([[1, 2, 3], [3, 2, 3]]),
    static_measurement_indices=torch.LongTensor([[2, 3, 4], [4, 3, 4]]),
)

BATCH_WITH_EXTRAS = dict(
    **BATCH_WITH_STATIC,
    time=torch.FloatTensor([[8, 9, 11], [7, 11, 16]]),
    start_time=torch.FloatTensor([8, 7]),
    stream_labels={"label1": torch.LongTensor([0, 1]), "label2": torch.LongTensor([3.4, 1.1])},
)


class TestPytorchBatch(ConfigComparisonsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.batch = PytorchBatch(**BATCH_WITH_STATIC)

    def test_getitem(self):
        cases = [
            {
                "msg": "Should error on invalid index type",
                "index": {
                    "invalid_type",
                },
                "should_raise": TypeError,
            },
            {
                "msg": "Should return element by name when given string index",
                "index": "event_mask",
                "want": self.batch["event_mask"],
            },
            {
                "msg": "Should error when given a string index that doesn't exist",
                "index": "invalid_index",
                "should_raise": KeyError,
            },
            {
                "msg": "Should slice batch when given a slice index",
                "index": slice(0, 1),
                "want": PytorchBatch(
                    event_mask=torch.BoolTensor([[True, True, True]]),
                    time_delta=torch.FloatTensor([[1, 2, 3]]),
                    static_indices=torch.LongTensor([[1, 2, 3]]),
                    static_measurement_indices=torch.LongTensor([[2, 3, 4]]),
                    dynamic_indices=torch.LongTensor(
                        [
                            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
                        ]
                    ),
                    dynamic_measurement_indices=torch.LongTensor(
                        [
                            [[[1, 1], [3, 3]], [[5, 5], [7, 7]], [[9, 9], [11, 11]]],
                        ]
                    ),
                    dynamic_values=torch.FloatTensor(
                        [
                            [[[1.1, 1], [3.1, 3]], [[5.1, 5], [7.1, 7]], [[9.1, 9], [11.1, 11]]],
                        ]
                    ),
                    dynamic_values_mask=torch.BoolTensor(
                        [
                            [
                                [[True, False], [True, False]],
                                [[True, False], [True, False]],
                                [[False, True], [False, False]],
                            ],
                        ]
                    ),
                ),
            },
            {
                "msg": "Should slice batch when given an integer index",
                "index": 1,
                "want": PytorchBatch(
                    event_mask=torch.BoolTensor([True, True, False]),
                    time_delta=torch.FloatTensor([4, 5, 6]),
                    static_indices=torch.LongTensor([3, 2, 3]),
                    static_measurement_indices=torch.LongTensor([4, 3, 4]),
                    dynamic_indices=torch.LongTensor(
                        [[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]],
                    ),
                    dynamic_measurement_indices=torch.LongTensor(
                        [[[13, 13], [15, 15]], [[17, 17], [19, 19]], [[21, 21], [23, 23]]],
                    ),
                    dynamic_values=torch.FloatTensor(
                        [
                            [[13.1, 13], [15.1, 15]],
                            [[17.1, 17], [19.1, 19]],
                            [[21.1, 21], [23.1, 23]],
                        ],
                    ),
                    dynamic_values_mask=torch.BoolTensor(
                        [
                            [[True, False], [True, True]],
                            [[True, False], [True, False]],
                            [[True, True], [False, True]],
                        ],
                    ),
                ),
            },
            {
                "msg": "Should error when given an empty index",
                "index": tuple(),
                "should_raise": ValueError,
            },
            {
                "msg": "Should error when given a too long index",
                "index": (1, 2, 3, 4),
                "should_raise": ValueError,
            },
            {
                "msg": "Should slice batch when given an tuple",
                "index": (slice(0, 1), slice(0, 2), 1),
                "want": PytorchBatch(
                    event_mask=torch.BoolTensor([[True, True]]),
                    time_delta=torch.FloatTensor([[1, 2]]),
                    static_indices=torch.LongTensor([[1, 2, 3]]),
                    static_measurement_indices=torch.LongTensor([[2, 3, 4]]),
                    dynamic_indices=torch.LongTensor([[[3, 4], [7, 8]]]),
                    dynamic_measurement_indices=torch.LongTensor([[[3, 3], [7, 7]]]),
                    dynamic_values=torch.FloatTensor([[[3.1, 3], [7.1, 7]]]),
                    dynamic_values_mask=torch.BoolTensor([[[True, False], [True, False]]]),
                ),
            },
            {
                "msg": "Should slice batch without static when given an tuple",
                "index": (slice(0, 1), slice(1, 2), 1),
                "do_include_extras": False,
                "do_include_static": False,
                "want": PytorchBatch(
                    event_mask=torch.BoolTensor([[True]]),
                    time_delta=torch.FloatTensor([[2]]),
                    dynamic_indices=torch.LongTensor([[[7, 8]]]),
                    dynamic_measurement_indices=torch.LongTensor([[[7, 7]]]),
                    dynamic_values=torch.FloatTensor([[[7.1, 7]]]),
                    dynamic_values_mask=torch.BoolTensor([[[True, False]]]),
                ),
            },
            {
                "msg": "Should slice batch, including extras, when given an tuple",
                "index": (slice(0, 1), slice(1, 2), 1),
                "do_include_extras": True,
                "want": PytorchBatch(
                    event_mask=torch.BoolTensor([[True]]),
                    time_delta=torch.FloatTensor([[2]]),
                    static_indices=torch.LongTensor([[1, 2, 3]]),
                    static_measurement_indices=torch.LongTensor([[2, 3, 4]]),
                    dynamic_indices=torch.LongTensor([[[7, 8]]]),
                    dynamic_measurement_indices=torch.LongTensor([[[7, 7]]]),
                    dynamic_values=torch.FloatTensor([[[7.1, 7]]]),
                    dynamic_values_mask=torch.BoolTensor([[[True, False]]]),
                    time=torch.FloatTensor([[9]]),
                    start_time=torch.FloatTensor([8]),
                    stream_labels={
                        "label1": torch.LongTensor([0]),
                        "label2": torch.LongTensor([3.4]),
                    },
                ),
            },
        ]

        for case in cases:
            with self.subTest(msg=case["msg"]):
                batch = self.batch
                if not case.get("do_include_static", True):
                    batch = PytorchBatch(**BATCH_WITHOUT_STATIC)
                elif case.get("do_include_extras", False):
                    batch = PytorchBatch(**BATCH_WITH_EXTRAS)

                if case.get("should_raise", None) is not None:
                    with self.assertRaises(case["should_raise"]):
                        batch[case["index"]]
                else:
                    got = batch[case["index"]]
                    self.assertEqual(case["want"], got)

    def test_last_sequence_element_unsqueezed(self):
        want = PytorchBatch(
            event_mask=torch.BoolTensor([[True], [False]]),
            time_delta=torch.FloatTensor([[3], [6]]),
            static_indices=torch.LongTensor([[1, 2, 3], [3, 2, 3]]),
            static_measurement_indices=torch.LongTensor([[2, 3, 4], [4, 3, 4]]),
            dynamic_indices=torch.LongTensor([[[[9, 10], [11, 12]]], [[[21, 22], [23, 24]]]]),
            dynamic_measurement_indices=torch.LongTensor([[[[9, 9], [11, 11]]], [[[21, 21], [23, 23]]]]),
            dynamic_values=torch.FloatTensor([[[[9.1, 9], [11.1, 11]]], [[[21.1, 21], [23.1, 23]]]]),
            dynamic_values_mask=torch.BoolTensor(
                [
                    [[[False, True], [False, False]]],
                    [[[True, True], [False, True]]],
                ]
            ),
        )

        got = self.batch.last_sequence_element_unsqueezed()
        self.assertEqual(want, got)

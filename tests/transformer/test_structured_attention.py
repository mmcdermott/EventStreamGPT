import sys

sys.path.append("../..")

import unittest

import torch

from EventStream.transformer.structured_attention import StructuredAttention

from ..utils import MLTypeEqualityCheckableMixin


class MockSeqModule(torch.nn.Module):
    def __init__(self, return_extra: bool = True):
        super().__init__()
        self.return_extra = return_extra

    def forward(self, x, return_str: str = "SeqModule", **kwargs):
        x = x.cumsum(dim=1)
        if self.return_extra:
            return x**2, return_str
        else:
            return x**2


class MockDepGraphModule(torch.nn.Module):
    def __init__(self, return_extra: bool = True):
        super().__init__()
        self.return_extra = return_extra

    def forward(self, x, return_str: str = "DepGraphModule", **kwargs):
        x = x.cumsum(dim=1)
        out = (-(x**2))[:, 1:, :]
        if self.return_extra:
            return out, return_str
        else:
            return out


class TestStructuredAttention(MLTypeEqualityCheckableMixin, unittest.TestCase):
    def _set_up(
        self,
        seq_return_extra: bool = True,
        dep_graph_return_extra: bool = True,
    ):
        self.seq_module = MockSeqModule(return_extra=seq_return_extra)
        self.dep_graph_module = MockDepGraphModule(return_extra=dep_graph_return_extra)

        self.M = StructuredAttention(
            seq_module=self.seq_module,
            dep_graph_module=self.dep_graph_module,
        )

    def test_e2e(self):
        # Input:
        # We'll build a hidden state with
        # - batch size = 2
        # - num events = 2
        # - dependency graph len = 2
        # - hidden size = 3
        hidden_states = torch.Tensor(
            [
                [[[1, 2, 3], [4, 5, 6]], [[2, 1, 0], [10, 11, 12]]],
                [[[1, 1, 2], [2, 3, 3]], [[4, 4, 5], [3, 2, 2]]],
            ]
        )

        # Computation Flow:
        # First step: Event pooling
        #  - pooled events: torch.Tensor([
        #      [[4, 5, 6], [10, 11, 12]],
        #      [[2, 3, 3], [3, 2, 2]],
        #    ])
        # Second step: Sequential module
        #  - seq_module output: torch.Tensor([
        #       [[4**2, 5**2, 6**2], [14**2, 16**2, 18**2]],
        #       [[2**2, 3**2, 3**2], [5**2, 5**2, 5**2]],
        #    ])
        # Third step: Combine the seq_module output and dep graph elements for dep graph pooling.
        # - concatenated seq_module output and dep graph elements: torch.Tensor([
        #     [
        #       [[0, 0, 0], [1, 2, 3], [4**2, 5**2, 6**2]],
        #       [[4**2, 5**2, 6**2], [2, 1, 0], [14**2, 16**2, 18**2]],
        #     ], [
        #       [[0, 0, 0], [1, 1, 2], [2**2, 3**2, 3**2]],
        #       [[2**2, 3**2, 3**2], [4, 4, 5], [5**2, 5**2, 5**2]],
        #     ],
        #   ])
        # Fourth step: Dep graph module:
        # - dep graph module output:
        # torch.Tensor([
        #     [
        #         [[-(1+0)**2, -(2+0)**2, -(3+0)**2], [-(4**2+1)**2, -(5**2+2)**2, -(6**2+3)**2]],
        #         [
        #             [-(2+4**2)**2, -(1+5**2)**2, -(0+6**2)**2],
        #             [-(2+4**2+14**2)**2, -(1+5**2+16**2)**2, -(0+6**2+18**2)**2],
        #         ],
        #     ], [
        #         [[-(1+0)**2, -(1+0)**2, -(2+0)**2], [-(2**2+1)**2, -(3**2+1)**2, -(3**2+2)**2]],
        #         [
        #             [-(4+2**2)**2, -(4+3**2)**2, -(5+3**2)**2],
        #             [-(4+2**2+5**2)**2, -(4+3**2+5**2)**2, -(5+3**2+5**2)**2],
        #         ],
        #     ],
        # ])

        want_output = torch.Tensor(
            [
                [
                    [
                        [-((1 + 0) ** 2), -((2 + 0) ** 2), -((3 + 0) ** 2)],
                        [-((4**2 + 1) ** 2), -((5**2 + 2) ** 2), -((6**2 + 3) ** 2)],
                    ],
                    [
                        [-((2 + 4**2) ** 2), -((1 + 5**2) ** 2), -((0 + 6**2) ** 2)],
                        [
                            -((2 + 4**2 + 14**2) ** 2),
                            -((1 + 5**2 + 16**2) ** 2),
                            -((0 + 6**2 + 18**2) ** 2),
                        ],
                    ],
                ],
                [
                    [
                        [-((1 + 0) ** 2), -((1 + 0) ** 2), -((2 + 0) ** 2)],
                        [-((2**2 + 1) ** 2), -((3**2 + 1) ** 2), -((3**2 + 2) ** 2)],
                    ],
                    [
                        [-((4 + 2**2) ** 2), -((4 + 3**2) ** 2), -((5 + 3**2) ** 2)],
                        [
                            -((4 + 2**2 + 5**2) ** 2),
                            -((4 + 3**2 + 5**2) ** 2),
                            -((5 + 3**2 + 5**2) ** 2),
                        ],
                    ],
                ],
            ]
        )

        # We can also use seq or dep graph masks. A event_mask would have shape (batch_size, num_events)
        event_mask = torch.BoolTensor([[True, False], [False, True]])

        # With masking, we'll have:
        # First step: Event pooling
        #  - pooled events: torch.Tensor([
        #      [[4, 5, 6], MASKED],
        #      [MASKED, [3, 2, 2]],
        #    ])
        # Second step: Sequential module
        #  - seq_module output: torch.Tensor([
        #       [[4**2, 5**2, 6**2], MASKED],
        #       [MASKED, [3**2, 2**2, 2**2]],
        #    ])
        # Third step: Combine the seq_module output and dep graph elements for dep graph pooling.
        # - concatenated seq_module output and dep graph elements: torch.Tensor([
        #     [
        #       [[0, 0, 0], [1, 2, 3], [4**2, 5**2, 6**2]],
        #       MASKED
        #     ], [
        #       MASKED
        #       [[0, 0, 0], [4, 4, 5], [3**2, 2**2, 2**2]],
        #     ],
        #   ])
        # Fourth step: Dep graph module:
        # - dep graph module output:
        # torch.Tensor([
        #     [
        #         [[-(1+0)**2, -(2+0)**2, -(3+0)**2], [-(4**2+1)**2, -(5**2+2)**2, -(6**2+3)**2]],
        #         MASKED,
        #     ], [
        #         MASKED
        #         [
        #             [-(4+0)**2, -(4+0)**2, -(5+0)**2],
        #             [-(4+0+3**2)**2, -(4+0+2**2)**2, -(5+0+2**2)**2],
        #         ],
        #     ],
        # ])

        want_output_with_event_mask = torch.Tensor(
            [
                [
                    [
                        [-((1 + 0) ** 2), -((2 + 0) ** 2), -((3 + 0) ** 2)],
                        [-((4**2 + 1) ** 2), -((5**2 + 2) ** 2), -((6**2 + 3) ** 2)],
                    ],
                    [[0, 0, 0], [0, 0, 0]],
                ],
                [
                    [[0, 0, 0], [0, 0, 0]],
                    [
                        [-((4 + 0) ** 2), -((4 + 0) ** 2), -((5 + 0) ** 2)],
                        [
                            -((4 + 0 + 3**2) ** 2),
                            -((4 + 0 + 2**2) ** 2),
                            -((5 + 0 + 2**2) ** 2),
                        ],
                    ],
                ],
            ]
        )

        cases = [
            {
                "msg": "When neither returns extra, should return the correct value",
                "want": (
                    want_output,
                    {"seq_module": None, "dep_graph_module": None},
                ),
            },
            {
                "msg": (
                    "When seq_module returns extra but dep_graph_module doesn't, should return the correct "
                    "value and only seq_module's extra keyword arg"
                ),
                "seq_return_extra": True,
                "dep_graph_return_extra": False,
                "want": (
                    want_output,
                    {"seq_module": "SeqModule", "dep_graph_module": None},
                ),
            },
            {
                "msg": (
                    "When seq_module doesn't return extra but dep_graph_module does, should return the "
                    "correct value and only dep_graph_module's extra keyword arg"
                ),
                "seq_return_extra": False,
                "dep_graph_return_extra": True,
                "want": (
                    want_output,
                    {"seq_module": None, "dep_graph_module": "DepGraphModule"},
                ),
            },
            {
                "msg": (
                    "When neither seq_module nor dep_graph_module returns extra, should return the "
                    "correct value and neither extra keyword arg"
                ),
                "seq_return_extra": False,
                "dep_graph_return_extra": False,
                "want": (
                    want_output,
                    {"seq_module": None, "dep_graph_module": None},
                ),
            },
            {
                "msg": "With seq masked, should return the correct value.",
                "event_mask": event_mask,
                "want": (
                    want_output_with_event_mask,
                    {"seq_module": None, "dep_graph_module": None},
                ),
            },
            {
                "msg": "Should respect seq_module_kwargs and dep_graph_module_kwargs",
                "seq_module_kwargs": {"return_str": "SeqModule2"},
                "seq_return_extra": True,
                "dep_graph_module_kwargs": {"return_str": "DepGraphModule2"},
                "dep_graph_return_extra": True,
                "want": (
                    want_output,
                    {"seq_module": "SeqModule2", "dep_graph_module": "DepGraphModule2"},
                ),
            },
        ]

        for case in cases:
            with self.subTest(msg=case["msg"]):
                self._set_up(
                    seq_return_extra=case.get("seq_return_extra", False),
                    dep_graph_return_extra=case.get("dep_graph_return_extra", False),
                )

                got = self.M(
                    hidden_states,
                    event_mask=case.get("event_mask", None),
                    seq_module_kwargs=case.get("seq_module_kwargs", None),
                    dep_graph_module_kwargs=case.get("dep_graph_module_kwargs", None),
                )
                want = case["want"]

                self.assertTrue(type(got) is type(want))
                self.assertEqual(len(got), len(want))
                self.assertEqual(want[0], got[0])
                self.assertEqual(want[1], got[1])


if __name__ == "__main__":
    unittest.main()

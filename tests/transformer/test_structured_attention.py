import sys

sys.path.append("../..")

import unittest

import torch

from EventStream.transformer.structured_attention import (
    StructuredAttention,
    TakeExistingEventEmbedding,
)

from ..mixins import MLTypeEqualityCheckableMixin


class TestTakeExistingEventEmbedding(MLTypeEqualityCheckableMixin, unittest.TestCase):
    def test_e2e(self):
        M = TakeExistingEventEmbedding()

        X = torch.Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

        got = M(X)

        want = torch.Tensor([[4, 5, 6], [10, 11, 12]])
        self.assertEqual(got, want)


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

        # TODO(mmd): All wrong!

        # Computation Flow:
        # First step: Event pooling
        #  - pooled events: torch.Tensor([
        #      [[], [12, 12, 12]],
        #      [[3, 4, 5], [7, 6, 7]],
        #    ])
        # Second step: Sequential module
        #  - seq_module output: torch.Tensor([
        #       [[5**2, 7**2, 9**2], [17**2, 19**2, 21**2]],
        #       [[3**2, 4**2, 5**2], [10**2, 10**2, 12**2]],
        #    ])
        # Third step: Combine the seq_module output and dep graph elements for dep graph pooling.
        # - concatenated seq_module output and dep graph elements: torch.Tensor([
        #     [
        #       [[0, 0, 0], [1, 2, 3], [5**2, 7**2, 9**2]],
        #       [[5**2, 7**2, 9**2], [2, 1, 0], [17**2, 19**2, 21**2]],
        #     ], [
        #       [[0, 0, 0], [1, 1, 2], [3**2, 4**2, 5**2]],
        #       [[3**2, 4**2, 5**2], [4, 4, 5], [10**2, 10**2, 12**2]],
        #     ],
        #   ])
        # Fourth step: Dep graph module:
        # - dep graph module output:
        # torch.Tensor([
        #     [
        #         [[-1, -4, -9], [-26**2, -51**2, -84**2]],
        #         [
        #             [-(5**2+2)**2, -(7**2+1)**2, -(9**2+0)**2],
        #             [-(5**2 + 2 + 17**2)**2, -(7**2 + 1 + 19**2)**2, -(9**2 + 0 + 21**2)**2]
        #         ],
        #     ], [
        #         [[-1, -1, -4], [-10**2, -17**2, -27**2]],
        #         [
        #             [-13**2, -20**2, -30**2],
        #             [-(3**2 + 4 + 10**2)**2, -(4**2 + 4 + 10**2)**2, -(5**2 + 5 + 12**2)**2]
        #         ],
        #     ],
        # ])

        want_output = torch.Tensor(
            [
                [
                    [[-1, -4, -9], [-(26**2), -(51**2), -(84**2)]],
                    [
                        [-((5**2 + 2) ** 2), -((7**2 + 1) ** 2), -((9**2 + 0) ** 2)],
                        [
                            -((5**2 + 2 + 17**2) ** 2),
                            -((7**2 + 1 + 19**2) ** 2),
                            -((9**2 + 0 + 21**2) ** 2),
                        ],
                    ],
                ],
                [
                    [[-1, -1, -4], [-(10**2), -(17**2), -(27**2)]],
                    [
                        [-(13**2), -(20**2), -(30**2)],
                        [
                            -((3**2 + 4 + 10**2) ** 2),
                            -((4**2 + 4 + 10**2) ** 2),
                            -((5**2 + 5 + 12**2) ** 2),
                        ],
                    ],
                ],
            ]
        )

        # We can also use seq or dep graph masks. A seq_mask would have shape (batch_size, num_events)
        # and a dep graph mask would have shape (batch_size, num_events, dep_graph_len).
        seq_mask = torch.Tensor([[1, 0], [0, 1]])

        # With masking, we'll have:
        # First step: Event pooling
        #  - pooled events: torch.Tensor([
        #      [[5, 7, 9], MASKED],
        #      [MASKED, [7, 6, 7]],
        #    ])
        # Second step: Sequential module
        #  - seq_module output: torch.Tensor([
        #       [[25, 49, 81], MASKED],
        #       [MASKED, [49, 36, 49]],
        #    ])
        # Third step: Combine the seq_module output and dep graph elements for dep graph pooling.
        # If we do update to contextualized event, then we'll have:
        # - concatenated seq_module output and dep graph elements: torch.Tensor([
        #     [
        #       [[0, 0, 0], [1, 2, 3], [25, 49, 81]],
        #       MASKED
        #     ], [
        #       MASKED
        #       [[0, 0, 0], [4, 4, 5], [49, 36, 49]],
        #     ],
        #   ])
        # Fourth step: Dep graph module:
        # With update to contextualized event:
        # - dep graph module output: torch.Tensor([
        #     [
        #         [[-1, -4, -9], [-26**2, -51**2, -84**2]],
        #         MASKED
        #     ], [
        #         MASKED
        #         [[-16, -16, -25], [-53**2, -40**2, -54**2]],
        #     ],
        # ])

        want_output_with_seq_mask = torch.Tensor(
            [
                [
                    [[-1, -4, -9], [-(26**2), -(51**2), -(84**2)]],
                    [[0, 0, 0], [0, 0, 0]],
                ],
                [
                    [[0, 0, 0], [0, 0, 0]],
                    [[-16, -16, -25], [-(53**2), -(40**2), -(54**2)]],
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
                "seq_mask": seq_mask,
                "want": (
                    want_output_with_seq_mask,
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
                    seq_mask=case.get("seq_mask", None),
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

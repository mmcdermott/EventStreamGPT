import sys

sys.path.append("../..")

import unittest

import torch

from EventStream.data.types import PytorchBatch
from EventStream.transformer.config import (
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
)
from EventStream.transformer.fine_tuning_model import ESTForStreamClassification

from ..utils import MLTypeEqualityCheckableMixin

TEST_MEASUREMENTS_IDXMAP = {
    "event_type": 1,
    "multi_label_col": 2,
    "regression_col": 3,
}
TEST_MEASUREMENTS_PER_DEP_GRAPH_LEVEL = [[], ["event_type"], ["multi_label_col", "regression_col"]]


class TestESTForStreamClassification(MLTypeEqualityCheckableMixin, unittest.TestCase):
    def test_forward(self):
        class HiddenStateNoDepGraphLevel:
            @property
            def last_hidden_state(self):
                return torch.Tensor(
                    [
                        [
                            [1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 10.0, 11.0, 12],
                        ],
                        [
                            [1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 10.0, 11.0, 12],
                        ],
                    ]
                )

        class HiddenStateDepGraphLevel:
            @property
            def last_hidden_state(self):
                return torch.Tensor(
                    [
                        [
                            [
                                [1.0, 2.0, 3.0, 4.0],
                                [1.0, 2.0, 3.0, 4.0],
                                [-1.0, 0.0, -3.0, 4.0],
                            ],
                            [
                                [5.0, 6.0, 7.0, 8.0],
                                [1.0, 2.0, 3.0, 4.0],
                                [-5.0, 5.0, -7.0, 8.0],
                            ],
                            [
                                [9.0, 10.0, 11.0, 12],
                                [1.0, 2.0, 3.0, 4.0],
                                [-9.0, 10.0, -11.0, 12],
                            ],
                        ],
                        [
                            [
                                [1.0, 2.0, 3.0, 4.0],
                                [1.0, 2.0, 3.0, 4.0],
                                [-1.0, 0.0, -3.0, 4.0],
                            ],
                            [
                                [5.0, 6.0, 7.0, 8.0],
                                [1.0, 2.0, 3.0, 4.0],
                                [-5.0, 5.0, -7.0, 8.0],
                            ],
                            [
                                [9.0, 10.0, 11.0, 12],
                                [1.0, 2.0, 3.0, 4.0],
                                [-9.0, 10.0, -11.0, 12],
                            ],
                        ],
                    ]
                )

        batch = {
            "stream_labels": {"test": torch.LongTensor([0, 2])},
            "event_mask": torch.BoolTensor([[True, True, True], [True, True, True]]),
        }
        default_cfg_kwargs = {
            "finetuning_task": "test",
            "num_classes": 4,
            "task_specific_params": {"pooling_method": "cls"},
            "structured_event_processing_mode": StructuredEventProcessingMode.NESTED_ATTENTION,
            "measurements_per_dep_graph_level": TEST_MEASUREMENTS_PER_DEP_GRAPH_LEVEL,
            "measurements_idxmap": TEST_MEASUREMENTS_IDXMAP,
        }
        default_weight = torch.nn.Parameter(torch.eye(4))
        default_bias = torch.nn.Parameter(torch.zeros(4))
        default_encoder = HiddenStateDepGraphLevel

        cases = [
            {
                "msg": "It should perform binary classification in the appropriate setting.",
                "kwargs": dict(
                    structured_event_processing_mode=StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT,
                    measurements_per_dep_graph_level=None,
                    do_full_block_in_seq_attention=None,
                    do_full_block_in_dep_graph_attention=None,
                    dep_graph_attention_types=None,
                    dep_graph_window_size=None,
                    num_classes=2,
                    id2label={0: False, 1: True},
                    label2id={False: 0, True: 1},
                ),
                "mock_encoder": HiddenStateNoDepGraphLevel,
                "weight": torch.nn.Parameter(torch.Tensor([[1, 0, 0, 0]])),
                "bias": torch.nn.Parameter(torch.Tensor([0])),
                "batch": {"stream_labels": {"test": torch.FloatTensor([0, 1])}},
                # event_encoded should be
                # torch.Tensor([
                #     [
                #         [1., 2., 3., 4.],
                #         [5., 6., 7., 8.],
                #         [9., 10., 11., 12],
                #     ], [
                #         [1., 2., 3., 4.],
                #         [5., 6., 7., 8.],
                #         [9., 10., 11., 12],
                #     ]
                # ])
                # After cls pooling, we have
                # torch.Tensor([
                #     [1., 2., 3., 4.],
                #     [1., 2., 3., 4.],
                # ])
                # After logit_layer, we have logits:
                # torch.Tensor([1, 1])
                # This yields loss 0.8132616281509399
                "want_preds": torch.Tensor(
                    [
                        1,
                        1,
                    ]
                ),
                "want_labels": torch.FloatTensor([0, 1]),
                "want_loss": torch.tensor(0.8132616281509399),
            },
            {
                "msg": "It should select from the sequence in the conditional independence, CLS case.",
                "kwargs": dict(
                    structured_event_processing_mode=StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT,
                    measurements_per_dep_graph_level=None,
                    do_full_block_in_seq_attention=None,
                    do_full_block_in_dep_graph_attention=None,
                    dep_graph_attention_types=None,
                    dep_graph_window_size=None,
                ),
                "mock_encoder": HiddenStateNoDepGraphLevel,
                # event_encoded should be
                # torch.Tensor([
                #     [
                #         [1., 2., 3., 4.],
                #         [5., 6., 7., 8.],
                #         [9., 10., 11., 12],
                #     ], [
                #         [1., 2., 3., 4.],
                #         [5., 6., 7., 8.],
                #         [9., 10., 11., 12],
                #     ]
                # ])
                # After cls pooling, we have
                # torch.Tensor([
                #     [1., 2., 3., 4.],
                #     [1., 2., 3., 4.],
                # ])
                # After logit_layer, we have logits:
                # torch.Tensor([
                #     [1., 2., 3., 4.],
                #     [1., 2., 3., 4.],
                # ])
                # This yields loss 2.4401895999908447
                "want_preds": torch.Tensor(
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [1.0, 2.0, 3.0, 4.0],
                    ]
                ),
                "want_labels": torch.LongTensor([0, 2]),
                "want_loss": torch.tensor(2.4401895999908447),
            },
            {
                "msg": "It should select from the dep_graph in the nested attention, CLS case.",
                "kwargs": dict(),
                # event_encoded should be
                # torch.Tensor([
                #     [
                #         [-1., 0., -3., 4.],
                #         [-5., 5., -7., 8.],
                #         [-9., 10., -11., 12],
                #     ], [
                #         [-1., 0., -3., 4.],
                #         [-5., 5., -7., 8.],
                #         [-9., 10., -11., 12],
                #     ]
                # ])
                # After cls pooling, we have
                # torch.Tensor([
                #     [-1., 0., -3., 4.],
                #     [-1., 0., -3., 4.],
                # ])
                # After logit_layer, we have logits:
                # torch.Tensor([
                #     [-1., 0., -3., 4.],
                #     [-1., 0., -3., 4.],
                # ])
                # This yields loss 6.025634288787842
                "want_preds": torch.Tensor(
                    [
                        [-1.0, 0.0, -3.0, 4.0],
                        [-1.0, 0.0, -3.0, 4.0],
                    ]
                ),
                "want_labels": torch.LongTensor([0, 2]),
                "want_loss": torch.tensor(6.025634288787842),
            },
            {
                "msg": "It should select from the dep_graph in the nested attention, mean case.",
                "kwargs": {"task_specific_params": {"pooling_method": "mean"}},
                # event_encoded should be
                # torch.Tensor([
                #     [
                #         [-1., 0., -3., 4.],
                #         [-5., 5., -7., 8.],
                #         [-9., 10., -11., 12],
                #     ], [
                #         [-1., 0., -3., 4.],
                #         [-5., 5., -7., 8.],
                #         [-9., 10., -11., 12],
                #     ]
                # ])
                # After mean pooling, we have
                # torch.Tensor([
                #     [-5., 5., -7., 8.],
                #     [-5., 5., -7., 8.],
                # ])
                # After logit_layer, we have logits:
                # torch.Tensor([
                #     [-5., 5., -7., 8.],
                #     [-5., 5., -7., 8.],
                # ])
                # This yields loss 14.048589706420898
                "want_preds": torch.Tensor(
                    [
                        [-5, 5, -7, 8],
                        [-5, 5, -7, 8],
                    ]
                ),
                "want_labels": torch.LongTensor([0, 2]),
                "want_loss": torch.tensor(14.048589706420898),
            },
            {
                "msg": "It should select from the dep_graph in the nested attention, max case.",
                "kwargs": {"task_specific_params": {"pooling_method": "max"}},
                # event_encoded should be
                # torch.Tensor([
                #     [
                #         [-1., 2., -3., 4.],
                #         [-5., 6., -7., 8.],
                #         [-9., 10., -11., 12],
                #     ], [
                #         [-1., 2., -3., 4.],
                #         [-5., 6., -7., 8.],
                #         [-9., 10., -11., 12],
                #     ]
                # ])
                # After max pooling, we have
                # torch.Tensor([
                #     [-1., 10., -3., 12.],
                #     [-1., 10., -3., 12.],
                # ])
                # After logit_layer, we have logits:
                # torch.Tensor([
                #     [-1., 10., -3., 12.],
                #     [-1., 10., -3., 12.],
                # ])
                # This yields loss 14.126930236816406
                "want_preds": torch.Tensor(
                    [
                        [-1.0, 10.0, -3.0, 12.0],
                        [-1.0, 10.0, -3.0, 12.0],
                    ]
                ),
                "want_labels": torch.LongTensor([0, 2]),
                "want_loss": torch.tensor(14.126930236816406),
            },
        ]

        for C in cases:
            with self.subTest(C["msg"]):
                C["batch"] = PytorchBatch(**C.get("batch", batch))
                cfg = StructuredTransformerConfig(**{**default_cfg_kwargs, **C["kwargs"]})
                M = ESTForStreamClassification(cfg)

                class MockEncoder(torch.nn.Module):
                    def __init__(self):
                        super().__init__()

                    def forward(self, *args, **kwargs):
                        return C.get("mock_encoder", default_encoder)()

                M.encoder = MockEncoder()
                M.logit_layer.weight = C.get("weight", default_weight)
                M.logit_layer.bias = C.get("bias", default_bias)

                out = M.forward(C.get("batch", batch))

                self.assertEqual(C["want_labels"], out.labels)
                self.assertEqual(C["want_preds"], out.preds)
                self.assertEqual(C["want_loss"], out.loss)


if __name__ == "__main__":
    unittest.main()

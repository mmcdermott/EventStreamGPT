import sys

sys.path.append("../..")

import unittest

import numpy as np
import torch
from pytorch_lognormal_mixture import LogNormalMixtureDistribution

from EventStream.data.types import DataModality, PytorchBatch
from EventStream.transformer.config import (
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
)
from EventStream.transformer.fine_tuning_model import ESTForStreamClassification
from EventStream.transformer.generative_model_base import GenerativeOutputLayerBase

from ..mixins import MLTypeEqualityCheckableMixin

TEST_MEASUREMENTS_PER_GEN_MODE = {
    DataModality.SINGLE_LABEL_CLASSIFICATION: ["event_type"],
    DataModality.MULTI_LABEL_CLASSIFICATION: ["multi_label_col", "regression_col"],
    DataModality.MULTIVARIATE_REGRESSION: ["regression_col"],
}
TEST_MEASUREMENTS_IDXMAP = {
    "event_type": 1,
    "multi_label_col": 2,
    "regression_col": 3,
}
TEST_MEASUREMENTS_PER_DEP_GRAPH_LEVEL = [[], ["event_type"], ["multi_label_col", "regression_col"]]
# These are all including the 'UNK' tokens. So, e.g., there are 2 real options for 'event_type'.
TEST_VOCAB_SIZES_BY_MEASUREMENT = {
    "event_type": 2,
    "multi_label_col": 3,
    "regression_col": 4,
}
TEST_VOCAB_OFFSETS_BY_MEASUREMENT = {
    "event_type": 1,
    "multi_label_col": 3,
    "regression_col": 6,
}
TEST_EVENT_TYPES_IDXMAP = {
    "event_A": 0,
    "event_B": 1,
}

BASE_CONFIG_KWARGS = dict(
    measurements_per_generative_mode=TEST_MEASUREMENTS_PER_GEN_MODE,
    vocab_sizes_by_measurement=TEST_VOCAB_SIZES_BY_MEASUREMENT,
    vocab_offsets_by_measurement=TEST_VOCAB_OFFSETS_BY_MEASUREMENT,
    measurements_idxmap=TEST_MEASUREMENTS_IDXMAP,
    event_types_idxmap=TEST_EVENT_TYPES_IDXMAP,
    hidden_size=4,
    head_dim=None,
    num_attention_heads=2,  # Needs to divide hidden_size.
    measurements_per_dep_graph_level=TEST_MEASUREMENTS_PER_DEP_GRAPH_LEVEL,
)

BASE_BATCH = {
    "event_mask": torch.BoolTensor([[True, True, True]]),
    "time_delta": torch.FloatTensor([[2, 3, 1]]),
    "dynamic_values_mask": torch.BoolTensor(
        [
            [
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, True, True, True],
            ],
        ]
    ),
    "dynamic_measurement_indices": torch.LongTensor(
        [
            [
                [1, 0, 0, 0, 0, 0],
                [1, 2, 0, 0, 0, 0],
                [1, 2, 2, 3, 3, 3],
            ],
        ]
    ),
    "dynamic_indices": torch.LongTensor(
        [
            [
                [1, 0, 0, 0, 0, 0],
                [2, 5, 0, 0, 0, 0],
                [2, 4, 5, 7, 8, 9],
            ],
        ]
    ),
    "dynamic_values": torch.Tensor(
        [
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1.1, -1.1, 0.0],
            ],
        ]
    ),
}


class TestGenerativeOutputLayerBase(MLTypeEqualityCheckableMixin, unittest.TestCase):
    """Tests the OutputLayer."""

    def test_constructs(self):
        """Tests that the Model Output Layer constructs given default configuration options."""
        config = StructuredTransformerConfig(**BASE_CONFIG_KWARGS)
        GenerativeOutputLayerBase(config)

    def test_get_event_type_mask_per_measurement(self):
        config = StructuredTransformerConfig(
            **{
                **BASE_CONFIG_KWARGS,
                "event_types_per_measurement": {
                    "event_type": ["event_A", "event_B"],
                    "multi_label_col": ["event_A"],
                    "regression_col": ["event_B"],
                },
            }
        )
        batch = PytorchBatch(
            **{
                **BASE_BATCH,
                "dynamic_measurement_indices": torch.LongTensor(
                    [
                        [
                            [1, 0, 0, 0, 0, 0],
                            [1, 2, 0, 0, 0, 0],
                            [1, 2, 2, 3, 3, 3],
                        ],
                    ]
                ),
                "dynamic_indices": torch.LongTensor(
                    [
                        [
                            [1, 0, 0, 0, 0, 0],
                            [2, 5, 0, 0, 0, 0],
                            [2, 4, 5, 7, 8, 9],
                        ],
                    ]
                ),
                "event_mask": torch.BoolTensor([[True, True, True]]),
            }
        )

        layer = GenerativeOutputLayerBase(config)

        # Recall these are our measurement types
        # TEST_MEASUREMENTS_IDXMAP = {
        #     'event_type': 1,
        #     'multi_label_col': 2,
        #     'regression_col': 3,
        # }
        # And that we have two options for event types:
        # TEST_EVENT_TYPES_IDXMAP = {
        #     'event_A': 0,
        #     'event_B': 1,
        # }
        want_masks = {
            "event_type": torch.BoolTensor([[True, True, True]]),
            "multi_label_col": torch.BoolTensor([[True, False, False]]),
            "regression_col": torch.BoolTensor([[False, True, True]]),
        }

        self.assertNestedDictEqual(want_masks, layer.get_event_type_mask_per_measurement(batch))

    def test_get_classification_outputs(self):
        cases = [
            {
                "message": "Model should yield the correct outputs given inputs",
                "batch": {**BASE_BATCH},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 0.0],
                            [0.0, 2.0, 5.0, 8.0, 1.0, 4.0, 7.0, 0.0, 3.0, 6.0],
                        ],
                    ]
                ),
                "valid_measurements": {"event_type", "multi_label_col", "regression_col"},
                "want_dists": {
                    # All dists are of shape batch X seq X vocab size.
                    "event_type": torch.distributions.Categorical(
                        logits=torch.FloatTensor(
                            [
                                [
                                    [0.0, 1.0],
                                    [1.0, 3.0],
                                    [2.0, 5.0],
                                ]
                            ]
                        )
                    ),
                    "multi_label_col": torch.distributions.Bernoulli(
                        logits=torch.FloatTensor(
                            [
                                [
                                    [2.0, 3.0, 4.0],
                                    [5.0, 7.0, 9.0],
                                    [8.0, 1.0, 4.0],
                                ]
                            ]
                        )
                    ),
                    "regression_col": torch.distributions.Bernoulli(
                        logits=torch.FloatTensor(
                            [
                                [
                                    [5.0, 6.0, 7.0, 8.0],
                                    [2.0, 4.0, 6.0, 0.0],
                                    [7.0, 0.0, 3.0, 6.0],
                                ]
                            ]
                        )
                    ),
                },
                "want_labels": {
                    # The single-label classification task has shape batch X seq and has labels indices in
                    # it (in long format).
                    # Recall that event_type has no ['UNK'] currently prepending the vocab.
                    # TODO(mmd): Should likely have one.
                    "event_type": torch.LongTensor(
                        [
                            [0, 1, 1],
                        ]
                    ),
                    # The multi-label classification tasks have shape batch X seq X vocab size with
                    # binary indicators. They are also in float format, not long format. Also, note that
                    # labels are only present (non-zero) when the batch's data_type matches the target.
                    "multi_label_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0],
                                [0, 0, 1],
                                [0, 1, 1],
                            ]
                        ]
                    ),
                    "regression_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 1, 1, 1],
                            ]
                        ]
                    ),
                },
                # Losses should be given as follows.
                "want_losses": {
                    # event_type has 3 pairs of (logit, label) across each event:
                    #   ([0.0, 1.0], 0), ([1.0, 3.0], 1), ([2.0, 5.0], 1).
                    # We want to compute the NLL of this setting, which should then be averaged across
                    # events. So we want:
                    #  1/3 * (
                    #    -math.log(math.exp(0)/(math.exp(0) + math.exp(1))) +
                    #    -math.log(math.exp(3)/(math.exp(1) + math.exp(3))) +
                    #    -math.log(math.exp(5)/(math.exp(2) + math.exp(5)))
                    #  )
                    "event_type": torch.tensor(0.49625901671164574),
                    # multi_label_col is has no positive labels for the first event (as it is actually not
                    # reported there), then has logits and labels for the last two events. Our code currently
                    # tasks the model with predicting on all events, including the first, just with all
                    # negative labels, as in theory the multi-label events that aren't observed there are
                    # valid instances of the labels not being present.
                    #
                    # TODO(mmd): Is this right for the multi-label? Or should it be not measured for the first
                    # event?
                    # (logits, labels):
                    #  ([2, 3, 4], [0, 0, 0]), ([5, 7, 9], [0, 0, 1]), ([8, 1, 4], [0, 1, 1])
                    # We want to compute the NLL of this setting, which should then be averaged first across
                    # all labels for the multi-label problem, then across only those events that are unmasked.
                    # 1/3 * (
                    #   1/3 * (
                    #     -math.log(1 - 1/(1 + math.exp(-2)))
                    #     -math.log(1 - 1/(1 + math.exp(-3)))
                    #     -math.log(1 - 1/(1 + math.exp(-4)))
                    #   ) + 1/3 * (
                    #     -math.log(1 - 1/(1 + math.exp(-5)))
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1/(1 + math.exp(-9)))
                    #   ) + 1/3 * (
                    #     -math.log(1 - 1/(1 + math.exp(-8)))
                    #     -math.log(1/(1 + math.exp(-1)))
                    #     -math.log(1/(1 + math.exp(-4)))
                    #   )
                    # ) = 3.2814624309539795
                    # If we instead only scored this on events with labels present:
                    #  ([5, 7, 9], [0, 0, 1]), ([8, 1, 4], [0, 1, 1])
                    # 1/2 * (
                    #   1/3 * (
                    #     -math.log(1 - 1/(1 + math.exp(-5)))
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1/(1 + math.exp(-9)))
                    #   ) + 1/3 * (
                    #     -math.log(1 - 1/(1 + math.exp(-8)))
                    #     -math.log(1/(1 + math.exp(-1)))
                    #     -math.log(1/(1 + math.exp(-4)))
                    #   )
                    # ) = 3.389916102091471
                    "multi_label_col": torch.tensor(3.2814624309539795),
                    # regression_col has no labels for the first two events, then has logits and labels for
                    # the last event as follows:
                    # ([5, 6, 7, 8], [0, 0, 0, 0]), ([2, 4, 6, 0], [0, 0, 0, 0]) ([7, 0, 3, 6], [0, 1, 1, 1])
                    # We want to compute the NLL of this setting, which should then be averaged first across
                    # all labels for the multi-label problem, then across only those events that are unmasked.
                    # 1/3 * (
                    #   1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-5)))
                    #     -math.log(1 - 1/(1 + math.exp(-6)))
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1 - 1/(1 + math.exp(-8)))
                    #   ) + 1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-2)))
                    #     -math.log(1 - 1/(1 + math.exp(-4)))
                    #     -math.log(1 - 1/(1 + math.exp(-6)))
                    #     -math.log(1 - 1/(1 + math.exp(-0)))
                    #   ) + 1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1/(1 + math.exp(-0)))
                    #     -math.log(1/(1 + math.exp(-3)))
                    #     -math.log(1/(1 + math.exp(-6)))
                    #   )
                    # ) = 3.883021699569808
                    # If we only wanted to do this on events with an event type for which regression_col is
                    # ever reported (event type 2, the last two events), we would have:
                    # 1/2 * (
                    #   1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-2)))
                    #     -math.log(1 - 1/(1 + math.exp(-4)))
                    #     -math.log(1 - 1/(1 + math.exp(-6)))
                    #     -math.log(1 - 1/(1 + math.exp(-0)))
                    #   ) + 1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1/(1 + math.exp(-0)))
                    #     -math.log(1/(1 + math.exp(-3)))
                    #     -math.log(1/(1 + math.exp(-6)))
                    #   )
                    # ) = 2.5732278110479676
                    # If we only did this on events with that measurement measured at all:
                    # ([7, 0, 3, 6], [0, 1, 1, 1])
                    #
                    # 1/4 * (
                    #   -math.log(1 - 1/(1 + math.exp(-7)))
                    #   -math.log(1/(1 + math.exp(-0)))
                    #   -math.log(1/(1+math.exp(-3)))
                    #   -math.log(1/(1 + math.exp(-6)))
                    # ) = 1.9362804209313096
                    "regression_col": torch.tensor(3.883021699569808),
                },
            },
            {
                "message": (
                    "Model should only compute losses over measurement-specific event types when computing "
                    "losses."
                ),
                # Recall these are our measurement types
                # TEST_MEASUREMENTS_IDXMAP = {
                #     'event_type': 1,
                #     'multi_label_col': 2,
                #     'regression_col': 3,
                # }
                # And that we have two options for event types:
                # TEST_EVENT_TYPES_IDXMAP = {
                #     'event_A': 0,
                #     'event_B': 1,
                # }
                "config_kwargs": {
                    "event_types_per_measurement": {
                        "event_type": ["event_A", "event_B"],
                        "multi_label_col": ["event_A"],
                        "regression_col": ["event_B"],
                    },
                },
                "include_event_types_mask": True,
                "batch": {
                    **BASE_BATCH,
                    "dynamic_measurement_indices": torch.LongTensor(
                        [
                            [
                                [1, 0, 0, 0, 0, 0],
                                [1, 2, 0, 0, 0, 0],
                                [1, 2, 2, 3, 3, 3],
                            ],
                        ]
                    ),
                    "dynamic_indices": torch.LongTensor(
                        [
                            [
                                [1, 0, 0, 0, 0, 0],
                                [2, 5, 0, 0, 0, 0],
                                [2, 4, 5, 7, 8, 9],
                            ],
                        ]
                    ),
                    "event_mask": torch.BoolTensor([[True, True, True]]),
                },
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 0.0],
                            [0.0, 2.0, 5.0, 8.0, 1.0, 4.0, 7.0, 0.0, 3.0, 6.0],
                        ],
                    ]
                ),
                "valid_measurements": {"event_type", "multi_label_col", "regression_col"},
                "want_dists": {
                    # All dists are of shape batch X seq X vocab size.
                    "event_type": torch.distributions.Categorical(
                        logits=torch.FloatTensor(
                            [
                                [
                                    [0.0, 1.0],
                                    [1.0, 3.0],
                                    [2.0, 5.0],
                                ]
                            ]
                        )
                    ),
                    "multi_label_col": torch.distributions.Bernoulli(
                        logits=torch.FloatTensor(
                            [
                                [
                                    [2.0, 3.0, 4.0],
                                    [5.0, 7.0, 9.0],
                                    [8.0, 1.0, 4.0],
                                ]
                            ]
                        )
                    ),
                    "regression_col": torch.distributions.Bernoulli(
                        logits=torch.FloatTensor(
                            [
                                [
                                    [5.0, 6.0, 7.0, 8.0],
                                    [2.0, 4.0, 6.0, 0.0],
                                    [7.0, 0.0, 3.0, 6.0],
                                ]
                            ]
                        )
                    ),
                },
                "want_labels": {
                    # Labels ignore event_mask, and only respect data mask and dynamic_measurement_indices, so
                    # these are unchanged from the prior test.
                    "event_type": torch.LongTensor(
                        [
                            [0, 1, 1],
                        ]
                    ),
                    "multi_label_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0],
                                [0, 0, 1],
                                [0, 1, 1],
                            ]
                        ]
                    ),
                    "regression_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 1, 1, 1],
                            ]
                        ]
                    ),
                },
                # Losses should be modified to ignore the components of the events not valid for that
                # measurement type.
                "want_losses": {
                    # (logits, labels):
                    #   ([0.0, 1.0], 0), ([1.0, 3.0], 1) [MASKED], ([2.0, 5.0], 1).
                    # NLL =
                    # 1/3 * (
                    #    -math.log(math.exp(0)/(math.exp(0) + math.exp(1))) +
                    #    -math.log(math.exp(3)/(math.exp(1) + math.exp(3))) +
                    #    -math.log(math.exp(5)/(math.exp(2) + math.exp(5)))
                    # ) = 0.49625901671164574
                    "event_type": torch.tensor(0.49625901671164574),
                    # (logits, labels):
                    #   ([2, 3, 4], [0, 0, 0]),
                    #   WRONG_EVENT_TYPE ([5, 7, 9], [0, 0, 1]),
                    #   WRONG_EVENT_TYPE ([8, 1, 4], [0, 1, 1])
                    # NLL =
                    # 1 * (
                    #   1/3 * (
                    #     -math.log(1 - 1/(1 + math.exp(-2)))
                    #     -math.log(1 - 1/(1 + math.exp(-3)))
                    #     -math.log(1 - 1/(1 + math.exp(-4)))
                    #   ) + 0 * ( # WRONG EVENT TYPE
                    #     -math.log(1 - 1/(1 + math.exp(-5)))
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1/(1 + math.exp(-9)))
                    #   ) + 0 * ( # WRONG EVENT TYPE
                    #     -math.log(1 - 1/(1 + math.exp(-8)))
                    #     -math.log(1/(1 + math.exp(-1)))
                    #     -math.log(1/(1 + math.exp(-4)))
                    #   )
                    # ) = 3.064555096844842
                    "multi_label_col": torch.tensor(3.064555096844842),
                    # (logits, labels):
                    #   WRONG EVENT TYPE([5, 6, 7, 8], [0, 0, 0, 0]),
                    #   ([2, 4, 6, 0], [0, 0, 0, 0]),
                    #   ([7, 0, 3, 6], [0, 1, 1, 1])
                    # NLL =
                    # 1/2 * (
                    #   0 * ( # WRONG EVENT TYPE
                    #     -math.log(1 - 1/(1 + math.exp(-5)))
                    #     -math.log(1 - 1/(1 + math.exp(-6)))
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1 - 1/(1 + math.exp(-8)))
                    #   ) + 1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-2)))
                    #     -math.log(1 - 1/(1 + math.exp(-4)))
                    #     -math.log(1 - 1/(1 + math.exp(-6)))
                    #     -math.log(1 - 1/(1 + math.exp(-0)))
                    #   ) + 1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1/(1 + math.exp(-0)))
                    #     -math.log(1/(1 + math.exp(-3)))
                    #     -math.log(1/(1 + math.exp(-6)))
                    #   ) = 2.573227811047968
                    "regression_col": torch.tensor(2.573227811047968),
                },
            },
            {
                "message": "Model should ignore masked events when computing losses.",
                "batch": {
                    **BASE_BATCH,
                    "event_mask": torch.BoolTensor([[True, False, True]]),
                },
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 0.0],
                            [0.0, 2.0, 5.0, 8.0, 1.0, 4.0, 7.0, 0.0, 3.0, 6.0],
                        ],
                    ]
                ),
                "valid_measurements": {"event_type", "multi_label_col", "regression_col"},
                "want_dists": {
                    # All dists are of shape batch X seq X vocab size.
                    "event_type": torch.distributions.Categorical(
                        logits=torch.FloatTensor(
                            [
                                [
                                    [0.0, 1.0],
                                    [1.0, 3.0],
                                    [2.0, 5.0],
                                ]
                            ]
                        )
                    ),
                    "multi_label_col": torch.distributions.Bernoulli(
                        logits=torch.FloatTensor(
                            [
                                [
                                    [2.0, 3.0, 4.0],
                                    [5.0, 7.0, 9.0],
                                    [8.0, 1.0, 4.0],
                                ]
                            ]
                        )
                    ),
                    "regression_col": torch.distributions.Bernoulli(
                        logits=torch.FloatTensor(
                            [
                                [
                                    [5.0, 6.0, 7.0, 8.0],
                                    [2.0, 4.0, 6.0, 0.0],
                                    [7.0, 0.0, 3.0, 6.0],
                                ]
                            ]
                        )
                    ),
                },
                "want_labels": {
                    # Labels ignore event_mask, and only respect data mask and dynamic_measurement_indices, so
                    # these are unchanged from the prior test.
                    "event_type": torch.LongTensor(
                        [
                            [0, 1, 1],
                        ]
                    ),
                    "multi_label_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0],
                                [0, 0, 1],
                                [0, 1, 1],
                            ]
                        ]
                    ),
                    "regression_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 1, 1, 1],
                            ]
                        ]
                    ),
                },
                # Losses should be modified to ignore the components of the first event.
                "want_losses": {
                    # (logits, labels):
                    #   ([0.0, 1.0], 0), ([1.0, 3.0], 1) [MASKED], ([2.0, 5.0], 1).
                    # NLL =
                    # 1/2 * (
                    #   -math.log(math.exp(0)/(math.exp(0) + math.exp(1))) +
                    #   0*(-math.log(math.exp(3)/(math.exp(1) + math.exp(3)))) + # MASKED EVENT
                    #   -math.log(math.exp(5)/(math.exp(2) + math.exp(5)))
                    # ) = 0.6809245195459824
                    "event_type": torch.tensor(0.6809245195459824),
                    # (logits, labels):
                    #   ([2, 3, 4], [0, 0, 0]),
                    #   MASKED ([5, 7, 9], [0, 0, 1]),
                    #   ([8, 1, 4], [0, 1, 1])
                    # NLL =
                    # 1/2 * (
                    #   1/3 * (
                    #     -math.log(1 - 1/(1 + math.exp(-2)))
                    #     -math.log(1 - 1/(1 + math.exp(-3)))
                    #     -math.log(1 - 1/(1 + math.exp(-4)))
                    #   ) + 0 * ( # MASKED EVENT
                    #     -math.log(1 - 1/(1 + math.exp(-5)))
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1/(1 + math.exp(-9)))
                    #   ) + 1/3 * (
                    #     -math.log(1 - 1/(1 + math.exp(-8)))
                    #     -math.log(1/(1 + math.exp(-1)))
                    #     -math.log(1/(1 + math.exp(-4)))
                    #   )
                    # ) = 2.9209020520572953
                    "multi_label_col": torch.tensor(2.9209020520572953),
                    # (logits, labels):
                    #   ([5, 6, 7, 8], [0, 0, 0, 0]),
                    #   MASKED ([2, 4, 6, 0], [0, 0, 0, 0]),
                    #   ([7, 0, 3, 6], [0, 1, 1, 1])
                    # NLL =
                    # 1/2 * (
                    #   1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-5)))
                    #     -math.log(1 - 1/(1 + math.exp(-6)))
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1 - 1/(1 + math.exp(-8)))
                    #   ) + 0 * ( # MASKED EVENT
                    #     -math.log(1 - 1/(1 + math.exp(-2)))
                    #     -math.log(1 - 1/(1 + math.exp(-4)))
                    #     -math.log(1 - 1/(1 + math.exp(-6)))
                    #     -math.log(1 - 1/(1 + math.exp(-0)))
                    #   ) + 1/4 * (
                    #     -math.log(1 - 1/(1 + math.exp(-7)))
                    #     -math.log(1/(1 + math.exp(-0)))
                    #     -math.log(1/(1 + math.exp(-3)))
                    #     -math.log(1/(1 + math.exp(-6)))
                    #   ) = 4.2194449487723995
                    "regression_col": torch.tensor(4.2194449487723995),
                },
            },
            {
                "message": "Model should only process selected data types.",
                "batch": {**BASE_BATCH},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 0.0],
                            [0.0, 2.0, 5.0, 8.0, 1.0, 4.0, 7.0, 0.0, 3.0, 6.0],
                        ],
                    ]
                ),
                "valid_measurements": {"event_type"},
                "want_dists": {
                    "event_type": torch.distributions.Categorical(
                        logits=torch.FloatTensor(
                            [
                                [
                                    [0.0, 1.0],
                                    [1.0, 3.0],
                                    [2.0, 5.0],
                                ]
                            ]
                        )
                    ),
                },
                "want_labels": {"event_type": torch.LongTensor([[0, 1, 1]])},
                "want_losses": {
                    # (logits, label): ([0.0, 1.0], 0), ([1.0, 3.0], 1), ([2.0, 5.0], 1)
                    # NLL = 1/3 * (
                    #   -math.log(math.exp(0)/(math.exp(0) + math.exp(1))) +
                    #   -math.log(math.exp(3)/(math.exp(1) + math.exp(3))) +
                    #   -math.log(math.exp(5)/(math.exp(2) + math.exp(5)))
                    # )
                    "event_type": torch.tensor(0.49625901671164574),
                },
            },
            {
                "message": "Model should skip events for single label classification tasks with no label.",
                "batch": {
                    **BASE_BATCH,
                    "dynamic_measurement_indices": torch.LongTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [1, 2, 0, 0, 0, 0],
                                [1, 2, 2, 3, 3, 3],
                            ],
                        ]
                    ),
                    "dynamic_indices": torch.LongTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [2, 5, 0, 0, 0, 0],
                                [2, 4, 5, 7, 8, 9],
                            ],
                        ]
                    ),
                },
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 0.0],
                            [0.0, 2.0, 5.0, 8.0, 1.0, 4.0, 7.0, 0.0, 3.0, 6.0],
                        ],
                    ]
                ),
                "valid_measurements": {"event_type"},
                "want_dists": {
                    "event_type": torch.distributions.Categorical(
                        logits=torch.FloatTensor(
                            [
                                [
                                    [0.0, 1.0],
                                    [1.0, 3.0],
                                    [2.0, 5.0],
                                ]
                            ]
                        )
                    ),
                },
                "want_labels": {"event_type": torch.LongTensor([[0, 1, 1]])},
                "want_losses": {
                    # event_type has 2 pairs of (logit, label) across only the last two events (the first
                    # event is not measured): ([1.0, 3.0], 1), ([2.0, 5.0], 1).
                    # NLL = 1/2 * (
                    #   -math.log(math.exp(3)/(math.exp(1) + math.exp(3)))
                    #   -math.log(math.exp(5)/(math.exp(2) + math.exp(5)))
                    # ) = 0.08775768130835727
                    "event_type": torch.tensor(0.08775768130835727),
                },
            },
            {
                "message": "Model should give a loss of 0 when no events have a single label task observed.",
                "batch": {
                    **BASE_BATCH,
                    "dynamic_measurement_indices": torch.LongTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                            ],
                        ]
                    ),
                    "dynamic_indices": torch.LongTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                            ],
                        ]
                    ),
                },
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 0.0],
                            [0.0, 2.0, 5.0, 8.0, 1.0, 4.0, 7.0, 0.0, 3.0, 6.0],
                        ],
                    ]
                ),
                "valid_measurements": {"event_type"},
                "want_dists": {
                    "event_type": torch.distributions.Categorical(
                        logits=torch.FloatTensor(
                            [
                                [
                                    [0.0, 1.0],
                                    [1.0, 3.0],
                                    [2.0, 5.0],
                                ]
                            ]
                        )
                    ),
                },
                "want_labels": {"event_type": torch.LongTensor([[0, 0, 0]])},
                "want_losses": {
                    # event_type has no valid observations, so should return a loss of 0.
                    "event_type": torch.tensor(0.0),
                },
            },
        ]

        for C in cases:
            with self.subTest(C["message"]):
                C["batch"] = PytorchBatch(**C["batch"])
                config = StructuredTransformerConfig(
                    **{
                        **BASE_CONFIG_KWARGS,
                        **C.get("config_kwargs", {}),
                        "hidden_size": 10,
                    }
                )

                # TODO(mmd): The config right now assumes the passed vocabulary sizes sum to the total vocab
                # size, but the model assumes there is one extra universally unused vocab element up front, so
                # we need to adjust that.
                config.vocab_size = 10

                layer = GenerativeOutputLayerBase(config)
                layer.ClassificationLayer.weight = torch.nn.Parameter(torch.eye(10))
                layer.ClassificationLayer.bias = torch.nn.Parameter(
                    torch.zeros_like(layer.ClassificationLayer.bias)
                )

                if C.get("include_event_types_mask", False):
                    event_type_mask_per_measurement = layer.get_event_type_mask_per_measurement(
                        C["batch"]
                    )
                    got_losses, got_dists, got_labels = layer.get_classification_outputs(
                        batch=C["batch"],
                        encoded=C["encoded"],
                        valid_measurements=C["valid_measurements"],
                        event_type_mask_per_measurement=event_type_mask_per_measurement,
                    )
                else:
                    got_losses, got_dists, got_labels = layer.get_classification_outputs(
                        batch=C["batch"],
                        encoded=C["encoded"],
                        valid_measurements=C["valid_measurements"],
                    )

                self.assertNestedDictEqual(C["want_labels"], got_labels)
                self.assertNestedDictEqual(C["want_dists"], got_dists)
                self.assertNestedDictEqual(C["want_losses"], got_losses)

    def test_get_TTE_outputs(self):
        shared_config_kwargs = {
            **BASE_CONFIG_KWARGS,
            "hidden_size": 6,
        }
        generation_specific_config_kwargs = {
            "exponential": {"TTE_lognormal_generation_num_components": None},
            "log_normal_mixture": {"TTE_lognormal_generation_num_components": 2},
        }

        cases = [
            {
                "message": "Model should yield the correct outputs given inputs for an Exponential TTE.",
                "TTE_generation_layer_type": "exponential",
                "batch": {**BASE_BATCH},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
                            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                        ],
                    ]
                ),
                # `rate` is given by torch.nn.elu(layer.proj @ encoded):
                "want_dist": torch.distributions.Exponential(
                    rate=torch.FloatTensor([[1.0, 2.0, 3.0]]),
                ),
                "want_label": torch.FloatTensor([[2, 3]]),
                # The average log likelihood of the true observed time to event over valid pairs of unmasked
                # events (in this case the transitions between the first and second events and second and
                # third events) is given according to the PDF of the exponential distribution, shown below:
                # p(x) = rate * math.exp(-rate * x) for x greater than or equal to 0, and 0 otherwise.
                # Given our rates above, that means that we should expect a LL of
                # 1/2 * (
                #   math.log(1.0 * math.exp(-1.0 * 2.0))
                #   +math.log(2.0 * math.exp(-2.0 * 3.0))
                # ) = -3.6534264097200273
                "want_LL": torch.tensor(-3.6534264097200273),
            },
            {
                "message": "Model should yield the correct outputs given inputs for an LogNormalMixture.",
                "TTE_generation_layer_type": "log_normal_mixture",
                "batch": {**BASE_BATCH},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
                            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                        ],
                    ]
                ),
                # `the params are given by alternating positions.
                "want_dist": LogNormalMixtureDistribution(
                    locs=torch.Tensor(
                        [
                            [
                                [0, 3],
                                [1, 7],
                                [2, 8],
                            ]
                        ]
                    ),
                    log_scales=torch.Tensor(
                        [
                            [
                                [1, 4],
                                [3, 9],
                                [4, 10],
                            ]
                        ]
                    ),
                    log_weights=torch.Tensor(
                        [
                            [
                                [2, 5],
                                [5, 11],
                                [6, 12],
                            ]
                        ]
                    ),
                    mean_log_inter_time=0,
                    std_log_inter_time=1,
                ),
                "want_label": torch.FloatTensor([[2, 3]]),
                # The average log likelihood of the true observed time to event over valid pairs of unmasked
                # events (in this case the transitions between the first and second events and second and
                # third events) is given according to the weighted sum of the two component lognormal
                # distributions, given by their parameters above (recall that loc is the mu of the underlying
                # normal distribution and log_scale is the logarithm of the standard deviation of the
                # underlying normal distribution, and the two columns in the parameters above correspond to
                # the parameters for each component, with log_weights being the logits for the component
                # distributions):
                # See here for pdf: https://en.wikipedia.org/wiki/Log-normal_distribution
                # It's formula is
                # pdf(x) = (1/(x*math.exp(scale)*math.sqrt(2*math.pi))) *
                #          math.exp(-(math.log(x) - loc)**2/(2*(math.exp(scale)**2)))
                # LL = 1/2 * (
                #   math.log(
                #     math.exp(2)/(math.exp(2) + math.exp(5)) * (
                #       1/(2*math.exp(1)*math.sqrt(2*math.pi))*math.exp(
                #           -((math.log(2) - 0)**2)/(2*math.exp(1)**2)
                #          )
                #     ) + math.exp(5) / (math.exp(2) + math.exp(5)) * (
                #       1/(2*math.exp(4)*math.sqrt(2*math.pi))*math.exp(
                #            -((math.log(2) - 3)**2)/(2*math.exp(4)**2))
                #     )
                #   ) + math.log(
                #     math.exp(5)/(math.exp(11) + math.exp(5)) * (
                #       1/(3*math.exp(3)*math.sqrt(2*math.pi))*math.exp(
                #            -((math.log(3) - 1)**2)/(2*math.exp(3)**2))
                #     ) + math.exp(11) / (math.exp(11) + math.exp(5)) * (
                #       1/(3*math.exp(9)*math.sqrt(2*math.pi))*math.exp(
                #            -((math.log(3) - 7)**2)/(2*math.exp(9)**2))
                #     )
                #  )
                # ) = -7.6554941334115565
                "want_LL": torch.tensor(-7.6554941334115565),
            },
            {
                "message": "Model should respect event masking.",
                "TTE_generation_layer_type": "exponential",
                "batch": {
                    **BASE_BATCH,
                    # TODO(mmd): Were there only one valid event, the model would return a NaN Loss here, as
                    # opposed to just zeroing out that component for that patient. Is that desired?
                    "event_mask": torch.BoolTensor([[True, True, False]]),
                },
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
                            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                        ],
                    ]
                ),
                # `rate` is given by torch.nn.elu(layer.proj @ encoded):
                "want_dist": torch.distributions.Exponential(
                    rate=torch.FloatTensor([[1.0, 2.0, 3.0]]),
                ),
                # The labels are padded with 1s in locations where the event_mask is False. This is necessary
                # as multiple batch elements may have different #s of valid events.
                "want_label": torch.FloatTensor([[2, 1]]),
                # In this case, only the transition between the first and second event is valid, so the LL
                # should be:
                # math.log(1.0 * math.exp(-1.0 * 2.0)) = -2.
                "want_LL": torch.tensor(-2.0),
            },
        ]

        for C in cases:
            with self.subTest(C["message"]):
                C["batch"] = PytorchBatch(**C["batch"])
                config = StructuredTransformerConfig(
                    **shared_config_kwargs,
                    **generation_specific_config_kwargs[C["TTE_generation_layer_type"]],
                    TTE_generation_layer_type=C["TTE_generation_layer_type"],
                )

                # TODO(mmd): The config right now assumes the passed vocabulary sizes sum to the total vocab
                # size, but the model assumes there is one extra universally unused vocab element up front, so
                # we need to adjust that.
                config.vocab_size = 10

                layer = GenerativeOutputLayerBase(config)
                layer.TTE_layer.proj.bias = torch.nn.Parameter(
                    torch.zeros_like(layer.TTE_layer.proj.bias)
                )

                if C["TTE_generation_layer_type"] == "exponential":
                    layer.TTE_layer.proj.weight = torch.nn.Parameter(
                        torch.Tensor([[1, 0, 0, 0, 0, 0]])
                    )
                elif C["TTE_generation_layer_type"] == "log_normal_mixture":
                    layer.TTE_layer.proj.weight = torch.nn.Parameter(torch.eye(6))
                else:
                    raise ValueError(
                        f"TTE_generation_layer_type of {C['TTE_generation_layer_type']} unrecognized."
                    )

                got_LL, got_dist, got_label = layer.get_TTE_outputs(
                    batch=C["batch"], encoded=C["encoded"]
                )

                self.assertEqual(C["want_label"], got_label)
                self.assertDistributionsEqual(C["want_dist"], got_dist)
                self.assertEqual(C["want_LL"], got_LL, "Log likelihoods differ")

    def test_get_regression_outputs(self):
        cases = [
            {
                "message": "Model should yield the correct outputs given inputs.",
                "batch": {
                    **BASE_BATCH,
                    # Replicated here for clarity
                    "dynamic_measurement_indices": torch.LongTensor(
                        [
                            [
                                [1, 0, 0, 0, 0, 0],
                                [1, 2, 0, 0, 0, 0],
                                [1, 2, 2, 3, 3, 3],
                            ]
                        ]
                    ),
                    "dynamic_indices": torch.LongTensor(
                        [
                            [
                                [1, 0, 0, 0, 0, 0],
                                [2, 5, 0, 0, 0, 0],
                                [2, 4, 5, 7, 8, 9],
                            ]
                        ]
                    ),
                    "dynamic_values_mask": torch.BoolTensor(
                        [
                            [
                                [False, False, False, False, False, False],
                                [False, False, False, False, False, False],
                                [False, False, False, True, True, True],
                            ]
                        ]
                    ),
                    "dynamic_values": torch.Tensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1.1, -1.1, 0.0],
                            ]
                        ]
                    ),
                },
                "valid_measurements": {"regression_col"},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
                            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                        ],
                    ]
                ),
                # `rate` is given by torch.nn.elu(layer.proj @ encoded):
                "want_dists": {
                    # The parameters are a little weird here, because of how the layer works and because we
                    # use 0 as an index to indicate "not present" for masked data elements. This, plus the
                    # gather operation, means that the output parameters will have the parameters for the
                    # first regression target (index zero) at all masked positions, which causes the odd
                    # structure here. The only parameters that really matter are in the unmasked data
                    # positions, which are the last three of the last batch element.
                    # Further, recall that scale is elu(proj(encoded)) + 1, so there will be a plus one
                    # modifier here too.
                    "regression_col": torch.distributions.Normal(
                        loc=torch.FloatTensor(
                            [
                                [
                                    [0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1],
                                    [2, 2, 2, 6, 10, 14],
                                ]
                            ]
                        ),
                        scale=torch.FloatTensor(
                            [
                                [
                                    [2, 2, 2, 2, 2, 2],
                                    [4, 4, 4, 4, 4, 4],
                                    [5, 5, 5, 9, 13, 17],
                                ]
                            ]
                        ),
                    ),
                },
                "want_labels": {
                    "regression_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1.1, -1.1, 0.0],
                            ]
                        ]
                    ),
                },
                "want_losses": {
                    # The average NLL over present events per patient, then over patients, is given as
                    # follows:
                    # 1/1 * (
                    #   1/3 * (
                    #     -math.log(1/(9*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((1.1 - 6)/9)**2))
                    #     -math.log(1/(13*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((-1.1 - 10)/13)**2))
                    #     -math.log(1/(17*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((0 - 14)/17)**2))
                    #   )
                    # ) = 3.734679909416965
                    "regression_col": torch.tensor(3.734679909416965),
                },
                "want_indices": {
                    "regression_col": torch.LongTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 2, 3],
                            ]
                        ]
                    ),
                },
            },
            {
                # This test is a little wonky as it is including events in cases where they aren't actually
                # supposed to be possible to measure, but it is still good to validate the functionality is
                # working.
                "message": "Model should only include losses over valid event types.",
                "batch": {
                    **BASE_BATCH,
                    # Replicated here for clarity
                    "dynamic_measurement_indices": torch.LongTensor(
                        [
                            [
                                [1, 0, 0, 0, 0, 0],
                                [1, 2, 3, 3, 0, 0],
                                [1, 2, 2, 3, 3, 3],
                            ]
                        ]
                    ),
                    "dynamic_indices": torch.LongTensor(
                        [
                            [
                                [1, 0, 0, 0, 0, 0],
                                [1, 5, 8, 9, 0, 0],
                                [2, 4, 5, 7, 8, 9],
                            ]
                        ]
                    ),
                    "dynamic_values_mask": torch.BoolTensor(
                        [
                            [
                                [False, False, False, False, False, False],
                                [False, False, True, True, False, False],
                                [False, False, False, True, True, True],
                            ]
                        ]
                    ),
                    "dynamic_values": torch.Tensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 1.1, -1.1, 0, 0],
                                [0, 0, 0, 1.1, -1.1, 0.0],
                            ]
                        ]
                    ),
                },
                # Recall these are our measurement types
                # TEST_MEASUREMENTS_IDXMAP = {
                #     'event_type': 1,
                #     'multi_label_col': 2,
                #     'regression_col': 3,
                # }
                # And that we have two options for event types:
                # TEST_EVENT_TYPES_IDXMAP = {
                #     'event_A': 0,
                #     'event_B': 1,
                # }
                "config_kwargs": {
                    "event_types_per_measurement": {
                        "event_type": ["event_A", "event_B"],
                        "multi_label_col": ["event_A"],
                        "regression_col": ["event_B"],
                    },
                },
                "include_event_types_mask": True,
                "valid_measurements": {"regression_col"},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
                            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                        ],
                    ]
                ),
                # `rate` is given by torch.nn.elu(layer.proj @ encoded):
                "want_dists": {
                    # The parameters are a little weird here, because of how the layer works and because we
                    # use 0 as an index to indicate "not present" for masked data elements. This, plus the
                    # gather operation, means that the output parameters will have the parameters for the
                    # first regression target (index zero) at all masked positions, which causes the odd
                    # structure here. The only parameters that really matter are in the unmasked data
                    # positions, which are the last three of the last batch element.
                    # Further, recall that scale is elu(proj(encoded)) + 1, so there will be a plus one
                    # modifier here too.
                    "regression_col": torch.distributions.Normal(
                        loc=torch.FloatTensor(
                            [
                                [
                                    [0, 0, 0, 0, 0, 0],
                                    [1, 1, 9, 13, 1, 1],
                                    [2, 2, 2, 6, 10, 14],
                                ]
                            ]
                        ),
                        scale=torch.FloatTensor(
                            [
                                [
                                    [2, 2, 2, 2, 2, 2],
                                    [4, 4, 12, 16, 4, 4],
                                    [5, 5, 5, 9, 13, 17],
                                ]
                            ]
                        ),
                    ),
                },
                "want_labels": {
                    "regression_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 1.1, -1.1, 0, 0],
                                [0, 0, 0, 1.1, -1.1, 0.0],
                            ]
                        ]
                    ),
                },
                "want_losses": {
                    # The average NLL over present, valid typed events per patient, then over patients, is
                    # given as follows (we only process the final event as the former is the wrong type):
                    # 1/1 * (
                    #   1/3 * (
                    #     -math.log(1/(9*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((1.1 - 6)/9)**2))
                    #     -math.log(1/(13*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((-1.1 - 10)/13)**2))
                    #     -math.log(1/(17*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((0 - 14)/17)**2))
                    #   )
                    # ) = 3.734679909416965
                    "regression_col": torch.tensor(3.734679909416965),
                },
                "want_indices": {
                    "regression_col": torch.LongTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 2, 3, 0, 0],
                                [0, 0, 0, 1, 2, 3],
                            ]
                        ]
                    ),
                },
            },
            {
                "message": "Model should appropriately average losses over events and data elements (1).",
                "batch": {
                    **BASE_BATCH,
                    "event_mask": torch.BoolTensor([[True, True, False]]),
                    "dynamic_measurement_indices": torch.LongTensor(
                        [
                            [
                                [1, 3, 3, 0, 0, 0],
                                [1, 3, 3, 3, 3, 0],
                                [1, 3, 3, 3, 3, 3],
                            ]
                        ]
                    ),
                    "dynamic_indices": torch.LongTensor(
                        [
                            [
                                [1, 7, 9, 0, 0, 0],
                                [2, 6, 7, 8, 7, 0],
                                [2, 7, 7, 7, 8, 9],
                            ]
                        ]
                    ),
                    "dynamic_values_mask": torch.BoolTensor(
                        [
                            [
                                [False, True, True, False, False, False],
                                [False, True, True, True, True, False],
                                [False, True, True, True, True, True],
                            ]
                        ]
                    ),
                    "dynamic_values": torch.Tensor(
                        [
                            [
                                [np.NaN, 2.5, 3.8, np.NaN, np.NaN, np.NaN],
                                [np.NaN, -1.2, 2.0, 4.5, -4.0, np.NaN],
                                [np.NaN, -1.2, 2.0, 4.5, -4.0, -5.0],
                            ]
                        ]
                    ),
                },
                "valid_measurements": {"regression_col"},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
                            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                        ],
                    ]
                ),
                # `rate` is given by torch.nn.elu(layer.proj @ encoded):
                "want_dists": {
                    "regression_col": torch.distributions.Normal(
                        loc=torch.FloatTensor(
                            [
                                [
                                    [0, 2, 6, 0, 0, 0],
                                    [1, 1, 5, 9, 5, 1],
                                    [2, 6, 6, 6, 10, 14],
                                ]
                            ]
                        ),
                        scale=torch.FloatTensor(
                            [
                                [
                                    [2, 4, 8, 2, 2, 2],
                                    [4, 4, 8, 12, 8, 4],
                                    [5, 9, 9, 9, 13, 17],
                                ]
                            ]
                        ),
                    ),
                },
                "want_labels": {
                    "regression_col": torch.FloatTensor(
                        [
                            [
                                [0, 2.5, 3.8, 0, 0, 0],
                                [0, -1.2, 2.0, 4.5, -4.0, 0],
                                [0, -1.2, 2.0, 4.5, -4.0, -5.0],
                            ]
                        ]
                    ),
                },
                "want_losses": {
                    # The average NLL over present events per patient, then over patients, is given as
                    # follows:
                    # 1/2 * (
                    #   1/2 * (
                    #     -math.log(1/(4*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((2.5 - 2)/4)**2))
                    #     -math.log(1/(8*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((3.8 - 6)/8)**2))
                    #   ) + 1/4 * (
                    #     -math.log(1/(4*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((-1.2 - 1)/4)**2))
                    #     -math.log(1/(8*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((2 - 5)/8)**2))
                    #     -math.log(1/(12*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((4.5 - 9)/12)**2))
                    #     -math.log(1/(8*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((-4 - 5)/8)**2))
                    #   )
                    # ) = 2.91612520818805
                    "regression_col": torch.tensor(2.91612520818805),
                },
                "want_indices": {
                    "regression_col": torch.LongTensor(
                        [
                            [
                                [0, 1, 3, 0, 0, 0],
                                [0, 0, 1, 2, 1, 0],
                                [0, 1, 1, 1, 2, 3],
                            ]
                        ]
                    ),
                },
            },
            {
                "message": "Model should appropriately average losses over events and data elements (2).",
                "batch": {
                    **BASE_BATCH,
                    "event_mask": torch.BoolTensor([[True, True, True]]),
                    "dynamic_measurement_indices": torch.LongTensor(
                        [
                            [
                                [1, 3, 3, 0, 0, 0],
                                [1, 3, 3, 3, 3, 0],
                                [1, 2, 2, 2, 2, 2],
                            ]
                        ]
                    ),
                    "dynamic_indices": torch.LongTensor(
                        [
                            [
                                [1, 7, 9, 0, 0, 0],
                                [2, 6, 7, 8, 7, 0],
                                [2, 4, 5, 4, 5, 4],
                            ]
                        ]
                    ),
                    "dynamic_values_mask": torch.BoolTensor(
                        [
                            [
                                [False, True, True, False, False, False],
                                [False, True, True, True, True, False],
                                [False, False, False, False, False, True],
                            ]
                        ]
                    ),
                    "dynamic_values": torch.Tensor(
                        [
                            [
                                [np.NaN, 2.5, 3.8, np.NaN, np.NaN, np.NaN],
                                [np.NaN, -1.2, 2.0, 4.5, -4.0, np.NaN],
                                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                            ]
                        ]
                    ),
                },
                "valid_measurements": {"regression_col"},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
                            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                        ],
                    ]
                ),
                # `rate` is given by torch.nn.elu(layer.proj @ encoded):
                "want_dists": {
                    "regression_col": torch.distributions.Normal(
                        loc=torch.FloatTensor(
                            [
                                [
                                    [0, 2, 6, 0, 0, 0],
                                    [1, 1, 5, 9, 5, 1],
                                    [2, 2, 2, 2, 2, 2],
                                ]
                            ]
                        ),
                        scale=torch.FloatTensor(
                            [
                                [
                                    [2, 4, 8, 2, 2, 2],
                                    [4, 4, 8, 12, 8, 4],
                                    [5, 5, 5, 5, 5, 5],
                                ]
                            ]
                        ),
                    ),
                },
                "want_labels": {
                    "regression_col": torch.FloatTensor(
                        [
                            [
                                [0, 2.5, 3.8, 0, 0, 0],
                                [0, -1.2, 2.0, 4.5, -4.0, 0],
                                [0, 0, 0, 0, 0, 0],
                            ]
                        ]
                    ),
                },
                "want_losses": {
                    # The average NLL over present events with a regression label per patient, then over
                    # patients, is given as
                    # follows:
                    # 1/2 * (
                    #   1/2 * (
                    #     -math.log(1/(4*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((2.5 - 2)/4)**2))
                    #     -math.log(1/(8*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((3.8 - 6)/8)**2))
                    #   ) + 1/4 * (
                    #     -math.log(1/(4*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((-1.2 - 1)/4)**2))
                    #     -math.log(1/(8*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((2 - 5)/8)**2))
                    #     -math.log(1/(12*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((4.5 - 9)/12)**2))
                    #     -math.log(1/(8*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((-4 - 5)/8)**2))
                    #   )
                    # ) = 2.91612520818805
                    "regression_col": torch.tensor(2.91612520818805),
                },
                "want_indices": {
                    "regression_col": torch.LongTensor(
                        [
                            [
                                [0, 1, 3, 0, 0, 0],
                                [0, 0, 1, 2, 1, 0],
                                [0, 0, 0, 0, 0, 0],
                            ]
                        ]
                    ),
                },
            },
            {
                "message": "Model should only return a loss where `dynamic_values_mask` is True.",
                "batch": {
                    **BASE_BATCH,
                    # Replicated here for clarity
                    "dynamic_measurement_indices": torch.LongTensor(
                        [
                            [
                                [1, 0, 0, 0, 0, 0],
                                [1, 2, 0, 0, 0, 0],
                                [1, 2, 2, 3, 3, 3],
                            ]
                        ]
                    ),
                    "dynamic_indices": torch.LongTensor(
                        [
                            [
                                [1, 0, 0, 0, 0, 0],
                                [2, 5, 0, 0, 0, 0],
                                [2, 4, 5, 7, 8, 9],
                            ]
                        ]
                    ),
                    "dynamic_values_mask": torch.BoolTensor(
                        [
                            [
                                [False, False, False, False, False, False],
                                [False, False, False, False, False, False],
                                [False, False, False, True, True, False],
                            ]
                        ]
                    ),
                    "dynamic_values": torch.Tensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1.1, -1.1, 0],
                            ]
                        ]
                    ),
                },
                "valid_measurements": {"regression_col"},
                "encoded": torch.Tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
                            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                        ],
                    ]
                ),
                # `rate` is given by torch.nn.elu(layer.proj @ encoded):
                "want_dists": {
                    "regression_col": torch.distributions.Normal(
                        loc=torch.FloatTensor(
                            [
                                [
                                    [0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1],
                                    [2, 2, 2, 6, 10, 2],
                                ]
                            ]
                        ),
                        scale=torch.FloatTensor(
                            [
                                [
                                    [2, 2, 2, 2, 2, 2],
                                    [4, 4, 4, 4, 4, 4],
                                    [5, 5, 5, 9, 13, 5],
                                ]
                            ]
                        ),
                    ),
                },
                "want_labels": {
                    "regression_col": torch.FloatTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1.1, -1.1, 0.0],
                            ]
                        ]
                    ),
                },
                "want_losses": {
                    # The average NLL over present events per patient, then over patients, *should be* given
                    # as follows:
                    # 1/1 * (
                    #   1/2 * (
                    #     -math.log(1/(9*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((1.1 - 6)/9)**2))
                    #     -math.log(1/(13*math.sqrt(2*math.pi)) * math.exp((-1/2) * ((-1.1 - 10)/13)**2))
                    #   )
                    # ) = 3.556393752484623
                    "regression_col": torch.tensor(3.556393752484623),
                },
                "want_indices": {
                    "regression_col": torch.LongTensor(
                        [
                            [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 2, 0],
                            ]
                        ]
                    ),
                },
            },
        ]

        for C in cases:
            with self.subTest(C["message"]):
                C["batch"] = PytorchBatch(**C["batch"])
                config = StructuredTransformerConfig(
                    **{
                        **BASE_CONFIG_KWARGS,
                        **C.get("config_kwargs", {}),
                        "hidden_size": 8,  # 2 * number of regression components (4)
                    }
                )

                # TODO(mmd): The config right now assumes the passed vocabulary sizes sum to the total vocab
                # size, but the model assumes there is one extra universally unused vocab element up front, so
                # we need to adjust that.
                config.vocab_size = 10

                layer = GenerativeOutputLayerBase(config)
                layer.regression_layers["regression_col"].proj.weight = torch.nn.Parameter(
                    torch.eye(8)
                )
                layer.regression_layers["regression_col"].proj.bias = torch.nn.Parameter(
                    torch.zeros_like(layer.regression_layers["regression_col"].proj.bias)
                )

                if C.get("include_event_types_mask", False):
                    event_type_mask_per_measurement = layer.get_event_type_mask_per_measurement(
                        C["batch"]
                    )
                    got_losses, got_dists, got_labels, got_indices = layer.get_regression_outputs(
                        batch=C["batch"],
                        encoded=C["encoded"],
                        valid_measurements=C["valid_measurements"],
                        event_type_mask_per_measurement=event_type_mask_per_measurement,
                    )
                else:
                    got_losses, got_dists, got_labels, got_indices = layer.get_regression_outputs(
                        batch=C["batch"],
                        encoded=C["encoded"],
                        valid_measurements=C["valid_measurements"],
                    )

                self.assertNestedDictEqual(C["want_labels"], got_labels, "Labels differ")
                self.assertNestedDictEqual(C["want_dists"], got_dists, "Distributions differ")
                self.assertNestedDictEqual(C["want_indices"], got_indices, "Indices differ")
                self.assertNestedDictEqual(C["want_losses"], got_losses, "Losses differ")


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
                    do_add_temporal_position_embeddings_to_data_embeddings=None,
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
                    do_add_temporal_position_embeddings_to_data_embeddings=None,
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

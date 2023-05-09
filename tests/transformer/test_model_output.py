import sys

sys.path.append("../..")

import unittest

import numpy as np
import pandas as pd
import torch
from pytorch_lognormal_mixture import LogNormalMixtureDistribution

from EventStream.data.types import DataModality, PytorchBatch, TemporalityType
from EventStream.data.time_dependent_functor import AgeFunctor, TimeOfDayFunctor
from EventStream.data.vocabulary import Vocabulary
from EventStream.data.config import MeasurementConfig
from EventStream.transformer.config import (
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
)
from EventStream.transformer.model import (
    ESTForStreamClassification,
    GenerativeOutputLayer,
)

from ..mixins import MLTypeEqualityCheckableMixin

TEST_MEASUREMENT_CONFIGS = {
    'static_clf': MeasurementConfig(
        temporality=TemporalityType.STATIC,
        modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
        vocabulary=Vocabulary(
            vocabulary=['UNK', 'static_clf_1', 'static_clf_2'],
            obs_frequencies=[0.1, 0.5, 0.4],
        ),
    ),
    'static_reg': MeasurementConfig(
        temporality=TemporalityType.STATIC,
        modality=DataModality.UNIVARIATE_REGRESSION,
        vocabulary=None,
        measurement_metadata=pd.Series({
            'value_type': 'float',
            'normalizer': {'mean_': 1.0, 'std_': 2.0},
            'outlier_model': {'thresh_large_': 3.0, 'thresh_small_': -1.0},
        }),
    ),
    'age': MeasurementConfig(
        temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
        functor=AgeFunctor('dob'),
        vocabulary=None,
        measurement_metadata=pd.Series({
            'value_type': 'float',
            'normalizer': {'mean_': 40.0, 'std_': 10.0},
            'outlier_model': {'thresh_large_': 90., 'thresh_small_': 18.},
        }),
    ),
    'tod': MeasurementConfig(
        temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
        functor=TimeOfDayFunctor(),
        vocabulary=Vocabulary(
            vocabulary=['UNK', 'EARLY_AM', 'LATE_PM', 'AM', 'PM'],
            obs_frequencies=[0.1, 0.3, 0.22, 0.2, 0.18],
        ),
    ),
    'dynamic_single_label_clf': MeasurementConfig(
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
        vocabulary=Vocabulary(
            vocabulary=['UNK', 'dynamic_single_label_1', 'dynamic_single_label_2'],
            obs_frequencies=[0.1, 0.5, 0.4],
        ),
    ),
    'dynamic_multi_label_clf': MeasurementConfig(
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.MULTI_LABEL_CLASSIFICATION,
        vocabulary=Vocabulary(
            vocabulary=['UNK', 'dynamic_multi_label_1', 'dynamic_multi_label_2', 'dynamic_multi_label_3'],
            obs_frequencies=[0.1, 0.5, 0.25, 0.15],
        ),
    ),
    'dynamic_univariate_reg': MeasurementConfig(
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.UNIVARIATE_REGRESSION,
        vocabulary=None,
        measurement_metadata=pd.Series({
            'value_type': 'float',
            'normalizer': {'mean_': 2.0, 'std_': 3.0},
            'outlier_model': {'thresh_large_': 7.0, 'thresh_small_': -3.0},
        }),
    ),
    'dynamic_multivariate_reg': MeasurementConfig(
        temporality=TemporalityType.DYNAMIC,
        modality=DataModality.MULTIVARIATE_REGRESSION,
        values_column='dynamic_multivariate_reg_values',
        vocabulary=Vocabulary(
            vocabulary=['UNK', 'dynamic_multivariate_reg_1', 'dynamic_multivariate_reg_2'],
            obs_frequencies=[0.1, 0.5, 0.4],
        ),
        measurement_metadata=pd.DataFrame(
            {
                'value_type': ['float', 'float'],
                'normalizer': [{'mean_': 2.0, 'std_': 3.0}, {'mean_': 4.0, 'std_': 5.0}],
                'outlier_model': [
                    {'thresh_large_': 7.0, 'thresh_small_': -3.0},
                    {'thresh_large_': 9.0, 'thresh_small_': -5.0}
                ],
            },
            index=pd.Index(
                ['dynamic_multivariate_reg_1', 'dynamic_multivariate_reg_2'], name='dynamic_multivariate_reg'
            ),
        ),
    ),
}

TEST_MEASUREMENTS_PER_GEN_MODE = {
    DataModality.SINGLE_LABEL_CLASSIFICATION: ["event_type", "dynamic_single_label_clf"],
    DataModality.MULTI_LABEL_CLASSIFICATION: ["dynamic_multi_label_clf", "dynamic_multivariate_reg"],
    DataModality.MULTIVARIATE_REGRESSION: ["dynamic_multivariate_reg"],
    DataModality.UNIVARIATE_REGRESSION: ["dynamic_univariate_reg"],
}
TEST_MEASUREMENTS_IDXMAP = {
    "event_type": 1,
    "static_clf": 2,
    "static_reg": 3,
    "age": 4,
    "tod": 5,
    "dynamic_single_label_clf": 6,
    "dynamic_multi_label_clf": 7,
    "dynamic_univariate_reg": 8,
    "dynamic_multivariate_reg": 9,
}
# These are all including the 'UNK' tokens. So, e.g., there are 2 real options for 'event_type'.
TEST_VOCAB_SIZES_BY_MEASUREMENT = {
    "event_type": 2,
    "static_clf": 3,
    "static_reg": 1,
    "age": 1,
    "tod": 5,
    "dynamic_single_label_clf": 3,
    "dynamic_multi_label_clf": 4,
    "dynamic_univariate_reg": 1,
    "dynamic_multivariate_reg": 3,
}
TEST_VOCAB_OFFSETS_BY_MEASUREMENT = {
    "event_type": 1,
    "static_clf": 3,
    "static_reg": 6,
    "age": 7,
    "tod": 8,
    "dynamic_single_label_clf": 13,
    "dynamic_multi_label_clf": 16,
    "dynamic_univariate_reg": 20,
    "dynamic_multivariate_reg": 21,
}
TEST_EVENT_TYPES_IDXMAP = {
    "event_A": 0,
    "event_B": 1,
}

UNIFIED_VOCABULARY = {
    "event_type": ["event_A", "event_B"],
    "static_clf": ["UNK", "static_clf_1", "static_clf_2"],
    "static_reg": None,
    "age": None,
    "tod": ['UNK', 'EARLY_AM', 'LATE_PM', 'AM', 'PM'],
    "dynamic_single_label_clf": ["UNK", "dynamic_single_label_1", "dynamic_single_label_2"],
    "dynamic_multi_label_clf": ["UNK", "dynamic_multi_label_1", "dynamic_multi_label_2", "dynamic_multi_label_3"],
    "dynamic_univariate_reg": None,
    "dynamic_multivariate_reg": ["UNK", "dynamic_multivariate_reg_1", "dynamic_multivariate_reg_2"],
}
UNIFIED_IDXMAP = {
    "event_type": {"event_A": 1, "event_B": 2},
    "static_clf": {"UNK": 3, "static_clf_1": 4, "static_clf_2": 5},
    "static_reg": {"UNK": 7, "static_reg_1": 8},
}

BASE_BATCH = {
    "event_mask": torch.BoolTensor([
        [False, True, True],
        [True, True, True]
    ]),
    "time_delta": torch.FloatTensor([
        [1, 2, 1],
        [1, 4, 1],
    ]),
    "static_indices": torch.LongTensor([
        [1, 2, 3],
        [1, 2, 3],
    ]),
    "static_measurement_indices": torch.FloatTensor([
        [1, 2, 3],
        [1, 2, 3],
    ]),
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

class TestGenerativeSequenceModelSamples(MLTypeEqualityCheckableMixin, unittest.TestCase):
    """Tests the OutputLayer."""

    def test_build_new_batch_element(self):


    def build_new_batch_element(
        self,
        batch: PytorchBatch,
        config: StructuredTransformerConfig,
        static_data: pd.DataFrame | None = None,
    ) -> tuple[
        torch.FloatTensor,
        torch.BoolTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.FloatTensor,
        torch.BoolTensor,
    ]:
        """This function is used for generation, and builds a new batch element from the prediction
        sample in this object."""

        # Add data elements (indices, values, types, values_mask)
        dynamic_measurement_indices = []
        dynamic_indices = []
        dynamic_values = []
        dynamic_values_mask = []

        # Add event_mask
        event_mask = self.event_mask

        # Add time-dependent values if present.
        for m, cfg in config.measurement_configs.items():
            if cfg.temporality != TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                continue
            if cfg.modality == DataModality.DROPPED:
                continue

            # Produce the functional, time-dependent outputs.

            # TODO(mmd): This may be wrong in some cases! Don't know the actual start time as it is
            # initialized to zero!
            fn = cfg.functor

            is_meas_map = (
                batch.dynamic_measurement_indices[:, -1, :] == config.measurements_idxmap[m]
            )
            indices = batch.dynamic_indices[:, -1, :]
            values = batch.dynamic_values[:, -1, :]
            values_mask = batch.dynamic_values_mask[:, -1, :]

            # We sum(-1) here as there must be exactly one time-dependent-event observation of a given type
            # per event, by definition.
            indices = torch.where(is_meas_map, indices, torch.zeros_like(indices)).sum(-1)
            vals = torch.where(is_meas_map & values_mask, values, torch.zeros_like(values)).sum(-1)

            offset = config.vocab_offsets_by_measurement[m]
            new_indices, new_values = fn.update_from_prior_timepoint(
                prior_indices=indices - offset,
                prior_values=vals,
                new_delta=self.time_to_event,
                new_time=batch.start_time + self.time_to_event,
                vocab=cfg.vocabulary,
                measurement_metadata=cfg.measurement_metadata,
            )

            new_indices = (new_indices + offset).unsqueeze(-1)
            new_values = new_values.unsqueeze(-1)
            new_measurement_indices = config.measurements_idxmap[m] * torch.ones_like(new_indices)

            dynamic_indices.append(new_indices)
            dynamic_values_mask.append(~torch.isnan(new_values))
            dynamic_values.append(torch.nan_to_num(new_values, 0))
            dynamic_measurement_indices.append(new_measurement_indices)

        dynamic_indices = torch.cat(dynamic_indices, 1)
        dynamic_measurement_indices = torch.cat(dynamic_measurement_indices, 1)
        dynamic_values = torch.cat(dynamic_values, 1)
        dynamic_values_mask = torch.cat(dynamic_values_mask, 1)

        return (
            self.time_to_event,
            event_mask,
            dynamic_indices,
            dynamic_measurement_indices,
            dynamic_values,
            dynamic_values_mask,
        )





if __name__ == "__main__":
    unittest.main()

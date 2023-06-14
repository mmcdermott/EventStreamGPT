import sys

sys.path.append("../..")

import unittest

from EventStream.data.types import DataModality
from EventStream.transformer.config import (
    OptimizationConfig,
    StructuredTransformerConfig,
)
from EventStream.transformer.lightning_modules.generative_modeling import (
    ESTForGenerativeSequenceModelingLM,
)

from ...utils import MLTypeEqualityCheckableMixin

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


class TestESTForGenerativeSequenceModelingLM(MLTypeEqualityCheckableMixin, unittest.TestCase):
    def test_constructs(self):
        """Tests that the Lightning Module constructs given default configuration options."""
        config = StructuredTransformerConfig(
            measurements_per_generative_mode=TEST_MEASUREMENTS_PER_GEN_MODE,
            vocab_sizes_by_measurement=TEST_VOCAB_SIZES_BY_MEASUREMENT,
            vocab_offsets_by_measurement=TEST_VOCAB_OFFSETS_BY_MEASUREMENT,
            measurements_idxmap=TEST_MEASUREMENTS_IDXMAP,
            measurements_per_dep_graph_level=TEST_MEASUREMENTS_PER_DEP_GRAPH_LEVEL,
        )
        optimization_config = OptimizationConfig()

        ESTForGenerativeSequenceModelingLM(
            config=config,
            optimization_config=optimization_config,
            metrics_config={},
        )


if __name__ == "__main__":
    unittest.main()

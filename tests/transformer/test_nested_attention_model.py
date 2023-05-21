import sys

sys.path.append("../..")

import unittest

from EventStream.transformer.config import (
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
)
from EventStream.transformer.nested_attention_model import (
    NAPPTForGenerativeSequenceModeling,
    NestedAttentionGenerativeOutputLayer,
)

from ..mixins import MLTypeEqualityCheckableMixin

DEFAULT_VALID_CONFIG_KWARGS = {
    "structured_event_processing_mode": StructuredEventProcessingMode.NESTED_ATTENTION,
    "measurements_per_dep_graph_level": [],
}


class TestNestedAttentionGenerativeOutputLayer(MLTypeEqualityCheckableMixin, unittest.TestCase):
    def test_constructs(self):
        NestedAttentionGenerativeOutputLayer(
            StructuredTransformerConfig(**DEFAULT_VALID_CONFIG_KWARGS)
        )

        with self.assertRaises(ValueError):
            NestedAttentionGenerativeOutputLayer(
                StructuredTransformerConfig(
                    structured_event_processing_mode=StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT
                )
            )


class TestNAPPTForGenerativeSequenceModeling(MLTypeEqualityCheckableMixin, unittest.TestCase):
    def test_constructs(self):
        NAPPTForGenerativeSequenceModeling(
            StructuredTransformerConfig(**DEFAULT_VALID_CONFIG_KWARGS)
        )

        with self.assertRaises(ValueError):
            NAPPTForGenerativeSequenceModeling(
                StructuredTransformerConfig(
                    structured_event_processing_mode=StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT
                )
            )


if __name__ == "__main__":
    unittest.main()

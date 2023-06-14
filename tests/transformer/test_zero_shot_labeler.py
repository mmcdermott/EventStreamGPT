import sys

sys.path.append("../..")

import unittest

import torch

from EventStream.data.types import PytorchBatch
from EventStream.transformer.config import StructuredTransformerConfig
from EventStream.transformer.zero_shot_labeler import Labeler


class LabelerMock(Labeler):
    def __call__(self, batch: PytorchBatch):
        return torch.LongTensor([0]), torch.BoolTensor([True])


class TestLabeler(unittest.TestCase):
    def test_constructs(self):
        LabelerMock(StructuredTransformerConfig())

        with self.assertRaises(TypeError):
            Labeler(StructuredTransformerConfig())


if __name__ == "__main__":
    unittest.main()

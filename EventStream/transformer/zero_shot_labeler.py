import abc

import torch

from ...data.types import PytorchBatch
from ..config import StructuredTransformerConfig


class Labeler(abc.ABC):
    def __init__(self, config: StructuredTransformerConfig):
        self.config = config

    @abc.abstractmethod
    def __call__(self, batch: PytorchBatch) -> tuple[torch.LongTensor, torch.BoolTensor]:
        raise NotImplementedError("Must be overwritten by a subclass!")

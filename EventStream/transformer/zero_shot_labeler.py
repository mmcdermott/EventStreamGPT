import torch
import abc
from .config import StructuredTransformerConfig
from ..data.types import PytorchBatch

class Labeler(abc.ABC):
    def __init__(self, input_seq_len: int, config: StructuredTransformerConfig):
        self.input_seq_len = input_seq_len
        self.config = config
        
    @abc.abstractmethod
    def __call__(self, batch: PytorchBatch) -> tuple[torch.LongTensor, torch.BoolTensor]:
        raise NotImplementedError(f"Must be overwritten by a subclass!")

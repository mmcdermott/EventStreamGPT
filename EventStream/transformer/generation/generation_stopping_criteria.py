# Sourced from
# https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/generation_stopping_criteria.py
# Then modified

from abc import ABC

from transformers.utils import add_start_docstrings

from ...data.types import PytorchBatch
from ..model_output import GenerativeSequenceModelPredictions

STOPPING_CRITERIA_INPUTS_DOCSTRING = r"""
    Args:
        `batch` (`PytorchBatch`): The input batch.
        `outputs` (`GenerativeSequenceModelPredictions`): The predicted outputs.
        kwargs:
            Additional stopping criteria specific kwargs.
    Return:
        `bool`. `False` indicates we should continue, `True` indicates we should stop.
"""


class StoppingCriteria(ABC):
    """Abstract base class for all stopping criteria that can be applied during generation."""

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, batch: PytorchBatch, outputs: GenerativeSequenceModelPredictions, **kwargs) -> bool:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class MaxLengthCriteria(StoppingCriteria):
    """This class can be used to stop generation whenever the full generated number of events exceeds
    `max_length`.

    Keep
    in mind for decoder-only type of transformers, this will include the initial prompted events.
    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of events.
    """

    def __init__(self, max_length: int):
        self.max_length = max_length

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, batch: PytorchBatch, outputs: GenerativeSequenceModelPredictions, **kwargs) -> bool:
        return batch.sequence_length >= self.max_length


class StoppingCriteriaList(list):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, batch: PytorchBatch, outputs: GenerativeSequenceModelPredictions, **kwargs) -> bool:
        return any(criteria(batch, outputs) for criteria in self)

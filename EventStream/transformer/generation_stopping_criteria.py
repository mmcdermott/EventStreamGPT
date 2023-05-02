# Sourced from https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/generation_stopping_criteria.py
# Then modified

import time
import warnings
from abc import ABC
from copy import deepcopy
from typing import Optional

from transformers.utils import add_start_docstrings

from ..data.types import PytorchBatch
from .model_output import GenerativeSequenceModelPredictions

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
    def __call__(
        self, batch: PytorchBatch, outputs: GenerativeSequenceModelPredictions, **kwargs
    ) -> bool:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class MaxLengthCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of events exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted events.
    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of events.
    """

    def __init__(self, max_length: int):
        self.max_length = max_length

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, batch: PytorchBatch, outputs: GenerativeSequenceModelPredictions, **kwargs
    ) -> bool:
        return batch.time.shape[-1] >= self.max_length

class MaxTimeCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the
    time will start being counted when you initialize this function. You can override this by passing an
    `initial_time`.
    Args:
        max_time (`float`):
            The maximum allowed time in seconds for the generation.
        initial_time (`float`, *optional*, defaults to `time.time()`):
            The start of the generation allowed time.
    """

    def __init__(self, max_time: float, initial_timestamp: Optional[float] = None):
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, batch: PytorchBatch, outputs: GenerativeSequenceModelPredictions, **kwargs
    ) -> bool:
        return time.time() - self.initial_timestamp > self.max_time


class StoppingCriteriaList(list):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, batch: PytorchBatch, outputs: GenerativeSequenceModelPredictions, **kwargs) -> bool:
        return any(criteria(batch, outputs) for criteria in self)

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
        return None

def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria

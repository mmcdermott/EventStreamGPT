# Sourced from https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/generation_logits_process.py
# Then modified.

# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

from transformers.utils import add_start_docstrings

from ..data.types import PytorchBatch
from .model_output import GenerativeSequenceModelPredictions

OUTPUTS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        batch (`PytorchBatch`):
            The input batch of the input sequence.
        output (`TODO(mmd)`):
            Prediction outputs for the final element in the batch.
        kwargs:
            Additional logits processor specific kwargs.
    Return:
        `TODO`: The processed prediction outputs.
"""


class OutputsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(OUTPUTS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, batch: PytorchBatch, outputs: GenerativeSequenceModelPredictions
    ) -> GenerativeSequenceModelPredictions:
        """Torch method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class OutputsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with
    multinomial sampling."""

    @add_start_docstrings(OUTPUTS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, batch: PytorchBatch, outputs: GenerativeSequenceModelPredictions
    ) -> GenerativeSequenceModelPredictions:
        """Torch method for warping logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class OutputsProcessorList(list):
    """This class can be used to create a list of [`OutputsProcessor`] or [`OutputsWarper`] to
    subsequently process a `outputs` input tensor. This class inherits from list and adds a
    specific *__call__* method to apply each.

    [`OutputsProcessor`] or [`OutputsWarper`] to the inputs.
    """

    @add_start_docstrings(OUTPUTS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, batch: PytorchBatch, outputs: GenerativeSequenceModelPredictions, **kwargs
    ) -> GenerativeSequenceModelPredictions:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                outputs = processor(batch, outputs, **kwargs)
            else:
                outputs = processor(batch, outputs)
        return outputs

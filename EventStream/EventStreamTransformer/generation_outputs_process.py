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

from ..EventStreamData.types import EventStreamPytorchBatch
from .model_output import GenerativeSequenceModelPredictions

OUTPUTS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        batch (`EventStreamPytorchBatch`):
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
        self, batch: EventStreamPytorchBatch, outputs: GenerativeSequenceModelPredictions
    ) -> GenerativeSequenceModelPredictions:
        """Torch method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class OutputsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    @add_start_docstrings(OUTPUTS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, batch: EventStreamPytorchBatch, outputs: GenerativeSequenceModelPredictions
    ) -> GenerativeSequenceModelPredictions:
        """Torch method for warping logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class OutputsProcessorList(list):
    """
    This class can be used to create a list of [`OutputsProcessor`] or [`OutputsWarper`] to subsequently process a
    `outputs` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`OutputsProcessor`] or [`OutputsWarper`] to the inputs.
    """

    @add_start_docstrings(OUTPUTS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, batch: EventStreamPytorchBatch, outputs: GenerativeSequenceModelPredictions, **kwargs
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


#class MinLengthOutputsProcessor(OutputsProcessor):
#    r"""
#    [`OutputsProcessor`] enforcing a min-length by setting EOS probability to 0.
#    Args:
#        min_length (`int`):
#            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
#        eos_token_id (`int`):
#            The id of the *end-of-sequence* token.
#    """
#
#    def __init__(self, min_length: int, eos_token_id: int):
#        if not isinstance(min_length, int) or min_length < 0:
#            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")
#
#        if not isinstance(eos_token_id, int) or eos_token_id < 0:
#            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")
#
#        self.min_length = min_length
#        self.eos_token_id = eos_token_id
#
#    def __call__(self, batch: EventStreamPytorchBatch, outputs: GenerativeSequenceModelPredictions) -> GenerativeSequenceModelPredictions:
#        cur_len = batch.shape[-1]
#        if cur_len < self.min_length:
#            outputs[:, self.eos_token_id] = -float("inf")
#        return outputs


#class TemperatureOutputsWarper(OutputsWarper):
#    r"""
#    [`OutputsWarper`] for temperature (exponential scaling output probability distribution).
#    Args:
#        temperature (`float`):
#            The value used to module the logits distribution.
#    """
#
#    def __init__(self, temperature: float):
#        if not isinstance(temperature, float) or not (temperature > 0):
#            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")
#
#        self.temperature = temperature
#
#    def __call__(self, batch: EventStreamPytorchBatch, outputs: GenerativeSequenceModelPredictions) -> GenerativeSequenceModelPredictions:
#        outputs = outputs / self.temperature
#        return outputs


#class TopPOutputsWarper(OutputsWarper):
#    """
#    [`OutputsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
#    Args:
#        top_p (`float`):
#            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
#            higher are kept for generation.
#        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
#            All filtered values will be set to this float value.
#        min_tokens_to_keep (`int`, *optional*, defaults to 1):
#            Minimum number of tokens that cannot be filtered.
#    """
#
#    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
#        top_p = float(top_p)
#        if top_p < 0 or top_p > 1.0:
#            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
#
#        self.top_p = top_p
#        self.filter_value = filter_value
#        self.min_tokens_to_keep = min_tokens_to_keep
#
#    def __call__(self, batch: EventStreamPytorchBatch, outputs: GenerativeSequenceModelPredictions) -> GenerativeSequenceModelPredictions:
#        sorted_logits, sorted_indices = torch.sort(outputs, descending=False)
#        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
#
#        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
#        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
#        if self.min_tokens_to_keep > 1:
#            # Keep at least min_tokens_to_keep
#            sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0
#
#        # scatter sorted tensors to original indexing
#        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
#        outputs = outputs.masked_fill(indices_to_remove, self.filter_value)
#        return outputs
#
#
#class TopKOutputsWarper(OutputsWarper):
#    r"""
#    [`OutputsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.
#    Args:
#        top_k (`int`):
#            The number of highest probability vocabulary tokens to keep for top-k-filtering.
#        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
#            All filtered values will be set to this float value.
#        min_tokens_to_keep (`int`, *optional*, defaults to 1):
#            Minimum number of tokens that cannot be filtered.
#    """
#
#    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
#        if not isinstance(top_k, int) or top_k <= 0:
#            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
#
#        self.top_k = top_k
#        self.filter_value = filter_value
#        self.min_tokens_to_keep = min_tokens_to_keep
#
#    def __call__(self, batch: EventStreamPytorchBatch, outputs: GenerativeSequenceModelPredictions) -> GenerativeSequenceModelPredictions:
#        top_k = min(max(self.top_k, self.min_tokens_to_keep), outputs.size(-1))  # Safety check
#        # Remove all tokens with a probability less than the last token of the top-k
#        indices_to_remove = outputs < torch.topk(outputs, top_k)[0][..., -1, None]
#        outputs = outputs.masked_fill(indices_to_remove, self.filter_value)
#        return outputs


#class ForcedBOSTokenOutputsProcessor(OutputsProcessor):
#    r"""
#    [`OutputsProcessor`] that enforces the specified token as the first generated token.
#    Args:
#        bos_token_id (`int`):
#            The id of the token to force as the first generated token.
#    """
#
#    def __init__(self, bos_token_id: int):
#        self.bos_token_id = bos_token_id
#
#    def __call__(self, batch: EventStreamPytorchBatch, outputs: GenerativeSequenceModelPredictions) -> GenerativeSequenceModelPredictions:
#        cur_len = batch.shape[-1]
#        if cur_len == 1:
#            num_tokens = outputs.shape[1]
#            outputs[:, [i for i in range(num_tokens) if i != self.bos_token_id]] = -float("inf")
#            outputs[:, self.bos_token_id] = 0
#        return outputs


#class ForcedEOSTokenOutputsProcessor(OutputsProcessor):
#    r"""
#    [`OutputsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.
#    Args:
#        max_length (`int`):
#            The maximum length of the sequence to be generated.
#        eos_token_id (`int`):
#            The id of the token to force as the last generated token when `max_length` is reached.
#    """
#
#    def __init__(self, max_length: int, eos_token_id: int):
#        self.max_length = max_length
#        self.eos_token_id = eos_token_id
#
#    def __call__(self, batch: EventStreamPytorchBatch, outputs: GenerativeSequenceModelPredictions) -> GenerativeSequenceModelPredictions:
#        cur_len = batch.shape[-1]
#        if cur_len == self.max_length - 1:
#            num_tokens = outputs.shape[1]
#            outputs[:, [i for i in range(num_tokens) if i != self.eos_token_id]] = -float("inf")
#            outputs[:, self.eos_token_id] = 0
#        return outputs


#class InfNanRemoveOutputsProcessor(OutputsProcessor):
#    r"""
#    [`OutputsProcessor`] that removes all `nan` and `inf` values to avoid the generation method to fail. Note that using
#    the logits processor should only be used if necessary since it can slow down the generation method.
#    """
#
#    def __call__(self, batch: EventStreamPytorchBatch, outputs: GenerativeSequenceModelPredictions) -> GenerativeSequenceModelPredictions:
#        # set all nan values to 0.0
#        outputs[outputs != outputs] = 0.0
#
#        # set all inf values to max possible value
#        outputs[outputs == float("inf")] = torch.finfo(outputs.dtype).max
#
#        return outputs


# class LogitNormalization(OutputsProcessor, OutputsWarper):
#     r"""
#     [`OutputsWarper`] and [`OutputsProcessor`] for normalizing the outputs using log-softmax. It's important to normalize
#     the outputs during beam search, after applying the logits processors or warpers, since the search algorithm used in
#     this library doesn't do it (it only does it before, but they may need re-normalization) but it still supposes that
#     the outputs are normalized when comparing the hypotheses.
#     """
#
#     def __call__(self, batch: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
#         outputs = outputs.log_softmax(dim=-1)
#         return outputs

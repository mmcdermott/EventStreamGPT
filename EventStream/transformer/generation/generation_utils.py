# Sourced from
# https://raw.githubusercontent.com/huggingface/transformers/v4.23.1/src/transformers/generation_utils.py
# Then modified.

# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc.
# team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from transformers.utils import ModelOutput

from ...data.types import PytorchBatch
from ..config import StructuredEventProcessingMode
from ..model_output import GenerativeSequenceModelPredictions
from .generation_stopping_criteria import MaxLengthCriteria, StoppingCriteriaList

logger = logging.getLogger(__name__)


@dataclass
class SampleDecoderOnlyOutput(ModelOutput):
    """Base class for outputs of decoder-only generation models using sampling.

    Args:
        batch (`PytorchBatch`):
            The generated sequences.
        scores (
            `tuple(GenerativeSequenceModelPredictions)` *optional*, returned when `output_scores=True` is
            passed or when `config.output_scores=True`
        ):
            Processed predictions of the generative sequence modeling head, as torch distributions at each
            generation step.
        attentions (
            `tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or
            `config.output_attentions=True`
        ):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder)
            of `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (
            `tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`
        ):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder)
            of `torch.FloatTensor` of shape `(batch_size, generated_length, dependency_graph_len,
            hidden_size)`.
    """

    scores: tuple[GenerativeSequenceModelPredictions] | None = None
    batch: PytorchBatch | None = None
    attentions: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = None


class StructuredGenerationMixin:
    """A class containing all functions for auto-regressive structured event stream generation, to be used as
    a mixin in [`PreTrainedModel`].

    The class exposes [`generate`], which can be used for:
        - *sampling* by calling [`sample`] if `do_sample=True`.
    """

    @staticmethod
    def _expand_inputs_for_generation(batch: PytorchBatch, expand_size: int = 1) -> PytorchBatch:
        return batch.repeat_batch_elements(expand_size)

    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: ModelOutput, model_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs["past_key_values"]
        else:
            model_kwargs["past"] = None

        return model_kwargs

    def prepare_inputs_for_generation(self, batch: PytorchBatch, **kwargs) -> dict[str, Any]:
        raise NotImplementedError(
            "A model class needs to define a `prepare_inputs_for_generation` method "
            "in order to use `.generate()`."
        )

    def _get_stopping_criteria(
        self,
        max_length: int | None,
        stopping_criteria: StoppingCriteriaList | None,
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if max_length is not None:
            criteria.append(MaxLengthCriteria(max_length=max_length))
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria

    def _merge_criteria_processor_list(
        self,
        default_list: StoppingCriteriaList,
        custom_list: StoppingCriteriaList,
    ) -> StoppingCriteriaList:
        if len(custom_list) == 0:
            return default_list
        default_list.extend(custom_list)
        return default_list

    @torch.no_grad()
    def generate(
        self,
        batch: PytorchBatch,
        max_length: int | None = None,
        do_sample: bool | None = True,
        num_return_sequences: int | None = None,
        max_new_events: int | None = None,
        use_cache: bool | None = None,
        stopping_criteria: StoppingCriteriaList | None = StoppingCriteriaList(),
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_scores: bool | None = None,
        return_dict_in_generate: bool | None = None,
        synced_gpus: bool | None = False,
        **model_kwargs,
    ) -> SampleDecoderOnlyOutput | PytorchBatch:
        # 1. Set generation parameters if not already defined
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        if not do_sample:
            raise ValueError("Only `do_sample=True` mode is currently supported")

        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.config.return_dict_in_generate
        )

        # 3. Define other model kwargs
        model_kwargs["use_cache"] = use_cache
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states

        # decoder-only models should use left-padding for generation
        if torch.any(~batch["event_mask"][:, -1]):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `seq_padding_side='left'` when initializing the data."
            )

        # 4. Prepare `max_length` depending on other stopping criteria.
        input_seq_length = batch.sequence_length
        if max_length is None and max_new_events is None:
            warnings.warn(
                "Neither `max_length` nor `max_new_events` has been set, `max_length` will default to "
                f"{self.config.max_length} (`self.config.max_length`). Controlling `max_length` via the "
                "config is deprecated and `max_length` will be removed from the config in v5 of Transformers "
                "-- we recommend using `max_new_events` to control the maximum length of the generation.",
                UserWarning,
            )
        elif max_length is None and max_new_events is not None:
            max_length = max_new_events + input_seq_length
        elif max_length is not None and max_new_events is not None:
            raise ValueError(
                "Both `max_new_events` and `max_length` have been set but they serve the same purpose -- "
                "setting a limit to the generated output length. Remove one of those arguments. Please "
                "refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        # default to config if still None
        max_length = max_length if max_length is not None else self.config.max_length

        if max_length is not None:
            if input_seq_length >= max_length:
                logger.warning(
                    f"Input length is {input_seq_length}, but `max_length` is set to"
                    f" {max_length}. This can lead to unexpected behavior. You should consider increasing "
                    "`max_new_events`."
                )
            if max_length > self.config.max_seq_len:
                raise ValueError(
                    "Can't run for a maximum length longer than the current maximum sequence length!"
                )

        # 7. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, stopping_criteria=stopping_criteria
        )
        # 8. go into different generation modes

        # 11. expand batch with `num_return_sequences` additional sequences per batch
        batch = self._expand_inputs_for_generation(batch, expand_size=num_return_sequences)

        match self.config.structured_event_processing_mode:
            case StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
                sample_fn = self._conditionally_independent_sample_event
            case StructuredEventProcessingMode.NESTED_ATTENTION:
                sample_fn = self._nested_attention_sample_event
            case _:
                raise ValueError(
                    "Unsupported structured event processing mode: "
                    f"{self.config.structured_event_processing_mode}"
                )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        unfinished_sequences = batch["event_mask"].new_ones(batch.batch_size)

        this_peer_finished = False  # used by synced_gpus only
        generated_event_index = 0
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(batch.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            for key in (
                "dynamic_indices",
                "dynamic_values",
                "dynamic_measurement_indices",
                "time_delta",
                "time",
            ):
                vals = batch[key]
                if isinstance(vals, torch.Tensor) and torch.isnan(vals).any():
                    raise ValueError(
                        f"{torch.isnan(vals).sum()} NaNs detected in {key} on index {generated_event_index}!"
                    )
                if isinstance(vals, torch.Tensor) and (~torch.isfinite(vals)).any():
                    raise ValueError(
                        f"{(~torch.isfinite(vals)).sum()} Non-finites detected in {key} on index "
                        f"{generated_event_index}."
                    )

            # forward pass to get next token
            batch, scores, attentions, hidden_states, model_kwargs = sample_fn(
                batch,
                generated_event_index,
                **model_kwargs,
            )

            if return_dict_in_generate:
                # We use the `scores` convention here as it is in the standard huggingface config.
                if output_scores:
                    scores += (scores,)
                if output_attentions:
                    decoder_attentions += (attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (hidden_states,)

            # if eos_token was found in one sentence, set sentence to finished
            # if eos_token_id is not None:
            #     unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(batch, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

            generated_event_index += 1

        if return_dict_in_generate:
            return SampleDecoderOnlyOutput(
                scores=scores,
                batch=batch,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
        else:
            return batch

    def _conditionally_independent_sample_event(
        self,
        batch: PytorchBatch,
        generated_event_index: int,
        **model_kwargs,
    ) -> tuple[
        PytorchBatch,
        GenerativeSequenceModelPredictions,
        torch.FloatTensor,
        torch.FloatTensor,
        dict[str, Any],
    ]:
        model_inputs = self.prepare_inputs_for_generation(batch, **model_kwargs)
        outputs = self(
            **model_inputs,
            return_dict=True,
            is_generation=True,
        )
        model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)

        next_event_preds = outputs.preds.slice((slice(None), -1))

        # Prediction
        next_event = next_event_preds.sample(batch.event_mask)

        batch = next_event.append_to_batch(batch, self.config)
        batch = next_event.update_last_event_data(batch, self.config)

        return batch, next_event_preds, outputs.attentions, outputs.hidden_states, model_kwargs

    def _nested_attention_sample_event(
        self,
        batch: PytorchBatch,
        generated_event_index: int,
        **model_kwargs,
    ) -> tuple[
        PytorchBatch,
        tuple[GenerativeSequenceModelPredictions],
        tuple[torch.FloatTensor],
        tuple[torch.FloatTensor],
        dict[str, Any],
    ]:
        # set measurements_to_fill
        # Recall that we assert that the first element of the dependency graph should encompass all the
        # functional time dependent metrics, so we omit that with the {"time"} component.
        measurements_to_fill_list = [{"time"}, *self.config.measurements_per_dep_graph_level[1:]]

        is_first = generated_event_index == 0

        scores = tuple()
        attentions = tuple()
        hidden_states = tuple()
        for dep_graph_el_target, measurements_to_fill in enumerate(measurements_to_fill_list):
            # forward pass to get next token
            if is_first and dep_graph_el_target == 0:
                dep_graph_el_target = None

            for key in (
                "dynamic_indices",
                "dynamic_values",
                "dynamic_measurement_indices",
                "time_delta",
                "time",
            ):
                vals = batch[key]
                if isinstance(vals, torch.Tensor) and torch.isnan(vals).any():
                    raise ValueError(
                        f"{torch.isnan(vals).sum()} NaNs detected in {key} on index {generated_event_index} "
                        f"for dep_graph_el_target {dep_graph_el_target} (is_first={is_first})!"
                    )
                if isinstance(vals, torch.Tensor) and (~torch.isfinite(vals)).any():
                    raise ValueError(
                        f"{(~torch.isfinite(vals)).sum()} Non-finites detected in {key} on index "
                        f"{generated_event_index} for dep_graph_el_target {dep_graph_el_target} "
                        f"(is_first={is_first})!"
                    )

            model_inputs = self.prepare_inputs_for_generation(
                batch, dep_graph_el_generation_target=dep_graph_el_target, **model_kwargs
            )
            outputs = self(
                **model_inputs,
                return_dict=True,
                is_generation=True,
            )

            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)

            next_event_preds = outputs.preds.slice((slice(None), -1))
            scores += (next_event_preds,)
            attentions += (outputs.attentions,)
            hidden_states += (outputs.hidden_states,)

            # Prediction
            next_event = next_event_preds.sample(batch.event_mask)

            # update batch for next step
            if measurements_to_fill == {"time"}:
                batch = next_event.append_to_batch(batch, self.config)
            else:
                batch = next_event.update_last_event_data(
                    batch,
                    self.config,
                    measurements_to_fill=measurements_to_fill,
                )

        return batch, scores, attentions, hidden_states, model_kwargs

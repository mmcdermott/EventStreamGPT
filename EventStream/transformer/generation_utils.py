# Sourced from https://raw.githubusercontent.com/huggingface/transformers/v4.23.1/src/transformers/generation_utils.py
# Then modified.

# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
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

import logging, warnings, pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from .generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
)
from .generation_outputs_process import OutputsProcessorList

from transformers.utils import ModelOutput

from ..data.types import PytorchBatch
from ..data.dataset_base import DatasetBase
from .model_output import GenerativeSequenceModelPredictions

logger = logging.getLogger(__name__)

# TODO(mmd): All Output classes are wrong!
@dataclass
class GreedySearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using greedy search.


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

    batch: Optional[PytorchBatch] = None
    scores: Optional[Tuple[GenerativeSequenceModelPredictions]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class SampleDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using sampling.


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

    scores: Optional[Tuple[GenerativeSequenceModelPredictions]] = None
    batch: Optional[PytorchBatch] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


GreedySearchOutput = GreedySearchDecoderOnlyOutput
SampleOutput = SampleDecoderOnlyOutput

class StructuredGenerationMixin:
    """
    A class containing all functions for auto-regressive structured event stream generation, to be used as a
    mixin in [`PreTrainedModel`].

    The class exposes [`generate`], which can be used for:
        - *greedy decoding* by calling [`greedy_search`] if `do_sample=False`.
        - *multinomial sampling* by calling [`sample`] if `do_sample=True`.
    """

    @staticmethod
    def _expand_inputs_for_generation(
        batch: PytorchBatch,
        expand_size: int = 1,
        **model_kwargs,
    ) -> Tuple[PytorchBatch, Dict[str, Any]]:
        expanded_return_idx = torch.arange(
            batch['time'].shape[0]
        ).view(
            -1, 1
        ).repeat(
            1, expand_size
        ).view(
            -1
        ).to(
            batch['time'].device
        )

        for k, v in batch.items():
            match v:
                case dict(): batch[k] = {kk: vv.index_select(0, expanded_return_idx) for kk, vv in v.items()}
                case torch.Tensor(): batch[k] = v.index_select(0, expanded_return_idx)
                case _: raise TypeError(f"{k}: {type(v)} not supported in batch for generation!")

        return batch, model_kwargs

    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return model_kwargs

    def _get_outputs_warper(
        self,
    ) -> OutputsProcessorList:
        """
        This class returns a [`OutputsProcessorList`] list object that contains all relevant [`OutputsWarper`] instances
        used for multinomial sampling.
        """

        # TODO(mmd): Does nothing for now.

        warpers = OutputsProcessorList()

        return warpers

    def _get_model_outputs_processor(
        self,
    ) -> OutputsProcessorList:
        """
        This class returns a [`OutputsProcessorList`] list object that contains all relevant
        [`OutputsProcessor`] instances used to modify the scores of the language model head.
        """
        # This is a placeholder for now, as no outputs processors are currently supported!
        processors = OutputsProcessorList()

        # init warp parameters
        return processors

    def _get_stopping_criteria(
        self,
        max_length: Optional[int],
        max_time: Optional[float],
        stopping_criteria: Optional[StoppingCriteriaList],
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if max_length is not None:
            criteria.append(MaxLengthCriteria(max_length=max_length))
        if max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=max_time))
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria

    def _merge_criteria_processor_list(
        self,
        default_list: Union[OutputsProcessorList, StoppingCriteriaList],
        custom_list: Union[OutputsProcessorList, StoppingCriteriaList],
    ) -> Union[OutputsProcessorList, StoppingCriteriaList]:
        if len(custom_list) == 0:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "outputs processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `generate`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `generate` instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list

    @torch.no_grad()
    def generate(
        self,
        batch: PytorchBatch,
        max_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_events: Optional[int] = None,
        use_cache: Optional[bool] = None,

        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,

        # TODO(mmd): Improve API so this isn't necessary!
        base_dataset: Optional[DatasetBase] = None,
        batch_schema: Optional[List[Tuple[int, datetime, datetime]]] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, torch.LongTensor]:
        r"""

        Generates continuous-time sequences of events for models with an  Generation head. The
        method supports the followign generation methods:

            - *greedy decoding* by calling [`greedy_search`] if `do_sample=False`.
            - *multinomial sampling* by calling [`sample`] if `do_sample=True`.

        <Tip warning={true}>

        Apart from `batch`, all the arguments below will default to the value of the attribute of the same
        name as defined in the model's config (`config.json`) which in turn defaults to the
        [`~modeling_utils.PretrainedConfig`] of the model.

        </Tip>

        Parameters:
            batch (`PytorchBatch`):
                The sequence used as a prompt for the generation or as model inputs to the encoder.
                Can't be `None`, currently.
            max_length (`int`, *optional*, defaults to `model.config.max_length`):
                The maximum length the generated stream can have. Corresponds to the length of the input
                prompt + `max_new_events`.
                In general, prefer the use of `max_new_events`, which ignores the number of tokens in the
                prompt.
            max_new_events (`int`, *optional*):
                The maximum numbers of tokens to generate, ignoring the number of events in the prompt.
            do_sample (
                `bool`, *optional*, defaults to `model.config.do_sample` or `False` if the config does not set
                any value
            ):
                Whether or not to use sampling ; use greedy decoding otherwise.
            temperature (
                `float`, *optional*, defaults to `model.config.temperature` or 1.0 if the config does not set
                any value
            ):
                The value used to modulate the next token probabilities.

            num_return_sequences(
                `int`, *optional*, defaults to `model.config.num_return_sequences` or 1 if the config does not
                set any value
            ):
                The number of independently computed returned sequences for each element in the batch.
            max_time(`float`, *optional*):
                The maximum amount of time you allow the computation to run for in seconds. generation will still
                finish the current pass after allocated time has been passed.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, 1
                for tokens that are not masked, and 0 for masked tokens. If not provided, will default to a
                tensor the same shape as `input_ids` that masks the pad token.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the
                model) to speed up decoding.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                 Custom stopping criteria that complement the default stopping criteria built from arguments
                 and a model's config. If a stopping criteria is passed that is already created with the
                 arguments or a model's config an error is thrown. This feature is intended for advanced
                 users.
            output_attentions (
                `bool`, *optional*, defaults to `model.config.output_attentions` or `False` if the config does
                not set any value
            ):
                Whether or not to return the attentions tensors of all attention layers. See `attentions`
                under returned tensors for more details.
            output_hidden_states (
                `bool`, *optional*, defaults to `model.config.output_hidden_states` or `False` if the config
                does not set any value
            ):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned
                tensors for more details.
            output_scores (
                `bool`, *optional*, defaults to `model.config.output_scores` or `False` if the config does not
                set any value
            ):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more
                details.
            return_dict_in_generate (
                `bool`, *optional*, defaults to `model.config.return_dict_in_generate` or `False` if the
                config does not set any value
            ):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)

            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If
                the model is an encoder-decoder model, encoder specific kwargs should not be prefixed and
                decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if
            `return_dict_in_generate=True` or when `config.return_dict_in_generate=True`) or a
            `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the
                possible [`~utils.ModelOutput`] types are:

                    - [`~generation_utils.GreedySearchDecoderOnlyOutput`],
                    - [`~generation_utils.SampleDecoderOnlyOutput`],
                    - [`~generation_utils.BeamSearchDecoderOnlyOutput`],
                    - [`~generation_utils.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the
                possible [`~utils.ModelOutput`] types are:

                    - [`~generation_utils.GreedySearchEncoderDecoderOutput`],
                    - [`~generation_utils.SampleEncoderDecoderOutput`],
                    - [`~generation_utils.BeamSearchEncoderDecoderOutput`],
                    - [`~generation_utils.BeamSampleEncoderDecoderOutput`]
        """
        assert base_dataset is not None, "base_dataset must be provided for structured event generation."
        assert batch_schema is not None,\
            "batch_schema must be provided for structured event generation."

        static_data = base_dataset.subjects_df.loc[[subj for subj, _, _ in batch_schema]]

        # 1. Set generation parameters if not already defined
        do_sample = do_sample if do_sample is not None else self.config.do_sample
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
            return_dict_in_generate if return_dict_in_generate is not None
            else self.config.return_dict_in_generate
        )

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache

        # decoder-only models should use left-padding for generation
        if torch.any(~batch['event_mask'][:, -1]):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `seq_padding_side='left'` when initializing the data."
            )

        # 4. Prepare `max_length` depending on other stopping criteria.
        input_seq_length = batch['time'].shape[-1]
        if max_length is None and max_new_events is None:
            warnings.warn(
                "Neither `max_length` nor `max_new_events` has been set, `max_length` will default to "
                f"{self.config.max_length} (`self.config.max_length`). Controlling `max_length` via the config is "
                "deprecated and `max_length` will be removed from the config in v5 of Transformers -- we recommend "
                "using `max_new_events` to control the maximum length of the generation.",
                UserWarning,
            )
        elif max_length is None and max_new_events is not None:
            max_length = max_new_events + input_seq_length
        elif max_length is not None and max_new_events is not None:
            raise ValueError(
                "Both `max_new_events` and `max_length` have been set but they serve the same purpose -- setting a"
                " limit to the generated output length. Remove one of those arguments. Please refer to the"
                " documentation for more information. "
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

        # 5. determine generation mode
        is_greedy_gen_mode = (do_sample is False)
        is_sample_gen_mode = (do_sample is True)

        # 6. prepare distribution pre_processing samplers
        # TODO(mmd): Right now, this does nothing, as we don't have any valid outputs processors, but
        # eventually we may.
        outputs_processor = self._get_model_outputs_processor()

        # 7. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        )
        # 8. go into different generation modes
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} "
                    f"when doing greedy search."
                )

            # 10. run greedy search
            return self.greedy_search(
                batch,
                outputs_processor=outputs_processor,
                stopping_criteria=stopping_criteria,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                base_dataset=base_dataset,
                batch_schema=batch_schema,
                static_data=static_data,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 10. prepare outputs warper
            outputs_warper = self._get_outputs_warper()

            # 11. expand batch with `num_return_sequences` additional sequences per batch
            batch, model_kwargs = self._expand_inputs_for_generation(
                batch,
                expand_size=num_return_sequences,
                **model_kwargs,
            )

            # 12. run sample
            return self.sample(
                batch,
                outputs_processor=outputs_processor,
                outputs_warper=outputs_warper,
                stopping_criteria=stopping_criteria,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                base_dataset=base_dataset,
                batch_schema=batch_schema,
                static_data=static_data,
                **model_kwargs,
            )

    def _search(
        self,
        batch: PytorchBatch,
        outputs_processor: Optional[OutputsProcessorList] = None,
        outputs_warper: Optional[OutputsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        sample_fn: str = 'mode',
        # TODO(mmd): Improve API -- this shouldn't be necessary.
        base_dataset: Optional[DatasetBase] = None,
        batch_schema: Optional[List[int]] = None,
        static_data: Optional[pd.DataFrame] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, PytorchBatch]:
        r"""
        Generates sequences of token ids for models with a generative sequence modeling head using either
        greedy or sample decoding.

        Parameters:
            batch (`PytorchBatch`):
                The sequence used as a prompt for the generation.
            outputs_processor (`OutputsProcessorList`, *optional*):
                An instance of [`OutputsProcessorList`]. List of instances of class derived from [`OutputsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.
        """
        assert sample_fn in ('greedy', 'sample')
        assert base_dataset is not None, "base_dataset must be provided for structured event generation."
        assert batch_schema is not None, \
            "batch_schema must be provided for structured event generation."
        assert static_data is not None, "static_data must be provided for structured event generation."

        # init values
        outputs_processor = outputs_processor if outputs_processor is not None else OutputsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        outputs_warper = outputs_warper if outputs_warper is not None else OutputsProcessorList()
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        unfinished_sequences = batch['time'].new_ones(batch['time'].shape[0])

        this_peer_finished = False  # used by synced_gpus only
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

            next_scores = ()

            match self.config.structured_event_processing_mode:
                case 'conditionally_independent':
                    measurements_to_fill_list = [{'time'}, set(self.config.measurements_idxmap.keys())]
                case 'nested_attention':
                    if self.config.measurements_per_dep_graph_level:
                        measurements_to_fill_list = [
                            {'time'}, *self.config.measurements_per_dep_graph_level[1:]
                        ]
                    else:
                        measurements_to_fill_list = [{'time'}, set(self.config.measurements_idxmap.keys())]

            for measurements_to_fill in measurements_to_fill_list:
                # TODO(mmd): Here -- need to loop over dependency graph elements.
                # forward pass to get next token
                outputs = self(
                    batch=batch,
                    **model_kwargs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    is_generation=True,
                )

                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need

                next_event_preds = outputs.preds.slice((slice(None), -1))

                # pre-process distribution
                next_event_preds = outputs_processor(batch, next_event_preds)
                next_event_preds = outputs_warper(batch, next_event_preds)

                if return_dict_in_generate:
                    # We use the `scores` convention here as it is in the standard huggingface config.
                    if output_scores: next_scores += (next_event_preds,)

                # Prediction
                # TODO(mmd): make this only output the appropriate data types
                match sample_fn:
                    case 'greedy': next_event = next_event_preds.mode(batch.event_mask)
                    case 'sample': next_event = next_event_preds.sample(batch.event_mask)

                # update batch for next step
                if measurements_to_fill == {'time'}:
                    batch = next_event.append_to_batch(
                        batch, self.config,
                        base_dataset=base_dataset,
                        batch_schema=batch_schema,
                        static_data=static_data
                    )
                else:
                    batch = next_event.update_last_event_data(
                        batch, self.config, measurements_to_fill=measurements_to_fill
                    )

            if return_dict_in_generate:
                # We use the `scores` convention here as it is in the standard huggingface config.
                if output_scores: scores += (next_scores,)
                if output_attentions: decoder_attentions += (outputs.attentions,)
                if output_hidden_states: decoder_hidden_states += (outputs.hidden_states,)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False
            )

            # if eos_token was found in one sentence, set sentence to finished
            # if eos_token_id is not None:
            #     unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(batch, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            cls = GreedySearchDecoderOnlyOutput if sample_fn == 'greedy' else SampleDecoderOnlyOutput
            return cls(
                scores=scores,
                batch=batch,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
        else:
            return batch

    def greedy_search(self, *args, **kwargs) -> Union[GreedySearchOutput, PytorchBatch]:
        return self._search(*args, **kwargs, sample_fn = 'greedy')

    def sample(self, *args, **kwargs) -> Union[SampleOutput, PytorchBatch]:
        return self._search(*args, **kwargs, sample_fn = 'sample')

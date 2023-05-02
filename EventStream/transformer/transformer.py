# Based on "
# https://raw.githubusercontent.com/huggingface/transformers/
# e3cc4487fe66e03ec85970ea2db8e5fb34c455f4/src/transformers/models/gpt_neo/modeling_gpt_neo.py
# "
""" PyTorch StructuredTransformer model."""

import math, torch, torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from typing import Dict, Optional, Tuple, Union

from ..data.data_embedding_layer import DataEmbeddingLayer
from ..data.types import PytorchBatch
from .config import StructuredTransformerConfig
from .model_output import TransformerOutputWithPast
from .structured_attention import StructuredAttention

logger = logging.get_logger(__name__)

# TODO(mmd): Can use `transformers.apply_chunking_to_forward` to save memory.

def expand_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    if mask is None: return None

    # We create a 3D attention mask from a 2D tensor mask.
    # Sizes are [batch_size, 1, 1, to_seq_length]
    # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    # this attention mask is more simple than the triangular masking of causal attention
    # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    attention_mask = mask[:, None, None, :]

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_mask = attention_mask.to(dtype=dtype)  # fp16 compatibility
    attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min

    return attention_mask

class InnerSelfAttention(nn.Module):
    def __init__(
        self, config: StructuredTransformerConfig, attention_type: str, window_size: int,
    ):
        super().__init__()

        max_seq_len = config.max_seq_len
        self.window_size = window_size
        bias = torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.uint8)).view(
            1, 1, max_seq_len, max_seq_len
        )

        # local causal self attention is a sliding window where each token can only attend to the previous
        # window_size tokens. This is implemented by updating the causal mask such that for each token
        # all other tokens are masked except the previous window_size tokens.
        if attention_type == "local":
            bias = torch.bitwise_xor(bias, torch.tril(bias, -window_size))

        self.register_buffer("bias", bias)
        self.register_buffer("masked_bias", torch.tensor(-1e9))

        self.attn_dropout = nn.Dropout(float(config.attention_dropout))
        self.resid_dropout = nn.Dropout(float(config.resid_dropout))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and "
                f"`num_heads`: {self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length].to(torch.bool)
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error:
        # `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        static_kv_first: bool = False,
    ):
        # In the case of attention over the dependency graph, the history embedding is dropped after
        # processing, so we want to only use it as a KV, not as a query. This is captured in the
        # `static_kv_first` arg.

        # TODO(mmd): Flash attention
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # query, key, and value are all of shape (batch, head, seq_length, head_features)

        if static_kv_first:
            # In this case, we are only interested in performing the attention update over the non-static
            # queries.
            query = query[:, :, 1:, :]

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = {'present_key_value': present}
        if output_attentions:
            outputs['attn_weights'] = attn_weights

        return attn_output, outputs  # a, {present, (attentions)}


class InnerAttention(nn.Module):
    def __init__(
        self, config: StructuredTransformerConfig, layer_id: int = 0, is_seq: bool = True
    ):
        super().__init__()
        self.layer_id = layer_id
        self.is_seq = is_seq
        self.attention_layers = config.seq_attention_layers if is_seq else config.dep_graph_attention_layers
        self.attention_type = self.attention_layers[layer_id]
        if self.attention_type == 'local':
            self.window_size = config.seq_window_size if is_seq else config.dep_graph_window_size
        else: self.window_size = None

        if self.attention_type in ["global", "local"]:
            self.attention = InnerSelfAttention(
                config,
                attention_type=self.attention_type,
                window_size=self.window_size
            )
        else:
            raise NotImplementedError(
                "Only attn layer types 'global' and 'local' exist, but got `config.attention_layers`: "
                f"{config.attention_layers}. Select attn layer types from ['global', 'local'] only."
            )

        # We put the layer norm in here as sometimes the attention layer is used independently of the full
        # block setup but we still want the layer norm to happen.
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        hidden_states = self.layer_norm(hidden_states)
        return self.attention(
            hidden_states,
            attention_mask=expand_mask(attention_mask, dtype=hidden_states.dtype),
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            static_kv_first=(not self.is_seq),
        )


class InnerMLP(nn.Module):
    def __init__(self, config: StructuredTransformerConfig):
        super().__init__()
        embed_dim = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * embed_dim

        self.c_fc = nn.Linear(embed_dim, inner_dim)
        self.c_proj = nn.Linear(inner_dim, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class InnerBlock(nn.Module):
    def __init__(self, config: StructuredTransformerConfig, layer_id: int, is_seq: bool):
        super().__init__()
        self.attn = InnerAttention(config, layer_id, is_seq)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = InnerMLP(config)
        self.static_kv_first = (not is_seq)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Note that attention_mask here is still not expanded; we do that internally here to account for the
        different mask shapes used in the structured transformer.
        """
        # If we have a static kv entry first, we don't want to process it in the rest of the block, so we drop
        # it from the residual.
        residual = hidden_states if not self.static_kv_first else hidden_states[:, 1:, :]

        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output, outputs = attn_outputs  # output_attn: a, {present, (attentions)}

        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if not use_cache: outputs.pop('present_key_value')
        return hidden_states, outputs


class StructuredTransformerBlock(nn.Module):
    def __init__(self, config: StructuredTransformerConfig, layer_id: int):
        super().__init__()

        if config.do_full_block_in_seq_attention:
            seq_block = InnerBlock(config, layer_id, is_seq=True)
        else:
            seq_block = InnerAttention(config, layer_id, is_seq=True)

        if config.do_full_block_in_dep_graph_attention:
            dep_graph_block = InnerBlock(config, layer_id, is_seq=False)
        else:
            dep_graph_block = InnerAttention(config, layer_id, is_seq=False)

        self.block = StructuredAttention(
            seq_module=seq_block, dep_graph_module=dep_graph_block,
        )

    def forward(
        self, *args, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Optional[Dict[str, Optional[torch.Tensor]]]]]:
        return self.block(*args, **kwargs)


class StructuredTransformerPreTrainedModel(PreTrainedModel):
    """
    The pre-trained model class for Transformer models.
    """

    config_class = StructuredTransformerConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["StructuredTransformerBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, StructuredTransformer):
            module.gradient_checkpointing = value

# Copied from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class TemporalPositionEncoding(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        device: str = ('cuda' if torch.cuda.is_available() else 'cpu'),
        max_timepoint: float = 10000.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        div_term = torch.nn.Parameter(
            torch.exp(
                torch.arange(0, embedding_dim, 2, device=device) * (-math.log(max_timepoint) / embedding_dim)
            ), requires_grad=False
        )

        # We still want this to work for odd embedding dimensions, so we'll lop off the end of the cos
        # embedding. This is not a principled decision, but enabling odd embedding dimensions helps avoid edge
        # cases during hyperparameter tuning when searching over possible embedding spaces.
        if self.embedding_dim % 2 == 0:
            self.sin_div_term = div_term
            self.cos_div_term = div_term
        else:
            self.sin_div_term = div_term
            self.cos_div_term = div_term[:-1]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t is the tensor of input timepoints, with shape (batch size, sequence length)
        """

        bsz, seq_len = t.shape

        # First, we unsqueeze it for broadcasting through the hidden dim.
        t = t.unsqueeze(-1)

        # temporal_embeddings will be our output container.
        # It should have shape (batch size, sequence length, embedding dim), and be on the same device as the
        # timepoints.
        temporal_embeddings = torch.zeros(bsz, seq_len, self.embedding_dim, device=t.device)

        temporal_embeddings[:, :, 0::2] = torch.sin(t * self.sin_div_term.unsqueeze(0).unsqueeze(0))
        temporal_embeddings[:, :, 1::2] = torch.cos(t * self.cos_div_term.unsqueeze(0).unsqueeze(0))

        return temporal_embeddings

class StructuredInputLayer(torch.nn.Module):
    """
    Takes as input a batch from an event-stream pytorch dataset and produces contextualized embeddings from it.
    """

    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__()

        self.config = config

        if config.static_embedding_mode in ('prepend', 'concat_all'):
            raise NotImplementedError(f"{config.static_embedding_mode} mode is not yet supported.")

        if config.measurements_per_dep_graph_level is not None:
            # We need to translate from measurement name to index here via config.measurements_idxmap
            split_by_measurement_indices = []
            for measurement_list in config.measurements_per_dep_graph_level:
                out_list = []
                for measurement in measurement_list:
                    if type(measurement) is str:
                        out_list.append(config.measurements_idxmap[measurement])
                    elif (type(measurement) in (tuple, list)) and (len(measurement) == 2):
                        out_list.append((
                            config.measurements_idxmap[measurement[0]], measurement[1]
                        ))
                    else:
                        raise ValueError(
                            f"Unexpected type {type(measurement)}: {measurement}\n"
                            f"{config.measurements_per_dep_graph_level}"
                        )
                split_by_measurement_indices.append(out_list)
        else: split_by_measurement_indices = None

        self.data_embedding_layer = DataEmbeddingLayer(
            n_total_embeddings = config.vocab_size,
            out_dim = config.hidden_size,
            categorical_embedding_dim = config.categorical_embedding_dim,
            numerical_embedding_dim = config.numerical_embedding_dim,
            static_embedding_mode = config.static_embedding_mode,
            split_by_measurement_indices = split_by_measurement_indices,
            do_normalize_by_measurement_index = config.do_normalize_by_measurement_index,
            static_weight = config.static_embedding_weight,
            dynamic_weight = config.dynamic_embedding_weight,
            categorical_weight = config.categorical_embedding_weight,
            numerical_weight = config.numerical_embedding_weight,
        )

        self.time_embedding_layer = TemporalPositionEncoding(embedding_dim = config.hidden_size)

        self.embedding_dropout = torch.nn.Dropout(p=config.input_dropout)

    def forward(self, batch: PytorchBatch) -> torch.Tensor:
        data_embed = self.data_embedding_layer(batch)
        # data_embed is either of shape (batch_size, sequence_length, config.hidden_size) or of shape
        # (batch_size, sequence_length, len(config.measurements_per_dep_graph_level), config.hidden_size)

        time_embed = self.time_embedding_layer(batch['time'])
        # time_embed is of shape (batch_size, sequence_length, config.hidden_size)

        if self.config.measurements_per_dep_graph_level is not None:
            # In this case, we are in a non-conditionally independent mode, with a specified dependency graph
            # split. We assume that the first element of the dependency graph split reflects those components
            # that should be lumped in with time (e.g., the functional time dependent variables). We perform a
            # cumsum in this case such that even in the first layer, our final embedding of the dep graph
            # reflects the entire event.
            # TODO(mmd): The cumsum here should probably be normalized? Leveraging some dep_graph_mask?
            data_embed = data_embed.cumsum(dim=2)
            data_embed += time_embed.unsqueeze(2)
        else:
            # In this case, if we are in a conditionally independent setting, we ultimately want to sum the
            # time and data embedding, and if not, the None split by indicates that we should have an implicit
            # dep graph of [time, contents]
            if self.config.structured_event_processing_mode == 'conditionally_independent':
                # In a conditionally independent model, we collapse the dependency graph structure and just
                # represent each event with a single embedding.
                data_embed += time_embed
            else:
                data_embed = torch.cat((time_embed.unsqueeze(2), data_embed.unsqueeze(2)), dim=2)

        return self.embedding_dropout(data_embed)

class StructuredTransformer(StructuredTransformerPreTrainedModel):
    def __init__(self, config: StructuredTransformerConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.input_layer = StructuredInputLayer(config)
        self.structured_event_processing_mode = config.structured_event_processing_mode

        # TODO(mmd): Replace this with InnerBlock for a non-structured version.
        if config.structured_event_processing_mode == 'nested_attention':
            self.h = nn.ModuleList([
                StructuredTransformerBlock(config, layer_id=i)
                for i in range(config.num_hidden_layers)
            ])
        elif config.structured_event_processing_mode == 'conditionally_independent':
            self.h = nn.ModuleList([
                InnerBlock(config, layer_id=i, is_seq=True)
                for i in range(config.num_hidden_layers)
            ])
        else: raise ValueError(
            "Invalid `config.structured_event_processing_mode`! Got "
            f"{config.structured_event_processing_mode}."
        )

        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        batch: Optional[PytorchBatch] = None,
        input_embeds: Optional[torch.Tensor] = None,
        past: Optional[Tuple[torch.FloatTensor]] = None,
        seq_mask: Optional[torch.Tensor] = None,
        dep_graph_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TransformerOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past is None: past = tuple([None] * len(self.h))

        if input_embeds is None:
            assert batch is not None
            assert seq_mask is None

            input_embeds = self.input_layer(batch)
            seq_mask = batch['event_mask']
        else: assert batch is None, "Can't specify both input_embeds and batch."

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = input_embeds
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. "
                        "Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                # We do this twice because the checkpointed process can't take keyword args, which is safer
                # and cleaner, in my opinion.
                if self.structured_event_processing_mode == 'nested_attention':
                    args = (
                        hidden_states,
                        dep_graph_mask,
                        seq_mask,
                        dict(
                            layer_past=layer_past,
                            head_mask=head_mask[i],
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                        ),
                        {},
                    )
                elif self.structured_event_processing_mode == 'conditionally_independent':
                    args = (
                        hidden_states,
                        seq_mask,
                        layer_past,
                        head_mask[i],
                        use_cache,
                        output_attentions,
                    )
                else: raise ValueError(
                    "Invalid `self.structured_event_processing_mode`! Got "
                    f"{self.structured_event_processing_mode}."
                )

                outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(block), *args)
            else:
                if self.structured_event_processing_mode == 'nested_attention':
                    kwargs = dict(
                        hidden_states=hidden_states,
                        dep_graph_mask=dep_graph_mask,
                        seq_mask=seq_mask,
                        seq_module_kwargs = dict(
                            layer_past=layer_past,
                            head_mask=head_mask[i],
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                        ),
                        dep_graph_module_kwargs = {}
                    )
                elif self.structured_event_processing_mode == 'conditionally_independent':
                    kwargs = dict(
                        hidden_states=hidden_states,
                        attention_mask=seq_mask,
                        layer_past=layer_past,
                        head_mask=head_mask[i],
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )
                else: raise ValueError(
                    "Invalid `self.structured_event_processing_mode`! Got "
                    f"{self.structured_event_processing_mode}."
                )
                outputs = block(**kwargs)

            hidden_states, extra_return_info = outputs
            if self.structured_event_processing_mode == 'nested_attention':
                extra_return_info = extra_return_info['seq_module']

            if use_cache is True:
                presents = presents + (extra_return_info['present_key_value'],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (extra_return_info['attn_weights'],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(input_embeds.size())
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return TransformerOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

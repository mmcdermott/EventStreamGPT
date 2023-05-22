from typing import Any

import torch


class StructuredAttention(torch.nn.Module):
    """This module performs a dependency-graph structured attention calculation, in which each
    sequence element is itself composed of objects with internal dependency structures that you want
    to respect during calculation.

    This module is basically just a container for appropriately shuffling the input tensors to pass
    them to the nested modules for pooling events, processing the event sequences, then processing
    the intra-event dependency graph objects.
    """

    def __init__(
        self,
        seq_module: torch.nn.Module,
        dep_graph_module: torch.nn.Module,
    ):
        super().__init__()

        self.seq_module = seq_module
        self.dep_graph_module = dep_graph_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
        seq_module_kwargs: dict[str, Any] | None = None,
        dep_graph_module_kwargs: dict[str, Any] | None = None,
        prepend_graph_with_history_embeddings: bool = True,
        update_last_graph_el_to_history_embedding: bool = True,
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor | None] | None]]:
        """
        Performs a structured attention forward pass, consistent with the internal sub-modules. E.g., given
        the notation defined in the hidden_state arg documentation, this module performs the following steps:
            1. Input events are summarized into event embeddings via their last entry:
                `x_i = [e_{i, 1}, ..., e_{i, m}, s_i][:, -1, :] = s_i
            2. These events are contextualized via the history:
                `h_i = self.seq_module([x_1, ..., x_i], seq_mask, **seq_module_kwargs)`
            3. Output embeddings are produced by processing the historical context and dep. graph structure:
                ```
                e_{i, j}^{out} = self.dep_graph_module(
                    [h_{i-1}, e_{i, 1}, ..., e_{i, j-1}], **dep_graph_module_kwargs
                )
                ```

        Args:
            hidden_states (`torch.Tensor` of shape (batch, seq len, dependency graph len, hidden size)):
                The input embeddings corresponding to the different elements of the structured dependency
                graph, with the last element of the graph corresponding to a whole-event embedding
                E.g., if the sequence length is n, then
                hidden_states = [...[
                    [e_{1, 1}, e_{2, 1}, ..., e_{n-1, 1}, e_{n, 1}],
                    [e_{1, 2}, e_{2, 2}, ..., e_{n-1, 2}, e_{n, 2}],
                    ...
                    [e_{1, m}, e_{2, m}, ..., e_{n-1, m}, e_{n, m}],
                    [s_{1},    s_{2},    ..., s_{n-1},    s_{n}],
                ]...]
                corresponds to a setting where each sequence element x_i consists of a set
                (e_{i, 1}, ..., e_{i, m}) such that the generative model states that
                (x_1, ..., x_{i-1}) -> x_i and, within x_i, the e_{i, j} elements obey dependency
                relationships specified within the dep_graph_module. and s_i reflects the embedding of the
                entire element x_i

            seq_mask (`torch.Tensor` of shape `(batch, seq len)`, *optional*):
                Mask to avoid processing on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            seq_module_kwargs (`Dict[str, Any]`, *optional*):
                additional keyword arguments to pass to the sequence module.

            dep_graph_module_kwargs (`Dict[str, Any]`, *optional*):
                additional keyword arguments to pass to the dependency graph module.
        """

        if seq_module_kwargs is None:
            seq_module_kwargs = {}
        if dep_graph_module_kwargs is None:
            dep_graph_module_kwargs = {}

        # First, we produce input event embeddings:
        bsz, seq_len, dep_graph_len, hidden_size = hidden_states.size()

        compute_contextualized_history_embeddings = (
            prepend_graph_with_history_embeddings or update_last_graph_el_to_history_embedding
        )

        if compute_contextualized_history_embeddings:
            # To pool the per-event data, we need to reshape the data so that the natural "sequence length"
            # dimension is the dependency graph length.
            for_per_event_pooling = torch.reshape(
                hidden_states, (bsz * seq_len, dep_graph_len, hidden_size)
            )

            # However, we don't want to process any padding elements, so we need to filter those out.
            # To do this, we'll use the seq_mask. It has shape (bsz, seq_len). We'll expand it then re-shape
            # it in a similar manner as the for_per_event_pooling tensor.
            if seq_mask is None:
                per_event_all = for_per_event_pooling[:, -1, :]
            else:
                flat_seq_mask = torch.reshape(seq_mask, (bsz * seq_len,)).bool()
                for_per_event_pooling_present = for_per_event_pooling[flat_seq_mask, :, :]
                per_event = for_per_event_pooling_present[:, -1, :]

                per_event_all = torch.zeros(
                    bsz * seq_len, hidden_size, dtype=per_event.dtype, device=per_event.device
                )
                per_event_all[flat_seq_mask, :] = per_event

            per_event_all = torch.reshape(per_event_all, (bsz, seq_len, hidden_size))

            # Next, we need to summarize the sequence of pooled event embeddings.
            contextualized_events = self.seq_module(
                per_event_all,
                attention_mask=seq_mask,
                **seq_module_kwargs,
            )
            # Some modules will return extra outputs (e.g., attention weights, past key values, etc.)
            if isinstance(contextualized_events, tuple):
                contextualized_events, seq_module_return_kwargs = contextualized_events
            else:
                seq_module_return_kwargs = None

            # contextualized_events is of shape (bsz, seq_len, hidden_size)

            if prepend_graph_with_history_embeddings:
                # To produce the contextualized view of the _history_ prior to an event i, we pad the start of
                # this set of contextualized events with a zero vector and drop the last event.
                contextualized_history = torch.cat(
                    (
                        torch.zeros_like(contextualized_events[:, :1, :]),
                        contextualized_events[:, :-1, :],
                    ),
                    dim=1,
                )
                # contextualized_history is of shape (batch size, seq len, hidden_size)

                # Finally, we produce output embeddings
                # e_{i,j}' = self.dep_graph_module(
                #   [h_{i-1}, e_{i, 1}, ..., e_{i, j-1}, s_{i}], **dep_graph_module_kwargs
                # ).

                # To do this, we first produce these augmented dependency graph sequences:
                dep_graph_seq = torch.cat(
                    (contextualized_history.unsqueeze(2), hidden_states), dim=2
                )
                # dep_graph_seq is now of shape (batch size, seq len, dependency graph len + 1, hidden size)
                static_kv_first = True
            else:
                dep_graph_seq = hidden_states
                static_kv_first = False

            if update_last_graph_el_to_history_embedding:
                # We also update the per-event embedding s_i to reflect the contextualized version already
                # produced. This is especially important for any model that uses `use_cache`, as from here we
                # source the key/value embeddings for the contextualized_history vector for dependency element
                # generation in the next element of the sequence!
                dep_graph_seq[:, :, -1, :] = contextualized_events
        else:
            # Now, we need to reshape these so that the dependency graph axis is again the sequence axis, and
            # we also need to drop the padding elements of the sequence once more.
            static_kv_first = False
            dep_graph_seq = hidden_states
            seq_module_return_kwargs = None

            if seq_mask is not None:
                flat_seq_mask = torch.reshape(seq_mask, (bsz * seq_len,)).bool()

        # Now, we need to reshape these so that the dependency graph axis is again the sequence axis, and
        # we also need to drop the padding elements of the sequence once more.
        dep_graph_seq = torch.reshape(dep_graph_seq, (bsz * seq_len, -1, hidden_size))

        if seq_mask is not None:
            dep_graph_seq = dep_graph_seq[flat_seq_mask, :, :]

        dep_graph_out = self.dep_graph_module(
            dep_graph_seq,
            attention_mask=None,
            static_kv_first=static_kv_first,
            **dep_graph_module_kwargs,
        )
        # Some modules will return extra outputs (e.g., attention weights, past key values, etc.)
        if isinstance(dep_graph_out, tuple):
            dep_graph_out, dep_graph_module_return_kwargs = dep_graph_out
        else:
            dep_graph_module_return_kwargs = None

        # dep_graph_out has shape (?, dep_graph_len, hidden_size), as the dep_graph_module should have dropped
        # the history embedding from the output.

        # Now we need to re-shape it back to the original, respecting the dropped sequence elements.
        if seq_mask is None:
            dep_graph_all = dep_graph_out
        else:
            dep_graph_all = torch.zeros(
                bsz * seq_len,
                dep_graph_len,
                hidden_size,
                dtype=dep_graph_out.dtype,
                device=dep_graph_out.device,
            )
            dep_graph_all[flat_seq_mask, :, :] = dep_graph_out

            if isinstance(dep_graph_module_return_kwargs, dict) and (
                "present_key_value" in dep_graph_module_return_kwargs
            ):
                present_key_value_tuple = dep_graph_module_return_kwargs["present_key_value"]
                out_tuple = []
                for present_key_value in present_key_value_tuple:
                    present_key_value_all = torch.zeros(
                        bsz * seq_len,
                        *present_key_value.shape[1:],
                        dtype=present_key_value.dtype,
                        device=present_key_value.device,
                    )
                    present_key_value_all[flat_seq_mask, ...] = present_key_value
                    out_tuple.append(present_key_value_all)
                dep_graph_module_return_kwargs["present_key_value"] = tuple(out_tuple)

        dep_graph_all = torch.reshape(dep_graph_all, (bsz, seq_len, dep_graph_len, hidden_size))

        # And, with that, we're done.
        extra_return_kwargs = {
            "seq_module": seq_module_return_kwargs,
            "dep_graph_module": dep_graph_module_return_kwargs,
        }

        return dep_graph_all, extra_return_kwargs

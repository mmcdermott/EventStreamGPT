from typing import Any

import torch

from ..data.types import DataModality, PytorchBatch
from .config import StructuredEventProcessingMode, StructuredTransformerConfig
from .generation.generation_utils import StructuredGenerationMixin
from .model_output import (
    GenerativeOutputLayerBase,
    GenerativeSequenceModelLabels,
    GenerativeSequenceModelLosses,
    GenerativeSequenceModelOutput,
    GenerativeSequenceModelPredictions,
)
from .transformer import (
    ConditionallyIndependentPointProcessTransformer,
    StructuredTransformerPreTrainedModel,
    expand_mask,
    time_from_deltas,
)


class ConditionallyIndependentGenerativeOutputLayer(GenerativeOutputLayerBase):
    # TODO(mmd): Allow for use of NLL-beta throughout?
    # TODO(mmd): Per-subject, NLL should be averaged over total duration, not # of events?
    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__(config)
        if config.structured_event_processing_mode != StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
            raise ValueError(f"{config.structured_event_processing_mode} invalid!")

    def forward(
        self,
        batch: PytorchBatch,
        encoded: torch.FloatTensor,
        is_generation: bool = False,
    ) -> GenerativeSequenceModelOutput:
        # These are the containers we'll use to process the outputs
        classification_dists_by_measurement = {}
        classification_losses_by_measurement = None if is_generation else {}
        classification_labels_by_measurement = None if is_generation else {}
        regression_dists = {}
        regression_loss_values = None if is_generation else {}
        regression_labels = None if is_generation else {}
        regression_indices = None if is_generation else {}

        classification_measurements = set(self.classification_mode_per_measurement.keys())
        regression_measurements = set(
            self.config.measurements_for(DataModality.MULTIVARIATE_REGRESSION)
            + self.config.measurements_for(DataModality.UNIVARIATE_REGRESSION)
        )

        # encoded is of shape: (batch size, sequence length, config.hidden_size)
        bsz, seq_len, _ = encoded.shape
        whole_event_encoded = encoded

        # In this case, the whole_event_encoded representation actually is used to predict the next event's
        # contents, so it is what we want if we are in generative mode, but if we are not in generative mode
        # then to make it align with the labels we need to shift it to be in the right form. In particular, we
        # prepend a vector of zeros to be used to predict the contents of the first event (excluding the TTE
        # of the first event which is guaranteed to be zero) and we _don't_ predict the contents of the event
        # after the end of this sequence (as we have no way to judge them).

        if is_generation:
            for_event_contents_prediction = whole_event_encoded
        else:
            for_event_contents_prediction = torch.cat(
                (
                    torch.zeros_like(whole_event_encoded[:, 0, :]).unsqueeze(1),
                    whole_event_encoded[:, :-1, :],
                ),
                dim=1,
            )

        classification_out = self.get_classification_outputs(
            batch,
            for_event_contents_prediction,
            classification_measurements,
        )
        classification_dists_by_measurement.update(classification_out[1])
        if not is_generation:
            classification_losses_by_measurement.update(classification_out[0])
            classification_labels_by_measurement.update(classification_out[2])

        regression_out = self.get_regression_outputs(
            batch,
            for_event_contents_prediction,
            regression_measurements,
            is_generation=is_generation,
        )
        regression_dists.update(regression_out[1])
        if not is_generation:
            regression_loss_values.update(regression_out[0])
            regression_labels.update(regression_out[2])
            regression_indices.update(regression_out[3])

        TTE_LL_overall, TTE_dist, TTE_true = self.get_TTE_outputs(
            batch,
            whole_event_encoded,
            is_generation=is_generation,
        )

        return GenerativeSequenceModelOutput(
            **{
                "loss": (
                    sum(classification_losses_by_measurement.values())
                    + sum(regression_loss_values.values())
                    - TTE_LL_overall
                )
                if not is_generation
                else None,
                "losses": GenerativeSequenceModelLosses(
                    **{
                        "classification": classification_losses_by_measurement,
                        "regression": regression_loss_values,
                        "time_to_event": None if is_generation else -TTE_LL_overall,
                    }
                ),
                "preds": GenerativeSequenceModelPredictions(
                    classification=classification_dists_by_measurement,
                    regression=regression_dists,
                    regression_indices=regression_indices,
                    time_to_event=TTE_dist,
                ),
                "labels": GenerativeSequenceModelLabels(
                    classification=classification_labels_by_measurement,
                    regression=regression_labels,
                    regression_indices=regression_indices,
                    time_to_event=None if is_generation else TTE_true,
                ),
                "event_mask": batch["event_mask"],
                "dynamic_values_mask": batch["dynamic_values_mask"],
            }
        )


class CIPPTForGenerativeSequenceModeling(StructuredGenerationMixin, StructuredTransformerPreTrainedModel):
    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__(config)

        if config.structured_event_processing_mode != StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
            raise ValueError(f"{config.structured_event_processing_mode} invalid!")

        self.encoder = ConditionallyIndependentPointProcessTransformer(config)
        self.output_layer = ConditionallyIndependentGenerativeOutputLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(self, batch: PytorchBatch, past=None, **kwargs) -> dict[str, Any]:
        # only last sequence element in the batch if past is defined in kwargs
        batch.time = time_from_deltas(batch)

        use_cache = kwargs.get("use_cache", False)
        if not use_cache:
            return {**kwargs, "batch": batch}

        seq_attention_mask = expand_mask(batch.event_mask, batch.time_delta.dtype)

        dep_graph_el_generation_target = kwargs.get("dep_graph_el_generation_target", None)
        if dep_graph_el_generation_target is not None:
            raise ValueError(
                f"Can't use dep_graph_el_generation_target ({dep_graph_el_generation_target}) "
                "in a conditionally independent model."
            )

        match past:
            case None:
                pass

            case tuple():
                batch = batch.last_sequence_element_unsqueezed()

            case _:
                raise ValueError(f"{past} malformed!")

        return {
            **kwargs,
            "seq_attention_mask": seq_attention_mask,
            "batch": batch,
            "past": past,
        }

    def forward(
        self, batch: PytorchBatch, is_generation: bool = False, **kwargs
    ) -> GenerativeSequenceModelOutput:
        use_cache = kwargs.get("use_cache", False)
        output_attentions = kwargs.get("output_attentions", False)
        output_hidden_states = kwargs.get("output_hidden_states", False)

        encoded = self.encoder(batch, **kwargs)

        output = self.output_layer(batch, encoded.last_hidden_state, is_generation=is_generation)

        if use_cache:
            output["past_key_values"] = encoded.past_key_values

        if output_attentions:
            output["attentions"] = encoded.attentions

        if output_hidden_states:
            output["hidden_states"] = encoded.hidden_states

        return output

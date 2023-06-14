"""The nested attention core event stream GPT model."""
from typing import Any

import torch

from ..data.data_embedding_layer import MeasIndexGroupOptions
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
    NestedAttentionPointProcessTransformer,
    StructuredTransformerPreTrainedModel,
    expand_mask,
    time_from_deltas,
)


class NestedAttentionGenerativeOutputLayer(GenerativeOutputLayerBase):
    """The output layer for the nested attention event stream model.

    TODO(mmcdermott):
        Allow for use of NLL-beta throughout? https://github.com/mmcdermott/EventStreamGPT/issues/26

    Args:
        config: The overall model configuration.

    Raises:
        ValueError: If the model configuration does not indicate nested attention mode.
    """

    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__(config)

        if config.structured_event_processing_mode != StructuredEventProcessingMode.NESTED_ATTENTION:
            raise ValueError(f"{config.structured_event_processing_mode} invalid for this model!")

    def forward(
        self,
        batch: PytorchBatch,
        encoded: torch.FloatTensor,
        is_generation: bool = False,
        dep_graph_el_generation_target: int | None = None,
    ) -> GenerativeSequenceModelOutput:
        """Returns the overall model output for the input batch.

        It takes the final hidden states from the encoder and runs them through various output layers to
        predict subsequent event timing and contents. It's difference from a conditionally independent variant
        is largely in that it predicts dependency graph elements sequentially, relying on prior graph elements
        at each stage.

        Args:
            batch: The batch of data to process.
            encoded: The encoded representation of the input data. This is of shape (batch size, sequence
                length, dependency graph len, config.hidden_size). The last element of the dependency graph is
                always the whole-event embedding, and the first element of the dependency graph is always
                assumed to capture the time of the event.
            is_generation: Whether or not we are in generation mode. If so, the output predictions are for the
                next event for both time and event contents; if not, then we shift the event contents
                predictoin back by one event in order to align with the labels.
            dep_graph_el_generation_target: If is_generation is True, this is the index of the dependency
                graph element for which we are generating for this pass. If None, we generate all elements of
                the dependency graph (even though, for a nested attention model, this is generally wrong as we
                need to generate dependency graph elements in dependency graph order).
        """

        if dep_graph_el_generation_target is not None and not is_generation:
            raise ValueError(
                f"If dep_graph_el_generation_target ({dep_graph_el_generation_target}) is not None, "
                f"is_generation ({is_generation}) must be True!"
            )
        torch._assert(
            ~torch.isnan(encoded).any(),
            f"{torch.isnan(encoded).sum()} NaNs in encoded (target={dep_graph_el_generation_target})",
        )

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

        bsz, seq_len, dep_graph_len, _ = encoded.shape

        if is_generation:
            if dep_graph_el_generation_target is None or dep_graph_el_generation_target == 0:
                dep_graph_loop = None
                do_TTE = True
            else:
                if dep_graph_len == 1:
                    # This case can trigger when use_cache is True.
                    dep_graph_loop = range(1, 2)
                else:
                    dep_graph_loop = range(dep_graph_el_generation_target, dep_graph_el_generation_target + 1)
                do_TTE = False
        else:
            dep_graph_loop = range(1, dep_graph_len)
            do_TTE = True

        if dep_graph_loop is not None:
            # Now we need to walk through the other elements of the dependency graph (omitting the first
            # entry, which reflects time-only dependent values and so is covered by predicting TTE).
            for i in dep_graph_loop:
                # In this case, this level of the dependency graph is presumed to be used to
                # predict the data types listed in `self.config.measurements_per_dep_graph_level`.
                dep_graph_level_encoded = encoded[:, :, i - 1, :]
                # dep_graph_level_encoded is of shape (batch size, sequence length, hidden size)

                if dep_graph_el_generation_target is not None:
                    target_idx = dep_graph_el_generation_target
                else:
                    target_idx = i

                categorical_measurements_in_level = set()
                numerical_measurements_in_level = set()
                for measurement in self.config.measurements_per_dep_graph_level[target_idx]:
                    if type(measurement) in (tuple, list):
                        measurement, mode = measurement
                    else:
                        mode = MeasIndexGroupOptions.CATEGORICAL_AND_NUMERICAL

                    match mode:
                        case MeasIndexGroupOptions.CATEGORICAL_AND_NUMERICAL:
                            categorical_measurements_in_level.add(measurement)
                            numerical_measurements_in_level.add(measurement)
                        case MeasIndexGroupOptions.CATEGORICAL_ONLY:
                            categorical_measurements_in_level.add(measurement)
                        case MeasIndexGroupOptions.NUMERICAL_ONLY:
                            numerical_measurements_in_level.add(measurement)
                        case _:
                            raise ValueError(f"Unknown mode {mode}")

                classification_measurements_in_level = categorical_measurements_in_level.intersection(
                    classification_measurements
                )
                regression_measurements_in_level = numerical_measurements_in_level.intersection(
                    regression_measurements
                )

                torch._assert(
                    ~torch.isnan(dep_graph_level_encoded).any(),
                    (
                        f"{torch.isnan(dep_graph_level_encoded).sum()} NaNs in dep_graph_level_encoded "
                        f"({target_idx}, {i})"
                    ),
                )
                classification_out = self.get_classification_outputs(
                    batch,
                    dep_graph_level_encoded,
                    classification_measurements_in_level,
                )
                classification_dists_by_measurement.update(classification_out[1])
                if not is_generation:
                    classification_losses_by_measurement.update(classification_out[0])
                    classification_labels_by_measurement.update(classification_out[2])

                regression_out = self.get_regression_outputs(
                    batch,
                    dep_graph_level_encoded,
                    regression_measurements_in_level,
                    is_generation=is_generation,
                )
                regression_dists.update(regression_out[1])
                if not is_generation:
                    regression_loss_values.update(regression_out[0])
                    regression_labels.update(regression_out[2])
                    regression_indices.update(regression_out[3])

        if do_TTE:
            whole_event_encoded = encoded[:, :, -1, :]
            TTE_LL_overall, TTE_dist, TTE_true = self.get_TTE_outputs(
                batch,
                whole_event_encoded,
                is_generation=is_generation,
            )
        else:
            TTE_LL_overall, TTE_dist, TTE_true = None, None, None

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


class NAPPTForGenerativeSequenceModeling(StructuredGenerationMixin, StructuredTransformerPreTrainedModel):
    """The end-to-end model for nested attention generative sequence modelling.

    This model is a subclass of :class:`~transformers.StructuredTransformerPreTrainedModel` and is designed
    for generative pre-training over "event-stream" data, with inputs in the form of `PytorchBatch` objects.
    It is trained to solve the generative, multivariate, masked temporal point process problem over the
    defined measurements in the input data. It does so while respecting intra-event causal dependencies
    specified through the measurements_per_dep_graph_level specified in the config (aka the dependency graph).

    This model largely simply passes the input data through a `NestedAttentionPointProcessTransformer`
    followed by a `NestedAttentionGenerativeOutputLayer`.

    Args:
        config: The overall model configuration.

    Raises:
        ValueError: If the model configuration does not indicate nested attention mode.
    """

    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__(config)

        if config.structured_event_processing_mode != StructuredEventProcessingMode.NESTED_ATTENTION:
            raise ValueError(f"{config.structured_event_processing_mode} invalid for this model!")

        self.encoder = NestedAttentionPointProcessTransformer(config)
        self.output_layer = NestedAttentionGenerativeOutputLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(
        self, batch: PytorchBatch, past: dict[str, tuple] | None = None, **kwargs
    ) -> dict[str, Any]:
        """Returns model keyword arguments that have been modified for generation purposes.

        Args:
            batch: The batch of data to be transformed.
            past: The past state of the model, if any. If specified, it must be a dictionary containing both
                the seq_past key (the past of the sequential attention module) and a dep_graph_past key (the
                past of the dependency graph attention module). These inner past encodings are tuples
                containing the past values over prior layers and heads.
            **kwargs: Additional keyword arguments. If "use_cache" is set in the kwargs to False, then the
                past state is ignored. If not, then the past state is passed through the model to accelerate
                generation, if past is not None then the batch is trimmed to the last element in the sequence,
                and the sequential attention mask is pre-computed.

        Raises:
            ValueError: If the past state is malformed or if there is a dep_graph_el_generation_target in the
                kwargs that is not None.
        """

        use_cache = kwargs.get("use_cache", False)
        if not use_cache:
            return {**kwargs, "batch": batch}

        dep_graph_el_generation_target = kwargs.get("dep_graph_el_generation_target", None)
        seq_attention_mask = expand_mask(batch.event_mask, batch.time_delta.dtype)

        match past:
            case None:
                dep_graph_past = None

                if dep_graph_el_generation_target is not None:
                    raise ValueError(f"Can't have dep target {dep_graph_el_generation_target} without past")

            case dict() as pasts_dict if "seq_past" in pasts_dict and "dep_graph_past" in pasts_dict:
                past = pasts_dict["seq_past"]

                # only last sequence element in the batch if past is defined in kwargs
                batch.time = time_from_deltas(batch)
                batch = batch.last_sequence_element_unsqueezed()

                dep_graph_past = pasts_dict["dep_graph_past"]
                if dep_graph_past is not None and dep_graph_el_generation_target is None:
                    raise ValueError(
                        "Trying to use generate with a past without a dep graph generation target!"
                    )
                elif dep_graph_past is None and dep_graph_el_generation_target is not None:
                    raise ValueError("Trying to target only one dep graph element without past!")

            case _:
                raise ValueError(f"{past} malformed!")

        return {
            **kwargs,
            "batch": batch,
            "past": past,
            "dep_graph_past": dep_graph_past,
            "seq_attention_mask": seq_attention_mask,
        }

    def forward(
        self, batch: PytorchBatch, is_generation: bool = False, **kwargs
    ) -> GenerativeSequenceModelOutput:
        """This runs the full forward pass of the model.

        Args:
            batch: The batch of data to be transformed.
            is_generation: Whether or not the model is being used for generation.
            **kwargs: Additional keyword arguments, which are used for output structuring and are forwarded to
                the encoder. The model specifically looks for use_cache, output_attentions, and
                output_hidden_states keyword arguments, which control whether additional properties should be
                added to the output. In addition, the model also looks for the dep_graph_el_generation_target
                keyword argument, which is passed to the output layer.

        Returns:
            The output of the model, which is a `GenerativeSequenceModelOutput` object.
        """

        use_cache = kwargs.get("use_cache", False)
        output_attentions = kwargs.get("output_attentions", False)
        output_hidden_states = kwargs.get("output_hidden_states", False)

        encoded = self.encoder(batch, **kwargs)

        output = self.output_layer(
            batch,
            encoded.last_hidden_state,
            is_generation=is_generation,
            dep_graph_el_generation_target=kwargs.get("dep_graph_el_generation_target", None),
        )

        if use_cache:
            output["past_key_values"] = encoded.past_key_values

        if output_attentions:
            output["attentions"] = encoded.attentions

        if output_hidden_states:
            output["hidden_states"] = encoded.hidden_states

        return output

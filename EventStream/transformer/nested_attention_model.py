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
    time_from_deltas,
)


class NestedAttentionGenerativeOutputLayer(GenerativeOutputLayerBase):
    # TODO(mmd): Allow for use of NLL-beta throughout?
    # TODO(mmd): Per-subject, NLL should be averaged over total duration, not # of events?
    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__(config)

        if (
            config.structured_event_processing_mode
            != StructuredEventProcessingMode.NESTED_ATTENTION
        ):
            raise ValueError(f"{config.structured_event_processing_mode} invalid for this model!")

    def forward(
        self,
        batch: PytorchBatch,
        encoded: torch.FloatTensor,
        is_generation: bool = False,
        dep_graph_el_generation_target: int | None = None,
    ) -> GenerativeSequenceModelOutput:
        # encoded is of shape:
        # (batch size, sequence length, dependency graph len, config.hidden_size)
        # In this case, the last element of the dependency graph is always the whole-event embedding, and the
        # first element of the dependency graph is always assumed to be the time of the event.

        if dep_graph_el_generation_target is not None and not is_generation:
            raise ValueError(
                f"If dep_graph_el_generation_target ({dep_graph_el_generation_target}) is not None, "
                f"is_generation ({is_generation}) must be True!"
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

        event_type_mask_per_measurement = self.get_event_type_mask_per_measurement(batch)

        bsz, seq_len, dep_graph_len, _ = encoded.shape

        if dep_graph_el_generation_target is not None:
            if dep_graph_el_generation_target != 0:
                assert dep_graph_len == 1
                dep_graph_loop = range(1, 2)
                do_TTE = False
            else:
                dep_graph_loop = None
                do_TTE = True
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

                if self.config.measurements_per_dep_graph_level is None:
                    raise ValueError
                else:
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

                    classification_measurements_in_level = (
                        categorical_measurements_in_level.intersection(classification_measurements)
                    )
                    regression_measurements_in_level = (
                        numerical_measurements_in_level.intersection(regression_measurements)
                    )

                classification_out = self.get_classification_outputs(
                    batch,
                    dep_graph_level_encoded,
                    classification_measurements_in_level,
                    event_type_mask_per_measurement=event_type_mask_per_measurement,
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
                    event_type_mask_per_measurement=event_type_mask_per_measurement,
                )
                regression_dists.update(regression_out[1])
                if not is_generation:
                    regression_loss_values.update(regression_out[0])
                    regression_labels.update(regression_out[2])
                    regression_indices.update(regression_out[3])

        if do_TTE:
            # Now we need to walk through the other elements of the dependency graph (omitting the first
            # `whole_event_encoded` is of shape (batch size, sequence length, hidden size)
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
                "event_type_mask_per_measurement": event_type_mask_per_measurement,
                "event_mask": batch["event_mask"],
                "dynamic_values_mask": batch["dynamic_values_mask"],
            }
        )


class NAPPTForGenerativeSequenceModeling(
    StructuredGenerationMixin, StructuredTransformerPreTrainedModel
):
    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__(config)

        if (
            config.structured_event_processing_mode
            != StructuredEventProcessingMode.NESTED_ATTENTION
        ):
            raise ValueError(f"{config.structured_event_processing_mode} invalid for this model!")

        self.encoder = NestedAttentionPointProcessTransformer(config)
        self.output_layer = NestedAttentionGenerativeOutputLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(
        self, batch: PytorchBatch, past=None, **kwargs
    ) -> dict[str, Any]:
        # only last sequence element in the batch if past is defined in kwargs
        batch.time = time_from_deltas(batch.time_delta)

        use_cache = kwargs.get("use_cache", False)
        if not use_cache:
            return {**kwargs, "batch": batch}

        dep_graph_el_generation_target = kwargs.get("dep_graph_el_generation_target", None)

        match past:
            case None:
                dep_graph_past = None

                if (
                    dep_graph_el_generation_target is not None
                    and dep_graph_el_generation_target != 0
                ):
                    raise ValueError(
                        f"Can't have dep target {dep_graph_el_generation_target} without past"
                    )

            case dict() as pasts_dict if "seq_past" in pasts_dict and "dep_graph_past" in pasts_dict:
                past = pasts_dict["seq_past"]
                batch = batch.last_sequence_element_unsqueezed()

                dep_graph_past = pasts_dict["dep_graph_past"]
                if dep_graph_el_generation_target is None:
                    if dep_graph_past is not None:
                        raise ValueError(
                            "Trying to use dep_graph_past without a generation target!"
                        )
                elif dep_graph_el_generation_target <= 1:
                    # We're on a new sequence or dep graph element, so any dep_graph_past that was retained is
                    # now null and void.
                    dep_graph_past = None
                else:
                    if dep_graph_past is None:
                        raise ValueError(
                            "Trying to target only one dep graph element without past!"
                        )

            case _:
                raise ValueError(f"{past} malformed!")

        return {
            **kwargs,
            "batch": batch,
            "past": past,
            "dep_graph_past": dep_graph_past,
        }

    def forward(
        self, batch: PytorchBatch, is_generation: bool = False, **kwargs
    ) -> GenerativeSequenceModelOutput:
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

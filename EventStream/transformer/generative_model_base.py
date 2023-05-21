import torch

from ..data.types import DataModality, PytorchBatch
from .config import StructuredTransformerConfig, TimeToEventGenerationHeadType
from .generative_layers import (
    ExponentialTTELayer,
    GaussianIndexedRegressionLayer,
    GaussianRegressionLayer,
    LogNormalMixtureTTELayer,
)
from .model_output import get_event_type_mask_per_measurement
from .utils import safe_weighted_avg, str_summary, weighted_loss


class GenerativeOutputLayerBase(torch.nn.Module):
    # TODO(mmd): Allow for use of NLL-beta throughout?
    # TODO(mmd): Per-subject, NLL should be averaged over total duration, not # of events?
    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__()

        self.config = config

        match self.config.TTE_generation_layer_type:
            case TimeToEventGenerationHeadType.LOG_NORMAL_MIXTURE:
                self.TTE_layer = LogNormalMixtureTTELayer(
                    in_dim=config.hidden_size,
                    num_components=config.TTE_lognormal_generation_num_components,
                    mean_log_inter_time=config.mean_log_inter_event_time_min,
                    std_log_inter_time=config.std_log_inter_event_time_min,
                )
            case TimeToEventGenerationHeadType.EXPONENTIAL:
                self.TTE_layer = ExponentialTTELayer(in_dim=config.hidden_size)
            case _:
                raise ValueError(
                    f"Invalid option for `config.TTE_generation_layer_type`. Must be "
                    f"a member of the `TimeToEventGenerationHeadType` enum: "
                    f"({TimeToEventGenerationHeadType.values()}). got {config.TTE_generation_layer_type}."
                )

        self.ClassificationLayer = torch.nn.Linear(config.hidden_size, config.vocab_size)

        self.classification_criteria = {}
        for measurement in config.measurements_for(DataModality.SINGLE_LABEL_CLASSIFICATION):
            self.classification_criteria[measurement] = torch.nn.CrossEntropyLoss(reduction="none")
        for measurement in config.measurements_for(DataModality.MULTI_LABEL_CLASSIFICATION):
            self.classification_criteria[measurement] = torch.nn.BCEWithLogitsLoss(
                reduction="none"
            )

        self.regression_layers = torch.nn.ModuleDict({})
        for measurement in config.measurements_for(DataModality.MULTIVARIATE_REGRESSION):
            self.regression_layers[measurement] = GaussianIndexedRegressionLayer(
                n_regression_targets=config.vocab_sizes_by_measurement[measurement],
                in_dim=config.hidden_size,
            )
        for measurement in config.measurements_for(DataModality.UNIVARIATE_REGRESSION):
            if measurement in self.regression_layers:
                raise ValueError(f"{measurement} duplicated!")
            self.regression_layers[measurement] = GaussianRegressionLayer(
                in_dim=config.hidden_size
            )

        self.classification_mode_per_measurement = {}
        for generative_mode, measurements in config.measurements_per_generative_mode.items():
            if generative_mode in (
                DataModality.MULTIVARIATE_REGRESSION,
                DataModality.UNIVARIATE_REGRESSION,
            ):
                continue
            for measurement in measurements:
                assert measurement not in self.classification_mode_per_measurement
                self.classification_mode_per_measurement[measurement] = generative_mode

    def get_TTE_outputs(
        self, batch: PytorchBatch, encoded: torch.FloatTensor, is_generation: bool = False
    ) -> tuple[torch.FloatTensor, torch.distributions.Distribution, torch.FloatTensor,]:
        """Produces time-to-event predictions and log likelihoods (_not NLLs!_) for the model.

        Args:
            `batch` (`PytorchBatch`):
                The batch of data for which the classification predictions are desired.
            `encoded` (`torch.FloatTensor`, shape is batch_size X sequence_length X hidden_dim):
                The final encodings _to be used to predict the time from the event at that position to the
                subsequent event_. For example, the vector `encoded[i][j]` (which is of size `hidden_dim` and
                corresponds to event `j` for batch element `i`) is
                _not_ used to predict the time from event `j-1` to event `j`, but rather is used to predict
                the time from event `j` to event `j+1` (all for batch index `i`, of course). _Note that this
                is shifted from how `encoded` is used in other functions in this class._

        Returns:
            `TTE_LL` (`torch.FloatTensor`):
                A torch scalar containing the average log-likelihood of observed time-to-events given the
                predicted distribution. Averaging is done over all unmasked events per batch element first,
                then in a macro manner over all batch elements.
                TODO(mmd): Should probably be NLL, not LL.
            `TTE_dist` (`torch.distributions.Distribution`):
                The predicted torch Distribution for modelling time-to-event. The distribution's shape is such
                that samples drawn from the distribution will have shape `[batch_size, sequence_length]` and
                `sample[i][j]` will be a prediction for the time between events `j` and `j+1` for batch
                element `i` (note that this includes a prediction for the time until the event after the end
                of the sequence, though such an event is naturally not observed).
            `TTE_true` (`torch.FloatTensor`):
                A tensor of shape `[batch_size, sequence_length - 1]` such that `TTE_true[i][j]` contains the
                observed time between events `j` and `j+1` for batch element `i`.
        """
        TTE_dist = self.TTE_layer(encoded)

        if is_generation:
            return None, TTE_dist, None

        # TTE_dist is a distribution with random variables of shape (batch size, sequence length)
        TTE_obs_mask = batch["event_mask"][:, 1:] & batch["event_mask"][:, :-1]
        TTE_delta = batch["time_delta"][:, :-1]
        TTE_true = torch.where(TTE_obs_mask, TTE_delta, torch.ones_like(TTE_delta))

        # As TTE_dist contains a predicted distribution for the last sequence element, which we want to return
        # for generative purposes, we add a fake observation to the last element.
        TTE_true_exp = torch.cat(
            (TTE_true, torch.ones_like(TTE_true[:, -1]).unsqueeze(-1)), dim=-1
        )
        TTE_obs_mask_exp = torch.cat(
            (TTE_obs_mask, torch.zeros_like(TTE_obs_mask[:, -1]).unsqueeze(-1)), dim=-1
        )

        # We skip the last event as we have no true time to event for that event.
        # TODO(mmd): Use NLL-\beta?
        try:
            TTE_LL = TTE_dist.log_prob(TTE_true_exp)
        except ValueError as e:
            print(f"Failed to compute TTE log prob on input {str_summary(TTE_true_exp)}: {e}")
            raise

        if TTE_obs_mask_exp.isnan().any():
            raise ValueError(f"NaNs in TTE_obs_mask_exp: {batch}")
        elif TTE_true_exp.isnan().any():
            raise ValueError(f"NaNs in TTE_true_exp: {batch}")
        elif TTE_LL.isnan().any():
            raise ValueError(f"NaNs in TTE_LL: {batch}")
        elif (TTE_obs_mask_exp.float().sum(-1) == 0).any():
            raise ValueError(f"No observed time-to-event for >= 1 patient in batch: {batch}")

        TTE_LL_per_patient = (TTE_LL * TTE_obs_mask_exp.float()).sum(
            -1
        ) / TTE_obs_mask_exp.float().sum(-1)
        TTE_LL_overall = TTE_LL_per_patient.mean()

        return TTE_LL_overall, TTE_dist, TTE_true

    def get_classification_outputs(
        self,
        batch: PytorchBatch,
        encoded: torch.FloatTensor,
        valid_measurements: set[str],
        event_type_mask_per_measurement: dict[str, torch.BoolTensor] | None = None,
    ) -> tuple[
        dict[str, torch.FloatTensor],
        dict[str, torch.FloatTensor],
        dict[str, torch.LongTensor | torch.FloatTensor],
    ]:
        """Produces classification predictions and losses for the model.

        Args:
            `batch` (`PytorchBatch`):
                The batch of data for which the classification predictions are desired.
            `encoded` (`torch.FloatTensor`, shape is batch_size X sequence_length X hidden_dim):
                The final encodings _to be used to predict for each position in the sequence_. For example,
                the vector `encoded[i][j]` (which is of size `hidden_dim`) is _not_ the summary encoding of
                the batch element at batch index `i` and sequence index `j`, but rather is the input to be
                used to form classification predictions corresponding to batch element `i` at sequence
                position `j`.
            `valid_measurements` (`Set[str]`):
                The classification measurements in the batch that should be predicted from this input
                `encoded`.
            `event_type_mask_per_measurement` (`Optional[Dict[str, torch.BoolTensor]]`, defaults to None):
                A dictionary from measurement to a tensor of shape `[batch_size, sequence_length]` such that
                `event_type_mask_per_measurement[measurement][i][j]` is `True` if the event at batch index `i`
                and sequence index `j` is of a type that should be used to form predictions for the
                measurement `measurement`. If `None`, then all events are used to form predictions for all
                measurements.

        Returns:
            `classification_losses_by_measurement` (`Dict[str, torch.FloatTensor]`):
                A dictionary from `measurement` to scalar tensors consisting of the average NLL of the data
                given the classiciation model. Averaging happens via the following procedure:
                  * For multi-label measurements:
                    1. NLL is averaged over labels per sequence event, for all unmasked sequence events (as in
                       theory any event could have observed labels for binary multi-lable predictions).
                       TODO(mmd): this should likely be specific to events with certain event types.
                    2. NLL is macro-averaged across unmasked sequence events per batch element.
                    3. NLL is macro-averaged across batch elements.
                  * For single-task measurements:
                    1. NLL is computed on any event that has a label for that task.
                       TODO(mmd): Maybe should be conditioned on specific event types too?
                    2. NLL is macro-averaged across events which had a label for that task per sequence.
                       Sequences without any events with that label receive a loss of zero.
                    3. NLL is macro-averaged across batch elements.
            `classification_dists_by_measurement` (`Dict[str, torch.FloatTensor]`):
                A dictionary from `measurement` to classification distributions of shape
                `[batch_size X sequence_length X vocabulary_size]` or `[batch_size X sequence_length]`
                reflecting the probabilities for each event for that measurement. Returns scores for all
                events, even those that are masked, including the final event.
            `classification_labels_by_measurement` (`Dict[str, Union[torch.LongTensor, torch.FloatTensor]]`):
                A dictionary from `measurement` to tensors of one of two types:
                  * For multi-label measurements, returns FloatTensors of shape
                    `[batch_size X sequence_length X vocabulary_size]` containing binary labels for each
                    vocabulary element for each event.
                  * For single-label measurements, returns LongTensors of shape
                    `[batch_size, sequence_length]` containing label indices for each event with that task
                    observed, otherwise contains zeros.
        """

        if not valid_measurements:
            return {}, {}, {}

        # Classification of what elements are going to occur:
        classification_scores = self.ClassificationLayer(encoded)

        classification_losses_by_measurement = {}
        classification_dists_by_measurement = {}
        classification_labels_by_measurement = {}

        for measurement, classification_mode in self.classification_mode_per_measurement.items():
            if measurement not in valid_measurements:
                continue

            if event_type_mask_per_measurement is not None and measurement != "event_type":
                event_mask = event_type_mask_per_measurement[measurement] & batch["event_mask"]
            else:
                event_mask = batch["event_mask"]

            measurement_idx = self.config.measurements_idxmap[measurement]
            vocab_start = self.config.vocab_offsets_by_measurement[measurement]
            vocab_end = min(
                o
                for o in list(self.config.vocab_offsets_by_measurement.values())
                + [self.config.vocab_size]
                if o > vocab_start
            )

            scores = classification_scores[:, :, vocab_start:vocab_end]
            # scores is of shape [batch X seq X vocab_end-vocab_start]

            # We don't need to shift here, as given this is a structured model, we'll always rely on elements
            # of the dependency graph that don't include these inputs to predict them (e.g., predict the
            # contents of the event given the time at which the event occurred).
            dynamic_indices = batch["dynamic_indices"]
            tensor_idx = batch["dynamic_measurement_indices"] == measurement_idx

            if classification_mode == DataModality.SINGLE_LABEL_CLASSIFICATION:
                # As there is only one index of this type for this setting,
                # we can directly multiply by the mask and sum
                events_with_label = tensor_idx.any(dim=-1)
                labels = (
                    (dynamic_indices.long() * tensor_idx.long()).sum(dim=-1) - vocab_start
                ) * events_with_label.long()
                # labels is of shape [batch X seq]

                try:
                    loss_per_event = self.classification_criteria[measurement](
                        scores.transpose(1, 2), labels
                    )
                except IndexError as e:
                    print(f"Failed to get loss for {measurement}: {e}!")
                    print(f"vocab_start: {vocab_start}, vocab_end: {vocab_end}")
                    print(f"max(labels): {labels.max()}, min(labels): {labels.min()}")
                    print(
                        f"max(dynamic_indices*tensor_idx): {((dynamic_indices*tensor_idx).max())}, "
                        f"min(dynamic_indices*tensor_idx): {((dynamic_indices*tensor_idx).min())}"
                    )
                    print(f"max(tensor_idx.sum(-1)): {tensor_idx.sum(-1).max()}")
                    print(f"scores.shape: {scores.shape}")
                    raise

                event_mask = event_mask & events_with_label

                dists = torch.distributions.Categorical(logits=scores)

            elif classification_mode == DataModality.MULTI_LABEL_CLASSIFICATION:
                data_labels_or_zero = torch.where(
                    tensor_idx,
                    dynamic_indices - vocab_start + 1,
                    torch.zeros_like(dynamic_indices),
                ).long()

                labels = torch.zeros(
                    scores.shape[0], scores.shape[1], 1 + scores.shape[2], device=scores.device
                ).scatter(
                    dim=2,
                    index=data_labels_or_zero,
                    value=1,
                )

                labels = labels[:, :, 1:]  # Drop the omitted labels...

                loss_per_label = self.classification_criteria[measurement](scores, labels)
                loss_per_event = loss_per_label.mean(dim=-1)

                dists = torch.distributions.Bernoulli(logits=scores)

            else:
                raise ValueError(f"Classification mode {classification_mode} Invalid!")

            loss_overall = weighted_loss(loss_per_event, event_mask)

            classification_losses_by_measurement[measurement] = loss_overall
            classification_dists_by_measurement[measurement] = dists
            classification_labels_by_measurement[measurement] = labels
        return (
            classification_losses_by_measurement,
            classification_dists_by_measurement,
            classification_labels_by_measurement,
        )

    def get_regression_outputs(
        self,
        batch: PytorchBatch,
        encoded: torch.FloatTensor,
        valid_measurements: set[str],
        is_generation: bool = False,
        event_type_mask_per_measurement: dict[str, torch.BoolTensor] | None = None,
    ) -> tuple[
        dict[str, torch.FloatTensor],
        dict[str, torch.distributions.Distribution],
        dict[str, torch.FloatTensor],
        dict[str, torch.LongTensor],
    ]:
        """Produces regression predictions and losses for the model.

        Args:
            `batch` (`PytorchBatch`):
                The batch of data for which the regression predictions are desired.
            `encoded` (`torch.FloatTensor`, shape is batch_size X sequence_length X hidden_dim):
                The final encodings _to be used to predict for each position in the sequence_. For example,
                the vector `encoded[i][j]` (which is of size `hidden_dim`) is _not_ the summary encoding of
                the batch element at batch index `i` and sequence index `j`, but rather is the input to be
                used to form regression predictions corresponding to batch element `i` at sequence
                position `j`.
            `valid_measurements` (`Set[str]`):
                The regression measurements in the batch that should be predicted from this input `encoded`.
            `event_type_mask_per_measurement` (`Optional[Dict[str, torch.BoolTensor]]`, defaults to None):
                A dictionary from measurement to a tensor of shape `[batch_size, sequence_length]` such that
                `event_type_mask_per_measurement[measurement][i][j]` is `True` if the event at batch index `i`
                and sequence index `j` is of a type that should be used to form predictions for the
                measurement `measurement`. If `None`, then all events are used to form predictions for all
                measurements.

        Returns:
            `regression_loss_values` (`Dict[str, torch.FloatTensor]`):
                A dictionary from `measurement` to scalar tensors consisting of the average NLL of the data
                given the regression model. Averaging happens via the following procedure:
                  1. NLL is averaged over data elements of the correct measurement per event.
                     TODO(mmd): This is likely a bit wrong; if a regression task has no observed value, that
                     should be taken into account here but I don't think it is currently.
                  2. Per-event NLLs are averaged over unmasked events with labels per batch element.
                  3. NLL is macro-averaged over the batch.
            `regression_dists` (`Dict[str, torch.distributions.Distribution]`):
                A dictionary from `measurement` to torch distributions modelling the regression targets for
                each data element in each event. In particular, samples from these distributions will have
                shape
                `[batch_size, sequence_length, num_data_elements_per_event]`, such that `sample[i][j][k]` will
                correspond to a prediction for the regression target indexed by
                `batch['dynamic_indices'][i][j][k]`.
            `regression_labels` (`Dict[str, torch.FloatTensor]`):
                A dictionary from `measurement` to tensors of shape
                `[batch_size, sequence_length, num_data_elements_per_event]` containing regression targets for
                each data element, or 0 if that regression target is unobserved.
            `regression_indices` (`Dict[str, torch.LongTensor]`):
                A dictionary from `measurement` to tensors of shape
                `[batch_size, sequence_length, num_data_elements_per_event]` containing the integer index of
                the regression component observed in that position, or 0 if that regression target is
                unobserved. E.g., if we have 200 laboratory tests that we are regressing over, these indices
                state to which laboratory test results the values in `regression_labels` correspond.
        """
        if not valid_measurements:
            return {}, {}, {}, {}

        regression_loss_values = {}
        regression_dists = {}
        regression_labels = {}
        regression_indices = {}
        for measurement in self.config.measurements_for(DataModality.MULTIVARIATE_REGRESSION):
            if measurement not in valid_measurements:
                continue

            if event_type_mask_per_measurement is not None:
                event_mask = event_type_mask_per_measurement[measurement] & batch["event_mask"]
            else:
                event_mask = batch["event_mask"]

            measurement_idx = self.config.measurements_idxmap[measurement]
            vocab_start = self.config.vocab_offsets_by_measurement[measurement]

            # TODO(mmd): If we wanted, we could have `indices_measured_or_zero` reflect just the former part
            # of this `&`, and thus have predictions on all indices, even for those we don't observe values
            # for, but for now this functionality is not required, so we standardize them.
            tensor_idx = (batch["dynamic_measurement_indices"] == measurement_idx) & batch[
                "dynamic_values_mask"
            ]

            indices_measured_or_zero = torch.where(
                tensor_idx,
                batch["dynamic_indices"] - vocab_start,
                torch.zeros_like(batch["dynamic_indices"]),
            ).long()

            regr_dist = self.regression_layers[measurement](
                X=encoded, idx=(None if is_generation else indices_measured_or_zero)
            )

            values_observed_or_zero = torch.where(
                tensor_idx,
                batch["dynamic_values"],
                torch.zeros_like(batch["dynamic_values"]),
            ).float()

            # We don't need to shift here, as given this is a structured model, we'll always rely on elements
            # of the dependency graph that don't include these inputs to predict them (e.g., predict the
            # contents of the event given the time at which the event occurred).

            # TODO(mmd): Use NLL-\beta?
            if is_generation:
                loss_overall = None
            else:
                loss_per_label = -regr_dist.log_prob(values_observed_or_zero)
                loss_per_event, _ = safe_weighted_avg(loss_per_label, tensor_idx)

                events_with_label = event_mask & tensor_idx.any(dim=-1)
                loss_overall = weighted_loss(loss_per_event, events_with_label)

            regression_loss_values[measurement] = loss_overall
            regression_dists[measurement] = regr_dist
            regression_labels[measurement] = values_observed_or_zero
            regression_indices[measurement] = indices_measured_or_zero

        for measurement in self.config.measurements_for(DataModality.UNIVARIATE_REGRESSION):
            if measurement not in valid_measurements:
                continue

            if event_type_mask_per_measurement is not None:
                event_mask = event_type_mask_per_measurement[measurement] & batch["event_mask"]
            else:
                event_mask = batch["event_mask"]

            measurement_idx = self.config.measurements_idxmap[measurement]

            # TODO(mmd): If we wanted, we could have `indices_measured_or_zero` reflect just the former part
            # of this `&`, and thus have predictions on all indices, even for those we don't observe values
            # for, but for now this functionality is not required, so we standardize them.
            tensor_idx = (batch["dynamic_measurement_indices"] == measurement_idx) & batch[
                "dynamic_values_mask"
            ]

            # As there is only one index of this type for this setting,
            # we can directly multiply by the mask and sum
            events_with_label = tensor_idx.any(dim=-1)
            event_mask = event_mask & events_with_label

            regr_dist = self.regression_layers[measurement](X=encoded)

            values_observed_or_zero = (
                torch.where(
                    tensor_idx,
                    batch["dynamic_values"],
                    torch.zeros_like(batch["dynamic_values"]),
                )
                .float()
                .sum(dim=-1)
                * events_with_label.float()
            )
            values_observed_or_zero = values_observed_or_zero.unsqueeze(-1)

            # We don't need to shift here, as given this is a structured model, we'll always rely on elements
            # of the dependency graph that don't include these inputs to predict them (e.g., predict the
            # contents of the event given the time at which the event occurred).

            # TODO(mmd): Use NLL-\beta?
            if is_generation:
                loss_overall = None
            else:
                loss_per_event = -regr_dist.log_prob(values_observed_or_zero).squeeze(-1)

                events_with_label = event_mask & tensor_idx.any(dim=-1)
                loss_overall = weighted_loss(loss_per_event, events_with_label)

            regression_loss_values[measurement] = loss_overall
            regression_dists[measurement] = regr_dist
            regression_labels[measurement] = values_observed_or_zero
            regression_indices[measurement] = None

        return (
            regression_loss_values,
            regression_dists,
            None if is_generation else regression_labels,
            None if is_generation else regression_indices,
        )

    def get_event_type_mask_per_measurement(
        self, batch: PytorchBatch
    ) -> dict[str, torch.BoolTensor | None]:
        return get_event_type_mask_per_measurement(
            batch.dynamic_measurement_indices, batch.dynamic_indices, self.config
        )

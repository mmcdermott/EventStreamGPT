"""Core EventStream GPT model configuration classes.

Attributes:
    MEAS_INDEX_GROUP_T: The type of acceptable measurement index group option specifications.
    ATTENTION_TYPES_LIST_T: The type of acceptable attention type configuration options.
"""
import dataclasses
import enum
import itertools
import math
from collections.abc import Hashable
from typing import Any, Union

from transformers import PretrainedConfig

from ..data.config import MeasurementConfig
from ..data.data_embedding_layer import MeasIndexGroupOptions, StaticEmbeddingMode
from ..data.pytorch_dataset import PytorchDataset
from ..data.types import DataModality
from ..utils import JSONableMixin, StrEnum, hydra_dataclass

MEAS_INDEX_GROUP_T = Union[str, tuple[str, MeasIndexGroupOptions]]


class Split(StrEnum):
    """What data split is being used."""

    TRAIN = enum.auto()
    """The train split."""

    TUNING = enum.auto()
    """The hyperparameter tuning split.

    Also often called "dev", "validation", or "val".
    """

    HELD_OUT = enum.auto()
    """The held out test set split.

    Also often called "test".
    """


class MetricCategories(StrEnum):
    """Describes different categories of metrics.

    Used for configuring what metrics to track.
    """

    LOSS_PARTS = enum.auto()
    """Track the different loss components."""

    TTE = "TTE"
    """Track metrics related to time-to-event prediction."""

    CLASSIFICATION = enum.auto()
    """Track metrics for generative prediction of classification metrics."""

    REGRESSION = enum.auto()
    """Track metrics for generative prediction of regression metrics."""


class Metrics(StrEnum):
    """Describes the different supported metric functions."""

    AUROC = "AUROC"
    """The area under the receiver operating characteristic.

    Also commonly called "AUC".
    """

    AUPRC = "AUPRC"
    """The area under the precision recall curve.

    Also commonly refferred to as "Average Precision".
    """

    ACCURACY = enum.auto()
    """Raw accuracy."""

    EXPLAINED_VARIANCE = enum.auto()
    """The extent to which the predicted regression label explains the variance in the true label."""

    MSE = "MSE"
    """The mean squared error between predicted and true regression labels."""

    MSLE = "MSLE"
    """The mean squared log error between predicted and true regression labels."""


class Averaging(StrEnum):
    """Describes the different ways metric values can be averaged in multi-class or multi-label settings."""

    MACRO = enum.auto()
    """Macro-averaging; Metrics across different labels are averaged without regard for label frequency."""

    MICRO = enum.auto()
    """Micro-averaging; Metrics across different labels are averaged without weighting."""

    WEIGHTED = enum.auto()
    """Weighted-averaging; Metrics across different labels are averaged weighted by label/class frequency."""


@hydra_dataclass
class MetricsConfig(JSONableMixin):
    """An overall configuration for what metrics should be tracked.

    Args:
        n_auc_thresholds: The number of thresholds to be used when computing AUROCs, for memory efficiency.
        do_skip_all_metrics: If `True`, all metrics will be skipped by the model. This can save significant
            time.
        do_validate_args: If `True`, `torchmetrics` metrics objects will validate their arguments during
            computation. This costs time.
        include_metrics: A dictionary detailing what metrics should be tracked over what splits, for what
            measurements, in what ways. If `do_skip_all_metrics`, this will be silently overwritten with {}.
            The format for this dictionary is as follows. The outermost level of keys is splits. Within each
            split, there is another dictionary, whose keys are metric categories that should be tracked in
            some form on that split. Each metric category maps to either the boolean `True`, in which case
            that metric category should be tracked across all relevant metrics, or to a dictionary mapping
            metric functions to either the boolean `True`, indicating they should be tracked over all relevant
            weightings, or to a list of weightings which should be tracked.
    """

    n_auc_thresholds: int | None = 50
    do_skip_all_metrics: bool = False
    do_validate_args: bool = False

    include_metrics: dict[
        # Split, Dict[MetricCategories, Union[bool, Dict[Metrics, Union[bool, List[Averaging]]]]]
        str,
        Any,
    ] = dataclasses.field(
        default_factory=lambda: (
            {
                Split.TUNING: {
                    MetricCategories.LOSS_PARTS: True,
                    MetricCategories.TTE: {Metrics.MSE: True, Metrics.MSLE: True},
                    MetricCategories.CLASSIFICATION: {
                        Metrics.AUROC: [Averaging.WEIGHTED],
                        Metrics.ACCURACY: True,
                    },
                    MetricCategories.REGRESSION: {Metrics.MSE: True},
                },
                Split.HELD_OUT: {
                    MetricCategories.LOSS_PARTS: True,
                    MetricCategories.TTE: {Metrics.MSE: True, Metrics.MSLE: True},
                    MetricCategories.CLASSIFICATION: {
                        Metrics.AUROC: [Averaging.WEIGHTED],
                        Metrics.ACCURACY: True,
                    },
                    MetricCategories.REGRESSION: {Metrics.MSE: True},
                },
            }
        )
    )

    def __post_init__(self):
        if self.do_skip_all_metrics:
            self.include_metrics = {}

    def do_log_only_loss(self, split: Split) -> bool:
        """Returns True if only loss should be logged for this split."""
        if (
            self.do_skip_all_metrics
            or split not in self.include_metrics
            or not self.include_metrics[split]
            or (
                (len(self.include_metrics[split]) == 1)
                and (MetricCategories.LOSS_PARTS in self.include_metrics[split])
            )
        ):
            return True
        else:
            return False

    def do_log(self, split: Split, cat: MetricCategories, metric_name: str | None = None) -> bool:
        """Returns True if `metric_name` should be tracked for `split` and `cat`."""
        if self.do_log_only_loss(split):
            return False

        inc_dict = self.include_metrics[split].get(cat, False)
        if not inc_dict:
            return False
        elif metric_name is None or inc_dict is True:
            return True

        has_averaging = "_" in metric_name.replace("explained_variance", "")
        if not has_averaging:
            return metric_name in inc_dict

        parts = metric_name.split("_")
        averaging = parts[0]
        metric = "_".join(parts[1:])

        permissible_averagings = inc_dict.get(metric, [])
        if (permissible_averagings is True) or (averaging in permissible_averagings):
            return True
        else:
            return False

    def do_log_any(self, cat: MetricCategories, metric_name: str | None = None) -> bool:
        """Returns True if `metric_name` should be tracked for `cat` and any split."""
        for split in Split.values():
            if self.do_log(split, cat, metric_name):
                return True
        return False


@hydra_dataclass
class OptimizationConfig(JSONableMixin):
    """Configuration for optimization variables for training a model.

    Args:
        init_lr: The initial learning rate used by the optimizer. Given warmup is used, this will be the peak
            learning rate after the warmup period.
        end_lr: The final learning rate at the end of all learning rate decay.
        end_lr_frac_of_init_lr: The fraction of the initial learning rate that the end learning rate should
            be. Must be consistent with end_lr, when both are set. If only one is set, the other will be
            correctly inferred upon initialization. This is largely useful during hyperparameter tuning, to
            avoid sampling hyperparameters where ``end_lr`` is larger than ``init_lr``, which is not
            compatible with the supported learning rate scheduler.
        max_epochs: The maximum number of training epochs.
        batch_size: The batch size used during stochastic gradient descent.
        validation_batch_size: The batch size used during evaluation.
        lr_frac_warmup_steps: What fraction of the total training steps should be spent increasing the
            learning rate during the learning rate warmup period. Should not be set simultaneously with
            `lr_num_warmup_steps`. This is largely used in the `set_tot_dataset` function which initializes
            missing parameters given the dataset size, such as inferring the `max_num_training_steps` and
            setting `lr_num_warmup_steps` given this parameter and the inferred `max_num_training_steps`.
        lr_num_warmup_steps: How many training steps should be spent on learning rate warmup. If this is set
            then `lr_frac_warmup_steps` should be set to None, and `lr_frac_warmup_steps` will be properly
            inferred during `set_to_dataset`.
        max_training_steps: The maximum number of training steps the system will run for given `max_epochs`,
            `batch_size`, and the size of the used dataset (as inferred via `set_to_dataset`). Generally
            should not be set at initialization.
        lr_decay_power: The decay power in the learning rate polynomial decay with warmup. 1.0 corresponds to
            linear decay.
        weight_decay: The L2 weight regularization penalty that is applied during training.
        patience: The number of epochs to wait before early stopping if the validation loss does not improve.
            If None, early stopping is not used.
        gradient_accumulation: The number of gradient accumulation steps to use. If None, gradient
            accumulation is not used.

    Raises:
        ValueError: If `end_lr`, `init_lr`, and `end_lr_frac_of_init_lr` are not consistent, or if `end_lr`
            and `end_lr_frac_of_init_lr` are both unset.
    """

    init_lr: float = 1e-2
    end_lr: float | None = None
    end_lr_frac_of_init_lr: float | None = 1e-3
    max_epochs: int = 100
    batch_size: int = 32
    validation_batch_size: int = 32
    lr_frac_warmup_steps: float | None = 0.01
    lr_num_warmup_steps: int | None = None
    max_training_steps: int | None = None
    lr_decay_power: float = 1.0
    weight_decay: float = 0.01
    patience: int | None = None
    gradient_accumulation: int | None = None

    num_dataloader_workers: int = 0

    def __post_init__(self):
        if self.end_lr_frac_of_init_lr is not None:
            if self.end_lr_frac_of_init_lr <= 0.0 or self.end_lr_frac_of_init_lr >= 1.0:
                raise ValueError("`end_lr_frac_of_init_lr` must be between 0.0 and 1.0!")
            if self.end_lr is not None:
                prod = self.end_lr_frac_of_init_lr * self.init_lr
                if not math.isclose(self.end_lr, prod):
                    raise ValueError(
                        "If both set, `end_lr` must be equal to `end_lr_frac_of_init_lr * init_lr`! Got "
                        f"end_lr={self.end_lr}, end_lr_frac_of_init_lr * init_lr = {prod}!"
                    )
            self.end_lr = self.end_lr_frac_of_init_lr * self.init_lr
        else:
            if self.end_lr is None:
                raise ValueError("Must set either end_lr or end_lr_frac_of_init_lr!")
            self.end_lr_frac_of_init_lr = self.end_lr / self.init_lr

    def set_to_dataset(self, dataset: PytorchDataset):
        """Sets parameters in the config to appropriate values given dataset.

        Some parameters for optimization are dependent upon the total size of the dataset (e.g., converting
        between a fraction of training and a concrete number of steps). This function sets these parameters
        based on dataset's size.

        Args:
            dataset: The dataset to set the internal parameters too.

        Raises:
            ValueError: If the setting process does not yield consistent results.
        """

        steps_per_epoch = int(math.ceil(len(dataset) / self.batch_size))

        if self.max_training_steps is None:
            self.max_training_steps = steps_per_epoch * self.max_epochs

        if self.lr_num_warmup_steps is None:
            assert self.lr_frac_warmup_steps is not None
            self.lr_num_warmup_steps = int(round(self.lr_frac_warmup_steps * self.max_training_steps))
        elif self.lr_frac_warmup_steps is None:
            self.lr_frac_warmup_steps = self.lr_num_warmup_steps / self.max_training_steps

        if not (
            math.floor(self.lr_frac_warmup_steps * self.max_training_steps) <= self.lr_num_warmup_steps
        ) and (math.ceil(self.lr_frac_warmup_steps * self.max_training_steps) >= self.lr_num_warmup_steps):
            raise ValueError(
                "`self.lr_frac_warmup_steps`, `self.max_training_steps`, and `self.lr_num_warmup_steps` "
                "should be consistent, but they aren't! Got\n"
                f"\tself.max_training_steps = {self.max_training_steps}\n"
                f"\tself.lr_frac_warmup_steps = {self.lr_frac_warmup_steps}\n"
                f"\tself.lr_num_warmup_steps = {self.lr_num_warmup_steps}"
            )


class StructuredEventProcessingMode(StrEnum):
    """Structured event sequence processing modes."""

    CONDITIONALLY_INDEPENDENT = enum.auto()
    """Intra-event covariates are independent of one another, conditioned on history."""

    NESTED_ATTENTION = enum.auto()
    """Intra-event covariates are predicted according to a user-specified intra-event dependency chain."""


class TimeToEventGenerationHeadType(StrEnum):
    """Options for model TTE generation heads."""

    EXPONENTIAL = enum.auto()
    """TTE is modeled by an exponential distribution with a model-determined rate parameter."""

    LOG_NORMAL_MIXTURE = enum.auto()
    """TTE is modeled by a mixture of log-normal distribiutions."""


class AttentionLayerType(StrEnum):
    """Attention layer type options."""

    GLOBAL = enum.auto()
    """Attention is global over all sequence elements (respecting a causal mask)."""

    LOCAL = enum.auto()
    """Attention is limited to a local window of a config-determined size."""


ATTENTION_TYPES_LIST_T = Union[
    # "global" -- all layers are global.
    AttentionLayerType,
    # ["global", "local"] -- alternate global and local layers until you run out of layers.
    list[AttentionLayerType],
    # [(["global", "local"], 2), (["global"], 1)]
    # Do 2 alternating global and local layers, then 1 global layer.
    list[tuple[list[AttentionLayerType], int]],
]


class StructuredTransformerConfig(PretrainedConfig):
    """The configuration class for Event Stream GPT models.

    It is used to instantiate a Transformer model according to the specified arguments. Depending on the use
    of the model, some parameters will be unused. For example, `measurements_per_generative_mode` and
    parameters in the Model Output Config section are only used for generative tasks, not fine-tuning tasks.

    Configuration objects inherit from `PretrainedConfig` can be used to control the model outputs. Read the
    documentation from `PretrainedConfig` for more information. Of particular interest, note that all
    `PretrainedConfig` objects inherit the following properties, to be used for fine-tuning tasks:

    * finetuning_task (str, optional) — Name of the task used to fine-tune the model. This can be used
      when converting from an original (TensorFlow or PyTorch) checkpoint.
    * id2label (Dict[int, str], optional) — A map from index (for instance prediction index, or target
      index) to label.
    * label2id (Dict[str, int], optional) — A map from label to index for the model.
    * num_labels (int, optional) — Number of labels to use in the last layer added to the model, typically
      for a classification task.
    * task_specific_params (Dict[str, Any], optional) — Additional keyword arguments to store for the
      current task.
    * problem_type (str, optional) — Problem type for fine-tuning models. Can be one of
      "regression", "single_label_classification" or "multi_label_classification".

    Args:
        vocab_sizes_by_measurement: The size of the vocabulary per data type.
        vocab_offsets_by_measurement: The vocab offset per data type.
        measurement_configs: A map per measurement to the fit, pre-processed configuration object for that
            measurement. Used only during generation.
        measurements_idxmap: A map per measurement of the integer index corresponding to that measurement in
            the unified measurements vocabulary.
        measurements_per_generative_mode: Which measurements (by str name) are generated in which mode.
        event_types_idxmap: A map of the integer index corresponding to each event type.
        measurements_per_dep_graph_level: A list of the measurements (by name) and whether or not categorical,
            numerical, or both associated values of that measurement are used in each dependency graph level.
            At the default, this assumes the dependency graph has exactly one non-whole-event level and uses
            that to predict the entirety of the event contents.
        max_seq_len: The maximum sequence length for the model.
        do_split_embeddings: Whether or not embeddings should be split into separate categorical and numerical
            embedding layers, or all embedded jointly. See `DataEmbeddingLayer` for more information.
        categoral_embedding_dim: If specified, the input embedding layer will use a split embedding layer,
            with one embedding for categorical data and one for continuous data.  The embedding dimension for
            the categorical data will be this value. In this case, numerical_embedding_dim must be specified.
        numerical_embedding_dim:
            If specified, the input embedding layer will use a split embedding layer, with one embedding for
            categorical data and one for continuous data.  The embedding dimension for the continuous data
            will be this value. In this case, categoral_embedding_dim must be specified.
        static_embedding_mode:
            Specifies how the static embeddings are combined with dynamic embeddings. Options and their
            effects are described in the `StaticEmbeddingMode` documentation.
        static_embedding_weight:
            The relative weight of the static embedding in the combined embedding.  Only used if the
            `static_embedding_mode` is not `StaticEmbeddingMode.DROP`.
        dynamic_embedding_weight:
            The relative weight of the dynamic embedding in the combined embedding.  Only used if the
            `static_embedding_mode` is not `StaticEmbeddingMode.DROP`.
        categorical_embedding_weight:
            The relative weight of the categorical embedding in the combined embedding.  Only used if
            `categoral_embedding_dim` and `numerical_embedding_dim` are not None.
        numerical_embedding_weight:
            The relative weight of the numerical embedding in the combined embedding.  Only used if
            `categoral_embedding_dim` and `numerical_embedding_dim` are not None.
        do_normalize_by_measurement_index:
            If True, the input embeddings are normalized such that each unique measurement index contributes
            equally to the embedding.
        do_use_learnable_sinusoidal_ATE:
            If True, then the model will produce temporal position embeddings via a sinnusoidal position
            embedding such that the frequencies are learnable, rather than fixed and regular.


        structured_event_processing_mode: Specifies how the internal event is processed internally by the
            model. Can be either:

            1. `StructuredEventProcessingMode.NESTED_ATTENTION`:
               In this case, the whole-event embeddings are processed via a sequential encoder first into
               historical embeddings, then the inter-event dependency graph elements are processed via a
               second sequential encoder alongside the relevant historical embedding.  Sequential processing
               types are either full attention / MLP blocks or just self attention layers, as controlled by
               `do_full_block_in_seq_attention` and `do_full_block_in_dep_graph_attention`.
            2. `StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT`
               In this case, the input dependency graph embedding elements are all summed and processed as a
               single event sequence, with each event's output embedding being used to simultaneously predict
               all elements of the subsequent event (thereby treating them all as conditionally independent).
               In this case, the following parameters should all be None:

               * `measurements_per_dep_graph_level`
               * `do_full_block_in_seq_attention`
               * `do_full_block_in_dep_graph_attention`
               * `dep_graph_attention_types`
               * `dep_graph_window_size`

        hidden_size: The hidden size of the model. Must be consistent with `head_dim`, if specified.
        head_dim: The hidden size per attention head. Useful for hyperparameter tuning to avoid setting
            infeasible hidden sizes. Must be consistent with hidden_size, if specified.
        num_hidden_layers: Number of encoder layers.
        num_attention_heads: Number of attention heads for each attention layer in the Transformer encoder.
        seq_attention_types: The type of attention for each sequence self attention layer.
        seq_window_size: The window size used in local attention for sequence self attention layers.
        dep_graph_attention_types: The type of attention for each dependency graph self attention layer.
            Defaults to global attention as dependency graph sare in general much shorter than sequences.
        dep_graph_window_size: The window size used in local attention for dependency graph self attention
            layers. Default is set much lower as dependency graphs are in general much shorter than sequences.
        do_full_block_in_seq_attention: If True, use a full attention block (including layer normalization and
            MLP layers) for the sequence processing module. If false, just use a self attention layer.
        do_full_block_in_dep_graph_attention: If True, use a full attention block (including layer
            normalization and MLP layers) for the dependency graph processing module. If false, just use a
            self attention layer.
        intermediate_size: Dimension of the "intermediate" (often named feed-forward) layer in encoder.
        activation_function: The non-linear activation function (function or string) in the encoder. If
            string, ``"gelu"`` and ``"relu"`` are supported.
        input_dropout: The dropout probability for the input layer.
        attention_dropout: The dropout probability for the attention probabilities.
        resid_dropout: The dropout probability used on the residual connections.
        layer_norm_epsilon: The epsilon used by the layer normalization layers.
        init_std: The standard deviation of the truncated normal weight initialization distribution.
        TTE_generation_layer_type: What kind of TTE generation layer to use.
        TTE_lognormal_generation_num_components: If the TTE generation layer is ``'log_normal_mixture'``, this
            specifies the number of mixture components to include. Must be `None` if
            ``TTE_generation_layer_type == 'exponential'``.
        mean_log_inter_event_time_min: The mean of the log of the time between events in the underlying data.
            Used for normalizing TTE predictions. Must be `None` if ``TTE_generation_layer_type ==
            'exponential'``.
        std_log_inter_event_time_min: The standard deviation of the log of the time between events in the
            underlying data. Used for normalizing TTE predictions. Must be `None` if
            ``TTE_generation_layer_type == 'exponential'``.
        use_cache: Whether to use the past key/values attentions (if applicable to the model) to speed up
            decoding.

    Raises:
        ValueError: If configuration parameters are not fully self consistent.
    """

    def __init__(
        self,
        # Data configuration
        vocab_sizes_by_measurement: dict[str, int] | None = None,
        vocab_offsets_by_measurement: dict[str, int] | None = None,
        measurement_configs: dict[str, MeasurementConfig] | None = None,
        measurements_idxmap: dict[str, dict[Hashable, int]] | None = None,
        measurements_per_generative_mode: dict[DataModality, list[str]] | None = None,
        event_types_idxmap: dict[str, int] | None = None,
        measurements_per_dep_graph_level: list[list[MEAS_INDEX_GROUP_T]] | None = None,
        max_seq_len: int = 256,
        do_split_embeddings: bool = False,
        categorical_embedding_dim: int | None = None,
        numerical_embedding_dim: int | None = None,
        static_embedding_mode: StaticEmbeddingMode = StaticEmbeddingMode.SUM_ALL,
        static_embedding_weight: float = 0.5,
        dynamic_embedding_weight: float = 0.5,
        categorical_embedding_weight: float = 0.5,
        numerical_embedding_weight: float = 0.5,
        do_normalize_by_measurement_index: bool = False,
        do_use_learnable_sinusoidal_ATE: bool = False,
        # Model configuration
        structured_event_processing_mode: StructuredEventProcessingMode = (
            StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT
        ),
        hidden_size: int | None = None,
        head_dim: int | None = 64,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        seq_attention_types: ATTENTION_TYPES_LIST_T | None = None,
        seq_window_size: int = 32,
        dep_graph_attention_types: ATTENTION_TYPES_LIST_T | None = None,
        dep_graph_window_size: int | None = 2,
        intermediate_size: int = 32,
        activation_function: str = "gelu",
        attention_dropout: float = 0.1,
        input_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        init_std: float = 0.02,
        layer_norm_epsilon: float = 1e-5,
        do_full_block_in_dep_graph_attention: bool | None = True,
        do_full_block_in_seq_attention: bool | None = False,
        # Model output configuration
        TTE_generation_layer_type: TimeToEventGenerationHeadType = "exponential",
        TTE_lognormal_generation_num_components: int | None = None,
        mean_log_inter_event_time_min: float | None = None,
        std_log_inter_event_time_min: float | None = None,
        # For decoding
        use_cache: bool = True,
        **kwargs,
    ):
        self.do_use_learnable_sinusoidal_ATE = do_use_learnable_sinusoidal_ATE
        # Resetting default values to appropriate types
        if vocab_sizes_by_measurement is None:
            vocab_sizes_by_measurement = {}
        if vocab_offsets_by_measurement is None:
            vocab_offsets_by_measurement = {}
        if measurements_idxmap is None:
            measurements_idxmap = {}
        if measurements_per_generative_mode is None:
            measurements_per_generative_mode = {}
        if event_types_idxmap is None:
            event_types_idxmap = {}
        if measurement_configs is None:
            measurement_configs = {}

        self.event_types_idxmap = event_types_idxmap

        if measurement_configs:
            new_meas_configs = {}
            for k, v in measurement_configs.items():
                if type(v) is dict:
                    new_meas_configs[k] = MeasurementConfig.from_dict(v)
                else:
                    new_meas_configs[k] = v
            measurement_configs = new_meas_configs
        self.measurement_configs = measurement_configs

        if do_split_embeddings:
            if not type(categorical_embedding_dim) is int and categorical_embedding_dim > 0:
                raise ValueError(
                    f"When do_split_embeddings={do_split_embeddings}, categorical_embedding_dim must be "
                    f"a positive integer. Got {categorical_embedding_dim}."
                )
            if not type(numerical_embedding_dim) is int and numerical_embedding_dim > 0:
                raise ValueError(
                    f"When do_split_embeddings={do_split_embeddings}, numerical_embedding_dim must be "
                    f"a positive integer. Got {numerical_embedding_dim}."
                )
        else:
            if categorical_embedding_dim is not None:
                print(
                    f"WARNING: categorical_embedding_dim is set to {categorical_embedding_dim} but "
                    f"do_split_embeddings={do_split_embeddings}. Setting categorical_embedding_dim to None."
                )
                categorical_embedding_dim = None
            if numerical_embedding_dim is not None:
                print(
                    f"WARNING: numerical_embedding_dim is set to {numerical_embedding_dim} but "
                    f"do_split_embeddings={do_split_embeddings}. Setting numerical_embedding_dim to None."
                )
                numerical_embedding_dim = None
        self.do_split_embeddings = do_split_embeddings

        self.categorical_embedding_dim = categorical_embedding_dim
        self.numerical_embedding_dim = numerical_embedding_dim
        self.static_embedding_mode = static_embedding_mode
        self.static_embedding_weight = static_embedding_weight
        self.dynamic_embedding_weight = dynamic_embedding_weight
        self.categorical_embedding_weight = categorical_embedding_weight
        self.numerical_embedding_weight = numerical_embedding_weight
        self.do_normalize_by_measurement_index = do_normalize_by_measurement_index

        missing_param_err_tmpl = f"For a {structured_event_processing_mode} model, {{}} should not be None"
        extra_param_err_tmpl = (
            f"WARNING: For a {structured_event_processing_mode} model, {{}} is not used; got {{}}. Setting "
            "to None."
        )
        match structured_event_processing_mode:
            case StructuredEventProcessingMode.NESTED_ATTENTION:
                if do_full_block_in_seq_attention is None:
                    raise ValueError(missing_param_err_tmpl.format("do_full_block_in_seq_attention"))
                if do_full_block_in_dep_graph_attention is None:
                    raise ValueError(missing_param_err_tmpl.format("do_full_block_in_dep_graph_attention"))
                if measurements_per_dep_graph_level is None:
                    raise ValueError(missing_param_err_tmpl.format("measurements_per_dep_graph_level"))

                proc_measurements_per_dep_graph_level = []
                for group in measurements_per_dep_graph_level:
                    proc_group = []
                    for meas_index in group:
                        match meas_index:
                            case str():
                                proc_group.append(meas_index)
                            case [str() as meas_index, (str() | MeasIndexGroupOptions()) as mode]:
                                assert mode in MeasIndexGroupOptions.values()
                                proc_group.append((meas_index, mode))
                            case _:
                                raise ValueError(
                                    f"Invalid `measurements_per_dep_graph_level` entry {meas_index}."
                                )
                    proc_measurements_per_dep_graph_level.append(proc_group)
                measurements_per_dep_graph_level = proc_measurements_per_dep_graph_level

            case StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
                if measurements_per_dep_graph_level is not None:
                    print(
                        extra_param_err_tmpl.format(
                            "measurements_per_dep_graph_level", measurements_per_dep_graph_level
                        )
                    )
                    measurements_per_dep_graph_level = None
                if do_full_block_in_seq_attention is not None:
                    print(
                        extra_param_err_tmpl.format(
                            "do_full_block_in_seq_attention", do_full_block_in_seq_attention
                        )
                    )
                    do_full_block_in_seq_attention = None
                if do_full_block_in_dep_graph_attention is not None:
                    print(
                        extra_param_err_tmpl.format(
                            "do_full_block_in_dep_graph_attention",
                            do_full_block_in_dep_graph_attention,
                        )
                    )
                    do_full_block_in_dep_graph_attention = None
                if dep_graph_attention_types is not None:
                    print(extra_param_err_tmpl.format("dep_graph_attention_types", dep_graph_attention_types))
                    dep_graph_attention_types = None
                if dep_graph_window_size is not None:
                    print(extra_param_err_tmpl.format("dep_graph_window_size", dep_graph_window_size))
                    dep_graph_window_size = None

            case _:
                raise ValueError(
                    "`structured_event_processing_mode` must be a valid `StructuredEventProcessingMode` "
                    f"enum member ({StructuredEventProcessingMode.values()}). Got "
                    f"{structured_event_processing_mode}."
                )

        self.structured_event_processing_mode = structured_event_processing_mode

        if (head_dim is None) and (hidden_size is None):
            raise ValueError("Must specify at least one of hidden size or head dim!")

        if hidden_size is None:
            hidden_size = head_dim * num_attention_heads
        elif head_dim is None:
            head_dim = hidden_size // num_attention_heads

        if head_dim * num_attention_heads != hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {hidden_size} "
                f"and `num_attention_heads`: {num_attention_heads})."
            )

        if type(num_hidden_layers) is not int:
            raise TypeError(f"num_hidden_layers must be an int! Got {type(num_hidden_layers)}.")
        elif num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be > 0! Got {num_hidden_layers}.")
        self.num_hidden_layers = num_hidden_layers

        if seq_attention_types is None:
            seq_attention_types = ["local", "global"]

        self.seq_attention_types = seq_attention_types
        self.seq_attention_layers = self.expand_attention_types_params(seq_attention_types)

        if len(self.seq_attention_layers) != num_hidden_layers:
            raise ValueError(
                "Configuration for module is incorrect. "
                "It is required that `len(config.seq_attention_layers)` == `config.num_hidden_layers` "
                f"but is `len(config.seq_attention_layers) = {len(self.seq_attention_layers)}`, "
                f"`config.num_layers = {num_hidden_layers}`. "
                "`config.seq_attention_layers` is prepared using `config.seq_attention_types`. "
                "Please verify the value of `config.seq_attention_types` argument."
            )

        if structured_event_processing_mode != "conditionally_independent":
            if dep_graph_attention_types is None:
                dep_graph_attention_types = "global"

            dep_graph_attention_layers = self.expand_attention_types_params(dep_graph_attention_types)

            if len(dep_graph_attention_layers) != num_hidden_layers:
                raise ValueError(
                    "Configuration for module is incorrect. It is required that "
                    "`len(config.dep_graph_attention_layers)` == `config.num_hidden_layers` "
                    f"but is `len(config.dep_graph_attention_layers) = {len(dep_graph_attention_layers)}`, "
                    f"`config.num_layers = {num_hidden_layers}`. "
                    "`config.dep_graph_attention_layers` is prepared using "
                    "`config.dep_graph_attention_types`. Please verify the value of "
                    "`config.dep_graph_attention_types` argument."
                )
        else:
            dep_graph_attention_layers = None

        self.dep_graph_attention_types = dep_graph_attention_types
        self.dep_graph_attention_layers = dep_graph_attention_layers

        self.seq_window_size = seq_window_size
        self.dep_graph_window_size = dep_graph_window_size

        missing_param_err_tmpl = f"For a {TTE_generation_layer_type} model, {{}} should not be None"
        extra_param_err_tmpl = (
            f"WARNING: For a {TTE_generation_layer_type} model, {{}} is not used; got {{}}. "
            "Setting to None."
        )
        match TTE_generation_layer_type:
            case TimeToEventGenerationHeadType.LOG_NORMAL_MIXTURE:
                if TTE_lognormal_generation_num_components is None:
                    raise ValueError(missing_param_err_tmpl.format("TTE_lognormal_generation_num_components"))
                if type(TTE_lognormal_generation_num_components) is not int:
                    raise TypeError(
                        f"`TTE_lognormal_generation_num_components` must be an int! "
                        f"Got: {type(TTE_lognormal_generation_num_components)}."
                    )
                elif TTE_lognormal_generation_num_components <= 0:
                    raise ValueError(
                        "`TTE_lognormal_generation_num_components` should be >0 "
                        f"got {TTE_lognormal_generation_num_components}."
                    )
                if mean_log_inter_event_time_min is None:
                    mean_log_inter_event_time_min = 0.0
                if std_log_inter_event_time_min is None:
                    std_log_inter_event_time_min = 1.0

            case TimeToEventGenerationHeadType.EXPONENTIAL:
                if TTE_lognormal_generation_num_components is not None:
                    print(
                        extra_param_err_tmpl.format(
                            "TTE_lognormal_generation_num_components",
                            TTE_lognormal_generation_num_components,
                        )
                    )
                    TTE_lognormal_generation_num_components = None
                if mean_log_inter_event_time_min is not None:
                    print(
                        extra_param_err_tmpl.format(
                            "mean_log_inter_event_time_min", mean_log_inter_event_time_min
                        )
                    )
                    mean_log_inter_event_time_min = None
                if std_log_inter_event_time_min is not None:
                    print(
                        extra_param_err_tmpl.format(
                            "std_log_inter_event_time_min", std_log_inter_event_time_min
                        )
                    )
                    std_log_inter_event_time_min = None

            case _:
                raise ValueError(
                    f"Invalid option for `TTE_generation_layer_type`. Must be in "
                    f"({TimeToEventGenerationHeadType.values()}). Got {TTE_generation_layer_type}."
                )

        self.TTE_generation_layer_type = TTE_generation_layer_type
        self.TTE_lognormal_generation_num_components = TTE_lognormal_generation_num_components
        self.mean_log_inter_event_time_min = mean_log_inter_event_time_min
        self.std_log_inter_event_time_min = std_log_inter_event_time_min

        self.init_std = init_std

        self.max_seq_len = max_seq_len
        self.vocab_sizes_by_measurement = vocab_sizes_by_measurement
        self.vocab_offsets_by_measurement = vocab_offsets_by_measurement
        self.measurements_idxmap = measurements_idxmap
        self.measurements_per_generative_mode = measurements_per_generative_mode
        self.measurements_per_dep_graph_level = measurements_per_dep_graph_level

        self.vocab_size = max(sum(self.vocab_sizes_by_measurement.values()), 1)

        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.attention_dropout = attention_dropout
        self.input_dropout = input_dropout
        self.resid_dropout = resid_dropout
        self.intermediate_size = intermediate_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.activation_function = activation_function
        self.do_full_block_in_seq_attention = do_full_block_in_seq_attention
        self.do_full_block_in_dep_graph_attention = do_full_block_in_dep_graph_attention

        self.use_cache = use_cache

        assert not kwargs.get("is_encoder_decoder", False), "Can't be used in encoder/decoder mode!"
        kwargs["is_encoder_decoder"] = False

        super().__init__(**kwargs)

    def measurements_for(self, modality: DataModality) -> list[str]:
        return self.measurements_per_generative_mode.get(modality, [])

    def expand_attention_types_params(
        self, attention_types: ATTENTION_TYPES_LIST_T
    ) -> list[AttentionLayerType]:
        """Expands the attention syntax from the easy-to-enter syntax to one for the model."""
        if isinstance(attention_types, str):
            return [attention_types] * self.num_hidden_layers

        if not isinstance(attention_types, list):
            raise TypeError(f"Config Invalid {attention_types} ({type(attention_types)}) is wrong type!")

        if isinstance(attention_types[0], str):
            return (attention_types * self.num_hidden_layers)[: self.num_hidden_layers]

        if isinstance(attention_types[0], (list, tuple)):
            attentions = []
            for sub_list, n_layers in attention_types:
                attentions.extend(list(sub_list) * n_layers)
            return attentions[: self.num_hidden_layers]

        raise TypeError(f"Config Invalid {attention_types} El 0 ({type(attention_types[0])}) is wrong type!")

    def set_to_dataset(self, dataset: PytorchDataset):
        """Set various configuration parameters to match `dataset`."""
        # TODO(mmd): The overlap of information here is getting large -- should likely be simplified and
        # streamlined.
        self.measurement_configs = dataset.measurement_configs
        self.measurements_idxmap = dataset.vocabulary_config.measurements_idxmap
        self.measurements_per_generative_mode = dataset.vocabulary_config.measurements_per_generative_mode
        for k in DataModality.values():
            if k not in self.measurements_per_generative_mode:
                self.measurements_per_generative_mode[k] = []

        if self.structured_event_processing_mode == StructuredEventProcessingMode.NESTED_ATTENTION:
            in_dep = {
                x[0] if isinstance(x, (list, tuple)) and len(x) == 2 else x
                for x in itertools.chain.from_iterable(self.measurements_per_dep_graph_level)
            }
            in_generative_mode = set(
                itertools.chain.from_iterable(self.measurements_per_generative_mode.values())
            )

            if not in_generative_mode.issubset(in_dep):
                raise ValueError(
                    "Config is attempting to generate something outside the dependency graph:\n"
                    f"{in_generative_mode - in_dep}"
                )

        self.event_types_idxmap = dataset.vocabulary_config.event_types_idxmap

        self.vocab_offsets_by_measurement = dataset.vocabulary_config.vocab_offsets_by_measurement
        self.vocab_sizes_by_measurement = dataset.vocabulary_config.vocab_sizes_by_measurement
        for k in set(self.vocab_offsets_by_measurement.keys()) - set(self.vocab_sizes_by_measurement.keys()):
            self.vocab_sizes_by_measurement[k] = 1

        self.vocab_size = dataset.vocabulary_config.total_vocab_size
        self.max_seq_len = dataset.max_seq_len

        if self.TTE_generation_layer_type == TimeToEventGenerationHeadType.LOG_NORMAL_MIXTURE:
            self.mean_log_inter_event_time_min = dataset.mean_log_inter_event_time_min
            self.std_log_inter_event_time_min = dataset.std_log_inter_event_time_min

        if dataset.has_task:
            if self.finetuning_task is None and len(dataset.tasks) == 1:
                self.finetuning_task = dataset.tasks[0]
            if self.finetuning_task is not None:
                # In the single-task fine-tuning case, we can infer a lot of this from the dataset.
                match dataset.task_types[self.finetuning_task]:
                    case "binary_classification" | "multi_class_classification":
                        self.id2label = {
                            i: v for i, v in enumerate(dataset.task_vocabs[self.finetuning_task])
                        }
                        self.label2id = {v: i for i, v in self.id2label.items()}
                        self.num_labels = len(self.id2label)
                        self.problem_type = "single_label_classification"
                    case "regression":
                        self.num_labels = 1
                        self.problem_type = "regression"
            elif all(t == "binary_classification" for t in dataset.task_types.values()):
                self.problem_type = "multi_label_classification"
                self.id2label = {0: False, 1: True}
                self.label2id = {v: i for i, v in self.id2label.items()}
                self.num_labels = len(dataset.tasks)
            elif all(t == "regression" for t in dataset.task_types.values()):
                self.num_labels = len(dataset.tasks)
                self.problem_type = "regression"

    def __eq__(self, other):
        """Checks equality in a type sensitive manner to avoid pytorch lightning issues."""
        if not isinstance(other, PretrainedConfig):
            return False
        else:
            return PretrainedConfig.__eq__(self, other)

    def to_dict(self) -> dict[str, Any]:
        as_dict = super().to_dict()
        if as_dict.get("measurement_configs", {}):
            new_meas_configs = {}
            for k, v in as_dict["measurement_configs"].items():
                new_meas_configs[k] = v if isinstance(v, dict) else v.to_dict()
            as_dict["measurement_configs"] = new_meas_configs
        return as_dict

    @classmethod
    def from_dict(cls, *args, **kwargs) -> "StructuredTransformerConfig":
        raw_from_dict = super().from_dict(*args, **kwargs)
        if raw_from_dict.measurmeent_configs:
            new_meas_configs = {}
            for k, v in raw_from_dict.measurement_configs.items():
                new_meas_configs[k] = MeasurementConfig.from_dict(v)
            raw_from_dict.measurement_configs = new_meas_configs

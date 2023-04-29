import dataclasses, torch, torchmetrics, wandb, lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from pathlib import Path
from torchmetrics.classification import (
    MulticlassAUROC,
    MultilabelAUROC,
    MulticlassAccuracy,
    MultilabelAccuracy,
    MulticlassAveragePrecision,
    MultilabelAveragePrecision,
)
from typing import Any, Dict, Optional, Sequence, Union
from transformers import get_polynomial_decay_schedule_with_warmup

from .config import StructuredEventStreamTransformerConfig, EventStreamOptimizationConfig
from .model import StructuredEventStreamTransformerForGenerativeSequenceModeling
from .model_output import EventStreamTransformerForGenerativeSequenceModelOutput
from .utils import expand_indexed_regression
from ..EventStreamData.types import DataModality, EventStreamPytorchBatch
from ..EventStreamData.config import EventStreamPytorchDatasetConfig
from ..EventStreamData.event_stream_pytorch_dataset import EventStreamPytorchDataset

def str_summary(T: torch.Tensor):
    return f"shape: {tuple(T.shape)}, type: {T.dtype}, range: {T.min():n}-{T.max():n}"

class StructuredEventStreamForGenerativeSequenceModelingLightningModule(L.LightningModule):
    """A PyTorch Lightning Module for a `StructuredEventStreamTransformerForGenerativeSequenceModeling`."""
    def __init__(
        self,
        config: Union[StructuredEventStreamTransformerConfig, Dict[str, Any]],
        optimization_config: Union[EventStreamOptimizationConfig, Dict[str, Any]],
        pretrained_weights_fp: Optional[Path] = None,
        do_debug_mode: bool = True,
        do_skip_all_metrics: bool = False,
    ):
        """
        Initializes the Lightning Module.

        Args:
            config (`Union[StructuredEventstreamTransformerConfig, Dict[str, Any]]`):
                The configuration for the underlying
                `StructuredEventStreamTransformerForGenerativeSequenceModeling` model. Should be
                in the dedicated `StructuredEventStreamTransformerConfig` class or be a dictionary
                parseable as such.
            optimization_config (`Union[EventStreamOptimizationConfig, Dict[str, Any]]`):
                The configuration for the optimization process handled by the Lightning module. Should
                be in the dedicated `EventStreamOptimizationConfig` class or be a dictionary parseable
                as such.
        """
        super().__init__()

        # If the configurations are dictionaries, convert them to class objects. They may be passed as
        # dictionaries when the lightning module is loaded from a checkpoint, so we need to support
        # this functionality.
        if type(config) is dict: config = StructuredEventStreamTransformerConfig(**config)
        if type(optimization_config) is dict:
            optimization_config = EventStreamOptimizationConfig(**optimization_config)

        self.config = config
        self.optimization_config = optimization_config
        self.do_debug_mode = do_debug_mode
        self.do_skip_all_metrics = do_skip_all_metrics

        self.save_hyperparameters({
            'config': config.to_dict(),
            'optimization_config': dataclasses.asdict(optimization_config),
        })
        if not self.do_skip_all_metrics: self.build_metrics()

        if pretrained_weights_fp is None:
            self.model = StructuredEventStreamTransformerForGenerativeSequenceModeling(config)
        else:
            self.model = StructuredEventStreamTransformerForGenerativeSequenceModeling.from_pretrained(
                pretrained_weights_fp, config=config
            )

    def save_pretrained(self, model_dir: Path):
        fp = model_dir / 'pretrained_weights'
        self.model.save_pretrained(fp)

    def build_metrics(self):
        """Build the various torchmetrics we'll use to track performance."""

        # For judging our ability to predict time-to-event, we'll use the following scores:
        #   1. Explained Variance:
        #      (e.g., https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html)
        #   2. Mean Squared Error
        #   3. Mean Squared Log Error
        self.tte_metrics = torch.nn.ModuleDict({
            'explained_var': torchmetrics.ExplainedVariance(),
            'MSE': torchmetrics.MeanSquaredError(),
            'MSLE': torchmetrics.MeanSquaredLogError(),
        })

        self.metrics = torch.nn.ModuleDict()
        for task_type, measurements in self.config.measurements_per_generative_mode.items():
            for measurement in measurements:
                vocab_size = self.config.vocab_sizes_by_measurement[measurement]

                if measurement not in self.metrics: self.metrics[measurement] = torch.nn.ModuleDict()
                if task_type not in self.metrics[measurement]:
                    self.metrics[measurement][task_type] = torch.nn.ModuleDict()

                if task_type == DataModality.SINGLE_LABEL_CLASSIFICATION:
                    metric_kwargs = {'num_classes': vocab_size, 'ignore_index': 0}
                    if not self.do_debug_mode: metric_kwargs['validate_args'] = False

                    # For judging classification, we'll use macro & weighted accuracy, AUROC, and AUPRC
                    self.metrics[measurement][task_type].update({
                        'macro_AUROC': MulticlassAUROC(**metric_kwargs, average='macro'),
                        'weighted_AUROC': MulticlassAUROC(**metric_kwargs, average='weighted'),
                        'macro_accuracy': MulticlassAccuracy(**metric_kwargs, average='macro'),
                        'weighted_accuracy': MulticlassAccuracy(**metric_kwargs, average='weighted'),
                        'micro_accuracy': MulticlassAccuracy(**metric_kwargs, average='micro'),
                        'macro_AUPRC': MulticlassAveragePrecision(**metric_kwargs, average='macro'),
                        'weighted_AUPRC': MulticlassAveragePrecision(**metric_kwargs, average='weighted'),
                    })

                elif task_type == DataModality.MULTI_LABEL_CLASSIFICATION:
                    metric_kwargs = {'num_labels': vocab_size}
                    if not self.do_debug_mode: metric_kwargs['validate_args'] = False

                    # For judging classification, we'll use macro & weighted accuracy, AUROC, and AUPRC
                    self.metrics[measurement][task_type].update({
                        'macro_AUROC': MultilabelAUROC(**metric_kwargs, average='macro'),
                        'weighted_AUROC': MultilabelAUROC(**metric_kwargs, average='weighted'),
                        'micro_AUROC': MultilabelAUROC(**metric_kwargs, average='micro'),
                        'macro_accuracy': MultilabelAccuracy(**metric_kwargs, average='macro'),
                        'weighted_accuracy': MultilabelAccuracy(**metric_kwargs, average='weighted'),
                        'micro_accuracy': MultilabelAccuracy(**metric_kwargs, average='micro'),
                        'macro_AUPRC': MultilabelAveragePrecision(**metric_kwargs, average='macro'),
                        'weighted_AUPRC': MultilabelAveragePrecision(**metric_kwargs, average='weighted'),
                        'micro_AUPRC': MultilabelAveragePrecision(**metric_kwargs, average='micro'),
                    })
                elif task_type in (
                    DataModality.MULTIVARIATE_REGRESSION, DataModality.UNIVARIATE_REGRESSION
                ):
                    # As we have multiple regression tasks here (unlike TTE), we have to use both macro and
                    # weighted explained variance. We also use MSE. For some reason (TODO(mmd)) MSLE was
                    # erroring out so is not tracked.
                    self.metrics[measurement][task_type].update({
                        'macro_explained_var': torchmetrics.ExplainedVariance(multioutput='uniform_average'),
                        'weighted_explained_var': torchmetrics.ExplainedVariance(
                            multioutput='variance_weighted'
                        ),
                        'MSE': torchmetrics.MeanSquaredError(),
                    })

    def _log_metric_dict(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        metrics: Dict[str, torchmetrics.Metric],
        skip_metrics: Sequence[str],
        prefix: str,
        measurement: str,
    ):
        """
        This helper function logs the set of named metrics for the predictions `preds` and labels `labels`.

        Args:
            `preds` (`torch.Tensor`): The predictions for this metric calculation.
            `labels` (`torch.Tensor`): The labels for this metric calculation.
            `metrics` (`Dict[str, torchmetrics.Metric]`): The metrics to log, by name.
            `skip_metrics` (`Sequence[str]`):
                A list of metrics to skip. Entries are not full metric names, but rather are partial names and
                any metric whose name contains an element of `skip_metrics` will be skipped.
                For example, if `skip_metrics = ['AUROC', 'AUPRC']`, then a metric with name `'macro_AUROC'`
                or `'micro_AUPRC'` would be skipped, whereas a metric named `'weighted_accuracy'` would not.
            `prefix` (`str`):
                The prefix that should be used when logging metric results. Will likely be 'train', 'tuning',
                or 'held_out', for example.
            `measurement` (`str`): The measurement of this metric calculation. Affects the log name.
        """
        for metric_name, metric in metrics.items():
            # We'll want to skip a metric if any element of our skip_metrics list is a substring of the metric
            # name:
            if any(to_skip in metric_name for to_skip in skip_metrics): continue

            try:
                metric(preds, labels)
                self.log(
                    f"{prefix}_{measurement}_{metric_name}", metric,
                    batch_size=self.optimization_config.batch_size
                )
            except (ValueError, IndexError) as e:
                print(
                    f"Failed to compute {metric_name} for {measurement} "
                    f"with preds ({str_summary(preds)}) and labels ({str_summary(labels)}): {e}."
                )

    def log_metrics(
        self,
        results: EventStreamTransformerForGenerativeSequenceModelOutput,
        skip_metrics: Sequence[str],
        prefix: str
    ):
        """
        Logs metric results for a given output result.

        Args:
            `results` (`EventStreamtransformerForGenerativeSequenceModelOutput`):
                The results to assess across the suite of metrics.
            `skip_metrics` (`Sequence[str]`):
                A list of metrics to skip. Entries are not full metric names, but rather are partial names and
                any metric whose name contains an element of `skip_metrics` will be skipped.
                For example, if `skip_metrics = ['AUROC', 'AUPRC']`, then a metric with name `'macro_AUROC'`
                or `'micro_AUPRC'` would be skipped, whereas a metric named `'weighted_accuracy'` would not.
            `prefix` (`str`):
                The prefix that should be used when logging metric results. Will likely be 'train', 'tuning',
                or 'held_out', for example.
        """
        # We start by logging the losses.
        self.log_dict(
            {f"{prefix}_{k}_cls_NLL": v for k, v in results['losses']['classification'].items()},
            batch_size=self.optimization_config.batch_size,
        )
        self.log_dict(
            {f"{prefix}_{k}_reg_NLL": v for k, v in results['losses']['regression'].items()},
            batch_size=self.optimization_config.batch_size,
        )
        self.log(
            f"{prefix}_TTE_reg_NLL", results['losses']['time_to_event'],
            batch_size=self.optimization_config.batch_size,
        )

        self.log(
            f"{prefix}_loss", results['loss'],
            batch_size=self.optimization_config.batch_size,
        )

        if self.do_skip_all_metrics: return
        # We'll commonly log metrics via the `self._log_metric_dict` helper, with some shared keyword
        # arguments.
        log_metric_kwargs = {'skip_metrics': skip_metrics, 'prefix': prefix}

        # Time-to-event
        # The output of the model for time-to-event (and for regression targets as well) are pytorch
        # distribution objects, not scalars. So, for some evaluation metrics, we need to sample values from
        # those distributions to assess the metric.
        # TODO(mmd): We should likely be able to control how many samples are used, to minimize variance of
        # these results.
        tte_dist = results['preds']['time_to_event']
        tte_preds = tte_dist.sample()

        # After sampling, we also need to slice this down to just those intra-event-times that are actually
        # observed. This means we should drop the last sequence element (slice to `[:, :-1]` (as our observed
        # intra-event-times will only exist for the interior of our sequence), then further filter down to
        # just elements of the prediction for which the next sequence element was not masked
        # (mask via `results['event_mask'][:, 1:]`). We also need to filter the observed labels down to also
        # only be present for sequence elements where the next sequence element was truly observed.
        tte_preds = tte_preds[:, :-1][results['event_mask'][:, 1:]]
        tte_labels = results['labels']['time_to_event'][results['event_mask'][:, 1:]]

        # Finally, we can log all relevant TTE metrics given these predictions and labels.
        self._log_metric_dict(
            preds=tte_preds, labels=tte_labels, metrics=self.tte_metrics, measurement='TTE',
            **log_metric_kwargs
        )

        # Per data type
        for measurement, metrics_dict in self.metrics.items():
            if (
                (results['event_type_mask_per_measurement'] is None) or
                (measurement not in results['event_type_mask_per_measurement'])
            ):
                mask = results['event_mask']
            else:
                mask = results['event_mask'] & results['event_type_mask_per_measurement'][measurement]

            if not mask.any(): continue

            for task_type, metrics in metrics_dict.items():
                if task_type in {
                    DataModality.SINGLE_LABEL_CLASSIFICATION, DataModality.MULTI_LABEL_CLASSIFICATION
                }:
                    preds = results['preds']['classification'][measurement].logits
                    labels = results['labels']['classification'][measurement]

                    # We need to filter these down to just those corresponding to observed events. Note that
                    # unlike TTE, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.

                    preds = preds[mask]
                    labels = labels[mask].long()

                    self._log_metric_dict(
                        preds=preds, labels=labels, metrics=metrics, measurement=measurement,
                        **log_metric_kwargs
                    )

                elif task_type == DataModality.MULTIVARIATE_REGRESSION:
                    vocab_size = self.config.vocab_sizes_by_measurement[measurement]

                    # Here, like for TTE, we need to sample from the returned distribution before we can use
                    # it directly. Here we also need to limit to just those events that are actually observed.
                    # Like above, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.
                    dist = results['preds']['regression'][measurement]
                    preds = dist.sample()[mask]
                    labels = results['labels']['regression'][measurement][mask]

                    # However, as our regression output is actually indexed only to the group keys that are
                    # actually measured (tracked in `results['preds']['regression_indices']`, we need to
                    # expand our predictions and labels to be in the full vocabulary space for the metrics to
                    # work naturally.
                    preds_indices = results['preds']['regression_indices'][measurement][mask]
                    labels_indices = results['labels']['regression_indices'][measurement][mask]

                    # We also need to reflect just those data elements for which values were observed:
                    data_el_mask = results['dynamic_values_mask'][mask]

                    preds = preds[data_el_mask]
                    labels = labels[data_el_mask]
                    preds_indices = preds_indices[data_el_mask]
                    labels_indices = labels_indices[data_el_mask]

                    preds_expanded = expand_indexed_regression(preds, preds_indices, vocab_size)
                    labels_expanded = expand_indexed_regression(labels, labels_indices, vocab_size)

                    self._log_metric_dict(
                        preds=preds_expanded, labels=labels_expanded, metrics=metrics,
                        measurement=measurement, **log_metric_kwargs
                    )
                if task_type == DataModality.UNIVARIATE_REGRESSION:
                    # Here, like for TTE, we need to sample from the returned distribution before we can use
                    # it directly. Here we also need to limit to just those events that are actually observed.
                    # Like above, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.
                    dist = results['preds']['regression'][measurement]
                    preds = dist.sample()[mask]
                    labels = results['labels']['regression'][measurement][mask]

                    self._log_metric_dict(
                        preds=preds, labels=labels, metrics=metrics, measurement=measurement,
                        **log_metric_kwargs
                    )

    def training_step(self, batch: EventStreamPytorchBatch, batch_idx: int) -> torch.Tensor:
        """Training step. Skips logging all AUROC, AUPRC, and per_class metric to save compute."""
        out = self.model(batch)
        self.log_metrics(out, skip_metrics=('AUROC', 'AUPRC', 'per_class'), prefix='train')

        return out['loss']

    def validation_step(self, batch: EventStreamPytorchBatch, batch_idx: int):
        """Validation step. Differs from training only in that it does not skip metrics."""
        out = self.model(batch)
        self.log_metrics(out, skip_metrics=[], prefix='tuning')

    def test_step(self, batch: EventStreamPytorchBatch, batch_idx: int):
        """Validation step. Differs from training only in that it does not skip metrics."""
        out = self.model(batch)
        self.log_metrics(out, skip_metrics=[], prefix='held_out')

    def configure_optimizers(self):
        """
        Configures optimizer and learning rate scheduler. Currently this module uses the AdamW optimizer, with
        configurable weight_decay, with a learning rate warming up from 0 on a per-step manner to the
        configurable `self.optimization_config.init_lr`, then undergoes polynomial decay as specified via
        `self.optimization_config`.
        """
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.optimization_config.init_lr,
            weight_decay = self.optimization_config.weight_decay,
        )
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer = opt,
            num_warmup_steps = self.optimization_config.lr_num_warmup_steps,
            num_training_steps = self.optimization_config.max_training_steps,
            power = self.optimization_config.lr_decay_power,
            lr_end = self.optimization_config.end_lr,
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }

def fit_generative_sequence_model(
    save_dir: Path,
    dataset_dir: Path,
    config: StructuredEventStreamTransformerConfig,
    optimization_config: EventStreamOptimizationConfig,
    data_config: EventStreamPytorchDatasetConfig,
    wandb_name: Optional[str] = 'generative_event_stream_transformer',
    wandb_project: Optional[str] = None,
    wandb_team: Optional[str] = None,
    num_dataloader_workers: int = 1,
    do_detect_anomaly: bool = False,
    log_every_n_steps: int = 50,
    return_early: bool = False,
    train_split: str = 'train',
    tuning_split: str = 'tuning',
    held_out_split: str = 'held_out',
    do_overwrite: bool = False,
    do_skip_all_metrics_in_train: bool = False,
    do_final_validation_on_metrics: bool = True,
    do_save_pretrained: bool = True,
):
    """
    Runs the end to end training procedure for the StructuredEventStreamForGenerativeSequenceModelingLightningModule model.

    Args:
        `save_dir` (`pathlib.Path`):
            In what directory this model and its configurationfiles should be saved. If the directory does not
            exist, it will be created. If it does exist already, some files within it may be overwritten.
        `dataset_dir` (`pathlib.Path`):
            The path within which the base dataset object is saved at filename
            `processed_event_stream_dataset.pkl` on which this model run will train. This dataset should
            already be fully processed, including being split into named `'train'`, `'tuning'`, and
            `'held_out'` splits and having metadata vocabularies fully specified.
        `config` (`StructuredEventStreamTransformerConfig`):
            The model configuration for this run. Will primarily be used to create the underlying encoder
            transformer.
        `optimization_config` (`EventStreamOptimizationConfig`):
            The optimization configuration for this run. Will primarily be used to set lightning module
            parameters for optimization.
        `data_config` (`EventStreamPytorchDatasetConfig`):
            The pytorch dataset configuration for this run. Will primarily be used to configure how the
            pytorch dataset parses the data in `dataset_dir`.
        `wandb_name` (`str`, *optional*, defaults to `'generative_mimic_model'`):
            What name to use for tracking this model in Weights & Biases.
        `wandb_project` (`str`, *optional*, defaults to `'medFMs'`):
            What project to store this run under in Weights & Biases.
        `num_dataloader_workers` (`int`, *optional*, defaults to 1):
            How many dataloader workers to use.
        `do_detect_anomaly` (`bool`, *optional*, defaults to False):
            Whether to use the detect anomaly feature in pytorch lightning for this run. Makes the run take
            longer, but will provide detailed traceback information in the event of a gradient NaN issue.
        `log_every_n_steps` (`int`, *optional*, defaults to 50):
            How frequently should this run report log data.

    This function performs the following steps:
        1. Builds train, tuning, and held out pytorch datasets.
        2. Sets the configuration objects to match those built datasets.
        3. Saves the configuration files to the `save_dir`.
        4. builds the pytorch lightning module and Weights & Biases logger.
        5. Trains the lightning module over the pytorch datasets via Lightning.
        6. Returns the updated configuration file and the fit lightning module.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Creating or loading training/tuning datasets
    train_pyd = EventStreamPytorchDataset(data_config, split='train')
    tuning_pyd = EventStreamPytorchDataset(data_config, split='tuning')
    held_out_pyd = EventStreamPytorchDataset(data_config, split='held_out')

    # Setting up configurations
    config.set_to_dataset(train_pyd)

    optimization_config.set_to_dataset(train_pyd)

    # We don't have 'do_overwrite' support in this class.
    config_fp = save_dir / 'config.json'
    if config_fp.exists() and not do_overwrite: raise FileExistsError(f"{config_fp} already exists!")
    else: config.to_json_file(save_dir / "config.json")

    data_config.to_json_file(save_dir / "data_config.json", do_overwrite=do_overwrite)
    optimization_config.to_json_file(save_dir / "optimization_config.json", do_overwrite=do_overwrite)

    # Model
    LM = StructuredEventStreamForGenerativeSequenceModelingLightningModule(
        config, optimization_config, do_skip_all_metrics=do_skip_all_metrics_in_train
    )

    do_use_wandb = wandb_name is not None
    if do_use_wandb:
        wandb_logger_savedir = save_dir  # Wandb automatically adds a "wandb" suffix.
        wandb_logger_savedir.mkdir(parents=True, exist_ok=True)
        wandb_logger = WandbLogger(
            name=wandb_name, project=wandb_project, entity=wandb_team, save_dir=wandb_logger_savedir,
            log_model=True
        )
        # Watching the model naturally tracks parameter values and gradients.
        wandb_logger.watch(LM, log='all', log_graph=True)
    else: wandb_logger = None

    # Setting up torch dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_pyd,
        batch_size = optimization_config.batch_size,
        num_workers = num_dataloader_workers,
        collate_fn = train_pyd.collate,
        shuffle = True,
    )
    tuning_dataloader = torch.utils.data.DataLoader(
        tuning_pyd,
        batch_size = optimization_config.batch_size // 2,
        num_workers = num_dataloader_workers,
        collate_fn = tuning_pyd.collate,
        shuffle = False,
    )
    held_out_dataloader = torch.utils.data.DataLoader(
        held_out_pyd,
        batch_size = optimization_config.batch_size // 2,
        num_workers = num_dataloader_workers,
        collate_fn = held_out_pyd.collate,
        shuffle = False,
    )

    # Setting up model configurations
    # This will track the learning rate value as it updates through warmup and decay.
    callbacks = [LearningRateMonitor(logging_interval='step')]
    if optimization_config.patience is not None:
        callbacks.append(EarlyStopping(
            monitor='tuning_loss', mode='min', patience=optimization_config.patience
        ))

    checkpoints_dir = save_dir / "model_checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    trainer_kwargs = dict(
        max_epochs = optimization_config.max_epochs,
        detect_anomaly = do_detect_anomaly,
        log_every_n_steps = log_every_n_steps,
        callbacks = callbacks,
        default_root_dir = checkpoints_dir,
    )

    if do_use_wandb:
        trainer_kwargs['logger'] = wandb_logger

    if (
        (optimization_config.gradient_accumulation is not None) and
        (optimization_config.gradient_accumulation > 1)
    ):
        trainer_kwargs['accumulate_grad_batches'] = optimization_config.gradient_accumulation

    if torch.cuda.is_available():
        trainer_kwargs.update({'accelerator': "gpu", 'devices': -1})

    if return_early:
        return (
            (train_pyd, tuning_pyd), (config, optimization_config, data_config),
            (train_dataloader, tuning_dataloader), (trainer_kwargs, L.Trainer(**trainer_kwargs)), LM
        )

    # Fitting model
    try:
        n_attempts = 0
        while n_attempts < 5:
            n_attempts += 1
            try:
                trainer = L.Trainer(**trainer_kwargs)
                trainer.fit(
                    model=LM,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=tuning_dataloader
                )

                if do_save_pretrained:
                    pretraining_save_dir = save_dir / 'pretrained_weights'
                    pretraining_save_dir.mkdir(exist_ok=True, parents=False)
                    LM.save_pretrained(save_dir)

                break
            except RuntimeError as e:
                if n_attempts >= 5: raise

                print(
                    f"Caught error {e} during training on attempt {n_attempts}. Retrying with gradient "
                    "accumulation..."
                )
                trainer_kwargs['accumulate_grad_batches'] = (
                    trainer_kwargs.get('accumulate_grad_batches', 1) * 2
                )
                optimization_config.gradient_accumulation = trainer_kwargs['accumulate_grad_batches']
                optimization_config.batch_size = optimization_config.batch_size // 2
                optimization_config.to_json_file(
                    save_dir / "optimization_config.json", do_overwrite=True
                )

                train_dataloader = torch.utils.data.DataLoader(
                    train_pyd,
                    batch_size = optimization_config.batch_size,
                    num_workers = num_dataloader_workers,
                    collate_fn = train_pyd.collate,
                    shuffle = True,
                )
                tuning_dataloader = torch.utils.data.DataLoader(
                    tuning_pyd,
                    batch_size = optimization_config.batch_size // 2,
                    num_workers = num_dataloader_workers,
                    collate_fn = tuning_pyd.collate,
                    shuffle = False,
                )

        if do_final_validation_on_metrics:
            if do_skip_all_metrics_in_train:
                LM.do_skip_all_metrics = False
                LM.build_metrics()

            trainer.validate(model=LM, dataloaders=tuning_dataloader)
            trainer.test(model=LM, dataloaders=held_out_dataloader)
    finally:
        # Even in the case of an error, we want to ensure that wandb exits successfully, otherwise it can lock
        # up Jupyter notebooks for some reason.
        if do_use_wandb:
            wandb_logger.experiment.unwatch(LM)
            wandb.finish()

    return config, LM

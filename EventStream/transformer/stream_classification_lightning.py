import dataclasses, torch, torchmetrics, wandb, pandas as pd, lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from pathlib import Path
from torchmetrics.classification import (
    BinaryAUROC,
    MulticlassAUROC,
    MultilabelAUROC,
    BinaryAccuracy,
    MulticlassAccuracy,
    MultilabelAccuracy,
    BinaryAveragePrecision,
    MulticlassAveragePrecision,
    MultilabelAveragePrecision,
)
from typing import Any, Dict, Optional, Sequence, Union
from transformers import get_polynomial_decay_schedule_with_warmup

from .config import StructuredTransformerConfig, OptimizationConfig
from .model import ESTForStreamClassification
from .model_output import StreamClassificationModelOutput
from ..data.dataset_polars import Dataset
from ..data.config import PytorchDatasetConfig
from ..data.pytorch_dataset import PytorchDataset

def str_summary(T: torch.Tensor):
    return f"shape: {tuple(T.shape)}, type: {T.dtype}, range: {T.min():n}-{T.max():n}"

class ESTForStreamClassificationLM(L.LightningModule):
    """A PyTorch Lightning Module for a `ESTForStreamClassification` model."""
    def __init__(
        self,
        config: Union[StructuredTransformerConfig, Dict[str, Any]],
        optimization_config: Union[OptimizationConfig, Dict[str, Any]],
        pretrained_weights_fp: Optional[Path] = None,
        do_debug_mode: bool = True
    ):
        """
        Initializes the Lightning Module.

        Args:
            config (`Union[StructuredEventstreamTransformerConfig, Dict[str, Any]]`):
                The configuration for the underlying
                `StructuredForStreamClassification` model. Should be
                in the dedicated `StructuredTransformerConfig` class or be a dictionary
                parseable as such.
            optimization_config (`Union[OptimizationConfig, Dict[str, Any]]`):
                The configuration for the optimization process handled by the Lightning module. Should
                be in the dedicated `OptimizationConfig` class or be a dictionary parseable
                as such.
        """
        super().__init__()

        # If the configurations are dictionaries, convert them to class objects. They may be passed as
        # dictionaries when the lightning module is loaded from a checkpoint, so we need to support
        # this functionality.
        if type(config) is dict: config = StructuredTransformerConfig(**config)
        if type(optimization_config) is dict:
            optimization_config = OptimizationConfig(**optimization_config)

        self.config = config
        self.optimization_config = optimization_config
        self.do_debug_mode = do_debug_mode

        self.save_hyperparameters({
            'config': config.to_dict(),
            'optimization_config': dataclasses.asdict(optimization_config),
        })
        self.build_metrics()

        if pretrained_weights_fp is None:
            self.model = StructuredTransformerForStreamClassification(config)
        else:
            self.model = StructuredTransformerForStreamClassification.from_pretrained(
                pretrained_weights_fp, config=config
            )

    def save_pretrained(self, model_dir: Path):
        fp = model_dir / 'pretrained_weights'
        self.model.save_pretrained(fp)

    def build_metrics(self):
        """Build the various torchmetrics we'll use to track performance."""

        if (
            (self.config.problem_type == 'single_label_classification') and
            (self.config.num_labels > 2)
        ):
            metric_kwargs = {'num_classes': self.config.num_labels}
            if not self.do_debug_mode: metric_kwargs['validate_args'] = False

            # For judging classification, we'll use macro & weighted accuracy, AUROC, and AUPRC
            self.metrics = torch.nn.ModuleDict({
                'macro_AUROC': MulticlassAUROC(**metric_kwargs, average='macro'),
                'weighted_AUROC': MulticlassAUROC(**metric_kwargs, average='weighted'),
                'macro_accuracy': MulticlassAccuracy(**metric_kwargs, average='macro'),
                'weighted_accuracy': MulticlassAccuracy(**metric_kwargs, average='weighted'),
                'micro_accuracy': MulticlassAccuracy(**metric_kwargs, average='micro'),
                'macro_AUPRC': MulticlassAveragePrecision(**metric_kwargs, average='macro'),
                'weighted_AUPRC': MulticlassAveragePrecision(**metric_kwargs, average='weighted'),
            })
        elif (
            (self.config.problem_type == 'single_label_classification') and
            (self.config.num_labels == 2)
        ):
            metric_kwargs = {}
            if not self.do_debug_mode: metric_kwargs['validate_args'] = False

            # For judging classification, we'll use macro & weighted accuracy, AUROC, and AUPRC
            self.metrics = torch.nn.ModuleDict({
                'AUROC': BinaryAUROC(**metric_kwargs),
                'accuracy': BinaryAccuracy(**metric_kwargs),
                'AUPRC': BinaryAveragePrecision(**metric_kwargs),
            })
        elif self.config.problem_type == 'multi_label_classification':
            metric_kwargs = {'num_labels': self.config.num_labels}
            if not self.do_debug_mode: metric_kwargs['validate_args'] = False

            # For judging classification, we'll use macro & weighted accuracy, AUROC, and AUPRC
            self.metrics = torch.nn.ModuleDict({
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
        else: raise ValueError(f"{self.config.problem_type} not valid")
        self.metrics_defined = False

    def define_metrics(self):
        if self.metrics_defined: return
        for metric_name, _ in self.metrics.items():
            for prefix in ('train', 'tuning', 'held_out'):
                wandb.define_metric(f'{prefix}_{metric_name}', summary='max')
        self.metrics_defined = True

    def _log_metric_dict(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        metrics: Dict[str, torchmetrics.Metric],
        skip_metrics: Sequence[str],
        prefix: str,
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
        """
        self.define_metrics()
        for metric_name, metric in metrics.items():
            # We'll want to skip a metric if any element of our skip_metrics list is a substring of the metric
            # name:
            if any(to_skip in metric_name for to_skip in skip_metrics): continue

            try:
                metric(preds, labels)
                self.log(f"{prefix}_{metric_name}", metric)
            except (ValueError, IndexError) as e:
                print(
                    f"Failed to compute {metric_name} "
                    f"with preds ({str_summary(preds)}) and labels ({str_summary(labels)}): {e}."
                )

    def log_metrics(
        self,
        results: StreamClassificationModelOutput,
        skip_metrics: Sequence[str],
        prefix: str
    ):
        """
        Logs metric results for a given output result.

        Args:
            `results` (`transformerForGenerativeSequenceModelOutput`):
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

        self._log_metric_dict(
            preds=results.preds, labels=results.labels, metrics=self.metrics, skip_metrics=skip_metrics,
            prefix=prefix
        )

        self.log(f"{prefix}_loss", results.loss)

    def training_step(self, batch, batch_idx):
        """Training step. Skips logging all AUROC, AUPRC, and per_class metric to save compute."""
        out = self.model(batch)
        self.log_metrics(out, skip_metrics=('AUROC', 'AUPRC', 'per_class'), prefix='train')

        return out['loss']

    def validation_step(self, batch, batch_idx):
        """Validation step. Differs from training only in that it does not skip metrics."""
        out = self.model(batch)
        self.log_metrics(out, skip_metrics=[], prefix='tuning')

    def test_step(self, batch, batch_idx):
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

def train(
    save_dir: Path,
    optimization_config: OptimizationConfig,

    task_df: Optional[pd.DataFrame] = None,
    config: Optional[StructuredTransformerConfig] = None,
    data_config: Optional[PytorchDatasetConfig] = None,
    wandb_name: str = 'generative_mimic_model',
    wandb_project: str = 'medFMs',
    num_dataloader_workers: int = 1,
    do_detect_anomaly: bool = False,
    log_every_n_steps: int = 50,
    return_early: bool = False,
    pretrained_weights_fp: Optional[Path] = None,
    do_overwrite: bool = False,
    train_split: str = 'train',
    tuning_split: str = 'tuning',
    held_out_split: str = 'held_out',
    pyds: Optional[Sequence[PytorchDataset]] = None,
    extra_wandb_log_params: Optional[Dict[str, Any]] = None,
):
    """
    Runs the end to end training procedure for the ESTForStreamClassification model.

    Args:
        `save_dir` (`pathlib.Path`):
            In what directory this model and its configurationfiles should be saved. If the directory does not
            exist, it will be created. If it does exist already, some files within it may be overwritten.
        `dataset` (`Dataset`):
            The base dataset object on which this model run will train. This dataset should already be fully
            processed, including being split into named `'train'`, `'tuning'`, and `'held_out'` splits and
            having metadata vocabularies fully specified.
        `config` (`StructuredTransformerConfig`):
            The model configuration for this run. Will primarily be used to create the underlying encoder
            transformer.
        `optimization_config` (`OptimizationConfig`):
            The optimization configuration for this run. Will primarily be used to set lightning module
            parameters for optimization.
        `data_config` (`PytorchDatasetConfig`):
            The pytorch dataset configuration for this run. Will primarily be used to configure how the
            pytorch dataset parses the data in `dataset`.
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
        1. Builds train, tuning, and held out pytorch datasets from the passed `Dataset` and
           `data_config`.
        2. Sets the configuration objects to match those built datasets.
        3. Saves the configuration files to the `save_dir`.
        4. builds the pytorch lightning module and Weights & Biases logger.
        5. Trains the lightning module over the pytorch datasets via Lightning.
        6. Returns the updated configuration file and the fit lightning module.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Creating or loading training/tuning datasets
    train_pyd = PytorchDataset(data_config, task_df=task_df, split='train')
    tuning_pyd = PytorchDataset(data_config, task_df=task_df, split='tuning')
    held_out_pyd = PytorchDataset(data_config, task_df=task_df, split='held_out')

    # Setting up configurations
    assert config is not None
    config.set_to_dataset(train_pyd)

    optimization_config.set_to_dataset(train_pyd)

    config.to_json_file(save_dir / "config.json")
    optimization_config.to_json_file(save_dir / "optimization_config.json", do_overwrite=do_overwrite)

    # Model
    LM = ESTForStreamClassificationLM(
        config, optimization_config, pretrained_weights_fp=pretrained_weights_fp
    )

    wandb_logger_savedir = save_dir  # Wandb automatically adds a "wandb" suffix.
    wandb_logger_savedir.mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(
        name=wandb_name, project=wandb_project, save_dir=wandb_logger_savedir,
        log_model=True
    )
    # Watching the model naturally tracks parameter values and gradients.
    wandb_logger.watch(LM, log='all', log_graph=True)
    if extra_wandb_log_params is not None:
        wandb_logger.experiment.config.update(extra_wandb_log_params)

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
        batch_size = optimization_config.batch_size,
        num_workers = num_dataloader_workers,
        collate_fn = tuning_pyd.collate,
        shuffle = False,
    )
    held_out_dataloader = torch.utils.data.DataLoader(
        held_out_pyd,
        batch_size = optimization_config.batch_size,
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
        logger = wandb_logger,
        log_every_n_steps = log_every_n_steps,
        callbacks = callbacks,
        default_root_dir = checkpoints_dir,
    )

    if torch.cuda.is_available():
        trainer_kwargs.update({'accelerator': "gpu", 'devices': -1})

    trainer = L.Trainer(**trainer_kwargs)

    if return_early:
        return (
            (train_pyd, tuning_pyd, held_out_pyd), (config, optimization_config, data_config),
            (train_dataloader, tuning_dataloader), (trainer_kwargs, trainer), LM
        )

    # Fitting model
    try:
        trainer.fit(model=LM, train_dataloaders=train_dataloader, val_dataloaders=tuning_dataloader)
        trainer.test(model=LM, dataloaders=held_out_dataloader)
    finally:
        # Even in the case of an error, we want to ensure that wandb exits successfully, otherwise it can lock
        # up Jupyter notebooks for some reason.
        wandb_logger.experiment.unwatch(LM)
        wandb.finish()

    return config, LM

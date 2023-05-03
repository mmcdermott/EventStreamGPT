import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import lightning as L
import omegaconf
import pandas as pd
import torch
import torchmetrics
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
)
from transformers import get_polynomial_decay_schedule_with_warmup

from ..data.config import PytorchDatasetConfig
from ..data.dataset_polars import Dataset
from ..data.pytorch_dataset import PytorchDataset
from ..utils import hydra_dataclass, task_wrapper
from .config import MetricsConfig, OptimizationConfig, StructuredTransformerConfig
from .model import ESTForStreamClassification
from .model_output import StreamClassificationModelOutput


def str_summary(T: torch.Tensor):
    return f"shape: {tuple(T.shape)}, type: {T.dtype}, range: {T.min():n}-{T.max():n}"


class ESTForStreamClassificationLM(L.LightningModule):
    """A PyTorch Lightning Module for a `ESTForStreamClassification` model."""

    def __init__(
        self,
        config: Union[StructuredTransformerConfig, Dict[str, Any]],
        optimization_config: Union[OptimizationConfig, Dict[str, Any]],
        pretrained_weights_fp: Optional[Path] = None,
        do_debug_mode: bool = True,
    ):
        """Initializes the Lightning Module.

        Args:
            config (`Union[StructuredTransformerConfig, Dict[str, Any]]`):
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
        if type(config) is dict:
            config = StructuredTransformerConfig(**config)
        if type(optimization_config) is dict:
            optimization_config = OptimizationConfig(**optimization_config)

        self.config = config
        self.optimization_config = optimization_config
        self.do_debug_mode = do_debug_mode

        self.save_hyperparameters(
            {
                "config": config.to_dict(),
                "optimization_config": dataclasses.asdict(optimization_config),
            }
        )
        self.build_metrics()

        if pretrained_weights_fp is None:
            self.model = ESTForStreamClassification(config)
        else:
            self.model = ESTForStreamClassification.from_pretrained(
                pretrained_weights_fp, config=config
            )

    def save_pretrained(self, model_dir: Path):
        fp = model_dir / "pretrained_weights"
        self.model.save_pretrained(fp)

    def build_metrics(self):
        """Build the various torchmetrics we'll use to track performance."""

        if (self.config.problem_type == "single_label_classification") and (
            self.config.num_labels > 2
        ):
            metric_kwargs = {"num_classes": self.config.num_labels}
            if not self.do_debug_mode:
                metric_kwargs["validate_args"] = False

            # For judging classification, we'll use macro & weighted accuracy, AUROC, and AUPRC
            self.metrics = torch.nn.ModuleDict(
                {
                    "macro_AUROC": MulticlassAUROC(**metric_kwargs, average="macro"),
                    "weighted_AUROC": MulticlassAUROC(**metric_kwargs, average="weighted"),
                    "macro_accuracy": MulticlassAccuracy(**metric_kwargs, average="macro"),
                    "weighted_accuracy": MulticlassAccuracy(**metric_kwargs, average="weighted"),
                    "micro_accuracy": MulticlassAccuracy(**metric_kwargs, average="micro"),
                    "macro_AUPRC": MulticlassAveragePrecision(**metric_kwargs, average="macro"),
                    "weighted_AUPRC": MulticlassAveragePrecision(
                        **metric_kwargs, average="weighted"
                    ),
                }
            )
        elif (self.config.problem_type == "single_label_classification") and (
            self.config.num_labels == 2
        ):
            metric_kwargs = {}
            if not self.do_debug_mode:
                metric_kwargs["validate_args"] = False

            # For judging classification, we'll use macro & weighted accuracy, AUROC, and AUPRC
            self.metrics = torch.nn.ModuleDict(
                {
                    "AUROC": BinaryAUROC(**metric_kwargs),
                    "accuracy": BinaryAccuracy(**metric_kwargs),
                    "AUPRC": BinaryAveragePrecision(**metric_kwargs),
                }
            )
        elif self.config.problem_type == "multi_label_classification":
            metric_kwargs = {"num_labels": self.config.num_labels}
            if not self.do_debug_mode:
                metric_kwargs["validate_args"] = False

            # For judging classification, we'll use macro & weighted accuracy, AUROC, and AUPRC
            self.metrics = torch.nn.ModuleDict(
                {
                    "macro_AUROC": MultilabelAUROC(**metric_kwargs, average="macro"),
                    "weighted_AUROC": MultilabelAUROC(**metric_kwargs, average="weighted"),
                    "micro_AUROC": MultilabelAUROC(**metric_kwargs, average="micro"),
                    "macro_accuracy": MultilabelAccuracy(**metric_kwargs, average="macro"),
                    "weighted_accuracy": MultilabelAccuracy(**metric_kwargs, average="weighted"),
                    "micro_accuracy": MultilabelAccuracy(**metric_kwargs, average="micro"),
                    "macro_AUPRC": MultilabelAveragePrecision(**metric_kwargs, average="macro"),
                    "weighted_AUPRC": MultilabelAveragePrecision(
                        **metric_kwargs, average="weighted"
                    ),
                    "micro_AUPRC": MultilabelAveragePrecision(**metric_kwargs, average="micro"),
                }
            )
        else:
            raise ValueError(f"{self.config.problem_type} not valid")
        self.metrics_defined = False

    def define_metrics(self):
        if self.metrics_defined:
            return
        for metric_name, _ in self.metrics.items():
            for prefix in ("train", "tuning", "held_out"):
                wandb.define_metric(f"{prefix}_{metric_name}", summary="max")
        self.metrics_defined = True

    def _log_metric_dict(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        metrics: Dict[str, torchmetrics.Metric],
        skip_metrics: Sequence[str],
        prefix: str,
    ):
        """This helper function logs the set of named metrics for the predictions `preds` and
        labels `labels`.

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
            if any(to_skip in metric_name for to_skip in skip_metrics):
                continue

            try:
                metric(preds, labels)
                self.log(f"{prefix}_{metric_name}", metric)
            except (ValueError, IndexError) as e:
                print(
                    f"Failed to compute {metric_name} "
                    f"with preds ({str_summary(preds)}) and labels ({str_summary(labels)}): {e}."
                )

    def log_metrics(
        self, results: StreamClassificationModelOutput, skip_metrics: Sequence[str], prefix: str
    ):
        """Logs metric results for a given output result.

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
            preds=results.preds,
            labels=results.labels,
            metrics=self.metrics,
            skip_metrics=skip_metrics,
            prefix=prefix,
        )

        self.log(f"{prefix}_loss", results.loss)

    def training_step(self, batch, batch_idx):
        """Training step.

        Skips logging all AUROC, AUPRC, and per_class metric to save compute.
        """
        out = self.model(batch)
        self.log_metrics(out, skip_metrics=("AUROC", "AUPRC", "per_class"), prefix="train")

        return out["loss"]

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Differs from training only in that it does not skip metrics.
        """
        out = self.model(batch)
        self.log_metrics(out, skip_metrics=[], prefix="tuning")

    def test_step(self, batch, batch_idx):
        """Validation step.

        Differs from training only in that it does not skip metrics.
        """
        out = self.model(batch)
        self.log_metrics(out, skip_metrics=[], prefix="held_out")

    def configure_optimizers(self):
        """Configures optimizer and learning rate scheduler.

        Currently this module uses the AdamW optimizer, with configurable weight_decay, with a
        learning rate warming up from 0 on a per-step manner to the configurable
        `self.optimization_config.init_lr`, then undergoes polynomial decay as specified via
        `self.optimization_config`.
        """
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimization_config.init_lr,
            weight_decay=self.optimization_config.weight_decay,
        )
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=opt,
            num_warmup_steps=self.optimization_config.lr_num_warmup_steps,
            num_training_steps=self.optimization_config.max_training_steps,
            power=self.optimization_config.lr_decay_power,
            lr_end=self.optimization_config.end_lr,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


@hydra_dataclass
class FinetuneConfig:
    save_dir: str = omegaconf.MISSING
    pretrained_weights_fp: Optional[Path] = (None,)
    do_overwrite: bool = False

    config: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "_target_": "EventStream.transformer.config.StructuredTransformerConfig",
        }
    )
    optimization_config: OptimizationConfig = OptimizationConfig()
    data_config: PytorchDatasetConfig = PytorchDatasetConfig()
    metrics_config: MetricsConfig = MetricsConfig()

    task_df_name: str = omegaconf.MISSING
    task_df_fp: Optional[
        Union[str, Path]
    ] = "${data_config.save_dir}/task_dfs/${task_df_name}.parquet"

    wandb_name: Optional[str] = "generative_event_stream_transformer"
    wandb_project: Optional[str] = None
    wandb_team: Optional[str] = None
    extra_wandb_log_params: Optional[Dict[str, Any]] = None
    log_every_n_steps: int = 50

    num_dataloader_workers: int = 1

    do_detect_anomaly: bool = False
    do_final_validation_on_metrics: bool = True

    def __post_init__(self):
        if type(self.save_dir) is str and self.save_dir != omegaconf.MISSING:
            self.save_dir = Path(self.save_dir)
        if type(self.task_df_fp) is str and self.task_df_fp != omegaconf.MISSING:
            self.task_df_fp = Path(self.task_df_fp)

        if self.task_df_name is None and self.task_df_fp is not None:
            self.task_df_name = self.task_df_fp.stem


@task_wrapper
def train(cfg: FinetuneConfig, return_early: bool = False):
    """Runs the end to end training procedure for the ESTForGenerativeSequenceModelingLM model.

    Args: TODO
    """
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    task_df = pd.read_parquet(cfg.task_df_fp)

    # Creating or loading training/tuning datasets
    train_pyd = PytorchDataset(cfg.data_config, task_df=task_df, split="train")
    tuning_pyd = PytorchDataset(cfg.data_config, task_df=task_df, split="tuning")
    held_out_pyd = PytorchDataset(cfg.data_config, task_df=task_df, split="held_out")

    config = cfg.config
    optimization_config = cfg.optimization_config
    metrics_config = cfg.metrics_config

    config.set_to_dataset(train_pyd)
    optimization_config.set_to_dataset(train_pyd)

    # We don't have 'do_overwrite' support in this class.
    config_fp = cfg.save_dir / "config.json"
    if config_fp.exists() and not cfg.do_overwrite:
        raise FileExistsError(f"{config_fp} already exists!")
    else:
        config.to_json_file(cfg.save_dir / "config.json")

    cfg.data_config.to_json_file(cfg.save_dir / "data_config.json", do_overwrite=cfg.do_overwrite)
    optimization_config.to_json_file(
        cfg.save_dir / "optimization_config.json", do_overwrite=cfg.do_overwrite
    )
    metrics_config.to_json_file(
        cfg.save_dir / "metrics_config.json", do_overwrite=cfg.do_overwrite
    )

    # Model
    LM = ESTForStreamClassificationLM(
        config=config,
        optimization_config=optimization_config,
        metrics_config=metrics_config,
        pretrained_weights_fp=cfg.pretrained_weights_fp,
    )

    # Setting up torch dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_pyd,
        batch_size=optimization_config.batch_size,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=train_pyd.collate,
        shuffle=True,
    )
    tuning_dataloader = torch.utils.data.DataLoader(
        tuning_pyd,
        batch_size=optimization_config.batch_size // 2,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=tuning_pyd.collate,
        shuffle=False,
    )
    held_out_dataloader = torch.utils.data.DataLoader(
        held_out_pyd,
        batch_size=optimization_config.batch_size // 2,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=held_out_pyd.collate,
        shuffle=False,
    )

    # Setting up model configurations
    # This will track the learning rate value as it updates through warmup and decay.
    callbacks = [LearningRateMonitor(logging_interval="step")]
    if optimization_config.patience is not None:
        callbacks.append(
            EarlyStopping(monitor="tuning_loss", mode="min", patience=optimization_config.patience)
        )

    checkpoints_dir = cfg.save_dir / "model_checkpoints"
    checkpoints_dir.mkdir(parents=False, exist_ok=True)

    trainer_kwargs = dict(
        max_epochs=optimization_config.max_epochs,
        detect_anomaly=cfg.do_detect_anomaly,
        log_every_n_steps=cfg.log_every_n_steps,
        callbacks=callbacks,
        default_root_dir=checkpoints_dir,
    )

    do_use_wandb = cfg.wandb_name is not None
    if do_use_wandb:
        wandb_logger_savedir = cfg.save_dir  # Wandb automatically adds a "wandb" suffix.
        wandb_logger = WandbLogger(
            name=cfg.wandb_name,
            project=cfg.wandb_project,
            entity=cfg.wandb_team,
            save_dir=wandb_logger_savedir,
            log_model=True,
        )
        # Watching the model naturally tracks parameter values and gradients.
        wandb_logger.watch(LM, log="all", log_graph=True)

        trainer_kwargs["logger"] = wandb_logger

        if cfg.extra_wandb_log_params is not None:
            wandb_logger.experiment.config.update(cfg.extra_wandb_log_params)

    if (optimization_config.gradient_accumulation is not None) and (
        optimization_config.gradient_accumulation > 1
    ):
        trainer_kwargs["accumulate_grad_batches"] = optimization_config.gradient_accumulation

    if torch.cuda.is_available():
        trainer_kwargs.update({"accelerator": "gpu", "devices": -1})

    if return_early:
        return (
            (train_pyd, tuning_pyd),
            (config, optimization_config, cfg.data_config),
            (train_dataloader, tuning_dataloader),
            (trainer_kwargs, L.Trainer(**trainer_kwargs)),
            LM,
        )

    # Fitting model
    n_attempts = 0
    while n_attempts < 5:
        n_attempts += 1
        try:
            trainer = L.Trainer(**trainer_kwargs)
            trainer.fit(
                model=LM, train_dataloaders=train_dataloader, val_dataloaders=tuning_dataloader
            )
            break
        except RuntimeError as e:
            if n_attempts >= 5:
                raise

            print(
                f"Caught error {e} during training on attempt {n_attempts}. Retrying with gradient "
                "accumulation..."
            )
            trainer_kwargs["accumulate_grad_batches"] = (
                trainer_kwargs.get("accumulate_grad_batches", 1) * 2
            )
            optimization_config.gradient_accumulation = trainer_kwargs["accumulate_grad_batches"]
            optimization_config.batch_size = optimization_config.batch_size // 2
            optimization_config.to_json_file(
                cfg.save_dir / "optimization_config.json", do_overwrite=True
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_pyd,
                batch_size=optimization_config.batch_size,
                num_workers=cfg.num_dataloader_workers,
                collate_fn=train_pyd.collate,
                shuffle=True,
            )
            tuning_dataloader = torch.utils.data.DataLoader(
                tuning_pyd,
                batch_size=optimization_config.batch_size // 2,
                num_workers=cfg.num_dataloader_workers,
                collate_fn=tuning_pyd.collate,
                shuffle=False,
            )

    if cfg.do_final_validation_on_metrics:
        trainer.validate(model=LM, dataloaders=tuning_dataloader)
        trainer.test(model=LM, dataloaders=held_out_dataloader)

    return config, LM

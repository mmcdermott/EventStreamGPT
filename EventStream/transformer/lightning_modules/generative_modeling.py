import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import lightning as L
import omegaconf
import torch
import torch.multiprocessing
import torchmetrics
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
)
from transformers import get_polynomial_decay_schedule_with_warmup

from ...data.config import PytorchDatasetConfig
from ...data.pytorch_dataset import PytorchDataset
from ...data.types import DataModality, PytorchBatch
from ...utils import hydra_dataclass, task_wrapper
from ..conditionally_independent_model import CIPPTForGenerativeSequenceModeling
from ..config import (
    Averaging,
    MetricCategories,
    Metrics,
    MetricsConfig,
    OptimizationConfig,
    Split,
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
)
from ..model_output import GenerativeSequenceModelOutput
from ..nested_attention_model import NAPPTForGenerativeSequenceModeling
from ..utils import expand_indexed_regression, str_summary


class ESTForGenerativeSequenceModelingLM(L.LightningModule):
    """A PyTorch Lightning Module for a `ESTForGenerativeSequenceModeling`."""

    TRAIN_SKIP_METRICS = ("AUROC", "AUPRC", "per_class")
    CLASSIFICATION = {
        DataModality.SINGLE_LABEL_CLASSIFICATION,
        DataModality.MULTI_LABEL_CLASSIFICATION,
    }

    def __init__(
        self,
        config: StructuredTransformerConfig | dict[str, Any],
        optimization_config: OptimizationConfig | dict[str, Any],
        metrics_config: MetricsConfig | dict[str, Any],
        pretrained_weights_fp: Path | None = None,
    ):
        """Initializes the Lightning Module.

        Args:
            config (`Union[StructuredEventstreamTransformerConfig, Dict[str, Any]]`):
                The configuration for the underlying
                `ESTForGenerativeSequenceModeling` model. Should be
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
        if type(metrics_config) is dict:
            metrics_config = MetricsConfig(**metrics_config)

        self.config = config
        self.optimization_config = optimization_config
        self.metrics_config = metrics_config

        self.save_hyperparameters(
            {
                "config": config.to_dict(),
                "optimization_config": dataclasses.asdict(optimization_config),
            }
        )
        self.build_metrics()

        match config.structured_event_processing_mode:
            case StructuredEventProcessingMode.NESTED_ATTENTION:
                model_cls = NAPPTForGenerativeSequenceModeling
            case StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
                model_cls = CIPPTForGenerativeSequenceModeling
            case _:
                raise ValueError(
                    f"Unsupported structured event processing mode: {config.structured_event_processing_mode}"
                )

        if pretrained_weights_fp is None:
            self.model = model_cls(config)
        else:
            self.model = model_cls.from_pretrained(pretrained_weights_fp, config=config)

    def save_pretrained(self, model_dir: Path):
        fp = model_dir / "pretrained_weights"
        self.model.save_pretrained(fp)

    def build_metrics(self):
        """Build the various torchmetrics we'll use to track performance."""

        # For judging our ability to predict time-to-event, we'll use the following scores:
        #   1. Explained Variance
        #   2. Mean Squared Error
        #   3. Mean Squared Log Error
        self.tte_metrics = torch.nn.ModuleDict(
            {
                "MSE": torchmetrics.MeanSquaredError(),
                "MSLE": torchmetrics.MeanSquaredLogError(),
                "explained_variance": torchmetrics.ExplainedVariance(),
            }
        )

        self.metrics = torch.nn.ModuleDict()
        for task_type, measurements in self.config.measurements_per_generative_mode.items():
            for measurement in measurements:
                vocab_size = self.config.vocab_sizes_by_measurement[measurement]

                if measurement not in self.metrics:
                    self.metrics[measurement] = torch.nn.ModuleDict()
                if task_type not in self.metrics[measurement]:
                    self.metrics[measurement][task_type] = torch.nn.ModuleDict()

                match task_type:
                    case DataModality.SINGLE_LABEL_CLASSIFICATION:
                        cat = MetricCategories.CLASSIFICATION
                        metrics = {
                            Metrics.ACCURACY: (
                                MulticlassAccuracy,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                            Metrics.AUROC: (
                                MulticlassAUROC,
                                [Averaging.MACRO, Averaging.WEIGHTED],
                            ),
                            Metrics.AUPRC: (
                                MulticlassAveragePrecision,
                                [Averaging.MACRO, Averaging.WEIGHTED],
                            ),
                        }
                        metric_kwargs = {"num_classes": vocab_size, "ignore_index": 0}
                    case DataModality.MULTI_LABEL_CLASSIFICATION:
                        cat = MetricCategories.CLASSIFICATION
                        metrics = {
                            Metrics.ACCURACY: (
                                MultilabelAccuracy,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                            Metrics.AUROC: (
                                MultilabelAUROC,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                            Metrics.AUPRC: (
                                MultilabelAveragePrecision,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                        }
                        metric_kwargs = {"num_labels": vocab_size}
                    case DataModality.UNIVARIATE_REGRESSION:
                        cat = MetricCategories.REGRESSION
                        metrics = {
                            Metrics.MSE: (torchmetrics.MeanSquaredError, [None]),
                            Metrics.EXPLAINED_VARIANCE: (torchmetrics.ExplainedVariance, [None]),
                        }
                        metric_kwargs = {}
                    case DataModality.MULTIVARIATE_REGRESSION:
                        cat = MetricCategories.REGRESSION
                        metrics = {
                            Metrics.MSE: (torchmetrics.MeanSquaredError, [None]),
                            Metrics.EXPLAINED_VARIANCE: (
                                torchmetrics.ExplainedVariance,
                                [Averaging.MACRO, Averaging.WEIGHTED],
                            ),
                        }
                        metric_kwargs = {}
                    case _:
                        raise ValueError(f"Unrecognized modality {task_type}!")

                if not self.metrics_config.do_validate_args:
                    metric_kwargs["validate_args"] = False

                auc_kwargs = {**metric_kwargs, "thresholds": self.metrics_config.n_auc_thresholds}
                for metric, (metric_cls, averagings) in metrics.items():
                    if metric in (Metrics.AUROC, Metrics.AUPRC):
                        metric_cls_kwargs = {**auc_kwargs}
                    else:
                        metric_cls_kwargs = {**metric_kwargs}

                    for averaging in averagings:
                        if averaging is None:
                            metric_name = str(metric)
                            averaging_kwargs = {}
                        else:
                            metric_name = f"{averaging}_{metric}"
                            if metric == Metrics.EXPLAINED_VARIANCE:
                                if averaging == Averaging.MACRO:
                                    avg_str = "uniform_average"
                                elif averaging == Averaging.WEIGHTED:
                                    avg_str = "variance_weighted"
                                else:
                                    raise ValueError(f"{averaging} not supported for explained variance.")

                                averaging_kwargs = {"multioutput": avg_str}
                            else:
                                averaging_kwargs = {"average": averaging}

                        if self.metrics_config.do_log_any(cat, metric_name):
                            self.metrics[measurement][task_type][metric_name] = metric_cls(
                                **metric_cls_kwargs, **averaging_kwargs
                            )

    def _log_metric_dict(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        metrics: dict[str, torchmetrics.Metric],
        split: Split,
        measurement: str,
        cat: MetricCategories,
    ):
        """This helper function logs the set of named metrics for the predictions `preds` and labels `labels`.

        Args:
            `preds` (`torch.Tensor`): The predictions for this metric calculation.
            `labels` (`torch.Tensor`): The labels for this metric calculation.
            `metrics` (`Dict[str, torchmetrics.Metric]`): The metrics to log, by name.
            `skip_metrics` (`Sequence[str]`):
                A list of metrics to skip. Entries are not full metric names, but rather are partial names and
                any metric whose name contains an element of `skip_metrics` will be skipped.
                For example, if `skip_metrics = ['AUROC', 'AUPRC']`, then a metric with name `'macro_AUROC'`
                or `'micro_AUPRC'` would be skipped, whereas a metric named `'weighted_accuracy'` would not.
            `split` (`str`): TODO
            `measurement` (`str`): The measurement of this metric calculation. Affects the log name.
        """
        for metric_name, metric in metrics.items():
            # We'll want to skip a metric if any element of our skip_metrics list is a substring of the metric
            # name:
            if not self.metrics_config.do_log(split, cat, metric_name):
                continue

            try:
                if split != Split.TRAIN:
                    # This is slightly more efficient if we only care about epoch-level outputs.
                    # Source: https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
                    metric.update(preds, labels)
                else:
                    metric(preds, labels)

                self.log(
                    f"{split}_{measurement}_{metric_name}",
                    metric,
                    batch_size=self.optimization_config.batch_size,
                    sync_dist=True,
                )
            except (ValueError, IndexError) as e:
                print(
                    f"Failed to compute {metric_name} for {measurement} "
                    f"with preds ({str_summary(preds)}) and labels ({str_summary(labels)}): {e}."
                )

    def log_tte_metrics(self, results: GenerativeSequenceModelOutput, split: Split):
        # The output of the model for time-to-event (and for regression targets as well) are pytorch
        # distribution objects, not scalars. So, for some evaluation metrics, we need to sample values from
        # those distributions to assess the metric.
        # TODO(mmd): We should likely be able to control how many samples are used, to minimize variance of
        # these results.
        tte_dist = results["preds"]["time_to_event"]
        tte_preds = tte_dist.sample()

        # After sampling, we also need to slice this down to just those intra-event-times that are actually
        # observed. This means we should drop the last sequence element (slice to `[:, :-1]` (as our observed
        # intra-event-times will only exist for the interior of our sequence), then further filter down to
        # just elements of the prediction for which the next sequence element was not masked
        # (mask via `results['event_mask'][:, 1:]`). We also need to filter the observed labels down to also
        # only be present for sequence elements where the next sequence element was truly observed.
        tte_preds = tte_preds[:, :-1][results["event_mask"][:, 1:]]
        tte_labels = results["labels"]["time_to_event"][results["event_mask"][:, 1:]]

        # Finally, we can log all relevant TTE metrics given these predictions and labels.
        self._log_metric_dict(
            preds=tte_preds,
            labels=tte_labels,
            metrics=self.tte_metrics,
            measurement="TTE",
            split=split,
            cat=MetricCategories.TTE,
        )

    def log_metrics(self, results: GenerativeSequenceModelOutput, split: Split):
        """Logs metric results for a given output result.

        Args:
            `results` (`transformerForGenerativeSequenceModelOutput`):
                The results to assess across the suite of metrics.
            `split` (`str`): The split that should be used when logging metric results.
        """

        # We always want to log the raw loss.
        log_kwargs = {"batch_size": self.optimization_config.batch_size, "sync_dist": True}
        self.log(f"{split}_loss", results["loss"], **log_kwargs)

        if self.metrics_config.do_log_only_loss(split):
            return

        # We start by logging the losses.
        if self.metrics_config.do_log(split, MetricCategories.LOSS_PARTS):
            self.log_dict(
                {f"{split}_{k}_cls_NLL": v for k, v in results["losses"]["classification"].items()},
                **log_kwargs,
            )
            self.log_dict(
                {f"{split}_{k}_reg_NLL": v for k, v in results["losses"]["regression"].items()},
                **log_kwargs,
            )
            self.log(f"{split}_TTE_reg_NLL", results["losses"]["time_to_event"], **log_kwargs)

        # Time-to-event
        if self.metrics_config.do_log(split, MetricCategories.TTE):
            self.log_tte_metrics(results, split)

        # Per data type
        for measurement, metrics_dict in self.metrics.items():
            mask = results["event_mask"]

            if not mask.any():
                continue

            for task_type, metrics in metrics_dict.items():
                if task_type in self.CLASSIFICATION and self.metrics_config.do_log(
                    split, MetricCategories.CLASSIFICATION
                ):
                    # For now, we ignore the is_observed distribution (the first element of the below tuple).
                    _, sample_dist = results["preds"]["classification"][measurement]
                    preds = sample_dist.logits
                    labels = results["labels"]["classification"][measurement]

                    # We need to filter these down to just those corresponding to observed events. Note that
                    # unlike TTE, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.

                    preds = preds[mask]
                    labels = labels[mask].long()

                    self._log_metric_dict(
                        preds=preds,
                        labels=labels,
                        metrics=metrics,
                        measurement=measurement,
                        split=split,
                        cat=MetricCategories.CLASSIFICATION,
                    )

                elif task_type == DataModality.MULTIVARIATE_REGRESSION and self.metrics_config.do_log(
                    split, MetricCategories.REGRESSION
                ):
                    vocab_size = self.config.vocab_sizes_by_measurement[measurement]

                    # Here, like for TTE, we need to sample from the returned distribution before we can use
                    # it directly. Here we also need to limit to just those events that are actually observed.
                    # Like above, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.
                    dist = results["preds"]["regression"][measurement]
                    preds = dist.sample()[mask]
                    labels = results["labels"]["regression"][measurement][mask]

                    # However, as our regression output is actually indexed only to the group keys that are
                    # actually measured (tracked in `results['preds']['regression_indices']`, we need to
                    # expand our predictions and labels to be in the full vocabulary space for the metrics to
                    # work naturally.
                    preds_indices = results["preds"]["regression_indices"][measurement][mask]
                    labels_indices = results["labels"]["regression_indices"][measurement][mask]

                    # We also need to reflect just those data elements for which values were observed:
                    data_el_mask = results["dynamic_values_mask"][mask]

                    preds = preds[data_el_mask]
                    labels = labels[data_el_mask]
                    preds_indices = preds_indices[data_el_mask]
                    labels_indices = labels_indices[data_el_mask]

                    preds_expanded = expand_indexed_regression(preds, preds_indices, vocab_size)
                    labels_expanded = expand_indexed_regression(labels, labels_indices, vocab_size)

                    self._log_metric_dict(
                        preds=preds_expanded,
                        labels=labels_expanded,
                        metrics=metrics,
                        measurement=measurement,
                        split=split,
                        cat=MetricCategories.REGRESSION,
                    )
                elif task_type == DataModality.UNIVARIATE_REGRESSION and self.metrics_config.do_log(
                    split, MetricCategories.REGRESSION
                ):
                    # Here, like for TTE, we need to sample from the returned distribution before we can use
                    # it directly. Here we also need to limit to just those events that are actually observed.
                    # Like above, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.
                    # We ignore the is observed distribution here.
                    _, dist = results["preds"]["regression"][measurement]
                    preds = dist.sample()[mask]
                    labels = results["labels"]["regression"][measurement][mask]

                    self._log_metric_dict(
                        preds=preds,
                        labels=labels,
                        metrics=metrics,
                        measurement=measurement,
                        split=split,
                        cat=MetricCategories.REGRESSION,
                    )

    def training_step(self, batch: PytorchBatch, batch_idx: int) -> torch.Tensor:
        """Training step.

        Skips logging all AUROC, AUPRC, and per_class metric to save compute.
        """
        out = self.model(batch)
        self.log_metrics(out, split=Split.TRAIN)

        return out["loss"]

    def validation_step(self, batch: PytorchBatch, batch_idx: int):
        """Validation step.

        Differs from training only in that it does not skip metrics.
        """
        out = self.model(batch)
        self.log_metrics(out, split=Split.TUNING)

    def test_step(self, batch: PytorchBatch, batch_idx: int):
        """Validation step.

        Differs from training only in that it does not skip metrics.
        """
        out = self.model(batch)
        self.log_metrics(out, split=Split.HELD_OUT)

    def configure_optimizers(self):
        """Configures optimizer and learning rate scheduler.

        Currently this module uses the AdamW optimizer, with configurable weight_decay, with a learning rate
        warming up from 0 on a per-step manner to the configurable `self.optimization_config.init_lr`, then
        undergoes polynomial decay as specified via `self.optimization_config`.
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


SKIP_CFG_PARAMS = {"seq_attention_layers", "dep_graph_attention_layers"}


@hydra_dataclass
class PretrainConfig:
    do_overwrite: bool = False
    seed: int = 1

    config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "_target_": "EventStream.transformer.config.StructuredTransformerConfig",
            **{
                k: v
                for k, v in StructuredTransformerConfig(measurements_per_dep_graph_level=[]).to_dict().items()
                if k not in SKIP_CFG_PARAMS
            },
        }
    )
    optimization_config: OptimizationConfig = OptimizationConfig()
    data_config: PytorchDatasetConfig = PytorchDatasetConfig()
    pretraining_metrics_config: MetricsConfig = MetricsConfig(do_skip_all_metrics=True)
    final_validation_metrics_config: MetricsConfig = MetricsConfig(do_skip_all_metrics=False)

    trainer_config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "accelerator": "auto",
            "devices": "auto",
            "detect_anomaly": False,
            "default_root_dir": "${save_dir}/model_checkpoints",
            "log_every_n_steps": 10,
        }
    )

    experiment_dir: str = omegaconf.MISSING
    save_dir: str = "${experiment_dir}/pretrain/${now:%Y-%m-%d_%H-%M-%S}"

    wandb_logger_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "generative_event_stream_transformer",
            "project": None,
            "team": None,
            "log_model": True,
            "do_log_graph": True,
        }
    )

    wandb_experiment_config_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "save_dir": "${save_dir}",
        }
    )

    num_dataloader_workers: int = 1

    do_final_validation_on_metrics: bool = True

    # compile: bool = True

    def __post_init__(self):
        if type(self.save_dir) is str and self.save_dir != omegaconf.MISSING:
            self.save_dir = Path(self.save_dir)
        if "max_epochs" in self.trainer_config:
            raise ValueError("Max epochs is set in the optimization_config, not the trainer config!")
        if "callbacks" in self.trainer_config:
            raise ValueError("Callbacks are built internally, not set via trainer_config!")


@task_wrapper
def train(cfg: PretrainConfig):
    """Runs the end to end training procedure for the pre-training model.

    Args:
        cfg: The pre-training config defining the generative modeling task.
    """

    L.seed_everything(cfg.seed)
    torch.multiprocessing.set_sharing_strategy("file_system")

    train_pyd = PytorchDataset(cfg.data_config, split="train")
    tuning_pyd = PytorchDataset(cfg.data_config, split="tuning")

    config = cfg.config
    optimization_config = cfg.optimization_config
    data_config = cfg.data_config

    config.set_to_dataset(train_pyd)
    optimization_config.set_to_dataset(train_pyd)

    if os.environ.get("LOCAL_RANK", "0") == "0":
        cfg.save_dir.mkdir(parents=True, exist_ok=True)

        print("Saving config files...")
        config_fp = cfg.save_dir / "config.json"
        if config_fp.exists() and not cfg.do_overwrite:
            raise FileExistsError(f"{config_fp} already exists!")
        else:
            print(f"Writing to {config_fp}")
            config.to_json_file(config_fp)

        data_config.to_json_file(cfg.save_dir / "data_config.json", do_overwrite=cfg.do_overwrite)
        optimization_config.to_json_file(
            cfg.save_dir / "optimization_config.json", do_overwrite=cfg.do_overwrite
        )
        cfg.pretraining_metrics_config.to_json_file(
            cfg.save_dir / "pretraining_metrics_config.json", do_overwrite=cfg.do_overwrite
        )
        cfg.final_validation_metrics_config.to_json_file(
            cfg.save_dir / "final_validation_metrics_config.json", do_overwrite=cfg.do_overwrite
        )

    # Model
    LM = ESTForGenerativeSequenceModelingLM(
        config=config,
        optimization_config=optimization_config,
        metrics_config=cfg.pretraining_metrics_config,
    )

    # TODO(mmd): Get this working!
    # if cfg.compile:
    #     print("Compiling model!")
    #     LM = torch.compile(LM)

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
        batch_size=optimization_config.validation_batch_size,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=tuning_pyd.collate,
        shuffle=False,
    )

    # Setting up model configurations
    # This will track the learning rate value as it updates through warmup and decay.
    callbacks = [LearningRateMonitor(logging_interval="step")]
    if optimization_config.patience is not None:
        callbacks.append(
            EarlyStopping(monitor="tuning_loss", mode="min", patience=optimization_config.patience)
        )

    trainer_kwargs = dict(
        **cfg.trainer_config,
        max_epochs=optimization_config.max_epochs,
        callbacks=callbacks,
    )

    if cfg.wandb_logger_kwargs.get("name", None):
        if "do_log_graph" in cfg.wandb_logger_kwargs:
            do_log_graph = cfg.wandb_logger_kwargs.pop("do_log_graph")
        else:
            do_log_graph = False

        wandb_logger = WandbLogger(
            **{k: v for k, v in cfg.wandb_logger_kwargs.items() if v is not None},
            save_dir=cfg.save_dir,
        )

        if os.environ.get("LOCAL_RANK", "0") == "0":
            if do_log_graph:
                # Watching the model naturally tracks parameter values and gradients.
                wandb_logger.watch(LM, log="all", log_graph=True)

            if cfg.wandb_experiment_config_kwargs:
                wandb_logger.experiment.config.update(cfg.wandb_experiment_config_kwargs)

        trainer_kwargs["logger"] = wandb_logger

    if (optimization_config.gradient_accumulation is not None) and (
        optimization_config.gradient_accumulation > 1
    ):
        trainer_kwargs["accumulate_grad_batches"] = optimization_config.gradient_accumulation

    # Fitting model
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(model=LM, train_dataloaders=train_dataloader, val_dataloaders=tuning_dataloader)

    LM.save_pretrained(cfg.save_dir)

    if cfg.do_final_validation_on_metrics:
        held_out_pyd = PytorchDataset(cfg.data_config, split="held_out")
        held_out_dataloader = torch.utils.data.DataLoader(
            held_out_pyd,
            batch_size=optimization_config.validation_batch_size,
            num_workers=optimization_config.num_dataloader_workers,
            collate_fn=held_out_pyd.collate,
            shuffle=False,
        )

        LM.metrics_config = cfg.final_validation_metrics_config
        LM.build_metrics()

        tuning_metrics = trainer.validate(model=LM, dataloaders=tuning_dataloader)
        held_out_metrics = trainer.test(model=LM, dataloaders=held_out_dataloader)

        if os.environ.get("LOCAL_RANK", "0") == "0":
            print("Saving final metrics...")

            with open(cfg.save_dir / "tuning_metrics.json", mode="w") as f:
                json.dump(tuning_metrics, f)
            with open(cfg.save_dir / "held_out_metrics.json", mode="w") as f:
                json.dump(held_out_metrics, f)

        return tuning_metrics[0]["tuning_loss"], tuning_metrics, held_out_metrics

    return None

import importlib.util
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.multiprocessing
import torchmetrics
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

from ...data.pytorch_dataset import PytorchDataset
from ...data.types import PytorchBatch
from ...utils import task_wrapper
from ..conditionally_independent_model import CIPPTForGenerativeSequenceModeling
from ..config import StructuredEventProcessingMode, StructuredTransformerConfig
from ..model_output import StreamClassificationModelOutput
from ..nested_attention_model import NAPPTForGenerativeSequenceModeling
from ..utils import safe_weighted_avg, str_summary
from ..zero_shot_labeler import Labeler
from .fine_tuning import FinetuneConfig


class ESTForZeroShotClassificationLM(L.LightningModule):
    """A PyTorch Lightning Module for a zero-shot classification via generation for an EST model."""

    def __init__(
        self,
        config: StructuredTransformerConfig | dict[str, Any],
        pretrained_weights_fp: Path,
        labeling_function: Labeler,
        num_samples: int = 10,
        max_new_events: int = 10,
    ):
        """Initializes the Lightning Module.

        Args:
            config (`Union[StructuredTransformerConfig, Dict[str, Any]]`):
                The configuration for the underlying
                `StructuredForStreamClassification` model. Should be
                in the dedicated `StructuredTransformerConfig` class or be a dictionary
                parseable as such.
        """
        super().__init__()

        # If the configurations are dictionaries, convert them to class objects. They may be passed as
        # dictionaries when the lightning module is loaded from a checkpoint, so we need to support
        # this functionality.
        if type(config) is dict:
            config = StructuredTransformerConfig(**config)

        self.config = config
        self.num_samples = config.task_specific_params["num_samples"]
        self.max_new_events = max_new_events
        self.labeling_function = labeling_function

        self.save_hyperparameters(
            {
                "config": config.to_dict(),
                "num_samples": num_samples,
                "max_new_events": max_new_events,
            }
        )
        self.build_metrics()

        if pretrained_weights_fp is None:
            raise ValueError("pretrained_weights_fp must be specified")
        elif self.config.structured_event_processing_mode == StructuredEventProcessingMode.NESTED_ATTENTION:
            self.model = NAPPTForGenerativeSequenceModeling.from_pretrained(
                pretrained_weights_fp, config=config
            )
        else:
            self.model = CIPPTForGenerativeSequenceModeling.from_pretrained(
                pretrained_weights_fp, config=config
            )

    def build_metrics(self):
        """Build the various torchmetrics we'll use to track performance."""

        if (self.config.problem_type == "single_label_classification") and (self.config.num_labels > 2):
            metric_kwargs = {"num_classes": self.config.num_labels}

            # For judging classification, we'll use macro & weighted accuracy, AUROC, and AUPRC
            self.metrics = torch.nn.ModuleDict(
                {
                    "macro_AUROC": MulticlassAUROC(**metric_kwargs, average="macro"),
                    "weighted_AUROC": MulticlassAUROC(**metric_kwargs, average="weighted"),
                    "macro_accuracy": MulticlassAccuracy(**metric_kwargs, average="macro"),
                    "weighted_accuracy": MulticlassAccuracy(**metric_kwargs, average="weighted"),
                    "micro_accuracy": MulticlassAccuracy(**metric_kwargs, average="micro"),
                    "macro_AUPRC": MulticlassAveragePrecision(**metric_kwargs, average="macro"),
                    "weighted_AUPRC": MulticlassAveragePrecision(**metric_kwargs, average="weighted"),
                }
            )
        elif (self.config.problem_type == "single_label_classification") and (self.config.num_labels == 2):
            metric_kwargs = {}

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
                    "weighted_AUPRC": MultilabelAveragePrecision(**metric_kwargs, average="weighted"),
                    "micro_AUPRC": MultilabelAveragePrecision(**metric_kwargs, average="micro"),
                }
            )
        else:
            raise ValueError(f"{self.config.problem_type} not valid")
        # self.metrics_defined = False

    def _log_metric_dict(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        metrics: dict[str, torchmetrics.Metric],
        skip_metrics: Sequence[str],
        prefix: str,
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
            `prefix` (`str`):
                The prefix that should be used when logging metric results. Will likely be 'train', 'tuning',
                or 'held_out', for example.
        """
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
        self,
        results: StreamClassificationModelOutput,
        unpredictable: torch.BoolTensor,
        skip_metrics: Sequence[str],
        prefix: str,
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

        self.log(
            f"{prefix}_frac_unpredictable",
            unpredictable.float().mean(),
            on_step=False,
            on_epoch=True,
        )

        self._log_metric_dict(
            preds=results.preds,
            labels=results.labels,
            metrics=self.metrics,
            skip_metrics=skip_metrics,
            prefix=prefix,
        )

    def get_generative_predictions(self, batch: PytorchBatch) -> StreamClassificationModelOutput:
        """# capture num_samples to generate"""

        empirical_labels, labels_unpredicted = self.labeling_function(
            self.model.generate(
                batch,
                max_new_events=self.max_new_events,
                do_sample=True,
                return_dict_in_generate=False,
                output_scores=False,
                num_return_sequences=self.num_samples,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=True,
            ),
            input_seq_len=batch.sequence_length,
        )

        # empirical_labels is of shape [batch_size * num_samples, num_labels], and stores a binary indicator
        # of a prediction for each generated sample.
        # labels_unpredicted is of shape [batch_size * num_samples], and stores a boolean indicator of whether
        # or not the label was able to be predicted from the generated output. If this is True, then the label
        # was not predicted, and we should not count it in our metrics. We also want to track how frequently
        # this happens.

        # We'll reshape empirical_labels to be of shape [batch_size, num_samples, num_labels], so that we can
        # average over the num_samples dimension. We'll then take the weighted mean over the num_samples
        # dimension (weighted by whether or not the label was predicted) to get the empirical label for each
        # batch.

        empirical_labels = empirical_labels.reshape(
            batch.batch_size, self.num_samples, self.config.num_labels
        ).float()
        weighting_factor = (
            ~labels_unpredicted.reshape(batch.batch_size, self.num_samples)
            .unsqueeze(-1)
            .expand_as(empirical_labels)
        )

        empirical_labels = empirical_labels.transpose(1, 2)
        weighting_factor = weighting_factor.transpose(1, 2)

        empirical_labels, denom = safe_weighted_avg(empirical_labels, weighting_factor)
        labels_unpredicted = (
            labels_unpredicted.reshape(batch.batch_size, self.num_samples).float().mean(dim=-1)
        )

        # Remove unpredicted labels
        empirical_labels = empirical_labels[labels_unpredicted != 1]
        true_labels = batch["stream_labels"][self.config.finetuning_task][labels_unpredicted != 1]

        is_binary = self.config.id2label == {0: False, 1: True}

        if is_binary:
            empirical_labels = empirical_labels[:, 1]
            true_labels = true_labels.long()

        output = StreamClassificationModelOutput(
            loss=float("nan") * (torch.ones_like(true_labels[0]).float()),
            preds=empirical_labels,
            labels=true_labels,
        )

        return output, labels_unpredicted

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Differs from training only in that it does not skip metrics.
        """

        output, unpredictable = self.get_generative_predictions(batch)
        self.log_metrics(output, unpredictable, skip_metrics=[], prefix="tuning")

    def test_step(self, batch, batch_idx):
        """Validation step.

        Differs from training only in that it does not skip metrics.
        """

        output, unpredictable = self.get_generative_predictions(batch)
        self.log_metrics(output, unpredictable, skip_metrics=[], prefix="held_out")


def import_class_from_file(module_path, class_name):
    spec = importlib.util.spec_from_file_location(class_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


@task_wrapper
def zero_shot_evaluation(cfg: FinetuneConfig):
    L.seed_everything(cfg.seed)
    torch.multiprocessing.set_sharing_strategy("file_system")

    tuning_pyd = PytorchDataset(cfg.data_config, split="tuning")
    held_out_pyd = PytorchDataset(cfg.data_config, split="held_out")

    config = cfg.config
    cfg.data_config
    batch_size = cfg.optimization_config.validation_batch_size
    num_dataloader_workers = cfg.optimization_config.num_dataloader_workers

    orig_max_seq_len = config.max_seq_len
    orig_mean_log_inter_event_time = config.mean_log_inter_event_time_min
    orig_std_log_inter_event_time = config.std_log_inter_event_time_min
    config.set_to_dataset(tuning_pyd)
    config.max_seq_len = orig_max_seq_len
    config.mean_log_inter_event_time_min = orig_mean_log_inter_event_time
    config.std_log_inter_event_time_min = orig_std_log_inter_event_time

    # Load the labeler
    labeler_fp = cfg.data_config.save_dir / "task_dfs" / f"{cfg.task_df_name}_labeler.py"
    labeler_cls = import_class_from_file(labeler_fp, "TaskLabeler")

    labeling_function = labeler_cls(config=config)

    # Model
    LM = ESTForZeroShotClassificationLM(
        config=config,
        pretrained_weights_fp=cfg.pretrained_weights_fp,
        labeling_function=labeling_function,
        max_new_events=(orig_max_seq_len - tuning_pyd.max_seq_len),
    )

    # Setting up torch dataloader
    tuning_dataloader = torch.utils.data.DataLoader(
        tuning_pyd,
        batch_size=batch_size,
        num_workers=num_dataloader_workers,
        collate_fn=tuning_pyd.collate,
        shuffle=False,
    )
    held_out_dataloader = torch.utils.data.DataLoader(
        held_out_pyd,
        batch_size=batch_size,
        num_workers=num_dataloader_workers,
        collate_fn=held_out_pyd.collate,
        shuffle=False,
    )

    trainer_kwargs = {**cfg.trainer_config}

    if cfg.wandb_logger_kwargs.get("name", None):
        if "do_log_graph" in cfg.wandb_logger_kwargs:
            do_log_graph = cfg.wandb_logger_kwargs.pop("do_log_graph")
        else:
            do_log_graph = False

        wandb_logger = WandbLogger(
            **{k: v for k, v in cfg.wandb_logger_kwargs.items() if v is not None},
            save_dir=cfg.save_dir,
        )

        if do_log_graph:
            # Watching the model naturally tracks parameter values and gradients.
            wandb_logger.watch(LM, log="all", log_graph=True)

        if cfg.wandb_experiment_config_kwargs:
            wandb_logger.experiment.config.update(cfg.wandb_experiment_config_kwargs)

        trainer_kwargs["logger"] = wandb_logger

    trainer = L.Trainer(**trainer_kwargs)

    tuning_metrics = trainer.validate(model=LM, dataloaders=tuning_dataloader)
    held_out_metrics = trainer.test(model=LM, dataloaders=held_out_dataloader)

    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("Saving final metrics...")
        cfg.save_dir.mkdir(parents=True, exist_ok=True)

        with open(cfg.save_dir / "zero_shot_tuning_metrics.json", mode="w") as f:
            json.dump(tuning_metrics, f)
        with open(cfg.save_dir / "zero_shot_held_out_metrics.json", mode="w") as f:
            json.dump(held_out_metrics, f)

    return tuning_metrics, held_out_metrics

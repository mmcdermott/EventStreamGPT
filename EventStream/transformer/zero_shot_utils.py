import dataclasses
import json
import os
import random
from collections.abc import Sequence
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
from ..data.pytorch_dataset import PytorchDataset
from ..data.types import DataModality, PytorchBatch
from ..utils import hydra_dataclass, task_wrapper
from .config import (
    Averaging,
    MetricCategories,
    Metrics,
    MetricsConfig,
    OptimizationConfig,
    Split,
    StructuredTransformerConfig,
)
from .model import ESTForGenerativeSequenceModeling
from .model_output import GenerativeSequenceModelOutput
from .model_output import StreamClassificationModelOutput
from ..utils import hydra_dataclass, task_wrapper
from .utils import expand_indexed_regression, str_summary
from .utils import str_summary
from .stream_classification_lightning import FinetuneConfig

import polars as pl

def get_death_event_from_config(config, verbose=False):
    """
    input:
        vocabulary_config.json
    returns:
        indices of all measaurements corresponding to death events.
    """

    death_event_indices=[]

    for k in config['vocab_offsets_by_measurement'].keys():
        if k not in config['event_types_per_measurement'].keys():continue
        if verbose:print(k)
        offset = config['vocab_offsets_by_measurement'][k]
        if verbose:[event for i, event in enumerate(config['event_types_per_measurement'][k]) if 'DEATH' in event]
        death_event_indices = death_event_indices+ [i+offset for i,event in enumerate(config['event_types_per_measurement'][k]) if 'DEATH' in event]

    # handle event type 1 to 688

    return death_event_indices


    



# def zero_shot_evaluation(model_output, task_df, vocabulary_config, event_name, time_frame=None):
#     """
#     Given a model output, lookup the event_name (to determine if it occured within the given timeframe). Evaluate against task_df.
#     """

#     if event_name == 'in-hospital mortality':
#         # get the end state of the model output, check the discharge disposition

#         # get the offsets
#         all_death_events = get_death_event_from_config(vocabulary_config)

#         model_output.classification[-1]

#         if model_output in all_death_events:
#             pred = 1.
#         else:
#             pred = 0.
    
#     # get patient id
#     label = task_df.loc['patient_id', 'in-hospital mortality']


#     elif event_name == 'imminent mortality':
#         assert time_frame is not None, print('must also pass a time frame in hours to evaluate')
#         # get the end state of the generated sequence
#         time_size = vocabulary_config['vocab_sizes_by_measurement']['time_of_day']
#         time_offset = vocabulary_config['vocab_offsets_by_measurement']['time_of_day']

#         # get death event
#     else:
#         raise NotImplementedError





# v0): something one can import in a python script, has the following signature.
# Assume easy access to MeasurementConfigs -- can pass in by hand for now.
# Inputs:
# Fine-tuning config (https://github.com/mmcdermott/EventStreamML/blob/dev/EventStream/transformer/stream_classification_lightning.py#L281)
# Dataloader can be built from fine-tuning config -- see stream classification for example.
# BUT fine_tuning config must overwrite data_config
# seq_padding_side in the dataset config to be SeqPaddingSide.RIGHT or ’right’ (https://github.com/mmcdermott/EventStreamML/blob/dev/EventStream/data/config.py#L416) 
# Do_include_start_time to be True (this is needed for generation).
# Function for empirical labeling:
# Inputs:
# Generation output: https://github.com/mmcdermott/EventStreamML/blob/dev/EventStream/transformer/generation_utils.py#L80 
# Original batch (PytorchBatch object) so that they can know at what time the original sequences ended. 
# Output: 
# Tensor that looks contains predicted labels per patient (as appropriate integer indices for class option) -- may need to know dataset vocab as well.
# Think about the time-dependent-functor 
# Outputs:
# Final performance metrics for a zero-shot system, using same metrics from fine-tuning (https://github.com/mmcdermott/EventStreamML/blob/dev/EventStream/transformer/stream_classification_lightning.py)
# (advanced): Sync to weights-and-biases under a given model config (much like FT does).



# @hydra_dataclass
# class FinetuneConfig:
#     save_dir: str = omegaconf.MISSING
#     load_from_model_dir: str | Path | None = omegaconf.MISSING
#     pretrained_weights_fp: Path | None = None

#     do_overwrite: bool = False

#     config: dict[str, Any] = dataclasses.field(
#         default_factory=lambda: {
#             "_target_": "EventStream.transformer.config.StructuredTransformerConfig",
#         }
#     )
#     optimization_config: OptimizationConfig = OptimizationConfig()
#     data_config: PytorchDatasetConfig = PytorchDatasetConfig()
#     metrics_config: MetricsConfig = MetricsConfig()

#     task_df_name: str = omegaconf.MISSING
#     task_df_fp: None | (str | Path) = "${data_config.save_dir}/task_dfs/${task_df_name}.parquet"

#     wandb_name: str | None = "generative_event_stream_transformer"
#     wandb_project: str | None = None
#     wandb_team: str | None = None
#     extra_wandb_log_params: dict[str, Any] | None = None
#     log_every_n_steps: int = 50

#     num_dataloader_workers: int = 1

#     do_detect_anomaly: bool = False
#     do_final_validation_on_metrics: bool = True

#     def __post_init__(self):
#         if type(self.save_dir) is str and self.save_dir != omegaconf.MISSING:
#             self.save_dir = Path(self.save_dir)
#         if type(self.task_df_fp) is str and self.task_df_fp != omegaconf.MISSING:
#             self.task_df_fp = Path(self.task_df_fp)

#         if self.task_df_name is None and self.task_df_fp is not None:
#             self.task_df_name = self.task_df_fp.stem



def template_labeling_function(generated_batch):
    """
    input:
       generated (GeneratedModelOutput): the generated batch
    returns:
       (Tensor) : which contains predicted labels per patient (as appropriate integer indices for class option)
    """

    # return a dummy tensor that is the same size as generated batch, of type longtensor, with 1's and 0's.

    dummy_tensor = torch.random.randint(0,2, generated_batc.shape)


    return dummy_tensor


# class TimeDependentFunctor(abc.ABC):
#     """An abstract base class defining the interface necessary for specifying time-dependent
#     functions."""

#     OUTPUT_MODALITY = DataModality.DROPPED

#     def __init__(self, **fn_params):
#         # Default to_dict/from_dict will only work if functions store all __init__ input params as class
#         # member variables, and use those to compute the function values in __call__...
#         for k, val in fn_params.items():
#             setattr(self, k, val)

#         self.link_static_cols = []

#     def to_dict(self) -> dict[str, Any]:
#         return {
#             "class": self.__class__.__name__,
#             "params": {k: v for k, v in vars(self).items() if k != "link_static_cols"},
#         }

#     @classmethod
#     def from_dict(cls, in_dict: dict[str, Any]) -> TimeDependentFunctor:
#         return cls(**in_dict["params"])

#     def __eq__(self, other: TimeDependentFunctor) -> bool:
#         return self.to_dict() == other.to_dict()


class ESTForZeroShotClassificationLM(L.LightningModule):
    """A PyTorch Lightning Module for a `ESTForStreamClassification` model."""

    def __init__(
        self,
        config: StructuredTransformerConfig | dict[str, Any],
        optimization_config: OptimizationConfig | dict[str, Any],
        pretrained_weights_fp: Path | None = None,
        do_debug_mode: bool = True,
        num_samples=1,
        labeling_function = None
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
        self.num_samples=num_samples

        self.save_hyperparameters(
            {
                "config": config.to_dict(),
                "optimization_config": dataclasses.asdict(optimization_config),
            }
        )
        self.build_metrics()

        if pretrained_weights_fp is None:
            self.model = ESTForGenerativeSequenceModeling(config)
        else:
            self.model = ESTForGenerativeSequenceModeling.from_pretrained(
                pretrained_weights_fp, config=config
            )
        assert labeling_function is not None
        self.labeling_function = labeling_function

    # def save_pretrained(self, model_dir: Path):
    #     fp = model_dir / "pretrained_weights"
    #     self.model.save_pretrained(fp)

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
        # self.metrics_defined = False


    def _log_metric_dict(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        metrics: dict[str, torchmetrics.Metric],
        skip_metrics: Sequence[str],
        prefix: str,
    ):
        """This helper function logs the set of named metrics for the predictions `preds` and labels
        `labels`.

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


    def get_generative_predictions(self,batch):
        """
        # capture num_samples to generate
        """

        out = self.labeling_function(M.generate(batch,
                                                max_new_events=self.max_new_events, 
                                                do_sample=True,
                                                return_dict_in_generate=True,
                                                output_scores=False))
        assert out.dtype==torch.Longtensor
        # streamclassificationmdeloutput
        output = StreamClassificationModelOutput(loss)
        output.loss = np.nan
        output.labels=batch['stream_labels'][self.task]
        # if num_return_sequences is larger

        # num_return_sequences >>> expanded batch tiles the model
        # count occurences of each label sum acrosss patient and create a logit.

        output.preds = torch.mean(out,dim=1) # torch.mean along one axis

        # assert output preds shape is the same as th labels shape
        assert len(output.preds.shape)==len(output.labels.shape)
        assert output.preds.shape[0] == output.labels.shape[0]
        assert output.preds.shape[1] == output.labels.shape[1]


        return output



    def validation_step(self, batch, batch_idx):
        """Validation step.

        Differs from training only in that it does not skip metrics.
        """
        # out = self.model(batch)
        ## TODO
        out = get_generative_predictions(batch)
        self.log_metrics(out, skip_metrics=[], prefix="tuning")


    def test_step(self, batch, batch_idx):
        """Validation step.

        Differs from training only in that it does not skip metrics.
        """
        # out = self.model(batch)
        out = get_generative_predictions(batch)
        # postprocess out to make it look like a prediction

        self.log_metrics(out, skip_metrics=[], prefix="held_out")

    





@task_wrapper
def zero_shot_evaluation(cfg: FinetuneConfig, labeling_function=get_death_event_from_config):
    """
    """

    # make sure config is in the right format
    print(type(cfg))
    print(cfg)
    # for k in cfg.keys():
    #     if isinstance(cfg[k],dict):
    #         print(sorted(list(cfg[k].keys())))
    assert cfg.data_config.seq_padding_side=='right', "seq_padding_side must be set to right for generation"
    assert cfg.data_config.do_include_start_time_min, "do_include_start_time_min must be True for generation" # data config overrides

    print(type(cfg))

    # print(cfg.keys())
    # for k in cfg.keys():
    #     if isinstance(cfg[k],dict):
    #         print(sorted(list(cfg[k].keys())))


    # build the dataloader

    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    # task_df = pl.read_parquet(cfg.task_df_name)
    # task_df = pd.read_parquet('/ais/bulbasaur/bretnestor/')

    # Creating or loading training/tuning datasets
    # held_out_pyd = PytorchDataset(cfg.data_config, task_df=task_df, split="held_out")
    tuning_pyd = PytorchDataset(cfg.data_config, split="tuning")
    # held_out_pyd = PytorchDataset(cfg.data_config,  split="held_out")



    config = cfg.config
    optimization_config = cfg.optimization_config
    metrics_config = cfg.metrics_config

    # config.set_to_dataset(train_pyd)
    # optimization_config.set_to_dataset(train_pyd)

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
    LM = ESTForZeroShotClassificationLM(
        config=config,
        optimization_config=optimization_config,
        metrics_config=metrics_config,
        pretrained_weights_fp=cfg.pretrained_weights_fp,
        num_samples=2
    )


    held_out_dataloader = torch.utils.data.DataLoader(
        held_out_pyd,
        batch_size=optimization_config.batch_size // 2,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=held_out_pyd.collate,
        shuffle=False,
    )


    if cfg.do_final_validation_on_metrics:
        trainer.validate(model=LM, dataloaders=tuning_dataloader)
        trainer.test(model=LM, dataloaders=held_out_dataloader)

    return config, LM


import dataclasses
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import lightning as L
import omegaconf
import polars as pl
import torch
import torch.multiprocessing

from ..data.config import PytorchDatasetConfig, SeqPaddingSide
from ..data.pytorch_dataset import PytorchDataset
from ..data.types import PytorchBatch
from ..transformer.conditionally_independent_model import (
    CIPPTForGenerativeSequenceModeling,
)
from ..transformer.config import (
    OptimizationConfig,
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
)
from ..transformer.nested_attention_model import NAPPTForGenerativeSequenceModeling
from ..utils import hydra_dataclass, task_wrapper


class ESTForTrajectoryGeneration(L.LightningModule):
    """A PyTorch Lightning Module for a zero-shot classification via generation for an EST model."""

    def __init__(
        self,
        config: StructuredTransformerConfig | dict[str, Any],
        pretrained_weights_fp: Path,
    ):
        """Initializes the Lightning Module.

        Args:
            config (`Union[StructuredTransformerConfig, Dict[str, Any]]`):
                The configuration for the underlying
                model. Should be in the dedicated `StructuredTransformerConfig` class or be a dictionary
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
        self.max_new_events = config.task_specific_params["max_new_events"]

        self.save_hyperparameters({"config": config.to_dict()})

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

    def predict_step(self, batch: PytorchBatch, batch_idx: int) -> list[PytorchBatch]:
        """Prediction step.

        Generates new samples and writes them out.
        """

        generated_expanded_batch = self.model.generate(
            batch,
            max_new_events=self.max_new_events,
            do_sample=True,
            return_dict_in_generate=False,
            output_scores=False,
            num_return_sequences=self.num_samples,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,
        )
        return generated_expanded_batch.split_repeated_batch(self.num_samples)


@hydra_dataclass
class GenerateConfig:
    load_from_model_dir: str | Path = omegaconf.MISSING
    seed: int = 1

    pretrained_weights_fp: Path | None = None
    save_dir: str | None = None

    do_overwrite: bool = False

    optimization_config: OptimizationConfig = OptimizationConfig()

    task_df_name: str | None = None

    data_config_overrides: dict[str, Any] | None = dataclasses.field(
        default_factory=lambda: {
            "seq_padding_side": SeqPaddingSide.LEFT,
            "do_include_start_time_min": True,
            "do_include_subsequence_indices": True,
            "do_include_subject_id": True,
        }
    )

    trainer_config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "accelerator": "auto",
            "devices": "auto",
            "detect_anomaly": False,
            "default_root_dir": None,
        }
    )

    task_specific_params: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "num_samples": omegaconf.MISSING,
            "max_new_events": omegaconf.MISSING,
        }
    )

    config_overrides: dict[str, Any] = dataclasses.field(default_factory=lambda: {})

    parallelize_conversion: int | None = None

    def __post_init__(self):
        if isinstance(self.save_dir, str):
            self.save_dir = Path(self.save_dir)

        if self.load_from_model_dir in (None, omegaconf.MISSING):
            raise ValueError("Must load from a model!")

        if type(self.load_from_model_dir) is str:
            self.load_from_model_dir = Path(self.load_from_model_dir)

        if self.pretrained_weights_fp is None:
            self.pretrained_weights_fp = self.load_from_model_dir / "pretrained_weights"
        if self.save_dir is None:
            if self.task_df_name is not None:
                self.save_dir = self.load_from_model_dir / "finetuning" / self.task_df_name
            else:
                self.save_dir = self.load_from_model_dir

        if self.trainer_config.get("default_root_dir", None) is None:
            self.trainer_config["default_root_dir"] = self.save_dir / "model_checkpoints"

        data_config_fp = self.load_from_model_dir / "data_config.json"
        print(f"Loading data_config from {data_config_fp}")
        self.data_config = PytorchDatasetConfig.from_json_file(data_config_fp)

        if self.task_df_name is not None:
            self.data_config.task_df_name = self.task_df_name

        for param, val in self.data_config_overrides.items():
            if param == "task_df_name":
                print(
                    f"WARNING: task_df_name is set in data_config_overrides to {val}! "
                    f"Original is {self.task_df_name}. Ignoring data_config_overrides..."
                )
                continue
            print(f"Overwriting {param} in data_config from {getattr(self.data_config, param)} to {val}")
            setattr(self.data_config, param, val)

        config_fp = self.load_from_model_dir / "config.json"
        print(f"Loading config from {config_fp}")
        self.config = StructuredTransformerConfig.from_json_file(config_fp)

        for param, val in self.config_overrides.items():
            print(f"Overwriting {param} in config from {getattr(self.config, param)} to {val}")
            setattr(self.config, param, val)

        if self.task_specific_params is None:
            raise ValueError("Must specify num samples to generate")

        if (
            self.data_config_overrides.get("max_seq_len", None) is None
            and self.task_specific_params.get("max_new_events", None) is not None
        ):
            self.data_config.max_seq_len = (
                self.config.max_seq_len - self.task_specific_params["max_new_events"]
            )

        implied_max_new_events = self.config.max_seq_len - self.data_config.max_seq_len
        if implied_max_new_events <= 0:
            raise ValueError("Implied to not be generating any new events!")

        if self.config.task_specific_params is None:
            self.config.task_specific_params = {}
        self.config.task_specific_params.update(self.task_specific_params)

        if self.task_specific_params.get("max_new_events", None) in (omegaconf.MISSING, None):
            self.config.task_specific_params["max_new_events"] = implied_max_new_events

        assert self.config.task_specific_params["max_new_events"] == implied_max_new_events


@task_wrapper
def generate_trajectories(cfg: GenerateConfig):
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

    output_dir = cfg.save_dir / "generated_trajectories"

    # Model
    LM = ESTForTrajectoryGeneration(
        config=config,
        pretrained_weights_fp=cfg.pretrained_weights_fp,
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

    trainer = L.Trainer(**cfg.trainer_config)
    tuning_trajectories = trainer.predict(model=LM, dataloaders=tuning_dataloader)

    local_rank = os.environ.get("LOCAL_RANK", "0")

    for samp_idx, gen_batches in enumerate(zip(*tuning_trajectories)):
        out_fp = output_dir / "tuning" / f"sample_{samp_idx}_local_rank_{local_rank}.parquet"
        out_fp.parent.mkdir(exist_ok=True, parents=True)

        st_convert = datetime.now()
        print(f"Converting to DFs for sample {samp_idx}...")
        if cfg.parallelize_conversion is not None and cfg.parallelize_conversion > 1:
            with Pool(cfg.parallelize_conversion) as p:
                dfs = p.map(PytorchBatch.convert_to_DL_DF, gen_batches)
        else:
            dfs = [B.convert_to_DL_DF() for B in gen_batches]
        print(f"Conversion done in {datetime.now() - st_convert}")

        st_write = datetime.now()
        print(f"Writing DF to {out_fp}...")
        pl.concat(dfs).write_parquet(out_fp)
        print(f"Writing done in {datetime.now() - st_write}")

    held_out_trajectories = trainer.predict(model=LM, dataloaders=held_out_dataloader)

    for samp_idx, gen_batches in enumerate(zip(*held_out_trajectories)):
        out_fp = output_dir / "held_out" / f"sample_{samp_idx}_local_rank_{local_rank}.parquet"
        out_fp.parent.mkdir(exist_ok=True, parents=True)

        st_convert = datetime.now()
        print(f"Converting to DFs for sample {samp_idx}...")
        if cfg.parallelize_conversion is not None and cfg.parallelize_conversion > 1:
            with Pool(cfg.parallelize_conversion) as p:
                dfs = p.map(PytorchBatch.convert_to_DL_DF, gen_batches)
        else:
            dfs = [B.convert_to_DL_DF() for B in gen_batches]
        print(f"Conversion done in {datetime.now() - st_convert}")
        print(f"Conversion done in {datetime.now() - st_convert}")

        st_write = datetime.now()
        print(f"Writing DF to {out_fp}...")
        pl.concat(dfs).write_parquet(out_fp)
        print(f"Writing done in {datetime.now() - st_write}")

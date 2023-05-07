import dataclasses
from pathlib import Path
from typing import Any

import lightning as L
import omegaconf
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..data.config import PytorchDatasetConfig
from ..data.pytorch_dataset import PytorchDataset
from ..utils import hydra_dataclass
from .config import StructuredEventProcessingMode, StructuredTransformerConfig
from .transformer import StructuredTransformer, StructuredTransformerPreTrainedModel
from .utils import safe_masked_max, safe_weighted_avg


class EmbeddingsOnlyModel(StructuredTransformerPreTrainedModel):
    def __init__(self, config: StructuredTransformerConfig):
        super().__init__(config)
        self.encoder = StructuredTransformer(config)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


class ESTForEmbedding(L.LightningModule):
    """A PyTorch Lightning Module for a `StructuredForStreamClassification` model."""

    def __init__(
        self,
        config: StructuredTransformerConfig | dict[str, Any],
        pretrained_weights_fp: Path,
    ):
        """Initializes the Lightning Module.

        Args:
            config (`Union[StructuredEventstreamTransformerConfig, Dict[str, Any]]`):
                The configuration for the underlying
                `StructuredTransformerForGenerativeSequenceModeling` model. Should be
                in the dedicated `StructuredTransformerConfig` class or be a dictionary
                parseable as such.
            pretrained_weights_fp (`pathlib.Path`):
                The path to the pre-trained model that should be loaded whose embeddings will be retrieved.
        """
        super().__init__()

        # If the configurations are dictionaries, convert them to class objects. They may be passed as
        # dictionaries when the lightning module is loaded from a checkpoint, so we need to support
        # this functionality.
        if type(config) is dict:
            config = StructuredTransformerConfig(**config)

        self.config = config

        self.uses_dep_graph = (
            self.config.structured_event_processing_mode
            == StructuredEventProcessingMode.NESTED_ATTENTION
        )
        self.pooling_method = config.task_specific_params["pooling_method"]

        self.model = EmbeddingsOnlyModel.from_pretrained(pretrained_weights_fp, config=config)

    def training_step(self, batch, batch_idx):
        """This should not be used."""
        raise NotImplementedError("This class can't train; only get pre-trained embeddings!")

    def validation_step(self, batch, batch_idx):
        """This should not be used."""
        raise NotImplementedError("This class can't validate; only get pre-trained embeddings!")

    def predict_step(self, batch, batch_idx):
        """Retrieves the embeddings and returns them."""
        encoded = self.model(batch).last_hidden_state
        event_encoded = encoded[:, :, -1, :] if self.uses_dep_graph else encoded

        # `event_encoded` is of shape [batch X seq X hidden_dim]. For pooling, I want to put the sequence
        # dimension as last, so we'll transpose.
        event_encoded = event_encoded.transpose(1, 2)

        match self.pooling_method:
            case "last":
                return event_encoded[:, :, -1]
            case "max":
                return safe_masked_max(event_encoded, batch["event_mask"])
            case "mean":
                return safe_weighted_avg(event_encoded, batch["event_mask"])[0]
            case "none":
                return event_encoded
            case _:
                raise ValueError(f"{self.pooling_method} is not a supported pooling method.")


@hydra_dataclass
class GetEmbeddingsConfig:
    save_dir: str | Path | None = None
    load_model_dir: str | Path = omegaconf.MISSING
    do_overwrite: bool = False

    batch_size: int = 32
    pooling_method: str = "last"
    splits: list[str] = dataclasses.field(defaultfactory=lambda: ["train", "tuning", "held_out"])

    data_config: PytorchDatasetConfig | None = None

    task_df_name: str = omegaconf.MISSING
    task_df_fp: str | Path | None = None

    wandb_name: str | None = "generative_event_stream_transformer"
    wandb_project: str | None = None
    wandb_team: str | None = None
    extra_wandb_log_params: dict[str, Any] | None = None
    log_every_n_steps: int = 50

    num_dataloader_workers: int = 1

    do_detect_anomaly: bool = False
    do_final_validation_on_metrics: bool = True
    do_load_only: bool = False

    def __post_init__(self):
        for param in ("save_dir", "task_df_fp", "load_model_dir"):
            val = getattr(self, param)
            if type(val) is str and val != omegaconf.MISSING:
                setattr(self, param, Path(val))

        if self.task_df_name is None and self.task_df_fp is not None:
            self.task_df_name = self.task_df_fp.stem


def get_embeddings(cfg: GetEmbeddingsConfig, return_early: bool = False):
    """Runs the end to end training procedure for the ESTForGenerativeSequenceModelingLM model.

    Args: TODO
    """

    if not cfg.load_model_dir.is_dir():
        raise FileNotFoundError(f"The model directory {cfg.load_model_dir} does not exist.")
    if cfg.save_dir is None:
        cfg.save_dir = cfg.load_model_dir / "embeddings"

    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    embeddings_fps = [cfg.save_dir / f"{split}.pt" for split in cfg.splits]
    all([fp.is_file() for fp in embeddings_fps])

    if cfg.do_load_only:
        for fp in embeddings_fps:
            if not fp.is_file():
                raise FileNotFoundError(f"Embeddings not found at {fp} but do_load_only=True.")
        return [torch.load(fp) for fp in embeddings_fps]

    if cfg.data_config is None:
        cfg.data_config = PytorchDatasetConfig.from_json_file(
            cfg.load_model_dir / "data_config.json"
        )

    config = StructuredTransformerConfig.from_json_file(cfg.load_model_dir / "config.json")
    if config.task_specific_params is None:
        config.task_specific_params = {}
        config.task_specific_params["pooling_method"] = cfg.pooling_method

        if config.max_seq_len != cfg.data_config.max_seq_len:
            print(
                f"Warning: `config.max_seq_len` ({config.max_seq_len}) != `data_config.max_seq_len` "
                f"({cfg.data_config.max_seq_len}). Resetting data_config to match."
            )
            cfg.data_config.max_seq_len = config.max_seq_len

    # Creating training/tuning datasets"${data_config.save_dir}/task_dfs/${task_df_name}.parquet"
    if cfg.task_df_fp is None:
        if cfg.task_df_name is None:
            raise ValueError("Either `task_df_fp` or `task_df_name` must be provided.")
        cfg.task_df_fp = cfg.data_config.save_dir / "task_dfs" / f"{cfg.task_df_name}.parquet"
    task_df = pd.read_parquet(cfg.task_df_fp)

    cfg.data_config.to_json_file(cfg.save_dir / "data_config.json", do_overwrite=cfg.do_overwrite)
    task_df.to_parquet(cfg.save_dir / "task_df.parquet")
    config.to_json_file(cfg.save_dir / "config.json")

    # Datasets
    # Model
    pretrained_weights_fp = cfg.load_model_dir / "pretrained_weights"
    if not pretrained_weights_fp.is_dir():
        raise FileNotFoundError(f"Couldn't find pretrained weights at {pretrained_weights_fp}.")
    LM = ESTForEmbedding(config, pretrained_weights_fp)

    out = []
    for split, embeddings_fp in tqdm(
        list(zip(cfg.splits, embeddings_fps)), desc="Embeddings", leave=False
    ):
        if embeddings_fp.is_file() and not cfg.do_overwrite:
            print(
                f"Embeddings already exist at {embeddings_fp}. To overwrite, set `do_overwrite=True`."
            )
            out.append(torch.load(embeddings_fp))
            continue

        pyd = PytorchDataset(config=cfg.data_config, split=split, task_df=task_df)

        # Setting up torch dataloader
        dataloader = torch.utils.data.DataLoader(
            pyd,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_dataloader_workers,
            collate_fn=pyd.collate,
            shuffle=False,
        )

        checkpoints_dir = cfg.save_dir / "model_checkpoints"

        trainer_kwargs = {"max_epochs": 1, "default_root_dir": checkpoints_dir}

        if torch.cuda.is_available():
            trainer_kwargs.update({"accelerator": "gpu", "devices": -1})

        trainer = L.Trainer(**trainer_kwargs)

        # Getting Embeddings model
        embeddings = torch.cat(trainer.predict(LM, dataloader), 0)

        torch.save(embeddings, embeddings_fp)
        out.append(embeddings)

    return out

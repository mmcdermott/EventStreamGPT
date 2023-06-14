import os
from pathlib import Path
from typing import Any

import lightning as L
import torch

from ...data.pytorch_dataset import PytorchDataset
from ..config import StructuredEventProcessingMode, StructuredTransformerConfig
from ..transformer import (
    ConditionallyIndependentPointProcessTransformer,
    NestedAttentionPointProcessTransformer,
    StructuredTransformerPreTrainedModel,
)
from ..utils import safe_masked_max, safe_weighted_avg
from .fine_tuning import FinetuneConfig


class EmbeddingsOnlyModel(StructuredTransformerPreTrainedModel):
    def __init__(self, config: StructuredTransformerConfig):
        super().__init__(config)
        if self.config.structured_event_processing_mode == StructuredEventProcessingMode.NESTED_ATTENTION:
            self.encoder = NestedAttentionPointProcessTransformer(config=config)
        else:
            self.encoder = ConditionallyIndependentPointProcessTransformer(config)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


class ESTForEmbedding(L.LightningModule):
    """A PyTorch Lightning Module for extracting embeddings only model."""

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
            self.config.structured_event_processing_mode == StructuredEventProcessingMode.NESTED_ATTENTION
        )
        self.pooling_method = config.task_specific_params["pooling_method"]

        self.model = EmbeddingsOnlyModel.from_pretrained(pretrained_weights_fp, config=config)

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


def get_embeddings(cfg: FinetuneConfig):
    """Gets embeddings.

    Writes embeddings to
    ``cfg.load_from_model_dir / "embeddings" / cfg.task_df_name / "{split}_embeddings.pt"``.

    Args:
        cfg: The fine-tuning configuration object specifying the cohort for which and model from which you
            wish to get embeddings.
    """

    torch.multiprocessing.set_sharing_strategy("file_system")

    train_pyd = PytorchDataset(cfg.data_config, split="train")
    tuning_pyd = PytorchDataset(cfg.data_config, split="tuning")
    held_out_pyd = PytorchDataset(cfg.data_config, split="held_out")

    config = cfg.config
    cfg.data_config
    optimization_config = cfg.optimization_config

    config.set_to_dataset(train_pyd)

    # Model
    LM = ESTForEmbedding(config, pretrained_weights_fp=cfg.pretrained_weights_fp)

    # Setting up torch dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_pyd,
        batch_size=optimization_config.validation_batch_size,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=train_pyd.collate,
        shuffle=False,
    )
    tuning_dataloader = torch.utils.data.DataLoader(
        tuning_pyd,
        batch_size=optimization_config.validation_batch_size,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=tuning_pyd.collate,
        shuffle=False,
    )
    held_out_dataloader = torch.utils.data.DataLoader(
        held_out_pyd,
        batch_size=optimization_config.validation_batch_size,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=held_out_pyd.collate,
        shuffle=False,
    )

    trainer = L.Trainer(**cfg.trainer_config)

    for sp, dataloader in (
        ("train", train_dataloader),
        ("tuning", tuning_dataloader),
        ("held_out", held_out_dataloader),
    ):
        # Getting Embeddings model
        embeddings = torch.cat(trainer.predict(LM, dataloader), 0)

        embeddings_fp = cfg.load_from_model_dir / "embeddings" / cfg.task_df_name / f"{sp}_embeddings.pt"

        if os.environ.get("LOCAL_RANK", "0") == "0":
            if embeddings_fp.is_file() and not cfg.do_overwrite:
                print(f"Embeddings already exist at {embeddings_fp}. To overwrite, set `do_overwrite=True`.")
            else:
                print(f"Saving {sp} embeddings to {embeddings_fp}.")
                torch.save(embeddings, embeddings_fp)

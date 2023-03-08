import dataclasses, json, torch, pandas as pd, pytorch_lightning as pl
from pathlib import Path
from tqdm.auto import tqdm
from transformers import get_polynomial_decay_schedule_with_warmup
from typing import Any, Dict, List, Optional, Sequence, Union

from .config import (
    StructuredEventProcessingMode,
    StructuredEventStreamTransformerConfig,
    EventStreamOptimizationConfig,
)
from .transformer import StructuredEventStreamTransformer, StructuredEventStreamTransformerPreTrainedModel
from .utils import safe_masked_max, safe_weighted_avg
from ..EventStreamData.event_stream_dataset import EventStreamDataset
from ..EventStreamData.config import EventStreamPytorchDatasetConfig
from ..EventStreamData.event_stream_pytorch_cached_dataset import EventStreamPytorchDataset

class EmbeddingsOnlyModel(StructuredEventStreamTransformerPreTrainedModel):
    def __init__(self, config: StructuredEventStreamTransformerConfig):
        super().__init__(config)
        self.encoder = StructuredEventStreamTransformer(config)

    def forward(self, *args, **kwargs): return self.encoder(*args, **kwargs)

class StructuredEventStreamForEmbeddingLightningModule(pl.LightningModule):
    """A PyTorch Lightning Module for a `StructuredEventStreamForStreamClassification` model."""
    def __init__(
        self,
        config: Union[StructuredEventStreamTransformerConfig, Dict[str, Any]],
        pretrained_weights_fp: Path,
    ):
        """
        Initializes the Lightning Module.

        Args:
            config (`Union[StructuredEventstreamTransformerConfig, Dict[str, Any]]`):
                The configuration for the underlying
                `StructuredEventStreamTransformerForGenerativeSequenceModeling` model. Should be
                in the dedicated `StructuredEventStreamTransformerConfig` class or be a dictionary 
                parseable as such.
            pretrained_weights_fp (`pathlib.Path`):
                The path to the pre-trained model that should be loaded whose embeddings will be retreived.
        """
        super().__init__()

        # If the configurations are dictionaries, convert them to class objects. They may be passed as
        # dictionaries when the lightning module is loaded from a checkpoint, so we need to support
        # this functionality.
        if type(config) is dict: config = StructuredEventStreamTransformerConfig(**config)

        self.config = config

        self.uses_dep_graph = (
            self.config.structured_event_processing_mode == StructuredEventProcessingMode.NESTED_ATTENTION
        )
        self.pooling_method = config.task_specific_params['pooling_method']

        self.model = EmbeddingsOnlyModel.from_pretrained(pretrained_weights_fp, config=config)

    def training_step(self, batch, batch_idx):
        """This should not be used."""
        raise NotImplementedError(f"This class can't train; only get pre-trained embeddings!")

    def validation_step(self, batch, batch_idx):
        """This should not be used."""
        raise NotImplementedError(f"This class can't validate; only get pre-trained embeddings!")

    def predict_step(self, batch, batch_idx):
        """Retrieves the embeddings and returns them."""
        encoded = self.model(batch).last_hidden_state
        event_encoded = encoded[:, :, -1, :] if self.uses_dep_graph else encoded

        # `event_encoded` is of shape [batch X seq X hidden_dim]. For pooling, I want to put the sequence
        # dimension as last, so we'll transpose.
        event_encoded = event_encoded.transpose(1, 2)

        match self.pooling_method:
            case 'cls': return event_encoded[:, :, 0]
            case 'last': return event_encoded[:, :, -1]
            case 'max': return safe_masked_max(event_encoded, batch['event_mask'])
            case 'mean': return safe_weighted_avg(event_encoded, batch['event_mask'])[0]
            case 'none': return event_encoded
            case _: raise ValueError(f"{self.pooling_method} is not a supported pooling method.")

def get_embeddings(
    load_model_dir: Path,
    dataset_dir: Optional[Path] = None,
    dataset: Optional[EventStreamDataset] = None,
    task_df: Optional[pd.DataFrame] = None,
    task_df_name: Optional[str] = None,
    batch_size: int = 32,
    num_dataloader_workers: int = 1,
    pooling_method: str = 'last',
    save_embeddings_dir: Optional[Path] = None,
    config: Optional[StructuredEventStreamTransformerConfig] = None,
    data_config: Optional[EventStreamPytorchDatasetConfig] = None,
    do_overwrite: bool = False,
    get_embeddings_on_splits: Sequence[str] = ['held_out'],
    pyds: Optional[Sequence[EventStreamPytorchDataset]] = None,
    do_load_only: bool = False,
):
    """
    Gets the embeddings for the model saved in `load_model_dir`.
    """

    assert load_model_dir.is_dir()

    if save_embeddings_dir is None:
        save_embeddings_dir = load_model_dir / 'embeddings'
        save_embeddings_dir.mkdir(parents=False, exist_ok=True)

    embeddings_fps = [save_embeddings_dir / f'{split}_embeddings.pt' for split in get_embeddings_on_splits]
    all_present = all([fp.is_file() for fp in embeddings_fps])

    if do_load_only:
        if all_present: return [torch.load(fp) for fp in embeddings_fps]
        else: raise FileNotFoundError(f"Embeddings not found at {embeddings_fps}.")
    elif all_present and not do_overwrite:
        print(f"Embeddings already exist at {embeddings_fps}. To overwrite, set `do_overwrite=True`.")
        return [torch.load(fp) for fp in embeddings_fps]


    if data_config is None:
        data_config = EventStreamPytorchDatasetConfig.from_json_file(load_model_dir / 'data_config.json')

    if config is None:
        config = StructuredEventStreamTransformerConfig.from_json_file(load_model_dir / 'config.json')
        if config.task_specific_params is None: config.task_specific_params = {}
        config.task_specific_params['pooling_method'] = pooling_method

        if config.max_seq_len != data_config.max_seq_len:
            print(f"Warning: `config.max_seq_len` ({config.max_seq_len}) != `data_config.max_seq_len` ({data_config.max_seq_len}).")
            data_config.max_seq_len = config.max_seq_len

    # Creating training/tuning datasets
    assert task_df_name is not None
    if task_df is None:
        assert dataset_dir is not None
        task_df = pd.read_csv(dataset_dir / 'task_dfs' / f'{task_df_name}.csv')

    data_config.to_json_file(save_embeddings_dir / "data_config.json", do_overwrite=do_overwrite)
    task_df.to_csv(save_embeddings_dir / "task_df.csv")
    config.to_json_file(save_embeddings_dir / "config.json")

    # Datasets
    if pyds is None:
        pyds = EventStreamPytorchDataset.load_or_create(
            data_dir=dataset_dir,
            config=data_config,
            splits=get_embeddings_on_splits,
            data=dataset,
            task_df=task_df,
            task_df_name=task_df_name,
        )

    # Model
    pretrained_weights_fp = load_model_dir / 'pretrained_weights'
    if not pretrained_weights_fp.is_dir():
        raise FileNotFoundError(f"Couldn't find pretrained weights at {pretrained_weights_fp}.")
    LM = StructuredEventStreamForEmbeddingLightningModule(config, pretrained_weights_fp)

    out = []
    for split, embeddings_fp, pyd in tqdm(
        list(zip(get_embeddings_on_splits, embeddings_fps, pyds)),
        desc="Getting Embeddings", leave=False
    ):
        if embeddings_fp.is_file() and not do_overwrite:
            print(f"Embeddings already exist at {embeddings_fp}. To overwrite, set `do_overwrite=True`.")
            out.append(torch.load(embeddings_fp))
            continue

        if pyd.max_seq_len != data_config.max_seq_len:
            print(f"Warning: `pyd.max_seq_len` ({pyd.max_seq_len}) != `data_config.max_seq_len` ({data_config.max_seq_len}).")
            pyd.max_seq_len = data_config.max_seq_len

        # Setting up torch dataloader
        dataloader = torch.utils.data.DataLoader(
            pyd,
            batch_size  = batch_size,
            num_workers = num_dataloader_workers,
            collate_fn  = pyd.collate,
            shuffle     = False,
        )

        checkpoints_dir = save_embeddings_dir / "model_checkpoints"

        trainer_kwargs = {'max_epochs': 1, 'default_root_dir': checkpoints_dir}

        if torch.cuda.is_available():
            trainer_kwargs.update({'accelerator': "gpu", 'devices': -1})

        trainer = pl.Trainer(**trainer_kwargs)

        # Getting Embeddings model
        embeddings = torch.cat(trainer.predict(LM, dataloader), 0)

        torch.save(embeddings, embeddings_fp)
        out.append(embeddings)

    return out

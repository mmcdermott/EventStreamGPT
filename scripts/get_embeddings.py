#!/usr/bin/env python
"""Gets the emeddings of a pre-trained model for a user-specified fine-tuning dataset."""

try:
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import hydra
import torch

from EventStream.logger import hydra_loguru_init
from EventStream.transformer.lightning_modules.embedding import (
    FinetuneConfig,
    get_embeddings,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_name="finetune_config")
def main(cfg: FinetuneConfig):
    hydra_loguru_init()
    if type(cfg) is not FinetuneConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")
    return get_embeddings(cfg)


if __name__ == "__main__":
    main()

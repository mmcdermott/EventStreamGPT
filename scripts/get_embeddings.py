#!/usr/bin/env python

try:
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import hydra
import torch

from EventStream.transformer.get_embeddings_lightning import (
    FinetuneConfig,
    get_embeddings,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_name="finetune_config")
def main(cfg: FinetuneConfig):
    if type(cfg) is not FinetuneConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")
    # TODO(mmd): This isn't the right return value for hyperparameter sweeps.
    return get_embeddings(cfg)


if __name__ == "__main__":
    main()

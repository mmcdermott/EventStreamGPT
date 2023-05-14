#!/usr/bin/env python

try:
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import hydra
import torch

from EventStream.transformer.get_embeddings_lightning import (
    GetEmbeddingsConfig,
    get_embeddings,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_name="get_embeddings_config")
def main(cfg: GetEmbeddingsConfig):
    if type(cfg) is not GetEmbeddingsConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")
    # TODO(mmd): This isn't the right return value for hyperparameter sweeps.
    return get_embeddings(cfg)


if __name__ == "__main__":
    main()

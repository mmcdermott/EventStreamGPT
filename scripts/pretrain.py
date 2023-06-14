#!/usr/bin/env python
"""Pre-trains a model from scartch."""

try:
    # This color-codes and prettifies error messages if the script fails.
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import copy
import os

import hydra
import torch
from omegaconf import OmegaConf

from EventStream.transformer.lightning_modules.generative_modeling import (
    PretrainConfig,
    train,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_name="pretrain_config")
def main(cfg: PretrainConfig):
    if type(cfg) is not PretrainConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")
    # TODO(mmd): This isn't the right return value for hyperparameter sweeps.

    if os.environ.get("LOCAL_RANK", "0") == "0":
        cfg_fp = cfg.save_dir / "pretrain_config.yaml"
        cfg_fp.parent.mkdir(exist_ok=True, parents=True)

        cfg_dict = copy.deepcopy(cfg)
        cfg_dict.config = cfg_dict.config.to_dict()
        OmegaConf.save(cfg_dict, cfg_fp)

    return train(cfg)


if __name__ == "__main__":
    main()

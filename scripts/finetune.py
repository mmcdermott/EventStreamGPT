#!/usr/bin/env python

try:
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import copy

import hydra
import torch
from omegaconf import OmegaConf

from EventStream.transformer.stream_classification_lightning import (
    FinetuneConfig,
    train,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_name="finetune_config")
def main(cfg: FinetuneConfig):
    if type(cfg) is not FinetuneConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")
    cfg_fp = cfg.save_dir / "finetune_{cfg.task_df_name}_config.yaml"
    cfg_fp.parent.mkdir(exist_ok=True, parents=True)

    cfg_dict = copy.deepcopy(cfg)
    cfg_dict.config = cfg_dict.config.to_dict()
    OmegaConf.save(cfg_dict, cfg_fp)

    return train(cfg)


if __name__ == "__main__":
    main()

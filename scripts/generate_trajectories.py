#!/usr/bin/env python
"""Fine-tunes a model on a user-specified downstream task."""

try:
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import copy
import os

import hydra
import torch
from omegaconf import OmegaConf

from EventStream.evaluation.general_generative_evaluation import (
    GenerateConfig,
    generate_trajectories,
)
from EventStream.logger import hydra_loguru_init

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_name="generate_config")
def main(cfg: GenerateConfig):
    hydra_loguru_init()
    if type(cfg) is not GenerateConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        cfg_fp = cfg.save_dir / "generate_config.yaml"
        cfg_fp.parent.mkdir(exist_ok=True, parents=True)

        cfg_dict = copy.deepcopy(cfg)
        cfg_dict.config = cfg_dict.config.to_dict()
        OmegaConf.save(cfg_dict, cfg_fp)

    return generate_trajectories(cfg)


if __name__ == "__main__":
    main()

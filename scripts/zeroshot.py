#!/usr/bin/env python
"""Runs zero-shot evaluation over the user-specified fine-tuning task."""

try:
    # This color-codes and prettifies error messages if the script fails.
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import hydra
import torch

from EventStream.transformer.lightning_modules.zero_shot_evaluator import (
    FinetuneConfig,
    zero_shot_evaluation,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_name="finetune_config")
def main(cfg: FinetuneConfig):
    if type(cfg) is not FinetuneConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")
    return zero_shot_evaluation(cfg)


if __name__ == "__main__":
    main()

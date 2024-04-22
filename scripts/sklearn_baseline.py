#!/usr/bin/env python
"""Fine-tunes a model on a user-specified downstream task."""

try:
    # This color-codes and prettifies error messages if the script fails.
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import hydra

from EventStream.baseline.FT_task_baseline import SklearnConfig, wandb_train_sklearn
from EventStream.logger import hydra_loguru_init


@hydra.main(version_base=None, config_name="sklearn_config")
def main(cfg: SklearnConfig):
    hydra_loguru_init()
    if type(cfg) is not SklearnConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")

    return wandb_train_sklearn(cfg)


if __name__ == "__main__":
    main()

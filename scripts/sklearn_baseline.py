#!/usr/bin/env python
"""Fine-tunes a model on a user-specified downstream task."""

try:
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import hydra

from EventStream.evaluation.wandb_sklearn import SklearnConfig, wandb_train_sklearn


@hydra.main(version_base=None, config_name="sklearn_config")
def main(cfg: SklearnConfig):
    if type(cfg) is not SklearnConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")

    return wandb_train_sklearn(cfg)


if __name__ == "__main__":
    main()

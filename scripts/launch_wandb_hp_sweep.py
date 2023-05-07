#!/usr/bin/env python

try:
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

from typing import Any

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

WANDB_SWEEP_KEYS = {"value", "values", "min", "max", "distribution"}


def collapse_cfg(k: str, v: Any) -> dict[str, dict[str, Any]]:
    if type(v) is not dict:
        raise ValueError(f"Misconfigured @ {k}")
    if len(WANDB_SWEEP_KEYS.intersection(v.keys())) > 0:
        return {k: v}

    out = {}
    if k:
        for kk, vv in v.items():
            out.update(collapse_cfg(f"{k}.{kk}", vv))
    else:
        for kk, vv in v.items():
            out.update(collapse_cfg(kk, vv))
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="hyperparameter_sweep")
def main(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg, _convert_="all")
    cfg["command"] = [
        # "WANDB_START_METHOD=thread",
        "${env}",
        "${interpreter}",
        "${program}",
        "${args_no_hyphens}",
    ]

    new_params = {}
    for k, v in cfg["parameters"].items():
        new_params.update(collapse_cfg(k, v))

    cfg["parameters"] = new_params

    sweep_id = wandb.sweep(sweep=cfg)
    return sweep_id


if __name__ == "__main__":
    main()

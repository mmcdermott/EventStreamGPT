#!/usr/bin/env python
"""Launches a [weights and biases](https://wandb.ai/) hyperparameter tuning sweep."""

try:
    # This color-codes and prettifies error messages if the script fails.
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

from typing import Any

import hydra
import wandb
from omegaconf import DictConfig

# This is a (non-exhaustive) set of weights and biases sweep parameter keywords, which is used to indicate
# when a configuration dictionary contains actual parameter choices, rather than further nested parameter
# groups.
WANDB_SWEEP_KEYS: set[str] = {"value", "values", "min", "max", "distribution"}


def collapse_cfg(k: str, v: dict[str, Any]) -> dict[str, Any]:
    """Collapses a nested config into the hydra parameter override syntax.

    The weights and biases sweep configuration system leverages nested parameter groups, but they are
    represented via a different syntax than that which Hydra uses for overrides (dot separated). This function
    converts the former to the latter in the sweep config so that program runs work down the line. The
    dictionary `v` is collapsed to leaves, where leaf is defined by when the dictionary `v` contains any
    sentinel `WANDB_SWEEP_KEYS` keys. If the dictionary `v` contains just the value `None` (`{'value': None}`)
    then an empty dictionary is returned to remove that parameter from the configuration.

    Args:
        k: The string name of the containing parameter group.
        v: The dictionary value of the nested sub-parameter group to be collapsed.

    Returns:
        A single dictionary with nested key strings and all leaf values represented.

    Raises:
        TypeError: if `v` is not a dictionary.

    Examples:
        >>> collapse_cfg("foo", None)
        Traceback (most recent call last):
            ...
        TypeError: Misconfigured @ foo: None (<class 'NoneType'>) is not a dict!
        >>> collapse_cfg("bar", {"values": "vals"})
        {'bar': {'values': 'vals'}}
        >>> collapse_cfg("foo", {"bar": {"baz": {"values": "vals"}}, "biz": {"max": "MX"}})
        {'foo.bar.baz': {'values': 'vals'}, 'foo.biz': {'max': 'MX'}}
        >>> collapse_cfg("foo", {"bar": {"value": None}})
        {}
    """
    if type(v) is not dict:
        raise TypeError(f"Misconfigured @ {k}: {v} ({type(v)}) is not a dict!")
    if len(WANDB_SWEEP_KEYS.intersection(v.keys())) > 0:
        if set(v.keys()) == {"value"} and v["value"] is None:
            return {}
        else:
            return {k: v}

    out = {}
    if k:
        for kk, vv in v.items():
            out.update(collapse_cfg(f"{k}.{kk}", vv))
    else:
        for kk, vv in v.items():
            out.update(collapse_cfg(kk, vv))
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="hyperparameter_sweep_base")
def main(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg, _convert_="all")
    cfg["command"] = [
        "${env}",
        "${interpreter}",
        "${program}",
        "${args_no_hyphens}",
    ]

    new_params = {}
    for k, v in cfg["parameters"].items():
        new_params.update(collapse_cfg(k, v))

    cfg["parameters"] = new_params

    if "cohort_name" in cfg:
        cfg.pop("cohort_name")

    sweep_kwargs = {}
    if "entity" in cfg:
        entity = cfg.pop("entity")
        if entity:
            sweep_kwargs["entity"] = entity
    if "project" in cfg:
        project = cfg.pop("project")
        if project:
            sweep_kwargs["project"] = project

    sweep_id = wandb.sweep(sweep=cfg, **sweep_kwargs)
    return sweep_id


if __name__ == "__main__":
    main()

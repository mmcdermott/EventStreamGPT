#!/usr/bin/env python
"""Builds a flat representation dataset given a hydra config file."""

try:
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

from pathlib import Path

import hydra
from omegaconf import DictConfig

from EventStream.data.dataset_polars import Dataset


@hydra.main(version_base=None, config_path="../configs", config_name="dataset_base")
def main(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg, _convert_="all")
    save_dir = Path(cfg.pop("save_dir"))
    window_sizes = cfg.pop("window_sizes")
    subjects_per_output_file = (
        cfg.pop("subjects_per_output_file") if "subjects_per_output_file" in cfg else None
    )

    # Build flat reps for specified task and window sizes
    ESD = Dataset.load(save_dir)
    feature_inclusion_frequency, include_only_measurements = ESD._resolve_flat_rep_cache_params(
        feature_inclusion_frequency=None, include_only_measurements=None
    )
    cache_kwargs = dict(
        subjects_per_output_file=subjects_per_output_file,
        feature_inclusion_frequency=feature_inclusion_frequency,  # 0.1
        window_sizes=window_sizes,
        include_only_measurements=include_only_measurements,
        do_overwrite=cfg.pop("do_overwrite"),
        do_update=cfg.pop("do_update"),
    )
    ESD.cache_flat_representation(**cache_kwargs)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Builds a dataset given a hydra config file."""

try:
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import math
import shutil
from pathlib import Path

import hydra
import numpy as np
import pyarrow.parquet
from loguru import logger
from tqdm.auto import tqdm

from EventStream.data.dataset_polars import Dataset
from EventStream.logger import hydra_loguru_init
from EventStream.utils import hydra_dataclass


@hydra_dataclass
class ConversionConfig:
    dataset_dir: str | Path
    ESDS_save_dir: str | Path
    do_overwrite: bool = False
    ESDS_chunk_size: int = 20000

    def __post_init__(self):
        if type(self.dataset_dir) is str:
            self.dataset_dir = Path(self.dataset_dir)
        if type(self.ESDS_save_dir) is str:
            self.ESDS_save_dir = Path(self.ESDS_save_dir)


@hydra.main(version_base=None, config_name="conversion_config")
def main(cfg: ConversionConfig):
    hydra_loguru_init()

    if type(cfg) is not ConversionConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")

    out_files = list(cfg.ESDS_save_dir.glob("*.parquet"))
    if len(out_files) > 0 and not cfg.do_overwrite:
        raise FileExistsError(
            f"cfg.do_overwrite={cfg.do_overwrite} but found extant files at {cfg.ESDS_save_dir}"
        )
    elif cfg.do_overwrite and cfg.ESDS_save_dir.is_dir():
        logger.info(f"Overwriting {cfg.ESDS_save_dir}")
        shutil.rmtree(cfg.ESDS_save_dir)

    cfg.ESDS_save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset from {cfg.dataset_dir}")
    ESGPT_dataset = Dataset.load(cfg.dataset_dir)

    for sp, subjs in tqdm(list(ESGPT_dataset.split_subjects.items())):
        n_chunks = int(math.ceil(len(subjs) / cfg.ESDS_chunk_size))
        logger.info(f"Splitting {sp} into {n_chunks} chunks")
        chunks = np.array_split(list(subjs), n_chunks)
        rng = tqdm(enumerate(chunks), total=len(chunks), leave=False, desc=f"Saving {sp}")
        sp_dir = cfg.ESDS_save_dir / sp
        sp_dir.mkdir(exist_ok=True, parents=False)

        for i, subjs_chunk in rng:
            df = ESGPT_dataset.build_ESDS_representation(do_sort_outputs=True, subject_ids=list(subjs_chunk))
            arr_table = df.to_arrow().cast(ESGPT_dataset.ESDS_schema)
            pyarrow.parquet.write_table(arr_table, sp_dir / f"{i}.parquet")


if __name__ == "__main__":
    main()

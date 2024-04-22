#!/usr/bin/env python
import polars as pl, numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig

from EventStream.data.dataset_polars import Dataset

@hydra.main(version_base=None)
def main(cfg: DictConfig):
    dataset_dir = Path(cfg.dataset_dir)

    ESD = Dataset.load(dataset_dir)

    rand = np.random.default_rng(1)

    tasks_labels_types = [
        ("multi_class_classification", lambda N: rand.choice([0, 1, 2], size=N), pl.UInt32),
        ("single_label_binary_classification", lambda N: rand.choice([True, False], size=N), pl.Boolean),
        ("univariate_regression", lambda N: rand.normal(size=N), pl.Float32),
    ]

    for task, label_fn, pl_dtype in tasks_labels_types:
        task_fp = dataset_dir / "task_dfs" / f"{task}.parquet"
        task_fp.parent.mkdir(exist_ok=True, parents=True)

        (
            ESD.events_df
            .group_by('subject_id')
            .agg(pl.col('timestamp').sample().first().alias('end_time'))
            .with_columns(
                pl.lit(label_fn(len(ESD.subject_ids))).cast(pl_dtype).alias('label'),
                pl.lit(None, dtype=pl.Datetime).alias('start_time')
            )
            .write_parquet(task_fp)
        )

if __name__ == "__main__":
    main()

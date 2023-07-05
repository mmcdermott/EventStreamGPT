# %%time
# %%memit

import os
from pathlib import Path

import polars as pl

from EventStream.data.dataset_polars import Dataset

COHORT_NAME = "MIMIC_IV/ESD_06-13-23_150GB_10cpu-1"
PROJECT_DIR = Path(os.environ["PROJECT_DIR"])
DATA_DIR = PROJECT_DIR / "data" / COHORT_NAME
assert DATA_DIR.is_dir()

TASK_DF_DIR = DATA_DIR / "task_dfs"
TASK_DF_DIR.mkdir(exist_ok=True, parents=False)

ESD = Dataset.load(DATA_DIR)


def has_event_type(type_str: str) -> pl.Expr:
    event_types = pl.col("event_type").cast(pl.Utf8).str.split("&")
    return event_types.list.contains(type_str)


events_df = ESD.events_df.lazy()

readmission_30d = (
    events_df.with_columns(
        has_event_type("DISCHARGE").alias("is_discharge"), has_event_type("ADMISSION").alias("is_admission")
    )
    .filter(pl.col("is_discharge") | pl.col("is_admission"))
    .sort(["subject_id", "timestamp"], descending=False)
    .with_columns(
        pl.when(pl.col("is_admission"))
        .then(pl.col("timestamp"))
        .otherwise(None)
        .alias("admission_time")
        .cast(pl.Datetime)
    )
    .with_columns(
        pl.col("admission_time")
        .fill_null(strategy="backward")
        .over("subject_id")
        .alias("next_admission_time"),
        pl.col("admission_time")
        .fill_null(strategy="forward")
        .over("subject_id")
        .alias("prev_admission_time"),
    )
    .with_columns(
        ((pl.col("next_admission_time") - pl.col("timestamp")) < pl.duration(days=30))
        .fill_null(False)
        .alias("30d_readmission")
    )
    .filter(pl.col("is_discharge"))
)

readmission_30d_all = readmission_30d.select(
    "subject_id",
    pl.lit(None).cast(pl.Datetime).alias("start_time"),
    pl.col("timestamp").alias("end_time"),
    "30d_readmission",
)

readmission_30d_all.collect().write_parquet(TASK_DF_DIR / "readmission_30d_all.parquet")

prevalence = readmission_30d_all.select(pl.col("30d_readmission").mean()).collect().item()
print(f"The {COHORT_NAME} cohort has a {prevalence*100:.1f}% 30d readmission prevalence.")

# Loading events from \
# /n/data1/hms/dbmi/zaklab/RAMMS/data/MIMIC_IV/ESD_06-13-23_150GB_10cpu-1/events_df.parquet...
# The MIMIC_IV/ESD_06-13-23_150GB_10cpu-1 cohort has a 32.6% 30d readmission prevalence.
# peak memory: 912.86 MiB, increment: 496.57 MiB
# CPU times: user 7.19 s, sys: 1.34 s, total: 8.53 s
# Wall time: 4.55 s

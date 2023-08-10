import os
import json
import pickle
from pathlib import Path
import polars as pl, numpy as np
import polars.selectors as cs

from EventStream.data.dataset_polars import Dataset
from EventStream.evaluation.FT_task_baseline import (
    load_flat_rep, fit_baseline_task_model, add_tasks_from, summarize_binary_task
)

from sklearn.ensemble import RandomForestClassifier

from pathlib import Path

import polars as pl
from EventStream.data.dataset_polars import Dataset

SUPPORTED_MODELS = {
    "RandomForestClassifier": RandomForestClassifier,
}

@hydra_dataclass
class TaskBaselineConfig:
    dataset_dir: str | Path = omegaconf.MISSING
    tasks: str | tuple[str, str] | list[tuple[str, str]]: = omegaconf.MISSING
    train_subset_size: int | float | None = None
    seed: int = 1
    window_sizes: list[str] = ["6h", "12h", "1d", "3d", "7d", "30d", "180d", "365d", "730d", "3650d", "FULL"]
    hyperparameter_search_budget: int = 10
    n_samples: int = 5

    model_cls: str = omegaconf.MISSING
    model_hyperparameter_dist: dict = field(default_factory=lambda: {})

    error_score: float | str = float("nan")
    verbose: int = 0

    def __post_init__(self):
        match self.tasks:
            case omegaconf.MISSING:
                raise ValueError("task_name must be specified")
            case (str(), str()):
                self.tasks = [self.task_name]
            case list():
                pass
            case _:
                raise ValueError("task_name must be a string or a list of strings")

        match self.dataset_dir:
            case omegaconf.MISSING:
                raise ValueError("dataset_dir must be specified")
            case str():
                self.dataset_dir = Path(self.dataset_dir)
            case Path():
                pass
            case _:
                raise ValueError("dataset_dir must be a string or a pathlib.Path")


def baseline_model(cfg: TaskBaselineConfig):
    ESD = Dataset.load(cfg.dataset_dir)
    all_tasks = add_tasks_from(cfg.dataset_dir / "task_dfs")

    tasks = {}
    for task_name, label_col in cfg.tasks:
        task_df = all_tasks[task_name]
        tasks[task_name] = (task_df, label_col)

    for task_name, (task_df, label_col) in tasks.items()
        out_pipe, subject_ids = fit_baseline_task_model(
            task_df, label_col, ESD, n_samples=cfg.n_samples,
            model_cls=SUPPORTED_MODELS[cfg.model_cls],
            model_param_distributions=cfg.model_hyperparameter_dist,
            verbose=cfg.verbose,
            hyperparameter_search_budget=cfg.hyperparameter_search_budget,
            error_score=cfg.error_score,
            window_size_options=cfg.window_sizes,
            seed=cfg.seed,
            train_subset_size=cfg.train_subset_size,
        )

        save_dir = cfg.dataset_dir / "task_baselines" / task_name
        if cfg.train_subset_size is not None:
            save_dir = save_dir / f"train_subset_size_{cfg.train_subset_size}"

        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "subject_ids.json", "w") as f:
            json.dump(subject_ids, f)

        with open(save_dir / "out_pipe.pkl", "wb") as f:
            pickle.dump(out_pipe, f)

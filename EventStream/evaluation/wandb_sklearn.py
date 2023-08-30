"""Utilities for collecting baseline performance of fine-tuning tasks defined over ESGPT datasets."""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import omegaconf
import polars as pl
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..data.dataset_polars import Dataset
from ..data.pytorch_dataset import PytorchDataset
from ..utils import task_wrapper
from .FT_task_baseline import ESDFlatFeatureLoader, add_tasks_from, load_flat_rep

pl.enable_string_cache(True)

import dataclasses
import inspect
import warnings
from abc import ABC

SKLEARN_CONFIG_MODULES = {}


def registered_sklearn_config(dataclass: Any) -> Any:
    """Decorator that allows you to use a dataclass as a hydra config via the `ConfigStore`

    Adds the decorated dataclass as a `Hydra StructuredConfig object`_ to the `Hydra ConfigStore`_.
    The name of the stored config in the ConfigStore is the snake case version of the CamelCase class name.

    .. _Hydra StructuredConfig object: https://hydra.cc/docs/tutorials/structured_config/intro/

    .. _Hydra ConfigStore: https://hydra.cc/docs/tutorials/structured_config/config_store/
    """

    dataclass = dataclasses.dataclass(dataclass)

    name = dataclass.__name__
    cls_name = name[: -len("Config")]

    if cls_name != dataclass().CLS:
        raise ValueError(f"CLS must be {cls_name} for config class named {name}")

    SKLEARN_CONFIG_MODULES[cls_name] = dataclass

    return dataclass


class BaseSklearnModuleConfig(ABC):
    SKLEARN_COMPONENTS = {
        cls.__name__: cls
        for cls in [
            PCA,
            NMF,
            SelectKBest,
            mutual_info_classif,
            KNNImputer,
            SimpleImputer,
            MinMaxScaler,
            StandardScaler,
            ESDFlatFeatureLoader,
            RandomForestClassifier,
        ]
    }
    SKIP_PARAMS = ["CLS", "SKLEARN_COMPONENTS", "SKIP_PARAMS"]

    CLS: str = omegaconf.MISSING

    def get_model(self, seed: int | None = None, **additional_kwargs) -> Any:
        cls = self.SKLEARN_COMPONENTS[self.CLS]

        kwargs = {**self.module_kwargs, **additional_kwargs}
        kwargs = {k: None if v in ("null", "None") else v for k, v in kwargs.items()}
        signature = inspect.signature(cls)
        for k in list(kwargs.keys()):
            if k not in signature.parameters:
                warnings.warn(f"Parameter {k} not in signature of {cls.__name__}. Dropping")
                del kwargs[k]
        if "random_state" in signature.parameters:
            kwargs["random_state"] = seed
        elif "seed" in signature.parameters:
            kwargs["seed"] = seed

        return self.SKLEARN_COMPONENTS[self.CLS](**kwargs)

    @property
    def module_kwargs(self) -> dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items() if k not in self.SKIP_PARAMS}


@registered_sklearn_config
class RandomForestClassifierConfig(BaseSklearnModuleConfig):
    CLS: str = "RandomForestClassifier"

    n_estimators: int = 100
    criterion: str = "gini"
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: str | None = "sqrt"
    max_leaf_nodes: int | None = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    class_weight: str | None = None
    ccp_alpha: float = 0.0
    max_samples: int | float | None = None


@registered_sklearn_config
class MinMaxScalerConfig(BaseSklearnModuleConfig):
    CLS: str = "MinMaxScaler"


@registered_sklearn_config
class StandardScalerConfig(BaseSklearnModuleConfig):
    CLS: str = "StandardScaler"


@registered_sklearn_config
class SimpleImputerConfig(BaseSklearnModuleConfig):
    CLS: str = "SimpleImputer"

    strategy: str = "constant"
    fill_value: float = 0
    add_indicator: bool = True


@registered_sklearn_config
class NMFConfig(BaseSklearnModuleConfig):
    CLS: str = "NMF"

    n_components: int = 2


@registered_sklearn_config
class PCAConfig(BaseSklearnModuleConfig):
    CLS: str = "PCA"

    n_components: int = 2


@registered_sklearn_config
class SelectKBestConfig(BaseSklearnModuleConfig):
    CLS: str = "SelectKBest"

    k: int = 2


@registered_sklearn_config
class KNNImputerConfig(BaseSklearnModuleConfig):
    CLS: str = "KNNImputer"

    n_neighbors: int = 5
    weights: str = "uniform"
    add_indicator: bool = True


@registered_sklearn_config
class ESDFlatFeatureLoaderConfig(BaseSklearnModuleConfig):
    CLS: str = "ESDFlatFeatureLoader"

    window_sizes: list[str] | None = None
    feature_inclusion_frequency: float | None = None
    include_only_measurements: list[str] | None = None
    convert_to_mean_var: bool = True


@dataclasses.dataclass
class SklearnConfig:
    PIPELINE_COMPONENTS = ["feature_selector", "scaling", "imputation", "dim_reduce", "model"]

    defaults: list[Any] = dataclasses.field(
        default_factory=lambda: [
            "_self_",
            {"feature_selector": "esd_flat_feature_loader"},
            {"scaling": "standard_scaler"},
            {"imputation": "simple_imputer"},
            {"dim_reduce": "pca"},
            {"model": "random_forest_classifier"},
        ]
    )

    seed: int = 1

    experiment_dir: str | Path = omegaconf.MISSING
    dataset_dir: str | Path = omegaconf.MISSING
    save_dir: str | Path = (
        "${experiment_dir}/sklearn_baselines/${task_df_name}/${finetuning_task_label}/"
        "${now:%Y-%m-%d_%H-%M-%S}"
    )

    train_subset_size: int | float | str | None = None

    do_overwrite: bool = False

    task_df_name: str | None = omegaconf.MISSING
    finetuning_task_label: str | None = omegaconf.MISSING

    feature_selector: Any = omegaconf.MISSING
    scaling: Any = None
    imputation: Any = None
    dim_reduce: Any = None
    model: Any = omegaconf.MISSING

    wandb_logger_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "${task_df_name}_sklearn",
            "project": None,
            "entity": None,
            "save_code": True,
        }
    )

    def __post_init__(self):
        if isinstance(self.save_dir, str):
            self.save_dir = Path(self.save_dir)
        if isinstance(self.dataset_dir, str):
            self.dataset_dir = Path(self.dataset_dir)

        match self.train_subset_size:
            case int() as n_subjects if n_subjects > 0:
                pass
            case float() as frac_subjects if 0 < frac_subjects and frac_subjects < 1:
                pass
            case "FULL" | None:
                pass
            case _:
                raise ValueError(
                    "train_subset_size invalid! Must be either None, a positive int, or a float "
                    f"between 0 and 1. Got {self.train_subset_size}."
                )

    def __get_component_model(self, component: str, **kwargs) -> Any:
        if component not in self.PIPELINE_COMPONENTS:
            raise ValueError(f"Unknown component {component}")

        component_val = getattr(self, component)
        match component_val:
            case None:
                return "passthrough"
            case BaseSklearnModuleConfig():
                pass
            case dict() | omegaconf.DictConfig():
                component_val = SKLEARN_CONFIG_MODULES[component_val["CLS"]](**component_val)
                setattr(self, component, component_val)
            case _:
                raise ValueError(
                    f"{component} can only be a SKlearnConfig or None (in which case it is omitted). "
                    f"Got {type(component_val)}({component_val})."
                )

        return component_val.get_model(seed=self.seed, **kwargs)

    def get_model(self, dataset: Dataset) -> Any:
        return Pipeline(
            [("feature_selector", self.__get_component_model("feature_selector", ESD=dataset))]
            + [(n, self.__get_component_model(n)) for n in self.PIPELINE_COMPONENTS[1:]]
        )


cs = ConfigStore.instance()
cs.store(name="sklearn_config", node=SklearnConfig)
cs.store(group="scaling", name="min_max_scaler", node=MinMaxScalerConfig)
cs.store(group="scaling", name="standard_scaler", node=StandardScalerConfig)
cs.store(group="imputation", name="simple_imputer", node=SimpleImputerConfig)
cs.store(group="imputation", name="knn_imputer", node=KNNImputerConfig)
cs.store(group="dim_reduce", name="nmf", node=NMFConfig)
cs.store(group="dim_reduce", name="pca", node=PCAConfig)
cs.store(group="dim_reduce", name="select_k_best", node=SelectKBestConfig)
cs.store(group="feature_selector", name="esd_flat_feature_loader", node=ESDFlatFeatureLoaderConfig)
cs.store(group="model", name="random_forest_classifier", node=RandomForestClassifierConfig)

METRIC_FNS = {
    "NLL": log_loss,
    "AUROC": roc_auc_score,
    "AUPRC": average_precision_score,
    "Accuracy": accuracy_score,
}


def eval_multi_class_classification(Y: np.ndarray, probs: np.ndarray, task_vocab: list[Any]):
    results = {}
    probs_metrics = [("NLL", {"labels": task_vocab})]

    for metric_n, metric_kwargs in [
        ("AUROC/OVO", {"multi_class": "ovo", "labels": task_vocab}),
        ("AUROC/OVR", {"multi_class": "ovr", "labels": task_vocab}),
        ("AUPRC/OVR", {}),
    ]:
        average_methods = ["weighted"]
        if metric_n.endswith("OVR"):
            average_methods.extend([None, "macro"])
        for average in average_methods:
            probs_metrics.append((f"{metric_n}/{average}", {"average": average, **metric_kwargs}))

    for metric_n, metric_kwargs in probs_metrics:
        metric_fn = METRIC_FNS[metric_n.split("/")[0]]
        try:
            output = metric_fn(Y, probs, **metric_kwargs)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed in evaluating {metric_fn}") from e
        if isinstance(output, (list, tuple, np.ndarray)):
            if not metric_n.endswith("/None"):
                raise ValueError(f"Metric {metric_n} returned a list output unexpectedly")
            if len(output) != len(task_vocab):
                raise ValueError(
                    f"Metric returned a sequence of inappropriate length {len(output)} (vocab length "
                    f"{len(task_vocab)}"
                )
            for v, o in zip(task_vocab, output):
                results[f"{metric_n[:-len('/None')]}/{v}"] = o
        else:
            results[metric_n] = output

    label_metrics = ["Accuracy"]
    for metric_n in label_metrics:
        results[metric_n] = METRIC_FNS[metric_n.split("/")[0]](Y, probs.argmax(axis=1))

    return results


def eval_binary_classification(Y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    results = {}
    probs_metrics = ["AUROC", "AUPRC", "NLL"]
    label_metrics = ["Accuracy"]

    for metric_n in probs_metrics:
        results[metric_n] = METRIC_FNS[metric_n.split("/")[0]](Y, probs[:, 1])

    for metric_n in label_metrics:
        results[metric_n] = METRIC_FNS[metric_n.split("/")[0]](Y, probs.argmax(axis=1))

    return results


def train_sklearn_pipeline(cfg: SklearnConfig):
    print(f"Saving config to {cfg.save_dir / 'config.yaml'}")
    cfg.save_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(cfg, cfg.save_dir / "config.yaml")

    ESD = Dataset.load(cfg.dataset_dir)

    task_dfs = add_tasks_from(ESD.config.save_dir / "task_dfs")
    task_df = task_dfs[cfg.task_df_name]

    task_type, normalized_label = PytorchDataset.normalize_task(
        pl.col(cfg.finetuning_task_label), task_df.schema[cfg.finetuning_task_label]
    )

    match task_type:
        case "binary_classification":
            task_vocab = [False, True]
        case "multi_class_classification":
            task_vocab = list(
                range(task_df.select(pl.col(cfg.finetuning_task_label).max()).collect().item() + 1)
            )
        case _:
            raise ValueError(f"Task type {task_type} not supported!")

    # TODO(mmd): Window sizes may violate start_time constraints in task dfs!

    print(f"Loading representations for {', '.join(cfg.feature_selector.window_sizes)}")
    task_df = task_df.select(
        "subject_id",
        pl.col("end_time").alias("timestamp"),
        normalized_label.alias(cfg.finetuning_task_label),
    )

    subjects_included = {}

    if cfg.train_subset_size not in (None, "FULL"):
        subject_ids = list(ESD.split_subjects["train"])
        prng = np.random.default_rng(cfg.seed)
        match cfg.train_subset_size:
            case int() as n_subjects if n_subjects > 1:
                subject_ids = prng.choice(subject_ids, size=n_subjects, replace=False)
            case float() as frac if 0 < frac < 1:
                subject_ids = prng.choice(subject_ids, size=int(frac * len(subject_ids)), replace=False)
            case _:
                raise ValueError(
                    f"train_subset_size must be either 'FULL', `None`, an int > 1, or a float in (0, 1); "
                    f"got {cfg.train_subset_size}"
                )
        subjects_included["train"] = [int(e) for e in subject_ids]
        subjects_included["tuning"] = [int(e) for e in ESD.split_subjects["tuning"]]
        subjects_included["held_out"] = [int(e) for e in ESD.split_subjects["held_out"]]

        all_subject_ids = list(
            set(subjects_included["train"])
            | set(subjects_included["tuning"])
            | set(subjects_included["held_out"])
        )
        task_df = task_df.filter(pl.col("subject_id").is_in(all_subject_ids))

    with open(cfg.save_dir / "subjects.json", mode="w") as f:
        json.dump(subjects_included, f)

    flat_reps = load_flat_rep(ESD, window_sizes=cfg.feature_selector.window_sizes, join_df=task_df)
    Xs_and_Ys = {}
    for split in ("train", "tuning", "held_out"):
        st = datetime.now()
        print(f"Loading dataset for {split}")
        df = flat_reps[split].collect()

        X = df.drop(["subject_id", "timestamp", cfg.finetuning_task_label])
        Y = df[cfg.finetuning_task_label].to_numpy()
        print(f"Done with {split} dataset with X of shape {X.shape} " f"(elapsed: {datetime.now() - st})")
        Xs_and_Ys[split] = (X, Y)

    print("Initializing model!")
    model = cfg.get_model(dataset=ESD)

    print("Fitting model!")
    model.fit(*Xs_and_Ys["train"])
    print(f"Saving model to {cfg.save_dir}")
    with open(cfg.save_dir / "model.pkl", mode="wb") as f:
        pickle.dump(model, f)

    print("Evaluating model!")
    all_metrics = {}
    for split in ("tuning", "held_out"):
        X, Y = Xs_and_Ys[split]
        probs = model.predict_proba(X)

        match task_type:
            case "binary_classification":
                all_metrics[split] = eval_binary_classification(Y, probs)
            case "multi_class_classification":
                all_metrics[split] = eval_multi_class_classification(Y, probs, task_vocab=task_vocab)

    with open(cfg.save_dir / "final_metrics.json", mode="w") as f:
        json.dump(all_metrics, f)

    return model, Xs_and_Ys, all_metrics


@task_wrapper
def wandb_train_sklearn(cfg: SklearnConfig):
    run = wandb.init(
        **cfg.wandb_logger_kwargs,
        config=dataclasses.asdict(cfg),
    )

    _, _, metrics = train_sklearn_pipeline(cfg)
    run.log(metrics)

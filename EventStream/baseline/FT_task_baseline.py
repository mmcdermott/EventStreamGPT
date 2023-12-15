"""Utilities for collecting baseline performance of fine-tuning tasks defined over ESGPT datasets."""

import dataclasses
import inspect
import json
import pickle
import warnings
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import omegaconf
import polars as pl
import polars.selectors as cs
import wandb
from hydra.core.config_store import ConfigStore
from loguru import logger
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
from ..data.pytorch_dataset import ConstructorPytorchDataset
from ..tasks.profile import add_tasks_from
from ..utils import task_wrapper

pl.enable_string_cache()


def load_flat_rep(
    ESD: Dataset,
    window_sizes: list[str],
    feature_inclusion_frequency: float | dict[str, float] | None = None,
    include_only_measurements: set[str] | None = None,
    do_update_if_missing: bool = True,
    task_df_name: str | None = None,
    do_cache_filtered_task: bool = True,
    subjects_included: dict[str, set[int]] | None = None,
) -> dict[str, pl.LazyFrame]:
    """Loads a set of flat representations from a passed dataset that satisfy the given constraints.

    Args:
        ESD: The dataset for which the flat representations should be loaded.
        window_sizes: Beyond writing out a raw, per-event flattened representation, the dataset also has
            the capability to summarize these flattened representations over the historical windows
            specified in this argument. These are strings specifying time deltas, using this syntax:
            `link_`. Each window size will be summarized to a separate directory, and will share the same
            subject file split as is used in the raw representation files.
        feature_inclusion_frequency: The base feature inclusion frequency that should be used to dictate
            what features can be included in the flat representation. It can either be a float, in which
            case it applies across all measurements, or `None`, in which case no filtering is applied, or
            a dictionary from measurement type to a float dictating a per-measurement-type inclusion
            cutoff.
        include_only_measurements: Measurement types can also be filtered out wholesale from both
            representations. If this list is not None, only these measurements will be included.
        do_update_if_missing: If `True`, then if any window sizes or features are missing, the function will
            try to update the stored flat representations to reflect these. If `False`, if information is
            missing, it will raise a `FileNotFoundError` instead.
        task_df_name: If specified, the flat representations loaded will be (inner) joined against the task
            dataframe of this name on the columns ``"subject_id"`` and ``"end_time"`` (which will be renamed
            to ``"timestamp"``). This is to avoid needing to load the full dataset in flattened form into
            memory. This is also used as a cache key; if a pre-filtered dataset is written to disk at a
            specified path for this task, then the data will be loaded from there, rather than from the base
            dataset.
        do_cache_filtered_task: If `True`, the flat representations will, after being filtered to just the
            relevant rows for the task, be cached to disk for faster re-use.
        subjects_included: A dictionary by split of the subjects to include in the task. Omitted splits are
            used wholesale.

    Raises:
        FileNotFoundError: If `do_update_if_missing` is `False` and the requested historical representations
            are not already written to disk.
    """
    if subjects_included is None:
        subjects_included = {}

    flat_dir = ESD.config.save_dir / "flat_reps"

    feature_inclusion_frequency, include_only_measurements = ESD._resolve_flat_rep_cache_params(
        feature_inclusion_frequency, include_only_measurements
    )

    cache_kwargs = dict(
        feature_inclusion_frequency=feature_inclusion_frequency,
        window_sizes=window_sizes,
        include_only_measurements=include_only_measurements,
        do_overwrite=False,
        do_update=True,
    )

    params_fp = flat_dir / "params.json"
    if not params_fp.is_file():
        if not do_update_if_missing:
            raise FileNotFoundError("Flat representation files haven't been written!")
        else:
            ESD.cache_flat_representation(**cache_kwargs)

    with open(params_fp) as f:
        params = json.load(f)

    if task_df_name is not None:
        task_dir = ESD.config.save_dir / "task_dfs"
        join_df = pl.scan_parquet(task_dir / f"{task_df_name}.parquet").rename({"end_time": "timestamp"})

    needs_more_measurements = not set(include_only_measurements).issubset(params["include_only_measurements"])
    needs_more_features = params["feature_inclusion_frequency"] is not None and (
        (feature_inclusion_frequency is None)
        or any(
            params["feature_inclusion_frequency"].get(m, float("inf")) > m_freq
            for m, m_freq in feature_inclusion_frequency.items()
        )
    )
    needs_more_windows = False
    for window_size in window_sizes:
        if not (flat_dir / "over_history" / "train" / window_size).is_dir():
            needs_more_windows = True

    if needs_more_measurements or needs_more_features or needs_more_windows:
        if do_update_if_missing:
            ESD.cache_flat_representation(**cache_kwargs)
            with open(params_fp) as f:
                params = json.load(f)
        else:
            raise FileNotFoundError(
                f"Missing files! Needs measurements: {needs_more_measurements}; Needs features: "
                f"{needs_more_features}; Needs windows: {needs_more_windows}."
            )

    allowed_features = ESD._get_flat_rep_feature_cols(
        feature_inclusion_frequency=feature_inclusion_frequency,
        window_sizes=window_sizes,
        include_only_measurements=include_only_measurements,
    )

    join_keys = ["subject_id", "timestamp"]

    by_split = {}
    for sp, all_sp_subjects in ESD.split_subjects.items():
        if task_df_name is not None:
            sp_join_df = join_df.filter(pl.col("subject_id").is_in(list(all_sp_subjects)))

        static_df = pl.scan_parquet(flat_dir / "static" / sp / "*.parquet")
        if task_df_name is not None:
            static_df = static_df.join(sp_join_df.select("subject_id").unique(), on="subject_id", how="inner")

        dfs = []
        for window_size in window_sizes:
            window_features = [c for c in allowed_features if c.startswith(f"{window_size}/")]

            if task_df_name is not None:
                task_window_dir = flat_dir / "task_histories" / task_df_name / sp / window_size

            window_dir = flat_dir / "over_history" / sp / window_size
            window_dfs = []
            for fp in window_dir.glob("*.parquet"):
                subjects_idx = int(fp.stem)
                subjects = params["subject_chunks_by_split"][sp][subjects_idx]

                if task_df_name is not None:
                    fn = fp.parts[-1]
                    cached_fp = task_window_dir / fn
                    if cached_fp.is_file():
                        df = pl.scan_parquet(cached_fp).select("subject_id", "timestamp", *window_features)
                        if subjects_included.get(sp, None) is not None:
                            subjects = list(set(subjects).intersection(subjects_included[sp]))
                            df = df.filter(pl.col("subject_id").is_in(subjects))
                        window_dfs.append(df)
                        continue

                df = pl.scan_parquet(fp)
                if task_df_name is not None:
                    filter_join_df = sp_join_df.select(join_keys).filter(pl.col("subject_id").is_in(subjects))

                    df = df.join(filter_join_df, on=join_keys, how="inner")

                    if do_cache_filtered_task:
                        cached_fp.parent.mkdir(exist_ok=True, parents=True)
                        df.collect().write_parquet(cached_fp, use_pyarrow=True)

                df = df.select("subject_id", "timestamp", *window_features)
                if subjects_included.get(sp, None) is not None:
                    subjects = list(set(subjects).intersection(subjects_included[sp]))
                    df = df.filter(pl.col("subject_id").is_in(subjects))

                window_dfs.append(df)

            dfs.append(pl.concat(window_dfs, how="vertical"))

        joined_df = dfs[0]
        for jdf in dfs[1:]:
            joined_df = joined_df.join(jdf, on=join_keys, how="inner")

        # Add in the labels
        if task_df_name is not None:
            joined_df = joined_df.join(sp_join_df, on=join_keys, how="inner")
            extra_cols = [c for c in sp_join_df.columns if c not in join_keys]
        else:
            extra_cols = []

        # Add in the static data
        by_split[sp] = (
            joined_df.join(static_df, on="subject_id", how="left")
            .with_columns(cs.ends_with("count").fill_null(0))
            .select(*join_keys, *extra_cols, *allowed_features)
        )

    return by_split


class ESDFlatFeatureLoader:
    """A flat feature pre-processor in line with scikit-learn's APIs.

    This can dynamically apply window size, feature inclusion frequency, measurement restrictions, and mean
    variable conversions to flat feature sets. All window sizes indicated in this featurizer must be included
    in the passed dataframes.
    """

    def __init__(
        self,
        ESD: Dataset,
        window_sizes: list[str],
        feature_inclusion_frequency: float | dict[str, float] | None = None,
        include_only_measurements: set[str] | None = None,
        convert_to_mean_var: bool = True,
        **kwargs,
    ):
        self.ESD = ESD
        if type(window_sizes) is not list:
            raise ValueError(f"window_sizes must be a list; got {type(window_sizes)}: {window_sizes}")
        self.window_sizes = window_sizes
        self.feature_inclusion_frequency = feature_inclusion_frequency
        self.include_only_measurements = include_only_measurements
        self.convert_to_mean_var = convert_to_mean_var

    def set_params(
        self,
        ESD: Dataset | None = None,
        window_sizes: list[str] | None = None,
        feature_inclusion_frequency: float | dict[str, float] | None = None,
        include_only_measurements: set[str] | None = None,
        convert_to_mean_var: bool | None = None,
    ):
        if ESD is not None:
            self.ESD = ESD
        if window_sizes is not None:
            self.window_sizes = window_sizes
        if feature_inclusion_frequency is not None:
            self.feature_inclusion_frequency = feature_inclusion_frequency
        if include_only_measurements is not None:
            self.include_only_measurements = include_only_measurements
        if convert_to_mean_var is not None:
            self.convert_to_mean_var = convert_to_mean_var

    def fit(self, flat_rep_df: pl.DataFrame, _) -> "ESDFlatFeatureLoader":
        self.feature_columns = self.ESD._get_flat_rep_feature_cols(
            feature_inclusion_frequency=self.feature_inclusion_frequency,
            window_sizes=self.window_sizes,
            include_only_measurements=self.include_only_measurements,
        )

        want_cols = set(self.feature_columns)
        have_cols = set(flat_rep_df.columns)
        if not want_cols.issubset(have_cols):
            missing_cols = list(want_cols - have_cols)
            raise ValueError(
                f"Missing {len(missing_cols)} required columns:\n"
                f"  {', '.join(missing_cols[:5])}{'...' if len(missing_cols) > 5 else ''}."
                f"Have columns:\n{', '.join(flat_rep_df.columns)}"
            )

        flat_rep_df = flat_rep_df.select(self.feature_columns)
        non_null_cols = [s.name for s in flat_rep_df if s.null_count() != flat_rep_df.height]

        self.feature_columns = non_null_cols

        return self

    def transform(self, flat_rep_df: pl.DataFrame) -> np.ndarray:
        out_df = flat_rep_df.lazy().select(self.feature_columns)

        if self.convert_to_mean_var:

            def last_part(s: str) -> str:
                return "/".join(s.split("/")[:-1])

            has_values_cols = {last_part(c) for c in self.feature_columns if c.endswith("has_values_count")}
            to_conv_mean_cols = []
            to_conv_var_cols = []

            for col in has_values_cols:
                if f"{col}/sum" in self.feature_columns:
                    to_conv_mean_cols.append(col)
                    if f"{col}/sum_sqd" in self.feature_columns:
                        to_conv_var_cols.append(col)

            out_df = (
                out_df.with_columns(
                    *[
                        (pl.col(f"{c}/sum") / pl.col(f"{c}/has_values_count")).alias(f"{c}/mean")
                        for c in to_conv_mean_cols
                    ],
                )
                .with_columns(
                    *[
                        (
                            (pl.col(f"{c}/sum_sqd") / pl.col(f"{c}/has_values_count"))
                            - (pl.col(f"{c}/mean") ** 2)
                        ).alias(f"{c}/var")
                        for c in to_conv_var_cols
                    ],
                )
                .drop(
                    *[f"{c}/sum" for c in to_conv_mean_cols],
                    *[f"{c}/sum_sqd" for c in to_conv_var_cols],
                    *[f"{c}/has_values_count" for c in has_values_cols],
                )
            )

        out_df = out_df

        return out_df.collect().to_numpy()


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


config_store = ConfigStore.instance()
config_store.store(name="sklearn_config", node=SklearnConfig)
config_store.store(group="scaling", name="min_max_scaler", node=MinMaxScalerConfig)
config_store.store(group="scaling", name="standard_scaler", node=StandardScalerConfig)
config_store.store(group="imputation", name="simple_imputer", node=SimpleImputerConfig)
config_store.store(group="imputation", name="knn_imputer", node=KNNImputerConfig)
config_store.store(group="dim_reduce", name="nmf", node=NMFConfig)
config_store.store(group="dim_reduce", name="pca", node=PCAConfig)
config_store.store(group="dim_reduce", name="select_k_best", node=SelectKBestConfig)
config_store.store(group="feature_selector", name="esd_flat_feature_loader", node=ESDFlatFeatureLoaderConfig)
config_store.store(group="model", name="random_forest_classifier", node=RandomForestClassifierConfig)

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
    logger.info(f"Saving config to {cfg.save_dir / 'config.yaml'}")
    cfg.save_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(cfg, cfg.save_dir / "config.yaml")

    ESD = Dataset.load(cfg.dataset_dir)

    task_dfs = add_tasks_from(ESD.config.save_dir / "task_dfs")
    task_df = task_dfs[cfg.task_df_name]

    task_type, normalized_label = ConstructorPytorchDataset.normalize_task(
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

    logger.info(f"Loading representations for {', '.join(cfg.feature_selector.window_sizes)}")
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

    with open(cfg.save_dir / "subjects.json", mode="w") as f:
        json.dump(subjects_included, f)

    flat_reps = load_flat_rep(
        ESD,
        window_sizes=cfg.feature_selector.window_sizes,
        task_df_name=cfg.task_df_name,
        subjects_included=subjects_included,
    )
    Xs_and_Ys = {}
    for split in ("train", "tuning", "held_out"):
        st = datetime.now()
        logger.info(f"Loading dataset for {split}")
        df = flat_reps[split].with_columns(normalized_label.alias(cfg.finetuning_task_label)).collect()

        X = df.drop(["subject_id", "timestamp", cfg.finetuning_task_label])
        Y = df[cfg.finetuning_task_label].to_numpy()
        logger.info(
            f"Done with {split} dataset with X of shape {X.shape} " f"(elapsed: {datetime.now() - st})"
        )
        Xs_and_Ys[split] = (X, Y)

    logger.info("Initializing model!")
    model = cfg.get_model(dataset=ESD)

    logger.info("Fitting model!")
    model.fit(*Xs_and_Ys["train"])
    logger.info(f"Saving model to {cfg.save_dir}")
    with open(cfg.save_dir / "model.pkl", mode="wb") as f:
        pickle.dump(model, f)

    logger.info("Evaluating model!")
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

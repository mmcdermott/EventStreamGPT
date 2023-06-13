#!/usr/bin/env python
"""Builds a dataset given a hydra config file."""

try:
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import inflect
from omegaconf import DictConfig, OmegaConf

from EventStream.data.config import (
    DatasetConfig,
    DatasetSchema,
    InputDFSchema,
    MeasurementConfig,
)
from EventStream.data.dataset_polars import Dataset, Query
from EventStream.data.types import (
    DataModality,
    InputDataType,
    InputDFType,
    TemporalityType,
)

inflect = inflect.engine()


def add_to_container(key: str, val: Any, cont: dict[str, Any]):
    """Adds key to container, unless it is already present in that container with a different value.

    If `key` is in `cont` with value `val`, prints a warning. If it is in `cont` with a different value,
    raises a `ValueError`. Otherwise adds `key` to `cont` with value `val`. Returns `None`

    Args:
        key: The key to add to the container `cont`.
        val: The value to associate with key `key` in container `cont`.
        cont: The container in which to store `key` and `value`.

    Raises:
        ValueError: If `key` is in `cont` with value not equal to `val`.

    Examples:
        >>> cont = {'foo': "bar"}
        >>> add_to_container('biz', 3, cont)
        >>> cont
        {'foo': 'bar', 'biz': 3}
        >>> add_to_container('biz', 3, cont)
        WARNING: biz is specified twice with value 3.
        >>> cont
        {'foo': 'bar', 'biz': 3}
        >>> add_to_container('foo', 3, cont)
        Traceback (most recent call last):
            ...
        ValueError: foo is specified twice (3 v. bar)
    """

    if key in cont:
        if cont[key] == val:
            print(f"WARNING: {key} is specified twice with value {val}.")
        else:
            raise ValueError(f"{key} is specified twice ({val} v. {cont[key]})")
    else:
        cont[key] = val


@hydra.main(version_base=None, config_path="../configs", config_name="dataset_base")
def main(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg, _convert_="all")

    cfg_fp = Path(cfg["save_dir"]) / "hydra_config.yaml"
    cfg_fp.parent.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(cfg, cfg_fp)

    # 1. Build measurement_configs and track input schemas
    subject_id_col = cfg.pop("subject_id_col")
    measurements_by_temporality = cfg.pop("measurements")

    static_sources = defaultdict(dict)
    dynamic_sources = defaultdict(dict)
    measurement_configs = {}

    if TemporalityType.FUNCTIONAL_TIME_DEPENDENT in measurements_by_temporality:
        time_dep_measurements = measurements_by_temporality.pop(TemporalityType.FUNCTIONAL_TIME_DEPENDENT)
    else:
        time_dep_measurements = {}

    for temporality, measurements_by_modality in measurements_by_temporality.items():
        schema_source = static_sources if temporality == TemporalityType.STATIC else dynamic_sources
        for modality, measurements_by_source in measurements_by_modality.items():
            if not measurements_by_source:
                continue
            for source_name, measurements in measurements_by_source.items():
                data_schema = schema_source[source_name]

                if type(measurements) is str:
                    measurements = [measurements]
                for m in measurements:
                    measurement_config_kwargs = {
                        "name": m,
                        "temporality": temporality,
                        "modality": modality,
                    }
                    if type(m) is dict:
                        m_dict = m
                        if m.get("values_column", None):
                            values_column = m_dict.pop("values_column")
                            m = [m_dict.pop("name"), values_column]
                        else:
                            m = m_dict.pop("name")
                        measurement_config_kwargs.update(m_dict)

                    match m, modality:
                        case str(), DataModality.UNIVARIATE_REGRESSION:
                            add_to_container(m, InputDataType.FLOAT, data_schema)
                        case [str() as m, str() as v], DataModality.MULTIVARIATE_REGRESSION:
                            add_to_container(m, InputDataType.CATEGORICAL, data_schema)
                            add_to_container(v, InputDataType.FLOAT, data_schema)
                            measurement_config_kwargs["values_column"] = v
                            measurement_config_kwargs["name"] = m
                        case str(), DataModality.SINGLE_LABEL_CLASSIFICATION:
                            add_to_container(m, InputDataType.CATEGORICAL, data_schema)
                        case str(), DataModality.MULTI_LABEL_CLASSIFICATION:
                            add_to_container(m, InputDataType.CATEGORICAL, data_schema)
                        case _:
                            raise ValueError(f"{m}, {modality} invalid! Must be in {DataModality.values()}!")

                    if m in measurement_configs:
                        if measurement_configs[m].to_dict() != measurement_config_kwargs:
                            raise ValueError(f"{m} differs across input sources!")
                    else:
                        measurement_configs[m] = MeasurementConfig(**measurement_config_kwargs)

    if len(static_sources) > 1:
        raise NotImplementedError(
            f"Currently, only 1 static source can be specified -- you have {static_sources}"
        )

    static_key = list(static_sources.keys())[0]
    static_col_schema = static_sources[static_key]

    for m, config in time_dep_measurements.items():
        if type(m) is not str:
            raise ValueError(f"{m} must be a string for time-dep measurement!")
        functor_class = config.pop("functor")
        functor_kwargs = config.pop("kwargs", {})

        measurement_config_kwargs = {
            "name": m,
            "temporality": TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
            "functor": MeasurementConfig.FUNCTORS[functor_class](**functor_kwargs),
        }

        necessary_static_measurements = config.pop("necessary_static_measurements", {})
        if necessary_static_measurements:
            for in_col, in_fmt in necessary_static_measurements.items():
                schema_key = in_col
                schema_val = (in_col, in_fmt)
                if in_col in static_col_schema and static_col_schema[schema_key] != schema_val:
                    raise ValueError(
                        f"Schema Collision! {schema_key}, {schema_val} v. {static_col_schema[schema_key]}"
                    )

                static_col_schema[schema_key] = schema_val

        if m in measurement_configs and measurement_configs[m].to_dict() != measurement_config_kwargs:
            raise ValueError(f"{m} differs across input sources!")
        measurement_configs[m] = MeasurementConfig(**measurement_config_kwargs)

    # 1. Build DatasetSchema
    connection_uri = cfg.pop("connection_uri", None)
    cfg.pop("raw_data_dir", None)

    def build_schema(
        col_schema: dict[str, InputDataType],
        source_schema: dict[str, Any],
        schema_name: str,
        **extra_kwargs,
    ) -> InputDFSchema:
        input_schema_kwargs = {}

        if "query" in source_schema:
            if "input_df" in source_schema:
                raise ValueError(
                    f"Can't specify both query {source_schema['query']} "
                    f"and input_df {source_schema['input_df']} at once!"
                )
            match source_schema["query"]:
                case str() | list() as query:
                    if not connection_uri:
                        raise ValueError("If providing a query string, must provide a connection_uri!")
                    if type(query) is list:
                        query = tuple(query)
                    input_schema_kwargs["input_df"] = Query(query=query, connection_uri=connection_uri)
                case dict() as query_kwargs:
                    if "connection_uri" not in query_kwargs:
                        query_kwargs["connection_uri"] = connection_uri
                    input_schema_kwargs["input_df"] = Query(**query_kwargs)
                case _:
                    raise ValueError(f"Cannot parse query {source_schema['query']}!")
        elif "input_df" in source_schema:
            input_schema_kwargs["input_df"] = source_schema["input_df"]
        else:
            raise ValueError("Must specify either a query or an input dataframe!")

        for param in (
            "start_ts_col",
            "end_ts_col",
            "ts_col",
            "event_type",
            "start_ts_format",
            "end_ts_format",
            "ts_format",
        ):
            if param in source_schema:
                input_schema_kwargs[param] = source_schema[param]

        if source_schema.get("start_ts_col", None):
            input_schema_kwargs["type"] = InputDFType.RANGE
        elif source_schema.get("ts_col", None):
            input_schema_kwargs["type"] = InputDFType.EVENT
        else:
            input_schema_kwargs["type"] = InputDFType.STATIC

        if input_schema_kwargs["type"] != InputDFType.STATIC and "event_type" not in input_schema_kwargs:
            if not inflect.singular_noun(schema_name):
                event_type = schema_name
            else:
                event_type = inflect.singular_noun(schema_name)
            input_schema_kwargs["event_type"] = event_type.upper()

        cols_covered = []
        any_schemas_present = False
        for n, cols_n in (
            ("start_data_schema", "start_columns"),
            ("end_data_schema", "end_columns"),
            ("data_schema", "columns"),
        ):
            if cols_n not in source_schema:
                continue
            cols = source_schema[cols_n]
            data_schema = {}

            match source_schema.get("event_type", None):
                case list():
                    for et in source_schema["event_type"]:
                        if et.startswith("COL:"):
                            event_type_col = et[len("COL:") :]
                            data_schema[event_type_col] = (event_type_col, InputDataType.CATEGORICAL)
                case str() as et if et.startswith("COL:"):
                    event_type_col = et[len("COL:") :]
                    data_schema[event_type_col] = (event_type_col, InputDataType.CATEGORICAL)

            if type(cols) is dict:
                cols = [list(t) for t in cols.items()]

            for col in cols:
                match col:
                    case [str() as in_name, str() as out_name] if out_name in col_schema:
                        schema_key = in_name
                        schema_val = (out_name, col_schema[out_name])
                    case str() as col_name if col_name in col_schema:
                        schema_key = col_name
                        schema_val = (col_name, col_schema[col_name])
                    case _:
                        raise ValueError(f"{col} unprocessable! Col schema: {col_schema}")

                cols_covered.append(schema_val[0])
                add_to_container(schema_key, schema_val, data_schema)
            input_schema_kwargs[n] = data_schema
            any_schemas_present = True

        if not any_schemas_present and (len(col_schema) > len(cols_covered)):
            input_schema_kwargs["data_schema"] = {}

        for col, dt in col_schema.items():
            if col in cols_covered:
                continue

            for schema in ("start_data_schema", "end_data_schema", "data_schema"):
                if schema in input_schema_kwargs:
                    input_schema_kwargs[schema][col] = dt

        must_have = source_schema.get("must_have", None)
        match must_have:
            case None:
                pass
            case list():
                input_schema_kwargs["must_have"] = must_have
            case dict() as must_have_dict:
                must_have = []
                for k, v in must_have_dict.items():
                    match v:
                        case True:
                            must_have.append(k)
                        case list():
                            must_have.append((k, v))
                        case _:
                            raise ValueError(f"{v} invalid for `must_have`")
                input_schema_kwargs["must_have"] = must_have

        return InputDFSchema(**input_schema_kwargs, **extra_kwargs)

    inputs = cfg.pop("inputs")
    dataset_schema = DatasetSchema(
        static=build_schema(
            col_schema=static_col_schema,
            source_schema=inputs.pop(static_key),
            subject_id_col=subject_id_col,
            schema_name=static_key,
        ),
        dynamic=[
            build_schema(
                col_schema=dynamic_sources.get(dynamic_key, {}),
                source_schema=source_schema,
                schema_name=dynamic_key,
            )
            for dynamic_key, source_schema in inputs.items()
        ],
    )

    # 2. Build Config
    split = cfg.pop("split", (0.8, 0.1))
    seed = cfg.pop("seed", 1)
    do_overwrite = cfg.pop("do_overwrite", False)
    cfg.pop("cohort_name")
    DL_chunk_size = cfg.pop("DL_chunk_size", 20000)

    valid_config_kwargs = {f.name for f in dataclasses.fields(DatasetConfig)}
    extra_kwargs = {k: v for k, v in cfg.items() if k not in valid_config_kwargs}
    config_kwargs = {k: v for k, v in cfg.items() if k in valid_config_kwargs}

    if extra_kwargs:
        print(f"Omitting {extra_kwargs} from config!")

    config = DatasetConfig(measurement_configs=measurement_configs, **config_kwargs)

    if config.save_dir is not None:
        dataset_schema.to_json_file(config.save_dir / "input_schema.json", do_overwrite=do_overwrite)

    ESD = Dataset(config=config, input_schema=dataset_schema)
    ESD.split(split, seed=seed)
    ESD.preprocess()
    ESD.save(do_overwrite=do_overwrite)
    ESD.cache_deep_learning_representation(DL_chunk_size, do_overwrite=do_overwrite)


if __name__ == "__main__":
    main()

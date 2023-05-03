from __future__ import annotations

import abc
from typing import Any, Dict

import pandas as pd
import polars as pl

from .types import DataModality


class TimeDependentFunctor(abc.ABC):
    """An abstract base class defining the interface necessary for specifying time-dependent
    functions."""

    OUTPUT_MODALITY = DataModality.DROPPED

    def __init__(self, **fn_params):
        # Default to_dict/from_dict will only work if functions store all __init__ input params as class
        # member variables, and use those to compute the function values in __call__...
        for k, val in fn_params.items():
            setattr(self, k, val)

        self.link_static_cols = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "params": {k: v for k, v in vars(self).items() if k != "link_static_cols"},
        }

    @classmethod
    def from_dict(cls, in_dict: dict[str, Any]) -> TimeDependentFunctor:
        return cls(**in_dict["params"])

    def __eq__(self, other: TimeDependentFunctor) -> bool:
        return self.to_dict() == other.to_dict()


class AgeFunctor(TimeDependentFunctor):
    """An example functor that returns the age of the subject when the event occurred."""

    OUTPUT_MODALITY = DataModality.UNIVARIATE_REGRESSION

    def __init__(self, dob_col: str):
        self.dob_col = dob_col
        self.link_static_cols = [dob_col]

    def pl_expr(self) -> pl.Expression:
        return (
            (pl.col("timestamp") - pl.col(self.dob_col)).dt.nanoseconds()
            / 1e9
            / 60
            / 60
            / 24
            / 365.25
        )


class TimeOfDayFunctor(TimeDependentFunctor):
    """An example functor that returns the time-of-day in 4 categories when the event occurred."""

    OUTPUT_MODALITY = DataModality.SINGLE_LABEL_CLASSIFICATION

    def pl_expr(self) -> pl.Expression:
        return (
            pl.when(pl.col("timestamp").dt.hour() < 6)
            .then("EARLY_AM")
            .when(pl.col("timestamp").dt.hour() < 12)
            .then("AM")
            .when(pl.col("timestamp").dt.hour() < 21)
            .then("PM")
            .otherwise("LATE_PM")
        )

# This file contains the abstract base class for polars pre-processors. It is just used to define the
# interface expected by the data preprocessing pipeline.

from abc import ABC, abstractmethod
from typing import Dict

import polars as pl


class Preprocessor(ABC):
    @classmethod
    @abstractmethod
    def params_schema(cls) -> Dict[str, pl.DataType]:
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def fit_from_polars(self, column: pl.Expr) -> pl.Expr:
        raise NotImplementedError("Subclass must implement abstract method")

    @classmethod
    @abstractmethod
    def predict_from_polars(cls, column: pl.Expr, model_column: pl.Expr) -> pl.Expr:
        raise NotImplementedError("Subclass must implement abstract method")

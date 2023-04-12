import polars as pl

from typing import Dict

from .preprocessor import Preprocessor

class StandardScaler(Preprocessor):
    """
    This normalizer computes the mean and standard deviation of the data, and normalizes it to have zero mean
    and unit variance on the output.
    """
    @classmethod
    def params_schema(cls) -> Dict[str, pl.DataType]: return {'mean_': pl.Float64, 'std_': pl.Float64}

    def fit_from_polars(self, column: pl.Expr) -> pl.Expr:
        return pl.struct([column.mean().alias('mean_'), column.std().alias('std_')])

    @classmethod
    def predict_from_polars(cls, column: pl.Expr, model_column: pl.Expr) -> pl.Expr:
        return (column - model_column.struct.field('mean_')) / model_column.struct.field('std_')

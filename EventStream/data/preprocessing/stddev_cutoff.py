import polars as pl

from typing import Dict

from .preprocessor import Preprocessor

class StddevCutoffOutlierDetector(Preprocessor):
    """
    This normalizer computes the mean and standard deviation of the data, and normalizes it to have zero mean
    and unit variance on the output.
    """

    def __init__(self, stddev_cutoff: float = 5.0):
        self.stddev_cutoff = stddev_cutoff

    @classmethod
    def params_schema(cls) -> Dict[str, pl.DataType]:
        return {'thresh_large_': pl.Float64, 'thresh_small_': pl.Float64}

    def fit_from_polars(self, column: pl.Expr) -> pl.Expr:
        mean, std = column.mean(), column.std()
        return pl.struct([
            (mean + self.stddev_cutoff*std).alias('thresh_large_'),
            (mean - self.stddev_cutoff*std).alias('thresh_small_')
        ])

    @classmethod
    def predict_from_polars(cls, column: pl.Expr, model_column: pl.Expr) -> pl.Expr:
        """
        This provides a polars expression capable of producing the predictions of a fitted
        StddevCutoffOutlierDetector instance. It can be used either on a raw column or within a groupby
        expression, and will output a polars boolean column where True indicates an outlier.
        """

        return (
            (column > model_column.struct.field('thresh_large_')) |
            (column < model_column.struct.field('thresh_small_'))
        ).alias('is_outlier')

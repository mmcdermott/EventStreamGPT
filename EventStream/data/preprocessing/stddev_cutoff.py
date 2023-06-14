"""Pre-processor that filters data to contain only values within a certain number of standard deviations from
the mean."""

import polars as pl

from .preprocessor import Preprocessor


class StddevCutoffOutlierDetector(Preprocessor):
    """Filters out data elements that are outside a specifiable number of standard deviations of the mean.

    This is a concrete implementation of the Preprocessor abstract class. It is a pre-processor that
    identifies outliers, here defined to be data points more than a specifiable number of standard deviations
    away from the mean. It is implemented as a Polars friendly pre-processor, meaning that it is implemented
    as a Polars expression that can be used in both a select and a groupby aggregation context.

    Attributes:
        stddev_cutoff: The number of standard deviations from the mean to use as the cutoff for identifying
            outliers. Defaults to 5.0.

    Examples:
        >>> import polars as pl
        >>> S = StddevCutoffOutlierDetector(stddev_cutoff=1.0)
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> params = S.fit_from_polars(pl.col("a")).alias("params")
        >>> df.select(params)["params"].to_list()
        [{'thresh_large_': 4.58113883008419, 'thresh_small_': 1.4188611699158102}]
        >>> outliers = S.predict_from_polars(pl.col("a"), params).alias("a_outliers")
        >>> df.select(outliers)["a_outliers"].to_list()
        [True, False, False, False, True]
    """

    def __init__(self, stddev_cutoff: float = 5.0):
        self.stddev_cutoff = stddev_cutoff

    @classmethod
    def params_schema(cls) -> dict[str, pl.DataType]:
        r"""Returns {"thresh_large\_": pl.Float64, "thresh_small\_": pl.Float64}."""
        return {"thresh_large_": pl.Float64, "thresh_small_": pl.Float64}

    def fit_from_polars(self, column: pl.Expr) -> pl.Expr:
        """Identify the configured large and small extreme value thresholds from the data in `column`.

        Arguments:
            column: The Polars expression for the column containing the raw data to be pre-processed.

        Returns:
            pl.Expr: A polars expression that will identify the mean plus or minus `self.stddev_cutoff` times
                the standard deviation of the data in `column`.
        """
        mean, std = column.mean(), column.std()
        return pl.struct(
            [
                (mean + self.stddev_cutoff * std).alias("thresh_large_"),
                (mean - self.stddev_cutoff * std).alias("thresh_small_"),
            ]
        )

    @classmethod
    def predict_from_polars(cls, column: pl.Expr, model_column: pl.Expr) -> pl.Expr:
        """Returns a column containing True if and only if the data in `column` is an outlier.

        Arguments:
            column: The Polars expression for the column containing the raw data to be checked for outliers.
            model_column: The Polars expression for the column containing the upper and lower thresholds for
                inliers.

        Returns:
            pl.Expr: A Polars expression that will return True if and only if the data in `column` is greater
                than the `"thresh_large"` field in the struct in `model_column` or less than the
                `"thresh_small"` field in the struct in `model_column`.
        """

        return (
            (column > model_column.struct.field("thresh_large_"))
            | (column < model_column.struct.field("thresh_small_"))
        ).alias("is_outlier")

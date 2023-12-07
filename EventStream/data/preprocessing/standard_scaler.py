"""Pre-processor that normalizes data to have zero mean and unit variance."""

import polars as pl

from .preprocessor import Preprocessor


class StandardScaler(Preprocessor):
    """Normalizes data to have zero mean and unit variance.

    This is a concrete implementation of the Preprocessor abstract class. It is a pre-processor that
    normalizes data to have zero mean and unit variance. It is implemented as a Polars friendly pre-processor,
    meaning that it is implemented as a Polars expression that can be used in both a select and a group_by
    aggregation context.

    Examples:
        >>> import polars as pl
        >>> S = StandardScaler()
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> params = S.fit_from_polars(pl.col("a")).alias("params")
        >>> df.select(params)["params"].to_list()
        [{'mean_': 3.0, 'std_': 1.5811388300841898}]
        >>> norm = S.predict_from_polars(pl.col("a"), params).alias("a_norm")
        >>> df.select(norm)["a_norm"].to_list()
        [-1.2649110640673518, -0.6324555320336759, 0.0, 0.6324555320336759, 1.2649110640673518]
    """

    @classmethod
    def params_schema(cls) -> dict[str, pl.DataType]:
        r"""Returns {"mean\_": pl.Float64, "std\_": pl.Float64}."""
        return {"mean_": pl.Float64, "std_": pl.Float64}

    def fit_from_polars(self, column: pl.Expr) -> pl.Expr:
        r"""Fit the mean and standard deviation of the data in `column`.

        Arguments:
            column: The Polars expression for the column containing the raw data to be pre-processed.

        Returns:
            pl.Expr: A polars expression for a struct column containing the mean and standard deviation of
                the data in `column` in fields named "mean\_" and "std\_" respectively.
        """
        return pl.struct([column.mean().alias("mean_"), column.std().alias("std_")])

    @classmethod
    def predict_from_polars(cls, column: pl.Expr, model_column: pl.Expr) -> pl.Expr:
        r"""Returns `(column - model_column.struct.field("mean_")) / model_column.struct.field("std_")`.

        Arguments:
            column: The Polars expression for the column containing the raw data to be centered and scaled.
            model_column: The Polars expression for a struct column containing "mean\_" and "std\_" fields.

        Returns:
            pl.Expr: `(column - model_column.struct.field("mean_")) / model_column.struct.field("std_")`
        """
        return (column - model_column.struct.field("mean_")) / model_column.struct.field("std_")

"""The base class for Polars friendly data pre-processors.

This file contains the abstract base class for polars pre-processors. It is just used to define the interface
expected by the data preprocessing pipeline. Subclasses (defined in other files in this module) contain actual
implementations of algorithms.
"""

from abc import ABC, abstractmethod

import polars as pl


class Preprocessor(ABC):
    """The base class for Polars friendly data pre-processors.

    This should be sub-classed by implementation classes for concrete implementations. Must define the schema
    of the output column produced by the pre-processor, the fit method which extracts those parameters from
    the raw data via a Polars expression, and the predict method which applies the pre-processing to a data
    column expression using another column containing the model parameters for that data element.
    """

    @classmethod
    @abstractmethod
    def params_schema(cls) -> dict[str, pl.DataType]:
        """The schema of the output column produced by the pre-processor.

        Must be implemented by a sub-class.

        Returns:
            dict[str, pl.DataType]:
                The schema of the output column produced by the pre-processor, as a mapping from field names
                to polars data types.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def fit_from_polars(self, column: pl.Expr) -> pl.Expr:
        """Fit the pre-processing model over the data contained in `column`.

        Performs the logic necessary to fit the pre-processing model over the data in the input column. As the
        input column is a polars expression, it does not contain materialized data, but rather just references
        a column operation that could be run to produce materialized data. The pre-processing logic must be
        consistent with that assumption. Must be implemented by a sub-class. The logic used in this method
        must be applicable for use in both a select and a group_by aggregation context.

        Arguments:
            column: The Polars expression for the column containing the raw data to be pre-processed.

        Returns:
            pl.Expr:
                The Polars expression for a column that would materialize the resulting pre-processing model
                parameters.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    @classmethod
    @abstractmethod
    def predict_from_polars(cls, column: pl.Expr, model_column: pl.Expr) -> pl.Expr:
        """Predicts for the data in `column` given the fit parameters in `model_column`.

        Performs the logic necessary to "predict" as defined by the implementing subclass over the data in the
        input column according to the parameters in the fit model column. As both input columns are polars
        expressions, they do not contain materialized data, but rather just references column operations that
        could be run to produce materialized data. The pre-processing logic must be consistent with that
        assumption. Must be implemented by a sub-class. The logic used in this method must be applicable for
        use in both a select and a group_by aggregation context.

        Arguments:
            column: The Polars expression for the column containing the raw data to be pre-processed.
            model_column: The Polars expression for the column containing the pre-processing model parameters.

        Returns:
            pl.Expr:
                The Polars expression for a column that would materialize the pre-processed outputs for the
                input data given the pre-processing model parameters.
        """
        raise NotImplementedError("Subclass must implement abstract method")

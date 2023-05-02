from __future__ import annotations

import abc, pandas as pd, polars as pl

from typing import Any, Dict

from .types import DataModality

class TimeDependentFunctor(abc.ABC):
    """An abstract base class defining the interface necessary for specifying time-dependent functions."""
    OUTPUT_MODALITY = DataModality.DROPPED

    def __init__(self, **fn_params):
        # Default to_dict/from_dict will only work if functions store all __init__ input params as class
        # member variables, and use those to compute the function values in __call__...
        for k, val in fn_params.items(): setattr(self, k, val)

        self.link_static_cols = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            'class': self.__class__.__name__, 'params': {
                k: v for k, v in vars(self).items() if k != 'link_static_cols'
            }
        }

    @classmethod
    def from_dict(cls, in_dict: Dict[str, Any]) -> 'TimeDependentFunctor': return cls(**in_dict['params'])

    @abc.abstractmethod
    def __call__(self, linked_time_df: pd.DataFrame) -> pd.Series:
        """
        Computes the value of the time-dependent function. Must be overwritten but retain the same signature.

        Args:
            `time` (`pd.Series`):
                A series wite multi-index consisting of (1) the event ID (unique, named `'event_id'`) and (2)
                the subject ID (not unique, named `'subject_id'`) with values corresponding to the timestamps
                of the events, in datetime format.
            `subject_df` (`pd.DataFrame`):
                A dataframe with index consisting of the subject ID (unique, named `'subject_id'`) and
                containing the values observed in the `Datasets` `subjects_df`.
        Returns:
            The result of the time-dependent function in question, as a `pd.Series` _with the same order as
            the input `time` series_.
        """
        raise NotImplementedError("Must overwrite in subclass!")

    def __eq__(self, other: 'TimeDependentFunctor') -> bool: return self.to_dict() == other.to_dict()

class AgeFunctor(TimeDependentFunctor):
    """An example functor that returns the age of the subject when the event occurred."""
    OUTPUT_MODALITY = DataModality.UNIVARIATE_REGRESSION

    def __init__(self, dob_col: str):
        self.dob_col = dob_col
        self.link_static_cols = [dob_col]

    def __call__(self, linked_time_df: pd.DataFrame) -> pd.Series:
        return (linked_time_df['timestamp'] - linked_time_df[self.dob_col]) / pd.to_timedelta(365.25, 'days')

    def pl_expr(self) -> pl.Expression:
        return (pl.col('timestamp') - pl.col(self.dob_col)).dt.nanoseconds() / 1e9 / 60 / 60 / 24 / 365.25

class TimeOfDayFunctor(TimeDependentFunctor):
    """An example functor that returns the time-of-day in 4 categories when the event occurred."""
    OUTPUT_MODALITY = DataModality.SINGLE_LABEL_CLASSIFICATION

    def __call__(self, linked_time_df: pd.DataFrame) -> pd.Series:
        return linked_time_df['timestamp'].apply(
            lambda dt: (
                'EARLY_AM' if dt.hour < 6 else
                'AM' if dt.hour < 12 else
                'PM' if dt.hour < 21 else
                'LATE_PM'
            )
        )

    def pl_expr(self) -> pl.Expression:
        return pl.when(
            pl.col('timestamp').dt.hour() < 6
        ).then('EARLY_AM').when(
            pl.col('timestamp').dt.hour() < 12
        ).then('AM').when(
            pl.col('timestamp').dt.hour() < 21
        ).then('PM').otherwise('LATE_PM')

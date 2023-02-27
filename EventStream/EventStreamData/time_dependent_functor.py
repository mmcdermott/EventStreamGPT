from __future__ import annotations

import abc, pandas as pd

from typing import Any, Dict

from .types import DataModality

class TimeDependentFunctor(abc.ABC):
    """An abstract base class defining the interface necessary for specifying time-dependent functions."""
    OUTPUT_MODALITY = DataModality.DROPPED

    @abc.abstractmethod
    def __init__(self, **fn_params):
        # Default to_dict/from_dict will only work if functions store all __init__ input params as class
        # member variables, and use those to compute the function values in __call__...
        for k, val in fn_params.items: setattr(self, k, val)

    def to_dict(self) -> Dict[str, Any]:
        return {'class': self.__class__.__name__, 'params': vars(self)}

    @classmethod
    def from_dict(cls, in_dict: Dict[str, Any]) -> 'TimeDependentFunctor': return cls(**in_dict['params'])

    @abc.abstractmethod
    def __call__(self, time: pd.Series, subject_df: pd.DataFrame) -> pd.Series:
        """
        Computes the value of the time-dependent function. Must be overwritten but retain the same signature.

        Args:
            `time` (`pd.Series`):
                A series wite multi-index consisting of (1) the event ID (unique, named `'event_id'`) and (2)
                the subject ID (not unique, named `'subject_id'`) with values corresponding to the timestamps
                of the events, in datetime format.
            `subject_df` (`pd.DataFrame`):
                A dataframe with index consisting of the subject ID (unique, named `'subject_id'`) and
                containing the values observed in the `EventStreamDatasets` `subjects_df`.
        Returns:
            The result of the time-dependent function in question, as a `pd.Series` _with the same order as
            the input `time` series_.
        """
        raise NotImplementedError(f"Must overwrite in subclass!")

    def __eq__(self, other: 'TimeDependentFunctor') -> bool: return self.to_dict() == other.to_dict()

class AgeFunctor(TimeDependentFunctor):
    """An example functor that returns the age of the subject when the event occurred."""
    OUTPUT_MODALITY = DataModality.UNIVARIATE_REGRESSION

    def __init__(self, dob_col: str):
        self.dob_col = dob_col

    def __call__(self, time: pd.Series, subject_df: pd.DataFrame) -> pd.Series:
        return (
            (time - subject_df.loc[time.index.get_level_values('subject_id'), self.dob_col].values) /
            pd.to_timedelta(365, 'days')
        )

class TimeOfDayFunctor(TimeDependentFunctor):
    """An example functor that returns the time-of-day in 4 categories when the event occurred."""
    OUTPUT_MODALITY = DataModality.SINGLE_LABEL_CLASSIFICATION

    def __init__(self): pass
    def __call__(self, time: pd.Series, _) -> pd.Series:
        return time.apply(
            lambda dt: (
                'EARLY_AM' if dt.hour < 6 else
                'AM' if dt.hour < 12 else
                'PM' if dt.hour < 21 else
                'LATE_PM'
            )
        )

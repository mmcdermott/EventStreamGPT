import copy, numpy as np, pandas as pd, warnings

from collections import Counter
from mixins import SeedableMixin, SaveableMixin, TimeableMixin
from sklearn.preprocessing import QuantileTransformer
from typing import Any, Dict, Hashable, List, Optional, Tuple, Sequence, Set, Union

from .expandable_df_dict import ExpandableDfDict
from .config import EventStreamDatasetConfig, MeasurementConfig
from .types import DataModality, TemporalityType, NumericDataModalitySubtype
from .vocabulary import Vocabulary

from ..utils import lt_count_or_proportion, flatten_dict, to_sklearn_np
from ..VarianceImpactOutlierDetector.variance_impact_outlier_detector import (
    VarianceImpactOutlierDetector
)

class EventStreamDataset(SeedableMixin, SaveableMixin, TimeableMixin):
    """
    A unified dataset object for storing event-stream data. Data is stored via three dataframes:
        1. `events_df`, which has integer index `event_id`, and 3 columns:
            * `subject_id` (any hashable type)
            * `timestamp` (pd.datetime)
            * `event_type` (str)
        2. `joint_metadata_df`, which has integer index `metadata_id`, and 3 mandatory columns, plus any
           number of user defined columns.
            * `event_id`, the integer index of the event to which this metadata element corresponds.
            * `event_type`, the type of event to which this metadata element corresponds.
            * `subject_id`, the subject_id of the event to which this metadata element corresponds.
        3. `subjects_df`, which has integer index `subject_id` and arbitrary other user-specified columns with
           subject-specific metadata.
    `joint_metadata_df` is joinable to `events_df` on `event_id`. One can also access
    `events_df_with_metadata`, which contains the columns of `events_df` plus an additional column,
    `metadata`, which contains an `ExpandableDfDict` view of the metadata dataframe rows corresponding to that
    event, with columns that are universally null removed. `events_df_with_metadata` is lazily constructed
    when accessed then cached, so first accesses may take a long time.

    TODO(mmd): Consider using pandas sparse matrices to simplify:
    https://pandas.pydata.org/docs/user_guide/sparse.html
    """

    # Dictates in which format the `_save` and `_load` methods will save/load objects of this class, as
    # defined in `SaveableMixin`.
    _PICKLER = 'dill'

    # Dictates what models can be fit on numerical metadata columns, for both outlier detection and
    # normalization.
    METADATA_MODELS = {
        # Outlier Detectors
        'variance_impact_outlier_detector': VarianceImpactOutlierDetector,

        # Normalizers
        'quantile_transformer': QuantileTransformer,
    }

    # This variable stores inferred upper and lower valid bounds for various units of measure. They are used
    # to drop outliers from observed numerical values.
    # TODO(mmd): Let this be set from a data file.
    UNIT_BOUNDS = {
        # (unit strings): [lower, lower_inclusive, upper, upper_inclusive],
        ('%', 'percent'): [0, False, 100, False],
    }

    @classmethod
    def infer_bounds_from_units_inplace(cls, measurement_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        This updates the bounds in measurement_metadata to reflect the implied bounds in `cls.UNIT_BOUNDS`.

        Args:
            `measurement_metadata` (`pd.DataFrame`):
                The initial bounds for each observed numeric key. Keys are stored in the index, and must have
                the following columns:
                    * `unit`: The unit of measure of this key.
                    * `drop_lower_bound`:
                        A lower bound such that values either below or at or below this level will be dropped
                        (key presence will be retained).
                    * `drop_lower_bound_inclusive`:
                        Is the drop lower bound inclusive or exclusive?
                    * `drop_upper_bound`:
                        An upper bound such that values either above or at or above this level will be dropped
                        (key presence will be retained).
                    * `drop_upper_bound_inclusive`:
                        Is the drop upper bound inclusive or exclusive?

        Returns:
            `pd.DataFrame`, of the same schema as `measurement_metadata` with the `drop_*` columns updated
            according to `cls.UNIT_BOUNDS`
        """

        unit_bounds = flatten_dict(cls.UNIT_BOUNDS)
        new_cols = [
            'unit_inferred_low', 'unit_inferred_low_inclusive', 'unit_inferred_high',
            'unit_inferred_high_inclusive'
        ]
        measurement_metadata[new_cols] = measurement_metadata.unit.apply(
            lambda u: pd.Series(unit_bounds.get(u, [None, None, None, None]))
        )


        for (old_bound, new_bound, min_max) in (
            ('drop_lower_bound', 'unit_inferred_low', 'max'),
            ('drop_upper_bound', 'unit_inferred_high', 'min'),
        ):
            old_incl = f"{old_bound}_inclusive"
            new_incl = f"{new_bound}_inclusive"

            if old_bound not in measurement_metadata.columns:
                measurement_metadata[old_bound] = measurement_metadata[new_bound]
                measurement_metadata[old_incl] = measurement_metadata[new_incl]
                continue

            if min_max == 'min':
                measurement_metadata[old_incl] = measurement_metadata[
                    [old_bound, new_bound, old_incl, new_incl]
                ].apply(
                    lambda r: (
                        r[old_incl] if (pd.isnull(r[new_bound])) else
                        r[new_incl] if (pd.isnull(r[old_bound])) else
                        r[old_incl] if (r[old_bound] < r[new_bound]) else
                        r[new_incl] if (r[new_bound] < r[old_bound]) else
                        (r[old_incl] or r[new_incl])
                    ), axis='columns'
                )
                measurement_metadata[old_bound] = measurement_metadata[[old_bound, new_bound]].min('columns')
            else:
                measurement_metadata[old_incl] = measurement_metadata[
                    [old_bound, new_bound, old_incl, new_incl]
                ].apply(
                    lambda r: (
                        r[old_incl] if (pd.isnull(r[new_bound])) else
                        r[new_incl] if (pd.isnull(r[old_bound])) else
                        r[old_incl] if (r[old_bound] > r[new_bound]) else
                        r[new_incl] if (r[new_bound] > r[old_bound]) else
                        (r[old_incl] or r[new_incl])
                    ), axis='columns'
                )
                measurement_metadata[old_bound] = measurement_metadata[[old_bound, new_bound]].max('columns')

            measurement_metadata.drop(columns=[new_bound, new_incl], inplace=True)

        return measurement_metadata

    @staticmethod
    def drop_or_censor_series(val: pd.Series, row: Union[pd.Series, Dict[str, Optional[float]]]) -> pd.Series:
        """
        Appropriately either drops (returns np.NaN) or censors (returns the censor value) the values in `val`
        based on the bounds in `row`. See `EventStreamDataset.drop_or_censor` for description of `row` bound
        keys and meaning.
        """
        return val.apply(EventStreamDataset.drop_or_censor, args=(row,))

    @staticmethod
    def drop_or_censor(val: float, row: Union[pd.Series, Dict[str, Optional[float]]]) -> float:
        """
        Appropriately either drops (returns np.NaN) or censors (returns the censor value) the value `val`
        based on the bounds in `row`.

        Args:
            `val` (`float`): The value to drop, censor, or return unchanged.
            `row` (`Union[pd.Series, Dict[str, Optional[float]]]`):
                The bounds for dropping and censoring. Must contain the following keys:
                    * `drop_lower_bound`:
                        A lower bound such that if `val` is either below or at or below this level, `np.NaN`
                        will be returned.
                        If `None` or `np.NaN`, no bound will be applied.
                    * `drop_lower_bound_inclusive`:
                        If `True`, returns `np.NaN` if `val <= row['drop_lower_bound']`. Else, returns
                        `np.NaN` if `val < row['drop_lower_bound']`.
                    * `drop_upper_bound`:
                        An upper bound such that if `val` is either above or at or above this level, `np.NaN`
                        will be returned.
                        If `None` or `np.NaN`, no bound will be applied.
                    * `drop_upper_bound_inclusive`:
                        If `True`, returns `np.NaN` if `val >= row['drop_upper_bound']`. Else, returns
                        `np.NaN` if `val > row['drop_upper_bound']`.
                    * `censor_lower_bound`:
                        A lower bound such that if `val` is below this level but above `drop_lower_bound`,
                        `censor_lower_bound` will be returned.
                        If `None` or `np.NaN`, no bound will be applied.
                    * `censor_upper_bound`:
                        An upper bound such that if `val` is above this level but below `drop_upper_bound`,
                        `censor_upper_bound` will be returned.
                        If `None` or `np.NaN`, no bound will be applied.

        """
        if (
            (
                ('drop_lower_bound' in row) and
                (not pd.isnull(row['drop_lower_bound'])) and (
                    (val < row['drop_lower_bound']) or
                    (row['drop_lower_bound_inclusive'] and val == row['drop_lower_bound'])
                )
            ) or (
                ('drop_upper_bound' in row) and
                (not pd.isnull(row['drop_upper_bound'])) and (
                    (val > row['drop_upper_bound']) or
                    (row['drop_upper_bound_inclusive'] and val == row['drop_upper_bound'])
                )
            )
        ): return np.NaN
        elif (
            ('censor_lower_bound' in row) and
            (not pd.isnull(row['censor_lower_bound'])) and
            (val < row['censor_lower_bound'])
        ): return row['censor_lower_bound']
        elif (
            ('censor_upper_bound' in row) and
            (not pd.isnull(row['censor_upper_bound'])) and
            (val > row['censor_upper_bound'])
        ): return row['censor_upper_bound']
        else: return val

    @staticmethod
    def drop_oob_and_censor_outliers(vals: pd.Series, measurement_metadata: pd.DataFrame) -> pd.Series:
        """
        Drops out-of-bounds data values and censors invalid data values according to measurement_metadata. Data
        are joined based on indices of `vals` and `measurement_metadata`.

        Args:
            `vals` (`pd.Series`): The values to be dropped or censored.
            `measurement_metadata` (`pd.DataFrame`):
                The dataframe containing the bounds to drop or censor `vals`. Must have one index level and
                the following column names:
                    * `drop_lower_bound`
                    * `drop_lower_bound_inclusive`
                    * `drop_upper_bound`
                    * `drop_upper_bound_inclusive`
                    * `censor_lower_bound`
                    * `censor_upper_bound`
                See `EventStreamDataset.drop_or_censor` for a description of what the columns mean individually.

        Returns: A `pd.Series` of the same schema as `vals` with the values either dropped or censored.
        """

        assert len(measurement_metadata.index.names) == 1

        orig_val_name = vals.name
        orig_val_index = vals.index
        orig_key_col_name = measurement_metadata.index.names[0]

        # We just want the unit bounds, and we need to standardize the index name
        cols = list(set(measurement_metadata.columns).intersection({
            'drop_lower_bound', 'drop_lower_bound_inclusive', 'drop_upper_bound',
            'drop_upper_bound_inclusive', 'censor_lower_bound', 'censor_upper_bound',
        }))
        if not cols: return vals

        vals = pd.DataFrame({'vals': vals})
        vals['int_index'] = np.arange(len(vals))

        vals.set_index('int_index', append=True, inplace=True)
        vals.index.names = ['key_col', 'int_index']

        measurement_metadata.index.names = ['key_col']
        processed_vals = vals.join(measurement_metadata[cols], on='key_col', how='left').apply(
            lambda r: EventStreamDataset.drop_or_censor(r['vals'], r),
            axis='columns'
        )

        measurement_metadata.index.names = [orig_key_col_name]

        processed_vals = processed_vals.loc[vals.index]
        processed_vals.name = orig_val_name
        processed_vals.index = orig_val_index

        return processed_vals

    @classmethod
    def _fit_metadata_model(cls, vals: pd.Series, model_config: Dict[str, Any]):
        """Fits a model as specified in `model_config` on the values in `vals`."""
        assert 'cls' in model_config
        assert model_config['cls'] in cls.METADATA_MODELS

        vals = to_sklearn_np(vals)
        if len(vals) == 0: return None

        N = len(vals)

        model_config = copy.deepcopy(
            {k: (v(N) if callable(v) else v) for k, v in model_config.items()}
        )
        model_cls = cls.METADATA_MODELS[model_config.pop('cls')]

        model = model_cls(**model_config)
        model.fit(vals)
        return model

    @classmethod
    def int_key_value_to_categorical(cls, key: Any, val: Union[int, float]) -> str:
        """Returns a string representation of a value, converted to an integer then a categorical key."""
        return f"{key}__EQ_{int(np.round(val)):d}"
    @classmethod
    def float_key_value_to_categorical(cls, key: Any, val: Union[int, float]) -> str:
        """Returns a string representation of a float value converted to a categorical key."""
        return f"{key}__EQ_{val}"

    @classmethod
    def transform_categorical_values_series(
        cls, measurement_metadata: pd.Series, vals: pd.Series
    ) -> Optional[pd.Series]:
        """
        Converts the observed values in `vals` to an appropriate numerical or categorical representation as
        dictated by `measurement_metadata['value_type']`.
        TODO(mmd): unify with dataframe function.

        Args:
            `measurement_metadata` (`pd.Series`):
                The series containing the value types dictating what conversions should apply.
                It must contain an index `'value_type'`.
                If `measurement_metadata.value_type == NumericDataModalitySubtype.CATEGORICAL_INTEGER`, the
                values will be converted to integers, then to string representations via the appropriate class
                method.
                If `measurement_metadata.value_type == NumericDataModalitySubtype.CATEGORICAL_FLOAT`, the
                values will be converted to string representations via the appropriate class method directly.
            `vals` (`pd.Series`):
            Contains the values to be modified. The name of the series will be used in the modification.

        Returns: A series containing the transformed values, if they warrant transformation, else None.
        """
        match measurement_metadata.value_type:
            case NumericDataModalitySubtype.CATEGORICAL_INTEGER:
                conversion_fn = cls.int_key_value_to_categorical
            case NumericDataModalitySubtype.CATEGORICAL_FLOAT:
                conversion_fn = cls.float_key_value_to_categorical
            case _: return None

        return vals.apply(lambda v: conversion_fn(key=vals.name, val=v))

    @classmethod
    def transform_categorical_key_values_df(
        cls, measurement_metadata: pd.DataFrame, kv_df: pd.DataFrame, key_col: str, val_col: str
    ) -> pd.DataFrame:
        """
        Converts the keys and values in `kv_df` to an appropriate numerical or categorical representation as
        dictated by `measurement_metadata['value_type']`.

        Args:
            `measurement_metadata` (`pd.DataFrame`):
                The dataframe containing the value types dictating what conversions should apply to what
                key-value pairs. The index of `measurement_metadata` must contain keys, and it must contain a
                column `'value_type'`.
                Keys with a `'value_type'` of `NumericDataModalitySubtype.CATEGORICAL_INTEGER` will be converted
                to integers, then to string representations via the appropriate class method.
                Keys with a `'value_type'` of `NumericDataModalitySubtype.CATEGORICAL_FLOAT` will be converted
                to string representations via the appropriate class method.
            `kv_df` (`pd.DataFrame`):
                Contains the key-value pairs (keys in column `key_col`, values in `val_col`) to be converted.
                This dataframe _will_ be modified by this function!
            `key_col` (`str`): The column of `kv_df` containing the keys.
            `val_col` (`str`): The column of `kv_df` containing the values.

        Returns: The modified `kv_df` with transformed keys, of type `pd.DataFrame`.
        """
        for value_type, conversion_fn in (
            (NumericDataModalitySubtype.CATEGORICAL_INTEGER, cls.int_key_value_to_categorical),
            (NumericDataModalitySubtype.CATEGORICAL_FLOAT, cls.float_key_value_to_categorical),
        ):
            keys_to_expand = set(measurement_metadata[measurement_metadata['value_type'] == value_type].index)
            if not keys_to_expand: continue

            keys_to_convert_idx = kv_df[key_col].isin(keys_to_expand)
            kv_df.loc[keys_to_convert_idx, key_col] = kv_df[keys_to_convert_idx].apply(
                lambda r: conversion_fn(key=r[key_col], val=r[val_col]),
                axis='columns'
            )
            kv_df.loc[keys_to_convert_idx, val_col] = np.NaN

        return kv_df

    @staticmethod
    def to_events(
        df:            pd.DataFrame,
        event_type:    str,
        subject_col:   str,
        time_col:      str,
        metadata_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Converts `df` into the format of an `events_df` as expected by `EventStreamDataset`.

        Args:
            `df` (`pd.DataFrame'):
                The dataframe to be converted. Must have one row per event (or be collapsible into such) and
                capture only a single event type.
            `event_type` (`str`): What type of event this dataframe captures.
            `subject_col` (`str`): The name of the column containing the subject ID.
            `time_col` (`str`): The name of the column containing the timestamp of the event.
            `metadata_cols` (`List[str]`, *optional*, default is `[]`):
                A list of the columns that should be captured as event-specific metadata. They will be
                extracted and converted to `ExpandableDfDict`s per the `EventStreamDataset` structure.

        Returns:
            * `events` (`pd.DataFrame`):
                A copy of of `df` with the following modifications:
                    * `time_col` renamed to `'timestamp'`
                    * `subject_col` renamed to `'subject_id'`
                    * An `'event_type'` column added added with the value `event_type`
                    * The index _overwritten_ with an `'event_id'` index which is a numerical index following
                      the order of records in the input `df`.
                    * A `'metadata'` column added whose values take on single-row `ExpandableDfDict` objects
                      corresponding to the columns and values of `df[metadata_cols]` for that row.
        """
        if metadata_cols is None: metadata_cols = []

        events = df[[subject_col, time_col]].rename(
            columns={time_col: 'timestamp', subject_col: 'subject_id'}
        ).copy()
        events['event_type'] = event_type
        events['metadata']   = df.apply(
            lambda r: ExpandableDfDict({k: [v] for k, v in r[metadata_cols].to_dict().items()}), axis=1
        )

        return events

    @staticmethod
    def to_events_and_metadata(
        df:            pd.DataFrame,
        event_type:    str,
        subject_col:   str,
        time_col:      str,
        metadata_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Converts `df` into the format of an `events_df`, `meatadata_df` pair as expected by
        `EventStreamDataset`.
        TODO(mmd): this function is inefficient, computationally. Should do computation on a single copy of
        the dataframe, then split into `events` vs. `metadata`.
        TODO(mmd): Should rename `events_df -> events_df` and `metadata -> joint_metadata_df` following
        EventStreamDataset convention or otherwise standardize.

        Args:
            `df` (`pd.DataFrame'):
                The dataframe to be converted. Must have one row per event (or be collapsible into such) and
                capture only a single event type.
            `event_type` (`str`): What type of event this dataframe captures.
            `subject_col` (`str`): The name of the column containing the subject ID.
            `time_col` (`str`): The name of the column containing the timestamp of the event.
            `metadata_cols` (`List[str]`, *optional*, default is `[]`):
                A list of the columns that should be captured as event-specific metadata. They will be
                extracted into a separate metadata_df with a shared index for the extracted events_df. As they
                are all sourced from the same underlying dataframe, they will have the same number of samples
                here, though that relationship need not hold in general within an `EventStreamDataset` object.

        Returns:
            * `events` (`pd.DataFrame`):
                A copy of `df` with the following modifications:
                    * `time_col` renamed to `'timestamp'`
                    * `subject_col` renamed to `'subject_id'`
                    * A column named `'event_type'` added with the value `event_type`
                    * The index _overwritten_ with an `'event_id'` index which is a range index following
                      the order of the records in `df`.
            * `metadata` (`pd.DataFrame`):
                A copy of `df[[subject_col] + metadata_cols]` with the following modifications:
                    * `subject_col` renamed to `subject_id`.
                    * Columns added corresponding to `'event_id'` and `'event_type'`, matching `events`.
                    * The index _overwritten_ with a `'metadata_id'` index which is a range index following
                      the order of the records in `df`.
        """
        if metadata_cols is None: metadata_cols = []

        events = df[[subject_col, time_col]].rename(
            columns={time_col: 'timestamp', subject_col: 'subject_id'}
        ).copy()
        events['event_type'] = event_type
        events['event_id'] = np.arange(len(df))
        events.set_index('event_id', inplace=True)

        metadata = df[[subject_col] + metadata_cols].copy().rename(columns={subject_col: 'subject_id'})
        metadata['event_id'] = np.arange(len(df))
        metadata['event_type'] = event_type
        metadata['metadata_id'] = np.arange(len(metadata))
        metadata.set_index('metadata_id', inplace=True)

        return events, metadata[['event_id', 'event_type', 'subject_id', *metadata_cols]]

    def __init__(
        self,
        config: EventStreamDatasetConfig,
        events_df: pd.DataFrame,
        metadata_df: Optional[pd.DataFrame] = None,
        subjects_df: Optional[pd.DataFrame] = None,
        do_copy: bool = True,
    ):
        """
        Builds the `EventStreamDataset` object.

        Args:
            `events_df` (`pd.DataFrame`):
                The underlying data. Should have the following columns:
                    * `subject_id`: The ID of the subject of the row.
                    * `event_type`: The type of the row's event.
                    * `timestamp`: The timestamp (in a `pd.to_datetime` parseable format) of the row's event.
                    * `metadata`:
                        Contains `ExpandableDfDict` objects capturing additional metadata elements for the
                        event. If `metadata_df` is `None`, this column must be present. Otherwise it is
                        unused.
            `metadata_df` (`Optional[pd.DataFrame]`, defaults to `None`):
                The associated underlying metadata. If present, one of the following two conditions must be
                true:
                    * It must contain a column `event_id` and `events_df` must have a single index columnh
                      named `event_id` which can be joined against this column. `subject_id` and `event_type`
                      will then be inferred (unless already present) in `metadata_df` on the basis of this
                      join key.
                    * It must be the same length as `events_df`, in which case it is assumed that they both
                      share the same index, and an appropriate `event_id` column is added to both to ensure
                      this.
            `config` (`EventStreamDatasetConfig`):
                Configuration objects for this dataset. Largely details how metadata should be processed.
            `do_copy` (`bool`, *optional*, defaults to True): Whether or not `events_df` should be copied.

        Upon instantiation, `events_df` will have timestamps converted to pandas datetime objects and will be
        sorted by subject and timestamp. The index of `events_df` will be discarded; however, an integer index
        will be assigned on the basis of the original order of `events_df` when passed in and will be retained
        as `event_id` (unless other operations reset this index), which can be used to resolve this object
        with any original index.
        """

        self.config = config
        self.metadata_is_fit = False

        # After pre-processing, we may infer new types or otherwise change measurement configuration, so
        # we store a separage configuration object for post-processing. It is initialized as empty as we have
        # not yet pre-processed anything.
        self.inferred_measurement_configs = {}

        if do_copy: events_df = events_df.copy()

        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])

        if metadata_df is not None:
            if 'event_id' in metadata_df.columns:
                assert events_df.index.names == ['event_id'], f"Got {events_df.index.names}"
                if 'event_type' not in metadata_df.columns:
                    metadata_df['event_type'] = events_df.loc[metadata_df.event_id, 'event_type']
                if 'subject_id' not in metadata_df.columns:
                    metadata_df['subject_id'] = events_df.loc[metadata_df.event_id, 'subject_id']
            else:
                assert len(metadata_df) == len(events_df)
                metadata_df['event_id'] = np.arange(len(events_df))
                metadata_df['event_type'] = events_df['event_type']

            non_event_cols = [c for c in metadata_df.columns if c not in ('event_id', 'event_type')]
            cols_order = ['event_id', 'event_type', *sorted(non_event_cols)]
            self.joint_metadata_df = metadata_df[cols_order].copy()
        elif 'metadata' not in events_df:
            self.joint_metadata_df = pd.DataFrame(
                {'event_id': [], 'event_type': []}, index=pd.Index([], name='metadata_id')
            )

        if subjects_df is not None: assert subjects_df.index.names == ['subject_id']

        self.subjects_df = subjects_df

        self.__events_df_with_metadata_stale = False
        self.events_df = events_df

        self.split_subjects = {}

    @property
    def has_static_measurements(self):
        return (
            (self.subjects_df is not None) and
            any(cfg.temporality == TemporalityType.STATIC for cfg in self.measurement_configs.values())
        )

    @property
    def passed_measurement_configs(self): return self.config.measurement_configs

    @TimeableMixin.TimeAs
    def __build_joint_metadata_df_from_events(self):
        """
        Builds a joint metadata dataframe by concatenating the `ExpandableDfDict` metadata objects present
        in each row of the events df, then saves that as `self.joint_metadata_df`.
        """
        def metadata_df_with_event_info(r: pd.Series):
            df = r.metadata.df()
            df['event_id'] = r.name
            df['event_type'] = r.event_type
            df['subject_id'] = r.subject_id
            return df
        metadata_df = pd.concat(
            (metadata_df_with_event_info(r) for _, r in self.events_df.iterrows()), ignore_index=True
        )
        metadata_df.index = np.arange(len(metadata_df))
        metadata_df.index.names = ['metadata_id']

        static_cols = ['event_id', 'event_type', 'subject_id']

        non_event_cols = [c for c in metadata_df.columns if c not in static_cols]
        cols_order = [*static_cols, *sorted(non_event_cols)]

        self.joint_metadata_df = metadata_df[cols_order]
        self.metadata_is_fit = False

    @property
    def events_df(self): return self._events_df

    @property
    def events_df_with_metadata(self):
        """
        If necessary, re-builds the events df view with metadata, then returns that view. Uses this method
        rather than `functools.cached_property` because clearing the `cached_property` version was (for some
        reason) very computationally expensive.
        """
        if self.__events_df_with_metadata_stale or not hasattr(self, '_events_df_with_metadata'):
            self._build_events_df_with_metadata()
            self.__events_df_with_metadata_stale = False

        return self._events_df_with_metadata

    @TimeableMixin.TimeAs
    def _build_events_df_with_metadata(self):
        """
        Builds the joint view of events with the `'metadata'` column that was originally used.
        Stored separately for ease of tracking timing even with the caching property above.
        """
        E = self._events_df.copy()
        if self.joint_metadata_df.shape[1] == 0:
            E['metadata'] = [ExpandableDfDict({}) for _ in range(len(E))]
        else:
            E['metadata'] = self.joint_metadata_df.drop(
                columns=['event_type', 'subject_id']
            ).groupby('event_id', group_keys=True).apply(ExpandableDfDict.from_df)
        self._events_df_with_metadata = E

    def __clear_events_with_metadata(self):
        """
        An internal endpoint for clearing the pre-cached events with metadata. Right now, this function may be
        not necessary given how rarely that view is used, and it may be removed in the future.
        """
        self.__events_df_with_metadata_stale = True

    @events_df.setter
    @TimeableMixin.TimeAs(key='events_df_setter')
    def events_df(self, new_df):
        """
        Sets `self.events_df`, and performs necessary associated operations:
          * Resetting the index to a new integer index (event_id).
          * Adding the new event_id to the metadata.
          * If necessary, re-building `self.joint_metadata_df`.
          * Clearing the stored `events_with_metadata` view.
          * Sorts the events by `'subject_id'` and `'timestamp'`.
        """
        self._events_df = new_df
        if self.events_df.index.names == ['event_id']:
            assert self.events_df.index.is_unique
        else:
            self.events_df.index = np.arange(len(self.events_df))
            self.events_df.index.names = ['event_id']

        if 'metadata' in self.events_df.columns:
            self.__build_joint_metadata_df_from_events()
            self.events_df.drop(columns='metadata', inplace=True)

        self.__clear_events_with_metadata()
        self.sort_events()

        self.event_types = [e for e, _ in Counter(self.events_df.event_type).most_common()]
        self.subject_ids = set(self.events_df.subject_id)
        self.n_events_per_subject = self.events_df.groupby('subject_id').timestamp.count().to_dict()

        if self.subjects_df is not None:
            subjects_with_no_events = set(self.subjects_df.index.values) - self.subject_ids
            for sid in subjects_with_no_events: self.n_events_per_subject[sid] = 0

            self.subject_ids.update(subjects_with_no_events)

    @TimeableMixin.TimeAs
    def sort_events(self):
        """Sorts events by subject ID and timestamp in ascending order."""
        self.events_df.sort_values(by=['subject_id', 'timestamp'], ascending=True, inplace=True)

    @TimeableMixin.TimeAs
    def agg_by_time_type(self):
        """
        Aggregates the events_df by subject_id, timestamp, and event_type, tracking all associated metadata.
        Note that no numerical aggregation (e.g., mean, etc.) happens here; duplicate entries will both be
        captured in the output metadata object.
        """

        with self._time_as('agg_by_time_type_group_by'):
            self.events_df = self.events_df.reset_index().groupby([
                'subject_id', 'timestamp', 'event_type'
            ]).event_id.agg(set).reset_index()
        self.events_df.rename(columns={'event_id': 'old_event_id'}, inplace=True)

        new_to_old_set = self.events_df['old_event_id'].to_dict()
        self.events_df.drop(columns=['old_event_id'], inplace=True)

        old_to_new = {}
        for new, olds in new_to_old_set.items():
            for old in olds: old_to_new[old] = new

        with self._time_as('agg_by_time_type_update_metadata_event_ids'):
            # This may cause unnecessary OOMs: https://github.com/pandas-dev/pandas/issues/6697
            #self.joint_metadata_df.event_id.replace(old_to_new, inplace=True)
            self.joint_metadata_df['event_id'] = self.joint_metadata_df.event_id.apply(
                lambda x: old_to_new[x]
            )
        self.__clear_events_with_metadata()

    @SeedableMixin.WithSeed
    @TimeableMixin.TimeAs
    def split(
        self,
        split_fracs: Sequence[float],
        split_names: Optional[Sequence[str]] = None,
    ):
        """
        Splits the underlying dataset into random sets by `subject_id`.
        Args:
            `split_fracs` (`Sequence[float]`, each split_frac must be >= 0 and the sum must be <= 1):
                The fractional sizes of the desired splits. If it sums to < 1, the remainder will be tracked
                in an extra split at the end of the list.
            `split_names` (`Sequence[str]`, *optional*, defaults to None):
                If specified, assigns the passed names to each split. Must be of the same size as
                `split_fracs` (after it is expanded to sum to 1 if necessary). If unset, and there are two
                splits, it defaults to [`train`, `held_out`]. If there are three, it defaults to `['train',
                'tuning', 'held_out']. If more than 3, it defaults to `['split_0', 'split_1', ...]`. Split
                names of `train`, `tuning`, and `held_out` have special significance and are used elsewhere in
                the model, so if `split_names` does not reflect those other things may not work down the line.
        """
        split_fracs = list(split_fracs)

        assert min(split_fracs) >= 0 and max(split_fracs) <= 1
        assert sum(split_fracs) <= 1
        if sum(split_fracs) < 1: split_fracs.append(1-sum(split_fracs))

        if split_names is None:
            if len(split_fracs) == 2: split_names = ['train', 'held_out']
            elif len(split_fracs) == 3: split_names = ['train', 'tuning', 'held_out']
            else: split_names = [f'split_{i}' for i in range(len(split_fracs))]
        else: assert len(split_names) == len(split_fracs)

        subjects   = np.random.permutation(list(self.subject_ids))
        split_lens = (np.array(split_fracs) * len(subjects)).round().astype(int)

        subjects_per_split = np.split(subjects, split_lens.cumsum())

        self.split_subjects = {k: set(v) for k, v in zip(split_names, subjects_per_split)}

    @property
    def splits(self): return set(self.split_subjects.keys())

    @TimeableMixin.TimeAs
    def subject_ids_for_split(self, split: Optional[str] = None, splits: Optional[str] = None) -> Set[int]:
        """Returns subjects in split `split` or `splits` (both cannot be set; returns all if neither)."""
        if (split is None) and (splits is None): return self.subject_ids

        assert not ((split is not None) and (splits is not None))
        if split is not None: splits = [split]

        for sp in splits: assert sp in self.splits, f"Split {sp} not found."

        return set().union(*(self.split_subjects[sp] for sp in splits))

    # We have special callouts for train, tuning, and held_out split subjects.
    @property
    def train_subject_ids(self): return self.subject_ids_for_split('train')
    @property
    def tuning_subject_ids(self): return self.subject_ids_for_split('tuning')
    @property
    def held_out_subject_ids(self): return self.subject_ids_for_split('held_out')

    @property
    def train_subjects_df(self):
        return self.subjects_df[
            self.subjects_df.index.isin(self.subject_ids_for_split('train'))
        ]
    @property
    def tuning_subjects_df(self):
        return self.subjects_df[
            self.subjects_df.index.isin(self.subject_ids_for_split('tuning'))
        ]
    @property
    def held_out_subjects_df(self):
        return self.subjects_df[
            self.subjects_df.index.isin(self.subject_ids_for_split('held_out'))
        ]

    @TimeableMixin.TimeAs
    def _events_for_split(
        self, split: Optional[str] = None, splits: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """Returns the events in split `split` or splits `splits` (can't set both), or all events."""
        if split is None and splits is None: return self.events_df
        return self.events_df[self.events_df.subject_id.isin(self.subject_ids_for_split(split, splits))]

    @TimeableMixin.TimeAs
    def __metadata_df_idx(
        self,
        event_types: Optional[Sequence[str]] = None,
        event_type: Optional[str] = None,
        splits: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
        subject_ids: Optional[Sequence[Hashable]] = None,
        subject_id: Optional[Hashable] = None,
    ) -> np.ndarray:
        """
        Returns a numpy array that serves as a valid index into `self.joint_metadata_df` corresponding to
        events following input constraints. The index returned is in the same order as
        `self.joint_metadata_df` as of the time of the function call.

        Args:
            * `event_types` (`Optional[Sequence[str]]`), *optional*, defaults to `None`:
                If specified, the returned index will only contain metadata events corresponding to events of
                a type in `event_types`.
                Cannot be simultanesouly set with `event_type`.
            * `event_type` (`Optional[str]`), *optional*, defaults to `None`:
                If specified, the returned index will only contain metadata events corresponding to events of
                type `event_type`.
                Cannot be simultanesouly set with `event_types`.
            * `splits` (`Optional[Sequence[str]]`), *optional*, defaults to `None`:
                If specified, the returned index will only contain metadata events corresponding to events of
                a subject in a split in `splits`.
                Cannot be simultanesouly set with `split`.
            * `splits` (`Optional[str]`), *optional*, defaults to `None`:
                If specified, the returned index will only contain metadata events corresponding to events of
                a subject in split `split`.
                Cannot be simultanesouly set with `split`.
            * `subject_ids` (`Optional[Sequence[Hashable]]`), *optional*, defaults to `None`:
                If specified, the returned index will only contain metadata events corresponding to events of
                a subject in `subject_ids`.
                Cannot be simultanesouly set with `subject_id`.
            * `subject_id` (`Optional[Hashable]`), *optional*, defaults to `None`:
                If specified, the returned index will only contain metadata events corresponding to events of
                subject `subject_id`.
                Cannot be simultanesouly set with `subject_ids`.
        Returns:
            A boolean index into `self.joint_metadata_df` which satisfies the constraints implied by the
            arguments.
        """

        assert not ((subject_id is not None) and (subject_ids is not None))
        assert not ((event_type is not None) and (event_types is not None))

        if subject_id is not None: subject_ids = [subject_id]
        if subject_ids is not None: assert (split is None) and (splits is None)
        elif ((split is not None) or (splits is not None)):
            subject_ids = self.subject_ids_for_split(split, splits)

        if event_type is not None: event_types = [event_type]

        event_types = set(event_types) if event_types is not None else None

        out_idx = None
        if event_types is not None:
            out_idx = self.joint_metadata_df.event_type.isin(event_types).values

        if subject_ids is not None:
            subjects_idx = self.joint_metadata_df.subject_id.isin(subject_ids).values
            if out_idx is None: out_idx = subjects_idx
            else: out_idx = out_idx & subjects_idx

        return out_idx

    @TimeableMixin.TimeAs
    def metadata_df(self, *args, **kwargs):
        """Retrieves restricted metadata records and drops nan columns"""
        idx = self.__metadata_df_idx(*args, **kwargs)
        if idx is None: df = self.joint_metadata_df
        else: df = self.joint_metadata_df.loc[idx]
        return df.dropna(axis=1, how='all')

    # Special accessors for train, tuning, and held-out splits.
    @property
    def train_events_df(self): return self._events_for_split('train')
    @property
    def tuning_events_df(self): return self._events_for_split('tuning')
    @property
    def held_out_events_df(self): return self._events_for_split('held_out')

    def _inter_event_times_for_split(self, split: str, unit: pd.Timedelta) -> pd.Series:
        """Returns the inter-event times for this dataset, in the specified units."""
        inter_event_times = self._events_for_split(split).groupby('subject_id').timestamp.diff().dropna()
        return inter_event_times / unit

    @property
    def train_mean_log_inter_event_time_min(self):
        """Returns the mean of the log inter-event times in the training set plus one (due to zeros)."""
        # TODO(mmd): This adding of one is necessary to deal with events at the same time; likely better to
        # not permit collisions like that at all...
        times = self._inter_event_times_for_split('train', pd.Timedelta(1, unit='minute')) + 1
        return np.log(times).mean()

    @property
    def train_std_log_inter_event_time_min(self):
        """Returns the std of the log inter-event times in the training set plus one (due to zeros)."""
        # TODO(mmd): This adding of one is necessary to deal with events at the same time; likely better to
        # not permit collisions like that at all...
        times = self._inter_event_times_for_split('train', pd.Timedelta(1, unit='minute')) + 1
        return np.log(times).std()

    @TimeableMixin.TimeAs
    def backup_numerical_metadata_columns(self):
        """Backs up the all numerical columns to avoid data loss during processing."""
        for gp_by_col, val_col in self.dynamic_numerical_columns:
            backup_gp_by_col, backup_val_col = f"__backup_{gp_by_col}", f"__backup_{val_col}"
            if backup_gp_by_col not in self.joint_metadata_df.columns:
                self.joint_metadata_df.loc[:, backup_gp_by_col] = self.joint_metadata_df[gp_by_col]
            if backup_val_col not in self.joint_metadata_df.columns:
                self.joint_metadata_df.loc[:, backup_val_col] = self.joint_metadata_df[val_col]

        for val_col in self.time_dependent_numerical_columns:
            backup_val_col = f"__backup_{val_col}"
            if backup_val_col not in self.events_df.columns:
                self.events_df.loc[:, backup_val_col] = self.events_df[val_col]

    @TimeableMixin.TimeAs
    def restore_numerical_metadata_columns(self):
        """Restores backed-up copies of all numerical columns."""
        for gp_by_col, val_col in self.dynamic_numerical_columns:
            backup_gp_by_col, backup_val_col = f"__backup_{gp_by_col}", f"__backup_{val_col}"
            assert backup_gp_by_col in self.joint_metadata_df.columns
            self.joint_metadata_df[gp_by_col] = self.joint_metadata_df[backup_gp_by_col]

            assert backup_val_col in self.joint_metadata_df.columns
            self.joint_metadata_df[val_col] = self.joint_metadata_df[backup_val_col]

            self.joint_metadata_df.drop(columns=[backup_gp_by_col, backup_val_col], inplace=True)

        for val_col in self.time_dependent_numerical_columns:
            backup_val_col = f"__backup_{val_col}"
            assert backup_val_col in self.events_df.columns
            self.events_df[val_col] = self.events_df[backup_val_col]

            self.events_df.drop(columns=[backup_val_col], inplace=True)
        self.__clear_events_with_metadata()

    @TimeableMixin.TimeAs
    def preprocess_metadata(self):
        """Fits all metadata over the train set, then transforms all metadata."""
        self.add_time_dependent_columns()
        self.fit_metadata()
        self.transform_metadata()

    @TimeableMixin.TimeAs
    def add_time_dependent_columns(self):
        timestamps_series = self.events_df.set_index('subject_id', append=True).timestamp
        for col, cfg in self.passed_measurement_configs.items():
            if cfg.temporality != TemporalityType.FUNCTIONAL_TIME_DEPENDENT: continue

            function_vals = cfg.functor(timestamps_series, self.subjects_df)
            function_vals.index = function_vals.index.get_level_values('event_id')
            self.events_df[col] = function_vals

    @TimeableMixin.TimeAs
    def fit_metadata(self):
        """
        Fits preprocessing models, variables, and vocabularies over all metadata, including both numerical and
        categorical columns, over the training split. Details of pre-processing are dictated by `self.config`.
        """
        self.metadata_is_fit = False
        self._fit_numerical_metadata()
        self._fit_categorical_metadata()
        self.metadata_is_fit = True

    @property
    def measurement_configs(self):
        """
        Returns the inferred configuration objects if the metadata preprocessors have been fit, otherwise
        the passed configuration objects.
        """
        return self.inferred_measurement_configs if self.metadata_is_fit else self.config.measurement_configs

    @property
    def measurements(self):
        """Returns all non-dropped metadata columns."""
        return [k for k, cfg in self.measurement_configs.items() if not cfg.is_dropped]

    @property
    def dynamic_numerical_columns(self):
        """Returns all numerical metadata column key-column, value-column pairs."""
        return [
            (k, cfg.values_column) for k, cfg in self.measurement_configs.items() \
                if (cfg.is_numeric and cfg.temporality == TemporalityType.DYNAMIC)
        ]
    @property
    def time_dependent_numerical_columns(self):
        """Returns all numerical metadata column key-column, value-column pairs."""
        return [
            k for k, cfg in self.measurement_configs.items() \
                if (cfg.is_numeric and cfg.temporality == TemporalityType.FUNCTIONAL_TIME_DEPENDENT)
        ]

    @property
    def measurement_idxmaps(self):
        """Accesses the fit vocabularies vocabulary idxmap objects, per measurement column."""
        idxmaps = {}
        for m in self.measurements:
            config = self.measurement_configs[m]
            if config.vocabulary is not None: idxmaps[m] = config.vocabulary.idxmap
        return idxmaps

    @property
    def measurement_vocabs(self):
        """Accesses the fit vocabularies vocabulary objects, per measurement column."""

        vocabs = {}
        for m in self.measurements:
            config = self.measurement_configs[m]
            if config.vocabulary is not None: vocabs[m] = config.vocabulary.vocabulary
        return vocabs

    @TimeableMixin.TimeAs
    def _fit_numerical_metadata(self):
        """
        Pre-processes numerical metadata columns, according to `self.config`. Iterates through individual
        numerical columns and processes them independently.
        """

        for k, v in self.dynamic_numerical_columns: self._fit_dynamic_numerical_metadata_column(k, v)
        for k in self.time_dependent_numerical_columns: self._fit_time_dependent_numerical_metadata_column(k)

    @TimeableMixin.TimeAs
    def _fit_dynamic_numerical_metadata_column(self, key_col: str, val_col: str):
        """
        Pre-processes a particular numerical metadata column, with key `key_col`.

        Performs the following steps:
            1. Infers additional bounds on the basis of metadata units.
            2. Eliminates hard outliers and performs censoring via specified config.
            3. Per-metadata key, processes the metadata values.

        Throughout these steps, the model also tracks the final configuration of the column in
        `self.inferred_measurement_configs`.

        Args:
            `key_col` (`str`): The column name of the governing key column.
            `val_col` (`str`):
                The column name of the governing values column. Note this should be identical to
                `self.passed_measurement_configs[key_col].values_column`.
        """
        with self._time_as('building_inferred_config'):
            passed_config = self.passed_measurement_configs[key_col]
            # TODO(mmd): Infer this
            event_types = passed_config.present_in_event_types

            inferred_config = copy.deepcopy(passed_config)
            self.inferred_measurement_configs[key_col] = inferred_config

        with self._time_as('get_kv_df'):
            kv_train_df = self.metadata_df(event_types=event_types, split='train')[[key_col, val_col]]

            N = len(kv_train_df[key_col].dropna())
            total_possible_events = len(kv_train_df)

        if lt_count_or_proportion(N, self.config.min_valid_column_observations, total_possible_events):
            # In this case, we're going to drop this column entirely, so we can return.
            return

        with self._time_as('build_measurement_metadata'):
            inferred_config.add_missing_mandatory_metadata_cols()
            measurement_metadata = inferred_config.measurement_metadata

        # 1. Infers additional bounds on the basis of metadata units.
        with self._time_as('check_units'):
            if 'unit' in measurement_metadata.columns:
                self.infer_bounds_from_units_inplace(measurement_metadata)

        vals = kv_train_df.set_index(key_col)[val_col]

        # 2. Eliminate hard outliers and perform censoring.
        with self._time_as('drop_oob_and_censor_outliers'):
            vals = self.drop_oob_and_censor_outliers(vals, measurement_metadata)

        # 3. Per-metadata key, process the metadata values.
        with warnings.catch_warnings():
            # As `_fit_dynamic_numerical_metadata_column_vals` returns None, pandas throws a warning about the
            # default dtype for an empty Series, so we add a filter to ignore that.
            warnings.simplefilter(action='ignore', category=FutureWarning)
            vals.groupby(level=key_col).transform(
                self._fit_dynamic_numerical_metadata_column_vals,
                key_col=key_col,
                total_col_obs=N
            )

    @TimeableMixin.TimeAs
    def _fit_dynamic_numerical_metadata_column_vals(
        self, vals: pd.Series, key_col: str, total_col_obs: int
    ):
        gp_key = vals.name
        measurement_metadata = self.inferred_measurement_configs[key_col].measurement_metadata

        if gp_key not in measurement_metadata.index:
            measurement_metadata.loc[gp_key] = pd.Series(
                [None for _ in measurement_metadata.columns], dtype=object
            )

        measurement_metadata_copy = measurement_metadata.loc[gp_key].copy()

        output = self._fit_numerical_metadata_column_vals(
            vals, measurement_metadata_copy, total_col_obs
        )

        measurement_metadata.loc[gp_key] = measurement_metadata_copy
        return output

    @TimeableMixin.TimeAs
    def _fit_time_dependent_numerical_metadata_column(self, col: str):
        """
        Pre-processes a particular numerical metadata column, with key `key_col`.

        Performs the following steps:
            1. Infers additional bounds on the basis of metadata units.
            2. Eliminates hard outliers and performs censoring via specified config.
            3. Per-metadata key, processes the metadata values.

        Throughout these steps, the model also tracks the final configuration of the column in
        `self.inferred_measurement_configs`.

        Args:
            `key_col` (`str`): The column name of the governing key column.
            `val_col` (`str`):
                The column name of the governing values column. Note this should be identical to
                `self.passed_measurement_configs[key_col].values_column`.
        """
        with self._time_as('building_inferred_config'):
            passed_config = self.passed_measurement_configs[col]

            inferred_config = copy.deepcopy(passed_config)
            self.inferred_measurement_configs[col] = inferred_config

        with self._time_as('get_vals'):
            vals = self.train_events_df[col]

            N = len(vals.dropna())
            total_possible_events = len(vals)

        if lt_count_or_proportion(N, self.config.min_valid_column_observations, total_possible_events):
            # In this case, we're going to drop this column entirely, so we can return.
            return

        with self._time_as('build_measurement_metadata'):
            inferred_config.add_missing_mandatory_metadata_cols()
            measurement_metadata = inferred_config.measurement_metadata

        # 1. Infers additional bounds on the basis of metadata units.
        with self._time_as('check_units'):
            if 'unit' in measurement_metadata:
                # TODO(mmd): unify with other function.
                unit_bounds = flatten_dict(self.UNIT_BOUNDS)
                new_cols = [
                    'unit_inferred_low', 'unit_inferred_low_inclusive', 'unit_inferred_high',
                    'unit_inferred_high_inclusive'
                ]
                measurement_metadata[new_cols] = unit_bounds.get(
                    measurement_metadata['unit'], [None, None, None, None]
                )

        # 2. Eliminate hard outliers and perform censoring.
        with self._time_as('drop_oob_and_censor_outliers'):
            vals = self.drop_or_censor_series(vals, measurement_metadata)

        # 3. Per-metadata key, process the metadata values.
        self._fit_numerical_metadata_column_vals(vals, measurement_metadata, N)

    @TimeableMixin.TimeAs
    def _fit_numerical_metadata_column_vals(
        self, vals: pd.Series, measurement_metadata: pd.Series, total_col_obs: int
    ):
        """
        Fits the requisite numerical preprocessors on the given metadata column values.
        TODO(mmd): It may be that this is not efficient, and instead configuration objects should be passed in
        as inputs to avoid needing the full class context across all groupby workers.

        Performs the following steps:
            1. Sets the column type if it is not pre-set. If necessary, converts the values to the appropriate
               type prior to subsequent processing, but does not alter persistent metadata.
            2. Fits an outlier detection model. If necessary, removes outliers in training prior to
               normalization, but does not alter persistent metadata.
            3. Fits a normalizer model.

        Args:
            `vals` (`pd.Series`): The values to be pre-processed.
            `key_col` (`str`): The name of the key column that is being processed.
            `total_col_obs` (`int`):
                The total number of column observations that were observed for this metadata column (_not_
                just this key!)
        """

        total_key_obs = len(vals)
        vals = vals.dropna()

        # 1. Sets the column type if it is not pre-set. If necessary, converts the values to the appropriate
        # type prior to subsequent processing, but does not alter persistent metadata.

        if pd.isnull(measurement_metadata['value_type']):
            measurement_metadata.loc['value_type'] = self._infer_val_type(vals, total_col_obs, total_key_obs)

        # After inferring the value type, we need to convert it or return if necessary.
        match measurement_metadata.loc['value_type']:
            case NumericDataModalitySubtype.INTEGER: vals = vals.round(0).astype(int)
            case NumericDataModalitySubtype.FLOAT: pass
            case _: return

        # 2. Fits an outlier detection model, then removes outliers locally prior to normalization.
        if self.config.outlier_detector_config is not None:
            with self._time_as('fit_outlier_detector'):
                outlier_model = self._fit_metadata_model(vals, self.config.outlier_detector_config)
                measurement_metadata.loc['outlier_model'] = outlier_model

                inliers = outlier_model.predict(to_sklearn_np(vals)).reshape(-1)
                if (inliers == -1).all():
                    measurement_metadata.loc['value_type'] = NumericDataModalitySubtype.DROPPED
                    return

                vals[inliers == -1] = np.NaN

        # 3. Fits a normalizer model.
        if self.config.normalizer_config is not None:
            with self._time_as('fit_normalizer'):
                normalizer_model = self._fit_metadata_model(vals, self.config.normalizer_config)
                measurement_metadata.loc['normalizer'] = normalizer_model

    @TimeableMixin.TimeAs
    def _infer_val_type(
        self, vals: pd.Series, total_col_obs: int, total_key_obs: int
    ) -> NumericDataModalitySubtype:
        """
        Infers the appropriate type of the passed metadata column values. Performs the following steps:
            1. Determines if the column should be dropped for having too few measurements.
            2. Determines if the column actually contains integral, not floating point values.
            3. Determines if the column should be partially or fully re-categorized as a categorical column.

        Args:
            `vals` (`pd.Series`): The values to be pre-processed.
            `total_col_obs` (`int`):
                The total number of column observations that were observed for this metadata column (_not_
                just this key!)
            `total_key_obs` (`int`):
                The total number of observations of this key that were observed, including those that
                lacked a value.

        Returns: The appropriate `NumericDataModalitySubtype` for the values.
        """
        # 1. Determines if the column should be dropped for having too few measurements.
        if lt_count_or_proportion(
            total_key_obs, self.config.min_valid_vocab_element_observations, total_col_obs
        ):
            # In this case, there are too few values to even form a valid observation, so we drop all numeric
            # values and return NaN. Presuming this is the only instance of the key column (which it should
            # be), this key will also be dropped during categorical processing.
            return NumericDataModalitySubtype.DROPPED

        value_type = NumericDataModalitySubtype.FLOAT
        vals = vals.dropna()

        # 2. Determine if the column actually contains integral, not floating point values.
        if self.config.min_true_float_frequency is not None:
            int_freq = (vals == np.floor(vals)).mean()
            if int_freq > 1 - self.config.min_true_float_frequency:
                vals = vals.round(0).astype(int)
                value_type = NumericDataModalitySubtype.INTEGER

        # 3. Determine if the column should be partially or fully re-categorized as a categorical column.
        value_counts = vals.value_counts(dropna=True)

        if (
            lt_count_or_proportion(
                len(value_counts), self.config.min_unique_numerical_observations, len(vals)
            ) or (
                (self.config.max_numerical_value_frequency is not None) and
                ((value_counts.iloc[0] / len(vals)) > self.config.max_numerical_value_frequency)
            )
        ):
            # Here, we convert the output to categorical.
            if len(value_counts) == 1:
                value_type = NumericDataModalitySubtype.DROPPED
            elif value_type == NumericDataModalitySubtype.INTEGER:
                value_type = NumericDataModalitySubtype.CATEGORICAL_INTEGER
            elif value_type == NumericDataModalitySubtype.FLOAT:
                value_type = NumericDataModalitySubtype.CATEGORICAL_FLOAT
            else:
                raise ValueError(f"Unrecognized value type: {value_type}")

        return value_type

    @TimeableMixin.TimeAs
    def _fit_categorical_metadata(self):
        """
        Pre-processes categorical metadata columns, according to `self.config`.

        Performs the following steps for each categorical metadata column:
            1. Set the overall observation frequency for the column.
            2. Drop the column if observations occur too rarely.
            3. Fit metadata vocabularies on the training set.
            4. Eliminate observations that occur too rarely.
            5. If all observations were eliminated, we need to drop the entire column.
        """
        for m in self.measurements: self._fit_categorical_metadata_column(m)

    @TimeableMixin.TimeAs
    def _fit_categorical_metadata_column(self, col: str):
        passed_config = self.passed_measurement_configs[col]

        if col not in self.inferred_measurement_configs:
            self.inferred_measurement_configs[col] = copy.deepcopy(passed_config)

        config = self.inferred_measurement_configs[col]

        match config.temporality:
            case TemporalityType.DYNAMIC:
                measurement_df = self.metadata_df(event_types=config.present_in_event_types, split='train')
            case TemporalityType.STATIC:
                measurement_df = self.train_subjects_df
            case TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                measurement_df = self.train_events_df

        if col not in measurement_df:
            config.drop()
            return

        total_possible_events = len(measurement_df)
        measurement_df = measurement_df[~pd.isnull(measurement_df[col])]

        match config.modality:
            case DataModality.DROPPED: return

            case DataModality.MULTIVARIATE_REGRESSION:
                kv_df = measurement_df[[col, config.values_column]].copy()
                kv_df = self.transform_categorical_key_values_df(
                    config.measurement_metadata, kv_df, col, config.values_column
                )
                observations = kv_df[col]

            case DataModality.UNIVARIATE_REGRESSION:
                observations = self.transform_categorical_values_series(
                    config.measurement_metadata, measurement_df[col].copy()
                )
                if observations is None: return

            case _: observations = measurement_df[col].copy()

        return self._fit_categorical_metadata_column_vals(observations, config, total_possible_events)

    @TimeableMixin.TimeAs
    def _fit_categorical_metadata_column_vals(
        self, vals: pd.Series, inferred_config: MeasurementConfig, total_possible_events: int
    ):
        # 1. Set the overall observation frequency for the column.
        N = len(vals)
        inferred_config.observation_frequency = N / total_possible_events

        # 2. Drop the column if observations occur too rarely.
        if lt_count_or_proportion(N, self.config.min_valid_column_observations, total_possible_events):
            inferred_config.drop()
            return

        # 3. Fit metadata vocabularies on the trianing set.
        if inferred_config.vocabulary is None:
            try:
                inferred_config.vocabulary = Vocabulary.build_vocab(vals)
            except AssertionError as e:
                raise AssertionError(f"Failed to build vocabulary for {col}") from e

        # 4. Eliminate observations that occur too rarely.
        if self.config.min_valid_vocab_element_observations is not None:
            inferred_config.vocabulary.filter(N, self.config.min_valid_vocab_element_observations)

        # 5. If all observations were eliminated, drop the column.
        if inferred_config.vocabulary.vocabulary == ['UNK']:
            inferred_config.drop()
            return

    @TimeableMixin.TimeAs
    def transform_metadata(self):
        """Transforms the entire dataset metadata given the fit pre-processors."""
        self.backup_numerical_metadata_columns()

        for k, v in self.dynamic_numerical_columns: 
            self._transform_dynamic_numerical_metadata_column(k, v)

        for k in self.time_dependent_numerical_columns:
            self._transform_time_dependent_numerical_metadata_column(k)

        self.__clear_events_with_metadata()

    @TimeableMixin.TimeAs
    def _transform_dynamic_numerical_metadata_column(self, key_col: str, val_col: str):
        """
        Transforms a particular numerical metadata column, with key `key_col`.

        Performs the following steps:
            1. Transforms keys to categorical representations for categorical keys.
            2. Eliminates any values associated with dropped or categorical keys.
            3. Eliminates hard outliers and performs censoring via specified config.
            4. Convert values to desired types.
            5. Add inlier/outlier indices and remove learned outliers.
            6. Normalize values.

        Args:
            `key_col` (`str`): The column name of the governing key column.
            `val_col` (`str`): The column name of the underlying values column.
        """

        config = self.measurement_configs[key_col]
        measurement_metadata = config.measurement_metadata

        event_types = config.present_in_event_types
        kv_df = self.metadata_df(event_types=event_types)[[key_col, val_col]]
        kv_df = kv_df[~kv_df[key_col].isna()].copy()

        # 1. Transforms keys to categorical representations for categorical keys.
        kv_df = self.transform_categorical_key_values_df(measurement_metadata, kv_df, key_col, val_col)

        # 2. Eliminates any values associated with dropped or categorical keys.
        kv_df.loc[~(kv_df[key_col].isin(config.vocabulary.vocab_set)), val_col] = np.NaN

        # Reset the true metadata from steps one and two before continuing.
        with warnings.catch_warnings():
            # This operation throws a Deprecation warning because pandas thinks the new columns are a
            # different type than the prior columns, so current behavior is to set this with a copy and new
            # behavior will be to set it in place. It actually isn't a new type though, but I think pandas
            # doesn't realize that for some reason. This rationale is an inference from the other source of
            # this warning in the code.
            warnings.simplefilter(action='ignore', category=DeprecationWarning)
            self.joint_metadata_df.loc[kv_df.index, kv_df.columns] = kv_df

        # 3. Eliminates hard outliers and performs censoring via specified config.
        present_idx = ~kv_df[val_col].isna()

        if not present_idx.any(): return

        kv_df = kv_df[present_idx]

        inlier_col = f"{val_col}_is_inlier"
        kv_df[inlier_col] = None
        kv_df.astype({inlier_col: 'boolean'}, copy=False)

        self.joint_metadata_df.loc[kv_df.index, [val_col, inlier_col]] = kv_df.groupby(
            key_col, group_keys=False
        ).apply(
            self._transform_numerical_metadata_column_vals_gp, measurement_metadata, val_col, inlier_col
        )

    @TimeableMixin.TimeAs
    def _transform_time_dependent_numerical_metadata_column(self, col: str):
        """
        Transforms a particular numerical metadata column, with key `key_col`.

        Performs the following steps:
            1. Eliminates any values associated with dropped or categorical keys.
            3. Eliminates hard outliers and performs censoring via specified config.
            4. Convert values to desired types.
            5. Add inlier/outlier indices and remove learned outliers.
            6. Normalize values.

        Args:
            `col` (`str`): The column name of the governing column.
        """

        config = self.measurement_configs[col]
        measurement_metadata = config.measurement_metadata

        vals_df = self.events_df[[col]]

        # 1. Transforms keys to categorical representations for categorical keys.
        transformed_vals = self.transform_categorical_values_series(
            measurement_metadata, vals_df[col]
        )
        if transformed_vals is not None:
            self.events_df[vals_df] = transformed_vals
            return

        # 2. Eliminates hard outliers and performs censoring via specified config.
        present_idx = ~vals_df[col].isna()

        if not present_idx.any(): return

        vals_df = vals_df[present_idx]

        inlier_col = f"{col}_is_inlier"
        vals_df[inlier_col] = pd.Series([np.NaN] * len(vals_df), dtype='boolean')

        with self._time_as('_transform_numerical_metadata_column_vals'):
            self.events_df.loc[
                vals_df.index,
                vals_df.columns
            ] = EventStreamDataset._transform_numerical_metadata_column_vals(
                vals_df, measurement_metadata, col, inlier_col
            )

    @TimeableMixin.TimeAs(key='_transform_numerical_metadata_column_vals')
    def _transform_numerical_metadata_column_vals_gp(
        self, vals: pd.DataFrame, measurement_metadata: pd.DataFrame, val_col: str, inlier_col: str,
    ) -> pd.Series:
        return EventStreamDataset._transform_numerical_metadata_column_vals(
            vals, measurement_metadata.loc[vals.name], val_col, inlier_col
        )

    @staticmethod
    def _transform_numerical_metadata_column_vals(
        vals_df: pd.DataFrame, measurement_metadata: pd.Series, val_col: str, inlier_col: str,
    ) -> pd.DataFrame:
        """
        Transforms a particular numerical metadata column, with key `key_col`.

        Performs the following steps:
            1. Transforms keys to categorical representations for categorical keys.
            2. Eliminates any values associated with dropped or categorical keys.
            3. Eliminates hard outliers and performs censoring via specified config.
            4. Convert values to desired types.
            5. Add inlier/outlier indices and remove learned outliers.
            6. Normalize values.

        Args:
            `key_col` (`str`): The column name of the governing key column.
            `val_col` (`str`): The column name of the underlying values column.
        """

        if measurement_metadata['value_type'] in {
            NumericDataModalitySubtype.DROPPED,
            NumericDataModalitySubtype.CATEGORICAL_INTEGER,
            NumericDataModalitySubtype.CATEGORICAL_FLOAT,
        }:
            vals_df[val_col] = np.NaN
            return

        vals_df[val_col] = EventStreamDataset.drop_or_censor_series(vals_df[val_col], measurement_metadata)

        # 4. Convert values to desired types:
        present_idx = ~vals_df[val_col].isna()
        if not present_idx.any(): return vals_df

        if measurement_metadata['value_type'] == NumericDataModalitySubtype.INTEGER:
            vals_df[val_col] = vals_df[val_col].round()

        # 5. Add inlier/outlier indices and remove learned outliers.
        if (
            ('outlier_model' in measurement_metadata.index) and
            (not pd.isnull(measurement_metadata['outlier_model']))
        ):
            inlier_idx = EventStreamDataset._get_is_inlier(
                vals_df.loc[present_idx, val_col], measurement_metadata
            )

            with warnings.catch_warnings():
                # This operation throws a Deprecation warning because inlier_idx is a different type than the
                # prior column in vals_df, so current behavior is to set this with a copy and new behavior
                # will be to set it in place. It actually isn't a new type though, but I think pandas doesn't
                # realize that for some reason. Making both columns simple 'bool's rather than 'boolean's
                # (which allow NaNs) solves the issue.
                warnings.simplefilter(action='ignore', category=DeprecationWarning)
                vals_df.loc[present_idx, inlier_col] = inlier_idx
            present_idx.loc[present_idx] = inlier_idx
            vals_df.loc[~present_idx, val_col] = np.NaN

            if not present_idx.any(): return vals_df

        # 6. Normalize values.
        if (
            ('normalizer' in measurement_metadata.index) and
            (not pd.isnull(measurement_metadata['normalizer']))
        ):
            vals_df.loc[present_idx, val_col] = EventStreamDataset._get_normalized_vals(
                vals_df.loc[present_idx, val_col], measurement_metadata
            )

        vals_df[val_col] = vals_df[val_col].astype(float)
        return vals_df

    @staticmethod
    def _get_is_inlier(vals: pd.Series, measurement_metadata: pd.Series) -> pd.Series:
        """
        Uses a pre-fit outlier detection model (if present) to return inlier predictions for a metadata
        column.

        Args:
            `vals` (`pd.Series`): The values to be processed.
            `key_col` (`str`): The metadata column in question.
        Returns:
            A boolean `pd.Series` with the same index and name as `vals` which is `True` if the value at that
            position is an inlier and `False` otherwise.
        """
        if (
            ('outlier_model' in measurement_metadata.index) and
            (not pd.isnull(measurement_metadata['outlier_model']))
        ):
            M = measurement_metadata['outlier_model']
            return pd.Series(
                M.predict(to_sklearn_np(vals)).reshape(-1)==1, index=vals.index, name=vals.name,
                dtype='boolean',
            )
        else:
            return pd.Series(
                [True for v in vals], index=vals.index, name=vals.name, dtype='boolean'
            )

    @staticmethod
    def _get_normalized_vals(vals: pd.Series, measurement_metadata: pd.Series) -> pd.Series:
        """
        Uses a pre-fit normalizer model (if present) to return normalized values for a metadata column.

        Args:
            `vals` (`pd.Series`): The values to be processed.
            `key_col` (`str`): The metadata column in question.
        Returns: A floating point `pd.Series` containing normalized versions of `vals`.
        """
        if (
            ('normalizer' in measurement_metadata.index) and
            (not pd.isnull(measurement_metadata['normalizer']))
        ):
            M = measurement_metadata['normalizer']
            return pd.Series(M.transform(to_sklearn_np(vals)).reshape(-1), index=vals.index, name=vals.name)
        else:
            return vals

import copy, torch, numpy as np, pandas as pd, polars as pl

from collections import defaultdict
from mixins import SaveableMixin, SeedableMixin, TimeableMixin
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .config import EventStreamPytorchDatasetConfig, VocabularyConfig
from .types import EventStreamPytorchBatch

DATA_ITEM_T = Dict[str, List[float]]

class EventStreamPytorchDataset(SaveableMixin, SeedableMixin, TimeableMixin, torch.utils.data.Dataset):
    """
    Ultimately, this class will produce a set of batches for ML that will have the following structure:
    {
        'event_mask': [batch_size X seq_len],
        'time': [batch_size X seq_len],

        'static_indices': [batch_size X N_static_data],
        'static_measurement_indices': [batch_size X N_static_data],

        'dynamic_indices': [batch_size X seq_len X N_data],
        'dynamic_values': [batch_size X seq_len X N_data],
        'dynamic_measurement_indices': [batch_size X seq_len X N_data],
        'dynamic_values_mask': [batch_size X seq_len X N_data],

        'stream_labels': {'task_name': [batch_size X N_labels]},
    }

    In this batch, the tensors will have the following types/properties:
    'event_mask'
        boolean type, represents whether or not an event is present at that index in the batch.
    'time'
        floating point type, represents the time in minutes since the start of that subject's data for the
        given event in the sequence.

    'dynamic_indices'
        integer type, represents the index of the particular data element measured for the subject's data
        at a given event in the sequence and data element within the event. This index is w.r.t. a unified
        vocabulary.
    'dynamic_values'
        float type, represents a numerical value associated with the data event at the corresponding index. If
        there is no associated numerical value with the given data event, will be `0`.
    'dynamic_measurement_indices'
        integer type, represents the index of the type of data element at the given index. Here, by `type`, we
        mean the underlying vocabulary.
    'dynamic_values_mask'
        boolean type, represents whether or not a data value is present at that position in the batch.

    'stream_labels'

    Note that the data elements, while ordered relative to one another, should be seen as an unordered
    set of data events overall, and processed accordingly.
    """

    TYPE_CHECKERS = {
        'multi_class_classification': [
            (pd.api.types.is_integer_dtype, None),
            (pd.api.types.is_categorical_dtype, lambda Y: Y.cat.codes.astype(int)),
        ],
        'binary_classification': [
            (pd.api.types.is_bool_dtype, lambda Y: Y.astype(float))
        ],
        'regression': [
            (pd.api.types.is_float_dtype, None),
        ],
    }

    @classmethod
    def normalize_task(cls, val: pd.Series) -> Tuple[str, pd.Series]:
        for task_type, checkers in cls.TYPE_CHECKERS.items():
            for check_fn, normalize_fn in checkers:
                if check_fn(val.dtype):
                    return task_type, (val if normalize_fn is None else normalize_fn(val))

        raise TypeError(f"Can't process label of {val.dtype} type!")

    def __init__(
        self,
        config: EventStreamPytorchDatasetConfig,
        split: Optional[str] = None,
        vocabulary_config: Optional[Union[VocabularyConfig, Path]] = None,
        data: Optional[Union[pl.DataFrame, Path]] = None,
        task_df: Optional[pd.DataFrame] = None,
    ):
        """
        Constructs the EventStreamPytorchDataset).
        Args:
            `data` (): The underlying source dataset.
            `config` (`EventStreamPytorchDatasetConfig`): Configuration options for the dataset.
            `task_df` (`Optional[pd.DataFrame]`, *optional*, defaults to `None`):
                If specified, this dictates how the raw data is sampled to produce batches and task labels
                (assumed to be stream-prediction level labels) are added to the output batches on the basis of
                the columns in `task_df`. `task_df` must have columns
                    * `subject_id`
                    * `start_time`
                    * `end_time`
                All other columns will be treated as labels and will be normalized then represented in the
                batch.
        """
        super().__init__()

        self.config = config

        if data is None:
            assert vocabulary_config is None
            assert self.config.save_dir is not None
            assert split is not None

            data = self.config.save_dir / 'DL_reps' / f'{split}.parquet'
            vocabulary_config = self.config.save_dir / 'vocabulary_config.json'
        else:
            assert vocabulary_config is not None
            assert split is None

        self.split = split

        if isinstance(data, pl.DataFrame): self.cached_data = data
        elif isinstance(data, Path): self.cached_data = pl.read_parquet(data)
        else: raise TypeError(f"Can't process data of type {type(data)}!")

        if isinstance(vocabulary_config, VocabularyConfig): self.vocabulary_config = vocabulary_config
        elif isinstance(vocabulary_config, Path):
            self.vocabulary_config = VocabularyConfig.from_json_file(vocabulary_config)
        else: raise TypeError(f"Can't process vocabulary_config of type {type(vocabulary_config)}!")

        self.do_produce_static_data = 'static_indices' in self.cached_data
        self.seq_padding_side = config.seq_padding_side
        self.max_seq_len = config.max_seq_len

        self.cached_data = self.cached_data.filter(pl.col('time').arr.lengths() >= config.min_seq_len)

        self.subject_ids = list(self.cached_data['subject_id'])

        self.task_types = {}
        self.task_vocabs = {}
        if task_df is None:
            self.task_df = None
            self.tasks = None
        else:
            self.task_df = task_df[task_df.subject_id.isin(self.subject_ids)].copy()
            self.tasks = [c for c in self.task_df if c not in ['subject_id', 'start_time', 'end_time']]
            task_schema = {'subject_id': self.cached_data['subject_id'].dtype}
            for t in self.tasks:
                task_type, normalized_vals = self.normalize_task(self.task_df[t])
                if task_type == 'multi_class_classification':
                    if pd.api.types.is_categorical_dtype(self.task_df[t]):
                        self.task_vocabs[t] = list(self.task_df[t].cat.categories.values)
                    else:
                        self.task_vocabs[t] = list(range(self.task_df[t].max()+1))
                elif task_type == 'binary_classification':
                    self.task_vocabs[t] = [False, True]
                    task_schema[t] = pl.Boolean

                self.task_types[t] = task_type
                self.task_df[t] = normalized_vals

            T_df = pl.from_pandas(self.task_df, schema_overrides=task_schema).with_row_count('task_row_num')
            time_dep_cols = ['time', 'dynamic_indices', 'dynamic_values', 'dynamic_measurement_indices']

            start_idx_expr = pl.col('time').arr.explode().search_sorted(pl.col('start_time_min').first())
            end_idx_expr = pl.col('time').arr.explode().search_sorted(pl.col('end_time_min').first())

            self.cached_data = self.cached_data.join(
                T_df, on='subject_id', how='inner', suffix='_task'
            ).with_columns(
                start_time_min = (pl.col('start_time_task') - pl.col('start_time')) / np.timedelta64(1, 'm'),
                end_time_min = (pl.col('end_time') - pl.col('start_time')) / np.timedelta64(1, 'm'),
            ).with_row_count(
                '__id'
            ).groupby('__id').agg(
                pl.col('task_row_num').first(),
                **{c: pl.col(c).first() for c in self.cached_data.columns if c not in time_dep_cols},
                **{c: pl.col(c).first() for c in self.tasks},
                **{
                    t: pl.col(t).arr.explode().slice(start_idx_expr, end_idx_expr - start_idx_expr)
                    for t in time_dep_cols
                },
            ).drop('__id')

            self.cached_data = self.cached_data.filter(
                pl.col('time').arr.lengths() >= config.min_seq_len
            ).sort('task_row_num').drop('task_row_num')

        with self._time_as('convert_to_rows'):
            self.cached_data = self.cached_data.drop('subject_id', 'start_time')
            self.columns = self.cached_data.columns
            self.cached_data = self.cached_data.rows()

    @property
    def has_task(self) -> bool: return (self.task_df is not None)

    def __len__(self): return len(self.cached_data)

    @SeedableMixin.WithSeed
    def subset(self, subset_size: int, do_replace: bool = False) -> 'EventStreamPytorchDataset':
        sub_idx = np.random.choice(len(self), subset_size, replace=do_replace)

        out = copy.deepcopy(self)
        if self.has_task:
            out.task_df = self.task_df.iloc[sub_idx]

        out.cached_data = [out.cached_data[i] for i in sub_idx]
        assert len(out) == subset_size

        return out

    def __getitem__(self, idx: int) -> Dict[str, list]: return self._seeded_getitem(idx)

    @SeedableMixin.WithSeed
    @TimeableMixin.TimeAs
    def _seeded_getitem(self, idx: int) -> Dict[str, list]:
        """
        Returns a dictionary corresponding to a batch element for a single patient. This will not be
        tensorized as that work will need to be re-done in the collate function regardless. The output will
        have structure:
        {
            'time': [seq_len],
            'dynamic_indices': [seq_len, n_data_per_event] (ragged),
            'dynamic_values': [seq_len, n_data_per_event] (ragged),
            'dynamic_measurement_indices': [seq_len, n_data_per_event] (ragged),
            'static_indices': [seq_len, n_data_per_event] (ragged),
            'static_measurement_indices': [seq_len, n_data_per_event] (ragged),
        }
          1. `time` captures the time of the sequence elements.
          2. `dynamic_indices` captures the categorical metadata elements listed in `self.data_cols` in a unified
             vocabulary space spanning all metadata vocabularies.
          3. `dynamic_values` captures the numerical metadata elements listed in `self.data_cols`. If no
             numerical elements are listed in `self.data_cols` for a given categorical column, the according
             index in this output will be `np.NaN`.
          4. `dynamic_measurement_indices` captures which metadata vocabulary was used to source a given data element.

        If `self.do_normalize_log_inter_event_times`, then `time` will be approximately modified as follows:
            1. `obs_TTE = time.diff()` Capture the observed inter_event_times.
            2. `mod_TTE = np.exp((np.log(obs_TTE + 1) - self.mean_log_inter...)/self.std_log_inter...)`:
               Modify the times between events by first padding them so none are <= 0, then ensuring that
               their log has mean 0 and standard deviation 1, then re-exponentiating them out of the log
               space.
            3. `mod_time = mod_TTE.cumsum()` Re-sum the modified inter-event times to get modified raw times.
        """

        full_subj_data = {c: v for c, v in zip(self.columns, self.cached_data[idx])}

        # If we need to truncate to `self.max_seq_len`, grab a random full-size span to capture that.
        # TODO(mmd): This will proportionally underweight the front and back ends of the subjects data
        # relative to the middle, as there are fewer full length sequences containing those elements.
        if len(full_subj_data['time']) > self.max_seq_len:
            with self._time_as('truncate_to_max_seq_len'):
                start_idx = np.random.choice(len(full_subj_data['time']) - self.max_seq_len)
                for k in ('time', 'dynamic_indices', 'dynamic_values', 'dynamic_measurement_indices'):
                    full_subj_data[k] = full_subj_data[k][start_idx:start_idx+self.max_seq_len]

        return full_subj_data

    def __static_and_dynamic_collate(self, batch: List[DATA_ITEM_T]) -> EventStreamPytorchBatch:
        out_batch = self.__dynamic_only_collate(batch)

        # Get the maximum number of static elements in the batch.
        max_n_static = max(len(e['static_indices']) for e in batch)

        # Walk through the batch and pad the associated tensors in all requisite dimensions.
        self._register_start('collate_static_padding')
        out = defaultdict(list)
        for e in batch:
            if self.do_produce_static_data:
                n_static = len(e['static_indices'])
                static_delta = max_n_static - n_static
                out['static_indices'].append(torch.nn.functional.pad(
                    torch.Tensor(e['static_indices']), (0, static_delta), value=np.NaN
                ))
                out['static_measurement_indices'].append(torch.nn.functional.pad(
                    torch.Tensor(e['static_measurement_indices']), (0, static_delta), value=np.NaN
                ))
        self._register_end('collate_static_padding')

        self._register_start('collate_static_post_padding')
        # Unsqueeze the padded tensors into the batch dimension and combine them.
        out = {k: torch.cat([T.unsqueeze(0) for T in Ts], dim=0) for k, Ts in out.items()}

        # Convert to the right types and add to the batch.
        out_batch['static_indices'] = torch.nan_to_num(out['static_indices'], nan=0).long()
        out_batch['static_measurement_indices'] = torch.nan_to_num(
            out['static_measurement_indices'], nan=0
        ).long()
        self._register_end('collate_static_post_padding')

        return out_batch

    def __dynamic_only_collate(self, batch: List[DATA_ITEM_T]) -> EventStreamPytorchBatch:
        # Get the local max sequence length and n_data elements for padding.
        max_seq_len = max(len(e['time']) for e in batch)

        max_n_data = 0
        for e in batch:
            for v in e['dynamic_indices']: max_n_data = max(max_n_data, len(v))
        if max_n_data == 0:
            raise ValueError(
                f"Batch has no dynamic measurements! Got:\n{batch[0]}\n{batch[1]}\n..."
            )

        # Walk through the batch and pad the associated tensors in all requisite dimensions.
        self._register_start('collate_dynamic_padding')
        out = defaultdict(list)
        for e in batch:
            seq_len   = len(e['time'])
            seq_delta = max_seq_len - seq_len

            if self.seq_padding_side == 'right':
                out['time'].append(torch.nn.functional.pad(
                    torch.Tensor(e['time']), (0, seq_delta), value=np.NaN
                ))
            else:
                out['time'].append(torch.nn.functional.pad(
                    torch.Tensor(e['time']), (seq_delta, 0), value=np.NaN
                ))

            data_elements = defaultdict(list)
            for k in ('dynamic_indices', 'dynamic_values', 'dynamic_measurement_indices'):
                for vs in e[k]:
                    if vs is None: vs = [np.NaN] * max_n_data

                    data_delta = max_n_data - len(vs)
                    vs = [v if v is not None else np.NaN for v in vs]

                    # We don't worry about seq_padding_side here as this is not the sequence dimension.
                    data_elements[k].append(torch.nn.functional.pad(
                        torch.Tensor(vs), (0, data_delta), value=np.NaN
                    ))

                if len(data_elements[k]) == 0:
                    raise ValueError(
                        f"Batch element has no {k}! Got:\n{e}."
                    )

                if self.seq_padding_side == 'right':
                    data_elements[k] = torch.nn.functional.pad(
                        torch.cat([T.unsqueeze(0) for T in data_elements[k]]),
                        (0, 0, 0, seq_delta), value=np.NaN
                    )
                else:
                    data_elements[k] = torch.nn.functional.pad(
                        torch.cat([T.unsqueeze(0) for T in data_elements[k]]),
                        (0, 0, seq_delta, 0), value=np.NaN
                    )

                out[k].append(data_elements[k])
        self._register_end('collate_dynamic_padding')

        self._register_start('collate_post_padding_processing')
        # Unsqueeze the padded tensors into the batch dimension and combine them.
        out_batch = {
            k: torch.cat([T.unsqueeze(0) for T in Ts], dim=0) for k, Ts in out.items()
        }

        # Add event and data masks on the basis of which elements are present, then convert the tensor
        # elements to the appropraite types.
        out_batch['event_mask'] = ~out_batch['time'].isnan()
        out_batch['dynamic_values_mask'] = ~out_batch['dynamic_values'].isnan()

        out_batch['time'] = torch.nan_to_num(out_batch['time'], nan=0)

        out_batch['dynamic_indices'] = torch.nan_to_num(out_batch['dynamic_indices'], nan=0).long()
        out_batch['dynamic_measurement_indices'] = torch.nan_to_num(
            out_batch['dynamic_measurement_indices'], nan=0
        ).long()
        out_batch['dynamic_values'] = torch.nan_to_num(out_batch['dynamic_values'], nan=0)

        out_batch = EventStreamPytorchBatch(**out_batch)
        self._register_end('collate_post_padding_processing')

        if not self.has_task: return out_batch

        self._register_start('collate_task_labels')
        out_labels = {}

        for task in self.tasks:
            task_type = self.task_types[task]

            out_labels[task] = []
            for e in batch: out_labels[task].append(e[task])

            match task_type:
                case 'multi_class_classification': out_labels[task] = torch.LongTensor(out_labels[task])
                case 'binary_classification': out_labels[task] = torch.FloatTensor(out_labels[task])
                case 'regression': out_labels[task] = torch.FloatTensor(out_labels[task])
                case _: raise TypeError(f"Don't know how to tensorify task of type {task_type}!")

        out_batch.stream_labels = out_labels
        self._register_end('collate_task_labels')

        return out_batch

    @TimeableMixin.TimeAs
    def collate(self, batch: List[DATA_ITEM_T]) -> EventStreamPytorchBatch:
        """ Combines the ragged dictionaries produced by __getitem__ into a tensorized batch."""
        if self.do_produce_static_data: return self.__static_and_dynamic_collate(batch)
        else: return self.__dynamic_only_collate(batch)

import copy, itertools, torch, numpy as np, pandas as pd

from collections import defaultdict
from datetime import datetime
from functools import cached_property
from mixins import SaveableMixin, SeedableMixin, TimeableMixin
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union

from .event_stream_dataset import EventStreamDataset
from .config import EventStreamPytorchDatasetConfig
from .types import DataModality, EventStreamPytorchBatch, TemporalityType

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

    TODO(mmd): finalize/improve.

    In this batch, the tensors will have the following types/properties:
    'event_mask'
        boolean type, represents whether or not an event is present at that index in the batch.
    'time'
        floating point type, represents the time in minutes since the start of that subject's data for the
        given event in the sequence.

    'dynamic_indices'
        integer type, represents the index of the particular data element measured for the subject's data
        at a given event in the sequence and data element within the event. This index is w.r.t. a unified
        vocabulary spanning all metadata vocabularies in the passed `EventStreamDataset`, `E`.
    'dynamic_values'
        float type, represents a numerical value associated with the data event at the corresponding index. If
        there is no associated numerical value with the given data event, will be `0`.
    'dynamic_measurement_indices'
        integer type, represents the index of the type of data element at the given index. Here, by `type`, we
        mean the underlying source metadata vocabulary from the passed `EventStreamDataset`, `E`.
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

    @classmethod
    def load_or_create(
        cls,
        data_dir: Path,
        config: EventStreamPytorchDatasetConfig,
        splits: Sequence[str],
        data: Optional[EventStreamDataset] = None,
        task_df: Optional[pd.DataFrame] = None,
        task_df_name: Optional[str] = None,
    ) -> 'EventStreamPytorchDataset':
        """
        Loads the dataset from disk if it exists, otherwise creates it and saves it to disk.
        """

        out = []
        for split in splits:
            if task_df_name is not None:
                pyd_fp = data_dir / f'{task_df_name}_{split}_pytorch_dataset.pkl'
            else:
                pyd_fp = data_dir / f'{split}_pytorch_dataset.pkl'
            if pyd_fp.is_file():
                start = datetime.now()
                loaded_pyd = cls._load(pyd_fp)
                print(f"Loaded pytorch dataset from {pyd_fp} in {datetime.now() - start}.")

                if loaded_pyd.max_seq_len != config.max_seq_len:
                    print(
                        f"Warning: max_seq_len mismatch for {split} split: "
                        f"{loaded_pyd.max_seq_len} vs {config.max_seq_len}!\n"
                        f"Overwriting with {config.max_seq_len}..."
                    )
                    loaded_pyd.max_seq_len = config.max_seq_len

                if loaded_pyd.do_normalize_log_inter_event_times != config.do_normalize_log_inter_event_times:
                    raise ValueError(
                        f"do_normalize_log_inter_event_times mismatch for {split} split: "
                        f"{loaded_pyd.do_normalize_log_inter_event_times} vs "
                        f"{config.do_normalize_log_inter_event_times}!"
                    )

                out.append(loaded_pyd)
                continue

            print(f"Creating pytorch dataset for {split} split...")

            if data is None:
                print(f"Loading event stream dataset...")
                start = datetime.now()
                data = EventStreamDataset._load(data_dir / 'processed_event_stream_dataset.pkl')
                print(f"Loaded event stream dataset in {datetime.now() - start}.")

            start = datetime.now()
            pyd = cls(data, config, split, task_df=task_df, task_df_name=task_df_name)
            print(f"Made pytorch dataset in {datetime.now() - start}.")

            start = datetime.now()
            pyd.save(data_dir)
            print(f"Saved pytorch dataset to {data_dir} in {datetime.now() - start}.")
            out.append(pyd)

        return out

    def save_path(self, data_dir: Path) -> Path:
        if self.has_task:
            return data_dir / f'{self.task_df_name}_{self.split}_pytorch_dataset.pkl'
        else:
            return data_dir / f'{self.split}_pytorch_dataset.pkl'

    def save(self, data_dir: Path, do_overwrite: bool = False):
        self._save(self.save_path(data_dir), do_overwrite=do_overwrite)

    def __init__(
        self,
        data: EventStreamDataset,
        config: EventStreamPytorchDatasetConfig,
        split: str,
        task_df: Optional[pd.DataFrame] = None,
        task_df_name: Optional[str] = None,
    ):
        """
        Constructs the EventStreamPytorchDataset).
        Args:
            `E` (`EventStreamDataset`): The underlying source dataset.
            `config` (`EventStreamPytorchDatasetConfig`): Configuration options for the dataset.
            `split` (`str`): Which data split (e.g., train, tuning, held_out) within `E` will be used.
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
        self.do_produce_static_data = data.has_static_measurements
        self.split = split

        self.event_types_idxmap = {et: i for i, et in enumerate(data.event_types)}

        self.event_cols = []
        self.static_cols = []
        self.dynamic_cols = []
        self.measurements_per_generative_mode = {modality: [] for modality in DataModality.values()}
        self.measurements_per_generative_mode[DataModality.SINGLE_LABEL_CLASSIFICATION].append('event_type')
        self.event_types_per_measurement = {'event_type': list(self.event_types_idxmap.keys())}

        for m, cfg in data.measurement_configs.items():
            if cfg.is_dropped: continue

            col = m
            if cfg.modality == DataModality.MULTIVARIATE_REGRESSION: col = (m, cfg.values_column)

            match cfg.temporality:
                case TemporalityType.FUNCTIONAL_TIME_DEPENDENT: self.event_cols.append(col)
                case TemporalityType.STATIC: self.static_cols.append(col)
                case TemporalityType.DYNAMIC:
                    self.dynamic_cols.append(col)
                    self.measurements_per_generative_mode[cfg.modality].append(m)
                    if cfg.modality == DataModality.MULTIVARIATE_REGRESSION:
                        self.measurements_per_generative_mode[
                            DataModality.MULTI_LABEL_CLASSIFICATION
                        ].append(m)

                    if cfg.present_in_event_types is None:
                        self.event_types_per_measurement[m] = list(
                            self.event_types_idxmap.keys()
                        )
                    else:
                        self.event_types_per_measurement[m] = list(cfg.present_in_event_types)

        self.measurements_per_generative_mode = {
            k: v for k, v in self.measurements_per_generative_mode.items() if v
        }
        self.data_cols = self.dynamic_cols + self.event_cols + self.static_cols

        self.seq_padding_side = config.seq_padding_side
        assert self.seq_padding_side in ('left', 'right'), f"{self.seq_padding_side} is invalid!"

        self.max_seq_len = config.max_seq_len
        self.do_normalize_log_inter_event_times = config.do_normalize_log_inter_event_times
        self.mean_log_inter_event_time_min = data.train_mean_log_inter_event_time_min
        self.std_log_inter_event_time_min = data.train_std_log_inter_event_time_min

        # We need thse to be ordered, so we ensure this is a list.
        self.subject_ids = [
            sid for sid in data.subject_ids_for_split(split) \
                if data.n_events_per_subject[sid] >= config.min_seq_len
        ]

        # event_type is tracked differently than other metadata elements, currently. TODO(mmd): Probably
        # sub-optimal.

        # Everything starts at 1 as we reserve 0 for padding.
        self.vocab_sizes_by_measurement = {
            'event_type': len(self.event_types_idxmap),
            **{k: len(v) for k, v in data.measurement_vocabs.items()},
        }

        self.measurement_vocab_offsets = {'event_type': 1}
        self.measurements_idxmap = {'event_type': 1}
        curr_end_of_vocab = len(self.event_types_idxmap) + 1

        # TODO(mmd): This is inconsistent with the logic in the config.
        for i, col in enumerate(self.data_cols):
            if type(col) is tuple: col  = col[0]
            self.measurements_idxmap[col] = i+2 # +2 to account for UNK (0) and event_type (1)

            self.measurement_vocab_offsets[col] = curr_end_of_vocab

            if data.measurement_configs[col].vocabulary is None:
                if data.measurement_configs[col].modality != DataModality.UNIVARIATE_REGRESSION:
                    raise ValueError(
                        f"Observed inappropriate config! {col} has no vocabulary but is not univariate "
                        f"regression, is {data.measurement_configs[col].modality}! Full config:\n"
                        f"{data.measurement_configs[col]}"
                    )
                vocab_size = 1
            else:
                vocab_size = len(data.measurement_idxmaps[col])
            curr_end_of_vocab += vocab_size

        self.total_vocab_size   = curr_end_of_vocab
        self.total_n_measurements = len(self.measurements_idxmap)

        self.task_types = {}
        self.task_vocabs = {}
        if task_df is None:
            self.task_df = None
            self.tasks = None
            assert task_df_name is None
            self.task_df_name = None
        else:
            assert task_df_name is not None
            self.task_df_name = task_df_name
            self.task_df = task_df[task_df.subject_id.isin(self.subject_ids)].copy()
            self.tasks = [c for c in self.task_df if c not in ['subject_id', 'start_time', 'end_time']]
            for t in self.tasks:
                task_type, normalized_vals = self.normalize_task(self.task_df[t])
                if task_type == 'multi_class_classification':
                    if pd.api.types.is_categorical_dtype(self.task_df[t]):
                        self.task_vocabs[t] = list(self.task_df[t].cat.categories.values)
                    else:
                        self.task_vocabs[t] = list(range(self.task_df[t].max()+1))
                elif task_type == 'binary_classification':
                    self.task_vocabs[t] = [False, True]

                self.task_types[t] = task_type
                self.task_df[t] = normalized_vals

        self._pre_cache_data_items(data)

    @property
    def has_task(self) -> bool: return (self.task_df is not None)

    def __len__(self): return len(self.task_df) if self.has_task else len(self.subject_ids)

    @SeedableMixin.WithSeed
    def subset(self, subset_size: int) -> 'EventStreamPytorchDataset':
        all_idx = np.arange(len(self.schema))
        sub_idx = np.random.permutation(all_idx)[:subset_size]

        out = copy.deepcopy(self)
        if self.has_task:
            out.task_df = self.task_df.iloc[sub_idx]
        else:
            out.subject_ids = [self.subject_ids[i] for i in sub_idx]
        del out.schema
        assert len(out) == subset_size

        out.cached_data = list(np.array(self.cached_data)[sub_idx])

        out.split = f"{self.split}_subset_{subset_size}"
        return out

    @cached_property
    def schema(self):
        if self.has_task:
            return list(
                self.task_df.apply(lambda r: (r.subject_id, r.start_time, r.end_time), axis='columns')
            )
        else:
            return [(sid, None, None) for sid in self.subject_ids]


    @TimeableMixin.TimeAs
    def _pre_cache_data_items(
        self,
        data: EventStreamDataset,
    ):
        """
        Pre-caches processed versions of the underlying data so that subsequent __get_item__ calls are faster.
        """

        self.cached_data = []

        for subject_id, start_time, end_time in tqdm(self.schema, leave=False):
            # First find the subject corresponding to this dataset element.
            with self._time_as('pre_cache_get_subj_data'):
                subj_data = data.events_df[data.events_df.subject_id == subject_id]
                if start_time is not None:
                    subj_data = subj_data[subj_data.timestamp >= start_time]
                if end_time is not None:
                    subj_data = subj_data[subj_data.timestamp <= end_time]

            with self._time_as('pre_cache_get_subj_dynamic_data'):
                subj_metadata = data.metadata_df(subject_id=subject_id)

            start_time = subj_data.timestamp.min()

            # Now we need to produce the 4 tensor elements in this dataset:
            # time, dynamic_indices, dynamic_values, and dynamic_measurement_indices

            self._register_start('pre_cache_get_dynamic_item_contents')
            # Normalize time to the start of the sequence and convert it to minutes.
            time_min = (subj_data.timestamp - start_time) / pd.to_timedelta(1, unit='minute')

            if self.do_normalize_log_inter_event_times:
                time_deltas = time_min.diff()
                # We do the + 1 here because it is possible that events have the same timestamp. This should be
                # mirrored in the calculation of mean_/std_log_inter_event_time_min.
                time_deltas = np.exp(
                    (np.log(time_deltas + 1) - self.mean_log_inter_event_time_min) /
                    self.std_log_inter_event_time_min
                )
                time_min = time_deltas.fillna(0).cumsum()

            # For data elements, we'll build a ragged representation for now,
            # then will convert everything to padded tensors in the collate function.

            dynamic_indices, dynamic_values, dynamic_measurement_indices = [], [], []
            # TODO(mmd): convert to apply to make this faster
            for event_id, r in subj_data.iterrows():
                # The first metadata element will always be the event type, so we initialize with that. It has no
                # value associated with it.
                event_dynamic_indices = [
                    self.measurement_vocab_offsets['event_type'] + self.event_types_idxmap[r['event_type']]
                ]
                event_dynamic_values = [np.NaN]
                event_dynamic_measurement_indices = [self.measurements_idxmap['event_type']]

                self._register_start('pre_cache_get_dynamic_item_contents__metadata_select')
                metadata = subj_metadata[subj_metadata.event_id == event_id]
                self._register_end('pre_cache_get_dynamic_item_contents__metadata_select')

                self._register_start('pre_cache_get_dynamic_item_contents__dynamic_cols')
                for col in self.dynamic_cols:
                    if type(col) is tuple: col, vals_col = col
                    else: vals_col = None

                    # A given metadata column may not exist in this event, which is fine and just means it won't
                    # contribute to the data values for this event, so we can continue.
                    if col not in metadata.columns: continue

                    # As we normalize everything to a single vocabulary, we need to grab the offset here.
                    offset = self.measurement_vocab_offsets[col]

                    vals = metadata[col].values
                    # Some values may be nested sequences, which we need to flatten.
                    if type(vals[0]) in (list, tuple): vals = list(itertools.chain.from_iterable(vals))

                    # Some values may be missing, which we need to ignore. We don't use pandas or numpy here as
                    # the type of vals is just a list.
                    vals_valid_idx = np.array([not pd.isnull(v) for v in vals])
                    vals = [v for v, b in zip(vals, vals_valid_idx) if b]

                    # As 0 is a sentinel vocabulary element of 'UNK' in all vocabularies, that is what we use of
                    # we don't find the associated key in the metadata idxmap.
                    new_indices = [
                        data.measurement_idxmaps[col].get(v, 0) + offset for v in vals
                    ]
                    event_dynamic_indices.extend(new_indices)
                    event_dynamic_measurement_indices.extend([self.measurements_idxmap[col] for _ in new_indices])

                    if (vals_col is None) or (vals_col not in metadata.columns):
                        event_dynamic_values.extend([np.NaN for _ in new_indices])
                    else:
                        # We normalize infinite values to missing values here as well. TODO(mmd): Log this?
                        event_dynamic_values.extend(
                            [np.NaN if v in (float('inf'), -float('inf')) else v for v, b in zip(
                                #M[vals_col], vals_valid_idx
                                metadata[vals_col], vals_valid_idx
                            ) if b]
                        )
                self._register_end('pre_cache_get_dynamic_item_contents__dynamic_cols')

                self._register_start('pre_cache_get_dynamic_item_contents__event_cols')
                for col in self.event_cols:
                    if type(col) is tuple: col, vals_col = col
                    else: vals_col = None

                    # A given metadata column may not exist in this event, which is fine and just means it won't
                    # contribute to the data values for this event, so we can continue.
                    if col not in r: continue

                    # As we normalize everything to a single vocabulary, we need to grab the offset here.
                    offset = self.measurement_vocab_offsets[col]

                    val = r[col]

                    if data.measurement_configs[col].modality == DataModality.UNIVARIATE_REGRESSION:
                        new_index = offset
                        new_val = val
                    else:
                        # As 0 is a sentinel vocabulary element of 'UNK' in all vocabularies, that is what we use
                        # of we don't find the associated key in the metadata idxmap.
                        new_index = data.measurement_idxmaps[col].get(val, 0) + offset
                        if (vals_col is None) or (vals_col not in metadata.columns): new_val = np.NaN
                        else: new_val = r[vals_col]

                    event_dynamic_indices.append(new_index)
                    event_dynamic_measurement_indices.append(self.measurements_idxmap[col])
                    event_dynamic_values.append(
                        np.NaN if new_val in (float('inf'), -float('inf')) else new_val
                    )
                self._register_end('pre_cache_get_dynamic_item_contents__event_cols')

                dynamic_indices.append(event_dynamic_indices)
                dynamic_values.append(event_dynamic_values)
                dynamic_measurement_indices.append(event_dynamic_measurement_indices)
            self._register_end('pre_cache_get_dynamic_item_contents')

            if not self.do_produce_static_data:
                out = {
                    'time': time_min.values,
                    'dynamic_indices': dynamic_indices,
                    'dynamic_values': dynamic_values,
                    'dynamic_measurement_indices': dynamic_measurement_indices,
                }
            else:
                self._register_start('pre_cache_get_static_item_contents')
                subj_static_data = data.subjects_df.loc[subject_id]

                static_indices = []
                static_measurement_indices = []
                for col in self.static_cols:
                    # A given metadata column may not exist in this event, which is fine and just means it
                    # won't contribute to the data values for this event, so we can continue.
                    if col not in subj_static_data.index: continue

                    # As we normalize everything to a single vocabulary, we need to grab the offset here.
                    offset = self.measurement_vocab_offsets[col]

                    val = subj_static_data[col]
                    static_indices.append(data.measurement_idxmaps[col].get(val, 0) + offset)
                    static_measurement_indices.append(self.measurements_idxmap[col])
                self._register_end('pre_cache_get_static_item_contents')

                out = {
                    'time': time_min.values,
                    'dynamic_indices': dynamic_indices,
                    'dynamic_values': dynamic_values,
                    'dynamic_measurement_indices': dynamic_measurement_indices,
                    'static_indices': static_indices,
                    'static_measurement_indices': static_measurement_indices,
                }
            self.cached_data.append(out)

    def __getitem__(self, idx: int) -> Dict[str, list]:
        batch = self._seeded_getitem_from_cached_idx(idx)

        if self.has_task:
            row = self.task_df.iloc[idx]
            for task in self.tasks: batch[task] = row[task]

        return batch

    @SeedableMixin.WithSeed
    @TimeableMixin.TimeAs
    def _seeded_getitem_from_cached_idx(self, idx: int) -> Dict[str, list]:
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

        full_subj_data = self.cached_data[idx]

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
                    data_delta = max_n_data - len(vs)

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

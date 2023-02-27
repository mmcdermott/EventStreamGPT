import itertools, torch, numpy as np, pandas as pd

from collections import defaultdict
from datetime import datetime
from functools import cached_property
from mixins import SeedableMixin

from .event_stream_dataset import EventStreamDataset
from .config import EventStreamPytorchDatasetConfig
from .types import DataModality, EventStreamPytorchBatch, TemporalityType

from typing import Dict, Hashable, List, Optional, Tuple, Union

DATA_ITEM_T = Dict[str, List[float]]

class EventStreamPytorchDataset(SeedableMixin, torch.utils.data.Dataset):
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

    def __init__(
        self,
        E: EventStreamDataset,
        config: EventStreamPytorchDatasetConfig,
        split: str,
        task_df: Optional[pd.DataFrame] = None,
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
        self.data = E

        self.seq_padding_side = config.seq_padding_side
        assert self.seq_padding_side in ('left', 'right'), f"{self.seq_padding_side} is invalid!"

        self.max_seq_len = config.max_seq_len
        self.do_normalize_log_inter_event_times = config.do_normalize_log_inter_event_times
        self.mean_log_inter_event_time_min = E.train_mean_log_inter_event_time_min
        self.std_log_inter_event_time_min = E.train_std_log_inter_event_time_min

        # We need thse to be ordered, so we ensure this is a list.
        self.subject_ids = [
            sid for sid in self.data.subject_ids_for_split(split) \
                if self.data.n_events_per_subject[sid] >= config.min_seq_len
        ]

        # event_type is tracked differently than other metadata elements, currently. TODO(mmd): Probably
        # sub-optimal.
        self.event_types_idxmap = {et: i for i, et in enumerate(self.data.event_types)}

        # Everything starts at 1 as we reserve 0 for padding.
        self.measurement_vocab_offsets = {'event_type': 1}
        self.measurements_idxmap = {'event_type': 1}
        curr_end_of_vocab = len(self.event_types_idxmap) + 1

        # TODO(mmd): This is inconsistent with the logic in the config.
        for i, col in enumerate(self.data_cols):
            if type(col) is tuple: col  = col[0]
            self.measurements_idxmap[col] = i+2 # +2 to account for UNK (0) and event_type (1)

            self.measurement_vocab_offsets[col] = curr_end_of_vocab

            if self.data.measurement_configs[col].vocabulary is None:
                if self.data.measurement_configs[col].modality != DataModality.UNIVARIATE_REGRESSION:
                    raise ValueError(
                        f"Observed inappropriate config! {col} has no vocabulary but is not univariate "
                        f"regression, is {self.data.measurement_configs[col].modality}! Full config:\n"
                        f"{self.data.measurement_configs[col]}"
                    )
                vocab_size = 1
            else:
                vocab_size = len(self.data.measurement_idxmaps[col])
            curr_end_of_vocab += vocab_size

        self.total_vocab_size   = curr_end_of_vocab
        self.total_n_measurements = len(self.measurements_idxmap)

        self.task_types = {}
        self.task_vocabs = {}
        if task_df is None:
            self.task_df = None
            self.tasks = None
        else:
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

    @property
    def do_produce_static_data(self): return self.data.has_static_measurements

    @cached_property
    def data_cols(self):
        return self.dynamic_cols + self.event_cols + self.static_cols

    @cached_property
    def event_cols(self):
        out = []
        for m, cfg in self.data.measurement_configs.items():
            if cfg.is_dropped: continue
            if cfg.temporality != TemporalityType.FUNCTIONAL_TIME_DEPENDENT: continue
            if cfg.modality == DataModality.MULTIVARIATE_REGRESSION: out.append((m, cfg.values_column))
            else: out.append(m)
        return out

    @cached_property
    def static_cols(self):
        out = []
        for m, cfg in self.data.measurement_configs.items():
            if cfg.is_dropped: continue
            if cfg.temporality != TemporalityType.STATIC: continue
            if cfg.modality == DataModality.MULTIVARIATE_REGRESSION: out.append((m, cfg.values_column))
            else: out.append(m)
        return out

    @cached_property
    def dynamic_cols(self):
        out = []
        for m, cfg in self.data.measurement_configs.items():
            if cfg.is_dropped: continue
            if cfg.temporality != TemporalityType.DYNAMIC: continue
            if cfg.modality == DataModality.MULTIVARIATE_REGRESSION: out.append((m, cfg.values_column))
            else: out.append(m)
        return out

    @cached_property
    def measurements_per_generative_mode(self) -> Dict[DataModality, List[str]]:
        # TODO(mmd): Should probably be unordered internally (e.g., defaultdict(set) vs. defaultdict(list))
        out = defaultdict(list)
        for dt in self.measurements_idxmap:
            if dt == 'event_type':
                out[DataModality.SINGLE_LABEL_CLASSIFICATION].append(dt)
                continue
            config = self.data.measurement_configs[dt]

            if config.temporality != TemporalityType.DYNAMIC:
                # These aren't generated.
                continue

            vt = self.data.measurement_configs[dt].modality
            out[vt].append(dt)
            if vt == DataModality.MULTIVARIATE_REGRESSION:
                out[DataModality.MULTI_LABEL_CLASSIFICATION].append(dt)

        return dict(out)

    @property
    def has_task(self) -> bool: return (self.task_df is not None)

    def __len__(self): return len(self.task_df) if self.has_task else len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, list]:
        if self.has_task:
            row = self.task_df.iloc[idx]
            batch = self._seeded_getitem_from_range(
                subject_id=row.subject_id, start_time=row.start_time, end_time=row.end_time
            )
            for task in self.tasks: batch[task] = row[task]
            return batch
        else:
            return self._seeded_getitem_from_range(self.subject_ids[idx])

    @SeedableMixin.WithSeed
    def _seeded_getitem_from_range(
        self,
        subject_id: Hashable,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, list]:
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
            'static_values': [seq_len, n_data_per_event] (ragged),
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

        # First find the subject corresponding to this dataset element.
        subj_data = self.data.events_df[self.data.events_df.subject_id == subject_id]
        if start_time is not None:
            subj_data = subj_data[subj_data.timestamp >= start_time]
        if end_time is not None:
            subj_data = subj_data[subj_data.timestamp <= end_time]

        subj_metadata = self.data.metadata_df(subject_id=subject_id)

        start_time = subj_data.timestamp.min()
        out = {}

        # If we need to truncate to `self.max_seq_len`, grab a random full-size span to capture that.
        # TODO(mmd): This will proportionally underweight the front and back ends of the subjects data
        # relative to the middle, as there are fewer full length sequences containing those elements.
        if len(subj_data) > self.max_seq_len:
            start_idx = np.random.choice(len(subj_data) - self.max_seq_len)
            subj_data = subj_data.iloc[start_idx:start_idx+self.max_seq_len]

        # Now we need to produce the 4 tensor elements in this dataset:
        # time, dynamic_indices, dynamic_values, and dynamic_measurement_indices

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
        out['time'] = time_min.values

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
            event_dynamic_values  = [np.NaN]
            event_dynamic_measurement_indices   = [self.measurements_idxmap['event_type']]

            metadata = subj_metadata[subj_metadata.event_id == event_id]

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
                    self.data.measurement_idxmaps[col].get(v, 0) + offset for v in vals
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

            for col in self.event_cols:
                if type(col) is tuple: col, vals_col = col
                else: vals_col = None

                # A given metadata column may not exist in this event, which is fine and just means it won't
                # contribute to the data values for this event, so we can continue.
                if col not in r: continue

                # As we normalize everything to a single vocabulary, we need to grab the offset here.
                offset = self.measurement_vocab_offsets[col]

                val = r[col]

                if self.data.measurement_configs[col].modality == DataModality.UNIVARIATE_REGRESSION:
                    new_index = offset
                    new_val = val
                else:
                    # As 0 is a sentinel vocabulary element of 'UNK' in all vocabularies, that is what we use
                    # of we don't find the associated key in the metadata idxmap.
                    new_index = self.data.measurement_idxmaps[col].get(val, 0) + offset
                    if (vals_col is None) or (vals_col not in metadata.columns): new_val = np.NaN
                    else: new_val = r[vals_col]

                event_dynamic_indices.append(new_index)
                event_dynamic_measurement_indices.append(self.measurements_idxmap[col])
                event_dynamic_values.append(
                    np.NaN if new_val in (float('inf'), -float('inf')) else new_val
                )

            dynamic_indices.append(event_dynamic_indices)
            dynamic_values.append(event_dynamic_values)
            dynamic_measurement_indices.append(event_dynamic_measurement_indices)

        if not self.do_produce_static_data:
            return {
                'time': time_min.values,
                'dynamic_indices': dynamic_indices,
                'dynamic_values': dynamic_values,
                'dynamic_measurement_indices': dynamic_measurement_indices,
            }

        subj_static_data = self.data.subjects_df.loc[subject_id]

        static_indices = []
        static_measurement_indices = []
        for col in self.static_cols:
            # A given metadata column may not exist in this event, which is fine and just means it won't
            # contribute to the data values for this event, so we can continue.
            if col not in subj_static_data.index: continue

            # As we normalize everything to a single vocabulary, we need to grab the offset here.
            offset = self.measurement_vocab_offsets[col]

            val = subj_static_data[col]
            static_indices.append(self.data.measurement_idxmaps[col].get(val, 0) + offset)
            static_measurement_indices.append(self.measurements_idxmap[col])

        return {
            'time': time_min.values,
            'dynamic_indices': dynamic_indices,
            'dynamic_values': dynamic_values,
            'dynamic_measurement_indices': dynamic_measurement_indices,
            'static_indices': static_indices,
            'static_measurement_indices': static_measurement_indices,
        }

    def __static_and_dynamic_collate(self, batch: List[DATA_ITEM_T]) -> EventStreamPytorchBatch:
        out_batch = self.__dynamic_only_collate(batch)

        # Get the maximum number of static elements in the batch.
        max_n_static = max(len(e['static_indices']) for e in batch)

        # Walk through the batch and pad the associated tensors in all requisite dimensions.
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

        # Unsqueeze the padded tensors into the batch dimension and combine them.
        out = {k: torch.cat([T.unsqueeze(0) for T in Ts], dim=0) for k, Ts in out.items()}

        # Convert to the right types and add to the batch.
        out_batch['static_indices'] = torch.nan_to_num(out['static_indices'], nan=0).long()
        out_batch['static_measurement_indices'] = torch.nan_to_num(
            out['static_measurement_indices'], nan=0
        ).long()

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

        if not self.has_task: return out_batch

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

        return out_batch

    def collate(self, batch: List[DATA_ITEM_T]) -> EventStreamPytorchBatch:
        """ Combines the ragged dictionaries produced by __getitem__ into a tensorized batch."""
        if self.do_produce_static_data: return self.__static_and_dynamic_collate(batch)
        else: return self.__dynamic_only_collate(batch)

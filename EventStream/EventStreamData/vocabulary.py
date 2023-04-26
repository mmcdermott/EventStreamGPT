from __future__ import annotations

import copy, dataclasses, numpy as np, pandas as pd

from collections import Counter
from functools import cached_property
from typing import Dict, Generic, List, Optional, Sequence, Set, TypeVar, Union

from ..utils import COUNT_OR_PROPORTION

VOCAB_ELEMENT = TypeVar('T')
NESTED_VOCAB_SEQUENCE = Union[VOCAB_ELEMENT, Sequence['NESTED_VOCAB_SEQUENCE']]
@dataclasses.dataclass
class Vocabulary(Generic[VOCAB_ELEMENT]):
    """
    Stores a vocabulary of observed elements of a type `VOCAB_ELEMENT`, alongside their relative frequencies.
    """

    # The vocabulary, beginning with 'UNK' and subsequently proceeding in order of most frequently observed to
    # least frequently observed.
    vocabulary: Optional[List[Union[str, VOCAB_ELEMENT]]] = None

    # The observed frequencies of elements of the vocabulary.
    obs_frequencies: Optional[np.ndarray] = None

    @cached_property
    def idxmap(self) -> Dict[VOCAB_ELEMENT, int]:
        """Returns a mapping from vocab element to vocabulary integer index."""
        return {v: i for i, v in enumerate(self.vocabulary)}

    @property
    def vocab_set(self) -> Set[VOCAB_ELEMENT]:
        """Returns as et representation of the vocabulary elements."""
        return set(self.idxmap.keys())

    def __getitem__(self, q: Union[int, VOCAB_ELEMENT]) -> Union[int, VOCAB_ELEMENT]:
        """
        Gets either the vocabulary element at the integer index `q` or the integer index corresponding to
        the vocabulary element `q`
        """
        if type(q) is int:
            return self.vocabulary[q]
        else:
            assert (type(q) in self.element_types) or (q == 'UNK')
            return self.idxmap.get(q, 0)

    def __len__(self): return len(self.vocabulary)

    def __eq__(self, other: 'Vocabulary'):
        return (
            (type(self) is type(other)) and
            (self.vocabulary == other.vocabulary) and
            (self.obs_frequencies.round(3) == other.obs_frequencies.round(3)).all()
        )

    def __post_init__(self):
        """Validates the vocabulary and sorts the vocabulary in the proper order."""
        assert len(self.vocabulary) > 0, "Empty vocabularies are not supported!"
        assert len(self.vocabulary) == len(self.obs_frequencies)

        vocab_set = set(self.vocabulary)
        assert len(self.vocabulary) == len(vocab_set)

        self.element_types = set(type(v) for v in self.vocabulary if v != 'UNK')
        assert int not in self.element_types, "Integer vocabularies are not supported."

        self.obs_frequencies = np.array(self.obs_frequencies)
        self.obs_frequencies = self.obs_frequencies / self.obs_frequencies.sum()

        vocab = copy.deepcopy(self.vocabulary)
        obs_frequencies = self.obs_frequencies

        if 'UNK' in vocab_set:
            unk_index = vocab.index('UNK')
            unk_freq = obs_frequencies[unk_index]
            obs_frequencies = np.delete(obs_frequencies, unk_index)
            del vocab[unk_index]
        else: unk_freq = 0

        idx = np.lexsort((vocab, obs_frequencies))[::-1]

        self.vocabulary = ['UNK'] + [vocab[i] for i in idx]
        self.obs_frequencies = np.concatenate(([unk_freq], obs_frequencies[idx]))

    def filter(self, total_observations: int, min_valid_element_freq: COUNT_OR_PROPORTION):
        """
        Filters the vocabulary elements to only those occurring sufficiently often, pushing the dropped
        elements into the `'UNK'` element.

        Args:
            `total_observations` (`int`): How many total observations were there of vocabulary elements.
            `min_valid_element_freq` (`COUNT_OR_PROPORTION`):
                How frequently must an element have been observed to be retained?
        """

        if type(min_valid_element_freq) is not float: min_valid_element_freq /= total_observations

        # np.searchsorted(a, v, side='right') returns i such that
        # a[i-1] <= v < a[i]
        # So, np.searchsorted(-self.obs_frequencies[1:], -min_valid_element_freq, side='left') returns i s.t.
        # -self.obs_frequencies[i+1-1] <= -min_valid_element_freq < -self.obs_frequencies[i+1]
        # <=>
        # self.obs_frequencies[i] >= min_valid_element_freq > self.obs_frequencies[i+1]
        # which is precisely the index i such that self.obs_frequencies[:i+1] are >= min_valid_element_freq
        # and self.obs_frequencies[i+1:] are < min_valid_element_freq
        idx = np.searchsorted(-self.obs_frequencies[1:], -min_valid_element_freq, side='right')

        # Now, we need to filter the vocabulary elements, but also put anything dropped in the UNK bucket.
        self.obs_frequencies[0] += self.obs_frequencies[idx+1:].sum()

        self.vocabulary = self.vocabulary[:idx+1]
        self.obs_frequencies = self.obs_frequencies[:idx+1]
        if hasattr(self, 'idxmap'): delattr(self, 'idxmap')

    @staticmethod
    def __nested_update_container(container: Union[set, Counter], val: NESTED_VOCAB_SEQUENCE):
        """
        If `val` is a scalar, adds `val` to `container`.
        If `val` is a sequence, then iterates through `val` and recursively adds its elements to `container`.
        """
        if isinstance(val, (float, np.float32, np.float64)) and np.isnan(val): return
        elif isinstance(val, (list, tuple, np.ndarray, pd.Series)):
            for v in val: Vocabulary.__nested_update_container(container, v)
        else: container.update([val])

    @classmethod
    def build_vocab(cls, observations: NESTED_VOCAB_SEQUENCE) -> 'Vocabulary':
        """Builds a vocabulary from a set of observed elements."""
        counter = Counter()
        cls.__nested_update_container(counter, observations)

        vocab = list(counter.keys())
        freq = [counter[k] / len(observations) for k in vocab]
        return cls(vocabulary=vocab, obs_frequencies=freq)

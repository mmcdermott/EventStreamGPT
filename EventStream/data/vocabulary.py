from __future__ import annotations

import copy
import dataclasses
import math
from collections import Counter
from functools import cached_property
from io import TextIOBase
from textwrap import shorten, wrap
from typing import Dict, Generic, List, Optional, Sequence, Set, TypeVar, Union

import numpy as np
import pandas as pd
from sparklines import sparklines

from ..utils import COUNT_OR_PROPORTION, num_initial_spaces

VOCAB_ELEMENT = TypeVar("T")
NESTED_VOCAB_SEQUENCE = Union[VOCAB_ELEMENT, Sequence["NESTED_VOCAB_SEQUENCE"]]


@dataclasses.dataclass
class Vocabulary(Generic[VOCAB_ELEMENT]):
    """Stores a vocabulary of observed elements of a type `VOCAB_ELEMENT`, alongside their relative
    frequencies."""

    # The vocabulary, beginning with 'UNK' and subsequently proceeding in order of most frequently observed to
    # least frequently observed.
    vocabulary: list[str | VOCAB_ELEMENT] | None = None

    # The observed frequencies of elements of the vocabulary.
    obs_frequencies: np.ndarray | None = None

    @cached_property
    def idxmap(self) -> dict[VOCAB_ELEMENT, int]:
        """Returns a mapping from vocab element to vocabulary integer index."""
        return {v: i for i, v in enumerate(self.vocabulary)}

    @property
    def vocab_set(self) -> set[VOCAB_ELEMENT]:
        """Returns as et representation of the vocabulary elements."""
        return set(self.idxmap.keys())

    def __getitem__(self, q: int | VOCAB_ELEMENT) -> int | VOCAB_ELEMENT:
        """Gets either the vocabulary element at the integer index `q` or the integer index
        corresponding to the vocabulary element `q`"""
        if type(q) is int:
            return self.vocabulary[q]
        else:
            assert (type(q) in self.element_types) or (q == "UNK")
            return self.idxmap.get(q, 0)

    def __len__(self):
        return len(self.vocabulary)

    def __eq__(self, other: Vocabulary):
        return (
            (type(self) is type(other))
            and (self.vocabulary == other.vocabulary)
            and (self.obs_frequencies.round(3) == other.obs_frequencies.round(3)).all()
        )

    def __post_init__(self):
        """Validates the vocabulary and sorts the vocabulary in the proper order."""
        assert len(self.vocabulary) > 0, "Empty vocabularies are not supported!"
        assert len(self.vocabulary) == len(self.obs_frequencies)

        vocab_set = set(self.vocabulary)
        assert len(self.vocabulary) == len(vocab_set)

        self.element_types = {type(v) for v in self.vocabulary if v != "UNK"}
        assert int not in self.element_types, "Integer vocabularies are not supported."

        self.obs_frequencies = np.array(self.obs_frequencies)
        self.obs_frequencies = self.obs_frequencies / self.obs_frequencies.sum()

        vocab = copy.deepcopy(self.vocabulary)
        obs_frequencies = self.obs_frequencies

        if "UNK" in vocab_set:
            unk_index = vocab.index("UNK")
            unk_freq = obs_frequencies[unk_index]
            obs_frequencies = np.delete(obs_frequencies, unk_index)
            del vocab[unk_index]
        else:
            unk_freq = 0

        idx = np.lexsort((vocab, obs_frequencies))[::-1]

        self.vocabulary = ["UNK"] + [vocab[i] for i in idx]
        self.obs_frequencies = np.concatenate(([unk_freq], obs_frequencies[idx]))

    def filter(self, total_observations: int, min_valid_element_freq: COUNT_OR_PROPORTION):
        """Filters the vocabulary elements to only those occurring sufficiently often, pushing the
        dropped elements into the `'UNK'` element.

        Args:
            `total_observations` (`int`): How many total observations were there of vocabulary elements.
            `min_valid_element_freq` (`COUNT_OR_PROPORTION`):
                How frequently must an element have been observed to be retained?
        """

        if type(min_valid_element_freq) is not float:
            min_valid_element_freq /= total_observations

        # np.searchsorted(a, v, side='right') returns i such that
        # a[i-1] <= v < a[i]
        # So, np.searchsorted(-self.obs_frequencies[1:], -min_valid_element_freq, side='left') returns i s.t.
        # -self.obs_frequencies[i+1-1] <= -min_valid_element_freq < -self.obs_frequencies[i+1]
        # <=>
        # self.obs_frequencies[i] >= min_valid_element_freq > self.obs_frequencies[i+1]
        # which is precisely the index i such that self.obs_frequencies[:i+1] are >= min_valid_element_freq
        # and self.obs_frequencies[i+1:] are < min_valid_element_freq
        idx = np.searchsorted(-self.obs_frequencies[1:], -min_valid_element_freq, side="right")

        # Now, we need to filter the vocabulary elements, but also put anything dropped in the UNK bucket.
        self.obs_frequencies[0] += self.obs_frequencies[idx + 1 :].sum()

        self.vocabulary = self.vocabulary[: idx + 1]
        self.obs_frequencies = self.obs_frequencies[: idx + 1]
        if hasattr(self, "idxmap"):
            delattr(self, "idxmap")

    @staticmethod
    def __nested_update_container(container: set | Counter, val: NESTED_VOCAB_SEQUENCE):
        """If `val` is a scalar, adds `val` to `container`.

        If `val` is a sequence, then iterates through `val` and recursively adds its elements to
        `container`.
        """
        if isinstance(val, (float, np.float32, np.float64)) and np.isnan(val):
            return
        elif isinstance(val, (list, tuple, np.ndarray, pd.Series)):
            for v in val:
                Vocabulary.__nested_update_container(container, v)
        else:
            container.update([val])

    @classmethod
    def build_vocab(cls, observations: NESTED_VOCAB_SEQUENCE) -> Vocabulary:
        """Builds a vocabulary from a set of observed elements."""
        counter = Counter()
        cls.__nested_update_container(counter, observations)

        vocab = list(counter.keys())
        freq = [counter[k] / len(observations) for k in vocab]
        return cls(vocabulary=vocab, obs_frequencies=freq)

    def describe(
        self,
        line_width: int = 60,
        wrap_lines: bool = True,
        n_head: int = 3,
        n_tail: int = 2,
        stream: TextIOBase | None = None,
    ) -> int | None:
        lines = []
        lines.append(f"{len(self)} elements, {self.obs_frequencies[0]*100:.1f}% UNKs")

        sparkline_prefix = "Frequencies:"
        W = line_width - len(sparkline_prefix) - 2

        if W > len(self):
            freqs = self.obs_frequencies[1:]
        else:
            freqs = self.obs_frequencies[1 : len(self) : int(math.ceil(len(self) / W))]

        lines.append(f"{sparkline_prefix} {sparklines(freqs)[0]}")
        if len(self) - 1 <= (n_head + n_tail):
            lines.append("Elements:")
            for v, f in zip(self.vocabulary[1:], self.obs_frequencies[1:]):
                lines.append(f"  ({f*100:.1f}%) {v}")
        else:
            lines.append("Examples:")
            for i in range(n_head):
                lines.append(f"  ({self.obs_frequencies[i+1]*100:.1f}%) {self.vocabulary[i+1]}")
            lines.append("  ...")
            for i in range(n_tail):
                lines.append(
                    f"  ({self.obs_frequencies[-n_tail+i]*100:.1f}%) {self.vocabulary[-n_tail+i]}"
                )

        line_indents = [num_initial_spaces(line) for line in lines]
        if wrap_lines:
            lines = [
                wrap(line, width=line_width, initial_indent="", subsequent_indent=(" " * ind))
                for line, ind in zip(lines, line_indents)
            ]
        else:
            lines = [
                shorten(line, width=line_width, initial_indent=(" " * ind))
                for line, ind in zip(lines, line_indents)
            ]

        desc = "\n".join(lines)
        if stream is None:
            print(desc)
            return
        return stream.write(desc)

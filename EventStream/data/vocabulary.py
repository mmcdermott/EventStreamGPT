"""A vocabulary class for easy management of categorical data element options."""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Sequence
from functools import cached_property
from io import TextIOBase
from textwrap import shorten, wrap
from typing import Generic, TypeVar, Union

import numpy as np
from sparklines import sparklines

from ..utils import COUNT_OR_PROPORTION, num_initial_spaces

VOCAB_ELEMENT = TypeVar("T")
NESTED_VOCAB_SEQUENCE = Union[VOCAB_ELEMENT, Sequence["NESTED_VOCAB_SEQUENCE"]]


@dataclasses.dataclass
class Vocabulary(Generic[VOCAB_ELEMENT]):
    """Stores a vocabulary of observed elements of type `VOCAB_ELEMENT` ordered by frequency.

    This class represents a vocabulary of observed elements of specifiable type `VOCAB_ELEMENT`. All
    vocabularies include an "unknown" option, codified as the string `'UNK'`. Upon construction, the
    vocabulary is sorted in order of decreasing frequency. The vocabulary can also be described for a
    text-based visual representation of the contained elements and their relative frequency distribution.
    Vocabulary elements can be arbitrary types _except_ for integers.

    Attributes:
        vocabulary: The vocabulary, stored as a plain list, beginning with 'UNK' and subsequently proceeding
            in order of most frequently observed to least frequently observed.
        obs_frequencies: The observed frequencies of elements of the vocabulary, stored as a plain list.
        element_types: A set of the types of elements that are allowed in this vocabulary.

    Raises:
        ValueError: If an empty vocabulary is passed, a vocabulary with duplicates is passed, a vocabulary
            with integer elements is passed, or a vocabulary whose length differs from the passed observation
            frequencies.

    Examples:
        >>> vocab = Vocabulary(vocabulary=['apple', 'banana', 'UNK'], obs_frequencies=[3, 5, 2])
        >>> vocab.vocabulary
        ['UNK', 'banana', 'apple']
        >>> vocab.obs_frequencies
        [0.2, 0.5, 0.3]
        >>> len(vocab)
        3
        >>> vocab = Vocabulary(vocabulary=[], obs_frequencies=[])
        Traceback (most recent call last):
            ...
        ValueError: Empty vocabularies are not supported.
        >>> vocab = Vocabulary(vocabulary=['apple'], obs_frequencies=[1, 2])
        Traceback (most recent call last):
            ...
        ValueError: self.vocabulary and self.obs_frequencies must have the same length. Got 1 and 2.
        >>> vocab = Vocabulary(vocabulary=['apple', 'apple'], obs_frequencies=[1, 2])
        Traceback (most recent call last):
            ...
        ValueError: Vocabulary has duplicates. len(self.vocabulary) = 2, but len(set(self.vocabulary)) = 1.
        >>> vocab = Vocabulary(vocabulary=['apple', 1], obs_frequencies=[1, 2])
        Traceback (most recent call last):
            ...
        ValueError: Integer elements in the vocabulary are not supported.
    """

    # The vocabulary, beginning with 'UNK' and subsequently proceeding in order of most frequently observed to
    # least frequently observed.
    vocabulary: list[str | VOCAB_ELEMENT] | None = None

    # The observed frequencies of elements of the vocabulary.
    obs_frequencies: np.ndarray | list[float] | None = None

    @cached_property
    def idxmap(self) -> dict[VOCAB_ELEMENT, int]:
        """Returns a mapping from vocab element to vocabulary integer index.

        Returns:
            Dictionary mapping vocabulary elements to their index.

        Example:
            >>> vocab = Vocabulary(vocabulary=['apple', 'banana', 'UNK'], obs_frequencies=[3, 5, 2])
            >>> vocab.idxmap
            {'UNK': 0, 'banana': 1, 'apple': 2}
        """

        return {v: i for i, v in enumerate(self.vocabulary)}

    def __getitem__(self, q: int | VOCAB_ELEMENT) -> int | VOCAB_ELEMENT:
        """Gets vocabulary element or corresponding integer index for `q`.

        If `q` is an integer index, returns the vocabulary element at that index. If it is a valid type to be
        a member of the vocabulary, returns the integer index associated with that element, or 0 if that
        element is not in the vocabulary (0 corresponds to the UNK index, so this is appropriate).

        Args:
            q: Query to fetch either the vocabulary element or its index.

        Returns:
            Vocabulary element at index q if q is an integer.
            Index of the vocabulary element if q is a string.

        Raises:
            TypeError: if the query element is not an integer, the UNK sentinel value, or a member of the
                allowed types for this vocabulary (`self.element_types`).

        Example:
            >>> vocab = Vocabulary(vocabulary=['apple', 'banana', 'UNK'], obs_frequencies=[3, 5, 2])
            >>> vocab[1]
            'banana'
            >>> vocab['apple']
            2
            >>> vocab[3.4]
            Traceback (most recent call last):
                ...
            TypeError: Type <class 'float'> is not a valid type for this vocabulary.
        """

        if type(q) is int:
            return self.vocabulary[q]
        else:
            if (type(q) not in self.element_types) and (q != "UNK"):
                raise TypeError(f"Type {type(q)} is not a valid type for this vocabulary.")
            return self.idxmap.get(q, 0)

    def __len__(self):
        """Returns the length of the vocabulary, including UNK."""
        return len(self.vocabulary)

    def __eq__(self, other: Vocabulary):
        """Returns True if other is an identical vocabulary.

        Returns:
            True if the type of self and other match, if their vocabulary lists are identical, and if their
            observed frequencies list are identical up to a precision of 3 decimal points.
        """
        return (
            (type(self) is type(other))
            and (self.vocabulary == other.vocabulary)
            and (np.array(self.obs_frequencies).round(3) == np.array(other.obs_frequencies).round(3)).all()
        )

    def __post_init__(self):
        """Validates and sorts the vocabulary."""
        if len(self.vocabulary) == 0:
            raise ValueError("Empty vocabularies are not supported.")
        if len(self.vocabulary) != len(self.obs_frequencies):
            raise ValueError(
                "self.vocabulary and self.obs_frequencies must have the same length. Got "
                f"{len(self.vocabulary)} and {len(self.obs_frequencies)}."
            )

        vocab_set = set(self.vocabulary)
        if len(self.vocabulary) != len(vocab_set):
            raise ValueError(
                f"Vocabulary has duplicates. len(self.vocabulary) = {len(self.vocabulary)}, but "
                f"len(set(self.vocabulary)) = {len(vocab_set)}."
            )

        self.element_types = {type(v) for v in self.vocabulary if v != "UNK"}
        if int in self.element_types:
            raise ValueError("Integer elements in the vocabulary are not supported.")

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
        self.obs_frequencies = list(np.concatenate(([unk_freq], obs_frequencies[idx])))

    def filter(
        self, total_observations: int | None, min_valid_element_freq: COUNT_OR_PROPORTION | None
    ) -> Vocabulary:
        """Filters the vocabulary elements to only those occurring sufficiently often.

        Filters out infrequent elements from the vocabulary, pushing the dropped elements into the UNK
        element. The cutoff frequency can be specified either as an integral count or as a floating point
        proportion. If specified as a count, it will be converted to a proportion via `total_observations`, as
        the internal observed frequency list is stored in terms of frequencies, not counts. Even if UNK occurs
        in the original vocabulary with frequency below this cut off, it will be retained as it is the
        destination element for filtered elements, and its output frequency will be updated accordingly.

        Args:
            total_observations: How many total observations were there of vocabulary elements.
            min_valid_element_freq: How frequently must an element have been observed to be retained?

        Example:
            >>> vocab = Vocabulary(vocabulary=['apple', 'banana', 'UNK'], obs_frequencies=[5, 3, 2])
            >>> vocab.filter(total_observations=10, min_valid_element_freq=0.4)
            >>> vocab.vocabulary
            ['UNK', 'apple']
            >>> vocab.obs_frequencies
            [0.5, 0.5]
            >>> vocab = Vocabulary(vocabulary=['apple', 'banana', 'UNK'], obs_frequencies=[5, 3, 2])
            >>> vocab.filter(total_observations=10, min_valid_element_freq=None)
            >>> vocab.vocabulary
            ['UNK', 'apple', 'banana']
        """

        if min_valid_element_freq is None:
            return

        try:
            if 0 < min_valid_element_freq and min_valid_element_freq < 1:
                pass
            elif min_valid_element_freq >= 1 and min_valid_element_freq == round(min_valid_element_freq):
                min_valid_element_freq /= total_observations
            else:
                raise ValueError(
                    "Can only filter vocabularies by floats in (0, 1) or ints > 1; got "
                    f"{type(min_valid_element_freq)}({min_valid_element_freq}."
                )
        except TypeError as e:
            raise ValueError(
                "Can only filter vocabularies by floats in (0, 1) or ints > 1; got "
                f"{type(min_valid_element_freq)}({min_valid_element_freq}."
            ) from e

        # np.searchsorted(a, v, side='right') returns i such that
        # a[i-1] <= v < a[i]
        # So, np.searchsorted(-self.obs_frequencies[1:], -min_valid_element_freq, side='left') returns i s.t.
        # -self.obs_frequencies[i+1-1] <= -min_valid_element_freq < -self.obs_frequencies[i+1]
        # <=>
        # self.obs_frequencies[i] >= min_valid_element_freq > self.obs_frequencies[i+1]
        # which is precisely the index i such that self.obs_frequencies[:i+1] are >= min_valid_element_freq
        # and self.obs_frequencies[i+1:] are < min_valid_element_freq
        self.obs_frequencies = np.array(self.obs_frequencies)
        idx = np.searchsorted(-self.obs_frequencies[1:], -min_valid_element_freq, side="right")

        # Now, we need to filter the vocabulary elements, but also put anything dropped in the UNK bucket.
        self.obs_frequencies[0] += self.obs_frequencies[idx + 1 :].sum()

        self.vocabulary = self.vocabulary[: idx + 1]
        self.obs_frequencies = self.obs_frequencies[: idx + 1]
        if hasattr(self, "idxmap"):
            delattr(self, "idxmap")
        self.obs_frequencies = list(self.obs_frequencies)

    def describe(
        self,
        line_width: int = 60,
        wrap_lines: bool = True,
        n_head: int = 3,
        n_tail: int = 2,
        stream: TextIOBase | None = None,
    ) -> int | None:
        """Prints or outputs to a stream a text-based visual representation of the vocabulary.

        This both lists the head and tail of the vocabulary but also produces a sparklines representation of
        the relative frequency distribution of vocabulary elements observed. In the printed head and tail
        elements, UNK is skipped. If more elements are in the vocabulary than the printed elements, ellipsis
        will denote the skipped elements.

        Args:
            line_width: The maximum width of each line in the description.
            wrap_lines: Whether to wrap lines that exceed the `line_width`.
            n_head: The number of high-frequency elements to include in the description.
            n_tail: The number of low-frequency elements to include in the description.
            stream: The stream to write the description to. If `None`, the description is printed to stdout.

        Returns:
            The number of characters written to the stream if a stream was provided, otherwise `None`.

        Example:
            >>> vocab = Vocabulary(
            ...     vocabulary=['apple', 'banana', 'pear', 'UNK'],
            ...     obs_frequencies=[3, 4, 1, 2],
            ... )
            >>> vocab.describe(n_head=2, n_tail=1, wrap_lines=False)
            4 elements, 20.0% UNKs
            Frequencies: █▆▁
            Elements:
              (40.0%) banana
              (30.0%) apple
              (10.0%) pear
            >>> vocab.describe(n_head=1, n_tail=0, wrap_lines=False)
            4 elements, 20.0% UNKs
            Frequencies: █▆▁
            Examples:
              (40.0%) banana
              ...
            >>> vocab.describe(n_head=1, n_tail=0, wrap_lines=False, line_width=10)
            4 [...]
            [...]
            Examples:
              [...]
              ...
            >>> vocab.describe(n_head=1, n_tail=0, wrap_lines=True, line_width=10)
            4
            elements,
            20.0% UNKs
            Frequencie
            s:
            Examples:
              (40.0%)
              banana
              ...
        """

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
                lines.append(f"  ({self.obs_frequencies[-n_tail+i]*100:.1f}%) {self.vocabulary[-n_tail+i]}")

        line_indents = [num_initial_spaces(line) for line in lines]
        if wrap_lines:
            new_lines = []
            for line, ind in zip(lines, line_indents):
                new_lines.extend(
                    wrap(line, width=line_width, initial_indent="", subsequent_indent=(" " * ind))
                )
            lines = new_lines
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

import sys

sys.path.append("../..")

import unittest

import numpy as np

from EventStream.data.vocabulary import Vocabulary

from ..mixins import MLTypeEqualityCheckableMixin


class TestVocabulary(MLTypeEqualityCheckableMixin, unittest.TestCase):
    """Tests the `Vocabulary` class."""

    def test_vocabulary(self):
        vocab = Vocabulary(vocabulary=["foo", "bar", "baz"], obs_frequencies=[0.2, 0.7, 0.1])

        self.assertEqual(vocab.element_types, {str})
        self.assertEqual(vocab.vocabulary, ["UNK", "bar", "foo", "baz"])
        self.assertEqual(vocab.obs_frequencies, np.array([0, 0.7, 0.2, 0.1]))
        self.assertEqual(vocab.idxmap, {"UNK": 0, "bar": 1, "foo": 2, "baz": 3})
        self.assertEqual(vocab.vocab_set, {"UNK", "bar", "foo", "baz"})
        self.assertEqual(len(vocab), 4)
        self.assertEqual(vocab["bar"], 1)
        self.assertEqual(vocab[3], "baz")
        self.assertEqual(vocab["fizzz"], 0)
        with self.assertRaises(AssertionError, msg="Should error on type mismatches."):
            vocab[3.2]

        vocab.filter(total_observations=10, min_valid_element_freq=0.25)
        self.assertEqual(vocab.vocabulary, ["UNK", "bar"])
        self.assertEqual(vocab.obs_frequencies, np.array([0.3, 0.7]))
        self.assertEqual(vocab.idxmap, {"UNK": 0, "bar": 1})
        self.assertEqual(vocab.vocab_set, {"UNK", "bar"})
        self.assertEqual(len(vocab), 2)
        self.assertEqual(vocab["bar"], 1)
        with self.assertRaises(IndexError, msg="Can't retrieve elements out of range."):
            vocab[3]
        self.assertEqual(vocab["foo"], 0)

        vocab = Vocabulary(vocabulary=[3.5, "UNK", 100.0], obs_frequencies=[0.2, 0.3, 0.5])

        self.assertEqual(vocab.element_types, {float})
        self.assertEqual(vocab.vocabulary, ["UNK", 100.0, 3.5])
        self.assertEqual(vocab.obs_frequencies, np.array([0.3, 0.5, 0.2]))
        self.assertEqual(vocab.idxmap, {"UNK": 0, 100.0: 1, 3.5: 2})
        self.assertEqual(vocab.vocab_set, {"UNK", 100.0, 3.5})
        self.assertEqual(len(vocab), 3)
        self.assertEqual(vocab["UNK"], 0)
        self.assertEqual(vocab[100.0], 1)
        self.assertEqual(vocab[2], 3.5)
        self.assertEqual(vocab[33.42], 0)
        with self.assertRaises(AssertionError, msg="Should error on type mismatches."):
            vocab["fizz_bang"]

        vocab.filter(total_observations=3, min_valid_element_freq=1)
        self.assertEqual(vocab.vocabulary, ["UNK", 100.0])
        self.assertEqual(vocab.obs_frequencies, np.array([0.5, 0.5]))
        self.assertEqual(vocab.idxmap, {"UNK": 0, 100.0: 1})

        with self.assertRaises(AssertionError, msg="Vocab can't have duplicates."):
            vocab = Vocabulary(["foo", "foo", "bar"], [0.3, 0.3, 0.4])
        with self.assertRaises(AssertionError, msg="Vocab can't be empty"):
            vocab = Vocabulary([], [])
        with self.assertRaises(
            AssertionError, msg="Vocab and frequencies must be the same length."
        ):
            vocab = Vocabulary(["foo", "bar", "baz"], [0.3, 0.7])
        with self.assertRaises(AssertionError, msg="Vocab doesn't support integer vocabularies."):
            vocab = Vocabulary([3, 4], [0.3, 0.7])

    def test_build_vocab(self):
        got_vocab = Vocabulary.build_vocab(["foo", "foo", "foo", "bar", "bar"])
        want_vocab = Vocabulary(["foo", "bar"], [3 / 5, 2 / 5])
        self.assertEqual(want_vocab, got_vocab)

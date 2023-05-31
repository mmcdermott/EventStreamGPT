import sys

sys.path.append("../..")

import unittest

from EventStream.data.vocabulary import Vocabulary

from ..utils import MLTypeEqualityCheckableMixin


def rounded_obs_freq(v: Vocabulary) -> list[float]:
    return [round(x, 5) for x in v.obs_frequencies]


class TestVocabulary(MLTypeEqualityCheckableMixin, unittest.TestCase):
    """Tests the `Vocabulary` class."""

    def test_vocabulary(self):
        vocab = Vocabulary(vocabulary=["foo", "bar", "baz"], obs_frequencies=[0.2, 0.7, 0.1])

        self.assertEqual(vocab.element_types, {str})
        self.assertEqual(vocab.vocabulary, ["UNK", "bar", "foo", "baz"])
        self.assertEqual(rounded_obs_freq(vocab), [0, 0.7, 0.2, 0.1])
        self.assertEqual(vocab.idxmap, {"UNK": 0, "bar": 1, "foo": 2, "baz": 3})
        self.assertEqual(len(vocab), 4)
        self.assertEqual(vocab["bar"], 1)
        self.assertEqual(vocab[3], "baz")
        self.assertEqual(vocab["fizzz"], 0)
        with self.assertRaises(TypeError, msg="Should error on type mismatches."):
            vocab[3.2]

        vocab.filter(total_observations=10, min_valid_element_freq=0.25)
        self.assertEqual(vocab.vocabulary, ["UNK", "bar"])
        self.assertEqual(rounded_obs_freq(vocab), [0.3, 0.7])
        self.assertEqual(vocab.idxmap, {"UNK": 0, "bar": 1})
        self.assertEqual(len(vocab), 2)
        self.assertEqual(vocab["bar"], 1)
        with self.assertRaises(IndexError, msg="Can't retrieve elements out of range."):
            vocab[3]
        self.assertEqual(vocab["foo"], 0)

        vocab = Vocabulary(vocabulary=[3.5, "UNK", 100.0], obs_frequencies=[0.2, 0.3, 0.5])

        self.assertEqual(vocab.element_types, {float})
        self.assertEqual(vocab.vocabulary, ["UNK", 100.0, 3.5])
        self.assertEqual(rounded_obs_freq(vocab), [0.3, 0.5, 0.2])
        self.assertEqual(vocab.idxmap, {"UNK": 0, 100.0: 1, 3.5: 2})
        self.assertEqual(len(vocab), 3)
        self.assertEqual(vocab["UNK"], 0)
        self.assertEqual(vocab[100.0], 1)
        self.assertEqual(vocab[2], 3.5)
        self.assertEqual(vocab[33.42], 0)
        with self.assertRaises(TypeError, msg="Should error on type mismatches."):
            vocab["fizz_bang"]

        vocab.filter(total_observations=3, min_valid_element_freq=1)
        self.assertEqual(vocab.vocabulary, ["UNK", 100.0])
        self.assertEqual(rounded_obs_freq(vocab), [0.5, 0.5])
        self.assertEqual(vocab.idxmap, {"UNK": 0, 100.0: 1})

        with self.assertRaises(ValueError, msg="Vocab can't have duplicates."):
            vocab = Vocabulary(["foo", "foo", "bar"], [0.3, 0.3, 0.4])
        with self.assertRaises(ValueError, msg="Vocab can't be empty"):
            vocab = Vocabulary([], [])
        with self.assertRaises(ValueError, msg="Vocab and frequencies must be the same length."):
            vocab = Vocabulary(["foo", "bar", "baz"], [0.3, 0.7])
        with self.assertRaises(ValueError, msg="Vocab doesn't support integer vocabularies."):
            vocab = Vocabulary([3, 4], [0.3, 0.7])

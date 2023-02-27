import sys
sys.path.append('../..')

import unittest, numpy as np, pandas as pd

from EventStream.EventStreamData.expandable_df_dict import (
    ExpandableDfDict,
)

class TestExpandableDfDict(unittest.TestCase):
    def assertDataframeEqual(self, a, b, msg):
        try:
            pd.testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)

    def test_constructs(self):
        E = ExpandableDfDict()

        self.assertEqual(E.n_rows, 0)
        self.assertEqual(E.df_dict, {})

    def test_equality(self):
        E1 = ExpandableDfDict({'a': [3, np.NaN], 'b': [None, 5]})
        E2 = ExpandableDfDict({'a': [3, np.NaN], 'b': [None, 5]})

        self.assertEqual(E1, E2)

    def test_append(self):
        E = ExpandableDfDict()

        # Append with vals
        E.append(vals={'a': 3, 'b': 5})
        self.assertEqual(E.n_rows, 1)
        self.assertEqual(E.df(), pd.DataFrame({'a': [3], 'b': [5]}))

        # Append with other vals
        E.append(c=3, b=6)
        self.assertEqual(E.n_rows, 2)
        self.assertEqual(E.df(), pd.DataFrame({'a': [3, None], 'b': [5, 6], 'c': [None, 3]}))

    def test_extend(self):
        E = ExpandableDfDict()

        # Append with vals
        E.extend(vals={'a': [1, 2], 'b': [3, 4]})
        self.assertEqual(E.n_rows, 2)
        self.assertEqual(E.df(), pd.DataFrame({'a': [1, 2], 'b': [3, 4]}))

        # Append with other vals
        E.extend(c=[1, 3], b=['a', 'b'])
        self.assertEqual(E.n_rows, 4)
        self.assertEqual(
            E.df(),
            pd.DataFrame({
                'a': [1, 2, None, None],
                'b': [3, 4, 'a', 'b'],
                'c': [None, None, 1, 3]
            }),
        )

        with self.assertRaises(AssertionError):
            E.extend(a=[1, 4], b = [3])

    def test_concatenate(self):
        E1 = ExpandableDfDict({'a': [1, 2], 'b': ['a', 'b'], 'c': [0, 1]})
        E2 = ExpandableDfDict({'a': [3],    'c': [0]})
        E3 = ExpandableDfDict({'a': [4, 5], 'b': ['c', 'd'], 'd': ['A', 'B']})

        E = ExpandableDfDict.concatenate((E1, E2, E3))
        self.assertEqual(E.n_rows, 5)
        self.assertEqual(
            E.df(),
            pd.DataFrame({
                'a': [1, 2, 3, 4, 5],
                'b': ['a', 'b', None, 'c', 'd'],
                'c': [0, 1, 0, None, None],
                'd': [None, None, None, 'A', 'B'],
            }),
        )

if __name__ == '__main__': unittest.main()

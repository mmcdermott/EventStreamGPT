import sys
sys.path.append('..')

import math, torch, numpy as np, pandas as pd, polars as pl
from polars.testing import assert_frame_equal as assert_pl_frame_equal

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from EventStream.EventStreamData.config import EventStreamDatasetConfig, MeasurementConfig
from EventStream.EventStreamData.vocabulary import Vocabulary
from EventStream.EventStreamTransformer.config import StructuredEventStreamTransformerConfig
from EventStream.EventStreamTransformer.model_output import EventStreamTransformerOutputWithPast

ASSERT_FN = Callable[[Any, Any, Optional[str]], None]

class MLTypeEqualityCheckableMixin():
    """
    This mixin provides capability to `unittest.TestCase` submodules to check various common ML types for
    equality, including:
      * `torch.Tensor`, via `torch.testing.assert_close`
      * `pd.DataFrame`, via `pd.testing.assert_frame_equal`
      * `pd.Series`, via `pd.testing.assert_series_equal`
      * `np.ndarray`, via `np.testing.assert_array_equal`
    """

    EQ_TYPE_CHECKERS = {
        torch.Tensor: (torch.testing.assert_close, {'equal_nan': True}),
        pd.DataFrame: pd.testing.assert_frame_equal,
        pl.DataFrame: (assert_pl_frame_equal, {'check_column_order': False}),
        pd.Series: pd.testing.assert_series_equal,
        np.ndarray: np.testing.assert_allclose,
    }

    def _typedAssertEqualFntr(
        self, assert_fn: Union[ASSERT_FN, Tuple[ASSERT_FN, Dict[str, Any]]]
    ) -> ASSERT_FN:
        if type(assert_fn) is tuple: assert_fn, assert_kwargs = assert_fn
        else: assert_kwargs = {}

        def f(want: Any, got: Any, msg: Optional[str] = None):
            try: assert_fn(want, got, **assert_kwargs)
            except Exception as e:
                if msg is None: msg = ''
                msg = f"{msg}\nWant:\n{want}\nGot:\n{got}"
                raise self.failureException(msg) from e

        return f

    def assertNestedEqual(
        self, want: Any, got: Any, msg: Optional[str] = None, check_like: bool = False
    ):
        m = msg
        if m is None: m = "Values aren't equal"

        types_match = isinstance(want, type(got)) or isinstance(got, type(want))

        try: types_match = (want == got) or types_match
        except: pass

        self.assertTrue(types_match, msg=f"{m}: Want type {type(want)}, got type {type(got)}")

        if (type(want) is pd.DataFrame) and check_like:
            try: pd.testing.assert_frame_equal(want, got, check_like=True)
            except Exception as e:
                if m is None: m = ''
                m = f"{m}\nWant:\n{want}\nGot:\n{got}"
                raise self.failureException(m) from e
        elif isinstance(want, dict):
            self.assertTrue(isinstance(got, dict), msg=m)
            self.assertNestedDictEqual(want, got, msg=m, check_like=check_like)
        elif isinstance(want, torch.distributions.Distribution):
            self.assertTrue(isinstance(got, torch.distributions.Distribution), msg=m)
            self.assertDistributionsEqual(want, got, msg=m)
        elif isinstance(want, str):
            self.assertTrue(isinstance(got, str), msg=m)
            self.assertEqual(want, got, msg=m)
        elif isinstance(want, Sequence):
            self.assertTrue(isinstance(got, Sequence), msg=m)
            self.assertEqual(len(want), len(got), msg=m)
            if not m: m = "Sequences aren't equal"
            for i, (want_i, got_i) in enumerate(zip(want, got)):
                self.assertNestedEqual(want_i, got_i, msg=f"{m} (index {i})", check_like=check_like)
        elif isinstance(want, float):
            if math.isnan(want): self.assertTrue(math.isnan(got), msg=m)
            else:
                self.assertFalse(math.isnan(got), msg=m)
                self.assertEqual(want, got, msg=m)
        else:
            m = f"{m}: Want {want}, got {got}"
            self.assertEqual(want, got, msg=m)

    def assertNestedDictEqual(
        self, want: dict, got: dict, msg: Optional[str] = None, check_like: bool = False
    ):
        """
        This assers that two dicionaries are equal using nested assert checks for the internal values. It is
        useful so that we can compare dictionaries of tensors or arrays with the type-specific comparators.
        """

        self.assertIsInstance(want, dict, msg)
        self.assertIsInstance(got, dict, msg)
        self.assertEqual(set(want.keys()), set(got.keys()), msg)

        for k in want.keys():
            want_val = want[k]
            got_val = got[k]
            if msg: m = f"{msg} (key {k})"
            else: m = f"Dictionaries aren't equal (key {k})"

            self.assertNestedEqual(want_val, got_val, m)


    def assertDistributionsEqual(
        self,
        want: torch.distributions.Distribution,
        got: torch.distributions.Distribution,
        msg: Optional[str] = None,
    ):
        m_type = f"Type of distributions does not match! want {type(want)}, got {type(got)}"
        if msg is not None: m_type = f"{msg}: {m_type}"
        self.assertEqual(type(want), type(got), m_type)

        m_vars = f"Parameters of distributions does not match! want {vars(want)}, got {vars(got)}"
        if msg is not None: m_vars = f"{msg}: {m_vars}"
        self.assertNestedDictEqual(vars(want), vars(got), m_vars)

    def setUp(self):
        for val_type, assert_fn in self.EQ_TYPE_CHECKERS.items():
            fn = self._typedAssertEqualFntr(assert_fn)
            self.addTypeEqualityFunc(val_type, fn)

        super().setUp()

class ConfigComparisonsMixin(MLTypeEqualityCheckableMixin):
    """
    This mixin provides capability to `unittest.TestCase` submodules to compare configuation objects for
    equality.
    """

    def assert_type_and_vars_equal(self, want: object, got: object, msg: Optional[str] = None):
        self.assertEqual(type(want), type(got), msg)
        self.assertNestedDictEqual(vars(want), vars(got), msg, check_like=True)

    def assert_vocabulary_equal(self, want: Vocabulary, got: Vocabulary, msg: Optional[str] = None):
        self.assertEqual(type(want), type(got), msg)
        self.assertEqual(want.vocabulary, got.vocabulary, msg)
        self.assertEqual(want.obs_frequencies, got.obs_frequencies, msg)

    def assert_measurement_config_equal(
        self, want: MeasurementConfig, got: MeasurementConfig, msg: Optional[str] = None
    ):
        if msg is None: msg = 'MeasurementConfigs are not equal'
        self.assertEqual(type(want), type(got), f"{msg}: Types {type(want)} and {type(got)} don't match")

        want_less_metadata = vars(want).copy()
        want_metadata = want_less_metadata.pop('measurement_metadata')
        got_less_metadata = vars(got).copy()
        got_metadata = got_less_metadata.pop('measurement_metadata')

        self.assertNestedDictEqual(
            want_less_metadata, got_less_metadata, msg=f"{msg}: Non-metadata keys aren't equal.",
            check_like=True
        )

        if want_metadata is None: self.assertIsNone(got_metadata, msg=f"{msg}: got metadata is not None")
        elif isinstance(want_metadata, pd.DataFrame):
            self.assertTrue(
                isinstance(got_metadata, pd.DataFrame), msg=f"{msg}: got metadata is not a DataFrame"
            )
            want_idx = want_metadata.index
            got_idx = got_metadata.index

            self.assertEqual(set(want_idx), set(got_idx), msg)
            reordered_got = got_metadata.reindex(want_idx).copy()
            try: pd.testing.assert_frame_equal(want_metadata, reordered_got, check_like=True)
            except Exception as e:
                if msg is None: msg = ''
                msg = f"{msg}\nWant:\n{want_metadata}\nGot:\n{reordered_got}"
                raise self.failureException(msg) from e
        else:
            self.assertTrue(isinstance(want_metadata, pd.Series), msg=f"{msg}: want metadata is not a Series")
            self.assertTrue(isinstance(got_metadata, pd.Series), msg=f"{msg}: got metadata is not a Series")
            self.assertEqual(want_metadata, got_metadata, msg=f"{msg}: Series metadata not equal")

    def setUp(self):
        self.addTypeEqualityFunc(StructuredEventStreamTransformerConfig, self.assert_type_and_vars_equal)
        self.addTypeEqualityFunc(MeasurementConfig, self.assert_measurement_config_equal)
        self.addTypeEqualityFunc(EventStreamDatasetConfig, self.assert_type_and_vars_equal)
        self.addTypeEqualityFunc(EventStreamTransformerOutputWithPast, self.assert_type_and_vars_equal)
        self.addTypeEqualityFunc(Vocabulary, self.assert_vocabulary_equal)
        super().setUp()

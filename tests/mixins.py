import sys
sys.path.append('..')

import torch, numpy as np, pandas as pd

from typing import Any, Callable, Dict, Optional, Tuple, Union

from EventStream.EventStreamData.config import EventStreamDatasetConfig, MeasurementConfig
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

    def assertNestedDictEqual(self, want: dict, got: dict, msg: Optional[str] = None):
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
            if isinstance(want_val, dict):
                self.assertTrue(isinstance(got_val, dict), msg=m)
                self.assertNestedDictEqual(want_val, got_val, msg=m)
            elif isinstance(want_val, torch.distributions.Distribution):
                self.assertTrue(isinstance(got_val, torch.distributions.Distribution), msg=m)
                self.assertDistributionsEqual(want_val, got_val, msg=m)
            else:
                self.assertEqual(want_val, got_val, msg=m)

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

class ConfigComparisonsMixin(MLTypeEqualityCheckableMixin):
    """
    This mixin provides capability to `unittest.TestCase` submodules to compare configuation objects for
    equality, including:
      * `pd.DataFrame`, via `pd.testing.assert_frame_equal`
      * `pd.Series`, via `pd.testing.assert_series_equal`
      * `MeasurementConfig`, via `self.assert_metadata_column_config_equal`
      * `EventStreamDatasetConfig`, via `self.assert_event_stream_dataset_config_equal`
    """

    def assert_type_and_vars_equal(self, want: object, got: object, msg: Optional[str] = None):
        self.assertEqual(type(want), type(got), msg)
        self.assertNestedDictEqual(vars(want), vars(got), msg)

    def setUp(self):
        super().setUp()
        self.addTypeEqualityFunc(StructuredEventStreamTransformerConfig, self.assert_type_and_vars_equal)
        self.addTypeEqualityFunc(MeasurementConfig, self.assert_type_and_vars_equal)
        self.addTypeEqualityFunc(EventStreamDatasetConfig, self.assert_type_and_vars_equal)
        self.addTypeEqualityFunc(EventStreamTransformerOutputWithPast, self.assert_type_and_vars_equal)

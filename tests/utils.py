import sys

sys.path.append("..")

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional
from unittest.mock import Mock, _Call

import numpy as np
import pandas as pd
import polars as pl
import torch
from polars.testing import assert_frame_equal as assert_pl_frame_equal

from EventStream.data.config import DatasetConfig, MeasurementConfig
from EventStream.data.types import PytorchBatch
from EventStream.data.vocabulary import Vocabulary
from EventStream.transformer.config import StructuredTransformerConfig
from EventStream.transformer.model_output import (
    GenerativeSequenceModelLabels,
    GenerativeSequenceModelLosses,
    GenerativeSequenceModelOutput,
    GenerativeSequenceModelPredictions,
    TransformerOutputWithPast,
)

ASSERT_FN = Callable[[Any, Any, Optional[str]], None]


def round_dict(d: dict[str, float]) -> dict[str, float]:
    """Rounds a dictionary of floats to five places.

    Args:
        d: The dictionary to round.

    Returns:
        A dictionary with the same keys as `d`, whose values are `None` if the associated value in `d` is
        `None`, and otherwise the value in `d` rounde dto five places.

    Examples:
        >>> round_dict({"a": 1.23456789, "b": 2.3456789})
        {'a': 1.23457, 'b': 2.34568}
        >>> round_dict({"a": None, "b": 1})
        {'a': None, 'b': 1}
    """
    return {k: None if v is None else round(v, 5) for k, v in d.items()}


class MockModule(Mock, torch.nn.Module):
    """Useful for mocking sub-modules of a `torch.nn.Module`."""

    def __init__(self, *args, **kwargs):
        Mock.__init__(self, *args, **kwargs)
        torch.nn.Module.__init__(self)


class MLTypeEqualityCheckableMixin:
    """This mixin provides capability to `unittest.TestCase` submodules to check various common ML types for
    equality, including:

    * `torch.Tensor`, via `torch.testing.assert_close`
    * `pd.DataFrame`, via `pd.testing.assert_frame_equal`
    * `pd.Series`, via `pd.testing.assert_series_equal`
    * `np.ndarray`, via `np.testing.assert_array_equal`
    """

    EQ_TYPE_CHECKERS = {
        torch.Tensor: (
            torch.testing.assert_close,
            {"equal_nan": True, "rtol": 1e-3, "atol": 1e-3},
        ),
        pd.DataFrame: pd.testing.assert_frame_equal,
        pl.DataFrame: (assert_pl_frame_equal, {"check_column_order": False}),
        pd.Series: pd.testing.assert_series_equal,
        np.ndarray: np.testing.assert_allclose,
    }

    def _typedAssertEqualFntr(self, assert_fn: ASSERT_FN | tuple[ASSERT_FN, dict[str, Any]]) -> ASSERT_FN:
        if type(assert_fn) is tuple:
            assert_fn, assert_kwargs = assert_fn
        else:
            assert_kwargs = {}

        def f(want: Any, got: Any, msg: str | None = None):
            try:
                assert_fn(want, got, **assert_kwargs)
            except Exception as e:
                if msg is None:
                    msg = ""
                msg = f"{msg}\nWant:\n{want}\nGot:\n{got}"
                raise self.failureException(msg) from e

        return f

    def assertNestedEqual(self, want: Any, got: Any, msg: str | None = None, check_like: bool = False):
        m = msg
        if m is None:
            m = "Values aren't equal"

        types_match = isinstance(want, type(got)) or isinstance(got, type(want))

        try:
            types_match = (want == got) or types_match
        except Exception:
            pass

        self.assertTrue(types_match, msg=f"{m}: Want type {type(want)}, got type {type(got)}")

        if (type(want) is pd.DataFrame) and check_like:
            try:
                pd.testing.assert_frame_equal(want, got, check_like=True)
            except Exception as e:
                if m is None:
                    m = ""
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
            if not m:
                m = "Sequences aren't equal"
            for i, (want_i, got_i) in enumerate(zip(want, got)):
                self.assertNestedEqual(want_i, got_i, msg=f"{m} (index {i})", check_like=check_like)
        elif isinstance(want, float):
            if math.isnan(want):
                self.assertTrue(math.isnan(got), msg=m)
            else:
                self.assertFalse(math.isnan(got), msg=m)
                self.assertEqual(want, got, msg=m)
        else:
            m = f"{m}: Want {want}, got {got}"
            self.assertEqual(want, got, msg=m)

    def assertNestedDictEqual(self, want: dict, got: dict, msg: str | None = None, check_like: bool = False):
        """This assers that two dictionaries are equal using nested assert checks for the internal values.

        It is useful so that we can compare dictionaries of tensors or arrays with the type- specific
        comparators.
        """

        self.assertIsInstance(want, dict, msg)
        self.assertIsInstance(got, dict, msg)
        self.assertEqual(set(want.keys()), set(got.keys()), msg)

        for k in want.keys():
            want_val = want[k]
            got_val = got[k]
            if msg:
                m = f"{msg} (key {k})"
            else:
                m = f"Dictionaries aren't equal (key {k})"

            self.assertNestedEqual(want_val, got_val, m)

    def assertDistributionsEqual(
        self,
        want: torch.distributions.Distribution,
        got: torch.distributions.Distribution,
        msg: str | None = None,
    ):
        m_type = f"Type of distributions does not match! want {type(want)}, got {type(got)}"
        if msg is not None:
            m_type = f"{msg}: {m_type}"
        self.assertEqual(type(want), type(got), m_type)

        m_vars = f"Parameters of distributions does not match! want {vars(want)}, got {vars(got)}"
        if msg is not None:
            m_vars = f"{msg}: {m_vars}"
        self.assertNestedDictEqual(vars(want), vars(got), m_vars)

    def assertNestedCalledWith(self, mock: Mock, want_calls: list[_Call]):
        self.assertEqual(mock.call_count, len(want_calls))
        for want, got in zip(want_calls, mock.mock_calls):
            want_name = ""
            if len(want) == 2:
                want_args, want_kwargs = want
            else:
                want_name, want_args, want_kwargs = want

            self.assertFalse(
                getattr(want, "_mock_parent", None)
                and getattr(got, "_mock_parent", None)
                and want._mock_parent != got._mock_parent
            )

            len_got = len(got)
            self.assertTrue(len_got <= 3)
            got_name = ""
            if len_got == 0:
                got_args, got_kwargs = (), {}
            elif len_got == 3:
                got_name, got_args, got_kwargs = got
            elif len_got == 1:
                (value,) = got
                if isinstance(value, tuple):
                    got_args = value
                    got_kwargs = {}
                elif isinstance(value, str):
                    got_name = value
                    got_args, got_kwargs = (), {}
                else:
                    got_args = ()
                    got_kwargs = value
            elif len_got == 2:
                # could be (name, args) or (name, kwargs) or (args, kwargs)
                first, second = got
                if isinstance(first, str):
                    got_name = first
                    if isinstance(second, tuple):
                        got_args, got_kwargs = second, {}
                    else:
                        got_args, got_kwargs = (), second
                else:
                    got_args, got_kwargs = first, second

            self.assertFalse(want_name and got_name != want_name)
            self.assertNestedEqual(want_args, got_args)
            self.assertNestedEqual(want_kwargs, got_kwargs)

    def assert_type_and_vars_equal(self, want: object, got: object, msg: str | None = None):
        self.assertEqual(type(want), type(got), msg)
        self.assertNestedDictEqual(vars(want), vars(got), msg, check_like=True)

    def setUp(self):
        for val_type, assert_fn in self.EQ_TYPE_CHECKERS.items():
            fn = self._typedAssertEqualFntr(assert_fn)
            self.addTypeEqualityFunc(val_type, fn)

        super().setUp()


class ConfigComparisonsMixin(MLTypeEqualityCheckableMixin):
    """This mixin provides capability to `unittest.TestCase` submodules to compare configuration objects for
    equality."""

    def assert_vocabulary_equal(self, want: Vocabulary, got: Vocabulary, msg: str | None = None):
        self.assertEqual(type(want), type(got), msg)
        self.assertEqual(want.vocabulary, got.vocabulary, msg)
        self.assertEqual(np.array(want.obs_frequencies), np.array(got.obs_frequencies), msg)

    def assert_measurement_config_equal(
        self, want: MeasurementConfig, got: MeasurementConfig, msg: str | None = None
    ):
        if msg is None:
            msg = "MeasurementConfigs are not equal"
        self.assertEqual(type(want), type(got), f"{msg}: Types {type(want)} and {type(got)} don't match")

        want_less_metadata = vars(want).copy()
        want_metadata = want_less_metadata.pop("_measurement_metadata")
        got_less_metadata = vars(got).copy()
        got_metadata = got_less_metadata.pop("_measurement_metadata")

        self.assertNestedDictEqual(
            want_less_metadata,
            got_less_metadata,
            msg=f"{msg}: Non-metadata keys aren't equal.",
            check_like=True,
        )

        match want_metadata:
            case None:
                self.assertIsNone(got_metadata, msg=f"{msg}: got metadata is not None")
            case str() | Path():
                self.assertEqual(
                    str(want_metadata),
                    str(got_metadata),
                    msg=f"{msg}: {want_metadata} != {got_metadata}!",
                )
            case pd.DataFrame():
                self.assertTrue(
                    isinstance(got_metadata, pd.DataFrame),
                    msg=f"{msg}: got metadata is not a DataFrame",
                )
                want_idx = want_metadata.index
                got_idx = got_metadata.index

                for model_col in ("outlier_model", "normalizer"):
                    self.assertEqual((model_col in want_metadata), (model_col in got_metadata))

                    if model_col not in want_metadata:
                        continue

                    want_metadata[model_col] = want_metadata[model_col].apply(round_dict)
                    got_metadata[model_col] = got_metadata[model_col].apply(round_dict)

                self.assertEqual(set(want_idx), set(got_idx), msg)
                # I don't know why, by the extra copy() is necessary to avoid the reindex sometimes not taking
                # and the resulting dataframes to not match index orders.
                reordered_got = got_metadata.copy().reindex(want_idx).copy()
                try:
                    pd.testing.assert_frame_equal(want_metadata, reordered_got, check_like=True)
                except Exception as e:
                    if msg is None:
                        msg = ""
                    msg = f"{msg}\nWant:\n{want_metadata}\nGot:\n{reordered_got}"
                    raise self.failureException(msg) from e
            case _:
                self.assertTrue(
                    isinstance(want_metadata, pd.Series),
                    msg=f"{msg}: want metadata is not a Series",
                )
                self.assertTrue(
                    isinstance(got_metadata, pd.Series), msg=f"{msg}: got metadata is not a Series"
                )
                want_metadata = want_metadata.to_dict()
                got_metadata = got_metadata.to_dict()
                for model_col in ("outlier_model", "normalizer"):
                    self.assertEqual((model_col in want_metadata), (model_col in got_metadata))

                    if model_col not in want_metadata:
                        continue

                    try:
                        want_metadata[model_col] = round_dict(want_metadata[model_col])
                    except AttributeError as e:
                        raise self.failureException(
                            f"Rounding dict failed for {model_col} on want!\n{want_metadata}"
                        ) from e

                    try:
                        got_metadata[model_col] = round_dict(got_metadata[model_col])
                    except AttributeError as e:
                        raise self.failureException(
                            f"Rounding dict failed for {model_col} on got!\n{got_metadata}"
                        ) from e

                for k in want_metadata:
                    self.assertTrue(k in got_metadata, f"{msg}: Key {k} in want but not got!")
                for k in got_metadata:
                    self.assertTrue(k in want_metadata, f"{msg}: Key {k} in got but not want!")

                for k in got_metadata:
                    want_v = want_metadata[k]
                    got_v = want_metadata[k]
                    self.assertEqual(
                        want_v,
                        got_v,
                        msg=f"{msg}: Series metadata values for {k} differ! {want_v} vs. {got_v}",
                    )

    def setUp(self):
        self.addTypeEqualityFunc(MeasurementConfig, self.assert_measurement_config_equal)
        self.addTypeEqualityFunc(DatasetConfig, self.assert_type_and_vars_equal)
        self.addTypeEqualityFunc(Vocabulary, self.assert_vocabulary_equal)
        self.addTypeEqualityFunc(PytorchBatch, self.assert_type_and_vars_equal)
        self.addTypeEqualityFunc(StructuredTransformerConfig, self.assert_type_and_vars_equal)
        self.addTypeEqualityFunc(TransformerOutputWithPast, self.assert_type_and_vars_equal)
        self.addTypeEqualityFunc(GenerativeSequenceModelLabels, self.assert_type_and_vars_equal)
        self.addTypeEqualityFunc(GenerativeSequenceModelLosses, self.assert_type_and_vars_equal)
        self.addTypeEqualityFunc(GenerativeSequenceModelOutput, self.assert_type_and_vars_equal)
        self.addTypeEqualityFunc(GenerativeSequenceModelPredictions, self.assert_type_and_vars_equal)
        super().setUp()

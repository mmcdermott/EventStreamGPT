import sys

sys.path.append("..")

import dataclasses
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from EventStream.utils import (
    JSONableMixin,
    count_or_proportion,
    flatten_dict,
    is_monotonically_nonincreasing,
    to_sklearn_np,
)

from .mixins import MLTypeEqualityCheckableMixin


class TestUtilFunctions(MLTypeEqualityCheckableMixin, unittest.TestCase):
    """Tests `EventStreamData.utils`."""

    def test_count_or_proportion(self):
        self.assertEqual(
            4, count_or_proportion(10, 4), msg="When passed an integer, should return same."
        )
        self.assertEqual(
            3,
            count_or_proportion(10, 1 / 3),
            msg="When passed a float, should return the proportion.",
        )

    def test_is_monotonically_nonincreasing(self):
        self.assertTrue(is_monotonically_nonincreasing(np.array([1, 0, 0, -1000, -float("inf")])))
        self.assertFalse(is_monotonically_nonincreasing(np.array([1, 2, 0, -1000, -float("inf")])))
        self.assertTrue(is_monotonically_nonincreasing(np.array([])))

    def test_flatten_dict(self):
        self.assertEqual({}, flatten_dict({}))
        self.assertEqual(
            {"foo": 3, "bar": 3, "biz": 4}, flatten_dict({("foo", "bar"): 3, ("biz",): 4})
        )

    def test_to_sklearn_np(self):
        self.assertEqual(
            np.array([], dtype=float).reshape(-1, 1), to_sklearn_np(pd.Series([], dtype=float))
        )
        self.assertEqual(
            np.array([[1], [-2], [3], [-4]]), to_sklearn_np(pd.Series([1, -2, 3, -4]))
        )
        self.assertEqual(
            np.array([[1], [2], [3], [4]]), to_sklearn_np(pd.Series([1, np.NaN, "foo", 2, 3, 4]))
        )


class TestJSONableMixin(unittest.TestCase):
    def test_save_load(self):
        class Derived(JSONableMixin):
            def __init__(self, a, b):
                self.a = a
                self.b = b

            def to_dict(self):
                return {"a": self.a, "b": self.b}

            def __eq__(self, other):
                return type(other) is type(self) and vars(other) == vars(self)

        obj = Derived(1, "2")

        with TemporaryDirectory() as d:
            save_path = Path(d) / "config.json"
            obj.to_json_file(save_path)
            got_obj = Derived.from_json_file(save_path)
            self.assertEqual(obj, got_obj)

            with self.assertRaises(FileExistsError):
                obj.to_json_file(save_path)

            obj.a = 4
            self.assertNotEqual(
                obj, got_obj, "These should no longer be equal given the modification."
            )

            obj.to_json_file(save_path, do_overwrite=True)
            got_obj = Derived.from_json_file(save_path)
            self.assertEqual(obj, got_obj)

            with self.assertRaises(FileNotFoundError):
                got_obj = Derived.from_json_file(Path(d) / "not_found.json")

            with self.assertRaises(FileNotFoundError):
                obj.to_json_file(Path(d) / "not_found" / "config.json")

    def test_auto_to_dict_for_dataclasses(self):
        @dataclasses.dataclass
        class Derived(JSONableMixin):
            a: int = 4
            b: str = "foo"

        obj = Derived(a=3, b="hi")
        self.assertEqual({"a": 3, "b": "hi"}, obj.to_dict())

import sys

sys.path.append("..")

import dataclasses
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from EventStream.utils import (
    JSONableMixin,
    count_or_proportion,
    lt_count_or_proportion,
    num_initial_spaces,
)

from .utils import MLTypeEqualityCheckableMixin


class TestUtilFunctions(MLTypeEqualityCheckableMixin, unittest.TestCase):
    """Tests `EventStreamData.utils`."""

    def test_count_or_proportion(self):
        self.assertEqual(4, count_or_proportion(10, 4))
        self.assertEqual(3, count_or_proportion(10, 1 / 3))
        with self.assertRaises(TypeError):
            count_or_proportion(10, "foo")
        with self.assertRaises(ValueError):
            count_or_proportion(10, 1.1)
        with self.assertRaises(ValueError):
            count_or_proportion(10, -0.1)
        with self.assertRaises(ValueError):
            count_or_proportion(10, 0)
        with self.assertRaises(TypeError):
            count_or_proportion("foo", 1 / 3)

    def test_lt_count_or_proportion(self):
        self.assertFalse(lt_count_or_proportion(10, None, 100))
        self.assertTrue(lt_count_or_proportion(10, 11, 100))
        self.assertFalse(lt_count_or_proportion(12, 11, 100))

    def test_num_initial_spaces(self):
        self.assertEqual(0, num_initial_spaces("foo"))
        self.assertEqual(3, num_initial_spaces("   \tfoo"))


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
            self.assertNotEqual(obj, got_obj, "These should no longer be equal given the modification.")

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

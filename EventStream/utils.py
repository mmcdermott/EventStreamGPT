"""Utility functions for the EventStream library."""


from __future__ import annotations

import dataclasses
import enum
import functools
import json
import re
import traceback
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any, TypeVar, Union

import hydra
import polars as pl
from loguru import logger

PROPORTION = float
COUNT_OR_PROPORTION = Union[int, PROPORTION]
WHOLE = Union[int, pl.Expr]


def count_or_proportion(N: WHOLE | None, cnt_or_prop: COUNT_OR_PROPORTION) -> int:
    """Returns `cnt_or_prop` if it is an integer or `int(N*cnt_or_prop)` if it is a float.

    Resolves cutoff variables that can either be passed as integer counts or fractions of a whole. E.g., the
    vocabulary should contain only elements that occur with count or proportion at least X, where X might be
    20 times, or 1%.

    Arguments:
        N: The total number of elements in the whole. Only used if `cnt_or_prop` is a proportion (float).
        cnt_or_prop: The cutoff value, either as an integer count or a proportion of the whole.

    Returns:
        The cutoff value as an integer count of the whole.

    Raises:
        TypeError: If `cnt_or_prop` is not an integer or a float or if `N` is needed and is not an integer or
            a polars Expression.
        ValueError: If `cnt_or_prop` is not a positive integer or a float between 0 and 1.

    Examples:
        >>> count_or_proportion(100, 0.1)
        10
        >>> count_or_proportion(None, 11)
        11
        >>> count_or_proportion(100, 0.116)
        12
        >>> count_or_proportion(None, 0)
        Traceback (most recent call last):
            ...
        ValueError: 0 must be positive if it is an integer
        >>> count_or_proportion(None, 1.3)
        Traceback (most recent call last):
            ...
        ValueError: 1.3 must be between 0 and 1 if it is a float
        >>> count_or_proportion(None, "a")
        Traceback (most recent call last):
            ...
        TypeError: a must be a positive integer or a float between 0 or 1
        >>> count_or_proportion("a", 0.2)
        Traceback (most recent call last):
            ...
        TypeError: a must be an integer or a polars.Expr when cnt_or_prop is a float!
    """

    match cnt_or_prop:
        case int() if 0 < cnt_or_prop:
            return cnt_or_prop
        case int():
            raise ValueError(f"{cnt_or_prop} must be positive if it is an integer")
        case float() if 0 < cnt_or_prop < 1:
            pass
        case float():
            raise ValueError(f"{cnt_or_prop} must be between 0 and 1 if it is a float")
        case _:
            raise TypeError(f"{cnt_or_prop} must be a positive integer or a float between 0 or 1")

    match N:
        case int():
            return int(round(cnt_or_prop * N))
        case pl.Expr():
            return (N * cnt_or_prop).round(0).cast(int)
        case _:
            raise TypeError(f"{N} must be an integer or a polars.Expr when cnt_or_prop is a float!")


def lt_count_or_proportion(
    N_obs: int, cnt_or_prop: COUNT_OR_PROPORTION | None, N_total: int | None = None
) -> bool:
    """Returns True if `N_obs` is less than the `cnt_or_prop` of `N_total`.

    Arguments:
        N_obs: The number of observations.
        cnt_or_prop: The cutoff value, either as an integer count or a proportion of the whole.
        N_total (optional; default is `None`): The total number of elements in the whole. Only used if
            `cnt_or_prop` is a proportion.

    Returns:
        If `cnt_or_prop` is `None`, return `False`. Otherwise, return `True` if `N_obs` is less than the
        `cnt_or_prop` (if it is a count) or `int(round(cnt_or_prop*N_total))` if it is a proportion.

    Examples:
        >>> lt_count_or_proportion(10, 0.1, 100)
        False
        >>> lt_count_or_proportion(10, 0.11, 100)
        True
        >>> lt_count_or_proportion(10, 11)
        True
        >>> lt_count_or_proportion(10, 9)
        False
        >>> lt_count_or_proportion(10, None)
        False
    """
    if cnt_or_prop is None:
        return False
    return N_obs < count_or_proportion(N_total, cnt_or_prop)


def num_initial_spaces(s: str) -> int:
    """Returns the number of initial spaces in `s`.

    Arguments:
        s: The string of which to count the initial spaces.

    Returns:
        The number of initial spaces in `s`.

    Examples:
        >>> num_initial_spaces("  a")
        2
        >>> num_initial_spaces("lorem ipsum    ")
        0
    """
    return len(s) - len(s.lstrip(" "))


class StrEnum(str, enum.Enum):
    """An enum object where members are stored as lowercase strings and can be used as strings.

    StrEnum is a Python ``enum.Enum`` that inherits from ``str``. This allows it to be compared identically
    with string objects, making it suitable for use for configuration values parsed from command line or other
    string arguments. The default ``auto()`` behavior uses the member name, lowercased, as its value.
    This code is sourced from
    https://github.com/irgeek/StrEnum/blob/0f868b68cb7cdab50a79117679a301f550a324bc/strenum/__init__.py#L21
    This is made obsolete by python 3.11, which has `enum.StrEnum` natively.

    Raises:
        TypeError if given enum variable values are not strings.

    Examples:
        >>> from enum import auto
        >>> class Example(StrEnum):
        ...     UPPER_CASE = auto()
        ...     lower_case = auto()
        ...     MixedCaseFixed = "MixedCaseFixed"
        >>> assert Example.UPPER_CASE == "upper_case"
        >>> assert Example.lower_case == "lower_case"
        >>> assert Example.MixedCaseFixed == "MixedCaseFixed"
        >>> class Example(StrEnum):
        ...     VAR_1 = 1
        Traceback (most recent call last):
            ...
        TypeError: Values of StrEnums must be strings: 1 is a <class 'int'>

    Todo:
        TODO(mattmcdermott8@gmail.com): Upgrade to python 3.11 and eliminate this class. See
            https://github.com/mmcdermott/EventStreamGPT/issues/23
    """

    def __new__(cls, value, *args, **kwargs):
        if not isinstance(value, (str, enum.auto)):
            raise TypeError(f"Values of StrEnums must be strings: {value!r} is a {type(value)}")
        return super().__new__(cls, value, *args, **kwargs)

    def __str__(self):
        return str(self.value)

    def _generate_next_value_(name, *_):
        return name.lower()

    @classmethod
    def values(cls):
        """Returns a list of enum class member options as strings.

        This method gives a list of possible options in the calling class; it is useful for validating a
        string is a member of the enum.

        Returns:
            A list of enum members as strings.

        Examples:
            >>> from enum import auto
            >>> class Example(StrEnum):
            ...     UPPER_CASE = auto()
            ...     lower_case = auto()
            ...     MixedCase = auto()
            >>> Example.values()
            ['upper_case', 'lower_case', 'mixedcase']
            >>> class Example(StrEnum):
            ...     var1 = "VAR_1"
            ...     Var2 = auto()
            >>> Example.values()
            ['VAR_1', 'var2']
        """
        return list(map(lambda c: c.value, cls))


# Create a generic variable that can be 'JSONableMixin', or any subclass.
JSONABLE_INSTANCE_T = TypeVar("JSONABLE_INSTANCE_T", bound="JSONableMixin")


class JSONableMixin:
    """A simple mixin to enable saving/loading of data container classes to json files.

    This mixin allows easy conversion between python objects and JSON format, facilitating
    their storage and retrieval. Subclasses must implement a `to_dict` method that
    defines how the object should be converted to a dictionary.

    Todo:
        TODO(mattmcdermott8@gmail.com): Investigate removing in favor of
        [OmegaConf](https://omegaconf.readthedocs.io/en/latest/index.html) throughout. See
        https://github.com/mmcdermott/EventStreamGPT/issues/24
    """

    @classmethod
    def from_dict(cls: type[JSONABLE_INSTANCE_T], as_dict: dict) -> JSONABLE_INSTANCE_T:
        """Converts a dictionary representation of an object into the calling class.

        By default, this method simply calls the calling class constructor with the arguments in `as_dict` as
        keyword arguments. Can be overwritten by subclasses for more complex use cases.

        Arguments:
            as_dict: A dictionary representation of an object.

        Returns:
            An instance of the calling class.

        Examples:
            >>> class MyData(JSONableMixin):
            ...     def __init__(self, name):
            ...         self.name = name
            >>> my_data = MyData.from_dict({'name': 'Test'})
            >>> my_data.name
            'Test'
        """
        return cls(**as_dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts the object into a dictionary.

        If the calling object is a `dataclasses.dataclass`, then this method just calls `dataclasses.asdict`.
        Otherwise, this method needs to be implemented by the subclasses.

        Returns:
            A dictionary representation of the object.

        Raises:
            NotImplementedError: If this method is not implemented by the subclass.

        Examples:
            >>> @dataclasses.dataclass
            ... class MyData(JSONableMixin):
            ...     name: str
            >>> MyData('Test').to_dict()
            {'name': 'Test'}
            >>> class MyData(JSONableMixin):
            ...     def __init__(self, name: str):
            ...         self.name = name
            ...     def to_dict(self) -> dict[str, str]:
            ...         return {"name": self.name}
            >>> MyData('Test2').to_dict()
            {'name': 'Test2'}
            >>> class MyData(JSONableMixin):
            ...     def __init__(self, name: str):
            ...         self.name = name
            >>> MyData('Test2').to_dict()
            Traceback (most recent call last):
                ...
            NotImplementedError: This must be overwritten in non-dataclass derived classes!
        """
        if dataclasses.is_dataclass(self):
            return dataclasses.asdict(self)
        raise NotImplementedError("This must be overwritten in non-dataclass derived classes!")

    def to_json_file(self, fp: Path, do_overwrite: bool = False):
        """Writes the object to a json file.

        Serializes the object as JSON and writes it to a file.

        Args:
            fp: The file path to write the JSON data.
            do_overwrite: If True, overwrites an existing file at the specified path. Defaults to False.

        Raises:
            FileExistsError: If the file already exists and do_overwrite is set to False.

        Examples:
            >>> import dataclasses
            >>> import tempfile
            >>> from pathlib import Path
            >>> @dataclasses.dataclass
            ... class MyData(JSONableMixin):
            ...     name: str
            >>> data = MyData('Test')
            >>> with tempfile.TemporaryDirectory() as tmp_dir:
            ...     fp = Path(tmp_dir) / 'test.json'
            ...     data.to_json_file(fp, do_overwrite=False)
            ...     with open(fp, mode='r') as f:
            ...         f.read()
            '{"name": "Test"}'
            >>> with tempfile.TemporaryDirectory() as tmp_dir:
            ...     fp = Path(tmp_dir) / 'test.json'
            ...     fp.touch()
            ...     data.to_json_file(fp, do_overwrite=False)
            Traceback (most recent call last):
                ...
            FileExistsError: ...test.json exists and do_overwrite = False
        """
        if (not do_overwrite) and fp.exists():
            raise FileExistsError(f"{fp} exists and do_overwrite = {do_overwrite}")
        with open(fp, mode="w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_json_file(cls: type[JSONABLE_INSTANCE_T], fp: Path) -> JSONABLE_INSTANCE_T:
        """Loads the object from a json file.

        Reads a JSON file and converts it into an object of the calling class.

        Args:
            fp: The file path to read the JSON data.

        Returns:
            An instance of the calling class.

        Raises:
            FileNotFoundError: If the passed file path does not exist.

        Examples:
            >>> import dataclasses
            >>> import tempfile
            >>> from pathlib import Path
            >>> @dataclasses.dataclass
            ... class MyData(JSONableMixin):
            ...     name: str
            >>> with tempfile.TemporaryDirectory() as tmp_dir:
            ...     fp = Path(tmp_dir) / 'test.json'
            ...     with open(fp, mode='w') as f:
            ...         _ = f.write('{"name": "Test"}')
            ...     data = MyData.from_json_file(fp)
            >>> data.to_dict()
            {'name': 'Test'}
            >>> with tempfile.TemporaryDirectory() as tmp_dir:
            ...     fp = Path(tmp_dir) / 'test.json'
            ...     MyData.from_json_file(fp)
            Traceback (most recent call last):
                ...
            FileNotFoundError: ...test.json...
        """
        with open(fp) as f:
            return cls.from_dict(json.load(f))


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    It ensures that weights and biases finish tracking any runs that were running, even in the case of an
    exception, to avoid multi-run failures due to weights and biases errors.
    """

    @functools.wraps(task_func)
    def wrap(*args, **kwargs):
        try:
            fn_return = task_func(*args, **kwargs)
        except Exception as ex:
            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            logger.error(f"EXCEPTION: {ex}\nTRACEBACK:\n{traceback.print_exc()}")
            raise ex
        finally:
            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    wandb.finish()

        return fn_return

    return wrap


def hydra_dataclass(dataclass: Any) -> Any:
    """Decorator that allows you to use a dataclass as a hydra config via the `ConfigStore`

    Adds the decorated dataclass as a `Hydra StructuredConfig object`_ to the `Hydra ConfigStore`_.
    The name of the stored config in the ConfigStore is the snake case version of the CamelCase class name.

    .. _Hydra StructuredConfig object: https://hydra.cc/docs/tutorials/structured_config/intro/

    .. _Hydra ConfigStore: https://hydra.cc/docs/tutorials/structured_config/config_store/
    """

    dataclass = dataclasses.dataclass(dataclass)

    name = dataclass.__name__
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    cs = hydra.core.config_store.ConfigStore.instance()
    cs.store(name=name, node=dataclass)

    return dataclass

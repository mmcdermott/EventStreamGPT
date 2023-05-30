from __future__ import annotations

import dataclasses
import enum
import functools
import json
import re
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any, TypeVar, Union

import hydra
import polars as pl

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
        >>> count_or_proportion("a", 10)
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
            raise TypeError(
                f"{N} must be an integer or a polars.Expr when cnt_or_prop is a float!"
            )


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
            https://github.com/mmcdermott/EventStreamML/issues/23
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

    Todo:
        TODO(mattmcdermott8@gmail.com): Investigate removing in favor of
        [OmegaConf](https://omegaconf.readthedocs.io/en/latest/index.html) throughout. See
        https://github.com/mmcdermott/EventStreamML/issues/24
    """

    @classmethod
    def from_dict(cls: type[JSONABLE_INSTANCE_T], as_dict: dict) -> JSONABLE_INSTANCE_T:
        """Returns a calling-class version of the data presented in `as_dict`.

        By default, this method simply calls the calling class constructor with the arguments in `as_dict` as
        keyword arguments. Can be overwritten by subclasses for more complex use cases.

        Arguments:
            as_dict: The data with which to instantiate the calling class.

        Returns:
            An instance of the calling class instantiated with the passed data.
        """
        return cls(**as_dict)

    def to_dict(self) -> dict[str, Any]:
        if dataclasses.is_dataclass(self):
            return dataclasses.asdict(self)
        raise NotImplementedError("This must be overwritten in non-dataclass derived classes!")

    def to_json_file(self, fp: Path, do_overwrite: bool = False):
        """Writes configuration object to a json file as a plain dictionary."""
        if (not do_overwrite) and fp.exists():
            raise FileExistsError(f"{fp} exists and do_overwrite = {do_overwrite}")
        with open(fp, mode="w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_json_file(cls: type[JSONABLE_INSTANCE_T], fp: Path) -> JSONABLE_INSTANCE_T:
        """Build configuration object from contents of `fp` interpreted as a dictionary stored in
        json."""
        with open(fp) as f:
            return cls.from_dict(json.load(f))


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        ...

        return metric_dict, object_dict
    ```
    """

    @functools.wraps(task_func)
    def wrap(*args, **kwargs):
        try:
            fn_return = task_func(*args, **kwargs)
        except Exception as ex:
            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
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
    """Decorator that allows you to use a dataclass as a hydra config by adding it to the hydra
    store.

    Example:
    ```
    @hydra_dataclass
    class MyConfig:
        foo: int = 1
        bar: str = "baz"

    # Name of the config is the snake case version of the (CamelCase) class name
    @hydra.main(config_name="my_config")
    def main(cfg: MyConfig) -> None:
        print(cfg.foo, cfg.bar)
    ```
    """

    dataclass = dataclasses.dataclass(dataclass)

    name = dataclass.__name__
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    cs = hydra.core.config_store.ConfigStore.instance()
    cs.store(name=name, node=dataclass)

    return dataclass

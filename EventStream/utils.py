from __future__ import annotations

import dataclasses, enum, json, numpy as np, pandas as pd

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Type, TypeVar, Union

PROPORTION = float
COUNT_OR_PROPORTION = Union[int, PROPORTION]

def count_or_proportion(N: Optional[int], cnt_or_prop: COUNT_OR_PROPORTION) -> int:
    """
    Returns either `cnt_or_prop` if it is an integer of `int(N*cnt_or_prop)` if it is a float.
    Used for resolving cutoff variables that can either be passed as integer counts or fractions of a whole.
    E.g., the vocabulary should contain only elements that occur with count or proportion at least X, where X
    might be 20 times, or 1%.
    """
    if type(cnt_or_prop) is int: return cnt_or_prop
    assert N is not None
    return int(cnt_or_prop * N)

def lt_count_or_proportion(
    N_obs: int, cnt_or_prop: Optional[COUNT_OR_PROPORTION], N_total: Optional[int] = None
) -> bool:
    if cnt_or_prop is None: return False
    return N_obs < count_or_proportion(N_total, cnt_or_prop)

def is_monotonically_nonincreasing(a: np.ndarray) -> bool:
    """Returns True if and only if a is monotonically non-increasing."""
    return np.all(a[:-1] >= a[1:])

def flatten_dict(exp_dict: Dict[Sequence[str], Any]) -> Dict[str, Any]:
    """
    Flattens a dictionary out by key.

    Args:
        `exp_dict` (`Dict[Sequence[Hashable], Any]`):
            A dictionary of the form `{[k1, k2, ...]: v}`

    Returns: A dictionary of the form `{k1: v, k2: v, ...}`
    """

    out = {}
    for ks, v in exp_dict.items():
        for k in ks: out[k] = v
    return out

def to_sklearn_np(vals: pd.Series):
    """
    Takes a pandas series and returns a view consisting of only non-null numeric types reshaped to a 2D view
    where the last dimension is 1 (which is needed for scikit-learn).
    """
    vals = vals.dropna()
    vals = vals[vals.apply(lambda v: pd.api.types.is_numeric_dtype(type(v)))]
    vals = vals.astype(float)
    return vals.values.reshape((-1, 1))


class StrEnum(str, enum.Enum):
    """
    This is made obsolete by python 3.11, which has `enum.StrEnum` natively. TODO(mmd): Upgrade to python 3.11
    and eliminate this class.

    This code is sourced from
    https://github.com/irgeek/StrEnum/blob/0f868b68cb7cdab50a79117679a301f550a324bc/strenum/__init__.py#L21

    StrEnum is a Python ``enum.Enum`` that inherits from ``str``. The default
    ``auto()`` behavior uses the member name as its value.
    Example usage::
        class Example(StrEnum):
            UPPER_CASE = auto()
            lower_case = auto()
            MixedCase = auto()
        assert Example.UPPER_CASE == "upper_case"
        assert Example.lower_case == "lower_case"
        assert Example.MixedCase == "mixedcase"
    """

    def __new__(cls, value, *args, **kwargs):
        if not isinstance(value, (str, enum.auto)):
            raise TypeError(
                f"Values of StrEnums must be strings: {value!r} is a {type(value)}"
            )
        return super().__new__(cls, value, *args, **kwargs)

    def __str__(self):
        return str(self.value)

    def _generate_next_value_(name, *_):
        return name.lower()

# Create a generic variable that can be 'JSONableMixin', or any subclass.
JSONABLE_INSTANCE_T = TypeVar('JSONABLE_INSTANCE_T', bound='JSONableMixin')

class JSONableMixin():
    """A simple mixin to enable saving/loading of dataclasses (or other classes in theory) to json files."""

    @classmethod
    def from_dict(cls: Type[JSONABLE_INSTANCE_T], as_dict: dict) -> JSONABLE_INSTANCE_T:
        """This is a default method that can be overwritten in derived classes."""
        return cls(**as_dict)

    def to_dict(self) -> Dict[str, Any]:
        if dataclasses.is_dataclass(self): return dataclasses.asdict(self)
        raise NotImplementedError(f"This must be overwritten in non-dataclass derived classes!")

    def to_json_file(self, fp: Path, do_overwrite: bool = False):
        """Writes configuration object to a json file as a plain dictionary."""
        if (not do_overwrite) and fp.exists():
            raise FileExistsError(f"{fp} exists and do_overwrite = {do_overwrite}")
        with open(fp, mode='w') as f: json.dump(self.to_dict(), f)

    @classmethod
    def from_json_file(cls: Type[JSONABLE_INSTANCE_T], fp: Path) -> JSONABLE_INSTANCE_T:
        """Build configuration object from contents of `fp` interpreted as a dictionary stored in json."""
        with open(fp, mode='r') as f: return cls.from_dict(json.load(f))

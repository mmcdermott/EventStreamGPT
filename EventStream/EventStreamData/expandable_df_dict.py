from __future__ import annotations

import copy, pandas as pd

from typing import Any, Dict, Optional

class ExpandableDfDict():
    """
    This class is just a dumber, lower-overhead version of a dataframe, stored as a dictionary. It is good for
    rapidly updating a dataframe like structure from a series of partially observed dictionaries or key-value
    pairs. Stores a structure such that the number of rows for all columns is constant after appends/extends,
    even if new columns are added or values are missing from existing columns on some appends/extends by
    filling in missing places with Nones.
    """
    def __init__(self, initial_dict: Optional[Dict[str, list]] = None):
        self.df_dict = {}
        self.n_rows  = 0

        if initial_dict is not None: self.extend(initial_dict)

    @property
    def n_cols(self) -> int: return len(self.df_dict)

    def df(self) -> pd.DataFrame: return pd.DataFrame(self.df_dict)

    def append(self, vals: Optional[dict] = None, **other_vals):
        """Appends the passed values into self."""
        if vals is None: vals = {}
        self.extend({k: [v] for k, v in vals.items()}, **{k: [v] for k, v in other_vals.items()})

    def extend(self, vals: Optional[dict] = None, **other_vals):
        """Extends self via the passed values."""
        if vals is None: vals = {}
        assert len(set(vals.keys()).intersection(set(other_vals.keys()))) == 0

        new_vals = {**other_vals, **vals}
        if not new_vals: return

        lens_added = set(len(v) for v in new_vals.values())
        assert len(lens_added) == 1

        len_added = list(lens_added)[0]

        for k, v in new_vals.items():
            if k not in self.df_dict: self.df_dict[k] = [None for _ in range(self.n_rows)]

            self.df_dict[k].extend(v)

        missed_keys = set(self.df_dict.keys()) - set(new_vals.keys())
        for k in missed_keys:
            self.df_dict[k].extend([None for _ in range(len_added)])

        self.n_rows += len_added

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> ExpandableDfDict:
        """Converts an existing dataframe into an ExpandableDfDict"""
        return cls(df.dropna(axis=1, how='all').to_dict(orient='list'))

    @classmethod
    def concatenate(cls, df_dicts: Sequence[ExpandableDfDict]) -> ExpandableDfDict:
        """Concatenates many ExpandableDfDicts together."""
        if len(df_dicts) == 0: return ExpandableDfDict

        if type(df_dicts) is pd.Series: df_dicts = list(df_dicts.values)

        out = copy.deepcopy(df_dicts[0])
        for df_dict in df_dicts[1:]: out.extend(**df_dict)

        return out

    def drop_col(self, col: str):
        """Drops a column from self."""
        self.df_dict.pop(col)
        return self

    # Mostly these operates just forward passed functions down to the underlying dictionary storage in
    # `self.df_dict`.
    def items(self) -> Sequence[Tuple[str, list]]: return self.df_dict.items()
    def values(self) -> Sequence[list]: return self.df_dict.values()
    def keys(self) -> Sequence[str]: return self.df_dict.keys()
    def __contains__(self, key: str) -> bool: return key in self.df_dict
    def __getitem__(self, key: str) -> list: return self.df_dict[key]
    def __setitem__(self, key: str, val: Any) -> list: self.df_dict[key] = val
    def __repr__(self): return f"ExpandableDfDict({repr(self.df_dict)})"
    def __str__(self): return str(self.df_dict)
    def __eq__(self, other: ExpandableDfDict) -> bool: return self.df_dict == other.df_dict
    def get(self, *args, **kwargs): return self.df_dict.get(*args, **kwargs)

"""Helpers for robust MLflow model signatures."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_integer_dtype


def make_nullable_safe_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Represent integer columns as floats so signatures accept missing values."""
    sample = data.copy()
    for column in sample.columns:
        if is_integer_dtype(sample[column].dtype):
            sample[column] = sample[column].astype("float64")
    return sample


__all__ = ["make_nullable_safe_sample"]

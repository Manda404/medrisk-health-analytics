"""Deterministic data contracts for training and scoring datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    """Result of a data contract evaluation."""

    passed: bool
    rows: int
    columns: int
    duplicate_rows: int
    missing_ratio: dict[str, float]
    violations: tuple[str, ...]

    def raise_for_failure(self) -> None:
        """Stop a pipeline when its input violates the contract."""
        if not self.passed:
            raise ValueError("Data quality contract failed: " + "; ".join(self.violations))


class DataQualityValidator:
    """Validate schema, completeness, uniqueness, finiteness, and target values."""

    def __init__(
        self,
        *,
        required_columns: Sequence[str],
        target_column: str | None = None,
        id_columns: Sequence[str] = (),
        max_missing_ratio: float = 0.05,
        allow_duplicate_rows: bool = False,
    ) -> None:
        if not 0 <= max_missing_ratio <= 1:
            raise ValueError("max_missing_ratio must be between 0 and 1")
        self.required_columns = tuple(required_columns)
        self.target_column = target_column
        self.id_columns = tuple(id_columns)
        self.max_missing_ratio = max_missing_ratio
        self.allow_duplicate_rows = allow_duplicate_rows

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        """Return a complete report without mutating the input."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        violations: list[str] = []
        missing_columns = sorted(set(self.required_columns) - set(df.columns))
        if missing_columns:
            violations.append(f"missing required columns: {missing_columns}")

        if df.empty:
            violations.append("dataset is empty")

        duplicate_rows = int(df.duplicated().sum())
        if duplicate_rows and not self.allow_duplicate_rows:
            violations.append(f"duplicate rows: {duplicate_rows}")

        missing_ratio = {str(c): float(v) for c, v in df.isna().mean().items()}
        excessive_missing = {
            column: ratio
            for column, ratio in missing_ratio.items()
            if ratio > self.max_missing_ratio
        }
        if excessive_missing:
            violations.append(f"columns above missing-value limit: {excessive_missing}")

        numeric = df.select_dtypes(include=[np.number])
        non_finite = int(np.isinf(numeric.to_numpy(dtype=float, copy=False)).sum())
        if non_finite:
            violations.append(f"non-finite numeric values: {non_finite}")

        duplicated_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicated_columns:
            violations.append(f"duplicate columns: {duplicated_columns}")

        for column in self.id_columns:
            if column in df.columns:
                null_ids = int(df[column].isna().sum())
                duplicate_ids = int(df[column].duplicated().sum())
                if null_ids:
                    violations.append(f"null identifiers in {column}: {null_ids}")
                if duplicate_ids:
                    violations.append(f"duplicate identifiers in {column}: {duplicate_ids}")

        if self.target_column and self.target_column in df.columns:
            values = set(df[self.target_column].dropna().unique().tolist())
            if values != {0, 1}:
                violations.append(
                    f"target {self.target_column} must contain both binary classes 0 and 1; "
                    f"found {sorted(values, key=str)}"
                )

        return DataQualityReport(
            passed=not violations,
            rows=len(df),
            columns=len(df.columns),
            duplicate_rows=duplicate_rows,
            missing_ratio=missing_ratio,
            violations=tuple(violations),
        )

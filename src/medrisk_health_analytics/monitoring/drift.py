"""Population Stability Index monitoring for tabular model inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftReport:
    """Per-feature drift measurements and the overall quality gate."""

    passed: bool
    drifted_features: tuple[str, ...]
    metrics: pd.DataFrame

    def raise_for_failure(self) -> None:
        """Raise when too many inputs moved beyond the configured threshold."""
        if not self.passed:
            raise ValueError(f"Data drift gate failed for: {list(self.drifted_features)}")


class DataDriftMonitor:
    """Compare a scoring population with its training reference using PSI."""

    def __init__(self, *, psi_threshold: float = 0.20, bins: int = 10) -> None:
        if psi_threshold <= 0:
            raise ValueError("psi_threshold must be positive")
        if bins < 2:
            raise ValueError("bins must be at least 2")
        self.psi_threshold = psi_threshold
        self.bins = bins

    def compare(self, reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
        """Calculate PSI for every shared feature."""
        if reference.empty or current.empty:
            raise ValueError("reference and current datasets must not be empty")
        shared = sorted(set(reference.columns) & set(current.columns))
        if not shared:
            raise ValueError("reference and current datasets have no shared columns")

        rows = []
        for column in shared:
            if pd.api.types.is_numeric_dtype(reference[column]):
                psi = self._numeric_psi(reference[column], current[column])
                kind = "numeric"
            else:
                psi = self._categorical_psi(reference[column], current[column])
                kind = "categorical"
            rows.append(
                {
                    "feature": column,
                    "feature_type": kind,
                    "psi": psi,
                    "drift_detected": psi >= self.psi_threshold,
                    "reference_rows": len(reference),
                    "current_rows": len(current),
                }
            )

        metrics = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
        drifted = tuple(metrics.loc[metrics["drift_detected"], "feature"].tolist())
        return DriftReport(passed=not drifted, drifted_features=drifted, metrics=metrics)

    def _numeric_psi(self, reference: pd.Series, current: pd.Series) -> float:
        ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
        cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
        if not len(ref) or not len(cur):
            return 0.0 if not len(ref) and not len(cur) else float("inf")
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, self.bins + 1)))
        if len(edges) < 2:
            return 0.0 if np.all(cur == ref[0]) else float("inf")
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)
        return self._psi_from_proportions(ref_counts, cur_counts)

    @staticmethod
    def _categorical_psi(reference: pd.Series, current: pd.Series) -> float:
        ref = reference.astype("string").fillna("__MISSING__")
        cur = current.astype("string").fillna("__MISSING__")
        categories = sorted(set(ref.unique()) | set(cur.unique()))
        ref_counts = ref.value_counts().reindex(categories, fill_value=0).to_numpy()
        cur_counts = cur.value_counts().reindex(categories, fill_value=0).to_numpy()
        return DataDriftMonitor._psi_from_proportions(ref_counts, cur_counts)

    @staticmethod
    def _psi_from_proportions(reference_counts: np.ndarray, current_counts: np.ndarray) -> float:
        epsilon = 1e-6
        ref = np.clip(reference_counts / reference_counts.sum(), epsilon, None)
        cur = np.clip(current_counts / current_counts.sum(), epsilon, None)
        return float(np.sum((cur - ref) * np.log(cur / ref)))

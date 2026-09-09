import numpy as np
import pandas as pd
import pytest

from medrisk_health_analytics import DataDriftMonitor, DataQualityValidator


def test_data_quality_contract_accepts_valid_binary_training_data():
    data = pd.DataFrame({"patient_id": [1, 2], "age": [40.0, 60.0], "target": [0, 1]})
    report = DataQualityValidator(
        required_columns=("patient_id", "age", "target"),
        target_column="target",
        id_columns=("patient_id",),
    ).validate(data)
    assert report.passed
    assert report.violations == ()


def test_data_quality_contract_reports_actionable_failures():
    data = pd.DataFrame({"patient_id": [1, 1], "age": [np.inf, np.inf], "target": [1, 1]})
    report = DataQualityValidator(
        required_columns=("patient_id", "age", "target", "missing"),
        target_column="target",
        id_columns=("patient_id",),
    ).validate(data)
    assert not report.passed
    assert any("missing required columns" in item for item in report.violations)
    assert any("non-finite" in item for item in report.violations)
    with pytest.raises(ValueError, match="Data quality contract failed"):
        report.raise_for_failure()


def test_drift_monitor_detects_shift_and_returns_feature_metrics():
    reference = pd.DataFrame(
        {"age": np.arange(100, dtype=float), "segment": ["A"] * 50 + ["B"] * 50}
    )
    current = pd.DataFrame({"age": np.arange(100, dtype=float) + 1000, "segment": ["C"] * 100})
    report = DataDriftMonitor(psi_threshold=0.2).compare(reference, current)
    assert not report.passed
    assert set(report.drifted_features) == {"age", "segment"}
    assert set(report.metrics.columns) >= {"feature", "psi", "drift_detected"}

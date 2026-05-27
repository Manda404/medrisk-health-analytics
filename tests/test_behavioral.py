"""Unit tests for BehavioralFeatureEngineer."""
import pandas as pd
import pytest
from medrisk_features.features import BehavioralFeatureEngineer


def test_behavioral_features_created(full_df, logger):
    fe = BehavioralFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)

    expected_columns = {
        "physical_activity_adequate",
        "screen_sleep_imbalance",
        "sedentary_risk",
    }
    for col in expected_columns:
        assert col in df_out.columns, f"Missing column: {col}"


def test_physical_activity_adequate_ratio(logger):
    """Someone doing 300 min/week (2x WHO) should have a ratio of 2.0."""
    df = pd.DataFrame({"physical_activity_minutes_per_week": [300, 150, 0]})
    fe = BehavioralFeatureEngineer(logger=logger)
    df_out = fe.transform(df)
    assert df_out["physical_activity_adequate"].iloc[0] == pytest.approx(2.0)
    assert df_out["physical_activity_adequate"].iloc[1] == pytest.approx(1.0)
    assert df_out["physical_activity_adequate"].iloc[2] == pytest.approx(0.0)


def test_physical_activity_adequate_capped_at_3(logger):
    """Values beyond 3x WHO should be capped at 3.0."""
    df = pd.DataFrame({"physical_activity_minutes_per_week": [9999]})
    fe = BehavioralFeatureEngineer(logger=logger)
    df_out = fe.transform(df)
    assert df_out["physical_activity_adequate"].iloc[0] == pytest.approx(3.0)


def test_sedentary_risk_high_screen_low_activity(logger):
    """High screen time + low activity → sedentary_risk = 1."""
    df = pd.DataFrame({
        "screen_time_hours_per_day": [8],
        "physical_activity_minutes_per_week": [30],
    })
    fe = BehavioralFeatureEngineer(logger=logger)
    df_out = fe.transform(df)
    assert df_out["sedentary_risk"].iloc[0] == 1


def test_sedentary_risk_active_person(logger):
    """Low screen + adequate activity → sedentary_risk = 0."""
    df = pd.DataFrame({
        "screen_time_hours_per_day": [2],
        "physical_activity_minutes_per_week": [200],
    })
    fe = BehavioralFeatureEngineer(logger=logger)
    df_out = fe.transform(df)
    assert df_out["sedentary_risk"].iloc[0] == 0


def test_missing_activity_column_no_crash(minimal_df, logger):
    """Missing behavioral columns should not raise exceptions."""
    fe = BehavioralFeatureEngineer(logger=logger)
    df_out = fe.transform(minimal_df)
    assert "physical_activity_adequate" not in df_out.columns
    assert "sedentary_risk" not in df_out.columns


def test_original_df_not_mutated(full_df, logger):
    original_cols = set(full_df.columns)
    fe = BehavioralFeatureEngineer(logger=logger)
    fe.transform(full_df)
    assert set(full_df.columns) == original_cols

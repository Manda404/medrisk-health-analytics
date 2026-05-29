"""Unit tests for LifestyleFeatureEngineer."""

import pandas as pd
import pytest
from medrisk_health_analytics.features import LifestyleFeatureEngineer


@pytest.fixture
def lifestyle_df():
    """DataFrame with all columns needed for lifestyle features."""
    return pd.DataFrame(
        {
            "diet_score": [8, 4],
            "physical_activity_minutes_per_week": [200, 50],
            "sleep_hours_per_day": [8, 5],
            "alcohol_consumption_per_week": [1, 6],
            "smoking_status": ["Never", "Current"],
            "screen_time_hours_per_day": [3, 10],
        }
    )


def test_lifestyle_score_created(lifestyle_df, logger):
    fe = LifestyleFeatureEngineer(logger=logger)
    df_out = fe.transform(lifestyle_df)
    assert "lifestyle_score" in df_out.columns


def test_lifestyle_score_perfect(logger):
    """A person meeting all 5 criteria should score 10."""
    df = pd.DataFrame(
        {
            "diet_score": [8],
            "physical_activity_minutes_per_week": [200],
            "sleep_hours_per_day": [8],
            "alcohol_consumption_per_week": [1],
            "smoking_status": ["Never"],
            "screen_time_hours_per_day": [2],
        }
    )
    fe = LifestyleFeatureEngineer(logger=logger)
    df_out = fe.transform(df)
    assert df_out["lifestyle_score"].iloc[0] == 10


def test_lifestyle_score_zero(logger):
    """A person meeting no criteria should score 0."""
    df = pd.DataFrame(
        {
            "diet_score": [2],
            "physical_activity_minutes_per_week": [10],
            "sleep_hours_per_day": [4],
            "alcohol_consumption_per_week": [10],
            "smoking_status": ["Current"],
            "screen_time_hours_per_day": [12],
        }
    )
    fe = LifestyleFeatureEngineer(logger=logger)
    df_out = fe.transform(df)
    assert df_out["lifestyle_score"].iloc[0] == 0


def test_lifestyle_score_range(lifestyle_df, logger):
    """Lifestyle score must always be between 0 and 10."""
    fe = LifestyleFeatureEngineer(logger=logger)
    df_out = fe.transform(lifestyle_df)
    assert df_out["lifestyle_score"].between(0, 10).all()


def test_sleep_efficiency_created(lifestyle_df, logger):
    fe = LifestyleFeatureEngineer(logger=logger)
    df_out = fe.transform(lifestyle_df)
    assert "sleep_efficiency" in df_out.columns


def test_sleep_efficiency_capped(logger):
    """sleep_efficiency should never exceed 2.0."""
    df = pd.DataFrame(
        {
            "sleep_hours_per_day": [9],
            "screen_time_hours_per_day": [0],
        }
    )
    fe = LifestyleFeatureEngineer(logger=logger)
    df_out = fe.transform(df)
    assert df_out["sleep_efficiency"].iloc[0] <= 2.0


def test_missing_lifestyle_columns_no_crash(minimal_df, logger):
    """Missing lifestyle columns should not raise exceptions."""
    fe = LifestyleFeatureEngineer(logger=logger)
    df_out = fe.transform(minimal_df)
    assert "lifestyle_score" not in df_out.columns


def test_original_df_not_mutated(lifestyle_df, logger):
    original_cols = set(lifestyle_df.columns)
    fe = LifestyleFeatureEngineer(logger=logger)
    fe.transform(lifestyle_df)
    assert set(lifestyle_df.columns) == original_cols

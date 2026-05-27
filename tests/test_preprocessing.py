"""Unit tests for preprocessing modules (categorical cleaning and leakage removal)."""

import pandas as pd
import pytest
from medrisk_features.preprocessing import clean_categorical_variables, drop_leakage_columns


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "Age": [40, 55],
            "gender": ["Male", "Other"],
            "employment_status": ["Retired", "Employed"],
            "smoking_status": ["Former", "Never"],
            "glucose_fasting": [110, 130],
            "bmi": [27, 32],
            "diabetes_stage": ["Stage 1", "Stage 2"],
            "diabetes_risk_score": [0.7, 0.9],
        }
    )


# ---------------------------------------------------------------------------
# Categorical cleaning tests
# ---------------------------------------------------------------------------


class TestCategoricalCleaning:
    def test_gender_other_mapped_to_unknown(self, raw_df, logger):
        df_out = clean_categorical_variables(raw_df, logger=logger)
        assert "Unknown" in df_out["gender"].values
        assert "Other" not in df_out["gender"].values

    def test_employment_retired_mapped_to_inactive(self, raw_df, logger):
        df_out = clean_categorical_variables(raw_df, logger=logger)
        assert "Inactive" in df_out["employment_status"].values
        assert "Retired" not in df_out["employment_status"].values

    def test_smoking_former_mapped_to_ex_smoker(self, raw_df, logger):
        df_out = clean_categorical_variables(raw_df, logger=logger)
        assert "Ex-Smoker" in df_out["smoking_status"].values
        assert "Former" not in df_out["smoking_status"].values

    def test_non_categorical_columns_unchanged(self, raw_df, logger):
        df_out = clean_categorical_variables(raw_df, logger=logger)
        pd.testing.assert_series_equal(df_out["glucose_fasting"], raw_df["glucose_fasting"])

    def test_original_df_not_mutated(self, raw_df, logger):
        original_gender = raw_df["gender"].copy()
        clean_categorical_variables(raw_df, logger=logger)
        pd.testing.assert_series_equal(raw_df["gender"], original_gender)


# ---------------------------------------------------------------------------
# Leakage removal tests
# ---------------------------------------------------------------------------


class TestLeakageRemoval:
    def test_leakage_columns_dropped(self, raw_df, logger):
        df_out = drop_leakage_columns(raw_df, logger=logger)
        assert "diabetes_stage" not in df_out.columns
        assert "diabetes_risk_score" not in df_out.columns

    def test_non_leakage_columns_preserved(self, raw_df, logger):
        df_out = drop_leakage_columns(raw_df, logger=logger)
        assert "glucose_fasting" in df_out.columns
        assert "bmi" in df_out.columns

    def test_no_leakage_columns_no_change(self, logger):
        df_clean = pd.DataFrame(
            {
                "Age": [40],
                "glucose_fasting": [110],
                "bmi": [27],
            }
        )
        df_out = drop_leakage_columns(df_clean, logger=logger)
        assert set(df_out.columns) == set(df_clean.columns)

    def test_original_df_not_mutated(self, raw_df, logger):
        original_cols = set(raw_df.columns)
        drop_leakage_columns(raw_df, logger=logger)
        assert set(raw_df.columns) == original_cols

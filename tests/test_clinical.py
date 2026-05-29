"""Unit tests for ClinicalFeatureEngineer."""

import pandas as pd
from medrisk_health_analytics.features import ClinicalFeatureEngineer


def test_lipid_ratio_hdl_ldl_created(full_df, logger):
    fe = ClinicalFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    assert "lipid_ratio_hdl_ldl" in df_out.columns


def test_lipid_ratio_hdl_ldl_values(full_df, logger):
    fe = ClinicalFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    # HDL / LDL should be positive for valid inputs
    assert (df_out["lipid_ratio_hdl_ldl"] > 0).all()


def test_cholesterol_hdl_ratio_created(full_df, logger):
    fe = ClinicalFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    assert "cholesterol_hdl_ratio" in df_out.columns


def test_bmi_glucose_interaction_created(full_df, logger):
    fe = ClinicalFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    assert "bmi_glucose_interaction" in df_out.columns


def test_bmi_glucose_interaction_value(full_df, logger):
    fe = ClinicalFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    # bmi_glucose_interaction = bmi * glucose_fasting
    expected = full_df["bmi"] * full_df["glucose_fasting"]
    pd.testing.assert_series_equal(
        df_out["bmi_glucose_interaction"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_glucose_variability_created(full_df, logger):
    fe = ClinicalFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    assert "glucose_variability" in df_out.columns


def test_glucose_variability_value(full_df, logger):
    fe = ClinicalFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    expected = full_df["glucose_postprandial"] - full_df["glucose_fasting"]
    pd.testing.assert_series_equal(
        df_out["glucose_variability"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_missing_cholesterol_no_ratio(minimal_df, logger):
    """lipid_ratio_hdl_ldl should not be created when cholesterol columns are absent."""
    fe = ClinicalFeatureEngineer(logger=logger)
    df_out = fe.transform(minimal_df)
    assert "lipid_ratio_hdl_ldl" not in df_out.columns
    assert "cholesterol_hdl_ratio" not in df_out.columns


def test_original_df_not_mutated(full_df, logger):
    original_cols = set(full_df.columns)
    fe = ClinicalFeatureEngineer(logger=logger)
    fe.transform(full_df)
    assert set(full_df.columns) == original_cols

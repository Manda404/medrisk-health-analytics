"""Unit tests for MedicalFeatureEngineer."""
import pandas as pd
import pytest
from medrisk_features.features import MedicalFeatureEngineer


def test_medical_features_created(full_df, logger):
    fe = MedicalFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)

    expected_columns = {
        "glucose_category",
        "hba1c_category",
        "homa_ir",
        "insulin_resistance_flag",
        "bmi_category",
        "bp_category",
        "metabolic_syndrome_flag",
    }
    for col in expected_columns:
        assert col in df_out.columns, f"Missing column: {col}"


def test_glucose_category_values(full_df, logger):
    """glucose_category values must follow ADA classification."""
    fe = MedicalFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    valid_values = {"Normal", "Pre-Diabetes", "Diabetes"}
    assert set(df_out["glucose_category"].dropna().astype(str)).issubset(valid_values)


def test_bmi_category_values(full_df, logger):
    fe = MedicalFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    valid_values = {"Underweight", "Normal", "Overweight", "Obese"}
    assert set(df_out["bmi_category"].dropna().astype(str)).issubset(valid_values)


def test_bp_category_values(full_df, logger):
    fe = MedicalFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    valid_values = {"Normal", "Pre-Hypertension", "Hypertension"}
    assert set(df_out["bp_category"].dropna().astype(str)).issubset(valid_values)


def test_homa_ir_positive(full_df, logger):
    fe = MedicalFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    assert (df_out["homa_ir"] > 0).all()


def test_insulin_resistance_flag_binary(full_df, logger):
    fe = MedicalFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    assert df_out["insulin_resistance_flag"].isin([0, 1]).all()


def test_metabolic_syndrome_flag_binary(full_df, logger):
    fe = MedicalFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    assert df_out["metabolic_syndrome_flag"].isin([0, 1]).all()


def test_missing_glucose_fasting_raises_key_error(full_df, logger):
    df = full_df.drop(columns=["glucose_fasting"])
    fe = MedicalFeatureEngineer(logger=logger)
    with pytest.raises(KeyError):
        fe.transform(df)


def test_missing_insulin_level_skips_homa_ir(minimal_df, logger):
    """homa_ir should not be computed when insulin_level is absent."""
    fe = MedicalFeatureEngineer(logger=logger)
    df_out = fe.transform(minimal_df)
    assert "homa_ir" not in df_out.columns


def test_original_df_not_mutated(full_df, logger):
    original_cols = set(full_df.columns)
    fe = MedicalFeatureEngineer(logger=logger)
    fe.transform(full_df)
    assert set(full_df.columns) == original_cols

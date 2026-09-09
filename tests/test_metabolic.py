"""Unit tests for MetabolicFeatureEngineer."""

from medrisk_health_analytics.features import MetabolicFeatureEngineer


def test_metabolic_features_created(full_df, logger):
    fe = MetabolicFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)

    expected_columns = {
        "tyg_index",
        "tyg_bmi_index",
        "dyslipidemia_flag",
        "cardiometabolic_burden",
        "blood_pressure_ratio",
    }
    for col in expected_columns:
        assert col in df_out.columns, f"Missing column: {col}"


def test_tyg_index_is_finite_and_positive(full_df, logger):
    fe = MetabolicFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    assert df_out["tyg_index"].notna().all()
    assert (df_out["tyg_index"] > 0).all()


def test_dyslipidemia_flag_binary(full_df, logger):
    fe = MetabolicFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    assert df_out["dyslipidemia_flag"].isin([0, 1]).all()


def test_cardiometabolic_burden_range(full_df, logger):
    fe = MetabolicFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    # Score is the sum of 5 binary flags → must be in [0, 5]
    assert df_out["cardiometabolic_burden"].between(0, 5).all()


def test_blood_pressure_ratio_positive(full_df, logger):
    fe = MetabolicFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)
    assert (df_out["blood_pressure_ratio"] > 0).all()


def test_missing_columns_gracefully_skipped(minimal_df, logger):
    """Features requiring optional columns should be skipped, not crash."""
    fe = MetabolicFeatureEngineer(logger=logger)
    df_out = fe.transform(minimal_df)
    assert "tyg_index" not in df_out.columns
    # dyslipidemia_flag requires triglycerides and hdl_cholesterol → absent
    assert "dyslipidemia_flag" not in df_out.columns


def test_original_df_not_mutated(full_df, logger):
    original_cols = set(full_df.columns)
    fe = MetabolicFeatureEngineer(logger=logger)
    fe.transform(full_df)
    assert set(full_df.columns) == original_cols

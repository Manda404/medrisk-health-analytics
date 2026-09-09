from medrisk_health_analytics import FeatureEngineeringPipeline


def test_pipeline_runs_end_to_end(full_df):
    pipeline = FeatureEngineeringPipeline()
    df_out = pipeline.transform(full_df)

    # Sanity checks
    assert df_out.shape[0] == full_df.shape[0]
    assert df_out.shape[1] > full_df.shape[1]

    # Key features
    assert "age_group" in df_out.columns
    assert "metabolic_syndrome_flag" in df_out.columns
    assert "lifestyle_score" in df_out.columns


def test_pipeline_outputs_mlflow_safe_string_categories(full_df):
    pipeline = FeatureEngineeringPipeline()
    df_out = pipeline.transform(full_df)

    assert str(df_out["age_group"].dtype) == "object"
    assert str(df_out["glucose_category"].dtype) == "object"

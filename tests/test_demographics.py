import pytest

from medrisk_health_analytics.features import DemographicsFeatureEngineer


def test_demographics_features_created(full_df, logger):
    fe = DemographicsFeatureEngineer(logger=logger)
    df_out = fe.transform(full_df)

    assert "age_group" in df_out.columns
    assert "age_squared" in df_out.columns
    assert "socioeconomic_vulnerability_flag" in df_out.columns


def test_missing_age_raises_error(full_df, logger):
    df = full_df.drop(columns=["Age"])
    fe = DemographicsFeatureEngineer(logger=logger)

    with pytest.raises(KeyError):
        fe.transform(df)


def test_lowercase_age_is_supported(full_df, logger):
    df = full_df.rename(columns={"Age": "age"})
    fe = DemographicsFeatureEngineer(logger=logger)

    df_out = fe.transform(df)

    assert "age_group" in df_out.columns
    assert "age_squared" in df_out.columns

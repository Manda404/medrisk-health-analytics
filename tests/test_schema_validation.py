import pandas as pd
import pytest

from medrisk_health_analytics.validation import DataSchemaValidator, SchemaValidationError


def test_schema_validation_passes(minimal_df, logger):
    validator = DataSchemaValidator(logger=logger)
    validator.validate(minimal_df)


def test_schema_validation_accepts_lowercase_age(minimal_df, logger):
    df = minimal_df.rename(columns={"Age": "age"})
    validator = DataSchemaValidator(logger=logger)

    validator.validate(df)


def test_schema_validation_fails(minimal_df, logger):
    df = minimal_df.drop(columns=["glucose_fasting"])
    validator = DataSchemaValidator(logger=logger)

    with pytest.raises(SchemaValidationError):
        validator.validate(df)


def test_schema_validation_rejects_non_numeric_core_column(minimal_df, logger):
    minimal_df["bmi"] = ["normal", "high"]

    with pytest.raises(SchemaValidationError, match="must be numeric"):
        DataSchemaValidator(logger=logger).validate(minimal_df)


def test_schema_validation_rejects_duplicate_columns(minimal_df, logger):
    duplicate_df = pd.concat([minimal_df, minimal_df[["bmi"]]], axis=1)

    with pytest.raises(SchemaValidationError, match="Duplicate column names"):
        DataSchemaValidator(logger=logger).validate(duplicate_df)


def test_schema_validation_can_enforce_physiological_ranges(minimal_df, logger):
    minimal_df.loc[0, "bmi"] = 200

    with pytest.raises(SchemaValidationError, match="expected range"):
        DataSchemaValidator(logger=logger, strict_ranges=True).validate(minimal_df)

import pytest
from medrisk_features.validation import DataSchemaValidator, SchemaValidationError


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

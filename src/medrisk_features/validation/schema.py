from typing import Set

from pandas import DataFrame

from medrisk_features.logging import get_logger
from medrisk_features.utils.exceptions import (
    MissingRequiredColumnError,
    SchemaValidationError,
)

# Re-export SchemaValidationError so existing imports from this module still work
__all__ = ["DataSchemaValidator", "SchemaValidationError"]


class DataSchemaValidator:
    """
    Validate input dataset schema before feature engineering.

    Purpose
    -------
    Detect missing critical columns early and prevent
    silent failures or inconsistent feature generation.

    Raises explicit, actionable errors rather than letting
    downstream steps fail with cryptic KeyErrors.
    """

    # Minimal required columns for the full pipeline.
    # `Age` is kept for backward compatibility with existing examples, while
    # `age` matches the provided data/patient.csv source file.
    REQUIRED_COLUMNS: Set[str] = {
        "glucose_fasting",
        "bmi",
    }
    REQUIRED_ALTERNATIVES = ({"Age", "age"},)

    def __init__(self, logger=None):
        self.logger = logger or get_logger(self.__class__.__name__)

    def validate_required_columns(self, df: DataFrame) -> None:
        """
        Check that all required columns are present in the DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input dataset.

        Raises
        ------
        MissingRequiredColumnError
            If one or more required columns are absent.
        """
        available_columns = set(df.columns)
        missing = self.REQUIRED_COLUMNS - available_columns
        for alternatives in self.REQUIRED_ALTERNATIVES:
            if not alternatives & available_columns:
                missing.add("/".join(sorted(alternatives)))

        if missing:
            self.logger.error(f"Missing required columns: {sorted(missing)}")
            raise MissingRequiredColumnError(list(missing))

        self.logger.info("All required columns are present.")

    def validate(self, df: DataFrame) -> None:
        """
        Run all schema validations.

        Parameters
        ----------
        df : DataFrame
            Input dataset.

        Raises
        ------
        SchemaValidationError
            If any schema validation check fails.
        """
        self.logger.info("Validating input data schema...")
        self.validate_required_columns(df)
        self.logger.info("Schema validation completed successfully.")

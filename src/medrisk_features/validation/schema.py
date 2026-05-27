from typing import Set
from pandas import DataFrame
from medrisk_features.logging import get_logger
from medrisk_features.utils.exceptions import (
    SchemaValidationError,
    MissingRequiredColumnError,
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

    # Minimal required columns for the full pipeline
    REQUIRED_COLUMNS: Set[str] = {
        "Age",
        "glucose_fasting",
        "bmi",
    }

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
        missing = self.REQUIRED_COLUMNS - set(df.columns)
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

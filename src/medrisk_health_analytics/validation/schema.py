from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from medrisk_health_analytics.logging import get_logger
from medrisk_health_analytics.utils.exceptions import (
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
    NUMERIC_RANGES: Dict[str, Tuple[float, float]] = {
        "Age": (0, 120),
        "age": (0, 120),
        "bmi": (10, 80),
        "glucose_fasting": (40, 600),
        "hba1c": (2, 25),
        "systolic_bp": (50, 300),
        "diastolic_bp": (30, 200),
    }

    def __init__(self, logger=None, strict_ranges: bool = False):
        self.logger = logger or get_logger(self.__class__.__name__)
        self.strict_ranges = strict_ranges

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

    def validate_column_names(self, df: DataFrame) -> None:
        """Reject duplicate labels because pandas may otherwise select ambiguous data."""
        duplicates = sorted(
            {
                str(column)
                for column, duplicated in zip(df.columns, df.columns.duplicated(), strict=True)
                if duplicated
            }
        )
        if duplicates:
            raise SchemaValidationError(f"Duplicate column names: {duplicates}")

    def validate_numeric_columns(self, df: DataFrame) -> None:
        """Validate numeric types and detect infinite or implausible values."""
        for column, (minimum, maximum) in self.NUMERIC_RANGES.items():
            if column not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise SchemaValidationError(f"Column '{column}' must be numeric")

            finite_values = df[column].dropna()
            if np.isinf(finite_values.to_numpy()).any():
                raise SchemaValidationError(f"Column '{column}' contains infinite values")

            invalid_count = int((~finite_values.between(minimum, maximum)).sum())
            if invalid_count:
                message = (
                    f"Column '{column}' contains {invalid_count} value(s) outside "
                    f"the expected range [{minimum}, {maximum}]"
                )
                if self.strict_ranges:
                    raise SchemaValidationError(message)
                self.logger.warning(message)

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
        if not isinstance(df, DataFrame):
            raise SchemaValidationError("Input must be a pandas DataFrame")
        self.validate_column_names(df)
        self.validate_required_columns(df)
        self.validate_numeric_columns(df)
        self.logger.info("Schema validation completed successfully.")

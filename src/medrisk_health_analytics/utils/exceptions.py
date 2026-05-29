"""
Custom exceptions for the medrisk-health-analytics package.

All exceptions inherit from MedRiskError to allow catching
the entire package's error family with a single except clause.
"""


class MedRiskError(Exception):
    """Base exception for all medrisk-health-analytics errors."""


class SchemaValidationError(MedRiskError):
    """
    Raised when the input DataFrame does not satisfy the
    required schema (missing columns, wrong types, etc.).

    Example
    -------
    >>> raise SchemaValidationError("Missing required columns: ['bmi', 'Age']")
    """


class FeatureEngineeringError(MedRiskError):
    """
    Raised when a feature engineering step fails due to
    unexpected data conditions (e.g., all-NaN column, invalid range).
    """


class MissingRequiredColumnError(SchemaValidationError):
    """
    Raised specifically when one or more required columns
    are absent from the input DataFrame.

    Parameters
    ----------
    missing_columns : list[str]
        Names of the columns that are missing.

    Example
    -------
    >>> raise MissingRequiredColumnError(["glucose_fasting", "bmi"])
    """

    def __init__(self, missing_columns: list) -> None:
        self.missing_columns = missing_columns
        message = f"Missing required columns: {sorted(missing_columns)}"
        super().__init__(message)


class InvalidConfigurationError(MedRiskError):
    """
    Raised when an invalid configuration value is passed to
    a pipeline component (e.g., unknown age_group_strategy).

    Example
    -------
    >>> raise InvalidConfigurationError(
    ...     "age_group_strategy", "unknown", ["detailed", "coarse"]
    ... )
    """

    def __init__(
        self,
        parameter: str,
        value: object,
        valid_values: list,
    ) -> None:
        self.parameter = parameter
        self.value = value
        self.valid_values = valid_values
        message = (
            f"Invalid value '{value}' for parameter '{parameter}'. "
            f"Valid options: {valid_values}"
        )
        super().__init__(message)

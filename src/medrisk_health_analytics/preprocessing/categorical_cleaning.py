from pandas import DataFrame

from medrisk_health_analytics.logging import get_logger


def clean_categorical_variables(
    df: DataFrame,
    logger=None,
) -> DataFrame:
    """
    Clean and harmonize categorical variable labels.

    Purpose
    -------
    Reduce semantic variability and ensure consistency
    before encoding or modeling.

    Medical relevance
    ------------------
    In public health, inconsistent labels (e.g. "Former" vs "Ex-Smoker")
    can lead to incorrect subgroup segmentation.

    Parameters
    ----------
    df : DataFrame
        Input dataset.
    logger :
        Optional Loguru logger instance.

    Returns
    -------
    DataFrame
        Cleaned dataset.
    """
    logger = logger or get_logger("categorical-cleaning")
    df = df.copy()

    logger.info("Cleaning categorical variables...")

    if "gender" in df.columns:
        df["gender"] = df["gender"].replace({"Other": "Unknown"})

    if "employment_status" in df.columns:
        df["employment_status"] = df["employment_status"].replace(
            {"Retired": "Inactive", "Unemployed": "Inactive"}
        )

    if "smoking_status" in df.columns:
        df["smoking_status"] = df["smoking_status"].replace({"Former": "Ex-Smoker"})

    logger.info("Categorical variables cleaned successfully.")
    return df

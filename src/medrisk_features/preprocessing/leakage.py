from pandas import DataFrame

from medrisk_features.logging import get_logger

LEAKAGE_COLUMNS = {
    "diabetes_stage",
    "diabetes_risk_score",
    # "diagnosed_diabetes",  # target → jamais dans X
}


def drop_leakage_columns(
    df: DataFrame,
    logger=None,
) -> DataFrame:
    """
    Remove columns that cause data leakage.

    Purpose
    -------
    Prevent features that directly or indirectly encode
    the prediction target from leaking into training data.

    Parameters
    ----------
    df : DataFrame
        Input dataset.
    logger :
        Optional Loguru logger instance.

    Returns
    -------
    DataFrame
        Dataset without leakage columns.
    """
    logger = logger or get_logger("leakage-removal")
    df = df.copy()

    present_columns = [c for c in LEAKAGE_COLUMNS if c in df.columns]

    if present_columns:
        logger.warning(f"Removing leakage columns: {present_columns}")
        df = df.drop(columns=present_columns)
    else:
        logger.info("No leakage columns detected.")

    return df

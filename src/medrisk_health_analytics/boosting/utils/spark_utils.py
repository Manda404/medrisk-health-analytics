"""
Spark ↔ Pandas conversion utilities for Databricks environments.

These helpers abstract the Spark/Pandas boundary so the rest of the
pipeline can work with plain Pandas DataFrames while still accepting
Spark DataFrames as inputs from Databricks notebooks.

Design rules
------------
1. Never import pyspark at module level — it crashes in non-Spark environments.
2. Always warn when converting large DataFrames (>10M rows).
3. Provide a safe estimate of DataFrame row count before conversion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import pandas as pd

if TYPE_CHECKING:
    # pyspark type hints without runtime import
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
    except ImportError:
        SparkDataFrame = None  # type: ignore

# Threshold above which a warning is emitted before .toPandas()
_LARGE_DF_ROW_WARNING = 5_000_000


def is_spark_dataframe(obj: object) -> bool:
    """Return True if obj is a PySpark DataFrame, without importing pyspark."""
    return type(obj).__name__ == "DataFrame" and hasattr(obj, "toPandas")


def to_pandas(
    df: Union[pd.DataFrame, SparkDataFrame],
    warn_rows: int = _LARGE_DF_ROW_WARNING,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convert a Spark or Pandas DataFrame to a Pandas DataFrame.

    If the input is already a Pandas DataFrame it is returned unchanged.

    Parameters
    ----------
    df : pd.DataFrame or pyspark.sql.DataFrame
        Input data. Accepts both Spark and Pandas DataFrames.
    warn_rows : int
        If the Spark DataFrame has more rows than this threshold, emit a
        warning before converting. Default: 5,000,000.
    sample_size : int or None
        If provided, sample this many rows from the Spark DataFrame before
        converting (useful for exploration and validation, not production).

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame ready for the training/inference pipeline.

    Raises
    ------
    TypeError
        If df is neither a Pandas nor a Spark DataFrame.
    """
    if isinstance(df, pd.DataFrame):
        return df

    if is_spark_dataframe(df):
        return _spark_to_pandas(df, warn_rows=warn_rows, sample_size=sample_size)

    raise TypeError(
        f"Expected a pandas or PySpark DataFrame, got {type(df).__name__}. "
        "Please convert your data to a supported format before calling this function."
    )


def to_spark(
    df: pd.DataFrame,
    spark=None,
):
    """
    Convert a Pandas DataFrame to a Spark DataFrame.

    Parameters
    ----------
    df : pd.DataFrame  Input Pandas DataFrame.
    spark : SparkSession or None
        Active SparkSession. If None, uses SparkSession.getActiveSession().

    Returns
    -------
    pyspark.sql.DataFrame
    """
    spark_session = spark or _get_active_spark_session()
    if spark_session is None:
        raise RuntimeError(
            "No active SparkSession found. "
            "Pass spark=spark explicitly or ensure Databricks is active."
        )
    return spark_session.createDataFrame(df)


def write_to_delta(
    df: pd.DataFrame,
    table_name: str,
    mode: str = "overwrite",
    spark=None,
) -> None:
    """
    Write a Pandas DataFrame to a Unity Catalog Delta table.

    Parameters
    ----------
    df : pd.DataFrame       Data to write.
    table_name : str        Full table name: 'catalog.schema.table'.
    mode : str              Spark write mode: 'overwrite', 'append'.
    spark : SparkSession    Active session (auto-detected if None).
    """
    spark_df = to_spark(df, spark=spark)
    (
        spark_df.write.format("delta")
        .mode(mode)
        .option("mergeSchema", "true")
        .saveAsTable(table_name)
    )


def read_from_delta(
    table_name: str,
    spark=None,
    convert_to_pandas: bool = True,
    sample_size: Optional[int] = None,
) -> Union[pd.DataFrame, SparkDataFrame]:
    """
    Read a Unity Catalog Delta table.

    Parameters
    ----------
    table_name : str          'catalog.schema.table'
    spark : SparkSession      Active session (auto-detected if None).
    convert_to_pandas : bool  If True, returns a Pandas DataFrame.
    sample_size : int or None If provided, sample before conversion.

    Returns
    -------
    pd.DataFrame or pyspark.sql.DataFrame
    """
    spark_session = spark or _get_active_spark_session()
    if spark_session is None:
        raise RuntimeError(
            "No active SparkSession found. "
            "Pass spark=spark explicitly or ensure Databricks is active."
        )

    spark_df = spark_session.table(table_name)

    if convert_to_pandas:
        return _spark_to_pandas(spark_df, sample_size=sample_size)

    return spark_df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _spark_to_pandas(
    spark_df,
    warn_rows: int = _LARGE_DF_ROW_WARNING,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """Internal: convert Spark → Pandas with optional sampling and warning."""
    import warnings

    if sample_size is not None:
        spark_df = spark_df.limit(sample_size)

    # Warn for large DataFrames (count() can be expensive — use explain heuristic)
    try:
        n_rows = spark_df.count()
        if n_rows > warn_rows:
            warnings.warn(
                f"Converting a large Spark DataFrame ({n_rows:,} rows) to Pandas. "
                "This may consume significant driver memory. "
                "Consider sampling with sample_size= or training on Spark directly.",
                ResourceWarning,
                stacklevel=3,
            )
    except Exception:
        pass  # count() may fail on some DataFrame types — non-blocking

    return spark_df.toPandas()


def _get_active_spark_session():
    """Return the active SparkSession or None if not in a Spark environment."""
    try:
        from pyspark.sql import SparkSession

        return SparkSession.getActiveSession()
    except ImportError:
        return None

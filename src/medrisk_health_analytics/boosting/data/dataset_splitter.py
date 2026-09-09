"""
Dataset splitting utilities for Databricks / Unity Catalog workflows.

This module covers the step before model training:
raw Unity Catalog table -> stratified train/test split -> Delta train/test tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from medrisk_health_analytics.boosting.utils.spark_utils import is_spark_dataframe


@dataclass
class DatasetSplitResult:
    """In-memory result returned by split_dataframe()."""

    train_df: Any
    test_df: Any
    train_count: int
    test_count: int
    target_column: str
    train_target_distribution: dict[str, int] = field(default_factory=dict)
    test_target_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class SplitTablesResult:
    """Unity Catalog table result returned by create_train_test_tables()."""

    source_table: str
    train_table: str
    test_table: str
    train_count: int
    test_count: int
    target_column: str
    train_target_distribution: dict[str, int] = field(default_factory=dict)
    test_target_distribution: dict[str, int] = field(default_factory=dict)


def split_dataframe(
    df: Any,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> DatasetSplitResult:
    """
    Split a Pandas or Spark DataFrame into train and test sets.

    Parameters
    ----------
    df : pandas.DataFrame or pyspark.sql.DataFrame
        Source dataset containing features and the target column.
    target_column : str
        Target column used for stratification.
    test_size : float
        Fraction of rows assigned to the test set. Must be between 0 and 1.
    random_state : int
        Seed used for reproducible splits.
    stratify : bool
        If True, preserve the target distribution as closely as possible.

    Returns
    -------
    DatasetSplitResult
        Train/test DataFrames plus row counts and target distributions.
    """
    _validate_split_inputs(df, target_column, test_size)

    if isinstance(df, pd.DataFrame):
        return _split_pandas_dataframe(
            df=df,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    if is_spark_dataframe(df):
        return _split_spark_dataframe(
            df=df,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    raise TypeError(f"Expected a pandas or PySpark DataFrame, got {type(df).__name__}.")


def create_train_test_tables(
    source_table: str,
    train_table: str,
    test_table: str,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    mode: str = "overwrite",
    spark=None,
) -> SplitTablesResult:
    """
    Read a Unity Catalog source table, split it, and save train/test Delta tables.

    Parameters
    ----------
    source_table : str
        Full Unity Catalog source table name, e.g. "catalog.schema.raw_patients".
    train_table : str
        Full output table name for the train split.
    test_table : str
        Full output table name for the test split.
    target_column : str
        Target column used for stratification.
    test_size : float
        Fraction of rows assigned to the test set.
    random_state : int
        Seed used for reproducible splits.
    stratify : bool
        If True, split within each target class.
    mode : str
        Spark save mode, typically "overwrite" or "append".
    spark : SparkSession or None
        Active SparkSession. If None, uses SparkSession.getActiveSession().

    Returns
    -------
    SplitTablesResult
        Output table names, row counts, and target distributions.
    """
    spark_session = spark or _get_active_spark_session()
    if spark_session is None:
        raise RuntimeError(
            "No active SparkSession found. Pass spark=spark explicitly or run inside Databricks."
        )

    source_df = spark_session.table(source_table)
    split = split_dataframe(
        source_df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    _write_spark_table(split.train_df, train_table, mode=mode)
    _write_spark_table(split.test_df, test_table, mode=mode)

    return SplitTablesResult(
        source_table=source_table,
        train_table=train_table,
        test_table=test_table,
        train_count=split.train_count,
        test_count=split.test_count,
        target_column=target_column,
        train_target_distribution=split.train_target_distribution,
        test_target_distribution=split.test_target_distribution,
    )


def _split_pandas_dataframe(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> DatasetSplitResult:
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for Pandas stratified splitting. "
            "Install with: pip install medrisk-health-analytics[boosting]"
        ) from exc

    stratify_values = df[target_column] if stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return DatasetSplitResult(
        train_df=train_df,
        test_df=test_df,
        train_count=len(train_df),
        test_count=len(test_df),
        target_column=target_column,
        train_target_distribution=_pandas_target_distribution(train_df, target_column),
        test_target_distribution=_pandas_target_distribution(test_df, target_column),
    )


def _split_spark_dataframe(
    df: Any,
    target_column: str,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> DatasetSplitResult:
    from pyspark.sql import Window
    from pyspark.sql import functions as f

    rand_col = "__medrisk_split_rand"
    is_test_col = "__medrisk_is_test"

    working = df.withColumn(rand_col, f.rand(random_state))

    if stratify:
        count_col = "__medrisk_class_count"
        row_col = "__medrisk_class_row"
        test_n_col = "__medrisk_class_test_n"

        class_window = Window.partitionBy(target_column)
        order_window = Window.partitionBy(target_column).orderBy(rand_col)

        working = (
            working.withColumn(count_col, f.count("*").over(class_window))
            .withColumn(row_col, f.row_number().over(order_window))
            .withColumn(
                test_n_col,
                f.when(f.col(count_col) <= 1, f.lit(0)).otherwise(
                    f.least(
                        f.col(count_col) - 1,
                        f.greatest(
                            f.lit(1),
                            f.floor(f.col(count_col) * f.lit(test_size)).cast("int"),
                        ),
                    )
                ),
            )
            .withColumn(is_test_col, f.col(row_col) <= f.col(test_n_col))
        )
        drop_cols = [rand_col, is_test_col, count_col, row_col, test_n_col]
    else:
        working = working.withColumn(is_test_col, f.col(rand_col) < f.lit(test_size))
        drop_cols = [rand_col, is_test_col]

    train_df = working.where(~f.col(is_test_col)).drop(*drop_cols)
    test_df = working.where(f.col(is_test_col)).drop(*drop_cols)

    return DatasetSplitResult(
        train_df=train_df,
        test_df=test_df,
        train_count=train_df.count(),
        test_count=test_df.count(),
        target_column=target_column,
        train_target_distribution=_spark_target_distribution(train_df, target_column),
        test_target_distribution=_spark_target_distribution(test_df, target_column),
    )


def _validate_split_inputs(df: Any, target_column: str, test_size: float) -> None:
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")

    if isinstance(df, pd.DataFrame):
        columns = set(df.columns)
    elif is_spark_dataframe(df):
        columns = set(df.columns)
    else:
        return

    if target_column not in columns:
        raise KeyError(f"Target column '{target_column}' not found in input dataset.")


def _pandas_target_distribution(df: pd.DataFrame, target_column: str) -> dict[str, int]:
    return {
        str(target): int(count)
        for target, count in df[target_column].value_counts(dropna=False).sort_index().items()
    }


def _spark_target_distribution(df: Any, target_column: str) -> dict[str, int]:
    rows = df.groupBy(target_column).count().collect()
    return {str(row[target_column]): int(row["count"]) for row in rows}


def _write_spark_table(df: Any, table_name: str, mode: str) -> None:
    (df.write.format("delta").mode(mode).option("overwriteSchema", "true").saveAsTable(table_name))


def _get_active_spark_session():
    try:
        from pyspark.sql import SparkSession

        return SparkSession.getActiveSession()
    except ImportError:
        return None

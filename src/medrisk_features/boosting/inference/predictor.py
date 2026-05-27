"""
predict_with_registered_model() — high-level inference entry point.

Loads a registered BoostingPyFuncModel from MLflow and applies it
to raw input data (Spark or Pandas), returning a clean output DataFrame.

Usage (Databricks notebook)
----------------------------
>>> from medrisk_features.boosting import predict_with_registered_model
>>>
>>> predictions = predict_with_registered_model(
...     model_uri="models:/workspace.schema.boosting_model@Champion",
...     input_df=new_spark_df,            # Spark or Pandas
... )
>>> display(predictions)
>>>
>>> # With Unity Catalog table as source and sink
>>> predict_with_registered_model(
...     model_uri="models:/workspace.schema.boosting_model@Champion",
...     input_table="workspace.schema.scoring_data",
...     output_table="workspace.schema.scoring_results",
...     mode="overwrite",
... )
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def predict_with_registered_model(
    model_uri: str,
    input_df: Optional[pd.DataFrame] = None,
    input_table: Optional[str] = None,
    output_table: Optional[str] = None,
    mode: str = "overwrite",
    spark=None,
) -> pd.DataFrame:
    """
    Load a registered BoostingPyFuncModel and run batch inference.

    One of input_df or input_table must be provided.

    Parameters
    ----------
    model_uri : str
        MLflow model URI. Supports all formats:
        - 'runs:/<run_id>/boosting_model'
        - 'models:/workspace.schema.model/Production'
        - 'models:/workspace.schema.model@Champion'
    input_df : pd.DataFrame or pyspark.sql.DataFrame or None
        Raw input data. Automatically converted from Spark if needed.
    input_table : str or None
        Unity Catalog table to load as input: 'catalog.schema.table'.
        Used when input_df is not provided.
    output_table : str or None
        Unity Catalog table to write output to. If None, returns DataFrame only.
    mode : str
        Spark write mode for output_table: 'overwrite' or 'append'.
    spark : SparkSession or None
        Active SparkSession for Spark operations. Auto-detected if None.

    Returns
    -------
    pd.DataFrame
        Output DataFrame with columns:
        <id_columns>, probability, priority, prediction_date, model_version.

    Raises
    ------
    ValueError
        If neither input_df nor input_table is provided.
    ImportError
        If mlflow is not installed.
    """
    try:
        import mlflow.pyfunc
    except ImportError as exc:
        raise ImportError(
            "mlflow is required for inference. "
            "Install with: pip install medrisk-features[mlflow]"
        ) from exc

    from medrisk_features.boosting.utils.spark_utils import (
        read_from_delta,
        to_pandas,
        write_to_delta,
    )

    # ── 1. Load input data ────────────────────────────────────────────────────
    if input_df is None and input_table is None:
        raise ValueError(
            "Provide either input_df (a DataFrame) or input_table (a Unity Catalog table name)."
        )

    if input_df is not None:
        df_pandas = to_pandas(input_df)
    else:
        if input_table is None:
            raise ValueError("input_table must be provided when input_df is None.")
        df_pandas = read_from_delta(
            table_name=input_table,
            spark=spark,
            convert_to_pandas=True,
        )

    # ── 2. Load the registered model ─────────────────────────────────────────
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # ── 3. Run inference ──────────────────────────────────────────────────────
    predictions = loaded_model.predict(df_pandas)

    # ── 4. Optionally write results to Unity Catalog ──────────────────────────
    if output_table is not None:
        write_to_delta(predictions, table_name=output_table, mode=mode, spark=spark)

    return predictions

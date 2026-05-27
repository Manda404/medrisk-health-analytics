"""
Holdout evaluation helpers for Databricks MLOps workflows.

The trainer can work only on a train table and leave the real test table
untouched. This module scores that independent holdout table and computes
binary classification metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class HoldoutEvaluationResult:
    """Result returned after scoring and evaluating a holdout dataset."""

    metrics: Dict[str, float]
    predictions: pd.DataFrame
    row_count: int
    target_column: str
    model_uri: Optional[str] = None
    output_table: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_binary_holdout_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
    prefix: str = "holdout",
) -> Dict[str, float]:
    """Compute binary classification metrics from holdout probabilities."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (pd.Series(y_score) >= threshold).astype(int)
    metrics: Dict[str, float] = {
        f"{prefix}_accuracy": round(accuracy_score(y_true, y_pred), 6),
        f"{prefix}_precision": round(precision_score(y_true, y_pred, zero_division=0), 6),
        f"{prefix}_recall": round(recall_score(y_true, y_pred, zero_division=0), 6),
        f"{prefix}_f1": round(f1_score(y_true, y_pred, zero_division=0), 6),
        f"{prefix}_roc_auc": round(roc_auc_score(y_true, y_score), 6),
        f"{prefix}_avg_precision": round(average_precision_score(y_true, y_score), 6),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics[f"{prefix}_tn"] = int(tn)
    metrics[f"{prefix}_fp"] = int(fp)
    metrics[f"{prefix}_fn"] = int(fn)
    metrics[f"{prefix}_tp"] = int(tp)
    return metrics


def evaluate_registered_model_on_holdout(
    model_uri: str,
    target_column: str,
    input_df: Optional[pd.DataFrame] = None,
    input_table: Optional[str] = None,
    output_table: Optional[str] = None,
    threshold: float = 0.5,
    mode: str = "overwrite",
    spark=None,
) -> HoldoutEvaluationResult:
    """
    Score a holdout dataset with a registered MLflow model and compute metrics.

    Provide either `input_df` or `input_table`. When `output_table` is set, the
    predictions plus target column are written as a Delta table.
    """
    try:
        import mlflow.pyfunc
    except ImportError as exc:
        raise ImportError(
            "mlflow is required for holdout evaluation. "
            "Install with: pip install medrisk-features[mlflow]"
        ) from exc

    from medrisk_features.boosting.utils.spark_utils import (
        read_from_delta,
        to_pandas,
        write_to_delta,
    )

    if input_df is None and input_table is None:
        raise ValueError("Provide either input_df or input_table.")

    if input_df is not None:
        df = to_pandas(input_df)
    else:
        if input_table is None:
            raise ValueError("input_table must be provided when input_df is None.")
        df = read_from_delta(input_table, spark=spark, convert_to_pandas=True)

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in holdout dataset.")

    y_true = df[target_column]
    X = df.drop(columns=[target_column])

    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(X)
    if "probability" not in predictions.columns:
        raise ValueError("Expected model output with a 'probability' column.")

    metrics = compute_binary_holdout_metrics(
        y_true=y_true,
        y_score=predictions["probability"],
        threshold=threshold,
        prefix="holdout",
    )

    predictions_with_target = pd.concat(
        [
            predictions.reset_index(drop=True),
            y_true.rename(target_column).reset_index(drop=True),
        ],
        axis=1,
    )

    if output_table:
        write_to_delta(predictions_with_target, table_name=output_table, mode=mode, spark=spark)

    return HoldoutEvaluationResult(
        metrics=metrics,
        predictions=predictions_with_target,
        row_count=len(predictions_with_target),
        target_column=target_column,
        model_uri=model_uri,
        output_table=output_table,
    )

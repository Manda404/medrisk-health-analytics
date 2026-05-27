"""
Model evaluation utilities for the medrisk boosting pipeline.

Computes all relevant metrics for classification and regression tasks
and returns a flat dict suitable for mlflow.log_metrics().
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    prefix: str = "train",
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Parameters
    ----------
    y_true  : Ground truth labels.
    y_pred  : Hard predictions.
    y_proba : Predicted probabilities of the positive class.
    prefix  : Metric name prefix: 'train', 'valid', or 'test'.

    Returns
    -------
    dict  Keys prefixed with `prefix_`, e.g. 'train_auc', 'train_f1'.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: Dict[str, float] = {
        f"{prefix}_accuracy":          round(accuracy_score(y_true, y_pred), 6),
        f"{prefix}_precision":         round(precision_score(y_true, y_pred, zero_division=0), 6),
        f"{prefix}_recall":            round(recall_score(y_true, y_pred, zero_division=0), 6),
        f"{prefix}_f1":                round(f1_score(y_true, y_pred, zero_division=0), 6),
        f"{prefix}_roc_auc":           round(roc_auc_score(y_true, y_proba), 6),
        f"{prefix}_avg_precision":     round(average_precision_score(y_true, y_proba), 6),
    }

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics[f"{prefix}_tn"] = int(tn)
    metrics[f"{prefix}_fp"] = int(fp)
    metrics[f"{prefix}_fn"] = int(fn)
    metrics[f"{prefix}_tp"] = int(tp)

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "train",
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Parameters
    ----------
    y_true  : Ground truth values.
    y_pred  : Predicted values.
    prefix  : Metric name prefix.

    Returns
    -------
    dict  Keys: 'mae', 'rmse', 'r2', 'mape'.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    metrics: Dict[str, float] = {
        f"{prefix}_mae":  round(mae, 6),
        f"{prefix}_rmse": round(rmse, 6),
        f"{prefix}_r2":   round(r2, 6),
    }

    # MAPE — guard against zero denominators
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        metrics[f"{prefix}_mape"] = round(float(mape), 6)

    return metrics


def evaluate_splits(
    model,
    splits: dict,
    task_type: str = "binary_classification",
) -> Dict[str, float]:
    """
    Evaluate a model on all available data splits (train, valid, test).

    Parameters
    ----------
    model     : Fitted BaseBoostingModel instance.
    splits    : Dict with keys 'train', 'valid', 'test' (optional).
                Each value is a dict with 'X' and 'y' keys.
    task_type : 'binary_classification' or 'regression'.

    Returns
    -------
    dict  All metrics from all splits, flat and MLflow-ready.
    """
    all_metrics: Dict[str, float] = {}

    for split_name, data in splits.items():
        X, y = data["X"], data["y"]
        y_pred  = model.predict(X)
        y_proba = model.predict_proba(X)

        if task_type == "regression":
            split_metrics = compute_regression_metrics(
                np.array(y), np.array(y_pred), prefix=split_name
            )
        else:
            split_metrics = compute_classification_metrics(
                np.array(y), np.array(y_pred), np.array(y_proba), prefix=split_name
            )

        all_metrics.update(split_metrics)

    return all_metrics

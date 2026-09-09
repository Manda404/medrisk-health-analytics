"""
Model evaluation utilities for the medrisk boosting pipeline.

Computes classification metrics and returns a flat dict suitable for
mlflow.log_metrics().
"""

from __future__ import annotations

from typing import Dict

import numpy as np


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
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: Dict[str, float] = {
        f"{prefix}_accuracy": round(accuracy_score(y_true, y_pred), 6),
        f"{prefix}_balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 6),
        f"{prefix}_precision": round(precision_score(y_true, y_pred, zero_division=0), 6),
        f"{prefix}_recall": round(recall_score(y_true, y_pred, zero_division=0), 6),
        f"{prefix}_f1": round(f1_score(y_true, y_pred, zero_division=0), 6),
        f"{prefix}_roc_auc": round(roc_auc_score(y_true, y_proba), 6),
        f"{prefix}_avg_precision": round(average_precision_score(y_true, y_proba), 6),
        f"{prefix}_mcc": round(matthews_corrcoef(y_true, y_pred), 6),
        f"{prefix}_brier_score": round(brier_score_loss(y_true, y_proba), 6),
    }

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics[f"{prefix}_specificity"] = round(tn / (tn + fp), 6) if tn + fp else 0.0
    metrics[f"{prefix}_npv"] = round(tn / (tn + fn), 6) if tn + fn else 0.0
    metrics[f"{prefix}_tn"] = int(tn)
    metrics[f"{prefix}_fp"] = int(fp)
    metrics[f"{prefix}_fn"] = int(fn)
    metrics[f"{prefix}_tp"] = int(tp)

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
    task_type : 'binary_classification' or 'multiclass_classification'.

    Returns
    -------
    dict  All metrics from all splits, flat and MLflow-ready.
    """
    all_metrics: Dict[str, float] = {}

    for split_name, data in splits.items():
        X, y = data["X"], data["y"]
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        split_metrics = compute_classification_metrics(
            np.array(y), np.array(y_pred), np.array(y_proba), prefix=split_name
        )

        all_metrics.update(split_metrics)

    return all_metrics

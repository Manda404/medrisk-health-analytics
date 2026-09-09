"""Model evaluation with metrics, diagnostics, and quality gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationReport:
    """Structured result produced by :class:`ModelEvaluator`."""

    model_name: str
    metrics: Mapping[str, float]
    confusion_matrix: pd.DataFrame
    classification_report: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.Series | None
    quality_gates: Mapping[str, Mapping[str, float | bool]]
    passed: bool
    metadata: Mapping[str, Any]


class ModelEvaluator:
    """Evaluate a fitted binary or multiclass classifier on test data."""

    def __init__(
        self,
        model: Any,
        *,
        model_name: str | None = None,
        threshold: float = 0.5,
        quality_gates: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.model = model
        self.model_name = model_name or model.__class__.__name__
        self.threshold = threshold
        self.quality_gates = dict(quality_gates or {})
        self.metadata = dict(metadata or {})

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> EvaluationReport:
        """Compute predictions, metrics, diagnostics, and gate status."""
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            balanced_accuracy_score,
            brier_score_loss,
            classification_report,
            confusion_matrix,
            f1_score,
            log_loss,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        self._validate_inputs(X_test, y_test)
        y_true = np.asarray(y_test)
        classes = np.unique(y_true)
        raw_probabilities = self._predict_probabilities(X_test)
        y_pred, score, probability_matrix = self._resolve_predictions(
            X_test, classes, raw_probabilities
        )
        binary = len(classes) == 2
        average = "binary" if binary else "weighted"
        metric_options = {"average": average, "zero_division": 0}
        if binary:
            metric_options["pos_label"] = classes[-1]

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, **metric_options)),
            "recall": float(recall_score(y_true, y_pred, **metric_options)),
            "f1": float(f1_score(y_true, y_pred, **metric_options)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }
        if score is not None:
            if binary:
                metrics["roc_auc"] = float(roc_auc_score(y_true, score))
                metrics["average_precision"] = float(average_precision_score(y_true, score))
                metrics["brier_score"] = float(brier_score_loss(y_true, score))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, probability_matrix, multi_class="ovr", average="weighted")
                )
            metrics["log_loss"] = float(log_loss(y_true, probability_matrix, labels=classes))

        matrix_values = confusion_matrix(y_true, y_pred, labels=classes)
        if binary:
            true_negative, false_positive, false_negative, true_positive = matrix_values.ravel()
            denominator = true_negative + false_positive
            metrics["specificity"] = float(true_negative / denominator) if denominator else 0.0
            npv_denominator = true_negative + false_negative
            metrics["negative_predictive_value"] = (
                float(true_negative / npv_denominator) if npv_denominator else 0.0
            )
            metrics["true_positive"] = float(true_positive)
            metrics["false_positive"] = float(false_positive)
            metrics["false_negative"] = float(false_negative)
            metrics["true_negative"] = float(true_negative)

        matrix = pd.DataFrame(
            matrix_values,
            index=[f"actual_{label}" for label in classes],
            columns=[f"predicted_{label}" for label in classes],
        )
        report = pd.DataFrame(
            classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        ).transpose()
        predictions = pd.DataFrame({"actual": y_true, "predicted": y_pred})
        if binary and score is not None:
            predictions["probability"] = score
        elif probability_matrix is not None:
            for index, label in enumerate(classes):
                predictions[f"probability_{label}"] = probability_matrix[:, index]

        gates = self._evaluate_quality_gates(metrics)
        return EvaluationReport(
            model_name=self.model_name,
            metrics=metrics,
            confusion_matrix=matrix,
            classification_report=report,
            predictions=predictions,
            feature_importance=self._get_feature_importance(),
            quality_gates=gates,
            passed=all(bool(gate["passed"]) for gate in gates.values()),
            metadata=self.metadata,
        )

    def _predict_probabilities(self, X_test: pd.DataFrame) -> np.ndarray | None:
        if not hasattr(self.model, "predict_proba"):
            return None
        return np.asarray(self.model.predict_proba(X_test))

    def _resolve_predictions(
        self,
        X_test: pd.DataFrame,
        classes: np.ndarray,
        probabilities: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        if probabilities is None:
            return np.asarray(self.model.predict(X_test)), None, None
        if probabilities.ndim == 1:
            if len(classes) != 2:
                raise ValueError("One-dimensional probabilities require a binary target")
            predictions = np.where(probabilities >= self.threshold, classes[-1], classes[0])
            return predictions, probabilities, np.column_stack([1 - probabilities, probabilities])
        if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
            raise ValueError("predict_proba output does not match the target classes")
        if len(classes) == 2:
            score = probabilities[:, 1]
            predictions = np.where(score >= self.threshold, classes[-1], classes[0])
            return predictions, score, probabilities
        return classes[np.argmax(probabilities, axis=1)], probabilities, probabilities

    def _get_feature_importance(self) -> pd.Series | None:
        try:
            if hasattr(self.model, "get_feature_importance"):
                return self.model.get_feature_importance()
        except RuntimeError:
            return None
        return None

    def _evaluate_quality_gates(
        self, metrics: Mapping[str, float]
    ) -> dict[str, Mapping[str, float | bool]]:
        unknown = sorted(set(self.quality_gates) - set(metrics))
        if unknown:
            raise ValueError(f"Quality gates reference unavailable metrics: {unknown}")
        return {
            metric: {
                "actual": metrics[metric],
                "minimum": minimum,
                "passed": metrics[metric] >= minimum,
            }
            for metric, minimum in self.quality_gates.items()
        }

    @staticmethod
    def _validate_inputs(X_test: pd.DataFrame, y_test: pd.Series) -> None:
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_test must be a pandas DataFrame")
        if not isinstance(y_test, pd.Series):
            raise TypeError("y_test must be a pandas Series")
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must contain the same number of rows")
        if X_test.empty:
            raise ValueError("Test data must not be empty")
        if y_test.isna().any():
            raise ValueError("y_test must not contain missing values")
        if y_test.nunique() < 2:
            raise ValueError("y_test must contain at least two classes")


__all__ = ["EvaluationReport", "ModelEvaluator"]

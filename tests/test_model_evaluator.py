import pandas as pd
import pytest

from medrisk_health_analytics import LogisticRegressionModel, ModelEvaluator


@pytest.fixture
def fitted_model_and_test_data():
    X = pd.DataFrame(
        {
            "age": [22, 28, 35, 41, 55, 62, 70, 77],
            "bmi": [19, 22, 24, 27, 31, 33, 36, 39],
        }
    )
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    return LogisticRegressionModel().fit(X, y), X, y


def test_model_evaluator_returns_complete_report(fitted_model_and_test_data):
    model, X_test, y_test = fitted_model_and_test_data
    evaluator = ModelEvaluator(
        model,
        model_name="diabetes-logistic",
        quality_gates={"roc_auc": 0.5, "recall": 0.5},
        metadata={"dataset": "holdout-v1"},
    )

    report = evaluator.evaluate(X_test, y_test)

    assert report.model_name == "diabetes-logistic"
    assert report.passed is True
    assert {
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "specificity",
        "mcc",
        "brier_score",
        "negative_predictive_value",
    } <= set(report.metrics)
    assert report.confusion_matrix.shape == (2, 2)
    assert list(report.predictions.columns) == ["actual", "predicted", "probability"]
    assert report.feature_importance is not None
    assert report.metadata == {"dataset": "holdout-v1"}


def test_model_evaluator_reports_failed_quality_gate(fitted_model_and_test_data):
    model, X_test, y_test = fitted_model_and_test_data
    report = ModelEvaluator(model, quality_gates={"recall": 1.1}).evaluate(X_test, y_test)

    assert report.passed is False
    assert report.quality_gates["recall"]["passed"] is False


def test_model_evaluator_rejects_unknown_gate(fitted_model_and_test_data):
    model, X_test, y_test = fitted_model_and_test_data

    with pytest.raises(ValueError, match="unavailable metrics"):
        ModelEvaluator(model, quality_gates={"unknown": 0.5}).evaluate(X_test, y_test)

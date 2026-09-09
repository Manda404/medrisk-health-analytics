from medrisk_health_analytics.boosting.training.trainer import _select_mlflow_metrics


def test_mlflow_tracks_only_decision_metrics():
    metrics = {
        "train_roc_auc": 0.95,
        "train_accuracy": 0.90,
        "train_tp": 100,
        "valid_roc_auc": 0.91,
        "valid_avg_precision": 0.89,
        "valid_recall": 0.85,
        "valid_f1": 0.86,
        "valid_specificity": 0.84,
        "valid_mcc": 0.70,
        "valid_brier_score": 0.12,
        "valid_accuracy": 0.88,
        "valid_tn": 80,
    }

    assert _select_mlflow_metrics(metrics) == {
        "valid_roc_auc": 0.91,
        "valid_avg_precision": 0.89,
        "valid_recall": 0.85,
        "valid_specificity": 0.84,
        "valid_mcc": 0.70,
        "valid_brier_score": 0.12,
    }

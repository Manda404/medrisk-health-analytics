import pandas as pd
from medrisk_features.boosting import compute_binary_holdout_metrics


def test_compute_binary_holdout_metrics():
    y_true = pd.Series([0, 0, 1, 1])
    y_score = pd.Series([0.1, 0.4, 0.8, 0.9])

    metrics = compute_binary_holdout_metrics(y_true, y_score, threshold=0.5)

    assert metrics["holdout_accuracy"] == 1.0
    assert metrics["holdout_precision"] == 1.0
    assert metrics["holdout_recall"] == 1.0
    assert metrics["holdout_f1"] == 1.0
    assert metrics["holdout_tp"] == 2
    assert metrics["holdout_tn"] == 2

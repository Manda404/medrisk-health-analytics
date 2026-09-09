import pandas as pd
import pytest

from medrisk_health_analytics.boosting import TrainingConfig, train_boosting_model


def test_training_rejects_missing_target():
    frame = pd.DataFrame({"age": [30, 40]})

    with pytest.raises(ValueError, match="missing required columns"):
        train_boosting_model(frame, TrainingConfig(target_column="target"))


def test_training_rejects_empty_data():
    frame = pd.DataFrame(columns=["age", "target"])

    with pytest.raises(ValueError, match="at least one row"):
        train_boosting_model(frame, TrainingConfig(target_column="target"))


def test_training_rejects_duplicate_columns():
    frame = pd.DataFrame([[30, 31, 0], [40, 41, 1]], columns=["age", "age", "target"])

    with pytest.raises(ValueError, match="duplicate columns"):
        train_boosting_model(frame, TrainingConfig(target_column="target"))


def test_training_rejects_missing_target_values():
    frame = pd.DataFrame({"age": [30, 40], "target": [0, None]})

    with pytest.raises(ValueError, match="contains missing values"):
        train_boosting_model(frame, TrainingConfig(target_column="target"))


def test_training_rejects_single_class_target():
    frame = pd.DataFrame({"age": [30, 40], "target": [1, 1]})

    with pytest.raises(ValueError, match="at least two classes"):
        train_boosting_model(frame, TrainingConfig(target_column="target"))


def test_binary_training_rejects_multiclass_target():
    frame = pd.DataFrame({"age": [30, 40, 50], "target": [0, 1, 2]})

    with pytest.raises(ValueError, match="exactly two target classes"):
        train_boosting_model(frame, TrainingConfig(target_column="target"))


def test_training_rejects_data_without_features():
    frame = pd.DataFrame({"target": [0, 1]})

    with pytest.raises(ValueError, match="usable feature columns"):
        train_boosting_model(frame, TrainingConfig(target_column="target"))

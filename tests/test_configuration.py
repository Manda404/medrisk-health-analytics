import pytest

from medrisk_health_analytics.boosting import InferenceConfig, SplitStrategy, TrainingConfig
from medrisk_health_analytics.utils import InvalidConfigurationError


def test_training_config_normalizes_enum_values():
    config = TrainingConfig(split_strategy="random", task_type="binary_classification")

    assert config.split_strategy is SplitStrategy.RANDOM
    assert config.task_type.value == "binary_classification"


@pytest.mark.parametrize(
    ("kwargs", "parameter"),
    [
        ({"test_size": 0}, "test_size"),
        ({"validation_size": 1}, "validation_size"),
        ({"split_strategy": "temporal"}, "temporal_column"),
        ({"categorical_encoding": "hash"}, "categorical_encoding"),
        ({"priority_medium_threshold": 0.9, "priority_high_threshold": 0.8}, "threshold"),
    ],
)
def test_training_config_rejects_invalid_values(kwargs, parameter):
    with pytest.raises(InvalidConfigurationError, match=parameter):
        TrainingConfig(**kwargs)


def test_training_config_rejects_overlapping_column_types():
    with pytest.raises(InvalidConfigurationError, match="disjoint"):
        TrainingConfig(numeric_columns=["age"], categorical_columns=["age"])


def test_training_config_rejects_target_as_feature():
    with pytest.raises(InvalidConfigurationError, match="target_column"):
        TrainingConfig(target_column="target", feature_columns=["age", "target"])


def test_inference_config_requires_model_uri():
    with pytest.raises(InvalidConfigurationError, match="model_uri"):
        InferenceConfig()


def test_inference_config_rejects_invalid_batch_size():
    with pytest.raises(InvalidConfigurationError, match="batch_size"):
        InferenceConfig(model_uri="runs:/abc/model", batch_size=0)

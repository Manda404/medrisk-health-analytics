import pytest
from pydantic import ValidationError

from medrisk_health_analytics.settings import RuntimeSettings


def test_settings_build_qualified_names():
    config = RuntimeSettings(catalog="health", schema="risk", model_alias="Champion")

    assert config.train_table == "health.risk.patient_train_features"
    assert config.registered_model_uri == "models:/health.risk.boosting_risk_model@Champion"
    assert config.data_quality_table == "health.risk.data_quality_metrics"
    assert config.drift_metrics_table == "health.risk.feature_drift_metrics"


def test_settings_load_environment_overrides():
    config = RuntimeSettings.from_env(
        {
            "MEDRISK_ENVIRONMENT": "stg",
            "MEDRISK_CATALOG": "health_stg",
            "MEDRISK_SCHEMA": "risk",
            "MEDRISK_ID_COLUMNS": "patient_id, encounter_id",
            "MEDRISK_VALIDATE_SCHEMA": "false",
            "MEDRISK_TEST_SIZE": "0.25",
        }
    )

    assert config.environment == "stg"
    assert config.id_columns == ("patient_id", "encounter_id")
    assert config.validate_schema is False
    assert config.test_size == 0.25


def test_settings_reject_invalid_boolean():
    with pytest.raises(ValidationError, match="validate_schema"):
        RuntimeSettings.from_env({"MEDRISK_VALIDATE_SCHEMA": "sometimes"})


def test_log_file_has_expected_name():
    config = RuntimeSettings(log_directory="custom-logs")

    assert config.log_file.as_posix() == "custom-logs/Medrisk-Health-Analytics.log"

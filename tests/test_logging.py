from medrisk_health_analytics.logging import configure_file_logging, get_logger
from medrisk_health_analytics.settings import settings


def test_named_loggers_are_cached():
    assert get_logger("pipeline") is get_logger("pipeline")


def test_different_names_get_different_contexts():
    assert get_logger("training") is not get_logger("inference")


def test_file_logging_is_configured_once():
    assert configure_file_logging() == configure_file_logging()
    assert settings.log_file.exists()

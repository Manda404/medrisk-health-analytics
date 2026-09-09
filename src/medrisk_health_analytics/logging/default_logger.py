from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from loguru import logger

from medrisk_health_analytics.settings import settings

# Keep the application's existing Loguru handlers and add one shared file sink.
_LOGGER_CACHE: Dict[str, Any] = {}
_LOGGER_CACHE_LOCK = Lock()
_FILE_HANDLER_ID: Optional[int] = None


def configure_file_logging() -> int:
    """Configure the shared rotating application log file exactly once."""
    global _FILE_HANDLER_ID
    with _LOGGER_CACHE_LOCK:
        if _FILE_HANDLER_ID is None:
            log_file = settings.log_file
            if str(Path.cwd()).startswith("/Workspace/"):
                log_file = Path("/tmp/medrisk/logs") / settings.log_file.name
            log_file.parent.mkdir(parents=True, exist_ok=True)
            _FILE_HANDLER_ID = logger.add(
                log_file,
                level=settings.log_level,
                rotation=settings.log_rotation,
                retention=settings.log_retention,
                encoding="utf-8",
                enqueue=True,
                backtrace=False,
                diagnose=False,
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {extra[logger_name]} | {message}"
                ),
            )
        return _FILE_HANDLER_ID


def get_logger(name: str = "medrisk-health-analytics"):
    """
    Return a Loguru logger bound to the given contextual name.

    The package adds one rotating file handler without removing the consuming
    application's existing handlers.
    Subsequent calls for the same ``name`` return the cached bound logger,
    which is safe in multi-threaded environments (Databricks Spark).

    Parameters
    ----------
    name : str
        Logical name shown in the ``logger_name`` field of each log record.

    Returns
    -------
    BoundLogger
        Loguru logger bound with ``logger_name=name``.
    """
    configure_file_logging()
    with _LOGGER_CACHE_LOCK:
        if name not in _LOGGER_CACHE:
            _LOGGER_CACHE[name] = logger.bind(logger_name=name)
        return _LOGGER_CACHE[name]

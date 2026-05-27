import sys
from typing import Any, Dict

from loguru import logger

# ---------------------------------------------------------------------------
# Module-level setup — executed ONCE at import time.
# Loguru uses a global singleton: logger.remove() / logger.add() must only be
# called once to avoid removing handlers from other components sharing the same
# process (e.g., multiple pipeline steps on Databricks Spark).
# ---------------------------------------------------------------------------
logger.remove()  # Remove the default Loguru handler
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "<cyan>{extra[logger_name]}</cyan> | "
        "<level>{message}</level>"
    ),
)

# Cache — avoids creating multiple bound loggers for the same name and ensures
# thread-safe reuse in multi-threaded Databricks environments.
_LOGGER_CACHE: Dict[str, Any] = {}


def get_logger(name: str = "medrisk-features"):
    """
    Return a Loguru logger bound to the given contextual name.

    The underlying handler is configured exactly once at module import time.
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
    if name not in _LOGGER_CACHE:
        _LOGGER_CACHE[name] = logger.bind(logger_name=name)
    return _LOGGER_CACHE[name]

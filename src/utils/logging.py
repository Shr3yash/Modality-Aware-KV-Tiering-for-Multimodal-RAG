from __future__ import annotations

import logging
from typing import Any

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structured logging with structlog and stdlib logging."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
    )

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger with the given name."""
    if not structlog.is_configured():
        configure_logging()
    return structlog.get_logger(name)


def log_exception(logger: structlog.stdlib.BoundLogger, msg: str, **kwargs: Any) -> None:
    """Helper to log an exception with consistent structure."""
    logger.exception(msg, **kwargs)


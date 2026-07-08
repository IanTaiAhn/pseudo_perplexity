# Structured logging setup
import logging
import os
import sys

import structlog

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def configure_logging() -> None:
    """Configure structlog to emit one JSON object per log event.

    Call once at process startup (api/main.py). Safe to call more than once —
    structlog.configure is idempotent.
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=_LOG_LEVEL,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(_LOG_LEVEL)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def bind_query_context(**kwargs) -> None:
    """Bind fields (e.g. query_id) to every log call for the current async task.

    Uses structlog's contextvars support so nested calls (orchestrator,
    generator) automatically include query_id without threading it through
    every function signature.
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_query_context() -> None:
    structlog.contextvars.clear_contextvars()

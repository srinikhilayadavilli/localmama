"""Structured-ish console logging with an explicit state-transition helper."""

from __future__ import annotations

import logging
import sys

from .config import settings

_CONFIGURED = False


def setup_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.handlers[:] = [handler]
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def log_transition(
    logger: logging.Logger,
    session_id: str,
    old_state: str,
    new_state: str,
    detail: str = "",
) -> None:
    """Requirement 13: log every workflow state transition clearly."""
    arrow = f"{old_state} -> {new_state}"
    suffix = f"  [{detail}]" if detail else ""
    logger.info("session=%s  %s%s", session_id[:8], arrow, suffix)

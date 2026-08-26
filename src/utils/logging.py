"""Structured logging utilities shared across the framework.

Provides a single :func:`get_logger` factory so every module logs through a
consistently configured handler instead of using ``print()``. Uses ``rich``
for readable console output when available, falling back to the standard
library formatter otherwise.
"""

from __future__ import annotations

import logging
from typing import Optional

try:  # rich is a declared dependency, but keep logging usable without it.
    from rich.logging import RichHandler

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when rich is absent.
    _RICH_AVAILABLE = False


_DEFAULT_LEVEL = logging.INFO
_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a configured logger for ``name``.

    The logger is configured exactly once per name; repeated calls return the
    same instance without stacking duplicate handlers.

    Args:
        name: Logger name, conventionally the caller's ``__name__``.
        level: Optional logging level; defaults to ``INFO`` when omitted.

    Returns:
        A ready-to-use :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level if level is not None else _DEFAULT_LEVEL)
    logger.propagate = False

    if _RICH_AVAILABLE:
        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    else:  # pragma: no cover - standard-library fallback path.
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    logger.addHandler(handler)
    return logger

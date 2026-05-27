"""Structured logging helpers for the MAPFM medical AI ecosystem."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

try:
    from loguru import logger as _loguru_logger
except Exception:  # pragma: no cover - fallback for minimal environments
    _loguru_logger = None


class _FallbackLogger:
    """Minimal logger with a Loguru-like API when loguru is unavailable."""

    def __init__(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        self._logger = logging.getLogger("mapfm")

    def add(self, *_: Any, **__: Any) -> None:
        return None

    def remove(self, *_: Any, **__: Any) -> None:
        return None

    def bind(self, **_: Any) -> "_FallbackLogger":
        return self

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(message.format(*args, **kwargs))

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(message.format(*args, **kwargs))

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(message.format(*args, **kwargs))

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(message.format(*args, **kwargs))

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(message.format(*args, **kwargs))


logger = _loguru_logger if _loguru_logger is not None else _FallbackLogger()


def configure_logging(log_dir: str | Path = "logs", level: str = "INFO") -> None:
    """Configure structured console and file logging.

    Args:
        log_dir: Folder where log files are written.
        level: Runtime logging level.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    if _loguru_logger is None:
        return
    logger.remove()
    fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {module} | {message}"
    logger.add(sys.stderr, level=level.upper(), format=fmt, backtrace=True, diagnose=False)
    logger.add(log_path / "system.log", level=level.upper(), format=fmt, rotation="10 MB", retention="14 days", backtrace=True)
    logger.add(log_path / "error.log", level="ERROR", format=fmt, rotation="10 MB", retention="30 days", backtrace=True)

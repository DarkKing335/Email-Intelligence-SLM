"""Structured logging configuration using Loguru.

Configures formatted console logging and JSON-structured file logs.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

from loguru import logger


def serialize_log(record: dict[str, Any]) -> str:
    """Format log record as a structured JSON line."""
    exception = record["exception"]
    exception_data = None
    if exception is not None:
        exception_data = {
            "type": str(exception.type),
            "value": str(exception.value),
            "traceback": bool(exception.traceback),
        }

    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
        "exception": exception_data,
        "extra": record["extra"],
    }
    return json.dumps(log_entry, ensure_ascii=False) + "\n"



def configure_logging(
    level: str = "INFO",
    log_file: str | Path = "logs/data_engineering.log",
    rotation: str = "50 MB",
    retention: str = "7 days",
    json_format: bool = True,
) -> None:
    """Configure Loguru sinks for console output and file writing.

    Parameters
    ----------
    level:
        Minimum log level (DEBUG, INFO, WARNING, ERROR).
    log_file:
        Filename path where logs will be stored.
    rotation:
        Condition when log file is rotated.
    retention:
        Condition when old log files are cleaned up.
    json_format:
        If True, log files are structured JSON. Otherwise, standard format.
    """
    # Remove existing handlers to avoid duplicates
    logger.remove()

    # 1. Console handler - human readable colorized output to stderr
    console_format = (
        "<green>{blue_time}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level> {extra}"
    )
    
    # Custom formatter to inject formatted time
    def console_formatter(record: dict[str, Any]) -> str:
        record["extra"]["blue_time"] = record["time"].strftime("%Y-%m-%d %H:%M:%S")
        return console_format + "\n"

    logger.add(
        sys.stderr,
        format=console_formatter,
        level=level,
        colorize=True,
        enqueue=True,
    )

    # 2. File handler - writes to file with rotation/retention if log_file is provided
    if not log_file:
        logger.debug("No log file path provided; skipping file logging handler.")
        return

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if json_format:
        # Structured JSON logger
        logger.add(
            str(log_path),
            format=serialize_log,
            level=level,
            rotation=rotation,
            retention=retention,
            enqueue=True,
        )
    else:
        # Standard file logger
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} - {message} {extra}"
        )
        logger.add(
            str(log_path),
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            enqueue=True,
        )

    logger.debug(
        "Logging configured",
        level=level,
        log_file=str(log_path),
        json_format=json_format,
    )

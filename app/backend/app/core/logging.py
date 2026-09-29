"""Structured logging configuration."""

import json
import logging
import sys
import time
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Format logs as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "document_id"):
            log_entry["document_id"] = getattr(record, "document_id")
        if hasattr(record, "job_id"):
            log_entry["job_id"] = getattr(record, "job_id")
        if hasattr(record, "stage"):
            log_entry["stage"] = getattr(record, "stage")
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = getattr(record, "duration_ms")
        if hasattr(record, "error_code"):
            log_entry["error_code"] = getattr(record, "error_code")

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(debug: bool = False) -> logging.Logger:
    """Initialize structured application logger."""
    level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger("document_ai")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger


logger = setup_logging()

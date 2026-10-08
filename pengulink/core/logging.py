"""PenguLink structured logging subsystem.

Provides console formatting via Rich, headless daemon log formats,
contextual device logger adapters, and credential redaction filters.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from rich.console import Console
from rich.logging import RichHandler


# Regex matching PEM private key blocks to prevent accidental leakage in logs
PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA )?PRIVATE KEY-----"
)


class RedactingFilter(logging.Filter):
    """Logging filter that sanitizes private keys and sensitive tokens."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = PRIVATE_KEY_RE.sub("[REDACTED_PRIVATE_KEY]", record.msg)
        if record.args:
            sanitized_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    sanitized_args.append(PRIVATE_KEY_RE.sub("[REDACTED_PRIVATE_KEY]", arg))
                else:
                    sanitized_args.append(arg)
            record.args = tuple(sanitized_args)
        return True


class DeviceLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically prefixes log messages with device ID context."""

    def process(self, msg: Any, kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        device_id = self.extra.get("device_id", "unknown") if self.extra else "unknown"
        prefix = f"[{device_id[:8]}]"
        return f"{prefix} {msg}", kwargs


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rich_console: bool = True,
) -> logging.Logger:
    """Configure the root 'pengulink' logger.

    Args:
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_file: Optional path to write persistent log output.
        rich_console: Use rich.logging.RichHandler if stdout/stderr is a TTY.

    Returns:
        The configured top-level 'pengulink' logger instance.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("pengulink")
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    logger.handlers.clear()
    logger.propagate = False

    redaction_filter = RedactingFilter()

    # 1. Console Handler
    if rich_console and sys.stderr.isatty():
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=False,
        )
        handler.setLevel(numeric_level)
        handler.addFilter(redaction_filter)
        logger.addHandler(handler)
    else:
        stream_handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(numeric_level)
        stream_handler.addFilter(redaction_filter)
        logger.addHandler(stream_handler)

    # 2. File Handler (Optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(numeric_level)
        file_handler.addFilter(redaction_filter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Obtain a namespaced logger under the 'pengulink' hierarchy."""
    if not name:
        return logging.getLogger("pengulink")
    if name.startswith("pengulink.") or name == "pengulink":
        return logging.getLogger(name)
    return logging.getLogger(f"pengulink.{name}")


def get_device_logger(device_id: str, subname: Optional[str] = None) -> DeviceLoggerAdapter:
    """Create a contextual logger adapter bound to a specific device ID."""
    base_name = f"pengulink.device.{subname}" if subname else "pengulink.device"
    base_logger = logging.getLogger(base_name)
    return DeviceLoggerAdapter(base_logger, {"device_id": device_id})


__all__ = [
    "PRIVATE_KEY_RE",
    "RedactingFilter",
    "DeviceLoggerAdapter",
    "setup_logging",
    "get_logger",
    "get_device_logger",
]

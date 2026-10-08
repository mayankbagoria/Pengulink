"""Unit tests for PenguLink structured logging and redaction."""

import logging
from pathlib import Path

from pengulink.core.logging import (
    DeviceLoggerAdapter,
    RedactingFilter,
    get_device_logger,
    get_logger,
    setup_logging,
)


def test_redaction_filter() -> None:
    """Assert RedactingFilter replaces private keys with [REDACTED_PRIVATE_KEY]."""
    rf = RedactingFilter()
    sample_key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA0Y3y...\n"
        "-----END RSA PRIVATE KEY-----"
    )

    # Test redaction in msg
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=f"Loading secret key: {sample_key}",
        args=(),
        exc_info=None,
    )
    rf.filter(record)
    assert "[REDACTED_PRIVATE_KEY]" in record.msg
    assert "MIIEowIBAAKCAQEA0Y3y" not in record.msg

    # Test redaction in args
    record_args = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Loading secret key: %s",
        args=(sample_key,),
        exc_info=None,
    )
    rf.filter(record_args)
    assert record_args.args is not None
    assert "[REDACTED_PRIVATE_KEY]" in record_args.args[0]
    assert "MIIEowIBAAKCAQEA0Y3y" not in record_args.args[0]


def test_device_logger_adapter() -> None:
    """Assert DeviceLoggerAdapter adds device ID prefix to log messages."""
    base_logger = logging.getLogger("test.device")
    adapter = DeviceLoggerAdapter(base_logger, {"device_id": "0123456789abcdef"})
    msg, kwargs = adapter.process("Connection established", {})
    assert msg == "[01234567] Connection established"


def test_get_logger_prefixing() -> None:
    """Assert get_logger automatically prefixes 'pengulink.'."""
    l1 = get_logger("core.config")
    assert l1.name == "pengulink.core.config"

    l2 = get_logger("pengulink.network.tcp")
    assert l2.name == "pengulink.network.tcp"

    l3 = get_logger()
    assert l3.name == "pengulink"


def test_setup_logging_file(tmp_path: Path) -> None:
    """Assert setup_logging writes output to a designated file with redaction."""
    log_file = tmp_path / "logs" / "daemon.log"
    logger = setup_logging(level="DEBUG", log_file=log_file, rich_console=False)

    secret = "-----BEGIN PRIVATE KEY-----\nMIGfMA0GCSqGSIb3DQEBAQUAA4GNAD...\n-----END PRIVATE KEY-----"
    logger.info("Initializing node with secret: %s", secret)

    # Flush handlers
    for h in logger.handlers:
        h.flush()

    assert log_file.is_file()
    content = log_file.read_text(encoding="utf-8")
    assert "Initializing node with secret: [REDACTED_PRIVATE_KEY]" in content
    assert "MIGfMA0GCSqGSIb3" not in content

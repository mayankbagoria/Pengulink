"""tests.adversarial.test_clipboard_privacy_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial privacy challenge test harness verifying strict zero-logging
guarantees for the PenguLink clipboard synchronization ecosystem.

Verifies:
1. Multi-canary token tracking across inbound packet, outbound watcher, IPC set, and CLI set.
2. In-memory LogRingBuffer audit ensuring zero cleartext leakage.
3. IPC error reporting and exception logging sanitization.
4. CLI set confirmation message output privacy.
5. ClipboardUpdatedEvent string and repr representation redaction.
6. Clean daemon lifecycle and zero dangling asyncio tasks.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.config import PenguLinkConfig
from pengulink.core.events import ClipboardUpdatedEvent
from pengulink.daemon.ipc import PenguLinkIPCClient
from pengulink.daemon.service import LogRingBuffer, PenguLinkDaemon, safe_redact_clipboard
from pengulink.network.packet import Packet
from pengulink.plugins.clipboard import InMemoryClipboardBackend, redact_clipboard

# =============================================================================
# Canary Secret Tokens
# =============================================================================

CANARY_INBOUND = "CANARY_TOKEN_INBOUND_9876543210_SECRET"
CANARY_OUTBOUND = "CANARY_TOKEN_OUTBOUND_1234567890_SECRET"
CANARY_IPC = "CANARY_TOKEN_IPC_SET_5555555555_SECRET"
CANARY_CLI = "CANARY_TOKEN_CLI_SET_7777777777_SECRET"
CANARY_OVERSIZED = "CANARY_TOKEN_OVERSIZED_8888888888_SECRET"
CANARY_SURROGATE = "CANARY_TOKEN_SURROGATE_\ud800_SECRET"


@pytest.fixture
def privacy_daemon_config(tmp_path: Path) -> PenguLinkConfig:
    """Construct isolated daemon config for privacy tests."""
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / "config"
    cfg.storage.data_dir = tmp_path / "data"
    cfg.storage.runtime_dir = tmp_path / "run"
    cfg.device.device_id = "privacy_daemon_device_id_12345678"
    cfg.device.device_name = "PenguLink-Privacy-Test"
    cfg.network.udp_port = 17180
    cfg.network.tcp_port_min = 17230
    cfg.network.tcp_port_max = 17235
    cfg.ensure_directories()
    return cfg


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing Click CLI runner."""
    return CliRunner(env={"COLUMNS": "160"})


# =============================================================================
# Adversarial Privacy Test Suite
# =============================================================================

async def test_adv_privacy_inbound_packet_zero_log_leak(
    privacy_daemon_config: PenguLinkConfig,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify inbound kdeconnect.clipboard packet does NOT leak canary cleartext into daemon logs or LogRingBuffer."""
    backend = InMemoryClipboardBackend()
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(
        config=privacy_daemon_config,
        socket_path=sock_path,
        clipboard_backend=backend,
    )

    caplog.set_level(logging.DEBUG)
    await daemon.start()
    try:
        inbound_pkt = Packet(
            type="kdeconnect.clipboard",
            body={"content": CANARY_INBOUND, "timestamp": 1728000000000},
        )

        send_cb = lambda pkt: None
        await daemon.plugin_registry.dispatch_packet(inbound_pkt, "remote_peer_1", send_cb)
        await asyncio.sleep(0.05)

        # 1. Assert cleartext canary is NEVER in any log record emitted
        for record in caplog.records:
            assert CANARY_INBOUND not in record.getMessage(), (
                f"Canary leaked in logger '{record.name}': {record.getMessage()}"
            )

        # 2. Assert cleartext canary is NEVER in daemon._log_buffer
        for entry in daemon._log_buffer.get_logs(limit=1000):
            assert CANARY_INBOUND not in entry.get("message", ""), (
                f"Canary leaked in LogRingBuffer: {entry.get('message')}"
            )

        # 3. Assert redacted digest prefix appears in records instead
        digest_prefix = hashlib.sha256(CANARY_INBOUND.encode("utf-8")).hexdigest()[:8]
        found_redacted = any(
            digest_prefix in r.getMessage() for r in caplog.records
        )
        assert found_redacted, "Expected redacted sha256 prefix in log output"

    finally:
        await daemon.stop()


async def test_adv_privacy_outbound_watcher_zero_log_leak(
    privacy_daemon_config: PenguLinkConfig,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify local OS clipboard change does NOT leak canary cleartext into daemon logs or LogRingBuffer."""
    backend = InMemoryClipboardBackend()
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(
        config=privacy_daemon_config,
        socket_path=sock_path,
        clipboard_backend=backend,
    )

    caplog.set_level(logging.DEBUG)
    await daemon.start()
    try:
        # Trigger local clipboard change
        await backend.set_text(CANARY_OUTBOUND)
        await asyncio.sleep(0.05)

        # Audit logs and buffer
        for record in caplog.records:
            assert CANARY_OUTBOUND not in record.getMessage(), (
                f"Canary leaked in logger '{record.name}': {record.getMessage()}"
            )

        for entry in daemon._log_buffer.get_logs(limit=1000):
            assert CANARY_OUTBOUND not in entry.get("message", ""), (
                f"Canary leaked in LogRingBuffer: {entry.get('message')}"
            )

    finally:
        await daemon.stop()


async def test_adv_privacy_ipc_set_zero_log_leak(
    privacy_daemon_config: PenguLinkConfig,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify set_clipboard over IPC does NOT leak canary cleartext into daemon logs or LogRingBuffer."""
    backend = InMemoryClipboardBackend()
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(
        config=privacy_daemon_config,
        socket_path=sock_path,
        clipboard_backend=backend,
    )

    caplog.set_level(logging.DEBUG)
    await daemon.start()
    try:
        client = PenguLinkIPCClient(socket_path=sock_path)
        res = await client.set_clipboard(CANARY_IPC)
        assert res.get("status") == "ok"
        await asyncio.sleep(0.05)

        # Audit all log records
        for record in caplog.records:
            assert CANARY_IPC not in record.getMessage(), (
                f"Canary leaked in logger '{record.name}': {record.getMessage()}"
            )

        for entry in daemon._log_buffer.get_logs(limit=1000):
            assert CANARY_IPC not in entry.get("message", ""), (
                f"Canary leaked in LogRingBuffer: {entry.get('message')}"
            )

    finally:
        await daemon.stop()


def test_adv_privacy_cli_set_zero_log_leak(
    cli_runner: CliRunner,
    privacy_daemon_config: PenguLinkConfig,
    tmp_path: Path,
) -> None:
    """Verify CLI clipboard set command does NOT echo canary cleartext in confirmation badge."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    with patch(
        "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
        return_value={"status": "ok", "bytes": len(CANARY_CLI.encode("utf-8"))},
    ):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "clipboard", "set", CANARY_CLI])
        assert result.exit_code == 0
        # CRITICAL PRIVACY ASSERTION: The sensitive string MUST NOT appear in the user confirmation
        assert CANARY_CLI not in result.output
        assert "Clipboard updated" in result.output
        assert f"{len(CANARY_CLI.encode('utf-8'))} bytes" in result.output


async def test_adv_privacy_ipc_error_zero_log_leak(
    privacy_daemon_config: PenguLinkConfig,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify oversized payload containing canary token is rejected without echoing cleartext into logs or error strings."""
    backend = InMemoryClipboardBackend()
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(
        config=privacy_daemon_config,
        socket_path=sock_path,
        clipboard_backend=backend,
    )

    caplog.set_level(logging.DEBUG)
    await daemon.start()
    try:
        # Create oversized payload (>120 KB) embedded with CANARY_OVERSIZED
        oversized_payload = f"{CANARY_OVERSIZED}_{'Z' * 125_000}"
        client = PenguLinkIPCClient(socket_path=sock_path)

        with pytest.raises(Exception) as exc_info:
            await client.set_clipboard(oversized_payload)

        # 1. Error message to client MUST NOT contain the canary text
        assert CANARY_OVERSIZED not in str(exc_info.value)

        # 2. Daemon logs and LogRingBuffer MUST NOT contain the canary text
        for record in caplog.records:
            assert CANARY_OVERSIZED not in record.getMessage(), (
                f"Oversized canary leaked in logger '{record.name}': {record.getMessage()}"
            )

        for entry in daemon._log_buffer.get_logs(limit=1000):
            assert CANARY_OVERSIZED not in entry.get("message", ""), (
                f"Oversized canary leaked in LogRingBuffer: {entry.get('message')}"
            )

    finally:
        await daemon.stop()


async def test_adv_privacy_log_ring_buffer_audit(
    privacy_daemon_config: PenguLinkConfig,
    tmp_path: Path,
) -> None:
    """Execute burst of 30 diverse clipboard updates and audit every entry in LogRingBuffer for zero plaintext."""
    backend = InMemoryClipboardBackend()
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(
        config=privacy_daemon_config,
        socket_path=sock_path,
        clipboard_backend=backend,
    )

    await daemon.start()
    canaries = [f"CANARY_SECRET_BURST_TOKEN_{idx:04d}_PASS" for idx in range(30)]
    try:
        for idx, token in enumerate(canaries):
            if idx % 2 == 0:
                await daemon.set_clipboard(token)
            else:
                pkt = Packet(
                    type="kdeconnect.clipboard",
                    body={"content": token, "timestamp": 1728000000000 + idx},
                )
                await daemon.plugin_registry.dispatch_packet(pkt, f"device_{idx}", lambda p: None)

        await asyncio.sleep(0.05)

        # Audit entire buffer
        buffer_entries = daemon._log_buffer.get_logs(limit=1000)
        assert len(buffer_entries) > 0

        for entry in buffer_entries:
            msg = entry.get("message", "")
            for token in canaries:
                assert token not in msg, f"Leak detected in ring buffer entry: {msg}"

    finally:
        await daemon.stop()


def test_adv_privacy_clipboard_event_repr_and_str() -> None:
    """Verify ClipboardUpdatedEvent strictly redacts content in __repr__ and __str__."""
    secret = "ULTRA_SENSITIVE_PASSWORD_12345"
    event = ClipboardUpdatedEvent(content=secret, device_id="device_xyz", timestamp=99999)

    repr_str = repr(event)
    str_str = str(event)

    assert secret not in repr_str
    assert secret not in str_str
    assert "[REDACTED" in repr_str
    assert "[REDACTED" in str_str


def test_adv_privacy_safe_redact_clipboard_surrogate_infallibility() -> None:
    """Verify safe_redact_clipboard handles lone surrogates, None, and arbitrary objects without throwing."""
    assert safe_redact_clipboard(None) == "[REDACTED, None]"
    assert "[REDACTED" in safe_redact_clipboard(CANARY_SURROGATE)
    assert "[REDACTED" in safe_redact_clipboard(123456)
    assert "[REDACTED" in safe_redact_clipboard(["list", "of", "items"])


async def test_adv_privacy_daemon_start_stop_zero_dangling_tasks(
    privacy_daemon_config: PenguLinkConfig,
    tmp_path: Path,
) -> None:
    """Verify daemon start and stop cycle leaves zero dangling background tasks."""
    backend = InMemoryClipboardBackend()
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(
        config=privacy_daemon_config,
        socket_path=sock_path,
        clipboard_backend=backend,
    )

    await daemon.start()
    await daemon.set_clipboard("Lifecycle Task Test")
    await asyncio.sleep(0.02)
    await daemon.stop()

    # Verify background tasks set is cleared
    assert len(daemon._background_tasks) == 0
    assert not daemon.clipboard_plugin.is_running

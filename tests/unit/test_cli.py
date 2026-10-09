"""tests.unit.test_cli
~~~~~~~~~~~~~~~~~~~
Unit tests for PenguLink interactive debug CLI (pengulink/cli/main.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing Click CLI test runner."""
    return CliRunner(env={"COLUMNS": "160"})


def test_cli_version(cli_runner: CliRunner) -> None:
    """Verify --version returns version string and exits cleanly."""
    result = cli_runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_daemon_not_running_error(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify graceful error when background daemon is not running."""
    nonexistent_socket = tmp_path / "nonexistent.sock"
    result = cli_runner.invoke(cli, ["-s", str(nonexistent_socket), "devices"])
    assert result.exit_code == 1
    assert (
        "PenguLink daemon is not running. Start with 'pengulink daemon run' or start the background service."
        in result.output
    )


def test_cli_discover_command(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify discover command displays discovered peers in Rich table."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    sample_peers = [
        {
            "device_id": "pixel_8_device_id_12345678",
            "device_name": "Pixel 8 Pro",
            "device_type": "phone",
            "sender_ip": "192.168.1.105",
            "tcp_port": 1716,
            "is_paired": False,
        }
    ]

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_peers):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "discover"])
        assert result.exit_code == 0
        assert "Discovered KDE Connect Peers" in result.output
        assert "Pixel 8 Pro" in result.output
        assert "192.168.1.105" in result.output
        assert "phone" in result.output


def test_cli_devices_command(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify devices command displays known devices with state badges and battery formatting."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    sample_devices = [
        {
            "device_id": "galaxy_s24_device_id_abcdef",
            "device_name": "Galaxy S24",
            "device_type": "phone",
            "state": "CONNECTED",
            "is_paired": True,
            "battery": {
                "current_charge": 88,
                "is_charging": True,
                "threshold_event": 0,
            },
        },
        {
            "device_id": "tablet_device_id_999999",
            "device_name": "Pixel Tablet",
            "device_type": "tablet",
            "state": "DISCONNECTED",
            "is_paired": False,
            "battery": {
                "current_charge": 12,
                "is_charging": False,
                "threshold_event": 1,
            },
        },
    ]

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_devices):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "devices"])
        assert result.exit_code == 0
        assert "Known & Paired Devices" in result.output
        assert "Galaxy S24" in result.output
        assert "CONNECTED" in result.output
        assert "88% ⚡ (Charging)" in result.output
        assert "12% 🪫 (Low)" in result.output


def test_cli_pair_command_renders_panel(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify pair command renders 8-character verification code in Rich panel."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    pair_result = {
        "device_id": "phone_123",
        "device_name": "Pixel 8",
        "verification_code": "8F3A1B9C",
        "status": "paired",
    }

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=pair_result):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "pair", "phone_123"])
        assert result.exit_code == 0
        assert "Pairing Verification" in result.output
        assert "8F3A1B9C" in result.output
        assert "Pairing successful" in result.output


def test_cli_notifications_otp_highlight(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify notifications command renders OTP highlighting and supports --otp-only."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    sample_notifs = [
        {
            "notification_id": "notif_otp_1",
            "device_id": "phone_12345678",
            "app_name": "Google Messages",
            "title": "Bank Security",
            "text": "Your verification code is 492810.",
            "auth_code": "492810",
            "request_reply_id": None,
            "time_str": "12:00",
        },
        {
            "notification_id": "notif_regular_2",
            "device_id": "phone_12345678",
            "app_name": "News App",
            "title": "Breaking News",
            "text": "Weather forecast for today.",
            "auth_code": None,
            "request_reply_id": None,
            "time_str": "12:05",
        },
    ]

    # Test all notifications
    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_notifs):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "notifications"])
        assert result.exit_code == 0
        assert "Active Mobile Notifications" in result.output
        assert "492810" in result.output
        assert "News App" in result.output

    # Test --otp-only filter
    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_notifs):
        result_otp = cli_runner.invoke(cli, ["-s", str(mock_socket), "notifications", "--otp-only"])
        assert result_otp.exit_code == 0
        assert "492810" in result_otp.output
        assert "News App" not in result_otp.output


def test_cli_ping_command(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify ping command invokes IPC send_ping and prints confirmation."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    ping_result = {"device_id": "phone_123", "success": True, "message": "Ping transmitted"}

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=ping_result):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "ping", "phone_123", "-m", "Hello!"])
        assert result.exit_code == 0
        assert "Ping successfully sent to 'phone_123'" in result.output


def test_cli_unpair_command(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify unpair command invokes unpair_device and reports success."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    unpair_result = {"device_id": "phone_123", "success": True}

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=unpair_result):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "unpair", "phone_123", "--yes"])
        assert result.exit_code == 0
        assert "Device 'phone_123' successfully unpaired" in result.output


def test_cli_connect_command(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify connect command invokes connect_device and reports success."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    connect_result = {"device_id": "phone_123", "state": "CONNECTED"}

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=connect_result):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "connect", "phone_123"])
        assert result.exit_code == 0
        assert "Connected to device 'phone_123'" in result.output


def test_cli_dismiss_and_reply_commands(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify dismiss and reply notification subcommands."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    # Dismiss
    dismiss_res = {"device_id": "phone_123", "notification_id": "notif_45", "success": True}
    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=dismiss_res):
        res1 = cli_runner.invoke(cli, ["-s", str(mock_socket), "dismiss", "phone_123", "notif_45"])
        assert res1.exit_code == 0
        assert "Notification 'notif_45' dismissed on device 'phone_123'" in res1.output

    # Reply
    reply_res = {"device_id": "phone_123", "reply_id": "reply_token_1", "success": True}
    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=reply_res):
        res2 = cli_runner.invoke(
            cli,
            ["-s", str(mock_socket), "reply", "phone_123", "reply_token_1", "On my way!"],
        )
        assert res2.exit_code == 0
        assert 'Reply sent to device \'phone_123\': "On my way!"' in res2.output


def test_cli_daemon_status_command(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify daemon status subcommand displays Rich status panel."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    status_data = {
        "running": True,
        "device_id": "0123456789abcdef",
        "device_name": "PenguLink-PC",
        "connected_devices": 1,
        "discovered_devices": 2,
        "known_devices": 3,
        "uptime": "2h 15m",
    }

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=status_data):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "daemon", "status"])
        assert result.exit_code == 0
        assert "PenguLink Daemon Status" in result.output
        assert "RUNNING (Active)" in result.output
        assert "0123456789abcdef" in result.output
        assert "PenguLink-PC" in result.output


def test_cli_logs_command(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify logs command renders recent records in Rich table."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    sample_logs = [
        {
            "time_str": "2026-10-04 04:00:00",
            "level": "INFO",
            "logger": "pengulink.daemon.service",
            "message": "Daemon started cleanly",
        },
        {
            "time_str": "2026-10-04 04:01:00",
            "level": "WARNING",
            "logger": "pengulink.network.tcp",
            "message": "Connection dropped by peer",
        },
    ]

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_logs):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "logs", "--limit", "10"])
        assert result.exit_code == 0
        assert "Recent Daemon Log Records" in result.output
        assert "Daemon started cleanly" in result.output
        assert "Connection dropped by peer" in result.output


def test_cli_json_output_mode(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify --json / --json-output produces valid JSON without Rich markup."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    sample_devices = [
        {
            "device_id": "test_id_123",
            "device_name": "Device 1",
            "state": "CONNECTED",
            "is_paired": True,
            "battery": None,
        }
    ]

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_devices):
        result = cli_runner.invoke(cli, ["--json", "-s", str(mock_socket), "devices"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert parsed[0]["device_id"] == "test_id_123"


def test_cli_defensive_none_handling(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify format_battery_badge and notifications command safely handle explicit None values."""
    from pengulink.cli.main import format_battery_badge

    # 1. battery with None current_charge
    assert format_battery_badge({"current_charge": None}) == "[dim]AC / Desktop[/]"

    # 2. notification with None device_id
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    sample_notifs = [
        {
            "notification_id": "notif_none_id",
            "device_id": None,
            "app_name": "TestApp",
            "title": "Alert",
            "text": "Code: 1234",
            "auth_code": "1234",
            "time_str": "12:00:00",
        }
    ]

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_notifs):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "notifications"])
        assert result.exit_code == 0
        assert "Active Mobile Notifications" in result.output
        assert "notif_none_id" in result.output


# =====================================================================
# Clipboard CLI Tests
# =====================================================================

def test_cli_clipboard_get_interactive(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify clipboard get displays Rich panel in interactive mode."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    sample_result = {
        "content": "Secret text here",
        "timestamp": 1728456789000,
        "digest": "abcdef123456",
        "bytes": 16,
    }

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_result):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "clipboard", "get"])
        assert result.exit_code == 0
        assert "Secret text here" in result.output


def test_cli_clipboard_get_raw(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify clipboard get --raw prints unadorned text without decorations."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    sample_result = {"content": "Exact raw clipboard text", "bytes": 24}

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_result):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "clipboard", "get", "--raw"])
        assert result.exit_code == 0
        assert result.output == "Exact raw clipboard text"


def test_cli_clipboard_get_json(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify clipboard get --json outputs machine-readable JSON."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    sample_result = {"content": "JSON clipboard text", "timestamp": 1000, "bytes": 19}

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_result):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "clipboard", "get", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["content"] == "JSON clipboard text"
        assert data["bytes"] == 19


def test_cli_clipboard_get_empty(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify clipboard get displays message when clipboard is empty."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    sample_result = {"content": "", "timestamp": 0, "bytes": 0}

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_result):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "clipboard", "get"])
        assert result.exit_code == 0
        assert "Clipboard is empty." in result.output


def test_cli_clipboard_set_success(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify clipboard set updates clipboard and outputs confirmation without echoing raw text."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    called_args: Dict[str, Any] = {}

    def mock_call(method: str, params: Any = None, timeout: float = 10.0) -> Any:
        called_args["method"] = method
        called_args["params"] = params
        return {"status": "ok", "bytes": len(params["content"].encode("utf-8")), "timestamp": 12345}

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=mock_call):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "clipboard", "set", "My Private Secret"])
        assert result.exit_code == 0
        assert called_args["method"] == "set_clipboard"
        assert called_args["params"]["content"] == "My Private Secret"
        # Zero-logging privacy verification: confirmation message must not echo the raw content
        assert "My Private Secret" not in result.output
        assert "Clipboard updated (17 bytes)" in result.output


def test_cli_clipboard_set_json(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify clipboard set --json outputs JSON result."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value={"status": "ok", "bytes": 5}):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "clipboard", "set", "hello", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"


def test_cli_clipboard_set_piped_stdin(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify clipboard set reads from stdin when text is '-' or piped."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    called_args: Dict[str, Any] = {}

    def mock_call(method: str, params: Any = None, timeout: float = 10.0) -> Any:
        called_args["params"] = params
        return {"status": "ok", "bytes": len(params["content"].encode("utf-8"))}

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=mock_call):
        result = cli_runner.invoke(
            cli,
            ["-s", str(mock_socket), "clipboard", "set", "-"],
            input="Piped Stdin Content",
        )
        assert result.exit_code == 0
        assert called_args["params"]["content"] == "Piped Stdin Content"


def test_cli_clipboard_set_oversized_validation(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify clipboard set rejects oversized text (> 120 KB)."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    oversized = "X" * 125_000
    result = cli_runner.invoke(cli, ["-s", str(mock_socket), "clipboard", "set", oversized])
    assert result.exit_code != 0
    assert "exceeds maximum allowed size" in result.output

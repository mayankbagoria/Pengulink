"""tests.unit.test_cli_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial challenge test harness for PenguLink CLI (pengulink.cli.main).
Tests daemon-down conditions, missing/invalid parameters, Rich vs JSON formatting,
8-character verification code panels, OTP badge filtering, IPC errors, and edge cases.
"""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import (
    cli,
    format_battery_badge,
    format_otp_badge,
    format_paired_badge,
    format_state_badge,
)
from pengulink.core.exceptions import (
    DaemonNotRunningError,
    DeviceNotFoundError,
    IPCError,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing Click CLI test runner with wide column configuration."""
    return CliRunner(env={"COLUMNS": "160"})


# =====================================================================
# 1. Daemon-Down Scenarios (All 11 Subcommands)
# =====================================================================

ALL_CLI_SUBCOMMANDS = [
    ["discover"],
    ["devices"],
    ["pair", "remote_dev_1"],
    ["unpair", "remote_dev_1", "--yes"],
    ["connect", "remote_dev_1"],
    ["ping", "remote_dev_1"],
    ["notifications"],
    ["dismiss", "remote_dev_1", "notif_99"],
    ["reply", "remote_dev_1", "reply_token_1", "Hello!"],
    ["daemon", "status"],
    ["logs"],
]


@pytest.mark.parametrize("subcommand", ALL_CLI_SUBCOMMANDS, ids=lambda c: "_".join(c[:2]))
def test_cli_all_subcommands_daemon_not_running_socket_missing(
    cli_runner: CliRunner, tmp_path: Path, subcommand: List[str]
) -> None:
    """Verify all 11 CLI subcommands exit with code 1 and clean user-facing error when socket does not exist."""
    nonexistent_socket = tmp_path / "absent_daemon.sock"
    args = ["-s", str(nonexistent_socket)] + subcommand
    result = cli_runner.invoke(cli, args)

    assert result.exit_code == 1, f"Failed on command {subcommand}: exit code was {result.exit_code}"
    assert "Traceback (most recent call last)" not in result.output, f"Traceback leaked on {subcommand}"
    assert "PenguLink daemon is not running" in result.output, f"Missing offline error message on {subcommand}"


@pytest.mark.parametrize("subcommand", ALL_CLI_SUBCOMMANDS, ids=lambda c: "_".join(c[:2]))
def test_cli_all_subcommands_daemon_not_running_connection_refused(
    cli_runner: CliRunner, tmp_path: Path, subcommand: List[str]
) -> None:
    """Verify all 11 CLI subcommands exit with code 1 and clean error when socket exists but is inactive."""
    stale_socket = tmp_path / "stale_daemon.sock"
    # Create and immediately close a Unix socket to leave a stale node
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(stale_socket))
    finally:
        sock.close()

    args = ["-s", str(stale_socket)] + subcommand
    result = cli_runner.invoke(cli, args)

    assert result.exit_code == 1, f"Failed on command {subcommand}: exit code was {result.exit_code}"
    assert "Traceback (most recent call last)" not in result.output, f"Traceback leaked on {subcommand}"
    assert "PenguLink daemon is not running" in result.output, f"Missing offline error message on {subcommand}"


# =====================================================================
# 2. Invalid Arguments and Missing Parameters
# =====================================================================

MISSING_PARAM_CASES = [
    (["pair"], "Missing argument 'DEVICE_ID'"),
    (["unpair"], "Missing argument 'DEVICE_ID'"),
    (["connect"], "Missing argument 'DEVICE_ID'"),
    (["ping"], "Missing argument 'DEVICE_ID'"),
    (["dismiss"], "Missing argument 'DEVICE_ID'"),
    (["dismiss", "dev_123"], "Missing argument 'NOTIFICATION_ID'"),
    (["reply"], "Missing argument 'DEVICE_ID'"),
    (["reply", "dev_123"], "Missing argument 'REPLY_ID'"),
    (["reply", "dev_123", "rep_1"], "Missing argument 'MESSAGE'"),
]


@pytest.mark.parametrize("args,expected_err", MISSING_PARAM_CASES, ids=lambda c: "_".join(c[0]))
def test_cli_missing_required_arguments(
    cli_runner: CliRunner, args: List[str], expected_err: str
) -> None:
    """Verify Click argument validation rejects missing parameters with exit code 2."""
    result = cli_runner.invoke(cli, args)
    assert result.exit_code == 2
    assert "Error:" in result.output
    assert expected_err in result.output


INVALID_OPTION_CASES = [
    (["discover", "--timeout", "not_a_float"], "Invalid value for '--timeout' / '-t'"),
    (["logs", "--limit", "abc"], "Invalid value for '--limit' / '-n'"),
    (["logs", "--level", "SUPER_CRITICAL"], "Invalid value for '--level' / '-l'"),
    (["daemon", "run", "--log-level", "VERBOSE"], "Invalid value for '--log-level' / '-l'"),
]


@pytest.mark.parametrize("args,expected_substr", INVALID_OPTION_CASES, ids=lambda c: "_".join(c[0]))
def test_cli_invalid_option_values(
    cli_runner: CliRunner, args: List[str], expected_substr: str
) -> None:
    """Verify Click option validation rejects invalid types and choices with exit code 2."""
    result = cli_runner.invoke(cli, args)
    assert result.exit_code == 2
    assert "Error:" in result.output
    assert expected_substr in result.output


def test_cli_unpair_abort_when_not_confirmed(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify unpair command aborts cleanly without calling IPC when user declines confirmation."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync") as mock_ipc:
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "unpair", "phone_abc"], input="n\n")
        assert result.exit_code == 1
        assert "Aborted" in result.output
        mock_ipc.assert_not_called()


# =====================================================================
# 3. Formatting Verification: JSON Mode vs Rich Output
# =====================================================================

SAMPLE_PEERS = [
    {
        "device_id": "pixel_9_dev_12345",
        "device_name": "Pixel 9 Pro",
        "device_type": "phone",
        "sender_ip": "192.168.1.50",
        "tcp_port": 1716,
        "is_paired": True,
    }
]

SAMPLE_DEVICES = [
    {
        "device_id": "dev_phone_1",
        "device_name": "Galaxy S23",
        "device_type": "phone",
        "state": "CONNECTED",
        "is_paired": True,
        "battery": {"current_charge": 77, "is_charging": False, "threshold_event": 0},
    }
]

SAMPLE_NOTIFICATIONS = [
    {
        "notification_id": "sms_notif_100",
        "device_id": "dev_phone_1",
        "app_name": "Messages",
        "title": "Auth Code",
        "text": "Your passcode is 819203",
        "auth_code": "819203",
        "request_reply_id": "reply_token_99",
        "time_str": "14:30:00",
    }
]

SAMPLE_LOGS = [
    {
        "time_str": "2026-10-04 10:00:00",
        "level": "INFO",
        "logger": "pengulink.service",
        "message": "Peer connected",
    }
]

JSON_MODE_TEST_CASES = [
    (["discover"], SAMPLE_PEERS, list),
    (["devices"], SAMPLE_DEVICES, list),
    (["pair", "dev_phone_1"], {"verification_code": "3E4F5A6B", "status": "paired"}, dict),
    (["unpair", "dev_phone_1", "--yes"], {"success": True}, dict),
    (["connect", "dev_phone_1"], {"state": "CONNECTED"}, dict),
    (["ping", "dev_phone_1"], {"success": True}, dict),
    (["notifications"], SAMPLE_NOTIFICATIONS, list),
    (["dismiss", "dev_phone_1", "sms_notif_100"], {"success": True}, dict),
    (["reply", "dev_phone_1", "reply_token_99", "Got it"], {"success": True}, dict),
    (["daemon", "status"], {"running": True, "device_id": "local_dev_1", "uptime": "1h"}, dict),
    (["logs"], SAMPLE_LOGS, list),
]


@pytest.mark.parametrize("subcommand,mock_res,expected_type", JSON_MODE_TEST_CASES)
def test_cli_json_output_mode_all_subcommands(
    cli_runner: CliRunner,
    tmp_path: Path,
    subcommand: List[str],
    mock_res: Any,
    expected_type: type,
) -> None:
    """Verify --json produces valid JSON matching expected schema for all subcommands."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=mock_res):
        args = ["--json", "-s", str(mock_socket)] + subcommand
        result = cli_runner.invoke(cli, args)
        assert result.exit_code == 0, f"Command {subcommand} failed in JSON mode: {result.output}"

        # Must parse as clean JSON
        parsed = json.loads(result.output)
        assert isinstance(parsed, expected_type)
        if expected_type is list:
            assert len(parsed) == len(mock_res)


# =====================================================================
# 4. 8-Character Visual Verification Code Panel & Pairing States
# =====================================================================

def test_cli_pair_success_with_verification_code_panel(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify pair command renders Rich panel with 8-character verification code and device name."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    pair_data = {
        "device_id": "pixel_phone_abc",
        "device_name": "Pixel 8 Pro",
        "verification_code": "8D3A1F4C",
        "status": "paired",
    }

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=pair_data):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "pair", "pixel_phone_abc"])
        assert result.exit_code == 0
        assert "Pairing Verification" in result.output
        assert "8D3A1F4C" in result.output
        assert "Pixel 8 Pro" in result.output
        assert "Pairing successful" in result.output


def test_cli_pair_remote_rejection(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify pair command terminates with exit code 1 when remote device rejects pairing."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    pair_data = {
        "device_id": "pixel_phone_abc",
        "device_name": "Pixel 8 Pro",
        "verification_code": "8D3A1F4C",
        "status": "rejected",
    }

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=pair_data):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "pair", "pixel_phone_abc"])
        assert result.exit_code == 1
        assert "Pairing rejected by device" in result.output


def test_cli_pair_remote_timeout(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify pair command terminates with exit code 1 when pairing handshake times out."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    pair_data = {
        "device_id": "pixel_phone_abc",
        "device_name": "Pixel 8 Pro",
        "verification_code": "8D3A1F4C",
        "status": "timeout",
    }

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=pair_data):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "pair", "pixel_phone_abc"])
        assert result.exit_code == 1
        assert "Pairing timed out waiting for device" in result.output


def test_cli_pair_pending_confirmation_poll_success(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify pair command polls daemon when status is pending and detects confirmed pairing."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    pair_init = {
        "device_id": "pixel_phone_abc",
        "device_name": "Pixel 8 Pro",
        "verification_code": "8D3A1F4C",
        "status": "pending",
    }

    devices_poll = [
        {"device_id": "pixel_phone_abc", "is_paired": True, "state": "CONNECTED"}
    ]

    def mock_ipc_call(method: str, params: Any = None, timeout: Any = None) -> Any:
        if method == "pair_device":
            return pair_init
        elif method == "get_devices":
            return devices_poll
        return {}

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=mock_ipc_call):
        with patch("time.sleep", return_value=None):  # Fast polling in tests
            result = cli_runner.invoke(cli, ["-s", str(mock_socket), "pair", "pixel_phone_abc", "-t", "5"])
            assert result.exit_code == 0
            assert "Pairing successful" in result.output


def test_cli_pair_pending_confirmation_poll_aborts_on_error(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify pair polling aborts cleanly with exit code 1 if device transitions to ERROR state."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    pair_init = {
        "device_id": "pixel_phone_abc",
        "device_name": "Pixel 8 Pro",
        "verification_code": "8D3A1F4C",
        "status": "pending",
    }

    devices_poll = [
        {"device_id": "pixel_phone_abc", "is_paired": False, "state": "ERROR"}
    ]

    def mock_ipc_call(method: str, params: Any = None, timeout: Any = None) -> Any:
        if method == "pair_device":
            return pair_init
        elif method == "get_devices":
            return devices_poll
        return {}

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=mock_ipc_call):
        with patch("time.sleep", return_value=None):
            result = cli_runner.invoke(cli, ["-s", str(mock_socket), "pair", "pixel_phone_abc", "-t", "5"])
            assert result.exit_code == 1
            assert "Pairing negotiation ended without confirmation" in result.output


# =====================================================================
# 5. OTP / 2FA Highlighting and --otp-only Filtering
# =====================================================================

def test_cli_notifications_otp_badge_and_filtering(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify OTP codes are formatted with high contrast and filtered by --otp-only."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    notifs = [
        {
            "notification_id": "n1",
            "device_id": "device_11112222",
            "app_name": "Messages",
            "title": "Bank Verification",
            "text": "Your OTP code is 948302.",
            "auth_code": "948302",
            "request_reply_id": None,
            "time_str": "12:00",
        },
        {
            "notification_id": "n2",
            "device_id": "device_11112222",
            "app_name": "Twitter",
            "title": "New Tweet",
            "text": "Check out this update!",
            "auth_code": None,
            "request_reply_id": "reply_tw",
            "time_str": "12:05",
        },
    ]

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=notifs):
        # 1. Full view: both present
        res_full = cli_runner.invoke(cli, ["-s", str(mock_socket), "notifications"])
        assert res_full.exit_code == 0
        assert "948302" in res_full.output
        assert "Twitter" in res_full.output

        # 2. OTP-only view: Twitter excluded
        res_otp = cli_runner.invoke(cli, ["-s", str(mock_socket), "notifications", "--otp-only"])
        assert res_otp.exit_code == 0
        assert "948302" in res_otp.output
        assert "Twitter" not in res_otp.output


def test_cli_notifications_empty_and_no_otp_found(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify clean display when notification list is empty or has no OTPs."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    # Empty list
    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=[]):
        res1 = cli_runner.invoke(cli, ["-s", str(mock_socket), "notifications"])
        assert res1.exit_code == 0
        assert "No active notifications found" in res1.output

    # Non-empty list, but none have OTP and --otp-only specified
    notifs_no_otp = [
        {
            "notification_id": "n2",
            "device_id": "device_1",
            "app_name": "Email",
            "title": "Hello",
            "text": "Meeting at 3",
            "auth_code": None,
            "time_str": "12:00",
        }
    ]
    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=notifs_no_otp):
        res2 = cli_runner.invoke(cli, ["-s", str(mock_socket), "notifications", "--otp-only"])
        assert res2.exit_code == 0
        assert "No active notifications found" in res2.output


# =====================================================================
# 6. Rich Badges and Formatting Helpers (All 8 States + Battery + OTP)
# =====================================================================

ALL_8_STATES = [
    ("CONNECTED", "● CONNECTED"),
    ("CONNECTING", "◌ CONNECTING"),
    ("PAIRING", "◌ PAIRING"),
    ("PAIRED", "● PAIRED"),
    ("DISCOVERED", "● DISCOVERED"),
    ("DISCONNECTING", "○ DISCONNECTING"),
    ("DISCONNECTED", "○ DISCONNECTED"),
    ("ERROR", "✕ ERROR"),
    ("UNKNOWN_STATE", "UNKNOWN_STATE"),
]


@pytest.mark.parametrize("state,expected_label", ALL_8_STATES)
def test_format_state_badge_all_states(state: str, expected_label: str) -> None:
    """Verify badge formatting for all 8 KDE Connect lifecycle states."""
    badge = format_state_badge(state)
    assert expected_label in badge


def test_format_battery_badge_cases() -> None:
    """Verify battery formatting for charging, low battery, normal, desktop/AC, and None."""
    assert format_battery_badge(None) == "[dim]N/A[/]"
    assert format_battery_badge({}) == "[dim]N/A[/]"
    assert format_battery_badge({"current_charge": -1}) == "[dim]AC / Desktop[/]"

    # Charging
    charging_badge = format_battery_badge({"current_charge": 50, "is_charging": True})
    assert "50%" in charging_badge
    assert "⚡ (Charging)" in charging_badge

    # Low battery (<= 15%)
    low_badge = format_battery_badge({"current_charge": 12, "is_charging": False})
    assert "12%" in low_badge
    assert "🪫 (Low)" in low_badge

    # Normal battery
    normal_badge = format_battery_badge({"current_charge": 80, "is_charging": False})
    assert "80%" in normal_badge
    assert "🔋" in normal_badge


def test_format_otp_badge_cases() -> None:
    """Verify OTP badge styling."""
    assert format_otp_badge(None) == "[dim]-[/]"
    assert format_otp_badge("") == "[dim]-[/]"
    assert "999888" in format_otp_badge("999888")


# =====================================================================
# 7. IPC Domain Errors & Traceback Suppression
# =====================================================================

def test_cli_ipc_domain_error_clean_exit(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify IPC errors (e.g. DeviceNotFound, PairingRejected) exit with code 1 and no tracebacks."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    with patch(
        "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
        side_effect=IPCError("Device 'ghost_id' not found in store"),
    ):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "ping", "ghost_id"])
        assert result.exit_code == 1
        assert "IPC Error:" in result.output
        assert "Device 'ghost_id' not found in store" in result.output
        assert "Traceback" not in result.output


def test_cli_daemon_status_stopped_display(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify daemon status subcommand displays STOPPED badge and exits with code 1 when daemon is down."""
    mock_socket = tmp_path / "nonexistent.sock"
    result = cli_runner.invoke(cli, ["-s", str(mock_socket), "daemon", "status"])
    assert result.exit_code == 1
    assert "STOPPED (Inactive)" in result.output
    assert "PenguLink daemon is not running" in result.output
    assert "Traceback" not in result.output


# =====================================================================
# 8. Logs Filtering & Empty State
# =====================================================================

def test_cli_logs_empty_state(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify logs command displays clean empty state message."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=[]):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "logs"])
        assert result.exit_code == 0
        assert "No log entries recorded yet" in result.output


# =====================================================================
# 9. Adversarial Resiliency: Malformed/Null Fields & Daemon Garbage
# =====================================================================

def test_cli_notifications_with_omitted_device_id(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify notifications command handles notification when device_id is omitted."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    notifs = [
        {
            "notification_id": "n_missing",
            # device_id omitted
            "app_name": "TestApp",
            "title": "Title",
            "text": "Text content",
            "auth_code": "112233",
            "request_reply_id": None,
            "time_str": "12:00",
        }
    ]

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=notifs):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "notifications"])
        assert result.exit_code == 0
        assert "TestApp" in result.output
        assert "112233" in result.output


def test_cli_devices_with_missing_battery(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify devices command displays N/A when battery information is omitted/None."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    devices = [
        {
            "device_id": "dev_no_bat",
            "device_name": "DesktopPC",
            "device_type": "desktop",
            "state": "CONNECTED",
            "is_paired": True,
            "battery": None,
        }
    ]

    with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=devices):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "devices"])
        assert result.exit_code == 0
        assert "DesktopPC" in result.output
        assert "N/A" in result.output


def test_cli_corrupted_ipc_response_handling(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Verify corrupted/non-JSON daemon response results in clean exit code 1 with no traceback."""
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    with patch(
        "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
        side_effect=json.JSONDecodeError("Expecting value", "bad json", 0),
    ):
        result = cli_runner.invoke(cli, ["-s", str(mock_socket), "devices"])
        assert result.exit_code == 1
        assert "Unexpected Error:" in result.output
        assert "Traceback" not in result.output


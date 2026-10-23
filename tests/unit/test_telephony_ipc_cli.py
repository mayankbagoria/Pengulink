"""tests.unit.test_telephony_ipc_cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit test suite for Milestone 29:
- Telephony & PauseMusic JSON-RPC IPC server handlers and client methods.
- Error codes -32017 (TELEPHONY_ERROR) and -32018 (PAUSE_MUSIC_ERROR).
- Direct PenguLinkDaemon methods for Telephony and PauseMusic operations.
- Click CLI commands: pengulink telephony (calls, mute-ringer, reject-call).
- Click CLI commands: pengulink pausemusic (status, configure).
- Rich formatting (tables, badges, panels, checkmarks) and clean JSON outputs.
- Error scenarios: daemon unreachable, offline devices, invalid choices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli, format_call_duration, format_call_state_badge
from pengulink.core.exceptions import (
    ConnectionClosedError,
    DaemonNotRunningError,
    DeviceNotFoundError,
    InvalidCallStateError,
    IPCError,
    PauseMusicConfigError,
    TelephonyError,
)
from pengulink.daemon.ipc import (
    DEVICE_NOT_CONNECTED,
    DEVICE_NOT_FOUND,
    PAUSE_MUSIC_ERROR,
    TELEPHONY_ERROR,
    PenguLinkIPCClient,
    PenguLinkIPCServer,
)
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.services.telephony import (
    AwaitableCallList,
    CallItem,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing Click CLI runner configured with standard terminal width."""
    return CliRunner(env={"COLUMNS": "160"})


# =====================================================================
# 1. Telephony Data Structures & Helpers Tests
# =====================================================================

class TestTelephonyDataStructures:
    """Test AwaitableCallList and CallItem dual-access behavior."""

    def test_call_item_dict_and_attribute_access(self) -> None:
        item = CallItem({
            "device_id": "dev_test",
            "phone_number": "+15551234",
            "contact_name": "Test User",
            "state": "RINGING",
            "duration": 12.5,
            "is_active": True,
        })
        assert item["device_id"] == "dev_test"
        assert item.device_id == "dev_test"
        assert item.phone_number == "+15551234"
        assert item.contact_name == "Test User"
        assert item.state == "RINGING"
        assert item.duration == 12.5
        assert item.is_active is True
        assert item.to_dict() == dict(item)

        with pytest.raises(AttributeError):
            _ = item.non_existent_key

    async def test_awaitable_call_list_sync_and_async(self) -> None:
        item1 = CallItem({"device_id": "d1", "phone_number": "+111"})
        item2 = CallItem({"device_id": "d2", "phone_number": "+222"})
        cl = AwaitableCallList([item1, item2])

        # Synchronous inspection
        assert len(cl) == 2
        assert cl[0].phone_number == "+111"
        assert cl[1].device_id == "d2"

        # Asynchronous awaiting
        awaited = await cl
        assert awaited is cl
        assert len(awaited) == 2

    def test_format_call_helpers(self) -> None:
        assert "RINGING" in format_call_state_badge("RINGING")
        assert "TALKING" in format_call_state_badge("TALKING")
        assert "MISSED" in format_call_state_badge("MISSED")
        assert "IDLE" in format_call_state_badge("IDLE")

        assert format_call_duration(14.0) == "00:14"
        assert format_call_duration(125.0) == "02:05"
        assert format_call_duration(3665.0) == "01:01:05"

    def test_format_call_duration_resilience(self) -> None:
        """Verify format_call_duration handles non-finite floats, negative values, and None safely."""
        # Non-finite floats return "00:00" without crashing
        assert format_call_duration(float("nan")) == "00:00"
        assert format_call_duration(float("inf")) == "00:00"
        assert format_call_duration(float("-inf")) == "00:00"

        # None and invalid inputs return "00:00"
        assert format_call_duration(None) == "00:00"
        assert format_call_duration("corrupted") == "00:00"  # type: ignore[arg-type]

        # Negative and zero values clamped to "00:00"
        assert format_call_duration(-1.0) == "00:00"
        assert format_call_duration(-999.0) == "00:00"
        assert format_call_duration(0.0) == "00:00"

        # Valid standard and long durations
        assert format_call_duration(14.0) == "00:14"
        assert format_call_duration(125.0) == "02:05"
        assert format_call_duration(3665.0) == "01:01:05"
        assert format_call_duration(360000.0) == "100:00:00"


# =====================================================================
# 2. Telephony IPC Server and Client Tests
# =====================================================================

class TestTelephonyIPCServerAndClient:
    """Test IPC server request routing, serialization, error codes, and client helpers."""

    async def test_ipc_telephony_get_calls_all(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_tel_calls.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            assert method in ("telephony_get_calls", "telephony_calls")
            return [
                {
                    "device_id": "phone_1",
                    "phone_number": "+15550011",
                    "contact_name": "Alice",
                    "state": "RINGING",
                    "duration": 5.2,
                    "is_active": True,
                },
                {
                    "device_id": "phone_2",
                    "phone_number": "+15550022",
                    "contact_name": "Bob",
                    "state": "TALKING",
                    "duration": 42.0,
                    "is_active": True,
                },
            ]

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            calls = await client.telephony_get_calls()
            assert len(calls) == 2
            assert calls[0]["contact_name"] == "Alice"
            assert calls[1]["state"] == "TALKING"
        finally:
            await server.stop()

    async def test_ipc_telephony_get_calls_filtered(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_tel_filter.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            assert params.get("device_id") == "phone_1"
            return [
                {
                    "device_id": "phone_1",
                    "phone_number": "+15550011",
                    "contact_name": "Alice",
                    "state": "RINGING",
                    "duration": 1.0,
                    "is_active": True,
                }
            ]

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            calls = await client.telephony_get_calls("phone_1")
            assert len(calls) == 1
            assert calls[0]["device_id"] == "phone_1"
        finally:
            await server.stop()

    async def test_ipc_telephony_calls_alias(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_tel_alias.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            return []

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            res = await client.telephony_calls()
            assert res == []
        finally:
            await server.stop()

    async def test_ipc_telephony_mute_ringer_success(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_tel_mute.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            assert method == "telephony_mute_ringer"
            assert params["device_id"] == "phone_1"
            return {"device_id": "phone_1", "action": "muteRinger", "success": True}

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            res = await client.telephony_mute_ringer("phone_1")
            assert res["success"] is True
            assert res["action"] == "muteRinger"
        finally:
            await server.stop()

    async def test_ipc_telephony_reject_call_success(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_tel_reject.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            assert method == "telephony_reject_call"
            assert params["device_id"] == "phone_1"
            assert params.get("phone_number") == "+15559999"
            return {"device_id": "phone_1", "action": "rejectCall", "success": True}

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            res = await client.telephony_reject_call("phone_1", phone_number="+15559999")
            assert res["success"] is True
            assert res["action"] == "rejectCall"
        finally:
            await server.stop()

    async def test_ipc_telephony_domain_error_32017(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_tel_err.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            raise TelephonyError("Call rejection failed on remote phone")

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc_info:
                await client.telephony_reject_call("phone_1")
            assert str(TELEPHONY_ERROR) in str(exc_info.value)
            assert "Call rejection failed" in str(exc_info.value)
        finally:
            await server.stop()

    async def test_ipc_telephony_invalid_call_state_error(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_tel_invalid_state.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            raise InvalidCallStateError("Cannot answer call in idle state")

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc_info:
                await client.telephony_get_calls("phone_1")
            assert str(TELEPHONY_ERROR) in str(exc_info.value)
        finally:
            await server.stop()

    async def test_ipc_telephony_device_not_found_32001(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_tel_notfound.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            raise DeviceNotFoundError("Unknown device id")

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc_info:
                await client.telephony_mute_ringer("ghost_device")
            assert str(DEVICE_NOT_FOUND) in str(exc_info.value)
        finally:
            await server.stop()

    async def test_ipc_telephony_device_not_connected_32002(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_tel_offline.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            raise ConnectionClosedError("Device is not connected")

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc_info:
                await client.telephony_mute_ringer("offline_device")
            assert str(DEVICE_NOT_CONNECTED) in str(exc_info.value)
        finally:
            await server.stop()


# =====================================================================
# 3. PauseMusic IPC Server and Client Tests
# =====================================================================

class TestPauseMusicIPCServerAndClient:
    """Test PauseMusic IPC status inspection, dynamic configuration, and error codes."""

    async def test_ipc_pausemusic_get_config_and_status(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_pm_status.sock"

        sample_status = {
            "condition": "ringing",
            "action": "pause",
            "resume": True,
            "condition_talking": False,
            "action_pause": True,
            "action_mute": False,
            "action_resume": True,
            "is_interrupted": True,
            "active_call_devices": ["phone_1"],
            "paused_players": ["vlc"],
            "muted_sinks": [],
            "config": {
                "condition": "ringing",
                "action": "pause",
                "condition_talking": False,
                "action_pause": True,
                "action_mute": False,
                "action_resume": True,
            },
        }

        async def handler(method: str, params: dict[str, Any]) -> Any:
            assert method in ("pausemusic_get_config", "pausemusic_status")
            return sample_status

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            res = await client.pausemusic_get_config()
            assert res["condition"] == "ringing"
            assert res["action"] == "pause"
            assert res["is_interrupted"] is True
            assert "phone_1" in res["active_call_devices"]
        finally:
            await server.stop()

    async def test_ipc_pausemusic_status_alias(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_pm_alias.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            return {"condition": "talking", "action": "both"}

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            res = await client.pausemusic_status()
            assert res["condition"] == "talking"
        finally:
            await server.stop()

    async def test_ipc_pausemusic_set_config_condition_and_actions(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_pm_set.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            assert method in ("pausemusic_set_config", "pausemusic_configure")
            assert params["condition"] == "talking"
            assert params["action"] == "mute"
            return {
                "condition": "talking",
                "action": "mute",
                "resume": True,
                "condition_talking": True,
                "action_pause": False,
                "action_mute": True,
                "action_resume": True,
            }

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            res = await client.pausemusic_set_config(condition="talking", action="mute")
            assert res["condition"] == "talking"
            assert res["action"] == "mute"
            assert res["condition_talking"] is True
        finally:
            await server.stop()

    async def test_ipc_pausemusic_configure_alias(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_pm_cfg_alias.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            return {"condition": "ringing", "action": "pause"}

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            res = await client.pausemusic_configure(condition="ringing")
            assert res["condition"] == "ringing"
        finally:
            await server.stop()

    async def test_ipc_pausemusic_domain_error_32018(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_pm_err.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            raise PauseMusicConfigError("Invalid condition parameter")

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc_info:
                await client.pausemusic_set_config(condition="invalid_choice")
            assert str(PAUSE_MUSIC_ERROR) in str(exc_info.value)
            assert "Invalid condition" in str(exc_info.value)
        finally:
            await server.stop()


# =====================================================================
# 4. Direct PenguLinkDaemon Methods & Capabilities Tests
# =====================================================================

class TestDaemonTelephonyAndPauseMusic:
    """Test direct daemon methods, wiring, and capability advertising."""

    async def test_daemon_telephony_and_pausemusic_instantiation(self, tmp_path: Path) -> None:
        daemon = PenguLinkDaemon(config_dir=tmp_path)
        assert hasattr(daemon, "telephony_service")
        assert hasattr(daemon, "telephony_plugin")
        assert hasattr(daemon, "pause_music_config")
        assert hasattr(daemon, "pause_music_service")
        assert hasattr(daemon, "pause_music_plugin")
        assert hasattr(daemon, "pausemusic_service")
        assert hasattr(daemon, "pausemusic_plugin")

        # Capabilities registration
        inc = daemon.plugin_registry.get_all_incoming_capabilities()
        outg = daemon.plugin_registry.get_all_outgoing_capabilities()
        assert "kdeconnect.telephony" in inc
        assert "kdeconnect.pausemusic" in inc
        assert "kdeconnect.telephony.request_mute" in outg
        assert "kdeconnect.telephony.request" in outg

        assert "kdeconnect.telephony" in daemon.config.device.incoming_capabilities
        assert "kdeconnect.pausemusic" in daemon.config.device.incoming_capabilities

    async def test_daemon_telephony_get_calls_empty_and_populated(self, tmp_path: Path) -> None:
        daemon = PenguLinkDaemon(config_dir=tmp_path)
        calls = await daemon.telephony_get_calls()
        assert calls == []

        # Simulate call in service
        await daemon.telephony_service.process_call_event(
            device_id="dev_1",
            event="ringing",
            phone_number="+15551234",
            contact_name="Grace Hopper",
        )
        calls = await daemon.telephony_get_calls()
        assert len(calls) == 1
        assert calls[0]["phone_number"] == "+15551234"
        assert calls[0]["contact_name"] == "Grace Hopper"

        # Filtered by device_id
        filtered = await daemon.telephony_get_calls("dev_1")
        assert len(filtered) == 1
        assert filtered[0]["device_id"] == "dev_1"

        empty = await daemon.telephony_get_calls("dev_none")
        assert empty == []

    async def test_daemon_telephony_mute_ringer_disconnected(self, tmp_path: Path) -> None:
        daemon = PenguLinkDaemon(config_dir=tmp_path)
        with pytest.raises(ConnectionClosedError):
            await daemon.telephony_mute_ringer("dev_offline")

    async def test_daemon_telephony_reject_call_disconnected(self, tmp_path: Path) -> None:
        daemon = PenguLinkDaemon(config_dir=tmp_path)
        with pytest.raises(ConnectionClosedError):
            await daemon.telephony_reject_call("dev_offline")

    async def test_daemon_telephony_missing_device_id_raises_value_error(self, tmp_path: Path) -> None:
        daemon = PenguLinkDaemon(config_dir=tmp_path)
        with pytest.raises(ValueError):
            await daemon.telephony_mute_ringer("")
        with pytest.raises(ValueError):
            await daemon.telephony_reject_call("")

    async def test_daemon_pausemusic_get_config_and_status(self, tmp_path: Path) -> None:
        daemon = PenguLinkDaemon(config_dir=tmp_path)
        cfg = await daemon.pausemusic_get_config()
        assert "condition" in cfg
        assert "action" in cfg
        assert "resume" in cfg
        assert "is_interrupted" in cfg
        assert "config" in cfg

        status = await daemon.pausemusic_status()
        assert status["condition"] == cfg["condition"]

    async def test_daemon_pausemusic_set_config(self, tmp_path: Path) -> None:
        daemon = PenguLinkDaemon(config_dir=tmp_path)
        res = await daemon.pausemusic_set_config(condition="talking", action="mute", resume=False)
        assert res["condition"] == "talking"
        assert res["action"] == "mute"
        assert res["resume"] is False
        assert res["condition_talking"] is True

    async def test_daemon_ipc_request_routing(self, tmp_path: Path) -> None:
        daemon = PenguLinkDaemon(config_dir=tmp_path)

        # Route telephony_get_calls
        res_calls = await daemon._handle_ipc_request("telephony_get_calls", {})
        assert isinstance(res_calls, list)

        # Route pausemusic_get_config
        res_pm = await daemon._handle_ipc_request("pausemusic_get_config", {})
        assert isinstance(res_pm, dict)
        assert "condition" in res_pm

        # Route pausemusic_set_config
        res_update = await daemon._handle_ipc_request(
            "pausemusic_set_config", {"condition": "talking", "action": "both"}
        )
        assert res_update["condition"] == "talking"
        assert res_update["action"] == "both"


# =====================================================================
# 5. Telephony CLI Commands Tests
# =====================================================================

class TestTelephonyCLI:
    """Test Click CLI commands for 'pengulink telephony'."""

    def test_cli_telephony_calls_rich_table_all(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_calls = [
            {
                "device_id": "phone_alpha",
                "phone_number": "+15551234567",
                "contact_name": "Ada Lovelace",
                "state": "RINGING",
                "duration": 14.0,
                "is_active": True,
            },
            {
                "device_id": "phone_beta",
                "phone_number": "+15559876543",
                "contact_name": "Charles Babbage",
                "state": "TALKING",
                "duration": 125.0,
                "is_active": True,
            },
        ]

        with patch("pengulink.cli.main.run_ipc_call", return_value=sample_calls) as mock_ipc:
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "telephony", "calls"])
            assert res.exit_code == 0
            assert "Active Phone Calls" in res.output
            assert "phone_alpha" in res.output
            assert "+15551234567" in res.output
            assert "Ada Lovelace" in res.output
            assert "RINGING" in res.output
            assert "00:14" in res.output
            assert "02:05" in res.output
            mock_ipc.assert_called_once_with(mock_ipc.call_args[0][0], "telephony_get_calls", {})

    def test_cli_telephony_calls_filter_device_id_option(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        with patch("pengulink.cli.main.run_ipc_call", return_value=[]) as mock_ipc:
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "telephony", "calls", "--device-id", "phone_123"])
            assert res.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0], "telephony_get_calls", {"device_id": "phone_123"}
            )
            assert "No active calls for device 'phone_123'" in res.output

    def test_cli_telephony_calls_filter_device_id_argument(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        with patch("pengulink.cli.main.run_ipc_call", return_value=[]) as mock_ipc:
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "telephony", "calls", "phone_xyz"])
            assert res.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0], "telephony_get_calls", {"device_id": "phone_xyz"}
            )
            assert "No active calls for device 'phone_xyz'" in res.output

    def test_cli_telephony_calls_empty_all(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        with patch("pengulink.cli.main.run_ipc_call", return_value=[]):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "telephony", "calls"])
            assert res.exit_code == 0
            assert "No active phone calls." in res.output

    def test_cli_telephony_calls_empty_filtered(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        with patch("pengulink.cli.main.run_ipc_call", return_value=[]):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "telephony", "calls", "--device-id", "dev_none"])
            assert res.exit_code == 0
            assert "No active calls for device 'dev_none'" in res.output

    def test_cli_telephony_calls_json_output(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_calls = [
            {
                "device_id": "phone_json",
                "phone_number": "+15554321000",
                "contact_name": "Margaret Hamilton",
                "state": "RINGING",
                "duration": 3.5,
                "is_active": True,
            }
        ]

        with patch("pengulink.cli.main.run_ipc_call", return_value=sample_calls):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "telephony", "calls", "--json"])
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["phone_number"] == "+15554321000"
            assert data[0]["contact_name"] == "Margaret Hamilton"

    def test_cli_telephony_mute_ringer_success_and_json(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        # Rich text check
        with patch("pengulink.cli.main.run_ipc_call", return_value={"device_id": "phone_1", "action": "muteRinger", "success": True}):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "telephony", "mute-ringer", "phone_1"])
            assert res.exit_code == 0
            assert "Requested ringer mute for device 'phone_1'" in res.output

        # JSON flag check
        with patch("pengulink.cli.main.run_ipc_call", return_value={"device_id": "phone_1", "action": "muteRinger", "success": True}):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "telephony", "mute-ringer", "phone_1", "--json"])
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert data["success"] is True
            assert data["action"] == "muteRinger"

    def test_cli_telephony_reject_call_success_and_json(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        # Rich text check
        with patch("pengulink.cli.main.run_ipc_call", return_value={"device_id": "phone_1", "action": "rejectCall", "success": True}):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "telephony", "reject-call", "phone_1"])
            assert res.exit_code == 0
            assert "Requested call rejection for device 'phone_1'" in res.output

        # JSON flag check with phone-number
        with patch("pengulink.cli.main.run_ipc_call", return_value={"device_id": "phone_1", "action": "rejectCall", "success": True}) as mock_ipc:
            res = cli_runner.invoke(
                cli,
                ["-s", str(mock_socket), "telephony", "reject-call", "phone_1", "--phone-number", "+15559999", "--json"],
            )
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert data["success"] is True
            assert data["action"] == "rejectCall"
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0], "telephony_reject_call", {"device_id": "phone_1", "phone_number": "+15559999"}
            )


# =====================================================================
# 6. PauseMusic CLI Commands Tests
# =====================================================================

class TestPauseMusicCLI:
    """Test Click CLI commands for 'pengulink pausemusic'."""

    def test_cli_pausemusic_status_rich_panel(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        status_data = {
            "condition": "ringing",
            "action": "pause",
            "resume": True,
            "condition_talking": False,
            "action_pause": True,
            "action_mute": False,
            "action_resume": True,
            "is_interrupted": True,
            "active_call_devices": ["phone_alpha"],
            "paused_players": ["Spotify"],
            "muted_sinks": [],
        }

        with patch("pengulink.cli.main.run_ipc_call", return_value=status_data):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "pausemusic", "status"])
            assert res.exit_code == 0
            assert "PauseMusic Status & Policy" in res.output
            assert "Trigger Condition:" in res.output
            assert "ringing" in res.output
            assert "Action Policy:" in res.output
            assert "ACTIVE INTERRUPTION" in res.output
            assert "phone_alpha" in res.output
            assert "Spotify" in res.output

    def test_cli_pausemusic_status_json_output(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        status_data = {
            "condition": "talking",
            "action": "both",
            "resume": True,
            "condition_talking": True,
            "action_pause": True,
            "action_mute": True,
            "action_resume": True,
            "is_interrupted": False,
            "active_call_devices": [],
            "paused_players": [],
            "muted_sinks": [],
        }

        with patch("pengulink.cli.main.run_ipc_call", return_value=status_data):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "pausemusic", "status", "--json"])
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert data["condition"] == "talking"
            assert data["action"] == "both"
            assert data["is_interrupted"] is False

    def test_cli_pausemusic_configure_flags_and_choices(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        captured: dict[str, Any] = {}

        def mock_call(ctx: Any, method: str, params: dict[str, Any]) -> Any:
            captured["method"] = method
            captured["params"] = params
            return {
                "condition": params.get("condition", "talking"),
                "action": params.get("action", "both"),
                "action_pause": params.get("action_pause", True),
                "action_mute": params.get("action_mute", True),
                "action_resume": params.get("action_resume", True),
            }

        with patch("pengulink.cli.main.run_ipc_call", side_effect=mock_call):
            res = cli_runner.invoke(
                cli,
                [
                    "-s", str(mock_socket),
                    "pausemusic", "configure",
                    "--condition-talking", "talking",
                    "--action-pause",
                    "--action-mute",
                    "--no-action-resume",
                ],
            )
            assert res.exit_code == 0
            assert captured["params"]["condition"] == "talking"
            assert captured["params"]["action_pause"] is True
            assert captured["params"]["action_mute"] is True
            assert captured["params"]["action_resume"] is False
            assert "Updated PauseMusic configuration" in res.output

    def test_cli_pausemusic_configure_json_output(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        updated_config = {
            "condition": "talking",
            "action": "mute",
            "resume": False,
            "action_pause": False,
            "action_mute": True,
            "action_resume": False,
        }

        with patch("pengulink.cli.main.run_ipc_call", return_value=updated_config):
            res = cli_runner.invoke(
                cli,
                [
                    "-s", str(mock_socket),
                    "pausemusic", "configure",
                    "--condition", "talking",
                    "--action", "mute",
                    "--no-resume",
                    "--json",
                ],
            )
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert data["condition"] == "talking"
            assert data["action"] == "mute"
            assert data["resume"] is False


# =====================================================================
# 7. CLI Error Handling Tests
# =====================================================================

class TestTelephonyCLIErrors:
    """Test error conditions, offline daemon, IPC exceptions, and invalid inputs."""

    def test_cli_daemon_unreachable_error(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        missing_socket = tmp_path / "nonexistent.sock"

        with patch("pengulink.cli.main.PenguLinkIPCClient.call_sync", side_effect=DaemonNotRunningError("Daemon offline")):
            res = cli_runner.invoke(cli, ["-s", str(missing_socket), "telephony", "calls"])
            assert res.exit_code == 1
            assert "PenguLink daemon is not running" in res.output

    def test_cli_telephony_ipc_error_32017(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        with patch("pengulink.cli.main.run_ipc_call", side_effect=SystemExit(1)):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "telephony", "reject-call", "offline_phone"])
            assert res.exit_code == 1

    def test_cli_pausemusic_ipc_error_32018(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        # Invoking configure with no options raises UsageError (exit code 2)
        res = cli_runner.invoke(cli, ["-s", str(mock_socket), "pausemusic", "configure"])
        assert res.exit_code != 0
        assert "At least one configuration option must be specified" in res.output

        # Invoking configure with invalid condition choice raises UsageError
        res_invalid = cli_runner.invoke(cli, ["-s", str(mock_socket), "pausemusic", "configure", "--condition", "invalid_opt"])
        assert res_invalid.exit_code != 0

    def test_cli_daemon_unreachable_error_json(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Verify that CLI outputs valid structured JSON when --json is provided and daemon is offline."""
        missing_socket = tmp_path / "nonexistent.sock"

        with patch("pengulink.cli.main.PenguLinkIPCClient.call_sync", side_effect=DaemonNotRunningError("Daemon offline")):
            # Global --json flag
            res = cli_runner.invoke(cli, ["-s", str(missing_socket), "--json", "telephony", "calls"])
            assert res.exit_code == 1
            data = json.loads(res.output)
            assert data["success"] is False
            assert data["code"] == -1
            assert "Daemon offline" in data["error"]

            # Subcommand --json flag
            res2 = cli_runner.invoke(cli, ["-s", str(missing_socket), "telephony", "calls", "--json"])
            assert res2.exit_code == 1
            data2 = json.loads(res2.output)
            assert data2["success"] is False
            assert data2["code"] == -1
            assert "Daemon offline" in data2["error"]

    def test_cli_telephony_ipc_error_json(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Verify that CLI outputs structured JSON with code when IPCError occurs."""
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        exc = IPCError("IPC Error [-32017]: Hardware ringer mute failed", context={"code": -32017})
        exc.code = -32017

        with patch("pengulink.cli.main.PenguLinkIPCClient.call_sync", side_effect=exc):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "telephony", "mute-ringer", "dev1", "--json"])
            assert res.exit_code == 1
            data = json.loads(res.output)
            assert data["success"] is False
            assert data["code"] == -32017
            assert "Hardware ringer mute failed" in data["error"]

    def test_cli_unexpected_error_json(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Verify that unexpected exceptions in run_ipc_call output structured JSON under --json."""
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        with patch("pengulink.cli.main.PenguLinkIPCClient.call_sync", side_effect=RuntimeError("Kernel socket failure")):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "--json", "telephony", "calls"])
            assert res.exit_code == 1
            data = json.loads(res.output)
            assert data["success"] is False
            assert data["code"] == -1
            assert "Kernel socket failure" in data["error"]

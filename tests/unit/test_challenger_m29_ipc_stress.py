"""tests.unit.test_challenger_m29_ipc_stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Challenger stress test suite for Milestone 29:
- Malformed JSON-RPC requests to telephony_* and pausemusic_*.
- Non-existent device IDs for telephony_mute_ringer and telephony_reject_call (-32002, -32001).
- Invalid configuration values to pausemusic_set_config / pausemusic_configure (-32018).
- Calling IPC when daemon is stopped or socket is missing (DaemonNotRunningError / clean exit).
- Empty call lists, unknown call IDs, boundary conditions, and concurrency stress bursts.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.config import PenguLinkConfig
from pengulink.core.exceptions import (
    CallNotFoundError,
    DaemonNotRunningError,
    DeviceNotFoundError,
    InvalidCallStateError,
    IPCError,
    PauseMusicConfigError,
    PauseMusicError,
    TelephonyError,
)
from pengulink.daemon.ipc import (
    DEVICE_NOT_CONNECTED,
    DEVICE_NOT_FOUND,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PAUSE_MUSIC_ERROR,
    STREAM_LIMIT_BYTES,
    TELEPHONY_ERROR,
    PenguLinkIPCClient,
    PenguLinkIPCServer,
)
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.services.telephony import (
    ActiveCall,
    CallItem,
    CallState,
    TelephonyService,
)

# Port counter to prevent port collisions between test daemons
_PORT_BASE = 28500


def _allocate_ports() -> tuple[int, int, int]:
    global _PORT_BASE
    _PORT_BASE += 10
    return (_PORT_BASE, _PORT_BASE + 1, _PORT_BASE + 5)


def _create_test_config(tmp_path: Path, test_id: str) -> PenguLinkConfig:
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / f"cfg_{test_id}"
    cfg.storage.data_dir = tmp_path / f"data_{test_id}"
    cfg.storage.runtime_dir = tmp_path / f"run_{test_id}"
    cfg.device.device_id = f"testdev_{test_id[:12]}"
    cfg.device.device_name = f"Challenger-M29-{test_id}"
    udp, tcp_min, tcp_max = _allocate_ports()
    cfg.network.udp_port = udp
    cfg.network.tcp_port_min = tcp_min
    cfg.network.tcp_port_max = tcp_max
    cfg.ensure_directories()
    return cfg


async def send_raw_ipc(
    socket_path: Path,
    raw_data: bytes | str,
    read_response: bool = True,
    timeout: float = 5.0,
) -> dict[str, Any] | None:
    """Helper to transmit raw byte frames directly over the Unix socket."""
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(socket_path), limit=STREAM_LIMIT_BYTES),
        timeout=timeout,
    )
    try:
        data_bytes = raw_data.encode("utf-8") if isinstance(raw_data, str) else raw_data
        if not data_bytes.endswith(b"\n"):
            data_bytes += b"\n"
        writer.write(data_bytes)
        await asyncio.wait_for(writer.drain(), timeout=timeout)
        if not read_response:
            await asyncio.sleep(0.05)
            return None
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not line:
            return None
        return json.loads(line.decode("utf-8"))
    finally:
        writer.close()
        with suppress(OSError, asyncio.TimeoutError):
            await asyncio.wait_for(writer.wait_closed(), timeout=1.0)


@asynccontextmanager
async def _running_daemon_ctx(
    tmp_path: Path,
    test_id: str,
) -> AsyncIterator[tuple[PenguLinkDaemon, Path]]:
    cfg = _create_test_config(tmp_path, test_id)
    socket_path = tmp_path / f"daemon_{test_id}.sock"
    daemon = PenguLinkDaemon(
        config=cfg,
        socket_path=socket_path,
        config_dir=tmp_path / f"cfg_{test_id}",
    )
    await daemon.start()
    try:
        yield daemon, socket_path
    finally:
        await daemon.stop()


# =====================================================================
# 1. Malformed JSON-RPC Requests Tests
# =====================================================================


class TestMalformedIPCRequests:
    """Empirical challenge: Malformed JSON-RPC requests to telephony_* and pausemusic_*."""

    async def test_malformed_json_syntax_returns_parse_error(self, tmp_path: Path) -> None:
        """Malformed JSON payloads must return standard PARSE_ERROR (-32700)."""
        async with _running_daemon_ctx(tmp_path, "fuzz_json") as (_, socket_path):
            malformed_payloads = [
                b"{bad json syntax",
                b'{"jsonrpc": "2.0", "method": "telephony_get_calls", "params": ',
                b"\xff\xfe\x00\x01\x80\x99",
                b"telephony_get_calls\n",
                b'{"jsonrpc": "2.0", "method": "pausemusic_get_config",,}',
            ]
            for payload in malformed_payloads:
                resp = await send_raw_ipc(socket_path, payload)
                assert resp is not None, f"Expected error response for {payload!r}"
                assert resp.get("jsonrpc") == "2.0"
                err = resp.get("error", {})
                assert err.get("code") == PARSE_ERROR
                assert "parse error" in err.get("message", "").lower()

    async def test_non_dict_root_returns_invalid_request(self, tmp_path: Path) -> None:
        """Non-dict root structures must return INVALID_REQUEST (-32600)."""
        async with _running_daemon_ctx(tmp_path, "fuzz_nondict") as (_, socket_path):
            bad_roots = [
                json.dumps(["telephony_get_calls"]),
                json.dumps("pausemusic_get_config"),
                json.dumps(12345),
                json.dumps(True),
            ]
            for bad_payload in bad_roots:
                resp = await send_raw_ipc(socket_path, bad_payload)
                assert resp is not None
                assert resp.get("jsonrpc") == "2.0"
                err = resp.get("error", {})
                assert err.get("code") == INVALID_REQUEST
                assert "invalid json-rpc" in err.get("message", "").lower()

    async def test_invalid_jsonrpc_version_returns_invalid_request(self, tmp_path: Path) -> None:
        """Missing or wrong jsonrpc version must return INVALID_REQUEST (-32600)."""
        async with _running_daemon_ctx(tmp_path, "fuzz_version") as (_, socket_path):
            bad_versions = [
                {"jsonrpc": "1.0", "method": "telephony_get_calls", "id": 1},
                {"method": "telephony_get_calls", "id": 2},
                {"jsonrpc": 2.0, "method": "pausemusic_get_config", "id": 3},
                {"jsonrpc": "", "method": "pausemusic_get_config", "id": 4},
            ]
            for bad_req in bad_versions:
                resp = await send_raw_ipc(socket_path, json.dumps(bad_req))
                assert resp is not None
                assert resp.get("error", {}).get("code") == INVALID_REQUEST

    async def test_invalid_or_missing_method_returns_invalid_request(self, tmp_path: Path) -> None:
        """Missing or non-string method names must return INVALID_REQUEST (-32600)."""
        async with _running_daemon_ctx(tmp_path, "fuzz_method") as (_, socket_path):
            bad_methods = [
                {"jsonrpc": "2.0", "id": 1},
                {"jsonrpc": "2.0", "method": 12345, "id": 2},
                {"jsonrpc": "2.0", "method": "", "id": 3},
                {"jsonrpc": "2.0", "method": None, "id": 4},
                {"jsonrpc": "2.0", "method": [], "id": 5},
            ]
            for bad_req in bad_methods:
                resp = await send_raw_ipc(socket_path, json.dumps(bad_req))
                assert resp is not None
                assert resp.get("error", {}).get("code") == INVALID_REQUEST

    async def test_non_dict_params_returns_invalid_params(self, tmp_path: Path) -> None:
        """Non-dict params (string, integer, list, bool) must return INVALID_PARAMS (-32602)."""
        async with _running_daemon_ctx(tmp_path, "fuzz_params") as (_, socket_path):
            bad_params_list = [
                {"jsonrpc": "2.0", "method": "telephony_get_calls", "params": "dev1", "id": 1},
                {"jsonrpc": "2.0", "method": "telephony_mute_ringer", "params": 42, "id": 2},
                {"jsonrpc": "2.0", "method": "pausemusic_get_config", "params": [1, 2], "id": 3},
                {"jsonrpc": "2.0", "method": "pausemusic_set_config", "params": True, "id": 4},
            ]
            for bad_req in bad_params_list:
                resp = await send_raw_ipc(socket_path, json.dumps(bad_req))
                assert resp is not None
                assert resp.get("error", {}).get("code") == INVALID_PARAMS
                assert "params must be an object" in resp.get("error", {}).get("message", "").lower()

    async def test_telephony_invalid_param_types_return_invalid_params(self, tmp_path: Path) -> None:
        """Type errors in telephony parameters must return INVALID_PARAMS (-32602)."""
        async with _running_daemon_ctx(tmp_path, "tel_bad_params") as (_, socket_path):
            # telephony_get_calls with non-string device_id
            resp = await send_raw_ipc(
                socket_path,
                json.dumps({
                    "jsonrpc": "2.0",
                    "method": "telephony_get_calls",
                    "params": {"device_id": 12345},
                    "id": 10,
                }),
            )
            assert resp is not None
            assert resp.get("error", {}).get("code") == INVALID_PARAMS
            assert "device_id" in resp.get("error", {}).get("message", "").lower()

            # telephony_mute_ringer with empty device_id
            resp = await send_raw_ipc(
                socket_path,
                json.dumps({
                    "jsonrpc": "2.0",
                    "method": "telephony_mute_ringer",
                    "params": {"device_id": ""},
                    "id": 11,
                }),
            )
            assert resp is not None
            assert resp.get("error", {}).get("code") == INVALID_PARAMS

            # telephony_mute_ringer missing device_id
            resp = await send_raw_ipc(
                socket_path,
                json.dumps({
                    "jsonrpc": "2.0",
                    "method": "telephony_mute_ringer",
                    "params": {},
                    "id": 12,
                }),
            )
            assert resp is not None
            assert resp.get("error", {}).get("code") == INVALID_PARAMS

            # telephony_reject_call with whitespace-only device_id
            resp = await send_raw_ipc(
                socket_path,
                json.dumps({
                    "jsonrpc": "2.0",
                    "method": "telephony_reject_call",
                    "params": {"device_id": "   "},
                    "id": 13,
                }),
            )
            assert resp is not None
            assert resp.get("error", {}).get("code") == INVALID_PARAMS

    async def test_unknown_method_returns_method_not_found(self, tmp_path: Path) -> None:
        """Unknown methods under telephony_* or pausemusic_* must return METHOD_NOT_FOUND (-32601)."""
        async with _running_daemon_ctx(tmp_path, "unknown_method") as (_, socket_path):
            unknowns = [
                "telephony_unknown_action",
                "pausemusic_restart_engine",
                "telephony_hangup_all",
            ]
            for m in unknowns:
                resp = await send_raw_ipc(
                    socket_path,
                    json.dumps({
                        "jsonrpc": "2.0",
                        "method": m,
                        "params": {},
                        "id": 99,
                    }),
                )
                assert resp is not None
                assert resp.get("error", {}).get("code") == METHOD_NOT_FOUND


# =====================================================================
# 2. Non-existent Device IDs Handling Tests
# =====================================================================


class TestNonExistentDeviceHandling:
    """Empirical challenge: Non-existent device IDs for telephony_mute_ringer and telephony_reject_call."""

    async def test_mute_ringer_nonexistent_device_returns_device_not_connected_32002(
        self, tmp_path: Path
    ) -> None:
        """Calling telephony_mute_ringer for offline/non-existent device returns -32002."""
        async with _running_daemon_ctx(tmp_path, "mute_ghost") as (_, socket_path):
            # Raw IPC test
            resp = await send_raw_ipc(
                socket_path,
                json.dumps({
                    "jsonrpc": "2.0",
                    "method": "telephony_mute_ringer",
                    "params": {"device_id": "ghost_phone_999"},
                    "id": 20,
                }),
            )
            assert resp is not None
            err = resp.get("error", {})
            assert err.get("code") == DEVICE_NOT_CONNECTED
            assert "ghost_phone_999" in err.get("message", "")
            assert "not connected" in err.get("message", "").lower()

            # Client test
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc_info:
                await client.telephony_mute_ringer("ghost_phone_999")
            assert exc_info.value.code == DEVICE_NOT_CONNECTED

    async def test_reject_call_nonexistent_device_returns_device_not_connected_32002(
        self, tmp_path: Path
    ) -> None:
        """Calling telephony_reject_call for offline/non-existent device returns -32002."""
        async with _running_daemon_ctx(tmp_path, "reject_ghost") as (_, socket_path):
            # Raw IPC test
            resp = await send_raw_ipc(
                socket_path,
                json.dumps({
                    "jsonrpc": "2.0",
                    "method": "telephony_reject_call",
                    "params": {"device_id": "ghost_phone_888", "phone_number": "+15550000"},
                    "id": 21,
                }),
            )
            assert resp is not None
            err = resp.get("error", {})
            assert err.get("code") == DEVICE_NOT_CONNECTED
            assert "ghost_phone_888" in err.get("message", "")
            assert "not connected" in err.get("message", "").lower()

            # Client test
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc_info:
                await client.telephony_reject_call("ghost_phone_888")
            assert exc_info.value.code == DEVICE_NOT_CONNECTED

    async def test_device_not_found_exception_mapping_32001(self, tmp_path: Path) -> None:
        """When DeviceNotFoundError is raised, server must map to DEVICE_NOT_FOUND (-32001)."""
        socket_path = tmp_path / "mock_dnf.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            raise DeviceNotFoundError(f"Device '{params.get('device_id')}' was never paired")

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()
        try:
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc_info:
                await client.telephony_mute_ringer("unpaired_device_1")
            assert exc_info.value.code == DEVICE_NOT_FOUND
            assert "was never paired" in str(exc_info.value)
        finally:
            await server.stop()

    async def test_telephony_domain_exception_mapping_32017(self, tmp_path: Path) -> None:
        """Telephony domain exceptions must map to TELEPHONY_ERROR (-32017)."""
        socket_path = tmp_path / "mock_tel_err.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            if method == "telephony_mute_ringer":
                raise TelephonyError("Hardware ringer mute failed")
            elif method == "telephony_reject_call":
                raise CallNotFoundError("No active call matching ID")
            elif method == "telephony_get_calls":
                raise InvalidCallStateError("Corrupt call state table")
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()
        try:
            client = PenguLinkIPCClient(socket_path)

            with pytest.raises(IPCError) as exc1:
                await client.telephony_mute_ringer("dev1")
            assert exc1.value.code == TELEPHONY_ERROR
            assert "Hardware ringer mute failed" in str(exc1.value)

            with pytest.raises(IPCError) as exc2:
                await client.telephony_reject_call("dev1")
            assert exc2.value.code == TELEPHONY_ERROR
            assert "No active call matching ID" in str(exc2.value)

            with pytest.raises(IPCError) as exc3:
                await client.telephony_get_calls("dev1")
            assert exc3.value.code == TELEPHONY_ERROR
            assert "Corrupt call state table" in str(exc3.value)
        finally:
            await server.stop()


# =====================================================================
# 3. Invalid Configuration Values Tests (-32018)
# =====================================================================


class TestPauseMusicConfigInvalidValues:
    """Empirical challenge: Invalid configuration values to pausemusic_set_config (-32018)."""

    async def test_invalid_condition_returns_pause_music_error_32018(
        self, tmp_path: Path
    ) -> None:
        """Invalid condition strings must return PAUSE_MUSIC_ERROR (-32018)."""
        async with _running_daemon_ctx(tmp_path, "pm_bad_cond") as (_, socket_path):
            invalid_conditions = ["exploding", "never", "unknown_cond", "12345"]
            for bad_c in invalid_conditions:
                resp = await send_raw_ipc(
                    socket_path,
                    json.dumps({
                        "jsonrpc": "2.0",
                        "method": "pausemusic_set_config",
                        "params": {"condition": bad_c},
                        "id": 30,
                    }),
                )
                assert resp is not None
                err = resp.get("error", {})
                assert err.get("code") == PAUSE_MUSIC_ERROR
                assert "must be 'ringing' or 'talking'" in err.get("message", "")

            # Verify client helper raises IPCError with -32018
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc_info:
                await client.pausemusic_set_config(condition="invalid_foo")
            assert exc_info.value.code == PAUSE_MUSIC_ERROR
            assert "must be 'ringing' or 'talking'" in str(exc_info.value)

    async def test_invalid_action_returns_pause_music_error_32018(
        self, tmp_path: Path
    ) -> None:
        """Invalid action strings must return PAUSE_MUSIC_ERROR (-32018)."""
        async with _running_daemon_ctx(tmp_path, "pm_bad_act") as (_, socket_path):
            invalid_actions = ["destroy", "unmute", "pause_all", "random_action"]
            for bad_a in invalid_actions:
                resp = await send_raw_ipc(
                    socket_path,
                    json.dumps({
                        "jsonrpc": "2.0",
                        "method": "pausemusic_set_config",
                        "params": {"action": bad_a},
                        "id": 31,
                    }),
                )
                assert resp is not None
                err = resp.get("error", {})
                assert err.get("code") == PAUSE_MUSIC_ERROR
                assert "must be 'pause', 'mute', 'both', or 'none'" in err.get("message", "")

            # Verify client helper raises IPCError with -32018
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc_info:
                await client.pausemusic_set_config(action="bad_act_bar")
            assert exc_info.value.code == PAUSE_MUSIC_ERROR
            assert "must be 'pause', 'mute', 'both', or 'none'" in str(exc_info.value)

    async def test_direct_daemon_pausemusic_set_config_validation(
        self, tmp_path: Path
    ) -> None:
        """Direct call to daemon.pausemusic_set_config raises PauseMusicConfigError."""
        daemon = PenguLinkDaemon(config_dir=tmp_path)
        with pytest.raises(PauseMusicConfigError):
            await daemon.pausemusic_set_config(condition="invalid_choice")
        with pytest.raises(PauseMusicConfigError):
            await daemon.pausemusic_set_config(action="invalid_action")

    async def test_pause_music_domain_exception_mapping_32018(self, tmp_path: Path) -> None:
        """PauseMusicError and PauseMusicConfigError must map to PAUSE_MUSIC_ERROR (-32018)."""
        socket_path = tmp_path / "mock_pm_err.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            if method == "pausemusic_get_config":
                raise PauseMusicError("Failed to query MPRIS player state")
            raise PauseMusicConfigError("Configuration syntax error")

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()
        try:
            client = PenguLinkIPCClient(socket_path)

            with pytest.raises(IPCError) as exc1:
                await client.pausemusic_get_config()
            assert exc1.value.code == PAUSE_MUSIC_ERROR
            assert "Failed to query MPRIS player state" in str(exc1.value)

            with pytest.raises(IPCError) as exc2:
                await client.pausemusic_set_config(condition="bad")
            assert exc2.value.code == PAUSE_MUSIC_ERROR
            assert "Configuration syntax error" in str(exc2.value)
        finally:
            await server.stop()


# =====================================================================
# 4. Daemon Stopped / Missing Socket Tests
# =====================================================================


class TestDaemonStoppedAndMissingSocket:
    """Empirical challenge: Calling IPC when daemon is stopped or socket is missing."""

    async def test_client_missing_socket_raises_daemon_not_running(self, tmp_path: Path) -> None:
        """Calling IPCClient when socket file does not exist raises DaemonNotRunningError."""
        missing_socket = tmp_path / "non_existent_daemon.sock"
        client = PenguLinkIPCClient(missing_socket)

        with pytest.raises(DaemonNotRunningError) as exc:
            await client.telephony_get_calls()
        assert "socket not found" in str(exc.value)

        with pytest.raises(DaemonNotRunningError):
            await client.telephony_mute_ringer("dev1")

        with pytest.raises(DaemonNotRunningError):
            await client.telephony_reject_call("dev1")

        with pytest.raises(DaemonNotRunningError):
            await client.pausemusic_get_config()

        with pytest.raises(DaemonNotRunningError):
            await client.pausemusic_set_config(condition="talking")

    async def test_client_stale_dead_socket_raises_daemon_not_running(self, tmp_path: Path) -> None:
        """Calling IPCClient when socket file exists on disk but daemon is stopped raises DaemonNotRunningError."""
        stale_socket = tmp_path / "stale_daemon.sock"
        # Bind and close to leave a dead socket file on filesystem
        dead_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        dead_sock.bind(str(stale_socket))
        dead_sock.close()

        assert stale_socket.exists()
        client = PenguLinkIPCClient(stale_socket)

        with pytest.raises(DaemonNotRunningError) as exc:
            await client.telephony_get_calls()
        assert "not running or not responding" in str(exc.value)

        with pytest.raises(DaemonNotRunningError):
            await client.pausemusic_get_config()

    def test_cli_clean_exit_when_daemon_stopped(self, tmp_path: Path) -> None:
        """CLI commands must print a friendly message and cleanly exit with code 1."""
        missing_socket = tmp_path / "missing_cli.sock"
        runner = CliRunner()

        cli_commands = [
            ["--socket-path", str(missing_socket), "telephony", "calls"],
            ["--socket-path", str(missing_socket), "telephony", "mute-ringer", "dev1"],
            ["--socket-path", str(missing_socket), "telephony", "reject-call", "dev1"],
            ["--socket-path", str(missing_socket), "pausemusic", "status"],
            ["--socket-path", str(missing_socket), "pausemusic", "configure", "--condition", "ringing"],
        ]

        for cmd in cli_commands:
            res = runner.invoke(cli, cmd)
            assert res.exit_code == 1
            assert "PenguLink daemon is not running" in res.output


# =====================================================================
# 5. Empty Call Lists, Unknown Call IDs & Boundary Conditions
# =====================================================================


class TestEmptyCallListsAndBoundaryConditions:
    """Empirical challenge: Empty call lists, unknown call IDs, boundary values, and concurrency."""

    async def test_empty_call_lists_across_daemon_and_ipc(self, tmp_path: Path) -> None:
        """telephony_get_calls returns clean empty lists on initial state and unknown devices."""
        async with _running_daemon_ctx(tmp_path, "empty_calls") as (daemon, socket_path):
            client = PenguLinkIPCClient(socket_path)

            # 1. Direct daemon empty calls
            assert await daemon.telephony_get_calls() == []
            assert await daemon.telephony_get_calls("nonexistent_device") == []

            # 2. Client over IPC empty calls
            res_all = await client.telephony_get_calls()
            assert res_all == []
            assert isinstance(res_all, list)

            res_filtered = await client.telephony_get_calls("nonexistent_device")
            assert res_filtered == []
            assert isinstance(res_filtered, list)

            # 3. Raw IPC query
            raw_res = await send_raw_ipc(
                socket_path,
                json.dumps({
                    "jsonrpc": "2.0",
                    "method": "telephony_get_calls",
                    "params": {},
                    "id": 40,
                }),
            )
            assert raw_res is not None
            assert raw_res.get("result") == []

    async def test_boundary_call_item_fields_and_formatting(self) -> None:
        """ActiveCall and CallItem boundary conditions: unicode, huge names, zero duration."""
        huge_name = "Alice " * 1000
        unicode_phone = "+1 (800) 555-0199 📱 🇯🇵"
        emoji_contact = "👨‍💻 Engineer Bob 🚀"

        now = time.time()
        call = ActiveCall(
            device_id="dev_extreme",
            phone_number=unicode_phone,
            contact_name=emoji_contact,
            state=CallState.RINGING,
            start_time=now + 10.0,  # future timestamp boundary
        )
        assert call.duration == 0.0  # max(0.0, time.time() - start_time)
        assert call.is_active is True

        call_dict = call.to_dict()
        assert call_dict["phone_number"] == unicode_phone
        assert call_dict["contact_name"] == emoji_contact
        assert call_dict["duration"] == 0.0

        item = CallItem(call_dict)
        assert item.contact_name == emoji_contact
        assert item["phone_number"] == unicode_phone

        # Oversized contact name
        item_huge = CallItem({"contact_name": huge_name, "duration": 12.0})
        assert len(item_huge.contact_name) == len(huge_name)

    async def test_unrecognized_telephony_event_handling(self) -> None:
        """TelephonyService safely ignores unrecognized wire event types without crash."""
        service = TelephonyService()
        result = await service.process_call_event(
            device_id="dev1",
            event="unrecognized_telephony_signal_xyz",
            phone_number="+15551234",
            contact_name="Test",
        )
        assert result is None
        calls = service.get_active_calls()
        assert len(calls) == 0

    async def test_multi_device_call_isolation_and_lifecycle(self) -> None:
        """Call state transitions across multiple devices maintain strict isolation."""
        service = TelephonyService()

        # Device 1: Incoming ring
        await service.process_call_event("dev1", "ringing", "+1111", "Alice")
        # Device 2: Talking
        await service.process_call_event("dev2", "talking", "+2222", "Bob")

        # Isolated queries
        d1_calls = service.get_active_calls("dev1")
        assert len(d1_calls) == 1
        assert d1_calls[0].phone_number == "+1111"
        assert d1_calls[0].state == "RINGING"

        d2_calls = service.get_active_calls("dev2")
        assert len(d2_calls) == 1
        assert d2_calls[0].phone_number == "+2222"
        assert d2_calls[0].state == "TALKING"

        # All calls
        all_calls = service.get_active_calls()
        assert len(all_calls) == 2

        # Device 1 answers call -> talking
        await service.process_call_event("dev1", "talking", "+1111", "Alice")
        assert service.get_active_calls("dev1")[0].state == "TALKING"

        # Device 1 hangs up -> idle
        await service.process_call_event("dev1", "idle", "+1111", "Alice")
        assert len(service.get_active_calls("dev1")) == 0
        assert len(service.get_active_calls("dev2")) == 1

        # Clear device 2
        await service.clear_device("dev2")
        assert len(service.get_active_calls()) == 0

    async def test_concurrency_stress_burst_daemon_ipc(self, tmp_path: Path) -> None:
        """Stress test: 50 concurrent tasks querying calls and modifying pausemusic config."""
        async with _running_daemon_ctx(tmp_path, "concurrency_burst") as (daemon, socket_path):
            client = PenguLinkIPCClient(socket_path)

            # Pre-populate calls
            await daemon.telephony_service.process_call_event("dev_a", "ringing", "+1001", "User A")
            await daemon.telephony_service.process_call_event("dev_b", "talking", "+1002", "User B")

            async def query_worker(i: int) -> bool:
                calls = await client.telephony_get_calls()
                assert len(calls) == 2
                cfg = await client.pausemusic_get_config()
                assert "condition" in cfg
                return True

            async def config_worker(i: int) -> bool:
                cond = "talking" if i % 2 == 0 else "ringing"
                act = "both" if i % 2 == 0 else "pause"
                res = await client.pausemusic_set_config(condition=cond, action=act)
                assert res["condition"] == cond
                assert res["action"] == act
                return True

            tasks = []
            for i in range(25):
                tasks.append(asyncio.create_task(query_worker(i)))
                tasks.append(asyncio.create_task(config_worker(i)))

            results = await asyncio.gather(*tasks)
            assert all(results)
            assert len(results) == 50

            # Final sanity verification
            final_calls = await client.telephony_get_calls()
            assert len(final_calls) == 2

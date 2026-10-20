"""tests.unit.test_challenger_m25_ipc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial stress-test, parameter fuzzing, error code assertions,
concurrency bursts, boundary values, and corrupted storage recovery suite
for PenguLink daemon JSON-RPC IPC endpoints:
  - systemvolume_* (SystemVolume local & remote controls, error -32015)
  - runcommand_* (RunCommand local store & process execution, error -32016)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Tuple, Union

import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.core.exceptions import (
    AudioBackendError,
    AudioUnavailableError,
    CommandExecutionError,
    CommandNotFoundError,
    ConnectionClosedError,
    DeviceNotFoundError,
    IPCError,
    RunCommandError,
    StorageCorruptedError,
    StorageError,
    StorageWriteError,
)
from pengulink.daemon.ipc import (
    AUDIO_SYSTEM_VOLUME_ERROR,
    DEVICE_NOT_CONNECTED,
    DEVICE_NOT_FOUND,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    RUN_COMMAND_ERROR,
    STREAM_LIMIT_BYTES,
    PenguLinkIPCClient,
    PenguLinkIPCServer,
)
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.services.audio import (
    AudioError,
    AudioSink,
    MockAudioSinkBackend,
)
from pengulink.services.command import CommandStore


# Port allocator to prevent network collisions between concurrent/sequential test daemons
_PORT_BASE = 17500


def _allocate_ports() -> Tuple[int, int, int]:
    global _PORT_BASE
    _PORT_BASE += 10
    return (_PORT_BASE, _PORT_BASE + 1, _PORT_BASE + 5)


async def send_raw_ipc(
    socket_path: Path,
    raw_data: Union[bytes, str],
    read_response: bool = True,
    timeout: float = 3.0,
) -> Optional[Dict[str, Any]]:
    """Helper to send raw payload bytes to IPC Unix socket and optionally receive JSON response."""
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
            # Brief yield to let server process notification
            await asyncio.sleep(0.05)
            return None
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not line:
            return None
        return json.loads(line.decode("utf-8"))
    finally:
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
        except Exception:
            pass


def _create_test_config(tmp_path: Path, test_id: str) -> PenguLinkConfig:
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / f"config_{test_id}"
    cfg.storage.data_dir = tmp_path / f"data_{test_id}"
    cfg.storage.runtime_dir = tmp_path / f"run_{test_id}"
    cfg.device.device_id = f"testdev_{test_id[:16]}"
    cfg.device.device_name = f"PenguLink-Challenger-{test_id}"
    udp, tcp_min, tcp_max = _allocate_ports()
    cfg.network.udp_port = udp
    cfg.network.tcp_port_min = tcp_min
    cfg.network.tcp_port_max = tcp_max
    cfg.ensure_directories()
    return cfg


def _create_mock_backend() -> MockAudioSinkBackend:
    return MockAudioSinkBackend(
        initial_sinks=[
            AudioSink(
                name="alsa_output.pci-0000.analog-stereo",
                description="Built-in Stereo",
                volume=32768,
                max_volume=65536,
                muted=False,
                enabled=True,
            ),
            AudioSink(
                name="bluez_output.headphones",
                description="Wireless Headphones",
                volume=45000,
                max_volume=65536,
                muted=True,
                enabled=False,
            ),
        ]
    )


@asynccontextmanager
async def _running_daemon_ctx(
    tmp_path: Path,
    test_id: str,
    cmd_store: Optional[CommandStore] = None,
    backend: Optional[MockAudioSinkBackend] = None,
) -> AsyncIterator[Tuple[PenguLinkDaemon, Path]]:
    cfg = _create_test_config(tmp_path, test_id)
    socket_path = tmp_path / f"daemon_{test_id}.sock"
    if cmd_store is None:
        cmd_store = CommandStore(storage_path=tmp_path / f"commands_{test_id}.json")
    if backend is None:
        backend = _create_mock_backend()
    daemon = PenguLinkDaemon(
        config=cfg,
        socket_path=socket_path,
        audio_backend=backend,
        command_store=cmd_store,
    )
    await daemon.start()
    try:
        yield daemon, socket_path
    finally:
        await daemon.stop()


# =====================================================================
# 1. Parameter Fuzzing & Malformed Frames Tests
# =====================================================================


class TestIPCParameterFuzzing:
    """Adversarial parameter fuzzing and malformed frame handling."""

    async def test_malformed_json_bytes_returns_parse_error(self, tmp_path: Path) -> None:
        """Malformed JSON input must return standard PARSE_ERROR (-32700)."""
        async with _running_daemon_ctx(tmp_path, "fuzz_malformed") as (_, socket_path):
            malformed_inputs = [
                b"{bad json syntax",
                b'{"jsonrpc": "2.0", "method": ',
                b"\xff\xfe\x00\x01\x80\x99",
                b"Hello World not json\n",
            ]
            for bad_raw in malformed_inputs:
                resp = await send_raw_ipc(socket_path, bad_raw)
                assert resp is not None, f"Expected error response for {bad_raw!r}"
                assert resp.get("jsonrpc") == "2.0"
                err = resp.get("error", {})
                assert err.get("code") == PARSE_ERROR
                assert "parse error" in err.get("message", "").lower()

    async def test_non_dict_root_returns_invalid_request(self, tmp_path: Path) -> None:
        """Non-dict root types must return INVALID_REQUEST (-32600)."""
        async with _running_daemon_ctx(tmp_path, "fuzz_nondict") as (_, socket_path):
            bad_roots = [
                json.dumps([1, 2, 3]),
                json.dumps("just a string"),
                json.dumps(12345),
                json.dumps(True),
            ]
            for bad_payload in bad_roots:
                resp = await send_raw_ipc(socket_path, bad_payload)
                assert resp is not None
                err = resp.get("error", {})
                assert err.get("code") == INVALID_REQUEST

    async def test_invalid_jsonrpc_version_returns_invalid_request(self, tmp_path: Path) -> None:
        """Missing or wrong jsonrpc version returns INVALID_REQUEST (-32600)."""
        async with _running_daemon_ctx(tmp_path, "fuzz_ver") as (_, socket_path):
            cases = [
                {"method": "runcommand_list", "id": 1},  # missing jsonrpc
                {"jsonrpc": "1.0", "method": "runcommand_list", "id": 1},
                {"jsonrpc": 2.0, "method": "runcommand_list", "id": 1},
                {"jsonrpc": "3.0", "method": "systemvolume_get_sinks", "id": 2},
            ]
            for c in cases:
                resp = await send_raw_ipc(socket_path, json.dumps(c))
                assert resp is not None
                assert resp.get("error", {}).get("code") == INVALID_REQUEST

    async def test_invalid_method_types_returns_invalid_request(self, tmp_path: Path) -> None:
        """Non-string or empty method returns INVALID_REQUEST (-32600)."""
        async with _running_daemon_ctx(tmp_path, "fuzz_method") as (_, socket_path):
            bad_methods = [None, 123, [], {}, True, ""]
            for bm in bad_methods:
                req = {"jsonrpc": "2.0", "method": bm, "id": 1}
                resp = await send_raw_ipc(socket_path, json.dumps(req))
                assert resp is not None
                assert resp.get("error", {}).get("code") == INVALID_REQUEST

    async def test_invalid_params_type_returns_invalid_params(self, tmp_path: Path) -> None:
        """Non-dict params returns INVALID_PARAMS (-32602)."""
        async with _running_daemon_ctx(tmp_path, "fuzz_params") as (_, socket_path):
            bad_params = ["not a dict", [1, 2], 42, True]
            for bp in bad_params:
                req = {
                    "jsonrpc": "2.0",
                    "method": "systemvolume_get_sinks",
                    "params": bp,
                    "id": 1,
                }
                resp = await send_raw_ipc(socket_path, json.dumps(req))
                assert resp is not None
                assert resp.get("error", {}).get("code") == INVALID_PARAMS

    async def test_missing_required_params_systemvolume(self, tmp_path: Path) -> None:
        """Missing required fields in systemvolume_* calls must return INVALID_PARAMS (-32602)."""
        async with _running_daemon_ctx(tmp_path, "fuzz_missing_sysvol") as (_, socket_path):
            cases = [
                ("systemvolume_set_volume", {}),
                ("systemvolume_set_volume", {"name": "sink1"}),
                ("systemvolume_set_volume", {"volume": 50}),
                ("systemvolume_set_muted", {}),
                ("systemvolume_set_muted", {"name": "sink1"}),
                ("systemvolume_set_default", {}),
                ("systemvolume_get_remote_sinks", {}),
                ("systemvolume_set_remote_volume", {}),
                ("systemvolume_set_remote_volume", {"device_id": "d1"}),
                ("systemvolume_set_remote_volume", {"device_id": "d1", "name": "s1"}),
                ("systemvolume_set_remote_muted", {"device_id": "d1"}),
                ("systemvolume_set_remote_default", {"device_id": "d1"}),
                ("systemvolume_request_remote_sinks", {}),
            ]
            for method, params in cases:
                req = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
                resp = await send_raw_ipc(socket_path, json.dumps(req))
                assert resp is not None, f"Expected response for {method} with {params}"
                err = resp.get("error", {})
                assert err.get("code") == INVALID_PARAMS, (
                    f"Method {method} returned {err} instead of INVALID_PARAMS (-32602)"
                )

    async def test_missing_required_params_runcommand(self, tmp_path: Path) -> None:
        """Missing required fields in runcommand_* calls must return INVALID_PARAMS (-32602)."""
        async with _running_daemon_ctx(tmp_path, "fuzz_missing_cmd") as (_, socket_path):
            cases = [
                ("runcommand_get", {}),
                ("runcommand_add", {}),
                ("runcommand_add", {"name": "Test"}),
                ("runcommand_add", {"command": "echo 1"}),
                ("runcommand_remove", {}),
                ("runcommand_exec", {}),
            ]
            for method, params in cases:
                req = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
                resp = await send_raw_ipc(socket_path, json.dumps(req))
                assert resp is not None
                err = resp.get("error", {})
                assert err.get("code") == INVALID_PARAMS, (
                    f"Method {method} returned {err} instead of INVALID_PARAMS (-32602)"
                )

    async def test_unexpected_param_types_return_invalid_params(self, tmp_path: Path) -> None:
        """Unexpected parameter types (arrays/dicts for string/int) return INVALID_PARAMS (-32602)."""
        async with _running_daemon_ctx(tmp_path, "fuzz_types") as (_, socket_path):
            cases = [
                (
                    "systemvolume_set_volume",
                    {"name": "sink1", "volume": "not_an_int"},
                ),
                (
                    "systemvolume_set_volume",
                    {"name": "sink1", "volume": [50]},
                ),
                (
                    "systemvolume_set_volume",
                    {"name": "sink1", "volume": {"val": 50}},
                ),
                ("runcommand_get", {"key": ["k1"]}),
                ("runcommand_get", {"key": {"key": "k1"}}),
                ("runcommand_add", {"name": 12345, "command": "echo 1"}),
                ("runcommand_add", {"name": "Test", "command": ["echo", "1"]}),
                ("runcommand_exec", {"key": [1, 2]}),
                ("systemvolume_get_remote_sinks", {"device_id": 9999}),
                ("systemvolume_get_remote_sinks", {"device_id": {"bad": "type"}}),
                ("systemvolume_set_remote_volume", {"device_id": [1], "name": "s1", "volume": 50}),
                ("systemvolume_set_remote_volume", {"device_id": "d1", "name": "s1", "volume": "bad"}),
                ("systemvolume_set_remote_volume", {"device_id": "d1", "name": "s1", "volume": [50]}),
            ]
            for method, params in cases:
                req = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
                resp = await send_raw_ipc(socket_path, json.dumps(req))
                assert resp is not None
                err = resp.get("error", {})
                assert err.get("code") == INVALID_PARAMS, (
                    f"Method {method} returned {err} instead of INVALID_PARAMS (-32602)"
                )

    async def test_empty_string_parameters_return_invalid_params(self, tmp_path: Path) -> None:
        """Empty string parameters where identifier/command expected return INVALID_PARAMS (-32602)."""
        async with _running_daemon_ctx(tmp_path, "fuzz_empty_str") as (_, socket_path):
            cases = [
                ("systemvolume_set_volume", {"name": "", "volume": 50}),
                ("systemvolume_set_muted", {"name": "", "muted": True}),
                ("systemvolume_set_default", {"name": ""}),
                ("systemvolume_get_remote_sinks", {"device_id": ""}),
                ("runcommand_get", {"key": ""}),
                ("runcommand_add", {"name": "", "command": "ls"}),
                ("runcommand_add", {"name": "ls", "command": ""}),
                ("runcommand_remove", {"key": ""}),
                ("runcommand_exec", {"key": ""}),
            ]
            for method, params in cases:
                req = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
                resp = await send_raw_ipc(socket_path, json.dumps(req))
                assert resp is not None
                err = resp.get("error", {})
                assert err.get("code") == INVALID_PARAMS, (
                    f"Method {method} with empty string returned {err}"
                )

    async def test_notification_suppression_on_fuzzing(self, tmp_path: Path) -> None:
        """JSON-RPC notifications (omitting 'id') MUST NOT produce responses even on error."""
        async with _running_daemon_ctx(tmp_path, "fuzz_notif") as (_, socket_path):
            failing_notifications = [
                {"jsonrpc": "2.0", "method": "systemvolume_set_volume", "params": {}},
                {"jsonrpc": "2.0", "method": "runcommand_get", "params": {"key": ""}},
                {"jsonrpc": "2.0", "method": "nonexistent_method", "params": {}},
            ]
            for notif in failing_notifications:
                reader, writer = await asyncio.open_unix_connection(
                    str(socket_path), limit=STREAM_LIMIT_BYTES
                )
                try:
                    # Write notification frame
                    writer.write((json.dumps(notif) + "\n").encode("utf-8"))
                    await writer.drain()

                    # Immediately write a normal request with id=999
                    ping_req = {
                        "jsonrpc": "2.0",
                        "method": "runcommand_list",
                        "params": {},
                        "id": 999,
                    }
                    writer.write((json.dumps(ping_req) + "\n").encode("utf-8"))
                    await writer.drain()

                    # Read exactly one response line
                    line = await asyncio.wait_for(reader.readline(), timeout=1.0)
                    assert line, "Expected response for ping request"
                    resp = json.loads(line.decode("utf-8"))

                    # Must be the ping response (notification had no response written)
                    assert resp.get("id") == 999
                    assert "result" in resp
                finally:
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except Exception:
                        pass


# =====================================================================
# 2. Domain Error Code Assertions (-32015, -32016, -32001, -32002)
# =====================================================================


class TestIPCErrorCodeAssertions:
    """Strict verification of domain error codes and exception mapping."""

    async def test_audio_error_code_mapping_32015(self, tmp_path: Path) -> None:
        """Audio exceptions must strictly map to AUDIO_SYSTEM_VOLUME_ERROR (-32015)."""
        socket_path = tmp_path / "test_audio_err_adv.sock"

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            if method == "systemvolume_set_volume":
                raise AudioError("Sound server daemon failed")
            elif method == "systemvolume_get_sinks":
                raise AudioBackendError("Backend wpctl exited with failure")
            elif method == "systemvolume_set_muted":
                raise AudioUnavailableError("No active sound server detected")
            elif method == "systemvolume_set_default":
                raise RuntimeError("Audio pipeline broken")
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()
        try:
            client = PenguLinkIPCClient(socket_path)

            # AudioError
            with pytest.raises(IPCError) as exc:
                await client.systemvolume_set_volume("sink1", 50)
            assert exc.value.code == AUDIO_SYSTEM_VOLUME_ERROR
            assert "Sound server daemon failed" in str(exc.value)

            # AudioBackendError
            with pytest.raises(IPCError) as exc:
                await client.systemvolume_get_sinks()
            assert exc.value.code == AUDIO_SYSTEM_VOLUME_ERROR
            assert "wpctl exited with failure" in str(exc.value)

            # AudioUnavailableError
            with pytest.raises(IPCError) as exc:
                await client.systemvolume_set_muted("sink1", True)
            assert exc.value.code == AUDIO_SYSTEM_VOLUME_ERROR
            assert "No active sound server" in str(exc.value)

            # Generic fallback for systemvolume_*
            with pytest.raises(IPCError) as exc:
                await client.systemvolume_set_default("sink1")
            assert exc.value.code == AUDIO_SYSTEM_VOLUME_ERROR
            assert "Audio pipeline broken" in str(exc.value)
        finally:
            await server.stop()

    async def test_run_command_error_code_mapping_32016(self, tmp_path: Path) -> None:
        """Command and storage exceptions must strictly map to RUN_COMMAND_ERROR (-32016)."""
        socket_path = tmp_path / "test_cmd_err_adv.sock"

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            if method == "runcommand_exec":
                raise CommandExecutionError("Process killed by SIGKILL")
            elif method == "runcommand_get":
                raise CommandNotFoundError(f"Command '{params.get('key')}' not found")
            elif method == "runcommand_add":
                raise StorageWriteError("Permission denied writing commands.json")
            elif method == "runcommand_list":
                raise StorageCorruptedError("commands.json is corrupted")
            elif method == "runcommand_remove":
                raise RunCommandError("Failed removing command")
            elif method == "runcommand_stop":
                raise ValueError("Command 'k1' is disabled")
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()
        try:
            client = PenguLinkIPCClient(socket_path)

            # CommandExecutionError
            with pytest.raises(IPCError) as exc:
                await client.runcommand_exec("k1")
            assert exc.value.code == RUN_COMMAND_ERROR
            assert "SIGKILL" in str(exc.value)

            # CommandNotFoundError
            with pytest.raises(IPCError) as exc:
                await client.runcommand_get("missing_k")
            assert exc.value.code == RUN_COMMAND_ERROR
            assert "not found" in str(exc.value)

            # StorageWriteError
            with pytest.raises(IPCError) as exc:
                await client.runcommand_add("New", "echo 1")
            assert exc.value.code == RUN_COMMAND_ERROR
            assert "Permission denied" in str(exc.value)

            # StorageCorruptedError
            with pytest.raises(IPCError) as exc:
                await client.runcommand_list()
            assert exc.value.code == RUN_COMMAND_ERROR
            assert "corrupted" in str(exc.value)

            # RunCommandError
            with pytest.raises(IPCError) as exc:
                await client.runcommand_remove("k1")
            assert exc.value.code == RUN_COMMAND_ERROR
            assert "Failed removing" in str(exc.value)

            # Disabled command ValueError
            with pytest.raises(IPCError) as exc:
                await client.runcommand_stop("1")
            assert exc.value.code == RUN_COMMAND_ERROR
            assert "disabled" in str(exc.value)
        finally:
            await server.stop()

    async def test_disconnected_device_error_mapping_32002(self, tmp_path: Path) -> None:
        """Remote audio operations on disconnected device return DEVICE_NOT_CONNECTED (-32002)."""
        cfg = _create_test_config(tmp_path, "disconn")
        socket_path = tmp_path / "daemon_disconn.sock"
        daemon = PenguLinkDaemon(
            config=cfg,
            socket_path=socket_path,
            audio_backend=_create_mock_backend(),
        )
        await daemon.start()
        try:
            client = PenguLinkIPCClient(socket_path)

            with pytest.raises(IPCError) as exc:
                await client.systemvolume_set_remote_volume("unpaired_device_1", "Speaker", 50)
            assert exc.value.code == DEVICE_NOT_CONNECTED
            assert "not connected" in str(exc.value).lower()

            with pytest.raises(IPCError) as exc:
                await client.systemvolume_set_remote_muted("unpaired_device_1", "Speaker", True)
            assert exc.value.code == DEVICE_NOT_CONNECTED

            with pytest.raises(IPCError) as exc:
                await client.systemvolume_set_remote_default("unpaired_device_1", "Speaker")
            assert exc.value.code == DEVICE_NOT_CONNECTED

            with pytest.raises(IPCError) as exc:
                await client.systemvolume_request_remote_sinks("unpaired_device_1")
            assert exc.value.code == DEVICE_NOT_CONNECTED
        finally:
            await daemon.stop()

    async def test_device_not_found_mapping_32001(self, tmp_path: Path) -> None:
        """Device lookup errors must map strictly to DEVICE_NOT_FOUND (-32001)."""
        socket_path = tmp_path / "daemon_dev_nf.sock"

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            raise DeviceNotFoundError(f"Device '{params.get('device_id')}' not found")

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()
        try:
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc:
                await client.call("some_device_lookup", {"device_id": "ghost_1"})
            assert exc.value.code == DEVICE_NOT_FOUND
            assert "ghost_1" in str(exc.value)
        finally:
            await server.stop()

    async def test_method_not_found_mapping_32601(self, tmp_path: Path) -> None:
        """Calling non-existent method returns METHOD_NOT_FOUND (-32601)."""
        cfg = _create_test_config(tmp_path, "method_nf")
        socket_path = tmp_path / "daemon_method_nf.sock"
        daemon = PenguLinkDaemon(
            config=cfg,
            socket_path=socket_path,
            audio_backend=_create_mock_backend(),
        )
        await daemon.start()
        try:
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc:
                await client.call("systemvolume_nonexistent_action", {})
            assert exc.value.code == METHOD_NOT_FOUND
            assert (
                "Method not found" in str(exc.value)
                or "Unknown JSON-RPC method" in str(exc.value)
            )
        finally:
            await daemon.stop()


# =====================================================================
# 3. Boundary Values & Line Overrun Stress Tests
# =====================================================================


class TestIPCBoundaryValues:
    """Boundary testing: volume ranges, extreme keys, special characters, and line overruns."""

    async def test_volume_integer_boundaries(self, tmp_path: Path) -> None:
        """Volume boundaries (negative, 0, 100, 65536, >65536, huge integers) clamp cleanly."""
        async with _running_daemon_ctx(tmp_path, "bounds_vol") as (daemon, socket_path):
            client = PenguLinkIPCClient(socket_path)
            sink_name = "alsa_output.pci-0000.analog-stereo"

            # 1. Zero volume
            res = await client.systemvolume_set_volume(sink_name, 0)
            assert res.get("success") is True
            sink = await daemon.audio_service.get_sink(sink_name)
            assert sink is not None and sink.volume == 0

            # 2. Percentage mode: 50% -> 32768
            res = await client.systemvolume_set_volume(sink_name, 50)
            assert res.get("success") is True
            sink = await daemon.audio_service.get_sink(sink_name)
            assert sink is not None and sink.volume == 32768

            # 3. Percentage mode: 100% -> 65536
            res = await client.systemvolume_set_volume(sink_name, 100)
            assert res.get("success") is True
            sink = await daemon.audio_service.get_sink(sink_name)
            assert sink is not None and sink.volume == 65536

            # 4. Raw volume mode: 45000 (> 100)
            res = await client.systemvolume_set_volume(sink_name, 45000)
            assert res.get("success") is True
            sink = await daemon.audio_service.get_sink(sink_name)
            assert sink is not None and sink.volume == 45000

            # 5. Exact max raw volume: 65536
            res = await client.systemvolume_set_volume(sink_name, 65536)
            assert res.get("success") is True
            sink = await daemon.audio_service.get_sink(sink_name)
            assert sink is not None and sink.volume == 65536

            # 6. Negative volume clamped to 0 (service clamps target_vol to 0)
            res = await client.systemvolume_set_volume(sink_name, -500)
            assert res.get("success") is True
            sink = await daemon.audio_service.get_sink(sink_name)
            assert sink is not None and sink.volume == 0

            # 7. Overflow volume clamped to 65536
            res = await client.systemvolume_set_volume(sink_name, 100000)
            assert res.get("success") is True
            sink = await daemon.audio_service.get_sink(sink_name)
            assert sink is not None and sink.volume == 65536

    async def test_command_keys_and_unicode_boundaries(self, tmp_path: Path) -> None:
        """Command keys with negative numeric strings, UUIDs, Unicode, and emojis."""
        async with _running_daemon_ctx(tmp_path, "bounds_cmd") as (_, socket_path):
            client = PenguLinkIPCClient(socket_path)

            test_commands = [
                ("🔊 Mute Audio", "echo 'mute toggle'"),
                ("🚀 Launch App 🎉", "echo 'PenguLink Rocket'"),
                ("Long_Name_" + "A" * 500, "echo 'long'"),
                ("Shell Metachars", 'echo "a; b && c || d | grep x"'),
                ("Numeric String Key", "echo 'numeric'"),
            ]

            added_keys = []
            for name, cmd in test_commands:
                entry = await client.runcommand_add(name, cmd)
                assert "key" in entry
                assert entry["name"] == name
                added_keys.append(entry["key"])

            # Verify all added commands exist in list
            cmd_list = await client.runcommand_list()
            keys_in_list = {c["key"] for c in cmd_list}
            for k in added_keys:
                assert k in keys_in_list

                # Verify retrieval
                fetched = await client.runcommand_get(k)
                assert fetched["key"] == k

                # Verify safe execution without shell expansion exploits
                res_exec = await client.runcommand_exec(k)
                assert res_exec.get("success") is True
                assert res_exec.get("exit_code") == 0

    async def test_stream_line_guard_boundary(self, tmp_path: Path) -> None:
        """Requests exceeding 1 MB line guard limit trigger clean disconnection without crash."""
        async with _running_daemon_ctx(tmp_path, "bounds_line") as (_, socket_path):
            # Large request below limit (~200 KB) should succeed
            large_str = "x" * 200_000
            req_ok = {
                "jsonrpc": "2.0",
                "method": "runcommand_add",
                "params": {"name": "LargeCmd", "command": f"echo '{large_str}'"},
                "id": 1,
            }
            resp = await send_raw_ipc(socket_path, json.dumps(req_ok))
            assert resp is not None
            assert "result" in resp

            # Massive request exceeding 1 MB limit (> 1_048_576 bytes)
            huge_str = "x" * (STREAM_LIMIT_BYTES + 1024)
            req_huge = {
                "jsonrpc": "2.0",
                "method": "runcommand_add",
                "params": {"name": "Huge", "command": huge_str},
                "id": 2,
            }
            # Connection should be terminated by daemon line limit guard
            resp_huge = await send_raw_ipc(socket_path, json.dumps(req_huge), timeout=1.0)
            assert resp_huge is None

            # Daemon must still be healthy and accepting subsequent requests
            client = PenguLinkIPCClient(socket_path)
            sinks = await client.systemvolume_get_sinks()
            assert len(sinks) >= 1


# =====================================================================
# 4. Concurrency Bursts & Socket Stress Tests
# =====================================================================


class TestIPCConcurrencyBursts:
    """Stress-test high-concurrency client bursts and pipelined socket traffic."""

    async def test_concurrency_burst_read_queries(self, tmp_path: Path) -> None:
        """Burst 50 concurrent async read calls to systemvolume and runcommand endpoints."""
        async with _running_daemon_ctx(tmp_path, "concurr_read") as (_, socket_path):
            async def worker(idx: int) -> bool:
                client = PenguLinkIPCClient(socket_path)
                if idx % 2 == 0:
                    sinks = await client.systemvolume_get_sinks()
                    return len(sinks) >= 2
                else:
                    cmds = await client.runcommand_list()
                    return isinstance(cmds, list)

            tasks = [asyncio.create_task(worker(i)) for i in range(50)]
            results = await asyncio.gather(*tasks)
            assert all(results), "All 50 concurrent calls must succeed"

    async def test_concurrency_burst_mutations(self, tmp_path: Path) -> None:
        """Burst 30 concurrent async mutation calls (set_volume, set_muted, runcommand_add)."""
        async with _running_daemon_ctx(tmp_path, "concurr_mut") as (_, socket_path):
            sink_name = "alsa_output.pci-0000.analog-stereo"

            async def volume_worker(idx: int) -> bool:
                client = PenguLinkIPCClient(socket_path)
                vol = (idx * 2000) % 65536
                res = await client.systemvolume_set_volume(sink_name, vol)
                return res.get("success") is True

            async def command_worker(idx: int) -> bool:
                client = PenguLinkIPCClient(socket_path)
                res = await client.runcommand_add(f"BurstCmd_{idx}", f"echo '{idx}'")
                return "key" in res

            tasks = []
            for i in range(15):
                tasks.append(asyncio.create_task(volume_worker(i)))
                tasks.append(asyncio.create_task(command_worker(i)))

            results = await asyncio.gather(*tasks)
            assert all(results), "All 30 mutation tasks must succeed without deadlock"

    async def test_socket_pipelining_single_connection(self, tmp_path: Path) -> None:
        """Pipeline 25 JSON-RPC requests back-to-back in a single buffer over one connection."""
        async with _running_daemon_ctx(tmp_path, "concurr_pipe") as (_, socket_path):
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            num_requests = 25
            try:
                # Batch write 25 requests
                batch_lines = []
                for i in range(num_requests):
                    req = {
                        "jsonrpc": "2.0",
                        "method": "systemvolume_get_sinks",
                        "params": {},
                        "id": i + 1,
                    }
                    batch_lines.append(json.dumps(req))

                writer.write(("\n".join(batch_lines) + "\n").encode("utf-8"))
                await writer.drain()

                # Read 25 responses sequentially
                received_ids = []
                for _ in range(num_requests):
                    line = await asyncio.wait_for(reader.readline(), timeout=3.0)
                    assert line, "Expected response line from pipelined stream"
                    resp = json.loads(line.decode("utf-8"))
                    assert "result" in resp
                    received_ids.append(resp.get("id"))

                assert received_ids == list(range(1, num_requests + 1))
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

    async def test_rapid_connection_churn(self, tmp_path: Path) -> None:
        """Rapidly open and immediately close 30 client connections without hanging server."""
        async with _running_daemon_ctx(tmp_path, "concurr_churn") as (_, socket_path):
            for _ in range(30):
                reader, writer = await asyncio.open_unix_connection(str(socket_path))
                writer.write(b'{"jsonrpc": "2.0", "method": "runcommand_list", "id": 1}\n')
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

            # Verify server is intact and fully responsive
            client = PenguLinkIPCClient(socket_path)
            res = await client.runcommand_list()
            assert isinstance(res, list)


# =====================================================================
# 5. Corrupted Storage Recovery via IPC
# =====================================================================


class TestIPCCorruptedStorageRecovery:
    """Verify IPC server and CommandStore recover gracefully from corrupted storage files."""

    async def test_corrupted_json_syntax_recovery_via_ipc(self, tmp_path: Path) -> None:
        """Corrupted JSON syntax in commands.json triggers backup and returns empty list via IPC."""
        cfg = _create_test_config(tmp_path, "corrupt_syntax")
        socket_path = tmp_path / "daemon_corrupt_syntax.sock"
        store_path = tmp_path / "commands_corrupt.json"

        # Plant corrupted JSON on disk
        store_path.write_text('{"commands": { "broken": syntax error here...', encoding="utf-8")

        cmd_store = CommandStore(storage_path=store_path)
        daemon = PenguLinkDaemon(
            config=cfg,
            socket_path=socket_path,
            command_store=cmd_store,
            audio_backend=_create_mock_backend(),
        )
        await daemon.start()
        try:
            client = PenguLinkIPCClient(socket_path)

            # 1. runcommand_list should recover and return empty list
            cmds = await client.runcommand_list()
            assert cmds == []

            # 2. Corrupt backup file must have been created
            backups = list(tmp_path.glob("commands_corrupt.corrupt.*.bak"))
            assert len(backups) >= 1

            # 3. Add command via IPC works post-corruption
            new_cmd = await client.runcommand_add("PostCorrupt", "echo recovered")
            assert new_cmd["name"] == "PostCorrupt"
            k = new_cmd["key"]

            # 4. Get command returns newly added entry
            fetched = await client.runcommand_get(k)
            assert fetched["key"] == k

            # 5. Disk file is valid JSON again
            saved_content = json.loads(store_path.read_text(encoding="utf-8"))
            assert "commands" in saved_content
            assert k in saved_content["commands"]
        finally:
            await daemon.stop()

    async def test_corrupted_root_array_recovery_via_ipc(self, tmp_path: Path) -> None:
        """Non-dict root in commands.json triggers safe recovery via IPC."""
        cfg = _create_test_config(tmp_path, "corrupt_root")
        socket_path = tmp_path / "daemon_corrupt_root.sock"
        store_path = tmp_path / "commands_root.json"

        # Plant non-dict JSON
        store_path.write_text("[1, 2, 3, 4, 5]", encoding="utf-8")

        cmd_store = CommandStore(storage_path=store_path)
        daemon = PenguLinkDaemon(
            config=cfg,
            socket_path=socket_path,
            command_store=cmd_store,
            audio_backend=_create_mock_backend(),
        )
        await daemon.start()
        try:
            client = PenguLinkIPCClient(socket_path)
            cmds = await client.runcommand_list()
            assert cmds == []
        finally:
            await daemon.stop()

    async def test_storage_write_permission_error_mapping_32016(self, tmp_path: Path) -> None:
        """Storage write failure maps strictly to RUN_COMMAND_ERROR (-32016)."""
        socket_path = tmp_path / "daemon_storage_err.sock"

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            if method == "runcommand_add":
                raise StorageWriteError("Permission denied: cannot write commands.json")
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()
        try:
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc:
                await client.runcommand_add("FailCmd", "echo 1")
            assert exc.value.code == RUN_COMMAND_ERROR
            assert "Permission denied" in str(exc.value)
        finally:
            await server.stop()

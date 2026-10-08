"""tests.unit.test_ipc_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial challenge test suite for Unix Domain Socket IPC and PenguLinkDaemon runtime:
- Socket collision edge cases (active daemon probe vs stale socket unlinking, corrupted files, deep directories)
- Stream buffer overruns (>1 MB line guard limit, chunked flooding, boundary payloads)
- Malformed JSON-RPC 2.0 requests (missing jsonrpc, invalid root types, non-string methods, invalid params)
- Error code compliance (PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS)
- Notification error handling (silence on errors for notifications)
- High-concurrency client burst and rapid churn over Unix Domain Socket
- Clean server teardown with active client connections
- Daemon service IPC method boundaries and error responses
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.core.exceptions import (
    DaemonAlreadyRunningError,
    DaemonNotRunningError,
    DeviceNotFoundError,
    IPCError,
)
from pengulink.daemon.ipc import (
    APPLICATION_ERROR,
    DEVICE_NOT_CONNECTED,
    DEVICE_NOT_FOUND,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    STREAM_LIMIT_BYTES,
    PenguLinkIPCClient,
    PenguLinkIPCServer,
)
from pengulink.daemon.service import PenguLinkDaemon


@pytest.fixture
def temp_ipc_socket(tmp_path: Path) -> Path:
    sock_dir = tmp_path / "runtime"
    return sock_dir / "daemon.sock"


# =====================================================================
# 1. Socket Collision & Filesystem Edge Cases
# =====================================================================

async def test_adv_active_daemon_probe_preserves_running_server(temp_ipc_socket: Path) -> None:
    """Stress-test active daemon collision detection.

    Starting a second server instance against an active server MUST fail with
    DaemonAlreadyRunningError, AND the original server MUST remain healthy
    and capable of serving requests without disconnection.
    """
    async def handler(method: str, params: Dict[str, Any]) -> Any:
        return {"status": "alive", "echo": params.get("val")}

    server1 = PenguLinkIPCServer(socket_path=temp_ipc_socket, handler=handler)
    await server1.start()
    assert server1.is_running

    # Second server attempts to start on same path
    server2 = PenguLinkIPCServer(socket_path=temp_ipc_socket, handler=handler)
    with pytest.raises(DaemonAlreadyRunningError):
        await server2.start()

    assert not server2.is_running
    assert server1.is_running

    # Original server must still respond to client requests
    client = PenguLinkIPCClient(socket_path=temp_ipc_socket)
    resp = await client.call("echo_test", {"val": "still_working"})
    assert resp == {"status": "alive", "echo": "still_working"}

    await server1.stop()
    assert not temp_ipc_socket.exists()


async def test_adv_stale_socket_recovery_from_abandoned_file(temp_ipc_socket: Path) -> None:
    """Test recovery from an unlinked or abandoned file left after process crash."""
    temp_ipc_socket.parent.mkdir(parents=True, exist_ok=True)
    temp_ipc_socket.write_text("abandoned_dead_process_socket_bytes")
    assert temp_ipc_socket.exists()

    async def recover_handler(m: str, p: Dict[str, Any]) -> Any:
        return {"recovered": True}

    server = PenguLinkIPCServer(socket_path=temp_ipc_socket, handler=recover_handler)
    await server.start()
    assert server.is_running

    client = PenguLinkIPCClient(socket_path=temp_ipc_socket)
    res = await client.call("ping")
    assert res == {"recovered": True}

    await server.stop()
    assert not temp_ipc_socket.exists()


async def test_adv_nested_nonexistent_parent_directories(tmp_path: Path) -> None:
    """Server start must create arbitrarily deep parent directory structure with 0700 permissions."""
    deep_path = tmp_path / "a" / "b" / "c" / "d" / "pengulink.sock"
    assert not deep_path.parent.exists()

    async def dummy_handler(m: str, p: Dict[str, Any]) -> Any:
        return "ok"

    server = PenguLinkIPCServer(socket_path=deep_path, handler=dummy_handler)
    await server.start()
    assert server.is_running
    assert deep_path.exists()
    assert (deep_path.parent.stat().st_mode & 0o777) == 0o700

    await server.stop()


async def test_adv_rapid_start_stop_cycling(temp_ipc_socket: Path) -> None:
    """Verify rapid consecutive start/stop cycles do not leak file descriptors or leave zombie state."""
    async def dummy_handler(m: str, p: Dict[str, Any]) -> Any:
        return "pong"

    for i in range(10):
        server = PenguLinkIPCServer(socket_path=temp_ipc_socket, handler=dummy_handler)
        await server.start()
        assert server.is_running
        assert temp_ipc_socket.exists()
        await server.stop()
        assert not server.is_running
        assert not temp_ipc_socket.exists()


# =====================================================================
# 2. Stream Overruns (>1 MB Line Limit)
# =====================================================================

async def test_adv_stream_overrun_server_drops_connection_and_survives(temp_ipc_socket: Path) -> None:
    """Sending a line exceeding STREAM_LIMIT_BYTES (1 MB) must trigger LimitOverrunError on server,

    close the rogue connection, and NOT crash the server or block other clients.
    """
    received_calls: List[str] = []

    async def handler(method: str, params: Dict[str, Any]) -> Any:
        received_calls.append(method)
        return {"received": method}

    server = PenguLinkIPCServer(socket_path=temp_ipc_socket, handler=handler)
    await server.start()

    # Client 1: Sends > 1 MB stream without newline
    reader1, writer1 = await asyncio.open_unix_connection(str(temp_ipc_socket))
    oversized_chunk = b"X" * (STREAM_LIMIT_BYTES + 4096)
    writer1.write(oversized_chunk)
    await writer1.drain()

    # The server should disconnect client 1 upon LimitOverrunError
    # Reading from client 1 should return EOF
    eof_line = await reader1.readline()
    assert eof_line == b"", "Server should have severed connection on stream overrun"
    writer1.close()
    await writer1.wait_closed()

    # Server MUST still be healthy for legitimate clients
    client2 = PenguLinkIPCClient(socket_path=temp_ipc_socket)
    resp = await client2.call("legitimate_call")
    assert resp == {"received": "legitimate_call"}
    assert "legitimate_call" in received_calls

    await server.stop()


async def test_adv_stream_boundary_payload_within_limit(temp_ipc_socket: Path) -> None:
    """Sending a payload close to the limit (~500 KB) must be processed cleanly."""
    async def handler(method: str, params: Dict[str, Any]) -> Any:
        return {"payload_len": len(params.get("data", ""))}

    server = PenguLinkIPCServer(socket_path=temp_ipc_socket, handler=handler)
    await server.start()

    client = PenguLinkIPCClient(socket_path=temp_ipc_socket)
    large_str = "A" * 500_000
    res = await client.call("large_payload", {"data": large_str})
    assert res == {"payload_len": 500_000}

    await server.stop()


# =====================================================================
# 3. Malformed JSON-RPC Requests & Specification Compliance
# =====================================================================

async def test_adv_malformed_json_syntax(temp_ipc_socket: Path) -> None:
    """Corrupted syntax or null bytes must return standard PARSE_ERROR (-32700)."""
    async def dummy_handler(m: str, p: Dict[str, Any]) -> Any:
        return "ok"

    server = PenguLinkIPCServer(socket_path=temp_ipc_socket, handler=dummy_handler)
    await server.start()

    reader, writer = await asyncio.open_unix_connection(str(temp_ipc_socket))

    malformed_inputs = [
        b"{invalid_json_payload\n",
        b'{"jsonrpc": "2.0", "method": "test", \x00\x01\x02}\n',
        b'{"jsonrpc": "2.0", "method": "test", "params": \n',
        b'<!DOCTYPE html><html><body>Error</body></html>\n',
    ]

    for bad_bytes in malformed_inputs:
        writer.write(bad_bytes)
        await writer.drain()

        line = await reader.readline()
        assert line, f"Expected response for bad input: {bad_bytes!r}"
        resp = json.loads(line.decode("utf-8"))
        assert resp.get("jsonrpc") == "2.0"
        assert resp["error"]["code"] == PARSE_ERROR
        assert resp.get("id") is None

    writer.close()
    await writer.wait_closed()
    await server.stop()


async def test_adv_invalid_jsonrpc_version_and_root_types(temp_ipc_socket: Path) -> None:
    """Missing or incorrect jsonrpc version, or non-object root JSON must return INVALID_REQUEST (-32600)."""
    async def dummy_handler(m: str, p: Dict[str, Any]) -> Any:
        return "ok"

    server = PenguLinkIPCServer(socket_path=temp_ipc_socket, handler=dummy_handler)
    await server.start()

    reader, writer = await asyncio.open_unix_connection(str(temp_ipc_socket))

    invalid_envelopes = [
        # Missing jsonrpc version
        b'{"method": "test", "params": {}, "id": 1}\n',
        # Wrong jsonrpc version
        b'{"jsonrpc": "1.0", "method": "test", "params": {}, "id": 2}\n',
        b'{"jsonrpc": "2.1", "method": "test", "params": {}, "id": 3}\n',
        # Non-dict root types
        b'[1, 2, 3]\n',
        b'"just a string"\n',
        b'12345\n',
        b'true\n',
        b'null\n',
    ]

    for env_bytes in invalid_envelopes:
        writer.write(env_bytes)
        await writer.drain()

        line = await reader.readline()
        assert line, f"Expected error response for: {env_bytes!r}"
        resp = json.loads(line.decode("utf-8"))
        assert resp["error"]["code"] == INVALID_REQUEST
        assert resp.get("id") is None

    writer.close()
    await writer.wait_closed()
    await server.stop()


async def test_adv_invalid_method_types(temp_ipc_socket: Path) -> None:
    """Method must be a non-empty string; anything else must return INVALID_REQUEST (-32600)."""
    async def dummy_handler(m: str, p: Dict[str, Any]) -> Any:
        return "ok"

    server = PenguLinkIPCServer(socket_path=temp_ipc_socket, handler=dummy_handler)
    await server.start()

    reader, writer = await asyncio.open_unix_connection(str(temp_ipc_socket))

    bad_methods = [
        b'{"jsonrpc": "2.0", "method": 12345, "id": 10}\n',
        b'{"jsonrpc": "2.0", "method": null, "id": 11}\n',
        b'{"jsonrpc": "2.0", "method": "", "id": 12}\n',
        b'{"jsonrpc": "2.0", "method": ["test"], "id": 13}\n',
        b'{"jsonrpc": "2.0", "method": {"name": "test"}, "id": 14}\n',
    ]

    for m_bytes in bad_methods:
        writer.write(m_bytes)
        await writer.drain()

        line = await reader.readline()
        resp = json.loads(line.decode("utf-8"))
        assert resp["error"]["code"] == INVALID_REQUEST
        assert "Method must be a non-empty string" in resp["error"]["message"]

    writer.close()
    await writer.wait_closed()
    await server.stop()


async def test_adv_invalid_params_types(temp_ipc_socket: Path) -> None:
    """Params must be an object (dict); non-dict params must return INVALID_PARAMS (-32602)."""
    async def dummy_handler(m: str, p: Dict[str, Any]) -> Any:
        return "ok"

    server = PenguLinkIPCServer(socket_path=temp_ipc_socket, handler=dummy_handler)
    await server.start()

    reader, writer = await asyncio.open_unix_connection(str(temp_ipc_socket))

    bad_params = [
        b'{"jsonrpc": "2.0", "method": "test", "params": "string_params", "id": 20}\n',
        b'{"jsonrpc": "2.0", "method": "test", "params": 123, "id": 21}\n',
        b'{"jsonrpc": "2.0", "method": "test", "params": [1, 2, 3], "id": 22}\n',
    ]

    for p_bytes in bad_params:
        writer.write(p_bytes)
        await writer.drain()

        line = await reader.readline()
        resp = json.loads(line.decode("utf-8"))
        assert resp["error"]["code"] == INVALID_PARAMS
        assert "Params must be an object" in resp["error"]["message"]

    writer.close()
    await writer.wait_closed()
    await server.stop()


async def test_adv_unknown_method_error_code(temp_ipc_socket: Path) -> None:
    """When a method is not recognized, verify server response error handling.

    Note: In JSON-RPC 2.0 specification, unknown method returns METHOD_NOT_FOUND (-32601).
    We test whether server responds cleanly with a structured error without crashing.
    """
    async def sample_handler(method: str, params: Dict[str, Any]) -> Any:
        if method == "known_method":
            return "ok"
        raise KeyError(f"Unknown JSON-RPC method '{method}'")

    server = PenguLinkIPCServer(socket_path=temp_ipc_socket, handler=sample_handler)
    await server.start()

    reader, writer = await asyncio.open_unix_connection(str(temp_ipc_socket))
    writer.write(b'{"jsonrpc": "2.0", "method": "non_existent_method", "id": 50}\n')
    await writer.drain()

    line = await reader.readline()
    resp = json.loads(line.decode("utf-8"))
    writer.close()
    await writer.wait_closed()
    await server.stop()

    assert resp["id"] == 50
    assert "error" in resp
    actual_code = resp["error"]["code"]
    assert actual_code == METHOD_NOT_FOUND


async def test_adv_notification_does_not_send_error_response(temp_ipc_socket: Path) -> None:
    """JSON-RPC 2.0 notifications (requests without 'id') MUST NEVER receive a response,

    even if an error (e.g. DeviceNotFoundError, invalid params, application error) occurs.
    """
    async def failing_handler(method: str, params: Dict[str, Any]) -> Any:
        raise DeviceNotFoundError("Device not found for notification")

    server = PenguLinkIPCServer(socket_path=temp_ipc_socket, handler=failing_handler)
    await server.start()

    reader, writer = await asyncio.open_unix_connection(str(temp_ipc_socket))

    # Send notification that causes DeviceNotFoundError
    writer.write(b'{"jsonrpc": "2.0", "method": "fail_notify", "params": {}}\n')
    await writer.drain()

    # Now send a regular request with id=999
    writer.write(b'{"jsonrpc": "2.0", "method": "fail_notify", "params": {}, "id": 999}\n')
    await writer.drain()

    # The first line read should correspond to id=999, NOT the notification
    line = await reader.readline()
    resp = json.loads(line.decode("utf-8"))
    assert resp["id"] == 999
    assert resp["error"]["code"] == DEVICE_NOT_FOUND

    writer.close()
    await writer.wait_closed()
    await server.stop()


# =====================================================================
# 4. Rapid Concurrent Client Connections & Churn
# =====================================================================

async def test_adv_concurrent_clients_burst(temp_ipc_socket: Path) -> None:
    """Stress-test with 50 concurrent client connections executing simultaneously."""
    call_counts = 0
    lock = asyncio.Lock()

    async def counting_handler(method: str, params: Dict[str, Any]) -> Any:
        nonlocal call_counts
        async with lock:
            call_counts += 1
        await asyncio.sleep(0.01)  # Simulate small async processing
        return {"client_id": params.get("client_id"), "count": call_counts}

    server = PenguLinkIPCServer(socket_path=temp_ipc_socket, handler=counting_handler)
    await server.start()

    async def client_worker(cid: int) -> Dict[str, Any]:
        c = PenguLinkIPCClient(socket_path=temp_ipc_socket)
        return await c.call("test_call", {"client_id": cid})

    tasks = [asyncio.create_task(client_worker(i)) for i in range(50)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 50
    for i, res in enumerate(results):
        assert res["client_id"] == i

    assert call_counts == 50

    await server.stop()


async def test_adv_rapid_connect_disconnect_churn(temp_ipc_socket: Path) -> None:
    """50 clients rapidly connect and disconnect immediately without sending data.

    Server must not leak client tasks or crash with unhandled connection errors.
    """
    async def dummy_handler(m: str, p: Dict[str, Any]) -> Any:
        return "ok"

    server = PenguLinkIPCServer(socket_path=temp_ipc_socket, handler=dummy_handler)
    await server.start()

    for _ in range(50):
        reader, writer = await asyncio.open_unix_connection(str(temp_ipc_socket))
        writer.close()
        await writer.wait_closed()

    # Verify server is still functional
    client = PenguLinkIPCClient(socket_path=temp_ipc_socket)
    assert await client.call("ping") == "ok"

    await server.stop()


# =====================================================================
# 5. Full Daemon Service Integration Stress
# =====================================================================

async def test_adv_daemon_ipc_method_error_handling(tmp_path: Path) -> None:
    """Adversarially challenge PenguLinkDaemon IPC method dispatcher with invalid inputs."""
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / "config"
    cfg.storage.data_dir = tmp_path / "data"
    cfg.storage.runtime_dir = tmp_path / "run"
    cfg.device.device_id = "testdaemon1234567890abcdef123456"
    cfg.device.device_name = "PenguLink-Daemon-Test"
    cfg.network.udp_port = 17169
    cfg.network.tcp_port_min = 17210
    cfg.network.tcp_port_max = 17215
    cfg.ensure_directories()

    sock_path = tmp_path / "daemon.sock"

    daemon = PenguLinkDaemon(config=cfg, socket_path=sock_path)
    await daemon.start()

    client = PenguLinkIPCClient(socket_path=sock_path)

    try:
        # 1. status works
        status = await client.status()
        assert status["running"] is True
        assert status["device_id"] == cfg.device.device_id

        # 2. ping on non-existent device raises DEVICE_NOT_CONNECTED or DEVICE_NOT_FOUND
        with pytest.raises(IPCError) as exc_info:
            await client.send_ping("non_existent_device_id")
        assert f"[{DEVICE_NOT_CONNECTED}]" in str(exc_info.value) or f"[{DEVICE_NOT_FOUND}]" in str(exc_info.value)

        # 3. dismiss notification on non-connected device raises DEVICE_NOT_CONNECTED
        with pytest.raises(IPCError) as exc_info:
            await client.dismiss_notification("device_999", "notif_123")
        assert f"[{DEVICE_NOT_CONNECTED}]" in str(exc_info.value)

        # 4. reply notification missing required message raises INVALID_PARAMS
        with pytest.raises(IPCError) as exc_info:
            await client.call("reply_notification", {"device_id": "d1", "reply_id": "r1"})
        assert f"[{INVALID_PARAMS}]" in str(exc_info.value)

        # 5. get_logs with custom limits
        logs = await client.get_logs(limit=5)
        assert isinstance(logs, list)

        # 6. get_devices online_only and paired_only filters
        devs = await client.get_devices(online_only=True)
        assert isinstance(devs, list)

        # 7. stop_daemon via IPC
        stop_res = await client.stop_daemon()
        assert stop_res["success"] is True

        # Wait for daemon to fully stop
        await daemon.wait_until_stopped()
        assert not daemon.is_running
        assert not sock_path.exists()

    finally:
        if daemon.is_running:
            await daemon.stop()

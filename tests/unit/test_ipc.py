"""tests.unit.test_ipc
~~~~~~~~~~~~~~~~~~~
Unit tests for Unix Domain Socket IPC (PenguLinkIPCServer, PenguLinkIPCClient).
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from pengulink.core.exceptions import (
    DaemonAlreadyRunningError,
    DaemonNotRunningError,
    DeviceNotFoundError,
    IPCError,
    PairingError,
)
from pengulink.daemon.ipc import (
    APPLICATION_ERROR,
    CLIPBOARD_ERROR,
    DEVICE_NOT_FOUND,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PenguLinkIPCClient,
    PenguLinkIPCServer,
    resolve_socket_path,
)
from pengulink.plugins.clipboard.backend import ClipboardError, ClipboardUnavailableError


@pytest.fixture
def temp_socket_path(tmp_path: Path) -> Path:
    """Provide a path for temporary unix domain socket within tmp_path."""
    sock_dir = tmp_path / "run"
    return sock_dir / "daemon.sock"


async def test_resolve_socket_path_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify resolve_socket_path hierarchy."""
    # 1. Explicit parameter
    explicit = tmp_path / "custom.sock"
    assert resolve_socket_path(explicit) == explicit

    # 2. Environment variable
    monkeypatch.setenv("PENGULINK_SOCKET_PATH", str(tmp_path / "env.sock"))
    assert resolve_socket_path() == tmp_path / "env.sock"
    monkeypatch.delenv("PENGULINK_SOCKET_PATH", raising=False)

    # 3. XDG_RUNTIME_DIR
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "xdg"))
    assert resolve_socket_path() == tmp_path / "xdg" / "pengulink" / "daemon.sock"
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

    # 4. Fallback ~/.config/pengulink/daemon.sock
    default_path = resolve_socket_path()
    assert default_path.name == "daemon.sock"


async def test_ipc_server_permissions_and_lifecycle(temp_socket_path: Path) -> None:
    """Verify IPC server enforces 0700 on dir and 0600 on socket file."""
    server = PenguLinkIPCServer(socket_path=temp_socket_path)
    try:
        await server.start()
        assert server.is_running
        assert temp_socket_path.exists()

        # Check POSIX permissions
        dir_mode = temp_socket_path.parent.stat().st_mode
        sock_mode = temp_socket_path.stat().st_mode

        # Directory permissions should have only user bits (0700)
        assert (dir_mode & 0o777) == 0o700

        # Socket file should be 0600 (or on linux 0600 permissions mask)
        assert (sock_mode & 0o077) == 0  # No group or world permissions
    finally:
        await server.stop()
        assert not server.is_running
        assert not temp_socket_path.exists()


async def test_ipc_collision_detection_active_daemon(temp_socket_path: Path) -> None:
    """Verify starting a second daemon raises DaemonAlreadyRunningError if live daemon exists."""
    server1 = PenguLinkIPCServer(socket_path=temp_socket_path)
    server2 = PenguLinkIPCServer(socket_path=temp_socket_path)

    await server1.start()
    try:
        with pytest.raises(DaemonAlreadyRunningError) as exc_info:
            await server2.start()
        assert "already active" in str(exc_info.value)
    finally:
        await server1.stop()


async def test_ipc_collision_detection_stale_socket_recovery(temp_socket_path: Path) -> None:
    """Verify stale unlinked socket file (from crash) is detected, unlinked, and rebound cleanly."""
    temp_socket_path.parent.mkdir(parents=True, exist_ok=True)
    temp_socket_path.write_text("stale_socket_content")
    assert temp_socket_path.exists()

    server = PenguLinkIPCServer(socket_path=temp_socket_path)
    try:
        await server.start()
        assert server.is_running
    finally:
        await server.stop()


async def test_ipc_stream_framing_and_client_roundtrip(temp_socket_path: Path) -> None:
    """Verify normal JSON-RPC request and response roundtrip with PenguLinkIPCClient."""
    async def sample_handler(method: str, params: Dict[str, Any]) -> Any:
        if method == "echo":
            return {"echoed": params.get("value")}
        elif method == "add":
            return params.get("a", 0) + params.get("b", 0)
        raise KeyError(f"Unknown method {method}")

    server = PenguLinkIPCServer(socket_path=temp_socket_path, handler=sample_handler)
    client = PenguLinkIPCClient(socket_path=temp_socket_path)

    await server.start()
    try:
        # Call echo
        res1 = await client.call("echo", {"value": "hello_ipc"})
        assert res1 == {"echoed": "hello_ipc"}

        # Call add
        res2 = await client.call("add", {"a": 10, "b": 25})
        assert res2 == 35

        # Check is_running
        async def handler_with_status(method: str, params: Dict[str, Any]) -> Any:
            if method == "status":
                return {"running": True}
            return await sample_handler(method, params)

        server.handler = handler_with_status
        assert await client.is_running()
    finally:
        await server.stop()


def test_ipc_call_sync_wrapper(temp_socket_path: Path) -> None:
    """Verify call_sync provides synchronous wrapper for CLI."""
    async def sample_handler(method: str, params: Dict[str, Any]) -> Any:
        return {"sync": True, "method": method}

    server = PenguLinkIPCServer(socket_path=temp_socket_path, handler=sample_handler)

    loop = asyncio.new_event_loop()
    def _run_server():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.start())
        loop.run_forever()

    th = threading.Thread(target=_run_server, daemon=True)
    th.start()

    # Wait for socket
    for _ in range(50):
        if temp_socket_path.exists():
            break
        time.sleep(0.02)

    try:
        client = PenguLinkIPCClient(socket_path=temp_socket_path)
        res = client.call_sync("test_sync", {"foo": "bar"})
        assert res == {"sync": True, "method": "test_sync"}
    finally:
        fut = asyncio.run_coroutine_threadsafe(server.stop(), loop)
        fut.result(timeout=2.0)
        loop.call_soon_threadsafe(loop.stop)
        th.join(timeout=2.0)


async def test_ipc_empty_lines_ignored(temp_socket_path: Path) -> None:
    """Verify empty lines and whitespace lines do not trigger errors."""
    server = PenguLinkIPCServer(socket_path=temp_socket_path, handler=lambda m, p: asyncio.sleep(0, result="ok"))
    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(str(temp_socket_path))
        writer.write(b"\n   \n\n")
        await writer.drain()

        writer.write(b'{"jsonrpc": "2.0", "method": "test", "params": {}, "id": 1}\n')
        await writer.drain()

        line = await reader.readline()
        res = json.loads(line.decode("utf-8"))
        assert res.get("result") == "ok"
        assert res.get("id") == 1

        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


async def test_ipc_jsonrpc_parse_and_invalid_request_errors(temp_socket_path: Path) -> None:
    """Verify standard JSON-RPC 2.0 error handling for bad JSON and invalid envelopes."""
    server = PenguLinkIPCServer(socket_path=temp_socket_path, handler=lambda m, p: None)
    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(str(temp_socket_path))

        # 1. Malformed JSON
        writer.write(b"{invalid_json\n")
        await writer.drain()
        line = await reader.readline()
        resp = json.loads(line.decode("utf-8"))
        assert resp["error"]["code"] == PARSE_ERROR
        assert resp["id"] is None

        # 2. Missing jsonrpc: "2.0"
        writer.write(b'{"method": "test", "id": 2}\n')
        await writer.drain()
        line = await reader.readline()
        resp = json.loads(line.decode("utf-8"))
        assert resp["error"]["code"] == INVALID_REQUEST

        # 3. Method not a string
        writer.write(b'{"jsonrpc": "2.0", "method": 12345, "id": 3}\n')
        await writer.drain()
        line = await reader.readline()
        resp = json.loads(line.decode("utf-8"))
        assert resp["error"]["code"] == INVALID_REQUEST

        # 4. Params not an object
        writer.write(b'{"jsonrpc": "2.0", "method": "test", "params": [1, 2], "id": 4}\n')
        await writer.drain()
        line = await reader.readline()
        resp = json.loads(line.decode("utf-8"))
        assert resp["error"]["code"] == INVALID_PARAMS

        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


async def test_ipc_domain_error_mapping(temp_socket_path: Path) -> None:
    """Verify domain exceptions are mapped to specific JSON-RPC codes."""
    async def error_handler(method: str, params: Dict[str, Any]) -> Any:
        if method == "not_found":
            raise DeviceNotFoundError("Device 12345 not found")
        elif method == "pairing_fail":
            raise PairingError("Pairing negotiation failed")
        elif method == "param_fail":
            raise ValueError("Invalid argument count")
        elif method == "unknown_meth":
            raise KeyError("Unknown JSON-RPC method 'unknown_meth'")
        raise RuntimeError("General error")

    server = PenguLinkIPCServer(socket_path=temp_socket_path, handler=error_handler)
    client = PenguLinkIPCClient(socket_path=temp_socket_path)

    await server.start()
    try:
        # DeviceNotFoundError -> DEVICE_NOT_FOUND (-32001)
        with pytest.raises(IPCError) as exc_info:
            await client.call("not_found")
        assert f"[{DEVICE_NOT_FOUND}]" in str(exc_info.value)

        # ValueError -> INVALID_PARAMS (-32602)
        with pytest.raises(IPCError) as exc_info:
            await client.call("param_fail")
        assert f"[{INVALID_PARAMS}]" in str(exc_info.value)

        # Unknown method -> METHOD_NOT_FOUND (-32601)
        with pytest.raises(IPCError) as exc_info:
            await client.call("unknown_meth")
        assert f"[{METHOD_NOT_FOUND}]" in str(exc_info.value)

        # General error -> APPLICATION_ERROR (-32000)
        with pytest.raises(IPCError) as exc_info:
            await client.call("other_fail")
        assert f"[{APPLICATION_ERROR}]" in str(exc_info.value)
    finally:
        await server.stop()


async def test_ipc_notification_no_response(temp_socket_path: Path) -> None:
    """Verify JSON-RPC notification (no 'id') produces no response line."""
    notified = False

    async def notify_handler(method: str, params: Dict[str, Any]) -> Any:
        nonlocal notified
        if method == "notify_me":
            notified = True
            return {"ignored": True}

    server = PenguLinkIPCServer(socket_path=temp_socket_path, handler=notify_handler)
    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(str(temp_socket_path))
        writer.write(b'{"jsonrpc": "2.0", "method": "notify_me", "params": {}}\n')
        await writer.drain()

        writer.write(b'{"jsonrpc": "2.0", "method": "notify_me", "params": {}, "id": 99}\n')
        await writer.drain()

        line = await reader.readline()
        resp = json.loads(line.decode("utf-8"))
        assert resp["id"] == 99
        assert notified is True

        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


async def test_ipc_daemon_not_running_error(temp_socket_path: Path) -> None:
    """Verify client raises DaemonNotRunningError when daemon socket does not exist or refuse connection."""
    client = PenguLinkIPCClient(socket_path=temp_socket_path)
    with pytest.raises(DaemonNotRunningError):
        await client.call("ping")


# =====================================================================
# Clipboard IPC Tests
# =====================================================================

async def test_ipc_clipboard_helpers(temp_socket_path: Path) -> None:
    """Verify PenguLinkIPCClient get_clipboard and set_clipboard helper methods."""
    captured: Dict[str, Any] = {}

    async def mock_handler(method: str, params: Dict[str, Any]) -> Any:
        if method == "get_clipboard":
            return {"content": "Stored clip", "timestamp": 123456789, "bytes": 11}
        elif method == "set_clipboard":
            captured["content"] = params.get("content")
            return {"status": "ok", "bytes": len(params.get("content", "").encode("utf-8"))}
        raise KeyError(method)

    server = PenguLinkIPCServer(socket_path=temp_socket_path, handler=mock_handler)
    client = PenguLinkIPCClient(socket_path=temp_socket_path)
    await server.start()
    try:
        get_res = await client.get_clipboard()
        assert get_res["content"] == "Stored clip"
        assert get_res["timestamp"] == 123456789

        set_res = await client.set_clipboard("New remote text")
        assert set_res["status"] == "ok"
        assert captured["content"] == "New remote text"
    finally:
        await server.stop()


async def test_ipc_clipboard_error_mappings(temp_socket_path: Path) -> None:
    """Verify ClipboardError maps to CLIPBOARD_ERROR and ClipboardUnavailableError maps to INTERNAL_ERROR."""
    async def mock_handler(method: str, params: Dict[str, Any]) -> Any:
        if method == "fail_backend":
            raise ClipboardError("Backend execution failure")
        elif method == "fail_unavailable":
            raise ClipboardUnavailableError("Clipboard plugin not active")
        raise KeyError(method)

    server = PenguLinkIPCServer(socket_path=temp_socket_path, handler=mock_handler)
    client = PenguLinkIPCClient(socket_path=temp_socket_path)
    await server.start()
    try:
        with pytest.raises(IPCError) as exc_info:
            await client.call("fail_backend")
        assert f"[{CLIPBOARD_ERROR}]" in str(exc_info.value)

        with pytest.raises(IPCError) as exc_info:
            await client.call("fail_unavailable")
        assert f"[{INTERNAL_ERROR}]" in str(exc_info.value)
    finally:
        await server.stop()


async def test_ipc_set_clipboard_sanitized_exception_logging(
    temp_socket_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify exception logging during set_clipboard does not leak sensitive content."""
    secret = "SUPER_SECRET_CANARY_TOKEN_9999"

    async def mock_handler(method: str, params: Dict[str, Any]) -> Any:
        if method == "set_clipboard":
            raise RuntimeError(f"Crashing with secret {secret}")
        return "ok"

    server = PenguLinkIPCServer(socket_path=temp_socket_path, handler=mock_handler)
    client = PenguLinkIPCClient(socket_path=temp_socket_path)
    await server.start()
    try:
        with caplog.at_level("WARNING"):
            with pytest.raises(IPCError):
                await client.set_clipboard(secret)

        # Assert the exception type was logged but not the secret value
        assert secret not in caplog.text
        assert "RuntimeError" in caplog.text
    finally:
        await server.stop()

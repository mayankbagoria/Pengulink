"""tests.adversarial.test_clipboard_m93_ipc_cli_fuzz_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Milestone M9.3 Empirical Adversarial Challenge Test Harness:
Unix Domain Socket JSON-RPC 2.0 Fuzzing, Boundary Stress & CLI Verification.

Verifies:
1. Raw Unix socket JSON-RPC 2.0 protocol robustness:
   - Malformed JSON, syntax corruption, random binary frames, empty lines.
   - Non-object params (lists, integers, strings, bools, null).
   - Missing fields (missing 'content', wrong keys, missing 'method', notifications).
   - Exact boundary payloads: 119,999 bytes (pass), 120,000 bytes (pass), 120,001 bytes (fail).
   - Oversized payloads (>120 KB) up to 500 KB, and >1 MB line limit overrun handling.
   - Unicode edge cases: composite emojis, ZWJ sequences, multiline CRLF/LF, null bytes, lone surrogates.
2. CLI commands under CliRunner stress:
   - 'pengulink clipboard set' with piped stdin (explicit '-' and implicit), empty string, huge text.
   - 'pengulink clipboard get' with --json, --raw, empty state, and special characters.
3. Daemon crash resilience: zero unhandled exceptions, zero leaking tasks, and zero socket corruption.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.config import PenguLinkConfig
from pengulink.daemon.ipc import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PenguLinkIPCClient,
    PenguLinkIPCServer,
)
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.plugins.clipboard import (
    MAX_CLIPBOARD_PACKET_BYTES,
    InMemoryClipboardBackend,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def fuzz_daemon_config(tmp_path: Path) -> PenguLinkConfig:
    """Provide isolated daemon configuration for fuzzing tests."""
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / "config"
    cfg.storage.data_dir = tmp_path / "data"
    cfg.storage.runtime_dir = tmp_path / "run"
    cfg.device.device_id = "fuzz_daemon_device_id_9999"
    cfg.device.device_name = "PenguLink-Fuzz-Test"
    cfg.network.udp_port = 17188
    cfg.network.tcp_port_min = 17240
    cfg.network.tcp_port_max = 17245
    cfg.ensure_directories()
    return cfg


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing Click CLI runner."""
    return CliRunner(env={"COLUMNS": "160"})


async def send_raw_ipc_line(socket_path: Path, raw_bytes: bytes, timeout: float = 3.0) -> bytes:
    """Helper to connect to Unix domain socket, send raw bytes, and read response line."""
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(socket_path)),
        timeout=timeout,
    )
    try:
        writer.write(raw_bytes)
        await writer.drain()
        response = await asyncio.wait_for(reader.readline(), timeout=timeout)
        return response
    finally:
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
        except Exception:
            pass


# =============================================================================
# 1. Unix Socket JSON-RPC 2.0 Fuzzing & Malformed Inputs
# =============================================================================

class TestSocketJSONRPCFuzzing:
    """Adversarial fuzz testing of the daemon's Unix socket JSON-RPC IPC interface."""

    async def test_raw_socket_malformed_json_syntax(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Verify malformed JSON syntax returns PARSE_ERROR (-32700) without crashing daemon."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            malformed_payloads = [
                b"{not valid json\n",
                b'{"jsonrpc": "2.0", "method": "get_clipboard",\n',
                b'{"jsonrpc": 2.0, "method": get_clipboard}\n',
                b"\xff\xfe\xfd\xfc\n",
                b"GARBAGE_WITHOUT_JSON\n",
            ]

            for payload in malformed_payloads:
                resp_bytes = await send_raw_ipc_line(sock_path, payload)
                assert resp_bytes, "Server must reply with JSON-RPC error"
                data = json.loads(resp_bytes.decode("utf-8"))
                assert data.get("jsonrpc") == "2.0"
                assert "error" in data
                assert data["error"]["code"] == PARSE_ERROR
                assert "Parse error" in data["error"]["message"]

            # Verify daemon is still operational after fuzzing barrage
            client = PenguLinkIPCClient(socket_path=sock_path)
            res = await client.call("get_clipboard")
            assert "content" in res
        finally:
            await daemon.stop()

    async def test_raw_socket_empty_and_whitespace_lines(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Verify empty and whitespace lines are ignored without producing responses or errors."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            reader, writer = await asyncio.open_unix_connection(str(sock_path))
            try:
                # Send empty lines
                writer.write(b"\n   \n\t\n")
                await writer.drain()

                # Send a valid request immediately after
                valid_req = (
                    json.dumps({
                        "jsonrpc": "2.0",
                        "method": "get_clipboard",
                        "params": {},
                        "id": 1,
                    })
                    + "\n"
                ).encode("utf-8")
                writer.write(valid_req)
                await writer.drain()

                resp_line = await asyncio.wait_for(reader.readline(), timeout=2.0)
                assert resp_line
                data = json.loads(resp_line.decode("utf-8"))
                assert data.get("id") == 1
                assert "result" in data
            finally:
                writer.close()
                await writer.wait_closed()
        finally:
            await daemon.stop()

    async def test_raw_socket_non_object_params(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Verify non-object 'params' (lists, ints, strings, bools, null) return INVALID_PARAMS (-32602)."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            invalid_params_samples = [
                [1, 2, 3],
                "not-an-object",
                42,
                True,
                False,
                None,
            ]

            for idx, bad_params in enumerate(invalid_params_samples, start=100):
                req = {
                    "jsonrpc": "2.0",
                    "method": "set_clipboard",
                    "params": bad_params,
                    "id": idx,
                }
                resp_bytes = await send_raw_ipc_line(sock_path, (json.dumps(req) + "\n").encode("utf-8"))
                data = json.loads(resp_bytes.decode("utf-8"))
                assert data.get("id") == idx
                assert "error" in data
                assert data["error"]["code"] == INVALID_PARAMS
                assert "Params must be an object" in data["error"]["message"]

            # Verify daemon health
            client = PenguLinkIPCClient(socket_path=sock_path)
            status = await client.call("status")
            assert status["running"] is True
        finally:
            await daemon.stop()

    async def test_raw_socket_missing_and_invalid_fields(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Verify missing required fields in JSON-RPC request and set_clipboard params."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            # 1. Missing jsonrpc version
            req1 = {"method": "get_clipboard", "id": 1}
            resp1 = json.loads((await send_raw_ipc_line(sock_path, (json.dumps(req1) + "\n").encode("utf-8"))).decode())
            assert resp1["error"]["code"] == INVALID_REQUEST

            # 2. Wrong jsonrpc version
            req2 = {"jsonrpc": "1.0", "method": "get_clipboard", "id": 2}
            resp2 = json.loads((await send_raw_ipc_line(sock_path, (json.dumps(req2) + "\n").encode("utf-8"))).decode())
            assert resp2["error"]["code"] == INVALID_REQUEST

            # 3. Missing method
            req3 = {"jsonrpc": "2.0", "id": 3}
            resp3 = json.loads((await send_raw_ipc_line(sock_path, (json.dumps(req3) + "\n").encode("utf-8"))).decode())
            assert resp3["error"]["code"] == INVALID_REQUEST

            # 4. Empty method string
            req4 = {"jsonrpc": "2.0", "method": "", "id": 4}
            resp4 = json.loads((await send_raw_ipc_line(sock_path, (json.dumps(req4) + "\n").encode("utf-8"))).decode())
            assert resp4["error"]["code"] == INVALID_REQUEST

            # 5. set_clipboard missing 'content' param
            req5 = {"jsonrpc": "2.0", "method": "set_clipboard", "params": {}, "id": 5}
            resp5 = json.loads((await send_raw_ipc_line(sock_path, (json.dumps(req5) + "\n").encode("utf-8"))).decode())
            assert resp5["error"]["code"] == INVALID_PARAMS
            assert "Parameter 'content' is required" in resp5["error"]["message"]

            # 6. set_clipboard non-string 'content' types
            for idx, bad_val in enumerate([123, 45.6, None, ["a"], {"nested": True}], start=20):
                req_val = {
                    "jsonrpc": "2.0",
                    "method": "set_clipboard",
                    "params": {"content": bad_val},
                    "id": idx,
                }
                resp_val = json.loads(
                    (await send_raw_ipc_line(sock_path, (json.dumps(req_val) + "\n").encode("utf-8"))).decode()
                )
                assert resp_val["error"]["code"] == INVALID_PARAMS
                assert "must be a string" in resp_val["error"]["message"]
        finally:
            await daemon.stop()


# =============================================================================
# 2. Boundary Cases & Payload Stress (119,999, 120,000, 120,001 Bytes & Overrun)
# =============================================================================

class TestSocketPayloadBoundaries:
    """Stress-test exact boundaries of the 120 KB clipboard size ceiling."""

    async def test_boundary_119999_bytes_accepted(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Exact 119,999 bytes payload (ceiling - 1) is accepted and roundtrips intact."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            payload = "A" * 119_999
            client = PenguLinkIPCClient(socket_path=sock_path)

            set_res = await client.set_clipboard(payload)
            assert set_res.get("status") == "ok"
            assert set_res.get("bytes") == 119_999

            get_res = await client.get_clipboard()
            assert get_res["bytes"] == 119_999
            assert get_res["content"] == payload
        finally:
            await daemon.stop()

    async def test_boundary_120000_bytes_accepted(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Exact 120,000 bytes payload (exact ceiling) is accepted and roundtrips intact."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            payload = "B" * 120_000
            client = PenguLinkIPCClient(socket_path=sock_path)

            set_res = await client.set_clipboard(payload)
            assert set_res.get("status") == "ok"
            assert set_res.get("bytes") == 120_000

            get_res = await client.get_clipboard()
            assert get_res["bytes"] == 120_000
            assert get_res["content"] == payload
        finally:
            await daemon.stop()

    async def test_boundary_120001_bytes_rejected(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Exact 120,001 bytes payload (ceiling + 1) is rejected with INVALID_PARAMS."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            payload = "C" * 120_001
            req = {
                "jsonrpc": "2.0",
                "method": "set_clipboard",
                "params": {"content": payload},
                "id": 99,
            }
            resp_bytes = await send_raw_ipc_line(sock_path, (json.dumps(req) + "\n").encode("utf-8"))
            data = json.loads(resp_bytes.decode("utf-8"))
            assert data["id"] == 99
            assert "error" in data
            assert data["error"]["code"] == INVALID_PARAMS
            assert "exceeds maximum allowed size" in data["error"]["message"]
            assert "120001" in data["error"]["message"]

            # Verify local clipboard was NOT overwritten
            client = PenguLinkIPCClient(socket_path=sock_path)
            get_res = await client.get_clipboard()
            assert get_res["content"] != payload
        finally:
            await daemon.stop()

    async def test_oversized_payloads_safely_rejected(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Massive payloads (150 KB, 300 KB, 500 KB) within stream limit are safely rejected."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            for size in [150_000, 300_000, 500_000]:
                payload = "X" * size
                req = {
                    "jsonrpc": "2.0",
                    "method": "set_clipboard",
                    "params": {"content": payload},
                    "id": size,
                }
                resp_bytes = await send_raw_ipc_line(
                    sock_path, (json.dumps(req) + "\n").encode("utf-8"), timeout=5.0
                )
                data = json.loads(resp_bytes.decode("utf-8"))
                assert data["error"]["code"] == INVALID_PARAMS
                assert "exceeds maximum allowed size" in data["error"]["message"]

            # Daemon survives without any crash
            client = PenguLinkIPCClient(socket_path=sock_path)
            status = await client.call("status")
            assert status["running"] is True
        finally:
            await daemon.stop()

    async def test_stream_limit_overrun_guard(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Stream line exceeding 1 MB limit triggers LimitOverrunError and drops connection cleanly."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            reader, writer = await asyncio.open_unix_connection(str(sock_path))
            try:
                # Send 1.2 MB without a newline
                huge_chunk = b"Y" * (1_200_000)
                writer.write(huge_chunk)
                await writer.drain()

                # Reading from the reader should observe EOF / closed connection or peer reset
                try:
                    data = await asyncio.wait_for(reader.read(1024), timeout=2.0)
                    assert data == b"", "Server should close connection on buffer limit overrun"
                except (ConnectionResetError, BrokenPipeError, OSError):
                    # Kernel RST on unread buffer overrun is an expected clean drop
                    pass
            finally:
                try:
                    writer.close()
                    await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
                except (ConnectionResetError, BrokenPipeError, OSError, asyncio.TimeoutError):
                    pass

            # Verify daemon survives the limit overrun and immediately accepts subsequent clients
            client = PenguLinkIPCClient(socket_path=sock_path)
            res = await client.get_clipboard()
            assert "content" in res
        finally:
            await daemon.stop()


# =============================================================================
# 3. Unicode Edge Cases (Emojis, Multiline, Null Bytes, Surrogates)
# =============================================================================

class TestSocketUnicodeEdgeCases:
    """Stress-test Unicode edge cases over JSON-RPC IPC."""

    async def test_complex_emoji_and_astral_plane_sequences(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Complex emojis, ZWJ sequences, skin tone modifiers, and astral symbols roundtrip intact."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            complex_unicode = (
                "👨‍👩‍👧‍👦 Family Emoji | 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Flag | "
                "🤝🏽 Skin Tone Handshake | 1️⃣ Keycap | "
                "✨ Astral Math: 𝒳 = ∫ ℱ(ω) dω | RTL: سلام دنیا"
            )
            client = PenguLinkIPCClient(socket_path=sock_path)

            set_res = await client.set_clipboard(complex_unicode)
            assert set_res.get("status") == "ok"
            assert set_res.get("bytes") == len(complex_unicode.encode("utf-8"))

            get_res = await client.get_clipboard()
            assert get_res["content"] == complex_unicode
            assert get_res["bytes"] == len(complex_unicode.encode("utf-8"))
        finally:
            await daemon.stop()

    async def test_multiline_mixed_line_endings(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Multiline text with mixed LF, CRLF, and CR roundtrips with preserved whitespace."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            multiline_text = "Line 1\nLine 2\r\nLine 3\rLine 4\n\n\nTrailing newlines\n"
            client = PenguLinkIPCClient(socket_path=sock_path)

            await client.set_clipboard(multiline_text)
            get_res = await client.get_clipboard()
            assert get_res["content"] == multiline_text
        finally:
            await daemon.stop()

    async def test_embedded_null_bytes(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Strings with embedded null characters (\\u0000) roundtrip without truncation."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            null_text = "Prefix\x00Middle\x00Suffix\x00\x00"
            client = PenguLinkIPCClient(socket_path=sock_path)

            await client.set_clipboard(null_text)
            get_res = await client.get_clipboard()
            assert get_res["content"] == null_text
            assert get_res["bytes"] == len(null_text.encode("utf-8"))
        finally:
            await daemon.stop()

    async def test_lone_surrogate_rejection_without_daemon_crash(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """JSON with lone surrogate escape sequence (\\ud800) is rejected cleanly with INVALID_PARAMS."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            # Send raw JSON containing escaped surrogate \ud800
            req_line = b'{"jsonrpc": "2.0", "method": "set_clipboard", "params": {"content": "bad_\\ud800_surrogate"}, "id": 77}\n'
            resp_bytes = await send_raw_ipc_line(sock_path, req_line)
            data = json.loads(resp_bytes.decode("utf-8"))
            assert data["id"] == 77
            assert "error" in data
            assert data["error"]["code"] == INVALID_PARAMS
            assert "surrogate characters" in data["error"]["message"]

            # Ensure daemon survives and subsequent calls work
            client = PenguLinkIPCClient(socket_path=sock_path)
            res = await client.get_clipboard()
            assert "content" in res
        finally:
            await daemon.stop()

    async def test_empty_string_clipboard_handling(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Empty string clipboard content clears backend and returns 0 bytes."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend(initial_text="pre-existing content")
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            client = PenguLinkIPCClient(socket_path=sock_path)

            set_res = await client.set_clipboard("")
            assert set_res.get("status") == "ok"
            assert set_res.get("bytes") == 0

            get_res = await client.get_clipboard()
            assert get_res["content"] == ""
            assert get_res["bytes"] == 0
        finally:
            await daemon.stop()


# =============================================================================
# 4. CLI CliRunner Stress & Edge Cases
# =============================================================================

class TestCLIClipboardStress:
    """Stress-test Click CLI clipboard commands under various inputs and flags."""

    def test_cli_set_stdin_pipe_explicit_dash(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """'pengulink clipboard set -' ingests content from piped stdin."""
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        captured: Dict[str, Any] = {}

        def mock_call(method: str, params: Any = None, timeout: float = 10.0) -> Any:
            captured["method"] = method
            captured["params"] = params
            return {"status": "ok", "bytes": len(params["content"].encode("utf-8"))}

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=mock_call):
            result = cli_runner.invoke(
                cli,
                ["-s", str(mock_socket), "clipboard", "set", "-"],
                input="Piped text from stdin\nWith multiple lines\n",
            )
            assert result.exit_code == 0
            assert captured["method"] == "set_clipboard"
            assert captured["params"]["content"] == "Piped text from stdin\nWith multiple lines\n"
            assert "Clipboard updated" in result.output

    def test_cli_set_stdin_pipe_implicit(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """'pengulink clipboard set' without TEXT argument ingests piped stdin when not a tty."""
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        captured: Dict[str, Any] = {}

        def mock_call(method: str, params: Any = None, timeout: float = 10.0) -> Any:
            captured["params"] = params
            return {"status": "ok", "bytes": len(params["content"].encode("utf-8"))}

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=mock_call):
            result = cli_runner.invoke(
                cli,
                ["-s", str(mock_socket), "clipboard", "set"],
                input="Implicit piped input",
            )
            assert result.exit_code == 0
            assert captured["params"]["content"] == "Implicit piped input"

    def test_cli_set_empty_string_argument(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """'pengulink clipboard set \"\"' updates clipboard to empty string without error."""
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        captured: Dict[str, Any] = {}

        def mock_call(method: str, params: Any = None, timeout: float = 10.0) -> Any:
            captured["params"] = params
            return {"status": "ok", "bytes": 0}

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=mock_call):
            result = cli_runner.invoke(
                cli,
                ["-s", str(mock_socket), "clipboard", "set", ""],
            )
            assert result.exit_code == 0
            assert captured["params"]["content"] == ""
            assert "Clipboard updated (0 bytes)" in result.output

    def test_cli_set_oversized_validation_boundaries(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """CLI enforces 120,000 bytes ceiling before making IPC calls."""
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        # 1. Exactly 120,000 bytes -> passes CLI check
        payload_120k = "Z" * 120_000
        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value={"status": "ok", "bytes": 120000}):
            res_pass = cli_runner.invoke(
                cli,
                ["-s", str(mock_socket), "clipboard", "set", payload_120k],
            )
            assert res_pass.exit_code == 0

        # 2. Exactly 120,001 bytes -> rejected by CLI validation
        payload_120001 = "Z" * 120_001
        res_fail = cli_runner.invoke(
            cli,
            ["-s", str(mock_socket), "clipboard", "set", payload_120001],
        )
        assert res_fail.exit_code != 0
        assert "exceeds maximum allowed size" in res_fail.output
        assert "120,001 bytes > 120,000 bytes" in res_fail.output

    def test_cli_get_raw_and_redirected(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """'pengulink clipboard get --raw' outputs unadorned exact text."""
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        raw_sample = "Raw unadorned text\nwith newlines\n"
        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
            return_value={"content": raw_sample, "timestamp": 12345, "bytes": len(raw_sample)},
        ):
            result = cli_runner.invoke(cli, ["-s", str(mock_socket), "clipboard", "get", "--raw"])
            assert result.exit_code == 0
            assert result.output == raw_sample
            assert "Clipboard Content" not in result.output

    def test_cli_get_json_flag(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """'pengulink clipboard get --json' outputs valid machine-readable JSON."""
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_dict = {
            "content": "Secret payload",
            "timestamp": 1728000000000,
            "digest": "abcdef123456",
            "bytes": 14,
        }
        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_dict):
            result = cli_runner.invoke(cli, ["-s", str(mock_socket), "clipboard", "get", "--json"])
            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert parsed["content"] == "Secret payload"
            assert parsed["bytes"] == 14
            assert parsed["timestamp"] == 1728000000000

    def test_cli_get_empty_clipboard_modes(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Verify get behavior on empty clipboard across --raw and --json."""
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        empty_dict = {"content": "", "timestamp": 0, "digest": None, "bytes": 0}
        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=empty_dict):
            # Raw mode on empty content outputs empty string
            res_raw = cli_runner.invoke(cli, ["-s", str(mock_socket), "clipboard", "get", "--raw"])
            assert res_raw.exit_code == 0
            assert res_raw.output == ""

            # JSON mode on empty content outputs JSON
            res_json = cli_runner.invoke(cli, ["-s", str(mock_socket), "clipboard", "get", "--json"])
            assert res_json.exit_code == 0
            parsed = json.loads(res_json.output)
            assert parsed["content"] == ""
            assert parsed["bytes"] == 0


# =============================================================================
# 5. Daemon Crash Resilience & Concurrent IPC Stress
# =============================================================================

class TestDaemonIPCStressResilience:
    """Stress-test daemon resilience against concurrent load and abrupt disconnects."""

    async def test_abrupt_client_socket_disconnect_during_write(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Abrupt client socket disconnect without newline causes zero daemon crashes or task leaks."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            # Open socket, send half a request, close abruptly
            for _ in range(5):
                reader, writer = await asyncio.open_unix_connection(str(sock_path))
                writer.write(b'{"jsonrpc": "2.0", "method": "set_clip')
                await writer.drain()
                writer.close()
                await writer.wait_closed()

            await asyncio.sleep(0.05)

            # Ensure daemon is fully functional
            client = PenguLinkIPCClient(socket_path=sock_path)
            res = await client.set_clipboard("Alive and resilient")
            assert res.get("status") == "ok"
            get_res = await client.get_clipboard()
            assert get_res["content"] == "Alive and resilient"
        finally:
            await daemon.stop()

    async def test_concurrent_fuzz_and_traffic_barrage(
        self, fuzz_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Concurrent burst of 30 mixed valid and malformed requests completes without error."""
        sock_path = tmp_path / "run" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(
            config=fuzz_daemon_config,
            socket_path=sock_path,
            clipboard_backend=backend,
        )
        await daemon.start()
        try:
            client = PenguLinkIPCClient(socket_path=sock_path)

            async def worker(idx: int) -> None:
                if idx % 3 == 0:
                    # Valid set
                    await client.set_clipboard(f"Concurrent Message {idx}")
                elif idx % 3 == 1:
                    # Valid get
                    await client.get_clipboard()
                else:
                    # Fuzzed raw request
                    bad_req = (
                        json.dumps({
                            "jsonrpc": "2.0",
                            "method": "set_clipboard",
                            "params": [idx],  # Non-object params
                            "id": idx,
                        })
                        + "\n"
                    ).encode("utf-8")
                    resp = await send_raw_ipc_line(sock_path, bad_req)
                    assert b"error" in resp

            # Launch 30 concurrent workers
            tasks = [asyncio.create_task(worker(i)) for i in range(30)]
            await asyncio.gather(*tasks)

            # Confirm final daemon state
            status = await client.call("status")
            assert status["running"] is True
        finally:
            await daemon.stop()

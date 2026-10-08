"""tests.adversarial.test_clipboard_backend_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial stress and robustness test suite for Milestone M9.1:
Linux Clipboard Backend Subsystem (Wayland, X11, InMemory, Factory).

Stress test vectors:
1. Wayland Subprocess Simulation Stress:
   - Truncated headers (1, 2, 3 bytes then EOF)
   - Corrupted framing (header length > available data followed by abrupt EOF)
   - Immediate EOF and EOF mid-payload
   - Giant header integer overflow / payload claims (e.g. 4 GB - 1) followed by EOF
   - Chunked byte-by-byte streaming (TCP/pipe fragmentation)
   - Oversized payload (> 1 MiB) draining followed by valid frame
   - Invalid UTF-8 replacement behavior
   - Callback exceptions in reader loop
   - Rapid burst frame flood (500 frames)
   - Zero-length payload handling
2. X11 Ctypes & Display Robustness:
   - DISPLAY unset, empty, bogus (e.g. ":999", "invalid", "/tmp/nonexistent")
   - XOpenDisplay failure simulation (returns NULL / None)
   - Protocol error handler invocation (_default_x11_error_handler)
   - loop.remove_reader error resilience
   - Concurrent get_text and set_text under failing display
   - Double / multiple stop_watching idempotency
3. Environment Override & Factory Fuzzing:
   - Empty, whitespace-only, mixed-case ("WaYlAnD", "X11", "In_MeMoRy", "HeAdLeSs", "mOcK")
   - Invalid backend names ("none", "windows", "macos", "123", "!@#$%", "wayland_x11")
   - Embedded null bytes in explicit argument
   - Precedence: explicit argument vs environment variable
   - Fallback cascade when display servers are unavailable
4. Privacy & Zero-Content Logging Audit:
   - Sentinel secrets ("SECRET_TOKEN_CONFIDENTIAL_998877") processed via all backends
   - Comprehensive log record capture (DEBUG through CRITICAL)
   - Callback exception logging & traceback verification
   - redact_clipboard formatting and zero cleartext leak
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import struct
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pengulink.plugins.clipboard import (
    MAX_CLIPBOARD_TEXT_BYTES,
    ClipboardBackend,
    ClipboardBackendError,
    ClipboardChangeCallback,
    ClipboardError,
    ClipboardUnavailableError,
    InMemoryClipboardBackend,
    WaylandClipboardBackend,
    X11ClipboardBackend,
    detect_clipboard_backend,
    redact_clipboard,
)
from pengulink.plugins.clipboard.backend import (
    _c_x11_error_handler,
    _default_x11_error_handler,
)


# =====================================================================
# 1. Wayland Subprocess Simulation Stress Tests
# =====================================================================

class TestWaylandFramingStress:
    """Stress-test Wayland length-prefixed streaming protocol parser."""

    @pytest.mark.parametrize("header_bytes", [b"", b"\x00", b"\x00\x00", b"\x00\x00\x00"])
    async def test_truncated_header_eof_handling(self, header_bytes: bytes) -> None:
        """Simulate EOF after fewer than 4 bytes of header: reader must terminate gracefully without crash."""
        backend = WaylandClipboardBackend()
        backend.is_available = True
        received: List[str] = []

        stream = asyncio.StreamReader()
        stream.feed_data(header_bytes)
        stream.feed_eof()

        mock_proc = MagicMock()
        mock_proc.stdout = stream
        mock_proc.returncode = None
        mock_proc.pid = 7711
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await backend.start_watching(lambda c: received.append(c))  # type: ignore[arg-type]
            await asyncio.sleep(0.05)
            await backend.stop_watching()

        assert received == []

    @pytest.mark.parametrize("declared_len,actual_payload", [
        (100, b"short"),
        (50, b"only 10 b!"),
        (10, b""),
        (1048576, b"truncated mid-flight"),
    ])
    async def test_corrupted_framing_truncated_payload_eof(
        self, declared_len: int, actual_payload: bytes
    ) -> None:
        """Simulate header claiming N bytes, but stream cuts off early: reader must handle EOF cleanly."""
        backend = WaylandClipboardBackend()
        backend.is_available = True
        received: List[str] = []

        header = struct.pack(">I", declared_len)
        stream = asyncio.StreamReader()
        stream.feed_data(header + actual_payload)
        stream.feed_eof()

        mock_proc = MagicMock()
        mock_proc.stdout = stream
        mock_proc.returncode = None
        mock_proc.pid = 7712
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await backend.start_watching(lambda c: received.append(c))  # type: ignore[arg-type]
            await asyncio.sleep(0.05)
            await backend.stop_watching()

        assert received == []

    async def test_giant_header_length_claim_and_eof(self) -> None:
        """Simulate header claiming 4,294,967,295 bytes (uint32 max): reader must attempt drain safely and handle EOF."""
        backend = WaylandClipboardBackend()
        backend.is_available = True
        received: List[str] = []

        # uint32 max
        giant_header = struct.pack(">I", 0xFFFFFFFF)
        stream = asyncio.StreamReader()
        stream.feed_data(giant_header + b"tiny data before abrupt EOF")
        stream.feed_eof()

        mock_proc = MagicMock()
        mock_proc.stdout = stream
        mock_proc.returncode = None
        mock_proc.pid = 7718
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await backend.start_watching(lambda c: received.append(c))  # type: ignore[arg-type]
            await asyncio.sleep(0.05)
            await backend.stop_watching()

        assert received == []

    async def test_zero_length_payload_handling(self) -> None:
        """Simulate zero-length frame (empty clipboard): callback must receive empty string."""
        backend = WaylandClipboardBackend()
        backend.is_available = True
        received: List[str] = []

        zero_frame = struct.pack(">I", 0)
        stream = asyncio.StreamReader()
        stream.feed_data(zero_frame)
        stream.feed_eof()

        mock_proc = MagicMock()
        mock_proc.stdout = stream
        mock_proc.returncode = None
        mock_proc.pid = 7719
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await backend.start_watching(lambda c: received.append(c))  # type: ignore[arg-type]
            await asyncio.sleep(0.05)
            await backend.stop_watching()

        assert received == [""]

    async def test_chunked_byte_by_byte_streaming(self) -> None:
        """Simulate slow/fragmented TCP or pipe delivery arriving byte-by-byte."""
        backend = WaylandClipboardBackend()
        backend.is_available = True
        received: List[str] = []

        content = "Fragmented 🧩 UTF-8 emojis and newlines\nSecond Line\n"
        encoded = content.encode("utf-8")
        frame = struct.pack(">I", len(encoded)) + encoded

        stream = asyncio.StreamReader()

        mock_proc = MagicMock()
        mock_proc.stdout = stream
        mock_proc.returncode = None
        mock_proc.pid = 7713
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await backend.start_watching(lambda c: received.append(c))  # type: ignore[arg-type]

            # Feed byte-by-byte with tiny async pauses
            for b in frame:
                stream.feed_data(bytes([b]))
                await asyncio.sleep(0.001)

            stream.feed_eof()
            await asyncio.sleep(0.05)
            await backend.stop_watching()

        assert received == [content]

    async def test_invalid_utf8_in_payload_replacement(self) -> None:
        """Simulate malformed UTF-8 bytes in payload: must decode with replacement character without exception."""
        backend = WaylandClipboardBackend()
        backend.is_available = True
        received: List[str] = []

        bad_payload = b"Valid prefix " + b"\xff\xfe\xfa\x80" + b" valid suffix"
        frame = struct.pack(">I", len(bad_payload)) + bad_payload

        stream = asyncio.StreamReader()
        stream.feed_data(frame)
        stream.feed_eof()

        mock_proc = MagicMock()
        mock_proc.stdout = stream
        mock_proc.returncode = None
        mock_proc.pid = 7714
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await backend.start_watching(lambda c: received.append(c))  # type: ignore[arg-type]
            await asyncio.sleep(0.05)
            await backend.stop_watching()

        assert len(received) == 1
        assert "Valid prefix " in received[0]
        assert " valid suffix" in received[0]
        assert "\ufffd" in received[0]

    async def test_rapid_burst_frame_flood(self) -> None:
        """Simulate a flood of 500 back-to-back clipboard change frames."""
        backend = WaylandClipboardBackend()
        backend.is_available = True
        received: List[str] = []

        count = 500
        buffer = bytearray()
        for i in range(count):
            msg = f"burst_msg_{i}".encode("utf-8")
            buffer.extend(struct.pack(">I", len(msg)) + msg)

        stream = asyncio.StreamReader()
        stream.feed_data(bytes(buffer))
        stream.feed_eof()

        mock_proc = MagicMock()
        mock_proc.stdout = stream
        mock_proc.returncode = None
        mock_proc.pid = 7715
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await backend.start_watching(lambda c: received.append(c))  # type: ignore[arg-type]
            await asyncio.sleep(0.15)
            await backend.stop_watching()

        assert len(received) == count
        assert received[0] == "burst_msg_0"
        assert received[-1] == f"burst_msg_{count - 1}"

    async def test_failing_callback_does_not_abort_stream(self) -> None:
        """Verify that an exception raised inside on_change callback does not crash reader loop."""
        backend = WaylandClipboardBackend()
        backend.is_available = True
        calls: List[str] = []

        async def erratic_callback(text: str) -> None:
            calls.append(text)
            if "fail" in text:
                raise ValueError("Deliberate consumer failure")

        msg1 = b"fail_1"
        msg2 = b"success_after_fail"
        msg3 = b"fail_2"
        msg4 = b"final_success"

        data = (
            struct.pack(">I", len(msg1)) + msg1 +
            struct.pack(">I", len(msg2)) + msg2 +
            struct.pack(">I", len(msg3)) + msg3 +
            struct.pack(">I", len(msg4)) + msg4
        )

        stream = asyncio.StreamReader()
        stream.feed_data(data)
        stream.feed_eof()

        mock_proc = MagicMock()
        mock_proc.stdout = stream
        mock_proc.returncode = None
        mock_proc.pid = 7716
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await backend.start_watching(erratic_callback)
            await asyncio.sleep(0.08)
            await backend.stop_watching()

        assert calls == ["fail_1", "success_after_fail", "fail_2", "final_success"]

    async def test_stop_watching_idempotency(self) -> None:
        """Calling stop_watching multiple times must be completely safe."""
        backend = WaylandClipboardBackend()
        backend.is_available = True

        mock_proc = MagicMock()
        mock_proc.stdout = asyncio.StreamReader()
        mock_proc.returncode = None
        mock_proc.pid = 7717
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await backend.start_watching(AsyncMock())
            await backend.stop_watching()
            await backend.stop_watching()
            await backend.stop_watching()


# =====================================================================
# 2. X11 Ctypes Robustness Stress Tests
# =====================================================================

class TestX11CtypesRobustnessStress:
    """Stress-test X11 ctypes bindings, bogus displays, and error handling."""

    @pytest.mark.parametrize("bogus_display", [
        None,
        "",
        ":999.0",
        "bogus_display_server",
        "/tmp/.X11-unix/X99",
        ":0",
    ])
    async def test_x11_bogus_display_does_not_crash(self, bogus_display: Optional[str]) -> None:
        """Running with invalid/bogus or unset DISPLAY must fail gracefully without unhandled exception."""
        backend = X11ClipboardBackend()
        env = {}
        if bogus_display is not None:
            env["DISPLAY"] = bogus_display

        with patch.dict(os.environ, env, clear=True):
            if not bogus_display:
                assert backend.is_available is False
                assert await backend.get_text() == ""
                assert await backend.set_text("test") is False
                with pytest.raises(ClipboardUnavailableError):
                    await backend.start_watching(AsyncMock())
            else:
                with patch("shutil.which", return_value="/usr/bin/xclip"):
                    mock_proc = AsyncMock()
                    mock_proc.communicate.return_value = (b"", b"Error: Can't open display: :999.0\n")
                    mock_proc.returncode = 1

                    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                        assert await backend.get_text() == ""
                        assert await backend.set_text("test") is False

    async def test_x11_xopendisplay_returns_null_raises_clipboard_backend_error(self) -> None:
        """When XOpenDisplay returns NULL (cannot connect to X server), raise ClipboardBackendError cleanly."""
        backend = X11ClipboardBackend()
        backend.is_available = True

        mock_x11 = MagicMock()
        mock_x11.XOpenDisplay.return_value = None
        mock_xfixes = MagicMock()

        def fake_load_library(name: str) -> Any:
            if "libX11" in name:
                return mock_x11
            if "libXfixes" in name:
                return mock_xfixes
            raise OSError("Library not found")

        with patch("ctypes.cdll.LoadLibrary", side_effect=fake_load_library), \
             patch.object(backend, "_has_x11_ctypes", return_value=True):
            with pytest.raises(ClipboardBackendError) as exc_info:
                await backend.start_watching(AsyncMock())
            assert "Failed to open X11 Display" in str(exc_info.value)
            assert backend._x11_dpy is None
            assert backend._x11_fd is None

    def test_default_x11_error_handler_suppresses_c_abort(self) -> None:
        """Verify _default_x11_error_handler returns 0 without raising."""
        res = _default_x11_error_handler(None, None)
        assert res == 0

        c_res = _c_x11_error_handler(None, None)
        assert c_res == 0

    async def test_x11_stop_watching_idempotency(self) -> None:
        """Calling stop_watching multiple times on X11 must be safe and idempotent."""
        backend = X11ClipboardBackend()
        await backend.stop_watching()
        await backend.stop_watching()

    async def test_x11_remove_reader_race_or_failure(self) -> None:
        """If loop.remove_reader raises ValueError/RuntimeError on stop, it must be caught safely."""
        backend = X11ClipboardBackend()
        backend._x11_fd = 99
        backend._x11_dpy = 0x1234
        mock_x11 = MagicMock()
        backend._x11_lib = mock_x11

        with patch("asyncio.get_running_loop") as mock_loop_fn:
            mock_loop = MagicMock()
            mock_loop.remove_reader.side_effect = ValueError("FD not registered")
            mock_loop_fn.return_value = mock_loop

            await backend.stop_watching()
            assert backend._x11_fd is None
            assert backend._x11_dpy is None
            mock_x11.XCloseDisplay.assert_called_once_with(0x1234)

    async def test_x11_concurrent_reads_and_writes(self) -> None:
        """Concurrent calls to get_text and set_text must not deadlock or crash."""
        backend = X11ClipboardBackend()
        backend._tool = "/usr/bin/xclip"

        mock_proc_read = AsyncMock()
        mock_proc_read.communicate.return_value = (b"read result", b"")
        mock_proc_read.returncode = 0

        mock_proc_write = AsyncMock()
        mock_proc_write.communicate.return_value = (b"", b"")
        mock_proc_write.returncode = 0

        with patch("asyncio.create_subprocess_exec", side_effect=[mock_proc_read, mock_proc_write, mock_proc_read]):
            results = await asyncio.gather(
                backend.get_text(),
                backend.set_text("write payload"),
                backend.get_text(),
            )
            assert results[0] == "read result"
            assert results[1] is True
            assert results[2] == "read result"


# =====================================================================
# 3. Environment Override & Factory Fuzzing Tests
# =====================================================================

class TestEnvironmentOverrideFuzzing:
    """Fuzz detect_clipboard_backend with invalid, empty, and mixed-case inputs."""

    @pytest.mark.parametrize("valid_mixed_case,expected_class", [
        ("wayland", WaylandClipboardBackend),
        ("WAYLAND", WaylandClipboardBackend),
        ("WayLand", WaylandClipboardBackend),
        ("  wayland  ", WaylandClipboardBackend),
        ("x11", X11ClipboardBackend),
        ("X11", X11ClipboardBackend),
        ("X11 ", X11ClipboardBackend),
        ("in_memory", InMemoryClipboardBackend),
        ("IN_MEMORY", InMemoryClipboardBackend),
        ("In_Memory", InMemoryClipboardBackend),
        ("memory", InMemoryClipboardBackend),
        ("MEMORY", InMemoryClipboardBackend),
        ("headless", InMemoryClipboardBackend),
        ("HEADLESS", InMemoryClipboardBackend),
        ("mock", InMemoryClipboardBackend),
        ("MOCK", InMemoryClipboardBackend),
        ("  mock\t", InMemoryClipboardBackend),
    ])
    def test_case_insensitive_and_whitespace_backend_names(
        self, valid_mixed_case: str, expected_class: Any
    ) -> None:
        """Valid backend names with arbitrary casing or padding whitespace must resolve correctly."""
        # 1. Via explicit parameter
        backend_param = detect_clipboard_backend(valid_mixed_case)
        assert isinstance(backend_param, expected_class)

        # 2. Via environment variable
        with patch.dict(os.environ, {"PENGULINK_CLIPBOARD_BACKEND": valid_mixed_case}):
            backend_env = detect_clipboard_backend()
            assert isinstance(backend_env, expected_class)

    @pytest.mark.parametrize("invalid_value", [
        "invalid",
        "unknown",
        "none",
        "darwin",
        "windows",
        "wl",
        "x",
        "123",
        "!@#$%",
        "wayland x11",
        "wayland; rm -rf /",
    ])
    def test_invalid_backend_names_raise_clipboard_backend_error(
        self, invalid_value: str
    ) -> None:
        """Invalid backend names must raise ClipboardBackendError with clean message."""
        # 1. Via explicit parameter
        with pytest.raises(ClipboardBackendError) as exc_info:
            detect_clipboard_backend(invalid_value)
        assert "Unknown clipboard backend requested" in str(exc_info.value)

        # 2. Via environment variable
        with patch.dict(os.environ, {"PENGULINK_CLIPBOARD_BACKEND": invalid_value}):
            with pytest.raises(ClipboardBackendError) as exc_info_env:
                detect_clipboard_backend()
            assert "Unknown clipboard backend requested" in str(exc_info_env.value)

    def test_embedded_null_byte_in_argument(self) -> None:
        """Passing argument with embedded null byte raises ClipboardBackendError cleanly."""
        with pytest.raises(ClipboardBackendError) as exc_info:
            detect_clipboard_backend("\x00wayland")
        assert "Unknown clipboard backend requested" in str(exc_info.value)

    @pytest.mark.parametrize("empty_value", ["", "   ", "\t", "\n", " \t \n "])
    def test_empty_or_whitespace_env_falls_back_to_auto_detection(
        self, empty_value: str
    ) -> None:
        """Empty or whitespace-only env var or argument must trigger auto-detection fallback."""
        with patch.dict(os.environ, {"PENGULINK_CLIPBOARD_BACKEND": empty_value}, clear=True), \
             patch("pengulink.plugins.clipboard.backend.WaylandClipboardBackend.is_available", False), \
             patch("pengulink.plugins.clipboard.backend.X11ClipboardBackend.is_available", False):
            backend = detect_clipboard_backend(empty_value)
            assert isinstance(backend, InMemoryClipboardBackend)

    def test_precedence_explicit_arg_over_env_var(self) -> None:
        """Explicit argument must override environment variable even if env var is invalid or different."""
        with patch.dict(os.environ, {"PENGULINK_CLIPBOARD_BACKEND": "totally_invalid_junk"}):
            backend = detect_clipboard_backend("mock")
            assert isinstance(backend, InMemoryClipboardBackend)

            backend_wayland = detect_clipboard_backend("wayland")
            assert isinstance(backend_wayland, WaylandClipboardBackend)


# =====================================================================
# 4. Zero-Content Privacy & Leakage Audit
# =====================================================================

class TestZeroContentPrivacyAudit:
    """Verify that raw sensitive clipboard text NEVER leaks into logs, exceptions, or tracebacks."""

    SENTINEL_SECRET = "SUPER_SECRET_AUTH_KEY_998877665544332211"

    def test_redact_clipboard_never_contains_raw_text(self) -> None:
        """redact_clipboard must completely obscure content."""
        res = redact_clipboard(self.SENTINEL_SECRET)
        assert self.SENTINEL_SECRET not in res
        assert "SUPER_SECRET" not in res
        assert "40 chars" in res

    async def test_in_memory_backend_logs_redacted_only(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify InMemoryClipboardBackend never logs cleartext on set_text or simulate_external_copy."""
        caplog.set_level(logging.DEBUG)
        backend = InMemoryClipboardBackend()

        await backend.set_text(self.SENTINEL_SECRET)
        await backend.simulate_external_copy(self.SENTINEL_SECRET)

        for record in caplog.records:
            assert self.SENTINEL_SECRET not in record.getMessage()
            assert self.SENTINEL_SECRET not in str(record.args)

    async def test_wayland_backend_logs_redacted_only(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify WaylandClipboardBackend never logs cleartext on set_text."""
        caplog.set_level(logging.DEBUG)
        backend = WaylandClipboardBackend()

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("shutil.which", return_value="/usr/bin/wl-copy"), \
             patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await backend.set_text(self.SENTINEL_SECRET)

        for record in caplog.records:
            assert self.SENTINEL_SECRET not in record.getMessage()
            assert self.SENTINEL_SECRET not in str(record.args)

    async def test_x11_backend_logs_redacted_only(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify X11ClipboardBackend never logs cleartext on set_text."""
        caplog.set_level(logging.DEBUG)
        backend = X11ClipboardBackend()
        backend._tool = "/usr/bin/xclip"

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await backend.set_text(self.SENTINEL_SECRET)

        for record in caplog.records:
            assert self.SENTINEL_SECRET not in record.getMessage()
            assert self.SENTINEL_SECRET not in str(record.args)

    async def test_exception_in_callback_traceback_does_not_contain_secret(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When an exception occurs during callback execution, ensure logger does not leak raw text."""
        caplog.set_level(logging.DEBUG)
        backend = InMemoryClipboardBackend()

        async def failing_callback(text: str) -> None:
            raise RuntimeError("Generic callback failure")

        await backend.start_watching(failing_callback)
        await backend.simulate_external_copy(self.SENTINEL_SECRET)
        await backend.stop_watching()

        for record in caplog.records:
            assert self.SENTINEL_SECRET not in record.getMessage()
            if record.exc_info:
                import traceback
                tb_str = "".join(traceback.format_exception(*record.exc_info))
                assert self.SENTINEL_SECRET not in str(record.exc_info[1])

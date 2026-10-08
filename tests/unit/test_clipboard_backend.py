"""tests.unit.test_clipboard_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive unit test suite for Milestone M9.1:
Linux Clipboard Backend Subsystem (ABC, Wayland, X11, InMemory, Factory).
"""

from __future__ import annotations

import asyncio
import os
import signal
import struct
import sys
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
    DEFAULT_SUBPROCESS_TIMEOUT,
    SUBPROCESS_TERMINATE_TIMEOUT,
)


# =====================================================================
# 1. ABC Contract Verification
# =====================================================================

class TestClipboardBackendABC:
    """Validate ClipboardBackend abstract base class contract."""

    def test_cannot_instantiate_abc(self) -> None:
        """Verify ClipboardBackend cannot be directly instantiated."""
        with pytest.raises(TypeError):
            ClipboardBackend()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_all_abstract_methods(self) -> None:
        """Verify incomplete subclasses cannot be instantiated."""
        class IncompleteBackend(ClipboardBackend):
            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteBackend()  # type: ignore[abstract]


# =====================================================================
# 2. Privacy & Redaction Helpers
# =====================================================================

class TestClipboardPrivacyAndConstants:
    """Validate redaction format, constants, and zero-content leakage."""

    def test_constants_values(self) -> None:
        """Verify expected buffer size caps and timeouts."""
        assert MAX_CLIPBOARD_TEXT_BYTES == 1_048_576
        assert DEFAULT_SUBPROCESS_TIMEOUT == 3.0
        assert SUBPROCESS_TERMINATE_TIMEOUT == 2.0

    def test_redact_clipboard_masks_sensitive_data(self) -> None:
        """Verify redact_clipboard does not leak cleartext and formats metadata."""
        secret = "Password123!_super_secret"
        redacted = redact_clipboard(secret)

        assert secret not in redacted
        assert f"{len(secret)} chars" in redacted
        assert "sha256:" in redacted
        assert redacted.startswith("[REDACTED, ")
        assert redacted.endswith("]")

    def test_redact_clipboard_empty_string(self) -> None:
        """Verify redact_clipboard handles empty string."""
        redacted = redact_clipboard("")
        assert "0 chars" in redacted
        assert "sha256:e3b0c442" in redacted

    def test_redact_clipboard_unicode_and_emojis(self) -> None:
        """Verify redact_clipboard handles complex Unicode characters."""
        text = "PenguLink 🐧🔗 🎉"
        redacted = redact_clipboard(text)
        assert text not in redacted
        assert f"{len(text)} chars" in redacted


# =====================================================================
# 3. InMemoryClipboardBackend Tests
# =====================================================================

class TestInMemoryClipboardBackend:
    """Comprehensive tests for the InMemoryClipboardBackend test double."""

    async def test_initialization_defaults(self) -> None:
        """Verify initial state with default and custom initial text."""
        backend_default = InMemoryClipboardBackend()
        assert backend_default.name == "in_memory"
        assert backend_default.is_available is True
        assert await backend_default.get_text() == ""
        assert backend_default.mutation_history == []

        backend_custom = InMemoryClipboardBackend(initial_text="seed")
        assert await backend_custom.get_text() == "seed"
        assert backend_custom.mutation_history == ["seed"]

    async def test_get_and_set_text(self) -> None:
        """Verify setting and reading clipboard text in memory."""
        backend = InMemoryClipboardBackend()
        success = await backend.set_text("first value")
        assert success is True
        assert await backend.get_text() == "first value"
        assert backend.mutation_history == ["first value"]

        success = await backend.set_text("second value")
        assert success is True
        assert await backend.get_text() == "second value"
        assert backend.mutation_history == ["first value", "second value"]

    async def test_set_text_does_not_fire_watcher_callback(self) -> None:
        """Verify set_text (remote update) does NOT trigger watcher callback."""
        backend = InMemoryClipboardBackend()
        events: List[str] = []

        async def on_change(content: str) -> None:
            events.append(content)

        await backend.start_watching(on_change)
        await backend.set_text("remote packet arriving")
        assert events == []
        assert await backend.get_text() == "remote packet arriving"
        await backend.stop_watching()

    async def test_simulate_external_copy_fires_watcher_callback(self) -> None:
        """Verify simulate_external_copy (local Ctrl+C) triggers watcher callback."""
        backend = InMemoryClipboardBackend()
        events: List[str] = []

        async def on_change(content: str) -> None:
            events.append(content)

        await backend.start_watching(on_change)
        await backend.simulate_external_copy("user copied this locally")
        assert events == ["user copied this locally"]
        assert await backend.get_text() == "user copied this locally"
        assert backend.mutation_history == ["user copied this locally"]

        # Stop watching and ensure subsequent copies do not fire callback
        await backend.stop_watching()
        await backend.simulate_external_copy("copied after stop")
        assert events == ["user copied this locally"]
        assert await backend.get_text() == "copied after stop"

    async def test_size_limit_rejection_boundary(self) -> None:
        """Verify rejection at exact 1 MiB boundary (1048576 vs 1048577 bytes)."""
        backend = InMemoryClipboardBackend()

        # Exactly 1 MiB should succeed
        exact_text = "A" * MAX_CLIPBOARD_TEXT_BYTES
        assert await backend.set_text(exact_text) is True
        assert await backend.get_text() == exact_text

        # 1 MiB + 1 byte should be rejected
        oversized_text = "A" * (MAX_CLIPBOARD_TEXT_BYTES + 1)
        assert await backend.set_text(oversized_text) is False
        # Content remains unchanged
        assert await backend.get_text() == exact_text

        # simulate_external_copy also respects limit
        events: List[str] = []
        await backend.start_watching(lambda c: events.append(c))  # type: ignore[arg-type]
        await backend.simulate_external_copy(oversized_text)
        assert events == []
        assert await backend.get_text() == exact_text
        await backend.stop_watching()

    async def test_unicode_emojis_and_multiline(self) -> None:
        """Verify multiline formatting and complex Unicode emojis are preserved."""
        backend = InMemoryClipboardBackend()
        multiline_emoji_text = "Line 1 🐧\nLine 2 🔗\r\nLine 3 🎉 — مرحبا / 日本語"

        assert await backend.set_text(multiline_emoji_text) is True
        assert await backend.get_text() == multiline_emoji_text

    async def test_clear_history(self) -> None:
        """Verify clear_history clears recorded mutations."""
        backend = InMemoryClipboardBackend(initial_text="init")
        await backend.set_text("update 1")
        await backend.set_text("update 2")
        assert len(backend.mutation_history) == 3

        backend.clear_history()
        assert backend.mutation_history == []

    async def test_callback_exception_isolation(self) -> None:
        """Verify errors raised inside subscriber callback do not crash backend."""
        backend = InMemoryClipboardBackend()

        async def failing_callback(content: str) -> None:
            raise RuntimeError("Subscriber explosion!")

        await backend.start_watching(failing_callback)
        # Should not raise exception
        await backend.simulate_external_copy("safe text")
        assert await backend.get_text() == "safe text"
        await backend.stop_watching()


# =====================================================================
# 4. WaylandClipboardBackend Tests
# =====================================================================

class TestWaylandClipboardBackend:
    """Comprehensive tests for Wayland clipboard backend."""

    def test_availability_matrix(self) -> None:
        """Verify is_available conditions across environment and PATH."""
        backend = WaylandClipboardBackend()

        # Both tools and WAYLAND_DISPLAY present -> True
        with patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}, clear=True), \
             patch("shutil.which", return_value="/usr/bin/tool"):
            assert backend.is_available is True

        # WAYLAND_DISPLAY unset -> False
        with patch.dict(os.environ, {}, clear=True), \
             patch("shutil.which", return_value="/usr/bin/tool"):
            assert backend.is_available is False

        # wl-copy missing -> False
        with patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}, clear=True), \
             patch("shutil.which", side_effect=lambda t: "/usr/bin/wl-paste" if t == "wl-paste" else None):
            assert backend.is_available is False

        # wl-paste missing -> False
        with patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}, clear=True), \
             patch("shutil.which", side_effect=lambda t: "/usr/bin/wl-copy" if t == "wl-copy" else None):
            assert backend.is_available is False

    async def test_get_text_success(self) -> None:
        """Verify get_text executes wl-paste and parses decoded stdout."""
        backend = WaylandClipboardBackend()

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"wayland text content", b"")
        mock_proc.returncode = 0

        with patch("shutil.which", return_value="/usr/bin/wl-paste"), \
             patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            result = await backend.get_text()
            assert result == "wayland text content"
            mock_exec.assert_awaited_once_with(
                "wl-paste",
                "--no-newline",
                "--type",
                "text/plain",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

    async def test_get_text_empty_clipboard_or_nonzero_exit(self) -> None:
        """Verify get_text returns empty string on exit code 1 or missing tool."""
        backend = WaylandClipboardBackend()

        # Tool missing in PATH
        with patch("shutil.which", return_value=None):
            assert await backend.get_text() == ""

        # Exit code 1 (no selection)
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"No selection")
        mock_proc.returncode = 1

        with patch("shutil.which", return_value="/usr/bin/wl-paste"), \
             patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            assert await backend.get_text() == ""

    async def test_get_text_timeout_handling(self) -> None:
        """Verify get_text kills process on timeout and returns empty string."""
        backend = WaylandClipboardBackend()

        mock_proc = AsyncMock()
        mock_proc.communicate.side_effect = asyncio.TimeoutError()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("shutil.which", return_value="/usr/bin/wl-paste"), \
             patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await backend.get_text()
            assert result == ""
            mock_proc.kill.assert_called_once()

    async def test_get_text_clamps_oversized_output(self) -> None:
        """Verify get_text truncates output exceeding 1 MiB."""
        backend = WaylandClipboardBackend()
        oversized = b"Z" * (MAX_CLIPBOARD_TEXT_BYTES + 500)

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (oversized, b"")
        mock_proc.returncode = 0

        with patch("shutil.which", return_value="/usr/bin/wl-paste"), \
             patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await backend.get_text()
            assert len(result.encode("utf-8")) == MAX_CLIPBOARD_TEXT_BYTES

    async def test_set_text_success(self) -> None:
        """Verify set_text pipes UTF-8 encoded bytes to wl-copy stdin."""
        backend = WaylandClipboardBackend()

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("shutil.which", return_value="/usr/bin/wl-copy"), \
             patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            success = await backend.set_text("test write content")
            assert success is True
            mock_exec.assert_awaited_once_with(
                "wl-copy",
                "--type",
                "text/plain",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            mock_proc.communicate.assert_awaited_once_with(input=b"test write content")

    async def test_set_text_oversized_rejection(self) -> None:
        """Verify set_text rejects oversized text without launching subprocess."""
        backend = WaylandClipboardBackend()
        oversized = "A" * (MAX_CLIPBOARD_TEXT_BYTES + 1)

        with patch("shutil.which", return_value="/usr/bin/wl-copy"), \
             patch("asyncio.create_subprocess_exec") as mock_exec:
            assert await backend.set_text(oversized) is False
            mock_exec.assert_not_called()

    async def test_set_text_timeout_handling(self) -> None:
        """Verify set_text terminates process on timeout and returns False."""
        backend = WaylandClipboardBackend()

        mock_proc = AsyncMock()
        mock_proc.communicate.side_effect = asyncio.TimeoutError()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("shutil.which", return_value="/usr/bin/wl-copy"), \
             patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            assert await backend.set_text("timeout payload") is False
            mock_proc.kill.assert_called_once()

    async def test_start_watching_unavailable_raises_error(self) -> None:
        """Verify start_watching raises ClipboardUnavailableError if unavailable."""
        backend = WaylandClipboardBackend()
        backend.is_available = False
        with pytest.raises(ClipboardUnavailableError):
            await backend.start_watching(AsyncMock())

    async def test_watcher_length_prefixed_framing(self) -> None:
        """Verify length-prefixed streaming protocol receives framed multiline and emojis."""
        backend = WaylandClipboardBackend()
        backend.is_available = True
        received: List[str] = []

        async def on_change(text: str) -> None:
            received.append(text)

        # Frame 1: Multiline text
        data1 = "Line 1\nLine 2\r\nLine 3".encode("utf-8")
        frame1 = struct.pack(">I", len(data1)) + data1

        # Frame 2: Unicode emojis
        data2 = "PenguLink 🐧 🔗".encode("utf-8")
        frame2 = struct.pack(">I", len(data2)) + data2

        # Frame 3: Empty string (0-length frame)
        frame3 = struct.pack(">I", 0)

        stream = asyncio.StreamReader()
        stream.feed_data(frame1 + frame2 + frame3)
        stream.feed_eof()

        mock_proc = MagicMock()
        mock_proc.stdout = stream
        mock_proc.returncode = None
        mock_proc.pid = 12345
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await backend.start_watching(on_change)
            await asyncio.sleep(0.05)  # Yield for reader task to process stream
            await backend.stop_watching()

        assert received == ["Line 1\nLine 2\r\nLine 3", "PenguLink 🐧 🔗", ""]

    async def test_watcher_oversized_frame_drain(self) -> None:
        """Verify oversized frames are drained without crashing stream synchronization."""
        backend = WaylandClipboardBackend()
        backend.is_available = True
        received: List[str] = []

        # Frame 1: Oversized frame (1 MiB + 10 bytes)
        oversized_len = MAX_CLIPBOARD_TEXT_BYTES + 10
        oversized_header = struct.pack(">I", oversized_len)
        oversized_body = b"X" * oversized_len

        # Frame 2: Valid follow-up frame
        valid_data = b"valid following frame"
        valid_frame = struct.pack(">I", len(valid_data)) + valid_data

        stream = asyncio.StreamReader()
        stream.feed_data(oversized_header + oversized_body + valid_frame)
        stream.feed_eof()

        mock_proc = MagicMock()
        mock_proc.stdout = stream
        mock_proc.returncode = None
        mock_proc.pid = 54321
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await backend.start_watching(lambda c: received.append(c))  # type: ignore[arg-type]
            await asyncio.sleep(0.05)
            await backend.stop_watching()

        # Oversized frame is skipped, but subsequent valid frame is received cleanly!
        assert received == ["valid following frame"]

    async def test_stop_watching_escalation(self) -> None:
        """Verify stop_watching escalates from SIGTERM to SIGKILL on timeout."""
        backend = WaylandClipboardBackend()

        mock_proc = MagicMock()
        mock_proc.pid = 9999
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        # First wait times out, second wait succeeds after SIGKILL
        mock_proc.wait = AsyncMock(side_effect=[asyncio.TimeoutError(), 0])

        backend._watch_process = mock_proc
        backend._watch_task = None
        backend._stopped = False

        with patch("os.killpg", side_effect=OSError()):  # Force fallback to terminate/kill
            await backend.stop_watching()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert backend._watch_process is None


# =====================================================================
# 5. X11ClipboardBackend Tests
# =====================================================================

class TestX11ClipboardBackend:
    """Comprehensive tests for X11 clipboard backend."""

    def test_availability_matrix(self) -> None:
        """Verify is_available conditions across DISPLAY, tools, and ctypes."""
        backend = X11ClipboardBackend()

        # All present -> True
        with patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True), \
             patch("shutil.which", return_value="/usr/bin/xclip"), \
             patch.object(backend, "_has_x11_ctypes", return_value=True):
            assert backend.is_available is True

        # Headless (DISPLAY unset) -> False
        with patch.dict(os.environ, {}, clear=True), \
             patch("shutil.which", return_value="/usr/bin/xclip"), \
             patch.object(backend, "_has_x11_ctypes", return_value=True):
            assert backend.is_available is False

        # No CLI tool -> False
        with patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True), \
             patch("shutil.which", return_value=None), \
             patch.object(backend, "_has_x11_ctypes", return_value=True):
            assert backend.is_available is False

    async def test_get_text_with_xclip(self) -> None:
        """Verify get_text executes xclip -selection clipboard -o."""
        backend = X11ClipboardBackend()
        backend._tool = "/usr/bin/xclip"

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"xclip content", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            text = await backend.get_text()
            assert text == "xclip content"
            mock_exec.assert_awaited_once_with(
                "/usr/bin/xclip",
                "-selection",
                "clipboard",
                "-o",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

    async def test_get_text_with_xsel(self) -> None:
        """Verify get_text executes xsel --clipboard --output when tool is xsel."""
        backend = X11ClipboardBackend()
        backend._tool = "/usr/bin/xsel"

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"xsel content", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            text = await backend.get_text()
            assert text == "xsel content"
            mock_exec.assert_awaited_once_with(
                "/usr/bin/xsel",
                "--clipboard",
                "--output",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

    async def test_set_text_with_xclip(self) -> None:
        """Verify set_text pipes data into xclip -selection clipboard -i."""
        backend = X11ClipboardBackend()
        backend._tool = "/usr/bin/xclip"

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            success = await backend.set_text("xclip input")
            assert success is True
            mock_exec.assert_awaited_once_with(
                "/usr/bin/xclip",
                "-selection",
                "clipboard",
                "-i",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            mock_proc.communicate.assert_awaited_once_with(input=b"xclip input")

    async def test_set_text_with_xsel(self) -> None:
        """Verify set_text pipes data into xsel --clipboard --input."""
        backend = X11ClipboardBackend()
        backend._tool = "/usr/bin/xsel"

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            success = await backend.set_text("xsel input")
            assert success is True
            mock_exec.assert_awaited_once_with(
                "/usr/bin/xsel",
                "--clipboard",
                "--input",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

    async def test_headless_graceful_fallback(self) -> None:
        """Verify headless (no tool/display) returns empty/false without crashing."""
        backend = X11ClipboardBackend()
        backend._tool = None
        backend.is_available = False

        assert await backend.get_text() == ""
        assert await backend.set_text("data") is False

        with pytest.raises(ClipboardUnavailableError):
            await backend.start_watching(AsyncMock())

    async def test_native_xfixes_setup_and_readable(self) -> None:
        """Verify native XFixes ctypes setup, loop.add_reader, and event draining."""
        backend = X11ClipboardBackend()
        backend._tool = "/usr/bin/xclip"
        backend.is_available = True

        mock_x11 = MagicMock()
        mock_xfixes = MagicMock()

        mock_x11.XOpenDisplay.return_value = 0x12345678
        mock_x11.XConnectionNumber.return_value = 7
        mock_x11.XDefaultRootWindow.return_value = 100
        mock_x11.XInternAtom.return_value = 42

        # Simulate 2 pending events drained, then 0
        mock_x11.XPending.side_effect = [2, 1, 0]

        def fake_load_library(name: str) -> Any:
            if "libX11" in name:
                return mock_x11
            if "libXfixes" in name:
                return mock_xfixes
            raise OSError()

        received: List[str] = []

        async def on_change(text: str) -> None:
            received.append(text)

        with patch("ctypes.cdll.LoadLibrary", side_effect=fake_load_library), \
             patch.object(backend, "_has_x11_ctypes", return_value=True), \
             patch.object(backend, "get_text", AsyncMock(return_value="x11 event text")):

            await backend.start_watching(on_change)
            assert backend._x11_fd == 7

            # Trigger readable callback
            backend._on_x11_readable()
            await asyncio.sleep(0.05)  # Wait for coalesced notify

            assert received == ["x11 event text"]
            assert mock_x11.XNextEvent.call_count == 2

            # Clean teardown
            await backend.stop_watching()
            assert backend._x11_fd is None
            assert backend._x11_dpy is None
            mock_x11.XCloseDisplay.assert_called_once_with(0x12345678)


# =====================================================================
# 6. Factory detect_clipboard_backend Tests
# =====================================================================

class TestDetectClipboardBackend:
    """Comprehensive tests for detect_clipboard_backend factory."""

    def test_explicit_argument_precedence(self) -> None:
        """Verify explicit backend_name takes precedence over environment variable."""
        with patch.dict(os.environ, {"PENGULINK_CLIPBOARD_BACKEND": "wayland"}):
            backend = detect_clipboard_backend("in_memory")
            assert isinstance(backend, InMemoryClipboardBackend)

        backend = detect_clipboard_backend("wayland")
        assert isinstance(backend, WaylandClipboardBackend)

        backend = detect_clipboard_backend("x11")
        assert isinstance(backend, X11ClipboardBackend)

    def test_aliases_for_in_memory(self) -> None:
        """Verify in_memory, memory, headless, mock all return InMemoryClipboardBackend."""
        for alias in ("in_memory", "memory", "headless", "mock", "IN_MEMORY"):
            b = detect_clipboard_backend(alias)
            assert isinstance(b, InMemoryClipboardBackend)

    def test_unknown_backend_raises_error(self) -> None:
        """Verify requesting unknown backend raises ClipboardBackendError."""
        with pytest.raises(ClipboardBackendError) as exc_info:
            detect_clipboard_backend("unknown_desktop")
        assert "Unknown clipboard backend requested" in str(exc_info.value)

        with patch.dict(os.environ, {"PENGULINK_CLIPBOARD_BACKEND": "bogus"}):
            with pytest.raises(ClipboardBackendError):
                detect_clipboard_backend()

    def test_env_variable_resolution(self) -> None:
        """Verify PENGULINK_CLIPBOARD_BACKEND resolution when no argument passed."""
        with patch.dict(os.environ, {"PENGULINK_CLIPBOARD_BACKEND": "wayland"}):
            assert isinstance(detect_clipboard_backend(), WaylandClipboardBackend)

        with patch.dict(os.environ, {"PENGULINK_CLIPBOARD_BACKEND": "x11"}):
            assert isinstance(detect_clipboard_backend(), X11ClipboardBackend)

        with patch.dict(os.environ, {"PENGULINK_CLIPBOARD_BACKEND": "mock"}):
            assert isinstance(detect_clipboard_backend(), InMemoryClipboardBackend)

    def test_auto_detection_cascade(self) -> None:
        """Verify auto-detection cascade: Wayland -> X11 -> InMemory fallback."""
        # 1. Wayland available -> Wayland
        with patch.dict(os.environ, {}, clear=True), \
             patch("pengulink.plugins.clipboard.backend.WaylandClipboardBackend.is_available", True):
            assert isinstance(detect_clipboard_backend(), WaylandClipboardBackend)

        # 2. Wayland unavailable, X11 available -> X11
        with patch.dict(os.environ, {}, clear=True), \
             patch("pengulink.plugins.clipboard.backend.WaylandClipboardBackend.is_available", False), \
             patch("pengulink.plugins.clipboard.backend.X11ClipboardBackend.is_available", True):
            assert isinstance(detect_clipboard_backend(), X11ClipboardBackend)

        # 3. Neither available -> InMemory
        with patch.dict(os.environ, {}, clear=True), \
             patch("pengulink.plugins.clipboard.backend.WaylandClipboardBackend.is_available", False), \
             patch("pengulink.plugins.clipboard.backend.X11ClipboardBackend.is_available", False):
            assert isinstance(detect_clipboard_backend(), InMemoryClipboardBackend)

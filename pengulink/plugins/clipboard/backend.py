"""pengulink.plugins.clipboard.backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pluggable desktop clipboard backend abstraction for PenguLink.

Supports:
- Wayland desktops via wl-clipboard (wl-copy and wl-paste --watch)
- X11 desktops via ctypes libX11 / libXfixes event-driven monitoring and xclip/xsel
- In-memory mock/headless clipboard for CI, Docker, and unit tests
- Automatic runtime detection with PENGULINK_CLIPBOARD_BACKEND environment override
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import logging
import os
import shutil
import signal
import struct
import sys
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, List, Optional, Set

from pengulink.core.exceptions import PluginError

logger = logging.getLogger("pengulink.plugins.clipboard.backend")

# Maximum allowed clipboard text size in bytes (1 MiB = 1,048,576 bytes)
MAX_CLIPBOARD_TEXT_BYTES: int = 1_048_576

# Subprocess timeouts in seconds
DEFAULT_SUBPROCESS_TIMEOUT: float = 3.0
SUBPROCESS_TERMINATE_TIMEOUT: float = 2.0

# Asynchronous callback invoked when clipboard content changes locally
ClipboardChangeCallback = Callable[[str], Awaitable[None]]


class ClipboardError(PluginError):
    """Base exception for clipboard plugin and backend subsystem."""


class ClipboardBackendError(ClipboardError):
    """Exception raised when a clipboard backend encounters an operational error."""


class ClipboardUnavailableError(ClipboardBackendError):
    """Raised when a requested clipboard backend is not available."""


def redact_clipboard(content: str) -> str:
    """Format clipboard content safely for logging without leaking sensitive data."""
    char_len = len(content)
    digest_prefix = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"[REDACTED, {char_len} chars, sha256:{digest_prefix}]"


# Ctypes X11 Error Handler Callback Type
_XERROR_HANDLER_T = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)


def _default_x11_error_handler(display_ptr: Any, error_event_ptr: Any) -> int:
    """Non-fatal X11 protocol error handler to prevent Xlib from calling exit(1)."""
    logger.debug("Suppressed asynchronous X11 protocol error")
    return 0


# Retain module-level reference to prevent garbage collection of C callback
_c_x11_error_handler = _XERROR_HANDLER_T(_default_x11_error_handler)


class _XEvent(ctypes.Structure):
    """Generic XEvent union buffer ensuring 192-byte padding against stack corruption."""

    _fields_ = [
        ("type", ctypes.c_int),
        ("pad", ctypes.c_byte * 192),
    ]


class ClipboardBackend(ABC):
    """Abstract interface defining the desktop clipboard backend contract."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifying name of the backend (e.g. 'wayland', 'x11', 'in_memory')."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Return True if required environment and tools are available."""
        ...

    @abstractmethod
    async def get_text(self) -> str:
        """Read current text from the system clipboard.

        Returns:
            Current clipboard text, or empty string if clipboard is empty or unreadable.
        """
        ...

    @abstractmethod
    async def set_text(self, text: str) -> bool:
        """Write text to the system clipboard.

        Args:
            text: UTF-8 string to write.

        Returns:
            True on success, False if write failed or exceeded size limits.
        """
        ...

    @abstractmethod
    async def start_watching(self, on_change: ClipboardChangeCallback) -> None:
        """Begin event-driven monitoring of clipboard changes without busy polling.

        Args:
            on_change: Async callback receiving the new text content when clipboard changes.
        """
        ...

    @abstractmethod
    async def stop_watching(self) -> None:
        """Stop monitoring and cleanly terminate child processes or event loops."""
        ...


class WaylandClipboardBackend(ClipboardBackend):
    """Wayland clipboard implementation using wl-clipboard (wl-copy, wl-paste)."""

    def __init__(self) -> None:
        self._watch_process: Optional[asyncio.subprocess.Process] = None
        self._watch_task: Optional[asyncio.Task[None]] = None
        self._callback: Optional[ClipboardChangeCallback] = None
        self._stopped: bool = False
        self._available_override: Optional[bool] = None

    @property
    def name(self) -> str:
        return "wayland"

    @property
    def is_available(self) -> bool:
        if self._available_override is not None:
            return self._available_override
        has_display = bool(os.environ.get("WAYLAND_DISPLAY"))
        has_tools = bool(shutil.which("wl-copy") and shutil.which("wl-paste"))
        return has_display and has_tools

    @is_available.setter
    def is_available(self, val: Optional[bool]) -> None:
        self._available_override = val

    @is_available.deleter
    def is_available(self) -> None:
        self._available_override = None

    async def get_text(self) -> str:
        if not shutil.which("wl-paste"):
            return ""
        try:
            proc = await asyncio.create_subprocess_exec(
                "wl-paste",
                "--no-newline",
                "--type",
                "text/plain",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=DEFAULT_SUBPROCESS_TIMEOUT
            )
            if proc.returncode != 0:
                return ""
            if len(stdout) > MAX_CLIPBOARD_TEXT_BYTES:
                stdout = stdout[:MAX_CLIPBOARD_TEXT_BYTES]
            return stdout.decode("utf-8", errors="replace")
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            logger.warning("Wayland wl-paste timed out")
            return ""
        except (FileNotFoundError, ProcessLookupError, OSError) as exc:
            logger.debug("Failed to read Wayland clipboard: %s", exc)
            return ""

    async def set_text(self, text: str) -> bool:
        if not shutil.which("wl-copy"):
            return False
        try:
            data = text.encode("utf-8")
        except UnicodeEncodeError:
            logger.warning("Wayland clipboard write rejected: text contains unencodable surrogates")
            return False
        if len(data) > MAX_CLIPBOARD_TEXT_BYTES:
            logger.warning(
                "Wayland clipboard write rejected: exceeds %d bytes",
                MAX_CLIPBOARD_TEXT_BYTES,
            )
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                "wl-copy",
                "--type",
                "text/plain",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(
                proc.communicate(input=data), timeout=DEFAULT_SUBPROCESS_TIMEOUT
            )
            logger.debug("Wrote text to Wayland clipboard: %s", redact_clipboard(text))
            return proc.returncode == 0
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            logger.warning("Wayland wl-copy timed out")
            return False
        except (FileNotFoundError, ProcessLookupError, OSError) as exc:
            logger.warning("Failed to write to Wayland clipboard: %s", exc)
            return False

    async def start_watching(self, on_change: ClipboardChangeCallback) -> None:
        if not self.is_available:
            raise ClipboardUnavailableError("Wayland clipboard tools not available")
        self._callback = on_change
        self._stopped = False

        # Length-prefixed binary framing helper over stdout
        cmd = [
            "wl-paste",
            "--type",
            "text/plain",
            "--watch",
            sys.executable,
            "-c",
            "import sys, struct; d = sys.stdin.buffer.read(); "
            "sys.stdout.buffer.write(struct.pack('>I', len(d)) + d); "
            "sys.stdout.buffer.flush()",
        ]
        self._watch_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        self._watch_task = asyncio.create_task(self._reader_loop())
        logger.debug("Started Wayland clipboard watcher (PID=%s)", getattr(self._watch_process, "pid", None))

    async def _reader_loop(self) -> None:
        proc = self._watch_process
        if not proc or not proc.stdout:
            return
        try:
            while not self._stopped and proc.returncode is None:
                header = await proc.stdout.readexactly(4)
                length = struct.unpack(">I", header)[0]
                if length > MAX_CLIPBOARD_TEXT_BYTES:
                    # Drain oversized payload safely in bounded chunks
                    remaining = length
                    while remaining > 0:
                        chunk_size = min(remaining, 65536)
                        await proc.stdout.readexactly(chunk_size)
                        remaining -= chunk_size
                    logger.warning("Dropped oversized clipboard change (%d bytes)", length)
                    continue

                payload = await proc.stdout.readexactly(length)
                content = payload.decode("utf-8", errors="replace")
                if self._callback and not self._stopped:
                    try:
                        await self._callback(content)
                    except Exception as exc:
                        logger.error("Error in clipboard change callback: %s", exc, exc_info=True)
        except (asyncio.IncompleteReadError, asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as exc:
            if not self._stopped:
                logger.error("Unexpected error in Wayland reader loop: %s", exc)

    async def stop_watching(self) -> None:
        self._stopped = True
        if self._watch_task and not self._watch_task.done():
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
        self._watch_task = None

        proc = self._watch_process
        if proc is not None and proc.returncode is None:
            try:
                # Stage 1: Graceful SIGTERM escalation to process group
                try:
                    if hasattr(proc, "pid") and proc.pid:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    else:
                        proc.terminate()
                except (ProcessLookupError, PermissionError, OSError, AttributeError):
                    proc.terminate()

                await asyncio.wait_for(
                    proc.wait(), timeout=SUBPROCESS_TERMINATE_TIMEOUT
                )
            except (asyncio.TimeoutError, ProcessLookupError):
                # Stage 2: Escalation to SIGKILL on timeout
                logger.warning(
                    "Wayland watcher failed to exit within %ss, escalating to SIGKILL",
                    SUBPROCESS_TERMINATE_TIMEOUT,
                )
                try:
                    try:
                        if hasattr(proc, "pid") and proc.pid:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        else:
                            proc.kill()
                    except (ProcessLookupError, PermissionError, OSError, AttributeError):
                        proc.kill()
                    await asyncio.wait_for(
                        proc.wait(), timeout=SUBPROCESS_TERMINATE_TIMEOUT
                    )
                except (asyncio.TimeoutError, ProcessLookupError) as exc:
                    logger.error("Failed to reap Wayland watcher child process: %s", exc)
        self._watch_process = None
        self._callback = None
        logger.debug("Stopped Wayland clipboard watcher")


class X11ClipboardBackend(ClipboardBackend):
    """X11 clipboard implementation supporting native ctypes XFixes and xclip/xsel."""

    def __init__(self) -> None:
        self._tool_override: Optional[str] = None
        self._available_override: Optional[bool] = None
        self._x11_dpy: Optional[Any] = None
        self._x11_fd: Optional[int] = None
        self._x11_lib: Any = None
        self._callback: Optional[ClipboardChangeCallback] = None
        self._stopped: bool = False
        self._notify_pending: bool = False
        self._tasks: Set[asyncio.Task[Any]] = set()

    @property
    def _tool(self) -> Optional[str]:
        if self._tool_override is not None:
            return self._tool_override
        return shutil.which("xclip") or shutil.which("xsel")

    @_tool.setter
    def _tool(self, val: Optional[str]) -> None:
        self._tool_override = val

    @_tool.deleter
    def _tool(self) -> None:
        self._tool_override = None

    @property
    def name(self) -> str:
        return "x11"

    @property
    def is_available(self) -> bool:
        if self._available_override is not None:
            return self._available_override
        has_display = bool(os.environ.get("DISPLAY"))
        has_io = bool(self._tool)
        has_watcher = self._has_x11_ctypes() or bool(shutil.which("clipnotify"))
        return has_display and has_io and has_watcher

    @is_available.setter
    def is_available(self, val: Optional[bool]) -> None:
        self._available_override = val

    @is_available.deleter
    def is_available(self) -> None:
        self._available_override = None

    def _has_x11_ctypes(self) -> bool:
        try:
            ctypes.cdll.LoadLibrary("libX11.so.6")
            ctypes.cdll.LoadLibrary("libXfixes.so.3")
            return True
        except OSError:
            return False

    async def get_text(self) -> str:
        tool = self._tool
        if not tool:
            return ""
        if "xclip" in tool:
            cmd = [tool, "-selection", "clipboard", "-o"]
        elif "xsel" in tool:
            cmd = [tool, "--clipboard", "--output"]
        else:
            return ""

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=DEFAULT_SUBPROCESS_TIMEOUT
            )
            if proc.returncode != 0:
                return ""
            if len(stdout) > MAX_CLIPBOARD_TEXT_BYTES:
                stdout = stdout[:MAX_CLIPBOARD_TEXT_BYTES]
            return stdout.decode("utf-8", errors="replace")
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            logger.warning("X11 get_text subprocess timed out")
            return ""
        except (FileNotFoundError, ProcessLookupError, OSError) as exc:
            logger.debug("Failed to read X11 clipboard: %s", exc)
            return ""

    async def set_text(self, text: str) -> bool:
        tool = self._tool
        if not tool:
            return False
        try:
            data = text.encode("utf-8")
        except UnicodeEncodeError:
            logger.warning("X11 clipboard write rejected: text contains unencodable surrogates")
            return False
        if len(data) > MAX_CLIPBOARD_TEXT_BYTES:
            logger.warning(
                "X11 clipboard write rejected: exceeds %d bytes",
                MAX_CLIPBOARD_TEXT_BYTES,
            )
            return False

        if "xclip" in tool:
            cmd = [tool, "-selection", "clipboard", "-i"]
        elif "xsel" in tool:
            cmd = [tool, "--clipboard", "--input"]
        else:
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(
                proc.communicate(input=data), timeout=DEFAULT_SUBPROCESS_TIMEOUT
            )
            logger.debug("Wrote text to X11 clipboard: %s", redact_clipboard(text))
            return proc.returncode == 0
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            logger.warning("X11 set_text subprocess timed out")
            return False
        except (FileNotFoundError, ProcessLookupError, OSError) as exc:
            logger.warning("Failed to write to X11 clipboard: %s", exc)
            return False

    async def start_watching(self, on_change: ClipboardChangeCallback) -> None:
        if not self.is_available:
            raise ClipboardUnavailableError("X11 clipboard display/tools not available")
        self._callback = on_change
        self._stopped = False

        if self._has_x11_ctypes():
            await self._setup_native_xfixes_watcher()
        else:
            raise ClipboardUnavailableError("X11 native XFixes libraries not available")

    async def _setup_native_xfixes_watcher(self) -> None:
        loop = asyncio.get_running_loop()
        x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
        xfixes = ctypes.cdll.LoadLibrary("libXfixes.so.3")

        # Set 64-bit safe ctypes function prototypes
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XConnectionNumber.argtypes = [ctypes.c_void_p]
        x11.XConnectionNumber.restype = ctypes.c_int
        x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        x11.XDefaultRootWindow.restype = ctypes.c_ulong
        x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        x11.XInternAtom.restype = ctypes.c_ulong
        x11.XSetErrorHandler.argtypes = [ctypes.c_void_p]
        x11.XSetErrorHandler.restype = ctypes.c_void_p
        xfixes.XFixesSelectSelectionInput.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong
        ]
        xfixes.XFixesSelectSelectionInput.restype = None
        x11.XFlush.argtypes = [ctypes.c_void_p]
        x11.XFlush.restype = ctypes.c_int
        x11.XPending.argtypes = [ctypes.c_void_p]
        x11.XPending.restype = ctypes.c_int
        x11.XNextEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        x11.XNextEvent.restype = ctypes.c_int
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay.restype = ctypes.c_int

        # Protect against Xlib fatal exit on protocol errors
        x11.XSetErrorHandler(_c_x11_error_handler)

        self._x11_lib = x11
        display_name = os.environ.get("DISPLAY")
        dpy_arg = display_name.encode("utf-8") if display_name else None
        self._x11_dpy = x11.XOpenDisplay(dpy_arg)
        if not self._x11_dpy:
            raise ClipboardBackendError("Failed to open X11 Display")

        self._x11_fd = x11.XConnectionNumber(self._x11_dpy)
        root = x11.XDefaultRootWindow(self._x11_dpy)
        clip_atom = x11.XInternAtom(self._x11_dpy, b"CLIPBOARD", 0)

        # XFixesSetSelectionOwnerNotifyMask = 1
        xfixes.XFixesSelectSelectionInput(self._x11_dpy, root, clip_atom, 1)
        x11.XFlush(self._x11_dpy)

        loop.add_reader(self._x11_fd, self._on_x11_readable)
        logger.debug("Started X11 XFixes clipboard watcher (FD=%s)", self._x11_fd)

    def _on_x11_readable(self) -> None:
        if self._stopped or not self._x11_dpy:
            return
        try:
            # Drain Xlib client event queue non-blockingly
            while self._x11_lib and self._x11_lib.XPending(self._x11_dpy) > 0:
                event_buf = _XEvent()
                self._x11_lib.XNextEvent(self._x11_dpy, ctypes.byref(event_buf))

            # Coalesce rapid notifications within a single event loop tick
            if not self._notify_pending:
                self._notify_pending = True
                task = asyncio.create_task(self._coalesced_notify())
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        except Exception as exc:
            logger.error("Error reading X11 event: %s", exc)

    async def _coalesced_notify(self) -> None:
        try:
            await asyncio.sleep(0.01)  # Brief settling window for event burst coalescing
            if self._stopped:
                return
            text = await self.get_text()
            if self._callback and not self._stopped:
                try:
                    await self._callback(text)
                except Exception as exc:
                    logger.error("Error in X11 change notification callback: %s", exc, exc_info=True)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Error in X11 coalesced notify: %s", exc, exc_info=True)
        finally:
            self._notify_pending = False

    async def stop_watching(self) -> None:
        self._stopped = True
        for t in list(self._tasks):
            if not t.done():
                t.cancel()
        self._tasks.clear()
        if self._x11_fd is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.remove_reader(self._x11_fd)
            except (RuntimeError, ValueError):
                pass
            self._x11_fd = None

        if self._x11_dpy and self._x11_lib:
            try:
                self._x11_lib.XCloseDisplay(self._x11_dpy)
            except Exception as exc:
                logger.debug("Error closing X11 display: %s", exc)
            self._x11_dpy = None

        self._callback = None
        logger.debug("Stopped X11 clipboard watcher")


class InMemoryClipboardBackend(ClipboardBackend):
    """Thread-safe, purely in-memory clipboard backend for CI, headless runs, and tests."""

    def __init__(self, initial_text: str = "") -> None:
        self._content: str = initial_text
        self._lock = asyncio.Lock()
        self._callback: Optional[ClipboardChangeCallback] = None
        self._watching: bool = False
        self.mutation_history: List[str] = [initial_text] if initial_text else []

    @property
    def name(self) -> str:
        return "in_memory"

    @property
    def is_available(self) -> bool:
        return True

    async def get_text(self) -> str:
        async with self._lock:
            return self._content

    async def set_text(self, text: str) -> bool:
        try:
            encoded = text.encode("utf-8")
        except UnicodeEncodeError:
            logger.warning("In-memory clipboard write rejected: text contains unencodable surrogates")
            return False
        if len(encoded) > MAX_CLIPBOARD_TEXT_BYTES:
            logger.warning(
                "In-memory clipboard write rejected: payload exceeds %d bytes",
                MAX_CLIPBOARD_TEXT_BYTES,
            )
            return False
        async with self._lock:
            self._content = text
            self.mutation_history.append(text)
        logger.debug("Set in-memory clipboard text: %s", redact_clipboard(text))
        return True

    async def simulate_external_copy(self, text: str) -> None:
        """Simulate a local desktop copy event (triggers watcher callback if watching)."""
        try:
            encoded = text.encode("utf-8")
        except UnicodeEncodeError:
            logger.warning("Simulated external copy rejected: text contains unencodable surrogates")
            return
        if len(encoded) > MAX_CLIPBOARD_TEXT_BYTES:
            logger.warning(
                "Simulated external copy rejected: payload exceeds %d bytes",
                MAX_CLIPBOARD_TEXT_BYTES,
            )
            return
        async with self._lock:
            self._content = text
            self.mutation_history.append(text)
        if self._watching and self._callback:
            try:
                await self._callback(text)
            except Exception as exc:
                logger.error("Error in clipboard callback: %s", exc, exc_info=True)

    async def start_watching(self, on_change: ClipboardChangeCallback) -> None:
        self._callback = on_change
        self._watching = True

    async def stop_watching(self) -> None:
        self._callback = None
        self._watching = False

    def clear_history(self) -> None:
        """Clear recorded mutation history."""
        self.mutation_history.clear()


def detect_clipboard_backend(backend_name: Optional[str] = None) -> ClipboardBackend:
    """Detect and instantiate the most appropriate clipboard backend.

    Resolution precedence:
    1. Explicit `backend_name` argument (if provided and non-empty).
    2. `PENGULINK_CLIPBOARD_BACKEND` environment variable.
       - 'wayland' -> WaylandClipboardBackend()
       - 'x11' -> X11ClipboardBackend()
       - 'in_memory' | 'memory' | 'headless' | 'mock' -> InMemoryClipboardBackend()
       - Unknown value -> raises ClipboardBackendError
    3. Auto-detection:
       - If WaylandClipboardBackend().is_available -> WaylandClipboardBackend()
       - If X11ClipboardBackend().is_available -> X11ClipboardBackend()
       - Fallback -> InMemoryClipboardBackend()

    Returns:
        An instantiated instance of ClipboardBackend.

    Raises:
        ClipboardBackendError: If an unknown backend name was explicitly requested.
    """
    req = (backend_name or os.environ.get("PENGULINK_CLIPBOARD_BACKEND", "")).strip().lower()

    if req:
        if req == "wayland":
            return WaylandClipboardBackend()
        elif req == "x11":
            return X11ClipboardBackend()
        elif req in ("in_memory", "memory", "headless", "mock"):
            return InMemoryClipboardBackend()
        else:
            raise ClipboardBackendError(f"Unknown clipboard backend requested: '{req}'")

    # Auto-detection cascade
    wayland = WaylandClipboardBackend()
    if wayland.is_available:
        logger.info("Auto-detected Wayland clipboard backend")
        return wayland

    x11 = X11ClipboardBackend()
    if x11.is_available:
        logger.info("Auto-detected X11 clipboard backend")
        return x11

    logger.info("No active Wayland or X11 tools detected; using InMemoryClipboardBackend")
    return InMemoryClipboardBackend()

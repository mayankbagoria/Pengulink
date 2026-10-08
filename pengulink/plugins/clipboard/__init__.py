"""pengulink.plugins.clipboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect clipboard synchronization plugin and desktop backend subsystem.
"""

from __future__ import annotations

from pengulink.plugins.clipboard.backend import (
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
from pengulink.plugins.clipboard.plugin import (
    CLIPBOARD_CONNECT_PACKET_TYPE,
    CLIPBOARD_PACKET_TYPE,
    DEFAULT_WRITE_IN_FLIGHT_TTL,
    MAX_CLIPBOARD_PACKET_BYTES,
    MAX_FUTURE_CLOCK_DRIFT_MS,
    ClipboardPlugin,
    InFlightWrite,
    LoopSuppressionEngine,
    RollingDigestLRUCache,
    create_clipboard_connect_packet,
    create_clipboard_packet,
)

__all__ = [
    # Backend exports
    "MAX_CLIPBOARD_TEXT_BYTES",
    "ClipboardBackend",
    "ClipboardChangeCallback",
    "WaylandClipboardBackend",
    "X11ClipboardBackend",
    "InMemoryClipboardBackend",
    "detect_clipboard_backend",
    "ClipboardError",
    "ClipboardBackendError",
    "ClipboardUnavailableError",
    "redact_clipboard",
    # Plugin & Engine exports
    "CLIPBOARD_PACKET_TYPE",
    "CLIPBOARD_CONNECT_PACKET_TYPE",
    "MAX_CLIPBOARD_PACKET_BYTES",
    "DEFAULT_WRITE_IN_FLIGHT_TTL",
    "MAX_FUTURE_CLOCK_DRIFT_MS",
    "ClipboardPlugin",
    "LoopSuppressionEngine",
    "RollingDigestLRUCache",
    "InFlightWrite",
    "create_clipboard_packet",
    "create_clipboard_connect_packet",
]

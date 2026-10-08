"""pengulink.plugins.clipboard.plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect Clipboard Synchronization plugin with 4-layer loop suppression.

Features:
- Bidirectional clipboard synchronization (Linux <-> Android/KDE Connect)
- Pluggable desktop backend integration (Wayland, X11, InMemory)
- 4-Layer Loop Suppression Engine:
  1. Rolling SHA-256 digest LRU cache (Capacity 32)
  2. Write-in-flight suppression flag with monotonic TTL window and expected digest matching
  3. Millisecond timestamp comparison (remote > local for inbound; monotonic advancement for local)
  4. Multi-device fanout with originator device ID exclusion
- 120 KB text payload ceiling guard (MAX_CLIPBOARD_PACKET_BYTES = 120_000)
- Full lifecycle management (start/stop) with event subscriptions
- Connection / reconnection state convergence (kdeconnect.clipboard.connect)
- Strict zero-log privacy: never log raw clipboard text (use redact_clipboard)
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from pengulink.core.events import (
    ClipboardUpdatedEvent,
    DeviceConnected,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
)
from pengulink.network.packet import Packet
from pengulink.plugins.base import Plugin, SendPacketCallable, safe_send_packet
from pengulink.plugins.clipboard.backend import (
    ClipboardBackend,
    detect_clipboard_backend,
    redact_clipboard,
)

logger = logging.getLogger("pengulink.plugins.clipboard.plugin")

# KDE Connect wire packet types
CLIPBOARD_PACKET_TYPE: str = "kdeconnect.clipboard"
CLIPBOARD_CONNECT_PACKET_TYPE: str = "kdeconnect.clipboard.connect"

# Wire text payload ceiling in UTF-8 bytes (120 KB)
MAX_CLIPBOARD_PACKET_BYTES: int = 120_000

# Default write-in-flight TTL window in seconds
DEFAULT_WRITE_IN_FLIGHT_TTL: float = 0.5

# Maximum allowed future clock drift for remote timestamps (24 hours in ms)
MAX_FUTURE_CLOCK_DRIFT_MS: int = 86_400_000


def create_clipboard_packet(content: str, timestamp: Optional[int] = None) -> Packet:
    """Create a kdeconnect.clipboard packet with the given content and timestamp."""
    now_ms = int(time.time() * 1000)
    ts = timestamp if timestamp is not None else now_ms
    return Packet(
        id=now_ms,
        type=CLIPBOARD_PACKET_TYPE,
        body={"content": content, "timestamp": ts},
    )


def create_clipboard_connect_packet(content: str, timestamp: Optional[int] = None) -> Packet:
    """Create a kdeconnect.clipboard.connect packet for initial synchronization."""
    now_ms = int(time.time() * 1000)
    ts = timestamp if timestamp is not None else now_ms
    return Packet(
        id=now_ms,
        type=CLIPBOARD_CONNECT_PACKET_TYPE,
        body={"content": content, "timestamp": ts},
    )


@dataclass(frozen=True)
class InFlightWrite:
    """Record of an active in-flight clipboard write to the OS."""

    expiry_monotonic: float
    expected_digest: str
    expected_normalized_digest: str


class RollingDigestLRUCache:
    """Layer 1: Rolling SHA-256 digest LRU cache (capacity 32)."""

    def __init__(self, capacity: int = 32) -> None:
        self.capacity: int = max(1, capacity)
        self._cache: collections.OrderedDict[str, None] = collections.OrderedDict()

    @staticmethod
    def hash_content(text: str) -> str:
        """Compute 64-char SHA-256 hex digest of text UTF-8 bytes."""
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    # Alias for convenience
    digest = hash_content

    def contains(self, digest: str) -> bool:
        """Check presence and update LRU position."""
        if digest in self._cache:
            self._cache.move_to_end(digest, last=True)
            return True
        return False

    def contains_text(self, text: str) -> bool:
        """Hash text and check presence in LRU."""
        return self.contains(self.hash_content(text))

    def add(self, digest: str) -> None:
        """Insert digest, evicting oldest entry if over capacity."""
        if digest in self._cache:
            self._cache.move_to_end(digest, last=True)
        else:
            self._cache[digest] = None
            if len(self._cache) > self.capacity:
                self._cache.popitem(last=False)

    def add_text(self, text: str) -> str:
        """Compute digest, add to cache, and return digest."""
        d = self.hash_content(text)
        self.add(d)
        return d

    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, digest: str) -> bool:
        return digest in self._cache


class LoopSuppressionEngine:
    """Coordinated 4-layer loop suppression engine for KDE Connect clipboard sync.

    Layers:
    1. Rolling SHA-256 Digest LRU Cache (Capacity 32)
    2. Write-in-Flight Monotonic TTL Suppression Window
    3. Strict Millisecond Timestamp Monotonicity
    4. Multi-Device Fanout Originator Exclusion
    """

    def __init__(
        self,
        cache_capacity: int = 32,
        write_in_flight_ttl: float = DEFAULT_WRITE_IN_FLIGHT_TTL,
        max_future_drift_ms: int = MAX_FUTURE_CLOCK_DRIFT_MS,
    ) -> None:
        self.cache_capacity: int = cache_capacity
        self.default_ttl: float = write_in_flight_ttl
        self.max_future_drift_ms: int = max_future_drift_ms

        # Layer 1
        self._digest_cache: RollingDigestLRUCache = RollingDigestLRUCache(capacity=cache_capacity)

        # Layer 2
        self._in_flight: Optional[InFlightWrite] = None

        # Layer 3
        self._local_timestamp_ms: int = 0

    @property
    def local_timestamp(self) -> int:
        """Authoritative local millisecond epoch timestamp."""
        return self._local_timestamp_ms

    @local_timestamp.setter
    def local_timestamp(self, value: int) -> None:
        self._local_timestamp_ms = max(0, int(value))

    @property
    def digest_cache(self) -> RollingDigestLRUCache:
        return self._digest_cache

    @property
    def in_flight(self) -> Optional[InFlightWrite]:
        return self._in_flight

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize line breaks and trailing whitespace for OS variations."""
        return text.replace("\r\n", "\n").rstrip()

    @staticmethod
    def current_wall_ms() -> int:
        """Return current wall-clock epoch in milliseconds."""
        return int(time.time() * 1000)

    # =========================================================================
    # Layer 2: Write-in-Flight Operations
    # =========================================================================

    def arm_write_in_flight(self, content: str, ttl: Optional[float] = None) -> None:
        """Arm write-in-flight suppression window when applying remote content."""
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry = time.monotonic() + effective_ttl
        raw_digest = RollingDigestLRUCache.hash_content(content)
        norm_digest = RollingDigestLRUCache.hash_content(self._normalize(content))
        self._in_flight = InFlightWrite(
            expiry_monotonic=expiry,
            expected_digest=raw_digest,
            expected_normalized_digest=norm_digest,
        )
        logger.debug(
            "Armed write-in-flight window (ttl=%.2fs, raw_digest=%s..)",
            effective_ttl,
            raw_digest[:8],
        )

    def is_write_in_flight_active(self, event_text: str) -> bool:
        """Check if an OS event matches an active write-in-flight window."""
        if self._in_flight is None:
            return False

        now = time.monotonic()
        if now > self._in_flight.expiry_monotonic:
            self._in_flight = None
            return False

        event_raw_digest = RollingDigestLRUCache.hash_content(event_text)
        event_norm_digest = RollingDigestLRUCache.hash_content(self._normalize(event_text))

        if (
            event_raw_digest == self._in_flight.expected_digest
            or event_norm_digest == self._in_flight.expected_normalized_digest
        ):
            logger.debug(
                "Suppressed OS reflection echo via Layer 2 (digest=%s..)",
                event_raw_digest[:8],
            )
            return True

        return False

    def disarm_write_in_flight(self) -> None:
        """Explicitly cancel in-flight suppression."""
        self._in_flight = None

    # =========================================================================
    # Inbound Evaluation Pipeline (Remote -> Linux)
    # =========================================================================

    def evaluate_inbound(
        self,
        content: Any,
        remote_timestamp: int,
        device_id: str,
        is_connect_packet: bool = False,
    ) -> Tuple[bool, str]:
        """Evaluate incoming remote clipboard packet against Layers 1 and 3.

        Returns:
            (is_accepted: bool, reason: str)
        """
        # 1. Type and Size Bounds
        if content is None or not isinstance(content, str):
            return False, "invalid_content_type"

        try:
            byte_len = len(content.encode("utf-8"))
        except UnicodeEncodeError:
            return False, "corrupted_surrogate_encoding"

        if byte_len > MAX_CLIPBOARD_PACKET_BYTES:
            return False, f"payload_exceeds_limit ({byte_len} bytes > {MAX_CLIPBOARD_PACKET_BYTES} bytes)"

        # 2. Layer 3: Sentinel / Invalid Timestamp Check
        if remote_timestamp <= 0:
            return False, "invalid_or_missing_timestamp"

        # 3. Layer 3: Future Clock Drift Check
        now_ms = self.current_wall_ms()
        if remote_timestamp > now_ms + self.max_future_drift_ms:
            return False, "future_clock_drift_exceeded"

        # 4. Layer 3: Timestamp Monotonicity
        if is_connect_packet:
            if remote_timestamp < self._local_timestamp_ms:
                return (
                    False,
                    f"stale_timestamp (remote={remote_timestamp} < local={self._local_timestamp_ms})",
                )
        else:
            if remote_timestamp <= self._local_timestamp_ms:
                return (
                    False,
                    f"stale_timestamp (remote={remote_timestamp} <= local={self._local_timestamp_ms})",
                )

        # 5. Layer 1: Rolling Digest Deduplication Check
        digest = RollingDigestLRUCache.hash_content(content)
        if self._digest_cache.contains(digest):
            return False, f"duplicate_content (digest={digest[:8]}..)"

        return True, "accepted"

    def record_inbound_accepted(
        self,
        content: str,
        remote_timestamp: int,
        ttl: Optional[float] = None,
    ) -> str:
        """Apply accepted remote update to engine state.

        - Updates local timestamp to remote_timestamp
        - Caches digest in Layer 1 LRU
        - Arms Layer 2 write-in-flight window
        """
        self._local_timestamp_ms = max(self._local_timestamp_ms, remote_timestamp)
        digest = self._digest_cache.add_text(content)
        self.arm_write_in_flight(content, ttl=ttl)
        return digest

    # =========================================================================
    # Outbound Evaluation Pipeline (Linux OS Change -> Broadcast)
    # =========================================================================

    def evaluate_outbound_os_event(self, content: Any) -> Tuple[bool, str]:
        """Evaluate local desktop clipboard change event against Layers 1 and 2.

        Returns:
            (is_accepted: bool, reason: str)
        """
        if content is None or not isinstance(content, str):
            return False, "invalid_content_type"
        if not content:
            return False, "empty_content"

        try:
            byte_len = len(content.encode("utf-8"))
        except UnicodeEncodeError:
            return False, "corrupted_surrogate_encoding"

        if byte_len > MAX_CLIPBOARD_PACKET_BYTES:
            return False, f"payload_exceeds_limit ({byte_len} bytes > {MAX_CLIPBOARD_PACKET_BYTES} bytes)"

        # Layer 2: Write-in-Flight Reflection Suppression
        if self.is_write_in_flight_active(content):
            return False, "write_in_flight_reflection"

        # Layer 1: Rolling Digest Deduplication Check
        digest = RollingDigestLRUCache.hash_content(content)
        if self._digest_cache.contains(digest):
            return False, f"duplicate_content (digest={digest[:8]}..)"

        return True, "accepted"

    def record_outbound_accepted(self, content: str) -> Tuple[str, int]:
        """Commit accepted local copy to engine state.

        - Advances local timestamp monotonically: max(local_ts + 1, now_ms)
        - Caches digest in Layer 1 LRU
        - Returns (digest, assigned_timestamp)
        """
        now_ms = self.current_wall_ms()
        self._local_timestamp_ms = max(self._local_timestamp_ms + 1, now_ms)
        digest = self._digest_cache.add_text(content)
        return digest, self._local_timestamp_ms

    # =========================================================================
    # Layer 4: Multi-Device Fanout Filtering
    # =========================================================================

    @staticmethod
    def get_fanout_recipients(
        active_devices: Any,
        originator_device_id: Optional[str] = None,
        capability_checker: Optional[Callable[[str, str], bool]] = None,
        required_capability: str = CLIPBOARD_PACKET_TYPE,
    ) -> List[str]:
        """Select eligible device IDs for fanout transmission, excluding originator."""
        device_ids: List[str] = []
        if isinstance(active_devices, dict):
            for dev_id, dev_obj in active_devices.items():
                is_conn = getattr(dev_obj, "is_connected", True)
                if callable(is_conn):
                    is_conn = is_conn()
                if not is_conn:
                    continue
                device_ids.append(dev_id)
        elif isinstance(active_devices, (set, list, tuple)):
            device_ids = list(active_devices)

        targets: List[str] = []
        for dev_id in device_ids:
            if originator_device_id is not None and dev_id == originator_device_id:
                continue
            if capability_checker is not None:
                if not capability_checker(dev_id, required_capability):
                    continue
            targets.append(dev_id)

        return targets

    def reset(self) -> None:
        """Reset all engine states."""
        self._digest_cache.clear()
        self._in_flight = None
        self._local_timestamp_ms = 0


class ClipboardPlugin(Plugin):
    """KDE Connect Clipboard Synchronization plugin.

    Provides bidirectional clipboard synchronization between Linux desktop
    sessions and connected KDE Connect peers with 4-layer loop suppression.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        backend: Optional[ClipboardBackend] = None,
        packet_sender: Optional[Callable[[str, Packet], Any]] = None,
        capability_checker: Optional[Callable[[str, str], bool]] = None,
        ttl_seconds: float = DEFAULT_WRITE_IN_FLIGHT_TTL,
        digest_cache_size: int = 32,
        max_future_drift_ms: int = MAX_FUTURE_CLOCK_DRIFT_MS,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self._backend: ClipboardBackend = backend or detect_clipboard_backend()
        self._packet_sender = packet_sender
        self._capability_checker = capability_checker

        self._suppression = LoopSuppressionEngine(
            cache_capacity=digest_cache_size,
            write_in_flight_ttl=ttl_seconds,
            max_future_drift_ms=max_future_drift_ms,
        )

        self._connected_devices: Set[str] = set()
        self._device_senders: Dict[str, SendPacketCallable] = {}
        self._last_content: str = ""
        self._last_digest: Optional[str] = None
        self._running: bool = False
        self._background_tasks: Set[asyncio.Task[Any]] = set()
        self._unsub_callbacks: List[Callable[[], None]] = []

    @property
    def plugin_id(self) -> str:
        return "clipboard"

    @property
    def incoming_capabilities(self) -> List[str]:
        return [CLIPBOARD_PACKET_TYPE, CLIPBOARD_CONNECT_PACKET_TYPE]

    @property
    def outgoing_capabilities(self) -> List[str]:
        return [CLIPBOARD_PACKET_TYPE, CLIPBOARD_CONNECT_PACKET_TYPE]

    @property
    def backend(self) -> ClipboardBackend:
        return self._backend

    @property
    def suppression(self) -> LoopSuppressionEngine:
        return self._suppression

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def latest_timestamp(self) -> int:
        return self._suppression.local_timestamp

    @property
    def last_content(self) -> str:
        return self._last_content

    @property
    def last_digest(self) -> Optional[str]:
        return self._last_digest

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def start(self) -> None:
        """Start clipboard backend watcher and subscribe to system events."""
        if self._running:
            return
        self._running = True

        # 1. Wire EventBus Subscriptions
        if self.event_bus:
            self._unsub_callbacks.append(
                self.event_bus.subscribe(DeviceConnected, self._on_device_connected)
            )
            self._unsub_callbacks.append(
                self.event_bus.subscribe(DeviceDisconnected, self._on_device_disconnected)
            )
            self._unsub_callbacks.append(
                self.event_bus.subscribe(DeviceUnpaired, self._on_device_unpaired)
            )

        # 2. Seed Initial Local Clipboard State
        try:
            initial_text = await self._backend.get_text()
            if initial_text and isinstance(initial_text, str):
                try:
                    encoded = initial_text.encode("utf-8")
                except UnicodeEncodeError:
                    encoded = None
                if encoded is not None and len(encoded) <= MAX_CLIPBOARD_PACKET_BYTES:
                    digest = hashlib.sha256(encoded).hexdigest()
                    self._last_content = initial_text
                    self._last_digest = digest
                    now_ms = int(time.time() * 1000)
                    self._suppression.local_timestamp = now_ms
                    self._suppression.digest_cache.add(digest)
        except Exception as exc:
            logger.warning("Could not read initial clipboard state: %s", exc)

        # 3. Start Backend Watcher
        try:
            await self._backend.start_watching(self._on_local_clipboard_changed)
            logger.info("ClipboardPlugin started using backend '%s'", self._backend.name)
        except Exception as exc:
            logger.error("Failed to start clipboard backend '%s': %s", self._backend.name, exc)

    async def stop(self) -> None:
        """Stop backend watcher, cancel pending tasks, and clean up state."""
        if not self._running:
            return
        self._running = False

        # 1. Stop Backend Watcher
        try:
            await self._backend.stop_watching()
        except Exception as exc:
            logger.debug("Error stopping clipboard backend: %s", exc)

        # 2. Unsubscribe EventBus Listeners
        for unsub in self._unsub_callbacks:
            try:
                unsub()
            except Exception:
                pass
        self._unsub_callbacks.clear()

        # 3. Drain and Cancel Background Broadcast Tasks
        tasks = list(self._background_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()

        # 4. Clear Ephemeral State
        self._connected_devices.clear()
        self._device_senders.clear()
        self._suppression.disarm_write_in_flight()
        logger.info("ClipboardPlugin stopped cleanly")

    # =========================================================================
    # Inbound Packet Handling
    # =========================================================================

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Process incoming kdeconnect.clipboard or kdeconnect.clipboard.connect packet."""
        if not self._running:
            logger.debug(
                "Discarding incoming clipboard packet from %s: ClipboardPlugin is not running",
                device_id,
            )
            return

        # Track device sender callback
        self._device_senders[device_id] = send_packet
        self._connected_devices.add(device_id)

        if packet.type not in (CLIPBOARD_PACKET_TYPE, CLIPBOARD_CONNECT_PACKET_TYPE):
            return

        if not isinstance(packet.body, dict):
            logger.warning("Malformed clipboard packet body from %s (not a dict)", device_id)
            return

        content = packet.body.get("content")
        if content is None or not isinstance(content, str):
            logger.warning(
                "Malformed clipboard packet from %s: missing or non-string 'content'",
                device_id,
            )
            return

        raw_ts = packet.body.get("timestamp")
        try:
            timestamp = int(raw_ts) if raw_ts is not None else int(packet.id)
        except (ValueError, TypeError):
            timestamp = int(packet.id) if isinstance(packet.id, (int, float)) and packet.id > 0 else 0

        is_connect = (packet.type == CLIPBOARD_CONNECT_PACKET_TYPE)

        # Evaluate against 4-layer loop suppression
        accepted, reason = self._suppression.evaluate_inbound(
            content=content,
            remote_timestamp=timestamp,
            device_id=device_id,
            is_connect_packet=is_connect,
        )
        if not accepted:
            logger.debug(
                "Discarded incoming clipboard packet from %s: %s (%s)",
                device_id,
                reason,
                redact_clipboard(content),
            )
            # Reconnection convergence: if remote sent stale clipboard on connect, push local
            if is_connect and "stale_timestamp" in reason and self._last_content:
                push_pkt = create_clipboard_connect_packet(
                    self._last_content,
                    self._suppression.local_timestamp,
                )
                task = asyncio.create_task(self._send_to_device(device_id, push_pkt))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            return

        # Record accepted update in suppression engine
        digest = self._suppression.record_inbound_accepted(content, timestamp)
        self._last_content = content
        self._last_digest = digest

        logger.info(
            "Accepted inbound clipboard from %s: %s, ts=%d",
            device_id,
            redact_clipboard(content),
            timestamp,
        )

        # Write to local OS clipboard
        try:
            await self._backend.set_text(content)
        except Exception as exc:
            logger.error("Failed to write remote clipboard text to backend: %s", exc)

        # Publish typed event to EventBus
        if self.event_bus:
            await self.event_bus.publish(
                ClipboardUpdatedEvent(
                    device_id=device_id,
                    content=content,
                    timestamp=timestamp,
                )
            )

        # Fan out to other connected devices (Layer 4: exclude originator)
        fwd_pkt = create_clipboard_packet(content, timestamp)
        await self._broadcast_packet(fwd_pkt, exclude_device_id=device_id)

    # =========================================================================
    # Outbound Local Change Handling
    # =========================================================================

    async def _on_local_clipboard_changed(self, text: str) -> None:
        """Callback invoked by the clipboard backend when local clipboard mutates."""
        if not self._running:
            logger.debug("Discarding local clipboard change: ClipboardPlugin is not running")
            return
        accepted, reason = self._suppression.evaluate_outbound_os_event(text)
        if not accepted:
            logger.debug(
                "Suppressed local clipboard broadcast: %s (%s)",
                reason,
                redact_clipboard(text),
            )
            return

        digest, assigned_ts = self._suppression.record_outbound_accepted(text)
        self._last_content = text
        self._last_digest = digest

        logger.info(
            "Local clipboard updated: %s, ts=%d",
            redact_clipboard(text),
            assigned_ts,
        )

        if self.event_bus:
            await self.event_bus.publish(
                ClipboardUpdatedEvent(
                    device_id=None,
                    content=text,
                    timestamp=assigned_ts,
                )
            )

        out_pkt = create_clipboard_packet(text, assigned_ts)
        await self._broadcast_packet(out_pkt, exclude_device_id=None)

    # =========================================================================
    # Device Management & Lifecycle Event Handlers
    # =========================================================================

    def set_packet_sender(self, sender: Callable[[str, Packet], Any]) -> None:
        """Set a fallback packet sender callback."""
        self._packet_sender = sender

    def set_capability_checker(self, checker: Callable[[str, str], bool]) -> None:
        """Set a peer capability checker callback."""
        self._capability_checker = checker

    def add_connected_device(
        self,
        device_id: str,
        send_packet: Optional[SendPacketCallable] = None,
    ) -> None:
        """Register a connected device and optional sender callback."""
        self._connected_devices.add(device_id)
        if send_packet is not None:
            self._device_senders[device_id] = send_packet

    def remove_connected_device(self, device_id: str) -> None:
        """Remove a disconnected device."""
        self._connected_devices.discard(device_id)
        self._device_senders.pop(device_id, None)

    def clear_device(self, device_id: str) -> None:
        """Clean up all state associated with an unpaired device."""
        self.remove_connected_device(device_id)

    def get_connected_devices(self) -> List[str]:
        """Return list of currently tracked connected device IDs."""
        return sorted(list(self._connected_devices))

    async def _on_device_connected(self, event: DeviceConnected) -> None:
        """Handle peer connection: track device and synchronize state."""
        if not self._running:
            return
        self._connected_devices.add(event.device_id)

        # Check if remote peer supports clipboard
        if self._capability_checker and not self._capability_checker(
            event.device_id, CLIPBOARD_PACKET_TYPE
        ):
            return

        # Synchronize latest local clipboard with newly connected peer
        if self._last_content and self._suppression.local_timestamp > 0:
            connect_pkt = create_clipboard_connect_packet(
                self._last_content,
                self._suppression.local_timestamp,
            )
            task = asyncio.create_task(self._send_to_device(event.device_id, connect_pkt))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _on_device_disconnected(self, event: DeviceDisconnected) -> None:
        """Handle peer disconnection."""
        self._connected_devices.discard(event.device_id)
        self._device_senders.pop(event.device_id, None)

    async def _on_device_unpaired(self, event: DeviceUnpaired) -> None:
        """Handle peer unpairing."""
        self.clear_device(event.device_id)

    # =========================================================================
    # Transmission & Broadcast Helpers
    # =========================================================================

    async def _send_to_device(self, device_id: str, packet: Packet) -> None:
        """Send a packet to a specific device via registered or fallback sender."""
        sender = self._device_senders.get(device_id)
        if sender:
            try:
                await safe_send_packet(sender, packet)
                return
            except Exception as exc:
                logger.debug("Failed sending packet to %s via registered sender: %s", device_id, exc)

        if self._packet_sender:
            try:
                res = self._packet_sender(device_id, packet)
                if inspect.isawaitable(res):
                    await res
            except Exception as exc:
                logger.debug("Failed sending packet to %s via packet_sender: %s", device_id, exc)

    async def _broadcast_packet(
        self,
        packet: Packet,
        exclude_device_id: Optional[str] = None,
    ) -> None:
        """Broadcast packet to all eligible connected devices, excluding originator."""
        if not self._running:
            return
        recipients = self._suppression.get_fanout_recipients(
            self._connected_devices,
            originator_device_id=exclude_device_id,
            capability_checker=self._capability_checker,
            required_capability=CLIPBOARD_PACKET_TYPE,
        )
        for dev_id in recipients:
            task = asyncio.create_task(self._send_to_device(dev_id, packet))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    # =========================================================================
    # Programmatic Inspection & Mutation
    # =========================================================================

    async def get_current_clipboard(self) -> Dict[str, Any]:
        """Return current clipboard state metadata."""
        return {
            "content": self._last_content,
            "timestamp": self._suppression.local_timestamp,
            "digest": self._last_digest,
        }

    async def set_local_clipboard(self, content: str) -> bool:
        """Programmatically set the local clipboard and broadcast if non-empty."""
        if not self._running:
            logger.warning("set_local_clipboard rejected: ClipboardPlugin is not running")
            return False

        if not isinstance(content, str):
            return False

        try:
            encoded = content.encode("utf-8")
        except UnicodeEncodeError:
            logger.warning(
                "set_local_clipboard rejected: content contains unencodable surrogate characters"
            )
            return False

        if len(encoded) > MAX_CLIPBOARD_PACKET_BYTES:
            logger.warning(
                "set_local_clipboard rejected: payload %d bytes exceeds %d bytes ceiling",
                len(encoded),
                MAX_CLIPBOARD_PACKET_BYTES,
            )
            return False

        success = await self._backend.set_text(content)
        if success:
            if content:
                await self._on_local_clipboard_changed(content)
            else:
                self._last_content = ""
                self._last_digest = hashlib.sha256(b"").hexdigest()
        return success

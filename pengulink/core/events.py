"""pengulink.core.events
~~~~~~~~~~~~~~~~~~~~~
Strongly typed asynchronous Event Bus and event hierarchy for PenguLink.

Decouples core networking/protocol from future GNOME/GTK or CLI interfaces.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

if TYPE_CHECKING:
    from pengulink.services.audio import AudioSink

logger = logging.getLogger("pengulink.core.events")


@dataclass(frozen=True)
class Event:
    """Base event for all PenguLink system events."""

    timestamp: float = field(default_factory=time.time)


# =====================================================================
# Device Lifecycle Events
# =====================================================================


@dataclass(frozen=True)
class DeviceDiscovered(Event):
    """Emitted when a new or existing device broadcasts a UDP discovery beacon."""

    device_id: str = ""
    device_name: str = ""
    device_type: str = "desktop"
    host: str = ""
    port: int = 1716
    protocol_version: int = 7


@dataclass(frozen=True)
class DeviceConnected(Event):
    """Emitted when mutual TLS and post-TLS identity authentication completes."""

    device_id: str = ""
    device_name: str = ""
    host: str = ""
    port: int = 1716


@dataclass(frozen=True)
class DeviceDisconnected(Event):
    """Emitted when a device TCP stream terminates or times out."""

    device_id: str = ""
    reason: str = "Connection closed"


@dataclass(frozen=True)
class PairingRequested(Event):
    """Emitted when an incoming pairing request arrives with visual verification code."""

    device_id: str = ""
    device_name: str = ""
    verification_code: str = ""
    pairing_timestamp: int = 0


@dataclass(frozen=True)
class DevicePaired(Event):
    """Emitted when pairing is mutually confirmed and trusted credentials are saved."""

    device_id: str = ""
    device_name: str = ""


@dataclass(frozen=True)
class DeviceUnpaired(Event):
    """Emitted when a device pairing is explicitly revoked or removed."""

    device_id: str = ""


# =====================================================================
# Plugin Events
# =====================================================================


@dataclass(frozen=True)
class PingReceived(Event):
    """Emitted when an incoming kdeconnect.ping packet is received."""

    device_id: str = ""
    message: str = "Ping!"


@dataclass(frozen=True)
class BatteryUpdated(Event):
    """Emitted when battery status information is received from a remote node."""

    device_id: str = ""
    current_charge: int = -1  # 0..100 (-1 = no battery / desktop)
    is_charging: bool = False
    threshold_event: int = 0  # 0 = none, 1 = battery low
    battery_quantity: int = 1


@dataclass(frozen=True)
class NotificationReceived(Event):
    """Emitted when a mobile application notification is delivered."""

    device_id: str = ""
    notification_id: str = ""
    app_name: str = ""
    title: str = ""
    text: str = ""
    ticker: str = ""
    is_clearable: bool = True
    silent: bool = False
    time_str: str = ""
    group_name: str | None = None
    actions: list[str] = field(default_factory=list)
    request_reply_id: str | None = None
    auth_code: str | None = None


@dataclass(frozen=True)
class NotificationCleared(Event):
    """Emitted when a remote notification is dismissed or cancelled."""

    device_id: str = ""
    notification_id: str = ""


@dataclass(frozen=True)
class ConnectivityReportReceived(Event):
    """Emitted when cellular connectivity information is received from a remote node."""

    device_id: str = ""
    signal_strengths: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ClipboardUpdatedEvent(Event):
    """Emitted when clipboard content is updated either locally or remotely."""

    device_id: str | None = None
    content: str = ""
    timestamp: int = 0

    def __repr__(self) -> str:
        digest = hashlib.sha256(
            self.content.encode("utf-8", errors="replace")
        ).hexdigest()[:8]
        return (
            f"ClipboardUpdatedEvent(device_id={self.device_id!r}, "
            f"content='[REDACTED, {len(self.content)} chars, sha256:{digest}]', "
            f"timestamp={self.timestamp})"
        )

    def __str__(self) -> str:
        return self.__repr__()


# Type variable for Event subscribers
E = TypeVar("E", bound=Event)
EventHandler = Callable[[Any], Coroutine[Any, Any, None] | None]


class EventBus:
    """Asynchronous, typed publish-subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: dict[type[Event], list[EventHandler]] = {}
        self._global_subscribers: list[EventHandler] = []
        self._lock = asyncio.Lock()

    def subscribe(
        self, event_type: type[E], handler: Callable[[E], Any]
    ) -> Callable[[], None]:
        """Subscribe a callable to a specific event type.

        Returns an unsubscribe callback.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

        def unsubscribe() -> None:
            self.unsubscribe(event_type, handler)

        return unsubscribe

    def subscribe_all(self, handler: EventHandler) -> Callable[[], None]:
        """Subscribe a callable to all published events."""
        self._global_subscribers.append(handler)

        def unsubscribe() -> None:
            if handler in self._global_subscribers:
                self._global_subscribers.remove(handler)

        return unsubscribe

    def unsubscribe(self, event_type: type[Event], handler: EventHandler) -> None:
        """Unregister an event handler."""
        if event_type in self._subscribers and handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]

    async def publish(self, event: Event) -> None:
        """Publish an event to all registered matching and global subscribers.

        Executes handlers concurrently with full exception protection.
        """
        handlers: list[EventHandler] = []

        # Snapshot subscribers under lock to prevent RuntimeError from concurrent dict mutation
        async with self._lock:
            subscribers_snapshot = [
                (registered_type, list(handler_list))
                for registered_type, handler_list in list(self._subscribers.items())
            ]
            global_snapshot = list(self._global_subscribers)

        # Find exact matches and subclass matches
        for registered_type, handler_list in subscribers_snapshot:
            if isinstance(event, registered_type):
                handlers.extend(handler_list)

        handlers.extend(global_snapshot)

        if not handlers:
            return

        async def _safe_invoke(h: EventHandler) -> None:
            try:
                res = h(event)
                if inspect.isawaitable(res):
                    await res
            except Exception:
                logger.exception(
                    "Unhandled exception in EventBus handler '%s' for event %s",
                    getattr(h, "__name__", str(h)),
                    type(event).__name__,
                )

        await asyncio.gather(
            *(_safe_invoke(h) for h in handlers), return_exceptions=True
        )


# =====================================================================
# Find My Phone & Alert Events
# =====================================================================


@dataclass(frozen=True)
class FindDeviceAlertTriggeredEvent(Event):
    """Emitted when the desktop Find This Device alert is triggered (remote or local)."""

    device_id: str = ""
    source: str = "remote"  # "remote" (Android request) or "local" (CLI test)
    max_duration: float = 30.0


@dataclass(frozen=True)
class FindDeviceAlertStoppedEvent(Event):
    """Emitted when an active Find This Device alert is cancelled or stopped."""

    device_id: str = ""
    reason: str = "toggle"  # "toggle", "manual", "timeout", "shutdown"
    duration_played: float = 0.0


# Canonical aliases for PROJECT.md and AlertService compatibility
AlertTriggered = FindDeviceAlertTriggeredEvent
AlertStopped = FindDeviceAlertStoppedEvent


# =====================================================================
# Share Plugin Events
# =====================================================================


@dataclass(frozen=True)
class ShareReceivedEvent(Event):
    """Emitted when incoming text, URL, or file share is received from a paired device."""

    device_id: str = ""
    share_type: str = "text"  # "text", "url", or "file"
    text: str | None = None
    url: str | None = None
    filename: str | None = None
    file_path: Path | None = None
    file_size: int | None = None
    mime_type: str | None = None


@dataclass(frozen=True)
class FileTransferProgress(Event):
    """Emitted during active binary payload streaming."""

    device_id: str = ""
    transfer_id: str = ""
    filename: str = ""
    direction: str = "incoming"  # "incoming" or "outgoing"
    bytes_transferred: int = 0
    total_bytes: int = 0
    percentage: float = 0.0


@dataclass(frozen=True)
class FileReceivedEvent(Event):
    """Emitted when a file transfer successfully completes and is staged into final path."""

    device_id: str = ""
    file_path: Path = field(default_factory=Path)
    file_size: int = 0
    filename: str = ""


# Aliases
ShareReceived = ShareReceivedEvent
FileReceived = FileReceivedEvent


# =====================================================================
# Contacts Plugin Events
# =====================================================================


@dataclass(frozen=True)
class ContactsSynchronizedEvent(Event):
    """Emitted when contacts synchronization with a device finishes."""

    device_id: str = ""
    total_contacts: int = 0
    contacts_added: int = 0
    contacts_updated: int = 0
    contacts_deleted: int = 0
    duration: float = 0.0


# =====================================================================
# MPRIS Media Control Events
# =====================================================================


@dataclass(frozen=True)
class LocalMediaPlayerChanged(Event):
    """Emitted when a host media player status, track metadata, or progress changes."""

    player: str = ""
    title: str = ""
    artist: str = ""
    album: str = ""
    album_art_url: str = ""
    is_playing: bool = False
    pos_ms: int = 0
    length_ms: int = -1
    volume: int = 100
    can_play: bool = True
    can_pause: bool = True
    can_go_next: bool = True
    can_go_previous: bool = True
    can_seek: bool = True
    loop_status: str = "None"
    shuffle: bool = False


@dataclass(frozen=True)
class LocalPlayerListChanged(Event):
    """Emitted when active media players appear or disappear on the host."""

    player_list: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MediaControlCommandReceived(Event):
    """Emitted when a remote device invokes an MPRIS control action or property change."""

    device_id: str = ""
    player: str = ""
    command: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RemotePlayerUpdatedEvent(Event):
    """Emitted when a remote device's media player status is received (MprisRemote)."""

    device_id: str = ""
    player: str = ""
    title: str = ""
    artist: str = ""
    album: str = ""
    album_art_url: str = ""
    is_playing: bool = False
    pos_ms: int = 0
    length_ms: int = -1
    volume: int = 100


@dataclass(frozen=True)
class RemotePlayerListUpdatedEvent(Event):
    """Emitted when a remote device's available media player list changes (MprisRemote)."""

    device_id: str = ""
    player_list: list[str] = field(default_factory=list)


# =====================================================================
# Remote Commands Events (Milestone 15)
# =====================================================================


@dataclass(frozen=True)
class RemoteCommandsReceivedEvent(Event):
    """Emitted when available commands are received from a remote device."""

    device_id: str = ""
    commands: dict[str, dict[str, Any]] = field(default_factory=dict)
    count: int = 0


@dataclass(frozen=True)
class RemoteCommandExecutedEvent(Event):
    """Emitted when an execution request for a remote command is dispatched."""

    device_id: str = ""
    key: str = ""


@dataclass(frozen=True)
class RemoteCommandOutputEvent(Event):
    """Emitted when execution output feedback is received from a remote device."""

    device_id: str = ""
    key: str = ""
    output: str = ""
    exit_code: int = 0


# =====================================================================
# RunCommand & Local Execution Events (Milestone 23)
# =====================================================================


@dataclass(frozen=True)
class CommandExecutedEvent(Event):
    """Emitted when a local command completes execution triggered by a remote device or locally."""

    device_id: str = ""
    key: str = ""
    command_name: str = ""
    success: bool = True
    exit_code: int = 0


@dataclass(frozen=True)
class CommandOutputEvent(Event):
    """Emitted when output is captured from a running local command process."""

    device_id: str = ""
    key: str = ""
    stream: str = "stdout"
    data: str = ""


# =====================================================================
# SystemVolume & Audio Events (Milestone 22)
# =====================================================================


@dataclass(frozen=True)
class AudioSinkUpdatedEvent(Event):
    """Emitted when an audio sink's volume, mute, or default status is modified."""

    device_id: str | None = None
    sink: AudioSink | None = None


@dataclass(frozen=True)
class AudioSinkListEvent(Event):
    """Emitted when the complete list of audio sinks is updated or enumerated."""

    device_id: str | None = None
    sinks: list[AudioSink] = field(default_factory=list)


# Aliases for event compatibility
LocalAudioSinkListChanged = AudioSinkListEvent
LocalAudioVolumeChanged = AudioSinkUpdatedEvent
DeviceUnpairedEvent = DeviceUnpaired


# =====================================================================
# RemoteSystemVolume Audio Events (Milestone 24)
# =====================================================================


@dataclass(frozen=True)
class RemoteAudioSinkUpdatedEvent(Event):
    """Emitted when a remote device audio sink's volume, mute, or state is updated."""

    device_id: str = ""
    sink: Any = None

    @property
    def sink_name(self) -> str:
        if hasattr(self.sink, "name"):
            return str(self.sink.name)
        return ""

    @property
    def volume(self) -> int:
        if hasattr(self.sink, "volume"):
            return int(self.sink.volume)
        return 0

    @property
    def max_volume(self) -> int:
        if hasattr(self.sink, "max_volume"):
            return int(self.sink.max_volume)
        return 65536

    @property
    def muted(self) -> bool:
        if hasattr(self.sink, "muted"):
            return bool(self.sink.muted)
        return False


@dataclass(frozen=True)
class RemoteAudioSinkListEvent(Event):
    """Emitted when the complete list of remote audio sinks is received or updated."""

    device_id: str = ""
    sinks: list[Any] = field(default_factory=list)


# =====================================================================
# Telephony Events (Milestone 27)
# =====================================================================


@dataclass(frozen=True)
class CallEvent(Event):
    """Emitted when an incoming telephony event packet is received from a paired device."""

    device_id: str = ""
    event_type: str = ""  # "ringing", "talking", "missedCall", "idle", "callEnded"
    phone_number: str = ""
    contact_name: str = ""
    is_cancel: bool = False
    phone_thumbnail: str | None = None

    @property
    def event(self) -> str:
        """Alias for event_type for compatibility."""
        return self.event_type


@dataclass(frozen=True)
class CallStateChangedEvent(Event):
    """Emitted when active call state changes for a device."""

    device_id: str = ""
    old_state: str = "IDLE"  # "IDLE", "RINGING", "TALKING", "MISSED"
    new_state: str = "IDLE"
    phone_number: str = ""
    contact_name: str = ""


__all__ = [
    "AlertStopped",
    "AlertTriggered",
    "AudioSinkListEvent",
    "AudioSinkUpdatedEvent",
    "BatteryUpdated",
    "CallEvent",
    "CallStateChangedEvent",
    "ClipboardUpdatedEvent",
    "CommandExecutedEvent",
    "CommandOutputEvent",
    "ConnectivityReportReceived",
    "ContactsSynchronizedEvent",
    "DeviceConnected",
    "DeviceDisconnected",
    "DeviceDiscovered",
    "DevicePaired",
    "DeviceUnpaired",
    "DeviceUnpairedEvent",
    "Event",
    "EventBus",
    "FileReceived",
    "FileReceivedEvent",
    "FileTransferProgress",
    "FindDeviceAlertStoppedEvent",
    "FindDeviceAlertTriggeredEvent",
    "LocalAudioSinkListChanged",
    "LocalAudioVolumeChanged",
    "LocalMediaPlayerChanged",
    "LocalPlayerListChanged",
    "MediaControlCommandReceived",
    "NotificationCleared",
    "NotificationReceived",
    "PairingRequested",
    "PingReceived",
    "RemoteAudioSinkListEvent",
    "RemoteAudioSinkUpdatedEvent",
    "RemoteCommandExecutedEvent",
    "RemoteCommandOutputEvent",
    "RemoteCommandsReceivedEvent",
    "RemotePlayerListUpdatedEvent",
    "RemotePlayerUpdatedEvent",
    "ShareReceived",
    "ShareReceivedEvent",
]


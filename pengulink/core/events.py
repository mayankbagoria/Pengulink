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
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Type, TypeVar, Union

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
    group_name: Optional[str] = None
    actions: List[str] = field(default_factory=list)
    request_reply_id: Optional[str] = None
    auth_code: Optional[str] = None


@dataclass(frozen=True)
class NotificationCleared(Event):
    """Emitted when a remote notification is dismissed or cancelled."""

    device_id: str = ""
    notification_id: str = ""


@dataclass(frozen=True)
class ConnectivityReportReceived(Event):
    """Emitted when cellular connectivity information is received from a remote node."""

    device_id: str = ""
    signal_strengths: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ClipboardUpdatedEvent(Event):
    """Emitted when clipboard content is updated either locally or remotely."""

    device_id: Optional[str] = None
    content: str = ""
    timestamp: int = 0

    def __repr__(self) -> str:
        digest = hashlib.sha256(self.content.encode("utf-8", errors="replace")).hexdigest()[:8]
        return (
            f"ClipboardUpdatedEvent(device_id={self.device_id!r}, "
            f"content='[REDACTED, {len(self.content)} chars, sha256:{digest}]', "
            f"timestamp={self.timestamp})"
        )

    def __str__(self) -> str:
        return self.__repr__()


# Type variable for Event subscribers
E = TypeVar("E", bound=Event)
EventHandler = Callable[[Any], Union[Coroutine[Any, Any, None], None]]


class EventBus:
    """Asynchronous, typed publish-subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: Dict[Type[Event], List[EventHandler]] = {}
        self._global_subscribers: List[EventHandler] = []
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: Type[E], handler: Callable[[E], Any]) -> Callable[[], None]:
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

    def unsubscribe(self, event_type: Type[Event], handler: EventHandler) -> None:
        """Unregister an event handler."""
        if event_type in self._subscribers and handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]

    async def publish(self, event: Event) -> None:
        """Publish an event to all registered matching and global subscribers.

        Executes handlers concurrently with full exception protection.
        """
        handlers: List[EventHandler] = []

        # Find exact matches and subclass matches
        for registered_type, handler_list in self._subscribers.items():
            if isinstance(event, registered_type):
                handlers.extend(handler_list)

        handlers.extend(self._global_subscribers)

        if not handlers:
            return

        async def _safe_invoke(h: EventHandler) -> None:
            try:
                res = h(event)
                if inspect.isawaitable(res):
                    await res
            except Exception as exc:
                logger.error(
                    "Unhandled exception in EventBus handler '%s' for event %s: %s",
                    getattr(h, "__name__", str(h)),
                    type(event).__name__,
                    exc,
                    exc_info=True,
                )

        await asyncio.gather(*(_safe_invoke(h) for h in handlers), return_exceptions=True)


__all__ = [
    "Event",
    "DeviceDiscovered",
    "DeviceConnected",
    "DeviceDisconnected",
    "PairingRequested",
    "DevicePaired",
    "DeviceUnpaired",
    "PingReceived",
    "BatteryUpdated",
    "NotificationReceived",
    "NotificationCleared",
    "ConnectivityReportReceived",
    "ClipboardUpdatedEvent",
    "EventBus",
]

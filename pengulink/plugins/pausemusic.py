"""pengulink.plugins.pausemusic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect PauseMusic plugin for host media playback and volume coordination.

Coordinates:
- Host media player pause/resume (MPRIS v2) during incoming or active phone calls.
- Host system audio sink mute/unmute during call interruptions.
- Subscribes to central EventBus domain events (CallStateChangedEvent, DeviceDisconnected, DeviceUnpaired).
- Delegates state transitions to PauseMusicService.
- Safely handles direct kdeconnect.pausemusic packets without error.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pengulink.core.events import (
    CallStateChangedEvent,
    DeviceDisconnected,
    DeviceUnpaired,
    DeviceUnpairedEvent,
    EventBus,
)
from pengulink.core.exceptions import PenguLinkError
from pengulink.network.packet import Packet
from pengulink.plugins.base import Plugin, SendPacketCallable

if TYPE_CHECKING:
    from pengulink.services.pausemusic import PauseMusicConfig, PauseMusicService

logger = logging.getLogger("pengulink.plugins.pausemusic")

# Upstream KDE Connect capability identifier
PAUSE_MUSIC_PACKET_TYPE = "kdeconnect.pausemusic"

LIFECYCLE_EXCEPTIONS = (
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    KeyError,
)

PLUGIN_DISPATCH_EXCEPTIONS = (
    PenguLinkError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    OSError,
    TimeoutError,
)


class PauseMusicPlugin(Plugin):
    """KDE Connect PauseMusic plugin for host media playback and volume coordination."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        service: PauseMusicService | None = None,
        pause_music_service: PauseMusicService | None = None,
    ) -> None:
        """Initialize PauseMusicPlugin.

        Args:
            event_bus: Optional central EventBus for lifecycle and call subscriptions.
            service: Optional PauseMusicService managing media/audio interruption state.
            pause_music_service: Alias for service parameter.
        """
        super().__init__(event_bus=event_bus)

        resolved_service = service or pause_music_service
        if resolved_service is None:
            try:
                from pengulink.services.pausemusic import PauseMusicService

                # Pass None for event_bus so PauseMusicPlugin manages the EventBus subscription
                resolved_service = PauseMusicService(event_bus=None)
            except ImportError:
                resolved_service = None

        self.service: PauseMusicService | None = resolved_service
        self._unsub_callbacks: list[Callable[[], None]] = []
        self._started: bool = False

    @property
    def plugin_id(self) -> str:
        """Unique plugin identifier matching KDE Connect plugin specification."""
        return "pausemusic"

    @property
    def incoming_capabilities(self) -> list[str]:
        """Packet types advertised in incoming capabilities for discovery identity."""
        return [PAUSE_MUSIC_PACKET_TYPE]

    @property
    def outgoing_capabilities(self) -> list[str]:
        """Packet types this plugin transmits to remote peers (none for host automation)."""
        return []

    @property
    def pause_music_service(self) -> Any | None:
        """Convenience alias for self.service."""
        return self.service

    @property
    def config(self) -> PauseMusicConfig | None:
        """Current PauseMusic configuration if service is present."""
        if self.service is not None and hasattr(self.service, "config"):
            return self.service.config
        return None

    @property
    def is_interrupted(self) -> bool:
        """True if an active media interruption is in progress."""
        if self.service is not None and hasattr(self.service, "is_interrupted"):
            return bool(self.service.is_interrupted)
        return False

    @property
    def active_call_devices(self) -> set[str]:
        """Set of device IDs currently causing active media interruption."""
        if self.service is not None and hasattr(self.service, "active_call_devices"):
            return set(self.service.active_call_devices)
        return set()

    @property
    def paused_players(self) -> set[str]:
        """Set of MPRIS players that were paused by the current interruption."""
        if self.service is not None and hasattr(self.service, "paused_players"):
            return set(self.service.paused_players)
        return set()

    @property
    def muted_sinks(self) -> set[str]:
        """Set of audio sinks that were muted by the current interruption."""
        if self.service is not None and hasattr(self.service, "muted_sinks"):
            return set(self.service.muted_sinks)
        return set()

    async def start(self) -> None:
        """Lifecycle hook: subscribe to call state changes and device lifecycle events."""
        await super().start()
        if self._started:
            return

        if self.event_bus:
            # Subscribe to call state transitions
            self._unsub_callbacks.append(
                self.event_bus.subscribe(CallStateChangedEvent, self._on_call_state_changed)
            )
            # Subscribe to device disconnect and unpair lifecycle events
            self._unsub_callbacks.append(
                self.event_bus.subscribe(DeviceDisconnected, self._on_device_disconnected)
            )
            self._unsub_callbacks.append(
                self.event_bus.subscribe(DeviceUnpaired, self._on_device_unpaired)
            )
            if DeviceUnpairedEvent is not DeviceUnpaired:
                self._unsub_callbacks.append(
                    self.event_bus.subscribe(DeviceUnpairedEvent, self._on_device_unpaired)
                )

        if self.service is not None and hasattr(self.service, "start") and callable(self.service.start):
            res = self.service.start()
            if inspect.isawaitable(res):
                await res

        self._started = True
        logger.info("PauseMusicPlugin started and subscribed to lifecycle events")

    async def stop(self) -> None:
        """Lifecycle hook: unsubscribe from events and restore media if active interruption exists."""
        for unsub in self._unsub_callbacks:
            try:
                unsub()
            except LIFECYCLE_EXCEPTIONS as exc:
                logger.debug("Error unsubscribing event in PauseMusicPlugin: %s", exc)
        self._unsub_callbacks.clear()

        # If an active interruption exists when stopping, restore host media playback and volume
        if self.service is not None:
            for method_name in ("restore_media", "resume_all", "clear_all", "stop"):
                method = getattr(self.service, method_name, None)
                if callable(method):
                    try:
                        res = method()
                        if inspect.isawaitable(res):
                            await res
                        break
                    except PLUGIN_DISPATCH_EXCEPTIONS as exc:
                        logger.error("Error restoring media in PauseMusicPlugin during stop: %s", exc)

        self._started = False
        await super().stop()
        logger.info("PauseMusicPlugin stopped cleanly")

    async def _on_call_state_changed(self, event: CallStateChangedEvent) -> None:
        """Process CallStateChangedEvent and delegate state transition to PauseMusicService."""
        device_id = getattr(event, "device_id", "")
        new_state = getattr(event, "new_state", "")

        if not device_id or not new_state:
            return

        clean_device_id = str(device_id).strip()
        clean_new_state = str(new_state).strip()

        if self.service is not None and hasattr(self.service, "handle_call_state_changed"):
            try:
                res = self.service.handle_call_state_changed(clean_device_id, clean_new_state)
                if inspect.isawaitable(res):
                    await res
            except PLUGIN_DISPATCH_EXCEPTIONS as exc:
                logger.warning(
                    "Error delegating call state changed (%s -> %s) in PauseMusicPlugin: %s",
                    clean_device_id,
                    clean_new_state,
                    exc,
                )

    async def _on_device_disconnected(self, event: DeviceDisconnected) -> None:
        """Handle device disconnection by clearing its call interruption and restoring media."""
        device_id = getattr(event, "device_id", "")
        if not device_id:
            return

        clean_device_id = str(device_id).strip()
        if self.service is not None:
            try:
                if hasattr(self.service, "handle_device_disconnected"):
                    res = self.service.handle_device_disconnected(clean_device_id)
                    if inspect.isawaitable(res):
                        await res
                elif hasattr(self.service, "clear_device"):
                    res = self.service.clear_device(clean_device_id)
                    if inspect.isawaitable(res):
                        await res
            except PLUGIN_DISPATCH_EXCEPTIONS as exc:
                logger.warning(
                    "Error handling device disconnection for %s in PauseMusicPlugin: %s",
                    clean_device_id,
                    exc,
                )

    async def _on_device_unpaired(self, event: DeviceUnpaired) -> None:
        """Handle device unpairing by clearing its call interruption and restoring media."""
        device_id = getattr(event, "device_id", "")
        if not device_id:
            return

        clean_device_id = str(device_id).strip()
        if self.service is not None:
            try:
                if hasattr(self.service, "handle_device_disconnected"):
                    res = self.service.handle_device_disconnected(clean_device_id)
                    if inspect.isawaitable(res):
                        await res
                elif hasattr(self.service, "clear_device"):
                    res = self.service.clear_device(clean_device_id)
                    if inspect.isawaitable(res):
                        await res
            except PLUGIN_DISPATCH_EXCEPTIONS as exc:
                logger.warning(
                    "Error handling device unpairing for %s in PauseMusicPlugin: %s",
                    clean_device_id,
                    exc,
                )

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Process direct incoming packet defensively.

        Safely ignores or acknowledges any direct kdeconnect.pausemusic packets without error.
        """
        clean_device_id = str(device_id).strip() if device_id else ""
        packet_type = getattr(packet, "type", "")

        if packet_type != PAUSE_MUSIC_PACKET_TYPE:
            logger.debug(
                "PauseMusicPlugin safely ignoring packet with unexpected type '%s' from device %s",
                packet_type,
                clean_device_id,
            )
            return

        if not clean_device_id:
            logger.warning("PauseMusicPlugin received packet with empty or missing device_id")
            return

        body = getattr(packet, "body", None)
        if not isinstance(body, dict):
            logger.warning(
                "PauseMusicPlugin ignoring packet with non-dict body from device %s",
                clean_device_id,
            )
            return

        logger.debug(
            "PauseMusicPlugin acknowledged incoming packet '%s' from %s (payload keys: %s)",
            packet_type,
            clean_device_id,
            list(body.keys()),
        )

    async def get_status(self) -> dict[str, Any]:
        """Return operational status and policy configuration of the PauseMusic subsystem."""
        if self.service is not None and hasattr(self.service, "get_status"):
            res = self.service.get_status()
            if inspect.isawaitable(res):
                return await res
            return res
        return {
            "is_interrupted": self.is_interrupted,
            "active_devices": list(self.active_call_devices),
            "paused_players": list(self.paused_players),
            "muted_sinks": list(self.muted_sinks),
        }

    async def update_config(self, **kwargs: Any) -> Any:
        """Update PauseMusic policy configuration."""
        if self.service is not None and hasattr(self.service, "update_config"):
            res = self.service.update_config(**kwargs)
            if inspect.isawaitable(res):
                return await res
            return res
        return None

    async def restore_media(self) -> None:
        """Manually restore media playback and unmute sinks, clearing active interruption."""
        if self.service is not None and hasattr(self.service, "restore_media"):
            res = self.service.restore_media()
            if inspect.isawaitable(res):
                await res


__all__ = [
    "PAUSE_MUSIC_PACKET_TYPE",
    "PauseMusicPlugin",
]

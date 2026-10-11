"""pengulink.plugins.findmyphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect Find My Phone and Find This Device plugin.

Handles:
- Outbound 'ring_phone': sends 'kdeconnect.findmyphone.request' to ring remote device.
- Inbound packet handling: receives 'kdeconnect.findmyphone.request' to ring desktop.
- Toggle cancellation: second incoming locate request silences active desktop alert.
- Lifecycle coordination with AlertService and EventBus.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

from pengulink.core.events import (
    EventBus,
    FindDeviceAlertStoppedEvent,
    FindDeviceAlertTriggeredEvent,
)
from pengulink.core.exceptions import ConnectionClosedError
from pengulink.network.packet import Packet
from pengulink.plugins.base import Plugin, SendPacketCallable, safe_send_packet
from pengulink.services.alert import _sanitize_log_str

logger = logging.getLogger("pengulink.plugins.findmyphone")

# Upstream KDE Connect packet types
FINDMYPHONE_REQUEST_TYPE = "kdeconnect.findmyphone.request"
FINDMYPHONE_LEGACY_TYPE = "kdeconnect.findmyphone"


def create_find_my_phone_request_packet(body: Optional[Dict[str, Any]] = None) -> Packet:
    """Construct a kdeconnect.findmyphone.request packet.

    Upstream specification mandates an empty body object by default.
    """
    payload = dict(body) if isinstance(body, dict) else {}
    return Packet(
        type=FINDMYPHONE_REQUEST_TYPE,
        body=payload,
    )


class FindMyPhonePlugin(Plugin):
    """Plugin implementing bidirectional Find My Phone and Find This Device operations."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        alert_service: Optional[Any] = None,
        packet_sender: Optional[SendPacketCallable] = None,
        is_paired_checker: Optional[Callable[[str], Union[bool, Coroutine[Any, Any, bool]]]] = None,
    ) -> None:
        """Initialize FindMyPhonePlugin.

        Args:
            event_bus: Optional EventBus for publishing alert events.
            alert_service: Optional AlertService instance managing audio & notification alerts.
            packet_sender: Optional default packet sender callback.
            is_paired_checker: Optional callback checking if remote device ID is authenticated/paired.
        """
        super().__init__(event_bus=event_bus)
        self.alert_service = alert_service
        self.packet_sender = packet_sender
        self.is_paired_checker = is_paired_checker
        self._fallback_active: bool = False
        self._lock = asyncio.Lock()

    @property
    def plugin_id(self) -> str:
        return "findmyphone"

    @property
    def incoming_capabilities(self) -> List[str]:
        """Packet types this plugin handles (Find This Device receiver)."""
        return [FINDMYPHONE_REQUEST_TYPE, FINDMYPHONE_LEGACY_TYPE]

    @property
    def outgoing_capabilities(self) -> List[str]:
        """Packet types this plugin can emit (Find My Phone sender)."""
        return [FINDMYPHONE_REQUEST_TYPE, FINDMYPHONE_LEGACY_TYPE]

    def _is_alert_active(self) -> bool:
        """Check whether alert is currently active."""
        if self.alert_service is not None:
            active_attr = getattr(self.alert_service, "is_active", None)
            if callable(active_attr):
                try:
                    return bool(active_attr())
                except Exception:
                    pass
            elif isinstance(active_attr, bool):
                return active_attr
            return bool(getattr(self.alert_service, "active", False))
        return self._fallback_active

    async def _start_alert(self, device_id: str, source: str = "remote") -> None:
        """Invoke alert start on AlertService or fallback state."""
        if self.alert_service is not None:
            start_fn = getattr(self.alert_service, "start_alert", None)
            if callable(start_fn):
                res = start_fn(device_id=device_id, source=source)
                if inspect.isawaitable(res):
                    await res
            has_real_bus = isinstance(getattr(self.alert_service, "_event_bus", None), EventBus)
            if self.event_bus and not has_real_bus:
                await self.event_bus.publish(
                    FindDeviceAlertTriggeredEvent(device_id=device_id, source=source)
                )
        else:
            self._fallback_active = True
            if self.event_bus:
                await self.event_bus.publish(
                    FindDeviceAlertTriggeredEvent(device_id=device_id, source=source)
                )

    async def _stop_alert(self, reason: str = "toggle") -> None:
        """Invoke alert stop on AlertService or fallback state."""
        if self.alert_service is not None:
            stop_fn = getattr(self.alert_service, "stop_alert", None)
            if callable(stop_fn):
                res = stop_fn(reason=reason)
                if inspect.isawaitable(res):
                    await res
            has_real_bus = isinstance(getattr(self.alert_service, "_event_bus", None), EventBus)
            if self.event_bus and not has_real_bus:
                await self.event_bus.publish(
                    FindDeviceAlertStoppedEvent(device_id="", reason=reason)
                )
        else:
            self._fallback_active = False
            if self.event_bus:
                await self.event_bus.publish(
                    FindDeviceAlertStoppedEvent(device_id="", reason=reason)
                )

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Handle incoming findmyphone request packet from remote device (Find This Device).

        Implements toggle cancellation:
        - If alert is inactive: starts alert and emits FindDeviceAlertTriggeredEvent.
        - If alert is already active: stops alert and emits FindDeviceAlertStoppedEvent.
        """
        if packet.type not in (FINDMYPHONE_REQUEST_TYPE, FINDMYPHONE_LEGACY_TYPE):
            logger.debug("FindMyPhonePlugin ignoring unhandled packet type '%s'", packet.type)
            return

        if not device_id or not isinstance(device_id, str) or not device_id.strip():
            logger.warning(
                "FindMyPhonePlugin ignoring locate request with empty or invalid device_id: %r",
                device_id,
            )
            return

        if hasattr(packet, "body") and not isinstance(packet.body, dict):
            logger.warning(
                "FindMyPhonePlugin ignoring packet with malformed non-dict body from device %s",
                _sanitize_log_str(device_id),
            )
            return

        if self.is_paired_checker is not None:
            check_res = self.is_paired_checker(device_id)
            if inspect.isawaitable(check_res):
                check_res = await check_res
            if not check_res:
                logger.warning(
                    "FindMyPhonePlugin rejecting locate packet from unauthenticated/unpaired device '%s'",
                    _sanitize_log_str(device_id),
                )
                return

        async with self._lock:
            currently_active = self._is_alert_active()

            if currently_active:
                logger.info(
                    "Find My Phone alert already active; toggling OFF on request from device %s",
                    _sanitize_log_str(device_id),
                )
                await self._stop_alert(reason="toggle")
            else:
                logger.info(
                    "Find My Phone alert triggered by remote device %s",
                    _sanitize_log_str(device_id),
                )
                await self._start_alert(device_id=device_id, source="remote")

    async def ring_phone(
        self,
        device_id: str,
        send_packet: Optional[SendPacketCallable] = None,
    ) -> Packet:
        """Send a findmyphone request packet to remote device (Find My Phone).

        Args:
            device_id: Target device identifier.
            send_packet: Callable or coroutine transmitting wire packet to device.

        Returns:
            Dispatched Packet.

        Raises:
            ConnectionClosedError: If peer is disconnected or transmission fails.
        """
        # Defensive parameter ordering support: allow ring_phone(send_packet, device_id)
        if callable(device_id) and isinstance(send_packet, str):
            device_id, send_packet = send_packet, device_id  # type: ignore

        if not device_id or not isinstance(device_id, str) or not device_id.strip():
            raise ConnectionClosedError(
                f"Cannot ring device '{device_id}': invalid or unresolvable device ID"
            )

        sender = send_packet or self.packet_sender
        if not callable(sender):
            raise ConnectionClosedError(
                f"Cannot ring device '{device_id}': send_packet callable is invalid or missing"
            )

        packet = create_find_my_phone_request_packet()
        logger.info("Transmitting Find My Phone request to device '%s'", _sanitize_log_str(device_id))

        try:
            try:
                await safe_send_packet(sender, packet)
            except TypeError:
                res = sender(device_id, packet)
                if inspect.isawaitable(res):
                    await res
        except ConnectionClosedError:
            raise
        except Exception as exc:
            logger.error("Failed to transmit ring request to device '%s': %s", _sanitize_log_str(device_id), exc)
            raise ConnectionClosedError(
                f"Failed to transmit ring request to device '{device_id}': {exc}"
            ) from exc

        return packet

    async def stop(self) -> None:
        """Lifecycle hook: cleanly terminate any running alert on shutdown."""
        async with self._lock:
            if self._is_alert_active():
                logger.info("FindMyPhonePlugin stopping; terminating active alert")
                await self._stop_alert(reason="shutdown")


__all__ = [
    "FINDMYPHONE_REQUEST_TYPE",
    "FINDMYPHONE_LEGACY_TYPE",
    "create_find_my_phone_request_packet",
    "FindMyPhonePlugin",
]

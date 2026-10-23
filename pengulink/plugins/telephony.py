"""pengulink.plugins.telephony
~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect Telephony plugin implementation.

Handles:
- Inbound 'kdeconnect.telephony' event packets from remote paired devices.
- Parsing call states: ringing, talking, missedCall, idle, callEnded.
- Safe defaults and coercion for phone number, contact name, and isCancel.
- Delegation to TelephonyService for call tracking and EventBus publication.
- Outbound action dispatch: 'mute_ringer' and 'reject_call'.
- Lifecycle tracking and multi-device connection isolation.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from pengulink.core.events import (
    DeviceDisconnected,
    DeviceUnpaired,
    DeviceUnpairedEvent,
    EventBus,
)
from pengulink.core.exceptions import ConnectionClosedError
from pengulink.network.packet import Packet
from pengulink.plugins.base import Plugin, SendPacketCallable, safe_send_packet
from pengulink.services.telephony import TelephonyService

logger = logging.getLogger("pengulink.plugins.telephony")

# Upstream KDE Connect packet types
TELEPHONY_PACKET_TYPE = "kdeconnect.telephony"
TELEPHONY_REQUEST_MUTE_TYPE = "kdeconnect.telephony.request_mute"
TELEPHONY_REQUEST_TYPE = "kdeconnect.telephony.request"

# Known valid call event types
KNOWN_CALL_EVENTS = frozenset({"ringing", "talking", "missedCall", "idle", "callEnded"})

# String bounds to prevent memory exhaustion / log injection
MAX_EVENT_LEN = 64
MAX_PHONE_NUMBER_LEN = 128
MAX_CONTACT_NAME_LEN = 1024
MAX_THUMBNAIL_LEN = 100000


def create_telephony_mute_packet() -> Packet:
    """Construct an outbound kdeconnect.telephony.request_mute packet."""
    return Packet(
        type=TELEPHONY_REQUEST_MUTE_TYPE,
        body={},
    )


def create_telephony_reject_packet(phone_number: str | None = None) -> Packet:
    """Construct an outbound kdeconnect.telephony.request packet to reject a call."""
    body: dict[str, Any] = {"action": "rejectCall"}
    if phone_number:
        body["phoneNumber"] = str(phone_number)[:MAX_PHONE_NUMBER_LEN]
    return Packet(
        type=TELEPHONY_REQUEST_TYPE,
        body=body,
    )


def coerce_boolean(val: Any) -> bool:
    """Safely coerce truthy string or boolean values matching KDE Connect semantics."""
    if val is True:
        return True
    if val is False or val is None:
        return False
    if isinstance(val, (int, float)):
        return val == 1
    val_str = str(val).strip().lower()
    return val_str in ("true", "1", "yes")


class TelephonyPlugin(Plugin):
    """KDE Connect Telephony plugin for handling calls and triggering call actions."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        telephony_service: TelephonyService | None = None,
        packet_sender: SendPacketCallable | None = None,
        is_paired_checker: Callable[[str], bool | Coroutine[Any, Any, bool]] | None = None,
    ) -> None:
        """Initialize TelephonyPlugin.

        Args:
            event_bus: Optional EventBus for lifecycle subscriptions and events.
            telephony_service: Optional TelephonyService managing active call states.
            packet_sender: Optional default packet sender callback.
            is_paired_checker: Optional callback checking if remote device ID is authenticated/paired.
        """
        super().__init__(event_bus=event_bus)
        self.telephony_service = telephony_service or TelephonyService(event_bus=event_bus)
        self.packet_sender = packet_sender
        self.is_paired_checker = is_paired_checker
        self._connected_peers: dict[str, SendPacketCallable] = {}
        self._unsub_callbacks: list[Callable[[], None]] = []

    @property
    def plugin_id(self) -> str:
        return "telephony"

    @property
    def incoming_capabilities(self) -> list[str]:
        """Packet types this plugin can receive from remote peers."""
        return [TELEPHONY_PACKET_TYPE]

    @property
    def outgoing_capabilities(self) -> list[str]:
        """Packet types this plugin can transmit to remote peers."""
        return [TELEPHONY_REQUEST_MUTE_TYPE, TELEPHONY_REQUEST_TYPE]

    async def start(self) -> None:
        """Lifecycle hook: subscribe to device lifecycle events."""
        await super().start()
        if self.event_bus:
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

    async def stop(self) -> None:
        """Lifecycle hook: unsubscribe from events and clear state."""
        for unsub in self._unsub_callbacks:
            try:
                unsub()
            except (AttributeError, RuntimeError, TypeError, ValueError, KeyError) as exc:
                logger.debug("Error unsubscribing in TelephonyPlugin: %s", exc)
        self._unsub_callbacks.clear()
        self._connected_peers.clear()
        if self.telephony_service:
            await self.telephony_service.clear_all()
        await super().stop()

    async def _on_device_disconnected(self, event: DeviceDisconnected) -> None:
        """Prune peer callback and clear call state upon device disconnection."""
        device_id = getattr(event, "device_id", "")
        if device_id:
            self._connected_peers.pop(device_id, None)
            if self.telephony_service and hasattr(self.telephony_service, "clear_device"):
                res = self.telephony_service.clear_device(device_id)
                if inspect.isawaitable(res):
                    await res

    async def _on_device_unpaired(self, event: DeviceUnpaired) -> None:
        """Prune peer callback and clear call state upon device unpairing."""
        device_id = getattr(event, "device_id", "")
        if device_id:
            self._connected_peers.pop(device_id, None)
            if self.telephony_service and hasattr(self.telephony_service, "clear_device"):
                res = self.telephony_service.clear_device(device_id)
                if inspect.isawaitable(res):
                    await res

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Process incoming kdeconnect.telephony packet defensively.

        Validates payload structure, extracts call event metadata with safe defaults,
        coerces isCancel, and delegates state transition to TelephonyService.
        """
        if packet.type != TELEPHONY_PACKET_TYPE:
            logger.debug("TelephonyPlugin ignoring packet of type '%s'", packet.type)
            return

        if not device_id or not isinstance(device_id, str) or not device_id.strip():
            logger.warning("TelephonyPlugin received packet with invalid device_id: %r", device_id)
            return

        clean_device_id = device_id.strip()

        if not hasattr(packet, "body") or not isinstance(packet.body, dict):
            logger.warning(
                "TelephonyPlugin ignoring packet with non-dict body from device %s",
                clean_device_id,
            )
            return

        if self.is_paired_checker is not None:
            check_res = self.is_paired_checker(clean_device_id)
            if inspect.isawaitable(check_res):
                check_res = await check_res
            if not check_res:
                logger.warning(
                    "TelephonyPlugin rejecting packet from unauthenticated device %s",
                    clean_device_id,
                )
                return

        if send_packet is not None:
            self._connected_peers[clean_device_id] = send_packet

        body = packet.body

        # Parse and coerce isCancel
        is_cancel = coerce_boolean(body.get("isCancel"))

        # Parse event
        raw_event = body.get("event")
        if raw_event is None:
            if is_cancel:
                event = "idle"
            else:
                logger.warning(
                    "TelephonyPlugin received packet missing 'event' from device %s",
                    clean_device_id,
                )
                return
        elif isinstance(raw_event, str):
            event = raw_event.strip()[:MAX_EVENT_LEN]
        else:
            logger.warning(
                "TelephonyPlugin received non-string event %r from device %s",
                raw_event,
                clean_device_id,
            )
            return

        # Legacy SMS filtering
        if event.lower() == "sms":
            logger.debug(
                "TelephonyPlugin safely ignoring legacy 'sms' event from device %s",
                clean_device_id,
            )
            return

        # Parse phone number with safe fallback
        raw_phone = body.get("phoneNumber")
        if raw_phone is not None and isinstance(raw_phone, (str, int)):
            phone_number = str(raw_phone).strip()[:MAX_PHONE_NUMBER_LEN]
            if not phone_number:
                phone_number = "unknown number"
        else:
            phone_number = "unknown number"

        # Parse contact name with fallback to phone_number
        raw_contact = body.get("contactName")
        if raw_contact is not None and isinstance(raw_contact, str) and raw_contact.strip():
            contact_name = raw_contact[:MAX_CONTACT_NAME_LEN]
        else:
            contact_name = phone_number

        # Parse phone thumbnail (base64)
        raw_thumb = body.get("phoneThumbnail")
        if raw_thumb is not None and isinstance(raw_thumb, str) and raw_thumb.strip():
            phone_thumbnail = raw_thumb.strip()[:MAX_THUMBNAIL_LEN]
        else:
            phone_thumbnail = None

        logger.info(
            "Telephony event '%s' from %s (number: %s, contact: %s, isCancel: %s)",
            event,
            clean_device_id,
            phone_number,
            contact_name,
            is_cancel,
        )

        if self.telephony_service is not None:
            try:
                res = self.telephony_service.process_call_event(
                    device_id=clean_device_id,
                    event=event,
                    phone_number=phone_number,
                    contact_name=contact_name,
                    is_cancel=is_cancel,
                    phone_thumbnail=phone_thumbnail,
                )
                if inspect.isawaitable(res):
                    await res
            except Exception:
                logger.exception(
                    "Error delegating telephony event to TelephonyService for device %s",
                    clean_device_id,
                )

    async def _send_to_device(
        self,
        device_id: str,
        packet: Packet,
        send_packet: SendPacketCallable | None = None,
    ) -> bool:
        """Route outbound packet via provided send_packet callback, cached peer, or packet_sender."""
        if not device_id or not isinstance(device_id, str) or not device_id.strip():
            raise ConnectionClosedError(
                f"Cannot transmit packet: invalid or missing device ID '{device_id}'"
            )

        clean_device_id = device_id.strip()
        sender = send_packet or self._connected_peers.get(clean_device_id) or self.packet_sender

        if sender is None or not callable(sender):
            raise ConnectionClosedError(
                f"Cannot transmit packet to device '{clean_device_id}': no active connection or packet sender"
            )

        try:
            try:
                await safe_send_packet(sender, packet)
            except TypeError:
                res = sender(clean_device_id, packet)
                if inspect.isawaitable(res):
                    await res
        except ConnectionClosedError:
            raise
        except Exception as exc:
            logger.error("Failed to transmit packet to device '%s': %s", clean_device_id, exc)
            raise ConnectionClosedError(
                f"Failed to transmit packet to device '{clean_device_id}': {exc}"
            ) from exc

        return True

    async def mute_ringer(
        self,
        device_id: str,
        send_packet: SendPacketCallable | None = None,
    ) -> bool:
        """Request the remote peer to silence its ringer during an incoming call.

        Args:
            device_id: Target remote device identifier.
            send_packet: Optional send packet callable or coroutine.

        Returns:
            True on successful transmission.

        Raises:
            ConnectionClosedError: If peer is disconnected or sender is unavailable.
        """
        # Defensive parameter ordering support: allow mute_ringer(send_packet, device_id)
        if callable(device_id) and isinstance(send_packet, str):
            device_id, send_packet = send_packet, device_id  # type: ignore

        packet = create_telephony_mute_packet()
        logger.info("Transmitting mute ringer request to device '%s'", device_id)
        return await self._send_to_device(device_id, packet, send_packet)

    async def reject_call(
        self,
        device_id: str,
        send_packet: SendPacketCallable | None = None,
        phone_number: str | None = None,
    ) -> bool:
        """Request the remote peer to reject or decline an incoming call.

        Args:
            device_id: Target remote device identifier.
            send_packet: Optional send packet callable or coroutine.
            phone_number: Optional phone number to include in reject request.

        Returns:
            True on successful transmission.

        Raises:
            ConnectionClosedError: If peer is disconnected or sender is unavailable.
        """
        # Defensive parameter ordering support: allow reject_call(send_packet, device_id)
        if callable(device_id) and isinstance(send_packet, str):
            device_id, send_packet = send_packet, device_id  # type: ignore

        packet = create_telephony_reject_packet(phone_number=phone_number)
        logger.info("Transmitting reject call request to device '%s'", device_id)
        return await self._send_to_device(device_id, packet, send_packet)

    async def get_active_calls(self, device_id: str | None = None) -> list[dict[str, Any]]:
        """Return active calls from TelephonyService, or empty list if unavailable."""
        if self.telephony_service is not None and hasattr(self.telephony_service, "get_active_calls"):
            res = self.telephony_service.get_active_calls(device_id)
            if inspect.isawaitable(res):
                return await res
            return res
        return []

    async def get_call(self, device_id: str) -> dict[str, Any] | None:
        """Return active call dictionary for specific device, or None."""
        if self.telephony_service is not None and hasattr(self.telephony_service, "get_call"):
            res = self.telephony_service.get_call(device_id)
            if inspect.isawaitable(res):
                return await res
            return res
        return None

    async def clear_device(self, device_id: str) -> bool:
        """Prune device from peer cache and clear active calls in service."""
        self._connected_peers.pop(device_id, None)
        if self.telephony_service is not None and hasattr(self.telephony_service, "clear_device"):
            res = self.telephony_service.clear_device(device_id)
            if inspect.isawaitable(res):
                return await res
            return bool(res)
        return False


__all__ = [
    "KNOWN_CALL_EVENTS",
    "TELEPHONY_PACKET_TYPE",
    "TELEPHONY_REQUEST_MUTE_TYPE",
    "TELEPHONY_REQUEST_TYPE",
    "TelephonyPlugin",
    "coerce_boolean",
    "create_telephony_mute_packet",
    "create_telephony_reject_packet",
]

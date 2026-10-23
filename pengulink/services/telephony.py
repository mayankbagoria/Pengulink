"""pengulink.services.telephony
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Core telephony call-state management and event coordination service.

Implements:
- CallState enum: IDLE, RINGING, TALKING, MISSED
- ActiveCall dataclass: device_id, phone_number, contact_name, state, start_time, last_updated
- TelephonyService:
  - Internal registry: _active_calls: Dict[str, ActiveCall] serialized with asyncio.Lock()
  - Multi-device call state isolation
  - process_call_event: processes inbound wire events, handles state transitions,
    and publishes CallEvent and CallStateChangedEvent on EventBus
  - Queries: get_active_calls(), get_call(), active_calls property
  - Lifecycle cleanup: clear_device(), clear_all(), automatic eviction on DeviceDisconnected/DeviceUnpaired
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pengulink.core.events import (
    CallEvent,
    CallStateChangedEvent,
    DeviceDisconnected,
    DeviceUnpaired,
    DeviceUnpairedEvent,
    EventBus,
)
from pengulink.core.exceptions import CallNotFoundError, InvalidCallStateError

logger = logging.getLogger("pengulink.services.telephony")

MAX_STRING_LEN = 1024


class CallState(str, Enum):
    """Enumeration of telephony call states."""

    IDLE = "IDLE"
    RINGING = "RINGING"
    TALKING = "TALKING"
    MISSED = "MISSED"


@dataclass
class ActiveCall:
    """Represents an active or in-progress telephone call on a connected device."""

    device_id: str
    phone_number: str
    contact_name: str
    state: CallState
    start_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    @property
    def duration(self) -> float:
        """Elapsed call duration in seconds."""
        return max(0.0, time.time() - self.start_time)

    @property
    def is_active(self) -> bool:
        """True if the call is currently ringing or talking."""
        return self.state in (CallState.RINGING, CallState.TALKING)

    def to_dict(self) -> dict[str, Any]:
        """Serialize call details for JSON-RPC IPC transmission or CLI formatting."""
        return {
            "device_id": self.device_id,
            "phone_number": self.phone_number,
            "contact_name": self.contact_name,
            "state": self.state.value if isinstance(self.state, Enum) else str(self.state),
            "start_time": self.start_time,
            "last_updated": self.last_updated,
            "duration": self.duration,
            "is_active": self.is_active,
        }


class CallItem(dict):
    """Dictionary supporting both item and attribute access for ActiveCall fields."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'CallItem' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> dict[str, Any]:
        """Return pure dictionary snapshot."""
        return dict(self)


class AwaitableCallList(list):
    """List supporting both synchronous inspection and await expressions."""

    def __await__(self):
        async def _identity():
            return self

        return _identity().__await__()


def coerce_is_cancel(val: Any) -> bool:
    """Safely coerce truthy string, int, or boolean values matching KDE Connect semantics."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val == 1
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return False


class TelephonyService:
    """Core service managing active call states and publishing domain events."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self.event_bus = event_bus
        self._active_calls: dict[str, ActiveCall] = {}
        self._lock = asyncio.Lock()
        self._unsub_callbacks: list[Callable[[], None]] = []

        if self.event_bus:
            self._subscribe_lifecycle_events()

    def _subscribe_lifecycle_events(self) -> None:
        """Subscribe to device disconnect and unpair events on EventBus."""
        if not self.event_bus:
            return
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

    async def _on_device_disconnected(self, event: DeviceDisconnected) -> None:
        """Handle device disconnection by clearing its active call state."""
        device_id = getattr(event, "device_id", "")
        if device_id:
            await self.clear_device(device_id)

    async def _on_device_unpaired(self, event: DeviceUnpaired) -> None:
        """Handle device unpairing by clearing its active call state."""
        device_id = getattr(event, "device_id", "")
        if device_id:
            await self.clear_device(device_id)

    async def process_call_event(
        self,
        device_id: str,
        event: str,
        phone_number: str | None = None,
        contact_name: str | None = None,
        is_cancel: bool | str | Any = False,
        phone_thumbnail: str | None = None,
    ) -> ActiveCall | None:
        """Process an inbound telephony packet event.

        Updates internal call state registry and publishes CallEvent and
        CallStateChangedEvent onto the EventBus.
        """
        if not device_id or not isinstance(device_id, str) or not device_id.strip():
            raise InvalidCallStateError("device_id must be a non-empty string")

        device_id = device_id.strip()
        raw_event = str(event).strip() if event is not None else ""
        event_lower = raw_event.lower()

        # Parse lenient boolean is_cancel
        is_cancel_bool = coerce_is_cancel(is_cancel)

        # Sanitize phone number and contact name fallbacks
        phone_str = str(phone_number).strip()[:MAX_STRING_LEN] if phone_number is not None else ""
        contact_str = str(contact_name)[:MAX_STRING_LEN] if contact_name is not None else ""

        if not contact_str.strip() and phone_str:
            contact_str = phone_str
        elif not contact_str.strip() and not phone_str:
            contact_str = "unknown number"

        if not phone_str:
            phone_str = "unknown number"

        # Silently ignore deprecated SMS events
        if event_lower == "sms":
            logger.debug("Ignored deprecated SMS event from device %s", device_id)
            return None

        call_event: CallEvent | None = None
        state_event: CallStateChangedEvent | None = None
        result_call: ActiveCall | None = None

        # Mutate active call state registry and prepare domain events under lock
        async with self._lock:
            if self.event_bus:
                call_event = CallEvent(
                    device_id=device_id,
                    event_type=raw_event,
                    phone_number=phone_str,
                    contact_name=contact_str,
                    is_cancel=is_cancel_bool,
                    phone_thumbnail=phone_thumbnail,
                )

            # Case A: Call cancelled or ended
            if is_cancel_bool or event_lower in ("idle", "callended", "call_ended", "end", "ended"):
                existing = self._active_calls.pop(device_id, None)
                if existing is not None:
                    old_state = (
                        existing.state.value
                        if isinstance(existing.state, Enum)
                        else str(existing.state)
                    )
                    existing.state = CallState.IDLE
                    existing.last_updated = time.time()
                    p_num = phone_str if phone_str != "unknown number" else existing.phone_number
                    c_name = (
                        contact_str
                        if contact_str != "unknown number"
                        else existing.contact_name
                    )
                    if self.event_bus:
                        state_event = CallStateChangedEvent(
                            device_id=device_id,
                            old_state=old_state,
                            new_state=CallState.IDLE.value,
                            phone_number=p_num,
                            contact_name=c_name,
                        )
                result_call = None

            # Case B: Incoming ringing call
            elif event_lower == "ringing":
                existing = self._active_calls.get(device_id)
                if existing is not None:
                    old_state = (
                        existing.state.value
                        if isinstance(existing.state, Enum)
                        else str(existing.state)
                    )
                    existing.state = CallState.RINGING
                    if phone_str and phone_str != "unknown number":
                        existing.phone_number = phone_str
                    if contact_str and contact_str != "unknown number":
                        existing.contact_name = contact_str
                    existing.last_updated = time.time()
                    call = existing
                else:
                    old_state = CallState.IDLE.value
                    call = ActiveCall(
                        device_id=device_id,
                        phone_number=phone_str,
                        contact_name=contact_str,
                        state=CallState.RINGING,
                        start_time=time.time(),
                        last_updated=time.time(),
                    )
                    self._active_calls[device_id] = call

                if old_state != CallState.RINGING.value and self.event_bus:
                    state_event = CallStateChangedEvent(
                        device_id=device_id,
                        old_state=old_state,
                        new_state=CallState.RINGING.value,
                        phone_number=call.phone_number,
                        contact_name=call.contact_name,
                    )
                result_call = call

            # Case C: Answered / talking call
            elif event_lower == "talking":
                existing = self._active_calls.get(device_id)
                if existing is not None:
                    old_state = (
                        existing.state.value
                        if isinstance(existing.state, Enum)
                        else str(existing.state)
                    )
                    existing.state = CallState.TALKING
                    if phone_str and phone_str != "unknown number":
                        existing.phone_number = phone_str
                    if contact_str and contact_str != "unknown number":
                        existing.contact_name = contact_str
                    existing.last_updated = time.time()
                    call = existing
                else:
                    old_state = CallState.IDLE.value
                    call = ActiveCall(
                        device_id=device_id,
                        phone_number=phone_str,
                        contact_name=contact_str,
                        state=CallState.TALKING,
                        start_time=time.time(),
                        last_updated=time.time(),
                    )
                    self._active_calls[device_id] = call

                if old_state != CallState.TALKING.value and self.event_bus:
                    state_event = CallStateChangedEvent(
                        device_id=device_id,
                        old_state=old_state,
                        new_state=CallState.TALKING.value,
                        phone_number=call.phone_number,
                        contact_name=call.contact_name,
                    )
                result_call = call

            # Case D: Missed call
            elif event_lower in ("missedcall", "missed_call"):
                existing = self._active_calls.pop(device_id, None)
                if existing is not None:
                    old_state = (
                        existing.state.value
                        if isinstance(existing.state, Enum)
                        else str(existing.state)
                    )
                    p_num = (
                        phone_str
                        if phone_str != "unknown number"
                        else existing.phone_number
                    )
                    c_name = (
                        contact_str
                        if contact_str != "unknown number"
                        else existing.contact_name
                    )
                else:
                    old_state = CallState.IDLE.value
                    p_num = phone_str
                    c_name = contact_str

                if self.event_bus:
                    state_event = CallStateChangedEvent(
                        device_id=device_id,
                        old_state=old_state,
                        new_state=CallState.MISSED.value,
                        phone_number=p_num,
                        contact_name=c_name,
                    )
                result_call = None

            # Case E: Unrecognized event
            else:
                logger.warning(
                    "Unrecognized telephony call event '%s' from device %s",
                    raw_event,
                    device_id,
                )
                result_call = None

        # Publish domain events strictly outside the lock to prevent re-entrancy deadlocks
        if self.event_bus:
            if call_event is not None:
                await self.event_bus.publish(call_event)
            if state_event is not None:
                await self.event_bus.publish(state_event)

        return result_call

    def get_active_calls(self, device_id: str | None = None) -> AwaitableCallList:
        """Return list of active call dictionaries, supporting both synchronous and async access."""
        if device_id is not None:
            call = self._active_calls.get(device_id)
            items = [CallItem(call.to_dict())] if call is not None else []
        else:
            items = [CallItem(c.to_dict()) for c in self._active_calls.values()]
        return AwaitableCallList(items)

    def get_active_calls_sync(self, device_id: str | None = None) -> AwaitableCallList:
        """Synchronously return snapshot of active call dictionaries."""
        return self.get_active_calls(device_id)

    async def get_call(self, device_id: str) -> CallItem | None:
        """Retrieve active call dictionary for device_id, or None if not found."""
        async with self._lock:
            call = self._active_calls.get(device_id)
            return CallItem(call.to_dict()) if call is not None else None

    def get_call_sync(self, device_id: str, raise_if_not_found: bool = False) -> CallItem | None:
        """Synchronously retrieve active call dictionary for device_id."""
        call = self._active_calls.get(device_id)
        if call is None:
            if raise_if_not_found:
                raise CallNotFoundError(f"No active call found for device '{device_id}'")
            return None
        return CallItem(call.to_dict())

    async def get_call_object(self, device_id: str, raise_if_not_found: bool = False) -> ActiveCall | None:
        """Retrieve ActiveCall dataclass instance for device_id."""
        async with self._lock:
            call = self._active_calls.get(device_id)
            if call is None and raise_if_not_found:
                raise CallNotFoundError(f"No active call found for device '{device_id}'")
            return call

    @property
    def active_calls(self) -> dict[str, ActiveCall]:
        """Dictionary snapshot of current active call instances."""
        return dict(self._active_calls)

    async def clear_device(self, device_id: str) -> bool:
        """Evict active call for device_id and emit transition to IDLE if call was active.

        Returns:
            True if an active call was cleared, False otherwise.
        """
        state_event: CallStateChangedEvent | None = None
        old_state: str = ""
        async with self._lock:
            call = self._active_calls.pop(device_id, None)
            if call is None:
                return False

            old_state = (
                call.state.value
                if isinstance(call.state, Enum)
                else str(call.state)
            )
            call.state = CallState.IDLE
            call.last_updated = time.time()
            if self.event_bus and old_state != CallState.IDLE.value:
                state_event = CallStateChangedEvent(
                    device_id=device_id,
                    old_state=old_state,
                    new_state=CallState.IDLE.value,
                    phone_number=call.phone_number,
                    contact_name=call.contact_name,
                )

        # Publish outside the lock to prevent re-entrancy deadlocks
        if self.event_bus and state_event is not None:
            await self.event_bus.publish(state_event)

        logger.debug("Cleared active call for device %s (was %s)", device_id, old_state)
        return True

    async def clear_all(self) -> None:
        """Evict all active calls across all devices and emit transition events."""
        events_to_publish: list[CallStateChangedEvent] = []
        async with self._lock:
            for device_id, call in list(self._active_calls.items()):
                old_state = (
                    call.state.value
                    if isinstance(call.state, Enum)
                    else str(call.state)
                )
                call.state = CallState.IDLE
                call.last_updated = time.time()
                if self.event_bus and old_state != CallState.IDLE.value:
                    events_to_publish.append(
                        CallStateChangedEvent(
                            device_id=device_id,
                            old_state=old_state,
                            new_state=CallState.IDLE.value,
                            phone_number=call.phone_number,
                            contact_name=call.contact_name,
                        )
                    )
            num_cleared = len(self._active_calls)
            self._active_calls.clear()

        if self.event_bus:
            for event in events_to_publish:
                await self.event_bus.publish(event)
        logger.debug("Cleared all active calls across %d devices", num_cleared)

    def close(self) -> None:
        """Clean up EventBus subscriptions."""
        for unsub in self._unsub_callbacks:
            try:
                unsub()
            except (AttributeError, RuntimeError, TypeError, ValueError, KeyError) as exc:
                logger.debug("Error unsubscribing event in TelephonyService: %s", exc)
        self._unsub_callbacks.clear()


__all__ = [
    "ActiveCall",
    "AwaitableCallList",
    "CallItem",
    "CallState",
    "TelephonyService",
    "coerce_is_cancel",
]

"""pengulink.core.state_machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Explicit 8-state Device State Machine for KDE Connect nodes.

States:
  DISCONNECTED, DISCOVERED, CONNECTING, PAIRING, PAIRED,
  CONNECTED, DISCONNECTING, ERROR.

Enforces valid transition matrix, prevents session hijacking (CVE-2020-26164),
and records state transition telemetry.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Union

from pengulink.core.exceptions import InvalidStateTransitionError

logger = logging.getLogger("pengulink.core.state_machine")


class DeviceState(str, Enum):
    """Explicit 8-state representation of a KDE Connect device node."""

    DISCONNECTED = "DISCONNECTED"
    DISCOVERED = "DISCOVERED"
    CONNECTING = "CONNECTING"
    PAIRING = "PAIRING"
    PAIRED = "PAIRED"
    CONNECTED = "CONNECTED"
    DISCONNECTING = "DISCONNECTING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class StateTransitionRecord:
    """Audit log entry for a device state machine transition."""

    from_state: DeviceState
    to_state: DeviceState
    timestamp: float
    reason: Optional[str] = None


class DeviceStateMachine:
    """Manages explicit lifecycle transitions for a single KDE Connect device."""

    # Explicit whitelist of legal transitions
    VALID_TRANSITIONS: Dict[DeviceState, Set[DeviceState]] = {
        DeviceState.DISCONNECTED: {
            DeviceState.DISCOVERED,
            DeviceState.CONNECTING,
        },
        DeviceState.DISCOVERED: {
            DeviceState.CONNECTING,
            DeviceState.DISCONNECTED,
        },
        DeviceState.CONNECTING: {
            DeviceState.CONNECTED,
            DeviceState.PAIRING,
            DeviceState.DISCONNECTED,
            DeviceState.ERROR,
        },
        DeviceState.PAIRING: {
            DeviceState.PAIRED,
            DeviceState.DISCONNECTING,
            DeviceState.DISCONNECTED,
        },
        DeviceState.PAIRED: {
            DeviceState.CONNECTED,
            DeviceState.DISCONNECTING,
        },
        DeviceState.CONNECTED: {
            DeviceState.DISCONNECTING,
            DeviceState.DISCONNECTED,
            DeviceState.ERROR,
        },
        DeviceState.DISCONNECTING: {
            DeviceState.DISCONNECTED,
            DeviceState.ERROR,
        },
        DeviceState.ERROR: {
            DeviceState.DISCONNECTED,
            DeviceState.CONNECTING,
        },
    }

    def __init__(
        self,
        device_id: str,
        initial_state: DeviceState = DeviceState.DISCONNECTED,
        on_transition: Optional[
            Callable[[DeviceState, DeviceState, Optional[str]], Union[Coroutine[Any, Any, None], None]]
        ] = None,
    ) -> None:
        self.device_id = device_id
        self._current_state = initial_state
        self._on_transition = on_transition
        self._history: List[StateTransitionRecord] = [
            StateTransitionRecord(
                from_state=initial_state,
                to_state=initial_state,
                timestamp=time.time(),
                reason="Initial state",
            )
        ]
        self._listeners: List[
            Callable[[DeviceState, DeviceState, Optional[str]], Union[Coroutine[Any, Any, None], None]]
        ] = []
        if on_transition:
            self._listeners.append(on_transition)

    @property
    def current_state(self) -> DeviceState:
        return self._current_state

    @property
    def state(self) -> DeviceState:
        return self._current_state

    @property
    def is_paired(self) -> bool:
        """Check whether the device state machine is currently in a paired or connected state."""
        return self._current_state in (DeviceState.PAIRED, DeviceState.CONNECTED)

    @property
    def history(self) -> List[StateTransitionRecord]:
        return list(self._history)

    def can_transition_to(self, new_state: DeviceState) -> bool:
        """Check if transition from current state to new_state is allowed."""
        if new_state == self._current_state:
            return True  # Idempotent state transition
        allowed = self.VALID_TRANSITIONS.get(self._current_state, set())
        return new_state in allowed

    async def transition_to(self, new_state: DeviceState, reason: Optional[str] = None) -> None:
        """Execute a state transition with invariant validation and listener dispatch."""
        if new_state == self._current_state:
            return

        if not self.can_transition_to(new_state):
            logger.error(
                "Illegal state transition for %s: %s -> %s (reason: %s)",
                self.device_id,
                self._current_state.value,
                new_state.value,
                reason,
            )
            raise InvalidStateTransitionError(
                device_id=self.device_id,
                current_state=self._current_state.value,
                attempted_state=new_state.value,
                context={"reason": reason},
            )

        old_state = self._current_state
        self._current_state = new_state
        now = time.time()

        record = StateTransitionRecord(
            from_state=old_state,
            to_state=new_state,
            timestamp=now,
            reason=reason,
        )
        self._history.append(record)

        logger.info(
            "Device %s state changed: %s -> %s (%s)",
            self.device_id,
            old_state.value,
            new_state.value,
            reason or "no reason given",
        )

        # Dispatch listeners
        for listener in list(self._listeners):
            try:
                res = listener(old_state, new_state, reason)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as exc:
                logger.warning(
                    "Error in state transition listener for %s (%s -> %s): %s",
                    self.device_id,
                    old_state.value,
                    new_state.value,
                    exc,
                )

    def add_listener(
        self,
        listener: Callable[[DeviceState, DeviceState, Optional[str]], Union[Coroutine[Any, Any, None], None]],
    ) -> None:
        """Register a callback invoked on every state transition."""
        self._listeners.append(listener)


__all__ = [
    "DeviceState",
    "StateTransitionRecord",
    "DeviceStateMachine",
]

"""Unit tests for pengulink.core.state_machine."""

from __future__ import annotations

import pytest

from pengulink.core.exceptions import InvalidStateTransitionError
from pengulink.core.state_machine import DeviceState, DeviceStateMachine


async def test_initial_state_and_history() -> None:
    fsm = DeviceStateMachine(device_id="dev_fsm_1")
    assert fsm.current_state == DeviceState.DISCONNECTED
    assert len(fsm.history) == 1
    assert fsm.history[0].from_state == DeviceState.DISCONNECTED
    assert fsm.history[0].to_state == DeviceState.DISCONNECTED


async def test_legal_pairing_lifecycle() -> None:
    transitions = []
    fsm = DeviceStateMachine(
        device_id="dev_fsm_2",
        on_transition=lambda old_s, new_s, r: transitions.append((old_s, new_s)),
    )

    # DISCONNECTED -> DISCOVERED -> CONNECTING -> PAIRING -> PAIRED -> CONNECTED
    await fsm.transition_to(DeviceState.DISCOVERED, reason="UDP beacon")
    assert fsm.current_state == DeviceState.DISCOVERED

    await fsm.transition_to(DeviceState.CONNECTING, reason="User pair clicked")
    assert fsm.current_state == DeviceState.CONNECTING

    await fsm.transition_to(DeviceState.PAIRING, reason="TLS TOFU established")
    assert fsm.current_state == DeviceState.PAIRING

    await fsm.transition_to(DeviceState.PAIRED, reason="User confirmed auth code")
    assert fsm.current_state == DeviceState.PAIRED

    await fsm.transition_to(DeviceState.CONNECTED, reason="Cert persisted, plugins active")
    assert fsm.current_state == DeviceState.CONNECTED

    # CONNECTED -> DISCONNECTING -> DISCONNECTED
    await fsm.transition_to(DeviceState.DISCONNECTING, reason="Graceful close")
    assert fsm.current_state == DeviceState.DISCONNECTING

    await fsm.transition_to(DeviceState.DISCONNECTED, reason="Transport closed")
    assert fsm.current_state == DeviceState.DISCONNECTED

    assert len(transitions) == 7


async def test_illegal_transition_raises() -> None:
    fsm = DeviceStateMachine(device_id="dev_fsm_illegal")
    assert fsm.current_state == DeviceState.DISCONNECTED

    # Illegal: DISCONNECTED directly to CONNECTED
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        await fsm.transition_to(DeviceState.CONNECTED)

    assert "Illegal transition for device 'dev_fsm_illegal'" in str(exc_info.value)
    assert fsm.current_state == DeviceState.DISCONNECTED  # State must not mutate on failure


async def test_illegal_pairing_to_connected_direct() -> None:
    fsm = DeviceStateMachine(device_id="dev_fsm_sandbox")
    await fsm.transition_to(DeviceState.DISCOVERED)
    await fsm.transition_to(DeviceState.CONNECTING)
    await fsm.transition_to(DeviceState.PAIRING)

    # Invariant: Must transition through PAIRED (to ensure trust was flushed) before CONNECTED
    with pytest.raises(InvalidStateTransitionError):
        await fsm.transition_to(DeviceState.CONNECTED)


async def test_error_state_and_recovery() -> None:
    fsm = DeviceStateMachine(device_id="dev_fsm_err")
    await fsm.transition_to(DeviceState.CONNECTING)
    await fsm.transition_to(DeviceState.ERROR, reason="TLS cert handshake failed")
    assert fsm.current_state == DeviceState.ERROR

    # From ERROR can recover to DISCONNECTED
    await fsm.transition_to(DeviceState.DISCONNECTED, reason="Cooldown expired")
    assert fsm.current_state == DeviceState.DISCONNECTED

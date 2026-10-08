"""Unit tests for pengulink.core.events."""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from pengulink.core.events import (
    BatteryUpdated,
    DeviceConnected,
    DeviceDiscovered,
    DeviceDisconnected,
    DevicePaired,
    DeviceUnpaired,
    Event,
    EventBus,
    NotificationCleared,
    NotificationReceived,
    PairingRequested,
    PingReceived,
)


async def test_event_bus_subscribe_and_publish() -> None:
    bus = EventBus()
    received: List[Event] = []

    bus.subscribe(PingReceived, lambda e: received.append(e))

    ping = PingReceived(device_id="dev_1", message="Hello!")
    await bus.publish(ping)

    assert len(received) == 1
    assert received[0] is ping


async def test_event_bus_unsubscribe() -> None:
    bus = EventBus()
    received: List[Event] = []

    unsub = bus.subscribe(PingReceived, lambda e: received.append(e))
    await bus.publish(PingReceived(device_id="dev_1", message="First"))
    assert len(received) == 1

    unsub()
    await bus.publish(PingReceived(device_id="dev_1", message="Second"))
    assert len(received) == 1  # No new event received


async def test_event_bus_subscribe_all() -> None:
    bus = EventBus()
    received: List[Event] = []

    bus.subscribe_all(lambda e: received.append(e))

    e1 = PingReceived(device_id="dev_1", message="Ping")
    e2 = BatteryUpdated(device_id="dev_1", current_charge=80, is_charging=True, threshold_event=0)

    await bus.publish(e1)
    await bus.publish(e2)

    assert len(received) == 2
    assert received[0] is e1
    assert received[1] is e2


async def test_event_bus_async_handler() -> None:
    bus = EventBus()
    flag = asyncio.Event()

    async def async_handler(e: PingReceived) -> None:
        await asyncio.sleep(0.01)
        flag.set()

    bus.subscribe(PingReceived, async_handler)
    await bus.publish(PingReceived(device_id="dev_1", message="Async"))

    assert flag.is_set()


async def test_event_bus_fault_isolation() -> None:
    bus = EventBus()
    received: List[Event] = []

    def failing_handler(e: PingReceived) -> None:
        raise RuntimeError("Crash in subscriber")

    def working_handler(e: PingReceived) -> None:
        received.append(e)

    bus.subscribe(PingReceived, failing_handler)
    bus.subscribe(PingReceived, working_handler)

    # Publishing must not raise; working handler must still execute
    await bus.publish(PingReceived(device_id="dev_1", message="Resilient"))
    assert len(received) == 1

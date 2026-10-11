"""tests.unit.test_findmyphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for FindMyPhonePlugin, protocol packets, toggle cancellation,
outbound ring_phone, and event dispatch.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List
import pytest

from pengulink.core.events import (
    EventBus,
    FindDeviceAlertStoppedEvent,
    FindDeviceAlertTriggeredEvent,
)
from pengulink.core.exceptions import ConnectionClosedError
from pengulink.network.packet import Packet
from pengulink.plugins.findmyphone import (
    FINDMYPHONE_LEGACY_TYPE,
    FINDMYPHONE_REQUEST_TYPE,
    FindMyPhonePlugin,
    create_find_my_phone_request_packet,
)
from pengulink.services.alert import AlertService, MockAudioBackend


def test_create_find_my_phone_request_packet() -> None:
    """Verify packet construction with default empty body."""
    pkt = create_find_my_phone_request_packet()
    assert pkt.type == FINDMYPHONE_REQUEST_TYPE
    assert pkt.body == {}
    assert pkt.payloadSize is None

    pkt_custom = create_find_my_phone_request_packet({"custom": "data"})
    assert pkt_custom.body == {"custom": "data"}


def test_findmyphone_plugin_capabilities() -> None:
    """Verify dual capability advertising for KDE Connect compatibility."""
    plugin = FindMyPhonePlugin()
    assert plugin.plugin_id == "findmyphone"
    assert FINDMYPHONE_REQUEST_TYPE in plugin.incoming_capabilities
    assert FINDMYPHONE_LEGACY_TYPE in plugin.incoming_capabilities
    assert FINDMYPHONE_REQUEST_TYPE in plugin.outgoing_capabilities
    assert FINDMYPHONE_LEGACY_TYPE in plugin.outgoing_capabilities


async def test_findmyphone_inbound_starts_alert(tmp_path: Path) -> None:
    """Inbound packet triggers start_alert when idle."""
    backend = MockAudioBackend(playback_duration=0.01)
    bus = EventBus()
    alert_service = AlertService(backend=backend, event_bus=bus, runtime_dir=tmp_path)
    plugin = FindMyPhonePlugin(event_bus=bus, alert_service=alert_service)

    pkt = create_find_my_phone_request_packet()
    await plugin.handle_packet(pkt, "remote_phone_1", lambda p: None)

    assert alert_service.is_active() is True
    assert alert_service.get_status()["device_id"] == "remote_phone_1"
    assert alert_service.get_status()["source"] == "remote"
    await alert_service.stop_alert()


async def test_findmyphone_inbound_toggle_cancellation(tmp_path: Path) -> None:
    """Second inbound packet toggles active alert OFF."""
    backend = MockAudioBackend(playback_duration=0.01)
    bus = EventBus()
    stopped_events: List[Any] = []
    bus.subscribe(FindDeviceAlertStoppedEvent, lambda ev: stopped_events.append(ev))

    alert_service = AlertService(backend=backend, event_bus=bus, runtime_dir=tmp_path)
    plugin = FindMyPhonePlugin(event_bus=bus, alert_service=alert_service)

    pkt = create_find_my_phone_request_packet()

    # 1. First packet rings
    await plugin.handle_packet(pkt, "remote_phone_1", lambda p: None)
    assert alert_service.is_active() is True

    # 2. Second packet cancels
    await plugin.handle_packet(pkt, "remote_phone_1", lambda p: None)
    assert alert_service.is_active() is False
    assert len(stopped_events) >= 1
    assert stopped_events[0].reason == "toggle"


async def test_findmyphone_outbound_ring_phone() -> None:
    """ring_phone dispatches kdeconnect.findmyphone.request packet."""
    dispatched: List[Packet] = []

    async def mock_sender(pkt: Packet) -> None:
        dispatched.append(pkt)

    plugin = FindMyPhonePlugin()
    packet = await plugin.ring_phone("target_device_42", mock_sender)

    assert packet.type == FINDMYPHONE_REQUEST_TYPE
    assert packet.body == {}
    assert len(dispatched) == 1
    assert dispatched[0].id == packet.id


async def test_findmyphone_outbound_ring_phone_with_injected_sender() -> None:
    """ring_phone uses plugin.packet_sender if send_packet arg is omitted."""
    dispatched: List[Packet] = []

    def mock_sender(device_id: str, pkt: Packet) -> None:
        dispatched.append(pkt)

    plugin = FindMyPhonePlugin(packet_sender=mock_sender)
    packet = await plugin.ring_phone("target_device_42")

    assert len(dispatched) == 1
    assert dispatched[0].id == packet.id


async def test_findmyphone_outbound_invalid_sender() -> None:
    """Calling ring_phone with no valid sender raises ConnectionClosedError."""
    plugin = FindMyPhonePlugin()
    with pytest.raises(ConnectionClosedError):
        await plugin.ring_phone("target_device_42")


async def test_findmyphone_outbound_broken_stream() -> None:
    """Broken connection raises ConnectionClosedError."""
    async def broken_sender(pkt: Packet) -> None:
        raise OSError("Broken pipe on TCP stream")

    plugin = FindMyPhonePlugin()
    with pytest.raises(ConnectionClosedError) as exc_info:
        await plugin.ring_phone("target_device_42", broken_sender)
    assert "Broken pipe" in str(exc_info.value)


async def test_findmyphone_fallback_without_alert_service() -> None:
    """Plugin operates in standalone fallback mode when AlertService is omitted."""
    bus = EventBus()
    triggered: List[Any] = []
    stopped: List[Any] = []
    bus.subscribe(FindDeviceAlertTriggeredEvent, lambda ev: triggered.append(ev))
    bus.subscribe(FindDeviceAlertStoppedEvent, lambda ev: stopped.append(ev))

    plugin = FindMyPhonePlugin(event_bus=bus)
    assert plugin._is_alert_active() is False

    # Start alert
    pkt = create_find_my_phone_request_packet()
    await plugin.handle_packet(pkt, "dev_standalone", lambda p: None)
    assert plugin._is_alert_active() is True
    assert len(triggered) == 1
    assert triggered[0].device_id == "dev_standalone"

    # Toggle alert off
    await plugin.handle_packet(pkt, "dev_standalone", lambda p: None)
    assert plugin._is_alert_active() is False
    assert len(stopped) == 1
    assert stopped[0].reason == "toggle"


async def test_findmyphone_malformed_body_tolerance(tmp_path: Path) -> None:
    """Non-empty or unexpected body contents do not cause errors."""
    backend = MockAudioBackend(playback_duration=0.01)
    service = AlertService(backend=backend, runtime_dir=tmp_path)
    plugin = FindMyPhonePlugin(alert_service=service)

    pkt = Packet(type=FINDMYPHONE_REQUEST_TYPE, body={"unexpected_key": 12345, "extra": "data"})
    await plugin.handle_packet(pkt, "dev_test", lambda p: None)
    assert service.is_active() is True
    await service.stop_alert()


async def test_findmyphone_shutdown_cleanup(tmp_path: Path) -> None:
    """plugin.stop() stops any active alert."""
    backend = MockAudioBackend(playback_duration=0.01)
    service = AlertService(backend=backend, runtime_dir=tmp_path)
    plugin = FindMyPhonePlugin(alert_service=service)

    pkt = create_find_my_phone_request_packet()
    await plugin.handle_packet(pkt, "dev_test", lambda p: None)
    assert service.is_active() is True

    await plugin.stop()
    assert service.is_active() is False

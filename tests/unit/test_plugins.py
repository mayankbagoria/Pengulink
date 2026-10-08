"""Unit tests for pengulink.plugins (Ping, Battery, Notification, Registry)."""

from __future__ import annotations

from typing import List
import pytest

from pengulink.core.events import (
    BatteryUpdated,
    Event,
    EventBus,
    NotificationCleared,
    NotificationReceived,
    PingReceived,
)
from pengulink.network.packet import Packet
from pengulink.plugins.base import PluginRegistry
from pengulink.plugins.battery import BatteryPlugin, create_battery_status_packet
from pengulink.plugins.notification import (
    NotificationPlugin,
    extract_otp_code,
    sanitize_notification_id,
)
from pengulink.plugins.ping import PingPlugin, create_ping_packet


async def test_plugin_registry_dispatch() -> None:
    bus = EventBus()
    registry = PluginRegistry(bus)

    ping_plugin = PingPlugin(bus)
    battery_plugin = BatteryPlugin(bus)
    registry.register(ping_plugin)
    registry.register(battery_plugin)

    assert "kdeconnect.ping" in registry.get_all_incoming_capabilities()
    assert "kdeconnect.battery" in registry.get_all_incoming_capabilities()

    sent_packets: List[Packet] = []

    async def mock_send(p: Packet) -> None:
        sent_packets.append(p)

    handled = await registry.dispatch_packet(
        Packet(type="kdeconnect.ping", body={"message": "Hey"}),
        device_id="dev_1",
        send_packet=mock_send,
    )
    assert handled is True

    # Unknown packet type returns False without crashing
    unknown_handled = await registry.dispatch_packet(
        Packet(type="kdeconnect.unknown", body={}),
        device_id="dev_1",
        send_packet=mock_send,
    )
    assert unknown_handled is False


async def test_ping_plugin() -> None:
    bus = EventBus()
    events: List[Event] = []
    bus.subscribe(PingReceived, lambda e: events.append(e))

    plugin = PingPlugin(bus)
    sent_packets: List[Packet] = []

    # Receive ping
    await plugin.handle_packet(
        create_ping_packet(message="Custom Ping"),
        device_id="phone_ping",
        send_packet=lambda p: sent_packets.append(p),
    )

    assert len(events) == 1
    assert isinstance(events[0], PingReceived)
    assert events[0].message == "Custom Ping"
    assert events[0].device_id == "phone_ping"

    # Send ping
    await plugin.send_ping(lambda p: sent_packets.append(p), message="Outbound Ping")
    assert len(sent_packets) == 1
    assert sent_packets[0].body.get("message") == "Outbound Ping"


async def test_battery_plugin() -> None:
    bus = EventBus()
    events: List[Event] = []
    bus.subscribe(BatteryUpdated, lambda e: events.append(e))

    plugin = BatteryPlugin(bus)
    sent_packets: List[Packet] = []

    # Receive battery status
    packet = create_battery_status_packet(
        current_charge=76,
        is_charging=True,
        threshold_event=0,
    )
    await plugin.handle_packet(packet, device_id="phone_bat", send_packet=lambda p: sent_packets.append(p))

    assert len(events) == 1
    assert isinstance(events[0], BatteryUpdated)
    assert events[0].current_charge == 76
    assert events[0].is_charging is True

    # Receive battery request query -> responds with local status
    plugin.local_charge = 99
    plugin.local_is_charging = False
    req_packet = Packet(type="kdeconnect.battery.request", body={"request": True})

    await plugin.handle_packet(req_packet, device_id="phone_bat", send_packet=lambda p: sent_packets.append(p))
    assert len(sent_packets) == 1
    assert sent_packets[0].type == "kdeconnect.battery"
    assert sent_packets[0].body["currentCharge"] == 99


async def test_notification_plugin_inbound_and_otp() -> None:
    bus = EventBus()
    events: List[Event] = []
    bus.subscribe(NotificationReceived, lambda e: events.append(e))
    bus.subscribe(NotificationCleared, lambda e: events.append(e))

    plugin = NotificationPlugin(bus)
    sent_packets: List[Packet] = []

    # Notification with Android prefix and OTP
    notif_packet = Packet(
        type="kdeconnect.notification",
        body={
            "id": "org.kde.kdeconnect_tp::4242",
            "appName": "Messages",
            "title": "Bank Security",
            "text": "Your verification code is 849201. Do not share.",
            "ticker": "Messages: Your verification code is 849201",
            "isClearable": True,
        },
    )

    await plugin.handle_packet(notif_packet, device_id="phone_notif", send_packet=lambda p: sent_packets.append(p))

    assert len(events) == 1
    received = events[0]
    assert isinstance(received, NotificationReceived)
    assert received.notification_id == "4242"
    assert received.auth_code == "849201"
    assert plugin.get_notification("4242") is not None

    # Remote cancellation
    cancel_packet = Packet(
        type="kdeconnect.notification",
        body={
            "id": "org.kde.kdeconnect_tp::4242",
            "isCancel": True,
        },
    )
    await plugin.handle_packet(cancel_packet, device_id="phone_notif", send_packet=lambda p: sent_packets.append(p))

    assert len(events) == 2
    cleared = events[1]
    assert isinstance(cleared, NotificationCleared)
    assert cleared.notification_id == "4242"
    assert plugin.get_notification("4242") is None


async def test_notification_plugin_outbound_helpers() -> None:
    plugin = NotificationPlugin()
    sent_packets: List[Packet] = []

    async def mock_send(p: Packet) -> None:
        sent_packets.append(p)

    # Dismiss
    await plugin.dismiss_notification(mock_send, "notif_99")
    assert sent_packets[0].type == "kdeconnect.notification.request"
    assert sent_packets[0].body["cancel"] == "notif_99"

    # Reply
    await plugin.reply_notification(mock_send, "reply_uuid_123", "Hello there")
    assert sent_packets[1].type == "kdeconnect.notification.reply"
    assert sent_packets[1].body["requestReplyId"] == "reply_uuid_123"
    assert sent_packets[1].body["message"] == "Hello there"

    # Action
    await plugin.trigger_action(mock_send, "notif_99", "Archive")
    assert sent_packets[2].type == "kdeconnect.notification.action"
    assert sent_packets[2].body["key"] == "notif_99"
    assert sent_packets[2].body["action"] == "Archive"


def test_notification_helpers() -> None:
    assert sanitize_notification_id("org.kde.kdeconnect_tp::123") == "123"
    assert sanitize_notification_id("standard_id") == "standard_id"

    # OTP extraction with invisible unicode characters
    dirty_text = "Your OTP\u2063 code is: 582914"
    assert extract_otp_code(dirty_text) == "582914"
    assert extract_otp_code("No digits here") is None

"""tests.e2e.test_plugins_e2e
~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-End integration tests for KDE Connect core plugins:
- PingPlugin: outbound and inbound ping/pong exchange.
- BatteryPlugin: mobile charge updates and remote queries.
- NotificationPlugin: notification dispatch, OTP/2FA extraction, remote dismissal,
  local cancellation, and quick replies.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pengulink.core.events import (
    BatteryUpdated,
    NotificationCleared,
    NotificationReceived,
    PingReceived,
)
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.network.packet import PacketEngine
from tests.e2e.conftest import e2e_environment, wait_for
from tests.mock.mock_device import MockDevice


async def _pair_and_connect(daemon: PenguLinkDaemon, mock: MockDevice) -> None:
    """Helper to establish mutual pairing and an active TLS connection."""
    await mock.broadcast_identity(
        target_ip="127.0.0.1",
        target_port=daemon.config.network.udp_port,
    )
    await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)
    res = await daemon.pair_device(mock.device_id, wait=True, timeout=5.0)
    assert res["paired"] is True
    assert daemon.is_device_connected(mock.device_id) is True


async def test_plugin_ping_roundtrip_daemon_to_mock(tmp_path: Path) -> None:
    """Verify outbound ping from daemon to MockDevice with automatic pong response.

    1. Daemon sends kdeconnect.ping packet with custom message.
    2. MockDevice receives ping and records message.
    3. MockDevice auto_pong responds with kdeconnect.ping message 'Pong from MockDevice'.
    4. Daemon receives pong and emits PingReceived on EventBus.
    """
    ping_events: list[PingReceived] = []

    async with e2e_environment(tmp_path, auto_pong=True) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        daemon.event_bus.subscribe(
            PingReceived,
            lambda ev: ping_events.append(ev),
        )

        # Transmit ping from daemon
        await daemon.send_ping(mock.device_id, message="Ping from PenguLink Daemon")

        # Verify MockDevice received ping
        await wait_for(
            lambda: mock.last_ping_message == "Ping from PenguLink Daemon",
            timeout=3.0,
            desc="mock receiving ping",
        )

        # Verify daemon received pong response
        await wait_for(
            lambda: any(e.message == "Pong from MockDevice" for e in ping_events),
            timeout=3.0,
            desc="daemon receiving pong",
        )
        assert any(e.device_id == mock.device_id for e in ping_events)


async def test_plugin_ping_roundtrip_mock_to_daemon(tmp_path: Path) -> None:
    """Verify inbound ping from MockDevice to daemon emits PingReceived on EventBus."""
    ping_events: list[PingReceived] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        daemon.event_bus.subscribe(
            PingReceived,
            lambda ev: ping_events.append(ev),
        )

        await mock.send_ping(message="Ping from Android Phone")

        await wait_for(
            lambda: any(e.message == "Ping from Android Phone" for e in ping_events),
            timeout=3.0,
            desc="daemon receiving inbound ping",
        )
        assert any(e.device_id == mock.device_id for e in ping_events)


async def test_plugin_battery_update_from_mock(tmp_path: Path) -> None:
    """Verify mobile battery status update dispatched to EventBus and cached in daemon."""
    battery_events: list[BatteryUpdated] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        daemon.event_bus.subscribe(
            BatteryUpdated,
            lambda ev: battery_events.append(ev),
        )

        # Mock transmits battery update
        await mock.send_battery_update(current_charge=88, is_charging=True)

        await wait_for(
            lambda: any(e.current_charge == 88 and e.is_charging for e in battery_events),
            timeout=3.0,
            desc="daemon receiving battery update",
        )

        # Verify cache in BatteryPlugin
        cached = daemon.battery_plugin.get_device_battery(mock.device_id)
        assert cached is not None
        assert cached.current_charge == 88
        assert cached.is_charging is True

        # Verify reported via IPC
        devices = await ipc_client.get_devices()
        matched = [d for d in devices if d["device_id"] == mock.device_id]
        assert len(matched) == 1
        bat = matched[0]["battery"]
        assert bat is not None
        assert bat["current_charge"] == 88
        assert bat["is_charging"] is True


async def test_plugin_battery_query_daemon_to_mock(tmp_path: Path) -> None:
    """Verify daemon queries remote battery status and updates local cache."""
    battery_events: list[BatteryUpdated] = []

    async with e2e_environment(tmp_path, auto_battery_reply=True) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        daemon.event_bus.subscribe(
            BatteryUpdated,
            lambda ev: battery_events.append(ev),
        )

        # Configure mock's current battery level
        mock.current_battery_charge = 65
        mock.is_battery_charging = False

        # Query battery from daemon
        conn = daemon._active_connections[mock.device_id]
        await daemon.battery_plugin.request_battery_status(
            lambda pkt: conn.write_line(PacketEngine.serialize(pkt))
        )

        # Wait for reply to arrive and be processed
        await wait_for(
            lambda: any(e.current_charge == 65 and not e.is_charging for e in battery_events),
            timeout=3.0,
            desc="daemon receiving battery reply",
        )

        cached = daemon.battery_plugin.get_device_battery(mock.device_id)
        assert cached is not None
        assert cached.current_charge == 65
        assert cached.is_charging is False


async def test_plugin_notification_dispatch_and_otp(tmp_path: Path) -> None:
    """Verify notification parsing, Android prefix sanitization, and 2FA OTP code extraction."""
    notif_events: list[NotificationReceived] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        daemon.event_bus.subscribe(
            NotificationReceived,
            lambda ev: notif_events.append(ev),
        )

        # Mock transmits notification with OTP
        await mock.send_notification(
            notification_id="security_alert_007",
            app_name="SMS Messages",
            title="Two-Factor Code",
            text="Your verification code is 948201 for logging in.",
        )

        await wait_for(
            lambda: len(notif_events) > 0,
            timeout=3.0,
            desc="notification received event",
        )

        ev = notif_events[0]
        assert ev.notification_id == "security_alert_007"
        assert ev.app_name == "SMS Messages"
        assert ev.auth_code == "948201"
        assert ev.title == "Two-Factor Code"

        # Verify accessible via IPC
        ipc_notifs = await ipc_client.get_notifications(mock.device_id)
        assert len(ipc_notifs) >= 1
        matched = [n for n in ipc_notifs if n["notification_id"] == "security_alert_007"]
        assert len(matched) == 1
        assert matched[0]["auth_code"] == "948201"


async def test_plugin_notification_dismissal_remote(tmp_path: Path) -> None:
    """Verify remote notification dismissal via IPC transmits cancel request to peer."""
    cleared_events: list[NotificationCleared] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        daemon.event_bus.subscribe(
            NotificationCleared,
            lambda ev: cleared_events.append(ev),
        )

        # 1. Post notification
        await mock.send_notification(
            notification_id="clear_me_123",
            app_name="Email",
            title="Meeting Request",
            text="Team standup in 10 minutes",
        )
        await wait_for(
            lambda: daemon.notification_plugin.get_notification("clear_me_123") is not None,
            timeout=3.0,
        )

        # 2. Dismiss notification via IPC
        res = await ipc_client.dismiss_notification(mock.device_id, "clear_me_123")
        assert res["success"] is True

        # 3. Verify cleared locally and event emitted
        assert daemon.notification_plugin.get_notification("clear_me_123") is None
        await wait_for(lambda: len(cleared_events) > 0, timeout=2.0)
        assert any(e.notification_id == "clear_me_123" for e in cleared_events)

        # 4. Verify cancel request packet sent to mock
        await wait_for(
            lambda: any(
                p.body.get("cancel") == "clear_me_123"
                for p in mock.get_received_packets("kdeconnect.notification.request")
            ),
            timeout=3.0,
            desc="mock receiving notification cancellation packet",
        )


async def test_plugin_notification_cancel_from_mock(tmp_path: Path) -> None:
    """Verify phone-initiated notification cancellation removes item from daemon cache."""
    cleared_events: list[NotificationCleared] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        daemon.event_bus.subscribe(
            NotificationCleared,
            lambda ev: cleared_events.append(ev),
        )

        # 1. Post notification
        await mock.send_notification(
            notification_id="cancel_from_phone",
            app_name="Messenger",
            title="New Message",
            text="Hey there!",
        )
        await wait_for(
            lambda: daemon.notification_plugin.get_notification("cancel_from_phone") is not None,
            timeout=3.0,
        )

        # 2. Cancel from mock
        await mock.cancel_notification("cancel_from_phone")

        # 3. Verify removed from daemon cache and event emitted
        await wait_for(
            lambda: daemon.notification_plugin.get_notification("cancel_from_phone") is None,
            timeout=3.0,
        )
        assert any(e.notification_id == "cancel_from_phone" for e in cleared_events)


async def test_plugin_notification_reply(tmp_path: Path) -> None:
    """Verify quick reply transmitted via IPC delivers kdeconnect.notification.reply to peer."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Mock sends notification with requestReplyId
        await mock.send_notification(
            notification_id="chat_msg_42",
            app_name="ChatApp",
            title="Alice",
            text="Are you free for lunch?",
            request_reply_id="reply_channel_42",
        )
        await wait_for(
            lambda: daemon.notification_plugin.get_notification("chat_msg_42") is not None,
            timeout=3.0,
        )

        # Transmit reply via IPC
        reply_res = await ipc_client.reply_notification(
            device_id=mock.device_id,
            reply_id="reply_channel_42",
            message="Yes, let's meet at 12:30!",
        )
        assert reply_res["success"] is True

        # Verify mock received the reply packet
        await wait_for(
            lambda: len(mock.get_received_packets("kdeconnect.notification.reply")) > 0,
            timeout=3.0,
            desc="mock receiving reply packet",
        )
        pkts = mock.get_received_packets("kdeconnect.notification.reply")
        last_pkt = pkts[-1]
        assert last_pkt.body.get("requestReplyId") == "reply_channel_42"
        assert last_pkt.body.get("message") == "Yes, let's meet at 12:30!"

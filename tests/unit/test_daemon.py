"""tests.unit.test_daemon
~~~~~~~~~~~~~~~~~~~~~~
Unit tests for PenguLinkDaemon runtime service, lifecycle, and IPC handlers.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict

import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.core.state_machine import DeviceState
from pengulink.daemon.ipc import PenguLinkIPCClient
from pengulink.daemon.service import LogRingBuffer, PenguLinkDaemon, format_uptime
from pengulink.network.packet import Packet
from pengulink.network.udp import DiscoveredDevice


@pytest.fixture
def test_daemon_config(tmp_path: Path) -> PenguLinkConfig:
    """Construct isolated daemon configuration in tmp_path."""
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / "config"
    cfg.storage.data_dir = tmp_path / "data"
    cfg.storage.runtime_dir = tmp_path / "run"
    cfg.device.device_id = "testdaemon1234567890abcdef123456"
    cfg.device.device_name = "PenguLink-Daemon-Test"
    cfg.network.udp_port = 17168
    cfg.network.tcp_port_min = 17200
    cfg.network.tcp_port_max = 17205
    cfg.ensure_directories()
    return cfg


def test_format_uptime() -> None:
    """Verify uptime duration formatter."""
    assert format_uptime(45) == "45s"
    assert format_uptime(125) == "2m 5s"
    assert format_uptime(3665) == "1h 1m 5s"
    assert format_uptime(90065) == "1d 1h 1m"


def test_log_ring_buffer_emission_and_filtering() -> None:
    """Verify LogRingBuffer captures and filters records."""
    buf = LogRingBuffer(capacity=5)
    test_logger = logging.getLogger("test_buf_logger")
    test_logger.addHandler(buf)
    test_logger.setLevel(logging.DEBUG)

    try:
        test_logger.debug("Debug msg 1")
        test_logger.info("Info msg 2")
        test_logger.warning("Warn msg 3")
        test_logger.error("Error msg 4")

        # Fetch all
        logs = buf.get_logs(limit=10)
        assert len(logs) == 4
        assert logs[0]["message"] == "Debug msg 1"
        assert logs[3]["level"] == "ERROR"

        # Filter by min_level WARNING
        warn_logs = buf.get_logs(limit=10, min_level="WARNING")
        assert len(warn_logs) == 2
        assert warn_logs[0]["level"] == "WARNING"
        assert warn_logs[1]["level"] == "ERROR"

        # Test capacity bounding
        test_logger.info("Msg 5")
        test_logger.info("Msg 6")
        test_logger.info("Msg 7")
        assert len(buf.get_logs(limit=10)) == 5
    finally:
        test_logger.removeHandler(buf)


async def test_daemon_initialization_and_credentials(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify daemon initializes storage, plugins, and 0600 cryptographic credentials."""
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)

    assert daemon.config.device_id == test_daemon_config.device_id
    assert daemon.private_key is not None
    assert daemon.certificate is not None
    assert daemon.local_key_pem.startswith(b"-----BEGIN PRIVATE KEY-----")
    assert daemon.local_cert_pem.startswith(b"-----BEGIN CERTIFICATE-----")

    # Verify POSIX 0600 key file permissions
    key_file = test_daemon_config.storage.config_dir / "private_key.pem"
    assert key_file.exists()
    st_mode = key_file.stat().st_mode
    assert (st_mode & 0o077) == 0  # No group or world permissions


async def test_daemon_start_and_stop_lifecycle(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify daemon starts and stops all network and IPC subsystems cleanly."""
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)

    assert not daemon.is_running
    assert not sock_path.exists()

    await daemon.start()
    try:
        assert daemon.is_running
        assert sock_path.exists()
        assert daemon.tcp_manager.is_running
        assert daemon.udp_discovery.is_running
        assert daemon.ipc_server.is_running
    finally:
        await daemon.stop()
        assert not daemon.is_running
        assert not sock_path.exists()
        assert not daemon.tcp_manager.is_running
        assert not daemon.udp_discovery.is_running
        assert not daemon.ipc_server.is_running


async def test_daemon_ipc_status_query(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify IPC 'status' method returns valid daemon metadata."""
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)

    await daemon.start()
    try:
        client = PenguLinkIPCClient(socket_path=sock_path)
        status_info = await client.status()

        assert status_info["running"] is True
        assert status_info["version"] == "0.1.0"
        assert status_info["device_id"] == test_daemon_config.device_id
        assert status_info["device_name"] == test_daemon_config.device_name
        assert status_info["connected_devices"] == 0
        assert status_info["known_devices"] == 0
        assert status_info["socket_path"] == str(sock_path)
    finally:
        await daemon.stop()


async def test_daemon_ipc_discover_and_get_devices(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify IPC 'discover' and 'get_devices' methods."""
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)

    await daemon.start()
    try:
        client = PenguLinkIPCClient(socket_path=sock_path)

        # 1. Empty discover initially
        discovered = await client.discover(timeout=0.0, refresh=False)
        assert isinstance(discovered, list)
        assert len(discovered) == 0

        # Inject simulated discovered device
        mock_discovered = DiscoveredDevice(
            device_id="peer_android_999",
            device_name="Pixel 8 Pro",
            device_type="phone",
            protocol_version=7,
            tcp_port=1716,
            sender_ip="192.168.1.88",
            incoming_capabilities=("kdeconnect.ping", "kdeconnect.battery"),
            outgoing_capabilities=("kdeconnect.ping", "kdeconnect.battery"),
            timestamp=1700000000.0,
        )
        daemon._discovered_devices[mock_discovered.device_id] = mock_discovered

        # Query discover again
        discovered2 = await client.discover(timeout=0.0, refresh=False)
        assert len(discovered2) == 1
        dev = discovered2[0]
        assert dev["device_id"] == "peer_android_999"
        assert dev["device_name"] == "Pixel 8 Pro"
        assert dev["state"] == "DISCOVERED"

        # 2. Query get_devices
        devices = await client.get_devices()
        assert len(devices) == 1
        assert devices[0]["device_id"] == "peer_android_999"

        # 3. Simulate battery status in plugin
        battery_pkt = Packet(
            type="kdeconnect.battery",
            body={"currentCharge": 76, "isCharging": True, "thresholdEvent": 0},
        )
        await daemon.battery_plugin.handle_packet(battery_pkt, "peer_android_999", lambda p: None)

        # Query devices again with battery
        devices_with_bat = await client.get_devices()
        assert devices_with_bat[0]["battery"]["current_charge"] == 76
        assert devices_with_bat[0]["battery"]["is_charging"] is True

    finally:
        await daemon.stop()


async def test_daemon_ipc_unpair_device(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify IPC 'unpair_device' purges credentials from storage."""
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)

    await daemon.start()
    try:
        client = PenguLinkIPCClient(socket_path=sock_path)

        # Pre-seed paired device
        await daemon.device_store.save_device(
            device_id="paired_peer_456",
            name="My Phone",
            device_type="phone",
            certificate_pem="DUMMY_PEM",
            is_paired=True,
        )
        assert await daemon.device_store.is_device_paired("paired_peer_456")

        # Call unpair
        res = await client.unpair_device("paired_peer_456")
        assert res["success"] is True
        assert not (await daemon.device_store.is_device_paired("paired_peer_456"))

    finally:
        await daemon.stop()


async def test_daemon_ipc_notifications(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify IPC 'get_notifications' and 'dismiss_notification'."""
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)

    await daemon.start()
    try:
        client = PenguLinkIPCClient(socket_path=sock_path)

        # Inject notification into plugin
        notif_pkt = Packet(
            type="kdeconnect.notification",
            body={
                "id": "org.kde.kdeconnect_tp::bank_alert",
                "appName": "BankApp",
                "title": "Security Code",
                "text": "Your login verification code is 849201.",
            },
        )
        await daemon.notification_plugin.handle_packet(notif_pkt, "test_device_1", lambda p: None)

        # Query notifications
        notifs = await client.get_notifications()
        assert len(notifs) == 1
        assert notifs[0]["notification_id"] == "bank_alert"
        assert notifs[0]["app_name"] == "BankApp"
        assert notifs[0]["auth_code"] == "849201"

    finally:
        await daemon.stop()


async def test_daemon_ipc_get_logs(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify IPC 'get_logs' retrieves records from LogRingBuffer."""
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)

    await daemon.start()
    try:
        client = PenguLinkIPCClient(socket_path=sock_path)

        # Emit log message via pengulink logger
        p_logger = logging.getLogger("pengulink.daemon.test")
        p_logger.info("Custom log entry for IPC test")

        logs = await client.get_logs(limit=50)
        assert any("Custom log entry for IPC test" in entry["message"] for entry in logs)

    finally:
        await daemon.stop()


async def test_daemon_ipc_stop_daemon(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify IPC 'stop_daemon' schedules graceful shutdown."""
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)

    await daemon.start()
    try:
        client = PenguLinkIPCClient(socket_path=sock_path)
        res = await client.stop_daemon()
        assert res["success"] is True

        # Wait for daemon to stop
        await asyncio.wait_for(daemon.wait_until_stopped(), timeout=2.0)
        assert not daemon.is_running
    finally:
        await daemon.stop()


async def test_daemon_safe_state_transitions(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify multi-step state transitions execute without InvalidStateTransitionError."""
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)

    dev_id = "test_fsm_device"
    sm = daemon._get_or_create_state_machine(dev_id)
    assert sm.current_state == DeviceState.DISCONNECTED

    # DISCONNECTED -> CONNECTED traversal
    await daemon._transition_device(dev_id, DeviceState.CONNECTED)
    assert sm.current_state == DeviceState.CONNECTED

    # CONNECTED -> DISCONNECTED
    await daemon._transition_device(dev_id, DeviceState.DISCONNECTED)
    assert sm.current_state == DeviceState.DISCONNECTED

    # DISCONNECTED -> PAIRING traversal
    await daemon._transition_device(dev_id, DeviceState.PAIRING)
    assert sm.current_state == DeviceState.PAIRING

    # PAIRING -> CONNECTED traversal
    await daemon._transition_device(dev_id, DeviceState.CONNECTED)
    assert sm.current_state == DeviceState.CONNECTED


async def test_daemon_outbound_tofu_connection(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify outbound TOFU connection to unpaired mock device succeeds without CertificateVerificationError."""
    from tests.mock.mock_device import MockDeviceConfig, MockKdeConnectDevice

    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)
    await daemon.start()

    mock = MockKdeConnectDevice(config=MockDeviceConfig(device_id="mock_phone_123456"))
    await mock.start(listen_tcp=True)

    try:
        conn = await daemon.connect_device("mock_phone_123456", host="127.0.0.1", port=mock.bound_tcp_port)
        assert conn is not None
        assert daemon.is_device_connected("mock_phone_123456")
    finally:
        await mock.stop()
        await daemon.stop()


async def test_daemon_pair_device_outbound(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify outbound TOFU pairing completes cleanly and persists device in DeviceStore."""
    from tests.mock.mock_device import MockDeviceConfig, MockKdeConnectDevice

    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)
    await daemon.start()

    mock = MockKdeConnectDevice(config=MockDeviceConfig(device_id="mock_phone_123456", auto_pair=True))
    await mock.start(listen_tcp=True)

    try:
        daemon._discovered_devices[mock.device_id] = DiscoveredDevice(
            device_id=mock.device_id,
            device_name=mock.config.device_name,
            device_type=mock.config.device_type,
            protocol_version=7,
            tcp_port=mock.bound_tcp_port,
            sender_ip="127.0.0.1",
        )
        res = await daemon.pair_device(mock.device_id, wait=True, timeout=5.0)
        assert res["paired"] is True
        assert res["status"] == "paired"
        assert await daemon.device_store.is_device_paired(mock.device_id) is True
    finally:
        await mock.stop()
        await daemon.stop()


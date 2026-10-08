"""tests.e2e.test_discovery
~~~~~~~~~~~~~~~~~~~~~~~~
End-to-End integration tests for UDP discovery between PenguLinkDaemon
and MockKdeConnectDevice across loopback (127.0.0.1).
"""

from __future__ import annotations

import asyncio
import socket
import uuid
from pathlib import Path

import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.core.events import DeviceDiscovered
from pengulink.core.state_machine import DeviceState
from pengulink.network.packet import Packet
from tests.e2e.conftest import e2e_environment, get_free_ports, wait_for
from tests.mock.mock_device import MockDevice, MockDeviceConfig


async def test_udp_discovery_mock_to_daemon(tmp_path: Path) -> None:
    """Verify end-to-end UDP discovery of MockDevice by PenguLinkDaemon.

    1. Mock broadcasts identity datagram targeting daemon's UDP port.
    2. Daemon receives packet, emits DeviceDiscovered on EventBus, updates
       internal table and state machine to DISCOVERED.
    3. IPC client discover() reports the discovered device with state 'DISCOVERED'.
    """
    discovered_events: list[DeviceDiscovered] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        daemon.event_bus.subscribe(
            DeviceDiscovered,
            lambda ev: discovered_events.append(ev),
        )

        # Mock broadcasts identity to daemon's UDP listener
        await mock.broadcast_identity(
            target_ip="127.0.0.1",
            target_port=daemon.config.network.udp_port,
        )

        # Wait for device to be discovered
        await wait_for(
            lambda: mock.device_id in daemon._discovered_devices,
            timeout=5.0,
            desc=f"discovery of {mock.device_id}",
        )

        dev_record = daemon._discovered_devices[mock.device_id]
        assert dev_record.device_id == mock.device_id
        assert dev_record.device_name == mock.config.device_name
        assert dev_record.device_type == mock.config.device_type
        assert dev_record.tcp_port == mock.bound_tcp_port
        assert dev_record.protocol_version == mock.config.protocol_version
        assert dev_record.sender_ip == "127.0.0.1"

        # Verify EventBus emission
        assert len(discovered_events) >= 1
        assert any(e.device_id == mock.device_id for e in discovered_events)

        # Verify state machine transition
        sm = daemon._state_machines.get(mock.device_id)
        assert sm is not None
        assert sm.current_state == DeviceState.DISCOVERED

        # Verify IPC query
        ipc_devices = await ipc_client.discover(timeout=0.0, refresh=False)
        matched = [d for d in ipc_devices if d["device_id"] == mock.device_id]
        assert len(matched) == 1
        assert matched[0]["device_name"] == mock.config.device_name
        assert matched[0]["state"] == "DISCOVERED"
        assert matched[0]["tcp_port"] == mock.bound_tcp_port


async def test_udp_discovery_daemon_to_loopback(tmp_path: Path) -> None:
    """Verify daemon identity broadcast payload and formatting.

    1. Standalone raw UDP socket binds on loopback.
    2. Daemon broadcast_now(target_ip="127.0.0.1") transmits identity datagram.
    3. Assert datagram is valid JSON with standard KDE Connect fields and bound TCP port.
    """
    target_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target_sock.bind(("127.0.0.1", 0))
    target_port = target_sock.getsockname()[1]
    target_sock.setblocking(False)

    tcp_base = get_free_ports(2)[0]
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / "config"
    cfg.storage.data_dir = tmp_path / "data"
    cfg.storage.runtime_dir = tmp_path / "run"
    cfg.device.device_id = f"daemon_bc_{uuid.uuid4().hex[:12]}"
    cfg.device.device_name = "PenguLink-Broadcaster"
    cfg.network.tcp_port_min = tcp_base
    cfg.network.tcp_port_max = tcp_base + 5
    cfg.network.udp_port = target_port

    async with e2e_environment(tmp_path, daemon_config=cfg) as (daemon, mock, ipc_client):
        try:
            # Trigger targeted broadcast to our listening socket
            await daemon.udp_discovery.broadcast_now(target_ip="127.0.0.1")

            loop = asyncio.get_running_loop()
            data = await asyncio.wait_for(loop.sock_recv(target_sock, 4096), timeout=3.0)

            pkt = Packet.from_wire(data)
            assert pkt.type == "kdeconnect.identity"
            body = pkt.body
            assert body["deviceId"] == daemon.config.device_id
            assert body["deviceName"] == daemon.config.device_name
            assert body["deviceType"] == daemon.config.device_type
            assert body["tcpPort"] == daemon.tcp_manager.bound_port
            assert body["protocolVersion"] == 7
            assert isinstance(body.get("incomingCapabilities"), list)
            assert isinstance(body.get("outgoingCapabilities"), list)
        finally:
            target_sock.close()


async def test_udp_discovery_debounce_and_reflection(tmp_path: Path) -> None:
    """Verify UDP self-reflection filtering and sliding debounce window."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        # 1. Local self-reflection packet
        self_packet = Packet(
            type="kdeconnect.identity",
            body={
                "deviceId": daemon.config.device_id,
                "deviceName": daemon.config.device_name,
                "deviceType": daemon.config.device_type,
                "protocolVersion": 7,
                "tcpPort": daemon.tcp_manager.bound_port,
            },
        )
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.sendto(
                self_packet.to_wire(),
                ("127.0.0.1", daemon.config.network.udp_port),
            )
        finally:
            s.close()

        await asyncio.sleep(0.05)
        # Daemon must ignore its own identity packet
        assert daemon.config.device_id not in daemon._discovered_devices

        # 2. Debounce burst: send 5 rapid broadcasts from mock
        debounce_events: list[DeviceDiscovered] = []
        daemon.event_bus.subscribe(
            DeviceDiscovered,
            lambda ev: debounce_events.append(ev),
        )

        for _ in range(5):
            await mock.broadcast_identity(
                target_ip="127.0.0.1",
                target_port=daemon.config.network.udp_port,
            )
            await asyncio.sleep(0.01)

        await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=3.0)
        # Even with 5 rapid broadcasts, sliding debounce window ensures only 1 event is published
        assert len(debounce_events) == 1
        # Device state machine remains stable in DISCOVERED
        sm = daemon._state_machines.get(mock.device_id)
        assert sm is not None
        assert sm.current_state == DeviceState.DISCOVERED


async def test_udp_discovery_multiple_mock_devices(tmp_path: Path) -> None:
    """Verify concurrent discovery and tracking of multiple distinct MockDevices."""
    async with e2e_environment(tmp_path) as (daemon, mock_a, ipc_client):
        # Instantiate second independent mock device
        m_cfg_b = MockDeviceConfig(
            device_id=f"mock_beta_{uuid.uuid4().hex[:12]}",
            device_name="Mock Android Tablet",
            device_type="tablet",
            auto_pair=True,
        )
        mock_b = MockDevice(config=m_cfg_b)
        await mock_b.start(listen_tcp=True, port=0)

        try:
            # Both broadcast to daemon
            await mock_a.broadcast_identity(
                target_ip="127.0.0.1",
                target_port=daemon.config.network.udp_port,
            )
            await mock_b.broadcast_identity(
                target_ip="127.0.0.1",
                target_port=daemon.config.network.udp_port,
            )

            await wait_for(
                lambda: mock_a.device_id in daemon._discovered_devices
                and mock_b.device_id in daemon._discovered_devices,
                timeout=5.0,
                desc="discovery of both mock devices",
            )

            # Check via IPC
            devs = await ipc_client.get_devices()
            dev_ids = {d["device_id"] for d in devs}
            assert mock_a.device_id in dev_ids
            assert mock_b.device_id in dev_ids
        finally:
            await mock_b.stop()


async def test_udp_discovery_malformed_datagram_resilience(tmp_path: Path) -> None:
    """Verify daemon does not crash when receiving corrupt or malformed UDP datagrams."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        corrupt_payloads = [
            b"NON_JSON_CORRUPTED_BYTES\n",
            b"\x00\xff\xfe\xfd\x01\x02\x03\n",
            b"{}\n",
            b'{"type": "kdeconnect.unknown_type", "body": {}}\n',
            b'{"type": "kdeconnect.identity"}\n',
            b'{"type": "kdeconnect.identity", "body": "invalid_string_not_dict"}\n',
            b'{"type": "kdeconnect.identity", "body": {"deviceId": ""}}\n',
        ]

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for payload in corrupt_payloads:
                s.sendto(payload, ("127.0.0.1", daemon.config.network.udp_port))
        finally:
            s.close()

        await asyncio.sleep(0.05)
        # Daemon survives corruption burst unharmed
        assert daemon.is_running is True

        # Subsequent legitimate broadcast succeeds normally
        await mock.broadcast_identity(
            target_ip="127.0.0.1",
            target_port=daemon.config.network.udp_port,
        )
        await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=3.0)
        assert mock.device_id in daemon._discovered_devices


async def test_udp_discovery_port_collision_fallback(tmp_path: Path) -> None:
    """Verify UDPDiscovery gracefully falls back to an ephemeral port when configured port is busy."""
    occupier_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    occupier_sock.bind(("127.0.0.1", 0))
    busy_port = occupier_sock.getsockname()[1]

    tcp_base = get_free_ports(2)[0]
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / "config"
    cfg.storage.data_dir = tmp_path / "data"
    cfg.storage.runtime_dir = tmp_path / "run"
    cfg.device.device_id = f"daemon_col_{uuid.uuid4().hex[:12]}"
    cfg.device.device_name = "PenguLink-Collision-Test"
    cfg.network.tcp_port_min = tcp_base
    cfg.network.tcp_port_max = tcp_base + 5
    cfg.network.udp_port = busy_port

    try:
        async with e2e_environment(tmp_path, daemon_config=cfg) as (daemon, mock, ipc_client):
            assert daemon.is_running is True
            assert daemon.udp_discovery.is_running is True
    finally:
        occupier_sock.close()

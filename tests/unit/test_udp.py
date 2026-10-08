"""Unit tests for PenguLink UDP Discovery."""

import asyncio
import json
import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.network.packet import Packet
from pengulink.network.udp import (
    DiscoveredDevice,
    UDPDiscovery,
    get_broadcast_addresses,
)


def test_get_broadcast_addresses() -> None:
    """Assert get_broadcast_addresses returns valid broadcast IP set."""
    broadcasts = get_broadcast_addresses()
    assert isinstance(broadcasts, set)
    assert "255.255.255.255" in broadcasts
    for ip in broadcasts:
        octets = ip.split(".")
        assert len(octets) == 4
        for o in octets:
            assert 0 <= int(o) <= 255


def test_build_identity_packet(sample_config: PenguLinkConfig) -> None:
    """Assert build_identity_packet constructs compliant wire identity."""
    discovery = UDPDiscovery(
        config=sample_config,
        bound_tcp_port_getter=lambda: 1718,
    )
    wire = discovery.build_identity_packet()
    assert wire.endswith(b"\n")

    pkt = Packet.from_wire(wire)
    assert pkt.type == "kdeconnect.identity"
    ident = pkt.parse_identity()
    assert ident.deviceId == sample_config.device.device_id
    assert ident.deviceName == sample_config.device.device_name
    assert ident.tcpPort == 1718
    assert ident.protocolVersion == 7


def test_local_reflection_deduplication(sample_config: PenguLinkConfig) -> None:
    """Assert incoming self-broadcast reflection is dropped silently."""
    discovered_list = []
    discovery = UDPDiscovery(
        config=sample_config,
        bound_tcp_port_getter=lambda: 1716,
        on_device_discovered=lambda dev: discovered_list.append(dev),
    )

    # Simulate datagram from local device ID
    self_packet = {
        "id": 1,
        "type": "kdeconnect.identity",
        "body": {
            "deviceId": sample_config.device.device_id,
            "deviceName": sample_config.device.device_name,
            "deviceType": "desktop",
            "protocolVersion": 7,
            "tcpPort": 1716,
        },
    }
    payload = json.dumps(self_packet).encode("utf-8") + b"\n"
    discovery._handle_datagram(payload, ("127.0.0.1", 1716))

    # Assert callback was NOT triggered
    assert len(discovered_list) == 0


def test_remote_device_discovery(sample_config: PenguLinkConfig) -> None:
    """Assert incoming remote identity datagram triggers on_device_discovered."""
    discovered_list = []
    discovery = UDPDiscovery(
        config=sample_config,
        bound_tcp_port_getter=lambda: 1716,
        on_device_discovered=lambda dev: discovered_list.append(dev),
    )

    remote_packet = {
        "id": 100,
        "type": "kdeconnect.identity",
        "body": {
            "deviceId": "remote_phone_99999999999999999999",
            "deviceName": "Pixel 8",
            "deviceType": "phone",
            "protocolVersion": 7,
            "tcpPort": 1720,
            "incomingCapabilities": ["kdeconnect.ping"],
            "outgoingCapabilities": ["kdeconnect.ping"],
        },
    }
    payload = json.dumps(remote_packet).encode("utf-8") + b"\n"
    discovery._handle_datagram(payload, ("192.168.1.50", 1720))

    assert len(discovered_list) == 1
    dev = discovered_list[0]
    assert dev.device_id == "remote_phone_99999999999999999999"
    assert dev.device_name == "Pixel 8"
    assert dev.device_type == "phone"
    assert dev.tcp_port == 1720
    assert dev.sender_ip == "192.168.1.50"
    assert dev.is_compatible() is True


def test_debounce_rapid_datagrams(sample_config: PenguLinkConfig) -> None:
    """Assert duplicate discovery packets within debounce window are suppressed."""
    discovered_list = []
    discovery = UDPDiscovery(
        config=sample_config,
        bound_tcp_port_getter=lambda: 1716,
        on_device_discovered=lambda dev: discovered_list.append(dev),
    )

    remote_packet = {
        "id": 101,
        "type": "kdeconnect.identity",
        "body": {
            "deviceId": "debounced_device_111111111111111111",
            "deviceName": "Galaxy Tab",
            "deviceType": "tablet",
            "protocolVersion": 7,
            "tcpPort": 1716,
        },
    }
    payload = json.dumps(remote_packet).encode("utf-8") + b"\n"

    # Send first time
    discovery._handle_datagram(payload, ("192.168.1.60", 1716))
    assert len(discovered_list) == 1

    # Send immediately again from same IP/port
    discovery._handle_datagram(payload, ("192.168.1.60", 1716))
    assert len(discovered_list) == 1  # Still 1, debounced!


def test_malformed_datagram_ignored(sample_config: PenguLinkConfig) -> None:
    """Assert invalid JSON or non-identity packets do not trigger callback."""
    discovered_list = []
    discovery = UDPDiscovery(
        config=sample_config,
        bound_tcp_port_getter=lambda: 1716,
        on_device_discovered=lambda dev: discovered_list.append(dev),
    )

    # 1. Invalid JSON
    discovery._handle_datagram(b"not json\n", ("192.168.1.70", 1716))
    # 2. Non-identity packet
    non_ident = json.dumps({"type": "kdeconnect.ping", "body": {}}).encode("utf-8")
    discovery._handle_datagram(non_ident, ("192.168.1.70", 1716))

    assert len(discovered_list) == 0


async def test_udp_start_and_stop(sample_config: PenguLinkConfig) -> None:
    """Assert UDPDiscovery starts and stops cleanly without dangling tasks."""
    # Use ephemeral port to avoid conflicts with system daemons
    sample_config.network.udp_port = 0
    discovery = UDPDiscovery(
        config=sample_config,
        bound_tcp_port_getter=lambda: 1716,
    )

    await discovery.start()
    assert discovery.is_running is True

    # Test single broadcast
    await discovery.broadcast_now(target_ip="127.0.0.1")

    await discovery.stop()
    assert discovery.is_running is False


def test_discovered_device_compatibility() -> None:
    """Assert DiscoveredDevice checks protocol version compatibility."""
    dev_v7 = DiscoveredDevice(
        device_id="dev_777777777777777777777777777777",
        device_name="Test Dev 7",
        device_type="phone",
        protocol_version=7,
        tcp_port=1716,
        sender_ip="192.168.1.10",
    )
    assert dev_v7.is_compatible() is True

    dev_v8 = DiscoveredDevice(
        device_id="dev_888888888888888888888888888888",
        device_name="Test Dev 8",
        device_type="desktop",
        protocol_version=8,
        tcp_port=1716,
        sender_ip="192.168.1.11",
    )
    assert dev_v8.is_compatible() is True

    dev_v5 = DiscoveredDevice(
        device_id="dev_555555555555555555555555555555",
        device_name="Test Dev 5",
        device_type="tablet",
        protocol_version=5,
        tcp_port=1716,
        sender_ip="192.168.1.12",
    )
    assert dev_v5.is_compatible() is False


async def test_udp_burst_and_sweep(sample_config: PenguLinkConfig) -> None:
    """Assert UDP burst and sweep dispatch functions execute properly."""
    sample_config.network.udp_port = 0
    discovery = UDPDiscovery(
        config=sample_config,
        bound_tcp_port_getter=lambda: 1716,
    )
    await discovery.start()

    # Trigger short burst
    await discovery.trigger_burst(count=2, delay=0.01)

    # Trigger short sweep
    await discovery.trigger_sweep(duration=0.05, interval=0.02)
    await asyncio.sleep(0.06)

    await discovery.stop()


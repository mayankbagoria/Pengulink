"""tests.unit.test_connectivity_report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive unit test suite for KDE Connect Connectivity Report plugin:
- Data models (SignalState, ConnectivityReport) and serialization (to_dict)
- Defensive parsing (parse_signal_state) with bounds clamping (0..4) and normalization
- Max subscription bounds (16 max) and string sanitization
- Packet creation (kdeconnect.connectivity_report.request)
- ConnectivityReportPlugin lifecycle, capability advertising, and EventBus integration
- Memory cache management and automatic eviction on DeviceUnpaired
- Malformed/adversarial wire payload resilience
- Daemon JSON-RPC endpoints (get_connectivity_report, request_connectivity_report)
- CLI formatting helpers and 'pengulink connectivity' command
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import (
    cli,
    format_network_badge,
    format_signal_bars,
)
from pengulink.core.config import PenguLinkConfig
from pengulink.core.events import (
    ConnectivityReportReceived,
    DeviceUnpaired,
    Event,
    EventBus,
)
from pengulink.core.exceptions import ConnectionClosedError, DeviceNotFoundError
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.network.packet import Packet
from pengulink.network.tcp import TCPConnection
from pengulink.plugins.base import PluginRegistry
from pengulink.plugins.connectivity_report import (
    CONNECTIVITY_REPORT_PACKET_TYPE,
    CONNECTIVITY_REPORT_REQUEST_TYPE,
    KNOWN_NETWORK_TYPES,
    MAX_NETWORK_TYPE_LEN,
    MAX_SUB_ID_LEN,
    MAX_SUBSCRIPTIONS,
    ConnectivityReport,
    ConnectivityReportPlugin,
    SignalState,
    create_connectivity_request_packet,
    parse_signal_state,
)


# =====================================================================
# 1. Models & Serialization Tests
# =====================================================================

def test_signal_state_defaults_and_custom_values() -> None:
    """Verify SignalState model defaults and custom initialization."""
    default_state = SignalState()
    assert default_state.network_type == "Unknown"
    assert default_state.signal_strength == 0

    custom_state = SignalState(network_type="5G", signal_strength=4)
    assert custom_state.network_type == "5G"
    assert custom_state.signal_strength == 4


def test_signal_state_serialization() -> None:
    """Verify SignalState.to_dict() provides both snake_case and camelCase keys."""
    state = SignalState(network_type="LTE", signal_strength=3)
    d = state.to_dict()
    assert d["network_type"] == "LTE"
    assert d["signal_strength"] == 3
    assert d["networkType"] == "LTE"
    assert d["signalStrength"] == 3


def test_signal_state_immutability() -> None:
    """Verify SignalState is a frozen dataclass."""
    state = SignalState(network_type="5G", signal_strength=4)
    with pytest.raises((AttributeError, TypeError)):
        state.signal_strength = 2  # type: ignore[misc]


def test_connectivity_report_defaults_and_serialization() -> None:
    """Verify ConnectivityReport model defaults and to_dict serialization."""
    before = time.time()
    report = ConnectivityReport(device_id="dev_pixel_9")
    after = time.time()

    assert report.device_id == "dev_pixel_9"
    assert report.signal_strengths == {}
    assert before <= report.timestamp <= after

    # Serialization
    sub1 = SignalState(network_type="5G", signal_strength=4)
    sub2 = SignalState(network_type="LTE", signal_strength=2)
    multi_report = ConnectivityReport(
        device_id="dev_pixel_9",
        signal_strengths={"0": sub1, "1": sub2},
        timestamp=1696412345.0,
    )
    serialized = multi_report.to_dict()

    assert serialized["device_id"] == "dev_pixel_9"
    assert serialized["timestamp"] == 1696412345.0
    assert "0" in serialized["signal_strengths"]
    assert "1" in serialized["signal_strengths"]
    assert serialized["signal_strengths"]["0"]["network_type"] == "5G"
    assert serialized["signal_strengths"]["0"]["signal_strength"] == 4
    assert serialized["signal_strengths"]["1"]["network_type"] == "LTE"
    assert serialized["signal_strengths"]["1"]["signal_strength"] == 2


def test_connectivity_report_immutability() -> None:
    """Verify ConnectivityReport is a frozen dataclass."""
    report = ConnectivityReport(device_id="dev_1")
    with pytest.raises((AttributeError, TypeError)):
        report.device_id = "dev_2"  # type: ignore[misc]


# =====================================================================
# 2. Defensive Parsing & Bounds Clamping (parse_signal_state)
# =====================================================================

def test_parse_signal_state_valid_inputs() -> None:
    """Verify parse_signal_state accurately decodes valid wire dictionaries."""
    res1 = parse_signal_state({"networkType": "5G", "signalStrength": 4})
    assert res1.network_type == "5G"
    assert res1.signal_strength == 4

    res2 = parse_signal_state({"networkType": "LTE", "signalStrength": 3})
    assert res2.network_type == "LTE"
    assert res2.signal_strength == 3

    # Snake_case fallback keys
    res3 = parse_signal_state({"network_type": "GSM", "signal_strength": 1})
    assert res3.network_type == "GSM"
    assert res3.signal_strength == 1


def test_parse_signal_state_clamping_bounds() -> None:
    """Verify signalStrength integer is clamped strictly to 0..4."""
    # Negative values clamp to 0
    assert parse_signal_state({"signalStrength": -1}).signal_strength == 0
    assert parse_signal_state({"signalStrength": -100}).signal_strength == 0

    # Values > 4 clamp to 4
    assert parse_signal_state({"signalStrength": 5}).signal_strength == 4
    assert parse_signal_state({"signalStrength": 999}).signal_strength == 4

    # Boundaries 0 and 4 are preserved
    assert parse_signal_state({"signalStrength": 0}).signal_strength == 0
    assert parse_signal_state({"signalStrength": 4}).signal_strength == 4


def test_parse_signal_state_numeric_conversions() -> None:
    """Verify floats and numeric strings are converted and rounded cleanly."""
    # Float rounding
    assert parse_signal_state({"signalStrength": 3.7}).signal_strength == 4
    assert parse_signal_state({"signalStrength": 2.2}).signal_strength == 2
    assert parse_signal_state({"signalStrength": 0.4}).signal_strength == 0
    assert parse_signal_state({"signalStrength": -0.8}).signal_strength == 0
    assert parse_signal_state({"signalStrength": 4.9}).signal_strength == 4

    # Numeric strings
    assert parse_signal_state({"signalStrength": "3"}).signal_strength == 3
    assert parse_signal_state({"signalStrength": " 2 "}).signal_strength == 2
    assert parse_signal_state({"signalStrength": "4.2"}).signal_strength == 4
    assert parse_signal_state({"signalStrength": "-2"}).signal_strength == 0
    assert parse_signal_state({"signalStrength": "10"}).signal_strength == 4


def test_parse_signal_state_non_numeric_defaults() -> None:
    """Verify invalid or non-numeric signalStrength defaults to 0 safely."""
    assert parse_signal_state({"signalStrength": "invalid"}).signal_strength == 0
    assert parse_signal_state({"signalStrength": None}).signal_strength == 0
    assert parse_signal_state({"signalStrength": []}).signal_strength == 0
    assert parse_signal_state({"signalStrength": {}}).signal_strength == 0
    assert parse_signal_state({}).signal_strength == 0


def test_parse_signal_state_network_type_normalization() -> None:
    """Verify networkType string uppercase normalization and whitespace stripping."""
    assert parse_signal_state({"networkType": "lte"}).network_type == "LTE"
    assert parse_signal_state({"networkType": "5g"}).network_type == "5G"
    assert parse_signal_state({"networkType": "  edge  "}).network_type == "EDGE"
    assert parse_signal_state({"networkType": "hspa"}).network_type == "HSPA"
    assert parse_signal_state({"networkType": "4g"}).network_type == "4G"
    assert parse_signal_state({"networkType": "3g"}).network_type == "3G"
    assert parse_signal_state({"networkType": "2g"}).network_type == "2G"

    # Preserves non-canonical future radio descriptor strings
    assert parse_signal_state({"networkType": "5G_NSA"}).network_type == "5G_NSA"
    assert parse_signal_state({"networkType": "LTE_ADVANCED"}).network_type == "LTE_ADVANCED"

    # Empty string or missing defaults to 'Unknown'
    assert parse_signal_state({"networkType": ""}).network_type == "Unknown"
    assert parse_signal_state({"networkType": "   "}).network_type == "Unknown"
    assert parse_signal_state({}).network_type == "Unknown"
    assert parse_signal_state({"networkType": None}).network_type == "Unknown"


def test_parse_signal_state_string_truncation() -> None:
    """Verify networkType strings exceeding MAX_NETWORK_TYPE_LEN (32) are truncated."""
    excessive_type = "A" * 100
    res = parse_signal_state({"networkType": excessive_type})
    assert len(res.network_type) == MAX_NETWORK_TYPE_LEN
    assert res.network_type == "A" * MAX_NETWORK_TYPE_LEN


def test_parse_signal_state_non_dict_fallback() -> None:
    """Verify parse_signal_state with non-dict input returns safe default SignalState."""
    assert parse_signal_state(None) == SignalState(network_type="Unknown", signal_strength=0)
    assert parse_signal_state("not_a_dict") == SignalState(network_type="Unknown", signal_strength=0)
    assert parse_signal_state(12345) == SignalState(network_type="Unknown", signal_strength=0)
    assert parse_signal_state([1, 2, 3]) == SignalState(network_type="Unknown", signal_strength=0)


# =====================================================================
# 3. Packet Builder & Capabilities Tests
# =====================================================================

def test_create_connectivity_request_packet() -> None:
    """Verify create_connectivity_request_packet creates valid query packet."""
    pkt = create_connectivity_request_packet()
    assert isinstance(pkt, Packet)
    assert pkt.type == CONNECTIVITY_REPORT_REQUEST_TYPE
    assert pkt.type == "kdeconnect.connectivity_report.request"
    assert pkt.body == {}


def test_plugin_capabilities_and_id() -> None:
    """Verify ConnectivityReportPlugin ID and capability advertisements."""
    plugin = ConnectivityReportPlugin()
    assert plugin.plugin_id == "connectivity_report"
    assert plugin.incoming_capabilities == [CONNECTIVITY_REPORT_PACKET_TYPE]
    assert plugin.incoming_capabilities == ["kdeconnect.connectivity_report"]
    assert plugin.outgoing_capabilities == [CONNECTIVITY_REPORT_REQUEST_TYPE]
    assert plugin.outgoing_capabilities == ["kdeconnect.connectivity_report.request"]


def test_plugin_registry_integration() -> None:
    """Verify ConnectivityReportPlugin registers cleanly in PluginRegistry."""
    bus = EventBus()
    registry = PluginRegistry(bus)
    plugin = ConnectivityReportPlugin(bus)

    registry.register(plugin)

    assert CONNECTIVITY_REPORT_PACKET_TYPE in registry.get_all_incoming_capabilities()
    assert CONNECTIVITY_REPORT_REQUEST_TYPE in registry.get_all_outgoing_capabilities()


# =====================================================================
# 4. Inbound Packet Handling & EventBus Tests
# =====================================================================

async def test_plugin_handle_packet_single_sim() -> None:
    """Verify processing single-SIM connectivity report updates cache and emits event."""
    bus = EventBus()
    events: List[ConnectivityReportReceived] = []
    bus.subscribe(ConnectivityReportReceived, lambda e: events.append(e))

    plugin = ConnectivityReportPlugin(bus)
    sent_packets: List[Packet] = []

    packet = Packet(
        type=CONNECTIVITY_REPORT_PACKET_TYPE,
        body={
            "signalStrengths": {
                "0": {
                    "networkType": "5G",
                    "signalStrength": 4,
                }
            }
        },
    )

    await plugin.handle_packet(packet, device_id="phone_alpha", send_packet=lambda p: sent_packets.append(p))

    # Verify cache
    cached = plugin.get_device_report("phone_alpha")
    assert cached is not None
    assert cached.device_id == "phone_alpha"
    assert len(cached.signal_strengths) == 1
    assert "0" in cached.signal_strengths
    assert cached.signal_strengths["0"].network_type == "5G"
    assert cached.signal_strengths["0"].signal_strength == 4

    # Verify EventBus emission
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, ConnectivityReportReceived)
    assert ev.device_id == "phone_alpha"
    assert "0" in ev.signal_strengths
    assert ev.signal_strengths["0"]["network_type"] == "5G"
    assert ev.signal_strengths["0"]["signal_strength"] == 4


async def test_plugin_handle_packet_multi_sim() -> None:
    """Verify processing multi-SIM connectivity report with distinct subscriptions."""
    bus = EventBus()
    events: List[ConnectivityReportReceived] = []
    bus.subscribe(ConnectivityReportReceived, lambda e: events.append(e))

    plugin = ConnectivityReportPlugin(bus)

    packet = Packet(
        type=CONNECTIVITY_REPORT_PACKET_TYPE,
        body={
            "signalStrengths": {
                "0": {"networkType": "5G", "signalStrength": 4},
                "1": {"networkType": "LTE", "signalStrength": 2},
                "custom_slot_2": {"networkType": "3G", "signalStrength": 1},
            }
        },
    )

    await plugin.handle_packet(packet, device_id="phone_dual", send_packet=lambda p: None)

    cached = plugin.get_device_report("phone_dual")
    assert cached is not None
    assert len(cached.signal_strengths) == 3
    assert cached.signal_strengths["0"].network_type == "5G"
    assert cached.signal_strengths["0"].signal_strength == 4
    assert cached.signal_strengths["1"].network_type == "LTE"
    assert cached.signal_strengths["1"].signal_strength == 2
    assert cached.signal_strengths["custom_slot_2"].network_type == "3G"
    assert cached.signal_strengths["custom_slot_2"].signal_strength == 1

    assert len(events) == 1
    assert len(events[0].signal_strengths) == 3


async def test_plugin_handle_packet_airplane_mode_empty_strengths() -> None:
    """Verify empty signalStrengths (airplane mode or tablet) creates valid 0-SIM report."""
    bus = EventBus()
    events: List[ConnectivityReportReceived] = []
    bus.subscribe(ConnectivityReportReceived, lambda e: events.append(e))

    plugin = ConnectivityReportPlugin(bus)

    packet = Packet(
        type=CONNECTIVITY_REPORT_PACKET_TYPE,
        body={"signalStrengths": {}},
    )

    await plugin.handle_packet(packet, device_id="tablet_wifi", send_packet=lambda p: None)

    cached = plugin.get_device_report("tablet_wifi")
    assert cached is not None
    assert cached.signal_strengths == {}

    assert len(events) == 1
    assert events[0].device_id == "tablet_wifi"
    assert events[0].signal_strengths == {}


async def test_plugin_handle_packet_max_subscriptions_limit() -> None:
    """Verify subscriptions beyond MAX_SUBSCRIPTIONS (16) are safely truncated."""
    plugin = ConnectivityReportPlugin()

    # Create 30 subscriptions
    excessive_strengths = {
        f"slot_{i}": {"networkType": "LTE", "signalStrength": 3}
        for i in range(30)
    }

    packet = Packet(
        type=CONNECTIVITY_REPORT_PACKET_TYPE,
        body={"signalStrengths": excessive_strengths},
    )

    await plugin.handle_packet(packet, device_id="phone_overflow", send_packet=lambda p: None)

    cached = plugin.get_device_report("phone_overflow")
    assert cached is not None
    assert len(cached.signal_strengths) == MAX_SUBSCRIPTIONS
    assert len(cached.signal_strengths) == 16


async def test_plugin_handle_packet_sub_id_sanitization() -> None:
    """Verify subscription identifier string sanitization and truncation."""
    plugin = ConnectivityReportPlugin()

    oversized_sub = "S" * 80
    packet = Packet(
        type=CONNECTIVITY_REPORT_PACKET_TYPE,
        body={
            "signalStrengths": {
                "  trimmed_slot  ": {"networkType": "5G", "signalStrength": 4},
                "": {"networkType": "LTE", "signalStrength": 2},
                oversized_sub: {"networkType": "GSM", "signalStrength": 1},
            }
        },
    )

    await plugin.handle_packet(packet, device_id="phone_sanitize", send_packet=lambda p: None)

    cached = plugin.get_device_report("phone_sanitize")
    assert cached is not None
    assert "trimmed_slot" in cached.signal_strengths
    assert "0" in cached.signal_strengths  # Empty sub_id defaults to "0"
    truncated_key = oversized_sub[:MAX_SUB_ID_LEN]
    assert truncated_key in cached.signal_strengths
    assert len(truncated_key) == MAX_SUB_ID_LEN


async def test_plugin_handle_packet_ignores_unrelated_types() -> None:
    """Verify plugin ignores packets that do not match CONNECTIVITY_REPORT_PACKET_TYPE."""
    bus = EventBus()
    events: List[ConnectivityReportReceived] = []
    bus.subscribe(ConnectivityReportReceived, lambda e: events.append(e))

    plugin = ConnectivityReportPlugin(bus)

    ping_packet = Packet(type="kdeconnect.ping", body={"message": "hello"})
    await plugin.handle_packet(ping_packet, device_id="phone_ping", send_packet=lambda p: None)

    assert plugin.get_device_report("phone_ping") is None
    assert len(events) == 0


# =====================================================================
# 5. Malformed Payload Handling & Resilience Tests
# =====================================================================

async def test_plugin_handle_packet_non_dict_body() -> None:
    """Verify plugin safely ignores packets with non-dictionary body without raising."""
    plugin = ConnectivityReportPlugin()

    packet = Packet.model_construct(type=CONNECTIVITY_REPORT_PACKET_TYPE, body="invalid_string_body")
    await plugin.handle_packet(packet, device_id="dev_corrupt", send_packet=lambda p: None)

    assert plugin.get_device_report("dev_corrupt") is None


async def test_plugin_handle_packet_missing_or_invalid_signal_strengths() -> None:
    """Verify missing, null, or non-dict signalStrengths does not crash."""
    plugin = ConnectivityReportPlugin()

    # Missing signalStrengths key
    p1 = Packet(type=CONNECTIVITY_REPORT_PACKET_TYPE, body={})
    await plugin.handle_packet(p1, device_id="dev_missing", send_packet=lambda p: None)
    rep1 = plugin.get_device_report("dev_missing")
    assert rep1 is not None
    assert rep1.signal_strengths == {}

    # signalStrengths is None
    p2 = Packet(type=CONNECTIVITY_REPORT_PACKET_TYPE, body={"signalStrengths": None})
    await plugin.handle_packet(p2, device_id="dev_null", send_packet=lambda p: None)
    rep2 = plugin.get_device_report("dev_null")
    assert rep2 is not None
    assert rep2.signal_strengths == {}

    # signalStrengths is a list or string
    p3 = Packet(type=CONNECTIVITY_REPORT_PACKET_TYPE, body={"signalStrengths": ["not", "a", "dict"]})
    await plugin.handle_packet(p3, device_id="dev_list", send_packet=lambda p: None)
    rep3 = plugin.get_device_report("dev_list")
    assert rep3 is not None
    assert rep3.signal_strengths == {}


# =====================================================================
# 6. State Cache & Lifecycle Cleanup Tests
# =====================================================================

async def test_plugin_cache_management_and_clear_device() -> None:
    """Verify get_device_report, clear_device, and clear_all methods."""
    plugin = ConnectivityReportPlugin()
    assert plugin.get_device_report("nonexistent") is None

    p1 = Packet(type=CONNECTIVITY_REPORT_PACKET_TYPE, body={"signalStrengths": {"0": {"networkType": "5G", "signalStrength": 4}}})
    await plugin.handle_packet(p1, device_id="dev_A", send_packet=lambda p: None)
    await plugin.handle_packet(p1, device_id="dev_B", send_packet=lambda p: None)

    assert plugin.get_device_report("dev_A") is not None
    assert plugin.get_device_report("dev_B") is not None

    # Clear single device
    assert plugin.clear_device("dev_A") is True
    assert plugin.clear_device("dev_A") is False
    assert plugin.get_device_report("dev_A") is None
    assert plugin.get_device_report("dev_B") is not None

    # Clear all
    plugin.clear_all()
    assert plugin.get_device_report("dev_B") is None


async def test_plugin_unpair_event_automatic_eviction() -> None:
    """Verify publishing DeviceUnpaired on EventBus triggers immediate cache eviction."""
    bus = EventBus()
    plugin = ConnectivityReportPlugin(bus)

    # Ingest reports for two devices
    pkt = Packet(type=CONNECTIVITY_REPORT_PACKET_TYPE, body={"signalStrengths": {"0": {"networkType": "5G", "signalStrength": 4}}})
    await plugin.handle_packet(pkt, device_id="dev_unpair_target", send_packet=lambda p: None)
    await plugin.handle_packet(pkt, device_id="dev_stay", send_packet=lambda p: None)

    assert plugin.get_device_report("dev_unpair_target") is not None
    assert plugin.get_device_report("dev_stay") is not None

    # Publish DeviceUnpaired event
    await bus.publish(DeviceUnpaired(device_id="dev_unpair_target"))

    # Assert target cleared, other device retained
    assert plugin.get_device_report("dev_unpair_target") is None
    assert plugin.get_device_report("dev_stay") is not None


async def test_plugin_stop_lifecycle() -> None:
    """Verify plugin.stop() unsubscribes from EventBus and purges all cached reports."""
    bus = EventBus()
    plugin = ConnectivityReportPlugin(bus)

    pkt = Packet(type=CONNECTIVITY_REPORT_PACKET_TYPE, body={"signalStrengths": {"0": {"networkType": "LTE", "signalStrength": 3}}})
    await plugin.handle_packet(pkt, device_id="dev_1", send_packet=lambda p: None)
    assert plugin.get_device_report("dev_1") is not None

    await plugin.stop()

    assert plugin.get_device_report("dev_1") is None
    assert plugin._unsub_unpair is None


# =====================================================================
# 7. Outbound Request Helper Tests
# =====================================================================

async def test_plugin_request_connectivity_report() -> None:
    """Verify request_connectivity_report transmits valid query packet."""
    plugin = ConnectivityReportPlugin()
    sent_packets: List[Packet] = []

    async def mock_send(p: Packet) -> None:
        sent_packets.append(p)

    pkt = await plugin.request_connectivity_report(mock_send)

    assert len(sent_packets) == 1
    assert sent_packets[0] is pkt
    assert sent_packets[0].type == CONNECTIVITY_REPORT_REQUEST_TYPE
    assert sent_packets[0].body == {}


# =====================================================================
# 8. Daemon JSON-RPC Integration Tests
# =====================================================================

@pytest.fixture
def test_daemon_config(tmp_path: Path) -> PenguLinkConfig:
    """Construct isolated daemon configuration in tmp_path."""
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / "config"
    cfg.storage.data_dir = tmp_path / "data"
    cfg.storage.runtime_dir = tmp_path / "run"
    cfg.device.device_id = "testdaemon1234567890abcdef123456"
    cfg.device.device_name = "PenguLink-Daemon-CR-Test"
    cfg.network.udp_port = 17169
    cfg.network.tcp_port_min = 17210
    cfg.network.tcp_port_max = 17215
    cfg.ensure_directories()
    return cfg


async def test_daemon_get_connectivity_report_missing_device_id(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify get_connectivity_report validation rejects empty or missing device_id."""
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)

    with pytest.raises(ValueError, match="Parameter 'device_id' is required"):
        await daemon._handle_ipc_request("get_connectivity_report", {"device_id": ""})

    with pytest.raises(ValueError, match="Parameter 'device_id' is required"):
        await daemon._handle_ipc_request("get_connectivity_report", {})

    with pytest.raises(ValueError, match="Parameter 'device_id' is required"):
        await daemon._handle_ipc_request("request_connectivity_report", {"device_id": ""})

    with pytest.raises(ValueError, match="Parameter 'device_id' is required"):
        await daemon._handle_ipc_request("request_connectivity_report", {})


async def test_daemon_get_connectivity_report_unknown_device_raises(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify get_connectivity_report raises DeviceNotFoundError for unknown device."""
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)

    with pytest.raises(DeviceNotFoundError, match="Device 'ghost_device' is unknown"):
        await daemon.get_connectivity_report("ghost_device")


async def test_daemon_get_connectivity_report_known_but_no_report(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify get_connectivity_report returns None when device is known but has not reported."""
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)

    # Seed discovered device
    daemon._discovered_devices["known_device_1"] = MagicMock()

    report = await daemon.get_connectivity_report("known_device_1")
    assert report is None


async def test_daemon_get_connectivity_report_cached_success(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify get_connectivity_report returns dictionary of cached report."""
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)

    daemon._discovered_devices["active_phone"] = MagicMock()

    # Ingest report
    pkt = Packet(
        type=CONNECTIVITY_REPORT_PACKET_TYPE,
        body={"signalStrengths": {"0": {"networkType": "5G", "signalStrength": 4}}},
    )
    await daemon.connectivity_plugin.handle_packet(pkt, device_id="active_phone", send_packet=lambda p: None)

    report_dict = await daemon.get_connectivity_report("active_phone")
    assert report_dict is not None
    assert report_dict["device_id"] == "active_phone"
    assert "0" in report_dict["signal_strengths"]
    assert report_dict["signal_strengths"]["0"]["network_type"] == "5G"
    assert report_dict["signal_strengths"]["0"]["signal_strength"] == 4

    # Also verify via central JSON-RPC dispatcher _handle_ipc_request
    rpc_res = await daemon._handle_ipc_request("get_connectivity_report", {"device_id": "active_phone"})
    assert rpc_res == report_dict


async def test_daemon_request_connectivity_report_not_connected(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify request_connectivity_report raises ConnectionClosedError if device is offline."""
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)

    with pytest.raises(ConnectionClosedError, match="Device offline_device is not connected"):
        await daemon.request_connectivity_report("offline_device")


async def test_daemon_request_connectivity_report_connected_success(test_daemon_config: PenguLinkConfig, tmp_path: Path) -> None:
    """Verify request_connectivity_report sends packet when device is actively connected."""
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(config=test_daemon_config, socket_path=sock_path)

    mock_conn = MagicMock(spec=TCPConnection)
    mock_conn.is_closed = False
    mock_conn.write_line = AsyncMock()
    daemon._active_connections["online_device"] = mock_conn

    res = await daemon.request_connectivity_report("online_device")
    assert res["success"] is True
    assert res["device_id"] == "online_device"
    assert mock_conn.write_line.called

    # Also verify via central JSON-RPC dispatcher _handle_ipc_request
    rpc_res = await daemon._handle_ipc_request("request_connectivity_report", {"device_id": "online_device"})
    assert rpc_res["success"] is True
    assert rpc_res["device_id"] == "online_device"


# =====================================================================
# 9. CLI Formatting & Command Tests
# =====================================================================

def test_cli_format_signal_bars() -> None:
    """Verify format_signal_bars produces correct Rich indicators."""
    assert "●●●●" in format_signal_bars(4)
    assert "(Great)" in format_signal_bars(4)

    assert "●●●○" in format_signal_bars(3)
    assert "(Good)" in format_signal_bars(3)

    assert "●●○○" in format_signal_bars(2)
    assert "(Moderate)" in format_signal_bars(2)

    assert "●○○○" in format_signal_bars(1)
    assert "(Poor)" in format_signal_bars(1)

    assert "○○○○" in format_signal_bars(0)
    assert "(No Signal)" in format_signal_bars(0)

    # Out of range fallback
    assert "○○○○" in format_signal_bars(-1)
    assert "○○○○" in format_signal_bars(5)


def test_cli_format_network_badge() -> None:
    """Verify format_network_badge colors radio generations accurately."""
    assert "5G" in format_network_badge("5G")
    assert "magenta" in format_network_badge("5G")

    assert "LTE" in format_network_badge("LTE")
    assert "cyan" in format_network_badge("4G")

    assert "3G" in format_network_badge("3G")
    assert "3G" in format_network_badge("HSPA")

    assert "2G" in format_network_badge("EDGE")
    assert "2G" in format_network_badge("GSM")

    assert "Unknown" in format_network_badge("Unknown")
    assert "CustomRadio" in format_network_badge("CustomRadio")


def test_cli_connectivity_command_rich_single_sim(tmp_path: Path) -> None:
    """Verify 'pengulink connectivity <dev>' renders single SIM Rich table panel."""
    runner = CliRunner(env={"COLUMNS": "160"})
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    sample_report = {
        "device_id": "pixel_8_device",
        "signal_strengths": {
            "0": {
                "network_type": "5G",
                "signal_strength": 4,
            }
        },
        "timestamp": time.time(),
    }

    with patch("pengulink.cli.main.run_ipc_call", return_value=sample_report):
        res = runner.invoke(cli, ["-s", str(mock_socket), "connectivity", "pixel_8_device"])
        assert res.exit_code == 0
        assert "Cellular Connectivity: pixel_8_device" in res.output
        assert "SIM 1 (Slot 0)" in res.output
        assert "5G" in res.output
        assert "●●●●" in res.output
        assert "Great" in res.output
        assert "Active" in res.output


def test_cli_connectivity_command_rich_multi_sim(tmp_path: Path) -> None:
    """Verify 'pengulink connectivity <dev>' renders multi-SIM table with all slots."""
    runner = CliRunner(env={"COLUMNS": "160"})
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    sample_report = {
        "device_id": "dual_sim_phone",
        "signal_strengths": {
            "0": {"network_type": "5G", "signal_strength": 4},
            "1": {"network_type": "LTE", "signal_strength": 2},
        },
        "timestamp": time.time(),
    }

    with patch("pengulink.cli.main.run_ipc_call", return_value=sample_report):
        res = runner.invoke(cli, ["-s", str(mock_socket), "connectivity", "dual_sim_phone"])
        assert res.exit_code == 0
        assert "SIM 1 (Slot 0)" in res.output
        assert "SIM 2 (Slot 1)" in res.output
        assert "5G" in res.output
        assert "LTE" in res.output
        assert "●●●●" in res.output
        assert "●●○○" in res.output


def test_cli_connectivity_command_json_output(tmp_path: Path) -> None:
    """Verify 'pengulink connectivity <dev> --json' outputs machine-readable JSON."""
    runner = CliRunner(env={"COLUMNS": "160"})
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    sample_report = {
        "device_id": "pixel_8_device",
        "signal_strengths": {
            "0": {"network_type": "5G", "signal_strength": 4}
        },
        "timestamp": 1696412345.0,
    }

    with patch("pengulink.cli.main.run_ipc_call", return_value=sample_report):
        res = runner.invoke(cli, ["-s", str(mock_socket), "connectivity", "pixel_8_device", "--json"])
        assert res.exit_code == 0
        parsed = json.loads(res.output)
        assert parsed["device_id"] == "pixel_8_device"
        assert parsed["signal_strengths"]["0"]["network_type"] == "5G"
        assert parsed["signal_strengths"]["0"]["signal_strength"] == 4


def test_cli_connectivity_command_airplane_mode(tmp_path: Path) -> None:
    """Verify 'pengulink connectivity <dev>' renders warning when signalStrengths is empty."""
    runner = CliRunner(env={"COLUMNS": "160"})
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    sample_report = {
        "device_id": "airplane_phone",
        "signal_strengths": {},
        "timestamp": time.time(),
    }

    with patch("pengulink.cli.main.run_ipc_call", return_value=sample_report):
        res = runner.invoke(cli, ["-s", str(mock_socket), "connectivity", "airplane_phone"])
        assert res.exit_code == 0
        assert "No active cellular subscriptions reported" in res.output
        assert "Airplane Mode" in res.output


def test_cli_connectivity_command_no_report_available(tmp_path: Path) -> None:
    """Verify 'pengulink connectivity <dev>' outputs prompt when no report has arrived."""
    runner = CliRunner(env={"COLUMNS": "160"})
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    with patch("pengulink.cli.main.run_ipc_call", return_value=None):
        res = runner.invoke(cli, ["-s", str(mock_socket), "connectivity", "new_device"])
        assert res.exit_code == 0
        assert "No connectivity report available yet for device 'new_device'" in res.output
        assert "--request" in res.output


def test_cli_connectivity_command_request_flag(tmp_path: Path) -> None:
    """Verify '--request' flag triggers request_connectivity_report prior to querying."""
    runner = CliRunner(env={"COLUMNS": "160"})
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    calls: List[str] = []

    def mock_ipc(ctx: Any, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Any:
        calls.append(method)
        if method == "request_connectivity_report":
            return {"success": True}
        elif method == "get_connectivity_report":
            return {
                "device_id": "req_device",
                "signal_strengths": {"0": {"network_type": "5G", "signal_strength": 4}},
                "timestamp": time.time(),
            }
        return None

    with patch("pengulink.cli.main.run_ipc_call", side_effect=mock_ipc):
        res = runner.invoke(cli, ["-s", str(mock_socket), "connectivity", "req_device", "-r"])
        assert res.exit_code == 0
        assert calls == ["request_connectivity_report", "get_connectivity_report"]
        assert "5G" in res.output

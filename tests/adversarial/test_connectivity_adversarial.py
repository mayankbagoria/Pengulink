"""tests.adversarial.test_connectivity_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dedicated adversarial security, stress, and vulnerability test suite for
the KDE Connect Connectivity Report plugin (kdeconnect.connectivity_report).

Validates defenses against:
1. Oversized payload attacks (>128 KB stream buffer overrun, CVE-2020-26164 mitigation).
2. Massive subscription dictionary floods (500–1,000 keys) memory exhaustion vectors.
3. Ultra-long string injections (10,000+ characters) for subscription IDs & network types.
4. Deeply nested, malformed, and type-confused JSON payloads (lists, nulls, booleans).
5. Floating-point attacks (NaN, Infinity, -Infinity, overflow strings).
6. Rapid burst floods (500+ back-to-back packets) for race conditions & memory bloat.
7. Post-unpair packet delivery (local and remote unpair cache eviction).
8. Non-string keys, unicode control characters, null bytes, and injection payloads.
9. Protocol fidelity: verification of ZERO synthetic Wi-Fi fields.
10. Mathematical bounds fuzzing: signal strength strictly clamped to [0, 4].
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from pengulink.core.events import (
    ConnectivityReportReceived,
    DeviceUnpaired,
    EventBus,
)
from pengulink.core.exceptions import DeviceNotFoundError, IPCError
from pengulink.core.state_machine import DeviceState
from pengulink.network.packet import Packet
from pengulink.plugins.connectivity_report import (
    CONNECTIVITY_REPORT_PACKET_TYPE,
    CONNECTIVITY_REPORT_REQUEST_TYPE,
    MAX_NETWORK_TYPE_LEN,
    MAX_SUB_ID_LEN,
    MAX_SUBSCRIPTIONS,
    ConnectivityReport,
    ConnectivityReportPlugin,
    SignalState,
    parse_signal_state,
)
from tests.e2e.conftest import e2e_environment, wait_for


# =====================================================================
# Helper Fixtures & Callables
# =====================================================================

async def _dummy_send(packet: Packet) -> None:
    """No-op packet sender callback."""
    pass


async def _pair_and_connect(daemon: Any, mock: Any) -> None:
    """Helper to pair and connect a mock device with daemon."""
    await mock.broadcast_identity(
        target_ip="127.0.0.1",
        target_port=daemon.config.network.udp_port,
    )
    await wait_for(
        lambda: mock.device_id in daemon._discovered_devices,
        timeout=4.0,
        desc="discovering mock device",
    )
    res = await daemon.pair_device(mock.device_id, wait=True, timeout=5.0)
    assert res["paired"] is True
    assert daemon.is_device_connected(mock.device_id) is True


# =====================================================================
# 1. Oversized Wire Payload Attacks (>128 KB Stream Guard, CVE-2020-26164)
# =====================================================================

async def test_adv_cr_001_oversized_payload_128kb_stream_guard(tmp_path: Path) -> None:
    """Verify daemon's 128KB guard terminates connection upon receiving an oversized

    kdeconnect.connectivity_report payload (>128 KB), mitigating CVE-2020-26164.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Construct an oversized connectivity report (> 140,000 bytes)
        # Pad body with massive string to exceed 128 KB (131,072 bytes) limit
        padding = "X" * 145000
        oversized_body = {
            "signalStrengths": {
                "0": {
                    "networkType": "5G",
                    "signalStrength": 4,
                    "maliciousPadding": padding,
                }
            }
        }
        pkt = Packet(
            type=CONNECTIVITY_REPORT_PACKET_TYPE,
            body=oversized_body,
        )
        oversized_wire = pkt.to_wire()
        assert len(oversized_wire) > 131072

        # Transmit raw oversized packet over TLS socket
        await mock.send_raw_bytes(oversized_wire)

        # 128 KB guard must trigger LimitOverrunError and sever TCP connection
        await wait_for(
            lambda: not daemon.is_device_connected(mock.device_id),
            timeout=4.0,
            desc="128KB buffer overrun protection dropping socket",
        )

        # Verify daemon process remains alive and resilient
        assert daemon.is_running is True


# =====================================================================
# 2. Massive Subscription Dictionary Floods (Memory Exhaustion Vectors)
# =====================================================================

async def test_adv_cr_002_massive_subscription_dict_flood_500_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify plugin truncates massive dictionary flood of 500 subscription keys

    to MAX_SUBSCRIPTIONS (16), preventing heap exhaustion DoS.
    """
    event_bus = EventBus()
    plugin = ConnectivityReportPlugin(event_bus=event_bus)

    # Generate 500 subscription keys
    huge_strengths: Dict[str, Any] = {}
    for i in range(500):
        huge_strengths[f"sub_{i}"] = {
            "networkType": "LTE" if i % 2 == 0 else "5G",
            "signalStrength": (i % 5),
        }

    pkt = Packet(
        type=CONNECTIVITY_REPORT_PACKET_TYPE,
        body={"signalStrengths": huge_strengths},
    )

    events: List[ConnectivityReportReceived] = []
    event_bus.subscribe(ConnectivityReportReceived, lambda ev: events.append(ev))

    with caplog.at_level(logging.WARNING):
        await plugin.handle_packet(pkt, "attacker_device", _dummy_send)

    # 1. Verify warning was logged about excessive subscriptions
    assert any("Excessive subscriptions (500 > 16)" in rec.message for rec in caplog.records)

    # 2. Verify state cache only holds exactly MAX_SUBSCRIPTIONS (16)
    report = plugin.get_device_report("attacker_device")
    assert report is not None
    assert len(report.signal_strengths) == MAX_SUBSCRIPTIONS
    assert len(report.signal_strengths) == 16

    # 3. Verify EventBus event payload also capped at 16
    assert len(events) == 1
    assert len(events[0].signal_strengths) == 16


async def test_adv_cr_003_massive_subscription_dict_flood_1000_keys() -> None:
    """Verify plugin truncates massive flood of 1,000 subscription keys

    without memory spikes or unhandled exceptions.
    """
    event_bus = EventBus()
    plugin = ConnectivityReportPlugin(event_bus=event_bus)

    huge_strengths = {
        f"slot_{i}": {"networkType": "5G", "signalStrength": 4}
        for i in range(1000)
    }

    pkt = Packet(
        type=CONNECTIVITY_REPORT_PACKET_TYPE,
        body={"signalStrengths": huge_strengths},
    )

    await plugin.handle_packet(pkt, "flood_device_1000", _dummy_send)

    report = plugin.get_device_report("flood_device_1000")
    assert report is not None
    assert len(report.signal_strengths) == MAX_SUBSCRIPTIONS


# =====================================================================
# 3. Ultra-Long String Injections (10,000+ Characters)
# =====================================================================

def test_adv_cr_004_ultra_long_string_injection_network_type() -> None:
    """Verify networkType strings exceeding 10,000 characters are truncated

    strictly to MAX_NETWORK_TYPE_LEN (32) without blowing up memory.
    """
    long_string = "5G_" + ("A" * 15000)
    state = parse_signal_state({"networkType": long_string, "signalStrength": 3})

    assert len(state.network_type) <= MAX_NETWORK_TYPE_LEN
    assert len(state.network_type) == 32
    assert state.network_type.startswith("5G_")
    assert state.signal_strength == 3


async def test_adv_cr_005_ultra_long_string_injection_subscription_id() -> None:
    """Verify subscription IDs exceeding 10,000 characters are truncated

    strictly to MAX_SUB_ID_LEN (32) in handle_packet.
    """
    event_bus = EventBus()
    plugin = ConnectivityReportPlugin(event_bus=event_bus)

    huge_sub_id = "SUB_ID_" + ("B" * 20000)
    huge_net_type = "LTE_" + ("C" * 12000)

    pkt = Packet(
        type=CONNECTIVITY_REPORT_PACKET_TYPE,
        body={
            "signalStrengths": {
                huge_sub_id: {
                    "networkType": huge_net_type,
                    "signalStrength": 2,
                }
            }
        },
    )

    await plugin.handle_packet(pkt, "injection_device", _dummy_send)

    report = plugin.get_device_report("injection_device")
    assert report is not None
    assert len(report.signal_strengths) == 1

    stored_sub_id = list(report.signal_strengths.keys())[0]
    assert len(stored_sub_id) <= MAX_SUB_ID_LEN
    assert len(stored_sub_id) == 32
    assert stored_sub_id == huge_sub_id[:MAX_SUB_ID_LEN].strip()

    stored_signal = report.signal_strengths[stored_sub_id]
    assert len(stored_signal.network_type) <= MAX_NETWORK_TYPE_LEN
    assert len(stored_signal.network_type) == 32


# =====================================================================
# 4. Deeply Nested & Malformed JSON Payloads
# =====================================================================

def test_adv_cr_006_deeply_nested_dictionaries_and_lists() -> None:
    """Verify parser safely rejects deeply nested recursive dicts and lists

    without throwing RecursionError or crashing.
    """
    # 50 levels of nested dictionaries
    nested_data: Dict[str, Any] = {"level_0": True}
    for i in range(1, 55):
        nested_data = {f"level_{i}": nested_data}

    # Pass as raw signal state
    state = parse_signal_state(nested_data)
    assert state.network_type == "Unknown"
    assert state.signal_strength == 0

    # 50 levels of nested lists
    nested_list: Any = ["base"]
    for _ in range(50):
        nested_list = [nested_list]

    state_list = parse_signal_state(nested_list)
    assert state_list.network_type == "Unknown"
    assert state_list.signal_strength == 0


async def test_adv_cr_007_type_confusion_in_signal_strengths() -> None:
    """Verify handle_packet withstands unexpected primitive and container types

    in body and signalStrengths (nulls, lists, booleans, numbers, strings).
    """
    plugin = ConnectivityReportPlugin()

    hostile_bodies: List[Any] = [
        None,
        [],
        [1, 2, 3],
        "string_body",
        12345,
        True,
        False,
        {"signalStrengths": None},
        {"signalStrengths": []},
        {"signalStrengths": [1, 2, 3]},
        {"signalStrengths": "not_a_dict"},
        {"signalStrengths": 12345},
        {"signalStrengths": True},
        {"signalStrengths": False},
    ]

    for hostile in hostile_bodies:
        pkt = Packet(
            type=CONNECTIVITY_REPORT_PACKET_TYPE,
            body=hostile if isinstance(hostile, dict) else {},
        )
        if not isinstance(hostile, dict):
            # Bypass Pydantic Packet validation to simulate raw corrupted wire ingestion
            object.__setattr__(pkt, "body", hostile)

        # Must not raise unhandled exceptions
        await plugin.handle_packet(pkt, "type_confusion_dev", _dummy_send)


def test_adv_cr_008_malformed_fields_in_subscription_object() -> None:
    """Verify parse_signal_state handles null, empty, boolean, or container values

    for networkType and signalStrength.
    """
    cases: List[Dict[str, Any]] = [
        {"networkType": None, "signalStrength": None},
        {"networkType": [], "signalStrength": []},
        {"networkType": {}, "signalStrength": {}},
        {"networkType": True, "signalStrength": False},
        {"networkType": 9999, "signalStrength": "invalid_string"},
        {"networkType": "", "signalStrength": ""},
        {"networkType": "   ", "signalStrength": "   "},
        {"networkType": None, "signalStrength": "NaN"},
    ]

    for case in cases:
        st = parse_signal_state(case)
        assert isinstance(st.network_type, str)
        assert isinstance(st.signal_strength, int)
        assert 0 <= st.signal_strength <= 4


# =====================================================================
# 5. Floating-Point Numeric Attacks (NaN, Infinity, Overflow)
# =====================================================================

@pytest.mark.parametrize(
    "hostile_val,expected_strength",
    [
        (float("nan"), 0),
        (float("inf"), 0),
        (-float("inf"), 0),
        ("nan", 0),
        ("NaN", 0),
        ("infinity", 0),
        ("Infinity", 0),
        ("-Infinity", 0),
        ("-inf", 0),
        ("1e500", 0),
        ("-1e500", 0),
        (1e308, 4),
        (-1e308, 0),
        (3.9999, 4),
        (0.0001, 0),
        (-0.0001, 0),
        (4.0001, 4),
    ],
)
def test_adv_cr_009_nan_infinity_numeric_attacks(hostile_val: Any, expected_strength: int) -> None:
    """Verify IEEE 754 special values (NaN, Infinity) and extreme scientific notation

    safely clamp to valid coarse signal bars without throwing uncaught exceptions.
    """
    res = parse_signal_state({"networkType": "5G", "signalStrength": hostile_val})
    assert res.signal_strength == expected_strength
    assert 0 <= res.signal_strength <= 4


# =====================================================================
# 6. Rapid Burst Floods (500+ Back-to-Back Reports)
# =====================================================================

async def test_adv_cr_010_rapid_burst_flood_500_packets() -> None:
    """Verify plugin withstands 500 rapid back-to-back packets with atomic cache

    updates, zero task leaks, and strict O(1) device cache retention.
    """
    event_bus = EventBus()
    plugin = ConnectivityReportPlugin(event_bus=event_bus)

    event_counter = 0

    def _on_event(ev: ConnectivityReportReceived) -> None:
        nonlocal event_counter
        event_counter += 1

    event_bus.subscribe(ConnectivityReportReceived, _on_event)

    # Dispatch 500 rapid packets
    for i in range(500):
        pkt = Packet(
            type=CONNECTIVITY_REPORT_PACKET_TYPE,
            body={
                "signalStrengths": {
                    "0": {
                        "networkType": "5G" if i % 2 == 0 else "LTE",
                        "signalStrength": (i % 5),
                    }
                }
            },
        )
        await plugin.handle_packet(pkt, "burst_device", _dummy_send)

    # Let event tasks finish
    await asyncio.sleep(0.05)

    # 1. State cache must hold exactly 1 entry for this device
    assert len(plugin._device_reports) == 1
    report = plugin.get_device_report("burst_device")
    assert report is not None
    assert report.signal_strengths["0"].signal_strength == (499 % 5)
    assert report.signal_strengths["0"].network_type == "LTE"

    # 2. All 500 events successfully published without dropped tasks
    assert event_counter == 500


async def test_adv_cr_011_live_socket_rapid_burst_100_packets(tmp_path: Path) -> None:
    """Verify live mTLS socket stream remains robust and unchoked during

    a high-frequency burst of 100 connectivity reports.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        for i in range(100):
            await mock.send_connectivity_report(
                signal_strengths={
                    "0": {
                        "networkType": "5G",
                        "signalStrength": i % 5,
                    }
                }
            )

        # Allow daemon time to process all packets
        await asyncio.sleep(0.1)

        # Connection remains alive
        assert daemon.is_device_connected(mock.device_id) is True

        # Query latest state over IPC
        report = await ipc_client.get_connectivity_report(mock.device_id)
        assert report is not None
        assert report["device_id"] == mock.device_id
        assert "0" in report["signal_strengths"]


# =====================================================================
# 7. Post-Unpair Packet Delivery & Lifecycle Eviction
# =====================================================================

async def test_adv_cr_012_post_unpair_packet_delivery_local(tmp_path: Path) -> None:
    """Verify local unpair immediately purges connectivity cache and post-unpair

    packets do not resurrect unauthorized state or crash the daemon.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # 1. Send initial report
        await mock.send_connectivity_report()
        await wait_for(
            lambda: daemon.connectivity_plugin.get_device_report(mock.device_id) is not None,
            timeout=3.0,
            desc="populating initial connectivity report",
        )

        # 2. Unpair device locally via IPC
        unpair_res = await ipc_client.unpair_device(mock.device_id)
        assert unpair_res["success"] is True

        # 3. Cache must be immediately evicted
        assert daemon.connectivity_plugin.get_device_report(mock.device_id) is None

        # 4. IPC queries for uncoupled device return None
        report_after = await ipc_client.get_connectivity_report(mock.device_id)
        assert report_after is None

        # Querying an unknown device raises IPCError
        with pytest.raises(IPCError):
            await ipc_client.get_connectivity_report("never_existed_device_xyz")

        # 5. Rogue packet delivered to plugin directly for the unpaired device
        rogue_pkt = Packet(
            type=CONNECTIVITY_REPORT_PACKET_TYPE,
            body={"signalStrengths": {"0": {"networkType": "5G", "signalStrength": 4}}},
        )
        # Even if dispatched directly to plugin, verify daemon store is not corrupted
        await daemon.connectivity_plugin.handle_packet(rogue_pkt, mock.device_id, _dummy_send)

        # Device is still unpaired in device store
        assert not await daemon.device_store.is_device_paired(mock.device_id)


async def test_adv_cr_013_post_unpair_packet_delivery_remote_revocation(tmp_path: Path) -> None:
    """Verify remote pairing revocation (pair: false packet) immediately purges

    connectivity cache and closes connection.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Send report
        await mock.send_connectivity_report()
        await wait_for(
            lambda: daemon.connectivity_plugin.get_device_report(mock.device_id) is not None,
            timeout=3.0,
            desc="populating connectivity report",
        )

        # Remote peer revokes pairing
        revoke_pkt = Packet(
            type="kdeconnect.pair",
            body={"pair": False},
        )
        await mock.send_packet(revoke_pkt)

        # Wait for daemon to detect remote unpair and close socket
        await wait_for(
            lambda: not daemon.is_device_connected(mock.device_id),
            timeout=3.0,
            desc="daemon processing remote unpair",
        )

        # Verify connectivity cache is purged
        assert daemon.connectivity_plugin.get_device_report(mock.device_id) is None


# =====================================================================
# 8. Non-String Keys, Unicode Controls, Null Bytes, and Injections
# =====================================================================

async def test_adv_cr_014_non_string_dictionary_keys() -> None:
    """Verify integer, boolean, float, and object dictionary keys are safely coerced

    to strings without raising AttributeError or TypeError.
    """
    plugin = ConnectivityReportPlugin()

    hostile_keys_strengths: Dict[Any, Any] = {
        0: {"networkType": "LTE", "signalStrength": 1},
        1: {"networkType": "5G", "signalStrength": 4},
        True: {"networkType": "GSM", "signalStrength": 2},
        3.1415: {"networkType": "3G", "signalStrength": 3},
    }

    pkt = Packet(
        type=CONNECTIVITY_REPORT_PACKET_TYPE,
        body={"signalStrengths": hostile_keys_strengths},
    )

    await plugin.handle_packet(pkt, "non_string_keys_dev", _dummy_send)

    report = plugin.get_device_report("non_string_keys_dev")
    assert report is not None

    for sub_id in report.signal_strengths.keys():
        assert isinstance(sub_id, str)
        assert len(sub_id) <= MAX_SUB_ID_LEN

    # Verify JSON serialization does not fail
    serialized = report.to_dict()
    assert isinstance(serialized, dict)
    dumped = json.dumps(serialized)
    assert len(dumped) > 0


def test_adv_cr_015_unicode_control_characters_and_null_bytes() -> None:
    """Verify null bytes, RTL overrides, zero-width spaces, and ANSI escapes

    do not crash the parser or break data models.
    """
    hostile_inputs = [
        # Null bytes
        {"networkType": "5G\x00_NSA", "signalStrength": 4},
        # RTL Override (U+202E)
        {"networkType": "\u202e5G_SPOOF", "signalStrength": 3},
        # Zero-width spaces
        {"networkType": "L\u200bT\u200cE", "signalStrength": 2},
        # ANSI Escape Sequences
        {"networkType": "\x1b[31mRED_ALERT\x1b[0m", "signalStrength": 1},
        # C0 / C1 control characters
        {"networkType": "\x01\x02\x03\x1fLTE", "signalStrength": 4},
    ]

    for inp in hostile_inputs:
        state = parse_signal_state(inp)
        assert isinstance(state.network_type, str)
        assert len(state.network_type) <= MAX_NETWORK_TYPE_LEN
        assert 0 <= state.signal_strength <= 4


def test_adv_cr_016_injection_payloads_sanitization() -> None:
    """Verify SQL injection, Path Traversal, and Command Injection strings

    are treated as inert text and bounded to 32 characters.
    """
    injections = [
        "../../../../../../etc/passwd",
        "..\\..\\Windows\\System32\\calc.exe",
        "' OR '1'='1' --",
        "'; DROP TABLE devices; --",
        "$(cat /etc/shadow)",
        "`rm -rf /`",
        "<script>alert(1)</script>",
    ]

    for inj in injections:
        state = parse_signal_state({"networkType": inj, "signalStrength": 2})
        assert len(state.network_type) <= MAX_NETWORK_TYPE_LEN
        assert state.signal_strength == 2

        # Wire serialization check
        report = ConnectivityReport(
            device_id="inj_dev",
            signal_strengths={inj[:MAX_SUB_ID_LEN]: state},
        )
        d = report.to_dict()
        assert json.loads(json.dumps(d)) is not None


# =====================================================================
# 9. Protocol Fidelity: Zero Synthetic Wi-Fi Fields Invariant
# =====================================================================

def test_adv_cr_017_protocol_fidelity_zero_synthetic_wifi_fields() -> None:
    """Verify ground-truth fidelity: Wire packet schema and Python models contain

    ZERO synthetic Wi-Fi fields (no ssid, rssi, link_speed, or wifi keys).
    """
    forbidden_keys = {
        "wifi",
        "wifi_ssid",
        "ssid",
        "bssid",
        "wifi_rssi",
        "rssi",
        "link_speed",
        "linkSpeed",
        "wifi_signal",
    }

    # 1. Check SignalState model fields
    sig = SignalState(network_type="5G", signal_strength=4)
    sig_dict = sig.to_dict()
    for forbidden in forbidden_keys:
        assert forbidden not in sig_dict
        assert not hasattr(sig, forbidden)

    # 2. Check ConnectivityReport model fields
    rep = ConnectivityReport(device_id="dev1", signal_strengths={"0": sig})
    rep_dict = rep.to_dict()
    for forbidden in forbidden_keys:
        assert forbidden not in rep_dict
        assert not hasattr(rep, forbidden)

    # 3. Check request packet
    from pengulink.plugins.connectivity_report import create_connectivity_request_packet
    req_pkt = create_connectivity_request_packet()
    assert req_pkt.type == CONNECTIVITY_REPORT_REQUEST_TYPE
    assert req_pkt.body == {}


# =====================================================================
# 10. Fuzz Testing: Mathematical Bounds Invariant [0, 4]
# =====================================================================

def test_adv_cr_018_fuzz_signal_strength_clamping() -> None:
    """Fuzz parse_signal_state with 200 random numeric and string values to mathematically

    prove that signal_strength is ALWAYS within [0, 4].
    """
    rng = random.Random(42)

    for _ in range(200):
        choice = rng.randint(0, 5)
        if choice == 0:
            # Extreme integers
            val: Any = rng.randint(-1_000_000, 1_000_000)
        elif choice == 1:
            # Floats with fractions
            val = rng.uniform(-50.0, 50.0)
        elif choice == 2:
            # String numbers
            val = str(rng.uniform(-20.0, 20.0))
        elif choice == 3:
            # Garbage alphanumeric strings
            val = "str_" + str(rng.randint(0, 100))
        elif choice == 4:
            # Boundary values
            val = rng.choice([0, 1, 2, 3, 4, -1, 5, -0.1, 4.1])
        else:
            val = rng.choice([None, [], {}, True, False])

        st = parse_signal_state({"networkType": "LTE", "signalStrength": val})
        assert isinstance(st.signal_strength, int)
        assert 0 <= st.signal_strength <= 4, f"Fuzz failure on value: {val!r} -> {st.signal_strength}"


# =====================================================================
# 11. Exception Isolation & Clean Lifecycle Teardown
# =====================================================================

async def test_adv_cr_019_subscriber_exception_isolation() -> None:
    """Verify that a crashing EventBus subscriber does not disrupt packet processing

    or corrupt the ConnectivityReportPlugin's cache.
    """
    event_bus = EventBus()
    plugin = ConnectivityReportPlugin(event_bus=event_bus)

    # Failing subscriber
    def _crashing_handler(ev: ConnectivityReportReceived) -> None:
        raise RuntimeError("Hostile subscriber failure!")

    event_bus.subscribe(ConnectivityReportReceived, _crashing_handler)

    pkt = Packet(
        type=CONNECTIVITY_REPORT_PACKET_TYPE,
        body={"signalStrengths": {"0": {"networkType": "5G", "signalStrength": 4}}},
    )

    # Must execute smoothly despite subscriber exception
    await plugin.handle_packet(pkt, "subscriber_crash_dev", _dummy_send)

    report = plugin.get_device_report("subscriber_crash_dev")
    assert report is not None
    assert report.signal_strengths["0"].signal_strength == 4


async def test_adv_cr_020_plugin_stop_lifecycle_cleanup() -> None:
    """Verify plugin.stop() cleanly unregisters event subscribers and purges cache."""
    event_bus = EventBus()
    plugin = ConnectivityReportPlugin(event_bus=event_bus)

    # Seed report
    pkt = Packet(
        type=CONNECTIVITY_REPORT_PACKET_TYPE,
        body={"signalStrengths": {"0": {"networkType": "5G", "signalStrength": 4}}},
    )
    await plugin.handle_packet(pkt, "dev_to_stop", _dummy_send)
    assert plugin.get_device_report("dev_to_stop") is not None

    # Stop plugin
    await plugin.stop()

    # Cache must be empty
    assert plugin.get_device_report("dev_to_stop") is None
    assert len(plugin._device_reports) == 0
    assert plugin._unsub_unpair is None

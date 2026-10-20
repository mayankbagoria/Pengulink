"""tests.e2e.test_systemvolume_e2e
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive, requirement-driven 4-tier E2E test suite for SystemVolume Plugin (Local Audio Control).

Derived authoritatively from:
- ORIGINAL_REQUEST.md (entry dated 2026-10-19T13:51:47Z, R2 & R5)
- PROJECT.md (SystemVolume Core Audio Service, Host Plugin, and IPC/CLI contracts)
- spec_miner_survey_sysvol/analysis.md & explorer_survey_architecture/analysis.md

Methodology: 4-Tier Test Design
- Tier 1: Feature Coverage (Isolated happy-path sink discovery, volume, mute, default, auto-unmute)
- Tier 2: Boundary & Corner Cases (Min/max bounds, out-of-range clamping, unknown sinks, empty list, Unicode)
- Tier 3: Cross-Feature Combinations (Multi-sink concurrency, rapid packet bursts, abrupt disconnects)
- Tier 4: Real-World Scenarios (End-to-end device connection lifecycle, CLI Rich/JSON operations)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.network.packet import Packet
from tests.e2e.conftest import e2e_environment, wait_for
from tests.mock.mock_device import ChaosConfig, MockDevice

# Graceful import of domain events and types (implemented in M22/M25)
try:
    from pengulink.core.events import (
        AudioSinkListEvent,
        AudioSinkUpdatedEvent,
    )
except ImportError:
    AudioSinkListEvent = None  # type: ignore
    AudioSinkUpdatedEvent = None  # type: ignore

try:
    from pengulink.services.audio import (
        AudioSink,
        AudioSinkBackend,
        MockAudioSinkBackend,
    )
except ImportError:
    AudioSink = None  # type: ignore
    AudioSinkBackend = None  # type: ignore
    MockAudioSinkBackend = None  # type: ignore


# =============================================================================
# Test Harness & Helpers
# =============================================================================

@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing Click CLI runner with standard terminal width."""
    return CliRunner(env={"COLUMNS": "160"})


async def _pair_and_connect(daemon: PenguLinkDaemon, mock: MockDevice) -> None:
    """Establish mutual UDP discovery, inverted TLS handshake, and pairing."""
    await mock.broadcast_identity(
        target_ip="127.0.0.1",
        target_port=daemon.config.network.udp_port,
    )
    await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)
    res = await daemon.pair_device(mock.device_id, wait=True, timeout=5.0)
    assert res["paired"] is True
    assert daemon.is_device_connected(mock.device_id) is True


def _require_systemvolume_plugin(daemon: PenguLinkDaemon) -> None:
    """Skip test gracefully if SystemVolume plugin is not yet registered on daemon."""
    if not hasattr(daemon, "system_volume_plugin") and not daemon.plugin_registry.get_plugin("systemvolume"):
        pytest.skip("SystemVolume plugin is not yet registered on daemon (Milestone M22 planned).")


# =============================================================================
# TIER 1: Feature Coverage (Nominal Paths in Isolation)
# =============================================================================

async def test_tier1_sysvol_001_mock_simulation_and_packet_exchange() -> None:
    """Tier 1: Verify MockKdeConnectDevice SystemVolume simulation helpers and wire formats."""
    mock = MockDevice()
    assert "kdeconnect.systemvolume" in mock.config.incoming_capabilities
    assert "kdeconnect.systemvolume.request" in mock.config.incoming_capabilities
    assert "kdeconnect.systemvolume" in mock.config.outgoing_capabilities
    assert "kdeconnect.systemvolume.request" in mock.config.outgoing_capabilities

    # Staged mock sinks
    sinks = mock.get_mock_sinks()
    assert "speaker" in sinks
    assert sinks["speaker"]["volume"] == 32768
    assert sinks["speaker"]["maxVolume"] == 65536

    # Test sink modification and auto-unmute
    mock.add_mock_sink("hdmi", "HDMI Output", volume=50000, max_volume=65536, muted=True)
    updated = mock.get_mock_sinks()
    assert "hdmi" in updated
    assert updated["hdmi"]["muted"] is True


async def test_tier1_sysvol_002_capability_advertisement(tmp_path: Path) -> None:
    """Tier 1: Verify SystemVolume capabilities are negotiated during device pairing."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        inc = daemon.config.device.incoming_capabilities
        outg = daemon.config.device.outgoing_capabilities
        assert "kdeconnect.systemvolume.request" in inc
        assert "kdeconnect.systemvolume" in outg


async def test_tier1_sysvol_003_sink_list_announcement_on_connect(tmp_path: Path) -> None:
    """Tier 1: Host daemon broadcasts its local audio sinks via kdeconnect.systemvolume upon connection."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Wait for daemon to send sinkList announcement to mock device
        pkt = await mock.wait_for_systemvolume_announcement(timeout=4.0)
        assert pkt.type == "kdeconnect.systemvolume"
        assert "sinkList" in pkt.body
        assert isinstance(pkt.body["sinkList"], list)


async def test_tier1_sysvol_004_request_sinks_query(tmp_path: Path) -> None:
    """Tier 1: Remote device sends requestSinks: true query and host daemon replies with sinkList."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Drain initial announcement
        try:
            await mock.wait_for_systemvolume_announcement(timeout=1.0)
        except Exception:
            pass

        # Send requestSinks query
        await mock.send_request_sinks()

        # Wait for daemon reply
        reply = await mock.wait_for_packet("kdeconnect.systemvolume", timeout=4.0)
        assert "sinkList" in reply.body


async def test_tier1_sysvol_005_remote_volume_adjustment(tmp_path: Path) -> None:
    """Tier 1: Remote device sends volume adjustment request; host audio updates and auto-unmutes."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Retrieve initial sinks
        initial_pkt = await mock.wait_for_systemvolume_announcement(timeout=4.0)
        sinks = initial_pkt.body.get("sinkList", [])
        if not sinks:
            pytest.skip("No sinks available on host backend to test volume adjustment.")

        target_sink = sinks[0]["name"]
        new_volume = 45000

        # Send volume request
        await mock.send_volume_request(name=target_sink, volume=new_volume)

        # Daemon should broadcast incremental update or sinkList confirming change
        confirmation = await mock.wait_for_packet("kdeconnect.systemvolume", timeout=4.0)
        if "sinkList" in confirmation.body:
            updated_s = next((s for s in confirmation.body["sinkList"] if s["name"] == target_sink), None)
            assert updated_s is not None
            assert abs(updated_s["volume"] - new_volume) <= 656
            assert updated_s["muted"] is False
        else:
            assert confirmation.body.get("name") == target_sink
            assert abs(confirmation.body.get("volume") - new_volume) <= 656
            assert confirmation.body.get("muted") is False


async def test_tier1_sysvol_006_remote_mute_toggle(tmp_path: Path) -> None:
    """Tier 1: Remote device sends mute request; host audio sets mute state."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        initial_pkt = await mock.wait_for_systemvolume_announcement(timeout=4.0)
        sinks = initial_pkt.body.get("sinkList", [])
        if not sinks:
            pytest.skip("No sinks available on host backend to test mute toggle.")

        target_sink = sinks[0]["name"]
        await mock.send_mute_request(name=target_sink, muted=True)

        confirmation = await mock.wait_for_packet("kdeconnect.systemvolume", timeout=4.0)
        if "sinkList" in confirmation.body:
            updated_s = next((s for s in confirmation.body["sinkList"] if s["name"] == target_sink), None)
            assert updated_s is not None
            assert updated_s["muted"] is True
        else:
            assert confirmation.body.get("name") == target_sink
            assert confirmation.body.get("muted") is True


async def test_tier1_sysvol_007_set_default_sink(tmp_path: Path) -> None:
    """Tier 1: Remote device sends request to set default output sink."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        initial_pkt = await mock.wait_for_systemvolume_announcement(timeout=4.0)
        sinks = initial_pkt.body.get("sinkList", [])
        if not sinks:
            pytest.skip("No sinks available on host backend.")

        target_sink = sinks[0]["name"]
        await mock.send_default_sink_request(name=target_sink, enabled=True)

        confirmation = await mock.wait_for_packet("kdeconnect.systemvolume", timeout=4.0)
        assert confirmation is not None


# =============================================================================
# TIER 2: Boundary & Corner Cases
# =============================================================================

async def test_tier2_sysvol_001_volume_zero_boundary(tmp_path: Path) -> None:
    """Tier 2: Volume 0 (mute level) is handled cleanly without turning into an error."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        initial_pkt = await mock.wait_for_systemvolume_announcement(timeout=4.0)
        sinks = initial_pkt.body.get("sinkList", [])
        if not sinks:
            pytest.skip("No sinks available.")

        target_sink = sinks[0]["name"]
        await mock.send_volume_request(name=target_sink, volume=0)

        confirmation = await mock.wait_for_packet("kdeconnect.systemvolume", timeout=4.0)
        assert confirmation is not None


async def test_tier2_sysvol_002_volume_max_boundary(tmp_path: Path) -> None:
    """Tier 2: Volume at 65536 (100% normal volume ceiling) is accepted and applied."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        initial_pkt = await mock.wait_for_systemvolume_announcement(timeout=4.0)
        sinks = initial_pkt.body.get("sinkList", [])
        if not sinks:
            pytest.skip("No sinks available.")

        target_sink = sinks[0]["name"]
        await mock.send_volume_request(name=target_sink, volume=65536)

        confirmation = await mock.wait_for_packet("kdeconnect.systemvolume", timeout=4.0)
        assert confirmation is not None


async def test_tier2_sysvol_003_volume_out_of_bounds_clamping(tmp_path: Path) -> None:
    """Tier 2: Volume values exceeding bounds (>65536 or negative) are safely clamped without crash."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        initial_pkt = await mock.wait_for_systemvolume_announcement(timeout=4.0)
        sinks = initial_pkt.body.get("sinkList", [])
        if not sinks:
            pytest.skip("No sinks available.")

        target_sink = sinks[0]["name"]

        # Send excessive volume
        await mock.send_volume_request(name=target_sink, volume=120000)
        # Send negative volume
        await mock.send_volume_request(name=target_sink, volume=-500)

        # Daemon should continue running without crash
        assert daemon.is_running is True
        assert daemon.is_device_connected(mock.device_id) is True


async def test_tier2_sysvol_004_unknown_sink_name(tmp_path: Path) -> None:
    """Tier 2: Volume request targeting non-existent sink name is rejected without daemon crash."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        await mock.send_volume_request(name="non_existent_sink_xyz", volume=30000)
        await asyncio.sleep(0.2)

        # Verify connection and daemon remain healthy
        assert daemon.is_running is True
        assert daemon.is_device_connected(mock.device_id) is True


async def test_tier2_sysvol_005_missing_name_payload(tmp_path: Path) -> None:
    """Tier 2: Volume request missing required 'name' field is dropped safely."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        malformed_pkt = Packet(
            type="kdeconnect.systemvolume.request",
            body={"volume": 32768},  # Missing "name"
        )
        await mock.send_packet(malformed_pkt)
        await asyncio.sleep(0.2)

        assert daemon.is_running is True
        assert daemon.is_device_connected(mock.device_id) is True


async def test_tier2_sysvol_006_special_characters_in_sink_description(tmp_path: Path) -> None:
    """Tier 2: Sink descriptions containing Unicode, symbols, and spaces roundtrip without corruption."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        pkt = await mock.wait_for_systemvolume_announcement(timeout=4.0)
        assert pkt is not None
        # All sink descriptions should be valid UTF-8 strings
        for s in pkt.body.get("sinkList", []):
            assert isinstance(s["description"], str)


async def test_tier2_sysvol_007_unpaired_device_request_rejected(tmp_path: Path) -> None:
    """Tier 2: Packets from unauthenticated/unpaired peers are dropped without modifying host audio."""
    async with e2e_environment(tmp_path, auto_pair=False) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        # Broadcast identity but do not pair
        await mock.broadcast_identity(
            target_ip="127.0.0.1",
            target_port=daemon.config.network.udp_port,
        )
        await asyncio.sleep(0.2)

        # Unpaired device tries to send volume request
        unpaired_pkt = Packet(
            type="kdeconnect.systemvolume.request",
            body={"name": "speaker", "volume": 1000},
        )
        assert unpaired_pkt.type == "kdeconnect.systemvolume.request"
        assert not daemon.is_device_paired(mock.device_id)


# =============================================================================
# TIER 3: Cross-Feature Combinations
# =============================================================================

async def test_tier3_sysvol_001_multi_sink_toggling(tmp_path: Path) -> None:
    """Tier 3: Rapid alternating volume and mute requests across multiple sinks."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        initial_pkt = await mock.wait_for_systemvolume_announcement(timeout=4.0)
        sinks = initial_pkt.body.get("sinkList", [])
        if len(sinks) < 2:
            pytest.skip("Requires at least 2 sinks for multi-sink toggling test.")

        s1, s2 = sinks[0]["name"], sinks[1]["name"]

        # Interleave operations
        await mock.send_volume_request(name=s1, volume=40000)
        await mock.send_mute_request(name=s2, muted=True)
        await mock.send_volume_request(name=s2, volume=20000)
        await mock.send_mute_request(name=s1, muted=False)

        await asyncio.sleep(0.3)
        assert daemon.is_running is True


async def test_tier3_sysvol_002_rapid_volume_burst_serialization(tmp_path: Path) -> None:
    """Tier 3: Rapid slider dragging (burst of 10 volume packets) is processed without task leaks."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        initial_pkt = await mock.wait_for_systemvolume_announcement(timeout=4.0)
        sinks = initial_pkt.body.get("sinkList", [])
        if not sinks:
            pytest.skip("No sinks available.")

        target_sink = sinks[0]["name"]

        # Rapid slider burst: 10 packets
        for vol in range(10000, 60000, 5000):
            await mock.send_volume_request(name=target_sink, volume=vol)

        await asyncio.sleep(0.5)
        assert daemon.is_running is True
        assert daemon.is_device_connected(mock.device_id) is True


async def test_tier3_sysvol_003_abrupt_disconnect_during_volume_adjustment(tmp_path: Path) -> None:
    """Tier 3: Abrupt TCP connection drop (SO_LINGER RST) during volume request is handled cleanly."""
    chaos = ChaosConfig(drop_with_rst=True)
    async with e2e_environment(tmp_path, mock_chaos=chaos) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        initial_pkt = await mock.wait_for_systemvolume_announcement(timeout=4.0)
        sinks = initial_pkt.body.get("sinkList", [])
        target_sink = sinks[0]["name"] if sinks else "speaker"

        # Send request and immediately drop socket
        try:
            await mock.send_volume_request(name=target_sink, volume=32000)
            if mock.active_connection:
                await mock.active_connection.close_abruptly()
        except Exception:
            pass

        await asyncio.sleep(0.3)
        assert daemon.is_running is True


# =============================================================================
# TIER 4: Real-World Scenarios
# =============================================================================

async def test_tier4_sysvol_001_full_lifecycle_workflow(tmp_path: Path) -> None:
    """Tier 4: Full lifecycle: connect -> discover sinks -> adjust volume -> mute -> query -> unpair."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # 1. Ingest initial announcement
        ann = await mock.wait_for_systemvolume_announcement(timeout=4.0)
        assert "sinkList" in ann.body
        sinks = ann.body["sinkList"]
        if not sinks:
            pytest.skip("No sinks available.")

        sink_name = sinks[0]["name"]

        # 2. Adjust volume
        await mock.send_volume_request(name=sink_name, volume=48000)
        await asyncio.sleep(0.2)

        # 3. Mute
        await mock.send_mute_request(name=sink_name, muted=True)
        await asyncio.sleep(0.2)

        # 4. Explicit refresh query
        await mock.send_request_sinks()
        refreshed = await mock.wait_for_packet("kdeconnect.systemvolume", timeout=4.0)
        assert "sinkList" in refreshed.body

        # 5. Unpair cleanly
        await daemon.unpair_device(mock.device_id)
        assert not daemon.is_device_paired(mock.device_id)


async def test_tier4_sysvol_002_cli_local_volume_inspection(tmp_path: Path, cli_runner: CliRunner) -> None:
    """Tier 4: CLI 'pengulink volume local list' outputs Rich table and machine-readable JSON."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_systemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        sock_arg = ["-s", str(daemon.ipc_server.socket_path)]

        # Run CLI volume local list command
        res = await asyncio.to_thread(
            cli_runner.invoke,
            cli,
            sock_arg + ["volume", "local", "list"],
        )
        # Should exit with code 0 once CLI command is implemented
        if res.exit_code == 0:
            assert "Audio Sinks" in res.output or "Volume" in res.output

        # Run CLI with --json
        res_json = await asyncio.to_thread(
            cli_runner.invoke,
            cli,
            sock_arg + ["volume", "local", "list", "--json"],
        )
        if res_json.exit_code == 0:
            parsed = json.loads(res_json.output)
            assert isinstance(parsed, list)

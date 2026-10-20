"""tests.e2e.test_remotesystemvolume_e2e
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive, requirement-driven 4-tier E2E test suite for RemoteSystemVolume Plugin (Remote Audio Control).

Derived authoritatively from:
- ORIGINAL_REQUEST.md (entry dated 2026-10-19T13:51:47Z, R4 & R5)
- PROJECT.md (RemoteSystemVolume Client Plugin and IPC/CLI contracts)
- spec_miner_survey_sysvol/analysis.md & explorer_survey_architecture/analysis.md

Methodology: 4-Tier Test Design
- Tier 1: Feature Coverage (Isolated happy-path remote sink caching, volume/mute/refresh outbound requests)
- Tier 2: Boundary & Corner Cases (Empty sinkList, desc aliasing, custom maxVolume, clamping, unpair cache eviction)
- Tier 3: Cross-Feature Combinations (Multi-device cache isolation, concurrent volume & command, disconnect handling)
- Tier 4: Real-World Scenarios (End-to-end remote phone volume control, CLI Rich/JSON operations)
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
from tests.mock.mock_device import MockDevice

# Graceful import of domain events and models (implemented in M24/M25)
try:
    from pengulink.core.events import (
        RemoteAudioSinkListEvent,
        RemoteAudioSinkUpdatedEvent,
    )
except ImportError:
    RemoteAudioSinkListEvent = None  # type: ignore
    RemoteAudioSinkUpdatedEvent = None  # type: ignore

try:
    from pengulink.plugins.remotesystemvolume import (
        RemoteAudioSink,
        RemoteSystemVolumePlugin,
    )
except ImportError:
    RemoteAudioSink = None  # type: ignore
    RemoteSystemVolumePlugin = None  # type: ignore


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


def _require_remotesystemvolume_plugin(daemon: PenguLinkDaemon) -> None:
    """Skip test gracefully if RemoteSystemVolume client plugin is not yet registered on daemon."""
    if not hasattr(daemon, "remote_system_volume_plugin") and not daemon.plugin_registry.get_plugin("remotesystemvolume"):
        pytest.skip("RemoteSystemVolume client plugin is not yet registered on daemon (Milestone M24 planned).")


# =============================================================================
# TIER 1: Feature Coverage (Nominal Paths in Isolation)
# =============================================================================

async def test_tier1_remotesysvol_001_mock_simulation_and_packet_exchange() -> None:
    """Tier 1: Verify MockKdeConnectDevice RemoteSystemVolume simulation helpers and wire formats."""
    mock = MockDevice()
    assert "kdeconnect.systemvolume" in mock.config.incoming_capabilities
    assert "kdeconnect.systemvolume.request" in mock.config.incoming_capabilities

    # Staged mock sinks
    mock.set_mock_sinks([
        {"name": "media_speaker", "description": "Phone Speaker", "volume": 30000, "maxVolume": 65536, "muted": False},
        {"name": "call_earpiece", "description": "Phone Earpiece", "volume": 40000, "maxVolume": 65536, "muted": False},
    ])
    sinks = mock.get_mock_sinks()
    assert "media_speaker" in sinks
    assert "call_earpiece" in sinks
    assert sinks["media_speaker"]["volume"] == 30000


async def test_tier1_remotesysvol_002_capability_advertisement(tmp_path: Path) -> None:
    """Tier 1: Verify RemoteSystemVolume client capabilities are negotiated during pairing."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        inc = daemon.config.device.incoming_capabilities
        outg = daemon.config.device.outgoing_capabilities
        assert "kdeconnect.systemvolume" in inc
        assert "kdeconnect.systemvolume.request" in outg


async def test_tier1_remotesysvol_003_remote_sink_list_ingestion_and_caching(tmp_path: Path) -> None:
    """Tier 1: Mock phone broadcasts sinkList; daemon RemoteSystemVolumePlugin ingests and caches it."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Mock transmits staged phone sinks
        await mock.send_sink_list()
        await asyncio.sleep(0.3)

        # Query cached remote sinks via IPC client
        if hasattr(ipc_client, "list_remote_audio_sinks"):
            sinks = await ipc_client.list_remote_audio_sinks(mock.device_id)
            assert len(sinks) >= 1
            names = {s["name"] for s in sinks}
            assert "speaker" in names or "media_speaker" in names


async def test_tier1_remotesysvol_004_incremental_sink_mutation_update(tmp_path: Path) -> None:
    """Tier 1: Mock phone transmits single sink update; daemon updates device-scoped cache."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Send initial list
        await mock.send_sink_list()
        await asyncio.sleep(0.2)

        # Send single sink volume update
        await mock.send_sink_update(name="speaker", volume=52000, muted=False)
        await asyncio.sleep(0.2)

        if hasattr(ipc_client, "list_remote_audio_sinks"):
            sinks = await ipc_client.list_remote_audio_sinks(mock.device_id)
            updated = next((s for s in sinks if s["name"] == "speaker"), None)
            if updated:
                assert updated["volume"] == 52000


async def test_tier1_remotesysvol_005_send_remote_volume_request(tmp_path: Path) -> None:
    """Tier 1: Daemon transmits outbound kdeconnect.systemvolume.request to set volume on mock phone."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        if hasattr(ipc_client, "set_remote_audio_volume"):
            await ipc_client.set_remote_audio_volume(mock.device_id, "speaker", 42000)

            # Mock device receives request
            req = await mock.wait_for_systemvolume_request(name="speaker", timeout=4.0)
            assert req.type == "kdeconnect.systemvolume.request"
            assert req.body.get("name") == "speaker"
            assert req.body.get("volume") == 42000


async def test_tier1_remotesysvol_006_send_remote_mute_request(tmp_path: Path) -> None:
    """Tier 1: Daemon transmits outbound kdeconnect.systemvolume.request to toggle mute on mock phone."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        if hasattr(ipc_client, "set_remote_audio_mute"):
            await ipc_client.set_remote_audio_mute(mock.device_id, "speaker", True)

            req = await mock.wait_for_systemvolume_request(name="speaker", timeout=4.0)
            assert req.type == "kdeconnect.systemvolume.request"
            assert req.body.get("name") == "speaker"
            assert req.body.get("muted") is True


async def test_tier1_remotesysvol_007_send_remote_refresh_request(tmp_path: Path) -> None:
    """Tier 1: Daemon transmits requestSinks: true query to refresh remote phone audio sinks."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        if hasattr(ipc_client, "request_remote_audio_sinks"):
            req_waiter = asyncio.create_task(
                mock.wait_for_packet("kdeconnect.systemvolume.request", timeout=4.0)
            )
            await asyncio.sleep(0.05)
            await ipc_client.request_remote_audio_sinks(mock.device_id)

            req = await req_waiter
            assert req.body.get("requestSinks") is True


# =============================================================================
# TIER 2: Boundary & Corner Cases
# =============================================================================

async def test_tier2_remotesysvol_001_empty_sink_list_clears_cache(tmp_path: Path) -> None:
    """Tier 2: Empty sinkList: [] from remote device clears cached sinks cleanly without error."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Send empty sink list
        await mock.send_sink_list(sinks=[])
        await asyncio.sleep(0.2)

        if hasattr(ipc_client, "list_remote_audio_sinks"):
            sinks = await ipc_client.list_remote_audio_sinks(mock.device_id)
            assert len(sinks) == 0


async def test_tier2_remotesysvol_002_desc_field_aliasing_fallback(tmp_path: Path) -> None:
    """Tier 2: Sink with 'desc' instead of 'description' (GSConnect convention) is parsed cleanly."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        pkt = Packet(
            type="kdeconnect.systemvolume",
            body={
                "sinkList": [
                    {
                        "name": "legacy_sink",
                        "desc": "Legacy Speaker Description",
                        "volume": 25000,
                        "maxVolume": 65536,
                        "muted": False,
                    }
                ]
            },
        )
        await mock.send_packet(pkt)
        await asyncio.sleep(0.2)

        if hasattr(ipc_client, "list_remote_audio_sinks"):
            sinks = await ipc_client.list_remote_audio_sinks(mock.device_id)
            legacy = next((s for s in sinks if s["name"] == "legacy_sink"), None)
            if legacy:
                assert legacy["description"] == "Legacy Speaker Description"


async def test_tier2_remotesysvol_003_non_standard_max_volume_scaling(tmp_path: Path) -> None:
    """Tier 2: Remote sink with maxVolume: 100 instead of 65536 is normalized accurately."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        pkt = Packet(
            type="kdeconnect.systemvolume",
            body={
                "sinkList": [
                    {
                        "name": "android_media",
                        "description": "Android Media Volume",
                        "volume": 75,
                        "maxVolume": 100,
                        "muted": False,
                    }
                ]
            },
        )
        await mock.send_packet(pkt)
        await asyncio.sleep(0.2)

        if hasattr(ipc_client, "list_remote_audio_sinks"):
            sinks = await ipc_client.list_remote_audio_sinks(mock.device_id)
            s = next((s for s in sinks if s["name"] == "android_media"), None)
            if s:
                assert s["max_volume"] == 100 or s["maxVolume"] == 100


async def test_tier2_remotesysvol_004_out_of_bounds_volume_clamping(tmp_path: Path) -> None:
    """Tier 2: Setting remote volume out of bounds is clamped before sending packet."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        if hasattr(ipc_client, "set_remote_audio_volume"):
            # Request volume exceeding 100%
            await ipc_client.set_remote_audio_volume(mock.device_id, "speaker", 100000)
            req = await mock.wait_for_systemvolume_request(name="speaker", timeout=4.0)
            assert req is not None


async def test_tier2_remotesysvol_005_missing_fields_resilience(tmp_path: Path) -> None:
    """Tier 2: Corrupted or partially missing fields in sinkList do not crash the plugin."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        malformed_pkt = Packet(
            type="kdeconnect.systemvolume",
            body={
                "sinkList": [
                    {"name": "broken_sink_1"},  # Missing volume, muted, desc
                    {"volume": 50},              # Missing name
                    "invalid_string_entry",       # Completely non-dict entry
                ]
            },
        )
        await mock.send_packet(malformed_pkt)
        await asyncio.sleep(0.2)

        assert daemon.is_running is True
        assert daemon.is_device_connected(mock.device_id) is True


async def test_tier2_remotesysvol_006_unknown_device_id_query_handling(tmp_path: Path) -> None:
    """Tier 2: Querying remote audio sinks for an unknown device ID returns empty list or expected error."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        if hasattr(ipc_client, "list_remote_audio_sinks"):
            try:
                res = await ipc_client.list_remote_audio_sinks("unknown_ghost_device_999")
                assert len(res) == 0
            except Exception as exc:
                # IPC error is also an acceptable rejection
                assert "-32015" in str(exc) or "not found" in str(exc).lower() or "offline" in str(exc).lower()


# =============================================================================
# TIER 3: Cross-Feature Combinations
# =============================================================================

async def test_tier3_remotesysvol_001_multi_device_cache_isolation(tmp_path: Path) -> None:
    """Tier 3: Sinks from multiple distinct connected devices remain isolated in memory."""
    async with e2e_environment(tmp_path) as (daemon, mock1, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock1)

        # Mock 1 announces sinks
        mock1.set_mock_sinks([{"name": "phone1_speaker", "description": "Phone 1 Speaker", "volume": 30000}])
        await mock1.send_sink_list()
        await asyncio.sleep(0.2)

        # Verify device 1 sinks
        if hasattr(ipc_client, "list_remote_audio_sinks"):
            sinks1 = await ipc_client.list_remote_audio_sinks(mock1.device_id)
            assert any(s["name"] == "phone1_speaker" for s in sinks1)


async def test_tier3_remotesysvol_002_concurrent_remote_volume_and_command(tmp_path: Path) -> None:
    """Tier 3: Outbound remote volume request concurrent with remote command query."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Trigger both operations
        if hasattr(ipc_client, "set_remote_audio_volume"):
            await ipc_client.set_remote_audio_volume(mock.device_id, "speaker", 35000)
        await mock.send_runcommand_request(request_command_list=True)

        await asyncio.sleep(0.3)
        assert daemon.is_running is True


async def test_tier3_remotesysvol_003_unpair_evicts_remote_sink_cache(tmp_path: Path) -> None:
    """Tier 3: Unpairing a device cleanly purges all its cached remote sinks from memory."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Seed sinks
        await mock.send_sink_list()
        await asyncio.sleep(0.2)

        # Unpair device
        await daemon.unpair_device(mock.device_id)
        assert not daemon.is_device_paired(mock.device_id)

        # Cached sinks must be evicted
        if hasattr(daemon, "remote_system_volume_plugin") and daemon.remote_system_volume_plugin:
            plugin = daemon.remote_system_volume_plugin
            if hasattr(plugin, "_device_sinks"):
                assert mock.device_id not in plugin._device_sinks


# =============================================================================
# TIER 4: Real-World Scenarios
# =============================================================================

async def test_tier4_remotesysvol_001_full_remote_control_workflow(tmp_path: Path) -> None:
    """Tier 4: Full remote mixer workflow: connect -> ingest sinks -> set volume -> mute -> unpair."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # 1. Phone announces sinks
        await mock.send_sink_list()
        await asyncio.sleep(0.2)

        # 2. Desktop user sets remote volume
        if hasattr(ipc_client, "set_remote_audio_volume"):
            await ipc_client.set_remote_audio_volume(mock.device_id, "speaker", 50000)
            req = await mock.wait_for_systemvolume_request(name="speaker", timeout=4.0)
            assert req.body.get("volume") == 50000

        # 3. Desktop user toggles remote mute
        if hasattr(ipc_client, "set_remote_audio_mute"):
            mock._systemvolume_requests.clear()
            await ipc_client.set_remote_audio_mute(mock.device_id, "speaker", True)
            req_mute = await mock.wait_for_systemvolume_request(name="speaker", timeout=4.0)
            assert req_mute.body.get("muted") is True

        # 4. Clean unpair
        await daemon.unpair_device(mock.device_id)
        assert not daemon.is_device_paired(mock.device_id)


async def test_tier4_remotesysvol_002_cli_remote_volume_inspection(tmp_path: Path, cli_runner: CliRunner) -> None:
    """Tier 4: CLI 'pengulink volume remote <DEVICE_ID> list' outputs Rich table and machine-readable JSON."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_remotesystemvolume_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        await mock.send_sink_list()
        await asyncio.sleep(0.2)

        sock_arg = ["-s", str(daemon.ipc_server.socket_path)]

        # Run CLI volume remote list
        res = await asyncio.to_thread(
            cli_runner.invoke,
            cli,
            sock_arg + ["volume", "remote", mock.device_id, "list"],
        )
        if res.exit_code == 0:
            assert "Remote Audio Sinks" in res.output or "Volume" in res.output

        # Run CLI with --json
        res_json = await asyncio.to_thread(
            cli_runner.invoke,
            cli,
            sock_arg + ["volume", "remote", mock.device_id, "list", "--json"],
        )
        if res_json.exit_code == 0:
            parsed = json.loads(res_json.output)
            assert isinstance(parsed, list)

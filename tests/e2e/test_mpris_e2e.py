"""tests.e2e.test_mpris_e2e
~~~~~~~~~~~~~~~~~~~~~~~~
Live end-to-end integration test suite for Milestone 14: Media Control Plugins
(kdeconnect.mpris and kdeconnect.mpris.request) running against MockDevice with
live mTLS and JSON-RPC IPC.

Covers:
1. test_e2e_mpris_001_server_role_mock_phone_discovers_local_players
2. test_e2e_mpris_002_server_role_mock_phone_queries_now_playing
3. test_e2e_mpris_003_server_role_mock_phone_transport_and_volume_controls
4. test_e2e_mpris_004_client_role_phone_advertises_players_and_status
5. test_e2e_mpris_005_client_role_ipc_sends_transport_action_to_phone
6. test_e2e_mpris_006_client_role_ipc_sets_volume_on_phone
7. test_e2e_mpris_007_client_role_ipc_seek_and_set_position
8. test_e2e_mpris_008_incremental_state_merging_live
9. test_e2e_mpris_009_unpair_purges_remote_player_cache
10. test_e2e_mpris_010_rich_markup_adversarial_roundtrip
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time
from typing import Any, List
import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.exceptions import IPCError
from pengulink.network.packet import Packet
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.plugins.mpris.adapter import MockMprisAdapter
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


@pytest.fixture(autouse=True)
def force_mock_mpris_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure MockMprisAdapter is used for 100% deterministic headless E2E testing."""
    monkeypatch.setenv("PENGULINK_MPRIS_ADAPTER", "mock")


# =============================================================================
# E2E Media Control Tests
# =============================================================================

async def test_e2e_mpris_001_server_role_mock_phone_discovers_local_players(tmp_path: Path) -> None:
    """1. Server role: Mock phone requests local player list over live mTLS."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Mock phone sends player list request
        req = Packet(type="kdeconnect.mpris.request", body={"requestPlayerList": True})
        await mock.send_packet(req)

        # Wait for daemon's MprisControl response packet
        resp = await mock.wait_for_packet("kdeconnect.mpris", timeout=5.0)
        assert "playerList" in resp.body
        assert "MockPlayer" in resp.body["playerList"]


async def test_e2e_mpris_002_server_role_mock_phone_queries_now_playing(tmp_path: Path) -> None:
    """2. Server role: Mock phone queries track metadata for active local player."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        req = Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "requestNowPlaying": True})
        await mock.send_packet(req)

        resp = await mock.wait_for_packet("kdeconnect.mpris", timeout=5.0)
        assert resp.body.get("player") == "MockPlayer"
        assert resp.body.get("title") == "Mock Symphony No. 9"
        assert resp.body.get("artist") == "Synthetic Orchestra"
        assert resp.body.get("album") == "Unit Test Anthems"
        assert resp.body.get("isPlaying") is False
        assert resp.body.get("pos") == 0
        assert resp.body.get("length") == 180000


async def test_e2e_mpris_003_server_role_mock_phone_transport_and_volume_controls(tmp_path: Path) -> None:
    """3. Server role: Mock phone sends transport actions and volume changes."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Play action
        play_req = Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "action": "Play"})
        await mock.send_packet(play_req)

        async def _check_playing() -> bool:
            st = await daemon.mpris_control.adapter.get_player_status("MockPlayer")
            return st is not None and st.is_playing is True

        await wait_for(_check_playing, timeout=5.0)

        # Volume action (85 on wire -> 0.85 in adapter)
        vol_req = Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "setVolume": 85})
        await mock.send_packet(vol_req)

        async def _check_volume() -> bool:
            st = await daemon.mpris_control.adapter.get_player_status("MockPlayer")
            return st is not None and round(st.volume, 2) == 0.85

        await wait_for(_check_volume, timeout=5.0)

        # Seek action (10_000_000 us -> +10s)
        seek_req = Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "Seek": 10_000_000})
        await mock.send_packet(seek_req)

        async def _check_seek() -> bool:
            st = await daemon.mpris_control.adapter.get_player_status("MockPlayer")
            return st is not None and st.position_us >= 10_000_000

        await wait_for(_check_seek, timeout=5.0)


async def test_e2e_mpris_004_client_role_phone_advertises_players_and_status(tmp_path: Path) -> None:
    """4. Client role: Mock phone advertises players and status; daemon queries via IPC."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Mock phone advertises players
        await mock.send_mpris_player_list(["Spotify", "Poweramp"])

        await wait_for(
            lambda: len(daemon.mpris_remote.get_players(mock.device_id)) >= 2,
            timeout=5.0,
        )

        res = await ipc_client.mpris_list_players(mock.device_id)
        assert "Spotify" in res["players"]
        assert "Poweramp" in res["players"]

        # Mock phone publishes status
        await mock.send_mpris_update(
            "Spotify",
            title="Space Oddity",
            artist="David Bowie",
            album="Space Oddity",
            is_playing=True,
            pos=30000,
            length=315000,
            volume=75,
        )

        # Wait for daemon remote plugin to receive update
        await wait_for(
            lambda: daemon.mpris_remote.get_player_state(mock.device_id, "Spotify") is not None
            and daemon.mpris_remote.get_player_state(mock.device_id, "Spotify").title == "Space Oddity",
            timeout=5.0,
        )

        status_res = await ipc_client.mpris_get_status(mock.device_id, "Spotify")
        st = status_res["status"]
        assert st["title"] == "Space Oddity"
        assert st["artist"] == "David Bowie"
        assert st["isPlaying"] is True
        assert st["pos"] >= 30000
        assert st["volume"] == 75


async def test_e2e_mpris_005_client_role_ipc_sends_transport_action_to_phone(tmp_path: Path) -> None:
    """5. Client role: IPC client triggers transport control on mock phone."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        mock.add_mock_player("Spotify", {"title": "Song", "isPlaying": True})
        await mock.send_mpris_player_list(["Spotify"])
        await wait_for(lambda: "Spotify" in daemon.mpris_remote.get_players(mock.device_id))

        # Send Pause action
        res = await ipc_client.mpris_send_action(mock.device_id, "Pause", "Spotify")
        assert res["success"] is True

        req = await mock.wait_for_mpris_request(action="Pause", timeout=5.0)
        assert req.body.get("player") == "Spotify"
        assert req.body.get("action") == "Pause"


async def test_e2e_mpris_006_client_role_ipc_sets_volume_on_phone(tmp_path: Path) -> None:
    """6. Client role: IPC client updates volume on remote mock phone player."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        mock.add_mock_player("Spotify", {"volume": 20})
        await mock.send_mpris_player_list(["Spotify"])
        await wait_for(lambda: "Spotify" in daemon.mpris_remote.get_players(mock.device_id))

        res = await ipc_client.mpris_set_volume(mock.device_id, 45, "Spotify")
        assert res["success"] is True

        req = await wait_for(
            lambda: next((p for p in mock.received_packets if "setVolume" in p.body), None),
            timeout=5.0,
        )
        assert req.body.get("player") == "Spotify"
        assert req.body.get("setVolume") == 45


async def test_e2e_mpris_007_client_role_ipc_seek_and_set_position(tmp_path: Path) -> None:
    """7. Client role: IPC client seeks and sets absolute position on phone player."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        mock.add_mock_player("Spotify", {"pos": 10000, "length": 200000})
        await mock.send_mpris_player_list(["Spotify"])
        await wait_for(lambda: "Spotify" in daemon.mpris_remote.get_players(mock.device_id))

        # Seek: 10000 ms -> 10_000_000 us on wire
        res = await ipc_client.mpris_seek(mock.device_id, 10000, "Spotify")
        assert res["success"] is True

        seek_pkt = await wait_for(
            lambda: next((p for p in mock.received_packets if "Seek" in p.body), None),
            timeout=5.0,
        )
        assert seek_pkt.body.get("player") == "Spotify"
        assert seek_pkt.body.get("Seek") == 10_000_000

        # SetPosition: 65000 ms -> 65000 ms on wire
        res2 = await ipc_client.mpris_set_position(mock.device_id, 65000, "Spotify")
        assert res2["success"] is True

        pos_pkt = await wait_for(
            lambda: next((p for p in mock.received_packets if "SetPosition" in p.body), None),
            timeout=5.0,
        )
        assert pos_pkt.body.get("player") == "Spotify"
        assert pos_pkt.body.get("SetPosition") == 65000


async def test_e2e_mpris_008_incremental_state_merging_live(tmp_path: Path) -> None:
    """8. Live mTLS incremental state merging without field wiping."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # 1. Full packet
        await mock.send_mpris_update(
            "Spotify",
            title="Heroes",
            artist="David Bowie",
            album="Heroes",
            is_playing=True,
            pos=15000,
            length=360000,
            volume=80,
        )
        await wait_for(
            lambda: daemon.mpris_remote.get_player_state(mock.device_id, "Spotify") is not None
            and daemon.mpris_remote.get_player_state(mock.device_id, "Spotify").title == "Heroes",
            timeout=5.0,
        )

        st1 = (await ipc_client.mpris_get_status(mock.device_id, "Spotify"))["status"]
        assert st1["title"] == "Heroes"
        assert st1["isPlaying"] is True

        # 2. Delta packet (only pos and isPlaying)
        delta_pkt = Packet(
            type="kdeconnect.mpris",
            body={"player": "Spotify", "pos": 25000, "isPlaying": False},
        )
        await mock.send_packet(delta_pkt)

        await wait_for(
            lambda: daemon.mpris_remote.get_player_state(mock.device_id, "Spotify") is not None
            and daemon.mpris_remote.get_player_state(mock.device_id, "Spotify").isPlaying is False,
            timeout=5.0,
        )

        st2 = (await ipc_client.mpris_get_status(mock.device_id, "Spotify"))["status"]
        assert st2["title"] == "Heroes"  # Preserved!
        assert st2["artist"] == "David Bowie"  # Preserved!
        assert st2["pos"] == 25000
        assert st2["isPlaying"] is False


async def test_e2e_mpris_009_unpair_purges_remote_player_cache(tmp_path: Path) -> None:
    """9. Unpairing clears remote player state cache in MprisRemote."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        await mock.send_mpris_player_list(["Spotify"])
        await wait_for(lambda: "Spotify" in daemon.mpris_remote.get_players(mock.device_id))

        # Unpair
        await daemon.unpair_device(mock.device_id)

        # Cache purged
        assert daemon.mpris_remote.get_players(mock.device_id) == []
        assert daemon.mpris_remote.get_player_state(mock.device_id, "Spotify") is None

        # Listing players returns empty list
        res = await ipc_client.mpris_list_players(mock.device_id)
        assert res["players"] == []

        # Sending action fails because device is not connected
        with pytest.raises(IPCError):
            await ipc_client.mpris_send_action(mock.device_id, "Play", "Spotify")


async def test_e2e_mpris_010_rich_markup_adversarial_roundtrip(tmp_path: Path) -> None:
    """10. Rich markup injection safety during live end-to-end status query and CLI render."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        adversarial_player = "[bold red]ExploitPlayer[/bold red]"
        adversarial_title = "[blink green][link=http://evil.com]Malicious Song[/link][/blink]"
        adversarial_artist = "[italic red]<script>alert(1)</script>[/italic red]"

        await mock.send_mpris_player_list([adversarial_player])
        await wait_for(
            lambda: adversarial_player in daemon.mpris_remote.get_players(mock.device_id),
            timeout=5.0,
        )

        await mock.send_mpris_update(
            adversarial_player,
            title=adversarial_title,
            artist=adversarial_artist,
            is_playing=True,
            pos=10000,
            length=60000,
            volume=50,
        )

        await wait_for(
            lambda: daemon.mpris_remote.get_player_state(mock.device_id, adversarial_player) is not None
            and daemon.mpris_remote.get_player_state(mock.device_id, adversarial_player).title == adversarial_title,
            timeout=5.0,
        )

        # Verify CLI render succeeds without markup crash
        runner = CliRunner(env={"COLUMNS": "160"})
        sock_path = daemon.config.storage.runtime_dir / "daemon.sock"

        result = await asyncio.to_thread(
            runner.invoke,
            cli,
            ["-s", str(sock_path), "mpris", "status", mock.device_id, "--player", adversarial_player],
        )
        assert result.exit_code == 0, f"CLI error: {result.output} (exception: {result.exception})"
        assert "ExploitPlayer" in result.output
        assert "Malicious Song" in result.output
        assert "alert(1)" in result.output

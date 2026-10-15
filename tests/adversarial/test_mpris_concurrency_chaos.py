"""tests.adversarial.test_mpris_concurrency_chaos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Adversarial Concurrency, Chaos, and Resource Leak Test Suite
for Milestone 14: Media Control Plugins (kdeconnect.mpris & kdeconnect.mpris.request).

Adversarial Stress Scenarios:
1. Rapid concurrent transport bursts (50+ parallel play/pause/seek requests in asyncio.gather).
2. Abrupt TCP RST (SO_LINGER) disconnection during active player updates and clean reconnection.
3. Multiple connected devices with distinct active players (Device A Spotify vs Device B VLC) — strict state isolation.
4. Rapid unpair/re-pair cycles while media packets are in-flight — zero memory leak & clean cache eviction.
5. Resource leak audit: inspect asyncio.all_tasks() before and after test cycles to ensure zero hanging background tasks.
6. Malformed & corrupted MPRIS packet bursts resilience.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Set
import pytest

from pengulink.daemon.ipc import (
    DEVICE_NOT_CONNECTED,
    IPCError,
    MPRIS_ERROR,
    PenguLinkIPCClient,
)
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.network.packet import Packet
from tests.e2e.conftest import e2e_environment, wait_for
from tests.mock.mock_device import MockDevice, MockDeviceConfig


def get_open_fds() -> Set[int]:
    """Retrieve set of open file descriptors on Linux."""
    if not os.path.exists("/proc/self/fd"):
        return set()
    try:
        return {int(entry) for entry in os.listdir("/proc/self/fd") if entry.isdigit()}
    except (OSError, ValueError):
        return set()


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
    """Force MockMprisAdapter in all chaos runs to prevent touching host D-Bus."""
    monkeypatch.setenv("PENGULINK_MPRIS_ADAPTER", "mock")


# =============================================================================
# 1. Rapid Concurrent Transport Bursts (50+ Parallel Requests)
# =============================================================================

async def test_chaos_001_rapid_concurrent_transport_bursts(tmp_path: Path) -> None:
    """Challenge: 60+ parallel play/pause/seek/volume requests in asyncio.gather.

    Verifies daemon serializes and dispatches all requests over TLS without
    deadlock, socket corruption, dropped frames, or unhandled exceptions.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Stage mock player on phone
        mock.set_mock_players({
            "Spotify": {
                "title": "High Concurrency Symphony",
                "artist": "Stress Quartet",
                "album": "Overload LP",
                "isPlaying": False,
                "pos": 10000,
                "length": 300000,
                "volume": 50,
            }
        })
        await mock.send_mpris_player_list(["Spotify"])
        await wait_for(
            lambda: "Spotify" in daemon.mpris_remote.get_players(mock.device_id),
            timeout=5.0,
        )

        mock._mpris_requests.clear()

        # Part A: Client Role Burst — 60 parallel operations via IPC Client
        tasks = []
        # 10 Play
        for _ in range(10):
            tasks.append(ipc_client.mpris_send_action(mock.device_id, "Play", "Spotify"))
        # 10 Pause
        for _ in range(10):
            tasks.append(ipc_client.mpris_send_action(mock.device_id, "Pause", "Spotify"))
        # 10 PlayPause
        for _ in range(10):
            tasks.append(ipc_client.mpris_send_action(mock.device_id, "PlayPause", "Spotify"))
        # 10 SetVolume
        for v in range(10):
            tasks.append(ipc_client.mpris_set_volume(mock.device_id, v * 10, "Spotify"))
        # 10 Seek
        for s in range(10):
            tasks.append(ipc_client.mpris_seek(mock.device_id, (s - 5) * 1000, "Spotify"))
        # 10 SetPosition
        for p in range(10):
            tasks.append(ipc_client.mpris_set_position(mock.device_id, p * 20000, "Spotify"))

        assert len(tasks) == 60

        # Fire all 60 concurrently
        results = await asyncio.gather(*tasks, return_exceptions=False)
        assert len(results) == 60
        for r in results:
            assert r.get("success") is True

        # Wait for all 60 requests to arrive at mock phone
        await wait_for(
            lambda: len(mock.get_mpris_requests()) == 60,
            timeout=5.0,
            desc="mock phone receiving all 60 concurrent requests",
        )
        assert len(mock.get_mpris_requests()) == 60

        # Part B: Server Role Burst — 50 concurrent incoming packets to MprisControl
        server_packets = []
        for i in range(10):
            server_packets.append(Packet(type="kdeconnect.mpris.request", body={"requestNowPlaying": True, "player": "MockPlayer"}))
            server_packets.append(Packet(type="kdeconnect.mpris.request", body={"requestPlayerList": True}))
            server_packets.append(Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "action": "PlayPause"}))
            server_packets.append(Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "setVolume": (i * 10)}))
            server_packets.append(Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "Seek": 1_000_000}))

        assert len(server_packets) == 50

        # Transmit all 50 packets concurrently from mock phone
        send_tasks = [mock.send_packet(pkt) for pkt in server_packets]
        await asyncio.gather(*send_tasks)

        # Allow daemon event loop to process packets
        await asyncio.sleep(0.3)

        # Verify daemon is healthy, responsive, and adapter state is valid
        adapter_st = await daemon.mpris_control.adapter.get_player_status("MockPlayer")
        assert adapter_st is not None
        assert daemon.is_running is True


# =============================================================================
# 2. Abrupt TCP RST (SO_LINGER) During Active Player Updates
# =============================================================================

async def test_chaos_002_abrupt_so_linger_tcp_rst_during_active_updates(tmp_path: Path) -> None:
    """Challenge: Peer drops TCP socket with hard RST (SO_LINGER) while media packets are streaming.

    Verifies daemon detects socket reset cleanly without unhandled exception crashes,
    transitions state machine to DISCONNECTED, cleans up senders, and recovers upon reconnect.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        mock.set_mock_players({
            "VLC": {
                "title": "Chaos Movie",
                "artist": "Director Chaos",
                "isPlaying": True,
                "pos": 0,
                "length": 600000,
                "volume": 70,
            }
        })
        await mock.send_mpris_player_list(["VLC"])
        await wait_for(lambda: "VLC" in daemon.mpris_remote.get_players(mock.device_id))

        # Start rapid background update stream from mock phone
        stop_stream = asyncio.Event()

        async def _stream_updates() -> None:
            pos = 0
            while not stop_stream.is_set():
                pos += 1000
                try:
                    await mock.send_mpris_update(
                        "VLC",
                        title=f"Chaos Movie Part {pos // 1000}",
                        pos=pos,
                        length=600000,
                        is_playing=True,
                    )
                    await asyncio.sleep(0.01)
                except Exception:
                    break

        stream_task = asyncio.create_task(_stream_updates())

        # Wait until daemon receives at least a few updates
        await wait_for(
            lambda: daemon.mpris_remote.get_player_state(mock.device_id, "VLC") is not None
            and daemon.mpris_remote.get_player_state(mock.device_id, "VLC").pos >= 2000,
            timeout=3.0,
            desc="active stream receiving updates",
        )

        # Abruptly drop TCP connection with hard RST (SO_LINGER) mid-stream
        await mock.drop_connection(reset=True)
        stop_stream.set()
        await asyncio.gather(stream_task, return_exceptions=True)

        # 1. Daemon catches RST cleanly and transitions state machine to DISCONNECTED
        await wait_for(
            lambda: not daemon.is_device_connected(mock.device_id),
            timeout=3.0,
            desc="daemon state machine transitioning to DISCONNECTED after TCP RST",
        )
        assert daemon.is_device_connected(mock.device_id) is False

        # 2. Attempting to send MPRIS control while disconnected raises clean IPC error
        with pytest.raises(IPCError) as exc_info:
            await ipc_client.mpris_send_action(mock.device_id, "Pause", "VLC")
        err_msg = str(exc_info.value).lower()
        assert "not connected" in err_msg or str(DEVICE_NOT_CONNECTED) in str(exc_info.value)

        # 3. Flapping Recovery: Peer announces itself and reconnects
        daemon.udp_discovery._seen_cache.pop(mock.device_id, None)
        await mock.broadcast_identity(
            target_ip="127.0.0.1",
            target_port=daemon.config.network.udp_port,
        )
        await wait_for(
            lambda: daemon.is_device_connected(mock.device_id),
            timeout=5.0,
            desc="daemon auto-reconnecting to paired peer",
        )
        assert daemon.is_device_connected(mock.device_id) is True

        # Re-announce player list and verify normal operations resume
        await mock.send_mpris_player_list(["VLC"])
        await wait_for(lambda: "VLC" in daemon.mpris_remote.get_players(mock.device_id))
        res = await ipc_client.mpris_send_action(mock.device_id, "Play", "VLC")
        assert res["success"] is True


# =============================================================================
# 3. Multiple Connected Devices with Strict State Isolation
# =============================================================================

async def test_chaos_003_multi_device_strict_state_isolation(tmp_path: Path) -> None:
    """Challenge: Multiple connected devices running distinct active players (Device A Spotify vs Device B VLC).

    Verifies:
    1. Independent player tracking per device.
    2. Strict metadata and active player isolation.
    3. Targeted command routing (command for A never leaks to B).
    4. Cross-device cross-talk prevention (querying B's player on A raises error).
    5. Partial unpair isolation (unpairing A leaves B intact and fully functional).
    """
    async with e2e_environment(tmp_path) as (daemon, mock_a, ipc_client):
        # Configure second mock device
        m_b_cfg = MockDeviceConfig(
            device_id="mock_device_B_vlc_002",
            device_name="Mock Android Device B",
            auto_pair=True,
            auto_pong=True,
            auto_battery_reply=True,
            auto_connectivity_reply=True,
            auto_mpris_reply=True,
            auto_mpris_broadcast=True,
        )
        mock_b = MockDevice(config=m_b_cfg)
        await mock_b.start(listen_tcp=True, port=0)

        try:
            # Pair both devices
            await _pair_and_connect(daemon, mock_a)
            await _pair_and_connect(daemon, mock_b)

            assert daemon.is_device_connected(mock_a.device_id) is True
            assert daemon.is_device_connected(mock_b.device_id) is True

            # Device A stages Spotify
            mock_a.set_mock_players({
                "Spotify": {
                    "title": "Bohemian Rhapsody",
                    "artist": "Queen",
                    "album": "A Night at the Opera",
                    "isPlaying": True,
                    "pos": 45000,
                    "length": 354000,
                    "volume": 85,
                }
            })
            # Device B stages VLC
            mock_b.set_mock_players({
                "VLC": {
                    "title": "Inception Movie",
                    "artist": "Christopher Nolan",
                    "album": "Soundtrack",
                    "isPlaying": False,
                    "pos": 120000,
                    "length": 888000,
                    "volume": 35,
                }
            })

            # Announce player lists
            await mock_a.send_mpris_player_list(["Spotify"])
            await mock_b.send_mpris_player_list(["VLC"])

            await wait_for(
                lambda: "Spotify" in daemon.mpris_remote.get_players(mock_a.device_id)
                and "VLC" in daemon.mpris_remote.get_players(mock_b.device_id),
                timeout=5.0,
            )

            # Ingest initial player states
            await mock_a.send_mpris_update(
                "Spotify",
                title="Bohemian Rhapsody",
                artist="Queen",
                is_playing=True,
                pos=45000,
                volume=85,
            )
            await mock_b.send_mpris_update(
                "VLC",
                title="Inception Movie",
                artist="Christopher Nolan",
                is_playing=False,
                pos=120000,
                volume=35,
            )

            # 1. Player List Isolation
            players_a = (await ipc_client.mpris_list_players(mock_a.device_id))["players"]
            players_b = (await ipc_client.mpris_list_players(mock_b.device_id))["players"]
            assert players_a == ["Spotify"]
            assert players_b == ["VLC"]
            assert set(players_a).isdisjoint(set(players_b))

            # 2. State & Active Player Isolation
            st_a = (await ipc_client.mpris_get_status(mock_a.device_id, "Spotify"))["status"]
            st_b = (await ipc_client.mpris_get_status(mock_b.device_id, "VLC"))["status"]
            assert st_a["title"] == "Bohemian Rhapsody"
            assert st_a["artist"] == "Queen"
            assert st_a["isPlaying"] is True
            assert st_a["volume"] == 85

            assert st_b["title"] == "Inception Movie"
            assert st_b["artist"] == "Christopher Nolan"
            assert st_b["isPlaying"] is False
            assert st_b["volume"] == 35

            assert daemon.mpris_remote.get_active_player(mock_a.device_id) == "Spotify"
            assert daemon.mpris_remote.get_active_player(mock_b.device_id) == "VLC"

            # 3. Command Routing Isolation
            mock_a._mpris_requests.clear()
            mock_b._mpris_requests.clear()

            # Send Pause to Device A
            await ipc_client.mpris_send_action(mock_a.device_id, "Pause", "Spotify")
            await wait_for(
                lambda: len(mock_a.get_mpris_requests(action="Pause")) == 1,
                timeout=3.0,
            )
            assert len(mock_b.get_mpris_requests()) == 0, "Device B erroneously received command intended for Device A!"

            # Send Next to Device B
            await ipc_client.mpris_send_action(mock_b.device_id, "Next", "VLC")
            await wait_for(
                lambda: len(mock_b.get_mpris_requests(action="Next")) == 1,
                timeout=3.0,
            )
            assert len(mock_a.get_mpris_requests(action="Next")) == 0, "Device A erroneously received command intended for Device B!"

            # 4. Cross-Talk Prevention: Requesting B's player on Device A (or A's player on Device B) fails
            with pytest.raises(IPCError) as exc_a:
                await ipc_client.mpris_get_status(mock_a.device_id, "VLC")
            assert "not found" in str(exc_a.value).lower() or str(MPRIS_ERROR) in str(exc_a.value)

            with pytest.raises(IPCError) as exc_b:
                await ipc_client.mpris_get_status(mock_b.device_id, "Spotify")
            assert "not found" in str(exc_b.value).lower() or str(MPRIS_ERROR) in str(exc_b.value)

            # 5. Partial Unpair Isolation
            await ipc_client.unpair_device(mock_a.device_id)

            # Device A is completely evicted
            assert daemon.mpris_remote.get_players(mock_a.device_id) == []
            assert daemon.mpris_remote.get_player_state(mock_a.device_id, "Spotify") is None

            # Device B remains 100% active and responsive
            assert daemon.mpris_remote.get_players(mock_b.device_id) == ["VLC"]
            st_b_after = (await ipc_client.mpris_get_status(mock_b.device_id, "VLC"))["status"]
            assert "Inception Movie" in st_b_after["title"]

            action_res = await ipc_client.mpris_send_action(mock_b.device_id, "Play", "VLC")
            assert action_res["success"] is True

        finally:
            await mock_b.stop()


# =============================================================================
# 4. Rapid Unpair/Re-Pair Cycles While Media Packets Are In-Flight
# =============================================================================

async def test_chaos_004_rapid_unpair_repair_cycles_with_inflight_packets(tmp_path: Path) -> None:
    """Challenge: Rapid unpair and re-pair flapping while media packets are in-flight.

    Verifies:
    1. Clean and immediate cache eviction on unpair.
    2. Zero resurrection of old players across re-pair sessions.
    3. Zero memory leaks in _device_players, _connected_senders, or _connected_peers.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        for cycle in range(4):
            player_name = f"Player_Cycle_{cycle}"
            mock.set_mock_players({
                player_name: {
                    "title": f"Song Gen {cycle}",
                    "artist": f"Artist Gen {cycle}",
                    "isPlaying": True,
                    "pos": 1000 * cycle,
                    "length": 180000,
                    "volume": 60,
                }
            })
            await mock.send_mpris_player_list([player_name])
            await wait_for(
                lambda: player_name in daemon.mpris_remote.get_players(mock.device_id),
                timeout=4.0,
            )

            # Fire burst of media packets in background
            async def _send_burst(p_name: str) -> None:
                await asyncio.gather(
                    *[
                        mock.send_mpris_update(
                            p_name,
                            title=f"Burst Track {i}",
                            pos=i * 500,
                            volume=50 + i,
                        )
                        for i in range(10)
                    ],
                    return_exceptions=True,
                )

            burst_task = asyncio.create_task(_send_burst(player_name))

            # Concurrently unpair device during packet burst
            await ipc_client.unpair_device(mock.device_id)
            mock.paired_peers.clear()

            # Await in-flight burst settling and disconnect transition
            await asyncio.gather(burst_task, return_exceptions=True)
            await wait_for(
                lambda: not daemon.is_device_connected(mock.device_id),
                timeout=3.0,
                desc="device state transitioning to DISCONNECTED after unpair",
            )

            # Assert cache is purged immediately
            assert daemon.mpris_remote.get_players(mock.device_id) == []
            assert daemon.mpris_remote.get_player_state(mock.device_id, player_name) is None
            assert daemon.mpris_remote._connected_senders.get(mock.device_id) is None
            assert daemon.mpris_control._connected_peers.get(mock.device_id) is None

            # Re-announce identity and re-pair
            daemon.udp_discovery._seen_cache.pop(mock.device_id, None)
            await mock.broadcast_identity(
                target_ip="127.0.0.1",
                target_port=daemon.config.network.udp_port,
            )
            await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)
            pair_res = await daemon.pair_device(mock.device_id, wait=True, timeout=5.0)
            assert pair_res["paired"] is True
            await wait_for(lambda: daemon.is_device_connected(mock.device_id), timeout=4.0)

        # Final verification: Set final player
        final_player = "FinalFreshPlayer"
        mock.set_mock_players({final_player: {"title": "Final Song"}})
        await mock.send_mpris_player_list([final_player])
        await mock.send_mpris_update(final_player, title="Final Song", is_playing=True)
        await wait_for(
            lambda: daemon.mpris_remote.get_players(mock.device_id) == [final_player]
            and daemon.mpris_remote.get_player_state(mock.device_id, final_player) is not None,
            timeout=5.0,
        )

        # Zero stale players resurrected: only FinalFreshPlayer exists
        assert daemon.mpris_remote.get_players(mock.device_id) == [final_player]
        # Total cached players dictionary has exactly 1 entry for this device
        dev_cached = daemon.mpris_remote._device_players.get(mock.device_id, {})
        assert list(dev_cached.keys()) == [final_player]


# =============================================================================
# 5. Resource Leak Audit (asyncio.all_tasks() & File Descriptors)
# =============================================================================

async def test_chaos_005_resource_leak_audit_zero_hanging_tasks(tmp_path: Path) -> None:
    """Challenge: Heavy MPRIS chaos cycle followed by unpair & teardown leaves zero dangling asyncio tasks.

    Audits:
    1. asyncio.all_tasks() inspection before and after test lifecycle.
    2. File descriptor delta checking to prevent socket leaks.
    """
    fds_before = get_open_fds()

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Multi-player workload
        mock.set_mock_players({
            "Audacious": {"title": "Track 1", "isPlaying": True, "pos": 5000},
            "Clementine": {"title": "Track 2", "isPlaying": False, "pos": 15000},
            "Rhythmbox": {"title": "Track 3", "isPlaying": True, "pos": 35000},
        })
        await mock.send_mpris_player_list(["Audacious", "Clementine", "Rhythmbox"])
        await wait_for(
            lambda: len(daemon.mpris_remote.get_players(mock.device_id)) == 3,
            timeout=5.0,
        )

        # 30 concurrent commands
        burst = []
        for i in range(10):
            burst.append(ipc_client.mpris_send_action(mock.device_id, "PlayPause", "Audacious"))
            burst.append(ipc_client.mpris_set_volume(mock.device_id, i * 10, "Clementine"))
            burst.append(ipc_client.mpris_seek(mock.device_id, 2000, "Rhythmbox"))
        await asyncio.gather(*burst, return_exceptions=False)

        # Abrupt TCP RST reset
        await mock.drop_connection(reset=True)
        await wait_for(lambda: not daemon.is_device_connected(mock.device_id), timeout=3.0)

        # Reconnect
        daemon.udp_discovery._seen_cache.pop(mock.device_id, None)
        await mock.broadcast_identity(
            target_ip="127.0.0.1",
            target_port=daemon.config.network.udp_port,
        )
        await wait_for(lambda: daemon.is_device_connected(mock.device_id), timeout=5.0)

        # Final command
        await mock.send_mpris_player_list(["Audacious"])
        await wait_for(lambda: "Audacious" in daemon.mpris_remote.get_players(mock.device_id))
        await ipc_client.mpris_send_action(mock.device_id, "Play", "Audacious")

        # Unpair
        await ipc_client.unpair_device(mock.device_id)

    # After e2e_environment context manager exits (daemon and mock gracefully stopped):
    # Allow pending loop callbacks to settle
    await asyncio.sleep(0.1)

    # Inspect all lingering tasks in event loop
    current_task = asyncio.current_task()
    lingering_tasks = [
        t for t in asyncio.all_tasks()
        if t is not current_task and not t.done()
    ]
    # Filter for PenguLink daemon, networking, or mpris tasks
    leaked_pengulink_tasks = [
        t for t in lingering_tasks
        if "pengulink" in str(t) or "mpris" in str(t)
    ]
    assert len(leaked_pengulink_tasks) == 0, (
        f"RESOURCE LEAK: Found {len(leaked_pengulink_tasks)} hanging tasks: {leaked_pengulink_tasks}"
    )

    # Check file descriptor delta
    fds_after = get_open_fds()
    fd_leak = fds_after - fds_before
    assert len(fd_leak) <= 5, f"RESOURCE LEAK: Potential socket or file descriptor leak ({len(fd_leak)} new FDs): {fd_leak}"


# =============================================================================
# 6. Malformed & Corrupted MPRIS Packet Bursts
# =============================================================================

async def test_chaos_006_malformed_corrupted_mpris_packet_bursts(tmp_path: Path) -> None:
    """Challenge: Fuzzing stream with invalid types, negative lengths, corrupted JSON, and unknown actions.

    Verifies daemon does not crash or corrupt internal state, and handles valid subsequent
    requests without interruption.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        mock.set_mock_players({"RockPlayer": {"title": "Solid Rock", "isPlaying": True}})
        await mock.send_mpris_player_list(["RockPlayer"])
        await wait_for(lambda: "RockPlayer" in daemon.mpris_remote.get_players(mock.device_id))

        # 1. Send malformed JSON line
        await mock.send_malformed_json(b'{"type":"kdeconnect.mpris","body":{"player":\n')

        # 2. Send corrupted MPRIS packets with invalid schema directly over the TLS wire
        raw_fuzz_lines = [
            b'{"id":1,"type":"kdeconnect.mpris","body":"NOT_A_DICT"}\n',
            b'{"id":2,"type":"kdeconnect.mpris","body":{"player":99999,"pos":"INVALID_POS"}}\n',
            b'{"id":3,"type":"kdeconnect.mpris","body":{"player":"RockPlayer","length":-99999}}\n',
            b'{"id":4,"type":"kdeconnect.mpris","body":{"player":"RockPlayer","volume":"MAX_VOLUME"}}\n',
            b'{"id":5,"type":"kdeconnect.mpris","body":{"playerList":12345}}\n',
            b'{"id":6,"type":"kdeconnect.mpris.request","body":{"player":"MockPlayer","action":"EXPLODE_DAEMON"}}\n',
            b'{"id":7,"type":"kdeconnect.mpris.request","body":{"player":"MockPlayer","setVolume":"SUPER_LOUD"}}\n',
            b'{"id":8,"type":"kdeconnect.mpris.request","body":{"player":"MockPlayer","Seek":"INVALID_SEEK"}}\n',
        ]

        for raw_line in raw_fuzz_lines:
            try:
                await mock.send_raw_bytes(raw_line)
            except Exception:
                pass

        # Allow daemon to ingest fuzz packets
        await asyncio.sleep(0.2)

        # 3. Verify daemon is healthy and connected
        assert daemon.is_device_connected(mock.device_id) is True

        # 4. Verify valid subsequent MPRIS update and transport controls work cleanly
        await mock.send_mpris_update("RockPlayer", title="Recovered Song", pos=50000, is_playing=True)
        await wait_for(
            lambda: daemon.mpris_remote.get_player_state(mock.device_id, "RockPlayer").title == "Recovered Song",
            timeout=4.0,
        )

        st = (await ipc_client.mpris_get_status(mock.device_id, "RockPlayer"))["status"]
        assert st["title"] == "Recovered Song"

        res = await ipc_client.mpris_send_action(mock.device_id, "Play", "RockPlayer")
        assert res["success"] is True

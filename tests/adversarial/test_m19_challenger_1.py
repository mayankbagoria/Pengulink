"""tests.adversarial.test_m19_challenger_1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Adversarial Challenge & Concurrency Stress Suite for Milestone 19:
Phase 3: Concurrency, Performance & Protocol Refinements.

Focus Areas:
1. CLUS-19 (BUG-HIGH-12, BUG-007): MPRIS D-Bus Blocking & Batch GetAll in `pengulink/plugins/mpris/adapter.py`
   - Real asyncio event loop responsiveness under simulated slow D-Bus operations (to_thread verification)
   - Single GetAll batch query verification without sequential Get calls
   - Missing optional properties matrix resilience (LoopStatus, Shuffle, Volume, Position, Metadata)
   - DBusException resilience (UnknownMethod, ServiceUnknown, D-Bus timeouts)
   - Extreme malformed payload fuzzing (corrupt types, nested structures, None values)
   - Massive concurrent barrage of controls, queries, volume changes, and seeks under D-Bus thread pool stress

2. CLUS-11 (BUG-HIGH-04): UDP Discovery DoS Mitigation & Concurrency Control in `pengulink/network/udp.py`
   - Massive UDP storm (5,000+ distinct device ID datagrams) enforcing strict _seen_cache bound (<= 1024)
   - FIFO / LRU eviction verification under continuous storm
   - Semaphore concurrency throttling strictly bound to <= 16 under bursts
   - Zero semaphore permit leakage under exploding callback exceptions
   - Cache TTL eviction and debounce window adversarial verification
   - Datagram fuzzing storm (truncated JSON, binary garbage, 64KB oversized payloads, missing/invalid keys)
   - Rapid stop() and task cancellation lifecycle under active discovery storm
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import dbus.exceptions
from pengulink.core.config import PenguLinkConfig
from pengulink.network.packet import Packet
from pengulink.network.udp import DiscoveredDevice, UDPDiscovery
from pengulink.plugins.mpris.adapter import DBusMprisAdapter, PlayerState


# =============================================================================
# Helper: Event Loop Lag Monitor
# =============================================================================

async def _measure_event_loop_jitter(
    stop_event: asyncio.Event,
    interval_s: float = 0.005,
) -> List[float]:
    """Measure event loop scheduling latency (jitter) continuously.

    If any synchronous call blocks the main thread, the measured jitter will spike.
    """
    jitters: List[float] = []
    while not stop_event.is_set():
        t0 = time.perf_counter()
        await asyncio.sleep(interval_s)
        t1 = time.perf_counter()
        jitter = max(0.0, (t1 - t0) - interval_s)
        jitters.append(jitter)
    return jitters


# =============================================================================
# 1. CLUS-19: MPRIS D-Bus Concurrency, Non-Blocking Event Loop & GetAll Stress
# =============================================================================

class TestMprisDBusAdversarialStress:
    """Empirical adversarial stress testing for DBusMprisAdapter."""

    async def test_event_loop_responsiveness_under_slow_dbus(self) -> None:
        """Verify event loop remains completely non-blocking while slow D-Bus calls execute.

        Simulates D-Bus methods that block the calling thread for 100ms via time.sleep.
        Measures event loop jitter: if executed on the main event loop thread, jitter would be >= 100ms.
        With asyncio.to_thread, event loop jitter must remain tiny (< 45ms).
        """
        adapter = DBusMprisAdapter()
        adapter._running = True

        mock_bus = MagicMock()
        mock_obj = MagicMock()
        mock_bus.get_object.return_value = mock_obj
        adapter._bus = mock_bus

        # D-Bus mock that deliberately blocks the executing thread for 100ms
        def slow_get_all(interface: str) -> Dict[str, Any]:
            time.sleep(0.10)  # Blocking synchronous C-extension simulation
            return {
                "PlaybackStatus": "Playing",
                "Metadata": {"xesam:title": "Slow Song"},
                "Position": 5000000,
                "Volume": 0.8,
            }

        def slow_list_names() -> List[str]:
            time.sleep(0.10)
            return ["org.mpris.MediaPlayer2.slowplayer"]

        with patch("dbus.Interface") as mock_iface_cls:
            mock_iface = MagicMock()
            mock_iface.GetAll.side_effect = slow_get_all
            mock_iface.ListNames.side_effect = slow_list_names
            mock_iface_cls.return_value = mock_iface

            stop_event = asyncio.Event()
            monitor_task = asyncio.create_task(_measure_event_loop_jitter(stop_event, 0.005))

            # Run 4 concurrent slow queries in parallel
            t_start = time.perf_counter()
            results = await asyncio.gather(
                adapter.list_players(),
                adapter.get_player_status("slowplayer"),
                adapter.get_player_status("slowplayer"),
                adapter.get_player_status("slowplayer"),
            )
            t_duration = time.perf_counter() - t_start

            stop_event.set()
            jitters = await monitor_task

            # Verify results were computed successfully
            assert results[0] == ["slowplayer"]
            assert results[1] is not None
            assert results[1].title == "Slow Song"

            # Max jitter on asyncio event loop must NOT spike to >= 100ms
            max_jitter = max(jitters) if jitters else 0.0
            assert max_jitter < 0.045, (
                f"Event loop blocked! Max jitter was {max_jitter*1000:.2f}ms, "
                f"exceeding 45ms limit under 100ms D-Bus blocking call."
            )

    async def test_get_player_status_batch_getall_vs_sequential_queries(self) -> None:
        """Verify get_player_status executes exactly 1 GetAll query and 0 individual Get queries."""
        adapter = DBusMprisAdapter()
        adapter._running = True

        mock_bus = MagicMock()
        mock_obj = MagicMock()
        mock_bus.get_object.return_value = mock_obj
        adapter._bus = mock_bus

        with patch("dbus.Interface") as mock_iface_cls:
            mock_props_iface = MagicMock()
            mock_props_iface.GetAll.return_value = {
                "PlaybackStatus": "Playing",
                "Metadata": {
                    "xesam:title": "Symphony No. 5",
                    "xesam:artist": ["Beethoven"],
                    "xesam:album": "Classical Masterpieces",
                    "mpris:length": 420000000,
                },
                "Position": 120000000,
                "Volume": 0.75,
                "CanPlay": True,
                "CanPause": True,
                "CanGoNext": True,
                "CanGoPrevious": True,
                "CanSeek": True,
                "LoopStatus": "Playlist",
                "Shuffle": True,
            }
            mock_iface_cls.return_value = mock_props_iface

            status = await adapter.get_player_status("vlc")

            # Must call GetAll exactly once
            assert mock_props_iface.GetAll.call_count == 1
            mock_props_iface.GetAll.assert_called_once_with("org.mpris.MediaPlayer2.Player")
            # Must NOT call props_iface.Get even once
            assert mock_props_iface.Get.call_count == 0

            assert status is not None
            assert status.player == "vlc"
            assert status.title == "Symphony No. 5"
            assert status.artist == "Beethoven"
            assert status.album == "Classical Masterpieces"
            assert status.is_playing is True
            assert status.position_us == 120000000
            assert status.length_us == 420000000
            assert status.volume == 0.75
            assert status.loop_status == "Playlist"
            assert status.shuffle is True
            assert status.can_go_next is True
            assert status.can_go_previous is True

    @pytest.mark.parametrize(
        "missing_key,expected_field,expected_val",
        [
            ("LoopStatus", "loop_status", "None"),
            ("Shuffle", "shuffle", False),
            ("Volume", "volume", 1.0),
            ("Position", "position_us", 0),
            ("Metadata", "title", ""),
            ("PlaybackStatus", "is_playing", False),
            ("CanPlay", "can_play", True),
            ("CanPause", "can_pause", True),
            ("CanGoNext", "can_go_next", False),
            ("CanGoPrevious", "can_go_previous", False),
            ("CanSeek", "can_seek", False),
        ],
    )
    async def test_get_player_status_missing_optional_properties_matrix(
        self, missing_key: str, expected_field: str, expected_val: Any
    ) -> None:
        """Matrix test: verify every optional property missing from GetAll falls back to safe default."""
        adapter = DBusMprisAdapter()
        adapter._running = True

        mock_bus = MagicMock()
        mock_obj = MagicMock()
        mock_bus.get_object.return_value = mock_obj
        adapter._bus = mock_bus

        full_props: Dict[str, Any] = {
            "PlaybackStatus": "Playing",
            "Metadata": {"xesam:title": "Test Title", "xesam:artist": ["Artist"]},
            "Position": 1000,
            "Volume": 0.5,
            "CanPlay": True,
            "CanPause": True,
            "CanGoNext": True,
            "CanGoPrevious": True,
            "CanSeek": True,
            "LoopStatus": "Track",
            "Shuffle": True,
        }
        # Deliberately omit target property
        del full_props[missing_key]

        with patch("dbus.Interface") as mock_iface_cls:
            mock_props = MagicMock()
            mock_props.GetAll.return_value = full_props
            mock_iface_cls.return_value = mock_props

            status = await adapter.get_player_status("target_player")
            assert status is not None
            assert getattr(status, expected_field) == expected_val

    async def test_get_player_status_completely_empty_dict(self) -> None:
        """GetAll returning completely empty dict {} yields valid PlayerState with all defaults."""
        adapter = DBusMprisAdapter()
        adapter._running = True
        mock_bus = MagicMock()
        adapter._bus = mock_bus

        with patch("dbus.Interface") as mock_iface_cls:
            mock_props = MagicMock()
            mock_props.GetAll.return_value = {}
            mock_iface_cls.return_value = mock_props

            status = await adapter.get_player_status("bare_player")
            assert status is not None
            assert status.player == "bare_player"
            assert status.is_playing is False
            assert status.title == ""
            assert status.artist == ""
            assert status.album == ""
            assert status.album_art_url == ""
            assert status.position_us == 0
            assert status.length_us == -1
            assert status.volume == 1.0
            assert status.loop_status == "None"
            assert status.shuffle is False
            assert status.can_play is True
            assert status.can_pause is True
            assert status.can_go_next is False
            assert status.can_go_previous is False
            assert status.can_seek is False

    async def test_get_player_status_dbus_exception_resilience(self) -> None:
        """GetAll raising DBusException triggers fallback to empty dict and returns valid PlayerState."""
        adapter = DBusMprisAdapter()
        adapter._running = True
        mock_bus = MagicMock()
        adapter._bus = mock_bus

        with patch("dbus.Interface") as mock_iface_cls:
            mock_props = MagicMock()
            mock_props.GetAll.side_effect = dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.UnknownMethod: Method GetAll is not implemented"
            )
            mock_iface_cls.return_value = mock_props

            status = await adapter.get_player_status("buggy_player")
            # Must not raise DBusException, must return default PlayerState
            assert status is not None
            assert status.player == "buggy_player"
            assert status.is_playing is False
            assert status.title == ""

    async def test_get_player_status_service_unknown_returns_none(self) -> None:
        """When get_object raises ServiceUnknown DBusException, returns None gracefully."""
        adapter = DBusMprisAdapter()
        adapter._running = True
        mock_bus = MagicMock()
        mock_bus.get_object.side_effect = dbus.exceptions.DBusException(
            "org.freedesktop.DBus.Error.ServiceUnknown: The name org.mpris.MediaPlayer2.ghost was not provided"
        )
        adapter._bus = mock_bus

        status = await adapter.get_player_status("ghost")
        assert status is None

    async def test_get_player_status_malformed_payload_fuzzing(self) -> None:
        """Fuzz GetAll dictionary with malformed types (nested strings, negative values, bad keys)."""
        adapter = DBusMprisAdapter()
        adapter._running = True
        mock_bus = MagicMock()
        adapter._bus = mock_bus

        fuzz_cases = [
            # Case 1: Metadata is not a dict
            {"Metadata": "corrupt_metadata_string"},
            # Case 2: Metadata is None
            {"Metadata": None},
            # Case 3: Metadata is integer
            {"Metadata": 12345},
            # Case 4: Non-integer Position and Volume
            {"Position": "non_int_pos", "Volume": "non_float_vol"},
            # Case 5: None values for scalar fields
            {"Position": None, "Volume": None, "LoopStatus": None, "Shuffle": None},
            # Case 6: Malformed artist (not a list, not a string)
            {"Metadata": {"xesam:artist": None, "mpris:length": "bad_len"}},
            # Case 7: Mixed types in artist list
            {"Metadata": {"xesam:artist": ["Valid Artist", 123, None]}},
            # Case 8: Malformed PlaybackStatus
            {"PlaybackStatus": 9999},
            # Case 9: Empty string LoopStatus
            {"LoopStatus": ""},
        ]

        with patch("dbus.Interface") as mock_iface_cls:
            mock_props = MagicMock()
            mock_iface_cls.return_value = mock_props

            for idx, malformed in enumerate(fuzz_cases):
                mock_props.GetAll.return_value = malformed
                status = await adapter.get_player_status(f"fuzz_{idx}")
                assert status is not None, f"Failed on fuzz case {idx}: {malformed}"
                assert isinstance(status.title, str)
                assert isinstance(status.artist, str)
                assert isinstance(status.album, str)
                assert isinstance(status.position_us, int)
                assert isinstance(status.length_us, int)
                assert isinstance(status.volume, float)
                assert isinstance(status.loop_status, str)
                assert isinstance(status.shuffle, bool)
                assert status.loop_status != ""

    async def test_concurrent_actions_barrage_stress(self) -> None:
        """Launch 100 concurrent tasks across all D-Bus methods without deadlock or errors."""
        adapter = DBusMprisAdapter()
        adapter._running = True
        mock_bus = MagicMock()
        mock_obj = MagicMock()
        mock_bus.get_object.return_value = mock_obj
        adapter._bus = mock_bus

        with patch("dbus.Interface") as mock_iface_cls:
            mock_iface = MagicMock()
            mock_iface.ListNames.return_value = [
                "org.mpris.MediaPlayer2.p1",
                "org.mpris.MediaPlayer2.p2",
            ]
            mock_iface.GetAll.return_value = {
                "PlaybackStatus": "Playing",
                "Metadata": {"xesam:title": "Barrage Track"},
                "Position": 1000,
                "Volume": 0.9,
            }
            mock_iface_cls.return_value = mock_iface

            async def _worker(i: int) -> bool:
                p = f"p{i % 2 + 1}"
                if i % 5 == 0:
                    players = await adapter.list_players()
                    return len(players) == 2
                elif i % 5 == 1:
                    st = await adapter.get_player_status(p)
                    return st is not None and st.title == "Barrage Track"
                elif i % 5 == 2:
                    return await adapter.send_action(p, "Play")
                elif i % 5 == 3:
                    return await adapter.set_volume(p, 0.5)
                else:
                    return await adapter.seek(p, 5000)

            tasks = [asyncio.create_task(_worker(i)) for i in range(100)]
            results = await asyncio.gather(*tasks)
            assert all(results)
            assert len(results) == 100


# =============================================================================
# 2. CLUS-11: UDP Discovery DoS Defense, Cache Bounds & Semaphore Throttling
# =============================================================================

class TestUDPDiscoveryStormAdversarialStress:
    """Empirical adversarial stress testing for UDPDiscovery (CLUS-11)."""

    def test_massive_udp_storm_seen_cache_bound(self) -> None:
        """Stress test with a UDP storm (5,000+ distinct datagrams) enforcing cache bound <= 1024.

        Validates:
        - len(_seen_cache) NEVER exceeds 1024 at ANY point during the storm.
        - Final size is exactly 1024.
        - FIFO / LRU eviction: the earliest devices are evicted, latest devices retained.
        """
        discovery = UDPDiscovery(config=PenguLinkConfig())
        assert discovery._seen_cache_max_size == 1024

        total_datagrams = 5500
        cache_sizes_sampled: List[int] = []

        # Stream 5,500 distinct identity packets
        for i in range(total_datagrams):
            dev_id = f"storm_device_{i:05d}"
            raw_pkt = Packet(
                type="kdeconnect.identity",
                body={
                    "deviceId": dev_id,
                    "deviceName": f"Storm Phone {i}",
                    "deviceType": "phone",
                    "protocolVersion": 7,
                    "tcpPort": 1716,
                },
            ).to_wire()

            discovery._handle_datagram(raw_pkt, ("192.168.1.50", 1716))

            # Sample cache size continuously
            if i % 100 == 0 or i > 1020:
                cur_len = len(discovery._seen_cache)
                assert cur_len <= 1024, (
                    f"Cache bound breached! Size {cur_len} > 1024 at iteration {i}"
                )
                cache_sizes_sampled.append(cur_len)

        # Invariant 1: Cache bound strictly capped
        assert len(discovery._seen_cache) == 1024
        assert max(cache_sizes_sampled) == 1024

        # Invariant 2: LRU eviction verification
        # Earliest 4,476 devices (0 to 4475) must have been evicted
        assert "storm_device_00000" not in discovery._seen_cache
        assert "storm_device_01000" not in discovery._seen_cache
        assert "storm_device_04475" not in discovery._seen_cache

        # Latest 1,024 devices (4476 to 5499) must be retained
        assert "storm_device_04476" in discovery._seen_cache
        assert "storm_device_05000" in discovery._seen_cache
        assert "storm_device_05499" in discovery._seen_cache

    async def test_discovery_semaphore_throttles_concurrency_under_burst(self) -> None:
        """Verify _discovery_semaphore strictly bounds concurrent callback execution to <= 16.

        Spawns 200 distinct discoveries in rapid burst. Tracks active concurrency inside callback.
        Verifies:
        - Max active concurrent callbacks is <= 16.
        - Total completed callbacks is exactly 200.
        - Semaphore permits are fully returned to initial value (16).
        - Background tasks are cleanly tracked and drained without leaks.
        """
        discovery = UDPDiscovery(config=PenguLinkConfig())
        concurrency_lock = asyncio.Lock()
        current_active = 0
        max_active = 0
        completed_count = 0

        async def _tracked_callback(dev: DiscoveredDevice) -> None:
            nonlocal current_active, max_active, completed_count
            async with concurrency_lock:
                current_active += 1
                if current_active > max_active:
                    max_active = current_active

            # Simulate I/O or state machine lookup
            await asyncio.sleep(0.01)

            async with concurrency_lock:
                current_active -= 1
                completed_count += 1

        discovery.on_device_discovered = _tracked_callback

        burst_size = 200
        for i in range(burst_size):
            raw_pkt = Packet(
                type="kdeconnect.identity",
                body={
                    "deviceId": f"burst_dev_{i:04d}",
                    "deviceName": f"Burst Device {i}",
                    "deviceType": "tablet",
                    "protocolVersion": 7,
                    "tcpPort": 1716,
                },
            ).to_wire()
            discovery._handle_datagram(raw_pkt, ("10.0.0.1", 1716))

        # In-flight tasks must be tracked
        assert len(discovery._background_tasks) > 0

        # Wait for all background tasks to complete
        await asyncio.gather(*discovery._background_tasks)

        # Invariant 1: Concurrency strictly bounded
        assert max_active <= 16, f"Semaphore exceeded! Max concurrent was {max_active} > 16"
        assert max_active > 1, f"Expected concurrency > 1, got {max_active}"

        # Invariant 2: All tasks completed without drop
        assert completed_count == burst_size
        assert current_active == 0

        # Invariant 3: Background task set drained cleanly
        assert len(discovery._background_tasks) == 0

        # Invariant 4: Semaphore permits fully restored
        assert discovery._discovery_semaphore._value == 16

    async def test_semaphore_release_under_exploding_callbacks(self) -> None:
        """Verify zero semaphore permit leak when callbacks raise unhandled exceptions."""
        discovery = UDPDiscovery(config=PenguLinkConfig())

        async def _exploding_callback(dev: DiscoveredDevice) -> None:
            await asyncio.sleep(0.005)
            raise RuntimeError(f"Simulated crash for {dev.device_id}")

        discovery.on_device_discovered = _exploding_callback

        # Send 50 datagrams that will trigger exceptions
        for i in range(50):
            raw_pkt = Packet(
                type="kdeconnect.identity",
                body={
                    "deviceId": f"exploding_dev_{i}",
                    "deviceName": f"Device {i}",
                    "deviceType": "phone",
                    "protocolVersion": 7,
                    "tcpPort": 1716,
                },
            ).to_wire()
            discovery._handle_datagram(raw_pkt, ("192.168.1.100", 1716))

        # Wait for all tasks with return_exceptions=True
        await asyncio.gather(*discovery._background_tasks, return_exceptions=True)

        # Invariant 1: Tasks drained from background set
        assert len(discovery._background_tasks) == 0

        # Invariant 2: Semaphore permits intact (value == 16)
        assert discovery._discovery_semaphore._value == 16

        # Invariant 3: New valid callback succeeds without deadlock
        recovered = False

        async def _healthy_callback(dev: DiscoveredDevice) -> None:
            nonlocal recovered
            recovered = True

        discovery.on_device_discovered = _healthy_callback
        healthy_pkt = Packet(
            type="kdeconnect.identity",
            body={
                "deviceId": "healthy_device_after_crash",
                "deviceName": "Healthy Phone",
                "deviceType": "phone",
                "protocolVersion": 7,
                "tcpPort": 1716,
            },
        ).to_wire()
        discovery._handle_datagram(healthy_pkt, ("192.168.1.100", 1716))
        await asyncio.gather(*discovery._background_tasks)
        assert recovered is True

    async def test_seen_cache_ttl_and_debounce_adversarial(self) -> None:
        """Verify debounce window suppression and TTL expiration mechanics."""
        discovery = UDPDiscovery(config=PenguLinkConfig())
        dispatched: List[str] = []

        async def _cb(dev: DiscoveredDevice) -> None:
            dispatched.append(dev.device_id)

        discovery.on_device_discovered = _cb

        now = time.time()
        # Seed cache with 100 entries that expired 301 seconds ago
        for i in range(100):
            discovery._seen_cache[f"expired_{i}"] = (now - 301.0, "192.168.1.1", 1716)

        assert len(discovery._seen_cache) == 100

        # Receive 1 fresh datagram: must prune expired entries
        fresh_pkt = Packet(
            type="kdeconnect.identity",
            body={
                "deviceId": "fresh_entry",
                "deviceName": "Fresh",
                "deviceType": "phone",
                "protocolVersion": 7,
                "tcpPort": 1716,
            },
        ).to_wire()
        discovery._handle_datagram(fresh_pkt, ("192.168.1.1", 1716))
        await asyncio.gather(*discovery._background_tasks)

        # All 100 expired entries must be pruned
        assert "fresh_entry" in discovery._seen_cache
        for i in range(100):
            assert f"expired_{i}" not in discovery._seen_cache
        assert len(discovery._seen_cache) == 1

        # Now test debounce window: send 50 duplicate packets for 'fresh_entry' within 2s
        for _ in range(50):
            discovery._handle_datagram(fresh_pkt, ("192.168.1.1", 1716))
        await asyncio.gather(*discovery._background_tasks)

        # Only 1 dispatch was made
        assert dispatched == ["fresh_entry"]

        # Different port from same device ID bypasses debounce and updates entry
        diff_port_pkt = Packet(
            type="kdeconnect.identity",
            body={
                "deviceId": "fresh_entry",
                "deviceName": "Fresh",
                "deviceType": "phone",
                "protocolVersion": 7,
                "tcpPort": 1750,
            },
        ).to_wire()
        discovery._handle_datagram(diff_port_pkt, ("192.168.1.1", 1750))
        await asyncio.gather(*discovery._background_tasks)

        assert len(dispatched) == 2
        assert discovery._seen_cache["fresh_entry"][2] == 1750

    async def test_malformed_udp_datagram_fuzz_storm(self) -> None:
        """Adversarial storm of 1,000 malformed datagrams handled without crashing or cache pollution."""
        discovery = UDPDiscovery(config=PenguLinkConfig())
        dispatched: List[str] = []

        async def _cb(dev: DiscoveredDevice) -> None:
            dispatched.append(dev.device_id)

        discovery.on_device_discovered = _cb

        fuzz_datagrams = [
            b"",
            b"   \r\n\t  ",
            b"\x00\xff\xfe\x01\x02\x03\x80\x90",  # Raw non-UTF-8 binary garbage
            b'{"incomplete": "json',
            b"[1, 2, 3]",  # Array JSON
            b'"just a string"',
            b"12345",
            b'{"type": "kdeconnect.ping"}',  # Non-identity packet
            b'{"type": "kdeconnect.identity"}',  # Missing body
            b'{"type": "kdeconnect.identity", "body": "string_not_dict"}',
            b'{"type": "kdeconnect.identity", "body": {}}',  # Missing deviceId
            b'{"type": "kdeconnect.identity", "body": {"deviceId": 12345}}',  # Int deviceId
            b'{"type": "kdeconnect.identity", "body": {"deviceId": ""}}',  # Empty deviceId
            b'{"type": "kdeconnect.identity", "body": {"deviceId": null}}',
            json.dumps({"type": "kdeconnect.identity", "body": {"deviceId": "a" * 65000}}).encode(),  # 65KB payload
        ]

        # Feed 1,000 malformed datagrams in a loop
        for i in range(1000):
            data = fuzz_datagrams[i % len(fuzz_datagrams)]
            # Must not raise any unhandled exception
            discovery._handle_datagram(data, ("192.168.1.200", 1716))

        await asyncio.gather(*discovery._background_tasks)

        # None of the malformed inputs should have dispatched or polluted cache
        # (Except the 65KB payload with valid string deviceId)
        assert len(discovery._seen_cache) <= 1

    async def test_discovery_stop_drains_and_cancels_inflight_tasks(self) -> None:
        """Stopping discovery service cancels in-flight tasks and cleans up cache without leaks."""
        discovery = UDPDiscovery(config=PenguLinkConfig())
        discovery._running = True

        cancelled_count = 0
        started_count = 0

        async def _lingering_callback(dev: DiscoveredDevice) -> None:
            nonlocal cancelled_count, started_count
            started_count += 1
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                cancelled_count += 1
                raise

        discovery.on_device_discovered = _lingering_callback

        # Spawn 30 tasks
        for i in range(30):
            raw_pkt = Packet(
                type="kdeconnect.identity",
                body={
                    "deviceId": f"stop_dev_{i}",
                    "deviceName": f"Device {i}",
                    "deviceType": "phone",
                    "protocolVersion": 7,
                    "tcpPort": 1716,
                },
            ).to_wire()
            discovery._handle_datagram(raw_pkt, ("192.168.1.1", 1716))

        assert len(discovery._background_tasks) == 30

        # Yield to let the semaphore allow up to 16 tasks to start
        await asyncio.sleep(0.01)
        assert started_count == 16, f"Expected 16 tasks to acquire semaphore, got {started_count}"

        # Now trigger stop
        await discovery.stop()

        # Invariant 1: Service is stopped
        assert discovery.is_running is False

        # Invariant 2: Background tasks are empty
        assert len(discovery._background_tasks) == 0

        # Invariant 3: The 16 started in-flight tasks caught CancelledError
        assert cancelled_count == 16

        # Invariant 4: Cache cleared
        assert len(discovery._seen_cache) == 0

"""tests.unit.test_phase3_defects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dedicated regression test suite for Milestone 19 (Phase 3: Concurrency, Performance & Protocol Refinement):
- CLUS-19 (BUG-HIGH-12, BUG-007): Synchronous Blocking D-Bus Calls & 11 Sequential Queries in DBusMprisAdapter
- CLUS-18 (BUG-HIGH-11): Blocking Synchronous Disk I/O in Async Stream Loops (payload.py, security.py)
- CLUS-17 (BUG-HIGH-10): Destination File Collision TOCTOU Race in Concurrent Staging (security.py, plugin.py)
- CLUS-16 (BUG-HIGH-09): Hidden UNIX Dotfile Creation via Length Truncation in sanitize_filename (security.py)
- CLUS-11 (BUG-HIGH-04): UDP Discovery Unbounded _seen_cache & Unthrottled Task Spawning (udp.py)
- CLUS-26 (BUG-009, BUG-012): Active Notification Cross-Device Collision & Unpair Memory Leaks (notification.py, battery.py)
- CLUS-27 (BUG-010): 24-Hour Clock Drift Window in Clipboard Sync (clipboard/plugin.py)
"""

from __future__ import annotations

import asyncio
import collections
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.core.events import (
    BatteryUpdated,
    DeviceUnpaired,
    EventBus,
    FileReceivedEvent,
    NotificationCleared,
    NotificationReceived,
    ShareReceivedEvent,
)
from pengulink.core.exceptions import PathTraversalError
from pengulink.network.packet import Packet
from pengulink.network.udp import DiscoveredDevice, UDPDiscovery
from pengulink.plugins.battery import BatteryPlugin, DeviceBatteryState
from pengulink.plugins.clipboard.plugin import (
    MAX_FUTURE_CLOCK_DRIFT_MS,
    LoopSuppressionEngine,
)
from pengulink.plugins.mpris.adapter import DBusMprisAdapter, PlayerState
from pengulink.plugins.notification import NotificationPlugin, StoredNotification
from pengulink.plugins.share.payload import PayloadServer
from pengulink.plugins.share.plugin import SharePlugin
from pengulink.plugins.share.security import (
    AtomicFileStager,
    AtomicStagingWriter,
    resolve_collision,
    sanitize_filename,
)


# =============================================================================
# 1. CLUS-19 (BUG-HIGH-12, BUG-007): DBusMprisAdapter Concurrency & GetAll Batching
# =============================================================================

class TestClus19MprisDBusConcurrency:
    """Verify offloading of D-Bus calls via to_thread and GetAll batching with fallbacks."""

    async def test_list_players_offloaded_to_thread(self) -> None:
        """list_players executes D-Bus query inside asyncio.to_thread."""
        adapter = DBusMprisAdapter()
        adapter._running = True
        mock_bus = MagicMock()
        mock_dbus_obj = MagicMock()
        mock_bus.get_object.return_value = mock_dbus_obj
        adapter._bus = mock_bus

        thread_runs: List[bool] = []

        with patch("pengulink.plugins.mpris.adapter.asyncio.to_thread") as mock_to_thread:
            async def _fake_to_thread(func, *args, **kwargs):
                thread_runs.append(True)
                return func(*args, **kwargs)

            mock_to_thread.side_effect = _fake_to_thread

            with patch("dbus.Interface") as mock_iface_cls:
                mock_iface = MagicMock()
                mock_iface.ListNames.return_value = [
                    "org.mpris.MediaPlayer2.vlc",
                    "org.mpris.MediaPlayer2.spotify",
                    "org.freedesktop.DBus",
                ]
                mock_iface_cls.return_value = mock_iface

                players = await adapter.list_players()

                assert mock_to_thread.called
                assert len(thread_runs) == 1
                assert players == ["spotify", "vlc"]

    async def test_get_player_status_uses_getall_and_thread(self) -> None:
        """get_player_status fetches properties via GetAll in to_thread, avoiding 11 sequential calls."""
        adapter = DBusMprisAdapter()
        adapter._running = True
        mock_bus = MagicMock()
        mock_player_obj = MagicMock()
        mock_bus.get_object.return_value = mock_player_obj
        adapter._bus = mock_bus

        thread_runs: List[bool] = []

        with patch("pengulink.plugins.mpris.adapter.asyncio.to_thread") as mock_to_thread:
            async def _fake_to_thread(func, *args, **kwargs):
                thread_runs.append(True)
                return func(*args, **kwargs)

            mock_to_thread.side_effect = _fake_to_thread

            with patch("dbus.Interface") as mock_iface_cls:
                mock_props_iface = MagicMock()
                # Single GetAll return value containing all properties
                mock_props_iface.GetAll.return_value = {
                    "PlaybackStatus": "Playing",
                    "Metadata": {
                        "xesam:title": "Test Title",
                        "xesam:artist": ["Test Artist"],
                        "xesam:album": "Test Album",
                        "mpris:artUrl": "file:///art.jpg",
                        "mpris:length": 180000000,
                        "xesam:url": "https://example.com/track",
                    },
                    "Position": 42000000,
                    "Volume": 0.85,
                    "CanPlay": True,
                    "CanPause": True,
                    "CanGoNext": True,
                    "CanGoPrevious": True,
                    "CanSeek": True,
                    "LoopStatus": "Track",
                    "Shuffle": True,
                }
                mock_iface_cls.return_value = mock_props_iface

                status = await adapter.get_player_status("spotify")

                assert mock_to_thread.called
                assert len(thread_runs) == 1
                mock_props_iface.GetAll.assert_called_once_with("org.mpris.MediaPlayer2.Player")
                assert not mock_props_iface.Get.called

                assert status is not None
                assert status.player == "spotify"
                assert status.title == "Test Title"
                assert status.artist == "Test Artist"
                assert status.album == "Test Album"
                assert status.is_playing is True
                assert status.position_us == 42000000
                assert status.volume == 0.85
                assert status.loop_status == "Track"
                assert status.shuffle is True

    async def test_get_player_status_missing_optional_properties_fallback(self) -> None:
        """Missing optional properties in GetAll fall back to safe defaults without crashing."""
        adapter = DBusMprisAdapter()
        adapter._running = True
        mock_bus = MagicMock()
        adapter._bus = mock_bus

        with patch("dbus.Interface") as mock_iface_cls:
            mock_props_iface = MagicMock()
            # Minimal dictionary omitting LoopStatus, Shuffle, Volume, Position, Metadata
            mock_props_iface.GetAll.return_value = {
                "PlaybackStatus": "Stopped",
            }
            mock_iface_cls.return_value = mock_props_iface

            status = await adapter.get_player_status("minimal_player")

            assert status is not None
            assert status.player == "minimal_player"
            assert status.is_playing is False
            assert status.title == ""
            assert status.artist == ""
            assert status.album == ""
            assert status.volume == 1.0
            assert status.position_us == 0
            assert status.loop_status == "None"
            assert status.shuffle is False

    async def test_send_action_volume_seek_offloaded_to_thread(self) -> None:
        """send_action, set_volume, seek, and set_position offload via asyncio.to_thread."""
        adapter = DBusMprisAdapter()
        adapter._running = True
        mock_bus = MagicMock()
        adapter._bus = mock_bus

        with patch("pengulink.plugins.mpris.adapter.asyncio.to_thread") as mock_to_thread:
            async def _fake_to_thread(func, *args, **kwargs):
                return func(*args, **kwargs)

            mock_to_thread.side_effect = _fake_to_thread

            with patch("dbus.Interface"):
                await adapter.send_action("spotify", "Play")
                assert mock_to_thread.called

                mock_to_thread.reset_mock()
                await adapter.set_volume("spotify", 0.5)
                assert mock_to_thread.called

                mock_to_thread.reset_mock()
                await adapter.seek("spotify", 5000000)
                assert mock_to_thread.called

                mock_to_thread.reset_mock()
                await adapter.set_position("spotify", 10000000)
                assert mock_to_thread.called


# =============================================================================
# 2. CLUS-18 (BUG-HIGH-11): Non-blocking Disk I/O in Async Stream Loops
# =============================================================================

class TestClus18AsyncDiskIO:
    """Verify that file reads and staging writes use asyncio.to_thread."""

    async def test_atomic_staging_writer_write_chunk_uses_to_thread(self, tmp_path: Path) -> None:
        """AtomicStagingWriter.write_chunk executes self.write via asyncio.to_thread."""
        dest_dir = tmp_path / "downloads"
        dest_dir.mkdir()
        final_path = dest_dir / "test.dat"

        writer = AtomicStagingWriter(dest_dir=dest_dir, final_path=final_path, expected_size=10)
        writer.open()

        try:
            with patch("pengulink.plugins.share.security.asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
                n = await writer.write_chunk(b"hello")
                assert n == 5
                assert mock_to_thread.called
                assert writer.bytes_written == 5
        finally:
            writer.close()
            writer.unlink_part()

    async def test_payload_server_send_file_uses_to_thread(self, tmp_path: Path) -> None:
        """PayloadServer.send_file reads chunks via asyncio.to_thread(f.read, CHUNK_SIZE)."""
        test_file = tmp_path / "sample.bin"
        test_file.write_bytes(b"A" * 1024)

        server = PayloadServer(
            local_cert_pem=b"cert",
            local_key_pem=b"key",
            peer_cert_pem=b"peer",
        )

        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_reader = MagicMock()
        server._client_connection = (mock_reader, mock_writer)
        server._connected_event.set()

        with patch("pengulink.plugins.share.payload.asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
            bytes_sent = await server.send_file(test_file, expected_size=1024)
            assert bytes_sent == 1024
            assert mock_to_thread.called
            mock_writer.write.assert_called()


# =============================================================================
# 3. CLUS-17 (BUG-HIGH-10): TOCTOU Collision Reservation in Concurrent Staging
# =============================================================================

class TestClus17CollisionReservation:
    """Verify that in-flight destination paths are reserved across concurrent transfers."""

    def test_resolve_collision_respects_reserved_paths(self, tmp_path: Path) -> None:
        """Candidate paths present in reserved_paths are avoided even before disk creation."""
        dest_dir = tmp_path / "downloads"
        dest_dir.mkdir()

        # File does not exist on disk yet
        initial = dest_dir / "document.pdf"
        assert not initial.exists()

        reserved: Set[Path] = {initial}
        res1 = resolve_collision(dest_dir, "document.pdf", reserved_paths=reserved)
        # Should resolve to counter (1)
        assert res1 == dest_dir / "document (1).pdf"

        # If (1) is also reserved, resolve to (2)
        reserved.add(res1)
        res2 = resolve_collision(dest_dir, "document.pdf", reserved_paths=reserved)
        assert res2 == dest_dir / "document (2).pdf"

    async def test_share_plugin_concurrent_inbound_files_reserve_distinct_paths(self, tmp_path: Path) -> None:
        """Simultaneous inbound file transfers for the same filename obtain distinct paths and clean up."""
        event_bus = EventBus()
        plugin = SharePlugin(event_bus=event_bus)

        # Override download dir
        with patch("pengulink.plugins.share.plugin.resolve_download_dir", return_value=tmp_path):
            packet1 = Packet(
                type="kdeconnect.share.request",
                body={"filename": "photo.jpg"},
                payloadSize=0,  # Zero-byte files finish immediately
            )
            packet2 = Packet(
                type="kdeconnect.share.request",
                body={"filename": "photo.jpg"},
                payloadSize=0,
            )

            # Concurrent transfers
            await asyncio.gather(
                plugin._handle_inbound_file(packet1, "dev1"),
                plugin._handle_inbound_file(packet2, "dev2"),
            )

            # Verify two distinct files were created on disk without overwriting
            files = sorted([p.name for p in tmp_path.iterdir() if p.is_file()])
            assert "photo.jpg" in files
            assert "photo (1).jpg" in files

            # Verify reserved set is cleanly discarded
            assert len(plugin._reserved_destinations) == 0


# =============================================================================
# 4. CLUS-16 (BUG-HIGH-09): Leading Dot Defense Post Byte-Clamping
# =============================================================================

class TestClus16HiddenDotfilePrevention:
    """Verify that sanitized filenames never begin with a dot, even under multi-byte truncation."""

    def test_sanitize_filename_multibyte_truncation_no_leading_dot(self) -> None:
        """Stem multi-byte truncation reducing stem to empty string never produces a dotfile."""
        # 60 emojis (240 bytes) with 238 'a's suffix (total 239 suffix bytes)
        raw = "😀" * 60 + "." + "a" * 238
        cleaned = sanitize_filename(raw, max_bytes=240)

        assert not cleaned.startswith(".")
        assert len(cleaned.encode("utf-8")) <= 240
        assert cleaned.startswith("file_")

    def test_sanitize_filename_all_dots(self) -> None:
        """Filename composed only of dots raises PathTraversalError; dots preceding text are sanitized."""
        with pytest.raises(PathTraversalError):
            sanitize_filename(".....", max_bytes=240)

        cleaned = sanitize_filename(".....test.txt", max_bytes=240)
        assert not cleaned.startswith(".")
        assert "test.txt" in cleaned

    def test_sanitize_filename_leading_dot_with_long_extension(self) -> None:
        """Leading dot file with long extension respects length limit and leading dot invariant."""
        raw = "." + "x" * 250
        cleaned = sanitize_filename(raw, max_bytes=240)
        assert not cleaned.startswith(".")
        assert len(cleaned.encode("utf-8")) <= 240


# =============================================================================
# 5. CLUS-11 (BUG-HIGH-04): UDP Discovery Bounded Seen Cache & Semaphore Throttling
# =============================================================================

class TestClus11UDPDiscoveryDoSDefense:
    """Verify bounded OrderedDict seen cache and semaphore concurrency throttling."""

    def test_seen_cache_capped_at_1024(self) -> None:
        """Inserting > 1024 distinct device IDs caps cache at 1024 and evicts oldest."""
        discovery = UDPDiscovery(config=PenguLinkConfig())
        assert isinstance(discovery._seen_cache, collections.OrderedDict)
        assert discovery._seen_cache_max_size == 1024

        # Simulate 1100 unique device datagrams
        for i in range(1100):
            data = Packet(
                type="kdeconnect.identity",
                body={
                    "deviceId": f"device_{i:04d}",
                    "deviceName": f"Device {i}",
                    "deviceType": "phone",
                    "protocolVersion": 7,
                    "tcpPort": 1716,
                },
            ).to_wire()
            discovery._handle_datagram(data, ("192.168.1.50", 1716))

        assert len(discovery._seen_cache) == 1024
        # Oldest items (device_0000 .. device_0075) must have been evicted
        assert "device_0000" not in discovery._seen_cache
        assert "device_0075" not in discovery._seen_cache
        assert "device_1099" in discovery._seen_cache

    def test_seen_cache_ttl_expiration(self) -> None:
        """Old entries past seen_cache_ttl are pruned upon receiving new datagrams."""
        discovery = UDPDiscovery(config=PenguLinkConfig())
        now = time.time()
        # Seed cache with an entry from 10 minutes ago
        discovery._seen_cache["stale_device"] = (now - 600.0, "192.168.1.20", 1716)

        # Receive a fresh datagram
        fresh_data = Packet(
            type="kdeconnect.identity",
            body={
                "deviceId": "fresh_device",
                "deviceName": "Fresh Phone",
                "deviceType": "phone",
                "protocolVersion": 7,
                "tcpPort": 1716,
            },
        ).to_wire()
        discovery._handle_datagram(fresh_data, ("192.168.1.30", 1716))

        # Stale device was pruned
        assert "stale_device" not in discovery._seen_cache
        assert "fresh_device" in discovery._seen_cache

    async def test_discovery_semaphore_throttles_concurrency(self) -> None:
        """Flood of novel discoveries is throttled by _discovery_semaphore."""
        discovery = UDPDiscovery(config=PenguLinkConfig())
        active_concurrent = 0
        max_seen_concurrent = 0

        async def _slow_callback(device: DiscoveredDevice) -> None:
            nonlocal active_concurrent, max_seen_concurrent
            active_concurrent += 1
            max_seen_concurrent = max(max_seen_concurrent, active_concurrent)
            await asyncio.sleep(0.02)
            active_concurrent -= 1

        discovery.on_device_discovered = _slow_callback

        # Send 30 distinct datagrams rapidly
        for i in range(30):
            data = Packet(
                type="kdeconnect.identity",
                body={
                    "deviceId": f"throttle_dev_{i}",
                    "deviceName": f"Device {i}",
                    "deviceType": "phone",
                    "protocolVersion": 7,
                    "tcpPort": 1716,
                },
            ).to_wire()
            discovery._handle_datagram(data, ("192.168.1.50", 1716))

        # Wait for all background tasks to finish
        if discovery._background_tasks:
            await asyncio.gather(*discovery._background_tasks, return_exceptions=True)

        assert max_seen_concurrent <= 16


# =============================================================================
# 6. CLUS-26 (BUG-009, BUG-012): Notification Composite Key & Unpair Memory Leaks
# =============================================================================

class TestClus26NotificationAndBatteryLifecycle:
    """Verify composite notification keying, backward compatibility, and DeviceUnpaired cleanups."""

    async def test_notification_composite_key_prevents_collision(self) -> None:
        """Two different devices emitting notification with ID '100' do not collide."""
        plugin = NotificationPlugin()
        dummy_sender = AsyncMock()

        packet1 = Packet(
            type="kdeconnect.notification",
            body={
                "id": "100",
                "appName": "WhatsApp",
                "title": "Alice",
                "text": "Hello from Phone A",
            },
        )
        packet2 = Packet(
            type="kdeconnect.notification",
            body={
                "id": "100",
                "appName": "Signal",
                "title": "Bob",
                "text": "Hello from Phone B",
            },
        )

        await plugin.handle_packet(packet1, "phone_a", dummy_sender)
        await plugin.handle_packet(packet2, "phone_b", dummy_sender)

        # Both notifications coexist in cache
        assert len(plugin.list_notifications()) == 2

        notif_a = plugin.get_notification("100", device_id="phone_a")
        notif_b = plugin.get_notification("100", device_id="phone_b")

        assert notif_a is not None and notif_a.app_name == "WhatsApp"
        assert notif_b is not None and notif_b.app_name == "Signal"

        # Backward compatible single-argument get_notification finds a matching notification
        any_notif = plugin.get_notification("100")
        assert any_notif is not None

    async def test_notification_device_unpaired_cleans_only_target_device(self) -> None:
        """DeviceUnpaired event purges all notifications for that device only."""
        event_bus = EventBus()
        plugin = NotificationPlugin(event_bus=event_bus)
        await plugin.start()
        dummy_sender = AsyncMock()

        # Add 2 notifications for phone_a and 1 for phone_b
        await plugin.handle_packet(
            Packet(type="kdeconnect.notification", body={"id": "1", "appName": "A1"}),
            "phone_a",
            dummy_sender,
        )
        await plugin.handle_packet(
            Packet(type="kdeconnect.notification", body={"id": "2", "appName": "A2"}),
            "phone_a",
            dummy_sender,
        )
        await plugin.handle_packet(
            Packet(type="kdeconnect.notification", body={"id": "1", "appName": "B1"}),
            "phone_b",
            dummy_sender,
        )

        assert len(plugin.list_notifications()) == 3

        # Unpair phone_a
        await event_bus.publish(DeviceUnpaired(device_id="phone_a"))

        # phone_a notifications removed; phone_b remains
        remaining = plugin.list_notifications()
        assert len(remaining) == 1
        assert remaining[0].device_id == "phone_b"
        assert remaining[0].app_name == "B1"

        await plugin.stop()

    async def test_battery_device_unpaired_cleans_device_state(self) -> None:
        """BatteryPlugin purges device battery state upon receiving DeviceUnpaired."""
        event_bus = EventBus()
        plugin = BatteryPlugin(event_bus=event_bus)
        await plugin.start()
        dummy_sender = AsyncMock()

        packet = Packet(
            type="kdeconnect.battery",
            body={"currentCharge": 85, "isCharging": True},
        )
        await plugin.handle_packet(packet, "phone_x", dummy_sender)
        assert plugin.get_device_battery("phone_x") is not None
        assert plugin.get_device_battery("phone_x").current_charge == 85

        # Unpair phone_x
        await event_bus.publish(DeviceUnpaired(device_id="phone_x"))
        assert plugin.get_device_battery("phone_x") is None

        await plugin.stop()


# =============================================================================
# 7. CLUS-27 (BUG-010): Clipboard Clock Drift Window & Clamp
# =============================================================================

class TestClus27ClipboardClockDrift:
    """Verify reduced clock drift window and timestamp clamping in LoopSuppressionEngine."""

    def test_max_future_clock_drift_constant_is_15_seconds(self) -> None:
        """MAX_FUTURE_CLOCK_DRIFT_MS must be 15,000 ms (15 seconds)."""
        assert MAX_FUTURE_CLOCK_DRIFT_MS == 15_000

    def test_evaluate_inbound_rejects_beyond_15s_drift(self) -> None:
        """Inbound packet with timestamp > now + 15s is rejected."""
        engine = LoopSuppressionEngine(max_future_drift_ms=MAX_FUTURE_CLOCK_DRIFT_MS)
        fixed_now = 1_000_000
        engine.current_wall_ms = lambda: fixed_now

        # At boundary: now + 15s -> accepted
        ok, reason = engine.evaluate_inbound("test content", fixed_now + 15_000, "peer1")
        assert ok is True
        assert reason == "accepted"

        # Beyond boundary: now + 15s + 1ms -> rejected
        ok, reason = engine.evaluate_inbound("future content", fixed_now + 15_001, "peer1")
        assert ok is False
        assert reason == "future_clock_drift_exceeded"

    def test_record_inbound_accepted_clamps_local_timestamp(self) -> None:
        """record_inbound_accepted clamps local timestamp to prevent future clock lockouts."""
        engine = LoopSuppressionEngine(max_future_drift_ms=15_000)
        now_ms = engine.current_wall_ms()

        # Suppose a packet arrived with timestamp slightly in the future: now + 5000 ms
        future_ts = now_ms + 5_000
        engine.record_inbound_accepted("some text", future_ts)

        # Local timestamp updated
        assert engine.local_timestamp == future_ts

        # Now test clamping when remote timestamp exceeds now_ms + max_future_drift_ms
        # e.g. caller passed remote_timestamp 100,000 ms in future
        engine.record_inbound_accepted("far future text", now_ms + 100_000)
        # Clamped to now_ms + 15_000 ms, not 100_000 ms
        current_wall = engine.current_wall_ms()
        assert engine.local_timestamp <= current_wall + 15_000

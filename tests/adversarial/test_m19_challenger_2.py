"""tests.adversarial.test_m19_challenger_2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial Stress & Empirical Challenge Suite for Milestone 19:
Phase 3 — Concurrency, Performance & Protocol Refinements.

Empirical Challenge Objectives:
1. CLUS-18 & CLUS-17: High Concurrency File Transfer TOCTOU Race & Async Disk I/O:
   - 50+ concurrent inbound file transfers with identical target filenames across
     devices to verify `_collision_lock` and `reserved_paths` allocate distinct,
     non-overlapping, non-colliding files on disk with zero data clobbering.
   - Concurrent downloads with simulated network and staging delay.
   - Transfer cancellation and error handling cleanly releases reserved paths.
   - Fallback UUID generation under collision counter exhaustion (10,000).
   - Async disk I/O offloading: verify `PayloadServer.send_file` and
     `AtomicStagingWriter.write_chunk` execute on worker threads (threading.get_ident()
     differs from main loop thread) and do not freeze event loop responsiveness.

2. CLUS-16: Filename Sanitization Byte Boundary Truncation Fuzzing:
   - Exhaustive fuzzing with 1,000+ hostile inputs: multibyte UTF-8 (2-byte Latin,
     3-byte CJK, 4-byte emojis), boundary slices at every byte offset, trailing dots,
     leading dots, sequences of dots, and long extensions.
   - Strictly assert that NO output filename ever starts with `.`, length never
     exceeds max_bytes, and no path separators or null bytes exist.

3. CLUS-26 & CLUS-27: Notifications, Battery State, and Clipboard Clock Drift:
   - 50+ devices emitting identical notification IDs simultaneously; verify zero
     cross-device overwrites, correct composite key lookup `(device_id, notification_id)`,
     and device-scoped dismissal/cancellation.
   - `DeviceUnpaired` cleanup: verify exact purge of target device's notifications
     and battery cache without touching other devices or leaking memory.
   - Clipboard clock drift clamping: verify 15-second boundary enforcement, rejection
     of timestamps > now + 15s, clamping in `record_inbound_accepted`, and multi-device
     clock convergence without 24-hour lockout.
"""

from __future__ import annotations

import asyncio
import os
import random
import string
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.core.events import (
    BatteryUpdated,
    DeviceUnpaired,
    EventBus,
    FileReceivedEvent,
    FileTransferProgress,
    NotificationCleared,
    NotificationReceived,
    ShareReceivedEvent,
)
from pengulink.core.exceptions import (
    PathTraversalError,
    PayloadSizeExceededError,
)
from pengulink.network.packet import Packet
from pengulink.plugins.battery import BatteryPlugin, DeviceBatteryState
from pengulink.plugins.clipboard.plugin import (
    MAX_CLIPBOARD_PACKET_BYTES,
    MAX_FUTURE_CLOCK_DRIFT_MS,
    ClipboardPlugin,
    LoopSuppressionEngine,
)
from pengulink.plugins.notification import NotificationPlugin, StoredNotification
from pengulink.plugins.share.payload import CHUNK_SIZE, PayloadClient, PayloadServer
from pengulink.plugins.share.plugin import SharePlugin
from pengulink.plugins.share.security import (
    MAX_COLLISION_ATTEMPTS,
    MAX_FILENAME_BYTES,
    AtomicFileStager,
    AtomicStagingWriter,
    resolve_collision,
    sanitize_filename,
    split_stem_and_suffix,
)


# =============================================================================
# 1. CLUS-18 & CLUS-17: High Concurrency File Transfer TOCTOU Race & Async Disk I/O
# =============================================================================

class TestClus17HighConcurrencyTOCTOU:
    """Stress-test concurrent inbound file transfers against TOCTOU race conditions."""

    async def test_high_concurrency_file_transfer_collision_60_transfers(self, tmp_path: Path) -> None:
        """60 concurrent inbound transfers with identical target filename must allocate

        60 distinct, non-overlapping files without clobbering or data loss.
        """
        event_bus = EventBus()
        plugin = SharePlugin(event_bus=event_bus)
        received_files: List[Path] = []

        async def _on_file_received(event: FileReceivedEvent) -> None:
            received_files.append(event.file_path)

        event_bus.subscribe(FileReceivedEvent, _on_file_received)

        target_filename = "confidential_quarterly_report.pdf"
        transfer_count = 60

        with patch("pengulink.plugins.share.plugin.resolve_download_dir", return_value=tmp_path):
            # Create 60 identical inbound file packets (zero-byte fast path)
            packets = [
                Packet(
                    type="kdeconnect.share.request",
                    body={"filename": target_filename},
                    payloadSize=0,
                )
                for _ in range(transfer_count)
            ]

            # Fire all 60 transfers simultaneously via asyncio.gather
            await asyncio.gather(
                *[
                    plugin._handle_inbound_file(pkt, f"device_{i:03d}")
                    for i, pkt in enumerate(packets)
                ]
            )

            # Verification 1: Exactly 60 events received
            assert len(received_files) == transfer_count

            # Verification 2: All 60 allocated paths are distinct
            allocated_set = set(received_files)
            assert len(allocated_set) == transfer_count, (
                f"Collision detected! Expected {transfer_count} distinct paths, got {len(allocated_set)}"
            )

            # Verification 3: All 60 files exist on disk
            for p in received_files:
                assert p.exists()
                assert p.is_file()

            # Verification 4: Filenames follow expected deterministic pattern:
            # report.pdf, report (1).pdf, ..., report (59).pdf
            file_names = {p.name for p in received_files}
            assert target_filename in file_names
            for c in range(1, transfer_count):
                expected_name = f"confidential_quarterly_report ({c}).pdf"
                assert expected_name in file_names, f"Missing expected collision candidate: {expected_name}"

            # Verification 5: In-flight reservation set must be completely empty
            assert len(plugin._reserved_destinations) == 0

    async def test_concurrent_transfers_with_simulated_payload_streaming(self, tmp_path: Path) -> None:
        """50 concurrent transfers downloading payload with simulated network delay.

        Verify in-flight reservation prevents collisions during active staging.
        """
        event_bus = EventBus()
        plugin = SharePlugin(event_bus=event_bus)

        # Mock peer_resolver so it returns dummy peer info
        plugin.peer_resolver = MagicMock(return_value=("127.0.0.1", b"fake_cert"))

        transfer_count = 50
        file_size = 256
        data_payload = b"X" * file_size
        target_name = "shared_dataset.tar.gz"

        allocated_paths: List[Path] = []
        reservation_sizes_during_transfer: List[int] = []

        # Custom fake download_to_staging that simulates network latency while staging
        async def _fake_download(
            target_host: str,
            target_port: int,
            expected_size: int,
            staging_writer: AtomicStagingWriter,
            target_device_id: str,
            filename: str,
            progress_callback: Any = None,
        ) -> Path:
            # Record reserved destinations size during active in-flight transfer
            reservation_sizes_during_transfer.append(len(plugin._reserved_destinations))

            staging_writer.open()
            # Simulate streaming chunks with brief yields
            await asyncio.sleep(0.005)
            await staging_writer.write_chunk(data_payload[:128])
            await asyncio.sleep(0.005)
            await staging_writer.write_chunk(data_payload[128:])
            final_p = staging_writer.commit()
            allocated_paths.append(final_p)
            return final_p

        with patch("pengulink.plugins.share.plugin.resolve_download_dir", return_value=tmp_path):
            with patch.object(PayloadClient, "download_to_staging", side_effect=_fake_download):
                packets = [
                    Packet(
                        type="kdeconnect.share.request",
                        body={"filename": target_name},
                        payloadSize=file_size,
                        payloadTransferInfo={"port": 1740 + (i % 20)},
                    )
                    for i in range(transfer_count)
                ]

                await asyncio.gather(
                    *[
                        plugin._handle_inbound_file(pkt, f"peer_{i}")
                        for i, pkt in enumerate(packets)
                    ]
                )

        # Verify 50 distinct files created on disk with exact payload
        assert len(allocated_paths) == transfer_count
        assert len(set(allocated_paths)) == transfer_count
        for p in allocated_paths:
            assert p.exists()
            assert p.read_bytes() == data_payload

        # Verify that during active transfers, multiple paths were concurrently reserved
        assert max(reservation_sizes_during_transfer) > 1

        # Verify reservation cleanup
        assert len(plugin._reserved_destinations) == 0

    async def test_transfer_failure_releases_reserved_path_cleanly(self, tmp_path: Path) -> None:
        """When an inbound transfer fails or raises mid-transfer, its reservation

        is discarded from _reserved_destinations, allowing a retry with the same name.
        """
        event_bus = EventBus()
        plugin = SharePlugin(event_bus=event_bus)
        plugin.peer_resolver = MagicMock(return_value=("127.0.0.1", b"fake_cert"))

        target_name = "critical_update.bin"

        # Mock download_to_staging that fails on first attempt
        attempt = 0

        async def _flaky_download(*args: Any, **kwargs: Any) -> Path:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                # First attempt fails with exception
                raise ConnectionResetError("Remote peer dropped during transfer")
            # Second attempt succeeds
            writer: AtomicStagingWriter = kwargs["staging_writer"]
            writer.open()
            await writer.write_chunk(b"SUCCESS")
            return writer.commit()

        with patch("pengulink.plugins.share.plugin.resolve_download_dir", return_value=tmp_path):
            with patch.object(PayloadClient, "download_to_staging", side_effect=_flaky_download):
                pkt = Packet(
                    type="kdeconnect.share.request",
                    body={"filename": target_name},
                    payloadSize=7,
                    payloadTransferInfo={"port": 1745},
                )

                # Attempt 1: Must fail
                with pytest.raises(ConnectionResetError):
                    await plugin._handle_inbound_file(pkt, "peer_err")

                # Verify reservation was freed
                assert len(plugin._reserved_destinations) == 0

                # Attempt 2: Must succeed and get the original un-suffixed target_name
                await plugin._handle_inbound_file(pkt, "peer_retry")

                final_file = tmp_path / target_name
                assert final_file.exists()
                assert final_file.read_bytes() == b"SUCCESS"
                assert len(plugin._reserved_destinations) == 0

    def test_collision_counter_exhaustion_uuid_fallback_with_reserved_paths(self, tmp_path: Path) -> None:
        """When collision counter reaches MAX_COLLISION_ATTEMPTS (10,000), fallback UUID

        generation resolves to a unique candidate that respects reserved_paths.
        """
        dest_dir = tmp_path / "coll_test"
        dest_dir.mkdir()
        base_name = "archive.zip"
        stem, suffix = split_stem_and_suffix(base_name)

        # Create base file and simulate all counter files (1..10000) being reserved
        reserved: Set[Path] = {dest_dir / base_name}
        for i in range(1, MAX_COLLISION_ATTEMPTS + 1):
            reserved.add(dest_dir / f"{stem} ({i}){suffix}")

        # Resolve collision with all 10,000 candidates reserved
        chosen = resolve_collision(dest_dir, base_name, reserved_paths=reserved)

        assert chosen not in reserved
        assert chosen.parent == dest_dir
        assert chosen.suffix == ".zip"
        # Must have stem_UUID.suffix format
        assert chosen.name.startswith("archive_")
        assert len(chosen.name) == len("archive_") + 8 + len(".zip")


class TestClus18AsyncDiskIOOffloading:
    """Empirical verification that disk read/write calls are offloaded to worker threads."""

    async def test_atomic_staging_writer_write_chunk_offloaded_to_thread(self, tmp_path: Path) -> None:
        """AtomicStagingWriter.write_chunk executes on a separate thread than the event loop."""
        main_thread_id = threading.get_ident()
        writer_thread_ids: List[int] = []

        dest_dir = tmp_path / "staging_threads"
        dest_dir.mkdir()
        target = dest_dir / "output.bin"

        writer = AtomicStagingWriter(dest_dir=dest_dir, final_path=target, expected_size=1000)
        writer.open()

        # Wrap writer.write to record thread ID
        original_write = writer.write

        def _recording_write(data: bytes) -> int:
            writer_thread_ids.append(threading.get_ident())
            return original_write(data)

        writer.write = _recording_write  # type: ignore

        try:
            for _ in range(5):
                await writer.write_chunk(b"A" * 200)
            writer.commit()

            assert len(writer_thread_ids) == 5
            # Verify ALL write executions occurred on worker threads, NOT main_thread_id
            for tid in writer_thread_ids:
                assert tid != main_thread_id, f"write executed on main thread {tid} == {main_thread_id}"
        finally:
            writer.close()
            if writer.part_path.exists():
                writer.part_path.unlink()

    async def test_payload_server_send_file_offloaded_to_thread(self, tmp_path: Path) -> None:
        """PayloadServer.send_file reads chunks via asyncio.to_thread off main thread."""
        main_thread_id = threading.get_ident()
        read_thread_ids: List[int] = []

        sample_file = tmp_path / "server_test.bin"
        sample_file.write_bytes(b"B" * 4096)

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

        # Intercept asyncio.to_thread in payload module
        real_to_thread = asyncio.to_thread

        async def _tracking_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
            read_thread_ids.append(threading.get_ident())
            return await real_to_thread(func, *args, **kwargs)

        with patch("pengulink.plugins.share.payload.asyncio.to_thread", side_effect=_tracking_to_thread):
            bytes_sent = await server.send_file(sample_file, expected_size=4096)
            assert bytes_sent == 4096
            assert len(read_thread_ids) > 0
            # Read was dispatched via asyncio.to_thread
            mock_writer.write.assert_called()

    async def test_event_loop_responsiveness_during_heavy_payload_stream(self, tmp_path: Path) -> None:
        """Event loop tick jitter remains low (<50ms) during heavy file chunk writes."""
        dest_dir = tmp_path / "jitter_test"
        dest_dir.mkdir()
        target = dest_dir / "large.dat"

        total_bytes = 1024 * 1024 * 2  # 2 MB
        chunk = b"Z" * (64 * 1024)      # 64 KB chunks
        num_chunks = total_bytes // len(chunk)

        writer = AtomicStagingWriter(dest_dir=dest_dir, final_path=target, expected_size=total_bytes)
        writer.open()

        max_tick_delay = 0.0
        stop_heartbeat = False

        async def _heartbeat() -> None:
            nonlocal max_tick_delay
            last = time.perf_counter()
            while not stop_heartbeat:
                await asyncio.sleep(0.005)
                now = time.perf_counter()
                delay = now - last - 0.005
                if delay > max_tick_delay:
                    max_tick_delay = delay
                last = now

        hb_task = asyncio.create_task(_heartbeat())

        try:
            for _ in range(num_chunks):
                await writer.write_chunk(chunk)
            writer.commit()
        finally:
            stop_heartbeat = True
            await hb_task
            writer.close()

        # Verify event loop tick delay did not stall beyond 50ms
        assert max_tick_delay < 0.05, f"Event loop stalled: max delay = {max_tick_delay*1000:.2f}ms"
        assert target.exists()
        assert target.stat().st_size == total_bytes


# =============================================================================
# 2. CLUS-16: Filename Sanitization Byte Boundary Truncation Fuzzing
# =============================================================================

class TestClus16FilenameSanitizationFuzzing:
    """Adversarial boundary fuzzing for sanitize_filename to ensure NO leading dots ever occur."""

    @pytest.mark.parametrize("max_bytes", [10, 30, 64, 128, 240])
    def test_fuzz_multibyte_utf8_truncation_invariants(self, max_bytes: int) -> None:
        """Fuzz multibyte UTF-8 sequences (2-byte, 3-byte, 4-byte) sliced at every boundary."""
        multibyte_chars = [
            "é", "ñ", "ü", "ß", "Д", "ж", "Ω", "λ",  # 2-byte
            "中", "文", "字", "符", "日", "本", "語", "한",  # 3-byte
            "😀", "🚀", "🐧", "🎉", "🔥", "🐱", "🌍", "💎",  # 4-byte
        ]

        extensions = [".txt", ".pdf", ".tar.gz", ".tar.xz", ".longextensionname", ""]

        rng = random.Random(42)

        for _ in range(200):
            # Construct random stem of 1 to 80 multibyte characters
            stem_len = rng.randint(1, 80)
            stem = "".join(rng.choice(multibyte_chars) for _ in range(stem_len))
            ext = rng.choice(extensions)
            raw = f"{stem}{ext}"

            cleaned = sanitize_filename(raw, max_bytes=max_bytes)

            # Invariant 1: NEVER starts with a dot
            assert not cleaned.startswith("."), (
                f"FAILED: Sanitized filename starts with dot: {cleaned!r} (from raw: {raw!r}, max_bytes={max_bytes})"
            )

            # Invariant 2: UTF-8 encoded byte length <= max_bytes
            encoded_len = len(cleaned.encode("utf-8"))
            assert encoded_len <= max_bytes, (
                f"FAILED: Exceeded max_bytes ({encoded_len} > {max_bytes}): {cleaned!r}"
            )

            # Invariant 3: Must not be empty
            assert len(cleaned) > 0

            # Invariant 4: No path separators or null bytes
            assert "/" not in cleaned and "\\" not in cleaned
            assert "\x00" not in cleaned

    def test_fuzz_dots_and_extensions_boundary_conditions(self) -> None:
        """Exhaustive boundary testing of filenames with leading, trailing, and clustered dots."""
        hostile_inputs = [
            # Trailing dots before/after stem
            "report.pdf....",
            "archive.tar.gz......",
            "test....txt",
            # Leading dot files with multi-byte stems
            ".hidden_file",
            "..hidden_double",
            "...hidden_triple",
            "." + "😀" * 60,
            ".." + "中" * 70,
            # Stem truncation reducing stem to 0 bytes leaving suffix
            "😀" * 60 + "." + "a" * 238,
            "🐧" * 59 + "." + "b" * 239,
            "🔥" * 30 + "." + "c" * 250,
            # Suffix itself exceeding max_bytes
            "short." + "x" * 260,
            "." + "y" * 250,
            # Dots mixed with spaces and control characters
            " . . hidden . . ",
            "   .bashrc   ",
            "...config.json...",
            "....leading....and....trailing....",
        ]

        for raw in hostile_inputs:
            cleaned = sanitize_filename(raw, max_bytes=240)
            assert not cleaned.startswith("."), f"Sanitized filename starts with dot: {cleaned!r} for raw {raw!r}"
            assert len(cleaned.encode("utf-8")) <= 240
            assert len(cleaned) > 0
            assert "/" not in cleaned and "\\" not in cleaned

    def test_pure_traversal_and_dot_sequences_raise_traversal_error(self) -> None:
        """Sequences consisting solely of dots or traversal components must raise PathTraversalError."""
        invalid_inputs = [
            ".",
            "..",
            "...",
            "....",
            ".....",
            "......",
            "/",
            "//",
            "///",
            "",
            "   ",
        ]

        for raw in invalid_inputs:
            with pytest.raises(PathTraversalError):
                sanitize_filename(raw, max_bytes=240)

    def test_exact_byte_boundary_truncation_preserves_valid_utf8(self) -> None:
        """Verify that slicing multi-byte characters at exact byte boundaries never

        produces invalid UTF-8 replacement or surrogate garbage in the output.
        """
        # 4-byte emoji: \xf0\x9f\x98\x80
        # If max_bytes cuts midway through (e.g. at byte 6 where emoji starts at byte 5),
        # errors='ignore' drops the partial byte sequence cleanly.
        for offset in range(1, 10):
            raw = "A" * offset + "😀🚀🐧"
            cleaned = sanitize_filename(raw, max_bytes=12)
            # Must roundtrip cleanly as valid UTF-8
            encoded = cleaned.encode("utf-8")
            assert encoded.decode("utf-8") == cleaned
            assert not cleaned.startswith(".")
            assert len(encoded) <= 12


# =============================================================================
# 3. CLUS-26: Cross-Device Notification Isolation & Lifecycle Cleanup
# =============================================================================

class TestClus26NotificationAndBatteryAdversarial:
    """Stress-test notification composite key isolation and DeviceUnpaired cleanup."""

    async def test_cross_device_notification_id_collision_50_devices(self) -> None:
        """50 distinct devices sending notifications with the EXACT same notification_id.

        Verify all 50 co-exist without overwriting, and retrieve by composite key.
        """
        plugin = NotificationPlugin()
        dummy_sender = AsyncMock()

        shared_notif_id = "global_system_alert_001"
        device_count = 50

        # Concurrently dispatch 50 notifications with identical ID
        packets = [
            (
                f"device_{i:03d}",
                Packet(
                    type="kdeconnect.notification",
                    body={
                        "id": shared_notif_id,
                        "appName": f"App_{i}",
                        "title": f"Title for Device {i}",
                        "text": f"Payload {i}",
                        "ticker": f"Ticker {i}",
                    },
                ),
            )
            for i in range(device_count)
        ]

        await asyncio.gather(
            *[plugin.handle_packet(pkt, dev_id, dummy_sender) for dev_id, pkt in packets]
        )

        # Verification 1: Exactly 50 active notifications stored
        all_notifs = plugin.list_notifications()
        assert len(all_notifs) == device_count

        # Verification 2: Each device's notification is uniquely retrievable
        for i in range(device_count):
            dev_id = f"device_{i:03d}"
            notif = plugin.get_notification(shared_notif_id, device_id=dev_id)
            assert notif is not None, f"Missing notification for {dev_id}"
            assert notif.device_id == dev_id
            assert notif.app_name == f"App_{i}"
            assert notif.title == f"Title for Device {i}"

        # Verification 3: list_notifications filtered by device_id returns exactly 1 item
        dev_0_notifs = plugin.list_notifications(device_id="device_000")
        assert len(dev_0_notifs) == 1
        assert dev_0_notifs[0].device_id == "device_000"

    async def test_targeted_notification_dismissal_preserves_peers(self) -> None:
        """Dismissing notification for device_A removes ONLY device_A's notification; peers intact."""
        plugin = NotificationPlugin()
        dummy_sender = AsyncMock()
        shared_id = "reminder_42"

        # Register notifications for dev_1, dev_2, and dev_3
        for dev in ["dev_1", "dev_2", "dev_3"]:
            await plugin.handle_packet(
                Packet(
                    type="kdeconnect.notification",
                    body={"id": shared_id, "appName": f"App_{dev}", "title": dev},
                ),
                dev,
                dummy_sender,
            )

        assert len(plugin.list_notifications()) == 3

        # Dismiss notification specifically for dev_2
        await plugin.dismiss_notification(dummy_sender, shared_id, device_id="dev_2")

        # dev_2 removed
        assert plugin.get_notification(shared_id, device_id="dev_2") is None
        # dev_1 and dev_3 remain intact
        assert plugin.get_notification(shared_id, device_id="dev_1") is not None
        assert plugin.get_notification(shared_id, device_id="dev_3") is not None
        assert len(plugin.list_notifications()) == 2

    async def test_remote_cancellation_preserves_peers(self) -> None:
        """Remote cancellation packet (isCancel=True) from device_X removes ONLY device_X's notification."""
        plugin = NotificationPlugin()
        dummy_sender = AsyncMock()
        shared_id = "incoming_call"

        for dev in ["phone_alpha", "phone_beta"]:
            await plugin.handle_packet(
                Packet(
                    type="kdeconnect.notification",
                    body={"id": shared_id, "appName": "Phone", "title": "Call"},
                ),
                dev,
                dummy_sender,
            )

        assert len(plugin.list_notifications()) == 2

        # phone_alpha sends cancel packet
        cancel_pkt = Packet(
            type="kdeconnect.notification",
            body={"id": shared_id, "isCancel": True},
        )
        await plugin.handle_packet(cancel_pkt, "phone_alpha", dummy_sender)

        # phone_alpha cleared, phone_beta intact
        assert plugin.get_notification(shared_id, device_id="phone_alpha") is None
        assert plugin.get_notification(shared_id, device_id="phone_beta") is not None
        assert len(plugin.list_notifications()) == 1

    async def test_device_unpaired_event_purges_only_target_device_state(self) -> None:
        """DeviceUnpaired purges all notification and battery state for target device; peers unaffected."""
        event_bus = EventBus()
        notif_plugin = NotificationPlugin(event_bus=event_bus)
        battery_plugin = BatteryPlugin(event_bus=event_bus)

        await notif_plugin.start()
        await battery_plugin.start()
        dummy_sender = AsyncMock()

        devices = ["dev_red", "dev_blue", "dev_green"]

        # Populate notifications (3 per device = 9 total)
        for dev in devices:
            for n in range(3):
                await notif_plugin.handle_packet(
                    Packet(
                        type="kdeconnect.notification",
                        body={"id": f"msg_{n}", "appName": f"App_{dev}_{n}"},
                    ),
                    dev,
                    dummy_sender,
                )
            # Populate battery
            await battery_plugin.handle_packet(
                Packet(
                    type="kdeconnect.battery",
                    body={"currentCharge": 80, "isCharging": False},
                ),
                dev,
                dummy_sender,
            )

        assert len(notif_plugin.list_notifications()) == 9
        for dev in devices:
            assert battery_plugin.get_device_battery(dev) is not None

        # Unpair dev_blue
        await event_bus.publish(DeviceUnpaired(device_id="dev_blue"))

        # Notifications: dev_blue cleared (0 remaining), dev_red (3) and dev_green (3) remain (total 6)
        assert len(notif_plugin.list_notifications(device_id="dev_blue")) == 0
        assert len(notif_plugin.list_notifications(device_id="dev_red")) == 3
        assert len(notif_plugin.list_notifications(device_id="dev_green")) == 3
        assert len(notif_plugin.list_notifications()) == 6

        # Battery: dev_blue cleared, others remain
        assert battery_plugin.get_device_battery("dev_blue") is None
        assert battery_plugin.get_device_battery("dev_red") is not None
        assert battery_plugin.get_device_battery("dev_green") is not None

        # Unpairing non-existent device is a graceful no-op
        await event_bus.publish(DeviceUnpaired(device_id="dev_nonexistent"))
        assert len(notif_plugin.list_notifications()) == 6

        await notif_plugin.stop()
        await battery_plugin.stop()


# =============================================================================
# 4. CLUS-27: Clipboard Clock Drift Clamping & Multi-Device Convergence
# =============================================================================

class TestClus27ClipboardClockDriftClamping:
    """Stress-test 15-second clock drift window, boundary conditions, and multi-device convergence."""

    def test_clock_drift_exact_boundary_conditions(self) -> None:
        """Exact boundary tests for 15,000ms maximum future clock drift window."""
        engine = LoopSuppressionEngine(max_future_drift_ms=15_000)
        wall_now = 5_000_000
        engine.current_wall_ms = lambda: wall_now

        # Boundary 1: Exactly at limit (now + 15,000ms) -> ACCEPTED
        ok, reason = engine.evaluate_inbound("border text 1", wall_now + 15_000, "peer_a")
        assert ok is True
        assert reason == "accepted"

        # Boundary 2: Limit + 1ms (now + 15,001ms) -> REJECTED
        ok, reason = engine.evaluate_inbound("border text 2", wall_now + 15_001, "peer_a")
        assert ok is False
        assert reason == "future_clock_drift_exceeded"

        # Boundary 3: Old 24-hour timestamp (now + 86,400,000ms) -> REJECTED
        ok, reason = engine.evaluate_inbound("24hr exploit text", wall_now + 86_400_000, "attacker")
        assert ok is False
        assert reason == "future_clock_drift_exceeded"

    def test_record_inbound_accepted_clamps_to_15s_ceiling(self) -> None:
        """record_inbound_accepted clamps local timestamp to now + 15s even if unvalidated timestamp passed."""
        engine = LoopSuppressionEngine(max_future_drift_ms=15_000)
        wall_now = 10_000_000
        engine.current_wall_ms = lambda: wall_now

        # Call record_inbound_accepted with an excessive future timestamp (e.g. +600,000ms)
        engine.record_inbound_accepted("malicious timestamp text", wall_now + 600_000)

        # Assert local timestamp was strictly clamped to wall_now + 15_000
        assert engine.local_timestamp == wall_now + 15_000
        assert engine.local_timestamp < wall_now + 600_000

    def test_multi_device_convergence_after_drift_lockout_expires(self) -> None:
        """Device A (clock +10s ahead) sends packet. After 15s, Device B (accurate clock)

        sends packet and is successfully accepted, confirming no 24-hour lockout.
        """
        engine = LoopSuppressionEngine(max_future_drift_ms=15_000)
        simulated_wall = 100_000

        def _get_wall() -> int:
            return simulated_wall

        engine.current_wall_ms = _get_wall

        # Time T0: Device A (fast by 10s = 10,000ms) sends clipboard
        fast_ts = simulated_wall + 10_000
        ok, reason = engine.evaluate_inbound("Clip from Fast Phone", fast_ts, "phone_fast")
        assert ok is True
        engine.record_inbound_accepted("Clip from Fast Phone", fast_ts)
        assert engine.local_timestamp == fast_ts

        # Time T0 + 1s: Device B (accurate clock) sends clipboard -> rejected as stale
        # because local timestamp is fast_ts (simulated_wall + 10,000)
        simulated_wall += 1_000
        ok, reason = engine.evaluate_inbound("Clip from Normal Phone", simulated_wall, "phone_normal")
        assert ok is False
        assert "stale_timestamp" in reason

        # Time T0 + 16s: 16 seconds have passed. Device B sends fresh clipboard.
        # Now simulated_wall = 100,000 + 16,000 = 116,000 > fast_ts (110,000).
        simulated_wall += 15_000  # Total elapsed: 16,000 ms
        ok, reason = engine.evaluate_inbound("Fresh Clip from Normal Phone", simulated_wall, "phone_normal")

        # Must be ACCEPTED! In the old 24h bug, this would remain rejected for 24 hours.
        assert ok is True
        assert reason == "accepted"

    async def test_clipboard_plugin_enforces_pairing_before_clock_drift_check(self) -> None:
        """ClipboardPlugin rejects unauthenticated / unpaired packets before evaluating clock drift."""
        plugin = ClipboardPlugin(
            backend=MagicMock(),
            is_paired_checker=lambda dev_id: False,  # Unpaired
        )
        plugin._running = True
        dummy_sender = AsyncMock()

        # Unpaired device sends packet with valid timestamp
        packet = Packet(
            type="kdeconnect.clipboard",
            body={"content": "malicious injection", "timestamp": int(time.time() * 1000)},
        )

        res = await plugin.handle_packet(packet, "unpaired_attacker", dummy_sender)
        assert res is True
        # Content was discarded; local suppression engine timestamp was NOT touched
        assert plugin._suppression.local_timestamp == 0

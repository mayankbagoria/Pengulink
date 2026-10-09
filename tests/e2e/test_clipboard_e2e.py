"""tests.e2e.test_clipboard_e2e
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-End integration test suite for KDE Connect Clipboard Synchronization plugin.
Executes live mTLS exchanges between PenguLinkDaemon and MockKdeConnectDevice over loopback.

Covers all 6 required milestone scenarios:
- test_e2e_clipboard_001_linux_to_android: Linux clipboard update propagates to connected mock device.
- test_e2e_clipboard_002_android_to_linux: Mock device packet updates Linux clipboard & emits event.
- test_e2e_clipboard_003_loop_suppression: Remote update never echoes; multi-device originator exclusion.
- test_e2e_clipboard_004_reconnection_sync: State convergence respecting timestamp monotonicity.
- test_e2e_clipboard_005_content_fidelity: Unicode, emojis, multiline strings, and empty strings roundtrip.
- test_e2e_clipboard_006_disconnection_resilience: Drops mid-broadcast, RST recovery, and zero leaks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any, List
import uuid

import pytest

from pengulink.core.events import ClipboardUpdatedEvent, DeviceConnected
from pengulink.core.state_machine import DeviceState
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.network.packet import Packet
from tests.e2e.conftest import e2e_environment, wait_for
from tests.mock.mock_device import MockDevice, MockDeviceConfig


@pytest.fixture(autouse=True)
def hermetic_clipboard_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure all tests strictly use InMemoryClipboardBackend so host clipboard is never touched."""
    monkeypatch.setenv("PENGULINK_CLIPBOARD_BACKEND", "in_memory")


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

    # Publish DeviceConnected on EventBus so all plugins register the new active connection
    await daemon.event_bus.publish(
        DeviceConnected(
            device_id=mock.device_id,
            host="127.0.0.1",
            port=mock.bound_tcp_port or 0,
        )
    )
    await asyncio.sleep(0.05)


# =============================================================================
# Scenario 1: Linux -> Android
# =============================================================================

async def test_e2e_clipboard_001_linux_to_android(tmp_path: Path) -> None:
    """Scenario 1: Setting clipboard locally in PenguLink propagates kdeconnect.clipboard to mock device.

    Verifies:
    1. IPC set_clipboard propagates over mTLS stream to MockKdeConnectDevice.
    2. Simulated external application copy (OS watcher event) propagates to MockKdeConnectDevice.
    3. Payload conforms to KDE Connect schema: body.content matches, body.timestamp is valid epoch ms.
    4. Zero-log privacy canary: raw clipboard secret token never appears in daemon logs.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # 1. Update clipboard via JSON-RPC IPC client
        test_text_ipc = "Hello from Linux Desktop via PenguLink IPC!"
        set_res = await ipc_client.set_clipboard(test_text_ipc)
        assert set_res.get("status") == "ok"
        assert set_res.get("bytes", 0) > 0

        # Verify MockDevice received the kdeconnect.clipboard packet
        await wait_for(
            lambda: any(
                p.type == "kdeconnect.clipboard" and p.body.get("content") == test_text_ipc
                for p in mock.received_packets
            ),
            timeout=4.0,
            desc="mock receiving kdeconnect.clipboard packet via IPC",
        )

        matching_pkts = [
            p for p in mock.received_packets
            if p.type == "kdeconnect.clipboard" and p.body.get("content") == test_text_ipc
        ]
        assert len(matching_pkts) >= 1
        rx_pkt = matching_pkts[0]
        assert rx_pkt.body["content"] == test_text_ipc
        assert isinstance(rx_pkt.body["timestamp"], int)
        assert rx_pkt.body["timestamp"] > 0

        # 2. Simulate external application copy event via local backend watcher
        test_text_os = "Copied text from Firefox browser application"
        await daemon.clipboard_plugin.backend.simulate_external_copy(test_text_os)

        await wait_for(
            lambda: any(
                p.type == "kdeconnect.clipboard" and p.body.get("content") == test_text_os
                for p in mock.received_packets
            ),
            timeout=4.0,
            desc="mock receiving kdeconnect.clipboard packet via OS watcher event",
        )

        # 3. Privacy Safeguard: Verify zero-log privacy rule with secret canary token
        canary_secret = f"CANARY_SECRET_{uuid.uuid4().hex[:16]}"
        await ipc_client.set_clipboard(canary_secret)

        await wait_for(
            lambda: any(
                p.type == "kdeconnect.clipboard" and p.body.get("content") == canary_secret
                for p in mock.received_packets
            ),
            timeout=4.0,
            desc="mock receiving canary clipboard packet",
        )

        # Ensure raw canary secret is strictly redacted from daemon logs
        logs = await ipc_client.get_logs()
        for log_entry in logs:
            msg = log_entry.get("message", "")
            assert canary_secret not in msg, f"Leak detected: canary token found in log: {msg}"


# =============================================================================
# Scenario 2: Android -> Linux
# =============================================================================

async def test_e2e_clipboard_002_android_to_linux(tmp_path: Path) -> None:
    """Scenario 2: Mock device sending kdeconnect.clipboard updates local daemon clipboard and triggers event.

    Verifies:
    1. Inbound kdeconnect.clipboard packet from MockDevice is accepted by daemon.
    2. ClipboardUpdatedEvent is dispatched to EventBus subscribers with accurate fields.
    3. Daemon local backend is updated with the received text.
    4. Daemon JSON-RPC get_clipboard IPC query returns the updated content and metadata.
    """
    events: List[ClipboardUpdatedEvent] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        daemon.event_bus.subscribe(
            ClipboardUpdatedEvent,
            lambda ev: events.append(ev),
        )

        remote_clip = "Hello from Android Mobile Phone! Verification Code: 492018."
        remote_ts = int(time.time() * 1000) + 2000

        # Mock sends kdeconnect.clipboard packet
        await mock.send_clipboard(content=remote_clip, timestamp=remote_ts)

        # Wait for ClipboardUpdatedEvent on daemon EventBus
        await wait_for(
            lambda: len(events) > 0,
            timeout=4.0,
            desc="daemon receiving ClipboardUpdatedEvent",
        )

        ev = events[0]
        assert ev.device_id == mock.device_id
        assert ev.content == remote_clip
        assert ev.timestamp == remote_ts

        # Verify local backend was mutated
        local_text = await daemon.clipboard_plugin.backend.get_text()
        assert local_text == remote_clip

        # Verify JSON-RPC IPC query returns the newly received clipboard
        ipc_clip = await ipc_client.get_clipboard()
        assert ipc_clip["content"] == remote_clip
        assert ipc_clip["timestamp"] == remote_ts


# =============================================================================
# Scenario 3: Loop Suppression
# =============================================================================

async def test_e2e_clipboard_003_loop_suppression(tmp_path: Path) -> None:
    """Scenario 3: Remote update applied to local daemon never echoes back to originating mock device.

    Verifies:
    1. Single Device Echo Suppression: Mock device sends clipboard update -> Daemon updates local clipboard ->
       Mock device receives ZERO echo packets back.
    2. Multi-Device Originator Exclusion: Mock A sends update -> Mock B receives update -> Mock A receives ZERO echo packets.
    3. Write-in-flight TTL suppression: Local OS backend reflection event within 500ms window does not emit duplicate packets.
    """
    async with e2e_environment(tmp_path) as (daemon, mock_a, ipc_client):
        # Create second independent mock device
        m_cfg_b = MockDeviceConfig(
            device_id=f"mock_device_b_{uuid.uuid4().hex[:12]}",
            device_name="Mock Android Phone B",
            auto_pair=True,
        )
        mock_b = MockDevice(config=m_cfg_b)
        await mock_b.start(listen_tcp=True, port=0)

        try:
            await _pair_and_connect(daemon, mock_a)
            await _pair_and_connect(daemon, mock_b)

            # Part 1 & 2: Mock A transmits clipboard update
            clip_content = "Multi-Device Sync Payload 999"
            clip_ts = int(time.time() * 1000) + 1500

            await mock_a.send_clipboard(content=clip_content, timestamp=clip_ts)

            # Wait for Mock B to receive the fanned out update
            await wait_for(
                lambda: any(
                    p.type == "kdeconnect.clipboard" and p.body.get("content") == clip_content
                    for p in mock_b.received_packets
                ),
                timeout=4.0,
                desc="mock_b receiving fanned-out clipboard update",
            )

            # Allow event loop settling time for any potential echoes
            await asyncio.sleep(0.3)

            # Verify Mock A (the originator) received ZERO echo packets
            current_mock_a_clips = len([
                p for p in mock_a.received_packets
                if p.type == "kdeconnect.clipboard" and p.body.get("content") == clip_content
            ])
            assert current_mock_a_clips == 0, "Loop suppression failure: Mock A received echo of its own clip!"

            # Part 3: Verify local backend state and Layer 2 Write-in-flight protection
            assert await daemon.clipboard_plugin.backend.get_text() == clip_content

            # Redundant immediate duplicate send from daemon is suppressed by Layer 1 LRU cache
            # Simulate local OS watcher firing with identical content
            await daemon.clipboard_plugin.backend.simulate_external_copy(clip_content)
            await asyncio.sleep(0.2)

            # Mock B should still have received only 1 packet with clip_content
            mock_b_matching = [
                p for p in mock_b.received_packets
                if p.type == "kdeconnect.clipboard" and p.body.get("content") == clip_content
            ]
            assert len(mock_b_matching) == 1, "Duplicate copy was not suppressed by Layer 1 LRU cache!"

        finally:
            if mock_b.is_running:
                await mock_b.stop()


# =============================================================================
# Scenario 4: Reconnection Synchronization
# =============================================================================

async def test_e2e_clipboard_004_reconnection_sync(tmp_path: Path) -> None:
    """Scenario 4: Freshly connected / reconnected mock device receives freshest clipboard state respecting monotonicity.

    Verifies:
    1. Pre-connection state synchronization: Daemon with local clipboard synchronizes with freshly connecting mock.
    2. Stale remote update rejection: Stale timestamp on connect packet is rejected, and daemon counter-pushes fresher state.
    3. Newer remote update acceptance: Newer timestamp on connect packet is accepted and updates local clipboard.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        # 1. Seed daemon clipboard before peer is connected
        initial_desktop_clip = "Pre-connection Desktop Content"
        set_ok = await daemon.set_clipboard(initial_desktop_clip)
        assert set_ok is True

        # Now pair and connect mock device
        await _pair_and_connect(daemon, mock)

        # Newly connected peer should receive kdeconnect.clipboard.connect packet with latest local state
        await wait_for(
            lambda: any(
                p.type == "kdeconnect.clipboard.connect" and p.body.get("content") == initial_desktop_clip
                for p in mock.received_packets
            ),
            timeout=4.0,
            desc="mock receiving initial state sync on connection",
        )

        local_ts = daemon.clipboard_plugin.latest_timestamp
        assert local_ts > 0

        # 2. Stale Remote Update: Mock sends connect packet with older timestamp (local_ts - 5000)
        stale_ts = local_ts - 5000
        stale_content = "Stale Mobile Content from yesterday"
        initial_connect_count = len(mock.get_received_packets("kdeconnect.clipboard.connect"))

        await mock.send_clipboard_connect(content=stale_content, timestamp=stale_ts)

        # Daemon should reject stale content and counter-push fresher local state back to mock
        await wait_for(
            lambda: len(mock.get_received_packets("kdeconnect.clipboard.connect")) > initial_connect_count,
            timeout=4.0,
            desc="daemon counter-pushing fresher state upon stale connect packet",
        )

        # Local daemon must still retain its fresher initial content
        assert await daemon.clipboard_plugin.backend.get_text() == initial_desktop_clip

        # 3. Newer Remote Update: Mock sends connect packet with newer timestamp (local_ts + 5000)
        newer_ts = local_ts + 5000
        newer_content = "Newer Mobile Content from Phone"
        await mock.send_clipboard_connect(content=newer_content, timestamp=newer_ts)

        # Daemon accepts newer update
        await wait_for(
            lambda: daemon.clipboard_plugin.last_content == newer_content,
            timeout=4.0,
            desc="daemon accepting newer connect update",
        )
        assert await daemon.clipboard_plugin.backend.get_text() == newer_content


# =============================================================================
# Scenario 5: Content Fidelity
# =============================================================================

async def test_e2e_clipboard_005_content_fidelity(tmp_path: Path) -> None:
    """Scenario 5: Exact Unicode, emojis, multiline strings, and empty strings roundtrip cleanly.

    Verifies:
    1. Diverse Unicode scripts (Arabic, Hebrew, Cyrillic, CJK) and complex multi-byte emojis.
    2. Complex multiline text with tabs, CRLF, and trailing whitespace.
    3. JSON code snippets with quotation and backslash escapes.
    4. Large text payload (~50 KB) under the 120 KB limit.
    5. Empty string handling: set_clipboard("") updates backend without broadcasting empty wire packet.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        test_payloads = [
            # 1. Unicode & Emojis
            "🎉 PenguLink 🐧 漢字, العربية, עִברִית, русский язык, 💻✨ 🔥🚀",
            # 2. Complex Multiline & Whitespace
            "Header Title\n\n\tIndented Item 1\r\n\tIndented Item 2\nTrailing whitespace   \nEnd of block.",
            # 3. Escaped JSON & Code
            '{"type": "source_code", "data": "def hello():\\n    print(\\"Hello, world!\\")\\n", "html": "<div id=\'test\'>&amp;</div>"}',
            # 4. Large text payload (~50 KB)
            ("The quick brown fox jumps over the lazy dog. 🦊 " * 1000),
        ]

        # Test each payload roundtrip: Linux -> Android -> Linux
        for idx, payload in enumerate(test_payloads):
            # A. Linux -> Android
            await daemon.set_clipboard(payload)
            await wait_for(
                lambda p=payload: any(
                    p_pkt.type == "kdeconnect.clipboard" and p_pkt.body.get("content") == p
                    for p_pkt in mock.received_packets
                ),
                timeout=4.0,
                desc=f"payload {idx} reaching mock device",
            )
            rx_mock = [p for p in mock.received_packets if p.body.get("content") == payload][0]
            assert rx_mock.body["content"] == payload

            # B. Android -> Linux
            remote_reply = f"{payload} [MODIFIED BY ANDROID]"
            reply_ts = daemon.clipboard_plugin.latest_timestamp + 1000
            await mock.send_clipboard(content=remote_reply, timestamp=reply_ts)

            await wait_for(
                lambda r=remote_reply: daemon.clipboard_plugin.last_content == r,
                timeout=4.0,
                desc=f"payload {idx} modified reply reaching daemon",
            )
            assert await daemon.clipboard_plugin.backend.get_text() == remote_reply

        # 5. Empty string handling
        # Local set empty string: updates backend without wire broadcast
        wire_count_before = len(mock.get_received_packets("kdeconnect.clipboard"))
        await daemon.set_clipboard("")
        assert await daemon.clipboard_plugin.backend.get_text() == ""
        await asyncio.sleep(0.2)
        wire_count_after = len(mock.get_received_packets("kdeconnect.clipboard"))
        assert wire_count_after == wire_count_before, "Empty string should not broadcast outbound packet"

        # Inbound empty string: accepted and clears local backend
        empty_ts = daemon.clipboard_plugin.latest_timestamp + 2000
        await mock.send_clipboard("", timestamp=empty_ts)
        await wait_for(
            lambda: daemon.clipboard_plugin.last_content == "",
            timeout=4.0,
            desc="daemon accepting inbound empty string",
        )
        assert await daemon.clipboard_plugin.backend.get_text() == ""


# =============================================================================
# Scenario 6: Disconnection Resilience
# =============================================================================

async def test_e2e_clipboard_006_disconnection_resilience(tmp_path: Path) -> None:
    """Scenario 6: Abrupt disconnection during active transmission handled gracefully without leaks or crashes.

    Verifies:
    1. Abrupt TCP drop via SO_LINGER (1, 0) during/prior to clipboard broadcast does not crash daemon.
    2. Truncated packet stream handled cleanly at EOF without unhandled exceptions.
    3. Daemon successfully accepts reconnection, restores state machine to CONNECTED, and resumes clipboard sync.
    4. Zero lingering tasks, zero leaked sockets, clean shutdown.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # 1. Transmit initial clipboard update to establish active state
        await daemon.set_clipboard("Before disconnect message")
        await wait_for(
            lambda: any(
                p.type == "kdeconnect.clipboard" and p.body.get("content") == "Before disconnect message"
                for p in mock.received_packets
            ),
            timeout=4.0,
            desc="mock receiving before-disconnect message",
        )

        # 2. Simulate abrupt TCP RST disconnection by peer
        await mock.drop_connection(reset=True)

        # Daemon read loop must catch connection reset and transition to DISCONNECTED
        await wait_for(
            lambda: not daemon.is_device_connected(mock.device_id),
            timeout=3.0,
            desc="daemon detecting abrupt disconnection",
        )
        assert daemon._state_machines[mock.device_id].current_state == DeviceState.DISCONNECTED

        # Broadcast during disconnected state: safely ignored by daemon without throwing or crashing
        await daemon.set_clipboard("Message while disconnected")
        assert daemon.is_running is True

        # 3. Reconnect device, then simulate peer transmitting truncated stream
        conn = await daemon.connect_device(mock.device_id, port=mock.bound_tcp_port)
        assert daemon.is_device_connected(mock.device_id) is True

        # Transmit partial JSON without newline and drop socket
        await mock.send_truncated_packet(b'{"type":"kdeconnect.clipboard","body":{"content":"incomplete')

        await wait_for(
            lambda: not daemon.is_device_connected(mock.device_id),
            timeout=3.0,
            desc="handling truncated packet EOF",
        )
        assert daemon.is_running is True

        # 4. Clean Reconnection and State Recovery
        conn = await daemon.connect_device(mock.device_id, port=mock.bound_tcp_port)
        assert not conn.is_closed
        assert daemon.is_device_connected(mock.device_id) is True

        # Clipboard synchronization resumes normally
        post_reconnect_msg = "Post-reconnect Synchronization Succeeded!"
        await daemon.set_clipboard(post_reconnect_msg)

        await wait_for(
            lambda: any(
                p.type == "kdeconnect.clipboard" and p.body.get("content") == post_reconnect_msg
                for p in mock.received_packets
            ),
            timeout=4.0,
            desc="mock receiving post-reconnect clipboard packet",
        )

        matching_pkts = [
            p for p in mock.received_packets
            if p.type == "kdeconnect.clipboard" and p.body.get("content") == post_reconnect_msg
        ]
        assert len(matching_pkts) >= 1
        assert matching_pkts[0].body["content"] == post_reconnect_msg

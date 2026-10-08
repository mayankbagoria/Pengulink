"""tests.unit.test_plugin_clipboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive unit test suite for ClipboardPlugin and 4-Layer Loop Suppression Engine.

Verifies:
- Inbound kdeconnect.clipboard & kdeconnect.clipboard.connect packet handling
- Outbound local clipboard change detection and fanout
- Layer 1: Rolling SHA-256 digest LRU cache (deduplication, capacity 32, eviction)
- Layer 2: Write-in-flight echo suppression with monotonic TTL & CRLF normalization
- Layer 3: Millisecond timestamp monotonicity (stale drop, drift ceiling, forward advance)
- Layer 4: Multi-device fanout with originator device ID exclusion
- 120 KB text payload ceiling guard (MAX_CLIPBOARD_PACKET_BYTES)
- UTF-8 Unicode, emojis, multiline text, and empty string handling
- Reconnection state synchronization on DeviceConnected
- Strict zero-log privacy (redact_clipboard & ClipboardUpdatedEvent __repr__/__str__)
- Complete plugin lifecycle management (start/stop idempotency, task drain)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
import pytest

from pengulink.core.events import (
    ClipboardUpdatedEvent,
    DeviceConnected,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
)
from pengulink.network.packet import Packet
from pengulink.plugins.base import PluginRegistry
from pengulink.plugins.clipboard import (
    CLIPBOARD_CONNECT_PACKET_TYPE,
    CLIPBOARD_PACKET_TYPE,
    DEFAULT_WRITE_IN_FLIGHT_TTL,
    MAX_CLIPBOARD_PACKET_BYTES,
    MAX_FUTURE_CLOCK_DRIFT_MS,
    ClipboardPlugin,
    InMemoryClipboardBackend,
    LoopSuppressionEngine,
    RollingDigestLRUCache,
    create_clipboard_connect_packet,
    create_clipboard_packet,
    redact_clipboard,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def memory_backend() -> InMemoryClipboardBackend:
    return InMemoryClipboardBackend()


@pytest.fixture
def plugin(event_bus: EventBus, memory_backend: InMemoryClipboardBackend) -> ClipboardPlugin:
    return ClipboardPlugin(
        event_bus=event_bus,
        backend=memory_backend,
        ttl_seconds=0.5,
        digest_cache_size=32,
    )


# =============================================================================
# 1. Plugin Capabilities & Registry Integration
# =============================================================================

class TestPluginCapabilitiesAndRegistry:
    def test_plugin_properties(self, plugin: ClipboardPlugin) -> None:
        assert plugin.plugin_id == "clipboard"
        assert CLIPBOARD_PACKET_TYPE in plugin.incoming_capabilities
        assert CLIPBOARD_CONNECT_PACKET_TYPE in plugin.incoming_capabilities
        assert CLIPBOARD_PACKET_TYPE in plugin.outgoing_capabilities
        assert CLIPBOARD_CONNECT_PACKET_TYPE in plugin.outgoing_capabilities
        assert not plugin.is_running
        assert plugin.latest_timestamp == 0

    async def test_plugin_registry_dispatch(
        self, event_bus: EventBus, memory_backend: InMemoryClipboardBackend
    ) -> None:
        registry = PluginRegistry(event_bus)
        plugin = ClipboardPlugin(event_bus=event_bus, backend=memory_backend)
        registry.register(plugin)
        await registry.start_all()

        all_incoming = registry.get_all_incoming_capabilities()
        assert CLIPBOARD_PACKET_TYPE in all_incoming
        assert CLIPBOARD_CONNECT_PACKET_TYPE in all_incoming

        sent: List[Packet] = []

        async def dummy_send(p: Packet) -> None:
            sent.append(p)

        packet = Packet(
            id=1000,
            type=CLIPBOARD_PACKET_TYPE,
            body={"content": "Synced via registry", "timestamp": 1000},
        )
        handled = await registry.dispatch_packet(packet, "phone_1", dummy_send)
        assert handled is True
        assert await memory_backend.get_text() == "Synced via registry"
        await registry.stop_all()

    def test_default_backend_detection(self) -> None:
        # Initializing without explicit backend should fallback safely to detected backend
        p = ClipboardPlugin()
        assert p.backend is not None
        assert p.backend.is_available


# =============================================================================
# 2. Inbound Packet Handling
# =============================================================================

class TestInboundPacketHandling:
    async def test_inbound_clipboard_packet_updates_backend_and_emits_event(
        self, plugin: ClipboardPlugin, event_bus: EventBus, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()
        events: List[ClipboardUpdatedEvent] = []

        def on_event(ev: ClipboardUpdatedEvent) -> None:
            events.append(ev)

        event_bus.subscribe(ClipboardUpdatedEvent, on_event)

        sent: List[Packet] = []

        async def send_cb(p: Packet) -> None:
            sent.append(p)

        packet = create_clipboard_packet("Hello from Android", timestamp=1000)
        await plugin.handle_packet(packet, device_id="android_dev", send_packet=send_cb)

        # Assert local clipboard changed
        assert await memory_backend.get_text() == "Hello from Android"
        assert plugin.latest_timestamp == 1000
        assert plugin.last_content == "Hello from Android"

        # Assert EventBus event emitted with correct data
        assert len(events) == 1
        assert events[0].device_id == "android_dev"
        assert events[0].content == "Hello from Android"
        assert events[0].timestamp == 1000
        await plugin.stop()

    async def test_inbound_missing_timestamp_fallback_to_packet_id(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()
        sent: List[Packet] = []

        async def send_cb(p: Packet) -> None:
            sent.append(p)

        packet = Packet(
            id=54321,
            type=CLIPBOARD_PACKET_TYPE,
            body={"content": "No timestamp in body"},
        )
        await plugin.handle_packet(packet, device_id="legacy_dev", send_packet=send_cb)
        assert await memory_backend.get_text() == "No timestamp in body"
        assert plugin.latest_timestamp == 54321
        await plugin.stop()

    async def test_inbound_connect_packet_sync(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()
        sent: List[Packet] = []

        async def send_cb(p: Packet) -> None:
            sent.append(p)

        connect_pkt = create_clipboard_connect_packet("Initial connect sync text", timestamp=2000)
        await plugin.handle_packet(connect_pkt, device_id="peer_dev", send_packet=send_cb)

        assert await memory_backend.get_text() == "Initial connect sync text"
        assert plugin.latest_timestamp == 2000
        await plugin.stop()

    async def test_inbound_connect_packet_stale_triggers_local_push(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()
        # Local has newer content
        plugin.suppression.local_timestamp = 5000
        plugin._last_content = "Fresh Linux Content"
        await memory_backend.set_text("Fresh Linux Content")

        sent: List[Packet] = []

        async def send_cb(p: Packet) -> None:
            sent.append(p)

        # Peer sends older connect packet
        stale_connect_pkt = create_clipboard_connect_packet("Old Phone Text", timestamp=3000)
        await plugin.handle_packet(stale_connect_pkt, device_id="phone_1", send_packet=send_cb)

        # Local content should NOT be overwritten
        assert await memory_backend.get_text() == "Fresh Linux Content"

        # Allow background push task to execute
        await asyncio.sleep(0.02)

        # Peer should receive convergence push of local clipboard
        assert len(sent) == 1
        assert sent[0].type == CLIPBOARD_CONNECT_PACKET_TYPE
        assert sent[0].body["content"] == "Fresh Linux Content"
        assert sent[0].body["timestamp"] == 5000
        await plugin.stop()

    async def test_inbound_invalid_or_zero_timestamp_rejected(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()
        sent: List[Packet] = []

        async def send_cb(p: Packet) -> None:
            sent.append(p)

        # Timestamp 0 rejected
        pkt_zero = Packet(
            id=0,
            type=CLIPBOARD_PACKET_TYPE,
            body={"content": "Should be rejected", "timestamp": 0},
        )
        await plugin.handle_packet(pkt_zero, device_id="dev_1", send_packet=send_cb)
        assert await memory_backend.get_text() == ""

        # Negative timestamp in body rejected
        pkt_neg = Packet(
            id=1,
            type=CLIPBOARD_PACKET_TYPE,
            body={"content": "Negative ts", "timestamp": -5},
        )
        await plugin.handle_packet(pkt_neg, device_id="dev_1", send_packet=send_cb)
        assert await memory_backend.get_text() == ""
        await plugin.stop()


# =============================================================================
# 3. Outbound Clipboard Dispatch
# =============================================================================

class TestOutboundClipboardDispatch:
    async def test_outbound_local_copy_broadcasts_packet(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()

        sent_dev1: List[Packet] = []
        sent_dev2: List[Packet] = []

        async def send1(p: Packet) -> None:
            sent_dev1.append(p)

        async def send2(p: Packet) -> None:
            sent_dev2.append(p)

        plugin.add_connected_device("dev_1", send1)
        plugin.add_connected_device("dev_2", send2)

        # Simulate user copying text on Linux desktop
        await memory_backend.simulate_external_copy("Copied from Firefox")

        # Allow async fanout tasks to execute
        await asyncio.sleep(0.02)

        assert len(sent_dev1) == 1
        assert sent_dev1[0].type == CLIPBOARD_PACKET_TYPE
        assert sent_dev1[0].body["content"] == "Copied from Firefox"
        assert sent_dev1[0].body["timestamp"] > 0

        assert len(sent_dev2) == 1
        assert sent_dev2[0].body["content"] == "Copied from Firefox"

        await plugin.stop()

    async def test_outbound_empty_string_suppressed(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()

        sent: List[Packet] = []

        async def send_cb(p: Packet) -> None:
            sent.append(p)

        plugin.add_connected_device("dev_1", send_cb)

        # Simulate empty clipboard copy (cleared selection)
        await memory_backend.simulate_external_copy("")
        await asyncio.sleep(0.02)

        # Outbound empty string should not be broadcast
        assert len(sent) == 0

        await plugin.stop()

    async def test_outbound_timestamp_monotonically_advances(
        self, plugin: ClipboardPlugin
    ) -> None:
        # Seed local timestamp in the future relative to wall-clock
        now_ms = plugin.suppression.current_wall_ms()
        future_base = now_ms + 10_000
        plugin.suppression.local_timestamp = future_base

        digest, assigned_ts = plugin.suppression.record_outbound_accepted("Local copy 1")
        assert assigned_ts == future_base + 1

        digest2, assigned_ts2 = plugin.suppression.record_outbound_accepted("Local copy 2")
        assert assigned_ts2 == future_base + 2


# =============================================================================
# 4. Layer 1: Rolling SHA-256 Digest LRU Cache
# =============================================================================

class TestLayer1RollingDigestLRU:
    def test_lru_cache_capacity_and_eviction(self) -> None:
        cache = RollingDigestLRUCache(capacity=32)

        # Insert 32 items
        for i in range(32):
            cache.add_text(f"item_{i}")
        assert len(cache) == 32
        assert cache.contains_text("item_0")
        assert cache.contains_text("item_31")

        # Insert 33rd item -> oldest LRU element should be evicted
        # Note: item_0 was accessed above via contains_text, so item_1 is now oldest!
        cache.add_text("item_32")
        assert len(cache) == 32
        assert not cache.contains_text("item_1")
        assert cache.contains_text("item_0")
        assert cache.contains_text("item_32")

    def test_lru_cache_recency_refresh(self) -> None:
        cache = RollingDigestLRUCache(capacity=3)
        cache.add_text("A")
        cache.add_text("B")
        cache.add_text("C")

        # Refresh "A" by checking contains
        assert cache.contains_text("A")

        # Add "D" -> "B" should be evicted, not "A"
        cache.add_text("D")
        assert not cache.contains_text("B")
        assert cache.contains_text("A")
        assert cache.contains_text("C")
        assert cache.contains_text("D")

    async def test_duplicate_outbound_suppressed(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()

        sent: List[Packet] = []

        async def send_cb(p: Packet) -> None:
            sent.append(p)

        plugin.add_connected_device("dev_1", send_cb)

        # Copy text once
        await memory_backend.simulate_external_copy("Duplicate Test")
        await asyncio.sleep(0.02)
        assert len(sent) == 1

        # Copy exact same text again -> Layer 1 deduplication suppresses
        await memory_backend.simulate_external_copy("Duplicate Test")
        await asyncio.sleep(0.02)
        assert len(sent) == 1

        # Copy new text -> sent
        await memory_backend.simulate_external_copy("New Different Text")
        await asyncio.sleep(0.02)
        assert len(sent) == 2

        await plugin.stop()

    async def test_duplicate_inbound_suppressed(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()
        sent: List[Packet] = []

        async def send_cb(p: Packet) -> None:
            sent.append(p)

        pkt1 = create_clipboard_packet("Inbound duplicate", timestamp=1000)
        await plugin.handle_packet(pkt1, "phone_1", send_cb)
        assert await memory_backend.get_text() == "Inbound duplicate"

        # Second packet with same content and newer timestamp
        pkt2 = create_clipboard_packet("Inbound duplicate", timestamp=2000)
        accepted, reason = plugin.suppression.evaluate_inbound("Inbound duplicate", 2000, "phone_1")
        assert not accepted
        assert "duplicate_content" in reason
        await plugin.stop()


# =============================================================================
# 5. Layer 2: Write-in-Flight Echo Suppression
# =============================================================================

class TestLayer2WriteInFlight:
    async def test_echo_reflection_suppressed_when_remote_writes_locally(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()

        sent_to_phone: List[Packet] = []

        async def phone_sender(p: Packet) -> None:
            sent_to_phone.append(p)

        plugin.add_connected_device("phone_1", phone_sender)

        # Inbound packet arrives from phone
        pkt = create_clipboard_packet("Remote Secret Clip", timestamp=1000)
        await plugin.handle_packet(pkt, "phone_1", phone_sender)

        assert await memory_backend.get_text() == "Remote Secret Clip"

        # Now simulate the OS backend reflection callback firing as a result of set_text
        await memory_backend.simulate_external_copy("Remote Secret Clip")
        await asyncio.sleep(0.02)

        # Assert no reflection packet was sent back to phone
        assert len(sent_to_phone) == 0

        await plugin.stop()

    def test_write_in_flight_expires_after_ttl(self) -> None:
        engine = LoopSuppressionEngine(write_in_flight_ttl=0.05)
        engine.arm_write_in_flight("Expiring Text", ttl=0.05)

        assert engine.is_write_in_flight_active("Expiring Text")

        time.sleep(0.08)

        # Window expired
        assert not engine.is_write_in_flight_active("Expiring Text")

    def test_write_in_flight_allows_different_content_during_window(self) -> None:
        engine = LoopSuppressionEngine(write_in_flight_ttl=1.0)
        engine.arm_write_in_flight("Remote Content", ttl=1.0)

        # User copies something different
        assert not engine.is_write_in_flight_active("User Different Copy")

    def test_crlf_normalization_matching(self) -> None:
        engine = LoopSuppressionEngine(write_in_flight_ttl=1.0)
        # Remote sent Windows CRLF
        engine.arm_write_in_flight("Multi\r\nLine\r\nText\r\n", ttl=1.0)

        # Linux OS watcher reflects normalized LF
        assert engine.is_write_in_flight_active("Multi\nLine\nText")


# =============================================================================
# 6. Layer 3: Timestamp Monotonicity
# =============================================================================

class TestLayer3TimestampMonotonicity:
    def test_stale_inbound_packets_rejected(self) -> None:
        engine = LoopSuppressionEngine()
        engine.local_timestamp = 2000

        # Older timestamp rejected
        accepted, reason = engine.evaluate_inbound("Text 1", 1999, "dev_1")
        assert not accepted
        assert "stale_timestamp" in reason

        # Equal timestamp rejected
        accepted, reason = engine.evaluate_inbound("Text 2", 2000, "dev_1")
        assert not accepted
        assert "stale_timestamp" in reason

        # Strictly newer accepted
        accepted, reason = engine.evaluate_inbound("Text 3", 2001, "dev_1")
        assert accepted

    def test_future_clock_drift_ceiling_rejected(self) -> None:
        engine = LoopSuppressionEngine(max_future_drift_ms=86_400_000)
        now_ms = int(time.time() * 1000)

        # Timestamp 48 hours into future rejected as clock poison attack
        future_ts = now_ms + (48 * 3600 * 1000)
        accepted, reason = engine.evaluate_inbound("Poison Clip", future_ts, "dev_1")
        assert not accepted
        assert "future_clock_drift_exceeded" in reason

    def test_local_clock_advancement_monotonicity(self) -> None:
        engine = LoopSuppressionEngine()
        future_base = engine.current_wall_ms() + 50_000
        engine.local_timestamp = future_base

        # When local user copies, clock must not step backwards
        _, assigned = engine.record_outbound_accepted("Local clip")
        assert assigned == future_base + 1


# =============================================================================
# 7. Layer 4: Multi-Device Originator Exclusion
# =============================================================================

class TestLayer4OriginatorExclusion:
    async def test_packet_from_dev_a_forwarded_to_dev_b_and_c_not_dev_a(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()
        sent_a: List[Packet] = []
        sent_b: List[Packet] = []
        sent_c: List[Packet] = []

        async def send_a(p: Packet) -> None:
            sent_a.append(p)

        async def send_b(p: Packet) -> None:
            sent_b.append(p)

        async def send_c(p: Packet) -> None:
            sent_c.append(p)

        plugin.add_connected_device("dev_a", send_a)
        plugin.add_connected_device("dev_b", send_b)
        plugin.add_connected_device("dev_c", send_c)

        # Dev A sends a clipboard update
        pkt = create_clipboard_packet("Fanout Broadcast Test", timestamp=1000)
        await plugin.handle_packet(pkt, "dev_a", send_a)

        # Allow fanout tasks to complete
        await asyncio.sleep(0.02)

        # Dev A must NOT receive its own packet reflected back!
        assert len(sent_a) == 0

        # Dev B and Dev C must receive the forwarded packet
        assert len(sent_b) == 1
        assert sent_b[0].body["content"] == "Fanout Broadcast Test"

        assert len(sent_c) == 1
        assert sent_c[0].body["content"] == "Fanout Broadcast Test"
        await plugin.stop()

    def test_capability_filtering(self) -> None:
        devices = {"phone": None, "tablet": None, "desktop": None}

        def caps_check(dev_id: str, cap: str) -> bool:
            return dev_id != "desktop"  # desktop lacks clipboard cap

        recipients = LoopSuppressionEngine.get_fanout_recipients(
            devices,
            originator_device_id="phone",
            capability_checker=caps_check,
            required_capability=CLIPBOARD_PACKET_TYPE,
        )
        assert recipients == ["tablet"]


# =============================================================================
# 8. Payload Limit Enforcement (120 KB Ceiling)
# =============================================================================

class TestPayloadLimitEnforcement:
    async def test_oversized_inbound_payload_rejected(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()
        sent: List[Packet] = []

        async def send_cb(p: Packet) -> None:
            sent.append(p)

        # 125 KB string
        oversized = "A" * (125 * 1024)
        pkt = create_clipboard_packet(oversized, timestamp=1000)
        await plugin.handle_packet(pkt, "phone_1", send_cb)

        # Backend should remain untouched
        assert await memory_backend.get_text() == ""
        await plugin.stop()

    async def test_oversized_outbound_payload_rejected(
        self, plugin: ClipboardPlugin
    ) -> None:
        await plugin.start()
        oversized = "B" * 120_001
        success = await plugin.set_local_clipboard(oversized)
        assert success is False
        await plugin.stop()

    async def test_payload_boundary_120k_accepted(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()
        sent: List[Packet] = []

        async def send_cb(p: Packet) -> None:
            sent.append(p)

        exact_120k = "C" * MAX_CLIPBOARD_PACKET_BYTES
        pkt = create_clipboard_packet(exact_120k, timestamp=1000)
        await plugin.handle_packet(pkt, "phone_1", send_cb)

        assert await memory_backend.get_text() == exact_120k
        await plugin.stop()


# =============================================================================
# 9. UTF-8 Unicode, Emojis, Multiline, and Empty Content
# =============================================================================

class TestUnicodeAndContentEdgeCases:
    async def test_utf8_emojis_and_multilingual_roundtrip(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()
        unicode_text = "PenguLink 🐧🔗🚀 — 中文, Русский, 日本語, العربية, עברית"
        sent: List[Packet] = []

        async def send_cb(p: Packet) -> None:
            sent.append(p)

        pkt = create_clipboard_packet(unicode_text, timestamp=1000)
        await plugin.handle_packet(pkt, "phone_1", send_cb)

        assert await memory_backend.get_text() == unicode_text
        await plugin.stop()

    async def test_multiline_newlines_and_tabs(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()
        multiline_text = "def hello():\n\tprint('line 1')\r\n\treturn True\n"
        sent: List[Packet] = []

        async def send_cb(p: Packet) -> None:
            sent.append(p)

        pkt = create_clipboard_packet(multiline_text, timestamp=1000)
        await plugin.handle_packet(pkt, "phone_1", send_cb)

        assert await memory_backend.get_text() == multiline_text
        await plugin.stop()

    async def test_empty_string_inbound_handling(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()
        sent: List[Packet] = []

        async def send_cb(p: Packet) -> None:
            sent.append(p)

        pkt = create_clipboard_packet("", timestamp=1000)
        await plugin.handle_packet(pkt, "phone_1", send_cb)

        # Inbound empty string accepted without crashing
        assert await memory_backend.get_text() == ""
        await plugin.stop()


# =============================================================================
# 10. Malformed Packet Resilience
# =============================================================================

class TestMalformedPacketResilience:
    async def test_non_dict_body(self, plugin: ClipboardPlugin) -> None:
        await plugin.start()
        class FakePacket:
            id = 1
            type = CLIPBOARD_PACKET_TYPE
            body = "not a dict"

        # Must not raise exception
        await plugin.handle_packet(FakePacket(), "dev_1", lambda p: None)  # type: ignore
        await plugin.stop()

    async def test_non_string_content(self, plugin: ClipboardPlugin) -> None:
        await plugin.start()
        for bad_content in [12345, True, ["list"], {"dict": 1}, None]:
            pkt = Packet(
                id=1,
                type=CLIPBOARD_PACKET_TYPE,
                body={"content": bad_content, "timestamp": 1000},
            )
            await plugin.handle_packet(pkt, "dev_1", lambda p: None)
            assert plugin.last_content == ""
        await plugin.stop()


# =============================================================================
# 11. Reconnection Convergence
# =============================================================================

class TestReconnectionConvergence:
    async def test_reconnection_sync_on_device_connected(
        self, plugin: ClipboardPlugin, event_bus: EventBus, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await plugin.start()

        # Seed local clipboard
        await plugin.set_local_clipboard("Latest Local Edit")

        sent: List[Packet] = []

        async def dev_sender(p: Packet) -> None:
            sent.append(p)

        plugin.set_packet_sender(lambda did, p: sent.append(p))

        # Device connects
        await event_bus.publish(DeviceConnected(device_id="reconnected_phone"))
        await asyncio.sleep(0.02)

        # Should receive connect packet with latest local clipboard
        assert len(sent) == 1
        assert sent[0].type == CLIPBOARD_CONNECT_PACKET_TYPE
        assert sent[0].body["content"] == "Latest Local Edit"
        assert sent[0].body["timestamp"] == plugin.latest_timestamp

        await plugin.stop()

    async def test_device_disconnected_and_unpaired(
        self, plugin: ClipboardPlugin, event_bus: EventBus
    ) -> None:
        await plugin.start()

        plugin.add_connected_device("phone_x", lambda p: None)
        assert "phone_x" in plugin.get_connected_devices()

        await event_bus.publish(DeviceDisconnected(device_id="phone_x"))
        assert "phone_x" not in plugin.get_connected_devices()

        plugin.add_connected_device("phone_y", lambda p: None)
        await event_bus.publish(DeviceUnpaired(device_id="phone_y"))
        assert "phone_y" not in plugin.get_connected_devices()

        await plugin.stop()


# =============================================================================
# 12. Privacy Audit & Redaction
# =============================================================================

class TestPrivacyAndZeroLogging:
    def test_event_repr_and_str_redacts_secret_content(self) -> None:
        secret = "MySecretBankPassword123!"
        event = ClipboardUpdatedEvent(device_id="phone_1", content=secret, timestamp=1000)

        r = repr(event)
        s = str(event)

        assert secret not in r
        assert secret not in s
        assert "[REDACTED" in r
        assert "[REDACTED" in s
        assert "chars" in r
        assert "sha256:" in r

        # But content is directly readable for authorized programmatic handlers
        assert event.content == secret

    async def test_logs_never_contain_raw_clipboard_text(
        self,
        plugin: ClipboardPlugin,
        memory_backend: InMemoryClipboardBackend,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.DEBUG)
        secret = "SuperSecretToken_XYZ987654321"

        await plugin.start()

        sent: List[Packet] = []

        async def send_cb(p: Packet) -> None:
            sent.append(p)

        # Inbound packet with secret
        pkt = create_clipboard_packet(secret, timestamp=1000)
        await plugin.handle_packet(pkt, "phone_1", send_cb)

        # Outbound copy with secret
        await plugin.set_local_clipboard("AnotherSecret_456")

        for record in caplog.records:
            assert secret not in record.message
            assert "AnotherSecret_456" not in record.message

        await plugin.stop()


# =============================================================================
# 13. Lifecycle Management
# =============================================================================

class TestLifecycleManagement:
    async def test_start_and_stop_idempotency(
        self, plugin: ClipboardPlugin
    ) -> None:
        assert not plugin.is_running

        await plugin.start()
        assert plugin.is_running

        # Calling start again is idempotent
        await plugin.start()
        assert plugin.is_running

        await plugin.stop()
        assert not plugin.is_running

        # Calling stop again is idempotent
        await plugin.stop()
        assert not plugin.is_running

    async def test_start_seeds_initial_clipboard(
        self, event_bus: EventBus, memory_backend: InMemoryClipboardBackend
    ) -> None:
        await memory_backend.set_text("Initial Seeded Text")

        p = ClipboardPlugin(event_bus=event_bus, backend=memory_backend)
        await p.start()

        assert p.last_content == "Initial Seeded Text"
        assert p.latest_timestamp > 0
        assert p.suppression.digest_cache.contains_text("Initial Seeded Text")

        await p.stop()

    async def test_handle_packet_rejected_when_not_running(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        """Packets dispatched before start() are safely dropped."""
        assert not plugin.is_running
        pkt = create_clipboard_packet("Before start", timestamp=1000)
        await plugin.handle_packet(pkt, "dev1", lambda p: None)
        assert await memory_backend.get_text() == ""
        assert plugin.get_connected_devices() == []

    async def test_handle_packet_rejected_after_stop(
        self, plugin: ClipboardPlugin, memory_backend: InMemoryClipboardBackend
    ) -> None:
        """Packets dispatched after stop() are safely dropped."""
        await plugin.start()
        await plugin.stop()
        assert not plugin.is_running
        pkt = create_clipboard_packet("After stop", timestamp=1000)
        await plugin.handle_packet(pkt, "dev1", lambda p: None)
        assert await memory_backend.get_text() == ""
        assert plugin.get_connected_devices() == []

    async def test_set_local_clipboard_rejected_when_not_running(
        self, plugin: ClipboardPlugin
    ) -> None:
        """Programmatic clipboard update returns False when plugin is not running."""
        assert not plugin.is_running
        ok = await plugin.set_local_clipboard("Test")
        assert ok is False

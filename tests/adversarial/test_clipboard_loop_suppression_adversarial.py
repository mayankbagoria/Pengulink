"""tests.adversarial.test_clipboard_loop_suppression_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial stress-test suite for the KDE Connect Clipboard 4-layer loop
suppression engine and ClipboardPlugin in pengulink/plugins/clipboard/plugin.py.

Stress Dimensions:
1. Concurrent local/remote echo reflection cycles & race conditions
2. LRU cache overflow, eviction, and boundary capacity (32, 33, 64, 10_000)
3. Timestamp monotonicity, NTP backward jumps, and clock skew boundaries
4. Multi-peer topologies (3+ devices), mesh reflections, and originator exclusion
5. Resource leaks, task lifecycle safety, and fault injection resilience
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Set
import pytest

from pengulink.core.events import (
    ClipboardUpdatedEvent,
    DeviceConnected,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
)
from pengulink.network.packet import Packet
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
# Helper Utilities & Test Doubles
# =============================================================================

class MockPeerDevice:
    """Mock peer tracking received packets and capable of simulating echo reflection."""

    def __init__(self, device_id: str, auto_echo: bool = False) -> None:
        self.device_id: str = device_id
        self.auto_echo: bool = auto_echo
        self.received_packets: List[Packet] = []
        self.echo_callback: Optional[Callable[[Packet], Any]] = None
        self.raise_on_send: Optional[Exception] = None

    async def send_packet(self, packet: Packet) -> None:
        """Simulate transmission over mTLS channel."""
        if self.raise_on_send is not None:
            raise self.raise_on_send
        self.received_packets.append(packet)
        if self.auto_echo and self.echo_callback:
            # Echo packet back to plugin immediately or in task
            asyncio.create_task(self.echo_callback(packet))


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def memory_backend() -> InMemoryClipboardBackend:
    return InMemoryClipboardBackend()


# =============================================================================
# Dimension 1: Concurrent Echo Reflection Cycles & Race Conditions
# =============================================================================

class TestConcurrentEchoReflectionCycles:
    """Stress-test concurrent echo reflections, ping-pong loops, and in-flight races."""

    async def test_echo_mirror_device_produces_zero_secondary_reflections(
        self, event_bus: EventBus, memory_backend: InMemoryClipboardBackend
    ) -> None:
        """Adversarial Echo Mirror: Remote peer immediately reflects any packet it receives.

        Verify that a local clipboard mutation triggers 1 broadcast to the mirror,
        the mirror reflects it back, and the 4-layer engine swallows the reflection
        with zero secondary broadcasts and zero infinite loops.
        """
        plugin = ClipboardPlugin(
            event_bus=event_bus,
            backend=memory_backend,
            ttl_seconds=0.5,
            digest_cache_size=32,
        )

        mirror_peer = MockPeerDevice("mirror-device-1", auto_echo=True)

        async def mirror_echo(pkt: Packet) -> None:
            # Mirror echoes the packet back to the plugin
            await plugin.handle_packet(pkt, mirror_peer.device_id, mirror_peer.send_packet)

        mirror_peer.echo_callback = mirror_echo
        plugin.add_connected_device(mirror_peer.device_id, mirror_peer.send_packet)
        await plugin.start()

        # Trigger local clipboard change
        success = await plugin.set_local_clipboard("Adversarial ping-pong test string 42")
        assert success is True

        # Allow async reflection cycles to settle
        await asyncio.sleep(0.1)

        # Mirror received the original broadcast
        assert len(mirror_peer.received_packets) == 1
        assert mirror_peer.received_packets[0].body["content"] == "Adversarial ping-pong test string 42"

        # Verify no further packets were sent (loop broken on first reflection)
        await asyncio.sleep(0.05)
        assert len(mirror_peer.received_packets) == 1

        await plugin.stop()

    async def test_high_concurrency_bidirectional_blast(
        self, event_bus: EventBus, memory_backend: InMemoryClipboardBackend
    ) -> None:
        """High-concurrency stress: 50 concurrent tasks (25 local writes + 25 remote packets).

        Ensure no race conditions, no unhandled exceptions, no corrupted LRU state,
        and clean state convergence.
        """
        plugin = ClipboardPlugin(
            event_bus=event_bus,
            backend=memory_backend,
            ttl_seconds=0.5,
            digest_cache_size=64,
        )

        peer1 = MockPeerDevice("peer-blast-1")
        peer2 = MockPeerDevice("peer-blast-2")
        plugin.add_connected_device(peer1.device_id, peer1.send_packet)
        plugin.add_connected_device(peer2.device_id, peer2.send_packet)
        await plugin.start()

        base_time = int(time.time() * 1000)

        async def perform_local_write(idx: int) -> None:
            await asyncio.sleep(0.001 * (idx % 5))
            await plugin.set_local_clipboard(f"local-concurrency-write-{idx}")

        async def perform_remote_packet(idx: int) -> None:
            await asyncio.sleep(0.001 * (idx % 7))
            sender = peer1 if idx % 2 == 0 else peer2
            pkt = create_clipboard_packet(
                f"remote-concurrency-packet-{idx}",
                timestamp=base_time + 10_000 + idx * 10,
            )
            await plugin.handle_packet(pkt, sender.device_id, sender.send_packet)

        # Launch 25 local writes and 25 remote packets concurrently
        tasks = []
        for i in range(25):
            tasks.append(perform_local_write(i))
            tasks.append(perform_remote_packet(i))

        # Gather without raising unhandled exceptions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                pytest.fail(f"Concurrent blast raised unhandled exception: {r}")

        # Settle pending broadcast tasks
        await asyncio.sleep(0.1)

        # Verify engine state integrity
        assert len(plugin.suppression.digest_cache) <= 64
        assert plugin.latest_timestamp > base_time

        await plugin.stop()

    async def test_interleaved_write_in_flight_queue_race(
        self, event_bus: EventBus, memory_backend: InMemoryClipboardBackend
    ) -> None:
        """Layer 2 stress: Rapid sequential remote writes before OS reflection events arrive.

        Remote sends: Msg1, Msg2, Msg3 rapidly.
        Then delayed OS echo events for Msg1, Msg2, Msg3 fire in reverse order.
        Verify that Layer 1 (digest cache) covers what Layer 2 (in-flight flag) might miss
        due to single-slot overwriting.
        """
        plugin = ClipboardPlugin(
            event_bus=event_bus,
            backend=memory_backend,
            ttl_seconds=1.0,
            digest_cache_size=32,
        )

        peer = MockPeerDevice("peer-queue-race")
        plugin.add_connected_device(peer.device_id, peer.send_packet)
        await plugin.start()

        now_ms = int(time.time() * 1000)
        messages = ["stream-msg-alpha", "stream-msg-beta", "stream-msg-gamma"]

        # Ingest 3 remote packets in quick succession
        for idx, msg in enumerate(messages):
            pkt = create_clipboard_packet(msg, timestamp=now_ms + 100 + idx * 10)
            await plugin.handle_packet(pkt, peer.device_id, peer.send_packet)

        # Clear peer's received packets to monitor any subsequent erroneous broadcasts
        peer.received_packets.clear()

        # Simulate delayed OS reflections arriving in reverse order
        for msg in reversed(messages):
            # Directly invoke the backend change callback to simulate OS reflection
            await plugin._on_local_clipboard_changed(msg)

        # Allow any potential broadcasts to execute
        await asyncio.sleep(0.05)

        # ALL 3 delayed OS events MUST be suppressed by Layer 1/2 combination!
        assert len(peer.received_packets) == 0, (
            f"Expected 0 echoes broadcast, got {len(peer.received_packets)}: "
            f"{[p.body['content'] for p in peer.received_packets]}"
        )

        await plugin.stop()

    async def test_rapid_alternating_oscillation_ping_pong(
        self, event_bus: EventBus, memory_backend: InMemoryClipboardBackend
    ) -> None:
        """Adversarial toggle: Rapid oscillation between two values 'PING' and 'PONG'.

        Verify that when user or remote flips between two strings, digests stay in LRU
        and transitions are tracked properly without infinite loops.
        """
        plugin = ClipboardPlugin(
            event_bus=event_bus,
            backend=memory_backend,
            ttl_seconds=0.2,
            digest_cache_size=32,
        )
        peer = MockPeerDevice("peer-toggle")
        plugin.add_connected_device(peer.device_id, peer.send_packet)
        await plugin.start()

        base_ts = int(time.time() * 1000)

        # 10 alternating updates
        for step in range(10):
            val = "PING" if step % 2 == 0 else "PONG"
            pkt = create_clipboard_packet(val, timestamp=base_ts + (step + 1) * 100)
            await plugin.handle_packet(pkt, peer.device_id, peer.send_packet)

        # Settle
        await asyncio.sleep(0.05)

        # The last value in clipboard must be PONG
        current = await plugin.get_current_clipboard()
        assert current["content"] == "PONG"

        await plugin.stop()

    def test_write_in_flight_boundary_precision(self) -> None:
        """Monotonic TTL expiration edge cases."""
        engine = LoopSuppressionEngine(write_in_flight_ttl=0.05)

        engine.arm_write_in_flight("test-content")
        # Immediately active
        assert engine.is_write_in_flight_active("test-content") is True

        # Different content during active window is NOT suppressed by Layer 2
        assert engine.is_write_in_flight_active("different-content") is False

        # Wait past TTL
        time.sleep(0.07)
        # Expired
        assert engine.is_write_in_flight_active("test-content") is False
        assert engine.in_flight is None


# =============================================================================
# Dimension 2: LRU Cache Overflow, Eviction, and Boundary Capacity
# =============================================================================

class TestLRUCacheCapacityAndOverflow:
    """Stress-test RollingDigestLRUCache across capacity boundaries and stress volumes."""

    @pytest.mark.parametrize("capacity", [1, 2, 31, 32, 33, 64])
    def test_exact_capacity_boundary_and_fifo_eviction(self, capacity: int) -> None:
        """Verify strict size invariance and FIFO eviction when items are not refreshed."""
        cache = RollingDigestLRUCache(capacity=capacity)
        assert len(cache) == 0

        # Fill up to capacity
        for i in range(capacity):
            cache.add_text(f"item-{i}")
            assert len(cache) == i + 1

        assert len(cache) == capacity
        # Oldest item is item-0
        assert cache.contains_text("item-0") is True

        # Add overflow item
        cache.add_text("overflow-item")
        assert len(cache) == capacity

        if capacity == 1:
            # Capacity 1: item-0 was evicted by overflow-item
            assert cache.contains_text("overflow-item") is True
            assert cache.contains_text("item-0") is False
        else:
            # Capacity > 1: item-0 was touched (promoted to MRU), so item-1 was evicted
            assert cache.contains_text("item-0") is True
            assert cache.contains_text("item-1") is False

    def test_lru_recency_promotion_prevents_eviction(self) -> None:
        """Verify that accessing an entry (via contains or add) promotes it to MRU."""
        cache = RollingDigestLRUCache(capacity=3)
        cache.add_text("A")
        cache.add_text("B")
        cache.add_text("C")

        # Order: [A, B, C] (A oldest)
        # Touch A
        assert cache.contains_text("A") is True
        # Order should now be: [B, C, A] (B oldest)

        # Add D -> B must be evicted, A and C must remain
        cache.add_text("D")
        assert len(cache) == 3
        assert cache.contains_text("B") is False
        assert cache.contains_text("A") is True
        assert cache.contains_text("C") is True
        assert cache.contains_text("D") is True

    def test_cache_flood_10000_distinct_items_no_memory_leak(self) -> None:
        """Throughput stress: 10,000 distinct strings inserted.

        Cache size must never exceed capacity 32, and all operations must run in O(1).
        """
        cache = RollingDigestLRUCache(capacity=32)
        start_time = time.perf_counter()

        for i in range(10_000):
            d = cache.add_text(f"stress-item-{i}-random-padding-xyz")
            if i % 100 == 0:
                assert cache.contains(d) is True
                assert len(cache) <= 32

        elapsed = time.perf_counter() - start_time
        assert len(cache) == 32
        # Must execute comfortably fast (< 0.5s for 10,000 hashes in Python)
        assert elapsed < 0.5

    async def test_eviction_echo_layer3_defense(
        self, event_bus: EventBus, memory_backend: InMemoryClipboardBackend
    ) -> None:
        """The Eviction-Reflection Attack:

        1. Remote peer sends string 'VULN'.
        2. Attacker floods 32 new strings, evicting 'VULN' from Layer 1 LRU cache.
        3. Attacker resends 'VULN' with the original stale timestamp.
        Verify that even though Layer 1 LRU evicted 'VULN', Layer 3 (timestamp monotonicity)
        STILL blocks the packet as 'stale_timestamp'!
        """
        plugin = ClipboardPlugin(
            event_bus=event_bus,
            backend=memory_backend,
            digest_cache_size=32,
        )
        peer = MockPeerDevice("peer-evict-attacker")
        plugin.add_connected_device(peer.device_id, peer.send_packet)
        await plugin.start()

        base_ts = int(time.time() * 1000)

        # Step 1: Send VULN
        pkt_vuln = create_clipboard_packet("VULN-PAYLOAD", timestamp=base_ts + 100)
        await plugin.handle_packet(pkt_vuln, peer.device_id, peer.send_packet)

        # Step 2: Flood 35 items to guarantee eviction of VULN from 32-capacity cache
        for i in range(35):
            pkt_flood = create_clipboard_packet(f"flood-{i}", timestamp=base_ts + 200 + i)
            await plugin.handle_packet(pkt_flood, peer.device_id, peer.send_packet)

        # Confirm VULN was evicted from Layer 1
        vuln_digest = RollingDigestLRUCache.hash_content("VULN-PAYLOAD")
        assert plugin.suppression.digest_cache.contains(vuln_digest) is False

        # Step 3: Resend VULN with original timestamp (base_ts + 100)
        # Layer 1 has forgotten it, but Layer 3 timestamp MUST DROP IT!
        accepted, reason = plugin.suppression.evaluate_inbound(
            content="VULN-PAYLOAD",
            remote_timestamp=base_ts + 100,
            device_id=peer.device_id,
        )
        assert accepted is False
        assert "stale_timestamp" in reason

        await plugin.stop()

    def test_degenerate_and_boundary_text_hashing(self) -> None:
        """Digest computation under degenerate content."""
        cache = RollingDigestLRUCache(capacity=32)

        # Empty string
        d_empty = cache.add_text("")
        assert d_empty == hashlib.sha256(b"").hexdigest()

        # Unicode surrogates and emojis
        d_emoji = cache.add_text("🔥🚀🦀\U0001F600")
        assert len(d_emoji) == 64

        # Maximum ceiling text (120 KB)
        huge_text = "A" * MAX_CLIPBOARD_PACKET_BYTES
        d_huge = cache.add_text(huge_text)
        assert len(d_huge) == 64

        # Re-adding returns same digest and does not duplicate
        assert cache.add_text("") == d_empty
        assert len(cache) == 3


# =============================================================================
# Dimension 3: Timestamp Monotonicity and Clock Skew
# =============================================================================

class TestTimestampMonotonicityAndClockSkew:
    """Stress-test timestamp evaluation, backward jumps, and clock drift ceilings."""

    def test_outbound_strictly_monotonic_advancement_1000_bursts(self) -> None:
        """1,000 rapid outbound events must produce strictly monotonically increasing timestamps."""
        engine = LoopSuppressionEngine()
        last_ts = 0

        for i in range(1, 1001):
            _, ts = engine.record_outbound_accepted(f"local-clip-{i}")
            assert ts > last_ts, f"Monotonicity violated at step {i}: {ts} <= {last_ts}"
            last_ts = ts

        assert engine.local_timestamp == last_ts

    def test_local_clock_backward_jump_resilience_ntp(self) -> None:
        """Simulate local system clock jumping backward by 1 hour (NTP correction).

        Verify that engine.record_outbound_accepted NEVER regresses backward.
        """
        engine = LoopSuppressionEngine()

        # Normal operation
        _, ts1 = engine.record_outbound_accepted("clip-before-jump")

        # Simulate system wall clock jumping backward by 3,600,000 ms (1 hour)
        simulated_jumped_now = ts1 - 3_600_000

        # Monkeypatch current_wall_ms to return the jumped backward time
        orig_wall = engine.current_wall_ms
        engine.current_wall_ms = lambda: simulated_jumped_now

        try:
            _, ts2 = engine.record_outbound_accepted("clip-after-jump-1")
            assert ts2 > ts1, f"Clock regressed after NTP jump: {ts2} <= {ts1}"

            _, ts3 = engine.record_outbound_accepted("clip-after-jump-2")
            assert ts3 > ts2, f"Clock regressed: {ts3} <= {ts2}"
        finally:
            engine.current_wall_ms = orig_wall

    def test_future_clock_drift_ceiling_boundary(self) -> None:
        """Test boundary condition around MAX_FUTURE_CLOCK_DRIFT_MS (24 hours)."""
        engine = LoopSuppressionEngine(max_future_drift_ms=MAX_FUTURE_CLOCK_DRIFT_MS)
        now_ms = engine.current_wall_ms()

        # Exactly at limit: now + 24 hours -> accepted
        accepted, reason = engine.evaluate_inbound("future-allowed", now_ms + MAX_FUTURE_CLOCK_DRIFT_MS, "peer-1")
        assert accepted is True

        # Exactly 1 ms beyond limit: now + 24 hours + 1ms -> rejected
        accepted, reason = engine.evaluate_inbound("future-forbidden", now_ms + MAX_FUTURE_CLOCK_DRIFT_MS + 1, "peer-1")
        assert accepted is False
        assert reason == "future_clock_drift_exceeded"

        # Absurd future: +30 days -> rejected
        accepted, reason = engine.evaluate_inbound("future-absurd", now_ms + 30 * 86_400_000, "peer-1")
        assert accepted is False
        assert reason == "future_clock_drift_exceeded"

    def test_clock_skew_blast_radius_analysis(self) -> None:
        """Adversarial analysis of legitimate peer suppression by skewed peer.

        If Peer A sends timestamp (now + 12 hours), local clock advances to (now + 12 hours).
        Subsequent packets from normal Peer B (with normal wall clock) will be rejected
        as 'stale_timestamp'.
        Document and verify this deterministic behavior.
        """
        engine = LoopSuppressionEngine()
        now_ms = engine.current_wall_ms()

        # Skewed peer attempts to advance local clock by +12 hours, but CLUS-27 clamps to max_future_drift_ms
        engine.record_inbound_accepted("skewed-text", now_ms + 12 * 3600 * 1000)
        assert engine.local_timestamp == now_ms + engine.max_future_drift_ms

        # Normal peer sends packet with current time -> REJECTED as stale within the 15s window
        accepted, reason = engine.evaluate_inbound("normal-text", now_ms + 5000, "normal-peer")
        assert accepted is False
        assert "stale_timestamp" in reason

    @pytest.mark.parametrize("bad_ts", [0, -1, -999999, -2**31])
    def test_negative_and_zero_timestamps_rejected(self, bad_ts: int) -> None:
        """Non-positive timestamps must be rejected immediately."""
        engine = LoopSuppressionEngine()
        accepted, reason = engine.evaluate_inbound("text", bad_ts, "dev-1")
        assert accepted is False
        assert reason == "invalid_or_missing_timestamp"

    def test_stale_timestamp_strictness(self) -> None:
        """Standard clipboard packets with remote_ts == local_ts must be rejected."""
        engine = LoopSuppressionEngine()
        engine.local_timestamp = 5000

        # Equal timestamp on standard packet -> rejected
        accepted, reason = engine.evaluate_inbound("text-1", 5000, "dev-1", is_connect_packet=False)
        assert accepted is False
        assert "stale_timestamp" in reason

        # Equal timestamp on connect packet -> accepted
        accepted, reason = engine.evaluate_inbound("text-2", 5000, "dev-1", is_connect_packet=True)
        assert accepted is True


# =============================================================================
# Dimension 4: Originator Exclusion in Multi-Peer Topologies
# =============================================================================

class TestMultiPeerTopologyOriginatorExclusion:
    """Stress-test fanout filtering, mesh reflection suppression, and peer topologies."""

    async def test_star_topology_originator_exclusion_10_devices(
        self, event_bus: EventBus, memory_backend: InMemoryClipboardBackend
    ) -> None:
        """Star Topology: 1 Linux daemon + 10 connected peers (Dev-0 to Dev-9).

        Dev-4 sends a clipboard update.
        Verify that Dev-4 receives ZERO packets, and all 9 other peers receive EXACTLY 1 packet.
        """
        plugin = ClipboardPlugin(event_bus=event_bus, backend=memory_backend)
        peers = [MockPeerDevice(f"dev-{i}") for i in range(10)]

        for p in peers:
            plugin.add_connected_device(p.device_id, p.send_packet)
        await plugin.start()

        now_ms = int(time.time() * 1000)
        originator = peers[4]

        # Originator Dev-4 sends packet
        pkt = create_clipboard_packet("Multicast broadcast message", timestamp=now_ms + 100)
        await plugin.handle_packet(pkt, originator.device_id, originator.send_packet)

        await asyncio.sleep(0.05)

        # Originator MUST NOT receive any packet
        assert len(originator.received_packets) == 0

        # All 9 other peers MUST receive exactly 1 packet
        for idx, p in enumerate(peers):
            if idx == 4:
                continue
            assert len(p.received_packets) == 1
            assert p.received_packets[0].body["content"] == "Multicast broadcast message"

        await plugin.stop()

    async def test_mesh_reflection_multi_peer_echo_storm(
        self, event_bus: EventBus, memory_backend: InMemoryClipboardBackend
    ) -> None:
        """Mesh Reflection Cycle: 4 Devices (A, B, C, D).

        1. A sends packet to Linux.
        2. Linux fans out to B, C, D.
        3. B, C, and D ALL reflect the packet back to Linux immediately.
        4. Linux must swallow all 3 reflections without re-broadcasting anything back to anyone.
        """
        plugin = ClipboardPlugin(event_bus=event_bus, backend=memory_backend)

        peer_a = MockPeerDevice("mesh-dev-a")
        peer_b = MockPeerDevice("mesh-dev-b", auto_echo=True)
        peer_c = MockPeerDevice("mesh-dev-c", auto_echo=True)
        peer_d = MockPeerDevice("mesh-dev-d", auto_echo=True)

        async def make_echo(dev: MockPeerDevice) -> Callable[[Packet], Any]:
            async def echo(pkt: Packet) -> None:
                await plugin.handle_packet(pkt, dev.device_id, dev.send_packet)
            return echo

        peer_b.echo_callback = await make_echo(peer_b)
        peer_c.echo_callback = await make_echo(peer_c)
        peer_d.echo_callback = await make_echo(peer_d)

        for p in [peer_a, peer_b, peer_c, peer_d]:
            plugin.add_connected_device(p.device_id, p.send_packet)
        await plugin.start()

        now_ms = int(time.time() * 1000)
        init_pkt = create_clipboard_packet("Mesh-Sync-Content-999", timestamp=now_ms + 100)

        # Dev A sends packet to Linux
        await plugin.handle_packet(init_pkt, peer_a.device_id, peer_a.send_packet)

        # Settle all reflections
        await asyncio.sleep(0.1)

        # Peer A must still have 0 received packets (never echoed back)
        assert len(peer_a.received_packets) == 0

        # B, C, D received 1 packet each initially, and their echoes produced NO subsequent packets
        assert len(peer_b.received_packets) == 1
        assert len(peer_c.received_packets) == 1
        assert len(peer_d.received_packets) == 1

        await plugin.stop()

    async def test_dynamic_peer_churn_during_active_fanout(
        self, event_bus: EventBus, memory_backend: InMemoryClipboardBackend
    ) -> None:
        """Dynamic Churn: Peers disconnect and unpair while broadcast tasks are running.

        Verify no exceptions, no leaked tasks, and graceful failure isolation.
        """
        plugin = ClipboardPlugin(event_bus=event_bus, backend=memory_backend)

        p1 = MockPeerDevice("churn-dev-1")
        p2 = MockPeerDevice("churn-dev-2")
        p3 = MockPeerDevice("churn-dev-3")

        # p2 will fail when sent to
        p2.raise_on_send = ConnectionResetError("Connection abruptly reset by peer")

        for p in [p1, p2, p3]:
            plugin.add_connected_device(p.device_id, p.send_packet)
        await plugin.start()

        # Concurrently perform local copy while unpairing p3
        async def do_unpair() -> None:
            await asyncio.sleep(0.001)
            plugin.clear_device(p3.device_id)

        async def do_copy() -> None:
            await plugin.set_local_clipboard("churn-test-content")

        await asyncio.gather(do_copy(), do_unpair(), return_exceptions=False)
        await asyncio.sleep(0.05)

        # p1 should receive packet cleanly
        assert len(p1.received_packets) == 1
        # p2 encountered connection reset safely without crashing
        assert len(p2.received_packets) == 0

        await plugin.stop()

    def test_fanout_capability_filtering(self) -> None:
        """Ensure devices lacking kdeconnect.clipboard are excluded from fanout."""
        caps = {
            "dev-full": ["kdeconnect.clipboard", "kdeconnect.ping"],
            "dev-ping-only": ["kdeconnect.ping"],
            "dev-battery-only": ["kdeconnect.battery"],
        }

        def check_cap(dev_id: str, cap: str) -> bool:
            return cap in caps.get(dev_id, [])

        active = ["dev-full", "dev-ping-only", "dev-battery-only"]
        targets = LoopSuppressionEngine.get_fanout_recipients(
            active_devices=active,
            originator_device_id=None,
            capability_checker=check_cap,
            required_capability=CLIPBOARD_PACKET_TYPE,
        )

        assert targets == ["dev-full"]


# =============================================================================
# Dimension 5: Chaos, Malformed Data & Resource Safety
# =============================================================================

class TestChaosAndResourceSafety:
    """Stress-test resource leaks, malformed payloads, and lifecycle safety."""

    async def test_zero_leaked_tasks_after_stop(
        self, event_bus: EventBus, memory_backend: InMemoryClipboardBackend
    ) -> None:
        """Verify plugin cleanly drains and cancels all internal background tasks."""
        plugin = ClipboardPlugin(event_bus=event_bus, backend=memory_backend)
        peer = MockPeerDevice("task-leak-peer")
        plugin.add_connected_device(peer.device_id, peer.send_packet)
        await plugin.start()

        # Trigger several operations
        for i in range(10):
            await plugin.set_local_clipboard(f"content-{i}")

        # Stop immediately while tasks may be in flight
        await plugin.stop()

        # Background tasks set must be empty and all done
        assert len(plugin._background_tasks) == 0

    @pytest.mark.parametrize(
        "bad_payload",
        [
            None,
            12345,
            [],
            {"content": 99999},  # non-string content
            {"content": None},
            {"content": "A" * 150_000},  # exceeds 120 KB ceiling
            {"content": "valid", "timestamp": -50},  # negative timestamp
            {"content": "valid", "timestamp": 0},    # zero timestamp
        ],
    )
    async def test_fuzzed_and_malformed_packets_handled_safely(
        self,
        event_bus: EventBus,
        memory_backend: InMemoryClipboardBackend,
        bad_payload: Any,
    ) -> None:
        """Send corrupted/adversarial packet bodies; plugin must reject safely without crashing."""
        plugin = ClipboardPlugin(event_bus=event_bus, backend=memory_backend)
        peer = MockPeerDevice("fuzz-peer")
        plugin.add_connected_device(peer.device_id, peer.send_packet)
        await plugin.start()

        # Use model_construct to bypass client-side Pydantic validation and test server resilience
        pkt = Packet.model_construct(id=1, type=CLIPBOARD_PACKET_TYPE, body=bad_payload)

        # Must not raise any unhandled exception
        await plugin.handle_packet(pkt, peer.device_id, peer.send_packet)

        # Local clipboard should remain untouched
        assert await memory_backend.get_text() == ""

        await plugin.stop()

    async def test_fallback_timestamp_corrupt_body_timestamp_with_packet_id(
        self,
        event_bus: EventBus,
        memory_backend: InMemoryClipboardBackend,
    ) -> None:
        """When body['timestamp'] is a corrupted string, KDE Connect protocol falls back to packet.id."""
        plugin = ClipboardPlugin(event_bus=event_bus, backend=memory_backend)
        peer = MockPeerDevice("fallback-peer")
        plugin.add_connected_device(peer.device_id, peer.send_packet)
        await plugin.start()

        pkt = Packet.model_construct(
            id=5000,
            type=CLIPBOARD_PACKET_TYPE,
            body={"content": "fallback_content", "timestamp": "corrupt_string"},
        )
        await plugin.handle_packet(pkt, peer.device_id, peer.send_packet)

        # Content accepted because packet.id (5000) was used as fallback timestamp
        assert await memory_backend.get_text() == "fallback_content"
        assert plugin.latest_timestamp == 5000

        await plugin.stop()

"""Adversarial Challenge Test Suite: Surrogate Code Points & Stopped-State Lifecycles.

Milestone M9.2 Iteration 2 Gate Challenger Suite:
1. All classes of surrogate code points:
   - Lone high surrogates (U+D800 to U+DBFF, boundary min, boundary max, private use)
   - Lone low surrogates (U+DC00 to U+DFFF, boundary min, boundary max)
   - Inverted surrogate pairs (low followed by high: U+DC00 U+D800, U+DE00 U+D83D)
   - Double high / double low surrogates
   - Separated surrogates interspersed with ASCII/multibyte Unicode
   - Surrogates embedded in large text (10 KB, 50 KB, 119 KB, 120 KB ceiling, 150 KB)
   - OS-level surrogateescape byte sequences (raw invalid UTF-8 bytes from Linux subsystems)
   - Multiline, control char, and null-byte surrogate strings
   - Verification across all entry points: inbound evaluation, handle_packet, outbound evaluation,
     local clipboard watcher callback, programmatic set_local_clipboard, redact_clipboard,
     start() seeding from pre-existing dirty backend, and desktop backends.

2. Stopped-state packet rejection & lifecycle containment:
   - Packets delivered before start(): strictly dropped, zero device tracking, zero mutations
   - Packets delivered after stop(): strictly dropped, zero device tracking, zero mutations
   - Concurrent packet floods colliding with stop(): zero task leaks, zero unclosed resources
   - Secondary hooks when stopped: _on_local_clipboard_changed, _on_device_connected,
     _broadcast_packet, set_local_clipboard
   - Multi-cycle reactivation (Unstarted -> Started -> Stopped -> Started -> Stopped)
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pytest

from pengulink.core.events import (
    ClipboardUpdatedEvent,
    DeviceConnected,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
)
from pengulink.network.packet import Packet, PacketEngine
from pengulink.plugins.clipboard.backend import (
    InMemoryClipboardBackend,
    WaylandClipboardBackend,
    X11ClipboardBackend,
    redact_clipboard,
)
from pengulink.plugins.clipboard.plugin import (
    CLIPBOARD_CONNECT_PACKET_TYPE,
    CLIPBOARD_PACKET_TYPE,
    MAX_CLIPBOARD_PACKET_BYTES,
    ClipboardPlugin,
    create_clipboard_connect_packet,
    create_clipboard_packet,
)


# =============================================================================
# Adversarial Fixtures
# =============================================================================

@pytest.fixture
def adversarial_env():
    """Create isolated test environment with EventBus, InMemory backend, and plugin."""
    bus = EventBus()
    backend = InMemoryClipboardBackend()
    captured_events: List[ClipboardUpdatedEvent] = []
    bus.subscribe(ClipboardUpdatedEvent, lambda e: captured_events.append(e))

    plugin = ClipboardPlugin(
        event_bus=bus,
        backend=backend,
        ttl_seconds=0.5,
        digest_cache_size=32,
    )
    return {
        "bus": bus,
        "backend": backend,
        "plugin": plugin,
        "events": captured_events,
    }


# =============================================================================
# 1. Exhaustive Surrogate Permutations Test Suite
# =============================================================================

class TestSurrogateCodePointAdversarial:
    """Stress-test all classes of surrogate code points across all boundaries."""

    SURROGATE_SAMPLES = [
        # --- Lone High Surrogates (U+D800 to U+DBFF) ---
        ("\ud800", "min high surrogate U+D800"),
        ("\ud801", "high surrogate U+D801"),
        ("\ud900", "high surrogate U+D900"),
        ("\udb7f", "max standard high surrogate U+DB7F"),
        ("\udb80", "min private use high surrogate U+DB80"),
        ("\udbff", "max private use high surrogate U+DBFF"),

        # --- Lone Low Surrogates (U+DC00 to U+DFFF) ---
        ("\udc00", "min low surrogate U+DC00"),
        ("\udc01", "low surrogate U+DC01"),
        ("\ude00", "mid low surrogate U+DE00"),
        ("\udeff", "low surrogate U+DEFF"),
        ("\udffe", "low surrogate U+DFFE"),
        ("\udfff", "max low surrogate U+DFFF"),

        # --- Inverted Surrogate Pairs (Low followed by High) ---
        ("\udc00\ud800", "inverted boundary pair U+DC00 U+D800"),
        ("\ude00\ud83d", "inverted emoji pair U+DE00 U+D83D"),
        ("\udfff\udbff", "inverted maximum pair U+DFFF U+DBFF"),

        # --- Double High / Double Low Surrogates ---
        ("\ud800\ud800", "double min high surrogate"),
        ("\ud83d\ud83d", "double emoji high surrogate"),
        ("\udbff\udbff", "double max high surrogate"),
        ("\udc00\udc00", "double min low surrogate"),
        ("\ude00\ude00", "double mid low surrogate"),
        ("\udfff\udfff", "double max low surrogate"),

        # --- High Surrogates without Low Surrogates, followed by normal characters ---
        ("\ud800A", "high surrogate followed by ASCII"),
        ("\ud800\u4e2d", "high surrogate followed by CJK character"),
        ("\ud800🌟", "high surrogate followed by valid SMP emoji"),

        # --- Low Surrogates preceded by normal characters ---
        ("A\udc00", "ASCII followed by low surrogate"),
        ("\u4e2d\udc00", "CJK followed by low surrogate"),
        ("🌟\udfff", "SMP emoji followed by low surrogate"),

        # --- Separated Surrogates ---
        ("\ud800_separator_\udc00", "high and low surrogates separated by ASCII"),
        ("prefix_\udc00_middle_\ud800_suffix", "inverted separated surrogates"),

        # --- Long repetitive surrogate sequences ---
        ("\ud800" * 200, "string of 200 high surrogates"),
        ("\udc00" * 200, "string of 200 low surrogates"),
        (("\udc00\ud800") * 100, "200 inverted surrogate sequence"),

        # --- Multiline & Control Characters with Surrogates ---
        ("Header\n\ud800\r\nBody\x00\udfff\tFooter", "multiline null-byte surrogate string"),

        # --- Linux OS surrogateescape Decoded Byte Sequences ---
        (b"\x80\xff\xfe\xc0".decode("utf-8", errors="surrogateescape"), "raw invalid bytes via surrogateescape"),
    ]

    @pytest.mark.parametrize("sample,desc", SURROGATE_SAMPLES)
    async def test_inbound_surrogate_rejection_pipeline(self, adversarial_env, sample: str, desc: str) -> None:
        """Verify inbound packet rejection across all surrogate permutations."""
        plugin = adversarial_env["plugin"]
        backend = adversarial_env["backend"]
        events = adversarial_env["events"]
        await plugin.start()

        # 1. evaluate_inbound returns (False, 'corrupted_surrogate_encoding')
        accepted, reason = plugin.suppression.evaluate_inbound(sample, 5000, "device_attacker")
        assert accepted is False, f"Inbound accepted surrogate: {desc}"
        assert reason == "corrupted_surrogate_encoding", f"Wrong rejection reason for {desc}: {reason}"

        # 2. handle_packet drops packet without exception, does not mutate backend, emits no events
        sent_pkts: List[Packet] = []
        in_pkt = create_clipboard_packet(sample, timestamp=5000)
        await plugin.handle_packet(in_pkt, "device_attacker", lambda p: sent_pkts.append(p))

        assert await backend.get_text() == "", f"Backend mutated by surrogate: {desc}"
        assert len(sent_pkts) == 0, f"Packet forwarded for surrogate: {desc}"
        assert len(events) == 0, f"Event published for surrogate: {desc}"

        await plugin.stop()

    @pytest.mark.parametrize("sample,desc", SURROGATE_SAMPLES)
    async def test_outbound_surrogate_rejection_pipeline(self, adversarial_env, sample: str, desc: str) -> None:
        """Verify outbound OS change rejection across all surrogate permutations."""
        plugin = adversarial_env["plugin"]
        events = adversarial_env["events"]
        await plugin.start()

        # 1. evaluate_outbound_os_event returns (False, 'corrupted_surrogate_encoding')
        accepted, reason = plugin.suppression.evaluate_outbound_os_event(sample)
        assert accepted is False, f"Outbound accepted surrogate: {desc}"
        assert reason == "corrupted_surrogate_encoding", f"Wrong rejection reason for {desc}: {reason}"

        # 2. _on_local_clipboard_changed drops without broadcasting or publishing
        await plugin._on_local_clipboard_changed(sample)
        assert plugin.last_content == "", f"last_content updated by surrogate: {desc}"
        assert len(events) == 0, f"Event published by surrogate outbound: {desc}"

        await plugin.stop()

    @pytest.mark.parametrize("sample,desc", SURROGATE_SAMPLES)
    async def test_programmatic_surrogate_rejection(self, adversarial_env, sample: str, desc: str) -> None:
        """Verify set_local_clipboard rejects surrogates without mutating backend."""
        plugin = adversarial_env["plugin"]
        backend = adversarial_env["backend"]
        events = adversarial_env["events"]
        await plugin.start()

        success = await plugin.set_local_clipboard(sample)
        assert success is False, f"set_local_clipboard succeeded for surrogate: {desc}"
        assert await backend.get_text() == "", f"Backend mutated by set_local_clipboard for {desc}"
        assert len(events) == 0, f"Event published for surrogate: {desc}"

        await plugin.stop()

    @pytest.mark.parametrize("sample,desc", SURROGATE_SAMPLES)
    def test_logger_redaction_surrogate_infallibility(self, sample: str, desc: str) -> None:
        """Verify redact_clipboard never raises UnicodeEncodeError on any surrogate string."""
        formatted = redact_clipboard(sample)
        assert isinstance(formatted, str)
        assert formatted.startswith("[REDACTED, ")
        assert f"{len(sample)} chars" in formatted
        assert "sha256:" in formatted

    async def test_surrogates_embedded_in_large_payloads(self, adversarial_env) -> None:
        """Test surrogates placed at start, middle, end of large texts up to 120 KB ceiling."""
        plugin = adversarial_env["plugin"]
        backend = adversarial_env["backend"]
        await plugin.start()

        test_payloads = [
            # 10 KB with surrogate at index 0
            ("\ud800" + ("A" * 10240), "10KB surrogate at start"),
            # 50 KB with surrogate in middle
            (("A" * 25000) + "\udfff" + ("B" * 25000), "50KB surrogate in middle"),
            # 119 KB with surrogate at end
            (("A" * 119000) + "\ud800", "119KB surrogate at end"),
            # Exactly 120 KB boundary with surrogate
            (("A" * 60000) + "\udc00\ud800" + ("B" * 59998), "120KB inverted surrogate in middle"),
            # 150 KB over-ceiling with surrogate (must reject safely)
            (("A" * 75000) + "\ud800" + ("B" * 75000), "150KB over-limit with surrogate"),
        ]

        for text, desc in test_payloads:
            acc, reason = plugin.suppression.evaluate_inbound(text, 10000, "dev_large")
            assert acc is False, f"Accepted large surrogate payload: {desc}"
            assert reason in ("corrupted_surrogate_encoding", "payload_exceeds_limit")

            # Must not crash handle_packet
            pkt = create_clipboard_packet(text, timestamp=10000)
            await plugin.handle_packet(pkt, "dev_large", lambda p: None)
            assert await backend.get_text() == ""

            # Must not crash set_local_clipboard
            assert await plugin.set_local_clipboard(text) is False

            # Redaction must not crash
            redacted = redact_clipboard(text)
            assert "[REDACTED," in redacted

        await plugin.stop()

    async def test_plugin_start_with_pre_existing_surrogates_in_backend(self) -> None:
        """Verify plugin.start() does not crash if backend already contains surrogate text."""
        bus = EventBus()
        backend = InMemoryClipboardBackend()
        # Seed backend directly with surrogate before plugin starts
        backend._content = "PreExisting\ud800Surrogate"

        plugin = ClipboardPlugin(event_bus=bus, backend=backend)
        # start() reads initial text from backend - must not raise UnicodeEncodeError
        await plugin.start()
        assert plugin.is_running is True
        # Initial text with surrogate must be ignored rather than seeded into last_content
        assert plugin.last_content == ""
        assert plugin.last_digest is None
        assert plugin.latest_timestamp == 0

        await plugin.stop()

    async def test_backend_direct_surrogate_write_rejection(self) -> None:
        """Verify all clipboard backends reject surrogates cleanly via set_text."""
        in_memory = InMemoryClipboardBackend()
        assert await in_memory.set_text("Bad\ud800Surrogate") is False
        assert await in_memory.get_text() == ""

        # simulate_external_copy on in_memory must also reject without crash
        callback_called = False
        async def dummy_cb(text: str) -> None:
            nonlocal callback_called
            callback_called = True

        await in_memory.start_watching(dummy_cb)
        await in_memory.simulate_external_copy("Copy\udfffSurrogate")
        assert callback_called is False
        assert await in_memory.get_text() == ""
        await in_memory.stop_watching()

        # Wayland backend set_text
        wayland = WaylandClipboardBackend()
        assert await wayland.set_text("Wayland\ud800Surrogate") is False

        # X11 backend set_text
        x11 = X11ClipboardBackend()
        assert await x11.set_text("X11\ud800Surrogate") is False

    async def test_reconnect_packet_with_surrogate_rejected_without_push(self, adversarial_env) -> None:
        """Verify kdeconnect.clipboard.connect packet with surrogates does not push local content."""
        plugin = adversarial_env["plugin"]
        backend = adversarial_env["backend"]
        await plugin.start()

        # Seed valid local clipboard
        await plugin.set_local_clipboard("Valid Local Text")
        assert plugin.last_content == "Valid Local Text"

        # Attacker connects with a surrogate connect packet
        sent_pkts: List[Packet] = []
        conn_pkt = create_clipboard_connect_packet("Attacker\ud800Connect", timestamp=1000)
        await plugin.handle_packet(conn_pkt, "dev_surrogate_peer", lambda p: sent_pkts.append(p))

        # Must not push local clipboard to attacker because reason is surrogate, not stale_timestamp
        assert len(sent_pkts) == 0
        assert await backend.get_text() == "Valid Local Text"

        await plugin.stop()


# =============================================================================
# 2. Stopped-State Packet Rejection & Lifecycle Containment Suite
# =============================================================================

class TestStoppedStateLifecycleContainment:
    """Stress-test packet rejection before start, after stop, and during stopping."""

    async def test_packet_delivery_before_start_strictly_rejected(self, adversarial_env) -> None:
        """Verify packet delivery to unstarted plugin leaves all states clean."""
        plugin = adversarial_env["plugin"]
        backend = adversarial_env["backend"]
        events = adversarial_env["events"]

        assert plugin.is_running is False
        assert plugin.get_connected_devices() == []

        forwarded_packets: List[Packet] = []
        pkt = create_clipboard_packet("Pushed Before Start", timestamp=1000)

        # Inbound packet arrives while plugin is NOT running
        await plugin.handle_packet(pkt, "peer_before_start", lambda p: forwarded_packets.append(p))

        # Strict checks:
        # 1. Backend must remain empty
        assert await backend.get_text() == ""
        # 2. Connected devices and sender registries must NOT register the sender
        assert plugin.get_connected_devices() == []
        assert "peer_before_start" not in plugin._device_senders
        # 3. No events published
        assert len(events) == 0
        # 4. No forwarded packets
        assert len(forwarded_packets) == 0
        # 5. Zero tracked background tasks
        assert len(plugin._background_tasks) == 0
        # 6. last_content remains empty
        assert plugin.last_content == ""

    async def test_packet_delivery_after_stop_strictly_rejected(self, adversarial_env) -> None:
        """Verify packet delivery after stop() leaves backend and registries clean."""
        plugin = adversarial_env["plugin"]
        backend = adversarial_env["backend"]
        events = adversarial_env["events"]

        # 1. Start plugin and establish valid state
        await plugin.start()
        assert plugin.is_running is True

        pkt1 = create_clipboard_packet("Active State Content", timestamp=2000)
        await plugin.handle_packet(pkt1, "peer_1", lambda p: None)
        assert await backend.get_text() == "Active State Content"
        assert plugin.get_connected_devices() == ["peer_1"]
        events.clear()

        # 2. Stop plugin cleanly
        await plugin.stop()
        assert plugin.is_running is False
        assert plugin.get_connected_devices() == []
        assert len(plugin._device_senders) == 0
        assert len(plugin._background_tasks) == 0

        # 3. Deliver new packet from new peer after stop()
        forwarded_pkts: List[Packet] = []
        pkt2 = create_clipboard_packet("Post Stop Infiltration", timestamp=3000)
        await plugin.handle_packet(pkt2, "peer_post_stop", lambda p: forwarded_pkts.append(p))

        # Strict assertions:
        # Backend still has previous content, NOT post-stop content
        assert await backend.get_text() == "Active State Content"
        # Peer was not added to tracked devices
        assert plugin.get_connected_devices() == []
        assert "peer_post_stop" not in plugin._device_senders
        # No events published
        assert len(events) == 0
        # No packets forwarded
        assert len(forwarded_pkts) == 0
        # Zero background tasks
        assert len(plugin._background_tasks) == 0

    async def test_secondary_methods_defended_when_stopped(self, adversarial_env) -> None:
        """Verify all secondary entry points return immediately when stopped."""
        plugin = adversarial_env["plugin"]
        backend = adversarial_env["backend"]
        events = adversarial_env["events"]

        # Ensure stopped
        assert plugin.is_running is False

        # 1. set_local_clipboard returns False
        ok = await plugin.set_local_clipboard("Programmatic Try")
        assert ok is False
        assert await backend.get_text() == ""

        # 2. _on_local_clipboard_changed no-ops
        await plugin._on_local_clipboard_changed("Simulated OS Change")
        assert plugin.last_content == ""
        assert len(events) == 0

        # 3. _on_device_connected does not track device or schedule tasks
        await plugin._on_device_connected(DeviceConnected(device_id="dev_ghost", device_name="Ghost"))
        assert plugin.get_connected_devices() == []
        assert len(plugin._background_tasks) == 0

        # 4. _broadcast_packet does not schedule tasks
        await plugin._broadcast_packet(create_clipboard_packet("Broadcast Try"))
        assert len(plugin._background_tasks) == 0

    async def test_concurrent_packet_flood_during_stop_lifecycle(self, adversarial_env) -> None:
        """Verify massive packet flood during stop() leaves zero leaked tasks or locks."""
        plugin = adversarial_env["plugin"]
        backend = adversarial_env["backend"]
        await plugin.start()

        # Connect 8 devices
        for i in range(8):
            plugin.add_connected_device(f"device_{i}", lambda p: None)

        stop_called = asyncio.Event()

        async def spammer(dev_idx: int) -> None:
            for j in range(40):
                if stop_called.is_set() and j > 20:
                    await asyncio.sleep(0.001)
                pkt = create_clipboard_packet(
                    f"Flood_{dev_idx}_{j}",
                    timestamp=50000 + dev_idx * 100 + j,
                )
                await plugin.handle_packet(pkt, f"device_{dev_idx}", lambda p: None)
                await asyncio.sleep(0.0005)

        # Launch 8 concurrent spammers (320 packets total)
        spammer_tasks = [asyncio.create_task(spammer(i)) for i in range(8)]

        # Let flood start
        await asyncio.sleep(0.01)

        # Trigger stop in parallel with flooding
        stop_called.set()
        await plugin.stop()

        # Await completion of all spammers
        await asyncio.gather(*spammer_tasks, return_exceptions=True)

        # Post-stop state verification
        assert plugin.is_running is False
        assert len(plugin._background_tasks) == 0, f"Leaked background tasks: {len(plugin._background_tasks)}"
        assert plugin.get_connected_devices() == []
        assert len(plugin._device_senders) == 0

    async def test_multi_cycle_restart_integrity(self, adversarial_env) -> None:
        """Verify repeatedly starting and stopping preserves isolation and functionality."""
        plugin = adversarial_env["plugin"]
        backend = adversarial_env["backend"]

        for cycle in range(1, 4):
            # Phase A: Stopped state rejection
            assert plugin.is_running is False
            pkt_stopped = create_clipboard_packet(f"Stopped_Payload_{cycle}", timestamp=cycle * 10000)
            await plugin.handle_packet(pkt_stopped, f"dev_stopped_{cycle}", lambda p: None)
            assert plugin.get_connected_devices() == []

            # Phase B: Start
            await plugin.start()
            assert plugin.is_running is True

            # Phase C: Normal packet accepted
            cur_ts = max(plugin.latest_timestamp + 100, cycle * 10000 + 500)
            pkt_active = create_clipboard_packet(f"Active_Payload_{cycle}", timestamp=cur_ts)
            await plugin.handle_packet(pkt_active, f"dev_active_{cycle}", lambda p: None)
            assert await backend.get_text() == f"Active_Payload_{cycle}"
            assert f"dev_active_{cycle}" in plugin.get_connected_devices()

            # Phase D: Surrogate packet rejected while active
            surrogate_pkt = create_clipboard_packet("Dirty\ud800Surrogate", timestamp=cur_ts + 100)
            await plugin.handle_packet(surrogate_pkt, f"dev_active_{cycle}", lambda p: None)
            # Backend must retain previous valid payload
            assert await backend.get_text() == f"Active_Payload_{cycle}"

            # Phase E: Stop
            await plugin.stop()
            assert plugin.is_running is False
            assert plugin.get_connected_devices() == []
            assert len(plugin._background_tasks) == 0


# =============================================================================
# 3. End-to-End Wire Framing & Fuzzing Challenge Suite
# =============================================================================

class TestEndToEndWireSurrogateAndLifecycleFuzzing:
    """End-to-end wire packet simulation and randomized surrogate fuzzing."""

    WIRE_SURROGATE_PAYLOADS = [
        # JSON wire encoded escape sequences
        b'{"id": 100, "type": "kdeconnect.clipboard", "body": {"content": "\\ud800", "timestamp": 1000}}\n',
        b'{"id": 101, "type": "kdeconnect.clipboard", "body": {"content": "\\udfff", "timestamp": 1001}}\n',
        b'{"id": 102, "type": "kdeconnect.clipboard", "body": {"content": "\\udc00\\ud800", "timestamp": 1002}}\n',
        b'{"id": 103, "type": "kdeconnect.clipboard", "body": {"content": "\\ud83d\\ud83d", "timestamp": 1003}}\n',
        b'{"id": 104, "type": "kdeconnect.clipboard", "body": {"content": "TextWith\\ud800Surrogate", "timestamp": 1004}}\n',
        b'{"id": 105, "type": "kdeconnect.clipboard.connect", "body": {"content": "\\ud800Connect", "timestamp": 1005}}\n',
    ]

    @pytest.mark.parametrize("wire_line", WIRE_SURROGATE_PAYLOADS)
    async def test_wire_parsed_surrogates_rejected_at_plugin_boundary(
        self, adversarial_env, wire_line: bytes
    ) -> None:
        """Wire bytes with JSON surrogate escapes parse cleanly but are safely dropped by plugin."""
        plugin = adversarial_env["plugin"]
        backend = adversarial_env["backend"]
        events = adversarial_env["events"]
        await plugin.start()

        # Parse wire bytes into Packet model
        packet = PacketEngine.parse(wire_line)
        assert isinstance(packet.body["content"], str)

        forwarded: List[Packet] = []
        await plugin.handle_packet(packet, "wire_peer", lambda p: forwarded.append(p))

        # Must not mutate backend, emit events, or forward
        assert await backend.get_text() == ""
        assert len(events) == 0
        assert len(forwarded) == 0

        await plugin.stop()

    async def test_multi_device_mesh_isolation_on_surrogate_attack(self, adversarial_env) -> None:
        """In a 5-device mesh, an attacker sending a surrogate must not reach the other 4 peers."""
        plugin = adversarial_env["plugin"]
        backend = adversarial_env["backend"]
        await plugin.start()

        received_by_peers: Dict[str, List[Packet]] = {f"peer_{i}": [] for i in range(1, 5)}
        for peer_id in received_by_peers:
            plugin.add_connected_device(peer_id, lambda p, pid=peer_id: received_by_peers[pid].append(p))

        # Attacker device sends surrogate
        attacker_pkt = create_clipboard_packet("Hostile\ud800Payload", timestamp=5000)
        await plugin.handle_packet(attacker_pkt, "attacker_dev", lambda p: None)

        # Verify none of the 4 peers received anything
        for peer_id, pkts in received_by_peers.items():
            assert len(pkts) == 0, f"Peer {peer_id} received forwarded surrogate packet!"

        assert await backend.get_text() == ""
        await plugin.stop()

    async def test_randomized_surrogate_fuzzing_harness(self, adversarial_env) -> None:
        """Generate 50 pseudorandom strings containing mixed surrogates and verify 100% rejection."""
        import random
        plugin = adversarial_env["plugin"]
        backend = adversarial_env["backend"]
        events = adversarial_env["events"]
        await plugin.start()

        rng = random.Random(42)  # Deterministic seed
        surrogate_pool = [
            "\ud800", "\ud801", "\ud900", "\udb7f", "\udb80", "\udbff",
            "\udc00", "\udc01", "\ude00", "\udeff", "\udffe", "\udfff",
        ]
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \t\n!@#$%^&*()_+😀🎉🔥"

        for iteration in range(50):
            # Construct random string with 1 to 5 surrogates randomly interspersed
            length = rng.randint(10, 200)
            chars = [rng.choice(alphabet) for _ in range(length)]
            num_surrogates = rng.randint(1, 5)
            for _ in range(num_surrogates):
                pos = rng.randint(0, len(chars))
                chars.insert(pos, rng.choice(surrogate_pool))
            fuzzed_str = "".join(chars)

            # Inbound evaluation must reject
            in_ok, reason = plugin.suppression.evaluate_inbound(fuzzed_str, 20000 + iteration, f"dev_fuzz_{iteration}")
            assert in_ok is False, f"Inbound accepted fuzzed surrogate at iteration {iteration}"
            assert reason == "corrupted_surrogate_encoding"

            # Outbound evaluation must reject
            out_ok, out_reason = plugin.suppression.evaluate_outbound_os_event(fuzzed_str)
            assert out_ok is False, f"Outbound accepted fuzzed surrogate at iteration {iteration}"
            assert out_reason == "corrupted_surrogate_encoding"

            # handle_packet must not raise or mutate backend
            pkt = create_clipboard_packet(fuzzed_str, timestamp=20000 + iteration)
            await plugin.handle_packet(pkt, f"dev_fuzz_{iteration}", lambda p: None)
            assert await backend.get_text() == ""

            # set_local_clipboard must return False
            assert await plugin.set_local_clipboard(fuzzed_str) is False

            # Redaction must not raise
            masked = redact_clipboard(fuzzed_str)
            assert "[REDACTED," in masked

        assert len(events) == 0
        await plugin.stop()

    async def test_rapid_lifecycle_flapping_stress(self, adversarial_env) -> None:
        """Rapidly flap start() and stop() 10 times under concurrent packet load."""
        plugin = adversarial_env["plugin"]
        backend = adversarial_env["backend"]

        for iteration in range(10):
            await plugin.start()
            assert plugin.is_running is True

            # Register 3 devices
            for d in range(3):
                plugin.add_connected_device(f"dev_flap_{iteration}_{d}", lambda p: None)

            # Send valid packet
            valid_ts = max(plugin.latest_timestamp + 10, iteration * 1000 + 1)
            valid_pkt = create_clipboard_packet(f"Flap_{iteration}", timestamp=valid_ts)
            await plugin.handle_packet(valid_pkt, f"dev_flap_{iteration}_0", lambda p: None)
            assert await backend.get_text() == f"Flap_{iteration}"

            # Stop
            await plugin.stop()
            assert plugin.is_running is False
            assert plugin.get_connected_devices() == []
            assert len(plugin._background_tasks) == 0

            # Send stopped packet
            stopped_pkt = create_clipboard_packet("Ignored_Flap", timestamp=valid_ts + 5)
            await plugin.handle_packet(stopped_pkt, f"dev_flap_{iteration}_1", lambda p: None)
            assert await backend.get_text() == f"Flap_{iteration}"
            assert plugin.get_connected_devices() == []

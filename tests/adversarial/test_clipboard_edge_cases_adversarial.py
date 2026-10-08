"""tests.adversarial.test_clipboard_edge_cases_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial stress-test suite for Milestone M9.2:
Edge Cases & Payload Stress Test of KDE Connect Clipboard Synchronization.

Test Dimensions:
1. Payloads near 120 KB ceiling (boundary checks, oversized rejection, multibyte straddling)
2. UTF-8 Unicode, astral plane codepoints, zero-width chars, emojis, null bytes, and surrogate handling
3. Empty, whitespace-only, and mixed line-ending edge cases
4. Rapid lifecycle start/stop cycles, concurrency storms, and task draining
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, List, Optional
import pytest

from pengulink.core.events import (
    ClipboardUpdatedEvent,
    EventBus,
)
from pengulink.network.packet import (
    MAX_PACKET_SIZE,
    Packet,
    PacketEngine,
)
from pengulink.plugins.clipboard import (
    CLIPBOARD_CONNECT_PACKET_TYPE,
    CLIPBOARD_PACKET_TYPE,
    MAX_CLIPBOARD_PACKET_BYTES,
    ClipboardPlugin,
    InMemoryClipboardBackend,
    LoopSuppressionEngine,
    create_clipboard_connect_packet,
    create_clipboard_packet,
    redact_clipboard,
)


# =============================================================================
# 1. Payloads Near 120 KB Ceiling (Boundary Checks & Oversized Rejection)
# =============================================================================

class TestPayloadCeilingBoundaries:
    """Stress-test the 120 KB text payload ceiling at exact boundary byte values."""

    @pytest.fixture
    def setup_plugin(self):
        backend = InMemoryClipboardBackend()
        bus = EventBus()
        plugin = ClipboardPlugin(event_bus=bus, backend=backend)
        return plugin, backend, bus

    async def test_exact_119999_bytes_accepted(self, setup_plugin) -> None:
        """119,999 bytes is 1 byte below the ceiling and must be accepted."""
        plugin, backend, bus = setup_plugin
        await plugin.start()

        payload = "A" * 119_999
        assert len(payload.encode("utf-8")) == 119_999

        # Inbound check
        acc, reason = plugin.suppression.evaluate_inbound(payload, 1000, "dev1")
        assert acc is True
        assert reason == "accepted"

        # Inbound handle_packet execution
        sent: List[Packet] = []
        pkt = create_clipboard_packet(payload, timestamp=1000)
        await plugin.handle_packet(pkt, "dev1", lambda p: sent.append(p))
        assert await backend.get_text() == payload
        assert plugin.last_content == payload

        # Outbound evaluation
        out_acc, out_reason = plugin.suppression.evaluate_outbound_os_event(payload)
        # Note: duplicate check will catch payload because it was already ingested, so test with new payload
        payload_alt = "B" * 119_999
        out_acc, out_reason = plugin.suppression.evaluate_outbound_os_event(payload_alt)
        assert out_acc is True
        assert out_reason == "accepted"

        # set_local_clipboard programmatic check
        set_ok = await plugin.set_local_clipboard(payload_alt)
        assert set_ok is True
        assert await backend.get_text() == payload_alt

        await plugin.stop()

    async def test_exact_120000_bytes_ceiling_accepted(self, setup_plugin) -> None:
        """Exact 120,000 bytes (MAX_CLIPBOARD_PACKET_BYTES) is on the boundary and must be accepted."""
        plugin, backend, bus = setup_plugin
        await plugin.start()

        payload = "C" * MAX_CLIPBOARD_PACKET_BYTES
        assert len(payload.encode("utf-8")) == 120_000

        acc, reason = plugin.suppression.evaluate_inbound(payload, 2000, "dev1")
        assert acc is True
        assert reason == "accepted"

        sent: List[Packet] = []
        pkt = create_clipboard_packet(payload, timestamp=2000)
        await plugin.handle_packet(pkt, "dev1", lambda p: sent.append(p))
        assert await backend.get_text() == payload

        payload_alt = "D" * MAX_CLIPBOARD_PACKET_BYTES
        set_ok = await plugin.set_local_clipboard(payload_alt)
        assert set_ok is True
        assert await backend.get_text() == payload_alt

        await plugin.stop()

    async def test_exact_120001_bytes_rejected(self, setup_plugin) -> None:
        """120,001 bytes is exactly 1 byte above the ceiling and must be strictly rejected."""
        plugin, backend, bus = setup_plugin
        await plugin.start()

        oversized = "E" * (MAX_CLIPBOARD_PACKET_BYTES + 1)
        assert len(oversized.encode("utf-8")) == 120_001

        # 1. Suppression engine inbound check
        acc, reason = plugin.suppression.evaluate_inbound(oversized, 3000, "dev1")
        assert acc is False
        assert "payload_exceeds_limit" in reason
        assert "120001 bytes > 120000 bytes" in reason

        # 2. Inbound packet execution discarded
        sent: List[Packet] = []
        pkt = create_clipboard_packet(oversized, timestamp=3000)
        await plugin.handle_packet(pkt, "dev1", lambda p: sent.append(p))
        assert await backend.get_text() == ""
        assert plugin.last_content == ""

        # 3. Outbound suppression
        out_acc, out_reason = plugin.suppression.evaluate_outbound_os_event(oversized)
        assert out_acc is False
        assert "payload_exceeds_limit" in out_reason

        # 4. Programmatic set_local_clipboard rejected
        set_ok = await plugin.set_local_clipboard(oversized)
        assert set_ok is False
        assert await backend.get_text() == ""

        await plugin.stop()

    async def test_150kb_and_massive_oversized_payloads_rejected(self, setup_plugin) -> None:
        """Test rejection of 150 KB (153,600 bytes), 500 KB, and 1 MB payloads."""
        plugin, backend, bus = setup_plugin
        await plugin.start()

        sizes = [
            150 * 1024,   # 150 KB = 153,600 bytes
            150_000,      # 150,000 decimal bytes
            500 * 1024,   # 500 KB
            1024 * 1024,  # 1 MB
        ]

        ts = 4000
        for sz in sizes:
            ts += 10
            payload = "X" * sz
            acc, reason = plugin.suppression.evaluate_inbound(payload, ts, "dev1")
            assert acc is False
            assert "payload_exceeds_limit" in reason
            assert f"{sz} bytes > 120000 bytes" in reason

            set_ok = await plugin.set_local_clipboard(payload)
            assert set_ok is False
            assert await backend.get_text() == ""

        await plugin.stop()

    async def test_multibyte_utf8_boundary_straddling(self, setup_plugin) -> None:
        """Multibyte UTF-8 characters must be measured by byte length, not char length."""
        plugin, backend, bus = setup_plugin
        await plugin.start()

        # 2-byte characters (Cyrillic 'ж', 2 bytes per char)
        # 60,000 chars * 2 bytes = 120,000 bytes -> ACCEPTED
        c2_valid = "ж" * 60_000
        assert len(c2_valid) == 60_000
        assert len(c2_valid.encode("utf-8")) == 120_000
        acc, _ = plugin.suppression.evaluate_inbound(c2_valid, 10000, "dev1")
        assert acc is True

        # 60,001 chars * 2 bytes = 120,002 bytes -> REJECTED
        c2_invalid = "ж" * 60_001
        assert len(c2_invalid.encode("utf-8")) == 120_002
        acc, reason = plugin.suppression.evaluate_inbound(c2_invalid, 10001, "dev1")
        assert acc is False
        assert "payload_exceeds_limit" in reason

        # 3-byte characters (CJK '中', 3 bytes per char)
        # 40,000 chars * 3 bytes = 120,000 bytes -> ACCEPTED
        c3_valid = "中" * 40_000
        assert len(c3_valid.encode("utf-8")) == 120_000
        acc, _ = plugin.suppression.evaluate_inbound(c3_valid, 10002, "dev1")
        assert acc is True

        # 40,001 chars * 3 bytes = 120,003 bytes -> REJECTED
        c3_invalid = "中" * 40_001
        assert len(c3_invalid.encode("utf-8")) == 120_003
        acc, reason = plugin.suppression.evaluate_inbound(c3_invalid, 10003, "dev1")
        assert acc is False
        assert "payload_exceeds_limit" in reason

        # 4-byte characters (Astral '🚀', 4 bytes per char)
        # 30,000 chars * 4 bytes = 120,000 bytes -> ACCEPTED
        c4_valid = "🚀" * 30_000
        assert len(c4_valid.encode("utf-8")) == 120_000
        acc, _ = plugin.suppression.evaluate_inbound(c4_valid, 10004, "dev1")
        assert acc is True

        # 30,001 chars * 4 bytes = 120,004 bytes -> REJECTED
        c4_invalid = "🚀" * 30_001
        assert len(c4_invalid.encode("utf-8")) == 120_004
        acc, reason = plugin.suppression.evaluate_inbound(c4_invalid, 10005, "dev1")
        assert acc is False
        assert "payload_exceeds_limit" in reason

        # Mixed ASCII + 4-byte character boundary
        # 119,996 ASCII + 1 4-byte char = 120,000 bytes -> ACCEPTED
        mixed_valid = "a" * 119_996 + "🐧"
        assert len(mixed_valid.encode("utf-8")) == 120_000
        acc, _ = plugin.suppression.evaluate_inbound(mixed_valid, 10006, "dev1")
        assert acc is True

        # 119,997 ASCII + 1 4-byte char = 120,001 bytes -> REJECTED
        mixed_invalid = "a" * 119_997 + "🐧"
        assert len(mixed_invalid.encode("utf-8")) == 120_001
        acc, reason = plugin.suppression.evaluate_inbound(mixed_invalid, 10007, "dev1")
        assert acc is False
        assert "payload_exceeds_limit" in reason

        await plugin.stop()

    def test_packet_envelope_framing_integrity_at_120kb(self) -> None:
        """A 120,000 byte payload packet must serialize within the 128 KB (131,072 bytes) framing ceiling."""
        payload = "Z" * MAX_CLIPBOARD_PACKET_BYTES
        pkt = create_clipboard_packet(payload, timestamp=123456789)
        wire_bytes = pkt.to_wire()

        # Must fit within 128 KB framing ceiling
        assert len(wire_bytes) <= MAX_PACKET_SIZE
        # Roundtrip parse through PacketEngine without buffer overrun
        parsed = PacketEngine.parse(wire_bytes)
        assert parsed.body["content"] == payload
        assert parsed.body["timestamp"] == 123456789


# =============================================================================
# 2. Complex Unicode: Astral Plane, Emojis, Zero-Width, and Surrogates
# =============================================================================

class TestUnicodeAstralAndSurrogates:
    """Stress-test complex Unicode variants, astral planes, zero-width characters, and surrogates."""

    @pytest.fixture
    def setup_plugin(self):
        backend = InMemoryClipboardBackend()
        bus = EventBus()
        plugin = ClipboardPlugin(event_bus=bus, backend=backend)
        return plugin, backend, bus

    async def test_astral_plane_codepoints_roundtrip(self, setup_plugin) -> None:
        """Astral plane codepoints U+10000..U+10FFFF must roundtrip without truncation."""
        plugin, backend, _ = setup_plugin
        await plugin.start()

        astral_samples = {
            "smp_linear_b": "\U00010000\U00010001\U00010002",
            "smp_hieroglyphs": "𓀀𓀁𓀂𓀃𓀄𓀅𓀆",
            "smp_musical": "𝄞𝄢𝄡𝄠𝄟𝄞",
            "smp_math_bold": "𝐀𝐁𝐂𝐃𝐄𝐅 𝐚𝐛𝐜𝐝𝐞𝐟 𝟎𝟏𝟐𝟑𝟒",
            "smp_mahjong_domino": "🀀🀁🀂🀃 🁣🁤🁥",
            "maximum_codepoint": "\U0010FFFF",
        }

        ts = 1000
        for name, text in astral_samples.items():
            ts += 10
            pkt = create_clipboard_packet(text, timestamp=ts)
            await plugin.handle_packet(pkt, "dev1", lambda p: None)
            assert await backend.get_text() == text
            assert plugin.last_content == text

        await plugin.stop()

    async def test_complex_multibyte_emoji_sequences(self, setup_plugin) -> None:
        """Complex emoji sequences (ZWJ, skin tones, flags, keycaps) must be handled safely."""
        plugin, backend, _ = setup_plugin
        await plugin.start()

        emoji_samples = [
            "👨‍👩‍👧‍👦",                        # Family ZWJ sequence (7 chars, 25 bytes)
            "👩‍❤️‍💋‍👨",                   # Couple with heart kiss (8 chars, 27 bytes)
            "🏳️‍🌈",                         # Rainbow flag (4 chars, 14 bytes)
            "🏴‍☠️",                         # Pirate flag (5 chars, 16 bytes)
            "👩🏿‍💻",                        # Woman technologist dark skin (4 chars, 15 bytes)
            "🇺🇸 🇬🇧 🇯🇵 🇩🇪 🇮🇳 🇫🇷",         # Regional flags
            "1️⃣ 2️⃣ 3️⃣ #️⃣ *️⃣",             # Keycap sequences
            "🤝🏻 🤝🏼 🤝🏽 🤝🏾 🤝🏿",          # Skin-tone modified gestures
        ]

        ts = 2000
        for em in emoji_samples:
            ts += 10
            pkt = create_clipboard_packet(em, timestamp=ts)
            await plugin.handle_packet(pkt, "dev1", lambda p: None)
            assert await backend.get_text() == em

        await plugin.stop()

    async def test_zero_width_and_bidi_characters(self, setup_plugin) -> None:
        """Zero-width spaces, joiners, BOM, and BiDi override controls must not corrupt state."""
        plugin, backend, _ = setup_plugin
        await plugin.start()

        zw_samples = [
            "\u200b",                         # Zero-width space
            "\u200c\u200d",                   # ZWNJ, ZWJ
            "\ufeff",                         # BOM (Zero-width no-break space)
            "\u2060",                         # Word joiner
            "\u202eRTL_OVERRIDE\u202c",       # Right-to-Left override and Pop Directional Formatting
            "\u200eLTR_MARK\u200fRTL_MARK",   # Directional marks
            "P\u200be\u200cn\u200dg\ufeffu\u2060l\u202ei\u202cnk", # Text interleaved with zero-width
        ]

        ts = 3000
        for zw in zw_samples:
            ts += 10
            pkt = create_clipboard_packet(zw, timestamp=ts)
            await plugin.handle_packet(pkt, "dev1", lambda p: None)
            assert await backend.get_text() == zw

        await plugin.stop()

    async def test_embedded_null_bytes_safety(self, setup_plugin) -> None:
        """Embedded null bytes (\\x00) must not cause C-string truncation or hashing discrepancies."""
        plugin, backend, _ = setup_plugin
        await plugin.start()

        null_samples = [
            "hello\x00world\x00!",
            "\x00" * 50,
            "prefix\x00\x00\x00\x00suffix",
        ]

        ts = 4000
        for text in null_samples:
            ts += 10
            pkt = create_clipboard_packet(text, timestamp=ts)
            await plugin.handle_packet(pkt, "dev1", lambda p: None)
            assert await backend.get_text() == text
            assert len(await backend.get_text()) == len(text)
            assert plugin.last_digest == hashlib.sha256(text.encode("utf-8")).hexdigest()

        await plugin.stop()

    def test_paired_surrogates_in_json_wire_decode_to_smp(self) -> None:
        """Paired surrogates in JSON wire format (e.g. \\ud83d\\ude00) cleanly decode to SMP 😀."""
        wire = b'{"type": "kdeconnect.clipboard", "body": {"content": "\\ud83d\\ude00", "timestamp": 5000}}\n'
        parsed = PacketEngine.parse(wire)
        assert parsed.body["content"] == "😀"
        assert len(parsed.body["content"]) == 1

    async def test_corrupted_lone_surrogates_failure_mode(self, setup_plugin) -> None:
        """Verify corrupted/lone surrogates are safely rejected without UnicodeEncodeError crashes.

        Remediation verification for Challenger 2 gate finding:
        When a packet body or input string contains unpaired surrogates (e.g. \\ud800 or \\udfff),
        evaluate_inbound, evaluate_outbound_os_event, set_local_clipboard, and handle_packet
        safely reject the payload without unhandled UnicodeEncodeError exceptions,
        and redact_clipboard safely masks the content with replacement characters.
        """
        plugin, backend, bus = setup_plugin
        await plugin.start()

        events: List[ClipboardUpdatedEvent] = []
        bus.subscribe(ClipboardUpdatedEvent, lambda e: events.append(e))

        corrupted_surrogates = [
            ("\ud800", "lone high surrogate"),
            ("\udfff", "lone low surrogate"),
            ("\ude00\ud83d", "inverted surrogate pair (low followed by high)"),
            ("\ud83d\ud83d", "double high surrogate"),
            ("\ud800\ud800", "double high surrogate duplicate"),
            ("\udfff\udfff", "double low surrogate duplicate"),
            ("ValidPrefix\ud800ValidSuffix", "embedded lone surrogate"),
        ]

        for s, desc in corrupted_surrogates:
            # 1. Inbound evaluation cleanly rejects without raising UnicodeEncodeError
            in_acc, in_reason = plugin.suppression.evaluate_inbound(s, 6000, "dev1")
            assert in_acc is False, f"Expected rejection for {desc}"
            assert in_reason == "corrupted_surrogate_encoding", f"Unexpected reason {in_reason} for {desc}"

            # 2. Outbound evaluation cleanly rejects without raising UnicodeEncodeError
            out_acc, out_reason = plugin.suppression.evaluate_outbound_os_event(s)
            assert out_acc is False, f"Expected rejection for {desc}"
            assert out_reason == "corrupted_surrogate_encoding", f"Unexpected reason {out_reason} for {desc}"

            # 3. set_local_clipboard safely returns False without raising UnicodeEncodeError
            set_ok = await plugin.set_local_clipboard(s)
            assert set_ok is False, f"Expected set_local_clipboard failure for {desc}"

            # 4. handle_packet safely discards without raising UnicodeEncodeError or mutating backend
            sent_packets: List[Packet] = []
            pkt = Packet(id=6000, type=CLIPBOARD_PACKET_TYPE, body={"content": s, "timestamp": 6000})
            await plugin.handle_packet(pkt, "dev1", lambda p: sent_packets.append(p))
            assert await backend.get_text() == "", f"Backend mutated by {desc}"
            assert len(sent_packets) == 0, f"Packet forwarded for {desc}"

            # 5. redact_clipboard safely formats without raising UnicodeEncodeError
            redacted = redact_clipboard(s)
            assert redacted.startswith("[REDACTED, "), f"Malformed redaction for {desc}"
            assert "sha256:" in redacted, f"Missing sha256 in redaction for {desc}"

        # 6. Ensure zero EventBus events were published for any surrogate payloads
        assert len(events) == 0, "ClipboardUpdatedEvent emitted for corrupted surrogate payload"

        await plugin.stop()


# =============================================================================
# 3. Empty, Whitespace-Only, and Mixed Line-Ending Edge Cases
# =============================================================================

class TestEmptyWhitespaceAndLineEndings:
    """Stress-test empty content, whitespace variants, and line-ending edge cases."""

    @pytest.fixture
    def setup_plugin(self):
        backend = InMemoryClipboardBackend()
        bus = EventBus()
        plugin = ClipboardPlugin(event_bus=bus, backend=backend)
        return plugin, backend, bus

    async def test_empty_string_inbound_vs_outbound_asymmetry(self, setup_plugin) -> None:
        """Inbound empty string is accepted; outbound empty string is suppressed."""
        plugin, backend, bus = setup_plugin
        await plugin.start()

        # Seed backend with some initial text
        await backend.set_text("Initial Text")

        # 1. Remote peer clears clipboard with empty string
        events: List[ClipboardUpdatedEvent] = []
        bus.subscribe(ClipboardUpdatedEvent, lambda e: events.append(e))

        pkt = create_clipboard_packet("", timestamp=1000)
        await plugin.handle_packet(pkt, "phone_1", lambda p: None)

        assert await backend.get_text() == ""
        assert plugin.last_content == ""
        assert len(events) == 1
        assert events[0].content == ""

        # 2. Local OS event with empty string must be suppressed (not broadcasted)
        acc, reason = plugin.suppression.evaluate_outbound_os_event("")
        assert acc is False
        assert reason == "empty_content"

        # 3. Programmatic set_local_clipboard("") clears backend and last_content
        await backend.set_text("Some text")
        set_ok = await plugin.set_local_clipboard("")
        assert set_ok is True
        assert await backend.get_text() == ""
        assert plugin.last_content == ""

        await plugin.stop()

    async def test_whitespace_only_strings_accepted_inbound_and_outbound(self, setup_plugin) -> None:
        """Whitespace-only strings (spaces, tabs, newlines) are truthy strings and accepted."""
        plugin, backend, _ = setup_plugin
        await plugin.start()

        whitespaces = [
            " ",
            "   ",
            "\t",
            "\t\t\t",
            "\n",
            "\r\n",
            " \t \r\n \n\t ",
        ]

        ts = 1000
        for ws in whitespaces:
            ts += 10
            # Inbound evaluation accepted
            in_acc, in_r = plugin.suppression.evaluate_inbound(ws, ts, "dev1")
            assert in_acc is True
            assert in_r == "accepted"

            # Inbound execution
            pkt = create_clipboard_packet(ws, timestamp=ts)
            await plugin.handle_packet(pkt, "dev1", lambda p: None)
            assert await backend.get_text() == ws

            # Outbound evaluation accepted
            out_acc, out_r = plugin.suppression.evaluate_outbound_os_event(ws + "extra")
            assert out_acc is True
            assert out_r == "accepted"

        await plugin.stop()

    async def test_line_ending_permutations_and_layer2_echo_matching(self, setup_plugin) -> None:
        """Layer 2 echo suppression normalizes \\r\\n to \\n and rstrips to match OS reflections."""
        engine = LoopSuppressionEngine(write_in_flight_ttl=1.0)

        remote_clip = "Line 1\r\nLine 2\r\nLine 3\r\n   "
        engine.arm_write_in_flight(remote_clip)

        # 1. Exact raw reflection matches
        assert engine.is_write_in_flight_active(remote_clip) is True

        # 2. OS normalized reflection with \n instead of \r\n and stripped trailing spaces matches
        os_reflected = "Line 1\nLine 2\nLine 3"
        assert engine.is_write_in_flight_active(os_reflected) is True

        # 3. Genuinely different text does NOT match
        assert engine.is_write_in_flight_active("Line 1\nLine 2\nLine 4") is False

    async def test_huge_whitespace_ceiling_boundary(self, setup_plugin) -> None:
        """Whitespace payloads at the 120 KB boundary."""
        plugin, backend, _ = setup_plugin
        await plugin.start()

        # 120,000 spaces -> accepted
        exact_120k_spaces = " " * MAX_CLIPBOARD_PACKET_BYTES
        acc, _ = plugin.suppression.evaluate_inbound(exact_120k_spaces, 1000, "dev1")
        assert acc is True

        # 120,001 spaces -> rejected
        over_spaces = " " * (MAX_CLIPBOARD_PACKET_BYTES + 1)
        acc, reason = plugin.suppression.evaluate_inbound(over_spaces, 2000, "dev1")
        assert acc is False
        assert "payload_exceeds_limit" in reason

        await plugin.stop()


# =============================================================================
# 4. Rapid Lifecycle Start/Stop Cycles, Concurrency Storms & Task Draining
# =============================================================================

class TestRapidLifecycleAndTaskDraining:
    """Stress-test rapid lifecycle churn, concurrent start/stop, and background task draining."""

    @pytest.fixture
    def setup_plugin(self):
        backend = InMemoryClipboardBackend()
        bus = EventBus()
        plugin = ClipboardPlugin(event_bus=bus, backend=backend)
        return plugin, backend, bus

    async def test_rapid_start_stop_50_cycles(self, setup_plugin) -> None:
        """50 rapid start/stop cycles must execute cleanly without leaking tasks or memory."""
        plugin, backend, _ = setup_plugin

        for i in range(50):
            await plugin.start()
            assert plugin.is_running is True
            assert len(plugin._unsub_callbacks) == 3

            await plugin.stop()
            assert plugin.is_running is False
            assert len(plugin._unsub_callbacks) == 0
            assert len(plugin._background_tasks) == 0

    async def test_idempotent_multiple_starts_and_stops(self, setup_plugin) -> None:
        """Repeated sequential calls to start() and stop() must be strictly idempotent."""
        plugin, _, _ = setup_plugin

        # 5 consecutive start() calls
        for _ in range(5):
            await plugin.start()
            assert plugin.is_running is True
            assert len(plugin._unsub_callbacks) == 3

        # 5 consecutive stop() calls
        for _ in range(5):
            await plugin.stop()
            assert plugin.is_running is False
            assert len(plugin._unsub_callbacks) == 0
            assert len(plugin._background_tasks) == 0

    async def test_background_task_cancellation_and_draining_under_slow_sender(self, setup_plugin) -> None:
        """Plugin stop() must cancel and await all in-flight broadcast tasks."""
        plugin, backend, _ = setup_plugin
        await plugin.start()

        started_slow_sends = 0
        slow_send_completed = 0
        slow_send_cancelled = 0

        async def slow_sender(p: Packet) -> None:
            nonlocal started_slow_sends, slow_send_completed, slow_send_cancelled
            started_slow_sends += 1
            try:
                await asyncio.sleep(0.5)
                slow_send_completed += 1
            except asyncio.CancelledError:
                slow_send_cancelled += 1
                raise

        # Add 5 devices with slow sender
        for i in range(5):
            plugin.add_connected_device(f"slow_dev_{i}", slow_sender)

        # Trigger outbound broadcast
        await backend.simulate_external_copy("Slow Broadcast Payload")
        # Give asyncio loop a moment to schedule broadcast tasks
        await asyncio.sleep(0.02)

        assert len(plugin._background_tasks) == 5
        assert started_slow_sends == 5

        # Stop plugin immediately: all background tasks must be cancelled and drained
        await plugin.stop()

        assert len(plugin._background_tasks) == 0
        assert slow_send_completed == 0
        assert slow_send_cancelled == 5

    async def test_concurrent_start_stop_storm(self, setup_plugin) -> None:
        """Concurrent storm of 20 tasks calling start() and stop() simultaneously."""
        plugin, backend, _ = setup_plugin

        async def worker(idx: int) -> None:
            for _ in range(10):
                if idx % 2 == 0:
                    await plugin.start()
                else:
                    await plugin.stop()
                await asyncio.sleep(0.001)

        tasks = [asyncio.create_task(worker(i)) for i in range(20)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Clean final shutdown
        await plugin.stop()
        assert plugin.is_running is False
        assert len(plugin._background_tasks) == 0
        assert len(plugin._unsub_callbacks) == 0

    async def test_packet_handling_when_plugin_stopped(self, setup_plugin) -> None:
        """Verify packet handling rejection when plugin is stopped.

        Packets received while stopped must be cleanly discarded without mutating backend
        state, emitting events, or leaking background tasks.
        """
        plugin, backend, bus = setup_plugin
        await plugin.start()

        events: List[ClipboardUpdatedEvent] = []
        bus.subscribe(ClipboardUpdatedEvent, lambda e: events.append(e))

        await plugin.stop()
        assert plugin.is_running is False

        pkt = create_clipboard_packet("Received while stopped", timestamp=1000)
        sent: List[Packet] = []
        await plugin.handle_packet(pkt, "dev1", lambda p: sent.append(p))

        # Backend is NOT updated
        assert await backend.get_text() == ""
        assert plugin.last_content == ""
        # Ephemeral device maps remain empty
        assert plugin.get_connected_devices() == []
        assert len(plugin._device_senders) == 0
        # Background tasks set is clean
        assert len(plugin._background_tasks) == 0
        assert len(sent) == 0
        assert len(events) == 0


# =============================================================================
# 5. Rigorous Surrogate & Lifecycle Edge Cases (M9.2 Iteration 2 Gate Hardening)
# =============================================================================

class TestSurrogateAndLifecycleGateHardening:
    """Rigorous gate-hardening test suite preventing regressions in surrogate resilience and stopped guards."""

    @pytest.fixture
    def setup_plugin(self):
        backend = InMemoryClipboardBackend()
        bus = EventBus()
        plugin = ClipboardPlugin(event_bus=bus, backend=backend)
        return plugin, backend, bus

    @pytest.mark.parametrize(
        "surrogate_sample,description",
        [
            ("\ud800", "minimum high surrogate U+D800"),
            ("\ud801", "high surrogate U+D801"),
            ("\udb7f", "high surrogate U+DB7F"),
            ("\udb80", "high private surrogate U+DB80"),
            ("\udbff", "maximum high surrogate U+DBFF"),
            ("\udc00", "minimum low surrogate U+DC00"),
            ("\udc01", "low surrogate U+DC01"),
            ("\udfff", "maximum low surrogate U+DFFF"),
            ("\ude00\ud83d", "inverted pair U+DE00 U+D83D"),
            ("\ud83d\ud83d", "double high surrogate"),
            ("\ude00\ude00", "double low surrogate"),
            ("A" * 100 + "\ud800" + "B" * 100, "surrogate embedded in 200 ascii chars"),
            ("\ud800" * 50, "string composed exclusively of 50 high surrogates"),
        ],
    )
    async def test_comprehensive_surrogate_samples_rejected(
        self, setup_plugin, surrogate_sample: str, description: str
    ) -> None:
        """Every surrogate permutation must be rejected cleanly by all inbound and outbound paths."""
        plugin, backend, _ = setup_plugin
        await plugin.start()

        # Inbound evaluation returns False
        in_acc, in_reason = plugin.suppression.evaluate_inbound(surrogate_sample, 1000, "dev1")
        assert in_acc is False, f"Failed for {description}"
        assert in_reason == "corrupted_surrogate_encoding", f"Failed for {description}"

        # Outbound evaluation returns False
        out_acc, out_reason = plugin.suppression.evaluate_outbound_os_event(surrogate_sample)
        assert out_acc is False, f"Failed for {description}"
        assert out_reason == "corrupted_surrogate_encoding", f"Failed for {description}"

        # Programmatic write returns False
        assert await plugin.set_local_clipboard(surrogate_sample) is False

        # Inbound packet execution leaves backend clean
        pkt = create_clipboard_packet(surrogate_sample, timestamp=1000)
        sent: List[Packet] = []
        await plugin.handle_packet(pkt, "dev1", lambda p: sent.append(p))
        assert await backend.get_text() == ""
        assert len(sent) == 0

        # Redact formatting never raises
        masked = redact_clipboard(surrogate_sample)
        assert "[REDACTED," in masked

        await plugin.stop()

    async def test_re_activation_lifecycle_cycle(self, setup_plugin) -> None:
        """Verify full transition cycle: Unstarted -> Running -> Stopped -> Re-started."""
        plugin, backend, _ = setup_plugin

        # 1. Unstarted: packet dropped
        pkt1 = create_clipboard_packet("Unstarted text", timestamp=1000)
        await plugin.handle_packet(pkt1, "dev1", lambda p: None)
        assert await backend.get_text() == ""

        # 2. Start: packet accepted
        await plugin.start()
        assert plugin.is_running is True
        pkt2 = create_clipboard_packet("Running text", timestamp=2000)
        await plugin.handle_packet(pkt2, "dev1", lambda p: None)
        assert await backend.get_text() == "Running text"

        # 3. Stop: packet dropped
        await plugin.stop()
        assert plugin.is_running is False
        pkt3 = create_clipboard_packet("Stopped text", timestamp=3000)
        await plugin.handle_packet(pkt3, "dev1", lambda p: None)
        assert await backend.get_text() == "Running text"  # Still previous text!

        # 4. Re-start: packet accepted again
        await plugin.start()
        assert plugin.is_running is True
        ts4 = plugin.suppression.local_timestamp + 1000
        pkt4 = create_clipboard_packet("Reactivated text", timestamp=ts4)
        await plugin.handle_packet(pkt4, "dev1", lambda p: None)
        assert await backend.get_text() == "Reactivated text"

        await plugin.stop()

    async def test_concurrent_packets_during_and_after_stop(self, setup_plugin) -> None:
        """High-concurrency packet flood arriving concurrently as stop() is triggered."""
        plugin, backend, _ = setup_plugin
        await plugin.start()

        # Add 5 devices
        for i in range(5):
            plugin.add_connected_device(f"dev_{i}", lambda p: None)

        async def worker(idx: int) -> None:
            for j in range(20):
                pkt = create_clipboard_packet(f"Payload_{idx}_{j}", timestamp=10000 + idx * 100 + j)
                await plugin.handle_packet(pkt, f"dev_{idx}", lambda p: None)
                await asyncio.sleep(0.001)

        tasks = [asyncio.create_task(worker(i)) for i in range(5)]
        # Allow worker tasks to start
        await asyncio.sleep(0.005)

        # Trigger stop while tasks are blasting
        await plugin.stop()
        await asyncio.gather(*tasks, return_exceptions=True)

        # Final state must be completely clean
        assert plugin.is_running is False
        assert len(plugin._background_tasks) == 0
        assert plugin.get_connected_devices() == []
        assert len(plugin._device_senders) == 0

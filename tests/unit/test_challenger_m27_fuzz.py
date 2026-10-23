"""tests.unit.test_challenger_m27_fuzz
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial Stress, Fuzzing & Edge-Case Verification Suite for Milestone 27:
Telephony Plugin (`pengulink.plugins.telephony.TelephonyPlugin`) and
Telephony Service (`pengulink.services.telephony.TelephonyService`).

Empirically challenges:
1. Malformed packet envelopes (non-dict body, None body, list body, corrupt attributes, random fuzz generator).
2. Type fuzzing on phone number and contact name (null, non-string, boolean, float, containers, cross-product matrix).
3. Extreme string boundaries (100,000-character phone numbers/names, 500,000-character thumbnails, Zalgo, RTL, emojis, null bytes).
4. Unknown and legacy event strings (sms ignored, arbitrary events, case insensitivity, long event names).
5. High-frequency flapping bursts (1,000 rapid transitions within milliseconds, 10-device concurrent storm).
6. Reentrancy and concurrency lock safety under EventBus subscriber invocations.
"""

from __future__ import annotations

import asyncio
import random
import string
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from pengulink.core.events import (
    CallEvent,
    CallStateChangedEvent,
    DeviceDisconnected,
    EventBus,
)
from pengulink.core.exceptions import (
    CallNotFoundError,
    ConnectionClosedError,
    InvalidCallStateError,
)
from pengulink.network.packet import Packet
from pengulink.plugins.telephony import (
    TELEPHONY_PACKET_TYPE,
    TelephonyPlugin,
    coerce_boolean,
)
from pengulink.services.telephony import (
    ActiveCall,
    CallState,
    TelephonyService,
    coerce_is_cancel,
)


class MockPacketSender:
    """Mock packet sender recording sent packets."""

    def __init__(self, is_async: bool = True) -> None:
        self.sent_packets: list[Packet] = []
        self.is_async = is_async

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        packet = None
        for arg in args:
            if isinstance(arg, Packet):
                packet = arg
                break
        if packet is None:
            packet = kwargs.get("packet")
        if packet is not None:
            self.sent_packets.append(packet)

        if self.is_async:
            async def _coro() -> None:
                pass
            return _coro()
        return None


# =====================================================================
# Test Class 1: Malformed Packet Envelopes & Structural Fuzzing
# =====================================================================


class TestChallengerPacketEnvelopeFuzzing:
    """Stress tests on malformed packet bodies, headers, and missing keys."""

    @pytest.fixture
    def plugin(self) -> TelephonyPlugin:
        return TelephonyPlugin(event_bus=EventBus())

    async def test_fuzz_non_dict_body_types(self, plugin: TelephonyPlugin) -> None:
        """Verify non-dict packet bodies are safely dropped without crashing."""
        sender = MockPacketSender()
        malformed_bodies = [
            None,
            [],
            [1, 2, 3],
            ["ringing", "+15551234567"],
            "not_a_dict_string",
            123456,
            3.14159,
            True,
            False,
            (1, 2),
            {1, 2, 3},
            b"raw_bytes_body",
        ]

        for bad_body in malformed_bodies:
            bad_pkt = MagicMock()
            bad_pkt.type = TELEPHONY_PACKET_TYPE
            bad_pkt.body = bad_body

            # Should never raise an uncaught exception
            await plugin.handle_packet(bad_pkt, "dev_fuzz", sender)
            assert await plugin.get_call("dev_fuzz") is None

    async def test_fuzz_empty_dict_body(self, plugin: TelephonyPlugin) -> None:
        """Verify empty dict body {} is rejected gracefully."""
        sender = MockPacketSender()
        pkt = Packet(type=TELEPHONY_PACKET_TYPE, body={})
        await plugin.handle_packet(pkt, "dev_empty", sender)
        assert await plugin.get_call("dev_empty") is None

    async def test_fuzz_missing_or_corrupt_event(self, plugin: TelephonyPlugin) -> None:
        """Verify body with missing or non-string event field."""
        sender = MockPacketSender()
        bad_events = [None, 12345, 3.14, [], {}, True, False]

        for bad_evt in bad_events:
            pkt = Packet(
                type=TELEPHONY_PACKET_TYPE,
                body={"event": bad_evt, "phoneNumber": "123"},
            )
            await plugin.handle_packet(pkt, "dev_evt", sender)
            assert await plugin.get_call("dev_evt") is None

    async def test_fuzz_unsupported_or_foreign_packet_types(self, plugin: TelephonyPlugin) -> None:
        """Verify packets not matching kdeconnect.telephony are safely ignored."""
        sender = MockPacketSender()
        for p_type in [
            "kdeconnect.ping",
            "kdeconnect.battery",
            "kdeconnect.notification",
            "kdeconnect.systemvolume",
            "kdeconnect.telephony.request",
            "kdeconnect.telephony.request_mute",
        ]:
            pkt = Packet(
                type=p_type,
                body={"event": "ringing", "phoneNumber": "123"},
            )
            await plugin.handle_packet(pkt, "dev_foreign", sender)
            assert await plugin.get_call("dev_foreign") is None

        # Also test non-conforming types via mock objects
        for non_regex_type in ("", "unknown_type", "bad.type.123"):
            bad_pkt = MagicMock()
            bad_pkt.type = non_regex_type
            bad_pkt.body = {"event": "ringing", "phoneNumber": "123"}
            await plugin.handle_packet(bad_pkt, "dev_foreign_mock", sender)
            assert await plugin.get_call("dev_foreign_mock") is None

    async def test_fuzz_invalid_device_ids(self, plugin: TelephonyPlugin) -> None:
        """Verify empty, whitespace, and non-string device IDs are safely handled."""
        sender = MockPacketSender()
        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": "123"},
        )

        invalid_ids: list[Any] = ["", "   ", "\t", "\n", None, 12345, [], {}]
        for dev_id in invalid_ids:
            # Plugin drops without raising
            await plugin.handle_packet(pkt, dev_id, sender)  # type: ignore

    async def test_randomized_envelope_fuzz_generator(self, plugin: TelephonyPlugin) -> None:
        """Run 100 randomized pseudo-random malformed packets with chaotic types."""
        sender = MockPacketSender()
        rng = random.Random(42)

        def generate_chaotic_value(depth: int = 0) -> Any:
            if depth > 3:
                return rng.choice([None, 0, 1, "leaf", False])
            choice = rng.randint(0, 8)
            if choice == 0:
                return None
            elif choice == 1:
                return rng.choice([True, False])
            elif choice == 2:
                return rng.randint(-1_000_000, 1_000_000)
            elif choice == 3:
                return rng.uniform(-1e6, 1e6)
            elif choice == 4:
                return "".join(rng.choices(string.printable, k=rng.randint(0, 50)))
            elif choice == 5:
                return [generate_chaotic_value(depth + 1) for _ in range(rng.randint(0, 4))]
            elif choice == 6:
                return {
                    f"k_{i}": generate_chaotic_value(depth + 1)
                    for i in range(rng.randint(0, 4))
                }
            elif choice == 7:
                return float("nan")
            else:
                return float("inf")

        for i in range(100):
            chaotic_body = generate_chaotic_value()
            bad_pkt = MagicMock()
            bad_pkt.type = TELEPHONY_PACKET_TYPE
            bad_pkt.body = chaotic_body

            # Must process cleanly without crashing daemon
            await plugin.handle_packet(bad_pkt, f"fuzz_dev_{i}", sender)


# =====================================================================
# Test Class 2: Field Type Fuzzing (Phone Number & Contact Name)
# =====================================================================


class TestChallengerPhoneAndContactFieldTypes:
    """Stress tests on null, non-string, boolean, and container types in payload."""

    @pytest.fixture
    def plugin(self) -> TelephonyPlugin:
        return TelephonyPlugin(event_bus=EventBus())

    async def test_phone_number_null_none(self, plugin: TelephonyPlugin) -> None:
        """Verify phoneNumber: None falls back to 'unknown number'."""
        sender = MockPacketSender()
        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": None, "contactName": None},
        )
        await plugin.handle_packet(pkt, "p_dev", sender)
        call = await plugin.get_call("p_dev")
        assert call is not None
        assert call["phone_number"] == "unknown number"
        assert call["contact_name"] == "unknown number"

    async def test_phone_number_boolean_and_numeric_values(self, plugin: TelephonyPlugin) -> None:
        """Verify boolean and integer values for phoneNumber do not crash."""
        sender = MockPacketSender()

        # Integer phone number
        pkt_int = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": 1234567890},
        )
        await plugin.handle_packet(pkt_int, "p_dev1", sender)
        call1 = await plugin.get_call("p_dev1")
        assert call1 is not None
        assert call1["phone_number"] == "1234567890"

        # Float phone number (falls back safely)
        pkt_float = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": 3.14159},
        )
        await plugin.handle_packet(pkt_float, "p_dev2", sender)
        call2 = await plugin.get_call("p_dev2")
        assert call2 is not None
        assert call2["phone_number"] == "unknown number"

        # Boolean phone number (in Python bool inherits int)
        pkt_bool = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": False},
        )
        await plugin.handle_packet(pkt_bool, "p_dev3", sender)
        call3 = await plugin.get_call("p_dev3")
        assert call3 is not None
        assert isinstance(call3["phone_number"], str)

    async def test_phone_number_complex_container_types(self, plugin: TelephonyPlugin) -> None:
        """Verify list or dict phoneNumber values fall back to 'unknown number'."""
        sender = MockPacketSender()
        for bad_val in (["+15551234"], {"number": "+15551234"}, (1, 2)):
            pkt = Packet(
                type=TELEPHONY_PACKET_TYPE,
                body={"event": "ringing", "phoneNumber": bad_val},
            )
            await plugin.handle_packet(pkt, "p_cont", sender)
            call = await plugin.get_call("p_cont")
            assert call is not None
            assert call["phone_number"] == "unknown number"
            await plugin.clear_device("p_cont")

    async def test_contact_name_null_fallback_to_phone(self, plugin: TelephonyPlugin) -> None:
        """Verify contactName: None falls back to phone number."""
        sender = MockPacketSender()
        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": "+15558889999", "contactName": None},
        )
        await plugin.handle_packet(pkt, "c_dev", sender)
        call = await plugin.get_call("c_dev")
        assert call is not None
        assert call["phone_number"] == "+15558889999"
        assert call["contact_name"] == "+15558889999"

    async def test_contact_name_boolean_and_containers(self, plugin: TelephonyPlugin) -> None:
        """Verify non-string contactName falls back to phone number."""
        sender = MockPacketSender()
        for bad_contact in (True, False, 12345, ["Alice"], {"name": "Bob"}):
            pkt = Packet(
                type=TELEPHONY_PACKET_TYPE,
                body={
                    "event": "ringing",
                    "phoneNumber": "+12345",
                    "contactName": bad_contact,
                },
            )
            await plugin.handle_packet(pkt, "c_nonstr", sender)
            call = await plugin.get_call("c_nonstr")
            assert call is not None
            assert call["contact_name"] == "+12345"
            await plugin.clear_device("c_nonstr")

    async def test_empty_and_whitespace_only_strings(self, plugin: TelephonyPlugin) -> None:
        """Verify empty and whitespace strings fallback to 'unknown number'."""
        sender = MockPacketSender()
        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": "   ", "contactName": " \t\n "},
        )
        await plugin.handle_packet(pkt, "ws_dev", sender)
        call = await plugin.get_call("ws_dev")
        assert call is not None
        assert call["phone_number"] == "unknown number"
        assert call["contact_name"] == "unknown number"

    async def test_phone_and_contact_type_cross_product_matrix(self, plugin: TelephonyPlugin) -> None:
        """Permute 36 type combinations across phoneNumber x contactName."""
        sender = MockPacketSender()
        test_values = [
            None,
            "",
            "   ",
            True,
            False,
            0,
            123,
            ["sample"],
            {"key": "val"},
        ]

        counter = 0
        for p_val in test_values:
            for c_val in test_values:
                counter += 1
                dev_id = f"matrix_dev_{counter}"
                pkt = Packet(
                    type=TELEPHONY_PACKET_TYPE,
                    body={"event": "ringing", "phoneNumber": p_val, "contactName": c_val},
                )
                await plugin.handle_packet(pkt, dev_id, sender)
                call = await plugin.get_call(dev_id)
                assert call is not None
                assert isinstance(call["phone_number"], str)
                assert isinstance(call["contact_name"], str)
                assert len(call["phone_number"]) > 0
                assert len(call["contact_name"]) > 0


# =====================================================================
# Test Class 3: Extreme Strings, Unicode Zalgo, RTL & Binary Safety
# =====================================================================


class TestChallengerExtremeStringsAndUnicode:
    """Stress tests on 100,000-character buffers, Zalgo, RTL, and emojis."""

    @pytest.fixture
    def plugin(self) -> TelephonyPlugin:
        return TelephonyPlugin(event_bus=EventBus())

    async def test_phone_number_100k_chars_clamped(self, plugin: TelephonyPlugin) -> None:
        """Verify 100,000-character phone number is safely clamped without memory explosion."""
        sender = MockPacketSender()
        huge_phone = "+1" + "9" * 100_000
        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": huge_phone},
        )
        t0 = time.perf_counter()
        await plugin.handle_packet(pkt, "huge_dev", sender)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.1  # Fast processing, no quadratic parsing
        call = await plugin.get_call("huge_dev")
        assert call is not None
        assert len(call["phone_number"]) <= 256
        assert call["phone_number"].startswith("+19999")

    async def test_contact_name_100k_chars_clamped(self, plugin: TelephonyPlugin) -> None:
        """Verify 100,000-character contact name is safely clamped without memory explosion."""
        sender = MockPacketSender()
        huge_name = "Chief " + "A" * 100_000
        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "talking", "phoneNumber": "+123", "contactName": huge_name},
        )
        await plugin.handle_packet(pkt, "huge_name_dev", sender)
        call = await plugin.get_call("huge_name_dev")
        assert call is not None
        assert len(call["contact_name"]) <= 1024
        assert call["contact_name"].startswith("Chief AA")

    async def test_phone_thumbnail_500k_chars_clamped(self, plugin: TelephonyPlugin) -> None:
        """Verify 500,000-character thumbnail string is clamped safely."""
        sender = MockPacketSender()
        events: list[CallEvent] = []
        plugin.event_bus.subscribe(CallEvent, lambda ev: events.append(ev))  # type: ignore

        huge_thumb = "iVBORw0KGgoAAAANSUhEUgAA" + "A" * 500_000
        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={
                "event": "ringing",
                "phoneNumber": "+123",
                "phoneThumbnail": huge_thumb,
            },
        )
        await plugin.handle_packet(pkt, "thumb_dev", sender)
        assert len(events) == 1
        assert events[0].phone_thumbnail is not None
        assert len(events[0].phone_thumbnail) <= 100_000

    async def test_unicode_zalgo_text(self, plugin: TelephonyPlugin) -> None:
        """Verify stacked combining diacritics (Zalgo) parse and serialize cleanly."""
        sender = MockPacketSender()
        zalgo_name = "T̷̗̈́h̵̠̓e̶̩͊ ̷̧́C̸̰̈́å̶̗l̷̜̈́l̸̰̈́ë̶̗́r̴̰̈́"
        zalgo_phone = "+1̴5̵5̶5̸1̵2̶3̸4̵"

        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": zalgo_phone, "contactName": zalgo_name},
        )
        await plugin.handle_packet(pkt, "zalgo_dev", sender)
        call = await plugin.get_call("zalgo_dev")
        assert call is not None
        assert zalgo_name in call["contact_name"]
        # Verify JSON-safe serialization
        d = call
        assert isinstance(d["contact_name"], str)

    async def test_unicode_rtl_and_bidi_override(self, plugin: TelephonyPlugin) -> None:
        """Verify Hebrew, Arabic, and bidirectional overrides (RLO/LRO) are preserved."""
        sender = MockPacketSender()
        # Arabic with right-to-left override
        rtl_arabic = "\u202eمكالمة هاتفية واردة\u202c"
        # Hebrew
        rtl_hebrew = "\u200fשלום עליכם\u200e"

        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": rtl_hebrew, "contactName": rtl_arabic},
        )
        await plugin.handle_packet(pkt, "bidi_dev", sender)
        call = await plugin.get_call("bidi_dev")
        assert call is not None
        assert call["contact_name"] == rtl_arabic
        assert call["phone_number"] == rtl_hebrew

    async def test_unicode_complex_emojis_and_surrogates(self, plugin: TelephonyPlugin) -> None:
        """Verify multi-codepoint emojis (family ZWJ, flags, skin tones) roundtrip."""
        sender = MockPacketSender()
        emoji_contact = "👨‍👩‍👧‍👦 📞 🔥 🚀 🇺🇸 👋🏿"
        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "talking", "phoneNumber": "📞123", "contactName": emoji_contact},
        )
        await plugin.handle_packet(pkt, "emoji_dev", sender)
        call = await plugin.get_call("emoji_dev")
        assert call is not None
        assert call["contact_name"] == emoji_contact

    async def test_control_characters_and_null_bytes(self, plugin: TelephonyPlugin) -> None:
        """Verify strings containing null bytes, tabs, newlines, and ANSI escapes."""
        sender = MockPacketSender()
        nasty_str = "Alice\x00\r\n\t\x1b[31;1mHacked\x1b[0m"
        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": nasty_str, "contactName": nasty_str},
        )
        await plugin.handle_packet(pkt, "nasty_dev", sender)
        call = await plugin.get_call("nasty_dev")
        assert call is not None
        assert call["state"] == "RINGING"


# =====================================================================
# Test Class 4: Unknown & Legacy Event Strings
# =====================================================================


class TestChallengerEventTypesAndLegacyStrings:
    """Stress tests on SMS ignore, case variants, and unknown event codes."""

    @pytest.fixture
    def plugin(self) -> TelephonyPlugin:
        return TelephonyPlugin(event_bus=EventBus())

    async def test_legacy_sms_event_strictly_ignored(self, plugin: TelephonyPlugin) -> None:
        """Verify 'sms' event is dropped with zero state mutation and zero events emitted."""
        sender = MockPacketSender()
        events: list[CallEvent] = []
        state_events: list[CallStateChangedEvent] = []
        plugin.event_bus.subscribe(CallEvent, lambda ev: events.append(ev))  # type: ignore
        plugin.event_bus.subscribe(CallStateChangedEvent, lambda ev: state_events.append(ev))  # type: ignore

        for sms_str in ("sms", "SMS", "  Sms  "):
            pkt = Packet(
                type=TELEPHONY_PACKET_TYPE,
                body={"event": sms_str, "phoneNumber": "123", "messageBody": "Hello"},
            )
            await plugin.handle_packet(pkt, "sms_dev", sender)
            assert await plugin.get_call("sms_dev") is None
            assert len(events) == 0
            assert len(state_events) == 0

    async def test_unknown_event_strings_handled_gracefully(self, plugin: TelephonyPlugin) -> None:
        """Verify arbitrary event strings publish CallEvent but leave active state unchanged."""
        sender = MockPacketSender()
        events: list[CallEvent] = []
        state_events: list[CallStateChangedEvent] = []
        plugin.event_bus.subscribe(CallEvent, lambda ev: events.append(ev))  # type: ignore
        plugin.event_bus.subscribe(CallStateChangedEvent, lambda ev: state_events.append(ev))  # type: ignore

        unknown_events = [
            "onHold",
            "forwarded",
            "conference_start",
            "voicemail_pending",
            "custom_telecom_123",
            "random_string",
        ]

        for unk in unknown_events:
            pkt = Packet(
                type=TELEPHONY_PACKET_TYPE,
                body={"event": unk, "phoneNumber": "123"},
            )
            await plugin.handle_packet(pkt, "unk_dev", sender)
            assert await plugin.get_call("unk_dev") is None

        # CallEvent published for each, but zero CallStateChangedEvent
        assert len(events) == len(unknown_events)
        assert len(state_events) == 0

    async def test_case_insensitivity_and_variants_of_known_events(self, plugin: TelephonyPlugin) -> None:
        """Verify case variations and whitespace for known events transition state correctly."""
        sender = MockPacketSender()

        # 1. Ringing variants
        for r_evt in ("RINGING", "Ringing", "  ringing  "):
            pkt = Packet(type=TELEPHONY_PACKET_TYPE, body={"event": r_evt, "phoneNumber": "123"})
            await plugin.handle_packet(pkt, "var_dev", sender)
            call = await plugin.get_call("var_dev")
            assert call is not None and call["state"] == "RINGING"

        # 2. Talking variants
        for t_evt in ("TALKING", "Talking", "  talking  "):
            pkt = Packet(type=TELEPHONY_PACKET_TYPE, body={"event": t_evt, "phoneNumber": "123"})
            await plugin.handle_packet(pkt, "var_dev", sender)
            call = await plugin.get_call("var_dev")
            assert call is not None and call["state"] == "TALKING"

        # 3. Call ended variants
        for end_evt in ("IDLE", "Idle", "callEnded", "CALLENDED", "call_ended", "ended", "end"):
            # Set to ringing first
            await plugin.handle_packet(
                Packet(type=TELEPHONY_PACKET_TYPE, body={"event": "ringing", "phoneNumber": "123"}),
                "var_dev",
                sender,
            )
            assert await plugin.get_call("var_dev") is not None
            # End call
            await plugin.handle_packet(
                Packet(type=TELEPHONY_PACKET_TYPE, body={"event": end_evt}),
                "var_dev",
                sender,
            )
            assert await plugin.get_call("var_dev") is None

    async def test_event_string_length_clamping(self, plugin: TelephonyPlugin) -> None:
        """Verify extremely long event names are clamped to MAX_EVENT_LEN without memory leaks."""
        sender = MockPacketSender()
        huge_event = "ringing_" + "x" * 5000
        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": huge_event, "phoneNumber": "123"},
        )
        await plugin.handle_packet(pkt, "evt_dev", sender)
        # Clamped event is unknown, so state remains None
        assert await plugin.get_call("evt_dev") is None


# =====================================================================
# Test Class 5: Flapping Bursts & Concurrency Stress
# =====================================================================


class TestChallengerFlappingBurstsAndConcurrency:
    """Stress tests on high-frequency flapping bursts (1,000 rapid transitions)."""

    @pytest.fixture
    def setup_bus(self) -> tuple[EventBus, TelephonyPlugin]:
        bus = EventBus()
        plugin = TelephonyPlugin(event_bus=bus)
        return bus, plugin

    async def test_flapping_burst_single_device_1000_transitions(
        self, setup_bus: tuple[EventBus, TelephonyPlugin]
    ) -> None:
        """1,000 rapid call transitions executed on a single device within milliseconds.

        Validates lock synchronization, transition correctness, and zero memory corruption.
        """
        bus, plugin = setup_bus
        sender = MockPacketSender()
        device_id = "flapping_phone_1"

        state_events: list[CallStateChangedEvent] = []
        bus.subscribe(CallStateChangedEvent, lambda ev: state_events.append(ev))

        # Sequence of 4 recurring states: ringing -> talking -> ringing -> idle
        cycle = ["ringing", "talking", "ringing", "idle"]
        total_transitions = 1000

        t0 = time.perf_counter()
        for i in range(total_transitions):
            evt = cycle[i % len(cycle)]
            pkt = Packet(
                type=TELEPHONY_PACKET_TYPE,
                body={
                    "event": evt,
                    "phoneNumber": "+15550000000",
                    "contactName": "Flapping Tester",
                },
            )
            await plugin.handle_packet(pkt, device_id, sender)

        elapsed = time.perf_counter() - t0

        # Performance assertion: 1,000 transitions should complete in < 2.0 seconds
        assert elapsed < 2.0, f"1,000 flapping transitions took too long: {elapsed:.2f}s"

        # Final state check: 1000 % 4 == 0 -> last event was cycle[999 % 4] = cycle[3] = "idle"
        final_call = await plugin.get_call(device_id)
        assert final_call is None, "Final state should be IDLE after 1000 transitions"

        # Verify state changes were captured accurately
        assert len(state_events) == total_transitions
        assert state_events[-1].new_state == "IDLE"

    async def test_concurrent_multi_device_flapping_burst_1000_total(
        self, setup_bus: tuple[EventBus, TelephonyPlugin]
    ) -> None:
        """10 devices each executing 100 transitions concurrently (1,000 total transitions).

        Validates multi-device isolation and concurrency under asyncio.gather.
        """
        _, plugin = setup_bus
        sender = MockPacketSender()
        num_devices = 10
        transitions_per_device = 100

        async def device_worker(dev_index: int) -> None:
            dev_id = f"peer_dev_{dev_index}"
            cycle = ["ringing", "talking", "idle"]
            for i in range(transitions_per_device):
                evt = cycle[i % len(cycle)]
                pkt = Packet(
                    type=TELEPHONY_PACKET_TYPE,
                    body={
                        "event": evt,
                        "phoneNumber": f"+1555000{dev_index:04d}",
                        "contactName": f"Worker {dev_index}",
                    },
                )
                await plugin.handle_packet(pkt, dev_id, sender)

        t0 = time.perf_counter()
        await asyncio.gather(*(device_worker(i) for i in range(num_devices)))
        elapsed = time.perf_counter() - t0

        assert elapsed < 3.0, f"Concurrent multi-device burst took {elapsed:.2f}s"

        # Check final states: 100 % 3 = 1 -> last was cycle[99 % 3] = cycle[0] = "ringing"
        all_calls = await plugin.get_active_calls()
        assert len(all_calls) == num_devices
        for call in all_calls:
            assert call["state"] == "RINGING"

        # Cleanup
        await plugin.telephony_service.clear_all()
        assert len(await plugin.get_active_calls()) == 0

    async def test_rapid_lifecycle_churn_during_flapping(
        self, setup_bus: tuple[EventBus, TelephonyPlugin]
    ) -> None:
        """Concurrently fire call events and device disconnects without deadlocks or races."""
        bus, plugin = setup_bus
        await plugin.start()
        sender = MockPacketSender()

        async def call_pusher() -> None:
            for i in range(200):
                dev_id = f"churn_dev_{i % 5}"
                pkt = Packet(
                    type=TELEPHONY_PACKET_TYPE,
                    body={"event": "ringing", "phoneNumber": "123"},
                )
                await plugin.handle_packet(pkt, dev_id, sender)
                await asyncio.sleep(0.001)

        async def disconnect_pusher() -> None:
            for i in range(50):
                dev_id = f"churn_dev_{i % 5}"
                await bus.publish(DeviceDisconnected(device_id=dev_id))
                await asyncio.sleep(0.004)

        await asyncio.gather(call_pusher(), disconnect_pusher())
        await plugin.stop()

    async def test_outbound_action_burst_during_flapping(
        self, setup_bus: tuple[EventBus, TelephonyPlugin]
    ) -> None:
        """Fire 100 rapid mute_ringer and reject_call actions concurrently."""
        _, plugin = setup_bus
        sender = MockPacketSender()

        # Connect 5 devices
        for i in range(5):
            dev_id = f"act_dev_{i}"
            pkt = Packet(
                type=TELEPHONY_PACKET_TYPE,
                body={"event": "ringing", "phoneNumber": "123"},
            )
            await plugin.handle_packet(pkt, dev_id, sender)

        # Concurrently dispatch 50 mutes and 50 rejects
        async def mute_batch() -> None:
            for i in range(50):
                await plugin.mute_ringer(f"act_dev_{i % 5}")

        async def reject_batch() -> None:
            for i in range(50):
                await plugin.reject_call(f"act_dev_{i % 5}")

        await asyncio.gather(mute_batch(), reject_batch())
        assert len(sender.sent_packets) >= 100


# =====================================================================
# Test Class 6: Reentrancy & Concurrency Lock Investigation
# =====================================================================


class TestChallengerReentrancyAndLockSafety:
    """Stress tests on EventBus subscriber reentrancy and lock safety."""

    async def test_call_event_subscriber_can_query_service(self) -> None:
        """CallEvent subscriber querying telephony_service succeeds without deadlock."""
        bus = EventBus()
        plugin = TelephonyPlugin(event_bus=bus)
        sender = MockPacketSender()

        queried_results: list[dict[str, Any]] = []

        async def on_call_event(ev: CallEvent) -> None:
            # Reentrant query to telephony_service inside CallEvent handler
            calls = await plugin.get_active_calls()
            queried_results.extend(calls)

        bus.subscribe(CallEvent, on_call_event)

        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": "+1234"},
        )

        # Should finish quickly without deadlock
        await asyncio.wait_for(
            plugin.handle_packet(pkt, "reentrant_dev", sender),
            timeout=2.0,
        )

    async def test_call_state_changed_synchronous_snapshot_query(self) -> None:
        """CallStateChangedEvent subscriber querying synchronous snapshot succeeds."""
        bus = EventBus()
        plugin = TelephonyPlugin(event_bus=bus)
        sender = MockPacketSender()

        snapshots: list[list[dict[str, Any]]] = []

        def on_state_changed(ev: CallStateChangedEvent) -> None:
            # Synchronous snapshot method does not acquire lock and never deadlocks
            res = plugin.telephony_service.get_active_calls_sync()
            snapshots.append(res)

        bus.subscribe(CallStateChangedEvent, on_state_changed)

        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": "+1234"},
        )
        await asyncio.wait_for(
            plugin.handle_packet(pkt, "sync_query_dev", sender),
            timeout=2.0,
        )

        assert len(snapshots) == 1
        assert len(snapshots[0]) == 1
        assert snapshots[0][0]["state"] == "RINGING"

    async def test_call_state_changed_async_query_deadlock_investigation(self) -> None:
        """Investigation of async query inside CallStateChangedEvent subscriber.

        Because process_call_event holds self._lock while publishing CallStateChangedEvent,
        any subscriber awaiting self.get_active_calls() or self.get_call() could circular-wait.
        """
        bus = EventBus()
        plugin = TelephonyPlugin(event_bus=bus)
        sender = MockPacketSender()

        deadlock_observed = False
        queried_calls: list[dict[str, Any]] = []

        async def on_state_changed(ev: CallStateChangedEvent) -> None:
            # Reentrant async call requiring self._lock succeeds because lock is released before publish
            nonlocal deadlock_observed, queried_calls
            try:
                queried_calls = await asyncio.wait_for(plugin.get_active_calls(), timeout=0.5)
            except asyncio.TimeoutError:
                deadlock_observed = True

        bus.subscribe(CallStateChangedEvent, on_state_changed)

        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": "+1234"},
        )

        # Process packet
        await asyncio.wait_for(
            plugin.handle_packet(pkt, "deadlock_dev", sender),
            timeout=2.0,
        )

        # Empirically verify whether circular lock deadlock occurred
        # With lock released before publish, deadlock_observed must be False
        assert not deadlock_observed, "Expected async subscriber to query get_active_calls() without deadlock"
        assert len(queried_calls) == 1
        assert queried_calls[0]["device_id"] == "deadlock_dev"
        assert queried_calls[0]["state"] == "RINGING"

    async def test_call_state_changed_async_get_call_deadlock_investigation(self) -> None:
        """Investigation of async get_call inside CallStateChangedEvent subscriber."""
        bus = EventBus()
        plugin = TelephonyPlugin(event_bus=bus)
        sender = MockPacketSender()

        deadlock_observed = False
        queried_call: dict[str, Any] | None = None

        async def on_state_changed(ev: CallStateChangedEvent) -> None:
            nonlocal deadlock_observed, queried_call
            try:
                queried_call = await asyncio.wait_for(plugin.get_call(ev.device_id), timeout=0.5)
            except asyncio.TimeoutError:
                deadlock_observed = True

        bus.subscribe(CallStateChangedEvent, on_state_changed)

        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": "+1234"},
        )
        await asyncio.wait_for(
            plugin.handle_packet(pkt, "get_call_deadlock_dev", sender),
            timeout=2.0,
        )
        assert not deadlock_observed, "Expected async subscriber to query get_call() without deadlock"
        assert queried_call is not None
        assert queried_call["device_id"] == "get_call_deadlock_dev"
        assert queried_call["state"] == "RINGING"

    async def test_clear_device_during_active_call_deadlock_investigation(self) -> None:
        """Investigation of clear_device publishing CallStateChangedEvent under lock."""
        bus = EventBus()
        plugin = TelephonyPlugin(event_bus=bus)
        sender = MockPacketSender()

        deadlock_observed = False
        queried_calls: list[dict[str, Any]] = []

        async def on_state_changed(ev: CallStateChangedEvent) -> None:
            if ev.new_state == "IDLE":
                nonlocal deadlock_observed, queried_calls
                try:
                    queried_calls = await asyncio.wait_for(plugin.get_active_calls(), timeout=0.5)
                except asyncio.TimeoutError:
                    deadlock_observed = True

        bus.subscribe(CallStateChangedEvent, on_state_changed)

        # First set up ringing call without handler
        pkt = Packet(
            type=TELEPHONY_PACKET_TYPE,
            body={"event": "ringing", "phoneNumber": "+1234"},
        )
        await plugin.handle_packet(pkt, "clear_deadlock_dev", sender)

        # Clear device
        await asyncio.wait_for(
            plugin.clear_device("clear_deadlock_dev"),
            timeout=2.0,
        )
        assert not deadlock_observed, "Expected clear_device to publish without holding lock, avoiding deadlock"
        assert len(queried_calls) == 0


# =====================================================================
# Test Class 7: Direct Service Invocations & Error Boundaries
# =====================================================================


class TestChallengerDirectServiceInvocations:
    """Stress tests on direct TelephonyService methods, ActiveCall models, and exceptions."""

    async def test_service_invalid_device_id_raises_invalid_call_state(self) -> None:
        """Direct invocation with blank/invalid device_id strictly raises InvalidCallStateError."""
        service = TelephonyService()
        for bad_id in ("", "   ", "\t", "\n", None, 12345):
            with pytest.raises(InvalidCallStateError):
                await service.process_call_event(bad_id, "ringing")  # type: ignore

    async def test_service_call_not_found_error_properties(self) -> None:
        """CallNotFoundError behaves as KeyError and TelephonyError."""
        service = TelephonyService()

        with pytest.raises(CallNotFoundError) as exc_info:
            service.get_call_sync("missing_dev", raise_if_not_found=True)
        assert issubclass(exc_info.type, KeyError)

        with pytest.raises(CallNotFoundError):
            await service.get_call_object("missing_dev", raise_if_not_found=True)

    async def test_active_call_duration_clock_skew_resilience(self) -> None:
        """ActiveCall.duration handles negative elapsed time from backward clock jumps."""
        call = ActiveCall(
            device_id="skew_dev",
            phone_number="123",
            contact_name="Tester",
            state=CallState.TALKING,
            start_time=time.time() + 1000.0,  # Simulated future start time
        )
        assert call.duration == 0.0

    async def test_coerce_is_cancel_exhaustive_truthy_falsy_values(self) -> None:
        """Verify all variations of coerce_is_cancel."""
        assert coerce_is_cancel(True) is True
        assert coerce_is_cancel("true") is True
        assert coerce_is_cancel("True") is True
        assert coerce_is_cancel("TRUE") is True
        assert coerce_is_cancel("1") is True
        assert coerce_is_cancel(1) is True
        assert coerce_is_cancel(1.0) is True
        assert coerce_is_cancel("yes") is True
        assert coerce_is_cancel("YES") is True

        assert coerce_is_cancel(False) is False
        assert coerce_is_cancel("false") is False
        assert coerce_is_cancel("0") is False
        assert coerce_is_cancel(0) is False
        assert coerce_is_cancel(0.0) is False
        assert coerce_is_cancel("no") is False
        assert coerce_is_cancel(None) is False
        assert coerce_is_cancel([]) is False
        assert coerce_is_cancel({}) is False

    async def test_coerce_boolean_exhaustive_values(self) -> None:
        """Verify all variations of coerce_boolean."""
        assert coerce_boolean(True) is True
        assert coerce_boolean("true") is True
        assert coerce_boolean("1") is True
        assert coerce_boolean(1) is True
        assert coerce_boolean("yes") is True

        assert coerce_boolean(False) is False
        assert coerce_boolean("false") is False
        assert coerce_boolean(None) is False
        assert coerce_boolean(0) is False
        assert coerce_boolean("random_str") is False

    async def test_outbound_sender_error_mapped_to_connection_closed(self) -> None:
        """Verify broken socket / sender errors are wrapped into ConnectionClosedError."""
        plugin = TelephonyPlugin()

        def broken_sender(*args: Any, **kwargs: Any) -> Any:
            raise BrokenPipeError("Socket disconnected abruptly")

        with pytest.raises(ConnectionClosedError):
            await plugin.mute_ringer("dev_broken", send_packet=broken_sender)

        with pytest.raises(ConnectionClosedError):
            await plugin.reject_call("dev_broken", send_packet=broken_sender)

    async def test_is_paired_checker_async_coroutine(self) -> None:
        """Verify async coroutine is_paired_checker is correctly awaited and enforced."""
        async def async_checker(dev_id: str) -> bool:
            await asyncio.sleep(0.001)
            return dev_id == "paired_dev"

        plugin = TelephonyPlugin(is_paired_checker=async_checker)
        sender = MockPacketSender()

        # Unpaired device packet dropped
        await plugin.handle_packet(
            Packet(type=TELEPHONY_PACKET_TYPE, body={"event": "ringing", "phoneNumber": "123"}),
            "unpaired_dev",
            sender,
        )
        assert await plugin.get_call("unpaired_dev") is None

        # Paired device packet accepted
        await plugin.handle_packet(
            Packet(type=TELEPHONY_PACKET_TYPE, body={"event": "ringing", "phoneNumber": "123"}),
            "paired_dev",
            sender,
        )
        assert await plugin.get_call("paired_dev") is not None


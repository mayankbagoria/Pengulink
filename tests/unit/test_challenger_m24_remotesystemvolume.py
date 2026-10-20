"""tests.unit.test_challenger_m24_remotesystemvolume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical adversarial challenger stress-test and fuzzing suite for:
`pengulink/plugins/remotesystemvolume.py`

Adversarial Vectors:
1. Malformed and adversarial sink inputs:
   - Negative volume, floating point volume (50.5), string volume ("75%"),
     infinity ("inf", "+inf", "-inf"), NaN ("NaN", "nan"), huge numbers (1e1000, 10**100).
   - Missing fields, null names, non-string names, huge descriptions (100KB), emojis.
   - Custom maxVolume: zero, negative, huge (1000000), non-integer, NaN, inf.
2. Inbound packet fuzzing:
   - Non-dict sinkList (e.g. sinkList: "not_a_list", sinkList: None, sinkList: 123).
   - Sinks containing nested garbage or non-dict items.
   - Missing or empty names, corrupted payloads, duplicate sink names.
   - Single sink mutation fuzzing and edge cases.
   - Pairing checker enforcement and exception handling.
3. Outbound request boundaries:
   - Requesting volume for non-existent sink; negative volume; volume exceeding max.
   - Requesting volume with float values.
   - Routing verification (explicit sender, cached peer, packet_sender fallback, PenguLinkError).
4. Multi-device cache isolation & lifecycle:
   - Strict device isolation under concurrent updates.
   - Disconnection (retains cache, clears sender) vs Unpairing (purges cache).
   - EventBus typed event verification.
5. Concurrency & Stress Harness:
   - Rapid concurrent burst of 100 malformed/valid packets.
   - Concurrent reader queries during high-frequency cache mutations.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from pengulink.core.events import (
    DeviceDisconnected,
    DeviceUnpaired,
    DeviceUnpairedEvent,
    EventBus,
    RemoteAudioSinkListEvent,
    RemoteAudioSinkUpdatedEvent,
)
from pengulink.core.exceptions import PenguLinkError
from pengulink.network.packet import Packet
from pengulink.plugins.remotesystemvolume import (
    DEFAULT_MAX_VOLUME,
    RemoteAudioSink,
    RemoteSystemVolumePlugin,
    _safe_int,
)


def _make_packet(body: Any, p_type: str = "kdeconnect.systemvolume") -> Packet:
    return Packet(type=p_type, body=body)


# =============================================================================
# 1. Adversarial Fuzzing for _safe_int
# =============================================================================


class TestAdversarialSafeInt:
    """Stress-test _safe_int with invalid types, infinities, NaNs, and boundary values."""

    @pytest.mark.parametrize(
        ("val", "default", "expected"),
        [
            # Basic numbers
            (0, 99, 0),
            (42, 0, 42),
            (-42, 0, -42),
            # Floats
            (42.0, 0, 42),
            (42.9, 0, 42),
            (-42.9, 0, -42),
            (0.0, 99, 0),
            (-0.0, 99, 0),
            # String numbers
            ("123", 0, 123),
            ("-123", 0, -123),
            ("  456  \n", 0, 456),
            ("50.5", 0, 50),
            ("-50.5", 0, -50),
            ("0", 99, 0),
            # Non-numeric strings
            ("75%", 10, 10),
            ("$100", 0, 0),
            ("not_a_number", 7, 7),
            ("", 5, 5),
            ("   \t\n  ", 3, 3),
            ("0x10", 0, 0),
            ("0b101", 0, 0),
            ("\x00", 0, 0),
            # Infinities (strings and float constants)
            ("inf", 0, 0),
            ("+inf", 0, 0),
            ("-inf", 0, 0),
            ("infinity", 0, 0),
            ("-infinity", 0, 0),
            ("INFINITY", 0, 0),
            ("-INFINITY", 0, 0),
            (float("inf"), 0, 0),
            (float("-inf"), 0, 0),
            # NaNs (strings and float constants)
            ("nan", 0, 0),
            ("NaN", 0, 0),
            ("NAN", 0, 0),
            ("-nan", 0, 0),
            (float("nan"), 0, 0),
            # Huge overflow numbers
            ("1e1000", 0, 0),
            ("-1e1000", 0, 0),
            (1e1000, 0, 0),  # In Python, float 1e1000 is inf
            (-1e1000, 0, 0),  # In Python, float -1e1000 is -inf
            # Large integers (safe in Python)
            (10**100, 0, 10**100),
            # Booleans (must NOT evaluate to 1 or 0)
            (True, 42, 42),
            (False, 42, 42),
            # None and collections
            (None, 15, 15),
            ([], 20, 20),
            ([1, 2], 20, 20),
            ({}, 25, 25),
            ({"a": 1}, 25, 25),
            (set(), 30, 30),
            ((), 35, 35),
            (object(), 40, 40),
        ],
    )
    def test_safe_int_fuzz_cases(self, val: Any, default: int, expected: int) -> None:
        assert _safe_int(val, default=default) == expected

    def test_safe_int_with_exception_raising_objects(self) -> None:
        class ToxicFloat:
            def __float__(self) -> float:
                raise ValueError("Explosive float conversion")

        class ToxicInt:
            def __int__(self) -> int:
                raise TypeError("Explosive int conversion")

        class ToxicOverflow:
            def __float__(self) -> float:
                raise OverflowError("Explosive overflow")

        assert _safe_int(ToxicFloat(), default=77) == 77
        assert _safe_int(ToxicInt(), default=88) == 88
        assert _safe_int(ToxicOverflow(), default=99) == 99


# =============================================================================
# 2. Adversarial Fuzzing for RemoteAudioSink & from_dict
# =============================================================================


class TestRemoteAudioSinkAdversarialFuzzing:
    """Stress-test RemoteAudioSink dataclass and from_dict parser with corrupted data."""

    def test_from_dict_non_dict_rejection(self) -> None:
        for bad_input in [
            None,
            "string_payload",
            12345,
            [{"name": "sink"}],
            True,
            False,
            (1, 2),
            set(),
        ]:
            with pytest.raises(TypeError, match="Expected dict"):
                RemoteAudioSink.from_dict(bad_input)  # type: ignore[arg-type]

    def test_from_dict_negative_volume_clamping(self) -> None:
        sink1 = RemoteAudioSink.from_dict({"name": "sink1", "volume": -1})
        assert sink1.volume == 0
        assert sink1.volume_percent == 0

        sink2 = RemoteAudioSink.from_dict({"name": "sink2", "volume": -999999})
        assert sink2.volume == 0

        sink3 = RemoteAudioSink.from_dict({"name": "sink3", "volume": "-500"})
        assert sink3.volume == 0

    def test_from_dict_floating_point_volume(self) -> None:
        sink_float = RemoteAudioSink.from_dict({"name": "sink_flt", "volume": 50.5})
        assert sink_float.volume == 50

        sink_str_float = RemoteAudioSink.from_dict(
            {"name": "sink_sflt", "volume": "75.8"}
        )
        assert sink_str_float.volume == 75

        sink_neg_float = RemoteAudioSink.from_dict(
            {"name": "sink_nflt", "volume": -10.9}
        )
        assert sink_neg_float.volume == 0

    def test_from_dict_non_numeric_string_volume(self) -> None:
        for bad_vol in ["75%", "max", "null", "undefined", "$50", "abc"]:
            sink = RemoteAudioSink.from_dict({"name": "sink", "volume": bad_vol})
            assert sink.volume == 0

    def test_from_dict_infinity_and_nan_volume(self) -> None:
        for inf_val in [
            "inf",
            "-inf",
            "+inf",
            "infinity",
            "-infinity",
            float("inf"),
            float("-inf"),
        ]:
            sink = RemoteAudioSink.from_dict({"name": "sink_inf", "volume": inf_val})
            assert sink.volume == 0

        for nan_val in ["nan", "NaN", "NAN", "-nan", float("nan")]:
            sink = RemoteAudioSink.from_dict({"name": "sink_nan", "volume": nan_val})
            assert sink.volume == 0

    def test_from_dict_huge_numbers(self) -> None:
        # String 1e1000 coerces via safe_int to 0 (overflows float to inf)
        sink_huge_str = RemoteAudioSink.from_dict(
            {"name": "sink_huge", "volume": "1e1000"}
        )
        assert sink_huge_str.volume == 0

        # Float 1e1000 is inf in Python, coerces to 0
        sink_huge_flt = RemoteAudioSink.from_dict(
            {"name": "sink_huge2", "volume": 1e1000}
        )
        assert sink_huge_flt.volume == 0

        # Enormous integer clamped to max_volume
        sink_huge_int = RemoteAudioSink.from_dict(
            {"name": "sink_huge3", "volume": 10**100, "maxVolume": 65536}
        )
        assert sink_huge_int.volume == 65536
        assert sink_huge_int.volume_percent == 100

    def test_from_dict_custom_max_volume_boundaries(self) -> None:
        # Zero maxVolume defaults to 65536
        sink_zero_max = RemoteAudioSink.from_dict(
            {"name": "sink_zm", "volume": 1000, "maxVolume": 0}
        )
        assert sink_zero_max.max_volume == DEFAULT_MAX_VOLUME
        assert sink_zero_max.volume == 1000

        # Negative maxVolume defaults to 65536
        sink_neg_max = RemoteAudioSink.from_dict(
            {"name": "sink_nm", "volume": 1000, "maxVolume": -50}
        )
        assert sink_neg_max.max_volume == DEFAULT_MAX_VOLUME

        # Huge maxVolume (1,000,000) accepted
        sink_huge_max = RemoteAudioSink.from_dict(
            {"name": "sink_hm", "volume": 500000, "maxVolume": 1000000}
        )
        assert sink_huge_max.max_volume == 1000000
        assert sink_huge_max.volume == 500000
        assert sink_huge_max.volume_percent == 50

        # Non-integer / invalid string maxVolume defaults to 65536
        for bad_mv in [
            "hundred",
            None,
            [],
            {},
            float("nan"),
            float("inf"),
            "nan",
            "inf",
        ]:
            sink_bad_mv = RemoteAudioSink.from_dict(
                {"name": "sink_bm", "volume": 500, "maxVolume": bad_mv}
            )
            assert sink_bad_mv.max_volume == DEFAULT_MAX_VOLUME

        # Float maxVolume string coerces integer part
        sink_flt_mv = RemoteAudioSink.from_dict(
            {"name": "sink_fm", "volume": 50, "maxVolume": "100.5"}
        )
        assert sink_flt_mv.max_volume == 100
        assert sink_flt_mv.volume == 50
        assert sink_flt_mv.volume_percent == 50

    def test_from_dict_missing_fields_and_null_names(self) -> None:
        # Empty dictionary
        sink_empty = RemoteAudioSink.from_dict({})
        assert sink_empty.name == ""
        assert sink_empty.description == "Remote Audio Output"
        assert sink_empty.volume == 0
        assert sink_empty.max_volume == DEFAULT_MAX_VOLUME
        assert sink_empty.muted is False

        # Null name
        sink_null_name = RemoteAudioSink.from_dict({"name": None})
        assert sink_null_name.name == ""
        assert sink_null_name.description == "Remote Audio Output"

        # Non-string name (int)
        sink_int_name = RemoteAudioSink.from_dict({"name": 12345})
        assert sink_int_name.name == "12345"
        assert sink_int_name.description == "12345"

    def test_from_dict_huge_description_and_unicode(self) -> None:
        huge_desc = "A" * 100_000  # 100 KB description
        sink_huge_desc = RemoteAudioSink.from_dict(
            {"name": "speaker", "description": huge_desc}
        )
        assert sink_huge_desc.description == huge_desc
        assert len(sink_huge_desc.description) == 100_000

        # Unicode and emojis
        emoji_desc = "🔊 Phone Speaker 日本語 🚀"
        sink_unicode = RemoteAudioSink.from_dict(
            {"name": "speaker_jp", "description": emoji_desc}
        )
        assert sink_unicode.description == emoji_desc

    def test_from_dict_description_alias_precedence(self) -> None:
        # description > desc > nick
        sink1 = RemoteAudioSink.from_dict(
            {"name": "s", "description": "primary", "desc": "alias1", "nick": "alias2"}
        )
        assert sink1.description == "primary"

        sink2 = RemoteAudioSink.from_dict(
            {"name": "s", "desc": "alias1", "nick": "alias2"}
        )
        assert sink2.description == "alias1"

        sink3 = RemoteAudioSink.from_dict({"name": "s", "nick": "alias2"})
        assert sink3.description == "alias2"

    def test_volume_percent_clamping_and_direct_mutation_resilience(self) -> None:
        sink = RemoteAudioSink(name="s", volume=50, max_volume=100)
        assert sink.volume_percent == 50

        # Direct mutation out of bounds
        sink.volume = 200
        assert sink.volume_percent == 100

        sink.volume = -50
        assert sink.volume_percent == 0

        sink.max_volume = 0
        assert sink.volume_percent == 0

        sink.max_volume = -10
        assert sink.volume_percent == 0

    def test_to_wire_dict_and_to_dict_schemas(self) -> None:
        sink = RemoteAudioSink(
            name="test_sink",
            description="Test Sink",
            volume=32768,
            max_volume=65536,
            muted=True,
            device_id="dev_xyz",
        )
        wire = sink.to_wire_dict()
        assert wire == {
            "name": "test_sink",
            "description": "Test Sink",
            "volume": 32768,
            "maxVolume": 65536,
            "muted": True,
        }

        d = sink.to_dict()
        assert d["name"] == "test_sink"
        assert d["description"] == "Test Sink"
        assert d["volume"] == 32768
        assert d["max_volume"] == 65536
        assert d["maxVolume"] == 65536
        assert d["muted"] is True
        assert d["device_id"] == "dev_xyz"
        assert d["volume_percent"] == 50


# =============================================================================
# 3. Inbound Packet Fuzzing
# =============================================================================


class TestInboundPacketFuzzing:
    """Fuzz RemoteSystemVolumePlugin.handle_packet with toxic, malformed, and adversarial payloads."""

    async def test_non_dict_and_none_packet_bodies_safely_ignored(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        for toxic_body in [
            None,
            "string_payload",
            12345,
            [],
            [{"name": "sink"}],
            True,
            False,
        ]:
            mock_pkt = MagicMock()
            mock_pkt.type = "kdeconnect.systemvolume"
            mock_pkt.body = toxic_body
            # Must return cleanly without raising
            await plugin.handle_packet(
                mock_pkt, device_id="dev_fuzz", send_packet=MagicMock()
            )
            assert len(plugin.get_sinks("dev_fuzz")) == 0

    async def test_non_system_volume_packet_type_ignored(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        pkt = _make_packet(
            {"sinkList": [{"name": "speaker", "volume": 1000}]},
            p_type="kdeconnect.ping",
        )
        await plugin.handle_packet(pkt, device_id="dev_fuzz", send_packet=MagicMock())
        assert len(plugin.get_sinks("dev_fuzz")) == 0

    async def test_sink_list_not_a_list_fuzzing(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        for bad_sinklist in [
            "not_a_list",
            None,
            12345,
            {"name": "speaker"},  # Dict instead of list
            True,
            False,
            3.14,
        ]:
            pkt = _make_packet({"sinkList": bad_sinklist})
            await plugin.handle_packet(
                pkt, device_id="dev_bad_sl", send_packet=MagicMock()
            )
            assert len(plugin.get_sinks("dev_bad_sl")) == 0

    async def test_sink_list_containing_nested_garbage_and_non_dict_items(
        self,
    ) -> None:
        plugin = RemoteSystemVolumePlugin()
        garbage_sink_list = [
            # 1. Non-dict primitives
            None,
            "just_a_string",
            12345,
            [],
            True,
            False,
            # 2. Dicts without name or empty/whitespace name
            {},
            {"volume": 10000},
            {"name": None, "volume": 10000},
            {"name": "", "volume": 10000},
            {"name": "   \t\n  ", "volume": 10000},
            # 3. Dicts with non-string non-int names
            {"name": [], "volume": 10000},
            {"name": {}, "volume": 10000},
            # 4. Valid sink alongside garbage
            {"name": "valid_sink_1", "volume": 25000},
            # 5. Adversarial sink with toxic volume values
            {"name": "valid_sink_2", "volume": "75%", "maxVolume": -50},
            {"name": "valid_sink_3", "volume": "inf", "maxVolume": "NaN"},
            # 6. More garbage
            set(),
            {"name": "valid_sink_4", "description": "X" * 10_000, "volume": -50},
        ]

        pkt = _make_packet({"sinkList": garbage_sink_list})
        await plugin.handle_packet(
            pkt, device_id="dev_garbage", send_packet=MagicMock()
        )

        sinks = plugin.get_sinks("dev_garbage")
        sink_names = {s.name for s in sinks}
        assert sink_names == {
            "valid_sink_1",
            "valid_sink_2",
            "valid_sink_3",
            "valid_sink_4",
        }

        s2 = plugin.get_sink("dev_garbage", "valid_sink_2")
        assert s2 is not None
        assert s2.volume == 0
        assert s2.max_volume == DEFAULT_MAX_VOLUME

        s3 = plugin.get_sink("dev_garbage", "valid_sink_3")
        assert s3 is not None
        assert s3.volume == 0
        assert s3.max_volume == DEFAULT_MAX_VOLUME

        s4 = plugin.get_sink("dev_garbage", "valid_sink_4")
        assert s4 is not None
        assert s4.volume == 0
        assert len(s4.description) == 10_000

    async def test_sink_list_with_duplicate_sink_names_deduplication(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        duplicate_sinks = [
            {"name": "speaker", "volume": 10000, "muted": False},
            {"name": "speaker", "volume": 20000, "muted": True},
            {"name": "speaker", "volume": 30000, "muted": False},
        ]
        pkt = _make_packet({"sinkList": duplicate_sinks})
        await plugin.handle_packet(pkt, device_id="dev_dup", send_packet=MagicMock())

        sinks = plugin.get_sinks("dev_dup")
        assert len(sinks) == 1
        assert sinks[0].name == "speaker"
        assert sinks[0].volume == 30000
        assert sinks[0].muted is False

    async def test_single_sink_update_fuzzing(self) -> None:
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        # Seed initial sink
        await plugin.handle_packet(
            _make_packet(
                {
                    "sinkList": [
                        {
                            "name": "headset",
                            "description": "Initial Headset",
                            "volume": 20000,
                            "maxVolume": 65536,
                            "muted": False,
                        }
                    ]
                }
            ),
            device_id="dev_single",
            send_packet=MagicMock(),
        )

        # 1. Update with invalid/NaN volume
        await plugin.handle_packet(
            _make_packet({"name": "headset", "volume": "NaN"}),
            device_id="dev_single",
            send_packet=MagicMock(),
        )
        sink = plugin.get_sink("dev_single", "headset")
        assert sink is not None
        # Invalid volume falls back to existing volume (20000)
        assert sink.volume == 20000

        # 2. Update with negative volume
        await plugin.handle_packet(
            _make_packet({"name": "headset", "volume": -100}),
            device_id="dev_single",
            send_packet=MagicMock(),
        )
        sink = plugin.get_sink("dev_single", "headset")
        assert sink is not None
        assert sink.volume == 0

        # 3. Update with negative maxVolume -> ignored, keeps existing
        await plugin.handle_packet(
            _make_packet({"name": "headset", "maxVolume": -50}),
            device_id="dev_single",
            send_packet=MagicMock(),
        )
        sink = plugin.get_sink("dev_single", "headset")
        assert sink is not None
        assert sink.max_volume == 65536

        # 4. Update with float volume string
        await plugin.handle_packet(
            _make_packet({"name": "headset", "volume": "40000.7"}),
            device_id="dev_single",
            send_packet=MagicMock(),
        )
        sink = plugin.get_sink("dev_single", "headset")
        assert sink is not None
        assert sink.volume == 40000

        # 5. Update with non-boolean mute values
        await plugin.handle_packet(
            _make_packet({"name": "headset", "muted": 1}),
            device_id="dev_single",
            send_packet=MagicMock(),
        )
        assert plugin.get_sink("dev_single", "headset").muted is True

        await plugin.handle_packet(
            _make_packet({"name": "headset", "muted": 0}),
            device_id="dev_single",
            send_packet=MagicMock(),
        )
        assert plugin.get_sink("dev_single", "headset").muted is False

        # 6. Malformed single sink packet with missing name or empty name
        await plugin.handle_packet(
            _make_packet({"volume": 50000}),
            device_id="dev_single",
            send_packet=MagicMock(),
        )
        await plugin.handle_packet(
            _make_packet({"name": "", "volume": 50000}),
            device_id="dev_single",
            send_packet=MagicMock(),
        )
        await plugin.handle_packet(
            _make_packet({"name": "   ", "volume": 50000}),
            device_id="dev_single",
            send_packet=MagicMock(),
        )
        # Volume remains unchanged
        assert plugin.get_sink("dev_single", "headset").volume == 40000

        await plugin.stop()

    async def test_pairing_checker_dropping_and_exception_handling(self) -> None:
        calls: list[str] = []

        def flaky_checker(dev_id: str) -> bool:
            calls.append(dev_id)
            if dev_id == "crasher":
                raise ValueError("Database corrupt")
            if dev_id == "runtime_crasher":
                raise RuntimeError("Lock poisoned")
            if dev_id == "pengulink_crasher":
                raise PenguLinkError("Storage unavailable")
            return dev_id == "paired"

        plugin = RemoteSystemVolumePlugin(is_paired_checker=flaky_checker)

        # 1. Unpaired device -> dropped
        pkt = _make_packet({"sinkList": [{"name": "s1", "volume": 1000}]})
        await plugin.handle_packet(pkt, device_id="unpaired", send_packet=MagicMock())
        assert len(plugin.get_sinks("unpaired")) == 0

        # 2. Paired device -> accepted
        await plugin.handle_packet(pkt, device_id="paired", send_packet=MagicMock())
        assert len(plugin.get_sinks("paired")) == 1

        # 3. Crashers -> safely caught and dropped without exception
        for crasher_id in ["crasher", "runtime_crasher", "pengulink_crasher"]:
            await plugin.handle_packet(
                pkt, device_id=crasher_id, send_packet=MagicMock()
            )
            assert len(plugin.get_sinks(crasher_id)) == 0


# =============================================================================
# 4. Outbound Request Boundaries & Edge Cases
# =============================================================================


class TestOutboundRequestBoundaries:
    """Stress-test outbound control methods with boundary and unexpected parameters."""

    async def test_request_remote_sinks_packet_format(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        outbox: list[Packet] = []

        async def capture(pkt: Packet) -> None:
            outbox.append(pkt)

        await plugin.request_remote_sinks("phone_out", send_packet=capture)
        assert len(outbox) == 1
        assert outbox[0].type == "kdeconnect.systemvolume.request"
        assert outbox[0].body == {"requestSinks": True}

    async def test_set_remote_volume_boundaries(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        outbox: list[Packet] = []

        async def capture(pkt: Packet) -> None:
            outbox.append(pkt)

        # 1. Non-existent sink: max_volume defaults to 65536
        await plugin.set_remote_volume(
            "unknown_dev", "unknown_sink", 40000, send_packet=capture
        )
        assert outbox[-1].body == {"name": "unknown_sink", "volume": 40000}

        # 2. Non-existent sink with negative volume -> clamped to 0
        await plugin.set_remote_volume(
            "unknown_dev", "unknown_sink", -500, send_packet=capture
        )
        assert outbox[-1].body == {"name": "unknown_sink", "volume": 0}

        # 3. Non-existent sink with volume > 65536 -> clamped to 65536
        await plugin.set_remote_volume(
            "unknown_dev", "unknown_sink", 100000, send_packet=capture
        )
        assert outbox[-1].body == {"name": "unknown_sink", "volume": 65536}

        # 4. Known sink with custom maxVolume (e.g. 100)
        await plugin.handle_packet(
            _make_packet(
                {"sinkList": [{"name": "media_stream", "maxVolume": 100, "volume": 50}]}
            ),
            device_id="known_dev",
            send_packet=capture,
        )

        # Volume 150 -> clamped to 100
        await plugin.set_remote_volume(
            "known_dev", "media_stream", 150, send_packet=capture
        )
        assert outbox[-1].body == {"name": "media_stream", "volume": 100}

        # Volume -10 -> clamped to 0
        await plugin.set_remote_volume(
            "known_dev", "media_stream", -10, send_packet=capture
        )
        assert outbox[-1].body == {"name": "media_stream", "volume": 0}

        # Float volume 75.6 -> coerced to integer 75
        await plugin.set_remote_volume(
            "known_dev",
            "media_stream",
            75.6,
            send_packet=capture,  # type: ignore[arg-type]
        )
        assert outbox[-1].body == {"name": "media_stream", "volume": 75}

    async def test_set_remote_muted_and_default(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        outbox: list[Packet] = []

        async def capture(pkt: Packet) -> None:
            outbox.append(pkt)

        await plugin.set_remote_muted("dev", "sink", True, send_packet=capture)
        assert outbox[-1].body == {"name": "sink", "muted": True}

        await plugin.set_remote_muted("dev", "sink", False, send_packet=capture)
        assert outbox[-1].body == {"name": "sink", "muted": False}

        await plugin.set_remote_default("dev", "sink", send_packet=capture)
        assert outbox[-1].body == {"name": "sink", "enabled": True, "default": True}

    async def test_outbound_routing_resilience(self) -> None:
        plugin = RemoteSystemVolumePlugin()

        # No connection or sender -> raises PenguLinkError
        with pytest.raises(PenguLinkError, match="Cannot send packet"):
            await plugin.set_remote_volume("dev_offline", "speaker", 50)

        # Fallback to plugin.packet_sender if provided
        sent_via_global: list[Packet] = []

        async def global_sender(dev_id: str, pkt: Packet) -> None:
            sent_via_global.append(pkt)

        plugin_with_sender = RemoteSystemVolumePlugin(packet_sender=global_sender)
        await plugin_with_sender.set_remote_volume("dev_target", "speaker", 25000)
        assert len(sent_via_global) == 1
        assert sent_via_global[0].body == {"name": "speaker", "volume": 25000}


# =============================================================================
# 5. Multi-Device Isolation & Lifecycle Verification
# =============================================================================


class TestMultiDeviceIsolationAndLifecycle:
    """Verify clean device-scoped isolation, event emission, and unpair teardown."""

    async def test_multi_device_cache_isolation(self) -> None:
        plugin = RemoteSystemVolumePlugin()

        # Seed 3 separate devices
        for i in range(1, 4):
            dev_id = f"phone_{i}"
            pkt = _make_packet(
                {
                    "sinkList": [
                        {
                            "name": f"sink_{i}",
                            "volume": 10000 * i,
                            "maxVolume": 65536,
                        }
                    ]
                }
            )
            await plugin.handle_packet(pkt, device_id=dev_id, send_packet=MagicMock())

        assert len(plugin.get_sinks("phone_1")) == 1
        assert len(plugin.get_sinks("phone_2")) == 1
        assert len(plugin.get_sinks("phone_3")) == 1

        assert plugin.get_sink("phone_1", "sink_1").volume == 10000
        assert plugin.get_sink("phone_2", "sink_2").volume == 20000
        assert plugin.get_sink("phone_3", "sink_3").volume == 30000

        # Incremental mutation on phone_2 does not bleed into phone_1 or phone_3
        await plugin.handle_packet(
            _make_packet({"name": "sink_2", "volume": 55555}),
            device_id="phone_2",
            send_packet=MagicMock(),
        )
        assert plugin.get_sink("phone_1", "sink_1").volume == 10000
        assert plugin.get_sink("phone_2", "sink_2").volume == 55555
        assert plugin.get_sink("phone_3", "sink_3").volume == 30000

        # Evicting phone_2 leaves phone_1 and phone_3 intact
        plugin.evict_device("phone_2")
        assert len(plugin.get_sinks("phone_1")) == 1
        assert len(plugin.get_sinks("phone_2")) == 0
        assert len(plugin.get_sinks("phone_3")) == 1

    async def test_event_bus_publishing_accuracy(self) -> None:
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        list_events: list[RemoteAudioSinkListEvent] = []
        update_events: list[RemoteAudioSinkUpdatedEvent] = []

        bus.subscribe(RemoteAudioSinkListEvent, list_events.append)
        bus.subscribe(RemoteAudioSinkUpdatedEvent, update_events.append)

        # 1. List packet publishes RemoteAudioSinkListEvent
        await plugin.handle_packet(
            _make_packet(
                {
                    "sinkList": [
                        {"name": "spk", "volume": 30000},
                        {"name": "mic", "volume": 10000},
                    ]
                }
            ),
            device_id="dev_evt",
            send_packet=MagicMock(),
        )

        assert len(list_events) == 1
        assert list_events[0].device_id == "dev_evt"
        assert len(list_events[0].sinks) == 2

        # 2. Update packet publishes RemoteAudioSinkUpdatedEvent
        await plugin.handle_packet(
            _make_packet({"name": "spk", "volume": 35000}),
            device_id="dev_evt",
            send_packet=MagicMock(),
        )

        assert len(update_events) == 1
        assert update_events[0].device_id == "dev_evt"
        assert update_events[0].sink_name == "spk"
        assert update_events[0].volume == 35000

        await plugin.stop()

    async def test_lifecycle_event_unpair_and_disconnect(self) -> None:
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        # Seed device state
        await plugin.handle_packet(
            _make_packet({"sinkList": [{"name": "s", "volume": 1000}]}),
            device_id="dev_life",
            send_packet=MagicMock(),
        )
        assert len(plugin.get_sinks("dev_life")) == 1
        assert "dev_life" in plugin._connected_peers

        # Disconnection: peer sender removed, cache preserved
        await bus.publish(DeviceDisconnected(device_id="dev_life"))
        assert "dev_life" not in plugin._connected_peers
        assert len(plugin.get_sinks("dev_life")) == 1

        # Unpairing: cache purged
        await bus.publish(DeviceUnpaired(device_id="dev_life"))
        assert len(plugin.get_sinks("dev_life")) == 0

        # Unpair alias event (DeviceUnpairedEvent)
        await plugin.handle_packet(
            _make_packet({"sinkList": [{"name": "s", "volume": 1000}]}),
            device_id="dev_alias",
            send_packet=MagicMock(),
        )
        assert len(plugin.get_sinks("dev_alias")) == 1
        await bus.publish(DeviceUnpairedEvent(device_id="dev_alias"))
        assert len(plugin.get_sinks("dev_alias")) == 0

        await plugin.stop()


# =============================================================================
# 6. Concurrency & Stress Harness
# =============================================================================


class TestStressAndConcurrencyHarness:
    """Stress-test plugin under rapid concurrent packet bursts and queries."""

    async def test_rapid_burst_fuzz_packets(self) -> None:
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        mock_send = MagicMock()

        # Generate 100 diverse packets: valid, corrupt, empty, partial
        packets: list[tuple[str, Packet]] = []
        for i in range(100):
            dev_id = f"burst_phone_{i % 5}"  # 5 rotating devices
            if i % 5 == 0:
                # Full sink list
                pkt = _make_packet(
                    {
                        "sinkList": [
                            {"name": "speaker", "volume": i * 500, "maxVolume": 65536},
                            {
                                "name": "earpiece",
                                "volume": i * 200,
                                "muted": i % 2 == 0,
                            },
                        ]
                    }
                )
            elif i % 5 == 1:
                # Incremental volume update
                pkt = _make_packet({"name": "speaker", "volume": i * 600})
            elif i % 5 == 2:
                # Malformed non-dict / garbage entry
                pkt = _make_packet(
                    {"sinkList": [None, "garbage", {"name": "extra", "volume": "75%"}]}
                )
            elif i % 5 == 3:
                # Incremental mute update
                pkt = _make_packet({"name": "speaker", "muted": bool(i % 2)})
            else:
                # Empty sink list
                pkt = _make_packet({"sinkList": []})

            packets.append((dev_id, pkt))

        # Launch all 100 packets concurrently via asyncio.gather
        tasks = [
            plugin.handle_packet(pkt, device_id=dev, send_packet=mock_send)
            for dev, pkt in packets
        ]
        await asyncio.gather(*tasks)

        # Confirm all 5 devices can be queried without errors or deadlocks
        for i in range(5):
            dev_id = f"burst_phone_{i}"
            sinks = plugin.get_sinks(dev_id)
            assert isinstance(sinks, list)
            for s in sinks:
                assert isinstance(s, RemoteAudioSink)
                assert 0 <= s.volume <= s.max_volume
                assert 0 <= s.volume_percent <= 100

        await plugin.stop()

    async def test_concurrent_readers_during_mutations(self) -> None:
        """Verify reading get_sinks / get_sink while packets arrive produces no dictionary iteration races."""
        plugin = RemoteSystemVolumePlugin()
        dev_id = "concurrent_phone"

        # Worker 1: Rapid packet mutations
        async def writer() -> None:
            for i in range(50):
                pkt = _make_packet(
                    {
                        "sinkList": [
                            {"name": f"sink_{j}", "volume": (i + j) * 100}
                            for j in range(5)
                        ]
                    }
                )
                await plugin.handle_packet(
                    pkt, device_id=dev_id, send_packet=MagicMock()
                )
                await asyncio.sleep(0.001)

        # Worker 2: Rapid reader queries
        async def reader() -> None:
            for _ in range(100):
                sinks = plugin.get_sinks(dev_id)
                assert isinstance(sinks, list)
                for s in sinks:
                    assert plugin.get_sink(dev_id, s.name) is not None
                await asyncio.sleep(0.0005)

        await asyncio.gather(writer(), reader())
        assert len(plugin.get_sinks(dev_id)) == 5

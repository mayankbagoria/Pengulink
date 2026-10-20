"""tests.unit.test_plugin_remotesystemvolume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit test suite for KDE Connect RemoteSystemVolume client plugin:
- RemoteAudioSink dataclass parsing, desc aliasing, custom maxVolume, int coercion, volume clamping
- Plugin metadata, capability advertisements, and lifecycle hooks
- Full sinkList ingestion, event publishing, and query methods
- Incremental single sink volume/mute updates
- Empty sinkList cache clearing
- Malformed packets, corrupted entries, and missing fields resilience
- Device pairing checks and unauthenticated packet dropping
- Multi-device cache isolation
- Outbound requests (requestSinks, volume, muted, default)
- Cache eviction on device unpair and disconnect handling
"""

from __future__ import annotations

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


def _create_packet(
    body: dict | None = None,
    packet_type: str = "kdeconnect.systemvolume",
) -> Packet:
    return Packet(type=packet_type, body=body or {})


# =============================================================================
# 1. RemoteAudioSink Data Model Tests
# =============================================================================


class TestRemoteAudioSinkDataModel:
    """Tests for RemoteAudioSink dataclass methods, properties, and parsing."""

    def test_nominal_initialization(self) -> None:
        sink = RemoteAudioSink(
            name="speaker",
            description="Phone Speaker",
            volume=32768,
            max_volume=65536,
            muted=False,
            device_id="phone_1",
        )
        assert sink.name == "speaker"
        assert sink.description == "Phone Speaker"
        assert sink.volume == 32768
        assert sink.max_volume == 65536
        assert sink.muted is False
        assert sink.device_id == "phone_1"
        assert sink.volume_percent == 50

    def test_default_values_and_fallback_description(self) -> None:
        sink = RemoteAudioSink(name="headset")
        assert sink.name == "headset"
        assert sink.description == "headset"
        assert sink.volume == 0
        assert sink.max_volume == DEFAULT_MAX_VOLUME
        assert sink.muted is False
        assert sink.device_id == ""
        assert sink.volume_percent == 0

    def test_volume_clamping_to_max_volume(self) -> None:
        sink_over = RemoteAudioSink(name="sink1", volume=99999, max_volume=65536)
        assert sink_over.volume == 65536
        assert sink_over.volume_percent == 100

        sink_under = RemoteAudioSink(name="sink2", volume=-500, max_volume=65536)
        assert sink_under.volume == 0
        assert sink_under.volume_percent == 0

    def test_non_positive_max_volume_defaults_to_65536(self) -> None:
        sink_zero = RemoteAudioSink(name="sink_zero", volume=1000, max_volume=0)
        assert sink_zero.max_volume == DEFAULT_MAX_VOLUME
        assert sink_zero.volume == 1000

        sink_neg = RemoteAudioSink(name="sink_neg", volume=1000, max_volume=-100)
        assert sink_neg.max_volume == DEFAULT_MAX_VOLUME
        assert sink_neg.volume == 1000

    def test_custom_max_volume_scaling(self) -> None:
        sink = RemoteAudioSink(
            name="android_media",
            description="Android Media Stream",
            volume=75,
            max_volume=100,
            muted=False,
        )
        assert sink.max_volume == 100
        assert sink.volume == 75
        assert sink.volume_percent == 75

    def test_to_dict_and_to_wire_dict_schemas(self) -> None:
        sink = RemoteAudioSink(
            name="speaker",
            description="Speaker",
            volume=32768,
            max_volume=65536,
            muted=True,
            device_id="dev_xyz",
        )
        d = sink.to_dict()
        assert d["name"] == "speaker"
        assert d["description"] == "Speaker"
        assert d["volume"] == 32768
        assert d["max_volume"] == 65536
        assert d["maxVolume"] == 65536
        assert d["muted"] is True
        assert d["device_id"] == "dev_xyz"
        assert d["volume_percent"] == 50

        wire = sink.to_wire_dict()
        assert wire["name"] == "speaker"
        assert wire["description"] == "Speaker"
        assert wire["volume"] == 32768
        assert wire["maxVolume"] == 65536
        assert wire["muted"] is True

    def test_from_dict_nominal_and_aliases(self) -> None:
        # Standard KDE Connect wire format
        data_standard = {
            "name": "media",
            "description": "Media Volume",
            "volume": 40000,
            "maxVolume": 65536,
            "muted": False,
        }
        sink = RemoteAudioSink.from_dict(data_standard, device_id="phone_a")
        assert sink.name == "media"
        assert sink.description == "Media Volume"
        assert sink.volume == 40000
        assert sink.max_volume == 65536
        assert sink.muted is False
        assert sink.device_id == "phone_a"

        # GSConnect 'desc' alias
        data_desc = {
            "name": "ringtone",
            "desc": "Ringtone Volume",
            "volume": 20000,
            "max_volume": 65536,
            "muted": True,
        }
        sink_desc = RemoteAudioSink.from_dict(data_desc, device_id="phone_b")
        assert sink_desc.description == "Ringtone Volume"
        assert sink_desc.muted is True

        # 'nick' alias fallback
        data_nick = {
            "name": "alarm",
            "nick": "Alarm Clock",
            "volume": 50000,
        }
        sink_nick = RemoteAudioSink.from_dict(data_nick)
        assert sink_nick.description == "Alarm Clock"

    def test_safe_int_coercion_and_fallbacks(self) -> None:
        assert _safe_int(123) == 123
        assert _safe_int("456") == 456
        assert _safe_int(78.9) == 78
        assert _safe_int("78.9") == 78
        assert _safe_int(None, default=10) == 10
        assert _safe_int("not_a_number", default=5) == 5
        assert _safe_int("inf", default=0) == 0
        assert _safe_int("-inf", default=0) == 0
        assert _safe_int("nan", default=0) == 0
        assert _safe_int(float("inf"), default=0) == 0
        assert _safe_int(float("nan"), default=0) == 0
        assert _safe_int(True, default=0) == 0
        assert _safe_int(False, default=0) == 0
        assert _safe_int([], default=0) == 0
        assert _safe_int({}, default=0) == 0

    def test_from_dict_with_dirty_coercions(self) -> None:
        dirty = {
            "name": "dirty_sink",
            "description": "Dirty Sink",
            "volume": "32000.5",
            "maxVolume": "100",
            "muted": 1,
        }
        sink = RemoteAudioSink.from_dict(dirty)
        assert sink.volume == 100  # clamped to max_volume=100
        assert sink.max_volume == 100
        assert sink.muted is True

    def test_from_dict_non_dict_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            RemoteAudioSink.from_dict("invalid_non_dict")  # type: ignore[arg-type]


# =============================================================================
# 2. Plugin Metadata, Capabilities, and Lifecycle Tests
# =============================================================================


class TestRemoteSystemVolumePluginMetadataAndLifecycle:
    """Tests for plugin attributes, registration capabilities, and lifecycle."""

    def test_metadata_and_capabilities(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        assert plugin.plugin_id == "remotesystemvolume"
        assert plugin.name == "remotesystemvolume"
        assert plugin.incoming_capabilities == ["kdeconnect.systemvolume"]
        assert plugin.outgoing_capabilities == ["kdeconnect.systemvolume.request"]

    async def test_lifecycle_subscriptions_and_teardown(self) -> None:
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)

        await plugin.start()
        # Should have registered subscriptions for disconnect and unpair
        assert len(plugin._unsub_callbacks) >= 2

        # Populate internal state
        plugin._device_sinks["dev_1"] = {
            "speaker": RemoteAudioSink(name="speaker", volume=1000)
        }
        plugin._connected_peers["dev_1"] = MagicMock()

        await plugin.stop()
        assert len(plugin._unsub_callbacks) == 0
        assert len(plugin._device_sinks) == 0
        assert len(plugin._connected_peers) == 0


# =============================================================================
# 3. Sink List Ingestion and Caching Tests
# =============================================================================


class TestRemoteSystemVolumePluginSinkListIngestion:
    """Tests for receiving full sinkList packets and updating the cache."""

    async def test_sink_list_ingestion_and_event_publishing(self) -> None:
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        received_events: list[RemoteAudioSinkListEvent] = []
        bus.subscribe(RemoteAudioSinkListEvent, received_events.append)

        pkt = _create_packet(
            {
                "sinkList": [
                    {
                        "name": "phone_speaker",
                        "description": "Phone Speaker",
                        "volume": 35000,
                        "maxVolume": 65536,
                        "muted": False,
                    },
                    {
                        "name": "phone_earpiece",
                        "description": "Phone Earpiece",
                        "volume": 20000,
                        "maxVolume": 65536,
                        "muted": True,
                    },
                ]
            }
        )

        mock_send = MagicMock()
        await plugin.handle_packet(pkt, device_id="phone_123", send_packet=mock_send)

        # Cache check
        sinks = plugin.get_sinks("phone_123")
        assert len(sinks) == 2
        names = {s.name for s in sinks}
        assert names == {"phone_speaker", "phone_earpiece"}

        speaker = plugin.get_sink("phone_123", "phone_speaker")
        assert speaker is not None
        assert speaker.volume == 35000
        assert speaker.muted is False

        earpiece = plugin.get_sink("phone_123", "phone_earpiece")
        assert earpiece is not None
        assert earpiece.volume == 20000
        assert earpiece.muted is True

        # Event emission check
        assert len(received_events) == 1
        evt = received_events[0]
        assert evt.device_id == "phone_123"
        assert len(evt.sinks) == 2

    async def test_empty_sink_list_clears_device_cache(self) -> None:
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        # Seed initial sinks
        initial_pkt = _create_packet(
            {
                "sinkList": [
                    {"name": "speaker", "volume": 30000},
                ]
            }
        )
        await plugin.handle_packet(
            initial_pkt, device_id="phone_abc", send_packet=MagicMock()
        )
        assert len(plugin.get_sinks("phone_abc")) == 1

        # Ingest empty sinkList
        clearing_events: list[RemoteAudioSinkListEvent] = []
        bus.subscribe(RemoteAudioSinkListEvent, clearing_events.append)

        empty_pkt = _create_packet({"sinkList": []})
        await plugin.handle_packet(
            empty_pkt, device_id="phone_abc", send_packet=MagicMock()
        )

        # Cache must be empty
        assert len(plugin.get_sinks("phone_abc")) == 0
        assert len(clearing_events) == 1
        assert clearing_events[0].device_id == "phone_abc"
        assert len(clearing_events[0].sinks) == 0

    async def test_subsequent_sink_list_replaces_device_cache(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        mock_send = MagicMock()

        # First list
        pkt1 = _create_packet(
            {"sinkList": [{"name": "sink_old_1"}, {"name": "sink_old_2"}]}
        )
        await plugin.handle_packet(pkt1, device_id="dev_1", send_packet=mock_send)
        assert len(plugin.get_sinks("dev_1")) == 2

        # Second list replaces
        pkt2 = _create_packet({"sinkList": [{"name": "sink_new_1"}]})
        await plugin.handle_packet(pkt2, device_id="dev_1", send_packet=mock_send)

        sinks = plugin.get_sinks("dev_1")
        assert len(sinks) == 1
        assert sinks[0].name == "sink_new_1"


# =============================================================================
# 4. Incremental Single Sink Update Tests
# =============================================================================


class TestRemoteSystemVolumePluginIncrementalUpdates:
    """Tests for single sink mutation packets (volume, muted, description)."""

    async def test_incremental_volume_and_mute_updates(self) -> None:
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        # Seed initial sink
        pkt_list = _create_packet(
            {
                "sinkList": [
                    {
                        "name": "speaker",
                        "description": "Phone Speaker",
                        "volume": 30000,
                        "maxVolume": 65536,
                        "muted": False,
                    }
                ]
            }
        )
        await plugin.handle_packet(pkt_list, device_id="dev_1", send_packet=MagicMock())

        updated_events: list[RemoteAudioSinkUpdatedEvent] = []
        bus.subscribe(RemoteAudioSinkUpdatedEvent, updated_events.append)

        # 1. Update volume
        pkt_vol = _create_packet({"name": "speaker", "volume": 45000})
        await plugin.handle_packet(pkt_vol, device_id="dev_1", send_packet=MagicMock())

        sink = plugin.get_sink("dev_1", "speaker")
        assert sink is not None
        assert sink.volume == 45000
        assert sink.description == "Phone Speaker"  # preserved
        assert sink.muted is False

        assert len(updated_events) == 1
        assert updated_events[0].device_id == "dev_1"
        assert updated_events[0].sink_name == "speaker"
        assert updated_events[0].volume == 45000

        # 2. Update mute
        pkt_mute = _create_packet({"name": "speaker", "muted": True})
        await plugin.handle_packet(pkt_mute, device_id="dev_1", send_packet=MagicMock())

        sink_muted = plugin.get_sink("dev_1", "speaker")
        assert sink_muted is not None
        assert sink_muted.volume == 45000  # preserved
        assert sink_muted.muted is True

        assert len(updated_events) == 2
        assert updated_events[1].muted is True

    async def test_single_sink_update_creates_sink_if_not_present(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        pkt = _create_packet(
            {
                "name": "bluetooth_headphones",
                "description": "BT Headphones",
                "volume": 25000,
                "muted": False,
            }
        )
        await plugin.handle_packet(pkt, device_id="dev_x", send_packet=MagicMock())

        sink = plugin.get_sink("dev_x", "bluetooth_headphones")
        assert sink is not None
        assert sink.name == "bluetooth_headphones"
        assert sink.volume == 25000
        assert sink.description == "BT Headphones"


# =============================================================================
# 5. Resilience, Edge Cases, and Security Tests
# =============================================================================


class TestRemoteSystemVolumePluginResilienceAndEdgeCases:
    """Tests for malformed inputs, bad types, pairing guards, and edge cases."""

    async def test_corrupted_entries_in_sink_list_are_safely_ignored(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        malformed_pkt = _create_packet(
            {
                "sinkList": [
                    {"name": "valid_sink_1", "volume": 10000},
                    {"volume": 5000},  # Missing name
                    "completely_invalid_string",  # Non-dict
                    None,  # None entry
                    {"name": ""},  # Empty name
                    {"name": "   "},  # Whitespace name
                    {"name": "valid_sink_2", "volume": "inf"},  # Infinite volume
                ]
            }
        )
        await plugin.handle_packet(
            malformed_pkt, device_id="dev_test", send_packet=MagicMock()
        )

        sinks = plugin.get_sinks("dev_test")
        assert len(sinks) == 2
        names = {s.name for s in sinks}
        assert names == {"valid_sink_1", "valid_sink_2"}

        sink2 = plugin.get_sink("dev_test", "valid_sink_2")
        assert sink2 is not None
        assert sink2.volume == 0  # inf defaulted to 0 safely

    async def test_non_dict_body_and_mismatched_packet_type_dropped(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        mock_send = MagicMock()

        # Non-dict body using mock packet
        bad_body_pkt = MagicMock()
        bad_body_pkt.type = "kdeconnect.systemvolume"
        bad_body_pkt.body = "string_not_dict"
        await plugin.handle_packet(
            bad_body_pkt, device_id="dev1", send_packet=mock_send
        )
        assert len(plugin.get_sinks("dev1")) == 0

        # Empty dict body
        empty_pkt = Packet(type="kdeconnect.systemvolume", body={})
        await plugin.handle_packet(empty_pkt, device_id="dev1", send_packet=mock_send)
        assert len(plugin.get_sinks("dev1")) == 0

        # Wrong packet type
        wrong_type_pkt = Packet(
            type="kdeconnect.ping", body={"sinkList": [{"name": "s1"}]}
        )
        await plugin.handle_packet(
            wrong_type_pkt, device_id="dev1", send_packet=mock_send
        )
        assert len(plugin.get_sinks("dev1")) == 0

    async def test_pairing_checker_drops_unpaired_device_packets(self) -> None:
        is_paired_map = {"paired_phone": True, "unpaired_phone": False}
        checker = lambda dev_id: is_paired_map.get(dev_id, False)

        plugin = RemoteSystemVolumePlugin(is_paired_checker=checker)
        pkt = _create_packet({"sinkList": [{"name": "speaker", "volume": 12345}]})

        # From unpaired phone -> dropped
        await plugin.handle_packet(
            pkt, device_id="unpaired_phone", send_packet=MagicMock()
        )
        assert len(plugin.get_sinks("unpaired_phone")) == 0

        # From paired phone -> accepted
        await plugin.handle_packet(
            pkt, device_id="paired_phone", send_packet=MagicMock()
        )
        assert len(plugin.get_sinks("paired_phone")) == 1

    async def test_pairing_checker_exception_handled_safely(self) -> None:
        def crashing_checker(dev_id: str) -> bool:
            raise RuntimeError("Database error during pairing lookup")

        plugin = RemoteSystemVolumePlugin(is_paired_checker=crashing_checker)
        pkt = _create_packet({"sinkList": [{"name": "speaker", "volume": 12345}]})

        await plugin.handle_packet(pkt, device_id="dev_any", send_packet=MagicMock())
        assert len(plugin.get_sinks("dev_any")) == 0

    def test_query_unknown_device_returns_empty_and_none(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        assert plugin.get_sinks("non_existent_device") == []
        assert plugin.get_sink("non_existent_device", "speaker") is None


# =============================================================================
# 6. Multi-Device Cache Isolation Tests
# =============================================================================


class TestRemoteSystemVolumePluginMultiDeviceIsolation:
    """Tests confirming distinct connected devices maintain isolated sink state."""

    async def test_multi_device_cache_isolation(self) -> None:
        plugin = RemoteSystemVolumePlugin()

        pkt1 = _create_packet({"sinkList": [{"name": "dev1_speaker", "volume": 20000}]})
        pkt2 = _create_packet({"sinkList": [{"name": "dev2_speaker", "volume": 50000}]})

        await plugin.handle_packet(pkt1, device_id="dev_1", send_packet=MagicMock())
        await plugin.handle_packet(pkt2, device_id="dev_2", send_packet=MagicMock())

        sinks1 = plugin.get_sinks("dev_1")
        sinks2 = plugin.get_sinks("dev_2")

        assert len(sinks1) == 1
        assert sinks1[0].name == "dev1_speaker"
        assert sinks1[0].volume == 20000

        assert len(sinks2) == 1
        assert sinks2[0].name == "dev2_speaker"
        assert sinks2[0].volume == 50000

        # Modifying dev_1 doesn't mutate dev_2
        pkt_mod = _create_packet({"name": "dev1_speaker", "volume": 30000})
        await plugin.handle_packet(pkt_mod, device_id="dev_1", send_packet=MagicMock())

        assert plugin.get_sink("dev_1", "dev1_speaker").volume == 30000
        assert plugin.get_sink("dev_2", "dev2_speaker").volume == 50000

        # Emptying dev_1 leaves dev_2 untouched
        await plugin.handle_packet(
            _create_packet({"sinkList": []}), device_id="dev_1", send_packet=MagicMock()
        )
        assert len(plugin.get_sinks("dev_1")) == 0
        assert len(plugin.get_sinks("dev_2")) == 1


# =============================================================================
# 7. Outbound Requests Tests
# =============================================================================


class TestRemoteSystemVolumePluginOutboundRequests:
    """Tests for outbound control operations to remote devices."""

    async def test_request_remote_sinks(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        sent: list[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent.append(pkt)

        await plugin.request_remote_sinks("phone_1", send_packet=mock_send)
        assert len(sent) == 1
        assert sent[0].type == "kdeconnect.systemvolume.request"
        assert sent[0].body == {"requestSinks": True}

    async def test_set_remote_volume_nominal_and_clamping(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        sent: list[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent.append(pkt)

        # Seed sink with maxVolume: 100
        pkt_seed = _create_packet(
            {"sinkList": [{"name": "android_vol", "maxVolume": 100, "volume": 50}]}
        )
        await plugin.handle_packet(pkt_seed, device_id="phone_1", send_packet=mock_send)

        # Setting volume 75 -> within bounds [0, 100]
        await plugin.set_remote_volume(
            "phone_1", "android_vol", 75, send_packet=mock_send
        )
        assert sent[-1].body == {"name": "android_vol", "volume": 75}

        # Setting volume 150 -> clamped to 100
        await plugin.set_remote_volume(
            "phone_1", "android_vol", 150, send_packet=mock_send
        )
        assert sent[-1].body == {"name": "android_vol", "volume": 100}

        # Setting volume -10 -> clamped to 0
        await plugin.set_remote_volume(
            "phone_1", "android_vol", -10, send_packet=mock_send
        )
        assert sent[-1].body == {"name": "android_vol", "volume": 0}

    async def test_set_remote_muted(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        sent: list[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent.append(pkt)

        await plugin.set_remote_muted("phone_1", "speaker", True, send_packet=mock_send)
        assert len(sent) == 1
        assert sent[0].type == "kdeconnect.systemvolume.request"
        assert sent[0].body == {"name": "speaker", "muted": True}

        await plugin.set_remote_muted(
            "phone_1", "speaker", False, send_packet=mock_send
        )
        assert len(sent) == 2
        assert sent[1].body == {"name": "speaker", "muted": False}

    async def test_set_remote_default(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        sent: list[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent.append(pkt)

        await plugin.set_remote_default("phone_1", "headphones", send_packet=mock_send)
        assert len(sent) == 1
        assert sent[0].type == "kdeconnect.systemvolume.request"
        assert sent[0].body == {"name": "headphones", "enabled": True, "default": True}

    async def test_outbound_routing_via_cached_peers_or_packet_sender(self) -> None:
        sent_peer: list[Packet] = []

        async def peer_sender(pkt: Packet) -> None:
            sent_peer.append(pkt)

        plugin = RemoteSystemVolumePlugin()
        # Register peer via incoming packet
        await plugin.handle_packet(
            _create_packet({"sinkList": []}),
            device_id="phone_peer",
            send_packet=peer_sender,
        )

        # Call outbound control without passing send_packet
        await plugin.request_remote_sinks("phone_peer")
        assert len(sent_peer) == 1
        assert sent_peer[0].body.get("requestSinks") is True

        # Call on unknown device without sender raises PenguLinkError
        with pytest.raises(PenguLinkError):
            await plugin.request_remote_sinks("unknown_phone")


# =============================================================================
# 8. Eviction and Teardown Tests
# =============================================================================


class TestRemoteSystemVolumePluginEvictionAndTeardown:
    """Tests for unpair cache eviction and lifecycle disconnection."""

    async def test_evict_device_explicitly(self) -> None:
        plugin = RemoteSystemVolumePlugin()
        await plugin.handle_packet(
            _create_packet({"sinkList": [{"name": "speaker", "volume": 1000}]}),
            device_id="phone_to_evict",
            send_packet=MagicMock(),
        )
        assert len(plugin.get_sinks("phone_to_evict")) == 1

        plugin.evict_device("phone_to_evict")
        assert len(plugin.get_sinks("phone_to_evict")) == 0
        assert "phone_to_evict" not in plugin._device_sinks
        assert "phone_to_evict" not in plugin._connected_peers

    async def test_unpair_event_evicts_cache_automatically(self) -> None:
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        # Seed device sinks
        await plugin.handle_packet(
            _create_packet({"sinkList": [{"name": "speaker", "volume": 5000}]}),
            device_id="phone_unpaired",
            send_packet=MagicMock(),
        )
        assert len(plugin.get_sinks("phone_unpaired")) == 1

        # Publish DeviceUnpaired
        await bus.publish(DeviceUnpaired(device_id="phone_unpaired"))
        assert len(plugin.get_sinks("phone_unpaired")) == 0
        assert "phone_unpaired" not in plugin._device_sinks

    async def test_device_unpaired_event_alias_evicts_cache(self) -> None:
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        await plugin.handle_packet(
            _create_packet({"sinkList": [{"name": "speaker", "volume": 5000}]}),
            device_id="phone_unpaired_alias",
            send_packet=MagicMock(),
        )
        assert len(plugin.get_sinks("phone_unpaired_alias")) == 1

        # Publish DeviceUnpairedEvent
        await bus.publish(DeviceUnpairedEvent(device_id="phone_unpaired_alias"))
        assert len(plugin.get_sinks("phone_unpaired_alias")) == 0
        assert "phone_unpaired_alias" not in plugin._device_sinks

    async def test_disconnect_event_cleans_up_sender_retains_cache(self) -> None:
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        mock_send = MagicMock()
        await plugin.handle_packet(
            _create_packet({"sinkList": [{"name": "speaker", "volume": 5000}]}),
            device_id="phone_disc",
            send_packet=mock_send,
        )
        assert "phone_disc" in plugin._connected_peers
        assert len(plugin.get_sinks("phone_disc")) == 1

        # Publish DeviceDisconnected
        await bus.publish(DeviceDisconnected(device_id="phone_disc"))

        # Peer sender removed, but sinks retained in cache for reconnection
        assert "phone_disc" not in plugin._connected_peers
        assert len(plugin.get_sinks("phone_disc")) == 1


# =============================================================================
# 10. Remediation & Hardening Tests (Reviewer 2 Findings)
# =============================================================================


class TestRemoteSystemVolumeHardeningRemediation:
    """Explicit tests for Reviewer 2 recommendations:
    1. _connected_peers registration strictly after pairing check and format validation.
    2. set_remote_default outbound format (enabled: True, default: True) and interop.
    3. set_remote_volume safe float/NaN/None coercion with _safe_int.
    4. RemoteAudioSink enabled attribute in dataclass, from_dict, to_dict, and single sink updates.
    """

    async def test_unauthenticated_peer_not_registered_in_connected_peers(self) -> None:
        """Unpaired device sending packets must never be registered in _connected_peers."""
        plugin = RemoteSystemVolumePlugin(is_paired_checker=lambda d: False)
        mock_send = MagicMock()
        pkt = _create_packet({"sinkList": [{"name": "sink1", "volume": 100}]})

        await plugin.handle_packet(pkt, "unpaired_attacker", send_packet=mock_send)
        assert "unpaired_attacker" not in plugin._connected_peers
        assert len(plugin.get_sinks("unpaired_attacker")) == 0

    async def test_invalid_packet_body_not_registered_in_connected_peers(self) -> None:
        """Peer sending non-dict or invalid packet must not be registered in _connected_peers."""
        plugin = RemoteSystemVolumePlugin(is_paired_checker=lambda d: True)
        mock_send = MagicMock()

        # Non-dict body
        bad_pkt = MagicMock()
        bad_pkt.type = "kdeconnect.systemvolume"
        bad_pkt.body = "not_a_dict"
        await plugin.handle_packet(bad_pkt, "dev_bad_body", send_packet=mock_send)
        assert "dev_bad_body" not in plugin._connected_peers

        # Wrong packet type
        wrong_type_pkt = Packet(type="kdeconnect.other", body={"sinkList": []})
        await plugin.handle_packet(
            wrong_type_pkt, "dev_wrong_type", send_packet=mock_send
        )
        assert "dev_wrong_type" not in plugin._connected_peers

    async def test_set_remote_default_emits_both_enabled_and_default(self) -> None:
        """set_remote_default must include both 'enabled': True and 'default': True for upstream compatibility."""
        plugin = RemoteSystemVolumePlugin()
        captured: list[Packet] = []

        async def capture(pkt: Packet) -> None:
            captured.append(pkt)

        await plugin.set_remote_default("dev_target", "speakers", send_packet=capture)
        assert len(captured) == 1
        assert captured[0].type == "kdeconnect.systemvolume.request"
        assert captured[0].body == {
            "name": "speakers",
            "enabled": True,
            "default": True,
        }

    async def test_set_remote_volume_safe_int_coercion(self) -> None:
        """set_remote_volume must safely handle NaN, None, non-numeric strings, and floats."""
        plugin = RemoteSystemVolumePlugin()
        captured: list[Packet] = []

        async def capture(pkt: Packet) -> None:
            captured.append(pkt)

        # NaN float -> safe default 0
        await plugin.set_remote_volume(
            "dev1", "sink1", float("nan"), send_packet=capture
        )  # type: ignore[arg-type]
        assert captured[-1].body["volume"] == 0

        # Positive infinity -> safe default 0
        await plugin.set_remote_volume(
            "dev1", "sink1", float("inf"), send_packet=capture
        )  # type: ignore[arg-type]
        assert captured[-1].body["volume"] == 0

        # Non-numeric string -> safe default 0
        await plugin.set_remote_volume(
            "dev1", "sink1", "not_a_number", send_packet=capture
        )  # type: ignore[arg-type]
        assert captured[-1].body["volume"] == 0

        # None -> safe default 0
        await plugin.set_remote_volume("dev1", "sink1", None, send_packet=capture)  # type: ignore[arg-type]
        assert captured[-1].body["volume"] == 0

        # String numeric -> successfully coerced
        await plugin.set_remote_volume("dev1", "sink1", "25000", send_packet=capture)  # type: ignore[arg-type]
        assert captured[-1].body["volume"] == 25000

    def test_remote_audio_sink_enabled_attribute_and_serialization(self) -> None:
        """RemoteAudioSink tracks enabled field and serializes cleanly."""
        sink_default = RemoteAudioSink(name="sink_def")
        assert sink_default.enabled is False
        assert sink_default.to_dict()["enabled"] is False

        sink_enabled = RemoteAudioSink(name="sink_en", enabled=True)
        assert sink_enabled.enabled is True
        assert sink_enabled.to_dict()["enabled"] is True

        # from_dict with enabled
        parsed_en = RemoteAudioSink.from_dict({"name": "s1", "enabled": True})
        assert parsed_en.enabled is True

        # from_dict with default alias
        parsed_def = RemoteAudioSink.from_dict({"name": "s2", "default": True})
        assert parsed_def.enabled is True

        # from_dict without enabled/default defaults to False
        parsed_none = RemoteAudioSink.from_dict({"name": "s3"})
        assert parsed_none.enabled is False

    async def test_single_sink_mutation_handles_enabled(self) -> None:
        """Single-sink mutation updates enabled status on existing sink."""
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        events: list[RemoteAudioSinkUpdatedEvent] = []
        bus.subscribe(RemoteAudioSinkUpdatedEvent, lambda e: events.append(e))

        # Initial sink list
        pkt1 = _create_packet(
            {
                "sinkList": [
                    {"name": "sink_a", "volume": 10000, "enabled": False},
                    {"name": "sink_b", "volume": 20000, "enabled": True},
                ]
            }
        )
        await plugin.handle_packet(pkt1, "phone_1", send_packet=MagicMock())
        sink_a = plugin.get_sink("phone_1", "sink_a")
        assert sink_a is not None and sink_a.enabled is False

        # Mutate sink_a with enabled: True
        pkt2 = _create_packet({"name": "sink_a", "enabled": True})
        await plugin.handle_packet(pkt2, "phone_1", send_packet=MagicMock())

        sink_a_updated = plugin.get_sink("phone_1", "sink_a")
        assert sink_a_updated is not None
        assert sink_a_updated.enabled is True
        assert len(events) == 1
        assert events[0].sink.name == "sink_a"
        assert events[0].sink.enabled is True

    async def test_set_remote_default_interop_with_system_volume_host(
        self,
    ) -> None:
        """Verify set_remote_default output is accepted and processed by SystemVolumePlugin."""
        from pengulink.plugins.systemvolume import SystemVolumePlugin
        from pengulink.services.audio import (
            AudioSink,
            LocalAudioService,
            MockAudioSinkBackend,
        )

        backend = MockAudioSinkBackend(
            [
                AudioSink(name="s1", volume=30000, enabled=True),
                AudioSink(name="s2", volume=30000, enabled=False),
            ]
        )
        service = LocalAudioService(backend=backend)
        host = SystemVolumePlugin(audio_service=service)
        client = RemoteSystemVolumePlugin()
        captured: list[Packet] = []

        await client.set_remote_default(
            "host", "s2", send_packet=lambda p: captured.append(p)
        )
        assert len(captured) == 1
        await host.handle_packet(captured[0], "client", MagicMock())
        sinks = await service.list_sinks()
        assert next(s for s in sinks if s.name == "s2").enabled is True

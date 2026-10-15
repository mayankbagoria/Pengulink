"""tests.unit.test_plugin_mpris
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for MPRIS2 media control subsystem:
- MprisHostAdapter ABC contract
- PlayerState data model & serialization
- MockMprisAdapter in-memory operation
- DBusMprisAdapter echo suppression & detection
- MprisControl server plugin packet handling & broadcasting
- MprisRemote client plugin state tracking, merging, interpolation & unpair eviction
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pengulink.core.events import (
    DeviceConnected,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
    LocalMediaPlayerChanged,
    LocalPlayerListChanged,
    MediaControlCommandReceived,
    RemotePlayerListUpdatedEvent,
    RemotePlayerUpdatedEvent,
)
from pengulink.core.exceptions import (
    ActionUnsupportedError,
    ConnectionClosedError,
    MprisActionError,
    MprisAdapterError,
    MprisAdapterUnavailableError,
    MprisError,
    MprisPlayerNotFoundError,
)
from pengulink.network.packet import Packet
from pengulink.plugins.mpris.adapter import (
    DBusMprisAdapter,
    MockMprisAdapter,
    MprisHostAdapter,
    PlayerState,
    detect_mpris_adapter,
)
from pengulink.plugins.mpris.control import MprisControl
from pengulink.plugins.mpris.remote import MprisRemote, RemotePlayerState


# =====================================================================
# Category 1: MprisHostAdapter ABC & PlayerState
# =====================================================================

class TestMprisHostAdapterABC:
    def test_abc_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            MprisHostAdapter()  # type: ignore[abstract]

    def test_subclass_must_implement_all_methods(self) -> None:
        class IncompleteAdapter(MprisHostAdapter):
            @property
            def name(self) -> str:
                return "incomplete"

            @property
            def is_available(self) -> bool:
                return True

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore[abstract]


class TestPlayerState:
    def test_player_state_defaults(self) -> None:
        st = PlayerState(player="Spotify")
        assert st.player == "Spotify"
        assert st.title == ""
        assert st.artist == ""
        assert st.is_playing is False
        assert st.position_us == 0
        assert st.length_us == -1
        assert st.volume == 1.0

    def test_get_current_position_us_paused(self) -> None:
        st = PlayerState(player="Spotify", is_playing=False, position_us=5_000_000, last_updated=time.time() - 10)
        assert st.get_current_position_us() == 5_000_000

    def test_get_current_position_us_playing_and_clamped(self) -> None:
        now = 1000.0
        st = PlayerState(
            player="Spotify",
            is_playing=True,
            position_us=5_000_000,
            length_us=10_000_000,
            last_updated=now - 2.0,  # 2 seconds ago
        )
        # 5s + 2s = 7s = 7,000,000 us
        pos = st.get_current_position_us(now=now)
        assert pos == 7_000_000

        # Beyond length
        pos_clamped = st.get_current_position_us(now=now + 10.0)
        assert pos_clamped == 10_000_000

    def test_to_wire_dict_conversions(self) -> None:
        st = PlayerState(
            player="Spotify",
            title="Blinding Lights",
            artist="The Weeknd",
            album="After Hours",
            is_playing=True,
            position_us=30_000_000,  # 30s
            length_us=200_000_000,   # 200s
            volume=0.85,
            last_updated=time.time(),
        )
        wire = st.to_wire_dict(include_metadata=True)
        assert wire["player"] == "Spotify"
        assert wire["isPlaying"] is True
        assert wire["pos"] >= 30_000  # ms
        assert wire["length"] == 200_000  # ms
        assert wire["volume"] == 85  # 0..100
        assert wire["title"] == "Blinding Lights"
        assert wire["artist"] == "The Weeknd"
        assert wire["album"] == "After Hours"
        assert wire["nowPlaying"] == "The Weeknd - Blinding Lights"


# =====================================================================
# Category 2: MockMprisAdapter In-Memory Operations
# =====================================================================

class TestMockMprisAdapter:
    async def test_init_and_start_stop(self) -> None:
        adapter = MockMprisAdapter(auto_default_player=True)
        assert adapter.name == "mock"
        assert adapter.is_available is True
        await adapter.start()
        players = await adapter.list_players()
        assert "MockPlayer" in players
        await adapter.stop()

    async def test_get_player_status(self) -> None:
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None
        assert st.player == "MockPlayer"
        assert st.title == "Mock Symphony No. 9"

        st_none = await adapter.get_player_status("NonExistent")
        assert st_none is None
        await adapter.stop()

    async def test_transport_actions(self) -> None:
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()

        callback_data: List[Dict[str, Any]] = []

        async def cb(player: str, delta: Dict[str, Any]) -> None:
            callback_data.append(delta)

        adapter.set_change_callback(cb)

        # Play
        ok = await adapter.send_action("MockPlayer", "Play")
        assert ok is True
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and st.is_playing is True
        assert len(callback_data) == 1
        assert callback_data[-1]["isPlaying"] is True

        # Pause
        ok = await adapter.send_action("MockPlayer", "Pause")
        assert ok is True
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and st.is_playing is False
        assert callback_data[-1]["isPlaying"] is False

        # PlayPause toggle
        ok = await adapter.send_action("MockPlayer", "PlayPause")
        assert ok is True
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and st.is_playing is True

        # Stop
        ok = await adapter.send_action("MockPlayer", "Stop")
        assert ok is True
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and st.is_playing is False
        assert st.position_us == 0

        # Invalid Action
        assert await adapter.send_action("MockPlayer", "Explode") is False
        # Non-existent player
        assert await adapter.send_action("Ghost", "Play") is False

        await adapter.stop()

    async def test_set_volume(self) -> None:
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()

        assert await adapter.set_volume("MockPlayer", 0.40) is True
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and st.volume == 0.40

        # Clamping
        assert await adapter.set_volume("MockPlayer", 1.50) is True
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and st.volume == 1.0

        assert await adapter.set_volume("MockPlayer", -0.50) is True
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and st.volume == 0.0

        # Non-existent player
        assert await adapter.set_volume("Ghost", 0.5) is False
        await adapter.stop()

    async def test_seek_and_set_position(self) -> None:
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()

        # Seek relative
        assert await adapter.seek("MockPlayer", 10_000_000) is True  # +10s
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and st.position_us == 10_000_000

        # Seek negative clamped to 0
        assert await adapter.seek("MockPlayer", -50_000_000) is True
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and st.position_us == 0

        # SetPosition absolute
        assert await adapter.set_position("MockPlayer", 45_000_000) is True
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and st.position_us == 45_000_000

        # Beyond length clamped
        assert await adapter.set_position("MockPlayer", 500_000_000) is True
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and st.position_us == 180_000_000  # length_us

        await adapter.stop()

    async def test_loop_and_shuffle(self) -> None:
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()

        assert await adapter.set_loop_status("MockPlayer", "Track") is True
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and st.loop_status == "Track"

        assert await adapter.set_shuffle("MockPlayer", True) is True
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and st.shuffle is True

        await adapter.stop()

    async def test_mock_player_lifecycle_helpers(self) -> None:
        adapter = MockMprisAdapter(auto_default_player=False)
        await adapter.start()

        adapter.add_mock_player("Custom1", PlayerState(player="Custom1", title="Custom Title"))
        players = await adapter.list_players()
        assert "Custom1" in players

        assert adapter.remove_mock_player("Custom1") is True
        assert "Custom1" not in await adapter.list_players()

        adapter.add_mock_player("Custom2")
        adapter.clear_mock_players()
        assert len(await adapter.list_players()) == 0

        await adapter.stop()


# =====================================================================
# Category 3: Adapter Factory & D-Bus Echo Suppression
# =====================================================================

class TestAdapterFactoryAndDBus:
    def test_detect_adapter_env_override(self) -> None:
        with patch.dict(os.environ, {"PENGULINK_MPRIS_ADAPTER": "mock"}):
            adapter = detect_mpris_adapter()
            assert isinstance(adapter, MockMprisAdapter)

    def test_detect_adapter_explicit_name(self) -> None:
        adapter = detect_mpris_adapter("mock")
        assert isinstance(adapter, MockMprisAdapter)

    def test_detect_adapter_unknown_raises(self) -> None:
        with pytest.raises(MprisAdapterError):
            detect_mpris_adapter("nonexistent_adapter_xyz")

    def test_detect_adapter_fallback_headless(self) -> None:
        with patch.dict(os.environ, {"DBUS_SESSION_BUS_ADDRESS": ""}, clear=True):
            adapter = detect_mpris_adapter()
            assert isinstance(adapter, MockMprisAdapter)

    def test_dbus_echo_suppression_filter(self) -> None:
        assert DBusMprisAdapter.should_ignore_service("org.mpris.MediaPlayer2.kdeconnect.mpris_remote") is True
        assert DBusMprisAdapter.should_ignore_service("org.mpris.MediaPlayer2.pengulink.echo") is True
        assert DBusMprisAdapter.should_ignore_service("org.mpris.MediaPlayer2.playerctld") is True
        assert DBusMprisAdapter.should_ignore_service("org.mpris.MediaPlayer2.spotify") is False
        assert DBusMprisAdapter.should_ignore_service("org.mpris.MediaPlayer2.vlc") is False


# =====================================================================
# Category 4: MprisControl Server Plugin
# =====================================================================

class TestMprisControlServer:
    async def test_capabilities(self) -> None:
        adapter = MockMprisAdapter()
        ctrl = MprisControl(adapter=adapter)
        assert ctrl.incoming_capabilities == ["kdeconnect.mpris.request"]
        assert ctrl.outgoing_capabilities == ["kdeconnect.mpris"]
        assert ctrl.plugin_id == "mpriscontrol"

    async def test_handle_request_player_list(self) -> None:
        adapter = MockMprisAdapter(auto_default_player=True)
        ctrl = MprisControl(adapter=adapter)
        await ctrl.start()

        sent_packets: List[Packet] = []

        async def dummy_sender(pkt: Packet) -> None:
            sent_packets.append(pkt)

        req = Packet(type="kdeconnect.mpris.request", body={"requestPlayerList": True})
        await ctrl.handle_packet(req, "test_phone", dummy_sender)

        assert len(sent_packets) == 1
        resp = sent_packets[0]
        assert resp.type == "kdeconnect.mpris"
        assert "playerList" in resp.body
        assert "MockPlayer" in resp.body["playerList"]
        await ctrl.stop()

    async def test_handle_request_now_playing(self) -> None:
        adapter = MockMprisAdapter(auto_default_player=True)
        ctrl = MprisControl(adapter=adapter)
        await ctrl.start()

        sent_packets: List[Packet] = []

        async def dummy_sender(pkt: Packet) -> None:
            sent_packets.append(pkt)

        req = Packet(
            type="kdeconnect.mpris.request",
            body={"player": "MockPlayer", "requestNowPlaying": True},
        )
        await ctrl.handle_packet(req, "test_phone", dummy_sender)

        assert len(sent_packets) == 1
        resp = sent_packets[0]
        assert resp.type == "kdeconnect.mpris"
        assert resp.body["player"] == "MockPlayer"
        assert resp.body["title"] == "Mock Symphony No. 9"
        await ctrl.stop()

    async def test_handle_transport_actions_and_volume(self) -> None:
        event_bus = EventBus()
        events_received: List[MediaControlCommandReceived] = []
        event_bus.subscribe(MediaControlCommandReceived, lambda ev: events_received.append(ev))

        adapter = MockMprisAdapter(auto_default_player=True)
        ctrl = MprisControl(event_bus=event_bus, adapter=adapter)
        await ctrl.start()

        sent_packets: List[Packet] = []

        async def dummy_sender(pkt: Packet) -> None:
            sent_packets.append(pkt)

        # Action: Play
        req = Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "action": "Play"})
        await ctrl.handle_packet(req, "test_phone", dummy_sender)
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and st.is_playing is True
        assert len(events_received) == 1
        assert events_received[0].command == "action:Play"

        # Action: setVolume 80 (0..100) -> 0.8
        req = Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "setVolume": 80})
        await ctrl.handle_packet(req, "test_phone", dummy_sender)
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and round(st.volume, 2) == 0.80

        # Action: Seek (us)
        req = Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "Seek": 5_000_000})
        await ctrl.handle_packet(req, "test_phone", dummy_sender)
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and st.position_us >= 5_000_000

        # Action: SetPosition (ms -> us)
        req = Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "SetPosition": 45_000})
        await ctrl.handle_packet(req, "test_phone", dummy_sender)
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None and 45_000_000 <= st.position_us <= 46_000_000

        await ctrl.stop()

    async def test_broadcast_on_adapter_change(self) -> None:
        adapter = MockMprisAdapter(auto_default_player=True)
        ctrl = MprisControl(adapter=adapter)
        await ctrl.start()

        sent_packets: List[Packet] = []

        async def dummy_sender(pkt: Packet) -> None:
            sent_packets.append(pkt)

        # Register peer connection via incoming packet
        req = Packet(type="kdeconnect.mpris.request", body={"requestPlayerList": True})
        await ctrl.handle_packet(req, "test_phone", dummy_sender)
        sent_packets.clear()

        # Trigger adapter action which fires change callback
        await adapter.send_action("MockPlayer", "Play")
        assert len(sent_packets) >= 1
        broadcast_pkt = sent_packets[-1]
        assert broadcast_pkt.type == "kdeconnect.mpris"
        assert broadcast_pkt.body.get("player") == "MockPlayer"
        assert broadcast_pkt.body.get("isPlaying") is True

        await ctrl.stop()


# =====================================================================
# Category 5: MprisRemote Client Plugin
# =====================================================================

class TestMprisRemoteClient:
    async def test_capabilities(self) -> None:
        remote = MprisRemote()
        assert remote.incoming_capabilities == ["kdeconnect.mpris"]
        assert remote.outgoing_capabilities == ["kdeconnect.mpris.request"]
        assert remote.plugin_id == "mprisremote"

    async def test_ingest_player_list(self) -> None:
        event_bus = EventBus()
        events: List[RemotePlayerListUpdatedEvent] = []
        event_bus.subscribe(RemotePlayerListUpdatedEvent, lambda ev: events.append(ev))

        remote = MprisRemote(event_bus=event_bus)
        await remote.start()

        pkt = Packet(type="kdeconnect.mpris", body={"playerList": ["Spotify", "VLC"]})

        async def dummy_sender(p: Packet) -> None:
            pass

        await remote.handle_packet(pkt, "phone_1", dummy_sender)

        assert remote.get_players("phone_1") == ["Spotify", "VLC"]
        assert remote.get_active_player("phone_1") == "Spotify"
        assert len(events) == 1
        assert events[0].player_list == ["Spotify", "VLC"]

        await remote.stop()

    async def test_incremental_state_merging(self) -> None:
        remote = MprisRemote()
        await remote.start()

        async def dummy_sender(p: Packet) -> None:
            pass

        # 1. Full packet
        pkt1 = Packet(
            type="kdeconnect.mpris",
            body={
                "player": "Spotify",
                "title": "Blinding Lights",
                "artist": "The Weeknd",
                "album": "After Hours",
                "isPlaying": True,
                "pos": 10000,
                "length": 200000,
                "volume": 75,
            },
        )
        await remote.handle_packet(pkt1, "phone_1", dummy_sender)

        st = remote.get_player_state("phone_1", "Spotify")
        assert st is not None
        assert st.title == "Blinding Lights"
        assert st.artist == "The Weeknd"
        assert st.volume == 75

        # 2. Incremental partial packet (only volume)
        pkt2 = Packet(type="kdeconnect.mpris", body={"player": "Spotify", "volume": 90})
        await remote.handle_packet(pkt2, "phone_1", dummy_sender)

        st_updated = remote.get_player_state("phone_1", "Spotify")
        assert st_updated is not None
        assert st_updated.volume == 90
        # Title and artist must be preserved
        assert st_updated.title == "Blinding Lights"
        assert st_updated.artist == "The Weeknd"

        await remote.stop()

    async def test_dynamic_position_interpolation(self) -> None:
        remote = MprisRemote()
        await remote.start()

        async def dummy_sender(p: Packet) -> None:
            pass

        # Case A: isPlaying == True
        pkt_playing = Packet(
            type="kdeconnect.mpris",
            body={"player": "Spotify", "isPlaying": True, "pos": 20000, "length": 100000},
        )
        await remote.handle_packet(pkt_playing, "phone_1", dummy_sender)
        st = remote.get_player_state("phone_1", "Spotify")
        assert st is not None

        # Interpolation with simulated +3.0 seconds
        future_now = st.last_pos_timestamp + 3.0
        interpolated = st.get_current_position_ms(now=future_now)
        assert interpolated == 23000

        # Clamped to length
        far_future = st.last_pos_timestamp + 120.0
        assert st.get_current_position_ms(now=far_future) == 100000

        # Case B: isPlaying == False
        pkt_paused = Packet(
            type="kdeconnect.mpris",
            body={"player": "Spotify", "isPlaying": False, "pos": 25000, "length": 100000},
        )
        await remote.handle_packet(pkt_paused, "phone_1", dummy_sender)
        st_paused = remote.get_player_state("phone_1", "Spotify")
        assert st_paused is not None
        assert st_paused.get_current_position_ms(now=future_now) == 25000

        await remote.stop()

    async def test_outbound_command_serialization(self) -> None:
        sent_packets: List[Packet] = []

        async def dummy_sender(p: Packet) -> None:
            sent_packets.append(p)

        remote = MprisRemote()
        await remote.start()

        # Ingest player first so active player resolves
        await remote.handle_packet(
            Packet(type="kdeconnect.mpris", body={"player": "Spotify", "isPlaying": True}),
            "phone_1",
            dummy_sender,
        )
        sent_packets.clear()

        # 1. Action
        await remote.send_action("phone_1", "PlayPause")
        assert sent_packets[-1].body == {"player": "Spotify", "action": "PlayPause"}

        # 2. Volume
        await remote.set_volume("phone_1", 85)
        assert sent_packets[-1].body == {"player": "Spotify", "setVolume": 85}

        # 3. Seek: offset_ms -> Seek (us)
        await remote.seek("phone_1", 5000)
        assert sent_packets[-1].body == {"player": "Spotify", "Seek": 5_000_000}

        # 4. SetPosition: position_ms -> SetPosition (ms)
        await remote.set_position("phone_1", 30000)
        assert sent_packets[-1].body == {"player": "Spotify", "SetPosition": 30000}

        # 5. Invalid action raises ActionUnsupportedError
        with pytest.raises(ActionUnsupportedError):
            await remote.send_action("phone_1", "InvalidAction")

        await remote.stop()

    async def test_unpair_cache_eviction(self) -> None:
        event_bus = EventBus()
        remote = MprisRemote(event_bus=event_bus)
        await remote.start()

        async def dummy_sender(p: Packet) -> None:
            pass

        await remote.handle_packet(
            Packet(type="kdeconnect.mpris", body={"playerList": ["Spotify"]}),
            "phone_1",
            dummy_sender,
        )
        await remote.handle_packet(
            Packet(type="kdeconnect.mpris", body={"player": "Spotify", "title": "Track 1"}),
            "phone_1",
            dummy_sender,
        )

        assert "Spotify" in remote.get_players("phone_1")
        assert remote.get_player_state("phone_1", "Spotify") is not None

        # Emit DeviceUnpaired
        await event_bus.publish(DeviceUnpaired(device_id="phone_1"))

        # Cache must be wiped
        assert remote.get_players("phone_1") == []
        assert remote.get_player_state("phone_1", "Spotify") is None

        await remote.stop()

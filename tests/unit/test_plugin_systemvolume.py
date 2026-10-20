"""tests.unit.test_plugin_systemvolume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit test suite for KDE Connect SystemVolume host plugin (pengulink.plugins.systemvolume):
- Plugin metadata, capability advertisements, and lifecycle hooks
- Initial sinkList announcement upon DeviceConnected
- requestSinks handling and AudioSinkListEvent emission
- Remote volume adjustment with automatic unmuting
- Remote mute toggling and default sink activation
- Single-sink incremental update packet generation
- Robustness against missing fields, malformed bodies, and edge-case values
- Multi-peer connection tracking, broadcasting, and disconnect cleanup
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from pengulink.core.events import (
    AudioSinkListEvent,
    AudioSinkUpdatedEvent,
    DeviceConnected,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
)
from pengulink.network.packet import Packet
from pengulink.plugins.systemvolume import (
    SYSTEM_VOLUME_PACKET_TYPE,
    SYSTEM_VOLUME_REQUEST_TYPE,
    SystemVolumePlugin,
)
from pengulink.services.audio import (
    MockAudioSinkBackend,
)


def _create_mock_packet(
    body: dict[str, Any] | None = None, packet_type: str = SYSTEM_VOLUME_REQUEST_TYPE
) -> Packet:
    return Packet(type=packet_type, body=body or {})


class TestSystemVolumePluginMetadataAndLifecycle:
    """Tests for plugin initialization, capabilities, and lifecycle events."""

    def test_metadata_and_capabilities(self) -> None:
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        assert plugin.plugin_id == "systemvolume"
        assert plugin.incoming_capabilities == ["kdeconnect.systemvolume.request"]
        assert plugin.outgoing_capabilities == ["kdeconnect.systemvolume"]

    async def test_lifecycle_subscriptions(self) -> None:
        bus = EventBus()
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(event_bus=bus, backend=backend)

        await plugin.start()
        assert len(plugin._unsub_callbacks) == 3

        # Simulate device connection registration
        plugin._connected_peers["dev1"] = MagicMock()
        assert "dev1" in plugin._connected_peers

        await plugin.stop()
        assert len(plugin._unsub_callbacks) == 0
        assert len(plugin._connected_peers) == 0


class TestSystemVolumePluginConnectionBroadcast:
    """Tests for automatic sinkList broadcast when paired devices connect."""

    async def test_on_device_connected_broadcasts_sink_list(self) -> None:
        bus = EventBus()
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(event_bus=bus, backend=backend)
        await plugin.start()

        sent_packets: list[Packet] = []

        async def mock_sender(pkt: Packet) -> None:
            sent_packets.append(pkt)

        plugin._connected_peers["peer_123"] = mock_sender

        list_events: list[AudioSinkListEvent] = []
        bus.subscribe(AudioSinkListEvent, list_events.append)

        # Trigger DeviceConnected event
        await bus.publish(DeviceConnected(device_id="peer_123", device_name="Phone"))

        assert len(sent_packets) == 1
        pkt = sent_packets[0]
        assert pkt.type == SYSTEM_VOLUME_PACKET_TYPE
        assert "sinkList" in pkt.body
        sinks = pkt.body["sinkList"]
        assert len(sinks) == 2
        assert sinks[0]["name"] == "alsa_output.pci-0000_00_1f.3.speaker"
        assert sinks[0]["maxVolume"] == 65536

        assert len(list_events) == 1
        assert list_events[0].device_id == "peer_123"
        assert len(list_events[0].sinks) == 2

        await plugin.stop()

    async def test_on_device_connected_via_packet_sender(self) -> None:
        bus = EventBus()
        backend = MockAudioSinkBackend()
        sent_packets: list[tuple[str, Packet]] = []

        async def mock_packet_sender(device_id: str, pkt: Packet) -> None:
            sent_packets.append((device_id, pkt))

        plugin = SystemVolumePlugin(
            event_bus=bus, backend=backend, packet_sender=mock_packet_sender
        )
        await plugin.start()

        await bus.publish(DeviceConnected(device_id="peer_456"))
        assert len(sent_packets) == 1
        assert sent_packets[0][0] == "peer_456"
        assert sent_packets[0][1].type == SYSTEM_VOLUME_PACKET_TYPE

        await plugin.stop()


class TestSystemVolumePluginPacketHandling:
    """Tests for inbound packet processing and control reactions."""

    async def test_request_sinks_query(self) -> None:
        bus = EventBus()
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(event_bus=bus, backend=backend)

        sent_packets: list[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        list_events: list[AudioSinkListEvent] = []
        bus.subscribe(AudioSinkListEvent, list_events.append)

        request_packet = _create_mock_packet(body={"requestSinks": True})
        await plugin.handle_packet(request_packet, "dev1", mock_send)

        assert len(sent_packets) == 1
        resp = sent_packets[0]
        assert resp.type == SYSTEM_VOLUME_PACKET_TYPE
        assert "sinkList" in resp.body
        assert len(resp.body["sinkList"]) == 2

        assert len(list_events) == 1
        assert list_events[0].device_id == "dev1"

    async def test_volume_adjustment_unmutes_automatically(self) -> None:
        bus = EventBus()
        backend = MockAudioSinkBackend()
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"

        # Initially mute the speaker
        await backend.set_muted(speaker, True)

        plugin = SystemVolumePlugin(event_bus=bus, backend=backend)
        sent_packets: list[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        updated_events: list[AudioSinkUpdatedEvent] = []
        bus.subscribe(AudioSinkUpdatedEvent, updated_events.append)

        # Send volume adjustment packet
        req = _create_mock_packet(body={"name": speaker, "volume": 45000})
        await plugin.handle_packet(req, "dev1", mock_send)

        # Verify backend was unmuted and volume updated
        sinks = await backend.list_sinks()
        updated_sink = next(s for s in sinks if s.name == speaker)
        assert updated_sink.volume == 45000
        assert updated_sink.muted is False

        # Verify response packet
        assert len(sent_packets) == 1
        resp = sent_packets[0]
        assert resp.type == SYSTEM_VOLUME_PACKET_TYPE
        assert resp.body["name"] == speaker
        assert resp.body["volume"] == 45000
        assert resp.body["muted"] is False

        # Verify EventBus event
        assert len(updated_events) == 1
        assert updated_events[0].device_id == "dev1"
        assert updated_events[0].sink is not None
        assert updated_events[0].sink.volume == 45000
        assert updated_events[0].sink.muted is False

    async def test_mute_toggling(self) -> None:
        bus = EventBus()
        backend = MockAudioSinkBackend()
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"
        plugin = SystemVolumePlugin(event_bus=bus, backend=backend)

        sent_packets: list[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        # Mute
        req = _create_mock_packet(body={"name": speaker, "muted": True})
        await plugin.handle_packet(req, "dev1", mock_send)

        sinks = await backend.list_sinks()
        assert next(s for s in sinks if s.name == speaker).muted is True

        assert len(sent_packets) == 1
        assert sent_packets[0].body["muted"] is True

        # Unmute
        sent_packets.clear()
        req2 = _create_mock_packet(body={"name": speaker, "muted": False})
        await plugin.handle_packet(req2, "dev1", mock_send)

        sinks = await backend.list_sinks()
        assert next(s for s in sinks if s.name == speaker).muted is False
        assert sent_packets[0].body["muted"] is False

    async def test_set_default_enabled(self) -> None:
        bus = EventBus()
        backend = MockAudioSinkBackend()
        hdmi = "alsa_output.pci-0000_00_1f.3.hdmi"
        plugin = SystemVolumePlugin(event_bus=bus, backend=backend)

        sent_packets: list[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        req = _create_mock_packet(body={"name": hdmi, "enabled": True})
        await plugin.handle_packet(req, "dev1", mock_send)

        sinks = await backend.list_sinks()
        assert next(s for s in sinks if s.name == hdmi).enabled is True

        assert len(sent_packets) == 1
        assert sent_packets[0].body["enabled"] is True

    async def test_ignored_when_packet_type_mismatched(self) -> None:
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        sent_packets: list[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        req = _create_mock_packet(
            body={"requestSinks": True}, packet_type="kdeconnect.ping"
        )
        await plugin.handle_packet(req, "dev1", mock_send)
        assert len(sent_packets) == 0


class TestSystemVolumePluginEdgeCasesAndResilience:
    """Tests for malformed inputs, missing fields, and robustness."""

    async def test_missing_or_empty_name_field(self) -> None:
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        sent_packets: list[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        # Missing name completely
        req1 = _create_mock_packet(body={"volume": 30000})
        await plugin.handle_packet(req1, "dev1", mock_send)
        assert len(sent_packets) == 0

        # Empty name string
        req2 = _create_mock_packet(body={"name": "   ", "volume": 30000})
        await plugin.handle_packet(req2, "dev1", mock_send)
        assert len(sent_packets) == 0

        # Non-string, non-int name
        req3 = _create_mock_packet(body={"name": {"nested": "dict"}, "volume": 30000})
        await plugin.handle_packet(req3, "dev1", mock_send)
        assert len(sent_packets) == 0

    async def test_unknown_sink_name(self) -> None:
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        sent_packets: list[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        req = _create_mock_packet(body={"name": "phantom_sink", "volume": 30000})
        await plugin.handle_packet(req, "dev1", mock_send)
        # Safe no-op, no crash
        assert len(sent_packets) == 0

    async def test_volume_bounds_and_type_coercion(self) -> None:
        backend = MockAudioSinkBackend()
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"
        plugin = SystemVolumePlugin(backend=backend)

        sent_packets: list[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        # String volume (common in some JSON clients)
        req1 = _create_mock_packet(body={"name": speaker, "volume": "50000"})
        await plugin.handle_packet(req1, "dev1", mock_send)
        sinks = await backend.list_sinks()
        assert next(s for s in sinks if s.name == speaker).volume == 50000

        # Volume above max (65536)
        req2 = _create_mock_packet(body={"name": speaker, "volume": 200000})
        await plugin.handle_packet(req2, "dev1", mock_send)
        sinks = await backend.list_sinks()
        assert next(s for s in sinks if s.name == speaker).volume == 65536

        # Negative volume
        req3 = _create_mock_packet(body={"name": speaker, "volume": -100})
        await plugin.handle_packet(req3, "dev1", mock_send)
        sinks = await backend.list_sinks()
        assert next(s for s in sinks if s.name == speaker).volume == 0

        # Invalid non-numeric volume
        req4 = _create_mock_packet(body={"name": speaker, "volume": "not_a_number"})
        await plugin.handle_packet(req4, "dev1", mock_send)
        # Volume remains at 0 from previous call
        sinks = await backend.list_sinks()
        assert next(s for s in sinks if s.name == speaker).volume == 0

    async def test_empty_or_none_packet_body(self) -> None:
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        sent_packets: list[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        empty_packet = Packet(type=SYSTEM_VOLUME_REQUEST_TYPE, body={})
        await plugin.handle_packet(empty_packet, "dev1", mock_send)
        assert len(sent_packets) == 0

    async def test_volume_infinity_and_overflow_protection(self) -> None:
        """Volume payloads with inf, -inf, or 1e1000 are safely ignored without OverflowError."""
        backend = MockAudioSinkBackend()
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"
        plugin = SystemVolumePlugin(backend=backend)

        sent_packets: list[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        # Baseline volume is 35389
        for val in ("inf", "-inf", "1e1000", 1e1000):
            req = _create_mock_packet(body={"name": speaker, "volume": val})
            await plugin.handle_packet(req, "dev1", mock_send)
            sinks = await backend.list_sinks()
            assert next(s for s in sinks if s.name == speaker).volume == 35389


class TestSystemVolumePluginPublicAPIsAndBroadcasting:
    """Tests for public helper methods used by CLI and daemon IPC."""

    async def test_public_crud_and_broadcasting(self) -> None:
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"

        peer1_packets: list[Packet] = []
        peer2_packets: list[Packet] = []

        async def send_peer1(pkt: Packet) -> None:
            peer1_packets.append(pkt)

        async def send_peer2(pkt: Packet) -> None:
            peer2_packets.append(pkt)

        plugin._connected_peers["peer1"] = send_peer1
        plugin._connected_peers["peer2"] = send_peer2

        # 1. list_sinks & get_sink
        sinks = await plugin.list_sinks()
        assert len(sinks) == 2
        sink = await plugin.get_sink(speaker)
        assert sink is not None
        assert sink.name == speaker

        # 2. set_volume broadcasts to both peers
        assert await plugin.set_volume(speaker, 48000) is True
        assert len(peer1_packets) == 1
        assert len(peer2_packets) == 1
        assert peer1_packets[0].body["volume"] == 48000
        assert peer2_packets[0].body["volume"] == 48000

        # 3. set_muted broadcasts to both peers
        assert await plugin.set_muted(speaker, True) is True
        assert len(peer1_packets) == 2
        assert peer1_packets[1].body["muted"] is True

        # 4. set_default broadcasts full sink list
        hdmi = "alsa_output.pci-0000_00_1f.3.hdmi"
        assert await plugin.set_default(hdmi) is True
        assert len(peer1_packets) == 3
        assert "sinkList" in peer1_packets[2].body

    async def test_broadcast_pruning_on_broken_pipe(self) -> None:
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"

        async def failing_sender(pkt: Packet) -> None:
            raise ConnectionResetError("Connection lost")

        plugin._connected_peers["broken_peer"] = failing_sender

        sink = await plugin.get_sink(speaker)
        assert sink is not None

        await plugin.broadcast_sink_update(sink)
        # Broken peer must have been evicted from connected peers
        assert "broken_peer" not in plugin._connected_peers

    async def test_disconnect_and_unpair_cleanup(self) -> None:
        bus = EventBus()
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(event_bus=bus, backend=backend)
        await plugin.start()

        plugin._connected_peers["dev_a"] = MagicMock()
        plugin._connected_peers["dev_b"] = MagicMock()

        await bus.publish(DeviceDisconnected(device_id="dev_a"))
        assert "dev_a" not in plugin._connected_peers
        assert "dev_b" in plugin._connected_peers

        await bus.publish(DeviceUnpaired(device_id="dev_b"))
        assert "dev_b" not in plugin._connected_peers

        await plugin.stop()

    async def test_multi_peer_volume_sync_fanout(self) -> None:
        """When peer1 alters volume via handle_packet, peer2 receives broadcast update."""
        backend = MockAudioSinkBackend()
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"
        plugin = SystemVolumePlugin(backend=backend)

        peer1_packets: list[Packet] = []
        peer2_packets: list[Packet] = []

        async def send_peer1(pkt: Packet) -> None:
            peer1_packets.append(pkt)

        async def send_peer2(pkt: Packet) -> None:
            peer2_packets.append(pkt)

        # Register peer2 as connected
        plugin._connected_peers["peer2"] = send_peer2

        req = _create_mock_packet(body={"name": speaker, "volume": 42000})
        await plugin.handle_packet(req, "peer1", send_peer1)

        # Peer1 receives direct reply
        assert len(peer1_packets) == 1
        assert peer1_packets[0].body["volume"] == 42000
        assert peer1_packets[0].body["muted"] is False

        # Peer2 receives fanout broadcast update
        assert len(peer2_packets) == 1
        assert peer2_packets[0].body["name"] == speaker
        assert peer2_packets[0].body["volume"] == 42000

    async def test_concurrent_peer_churn_during_broadcast(self) -> None:
        """Adding a peer while a broadcast is awaiting send does not raise RuntimeError."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"

        received: list[Packet] = []

        async def slow_sender(pkt: Packet) -> None:
            received.append(pkt)
            await asyncio.sleep(0.01)

        plugin._connected_peers["slow_peer"] = slow_sender

        sink = await plugin.get_sink(speaker)
        assert sink is not None

        async def add_peer() -> None:
            await asyncio.sleep(0.002)
            plugin._connected_peers["late_peer"] = slow_sender

        await asyncio.gather(
            plugin.broadcast_sink_update(sink),
            add_peer(),
        )
        assert len(received) == 1
        assert "late_peer" in plugin._connected_peers

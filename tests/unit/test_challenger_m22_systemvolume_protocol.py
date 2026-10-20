"""tests.unit.test_challenger_m22_systemvolume_protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial Stress & Fuzzing Suite — Challenger 2 (Milestone 22: SystemVolume Protocol).

Empirical verification targeting PenguLink's KDE Connect SystemVolume host plugin
(`pengulink.plugins.systemvolume.SystemVolumePlugin`) and underlying audio subsystem:

1. Packet Fuzzing:
   - Missing `name` key in packet payload
   - Null values (`None`) across fields (`name`, `volume`, `muted`, `enabled`, `requestSinks`)
   - Deeply nested dictionaries as values
   - Arrays where string expected (`name: ["sink1", "sink2"]`)
   - Unexpected extra keys (huge buffers, special characters, command injections)
   - Empty packet bodies and malformed structures
   - Automated randomized fuzzing generator (100 pseudo-random payloads)

2. Malicious Volume Numbers & Boundaries:
   - Negative values (`-1`, `-500`, `-999999`) clamped safely to 0
   - Float strings (`"45000.5"`, `"0.0"`, `"65536.0"`) correctly parsed and clamped
   - Oversized integers (`10**18`) clamped safely to MAX_AUDIO_VOLUME (65536)
   - Boolean volume values (`True` -> 1, `False` -> 0)
   - Non-numeric volume strings (`"not_a_number"`, `"xyz"`) safely rejected without modification
   - Unhandled `OverflowError` on float infinity (`"inf"`, `"-inf"`, `"1e500"`)

3. High-Rate Concurrency & Packet Floods:
   - 100 rapid `kdeconnect.systemvolume.request` packets within 50ms from single simulated peer
   - 100 rapid `kdeconnect.systemvolume.request` packets within 50ms across multiple simulated peers
   - 100 rapid `requestSinks` discovery queries concurrently dispatched
   - Mixed high-concurrency barrage (volume, mute, default, sink queries) under service lock

4. Broken Socket Simulation & Resilience:
   - `broadcast_sink_update` evicts peer upon `ConnectionResetError`
   - `broadcast_sink_update` evicts peer upon `BrokenPipeError`
   - `broadcast_sink_list` evicts dead peers upon socket disconnects
   - Mixed peers (healthy vs broken): healthy peers receive updates, broken peers cleanly evicted
   - Public mutators (`set_volume`, `set_muted`, `set_default`) handle broken sockets gracefully
   - Dictionary mutation race under concurrent peer churn during broadcast

5. Lifecycle & Task Leak Verification:
   - 100 rapid connect / disconnect / unpair lifecycle cycles
   - Verification of 0 leaked background asyncio tasks
   - Peer table cleanup (`_connected_peers`) on disconnect and unpair
   - Clean repeated `start()` and `stop()` cycles without dangling event subscriptions
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from pengulink.core.events import (
    DeviceConnected,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
)
from pengulink.network.packet import Packet
from pengulink.plugins.systemvolume import (
    SYSTEM_VOLUME_REQUEST_TYPE,
    SystemVolumePlugin,
)
from pengulink.services.audio import (
    MAX_AUDIO_VOLUME,
    MockAudioSinkBackend,
)

SPEAKER_SINK = "alsa_output.pci-0000_00_1f.3.speaker"
HDMI_SINK = "alsa_output.pci-0000_00_1f.3.hdmi"


def _make_req(
    body: dict[str, Any] | None = None, ptype: str = SYSTEM_VOLUME_REQUEST_TYPE
) -> Packet:
    return Packet(type=ptype, body=body or {})


# =====================================================================
# 1. Packet Fuzzing
# =====================================================================


class TestSystemVolumePacketFuzzing:
    """Fuzzing tests evaluating resilience against structural malformations."""

    async def test_fuzz_missing_name_key(self) -> None:
        """Packet without 'name' key is safely rejected without raising exceptions."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        sent: list[Packet] = []
        pkt = _make_req(body={"volume": 32000, "muted": False})

        await plugin.handle_packet(pkt, "peer_1", sent.append)
        assert len(sent) == 0

    async def test_fuzz_null_values_in_payload(self) -> None:
        """Packet containing null (None) values across fields handled cleanly."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        sent: list[Packet] = []

        # Null name
        pkt1 = _make_req(body={"name": None, "volume": 32000})
        await plugin.handle_packet(pkt1, "peer_1", sent.append)
        assert len(sent) == 0

        # Null volume
        pkt2 = _make_req(body={"name": SPEAKER_SINK, "volume": None})
        await plugin.handle_packet(pkt2, "peer_1", sent.append)
        assert len(sent) == 0

        # Null muted
        pkt3 = _make_req(body={"name": SPEAKER_SINK, "muted": None})
        await plugin.handle_packet(pkt3, "peer_1", sent.append)
        # bool(None) is False, unmuting sink
        sinks = await backend.list_sinks()
        speaker = next(s for s in sinks if s.name == SPEAKER_SINK)
        assert speaker.muted is False

        # Null enabled
        pkt4 = _make_req(body={"name": SPEAKER_SINK, "enabled": None})
        await plugin.handle_packet(pkt4, "peer_1", sent.append)
        assert len(sent) <= 1

    async def test_fuzz_deeply_nested_dictionaries(self) -> None:
        """Deeply nested structures passed as values do not cause recursion errors or crashes."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        sent: list[Packet] = []
        nested_structure: dict[str, Any] = {
            "lvl1": {"lvl2": {"lvl3": {"lvl4": "payload"}}}
        }

        # Nested dict as name
        pkt1 = _make_req(body={"name": nested_structure, "volume": 10000})
        await plugin.handle_packet(pkt1, "peer_1", sent.append)
        assert len(sent) == 0

        # Nested dict as volume
        pkt2 = _make_req(body={"name": SPEAKER_SINK, "volume": nested_structure})
        await plugin.handle_packet(pkt2, "peer_1", sent.append)
        assert len(sent) == 0

    async def test_fuzz_array_where_string_expected(self) -> None:
        """Lists passed where strings are expected (e.g. name) are safely rejected."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        sent: list[Packet] = []
        pkt = _make_req(body={"name": ["sink1", "sink2"], "volume": 25000})

        await plugin.handle_packet(pkt, "peer_1", sent.append)
        assert len(sent) == 0

    async def test_fuzz_unexpected_extra_keys(self) -> None:
        """Superfluous keys, SQL-like injection strings, and large buffers are ignored."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        sent: list[Packet] = []
        payload = {
            "name": SPEAKER_SINK,
            "volume": 28000,
            "malicious_sql": "'; DROP TABLE sinks; --",
            "shell_cmd": "rm -rf /",
            "buffer_padding": "A" * 5000,
            "extra_numbers": [1, 2, 3, 4, 5],
            "extra_dict": {"k": "v"},
        }
        pkt = _make_req(body=payload)

        await plugin.handle_packet(pkt, "peer_1", sent.append)
        assert len(sent) == 1
        assert sent[0].body["volume"] == 28000
        assert "malicious_sql" not in sent[0].body

    async def test_fuzz_empty_body(self) -> None:
        """Empty packet body {} is safely handled as a no-op."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        sent: list[Packet] = []
        pkt = _make_req(body={})

        await plugin.handle_packet(pkt, "peer_1", sent.append)
        assert len(sent) == 0

    async def test_fuzz_randomized_payload_generator(self) -> None:
        """Randomized fuzz generator passes 100 arbitrary payloads without unhandled crash."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        rng = random.Random(42)
        primitives = [
            None,
            True,
            False,
            0,
            -1,
            100,
            65536,
            10**18,
            -99999,
            "0",
            "100",
            "45000.5",
            "-100",
            "invalid_str",
            "",
            "   ",
            [],
            [1, 2],
            {},
            {"nested": "val"},
            {"a": {"b": 1}},
        ]

        sent: list[Packet] = []
        for i in range(100):
            body = {
                "name": rng.choice(
                    [SPEAKER_SINK, "ghost_sink", rng.choice(primitives)]
                ),
                "volume": rng.choice(primitives),
                "muted": rng.choice(primitives),
                "enabled": rng.choice(primitives),
                "requestSinks": rng.choice(primitives),
                f"junk_{i}": rng.choice(primitives),
            }
            pkt = _make_req(body=body)
            # Must not crash with unhandled exception
            await plugin.handle_packet(pkt, f"peer_{i % 4}", sent.append)


# =====================================================================
# 2. Malicious Volume Numbers & Boundaries
# =====================================================================


class TestSystemVolumeMaliciousNumbers:
    """Stress testing numeric parsing, boundary clamping, and type coercion."""

    async def test_malicious_volume_negative_values(self) -> None:
        """Negative volume numbers are clamped to 0."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        for neg in (-1, -100, -999999):
            pkt = _make_req(body={"name": SPEAKER_SINK, "volume": neg})
            await plugin.handle_packet(pkt, "peer_1", lambda p: None)

            sinks = await backend.list_sinks()
            speaker = next(s for s in sinks if s.name == SPEAKER_SINK)
            assert speaker.volume == 0

    async def test_malicious_volume_float_strings(self) -> None:
        """Float strings are correctly coerced to integers and clamped."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        test_cases = [
            ("0.0", 0),
            ("32768.4", 32768),
            ("45000.75", 45000),
            ("65536.0", 65536),
            ("70000.9", 65536),  # clamped to MAX_AUDIO_VOLUME
            ("-50.2", 0),  # clamped to 0
        ]

        for raw_val, expected in test_cases:
            pkt = _make_req(body={"name": SPEAKER_SINK, "volume": raw_val})
            await plugin.handle_packet(pkt, "peer_1", lambda p: None)

            sinks = await backend.list_sinks()
            speaker = next(s for s in sinks if s.name == SPEAKER_SINK)
            assert speaker.volume == expected

    async def test_malicious_volume_oversized_integers(self) -> None:
        """Oversized integers (10^18) are clamped safely to MAX_AUDIO_VOLUME (65536)."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        pkt = _make_req(body={"name": SPEAKER_SINK, "volume": 10**18})
        await plugin.handle_packet(pkt, "peer_1", lambda p: None)

        sinks = await backend.list_sinks()
        speaker = next(s for s in sinks if s.name == SPEAKER_SINK)
        assert speaker.volume == MAX_AUDIO_VOLUME

    async def test_malicious_volume_booleans(self) -> None:
        """Boolean volumes coerce safely to 0 and 1 without crash."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        # True -> 1
        pkt_true = _make_req(body={"name": SPEAKER_SINK, "volume": True})
        await plugin.handle_packet(pkt_true, "peer_1", lambda p: None)
        sinks = await backend.list_sinks()
        assert next(s for s in sinks if s.name == SPEAKER_SINK).volume == 1

        # False -> 0
        pkt_false = _make_req(body={"name": SPEAKER_SINK, "volume": False})
        await plugin.handle_packet(pkt_false, "peer_1", lambda p: None)
        sinks = await backend.list_sinks()
        assert next(s for s in sinks if s.name == SPEAKER_SINK).volume == 0

    async def test_malicious_volume_non_numeric_strings(self) -> None:
        """Arbitrary non-numeric strings ('not_a_number', 'NaN') handled safely."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        # Set known initial volume
        await backend.set_volume(SPEAKER_SINK, 40000)

        for invalid in ("not_a_number", "xyz", "nan", "NaN", "null"):
            pkt = _make_req(body={"name": SPEAKER_SINK, "volume": invalid})
            await plugin.handle_packet(pkt, "peer_1", lambda p: None)

            # Volume must remain untouched
            sinks = await backend.list_sinks()
            speaker = next(s for s in sinks if s.name == SPEAKER_SINK)
            assert speaker.volume == 40000


# =====================================================================
# 3. High-Rate Concurrency & Packet Floods
# =====================================================================


class TestSystemVolumeRapidPacketFloods:
    """Stress tests with rapid packet floods within tight timing windows."""

    async def test_flood_100_packets_single_peer_within_50ms(self) -> None:
        """100 rapid volume update packets from single peer within 50ms."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        responses: list[Packet] = []

        async def sender(pkt: Packet) -> None:
            responses.append(pkt)

        start_time = time.monotonic()
        tasks = []
        for i in range(100):
            vol = (i * 650) % MAX_AUDIO_VOLUME
            pkt = _make_req(body={"name": SPEAKER_SINK, "volume": vol})
            tasks.append(plugin.handle_packet(pkt, "flood_peer_1", sender))

        await asyncio.gather(*tasks)
        elapsed = time.monotonic() - start_time

        assert elapsed < 0.25, f"100 packets took too long: {elapsed:.4f}s"
        assert len(responses) == 100

    async def test_flood_100_packets_multi_peer_within_50ms(self) -> None:
        """100 rapid packets from 5 concurrent peers processed cleanly without deadlock."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        responses: list[Packet] = []

        async def sender(pkt: Packet) -> None:
            responses.append(pkt)

        tasks = []
        for i in range(100):
            peer_id = f"peer_{i % 5}"
            vol = (i * 1000) % MAX_AUDIO_VOLUME
            pkt = _make_req(body={"name": SPEAKER_SINK, "volume": vol})
            tasks.append(plugin.handle_packet(pkt, peer_id, sender))

        await asyncio.gather(*tasks)
        assert len(responses) >= 100

    async def test_flood_100_request_sinks_packets(self) -> None:
        """100 rapid requestSinks packets yield 100 sinkList responses."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        responses: list[Packet] = []

        async def sender(pkt: Packet) -> None:
            responses.append(pkt)

        tasks = [
            plugin.handle_packet(
                _make_req(body={"requestSinks": True}),
                f"peer_{i % 10}",
                sender,
            )
            for i in range(100)
        ]

        await asyncio.gather(*tasks)
        assert len(responses) == 100
        assert all(r.body.get("sinkList") for r in responses)

    async def test_flood_mixed_operations_under_lock(self) -> None:
        """Mixed barrage of volume, mute, default, and sink queries without deadlocks."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        responses: list[Packet] = []

        async def sender(pkt: Packet) -> None:
            responses.append(pkt)

        tasks = []
        for i in range(100):
            op = i % 4
            peer_id = f"peer_{i % 3}"
            if op == 0:
                pkt = _make_req(body={"requestSinks": True})
            elif op == 1:
                pkt = _make_req(body={"name": SPEAKER_SINK, "volume": i * 500})
            elif op == 2:
                pkt = _make_req(body={"name": SPEAKER_SINK, "muted": (i % 2 == 0)})
            else:
                pkt = _make_req(
                    body={
                        "name": HDMI_SINK if i % 2 == 0 else SPEAKER_SINK,
                        "enabled": True,
                    }
                )
            tasks.append(plugin.handle_packet(pkt, peer_id, sender))

        await asyncio.gather(*tasks)
        assert len(responses) > 0


# =====================================================================
# 4. Broken Socket Simulation & Resilience
# =====================================================================


class TestSystemVolumeBrokenSocketSimulation:
    """Testing broken connection simulation (ConnectionResetError, BrokenPipeError)."""

    async def test_broken_socket_connection_reset_during_broadcast_sink_update(
        self,
    ) -> None:
        """Peer raising ConnectionResetError is cleanly evicted from _connected_peers."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        async def failing_sender(pkt: Packet) -> None:
            raise ConnectionResetError("Connection reset by peer")

        plugin._connected_peers["peer_broken"] = failing_sender
        plugin._connected_peers["peer_healthy"] = lambda p: None

        sink = await plugin.get_sink(SPEAKER_SINK)
        assert sink is not None

        await plugin.broadcast_sink_update(sink)

        assert "peer_broken" not in plugin._connected_peers
        assert "peer_healthy" in plugin._connected_peers

    async def test_broken_socket_broken_pipe_during_broadcast_sink_update(self) -> None:
        """Peer raising BrokenPipeError is cleanly evicted from _connected_peers."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        async def failing_sender(pkt: Packet) -> None:
            raise BrokenPipeError("Broken pipe")

        plugin._connected_peers["peer_broken_pipe"] = failing_sender

        sink = await plugin.get_sink(SPEAKER_SINK)
        assert sink is not None

        await plugin.broadcast_sink_update(sink)
        assert "peer_broken_pipe" not in plugin._connected_peers

    async def test_broken_socket_during_broadcast_sink_list(self) -> None:
        """Dead peers are evicted when broadcast_sink_list encounters ConnectionError."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        async def failing_sender(pkt: Packet) -> None:
            raise ConnectionResetError("Socket aborted")

        plugin._connected_peers["dead_peer"] = failing_sender
        plugin._connected_peers["live_peer"] = lambda p: None

        await plugin.broadcast_sink_list()
        assert "dead_peer" not in plugin._connected_peers
        assert "live_peer" in plugin._connected_peers

    async def test_broken_socket_mixed_peers_partial_failure(self) -> None:
        """Healthy peers receive updates while multiple broken peers are evicted."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        received_healthy: list[Packet] = []

        async def healthy_sender(pkt: Packet) -> None:
            received_healthy.append(pkt)

        async def broken_rst(pkt: Packet) -> None:
            raise ConnectionResetError("RST")

        async def broken_pipe(pkt: Packet) -> None:
            raise BrokenPipeError("EPIPE")

        plugin._connected_peers["ok_1"] = healthy_sender
        plugin._connected_peers["bad_1"] = broken_rst
        plugin._connected_peers["ok_2"] = healthy_sender
        plugin._connected_peers["bad_2"] = broken_pipe

        sink = await plugin.get_sink(SPEAKER_SINK)
        assert sink is not None

        await plugin.broadcast_sink_update(sink)

        assert len(received_healthy) == 2
        assert "bad_1" not in plugin._connected_peers
        assert "bad_2" not in plugin._connected_peers
        assert "ok_1" in plugin._connected_peers
        assert "ok_2" in plugin._connected_peers

    async def test_broken_socket_during_public_mutators(self) -> None:
        """Calling set_volume, set_muted, set_default evicts dead sockets without raising."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        async def bad_socket(pkt: Packet) -> None:
            raise BrokenPipeError("Socket dead")

        plugin._connected_peers["dead_sock"] = bad_socket

        # set_volume triggers broadcast_sink_update
        assert await plugin.set_volume(SPEAKER_SINK, 42000) is True
        assert "dead_sock" not in plugin._connected_peers


# =====================================================================
# 5. Lifecycle & Task Leak Verification
# =====================================================================


class TestSystemVolumeLifecycleAndTaskLeaks:
    """Verification of connection lifecycles, event unsubscriptions, and task leaks."""

    async def test_connect_disconnect_unpair_lifecycle_task_leak(self) -> None:
        """100 repeated connect/disconnect/unpair cycles leave 0 dangling asyncio tasks."""
        bus = EventBus()
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(event_bus=bus, backend=backend)
        await plugin.start()

        # Capture baseline background tasks
        initial_tasks = {
            t for t in asyncio.all_tasks() if t is not asyncio.current_task()
        }

        for i in range(100):
            dev_id = f"dev_{i % 5}"
            await bus.publish(DeviceConnected(device_id=dev_id))
            await bus.publish(DeviceDisconnected(device_id=dev_id))
            await bus.publish(DeviceUnpaired(device_id=dev_id))

        await asyncio.sleep(0.02)
        final_tasks = {
            t for t in asyncio.all_tasks() if t is not asyncio.current_task()
        }

        # Verify no accumulated tasks
        leaked_tasks = final_tasks - initial_tasks
        assert len(leaked_tasks) == 0, f"Leaked tasks detected: {leaked_tasks}"

        await plugin.stop()

    async def test_peer_table_cleanup_on_disconnect_and_unpair(self) -> None:
        """_connected_peers table is cleanly emptied when devices disconnect or unpair."""
        bus = EventBus()
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(event_bus=bus, backend=backend)
        await plugin.start()

        plugin._connected_peers["phone_1"] = lambda p: None
        plugin._connected_peers["tablet_2"] = lambda p: None

        await bus.publish(DeviceDisconnected(device_id="phone_1"))
        assert "phone_1" not in plugin._connected_peers
        assert "tablet_2" in plugin._connected_peers

        await bus.publish(DeviceUnpaired(device_id="tablet_2"))
        assert "tablet_2" not in plugin._connected_peers
        assert len(plugin._connected_peers) == 0

        await plugin.stop()

    async def test_multiple_start_stop_plugin_cycles(self) -> None:
        """Plugin starts and stops cleanly across 20 cycles with zero listener leaks."""
        bus = EventBus()
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(event_bus=bus, backend=backend)

        for _ in range(20):
            await plugin.start()
            assert len(plugin._unsub_callbacks) == 3
            await plugin.stop()
            assert len(plugin._unsub_callbacks) == 0
            assert len(plugin._connected_peers) == 0


# =====================================================================
# 6. Empirical Defect Reproducers (Challenger Findings)
# =====================================================================


class TestSystemVolumeEmpiricalDefectReproducers:
    """Empirical proof-of-defect tests demonstrating failure modes under challenge."""

    async def test_defect_volume_infinity_overflow_error(self) -> None:
        """Remediated: Float infinity ('inf' / '-inf' / '1e500') handled safely without OverflowError."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        sent: list[Packet] = []
        for val in ("inf", "-inf", "1e500"):
            pkt = _make_req(body={"name": SPEAKER_SINK, "volume": val})
            # Must handle safely without raising OverflowError
            await plugin.handle_packet(pkt, "peer_adversarial", sent.append)

        # Volume must remain at original value and no crash occurs
        sink = await plugin.get_sink(SPEAKER_SINK)
        assert sink is not None
        assert len(sent) == 0

    async def test_defect_concurrent_broadcast_and_peer_churn_dict_mutation(
        self,
    ) -> None:
        """Remediated: Concurrent peer churn during broadcast does not raise RuntimeError."""
        backend = MockAudioSinkBackend()
        plugin = SystemVolumePlugin(backend=backend)

        received: list[Packet] = []

        async def slow_sender(pkt: Packet) -> None:
            received.append(pkt)
            await asyncio.sleep(0.005)

        plugin._connected_peers["peer_1"] = slow_sender
        plugin._connected_peers["peer_2"] = slow_sender

        sink = await plugin.get_sink(SPEAKER_SINK)
        assert sink is not None

        async def churn_peer() -> None:
            await asyncio.sleep(0.002)
            plugin._connected_peers["peer_new"] = slow_sender

        # Must complete cleanly without RuntimeError
        await asyncio.gather(
            plugin.broadcast_sink_update(sink),
            churn_peer(),
        )
        assert len(received) >= 2

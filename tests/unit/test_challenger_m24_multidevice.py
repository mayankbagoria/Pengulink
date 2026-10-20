"""tests.unit.test_challenger_m24_multidevice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial Concurrency, Multi-Device Isolation, and Security Stress Suite.
Milestone 24: RemoteSystemVolume Client Plugin.

Targeting: `pengulink.plugins.remotesystemvolume.RemoteSystemVolumePlugin`

Test Matrix:
1. Multi-Device Cache Cross-Talk:
   - 25+ concurrent devices connecting, advertising identical sink names ("speaker", "headphones")
   - Disparate volumes, max_volumes, mute statuses, descriptions
   - Concurrent updates via asyncio.gather with ZERO cross-talk or leakage
   - Prefix/substring device ID collision resistance ("dev_1", "dev_10", "dev_100")
   - Case-insensitive lookups per-device isolation
2. Unpaired Device Rejection & Cache Poisoning:
   - Unpaired devices sending full sinkList or single sink updates strictly dropped
   - Cache poisoning attempts: spoofing another device's ID in packet body
   - Unpaired devices sending empty sinkList to wipe cache
   - Fail-closed security on pairing checker exceptions (KeyError, RuntimeError, etc.)
   - Dynamic mid-stream unpairing dropping subsequent packets
3. Concurrent Rapid Churn & Race Conditions:
   - Simultaneous burst of sink updates, queries, and unpair events
   - Testing for `RuntimeError: dictionary changed size during iteration` or deadlocks
   - Reentrant EventBus handlers calling plugin mutators during event emission
   - Rapid connect/disconnect/unpair lifecycle loops (zero resource/sender leaks)
   - Idempotent concurrent unpair events
4. Outbound Request Error Handling & Socket Resilience:
   - Calling `set_remote_volume` when `send_packet` raises ConnectionResetError, BrokenPipeError, OSError
   - Exception propagation without state corruption or collateral damage to other devices
   - Outbound requests to evicted/unpaired devices raise PenguLinkError
   - Sync and async sender error handling symmetry
   - Partial socket failure barrage across 20 devices
5. Chaotic Multi-Device Stress Matrix:
   - High-rate chaotic interleaving of valid updates, malformed payloads, disconnects, and queries
"""

from __future__ import annotations

import asyncio
import random
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
from pengulink.plugins.base import SendPacketCallable
from pengulink.plugins.remotesystemvolume import (
    RemoteAudioSink,
    RemoteSystemVolumePlugin,
)


def _make_packet(
    body: dict[str, Any] | None = None,
    ptype: str = "kdeconnect.systemvolume",
) -> Packet:
    return Packet(type=ptype, body=body or {})


# =============================================================================
# 1. Multi-Device Cache Isolation & Cross-Talk Prevention (25+ Concurrent Devices)
# =============================================================================


class TestMultiDeviceCacheCrossTalk:
    """Stress tests proving complete isolation across 25+ concurrent devices."""

    async def test_twenty_five_concurrent_devices_identical_sink_names(self) -> None:
        """25 devices simultaneously advertising identical sink names maintain 100% isolation."""
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        num_devices = 25
        device_ids = [f"device_{i:02d}" for i in range(num_devices)]

        # Every device has identical sink names: "speaker", "headphones", "custom"
        # but completely distinct volumes, mute states, and descriptions.
        async def connect_device(idx: int, dev_id: str) -> None:
            pkt = _make_packet(
                {
                    "sinkList": [
                        {
                            "name": "speaker",
                            "description": f"Speaker for {dev_id}",
                            "volume": 1000 * (idx + 1),
                            "maxVolume": 65536,
                            "muted": (idx % 2 == 0),
                        },
                        {
                            "name": "headphones",
                            "description": f"Headphones for {dev_id}",
                            "volume": 2 * (idx + 1),
                            "maxVolume": 100,
                            "muted": (idx % 3 == 0),
                        },
                        {
                            "name": "custom",
                            "description": f"Custom {dev_id}",
                            "volume": 100 * (idx + 1),
                            "maxVolume": 5000,
                            "muted": False,
                        },
                    ]
                }
            )
            mock_send = MagicMock()
            await plugin.handle_packet(pkt, device_id=dev_id, send_packet=mock_send)

        # Launch all 25 connections concurrently
        tasks = [connect_device(i, dev_ids) for i, dev_ids in enumerate(device_ids)]
        await asyncio.gather(*tasks)

        # Verify strict isolation across all devices
        for idx, dev_id in enumerate(device_ids):
            sinks = plugin.get_sinks(dev_id)
            assert len(sinks) == 3, f"Device {dev_id} must have exactly 3 sinks"

            # Check speaker sink
            speaker = plugin.get_sink(dev_id, "speaker")
            assert speaker is not None
            assert speaker.volume == 1000 * (idx + 1)
            assert speaker.description == f"Speaker for {dev_id}"
            assert speaker.muted is (idx % 2 == 0)
            assert speaker.device_id == dev_id

            # Check headphones sink
            headphones = plugin.get_sink(dev_id, "headphones")
            assert headphones is not None
            assert headphones.volume == 2 * (idx + 1)
            assert headphones.description == f"Headphones for {dev_id}"
            assert headphones.max_volume == 100
            assert headphones.muted is (idx % 3 == 0)
            assert headphones.device_id == dev_id

            # Check custom sink
            custom = plugin.get_sink(dev_id, "custom")
            assert custom is not None
            assert custom.volume == 100 * (idx + 1)
            assert custom.device_id == dev_id

        await plugin.stop()

    async def test_concurrent_single_sink_mutations_no_leakage(self) -> None:
        """25 devices concurrently receiving single-sink mutations with identical sink names."""
        plugin = RemoteSystemVolumePlugin()
        num_devices = 25
        device_ids = [f"peer_{i:02d}" for i in range(num_devices)]

        # Seed initial sinks
        seed_tasks = [
            plugin.handle_packet(
                _make_packet(
                    {
                        "sinkList": [
                            {
                                "name": "speaker",
                                "volume": 1000,
                                "description": f"Initial {dev_id}",
                            }
                        ]
                    }
                ),
                device_id=dev_id,
                send_packet=MagicMock(),
            )
            for dev_id in device_ids
        ]
        await asyncio.gather(*seed_tasks)

        # Concurrently mutate "speaker" to unique values on each device
        async def mutate_sink(idx: int, dev_id: str) -> None:
            new_vol = 2500 * (idx + 1)
            pkt = _make_packet(
                {
                    "name": "speaker",
                    "volume": new_vol,
                    "muted": bool(idx % 2),
                }
            )
            await plugin.handle_packet(pkt, device_id=dev_id, send_packet=MagicMock())

        mutate_tasks = [mutate_sink(i, dev_id) for i, dev_id in enumerate(device_ids)]
        await asyncio.gather(*mutate_tasks)

        # Assert no value leaked from one device to another
        for idx, dev_id in enumerate(device_ids):
            sink = plugin.get_sink(dev_id, "speaker")
            assert sink is not None
            assert sink.volume == 2500 * (idx + 1), (
                f"Cross-talk detected on {dev_id}: expected {2500 * (idx + 1)}, got {sink.volume}"
            )
            assert sink.muted == bool(idx % 2)
            # Original description must have been preserved
            assert sink.description == f"Initial {dev_id}"

    async def test_interleaved_sinklist_and_incremental_updates(self) -> None:
        """Device A gets incremental updates while Device B gets a full sinkList reset."""
        plugin = RemoteSystemVolumePlugin()
        dev_a = "device_alpha"
        dev_b = "device_beta"

        # Seed both
        await plugin.handle_packet(
            _make_packet({"sinkList": [{"name": "speaker", "volume": 10000}]}),
            dev_a,
            MagicMock(),
        )
        await plugin.handle_packet(
            _make_packet({"sinkList": [{"name": "speaker", "volume": 20000}]}),
            dev_b,
            MagicMock(),
        )

        # Device A receives incremental mutation
        pkt_a = _make_packet({"name": "speaker", "volume": 15000, "muted": True})
        # Device B receives full list replacement with completely different sinks
        pkt_b = _make_packet(
            {"sinkList": [{"name": "hdmi", "volume": 40000, "maxVolume": 65536}]}
        )

        await asyncio.gather(
            plugin.handle_packet(pkt_a, dev_a, MagicMock()),
            plugin.handle_packet(pkt_b, dev_b, MagicMock()),
        )

        sink_a = plugin.get_sink(dev_a, "speaker")
        assert sink_a is not None
        assert sink_a.volume == 15000
        assert sink_a.muted is True
        assert plugin.get_sink(dev_a, "hdmi") is None

        sink_b = plugin.get_sink(dev_b, "hdmi")
        assert sink_b is not None
        assert sink_b.volume == 40000
        assert plugin.get_sink(dev_b, "speaker") is None

    async def test_cache_wipe_on_one_device_preserves_others(self) -> None:
        """Empty sinkList on one device clears only that device's cache."""
        plugin = RemoteSystemVolumePlugin()
        devices = [f"node_{i}" for i in range(10)]

        for dev in devices:
            await plugin.handle_packet(
                _make_packet({"sinkList": [{"name": "main_sink", "volume": 12345}]}),
                dev,
                MagicMock(),
            )
            assert len(plugin.get_sinks(dev)) == 1

        # Clear node_3 only
        await plugin.handle_packet(
            _make_packet({"sinkList": []}),
            "node_3",
            MagicMock(),
        )

        # node_3 is empty, all others intact
        assert len(plugin.get_sinks("node_3")) == 0
        for dev in devices:
            if dev != "node_3":
                assert len(plugin.get_sinks(dev)) == 1
                assert plugin.get_sink(dev, "main_sink").volume == 12345

    def test_case_insensitive_name_lookup_isolation(self) -> None:
        """Case-insensitive lookups on device A do not leak or resolve to device B sinks."""
        plugin = RemoteSystemVolumePlugin()
        sink_a = RemoteAudioSink(name="Speaker", volume=11111, device_id="dev_a")
        sink_b = RemoteAudioSink(name="SPEAKER", volume=22222, device_id="dev_b")

        plugin._device_sinks["dev_a"] = {"Speaker": sink_a}
        plugin._device_sinks["dev_b"] = {"SPEAKER": sink_b}

        res_a = plugin.get_sink("dev_a", "speaker")
        assert res_a is not None
        assert res_a.volume == 11111
        assert res_a.device_id == "dev_a"

        res_b = plugin.get_sink("dev_b", "speaker")
        assert res_b is not None
        assert res_b.volume == 22222
        assert res_b.device_id == "dev_b"

    async def test_prefix_and_substring_device_ids_strict_isolation(self) -> None:
        """Devices with prefix or substring names do not cross-contaminate."""
        plugin = RemoteSystemVolumePlugin()
        tricky_ids = ["dev_1", "dev_10", "dev_100", "dev_1_extra", "dev_10_alt"]

        for idx, dev_id in enumerate(tricky_ids):
            await plugin.handle_packet(
                _make_packet(
                    {"sinkList": [{"name": "output", "volume": (idx + 1) * 7000}]}
                ),
                dev_id,
                MagicMock(),
            )

        for idx, dev_id in enumerate(tricky_ids):
            sink = plugin.get_sink(dev_id, "output")
            assert sink is not None
            assert sink.volume == (idx + 1) * 7000
            assert sink.device_id == dev_id


# =============================================================================
# 2. Unpaired Device Rejection & Cache Poisoning Defense
# =============================================================================


class TestUnpairedDeviceRejectionAndCachePoisoning:
    """Security tests verifying strict enforcement of pairing checks and anti-poisoning."""

    async def test_unpaired_device_sink_list_strictly_rejected(self) -> None:
        """Unpaired device cannot inject sinks into cache or emit events."""
        bus = EventBus()
        paired_devices = {"legit_device": True}
        plugin = RemoteSystemVolumePlugin(
            event_bus=bus,
            is_paired_checker=lambda dev: paired_devices.get(dev, False),
        )
        await plugin.start()

        events_captured: list[Any] = []
        bus.subscribe(RemoteAudioSinkListEvent, events_captured.append)
        bus.subscribe(RemoteAudioSinkUpdatedEvent, events_captured.append)

        # Attacker attempts to register sinks
        attacker_pkt = _make_packet(
            {"sinkList": [{"name": "evil_sink", "volume": 666}]}
        )
        await plugin.handle_packet(
            attacker_pkt, device_id="unpaired_mallory", send_packet=MagicMock()
        )

        assert plugin.get_sinks("unpaired_mallory") == []
        assert len(events_captured) == 0

        # Legit device registers sinks
        legit_pkt = _make_packet({"sinkList": [{"name": "good_sink", "volume": 30000}]})
        await plugin.handle_packet(
            legit_pkt, device_id="legit_device", send_packet=MagicMock()
        )

        assert len(plugin.get_sinks("legit_device")) == 1
        assert len(events_captured) == 1
        assert events_captured[0].device_id == "legit_device"

        await plugin.stop()

    async def test_unpaired_device_single_sink_mutation_rejected(self) -> None:
        """Unpaired device cannot send incremental sink updates."""
        paired_devices = {"device_1": True}
        plugin = RemoteSystemVolumePlugin(
            is_paired_checker=lambda dev: paired_devices.get(dev, False)
        )

        attacker_pkt = _make_packet({"name": "speaker", "volume": 50000})
        await plugin.handle_packet(
            attacker_pkt, device_id="unpaired_device", send_packet=MagicMock()
        )

        assert plugin.get_sinks("unpaired_device") == []
        assert plugin.get_sink("unpaired_device", "speaker") is None

    async def test_cache_poisoning_with_spoofed_device_id_in_payload(self) -> None:
        """Attacker embedding victim's device_id inside payload cannot alter victim cache."""
        paired_devices = {"victim_phone": True, "attacker_phone": False}
        plugin = RemoteSystemVolumePlugin(
            is_paired_checker=lambda dev: paired_devices.get(dev, False)
        )

        # Seed victim phone
        await plugin.handle_packet(
            _make_packet({"sinkList": [{"name": "speaker", "volume": 40000}]}),
            device_id="victim_phone",
            send_packet=MagicMock(),
        )
        assert plugin.get_sink("victim_phone", "speaker").volume == 40000

        # Attacker sends packet with "device_id": "victim_phone" inside JSON body
        poison_pkt = _make_packet(
            {
                "sinkList": [
                    {
                        "name": "speaker",
                        "volume": 0,
                        "device_id": "victim_phone",
                    }
                ],
                "device_id": "victim_phone",
            }
        )
        # Transport level connection is authenticated as attacker_phone
        await plugin.handle_packet(
            poison_pkt, device_id="attacker_phone", send_packet=MagicMock()
        )

        # Victim's cache MUST remain untouched
        victim_sink = plugin.get_sink("victim_phone", "speaker")
        assert victim_sink is not None
        assert victim_sink.volume == 40000, "Victim sink was poisoned by attacker!"

        # Attacker has no sinks
        assert plugin.get_sinks("attacker_phone") == []

    async def test_pairing_checker_exceptions_fail_closed(self) -> None:
        """Exceptions in pairing checker fail closed (drop packet, no crash)."""
        exception_types = [
            RuntimeError("DB lock"),
            KeyError("device_missing"),
            ValueError("invalid dev id"),
            TypeError("unhashable"),
            PenguLinkError("internal error"),
        ]

        for exc in exception_types:

            def bad_checker(dev: str, error_to_raise: Exception = exc) -> bool:
                raise error_to_raise

            plugin = RemoteSystemVolumePlugin(is_paired_checker=bad_checker)
            pkt = _make_packet({"sinkList": [{"name": "speaker", "volume": 1234}]})

            # Should not raise exception
            await plugin.handle_packet(
                pkt, device_id="any_dev", send_packet=MagicMock()
            )
            assert plugin.get_sinks("any_dev") == []

    async def test_dynamic_unpairing_drops_future_packets(self) -> None:
        """Device that unpairs stops having its packets accepted."""
        paired_status = {"phone_a": True}
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(
            event_bus=bus,
            is_paired_checker=lambda dev: paired_status.get(dev, False),
        )
        await plugin.start()

        # Step 1: Paired -> packet accepted
        await plugin.handle_packet(
            _make_packet({"sinkList": [{"name": "speaker", "volume": 20000}]}),
            "phone_a",
            MagicMock(),
        )
        assert len(plugin.get_sinks("phone_a")) == 1

        # Step 2: Unpair event
        paired_status["phone_a"] = False
        await bus.publish(DeviceUnpaired(device_id="phone_a"))
        assert len(plugin.get_sinks("phone_a")) == 0

        # Step 3: Device attempts new packet -> rejected
        await plugin.handle_packet(
            _make_packet({"sinkList": [{"name": "speaker", "volume": 30000}]}),
            "phone_a",
            MagicMock(),
        )
        assert len(plugin.get_sinks("phone_a")) == 0

        await plugin.stop()

    async def test_unpaired_device_empty_sinklist_does_not_emit_event(self) -> None:
        """Empty sinkList from unpaired device does not publish RemoteAudioSinkListEvent."""
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(
            event_bus=bus, is_paired_checker=lambda dev: False
        )
        await plugin.start()

        events: list[Any] = []
        bus.subscribe(RemoteAudioSinkListEvent, events.append)

        await plugin.handle_packet(
            _make_packet({"sinkList": []}),
            "unpaired_dev",
            MagicMock(),
        )
        assert len(events) == 0

        await plugin.stop()


# =============================================================================
# 3. Concurrent Rapid Churn & Race Conditions
# =============================================================================


class TestConcurrentRapidChurnAndRaceConditions:
    """Stress tests for concurrency races, dictionary mutations, and reentrancy."""

    async def test_concurrent_burst_updates_evictions_and_queries(self) -> None:
        """Simultaneous burst of sinkList, incremental updates, queries, and unpairs.

        Verifies no `RuntimeError: dictionary changed size during iteration` or deadlocks.
        """
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        devices = [f"churn_dev_{i:02d}" for i in range(20)]
        stop_flag = asyncio.Event()

        # Worker 1: Rapid full sinkList updates
        async def sink_list_worker(dev: str) -> None:
            for seq in range(30):
                if stop_flag.is_set():
                    break
                pkt = _make_packet(
                    {
                        "sinkList": [
                            {
                                "name": f"sink_{j}",
                                "volume": seq * 100 + j,
                                "maxVolume": 65536,
                            }
                            for j in range(random.randint(1, 5))
                        ]
                    }
                )
                await plugin.handle_packet(pkt, dev, MagicMock())
                await asyncio.sleep(0.001)

        # Worker 2: Rapid single sink mutations
        async def single_mutation_worker(dev: str) -> None:
            for seq in range(30):
                if stop_flag.is_set():
                    break
                pkt = _make_packet(
                    {
                        "name": "sink_0",
                        "volume": seq * 200,
                        "muted": (seq % 2 == 0),
                    }
                )
                await plugin.handle_packet(pkt, dev, MagicMock())
                await asyncio.sleep(0.001)

        # Worker 3: Constant queries
        async def query_worker(dev: str) -> None:
            for _ in range(50):
                if stop_flag.is_set():
                    break
                _ = plugin.get_sinks(dev)
                _ = plugin.get_sink(dev, "sink_0")
                _ = plugin.get_sink(dev, "NON_EXISTENT")
                await asyncio.sleep(0.001)

        # Worker 4: Occasional unpair / evictions
        async def unpair_worker(dev: str) -> None:
            for _ in range(5):
                if stop_flag.is_set():
                    break
                await asyncio.sleep(0.01)
                await bus.publish(DeviceUnpaired(device_id=dev))

        tasks = []
        for dev in devices:
            tasks.append(sink_list_worker(dev))
            tasks.append(single_mutation_worker(dev))
            tasks.append(query_worker(dev))
            tasks.append(unpair_worker(dev))

        # Run all 80 tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        stop_flag.set()

        # Assert no unexpected exceptions were raised
        for r in results:
            if isinstance(r, Exception):
                raise r

        await plugin.stop()

    async def test_reentrant_event_handler_calling_plugin_methods(self) -> None:
        """EventBus subscriber calling evict_device and get_sinks inside event handler."""
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        reentrant_logs: list[str] = []

        async def on_updated(evt: RemoteAudioSinkUpdatedEvent) -> None:
            # Reentrant operations while event is being dispatched
            sinks = plugin.get_sinks(evt.device_id)
            reentrant_logs.append(f"sinks_count:{len(sinks)}")
            # Evict during event dispatch
            plugin.evict_device(evt.device_id)
            reentrant_logs.append(f"after_evict:{len(plugin.get_sinks(evt.device_id))}")

        bus.subscribe(RemoteAudioSinkUpdatedEvent, on_updated)

        # Seed device
        await plugin.handle_packet(
            _make_packet({"sinkList": [{"name": "speaker", "volume": 1000}]}),
            "reentrant_dev",
            MagicMock(),
        )

        # Trigger single sink update to fire RemoteAudioSinkUpdatedEvent
        await plugin.handle_packet(
            _make_packet({"name": "speaker", "volume": 5000}),
            "reentrant_dev",
            MagicMock(),
        )

        assert "sinks_count:1" in reentrant_logs
        assert "after_evict:0" in reentrant_logs
        assert plugin.get_sinks("reentrant_dev") == []

        await plugin.stop()

    async def test_repeated_lifecycle_churn_no_resource_leaks(self) -> None:
        """10 rapid connect -> update -> disconnect -> unpair cycles across 10 devices."""
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        devices = [f"lifecycle_dev_{i}" for i in range(10)]

        for cycle in range(5):
            # Connect & populate
            for dev in devices:
                mock_send = MagicMock()
                await plugin.handle_packet(
                    _make_packet(
                        {
                            "sinkList": [
                                {
                                    "name": "spk",
                                    "volume": cycle * 1000,
                                    "maxVolume": 65536,
                                }
                            ]
                        }
                    ),
                    dev,
                    mock_send,
                )
                assert dev in plugin._connected_peers
                assert len(plugin.get_sinks(dev)) == 1

            # Disconnect: clears peers, retains cache
            for dev in devices:
                await bus.publish(DeviceDisconnected(device_id=dev))
                assert dev not in plugin._connected_peers
                assert len(plugin.get_sinks(dev)) == 1

            # Unpair: clears cache
            for dev in devices:
                await bus.publish(DeviceUnpaired(device_id=dev))
                assert len(plugin.get_sinks(dev)) == 0

        assert len(plugin._connected_peers) == 0
        assert len(plugin._device_sinks) == 0

        await plugin.stop()

    async def test_concurrent_duplicate_unpair_events_idempotency(self) -> None:
        """Multiple simultaneous DeviceUnpaired and DeviceUnpairedEvent triggers."""
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        await plugin.handle_packet(
            _make_packet({"sinkList": [{"name": "sink1", "volume": 1000}]}),
            "dup_unpair_dev",
            MagicMock(),
        )

        # Emit 20 unpair events in parallel
        events = [
            bus.publish(DeviceUnpaired(device_id="dup_unpair_dev")) for _ in range(10)
        ] + [
            bus.publish(DeviceUnpairedEvent(device_id="dup_unpair_dev"))
            for _ in range(10)
        ]
        await asyncio.gather(*events)

        assert plugin.get_sinks("dup_unpair_dev") == []
        assert "dup_unpair_dev" not in plugin._device_sinks

        await plugin.stop()


# =============================================================================
# 4. Outbound Request Error Handling & Socket Resilience
# =============================================================================


class TestOutboundRequestErrorHandlingAndSocketResilience:
    """Tests evaluating resilience against socket errors and dead transports."""

    async def test_connection_reset_error_propagates_and_preserves_state(
        self,
    ) -> None:
        """ConnectionResetError on set_remote_volume propagates to caller without corrupting cache."""
        plugin = RemoteSystemVolumePlugin()
        dev_id = "phone_dead_sock"

        # Seed sink
        await plugin.handle_packet(
            _make_packet(
                {
                    "sinkList": [
                        {
                            "name": "speaker",
                            "volume": 30000,
                            "maxVolume": 65536,
                        }
                    ]
                }
            ),
            dev_id,
            MagicMock(),
        )

        async def dead_send(pkt: Packet) -> None:
            raise ConnectionResetError("Connection reset by peer")

        # Attempt volume adjustment
        with pytest.raises(ConnectionResetError) as exc_info:
            await plugin.set_remote_volume(
                dev_id, "speaker", 45000, send_packet=dead_send
            )
        assert "Connection reset by peer" in str(exc_info.value)

        # Cache must remain consistent with last known state
        sink = plugin.get_sink(dev_id, "speaker")
        assert sink is not None
        assert sink.volume == 30000

    async def test_broken_pipe_error_on_outbound_controls(self) -> None:
        """BrokenPipeError on set_remote_muted propagates cleanly."""
        plugin = RemoteSystemVolumePlugin()
        dev_id = "phone_broken_pipe"

        async def broken_pipe_send(pkt: Packet) -> None:
            raise BrokenPipeError("Broken pipe")

        with pytest.raises(BrokenPipeError):
            await plugin.set_remote_muted(
                dev_id, "speaker", True, send_packet=broken_pipe_send
            )

        with pytest.raises(BrokenPipeError):
            await plugin.set_remote_default(
                dev_id, "speaker", send_packet=broken_pipe_send
            )

        with pytest.raises(BrokenPipeError):
            await plugin.request_remote_sinks(dev_id, send_packet=broken_pipe_send)

    async def test_concurrent_outbound_partial_socket_failures(self) -> None:
        """Outbound calls across 20 devices where half have dead sockets.

        Healthy devices must succeed; dead devices fail without cross-impact.
        """
        plugin = RemoteSystemVolumePlugin()
        num_devices = 20
        device_ids = [f"socket_dev_{i:02d}" for i in range(num_devices)]

        healthy_sent: list[tuple[str, Packet]] = []

        def make_sender(idx: int, dev_id: str) -> SendPacketCallable:
            if idx % 2 == 0:
                # Healthy sender
                async def ok_send(pkt: Packet) -> None:
                    healthy_sent.append((dev_id, pkt))

                return ok_send
            else:
                # Broken sender
                async def broken_send(pkt: Packet) -> None:
                    raise OSError(104, f"Connection reset by peer on {dev_id}")

                return broken_send

        # Seed devices with their respective senders
        for idx, dev_id in enumerate(device_ids):
            sender = make_sender(idx, dev_id)
            await plugin.handle_packet(
                _make_packet(
                    {"sinkList": [{"name": "speaker", "volume": 1000 * (idx + 1)}]}
                ),
                dev_id,
                send_packet=sender,
            )

        # Concurrently dispatch set_remote_volume to all 20 devices
        async def call_volume(dev_id: str) -> None:
            await plugin.set_remote_volume(dev_id, "speaker", 50000)

        tasks = [call_volume(dev_id) for dev_id in device_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Assert: 10 even devices succeeded, 10 odd devices raised OSError
        for idx, (dev_id, res) in enumerate(zip(device_ids, results)):
            if idx % 2 == 0:
                assert res is None, f"Expected success for {dev_id}, got {res}"
            else:
                assert isinstance(res, OSError), (
                    f"Expected OSError for {dev_id}, got {res}"
                )

        assert len(healthy_sent) == 10

    async def test_outbound_to_evicted_device_raises_pengulink_error(
        self,
    ) -> None:
        """Calling outbound control on device after eviction raises PenguLinkError."""
        plugin = RemoteSystemVolumePlugin()
        dev_id = "phone_to_evict"

        await plugin.handle_packet(
            _make_packet({"sinkList": [{"name": "speaker", "volume": 5000}]}),
            dev_id,
            send_packet=MagicMock(),
        )

        # Evict
        plugin.evict_device(dev_id)

        # Outbound call without explicit sender should raise PenguLinkError
        with pytest.raises(PenguLinkError) as exc_info:
            await plugin.set_remote_volume(dev_id, "speaker", 10000)
        assert f"Cannot send packet to {dev_id}: no active connection" in str(
            exc_info.value
        )

    async def test_synchronous_sender_raising_exception_handled_identically(
        self,
    ) -> None:
        """Synchronous send_packet raising exception propagates identically to async."""
        plugin = RemoteSystemVolumePlugin()

        def sync_raising_sender(pkt: Packet) -> None:
            raise ConnectionAbortedError("Software caused connection abort")

        with pytest.raises(ConnectionAbortedError):
            await plugin.set_remote_volume(
                "dev_sync", "speaker", 12345, send_packet=sync_raising_sender
            )

    async def test_packet_sender_fallback_dispatch(self) -> None:
        """Global packet_sender fallback behaves correctly when peer sender is absent."""
        sent_packets: list[tuple[str, Packet]] = []

        # packet_sender taking (device_id, packet)
        async def global_sender(target_dev: str, pkt: Packet) -> None:
            sent_packets.append((target_dev, pkt))

        plugin = RemoteSystemVolumePlugin(packet_sender=global_sender)

        await plugin.request_remote_sinks("remote_peer_xyz")
        assert len(sent_packets) == 1
        assert sent_packets[0][0] == "remote_peer_xyz"
        assert sent_packets[0][1].body.get("requestSinks") is True


# =============================================================================
# 5. Chaotic Multi-Device Stress Matrix
# =============================================================================


class TestChaoticMultiDeviceStressMatrix:
    """Chaotic high-concurrency barrage mixing valid, invalid, socket crashes, and unpairs."""

    async def test_chaotic_multidevice_battery(self) -> None:
        """25 devices under continuous chaos: rapid packets, random unpairs, queries."""
        bus = EventBus()
        paired_map: dict[str, bool] = {f"device_{i:02d}": True for i in range(25)}
        # Mark odd devices above 20 as unpaired
        paired_map["device_21"] = False
        paired_map["device_23"] = False

        plugin = RemoteSystemVolumePlugin(
            event_bus=bus,
            is_paired_checker=lambda dev: paired_map.get(dev, False),
        )
        await plugin.start()

        num_devices = 25
        device_ids = [f"device_{i:02d}" for i in range(num_devices)]

        async def device_chaos_worker(dev_id: str) -> None:
            for op in range(25):
                choice = op % 5
                if choice == 0:
                    # Valid sinkList
                    pkt = _make_packet(
                        {
                            "sinkList": [
                                {
                                    "name": "spk",
                                    "volume": random.randint(0, 65536),
                                    "maxVolume": 65536,
                                },
                                {
                                    "name": "bt",
                                    "volume": random.randint(0, 100),
                                    "maxVolume": 100,
                                },
                            ]
                        }
                    )
                    await plugin.handle_packet(pkt, dev_id, MagicMock())
                elif choice == 1:
                    # Single mutation
                    pkt = _make_packet(
                        {
                            "name": "spk",
                            "volume": random.randint(0, 65536),
                            "muted": bool(op % 2),
                        }
                    )
                    await plugin.handle_packet(pkt, dev_id, MagicMock())
                elif choice == 2:
                    # Fuzzed/dirty packet
                    dirty_pkt = _make_packet(
                        {
                            "sinkList": [
                                {"name": "valid", "volume": "45000"},
                                {"name": "", "volume": -100},  # empty name
                                None,  # non-dict
                                {"volume": 1234},  # missing name
                                {
                                    "name": "nan_sink",
                                    "volume": "nan",
                                    "maxVolume": "inf",
                                },
                            ]
                        }
                    )
                    await plugin.handle_packet(dirty_pkt, dev_id, MagicMock())
                elif choice == 3:
                    # Query state
                    _ = plugin.get_sinks(dev_id)
                    _ = plugin.get_sink(dev_id, "spk")
                    _ = plugin.get_sink(dev_id, "unknown")
                elif choice == 4:
                    # Lifecycle event
                    if op % 2 == 0:
                        await bus.publish(DeviceDisconnected(device_id=dev_id))
                    else:
                        await bus.publish(DeviceUnpaired(device_id=dev_id))
                await asyncio.sleep(0.001)

        tasks = [device_chaos_worker(dev_id) for dev_id in device_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                raise r

        # Verify that unpaired devices have zero sinks
        assert plugin.get_sinks("device_21") == []
        assert plugin.get_sinks("device_23") == []

        await plugin.stop()

    async def test_fifty_concurrent_devices_stress(self) -> None:
        """50 concurrent devices connected simultaneously with identical sink names."""
        plugin = RemoteSystemVolumePlugin()
        num_devices = 50
        device_ids = [f"node_{i:03d}" for i in range(num_devices)]

        async def setup_node(idx: int, dev_id: str) -> None:
            pkt = _make_packet(
                {
                    "sinkList": [
                        {
                            "name": "master_output",
                            "volume": idx * 1000,
                            "maxVolume": 65536,
                            "muted": (idx % 2 == 0),
                        }
                    ]
                }
            )
            await plugin.handle_packet(pkt, dev_id, MagicMock())

        await asyncio.gather(
            *[setup_node(i, dev_id) for i, dev_id in enumerate(device_ids)]
        )

        assert len(plugin._device_sinks) == 50
        for idx, dev_id in enumerate(device_ids):
            sink = plugin.get_sink(dev_id, "master_output")
            assert sink is not None
            assert sink.volume == idx * 1000
            assert sink.muted is (idx % 2 == 0)
            assert sink.device_id == dev_id

    async def test_timeout_and_cancelled_error_on_outbound_controls(self) -> None:
        """asyncio.TimeoutError and asyncio.CancelledError propagate without corrupting state."""
        plugin = RemoteSystemVolumePlugin()
        dev_id = "timeout_dev"

        await plugin.handle_packet(
            _make_packet({"sinkList": [{"name": "speaker", "volume": 12000}]}),
            dev_id,
            MagicMock(),
        )

        async def timing_out_sender(pkt: Packet) -> None:
            raise TimeoutError("Send timed out")

        with pytest.raises(TimeoutError):
            await plugin.set_remote_volume(
                dev_id, "speaker", 50000, send_packet=timing_out_sender
            )

        assert plugin.get_sink(dev_id, "speaker").volume == 12000

        async def cancelling_sender(pkt: Packet) -> None:
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await plugin.set_remote_volume(
                dev_id, "speaker", 50000, send_packet=cancelling_sender
            )

        assert plugin.get_sink(dev_id, "speaker").volume == 12000

    async def test_stop_during_active_concurrent_traffic(self) -> None:
        """Calling plugin.stop() while packets were recently in flight cleans up properly."""
        bus = EventBus()
        plugin = RemoteSystemVolumePlugin(event_bus=bus)
        await plugin.start()

        devices = [f"stop_test_dev_{i}" for i in range(10)]
        stopping = asyncio.Event()

        async def traffic_worker() -> None:
            for i in range(20):
                if stopping.is_set():
                    break
                dev = random.choice(devices)
                pkt = _make_packet({"sinkList": [{"name": "spk", "volume": i * 100}]})
                await plugin.handle_packet(pkt, dev, MagicMock())
                await asyncio.sleep(0.001)

        traffic_task = asyncio.create_task(traffic_worker())
        await asyncio.sleep(0.005)
        # Halt traffic and stop plugin
        stopping.set()
        await traffic_task
        await plugin.stop()

        assert len(plugin._unsub_callbacks) == 0
        assert len(plugin._connected_peers) == 0
        assert len(plugin._device_sinks) == 0

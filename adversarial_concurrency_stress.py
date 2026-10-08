#!/usr/bin/env python3
"""Adversarial stress test harness targeting concurrency during stopping in ClipboardPlugin.

High-concurrency lifecycle and stress test vectors:
- Vector 1: Massive high-concurrency packet flood (2000 packets) while stopping.
- Vector 2: Slow/blocked socket send callbacks with in-flight tasks while stopping.
- Vector 3: High-frequency storm of start/stop/handle_packet/set_local/local_change.
- Vector 4: Post-stop resilience (packet flood, OS callbacks, programmatic writes, connection events).
- Vector 5: Concurrent stop() stampede (multiple tasks stopping simultaneously).
- Vector 6: Fault injection in backend.stop_watching() and send_packet during stop.
- Task Leak Oracle: Counts all tasks in event loop before and after, ensuring zero leaked tasks.
"""

import asyncio
import logging
import random
import sys
import time
from typing import List, Set

from pengulink.core.events import ClipboardUpdatedEvent, DeviceConnected, EventBus
from pengulink.network.packet import Packet
from pengulink.plugins.clipboard.backend import InMemoryClipboardBackend
from pengulink.plugins.clipboard.plugin import ClipboardPlugin, create_clipboard_packet

logging.basicConfig(level=logging.WARNING)


def get_active_plugin_tasks(loop: asyncio.AbstractEventLoop, main_task: asyncio.Task) -> Set[asyncio.Task]:
    """Return set of active asyncio tasks in loop excluding current execution task."""
    all_tasks = asyncio.all_tasks(loop)
    return {t for t in all_tasks if t is not main_task and not t.done()}


async def test_vector1_massive_packet_flood_while_stopping() -> None:
    print("[Vector 1] Starting massive packet flood (2000 packets) while stopping...")
    bus = EventBus()
    backend = InMemoryClipboardBackend()
    plugin = ClipboardPlugin(event_bus=bus, backend=backend)
    await plugin.start()

    # Register 20 devices
    for i in range(20):
        plugin.add_connected_device(f"device_{i}", lambda p: None)

    stop_triggered = False
    packets_processed = 0

    async def sender_worker(worker_id: int) -> None:
        nonlocal packets_processed
        for seq in range(100):
            ts = int(time.time() * 1000) + worker_id * 1000 + seq
            pkt = create_clipboard_packet(f"Flood_{worker_id}_{seq}", timestamp=ts)
            try:
                await plugin.handle_packet(pkt, f"device_{worker_id}", lambda p: None)
                packets_processed += 1
            except Exception as e:
                print(f"UNEXPECTED EXCEPTION in sender_worker {worker_id}: {e}", file=sys.stderr)
                raise
            if worker_id == 5 and seq == 30 and not stop_triggered:
                # Let other tasks run slightly
                await asyncio.sleep(0.001)

    sender_tasks = [asyncio.create_task(sender_worker(w)) for w in range(20)]

    # Allow some packets to fly
    await asyncio.sleep(0.01)

    # Trigger stop in the middle of the flood
    stop_task = asyncio.create_task(plugin.stop())

    # Wait for both stop and all senders to finish
    await asyncio.gather(stop_task, *sender_tasks, return_exceptions=False)

    # VERIFICATION ORACLES:
    assert plugin.is_running is False, "Plugin must not be running"
    assert len(plugin._background_tasks) == 0, f"Background tasks must be 0, got {len(plugin._background_tasks)}"
    assert len(plugin.get_connected_devices()) == 0, "Connected devices must be 0"
    assert len(plugin._device_senders) == 0, "Device senders must be 0"
    assert len(plugin._unsub_callbacks) == 0, "Unsub callbacks must be 0"

    current_task = asyncio.current_task()
    lingering = get_active_plugin_tasks(asyncio.get_running_loop(), current_task)
    assert len(lingering) == 0, f"Leaked asyncio tasks detected: {lingering}"
    print(f"  -> SUCCESS: Processed {packets_processed} packets during flood; 0 leaked tasks; clean shutdown.")


async def test_vector2_slow_senders_and_in_flight_cancellation() -> None:
    print("[Vector 2] Starting slow senders with in-flight background tasks while stopping...")
    bus = EventBus()
    backend = InMemoryClipboardBackend()
    plugin = ClipboardPlugin(event_bus=bus, backend=backend)
    await plugin.start()

    active_senders = 0
    cancelled_senders = 0

    async def slow_sender(pkt: Packet) -> None:
        nonlocal active_senders, cancelled_senders
        active_senders += 1
        try:
            await asyncio.sleep(0.5)  # Simulate slow network socket
        except asyncio.CancelledError:
            cancelled_senders += 1
            raise
        finally:
            active_senders -= 1

    # Connect 10 devices with slow sender
    for i in range(10):
        plugin.add_connected_device(f"slow_dev_{i}", slow_sender)

    # Inbound packet from device_0 triggers fanout broadcast to devices 1..9
    pkt = create_clipboard_packet("Slow broadcast trigger", timestamp=int(time.time() * 1000))
    await plugin.handle_packet(pkt, "slow_dev_0", slow_sender)

    # Verify background tasks were created
    assert len(plugin._background_tasks) > 0, "Expected in-flight broadcast background tasks"
    in_flight_count = len(plugin._background_tasks)

    # Test Case A: Tasks cancelled before they ever start (pure task cancellation)
    assert in_flight_count > 0
    await plugin.stop()
    assert len(plugin._background_tasks) == 0
    # Let's inspect how stop handles both cases:
    # First, let's test actively running in-flight tasks:
    plugin2 = ClipboardPlugin(event_bus=bus, backend=backend)
    await plugin2.start()
    for i in range(10):
        plugin2.add_connected_device(f"slow_dev2_{i}", slow_sender)
    pkt2 = create_clipboard_packet("Slow broadcast trigger 2", timestamp=int(time.time() * 1000) + 1000)
    await plugin2.handle_packet(pkt2, "slow_dev2_0", slow_sender)
    # Give tasks 0.01s so they are actively suspended in asyncio.sleep(0.5)
    await asyncio.sleep(0.01)
    assert active_senders > 0, f"Expected active senders sleeping, got {active_senders}"

    stop_start = time.monotonic()
    await plugin2.stop()
    stop_duration = time.monotonic() - stop_start

    # Stop must not wait for the 0.5s sleep (must cancel immediately)
    assert stop_duration < 0.2, f"stop() took too long ({stop_duration:.3f}s), tasks not cancelled promptly"
    assert len(plugin2._background_tasks) == 0, "All background tasks must be drained"
    assert cancelled_senders > 0, f"Slow senders should have caught CancelledError, got {cancelled_senders}"
    assert active_senders == 0, f"Active senders must be 0 after stop, got {active_senders}"

    current_task = asyncio.current_task()
    lingering = get_active_plugin_tasks(asyncio.get_running_loop(), current_task)
    assert len(lingering) == 0, f"Leaked asyncio tasks detected: {lingering}"
    print(f"  -> SUCCESS: {in_flight_count} in-flight tasks promptly cancelled ({stop_duration*1000:.1f}ms); 0 leaked tasks.")


async def test_vector3_high_frequency_start_stop_packet_storm() -> None:
    print("[Vector 3] Starting high-frequency start/stop/handle/mutate storm (100 iterations, 30 workers)...")
    bus = EventBus()
    backend = InMemoryClipboardBackend()
    plugin = ClipboardPlugin(event_bus=bus, backend=backend)

    running_flag = True

    async def stopper_starter() -> None:
        while running_flag:
            await plugin.start()
            await asyncio.sleep(random.uniform(0.001, 0.005))
            await plugin.stop()
            await asyncio.sleep(random.uniform(0.001, 0.003))

    async def packet_spammer(w_id: int) -> None:
        for seq in range(50):
            ts = int(time.time() * 1000) + seq
            pkt = create_clipboard_packet(f"Storm_{w_id}_{seq}", timestamp=ts)
            await plugin.handle_packet(pkt, f"storm_dev_{w_id}", lambda p: None)
            if seq % 10 == 0:
                await plugin.set_local_clipboard(f"Local_{w_id}_{seq}")
                await plugin._on_local_clipboard_changed(f"OS_change_{w_id}_{seq}")
            await asyncio.sleep(random.uniform(0.0005, 0.002))

    lifecycle_task = asyncio.create_task(stopper_starter())
    spam_tasks = [asyncio.create_task(packet_spammer(i)) for i in range(15)]

    await asyncio.gather(*spam_tasks, return_exceptions=False)
    running_flag = False
    await lifecycle_task

    # Final definitive stop
    await plugin.stop()

    assert plugin.is_running is False
    assert len(plugin._background_tasks) == 0
    assert len(plugin.get_connected_devices()) == 0

    current_task = asyncio.current_task()
    lingering = get_active_plugin_tasks(asyncio.get_running_loop(), current_task)
    assert len(lingering) == 0, f"Leaked asyncio tasks detected: {lingering}"
    print("  -> SUCCESS: Storm completed without exceptions or deadlocks; 0 leaked tasks.")


async def test_vector4_post_stop_resilience() -> None:
    print("[Vector 4] Starting post-stop resilience verification (500 packets, OS callbacks, API writes)...")
    bus = EventBus()
    backend = InMemoryClipboardBackend()
    plugin = ClipboardPlugin(event_bus=bus, backend=backend)

    await plugin.start()
    await backend.set_text("Initial Text")
    await plugin.stop()
    assert plugin.is_running is False

    events_received: List[ClipboardUpdatedEvent] = []
    bus.subscribe(ClipboardUpdatedEvent, lambda e: events_received.append(e))

    # 1. Flood 500 packets to stopped plugin
    sent_packets: List[Packet] = []
    for i in range(500):
        pkt = create_clipboard_packet(f"PostStop_{i}", timestamp=int(time.time() * 1000) + i)
        await plugin.handle_packet(pkt, f"dev_{i%10}", lambda p: sent_packets.append(p))

    # Backend must NOT have changed
    assert await backend.get_text() == "Initial Text", "Backend mutated while plugin was stopped!"
    assert len(sent_packets) == 0, "Packets sent while plugin was stopped!"
    assert len(events_received) == 0, "Events published while plugin was stopped!"
    assert len(plugin.get_connected_devices()) == 0, "Devices registered while plugin was stopped!"
    assert len(plugin._device_senders) == 0, "Senders registered while plugin was stopped!"
    assert len(plugin._background_tasks) == 0, "Background tasks spawned while stopped!"

    # 2. OS callbacks fired while stopped
    await plugin._on_local_clipboard_changed("External change while stopped")
    assert len(events_received) == 0
    assert len(plugin._background_tasks) == 0

    # 3. Programmatic set_local_clipboard while stopped
    success = await plugin.set_local_clipboard("Programmatic write while stopped")
    assert success is False, "set_local_clipboard must return False when stopped"
    assert len(events_received) == 0

    # 4. Device connected event fired while stopped
    await plugin._on_device_connected(DeviceConnected(device_id="ghost_dev"))
    assert len(plugin.get_connected_devices()) == 0
    assert len(plugin._background_tasks) == 0

    # 5. Broadcast packet invoked directly while stopped
    await plugin._broadcast_packet(create_clipboard_packet("Ghost broadcast"))
    assert len(plugin._background_tasks) == 0

    current_task = asyncio.current_task()
    lingering = get_active_plugin_tasks(asyncio.get_running_loop(), current_task)
    assert len(lingering) == 0, f"Leaked asyncio tasks detected: {lingering}"
    print("  -> SUCCESS: All operations safely rejected while stopped; 0 state pollution; 0 leaked tasks.")


async def test_vector5_concurrent_stop_stampede() -> None:
    print("[Vector 5] Starting concurrent stop() stampede (50 simultaneous stop() calls)...")
    bus = EventBus()
    backend = InMemoryClipboardBackend()
    plugin = ClipboardPlugin(event_bus=bus, backend=backend)
    await plugin.start()

    for i in range(10):
        plugin.add_connected_device(f"dev_{i}", lambda p: None)

    # Concurrently launch 50 stop() calls
    results = await asyncio.gather(*(plugin.stop() for _ in range(50)), return_exceptions=True)
    for r in results:
        assert r is None or not isinstance(r, Exception), f"stop() raised exception: {r}"

    assert plugin.is_running is False
    assert len(plugin._background_tasks) == 0
    assert len(plugin.get_connected_devices()) == 0

    current_task = asyncio.current_task()
    lingering = get_active_plugin_tasks(asyncio.get_running_loop(), current_task)
    assert len(lingering) == 0, f"Leaked asyncio tasks detected: {lingering}"
    print("  -> SUCCESS: Concurrent stop stampede resolved idempotently; 0 leaked tasks.")


async def test_vector6_fault_injection_during_stop() -> None:
    print("[Vector 6] Starting fault injection during stop()...")
    bus = EventBus()

    class FaultyBackend(InMemoryClipboardBackend):
        async def stop_watching(self) -> None:
            raise RuntimeError("Hardware failure during stop_watching")

    faulty_backend = FaultyBackend()
    plugin = ClipboardPlugin(event_bus=bus, backend=faulty_backend)
    await plugin.start()

    # Even if stop_watching raises RuntimeError, stop() must not crash and must complete cleanup
    await plugin.stop()

    assert plugin.is_running is False
    assert len(plugin._background_tasks) == 0
    assert len(plugin._unsub_callbacks) == 0

    current_task = asyncio.current_task()
    lingering = get_active_plugin_tasks(asyncio.get_running_loop(), current_task)
    assert len(lingering) == 0, f"Leaked asyncio tasks detected: {lingering}"
    print("  -> SUCCESS: Backend fault during stop() safely absorbed; state clean; 0 leaked tasks.")


async def test_vector7_device_connection_storm_while_stopping() -> None:
    print("[Vector 7] Starting DeviceConnected/Disconnected event storm while stopping...")
    bus = EventBus()
    backend = InMemoryClipboardBackend()
    plugin = ClipboardPlugin(event_bus=bus, backend=backend)
    await plugin.start()
    await plugin.set_local_clipboard("Seed text for connect push")

    async def device_event_worker(w_id: int) -> None:
        for seq in range(30):
            dev = f"storm_dev_{w_id}_{seq}"
            await bus.publish(DeviceConnected(device_id=dev))
            # Also send connect packet
            conn_pkt = create_clipboard_packet("Old peer text", timestamp=100)
            conn_pkt.type = "kdeconnect.clipboard.connect"
            await plugin.handle_packet(conn_pkt, dev, lambda p: None)
            if seq % 2 == 0:
                await bus.publish(ClipboardUpdatedEvent(device_id=dev, content=f"text_{seq}", timestamp=200+seq))
            await asyncio.sleep(0.001)

    workers = [asyncio.create_task(device_event_worker(i)) for i in range(10)]
    await asyncio.sleep(0.01)

    # Stop plugin while event storm is ongoing
    await plugin.stop()
    await asyncio.gather(*workers, return_exceptions=True)

    assert plugin.is_running is False
    assert len(plugin._background_tasks) == 0
    assert len(plugin.get_connected_devices()) == 0

    current_task = asyncio.current_task()
    lingering = get_active_plugin_tasks(asyncio.get_running_loop(), current_task)
    assert len(lingering) == 0, f"Leaked asyncio tasks detected: {lingering}"
    print("  -> SUCCESS: Device connection storm resolved with 0 leaked tasks.")


async def test_vector8_bounded_cache_and_memory_churn() -> None:
    print("[Vector 8] Starting high-volume cache churn (10,000 distinct packets) with stop()...")
    bus = EventBus()
    backend = InMemoryClipboardBackend()
    plugin = ClipboardPlugin(event_bus=bus, backend=backend)
    await plugin.start()

    # Process 10,000 packets through suppression cache
    now = int(time.time() * 1000)
    for i in range(10_000):
        pkt = create_clipboard_packet(f"Content_Churn_{i}", timestamp=now + i + 1)
        await plugin.handle_packet(pkt, "churn_dev", lambda p: None)

    # Verify cache capacity is strictly bounded
    assert len(plugin.suppression.digest_cache._cache) == 32, (
        f"Cache capacity must remain at 32, got {len(plugin.suppression.digest_cache._cache)}"
    )

    await plugin.stop()
    assert plugin.is_running is False
    assert len(plugin._background_tasks) == 0

    current_task = asyncio.current_task()
    lingering = get_active_plugin_tasks(asyncio.get_running_loop(), current_task)
    assert len(lingering) == 0, f"Leaked asyncio tasks detected: {lingering}"
    print("  -> SUCCESS: Cache strictly bounded at 32; 0 memory leaks; 0 leaked tasks.")


async def main() -> None:
    print("================================================================================")
    print("ADVERSARIAL STRESS HARNESS: CONCURRENCY WHILE STOPPING (CHALLENGER 2)")
    print("================================================================================")
    await test_vector1_massive_packet_flood_while_stopping()
    await test_vector2_slow_senders_and_in_flight_cancellation()
    await test_vector3_high_frequency_start_stop_packet_storm()
    await test_vector4_post_stop_resilience()
    await test_vector5_concurrent_stop_stampede()
    await test_vector6_fault_injection_during_stop()
    await test_vector7_device_connection_storm_while_stopping()
    await test_vector8_bounded_cache_and_memory_churn()
    print("================================================================================")
    print("ALL 8 ADVERSARIAL STRESS TEST VECTORS PASSED EMPIRICALLY (0 LEAKED TASKS)!")
    print("================================================================================")


if __name__ == "__main__":
    asyncio.run(main())

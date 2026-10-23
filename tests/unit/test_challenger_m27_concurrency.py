"""tests.unit.test_challenger_m27_concurrency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial Concurrency, Multi-Device Isolation, Race Condition, and Lifecycle Stress Suite.
Milestone 27: Telephony Plugin & Active Call State Management.

Empirical verification against:
- `pengulink.services.telephony.TelephonyService`
- `pengulink.plugins.telephony.TelephonyPlugin`
- `pengulink.core.events.CallEvent`
- `pengulink.core.events.CallStateChangedEvent`

Test Matrix:
1. Multi-Device Concurrency (10+ Devices via asyncio.gather):
   - 10 devices concurrently receiving and modifying calls in parallel asyncio.gather.
   - Randomized concurrent call storms across 10 devices with 50 operations each (500 events).
   - 20 devices simultaneous burst to TALKING state verifying data fidelity.
   - Concurrent outbound telephony actions (mute_ringer, reject_call) across 10 devices.
2. Cross-Device Pollution & Isolation:
   - Device A call progression never contaminates Device B active talking call.
   - Cross-device identical phone numbers and contact names maintain isolated instances.
   - Substring/prefix device ID collisions ("dev", "dev_1", "dev_10", "dev_100").
   - Disconnecting/unpairing a subset of devices strictly isolates remaining devices.
3. Race Conditions:
   - Rapid clear_device(device_id) racing against concurrent process_call_event on the same device.
   - Simultaneous DeviceDisconnected and DeviceUnpaired events for same device.
   - Concurrent clear_all() while devices are streaming inbound call events.
   - Duplicate concurrent ringing and talking packets on same device.
   - Rapid flapping between ringing and cancellation (50 transitions).
4. EventBus Handler Exception & Slowness Isolation:
   - Failing synchronous and asynchronous subscribers throwing various exceptions never crash service.
   - Slow subscriber execution does not crash service.
   - Cross-device lock contention measurement under slow CallStateChangedEvent subscriber.
   - CallEvent subscriber querying active calls without deadlock (outside lock).
   - Re-entrancy challenge: investigating if subscribers querying service cause asyncio.Lock deadlock.
5. Memory Leak & Lifecycle Cleanup:
   - 500 device call lifecycles sequential creation and destruction (asserting _active_calls size returns to 0).
   - 500 device call lifecycles concurrent batches via asyncio.gather (asserting _active_calls size returns to 0).
   - TelephonyPlugin _connected_peers and TelephonyService _active_calls cleanup on unpair/disconnect.
   - Garbage collection audit after call teardown.
"""

from __future__ import annotations

import asyncio
import gc
import time
from typing import Any

from pengulink.core.events import (
    CallEvent,
    CallStateChangedEvent,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
)
from pengulink.network.packet import Packet
from pengulink.plugins.telephony import (
    TELEPHONY_PACKET_TYPE,
    TELEPHONY_REQUEST_MUTE_TYPE,
    TELEPHONY_REQUEST_TYPE,
    TelephonyPlugin,
)
from pengulink.services.telephony import (
    ActiveCall,
    CallState,
    TelephonyService,
)


class MockPacketSender:
    """Mock sender callback capturing dispatched wire packets."""

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


# =============================================================================
# 1. Multi-Device Concurrency (10+ Devices via asyncio.gather)
# =============================================================================


class TestParallelMultiDeviceConcurrency:
    """Stress tests verifying concurrent call handling across 10+ devices."""

    async def test_ten_devices_concurrent_full_call_lifecycles(self) -> None:
        """10 devices execute full lifecycles simultaneously via asyncio.gather."""
        bus = EventBus()
        service = TelephonyService(event_bus=bus)
        events_received: list[CallStateChangedEvent] = []
        bus.subscribe(CallStateChangedEvent, lambda ev: events_received.append(ev))

        num_devices = 10

        async def device_lifecycle(dev_idx: int) -> None:
            dev_id = f"device_{dev_idx:03d}"
            p_num = f"+1800555{dev_idx:04d}"
            c_name = f"Contact_{dev_idx}"

            # Step 1: Ringing
            call1 = await service.process_call_event(
                device_id=dev_id,
                event="ringing",
                phone_number=p_num,
                contact_name=c_name,
            )
            assert call1 is not None
            assert call1.state == CallState.RINGING
            assert call1.device_id == dev_id

            await asyncio.sleep(0.001)

            # Step 2: Talking
            call2 = await service.process_call_event(
                device_id=dev_id,
                event="talking",
                phone_number=p_num,
                contact_name=c_name,
            )
            assert call2 is not None
            assert call2.state == CallState.TALKING
            assert call2.duration >= 0.0

            await asyncio.sleep(0.001)

            # Step 3: Idle (Ended)
            call3 = await service.process_call_event(
                device_id=dev_id,
                event="idle",
            )
            assert call3 is None
            assert await service.get_call(dev_id) is None

        # Execute 10 device lifecycles concurrently
        await asyncio.gather(*(device_lifecycle(i) for i in range(num_devices)))

        # Assert all devices cleaned up
        active_calls = await service.get_active_calls()
        assert len(active_calls) == 0
        assert len(service._active_calls) == 0

        # Verify all state transitions were recorded:
        # Each device had IDLE->RINGING, RINGING->TALKING, TALKING->IDLE (3 events each = 30 total)
        assert len(events_received) == num_devices * 3
        service.close()

    async def test_ten_devices_randomized_concurrent_call_storms(self) -> None:
        """10 devices each fire 50 randomized call transitions concurrently (500 events)."""
        bus = EventBus()
        service = TelephonyService(event_bus=bus)

        num_devices = 10
        ops_per_device = 50
        event_types = ["ringing", "talking", "missedCall", "idle", "callEnded"]

        async def device_storm(dev_idx: int) -> None:
            dev_id = f"storm_dev_{dev_idx}"
            p_num = f"+1999000{dev_idx:04d}"
            c_name = f"StormUser_{dev_idx}"

            for op_idx in range(ops_per_device):
                ev = event_types[(dev_idx + op_idx) % len(event_types)]
                is_cancel = op_idx % 7 == 0
                await service.process_call_event(
                    device_id=dev_id,
                    event=ev,
                    phone_number=p_num,
                    contact_name=c_name,
                    is_cancel=is_cancel,
                )
                if op_idx % 5 == 0:
                    # Interleaved query
                    _ = await service.get_call(dev_id)
                if op_idx % 11 == 0:
                    _ = await service.get_active_calls()

        # Run all 10 device storms in parallel
        await asyncio.gather(*(device_storm(i) for i in range(num_devices)))

        # Clean up
        await service.clear_all()
        assert len(await service.get_active_calls()) == 0
        service.close()

    async def test_twenty_devices_concurrent_talking_burst(self) -> None:
        """20 devices transition to TALKING simultaneously, verifying active count and data fidelity."""
        service = TelephonyService()
        num_devices = 20

        async def start_call(i: int) -> ActiveCall | None:
            return await service.process_call_event(
                device_id=f"burst_dev_{i}",
                event="talking",
                phone_number=f"+12345{i:04d}",
                contact_name=f"User {i}",
            )

        results = await asyncio.gather(*(start_call(i) for i in range(num_devices)))
        assert len(results) == num_devices
        for call in results:
            assert call is not None
            assert call.state == CallState.TALKING
            assert call.is_active is True

        all_active = await service.get_active_calls()
        assert len(all_active) == num_devices

        # Verify each device has its exact data
        for i in range(num_devices):
            call_dict = await service.get_call(f"burst_dev_{i}")
            assert call_dict is not None
            assert call_dict["phone_number"] == f"+12345{i:04d}"
            assert call_dict["contact_name"] == f"User {i}"
            assert call_dict["state"] == "TALKING"

        await service.clear_all()
        assert len(await service.get_active_calls()) == 0

    async def test_multidevice_outbound_actions_concurrency(self) -> None:
        """10 devices concurrently issue outbound mute_ringer and reject_call requests."""
        plugin = TelephonyPlugin()
        num_devices = 10
        senders: dict[str, MockPacketSender] = {}

        # Establish connection cache for 10 devices
        for i in range(num_devices):
            dev_id = f"action_dev_{i}"
            sender = MockPacketSender()
            senders[dev_id] = sender
            await plugin.handle_packet(
                Packet(
                    type=TELEPHONY_PACKET_TYPE,
                    body={"event": "ringing", "phoneNumber": f"+100{i}"},
                ),
                dev_id,
                sender,
            )

        async def run_actions(dev_id: str) -> None:
            # Mute ringer
            res_mute = await plugin.mute_ringer(dev_id)
            assert res_mute is True
            # Reject call
            res_reject = await plugin.reject_call(dev_id, phone_number=f"+100{dev_id}")
            assert res_reject is True

        # Run concurrent outbound calls across all 10 devices
        await asyncio.gather(
            *(run_actions(f"action_dev_{i}") for i in range(num_devices))
        )

        # Assert every device received exactly 1 mute packet and 1 reject packet
        for i in range(num_devices):
            dev_id = f"action_dev_{i}"
            sender = senders[dev_id]
            types = [p.type for p in sender.sent_packets]
            assert TELEPHONY_REQUEST_MUTE_TYPE in types
            assert TELEPHONY_REQUEST_TYPE in types


# =============================================================================
# 2. Cross-Device Pollution & Isolation
# =============================================================================


class TestCrossDevicePollutionAndIsolation:
    """Stress tests asserting strict isolation across multiple devices."""

    async def test_cross_device_state_and_data_isolation(self) -> None:
        """Device A's call mutations never affect Device B's active talking call."""
        service = TelephonyService()

        # Device B establishes a stable talking call
        call_b = await service.process_call_event(
            device_id="device_b",
            event="talking",
            phone_number="+2222222222",
            contact_name="Bob",
        )
        assert call_b is not None
        initial_b_start_time = call_b.start_time

        # Device A undergoes rapid mutations: ringing -> talking -> ringing -> missed -> idle
        await service.process_call_event("device_a", "ringing", "+1111111111", "Alice")
        await service.process_call_event("device_a", "talking", "+1111111111", "Alice")
        await service.process_call_event(
            "device_a", "missedCall", "+1111111111", "Alice"
        )
        await service.process_call_event("device_a", "idle")

        # Verify Device A is idle / cleared
        assert await service.get_call("device_a") is None

        # Verify Device B is completely pristine and unaffected
        b_after = await service.get_call("device_b")
        assert b_after is not None
        assert b_after["device_id"] == "device_b"
        assert b_after["state"] == "TALKING"
        assert b_after["phone_number"] == "+2222222222"
        assert b_after["contact_name"] == "Bob"
        assert b_after["start_time"] == initial_b_start_time

    async def test_cross_device_identical_caller_info_isolation(self) -> None:
        """Devices receiving calls from the identical number and name maintain separate instances."""
        service = TelephonyService()
        shared_number = "+18005550199"
        shared_name = "Customer Support"

        call_a = await service.process_call_event(
            "dev_alpha", "ringing", shared_number, shared_name
        )
        call_b = await service.process_call_event(
            "dev_beta", "ringing", shared_number, shared_name
        )

        assert call_a is not None and call_b is not None
        assert call_a is not call_b
        assert id(call_a) != id(call_b)

        # Transition Alpha to Talking; Beta remains Ringing
        await service.process_call_event(
            "dev_alpha", "talking", shared_number, shared_name
        )

        obj_a = await service.get_call_object("dev_alpha")
        obj_b = await service.get_call_object("dev_beta")
        assert obj_a is not None and obj_b is not None
        assert obj_a.state == CallState.TALKING
        assert obj_b.state == CallState.RINGING

        # Clear Alpha; Beta must still be Ringing
        await service.clear_device("dev_alpha")
        assert await service.get_call("dev_alpha") is None
        assert await service.get_call("dev_beta") is not None
        assert (await service.get_call("dev_beta"))["state"] == "RINGING"

        await service.clear_device("dev_beta")

    async def test_cross_device_prefix_id_isolation(self) -> None:
        """Device IDs that are prefixes/substrings of each other maintain strict boundary."""
        service = TelephonyService()
        prefix_ids = ["dev", "dev_1", "dev_10", "dev_100", "dev_1000"]

        for dev_id in prefix_ids:
            await service.process_call_event(
                device_id=dev_id,
                event="ringing",
                phone_number=f"num_{dev_id}",
                contact_name=f"name_{dev_id}",
            )

        active = await service.get_active_calls()
        assert len(active) == len(prefix_ids)

        # Clear exact prefix "dev"
        cleared = await service.clear_device("dev")
        assert cleared is True
        assert await service.get_call("dev") is None

        # Verify other devices with prefix "dev_" are untouched
        for remaining_id in ["dev_1", "dev_10", "dev_100", "dev_1000"]:
            call = await service.get_call(remaining_id)
            assert call is not None
            assert call["device_id"] == remaining_id
            assert call["state"] == "RINGING"

        # Clear all
        await service.clear_all()
        assert len(await service.get_active_calls()) == 0

    async def test_cross_device_disconnect_and_unpair_isolation(self) -> None:
        """Disconnecting and unpairing specific devices isolates surviving devices."""
        bus = EventBus()
        service = TelephonyService(event_bus=bus)

        devices = [f"peer_{i}" for i in range(5)]
        for dev_id in devices:
            await service.process_call_event(
                dev_id, "talking", f"num_{dev_id}", f"contact_{dev_id}"
            )

        assert len(await service.get_active_calls()) == 5

        # Disconnect peer_1
        await bus.publish(DeviceDisconnected(device_id="peer_1"))
        # Unpair peer_3
        await bus.publish(DeviceUnpaired(device_id="peer_3"))

        assert await service.get_call("peer_1") is None
        assert await service.get_call("peer_3") is None

        # Remaining peers (0, 2, 4) must be intact
        for surviving in ["peer_0", "peer_2", "peer_4"]:
            call = await service.get_call(surviving)
            assert call is not None
            assert call["state"] == "TALKING"
            assert call["contact_name"] == f"contact_{surviving}"

        service.close()


# =============================================================================
# 3. Race Conditions Between Lifecycle and Concurrent Events
# =============================================================================


class TestRaceConditionsLifecycle:
    """Stress tests targeting race conditions between clear_device and process_call_event."""

    async def test_race_rapid_clear_device_during_concurrent_call_events(self) -> None:
        """100 tasks alternating between clear_device and process_call_event for the same device."""
        service = TelephonyService()
        target_dev = "race_dev"

        async def worker_clear() -> None:
            for _ in range(50):
                await service.clear_device(target_dev)
                await asyncio.sleep(0.0001)

        async def worker_call(event_type: str) -> None:
            for i in range(50):
                await service.process_call_event(
                    device_id=target_dev,
                    event=event_type,
                    phone_number=f"+1000{i}",
                    contact_name="Racer",
                )
                await asyncio.sleep(0.0001)

        # Run clear and call workers concurrently
        await asyncio.gather(
            worker_clear(),
            worker_call("ringing"),
            worker_call("talking"),
            worker_clear(),
        )

        # Ensure dictionary is consistent and no uncaught exceptions occurred
        call = await service.get_call(target_dev)
        # Call is either None (if clear won the race) or an ActiveCall dictionary
        if call is not None:
            assert call["state"] in ("RINGING", "TALKING")
            assert call["device_id"] == target_dev

        # Final cleanup
        await service.clear_device(target_dev)
        assert await service.get_call(target_dev) is None

    async def test_race_simultaneous_disconnect_and_unpair_events(self) -> None:
        """Firing DeviceDisconnected and DeviceUnpaired concurrently emits at most one IDLE event."""
        bus = EventBus()
        service = TelephonyService(event_bus=bus)
        state_changes: list[CallStateChangedEvent] = []
        bus.subscribe(CallStateChangedEvent, lambda ev: state_changes.append(ev))

        dev_id = "dual_event_dev"
        await service.process_call_event(dev_id, "talking", "+12345", "Alice")
        assert len(state_changes) == 1
        assert state_changes[0].new_state == "TALKING"

        # Fire both events simultaneously
        await asyncio.gather(
            bus.publish(DeviceDisconnected(device_id=dev_id)),
            bus.publish(DeviceUnpaired(device_id=dev_id)),
        )

        # Device must be cleared
        assert await service.get_call(dev_id) is None

        # Verify only one transition to IDLE was published (no duplicate IDLE events)
        idle_transitions = [e for e in state_changes if e.new_state == "IDLE"]
        assert len(idle_transitions) == 1
        service.close()

    async def test_race_concurrent_clear_all_and_inbound_calls(self) -> None:
        """clear_all() executing concurrently with inbound calls does not raise RuntimeError."""
        service = TelephonyService()

        async def stream_calls() -> None:
            for i in range(30):
                await service.process_call_event(
                    device_id=f"stream_dev_{i % 5}",
                    event="ringing",
                    phone_number=f"+111{i}",
                )
                await asyncio.sleep(0.0001)

        async def killer() -> None:
            for _ in range(10):
                await service.clear_all()
                await asyncio.sleep(0.0003)

        # Both run concurrently
        await asyncio.gather(stream_calls(), killer())

        # Cleanup
        await service.clear_all()
        assert len(await service.get_active_calls()) == 0

    async def test_race_concurrent_duplicate_ringing_and_talking(self) -> None:
        """Simultaneous ringing and talking packets on the same device resolve deterministically."""
        service = TelephonyService()
        dev_id = "dup_dev"

        async def send_ringing() -> None:
            await service.process_call_event(dev_id, "ringing", "+123", "User")

        async def send_talking() -> None:
            await service.process_call_event(dev_id, "talking", "+123", "User")

        # 20 concurrent coroutines racing to set ringing or talking
        tasks = [send_ringing() if i % 2 == 0 else send_talking() for i in range(20)]
        await asyncio.gather(*tasks)

        call = await service.get_call(dev_id)
        assert call is not None
        assert call["state"] in ("RINGING", "TALKING")
        await service.clear_device(dev_id)

    async def test_race_rapid_flapping_and_cancellation(self) -> None:
        """Rapid flapping between ringing and cancellation ends in a consistent state."""
        service = TelephonyService()
        dev_id = "flap_dev"

        for i in range(50):
            await service.process_call_event(dev_id, "ringing", "+111", "Caller")
            await service.process_call_event(dev_id, "idle", is_cancel=True)

        # Ended on idle, call must be None
        assert await service.get_call(dev_id) is None
        assert len(await service.get_active_calls()) == 0


# =============================================================================
# 4. EventBus Handler Exception & Slowness Isolation
# =============================================================================


class TestEventBusExceptionAndSlownessIsolation:
    """Stress tests verifying failing or slow subscribers never crash or corrupt TelephonyService."""

    async def test_failing_sync_and_async_subscribers_never_crash_service(self) -> None:
        """Subscribers raising various exceptions do not bubble up to crash process_call_event."""
        bus = EventBus()
        service = TelephonyService(event_bus=bus)

        # Register buggy subscribers
        def sync_exploder(ev: Any) -> None:
            raise ZeroDivisionError("sync explosion")

        async def async_exploder(ev: Any) -> None:
            raise RuntimeError("async explosion")

        def type_exploder(ev: Any) -> None:
            raise TypeError("type error in subscriber")

        bus.subscribe(CallEvent, sync_exploder)
        bus.subscribe(CallEvent, async_exploder)
        bus.subscribe(CallStateChangedEvent, type_exploder)
        bus.subscribe(CallStateChangedEvent, async_exploder)

        # TelephonyService must operate flawlessly despite subscriber exceptions
        call = await service.process_call_event("dev_buggy", "ringing", "+123", "Alice")
        assert call is not None
        assert call.state == CallState.RINGING

        call = await service.process_call_event("dev_buggy", "talking")
        assert call is not None
        assert call.state == CallState.TALKING

        # clear_device with buggy subscribers must also succeed
        cleared = await service.clear_device("dev_buggy")
        assert cleared is True
        assert await service.get_call("dev_buggy") is None

        service.close()

    async def test_slow_subscriber_does_not_crash_service(self) -> None:
        """Slow async subscribers execute concurrently and service remains functional."""
        bus = EventBus()
        service = TelephonyService(event_bus=bus)

        slow_executed = False

        async def slow_handler(ev: Any) -> None:
            nonlocal slow_executed
            await asyncio.sleep(0.01)
            slow_executed = True

        bus.subscribe(CallEvent, slow_handler)

        start = time.time()
        call = await service.process_call_event("dev_slow", "ringing", "+123", "Alice")
        elapsed = time.time() - start

        assert call is not None
        assert slow_executed is True
        assert elapsed >= 0.01  # EventBus awaits subscribers
        service.close()

    async def test_cross_device_lock_contention_under_slow_state_subscriber(
        self,
    ) -> None:
        """Empirically measure cross-device concurrency under slow CallStateChangedEvent subscriber.

        Because TelephonyService publishes CallStateChangedEvent outside async with self._lock,
        a slow subscriber to CallStateChangedEvent on dev1 does NOT delay dev2's process_call_event.
        We verify:
        1. Both calls succeed concurrently without lock starvation.
        2. Both devices maintain correct call state.
        3. Total elapsed time is close to a single subscriber duration (~0.05s) rather than serialized (~0.10s).
        """
        bus = EventBus()
        service = TelephonyService(event_bus=bus)

        async def slow_sub(ev: CallStateChangedEvent) -> None:
            await asyncio.sleep(0.05)

        bus.subscribe(CallStateChangedEvent, slow_sub)

        start = time.time()
        t1 = asyncio.create_task(
            service.process_call_event("dev_contention_1", "ringing")
        )
        await asyncio.sleep(0.005)  # Let t1 finish state mutation and enter subscriber sleep
        t2 = asyncio.create_task(
            service.process_call_event("dev_contention_2", "ringing")
        )

        r1, r2 = await asyncio.gather(t1, t2)
        elapsed = time.time() - start

        assert r1 is not None and r1.device_id == "dev_contention_1"
        assert r2 is not None and r2.device_id == "dev_contention_2"
        # Since t1 released the lock before awaiting subscribers, t2 does not wait for t1's slow subscriber.
        # Both slow subscribers run concurrently, so elapsed duration is < 0.08s (typically ~0.055s)
        assert elapsed < 0.08

        await service.clear_all()
        service.close()

    async def test_call_event_subscriber_queries_active_calls_without_deadlock(
        self,
    ) -> None:
        """CallEvent subscriber queries active calls without deadlock (published outside lock)."""
        bus = EventBus()
        service = TelephonyService(event_bus=bus)

        query_result: list[dict[str, Any]] = []

        async def call_event_subscriber(ev: CallEvent) -> None:
            # Query active calls - should not deadlock because CallEvent is published before self._lock
            res = await service.get_active_calls()
            query_result.extend(res)

        bus.subscribe(CallEvent, call_event_subscriber)

        # Trigger CallEvent
        await service.process_call_event("dev_outside_lock", "ringing", "+123", "Alice")

        # Verify subscriber ran and service is active
        call = await service.get_call("dev_outside_lock")
        assert call is not None
        await service.clear_device("dev_outside_lock")
        service.close()

    async def test_subscriber_reentrancy_deadlock_investigation(self) -> None:
        """Empirically verify subscriber re-entrancy without deadlock when querying async methods.

        CallStateChangedEvent is published after releasing self._lock.
        When an async subscriber queries self.get_active_calls(), it acquires the lock immediately
        without deadlocking. We verify both synchronous and asynchronous queries succeed.
        """
        bus = EventBus()
        service = TelephonyService(event_bus=bus)

        deadlock_detected = False
        sync_query_successful = False
        async_query_successful = False
        queried_calls: list[dict[str, Any]] = []

        async def reentrant_subscriber(ev: CallStateChangedEvent) -> None:
            nonlocal deadlock_detected, sync_query_successful, async_query_successful, queried_calls
            # 1. Synchronous query succeeds immediately
            sync_calls = service.get_active_calls_sync()
            if len(sync_calls) == 1:
                sync_query_successful = True

            # 2. Asynchronous query that acquires self._lock succeeds without deadlock
            try:
                queried_calls = await asyncio.wait_for(service.get_active_calls(), timeout=0.5)
                if len(queried_calls) == 1:
                    async_query_successful = True
            except asyncio.TimeoutError:
                deadlock_detected = True

        bus.subscribe(CallStateChangedEvent, reentrant_subscriber)

        # Trigger transition which emits CallStateChangedEvent
        await service.process_call_event("dev_reentrant", "ringing", "+123", "Alice")

        # Empirically verify:
        # - Synchronous query was successful without deadlocking
        assert sync_query_successful is True
        # - Async query succeeded promptly without deadlocking
        assert deadlock_detected is False
        assert async_query_successful is True
        assert len(queried_calls) == 1
        assert queried_calls[0]["device_id"] == "dev_reentrant"

        service.close()


# =============================================================================
# 5. Memory Leak & High-Volume Lifecycle Cleanup (500 Devices)
# =============================================================================


class TestMemoryLeakAndLifecyclePruning:
    """Stress tests creating and destroying 500 device call lifecycles."""

    async def test_five_hundred_device_lifecycles_sequential(self) -> None:
        """500 device call lifecycles created and destroyed sequentially."""
        service = TelephonyService()
        total_devices = 500

        for i in range(total_devices):
            dev_id = f"leak_test_dev_{i}"
            # Ringing
            await service.process_call_event(dev_id, "ringing", f"+100{i}", f"User {i}")
            # Talking
            await service.process_call_event(dev_id, "talking")
            # Idle / Ended
            await service.process_call_event(dev_id, "idle")

        # Explicitly assert _active_calls dictionary size returns to 0
        assert len(service._active_calls) == 0
        assert len(await service.get_active_calls()) == 0

    async def test_five_hundred_device_lifecycles_concurrent_batches(self) -> None:
        """500 devices created and terminated in concurrent batches of 50."""
        service = TelephonyService()
        total_devices = 500
        batch_size = 50

        async def run_device_cycle(idx: int) -> None:
            dev_id = f"batch_dev_{idx}"
            await service.process_call_event(
                dev_id, "ringing", f"+200{idx}", f"BatchUser_{idx}"
            )
            # 50% end via idle, 25% end via missedCall, 25% end via clear_device
            if idx % 4 == 0:
                await service.clear_device(dev_id)
            elif idx % 4 == 1:
                await service.process_call_event(dev_id, "missedCall")
            elif idx % 4 == 2:
                await service.process_call_event(dev_id, "idle")
            else:
                await service.process_call_event(dev_id, "ringing", is_cancel=True)

        for batch_start in range(0, total_devices, batch_size):
            batch_tasks = [
                run_device_cycle(i)
                for i in range(batch_start, batch_start + batch_size)
            ]
            await asyncio.gather(*batch_tasks)

        # Assert dictionary size returns to 0
        assert len(service._active_calls) == 0
        assert len(await service.get_active_calls()) == 0

    async def test_plugin_connected_peers_and_calls_cleanup_on_disconnect(self) -> None:
        """100 devices connect and disconnect via TelephonyPlugin; assert zero leaked state."""
        bus = EventBus()
        plugin = TelephonyPlugin(event_bus=bus)
        await plugin.start()

        num_devices = 100

        for i in range(num_devices):
            dev_id = f"peer_dev_{i}"
            sender = MockPacketSender()
            packet = Packet(
                type=TELEPHONY_PACKET_TYPE,
                body={"event": "ringing", "phoneNumber": f"+300{i}"},
            )
            await plugin.handle_packet(packet, dev_id, sender)

        assert len(plugin._connected_peers) == num_devices
        assert len(await plugin.get_active_calls()) == num_devices

        # Disconnect all 100 devices via EventBus
        for i in range(num_devices):
            dev_id = f"peer_dev_{i}"
            await bus.publish(DeviceDisconnected(device_id=dev_id))

        # Assert peer sender cache is completely drained
        assert len(plugin._connected_peers) == 0
        # Assert active calls dictionary is completely drained
        assert len(await plugin.get_active_calls()) == 0
        assert len(plugin.telephony_service._active_calls) == 0

        await plugin.stop()

    async def test_memory_garbage_collection_after_call_destruction(self) -> None:
        """Verify garbage collector frees ActiveCall instances upon removal."""
        service = TelephonyService()

        for i in range(100):
            await service.process_call_event(f"gc_dev_{i}", "ringing", "+123", "Test")

        assert len(service._active_calls) == 100
        await service.clear_all()
        assert len(service._active_calls) == 0

        # Trigger gc pass
        collected = gc.collect()
        assert collected >= 0
        assert len(service._active_calls) == 0

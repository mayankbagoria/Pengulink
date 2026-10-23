"""tests.unit.test_challenger_m28_iter2_stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Milestone 28 Iteration 2 Empirical Stress Test Suite:
High-contention reentrancy safety and non-reentrant lock verification.

Authored by Challenger M28-Iter2-1:
- Stress Test 1: 50 concurrent gather start/end calls with nested get_status & active_call_devices queries
- Stress Test 2: 50 concurrent interleaved start/end calls with micro-delays & lock assertion
- Stress Test 3: 50 concurrent calls with nested reentrant state mutations during external I/O
- Stress Test 4: 50 concurrent calls with EventBus AudioSinkUpdatedEvent cascading callbacks
- Stress Test 5: 50 concurrent calls racing with restore_media & device disconnects
"""

from __future__ import annotations

import asyncio
from typing import Any

from pengulink.core.events import (
    AudioSinkUpdatedEvent,
    EventBus,
)
from pengulink.plugins.mpris.adapter import MockMprisAdapter, PlayerState
from pengulink.services.audio import (
    AudioSink,
    LocalAudioService,
    MockAudioSinkBackend,
)
from pengulink.services.pausemusic import (
    PauseMusicConfig,
    PauseMusicService,
)


class StressMprisAdapter(MockMprisAdapter):
    """Mock MPRIS adapter instrumented with strict lock assertion and nested query hooks."""

    def __init__(self, svc_ref: list[PauseMusicService | None], delay: float = 0.0) -> None:
        super().__init__(auto_default_player=False)
        self.svc_ref = svc_ref
        self.delay = delay
        self.query_count = 0
        self.nested_observations: list[dict[str, Any]] = []

    @property
    def svc(self) -> PauseMusicService:
        assert self.svc_ref[0] is not None
        return self.svc_ref[0]

    async def list_players(self) -> list[str]:
        assert self.svc._lock.locked() is False, "Lock held during list_players()!"
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return await super().list_players()

    async def get_player_status(self, player: str) -> PlayerState | None:
        assert self.svc._lock.locked() is False, f"Lock held during get_player_status({player})!"
        # Execute nested queries
        status = self.svc.get_status()
        active = self.svc.active_call_devices
        interrupted = self.svc.is_interrupted
        self.nested_observations.append({
            "stage": "get_player_status",
            "active_len": len(active),
            "status_interrupted": status["is_interrupted"],
            "interrupted": interrupted,
        })
        self.query_count += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return await super().get_player_status(player)

    async def send_action(self, player: str, action: str) -> bool:
        assert self.svc._lock.locked() is False, f"Lock held during send_action({player}, {action})!"
        # Execute nested queries
        status = self.svc.get_status()
        active = self.svc.active_call_devices
        interrupted = self.svc.is_interrupted
        self.nested_observations.append({
            "stage": f"send_action_{action}",
            "active_len": len(active),
            "status_interrupted": status["is_interrupted"],
            "interrupted": interrupted,
        })
        self.query_count += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return await super().send_action(player, action)


class StressAudioService(LocalAudioService):
    """Mock Audio service instrumented with strict lock assertion and nested query hooks."""

    def __init__(
        self,
        backend: MockAudioSinkBackend,
        event_bus: EventBus | None,
        svc_ref: list[PauseMusicService | None],
        delay: float = 0.0,
    ) -> None:
        super().__init__(backend=backend, event_bus=event_bus)
        self.svc_ref = svc_ref
        self.delay = delay
        self.query_count = 0
        self.nested_observations: list[dict[str, Any]] = []

    @property
    def svc(self) -> PauseMusicService:
        assert self.svc_ref[0] is not None
        return self.svc_ref[0]

    async def list_sinks(self) -> list[AudioSink]:
        assert self.svc._lock.locked() is False, "Lock held during list_sinks()!"
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return await super().list_sinks()

    async def set_muted(self, name: str, muted: bool) -> bool:
        assert self.svc._lock.locked() is False, f"Lock held during set_muted({name}, {muted})!"
        status = self.svc.get_status()
        active = self.svc.active_call_devices
        interrupted = self.svc.is_interrupted
        self.nested_observations.append({
            "stage": f"set_muted_{muted}",
            "active_len": len(active),
            "status_interrupted": status["is_interrupted"],
            "interrupted": interrupted,
        })
        self.query_count += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return await super().set_muted(name, muted)


class TestPauseMusicHighContentionStress:
    """Empirical Stress Tests: 50 concurrent calls with nested reentrant queries and lock assertions."""

    async def test_50_concurrent_gather_start_and_end_with_nested_queries(self) -> None:
        """50 simultaneous start calls followed by 50 simultaneous end calls via asyncio.gather.

        Asserts:
        - svc._lock.locked() is False during every external operation
        - nested callbacks querying get_status() and active_call_devices complete cleanly
        - zero deadlocks or timeouts
        - reference count transitions 0 -> 1 and N -> 0 accurately
        """
        svc_holder: list[PauseMusicService | None] = [None]
        mpris = StressMprisAdapter(svc_holder, delay=0.0005)
        mpris.add_mock_player("spotify", PlayerState(player="spotify", is_playing=True))

        backend = MockAudioSinkBackend([
            AudioSink(name="sink_main", description="Main Output", volume=50000, muted=False)
        ])
        bus = EventBus()
        audio = StressAudioService(backend, bus, svc_holder, delay=0.0005)

        cfg = PauseMusicConfig(action_pause=True, action_mute=True, action_resume=True)
        svc = PauseMusicService(config=cfg, mpris_adapter=mpris, audio_service=audio, event_bus=bus)
        svc_holder[0] = svc

        # 1. 50 concurrent start calls
        start_tasks = [svc.handle_call_started(f"device_{i}") for i in range(50)]
        results_start = await asyncio.wait_for(asyncio.gather(*start_tasks), timeout=5.0)

        assert len(results_start) == 50
        # Exactly one call should have returned True (the 0 -> 1 transition)
        assert results_start.count(True) == 1
        assert results_start.count(False) == 49

        assert svc.is_interrupted is True
        assert len(svc.active_call_devices) == 50
        assert svc._lock.locked() is False
        assert mpris.query_count > 0
        assert audio.query_count > 0

        # Player should be paused, sink should be muted
        st = await mpris.get_player_status("spotify")
        assert st is not None and st.is_playing is False
        sinks = await audio.list_sinks()
        assert sinks[0].muted is True

        # 2. 50 concurrent end calls
        end_tasks = [svc.handle_call_ended(f"device_{i}") for i in range(50)]
        results_end = await asyncio.wait_for(asyncio.gather(*end_tasks), timeout=5.0)

        assert len(results_end) == 50
        # Exactly one call should have returned True (the 1 -> 0 transition)
        assert results_end.count(True) == 1
        assert results_end.count(False) == 49

        assert svc.is_interrupted is False
        assert len(svc.active_call_devices) == 0
        assert svc._lock.locked() is False

        # Player should be resumed, sink should be unmuted
        st_end = await mpris.get_player_status("spotify")
        assert st_end is not None and st_end.is_playing is True
        sinks_end = await audio.list_sinks()
        assert sinks_end[0].muted is False

    async def test_50_concurrent_interleaved_start_and_end(self) -> None:
        """50 tasks concurrently interleaving start and end calls on active and inactive devices.

        Simulates chaotic multi-device storms under heavy contention.
        Asserts:
        - svc._lock.locked() is False continuously
        - zero deadlocks
        - eventual consistency after all tasks settle
        """
        svc_holder: list[PauseMusicService | None] = [None]
        mpris = StressMprisAdapter(svc_holder, delay=0.0002)
        mpris.add_mock_player("vlc", PlayerState(player="vlc", is_playing=True))

        backend = MockAudioSinkBackend([
            AudioSink(name="sink_hdmi", description="HDMI Output", volume=60000, muted=False)
        ])
        bus = EventBus()
        audio = StressAudioService(backend, bus, svc_holder, delay=0.0002)

        cfg = PauseMusicConfig(action_pause=True, action_mute=True, action_resume=True)
        svc = PauseMusicService(config=cfg, mpris_adapter=mpris, audio_service=audio, event_bus=bus)
        svc_holder[0] = svc

        # Pre-seed 25 active devices
        for i in range(25):
            await svc.handle_call_started(f"init_dev_{i}")
        assert len(svc.active_call_devices) == 25
        assert svc.is_interrupted is True

        # Launch 50 interleaved actions:
        # 25 tasks end the init devices
        # 25 tasks start new devices
        tasks = []
        for i in range(25):
            tasks.append(svc.handle_call_ended(f"init_dev_{i}"))
            tasks.append(svc.handle_call_started(f"storm_dev_{i}"))

        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

        # After gather, all init_devs should be gone, and all storm_devs present
        assert svc.is_interrupted is True
        assert len(svc.active_call_devices) == 25
        assert svc.active_call_devices == {f"storm_dev_{i}" for i in range(25)}
        assert svc._lock.locked() is False

        # Clean up remaining 25 storm devices concurrently
        cleanup_tasks = [svc.handle_call_ended(f"storm_dev_{i}") for i in range(25)]
        await asyncio.wait_for(asyncio.gather(*cleanup_tasks), timeout=5.0)

        assert svc.is_interrupted is False
        assert len(svc.active_call_devices) == 0
        assert svc._lock.locked() is False

        # Assert final resumed state
        st = await mpris.get_player_status("vlc")
        assert st is not None and st.is_playing is True
        sinks = await audio.list_sinks()
        assert sinks[0].muted is False

    async def test_50_concurrent_calls_with_nested_reentrant_mutations(self) -> None:
        """50 concurrent calls where external MPRIS send_action triggers reentrant call state mutations."""
        svc_holder: list[PauseMusicService | None] = [None]

        class ReentrantMutatingMpris(MockMprisAdapter):
            def __init__(self, svc_ref: list[PauseMusicService | None]) -> None:
                super().__init__(auto_default_player=False)
                self.svc_ref = svc_ref
                self.reentrant_calls_fired = 0

            @property
            def svc(self) -> PauseMusicService:
                assert self.svc_ref[0] is not None
                return self.svc_ref[0]

            async def send_action(self, player: str, action: str) -> bool:
                assert self.svc._lock.locked() is False, f"Lock held on {action}!"
                # Trigger re-entrant call state change if action is Pause
                if action == "Pause" and self.reentrant_calls_fired < 5:
                    self.reentrant_calls_fired += 1
                    # Reentrant call on a sub-device
                    await self.svc.handle_call_state_changed(
                        f"sub_dev_{self.reentrant_calls_fired}", "RINGING"
                    )
                return await super().send_action(player, action)

        mpris = ReentrantMutatingMpris(svc_holder)
        mpris.add_mock_player("rhythmbox", PlayerState(player="rhythmbox", is_playing=True))

        cfg = PauseMusicConfig(action_pause=True, action_resume=True)
        svc = PauseMusicService(config=cfg, mpris_adapter=mpris)
        svc_holder[0] = svc

        # 50 concurrent start calls
        start_tasks = [svc.handle_call_started(f"main_dev_{i}") for i in range(50)]
        await asyncio.wait_for(asyncio.gather(*start_tasks), timeout=5.0)

        assert svc.is_interrupted is True
        assert svc._lock.locked() is False
        # Sub-devices were added reentrantly without deadlock
        for s in range(1, mpris.reentrant_calls_fired + 1):
            assert f"sub_dev_{s}" in svc.active_call_devices

        # Clean up all devices
        all_devs = list(svc.active_call_devices)
        cleanup_tasks = [svc.handle_call_ended(d) for d in all_devs]
        await asyncio.wait_for(asyncio.gather(*cleanup_tasks), timeout=5.0)

        assert svc.is_interrupted is False
        assert len(svc.active_call_devices) == 0
        assert svc._lock.locked() is False

    async def test_50_concurrent_calls_with_audio_eventbus_cascading(self) -> None:
        """50 concurrent calls with AudioSinkUpdatedEvent listener executing reentrant queries and calls."""
        svc_holder: list[PauseMusicService | None] = [None]
        bus = EventBus()
        backend = MockAudioSinkBackend([
            AudioSink(name="sink_pulse", description="Pulse", volume=40000, muted=False)
        ])
        audio = LocalAudioService(backend=backend, event_bus=bus)

        cfg = PauseMusicConfig(action_pause=False, action_mute=True, action_resume=True)
        svc = PauseMusicService(config=cfg, audio_service=audio, event_bus=bus)
        svc_holder[0] = svc

        event_queries: list[dict[str, Any]] = []

        async def on_audio_updated(event: AudioSinkUpdatedEvent) -> None:
            assert svc._lock.locked() is False, "Lock held during AudioSinkUpdatedEvent dispatch!"
            status = svc.get_status()
            active = svc.active_call_devices
            event_queries.append({
                "sink": event.sink.name if event.sink else None,
                "muted": event.sink.muted if event.sink else None,
                "active_count": len(active),
                "is_interrupted": status["is_interrupted"],
            })

        bus.subscribe(AudioSinkUpdatedEvent, on_audio_updated)

        # 50 concurrent start calls
        start_tasks = [svc.handle_call_started(f"event_dev_{i}") for i in range(50)]
        await asyncio.wait_for(asyncio.gather(*start_tasks), timeout=5.0)

        assert svc.is_interrupted is True
        assert len(svc.active_call_devices) == 50
        assert svc._lock.locked() is False
        assert len(event_queries) >= 1
        assert event_queries[0]["muted"] is True

        # 50 concurrent end calls
        end_tasks = [svc.handle_call_ended(f"event_dev_{i}") for i in range(50)]
        await asyncio.wait_for(asyncio.gather(*end_tasks), timeout=5.0)

        assert svc.is_interrupted is False
        assert len(svc.active_call_devices) == 0
        assert svc._lock.locked() is False
        assert event_queries[-1]["muted"] is False

    async def test_50_concurrent_calls_racing_with_restore_media(self) -> None:
        """50 concurrent calls racing with restore_media() invocations.

        Verifies that restore_media() and concurrent start/end calls never deadlock,
        and that lock is released during restore actions.
        """
        svc_holder: list[PauseMusicService | None] = [None]
        mpris = StressMprisAdapter(svc_holder, delay=0.0001)
        mpris.add_mock_player("clementine", PlayerState(player="clementine", is_playing=True))

        cfg = PauseMusicConfig(action_pause=True, action_resume=True)
        svc = PauseMusicService(config=cfg, mpris_adapter=mpris)
        svc_holder[0] = svc

        # Pre-seed 10 calls
        for i in range(10):
            await svc.handle_call_started(f"seed_{i}")

        # Launch 50 concurrent start calls alongside 2 restore_media calls
        tasks = [svc.handle_call_started(f"race_dev_{i}") for i in range(50)]
        tasks.append(svc.restore_media())
        tasks.append(svc.restore_media())

        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

        # Zero deadlocks, lock is released
        assert svc._lock.locked() is False

        # Clean up everything cleanly
        await asyncio.wait_for(svc.restore_media(), timeout=2.0)
        assert svc.is_interrupted is False
        assert len(svc.active_call_devices) == 0
        assert svc._lock.locked() is False

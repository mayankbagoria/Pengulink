"""tests.unit.test_challenger_m28_concurrency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Milestone 28 Concurrency, Overlapping Multi-Call Ref-Counting, Flapping,
and Lifecycle Race Condition Empirical Stress Test Suite.

Authored by Challenger M28-2 (Empirical Challenger):
- Category 1: Massive Overlapping Calls Across 5–10 Virtual Devices (Tests 1–6)
- Category 2: Rapid Flapping & High-Frequency Bursts (Tests 7–12)
- Category 3: Abrupt Disconnect and Unpair Races (Tests 13–17)
- Category 4: Plugin Start/Stop Races & Lifecycle (Tests 18–22)
- Category 5: Reentrancy Safety & Callback Integrity (Tests 23–26)
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pengulink.core.events import (
    CallStateChangedEvent,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
)
from pengulink.plugins.mpris.adapter import MockMprisAdapter, PlayerState
from pengulink.plugins.pausemusic import (
    PauseMusicPlugin,
)
from pengulink.services.audio import (
    AudioSink,
    LocalAudioService,
    MockAudioSinkBackend,
)
from pengulink.services.pausemusic import (
    PauseMusicConfig,
    PauseMusicService,
)

# ---------------------------------------------------------------------
# Synchronous Test Fixtures (conforming to repo runner conventions)
# ---------------------------------------------------------------------


@pytest.fixture
def event_bus() -> EventBus:
    """Isolated EventBus instance for concurrency tests."""
    return EventBus()


@pytest.fixture
def mock_mpris() -> MockMprisAdapter:
    """MockMprisAdapter with custom test players."""
    adapter = MockMprisAdapter(auto_default_player=False)
    adapter._players["Spotify"] = PlayerState(
        player="Spotify",
        title="Active Track",
        artist="Artist A",
        album="Album A",
        is_playing=True,
    )
    adapter._players["VLC"] = PlayerState(
        player="VLC",
        title="Paused Video",
        artist="Artist B",
        album="Album B",
        is_playing=False,
    )
    return adapter


@pytest.fixture
def mock_audio_backend() -> MockAudioSinkBackend:
    """MockAudioSinkBackend fixture with unmuted and muted sinks."""
    return MockAudioSinkBackend(
        initial_sinks=[
            AudioSink(
                name="alsa_output.pci-0000_00_1f.3.analog-stereo",
                description="Built-in Audio Analog Stereo",
                volume=32768,
                muted=False,
            ),
            AudioSink(
                name="alsa_output.pci-0000_01_00.1.hdmi-stereo",
                description="HDMI Audio Output",
                volume=65536,
                muted=True,
            ),
        ]
    )


@pytest.fixture
def local_audio(
    mock_audio_backend: MockAudioSinkBackend, event_bus: EventBus
) -> LocalAudioService:
    """LocalAudioService backed by MockAudioSinkBackend."""
    return LocalAudioService(backend=mock_audio_backend, event_bus=event_bus)


@pytest.fixture
def pausemusic_service(
    mock_mpris: MockMprisAdapter,
    local_audio: LocalAudioService,
    event_bus: EventBus,
) -> PauseMusicService:
    """PauseMusicService with pause and mute actions enabled for full verification."""
    config = PauseMusicConfig(
        condition_talking=False,
        action_pause=True,
        action_mute=True,
        action_resume=True,
    )
    return PauseMusicService(
        config=config,
        mpris_adapter=mock_mpris,
        audio_service=local_audio,
        event_bus=event_bus,
    )


# ---------------------------------------------------------------------
# Category 1: Massive Overlapping Calls Across 5–10 Virtual Devices
# ---------------------------------------------------------------------


class TestMassiveOverlappingCalls:
    """Empirical stress testing of overlapping multi-device call ref-counting."""

    async def test_overlapping_calls_5_devices_arbitrary_order(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
    ) -> None:
        """5 devices start, transition, and end in interleaved arbitrary order.

        Playback and unmuting must resume strictly when the 5th and final call ends.
        """
        service = pausemusic_service
        devices = [f"device_{i}" for i in range(1, 6)]

        # Step 1: dev1 starts ringing -> 0 -> 1 transition
        started = await service.handle_call_started(devices[0], "RINGING")
        assert started is True
        assert service.is_interrupted is True
        assert len(service.active_call_devices) == 1
        st_spotify = await mock_mpris.get_player_status("Spotify")
        assert st_spotify is not None and not st_spotify.is_playing
        sinks = await local_audio.list_sinks()
        assert any(s.name.endswith("analog-stereo") and s.muted for s in sinks)

        # Step 2: dev2, dev3, dev4, dev5 join at staggered intervals (N -> N+1)
        for dev in devices[1:]:
            res = await service.handle_call_started(dev, "RINGING")
            assert res is False  # N -> N+1 returns False (no new snapshot taken)
            assert dev in service.active_call_devices

        assert len(service.active_call_devices) == 5
        assert service.paused_players == {"Spotify"}

        # Step 3: dev2, dev4 transition to TALKING (idempotent duplicate join)
        res_talk2 = await service.handle_call_started(devices[1], "TALKING")
        assert res_talk2 is False
        assert len(service.active_call_devices) == 5

        # Step 4: Calls end in arbitrary non-FIFO order (dev3, dev1, dev5, dev2)
        end_order = [devices[2], devices[0], devices[4], devices[1]]
        for dev in end_order:
            ended = await service.handle_call_ended(dev)
            assert ended is False  # N -> N-1 returns False while N > 1
            assert dev not in service.active_call_devices
            assert service.is_interrupted is True
            # Playback MUST NOT resume while other calls are active
            st = await mock_mpris.get_player_status("Spotify")
            assert st is not None and not st.is_playing
            sinks = await local_audio.list_sinks()
            assert any(s.name.endswith("analog-stereo") and s.muted for s in sinks)

        # Exactly 1 active call device remains: devices[3] (dev4)
        assert service.active_call_devices == {devices[3]}
        assert len(service.active_call_devices) == 1

        # Step 5: Final call ends -> 1 -> 0 transition
        final_end = await service.handle_call_ended(devices[3])
        assert final_end is True
        assert service.is_interrupted is False
        assert len(service.active_call_devices) == 0
        assert len(service.paused_players) == 0
        assert len(service.muted_sinks) == 0

        # Strict playback and audio unmuting restoration verification
        st_final = await mock_mpris.get_player_status("Spotify")
        assert st_final is not None and st_final.is_playing is True
        sinks_final = await local_audio.list_sinks()
        assert any(s.name.endswith("analog-stereo") and not s.muted for s in sinks_final)
        # HDMI was muted prior to the call; verify it remained muted
        assert any(s.name.endswith("hdmi-stereo") and s.muted for s in sinks_final)

    async def test_overlapping_calls_10_devices_concurrent_gather(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
    ) -> None:
        """10 devices initiate calls concurrently via asyncio.gather.

        Verify snapshot is taken exactly once, all 10 are tracked, and concurrent
        resolution releases media cleanly.
        """
        service = pausemusic_service
        devices = [f"peer_device_{i:02d}" for i in range(10)]

        # Concurrently initiate 10 calls
        results = await asyncio.gather(
            *[service.handle_call_started(dev, "RINGING") for dev in devices]
        )

        # Exactly one call must have executed the 0 -> 1 transition
        assert results.count(True) == 1
        assert results.count(False) == 9
        assert len(service.active_call_devices) == 10
        assert service.paused_players == {"Spotify"}

        # Concurrently end the first 9 calls
        end_first_9 = await asyncio.gather(
            *[service.handle_call_ended(dev) for dev in devices[:9]]
        )
        assert all(res is False for res in end_first_9)
        assert len(service.active_call_devices) == 1
        assert service.is_interrupted is True

        # Media is still paused
        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and not st.is_playing

        # End the 10th call
        last_end = await service.handle_call_ended(devices[9])
        assert last_end is True
        assert service.is_interrupted is False
        assert len(service.active_call_devices) == 0

        # Media is now playing
        st_final = await mock_mpris.get_player_status("Spotify")
        assert st_final is not None and st_final.is_playing is True

    async def test_snapshot_preservation_under_multi_call_storm(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
    ) -> None:
        """Verify pre-call snapshot is NEVER overwritten or corrupted by subsequent calls.

        Initial state: Spotify=playing, VLC=paused.
        Sinks: Analog=unmuted, HDMI=muted.
        """
        service = pausemusic_service

        # Add a 3rd player that was STOPPED before any call
        mock_mpris._players["Audacious"] = PlayerState(
            player="Audacious",
            title="Stopped Track",
            is_playing=False,
            can_play=True,
            can_pause=True,
        )

        # Device 1 rings -> snapshots Spotify and Analog
        await service.handle_call_started("dev1", "RINGING")
        assert service.paused_players == {"Spotify"}
        assert service.muted_sinks == {"alsa_output.pci-0000_00_1f.3.analog-stereo"}

        # Devices 2 through 8 join
        for i in range(2, 9):
            await service.handle_call_started(f"dev{i}", "RINGING")

        # Snapshot sets MUST remain unchanged despite multiple active devices
        assert service.paused_players == {"Spotify"}
        assert service.muted_sinks == {"alsa_output.pci-0000_00_1f.3.analog-stereo"}

        # End all calls
        for i in range(1, 9):
            await service.handle_call_ended(f"dev{i}")

        # Post-interruption verification:
        # Spotify was resumed
        st_spot = await mock_mpris.get_player_status("Spotify")
        assert st_spot is not None and st_spot.is_playing is True

        # VLC was paused before calls and must NOT have been resumed
        st_vlc = await mock_mpris.get_player_status("VLC")
        assert st_vlc is not None and st_vlc.is_playing is False

        # Audacious was stopped before calls and must NOT have been resumed
        st_aud = await mock_mpris.get_player_status("Audacious")
        assert st_aud is not None and st_aud.is_playing is False

        # Analog was unmuted
        sinks = await local_audio.list_sinks()
        assert any(s.name.endswith("analog-stereo") and not s.muted for s in sinks)
        # HDMI was muted before calls and must remain muted
        assert any(s.name.endswith("hdmi-stereo") and s.muted for s in sinks)

    async def test_talking_condition_policy_with_multiple_devices(
        self,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
    ) -> None:
        """When condition_talking=True, ringing calls must not trigger pause.

        Only answered calls (TALKING) trigger interruption.
        """
        config = PauseMusicConfig(condition_talking=True, action_pause=True, action_resume=True)
        service = PauseMusicService(
            config=config, mpris_adapter=mock_mpris, audio_service=local_audio
        )

        # Device A rings -> ignored under talking condition
        res_a1 = await service.handle_call_state_changed("devA", "RINGING")
        assert res_a1 is False
        assert service.is_interrupted is False
        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and st.is_playing is True

        # Device B rings -> also ignored
        res_b1 = await service.handle_call_state_changed("devB", "RINGING")
        assert res_b1 is False
        assert service.is_interrupted is False

        # Device A answers (TALKING) -> triggers interruption
        res_a2 = await service.handle_call_state_changed("devA", "TALKING")
        assert res_a2 is True
        assert service.is_interrupted is True
        assert service.active_call_devices == {"devA"}
        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and not st.is_playing

        # Device B answers (TALKING) -> joins interruption (N -> N+1)
        res_b2 = await service.handle_call_state_changed("devB", "TALKING")
        assert res_b2 is False
        assert service.active_call_devices == {"devA", "devB"}

        # Device A ends -> Device B still talking, media remains paused
        await service.handle_call_state_changed("devA", "IDLE")
        assert service.is_interrupted is True
        assert service.active_call_devices == {"devB"}
        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and not st.is_playing

        # Device B ends -> 1 -> 0 transition, media resumes
        await service.handle_call_state_changed("devB", "IDLE")
        assert service.is_interrupted is False
        assert len(service.active_call_devices) == 0
        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and st.is_playing is True

    async def test_continuous_churn_around_anchor_device(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """An anchor device stays on a call while 15 transient devices churn around it.

        Media must remain paused continuously without flickering until anchor ends.
        """
        service = pausemusic_service

        # Anchor device starts call
        await service.handle_call_started("anchor_device", "TALKING")
        assert service.is_interrupted is True

        # Rapidly churn 15 transient devices
        for i in range(15):
            transient_id = f"transient_{i}"
            await service.handle_call_started(transient_id, "RINGING")
            assert service.is_interrupted is True
            # Media must still be paused
            st = await mock_mpris.get_player_status("Spotify")
            assert st is not None and not st.is_playing

            await service.handle_call_ended(transient_id)
            # Anchor is still active, media must NOT resume
            assert service.is_interrupted is True
            st = await mock_mpris.get_player_status("Spotify")
            assert st is not None and not st.is_playing

        # Only anchor remains
        assert service.active_call_devices == {"anchor_device"}

        # Conclude anchor call
        await service.handle_call_ended("anchor_device")
        assert service.is_interrupted is False
        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and st.is_playing is True

    async def test_permutation_matrix_reproducibility(
        self,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
    ) -> None:
        """Run 5 distinct shuffled permutations of 4 devices starting and ending calls.

        Verify invariant: media resumes iff all active devices have ended.
        """
        permutations = [
            # Starts, Ends
            (["d1", "d2", "d3", "d4"], ["d1", "d2", "d3", "d4"]),
            (["d4", "d3", "d2", "d1"], ["d1", "d3", "d2", "d4"]),
            (["d2", "d4", "d1", "d3"], ["d4", "d2", "d3", "d1"]),
            (["d3", "d1", "d4", "d2"], ["d3", "d4", "d1", "d2"]),
            (["d1", "d4", "d2", "d3"], ["d2", "d1", "d4", "d3"]),
        ]

        for starts, ends in permutations:
            # Reset player to playing before each test iteration
            p = mock_mpris._players["Spotify"]
            p.is_playing = True
            service = PauseMusicService(
                config=PauseMusicConfig(action_pause=True, action_resume=True),
                mpris_adapter=mock_mpris,
                audio_service=local_audio,
            )

            # Start calls in specified order
            for dev in starts:
                await service.handle_call_started(dev, "RINGING")
                assert service.is_interrupted is True

            # End calls in specified order
            for i, dev in enumerate(ends):
                await service.handle_call_ended(dev)
                is_last = (i == len(ends) - 1)
                st = await mock_mpris.get_player_status("Spotify")
                assert st is not None
                if not is_last:
                    assert st.is_playing is False
                    assert service.is_interrupted is True
                else:
                    assert st.is_playing is True
                    assert service.is_interrupted is False


# ---------------------------------------------------------------------
# Category 2: Rapid Flapping & High-Frequency Bursts
# ---------------------------------------------------------------------


class TestRapidFlappingAndBursts:
    """Stress tests simulating high-frequency, chaotic call state transitions."""

    async def test_rapid_flapping_burst_single_device_50_transitions_concurrent(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """Burst of 50 rapid state transitions on a single device fired concurrently.

        Verify no deadlocks, no exceptions, and state coherency.
        """
        service = pausemusic_service
        device = "flapping_device_01"

        # Prepare 50 alternating states
        states = ["RINGING", "TALKING", "IDLE"] * 16 + ["RINGING", "IDLE"]
        assert len(states) == 50

        # Fire all 50 transitions concurrently
        await asyncio.gather(
            *[service.handle_call_state_changed(device, state) for state in states]
        )

        # After the concurrent burst, send a deterministic IDLE to settle
        await service.handle_call_state_changed(device, "IDLE")
        assert service.is_interrupted is False
        assert len(service.active_call_devices) == 0

        # Player must be playing
        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and st.is_playing is True

    async def test_rapid_flapping_burst_multi_device_100_transitions_concurrent(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """100 transitions distributed across 10 devices fired concurrently via asyncio.gather.

        Verify zero deadlocks, zero exceptions, and clean post-burst restoration.
        """
        service = pausemusic_service
        devices = [f"dev_{i}" for i in range(10)]

        tasks = []
        for i in range(100):
            dev = devices[i % 10]
            state = "RINGING" if (i % 2 == 0) else "IDLE"
            tasks.append(service.handle_call_state_changed(dev, state))

        # Execute burst
        await asyncio.gather(*tasks)

        # Coherency check: status matches internal tracking
        status = service.get_status()
        assert status["is_interrupted"] == (len(service.active_call_devices) > 0)

        # Settle all devices to IDLE cleanly
        settle_tasks = [
            service.handle_call_state_changed(dev, "IDLE")
            for dev in list(service.active_call_devices)
        ]
        await asyncio.gather(*settle_tasks)

        assert service.is_interrupted is False
        assert len(service.active_call_devices) == 0
        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and st.is_playing is True

    async def test_rapid_flapping_burst_identical_ringing_idempotence(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """50 identical RINGING events fired concurrently for the same device.

        Verify reference set contains exactly 1 entry, and media is paused once.
        """
        service = pausemusic_service
        device = "spam_ringer"

        res = await asyncio.gather(
            *[service.handle_call_started(device, "RINGING") for _ in range(50)]
        )
        assert res.count(True) == 1
        assert res.count(False) == 49
        assert service.active_call_devices == {device}

        # Concurrently fire 50 IDLE events
        res_end = await asyncio.gather(
            *[service.handle_call_ended(device) for _ in range(50)]
        )
        assert res_end.count(True) == 1
        assert res_end.count(False) == 49
        assert len(service.active_call_devices) == 0
        assert service.is_interrupted is False

    async def test_rapid_sequential_ringing_idle_cycles(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """30 rapid sequential cycles of RINGING -> IDLE.

        Verify snapshot is cleanly captured and restored on every cycle without leaks.
        """
        service = pausemusic_service
        device = "cycle_device"

        for cycle in range(30):
            # Ringing
            start_ok = await service.handle_call_state_changed(device, "RINGING")
            assert start_ok is True
            assert service.is_interrupted is True
            st_pause = await mock_mpris.get_player_status("Spotify")
            assert st_pause is not None and not st_pause.is_playing

            # Idle
            end_ok = await service.handle_call_state_changed(device, "IDLE")
            assert end_ok is True
            assert service.is_interrupted is False
            st_play = await mock_mpris.get_player_status("Spotify")
            assert st_play is not None and st_play.is_playing is True

            # Snapshot sets must be completely empty after each cycle
            assert len(service.paused_players) == 0
            assert len(service.muted_sinks) == 0

    async def test_rapid_flapping_with_delayed_mpris_backend(
        self,
        local_audio: LocalAudioService,
    ) -> None:
        """Simulate an MPRIS backend with artificial async delays in all methods.

        Verify lock properly serializes snapshots without corrupted state.
        """

        class DelayedMockMprisAdapter(MockMprisAdapter):
            async def list_players(self) -> list[str]:
                await asyncio.sleep(0.005)
                return await super().list_players()

            async def get_player_status(self, player: str) -> PlayerState | None:
                await asyncio.sleep(0.005)
                return await super().get_player_status(player)

            async def send_action(self, player: str, action: str) -> bool:
                await asyncio.sleep(0.005)
                return await super().send_action(player, action)

        delayed_mpris = DelayedMockMprisAdapter(auto_default_player=False)
        delayed_mpris._players["SlowPlayer"] = PlayerState(
            player="SlowPlayer",
            is_playing=True,
        )

        service = PauseMusicService(
            config=PauseMusicConfig(action_pause=True, action_resume=True),
            mpris_adapter=delayed_mpris,
            audio_service=local_audio,
        )

        # Concurrently fire 5 devices starting calls against slow adapter
        devices = [f"slow_dev_{i}" for i in range(5)]
        results = await asyncio.gather(
            *[service.handle_call_started(d, "RINGING") for d in devices]
        )
        assert results.count(True) == 1
        assert len(service.active_call_devices) == 5
        assert service.paused_players == {"SlowPlayer"}

        st = await delayed_mpris.get_player_status("SlowPlayer")
        assert st is not None and not st.is_playing

        # Concurrently end all 5 calls
        end_results = await asyncio.gather(
            *[service.handle_call_ended(d) for d in devices]
        )
        assert end_results.count(True) == 1
        assert len(service.active_call_devices) == 0
        assert service.is_interrupted is False

        st_final = await delayed_mpris.get_player_status("SlowPlayer")
        assert st_final is not None and st_final.is_playing is True

    async def test_intermittent_mpris_errors_under_burst(
        self,
        local_audio: LocalAudioService,
    ) -> None:
        """MPRIS adapter intermittently raises exceptions during action dispatches.

        Verify PauseMusicService handles errors defensively without deadlocking.
        """

        class FaultyMprisAdapter(MockMprisAdapter):
            def __init__(self) -> None:
                super().__init__(auto_default_player=False)
                self.error_countdown = 0

            async def send_action(self, player: str, action: str) -> bool:
                if self.error_countdown > 0:
                    self.error_countdown -= 1
                    raise RuntimeError("Simulated D-Bus timeout")
                return await super().send_action(player, action)

        faulty = FaultyMprisAdapter()
        faulty._players["FaultyPlayer"] = PlayerState(
            player="FaultyPlayer",
            is_playing=True,
        )

        service = PauseMusicService(
            config=PauseMusicConfig(action_pause=True, action_resume=True),
            mpris_adapter=faulty,
            audio_service=local_audio,
        )

        # Trigger pause with faulty adapter
        faulty.error_countdown = 1
        # Should not raise exception
        await service.handle_call_started("dev_fault", "RINGING")
        assert service.is_interrupted is True

        # Trigger resume with faulty adapter
        faulty.error_countdown = 1
        # Should not raise exception
        await service.handle_call_ended("dev_fault")
        assert service.is_interrupted is False


# ---------------------------------------------------------------------
# Category 3: Abrupt Disconnect and Unpair Races
# ---------------------------------------------------------------------


class TestAbruptDisconnectAndUnpairRaces:
    """Stress tests simulating abrupt disconnections and unpairing while active."""

    async def test_disconnect_racing_with_call_started(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """Call started races with device disconnect on the same device.

        Verify no unhandled exceptions, zero deadlocks, and coherent outcome.
        """
        service = pausemusic_service
        device = "racing_device_01"

        await asyncio.gather(
            service.handle_call_started(device, "RINGING"),
            service.handle_device_disconnected(device),
        )

        # Coherency assertion: if still interrupted, dev must be in active_devices
        if service.is_interrupted:
            assert device in service.active_call_devices
            # Clean up
            await service.handle_device_disconnected(device)
            assert service.is_interrupted is False
        else:
            assert len(service.active_call_devices) == 0

        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and st.is_playing is True

    async def test_unpair_racing_with_call_ended(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """Call ended races with device unpair on the same device."""
        service = pausemusic_service
        device = "racing_device_02"

        # Start call first
        await service.handle_call_started(device, "TALKING")
        assert service.is_interrupted is True

        # Race call ended with device unpair (both call handle_call_ended under the hood)
        res = await asyncio.gather(
            service.handle_call_ended(device),
            service.handle_device_disconnected(device),
        )

        assert res.count(True) == 1
        assert res.count(False) == 1
        assert service.is_interrupted is False
        assert len(service.active_call_devices) == 0

        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and st.is_playing is True

    async def test_multi_device_abrupt_disconnect_storm(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """8 devices have active calls; all 8 abruptly disconnect simultaneously.

        Verify all 8 are cleared and media playback is restored cleanly.
        """
        service = pausemusic_service
        devices = [f"storm_dev_{i}" for i in range(8)]

        for dev in devices:
            await service.handle_call_started(dev, "RINGING")

        assert len(service.active_call_devices) == 8
        assert service.is_interrupted is True

        # Disconnect all 8 simultaneously
        results = await asyncio.gather(
            *[service.handle_device_disconnected(dev) for dev in devices]
        )

        # Exactly 1 disconnect should have triggered the final 1 -> 0 media restoration
        assert results.count(True) == 1
        assert results.count(False) == 7
        assert len(service.active_call_devices) == 0
        assert service.is_interrupted is False

        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and st.is_playing is True

    async def test_disconnect_unrelated_device_during_active_call(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """Disconnecting devices that have no active call must not alter interruption state."""
        service = pausemusic_service

        await service.handle_call_started("active_dev", "RINGING")
        assert service.is_interrupted is True
        assert service.active_call_devices == {"active_dev"}

        # Disconnect 5 unrelated devices
        unrelated = [f"unrelated_{i}" for i in range(5)]
        for dev in unrelated:
            res = await service.handle_device_disconnected(dev)
            assert res is False
            assert service.is_interrupted is True
            assert service.active_call_devices == {"active_dev"}

        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and not st.is_playing

        # Clean up active call
        await service.handle_call_ended("active_dev")
        assert service.is_interrupted is False
        st_final = await mock_mpris.get_player_status("Spotify")
        assert st_final is not None and st_final.is_playing is True

    async def test_eventbus_concurrent_disconnect_and_call_events(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
    ) -> None:
        """EventBus publishes CallStateChangedEvent and DeviceDisconnected concurrently.

        Verify subscriber exception isolation and clean handling.
        """
        plugin = PauseMusicPlugin(event_bus=event_bus)
        # Wire mock services
        service = PauseMusicService(
            config=PauseMusicConfig(action_pause=True, action_resume=True),
            mpris_adapter=mock_mpris,
            audio_service=local_audio,
        )
        plugin.service = service
        await plugin.start()

        # Concurrently publish call ringing for dev1 and dev2, and disconnect for dev1
        events = [
            CallStateChangedEvent(device_id="dev1", old_state="IDLE", new_state="RINGING"),
            CallStateChangedEvent(device_id="dev2", old_state="IDLE", new_state="RINGING"),
            DeviceDisconnected(device_id="dev1", reason="Socket RST"),
        ]

        await asyncio.gather(*[event_bus.publish(ev) for ev in events])

        # dev1 must be disconnected, dev2 must remain active
        assert "dev1" not in service.active_call_devices
        assert "dev2" in service.active_call_devices
        assert service.is_interrupted is True

        # Now disconnect dev2 via EventBus
        await event_bus.publish(DeviceUnpaired(device_id="dev2"))
        assert len(service.active_call_devices) == 0
        assert service.is_interrupted is False

        await plugin.stop()


# ---------------------------------------------------------------------
# Category 4: Plugin Start/Stop Races & Lifecycle
# ---------------------------------------------------------------------


class TestPluginStartStopRaces:
    """Stress tests on PauseMusicPlugin lifecycle under active concurrent traffic."""

    async def test_plugin_stop_racing_with_incoming_call(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
    ) -> None:
        """Stopping plugin concurrently with an arriving CallStateChangedEvent.

        Verify no exceptions escape, no deadlocks, and clean shutdown.
        """
        plugin = PauseMusicPlugin(event_bus=event_bus)
        service = PauseMusicService(
            config=PauseMusicConfig(action_pause=True, action_resume=True),
            mpris_adapter=mock_mpris,
            audio_service=local_audio,
        )
        plugin.service = service
        await plugin.start()

        call_ev = CallStateChangedEvent(
            device_id="race_dev", old_state="IDLE", new_state="RINGING"
        )

        await asyncio.gather(
            plugin.stop(),
            event_bus.publish(call_ev),
        )

        # After stop, plugin must be stopped and media restored
        assert plugin._started is False
        assert len(plugin._unsub_callbacks) == 0

        # Ensure media is restored if it was interrupted right before stop
        if service.is_interrupted:
            await service.restore_media()

        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and st.is_playing is True

    async def test_plugin_stop_during_multi_device_interruption(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
    ) -> None:
        """When plugin.stop() is invoked during an active 3-device interruption,

        it must automatically restore media playback and unmute sinks.
        """
        plugin = PauseMusicPlugin(event_bus=event_bus)
        service = PauseMusicService(
            config=PauseMusicConfig(action_pause=True, action_mute=True, action_resume=True),
            mpris_adapter=mock_mpris,
            audio_service=local_audio,
        )
        plugin.service = service
        await plugin.start()

        # 3 devices active
        await service.handle_call_started("dev1", "TALKING")
        await service.handle_call_started("dev2", "TALKING")
        await service.handle_call_started("dev3", "TALKING")

        assert service.is_interrupted is True
        st_pause = await mock_mpris.get_player_status("Spotify")
        assert st_pause is not None and not st_pause.is_playing

        # Stop plugin
        await plugin.stop()

        # Stop must have invoked restore_media() on the service
        assert plugin._started is False
        st_resume = await mock_mpris.get_player_status("Spotify")
        assert st_resume is not None and st_resume.is_playing is True
        sinks = await local_audio.list_sinks()
        assert any(s.name.endswith("analog-stereo") and not s.muted for s in sinks)

    async def test_rapid_plugin_start_stop_cycles(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
    ) -> None:
        """Rapid start() -> stop() -> start() -> stop() cycles under background event publishing."""
        plugin = PauseMusicPlugin(event_bus=event_bus)
        service = PauseMusicService(
            config=PauseMusicConfig(action_pause=True, action_resume=True),
            mpris_adapter=mock_mpris,
            audio_service=local_audio,
        )
        plugin.service = service

        for cycle in range(10):
            await plugin.start()
            assert plugin._started is True
            assert len(plugin._unsub_callbacks) >= 3

            # Publish event while started
            await event_bus.publish(
                CallStateChangedEvent(
                    device_id=f"cycle_{cycle}", old_state="IDLE", new_state="RINGING"
                )
            )

            await plugin.stop()
            assert plugin._started is False
            assert len(plugin._unsub_callbacks) == 0

        # After all cycles, media should be restored
        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and st.is_playing is True

    async def test_concurrent_plugin_stop_idempotency(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
    ) -> None:
        """Multiple concurrent invocations of plugin.stop()."""
        plugin = PauseMusicPlugin(event_bus=event_bus)
        service = PauseMusicService(
            config=PauseMusicConfig(action_pause=True, action_resume=True),
            mpris_adapter=mock_mpris,
            audio_service=local_audio,
        )
        plugin.service = service
        await plugin.start()

        # Interrupt
        await service.handle_call_started("dev_stop_race", "RINGING")
        assert service.is_interrupted is True

        # Stop 5 times concurrently
        await asyncio.gather(*[plugin.stop() for _ in range(5)])

        assert plugin._started is False
        assert len(plugin._unsub_callbacks) == 0
        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and st.is_playing is True

    async def test_service_stop_cleans_up_and_resumes(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
    ) -> None:
        """Calling service.stop() directly restores media and unsubscribes from EventBus."""
        service = pausemusic_service
        await service.handle_call_started("dev_srv_stop", "RINGING")
        assert service.is_interrupted is True

        await service.stop()
        assert len(service._unsub_callbacks) == 0
        assert service.is_interrupted is False

        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and st.is_playing is True


# ---------------------------------------------------------------------
# Category 5: Reentrancy Safety & Callback Integrity
# ---------------------------------------------------------------------


class TestReentrancyAndCallbackSafety:
    """Stress tests verifying that external callbacks and subscribers calling back

    into PauseMusicService or PauseMusicPlugin do not cause deadlocks.
    """

    async def test_reentrant_eventbus_subscriber_queries_service_during_call_event(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
    ) -> None:
        """An external subscriber to CallStateChangedEvent queries service and plugin status.

        Since EventBus invokes subscribers concurrently via asyncio.gather, this verifies
        status queries do not deadlock while handle_call_state_changed executes.
        """
        service = PauseMusicService(
            config=PauseMusicConfig(action_pause=True, action_resume=True),
            mpris_adapter=mock_mpris,
            audio_service=local_audio,
            event_bus=event_bus,
        )
        plugin = PauseMusicPlugin(event_bus=event_bus, service=service)
        await plugin.start()

        queried_statuses: list[dict[str, Any]] = []

        async def external_observer(event: CallStateChangedEvent) -> None:
            # Query status on service and plugin
            srv_status = service.get_status()
            plugin_status = await plugin.get_status()
            queried_statuses.append({
                "srv_interrupted": srv_status["is_interrupted"],
                "plugin_interrupted": plugin_status["is_interrupted"],
                "active_devices": list(service.active_call_devices),
            })

        event_bus.subscribe(CallStateChangedEvent, external_observer)

        # Publish call event
        await event_bus.publish(
            CallStateChangedEvent(device_id="dev_obs", old_state="IDLE", new_state="RINGING")
        )

        assert len(queried_statuses) >= 1
        assert service.is_interrupted is True

        # Clean up
        await event_bus.publish(
            CallStateChangedEvent(device_id="dev_obs", old_state="RINGING", new_state="IDLE")
        )
        assert service.is_interrupted is False
        await plugin.stop()

    async def test_reentrant_subscriber_triggers_nested_call_state_change(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
    ) -> None:
        """A subscriber receives CallStateChangedEvent for dev1 and publishes for dev2.

        Verify nested/cascading event dispatch completes cleanly without deadlock.
        """
        service = PauseMusicService(
            config=PauseMusicConfig(action_pause=True, action_resume=True),
            mpris_adapter=mock_mpris,
            audio_service=local_audio,
            event_bus=event_bus,
        )

        nested_published = False

        async def cascade_subscriber(event: CallStateChangedEvent) -> None:
            nonlocal nested_published
            if event.device_id == "cascade_dev1" and not nested_published:
                nested_published = True
                await event_bus.publish(
                    CallStateChangedEvent(
                        device_id="cascade_dev2", old_state="IDLE", new_state="RINGING"
                    )
                )

        event_bus.subscribe(CallStateChangedEvent, cascade_subscriber)

        # Fire dev1
        await event_bus.publish(
            CallStateChangedEvent(
                device_id="cascade_dev1", old_state="IDLE", new_state="RINGING"
            )
        )

        # Both devices should now be active
        assert service.is_interrupted is True
        assert "cascade_dev1" in service.active_call_devices
        assert "cascade_dev2" in service.active_call_devices
        assert len(service.active_call_devices) == 2

        # End both
        await event_bus.publish(
            CallStateChangedEvent(
                device_id="cascade_dev1", old_state="RINGING", new_state="IDLE"
            )
        )
        assert service.is_interrupted is True
        await event_bus.publish(
            CallStateChangedEvent(
                device_id="cascade_dev2", old_state="RINGING", new_state="IDLE"
            )
        )
        assert service.is_interrupted is False

    async def test_reentrant_mpris_change_callback_queries_service(
        self,
        local_audio: LocalAudioService,
    ) -> None:
        """MockMprisAdapter dispatches change callback on send_action.

        The change callback queries service.get_status().
        Verify no deadlock occurs during snapshot or resume.
        """
        adapter = MockMprisAdapter(auto_default_player=False)
        await adapter.start()
        adapter._players["CallbackPlayer"] = PlayerState(
            player="CallbackPlayer",
            is_playing=True,
        )

        service = PauseMusicService(
            config=PauseMusicConfig(action_pause=True, action_resume=True),
            mpris_adapter=adapter,
            audio_service=local_audio,
        )

        callback_invocations: list[dict[str, Any]] = []

        async def on_mpris_change(player: str, delta: dict[str, Any]) -> None:
            status = service.get_status()
            callback_invocations.append({
                "player": player,
                "delta": delta,
                "service_interrupted": status["is_interrupted"],
            })

        adapter.set_change_callback(on_mpris_change)

        # Start call: triggers pause -> triggers on_mpris_change
        await service.handle_call_started("dev_cb", "RINGING")
        assert len(callback_invocations) >= 1
        assert service.is_interrupted is True

        # End call: triggers play -> triggers on_mpris_change
        await service.handle_call_ended("dev_cb")
        assert len(callback_invocations) >= 2
        assert service.is_interrupted is False

    async def test_reentrant_eventbus_subscriber_disconnects_device_during_call(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
    ) -> None:
        """A subscriber to CallStateChangedEvent immediately disconnects the device.

        Verify no deadlock occurs between call start and immediate disconnect.
        """
        service = PauseMusicService(
            config=PauseMusicConfig(action_pause=True, action_resume=True),
            mpris_adapter=mock_mpris,
            audio_service=local_audio,
            event_bus=event_bus,
        )

        async def immediate_disconnect_handler(event: CallStateChangedEvent) -> None:
            if event.new_state == "RINGING":
                # Immediately disconnect device
                await service.handle_device_disconnected(event.device_id)

        event_bus.subscribe(CallStateChangedEvent, immediate_disconnect_handler)

        await event_bus.publish(
            CallStateChangedEvent(
                device_id="ephemeral_dev", old_state="IDLE", new_state="RINGING"
            )
        )

        # After both handlers complete, device must be cleaned up
        assert "ephemeral_dev" not in service.active_call_devices
        assert service.is_interrupted is False
        st = await mock_mpris.get_player_status("Spotify")
        assert st is not None and st.is_playing is True

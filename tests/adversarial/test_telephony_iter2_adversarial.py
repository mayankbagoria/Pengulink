"""tests.adversarial.test_telephony_iter2_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Milestone 30 Iteration 2 Adversarial Stress Suite:
Deep Reentrancy, Multi-Device Chaos Storms, Nested Policy Mutation Race
Conditions, Exception Isolation, and CLI --json Resilience.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli, format_call_duration
from pengulink.core.events import (
    CallStateChangedEvent,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
)
from pengulink.core.exceptions import DaemonNotRunningError, IPCError
from pengulink.plugins.mpris.adapter import MockMprisAdapter, PlayerState
from pengulink.services.audio import AudioSink
from pengulink.services.pausemusic import PauseMusicConfig, PauseMusicService
from pengulink.services.telephony import TelephonyService


class RobustMockAudioService:
    """Mock LocalAudioService tracking mute states and call histories."""

    def __init__(self) -> None:
        self.sinks: dict[str, AudioSink] = {}
        self.fail_sink_name: str | None = None

    def add_sink(self, sink: AudioSink) -> None:
        self.sinks[sink.name] = sink

    def list_sinks(self) -> list[AudioSink]:
        return list(self.sinks.values())

    async def set_muted(self, name: str, muted: bool) -> bool:
        if name == self.fail_sink_name:
            raise RuntimeError(f"Simulated audio daemon crash on sink {name}")
        if name in self.sinks:
            self.sinks[name].muted = muted
            return True
        return False

    async def get_sink(self, name: str) -> AudioSink | None:
        return self.sinks.get(name)


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def mock_mpris() -> MockMprisAdapter:
    return MockMprisAdapter(auto_default_player=False)


@pytest.fixture
def mock_audio() -> RobustMockAudioService:
    audio = RobustMockAudioService()
    audio.add_sink(AudioSink(name="sink_speaker", description="Speaker", muted=False))
    audio.add_sink(AudioSink(name="sink_headphones", description="Headphones", muted=False))
    audio.add_sink(AudioSink(name="sink_premute", description="Pre-muted Sink", muted=True))
    return audio


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner(env={"COLUMNS": "160"})


class TestDeepReentrancyAndZeroLockHold:
    """Challenge the Zero-Lock-Hold invariant under nested, multi-level re-entrant execution."""

    async def test_deep_nested_reentrancy_cascade(
        self,
        event_bus: EventBus,
    ) -> None:
        """Adversarial Challenge: 4-level deep reentrant cascade across EventBus and Services.

        Level 1: TelephonyService receives call dev1 (RINGING).
                 EventBus subscriber fires CallStateChangedEvent(dev1).
        Level 2: Subscriber synchronously invokes PauseMusicService.handle_call_started(dev2).
                 PauseMusic calls MPRIS.send_action(player1).
        Level 3: MPRIS adapter invokes TelephonyService.process_call_event(dev3, TALKING).
                 EventBus subscriber fires CallStateChangedEvent(dev3).
        Level 4: Subscriber invokes TelephonyService.get_active_calls() and mutates PauseMusicConfig.

        Any lock hold across any level results in instantaneous deadlock.
        """
        tel_service = TelephonyService(event_bus=event_bus)

        reentrancy_trace: list[str] = []

        class NestedMockMpris(MockMprisAdapter):
            async def send_action(self, player: str, action: str) -> bool:
                reentrancy_trace.append(f"mpris_action:{player}:{action}")
                # Level 3 re-entry: initiate call for dev3 inside MPRIS callback!
                await tel_service.process_call_event("dev3", "talking", "+15553333333")
                return await super().send_action(player, action)

        mpris = NestedMockMpris(auto_default_player=False)
        mpris.add_mock_player(
            "org.mpris.MediaPlayer2.nested",
            PlayerState(player="org.mpris.MediaPlayer2.nested", is_playing=True),
        )

        pm_service = PauseMusicService(mpris_adapter=mpris, event_bus=event_bus)

        async def on_call_state_changed(event: CallStateChangedEvent) -> None:
            reentrancy_trace.append(f"event:{event.device_id}:{event.new_state}")
            if event.device_id == "dev1" and event.new_state == "RINGING":
                # Level 2 re-entry: PauseMusic handle_call_started inside subscriber!
                await pm_service.handle_call_started("dev2")
            elif event.device_id == "dev3" and event.new_state == "TALKING":
                # Level 4 re-entry: Query active calls and mutate config!
                active = tel_service.get_active_calls()
                reentrancy_trace.append(f"level4_active_count:{len(active)}")
                pm_service.update_config(action="mute")

        event_bus.subscribe(CallStateChangedEvent, on_call_state_changed)

        # Trigger Level 1: Must complete within 2.0s without deadlocking
        await asyncio.wait_for(
            tel_service.process_call_event("dev1", "ringing", "+15551111111"),
            timeout=2.0,
        )

        # Verify all 4 levels executed cleanly
        assert "event:dev1:RINGING" in reentrancy_trace
        assert "mpris_action:org.mpris.MediaPlayer2.nested:Pause" in reentrancy_trace
        assert "event:dev3:TALKING" in reentrancy_trace
        assert any(t.startswith("level4_active_count:") for t in reentrancy_trace)

        # Verify internal states converged
        active_calls = {c["device_id"]: c["state"] for c in tel_service.get_active_calls()}
        assert active_calls["dev1"] == "RINGING"
        assert active_calls["dev3"] == "TALKING"
        assert pm_service.is_interrupted is True
        assert "dev2" in pm_service.active_call_devices


class TestMultiDeviceChaosStormWithPolicySwapping:
    """Stress test with high-concurrency call churn and simultaneous policy swapping."""

    async def test_25_device_call_storm_with_chaotic_policy_mutations(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
        mock_audio: RobustMockAudioService,
    ) -> None:
        """Adversarial Challenge: 25 devices hammering calls while policy is mutated concurrently.

        25 parallel coroutines start calls, transition to talking, end calls, disconnect, or unpair.
        5 policy-mutator coroutines continuously reconfigure condition, action, and resume.
        Verifies:
        - Zero deadlocks or race conditions.
        - Proper cleanup upon device disconnection / unpair.
        - Final state converges: all calls ended, media restored when resume=True.
        """
        # Populate media players
        for i in range(3):
            mock_mpris.add_mock_player(
                f"org.mpris.MediaPlayer2.p{i}",
                PlayerState(player=f"org.mpris.MediaPlayer2.p{i}", is_playing=True),
            )
        # One player was already paused
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.stopped",
            PlayerState(player="org.mpris.MediaPlayer2.stopped", is_playing=False),
        )

        cfg = PauseMusicConfig(action_pause=True, action_mute=True, action_resume=True)
        pm_service = PauseMusicService(
            config=cfg,
            mpris_adapter=mock_mpris,
            audio_service=mock_audio,  # type: ignore
            event_bus=event_bus,
        )
        tel_service = TelephonyService(event_bus=event_bus)

        stop_chaos = asyncio.Event()

        # Policy mutator workers
        async def policy_mutator(worker_id: int) -> None:
            actions = ["pause", "mute", "both"]
            conditions = ["ringing", "talking"]
            while not stop_chaos.is_set():
                act = random.choice(actions)
                cond = random.choice(conditions)
                pm_service.update_config(action=act, condition=cond)
                await asyncio.sleep(0.001)

        mutator_tasks = [asyncio.create_task(policy_mutator(i)) for i in range(5)]

        # 25 device call workers
        async def device_lifecycle(dev_id: str) -> None:
            # 1. Ringing
            await tel_service.process_call_event(dev_id, "ringing", "+15550009999")
            await pm_service.handle_call_started(dev_id)
            await asyncio.sleep(0.002)

            # 2. Talking
            await tel_service.process_call_event(dev_id, "talking", "+15550009999")
            await asyncio.sleep(0.002)

            # Randomly either cleanly end, disconnect, or unpair
            action_choice = random.choice(["end", "disconnect", "unpair"])
            if action_choice == "end":
                await tel_service.process_call_event(dev_id, "idle")
                await pm_service.handle_call_ended(dev_id)
            elif action_choice == "disconnect":
                await event_bus.publish(DeviceDisconnected(device_id=dev_id))
            else:
                await event_bus.publish(DeviceUnpaired(device_id=dev_id))

        dev_tasks = [device_lifecycle(f"chaos_dev_{i:02d}") for i in range(25)]

        # Run with strict 10.0s timeout
        await asyncio.wait_for(asyncio.gather(*dev_tasks), timeout=10.0)

        # Stop policy mutators
        stop_chaos.set()
        await asyncio.gather(*mutator_tasks)

        # Ensure all active calls are cleaned up
        assert len(tel_service.get_active_calls()) == 0
        assert len(pm_service.active_call_devices) == 0

        # Now test that calling restore_media() ensures pre-call playing players are resumed
        pm_service.update_config(action_resume=True)
        await pm_service.restore_media()

        # All 3 initially playing players must be playing
        for i in range(3):
            st = await mock_mpris.get_player_status(f"org.mpris.MediaPlayer2.p{i}")
            assert st.is_playing is True, f"Player p{i} must be resumed!"

        # Stopped player must remain stopped
        st_stopped = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.stopped")
        assert st_stopped.is_playing is False, "Stopped player must remain stopped!"

        # Pre-muted sink must remain muted
        sink_premute = await mock_audio.get_sink("sink_premute")
        assert sink_premute.muted is True, "Pre-muted sink must remain muted!"


class TestExceptionSafetyAndPartialFailureIsolation:
    """Stress test exception handling when underlying subsystems throw errors."""

    async def test_resumption_continues_when_one_player_fails(
        self,
        event_bus: EventBus,
    ) -> None:
        """Adversarial Challenge: One MPRIS player raises an exception during send_action('Play').

        Verify that remaining players are still resumed and the exception does not
        leave PauseMusicService in an inconsistent or permanently locked state.
        """
        class FlakyMockMpris(MockMprisAdapter):
            async def send_action(self, player: str, action: str) -> bool:
                if player == "org.mpris.MediaPlayer2.broken" and action == "Play":
                    raise ConnectionError("D-Bus communication severed for broken player")
                return await super().send_action(player, action)

        mpris = FlakyMockMpris(auto_default_player=False)
        mpris.add_mock_player(
            "org.mpris.MediaPlayer2.healthy1",
            PlayerState(player="org.mpris.MediaPlayer2.healthy1", is_playing=True),
        )
        mpris.add_mock_player(
            "org.mpris.MediaPlayer2.broken",
            PlayerState(player="org.mpris.MediaPlayer2.broken", is_playing=True),
        )
        mpris.add_mock_player(
            "org.mpris.MediaPlayer2.healthy2",
            PlayerState(player="org.mpris.MediaPlayer2.healthy2", is_playing=True),
        )

        pm_service = PauseMusicService(mpris_adapter=mpris, event_bus=event_bus)

        # Call starts: all 3 are paused
        await pm_service.handle_call_started("dev1")
        assert (await mpris.get_player_status("org.mpris.MediaPlayer2.healthy1")).is_playing is False
        assert (await mpris.get_player_status("org.mpris.MediaPlayer2.broken")).is_playing is False
        assert (await mpris.get_player_status("org.mpris.MediaPlayer2.healthy2")).is_playing is False

        # Call ends: broken player will raise ConnectionError
        # The service must catch it, log it, and continue resuming healthy players!
        res = await pm_service.handle_call_ended("dev1")
        assert res is True
        assert pm_service.is_interrupted is False

        # Verify healthy players were safely resumed
        assert (await mpris.get_player_status("org.mpris.MediaPlayer2.healthy1")).is_playing is True
        assert (await mpris.get_player_status("org.mpris.MediaPlayer2.healthy2")).is_playing is True
        # Lock must NOT remain held
        assert pm_service._lock.locked() is False

    async def test_audio_sink_exception_isolation_during_resume(
        self,
        event_bus: EventBus,
        mock_audio: RobustMockAudioService,
    ) -> None:
        """Adversarial Challenge: Audio sink failure during unmute does not block other sinks."""
        cfg = PauseMusicConfig(action_pause=False, action_mute=True, action_resume=True)
        pm_service = PauseMusicService(
            config=cfg,
            audio_service=mock_audio,  # type: ignore
            event_bus=event_bus,
        )

        # Call starts normally: both sinks are muted
        await pm_service.handle_call_started("dev1")
        assert (await mock_audio.get_sink("sink_speaker")).muted is True
        assert (await mock_audio.get_sink("sink_headphones")).muted is True

        # Now inject failure on sink_speaker during the unmute phase
        mock_audio.fail_sink_name = "sink_speaker"

        # Call ends: speaker unmute will fail with RuntimeError, but headphones must still unmute!
        res = await pm_service.handle_call_ended("dev1")
        assert res is True

        # Speaker remains muted due to simulated daemon error, headphones safely unmuted!
        assert (await mock_audio.get_sink("sink_speaker")).muted is True
        assert (await mock_audio.get_sink("sink_headphones")).muted is False
        assert pm_service._lock.locked() is False

    async def test_audio_sink_mute_loop_abort_behavior_analysis(
        self,
        event_bus: EventBus,
        mock_audio: RobustMockAudioService,
    ) -> None:
        """Adversarial Analysis: If set_muted fails during call arrival (mute phase),
        verify error is safely caught by COORDINATION_EXCEPTIONS without crashing or locking.
        """
        mock_audio.fail_sink_name = "sink_speaker"

        cfg = PauseMusicConfig(action_pause=False, action_mute=True, action_resume=True)
        pm_service = PauseMusicService(
            config=cfg,
            audio_service=mock_audio,  # type: ignore
            event_bus=event_bus,
        )

        # Call starts: sink_speaker fails, error caught, lock released
        res = await pm_service.handle_call_started("dev1")
        assert res is True
        assert pm_service._lock.locked() is False

        # Ending call cleans up safely without deadlocks
        res_end = await pm_service.handle_call_ended("dev1")
        assert res_end is True
        assert pm_service._lock.locked() is False


class TestCLIJsonErrorFormattingAdversarial:
    """Stress tests verifying strict JSON output on errors when --json is used."""

    def test_cli_json_error_handling_across_all_exception_types(
        self,
        cli_runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Adversarial Challenge: Verify --json outputs valid JSON on DaemonNotRunning, IPCError, and generic exceptions.

        Tests multiple invocation patterns:
        - pengulink --json telephony calls
        - pengulink telephony calls --json
        - pengulink --json pausemusic status
        - pengulink pausemusic configure --json
        """
        # Test 1: DaemonNotRunningError
        def mock_daemon_not_running(*args: Any, **kwargs: Any) -> Any:
            raise DaemonNotRunningError("Daemon socket /run/user/1000/pengulink.sock not found")

        monkeypatch.setattr("pengulink.cli.main.PenguLinkIPCClient.call_sync", mock_daemon_not_running)

        res1 = cli_runner.invoke(cli, ["--json", "telephony", "calls"])
        assert res1.exit_code == 1
        data1 = json.loads(res1.output)
        assert data1["success"] is False
        assert "not found" in data1["error"]

        res2 = cli_runner.invoke(cli, ["telephony", "calls", "--json"])
        assert res2.exit_code == 1
        data2 = json.loads(res2.output)
        assert data2["success"] is False

        # Test 2: IPCError with structured error code
        def mock_ipc_error(*args: Any, **kwargs: Any) -> Any:
            err = IPCError("Device not paired or unknown")
            err.code = -32017
            raise err

        monkeypatch.setattr("pengulink.cli.main.PenguLinkIPCClient.call_sync", mock_ipc_error)

        res3 = cli_runner.invoke(cli, ["telephony", "mute-ringer", "dev_unknown", "--json"])
        assert res3.exit_code == 1
        data3 = json.loads(res3.output)
        assert data3["success"] is False
        assert data3["code"] == -32017
        assert "Device not paired" in data3["error"]

        # Test 3: Generic unexpected exception
        def mock_unexpected_error(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Unexpected internal crash inside kernel interface")

        monkeypatch.setattr("pengulink.cli.main.PenguLinkIPCClient.call_sync", mock_unexpected_error)

        res4 = cli_runner.invoke(cli, ["--json", "pausemusic", "status"])
        assert res4.exit_code == 1
        data4 = json.loads(res4.output)
        assert data4["success"] is False
        assert "Unexpected internal crash" in data4["error"]

    def test_cli_format_call_duration_extreme_fuzz(self) -> None:
        """Adversarial Challenge: Extreme fuzzing of format_call_duration."""
        # Non-finite and exotic inputs
        assert format_call_duration(float("nan")) == "00:00"
        assert format_call_duration(float("inf")) == "00:00"
        assert format_call_duration(float("-inf")) == "00:00"
        assert format_call_duration(None) == "00:00"
        assert format_call_duration("not_a_number") == "00:00"
        assert format_call_duration(-999999.99) == "00:00"
        assert format_call_duration(object()) == "00:00"
        assert format_call_duration([]) == "00:00"
        assert format_call_duration({"duration": 12}) == "00:00"

        # Valid high values
        assert format_call_duration(86400) == "24:00:00"
        assert format_call_duration(3600 * 999 + 59 * 60 + 59) == "999:59:59"

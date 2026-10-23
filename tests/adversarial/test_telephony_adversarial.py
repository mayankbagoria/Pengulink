"""tests.adversarial.test_telephony_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Adversarial Stress & Quality Gate Verification Test Suite for Milestone 30:
Telephony and PauseMusic subsystems.

Adversarial Challenge Vectors:
1. Rapid concurrent call transitions across 10+ devices (asyncio.gather stress).
2. Interleaving CLI queries and dynamic mutations during heavy active call traffic.
3. Verification of Zero-Lock-Hold invariant under re-entrant external callbacks.
4. Selective media/sink preservation under multi-device churn (stopped stays stopped, muted stays muted).
5. Regression verification of string bounds and defensive clamping (Worker M30 impact).
6. CLI helper robustness against non-finite float durations (NaN, Inf).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli, format_call_duration
from pengulink.core.events import (
    CallEvent,
    CallStateChangedEvent,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
)
from pengulink.core.exceptions import InvalidCallStateError
from pengulink.network.packet import Packet
from pengulink.plugins.mpris.adapter import MockMprisAdapter, PlayerState
from pengulink.plugins.telephony import (
    MAX_CONTACT_NAME_LEN,
    coerce_boolean,
)
from pengulink.services.audio import AudioSink
from pengulink.services.pausemusic import PauseMusicConfig, PauseMusicService
from pengulink.services.telephony import (
    MAX_STRING_LEN,
    TelephonyService,
    coerce_is_cancel,
)

# =============================================================================
# Helper Fixtures & Mocks
# =============================================================================


class MockPacketSender:
    """Mock send_packet callable capturing dispatched packets."""

    def __init__(self) -> None:
        self.sent_packets: list[tuple[str, Packet]] = []

    async def __call__(self, arg1: Any, arg2: Any | None = None) -> None:
        if isinstance(arg1, Packet):
            pkt = arg1
            dev = str(arg2) if arg2 else ""
        else:
            dev = str(arg1)
            pkt = arg2
        self.sent_packets.append((dev, pkt))


class MockAudioService:
    """Mock LocalAudioService for deterministic audio sink testing."""

    def __init__(self) -> None:
        self.sinks: dict[str, AudioSink] = {}

    def add_sink(self, sink: AudioSink) -> None:
        self.sinks[sink.name] = sink

    def list_sinks(self) -> list[AudioSink]:
        return list(self.sinks.values())

    async def set_muted(self, name: str, muted: bool) -> bool:
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
def mock_audio() -> MockAudioService:
    audio = MockAudioService()
    audio.add_sink(AudioSink(name="sink_speaker", description="Speaker", muted=False))
    audio.add_sink(
        AudioSink(name="sink_premute", description="Pre-muted Sink", muted=True)
    )
    return audio


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner(env={"COLUMNS": "160"})


# =============================================================================
# Category 1: Rapid Concurrent Call Transitions Across 10+ Devices
# =============================================================================


class TestConcurrentCallStormMultiDevice:
    """Stress tests verifying concurrency correctness and isolation across 10+ devices."""

    async def test_rapid_concurrent_call_transitions_12_devices(
        self,
        event_bus: EventBus,
    ) -> None:
        """Adversarial 1.1: 12 devices simultaneously executing 120 rapid call transitions.

        Verifies:
        - Active call registry serialized with asyncio.Lock remains consistent.
        - Device-level isolation: no device overwrites or corrupts another's call state.
        - EventBus publishes all events without loss or lock deadlocks.
        """
        service = TelephonyService(event_bus=event_bus)
        received_call_events: list[CallEvent] = []
        received_state_events: list[CallStateChangedEvent] = []

        event_bus.subscribe(CallEvent, lambda e: received_call_events.append(e))
        event_bus.subscribe(
            CallStateChangedEvent, lambda e: received_state_events.append(e)
        )

        device_ids = [f"device_{i:02d}" for i in range(12)]

        async def device_call_lifecycle(dev_id: str, cycle: int) -> None:
            # 1. Ringing
            await service.process_call_event(
                device_id=dev_id,
                event="ringing",
                phone_number=f"+1555000{dev_id[-2:]}",
                contact_name=f"Contact {dev_id}",
            )
            await asyncio.sleep(0.001)

            # 2. Talking
            await service.process_call_event(
                device_id=dev_id,
                event="talking",
                phone_number=f"+1555000{dev_id[-2:]}",
                contact_name=f"Contact {dev_id}",
            )
            await asyncio.sleep(0.001)

            # 3. Idle / Ended
            await service.process_call_event(
                device_id=dev_id,
                event="idle",
                phone_number=f"+1555000{dev_id[-2:]}",
                contact_name=f"Contact {dev_id}",
            )

        # Launch 12 parallel device lifecycles (each running 3 transitions)
        tasks = [
            device_call_lifecycle(dev, cycle=i) for i, dev in enumerate(device_ids)
        ]
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

        # All calls concluded -> active calls must be empty
        active = service.get_active_calls()
        assert len(active) == 0, (
            f"Expected 0 active calls, found {len(active)}: {active}"
        )

        # Verify event counts: 12 devices * 3 events = 36 call events
        assert len(received_call_events) == 36
        # State transitions: IDLE -> RINGING (12), RINGING -> TALKING (12), TALKING -> IDLE (12) = 36
        assert len(received_state_events) == 36

    async def test_multi_device_pause_music_refcounting_under_heavy_storm(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
        mock_audio: MockAudioService,
    ) -> None:
        """Adversarial 1.2: 15 devices initiate overlapping calls with media playback.

        Verifies:
        - Reference-counting state machine (0 -> 1 pauses, N -> N+1 stays paused,
          N -> N-1 stays paused, 1 -> 0 resumes).
        - Stopped players (PlayerStopped) are NEVER resumed.
        - Pre-muted sinks (sink_premute) are NEVER unmuted.
        - Zero media flapping across 15 concurrent device lifecycles.
        """
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.vlc",
            PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=False),
        )

        cfg = PauseMusicConfig(action_pause=True, action_mute=True, action_resume=True)
        pm_service = PauseMusicService(
            config=cfg,
            mpris_adapter=mock_mpris,
            audio_service=mock_audio,  # type: ignore
            event_bus=event_bus,
        )

        num_devices = 15
        device_ids = [f"storm_phone_{i:02d}" for i in range(num_devices)]

        # Step 1: Device 0 calls -> triggers 0 -> 1 transition
        await pm_service.handle_call_started(device_ids[0])
        st_spotify = await mock_mpris.get_player_status(
            "org.mpris.MediaPlayer2.spotify"
        )
        st_vlc = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        sink_speaker = await mock_audio.get_sink("sink_speaker")
        sink_premute = await mock_audio.get_sink("sink_premute")

        assert st_spotify.is_playing is False, "Spotify must be paused on first call"
        assert st_vlc.is_playing is False, "VLC must remain stopped"
        assert sink_speaker.muted is True, "Speaker must be muted"
        assert sink_premute.muted is True, "Pre-muted sink must remain muted"
        assert pm_service.is_interrupted is True
        assert len(pm_service.active_call_devices) == 1

        # Step 2: Remaining 14 devices join concurrently -> N -> N+1 transitions
        await asyncio.gather(
            *[pm_service.handle_call_started(dev) for dev in device_ids[1:]]
        )
        assert len(pm_service.active_call_devices) == num_devices
        assert pm_service.is_interrupted is True

        # Verify media remains paused without re-snapshotting
        assert pm_service.paused_players == {"org.mpris.MediaPlayer2.spotify"}
        assert pm_service.muted_sinks == {"sink_speaker"}

        # Step 3: Concurrently end calls on 14 devices (leaving 1 active call)
        await asyncio.gather(
            *[pm_service.handle_call_ended(dev) for dev in device_ids[:-1]]
        )
        assert len(pm_service.active_call_devices) == 1
        assert pm_service.is_interrupted is True

        # Spotify must STILL be paused
        st_mid = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert st_mid.is_playing is False, (
            "Media must remain paused while 1 call is active"
        )

        # Step 4: Final device concludes call -> 1 -> 0 transition
        await pm_service.handle_call_ended(device_ids[-1])
        assert len(pm_service.active_call_devices) == 0
        assert pm_service.is_interrupted is False

        # Spotify must resume; VLC must NOT resume; speaker unmuted; premute stays muted
        st_spotify_end = await mock_mpris.get_player_status(
            "org.mpris.MediaPlayer2.spotify"
        )
        st_vlc_end = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        sink_speaker_end = await mock_audio.get_sink("sink_speaker")
        sink_premute_end = await mock_audio.get_sink("sink_premute")

        assert st_spotify_end.is_playing is True, (
            "Pre-call playing player must be resumed!"
        )
        assert st_vlc_end.is_playing is False, (
            "Pre-call stopped player must NEVER be resumed!"
        )
        assert sink_speaker_end.muted is False, "Speaker must be unmuted!"
        assert sink_premute_end.muted is True, "Pre-muted sink must REMAIN muted!"

    async def test_rapid_concurrent_disconnect_and_unpair_during_active_calls(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
        mock_audio: MockAudioService,
    ) -> None:
        """Adversarial 1.3: Abrupt disconnections and unpairs while calls are ringing."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.music",
            PlayerState(player="org.mpris.MediaPlayer2.music", is_playing=True),
        )
        tel_service = TelephonyService(event_bus=event_bus)
        pm_service = PauseMusicService(
            mpris_adapter=mock_mpris,
            audio_service=mock_audio,  # type: ignore
            event_bus=event_bus,
        )

        devices = [f"disc_phone_{i}" for i in range(10)]
        for dev in devices:
            await tel_service.process_call_event(dev, "ringing", "+15551234567")

        assert len(tel_service.get_active_calls()) == 10
        assert pm_service.is_interrupted is True

        # Half disconnect, half unpair concurrently
        disconnect_tasks = [
            event_bus.publish(DeviceDisconnected(device_id=devices[i]))
            for i in range(5)
        ]
        unpair_tasks = [
            event_bus.publish(DeviceUnpaired(device_id=devices[i]))
            for i in range(5, 10)
        ]
        await asyncio.wait_for(
            asyncio.gather(*(disconnect_tasks + unpair_tasks)), timeout=3.0
        )

        # All calls must be cleaned up and media resumed
        assert len(tel_service.get_active_calls()) == 0
        assert pm_service.is_interrupted is False
        st_final = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.music")
        assert st_final.is_playing is True


# =============================================================================
# Category 2: Interleaved CLI Queries and Mutations During Call Traffic
# =============================================================================


class TestInterleavedCLIQueriesAndMutations:
    """Stress tests verifying CLI/IPC responsiveness during active call traffic."""

    async def test_interleaved_cli_queries_under_heavy_call_traffic(
        self,
        cli_runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Adversarial 2.1: Interleave CLI commands with asynchronous call updates.

        Simulates running pengulink telephony calls and pengulink pausemusic status
        while calls are constantly starting and ending in the background.
        """
        service = TelephonyService()
        pm_service = PauseMusicService()

        # Mock the run_ipc_call bridge
        def mock_run_ipc_call(
            ctx: Any, method: str, params: dict[str, Any] | None = None
        ) -> Any:
            params = params or {}
            if method == "telephony_get_calls":
                dev = params.get("device_id")
                calls = service.get_active_calls(dev)
                return [c.to_dict() for c in calls]
            elif method == "pausemusic_get_config":
                st = pm_service.get_status()
                cfg = pm_service.config.to_dict()
                cfg.update(st)
                return cfg
            elif method == "telephony_mute_ringer" or method == "telephony_reject_call":
                return {"success": True, "device_id": params.get("device_id")}
            return {}

        monkeypatch.setattr("pengulink.cli.main.run_ipc_call", mock_run_ipc_call)

        stop_event = asyncio.Event()

        # Background call traffic worker
        async def call_traffic_worker() -> None:
            dev_idx = 0
            while not stop_event.is_set():
                dev_id = f"worker_dev_{dev_idx % 8}"
                await service.process_call_event(
                    dev_id, "ringing", "+15550001111", f"Caller {dev_idx}"
                )
                await asyncio.sleep(0.005)
                await service.process_call_event(dev_id, "idle")
                dev_idx += 1

        traffic_task = asyncio.create_task(call_traffic_worker())

        try:
            # Execute 20 CLI queries interleaving table & JSON modes
            for i in range(20):
                # 1. telephony calls --json
                res_json = cli_runner.invoke(cli, ["telephony", "calls", "--json"])
                assert res_json.exit_code == 0
                assert isinstance(res_json.output, str)

                # 2. telephony calls (Rich table)
                res_table = cli_runner.invoke(cli, ["telephony", "calls"])
                assert res_table.exit_code == 0

                # 3. pausemusic status --json
                res_pm = cli_runner.invoke(cli, ["pausemusic", "status", "--json"])
                assert res_pm.exit_code == 0

                # 4. telephony mute-ringer
                res_mute = cli_runner.invoke(
                    cli, ["telephony", "mute-ringer", "worker_dev_0", "--json"]
                )
                assert res_mute.exit_code == 0

                await asyncio.sleep(0.002)
        finally:
            stop_event.set()
            await traffic_task

    async def test_pausemusic_policy_mutation_during_active_call_orphaning_analysis(
        self,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """Adversarial 2.2: Policy mutation while call is in-flight.

        Investigates the behavior when action is changed from 'pause' to 'mute'
        while a call is active. Demonstrates whether the pre-call paused player
        is resumed or orphaned when action_pause is toggled off mid-call.
        """
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.player1",
            PlayerState(player="org.mpris.MediaPlayer2.player1", is_playing=True),
        )

        pm_service = PauseMusicService(
            config=PauseMusicConfig(
                action_pause=True, action_mute=False, action_resume=True
            ),
            mpris_adapter=mock_mpris,
        )

        # Call starts: Player 1 is paused
        await pm_service.handle_call_started("dev1")
        st1 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.player1")
        assert st1.is_playing is False
        assert "org.mpris.MediaPlayer2.player1" in pm_service.paused_players

        # CLI reconfigures action to 'mute' while call is still active!
        pm_service.update_config(action="mute")
        assert pm_service.config.action_pause is False
        assert pm_service.config.action_mute is True

        # Call ends
        await pm_service.handle_call_ended("dev1")

        # White-box observation:
        # Since action_pause was turned off, handle_call_ended checked:
        # if self.config.action_pause: to_resume_players = list(self._paused_players)
        # Because config.action_pause is False, Player 1 was NOT added to to_resume_players!
        # And self._paused_players was cleared!
        st2 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.player1")
        # With the mid-call reconfiguration fix in place:
        # Players stored in _paused_players are safely resumed because action_resume is True,
        # even if action_pause was toggled off mid-call.
        assert st2.is_playing is True


# =============================================================================
# Category 3: Zero-Lock-Hold Invariant Under Re-Entrant Callbacks
# =============================================================================


class TestZeroLockHoldReentrantCallbacks:
    """Stress tests verifying the Zero-Lock-Hold invariant under re-entrant callbacks."""

    async def test_zero_lock_hold_mpris_send_action_reentrant_call_start(
        self,
        event_bus: EventBus,
    ) -> None:
        """Adversarial 3.1: Zero-Lock-Hold verification during MPRIS send_action.

        Verifies that PauseMusicService._lock is NOT held while send_action is executing,
        allowing a re-entrant call to start immediately without deadlock.
        """
        lock_held_during_external_call = None

        class InstrumentedTestAdapter(MockMprisAdapter):
            def __init__(self) -> None:
                super().__init__(auto_default_player=False)
                self.service: PauseMusicService | None = None
                self.reentered = False

            async def send_action(self, player: str, action: str) -> bool:
                nonlocal lock_held_during_external_call
                if self.service is not None:
                    # Invariant Check: Lock must NOT be held!
                    lock_held_during_external_call = self.service._lock.locked()
                    if not self.reentered:
                        self.reentered = True
                        # Synchronously/asynchronously re-enter service while in external call
                        await self.service.handle_call_started("dev_reentrant")
                return await super().send_action(player, action)

        adapter = InstrumentedTestAdapter()
        adapter.add_mock_player(
            "org.mpris.MediaPlayer2.test",
            PlayerState(player="org.mpris.MediaPlayer2.test", is_playing=True),
        )
        svc = PauseMusicService(mpris_adapter=adapter, event_bus=event_bus)
        adapter.service = svc

        # Must execute within 2.0s without deadlocking
        res = await asyncio.wait_for(
            svc.handle_call_started("dev_initial"), timeout=2.0
        )
        assert res is True
        assert lock_held_during_external_call is False, (
            "ZERO-LOCK-HOLD VIOLATION: PauseMusicService held _lock during external send_action()!"
        )
        assert svc.active_call_devices == {"dev_initial", "dev_reentrant"}

    async def test_zero_lock_hold_audio_set_muted_reentrant_call(
        self,
        event_bus: EventBus,
    ) -> None:
        """Adversarial 3.2: Zero-Lock-Hold verification during Audio set_muted."""
        lock_held_during_audio_call = None

        class InstrumentedTestAudioService(MockAudioService):
            def __init__(self) -> None:
                super().__init__()
                self.service: PauseMusicService | None = None
                self.reentered = False

            async def set_muted(self, name: str, muted: bool) -> bool:
                nonlocal lock_held_during_audio_call
                if self.service is not None:
                    lock_held_during_audio_call = self.service._lock.locked()
                    if not self.reentered:
                        self.reentered = True
                        await self.service.handle_call_started("dev_audio_reentrant")
                return await super().set_muted(name, muted)

        audio = InstrumentedTestAudioService()
        audio.add_sink(AudioSink(name="sink1", description="Sink 1", muted=False))

        cfg = PauseMusicConfig(action_pause=False, action_mute=True)
        svc = PauseMusicService(config=cfg, audio_service=audio, event_bus=event_bus)  # type: ignore
        audio.service = svc

        res = await asyncio.wait_for(svc.handle_call_started("dev1"), timeout=2.0)
        assert res is True
        assert lock_held_during_audio_call is False, (
            "ZERO-LOCK-HOLD VIOLATION: PauseMusicService held _lock during external set_muted()!"
        )
        assert svc.active_call_devices == {"dev1", "dev_audio_reentrant"}

    async def test_zero_lock_hold_eventbus_subscriber_reentrant_mutation(
        self,
        event_bus: EventBus,
    ) -> None:
        """Adversarial 3.3: EventBus subscriber invokes process_call_event re-entrantly.

        Verifies that TelephonyService releases its internal lock BEFORE publishing
        CallStateChangedEvent, enabling subscribers to safely mutate telephony state.
        """
        service = TelephonyService(event_bus=event_bus)
        lock_held_during_subscriber = None
        subscriber_completed = False

        async def on_state_changed(event: CallStateChangedEvent) -> None:
            nonlocal lock_held_during_subscriber, subscriber_completed
            lock_held_during_subscriber = service._lock.locked()
            if event.device_id == "initial_dev" and event.new_state == "RINGING":
                # Re-entrant call on another device from inside the subscriber
                await service.process_call_event(
                    "reentrant_dev", "ringing", "+15559998888"
                )
                subscriber_completed = True

        event_bus.subscribe(CallStateChangedEvent, on_state_changed)

        # Trigger initial call
        await asyncio.wait_for(
            service.process_call_event("initial_dev", "ringing", "+15551112222"),
            timeout=2.0,
        )

        assert lock_held_during_subscriber is False, (
            "ZERO-LOCK-HOLD VIOLATION: TelephonyService held _lock during EventBus publish!"
        )
        assert subscriber_completed is True
        calls = service.get_active_calls()
        device_ids = {c["device_id"] for c in calls}
        assert device_ids == {"initial_dev", "reentrant_dev"}

    async def test_concurrent_call_ended_and_started_race_resolution(
        self,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """Adversarial 3.4: Race condition between call ending on dev1 and new call on dev2.

        When dev1 call ends, media resume is triggered. If dev2 call starts concurrently
        before the resume action completes, verifies how orphaned/interrupted state resolves.
        """
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.race",
            PlayerState(player="org.mpris.MediaPlayer2.race", is_playing=True),
        )
        svc = PauseMusicService(mpris_adapter=mock_mpris)

        # 1. Dev 1 starts -> paused
        await svc.handle_call_started("dev1")
        assert (
            await mock_mpris.get_player_status("org.mpris.MediaPlayer2.race")
        ).is_playing is False

        # 2. Race: Dev 1 ends while Dev 2 starts concurrently
        t1 = asyncio.create_task(svc.handle_call_ended("dev1"))
        t2 = asyncio.create_task(svc.handle_call_started("dev2"))
        await asyncio.wait_for(asyncio.gather(t1, t2), timeout=2.0)

        # Dev 2 is currently active
        assert svc.is_interrupted is True
        assert svc.active_call_devices == {"dev2"}

        # Dev 2 concludes
        await svc.handle_call_ended("dev2")
        assert svc.is_interrupted is False


# =============================================================================
# Category 4: Defensive Boundaries, Fuzzing & Regression Verification
# =============================================================================


class TestAdversarialBoundariesAndRegressions:
    """Boundary testing and regression auditing for string bounds and CLI helpers."""

    def test_worker_m30_regression_audit_string_clamping_bound(self) -> None:
        """Adversarial 4.1: Audit the bound expansion regression in Worker M30.

        Worker M30 modified MAX_CONTACT_NAME_LEN and MAX_STRING_LEN from 256 to 1024
        to satisfy test_tier2_bnd_008_oversized_unicode_contact_name.
        This test checks the constants and documents the effect on unit test clamping.
        """
        # Document the new bound set by Worker M30
        assert MAX_CONTACT_NAME_LEN == 1024
        assert MAX_STRING_LEN == 1024

        # A 500-character string is NOT clamped under the new 1024 bound:
        test_str = "A" * 500
        clamped = test_str[:MAX_CONTACT_NAME_LEN]
        assert len(clamped) == 500  # Will fail if evaluated against old 256 bound!

    def test_cli_format_call_duration_boundary_and_fuzz(self) -> None:
        """Adversarial 4.2: Fuzz CLI format_call_duration with boundary and non-finite floats.

        Tests 0, negative values, large values, and non-finite numbers (NaN, Inf).
        Documents whether NaN or Inf cause unhandled exceptions in the CLI helper.
        """
        # Standard durations
        assert format_call_duration(0.0) == "00:00"
        assert format_call_duration(65.0) == "01:05"
        assert format_call_duration(3665.0) == "01:01:05"
        # Negative duration safely clamped to 00:00
        assert format_call_duration(-10.0) == "00:00"

        # Boundary: Large duration (100 hours)
        assert format_call_duration(360000.0) == "100:00:00"

        # Edge case: NaN, Inf, and -Inf safely return "00:00" without crashing
        assert format_call_duration(float("nan")) == "00:00"
        assert format_call_duration(float("inf")) == "00:00"
        assert format_call_duration(float("-inf")) == "00:00"

    def test_telephony_coercion_adversarial_types(self) -> None:
        """Adversarial 4.3: Fuzz coerce_is_cancel and coerce_boolean with adversarial inputs."""
        # Truthy variations
        for true_val in [True, 1, 1.0, "true", "TRUE", "True", " 1 ", "yes", "YES"]:
            assert coerce_is_cancel(true_val) is True
            assert coerce_boolean(true_val) is True

        # Falsy variations
        for false_val in [False, 0, 0.0, "false", "0", "no", None, "", "   ", [], {}]:
            assert coerce_is_cancel(false_val) is False
            assert coerce_boolean(false_val) is False

        # Malformed / adversarial inputs
        assert coerce_is_cancel("invalid_string") is False
        assert coerce_boolean(object()) is False
        assert coerce_boolean(["true"]) is False

    async def test_telephony_service_empty_device_id_rejected(self) -> None:
        """Adversarial 4.4: TelephonyService strictly rejects empty or whitespace device IDs."""
        svc = TelephonyService()
        with pytest.raises(InvalidCallStateError):
            await svc.process_call_event("", "ringing")

        with pytest.raises(InvalidCallStateError):
            await svc.process_call_event("   ", "ringing")

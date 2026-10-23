"""tests.unit.test_plugin_pausemusic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive unit test suite for KDE Connect PauseMusic subsystem:
- Category 1: TestPauseMusicConfig (6 tests)
- Category 2: TestPauseConditionPolicy (5 tests)
- Category 3: TestPauseActionPolicy (5 tests)
- Category 4: TestSelectiveMediaResumption (6 tests)
- Category 5: TestSelectiveAudioUnmuting (5 tests)
- Category 6: TestReflectionSuppression (4 tests)
- Category 7: TestOverlappingCallsAndMultiDevice (6 tests)
- Category 8: TestFlappingTransitions (4 tests)
- Category 9: TestLifecycleDisconnectAndUnpair (5 tests)
- Category 10: TestPauseMusicPluginIntegration (6 tests)
- Category 11: TestPauseMusicLockReentrancyAndConcurrency (11 tests)
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pengulink.core.events import (
    AudioSinkUpdatedEvent,
    CallStateChangedEvent,
    DeviceDisconnected,
    DeviceUnpaired,
    DeviceUnpairedEvent,
    EventBus,
)
from pengulink.core.exceptions import (
    AudioUnavailableError,
    PauseMusicConfigError,
)
from pengulink.network.packet import Packet
from pengulink.plugins.mpris.adapter import MockMprisAdapter, PlayerState
from pengulink.plugins.pausemusic import (
    PAUSE_MUSIC_PACKET_TYPE,
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
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def event_bus() -> EventBus:
    """Isolated EventBus instance for unit tests."""
    return EventBus()


@pytest.fixture
def mock_mpris() -> MockMprisAdapter:
    """In-memory MockMprisAdapter fixture."""
    return MockMprisAdapter(auto_default_player=False)


@pytest.fixture
def mock_audio_backend() -> MockAudioSinkBackend:
    """MockAudioSinkBackend fixture with predefined unmuted speaker and muted HDMI."""
    return MockAudioSinkBackend(
        initial_sinks=[
            AudioSink(
                name="alsa_output.speaker",
                description="Speakers",
                volume=35389,
                muted=False,
            ),
            AudioSink(
                name="alsa_output.hdmi",
                description="HDMI",
                volume=65536,
                muted=True,
            ),
        ]
    )


@pytest.fixture
def local_audio(mock_audio_backend: MockAudioSinkBackend, event_bus: EventBus) -> LocalAudioService:
    """LocalAudioService backed by MockAudioSinkBackend."""
    return LocalAudioService(backend=mock_audio_backend, event_bus=event_bus)


@pytest.fixture
def pausemusic_service(
    mock_mpris: MockMprisAdapter,
    local_audio: LocalAudioService,
    event_bus: EventBus,
) -> PauseMusicService:
    """PauseMusicService fixture with standard configuration and EventBus integration."""
    config = PauseMusicConfig()
    return PauseMusicService(
        config=config,
        mpris_adapter=mock_mpris,
        audio_service=local_audio,
        event_bus=event_bus,
    )


@pytest.fixture
def pausemusic_plugin(
    pausemusic_service: PauseMusicService,
    event_bus: EventBus,
) -> PauseMusicPlugin:
    """PauseMusicPlugin fixture initialized with pausemusic_service."""
    return PauseMusicPlugin(service=pausemusic_service, event_bus=event_bus)


# =====================================================================
# Category 1: TestPauseMusicConfig (6 tests)
# =====================================================================


class TestPauseMusicConfig:
    def test_default_config_values(self) -> None:
        """1.1 Verify default policy values matching KDE Connect QML specification."""
        cfg = PauseMusicConfig()
        assert cfg.condition_talking is False
        assert cfg.action_pause is True
        assert cfg.action_mute is False
        assert cfg.action_resume is True
        assert cfg.condition == "ringing"
        assert cfg.action == "pause"
        assert cfg.resume is True

    def test_config_to_dict(self) -> None:
        """1.2 Verify serialization produces snake_case, camelCase, and semantic string keys."""
        cfg = PauseMusicConfig(
            condition_talking=True,
            action_pause=False,
            action_mute=True,
            action_resume=False,
        )
        d = cfg.to_dict()
        # Snake case
        assert d["condition_talking"] is True
        assert d["action_pause"] is False
        assert d["action_mute"] is True
        assert d["action_resume"] is False
        # Semantic strings
        assert d["condition"] == "talking"
        assert d["action"] == "mute"
        assert d["resume"] is False
        # Camel case (KDE Connect QML)
        assert d["conditionTalking"] is True
        assert d["actionPause"] is False
        assert d["actionMute"] is True
        assert d["actionResume"] is False

    def test_config_from_dict_standard(self) -> None:
        """1.3 Verify instantiation from standard dictionary with explicit booleans."""
        data_snake = {
            "condition_talking": True,
            "action_pause": True,
            "action_mute": True,
            "action_resume": False,
        }
        cfg1 = PauseMusicConfig.from_dict(data_snake)
        assert cfg1.condition_talking is True
        assert cfg1.action_pause is True
        assert cfg1.action_mute is True
        assert cfg1.action_resume is False
        assert cfg1.action == "both"

        data_camel = {
            "conditionTalking": False,
            "actionPause": False,
            "actionMute": True,
            "actionResume": True,
        }
        cfg2 = PauseMusicConfig.from_dict(data_camel)
        assert cfg2.condition_talking is False
        assert cfg2.action_pause is False
        assert cfg2.action_mute is True
        assert cfg2.action_resume is True

    def test_config_from_dict_coercion(self) -> None:
        """1.4 Verify lenient boolean coercion from string representations."""
        data = {
            "condition_talking": "true",
            "action_pause": "0",
            "action_mute": "yes",
            "action_resume": "false",
        }
        cfg = PauseMusicConfig.from_dict(data)
        assert cfg.condition_talking is True
        assert cfg.action_pause is False
        assert cfg.action_mute is True
        assert cfg.action_resume is False

    def test_config_from_dict_missing_fields_defaults(self) -> None:
        """1.5 Verify partial or empty dictionaries fall back gracefully to defaults."""
        cfg_empty = PauseMusicConfig.from_dict({})
        assert cfg_empty.condition_talking is False
        assert cfg_empty.action_pause is True
        assert cfg_empty.action_mute is False
        assert cfg_empty.action_resume is True

        cfg_partial = PauseMusicConfig.from_dict({"action": "both"})
        assert cfg_partial.action_pause is True
        assert cfg_partial.action_mute is True
        assert cfg_partial.condition_talking is False
        assert cfg_partial.action_resume is True

        with pytest.raises(PauseMusicConfigError):
            PauseMusicConfig.from_dict("not_a_dict")  # type: ignore[arg-type]

        with pytest.raises(PauseMusicConfigError):
            PauseMusicConfig.from_dict({"condition": "invalid_condition"})

        with pytest.raises(PauseMusicConfigError):
            PauseMusicConfig.from_dict({"action": "invalid_action"})

    def test_config_dynamic_mutation(self, pausemusic_service: PauseMusicService) -> None:
        """1.6 Verify mutating configuration updates operational properties and validates input."""
        cfg = pausemusic_service.config
        cfg.condition = "talking"
        assert cfg.condition_talking is True
        assert cfg.condition == "talking"

        cfg.condition = "ringing"
        assert cfg.condition_talking is False
        assert cfg.condition == "ringing"

        cfg.action = "both"
        assert cfg.action_pause is True and cfg.action_mute is True
        assert cfg.action == "both"

        cfg.action = "mute"
        assert cfg.action_pause is False and cfg.action_mute is True
        assert cfg.action == "mute"

        cfg.action = "none"
        assert cfg.action_pause is False and cfg.action_mute is False
        assert cfg.action == "none"

        cfg.action = "pause"
        assert cfg.action_pause is True and cfg.action_mute is False
        assert cfg.action == "pause"

        cfg.resume = False
        assert cfg.action_resume is False
        assert cfg.resume is False

        with pytest.raises(PauseMusicConfigError):
            cfg.condition = "invalid"

        with pytest.raises(PauseMusicConfigError):
            cfg.action = "unsupported"

        with pytest.raises(PauseMusicConfigError):
            cfg.resume = "not_a_bool"  # type: ignore[assignment]

        pausemusic_service.update_config(condition="talking", action="mute", resume=True)
        assert pausemusic_service.config.condition_talking is True
        assert pausemusic_service.config.action_pause is False
        assert pausemusic_service.config.action_mute is True
        assert pausemusic_service.config.action_resume is True


# =====================================================================
# Category 2: TestPauseConditionPolicy (5 tests)
# =====================================================================


class TestPauseConditionPolicy:
    async def test_condition_talking_false_pauses_on_ringing(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """2.1 Default policy (condition_talking=False): incoming RINGING pauses media immediately."""
        pausemusic_service.config.condition_talking = False
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(
            CallStateChangedEvent(
                device_id="dev1",
                old_state="IDLE",
                new_state="RINGING",
            )
        )

        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert status is not None and status.is_playing is False
        assert pausemusic_service.is_interrupted is True
        assert pausemusic_service.paused_players == {"org.mpris.MediaPlayer2.spotify"}

    async def test_condition_talking_false_talking_maintains_pause(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """2.2 Transition from RINGING -> TALKING maintains media in paused state without re-snapshotting."""
        pausemusic_service.config.condition_talking = False
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", old_state="IDLE", new_state="RINGING")
        )
        assert pausemusic_service.paused_players == {"org.mpris.MediaPlayer2.spotify"}

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", old_state="RINGING", new_state="TALKING")
        )
        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert status is not None and status.is_playing is False
        assert pausemusic_service.is_interrupted is True
        assert pausemusic_service.paused_players == {"org.mpris.MediaPlayer2.spotify"}

    async def test_condition_talking_true_ignores_ringing(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """2.3 condition_talking=True: incoming call in RINGING state is ignored; media continues playing."""
        pausemusic_service.config.condition_talking = True
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", old_state="IDLE", new_state="RINGING")
        )

        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert status is not None and status.is_playing is True
        assert pausemusic_service.is_interrupted is False
        assert len(pausemusic_service.paused_players) == 0

    async def test_condition_talking_true_pauses_on_talking(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """2.4 condition_talking=True: media pauses only once call transitions to TALKING."""
        pausemusic_service.config.condition_talking = True
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", old_state="IDLE", new_state="RINGING")
        )
        status_ringing = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert status_ringing is not None and status_ringing.is_playing is True

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", old_state="RINGING", new_state="TALKING")
        )
        status_talking = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert status_talking is not None and status_talking.is_playing is False
        assert pausemusic_service.is_interrupted is True
        assert pausemusic_service.paused_players == {"org.mpris.MediaPlayer2.spotify"}

    async def test_condition_talking_true_unanswered_call_zero_interruption(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """2.5 condition_talking=True: unanswered call (RINGING -> IDLE) causes zero media interruption."""
        pausemusic_service.config.condition_talking = True
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", old_state="IDLE", new_state="RINGING")
        )
        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", old_state="RINGING", new_state="IDLE")
        )

        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert status is not None and status.is_playing is True
        assert pausemusic_service.is_interrupted is False


# =====================================================================
# Category 3: TestPauseActionPolicy (5 tests)
# =====================================================================


class TestPauseActionPolicy:
    async def test_action_pause_true_action_mute_false_only_pauses_mpris(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
        event_bus: EventBus,
    ) -> None:
        """3.1 action_pause=True, action_mute=False: pauses MPRIS players; leaves audio sinks untouched."""
        pausemusic_service.config.action_pause = True
        pausemusic_service.config.action_mute = False
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.vlc",
            PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=True),
        )

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", new_state="RINGING")
        )
        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert status is not None and status.is_playing is False
        sink = await local_audio.get_sink("alsa_output.speaker")
        assert sink is not None and sink.muted is False

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", new_state="IDLE")
        )
        status_post = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert status_post is not None and status_post.is_playing is True
        sink_post = await local_audio.get_sink("alsa_output.speaker")
        assert sink_post is not None and sink_post.muted is False

    async def test_action_pause_false_action_mute_true_only_mutes_audio(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
        event_bus: EventBus,
    ) -> None:
        """3.2 action_pause=False, action_mute=True: leaves MPRIS playing; mutes host audio sinks."""
        pausemusic_service.config.action_pause = False
        pausemusic_service.config.action_mute = True
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.vlc",
            PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=True),
        )

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", new_state="RINGING")
        )
        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert status is not None and status.is_playing is True
        sink = await local_audio.get_sink("alsa_output.speaker")
        assert sink is not None and sink.muted is True

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", new_state="IDLE")
        )
        sink_post = await local_audio.get_sink("alsa_output.speaker")
        assert sink_post is not None and sink_post.muted is False

    async def test_action_pause_true_action_mute_true_both_pause_and_mute(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
        event_bus: EventBus,
    ) -> None:
        """3.3 action_pause=True, action_mute=True: simultaneously pauses MPRIS and mutes audio sinks."""
        pausemusic_service.config.action_pause = True
        pausemusic_service.config.action_mute = True
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.vlc",
            PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=True),
        )

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", new_state="RINGING")
        )
        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert status is not None and status.is_playing is False
        sink = await local_audio.get_sink("alsa_output.speaker")
        assert sink is not None and sink.muted is True

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", new_state="IDLE")
        )
        status_post = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert status_post is not None and status_post.is_playing is True
        sink_post = await local_audio.get_sink("alsa_output.speaker")
        assert sink_post is not None and sink_post.muted is False

    async def test_action_pause_false_action_mute_false_no_action_taken(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
        event_bus: EventBus,
    ) -> None:
        """3.4 action_pause=False, action_mute=False: passive tracking without altering media or audio."""
        pausemusic_service.config.action_pause = False
        pausemusic_service.config.action_mute = False
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.vlc",
            PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=True),
        )

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", new_state="RINGING")
        )
        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert status is not None and status.is_playing is True
        sink = await local_audio.get_sink("alsa_output.speaker")
        assert sink is not None and sink.muted is False

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", new_state="IDLE")
        )
        assert len(pausemusic_service.paused_players) == 0
        assert len(pausemusic_service.muted_sinks) == 0

    async def test_action_resume_false_leaves_players_paused_and_sinks_muted(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
        event_bus: EventBus,
    ) -> None:
        """3.5 action_resume=False: media remains paused and sinks remain muted after call ends."""
        pausemusic_service.config.action_pause = True
        pausemusic_service.config.action_mute = True
        pausemusic_service.config.action_resume = False
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.vlc",
            PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=True),
        )

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", new_state="RINGING")
        )
        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", new_state="IDLE")
        )

        status_post = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert status_post is not None and status_post.is_playing is False
        sink_post = await local_audio.get_sink("alsa_output.speaker")
        assert sink_post is not None and sink_post.muted is True
        assert len(pausemusic_service.paused_players) == 0
        assert len(pausemusic_service.muted_sinks) == 0

    async def test_action_mutation_mid_call_safely_resumes_paused_players(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
        event_bus: EventBus,
    ) -> None:
        """3.6 Mid-call policy reconfiguration: mutating action (pause -> mute) mid-call
        safely resumes players stored in _paused_players when action_resume is True."""
        pausemusic_service.config.action_pause = True
        pausemusic_service.config.action_mute = False
        pausemusic_service.config.action_resume = True

        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.vlc",
            PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=True),
        )

        # 1. Incoming call begins -> player is paused
        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", new_state="RINGING")
        )
        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert status is not None and status.is_playing is False
        assert "org.mpris.MediaPlayer2.vlc" in pausemusic_service.paused_players

        # 2. Operator mutates action policy mid-call from 'pause' to 'mute'
        pausemusic_service.update_config(action="mute")
        assert pausemusic_service.config.action_pause is False
        assert pausemusic_service.config.action_mute is True

        # 3. Call ends -> player must be safely resumed despite action_pause being False
        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", new_state="IDLE")
        )
        status_post = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert status_post is not None and status_post.is_playing is True
        assert len(pausemusic_service.paused_players) == 0

    async def test_action_mutation_mid_call_safely_unmutes_sinks(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
        event_bus: EventBus,
    ) -> None:
        """3.7 Mid-call policy reconfiguration: mutating action (mute -> pause) mid-call
        safely unmutes audio sinks stored in _muted_sinks when action_resume is True."""
        pausemusic_service.config.action_pause = False
        pausemusic_service.config.action_mute = True
        pausemusic_service.config.action_resume = True

        # Initial state: speaker unmuted
        sink = await local_audio.get_sink("alsa_output.speaker")
        assert sink is not None and sink.muted is False

        # 1. Incoming call begins -> speaker is muted
        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", new_state="RINGING")
        )
        sink_during = await local_audio.get_sink("alsa_output.speaker")
        assert sink_during is not None and sink_during.muted is True
        assert "alsa_output.speaker" in pausemusic_service.muted_sinks

        # 2. Operator mutates action policy mid-call from 'mute' to 'pause'
        pausemusic_service.update_config(action="pause")
        assert pausemusic_service.config.action_pause is True
        assert pausemusic_service.config.action_mute is False

        # 3. Call ends -> speaker must be safely unmuted despite action_mute being False
        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", new_state="IDLE")
        )
        sink_post = await local_audio.get_sink("alsa_output.speaker")
        assert sink_post is not None and sink_post.muted is False
        assert len(pausemusic_service.muted_sinks) == 0


# =====================================================================
# Category 4: TestSelectiveMediaResumption (6 tests)
# =====================================================================


class TestSelectiveMediaResumption:
    async def test_single_playing_player_is_paused_and_resumed(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """4.1 Single playing player is paused on call start and resumed on call end."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )
        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        s1 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s1 is not None and s1.is_playing is False

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))
        s2 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s2 is not None and s2.is_playing is True

    async def test_pre_call_paused_player_is_never_resumed(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """4.2 CRITICAL INVARIANT: Player that was already paused before the call MUST NOT be resumed."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.vlc",
            PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=False),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        assert "org.mpris.MediaPlayer2.vlc" not in pausemusic_service.paused_players

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))
        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert status is not None and status.is_playing is False

    async def test_pre_call_stopped_player_is_never_resumed(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """4.3 Player in stopped state before the call is skipped and never resumed."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.audacious",
            PlayerState(player="org.mpris.MediaPlayer2.audacious", is_playing=False),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))

        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.audacious")
        assert status is not None and status.is_playing is False

    async def test_multiple_players_mixed_playing_and_paused(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """4.4 Multi-player environment: only actively playing players are captured and resumed."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.p1",
            PlayerState(player="org.mpris.MediaPlayer2.p1", is_playing=True),
        )
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.p2",
            PlayerState(player="org.mpris.MediaPlayer2.p2", is_playing=False),
        )
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.p3",
            PlayerState(player="org.mpris.MediaPlayer2.p3", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        assert pausemusic_service.paused_players == {
            "org.mpris.MediaPlayer2.p1",
            "org.mpris.MediaPlayer2.p3",
        }

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))
        s1 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.p1")
        s2 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.p2")
        s3 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.p3")
        assert s1 is not None and s1.is_playing is True
        assert s2 is not None and s2.is_playing is False
        assert s3 is not None and s3.is_playing is True

    async def test_player_cannot_pause_uses_stop_action_and_resumes_with_play(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """4.5 Player with can_pause=False, can_stop=True receives Stop on pause and Play on resume."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.radio",
            PlayerState(
                player="org.mpris.MediaPlayer2.radio",
                is_playing=True,
                can_pause=False,
            ),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        s1 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.radio")
        assert s1 is not None and s1.is_playing is False

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))
        s2 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.radio")
        assert s2 is not None and s2.is_playing is True

    async def test_player_vanishes_during_call_handled_gracefully(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """4.6 Player terminates during call: resumption catches error and resumes remaining players."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.p1",
            PlayerState(player="org.mpris.MediaPlayer2.p1", is_playing=True),
        )
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.p2",
            PlayerState(player="org.mpris.MediaPlayer2.p2", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        # Player 1 process exits while on call
        mock_mpris.remove_mock_player("org.mpris.MediaPlayer2.p1")

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))
        s2 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.p2")
        assert s2 is not None and s2.is_playing is True


# =====================================================================
# Category 5: TestSelectiveAudioUnmuting (5 tests)
# =====================================================================


class TestSelectiveAudioUnmuting:
    async def test_single_unmuted_sink_is_muted_and_unmuted(
        self,
        pausemusic_service: PauseMusicService,
        local_audio: LocalAudioService,
        event_bus: EventBus,
    ) -> None:
        """5.1 Unmuted audio output sink is muted on call start and unmuted on call conclusion."""
        pausemusic_service.config.action_mute = True
        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))

        sink1 = await local_audio.get_sink("alsa_output.speaker")
        assert sink1 is not None and sink1.muted is True
        assert pausemusic_service.muted_sinks == {"alsa_output.speaker"}

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))
        sink2 = await local_audio.get_sink("alsa_output.speaker")
        assert sink2 is not None and sink2.muted is False

    async def test_pre_call_muted_sink_remains_muted_after_call(
        self,
        pausemusic_service: PauseMusicService,
        local_audio: LocalAudioService,
        event_bus: EventBus,
    ) -> None:
        """5.2 CRITICAL INVARIANT: Pre-call muted sink MUST NOT be unmuted when call finishes."""
        pausemusic_service.config.action_mute = True
        # HDMI is initially muted in fixture
        sink_initial = await local_audio.get_sink("alsa_output.hdmi")
        assert sink_initial is not None and sink_initial.muted is True

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        assert "alsa_output.hdmi" not in pausemusic_service.muted_sinks

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))
        sink_post = await local_audio.get_sink("alsa_output.hdmi")
        assert sink_post is not None and sink_post.muted is True

    async def test_multiple_sinks_mixed_mute_states(
        self,
        pausemusic_service: PauseMusicService,
        mock_audio_backend: MockAudioSinkBackend,
        local_audio: LocalAudioService,
        event_bus: EventBus,
    ) -> None:
        """5.3 Multi-sink: only pre-call unmuted sinks are snapshotted and restored."""
        pausemusic_service.config.action_mute = True
        mock_audio_backend.add_sink(
            AudioSink(name="sink.usb", description="USB", volume=40000, muted=False)
        )

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        assert pausemusic_service.muted_sinks == {"alsa_output.speaker", "sink.usb"}

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))
        s_speaker = await local_audio.get_sink("alsa_output.speaker")
        s_usb = await local_audio.get_sink("sink.usb")
        s_hdmi = await local_audio.get_sink("alsa_output.hdmi")
        assert s_speaker is not None and s_speaker.muted is False
        assert s_usb is not None and s_usb.muted is False
        assert s_hdmi is not None and s_hdmi.muted is True

    async def test_audio_service_unavailable_handled_gracefully(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """5.4 Sound server backend raising AudioUnavailableError fails gracefully without crash."""
        pausemusic_service.config.action_mute = True
        pausemusic_service.config.action_pause = True

        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        # Mock audio service raising error
        faulty_audio = AsyncMock(spec=LocalAudioService)
        faulty_audio.list_sinks.side_effect = AudioUnavailableError("Sound daemon unreachable")
        pausemusic_service._audio_service = faulty_audio

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        # MPRIS player still successfully pauses
        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert status is not None and status.is_playing is False

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))
        status_post = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert status_post is not None and status_post.is_playing is True

    async def test_new_sink_plugged_during_call_not_unmuted(
        self,
        pausemusic_service: PauseMusicService,
        mock_audio_backend: MockAudioSinkBackend,
        local_audio: LocalAudioService,
        event_bus: EventBus,
    ) -> None:
        """5.5 Audio device plugged in during call (not in snapshot) is not altered on call end."""
        pausemusic_service.config.action_mute = True

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        # Device plugged in while call is active
        mock_audio_backend.add_sink(
            AudioSink(name="sink.headphones", description="Headphones", volume=50000, muted=True)
        )

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))
        sink_hp = await local_audio.get_sink("sink.headphones")
        assert sink_hp is not None and sink_hp.muted is True


# =====================================================================
# Category 6: TestReflectionSuppression (4 tests)
# =====================================================================


class TestReflectionSuppression:
    async def test_kdeconnect_virtual_player_prefix_ignored(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """6.1 Virtual KDE Connect MPRIS player prefix org.mpris.MediaPlayer2.kdeconnect.* is ignored."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.kdeconnect.device_abc",
            PlayerState(player="org.mpris.MediaPlayer2.kdeconnect.device_abc", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        assert "org.mpris.MediaPlayer2.kdeconnect.device_abc" not in pausemusic_service.paused_players
        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.kdeconnect.device_abc")
        assert status is not None and status.is_playing is True

    async def test_pengulink_virtual_player_prefix_ignored(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """6.2 Virtual PenguLink MPRIS player prefix org.mpris.MediaPlayer2.pengulink.* is ignored."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.pengulink.virtual",
            PlayerState(player="org.mpris.MediaPlayer2.pengulink.virtual", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        assert "org.mpris.MediaPlayer2.pengulink.virtual" not in pausemusic_service.paused_players

    async def test_playerctld_reflection_ignored(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """6.3 org.mpris.MediaPlayer2.playerctld daemon player is ignored to avoid duplicate commands."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.playerctld",
            PlayerState(player="org.mpris.MediaPlayer2.playerctld", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        assert "org.mpris.MediaPlayer2.playerctld" not in pausemusic_service.paused_players

    async def test_mixed_real_and_reflected_players_only_real_affected(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """6.4 Mixed real and virtual players: only the real host player is paused and resumed."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.kdeconnect.remote_phone",
            PlayerState(player="org.mpris.MediaPlayer2.kdeconnect.remote_phone", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        s_spotify = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        s_remote = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.kdeconnect.remote_phone")
        assert s_spotify is not None and s_spotify.is_playing is False
        assert s_remote is not None and s_remote.is_playing is True

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))
        s_spotify_post = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s_spotify_post is not None and s_spotify_post.is_playing is True


# =====================================================================
# Category 7: TestOverlappingCallsAndMultiDevice (6 tests)
# =====================================================================


class TestOverlappingCallsAndMultiDevice:
    async def test_two_devices_overlapping_calls_resumes_only_when_both_end(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """7.1 OVERLAPPING CALL INVARIANT: Media stays paused until the final active call concludes."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        # 1. Dev A rings: 0 -> 1 transition (music pauses)
        await event_bus.publish(CallStateChangedEvent(device_id="devA", new_state="RINGING"))
        s1 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s1 is not None and s1.is_playing is False
        assert pausemusic_service.active_call_devices == {"devA"}

        # 2. Dev B rings: 1 -> 2 transition (music stays paused)
        await event_bus.publish(CallStateChangedEvent(device_id="devB", new_state="RINGING"))
        assert pausemusic_service.active_call_devices == {"devA", "devB"}
        s2 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s2 is not None and s2.is_playing is False

        # 3. Dev A call ends: 2 -> 1 transition (music STAYS PAUSED!)
        await event_bus.publish(CallStateChangedEvent(device_id="devA", new_state="IDLE"))
        assert pausemusic_service.active_call_devices == {"devB"}
        s3 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s3 is not None and s3.is_playing is False

        # 4. Dev B call ends: 1 -> 0 transition (music RESUMES!)
        await event_bus.publish(CallStateChangedEvent(device_id="devB", new_state="IDLE"))
        assert len(pausemusic_service.active_call_devices) == 0
        s4 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s4 is not None and s4.is_playing is True

    async def test_overlapping_call_does_not_overwrite_playing_snapshot(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """7.2 RE-ENTRANT SNAPSHOT PROTECTION: Dev B ringing while interrupted does not wipe snapshot."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="devA", new_state="RINGING"))
        assert pausemusic_service.paused_players == {"org.mpris.MediaPlayer2.spotify"}

        # While media is already paused, Dev B starts call
        await event_bus.publish(CallStateChangedEvent(device_id="devB", new_state="RINGING"))
        # Snapshot MUST still preserve Spotify!
        assert pausemusic_service.paused_players == {"org.mpris.MediaPlayer2.spotify"}

        await event_bus.publish(CallStateChangedEvent(device_id="devA", new_state="IDLE"))
        await event_bus.publish(CallStateChangedEvent(device_id="devB", new_state="IDLE"))

        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert status is not None and status.is_playing is True

    async def test_device_a_talking_device_b_ringing_order_of_events(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """7.3 Dev A answers (TALKING), Dev B rings (RINGING), Dev B ends, Dev A ends."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="devA", new_state="TALKING"))
        await event_bus.publish(CallStateChangedEvent(device_id="devB", new_state="RINGING"))
        await event_bus.publish(CallStateChangedEvent(device_id="devB", new_state="IDLE"))

        s_mid = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s_mid is not None and s_mid.is_playing is False

        await event_bus.publish(CallStateChangedEvent(device_id="devA", new_state="IDLE"))
        s_end = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s_end is not None and s_end.is_playing is True

    async def test_multi_device_rapid_staggered_calls(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """7.4 Three devices initiate and conclude calls in staggered interleaving."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        await event_bus.publish(CallStateChangedEvent(device_id="dev2", new_state="RINGING"))
        await event_bus.publish(CallStateChangedEvent(device_id="dev3", new_state="RINGING"))
        assert len(pausemusic_service.active_call_devices) == 3

        await event_bus.publish(CallStateChangedEvent(device_id="dev2", new_state="IDLE"))
        assert len(pausemusic_service.active_call_devices) == 2
        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))
        assert len(pausemusic_service.active_call_devices) == 1

        s_still_paused = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s_still_paused is not None and s_still_paused.is_playing is False

        await event_bus.publish(CallStateChangedEvent(device_id="dev3", new_state="IDLE"))
        assert len(pausemusic_service.active_call_devices) == 0
        s_resumed = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s_resumed is not None and s_resumed.is_playing is True

    async def test_active_call_device_tracking_and_refcount(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """7.5 Service active_call_devices set and is_interrupted reflect real-time active devices."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )
        assert pausemusic_service.is_interrupted is False
        assert len(pausemusic_service.active_call_devices) == 0

        await event_bus.publish(CallStateChangedEvent(device_id="devX", new_state="RINGING"))
        assert pausemusic_service.is_interrupted is True
        assert pausemusic_service.active_call_devices == {"devX"}

        status = pausemusic_service.get_status()
        assert status["is_interrupted"] is True
        assert status["active_call_devices"] == ["devX"]

        await event_bus.publish(CallStateChangedEvent(device_id="devX", new_state="IDLE"))
        assert pausemusic_service.is_interrupted is False
        assert len(pausemusic_service.active_call_devices) == 0

    async def test_same_device_call_waiting_transitions(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """7.6 Same device transitions maintain active interruption until termination."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="TALKING"))
        # Call waiting incoming ringing transition on same device
        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="TALKING"))
        assert pausemusic_service.is_interrupted is True

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))
        assert pausemusic_service.is_interrupted is False
        s = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s is not None and s.is_playing is True


# =====================================================================
# Category 8: TestFlappingTransitions (4 tests)
# =====================================================================


class TestFlappingTransitions:
    async def test_rapid_ringing_talking_idle_roundtrip(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """8.1 Rapid RINGING -> TALKING -> IDLE sequence completes cleanly with music resumed."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        for _ in range(5):
            await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
            await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="TALKING"))
            await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))

        assert pausemusic_service.is_interrupted is False
        s = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s is not None and s.is_playing is True
        assert len(pausemusic_service.paused_players) == 0

    async def test_rapid_successive_calls_ringing_idle_cycles(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """8.2 Back-to-back calls capture snapshot cleanly, resume cleanly, and leave no state leaks."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        # Call 1
        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))

        # Call 2 immediately follows
        await event_bus.publish(CallStateChangedEvent(device_id="dev2", new_state="RINGING"))
        s2 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s2 is not None and s2.is_playing is False

        await event_bus.publish(CallStateChangedEvent(device_id="dev2", new_state="IDLE"))
        s_end = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s_end is not None and s_end.is_playing is True

    async def test_duplicate_ringing_events_are_idempotent(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """8.3 Duplicate RINGING events from same device do not increment refcount or re-trigger actions."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))

        assert pausemusic_service.active_call_devices == {"dev1"}
        assert pausemusic_service.paused_players == {"org.mpris.MediaPlayer2.spotify"}

    async def test_duplicate_idle_events_are_idempotent(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """8.4 Duplicate IDLE events do not cause errors or invalid state transitions."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="RINGING"))
        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))
        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="IDLE"))

        assert len(pausemusic_service.active_call_devices) == 0
        s = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s is not None and s.is_playing is True


# =====================================================================
# Category 9: TestLifecycleDisconnectAndUnpair (5 tests)
# =====================================================================


class TestLifecycleDisconnectAndUnpair:
    async def test_device_disconnected_during_active_call_resumes_media(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """9.1 DISCONNECT FAIL-SAFE: Device drops connection during call -> media is resumed."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="devA", new_state="RINGING"))
        s_mid = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s_mid is not None and s_mid.is_playing is False

        await event_bus.publish(DeviceDisconnected(device_id="devA"))
        assert len(pausemusic_service.active_call_devices) == 0
        s_post = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s_post is not None and s_post.is_playing is True

    async def test_device_unpaired_during_active_call_resumes_media(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """9.2 UNPAIR FAIL-SAFE: Device is unpaired during call -> media is resumed."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="devA", new_state="TALKING"))
        assert pausemusic_service.is_interrupted is True

        await event_bus.publish(DeviceUnpaired(device_id="devA"))
        assert len(pausemusic_service.active_call_devices) == 0
        s_post = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s_post is not None and s_post.is_playing is True

        # Also test DeviceUnpairedEvent alias
        await event_bus.publish(CallStateChangedEvent(device_id="devA", new_state="TALKING"))
        assert pausemusic_service.is_interrupted is True
        await event_bus.publish(DeviceUnpairedEvent(device_id="devA"))
        assert len(pausemusic_service.active_call_devices) == 0

    async def test_one_device_disconnects_while_second_call_active_remains_paused(
        self,
        pausemusic_service: PauseMusicService,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """9.3 Multi-call disconnect: Dev A disconnects while Dev B is active -> media stays paused."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="devA", new_state="RINGING"))
        await event_bus.publish(CallStateChangedEvent(device_id="devB", new_state="RINGING"))

        await event_bus.publish(DeviceDisconnected(device_id="devA"))
        assert pausemusic_service.active_call_devices == {"devB"}
        s_mid = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s_mid is not None and s_mid.is_playing is False

        await event_bus.publish(CallStateChangedEvent(device_id="devB", new_state="IDLE"))
        assert len(pausemusic_service.active_call_devices) == 0
        s_end = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s_end is not None and s_end.is_playing is True

    async def test_plugin_stop_restores_media_if_interrupted(
        self,
        pausemusic_service: PauseMusicService,
        pausemusic_plugin: PauseMusicPlugin,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """9.4 Daemon / plugin shutdown: plugin.stop() restores media if call was active."""
        await pausemusic_plugin.start()
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(CallStateChangedEvent(device_id="devA", new_state="RINGING"))
        s_mid = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s_mid is not None and s_mid.is_playing is False

        await pausemusic_plugin.stop()
        s_end = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s_end is not None and s_end.is_playing is True

    async def test_idempotent_duplicate_disconnect_events(
        self,
        pausemusic_service: PauseMusicService,
        event_bus: EventBus,
    ) -> None:
        """9.5 Multiple disconnect events for the same device execute safely without error."""
        await event_bus.publish(DeviceDisconnected(device_id="devA"))
        await event_bus.publish(DeviceDisconnected(device_id="devA"))
        await event_bus.publish(DeviceUnpaired(device_id="devA"))
        assert len(pausemusic_service.active_call_devices) == 0


# =====================================================================
# Category 10: TestPauseMusicPluginIntegration (6 tests)
# =====================================================================


class TestPauseMusicPluginIntegration:
    async def test_plugin_metadata_and_capabilities(
        self,
        pausemusic_plugin: PauseMusicPlugin,
    ) -> None:
        """10.1 Verify plugin declarations match KDE Connect protocol specification."""
        assert pausemusic_plugin.plugin_id == "pausemusic"
        assert pausemusic_plugin.incoming_capabilities == [PAUSE_MUSIC_PACKET_TYPE]
        assert pausemusic_plugin.outgoing_capabilities == []
        assert pausemusic_plugin.pause_music_service is pausemusic_plugin.service

    async def test_plugin_eventbus_subscription_lifecycle(
        self,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
    ) -> None:
        """10.2 Verify plugin.start() subscribes to EventBus and stop() unsubscribes cleanly."""
        bus = EventBus()
        service = PauseMusicService(
            mpris_adapter=mock_mpris,
            audio_service=local_audio,
        )
        plugin = PauseMusicPlugin(service=service, event_bus=bus)

        initial_count = len(bus._subscribers.get(CallStateChangedEvent, []))
        assert initial_count == 0

        await plugin.start()
        start_count = len(bus._subscribers.get(CallStateChangedEvent, []))
        assert start_count == 1

        await plugin.stop()
        stop_count = len(bus._subscribers.get(CallStateChangedEvent, []))
        assert stop_count == 0

    async def test_plugin_handle_packet_safe_acknowledgment(
        self,
        pausemusic_plugin: PauseMusicPlugin,
    ) -> None:
        """10.3 Direct inbound kdeconnect.pausemusic packet is acknowledged without exception."""
        packet = Packet(
            type=PAUSE_MUSIC_PACKET_TYPE,
            body={"action": "probe"},
        )

        async def _dummy_sender(*_args: Any, **_kwargs: Any) -> None:
            pass

        # Should execute cleanly without error
        await pausemusic_plugin.handle_packet(packet, "dev1", _dummy_sender)

        # Packets with different types should be safely handled
        other_packet = Packet(type="kdeconnect.ping", body={})
        await pausemusic_plugin.handle_packet(other_packet, "dev1", _dummy_sender)

        empty_dev_packet = Packet(type=PAUSE_MUSIC_PACKET_TYPE, body={})
        await pausemusic_plugin.handle_packet(empty_dev_packet, "", _dummy_sender)

        non_dict_body = Packet.model_construct(type=PAUSE_MUSIC_PACKET_TYPE, body="string")
        await pausemusic_plugin.handle_packet(non_dict_body, "dev1", _dummy_sender)

    async def test_plugin_routes_call_state_changed_event_to_service(
        self,
        pausemusic_plugin: PauseMusicPlugin,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """10.4 EventBus CallStateChangedEvent routes through plugin to pause mock player."""
        await pausemusic_plugin.start()
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", old_state="IDLE", new_state="RINGING")
        )

        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert status is not None and status.is_playing is False
        assert pausemusic_plugin.is_interrupted is True
        assert pausemusic_plugin.paused_players == {"org.mpris.MediaPlayer2.spotify"}
        await pausemusic_plugin.stop()

    async def test_plugin_routes_device_disconnected_to_service(
        self,
        pausemusic_plugin: PauseMusicPlugin,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """10.5 EventBus DeviceDisconnected routes through plugin to resume paused player."""
        await pausemusic_plugin.start()
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", old_state="IDLE", new_state="RINGING")
        )
        assert pausemusic_plugin.is_interrupted is True

        await event_bus.publish(DeviceDisconnected(device_id="dev1"))
        assert pausemusic_plugin.is_interrupted is False
        status = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert status is not None and status.is_playing is True
        await pausemusic_plugin.stop()

    async def test_plugin_dynamic_config_update(
        self,
        pausemusic_plugin: PauseMusicPlugin,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """10.6 Dynamically updating plugin configuration modifies service behavior on subsequent events."""
        await pausemusic_plugin.start()
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )

        await pausemusic_plugin.update_config(condition="talking")
        assert pausemusic_plugin.config is not None
        assert pausemusic_plugin.config.condition_talking is True

        # Ringing does not pause
        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", old_state="IDLE", new_state="RINGING")
        )
        s1 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s1 is not None and s1.is_playing is True

        # Talking pauses
        await event_bus.publish(
            CallStateChangedEvent(device_id="dev1", old_state="RINGING", new_state="TALKING")
        )
        s2 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s2 is not None and s2.is_playing is False

        # Status inspection
        status = await pausemusic_plugin.get_status()
        assert status["is_interrupted"] is True
        assert "dev1" in status["active_call_devices"]

        # Restore media manually
        await pausemusic_plugin.restore_media()
        s3 = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert s3 is not None and s3.is_playing is True
        assert pausemusic_plugin.is_interrupted is False
        await pausemusic_plugin.stop()


# =====================================================================
# Category 11: TestPauseMusicLockReentrancyAndConcurrency (11 tests)
# =====================================================================


class TestPauseMusicLockReentrancyAndConcurrency:
    """Category 11: Explicit regression verification against lock reentrancy deadlocks.

    Ensures asyncio.Lock in PauseMusicService is NEVER held during external async
    operations (send_action, set_muted, list_players, list_sinks, EventBus publishes),
    preventing reentrant deadlocks during MPRIS callbacks, EventBus subscribers,
    and multi-coroutine interleavings.
    """

    async def test_reentrant_call_state_changed_during_mpris_send_action(
        self,
        event_bus: EventBus,
    ) -> None:
        """11.1 Verify that an MPRIS action triggering a re-entrant call does not deadlock."""
        class ReentrantMprisAdapter(MockMprisAdapter):
            def __init__(self) -> None:
                super().__init__(auto_default_player=False)
                self.service: PauseMusicService | None = None
                self.callback_executed = False

            async def send_action(self, player: str, action: str) -> bool:
                if action == "Pause" and self.service and not self.callback_executed:
                    self.callback_executed = True
                    assert self.service.is_interrupted is True
                    # Attempt re-entrant call during external action
                    await self.service.handle_call_state_changed("dev2", "RINGING")
                return await super().send_action(player, action)

        adapter = ReentrantMprisAdapter()
        adapter.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )
        svc = PauseMusicService(mpris_adapter=adapter, event_bus=event_bus)
        adapter.service = svc

        # Must complete cleanly without timeout
        res = await asyncio.wait_for(svc.handle_call_started("dev1"), timeout=1.0)
        assert res is True
        assert adapter.callback_executed is True
        assert svc.active_call_devices == {"dev1", "dev2"}
        assert svc.is_interrupted is True
        assert "org.mpris.MediaPlayer2.spotify" in svc.paused_players

        # Verify overlapping call lifecycle cleans up cleanly
        await asyncio.wait_for(svc.handle_call_ended("dev1"), timeout=1.0)
        assert svc.active_call_devices == {"dev2"}
        status_mid = await adapter.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert status_mid is not None and status_mid.is_playing is False

        await asyncio.wait_for(svc.handle_call_ended("dev2"), timeout=1.0)
        assert len(svc.active_call_devices) == 0
        status_end = await adapter.get_player_status("org.mpris.MediaPlayer2.spotify")
        assert status_end is not None and status_end.is_playing is True

    async def test_reentrancy_mpris_send_action_play_triggers_new_call(
        self,
        event_bus: EventBus,
    ) -> None:
        """11.2 Re-entrant MPRIS callback: send_action('Play') on call end triggers a new call."""
        class ReentrantResumeMprisAdapter(MockMprisAdapter):
            def __init__(self) -> None:
                super().__init__(auto_default_player=False)
                self.service: PauseMusicService | None = None
                self.play_reentered = False

            async def send_action(self, player: str, action: str) -> bool:
                if action == "Play" and self.service and not self.play_reentered:
                    self.play_reentered = True
                    # Re-enter service: dev2 starts ringing while dev1 resume is in flight
                    await self.service.handle_call_state_changed("dev2", "RINGING")
                return await super().send_action(player, action)

        adapter = ReentrantResumeMprisAdapter()
        adapter.add_mock_player(
            "org.mpris.MediaPlayer2.vlc",
            PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=True),
        )
        svc = PauseMusicService(mpris_adapter=adapter, event_bus=event_bus)
        adapter.service = svc

        # Interrupt dev1
        await svc.handle_call_started("dev1")
        assert svc.is_interrupted is True

        # End dev1, triggering resume and re-entrant dev2 call
        await asyncio.wait_for(svc.handle_call_ended("dev1"), timeout=1.0)
        assert adapter.play_reentered is True
        assert svc.active_call_devices == {"dev2"}

        # End dev2
        await asyncio.wait_for(svc.handle_call_ended("dev2"), timeout=1.0)
        assert len(svc.active_call_devices) == 0

    async def test_reentrancy_mpris_status_query_during_send_action(
        self,
        event_bus: EventBus,
    ) -> None:
        """11.3 Status queries during MPRIS send_action execute non-blockingly."""
        class StatusQueryMprisAdapter(MockMprisAdapter):
            def __init__(self) -> None:
                super().__init__(auto_default_player=False)
                self.service: PauseMusicService | None = None
                self.observed_status: dict[str, Any] = {}

            async def send_action(self, player: str, action: str) -> bool:
                if self.service:
                    self.observed_status = self.service.get_status()
                    assert self.service.is_interrupted is True
                    assert self.service.active_call_devices == {"dev1"}
                return await super().send_action(player, action)

        adapter = StatusQueryMprisAdapter()
        adapter.add_mock_player(
            "org.mpris.MediaPlayer2.rhythmbox",
            PlayerState(player="org.mpris.MediaPlayer2.rhythmbox", is_playing=True),
        )
        svc = PauseMusicService(mpris_adapter=adapter, event_bus=event_bus)
        adapter.service = svc

        await asyncio.wait_for(svc.handle_call_started("dev1"), timeout=1.0)
        assert adapter.observed_status["is_interrupted"] is True
        assert adapter.observed_status["active_call_devices"] == ["dev1"]

    async def test_reentrant_call_ended_during_audio_set_muted_eventbus(
        self,
        event_bus: EventBus,
        mock_audio_backend: MockAudioSinkBackend,
    ) -> None:
        """11.4 Re-entrant EventBus callback: AudioSinkUpdatedEvent invokes handle_call_ended."""
        audio = LocalAudioService(backend=mock_audio_backend, event_bus=event_bus)
        cfg = PauseMusicConfig(action_pause=False, action_mute=True)
        svc = PauseMusicService(config=cfg, audio_service=audio, event_bus=event_bus)

        subscriber_called = False

        async def on_audio_updated(event: AudioSinkUpdatedEvent) -> None:
            nonlocal subscriber_called
            if event.sink and event.sink.muted and not subscriber_called:
                subscriber_called = True
                await svc.handle_call_ended("dev1")

        event_bus.subscribe(AudioSinkUpdatedEvent, on_audio_updated)

        # Must complete cleanly without timeout
        res = await asyncio.wait_for(svc.handle_call_started("dev1"), timeout=1.0)
        assert res is True
        assert subscriber_called is True
        assert "dev1" not in svc.active_call_devices
        assert svc.is_interrupted is False

    async def test_reentrancy_eventbus_audio_sink_updated_triggers_new_call(
        self,
        event_bus: EventBus,
    ) -> None:
        """11.5 Re-entrant EventBus callback: AudioSinkUpdatedEvent initiates new call for dev2."""
        backend = MockAudioSinkBackend([
            AudioSink(name="sink.headphones", description="Headphones", volume=50000, muted=False)
        ])
        audio = LocalAudioService(backend=backend, event_bus=event_bus)
        cfg = PauseMusicConfig(action_pause=False, action_mute=True)
        svc = PauseMusicService(config=cfg, audio_service=audio, event_bus=event_bus)

        dev2_registered = False

        async def on_audio_sink_updated(event: AudioSinkUpdatedEvent) -> None:
            nonlocal dev2_registered
            if event.sink and event.sink.muted and not dev2_registered:
                dev2_registered = True
                await svc.handle_call_state_changed("dev2", "RINGING")

        event_bus.subscribe(AudioSinkUpdatedEvent, on_audio_sink_updated)

        await asyncio.wait_for(svc.handle_call_started("dev1"), timeout=1.0)
        assert dev2_registered is True
        assert svc.active_call_devices == {"dev1", "dev2"}
        assert svc.is_interrupted is True

        # Clean up
        await asyncio.wait_for(svc.handle_call_ended("dev1"), timeout=1.0)
        assert svc.active_call_devices == {"dev2"}
        await asyncio.wait_for(svc.handle_call_ended("dev2"), timeout=1.0)
        assert len(svc.active_call_devices) == 0

    async def test_reentrancy_eventbus_audio_sink_updated_queries_status(
        self,
        event_bus: EventBus,
    ) -> None:
        """11.6 Re-entrant EventBus callback: AudioSinkUpdatedEvent queries get_status()."""
        backend = MockAudioSinkBackend([
            AudioSink(name="sink.lineout", description="Line Out", volume=60000, muted=False)
        ])
        audio = LocalAudioService(backend=backend, event_bus=event_bus)
        cfg = PauseMusicConfig(action_pause=False, action_mute=True)
        svc = PauseMusicService(config=cfg, audio_service=audio, event_bus=event_bus)

        queried_status: dict[str, Any] = {}

        async def on_audio_sink_updated(event: AudioSinkUpdatedEvent) -> None:
            nonlocal queried_status
            queried_status = svc.get_status()

        event_bus.subscribe(AudioSinkUpdatedEvent, on_audio_sink_updated)

        await asyncio.wait_for(svc.handle_call_started("dev1"), timeout=1.0)
        assert queried_status.get("is_interrupted") is True
        assert "dev1" in queried_status.get("active_call_devices", [])

    async def test_reentrant_call_during_restore_media(
        self,
        event_bus: EventBus,
    ) -> None:
        """11.7 Verify that restore_media executes external actions outside lock without deadlock."""
        class ReentrantRestoreMprisAdapter(MockMprisAdapter):
            def __init__(self) -> None:
                super().__init__(auto_default_player=False)
                self.service: PauseMusicService | None = None

            async def send_action(self, player: str, action: str) -> bool:
                if self.service is not None and action == "Play":
                    # Re-entrant query/transition during restore_media
                    _ = self.service.get_status()
                    await self.service.handle_device_disconnected("dev1")
                return await super().send_action(player, action)

        adapter = ReentrantRestoreMprisAdapter()
        adapter.add_mock_player(
            "org.mpris.MediaPlayer2.vlc",
            PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=True),
        )
        svc = PauseMusicService(mpris_adapter=adapter, event_bus=event_bus)
        adapter.service = svc

        await svc.handle_call_started("dev1")
        assert "org.mpris.MediaPlayer2.vlc" in svc.paused_players

        # restore_media must complete without deadlock
        await asyncio.wait_for(svc.restore_media(), timeout=1.0)
        assert svc.is_interrupted is False
        assert len(svc.paused_players) == 0

    async def test_concurrent_call_started_and_ended_interleaving(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """11.8 Verify that concurrent started and ended calls do not deadlock or orphan paused players."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.vlc",
            PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=True),
        )
        svc = PauseMusicService(mpris_adapter=mock_mpris, event_bus=event_bus)

        # Run concurrent overlapping start and end coroutines
        await asyncio.gather(
            svc.handle_call_started("dev1"),
            svc.handle_call_started("dev2"),
            svc.handle_call_ended("dev1"),
            svc.handle_call_ended("dev2"),
        )

        assert svc.is_interrupted is False
        assert len(svc.active_call_devices) == 0
        vlc_state = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert vlc_state is not None and vlc_state.is_playing is True

    async def test_lock_never_held_during_mpris_external_operations(
        self,
        event_bus: EventBus,
    ) -> None:
        """11.9 Lock assertion: PauseMusicService._lock is False during all MPRIS external calls."""
        class LockAssertMprisAdapter(MockMprisAdapter):
            def __init__(self, svc: PauseMusicService) -> None:
                super().__init__(auto_default_player=False)
                self.svc = svc
                self.list_players_called = 0
                self.get_player_status_called = 0
                self.send_action_called = 0

            async def list_players(self) -> list[str]:
                assert self.svc._lock.locked() is False, "Lock held during list_players()!"
                self.list_players_called += 1
                return await super().list_players()

            async def get_player_status(self, player: str) -> PlayerState | None:
                assert self.svc._lock.locked() is False, f"Lock held during get_player_status({player})!"
                self.get_player_status_called += 1
                return await super().get_player_status(player)

            async def send_action(self, player: str, action: str) -> bool:
                assert self.svc._lock.locked() is False, f"Lock held during send_action({player}, {action})!"
                self.send_action_called += 1
                return await super().send_action(player, action)

        svc = PauseMusicService(config=PauseMusicConfig(action_pause=True, action_mute=False))
        adapter = LockAssertMprisAdapter(svc)
        adapter.add_mock_player(
            "org.mpris.MediaPlayer2.mpv",
            PlayerState(player="org.mpris.MediaPlayer2.mpv", is_playing=True),
        )
        svc._mpris_adapter = adapter

        # 1. During handle_call_started: list_players, get_player_status, send_action('Pause')
        await asyncio.wait_for(svc.handle_call_started("dev1"), timeout=1.0)
        assert adapter.list_players_called >= 1
        assert adapter.get_player_status_called >= 1
        assert adapter.send_action_called >= 1

        # 2. During handle_call_ended: send_action('Play')
        await asyncio.wait_for(svc.handle_call_ended("dev1"), timeout=1.0)
        assert adapter.send_action_called >= 2

        # 3. During restore_media: send_action('Play')
        await svc.handle_call_started("dev1")
        await asyncio.wait_for(svc.restore_media(), timeout=1.0)
        assert adapter.send_action_called >= 4

    async def test_lock_never_held_during_audio_external_operations(
        self,
        event_bus: EventBus,
    ) -> None:
        """11.10 Lock assertion: PauseMusicService._lock is False during all Audio external calls."""
        class LockAssertAudioService(LocalAudioService):
            def __init__(self, backend: MockAudioSinkBackend, svc: PauseMusicService) -> None:
                super().__init__(backend=backend)
                self.svc = svc
                self.list_sinks_called = 0
                self.set_muted_called = 0

            async def list_sinks(self) -> list[AudioSink]:
                assert self.svc._lock.locked() is False, "Lock held during list_sinks()!"
                self.list_sinks_called += 1
                return await super().list_sinks()

            async def set_muted(self, name: str, muted: bool) -> bool:
                assert self.svc._lock.locked() is False, f"Lock held during set_muted({name}, {muted})!"
                self.set_muted_called += 1
                return await super().set_muted(name, muted)

        svc = PauseMusicService(config=PauseMusicConfig(action_pause=False, action_mute=True))
        backend = MockAudioSinkBackend([
            AudioSink(name="sink.analog", description="Analog Output", volume=45000, muted=False)
        ])
        audio = LockAssertAudioService(backend, svc)
        svc._audio_service = audio

        # 1. During handle_call_started: list_sinks, set_muted(True)
        await asyncio.wait_for(svc.handle_call_started("dev1"), timeout=1.0)
        assert audio.list_sinks_called >= 1
        assert audio.set_muted_called >= 1

        # 2. During handle_call_ended: set_muted(False)
        await asyncio.wait_for(svc.handle_call_ended("dev1"), timeout=1.0)
        assert audio.set_muted_called >= 2

        # 3. During restore_media: set_muted(False)
        await svc.handle_call_started("dev1")
        await asyncio.wait_for(svc.restore_media(), timeout=1.0)
        assert audio.set_muted_called >= 4

    async def test_lock_never_held_during_simultaneous_mpris_and_audio(
        self,
        event_bus: EventBus,
    ) -> None:
        """11.11 Lock assertion: Simultaneous MPRIS and Audio policies with lock verification."""
        class CombinedMpris(MockMprisAdapter):
            def __init__(self, svc: PauseMusicService) -> None:
                super().__init__(auto_default_player=False)
                self.svc = svc
                self.actions: list[tuple[str, str]] = []

            async def send_action(self, player: str, action: str) -> bool:
                assert self.svc._lock.locked() is False, f"Lock held on {action}!"
                self.actions.append((player, action))
                return await super().send_action(player, action)

        class CombinedAudio(LocalAudioService):
            def __init__(self, backend: MockAudioSinkBackend, svc: PauseMusicService) -> None:
                super().__init__(backend=backend)
                self.svc = svc
                self.mutes: list[tuple[str, bool]] = []

            async def set_muted(self, name: str, muted: bool) -> bool:
                assert self.svc._lock.locked() is False, f"Lock held on set_muted {muted}!"
                self.mutes.append((name, muted))
                return await super().set_muted(name, muted)

        svc = PauseMusicService(config=PauseMusicConfig(action_pause=True, action_mute=True))
        mpris = CombinedMpris(svc)
        mpris.add_mock_player(
            "org.mpris.MediaPlayer2.clementine",
            PlayerState(player="org.mpris.MediaPlayer2.clementine", is_playing=True),
        )
        backend = MockAudioSinkBackend([
            AudioSink(name="sink.speaker", description="Speaker", volume=50000, muted=False)
        ])
        audio = CombinedAudio(backend, svc)
        svc._mpris_adapter = mpris
        svc._audio_service = audio

        await asyncio.wait_for(svc.handle_call_started("dev1"), timeout=1.0)
        assert ("org.mpris.MediaPlayer2.clementine", "Pause") in mpris.actions
        assert ("sink.speaker", True) in audio.mutes

        await asyncio.wait_for(svc.handle_call_ended("dev1"), timeout=1.0)
        assert ("org.mpris.MediaPlayer2.clementine", "Play") in mpris.actions
        assert ("sink.speaker", False) in audio.mutes

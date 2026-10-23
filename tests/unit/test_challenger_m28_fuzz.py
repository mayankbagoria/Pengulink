"""tests.unit.test_challenger_m28_fuzz
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial Stress, Fuzzing & Edge-Case Verification Suite for Milestone 28:
PauseMusic Subsystem (`pengulink.services.pausemusic.PauseMusicService`,
`pengulink.services.pausemusic.PauseMusicConfig`, and
`pengulink.plugins.pausemusic.PauseMusicPlugin`).

Empirically challenges:
1. Combinatorial matrix of PauseCondition ("ringing", "talking", case insensitivity,
   whitespace, invalid types/values), PauseAction ("pause", "mute", "both",
   "pause_and_mute", "none", invalid), and action_resume (True, False).
2. Malformed, noisy, or extreme inputs in to_dict / from_dict (fuzzing types,
   coercion of boolean strings/numbers, precedence rules, schema completeness, roundtrips).
3. Edge cases in player reflection suppression: random variations of player names
   (org.mpris.MediaPlayer2.kdeconnect.*, org.mpris.MediaPlayer2.pengulink.*,
   org.mpris.MediaPlayer2.playerctld, pseudo-reflections, case sensitivity, 50-player mix).
4. Boundary cases where players change state externally while a call is active
   (player vanishes, player manually resumed/stopped, can_pause=False, D-Bus exceptions).
5. Audio sink backend failures or missing sinks (empty sinks, audio unavailable,
   OS errors, sink unplugged mid-call, timeouts, pre-muted sinks).
"""

from __future__ import annotations

import random
import string
from typing import Any
from unittest.mock import patch

import pytest

from pengulink.core.events import (
    CallStateChangedEvent,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
)
from pengulink.core.exceptions import (
    AudioUnavailableError,
    PauseMusicConfigError,
)
from pengulink.plugins.mpris.adapter import MockMprisAdapter, PlayerState
from pengulink.services.audio import (
    AudioSink,
    LocalAudioService,
    MockAudioSinkBackend,
)
from pengulink.services.pausemusic import (
    REFLECTION_EXACT,
    REFLECTION_PREFIXES,
    PauseMusicConfig,
    PauseMusicService,
)

# ---------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def event_bus() -> EventBus:
    """Isolated EventBus instance."""
    return EventBus()


@pytest.fixture
def mock_mpris() -> MockMprisAdapter:
    """MockMprisAdapter fixture with no automatic default player."""
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
    """Standard PauseMusicService fixture."""
    return PauseMusicService(
        config=PauseMusicConfig(),
        mpris_adapter=mock_mpris,
        audio_service=local_audio,
        event_bus=event_bus,
    )


# =====================================================================
# Focus Area 1: Combinatorial Policy Matrix & Permutations (12 tests)
# =====================================================================


class TestChallengerPolicyMatrix:
    """Empirically test the combinatorial matrix of pause policy configurations."""

    def test_combinatorial_valid_policy_tuples(self) -> None:
        """1.1 Verify all 16 valid policy combinations of condition, action, and resume."""
        conditions = ["ringing", "talking"]
        actions = ["pause", "mute", "both", "none"]
        resumes = [True, False]

        for cond in conditions:
            for act in actions:
                for res in resumes:
                    cfg = PauseMusicConfig()
                    cfg.condition = cond
                    cfg.action = act
                    cfg.resume = res

                    assert cfg.condition == cond
                    assert cfg.action == act
                    assert cfg.resume is res
                    assert cfg.condition_talking == (cond == "talking")
                    assert cfg.action_pause == (act in ("pause", "both"))
                    assert cfg.action_mute == (act in ("mute", "both"))
                    assert cfg.action_resume is res

    def test_condition_case_insensitivity_and_whitespace(self) -> None:
        """1.2 Test condition setter handles arbitrary casing and whitespace."""
        cfg = PauseMusicConfig()
        for variant in ["RINGING", "Ringing", "rInGiNg", "  ringing  ", "\tringing\n"]:
            cfg.condition = variant
            assert cfg.condition == "ringing"
            assert cfg.condition_talking is False

        for variant in ["TALKING", "Talking", "TaLkInG", "  talking  ", "\ttalking\n"]:
            cfg.condition = variant
            assert cfg.condition == "talking"
            assert cfg.condition_talking is True

    def test_condition_invalid_values_raise_config_error(self) -> None:
        """1.3 Test condition setter rejects invalid string values."""
        cfg = PauseMusicConfig()
        invalid_values = ["idle", "missed", "active", "call", "paused", "", "   ", "random", "both"]
        for bad in invalid_values:
            with pytest.raises(PauseMusicConfigError):
                cfg.condition = bad

    def test_condition_invalid_types_raise_config_error(self) -> None:
        """1.4 Test condition setter rejects non-string types."""
        cfg = PauseMusicConfig()
        invalid_types: list[Any] = [None, 123, 0, True, False, 3.14, ["talking"], {"cond": "ringing"}]
        for bad in invalid_types:
            with pytest.raises(PauseMusicConfigError):
                cfg.condition = bad

    def test_action_case_insensitivity_and_whitespace(self) -> None:
        """1.5 Test action setter handles arbitrary casing and whitespace."""
        cfg = PauseMusicConfig()
        cfg.action = "  PAUSE  "
        assert cfg.action == "pause"
        assert cfg.action_pause is True
        assert cfg.action_mute is False

        cfg.action = "\tMute\n"
        assert cfg.action == "mute"
        assert cfg.action_pause is False
        assert cfg.action_mute is True

        cfg.action = "  BoTh "
        assert cfg.action == "both"
        assert cfg.action_pause is True
        assert cfg.action_mute is True

        cfg.action = "  NoNe  "
        assert cfg.action == "none"
        assert cfg.action_pause is False
        assert cfg.action_mute is False

        cfg.action = ""
        assert cfg.action == "none"

    def test_action_invalid_values_raise_config_error(self) -> None:
        """1.6 Test action setter rejects invalid action strings."""
        cfg = PauseMusicConfig()
        for bad in ["play", "stop", "unmute", "disable", "all", "invalid_action", "pause-mute"]:
            with pytest.raises(PauseMusicConfigError):
                cfg.action = bad

    def test_action_pause_and_mute_handling(self) -> None:
        """1.7 Verify behavior when 'pause_and_mute' is supplied: rejects non-canonical action string."""
        cfg = PauseMusicConfig()
        with pytest.raises(PauseMusicConfigError):
            cfg.action = "pause_and_mute"
        with pytest.raises(PauseMusicConfigError):
            PauseMusicConfig.from_dict({"action": "pause_and_mute"})

    def test_action_invalid_types_raise_config_error(self) -> None:
        """1.8 Test action setter rejects non-string types."""
        cfg = PauseMusicConfig()
        for bad in [None, 0, 1, True, False, 2.5, ["both"], {"action": "pause"}]:
            with pytest.raises(PauseMusicConfigError):
                cfg.action = bad

    def test_resume_boolean_setter_and_type_enforcement(self) -> None:
        """1.9 Test resume setter enforces strict boolean typing."""
        cfg = PauseMusicConfig()
        cfg.resume = True
        assert cfg.resume is True
        cfg.resume = False
        assert cfg.resume is False

        for bad in ["True", "true", 1, 0, None, 1.0, [True]]:
            with pytest.raises(PauseMusicConfigError):
                cfg.resume = bad

    async def test_service_full_matrix_behavior_pause_action(
        self,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
        event_bus: EventBus,
    ) -> None:
        """1.10 Empirically test action='pause' with resume=True vs resume=False."""
        for resume_flag in (True, False):
            mock_mpris.clear_mock_players()
            mock_mpris.add_mock_player(
                "org.mpris.MediaPlayer2.vlc",
                PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=True),
            )
            svc = PauseMusicService(
                config=PauseMusicConfig(action_pause=True, action_mute=False, action_resume=resume_flag),
                mpris_adapter=mock_mpris,
                audio_service=local_audio,
                event_bus=event_bus,
            )

            # Call starts
            await svc.handle_call_started("dev1", "RINGING")
            st = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
            assert st is not None and st.is_playing is False
            sink = await local_audio.get_sink("alsa_output.speaker")
            assert sink is not None and sink.muted is False

            # Call ends
            await svc.handle_call_ended("dev1")
            st_after = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
            assert st_after is not None and st_after.is_playing is resume_flag

    async def test_service_full_matrix_behavior_mute_action(
        self,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """1.11 Empirically test action='mute' with resume=True vs resume=False."""
        for resume_flag in (True, False):
            mock_mpris.clear_mock_players()
            mock_mpris.add_mock_player(
                "org.mpris.MediaPlayer2.vlc",
                PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=True),
            )
            backend = MockAudioSinkBackend(
                initial_sinks=[
                    AudioSink(name="alsa_output.speaker", volume=30000, muted=False),
                ]
            )
            audio_svc = LocalAudioService(backend=backend, event_bus=event_bus)
            svc = PauseMusicService(
                config=PauseMusicConfig(action_pause=False, action_mute=True, action_resume=resume_flag),
                mpris_adapter=mock_mpris,
                audio_service=audio_svc,
                event_bus=event_bus,
            )

            # Call starts
            await svc.handle_call_started("dev1", "RINGING")
            st = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
            assert st is not None and st.is_playing is True
            sink = await audio_svc.get_sink("alsa_output.speaker")
            assert sink is not None and sink.muted is True

            # Call ends
            await svc.handle_call_ended("dev1")
            sink_after = await audio_svc.get_sink("alsa_output.speaker")
            # If resume is True, sink is unmuted (muted=False); if resume is False, remains muted (muted=True)
            assert sink_after is not None and sink_after.muted is not resume_flag

    async def test_service_full_matrix_behavior_none_action(
        self,
        mock_mpris: MockMprisAdapter,
        local_audio: LocalAudioService,
        event_bus: EventBus,
    ) -> None:
        """1.12 Empirically test action='none' takes zero action on both media and audio."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.vlc",
            PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=True),
        )
        svc = PauseMusicService(
            config=PauseMusicConfig(action_pause=False, action_mute=False, action_resume=True),
            mpris_adapter=mock_mpris,
            audio_service=local_audio,
            event_bus=event_bus,
        )
        await svc.handle_call_started("dev1", "RINGING")
        st = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert st is not None and st.is_playing is True
        sink = await local_audio.get_sink("alsa_output.speaker")
        assert sink is not None and sink.muted is False

        await svc.handle_call_ended("dev1")
        st_after = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert st_after is not None and st_after.is_playing is True
        sink_after = await local_audio.get_sink("alsa_output.speaker")
        assert sink_after is not None and sink_after.muted is False


# =====================================================================
# Focus Area 2: Malformed, Noisy & Extreme Inputs in to_dict / from_dict (10 tests)
# =====================================================================


class TestChallengerConfigSerializationFuzzing:
    """Stress tests and fuzzing for PauseMusicConfig serialization and deserialization."""

    def test_from_dict_non_dict_inputs_raise_error(self) -> None:
        """2.1 Verify from_dict rejects non-dictionary inputs."""
        non_dicts: list[Any] = [None, [], [1, 2, 3], "invalid_string", 12345, 3.14, True, False]
        for bad in non_dicts:
            with pytest.raises(PauseMusicConfigError):
                PauseMusicConfig.from_dict(bad)

    def test_from_dict_empty_dict_yields_defaults(self) -> None:
        """2.2 Verify from_dict on an empty dict yields default policy settings."""
        cfg = PauseMusicConfig.from_dict({})
        assert cfg.condition == "ringing"
        assert cfg.condition_talking is False
        assert cfg.action == "pause"
        assert cfg.action_pause is True
        assert cfg.action_mute is False
        assert cfg.action_resume is True

    def test_from_dict_fuzzy_boolean_coercion(self) -> None:
        """2.3 Fuzz boolean fields with truthy and falsy string/numeric representations."""
        truthy_inputs = ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON", "enabled", 1, 1.0]
        for val in truthy_inputs:
            cfg = PauseMusicConfig.from_dict({"condition_talking": val, "action_mute": val})
            assert cfg.condition_talking is True
            assert cfg.action_mute is True

        falsy_inputs = ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF", "disabled", "", 0, 0.0]
        for val in falsy_inputs:
            cfg = PauseMusicConfig.from_dict({"condition_talking": val, "action_pause": val})
            assert cfg.condition_talking is False
            assert cfg.action_pause is False

    def test_from_dict_camel_case_kdeconnect_keys(self) -> None:
        """2.4 Verify from_dict accepts upstream KDE Connect QML camelCase keys."""
        data = {
            "conditionTalking": True,
            "actionPause": False,
            "actionMute": True,
            "actionResume": False,
        }
        cfg = PauseMusicConfig.from_dict(data)
        assert cfg.condition_talking is True
        assert cfg.action_pause is False
        assert cfg.action_mute is True
        assert cfg.action_resume is False

    def test_from_dict_key_precedence(self) -> None:
        """2.5 Verify precedence: semantic strings vs boolean flags in from_dict."""
        # 'condition' takes precedence over 'condition_talking'
        data1 = {"condition": "talking", "condition_talking": False}
        cfg1 = PauseMusicConfig.from_dict(data1)
        assert cfg1.condition == "talking"
        assert cfg1.condition_talking is True

        # 'action_pause' takes precedence over 'action' when explicitly provided
        data2 = {"action": "mute", "action_pause": True}
        cfg2 = PauseMusicConfig.from_dict(data2)
        assert cfg2.action_pause is True
        assert cfg2.action_mute is False

    def test_from_dict_invalid_condition_value(self) -> None:
        """2.6 Verify invalid condition string in dict raises PauseMusicConfigError."""
        with pytest.raises(PauseMusicConfigError):
            PauseMusicConfig.from_dict({"condition": "hangup"})

    def test_from_dict_invalid_action_value(self) -> None:
        """2.7 Verify invalid action string in dict raises PauseMusicConfigError."""
        with pytest.raises(PauseMusicConfigError):
            PauseMusicConfig.from_dict({"action": "destroy"})

    def test_from_dict_noisy_and_extreme_keys(self) -> None:
        """2.8 Verify resilient handling of large dicts with noise, Unicode, and extreme payloads."""
        noisy_dict: dict[str, Any] = {
            "condition": "talking",
            "action": "both",
            "action_resume": True,
            # Extraneous noise
            "unknown_key": "some_value",
            "giant_payload": "X" * 100_000,
            "emoji_key_🚀": "🔥",
            "deeply_nested": {"a": {"b": {"c": 123}}},
            "null_field": None,
        }
        # Add 100 random keys
        for i in range(100):
            noisy_dict[f"noise_key_{i}"] = i * 42

        cfg = PauseMusicConfig.from_dict(noisy_dict)
        assert cfg.condition == "talking"
        assert cfg.action == "both"
        assert cfg.action_resume is True

    def test_to_dict_schema_completeness(self) -> None:
        """2.9 Verify to_dict output includes all expected canonical and QML keys."""
        cfg = PauseMusicConfig(
            condition_talking=True,
            action_pause=True,
            action_mute=True,
            action_resume=False,
        )
        d = cfg.to_dict()
        expected_keys = {
            "condition_talking",
            "action_pause",
            "action_mute",
            "action_resume",
            "condition",
            "action",
            "resume",
            "conditionTalking",
            "actionPause",
            "actionMute",
            "actionResume",
        }
        assert expected_keys.issubset(d.keys())
        assert d["condition"] == "talking"
        assert d["action"] == "both"
        assert d["action_resume"] is False
        assert d["resume"] is False

    def test_config_serialization_roundtrip(self) -> None:
        """2.10 Verify lossless roundtrip serialization (to_dict -> from_dict -> to_dict)."""
        test_cases = [
            PauseMusicConfig(condition_talking=False, action_pause=True, action_mute=False, action_resume=True),
            PauseMusicConfig(condition_talking=True, action_pause=False, action_mute=True, action_resume=False),
            PauseMusicConfig(condition_talking=False, action_pause=True, action_mute=True, action_resume=True),
            PauseMusicConfig(condition_talking=True, action_pause=False, action_mute=False, action_resume=False),
        ]
        for original in test_cases:
            serialized = original.to_dict()
            restored = PauseMusicConfig.from_dict(serialized)
            assert restored.condition_talking == original.condition_talking
            assert restored.action_pause == original.action_pause
            assert restored.action_mute == original.action_mute
            assert restored.action_resume == original.action_resume
            assert restored.to_dict() == serialized


# =====================================================================
# Focus Area 3: Player Reflection Suppression Edge Cases (8 tests)
# =====================================================================


class TestChallengerReflectionSuppression:
    """Stress tests on player reflection suppression and namespace boundary cases."""

    async def test_kdeconnect_prefix_random_hashes_and_uuids(
        self,
        mock_mpris: MockMprisAdapter,
        pausemusic_service: PauseMusicService,
    ) -> None:
        """3.1 Verify arbitrary KDE Connect device player names are strictly suppressed."""
        for i in range(10):
            rand_hash = "".join(random.choices(string.hexdigits.lower(), k=16))
            player_name = f"org.mpris.MediaPlayer2.kdeconnect.a{rand_hash}"
            mock_mpris.add_mock_player(
                player_name,
                PlayerState(player=player_name, is_playing=True),
            )

        # Also add a real player
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.rhythmbox",
            PlayerState(player="org.mpris.MediaPlayer2.rhythmbox", is_playing=True),
        )

        await pausemusic_service.handle_call_started("dev1", "RINGING")

        # All 10 kdeconnect reflection players must remain is_playing=True (untouched)
        for p in await mock_mpris.list_players():
            st = await mock_mpris.get_player_status(p)
            assert st is not None
            if p.startswith("org.mpris.MediaPlayer2.kdeconnect."):
                assert st.is_playing is True
            elif p == "org.mpris.MediaPlayer2.rhythmbox":
                assert st.is_playing is False

        assert pausemusic_service.paused_players == {"org.mpris.MediaPlayer2.rhythmbox"}

    async def test_kdeconnect_prefix_with_special_chars(
        self,
        mock_mpris: MockMprisAdapter,
        pausemusic_service: PauseMusicService,
    ) -> None:
        """3.2 Test reflection suppression handles special characters in device suffixes."""
        special_names = [
            "org.mpris.MediaPlayer2.kdeconnect._device-test@domain:123",
            "org.mpris.MediaPlayer2.kdeconnect.手机_phone",
            "org.mpris.MediaPlayer2.kdeconnect.",
        ]
        for name in special_names:
            mock_mpris.add_mock_player(name, PlayerState(player=name, is_playing=True))

        await pausemusic_service.handle_call_started("dev1", "RINGING")
        assert len(pausemusic_service.paused_players) == 0

    async def test_pengulink_prefix_variations(
        self,
        mock_mpris: MockMprisAdapter,
        pausemusic_service: PauseMusicService,
    ) -> None:
        """3.3 Test pengulink daemon internal proxy reflections are suppressed."""
        pengulink_players = [
            "org.mpris.MediaPlayer2.pengulink.daemon",
            "org.mpris.MediaPlayer2.pengulink.subsystem.test",
            "org.mpris.MediaPlayer2.pengulink.instance_1",
        ]
        for name in pengulink_players:
            mock_mpris.add_mock_player(name, PlayerState(player=name, is_playing=True))

        await pausemusic_service.handle_call_started("dev1", "RINGING")
        assert len(pausemusic_service.paused_players) == 0

    async def test_playerctld_exact_match_ignored(
        self,
        mock_mpris: MockMprisAdapter,
        pausemusic_service: PauseMusicService,
    ) -> None:
        """3.4 Test org.mpris.MediaPlayer2.playerctld exact match is suppressed."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.playerctld",
            PlayerState(player="org.mpris.MediaPlayer2.playerctld", is_playing=True),
        )
        await pausemusic_service.handle_call_started("dev1", "RINGING")
        assert "org.mpris.MediaPlayer2.playerctld" not in pausemusic_service.paused_players

    async def test_playerctld_subpath_or_prefix_not_suppressed(
        self,
        mock_mpris: MockMprisAdapter,
        pausemusic_service: PauseMusicService,
    ) -> None:
        """3.5 Test that playerctld is an exact match: players named differently are NOT suppressed."""
        name = "org.mpris.MediaPlayer2.playerctld_custom"
        mock_mpris.add_mock_player(name, PlayerState(player=name, is_playing=True))
        await pausemusic_service.handle_call_started("dev1", "RINGING")
        assert name in pausemusic_service.paused_players

    async def test_legitimate_players_resembling_reflections_are_not_suppressed(
        self,
        mock_mpris: MockMprisAdapter,
        pausemusic_service: PauseMusicService,
    ) -> None:
        """3.6 Verify that legitimate players missing the trailing dot are NOT mistakenly suppressed."""
        # Upstream prefix is 'org.mpris.MediaPlayer2.kdeconnect.' with a dot.
        # A hypothetical player named 'org.mpris.MediaPlayer2.kdeconnect_extra' must NOT be suppressed.
        names = [
            "org.mpris.MediaPlayer2.kdeconnect_extra",
            "org.mpris.MediaPlayer2.pengulink_client",
            "org.mpris.MediaPlayer2.spotify",
            "org.mpris.MediaPlayer2.vlc.instance_42",
        ]
        for name in names:
            mock_mpris.add_mock_player(name, PlayerState(player=name, is_playing=True))

        await pausemusic_service.handle_call_started("dev1", "RINGING")
        assert set(names).issubset(pausemusic_service.paused_players)

    async def test_case_sensitivity_of_dbus_reflection_prefixes(
        self,
        mock_mpris: MockMprisAdapter,
        pausemusic_service: PauseMusicService,
    ) -> None:
        """3.7 Verify D-Bus case sensitivity: lowercase 'mediaplayer2' is treated as distinct."""
        # D-Bus well-known names are case-sensitive.
        name = "org.mpris.mediaplayer2.kdeconnect.device1"
        mock_mpris.add_mock_player(name, PlayerState(player=name, is_playing=True))

        await pausemusic_service.handle_call_started("dev1", "RINGING")
        # Since it does not start with exact "org.mpris.MediaPlayer2.kdeconnect.", it is treated as a real player
        assert name in pausemusic_service.paused_players

    async def test_stress_mixed_roster_50_players(
        self,
        mock_mpris: MockMprisAdapter,
        pausemusic_service: PauseMusicService,
    ) -> None:
        """3.8 Stress test mixed roster of 50 players: 25 reflections and 25 legitimate players."""
        expected_paused: set[str] = set()

        for i in range(25):
            ref_name = f"org.mpris.MediaPlayer2.kdeconnect.peer_{i}"
            mock_mpris.add_mock_player(ref_name, PlayerState(player=ref_name, is_playing=True))

            real_name = f"org.mpris.MediaPlayer2.player_{i}"
            mock_mpris.add_mock_player(real_name, PlayerState(player=real_name, is_playing=True))
            expected_paused.add(real_name)

        await pausemusic_service.handle_call_started("dev1", "RINGING")
        assert pausemusic_service.paused_players == expected_paused

        await pausemusic_service.handle_call_ended("dev1")
        for p in expected_paused:
            st = await mock_mpris.get_player_status(p)
            assert st is not None and st.is_playing is True

    def test_reflection_constants_definitions(self) -> None:
        """3.9 Verify reflection constants structure and essential entries."""
        assert isinstance(REFLECTION_PREFIXES, tuple)
        assert "org.mpris.MediaPlayer2.kdeconnect." in REFLECTION_PREFIXES
        assert "org.mpris.MediaPlayer2.pengulink." in REFLECTION_PREFIXES
        assert isinstance(REFLECTION_EXACT, set)
        assert "org.mpris.MediaPlayer2.playerctld" in REFLECTION_EXACT


# =====================================================================
# Focus Area 4: External Player State Mutations During Active Call (7 tests)
# =====================================================================


class TestChallengerExternalPlayerMutations:
    """Boundary cases where players change state externally while a call is active."""

    async def test_player_vanishes_abruptly_during_call(
        self,
        mock_mpris: MockMprisAdapter,
        pausemusic_service: PauseMusicService,
    ) -> None:
        """4.1 Player was playing, paused by call, then abruptly killed before call ends."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.spotify",
            PlayerState(player="org.mpris.MediaPlayer2.spotify", is_playing=True),
        )
        await pausemusic_service.handle_call_started("dev1", "RINGING")
        assert "org.mpris.MediaPlayer2.spotify" in pausemusic_service.paused_players

        # Player process terminates externally
        mock_mpris.remove_mock_player("org.mpris.MediaPlayer2.spotify")

        # Call ends: service handles missing player gracefully without raising
        result = await pausemusic_service.handle_call_ended("dev1")
        assert result is True
        assert len(pausemusic_service.paused_players) == 0

    async def test_player_manually_resumed_by_user_during_call(
        self,
        mock_mpris: MockMprisAdapter,
        pausemusic_service: PauseMusicService,
    ) -> None:
        """4.2 User manually clicks Play on desktop while call is active."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.vlc",
            PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=True),
        )
        await pausemusic_service.handle_call_started("dev1", "RINGING")

        # User unpauses VLC manually during active call
        await mock_mpris.send_action("org.mpris.MediaPlayer2.vlc", "Play")
        st_mid = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert st_mid is not None and st_mid.is_playing is True

        # When call ends, service sends Play (idempotent), snapshot cleared
        await pausemusic_service.handle_call_ended("dev1")
        st_end = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert st_end is not None and st_end.is_playing is True
        assert len(pausemusic_service.paused_players) == 0

    async def test_pre_call_paused_player_started_by_user_during_call(
        self,
        mock_mpris: MockMprisAdapter,
        pausemusic_service: PauseMusicService,
    ) -> None:
        """4.3 Player was paused before call, started mid-call: not affected on call end."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.mpv",
            PlayerState(player="org.mpris.MediaPlayer2.mpv", is_playing=False),
        )
        await pausemusic_service.handle_call_started("dev1", "RINGING")
        assert "org.mpris.MediaPlayer2.mpv" not in pausemusic_service.paused_players

        # User starts mpv during the call
        await mock_mpris.send_action("org.mpris.MediaPlayer2.mpv", "Play")

        # Call ends: mpv was not in initial paused_players snapshot, stays playing
        await pausemusic_service.handle_call_ended("dev1")
        st = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.mpv")
        assert st is not None and st.is_playing is True

    async def test_player_can_pause_false_can_stop_true_external_change(
        self,
        mock_mpris: MockMprisAdapter,
        pausemusic_service: PauseMusicService,
    ) -> None:
        """4.4 Player with can_pause=False uses Stop instead of Pause, resumes with Play."""
        st_radio = PlayerState(
            player="org.mpris.MediaPlayer2.radio",
            is_playing=True,
            can_pause=False,
        )
        st_radio.can_stop = True
        mock_mpris.add_mock_player("org.mpris.MediaPlayer2.radio", st_radio)

        await pausemusic_service.handle_call_started("dev1", "RINGING")
        st = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.radio")
        assert st is not None and st.is_playing is False

        await pausemusic_service.handle_call_ended("dev1")
        st_after = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.radio")
        assert st_after is not None and st_after.is_playing is True

    async def test_player_action_raises_runtime_error_on_pause(
        self,
        mock_mpris: MockMprisAdapter,
        pausemusic_service: PauseMusicService,
    ) -> None:
        """4.5 send_action raises RuntimeError during pause: caught and handled safely."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.faulty",
            PlayerState(player="org.mpris.MediaPlayer2.faulty", is_playing=True),
        )
        with patch.object(mock_mpris, "send_action", side_effect=RuntimeError("D-Bus transport dead")):
            result = await pausemusic_service.handle_call_started("dev1", "RINGING")
            assert result is True

    async def test_player_action_raises_timeout_error_on_resume(
        self,
        mock_mpris: MockMprisAdapter,
        pausemusic_service: PauseMusicService,
    ) -> None:
        """4.6 send_action raises TimeoutError during resume: caught safely and snapshot cleared."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.hanging",
            PlayerState(player="org.mpris.MediaPlayer2.hanging", is_playing=True),
        )
        await pausemusic_service.handle_call_started("dev1", "RINGING")
        assert "org.mpris.MediaPlayer2.hanging" in pausemusic_service.paused_players

        with patch.object(mock_mpris, "send_action", side_effect=TimeoutError("Timeout resuming player")):
            result = await pausemusic_service.handle_call_ended("dev1")
            assert result is True
            # Snapshot must still be cleared
            assert len(pausemusic_service.paused_players) == 0

    async def test_new_player_appears_and_plays_during_call(
        self,
        mock_mpris: MockMprisAdapter,
        pausemusic_service: PauseMusicService,
    ) -> None:
        """4.7 Brand new player started mid-call is not in snapshot and not altered on call end."""
        await pausemusic_service.handle_call_started("dev1", "RINGING")

        # New player spawned while call is active
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.newapp",
            PlayerState(player="org.mpris.MediaPlayer2.newapp", is_playing=True),
        )

        await pausemusic_service.handle_call_ended("dev1")
        st = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.newapp")
        assert st is not None and st.is_playing is True

    async def test_eventbus_malformed_event_dispatch(
        self,
        pausemusic_service: PauseMusicService,
        event_bus: EventBus,
    ) -> None:
        """4.8 Fuzz EventBus events with empty device_ids, unknown states, and unexpected payloads."""
        # Empty device ID should not crash service or trigger interruption
        await event_bus.publish(CallStateChangedEvent(device_id="", new_state="RINGING"))
        assert pausemusic_service.is_interrupted is False

        # Unknown state string should be safely ignored
        await event_bus.publish(CallStateChangedEvent(device_id="dev1", new_state="UNKNOWN_STATE"))
        assert pausemusic_service.is_interrupted is False

        # DeviceDisconnected with empty device_id
        await event_bus.publish(DeviceDisconnected(device_id=""))
        assert pausemusic_service.is_interrupted is False

        # DeviceUnpaired with empty device_id
        await event_bus.publish(DeviceUnpaired(device_id=""))
        assert pausemusic_service.is_interrupted is False


# =====================================================================
# Focus Area 5: Audio Sink Backend Failures & Extreme Edge Cases (7 tests)
# =====================================================================


class TestChallengerAudioSinkFailures:
    """Stress tests on audio sink backend failures and missing sinks."""

    async def test_audio_service_returns_empty_sink_list(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """5.1 Backend has zero sinks (headless server): clean execution without error."""
        backend = MockAudioSinkBackend(initial_sinks=[])
        audio_svc = LocalAudioService(backend=backend, event_bus=event_bus)
        svc = PauseMusicService(
            config=PauseMusicConfig(action_pause=False, action_mute=True),
            mpris_adapter=mock_mpris,
            audio_service=audio_svc,
            event_bus=event_bus,
        )

        res_start = await svc.handle_call_started("dev1", "RINGING")
        assert res_start is True
        assert len(svc.muted_sinks) == 0

        res_end = await svc.handle_call_ended("dev1")
        assert res_end is True

    async def test_audio_service_list_sinks_raises_audio_unavailable(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """5.2 list_sinks raises AudioUnavailableError: caught and does not crash."""
        audio_svc = LocalAudioService(event_bus=event_bus)
        svc = PauseMusicService(
            config=PauseMusicConfig(action_pause=False, action_mute=True),
            mpris_adapter=mock_mpris,
            audio_service=audio_svc,
            event_bus=event_bus,
        )

        with patch.object(audio_svc, "list_sinks", side_effect=AudioUnavailableError("PipeWire daemon crashed")):
            result = await svc.handle_call_started("dev1", "RINGING")
            assert result is True
            assert len(svc.muted_sinks) == 0

    async def test_audio_service_list_sinks_raises_os_error(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """5.3 list_sinks raises OSError: caught by COORDINATION_EXCEPTIONS."""
        audio_svc = LocalAudioService(event_bus=event_bus)
        svc = PauseMusicService(
            config=PauseMusicConfig(action_pause=False, action_mute=True),
            mpris_adapter=mock_mpris,
            audio_service=audio_svc,
            event_bus=event_bus,
        )

        with patch.object(audio_svc, "list_sinks", side_effect=OSError("Connection reset by peer")):
            result = await svc.handle_call_started("dev1", "RINGING")
            assert result is True

    async def test_sink_unplugged_mid_call(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """5.4 Sink muted at call start is removed mid-call: unmuting handled gracefully."""
        backend = MockAudioSinkBackend(
            initial_sinks=[
                AudioSink(name="usb_headset", volume=50000, muted=False),
            ]
        )
        audio_svc = LocalAudioService(backend=backend, event_bus=event_bus)
        svc = PauseMusicService(
            config=PauseMusicConfig(action_pause=False, action_mute=True),
            mpris_adapter=mock_mpris,
            audio_service=audio_svc,
            event_bus=event_bus,
        )

        await svc.handle_call_started("dev1", "RINGING")
        assert "usb_headset" in svc.muted_sinks

        # User unplugs the USB headset mid-call
        backend.remove_sink("usb_headset")

        # Call ends: set_muted for missing sink returns False, service does not crash
        result = await svc.handle_call_ended("dev1")
        assert result is True
        assert len(svc.muted_sinks) == 0

    async def test_sink_set_muted_raises_timeout(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """5.5 set_muted raises TimeoutError: caught safely and does not crash service."""
        backend = MockAudioSinkBackend(
            initial_sinks=[
                AudioSink(name="sink_slow", volume=50000, muted=False),
                AudioSink(name="sink_fast", volume=40000, muted=False),
            ]
        )
        audio_svc = LocalAudioService(backend=backend, event_bus=event_bus)
        svc = PauseMusicService(
            config=PauseMusicConfig(action_pause=False, action_mute=True),
            mpris_adapter=mock_mpris,
            audio_service=audio_svc,
            event_bus=event_bus,
        )

        with patch.object(audio_svc, "set_muted", side_effect=TimeoutError("IPC call timed out")):
            result = await svc.handle_call_started("dev1", "RINGING")
            # Coordination exception caught by COORDINATION_EXCEPTIONS without throwing
            assert result is True

    async def test_all_sinks_already_muted_before_call(
        self,
        event_bus: EventBus,
        mock_mpris: MockMprisAdapter,
    ) -> None:
        """5.6 Sinks that were already muted before the call are never unmuted on call end."""
        backend = MockAudioSinkBackend(
            initial_sinks=[
                AudioSink(name="sink1", volume=50000, muted=True),
                AudioSink(name="sink2", volume=40000, muted=True),
            ]
        )
        audio_svc = LocalAudioService(backend=backend, event_bus=event_bus)
        svc = PauseMusicService(
            config=PauseMusicConfig(action_pause=False, action_mute=True, action_resume=True),
            mpris_adapter=mock_mpris,
            audio_service=audio_svc,
            event_bus=event_bus,
        )

        await svc.handle_call_started("dev1", "RINGING")
        assert len(svc.muted_sinks) == 0

        await svc.handle_call_ended("dev1")
        sink1 = await audio_svc.get_sink("sink1")
        sink2 = await audio_svc.get_sink("sink2")
        assert sink1 is not None and sink1.muted is True
        assert sink2 is not None and sink2.muted is True

    async def test_audio_service_none_handled_safely(
        self,
        mock_mpris: MockMprisAdapter,
        event_bus: EventBus,
    ) -> None:
        """5.7 When _audio_service is None, action_mute=True executes safely without AttributeError."""
        mock_mpris.add_mock_player(
            "org.mpris.MediaPlayer2.vlc",
            PlayerState(player="org.mpris.MediaPlayer2.vlc", is_playing=True),
        )
        svc = PauseMusicService(
            config=PauseMusicConfig(action_pause=True, action_mute=True),
            mpris_adapter=mock_mpris,
            event_bus=event_bus,
        )
        # Explicitly decouple audio service to test None guard
        svc._audio_service = None

        res_start = await svc.handle_call_started("dev1", "RINGING")
        assert res_start is True
        st = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert st is not None and st.is_playing is False
        assert len(svc.muted_sinks) == 0

        res_end = await svc.handle_call_ended("dev1")
        assert res_end is True
        st_after = await mock_mpris.get_player_status("org.mpris.MediaPlayer2.vlc")
        assert st_after is not None and st_after.is_playing is True

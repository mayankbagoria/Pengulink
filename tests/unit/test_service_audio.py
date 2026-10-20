"""tests.unit.test_service_audio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit test suite for PenguLink local audio subsystem (pengulink.services.audio):
- AudioSink data model, property scaling, and JSON wire serialization
- MockAudioSinkBackend in-memory fixture management and mutators
- SilentFallbackSinkBackend no-op safety
- PipeWireAudioBackend wpctl status parsing and command dispatch
- PulseAudioBackend JSON and text fallback parsing and command dispatch
- detect_audio_backend auto-detection and environment overrides
- LocalAudioService concurrency protection and EventBus publishing
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from pengulink.core.events import (
    AudioSinkListEvent,
    AudioSinkUpdatedEvent,
    EventBus,
)
from pengulink.core.exceptions import AudioBackendError
from pengulink.services.audio import (
    MAX_AUDIO_VOLUME,
    AudioSink,
    LocalAudioService,
    MockAudioSinkBackend,
    PipeWireAudioBackend,
    PulseAudioBackend,
    SilentFallbackSinkBackend,
    detect_audio_backend,
    detect_audio_sink_backend,
)

SAMPLE_WPCTL_STATUS = """
PipeWire 'pipewire-0' [1.6.2, user@host, cookie:12345]
 └─ Clients:
        33. WirePlumber                         [1.6.2, user@host, pid:1234]

Audio
 ├─ Devices:
 │      48. High Definition Audio Controller [alsa]
 │  
 ├─ Sinks:
 │      58. HDMI / DisplayPort 3 Output [vol: 1.00]
 │      59. HDMI / DisplayPort 2 Output [vol: 0.75]
 │  *   61. Internal Speaker [vol: 0.54 MUTED]
 │  
 ├─ Sources:
 │  *   62. Headset Microphone [vol: 1.00]
 │  
 ├─ Filters:
 │  
 └─ Streams:
"""

SAMPLE_PACTL_JSON_SINKS = [
    {
        "name": "alsa_output.pci-0000_00_1f.3.speaker",
        "description": "Internal Speaker",
        "mute": True,
        "volume": {
            "front-left": {"value": 35389, "value_percent": "54%"},
            "front-right": {"value": 35389, "value_percent": "54%"},
        },
        "base_volume": 65536,
    },
    {
        "name": "alsa_output.pci-0000_00_1f.3.hdmi",
        "description": "HDMI Audio",
        "mute": False,
        "volume": {
            "front-left": {"value": 65536, "value_percent": "100%"},
            "front-right": {"value": 65536, "value_percent": "100%"},
        },
        "base_volume": 65536,
    },
]

SAMPLE_PACTL_TEXT_SINKS = """
Sink #1
\tState: RUNNING
\tName: alsa_output.pci-0000_00_1f.3.speaker
\tDescription: Internal Speaker
\tMute: yes
\tVolume: front-left: 35389 /  54% / -16.03 dB,   front-right: 35389 /  54% / -16.03 dB
\tBase Volume: 65536 / 100% / 0.00 dB

Sink #2
\tState: IDLE
\tName: alsa_output.pci-0000_00_1f.3.hdmi
\tDescription: HDMI Output
\tMute: no
\tVolume: front-left: 65536 / 100% / 0.00 dB,   front-right: 65536 / 100% / 0.00 dB
"""


class TestAudioSinkDataModel:
    """Tests for the AudioSink dataclass and serialization methods."""

    def test_audio_sink_defaults_and_clamping(self) -> None:
        sink = AudioSink(name="sink1")
        assert sink.name == "sink1"
        assert sink.description == "sink1"
        assert sink.volume == 0
        assert sink.max_volume == MAX_AUDIO_VOLUME
        assert not sink.muted
        assert not sink.enabled
        assert sink.volume_percent == 0

        # Clamping above max_volume
        over_sink = AudioSink(name="over", volume=100000, max_volume=65536)
        assert over_sink.volume == 65536
        assert over_sink.volume_percent == 100

        # Clamping negative
        under_sink = AudioSink(name="under", volume=-50, max_volume=65536)
        assert under_sink.volume == 0
        assert under_sink.volume_percent == 0

    def test_volume_percent_calculation(self) -> None:
        sink = AudioSink(name="sink", volume=32768, max_volume=65536)
        assert sink.volume_percent == 50

        sink_zero_max = AudioSink(name="zero", volume=0, max_volume=0)
        assert sink_zero_max.max_volume == MAX_AUDIO_VOLUME

    def test_to_wire_dict_conforms_to_kdeconnect_spec(self) -> None:
        sink = AudioSink(
            name="alsa.speaker",
            description="Speakers",
            volume=35389,
            max_volume=65536,
            muted=True,
            enabled=True,
        )
        wire = sink.to_wire_dict()
        assert wire == {
            "name": "alsa.speaker",
            "description": "Speakers",
            "muted": True,
            "volume": 35389,
            "maxVolume": 65536,
            "enabled": True,
        }
        assert "maxVolume" in wire

    def test_to_dict_contains_extended_fields(self) -> None:
        sink = AudioSink(
            name="alsa.speaker",
            description="Speakers",
            volume=32768,
            max_volume=65536,
            muted=False,
            enabled=True,
        )
        d = sink.to_dict()
        assert d["name"] == "alsa.speaker"
        assert d["volume_percent"] == 50
        assert d["maxVolume"] == 65536
        assert d["max_volume"] == 65536
        assert d["enabled"] is True

    def test_from_dict_parsing_variants(self) -> None:
        # 1. Standard wire dictionary
        wire_data = {
            "name": "speaker1",
            "description": "Internal Speaker",
            "muted": False,
            "volume": 32768,
            "maxVolume": 65536,
            "enabled": True,
        }
        sink1 = AudioSink.from_dict(wire_data)
        assert sink1.name == "speaker1"
        assert sink1.description == "Internal Speaker"
        assert sink1.volume == 32768
        assert sink1.max_volume == 65536
        assert sink1.enabled is True

        # 2. Legacy / alternate field names (desc instead of description)
        legacy_data = {
            "name": "speaker2",
            "desc": "Legacy Speaker Desc",
            "muted": True,
            "volume": "16384",
            "max_volume": 65536,
        }
        sink2 = AudioSink.from_dict(legacy_data)
        assert sink2.description == "Legacy Speaker Desc"
        assert sink2.volume == 16384
        assert sink2.muted is True

        # 3. Float volume and fallback description
        float_data = {
            "name": "speaker3",
            "volume": 42000.5,
        }
        sink3 = AudioSink.from_dict(float_data)
        assert sink3.volume == 42000
        assert sink3.description == "speaker3"

        # 4. Non-numeric or invalid maxVolume / volume fallbacks
        bad_data = {
            "name": "speaker4",
            "volume": "inf",
            "maxVolume": "invalid",
        }
        sink4 = AudioSink.from_dict(bad_data)
        assert sink4.volume == 0
        assert sink4.max_volume == MAX_AUDIO_VOLUME

        overflow_data = {
            "name": "speaker5",
            "volume": 1e1000,
            "maxVolume": -10,
        }
        sink5 = AudioSink.from_dict(overflow_data)
        assert sink5.volume == 0
        assert sink5.max_volume == MAX_AUDIO_VOLUME


class TestMockAudioSinkBackend:
    """Tests for MockAudioSinkBackend in-memory operations."""

    async def test_initial_sinks_and_availability(self) -> None:
        backend = MockAudioSinkBackend()
        assert await backend.is_available() is True
        sinks = await backend.list_sinks()
        assert len(sinks) == 2
        sink_names = [s.name for s in sinks]
        assert "alsa_output.pci-0000_00_1f.3.speaker" in sink_names
        assert "alsa_output.pci-0000_00_1f.3.hdmi" in sink_names

    async def test_custom_initial_sinks(self) -> None:
        custom_sink = AudioSink(name="custom1", description="Custom", volume=1000)
        backend = MockAudioSinkBackend(initial_sinks=[custom_sink])
        sinks = await backend.list_sinks()
        assert len(sinks) == 1
        assert sinks[0].name == "custom1"

    async def test_add_remove_reset_sinks(self) -> None:
        backend = MockAudioSinkBackend()
        backend.reset()
        assert len(await backend.list_sinks()) == 0

        new_sink = AudioSink(name="new_sink", volume=50000)
        backend.add_sink(new_sink)
        sinks = await backend.list_sinks()
        assert len(sinks) == 1
        assert sinks[0].name == "new_sink"

        assert backend.remove_sink("new_sink") is True
        assert backend.remove_sink("nonexistent") is False
        assert len(await backend.list_sinks()) == 0

    async def test_set_volume_unmutes_automatically(self) -> None:
        backend = MockAudioSinkBackend()
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"

        # Mute first
        await backend.set_muted(speaker, True)
        sinks = await backend.list_sinks()
        assert next(s for s in sinks if s.name == speaker).muted is True

        # Setting volume must automatically unmute
        success = await backend.set_volume(speaker, 45000)
        assert success is True
        sinks_after = await backend.list_sinks()
        updated = next(s for s in sinks_after if s.name == speaker)
        assert updated.volume == 45000
        assert updated.muted is False

    async def test_set_volume_clamping_and_missing_sink(self) -> None:
        backend = MockAudioSinkBackend()
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"

        # Over max
        assert await backend.set_volume(speaker, 999999) is True
        sinks = await backend.list_sinks()
        assert next(s for s in sinks if s.name == speaker).volume == 65536

        # Missing sink
        assert await backend.set_volume("no_such_sink", 30000) is False

    async def test_set_muted_and_missing_sink(self) -> None:
        backend = MockAudioSinkBackend()
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"

        assert await backend.set_muted(speaker, True) is True
        sinks = await backend.list_sinks()
        assert next(s for s in sinks if s.name == speaker).muted is True

        # set_mute alias
        assert await backend.set_mute(speaker, False) is True
        sinks = await backend.list_sinks()
        assert next(s for s in sinks if s.name == speaker).muted is False

        assert await backend.set_muted("unknown", True) is False

    async def test_set_default_sink(self) -> None:
        backend = MockAudioSinkBackend()
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"
        hdmi = "alsa_output.pci-0000_00_1f.3.hdmi"

        assert await backend.set_default(hdmi) is True
        sinks = await backend.list_sinks()
        assert next(s for s in sinks if s.name == hdmi).enabled is True
        assert next(s for s in sinks if s.name == speaker).enabled is False

        assert await backend.set_default("unknown_sink") is False


class TestSilentFallbackSinkBackend:
    """Tests for SilentFallbackSinkBackend no-op operations."""

    async def test_silent_fallback_behavior(self) -> None:
        backend = SilentFallbackSinkBackend()
        assert await backend.is_available() is True
        assert await backend.list_sinks() == []
        assert await backend.set_volume("sink1", 50000) is False
        assert await backend.set_muted("sink1", True) is False
        assert await backend.set_default("sink1") is False


class TestPipeWireAudioBackend:
    """Tests for PipeWireAudioBackend wpctl command parsing and execution."""

    async def test_is_available_when_wpctl_succeeds(self) -> None:
        backend = PipeWireAudioBackend()
        with (
            patch.object(backend, "is_supported", return_value=True),
            patch.object(backend, "_run_command", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = (0, SAMPLE_WPCTL_STATUS, "")
            assert await backend.is_available() is True

            mock_run.return_value = (1, "", "wpctl not found")
            assert await backend.is_available() is False

    async def test_list_sinks_parsing(self) -> None:
        backend = PipeWireAudioBackend()
        with patch.object(backend, "_run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, SAMPLE_WPCTL_STATUS, "")
            sinks = await backend.list_sinks()

            assert len(sinks) == 3
            # Check HDMI 3 Output (ID 58)
            hdmi3 = next(s for s in sinks if "58" in s.name or "HDMI" in s.description)
            assert hdmi3.volume == 65536
            assert not hdmi3.muted
            assert not hdmi3.enabled

            # Check HDMI 2 Output (ID 59, 75% volume = ~49152)
            hdmi2 = next(
                s
                for s in sinks
                if "59" in s.name or "HDMI / DisplayPort 2" in s.description
            )
            assert hdmi2.volume == round(0.75 * 65536)
            assert not hdmi2.muted

            # Check Internal Speaker (ID 61, marked with * for default, 54% volume MUTED)
            speaker = next(
                s for s in sinks if "61" in s.name or "Speaker" in s.description
            )
            assert speaker.volume == round(0.54 * 65536)
            assert speaker.muted is True
            assert speaker.enabled is True

    async def test_pw_dump_node_name_enrichment(self) -> None:
        backend = PipeWireAudioBackend()
        pw_dump_sample = json.dumps(
            [
                {
                    "id": 61,
                    "info": {
                        "props": {
                            "media.class": "Audio/Sink",
                            "node.name": "alsa_output.pci.speaker.sink",
                        }
                    },
                }
            ]
        )

        async def fake_run(cmd: list[str]):
            if cmd[0] == "wpctl":
                return 0, SAMPLE_WPCTL_STATUS, ""
            if cmd[0] == "pw-dump":
                return 0, pw_dump_sample, ""
            return 1, "", ""

        with (
            patch("shutil.which", return_value="/usr/bin/pw-dump"),
            patch.object(backend, "_run_command", side_effect=fake_run),
        ):
            sinks = await backend.list_sinks()
            speaker = next(s for s in sinks if "Speaker" in s.description)
            assert speaker.name == "alsa_output.pci.speaker.sink"
            assert speaker.enabled is True

    async def test_pw_dump_with_null_props_safe_handling(self) -> None:
        backend = PipeWireAudioBackend()
        pw_dump_sample = json.dumps(
            [
                {
                    "id": 99,
                    "info": {
                        "props": None,
                    },
                },
                {
                    "id": 100,
                    "info": None,
                },
                {
                    "id": 61,
                    "info": {
                        "props": {
                            "media.class": "Audio/Sink",
                            "node.name": "alsa_output.pci.speaker.sink",
                        }
                    },
                },
            ]
        )

        async def fake_run(cmd: list[str]):
            if cmd[0] == "wpctl":
                return 0, SAMPLE_WPCTL_STATUS, ""
            if cmd[0] == "pw-dump":
                return 0, pw_dump_sample, ""
            return 1, "", ""

        with (
            patch("shutil.which", return_value="/usr/bin/pw-dump"),
            patch.object(backend, "_run_command", side_effect=fake_run),
        ):
            sinks = await backend.list_sinks()
            speaker = next(s for s in sinks if "Speaker" in s.description)
            assert speaker.name == "alsa_output.pci.speaker.sink"

    async def test_set_volume_invokes_wpctl_and_unmutes(self) -> None:
        backend = PipeWireAudioBackend()
        with patch.object(backend, "_run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, "", "")
            success = await backend.set_volume("61", 32768)
            assert success is True

            # Must have called wpctl set-volume with 0.50 and wpctl set-mute with 0
            assert mock_run.call_count == 2
            set_vol_call = mock_run.call_args_list[0][0][0]
            assert set_vol_call == ["wpctl", "set-volume", "-l", "1.5", "61", "0.50"]
            set_mute_call = mock_run.call_args_list[1][0][0]
            assert set_mute_call == ["wpctl", "set-mute", "61", "0"]

    async def test_set_muted_invokes_wpctl(self) -> None:
        backend = PipeWireAudioBackend()
        with patch.object(backend, "_run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, "", "")
            assert await backend.set_muted("61", True) is True
            mock_run.assert_called_with(["wpctl", "set-mute", "61", "1"])

            assert await backend.set_muted("61", False) is True
            mock_run.assert_called_with(["wpctl", "set-mute", "61", "0"])

    async def test_set_default_invokes_wpctl(self) -> None:
        backend = PipeWireAudioBackend()
        with patch.object(backend, "_run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, "", "")
            assert await backend.set_default("61") is True
            mock_run.assert_called_with(["wpctl", "set-default", "61"])


class TestPulseAudioBackend:
    """Tests for PulseAudioBackend pactl JSON and text output parsing."""

    async def test_is_available_when_pactl_info_succeeds(self) -> None:
        backend = PulseAudioBackend()
        with (
            patch.object(backend, "is_supported", return_value=True),
            patch.object(backend, "_run_command", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = (
                0,
                "Server String: ...\nDefault Sink: alsa.speaker\n",
                "",
            )
            assert await backend.is_available() is True

            mock_run.return_value = (1, "", "Connection refused")
            assert await backend.is_available() is False

    async def test_list_sinks_json_parsing(self) -> None:
        backend = PulseAudioBackend()

        async def fake_run(cmd: list[str]):
            if cmd == ["pactl", "info"]:
                return 0, "Default Sink: alsa_output.pci-0000_00_1f.3.speaker\n", ""
            if cmd == ["pactl", "--format=json", "list", "sinks"]:
                return 0, json.dumps(SAMPLE_PACTL_JSON_SINKS), ""
            return 1, "", ""

        with patch.object(backend, "_run_command", side_effect=fake_run):
            sinks = await backend.list_sinks()
            assert len(sinks) == 2

            speaker = next(
                s for s in sinks if s.name == "alsa_output.pci-0000_00_1f.3.speaker"
            )
            assert speaker.volume == 35389
            assert speaker.muted is True
            assert speaker.enabled is True

            hdmi = next(
                s for s in sinks if s.name == "alsa_output.pci-0000_00_1f.3.hdmi"
            )
            assert hdmi.volume == 65536
            assert hdmi.muted is False
            assert hdmi.enabled is False

    async def test_list_sinks_text_fallback_parsing(self) -> None:
        backend = PulseAudioBackend()

        async def fake_run(cmd: list[str]):
            if cmd == ["pactl", "info"]:
                return 0, "Default Sink: alsa_output.pci-0000_00_1f.3.speaker\n", ""
            if cmd == ["pactl", "--format=json", "list", "sinks"]:
                return 1, "", "unrecognized option --format=json"
            if cmd == ["pactl", "list", "sinks"]:
                return 0, SAMPLE_PACTL_TEXT_SINKS, ""
            return 1, "", ""

        with patch.object(backend, "_run_command", side_effect=fake_run):
            sinks = await backend.list_sinks()
            assert len(sinks) == 2

            speaker = next(s for s in sinks if "speaker" in s.name)
            assert speaker.description == "Internal Speaker"
            assert speaker.muted is True
            assert speaker.enabled is True
            assert speaker.volume == round(0.54 * 65536)

    async def test_set_volume_and_unmute_in_pulse(self) -> None:
        backend = PulseAudioBackend()
        with patch.object(backend, "_run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, "", "")
            assert await backend.set_volume("sink1", 40000) is True

            assert mock_run.call_count == 2
            mock_run.assert_any_call(["pactl", "set-sink-volume", "sink1", "40000"])
            mock_run.assert_any_call(["pactl", "set-sink-mute", "sink1", "0"])

    async def test_set_muted_and_default_in_pulse(self) -> None:
        backend = PulseAudioBackend()
        with patch.object(backend, "_run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, "", "")
            assert await backend.set_muted("sink1", True) is True
            mock_run.assert_called_with(["pactl", "set-sink-mute", "sink1", "1"])

            assert await backend.set_default("sink1") is True
            mock_run.assert_called_with(["pactl", "set-default-sink", "sink1"])


class TestDetectAudioBackend:
    """Tests for backend selection and auto-detection."""

    def test_explicit_argument_selection(self) -> None:
        assert isinstance(detect_audio_backend("mock"), MockAudioSinkBackend)
        assert isinstance(detect_audio_backend("silent"), SilentFallbackSinkBackend)
        assert isinstance(detect_audio_backend("pipewire"), PipeWireAudioBackend)
        assert isinstance(detect_audio_backend("pulse"), PulseAudioBackend)

    def test_environment_variable_override(self) -> None:
        with patch.dict(os.environ, {"PENGULINK_AUDIO_BACKEND": "mock"}):
            assert isinstance(detect_audio_backend(), MockAudioSinkBackend)

        with patch.dict(os.environ, {"PENGULINK_AUDIO_BACKEND": "silent"}):
            assert isinstance(detect_audio_backend(), SilentFallbackSinkBackend)

    def test_unknown_backend_raises_error(self) -> None:
        with pytest.raises(AudioBackendError):
            detect_audio_backend("invalid_backend_xyz")

    def test_alias_detect_audio_sink_backend(self) -> None:
        assert isinstance(detect_audio_sink_backend("mock"), MockAudioSinkBackend)

    def test_auto_detection_cascade(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # If wpctl supported -> PipeWire
            with patch.object(PipeWireAudioBackend, "is_supported", return_value=True):
                assert isinstance(detect_audio_backend(), PipeWireAudioBackend)

            # If only pactl supported -> PulseAudio
            with (
                patch.object(PipeWireAudioBackend, "is_supported", return_value=False),
                patch.object(PulseAudioBackend, "is_supported", return_value=True),
            ):
                assert isinstance(detect_audio_backend(), PulseAudioBackend)

            # Neither -> SilentFallback
            with (
                patch.object(PipeWireAudioBackend, "is_supported", return_value=False),
                patch.object(PulseAudioBackend, "is_supported", return_value=False),
            ):
                assert isinstance(detect_audio_backend(), SilentFallbackSinkBackend)


class TestLocalAudioService:
    """Tests for LocalAudioService locking and EventBus publication."""

    async def test_service_crud_and_percent_volume(self) -> None:
        backend = MockAudioSinkBackend()
        service = LocalAudioService(backend=backend)

        sinks = await service.list_sinks()
        assert len(sinks) == 2
        first_name = sinks[0].name

        sink = await service.get_sink(first_name)
        assert sink is not None
        assert sink.name == first_name

        # Set volume as percentage (50% = 32768)
        assert await service.set_volume(first_name, 50, is_percent=True) is True
        updated = await service.get_sink(first_name)
        assert updated is not None
        assert updated.volume == 32768
        assert updated.volume_percent == 50

        # Mute
        assert await service.set_muted(first_name, True) is True
        assert (await service.get_sink(first_name)).muted is True

    async def test_service_event_bus_publishing(self) -> None:
        bus = EventBus()
        backend = MockAudioSinkBackend()
        service = LocalAudioService(backend=backend, event_bus=bus)

        updated_events: list[AudioSinkUpdatedEvent] = []
        list_events: list[AudioSinkListEvent] = []

        bus.subscribe(AudioSinkUpdatedEvent, updated_events.append)
        bus.subscribe(AudioSinkListEvent, list_events.append)

        speaker = "alsa_output.pci-0000_00_1f.3.speaker"
        hdmi = "alsa_output.pci-0000_00_1f.3.hdmi"

        # 1. Volume update triggers AudioSinkUpdatedEvent
        await service.set_volume(speaker, 40000)
        assert len(updated_events) == 1
        assert updated_events[0].sink is not None
        assert updated_events[0].sink.volume == 40000

        # 2. Mute update triggers AudioSinkUpdatedEvent
        await service.set_muted(speaker, True)
        assert len(updated_events) == 2
        assert updated_events[1].sink.muted is True

        # 3. Default sink switch triggers AudioSinkListEvent
        await service.set_default(hdmi)
        assert len(list_events) == 1
        assert len(list_events[0].sinks) == 2

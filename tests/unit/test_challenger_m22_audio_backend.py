"""tests.unit.test_challenger_m22_audio_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical challenger stress-test suite for PenguLink local audio subsystem:
- Boundary volume numbers (0, 65536, negative, overflows, floats, non-numeric strings)
- External command failure handling (non-zero exits, timeouts, OS errors)
- Unexpected and malformed command outputs (garbled wpctl status, corrupted pactl JSON/text)
- LocalAudioService concurrency serialization, lock mutual exclusion, and deadlock resistance
- Adversarial edge cases: pw-dump objects with null props, non-numeric maxVolume
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from pengulink.core.events import (
    AudioSinkUpdatedEvent,
    EventBus,
)
from pengulink.services.audio import (
    MAX_AUDIO_VOLUME,
    AudioSink,
    AudioSinkBackend,
    LocalAudioService,
    MockAudioSinkBackend,
    PipeWireAudioBackend,
    PulseAudioBackend,
    SilentFallbackSinkBackend,
)


class ConcurrencyInstrumentedBackend(AudioSinkBackend):
    """Test backend instrumented to detect concurrent execution inside operations."""

    def __init__(self, initial_sink: AudioSink) -> None:
        self.sink = AudioSink(
            name=initial_sink.name,
            description=initial_sink.description,
            volume=initial_sink.volume,
            max_volume=initial_sink.max_volume,
            muted=initial_sink.muted,
            enabled=initial_sink.enabled,
        )
        self.active_calls: int = 0
        self.max_observed_concurrency: int = 0
        self.total_calls: int = 0
        self._delay: float = 0.005  # 5ms delay to expose race conditions

    async def _enter(self) -> None:
        self.active_calls += 1
        self.total_calls += 1
        self.max_observed_concurrency = max(
            self.max_observed_concurrency, self.active_calls
        )
        if self._delay > 0:
            await asyncio.sleep(self._delay)

    async def _exit(self) -> None:
        self.active_calls -= 1

    async def is_available(self) -> bool:
        return True

    async def list_sinks(self) -> list[AudioSink]:
        await self._enter()
        try:
            return [
                AudioSink(
                    name=self.sink.name,
                    description=self.sink.description,
                    volume=self.sink.volume,
                    max_volume=self.sink.max_volume,
                    muted=self.sink.muted,
                    enabled=self.sink.enabled,
                )
            ]
        finally:
            await self._exit()

    async def set_volume(self, name: str, volume: int) -> bool:
        await self._enter()
        try:
            if name != self.sink.name:
                return False
            self.sink.volume = max(0, min(volume, self.sink.max_volume))
            self.sink.muted = False
            return True
        finally:
            await self._exit()

    async def set_muted(self, name: str, muted: bool) -> bool:
        await self._enter()
        try:
            if name != self.sink.name:
                return False
            self.sink.muted = bool(muted)
            return True
        finally:
            await self._exit()

    async def set_default(self, name: str) -> bool:
        await self._enter()
        try:
            if name != self.sink.name:
                return False
            self.sink.enabled = True
            return True
        finally:
            await self._exit()


# ============================================================================
# 1. Boundary Volume Numbers & Input Coercion
# ============================================================================


class TestAudioSinkBoundaries:
    """Stress tests for AudioSink edge-case numerical and type boundaries."""

    @pytest.mark.parametrize(
        ("raw_volume", "expected_volume", "expected_percent"),
        [
            (0, 0, 0),
            (1, 1, 0),
            (655, 655, 1),
            (32768, 32768, 50),
            (65535, 65535, 100),
            (65536, 65536, 100),
            # Negative boundaries -> clamped to 0
            (-1, 0, 0),
            (-65536, 0, 0),
            (-999999999, 0, 0),
            # Overflow boundaries -> clamped to max_volume (65536)
            (65537, 65536, 100),
            (100000, 65536, 100),
            (2**31 - 1, 65536, 100),
            (10**12, 65536, 100),
        ],
    )
    def test_audio_sink_volume_clamping_boundaries(
        self, raw_volume: int, expected_volume: int, expected_percent: int
    ) -> None:
        sink = AudioSink(name="boundary_sink", volume=raw_volume)
        assert sink.volume == expected_volume
        assert sink.volume_percent == expected_percent

    @pytest.mark.parametrize("invalid_max", [0, -1, -65536, -99999])
    def test_audio_sink_non_positive_max_volume_fallback(
        self, invalid_max: int
    ) -> None:
        sink = AudioSink(name="fallback_sink", volume=32768, max_volume=invalid_max)
        assert sink.max_volume == MAX_AUDIO_VOLUME
        assert sink.volume == 32768
        assert sink.volume_percent == 50

    @pytest.mark.parametrize(
        ("input_dict", "expected_vol", "expected_muted", "expected_desc"),
        [
            # String volume conversions
            ({"name": "s1", "volume": "0"}, 0, False, "s1"),
            ({"name": "s1", "volume": "65536"}, 65536, False, "s1"),
            ({"name": "s1", "volume": "32768.4"}, 32768, False, "s1"),
            ({"name": "s1", "volume": "32768.9"}, 32768, False, "s1"),
            ({"name": "s1", "volume": "-500"}, 0, False, "s1"),
            ({"name": "s1", "volume": "999999"}, 65536, False, "s1"),
            # Corrupted string volumes fallback to 0
            ({"name": "s1", "volume": "not_a_number"}, 0, False, "s1"),
            ({"name": "s1", "volume": ""}, 0, False, "s1"),
            ({"name": "s1", "volume": None}, 0, False, "s1"),
            ({"name": "s1", "volume": []}, 0, False, "s1"),
            ({"name": "s1", "volume": {}}, 0, False, "s1"),
            # Float volume representations
            ({"name": "s1", "volume": 0.0}, 0, False, "s1"),
            ({"name": "s1", "volume": 65536.0}, 65536, False, "s1"),
            ({"name": "s1", "volume": -100.5}, 0, False, "s1"),
            # Description aliases: description > desc > nick > name > fallback
            ({"name": "s1", "desc": "Desc Alias"}, 0, False, "Desc Alias"),
            ({"name": "s1", "nick": "Nick Alias"}, 0, False, "Nick Alias"),
            ({"name": "", "description": ""}, 0, False, "Audio Output"),
            ({"name": "   trimmed_sink   "}, 0, False, "trimmed_sink"),
        ],
    )
    def test_from_dict_boundary_coercion(
        self,
        input_dict: dict[str, Any],
        expected_vol: int,
        expected_muted: bool,
        expected_desc: str,
    ) -> None:
        sink = AudioSink.from_dict(input_dict)
        assert sink.volume == expected_vol
        assert sink.muted is expected_muted
        assert sink.description == expected_desc

    def test_wire_and_dict_serialization_fidelity(self) -> None:
        sink = AudioSink(
            name="test_sink",
            description="Test Description",
            volume=65536,
            max_volume=65536,
            muted=True,
            enabled=True,
        )
        wire = sink.to_wire_dict()
        assert wire["maxVolume"] == 65536
        assert wire["volume"] == 65536
        assert wire["muted"] is True
        assert wire["enabled"] is True

        d = sink.to_dict()
        assert d["maxVolume"] == 65536
        assert d["max_volume"] == 65536
        assert d["volume_percent"] == 100

    def test_from_dict_non_numeric_max_volume_adversarial(self) -> None:
        """Adversarial challenge: AudioSink.from_dict must safely coerce non-numeric maxVolume.

        When remote peers send malformed string maxVolume (e.g. 'invalid'), from_dict
        should safely fall back to MAX_AUDIO_VOLUME without raising unhandled ValueError.
        """
        sink = AudioSink.from_dict(
            {"name": "sink_malformed_max", "maxVolume": "invalid"}
        )
        assert sink.max_volume == MAX_AUDIO_VOLUME


class TestLocalAudioServiceBoundaries:
    """Boundary tests for LocalAudioService set_volume and input validation."""

    async def test_set_volume_raw_boundaries(self) -> None:
        backend = MockAudioSinkBackend()
        service = LocalAudioService(backend=backend)
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"

        # Exact min boundary
        assert await service.set_volume(speaker, 0) is True
        sink = await service.get_sink(speaker)
        assert sink is not None
        assert sink.volume == 0
        assert sink.volume_percent == 0

        # Exact max boundary
        assert await service.set_volume(speaker, 65536) is True
        sink = await service.get_sink(speaker)
        assert sink is not None
        assert sink.volume == 65536
        assert sink.volume_percent == 100

        # Negative volume clamped to 0
        assert await service.set_volume(speaker, -5000) is True
        sink = await service.get_sink(speaker)
        assert sink is not None
        assert sink.volume == 0

        # Overflow volume clamped to 65536
        assert await service.set_volume(speaker, 999999) is True
        sink = await service.get_sink(speaker)
        assert sink is not None
        assert sink.volume == 65536

    async def test_set_volume_percent_boundaries(self) -> None:
        backend = MockAudioSinkBackend()
        service = LocalAudioService(backend=backend)
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"

        # 0%
        assert await service.set_volume(speaker, 0, is_percent=True) is True
        sink = await service.get_sink(speaker)
        assert sink is not None
        assert sink.volume == 0

        # 100%
        assert await service.set_volume(speaker, 100, is_percent=True) is True
        sink = await service.get_sink(speaker)
        assert sink is not None
        assert sink.volume == 65536

        # Negative percent clamped to 0%
        assert await service.set_volume(speaker, -25, is_percent=True) is True
        sink = await service.get_sink(speaker)
        assert sink is not None
        assert sink.volume == 0

        # Over 100% clamped to 100%
        assert await service.set_volume(speaker, 150, is_percent=True) is True
        sink = await service.get_sink(speaker)
        assert sink is not None
        assert sink.volume == 65536

        # 50%
        assert await service.set_volume(speaker, 50, is_percent=True) is True
        sink = await service.get_sink(speaker)
        assert sink is not None
        assert sink.volume == 32768

    async def test_operations_on_nonexistent_and_empty_sink(self) -> None:
        backend = MockAudioSinkBackend()
        service = LocalAudioService(backend=backend)

        assert await service.get_sink("nonexistent") is None
        assert await service.get_sink("") is None

        assert await service.set_volume("nonexistent", 30000) is False
        assert await service.set_volume("", 30000) is False

        assert await service.set_muted("nonexistent", True) is False
        assert await service.set_muted("", True) is False

        assert await service.set_default("nonexistent") is False
        assert await service.set_default("") is False


# ============================================================================
# 2. External Command Handling & Failure Resilience
# ============================================================================


class TestPipeWireBackendCommandFailures:
    """Stress tests for PipeWireAudioBackend external command failures and malformed outputs."""

    async def test_wpctl_nonzero_exit_status(self) -> None:
        backend = PipeWireAudioBackend()
        with patch.object(backend, "_run_command", new_callable=AsyncMock) as mock_run:
            # Status fails with error exit code 1
            mock_run.return_value = (1, "", "Error: WirePlumber service unreachable")
            assert await backend.is_available() is False
            assert await backend.list_sinks() == []

            # Set-volume fails with error exit code 2
            mock_run.return_value = (2, "", "Node 42 not found")
            assert await backend.set_volume("42", 32768) is False

            # Set-mute fails
            mock_run.return_value = (1, "", "Permission denied")
            assert await backend.set_muted("42", True) is False

            # Set-default fails
            mock_run.return_value = (127, "", "command failed")
            assert await backend.set_default("42") is False

    async def test_wpctl_subprocess_timeout(self) -> None:
        backend = PipeWireAudioBackend(command_timeout=0.01)

        async def timeout_exec(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0.05)
            raise asyncio.TimeoutError("Execution timed out")

        with patch("asyncio.create_subprocess_exec", side_effect=timeout_exec):
            code, _stdout, stderr = await backend._run_command(["wpctl", "status"])
            assert code != 0
            assert "Execution timed out" in stderr

            # High-level methods return failure safely without raising unhandled exceptions
            assert await backend.is_available() is False
            assert await backend.list_sinks() == []
            assert await backend.set_volume("42", 30000) is False

    async def test_wpctl_subprocess_os_exceptions(self) -> None:
        backend = PipeWireAudioBackend()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("No such file or directory: 'wpctl'"),
        ):
            code, _stdout, stderr = await backend._run_command(["wpctl", "status"])
            assert code != 0
            assert "No such file" in stderr
            assert await backend.is_available() is False

    async def test_wpctl_status_garbled_and_edge_case_formats(self) -> None:
        backend = PipeWireAudioBackend()

        # 1. Output without Audio section
        no_audio_output = "Clients:\n  33. WirePlumber\nVideo\n  Devices:\n"
        with patch.object(backend, "_run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, no_audio_output, "")
            assert await backend.list_sinks() == []

        # 2. Output with Audio but no Sinks section (only Sources)
        no_sinks_output = "Audio\n  Sources:\n  * 62. Headset Mic [vol: 1.00]\n"
        with patch.object(backend, "_run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, no_sinks_output, "")
            assert await backend.list_sinks() == []

        # 3. Sinks section with malformed lines, overamplification, and invalid volumes
        corrupted_status = """Audio
 ├─ Devices:
 │      48. Audio Controller [alsa]
 │  
 ├─ Sinks:
 │      50. Malformed Line Without Volume Brackets
 │      51. Corrupted Float Sink [vol: NaN]
 │      52. String Volume Sink [vol: invalid]
 │      53. Negative Float Sink [vol: -0.25]
 │      54. Overamplified Float Sink [vol: 1.85]
 │  *   55. Normal Overamplified Default [vol: 1.20 MUTED]
 │  
 ├─ Sources:
 │      62. Headset Mic [vol: 1.00]
"""
        with patch.object(backend, "_run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, corrupted_status, "")
            sinks = await backend.list_sinks()
            names = [s.name for s in sinks]

            # Malformed or non-numeric lines are safely rejected by regex pattern
            assert "50" not in names
            assert "51" not in names
            assert "52" not in names
            assert "53" not in names

            # Overamplified volume (1.85) clamped to MAX_AUDIO_VOLUME (65536)
            s54 = next(s for s in sinks if s.name == "54")
            assert s54.volume == MAX_AUDIO_VOLUME
            assert s54.volume_percent == 100

            # Default sink (55) with 1.20 clamped to 65536, muted, enabled
            s55 = next(s for s in sinks if s.name == "55")
            assert s55.volume == MAX_AUDIO_VOLUME
            assert s55.muted is True
            assert s55.enabled is True

    async def test_pw_dump_corrupted_json_handling(self) -> None:
        backend = PipeWireAudioBackend()

        async def run_corrupted(cmd: list[str]) -> tuple[int, str, str]:
            if cmd[0] == "wpctl":
                return 0, "Audio\n ├─ Sinks:\n │  * 61. Speaker [vol: 0.50]\n", ""
            if cmd[0] == "pw-dump":
                # Return corrupted JSON syntax
                return 0, "{ corrupted json syntax ...", ""
            return 1, "", ""

        with (
            patch("shutil.which", return_value="/usr/bin/pw-dump"),
            patch.object(backend, "_run_command", side_effect=run_corrupted),
        ):
            # Must not raise JSONDecodeError; falls back to node ID
            sinks = await backend.list_sinks()
            assert len(sinks) == 1
            assert sinks[0].name == "61"

    async def test_pw_dump_with_null_props_adversarial(self) -> None:
        """Adversarial challenge: pw-dump outputs objects with props: null or non-dict props.

        In a live PipeWire system, pw-dump dumps graph objects where info.props can be null.
        PipeWireAudioBackend._refresh_node_names must handle this without raising AttributeError.
        """
        backend = PipeWireAudioBackend()

        async def run_with_null_props(cmd: list[str]) -> tuple[int, str, str]:
            if cmd[0] == "wpctl":
                return 0, "Audio\n ├─ Sinks:\n │  * 61. Speaker [vol: 0.50]\n", ""
            if cmd[0] == "pw-dump":
                return (
                    0,
                    json.dumps(
                        [
                            12345,  # Non-dict item
                            "string_item",
                            {"info": None},  # info is None
                            {
                                "id": 60,
                                "info": {
                                    "props": None
                                },  # props is None (CRITICAL EDGE CASE)
                            },
                            {
                                "id": 61,
                                "info": {
                                    "props": {
                                        "media.class": "Audio/Sink",
                                        "node.name": "alsa.valid.speaker",
                                    }
                                },
                            },
                        ]
                    ),
                    "",
                )
            return 1, "", ""

        with (
            patch("shutil.which", return_value="/usr/bin/pw-dump"),
            patch.object(backend, "_run_command", side_effect=run_with_null_props),
        ):
            sinks = await backend.list_sinks()
            assert len(sinks) == 1
            assert sinks[0].name == "alsa.valid.speaker"

    def test_resolve_target_id_special_names(self) -> None:
        backend = PipeWireAudioBackend()
        assert backend._resolve_target_id("@DEFAULT@") == "@DEFAULT_AUDIO_SINK@"
        assert backend._resolve_target_id("@DEFAULT_SINK@") == "@DEFAULT_AUDIO_SINK@"
        assert (
            backend._resolve_target_id("@DEFAULT_AUDIO_SINK@") == "@DEFAULT_AUDIO_SINK@"
        )
        assert backend._resolve_target_id("  @DEFAULT@  ") == "@DEFAULT_AUDIO_SINK@"
        assert backend._resolve_target_id("61") == "61"


class TestPulseAudioBackendCommandFailures:
    """Stress tests for PulseAudioBackend external command failures and malformed outputs."""

    async def test_pactl_nonzero_exit_status(self) -> None:
        backend = PulseAudioBackend()
        with patch.object(backend, "_run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (1, "", "Connection refused")
            assert await backend.is_available() is False

            mock_run.return_value = (1, "", "No such sink")
            assert await backend.set_volume("sink1", 30000) is False
            assert await backend.set_muted("sink1", True) is False
            assert await backend.set_default("sink1") is False

    async def test_pactl_subprocess_timeout_and_error(self) -> None:
        backend = PulseAudioBackend(command_timeout=0.01)

        async def timeout_exec(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0.05)
            raise asyncio.TimeoutError("pactl command timed out")

        with patch("asyncio.create_subprocess_exec", side_effect=timeout_exec):
            code, _stdout, stderr = await backend._run_command(["pactl", "info"])
            assert code != 0
            assert "pactl command timed out" in stderr
            assert await backend.is_available() is False

    async def test_pactl_json_malformed_payloads_fallback_to_text(self) -> None:
        backend = PulseAudioBackend()

        # pactl --format=json returns invalid JSON; pactl list sinks text succeeds
        fallback_text = """Sink #1
\tName: alsa_output.pci.speaker
\tDescription: Fallback Speaker
\tMute: no
\tVolume: front-left: 32768 / 50% / -6.00 dB
"""

        async def fake_run(cmd: list[str]) -> tuple[int, str, str]:
            if cmd == ["pactl", "info"]:
                return 0, "Default Sink: alsa_output.pci.speaker\n", ""
            if cmd == ["pactl", "--format=json", "list", "sinks"]:
                return 0, "{ corrupted json without list ...", ""
            if cmd == ["pactl", "list", "sinks"]:
                return 0, fallback_text, ""
            return 1, "", ""

        with patch.object(backend, "_run_command", side_effect=fake_run):
            sinks = await backend.list_sinks()
            assert len(sinks) == 1
            assert sinks[0].name == "alsa_output.pci.speaker"
            assert sinks[0].volume == 32768
            assert sinks[0].enabled is True

    async def test_pactl_json_items_with_missing_and_corrupted_fields(
        self,
    ) -> None:
        backend = PulseAudioBackend()

        corrupted_json_sinks = [
            # 1. Non-dict element
            "invalid_string_element",
            # 2. Missing volume dictionary
            {
                "name": "sink_no_vol",
                "description": "No Volume",
                "mute": False,
            },
            # 3. Volume dictionary with non-dict channel or missing value
            {
                "name": "sink_bad_vol",
                "description": "Bad Volume Dict",
                "volume": {
                    "front-left": "not_a_dict",
                    "front-right": {"missing_value": True},
                },
            },
            # 4. Volume value exceeding MAX_AUDIO_VOLUME
            {
                "name": "sink_overflow_vol",
                "volume": {
                    "front-left": {"value": 150000},
                },
            },
        ]

        async def fake_run(cmd: list[str]) -> tuple[int, str, str]:
            if cmd == ["pactl", "info"]:
                return 0, "", ""
            if cmd == ["pactl", "--format=json", "list", "sinks"]:
                return 0, json.dumps(corrupted_json_sinks), ""
            return 1, "", ""

        with patch.object(backend, "_run_command", side_effect=fake_run):
            sinks = await backend.list_sinks()
            assert len(sinks) == 3

            s_no_vol = next(s for s in sinks if s.name == "sink_no_vol")
            assert s_no_vol.volume == 0

            s_bad_vol = next(s for s in sinks if s.name == "sink_bad_vol")
            assert s_bad_vol.volume == 0

            s_over_vol = next(s for s in sinks if s.name == "sink_overflow_vol")
            assert s_over_vol.volume == MAX_AUDIO_VOLUME

    async def test_pactl_text_parsing_corrupted_and_empty(self) -> None:
        backend = PulseAudioBackend()

        # Completely empty text
        assert backend._parse_pactl_text_sinks("", "default") == []

        # Text with no recognized keys
        garbage_text = "Some random text from an unsupported sound daemon\nLine 2"
        assert backend._parse_pactl_text_sinks(garbage_text, "default") == []

        # Sink with missing volume line
        no_vol_text = """Sink #1
\tName: sink_without_volume
\tDescription: No Vol Sink
\tMute: yes
"""
        sinks = backend._parse_pactl_text_sinks(no_vol_text, "")
        assert len(sinks) == 1
        assert sinks[0].name == "sink_without_volume"
        assert sinks[0].volume == 0
        assert sinks[0].muted is True


class TestSilentFallbackSinkBackendEdgeCases:
    """Tests for SilentFallbackSinkBackend safety under all inputs."""

    async def test_silent_fallback_never_raises(self) -> None:
        backend = SilentFallbackSinkBackend()
        assert await backend.is_available() is True
        assert await backend.list_sinks() == []
        assert await backend.set_volume("", -100) is False
        assert await backend.set_volume("any", 999999) is False
        assert await backend.set_muted("", True) is False
        assert await backend.set_default("") is False


# ============================================================================
# 3. Concurrency Serialization & Deadlock Resistance
# ============================================================================


class TestLocalAudioServiceConcurrency:
    """Adversarial concurrency and race verification for LocalAudioService."""

    async def test_concurrency_serialization_mutual_exclusion(self) -> None:
        """Verify that LocalAudioService strictly serializes concurrent calls.

        The ConcurrencyInstrumentedBackend records the maximum number of concurrent
        active calls. With LocalAudioService's lock, max_observed_concurrency MUST be 1.
        """
        init_sink = AudioSink(name="sink1", volume=10000, max_volume=65536)
        backend = ConcurrencyInstrumentedBackend(init_sink)
        service = LocalAudioService(backend=backend)

        # Launch 40 concurrent volume changes with interleaved values
        tasks = [service.set_volume("sink1", 1000 * i) for i in range(1, 41)]
        results = await asyncio.gather(*tasks)

        assert all(results)
        # Invariant: Locking must ensure no more than 1 operation executes in backend at once
        assert backend.max_observed_concurrency == 1
        assert backend.total_calls == 40
        assert backend.active_calls == 0

    async def test_concurrent_volume_and_mute_mixed_burst(self) -> None:
        """Verify rapid interleaved volume, mute, default, and get_sink queries."""
        init_sink = AudioSink(name="sink1", volume=10000, max_volume=65536)
        backend = ConcurrencyInstrumentedBackend(init_sink)
        service = LocalAudioService(backend=backend)

        async def worker(op_idx: int) -> bool:
            mod = op_idx % 4
            if mod == 0:
                return await service.set_volume("sink1", (op_idx * 1500) % 65536)
            elif mod == 1:
                return await service.set_muted("sink1", op_idx % 2 == 0)
            elif mod == 2:
                sink = await service.get_sink("sink1")
                return sink is not None
            else:
                sinks = await service.list_sinks()
                return len(sinks) == 1

        # Fire 80 concurrent mixed operations
        tasks = [worker(i) for i in range(80)]
        results = await asyncio.gather(*tasks)

        assert all(results)
        # Lock mutual exclusion guaranteed throughout
        assert backend.max_observed_concurrency == 1
        assert backend.active_calls == 0

    async def test_event_bus_publishing_deadlock_freedom(self) -> None:
        """Verify that event subscribers can query or mutate the service without deadlocking.

        Because LocalAudioService publishes events outside of `self._lock`, an event subscriber
        re-entering the service (e.g. calling `get_sink` or `set_volume` in response to an
        `AudioSinkUpdatedEvent`) must not deadlock.
        """
        bus = EventBus()
        backend = MockAudioSinkBackend()
        service = LocalAudioService(backend=backend, event_bus=bus)
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"

        reentrant_query_success = False
        reentrant_sink_volume: int | None = None

        async def on_sink_updated(event: AudioSinkUpdatedEvent) -> None:
            nonlocal reentrant_query_success, reentrant_sink_volume
            # Subscriber immediately calls service.get_sink while processing event
            sink = await service.get_sink(speaker)
            if sink is not None:
                reentrant_query_success = True
                reentrant_sink_volume = sink.volume

        bus.subscribe(AudioSinkUpdatedEvent, on_sink_updated)

        # Trigger volume change; this will publish AudioSinkUpdatedEvent
        # Wrap in wait_for to fail quickly if a deadlock occurs
        success = await asyncio.wait_for(
            service.set_volume(speaker, 48000),
            timeout=2.0,
        )

        assert success is True
        assert reentrant_query_success is True
        assert reentrant_sink_volume == 48000

    async def test_high_volume_rapid_slider_drag_simulation(self) -> None:
        """Simulate high-frequency UI slider drag events (150 rapid updates)."""
        backend = MockAudioSinkBackend()
        service = LocalAudioService(backend=backend)
        speaker = "alsa_output.pci-0000_00_1f.3.speaker"

        # Rapidly fire 150 volume set calls representing slider movements
        tasks = [
            service.set_volume(speaker, int((i / 150.0) * MAX_AUDIO_VOLUME))
            for i in range(150)
        ]
        results = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=5.0,
        )

        assert all(results)
        final_sink = await service.get_sink(speaker)
        assert final_sink is not None
        assert 0 <= final_sink.volume <= MAX_AUDIO_VOLUME
        # Volume set automatically unmutes
        assert final_sink.muted is False

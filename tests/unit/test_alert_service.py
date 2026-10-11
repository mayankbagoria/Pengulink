"""tests.unit.test_alert_service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for AlertService, AudioBackend hierarchy, synthetic WAV chime generator,
process escalation, and runaway timeout protection.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal
import time
from typing import Any, List
import wave
import pytest

from pengulink.core.events import (
    AlertStopped,
    AlertTriggered,
    EventBus,
    FindDeviceAlertStoppedEvent,
    FindDeviceAlertTriggeredEvent,
)
from pengulink.core.exceptions import AudioBackendError
from pengulink.services.alert import (
    DEFAULT_MAX_ALERT_DURATION,
    INTER_CHIME_DELAY,
    AlertService,
    AlsaAudioBackend,
    AudioBackend,
    MockAudioBackend,
    MockAudioProcess,
    PipeWireAudioBackend,
    PulseAudioAudioBackend,
    SilentAudioProcess,
    SilentFallbackBackend,
    detect_audio_backend,
    ensure_chime_file,
    generate_alert_chime,
)


def test_synthetic_wav_chime_generation(tmp_path: Path) -> None:
    """Validate generated PCM WAV file structure and acoustics."""
    target_path = tmp_path / "test_chime.wav"
    result = generate_alert_chime(target_path, sample_rate=44100, duration=1.2)
    assert result == target_path
    assert target_path.exists()
    assert target_path.stat().st_size > 1000

    with wave.open(str(target_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 44100
        nframes = wf.getnframes()
        assert nframes == int(44100 * 1.2)
        frames = wf.readframes(nframes)
        assert len(frames) == nframes * 2
        # Verify non-silent samples
        assert any(b != 0 for b in frames)


def test_ensure_chime_file_caching(tmp_path: Path) -> None:
    """Calling ensure_chime_file twice reuses the cached file without re-generating."""
    path1 = ensure_chime_file(tmp_path)
    mtime1 = path1.stat().st_mtime_ns
    assert path1.exists()

    path2 = ensure_chime_file(tmp_path)
    mtime2 = path2.stat().st_mtime_ns
    assert path1 == path2
    assert mtime1 == mtime2


async def test_audio_backend_hierarchy() -> None:
    """Verify backend name and availability interfaces."""
    pw = PipeWireAudioBackend()
    assert pw.name == "pipewire"
    assert isinstance(await pw.is_available(), bool)

    pulse = PulseAudioAudioBackend()
    assert pulse.name == "pulseaudio"
    assert isinstance(await pulse.is_available(), bool)

    alsa = AlsaAudioBackend()
    assert alsa.name == "alsa"
    assert isinstance(await alsa.is_available(), bool)

    mock = MockAudioBackend()
    assert mock.name == "mock"
    assert await mock.is_available() is True

    silent = SilentFallbackBackend()
    assert silent.name == "silent"
    assert await silent.is_available() is True


def test_detect_audio_backend_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify PENGULINK_ALERT_BACKEND selects corresponding backend."""
    monkeypatch.setenv("PENGULINK_ALERT_BACKEND", "mock")
    backend = detect_audio_backend()
    assert isinstance(backend, MockAudioBackend)

    monkeypatch.setenv("PENGULINK_ALERT_BACKEND", "silent")
    backend = detect_audio_backend()
    assert isinstance(backend, SilentFallbackBackend)

    monkeypatch.setenv("PENGULINK_ALERT_BACKEND", "alsa")
    backend = detect_audio_backend()
    assert isinstance(backend, AlsaAudioBackend)

    monkeypatch.setenv("PENGULINK_ALERT_BACKEND", "invalid_backend_name_xyz")
    with pytest.raises(AudioBackendError):
        detect_audio_backend()


async def test_mock_audio_backend_play(tmp_path: Path) -> None:
    """MockAudioBackend records playback without spawning OS subprocess."""
    backend = MockAudioBackend(playback_duration=0.01)
    chime = ensure_chime_file(tmp_path)
    proc = await backend.play(chime)
    assert isinstance(proc, MockAudioProcess)
    assert backend.play_count == 1
    assert backend.played_files == [chime]
    rc = await proc.wait()
    assert rc == 0


async def test_silent_fallback_backend_play(tmp_path: Path) -> None:
    """SilentFallbackBackend executes gracefully."""
    backend = SilentFallbackBackend(simulated_duration=0.01)
    chime = ensure_chime_file(tmp_path)
    proc = await backend.play(chime)
    assert isinstance(proc, SilentAudioProcess)
    rc = await proc.wait()
    assert rc == 0


async def test_alert_service_start_and_status(tmp_path: Path) -> None:
    """Start alert and introspect status attributes."""
    backend = MockAudioBackend(playback_duration=0.05)
    bus = EventBus()
    service = AlertService(backend=backend, event_bus=bus, runtime_dir=tmp_path)

    assert service.is_active() is False
    status_idle = service.get_status()
    assert status_idle["active"] is False

    started = await service.start_alert(device_id="test_dev", source="remote", max_duration=10.0)
    assert started is True
    assert service.is_active() is True

    status_active = service.get_status()
    assert status_active["active"] is True
    assert status_active["device_id"] == "test_dev"
    assert status_active["source"] == "remote"
    assert status_active["backend"] == "mock"
    assert status_active["remaining"] > 0
    assert status_active["duration"] == 10.0

    await service.stop_alert(reason="manual")
    assert service.is_active() is False


async def test_alert_service_runaway_timeout(tmp_path: Path) -> None:
    """Alert automatically silences itself after duration expires."""
    backend = MockAudioBackend(playback_duration=0.01)
    bus = EventBus()
    stopped_events: List[Any] = []
    bus.subscribe(FindDeviceAlertStoppedEvent, lambda ev: stopped_events.append(ev))

    service = AlertService(backend=backend, event_bus=bus, runtime_dir=tmp_path)
    await service.start_alert(device_id="test_dev", source="test", max_duration=0.1)
    assert service.is_active() is True

    # Await runaway timeout
    await asyncio.sleep(0.2)
    assert service.is_active() is False
    assert len(stopped_events) == 1
    assert stopped_events[0].reason == "timeout"


async def test_alert_service_30s_hard_cap(tmp_path: Path) -> None:
    """Requests exceeding 30.0s are clamped to 30.0s."""
    backend = MockAudioBackend(playback_duration=0.01)
    service = AlertService(backend=backend, runtime_dir=tmp_path)
    await service.start_alert(device_id="dev1", max_duration=9999.0)
    status = service.get_status()
    assert status["duration"] == DEFAULT_MAX_ALERT_DURATION
    await service.stop_alert()


async def test_alert_service_manual_stop(tmp_path: Path) -> None:
    """stop_alert immediately halts playback loop and terminates process."""
    backend = MockAudioBackend(playback_duration=0.1)
    service = AlertService(backend=backend, runtime_dir=tmp_path)
    await service.start_alert(device_id="dev1", max_duration=10.0)
    assert service.is_active() is True

    stopped = await service.stop_alert(reason="manual")
    assert stopped is True
    assert service.is_active() is False

    # Calling stop again when already inactive returns False
    stopped_again = await service.stop_alert(reason="manual")
    assert stopped_again is False


async def test_alert_service_concurrency_mutex(tmp_path: Path) -> None:
    """Repeated start_alert calls refresh duration without spawning duplicate loops."""
    backend = MockAudioBackend(playback_duration=0.02)
    service = AlertService(backend=backend, runtime_dir=tmp_path)

    await service.start_alert(device_id="dev1", max_duration=5.0)
    loop_task_1 = service._loop_task

    await service.start_alert(device_id="dev1", max_duration=8.0)
    loop_task_2 = service._loop_task

    assert loop_task_1 is loop_task_2
    assert service.get_status()["duration"] == 8.0
    await service.stop_alert()


async def test_alert_service_eventbus_emission(tmp_path: Path) -> None:
    """Verify AlertTriggered and AlertStopped events are dispatched."""
    backend = MockAudioBackend(playback_duration=0.01)
    bus = EventBus()
    triggered: List[Any] = []
    stopped: List[Any] = []

    bus.subscribe(AlertTriggered, lambda ev: triggered.append(ev))
    bus.subscribe(AlertStopped, lambda ev: stopped.append(ev))

    service = AlertService(backend=backend, event_bus=bus, runtime_dir=tmp_path)
    await service.start_alert(device_id="phone_123", source="remote", max_duration=1.0)

    assert len(triggered) == 1
    assert triggered[0].device_id == "phone_123"
    assert triggered[0].source == "remote"

    await service.stop_alert(reason="manual")
    assert len(stopped) == 1
    assert stopped[0].device_id == "phone_123"
    assert stopped[0].reason == "manual"


async def test_alert_service_process_termination_escalation() -> None:
    """Subprocess termination escalates from terminate() to kill() if needed."""
    class StubbornProcess:
        def __init__(self) -> None:
            self.pid = 88888
            self.returncode = None
            self.terminated = False
            self.killed = False

        def terminate(self) -> None:
            self.terminated = True
            # Does not set returncode (simulating stuck process)

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            if self.killed:
                return -9
            await asyncio.sleep(2.0)
            return 0

    proc = StubbornProcess()
    service = AlertService()
    # Mock timeout to 0.05s for fast escalation
    orig_timeout = AlertService._terminate_process.__globals__["SUBPROCESS_TERMINATE_TIMEOUT"]
    try:
        AlertService._terminate_process.__globals__["SUBPROCESS_TERMINATE_TIMEOUT"] = 0.05
        await service._terminate_process(proc)
        assert proc.terminated is True
        assert proc.killed is True
    finally:
        AlertService._terminate_process.__globals__["SUBPROCESS_TERMINATE_TIMEOUT"] = orig_timeout


async def test_alert_service_daemon_shutdown(tmp_path: Path) -> None:
    """service.stop() cleanly shuts down active alerts."""
    backend = MockAudioBackend(playback_duration=0.02)
    service = AlertService(backend=backend, runtime_dir=tmp_path)
    await service.start_alert(device_id="dev_shutdown", max_duration=10.0)
    assert service.is_active() is True

    await service.stop()
    assert service.is_active() is False

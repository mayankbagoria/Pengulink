"""pengulink.services.alert
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Desktop alert service and audio playback subsystem for PenguLink.

Provides:
- AudioBackend hierarchy: PipeWire, PulseAudio, ALSA, Mock, and Silent Fallback
- Zero-dependency synthetic PCM WAV chime generator (stdlib wave, struct, math)
- AlertService: looping audio playback, 30s runaway protection, asyncio.Lock mutex,
  2-stage subprocess termination (SIGTERM -> SIGKILL), non-blocking notify-send,
  and complete status introspection.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import time
from typing import Any, Dict, List, Optional
import wave
from abc import ABC, abstractmethod

from pengulink.core.events import AlertStopped, AlertTriggered, EventBus
from pengulink.core.exceptions import (
    AlertError,
    AudioBackendError,
    AudioUnavailableError,
)

logger = logging.getLogger("pengulink.services.alert")

# Maximum runaway alert duration ceiling in seconds
DEFAULT_MAX_ALERT_DURATION: float = 30.0
# Inter-chime delay between playback iterations in seconds
INTER_CHIME_DELAY: float = 0.5
# Subprocess termination timeout in seconds
SUBPROCESS_TERMINATE_TIMEOUT: float = 1.0

# Regex pattern matching sensitive authentication tokens, keys, and secrets
SENSITIVE_PATTERN = re.compile(
    r"(-----BEGIN (?:RSA )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA )?PRIVATE KEY-----"
    r"|bearer\s+[a-zA-Z0-9_\-\.]+"
    r"|[a-zA-Z0-9_-]*(?:secret|token|password|auth|private)[a-zA-Z0-9_-]*)",
    re.IGNORECASE,
)


def _sanitize_log_str(val: Optional[str]) -> str:
    """Sanitize sensitive credentials, auth tokens, and secrets from log strings."""
    if not val:
        return ""
    return SENSITIVE_PATTERN.sub("[REDACTED]", str(val))


# =====================================================================
# Synthetic PCM WAV Chime Generator
# =====================================================================

def generate_alert_chime(
    output_path: Path,
    sample_rate: int = 44100,
    duration: float = 1.2,
) -> Path:
    """Generate a clean, pleasant 3-note ascending PCM WAV chime in Python stdlib.

    Zero external dependencies: uses wave, struct, math.

    Notes:
    - C5 (523.25 Hz): 0.0s - 0.4s
    - E5 (659.25 Hz): 0.2s - 0.6s
    - G5 (783.99 Hz): 0.4s - 1.2s
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    num_samples = int(sample_rate * duration)
    notes = [
        (0.0, 0.4, 523.25),   # C5
        (0.2, 0.6, 659.25),   # E5
        (0.4, 1.2, 783.99),   # G5
    ]

    samples = [0.0] * num_samples
    for start_t, end_t, freq in notes:
        start_idx = int(start_t * sample_rate)
        end_idx = min(int(end_t * sample_rate), num_samples)
        note_dur = end_t - start_t
        for i in range(start_idx, end_idx):
            t = (i - start_idx) / sample_rate
            # Exponential decay envelope: smooth attack & natural decay
            envelope = math.exp(-3.5 * (t / note_dur))
            samples[i] += 0.4 * math.sin(2.0 * math.pi * freq * t) * envelope

    # Write to temporary file first for atomic replacement
    temp_path = output_path.with_suffix(".tmp.wav")
    try:
        with wave.open(str(temp_path), "wb") as wf:
            wf.setnchannels(1)      # Mono
            wf.setsampwidth(2)      # 16-bit
            wf.setframerate(sample_rate)
            max_amp = 32767.0 * 0.8  # 80% volume headroom
            raw_frames = bytearray()
            for s in samples:
                val = max(-32768, min(32767, int(s * max_amp)))
                raw_frames.extend(struct.pack("<h", val))
            wf.writeframes(raw_frames)

        temp_path.replace(output_path)
        logger.debug("Generated synthetic alert chime at %s (%d bytes)", output_path, output_path.stat().st_size)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return output_path


def ensure_chime_file(runtime_dir: Optional[Path] = None) -> Path:
    """Ensure a valid alert chime WAV file exists and return its path."""
    target_dir = Path(runtime_dir) if runtime_dir else Path.home() / ".config" / "pengulink"
    target_path = target_dir / "alert.wav"
    if target_path.exists() and target_path.stat().st_size > 44:
        return target_path
    return generate_alert_chime(target_path)


# =====================================================================
# Audio Process Abstraction & Mock Processes
# =====================================================================

class MockAudioProcess:
    """Mock process object simulating an audio player subprocess."""

    def __init__(self, duration: float = 0.05) -> None:
        self.pid: int = 99999
        self.returncode: Optional[int] = None
        self.duration: float = duration
        self.terminated: bool = False
        self.killed: bool = False
        self._done_event = asyncio.Event()

    async def wait(self) -> int:
        try:
            await asyncio.wait_for(self._done_event.wait(), timeout=self.duration)
        except asyncio.TimeoutError:
            pass
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        self._done_event.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._done_event.set()


class SilentAudioProcess:
    """Simulates a silent audio playback iteration without noise or busy spinning."""

    def __init__(self, duration: float = 1.0) -> None:
        self.pid: int = 99998
        self.returncode: Optional[int] = None
        self.duration: float = duration
        self._done_event = asyncio.Event()

    async def wait(self) -> int:
        try:
            await asyncio.wait_for(self._done_event.wait(), timeout=self.duration)
        except asyncio.TimeoutError:
            pass
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15
        self._done_event.set()

    def kill(self) -> None:
        self.returncode = -9
        self._done_event.set()


# =====================================================================
# AudioBackend Hierarchy
# =====================================================================

class AudioBackend(ABC):
    """Abstract interface defining the audio playback backend contract."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifying name of the backend."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Return True if backend tools are available on the host."""
        ...

    @abstractmethod
    async def play(self, wav_path: Path) -> Any:
        """Spawn audio playback process for the given WAV file.

        Returns an object with wait(), terminate(), and kill() methods.
        """
        ...


class PipeWireAudioBackend(AudioBackend):
    """PipeWire audio playback using pw-cat -p."""

    @property
    def name(self) -> str:
        return "pipewire"

    async def is_available(self) -> bool:
        return shutil.which("pw-cat") is not None

    async def play(self, wav_path: Path) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            "pw-cat",
            "-p",
            str(wav_path),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )


class PulseAudioAudioBackend(AudioBackend):
    """PulseAudio audio playback using paplay."""

    @property
    def name(self) -> str:
        return "pulseaudio"

    async def is_available(self) -> bool:
        return shutil.which("paplay") is not None

    async def play(self, wav_path: Path) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            "paplay",
            str(wav_path),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )


class AlsaAudioBackend(AudioBackend):
    """ALSA audio playback using aplay -q."""

    @property
    def name(self) -> str:
        return "alsa"

    async def is_available(self) -> bool:
        return shutil.which("aplay") is not None

    async def play(self, wav_path: Path) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            "aplay",
            "-q",
            str(wav_path),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )


class MockAudioBackend(AudioBackend):
    """In-memory mock backend for automated testing and CI."""

    def __init__(self, playback_duration: float = 0.05) -> None:
        self.playback_duration: float = playback_duration
        self.played_files: List[Path] = []
        self.play_count: int = 0
        self.last_process: Optional[MockAudioProcess] = None

    @property
    def name(self) -> str:
        return "mock"

    async def is_available(self) -> bool:
        return True

    async def play(self, wav_path: Path) -> MockAudioProcess:
        self.played_files.append(Path(wav_path))
        self.play_count += 1
        proc = MockAudioProcess(duration=self.playback_duration)
        self.last_process = proc
        return proc

    def reset(self) -> None:
        self.played_files.clear()
        self.play_count = 0
        self.last_process = None


class SilentFallbackBackend(AudioBackend):
    """Fallback backend that logs a warning and performs graceful no-op playback."""

    def __init__(self, simulated_duration: float = 1.0) -> None:
        self.simulated_duration: float = simulated_duration
        self._logged_warning: bool = False

    @property
    def name(self) -> str:
        return "silent"

    async def is_available(self) -> bool:
        return True

    async def play(self, wav_path: Path) -> SilentAudioProcess:
        if not self._logged_warning:
            logger.warning("No functional audio backend detected; alert running in silent fallback mode")
            self._logged_warning = True
        return SilentAudioProcess(duration=self.simulated_duration)


def detect_audio_backend(backend_name: Optional[str] = None) -> AudioBackend:
    """Detect and instantiate the appropriate audio backend.

    Resolution precedence:
    1. Explicit `backend_name` parameter.
    2. `PENGULINK_ALERT_BACKEND` environment variable.
    3. Auto-detection: PipeWire -> PulseAudio -> ALSA -> SilentFallbackBackend.
    """
    req = (backend_name or os.environ.get("PENGULINK_ALERT_BACKEND", "")).strip().lower()

    if req:
        if req == "pipewire":
            return PipeWireAudioBackend()
        elif req in ("pulseaudio", "pulse"):
            return PulseAudioAudioBackend()
        elif req == "alsa":
            return AlsaAudioBackend()
        elif req in ("mock", "test"):
            return MockAudioBackend()
        elif req in ("silent", "none"):
            return SilentFallbackBackend()
        else:
            raise AudioBackendError(f"Unknown audio backend requested: '{req}'")

    if shutil.which("pw-cat") is not None:
        logger.debug("Auto-detected PipeWire audio backend")
        return PipeWireAudioBackend()
    if shutil.which("paplay") is not None:
        logger.debug("Auto-detected PulseAudio audio backend")
        return PulseAudioAudioBackend()
    if shutil.which("aplay") is not None:
        logger.debug("Auto-detected ALSA audio backend")
        return AlsaAudioBackend()

    logger.info("No audio tools detected; falling back to SilentFallbackBackend")
    return SilentFallbackBackend()


# =====================================================================
# AlertService Core
# =====================================================================

class AlertService:
    """UI-independent desktop alert service coordinating audio and visual locating alerts."""

    def __init__(
        self,
        backend: Optional[AudioBackend] = None,
        event_bus: Optional[EventBus] = None,
        runtime_dir: Optional[Path] = None,
    ) -> None:
        self._backend: AudioBackend = backend or detect_audio_backend()
        self._event_bus: Optional[EventBus] = event_bus
        self._runtime_dir: Optional[Path] = runtime_dir
        self._wav_path: Optional[Path] = None

        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._loop_task: Optional[asyncio.Task[None]] = None
        self._notification_task: Optional[asyncio.Task[None]] = None
        self._current_process: Optional[Any] = None


        self._is_active: bool = False
        self._device_id: Optional[str] = None
        self._source: Optional[str] = None
        self._start_time: float = 0.0
        self._max_duration: float = DEFAULT_MAX_ALERT_DURATION
        self._stop_reason: str = "manual"

    @property
    def backend(self) -> AudioBackend:
        return self._backend

    @property
    def active(self) -> bool:
        """Property alias for active state."""
        return self._is_active

    def is_active(self) -> bool:
        """Return True if an alert is currently ringing."""
        return self._is_active and not self._stop_event.is_set()

    def get_status(self) -> Dict[str, Any]:
        """Introspect full status of the alert service."""
        active = self._is_active and not self._stop_event.is_set()
        remaining: float = 0.0
        if active:
            elapsed = time.monotonic() - self._start_time
            remaining = max(0.0, self._max_duration - elapsed)
        return {
            "active": active,
            "device_id": self._device_id if active else None,
            "source": self._source if active else None,
            "duration": self._max_duration if active else 0.0,
            "remaining": round(remaining, 2),
            "remaining_seconds": round(remaining, 2),
            "backend": self._backend.name,
            "started_at": self._start_time if active else None,
        }

    async def start_alert(
        self,
        device_id: str = "local_cli",
        source: str = "remote",
        max_duration: float = DEFAULT_MAX_ALERT_DURATION,
    ) -> bool:
        """Trigger or refresh the locating alert.

        Guaranteed 30.0s hard cap runaway protection and single-instance mutex.
        """
        # If an active alert is in the process of stopping, wait for loop task to drain cleanly
        if self._is_active and self._stop_event.is_set() and self._loop_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._loop_task), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        async with self._lock:
            # Enforce hard 30s runaway protection cap
            clamped_duration = min(max(max_duration, 0.05), DEFAULT_MAX_ALERT_DURATION)

            if self._is_active and not self._stop_event.is_set():
                logger.info(
                    "Alert already active from '%s' (%s); refreshing countdown to %.1fs",
                    _sanitize_log_str(self._device_id),
                    _sanitize_log_str(self._source),
                    clamped_duration,
                )
                self._max_duration = clamped_duration
                self._start_time = time.monotonic()
                return True

            # Ensure synthetic PCM WAV file exists
            if self._wav_path is None or not self._wav_path.exists():
                self._wav_path = ensure_chime_file(self._runtime_dir)

            self._is_active = True
            self._device_id = device_id
            self._source = source
            self._start_time = time.monotonic()
            self._max_duration = clamped_duration
            self._stop_reason = "timeout"
            self._stop_event.clear()

            # Publish AlertTriggered event
            if self._event_bus:
                await self._event_bus.publish(
                    AlertTriggered(
                        device_id=device_id,
                        source=source,
                        max_duration=clamped_duration,
                    )
                )

            # Spawn non-blocking visual companion notification
            if self._notification_task is not None and not self._notification_task.done():
                self._notification_task.cancel()
            self._notification_task = asyncio.create_task(self._send_desktop_notification(device_id, source))

            # Spawn playback loop task
            self._loop_task = asyncio.create_task(self._playback_loop(clamped_duration))
            logger.info(
                "Started alert for '%s' (source=%s, duration=%.1fs, backend=%s)",
                _sanitize_log_str(device_id),
                _sanitize_log_str(source),
                clamped_duration,
                self._backend.name,
            )
            return True

    async def stop_alert(self, reason: str = "manual") -> bool:
        """Stop the active alert immediately, terminating child processes cleanly."""
        if not self._is_active:
            return False

        loop_task = self._loop_task
        notification_task = self._notification_task

        self._stop_reason = reason
        self._stop_event.set()
        self._is_active = False

        if notification_task is not None and not notification_task.done():
            notification_task.cancel()
            try:
                await notification_task
            except (asyncio.CancelledError, Exception):
                pass

        # Immediate process termination (SIGTERM -> SIGKILL)
        proc = self._current_process

        if proc is not None:
            await self._terminate_process(proc)

        # Await loop task completion cleanly without deadlock
        if loop_task is not None and not loop_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(loop_task), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                if not loop_task.done():
                    loop_task.cancel()
                    try:
                        await loop_task
                    except asyncio.CancelledError:
                        pass

        logger.info("Alert stopped (reason=%s)", reason)
        return True

    async def stop(self) -> None:
        """Lifecycle hook invoked during daemon shutdown."""
        await self.stop_alert(reason="shutdown")

    async def _playback_loop(self, duration: float) -> None:
        """Looping playback engine until stop_event or runaway timeout."""
        start_time = self._start_time
        stop_reason = "timeout"

        try:
            while not self._stop_event.is_set():
                elapsed = time.monotonic() - self._start_time
                remaining = self._max_duration - elapsed
                if remaining <= 0:
                    stop_reason = "timeout"
                    break

                try:
                    proc = await self._backend.play(self._wav_path)  # type: ignore
                    self._current_process = proc
                except Exception as exc:
                    logger.error("Audio backend '%s' failed playback: %s", self._backend.name, exc)
                    stop_reason = "error"
                    break

                # Wait for chime playback completion or stop_event
                wait_proc = asyncio.create_task(proc.wait())
                wait_stop = asyncio.create_task(self._stop_event.wait())

                done, pending = await asyncio.wait(
                    [wait_proc, wait_stop],
                    timeout=max(0.01, remaining),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

                if self._stop_event.is_set():
                    stop_reason = self._stop_reason
                    await self._terminate_process(proc)
                    break

                if wait_proc not in done:
                    # Chime playback exceeded remaining runaway window
                    stop_reason = "timeout"
                    await self._terminate_process(proc)
                    break

                # Retrieve result from wait_proc to avoid unretrieved exception and check return code
                try:
                    ret = wait_proc.result()
                    if ret not in (0, None) and not self._stop_event.is_set():
                        logger.error("Audio process '%s' exited with non-zero exit code: %s", self._backend.name, ret)
                        stop_reason = "error"
                        break
                except Exception as exc:
                    logger.error("Audio playback process error: %s", exc)
                    stop_reason = "error"
                    break

                # Inter-chime interval (wait on stop_event)
                elapsed = time.monotonic() - self._start_time
                remaining = self._max_duration - elapsed
                if remaining <= 0 or self._stop_event.is_set():
                    break

                pause_duration = min(INTER_CHIME_DELAY, remaining)
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=pause_duration)
                    stop_reason = self._stop_reason
                    break
                except asyncio.TimeoutError:
                    pass

        finally:
            if self._stop_event.is_set() and self._stop_reason:
                stop_reason = self._stop_reason
            async with self._lock:
                played_duration = time.monotonic() - start_time

                saved_device_id = self._device_id or ""
                self._is_active = False
                self._current_process = None
                self._device_id = None
                self._source = None

                if self._event_bus:
                    await self._event_bus.publish(
                        AlertStopped(
                            device_id=saved_device_id,
                            reason=stop_reason,
                            duration_played=round(played_duration, 2),
                        )
                    )
                self._loop_task = None

    async def _terminate_process(self, proc: Any) -> None:
        """2-stage subprocess termination: SIGTERM -> SIGKILL."""
        if proc is None:
            return
        if hasattr(proc, "returncode") and proc.returncode is not None:
            return

        # Stage 1: Graceful SIGTERM
        try:
            if hasattr(proc, "pid") and proc.pid:
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    if hasattr(proc, "terminate"):
                        proc.terminate()
            elif hasattr(proc, "terminate"):
                proc.terminate()

            if hasattr(proc, "wait"):
                await asyncio.wait_for(proc.wait(), timeout=SUBPROCESS_TERMINATE_TIMEOUT)
                return
        except (asyncio.TimeoutError, ProcessLookupError, OSError, Exception):
            pass

        # Stage 2: Forceful SIGKILL
        try:
            if hasattr(proc, "pid") and proc.pid:
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    if hasattr(proc, "kill"):
                        proc.kill()
            elif hasattr(proc, "kill"):
                proc.kill()

            if hasattr(proc, "wait"):
                await asyncio.wait_for(proc.wait(), timeout=SUBPROCESS_TERMINATE_TIMEOUT)
        except Exception as exc:
            logger.debug("Suppressed error during audio process kill escalation: %s", exc)

    async def _send_desktop_notification(self, device_id: str, source: str) -> None:
        """Fire-and-forget visual notification companion via notify-send."""
        if shutil.which("notify-send") is None:
            return
        proc: Optional[asyncio.subprocess.Process] = None
        try:
            cmd = [
                "notify-send",
                "-u", "critical",
                "-a", "PenguLink",
                "-i", "dialog-information",
                "Find This Device",
                f"Locate alert triggered by {_sanitize_log_str(device_id)} ({_sanitize_log_str(source)})",
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.CancelledError:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("Desktop notify-send companion failed (non-fatal): %s", exc)



__all__ = [
    "DEFAULT_MAX_ALERT_DURATION",
    "INTER_CHIME_DELAY",
    "generate_alert_chime",
    "ensure_chime_file",
    "MockAudioProcess",
    "SilentAudioProcess",
    "AudioBackend",
    "PipeWireAudioBackend",
    "PulseAudioAudioBackend",
    "AlsaAudioBackend",
    "MockAudioBackend",
    "SilentFallbackBackend",
    "detect_audio_backend",
    "AlertService",
]

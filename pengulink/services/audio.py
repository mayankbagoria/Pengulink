"""pengulink.services.audio
~~~~~~~~~~~~~~~~~~~~~~~~
Host Linux audio sink discovery, inspection, and volume control subsystem.

Provides:
- AudioSink: Strongly typed immutable/serializable representation of an audio output sink
- AudioSinkBackend: Abstract interface for Linux sound server controllers
- PipeWireAudioBackend: WirePlumber / PipeWire controller via wpctl / pw-dump
- PulseAudioBackend: PulseAudio controller via pactl
- MockAudioSinkBackend: Deterministic in-memory backend for testing and headless CI
- SilentFallbackSinkBackend: Graceful no-op fallback when audio services are absent
- LocalAudioService: High-level concurrency-safe audio service with EventBus integration
- detect_audio_backend: Auto-detecting backend factory respecting PENGULINK_AUDIO_BACKEND
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pengulink.core.events import (
    AudioSinkListEvent,
    AudioSinkUpdatedEvent,
    EventBus,
)
from pengulink.core.exceptions import (
    AudioBackendError,
    AudioUnavailableError,
    PenguLinkError,
)

logger = logging.getLogger("pengulink.services.audio")

# Standard volume scale ceiling matching PulseAudio PA_VOLUME_NORM
MAX_AUDIO_VOLUME: int = 65536
DEFAULT_COMMAND_TIMEOUT: float = 3.0


class AudioError(PenguLinkError):
    """Base exception for audio service errors."""


@dataclass
class AudioSink:
    """Represents a physical or virtual audio output sink."""

    name: str
    description: str = ""
    volume: int = 0
    max_volume: int = MAX_AUDIO_VOLUME
    muted: bool = False
    enabled: bool = False

    def __post_init__(self) -> None:
        if self.max_volume <= 0:
            self.max_volume = MAX_AUDIO_VOLUME
        self.volume = max(0, min(int(self.volume), self.max_volume))
        if not self.description:
            self.description = self.name or "Audio Output"

    @property
    def volume_percent(self) -> int:
        """Normalized integer percentage between 0 and 100."""
        if self.max_volume <= 0:
            return 0
        pct = round((self.volume / self.max_volume) * 100)
        return max(0, min(100, pct))

    def to_wire_dict(self) -> dict[str, Any]:
        """Serialize for upstream KDE Connect wire packets (camelCase maxVolume)."""
        return {
            "name": self.name,
            "description": self.description,
            "muted": self.muted,
            "volume": self.volume,
            "maxVolume": self.max_volume,
            "enabled": self.enabled,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize for IPC, CLI, and debugging representations."""
        return {
            "name": self.name,
            "description": self.description,
            "muted": self.muted,
            "volume": self.volume,
            "maxVolume": self.max_volume,
            "max_volume": self.max_volume,
            "volume_percent": self.volume_percent,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioSink:
        """Create an AudioSink instance from a dictionary safely."""
        name = str(data.get("name", "")).strip()
        description = str(
            data.get("description")
            or data.get("desc")
            or data.get("nick")
            or name
            or "Audio Output"
        )
        max_vol_raw = (
            data.get("maxVolume") or data.get("max_volume") or MAX_AUDIO_VOLUME
        )
        try:
            max_vol = int(float(max_vol_raw))
        except (ValueError, TypeError, OverflowError):
            max_vol = MAX_AUDIO_VOLUME
        if max_vol <= 0:
            max_vol = MAX_AUDIO_VOLUME
        raw_vol = data.get("volume", 0)
        try:
            vol = int(float(raw_vol))
        except (ValueError, TypeError, OverflowError):
            vol = 0
        vol = max(0, min(vol, max_vol))
        muted = bool(data.get("muted", False))
        enabled = bool(data.get("enabled", False))
        return cls(
            name=name,
            description=description,
            volume=vol,
            max_volume=max_vol,
            muted=muted,
            enabled=enabled,
        )


class AudioSinkBackend(ABC):
    """Abstract base class for audio server backends."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this audio backend is available on the host system."""
        ...

    @abstractmethod
    async def list_sinks(self) -> list[AudioSink]:
        """List all available audio output sinks."""
        ...

    @abstractmethod
    async def set_volume(self, name: str, volume: int) -> bool:
        """Set volume of a sink (0..max_volume).

        Upstream KDE Connect invariant: setting volume automatically unmutes the sink.
        """
        ...

    @abstractmethod
    async def set_muted(self, name: str, muted: bool) -> bool:
        """Set mute state of a sink."""
        ...

    async def set_mute(self, name: str, muted: bool) -> bool:
        """Alias for set_muted."""
        return await self.set_muted(name, muted)

    @abstractmethod
    async def set_default(self, name: str) -> bool:
        """Set sink as the default output device."""
        ...


class MockAudioSinkBackend(AudioSinkBackend):
    """In-memory audio backend for testing and headless CI environments."""

    def __init__(self, initial_sinks: list[AudioSink] | None = None) -> None:
        self._sinks: dict[str, AudioSink] = {}
        if initial_sinks is not None:
            for s in initial_sinks:
                self.add_sink(s)
        else:
            self.add_sink(
                AudioSink(
                    name="alsa_output.pci-0000_00_1f.3.speaker",
                    description="Internal Speakers",
                    volume=35389,
                    max_volume=MAX_AUDIO_VOLUME,
                    muted=False,
                    enabled=True,
                )
            )
            self.add_sink(
                AudioSink(
                    name="alsa_output.pci-0000_00_1f.3.hdmi",
                    description="HDMI / DisplayPort Audio",
                    volume=65536,
                    max_volume=MAX_AUDIO_VOLUME,
                    muted=False,
                    enabled=False,
                )
            )

    async def is_available(self) -> bool:
        return True

    def add_sink(self, sink: AudioSink) -> None:
        """Add or update an in-memory sink fixture."""
        self._sinks[sink.name] = AudioSink(
            name=sink.name,
            description=sink.description,
            volume=sink.volume,
            max_volume=sink.max_volume,
            muted=sink.muted,
            enabled=sink.enabled,
        )

    def remove_sink(self, name: str) -> bool:
        """Remove a sink fixture by name."""
        return self._sinks.pop(name, None) is not None

    def reset(self) -> None:
        """Clear all registered sinks."""
        self._sinks.clear()

    async def list_sinks(self) -> list[AudioSink]:
        return [
            AudioSink(
                name=s.name,
                description=s.description,
                volume=s.volume,
                max_volume=s.max_volume,
                muted=s.muted,
                enabled=s.enabled,
            )
            for s in self._sinks.values()
        ]

    async def set_volume(self, name: str, volume: int) -> bool:
        sink = self._sinks.get(name)
        if not sink:
            return False
        clamped = max(0, min(int(volume), sink.max_volume))
        sink.volume = clamped
        # Upstream invariant: setting volume automatically unmutes
        sink.muted = False
        return True

    async def set_muted(self, name: str, muted: bool) -> bool:
        sink = self._sinks.get(name)
        if not sink:
            return False
        sink.muted = bool(muted)
        return True

    async def set_default(self, name: str) -> bool:
        if name not in self._sinks:
            return False
        for s in self._sinks.values():
            s.enabled = s.name == name
        return True


class SilentFallbackSinkBackend(AudioSinkBackend):
    """Graceful no-op fallback when no sound server is available."""

    async def is_available(self) -> bool:
        return True

    async def list_sinks(self) -> list[AudioSink]:
        return []

    async def set_volume(self, name: str, volume: int) -> bool:
        return False

    async def set_muted(self, name: str, muted: bool) -> bool:
        return False

    async def set_default(self, name: str) -> bool:
        return False


class PipeWireAudioBackend(AudioSinkBackend):
    """WirePlumber / PipeWire audio backend using wpctl and pw-dump."""

    def __init__(self, command_timeout: float = DEFAULT_COMMAND_TIMEOUT) -> None:
        self.command_timeout = command_timeout
        self._name_to_id_cache: dict[str, str] = {}
        self._id_to_name_cache: dict[str, str] = {}

    @classmethod
    def is_supported(cls) -> bool:
        """Check if wpctl is installed on PATH."""
        return shutil.which("wpctl") is not None

    async def _run_command(self, cmd: list[str]) -> tuple[int, str, str]:
        """Execute a subprocess asynchronously with bounded timeout."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.command_timeout,
            )
            return (
                proc.returncode if proc.returncode is not None else 1,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except (OSError, asyncio.TimeoutError, RuntimeError, ValueError) as exc:
            logger.debug("PipeWireAudioBackend command %s failed: %s", cmd, exc)
            return 1, "", str(exc)

    async def is_available(self) -> bool:
        if not self.is_supported():
            return False
        code, stdout, _ = await self._run_command(["wpctl", "status"])
        return code == 0 and bool(stdout.strip())

    async def _refresh_node_names(self) -> None:
        """Try to resolve canonical node.name from pw-dump if available."""
        if not shutil.which("pw-dump"):
            return
        code, stdout, _ = await self._run_command(["pw-dump"])
        if code != 0 or not stdout.strip():
            return
        try:
            data = json.loads(stdout)
            if not isinstance(data, list):
                return
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                info = obj.get("info") or {}
                if not isinstance(info, dict):
                    info = {}
                props = info.get("props") or {}
                if not isinstance(props, dict):
                    props = {}
                if props.get("media.class") == "Audio/Sink":
                    node_id = str(obj.get("id", "")).strip()
                    node_name = str(props.get("node.name", "")).strip()
                    if node_id and node_name:
                        self._name_to_id_cache[node_name] = node_id
                        self._id_to_name_cache[node_id] = node_name
        except (
            json.JSONDecodeError,
            KeyError,
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            logger.debug("Failed to parse pw-dump JSON: %s", exc)

    async def list_sinks(self) -> list[AudioSink]:
        code, stdout, _ = await self._run_command(["wpctl", "status"])
        if code != 0 or not stdout:
            return []

        await self._refresh_node_names()

        in_audio = False
        in_sinks = False
        sinks: list[AudioSink] = []

        sink_pattern = re.compile(
            r"(\*?)\s*(\d+)\.\s+(.*?)\s*\[vol:\s*([0-9.]+)(?:\s+(MUTED))?\]"
        )

        for line in stdout.splitlines():
            if "Audio" in line and not line.startswith(" "):
                in_audio = True
                continue
            if in_audio:
                if "Video" in line and not line.startswith(" "):
                    break
                if "Sinks:" in line:
                    in_sinks = True
                    continue
                if in_sinks and (
                    "Sources:" in line or "Filters:" in line or "Streams:" in line
                ):
                    in_sinks = False
                    continue
                if in_sinks:
                    match = sink_pattern.search(line)
                    if match:
                        prefix_star = match.group(1) or (
                            "*" in line[: line.find(match.group(2))]
                        )
                        node_id = match.group(2)
                        desc = match.group(3).strip()
                        try:
                            vol_float = float(match.group(4))
                        except (ValueError, TypeError):
                            vol_float = 0.0
                        muted = bool(match.group(5))
                        vol_int = round(vol_float * MAX_AUDIO_VOLUME)
                        vol_int = max(0, min(vol_int, MAX_AUDIO_VOLUME))

                        # Map to canonical node.name if known, else node_id
                        canonical_name = self._id_to_name_cache.get(node_id, node_id)
                        self._name_to_id_cache[canonical_name] = node_id
                        self._name_to_id_cache[node_id] = node_id

                        sinks.append(
                            AudioSink(
                                name=canonical_name,
                                description=desc or canonical_name,
                                volume=vol_int,
                                max_volume=MAX_AUDIO_VOLUME,
                                muted=muted,
                                enabled=bool(prefix_star),
                            )
                        )

        return sinks

    def _resolve_target_id(self, name: str) -> str:
        """Resolve sink name or alias to a WirePlumber target ID."""
        cleaned = name.strip()
        if cleaned in ("@DEFAULT_AUDIO_SINK@", "@DEFAULT_SINK@", "@DEFAULT@"):
            return "@DEFAULT_AUDIO_SINK@"
        if cleaned in self._name_to_id_cache:
            return self._name_to_id_cache[cleaned]
        if cleaned.isdigit():
            return cleaned
        return cleaned

    async def set_volume(self, name: str, volume: int) -> bool:
        target_id = self._resolve_target_id(name)
        vol_float = max(0.0, min(1.5, volume / float(MAX_AUDIO_VOLUME)))
        vol_str = f"{vol_float:.2f}"
        code, _, _ = await self._run_command(
            ["wpctl", "set-volume", "-l", "1.5", target_id, vol_str]
        )
        if code != 0:
            return False
        # Upstream KDE Connect invariant: setting volume automatically unmutes
        await self._run_command(["wpctl", "set-mute", target_id, "0"])
        return True

    async def set_muted(self, name: str, muted: bool) -> bool:
        target_id = self._resolve_target_id(name)
        flag = "1" if muted else "0"
        code, _, _ = await self._run_command(["wpctl", "set-mute", target_id, flag])
        return code == 0

    async def set_default(self, name: str) -> bool:
        target_id = self._resolve_target_id(name)
        code, _, _ = await self._run_command(["wpctl", "set-default", target_id])
        return code == 0


class PulseAudioBackend(AudioSinkBackend):
    """PulseAudio backend using pactl."""

    def __init__(self, command_timeout: float = DEFAULT_COMMAND_TIMEOUT) -> None:
        self.command_timeout = command_timeout

    @classmethod
    def is_supported(cls) -> bool:
        """Check if pactl is installed on PATH."""
        return shutil.which("pactl") is not None

    async def _run_command(self, cmd: list[str]) -> tuple[int, str, str]:
        """Execute a pactl subprocess asynchronously."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.command_timeout,
            )
            return (
                proc.returncode if proc.returncode is not None else 1,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except (OSError, asyncio.TimeoutError, RuntimeError, ValueError) as exc:
            logger.debug("PulseAudioBackend command %s failed: %s", cmd, exc)
            return 1, "", str(exc)

    async def is_available(self) -> bool:
        if not self.is_supported():
            return False
        code, stdout, _ = await self._run_command(["pactl", "info"])
        return code == 0 and bool(stdout.strip())

    async def _get_default_sink_name(self) -> str:
        """Query the default sink name via pactl info."""
        code, stdout, _ = await self._run_command(["pactl", "info"])
        if code == 0:
            for line in stdout.splitlines():
                if line.startswith("Default Sink:"):
                    return line.split(":", 1)[1].strip()
        return ""

    async def list_sinks(self) -> list[AudioSink]:
        default_sink = await self._get_default_sink_name()

        # Try JSON format first (supported in modern PulseAudio)
        code, stdout, _ = await self._run_command(
            ["pactl", "--format=json", "list", "sinks"]
        )
        if code == 0 and stdout.strip():
            try:
                data = json.loads(stdout)
                if isinstance(data, list):
                    sinks: list[AudioSink] = []
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("name", "")).strip()
                        desc = str(item.get("description", "")).strip() or name
                        muted = bool(item.get("mute", False))

                        # Volume extraction
                        vol_val = 0
                        vol_dict = item.get("volume", {})
                        if isinstance(vol_dict, dict):
                            # Look at first channel value or front-left
                            values = []
                            for ch_data in vol_dict.values():
                                if isinstance(ch_data, dict) and "value" in ch_data:
                                    values.append(int(ch_data["value"]))
                            if values:
                                vol_val = int(sum(values) / len(values))
                        vol_val = max(0, min(vol_val, MAX_AUDIO_VOLUME))

                        is_default = (name == default_sink) if default_sink else False
                        sinks.append(
                            AudioSink(
                                name=name,
                                description=desc,
                                volume=vol_val,
                                max_volume=MAX_AUDIO_VOLUME,
                                muted=muted,
                                enabled=is_default,
                            )
                        )
                    return sinks
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                logger.debug(
                    "Failed parsing pactl JSON sinks, trying text parser: %s", exc
                )

        # Fallback to text parsing
        code, stdout, _ = await self._run_command(["pactl", "list", "sinks"])
        if code != 0 or not stdout:
            return []

        return self._parse_pactl_text_sinks(stdout, default_sink)

    def _parse_pactl_text_sinks(self, text: str, default_sink: str) -> list[AudioSink]:
        sinks: list[AudioSink] = []
        current: dict[str, Any] = {}

        for line in text.splitlines():
            line_str = line.strip()
            if line.startswith("Sink #"):
                if "name" in current:
                    sinks.append(
                        self._build_sink_from_pactl_dict(current, default_sink)
                    )
                current = {}
            elif line_str.startswith("Name:"):
                current["name"] = line_str.split(":", 1)[1].strip()
            elif line_str.startswith("Description:"):
                current["description"] = line_str.split(":", 1)[1].strip()
            elif line_str.startswith("Mute:"):
                current["muted"] = line_str.split(":", 1)[1].strip().lower() == "yes"
            elif line_str.startswith("Volume:"):
                # Volume: front-left: 35389 / 54% / -16.03 dB, ...
                vol_match = re.search(r"/\s*(\d+)%", line_str)
                if vol_match:
                    pct = int(vol_match.group(1))
                    current["volume"] = round((pct / 100.0) * MAX_AUDIO_VOLUME)
                else:
                    raw_match = re.search(r":\s*(\d+)\s*/", line_str)
                    if raw_match:
                        current["volume"] = int(raw_match.group(1))

        if "name" in current:
            sinks.append(self._build_sink_from_pactl_dict(current, default_sink))

        return sinks

    def _build_sink_from_pactl_dict(
        self, d: dict[str, Any], default_sink: str
    ) -> AudioSink:
        name = d.get("name", "")
        desc = d.get("description", "") or name
        vol = max(0, min(int(d.get("volume", 0)), MAX_AUDIO_VOLUME))
        muted = bool(d.get("muted", False))
        enabled = (name == default_sink) if default_sink else False
        return AudioSink(
            name=name,
            description=desc,
            volume=vol,
            max_volume=MAX_AUDIO_VOLUME,
            muted=muted,
            enabled=enabled,
        )

    async def set_volume(self, name: str, volume: int) -> bool:
        clamped = max(0, min(int(volume), MAX_AUDIO_VOLUME))
        code, _, _ = await self._run_command(
            ["pactl", "set-sink-volume", name, str(clamped)]
        )
        if code != 0:
            return False
        # Upstream invariant: setting volume automatically unmutes
        await self._run_command(["pactl", "set-sink-mute", name, "0"])
        return True

    async def set_muted(self, name: str, muted: bool) -> bool:
        flag = "1" if muted else "0"
        code, _, _ = await self._run_command(["pactl", "set-sink-mute", name, flag])
        return code == 0

    async def set_default(self, name: str) -> bool:
        code, _, _ = await self._run_command(["pactl", "set-default-sink", name])
        return code == 0


def detect_audio_backend(preferred_backend: str | None = None) -> AudioSinkBackend:
    """Detect and instantiate the most suitable audio backend.

    Resolution precedence:
    1. Explicit `preferred_backend` argument.
    2. `PENGULINK_AUDIO_BACKEND` environment variable.
       - 'pipewire' | 'wpctl' -> PipeWireAudioBackend()
       - 'pulse' | 'pulseaudio' | 'pactl' -> PulseAudioBackend()
       - 'mock' | 'memory' | 'test' -> MockAudioSinkBackend()
       - 'silent' | 'null' | 'none' -> SilentFallbackSinkBackend()
    3. Auto-detection cascade:
       - PipeWireAudioBackend if `wpctl` is present.
       - PulseAudioBackend if `pactl` is present.
       - SilentFallbackSinkBackend fallback.

    Raises:
        AudioBackendError: If an unrecognized backend name was explicitly requested.
    """
    req = (
        (preferred_backend or os.environ.get("PENGULINK_AUDIO_BACKEND", ""))
        .strip()
        .lower()
    )

    if req:
        if req in ("pipewire", "wpctl"):
            return PipeWireAudioBackend()
        elif req in ("pulse", "pulseaudio", "pactl"):
            return PulseAudioBackend()
        elif req in ("mock", "memory", "test"):
            return MockAudioSinkBackend()
        elif req in ("silent", "null", "none"):
            return SilentFallbackSinkBackend()
        else:
            raise AudioBackendError(f"Unknown audio backend requested: '{req}'")

    if PipeWireAudioBackend.is_supported():
        logger.debug("Auto-detected PipeWire audio backend via wpctl")
        return PipeWireAudioBackend()

    if PulseAudioBackend.is_supported():
        logger.debug("Auto-detected PulseAudio backend via pactl")
        return PulseAudioBackend()

    logger.debug(
        "No sound server CLI detected; falling back to SilentFallbackSinkBackend"
    )
    return SilentFallbackSinkBackend()


# Alias for factory function
detect_audio_sink_backend = detect_audio_backend


class AwaitableSinkList(list):
    """A list of AudioSink items that can be both iterated synchronously and awaited asynchronously."""

    def __init__(
        self,
        initial_items: list[AudioSink] | None = None,
        async_func: Any | None = None,
    ) -> None:
        super().__init__(initial_items or [])
        self._async_func = async_func

    def __await__(self):
        async def _resolve():
            if self._async_func is not None:
                res = await self._async_func()
                self.clear()
                self.extend(res)
            return self

        return _resolve().__await__()


class LocalAudioService:
    """High-level, thread/asyncio-safe local audio controller service."""

    def __init__(
        self,
        backend: AudioSinkBackend | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.backend = backend or detect_audio_backend()
        self.event_bus = event_bus
        self._lock = asyncio.Lock()
        self._cached_sinks: list[AudioSink] = []

    def _get_sync_sinks(self) -> list[AudioSink]:
        if hasattr(self.backend, "_sinks") and isinstance(self.backend._sinks, dict):
            return [
                AudioSink(
                    name=s.name,
                    description=s.description,
                    volume=s.volume,
                    max_volume=s.max_volume,
                    muted=s.muted,
                    enabled=s.enabled,
                )
                for s in self.backend._sinks.values()
            ]
        return list(self._cached_sinks)

    async def _list_sinks_unlocked(self) -> list[AudioSink]:
        sinks = await self.backend.list_sinks()
        self._cached_sinks = list(sinks)
        return sinks

    async def _get_sink_unlocked(self, name: str) -> AudioSink | None:
        sinks = await self._list_sinks_unlocked()
        for s in sinks:
            if s.name == name:
                return s
        return None

    async def is_available(self) -> bool:
        """Check if underlying backend is responsive."""
        return await self.backend.is_available()

    def list_sinks(self) -> Any:
        """List all audio output sinks concurrency-safely. Supports both await and synchronous iteration."""
        async def _async_runner() -> list[AudioSink]:
            async with self._lock:
                return await self._list_sinks_unlocked()

        return AwaitableSinkList(
            initial_items=self._get_sync_sinks(),
            async_func=_async_runner,
        )

    async def get_sink(self, name: str) -> AudioSink | None:
        """Retrieve a specific sink by name or node ID."""
        async with self._lock:
            return await self._get_sink_unlocked(name)

    async def set_volume(
        self,
        name: str,
        volume: int,
        is_percent: bool = False,
    ) -> bool:
        """Set volume of a sink by raw volume (0..65536) or percentage (0..100)."""
        target_vol = volume
        if is_percent:
            clamped_pct = max(0, min(100, int(volume)))
            target_vol = round((clamped_pct / 100.0) * MAX_AUDIO_VOLUME)
        target_vol = max(0, min(int(target_vol), MAX_AUDIO_VOLUME))

        updated_sink: AudioSink | None = None
        async with self._lock:
            success = await self.backend.set_volume(name, target_vol)
            if success and self.event_bus:
                updated_sink = await self._get_sink_unlocked(name)

        if updated_sink and self.event_bus:
            await self.event_bus.publish(
                AudioSinkUpdatedEvent(device_id=None, sink=updated_sink)
            )
        return success

    async def set_muted(self, name: str, muted: bool) -> bool:
        """Set mute state on a sink."""
        updated_sink: AudioSink | None = None
        async with self._lock:
            success = await self.backend.set_muted(name, bool(muted))
            if success and self.event_bus:
                updated_sink = await self._get_sink_unlocked(name)

        if updated_sink and self.event_bus:
            await self.event_bus.publish(
                AudioSinkUpdatedEvent(device_id=None, sink=updated_sink)
            )
        return success

    async def set_default(self, name: str) -> bool:
        """Set default output sink."""
        sinks: list[AudioSink] | None = None
        async with self._lock:
            success = await self.backend.set_default(name)
            if success and self.event_bus:
                sinks = await self._list_sinks_unlocked()

        if sinks is not None and self.event_bus:
            await self.event_bus.publish(
                AudioSinkListEvent(device_id=None, sinks=sinks)
            )
        return success


# Aliases
AudioService = LocalAudioService

__all__ = [
    "MAX_AUDIO_VOLUME",
    "AudioBackendError",
    "AudioError",
    "AudioService",
    "AudioSink",
    "AudioSinkBackend",
    "AudioUnavailableError",
    "AwaitableSinkList",
    "LocalAudioService",
    "MockAudioSinkBackend",
    "PipeWireAudioBackend",
    "PulseAudioBackend",
    "SilentFallbackSinkBackend",
    "detect_audio_backend",
    "detect_audio_sink_backend",
]

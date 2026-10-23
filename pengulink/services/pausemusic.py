"""pengulink.services.pausemusic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PauseMusic core service coordinating local host media player playback and audio muting on phone calls.

Provides:
- PauseMusicConfig: Dataclass model representing pause policies matching KDE Connect QML settings.
- PauseMusicService: Core concurrency-safe media and audio interruption coordinator.
- Selective playback resumption: Only players that were actively playing before the call are resumed.
- Selective audio unmuting: Only audio sinks that were unmuted before the call are unmuted.
- Idempotent overlapping call ref-counting: Zero media flapping across multi-call and multi-device transitions.
- Reflection suppression: Virtual KDE Connect players (org.mpris.MediaPlayer2.kdeconnect.*) are ignored.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pengulink.core.events import (
    CallStateChangedEvent,
    DeviceDisconnected,
    DeviceUnpaired,
    DeviceUnpairedEvent,
    EventBus,
)
from pengulink.core.exceptions import (
    PauseMusicConfigError,
    PenguLinkError,
)
from pengulink.plugins.mpris.adapter import MprisHostAdapter, detect_mpris_adapter
from pengulink.services.audio import LocalAudioService

logger = logging.getLogger("pengulink.services.pausemusic")

# Reflection suppression prefixes and exact player names
REFLECTION_PREFIXES: tuple[str, ...] = (
    "org.mpris.MediaPlayer2.kdeconnect.",
    "org.mpris.MediaPlayer2.pengulink.",
)
REFLECTION_EXACT: set[str] = {
    "org.mpris.MediaPlayer2.playerctld",
}


# Exception types caught during external media/audio coordination
COORDINATION_EXCEPTIONS = (
    PenguLinkError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    OSError,
    TimeoutError,
)
LIFECYCLE_EXCEPTIONS = (
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    KeyError,
)


def _coerce_bool(val: Any, default: bool = False) -> bool:
    """Coerce various string or numeric representations into a boolean."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "1", "yes", "on", "enabled"):
            return True
        if v in ("false", "0", "no", "off", "disabled", ""):
            return False
    return default


@dataclass
class PauseMusicConfig:
    """Configuration model for automatic media playback and audio pausing on phone calls.

    Canonical attributes:
        condition_talking: If False (default), pause on ringing or talking.
                           If True, pause only once the call is answered (talking).
        action_pause: If True (default), pause/resume local MPRIS media players.
        action_mute: If True, mute/unmute local audio output sinks. Default False.
        action_resume: If True (default), automatically resume media and restore
                       sink mute state when all calls conclude.
    """

    condition_talking: bool = False
    action_pause: bool = True
    action_mute: bool = False
    action_resume: bool = True

    # -----------------------------------------------------------------
    # Semantic Property Getters and Setters
    # -----------------------------------------------------------------

    @property
    def condition(self) -> str:
        """Semantic condition string: 'talking' or 'ringing'."""
        return "talking" if self.condition_talking else "ringing"

    @condition.setter
    def condition(self, value: str) -> None:
        if not isinstance(value, str):
            raise PauseMusicConfigError(f"condition must be a string, got {type(value).__name__}")
        norm = value.strip().lower()
        if norm == "talking":
            self.condition_talking = True
        elif norm == "ringing":
            self.condition_talking = False
        else:
            raise PauseMusicConfigError(f"Invalid condition '{value}'; must be 'ringing' or 'talking'")

    @property
    def action(self) -> str:
        """Semantic action string: 'both', 'pause', 'mute', or 'none'."""
        if self.action_pause and self.action_mute:
            return "both"
        if self.action_pause:
            return "pause"
        if self.action_mute:
            return "mute"
        return "none"

    @action.setter
    def action(self, value: str) -> None:
        if not isinstance(value, str):
            raise PauseMusicConfigError(f"action must be a string, got {type(value).__name__}")
        norm = value.strip().lower()
        if norm == "both":
            self.action_pause = True
            self.action_mute = True
        elif norm == "pause":
            self.action_pause = True
            self.action_mute = False
        elif norm == "mute":
            self.action_pause = False
            self.action_mute = True
        elif norm in ("none", ""):
            self.action_pause = False
            self.action_mute = False
        else:
            raise PauseMusicConfigError(
                f"Invalid action '{value}'; must be 'pause', 'mute', 'both', or 'none'"
            )

    @property
    def resume(self) -> bool:
        """Alias for action_resume."""
        return self.action_resume

    @resume.setter
    def resume(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise PauseMusicConfigError(f"resume must be a bool, got {type(value).__name__}")
        self.action_resume = value

    # -----------------------------------------------------------------
    # Serialization and Deserialization
    # -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a comprehensive dictionary."""
        return {
            "condition_talking": self.condition_talking,
            "action_pause": self.action_pause,
            "action_mute": self.action_mute,
            "action_resume": self.action_resume,
            "condition": self.condition,
            "action": self.action,
            "resume": self.action_resume,
            # Upstream KDE Connect QML / JSON keys
            "conditionTalking": self.condition_talking,
            "actionPause": self.action_pause,
            "actionMute": self.action_mute,
            "actionResume": self.action_resume,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PauseMusicConfig:
        """Construct PauseMusicConfig from dictionary with lenient key mapping."""
        if not isinstance(data, dict):
            raise PauseMusicConfigError(f"data must be a dict, got {type(data).__name__}")

        # Resolve condition_talking
        condition_talking = False
        if "condition" in data:
            val = str(data["condition"]).strip().lower()
            if val == "talking":
                condition_talking = True
            elif val == "ringing":
                condition_talking = False
            else:
                raise PauseMusicConfigError(f"Invalid condition '{data['condition']}'")
        elif "condition_talking" in data:
            condition_talking = _coerce_bool(data["condition_talking"], default=False)
        elif "conditionTalking" in data:
            condition_talking = _coerce_bool(data["conditionTalking"], default=False)

        # Resolve action_pause and action_mute
        action_pause = True
        action_mute = False
        if "action" in data and not ("action_pause" in data or "actionPause" in data):
            act = str(data["action"]).strip().lower()
            if act == "both":
                action_pause = True
                action_mute = True
            elif act == "pause":
                action_pause = True
                action_mute = False
            elif act == "mute":
                action_pause = False
                action_mute = True
            elif act in ("none", ""):
                action_pause = False
                action_mute = False
            else:
                raise PauseMusicConfigError(f"Invalid action '{data['action']}'")
        else:
            if "action_pause" in data:
                action_pause = _coerce_bool(data["action_pause"], default=True)
            elif "actionPause" in data:
                action_pause = _coerce_bool(data["actionPause"], default=True)

            if "action_mute" in data:
                action_mute = _coerce_bool(data["action_mute"], default=False)
            elif "actionMute" in data:
                action_mute = _coerce_bool(data["actionMute"], default=False)

        # Resolve action_resume
        action_resume = True
        if "action_resume" in data:
            action_resume = _coerce_bool(data["action_resume"], default=True)
        elif "actionResume" in data:
            action_resume = _coerce_bool(data["actionResume"], default=True)
        elif "resume" in data:
            action_resume = _coerce_bool(data["resume"], default=True)

        return cls(
            condition_talking=condition_talking,
            action_pause=action_pause,
            action_mute=action_mute,
            action_resume=action_resume,
        )


class PauseMusicService:
    """Core media playback and system volume coordination service on phone calls.

    Guarantees:
    - Overlapping call ref-counting invariant: Snapshot is strictly captured on
      0 -> 1 transition and resumed strictly on 1 -> 0 transition.
    - Stopped/paused media players and pre-muted sinks are preserved and never auto-resumed.
    - KDE Connect reflection players (org.mpris.MediaPlayer2.kdeconnect.*) are suppressed.
    - Concurrency safety via asyncio.Lock() without lock reentrancy deadlocks.
    """

    def __init__(
        self,
        config: PauseMusicConfig | None = None,
        mpris_adapter: MprisHostAdapter | None = None,
        audio_service: LocalAudioService | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.config = config or PauseMusicConfig()
        self._mpris_adapter = mpris_adapter or detect_mpris_adapter()
        self._audio_service = audio_service or LocalAudioService()
        self.event_bus = event_bus

        self._lock = asyncio.Lock()
        self._active_call_devices: set[str] = set()
        self._paused_players: set[str] = set()
        self._muted_sinks: set[str] = set()
        self._unsub_callbacks: list[Callable[[], None]] = []

        if self.event_bus:
            self._subscribe_events()

    @property
    def is_interrupted(self) -> bool:
        """True if an active media interruption is in progress."""
        return len(self._active_call_devices) > 0

    @property
    def active_call_devices(self) -> set[str]:
        """Set of device IDs currently causing active media interruption."""
        return set(self._active_call_devices)

    @property
    def paused_players(self) -> set[str]:
        """Set of MPRIS players that were paused by the current interruption."""
        return set(self._paused_players)

    @property
    def muted_sinks(self) -> set[str]:
        """Set of audio sinks that were muted by the current interruption."""
        return set(self._muted_sinks)

    def _subscribe_events(self) -> None:
        """Subscribe to call and device lifecycle events on EventBus."""
        if not self.event_bus:
            return
        self._unsub_callbacks.append(
            self.event_bus.subscribe(CallStateChangedEvent, self._on_call_state_changed)
        )
        self._unsub_callbacks.append(
            self.event_bus.subscribe(DeviceDisconnected, self._on_device_disconnected)
        )
        self._unsub_callbacks.append(
            self.event_bus.subscribe(DeviceUnpaired, self._on_device_unpaired)
        )
        if DeviceUnpairedEvent is not DeviceUnpaired:
            self._unsub_callbacks.append(
                self.event_bus.subscribe(DeviceUnpairedEvent, self._on_device_unpaired)
            )

    def _unsubscribe_events(self) -> None:
        """Clean up EventBus subscriptions."""
        for unsub in self._unsub_callbacks:
            try:
                unsub()
            except LIFECYCLE_EXCEPTIONS as exc:
                logger.debug("Error unsubscribing PauseMusicService event: %s", exc)
        self._unsub_callbacks.clear()

    async def _on_call_state_changed(self, event: CallStateChangedEvent) -> None:
        """Handle incoming CallStateChangedEvent from TelephonyService."""
        device_id = getattr(event, "device_id", "")
        new_state = getattr(event, "new_state", "")
        if device_id and new_state:
            await self.handle_call_state_changed(device_id, new_state)

    async def _on_device_disconnected(self, event: DeviceDisconnected) -> None:
        """Handle device disconnection."""
        device_id = getattr(event, "device_id", "")
        if device_id:
            await self.handle_device_disconnected(device_id)

    async def _on_device_unpaired(self, event: DeviceUnpaired) -> None:
        """Handle device unpairing."""
        device_id = getattr(event, "device_id", "")
        if device_id:
            await self.handle_device_disconnected(device_id)

    async def handle_call_state_changed(self, device_id: str, new_state: str) -> bool:
        """Process telephony call state transition for device_id."""
        clean_dev = str(device_id).strip()
        state = str(new_state).strip().upper()
        if not clean_dev:
            return False

        if self.config.condition_talking:
            if state == "TALKING":
                return await self.handle_call_started(clean_dev, state)
            if state in ("IDLE", "MISSED"):
                return await self.handle_call_ended(clean_dev)
            return False

        # Default policy: pause on ringing or talking
        if state in ("RINGING", "TALKING"):
            return await self.handle_call_started(clean_dev, state)
        if state in ("IDLE", "MISSED"):
            return await self.handle_call_ended(clean_dev)
        return False

    async def handle_call_started(self, device_id: str, state: str = "RINGING") -> bool:
        """Register an active call for device_id and apply pause/mute policies on 0 -> 1."""
        clean_dev = str(device_id).strip()
        if not clean_dev:
            return False

        async with self._lock:
            if clean_dev in self._active_call_devices:
                # Already tracked as active interruption for this device
                return False

            was_interrupted = len(self._active_call_devices) > 0
            self._active_call_devices.add(clean_dev)

            # Strict N -> N+1 Transition: DO NOT re-snapshot!
            if was_interrupted:
                logger.debug(
                    "Device %s joined active interruption. Total active call devices: %d",
                    clean_dev,
                    len(self._active_call_devices),
                )
                return False

            # Strict 0 -> 1 Transition: First interruption begins
            logger.info(
                "Call interruption started (device: %s, state: %s). Capturing pre-call snapshot.",
                clean_dev,
                state,
            )

        # LOCK RELEASED BEFORE EXTERNAL QUERIES AND ACTIONS

        # 1. Snapshot & Pause MPRIS media players
        if self.config.action_pause and self._mpris_adapter:
            try:
                players = await self._mpris_adapter.list_players()
                for p in players:
                    # Reflection suppression
                    if (
                        any(p.startswith(prefix) for prefix in REFLECTION_PREFIXES)
                        or p in REFLECTION_EXACT
                    ):
                        logger.debug("Skipping reflection player: %s", p)
                        continue
                    try:
                        status = await self._mpris_adapter.get_player_status(p)
                        if status and status.is_playing:
                            can_pause = getattr(status, "can_pause", True)
                            can_stop = getattr(status, "can_stop", True)
                            act = "Stop" if (not can_pause and can_stop) else "Pause"

                            # Short lock check: verify interruption is still active before pausing
                            async with self._lock:
                                if len(self._active_call_devices) == 0:
                                    break

                            await self._mpris_adapter.send_action(p, act)
                            logger.debug("Paused media player: %s", p)

                            # Record paused player under short lock
                            must_resume_orphaned = False
                            async with self._lock:
                                if len(self._active_call_devices) > 0:
                                    self._paused_players.add(p)
                                else:
                                    must_resume_orphaned = True

                            # If all calls ended while send_action was in flight, immediately restore
                            if must_resume_orphaned and self.config.action_resume:
                                try:
                                    await self._mpris_adapter.send_action(p, "Play")
                                except COORDINATION_EXCEPTIONS as resume_ex:
                                    logger.warning("Failed to resume orphaned player %s: %s", p, resume_ex)

                    except COORDINATION_EXCEPTIONS as ex:
                        logger.warning("Failed to query/pause player %s: %s", p, ex)
            except COORDINATION_EXCEPTIONS as ex:
                logger.warning("Error querying MPRIS player list: %s", ex)

        # 2. Snapshot & Mute Audio Sinks
        if self.config.action_mute and self._audio_service:
            try:
                sinks_res = self._audio_service.list_sinks()
                sinks = await sinks_res if inspect.isawaitable(sinks_res) else sinks_res
                for sink in sinks:
                    if not sink.muted:
                        # Short lock check: verify interruption is still active before muting
                        async with self._lock:
                            if len(self._active_call_devices) == 0:
                                break

                        await self._audio_service.set_muted(sink.name, True)
                        logger.debug("Muted audio sink: %s", sink.name)

                        # Record muted sink under short lock
                        must_unmute_orphaned = False
                        async with self._lock:
                            if len(self._active_call_devices) > 0:
                                self._muted_sinks.add(sink.name)
                            else:
                                must_unmute_orphaned = True

                        # If all calls ended while set_muted was in flight, immediately unmute
                        if must_unmute_orphaned and self.config.action_resume:
                            try:
                                await self._audio_service.set_muted(sink.name, False)
                            except COORDINATION_EXCEPTIONS as unmute_ex:
                                logger.warning("Failed to unmute orphaned sink %s: %s", sink.name, unmute_ex)

            except COORDINATION_EXCEPTIONS as ex:
                logger.warning("Error querying/muting audio sinks: %s", ex)

        return True

    async def handle_call_ended(self, device_id: str) -> bool:
        """Remove active call for device_id and resume/unmute on 1 -> 0."""
        clean_dev = str(device_id).strip()
        if not clean_dev:
            return False

        to_resume_players: list[str] = []
        to_unmute_sinks: list[str] = []
        should_resume: bool = False

        async with self._lock:
            if clean_dev not in self._active_call_devices:
                return False

            self._active_call_devices.remove(clean_dev)

            # Strict N -> N-1 Transition (N > 1): Other active calls remain
            if len(self._active_call_devices) > 0:
                logger.debug(
                    "Call ended for device %s, but %d active devices remain. Keeping media paused.",
                    clean_dev,
                    len(self._active_call_devices),
                )
                return False

            # Strict 1 -> 0 Transition: All active calls have concluded
            logger.info("All call interruptions ended. Evaluating auto-resume policy.")
            should_resume = bool(self.config.action_resume)
            if should_resume:
                # Safely collect all previously paused players and muted sinks,
                # even if action_pause or action_mute was modified mid-call.
                to_resume_players = list(self._paused_players)
                to_unmute_sinks = list(self._muted_sinks)

            # Clear snapshots under lock regardless of action_resume setting
            self._paused_players.clear()
            self._muted_sinks.clear()

        # LOCK RELEASED BEFORE EXTERNAL PLAY AND UNMUTE CALLS
        if should_resume:
            # 1. Resume strictly pre-call playing MPRIS players
            if self._mpris_adapter and to_resume_players:
                for p in to_resume_players:
                    try:
                        await self._mpris_adapter.send_action(p, "Play")
                        logger.debug("Resumed media player: %s", p)
                    except COORDINATION_EXCEPTIONS as ex:
                        logger.warning("Failed to resume player %s: %s", p, ex)

            # 2. Unmute strictly pre-call unmuted audio sinks
            if self._audio_service and to_unmute_sinks:
                for s_name in to_unmute_sinks:
                    try:
                        await self._audio_service.set_muted(s_name, False)
                        logger.debug("Unmuted audio sink: %s", s_name)
                    except COORDINATION_EXCEPTIONS as ex:
                        logger.warning("Failed to unmute sink %s: %s", s_name, ex)

        return True

    async def handle_device_disconnected(self, device_id: str) -> bool:
        """Clear active call interruption for disconnected or unpaired device."""
        return await self.handle_call_ended(device_id)

    async def clear_device(self, device_id: str) -> bool:
        """Alias for handle_device_disconnected."""
        return await self.handle_device_disconnected(device_id)

    async def restore_media(self) -> None:
        """Manually restore media playback and unmute sinks, clearing active interruption."""
        to_resume_players: list[str] = []
        to_unmute_sinks: list[str] = []

        async with self._lock:
            to_resume_players = list(self._paused_players)
            to_unmute_sinks = list(self._muted_sinks)
            self._active_call_devices.clear()
            self._paused_players.clear()
            self._muted_sinks.clear()

        # LOCK RELEASED BEFORE EXTERNAL PLAY AND UNMUTE CALLS
        if self._mpris_adapter and to_resume_players:
            for p in to_resume_players:
                try:
                    await self._mpris_adapter.send_action(p, "Play")
                except COORDINATION_EXCEPTIONS as ex:
                    logger.debug("Error resuming player %s in restore_media: %s", p, ex)

        if self._audio_service and to_unmute_sinks:
            for s in to_unmute_sinks:
                try:
                    await self._audio_service.set_muted(s, False)
                except COORDINATION_EXCEPTIONS as ex:
                    logger.debug("Error unmuting sink %s in restore_media: %s", s, ex)

    async def clear_all(self) -> None:
        """Alias for restore_media."""
        await self.restore_media()

    async def reset(self) -> None:
        """Alias for restore_media."""
        await self.restore_media()

    async def resume_all(self) -> None:
        """Alias for restore_media."""
        await self.restore_media()

    def get_status(self) -> dict[str, Any]:
        """Return runtime status snapshot for IPC and CLI."""
        return {
            "config": self.config.to_dict(),
            "is_interrupted": len(self._active_call_devices) > 0,
            "active_call_devices": list(self._active_call_devices),
            "paused_players": list(self._paused_players),
            "muted_sinks": list(self._muted_sinks),
        }

    def update_config(self, **kwargs: Any) -> PauseMusicConfig:
        """Update configuration dynamically."""
        if "condition" in kwargs and kwargs["condition"] is not None:
            self.config.condition = str(kwargs["condition"])
        if "condition_talking" in kwargs and kwargs["condition_talking"] is not None:
            self.config.condition_talking = _coerce_bool(kwargs["condition_talking"])
        if "conditionTalking" in kwargs and kwargs["conditionTalking"] is not None:
            self.config.condition_talking = _coerce_bool(kwargs["conditionTalking"])

        if "action" in kwargs and kwargs["action"] is not None:
            self.config.action = str(kwargs["action"])
        if "action_pause" in kwargs and kwargs["action_pause"] is not None:
            self.config.action_pause = _coerce_bool(kwargs["action_pause"])
        if "actionPause" in kwargs and kwargs["actionPause"] is not None:
            self.config.action_pause = _coerce_bool(kwargs["actionPause"])
        if "action_mute" in kwargs and kwargs["action_mute"] is not None:
            self.config.action_mute = _coerce_bool(kwargs["action_mute"])
        if "actionMute" in kwargs and kwargs["actionMute"] is not None:
            self.config.action_mute = _coerce_bool(kwargs["actionMute"])

        if "action_resume" in kwargs and kwargs["action_resume"] is not None:
            self.config.action_resume = _coerce_bool(kwargs["action_resume"])
        if "actionResume" in kwargs and kwargs["actionResume"] is not None:
            self.config.action_resume = _coerce_bool(kwargs["actionResume"])
        if "resume" in kwargs and kwargs["resume"] is not None:
            self.config.action_resume = _coerce_bool(kwargs["resume"])

        return self.config

    async def stop(self) -> None:
        """Clean up EventBus subscriptions and restore media if interrupted."""
        self._unsubscribe_events()
        if self._paused_players or self._muted_sinks:
            await self.restore_media()


__all__ = [
    "REFLECTION_EXACT",
    "REFLECTION_PREFIXES",
    "PauseMusicConfig",
    "PauseMusicService",
]

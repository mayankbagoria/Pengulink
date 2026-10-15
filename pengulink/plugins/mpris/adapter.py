"""pengulink.plugins.mpris.adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MPRIS2 Host Adapter Subsystem.

Provides an abstract interface (MprisHostAdapter) and implementations:
- MockMprisAdapter: Deterministic in-memory media player simulator for headless CI/tests.
- DBusMprisAdapter: Desktop D-Bus session bus integration with loop & echo suppression.
- detect_mpris_adapter: Auto-detection factory with environment override.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
import inspect
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from pengulink.core.exceptions import (
    MprisActionError,
    MprisAdapterError,
    MprisAdapterUnavailableError,
    MprisError,
    MprisPlayerNotFoundError,
)

logger = logging.getLogger("pengulink.plugins.mpris.adapter")


# =====================================================================
# PlayerState Data Model
# =====================================================================

@dataclass
class PlayerState:
    """Internal normalized representation of a host media player state."""

    player: str
    title: str = ""
    artist: str = ""
    album: str = ""
    album_art_url: str = ""
    is_playing: bool = False
    position_us: int = 0         # Microseconds (internal precision)
    length_us: int = -1          # Microseconds (-1 if unknown)
    volume: float = 1.0          # 0.0 .. 1.0 (MPRIS2 standard)
    can_play: bool = True
    can_pause: bool = True
    can_go_next: bool = True
    can_go_previous: bool = True
    can_seek: bool = True
    loop_status: str = "None"    # "None" | "Track" | "Playlist"
    shuffle: bool = False
    url: str = ""
    last_updated: float = field(default_factory=time.time)

    def get_current_position_us(self, now: Optional[float] = None) -> int:
        """Compute real-time track position in microseconds considering playback state."""
        if not self.is_playing:
            return max(0, self.position_us)
        curr_time = now if now is not None else time.time()
        elapsed_us = int((curr_time - self.last_updated) * 1_000_000)
        pos = self.position_us + elapsed_us
        if self.length_us > 0:
            pos = min(pos, self.length_us)
        return max(0, pos)

    def to_wire_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """Convert internal state to KDE Connect kdeconnect.mpris wire dictionary."""
        pos_ms = max(0, self.get_current_position_us() // 1000)
        length_ms = self.length_us // 1000 if self.length_us >= 0 else -1
        vol_pct = max(0, min(100, int(round(self.volume * 100))))

        wire: Dict[str, Any] = {
            "player": self.player,
            "isPlaying": bool(self.is_playing),
            "pos": pos_ms,
            "length": length_ms,
            "volume": vol_pct,
            "canPlay": bool(self.can_play),
            "canPause": bool(self.can_pause),
            "canGoNext": bool(self.can_go_next),
            "canGoPrevious": bool(self.can_go_previous),
            "canSeek": bool(self.can_seek),
            "loopStatus": self.loop_status,
            "shuffle": bool(self.shuffle),
        }

        if include_metadata:
            wire["title"] = self.title
            wire["artist"] = self.artist
            wire["album"] = self.album
            wire["albumArtUrl"] = self.album_art_url
            if self.artist and self.title:
                wire["nowPlaying"] = f"{self.artist} - {self.title}"
            elif self.title or self.artist:
                wire["nowPlaying"] = self.title or self.artist
            else:
                wire["nowPlaying"] = ""
            if self.url:
                wire["url"] = self.url

        return wire


# =====================================================================
# MprisHostAdapter Abstract Base Class
# =====================================================================

class MprisHostAdapter(ABC):
    """Abstract interface to host MPRIS2 media players."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifying name of the adapter ('mock', 'dbus')."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Return True if required environment and tools are operational."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Lifecycle hook: initialize listeners, connections, or event monitors."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Lifecycle hook: clean up resources and cancel tasks."""
        ...

    @abstractmethod
    async def list_players(self) -> List[str]:
        """Enumerate active player names currently available."""
        ...

    @abstractmethod
    async def get_player_status(self, player: str) -> Optional[PlayerState]:
        """Query state and metadata for a specific player."""
        ...

    @abstractmethod
    async def send_action(self, player: str, action: str) -> bool:
        """Execute a transport action ('Play', 'Pause', 'PlayPause', 'Stop', 'Next', 'Previous')."""
        ...

    @abstractmethod
    async def set_volume(self, player: str, volume: float) -> bool:
        """Set player volume as a float between 0.0 and 1.0."""
        ...

    @abstractmethod
    async def seek(self, player: str, offset_us: int) -> bool:
        """Relative seek forward or backward in microseconds."""
        ...

    @abstractmethod
    async def set_position(self, player: str, position_us: int) -> bool:
        """Set absolute track position in microseconds."""
        ...

    @abstractmethod
    def set_change_callback(
        self,
        callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]],
    ) -> None:
        """Register async callback invoked when player state or player list changes."""
        ...

    async def set_loop_status(self, player: str, loop_status: str) -> bool:
        """Set repeat/loop mode ('None', 'Track', 'Playlist'). Default returns False."""
        return False

    async def set_shuffle(self, player: str, shuffle: bool) -> bool:
        """Enable or disable shuffle. Default returns False."""
        return False


# =====================================================================
# MockMprisAdapter Implementation
# =====================================================================

class MockMprisAdapter(MprisHostAdapter):
    """In-memory mock adapter for CI/Docker testing without D-Bus."""

    def __init__(self, auto_default_player: bool = True) -> None:
        self._players: Dict[str, PlayerState] = {}
        self._callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None
        self._running: bool = False
        self._lock = asyncio.Lock()
        self._auto_default_player = auto_default_player

    @property
    def name(self) -> str:
        return "mock"

    @property
    def is_available(self) -> bool:
        return True

    async def start(self) -> None:
        self._running = True
        if self._auto_default_player and not self._players:
            self._players["MockPlayer"] = PlayerState(
                player="MockPlayer",
                title="Mock Symphony No. 9",
                artist="Synthetic Orchestra",
                album="Unit Test Anthems",
                length_us=180_000_000,  # 180 seconds in microseconds
                volume=0.75,
                is_playing=False,
            )

    async def stop(self) -> None:
        self._running = False

    async def list_players(self) -> List[str]:
        async with self._lock:
            return list(self._players.keys())

    async def get_player_status(self, player: str) -> Optional[PlayerState]:
        async with self._lock:
            st = self._players.get(player)
            if not st:
                return None
            res = copy.copy(st)
            res.position_us = st.get_current_position_us()
            return res

    async def send_action(self, player: str, action: str) -> bool:
        async with self._lock:
            st = self._players.get(player)
            if not st:
                return False

            now = time.time()
            act = action.strip()
            if act == "Play":
                st.is_playing = True
                st.last_updated = now
            elif act == "Pause":
                st.position_us = st.get_current_position_us(now)
                st.is_playing = False
                st.last_updated = now
            elif act == "PlayPause":
                if st.is_playing:
                    st.position_us = st.get_current_position_us(now)
                    st.is_playing = False
                else:
                    st.is_playing = True
                st.last_updated = now
            elif act == "Stop":
                st.is_playing = False
                st.position_us = 0
                st.last_updated = now
            elif act == "Next":
                st.title = f"Track {int(now) % 100 + 1}"
                st.position_us = 0
                st.last_updated = now
            elif act == "Previous":
                st.position_us = 0
                st.last_updated = now
            else:
                return False

            wire = st.to_wire_dict()

        await self._dispatch_change(player, wire)
        return True

    async def set_volume(self, player: str, volume: float) -> bool:
        async with self._lock:
            st = self._players.get(player)
            if not st:
                return False
            clamped = max(0.0, min(1.0, float(volume)))
            st.volume = clamped
            vol_pct = int(round(clamped * 100))
            delta = {"player": player, "volume": vol_pct}

        await self._dispatch_change(player, delta)
        return True

    async def seek(self, player: str, offset_us: int) -> bool:
        async with self._lock:
            st = self._players.get(player)
            if not st:
                return False
            now = time.time()
            curr = st.get_current_position_us(now)
            target = max(0, curr + int(offset_us))
            if st.length_us > 0:
                target = min(target, st.length_us)
            st.position_us = target
            st.last_updated = now
            delta = {"player": player, "pos": target // 1000}

        await self._dispatch_change(player, delta)
        return True

    async def set_position(self, player: str, position_us: int) -> bool:
        async with self._lock:
            st = self._players.get(player)
            if not st:
                return False
            now = time.time()
            target = max(0, int(position_us))
            if st.length_us > 0:
                target = min(target, st.length_us)
            st.position_us = target
            st.last_updated = now
            delta = {"player": player, "pos": target // 1000}

        await self._dispatch_change(player, delta)
        return True

    async def set_loop_status(self, player: str, loop_status: str) -> bool:
        async with self._lock:
            st = self._players.get(player)
            if not st:
                return False
            valid = {"None", "Track", "Playlist"}
            status = loop_status if loop_status in valid else "None"
            st.loop_status = status
            delta = {"player": player, "loopStatus": status}

        await self._dispatch_change(player, delta)
        return True

    async def set_shuffle(self, player: str, shuffle: bool) -> bool:
        async with self._lock:
            st = self._players.get(player)
            if not st:
                return False
            st.shuffle = bool(shuffle)
            delta = {"player": player, "shuffle": bool(shuffle)}

        await self._dispatch_change(player, delta)
        return True

    def set_change_callback(
        self,
        callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]],
    ) -> None:
        self._callback = callback

    # Test Helper APIs
    def add_mock_player(self, player: str, state: Optional[PlayerState] = None) -> None:
        """Add or replace a mock player and trigger playerList notification."""
        self._players[player] = state or PlayerState(player=player)
        self._schedule_notification("", {"playerList": list(self._players.keys())})

    def remove_mock_player(self, player: str) -> bool:
        """Remove a mock player and trigger playerList notification."""
        removed = self._players.pop(player, None) is not None
        if removed:
            self._schedule_notification("", {"playerList": list(self._players.keys())})
        return removed

    def clear_mock_players(self) -> None:
        """Clear all mock players and trigger playerList notification."""
        self._players.clear()
        self._schedule_notification("", {"playerList": []})

    async def _dispatch_change(self, player: str, delta: Dict[str, Any]) -> None:
        if self._callback and self._running:
            try:
                res = self._callback(player, delta)
                if inspect.isawaitable(res):
                    await res
            except Exception as exc:
                logger.error("Error in MPRIS change callback: %s", exc, exc_info=True)

    def _schedule_notification(self, player: str, delta: Dict[str, Any]) -> None:
        if self._callback and self._running:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._dispatch_change(player, delta))
            except RuntimeError:
                pass


# =====================================================================
# DBusMprisAdapter Implementation with Echo Suppression
# =====================================================================

class DBusMprisAdapter(MprisHostAdapter):
    """Session D-Bus MPRIS2 adapter with echo suppression."""

    # Echo suppression blacklist: do NOT export virtual remote players or playerctld
    IGNORE_PREFIXES = (
        "org.mpris.MediaPlayer2.kdeconnect.",
        "org.mpris.MediaPlayer2.pengulink.",
    )
    IGNORE_EXACT = {
        "org.mpris.MediaPlayer2.playerctld",
    }

    def __init__(self) -> None:
        self._callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None
        self._running: bool = False
        self._bus: Any = None
        self._check_availability()

    @property
    def name(self) -> str:
        return "dbus"

    @property
    def is_available(self) -> bool:
        return self._available

    def _check_availability(self) -> None:
        """Check if D-Bus environment and Python bindings are operational."""
        bus_addr = os.environ.get("DBUS_SESSION_BUS_ADDRESS", "").strip()
        if not bus_addr:
            self._available = False
            return

        try:
            import dbus  # type: ignore[import-untyped]
            self._available = True
        except ImportError:
            self._available = False

    @classmethod
    def should_ignore_service(cls, bus_name: str) -> bool:
        """Check if D-Bus bus name should be ignored to prevent feedback loops."""
        if bus_name in cls.IGNORE_EXACT:
            return True
        for prefix in cls.IGNORE_PREFIXES:
            if bus_name.startswith(prefix):
                return True
        return False

    async def start(self) -> None:
        if not self.is_available:
            raise MprisAdapterUnavailableError("D-Bus session bus is unavailable on host")
        self._running = True
        try:
            import dbus  # type: ignore[import-untyped]
            self._bus = dbus.SessionBus()
        except Exception as exc:
            self._running = False
            raise MprisAdapterUnavailableError(f"Failed to connect to D-Bus SessionBus: {exc}") from exc

    async def stop(self) -> None:
        self._running = False
        self._bus = None

    async def list_players(self) -> List[str]:
        if not self._running or not self._bus:
            return []
        try:
            import dbus  # type: ignore[import-untyped]
            dbus_obj = self._bus.get_object("org.freedesktop.DBus", "/org/freedesktop/DBus")
            dbus_iface = dbus.Interface(dbus_obj, "org.freedesktop.DBus")
            services = dbus_iface.ListNames()
            players = []
            for s in services:
                name_str = str(s)
                if name_str.startswith("org.mpris.MediaPlayer2.") and not self.should_ignore_service(name_str):
                    # Strip prefix to get human-readable player name
                    clean_name = name_str[len("org.mpris.MediaPlayer2."):]
                    players.append(clean_name)
            return sorted(players)
        except Exception as exc:
            logger.warning("Error listing D-Bus MPRIS players: %s", exc)
            return []

    async def get_player_status(self, player: str) -> Optional[PlayerState]:
        if not self._running or not self._bus:
            return None
        bus_name = f"org.mpris.MediaPlayer2.{player}"
        if self.should_ignore_service(bus_name):
            return None
        try:
            import dbus  # type: ignore[import-untyped]
            player_obj = self._bus.get_object(bus_name, "/org/mpris/MediaPlayer2")
            props_iface = dbus.Interface(player_obj, "org.freedesktop.DBus.Properties")

            playback_status = str(props_iface.Get("org.mpris.MediaPlayer2.Player", "PlaybackStatus"))
            is_playing = (playback_status.lower() == "playing")

            metadata = props_iface.Get("org.mpris.MediaPlayer2.Player", "Metadata")
            title = str(metadata.get("xesam:title", ""))
            artist_raw = metadata.get("xesam:artist", [])
            artist = ", ".join(str(a) for a in artist_raw) if isinstance(artist_raw, (list, tuple)) else str(artist_raw or "")
            album = str(metadata.get("xesam:album", ""))
            art_url = str(metadata.get("mpris:artUrl", ""))
            length_us = int(metadata.get("mpris:length", -1))
            url = str(metadata.get("xesam:url", ""))

            pos_us = int(props_iface.Get("org.mpris.MediaPlayer2.Player", "Position"))
            volume = float(props_iface.Get("org.mpris.MediaPlayer2.Player", "Volume"))

            can_play = bool(props_iface.Get("org.mpris.MediaPlayer2.Player", "CanPlay"))
            can_pause = bool(props_iface.Get("org.mpris.MediaPlayer2.Player", "CanPause"))
            can_go_next = bool(props_iface.Get("org.mpris.MediaPlayer2.Player", "CanGoNext"))
            can_go_prev = bool(props_iface.Get("org.mpris.MediaPlayer2.Player", "CanGoPrevious"))
            can_seek = bool(props_iface.Get("org.mpris.MediaPlayer2.Player", "CanSeek"))

            loop_status = str(props_iface.Get("org.mpris.MediaPlayer2.Player", "LoopStatus") or "None")
            shuffle = bool(props_iface.Get("org.mpris.MediaPlayer2.Player", "Shuffle") or False)

            return PlayerState(
                player=player,
                title=title,
                artist=artist,
                album=album,
                album_art_url=art_url,
                is_playing=is_playing,
                position_us=pos_us,
                length_us=length_us,
                volume=volume,
                can_play=can_play,
                can_pause=can_pause,
                can_go_next=can_go_next,
                can_go_previous=can_go_prev,
                can_seek=can_seek,
                loop_status=loop_status,
                shuffle=shuffle,
                url=url,
            )
        except Exception as exc:
            logger.debug("Could not get player status for '%s': %s", player, exc)
            return None

    async def send_action(self, player: str, action: str) -> bool:
        if not self._running or not self._bus:
            return False
        bus_name = f"org.mpris.MediaPlayer2.{player}"
        if self.should_ignore_service(bus_name):
            return False
        try:
            import dbus  # type: ignore[import-untyped]
            player_obj = self._bus.get_object(bus_name, "/org/mpris/MediaPlayer2")
            player_iface = dbus.Interface(player_obj, "org.mpris.MediaPlayer2.Player")

            action_map = {
                "Play": "Play",
                "Pause": "Pause",
                "PlayPause": "PlayPause",
                "Stop": "Stop",
                "Next": "Next",
                "Previous": "Previous",
            }
            method_name = action_map.get(action)
            if not method_name:
                return False
            method = getattr(player_iface, method_name)
            method()
            return True
        except Exception as exc:
            logger.warning("Failed to invoke action '%s' on player '%s': %s", action, player, exc)
            return False

    async def set_volume(self, player: str, volume: float) -> bool:
        if not self._running or not self._bus:
            return False
        bus_name = f"org.mpris.MediaPlayer2.{player}"
        if self.should_ignore_service(bus_name):
            return False
        try:
            import dbus  # type: ignore[import-untyped]
            player_obj = self._bus.get_object(bus_name, "/org/mpris/MediaPlayer2")
            props_iface = dbus.Interface(player_obj, "org.freedesktop.DBus.Properties")
            clamped = max(0.0, min(1.0, float(volume)))
            props_iface.Set("org.mpris.MediaPlayer2.Player", "Volume", dbus.Double(clamped))
            return True
        except Exception as exc:
            logger.warning("Failed to set volume on player '%s': %s", player, exc)
            return False

    async def seek(self, player: str, offset_us: int) -> bool:
        if not self._running or not self._bus:
            return False
        bus_name = f"org.mpris.MediaPlayer2.{player}"
        if self.should_ignore_service(bus_name):
            return False
        try:
            import dbus  # type: ignore[import-untyped]
            player_obj = self._bus.get_object(bus_name, "/org/mpris/MediaPlayer2")
            player_iface = dbus.Interface(player_obj, "org.mpris.MediaPlayer2.Player")
            player_iface.Seek(dbus.Int64(offset_us))
            return True
        except Exception as exc:
            logger.warning("Failed to seek on player '%s': %s", player, exc)
            return False

    async def set_position(self, player: str, position_us: int) -> bool:
        if not self._running or not self._bus:
            return False
        bus_name = f"org.mpris.MediaPlayer2.{player}"
        if self.should_ignore_service(bus_name):
            return False
        try:
            import dbus  # type: ignore[import-untyped]
            player_obj = self._bus.get_object(bus_name, "/org/mpris/MediaPlayer2")
            player_iface = dbus.Interface(player_obj, "org.mpris.MediaPlayer2.Player")
            track_id = "/org/mpris/MediaPlayer2/TrackList/NoTrack"
            try:
                props_iface = dbus.Interface(player_obj, "org.freedesktop.DBus.Properties")
                meta = props_iface.Get("org.mpris.MediaPlayer2.Player", "Metadata")
                track_id = meta.get("mpris:trackid", track_id)
            except Exception:
                pass
            player_iface.SetPosition(track_id, dbus.Int64(position_us))
            return True
        except Exception as exc:
            logger.warning("Failed to set position on player '%s': %s", player, exc)
            return False

    def set_change_callback(
        self,
        callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]],
    ) -> None:
        self._callback = callback


# =====================================================================
# Factory / Auto-Detection
# =====================================================================

def detect_mpris_adapter(adapter_name: Optional[str] = None) -> MprisHostAdapter:
    """Detect and instantiate the appropriate MPRIS host adapter.

    Resolution:
    1. Explicit adapter_name argument.
    2. PENGULINK_MPRIS_ADAPTER environment variable ('mock' | 'dbus').
    3. Auto-detection cascade:
       - If DBUS_SESSION_BUS_ADDRESS is present and DBusMprisAdapter is available -> DBusMprisAdapter()
       - Fallback -> MockMprisAdapter()
    """
    req = (adapter_name or os.environ.get("PENGULINK_MPRIS_ADAPTER", "")).strip().lower()

    if req:
        if req in ("mock", "in_memory", "headless"):
            return MockMprisAdapter()
        elif req == "dbus":
            dbus_adapter = DBusMprisAdapter()
            if not dbus_adapter.is_available:
                raise MprisAdapterUnavailableError("D-Bus session bus is not available on host")
            return dbus_adapter
        else:
            raise MprisAdapterError(f"Unknown MPRIS adapter requested: '{req}'")

    # Auto-detection cascade
    dbus_adapter = DBusMprisAdapter()
    if dbus_adapter.is_available:
        logger.info("Auto-detected active desktop D-Bus MPRIS adapter")
        return dbus_adapter

    logger.info("No active D-Bus session detected; falling back to MockMprisAdapter")
    return MockMprisAdapter()


__all__ = [
    "PlayerState",
    "MprisHostAdapter",
    "MockMprisAdapter",
    "DBusMprisAdapter",
    "detect_mpris_adapter",
]

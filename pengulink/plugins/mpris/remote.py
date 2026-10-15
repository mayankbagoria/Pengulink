"""pengulink.plugins.mpris.remote
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect MprisRemote Client Plugin.

Client role: Ingests remote media player updates (kdeconnect.mpris) from connected
peers, maintains per-device player states with real-time position interpolation
and incremental state merging, and transmits transport control requests (kdeconnect.mpris.request).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from pengulink.core.events import (
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
    RemotePlayerListUpdatedEvent,
    RemotePlayerUpdatedEvent,
)
from pengulink.core.exceptions import (
    ActionUnsupportedError,
    ConnectionClosedError,
    MprisPlayerNotFoundError,
)
from pengulink.plugins.base import Packet, Plugin, SendPacketCallable, safe_send_packet

logger = logging.getLogger("pengulink.plugins.mpris.remote")

ALLOWED_ACTIONS = {"Play", "Pause", "PlayPause", "Stop", "Next", "Previous"}


# =====================================================================
# RemotePlayerState Data Model
# =====================================================================

@dataclass
class RemotePlayerState:
    """Cached playback state for an individual media player on a remote device."""

    player: str                                  # Unique player name (e.g., "Spotify")
    title: str = ""                              # Track title
    artist: str = ""                             # Artist name
    album: str = ""                              # Album title
    albumArtUrl: str = ""                        # Cover art URI (http/https/file)
    isPlaying: bool = False                      # Playback status
    pos: int = 0                                 # Track position in ms at last_pos_timestamp
    length: int = -1                             # Total track duration in ms (-1 if unknown)
    volume: int = 100                            # Volume percentage (0..100)
    canPlay: bool = True                         # Capability flag
    canPause: bool = True                        # Capability flag
    canGoNext: bool = True                       # Capability flag
    canGoPrevious: bool = True                   # Capability flag
    canSeek: bool = True                         # Capability flag
    loopStatus: str = "None"                     # "None" | "Track" | "Playlist"
    shuffle: bool = False                        # Shuffle enabled
    last_pos_timestamp: float = field(default_factory=time.time)

    def get_current_position_ms(self, now: Optional[float] = None) -> int:
        """Calculate dynamic wall-clock position in milliseconds.

        When isPlaying is True:
            pos_now = last_pos + (now - last_pos_timestamp) * 1000
        When paused or stopped:
            pos_now = last_pos
        Clamped to [0, length] when length > 0, or [0, inf) if length is unknown.
        """
        curr_time = now if now is not None else time.time()
        if self.isPlaying:
            elapsed_ms = int((curr_time - self.last_pos_timestamp) * 1000)
            pos_now = self.pos + elapsed_ms
        else:
            pos_now = self.pos

        if self.length > 0:
            return max(0, min(pos_now, self.length))
        return max(0, pos_now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for JSON-RPC IPC transmission."""
        return {
            "player": self.player,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "albumArtUrl": self.albumArtUrl,
            "isPlaying": self.isPlaying,
            "pos": self.get_current_position_ms(),
            "raw_pos": self.pos,
            "length": self.length,
            "volume": self.volume,
            "canPlay": self.canPlay,
            "canPause": self.canPause,
            "canGoNext": self.canGoNext,
            "canGoPrevious": self.canGoPrevious,
            "canSeek": self.canSeek,
            "loopStatus": self.loopStatus,
            "shuffle": self.shuffle,
            "last_pos_timestamp": self.last_pos_timestamp,
        }


# =====================================================================
# MprisRemote Client Plugin Implementation
# =====================================================================

class MprisRemote(Plugin):
    """KDE Connect MprisRemote Client Plugin.

    Client role: Receives kdeconnect.mpris packets from connected peers,
    tracks per-device remote player state, interpolates live track position,
    and sends transport requests via kdeconnect.mpris.request.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        packet_sender: Optional[SendPacketCallable] = None,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self.packet_sender = packet_sender
        self._device_players: Dict[str, Dict[str, RemotePlayerState]] = {}
        self._device_player_lists: Dict[str, List[str]] = {}
        self._device_active_player: Dict[str, Optional[str]] = {}
        self._connected_senders: Dict[str, SendPacketCallable] = {}
        self._unsub_callbacks: List[Callable[[], None]] = []

    @property
    def plugin_id(self) -> str:
        return "mprisremote"

    @property
    def incoming_capabilities(self) -> List[str]:
        return ["kdeconnect.mpris"]

    @property
    def outgoing_capabilities(self) -> List[str]:
        return ["kdeconnect.mpris.request"]

    async def start(self) -> None:
        """Subscribe to device lifecycle events."""
        if self.event_bus:
            self._unsub_callbacks.append(
                self.event_bus.subscribe(DeviceDisconnected, self._on_device_disconnected)
            )
            self._unsub_callbacks.append(
                self.event_bus.subscribe(DeviceUnpaired, self._on_device_unpaired)
            )

    async def stop(self) -> None:
        """Unsubscribe from lifecycle events and purge memory caches."""
        for unsub in self._unsub_callbacks:
            try:
                unsub()
            except Exception:
                pass
        self._unsub_callbacks.clear()
        self._connected_senders.clear()
        self._device_players.clear()
        self._device_player_lists.clear()
        self._device_active_player.clear()

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Ingest incoming kdeconnect.mpris packet and update cache."""
        self._connected_senders[device_id] = send_packet
        body = packet.body or {}

        # 1. Handle playerList packet
        if "playerList" in body:
            raw_list = [str(p) for p in body.get("playerList", [])]
            self._device_player_lists[device_id] = raw_list

            # Prune cached players that no longer exist
            dev_players = self._device_players.setdefault(device_id, {})
            pruned_names = [name for name in dev_players if name not in raw_list]
            for name in pruned_names:
                dev_players.pop(name, None)

            # Update active player if pruned or unset
            active = self._device_active_player.get(device_id)
            if not active or active not in raw_list:
                self._device_active_player[device_id] = raw_list[0] if raw_list else None

            if self.event_bus:
                await self.event_bus.publish(
                    RemotePlayerListUpdatedEvent(
                        device_id=device_id,
                        player_list=list(raw_list),
                    )
                )

        # 2. Handle player state packet (full or incremental delta)
        if "player" in body:
            player_name = str(body["player"])
            dev_players = self._device_players.setdefault(device_id, {})
            now = time.time()

            if player_name in dev_players:
                st = dev_players[player_name]
                # Position & playback state handling
                if "pos" in body:
                    new_pos = int(body["pos"])
                    new_ts = now
                elif "isPlaying" in body and bool(body["isPlaying"]) != st.isPlaying:
                    # Freeze interpolated position on state change if pos omitted
                    new_pos = st.get_current_position_ms(now)
                    new_ts = now
                else:
                    new_pos = st.pos
                    new_ts = st.last_pos_timestamp

                st.pos = new_pos
                st.last_pos_timestamp = new_ts

                if "title" in body:
                    st.title = str(body["title"])
                if "artist" in body:
                    st.artist = str(body["artist"])
                if "album" in body:
                    st.album = str(body["album"])
                if "albumArtUrl" in body:
                    st.albumArtUrl = str(body["albumArtUrl"])
                if "isPlaying" in body:
                    st.isPlaying = bool(body["isPlaying"])
                if "length" in body:
                    st.length = int(body["length"])
                if "volume" in body:
                    st.volume = max(0, min(100, int(body["volume"])))
                if "canPlay" in body:
                    st.canPlay = bool(body["canPlay"])
                if "canPause" in body:
                    st.canPause = bool(body["canPause"])
                if "canGoNext" in body:
                    st.canGoNext = bool(body["canGoNext"])
                if "canGoPrevious" in body:
                    st.canGoPrevious = bool(body["canGoPrevious"])
                if "canSeek" in body:
                    st.canSeek = bool(body["canSeek"])
                if "loopStatus" in body:
                    st.loopStatus = str(body["loopStatus"])
                if "shuffle" in body:
                    st.shuffle = bool(body["shuffle"])
            else:
                st = RemotePlayerState(
                    player=player_name,
                    title=str(body.get("title", "")),
                    artist=str(body.get("artist", "")),
                    album=str(body.get("album", "")),
                    albumArtUrl=str(body.get("albumArtUrl", "")),
                    isPlaying=bool(body.get("isPlaying", False)),
                    pos=int(body.get("pos", 0)),
                    length=int(body.get("length", -1)),
                    volume=max(0, min(100, int(body.get("volume", 100)))),
                    canPlay=bool(body.get("canPlay", True)),
                    canPause=bool(body.get("canPause", True)),
                    canGoNext=bool(body.get("canGoNext", True)),
                    canGoPrevious=bool(body.get("canGoPrevious", True)),
                    canSeek=bool(body.get("canSeek", True)),
                    loopStatus=str(body.get("loopStatus", "None")),
                    shuffle=bool(body.get("shuffle", False)),
                    last_pos_timestamp=now,
                )
                dev_players[player_name] = st

            # Ensure player is in the list
            plist = self._device_player_lists.setdefault(device_id, [])
            if player_name not in plist:
                plist.append(player_name)

            # Auto-focus playing player
            if st.isPlaying or self._device_active_player.get(device_id) is None:
                self._device_active_player[device_id] = player_name

            if self.event_bus:
                await self.event_bus.publish(
                    RemotePlayerUpdatedEvent(
                        device_id=device_id,
                        player=st.player,
                        title=st.title,
                        artist=st.artist,
                        album=st.album,
                        album_art_url=st.albumArtUrl,
                        is_playing=st.isPlaying,
                        pos_ms=st.get_current_position_ms(now),
                        length_ms=st.length,
                        volume=st.volume,
                    )
                )

    # =================================================================
    # Query APIs
    # =================================================================

    def get_players(self, device_id: str) -> List[str]:
        """Return list of available player names for device."""
        return list(self._device_player_lists.get(device_id, []))

    def get_player_list(self, device_id: str) -> List[str]:
        """Alias for get_players."""
        return self.get_players(device_id)

    def get_active_player(self, device_id: str) -> Optional[str]:
        """Return active / focused player for device."""
        active = self._device_active_player.get(device_id)
        if active:
            return active
        plist = self._device_player_lists.get(device_id, [])
        return plist[0] if plist else None

    def get_player_state(self, device_id: str, player: Optional[str] = None) -> Optional[RemotePlayerState]:
        """Return cached state for a player, defaulting to active player."""
        target = player or self.get_active_player(device_id)
        if not target:
            return None
        return self._device_players.get(device_id, {}).get(target)

    # =================================================================
    # Command Dispatch APIs
    # =================================================================

    def _resolve_sender(self, device_id: str, explicit_sender: Optional[SendPacketCallable] = None) -> SendPacketCallable:
        sender = explicit_sender or self._connected_senders.get(device_id) or self.packet_sender
        if not sender:
            raise ConnectionClosedError(f"No active connection to device '{device_id}'")
        return sender

    def _resolve_target_player(self, device_id: str, player: Optional[str] = None) -> str:
        target = player or self.get_active_player(device_id)
        if not target:
            raise MprisPlayerNotFoundError(f"No active media player found on device '{device_id}'")
        return target

    async def request_player_list(
        self,
        device_id: str,
        send_packet: Optional[SendPacketCallable] = None,
    ) -> Packet:
        """Transmit requestPlayerList packet to remote device."""
        sender = self._resolve_sender(device_id, send_packet)
        pkt = Packet(type="kdeconnect.mpris.request", body={"requestPlayerList": True})
        await safe_send_packet(sender, pkt)
        return pkt

    async def request_now_playing(
        self,
        device_id: str,
        player: Optional[str] = None,
        send_packet: Optional[SendPacketCallable] = None,
    ) -> Packet:
        """Transmit requestNowPlaying packet for player to remote device."""
        sender = self._resolve_sender(device_id, send_packet)
        target = self._resolve_target_player(device_id, player)
        pkt = Packet(type="kdeconnect.mpris.request", body={"player": target, "requestNowPlaying": True})
        await safe_send_packet(sender, pkt)
        return pkt

    async def send_action(
        self,
        device_id: str,
        action: str,
        player: Optional[str] = None,
        send_packet: Optional[SendPacketCallable] = None,
    ) -> Packet:
        """Send transport control action (Play, Pause, PlayPause, Stop, Next, Previous)."""
        if action not in ALLOWED_ACTIONS:
            raise ActionUnsupportedError(f"Unsupported media action '{action}'. Must be one of {sorted(ALLOWED_ACTIONS)}")
        sender = self._resolve_sender(device_id, send_packet)
        target = self._resolve_target_player(device_id, player)
        pkt = Packet(type="kdeconnect.mpris.request", body={"player": target, "action": action})
        await safe_send_packet(sender, pkt)
        return pkt

    async def set_volume(
        self,
        device_id: str,
        volume: int,
        player: Optional[str] = None,
        send_packet: Optional[SendPacketCallable] = None,
    ) -> Packet:
        """Set volume percentage (0..100) on remote player."""
        clamped = max(0, min(100, int(volume)))
        sender = self._resolve_sender(device_id, send_packet)
        target = self._resolve_target_player(device_id, player)
        pkt = Packet(type="kdeconnect.mpris.request", body={"player": target, "setVolume": clamped})
        await safe_send_packet(sender, pkt)
        return pkt

    async def seek(
        self,
        device_id: str,
        offset_ms: int,
        player: Optional[str] = None,
        send_packet: Optional[SendPacketCallable] = None,
    ) -> Packet:
        """Seek relative track position in milliseconds (wire unit is microseconds)."""
        offset_us = int(offset_ms) * 1000
        sender = self._resolve_sender(device_id, send_packet)
        target = self._resolve_target_player(device_id, player)
        pkt = Packet(type="kdeconnect.mpris.request", body={"player": target, "Seek": offset_us})
        await safe_send_packet(sender, pkt)
        return pkt

    async def set_position(
        self,
        device_id: str,
        position_ms: int,
        player: Optional[str] = None,
        send_packet: Optional[SendPacketCallable] = None,
    ) -> Packet:
        """Set absolute track position in milliseconds (wire unit is milliseconds)."""
        pos_ms = max(0, int(position_ms))
        sender = self._resolve_sender(device_id, send_packet)
        target = self._resolve_target_player(device_id, player)
        pkt = Packet(type="kdeconnect.mpris.request", body={"player": target, "SetPosition": pos_ms})
        await safe_send_packet(sender, pkt)
        return pkt

    async def set_loop_status(
        self,
        device_id: str,
        loop_status: str,
        player: Optional[str] = None,
        send_packet: Optional[SendPacketCallable] = None,
    ) -> Packet:
        """Set repeat/loop mode ('None', 'Track', 'Playlist')."""
        valid = {"None", "Track", "Playlist"}
        status = loop_status if loop_status in valid else "None"
        sender = self._resolve_sender(device_id, send_packet)
        target = self._resolve_target_player(device_id, player)
        pkt = Packet(type="kdeconnect.mpris.request", body={"player": target, "setLoopStatus": status})
        await safe_send_packet(sender, pkt)
        return pkt

    async def set_shuffle(
        self,
        device_id: str,
        shuffle: bool,
        player: Optional[str] = None,
        send_packet: Optional[SendPacketCallable] = None,
    ) -> Packet:
        """Enable or disable shuffle on remote player."""
        sender = self._resolve_sender(device_id, send_packet)
        target = self._resolve_target_player(device_id, player)
        pkt = Packet(type="kdeconnect.mpris.request", body={"player": target, "setShuffle": bool(shuffle)})
        await safe_send_packet(sender, pkt)
        return pkt

    # =================================================================
    # Lifecycle & Cleanup Handlers
    # =================================================================

    def on_device_unpaired(self, device_id: str) -> None:
        """Purge all cached player states for unpaired device."""
        self._device_players.pop(device_id, None)
        self._device_player_lists.pop(device_id, None)
        self._device_active_player.pop(device_id, None)
        self._connected_senders.pop(device_id, None)

    async def _on_device_disconnected(self, event: DeviceDisconnected) -> None:
        self._connected_senders.pop(event.device_id, None)

    async def _on_device_unpaired(self, event: DeviceUnpaired) -> None:
        self.on_device_unpaired(event.device_id)


__all__ = [
    "RemotePlayerState",
    "MprisRemote",
    "ALLOWED_ACTIONS",
]

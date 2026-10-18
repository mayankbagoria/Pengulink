"""pengulink.plugins.mpris.control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect MprisControl Server Plugin.

Exposes local host media players to connected remote devices via kdeconnect.mpris
and processes remote transport control requests via kdeconnect.mpris.request.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

from pengulink.core.events import (
    DeviceConnected,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
    LocalMediaPlayerChanged,
    LocalPlayerListChanged,
    MediaControlCommandReceived,
)
from pengulink.plugins.base import Packet, Plugin, SendPacketCallable, safe_send_packet
from pengulink.plugins.mpris.adapter import MprisHostAdapter, detect_mpris_adapter

logger = logging.getLogger("pengulink.plugins.mpris.control")


class MprisControl(Plugin):
    """KDE Connect MprisControl Server Plugin.

    Server role: Exposes host Linux media players over MPRIS to connected peers
    and processes inbound transport controls (play, pause, seek, volume, etc.).
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        adapter: Optional[MprisHostAdapter] = None,
        packet_sender: Optional[SendPacketCallable] = None,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self.adapter: MprisHostAdapter = adapter or detect_mpris_adapter()
        self.packet_sender = packet_sender
        self._connected_peers: Dict[str, SendPacketCallable] = {}
        self._unsub_callbacks: List[Callable[[], None]] = []

    @property
    def plugin_id(self) -> str:
        return "mpriscontrol"

    @property
    def incoming_capabilities(self) -> List[str]:
        return ["kdeconnect.mpris.request"]

    @property
    def outgoing_capabilities(self) -> List[str]:
        return ["kdeconnect.mpris"]

    async def start(self) -> None:
        """Start host adapter and subscribe to device lifecycle events."""
        self.adapter.set_change_callback(self._on_adapter_change)
        await self.adapter.start()

        if self.event_bus:
            self._unsub_callbacks.append(
                self.event_bus.subscribe(DeviceConnected, self._on_device_connected)
            )
            self._unsub_callbacks.append(
                self.event_bus.subscribe(DeviceDisconnected, self._on_device_disconnected)
            )
            self._unsub_callbacks.append(
                self.event_bus.subscribe(DeviceUnpaired, self._on_device_unpaired)
            )

    async def stop(self) -> None:
        """Stop adapter and unsubscribe from event bus."""
        for unsub in self._unsub_callbacks:
            try:
                unsub()
            except Exception:
                pass
        self._unsub_callbacks.clear()
        self._connected_peers.clear()
        await self.adapter.stop()

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Process incoming kdeconnect.mpris.request packet."""
        self._connected_peers[device_id] = send_packet
        body = packet.body or {}

        # 1. requestPlayerList
        if body.get("requestPlayerList"):
            players = await self.adapter.list_players()
            resp = Packet(
                type="kdeconnect.mpris",
                body={"playerList": players, "supportAlbumArtPayload": True},
            )
            await safe_send_packet(send_packet, resp)
            return

        target_player = body.get("player", "")

        # 2. requestNowPlaying
        if body.get("requestNowPlaying"):
            st = await self.adapter.get_player_status(target_player)
            if st:
                resp = Packet(type="kdeconnect.mpris", body=st.to_wire_dict())
            else:
                players = await self.adapter.list_players()
                resp = Packet(
                    type="kdeconnect.mpris",
                    body={"playerList": players, "supportAlbumArtPayload": True},
                )
            await safe_send_packet(send_packet, resp)
            return

        # 3. requestVolume
        if body.get("requestVolume"):
            st = await self.adapter.get_player_status(target_player)
            if st:
                vol_pct = max(0, min(100, int(round(st.volume * 100))))
                resp = Packet(
                    type="kdeconnect.mpris",
                    body={"player": target_player, "volume": vol_pct},
                )
                await safe_send_packet(send_packet, resp)
            return

        # 4. Transport action
        action = body.get("action")
        if action and action in {"Play", "Pause", "PlayPause", "Stop", "Next", "Previous"}:
            success = await self.adapter.send_action(target_player, action)
            if self.event_bus:
                await self.event_bus.publish(
                    MediaControlCommandReceived(
                        device_id=device_id,
                        player=target_player,
                        command=f"action:{action}",
                        params={"action": action, "success": success},
                    )
                )

        # 5. setVolume (0..100 -> float 0.0..1.0)
        if "setVolume" in body:
            raw_vol = body["setVolume"]
            if isinstance(raw_vol, (int, float)):
                clamped_pct = max(0, min(100, int(raw_vol)))
                vol_float = clamped_pct / 100.0
                await self.adapter.set_volume(target_player, vol_float)
                if self.event_bus:
                    await self.event_bus.publish(
                        MediaControlCommandReceived(
                            device_id=device_id,
                            player=target_player,
                            command="setVolume",
                            params={"volume": clamped_pct},
                        )
                    )

        # 6. Seek (microseconds)
        if "Seek" in body:
            raw_seek = body["Seek"]
            if isinstance(raw_seek, int):
                await self.adapter.seek(target_player, raw_seek)
                if self.event_bus:
                    await self.event_bus.publish(
                        MediaControlCommandReceived(
                            device_id=device_id,
                            player=target_player,
                            command="Seek",
                            params={"offset_us": raw_seek},
                        )
                    )

        # 7. SetPosition (milliseconds -> microseconds)
        if "SetPosition" in body:
            raw_pos_ms = body["SetPosition"]
            if isinstance(raw_pos_ms, int):
                pos_us = raw_pos_ms * 1000
                await self.adapter.set_position(target_player, pos_us)
                if self.event_bus:
                    await self.event_bus.publish(
                        MediaControlCommandReceived(
                            device_id=device_id,
                            player=target_player,
                            command="SetPosition",
                            params={"position_ms": raw_pos_ms},
                        )
                    )

        # 8. setLoopStatus
        if "setLoopStatus" in body:
            loop_st = str(body["setLoopStatus"])
            await self.adapter.set_loop_status(target_player, loop_st)

        # 9. setShuffle
        if "setShuffle" in body:
            shuffle_val = bool(body["setShuffle"])
            await self.adapter.set_shuffle(target_player, shuffle_val)

    async def _on_adapter_change(self, player: str, delta: Dict[str, Any]) -> None:
        """Broadcast player state changes or playerList updates to all connected peers."""
        packet = Packet(type="kdeconnect.mpris", body=delta)
        dead_peers: List[str] = []

        for dev_id, sender in list(self._connected_peers.items()):
            try:
                await safe_send_packet(sender, packet)
            except Exception:
                dead_peers.append(dev_id)

        for dev_id in dead_peers:
            self._connected_peers.pop(dev_id, None)

        if self.event_bus:
            if "playerList" in delta:
                await self.event_bus.publish(
                    LocalPlayerListChanged(player_list=delta.get("playerList", []))
                )
            elif "player" in delta:
                await self.event_bus.publish(
                    LocalMediaPlayerChanged(
                        player=delta.get("player", ""),
                        title=delta.get("title", ""),
                        artist=delta.get("artist", ""),
                        album=delta.get("album", ""),
                        album_art_url=delta.get("albumArtUrl", ""),
                        is_playing=delta.get("isPlaying", False),
                        pos_ms=delta.get("pos", 0),
                        length_ms=delta.get("length", -1),
                        volume=delta.get("volume", 100),
                        can_play=delta.get("canPlay", True),
                        can_pause=delta.get("canPause", True),
                        can_go_next=delta.get("canGoNext", True),
                        can_go_previous=delta.get("canGoPrevious", True),
                        can_seek=delta.get("canSeek", True),
                        loop_status=delta.get("loopStatus", "None"),
                        shuffle=delta.get("shuffle", False),
                    )
                )

    async def _on_device_connected(self, event: DeviceConnected) -> None:
        """Broadcast initial playerList to freshly connected peer."""
        if not event.device_id:
            return
        sender = self._connected_peers.get(event.device_id)
        players = await self.adapter.list_players()
        pkt = Packet(
            type="kdeconnect.mpris",
            body={"playerList": players, "supportAlbumArtPayload": True},
        )
        if sender:
            try:
                await safe_send_packet(sender, pkt)
            except Exception:
                pass
        elif self.packet_sender:
            try:
                try:
                    res = self.packet_sender(event.device_id, pkt)
                except TypeError:
                    res = self.packet_sender(pkt)
                if inspect.isawaitable(res):
                    await res
            except Exception:
                pass

    async def _on_device_disconnected(self, event: DeviceDisconnected) -> None:
        self._connected_peers.pop(event.device_id, None)

    async def _on_device_unpaired(self, event: DeviceUnpaired) -> None:
        self._connected_peers.pop(event.device_id, None)


__all__ = ["MprisControl"]

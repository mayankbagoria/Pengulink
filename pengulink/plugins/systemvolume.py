"""pengulink.plugins.systemvolume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect SystemVolume host plugin for local Linux audio control.

Implements:
- Capability: outgoing "kdeconnect.systemvolume", incoming "kdeconnect.systemvolume.request"
- Broadcasts complete sinkList upon connection and in response to requestSinks
- Handles inbound volume adjustments, mute toggling, and default sink changes
- Upstream KDE Connect invariant: setting volume automatically unmutes the sink
- Emits strongly typed AudioSinkListEvent and AudioSinkUpdatedEvent on EventBus
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from typing import Any

from pengulink.core.events import (
    AudioSinkListEvent,
    AudioSinkUpdatedEvent,
    DeviceConnected,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
)
from pengulink.core.exceptions import PenguLinkError
from pengulink.network.packet import Packet
from pengulink.plugins.base import Plugin, SendPacketCallable, safe_send_packet
from pengulink.services.audio import (
    MAX_AUDIO_VOLUME,
    AudioSink,
    AudioSinkBackend,
    LocalAudioService,
    detect_audio_backend,
)

logger = logging.getLogger("pengulink.plugins.systemvolume")

SYSTEM_VOLUME_PACKET_TYPE = "kdeconnect.systemvolume"
SYSTEM_VOLUME_REQUEST_TYPE = "kdeconnect.systemvolume.request"


class SystemVolumePlugin(Plugin):
    """Host plugin exposing local Linux audio sinks and mixer controls."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        backend: AudioSinkBackend | None = None,
        audio_service: LocalAudioService | None = None,
        packet_sender: Any | None = None,
    ) -> None:
        super().__init__(event_bus)
        if audio_service is not None:
            self.audio_service = audio_service
        elif backend is not None:
            self.audio_service = LocalAudioService(backend=backend, event_bus=None)
        else:
            self.audio_service = LocalAudioService(
                backend=detect_audio_backend(), event_bus=None
            )

        self.backend = self.audio_service.backend
        self.packet_sender = packet_sender
        self._connected_peers: dict[str, SendPacketCallable] = {}
        self._unsub_callbacks: list[Callable[[], None]] = []

    @property
    def plugin_id(self) -> str:
        return "systemvolume"

    @property
    def incoming_capabilities(self) -> list[str]:
        return [SYSTEM_VOLUME_REQUEST_TYPE]

    @property
    def outgoing_capabilities(self) -> list[str]:
        return [SYSTEM_VOLUME_PACKET_TYPE]

    async def start(self) -> None:
        """Subscribe to device lifecycle events."""
        await super().start()
        if self.event_bus:
            self._unsub_callbacks.append(
                self.event_bus.subscribe(DeviceConnected, self._on_device_connected)
            )
            self._unsub_callbacks.append(
                self.event_bus.subscribe(
                    DeviceDisconnected, self._on_device_disconnected
                )
            )
            self._unsub_callbacks.append(
                self.event_bus.subscribe(DeviceUnpaired, self._on_device_unpaired)
            )

    async def stop(self) -> None:
        """Unsubscribe from lifecycle events and clear connected peers."""
        for unsub in self._unsub_callbacks:
            try:
                unsub()
            except (
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
                KeyError,
            ) as exc:
                logger.debug("Suppressed error unsubscribing event: %s", exc)
        self._unsub_callbacks.clear()
        self._connected_peers.clear()
        await super().stop()

    async def _on_device_connected(self, event: DeviceConnected) -> None:
        """Broadcast initial sink list when a paired device connects."""
        if not event.device_id:
            return
        sinks = await self.audio_service.list_sinks()
        pkt = Packet(
            type=SYSTEM_VOLUME_PACKET_TYPE,
            body={"sinkList": [s.to_wire_dict() for s in sinks]},
        )
        sender = self._connected_peers.get(event.device_id)
        if sender:
            try:
                await safe_send_packet(sender, pkt)
            except (
                OSError,
                ConnectionError,
                RuntimeError,
                PenguLinkError,
                asyncio.TimeoutError,
            ) as exc:
                logger.debug(
                    "Failed sending initial sinkList to %s: %s", event.device_id, exc
                )
        elif self.packet_sender:
            try:
                try:
                    res = self.packet_sender(event.device_id, pkt)
                except TypeError:
                    res = self.packet_sender(pkt)
                if inspect.isawaitable(res):
                    await res
            except (
                OSError,
                ConnectionError,
                RuntimeError,
                PenguLinkError,
                TypeError,
            ) as exc:
                logger.debug(
                    "Failed sending initial sinkList to %s via packet_sender: %s",
                    event.device_id,
                    exc,
                )

        if self.event_bus:
            await self.event_bus.publish(
                AudioSinkListEvent(device_id=event.device_id, sinks=sinks)
            )

    async def _on_device_disconnected(self, event: DeviceDisconnected) -> None:
        self._connected_peers.pop(event.device_id, None)

    async def _on_device_unpaired(self, event: DeviceUnpaired) -> None:
        self._connected_peers.pop(event.device_id, None)

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Process incoming kdeconnect.systemvolume.request packet."""
        self._connected_peers[device_id] = send_packet
        if packet.type != SYSTEM_VOLUME_REQUEST_TYPE:
            logger.debug("Ignoring packet with unexpected type: %s", packet.type)
            return

        body = packet.body or {}

        # 1. requestSinks: full sink list enumeration query
        if body.get("requestSinks"):
            sinks = await self.audio_service.list_sinks()
            pkt = Packet(
                type=SYSTEM_VOLUME_PACKET_TYPE,
                body={"sinkList": [s.to_wire_dict() for s in sinks]},
            )
            try:
                await safe_send_packet(send_packet, pkt)
            except (
                OSError,
                ConnectionError,
                RuntimeError,
                PenguLinkError,
                asyncio.TimeoutError,
            ) as exc:
                logger.debug("Failed sending sinkList to %s: %s", device_id, exc)
                self._connected_peers.pop(device_id, None)
            if self.event_bus:
                await self.event_bus.publish(
                    AudioSinkListEvent(device_id=device_id, sinks=sinks)
                )
            return

        # 2. Sink volume/mute/enabled mutation request
        raw_name = body.get("name")
        if not raw_name or not isinstance(raw_name, (str, int)):
            logger.warning(
                "kdeconnect.systemvolume.request missing 'name' field: %s", str(body)
            )
            return

        sink_name = str(raw_name).strip()
        if not sink_name:
            logger.warning("kdeconnect.systemvolume.request contains empty 'name'")
            return

        volume_updated = False
        muted_updated = False
        enabled_updated = False

        # Volume adjustment (upstream invariant: setting volume automatically unmutes)
        if "volume" in body:
            try:
                vol = int(float(body["volume"]))
                vol = max(0, min(vol, MAX_AUDIO_VOLUME))
                success = await self.audio_service.set_volume(sink_name, vol)
                if success:
                    volume_updated = True
                else:
                    logger.warning("Failed to set volume on sink: %s", sink_name)
            except (ValueError, TypeError, OverflowError) as exc:
                logger.warning(
                    "Invalid volume value in request for %s: %s", sink_name, exc
                )

        # Mute adjustment
        if "muted" in body:
            try:
                muted_val = bool(body["muted"])
                success = await self.audio_service.set_muted(sink_name, muted_val)
                if success:
                    muted_updated = True
                else:
                    logger.warning("Failed to set muted on sink: %s", sink_name)
            except (ValueError, TypeError, PenguLinkError, OSError) as exc:
                logger.warning(
                    "Invalid muted value in request for %s: %s", sink_name, exc
                )

        # Enabled / default sink adjustment
        if body.get("enabled"):
            try:
                success = await self.audio_service.set_default(sink_name)
                if success:
                    enabled_updated = True
                else:
                    logger.warning("Failed to set default sink: %s", sink_name)
            except (ValueError, TypeError, PenguLinkError, OSError) as exc:
                logger.warning("Error setting default sink %s: %s", sink_name, exc)

        # If any modification was applied, respond with updated sink state
        if volume_updated or muted_updated or enabled_updated:
            current_sink = await self.audio_service.get_sink(sink_name)
            if not current_sink:
                all_sinks = await self.audio_service.list_sinks()
                for s in all_sinks:
                    if s.name == sink_name or sink_name in s.name:
                        current_sink = s
                        break

            if current_sink:
                update_body: dict[str, Any] = {"name": current_sink.name}
                if volume_updated:
                    update_body["volume"] = current_sink.volume
                    update_body["muted"] = current_sink.muted
                if muted_updated:
                    update_body["muted"] = current_sink.muted
                if enabled_updated:
                    update_body["enabled"] = current_sink.enabled

                reply_pkt = Packet(type=SYSTEM_VOLUME_PACKET_TYPE, body=update_body)
                try:
                    await safe_send_packet(send_packet, reply_pkt)
                except (
                    OSError,
                    ConnectionError,
                    RuntimeError,
                    PenguLinkError,
                    asyncio.TimeoutError,
                ) as exc:
                    logger.debug(
                        "Failed sending update reply to %s: %s", device_id, exc
                    )
                    self._connected_peers.pop(device_id, None)

                # Fanout update to other connected peers so their UI remains synchronized
                await self.broadcast_sink_update(
                    current_sink, exclude_device_id=device_id
                )

                if self.event_bus:
                    await self.event_bus.publish(
                        AudioSinkUpdatedEvent(device_id=device_id, sink=current_sink)
                    )

    # -----------------------------------------------------------------
    # Public Control APIs for Daemon IPC and CLI
    # -----------------------------------------------------------------

    async def list_sinks(self) -> list[AudioSink]:
        """List all available audio sinks."""
        return await self.audio_service.list_sinks()

    async def get_sink(self, name: str) -> AudioSink | None:
        """Get an audio sink by name."""
        return await self.audio_service.get_sink(name)

    async def set_volume(
        self,
        name: str,
        volume: int,
        is_percent: bool = False,
    ) -> bool:
        """Set sink volume and broadcast update to connected peers."""
        success = await self.audio_service.set_volume(
            name=name, volume=volume, is_percent=is_percent
        )
        if success:
            sink = await self.audio_service.get_sink(name)
            if sink:
                await self.broadcast_sink_update(sink)
                if self.event_bus:
                    await self.event_bus.publish(
                        AudioSinkUpdatedEvent(device_id=None, sink=sink)
                    )
        return success

    async def set_muted(self, name: str, muted: bool) -> bool:
        """Set sink mute state and broadcast update to connected peers."""
        success = await self.audio_service.set_muted(name=name, muted=muted)
        if success:
            sink = await self.audio_service.get_sink(name)
            if sink:
                await self.broadcast_sink_update(sink)
                if self.event_bus:
                    await self.event_bus.publish(
                        AudioSinkUpdatedEvent(device_id=None, sink=sink)
                    )
        return success

    async def set_default(self, name: str) -> bool:
        """Set default sink and broadcast sink list to connected peers."""
        success = await self.audio_service.set_default(name=name)
        if success:
            await self.broadcast_sink_list()
            if self.event_bus:
                sinks = await self.audio_service.list_sinks()
                await self.event_bus.publish(
                    AudioSinkListEvent(device_id=None, sinks=sinks)
                )
        return success

    async def broadcast_sink_update(
        self, sink: AudioSink, exclude_device_id: str | None = None
    ) -> None:
        """Broadcast single sink mutation update to all connected peers."""
        pkt = Packet(
            type=SYSTEM_VOLUME_PACKET_TYPE,
            body={
                "name": sink.name,
                "volume": sink.volume,
                "muted": sink.muted,
                "enabled": sink.enabled,
            },
        )
        failed_devices = []
        for dev_id, sender in list(self._connected_peers.items()):
            if exclude_device_id and dev_id == exclude_device_id:
                continue
            try:
                await safe_send_packet(sender, pkt)
            except (
                OSError,
                ConnectionError,
                RuntimeError,
                PenguLinkError,
                asyncio.TimeoutError,
            ) as exc:
                logger.debug("Failed broadcasting sink update to %s: %s", dev_id, exc)
                failed_devices.append(dev_id)

        for dev_id in failed_devices:
            self._connected_peers.pop(dev_id, None)

    async def broadcast_sink_list(self, exclude_device_id: str | None = None) -> None:
        """Broadcast complete sink list to all connected peers."""
        sinks = await self.audio_service.list_sinks()
        pkt = Packet(
            type=SYSTEM_VOLUME_PACKET_TYPE,
            body={"sinkList": [s.to_wire_dict() for s in sinks]},
        )
        failed_devices = []
        for dev_id, sender in list(self._connected_peers.items()):
            if exclude_device_id and dev_id == exclude_device_id:
                continue
            try:
                await safe_send_packet(sender, pkt)
            except (
                OSError,
                ConnectionError,
                RuntimeError,
                PenguLinkError,
                asyncio.TimeoutError,
            ) as exc:
                logger.debug("Failed broadcasting sink list to %s: %s", dev_id, exc)
                failed_devices.append(dev_id)

        for dev_id in failed_devices:
            self._connected_peers.pop(dev_id, None)


__all__ = ["SystemVolumePlugin"]

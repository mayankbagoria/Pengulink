"""pengulink.plugins.remotesystemvolume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect RemoteSystemVolume client plugin for remote device audio control.

Implements:
- Capability: incoming "kdeconnect.systemvolume", outgoing "kdeconnect.systemvolume.request"
- RemoteAudioSink dataclass with robust parser and volume clamping
- Ingests and caches remote device audio sinks (sinkList and incremental updates)
- Device-scoped cache isolation and clean unpair eviction
- Outbound controls: requestSinks, volume, muted, default
- Publishes RemoteAudioSinkListEvent and RemoteAudioSinkUpdatedEvent on EventBus
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pengulink.core.events import (
    DeviceDisconnected,
    DeviceUnpaired,
    DeviceUnpairedEvent,
    EventBus,
    RemoteAudioSinkListEvent,
    RemoteAudioSinkUpdatedEvent,
)
from pengulink.core.exceptions import PenguLinkError
from pengulink.network.packet import Packet
from pengulink.plugins.base import Plugin, SendPacketCallable, safe_send_packet

BasePlugin = Plugin

logger = logging.getLogger("pengulink.plugins.remotesystemvolume")

SYSTEM_VOLUME_PACKET_TYPE = "kdeconnect.systemvolume"
SYSTEM_VOLUME_REQUEST_TYPE = "kdeconnect.systemvolume.request"
DEFAULT_MAX_VOLUME = 65536


def _safe_int(val: Any, default: int = 0) -> int:
    """Coerce value to int safely, handling floats, strings, inf/nan, and invalid types."""
    if val is None or isinstance(val, (dict, list, set)):
        return default
    if isinstance(val, bool):
        return default
    if isinstance(val, int):
        return val
    try:
        if isinstance(val, float):
            if math.isnan(val) or math.isinf(val):
                return default
            return int(val)
        if isinstance(val, str):
            s = val.strip()
            if not s or s.lower() in (
                "nan",
                "inf",
                "-inf",
                "+inf",
                "infinity",
                "-infinity",
            ):
                return default
            f = float(s)
            if math.isnan(f) or math.isinf(f):
                return default
            return int(f)
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return int(f)
    except (ValueError, TypeError, OverflowError):
        return default


@dataclass
class RemoteAudioSink:
    """Represents an audio output sink on a connected remote KDE Connect device."""

    name: str
    description: str = ""
    volume: int = 0
    max_volume: int = DEFAULT_MAX_VOLUME
    muted: bool = False
    enabled: bool = False
    device_id: str = ""

    def __post_init__(self) -> None:
        if self.max_volume <= 0:
            self.max_volume = DEFAULT_MAX_VOLUME
        self.volume = max(0, min(int(self.volume), self.max_volume))
        if not self.description:
            self.description = self.name or "Remote Audio Output"

    @property
    def volume_percent(self) -> int:
        """Normalized integer volume percentage between 0 and 100."""
        if self.max_volume <= 0:
            return 0
        pct = round((self.volume / self.max_volume) * 100)
        return max(0, min(100, pct))

    def to_wire_dict(self) -> dict[str, Any]:
        """Serialize for upstream KDE Connect wire protocol packets."""
        return {
            "name": self.name,
            "description": self.description,
            "volume": self.volume,
            "maxVolume": self.max_volume,
            "muted": self.muted,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize for IPC, CLI, and debugging representations."""
        return {
            "name": self.name,
            "description": self.description,
            "volume": self.volume,
            "max_volume": self.max_volume,
            "maxVolume": self.max_volume,
            "muted": self.muted,
            "enabled": self.enabled,
            "device_id": self.device_id,
            "volume_percent": self.volume_percent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], device_id: str = "") -> RemoteAudioSink:
        """Parse dictionary into RemoteAudioSink safely with fallbacks."""
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected dict for RemoteAudioSink.from_dict, got {type(data).__name__}"
            )

        raw_name = data.get("name", "")
        name = str(raw_name).strip() if raw_name is not None else ""

        raw_desc = data.get("description")
        if raw_desc is None:
            raw_desc = data.get("desc")
        if raw_desc is None:
            raw_desc = data.get("nick")
        description = str(raw_desc).strip() if raw_desc is not None else ""
        if not description:
            description = name or "Remote Audio Output"

        raw_max = data.get("maxVolume")
        if raw_max is None:
            raw_max = data.get("max_volume")
        max_vol = _safe_int(raw_max, default=DEFAULT_MAX_VOLUME)
        if max_vol <= 0:
            max_vol = DEFAULT_MAX_VOLUME

        raw_vol = data.get("volume")
        volume = _safe_int(raw_vol, default=0)
        volume = max(0, min(volume, max_vol))

        raw_muted = data.get("muted", False)
        muted = bool(raw_muted)

        raw_enabled = data.get("enabled")
        if raw_enabled is None:
            raw_enabled = data.get("default", False)
        enabled = bool(raw_enabled)

        dev_id = str(data.get("device_id") or device_id or "").strip()

        return cls(
            name=name,
            description=description,
            volume=volume,
            max_volume=max_vol,
            muted=muted,
            enabled=enabled,
            device_id=dev_id,
        )


class RemoteSystemVolumePlugin(BasePlugin):
    """Client plugin managing audio sinks on remote KDE Connect devices."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        packet_sender: Any | None = None,
        is_paired_checker: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(event_bus)
        self.packet_sender = packet_sender
        self.is_paired_checker = is_paired_checker
        self._device_sinks: dict[str, dict[str, RemoteAudioSink]] = {}
        self._connected_peers: dict[str, SendPacketCallable] = {}
        self._unsub_callbacks: list[Callable[[], None]] = []

    @property
    def plugin_id(self) -> str:
        return "remotesystemvolume"

    @property
    def name(self) -> str:
        return self.plugin_id

    @property
    def incoming_capabilities(self) -> list[str]:
        return [SYSTEM_VOLUME_PACKET_TYPE]

    @property
    def outgoing_capabilities(self) -> list[str]:
        return [SYSTEM_VOLUME_REQUEST_TYPE]

    async def start(self) -> None:
        """Lifecycle hook: subscribe to device lifecycle events."""
        await super().start()
        if self.event_bus:
            self._unsub_callbacks.append(
                self.event_bus.subscribe(
                    DeviceDisconnected, self._on_device_disconnected
                )
            )
            self._unsub_callbacks.append(
                self.event_bus.subscribe(DeviceUnpaired, self._on_device_unpaired)
            )
            if DeviceUnpairedEvent is not DeviceUnpaired:
                self._unsub_callbacks.append(
                    self.event_bus.subscribe(
                        DeviceUnpairedEvent, self._on_device_unpaired
                    )
                )

    async def stop(self) -> None:
        """Lifecycle hook: clean up resources on daemon shutdown."""
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
                logger.debug(
                    "Error unsubscribing event in RemoteSystemVolumePlugin: %s", exc
                )
        self._unsub_callbacks.clear()
        self._connected_peers.clear()
        self._device_sinks.clear()
        await super().stop()

    async def _on_device_disconnected(self, event: DeviceDisconnected) -> None:
        dev_id = getattr(event, "device_id", "")
        if dev_id:
            self._connected_peers.pop(dev_id, None)

    async def _on_device_unpaired(self, event: DeviceUnpaired) -> None:
        dev_id = getattr(event, "device_id", "")
        if dev_id:
            self.evict_device(dev_id)

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Process incoming kdeconnect.systemvolume packet from a remote device."""
        if not isinstance(packet.body, dict):
            logger.debug(
                "Dropping RemoteSystemVolume packet from %s: body is not a dict (%s)",
                device_id,
                type(packet.body).__name__,
            )
            return

        if packet.type != SYSTEM_VOLUME_PACKET_TYPE:
            logger.debug("Ignoring packet with unexpected type: %s", packet.type)
            return

        # Pairing check: if is_paired_checker is configured, drop packets from unpaired devices
        if self.is_paired_checker is not None:
            try:
                is_paired = self.is_paired_checker(device_id)
            except (
                KeyError,
                ValueError,
                TypeError,
                RuntimeError,
                PenguLinkError,
            ) as exc:
                logger.debug("Error checking pairing status for %s: %s", device_id, exc)
                is_paired = False
            if not is_paired:
                logger.debug(
                    "Dropping RemoteSystemVolume packet from unpaired device: %s",
                    device_id,
                )
                return

        # Peer is authenticated and packet format valid: register send_packet callback
        if send_packet is not None:
            self._connected_peers[device_id] = send_packet

        body = packet.body

        # Case 1: Full sink list enumeration in packet body
        if "sinkList" in body:
            raw_list = body.get("sinkList")
            if not isinstance(raw_list, list):
                logger.debug("sinkList is not a list in packet from %s", device_id)
                return

            if len(raw_list) == 0:
                self._device_sinks[device_id] = {}
                if self.event_bus:
                    await self.event_bus.publish(
                        RemoteAudioSinkListEvent(device_id=device_id, sinks=[])
                    )
                return

            parsed_sinks: dict[str, RemoteAudioSink] = {}
            for item in raw_list:
                if not isinstance(item, dict):
                    continue
                name_val = item.get("name")
                if not name_val or not isinstance(name_val, (str, int)):
                    continue
                name_str = str(name_val).strip()
                if not name_str:
                    continue
                try:
                    sink = RemoteAudioSink.from_dict(item, device_id=device_id)
                    parsed_sinks[sink.name] = sink
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    OverflowError,
                ) as exc:
                    logger.debug("Failed parsing sink entry %r: %s", item, exc)
                    continue

            self._device_sinks[device_id] = parsed_sinks
            if self.event_bus:
                await self.event_bus.publish(
                    RemoteAudioSinkListEvent(
                        device_id=device_id, sinks=list(parsed_sinks.values())
                    )
                )
            return

        # Case 2: Single sink mutation update
        raw_name = body.get("name")
        has_volume = "volume" in body
        has_muted = "muted" in body
        has_max_vol = "maxVolume" in body or "max_volume" in body
        has_desc = "description" in body or "desc" in body
        has_enabled = "enabled" in body or "default" in body

        if (
            raw_name is not None
            and isinstance(raw_name, (str, int))
            and (has_volume or has_muted or has_max_vol or has_desc or has_enabled)
        ):
            sink_name = str(raw_name).strip()
            if not sink_name:
                return

            device_cache = self._device_sinks.setdefault(device_id, {})
            existing = device_cache.get(sink_name)

            if existing is not None:
                max_vol = existing.max_volume
                if has_max_vol:
                    raw_mv = (
                        body.get("maxVolume")
                        if "maxVolume" in body
                        else body.get("max_volume")
                    )
                    mv = _safe_int(raw_mv, default=existing.max_volume)
                    if mv > 0:
                        max_vol = mv

                vol = existing.volume
                if has_volume:
                    vol = _safe_int(body.get("volume"), default=existing.volume)
                    vol = max(0, min(vol, max_vol))

                muted = existing.muted
                if has_muted:
                    muted = bool(body.get("muted"))

                enabled = existing.enabled
                if has_enabled:
                    raw_en = (
                        body.get("enabled")
                        if "enabled" in body
                        else body.get("default")
                    )
                    enabled = bool(raw_en)

                desc = existing.description
                if has_desc:
                    raw_desc = (
                        body.get("description")
                        if "description" in body
                        else body.get("desc")
                    )
                    if raw_desc is not None:
                        desc = str(raw_desc).strip() or desc

                updated_sink = RemoteAudioSink(
                    name=sink_name,
                    description=desc,
                    volume=vol,
                    max_volume=max_vol,
                    muted=muted,
                    enabled=enabled,
                    device_id=device_id,
                )
            else:
                updated_sink = RemoteAudioSink.from_dict(body, device_id=device_id)

            device_cache[sink_name] = updated_sink
            if self.event_bus:
                await self.event_bus.publish(
                    RemoteAudioSinkUpdatedEvent(device_id=device_id, sink=updated_sink)
                )
            return

    async def _send_to_device(
        self,
        device_id: str,
        packet: Packet,
        send_packet: SendPacketCallable | None = None,
    ) -> None:
        """Route outbound packet via provided send_packet callback, cached peer, or packet_sender."""
        sender = send_packet or self._connected_peers.get(device_id)
        if sender is not None:
            await safe_send_packet(sender, packet)
            return

        if self.packet_sender is not None:
            try:
                res = self.packet_sender(device_id, packet)
            except TypeError:
                res = self.packet_sender(packet)
            if inspect.isawaitable(res):
                await res
            return

        raise PenguLinkError(
            f"Cannot send packet to {device_id}: no active connection or packet sender"
        )

    async def request_remote_sinks(
        self, device_id: str, send_packet: SendPacketCallable | None = None
    ) -> None:
        """Request the complete list of audio sinks from a remote device."""
        pkt = Packet(
            type=SYSTEM_VOLUME_REQUEST_TYPE,
            body={"requestSinks": True},
        )
        await self._send_to_device(device_id, pkt, send_packet)

    async def set_remote_volume(
        self,
        device_id: str,
        name: str,
        volume: int,
        send_packet: SendPacketCallable | None = None,
    ) -> None:
        """Set volume on a remote audio sink, clamping to the sink's max_volume."""
        sink = self.get_sink(device_id, name)
        max_vol = sink.max_volume if sink else DEFAULT_MAX_VOLUME
        vol = _safe_int(volume, default=0)
        clamped_vol = max(0, min(vol, max_vol))
        pkt = Packet(
            type=SYSTEM_VOLUME_REQUEST_TYPE,
            body={"name": name, "volume": clamped_vol},
        )
        await self._send_to_device(device_id, pkt, send_packet)

    async def set_remote_muted(
        self,
        device_id: str,
        name: str,
        muted: bool,
        send_packet: SendPacketCallable | None = None,
    ) -> None:
        """Set mute state on a remote audio sink."""
        pkt = Packet(
            type=SYSTEM_VOLUME_REQUEST_TYPE,
            body={"name": name, "muted": bool(muted)},
        )
        await self._send_to_device(device_id, pkt, send_packet)

    async def set_remote_default(
        self,
        device_id: str,
        name: str,
        send_packet: SendPacketCallable | None = None,
    ) -> None:
        """Set a remote audio sink as the default output."""
        pkt = Packet(
            type=SYSTEM_VOLUME_REQUEST_TYPE,
            body={"name": name, "enabled": True, "default": True},
        )
        await self._send_to_device(device_id, pkt, send_packet)

    def get_sinks(self, device_id: str) -> list[RemoteAudioSink]:
        """Return list of cached remote audio sinks for a device, or [] if unknown."""
        return list(self._device_sinks.get(device_id, {}).values())

    def get_sink(self, device_id: str, name: str) -> RemoteAudioSink | None:
        """Return specific cached remote audio sink by name, or None."""
        device_sinks = self._device_sinks.get(device_id, {})
        if name in device_sinks:
            return device_sinks[name]
        name_lower = name.lower()
        for s in device_sinks.values():
            if s.name.lower() == name_lower:
                return s
        return None

    def evict_device(self, device_id: str) -> None:
        """Purge all cached audio sinks and peer sender for a device."""
        self._device_sinks.pop(device_id, None)
        self._connected_peers.pop(device_id, None)

    # Alias for Daemon IPC and CLI
    get_remote_sinks = get_sinks



__all__ = [
    "DEFAULT_MAX_VOLUME",
    "BasePlugin",
    "RemoteAudioSink",
    "RemoteSystemVolumePlugin",
]

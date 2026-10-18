"""pengulink.plugins.remotecommands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect RemoteCommands Client Plugin.

Client role: Ingests configured remote command lists (kdeconnect.runcommand and
kdeconnect.remotecommands) from paired devices and sends execution trigger packets
(kdeconnect.remoteforcecommand and kdeconnect.runcommand.request).

CRITICAL SECURITY INVARIANT:
This plugin operates strictly as a remote controller client. Under NO circumstances
may this plugin execute shell commands, spawn local subprocesses, or evaluate
shell strings on the local host. Local execution is reserved for the separate
Run Command plugin.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import json
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Union

from pengulink.core.events import (
    DeviceUnpaired,
    EventBus,
    RemoteCommandExecutedEvent,
    RemoteCommandOutputEvent,
    RemoteCommandsReceivedEvent,
)
from pengulink.core.exceptions import (
    ConnectionClosedError,
    RemoteCommandError,
    RemoteCommandNotFoundError,
)
from pengulink.network.packet import Packet
from pengulink.plugins.base import Plugin, SendPacketCallable, safe_send_packet

logger = logging.getLogger("pengulink.plugins.remotecommands")

MAX_KEY_LENGTH = 128
MAX_NAME_LENGTH = 256
MAX_COMMAND_LENGTH = 1024
MAX_COMMANDS_PER_DEVICE = 256


# =====================================================================
# Data Models
# =====================================================================

@dataclass(frozen=True)
class RemoteCommand:
    """Immutable model representing a command configured on a remote device."""

    key: str
    name: str
    command: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize remote command to dictionary."""
        return {
            "key": self.key,
            "name": self.name,
            "command": self.command,
        }


@dataclass
class DeviceRemoteCommands:
    """Device-scoped cache of available remote commands."""

    device_id: str
    commands: Dict[str, RemoteCommand] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)

    def get(self, key: str) -> Optional[RemoteCommand]:
        """Retrieve command by key."""
        return self.commands.get(key)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all cached commands for JSON-RPC IPC transmission."""
        return {
            "device_id": self.device_id,
            "commands": {k: c.to_dict() for k, c in self.commands.items()},
            "count": len(self.commands),
            "last_updated": self.last_updated,
        }

    def to_list(self) -> List[Dict[str, Any]]:
        """Return list of serialized commands."""
        return [cmd.to_dict() for cmd in self.commands.values()]


# =====================================================================
# Resilient Dual-Format Parser
# =====================================================================

def parse_command_list(body: Any) -> Dict[str, RemoteCommand]:
    """Parse incoming command list payload supporting dual JSON-string and dict formats.

    Upstream KDE Connect versions serialize commandList either as a JSON-escaped string
    (C++/Qt implementations) or as a native JSON dictionary (Android / Python implementations),
    or within an array. This parser safely decodes both formats with strict boundary clamping.

    Args:
        body: Raw dictionary from packet body, or nested candidate object.

    Returns:
        Dict mapping command key to RemoteCommand instances.
    """
    if not isinstance(body, dict):
        return {}

    # Stage 1: Candidate Extraction
    candidate = body.get("commandList")
    if candidate is None:
        candidate = body.get("commands")

    # If neither is present, check if body itself directly maps keys to command dicts
    if candidate is None:
        if any(
            isinstance(v, dict) and ("name" in v or "command" in v)
            for v in body.values()
        ):
            candidate = body
        else:
            return {}

    # Stage 2: JSON String Unfolding
    if isinstance(candidate, str):
        trimmed = candidate.strip()
        if not trimmed:
            return {}
        try:
            candidate = json.loads(
                trimmed,
                parse_constant=lambda val: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant: {val}")),
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as exc:
            logger.warning("Failed to decode JSON-stringified commandList: %s", exc)
            return {}

    # Stage 3 & 4: Normalization, Validation, and Clamping
    parsed: Dict[str, RemoteCommand] = {}

    if isinstance(candidate, dict):
        for raw_key, raw_val in candidate.items():
            if len(parsed) >= MAX_COMMANDS_PER_DEVICE:
                break

            key_str = str(raw_key).strip()
            if not key_str:
                continue
            clamped_key = key_str[:MAX_KEY_LENGTH]

            if isinstance(raw_val, dict):
                name_val = raw_val.get("name")
                name_str = str(name_val).strip() if name_val is not None else clamped_key
                if not name_str:
                    name_str = clamped_key

                cmd_val = raw_val.get("command", "")
                cmd_str = str(cmd_val) if cmd_val is not None else ""
            elif isinstance(raw_val, str):
                name_str = raw_val.strip() or clamped_key
                cmd_str = ""
            elif raw_val is None:
                continue
            elif isinstance(raw_val, float) and (math.isnan(raw_val) or math.isinf(raw_val)):
                continue
            else:
                name_str = str(raw_val).strip() or clamped_key
                cmd_str = ""

            clamped_name = name_str[:MAX_NAME_LENGTH]
            clamped_cmd = cmd_str[:MAX_COMMAND_LENGTH]

            parsed[clamped_key] = RemoteCommand(
                key=clamped_key,
                name=clamped_name,
                command=clamped_cmd,
            )

    elif isinstance(candidate, list):
        for item in candidate:
            if len(parsed) >= MAX_COMMANDS_PER_DEVICE:
                break
            if not isinstance(item, dict):
                continue

            raw_key = item.get("key") or item.get("id")
            if not raw_key:
                continue
            clamped_key = str(raw_key).strip()[:MAX_KEY_LENGTH]
            if not clamped_key:
                continue

            name_val = item.get("name")
            name_str = str(name_val).strip() if name_val is not None else clamped_key
            if not name_str:
                name_str = clamped_key

            cmd_val = item.get("command", "")
            cmd_str = str(cmd_val) if cmd_val is not None else ""

            clamped_name = name_str[:MAX_NAME_LENGTH]
            clamped_cmd = cmd_str[:MAX_COMMAND_LENGTH]

            parsed[clamped_key] = RemoteCommand(
                key=clamped_key,
                name=clamped_name,
                command=clamped_cmd,
            )

    return parsed


# =====================================================================
# Packet Factory Helpers
# =====================================================================

def create_request_commands_packet(legacy: bool = False) -> Packet:
    """Construct an outbound packet requesting remote device to advertise its command list.

    Args:
        legacy: If True, uses kdeconnect.remotecommands.request; otherwise kdeconnect.runcommand.request.
    """
    pkt_type = (
        "kdeconnect.remotecommands.request"
        if legacy
        else "kdeconnect.runcommand.request"
    )
    return Packet(type=pkt_type, body={"requestCommandList": True})


def create_execute_command_packet(
    key: str,
    packet_type: str = "kdeconnect.remoteforcecommand",
) -> Packet:
    """Construct an outbound packet requesting execution of a configured remote command.

    Args:
        key: Unique command key string configured on the remote peer.
        packet_type: kdeconnect.remoteforcecommand (direct) or kdeconnect.runcommand.request.
    """
    if not isinstance(key, str) or not key.strip():
        raise RemoteCommandError("Command key must be a non-empty string")
    return Packet(type=packet_type, body={"key": key.strip()})


# =====================================================================
# RemoteCommandsPlugin Implementation
# =====================================================================

class RemoteCommandsPlugin(Plugin):
    """KDE Connect RemoteCommands client plugin.

    Discovers available commands on paired devices and triggers remote execution.
    Acts strictly as a remote controller client with zero local shell execution.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        packet_sender: Optional[SendPacketCallable] = None,
        is_paired_checker: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.event_bus = event_bus
        self.packet_sender = packet_sender
        self.is_paired_checker = is_paired_checker
        self._device_commands: Dict[str, DeviceRemoteCommands] = {}
        self._connected_senders: Dict[str, SendPacketCallable] = {}
        self._lock = asyncio.Lock()

    @property
    def plugin_id(self) -> str:
        return "remotecommands"

    @property
    def incoming_capabilities(self) -> List[str]:
        return [
            "kdeconnect.remotecommands",
            "kdeconnect.runcommand",
            "kdeconnect.runcommand.output",
        ]

    @property
    def outgoing_capabilities(self) -> List[str]:
        return [
            "kdeconnect.remotecommands.request",
            "kdeconnect.remoteforcecommand",
            "kdeconnect.runcommand.request",
        ]

    async def start(self) -> None:
        """Subscribe to lifecycle events on start."""
        if self.event_bus:
            self.event_bus.subscribe(DeviceUnpaired, self._on_device_unpaired_event)

    async def stop(self) -> None:
        """Unsubscribe from events and clear cached states on stop."""
        if self.event_bus:
            self.event_bus.unsubscribe(DeviceUnpaired, self._on_device_unpaired_event)
        async with self._lock:
            self._device_commands.clear()
            self._connected_senders.clear()

    async def _on_device_unpaired_event(self, event: DeviceUnpaired) -> None:
        """Purge remote command caches when a device is unpaired."""
        if event and event.device_id:
            self.on_device_unpaired(event.device_id)

    def on_device_unpaired(self, device_id: str) -> None:
        """Synchronously purge device state upon unpairing."""
        self._device_commands.pop(device_id, None)
        self._connected_senders.pop(device_id, None)
        logger.debug("Purged remote command state for unpaired device '%s'", device_id)

    def _resolve_sender(
        self,
        device_id: str,
        explicit_sender: Optional[SendPacketCallable] = None,
    ) -> SendPacketCallable:
        """Resolve packet sender callback for device."""
        if explicit_sender:
            return explicit_sender
        if device_id in self._connected_senders:
            return self._connected_senders[device_id]
        if self.packet_sender:
            return lambda pkt: self.packet_sender(device_id, pkt)
        raise ConnectionClosedError(f"No active connection to device '{device_id}'")

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Route incoming remote commands packets and publish corresponding events.

        CRITICAL SECURITY INVARIANT:
        Never execute local subprocesses or shell commands under any circumstance.
        """
        if self.is_paired_checker is not None:
            check_res = self.is_paired_checker(device_id)
            if inspect.isawaitable(check_res):
                check_res = await check_res
            if not check_res:
                logger.warning(
                    "RemoteCommandsPlugin rejecting packet from unauthenticated/unpaired device '%s'",
                    device_id,
                )
                return True

        self._connected_senders[device_id] = send_packet
        body = packet.body or {}

        # Discard execution requests from remote peers (zero local execution invariant)
        if packet.type in (
            "kdeconnect.runcommand.request",
            "kdeconnect.remotecommands.request",
            "kdeconnect.remoteforcecommand",
        ):
            if "key" in body and not body.get("requestCommandList"):
                logger.debug(
                    "Ignored incoming execution request from '%s' (RemoteCommands is client-only)",
                    device_id,
                )
                return

        # Handle command list packets
        if packet.type in ("kdeconnect.runcommand", "kdeconnect.remotecommands"):
            parsed_commands = parse_command_list(body)
            async with self._lock:
                dev_cmds = DeviceRemoteCommands(
                    device_id=device_id,
                    commands=parsed_commands,
                    last_updated=time.time(),
                )
                self._device_commands[device_id] = dev_cmds

            if self.event_bus:
                await self.event_bus.publish(
                    RemoteCommandsReceivedEvent(
                        device_id=device_id,
                        commands={k: c.to_dict() for k, c in parsed_commands.items()},
                        count=len(parsed_commands),
                    )
                )

        # Handle command output feedback packets
        elif packet.type == "kdeconnect.runcommand.output":
            key = str(body.get("key", ""))
            raw_output = body.get("output") or body.get("result") or body.get("stdout") or ""
            if self.is_paired_checker is not None:
                output_str = str(raw_output)[:65536]
            else:
                output_str = str(raw_output)

            raw_code = body.get("exitCode") or body.get("exit_code") or body.get("status") or 0
            try:
                exit_code = int(raw_code)
            except (ValueError, TypeError):
                exit_code = 0

            if self.event_bus:
                await self.event_bus.publish(
                    RemoteCommandOutputEvent(
                        device_id=device_id,
                        key=key,
                        output=output_str,
                        exit_code=exit_code,
                    )
                )

    def get_commands(self, device_id: str) -> List[RemoteCommand]:
        """Return list of cached RemoteCommand instances for device."""
        dev = self._device_commands.get(device_id)
        return list(dev.commands.values()) if dev else []

    def get_command(self, device_id: str, key: str) -> Optional[RemoteCommand]:
        """Retrieve a specific cached RemoteCommand by key."""
        dev = self._device_commands.get(device_id)
        return dev.get(key) if dev else None

    async def request_command_list(
        self,
        device_id: str,
        send_packet: Optional[SendPacketCallable] = None,
        legacy: bool = False,
    ) -> Packet:
        """Transmit request to remote peer to advertise its configured command list."""
        sender = self._resolve_sender(device_id, send_packet)
        pkt = create_request_commands_packet(legacy=legacy)
        await safe_send_packet(sender, pkt)
        return pkt

    async def execute_command(
        self,
        device_id: str,
        key: str,
        send_packet: Optional[SendPacketCallable] = None,
        packet_type: str = "kdeconnect.remoteforcecommand",
    ) -> Packet:
        """Transmit command execution request to remote device.

        Raises:
            RemoteCommandError: If key is empty or invalid.
            ConnectionClosedError: If device has no active sender.
        """
        if not key or not str(key).strip():
            raise RemoteCommandError("Command key must be a non-empty string")

        sender = self._resolve_sender(device_id, send_packet)
        pkt = create_execute_command_packet(key=key, packet_type=packet_type)
        await safe_send_packet(sender, pkt)

        if self.event_bus:
            await self.event_bus.publish(
                RemoteCommandExecutedEvent(
                    device_id=device_id,
                    key=key.strip(),
                )
            )

        return pkt


__all__ = [
    "MAX_KEY_LENGTH",
    "MAX_NAME_LENGTH",
    "MAX_COMMAND_LENGTH",
    "MAX_COMMANDS_PER_DEVICE",
    "RemoteCommand",
    "DeviceRemoteCommands",
    "parse_command_list",
    "create_request_commands_packet",
    "create_execute_command_packet",
    "RemoteCommandsPlugin",
]

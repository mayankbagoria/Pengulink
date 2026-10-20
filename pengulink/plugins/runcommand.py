"""pengulink.plugins.runcommand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect RunCommand Host Plugin for local command execution.

Implements:
- Server / Host role: Advertises local host commands to paired peers (kdeconnect.runcommand)
- Listens for remote trigger requests (kdeconnect.runcommand.request)
- Streams 3-phase execution feedback (kdeconnect.runcommand.output):
    Phase A: commandStarted (started, command, key, id)
    Phase B: commandOutput (output, stdout, stderr, key, id)
    Phase C: commandFinished (finished, success, exitCode, key, output, id)
- Enforces strict security invariants:
    1. Direct execution via shlex.split without shell=True
    2. Zero remote string evaluation (never execute remote 'command' or 'cmd' payloads)
    3. Strict local whitelist dictionary lookup by key
    4. Unpaired / unauthenticated peer packet dropping
    5. Clean process group termination on stop packet or unpair
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from typing import Any

from pengulink.core.events import (
    DeviceConnected,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
)
from pengulink.core.exceptions import PenguLinkError
from pengulink.network.packet import Packet
from pengulink.plugins.base import Plugin, SendPacketCallable, safe_send_packet
from pengulink.services.command import (
    CommandEntry,
    CommandStore,
    LocalCommandService,
)

logger = logging.getLogger("pengulink.plugins.runcommand")

RUNCOMMAND_PACKET_TYPE = "kdeconnect.runcommand"
RUNCOMMAND_REQUEST_TYPE = "kdeconnect.runcommand.request"
RUNCOMMAND_OUTPUT_TYPE = "kdeconnect.runcommand.output"
REMOTE_FORCE_COMMAND_TYPE = "kdeconnect.remoteforcecommand"


class RunCommandPlugin(Plugin):
    """Host plugin exposing locally configured commands to paired peers."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        command_store: CommandStore | None = None,
        command_service: LocalCommandService | None = None,
        packet_sender: Any | None = None,
        is_paired_checker: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(event_bus)
        self.command_store = command_store or CommandStore()
        self.command_service = command_service or LocalCommandService(
            event_bus=event_bus, command_store=self.command_store
        )
        self.packet_sender = packet_sender
        self.is_paired_checker = is_paired_checker

        self._connected_peers: dict[str, SendPacketCallable] = {}
        self._device_running_processes: dict[str, set[str]] = {}
        self._device_active_tasks: dict[str, set[asyncio.Task[Any]]] = {}
        self._unsub_callbacks: list[Callable[[], None]] = []
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self._next_process_id: int = 1

    def _generate_process_id(self) -> int:
        """Generate next monotonic integer process ID for wire packets."""
        pid = self._next_process_id
        self._next_process_id += 1
        return pid

    @property
    def plugin_id(self) -> str:
        return "runcommand"

    @property
    def incoming_capabilities(self) -> list[str]:
        return [RUNCOMMAND_REQUEST_TYPE, REMOTE_FORCE_COMMAND_TYPE]

    @property
    def outgoing_capabilities(self) -> list[str]:
        return [RUNCOMMAND_PACKET_TYPE, RUNCOMMAND_OUTPUT_TYPE]

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
        """Tear down subscriptions, terminate running processes, and clean up."""
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
                logger.debug("Error unsubscribing event in RunCommandPlugin: %s", exc)
        self._unsub_callbacks.clear()

        # Stop active running processes
        await self.command_service.stop_all()

        # Cancel any active background streaming tasks
        for task in list(self._bg_tasks):
            if not task.done():
                task.cancel()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
            self._bg_tasks.clear()

        self._connected_peers.clear()
        self._device_running_processes.clear()
        self._device_active_tasks.clear()
        await super().stop()

    def _create_command_list_packet(self) -> Packet:
        """Create kdeconnect.runcommand packet conforming to Android stringified JSON requirement."""
        wire_dict = self.command_store.to_wire_dict()
        return Packet(
            type=RUNCOMMAND_PACKET_TYPE,
            body={
                "commandList": json.dumps(wire_dict),
                "canAddCommand": True,
            },
        )

    async def send_command_list(
        self,
        device_id: str | None = None,
        send_packet: SendPacketCallable | None = None,
    ) -> None:
        """Transmit current commandList to a device or via provided send_packet."""
        pkt = self._create_command_list_packet()

        if send_packet is not None:
            try:
                await safe_send_packet(send_packet, pkt)
                return
            except (
                OSError,
                ConnectionError,
                RuntimeError,
                PenguLinkError,
                asyncio.TimeoutError,
            ) as exc:
                logger.debug("Failed sending command list via callback: %s", exc)

        if device_id and device_id in self._connected_peers:
            sender = self._connected_peers[device_id]
            try:
                await safe_send_packet(sender, pkt)
                return
            except (
                OSError,
                ConnectionError,
                RuntimeError,
                PenguLinkError,
                asyncio.TimeoutError,
            ) as exc:
                logger.debug(
                    "Failed sending command list to peer %s: %s", device_id, exc
                )

        if self.packet_sender and device_id:
            try:
                try:
                    res = self.packet_sender(device_id, pkt)
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
                    "Failed sending command list to %s via packet_sender: %s",
                    device_id,
                    exc,
                )

    async def broadcast_command_list(self) -> None:
        """Broadcast current commandList to all connected peers."""
        pkt = self._create_command_list_packet()
        peers_snapshot = list(self._connected_peers.items())
        for peer_id, sender in peers_snapshot:
            try:
                await safe_send_packet(sender, pkt)
            except (
                OSError,
                ConnectionError,
                RuntimeError,
                PenguLinkError,
                asyncio.TimeoutError,
            ) as exc:
                logger.debug("Failed broadcasting commandList to %s: %s", peer_id, exc)

    async def _on_device_connected(self, event: DeviceConnected) -> None:
        """Broadcast command list when a paired device connects."""
        if not event.device_id:
            return
        await self.send_command_list(device_id=event.device_id)

    async def _on_device_disconnected(self, event: DeviceDisconnected) -> None:
        """Clean up peer connection tracking."""
        self._connected_peers.pop(event.device_id, None)

    async def on_device_unpaired(self, device_id: str) -> None:
        """Clean up peer state and terminate active processes for device_id."""
        self._connected_peers.pop(device_id, None)
        await self.stop_device_processes(device_id)

    async def _on_device_unpaired(self, event: DeviceUnpaired) -> None:
        """Clean up peer state and terminate active processes launched by that device."""
        await self.on_device_unpaired(event.device_id)

    async def stop_device_processes(self, device_id: str) -> None:
        """Terminate all child processes and cancel execution tasks for a specific device."""
        tasks = self._device_active_tasks.pop(device_id, set())
        for task in tasks:
            if not task.done():
                task.cancel()

        pids = self._device_running_processes.pop(device_id, set())
        for pid in pids:
            try:
                await self.command_service.stop_process(pid)
            except (
                ProcessLookupError,
                OSError,
                asyncio.TimeoutError,
                RuntimeError,
            ) as exc:
                logger.debug(
                    "Error stopping process %s for device %s: %s", pid, device_id, exc
                )

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Process incoming RunCommand request packet."""
        # Security check: verify peer authentication / pairing
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
                logger.error("Error checking pairing status for %s: %s", device_id, exc)
                is_paired = False
            if not is_paired:
                logger.warning(
                    "Rejecting RunCommand packet from unauthenticated/unpaired device: %s",
                    device_id,
                )
                return

        if packet.type not in self.incoming_capabilities:
            logger.debug("Ignoring packet with unsupported type: %s", packet.type)
            return

        # Peer is authenticated and capability supported: record peer sender
        self._connected_peers[device_id] = send_packet

        body = packet.body
        if not isinstance(body, dict):
            if body is not None:
                logger.warning(
                    "Rejecting RunCommand packet with non-dict body from %s",
                    device_id,
                )
            return

        # 1. requestCommandList: Query available commands
        if body.get("requestCommandList"):
            logger.debug("Device %s requested command list enumeration", device_id)
            await self.send_command_list(device_id=device_id, send_packet=send_packet)
            return

        # 2. stop: Stop currently active running processes for this device
        if body.get("stop"):
            logger.info("Device %s requested stop of running processes", device_id)
            await self.stop_device_processes(device_id)
            return

        # 3. setup: Desktop configuration UI request (safe no-op in headless core)
        if body.get("setup"):
            logger.debug(
                "Device %s requested setup UI (ignored in headless core)",
                device_id,
            )
            return

        # 4. key: Trigger execution of locally configured command
        if "key" in body and body["key"] is not None:
            key = str(body["key"]).strip()
            entry = self.command_store.get_command(key)
            if not entry:
                logger.warning(
                    "Command key '%s' requested by %s is not in local store. Rejection enforced.",
                    key,
                    device_id,
                )
                return
            if not entry.enabled:
                logger.warning(
                    "Command key '%s' requested by %s is disabled. Execution prevented.",
                    key,
                    device_id,
                )
                return

            # Spawn async execution task to avoid blocking packet loop
            task = asyncio.create_task(
                self._execute_and_stream(
                    entry=entry,
                    device_id=device_id,
                    send_packet=send_packet,
                )
            )
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

            if device_id not in self._device_active_tasks:
                self._device_active_tasks[device_id] = set()
            self._device_active_tasks[device_id].add(task)
            task.add_done_callback(
                lambda t, did=device_id: self._device_active_tasks.get(
                    did, set()
                ).discard(t)
            )
            return

        # 5. Security Guard: Discard arbitrary remote command strings
        if "command" in body or "cmd" in body:
            logger.warning(
                "SECURITY GUARD: Ignored remote command payload from %s (command=%r, cmd=%r). "
                "Only locally configured keys are executable.",
                device_id,
                body.get("command"),
                body.get("cmd"),
            )
            return

    async def _execute_and_stream(
        self,
        entry: CommandEntry,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Execute local command and stream 3-phase output packets."""
        proc_id = self._generate_process_id()
        proc_id_str = str(proc_id)
        if device_id not in self._device_running_processes:
            self._device_running_processes[device_id] = set()
        self._device_running_processes[device_id].add(proc_id_str)

        # Phase A: commandStarted
        start_pkt = Packet(
            type=RUNCOMMAND_OUTPUT_TYPE,
            body={
                "commandStarted": True,
                "command": entry.command,
                "key": entry.key,
                "id": proc_id,
            },
        )
        try:
            await safe_send_packet(send_packet, start_pkt)
        except (
            OSError,
            ConnectionError,
            RuntimeError,
            PenguLinkError,
            asyncio.TimeoutError,
        ) as exc:
            logger.debug(
                "Failed sending commandStarted packet to %s: %s", device_id, exc
            )

        # Concurrency race guard: check if process or device was stopped during yield
        if (
            device_id not in self._device_running_processes
            or proc_id_str not in self._device_running_processes[device_id]
        ):
            logger.info(
                "Process %s for device %s was stopped prior to spawn; aborting",
                proc_id,
                device_id,
            )
            return

        # Callback for Phase B: commandOutput
        async def _on_output(
            key: str,
            stdout_lines: list[str],
            stderr_lines: list[str],
            pid: str,
        ) -> None:
            if not stdout_lines and not stderr_lines:
                return
            out_pkt = Packet(
                type=RUNCOMMAND_OUTPUT_TYPE,
                body={
                    "commandOutput": True,
                    "stdout": stdout_lines,
                    "stderr": stderr_lines,
                    "key": entry.key,
                    "id": proc_id,
                },
            )
            try:
                await safe_send_packet(send_packet, out_pkt)
            except (
                OSError,
                ConnectionError,
                RuntimeError,
                PenguLinkError,
                asyncio.TimeoutError,
            ) as exc:
                logger.debug(
                    "Failed sending commandOutput packet to %s: %s", device_id, exc
                )

        exit_code = 1
        success = False
        combined_output = ""
        try:
            result = await self.command_service.execute_command_entry(
                entry=entry,
                device_id=device_id,
                process_id=proc_id_str,
                send_output=_on_output,
            )
            exit_code = result.exit_code
            success = result.success
            combined_output = "\n".join(result.stdout + result.stderr)
        except asyncio.CancelledError:
            logger.info("Command execution task cancelled for key '%s'", entry.key)
            exit_code = 1
            success = False
            combined_output = "Task cancelled"
            raise
        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            asyncio.TimeoutError,
            PenguLinkError,
        ) as exc:
            logger.error("Exception during command execution '%s': %s", entry.key, exc)
            exit_code = 1
            success = False
            combined_output = str(exc)
        finally:
            if device_id in self._device_running_processes:
                self._device_running_processes[device_id].discard(proc_id_str)

        # Phase C: commandFinished
        finish_pkt = Packet(
            type=RUNCOMMAND_OUTPUT_TYPE,
            body={
                "commandFinished": True,
                "success": success,
                "exitCode": exit_code,
                "key": entry.key,
                "output": combined_output,
                "id": proc_id,
            },
        )
        try:
            await safe_send_packet(send_packet, finish_pkt)
        except (
            OSError,
            ConnectionError,
            RuntimeError,
            PenguLinkError,
            asyncio.TimeoutError,
        ) as exc:
            logger.debug(
                "Failed sending commandFinished packet to %s: %s",
                device_id,
                exc,
            )


__all__ = [
    "REMOTE_FORCE_COMMAND_TYPE",
    "RUNCOMMAND_OUTPUT_TYPE",
    "RUNCOMMAND_PACKET_TYPE",
    "RUNCOMMAND_REQUEST_TYPE",
    "RunCommandPlugin",
]

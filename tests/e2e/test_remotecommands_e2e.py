"""tests.e2e.test_remotecommands_e2e
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Live end-to-end integration test suite for Milestone 15: RemoteCommands Plugin
running against MockKdeConnectDevice across loopback with live mTLS and JSON-RPC IPC.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, List
import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.events import RemoteCommandOutputEvent, RemoteCommandsReceivedEvent
from pengulink.core.exceptions import IPCError
from pengulink.network.packet import Packet
from pengulink.daemon.service import PenguLinkDaemon
from tests.e2e.conftest import e2e_environment, wait_for
from tests.mock.mock_device import MockDevice


async def _pair_and_connect(daemon: PenguLinkDaemon, mock: MockDevice) -> None:
    """Establish mutual UDP discovery, mTLS handshake, and pairing."""
    await mock.broadcast_identity(
        target_ip="127.0.0.1",
        target_port=daemon.config.network.udp_port,
    )
    await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)
    res = await daemon.pair_device(mock.device_id, wait=True, timeout=5.0)
    assert res["paired"] is True
    assert daemon.is_device_connected(mock.device_id) is True


# =============================================================================
# E2E Remote Commands Test Suite
# =============================================================================

async def test_e2e_remotecommands_001_discovery_and_capability_exchange(tmp_path: Path) -> None:
    """1. Verify discovery, pairing, and mutual RemoteCommands capability negotiation."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Verify daemon capabilities
        inc = daemon.config.device.incoming_capabilities
        outg = daemon.config.device.outgoing_capabilities
        assert "kdeconnect.runcommand" in inc
        assert "kdeconnect.remotecommands" in inc
        assert "kdeconnect.remoteforcecommand" in outg
        assert "kdeconnect.runcommand.request" in outg


async def test_e2e_remotecommands_002_mock_advertises_dict_format_command_list(tmp_path: Path) -> None:
    """2. Mock device advertises commands as native JSON dictionary."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Track event emission
        events: List[RemoteCommandsReceivedEvent] = []
        daemon.event_bus.subscribe(RemoteCommandsReceivedEvent, lambda ev: events.append(ev))

        # Mock transmits staged commands
        await mock.send_command_list(format="dict")

        # Wait for daemon to ingest and publish event
        await wait_for(lambda: len(events) > 0, timeout=4.0)
        assert events[0].count == 3

        # Query IPC
        cmds = await ipc_client.list_remote_commands(mock.device_id)
        assert len(cmds) == 3
        keys = {c["key"] for c in cmds}
        assert keys == {"cmd_lock", "cmd_suspend", "cmd_ring"}


async def test_e2e_remotecommands_003_mock_advertises_string_format_command_list(tmp_path: Path) -> None:
    """3. Mock device advertises commands as JSON-escaped string (C++/Qt convention)."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        events: List[RemoteCommandsReceivedEvent] = []
        daemon.event_bus.subscribe(RemoteCommandsReceivedEvent, lambda ev: events.append(ev))

        # Mock transmits staged commands as stringified JSON
        await mock.send_command_list(format="string")

        await wait_for(lambda: len(events) > 0, timeout=4.0)
        assert events[0].count == 3

        cmds = await ipc_client.list_remote_commands(mock.device_id)
        assert len(cmds) == 3
        cmd_map = {c["key"]: c for c in cmds}
        assert cmd_map["cmd_lock"]["name"] == "Lock Screen"
        assert cmd_map["cmd_suspend"]["name"] == "Suspend System"


async def test_e2e_remotecommands_004_cli_command_list_rich_and_json(tmp_path: Path) -> None:
    """4. CLI command list renders Rich table and machine-readable JSON."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        await mock.send_command_list(format="dict")
        await wait_for(
            lambda: len(daemon.remote_commands_plugin.get_commands(mock.device_id)) > 0,
            timeout=4.0,
        )

        runner = CliRunner(env={"COLUMNS": "160"})
        sock_arg = ["-s", str(daemon.ipc_server.socket_path)]

        # Rich Table Output
        res = await asyncio.to_thread(runner.invoke, cli, sock_arg + ["command", "list", mock.device_id])
        assert res.exit_code == 0
        assert "Remote Commands" in res.output
        assert "cmd_lock" in res.output
        assert "Lock Screen" in res.output

        # JSON Output
        res_json = await asyncio.to_thread(runner.invoke, cli, sock_arg + ["command", "list", mock.device_id, "--json"])
        assert res_json.exit_code == 0
        parsed = json.loads(res_json.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 3


async def test_e2e_remotecommands_005_cli_command_exec_triggers_wire_packet(tmp_path: Path) -> None:
    """5. CLI command exec triggers outbound wire packet arriving at mock device."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        await mock.send_command_list(format="dict")
        await wait_for(
            lambda: len(daemon.remote_commands_plugin.get_commands(mock.device_id)) > 0,
            timeout=4.0,
        )

        runner = CliRunner(env={"COLUMNS": "160"})
        sock_arg = ["-s", str(daemon.ipc_server.socket_path)]

        # Exec cmd_lock
        res = await asyncio.to_thread(runner.invoke, cli, sock_arg + ["command", "exec", mock.device_id, "cmd_lock"])
        assert res.exit_code == 0
        assert "Remote Command Executed" in res.output

        # Verify mock device received the execution packet
        pkt = await mock.wait_for_command_exec("cmd_lock", timeout=5.0)
        assert pkt.body.get("key") == "cmd_lock"
        assert "cmd_lock" in mock.executed_commands


async def test_e2e_remotecommands_006_cli_command_run_alias(tmp_path: Path) -> None:
    """6. CLI command run alias triggers execution packet identically to exec."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        await mock.send_command_list(format="dict")
        await wait_for(
            lambda: len(daemon.remote_commands_plugin.get_commands(mock.device_id)) > 0,
            timeout=4.0,
        )

        runner = CliRunner(env={"COLUMNS": "160"})
        sock_arg = ["-s", str(daemon.ipc_server.socket_path)]

        # Run cmd_suspend
        res = await asyncio.to_thread(runner.invoke, cli, sock_arg + ["command", "run", mock.device_id, "cmd_suspend"])
        assert res.exit_code == 0
        assert "Remote Command Executed" in res.output

        pkt = await mock.wait_for_command_exec("cmd_suspend", timeout=5.0)
        assert pkt.body.get("key") == "cmd_suspend"
        assert "cmd_suspend" in mock.executed_commands


async def test_e2e_remotecommands_007_ipc_exec_with_command_output_feedback(tmp_path: Path) -> None:
    """7. Remote device emits command output packet; daemon publishes event."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        # Configure mock device to automatically send execution output feedback
        mock.config.auto_command_output = True
        await _pair_and_connect(daemon, mock)
        await mock.send_command_list(format="dict")

        events: List[RemoteCommandOutputEvent] = []
        daemon.event_bus.subscribe(RemoteCommandOutputEvent, lambda ev: events.append(ev))

        # Execute command via IPC
        res = await ipc_client.execute_remote_command(mock.device_id, "cmd_ring")
        assert res["status"] == "transmitted"

        # Wait for output packet arrival at daemon
        await wait_for(lambda: len(events) > 0, timeout=4.0)
        assert events[0].key == "cmd_ring"
        assert events[0].exit_code == 0
        assert "Executed 'cmd_ring' successfully" in events[0].output


async def test_e2e_remotecommands_008_daemon_explicit_refresh_query(tmp_path: Path) -> None:
    """8. Daemon explicitly queries mock device to advertise command list."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Trigger refresh query
        res = await ipc_client.request_remote_commands(mock.device_id)
        assert res["status"] == "requested"

        # Mock should auto-reply with command list
        await wait_for(
            lambda: len(daemon.remote_commands_plugin.get_commands(mock.device_id)) > 0,
            timeout=4.0,
        )
        cmds = await ipc_client.list_remote_commands(mock.device_id)
        assert len(cmds) == 3


async def test_e2e_remotecommands_009_unpair_purges_remote_command_cache(tmp_path: Path) -> None:
    """9. Explicit unpairing immediately purges all stored commands for the device."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        await mock.send_command_list(format="dict")

        await wait_for(
            lambda: len(daemon.remote_commands_plugin.get_commands(mock.device_id)) > 0,
            timeout=4.0,
        )

        # Unpair device
        unpair_res = await daemon.unpair_device(mock.device_id)
        assert unpair_res["success"] is True

        # Cache should be wiped
        assert len(daemon.remote_commands_plugin.get_commands(mock.device_id)) == 0

        # Attempting execution should fail because device is not connected/unpaired
        with pytest.raises(IPCError):
            await ipc_client.execute_remote_command(mock.device_id, "cmd_lock")


async def test_e2e_remotecommands_010_rich_markup_adversarial_roundtrip(tmp_path: Path) -> None:
    """10. Malicious Rich markup in command names/commands rendered safely."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        hostile_cmds = {
            "cmd_pwn": {
                "key": "cmd_pwn",
                "name": "[bold red]System Exploit[/bold red]",
                "command": "[link=http://evil.com]rm -rf /[/link]",
            }
        }
        await mock.send_command_list(commands=hostile_cmds, format="dict")

        await wait_for(
            lambda: len(daemon.remote_commands_plugin.get_commands(mock.device_id)) > 0,
            timeout=4.0,
        )

        runner = CliRunner(env={"COLUMNS": "160"})
        sock_arg = ["-s", str(daemon.ipc_server.socket_path)]
        res = await asyncio.to_thread(runner.invoke, cli, sock_arg + ["command", "list", mock.device_id])
        assert res.exit_code == 0
        assert "System Exploit" in res.output

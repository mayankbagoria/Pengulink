"""tests.unit.test_cli_remotecommands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Milestone 15: RemoteCommands JSON-RPC IPC server/client
and Click/Rich CLI (pengulink command [list|exec|run]).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.exceptions import (
    ConnectionClosedError,
    DeviceNotFoundError,
    IPCError,
    RemoteCommandError,
    RemoteCommandNotFoundError,
)
from pengulink.daemon.ipc import (
    DEVICE_NOT_CONNECTED,
    DEVICE_NOT_FOUND,
    REMOTE_COMMANDS_ERROR,
    PenguLinkIPCClient,
    PenguLinkIPCServer,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner(env={"COLUMNS": "160"})


# =====================================================================
# 1. JSON-RPC IPC Server & Client Integration
# =====================================================================

class TestRemoteCommandsIPCServerAndClient:
    """Test IPC endpoints, client methods, error codes, and serialization."""

    async def test_ipc_list_remote_commands(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_remotecommands.sock"

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            if method == "list_remote_commands":
                assert params["device_id"] == "phone_1"
                return [
                    {"key": "cmd_lock", "name": "Lock Screen", "command": "loginctl lock-session"},
                    {"key": "cmd_suspend", "name": "Suspend System", "command": "systemctl suspend"},
                ]
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            res = await client.list_remote_commands("phone_1")
            assert len(res) == 2
            assert res[0]["key"] == "cmd_lock"
            assert res[0]["name"] == "Lock Screen"
            assert res[1]["key"] == "cmd_suspend"

            # Alias test
            res_alias = await client.remotecommands_list("phone_1")
            assert len(res_alias) == 2
        finally:
            await server.stop()

    async def test_ipc_request_remote_commands(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_request.sock"

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            if method == "request_remote_commands":
                assert params["device_id"] == "phone_1"
                return {
                    "device_id": "phone_1",
                    "status": "requested",
                    "message": "Command list request transmitted to device 'phone_1'",
                }
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            res = await client.request_remote_commands("phone_1")
            assert res["status"] == "requested"
            assert res["device_id"] == "phone_1"
        finally:
            await server.stop()

    async def test_ipc_execute_remote_command(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_exec.sock"

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            if method == "execute_remote_command":
                assert params["device_id"] == "phone_1"
                assert params["key"] == "cmd_lock"
                return {
                    "device_id": "phone_1",
                    "key": "cmd_lock",
                    "status": "transmitted",
                    "message": "Execution request for command 'cmd_lock' transmitted",
                }
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            res = await client.execute_remote_command("phone_1", "cmd_lock")
            assert res["status"] == "transmitted"
            assert res["key"] == "cmd_lock"

            # Alias test
            res_alias = await client.remotecommands_run("phone_1", "cmd_lock")
            assert res_alias["status"] == "transmitted"
        finally:
            await server.stop()

    async def test_ipc_remote_command_not_found_error(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_err_not_found.sock"

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            if method == "execute_remote_command":
                raise RemoteCommandNotFoundError(f"Command '{params.get('key')}' not found on device 'phone_1'")
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc_info:
                await client.execute_remote_command("phone_1", "invalid_key")
            assert str(REMOTE_COMMANDS_ERROR) in str(exc_info.value)
            assert "not found" in str(exc_info.value).lower()
        finally:
            await server.stop()

    async def test_ipc_device_not_found_error(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_err_dev.sock"

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            if method == "list_remote_commands":
                raise DeviceNotFoundError("Device 'missing_dev' not found")
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc_info:
                await client.list_remote_commands("missing_dev")
            assert str(DEVICE_NOT_FOUND) in str(exc_info.value)
        finally:
            await server.stop()

    async def test_ipc_device_not_connected_error(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_err_conn.sock"

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            if method == "execute_remote_command":
                raise ConnectionClosedError("Device 'phone_1' is not connected")
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc_info:
                await client.execute_remote_command("phone_1", "cmd_lock")
            assert str(DEVICE_NOT_CONNECTED) in str(exc_info.value)
        finally:
            await server.stop()


# =====================================================================
# 2. Click CLI: pengulink command list
# =====================================================================

class TestCliCommandList:
    """Test pengulink command list command."""

    def test_cli_command_list_table_display(self, cli_runner: CliRunner) -> None:
        mock_commands = [
            {"key": "cmd_lock", "name": "Lock Screen", "command": "loginctl lock-session"},
            {"key": "cmd_suspend", "name": "Suspend System", "command": "systemctl suspend"},
            {"key": "cmd_hidden", "name": "Hidden Script", "command": ""},
        ]
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_commands):
            result = cli_runner.invoke(cli, ["command", "list", "phone_1"])
            assert result.exit_code == 0
            assert "Remote Commands — phone_1" in result.output
            assert "Key" in result.output
            assert "Name" in result.output
            assert "Command" in result.output
            assert "cmd_lock" in result.output
            assert "Lock Screen" in result.output
            assert "loginctl lock-session" in result.output
            assert "cmd_suspend" in result.output
            assert "[hidden]" in result.output

    def test_cli_command_list_empty(self, cli_runner: CliRunner) -> None:
        with patch("pengulink.cli.main.run_ipc_call", return_value=[]):
            result = cli_runner.invoke(cli, ["command", "list", "phone_1"])
            assert result.exit_code == 0
            assert "No remote commands available for device 'phone_1'" in result.output

    def test_cli_command_list_json(self, cli_runner: CliRunner) -> None:
        mock_commands = [
            {"key": "cmd_lock", "name": "Lock Screen", "command": "loginctl lock-session"},
        ]
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_commands):
            result = cli_runner.invoke(cli, ["command", "list", "phone_1", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["key"] == "cmd_lock"

    def test_cli_command_list_missing_device_id(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(cli, ["command", "list", "   "])
        assert result.exit_code != 0

    def test_cli_command_list_device_offline_error(self, cli_runner: CliRunner) -> None:
        with patch("pengulink.cli.main.run_ipc_call", side_effect=IPCError("Device not connected")):
            result = cli_runner.invoke(cli, ["command", "list", "phone_1"])
            assert result.exit_code == 1


# =====================================================================
# 3. Click CLI: pengulink command exec & run
# =====================================================================

class TestCliCommandExecAndRun:
    """Test pengulink command exec and run alias commands."""

    def test_cli_command_exec_success(self, cli_runner: CliRunner) -> None:
        mock_res = {
            "device_id": "phone_1",
            "key": "cmd_lock",
            "status": "transmitted",
            "message": "Execution request transmitted",
        }
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_res):
            result = cli_runner.invoke(cli, ["command", "exec", "phone_1", "cmd_lock"])
            assert result.exit_code == 0
            assert "Remote Command Executed" in result.output
            assert "cmd_lock" in result.output
            assert "phone_1" in result.output

    def test_cli_command_run_alias(self, cli_runner: CliRunner) -> None:
        mock_res = {
            "device_id": "phone_1",
            "key": "cmd_lock",
            "status": "transmitted",
            "message": "Execution request transmitted",
        }
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_res):
            result = cli_runner.invoke(cli, ["command", "run", "phone_1", "cmd_lock"])
            assert result.exit_code == 0
            assert "Remote Command Executed" in result.output
            assert "cmd_lock" in result.output
            assert "phone_1" in result.output

    def test_cli_command_exec_json(self, cli_runner: CliRunner) -> None:
        mock_res = {
            "device_id": "phone_1",
            "key": "cmd_lock",
            "status": "transmitted",
            "message": "Execution request transmitted",
        }
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_res):
            result = cli_runner.invoke(cli, ["command", "exec", "phone_1", "cmd_lock", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["status"] == "transmitted"
            assert data["key"] == "cmd_lock"

    def test_cli_command_exec_missing_args(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(cli, ["command", "exec", "phone_1"])
        assert result.exit_code == 2

    def test_cli_command_exec_offline_failure(self, cli_runner: CliRunner) -> None:
        with patch("pengulink.cli.main.run_ipc_call", side_effect=IPCError("Device not connected")):
            result = cli_runner.invoke(cli, ["command", "exec", "phone_1", "cmd_lock"])
            assert result.exit_code == 1


# =====================================================================
# 4. Rich Markup Protection
# =====================================================================

class TestCliRichMarkupProtection:
    """Test immunity against Rich markup injection in untrusted device strings."""

    def test_cli_markup_escape_in_command_list(self, cli_runner: CliRunner) -> None:
        hostile_commands = [
            {
                "key": "[bold red]evil_key[/bold red]",
                "name": "[link=http://attacker.com]Malicious Link[/link]",
                "command": "[italic]malicious_cmd[/italic]",
            }
        ]
        with patch("pengulink.cli.main.run_ipc_call", return_value=hostile_commands):
            result = cli_runner.invoke(cli, ["command", "list", "phone_1"])
            assert result.exit_code == 0
            # Verifies string is escaped rather than crashing Rich console parser
            assert "evil_key" in result.output

    def test_cli_markup_escape_in_command_exec(self, cli_runner: CliRunner) -> None:
        mock_res = {
            "device_id": "phone_[red]test[/red]",
            "key": "[blink]cmd_hack[/blink]",
            "status": "transmitted",
        }
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_res):
            result = cli_runner.invoke(cli, ["command", "exec", "phone_[red]test[/red]", "[blink]cmd_hack[/blink]"])
            assert result.exit_code == 0
            assert "cmd_hack" in result.output

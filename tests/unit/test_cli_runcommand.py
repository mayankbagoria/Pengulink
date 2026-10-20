"""tests.unit.test_cli_runcommand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Milestone 25: Click/Rich CLI runcommand command group
(pengulink runcommand [list|add|remove|run]).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.exceptions import IPCError


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner(env={"COLUMNS": "160"})


# =====================================================================
# 1. runcommand list
# =====================================================================


class TestRuncommandList:
    """Test pengulink runcommand list command."""

    def test_runcommand_list_table_display(self, cli_runner: CliRunner) -> None:
        mock_commands = [
            {
                "key": "cmd_lock",
                "name": "Lock Screen",
                "command": "loginctl lock-session",
                "enabled": True,
            },
            {
                "key": "cmd_suspend",
                "name": "Suspend System",
                "command": "systemctl suspend",
                "enabled": False,
            },
        ]
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value=mock_commands
        ) as mock_ipc:
            result = cli_runner.invoke(cli, ["runcommand", "list"])
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0], "runcommand_list"
            )
            assert "Local Commands" in result.output
            assert "cmd_lock" in result.output
            assert "Lock Screen" in result.output
            assert "loginctl lock-session" in result.output
            assert "cmd_suspend" in result.output
            assert "Yes" in result.output
            assert "No" in result.output

    def test_runcommand_list_empty(self, cli_runner: CliRunner) -> None:
        with patch("pengulink.cli.main.run_ipc_call", return_value=[]):
            result = cli_runner.invoke(cli, ["runcommand", "list"])
            assert result.exit_code == 0
            assert "No local commands configured" in result.output

    def test_runcommand_list_json(self, cli_runner: CliRunner) -> None:
        mock_commands = [
            {"key": "k1", "name": "Test", "command": "echo test", "enabled": True}
        ]
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_commands):
            result = cli_runner.invoke(cli, ["runcommand", "list", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["key"] == "k1"

    def test_runcommand_list_ipc_error(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", side_effect=IPCError("Daemon error")
        ):
            result = cli_runner.invoke(cli, ["runcommand", "list"])
            assert result.exit_code == 1


# =====================================================================
# 2. runcommand add
# =====================================================================


class TestRuncommandAdd:
    """Test pengulink runcommand add command."""

    def test_runcommand_add_success(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call",
            return_value={
                "key": "cmd_1234",
                "name": "Lock",
                "command": "loginctl lock-session",
            },
        ) as mock_ipc:
            result = cli_runner.invoke(
                cli, ["runcommand", "add", "Lock", "loginctl lock-session"]
            )
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "runcommand_add",
                {"name": "Lock", "command": "loginctl lock-session"},
            )
            assert "Added command 'Lock' with key 'cmd_1234'" in result.output

    def test_runcommand_add_json(self, cli_runner: CliRunner) -> None:
        mock_res = {"key": "cmd_5678", "name": "Sleep", "command": "systemctl suspend"}
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_res):
            result = cli_runner.invoke(
                cli, ["runcommand", "add", "Sleep", "systemctl suspend", "--json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["key"] == "cmd_5678"
            assert data["name"] == "Sleep"

    def test_runcommand_add_empty_name(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(cli, ["runcommand", "add", "   ", "ls -la"])
        assert result.exit_code != 0
        assert "Argument 'NAME' must not be empty" in result.output

    def test_runcommand_add_empty_command(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(cli, ["runcommand", "add", "Listing", "   "])
        assert result.exit_code != 0
        assert "Argument 'COMMAND' must not be empty" in result.output

    def test_runcommand_add_ipc_error(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", side_effect=IPCError("Write failed")
        ):
            result = cli_runner.invoke(cli, ["runcommand", "add", "Test", "echo 1"])
            assert result.exit_code == 1


# =====================================================================
# 3. runcommand remove
# =====================================================================


class TestRuncommandRemove:
    """Test pengulink runcommand remove command."""

    def test_runcommand_remove_success(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            result = cli_runner.invoke(cli, ["runcommand", "remove", "cmd_lock"])
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0], "runcommand_remove", {"key": "cmd_lock"}
            )
            assert "Removed command 'cmd_lock'" in result.output

    def test_runcommand_remove_not_found(self, cli_runner: CliRunner) -> None:
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": False}):
            result = cli_runner.invoke(cli, ["runcommand", "remove", "nonexistent"])
            assert result.exit_code == 0
            assert "Command 'nonexistent' not found" in result.output

    def test_runcommand_remove_json(self, cli_runner: CliRunner) -> None:
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            result = cli_runner.invoke(
                cli, ["runcommand", "remove", "cmd_lock", "--json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["key"] == "cmd_lock"
            assert data["success"] is True

    def test_runcommand_remove_empty_key(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(cli, ["runcommand", "remove", "   "])
        assert result.exit_code != 0
        assert "Argument 'KEY' must not be empty" in result.output

    def test_runcommand_remove_ipc_error(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", side_effect=IPCError("Daemon error")
        ):
            result = cli_runner.invoke(cli, ["runcommand", "remove", "cmd_lock"])
            assert result.exit_code == 1


# =====================================================================
# 4. runcommand run
# =====================================================================


class TestRuncommandRun:
    """Test pengulink runcommand run command."""

    def test_runcommand_run_success(self, cli_runner: CliRunner) -> None:
        mock_res = {
            "success": True,
            "exit_code": 0,
            "stdout": ["Line 1 of output", "Line 2 of output"],
            "stderr": [],
        }
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value=mock_res
        ) as mock_ipc:
            result = cli_runner.invoke(cli, ["runcommand", "run", "cmd_echo"])
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0], "runcommand_exec", {"key": "cmd_echo"}
            )
            assert "Command succeeded (exit code 0)" in result.output
            assert "Line 1 of output" in result.output
            assert "Line 2 of output" in result.output

    def test_runcommand_run_failure(self, cli_runner: CliRunner) -> None:
        mock_res = {
            "success": False,
            "exit_code": 127,
            "stdout": [],
            "stderr": ["command not found: foobar"],
        }
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_res):
            result = cli_runner.invoke(cli, ["runcommand", "run", "cmd_bad"])
            assert result.exit_code == 0
            assert "Command failed (exit code 127)" in result.output
            assert "command not found: foobar" in result.output

    def test_runcommand_run_json(self, cli_runner: CliRunner) -> None:
        mock_res = {
            "success": True,
            "exit_code": 0,
            "stdout": ["hello"],
            "stderr": [],
        }
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_res):
            result = cli_runner.invoke(cli, ["runcommand", "run", "cmd_echo", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["success"] is True
            assert data["exit_code"] == 0
            assert data["stdout"] == ["hello"]

    def test_runcommand_run_empty_key(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(cli, ["runcommand", "run", "   "])
        assert result.exit_code != 0
        assert "Argument 'KEY' must not be empty" in result.output

    def test_runcommand_run_ipc_error(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", side_effect=IPCError("Command not found")
        ):
            result = cli_runner.invoke(cli, ["runcommand", "run", "cmd_missing"])
            assert result.exit_code == 1

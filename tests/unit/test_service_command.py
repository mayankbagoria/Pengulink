"""tests.unit.test_service_command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit test suite for PenguLink local command execution service and store:
- CommandEntry dataclass, dictionary parsing, wire serialization, and validation
- CommandStore persistence, POSIX 0600 permissions, atomic writes, corruption recovery
- LocalCommandService subprocess execution, shell=False safety, output streaming,
  process group termination, timeout handling, and EventBus emission.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from pengulink.core.events import (
    CommandExecutedEvent,
    CommandOutputEvent,
    EventBus,
)
from pengulink.services.command import (
    DEFAULT_SAMPLE_COMMANDS,
    CommandEntry,
    CommandStore,
    ExecutionResult,
    LocalCommandService,
)

# =============================================================================
# Test CommandEntry Data Model
# =============================================================================


class TestCommandEntryDataModel:
    """Tests for CommandEntry serialization, validation, and wire encoding."""

    def test_command_entry_initialization_and_to_dict(self) -> None:
        entry = CommandEntry(
            key="cmd_lock",
            name="Lock Screen",
            command="loginctl lock-session",
            enabled=True,
        )
        assert entry.key == "cmd_lock"
        assert entry.name == "Lock Screen"
        assert entry.command == "loginctl lock-session"
        assert entry.enabled is True

        d = entry.to_dict()
        assert d == {
            "key": "cmd_lock",
            "name": "Lock Screen",
            "command": "loginctl lock-session",
            "enabled": True,
        }

    def test_to_wire_dict_conforms_to_kdeconnect_spec(self) -> None:
        entry = CommandEntry(
            key="cmd_suspend",
            name="Suspend",
            command="systemctl suspend",
        )
        wire = entry.to_wire_dict()
        assert wire == {
            "name": "Suspend",
            "command": "systemctl suspend",
        }
        assert "key" not in wire
        assert "enabled" not in wire

    def test_from_dict_parsing_nominal(self) -> None:
        raw = {
            "key": "cmd_test",
            "name": "Run Test",
            "command": "pytest -v",
            "enabled": True,
        }
        entry = CommandEntry.from_dict(raw)
        assert entry.key == "cmd_test"
        assert entry.name == "Run Test"
        assert entry.command == "pytest -v"
        assert entry.enabled is True

    def test_from_dict_key_override(self) -> None:
        raw = {
            "name": "Echo Name",
            "command": "echo 1",
        }
        entry = CommandEntry.from_dict(raw, key_override="custom_key")
        assert entry.key == "custom_key"
        assert entry.name == "Echo Name"
        assert entry.command == "echo 1"
        assert entry.enabled is True

    def test_from_dict_validation_errors(self) -> None:
        with pytest.raises(TypeError, match="Expected dict"):
            CommandEntry.from_dict("not_a_dict")  # type: ignore

        with pytest.raises(ValueError, match="key cannot be empty"):
            CommandEntry.from_dict({"name": "Test", "command": "ls"})

        with pytest.raises(ValueError, match="name cannot be empty"):
            CommandEntry.from_dict({"key": "k", "name": "", "command": "ls"})

        with pytest.raises(ValueError, match="command cannot be empty"):
            CommandEntry.from_dict({"key": "k", "name": "Test", "command": " "})

        with pytest.raises(ValueError, match="key cannot be empty"):
            CommandEntry.from_dict({"key": None, "name": "Test", "command": "ls"})

        with pytest.raises(ValueError, match="name cannot be empty"):
            CommandEntry.from_dict({"key": "k", "name": None, "command": "ls"})

        with pytest.raises(ValueError, match="command cannot be empty"):
            CommandEntry.from_dict({"key": "k", "name": "Test", "command": None})

        with pytest.raises(ValueError, match="key cannot be empty"):
            CommandEntry.from_dict({"key": 12345, "name": "Test", "command": "ls"})


# =============================================================================
# Test CommandStore Persistence & Security
# =============================================================================


class TestCommandStore:
    """Tests for CommandStore CRUD, atomic writes, and 0o600 permissions."""

    def test_empty_store_initialization(self, tmp_path: Path) -> None:
        store_path = tmp_path / "commands.json"
        store = CommandStore(storage_path=store_path, populate_defaults=False)
        assert store.list_commands() == []
        assert store.to_wire_dict() == {}
        assert store.to_android_json_string() == "{}"

    def test_populate_defaults(self, tmp_path: Path) -> None:
        store_path = tmp_path / "commands.json"
        store = CommandStore(storage_path=store_path, populate_defaults=True)
        cmds = store.list_commands()
        assert len(cmds) == len(DEFAULT_SAMPLE_COMMANDS)
        assert store.get_command("cmd_lock") is not None
        assert store.get_command("cmd_suspend") is not None

    def test_add_get_remove_command(self, tmp_path: Path) -> None:
        store_path = tmp_path / "commands.json"
        store = CommandStore(storage_path=store_path)

        # 1. Add command with explicit key
        entry = store.add_command(name="List Files", command="ls -la", key="cmd_ls")
        assert entry.key == "cmd_ls"
        assert entry.name == "List Files"
        assert entry.command == "ls -la"

        # 2. Add command with auto-generated key
        auto_entry = store.add_command(name="Date", command="date")
        assert len(auto_entry.key) == 12

        # 3. Get command
        assert store.get_command("cmd_ls") == entry
        assert store.get_command("nonexistent") is None

        # 4. List commands
        all_cmds = store.list_commands()
        assert len(all_cmds) == 2

        # 5. Remove command
        assert store.remove_command("cmd_ls") is True
        assert store.remove_command("cmd_ls") is False
        assert store.get_command("cmd_ls") is None
        assert len(store.list_commands()) == 1

    def test_atomic_file_write_and_posix_permissions(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "secure_config"
        store_path = store_dir / "commands.json"
        store = CommandStore(storage_path=store_path)

        store.add_command(name="Test", command="echo 1", key="t1")

        # Verify file exists on disk
        assert store_path.exists()

        # Check POSIX 0600 file permissions
        file_mode = stat.S_IMODE(os.stat(store_path).st_mode)
        assert file_mode & 0o777 == 0o600

        # Check POSIX 0700 parent directory permissions
        dir_mode = stat.S_IMODE(os.stat(store_dir).st_mode)
        assert dir_mode & 0o777 == 0o700

        # Reload fresh instance from disk
        store2 = CommandStore(storage_path=store_path)
        loaded_entry = store2.get_command("t1")
        assert loaded_entry is not None
        assert loaded_entry.name == "Test"
        assert loaded_entry.command == "echo 1"

    def test_add_command_validation_errors(self, tmp_path: Path) -> None:
        store = CommandStore(storage_path=tmp_path / "commands.json")
        with pytest.raises(ValueError, match="cannot be empty"):
            store.add_command(name=" ", command="ls")
        with pytest.raises(ValueError, match="cannot be empty"):
            store.add_command(name="Name", command=" ")

    def test_to_wire_dict_filters_disabled_commands(self, tmp_path: Path) -> None:
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(name="Active", command="echo 1", key="c1", enabled=True)
        store.add_command(name="Disabled", command="echo 2", key="c2", enabled=False)

        wire = store.to_wire_dict()
        assert "c1" in wire
        assert "c2" not in wire

        android_str = store.to_android_json_string()
        parsed = json.loads(android_str)
        assert "c1" in parsed
        assert "c2" not in parsed

    def test_corruption_recovery(self, tmp_path: Path) -> None:
        store_path = tmp_path / "corrupt_commands.json"
        # Write corrupted JSON
        store_path.write_text("{ unclosed bad json", encoding="utf-8")

        store = CommandStore(storage_path=store_path)
        # Should gracefully recover to empty dict
        assert store.list_commands() == []

        # Corrupt backup file should have been created
        backups = list(tmp_path.glob("corrupt_commands.corrupt.*.bak"))
        assert len(backups) == 1


# =============================================================================
# Test LocalCommandService Execution & Lifecycle
# =============================================================================


class TestLocalCommandService:
    """Tests for safe subprocess execution, process groups, and cancellation."""

    async def test_execute_simple_echo_command(self) -> None:
        service = LocalCommandService()
        result = await service.run_command_string(
            command="echo 'Hello PenguLink'",
            key="test_echo",
        )
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.exit_code == 0
        assert "Hello PenguLink" in result.stdout

    async def test_execute_non_zero_exit_code(self) -> None:
        service = LocalCommandService()
        py_cmd = f"{sys.executable} -c 'import sys; sys.exit(7)'"
        result = await service.run_command_string(
            command=py_cmd,
            key="test_fail",
        )
        assert result.success is False
        assert result.exit_code == 7

    async def test_execute_non_existent_binary_returns_127(self) -> None:
        service = LocalCommandService()
        result = await service.run_command_string(
            command="/usr/bin/non_existent_binary_99999 --arg",
            key="missing_bin",
        )
        assert result.success is False
        assert result.exit_code == 127
        assert any("Command not found" in line for line in result.stderr)

    async def test_shell_metacharacters_not_evaluated_by_shell(self) -> None:
        """Verify shell=False prevents command chaining or evaluation."""
        service = LocalCommandService()
        # In a real shell, 'echo a; echo b' would run two commands.
        # With create_subprocess_exec(*shlex.split()), '; echo b' are arguments to echo.
        result = await service.run_command_string(
            command='echo "arg1; arg2 | arg3"',
            key="test_injection",
        )
        assert result.success is True
        assert result.exit_code == 0
        assert "arg1; arg2 | arg3" in result.stdout[0]

    async def test_empty_command_raises_value_error(self) -> None:
        service = LocalCommandService()
        with pytest.raises(ValueError, match="empty command string"):
            await service.run_command_string(command="  ")

    async def test_streaming_output_callback(self) -> None:
        service = LocalCommandService()
        captured_stdout: list[str] = []
        captured_stderr: list[str] = []

        async def output_cb(key: str, out: list[str], err: list[str], pid: str) -> None:
            captured_stdout.extend(out)
            captured_stderr.extend(err)

        py_cmd = f'{sys.executable} -c \'print("line1"); print("line2"); import sys; sys.stderr.write("err1\\n")\''
        result = await service.run_command_string(
            command=py_cmd,
            key="stream_test",
            send_output=output_cb,
        )
        assert result.success is True
        assert "line1" in captured_stdout
        assert "line2" in captured_stdout
        assert "err1" in captured_stderr

    async def test_event_bus_publishing(self) -> None:
        bus = EventBus()
        service = LocalCommandService(event_bus=bus)

        output_events: list[CommandOutputEvent] = []
        exec_events: list[CommandExecutedEvent] = []

        bus.subscribe(CommandOutputEvent, output_events.append)
        bus.subscribe(CommandExecutedEvent, exec_events.append)

        py_cmd = f"{sys.executable} -c 'print(\"bus_line\"); import sys; sys.exit(0)'"
        result = await service.run_command_string(
            command=py_cmd,
            key="bus_test",
            command_name="Bus Command",
            device_id="dev_42",
        )
        assert result.success is True
        assert len(output_events) >= 1
        assert output_events[0].data == "bus_line"
        assert output_events[0].device_id == "dev_42"
        assert output_events[0].key == "bus_test"

        assert len(exec_events) == 1
        assert exec_events[0].command_name == "Bus Command"
        assert exec_events[0].success is True
        assert exec_events[0].exit_code == 0
        assert exec_events[0].device_id == "dev_42"

    async def test_stop_process_terminates_running_process(self) -> None:
        service = LocalCommandService()
        py_cmd = f"{sys.executable} -c 'import time; time.sleep(30)'"

        # Launch command in background task
        task = asyncio.create_task(
            service.run_command_string(
                command=py_cmd,
                key="sleep_test",
                process_id="sleep_proc_1",
            )
        )

        # Wait until process is active
        await asyncio.sleep(0.15)
        assert "sleep_proc_1" in service._active_processes

        # Terminate process
        stopped = await service.stop_process("sleep_proc_1", timeout=0.5)
        assert stopped is True

        res = await task
        assert res.success is False
        assert res.exit_code != 0

    async def test_stop_all_terminates_multiple_processes(self) -> None:
        service = LocalCommandService()
        py_cmd = f"{sys.executable} -c 'import time; time.sleep(30)'"

        t1 = asyncio.create_task(
            service.run_command_string(command=py_cmd, process_id="proc_1")
        )
        t2 = asyncio.create_task(
            service.run_command_string(command=py_cmd, process_id="proc_2")
        )

        await asyncio.sleep(0.15)
        assert len(service._active_processes) == 2

        await service.stop_all(timeout=0.5)

        r1 = await t1
        r2 = await t2
        assert r1.success is False
        assert r2.success is False
        assert len(service._active_processes) == 0

    async def test_execution_timeout(self) -> None:
        service = LocalCommandService(default_timeout=0.2)
        py_cmd = f"{sys.executable} -c 'import time; time.sleep(10)'"

        result = await service.run_command_string(
            command=py_cmd,
            key="timeout_test",
            timeout=0.2,
        )
        assert result.success is False
        assert result.exit_code == 124

    async def test_execute_command_from_store_lookup(self, tmp_path: Path) -> None:
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(name="Echo", command="echo 'from store'", key="cmd_echo")

        service = LocalCommandService(command_store=store)
        res = await service.execute_command("cmd_echo")
        assert res.success is True
        assert "from store" in res.stdout

        with pytest.raises(KeyError, match="not found in store"):
            await service.execute_command("invalid_key")

    async def test_task_cancellation_terminates_subprocess(self) -> None:
        service = LocalCommandService()
        py_cmd = f"{sys.executable} -u -c 'import time; time.sleep(60)'"

        task = asyncio.create_task(
            service.run_command_string(command=py_cmd, process_id="cancel_test")
        )
        await asyncio.sleep(0.1)
        assert "cancel_test" in service._active_processes
        pid = service._active_processes["cancel_test"].pid

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert "cancel_test" not in service._active_processes
        await asyncio.sleep(0.1)
        try:
            os.kill(pid, 0)
            is_alive = True
        except ProcessLookupError:
            is_alive = False
        assert is_alive is False, (
            "Subprocess was expected to be terminated on task cancellation"
        )

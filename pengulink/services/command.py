"""pengulink.services.command
~~~~~~~~~~~~~~~~~~~~~~~~~~
Local command configuration storage and secure subprocess execution service.

Implements:
- CommandEntry: Strongly typed data model with KDE Connect wire serialization.
- CommandStore: POSIX 0o600 atomic JSON storage with corruption recovery and migration.
- LocalCommandService: Safe subprocess execution (shell=False, process group isolation,
  streaming stdout/stderr chunks, SIGTERM/SIGKILL termination, EventBus publishing).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import shlex
import shutil
import signal
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pengulink.core.events import (
    CommandExecutedEvent,
    CommandOutputEvent,
    EventBus,
)
from pengulink.core.exceptions import StorageCorruptedError, StorageWriteError

logger = logging.getLogger("pengulink.services.command")

DEFAULT_COMMANDS_PATH = Path.home() / ".config" / "pengulink" / "commands.json"
FALLBACK_COMMANDS_PATH = (
    Path.home() / ".config" / "pengulink" / "runcommand" / "commands.json"
)

DEFAULT_SAMPLE_COMMANDS: dict[str, dict[str, str]] = {
    "cmd_lock": {"name": "Lock Screen", "command": "loginctl lock-session"},
    "cmd_suspend": {"name": "Suspend System", "command": "systemctl suspend"},
}

# Max output captured per process to prevent memory exhaustion / Wi-Fi flooding
MAX_PROCESS_OUTPUT_BYTES = 64 * 1024
DEFAULT_STREAM_CHUNK_LINES = 5


@dataclass
class CommandEntry:
    """Represents a locally configured executable command."""

    key: str
    name: str
    command: str
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return complete dictionary representation."""
        return {
            "key": self.key,
            "name": self.name,
            "command": self.command,
            "enabled": self.enabled,
        }

    def to_wire_dict(self) -> dict[str, str]:
        """Return KDE Connect protocol wire dictionary."""
        return {
            "name": self.name,
            "command": self.command,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], key_override: str | None = None
    ) -> CommandEntry:
        """Parse CommandEntry from dictionary."""
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected dict for CommandEntry, got {type(data).__name__}"
            )
        raw_key = key_override if key_override is not None else data.get("key")
        if raw_key is None or not isinstance(raw_key, (str, bytes)):
            raise ValueError("CommandEntry key cannot be empty")
        key = (
            raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key
        ).strip()
        if not key:
            raise ValueError("CommandEntry key cannot be empty")

        raw_name = data.get("name")
        if raw_name is None or not isinstance(raw_name, (str, bytes)):
            raise ValueError("CommandEntry name cannot be empty")
        name = (
            raw_name.decode("utf-8") if isinstance(raw_name, bytes) else raw_name
        ).strip()
        if not name:
            raise ValueError("CommandEntry name cannot be empty")

        raw_cmd = data.get("command")
        if raw_cmd is None or not isinstance(raw_cmd, (str, bytes)):
            raise ValueError("CommandEntry command cannot be empty")
        command = (
            raw_cmd.decode("utf-8") if isinstance(raw_cmd, bytes) else raw_cmd
        ).strip()
        if not command:
            raise ValueError("CommandEntry command cannot be empty")
        enabled = bool(data.get("enabled", True))
        return cls(key=key, name=name, command=command, enabled=enabled)


class CommandStore:
    """Persistent storage for local commands with POSIX 0o600 atomic file writes."""

    def __init__(
        self,
        storage_path: Path | str | None = None,
        populate_defaults: bool = False,
    ) -> None:
        if storage_path is not None:
            self.storage_path = Path(storage_path)
        else:
            if DEFAULT_COMMANDS_PATH.exists():
                self.storage_path = DEFAULT_COMMANDS_PATH
            elif FALLBACK_COMMANDS_PATH.exists():
                self.storage_path = FALLBACK_COMMANDS_PATH
            else:
                self.storage_path = DEFAULT_COMMANDS_PATH

        self.populate_defaults = populate_defaults
        self._commands: dict[str, CommandEntry] = {}
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load commands from JSON storage file, recovering safely from corruption."""
        if not self.storage_path.exists():
            if self.populate_defaults:
                for k, v in DEFAULT_SAMPLE_COMMANDS.items():
                    self._commands[k] = CommandEntry(
                        key=k, name=v["name"], command=v["command"]
                    )
            return

        try:
            content = self.storage_path.read_text(encoding="utf-8")
            if not content.strip():
                self._commands = {}
                return
            data = json.loads(content)
            if not isinstance(data, dict):
                raise StorageCorruptedError(
                    f"Root element in {self.storage_path} is not a dictionary"
                )

            # Handle both {"commands": {...}} schema and direct key-entry mapping
            raw_cmds: dict[str, Any] = (
                data.get("commands", {}) if "commands" in data else data
            )
            if not isinstance(raw_cmds, dict):
                raise StorageCorruptedError("commands section is not a dictionary")

            loaded: dict[str, CommandEntry] = {}
            for k, v in raw_cmds.items():
                if isinstance(v, dict):
                    try:
                        entry = CommandEntry.from_dict(v, key_override=k)
                        loaded[entry.key] = entry
                    except ValueError as err:
                        logger.warning(
                            "Skipping invalid command entry '%s': %s", k, err
                        )
            self._commands = loaded
        except (
            json.JSONDecodeError,
            StorageCorruptedError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            corrupt_backup = self.storage_path.with_suffix(
                f".corrupt.{int(time.time())}.bak"
            )
            logger.error(
                "Commands file %s is corrupt (%s). Backing up to %s",
                self.storage_path,
                exc,
                corrupt_backup,
            )
            try:
                shutil.copy2(self.storage_path, corrupt_backup)
            except OSError:
                pass
            self._commands = {}

    def _save_to_disk(self) -> None:
        """Write JSON file atomically using a temporary file with secure 0o600 permissions."""
        parent_dir = self.storage_path.parent
        try:
            parent_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            if parent_dir.resolve() != Path.home().resolve():
                try:
                    os.chmod(parent_dir, 0o700)
                except OSError:
                    pass
        except (OSError, PermissionError) as exc:
            raise StorageWriteError(
                f"Failed to create storage directory {parent_dir}: {exc}"
            ) from exc

        tmp_file = self.storage_path.with_suffix(f".tmp.{os.getpid()}.{time.time_ns()}")
        try:
            payload = {
                "version": 1,
                "last_updated": time.time(),
                "commands": {
                    entry.key: entry.to_dict() for entry in self._commands.values()
                },
            }
            content = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
            fd = os.open(tmp_file, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            try:
                os.write(fd, content)
            finally:
                os.close(fd)

            try:
                os.chmod(tmp_file, 0o600)
            except OSError:
                pass

            tmp_file.replace(self.storage_path)

            try:
                os.chmod(self.storage_path, 0o600)
            except OSError:
                pass
        except (OSError, TypeError, ValueError) as exc:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass
            raise StorageWriteError(
                f"Failed writing {self.storage_path}: {exc}"
            ) from exc

    def load_commands(self) -> dict[str, CommandEntry]:
        """Reload from disk and return copy of command entries."""
        self._load_from_disk()
        return dict(self._commands)

    def save_commands(self, commands: dict[str, CommandEntry] | None = None) -> None:
        """Save commands dictionary to disk."""
        if commands is not None:
            self._commands = dict(commands)
        self._save_to_disk()

    def add_command(
        self,
        name: str,
        command: str,
        key: str | None = None,
        enabled: bool = True,
    ) -> CommandEntry:
        """Add a command entry, persisting atomically to disk."""
        clean_name = name.strip()
        clean_command = command.strip()
        if not clean_name:
            raise ValueError("Command name cannot be empty")
        if not clean_command:
            raise ValueError("Command string cannot be empty")

        clean_key = (key or "").strip()
        if not clean_key:
            clean_key = uuid.uuid4().hex[:12]

        entry = CommandEntry(
            key=clean_key,
            name=clean_name,
            command=clean_command,
            enabled=enabled,
        )
        self._commands[clean_key] = entry
        self._save_to_disk()
        logger.info("Added command '%s' (key=%s)", clean_name, clean_key)
        return entry

    def remove_command(self, key: str) -> bool:
        """Remove a command entry by key."""
        if key in self._commands:
            del self._commands[key]
            self._save_to_disk()
            logger.info("Removed command key '%s'", key)
            return True
        return False

    def get_command(self, key: str) -> CommandEntry | None:
        """Retrieve a command entry by key."""
        return self._commands.get(key)

    def list_commands(self) -> list[CommandEntry]:
        """Return list of all configured command entries."""
        return list(self._commands.values())

    def to_wire_dict(self) -> dict[str, dict[str, str]]:
        """Return dictionary conforming to KDE Connect wire specification."""
        return {
            entry.key: entry.to_wire_dict()
            for entry in self._commands.values()
            if entry.enabled
        }

    def to_android_json_string(self) -> str:
        """Serialize commandList as compact JSON string (Android requirement)."""
        return json.dumps(self.to_wire_dict())


@dataclass
class ExecutionResult:
    """Outcome of a local command execution."""

    process_id: str
    key: str
    command: str
    exit_code: int
    success: bool
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)


OutputCallback = Callable[
    [str, list[str], list[str], str],
    Coroutine[Any, Any, None] | None,
]


class LocalCommandService:
    """Safe asynchronous execution service for local host commands."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        command_store: CommandStore | None = None,
        default_timeout: float = 60.0,
        output_chunk_size: int = DEFAULT_STREAM_CHUNK_LINES,
        max_output_bytes: int = MAX_PROCESS_OUTPUT_BYTES,
    ) -> None:
        self.event_bus = event_bus
        self.command_store = command_store or CommandStore()
        self.default_timeout = default_timeout
        self.output_chunk_size = output_chunk_size
        self.max_output_bytes = max_output_bytes

        self._active_processes: dict[str, asyncio.subprocess.Process] = {}
        self._active_tasks: dict[str, asyncio.Task[Any]] = {}
        self._next_id: int = 1
        self._lock = asyncio.Lock()

    def _generate_process_id(self, key: str) -> str:
        asyncio.get_running_loop()
        pid = self._next_id
        self._next_id += 1
        return str(pid)

    def _build_env(self) -> dict[str, str]:
        """Construct sanitized environment preserving desktop session variables."""
        allowed_keys = {
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "SHELL",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TERM",
            "XDG_RUNTIME_DIR",
            "XDG_SESSION_TYPE",
            "XDG_CURRENT_DESKTOP",
            "DBUS_SESSION_BUS_ADDRESS",
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "PULSE_SERVER",
            "PIPEWIRE_REMOTE",
        }
        env = {k: v for k, v in os.environ.items() if k in allowed_keys}
        if "PATH" not in env:
            env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
        return env

    async def execute_command(
        self,
        key: str,
        device_id: str = "",
        send_output: OutputCallback | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Look up command by key from store and execute safely."""
        entry = self.command_store.get_command(key)
        if not entry:
            raise KeyError(f"Command with key '{key}' not found in store")
        if not entry.enabled:
            raise ValueError(f"Command with key '{key}' is disabled")

        return await self.execute_command_entry(
            entry=entry,
            device_id=device_id,
            send_output=send_output,
            timeout=timeout,
        )

    async def execute_command_entry(
        self,
        entry: CommandEntry,
        device_id: str = "",
        process_id: str | None = None,
        send_output: OutputCallback | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute a specific CommandEntry."""
        return await self.run_command_string(
            command=entry.command,
            key=entry.key,
            command_name=entry.name,
            device_id=device_id,
            process_id=process_id,
            send_output=send_output,
            timeout=timeout,
        )

    async def run_command_string(
        self,
        command: str,
        key: str = "",
        command_name: str = "",
        device_id: str = "",
        process_id: str | None = None,
        send_output: OutputCallback | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Spawn and track a subprocess without shell evaluation."""
        clean_command = command.strip()
        if not clean_command:
            raise ValueError("Cannot execute empty command string")

        argv = shlex.split(clean_command)
        if not argv:
            raise ValueError("Parsed argument list is empty")

        pid_str = process_id or self._generate_process_id(key)
        effective_timeout = timeout if timeout is not None else self.default_timeout
        display_name = command_name or argv[0]

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        total_captured_bytes = 0

        # Subprocess spawning
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                close_fds=True,
                env=self._build_env(),
            )
        except FileNotFoundError:
            err_msg = f"Command not found: {argv[0]}"
            logger.warning(
                "Execution failed: %s (key=%s, device=%s)",
                err_msg,
                key,
                device_id,
            )
            stderr_lines.append(err_msg)
            if self.event_bus:
                await self.event_bus.publish(
                    CommandOutputEvent(
                        device_id=device_id,
                        key=key,
                        stream="stderr",
                        data=err_msg,
                    )
                )
                await self.event_bus.publish(
                    CommandExecutedEvent(
                        device_id=device_id,
                        key=key,
                        command_name=display_name,
                        success=False,
                        exit_code=127,
                    )
                )
            return ExecutionResult(
                process_id=pid_str,
                key=key,
                command=command,
                exit_code=127,
                success=False,
                stdout=[],
                stderr=stderr_lines,
            )
        except PermissionError:
            err_msg = f"Permission denied executing command: {argv[0]}"
            logger.warning("Execution permission denied: %s (key=%s)", err_msg, key)
            stderr_lines.append(err_msg)
            if self.event_bus:
                await self.event_bus.publish(
                    CommandOutputEvent(
                        device_id=device_id,
                        key=key,
                        stream="stderr",
                        data=err_msg,
                    )
                )
                await self.event_bus.publish(
                    CommandExecutedEvent(
                        device_id=device_id,
                        key=key,
                        command_name=display_name,
                        success=False,
                        exit_code=126,
                    )
                )
            return ExecutionResult(
                process_id=pid_str,
                key=key,
                command=command,
                exit_code=126,
                success=False,
                stdout=[],
                stderr=stderr_lines,
            )

        async with self._lock:
            self._active_processes[pid_str] = proc

        # Stream reader helper
        async def _read_stream(
            reader: asyncio.StreamReader | None,
            stream_name: str,
            dest: list[str],
        ) -> None:
            nonlocal total_captured_bytes
            if reader is None:
                return

            chunk_buffer: list[str] = []

            async def _flush() -> None:
                if not chunk_buffer:
                    return
                lines_to_send = list(chunk_buffer)
                chunk_buffer.clear()
                if send_output:
                    out_chunk = lines_to_send if stream_name == "stdout" else []
                    err_chunk = lines_to_send if stream_name == "stderr" else []
                    res = send_output(key, out_chunk, err_chunk, pid_str)
                    if inspect.isawaitable(res):
                        await res

            buf = b""
            while not reader.at_eof():
                try:
                    chunk = await reader.read(65536)
                except (asyncio.LimitOverrunError, ValueError):
                    try:
                        chunk = await reader.read(4096)
                    except (asyncio.LimitOverrunError, ValueError):
                        break
                if not chunk:
                    break

                buf += chunk
                while b"\n" in buf:
                    raw_line, buf = buf.split(b"\n", 1)
                    if total_captured_bytes < self.max_output_bytes:
                        total_captured_bytes += len(raw_line)
                        line_str = raw_line.decode("utf-8", errors="replace").rstrip(
                            "\r"
                        )
                        dest.append(line_str)
                        chunk_buffer.append(line_str)

                        if self.event_bus:
                            await self.event_bus.publish(
                                CommandOutputEvent(
                                    device_id=device_id,
                                    key=key,
                                    stream=stream_name,
                                    data=line_str,
                                )
                            )

                        if len(chunk_buffer) >= self.output_chunk_size:
                            await _flush()

                # Slice chunks >= 64KB without newline to avoid memory overrun
                while len(buf) >= 65536:
                    raw_chunk = buf[:65536]
                    buf = buf[65536:]
                    if total_captured_bytes < self.max_output_bytes:
                        total_captured_bytes += len(raw_chunk)
                        line_str = raw_chunk.decode("utf-8", errors="replace").rstrip(
                            "\r"
                        )
                        dest.append(line_str)
                        chunk_buffer.append(line_str)

                        if self.event_bus:
                            await self.event_bus.publish(
                                CommandOutputEvent(
                                    device_id=device_id,
                                    key=key,
                                    stream=stream_name,
                                    data=line_str,
                                )
                            )

                        if len(chunk_buffer) >= self.output_chunk_size:
                            await _flush()

            if buf and total_captured_bytes < self.max_output_bytes:
                total_captured_bytes += len(buf)
                line_str = buf.decode("utf-8", errors="replace").rstrip("\r")
                dest.append(line_str)
                chunk_buffer.append(line_str)

                if self.event_bus:
                    await self.event_bus.publish(
                        CommandOutputEvent(
                            device_id=device_id,
                            key=key,
                            stream=stream_name,
                            data=line_str,
                        )
                    )

            await _flush()

        # Concurrent stream reading
        read_task = asyncio.gather(
            _read_stream(proc.stdout, "stdout", stdout_lines),
            _read_stream(proc.stderr, "stderr", stderr_lines),
        )

        timed_out = False
        try:
            if effective_timeout and effective_timeout > 0:
                await asyncio.wait_for(
                    asyncio.gather(read_task, proc.wait()),
                    timeout=effective_timeout,
                )
            else:
                await asyncio.gather(read_task, proc.wait())
        except asyncio.TimeoutError:
            timed_out = True
            logger.warning(
                "Command '%s' (pid=%s) timed out after %ss, terminating",
                key,
                pid_str,
                effective_timeout,
            )
            await self.stop_process(pid_str)
            read_task.cancel()
            try:
                await read_task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            await self.stop_process(pid_str)
            read_task.cancel()
            try:
                await read_task
            except asyncio.CancelledError:
                pass
            raise
        except Exception as exc:
            logger.error(
                "Exception executing command '%s' (pid=%s): %s",
                key,
                pid_str,
                exc,
            )
            await self.stop_process(pid_str)
            read_task.cancel()
            try:
                await read_task
            except asyncio.CancelledError:
                pass
            raise
        finally:
            if proc.returncode is None:
                await self.stop_process(pid_str)
            async with self._lock:
                self._active_processes.pop(pid_str, None)

        exit_code = 124 if timed_out else (proc.returncode or 0)
        success = (exit_code == 0) and not timed_out

        if self.event_bus:
            await self.event_bus.publish(
                CommandExecutedEvent(
                    device_id=device_id,
                    key=key,
                    command_name=display_name,
                    success=success,
                    exit_code=exit_code,
                )
            )

        return ExecutionResult(
            process_id=pid_str,
            key=key,
            command=command,
            exit_code=exit_code,
            success=success,
            stdout=stdout_lines,
            stderr=stderr_lines,
        )

    async def stop_process(self, process_id: str | int, timeout: float = 1.0) -> bool:
        """Terminate an active child process group using SIGTERM followed by SIGKILL."""
        pid_str = str(process_id)
        async with self._lock:
            proc = self._active_processes.get(pid_str)

        if not proc:
            return False

        # In start_new_session=True, proc.pid is the process group ID
        pgid = proc.pid

        # If leader already exited, check if any grandchildren survive in the process group
        if proc.returncode is not None:
            try:
                os.killpg(pgid, 0)
                os.killpg(pgid, signal.SIGKILL)
                return True
            except (ProcessLookupError, OSError):
                return False

        # Attempt process group SIGTERM
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            try:
                proc.terminate()
            except OSError:
                pass

        # Wait for leader process to exit
        leader_exited = False
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            leader_exited = True
        except asyncio.TimeoutError:
            leader_exited = False

        # Check if the process group still contains living processes (e.g. grandchildren)
        pg_alive = False
        try:
            os.killpg(pgid, 0)
            pg_alive = True
        except (ProcessLookupError, OSError):
            pg_alive = False

        # Escalate to SIGKILL if leader hasn't exited or grandchild processes survive
        if not leader_exited or pg_alive:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass

            try:
                await asyncio.wait_for(proc.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass

        return True

    async def stop_all(self, timeout: float = 1.0) -> None:
        """Terminate all active child process groups."""
        async with self._lock:
            active_pids = list(self._active_processes.keys())

        for pid in active_pids:
            await self.stop_process(pid, timeout=timeout)


__all__ = [
    "DEFAULT_COMMANDS_PATH",
    "DEFAULT_SAMPLE_COMMANDS",
    "FALLBACK_COMMANDS_PATH",
    "CommandEntry",
    "CommandStore",
    "ExecutionResult",
    "LocalCommandService",
]

"""tests.unit.test_challenger_m23_command_service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial stress-test suite for PenguLink LocalCommandService and CommandStore (Milestone 23):
1. Long-running & hung processes: SIGTERM -> SIGKILL escalation on direct and child processes, timeout handling.
2. Zombie processes & process group detachment: Sub-children termination, process group cleanup, stop_all().
3. Rapid concurrent executions: Burst of 30+ parallel commands, EventBus stream isolation, unique process IDs.
4. Large stdout/stderr output: Megabyte outputs, interleaved streams, buffer capping, un-delimited line limit overrun.
5. Corrupted commands.json recovery: Truncated JSON, binary garbage, bad root types, partial invalid entries, POSIX permissions.
6. Unicode & special characters: Emojis, CJK, Arabic, shell metacharacters, quotes, empty inputs.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import stat
import sys
import time
from pathlib import Path
from unittest.mock import patch

import psutil
import pytest

from pengulink.core.events import (
    CommandExecutedEvent,
    CommandOutputEvent,
    EventBus,
)
from pengulink.core.exceptions import StorageWriteError
from pengulink.services.command import (
    CommandStore,
    ExecutionResult,
    LocalCommandService,
)

# =============================================================================
# Helper Utilities
# =============================================================================


def get_child_pids_for_pgid(pgid: int) -> list[int]:
    """Find all active process IDs belonging to the given process group ID."""
    pids = []
    for p in psutil.process_iter(["pid"]):
        try:
            if os.getpgid(p.info["pid"]) == pgid:
                pids.append(p.info["pid"])
        except (ProcessLookupError, PermissionError, OSError):
            pass
    return pids


def clean_kill_pids(pids: list[int]) -> None:
    """Force kill a list of process IDs to prevent test process leakage."""
    for pid in pids:
        try:
            os.kill(pid, 9)
        except (ProcessLookupError, PermissionError, OSError):
            pass


# =============================================================================
# Vector 1: Long-Running and Hung Processes (SIGKILL Escalation)
# =============================================================================


class TestChallengerHungProcessesAndEscalation:
    """Stress tests for process termination, hung processes, and SIGKILL escalation."""

    async def test_direct_process_ignoring_sigterm_escalates_to_sigkill(self) -> None:
        """Verify that a direct process ignoring SIGTERM is forcefully killed by SIGKILL."""
        svc = LocalCommandService()
        # Process ignores SIGTERM and sleeps
        py_code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
        cmd = f"{sys.executable} -c {shlex.quote(py_code)}"

        task = asyncio.create_task(
            svc.run_command_string(cmd, key="ignore_term", process_id="hung_proc_1")
        )

        await asyncio.sleep(0.3)
        assert "hung_proc_1" in svc._active_processes
        proc = svc._active_processes["hung_proc_1"]
        pid = proc.pid
        assert psutil.pid_exists(pid)

        # Stop process with short timeout (0.2s) so it escalates to SIGKILL
        t0 = time.time()
        stopped = await svc.stop_process("hung_proc_1", timeout=0.2)
        elapsed = time.time() - t0

        assert stopped is True
        # Should have waited ~0.2s for SIGTERM before escalating to SIGKILL
        assert elapsed >= 0.18

        res = await task
        assert res.success is False
        assert res.exit_code == -9 or res.exit_code == 137  # SIGKILL

        # Verify process is dead
        await asyncio.sleep(0.1)
        assert not psutil.pid_exists(pid)

    async def test_long_running_process_timeout_triggers_termination(self) -> None:
        """Verify execution timeout terminates the process group and returns exit_code=124."""
        svc = LocalCommandService(default_timeout=0.3)
        py_code = "import time; time.sleep(30)"
        cmd = f"{sys.executable} -c {shlex.quote(py_code)}"

        t0 = time.time()
        res = await svc.run_command_string(cmd, key="timeout_proc", timeout=0.3)
        elapsed = time.time() - t0

        assert 0.25 <= elapsed <= 1.5
        assert res.success is False
        assert res.exit_code == 124
        assert len(svc._active_processes) == 0

    async def test_stop_process_on_already_exited_process_returns_false(self) -> None:
        """Calling stop_process on an already exited process returns False safely."""
        svc = LocalCommandService()
        res = await svc.run_command_string(
            "echo done", key="quick", process_id="quick_1"
        )
        assert res.success is True

        # Process is already completed and removed from _active_processes
        stopped = await svc.stop_process("quick_1")
        assert stopped is False

    async def test_stop_process_on_nonexistent_pid_returns_false(self) -> None:
        """Calling stop_process on an invalid process ID returns False safely."""
        svc = LocalCommandService()
        stopped = await svc.stop_process("nonexistent_pid_999999")
        assert stopped is False

    async def test_hung_subchild_process_group_escalation_gap_reproducer(self) -> None:
        """Empirical reproducer: When a leader (bash) dies on SIGTERM, stop_process returns True

        immediately without waiting for sub-children that ignored SIGTERM, leaving the child
        orphaned in the background.
        """
        svc = LocalCommandService()
        # Bash spawns a background child that ignores SIGTERM, then waits
        child_code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
        cmd = f"bash -c '{sys.executable} -c \"{child_code}\" & wait'"

        task = asyncio.create_task(
            svc.run_command_string(cmd, key="subchild_hung", process_id="bash_hung")
        )

        await asyncio.sleep(0.4)
        proc = svc._active_processes.get("bash_hung")
        assert proc is not None
        bash_pid = proc.pid

        # Find child processes in this process group
        child_pids = [
            p.info["pid"]
            for p in psutil.process_iter(["pid"])
            if p.info["pid"] != bash_pid and os.getpgid(p.info["pid"]) == bash_pid
        ]
        assert len(child_pids) >= 1, "Child process failed to spawn"
        orphan_pid = child_pids[0]

        try:
            # stop_process is called. Bash receives SIGTERM and exits immediately.
            # stop_process verifies pgid and escalates to SIGKILL on child!
            stopped = await svc.stop_process("bash_hung", timeout=0.2)
            assert stopped is True

            # Child process was terminated because stop_process escalated SIGKILL to pgid
            await asyncio.sleep(0.2)
            child_survived = psutil.pid_exists(orphan_pid)
            assert child_survived is False, (
                "Child process was expected to be terminated by SIGKILL escalation"
            )
            await task
        finally:
            clean_kill_pids([orphan_pid])


# =============================================================================
# Vector 2: Zombie Processes & Process Group Detachment
# =============================================================================


class TestChallengerProcessGroupAndZombieCleanup:
    """Stress tests for multi-process tree cleanup and process group detachment."""

    async def test_bash_multi_child_tree_clean_termination(self) -> None:
        """Spawning multiple background children and stopping terminates the entire group."""
        svc = LocalCommandService()
        cmd = "bash -c 'sleep 100 & sleep 100 & wait'"

        task = asyncio.create_task(
            svc.run_command_string(cmd, key="tree_multi", process_id="tree_proc")
        )

        await asyncio.sleep(0.3)
        proc = svc._active_processes.get("tree_proc")
        assert proc is not None
        pgid = proc.pid

        children = [pid for pid in get_child_pids_for_pgid(pgid) if pid != pgid]
        assert len(children) >= 2, f"Expected at least 2 children, found {children}"

        try:
            stopped = await svc.stop_process("tree_proc", timeout=0.5)
            assert stopped is True
            await task

            # Check that all children in process group are terminated
            await asyncio.sleep(0.2)
            for child_pid in children:
                if psutil.pid_exists(child_pid):
                    p = psutil.Process(child_pid)
                    # If process exists, it must be zombie awaiting reaper or dead
                    assert p.status() == psutil.STATUS_ZOMBIE or not psutil.pid_exists(
                        child_pid
                    )
        finally:
            clean_kill_pids(children)

    async def test_stop_all_terminates_multiple_active_process_groups(self) -> None:
        """Calling stop_all cleanly terminates multiple active process groups."""
        svc = LocalCommandService()
        py_cmd = f"{sys.executable} -c 'import time; time.sleep(60)'"

        tasks = []
        pids = []
        for i in range(5):
            pid_str = f"grp_proc_{i}"
            tasks.append(
                asyncio.create_task(
                    svc.run_command_string(py_cmd, key=f"key_{i}", process_id=pid_str)
                )
            )

        await asyncio.sleep(0.3)
        assert len(svc._active_processes) == 5
        pids = [proc.pid for proc in svc._active_processes.values()]

        try:
            await svc.stop_all(timeout=0.5)
            results = await asyncio.gather(*tasks)

            assert len(svc._active_processes) == 0
            for r in results:
                assert r.success is False

            # Verify all OS pids are dead
            await asyncio.sleep(0.1)
            for pid in pids:
                assert not psutil.pid_exists(pid)
        finally:
            clean_kill_pids(pids)

    async def test_process_lifecycle_cleans_active_processes_dict_on_exit(self) -> None:
        """Verify _active_processes dict is cleared immediately upon normal exit."""
        svc = LocalCommandService()
        res = await svc.run_command_string("echo normal_exit", process_id="exit_1")
        assert res.success is True
        assert "exit_1" not in svc._active_processes
        assert len(svc._active_processes) == 0


# =============================================================================
# Vector 3: Rapid Concurrent Executions & Cross-Talk Isolation
# =============================================================================


class TestChallengerRapidConcurrentExecutions:
    """Stress tests for concurrent process executions, PID collisions, and stream isolation."""

    async def test_burst_30_concurrent_executions_no_crosstalk(self) -> None:
        """Execute 30 concurrent commands in parallel with zero stream cross-talk."""
        svc = LocalCommandService()

        async def run_one(i: int) -> ExecutionResult:
            cmd = f"echo out_{i}"
            res = await svc.run_command_string(
                cmd,
                key=f"burst_{i}",
                command_name=f"Burst {i}",
            )
            assert res.success is True
            assert res.stdout == [f"out_{i}"], (
                f"Stream cross-talk detected: {res.stdout} != out_{i}"
            )
            return res

        results = await asyncio.gather(*[run_one(i) for i in range(30)])
        assert len(results) == 30

        # Verify all process IDs were distinct
        pids = [r.process_id for r in results]
        assert len(set(pids)) == 30

    async def test_burst_concurrent_eventbus_emission_isolation(self) -> None:
        """Verify 25 concurrent executions emit isolated events to EventBus without racing."""
        bus = EventBus()
        svc = LocalCommandService(event_bus=bus)

        events_by_key: dict[str, list[str]] = {}
        executed_events: dict[str, CommandExecutedEvent] = {}

        def on_output(event: CommandOutputEvent) -> None:
            events_by_key.setdefault(event.key, []).append(event.data)

        def on_exec(event: CommandExecutedEvent) -> None:
            executed_events[event.key] = event

        bus.subscribe(CommandOutputEvent, on_output)
        bus.subscribe(CommandExecutedEvent, on_exec)

        async def run_one(i: int) -> None:
            cmd = f"echo event_val_{i}"
            res = await svc.run_command_string(
                cmd,
                key=f"k_{i}",
                command_name=f"Name {i}",
                device_id="dev_stress",
            )
            assert res.success is True

        await asyncio.gather(*[run_one(i) for i in range(25)])

        # Verify every key received exact corresponding event
        for i in range(25):
            k = f"k_{i}"
            assert events_by_key.get(k) == [f"event_val_{i}"], (
                f"Event cross-talk on key {k}"
            )
            assert k in executed_events
            assert executed_events[k].success is True
            assert executed_events[k].device_id == "dev_stress"

    async def test_concurrent_same_key_executions_generate_independent_pids(
        self,
    ) -> None:
        """Executing the same key multiple times concurrently produces distinct process IDs."""
        svc = LocalCommandService()
        tasks = [
            svc.run_command_string("echo same_key", key="duplicate_key")
            for _ in range(15)
        ]
        results = await asyncio.gather(*tasks)
        pids = [r.process_id for r in results]
        assert len(set(pids)) == 15, "Process IDs collided for identical keys"

    async def test_concurrent_executions_interleaved_with_stop_all(self) -> None:
        """Verify starting 10 long-running processes and immediately stopping all causes no deadlock."""
        svc = LocalCommandService()
        py_cmd = f"{sys.executable} -c 'import time; time.sleep(30)'"

        tasks = [
            asyncio.create_task(svc.run_command_string(py_cmd, key=f"interleaved_{i}"))
            for i in range(10)
        ]

        await asyncio.sleep(0.2)
        await svc.stop_all(timeout=0.5)

        results = await asyncio.gather(*tasks)
        for r in results:
            assert r.success is False
        assert len(svc._active_processes) == 0


# =============================================================================
# Vector 4: Large Output Buffers, Interleaved Streams & Overruns
# =============================================================================


class TestChallengerLargeOutputBuffers:
    """Stress tests for megabyte outputs, stream draining, deadlocks, and buffer overruns."""

    async def test_large_stdout_output_stream_bounded_at_max_bytes(self) -> None:
        """Subprocess outputting 5MB with newlines completes quickly and respects max_output_bytes."""
        svc = LocalCommandService(max_output_bytes=64 * 1024)
        # 50,000 lines of 100 bytes each = 5 MB
        py_code = (
            "import sys; [sys.stdout.write('A' * 99 + '\\n') for _ in range(50000)]"
        )
        cmd = f"{sys.executable} -c {shlex.quote(py_code)}"

        t0 = time.time()
        res = await svc.run_command_string(cmd, key="huge_out")
        elapsed = time.time() - t0

        assert res.success is True
        assert res.exit_code == 0
        assert elapsed < 3.0  # Fast stream draining without deadlock

        total_captured = sum(len(line) for line in res.stdout)
        # Should be bounded by max_output_bytes
        assert total_captured <= svc.max_output_bytes + 500

    async def test_large_interleaved_stdout_and_stderr_no_deadlock(self) -> None:
        """Subprocess outputting tens of thousands of lines on both streams does not deadlock."""
        svc = LocalCommandService()
        py_code = (
            "import sys\n"
            "for i in range(5000):\n"
            "    sys.stdout.write(f'out {i}\\n')\n"
            "    sys.stderr.write(f'err {i}\\n')\n"
        )
        cmd = f"{sys.executable} -c {shlex.quote(py_code)}"

        res = await svc.run_command_string(cmd, key="interleaved_stream")
        assert res.success is True
        assert res.exit_code == 0
        assert len(res.stdout) > 0
        assert len(res.stderr) > 0

    async def test_custom_max_output_bytes_strictly_respected(self) -> None:
        """Custom small max_output_bytes (1KB) caps captured lines accurately."""
        svc = LocalCommandService(max_output_bytes=1024)
        py_code = "for i in range(500): print(f'line_{i:04d}_' + 'x' * 80)"
        cmd = f"{sys.executable} -c {shlex.quote(py_code)}"

        res = await svc.run_command_string(cmd, key="capped_out")
        assert res.success is True
        total_captured = sum(len(line) for line in res.stdout)
        assert total_captured <= 1024 + 100

    async def test_large_line_exceeding_streamreader_limit_reproducer(self) -> None:
        """Empirical reproducer: A line exceeding 64KB without newlines causes reader.readline()

        to raise ValueError: Separator is not found, and chunk exceed the limit.
        Because run_command_string does not catch this, the process leaks and an unhandled
        exception is raised to the caller.
        """
        svc = LocalCommandService()
        # Output 70,000 characters without newline
        py_code = "import sys, time; sys.stdout.write('X' * 70000); sys.stdout.flush(); time.sleep(30)"
        cmd = f"{sys.executable} -c {shlex.quote(py_code)}"

        # Find any leaked pids after execution
        curr_pid = os.getpid()
        leaked_pids: list[int] = []

        try:
            res = await svc.run_command_string(cmd, key="line_overrun", timeout=1.0)
            assert res.stdout or res.exit_code == 124

            # Verify zero child processes leaked
            await asyncio.sleep(0.2)
            for p in psutil.process_iter(["pid", "ppid", "cmdline"]):
                try:
                    if p.info["ppid"] == curr_pid and "line_overrun" not in str(
                        p.info["cmdline"]
                    ):
                        cmdline_str = " ".join(p.info["cmdline"] or [])
                        if "70000" in cmdline_str:
                            leaked_pids.append(p.info["pid"])
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    ProcessLookupError,
                    KeyError,
                ):
                    continue

            assert len(leaked_pids) == 0, "Expected zero child processes to be leaked"
        finally:
            clean_kill_pids(leaked_pids)


# =============================================================================
# Vector 5: Corrupted commands.json Recovery & POSIX Permissions
# =============================================================================


class TestChallengerCorruptedStorageRecovery:
    """Stress tests for corrupted JSON storage files, binary garbage, and POSIX permissions."""

    def test_truncated_json_recovers_to_empty_and_creates_backup(
        self, tmp_path: Path
    ) -> None:
        """Truncated unparseable JSON file creates backup and resets to empty."""
        store_path = tmp_path / "commands.json"
        store_path.write_text('{"commands": {"cmd1": {"name": "Test"', encoding="utf-8")

        store = CommandStore(storage_path=store_path)
        assert store.list_commands() == []

        backups = list(tmp_path.glob("commands.corrupt.*.bak"))
        assert len(backups) == 1
        assert '{"commands":' in backups[0].read_text(encoding="utf-8")

    def test_binary_garbage_recovers_safely(self, tmp_path: Path) -> None:
        """Binary garbage (non-UTF8 bytes) creates backup and recovers safely."""
        store_path = tmp_path / "commands.json"
        store_path.write_bytes(os.urandom(1024))

        store = CommandStore(storage_path=store_path)
        assert store.list_commands() == []

        backups = list(tmp_path.glob("commands.corrupt.*.bak"))
        assert len(backups) == 1

    def test_root_element_not_dictionary_recovers_safely(self, tmp_path: Path) -> None:
        """Root JSON array instead of dict creates backup and recovers."""
        store_path = tmp_path / "commands.json"
        store_path.write_text('["cmd1", "cmd2", "cmd3"]', encoding="utf-8")

        store = CommandStore(storage_path=store_path)
        assert store.list_commands() == []

        backups = list(tmp_path.glob("commands.corrupt.*.bak"))
        assert len(backups) == 1

    def test_commands_section_not_dictionary_recovers_safely(
        self, tmp_path: Path
    ) -> None:
        """Commands section as integer or array creates backup and recovers."""
        store_path = tmp_path / "commands.json"
        store_path.write_text('{"commands": 42}', encoding="utf-8")

        store = CommandStore(storage_path=store_path)
        assert store.list_commands() == []

        backups = list(tmp_path.glob("commands.corrupt.*.bak"))
        assert len(backups) == 1

    def test_partial_corrupted_records_skipped_while_valid_loaded(
        self, tmp_path: Path
    ) -> None:
        """Individual invalid entries (empty name, missing command) are skipped, valid preserved."""
        store_path = tmp_path / "commands.json"
        payload = {
            "commands": {
                "ok1": {"name": "Valid 1", "command": "echo 1"},
                "bad_name": {"name": "", "command": "echo 2"},
                "bad_cmd": {"name": "No Command"},
                "bad_type": "string_instead_of_dict",
                "ok2": {"name": "Valid 2", "command": "echo 3"},
            }
        }
        store_path.write_text(json.dumps(payload), encoding="utf-8")

        store = CommandStore(storage_path=store_path)
        cmds = store.list_commands()
        assert len(cmds) == 2
        keys = {c.key for c in cmds}
        assert keys == {"ok1", "ok2"}

    def test_completely_empty_file_handled_without_backup(self, tmp_path: Path) -> None:
        """0-byte empty file loads cleanly as empty dict without creating corrupt backup."""
        store_path = tmp_path / "commands.json"
        store_path.write_text("", encoding="utf-8")

        store = CommandStore(storage_path=store_path)
        assert store.list_commands() == []
        backups = list(tmp_path.glob("commands.corrupt.*.bak"))
        assert len(backups) == 0

    def test_permission_denied_on_load_handled_gracefully(self, tmp_path: Path) -> None:
        """File with 000 permissions defaults to empty dict without crashing."""
        store_path = tmp_path / "commands.json"
        store_path.write_text('{"commands": {}}', encoding="utf-8")
        store_path.chmod(0o000)

        try:
            store = CommandStore(storage_path=store_path)
            assert store.list_commands() == []
        finally:
            store_path.chmod(0o600)

    def test_permission_denied_on_save_raises_storage_write_error(
        self, tmp_path: Path
    ) -> None:
        """Verify write errors (e.g., target path is a directory or os.open fails) raise StorageWriteError."""
        # 1. Target path is a directory -> replace() fails with IsADirectoryError -> StorageWriteError
        target_dir = tmp_path / "directory_as_target"
        target_dir.mkdir()
        store = CommandStore(storage_path=target_dir)

        with pytest.raises(StorageWriteError, match="Is a directory"):
            store.add_command(name="Test", command="echo 1")

        # 2. PermissionError during os.open inside try block raises StorageWriteError
        store_path = tmp_path / "mock_perm" / "commands.json"
        store2 = CommandStore(storage_path=store_path)
        with (
            patch("os.open", side_effect=PermissionError("Mock Permission Denied")),
            pytest.raises(StorageWriteError, match="Mock Permission Denied"),
        ):
            store2.add_command(name="Test2", command="echo 2")

    def test_parent_directory_mkdir_unwrapped_permission_error_reproducer(self) -> None:
        """Empirical reproducer: When parent directory creation fails (e.g., in /root/...),

        parent_dir.mkdir() is outside the try/except block in CommandStore._save_to_disk,
        so it raises raw PermissionError instead of pengulink.core.exceptions.StorageWriteError.
        """
        store = CommandStore(
            storage_path="/root/forbidden_pengulink_test/commands.json"
        )
        # Verify directory creation error raises StorageWriteError
        with pytest.raises(StorageWriteError):
            store.add_command(name="Test", command="echo 1")

    def test_posix_permissions_0600_enforced_on_write(self, tmp_path: Path) -> None:
        """CommandStore creates 0o600 file and 0o700 directory."""
        sub_dir = tmp_path / "posix_test"
        store_path = sub_dir / "commands.json"

        store = CommandStore(storage_path=store_path)
        store.add_command(name="Secure", command="echo safe")

        file_mode = stat.S_IMODE(os.stat(store_path).st_mode)
        assert file_mode & 0o777 == 0o600

        dir_mode = stat.S_IMODE(os.stat(sub_dir).st_mode)
        assert dir_mode & 0o777 == 0o700


# =============================================================================
# Vector 6: Unicode, Special Characters & Shell Metacharacters
# =============================================================================


class TestChallengerUnicodeAndSpecialCharacters:
    """Stress tests for Unicode character sets, emojis, quotes, and metacharacters."""

    async def test_unicode_emojis_cjk_arabic_in_names_and_commands(
        self, tmp_path: Path
    ) -> None:
        """Verify full Unicode round-trip across storage, wire dict, and execution."""
        store_path = tmp_path / "unicode_commands.json"
        store = CommandStore(storage_path=store_path)

        entry = store.add_command(
            name="🚀 Super CJK 測試 & Arabic تجربة",
            command="echo 'Hello 🌍 測試 تجربة!'",
            key="key_🚀_1",
        )
        assert entry.key == "key_🚀_1"

        # Reload from disk
        store2 = CommandStore(storage_path=store_path)
        loaded = store2.get_command("key_🚀_1")
        assert loaded is not None
        assert loaded.name == entry.name

        # Wire serialization
        wire = store2.to_wire_dict()
        assert "key_🚀_1" in wire

        # Android JSON serialization round-trips
        android_json = store2.to_android_json_string()
        parsed = json.loads(android_json)
        assert "key_🚀_1" in parsed
        assert parsed["key_🚀_1"]["name"] == entry.name

        # Execution
        svc = LocalCommandService(command_store=store2)
        res = await svc.execute_command("key_🚀_1")
        assert res.success is True
        assert res.stdout == ["Hello 🌍 測試 تجربة!"]

    async def test_posix_shell_metacharacters_passed_as_literal_strings(self) -> None:
        """Verify shell=False prevents shell expansion of ;, |, &&, $(), *, and backticks."""
        svc = LocalCommandService()
        # In shell=True, ; would execute touch, | would pipe, $() would evaluate
        dangerous_cmd = (
            'echo "arg1; touch /tmp/pwned | id && $(uname) `whoami` * ? [a-z]"'
        )
        res = await svc.run_command_string(dangerous_cmd, key="meta_test")

        assert res.success is True
        assert res.exit_code == 0
        assert (
            "arg1; touch /tmp/pwned | id && $(uname) `whoami` * ? [a-z]"
            in res.stdout[0]
        )
        # Verify /tmp/pwned was not created
        assert not Path("/tmp/pwned").exists()

    async def test_nested_quotes_and_complex_spacing_in_arguments(self) -> None:
        """Verify arguments with internal quotes and spacing are correctly preserved."""
        svc = LocalCommandService()
        cmd = 'echo "first arg" \'second arg with "nested" quotes\' third'
        res = await svc.run_command_string(cmd, key="quotes_test")

        assert res.success is True
        assert res.stdout == ['first arg second arg with "nested" quotes third']

    async def test_empty_and_whitespace_command_rejected(self) -> None:
        """Empty or whitespace command string raises ValueError."""
        svc = LocalCommandService()
        with pytest.raises(ValueError, match="Cannot execute empty command string"):
            await svc.run_command_string("   ")

    async def test_unclosed_quote_raises_value_error_from_shlex(self) -> None:
        """Unclosed quotes in command string raise ValueError from shlex.split."""
        svc = LocalCommandService()
        with pytest.raises(ValueError, match="No closing quotation"):
            await svc.run_command_string('echo "unclosed quote')

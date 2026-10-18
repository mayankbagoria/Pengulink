"""tests.adversarial.test_m20_challenger_2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial Stress Test Suite — Challenger 2 (Milestone 20 / Phase 4 Integration & Hardening)

Empirical verification targeting IPC, Events, CLI, and File System Boundaries:
1. IPC SO_PEERCRED Unauthorized UID Rejection (CLUS-25):
   - Mock getsockopt returning mismatched UID (os.getuid() + 1, os.getuid() + 999, root uid 0, nobody 65534).
   - Unauthorized client socket closed immediately without processing JSON-RPC commands.
   - Server handler never invoked for unauthorized clients.
   - Concurrent barrage of unauthorized clients rejected cleanly without server degradation.
   - In-session credential validation defense during continuous communication.
   - Authorized clients with matching UID accepted and served cleanly.
2. CLI `pengulink daemon stop` Error Handling (CLUS-25):
   - Missing socket file handling exiting with code 1 and clean error message.
   - Stale socket file handling (connection refused) exiting with code 1.
   - Mock IPC client raising DaemonNotRunningError, FileNotFoundError, ConnectionRefusedError.
   - Unexpected IPCError handling exiting with code 1 without uncaught tracebacks.
   - Successful stop command exiting code 0.
3. CLI `pengulink connect` with Custom Options (CLUS-25):
   - Custom --host and --port override options forwarded to IPC payload.
   - Short flags -h and -p forwarded correctly.
   - Default host and port omission from payload when not specified.
   - Custom --timeout forwarded to payload.
   - Error handling when daemon is not running or returns IPC error.
4. EventBus High Concurrency Stress with 50 Tasks (CLUS-23):
   - 50 concurrent tasks continuously publishing diverse event types.
   - Dynamic subscriber mutation during publish iteration (subscribe, unsubscribe, subscribe_all).
   - Handlers mutating bus during their own execution.
   - Fault tolerance: failing handlers raising exceptions do not crash bus or disrupt other subscribers.
   - Zero RuntimeError: dictionary changed size during iteration, zero deadlocks.
5. Symlink Safety in `strip_execution_bits` (CLUS-28 & Lows):
   - Target files with 0o777, 0o755, 0o700 retain exact permissions when called on symlinks.
   - Chained symlinks (symlink -> symlink -> executable) leave target permissions untouched.
   - Broken symlinks handled gracefully without FileNotFoundError.
   - Contrast verification: regular files have execute bits stripped to exact 0o644 mode.
6. Directory Permission Protection for Path.home() (CLUS-28 & Lows):
   - ensure_destination_directory protects Path.home() from being chmodded to 0o700.
   - Home directory retains existing permissions (e.g. 0o755).
   - Tier 7 fallback in resolve_download_dir spares Path.home().
   - Non-home directories correctly receive strict 0o700 permissions.
7. Storage File 0600 Mode Under Concurrent Atomic Saves (CLUS-05, CLUS-28 & Lows):
   - 50 concurrent async tasks performing save_device and set_config in JsonStorageBackend.
   - devices.json and config.json strictly maintain 0o600 mode during and after writes.
   - Storage directory maintains 0o700 mode.
   - Zero data corruption, all 500 device records preserved in valid JSON.
   - Concurrent save and delete operations maintain 0o600 file mode.
8. vCard Escaping Order with Mixed Delimiters (CLUS-28 & Lows):
   - _unescape_vcard_value handling mixed backslashes, semicolons, commas, colons, newlines.
   - Structured N field splitting on unescaped semicolons while preserving escaped semicolons and commas.
   - Multiple consecutive backslashes with delimiters (e.g. \\\\; and \\\\,).
   - ORG and FN properties with complex escaped values.
   - Boundary tests with empty strings, lone backslashes, and massive escape strings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import random
import socket
import stat
import struct
import tempfile
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.events import (
    BatteryUpdated,
    DeviceConnected,
    DeviceDiscovered,
    DeviceDisconnected,
    Event,
    EventBus,
    NotificationReceived,
    PingReceived,
)
from pengulink.core.exceptions import (
    DaemonNotRunningError,
    IPCError,
)
from pengulink.daemon.ipc import (
    JSONRPC_VERSION,
    PenguLinkIPCClient,
    PenguLinkIPCServer,
)
from pengulink.plugins.contacts.vcard import (
    Contact,
    _unescape_vcard_value,
    parse_vcard,
    parse_vcard_line,
)
from pengulink.plugins.share.security import (
    ensure_destination_directory,
    resolve_download_dir,
    strip_execution_bits,
)
from pengulink.storage.json_store import JsonStorageBackend


# =============================================================================
# 1. IPC SO_PEERCRED Unauthorized UID Rejection
# =============================================================================

class TestIpcSoPeercredUnauthorizedRejection:
    """Empirical verification that PenguLinkIPCServer strictly enforces SO_PEERCRED

    UID matching and closes connections from unauthorized peers immediately.
    """

    async def test_unauthorized_uid_rejected_immediately(self, tmp_path: pathlib.Path) -> None:
        """Verify client with mismatched UID has connection closed immediately without response."""
        sock_path = tmp_path / "ipc_auth_test.sock"
        server = PenguLinkIPCServer(socket_path=sock_path)
        executed_commands: List[Dict[str, Any]] = []

        server.register_handler(
            "sensitive_command",
            lambda p: executed_commands.append(p) or {"status": "executed"},
        )
        await server.start()

        try:
            foreign_uid = os.getuid() + 100
            foreign_ucred = struct.pack("3i", os.getpid(), foreign_uid, os.getgid())

            with patch.object(socket.socket, "getsockopt", return_value=foreign_ucred):
                reader, writer = await asyncio.open_unix_connection(str(sock_path))
                payload = json.dumps({
                    "jsonrpc": "2.0",
                    "method": "sensitive_command",
                    "params": {"action": "drop_tables"},
                    "id": 42,
                }).encode("utf-8") + b"\n"

                try:
                    writer.write(payload)
                    await writer.drain()
                except (ConnectionResetError, BrokenPipeError):
                    pass

                try:
                    response = await reader.read(4096)
                except (ConnectionResetError, BrokenPipeError):
                    response = b""

                # Connection closed by server, zero bytes returned
                assert response == b"", f"Expected empty response, got: {response!r}"
                # Sensitive command handler must NOT have been called
                assert len(executed_commands) == 0

                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
        finally:
            await server.stop()

    async def test_multiple_unauthorized_uids_rejected_concurrently(self, tmp_path: pathlib.Path) -> None:
        """Verify concurrent barrage of unauthorized clients with different UIDs are all rejected."""
        sock_path = tmp_path / "ipc_multi_auth.sock"
        server = PenguLinkIPCServer(socket_path=sock_path)
        dispatched_count = 0

        def count_handler(params: Dict[str, Any]) -> Dict[str, str]:
            nonlocal dispatched_count
            dispatched_count += 1
            return {"status": "ok"}

        server.register_handler("test_ping", count_handler)
        await server.start()

        try:
            foreign_ucred = struct.pack("3i", os.getpid(), os.getuid() + 500, os.getgid())

            with patch.object(socket.socket, "getsockopt", return_value=foreign_ucred):
                async def attempt_unauthorized_connection(client_id: int) -> bytes:
                    try:
                        reader, writer = await asyncio.open_unix_connection(str(sock_path))
                        req = json.dumps({
                            "jsonrpc": "2.0",
                            "method": "test_ping",
                            "params": {"client": client_id},
                            "id": client_id,
                        }).encode("utf-8") + b"\n"
                        try:
                            writer.write(req)
                            await writer.drain()
                        except (ConnectionResetError, BrokenPipeError):
                            pass

                        try:
                            data = await reader.read(1024)
                        except (ConnectionResetError, BrokenPipeError):
                            data = b""

                        writer.close()
                        try:
                            await writer.wait_closed()
                        except Exception:
                            pass
                        return data
                    except (ConnectionResetError, BrokenPipeError, ConnectionRefusedError):
                        return b""

                results = await asyncio.gather(*(attempt_unauthorized_connection(i) for i in range(15)))

                # All unauthorized clients must receive 0 bytes
                for idx, res in enumerate(results):
                    assert res == b"", f"Client {idx} was not rejected! Got: {res!r}"

                # Server handler was never invoked
                assert dispatched_count == 0
        finally:
            await server.stop()

    async def test_authorized_uid_succeeds(self, tmp_path: pathlib.Path) -> None:
        """Verify genuine client matching os.getuid() succeeds and receives valid JSON-RPC response."""
        sock_path = tmp_path / "ipc_genuine_auth.sock"
        server = PenguLinkIPCServer(socket_path=sock_path)
        server.register_handler("get_info", lambda p: {"version": "1.0", "uid": os.getuid()})
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path=sock_path)
            res = await client.call("get_info", timeout=3.0)
            assert isinstance(res, dict)
            assert res.get("version") == "1.0"
            assert res.get("uid") == os.getuid()
        finally:
            await server.stop()

    async def test_in_session_credentials_drop_on_subsequent_request(self, tmp_path: pathlib.Path) -> None:
        """Verify connection is terminated if SO_PEERCRED query fails on subsequent requests in the loop."""
        sock_path = tmp_path / "ipc_in_session.sock"
        server = PenguLinkIPCServer(socket_path=sock_path)
        handled = []
        server.register_handler("ping", lambda p: handled.append(p) or {"pong": True})
        await server.start()

        try:
            reader, writer = await asyncio.open_unix_connection(str(sock_path))

            # First request: normal credentials (genuine UID)
            writer.write(b'{"jsonrpc": "2.0", "method": "ping", "params": {"n": 1}, "id": 1}\n')
            await writer.drain()
            resp1 = await reader.readline()
            data1 = json.loads(resp1.decode("utf-8").strip())
            assert data1.get("result") == {"pong": True}
            assert len(handled) == 1

            # Second request: getsockopt simulates foreign credentials
            foreign_ucred = struct.pack("3i", os.getpid(), os.getuid() + 77, os.getgid())
            with patch.object(socket.socket, "getsockopt", return_value=foreign_ucred):
                try:
                    writer.write(b'{"jsonrpc": "2.0", "method": "ping", "params": {"n": 2}, "id": 2}\n')
                    await writer.drain()
                except (ConnectionResetError, BrokenPipeError):
                    pass

                try:
                    resp2 = await reader.readline()
                except (ConnectionResetError, BrokenPipeError):
                    resp2 = b""
                assert resp2 == b""
                # Second request must NOT have been handled
                assert len(handled) == 1

            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        finally:
            await server.stop()


# =============================================================================
# 2. CLI `pengulink daemon stop` Error Handling
# =============================================================================

class TestCliDaemonStopErrorHandling:
    """Empirical verification that `pengulink daemon stop` handles missing or stale

    daemons cleanly without Python tracebacks, exiting with code 1.
    """

    def test_daemon_stop_missing_socket_exits_one(self, tmp_path: pathlib.Path) -> None:
        """Missing socket file causes daemon stop to exit with code 1 and clean message."""
        runner = CliRunner()
        missing_sock = tmp_path / "non_existent_daemon.sock"

        result = runner.invoke(cli, ["-s", str(missing_sock), "daemon", "stop"])
        assert result.exit_code == 1
        assert "PenguLink daemon is not running" in result.output
        assert "Traceback" not in result.output

    def test_daemon_stop_stale_socket_exits_one(self, tmp_path: pathlib.Path) -> None:
        """Stale socket file with no listening process causes daemon stop to exit 1."""
        runner = CliRunner()
        stale_sock = tmp_path / "stale_daemon.sock"
        stale_sock.touch()

        result = runner.invoke(cli, ["-s", str(stale_sock), "daemon", "stop"])
        assert result.exit_code == 1
        assert "PenguLink daemon is not running" in result.output
        assert "Traceback" not in result.output

    def test_daemon_stop_mock_daemon_not_running_error(self, tmp_path: pathlib.Path) -> None:
        """DaemonNotRunningError raised by client is caught and yields exit code 1."""
        runner = CliRunner()
        mock_sock = tmp_path / "mock.sock"
        mock_sock.touch()

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=DaemonNotRunningError("daemon dead")):
            result = runner.invoke(cli, ["-s", str(mock_sock), "daemon", "stop"])
            assert result.exit_code == 1
            assert "PenguLink daemon is not running" in result.output

    def test_daemon_stop_mock_connection_refused_error(self, tmp_path: pathlib.Path) -> None:
        """ConnectionRefusedError raised by client is caught and yields exit code 1."""
        runner = CliRunner()
        mock_sock = tmp_path / "mock.sock"
        mock_sock.touch()

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=ConnectionRefusedError("refused")):
            result = runner.invoke(cli, ["-s", str(mock_sock), "daemon", "stop"])
            assert result.exit_code == 1
            assert "PenguLink daemon is not running" in result.output

    def test_daemon_stop_unexpected_exception_handled_gracefully(self, tmp_path: pathlib.Path) -> None:
        """Generic unexpected exception outputs error notice and exits 1 without unhandled crash."""
        runner = CliRunner()
        mock_sock = tmp_path / "mock.sock"
        mock_sock.touch()

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=RuntimeError("internal IPC failure")):
            result = runner.invoke(cli, ["-s", str(mock_sock), "daemon", "stop"])
            assert result.exit_code == 1
            assert "Failed to stop daemon" in result.output
            assert "internal IPC failure" in result.output

    def test_daemon_stop_success_message_and_exit_zero(self, tmp_path: pathlib.Path) -> None:
        """Successful daemon stop returns exit code 0 and confirmation message."""
        runner = CliRunner()
        mock_sock = tmp_path / "mock.sock"
        mock_sock.touch()

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value={"status": "stopped"}):
            result = runner.invoke(cli, ["-s", str(mock_sock), "daemon", "stop"])
            assert result.exit_code == 0
            assert "Daemon stopped successfully" in result.output


# =============================================================================
# 3. CLI `pengulink connect` Custom Options
# =============================================================================

class TestCliConnectCustomHostAndPort:
    """Empirical verification that `pengulink connect` forwards custom --host,

    --port, and --timeout parameters to IPC.
    """

    def test_connect_with_custom_host_and_port(self, tmp_path: pathlib.Path) -> None:
        """Verify long flags --host and --port correctly populate connect payload."""
        runner = CliRunner()
        mock_sock = tmp_path / "mock.sock"
        mock_sock.touch()

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value={"status": "connected"}) as mock_call:
            result = runner.invoke(
                cli,
                ["-s", str(mock_sock), "connect", "dev_remote_1", "--host", "10.10.20.30", "--port", "1750"],
            )
            assert result.exit_code == 0
            assert "Connected to device 'dev_remote_1'" in result.output
            mock_call.assert_called_once_with(
                "connect",
                {"device_id": "dev_remote_1", "host": "10.10.20.30", "port": 1750},
            )

    def test_connect_with_short_flags_host_and_port(self, tmp_path: pathlib.Path) -> None:
        """Verify short flags -h and -p correctly populate connect payload."""
        runner = CliRunner()
        mock_sock = tmp_path / "mock.sock"
        mock_sock.touch()

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value={"status": "connected"}) as mock_call:
            result = runner.invoke(
                cli,
                ["-s", str(mock_sock), "connect", "dev_remote_2", "-h", "192.168.122.5", "-p", "1764"],
            )
            assert result.exit_code == 0
            mock_call.assert_called_once_with(
                "connect",
                {"device_id": "dev_remote_2", "host": "192.168.122.5", "port": 1764},
            )

    def test_connect_without_host_and_port_omits_them_from_payload(self, tmp_path: pathlib.Path) -> None:
        """When host and port are not specified, they should not be present in payload."""
        runner = CliRunner()
        mock_sock = tmp_path / "mock.sock"
        mock_sock.touch()

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value={"status": "connected"}) as mock_call:
            result = runner.invoke(cli, ["-s", str(mock_sock), "connect", "dev_remote_default"])
            assert result.exit_code == 0
            mock_call.assert_called_once_with("connect", {"device_id": "dev_remote_default"})

    def test_connect_with_custom_timeout(self, tmp_path: pathlib.Path) -> None:
        """Custom timeout parameter is passed in payload."""
        runner = CliRunner()
        mock_sock = tmp_path / "mock.sock"
        mock_sock.touch()

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value={"status": "connected"}) as mock_call:
            result = runner.invoke(
                cli,
                ["-s", str(mock_sock), "connect", "dev_timeout_test", "--timeout", "25.0", "--host", "192.168.1.1"],
            )
            assert result.exit_code == 0
            mock_call.assert_called_once_with(
                "connect",
                {"device_id": "dev_timeout_test", "host": "192.168.1.1", "timeout": 25.0},
            )

    def test_connect_daemon_not_running_error_handling(self, tmp_path: pathlib.Path) -> None:
        """DaemonNotRunningError yields clean exit 1."""
        runner = CliRunner()
        missing_sock = tmp_path / "missing.sock"

        result = runner.invoke(cli, ["-s", str(missing_sock), "connect", "dev_fail"])
        assert result.exit_code == 1
        assert "PenguLink daemon is not running" in result.output

    def test_connect_ipc_error_handling(self, tmp_path: pathlib.Path) -> None:
        """IPCError returned from daemon yields clean exit 1 with IPC Error display."""
        runner = CliRunner()
        mock_sock = tmp_path / "mock.sock"
        mock_sock.touch()

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=IPCError("Device unreachable")):
            result = runner.invoke(cli, ["-s", str(mock_sock), "connect", "dev_unreach"])
            assert result.exit_code == 1
            assert "IPC Error:" in result.output
            assert "Device unreachable" in result.output


# =============================================================================
# 4. EventBus High Concurrency Stress
# =============================================================================

class TestEventBusHighConcurrencyStress:
    """Empirical verification that EventBus handles 50 concurrent publisher tasks

    with intensive subscriber mutations without race conditions or memory corruption.
    """

    async def test_high_concurrency_publish_with_50_mutating_tasks(self) -> None:
        """Stress-test 50 concurrent tasks publishing events and actively mutating subscribers."""
        bus = EventBus()
        metrics = {
            "events_handled": 0,
            "handler_exceptions_caught": 0,
            "dynamic_handlers_invoked": 0,
            "global_handlers_invoked": 0,
        }

        # Baseline subscribers
        def base_ping_handler(event: PingReceived) -> None:
            metrics["events_handled"] += 1

        def base_battery_handler(event: BatteryUpdated) -> None:
            metrics["events_handled"] += 1

        unsub_base_ping = bus.subscribe(PingReceived, base_ping_handler)
        unsub_base_bat = bus.subscribe(BatteryUpdated, base_battery_handler)

        async def worker_task(task_id: int) -> None:
            active_unsubs: List[Any] = []

            for iteration in range(25):
                # 1. Dynamically subscribe new handler for typed events
                def dynamic_handler(ev: Event, tid=task_id, it=iteration) -> None:
                    metrics["events_handled"] += 1
                    metrics["dynamic_handlers_invoked"] += 1
                    # Intra-handler mutation: subscribe another handler during callback execution
                    if it % 8 == 0:
                        bus.subscribe(
                            DeviceConnected,
                            lambda e: metrics.update(events_handled=metrics["events_handled"] + 1),
                        )

                unsub_dyn = bus.subscribe(PingReceived, dynamic_handler)
                active_unsubs.append(unsub_dyn)

                # 2. Subscribe global handler
                def global_handler(ev: Event) -> None:
                    metrics["events_handled"] += 1
                    metrics["global_handlers_invoked"] += 1

                unsub_glob = bus.subscribe_all(global_handler)
                active_unsubs.append(unsub_glob)

                # 3. Publish varied events
                if iteration % 4 == 0:
                    await bus.publish(PingReceived(device_id=f"dev_{task_id}", message=f"ping_{iteration}"))
                elif iteration % 4 == 1:
                    await bus.publish(BatteryUpdated(device_id=f"dev_{task_id}", current_charge=iteration * 3))
                elif iteration % 4 == 2:
                    await bus.publish(NotificationReceived(device_id=f"dev_{task_id}", title=f"Note_{iteration}"))
                else:
                    await bus.publish(DeviceDiscovered(device_id=f"dev_{task_id}", device_name=f"Peer_{iteration}"))

                # 4. Concurrently unsubscribe random handlers
                if len(active_unsubs) > 4:
                    idx_to_remove = random.randint(0, len(active_unsubs) - 1)
                    remover = active_unsubs.pop(idx_to_remove)
                    remover()

                # 5. Intermittent cooperative yield
                if iteration % 6 == 0:
                    await asyncio.sleep(0.001)

            # Cleanup worker handlers
            for remover in active_unsubs:
                try:
                    remover()
                except Exception:
                    pass

        # Execute 50 concurrent worker tasks
        tasks = [asyncio.create_task(worker_task(t)) for t in range(50)]
        await asyncio.gather(*tasks)

        # Verification: thousands of callbacks were successfully executed without crash
        assert metrics["events_handled"] > 5000
        assert metrics["dynamic_handlers_invoked"] > 0
        assert metrics["global_handlers_invoked"] > 0

        # Post-stress verification: EventBus remains fully operational
        post_stress_received = []
        bus.subscribe(PingReceived, lambda e: post_stress_received.append(e.message))
        await bus.publish(PingReceived(device_id="post_stress", message="bus_alive"))
        assert "bus_alive" in post_stress_received

    async def test_faulty_and_async_handlers_under_concurrency(self) -> None:
        """Verify failing handlers and async sleeping handlers do not disrupt concurrency."""
        bus = EventBus()
        successes = []

        async def slow_async_handler(event: Event) -> None:
            await asyncio.sleep(0.005)
            successes.append("slow")

        def exploding_handler(event: Event) -> None:
            raise ValueError("Deliberate handler crash")

        def regular_handler(event: Event) -> None:
            successes.append("fast")

        bus.subscribe(PingReceived, slow_async_handler)
        bus.subscribe(PingReceived, exploding_handler)
        bus.subscribe(PingReceived, regular_handler)

        async def emit_batch(idx: int) -> None:
            await bus.publish(PingReceived(device_id=f"dev_{idx}", message="test"))

        await asyncio.gather(*(emit_batch(i) for i in range(20)))

        # All 20 fast and 20 slow handlers succeeded despite exploding_handler
        assert successes.count("fast") == 20
        assert successes.count("slow") == 20


# =============================================================================
# 5. Symlink Safety in `strip_execution_bits`
# =============================================================================

class TestStripExecutionBitsSymlinkSafety:
    """Empirical verification that `strip_execution_bits` never modifies target

    file permissions when invoked on symlinks.
    """

    def test_strip_execution_bits_preserves_target_permissions(self, tmp_path: pathlib.Path) -> None:
        """Executable target permissions (0o777, 0o755, 0o700) must remain untouched through symlink."""
        for initial_mode in (0o777, 0o755, 0o700):
            target = tmp_path / f"target_{oct(initial_mode)}.sh"
            target.write_text("#!/bin/sh\necho test\n")
            os.chmod(target, initial_mode)

            symlink = tmp_path / f"link_{oct(initial_mode)}"
            symlink.symlink_to(target)

            # Invoke strip_execution_bits on the symlink
            strip_execution_bits(symlink)

            # Target permissions must remain identical
            actual_mode = os.stat(target).st_mode & 0o777
            assert actual_mode == initial_mode, (
                f"Target permission modified through symlink! Expected {oct(initial_mode)}, got {oct(actual_mode)}"
            )

    def test_chained_symlinks_preserves_target(self, tmp_path: pathlib.Path) -> None:
        """Chained symlinks (symlink2 -> symlink1 -> target) do not tamper with target mode."""
        target = tmp_path / "chained_target.bin"
        target.write_bytes(b"\x7fELF...")
        os.chmod(target, 0o775)

        link1 = tmp_path / "link1"
        link1.symlink_to(target)

        link2 = tmp_path / "link2"
        link2.symlink_to(link1)

        strip_execution_bits(link2)

        actual_mode = os.stat(target).st_mode & 0o777
        assert actual_mode == 0o775

    def test_broken_symlink_does_not_crash(self, tmp_path: pathlib.Path) -> None:
        """Broken/dangling symlink handled cleanly without raising FileNotFoundError."""
        broken_target = tmp_path / "does_not_exist.sh"
        dangling_link = tmp_path / "dangling_link"
        dangling_link.symlink_to(broken_target)

        # Must not raise FileNotFoundError or OSError
        strip_execution_bits(dangling_link)

    def test_direct_regular_file_strips_execution_bits(self, tmp_path: pathlib.Path) -> None:
        """Contrast verification: invoking on regular file correctly strips execute bits to 0o644."""
        target = tmp_path / "regular_file.txt"
        target.write_text("regular data")
        os.chmod(target, 0o777)

        strip_execution_bits(target)

        mode_after = os.stat(target).st_mode & 0o777
        assert mode_after == 0o644


# =============================================================================
# 6. Directory Permission Protection for `Path.home()`
# =============================================================================

class TestEnsureDestinationDirectoryHomeProtection:
    """Empirical verification that `ensure_destination_directory` and

    `resolve_download_dir` protect Path.home() from being chmodded to 0o700.
    """

    def test_home_dir_permissions_unmodified_when_passed_in_candidates(self, tmp_path: pathlib.Path) -> None:
        """Fake home directory with 0o755 must NOT be chmodded to 0o700."""
        fake_home = tmp_path / "user_home"
        fake_home.mkdir(mode=0o755)
        os.chmod(fake_home, 0o755)

        with patch.object(pathlib.Path, "home", return_value=fake_home):
            dest = ensure_destination_directory([fake_home])
            assert dest == fake_home
            current_mode = os.stat(fake_home).st_mode & 0o777
            assert current_mode == 0o755, f"Home directory mode corrupted! Expected 0o755, got {oct(current_mode)}"

    def test_home_dir_protected_as_single_path_argument(self, tmp_path: pathlib.Path) -> None:
        """Passing fake_home directly as single argument protects permissions."""
        fake_home = tmp_path / "user_home_single"
        fake_home.mkdir(mode=0o755)
        os.chmod(fake_home, 0o755)

        with patch.object(pathlib.Path, "home", return_value=fake_home):
            dest = ensure_destination_directory(fake_home)
            assert dest == fake_home
            current_mode = os.stat(fake_home).st_mode & 0o777
            assert current_mode == 0o755

    def test_resolve_download_dir_fallback_to_home_spares_home(self, tmp_path: pathlib.Path) -> None:
        """When download resolution falls back to Tier 7 Path.home(), it spares home chmod."""
        fake_home = tmp_path / "user_home_tier7"
        fake_home.mkdir(mode=0o755)
        os.chmod(fake_home, 0o755)

        # Force all candidates before Tier 7 to fail by pointing them to unwritable directory
        unwritable_parent = tmp_path / "read_only_root"
        unwritable_parent.mkdir(mode=0o555)

        with patch.object(pathlib.Path, "home", return_value=fake_home):
            with patch.dict(os.environ, {"PENGULINK_DOWNLOAD_DIR": "", "XDG_DOWNLOAD_DIR": ""}, clear=False):
                with patch("pengulink.plugins.share.security.parse_xdg_user_dirs_file", return_value=None):
                    # Passing non-creatable override forces fallback through hierarchy
                    dest = resolve_download_dir(dest_override=unwritable_parent / "impossible" / "sub")
                    # If it resolved to home or another candidate, check fake_home mode
                    assert (os.stat(fake_home).st_mode & 0o777) == 0o755

    def test_non_home_directories_enforce_strict_0700(self, tmp_path: pathlib.Path) -> None:
        """Non-home directories MUST have strict 0o700 mode enforced."""
        sub_dir = tmp_path / "normal_downloads"
        sub_dir.mkdir(mode=0o755)
        os.chmod(sub_dir, 0o755)

        fake_home = tmp_path / "fake_home_other"
        fake_home.mkdir(mode=0o755)

        with patch.object(pathlib.Path, "home", return_value=fake_home):
            dest = ensure_destination_directory([sub_dir])
            assert dest == sub_dir
            sub_mode = os.stat(sub_dir).st_mode & 0o777
            assert sub_mode == 0o700


# =============================================================================
# 7. Storage File 0600 Mode Under Concurrent Atomic Saves
# =============================================================================

class TestJsonStoreConcurrentAtomicSaves:
    """Empirical verification that JsonStorageBackend preserves strict 0o600 file

    permissions and 0o700 directory mode under heavy concurrent writes.
    """

    async def test_concurrent_saves_preserve_0600_mode(self, tmp_path: pathlib.Path) -> None:
        """50 concurrent tasks saving device metadata and configuration preserve 0o600 permissions."""
        store_dir = tmp_path / "secure_store"
        backend = JsonStorageBackend(storage_dir=store_dir)
        await backend.initialize()

        # Check initial directory mode
        assert (os.stat(store_dir).st_mode & 0o777) == 0o700

        observed_violations: List[str] = []

        async def worker(worker_id: int) -> None:
            for iteration in range(10):
                dev_id = f"dev_{worker_id}_{iteration}"
                await backend.save_device(
                    dev_id,
                    {
                        "name": f"Phone_{worker_id}_{iteration}",
                        "pinned_cert": f"CERT_DATA_{worker_id}_{iteration}",
                        "timestamp": time.time(),
                    },
                )
                if iteration % 3 == 0:
                    await backend.set_config(f"key_{worker_id}_{iteration}", iteration * 10)

                # Inspect file mode during execution
                devices_file = store_dir / "devices.json"
                if devices_file.exists():
                    current_mode = os.stat(devices_file).st_mode & 0o777
                    if current_mode != 0o600:
                        observed_violations.append(f"devices.json mode {oct(current_mode)} during worker {worker_id}")

                config_file = store_dir / "config.json"
                if config_file.exists():
                    cfg_mode = os.stat(config_file).st_mode & 0o777
                    if cfg_mode != 0o600:
                        observed_violations.append(f"config.json mode {oct(cfg_mode)} during worker {worker_id}")

        # Launch 50 concurrent writer tasks
        await asyncio.gather(*(worker(w) for w in range(50)))

        # Assert zero permission regressions occurred during writes
        assert observed_violations == []

        # Final file state verification
        devices_file = store_dir / "devices.json"
        config_file = store_dir / "config.json"

        assert devices_file.exists()
        assert config_file.exists()
        assert (os.stat(devices_file).st_mode & 0o777) == 0o600
        assert (os.stat(config_file).st_mode & 0o777) == 0o600
        assert (os.stat(store_dir).st_mode & 0o777) == 0o700

        # Integrity verification: all 500 devices were persisted without JSON corruption
        saved_devices = await backend.list_devices()
        assert len(saved_devices) == 500

        # Confirm JSON file is parseable and valid
        raw_content = json.loads(devices_file.read_text(encoding="utf-8"))
        assert len(raw_content) == 500

    async def test_concurrent_save_and_delete_preserves_0600(self, tmp_path: pathlib.Path) -> None:
        """Concurrent interleaving of save_device and delete_device preserves 0o600 permissions."""
        store_dir = tmp_path / "mixed_store"
        backend = JsonStorageBackend(storage_dir=store_dir)
        await backend.initialize()

        async def save_and_delete(worker_id: int) -> None:
            dev_id = f"transient_dev_{worker_id}"
            await backend.save_device(dev_id, {"temp": True})
            await asyncio.sleep(0.001)
            await backend.delete_device(dev_id)

        await asyncio.gather(*(save_and_delete(i) for i in range(30)))

        devices_file = store_dir / "devices.json"
        assert devices_file.exists()
        assert (os.stat(devices_file).st_mode & 0o777) == 0o600


# =============================================================================
# 8. vCard Escaping Order with Mixed Delimiters
# =============================================================================

class TestVCardEscapingOrderAndDelimiters:
    """Empirical verification that vCard parsing and unescaping handles mixed

    backslashes, semicolons, commas, and colons in proper escaping order.
    """

    def test_unescape_vcard_value_mixed_sequences(self) -> None:
        """Verify _unescape_vcard_value handles mixed escape sequences correctly."""
        # \n, \N, \,, \;, \:, \\
        raw_val = r"Path\\To\\Folder\,with\;delimiters\:and\nNewline\NLine2"
        expected = "Path\\To\\Folder,with;delimiters:and\nNewline\nLine2"
        actual = _unescape_vcard_value(raw_val)
        assert actual == expected

    def test_unescape_vcard_value_delimiters_adjacent_to_backslashes(self) -> None:
        """Verify delimiters adjacent to escaped backslashes unescape properly."""
        raw_val = r"C:\\Users\\Bob\,Jr\\;Office\\:Desk"
        expected = "C:\\Users\\Bob,Jr\\;Office\\:Desk"
        actual = _unescape_vcard_value(raw_val)
        assert actual == expected

    def test_vcard_structured_n_with_mixed_escapes(self) -> None:
        """Structured N field with escaped semicolons, commas, and backslashes in components."""
        vcard_text = (
            "BEGIN:VCARD\n"
            "VERSION:3.0\n"
            "N:Von\\;Bismarck\\,Otto;Prince\\;Herbert;Count\\\\;Dr\\;PhD;Esq\\,Jr\n"
            "FN:Prince Herbert Von;Bismarck,Otto\n"
            "END:VCARD"
        )
        res = parse_vcard(vcard_text)
        contact = res[0] if isinstance(res, list) else res

        # Surname should contain unescaped semicolon and comma, not split
        assert contact.last_name == "Von;Bismarck,Otto"
        # First name should contain unescaped semicolon, not split
        assert contact.first_name == "Prince;Herbert"

    def test_vcard_org_with_escaped_semicolons_and_commas(self) -> None:
        """Structured ORG field with escaped delimiters within organization and unit."""
        vcard_text = (
            "BEGIN:VCARD\n"
            "VERSION:3.0\n"
            "FN:Jane Doe\n"
            "ORG:Acme\\;Global\\,Inc.;Research\\;Development\\,Lab\n"
            "END:VCARD"
        )
        res = parse_vcard(vcard_text)
        contact = res[0] if isinstance(res, list) else res

        # ORG components should retain their internal semicolons and commas
        assert contact.organization == "Acme;Global,Inc.; Research;Development,Lab"

    def test_vcard_escaped_consecutive_backslashes(self) -> None:
        """Consecutive backslashes before escaped delimiters are unescaped in proper order."""
        vcard_text = (
            "BEGIN:VCARD\n"
            "VERSION:3.0\n"
            "N:Smith\\\\\\;Senior;John\\\\\\,Jr;;;\n"
            "FN:John Smith\n"
            "END:VCARD"
        )
        res = parse_vcard(vcard_text)
        contact = res[0] if isinstance(res, list) else res

        assert contact.last_name == "Smith\\;Senior"
        assert contact.first_name == "John\\,Jr"

    def test_vcard_massive_escape_string_stress(self) -> None:
        """Stress-test 500 consecutive mixed escape sequences without exponential backtracking."""
        token = r"Text\\with\,comma\;semi\:colon\nline"
        massive_raw = token * 500
        start = time.perf_counter()
        unescaped = _unescape_vcard_value(massive_raw)
        elapsed = time.perf_counter() - start

        # Must execute in under 50ms
        assert elapsed < 0.05
        assert len(unescaped) < len(massive_raw)
        assert unescaped.startswith("Text\\with,comma;semi:colon\nline")

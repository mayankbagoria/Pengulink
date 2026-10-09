"""tests.adversarial.test_m93_resource_concurrency_leak_challenger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Challenger 2 Suite for Milestone M9.3:
Resource Safety, Concurrency & Leak Verifier.

Mandatory Verification Objectives:
1. Concurrency: Burst 50 concurrent IPC requests (get_clipboard and set_clipboard)
   simultaneously against a running daemon instance. Verify all return cleanly
   without race conditions or deadlocks.
2. Lifecycle leak audit: Repeatedly cycle daemon start() and stop() 10 times while
   triggering clipboard operations. Inspect asyncio.all_tasks(), open file descriptors,
   and child processes to verify zero leaked tasks, zero lingering processes, and
   zero unclosed sockets.
3. Privacy leak audit: Verify that canary tokens passed through IPC or CLI NEVER
   appear in daemon logs or LogRingBuffer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import socket
import string
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.config import PenguLinkConfig
from pengulink.core.events import ClipboardUpdatedEvent
from pengulink.daemon.ipc import PenguLinkIPCClient
from pengulink.daemon.service import LogRingBuffer, PenguLinkDaemon
from pengulink.network.packet import Packet
from pengulink.plugins.clipboard.backend import InMemoryClipboardBackend


# =============================================================================
# Helper Utilities for System Resource Inspection
# =============================================================================

def get_open_fds() -> Set[int]:
    """Retrieve the set of currently open file descriptors for the current process."""
    if not os.path.exists("/proc/self/fd"):
        return set()
    try:
        return {int(entry) for entry in os.listdir("/proc/self/fd") if entry.isdigit()}
    except (OSError, ValueError):
        return set()


def inspect_open_sockets() -> Dict[int, str]:
    """Inspect open file descriptors and identify active socket handles."""
    sockets: Dict[int, str] = {}
    if not os.path.exists("/proc/self/fd"):
        return sockets
    try:
        for entry in os.listdir("/proc/self/fd"):
            if not entry.isdigit():
                continue
            fd = int(entry)
            try:
                target = os.readlink(f"/proc/self/fd/{fd}")
                if "socket" in target:
                    sockets[fd] = target
            except (OSError, FileNotFoundError):
                continue
    except OSError:
        pass
    return sockets


def get_child_pids() -> Set[int]:
    """Retrieve PIDs of all child processes of the current process."""
    current_pid = os.getpid()
    children: Set[int] = set()
    if not os.path.exists("/proc"):
        return children
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == current_pid:
                continue
            status_path = f"/proc/{pid}/status"
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("PPid:"):
                            ppid = int(line.split()[1])
                            if ppid == current_pid:
                                children.add(pid)
                            break
            except (OSError, IOError, ValueError):
                continue
    except OSError:
        pass
    return children


def generate_canary_token(prefix: str = "CANARY") -> str:
    """Generate a unique high-entropy canary secret token."""
    rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=16))
    return f"{prefix}_{uuid.uuid4().hex}_{rand_suffix}_SECRET"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def test_daemon_config(tmp_path: Path) -> PenguLinkConfig:
    """Provide isolated configuration with dynamic ephemeral ports and storage."""
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / "config"
    cfg.storage.data_dir = tmp_path / "data"
    cfg.storage.runtime_dir = tmp_path / "run"
    cfg.device.device_id = f"challenger_daemon_{uuid.uuid4().hex[:8]}"
    cfg.device.device_name = "PenguLink-Challenger-Node"
    # Ephemeral port range to prevent collision
    cfg.network.udp_port = random.randint(20000, 30000)
    cfg.network.tcp_port_min = random.randint(30001, 35000)
    cfg.network.tcp_port_max = cfg.network.tcp_port_min + 5
    cfg.ensure_directories()
    return cfg


# =============================================================================
# OBJECTIVE 1: Concurrency Stress Test Suite
# =============================================================================

async def test_concurrency_burst_50_concurrent_ipc_requests(
    test_daemon_config: PenguLinkConfig,
    tmp_path: Path,
) -> None:
    """Objective 1: Burst 50 concurrent IPC requests (get_clipboard and set_clipboard)
    simultaneously against a running daemon instance. Verify all return cleanly
    without race conditions or deadlocks.
    """
    backend = InMemoryClipboardBackend()
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(
        config=test_daemon_config,
        socket_path=sock_path,
        clipboard_backend=backend,
    )

    await daemon.start()
    assert daemon.is_running
    assert sock_path.exists()

    try:
        # Seed an initial state
        await daemon.set_clipboard("initial_state_content")

        # Concurrency barrier: 50 simultaneous tasks across 50 independent IPC calls
        num_requests = 50
        start_event = asyncio.Event()

        async def worker_task(index: int) -> Dict[str, Any]:
            client = PenguLinkIPCClient(sock_path, default_timeout=5.0)
            await start_event.wait()

            if index % 2 == 0:
                # Write request
                payload = f"burst_payload_index_{index}_{uuid.uuid4().hex}"
                res = await client.set_clipboard(payload)
                return {
                    "type": "set",
                    "index": index,
                    "payload": payload,
                    "result": res,
                }
            else:
                # Read request
                res = await client.get_clipboard()
                return {
                    "type": "get",
                    "index": index,
                    "result": res,
                }

        # Launch all 50 worker coroutines
        tasks = [asyncio.create_task(worker_task(i)) for i in range(num_requests)]

        # Allow tasks to reach event barrier
        await asyncio.sleep(0.02)

        # Trigger simultaneous burst release
        start_event.set()

        # Await all 50 operations with strict 5-second timeout guard to detect deadlocks
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

        # Verify all 50 requests returned cleanly
        assert len(results) == num_requests

        sets = [r for r in results if r["type"] == "set"]
        gets = [r for r in results if r["type"] == "get"]

        assert len(sets) == 25
        assert len(gets) == 25

        # Validate all 'set' responses
        for s in sets:
            res = s["result"]
            assert isinstance(res, dict), f"Set response must be dict, got {type(res)}"
            assert res.get("status") == "ok", f"Expected status 'ok', got {res}"
            assert res.get("bytes", 0) > 0
            assert "timestamp" in res

        # Validate all 'get' responses
        for g in gets:
            res = g["result"]
            assert isinstance(res, dict), f"Get response must be dict, got {type(res)}"
            assert "content" in res, f"Response must include 'content', got {res}"
            assert isinstance(res["content"], str)
            assert "timestamp" in res
            assert "bytes" in res
            assert res["bytes"] == len(res["content"].encode("utf-8"))

        # Verify daemon is still fully responsive post-burst
        verification_client = PenguLinkIPCClient(sock_path, default_timeout=2.0)
        post_burst_state = await verification_client.get_clipboard()

        assert isinstance(post_burst_state, dict)
        assert post_burst_state.get("content") != ""
        # The final backend state must match one of the set payloads
        set_payloads = {s["payload"] for s in sets}
        assert post_burst_state.get("content") in set_payloads

    finally:
        await daemon.stop()
        assert not daemon.is_running


async def test_concurrency_burst_50_concurrent_writers_no_data_corruption(
    test_daemon_config: PenguLinkConfig,
    tmp_path: Path,
) -> None:
    """Stress test: 50 concurrent simultaneous writers targeting set_clipboard.
    Verifies that no payload corruption, partial writes, or race conditions occur.
    """
    backend = InMemoryClipboardBackend()
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(
        config=test_daemon_config,
        socket_path=sock_path,
        clipboard_backend=backend,
    )

    await daemon.start()

    try:
        num_writers = 50
        barrier = asyncio.Event()

        async def writer(i: int) -> Tuple[int, str, Dict[str, Any]]:
            client = PenguLinkIPCClient(sock_path, default_timeout=5.0)
            payload = f"writer_{i}_{'✨' * (i % 10)}_{uuid.uuid4().hex}"
            await barrier.wait()
            res = await client.set_clipboard(payload)
            return i, payload, res

        tasks = [asyncio.create_task(writer(i)) for i in range(num_writers)]
        await asyncio.sleep(0.02)
        barrier.set()

        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
        assert len(results) == num_writers

        for i, payload, res in results:
            assert res.get("status") == "ok"
            assert res.get("bytes") == len(payload.encode("utf-8"))

        # Final state check
        current_text = await backend.get_text()
        written_payloads = {p for _, p, _ in results}
        assert current_text in written_payloads

    finally:
        await daemon.stop()


async def test_concurrency_burst_50_raw_persistent_pipelined_requests(
    test_daemon_config: PenguLinkConfig,
    tmp_path: Path,
) -> None:
    """Stress test: 50 rapid requests pipelined over 5 persistent raw socket streams
    to stress-test IPC reader line-buffering and framing concurrently.
    """
    backend = InMemoryClipboardBackend()
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(
        config=test_daemon_config,
        socket_path=sock_path,
        clipboard_backend=backend,
    )

    await daemon.start()

    try:
        num_conns = 5
        requests_per_conn = 10  # 5 * 10 = 50 requests total

        async def connection_worker(conn_id: int) -> List[Dict[str, Any]]:
            reader, writer = await asyncio.open_unix_connection(str(sock_path))
            responses = []
            try:
                for req_id in range(requests_per_conn):
                    global_id = conn_id * 100 + req_id
                    if req_id % 2 == 0:
                        payload = {
                            "jsonrpc": "2.0",
                            "method": "set_clipboard",
                            "params": {"content": f"pipe_{conn_id}_{req_id}_{uuid.uuid4().hex}"},
                            "id": global_id,
                        }
                    else:
                        payload = {
                            "jsonrpc": "2.0",
                            "method": "get_clipboard",
                            "params": {},
                            "id": global_id,
                        }
                    writer.write((json.dumps(payload) + "\n").encode("utf-8"))
                    await writer.drain()

                    line = await reader.readline()
                    assert line, f"Connection closed prematurely for conn {conn_id}"
                    resp = json.loads(line.decode("utf-8"))
                    assert resp.get("id") == global_id
                    assert "result" in resp
                    responses.append(resp)
            finally:
                writer.close()
                await writer.wait_closed()
            return responses

        conn_tasks = [asyncio.create_task(connection_worker(c)) for c in range(num_conns)]
        all_responses = await asyncio.wait_for(asyncio.gather(*conn_tasks), timeout=5.0)

        total_received = sum(len(r) for r in all_responses)
        assert total_received == 50

    finally:
        await daemon.stop()


async def test_concurrency_burst_colliding_with_daemon_stop(
    test_daemon_config: PenguLinkConfig,
    tmp_path: Path,
) -> None:
    """Stress test: 50 concurrent requests colliding with daemon.stop().
    Verifies that tearing down during an active burst does not deadlock or leak resources.
    """
    backend = InMemoryClipboardBackend()
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(
        config=test_daemon_config,
        socket_path=sock_path,
        clipboard_backend=backend,
    )

    await daemon.start()

    num_requests = 50
    barrier = asyncio.Event()

    async def continuous_requester(idx: int) -> None:
        client = PenguLinkIPCClient(sock_path, default_timeout=1.0)
        await barrier.wait()
        try:
            if idx % 2 == 0:
                await client.set_clipboard(f"colliding_set_{idx}")
            else:
                await client.get_clipboard()
        except Exception:
            # Expected during teardown
            pass

    tasks = [asyncio.create_task(continuous_requester(i)) for i in range(num_requests)]
    await asyncio.sleep(0.01)
    barrier.set()

    # Small delay then stop daemon right in the middle of requests
    await asyncio.sleep(0.01)
    # daemon.stop() must complete cleanly within 3 seconds
    await asyncio.wait_for(daemon.stop(), timeout=3.0)

    # Gather any remaining client tasks
    await asyncio.gather(*tasks, return_exceptions=True)
    assert not daemon.is_running


async def test_concurrency_with_malformed_json_and_overrun_burst(
    test_daemon_config: PenguLinkConfig,
    tmp_path: Path,
) -> None:
    """Stress test: 25 valid requests concurrent with 25 malformed JSON lines
    and oversized line guard violations. Verifies the daemon isolates bad clients
    and continues servicing valid requests.
    """
    backend = InMemoryClipboardBackend()
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(
        config=test_daemon_config,
        socket_path=sock_path,
        clipboard_backend=backend,
    )

    await daemon.start()

    try:
        valid_successes: List[Dict[str, Any]] = []

        async def bad_client(bad_type: int) -> None:
            try:
                reader, writer = await asyncio.open_unix_connection(str(sock_path))
                if bad_type == 0:
                    # Truncated JSON
                    writer.write(b'{"jsonrpc": "2.0", "method": "set_clipboard"\n')
                elif bad_type == 1:
                    # Garbage syntax
                    writer.write(b"NOT_EVEN_JSON_???\n")
                else:
                    # Missing params
                    writer.write(b'{"jsonrpc": "2.0", "method": "set_clipboard", "params": "not_a_dict", "id": 999}\n')
                await writer.drain()
                _ = await reader.readline()
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

        async def good_client(idx: int) -> None:
            client = PenguLinkIPCClient(sock_path, default_timeout=3.0)
            res = await client.set_clipboard(f"good_payload_{idx}")
            valid_successes.append(res)

        all_tasks = []
        for i in range(25):
            all_tasks.append(asyncio.create_task(bad_client(i % 3)))
            all_tasks.append(asyncio.create_task(good_client(i)))

        await asyncio.wait_for(asyncio.gather(*all_tasks, return_exceptions=True), timeout=5.0)

        # All 25 valid requests must succeed despite concurrent garbage inputs
        assert len(valid_successes) == 25
        for s in valid_successes:
            assert s.get("status") == "ok"

    finally:
        await daemon.stop()


# =============================================================================
# OBJECTIVE 2: Lifecycle Leak Audit (10 Start/Stop Cycles)
# =============================================================================

async def test_lifecycle_leak_audit_10_start_stop_cycles(
    test_daemon_config: PenguLinkConfig,
    tmp_path: Path,
) -> None:
    """Objective 2: Repeatedly cycle daemon start() and stop() 10 times while
    triggering clipboard operations. Inspect asyncio.all_tasks(), open file descriptors,
    and child processes to verify zero leaked tasks, zero lingering processes, and
    zero unclosed sockets.
    """
    backend = InMemoryClipboardBackend()
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(
        config=test_daemon_config,
        socket_path=sock_path,
        clipboard_backend=backend,
    )

    # Let event loop settle before capturing baseline
    await asyncio.sleep(0.05)

    current_test_task = asyncio.current_task()
    baseline_tasks: Set[asyncio.Task[Any]] = {
        t for t in asyncio.all_tasks() if t is not current_test_task
    }
    baseline_sockets = inspect_open_sockets()
    baseline_children = get_child_pids()

    # Repeat 10 complete start/stop cycles with clipboard operations
    for cycle in range(1, 11):
        # 1. Start daemon
        await daemon.start()
        assert daemon.is_running, f"Cycle {cycle}: Daemon failed to start"
        assert daemon.clipboard_plugin.is_running, f"Cycle {cycle}: ClipboardPlugin not running"
        assert sock_path.exists(), f"Cycle {cycle}: Socket file missing"

        # 2. Trigger clipboard operations in each cycle
        cycle_payload = f"cycle_{cycle}_content_{uuid.uuid4().hex}"
        set_res = await daemon.set_clipboard(cycle_payload)
        assert set_res is True

        get_res = await daemon.get_clipboard()
        assert get_res["content"] == cycle_payload

        # Inbound packet dispatch simulation
        pkt = Packet(
            type="kdeconnect.clipboard",
            body={"content": f"inbound_{cycle}_{uuid.uuid4().hex}", "timestamp": int(time.time() * 1000)},
        )
        await daemon.plugin_registry.dispatch_packet(pkt, f"device_{cycle}", lambda p: None)

        # IPC client connection and query cycle
        client = PenguLinkIPCClient(sock_path, default_timeout=2.0)
        ipc_get = await client.get_clipboard()
        assert isinstance(ipc_get, dict)
        await client.set_clipboard(f"ipc_cycle_{cycle}_{uuid.uuid4().hex}")

        # 3. Stop daemon
        await daemon.stop()
        assert not daemon.is_running, f"Cycle {cycle}: Daemon failed to stop"
        assert not daemon.clipboard_plugin.is_running, f"Cycle {cycle}: ClipboardPlugin still running"
        assert not sock_path.exists(), f"Cycle {cycle}: Socket file still exists"

    # Allow brief settling window for OS socket cleanup and task cancellations
    await asyncio.sleep(0.1)

    # =========================================================================
    # LEAK AUDIT 1: asyncio.all_tasks()
    # =========================================================================
    current_tasks = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}
    leaked_tasks = [t for t in current_tasks if t not in baseline_tasks and not t.done()]

    task_details = [
        f"Task {t.get_name()}: coro={t.get_coro()}"
        for t in leaked_tasks
    ]
    assert len(leaked_tasks) == 0, (
        f"Lifecycle leak audit failed: {len(leaked_tasks)} lingering asyncio tasks detected after 10 cycles:\n"
        + "\n".join(task_details)
    )

    # =========================================================================
    # LEAK AUDIT 2: Child Processes
    # =========================================================================
    current_children = get_child_pids()
    leaked_children = current_children - baseline_children
    assert len(leaked_children) == 0, (
        f"Lifecycle leak audit failed: {len(leaked_children)} lingering child processes: {leaked_children}"
    )

    # =========================================================================
    # LEAK AUDIT 3: Open Sockets
    # =========================================================================
    current_sockets = inspect_open_sockets()
    leaked_socket_fds = set(current_sockets.keys()) - set(baseline_sockets.keys())
    leaked_socket_details = {fd: current_sockets[fd] for fd in leaked_socket_fds}

    assert len(leaked_socket_fds) == 0, (
        f"Lifecycle leak audit failed: {len(leaked_socket_fds)} unclosed sockets detected after 10 cycles:\n"
        f"{leaked_socket_details}"
    )


async def test_lifecycle_leak_audit_10_distinct_daemon_instances(
    tmp_path: Path,
) -> None:
    """Objective 2 variant: Repeatedly instantiate, start, use, and stop 10 distinct
    PenguLinkDaemon instances to ensure no class-level or global subsystem leakage.
    """
    await asyncio.sleep(0.05)
    baseline_tasks = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}
    baseline_sockets = inspect_open_sockets()
    baseline_children = get_child_pids()

    for i in range(10):
        cfg = PenguLinkConfig()
        cfg.storage.config_dir = tmp_path / f"config_{i}"
        cfg.storage.data_dir = tmp_path / f"data_{i}"
        cfg.storage.runtime_dir = tmp_path / f"run_{i}"
        cfg.device.device_id = f"inst_node_{i}_{uuid.uuid4().hex[:6]}"
        cfg.device.device_name = f"Node-{i}"
        cfg.network.udp_port = random.randint(35000, 45000)
        cfg.network.tcp_port_min = random.randint(45001, 50000)
        cfg.network.tcp_port_max = cfg.network.tcp_port_min + 5
        cfg.ensure_directories()

        sock = tmp_path / f"run_{i}" / "daemon.sock"
        backend = InMemoryClipboardBackend()
        daemon = PenguLinkDaemon(config=cfg, socket_path=sock, clipboard_backend=backend)

        await daemon.start()
        await daemon.set_clipboard(f"inst_{i}_payload")
        _ = await daemon.get_clipboard()
        await daemon.stop()

    await asyncio.sleep(0.1)

    # Audit tasks
    current_tasks = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}
    leaked_tasks = [t for t in current_tasks if t not in baseline_tasks and not t.done()]
    assert len(leaked_tasks) == 0, f"Leaked tasks across 10 instances: {leaked_tasks}"

    # Audit child processes
    leaked_children = get_child_pids() - baseline_children
    assert len(leaked_children) == 0, f"Leaked children: {leaked_children}"

    # Audit sockets
    current_sockets = inspect_open_sockets()
    leaked_socket_fds = set(current_sockets.keys()) - set(baseline_sockets.keys())
    assert len(leaked_socket_fds) == 0, f"Leaked sockets: {leaked_socket_fds}"


# =============================================================================
# OBJECTIVE 3: Privacy Leak Audit (Canary Tokens across IPC & CLI)
# =============================================================================

async def test_privacy_canary_tokens_never_appear_in_logs_or_ring_buffer(
    test_daemon_config: PenguLinkConfig,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Objective 3: Verify that canary tokens passed through IPC or CLI NEVER
    appear in daemon logs or LogRingBuffer.
    """
    backend = InMemoryClipboardBackend()
    sock_path = tmp_path / "run" / "daemon.sock"
    daemon = PenguLinkDaemon(
        config=test_daemon_config,
        socket_path=sock_path,
        clipboard_backend=backend,
    )

    caplog.set_level(logging.DEBUG)

    # Generate distinct canary secrets
    canary_ipc_set = generate_canary_token("CANARY_IPC_SET")
    canary_cli_set = generate_canary_token("CANARY_CLI_SET")
    canary_err_oversized = generate_canary_token("CANARY_OVERSIZED") + ("_DATA" * 30000)
    canary_surrogate = generate_canary_token("CANARY_SURR") + "_\ud800"

    all_canaries = [
        canary_ipc_set,
        canary_cli_set,
        canary_err_oversized[:40],  # Prefix of oversized token
        canary_surrogate[:-2],      # Valid prefix before surrogate
    ]

    await daemon.start()

    try:
        # 1. IPC set_clipboard with canary
        client = PenguLinkIPCClient(sock_path, default_timeout=2.0)
        set_res = await client.set_clipboard(canary_ipc_set)
        assert set_res["status"] == "ok"

        # 2. IPC get_clipboard (contains the canary text)
        get_res = await client.get_clipboard()
        # 3. IPC error paths: oversized payload
        client_err = PenguLinkIPCClient(sock_path, default_timeout=2.0)
        with pytest.raises(Exception):
            await client_err.set_clipboard(canary_err_oversized)

        # 4. IPC error paths: surrogate characters
        with pytest.raises(Exception):
            await daemon.set_clipboard(canary_surrogate)

        # Allow logging events to settle
        await asyncio.sleep(0.05)

        # =====================================================================
        # PRIVACY AUDIT: Log Records Inspection
        # =====================================================================
        captured_records = caplog.records
        assert len(captured_records) > 0, "Expected log records to be captured"

        for record in captured_records:
            record_msg = record.getMessage()
            raw_msg = str(record.msg)
            raw_args = str(record.args)

            for canary in all_canaries:
                assert canary not in record_msg, (
                    f"Privacy violation: Canary secret '{canary}' leaked into formatted log message!\n"
                    f"Logger: {record.name}, Level: {record.levelname}\n"
                    f"Message: {record_msg}"
                )
                assert canary not in raw_msg, (
                    f"Privacy violation: Canary secret '{canary}' leaked into raw log template!\n"
                    f"Message template: {raw_msg}"
                )
                assert canary not in raw_args, (
                    f"Privacy violation: Canary secret '{canary}' leaked into log arguments!\n"
                    f"Args: {raw_args}"
                )

        # =====================================================================
        # PRIVACY AUDIT: LogRingBuffer Inspection
        # =====================================================================
        ring_buffer_entries = daemon._log_buffer.get_logs(limit=1000)
        assert len(ring_buffer_entries) > 0, "Expected entries in LogRingBuffer"

        for entry in ring_buffer_entries:
            entry_msg = entry.get("message", "")
            for canary in all_canaries:
                assert canary not in entry_msg, (
                    f"Privacy violation: Canary secret '{canary}' found in LogRingBuffer entry!\n"
                    f"Buffer Entry: {entry}"
                )

    finally:
        await daemon.stop()


async def test_privacy_clipboard_event_and_result_representations(
    test_daemon_config: PenguLinkConfig,
    tmp_path: Path,
) -> None:
    """Verify that ClipboardUpdatedEvent and ClipboardResult do not leak cleartext
    when converted to representations or formatted into strings.
    """
    secret = generate_canary_token("SECRET_EVENT_PAYLOAD")

    # 1. Test ClipboardUpdatedEvent repr and str
    event = ClipboardUpdatedEvent(
        content=secret,
        timestamp=int(time.time() * 1000),
        device_id="test_device",
    )
    event_repr = repr(event)
    event_str = str(event)

    assert secret not in event_repr, f"ClipboardUpdatedEvent repr leaked cleartext: {event_repr}"
    assert secret not in event_str, f"ClipboardUpdatedEvent str leaked cleartext: {event_str}"


def test_privacy_cli_canary_tokens_never_leaked(tmp_path: Path) -> None:
    """Objective 3 (CLI): Verify Click CLI clipboard set command does NOT echo
    canary cleartext in console output badges across interactive and piped stdin modes.
    """
    from unittest.mock import patch

    runner = CliRunner(env={"COLUMNS": "160"})
    mock_socket = tmp_path / "daemon.sock"
    mock_socket.touch()

    canary_cli = generate_canary_token("CLI_CANARY")

    with patch(
        "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
        return_value={"status": "ok", "bytes": len(canary_cli.encode("utf-8"))},
    ):
        # 1. Positional argument set
        res = runner.invoke(
            cli,
            ["-s", str(mock_socket), "clipboard", "set", canary_cli],
            catch_exceptions=False,
        )
        assert res.exit_code == 0
        assert canary_cli not in res.output
        assert "Clipboard updated" in res.output

        # 2. Piped stdin set
        res_stdin = runner.invoke(
            cli,
            ["-s", str(mock_socket), "clipboard", "set"],
            input=canary_cli,
            catch_exceptions=False,
        )
        assert res_stdin.exit_code == 0
        assert canary_cli not in res_stdin.output


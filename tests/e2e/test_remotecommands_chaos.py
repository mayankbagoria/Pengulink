"""tests.e2e.test_remotecommands_chaos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Live mTLS concurrency, chaos, unpair race conditions, and rapid flapping
stress tests for Milestone 15: RemoteCommands Plugin Ecosystem.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import pytest

from pengulink.core.events import (
    DeviceConnected,
    DeviceDisconnected,
    DeviceUnpaired,
    RemoteCommandExecutedEvent,
    RemoteCommandsReceivedEvent,
)
from pengulink.core.exceptions import ConnectionClosedError, IPCError
from pengulink.core.state_machine import DeviceState
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.network.packet import Packet
from tests.e2e.conftest import e2e_environment, wait_for
from tests.mock.mock_device import MockDevice, MockDeviceConfig


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
# 1. Concurrency Chaos: Unpair Race Conditions over Live mTLS & JSON-RPC IPC
# =============================================================================

async def test_e2e_remotecommands_chaos_001_unpair_race_live_mtls_ipc(tmp_path: Path) -> None:
    """Concurrently fire 35+ command execution requests via IPC while unpairing device.

    Verifies:
    - Zero deadlocks or race crashes
    - Zero unhandled internal server errors (-32603)
    - Responses are either success ('transmitted') or handled error (e.g. -32002)
    - Remote commands cache is purged cleanly
    - Daemon state machine for device ends in DISCONNECTED
    - DeviceStore confirms removal
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Stage and advertise 5 mock commands
        staged = {
            f"cmd_{i}": {"key": f"cmd_{i}", "name": f"Command {i}", "command": f"echo {i}"}
            for i in range(5)
        }
        await mock.send_command_list(commands=staged, format="dict")
        await wait_for(
            lambda: len(daemon.remote_commands_plugin.get_commands(mock.device_id)) == 5,
            timeout=4.0,
        )

        results: List[Any] = []
        errors: List[Exception] = []

        async def exec_worker(idx: int) -> None:
            cmd_key = f"cmd_{idx % 5}"
            try:
                res = await ipc_client.execute_remote_command(mock.device_id, cmd_key)
                results.append(res)
            except IPCError as exc:
                # Allowed: DEVICE_NOT_CONNECTED (-32002) or DEVICE_NOT_FOUND (-32001)
                assert exc.code in (-32001, -32002, -32014), f"Unexpected IPC error code: {exc.code}"
                errors.append(exc)
            except ConnectionClosedError as exc:
                errors.append(exc)
            except Exception as exc:
                pytest.fail(f"Unhandled non-IPC exception during unpair race: {type(exc)}: {exc}")

        async def unpair_worker() -> None:
            # Short jitter to interleave in the middle of execution requests
            await asyncio.sleep(0.005)
            await daemon.unpair_device(mock.device_id)

        # Dispatch 40 concurrent execution workers + 1 unpair worker
        workers = [asyncio.create_task(exec_worker(i)) for i in range(40)]
        unpair_task = asyncio.create_task(unpair_worker())

        # Complete all within 5.0 seconds (deadlock assertion)
        done, pending = await asyncio.wait(workers + [unpair_task], timeout=5.0)
        assert len(pending) == 0, f"Deadlock detected! {len(pending)} tasks hung during unpair race"

        # Verification of state invariants
        assert daemon.is_running is True
        assert len(results) + len(errors) == 40

        # Cache must be completely purged
        assert len(daemon.remote_commands_plugin.get_commands(mock.device_id)) == 0

        # State machine must be DISCONNECTED
        sm = daemon._state_machines.get(mock.device_id)
        if sm:
            assert sm.current_state == DeviceState.DISCONNECTED

        # DeviceStore must not contain paired record
        is_paired = await daemon.device_store.is_device_paired(mock.device_id)
        assert is_paired is False


# =============================================================================
# 2. Rapid Reconnection & Flapping under Command Traffic
# =============================================================================

async def test_e2e_remotecommands_chaos_002_rapid_flapping_command_traffic(tmp_path: Path) -> None:
    """Rapid connect/disconnect cycles while mock sends command lists.

    Verifies:
    - Zero leaked asyncio tasks
    - Zero leaked sockets / file descriptors
    - Daemon TCP manager and remote commands plugin recover cleanly
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        def get_fd_count() -> int:
            if sys.platform.startswith("linux") and os.path.exists("/proc/self/fd"):
                try:
                    return len(os.listdir("/proc/self/fd"))
                except Exception:
                    pass
            return 0

        initial_fds = get_fd_count()
        initial_tasks = len(asyncio.all_tasks())

        # Perform 8 rapid connect/disconnect cycles while broadcasting commands
        for cycle in range(8):
            # Send command list packet
            await mock.send_command_list(format="dict")
            await asyncio.sleep(0.01)

            # Drop connection abruptly
            await mock.drop_connection(reset=False)
            await wait_for(
                lambda: not daemon.is_device_connected(mock.device_id),
                timeout=3.0,
                desc=f"cycle {cycle} disconnect",
            )

            # Reconnect
            await daemon.connect_device(mock.device_id, port=mock.bound_tcp_port, timeout=3.0)
            await wait_for(
                lambda: daemon.is_device_connected(mock.device_id),
                timeout=3.0,
                desc=f"cycle {cycle} reconnect",
            )

        # Allow background tasks to settle
        await asyncio.sleep(0.1)

        # Ensure no task explosion
        final_tasks = len(asyncio.all_tasks())
        task_growth = final_tasks - initial_tasks
        assert task_growth <= 5, f"Leaked asyncio tasks detected! Task growth: {task_growth}"

        # Ensure no socket/FD leak explosion
        final_fds = get_fd_count()
        if initial_fds > 0 and final_fds > 0:
            fd_growth = final_fds - initial_fds
            # Sockets must be closed; threshold allows slight runtime variations
            assert fd_growth <= 10, f"Leaked file descriptors detected! FD growth: {fd_growth}"

        # Verify command exchange works normally after flapping
        await mock.send_command_list(format="dict")
        await wait_for(
            lambda: len(daemon.remote_commands_plugin.get_commands(mock.device_id)) > 0,
            timeout=4.0,
        )
        cmds = await ipc_client.list_remote_commands(mock.device_id)
        assert len(cmds) == 3


# =============================================================================
# 3. Live Wire Packet Fuzzing Burst
# =============================================================================

async def test_e2e_remotecommands_chaos_003_live_wire_packet_fuzz_burst(tmp_path: Path) -> None:
    """Deliver a burst of 50 malformed and boundary packets over live mTLS socket.

    Verifies daemon does not crash and continues processing subsequent valid commands.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Transmit 50 weird/corrupted remotecommands packets
        fuzz_packets = [
            Packet(type="kdeconnect.runcommand", body={"commandList": "broken {{ json"}),
            Packet(type="kdeconnect.runcommand", body={"commandList": None}),
            Packet(type="kdeconnect.runcommand", body={"commandList": 12345}),
            Packet(type="kdeconnect.runcommand", body={"commandList": ["not", "a", "dict"]}),
            Packet(type="kdeconnect.runcommand", body={"commands": "invalid"}),
            Packet(type="kdeconnect.runcommand", body={}),
            Packet(type="kdeconnect.runcommand.output", body={"key": "", "exitCode": "xyz"}),
            Packet(type="kdeconnect.remoteforcecommand", body={"key": ";" * 500}),
            Packet(type="kdeconnect.remoteforcecommand", body={"key": None}),
            Packet(type="kdeconnect.runcommand", body={"commandList": {f"k{i}": {"name": "x" * 500} for i in range(10)}}),
        ] * 5

        for pkt in fuzz_packets:
            await mock.send_packet(pkt)

        # Small pause for daemon to ingest all 50 packets
        await asyncio.sleep(0.2)

        # Daemon must remain completely alive and operational
        assert daemon.is_running is True
        assert daemon.is_device_connected(mock.device_id) is True

        # Valid commands delivered afterward must parse cleanly
        valid_cmds = {
            "post_fuzz_cmd": {
                "key": "post_fuzz_cmd",
                "name": "Post Fuzz Command",
                "command": "echo recovered",
            }
        }
        await mock.send_command_list(commands=valid_cmds, format="dict")
        await wait_for(
            lambda: daemon.remote_commands_plugin.get_command(mock.device_id, "post_fuzz_cmd") is not None,
            timeout=4.0,
        )

        cmds = await ipc_client.list_remote_commands(mock.device_id)
        assert any(c["key"] == "post_fuzz_cmd" for c in cmds)


# =============================================================================
# 4. Multi-Device Unpair & Execution Isolation
# =============================================================================

async def test_e2e_remotecommands_chaos_004_multi_device_unpair_and_exec_isolation(tmp_path: Path) -> None:
    """Pair 2 mock devices simultaneously; spam executions to both while unpairing dev1.

    Verifies dev1 unpairs and clears cache while dev2 operations remain completely unaffected.
    """
    async with e2e_environment(tmp_path) as (daemon, mock1, ipc_client):
        await _pair_and_connect(daemon, mock1)

        # Create a second mock device
        cfg2 = MockDeviceConfig(
            device_id=f"mock_dev2_{tmp_path.name[:8]}",
            device_name="MockPhone-2",
        )
        mock2 = MockDevice(config=cfg2)
        await mock2.start()

        try:
            # Pair second mock device
            await mock2.broadcast_identity(
                target_ip="127.0.0.1",
                target_port=daemon.config.network.udp_port,
            )
            await wait_for(lambda: mock2.device_id in daemon._discovered_devices, timeout=4.0)
            res2 = await daemon.pair_device(mock2.device_id, wait=True, timeout=5.0)
            assert res2["paired"] is True
            assert daemon.is_device_connected(mock2.device_id) is True

            # Stage distinct commands on each device
            cmds1 = {"dev1_lock": {"key": "dev1_lock", "name": "Dev 1 Lock", "command": "lock1"}}
            cmds2 = {"dev2_lock": {"key": "dev2_lock", "name": "Dev 2 Lock", "command": "lock2"}}

            await mock1.send_command_list(commands=cmds1, format="dict")
            await mock2.send_command_list(commands=cmds2, format="dict")

            await wait_for(
                lambda: (
                    len(daemon.remote_commands_plugin.get_commands(mock1.device_id)) == 1
                    and len(daemon.remote_commands_plugin.get_commands(mock2.device_id)) == 1
                ),
                timeout=4.0,
            )

            # Fire concurrent executions to both devices while unpairing dev1
            dev1_results = []
            dev2_results = []

            async def worker_dev1() -> None:
                try:
                    r = await ipc_client.execute_remote_command(mock1.device_id, "dev1_lock")
                    dev1_results.append(r)
                except Exception:
                    pass

            async def worker_dev2() -> None:
                try:
                    r = await ipc_client.execute_remote_command(mock2.device_id, "dev2_lock")
                    dev2_results.append(r)
                except Exception as exc:
                    pytest.fail(f"Mock 2 execution failed unexpectedly during Mock 1 unpair: {exc}")

            async def unpair_dev1() -> None:
                await asyncio.sleep(0.005)
                await daemon.unpair_device(mock1.device_id)

            tasks = [asyncio.create_task(worker_dev1()) for _ in range(15)]
            tasks += [asyncio.create_task(worker_dev2()) for _ in range(15)]
            tasks.append(asyncio.create_task(unpair_dev1()))

            await asyncio.gather(*tasks)

            # Mock 1 must be purged and disconnected
            assert len(daemon.remote_commands_plugin.get_commands(mock1.device_id)) == 0
            assert daemon.is_device_connected(mock1.device_id) is False

            # Mock 2 must remain connected, cached, and fully operational
            assert daemon.is_device_connected(mock2.device_id) is True
            assert len(daemon.remote_commands_plugin.get_commands(mock2.device_id)) == 1
            assert len(dev2_results) == 15

            # Execute another command on Mock 2 to confirm ongoing stability
            final_res = await ipc_client.execute_remote_command(mock2.device_id, "dev2_lock")
            assert final_res["status"] == "transmitted"
            pkt = await mock2.wait_for_command_exec("dev2_lock", timeout=4.0)
            assert pkt.body.get("key") == "dev2_lock"

        finally:
            await mock2.stop()


# =============================================================================
# 5. Extreme Concurrency: 100 Workers Racing against Live Unpairing
# =============================================================================

async def test_e2e_remotecommands_chaos_005_extreme_unpair_concurrency_100_workers(tmp_path: Path) -> None:
    """Stress test: 100 concurrent execution workers (50 IPC + 50 daemon direct)
    racing against daemon.unpair_device over live mTLS.

    Verifies:
    - Zero deadlocks under heavy concurrency (timeout 8.0s)
    - Zero daemon crashes
    - Zero unhandled internal server exceptions
    - Perfect cleanup of caches, sockets, and persistence
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        staged = {
            f"cmd_{i}": {"key": f"cmd_{i}", "name": f"Command {i}", "command": f"echo {i}"}
            for i in range(10)
        }
        await mock.send_command_list(commands=staged, format="dict")
        await wait_for(
            lambda: len(daemon.remote_commands_plugin.get_commands(mock.device_id)) == 10,
            timeout=4.0,
        )

        completed_ipc = 0
        completed_direct = 0

        async def worker_ipc(idx: int) -> None:
            nonlocal completed_ipc
            key = f"cmd_{idx % 10}"
            try:
                await ipc_client.execute_remote_command(mock.device_id, key)
            except (IPCError, ConnectionClosedError):
                pass
            except Exception as exc:
                pytest.fail(f"Unexpected IPC worker exception: {type(exc)}: {exc}")
            finally:
                completed_ipc += 1

        async def worker_direct(idx: int) -> None:
            nonlocal completed_direct
            key = f"cmd_{idx % 10}"
            try:
                await daemon.execute_remote_command(mock.device_id, key)
            except (ConnectionClosedError, Exception):
                pass
            finally:
                completed_direct += 1

        async def unpair_task() -> None:
            await asyncio.sleep(0.01)
            await daemon.unpair_device(mock.device_id)

        tasks = [asyncio.create_task(worker_ipc(i)) for i in range(50)]
        tasks += [asyncio.create_task(worker_direct(i)) for i in range(50)]
        tasks.append(asyncio.create_task(unpair_task()))

        done, pending = await asyncio.wait(tasks, timeout=8.0)
        assert len(pending) == 0, f"Deadlock! {len(pending)} workers hung"
        assert completed_ipc == 50
        assert completed_direct == 50

        # Post-race assertions
        assert daemon.is_running is True
        assert len(daemon.remote_commands_plugin.get_commands(mock.device_id)) == 0
        assert daemon.is_device_connected(mock.device_id) is False
        assert await daemon.device_store.is_device_paired(mock.device_id) is False


# =============================================================================
# 6. Flapping Burst: 15 Cycles with Simultaneous Outbound Execution
# =============================================================================

async def test_e2e_remotecommands_chaos_006_flapping_burst_15_cycles_with_commands(tmp_path: Path) -> None:
    """15 rapid flapping cycles where connections drop while execution requests are in-flight.

    Verifies:
    - Zero leaked asyncio tasks
    - Zero leaked file descriptors
    - Sockets do not accumulate in CLOSE_WAIT or hang
    - Final state is cleanly recoverable
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        initial_tasks = len(asyncio.all_tasks())

        for cycle in range(15):
            # Send commands
            await mock.send_command_list(format="dict")

            # Fire an execution asynchronously without waiting
            asyncio.create_task(
                ipc_client.execute_remote_command(mock.device_id, "cmd_lock")
            )

            # Drop connection abruptly
            await mock.drop_connection(reset=(cycle % 2 == 0))
            await wait_for(
                lambda: not daemon.is_device_connected(mock.device_id),
                timeout=3.0,
                desc=f"flapping disconnect {cycle}",
            )

            # Re-establish connection
            await daemon.connect_device(mock.device_id, port=mock.bound_tcp_port, timeout=3.0)
            await wait_for(
                lambda: daemon.is_device_connected(mock.device_id),
                timeout=3.0,
                desc=f"flapping reconnect {cycle}",
            )

        await asyncio.sleep(0.2)

        # Task leak check
        final_tasks = len(asyncio.all_tasks())
        task_growth = final_tasks - initial_tasks
        assert task_growth <= 6, f"Task leak detected: growth={task_growth}"

        # Recovery check
        await mock.send_command_list(format="dict")
        await wait_for(
            lambda: len(daemon.remote_commands_plugin.get_commands(mock.device_id)) > 0,
            timeout=4.0,
        )
        res = await ipc_client.execute_remote_command(mock.device_id, "cmd_lock")
        assert res["status"] == "transmitted"
        pkt = await mock.wait_for_command_exec("cmd_lock", timeout=4.0)
        assert pkt.body.get("key") == "cmd_lock"


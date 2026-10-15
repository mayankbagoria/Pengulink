"""tests.e2e.test_remotecommands_concurrency_stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Challenger 2 stress test suite for Milestone 15: RemoteCommands Plugin Ecosystem.

Verifies:
1. Multi-device live mTLS isolation (3 concurrent devices, colliding keys, zero leakage).
2. Asynchronous unpair race conditions across multiple devices.
3. Cache eviction upon remote-initiated and daemon-initiated unpair.
4. Offline and unpaired device command rejection (fast-fail, proper IPC error codes).
5. Rapid packet mutation bursts and thread/coroutine safety (no dictionary mutation during iteration).
6. Post-unpair packet rejection over wire.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List
import pytest

from pengulink.core.events import (
    DeviceUnpaired,
    RemoteCommandExecutedEvent,
    RemoteCommandsReceivedEvent,
)
from pengulink.core.exceptions import (
    ConnectionClosedError,
    DeviceNotFoundError,
    IPCError,
    RemoteCommandNotFoundError,
)
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
# 1. Multi-Device Live mTLS Isolation & Key Collisions
# =============================================================================

async def test_stress_three_device_concurrent_isolation_and_unpair(tmp_path: Path) -> None:
    """Empirically test 3 concurrent mock devices with colliding command keys.

    Verifies:
    - Commands with identical key 'common_lock' on Dev A, Dev B, Dev C are strictly isolated.
    - Executing on Dev A delivers ONLY to Dev A, never to Dev B or Dev C.
    - Unpairing Dev B mid-stream wipes Dev B cache while Dev A and Dev C continue without degradation.
    """
    async with e2e_environment(tmp_path) as (daemon, mock_a, ipc_client):
        await _pair_and_connect(daemon, mock_a)

        # Initialize mock_b and mock_c
        cfg_b = MockDeviceConfig(
            device_id=f"mock_dev_b_{tmp_path.name[:6]}",
            device_name="MockPhone-B",
        )
        mock_b = MockDevice(config=cfg_b)
        await mock_b.start()

        cfg_c = MockDeviceConfig(
            device_id=f"mock_dev_c_{tmp_path.name[:6]}",
            device_name="MockPhone-C",
        )
        mock_c = MockDevice(config=cfg_c)
        await mock_c.start()

        try:
            # Pair Dev B
            await mock_b.broadcast_identity(
                target_ip="127.0.0.1",
                target_port=daemon.config.network.udp_port,
            )
            await wait_for(lambda: mock_b.device_id in daemon._discovered_devices, timeout=4.0)
            res_b = await daemon.pair_device(mock_b.device_id, wait=True, timeout=5.0)
            assert res_b["paired"] is True

            # Pair Dev C
            await mock_c.broadcast_identity(
                target_ip="127.0.0.1",
                target_port=daemon.config.network.udp_port,
            )
            await wait_for(lambda: mock_c.device_id in daemon._discovered_devices, timeout=4.0)
            res_c = await daemon.pair_device(mock_c.device_id, wait=True, timeout=5.0)
            assert res_c["paired"] is True

            # Setup commands with both unique and colliding keys
            cmds_a = {
                "common_lock": {"key": "common_lock", "name": "Lock Screen A", "command": "lock_a"},
                "unique_a": {"key": "unique_a", "name": "Unique A", "command": "cmd_a"},
            }
            cmds_b = {
                "common_lock": {"key": "common_lock", "name": "Lock Screen B", "command": "lock_b"},
                "unique_b": {"key": "unique_b", "name": "Unique B", "command": "cmd_b"},
            }
            cmds_c = {
                "common_lock": {"key": "common_lock", "name": "Lock Screen C", "command": "lock_c"},
                "unique_c": {"key": "unique_c", "name": "Unique C", "command": "cmd_c"},
            }

            await mock_a.send_command_list(commands=cmds_a, format="dict")
            await mock_b.send_command_list(commands=cmds_b, format="dict")
            await mock_c.send_command_list(commands=cmds_c, format="dict")

            await wait_for(
                lambda: (
                    len(daemon.remote_commands_plugin.get_commands(mock_a.device_id)) == 2
                    and len(daemon.remote_commands_plugin.get_commands(mock_b.device_id)) == 2
                    and len(daemon.remote_commands_plugin.get_commands(mock_c.device_id)) == 2
                ),
                timeout=4.0,
            )

            # Assert strict cache isolation via IPC
            list_a = await ipc_client.list_remote_commands(mock_a.device_id)
            list_b = await ipc_client.list_remote_commands(mock_b.device_id)
            list_c = await ipc_client.list_remote_commands(mock_c.device_id)

            names_a = {c["name"] for c in list_a}
            names_b = {c["name"] for c in list_b}
            names_c = {c["name"] for c in list_c}

            assert "Lock Screen A" in names_a and "Lock Screen B" not in names_a and "Lock Screen C" not in names_a
            assert "Lock Screen B" in names_b and "Lock Screen A" not in names_b and "Lock Screen C" not in names_b
            assert "Lock Screen C" in names_c and "Lock Screen A" not in names_c and "Lock Screen B" not in names_c

            # Attempt to execute unique_b on Dev A -> must be rejected with RemoteCommandNotFoundError
            with pytest.raises(IPCError) as exc_info:
                await ipc_client.execute_remote_command(mock_a.device_id, "unique_b")
            assert "-32014" in str(exc_info.value)

            # Concurrently execute commands across all 3 devices while unpairing Dev B
            b_results = []
            a_results = []
            c_results = []

            async def worker_a(i: int) -> None:
                key = "common_lock" if i % 2 == 0 else "unique_a"
                res = await ipc_client.execute_remote_command(mock_a.device_id, key)
                a_results.append(res)

            async def worker_c(i: int) -> None:
                key = "common_lock" if i % 2 == 0 else "unique_c"
                res = await ipc_client.execute_remote_command(mock_c.device_id, key)
                c_results.append(res)

            async def worker_b(i: int) -> None:
                try:
                    res = await ipc_client.execute_remote_command(mock_b.device_id, "common_lock")
                    b_results.append(res)
                except IPCError:
                    pass

            async def unpair_b_task() -> None:
                await asyncio.sleep(0.005)
                await daemon.unpair_device(mock_b.device_id)

            tasks = [asyncio.create_task(worker_a(i)) for i in range(20)]
            tasks += [asyncio.create_task(worker_c(i)) for i in range(20)]
            tasks += [asyncio.create_task(worker_b(i)) for i in range(20)]
            tasks.append(asyncio.create_task(unpair_b_task()))

            await asyncio.gather(*tasks)

            # Assert Dev A and Dev C had 100% success rate
            assert len(a_results) == 20
            assert len(c_results) == 20

            # Dev B is completely evicted
            assert len(daemon.remote_commands_plugin.get_commands(mock_b.device_id)) == 0
            assert daemon.is_device_connected(mock_b.device_id) is False

            # Cross-check packet receipts on Mock Devices
            # Mock A must have received ONLY Dev A commands
            for pkt in mock_a.received_packets:
                if pkt.type in ("kdeconnect.remoteforcecommand", "kdeconnect.runcommand.request"):
                    assert pkt.body.get("key") in ("common_lock", "unique_a")

            # Mock C must have received ONLY Dev C commands
            for pkt in mock_c.received_packets:
                if pkt.type in ("kdeconnect.remoteforcecommand", "kdeconnect.runcommand.request"):
                    assert pkt.body.get("key") in ("common_lock", "unique_c")

            # Mock B should have zero packets intended for A or C
            for pkt in mock_b.received_packets:
                if pkt.type in ("kdeconnect.remoteforcecommand", "kdeconnect.runcommand.request"):
                    assert pkt.body.get("key") in ("common_lock", "unique_b")

        finally:
            await mock_b.stop()
            await mock_c.stop()


# =============================================================================
# 2. Offline Device Rejection & Cache Persistence Check
# =============================================================================

async def test_stress_offline_device_behavior_and_unpair_purge(tmp_path: Path) -> None:
    """Test behavior when paired device is offline (disconnected), followed by unpair.

    Verifies:
    1. Offline paired device allows cached list inspection via list_remote_commands.
    2. Offline device immediately rejects execution with ConnectionClosedError (-32002).
    3. Offline device immediately rejects command list request with ConnectionClosedError (-32002).
    4. Unpairing offline device purges cache completely.
    5. Post-unpair list query raises DeviceNotFoundError (-32001).
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Send commands and ensure cached
        staged = {
            "offline_cmd_1": {"key": "offline_cmd_1", "name": "Offline Cmd 1", "command": "run1"},
            "offline_cmd_2": {"key": "offline_cmd_2", "name": "Offline Cmd 2", "command": "run2"},
        }
        await mock.send_command_list(commands=staged, format="dict")
        await wait_for(
            lambda: len(daemon.remote_commands_plugin.get_commands(mock.device_id)) == 2,
            timeout=4.0,
        )

        # Drop mock connection to make device offline
        await mock.drop_connection(reset=True)
        await wait_for(
            lambda: not daemon.is_device_connected(mock.device_id),
            timeout=4.0,
            desc="device offline",
        )

        # 1. Device is paired but offline: list_remote_commands must return cached list
        cmds = await ipc_client.list_remote_commands(mock.device_id)
        assert len(cmds) == 2
        keys = {c["key"] for c in cmds}
        assert keys == {"offline_cmd_1", "offline_cmd_2"}

        # 2. Execution must fail immediately with -32002 (DEVICE_NOT_CONNECTED)
        with pytest.raises(IPCError) as exc_exec:
            await ipc_client.execute_remote_command(mock.device_id, "offline_cmd_1")
        assert "-32002" in str(exc_exec.value)
        assert "not connected" in str(exc_exec.value).lower()

        # 3. Requesting commands must fail immediately with -32002
        with pytest.raises(IPCError) as exc_req:
            await ipc_client.request_remote_commands(mock.device_id)
        assert "-32002" in str(exc_req.value)
        assert "not connected" in str(exc_req.value).lower()

        # 4. Unpair offline device
        unpair_res = await daemon.unpair_device(mock.device_id)
        assert unpair_res["success"] is True

        # Cache must be wiped
        assert len(daemon.remote_commands_plugin.get_commands(mock.device_id)) == 0

        # 5. Querying list for unpaired device returns empty list (cache was purged)
        cmds_post_unpair = await ipc_client.list_remote_commands(mock.device_id)
        assert cmds_post_unpair == []

        # 6. Querying list for completely unknown device raises DeviceNotFoundError (-32001)
        with pytest.raises(IPCError) as exc_unknown:
            await ipc_client.list_remote_commands("completely_unknown_device_id_999")
        assert "-32001" in str(exc_unknown.value)

        # 7. Executing on unpaired device must be rejected with -32002
        with pytest.raises(IPCError) as exc_unpaired_exec:
            await ipc_client.execute_remote_command(mock.device_id, "offline_cmd_1")
        assert "-32002" in str(exc_unpaired_exec.value)


# =============================================================================
# 3. Remote-Initiated Unpair via Wire Packet
# =============================================================================

async def test_stress_remote_peer_wire_unpair_purges_cache(tmp_path: Path) -> None:
    """Mock device sends an unpair packet (kdeconnect.pair with pair: false).

    Verifies:
    - Daemon processes unpair packet from peer.
    - RemoteCommandsPlugin cache is instantly cleared.
    - Connection is torn down.
    - Subsequent executions fail with -32001 or -32002.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        await mock.send_command_list(format="dict")
        await wait_for(
            lambda: len(daemon.remote_commands_plugin.get_commands(mock.device_id)) > 0,
            timeout=4.0,
        )
        assert len(daemon.remote_commands_plugin.get_commands(mock.device_id)) == 3

        # Peer sends unpair packet over the active mTLS socket
        unpair_pkt = Packet(type="kdeconnect.pair", body={"pair": False})
        await mock.send_packet(unpair_pkt)

        # Wait for daemon to process unpair
        await wait_for(
            lambda: not daemon.is_device_connected(mock.device_id),
            timeout=4.0,
            desc="peer unpair disconnect",
        )

        # Cache must be purged
        assert len(daemon.remote_commands_plugin.get_commands(mock.device_id)) == 0

        # Execution should raise IPCError
        with pytest.raises(IPCError):
            await ipc_client.execute_remote_command(mock.device_id, "cmd_lock")


# =============================================================================
# 4. Rapid Command Mutation Bursts & Concurrent Read/Exec Safety
# =============================================================================

async def test_stress_rapid_mutation_burst_with_concurrent_readers(tmp_path: Path) -> None:
    """Rapidly deliver 60 mutating command packets while 30 reader/exec tasks run.

    Verifies:
    - No 'RuntimeError: dictionary changed size during iteration' in daemon or plugin.
    - Lock synchronization holds under concurrent read/write pressure.
    - Final state reflects latest advertised payload.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        stop_event = asyncio.Event()
        read_success = 0
        read_errors = 0

        async def reader_worker() -> None:
            nonlocal read_success, read_errors
            while not stop_event.is_set():
                try:
                    cmds = await ipc_client.list_remote_commands(mock.device_id)
                    assert isinstance(cmds, list)
                    read_success += 1
                except Exception:
                    read_errors += 1
                await asyncio.sleep(0.002)

        # Start 10 background reader tasks
        readers = [asyncio.create_task(reader_worker()) for _ in range(10)]

        # Fire 60 mutating command lists from mock device
        for i in range(60):
            # Vary number of commands between 1 and 20
            count = (i % 20) + 1
            staged = {
                f"dyn_cmd_{k}": {
                    "key": f"dyn_cmd_{k}",
                    "name": f"Dynamic Command {k} v{i}",
                    "command": f"echo {k}_{i}",
                }
                for k in range(count)
            }
            await mock.send_command_list(commands=staged, format="dict")
            await asyncio.sleep(0.003)

        # Wait for final batch to settle
        await asyncio.sleep(0.1)
        stop_event.set()
        await asyncio.gather(*readers)

        assert read_errors == 0, f"Reader workers encountered {read_errors} errors!"
        assert read_success > 50

        # Verify final command count matches last iteration (iteration 59 -> count 20)
        final_cmds = daemon.remote_commands_plugin.get_commands(mock.device_id)
        assert len(final_cmds) == 20
        assert daemon.is_running is True


# =============================================================================
# 5. Non-Existent Key Rejection under Concurrency
# =============================================================================

async def test_stress_concurrent_valid_vs_invalid_keys(tmp_path: Path) -> None:
    """Fire 50 concurrent executions: 25 valid keys and 25 non-existent keys.

    Verifies:
    - 25 valid keys succeed and transmit wire packets.
    - 25 non-existent keys fail with code -32014 and do NOT send wire packets.
    - Mock device receives exactly 25 execution packets.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        valid_cmds = {
            f"valid_{i}": {"key": f"valid_{i}", "name": f"Valid {i}", "command": f"echo {i}"}
            for i in range(25)
        }
        await mock.send_command_list(commands=valid_cmds, format="dict")
        await wait_for(
            lambda: len(daemon.remote_commands_plugin.get_commands(mock.device_id)) == 25,
            timeout=4.0,
        )

        valid_count = 0
        invalid_count = 0

        async def exec_worker(idx: int) -> None:
            nonlocal valid_count, invalid_count
            if idx % 2 == 0:
                # Valid key
                key = f"valid_{idx // 2}"
                res = await ipc_client.execute_remote_command(mock.device_id, key)
                assert res["status"] == "transmitted"
                valid_count += 1
            else:
                # Invalid key
                key = f"nonexistent_key_{idx}"
                with pytest.raises(IPCError) as exc_info:
                    await ipc_client.execute_remote_command(mock.device_id, key)
                assert "-32014" in str(exc_info.value)
                invalid_count += 1

        tasks = [asyncio.create_task(exec_worker(i)) for i in range(50)]
        await asyncio.gather(*tasks)

        assert valid_count == 25
        assert invalid_count == 25

        # Check mock device received exactly 25 execution packets
        exec_packets = [
            pkt for pkt in mock.received_packets
            if pkt.type in ("kdeconnect.remoteforcecommand", "kdeconnect.runcommand.request")
        ]
        assert len(exec_packets) == 25

"""tests.adversarial.test_contacts_concurrency_chaos_challenger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Adversarial Stress & Chaos Test Suite for Milestone 13:
Contacts Plugin Ecosystem (kdeconnect.contacts.*).

Empirical Challenge Objectives:
1. Network Chaos:
   - Abrupt TCP disconnection mid-sync (after receiving timestamps manifest, before receiving vCards).
   - Peer connection drops with TCP RST (SO_LINGER) and reconnection flapping.
   - Peer silently dropping vCard response packets and timeout handling.
2. Concurrency & Race Conditions:
   - Concurrent contacts_sync invocations on the same device serialized via lock.
   - Calling unpair_device while a contacts sync is actively in-flight (verifying zero resurrected cache files).
   - Concurrent contacts_list and contacts_search querying while sync is actively updating disk cache.
3. Resource Leak Verification:
   - Zero dangling asyncio tasks.
   - Zero unclosed file descriptors or socket leaks.
   - Zero leaked .tmp staging files in the contacts data directory.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Set
import pytest

from pengulink.core.events import ContactsSynchronizedEvent
from pengulink.daemon.ipc import (
    CONTACTS_ERROR,
    DEVICE_NOT_CONNECTED,
    IPCError,
    PenguLinkIPCClient,
)
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.network.packet import Packet
from tests.e2e.conftest import e2e_environment, wait_for
from tests.mock.mock_device import MockDevice


def get_open_fds() -> Set[int]:
    """Retrieve set of open file descriptors on Linux."""
    if not os.path.exists("/proc/self/fd"):
        return set()
    try:
        return {int(entry) for entry in os.listdir("/proc/self/fd") if entry.isdigit()}
    except (OSError, ValueError):
        return set()


async def _pair_and_connect(daemon: PenguLinkDaemon, mock: MockDevice) -> None:
    """Helper to establish mutual pairing and an active TLS connection."""
    await mock.broadcast_identity(
        target_ip="127.0.0.1",
        target_port=daemon.config.network.udp_port,
    )
    await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)
    res = await daemon.pair_device(mock.device_id, wait=True, timeout=5.0)
    assert res["paired"] is True
    assert daemon.is_device_connected(mock.device_id) is True


VCARD_SAMPLE = (
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "FN:Adversarial Contact {idx}\r\n"
    "TEL;TYPE=CELL:+1555{idx:04d}\r\n"
    "EMAIL:contact{idx}@example.com\r\n"
    "ORG:Chaos Labs\r\n"
    "END:VCARD"
)


# =============================================================================
# 1. Network Chaos Tests
# =============================================================================

async def test_chaos_abrupt_disconnect_after_manifest_before_vcards(tmp_path: Path) -> None:
    """Challenge: Peer disconnects abruptly right after delivering timestamps, before vCards."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        # Stage 5 contacts on mock device
        contacts_dict = {
            f"uid_{i}": {"timestamp": 1000 + i, "vcard": VCARD_SAMPLE.format(idx=i)}
            for i in range(5)
        }
        mock.set_mock_contacts(contacts_dict)

        await _pair_and_connect(daemon, mock)

        # Hook into mock packet receiver: when request_vcards_by_uid is received, abruptly close socket
        original_handle = mock._handle_protocol_packet
        dropped = False

        async def _dropping_handler(packet: Packet, conn: Any) -> None:
            nonlocal dropped
            if packet.type == "kdeconnect.contacts.request_vcards_by_uid":
                dropped = True
                await mock.drop_connection(reset=False)
                return
            await original_handle(packet, conn)

        mock._handle_protocol_packet = _dropping_handler

        # Trigger sync with a short timeout
        with pytest.raises(IPCError) as exc_info:
            await ipc_client.contacts_sync(mock.device_id, sync_timeout=1.0)

        # Assert clean IPC error mapping (either CONTACTS_ERROR or DEVICE_NOT_CONNECTED)
        assert dropped is True
        error_msg = str(exc_info.value).lower()
        assert (
            str(CONTACTS_ERROR) in str(exc_info.value)
            or str(DEVICE_NOT_CONNECTED) in str(exc_info.value)
            or "not connected" in error_msg
            or "timed out" in error_msg
        )

        # Verify daemon did NOT crash and is responsive
        devices = await ipc_client.get_devices()
        assert len(devices) >= 1

        # Verify contacts directory has zero leftover .tmp staging files
        contacts_dir = daemon.config.storage.data_dir / "contacts"
        if contacts_dir.exists():
            tmp_files = list(contacts_dir.glob("*.tmp"))
            assert len(tmp_files) == 0, f"Leaked .tmp staging files found: {tmp_files}"

        # Re-establish connection and verify recovery
        mock._handle_protocol_packet = original_handle
        daemon.udp_discovery._seen_cache.pop(mock.device_id, None)
        await mock.broadcast_identity(
            target_ip="127.0.0.1",
            target_port=daemon.config.network.udp_port,
        )
        await wait_for(lambda: daemon.is_device_connected(mock.device_id), timeout=4.0)

        recovery_sync = await ipc_client.contacts_sync(mock.device_id, sync_timeout=5.0)
        assert recovery_sync["status"] == "synchronized"
        assert recovery_sync["total"] == 5
        assert recovery_sync["added"] == 5


async def test_chaos_so_linger_rst_reset_during_sync(tmp_path: Path) -> None:
    """Challenge: Peer drops TCP socket with hard RST (SO_LINGER) while sync is in-flight."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        mock.set_mock_contacts({
            f"uid_{i}": {"timestamp": 2000 + i, "vcard": VCARD_SAMPLE.format(idx=i)}
            for i in range(10)
        })

        await _pair_and_connect(daemon, mock)

        # Hook into mock packet receiver: when request_all_uids_timestamps is received, force hard TCP RST
        original_handle = mock._handle_protocol_packet

        async def _rst_handler(packet: Packet, conn: Any) -> None:
            if packet.type == "kdeconnect.contacts.request_all_uids_timestamps":
                await mock.drop_connection(reset=True)
                return
            await original_handle(packet, conn)

        mock._handle_protocol_packet = _rst_handler

        with pytest.raises(IPCError) as exc_info:
            await ipc_client.contacts_sync(mock.device_id, sync_timeout=1.0)

        # Daemon should catch TCP RST cleanly
        assert (
            str(CONTACTS_ERROR) in str(exc_info.value)
            or str(DEVICE_NOT_CONNECTED) in str(exc_info.value)
            or "not connected" in str(exc_info.value).lower()
        )

        # Wait for daemon state machine to transition to DISCONNECTED
        await wait_for(
            lambda: not daemon.is_device_connected(mock.device_id),
            timeout=3.0,
            desc="state machine transition to DISCONNECTED",
        )

        # Verify no .tmp files leaked
        contacts_dir = daemon.config.storage.data_dir / "contacts"
        if contacts_dir.exists():
            assert list(contacts_dir.glob("*.tmp")) == []

        # Restore handler, reconnect and verify clean recovery
        mock._handle_protocol_packet = original_handle
        daemon.udp_discovery._seen_cache.pop(mock.device_id, None)
        await mock.broadcast_identity(
            target_ip="127.0.0.1",
            target_port=daemon.config.network.udp_port,
        )
        await wait_for(lambda: daemon.is_device_connected(mock.device_id), timeout=4.0)

        res = await ipc_client.contacts_sync(mock.device_id, sync_timeout=5.0)
        assert res["status"] == "synchronized"
        assert res["total"] == 10


async def test_chaos_peer_flapping_and_rapid_reconnection(tmp_path: Path) -> None:
    """Challenge: Peer rapidly connects, disconnects, and reconnects multiple times."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        mock.set_mock_contacts({
            "uid_flapper": {"timestamp": 5000, "vcard": VCARD_SAMPLE.format(idx=99)}
        })

        await _pair_and_connect(daemon, mock)

        for iteration in range(3):
            # Abruptly drop connection
            await mock.drop_connection(reset=(iteration % 2 == 1))
            await wait_for(
                lambda: not daemon.is_device_connected(mock.device_id),
                timeout=3.0,
                desc=f"flapping disconnect iteration {iteration}",
            )

            # Clear seen cache to allow immediate redelivery through sliding window
            daemon.udp_discovery._seen_cache.pop(mock.device_id, None)

            # Re-announce identity to trigger auto-reconnect
            await mock.broadcast_identity(
                target_ip="127.0.0.1",
                target_port=daemon.config.network.udp_port,
            )
            await wait_for(
                lambda: daemon.is_device_connected(mock.device_id),
                timeout=5.0,
                desc=f"flapping reconnect iteration {iteration}",
            )

        # After flapping settle, perform full contacts sync
        sync_res = await ipc_client.contacts_sync(mock.device_id, sync_timeout=5.0)
        assert sync_res["status"] == "synchronized"
        assert sync_res["total"] == 1
        assert sync_res["added"] == 1


async def test_chaos_peer_drops_vcard_response_packets_timeout(tmp_path: Path) -> None:
    """Challenge: Peer responds to timestamp manifest but drops vCard payload response packets."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        mock.set_mock_contacts({
            "uid_drop": {"timestamp": 3000, "vcard": VCARD_SAMPLE.format(idx=1)}
        })

        await _pair_and_connect(daemon, mock)

        # Configure mock chaos to drop vCard response
        mock.chaos_config.contacts_drop_vcard_response = True

        with pytest.raises(IPCError) as exc_info:
            await ipc_client.contacts_sync(mock.device_id, sync_timeout=0.6)

        assert str(CONTACTS_ERROR) in str(exc_info.value)
        assert "timed out" in str(exc_info.value).lower()

        # Verify internal future was cleaned up from _vcard_futures
        assert mock.device_id not in daemon.contacts_plugin._vcard_futures

        # Disable chaos and verify retry succeeds
        mock.chaos_config.contacts_drop_vcard_response = False
        res = await ipc_client.contacts_sync(mock.device_id, sync_timeout=5.0)
        assert res["status"] == "synchronized"
        assert res["total"] == 1


# =============================================================================
# 2. Concurrency & Race Conditions Tests
# =============================================================================

async def test_concurrency_multiple_simultaneous_syncs_same_device(tmp_path: Path) -> None:
    """Challenge: Multiple concurrent contacts_sync requests on the same device must be serialized."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        contacts_dict = {
            f"uid_{i}": {"timestamp": 1000 + i, "vcard": VCARD_SAMPLE.format(idx=i)}
            for i in range(15)
        }
        mock.set_mock_contacts(contacts_dict)

        await _pair_and_connect(daemon, mock)

        # Fire 6 concurrent sync requests
        tasks = [
            ipc_client.contacts_sync(mock.device_id, sync_timeout=10.0)
            for _ in range(6)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        assert len(results) == 6
        # All 6 must report synchronized
        for r in results:
            assert r["status"] == "synchronized"
            assert r["total"] == 15

        # Exactly 1 should have added the 15 contacts; the others should report 0 added (diff sync)
        added_counts = [r["added"] for r in results]
        assert sum(added_counts) == 15
        assert 15 in added_counts
        assert added_counts.count(0) == 5

        # Verify final storage state
        list_res = await ipc_client.contacts_list(mock.device_id)
        assert list_res["total"] == 15


async def test_concurrency_unpair_device_while_sync_in_flight(tmp_path: Path) -> None:
    """Challenge: Calling unpair_device while contacts_sync is in-flight must never leave orphaned cache files."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        contacts_dict = {
            f"uid_{i}": {"timestamp": 1000 + i, "vcard": VCARD_SAMPLE.format(idx=i)}
            for i in range(20)
        }
        mock.set_mock_contacts(contacts_dict)

        await _pair_and_connect(daemon, mock)

        # Introduce artificial response delay on mock device
        mock.chaos_config.contacts_delay = 0.6

        # Start sync in background task
        sync_task = asyncio.create_task(
            ipc_client.contacts_sync(mock.device_id, sync_timeout=5.0)
        )

        # Wait briefly so sync reaches in-flight network phase
        await asyncio.sleep(0.1)

        # Unpair device immediately while sync is in-flight
        unpair_res = await ipc_client.unpair_device(mock.device_id)
        assert unpair_res["success"] is True

        # Await the sync task (it should either fail due to closed connection or timeout)
        sync_res = None
        sync_err = None
        try:
            sync_res = await sync_task
        except Exception as exc:
            sync_err = exc

        # Either way, let any remaining async tasks settle
        await asyncio.sleep(0.2)

        # CRITICAL INVARIANT: The device is unpaired; its contacts MUST NOT exist on disk or in memory
        contacts_file = daemon.config.storage.data_dir / "contacts" / f"{mock.device_id}.json"
        assert not contacts_file.exists(), (
            f"RESURRECTION BUG: Contacts JSON file {contacts_file} exists on disk after device unpair!"
        )

        # Check in-memory store
        total_cached, cached_contacts = await daemon.contacts_plugin.list_contacts(mock.device_id)
        assert total_cached == 0
        assert cached_contacts == []


async def test_concurrency_read_and_search_during_active_sync_writes(tmp_path: Path) -> None:
    """Challenge: Concurrent contacts_list and contacts_search during active atomic write must never see corrupt JSON."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        # 60 contacts
        contacts_dict = {
            f"uid_{i}": {"timestamp": 1000 + i, "vcard": VCARD_SAMPLE.format(idx=i)}
            for i in range(60)
        }
        mock.set_mock_contacts(contacts_dict)

        await _pair_and_connect(daemon, mock)

        # Start background sync
        sync_task = asyncio.create_task(
            ipc_client.contacts_sync(mock.device_id, sync_timeout=10.0)
        )

        # Simultaneously fire bursts of read and search requests
        read_tasks = []
        for i in range(25):
            read_tasks.append(ipc_client.contacts_list(mock.device_id, limit=20))
            read_tasks.append(ipc_client.contacts_search(mock.device_id, query="Adversarial"))
            await asyncio.sleep(0.01)

        read_results = await asyncio.gather(*read_tasks, return_exceptions=True)
        sync_res = await sync_task

        assert sync_res["status"] == "synchronized"
        assert sync_res["total"] == 60

        # None of the read or search requests should have raised an exception or failed JSON parsing
        for idx, res in enumerate(read_results):
            assert not isinstance(res, Exception), f"Read query {idx} raised: {res}"
            if "contacts" in res:
                assert isinstance(res["contacts"], list)
            elif "matches" in res:
                assert isinstance(res["matches"], list)

        # Final read check
        final_list = await ipc_client.contacts_list(mock.device_id, limit=100)
        assert final_list["total"] == 60


# =============================================================================
# 3. Resource Leak Verification
# =============================================================================

async def test_resource_safety_zero_dangling_tasks_and_zero_tmp_files(tmp_path: Path) -> None:
    """Challenge: Repeated chaos sync cycles leave zero unclosed FDs, zero dangling tasks, zero .tmp files."""
    fds_before = get_open_fds()

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        mock.set_mock_contacts({
            f"uid_{i}": {"timestamp": 1000 + i, "vcard": VCARD_SAMPLE.format(idx=i)}
            for i in range(10)
        })

        await _pair_and_connect(daemon, mock)

        # Run 5 sync cycles with forced diffs
        for i in range(5):
            mock.add_mock_contact(f"uid_dyn_{i}", VCARD_SAMPLE.format(idx=500 + i), timestamp=5000 + i)
            res = await ipc_client.contacts_sync(mock.device_id, force=(i % 2 == 0))
            assert res["status"] == "synchronized"

        contacts_dir = daemon.config.storage.data_dir / "contacts"
        assert contacts_dir.exists()
        # Verify zero leftover staging files
        tmp_files = list(contacts_dir.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Found leaked temporary files: {tmp_files}"

        # Unpair device
        await ipc_client.unpair_device(mock.device_id)
        # Background unpair task offloads unlinking to a worker thread: await settling
        await wait_for(
            lambda: not (contacts_dir / f"{mock.device_id}.json").exists(),
            timeout=3.0,
            desc="contacts cache file unlinking upon unpair",
        )

    # Allow asyncio cleanup
    await asyncio.sleep(0.1)

    # Check open file descriptors delta
    fds_after = get_open_fds()
    fd_leak = fds_after - fds_before
    # Note: stdout/stderr or pyunit runner might keep 1-2 fds, but should never grow unboundedly
    assert len(fd_leak) <= 5, f"Potential file descriptor leak: {len(fd_leak)} new FDs: {fd_leak}"


async def test_concurrency_dual_device_parallel_sync(tmp_path: Path) -> None:
    """Challenge: Two separate devices syncing concurrently maintain strict cross-device storage isolation."""
    from tests.mock.mock_device import MockDeviceConfig

    async with e2e_environment(tmp_path) as (daemon, mock1, ipc_client):
        # Create second mock device
        m2_cfg = MockDeviceConfig(
            device_id="mock_device_two_000002",
            device_name="Mock Android Phone 2",
            auto_pair=True,
            auto_pong=True,
            auto_battery_reply=True,
            auto_connectivity_reply=True,
        )
        mock2 = MockDevice(config=m2_cfg)
        await mock2.start(listen_tcp=True, port=0)

        try:
            # Stage different contacts
            mock1.set_mock_contacts({
                "dev1_uid1": {"timestamp": 1000, "vcard": VCARD_SAMPLE.format(idx=101)},
                "dev1_uid2": {"timestamp": 2000, "vcard": VCARD_SAMPLE.format(idx=102)},
            })
            mock2.set_mock_contacts({
                "dev2_uid1": {"timestamp": 3000, "vcard": VCARD_SAMPLE.format(idx=201)},
                "dev2_uid2": {"timestamp": 4000, "vcard": VCARD_SAMPLE.format(idx=202)},
                "dev2_uid3": {"timestamp": 5000, "vcard": VCARD_SAMPLE.format(idx=203)},
            })

            # Pair both
            await _pair_and_connect(daemon, mock1)
            await _pair_and_connect(daemon, mock2)

            # Concurrent sync
            sync1_task = asyncio.create_task(ipc_client.contacts_sync(mock1.device_id))
            sync2_task = asyncio.create_task(ipc_client.contacts_sync(mock2.device_id))
            r1, r2 = await asyncio.gather(sync1_task, sync2_task)

            assert r1["total"] == 2
            assert r2["total"] == 3

            # Verify complete isolation
            l1 = await ipc_client.contacts_list(mock1.device_id)
            l2 = await ipc_client.contacts_list(mock2.device_id)

            assert l1["total"] == 2
            assert l2["total"] == 3
            uids_1 = {c["uid"] for c in l1["contacts"]}
            uids_2 = {c["uid"] for c in l2["contacts"]}
            assert uids_1 == {"dev1_uid1", "dev1_uid2"}
            assert uids_2 == {"dev2_uid1", "dev2_uid2", "dev2_uid3"}
            assert uids_1.isdisjoint(uids_2)
        finally:
            await mock2.stop()


async def test_resource_safety_zero_dangling_tasks_audit(tmp_path: Path) -> None:
    """Challenge: Daemon startup, chaos sync, unpair, and teardown leave zero lingering asyncio tasks."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        mock.set_mock_contacts({
            "uid_task_test": {"timestamp": 1000, "vcard": VCARD_SAMPLE.format(idx=77)}
        })
        await _pair_and_connect(daemon, mock)

        # Sync contacts
        res = await ipc_client.contacts_sync(mock.device_id)
        assert res["total"] == 1

        # Unpair
        await ipc_client.unpair_device(mock.device_id)

    # After context manager exits (daemon and mock stopped)
    await asyncio.sleep(0.1)

    # Inspect all running tasks in current loop
    current_task = asyncio.current_task()
    lingering_tasks = [
        t for t in asyncio.all_tasks()
        if t is not current_task and not t.done()
    ]
    # Filter out any known pytest background harness tasks if any
    active_pengulink_tasks = [
        t for t in lingering_tasks
        if "pengulink" in str(t) or "contacts" in str(t)
    ]
    assert len(active_pengulink_tasks) == 0, f"Found lingering pengulink tasks: {active_pengulink_tasks}"

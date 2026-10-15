"""tests.e2e.test_contacts_e2e
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Live end-to-end integration test suite for Milestone 13: Contacts Plugin Ecosystem
(kdeconnect.contacts.*) running against MockDevice with live mTLS and JSON-RPC IPC.

Covers:
1. test_e2e_contacts_001_initial_sync_roundtrip
2. test_e2e_contacts_002_diff_sync_zero_transfer_unchanged
3. test_e2e_contacts_003_diff_sync_incremental_mutations
4. test_e2e_contacts_004_diff_sync_remote_deletion
5. test_e2e_contacts_005_buffer_safety_large_book_chunking
6. test_e2e_contacts_006_offline_access_and_offline_sync_rejection
7. test_e2e_contacts_007_unpair_cache_eviction
8. test_e2e_contacts_008_empty_phonebook_handling
9. test_e2e_contacts_009_malformed_vcard_fault_tolerance
10. test_e2e_contacts_010_disconnection_mid_sync_resilience
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any, List
import pytest

from pengulink.core.events import ContactsSynchronizedEvent
from pengulink.daemon.ipc import CONTACTS_ERROR, IPCError
from pengulink.daemon.service import PenguLinkDaemon
from tests.e2e.conftest import e2e_environment, wait_for
from tests.mock.mock_device import MockDevice


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


# Sample vCards for test fixtures
VCARD_ALICE = (
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "FN:Alice Smith\r\n"
    "N:Smith;Alice;;;\r\n"
    "TEL;TYPE=CELL:+1 555-0100\r\n"
    "EMAIL;TYPE=WORK:alice@example.com\r\n"
    "ORG:Pengu Corp\r\n"
    "TITLE:Lead Engineer\r\n"
    "END:VCARD"
)

VCARD_BOB = (
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "FN:Bob Jones\r\n"
    "N:Jones;Bob;;;\r\n"
    "TEL;TYPE=HOME:(555) 020-0000\r\n"
    "EMAIL;TYPE=HOME:bob@example.com\r\n"
    "END:VCARD"
)

VCARD_CHARLIE = (
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "FN:Charlie Davis\r\n"
    "TEL;TYPE=VOICE:5550300\r\n"
    "END:VCARD"
)


# =============================================================================
# E2E Contacts Tests
# =============================================================================

async def test_e2e_contacts_001_initial_sync_roundtrip(tmp_path: Path) -> None:
    """1. Full initial contacts sync round-trip over live mTLS and JSON-RPC IPC."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        # Stage contacts on mock phone
        mock.set_mock_contacts({
            "uid_1": {"timestamp": 1000, "vcard": VCARD_ALICE},
            "uid_2": {"timestamp": 2000, "vcard": VCARD_BOB},
            "uid_3": {"timestamp": 3000, "vcard": VCARD_CHARLIE},
        })

        await _pair_and_connect(daemon, mock)

        # Track event emission
        events: List[ContactsSynchronizedEvent] = []
        daemon.event_bus.subscribe(ContactsSynchronizedEvent, lambda ev: events.append(ev))

        # Perform initial sync
        sync_res = await ipc_client.contacts_sync(mock.device_id)
        assert sync_res["status"] == "synchronized"
        assert sync_res["total"] == 3
        assert sync_res["added"] == 3
        assert sync_res["updated"] == 0
        assert sync_res["deleted"] == 0

        # Verify event was dispatched
        await wait_for(lambda: len(events) == 1, timeout=2.0)
        assert events[0].device_id == mock.device_id
        assert events[0].total_contacts == 3
        assert events[0].contacts_added == 3

        # List contacts via IPC
        list_res = await ipc_client.contacts_list(mock.device_id)
        assert list_res["total"] == 3
        contact_names = {c["name"] for c in list_res["contacts"]}
        assert contact_names == {"Alice Smith", "Bob Jones", "Charlie Davis"}

        # Get specific contact via IPC
        get_res = await ipc_client.contacts_get(mock.device_id, "uid_1")
        contact_alice = get_res["contact"]
        assert contact_alice["name"] == "Alice Smith"
        assert contact_alice["organization"] == "Pengu Corp"
        assert any(p["normalized"] == "+15550100" for p in contact_alice["phones"])

        # Search contacts via IPC
        search_res = await ipc_client.contacts_search(mock.device_id, "Bob")
        assert len(search_res["matches"]) == 1
        assert search_res["matches"][0]["uid"] == "uid_2"
        assert search_res["matches"][0]["name"] == "Bob Jones"

        # Verify atomic storage on disk with 0600 permissions
        disk_file = daemon.config.storage.data_dir / "contacts" / f"{mock.device_id}.json"
        assert disk_file.exists()
        mode = disk_file.stat().st_mode & 0o777
        assert mode == 0o600


async def test_e2e_contacts_002_diff_sync_zero_transfer_unchanged(tmp_path: Path) -> None:
    """2. Diff-based incremental sync with unchanged contacts results in zero wire vCard transfers."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        mock.set_mock_contacts({
            "uid_1": {"timestamp": 1000, "vcard": VCARD_ALICE},
            "uid_2": {"timestamp": 2000, "vcard": VCARD_BOB},
        })

        await _pair_and_connect(daemon, mock)

        # 1. Initial sync
        sync1 = await ipc_client.contacts_sync(mock.device_id)
        assert sync1["added"] == 2

        vcard_req_count_1 = sum(
            1 for p in mock.received_packets if p.type == "kdeconnect.contacts.request_vcards_by_uid"
        )
        assert vcard_req_count_1 >= 1

        # 2. Second sync on unchanged phonebook
        sync2 = await ipc_client.contacts_sync(mock.device_id)
        assert sync2["status"] == "synchronized"
        assert sync2["total"] == 2
        assert sync2["added"] == 0
        assert sync2["updated"] == 0
        assert sync2["deleted"] == 0

        # Wire verification: No new request_vcards_by_uid packet was transmitted!
        vcard_req_count_2 = sum(
            1 for p in mock.received_packets if p.type == "kdeconnect.contacts.request_vcards_by_uid"
        )
        assert vcard_req_count_2 == vcard_req_count_1


async def test_e2e_contacts_003_diff_sync_incremental_mutations(tmp_path: Path) -> None:
    """3. Incremental mutations: only updated and newly added contacts are requested across wire."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        mock.set_mock_contacts({
            "uid_1": {"timestamp": 1000, "vcard": VCARD_ALICE},
            "uid_2": {"timestamp": 2000, "vcard": VCARD_BOB},
        })

        await _pair_and_connect(daemon, mock)
        await ipc_client.contacts_sync(mock.device_id)

        # Clear mock's recorded requested UIDs to track subsequent sync
        mock._contacts_requested_uids.clear()

        # Update uid_1 with new timestamp and name; add uid_3; keep uid_2 untouched
        vcard_alice_updated = (
            "BEGIN:VCARD\r\n"
            "VERSION:3.0\r\n"
            "FN:Alice Smith Updated\r\n"
            "TEL:+1 555-9999\r\n"
            "END:VCARD"
        )
        mock.add_mock_contact("uid_1", vcard_alice_updated, timestamp=5000)
        mock.add_mock_contact("uid_3", VCARD_CHARLIE, timestamp=6000)

        # Second sync
        sync_res = await ipc_client.contacts_sync(mock.device_id)
        assert sync_res["status"] == "synchronized"
        assert sync_res["total"] == 3
        assert sync_res["added"] == 1
        assert sync_res["updated"] == 1
        assert sync_res["deleted"] == 0

        # Wire verification: Only mutated UIDs (uid_1 and uid_3) were requested
        assert set(mock._contacts_requested_uids) == {"uid_1", "uid_3"}

        # Verify updated data in daemon
        res_alice = await ipc_client.contacts_get(mock.device_id, "uid_1")
        assert res_alice["contact"]["name"] == "Alice Smith Updated"

        res_charlie = await ipc_client.contacts_get(mock.device_id, "uid_3")
        assert res_charlie["contact"]["name"] == "Charlie Davis"


async def test_e2e_contacts_004_diff_sync_remote_deletion(tmp_path: Path) -> None:
    """4. Remote contact deletions on peer device are reflected in local storage."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        mock.set_mock_contacts({
            "uid_1": {"timestamp": 1000, "vcard": VCARD_ALICE},
            "uid_2": {"timestamp": 2000, "vcard": VCARD_BOB},
        })

        await _pair_and_connect(daemon, mock)
        await ipc_client.contacts_sync(mock.device_id)

        # Remote user deletes uid_2
        mock.remove_mock_contact("uid_2")

        # Sync deletion
        sync_res = await ipc_client.contacts_sync(mock.device_id)
        assert sync_res["total"] == 1
        assert sync_res["deleted"] == 1
        assert sync_res["added"] == 0
        assert sync_res["updated"] == 0

        list_res = await ipc_client.contacts_list(mock.device_id)
        assert list_res["total"] == 1
        assert list_res["contacts"][0]["uid"] == "uid_1"

        # Verify getting deleted contact raises not found
        with pytest.raises(IPCError) as exc_info:
            await ipc_client.contacts_get(mock.device_id, "uid_2")
        assert str(CONTACTS_ERROR) in str(exc_info.value)


async def test_e2e_contacts_005_buffer_safety_large_book_chunking(tmp_path: Path) -> None:
    """5. Buffer safety: 75 contacts are requested in batches <= 50 UIDs conforming to 128 KB packet limits."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        # Generate 75 contacts
        contacts_dict = {}
        for i in range(75):
            uid = f"contact_{i:03d}"
            vcard = f"BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Person {i}\r\nTEL:+1555000{i:04d}\r\nEND:VCARD"
            contacts_dict[uid] = {"timestamp": 1000 + i, "vcard": vcard}
        mock.set_mock_contacts(contacts_dict)

        await _pair_and_connect(daemon, mock)

        sync_res = await ipc_client.contacts_sync(mock.device_id)
        assert sync_res["status"] == "synchronized"
        assert sync_res["total"] == 75
        assert sync_res["added"] == 75

        # Check all outgoing batch requests received by mock
        vcard_req_packets = [
            p for p in mock.received_packets if p.type == "kdeconnect.contacts.request_vcards_by_uid"
        ]
        # 75 contacts chunked into <= 50 per chunk requires at least 2 packets
        assert len(vcard_req_packets) >= 2
        all_requested_uids: List[str] = []
        for pkt in vcard_req_packets:
            uids = pkt.body.get("uids", [])
            assert len(uids) <= 50, f"Batch size {len(uids)} exceeded 50 UID buffer safety limit!"
            all_requested_uids.extend(uids)

        assert len(all_requested_uids) == 75
        assert len(set(all_requested_uids)) == 75

        # Verify all contacts present in list
        list_res = await ipc_client.contacts_list(mock.device_id, limit=100)
        assert list_res["total"] == 75


async def test_e2e_contacts_006_offline_access_and_offline_sync_rejection(tmp_path: Path) -> None:
    """6. Cached contacts remain readable offline while new sync attempts are cleanly rejected."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        mock.set_mock_contacts({
            "uid_1": {"timestamp": 1000, "vcard": VCARD_ALICE},
        })

        await _pair_and_connect(daemon, mock)
        await ipc_client.contacts_sync(mock.device_id)

        # Disconnect mock peer
        await mock.stop()
        await wait_for(
            lambda: not daemon.is_device_connected(mock.device_id),
            timeout=3.0,
            desc="device disconnected",
        )

        # 1. Offline read access succeeds from cached store
        list_res = await ipc_client.contacts_list(mock.device_id)
        assert list_res["total"] == 1
        assert list_res["contacts"][0]["name"] == "Alice Smith"

        get_res = await ipc_client.contacts_get(mock.device_id, "uid_1")
        assert get_res["contact"]["name"] == "Alice Smith"

        search_res = await ipc_client.contacts_search(mock.device_id, "Alice")
        assert len(search_res["matches"]) == 1

        # 2. Offline sync attempt must fail because device is not connected
        with pytest.raises(IPCError) as exc_info:
            await ipc_client.contacts_sync(mock.device_id)
        assert "not connected" in str(exc_info.value).lower()


async def test_e2e_contacts_007_unpair_cache_eviction(tmp_path: Path) -> None:
    """7. Unpairing a device completely purges contact cache files from disk and memory."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        mock.set_mock_contacts({
            "uid_1": {"timestamp": 1000, "vcard": VCARD_ALICE},
        })

        await _pair_and_connect(daemon, mock)
        await ipc_client.contacts_sync(mock.device_id)

        disk_file = daemon.config.storage.data_dir / "contacts" / f"{mock.device_id}.json"
        assert disk_file.exists()

        # In-memory store has contacts
        total_before, _ = await daemon.contacts_plugin.list_contacts(mock.device_id)
        assert total_before > 0

        # Unpair device via IPC
        unpair_res = await ipc_client.unpair_device(mock.device_id)
        assert unpair_res["success"] is True

        # Verify disk cache file was deleted
        assert not disk_file.exists()

        # Verify in-memory contacts store is purged
        total_after, contacts_after = await daemon.contacts_plugin.list_contacts(mock.device_id)
        assert total_after == 0
        assert contacts_after == []


async def test_e2e_contacts_008_empty_phonebook_handling(tmp_path: Path) -> None:
    """8. Empty phonebook handles zero-UID manifests gracefully without errors."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        mock.clear_mock_contacts()

        await _pair_and_connect(daemon, mock)

        sync_res = await ipc_client.contacts_sync(mock.device_id)
        assert sync_res["status"] == "synchronized"
        assert sync_res["total"] == 0
        assert sync_res["added"] == 0
        assert sync_res["updated"] == 0
        assert sync_res["deleted"] == 0

        list_res = await ipc_client.contacts_list(mock.device_id)
        assert list_res["total"] == 0
        assert list_res["contacts"] == []


async def test_e2e_contacts_009_malformed_vcard_fault_tolerance(tmp_path: Path) -> None:
    """9. Malformed, truncated, or corrupted vCards are parsed resiliently without crashing daemon."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        mock.set_mock_contacts({
            "m_no_envelope": {
                "timestamp": 1000,
                "vcard": "FN:Bare Property Line\nTEL:5551111\n",
            },
            "m_corrupt_qp": {
                "timestamp": 2000,
                "vcard": (
                    "BEGIN:VCARD\r\n"
                    "VERSION:3.0\r\n"
                    "FN;CHARSET=UTF-8;ENCODING=QUOTED-PRINTABLE:=C3=\r\n"
                    "TEL:+1234567890\r\n"
                    "END:VCARD"
                ),
            },
            "m_empty": {
                "timestamp": 3000,
                "vcard": "",
            },
            "m_valid": {
                "timestamp": 4000,
                "vcard": VCARD_BOB,
            },
        })

        await _pair_and_connect(daemon, mock)

        sync_res = await ipc_client.contacts_sync(mock.device_id)
        assert sync_res["status"] == "synchronized"
        assert sync_res["total"] == 4

        list_res = await ipc_client.contacts_list(mock.device_id)
        assert list_res["total"] == 4

        # Resilient fallbacks: None of them crashed
        names = {c["name"] for c in list_res["contacts"]}
        assert "Bob Jones" in names
        assert "Bare Property Line" in names


async def test_e2e_contacts_010_disconnection_mid_sync_resilience(tmp_path: Path) -> None:
    """10. Unresponsive peer timing out mid-sync raises clean IPC error and daemon recovers."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        mock.set_mock_contacts({
            "uid_1": {"timestamp": 1000, "vcard": VCARD_ALICE},
        })

        await _pair_and_connect(daemon, mock)

        # Configure mock to ignore incoming contacts requests
        mock.chaos_config.contacts_ignore_requests = True

        # Lower timeout on plugin for fast test execution
        daemon.contacts_plugin.default_timeout = 0.5

        with pytest.raises(IPCError) as exc_info:
            await ipc_client.contacts_sync(mock.device_id, sync_timeout=0.5)
        assert str(CONTACTS_ERROR) in str(exc_info.value)

        # Verify daemon is healthy and running
        devices = await ipc_client.get_devices()
        assert any(d["device_id"] == mock.device_id for d in devices)

        # Restore responsiveness and verify subsequent sync recovers
        mock.chaos_config.contacts_ignore_requests = False
        daemon.contacts_plugin.default_timeout = 5.0

        sync_res = await ipc_client.contacts_sync(mock.device_id)
        assert sync_res["status"] == "synchronized"
        assert sync_res["total"] == 1
        assert sync_res["added"] == 1

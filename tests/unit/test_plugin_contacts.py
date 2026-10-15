"""Unit tests for ContactsPlugin, ContactStore, Diff Sync, and Storage."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from pengulink.core.events import ContactsSynchronizedEvent, DeviceUnpaired, EventBus
from pengulink.core.exceptions import ConnectionClosedError, SyncTimeoutError
from pengulink.network.packet import Packet
from pengulink.plugins.contacts.plugin import (
    DEFAULT_BATCH_SIZE,
    LEGACY_CONTACTS,
    MAX_BATCH_SIZE,
    REQUEST_ALL_UIDS_TIMESTAMPS,
    REQUEST_VCARDS_BY_UID,
    RESPONSE_UIDS_TIMESTAMPS,
    RESPONSE_VCARDS,
    ContactsPlugin,
    ContactsSyncResult,
)
from pengulink.plugins.contacts.store import ContactStore, SyncDiff, chunk_list
from pengulink.plugins.contacts.vcard import Contact, PhoneNumber


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def contacts_plugin(event_bus: EventBus, tmp_path: Path) -> ContactsPlugin:
    return ContactsPlugin(event_bus=event_bus, data_dir=tmp_path)


class TestPluginBasics:
    def test_plugin_id_and_capabilities(self, contacts_plugin: ContactsPlugin) -> None:
        assert contacts_plugin.plugin_id == "contacts"
        assert RESPONSE_UIDS_TIMESTAMPS in contacts_plugin.incoming_capabilities
        assert RESPONSE_VCARDS in contacts_plugin.incoming_capabilities
        assert REQUEST_ALL_UIDS_TIMESTAMPS in contacts_plugin.outgoing_capabilities
        assert REQUEST_VCARDS_BY_UID in contacts_plugin.outgoing_capabilities
        assert LEGACY_CONTACTS in contacts_plugin.incoming_capabilities
        assert LEGACY_CONTACTS in contacts_plugin.outgoing_capabilities


class TestPacketRouting:
    async def test_handle_response_uids_timestamps(self, contacts_plugin: ContactsPlugin) -> None:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Dict[str, int]] = loop.create_future()
        contacts_plugin._timestamp_futures["dev1"] = fut

        pkt = Packet(
            type=RESPONSE_UIDS_TIMESTAMPS,
            body={"uids": ["101", "102"], "101": 1712000000000, "102": "1712000005000"},
        )
        await contacts_plugin.handle_packet(pkt, "dev1", AsyncMock())

        res = await fut
        assert res == {"101": 1712000000000, "102": 1712000005000}

    async def test_handle_response_vcards(self, contacts_plugin: ContactsPlugin) -> None:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Dict[str, str]] = loop.create_future()
        contacts_plugin._vcard_futures["dev1"] = fut

        pkt = Packet(
            type=RESPONSE_VCARDS,
            body={
                "uids": ["101"],
                "101": "BEGIN:VCARD\r\nFN:Alice\r\nEND:VCARD\r\n",
            },
        )
        await contacts_plugin.handle_packet(pkt, "dev1", AsyncMock())

        res = await fut
        assert "101" in res
        assert "Alice" in res["101"]

    async def test_handle_unrecognized_packet_safe(self, contacts_plugin: ContactsPlugin) -> None:
        pkt = Packet(type="kdeconnect.contacts.unknown_subpacket", body={})
        # Should not raise exception
        await contacts_plugin.handle_packet(pkt, "dev1", AsyncMock())

    async def test_server_request_all_uids_timestamps(self, contacts_plugin: ContactsPlugin) -> None:
        manifest = {"101": 1712000000000}
        contacts = {"101": Contact(uid="101", name="Alice", raw_vcard="VCARD_ALICE")}
        await contacts_plugin.store.save_device("dev1", manifest, contacts)

        sent_packets: List[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        req = Packet(type=REQUEST_ALL_UIDS_TIMESTAMPS, body={})
        await contacts_plugin.handle_packet(req, "dev1", mock_send)

        assert len(sent_packets) == 1
        resp = sent_packets[0]
        assert resp.type == RESPONSE_UIDS_TIMESTAMPS
        assert resp.body["uids"] == ["101"]
        assert resp.body["101"] == 1712000000000

    async def test_server_request_vcards_by_uid(self, contacts_plugin: ContactsPlugin) -> None:
        manifest = {"101": 1712000000000}
        contacts = {"101": Contact(uid="101", name="Alice", raw_vcard="VCARD_ALICE")}
        await contacts_plugin.store.save_device("dev1", manifest, contacts)

        sent_packets: List[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        req = Packet(type=REQUEST_VCARDS_BY_UID, body={"uids": ["101"]})
        await contacts_plugin.handle_packet(req, "dev1", mock_send)

        assert len(sent_packets) == 1
        resp = sent_packets[0]
        assert resp.type == RESPONSE_VCARDS
        assert resp.body["uids"] == ["101"]
        assert resp.body["101"] == "VCARD_ALICE"


class TestDiffSync:
    async def test_initial_sync_fetches_all(self, contacts_plugin: ContactsPlugin) -> None:
        sent_packets: List[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)
            if pkt.type == REQUEST_ALL_UIDS_TIMESTAMPS:
                asyncio.create_task(
                    contacts_plugin.handle_packet(
                        Packet(
                            type=RESPONSE_UIDS_TIMESTAMPS,
                            body={"uids": ["1", "2"], "1": 100, "2": 200},
                        ),
                        "dev1",
                        mock_send,
                    )
                )
            elif pkt.type == REQUEST_VCARDS_BY_UID:
                req_uids = pkt.body.get("uids", [])
                body = {
                    "uids": req_uids,
                    "1": "BEGIN:VCARD\r\nFN:Contact 1\r\nEND:VCARD\r\n",
                    "2": "BEGIN:VCARD\r\nFN:Contact 2\r\nEND:VCARD\r\n",
                }
                asyncio.create_task(
                    contacts_plugin.handle_packet(
                        Packet(type=RESPONSE_VCARDS, body=body),
                        "dev1",
                        mock_send,
                    )
                )

        res = await contacts_plugin.sync("dev1", mock_send)
        assert res.total_contacts == 2
        assert res.contacts_added == 2
        assert res.contacts_updated == 0
        assert res.contacts_deleted == 0

        # Verify disk
        total, contacts = await contacts_plugin.store.list_contacts("dev1")
        assert total == 2

    async def test_unchanged_sync_zero_vcard_requests(self, contacts_plugin: ContactsPlugin) -> None:
        # Pre-seed cache
        manifest = {"1": 100, "2": 200}
        contacts = {
            "1": Contact(uid="1", name="Contact 1", timestamp=100),
            "2": Contact(uid="2", name="Contact 2", timestamp=200),
        }
        await contacts_plugin.store.save_device("dev1", manifest, contacts)

        sent_packets: List[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)
            if pkt.type == REQUEST_ALL_UIDS_TIMESTAMPS:
                asyncio.create_task(
                    contacts_plugin.handle_packet(
                        Packet(
                            type=RESPONSE_UIDS_TIMESTAMPS,
                            body={"uids": ["1", "2"], "1": 100, "2": 200},
                        ),
                        "dev1",
                        mock_send,
                    )
                )

        res = await contacts_plugin.sync("dev1", mock_send)
        assert res.total_contacts == 2
        assert res.contacts_added == 0
        assert res.contacts_updated == 0
        assert res.contacts_deleted == 0

        # Exactly 1 request packet sent (REQUEST_ALL_UIDS_TIMESTAMPS), zero vCard requests
        assert len(sent_packets) == 1
        assert sent_packets[0].type == REQUEST_ALL_UIDS_TIMESTAMPS

    async def test_partial_update_diff(self, contacts_plugin: ContactsPlugin) -> None:
        # Pre-seed
        manifest = {"1": 100, "2": 200}
        contacts = {
            "1": Contact(uid="1", name="Contact 1", timestamp=100),
            "2": Contact(uid="2", name="Old Contact 2", timestamp=200),
        }
        await contacts_plugin.store.save_device("dev1", manifest, contacts)

        sent_packets: List[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)
            if pkt.type == REQUEST_ALL_UIDS_TIMESTAMPS:
                # "2" is updated (250 > 200), "3" is brand new
                asyncio.create_task(
                    contacts_plugin.handle_packet(
                        Packet(
                            type=RESPONSE_UIDS_TIMESTAMPS,
                            body={"uids": ["1", "2", "3"], "1": 100, "2": 250, "3": 300},
                        ),
                        "dev1",
                        mock_send,
                    )
                )
            elif pkt.type == REQUEST_VCARDS_BY_UID:
                req_uids = pkt.body.get("uids", [])
                body = {
                    "uids": req_uids,
                    "2": "BEGIN:VCARD\r\nFN:Updated Contact 2\r\nEND:VCARD\r\n",
                    "3": "BEGIN:VCARD\r\nFN:New Contact 3\r\nEND:VCARD\r\n",
                }
                asyncio.create_task(
                    contacts_plugin.handle_packet(
                        Packet(type=RESPONSE_VCARDS, body=body),
                        "dev1",
                        mock_send,
                    )
                )

        res = await contacts_plugin.sync("dev1", mock_send)
        assert res.total_contacts == 3
        assert res.contacts_added == 1
        assert res.contacts_updated == 1
        assert res.contacts_deleted == 0

        # Verify contact 1 was not re-fetched
        vcard_req = [p for p in sent_packets if p.type == REQUEST_VCARDS_BY_UID][0]
        assert set(vcard_req.body["uids"]) == {"2", "3"}

    async def test_deletion_diff(self, contacts_plugin: ContactsPlugin) -> None:
        # Pre-seed with contacts 1, 2, 3
        manifest = {"1": 100, "2": 200, "3": 300}
        contacts = {
            "1": Contact(uid="1", name="C1", timestamp=100),
            "2": Contact(uid="2", name="C2", timestamp=200),
            "3": Contact(uid="3", name="C3", timestamp=300),
        }
        await contacts_plugin.store.save_device("dev1", manifest, contacts)

        async def mock_send(pkt: Packet) -> None:
            if pkt.type == REQUEST_ALL_UIDS_TIMESTAMPS:
                # 2 was deleted on remote
                asyncio.create_task(
                    contacts_plugin.handle_packet(
                        Packet(
                            type=RESPONSE_UIDS_TIMESTAMPS,
                            body={"uids": ["1", "3"], "1": 100, "3": 300},
                        ),
                        "dev1",
                        mock_send,
                    )
                )

        res = await contacts_plugin.sync("dev1", mock_send)
        assert res.total_contacts == 2
        assert res.contacts_deleted == 1

        _, contacts_now = await contacts_plugin.store.load_device("dev1")
        assert "2" not in contacts_now
        assert "1" in contacts_now and "3" in contacts_now

    async def test_force_sync_fetches_all(self, contacts_plugin: ContactsPlugin) -> None:
        manifest = {"1": 100}
        contacts = {"1": Contact(uid="1", name="C1", timestamp=100)}
        await contacts_plugin.store.save_device("dev1", manifest, contacts)

        sent_packets: List[Packet] = []

        async def mock_send(pkt: Packet) -> None:
            sent_packets.append(pkt)
            if pkt.type == REQUEST_ALL_UIDS_TIMESTAMPS:
                asyncio.create_task(
                    contacts_plugin.handle_packet(
                        Packet(
                            type=RESPONSE_UIDS_TIMESTAMPS,
                            body={"uids": ["1"], "1": 100},
                        ),
                        "dev1",
                        mock_send,
                    )
                )
            elif pkt.type == REQUEST_VCARDS_BY_UID:
                asyncio.create_task(
                    contacts_plugin.handle_packet(
                        Packet(
                            type=RESPONSE_VCARDS,
                            body={"uids": ["1"], "1": "BEGIN:VCARD\r\nFN:C1 Force\r\nEND:VCARD\r\n"},
                        ),
                        "dev1",
                        mock_send,
                    )
                )

        res = await contacts_plugin.sync("dev1", mock_send, force=True)
        assert res.contacts_updated == 1
        assert any(p.type == REQUEST_VCARDS_BY_UID for p in sent_packets)


class TestBatchingSafety:
    def test_chunk_list_helper(self) -> None:
        items = list(range(100))
        chunks = chunk_list(items, chunk_size=25)
        assert len(chunks) == 4
        assert all(len(c) == 25 for c in chunks)

        chunks_50 = chunk_list(items, chunk_size=50)
        assert len(chunks_50) == 2

        with pytest.raises(ValueError):
            chunk_list(items, chunk_size=0)

    async def test_120_contacts_chunks_into_packets(self, contacts_plugin: ContactsPlugin) -> None:
        uids = [str(i) for i in range(120)]
        remote_manifest = {uid: 1000 + int(uid) for uid in uids}

        vcard_batches_sent: List[List[str]] = []

        async def mock_send(pkt: Packet) -> None:
            if pkt.type == REQUEST_ALL_UIDS_TIMESTAMPS:
                asyncio.create_task(
                    contacts_plugin.handle_packet(
                        Packet(type=RESPONSE_UIDS_TIMESTAMPS, body={"uids": uids, **remote_manifest}),
                        "dev1",
                        mock_send,
                    )
                )
            elif pkt.type == REQUEST_VCARDS_BY_UID:
                batch = pkt.body.get("uids", [])
                vcard_batches_sent.append(batch)
                body: Dict[str, Any] = {"uids": batch}
                for uid in batch:
                    body[uid] = f"BEGIN:VCARD\r\nFN:User {uid}\r\nEND:VCARD\r\n"
                asyncio.create_task(
                    contacts_plugin.handle_packet(
                        Packet(type=RESPONSE_VCARDS, body=body),
                        "dev1",
                        mock_send,
                    )
                )

        res = await contacts_plugin.sync("dev1", mock_send, batch_size=50)
        assert res.total_contacts == 120
        # 120 split with batch_size 50 => 50, 50, 20 (3 batches)
        assert len(vcard_batches_sent) == 3
        assert len(vcard_batches_sent[0]) == 50
        assert len(vcard_batches_sent[1]) == 50
        assert len(vcard_batches_sent[2]) == 20


class TestAtomicStorage:
    async def test_atomic_temp_replace(self, contacts_plugin: ContactsPlugin) -> None:
        store = contacts_plugin.store
        manifest = {"1": 100}
        contacts = {"1": Contact(uid="1", name="Alice")}
        await store.save_device("dev1", manifest, contacts)

        target_file = store.contacts_dir / "dev1.json"
        assert target_file.exists()

        raw = json.loads(target_file.read_text(encoding="utf-8"))
        assert raw["device_id"] == "dev1"
        assert raw["manifest"] == {"1": 100}
        assert raw["contacts"]["1"]["name"] == "Alice"

    async def test_storage_permissions_0600(self, contacts_plugin: ContactsPlugin) -> None:
        store = contacts_plugin.store
        await store.save_device("dev1", {"1": 100}, {"1": Contact(uid="1", name="Alice")})

        target_file = store.contacts_dir / "dev1.json"
        mode = oct(target_file.stat().st_mode & 0o777)
        assert mode == "0o600"

    async def test_multi_device_isolation(self, contacts_plugin: ContactsPlugin) -> None:
        store = contacts_plugin.store
        await store.save_device("phone_A", {"1": 10}, {"1": Contact(uid="1", name="Alice A")})
        await store.save_device("phone_B", {"2": 20}, {"2": Contact(uid="2", name="Bob B")})

        _, contacts_a = await store.load_device("phone_A")
        _, contacts_b = await store.load_device("phone_B")

        assert len(contacts_a) == 1
        assert "1" in contacts_a and "2" not in contacts_a
        assert len(contacts_b) == 1
        assert "2" in contacts_b and "1" not in contacts_b

    async def test_corrupt_file_recovery(self, contacts_plugin: ContactsPlugin) -> None:
        store = contacts_plugin.store
        await store.initialize()
        file_path = store.contacts_dir / "corrupt_dev.json"
        file_path.write_text("{NOT_VALID_JSON...", encoding="utf-8")

        manifest, contacts = await store.load_device("corrupt_dev")
        assert manifest == {}
        assert contacts == {}

    async def test_list_and_search_contacts(self, contacts_plugin: ContactsPlugin) -> None:
        store = contacts_plugin.store
        contacts = {
            "1": Contact(
                uid="1",
                name="Zara Phillips",
                phones=[PhoneNumber(number="+1-555-0101", normalized="+15550101", type="CELL")],
            ),
            "2": Contact(
                uid="2",
                name="Adam Baker",
                phones=[PhoneNumber(number="555-0202", normalized="5550202", type="WORK")],
            ),
            "3": Contact(
                uid="3",
                name="Charlie Cox",
                organization="Netflix",
            ),
        }
        await store.save_device("dev1", {"1": 1, "2": 2, "3": 3}, contacts)

        # list_contacts sorting
        total, lst = await store.list_contacts("dev1", limit=2, offset=0)
        assert total == 3
        assert len(lst) == 2
        assert lst[0].name == "Adam Baker"
        assert lst[1].name == "Charlie Cox"

        # search by name
        res = await store.search_contacts("dev1", "zara")
        assert len(res) == 1
        assert res[0].name == "Zara Phillips"

        # search by normalized digits
        res_digits = await store.search_contacts("dev1", "0202")
        assert len(res_digits) == 1
        assert res_digits[0].name == "Adam Baker"

        # search by org
        res_org = await store.search_contacts("dev1", "netflix")
        assert len(res_org) == 1
        assert res_org[0].name == "Charlie Cox"


class TestUnpairEviction:
    async def test_on_device_unpaired_removes_file(self, contacts_plugin: ContactsPlugin) -> None:
        store = contacts_plugin.store
        await store.save_device("dev1", {"1": 100}, {"1": Contact(uid="1", name="Alice")})
        file_path = store.contacts_dir / "dev1.json"
        assert file_path.exists()

        await contacts_plugin.clear_device("dev1")
        assert not file_path.exists()

    async def test_on_device_unpaired_clears_memory(self, contacts_plugin: ContactsPlugin) -> None:
        store = contacts_plugin.store
        await store.save_device("dev1", {"1": 100}, {"1": Contact(uid="1", name="Alice")})
        assert "dev1" in store._cache

        await contacts_plugin.clear_device("dev1")
        assert "dev1" not in store._cache


class TestEventBus:
    async def test_contacts_synchronized_event_emitted(
        self, contacts_plugin: ContactsPlugin, event_bus: EventBus
    ) -> None:
        received_events: List[ContactsSynchronizedEvent] = []
        event_bus.subscribe(ContactsSynchronizedEvent, lambda ev: received_events.append(ev))

        async def mock_send(pkt: Packet) -> None:
            if pkt.type == REQUEST_ALL_UIDS_TIMESTAMPS:
                asyncio.create_task(
                    contacts_plugin.handle_packet(
                        Packet(type=RESPONSE_UIDS_TIMESTAMPS, body={"uids": ["1"], "1": 100}),
                        "dev1",
                        mock_send,
                    )
                )
            elif pkt.type == REQUEST_VCARDS_BY_UID:
                asyncio.create_task(
                    contacts_plugin.handle_packet(
                        Packet(
                            type=RESPONSE_VCARDS,
                            body={"uids": ["1"], "1": "BEGIN:VCARD\r\nFN:Alice\r\nEND:VCARD\r\n"},
                        ),
                        "dev1",
                        mock_send,
                    )
                )

        await contacts_plugin.sync("dev1", mock_send)
        await asyncio.sleep(0.01)

        assert len(received_events) == 1
        ev = received_events[0]
        assert ev.device_id == "dev1"
        assert ev.total_contacts == 1
        assert ev.contacts_added == 1
        assert ev.duration >= 0.0


class TestTimeoutsAndErrors:
    async def test_timestamp_request_timeout(self, contacts_plugin: ContactsPlugin) -> None:
        async def mock_send_blackhole(pkt: Packet) -> None:
            pass  # Never reply

        with pytest.raises(SyncTimeoutError):
            await contacts_plugin.sync("dev1", mock_send_blackhole, timeout=0.05)

    async def test_vcard_batch_timeout(self, contacts_plugin: ContactsPlugin) -> None:
        async def mock_send(pkt: Packet) -> None:
            if pkt.type == REQUEST_ALL_UIDS_TIMESTAMPS:
                asyncio.create_task(
                    contacts_plugin.handle_packet(
                        Packet(type=RESPONSE_UIDS_TIMESTAMPS, body={"uids": ["1"], "1": 100}),
                        "dev1",
                        mock_send,
                    )
                )
            # Never reply to REQUEST_VCARDS_BY_UID

        with pytest.raises(SyncTimeoutError):
            await contacts_plugin.sync("dev1", mock_send, timeout=0.05)

    async def test_sync_unpaired_missing_send_packet(self, contacts_plugin: ContactsPlugin) -> None:
        with pytest.raises(ConnectionClosedError):
            await contacts_plugin.sync("dev1", None)  # type: ignore[arg-type]

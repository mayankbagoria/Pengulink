"""pengulink.plugins.contacts.plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect Contacts synchronization plugin for PenguLink.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pengulink.core.config import Config
from pengulink.core.events import ContactsSynchronizedEvent, DeviceUnpaired, EventBus
from pengulink.core.exceptions import (
    ConnectionClosedError,
    ConnectionTimeoutError,
    ContactsError,
    ContactNotFoundError,
    SyncTimeoutError,
)
from pengulink.network.packet import Packet
from pengulink.plugins.base import Plugin, SendPacketCallable, safe_send_packet
from pengulink.plugins.contacts.store import ContactStore, SyncDiff, chunk_list
from pengulink.plugins.contacts.vcard import Contact, parse_vcard

logger = logging.getLogger("pengulink.plugins.contacts")

REQUEST_ALL_UIDS_TIMESTAMPS = "kdeconnect.contacts.request_all_uids_timestamps"
RESPONSE_UIDS_TIMESTAMPS = "kdeconnect.contacts.response_uids_timestamps"
REQUEST_VCARDS_BY_UID = "kdeconnect.contacts.request_vcards_by_uid"
RESPONSE_VCARDS = "kdeconnect.contacts.response_vcards"
LEGACY_CONTACTS = "kdeconnect.contacts"

DEFAULT_BATCH_SIZE = 25
MAX_BATCH_SIZE = 50
DEFAULT_SYNC_TIMEOUT = 15.0


@dataclass(frozen=True)
class ContactsSyncResult:
    """Summary of a completed contact synchronization session."""

    device_id: str
    status: str
    total_contacts: int
    contacts_added: int
    contacts_updated: int
    contacts_deleted: int
    duration: float

    @property
    def total(self) -> int:
        return self.total_contacts

    @property
    def added(self) -> int:
        return self.contacts_added

    @property
    def updated(self) -> int:
        return self.contacts_updated

    @property
    def deleted(self) -> int:
        return self.contacts_deleted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "status": self.status,
            "total_contacts": self.total_contacts,
            "total": self.total_contacts,
            "contacts_added": self.contacts_added,
            "added": self.contacts_added,
            "contacts_updated": self.contacts_updated,
            "updated": self.contacts_updated,
            "contacts_deleted": self.contacts_deleted,
            "deleted": self.contacts_deleted,
            "duration": round(self.duration, 3),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


class ContactsPlugin(Plugin):
    """Contacts plugin managing vCard synchronization and address book queries."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        config: Optional[Config] = None,
        data_dir: Optional[Path | str] = None,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self.config = config
        resolved_data_dir = data_dir or (
            config.storage.data_dir
            if config and hasattr(config, "storage") and hasattr(config.storage, "data_dir")
            else Path.home() / ".local" / "share" / "pengulink"
        )
        self.store = ContactStore(data_dir=resolved_data_dir)
        self.default_timeout: float = DEFAULT_SYNC_TIMEOUT

        self._timestamp_futures: Dict[str, asyncio.Future[Dict[str, int]]] = {}
        self._vcard_futures: Dict[str, asyncio.Future[Dict[str, str]]] = {}
        self._device_sync_locks: Dict[str, asyncio.Lock] = {}
        self._unsub_unpair: Optional[Callable[[], None]] = None

        if self.event_bus:
            self._unsub_unpair = self.event_bus.subscribe(DeviceUnpaired, self._on_device_unpaired_event)

    @property
    def plugin_id(self) -> str:
        return "contacts"

    @property
    def incoming_capabilities(self) -> List[str]:
        return [
            RESPONSE_UIDS_TIMESTAMPS,
            RESPONSE_VCARDS,
            REQUEST_ALL_UIDS_TIMESTAMPS,
            REQUEST_VCARDS_BY_UID,
            LEGACY_CONTACTS,
        ]

    @property
    def outgoing_capabilities(self) -> List[str]:
        return [
            REQUEST_ALL_UIDS_TIMESTAMPS,
            REQUEST_VCARDS_BY_UID,
            RESPONSE_UIDS_TIMESTAMPS,
            RESPONSE_VCARDS,
            LEGACY_CONTACTS,
        ]

    def _on_device_unpaired_event(self, event: DeviceUnpaired) -> None:
        if event and event.device_id:
            self.on_device_unpaired(event.device_id)

    def on_device_unpaired(self, device_id: str) -> None:
        """Purge cached contacts upon device unpair."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.clear_device(device_id))
        except RuntimeError:
            asyncio.run(self.clear_device(device_id))

    async def clear_device(self, device_id: str) -> bool:
        """Purge cached contacts store and in-memory indices for target device."""
        return await self.store.clear_device(device_id)

    async def start(self) -> None:
        await self.store.initialize()

    async def stop(self) -> None:
        if self._unsub_unpair:
            try:
                self._unsub_unpair()
            except Exception:
                pass
            self._unsub_unpair = None

        for fut in list(self._timestamp_futures.values()):
            if not fut.done():
                fut.cancel()
        self._timestamp_futures.clear()

        for fut in list(self._vcard_futures.values()):
            if not fut.done():
                fut.cancel()
        self._vcard_futures.clear()

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Process incoming contact wire packets."""
        if not device_id:
            return

        body = packet.body if isinstance(packet.body, dict) else {}

        # 1. Incoming timestamp manifest response
        if packet.type == RESPONSE_UIDS_TIMESTAMPS:
            uids_list = body.get("uids", [])
            manifest: Dict[str, int] = {}
            if isinstance(uids_list, list):
                for uid in uids_list:
                    uid_str = str(uid)
                    raw_val = body.get(uid_str, 0)
                    try:
                        manifest[uid_str] = int(round(float(raw_val)))
                    except (ValueError, TypeError, OverflowError):
                        manifest[uid_str] = 0

            fut = self._timestamp_futures.get(device_id)
            if fut and not fut.done():
                fut.set_result(manifest)
            return

        # 2. Incoming vCard payload batch response
        if packet.type == RESPONSE_VCARDS:
            uids_list = body.get("uids", [])
            vcards: Dict[str, str] = {}
            if isinstance(uids_list, list):
                for uid in uids_list:
                    uid_str = str(uid)
                    vcards[uid_str] = str(body.get(uid_str, ""))

            fut = self._vcard_futures.get(device_id)
            if fut and not fut.done():
                fut.set_result(vcards)
            return

        # 3. Server role: peer requests our timestamps
        if packet.type == REQUEST_ALL_UIDS_TIMESTAMPS:
            manifest, _ = await self.store.load_device(device_id)
            resp = Packet(
                type=RESPONSE_UIDS_TIMESTAMPS,
                body={"uids": list(manifest.keys()), **manifest},
            )
            await safe_send_packet(send_packet, resp)
            return

        # 4. Server role: peer requests vCards by UID
        if packet.type == REQUEST_VCARDS_BY_UID:
            req_uids = body.get("uids", [])
            _, contacts = await self.store.load_device(device_id)
            resp_body: Dict[str, Any] = {"uids": []}
            if isinstance(req_uids, list):
                for uid in req_uids:
                    uid_str = str(uid)
                    c = contacts.get(uid_str)
                    if c:
                        resp_body["uids"].append(uid_str)
                        resp_body[uid_str] = c.raw_vcard
            resp = Packet(type=RESPONSE_VCARDS, body=resp_body)
            await safe_send_packet(send_packet, resp)
            return

    def _get_sync_lock(self, device_id: str) -> asyncio.Lock:
        if device_id not in self._device_sync_locks:
            self._device_sync_locks[device_id] = asyncio.Lock()
        return self._device_sync_locks[device_id]

    async def sync(
        self,
        device_id: str,
        send_packet: SendPacketCallable,
        force: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: Optional[float] = None,
    ) -> ContactsSyncResult:
        """Execute full or incremental synchronization with target device."""
        if not callable(send_packet):
            raise ConnectionClosedError(
                f"Cannot sync contacts: send_packet callable missing for {device_id}"
            )

        batch_size = max(1, min(MAX_BATCH_SIZE, batch_size))
        eff_timeout = timeout if timeout is not None else self.default_timeout
        start_time = time.time()

        async with self._get_sync_lock(device_id):
            loop = asyncio.get_running_loop()

            # Phase 1: Request manifest of UIDs and timestamps
            ts_fut: asyncio.Future[Dict[str, int]] = loop.create_future()
            self._timestamp_futures[device_id] = ts_fut
            req_pkt = Packet(type=REQUEST_ALL_UIDS_TIMESTAMPS, body={})

            try:
                await safe_send_packet(send_packet, req_pkt)
                remote_manifest = await asyncio.wait_for(ts_fut, timeout=eff_timeout)
            except (asyncio.TimeoutError, TimeoutError) as exc:
                raise SyncTimeoutError(
                    f"Timed out waiting for timestamp manifest from {device_id}"
                ) from exc
            finally:
                self._timestamp_futures.pop(device_id, None)

            # Phase 2: Compute diff with local cached store
            cached_manifest, cached_contacts = await self.store.load_device(device_id)
            diff = self.store.compute_diff(cached_manifest, remote_manifest, force=force)

            # Phase 3: Purge deleted contacts locally
            for uid in diff.uids_to_delete:
                cached_contacts.pop(uid, None)

            # Phase 4: Fetch batches of new or modified vCards
            added_count = 0
            updated_count = 0

            if diff.uids_to_fetch:
                chunks = chunk_list(diff.uids_to_fetch, chunk_size=batch_size)
                for chunk in chunks:
                    vcard_fut: asyncio.Future[Dict[str, str]] = loop.create_future()
                    self._vcard_futures[device_id] = vcard_fut
                    vcard_req = Packet(type=REQUEST_VCARDS_BY_UID, body={"uids": chunk})

                    try:
                        await safe_send_packet(send_packet, vcard_req)
                        batch_vcards = await asyncio.wait_for(vcard_fut, timeout=eff_timeout)
                    except (asyncio.TimeoutError, TimeoutError) as exc:
                        raise SyncTimeoutError(
                            f"Timed out waiting for vCard batch from {device_id}"
                        ) from exc
                    finally:
                        self._vcard_futures.pop(device_id, None)

                    for uid, raw_vcard in batch_vcards.items():
                        ts = remote_manifest.get(uid, 0)
                        contact = parse_vcard(uid, raw_vcard, device_id=device_id, timestamp=ts)
                        if uid in cached_contacts:
                            updated_count += 1
                        else:
                            added_count += 1
                        cached_contacts[uid] = contact

            # Phase 5: Persist updated store atomically
            await self.store.save_device(
                device_id=device_id,
                manifest=remote_manifest,
                contacts=cached_contacts,
            )

            duration = time.time() - start_time
            result = ContactsSyncResult(
                device_id=device_id,
                status="synchronized",
                total_contacts=len(cached_contacts),
                contacts_added=added_count,
                contacts_updated=updated_count,
                contacts_deleted=len(diff.uids_to_delete),
                duration=duration,
            )

            # Phase 6: Publish event to EventBus
            if self.event_bus:
                await self.event_bus.publish(
                    ContactsSynchronizedEvent(
                        device_id=device_id,
                        total_contacts=result.total_contacts,
                        contacts_added=result.contacts_added,
                        contacts_updated=result.contacts_updated,
                        contacts_deleted=result.contacts_deleted,
                        duration=result.duration,
                    )
                )

            return result

    async def list_contacts(
        self,
        device_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Tuple[int, List[Contact]]:
        """List cached contacts for device."""
        return await self.store.list_contacts(device_id, limit=limit, offset=offset)

    async def search_contacts(self, device_id: str, query: str) -> List[Contact]:
        """Search cached contacts for device."""
        return await self.store.search_contacts(device_id, query)

    async def get_contact(self, device_id: str, uid: str) -> Optional[Contact]:
        """Fetch single cached contact by UID."""
        return await self.store.get_contact(device_id, uid)


__all__ = [
    "REQUEST_ALL_UIDS_TIMESTAMPS",
    "RESPONSE_UIDS_TIMESTAMPS",
    "REQUEST_VCARDS_BY_UID",
    "RESPONSE_VCARDS",
    "LEGACY_CONTACTS",
    "DEFAULT_BATCH_SIZE",
    "MAX_BATCH_SIZE",
    "DEFAULT_SYNC_TIMEOUT",
    "ContactsSyncResult",
    "ContactsPlugin",
]

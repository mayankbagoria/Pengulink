"""pengulink.plugins.contacts.store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Device-scoped JSON persistence and diff synchronization for Contacts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar

from pengulink.core.exceptions import PathTraversalError, StorageError, StorageWriteError
from pengulink.plugins.contacts.vcard import Contact

logger = logging.getLogger("pengulink.plugins.contacts.store")

_DEVICE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

T = TypeVar("T")


def chunk_list(items: List[T], chunk_size: int = 25) -> List[List[T]]:
    """Partition a list into sub-lists of at most chunk_size elements."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


@dataclass(frozen=True)
class SyncDiff:
    """Result of computing diff between remote manifest and cached manifest."""

    uids_to_fetch: List[str]
    uids_to_delete: Set[str]
    unchanged_uids: Set[str]

    @property
    def total_remote(self) -> int:
        return len(self.uids_to_fetch) + len(self.unchanged_uids)


class ContactStore:
    """Device-scoped persistent storage and diff manager for contact books."""

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self.contacts_dir = self.data_dir / "contacts"
        self._locks: Dict[str, asyncio.Lock] = {}
        self._cache: Dict[str, Dict[str, Any]] = {}  # device_id -> {manifest, contacts, last_sync}
        self._initialized = False

    async def initialize(self) -> None:
        """Ensure storage directory exists with restrictive 0700 permissions."""
        if self._initialized:
            return
        self.contacts_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.contacts_dir, 0o700)
        except OSError:
            pass
        self._initialized = True

    def _get_device_lock(self, device_id: str) -> asyncio.Lock:
        if device_id not in self._locks:
            self._locks[device_id] = asyncio.Lock()
        return self._locks[device_id]

    def _get_device_file(self, device_id: str) -> Path:
        if not device_id or not _DEVICE_ID_RE.match(device_id):
            raise PathTraversalError(f"Invalid device_id '{device_id}' contains forbidden characters")
        file_path = self.contacts_dir / f"{device_id}.json"
        if not file_path.resolve().is_relative_to(self.contacts_dir.resolve()):
            raise PathTraversalError(f"Target path '{file_path}' escapes contacts directory")
        return file_path

    async def load_device(self, device_id: str) -> Tuple[Dict[str, int], Dict[str, Contact]]:
        """Load manifest and contacts from in-memory cache or disk."""
        async with self._get_device_lock(device_id):
            if device_id in self._cache:
                data = self._cache[device_id]
                return dict(data["manifest"]), dict(data["contacts"])

            file_path = self._get_device_file(device_id)
            if not file_path.exists():
                self._cache[device_id] = {"manifest": {}, "contacts": {}, "last_sync": 0.0}
                return {}, {}

            try:
                content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
                raw = json.loads(content)
                manifest: Dict[str, int] = {
                    str(k): int(v) for k, v in raw.get("manifest", {}).items()
                }
                contacts: Dict[str, Contact] = {
                    str(k): Contact.from_dict(v)
                    for k, v in raw.get("contacts", {}).items()
                    if isinstance(v, dict)
                }
                last_sync = float(raw.get("last_sync", 0.0))
                self._cache[device_id] = {
                    "manifest": manifest,
                    "contacts": contacts,
                    "last_sync": last_sync,
                }
                return dict(manifest), dict(contacts)
            except Exception as exc:
                logger.error("Failed loading contact cache for device %s: %s", device_id, exc)
                self._cache[device_id] = {"manifest": {}, "contacts": {}, "last_sync": 0.0}
                return {}, {}

    async def save_device(
        self,
        device_id: str,
        manifest: Dict[str, int],
        contacts: Dict[str, Contact],
    ) -> None:
        """Atomically persist device contacts to disk via temporary file."""
        await self.initialize()
        async with self._get_device_lock(device_id):
            now = time.time()
            self._cache[device_id] = {
                "manifest": dict(manifest),
                "contacts": dict(contacts),
                "last_sync": now,
            }

            file_path = self._get_device_file(device_id)
            tmp_path = file_path.with_suffix(f".{os.getpid()}.{time.time_ns()}.tmp")

            payload = {
                "device_id": device_id,
                "last_sync": now,
                "manifest": manifest,
                "contacts": {uid: c.to_dict() for uid, c in contacts.items()},
            }

            def _write_and_replace() -> None:
                try:
                    content = json.dumps(payload, indent=2, ensure_ascii=False)
                    tmp_path.write_text(content, encoding="utf-8")
                    try:
                        os.chmod(tmp_path, 0o600)
                    except OSError:
                        pass
                    os.replace(tmp_path, file_path)
                except Exception as exc:
                    if tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except OSError:
                            pass
                    raise StorageWriteError(
                        f"Failed atomic write for contacts {file_path}: {exc}"
                    ) from exc

            await asyncio.to_thread(_write_and_replace)

    def compute_diff(
        self,
        cached_manifest: Dict[str, int],
        remote_manifest: Dict[str, int],
        force: bool = False,
    ) -> SyncDiff:
        """Compute set of UIDs needing vCard fetch, deletion, or no change."""
        if force:
            return SyncDiff(
                uids_to_fetch=list(remote_manifest.keys()),
                uids_to_delete=set(cached_manifest.keys()) - set(remote_manifest.keys()),
                unchanged_uids=set(),
            )

        uids_to_delete = set(cached_manifest.keys()) - set(remote_manifest.keys())
        uids_to_fetch = [
            uid
            for uid, ts in remote_manifest.items()
            if uid not in cached_manifest or ts > cached_manifest[uid]
        ]
        unchanged_uids = set(remote_manifest.keys()) - set(uids_to_fetch)
        return SyncDiff(
            uids_to_fetch=uids_to_fetch,
            uids_to_delete=uids_to_delete,
            unchanged_uids=unchanged_uids,
        )

    async def clear_device(self, device_id: str) -> bool:
        """Purge disk file and in-memory cache upon device unpair."""
        async with self._get_device_lock(device_id):
            had_mem = self._cache.pop(device_id, None) is not None
            file_path = self._get_device_file(device_id)
            had_disk = False
            if file_path.exists():
                try:
                    await asyncio.to_thread(file_path.unlink)
                    had_disk = True
                except OSError as exc:
                    logger.warning("Could not unlink contact store %s: %s", file_path, exc)
            return had_mem or had_disk

    async def get_contact(self, device_id: str, uid: str) -> Optional[Contact]:
        """Fetch a single contact by UID."""
        _, contacts = await self.load_device(device_id)
        return contacts.get(uid)

    async def list_contacts(
        self,
        device_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Tuple[int, List[Contact]]:
        """Return total contact count and paginated list sorted by name."""
        _, contacts_map = await self.load_device(device_id)
        sorted_contacts = sorted(contacts_map.values(), key=lambda c: c.name.lower())
        total = len(sorted_contacts)
        if offset > 0:
            sorted_contacts = sorted_contacts[offset:]
        if limit is not None and limit >= 0:
            sorted_contacts = sorted_contacts[:limit]
        return total, sorted_contacts

    async def search_contacts(self, device_id: str, query: str) -> List[Contact]:
        """Case-insensitive search across name, phones, emails, and organization."""
        _, contacts_map = await self.load_device(device_id)
        q = query.strip().lower()
        if not q:
            return sorted(contacts_map.values(), key=lambda c: c.name.lower())

        digits = re.sub(r"[^\d]", "", q)
        matches: List[Contact] = []
        for c in contacts_map.values():
            if (
                q in c.name.lower()
                or (c.first_name and q in c.first_name.lower())
                or (c.last_name and q in c.last_name.lower())
                or (c.organization and q in c.organization.lower())
                or (c.title and q in c.title.lower())
                or any(q in em.email.lower() for em in c.emails)
                or any(q in p.number.lower() for p in c.phones)
            ):
                matches.append(c)
                continue

            # Normalized phone digit matching
            if digits and any(digits in p.normalized for p in c.phones):
                matches.append(c)

        return sorted(matches, key=lambda c: c.name.lower())


__all__ = [
    "chunk_list",
    "SyncDiff",
    "ContactStore",
]

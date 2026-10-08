"""pengulink.storage
~~~~~~~~~~~~~~~~~
Pluggable persistence layer for trusted devices, certificates, and configuration.
"""

from __future__ import annotations

from pengulink.storage.base import DeviceStore, StorageBackend
from pengulink.storage.json_store import JsonStorageBackend
from pengulink.storage.sqlite_store import SqliteStorageBackend

__all__ = [
    "StorageBackend",
    "DeviceStore",
    "JsonStorageBackend",
    "SqliteStorageBackend",
]

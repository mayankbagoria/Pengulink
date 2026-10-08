"""pengulink.storage.sqlite_store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SQLite persistence backend for PenguLink devices and configuration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pengulink.core.exceptions import StorageError
from pengulink.storage.base import StorageBackend

logger = logging.getLogger("pengulink.storage.sqlite_store")


class SqliteStorageBackend(StorageBackend):
    """SQLite-backed storage implementation."""

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self._lock = asyncio.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    async def initialize(self) -> None:
        """Create database tables."""
        async with self._lock:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(self._init_db_sync)

    def _init_db_sync(self) -> None:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        self._conn = conn
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS devices (
                    device_id TEXT PRIMARY KEY,
                    device_name TEXT NOT NULL,
                    device_type TEXT NOT NULL,
                    certificate_pem TEXT,
                    is_paired INTEGER NOT NULL DEFAULT 0,
                    data_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                )
            """)
        logger.debug("SqliteStorageBackend initialized at %s", self.db_path)

    async def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return await asyncio.to_thread(self._get_device_sync, device_id)

    def _get_device_sync(self, device_id: str) -> Optional[Dict[str, Any]]:
        if not self._conn:
            return None
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM devices WHERE device_id = ?", (device_id,))
        row = cur.fetchone()
        if not row:
            return None
        res = {
            "device_id": row["device_id"],
            "device_name": row["device_name"],
            "device_type": row["device_type"],
            "certificate_pem": row["certificate_pem"],
            "is_paired": bool(row["is_paired"]),
        }
        if row["data_json"]:
            try:
                extra = json.loads(row["data_json"])
                if isinstance(extra, dict):
                    res.update(extra)
            except Exception:
                pass
        return res

    async def save_device(self, device_id: str, data: Dict[str, Any]) -> None:
        async with self._lock:
            await asyncio.to_thread(self._save_device_sync, device_id, data)

    def _save_device_sync(self, device_id: str, data: Dict[str, Any]) -> None:
        if not self._conn:
            return
        name = data.get("device_name", "Unknown")
        dev_type = data.get("device_type", "desktop")
        cert_pem = data.get("certificate_pem")
        is_paired = 1 if data.get("is_paired") else 0

        # Store extra keys in data_json
        extra = {k: v for k, v in data.items() if k not in ("device_id", "device_name", "device_type", "certificate_pem", "is_paired")}
        extra_json = json.dumps(extra) if extra else None

        with self._conn:
            self._conn.execute("""
                INSERT INTO devices (device_id, device_name, device_type, certificate_pem, is_paired, data_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(device_id) DO UPDATE SET
                    device_name = excluded.device_name,
                    device_type = excluded.device_type,
                    certificate_pem = excluded.certificate_pem,
                    is_paired = excluded.is_paired,
                    data_json = excluded.data_json
            """, (device_id, name, dev_type, cert_pem, is_paired, extra_json))

    async def delete_device(self, device_id: str) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self._delete_device_sync, device_id)

    def _delete_device_sync(self, device_id: str) -> bool:
        if not self._conn:
            return False
        with self._conn:
            cur = self._conn.execute("DELETE FROM devices WHERE device_id = ?", (device_id,))
            return cur.rowcount > 0

    async def list_devices(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return await asyncio.to_thread(self._list_devices_sync)

    def _list_devices_sync(self) -> List[Dict[str, Any]]:
        if not self._conn:
            return []
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM devices")
        rows = cur.fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = {
                "device_id": row["device_id"],
                "device_name": row["device_name"],
                "device_type": row["device_type"],
                "certificate_pem": row["certificate_pem"],
                "is_paired": bool(row["is_paired"]),
            }
            if row["data_json"]:
                try:
                    extra = json.loads(row["data_json"])
                    if isinstance(extra, dict):
                        item.update(extra)
                except Exception:
                    pass
            result.append(item)
        return result

    async def get_config(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return await asyncio.to_thread(self._get_config_sync, key, default)

    def _get_config_sync(self, key: str, default: Any = None) -> Any:
        if not self._conn:
            return default
        cur = self._conn.cursor()
        cur.execute("SELECT value_json FROM config WHERE key = ?", (key,))
        row = cur.fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value_json"])
        except Exception:
            return default

    async def set_config(self, key: str, value: Any) -> None:
        async with self._lock:
            await asyncio.to_thread(self._set_config_sync, key, value)

    def _set_config_sync(self, key: str, value: Any) -> None:
        if not self._conn:
            return
        val_json = json.dumps(value)
        with self._conn:
            self._conn.execute("""
                INSERT INTO config (key, value_json)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
            """, (key, val_json))

    async def close(self) -> None:
        """Close SQLite database connection."""
        async with self._lock:
            if self._conn:
                await asyncio.to_thread(self._conn.close)
                self._conn = None

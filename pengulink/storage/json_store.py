"""pengulink.storage.json_store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Atomic JSON file storage backend with corruption recovery.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pengulink.core.exceptions import StorageCorruptedError, StorageError, StorageWriteError
from pengulink.storage.base import StorageBackend

logger = logging.getLogger("pengulink.storage.json_store")


class JsonStorageBackend(StorageBackend):
    """File-based JSON storage implementation."""

    def __init__(self, storage_dir: Union[str, Path]) -> None:
        self.storage_dir = Path(storage_dir)
        self.devices_file = self.storage_dir / "devices.json"
        self.config_file = self.storage_dir / "config.json"
        self._lock = asyncio.Lock()
        self._devices_cache: Dict[str, Dict[str, Any]] = {}
        self._config_cache: Dict[str, Any] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Create directory and load cached files."""
        async with self._lock:
            if self._initialized:
                return
            self.storage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            if self.storage_dir.resolve() != Path.home().resolve():
                try:
                    os.chmod(self.storage_dir, 0o700)
                except OSError:
                    pass
            self._devices_cache = await asyncio.to_thread(self._load_json_file, self.devices_file)
            self._config_cache = await asyncio.to_thread(self._load_json_file, self.config_file)
            self._initialized = True
            logger.debug("JsonStorageBackend initialized at %s", self.storage_dir)

    def _load_json_file(self, path: Path) -> Dict[str, Any]:
        """Load and parse JSON file, backing up and recovering if corrupt."""
        if not path.exists():
            return {}
        try:
            content = path.read_text(encoding="utf-8")
            if not content.strip():
                return {}
            data = json.loads(content)
            if not isinstance(data, dict):
                raise StorageCorruptedError(f"Root JSON element in {path} is not a dictionary")
            return data
        except Exception as exc:
            corrupt_backup = path.with_suffix(f".corrupt.{int(time.time())}.bak")
            logger.error("JSON file %s is corrupt (%s). Backing up to %s", path, exc, corrupt_backup)
            try:
                shutil.copy2(path, corrupt_backup)
            except OSError:
                pass
            return {}

    def _save_json_file(self, path: Path, data: Dict[str, Any]) -> None:
        """Write JSON file atomically using a temporary file with secure 0o600 permissions."""
        tmp_file = path.with_suffix(f".tmp.{os.getpid()}.{time.time_ns()}")
        try:
            content = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
            fd = os.open(tmp_file, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            try:
                os.write(fd, content)
            finally:
                os.close(fd)
            try:
                os.chmod(tmp_file, 0o600)
            except OSError:
                pass
            tmp_file.replace(path)
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        except Exception as exc:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass
            raise StorageWriteError(f"Failed writing {path}: {exc}") from exc

    async def _ensure_initialized(self) -> None:
        """Ensure storage directory and cache are loaded before any operation."""
        if not self._initialized:
            await self.initialize()

    async def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_initialized()
        async with self._lock:
            return self._devices_cache.get(device_id)

    async def save_device(self, device_id: str, data: Dict[str, Any]) -> None:
        await self._ensure_initialized()
        async with self._lock:
            self._devices_cache[device_id] = dict(data)
            await asyncio.to_thread(self._save_json_file, self.devices_file, self._devices_cache)

    async def delete_device(self, device_id: str) -> bool:
        await self._ensure_initialized()
        async with self._lock:
            if device_id in self._devices_cache:
                del self._devices_cache[device_id]
                await asyncio.to_thread(self._save_json_file, self.devices_file, self._devices_cache)
                return True
            return False

    async def list_devices(self) -> List[Dict[str, Any]]:
        await self._ensure_initialized()
        async with self._lock:
            return list(self._devices_cache.values())

    async def get_config(self, key: str, default: Any = None) -> Any:
        await self._ensure_initialized()
        async with self._lock:
            return self._config_cache.get(key, default)

    async def set_config(self, key: str, value: Any) -> None:
        await self._ensure_initialized()
        async with self._lock:
            self._config_cache[key] = value
            await asyncio.to_thread(self._save_json_file, self.config_file, self._config_cache)

    async def close(self) -> None:
        """Clean close."""
        self._initialized = False

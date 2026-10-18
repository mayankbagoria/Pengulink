"""pengulink.storage.base
~~~~~~~~~~~~~~~~~~~~~~
Abstract persistence backend interface and high-level DeviceStore manager.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pengulink.storage.base")


class StorageBackend(ABC):
    """Abstract interface for pluggable storage implementations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize storage resources (directories, database tables, etc.)."""
        ...

    @abstractmethod
    async def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored metadata for a device."""
        ...

    @abstractmethod
    async def save_device(self, device_id: str, data: Dict[str, Any]) -> None:
        """Save or update device record."""
        ...

    @abstractmethod
    async def delete_device(self, device_id: str) -> bool:
        """Delete device record. Returns True if existed and deleted."""
        ...

    @abstractmethod
    async def list_devices(self) -> List[Dict[str, Any]]:
        """Retrieve all stored devices."""
        ...

    @abstractmethod
    async def get_config(self, key: str, default: Any = None) -> Any:
        """Get an application configuration value."""
        ...

    @abstractmethod
    async def set_config(self, key: str, value: Any) -> None:
        """Set an application configuration value."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Cleanly close storage backend handles."""
        ...


class DeviceStore:
    """High-level persistence manager wrapping a StorageBackend."""

    def __init__(self, backend: StorageBackend) -> None:
        self.backend = backend

    async def initialize(self) -> None:
        """Initialize underlying backend."""
        await self.backend.initialize()

    async def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored device information."""
        return await self.backend.get_device(device_id)

    async def save_device(
        self,
        device_id: str,
        name: str,
        device_type: str = "desktop",
        certificate_pem: Optional[str] = None,
        is_paired: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist or update device record, preserving existing credentials and pairing state."""
        existing = await self.get_device(device_id)
        if existing:
            final_device_type = existing.get("device_type", device_type) if device_type == "desktop" else device_type
            final_cert = existing.get("certificate_pem") if certificate_pem is None else certificate_pem
            final_is_paired = existing.get("is_paired", False) if not is_paired else True
            record: Dict[str, Any] = dict(existing)
            record.update({
                "device_id": device_id,
                "device_name": name,
                "device_type": final_device_type,
                "certificate_pem": final_cert,
                "is_paired": final_is_paired,
            })
            if extra:
                record.update(extra)
        else:
            record = {
                "device_id": device_id,
                "device_name": name,
                "device_type": device_type,
                "certificate_pem": certificate_pem,
                "is_paired": is_paired,
            }
            if extra:
                record.update(extra)
        await self.backend.save_device(device_id, record)

    async def delete_device(self, device_id: str) -> bool:
        """Delete device from persistent storage."""
        return await self.backend.delete_device(device_id)

    async def list_devices(self) -> List[Dict[str, Any]]:
        """List all devices."""
        return await self.backend.list_devices()

    async def is_device_paired(self, device_id: str) -> bool:
        """Check whether a device is trusted and paired."""
        dev = await self.backend.get_device(device_id)
        if not dev:
            return False
        return bool(dev.get("is_paired", False))

    async def get_device_certificate_pem(self, device_id: str) -> Optional[bytes]:
        """Retrieve pinned certificate PEM bytes for a device."""
        dev = await self.backend.get_device(device_id)
        if not dev:
            return None
        cert_pem = dev.get("certificate_pem")
        if not cert_pem:
            return None
        return cert_pem.encode("utf-8") if isinstance(cert_pem, str) else cert_pem

    async def close(self) -> None:
        """Close storage backend."""
        await self.backend.close()

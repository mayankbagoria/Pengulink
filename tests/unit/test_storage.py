"""Unit tests for pengulink.storage (JSON and SQLite backends)."""

from __future__ import annotations

from pathlib import Path
import pytest

from pengulink.storage.base import DeviceStore
from pengulink.storage.json_store import JsonStorageBackend
from pengulink.storage.sqlite_store import SqliteStorageBackend


async def test_json_storage_crud(tmp_path: Path) -> None:
    backend = JsonStorageBackend(tmp_path / "json_store")
    await backend.initialize()

    # Save device
    dev_data = {
        "device_id": "phone_1",
        "device_name": "Pixel 8",
        "device_type": "phone",
        "certificate_pem": "-----BEGIN CERTIFICATE-----\nTEST\n-----END CERTIFICATE-----",
        "is_paired": True,
    }
    await backend.save_device("phone_1", dev_data)

    # Get device
    loaded = await backend.get_device("phone_1")
    assert loaded is not None
    assert loaded["device_name"] == "Pixel 8"
    assert loaded["is_paired"] is True

    # List devices
    devs = await backend.list_devices()
    assert len(devs) == 1
    assert devs[0]["device_id"] == "phone_1"

    # Config get/set
    await backend.set_config("theme", "dark")
    assert await backend.get_config("theme") == "dark"
    assert await backend.get_config("nonexistent", "default") == "default"

    # Delete device
    deleted = await backend.delete_device("phone_1")
    assert deleted is True
    assert await backend.get_device("phone_1") is None
    assert len(await backend.list_devices()) == 0

    await backend.close()


async def test_json_storage_corruption_recovery(tmp_path: Path) -> None:
    store_dir = tmp_path / "corrupt_store"
    store_dir.mkdir(parents=True, exist_ok=True)
    devices_file = store_dir / "devices.json"

    # Write corrupt JSON bytes
    devices_file.write_text("{{corrupted json garbage", encoding="utf-8")

    backend = JsonStorageBackend(store_dir)
    await backend.initialize()

    # Backend should recover cleanly with empty cache and preserve corrupt backup
    assert await backend.get_device("any_id") is None
    backups = list(store_dir.glob("devices.corrupt.*.bak"))
    assert len(backups) == 1

    await backend.close()


async def test_sqlite_storage_crud(tmp_path: Path) -> None:
    db_path = tmp_path / "pengulink.db"
    backend = SqliteStorageBackend(db_path)
    await backend.initialize()

    dev_data = {
        "device_id": "phone_sqlite_1",
        "device_name": "Galaxy S24",
        "device_type": "phone",
        "certificate_pem": "CERT_DATA",
        "is_paired": True,
        "custom_note": "Work phone",
    }
    await backend.save_device("phone_sqlite_1", dev_data)

    loaded = await backend.get_device("phone_sqlite_1")
    assert loaded is not None
    assert loaded["device_name"] == "Galaxy S24"
    assert loaded["is_paired"] is True
    assert loaded.get("custom_note") == "Work phone"

    # Config
    await backend.set_config("auto_pair", True)
    assert await backend.get_config("auto_pair") is True

    # Delete
    assert await backend.delete_device("phone_sqlite_1") is True
    assert await backend.get_device("phone_sqlite_1") is None

    await backend.close()


async def test_device_store_facade(tmp_path: Path) -> None:
    backend = JsonStorageBackend(tmp_path / "facade_store")
    store = DeviceStore(backend)
    await store.initialize()

    await store.save_device(
        device_id="dev_facade",
        name="Laptop",
        device_type="laptop",
        certificate_pem="CERT_PEM_BYTES",
        is_paired=True,
    )

    assert await store.is_device_paired("dev_facade") is True
    assert await store.is_device_paired("unknown_device") is False

    cert_bytes = await store.get_device_certificate_pem("dev_facade")
    assert cert_bytes == b"CERT_PEM_BYTES"

    await store.close()

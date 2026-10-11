"""tests.unit.test_plugin_share
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for SharePlugin, KDE Connect packet constructors, inbound/outbound
text and URL sharing, zero-byte file handling, permission checking, and lifecycle.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any, List, Optional
import pytest

from pengulink.core.events import (
    EventBus,
    FileReceivedEvent,
    FileTransferProgress,
    ShareReceivedEvent,
)
from pengulink.core.exceptions import ShareError
from pengulink.network.packet import Packet
from pengulink.plugins.share import (
    SHARE_LEGACY_TYPE,
    SHARE_REQUEST_TYPE,
    SHARE_UPDATE_TYPE,
    SharePlugin,
    create_share_file_packet,
    create_share_text_packet,
    create_share_url_packet,
)


def test_share_packet_builders() -> None:
    """Verify standard KDE Connect packet builders produce expected payload schemas."""
    # Text packet
    p_text = create_share_text_packet("Hello KDE Connect")
    assert p_text.type == SHARE_REQUEST_TYPE
    assert p_text.body == {"text": "Hello KDE Connect"}
    assert p_text.payloadSize is None

    # URL packet
    p_url = create_share_url_packet("https://kde.org")
    assert p_url.type == SHARE_REQUEST_TYPE
    assert p_url.body == {"url": "https://kde.org"}
    assert p_url.payloadSize is None

    # File packet
    p_file = create_share_file_packet(
        filename="document.pdf",
        payload_size=1024,
        payload_port=1740,
        number_of_files=2,
    )
    assert p_file.type == SHARE_REQUEST_TYPE
    assert p_file.body["filename"] == "document.pdf"
    assert p_file.body["numberOfFiles"] == 2
    assert p_file.payloadSize == 1024
    assert p_file.payloadTransferInfo == {"port": 1740}


def test_share_plugin_capabilities() -> None:
    """Verify SharePlugin capabilities conform to KDE Connect specification."""
    plugin = SharePlugin()
    assert plugin.plugin_id == "share"
    for cap in (SHARE_REQUEST_TYPE, SHARE_UPDATE_TYPE, SHARE_LEGACY_TYPE):
        assert cap in plugin.incoming_capabilities
        assert cap in plugin.outgoing_capabilities


async def test_inbound_text_share(tmp_path: Path) -> None:
    """Inbound text packet dispatches ShareReceivedEvent."""
    bus = EventBus()
    events: List[Any] = []
    bus.subscribe(ShareReceivedEvent, lambda ev: events.append(ev))

    plugin = SharePlugin(event_bus=bus)
    pkt = create_share_text_packet("Synchronized text note")

    await plugin.handle_packet(pkt, "remote_phone_1", lambda p: None)

    assert len(events) == 1
    ev = events[0]
    assert ev.device_id == "remote_phone_1"
    assert ev.share_type == "text"
    assert ev.text == "Synchronized text note"


async def test_inbound_url_share(tmp_path: Path) -> None:
    """Inbound URL packet dispatches ShareReceivedEvent."""
    bus = EventBus()
    events: List[Any] = []
    bus.subscribe(ShareReceivedEvent, lambda ev: events.append(ev))

    plugin = SharePlugin(event_bus=bus)
    pkt = create_share_url_packet("https://archlinux.org")

    await plugin.handle_packet(pkt, "remote_phone_2", lambda p: None)

    assert len(events) == 1
    ev = events[0]
    assert ev.device_id == "remote_phone_2"
    assert ev.share_type == "url"
    assert ev.url == "https://archlinux.org"


async def test_inbound_zero_byte_file_optimization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-byte file bypasses socket streaming and stages atomic empty file directly."""
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PENGULINK_DOWNLOAD_DIR", str(download_dir))

    bus = EventBus()
    file_events: List[Any] = []
    bus.subscribe(FileReceivedEvent, lambda ev: file_events.append(ev))

    plugin = SharePlugin(event_bus=bus)
    pkt = Packet(
        id=int(time.time() * 1000),
        type=SHARE_REQUEST_TYPE,
        body={"filename": "empty_marker.txt"},
        payloadSize=0,
    )

    await plugin.handle_packet(pkt, "remote_device", lambda p: None)

    created_file = download_dir / "empty_marker.txt"
    assert created_file.exists()
    assert created_file.stat().st_size == 0
    # Mode 0644
    assert (created_file.stat().st_mode & 0o777) == 0o644
    assert len(file_events) == 1
    assert file_events[0].filename == "empty_marker.txt"


async def test_unpaired_device_packet_rejected() -> None:
    """Packets from unpaired devices are dropped when checker returns False."""
    bus = EventBus()
    events: List[Any] = []
    bus.subscribe(ShareReceivedEvent, lambda ev: events.append(ev))

    async def _mock_is_paired(dev_id: str) -> bool:
        return False

    plugin = SharePlugin(event_bus=bus, is_paired_checker=_mock_is_paired)
    pkt = create_share_text_packet("Unauthorized message")

    await plugin.handle_packet(pkt, "rogue_device", lambda p: None)
    assert len(events) == 0


async def test_outbound_share_text() -> None:
    """share_text dispatches kdeconnect.share.request packet via callback."""
    sent: List[Packet] = []

    async def _send(pkt: Packet) -> None:
        sent.append(pkt)

    plugin = SharePlugin()
    res = await plugin.share_text("device_123", "Hello out there", send_packet=_send)
    assert res["status"] in ("transmitted", "ok")
    assert len(sent) == 1
    assert sent[0].type == SHARE_REQUEST_TYPE
    assert sent[0].body["text"] == "Hello out there"


async def test_outbound_share_url() -> None:
    """share_url dispatches kdeconnect.share.request packet via callback."""
    sent: List[Packet] = []

    async def _send(pkt: Packet) -> None:
        sent.append(pkt)

    plugin = SharePlugin()
    res = await plugin.share_url("device_123", "https://github.com", send_packet=_send)
    assert res["status"] in ("transmitted", "ok")
    assert len(sent) == 1
    assert sent[0].type == SHARE_REQUEST_TYPE
    assert sent[0].body["url"] == "https://github.com"


async def test_outbound_share_file_nonexistent_raises() -> None:
    """Attempting to share non-existent file raises FileNotFoundError."""
    plugin = SharePlugin()
    with pytest.raises(FileNotFoundError):
        await plugin.share_file("device_123", Path("/nonexistent/path/file.txt"))


async def test_cancel_device_transfers_and_stop() -> None:
    """cancel_device_transfers and stop drain tasks cleanly."""
    plugin = SharePlugin()
    # No active transfers initially
    await plugin.cancel_device_transfers("device_abc")
    await plugin.stop()
    assert len(plugin._active_transfers) == 0

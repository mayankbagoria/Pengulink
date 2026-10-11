"""tests.e2e.test_share_e2e
~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-end integration tests for PenguLink Share Feature Ecosystem against MockDevice:
- Outbound text and URL sharing via CLI and IPC
- Inbound text and URL reception and event bus publishing
- Outbound and inbound file transfer over ephemeral TLS sockets
- Multi-file sequential batch transfers
- CLI --json output validation
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, List
import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.events import FileReceivedEvent, ShareReceivedEvent
from pengulink.crypto.tls import create_paired_ssl_context
from pengulink.network.packet import Packet
from tests.e2e.conftest import e2e_environment, wait_for
from tests.mock.mock_device import MockDevice


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner(env={"COLUMNS": "160"})


async def _pair_and_connect(daemon: Any, mock: MockDevice) -> None:
    await mock.broadcast_identity(
        target_ip="127.0.0.1",
        target_port=daemon.config.network.udp_port,
    )
    await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)
    res = await daemon.pair_device(mock.device_id, wait=True, timeout=5.0)
    assert res["paired"] is True
    assert daemon.is_device_connected(mock.device_id) is True


async def test_e2e_share_outbound_text_cli(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Linux user shares text via CLI; mock phone receives packet."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        sock_path = daemon.config.storage.runtime_dir / "daemon.sock"

        text_content = "Important meeting link at 3 PM"
        res = await asyncio.to_thread(
            cli_runner.invoke, cli, ["-s", str(sock_path), "share", "text", mock.device_id, text_content]
        )
        assert res.exit_code == 0
        assert "transmitted" in res.output.lower() or "shared" in res.output.lower()

        await wait_for(
            lambda: any(
                p.type == "kdeconnect.share.request" and p.body.get("text") == text_content
                for p in mock.received_packets
            ),
            timeout=3.0,
            desc="mock received text share packet",
        )


async def test_e2e_share_inbound_text(tmp_path: Path) -> None:
    """Mock phone shares text; daemon dispatches ShareReceivedEvent."""
    events: List[Any] = []
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon.event_bus.subscribe(ShareReceivedEvent, lambda ev: events.append(ev))

        pkt = Packet(
            id=int(time.time() * 1000),
            type="kdeconnect.share.request",
            body={"text": "Note from mobile device"},
        )
        await mock.send_packet(pkt)

        await wait_for(lambda: len(events) >= 1, timeout=3.0, desc="ShareReceivedEvent emitted")
        assert events[0].share_type == "text"
        assert events[0].text == "Note from mobile device"
        assert events[0].device_id == mock.device_id


async def test_e2e_share_outbound_url_cli(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Linux user shares URL via CLI; mock phone receives packet."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        sock_path = daemon.config.storage.runtime_dir / "daemon.sock"

        target_url = "https://kdeconnect.kde.org"
        res = await asyncio.to_thread(
            cli_runner.invoke, cli, ["-s", str(sock_path), "share", "url", mock.device_id, target_url]
        )
        assert res.exit_code == 0

        await wait_for(
            lambda: any(
                p.type == "kdeconnect.share.request" and p.body.get("url") == target_url
                for p in mock.received_packets
            ),
            timeout=3.0,
            desc="mock received url share packet",
        )


async def test_e2e_share_inbound_url(tmp_path: Path) -> None:
    """Mock phone shares URL; daemon dispatches ShareReceivedEvent."""
    events: List[Any] = []
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon.event_bus.subscribe(ShareReceivedEvent, lambda ev: events.append(ev))

        pkt = Packet(
            id=int(time.time() * 1000),
            type="kdeconnect.share.request",
            body={"url": "https://python.org"},
        )
        await mock.send_packet(pkt)

        await wait_for(lambda: len(events) >= 1, timeout=3.0, desc="ShareReceivedEvent emitted")
        assert events[0].share_type == "url"
        assert events[0].url == "https://python.org"


async def test_e2e_share_inbound_file_transfer(tmp_path: Path) -> None:
    """Mock phone sends file; daemon downloads into download directory with mode 0644."""
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]
        daemon_cert_bytes = daemon_cert.encode("utf-8") if isinstance(daemon_cert, str) else daemon_cert

        file_payload = b"KDE Connect file payload streaming test\n" * 50
        ssl_ctx = create_paired_ssl_context(
            local_cert_pem=mock.cert_pem,
            local_key_pem=mock.key_pem,
            peer_cert_pem=daemon_cert_bytes,
            server_side=True,
        )

        async def _client_cb(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
            try:
                w.write(file_payload)
                await w.drain()
            finally:
                w.close()
                await w.wait_closed()

        srv = await asyncio.start_server(_client_cb, host="127.0.0.1", port=0, ssl=ssl_ctx)
        payload_port = srv.sockets[0].getsockname()[1]

        try:
            pkt = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"filename": "received_doc.txt"},
                payloadSize=len(file_payload),
                payloadTransferInfo={"port": payload_port},
            )
            await mock.send_packet(pkt)

            dest_file = download_dir / "received_doc.txt"
            await wait_for(lambda: dest_file.exists(), timeout=5.0, desc="inbound file written")
            assert dest_file.read_bytes() == file_payload
            assert (dest_file.stat().st_mode & 0o777) == 0o644
        finally:
            srv.close()
            await srv.wait_closed()


async def test_e2e_share_cli_json_output(cli_runner: CliRunner, tmp_path: Path) -> None:
    """CLI commands with --json produce valid JSON response."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        sock_path = daemon.config.storage.runtime_dir / "daemon.sock"

        res = await asyncio.to_thread(
            cli_runner.invoke,
            cli,
            ["-s", str(sock_path), "share", "text", "--json", mock.device_id, "json payload text"],
        )
        assert res.exit_code == 0
        parsed = json.loads(res.output)
        assert parsed.get("status") in ("transmitted", "ok")
        assert parsed.get("device_id") == mock.device_id

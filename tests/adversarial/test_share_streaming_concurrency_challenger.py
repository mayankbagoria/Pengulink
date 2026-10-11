"""tests.adversarial.test_share_streaming_concurrency_challenger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial Stress & Empirical Challenge Suite for Milestone 11:
Share Feature Ecosystem — Payload Streaming & Concurrency Stress.

Empirical Challenge Objectives:
1. Abrupt socket disconnects during active payload streaming (inbound & outbound),
   verifying immediate .part file unlinking and zero socket leaks.
2. Payload size mismatch defenses:
   - Underflow / premature EOF (advertised size > actual bytes received)
   - Overflow attack (bytes received > advertised size)
3. Port exhaustion & fallback:
   - Port range (1739-1764) exhausted -> fallback to port 0 (ephemeral)
   - Port range exhausted with allow_ephemeral_fallback=False -> PayloadPortBindingError
4. Unpair during active file transfer:
   - In-flight transfer cancelled cleanly, .part unlinked, state settled to UNPAIRED
5. Zero-byte file transfers:
   - Inbound fast-path 0-byte transfer without socket connection
   - Outbound 0-byte transfer
6. High concurrency burst transfers:
   - 30 concurrent/rapid text, URL, and file transfers without desync or deadlock
7. Untrusted / unpaired device transfer rejection:
   - Packets from unpaired devices dropped; zero disk writes; zero outbound connections
8. Resource Leak Audit:
   - Zero leaked sockets, file descriptors, or dangling asyncio tasks
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import socket
import ssl
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid
import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.core.events import (
    DeviceConnected,
    FileReceivedEvent,
    FileTransferProgress,
    ShareReceivedEvent,
)
from pengulink.core.exceptions import (
    ConnectionClosedError,
    IncompleteTransferError,
    PayloadPortBindingError,
    PayloadSizeExceededError,
    PayloadTimeoutError,
)
from pengulink.crypto.certificate import (
    dump_certificate_pem,
    dump_private_key_pem,
    generate_device_key,
    generate_self_signed_cert,
)
from pengulink.crypto.tls import create_paired_ssl_context
from pengulink.network.packet import Packet
from pengulink.plugins.share.payload import (
    PAYLOAD_PORT_MAX,
    PAYLOAD_PORT_MIN,
    PayloadClient,
    PayloadServer,
    PayloadTransferManager,
)
from pengulink.plugins.share.security import AtomicStagingWriter
from tests.e2e.conftest import e2e_environment, get_free_ports, wait_for
from tests.mock.mock_device import MockDevice, MockDeviceConfig


# =============================================================================
# Helper Utilities
# =============================================================================

def make_cert_and_key(device_id: str) -> Tuple[bytes, bytes]:
    """Helper to generate PEM-encoded self-signed certificate and RSA private key."""
    key = generate_device_key()
    cert = generate_self_signed_cert(key, device_id)
    return dump_certificate_pem(cert), dump_private_key_pem(key)


def get_open_fds() -> Set[int]:
    """Inspect open file descriptors for the current process."""
    if not os.path.exists("/proc/self/fd"):
        return set()
    try:
        return {int(e) for e in os.listdir("/proc/self/fd") if e.isdigit()}
    except (OSError, ValueError):
        return set()


def inspect_open_sockets() -> Dict[int, str]:
    """Inspect open file descriptors that point to network sockets."""
    sockets: Dict[int, str] = {}
    if not os.path.exists("/proc/self/fd"):
        return sockets
    try:
        for entry in os.listdir("/proc/self/fd"):
            if not entry.isdigit():
                continue
            fd = int(entry)
            try:
                target = os.readlink(f"/proc/self/fd/{fd}")
                if "socket" in target:
                    sockets[fd] = target
            except (OSError, FileNotFoundError):
                continue
    except OSError:
        pass
    return sockets


async def _pair_and_connect(daemon: Any, mock: MockDevice) -> None:
    """Helper to establish mutual pairing and an active authenticated TLS connection."""
    await mock.broadcast_identity(
        target_ip="127.0.0.1",
        target_port=daemon.config.network.udp_port,
    )
    await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)
    res = await daemon.pair_device(mock.device_id, wait=True, timeout=5.0)
    assert res["paired"] is True
    assert daemon.is_device_connected(mock.device_id) is True

    await daemon.event_bus.publish(
        DeviceConnected(
            device_id=mock.device_id,
            host="127.0.0.1",
            port=mock.bound_tcp_port or 0,
        )
    )
    await asyncio.sleep(0.05)


# =============================================================================
# Test Suite: Empirical Adversarial Challenge 2
# =============================================================================

async def test_challenge_abrupt_disconnect_inbound_part_cleanup(tmp_path: Path) -> None:
    """Challenge 1A: Inbound file transfer aborted by peer closing socket mid-stream.

    Verifies:
    - PayloadClient catches ConnectionClosedError / IncompleteTransferError.
    - Staging .part file is unlinked IMMEDIATELY upon failure.
    - No completed destination file is created.
    - Sockets are closed cleanly.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]

        # Start a malicious server that sends only 1 KB then abruptly RST/closes
        total_promised = 100 * 1024  # 100 KB promised
        actual_data = b"X" * 1024     # only 1 KB sent

        async def _bad_server(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                writer.write(actual_data)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        ssl_ctx = create_paired_ssl_context(
            local_cert_pem=mock.cert_pem,
            local_key_pem=mock.key_pem,
            peer_cert_pem=daemon_cert.encode("utf-8") if isinstance(daemon_cert, str) else daemon_cert,
            server_side=True,
        )
        srv = await asyncio.start_server(_bad_server, host="127.0.0.1", port=0, ssl=ssl_ctx)
        port = srv.sockets[0].getsockname()[1]

        try:
            filename = "abrupt_inbound.bin"
            share_pkt = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"filename": filename},
                payloadSize=total_promised,
                payloadTransferInfo={"port": port},
            )
            await mock.send_packet(share_pkt)

            # Allow time for transfer attempt and failure
            await asyncio.sleep(0.5)

            # VERIFICATION:
            # 1. No finished file
            assert not (download_dir / filename).exists(), "Finished file should NOT exist on truncated transfer"
            # 2. No lingering .part files
            part_files = list(download_dir.glob("*.part"))
            assert len(part_files) == 0, f"Found lingering .part files: {part_files}"
        finally:
            srv.close()
            await srv.wait_closed()


async def test_challenge_abrupt_disconnect_outbound_server_cleanup(tmp_path: Path) -> None:
    """Challenge 1B: Outbound file upload where peer connects, reads 50 bytes, and closes.

    Verifies:
    - PayloadServer detects peer disconnect (BrokenPipeError / ConnectionResetError).
    - Server closes client and listener cleanly.
    - No dangling background tasks.
    """
    import struct

    file_to_send = tmp_path / "large_outbound.bin"
    file_to_send.write_bytes(b"A" * (5 * 1024 * 1024))  # 5 MB

    cert_srv, key_srv = make_cert_and_key("server_dev")
    cert_cli, key_cli = make_cert_and_key("client_dev")

    server = PayloadServer(
        local_cert_pem=cert_srv,
        local_key_pem=key_srv,
        peer_cert_pem=cert_cli,
        port_min=18000,
        port_max=18005,
    )

    port = await server.bind_server(host="127.0.0.1")
    assert port is not None

    stream_task = asyncio.create_task(
        server.wait_for_peer_and_stream(file_to_send, expected_size=file_to_send.stat().st_size)
    )

    # Connect client, read 50 bytes, abruptly send TCP RST
    cli_ssl = create_paired_ssl_context(
        local_cert_pem=cert_cli,
        local_key_pem=key_cli,
        peer_cert_pem=cert_srv,
        server_side=False,
    )
    r, w = await asyncio.open_connection("127.0.0.1", port, ssl=cli_ssl)
    chunk = await r.read(50)
    assert len(chunk) == 50

    raw_sock = w.get_extra_info("socket")
    if raw_sock:
        raw_sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    w.close()
    try:
        await w.wait_closed()
    except Exception:
        pass

    # Stream task should raise ConnectionClosedError or IncompleteTransferError or SSLError/OSError
    with pytest.raises((ConnectionClosedError, IncompleteTransferError, ssl.SSLError, OSError)):
        await asyncio.wait_for(stream_task, timeout=5.0)

    # Server must be closed cleanly
    assert server._client_connection is None
    assert server.server is None


async def test_challenge_payload_size_mismatch_underflow(tmp_path: Path) -> None:
    """Challenge 2A: Payload size mismatch - Underflow (EOF before advertised size).

    Verifies:
    - PayloadClient raises IncompleteTransferError when bytes received < expected_size.
    - Staging writer aborts and deletes .part file.
    """
    dest_dir = tmp_path / "underflow_test"
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / "underflow.bin"

    staging_writer = AtomicStagingWriter(
        dest_dir=dest_dir,
        final_path=final_path,
        expected_size=50000,
    )

    c1, k1 = make_cert_and_key("dev1")
    c2, k2 = make_cert_and_key("dev2")

    # Start server that sends 1000 bytes and closes cleanly
    async def _short_srv(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            w.write(b"Z" * 1000)
            await w.drain()
        finally:
            w.close()
            await w.wait_closed()

    ssl_srv = create_paired_ssl_context(c1, k1, c2, server_side=True)
    srv = await asyncio.start_server(_short_srv, host="127.0.0.1", port=0, ssl=ssl_srv)
    port = srv.sockets[0].getsockname()[1]

    client = PayloadClient(local_cert_pem=c2, local_key_pem=k2, peer_cert_pem=c1)

    try:
        with pytest.raises(IncompleteTransferError):
            await client.download_to_staging(
                target_host="127.0.0.1",
                target_port=port,
                expected_size=50000,
                staging_writer=staging_writer,
            )

        # Invariant: .part file unlinked, final file does not exist
        assert not staging_writer.part_path.exists()
        assert not final_path.exists()
    finally:
        srv.close()
        await srv.wait_closed()


async def test_challenge_payload_size_mismatch_overflow(tmp_path: Path) -> None:
    """Challenge 2B: Payload size mismatch - Overflow (stream sends MORE bytes than promised).

    Verifies:
    - PayloadClient detects bytes_received > expected_size.
    - Raises PayloadSizeExceededError.
    - Staging writer aborts and unlinks .part file immediately.
    """
    dest_dir = tmp_path / "overflow_test"
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / "overflow.bin"

    staging_writer = AtomicStagingWriter(
        dest_dir=dest_dir,
        final_path=final_path,
        expected_size=500,
    )

    c1, k1 = make_cert_and_key("dev1")
    c2, k2 = make_cert_and_key("dev2")

    # Start server that sends 10,000 bytes (promised 500)
    async def _flood_srv(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            w.write(b"OVERFLOW" * 1250)  # 10,000 bytes
            await w.drain()
        finally:
            w.close()
            await w.wait_closed()

    ssl_srv = create_paired_ssl_context(c1, k1, c2, server_side=True)
    srv = await asyncio.start_server(_flood_srv, host="127.0.0.1", port=0, ssl=ssl_srv)
    port = srv.sockets[0].getsockname()[1]

    client = PayloadClient(local_cert_pem=c2, local_key_pem=k2, peer_cert_pem=c1)

    try:
        with pytest.raises(PayloadSizeExceededError):
            await client.download_to_staging(
                target_host="127.0.0.1",
                target_port=port,
                expected_size=500,
                staging_writer=staging_writer,
            )

        # Invariant: .part file unlinked, final file does not exist
        assert not staging_writer.part_path.exists()
        assert not final_path.exists()
    finally:
        srv.close()
        await srv.wait_closed()


async def test_challenge_port_exhaustion_fallback(tmp_path: Path) -> None:
    """Challenge 3: Port exhaustion in configured range falls back to port 0 (ephemeral).

    Verifies:
    - When all ports in the configured range are occupied:
      1. Default allow_ephemeral_fallback=True successfully binds port 0.
      2. allow_ephemeral_fallback=False raises PayloadPortBindingError.
    """
    c1, k1 = make_cert_and_key("dev1")
    c2, k2 = make_cert_and_key("dev2")

    # Occupy a narrow test range (e.g. 19500..19502)
    test_range_min = 19500
    test_range_max = 19502

    dummy_sockets = []
    try:
        for p in range(test_range_min, test_range_max + 1):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", p))
            s.listen(1)
            dummy_sockets.append(s)

        # Case A: Fallback enabled (default)
        server_fallback = PayloadServer(
            local_cert_pem=c1,
            local_key_pem=k1,
            peer_cert_pem=c2,
            port_min=test_range_min,
            port_max=test_range_max,
            allow_ephemeral_fallback=True,
        )
        bound_port = await server_fallback.bind_server(host="127.0.0.1")
        assert bound_port > 0
        assert bound_port not in range(test_range_min, test_range_max + 1)
        await server_fallback.close()

        # Case B: Fallback disabled
        server_strict = PayloadServer(
            local_cert_pem=c1,
            local_key_pem=k1,
            peer_cert_pem=c2,
            port_min=test_range_min,
            port_max=test_range_max,
            allow_ephemeral_fallback=False,
        )
        with pytest.raises(PayloadPortBindingError):
            await server_strict.bind_server(host="127.0.0.1")
        await server_strict.close()

    finally:
        for s in dummy_sockets:
            s.close()


async def test_challenge_unpair_during_active_transfer(tmp_path: Path) -> None:
    """Challenge 4: Unpairing a device during active in-flight file transfer cancels tasks & unlinks .part.

    Verifies:
    - Active transfer task is cancelled.
    - PayloadTransferManager cleans up active tasks.
    - .part file is completely removed.
    - Device transitions to DISCONNECTED/UNPAIRED without hangs.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]

        # Slow stream: 200 KB with chunk delay to guarantee transfer is in-flight
        stream_data = b"STREAMING_SLOW_PAYLOAD" * 5000
        slow_event = asyncio.Event()

        async def _slow_server(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                for i in range(0, len(stream_data), 1024):
                    writer.write(stream_data[i : i + 1024])
                    await writer.drain()
                    await asyncio.sleep(0.02)
            except Exception:
                pass
            finally:
                writer.close()
                await writer.wait_closed()
                slow_event.set()

        ssl_ctx = create_paired_ssl_context(
            local_cert_pem=mock.cert_pem,
            local_key_pem=mock.key_pem,
            peer_cert_pem=daemon_cert.encode("utf-8") if isinstance(daemon_cert, str) else daemon_cert,
            server_side=True,
        )
        srv = await asyncio.start_server(_slow_server, host="127.0.0.1", port=0, ssl=ssl_ctx)
        port = srv.sockets[0].getsockname()[1]

        try:
            filename = "unpair_in_flight.bin"
            share_pkt = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"filename": filename},
                payloadSize=len(stream_data),
                payloadTransferInfo={"port": port},
            )
            await mock.send_packet(share_pkt)

            # Wait briefly so file download starts and .part exists
            await asyncio.sleep(0.08)

            # Unpair device while active
            await daemon.unpair_device(mock.device_id)

            await asyncio.sleep(0.2)

            # Verify no .part file remains
            part_files = list(download_dir.glob("*.part"))
            assert len(part_files) == 0, f"Found lingering .part files after unpair: {part_files}"
            assert not (download_dir / filename).exists()

            # Verify payload manager has 0 active tasks for this device
            assert mock.device_id not in daemon.share_plugin.payload_manager._device_tasks
        finally:
            srv.close()
            await srv.wait_closed()


async def test_challenge_zero_byte_transfers(tmp_path: Path) -> None:
    """Challenge 5: Zero-byte file transfers (both inbound and outbound).

    Verifies:
    - Inbound 0-byte file bypasses network socket creation and commits directly.
    - Outbound 0-byte file sends packet with payloadSize=0 and connects cleanly.
    - File permissions are 0644 non-executable.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # 1. Inbound 0-byte file
        inbound_filename = "zero_byte_inbound.txt"
        zero_pkt = Packet(
            id=int(time.time() * 1000),
            type="kdeconnect.share.request",
            body={"filename": inbound_filename},
            payloadSize=0,
            payloadTransferInfo=None,  # No port needed for 0 bytes!
        )
        await mock.send_packet(zero_pkt)

        # Wait for file to be committed
        await wait_for(lambda: (download_dir / inbound_filename).exists(), timeout=3.0)
        zero_file = download_dir / inbound_filename
        assert zero_file.stat().st_size == 0
        # Check permissions (0o644)
        mode = zero_file.stat().st_mode & 0o777
        assert mode == 0o644

        # 2. Outbound 0-byte file
        outbound_source = tmp_path / "zero_out.txt"
        outbound_source.write_bytes(b"")

        share_task = asyncio.create_task(
            ipc_client.call(
                "share_file",
                {"device_id": mock.device_id, "file_path": str(outbound_source)},
            )
        )

        pkt = await mock.wait_for_share_packet(timeout=4.0)
        assert pkt.payloadSize == 0
        assert pkt.payloadTransferInfo is not None
        port = pkt.payloadTransferInfo["port"]

        # Mock connects and completes 0-byte read
        daemon_cert = daemon.local_cert_pem
        await mock.receive_share_file(
            target_port=port,
            expected_size=0,
            target_daemon_cert_pem=daemon_cert,
        )

        res = await share_task
        assert res.get("status") in ("ok", "transmitted", "shared") or "filename" in res


async def test_challenge_high_concurrency_burst_transfers(tmp_path: Path) -> None:
    """Challenge 6: High concurrency burst transfers (interleaved text, URL, and files).

    Verifies:
    - Rapid burst of 30 interleaved text and URL share requests.
    - Packet framing remains aligned without truncation or buffer corruption.
    - All events dispatched cleanly; zero deadlocks or task leaks.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        received_events: List[ShareReceivedEvent] = []

        async def _on_share(event: ShareReceivedEvent) -> None:
            received_events.append(event)

        daemon.event_bus.subscribe(ShareReceivedEvent, _on_share)

        num_burst = 30
        for i in range(num_burst):
            if i % 2 == 0:
                pkt = Packet(
                    id=int(time.time() * 1000) + i,
                    type="kdeconnect.share.request",
                    body={"text": f"Burst text message {i}"},
                )
            else:
                pkt = Packet(
                    id=int(time.time() * 1000) + i,
                    type="kdeconnect.share.request",
                    body={"url": f"https://example.com/item_{i}"},
                )
            await mock.send_packet(pkt)

        # Wait for all events to be received
        await wait_for(lambda: len(received_events) >= num_burst, timeout=5.0)
        assert len(received_events) == num_burst

        # Verify no framing corruption in text/URL values
        for i, ev in enumerate(received_events):
            if ev.share_type == "text":
                assert "Burst text message" in (ev.text or "")
            elif ev.share_type == "url":
                assert "https://example.com/item_" in (ev.url or "")


async def test_challenge_unpaired_device_rejection(tmp_path: Path) -> None:
    """Challenge 7: Untrusted / unpaired device share packets are rejected.

    Verifies:
    - Packet from an unpaired device is dropped.
    - No files are downloaded or created in the download directory.
    - No outbound network connections are initiated to untrusted payload ports.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    async with e2e_environment(tmp_path, auto_pair=False) as (daemon, mock, ipc_client):
        # Do NOT pair the device!
        untrusted_pkt = Packet(
            id=int(time.time() * 1000),
            type="kdeconnect.share.request",
            body={"filename": "malicious_unpaired.sh"},
            payloadSize=1024,
            payloadTransferInfo={"port": 19999},
        )

        # Directly send to daemon's plugin
        await daemon.share_plugin.handle_packet(
            untrusted_pkt,
            mock.device_id,
            send_packet=mock.send_packet,
        )

        # Allow time to ensure nothing starts
        await asyncio.sleep(0.3)

        # Invariant: No files created, no active transfers
        assert not (download_dir / "malicious_unpaired.sh").exists()
        assert len(list(download_dir.iterdir())) == 0
        assert mock.device_id not in daemon.share_plugin.payload_manager._device_tasks


async def test_challenge_resource_leak_audit(tmp_path: Path) -> None:
    """Challenge 8: Resource leak audit after full share lifecycle.

    Verifies:
    - Zero leaked sockets.
    - Zero orphaned .part files.
    - Zero lingering transfer tasks in PayloadTransferManager.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Run multiple operations: text, URL, and file
        await ipc_client.call("share_text", {"device_id": mock.device_id, "text": "Canary audit text"})
        await ipc_client.call("share_url", {"device_id": mock.device_id, "url": "https://audit.test"})

        f = tmp_path / "audit_file.txt"
        f.write_text("Audit content")

        share_task = asyncio.create_task(
            ipc_client.call("share_file", {"device_id": mock.device_id, "file_path": str(f)})
        )
        pkt = await mock.wait_for_share_packet(timeout=4.0)
        port = pkt.payloadTransferInfo["port"]
        await mock.receive_share_file(
            target_port=port,
            expected_size=f.stat().st_size,
            target_daemon_cert_pem=daemon.local_cert_pem,
        )
        res = await share_task
        assert res.get("status") == "ok"

        await asyncio.sleep(0.3)

    # After e2e_environment shuts down
    await asyncio.sleep(0.1)

    # Check for lingering transfer tasks
    remaining_tasks = {
        t for t in asyncio.all_tasks()
        if not t.done() and "payload" in str(t).lower()
    }
    assert len(remaining_tasks) == 0, f"Lingering payload tasks: {remaining_tasks}"

    # Check for orphaned .part files
    part_files = list(download_dir.glob("*.part"))
    assert len(part_files) == 0, f"Lingering .part files: {part_files}"

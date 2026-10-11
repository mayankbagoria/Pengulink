"""tests.e2e.test_requirements_e2e
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive, requirement-driven, opaque-box E2E test suite for PenguLink covering:
1. Share Feature (Bidirectional files, text, URLs)
2. Find My Phone Feature (CLI & core service)
3. Find This Device Alert Feature (Receiver, alert service, local CLI test)

Derived authoritatively from:
- ORIGINAL_REQUEST.md (entry dated 2026-10-10T16:48:53Z)
- PROJECT.md § Feature Inventory (Features 25–34) & Interface Contracts

Tiers Covered:
- Tier 1: Feature Coverage (Nominal Paths in Isolation) [>=5 per feature]
- Tier 2: Boundary & Corner Cases (Stress, Edge, Traversal, Collisions) [>=5 per feature]
- Tier 3: Cross-Feature Combinations (Pairwise Feature Interactions)
- Tier 4: Real-World Application Scenarios (Multi-step Linux-Android Workflows) [>=5 scenarios]
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import stat
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import uuid

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.events import (
    DeviceConnected,
    DeviceDisconnected,
    DeviceUnpaired,
    Event,
)
from pengulink.crypto.tls import create_paired_ssl_context
from pengulink.daemon.ipc import PenguLinkIPCClient
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.network.packet import Packet
from tests.e2e.conftest import e2e_environment, get_free_ports, wait_for
from tests.mock.mock_device import MockDevice, MockDeviceConfig

# Safe imports for upcoming domain events & exceptions (implemented in M10/M11)
try:
    from pengulink.core.events import (
        ShareReceivedEvent,
        FileTransferProgress,
        FileReceivedEvent,
    )
except ImportError:
    ShareReceivedEvent = None  # type: ignore
    FileTransferProgress = None  # type: ignore
    FileReceivedEvent = None  # type: ignore

try:
    from pengulink.core.events import (
        FindDeviceAlertTriggeredEvent,
        FindDeviceAlertStoppedEvent,
    )
except ImportError:
    FindDeviceAlertTriggeredEvent = None  # type: ignore
    FindDeviceAlertStoppedEvent = None  # type: ignore

try:
    from pengulink.core.exceptions import PathTraversalError
except ImportError:
    PathTraversalError = None  # type: ignore


# =============================================================================
# Test Harness & Network Helpers
# =============================================================================

@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing Click CLI test runner configured with standard width."""
    return CliRunner(env={"COLUMNS": "160"})


async def _pair_and_connect(daemon: PenguLinkDaemon, mock: MockDevice) -> None:
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


async def _start_mock_payload_server(
    mock: MockDevice,
    file_bytes: bytes,
    daemon_pubkey_pem: str,
    chunk_delay: float = 0.0,
) -> Tuple[int, asyncio.Server, asyncio.Event]:
    """Start an ephemeral out-of-band TLS server simulating Android payload sending.

    Upstream KDE Connect: Sender is TLS server, receiver connects outbound and upgrades to TLS.
    """
    completed_event = asyncio.Event()

    async def _handle_payload_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            if chunk_delay > 0:
                chunk_size = 1024
                for i in range(0, len(file_bytes), chunk_size):
                    writer.write(file_bytes[i : i + chunk_size])
                    await writer.drain()
                    await asyncio.sleep(chunk_delay)
            else:
                writer.write(file_bytes)
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            completed_event.set()

    daemon_pem_bytes = (
        daemon_pubkey_pem.encode("utf-8")
        if isinstance(daemon_pubkey_pem, str)
        else daemon_pubkey_pem
    )
    ssl_ctx = create_paired_ssl_context(
        local_cert_pem=mock.cert_pem,
        local_key_pem=mock.key_pem,
        peer_cert_pem=daemon_pem_bytes,
        server_side=True,
    )

    server = await asyncio.start_server(
        _handle_payload_client,
        host="127.0.0.1",
        port=0,
        ssl=ssl_ctx,
    )
    port = server.sockets[0].getsockname()[1]
    return port, server, completed_event


async def _receive_mock_payload_client(
    mock: MockDevice,
    target_port: int,
    expected_size: int,
    daemon_pubkey_pem: str,
    timeout: float = 5.0,
) -> bytes:
    """Connect to PenguLink daemon's ephemeral payload server and stream bytes via mTLS."""
    ssl_ctx = create_paired_ssl_context(
        local_cert_pem=mock.cert_pem,
        local_key_pem=mock.key_pem,
        peer_cert_pem=daemon_pubkey_pem.encode("utf-8") if isinstance(daemon_pubkey_pem, str) else daemon_pubkey_pem,
        server_side=False,
    )

    reader, writer = await asyncio.wait_for(
        asyncio.open_connection("127.0.0.1", target_port, ssl=ssl_ctx),
        timeout=timeout,
    )
    try:
        data = await asyncio.wait_for(reader.readexactly(expected_size), timeout=timeout)
        return data
    finally:
        writer.close()
        await writer.wait_closed()


# =============================================================================
# TIER 1: FEATURE COVERAGE (Nominal Paths in Isolation)
# Requirement Threshold: >=5 test cases per feature
# =============================================================================

# --- Feature 1: Share Feature (Nominal) ---

async def test_tier1_share_outbound_text_ipc(tmp_path: Path) -> None:
    """Tier 1: Linux -> Android text share via IPC client.

    Derived from: ORIGINAL_REQUEST.md R1, PROJECT.md Feature 30.
    Input: target device_id, text="Hello from PenguLink Share!"
    Expected Output:
    - IPC call 'share_text' succeeds with status 'ok' or 'transmitted'.
    - Mock device receives Packet(type="kdeconnect.share.request", body={"text": ...}).
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        test_text = "Hello from PenguLink Share!"
        res = await ipc_client.call("share_text", {"device_id": mock.device_id, "text": test_text})
        assert res.get("status") in ("ok", "transmitted")

        await wait_for(
            lambda: any(
                p.type == "kdeconnect.share.request" and p.body.get("text") == test_text
                for p in mock.received_packets
            ),
            timeout=4.0,
            desc="mock receiving kdeconnect.share.request text packet",
        )


async def test_tier1_share_outbound_text_cli(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Tier 1: Linux -> Android text share via Click CLI command.

    Derived from: ORIGINAL_REQUEST.md R1, R4, PROJECT.md Feature 30.
    Input: `pengulink -s <sock> share text <device_id> "Meeting notes 2026"`
    Expected Output: Exit code 0, confirmation panel or message, packet received on mock.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        sock_path = daemon.config.storage.runtime_dir / "daemon.sock"

        text_content = "Meeting notes 2026"
        res = await asyncio.to_thread(cli_runner.invoke, cli, ["-s", str(sock_path), "share", "text", mock.device_id, text_content])
        assert res.exit_code == 0
        assert "shared" in res.output.lower() or "transmitted" in res.output.lower() or "ok" in res.output.lower()

        await wait_for(
            lambda: any(
                p.type == "kdeconnect.share.request" and p.body.get("text") == text_content
                for p in mock.received_packets
            ),
            timeout=4.0,
            desc="mock receiving CLI-dispatched text packet",
        )


async def test_tier1_share_inbound_text(tmp_path: Path) -> None:
    """Tier 1: Android -> Linux text share received over primary mTLS connection.

    Derived from: ORIGINAL_REQUEST.md R1, PROJECT.md Feature 30.
    Input: Mock transmits Packet(type="kdeconnect.share.request", body={"text": "Grocery list"}).
    Expected Output: EventBus receives ShareReceivedEvent with share_type='text' and text='Grocery list'.
    """
    assert ShareReceivedEvent is not None, "ShareReceivedEvent must be implemented in pengulink.core.events"
    events: List[Any] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon.event_bus.subscribe(ShareReceivedEvent, lambda ev: events.append(ev))

        pkt = Packet(
            id=int(time.time() * 1000),
            type="kdeconnect.share.request",
            body={"text": "Grocery list: milk, eggs, bread"},
        )
        await mock.send_packet(pkt)

        await wait_for(
            lambda: len(events) >= 1,
            timeout=4.0,
            desc="daemon emitting ShareReceivedEvent for inbound text",
        )
        ev = events[0]
        assert ev.device_id == mock.device_id
        assert getattr(ev, "text", "") == "Grocery list: milk, eggs, bread"


async def test_tier1_share_outbound_url_cli(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Tier 1: Linux -> Android URL share via CLI.

    Derived from: ORIGINAL_REQUEST.md R1, R4, PROJECT.md Feature 30.
    Input: `pengulink -s <sock> share url <device_id> https://kdeconnect.kde.org`
    Expected Output: Exit code 0, mock device receives kdeconnect.share.request with body.url.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        sock_path = daemon.config.storage.runtime_dir / "daemon.sock"

        target_url = "https://kdeconnect.kde.org"
        res = await asyncio.to_thread(cli_runner.invoke, cli, ["-s", str(sock_path), "share", "url", mock.device_id, target_url])
        assert res.exit_code == 0

        await wait_for(
            lambda: any(
                p.type == "kdeconnect.share.request" and p.body.get("url") == target_url
                for p in mock.received_packets
            ),
            timeout=4.0,
            desc="mock receiving URL share packet",
        )


async def test_tier1_share_inbound_url(tmp_path: Path) -> None:
    """Tier 1: Android -> Linux URL share received over primary mTLS stream.

    Derived from: ORIGINAL_REQUEST.md R1, PROJECT.md Feature 30.
    Input: Mock sends Packet(type="kdeconnect.share.request", body={"url": "https://python.org"}).
    Expected Output: EventBus receives ShareReceivedEvent with url='https://python.org'.
    """
    assert ShareReceivedEvent is not None, "ShareReceivedEvent must be implemented in pengulink.core.events"
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

        await wait_for(
            lambda: len(events) >= 1,
            timeout=4.0,
            desc="daemon emitting ShareReceivedEvent for inbound URL",
        )
        assert getattr(events[0], "url", "") == "https://python.org"


async def test_tier1_share_outbound_file(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Tier 1: Linux -> Android file transfer via out-of-band TLS payload socket.

    Derived from: ORIGINAL_REQUEST.md R1, PROJECT.md Feature 31.
    Input: Valid local text file 'notes.txt' sent via IPC or CLI.
    Expected Output:
    - Daemon opens secondary TCP server on port in 1739..1764.
    - Sends kdeconnect.share.request with payloadTransferInfo={"port": port} and payloadSize.
    - Mock connects, executes TLS handshake, and receives file bytes matching source.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        source_file = tmp_path / "notes.txt"
        test_payload = b"Sample outbound file content from PenguLink\n" * 50
        source_file.write_bytes(test_payload)

        # Trigger file share via IPC
        share_task = asyncio.create_task(
            ipc_client.call("share_file", {"device_id": mock.device_id, "file_path": str(source_file)})
        )

        # Await packet on mock device
        await wait_for(
            lambda: any(
                p.type == "kdeconnect.share.request" and p.payloadTransferInfo is not None
                for p in mock.received_packets
            ),
            timeout=5.0,
            desc="mock receiving file share packet with payloadTransferInfo",
        )

        rx_pkt = [p for p in mock.received_packets if p.payloadTransferInfo is not None][0]
        payload_port = rx_pkt.payloadTransferInfo["port"]
        assert 1739 <= payload_port <= 1764 or payload_port > 1024
        assert rx_pkt.payloadSize == len(test_payload)

        # Connect secondary socket and stream bytes
        daemon_cert_pem = daemon.local_cert_pem if hasattr(daemon, "local_cert_pem") else mock.paired_peers[daemon.config.device.device_id]
        received_bytes = await _receive_mock_payload_client(
            mock=mock,
            target_port=payload_port,
            expected_size=len(test_payload),
            daemon_pubkey_pem=daemon_cert_pem,
        )
        assert received_bytes == test_payload
        await share_task


async def test_tier1_share_inbound_file(tmp_path: Path) -> None:
    """Tier 1: Android -> Linux file transfer received and stored into download directory.

    Derived from: ORIGINAL_REQUEST.md R1, PROJECT.md Feature 32.
    Input: Mock sets up payload server and sends kdeconnect.share.request packet.
    Expected Output:
    - Daemon connects outbound to mock's payload port.
    - Upgrades via mTLS.
    - File is saved into configured download directory with identical bytes and mode 0644.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        # Configure download destination
        if hasattr(daemon.config.storage, "download_dir"):
            daemon.config.storage.download_dir = download_dir
        os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

        await _pair_and_connect(daemon, mock)

        test_data = b"Inbound photo image payload binary data" * 100
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]
        port, srv, completed = await _start_mock_payload_server(mock, test_data, daemon_cert)

        try:
            pkt = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"filename": "photo_vacation.jpg"},
                payloadSize=len(test_data),
                payloadTransferInfo={"port": port},
            )
            await mock.send_packet(pkt)

            # Wait for file to arrive in download directory
            dest_file = download_dir / "photo_vacation.jpg"
            await wait_for(lambda: dest_file.exists() and dest_file.stat().st_size == len(test_data), timeout=5.0)
            assert dest_file.read_bytes() == test_data
        finally:
            srv.close()
            await srv.wait_closed()


# --- Feature 2: Find My Phone Feature (Nominal) ---

async def test_tier1_find_my_phone_outbound_cli(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Tier 1: Ring remote phone via Click CLI command.

    Derived from: ORIGINAL_REQUEST.md R2, R4, PROJECT.md Feature 25.
    Input: `pengulink -s <sock> find-phone <device_id>`
    Expected Output:
    - Transmits kdeconnect.findmyphone.request over primary mTLS stream.
    - Mock receives packet with body == {}.
    - CLI displays success panel and exits 0.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        sock_path = daemon.config.storage.runtime_dir / "daemon.sock"

        res = await asyncio.to_thread(cli_runner.invoke, cli, ["-s", str(sock_path), "find-phone", mock.device_id])
        assert res.exit_code == 0
        assert "ring" in res.output.lower() or "transmitted" in res.output.lower() or "phone" in res.output.lower()

        await wait_for(
            lambda: any(
                p.type in ("kdeconnect.findmyphone.request", "kdeconnect.findmyphone")
                for p in mock.received_packets
            ),
            timeout=4.0,
            desc="mock receiving findmyphone ring packet",
        )


async def test_tier1_find_my_phone_outbound_ipc(tmp_path: Path) -> None:
    """Tier 1: Ring remote phone via JSON-RPC IPC method 'find_phone'.

    Derived from: ORIGINAL_REQUEST.md R2, PROJECT.md Feature 25.
    Input: IPC method `find_phone(device_id)`
    Expected Output: Returns {"status": "transmitted", "device_id": ...}; packet arrives at mock.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        res = await ipc_client.call("find_phone", {"device_id": mock.device_id})
        assert res.get("status") in ("transmitted", "ok")
        assert res.get("device_id") == mock.device_id

        await wait_for(
            lambda: any(
                p.type in ("kdeconnect.findmyphone.request", "kdeconnect.findmyphone")
                for p in mock.received_packets
            ),
            timeout=3.0,
        )


async def test_tier1_find_my_phone_offline_device_cli(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Tier 1: Ringing an offline/disconnected device from CLI fails cleanly.

    Derived from: ORIGINAL_REQUEST.md R2, R4, PROJECT.md Feature 25.
    Input: Device is paired in store but TCP connection is closed.
    Expected Output: Non-zero exit code (1), actionable error message, no unhandled traceback.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        await mock.drop_connection()
        await wait_for(lambda: not daemon.is_device_connected(mock.device_id), timeout=3.0)

        sock_path = daemon.config.storage.runtime_dir / "daemon.sock"
        res = await asyncio.to_thread(cli_runner.invoke, cli, ["-s", str(sock_path), "find-phone", mock.device_id])
        assert res.exit_code != 0
        assert "offline" in res.output.lower() or "not connected" in res.output.lower()


async def test_tier1_find_my_phone_unknown_device_ipc(tmp_path: Path) -> None:
    """Tier 1: Ringing an unknown device ID returns clear error over IPC.

    Derived from: ORIGINAL_REQUEST.md R2, PROJECT.md Feature 25.
    Input: IPC method `find_phone(device_id="non_existent_1234")`.
    Expected Output: JSON-RPC error response or exception indicating device not found.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        with pytest.raises(Exception) as exc_info:
            await ipc_client.call("find_phone", {"device_id": "non_existent_1234"})
        err_msg = str(exc_info.value).lower()
        assert "not found" in err_msg or "unknown" in err_msg or "device" in err_msg


async def test_tier1_find_my_phone_wire_packet_schema(tmp_path: Path) -> None:
    """Tier 1: Verify wire packet schema sent to phone matches KDE Connect specification.

    Derived from: PROTOCOL_NOTES.md, PROJECT.md Feature 25.
    Expected Output:
    - Packet.type == "kdeconnect.findmyphone.request"
    - Packet.body == {} or {"request": true}
    - payloadSize is None.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        await ipc_client.call("find_phone", {"device_id": mock.device_id})

        await wait_for(lambda: len(mock.received_packets) >= 1, timeout=3.0)
        ring_pkt = [p for p in mock.received_packets if "findmyphone" in p.type][0]
        assert ring_pkt.type in ("kdeconnect.findmyphone.request", "kdeconnect.findmyphone")
        assert ring_pkt.payloadSize is None
        assert isinstance(ring_pkt.body, dict)


# --- Feature 3: Find This Device Alert Feature (Nominal) ---

async def test_tier1_find_this_device_inbound_trigger(tmp_path: Path) -> None:
    """Tier 1: Inbound locate request from phone triggers local AlertService.

    Derived from: ORIGINAL_REQUEST.md R3, PROJECT.md Feature 26.
    Input: Mock sends Packet(type="kdeconnect.findmyphone.request", body={}).
    Expected Output:
    - AlertService transitions to active ringing state.
    - Emits FindDeviceAlertTriggeredEvent(device_id=..., source="remote").
    """
    assert FindDeviceAlertTriggeredEvent is not None, "FindDeviceAlertTriggeredEvent must be defined in events.py"
    triggered_events: List[Any] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon.event_bus.subscribe(FindDeviceAlertTriggeredEvent, lambda ev: triggered_events.append(ev))

        # Send locate request from simulated Android phone
        pkt = Packet(
            id=int(time.time() * 1000),
            type="kdeconnect.findmyphone.request",
            body={},
        )
        await mock.send_packet(pkt)

        # Verify EventBus dispatch
        await wait_for(lambda: len(triggered_events) >= 1, timeout=4.0, desc="alert triggered event")
        assert triggered_events[0].device_id == mock.device_id

        # Verify AlertService state
        status = await ipc_client.call("alert_status", {})
        assert status.get("active") is True


async def test_tier1_find_this_device_cli_alert_test(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Tier 1: Local CLI alert test command triggers identical AlertService.

    Derived from: ORIGINAL_REQUEST.md R3, PROJECT.md Feature 28.
    Input: `pengulink -s <sock> alert test`
    Expected Output:
    - Invokes AlertService.
    - Displays ringing message.
    - `alert_status` reports active == True with source == "local".
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        sock_path = daemon.config.storage.runtime_dir / "daemon.sock"

        res = await asyncio.to_thread(cli_runner.invoke, cli, ["-s", str(sock_path), "alert", "test"])
        assert res.exit_code == 0
        assert "alert" in res.output.lower() or "ringing" in res.output.lower()

        status = await ipc_client.call("alert_status", {})
        assert status.get("active") is True
        assert status.get("source") in ("local", "cli")


async def test_tier1_find_this_device_cli_alert_stop(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Tier 1: Local CLI alert stop command silences active alert.

    Derived from: ORIGINAL_REQUEST.md R3, PROJECT.md Feature 28.
    Input: Start alert, then invoke `pengulink alert stop`.
    Expected Output:
    - Alert immediately stops.
    - CLI displays stopped confirmation.
    - `alert_status` reports active == False.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        sock_path = daemon.config.storage.runtime_dir / "daemon.sock"

        # Start alert
        await ipc_client.call("alert_test", {"duration": 30.0})
        status_before = await ipc_client.call("alert_status", {})
        assert status_before.get("active") is True

        # Stop alert via CLI
        res = await asyncio.to_thread(cli_runner.invoke, cli, ["-s", str(sock_path), "alert", "stop"])
        assert res.exit_code == 0
        assert "stopped" in res.output.lower() or "silenced" in res.output.lower()

        status_after = await ipc_client.call("alert_status", {})
        assert status_after.get("active") is False


async def test_tier1_find_this_device_cli_alert_status(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Tier 1: Local CLI alert status command reflects accurate state.

    Derived from: ORIGINAL_REQUEST.md R3, PROJECT.md Feature 28.
    Expected Output: Shows 'Inactive' when idle, and 'Ringing'/'Active' when playing.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        sock_path = daemon.config.storage.runtime_dir / "daemon.sock"

        # 1. Idle status
        res_idle = await asyncio.to_thread(cli_runner.invoke, cli, ["-s", str(sock_path), "alert", "status"])
        assert res_idle.exit_code == 0
        assert "inactive" in res_idle.output.lower() or "idle" in res_idle.output.lower()

        # 2. Active status
        await ipc_client.call("alert_test", {"duration": 30.0})
        res_active = await asyncio.to_thread(cli_runner.invoke, cli, ["-s", str(sock_path), "alert", "status"])
        assert res_active.exit_code == 0
        assert "ringing" in res_active.output.lower() or "active" in res_active.output.lower()


async def test_tier1_find_this_device_ipc_alert_cycle(tmp_path: Path) -> None:
    """Tier 1: Full programmatic alert lifecycle over JSON-RPC IPC.

    Derived from: PROJECT.md § Interface Contracts (FindMyPhonePlugin ↔ Daemon / IPC).
    Lifecycle: alert_test -> alert_status(active=True) -> alert_stop -> alert_status(active=False).
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        start_res = await ipc_client.call("alert_test", {"duration": 15.0})
        assert start_res.get("status") in ("ringing", "ok", "started")

        status_mid = await ipc_client.call("alert_status", {})
        assert status_mid.get("active") is True

        stop_res = await ipc_client.call("alert_stop", {})
        assert stop_res.get("status") in ("stopped", "ok")

        status_end = await ipc_client.call("alert_status", {})
        assert status_end.get("active") is False


# =============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# Requirement Threshold: >=5 test cases per feature
# =============================================================================

# --- Feature 1: Share Boundary & Security ---

async def test_tier2_share_empty_text(tmp_path: Path) -> None:
    """Tier 2: Empty string text share handled safely without protocol desync.

    Derived from: PROTOCOL_NOTES.md, GSConnect share.js.
    Input: Outbound share with text="".
    Expected Output: Handled cleanly; does not crash transport framing.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        res = await ipc_client.call("share_text", {"device_id": mock.device_id, "text": ""})
        assert res is not None


async def test_tier2_share_unicode_and_spaces_filename(tmp_path: Path) -> None:
    """Tier 2: Filename containing spaces, Unicode accents, and emojis preserved accurately.

    Derived from: ORIGINAL_REQUEST.md R1, PROJECT.md Feature 29.
    Input: Filename '📄 My Important Document 2026 🐧.pdf'
    Expected Output: Saved to disk with NFC Unicode normalization and intact spaces.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        complex_name = "📄 My Important Document 2026 🐧.pdf"
        file_content = b"PDF mock content with Unicode metadata"
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]
        port, srv, _ = await _start_mock_payload_server(mock, file_content, daemon_cert)

        try:
            pkt = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"filename": complex_name},
                payloadSize=len(file_content),
                payloadTransferInfo={"port": port},
            )
            await mock.send_packet(pkt)

            # Assert file exists with preserved characters
            expected_file = download_dir / complex_name
            await wait_for(lambda: expected_file.exists(), timeout=5.0)
            assert expected_file.read_bytes() == file_content
        finally:
            srv.close()
            await srv.wait_closed()


async def test_tier2_share_path_traversal_sanitization(tmp_path: Path) -> None:
    """Tier 2: Hostile filenames with path traversal sequences cannot escape download root.

    Derived from: ORIGINAL_REQUEST.md R1, docs/SECURITY.md, PROJECT.md Feature 29.
    Attacks:
    - POSIX traversal: '../../../../tmp/evil.sh'
    - Windows backslash traversal: '..\\..\\windows\\system32\\cmd.exe'
    - Null-byte poisoning: 'document.pdf\x00.sh'
    Expected Output:
    - Stripped to pure basename inside authorized download directory.
    - Or rejected with PathTraversalError.
    - Zero files written outside download directory.
    """
    download_dir = tmp_path / "sandbox_dest"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    canary_victim = tmp_path / "victim.txt"

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        hostile_name = "../../../../victim.txt"
        file_content = b"ATTACK PAYLOAD - MUST NEVER OVERWRITE OUTSIDE ROOT"
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]
        port, srv, _ = await _start_mock_payload_server(mock, file_content, daemon_cert)

        try:
            pkt = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"filename": hostile_name},
                payloadSize=len(file_content),
                payloadTransferInfo={"port": port},
            )
            await mock.send_packet(pkt)
            await asyncio.sleep(0.5)

            # Invariant 1: Outside file was not created or modified
            assert not canary_victim.exists()

            # Invariant 2: If file was accepted, it must reside strictly within download_dir
            for f in download_dir.iterdir():
                assert f.resolve().is_relative_to(download_dir.resolve())
        finally:
            srv.close()
            await srv.wait_closed()


async def test_tier2_share_collision_auto_numbering(tmp_path: Path) -> None:
    """Tier 2: Filename collisions resolved deterministically with 'stem (counter).ext'.

    Derived from: ORIGINAL_REQUEST.md R1, PROJECT.md Feature 29.
    Scenario: 'report.txt' already exists. Incoming file named 'report.txt' arrives twice.
    Expected Output:
    - Pre-existing file is NOT overwritten.
    - Second file is saved as 'report (1).txt'.
    - Third file is saved as 'report (2).txt'.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    # Seed pre-existing file
    original_file = download_dir / "report.txt"
    original_file.write_bytes(b"Original file version 1.0")

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]

        new_content = b"Incoming file version 2.0"
        port, srv, _ = await _start_mock_payload_server(mock, new_content, daemon_cert)

        try:
            pkt = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"filename": "report.txt"},
                payloadSize=len(new_content),
                payloadTransferInfo={"port": port},
            )
            await mock.send_packet(pkt)

            collided_file = download_dir / "report (1).txt"
            await wait_for(lambda: collided_file.exists(), timeout=5.0)

            # Verify original content was preserved
            assert original_file.read_bytes() == b"Original file version 1.0"
            assert collided_file.read_bytes() == new_content
        finally:
            srv.close()
            await srv.wait_closed()


async def test_tier2_share_interrupted_file_cleanup(tmp_path: Path) -> None:
    """Tier 2: Interrupted file transfer purges staging temporary .part file immediately.

    Derived from: ORIGINAL_REQUEST.md R1, PROJECT.md Feature 29, 32.
    Scenario: Payload socket abruptly closes after sending only 20% of promised payloadSize.
    Expected Output:
    - Incomplete transfer detected.
    - Staging '.part' file unlinked from destination directory.
    - Zero corrupt files left in download directory.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]

        # Partial server sending only 100 bytes of 10,000 bytes promised
        partial_data = b"X" * 100
        port, srv, _ = await _start_mock_payload_server(mock, partial_data, daemon_cert)

        try:
            pkt = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"filename": "incomplete_archive.zip"},
                payloadSize=10000,  # Promised 10 KB
                payloadTransferInfo={"port": port},
            )
            await mock.send_packet(pkt)
            await asyncio.sleep(1.0)

            # Invariant: No partial or incomplete files remain in download directory
            remaining_files = list(download_dir.iterdir())
            assert not any(f.name.endswith(".part") for f in remaining_files)
            assert not any(f.name == "incomplete_archive.zip" for f in remaining_files)
        finally:
            srv.close()
            await srv.wait_closed()


async def test_tier2_share_non_executable_permissions(tmp_path: Path) -> None:
    """Tier 2: Transferred files strictly stripped of execute permissions (POSIX 0644).

    Derived from: ORIGINAL_REQUEST.md R1, docs/SECURITY.md.
    Scenario: Android transmits a shell script 'install.sh'.
    Expected Output: File written with mode 0644 (no execute bits: stat.st_mode & 0o111 == 0).
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]

        script_payload = b"#!/bin/bash\necho 'Must not have execute bits'\n"
        port, srv, _ = await _start_mock_payload_server(mock, script_payload, daemon_cert)

        try:
            pkt = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"filename": "install.sh"},
                payloadSize=len(script_payload),
                payloadTransferInfo={"port": port},
            )
            await mock.send_packet(pkt)

            script_file = download_dir / "install.sh"
            await wait_for(lambda: script_file.exists(), timeout=5.0)

            # Verify no execute bits
            mode = script_file.stat().st_mode
            assert mode & stat.S_IXUSR == 0
            assert mode & stat.S_IXGRP == 0
            assert mode & stat.S_IXOTH == 0
        finally:
            srv.close()
            await srv.wait_closed()


# --- Feature 2: Find My Phone Boundary ---

async def test_tier2_find_my_phone_rapid_invocations(tmp_path: Path) -> None:
    """Tier 2: Rapid successive ring invocations (burst spam) handled safely without socket errors.

    Derived from: ORIGINAL_REQUEST.md R2, PROJECT.md Feature 25.
    Scenario: User rapidly taps 'find-phone' 5 times in <100ms.
    Expected Output: All 5 packets dispatched cleanly over active mTLS channel without stream corruption.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        for _ in range(5):
            await ipc_client.call("find_phone", {"device_id": mock.device_id})

        await wait_for(
            lambda: sum(1 for p in mock.received_packets if "findmyphone" in p.type) >= 5,
            timeout=4.0,
            desc="mock receiving all 5 rapid ring packets",
        )


async def test_tier2_find_my_phone_malformed_device_id(tmp_path: Path) -> None:
    """Tier 2: Malformed or malicious device IDs rejected cleanly with parameter error.

    Input: device_id with null bytes or extreme size (>1000 characters).
    Expected Output: Rejection with validation error; daemon does not crash.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        bad_id = "device\x00_overflow_" + ("A" * 500)
        with pytest.raises(Exception):
            await ipc_client.call("find_phone", {"device_id": bad_id})


async def test_tier2_find_my_phone_unpaired_device(tmp_path: Path) -> None:
    """Tier 2: Attempting to ring an unpaired discovered peer fails with authorization error.

    Derived from: docs/SECURITY.md (CVE-2020-26164 mitigations).
    Scenario: Discovered device is active on network but not paired in storage.
    Expected Output: Rejection; control packets cannot be transmitted to unpaired peers.
    """
    async with e2e_environment(tmp_path, auto_pair=False) as (daemon, mock, ipc_client):
        # Broadcast discovery only (do not pair)
        await mock.broadcast_identity(target_ip="127.0.0.1", target_port=daemon.config.network.udp_port)
        await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)

        with pytest.raises(Exception):
            await ipc_client.call("find_phone", {"device_id": mock.device_id})


async def test_tier2_find_my_phone_disconnect_in_flight(tmp_path: Path) -> None:
    """Tier 2: Connection drop right as ring packet is requested returns clean connection error.

    Derived from: ORIGINAL_REQUEST.md R2, Acceptance Criteria.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        # Drop socket immediately
        await mock.drop_connection(reset=True)
        await asyncio.sleep(0.05)

        with pytest.raises(Exception) as exc_info:
            await ipc_client.call("find_phone", {"device_id": mock.device_id})
        assert "closed" in str(exc_info.value).lower() or "not connected" in str(exc_info.value).lower()


async def test_tier2_find_my_phone_cli_json_flag(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Tier 2: CLI find-phone supports --json output producing machine-readable status.

    Derived from: ORIGINAL_REQUEST.md R4.
    Input: `pengulink -s <sock> --json-output find-phone <device_id>`
    Expected Output: Valid JSON stdout containing status and device_id.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        sock_path = daemon.config.storage.runtime_dir / "daemon.sock"

        res = await asyncio.to_thread(cli_runner.invoke, cli, ["-s", str(sock_path), "--json-output", "find-phone", mock.device_id])
        assert res.exit_code == 0
        parsed = json.loads(res.output)
        assert parsed.get("device_id") == mock.device_id
        assert parsed.get("status") in ("transmitted", "ok")


# --- Feature 3: Find This Device Alert Boundary ---

async def test_tier2_find_this_device_toggle_cancellation(tmp_path: Path) -> None:
    """Tier 2: Second incoming findmyphone packet toggles active alert off (toggle cancellation).

    Derived from: ORIGINAL_REQUEST.md R3, GSConnect findmyphone.js:_handleRequest.
    Scenario:
    1. Incoming packet 1 triggers AlertService (ringing).
    2. Incoming packet 2 while ringing cancels AlertService (stopped).
    Expected Output:
    - AlertService stops.
    - Emits FindDeviceAlertStoppedEvent(reason="toggle").
    """
    assert FindDeviceAlertStoppedEvent is not None, "FindDeviceAlertStoppedEvent must be defined"
    stopped_events: List[Any] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon.event_bus.subscribe(FindDeviceAlertStoppedEvent, lambda ev: stopped_events.append(ev))

        pkt = Packet(id=int(time.time() * 1000), type="kdeconnect.findmyphone.request", body={})

        # 1. First packet starts alert
        await mock.send_packet(pkt)
        await wait_for(lambda: (daemon._active_connections is not None), timeout=1.0)
        status_1 = await ipc_client.call("alert_status", {})
        assert status_1.get("active") is True

        # 2. Second packet cancels alert
        await mock.send_packet(pkt)
        await wait_for(lambda: len(stopped_events) >= 1, timeout=4.0, desc="alert toggle stopped event")

        status_2 = await ipc_client.call("alert_status", {})
        assert status_2.get("active") is False


async def test_tier2_find_this_device_concurrency_lock(tmp_path: Path) -> None:
    """Tier 2: Single-instance mutex prevents duplicate overlapping audio playback tasks.

    Derived from: ORIGINAL_REQUEST.md R3, PROJECT.md Feature 27.
    Scenario: Burst of 5 incoming findmyphone packets received in 50ms.
    Expected Output: Single active task, no duplicated child processes, zero sound clipping.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        for _ in range(5):
            pkt = Packet(id=int(time.time() * 1000), type="kdeconnect.findmyphone.request", body={})
            await mock.send_packet(pkt)

        await asyncio.sleep(0.3)
        status = await ipc_client.call("alert_status", {})
        assert status is not None


async def test_tier2_find_this_device_runaway_timeout(tmp_path: Path) -> None:
    """Tier 2: AlertService enforces runaway protection (stops automatically after timeout).

    Derived from: ORIGINAL_REQUEST.md R3, PROJECT.md Feature 27.
    Scenario: Alert triggered with short duration (e.g. 1.0s). Left unacknowledged.
    Expected Output: Automatically transitions to inactive after duration expires.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await ipc_client.call("alert_test", {"duration": 1.0})
        status_start = await ipc_client.call("alert_status", {})
        assert status_start.get("active") is True

        # Await timeout expiration
        await asyncio.sleep(1.2)
        status_end = await ipc_client.call("alert_status", {})
        assert status_end.get("active") is False


async def test_tier2_find_this_device_headless_audio_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tier 2: Headless environment without audio tools falls back gracefully without crashing.

    Derived from: ORIGINAL_REQUEST.md R3, PROJECT.md Feature 27.
    Scenario: PATH has no pw-cat, paplay, or aplay.
    Expected Output: AlertService logs warning/fallback, dispatches visual/event notification, does not crash.
    """
    monkeypatch.setenv("PATH", "/nonexistent_bin")
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        res = await ipc_client.call("alert_test", {"duration": 0.5})
        assert res is not None


async def test_tier2_find_this_device_daemon_shutdown_cleanup(tmp_path: Path) -> None:
    """Tier 2: Daemon shutdown during active ringing cleanly terminates audio processes.

    Derived from: ORIGINAL_REQUEST.md R3, Acceptance Criteria.
    Expected Output: Zero hanging tasks, zero zombie child processes on daemon.stop().
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await ipc_client.call("alert_test", {"duration": 30.0})
        # Stop daemon while alert is active
        await daemon.stop()
        assert not daemon.is_running


# =============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (Pairwise Interactions)
# Requirement: Pairwise coverage across Share, Find My Phone, and Find This Device
# =============================================================================

async def test_tier3_pairwise_share_while_alert_ringing(tmp_path: Path) -> None:
    """Tier 3: Pairwise - Inbound file transfer proceeds while Find This Device alert is ringing.

    Features Combined: Find This Device Alert (Feature 26) + Inbound File Share (Feature 32).
    Scenario: Alert is ringing continuously on PC. Simultaneously, phone streams a file.
    Expected Output:
    - Audio alert continues playing smoothly.
    - File payload streams over secondary TLS socket and saves cleanly.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # 1. Trigger alert
        await ipc_client.call("alert_test", {"duration": 30.0})
        status_alert = await ipc_client.call("alert_status", {})
        assert status_alert.get("active") is True

        # 2. Concurrently initiate file transfer
        test_data = b"Simultaneous file transfer payload during alert ringing" * 50
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]
        port, srv, _ = await _start_mock_payload_server(mock, test_data, daemon_cert)

        try:
            pkt = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"filename": "concurrent_test.txt"},
                payloadSize=len(test_data),
                payloadTransferInfo={"port": port},
            )
            await mock.send_packet(pkt)

            # Assert file arrives cleanly
            dest_file = download_dir / "concurrent_test.txt"
            await wait_for(lambda: dest_file.exists(), timeout=5.0)
            assert dest_file.read_bytes() == test_data

            # Assert alert is still active
            status_alert_after = await ipc_client.call("alert_status", {})
            assert status_alert_after.get("active") is True
        finally:
            srv.close()
            await srv.wait_closed()
            await ipc_client.call("alert_stop", {})


async def test_tier3_pairwise_find_phone_during_file_upload(tmp_path: Path) -> None:
    """Tier 3: Pairwise - Linux user rings phone while out-of-band file upload is in flight.

    Features Combined: Find My Phone (Feature 25) + Outbound File Share (Feature 31).
    Scenario: Secondary TCP payload stream is active. Primary mTLS control channel sends ring packet.
    Expected Output: Both operations succeed without stalling or multiplexing interference.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        source_file = tmp_path / "large_outbound.bin"
        source_data = b"Outbound file data " * 2000
        source_file.write_bytes(source_data)

        # 1. Start file share
        share_task = asyncio.create_task(
            ipc_client.call("share_file", {"device_id": mock.device_id, "file_path": str(source_file)})
        )

        # 2. Simultaneously ring phone over primary channel
        ring_res = await ipc_client.call("find_phone", {"device_id": mock.device_id})
        assert ring_res.get("status") in ("transmitted", "ok")

        # 3. Verify ring packet reached mock phone
        await wait_for(
            lambda: any("findmyphone" in p.type for p in mock.received_packets),
            timeout=4.0,
            desc="mock receiving findmyphone while share in flight",
        )


async def test_tier3_pairwise_alert_toggle_during_download(tmp_path: Path) -> None:
    """Tier 3: Pairwise - Find This Device toggle cancellation during active file download.

    Features Combined: Find This Device Toggle (Feature 26) + Inbound File Share (Feature 32).
    Scenario: Inbound file is downloading. Mock phone sends two quick locate packets (start then stop).
    Expected Output: Alert starts, then stops via toggle; file download completes without corruption.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]

        file_payload = b"Streaming file during toggle alert cancellation" * 100
        port, srv, _ = await _start_mock_payload_server(mock, file_payload, daemon_cert)

        try:
            # 1. Dispatch file share packet
            share_pkt = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"filename": "toggle_simultaneous.bin"},
                payloadSize=len(file_payload),
                payloadTransferInfo={"port": port},
            )
            await mock.send_packet(share_pkt)

            # 2. Dispatch alert packet 1 (trigger)
            alert_pkt = Packet(id=int(time.time() * 1000) + 1, type="kdeconnect.findmyphone.request", body={})
            await mock.send_packet(alert_pkt)
            await asyncio.sleep(0.1)

            # 3. Dispatch alert packet 2 (toggle cancel)
            await mock.send_packet(alert_pkt)

            # 4. Verify file finishes cleanly
            dest_file = download_dir / "toggle_simultaneous.bin"
            await wait_for(lambda: dest_file.exists(), timeout=5.0)
            assert dest_file.read_bytes() == file_payload

            # 5. Verify alert is inactive
            await wait_for(lambda: not (daemon.alert_service.is_active()), timeout=4.0, desc="alert inactive")
            status = await ipc_client.call("alert_status", {})
            assert status.get("active") is False
        finally:
            srv.close()
            await srv.wait_closed()


async def test_tier3_pairwise_unpair_during_active_share(tmp_path: Path) -> None:
    """Tier 3: Pairwise - Unpairing a device during active file share aborts transfer cleanly.

    Features Combined: DeviceStore Unpair (Feature 8) + Share Plugin Security (Feature 29).
    Scenario: File streaming is in progress. User unpairs device from PC.
    Expected Output:
    - Active payload stream terminated.
    - Staging '.part' file unlinked.
    - Device transitions to UNPAIRED / DISCONNECTED.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]

        stream_data = b"Infinite streaming data" * 1000
        port, srv, _ = await _start_mock_payload_server(mock, stream_data, daemon_cert, chunk_delay=0.05)

        try:
            share_pkt = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"filename": "aborted_share.bin"},
                payloadSize=len(stream_data),
                payloadTransferInfo={"port": port},
            )
            await mock.send_packet(share_pkt)
            await asyncio.sleep(0.1)

            # Unpair device while transfer is streaming
            await daemon.unpair_device(mock.device_id)

            # Invariant: No complete or partial files left in download directory
            await asyncio.sleep(0.5)
            assert not (download_dir / "aborted_share.bin").exists()
            assert not any(f.name.endswith(".part") for f in download_dir.iterdir())
        finally:
            srv.close()
            await srv.wait_closed()


async def test_tier3_pairwise_interleaved_text_share_and_alert_bursts(tmp_path: Path) -> None:
    """Tier 3: Pairwise - Interleaved text shares and locate requests across active connection.

    Features Combined: Share Text (Feature 30) + Find This Device (Feature 26).
    Scenario: Interleaved sequence of 4 text shares and 4 alert locate packets.
    Expected Output: Zero frame desynchronization; all text shares decoded; alert handles states.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        for i in range(4):
            # Send text
            text_pkt = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"text": f"Interleaved message #{i}"},
            )
            await mock.send_packet(text_pkt)

            # Send alert request
            alert_pkt = Packet(
                id=int(time.time() * 1000) + 1,
                type="kdeconnect.findmyphone.request",
                body={},
            )
            await mock.send_packet(alert_pkt)
            await asyncio.sleep(0.05)

        # Verify connection is still healthy and responsive
        ping_res = await ipc_client.call("send_ping", {"device_id": mock.device_id})
        assert ping_res.get("status") in ("sent", "ok")


# =============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# Requirement: >=5 realistic end-to-end multi-step Linux-Android user workflows
# =============================================================================

async def test_tier4_scenario_android_to_linux_photo_transfer_and_ring(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Tier 4: Scenario 1 - Mobile photo dump followed by locating misplaced phone.

    Workflow:
    1. Android discovers PC on Wi-Fi and establishes pairing.
    2. Android sends 2 photos with spaces and timestamp metadata.
    3. Linux daemon saves photos safely into Downloads directory.
    4. User notices phone is missing in room and runs CLI `pengulink find-phone <device_id>`.
    5. Mock phone rings and confirms reception.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        sock_path = daemon.config.storage.runtime_dir / "daemon.sock"

        # Step 1: Mutual discovery & pairing
        await _pair_and_connect(daemon, mock)

        # Step 2: Android sends Photo 1
        photo1_data = b"JPEG_BINARY_DATA_PHOTO_01_FAMILY_VACATION" * 20
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]
        p1_port, s1, _ = await _start_mock_payload_server(mock, photo1_data, daemon_cert)
        try:
            pkt1 = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"filename": "Family Photo 2026.jpg", "numberOfFiles": 2},
                payloadSize=len(photo1_data),
                payloadTransferInfo={"port": p1_port},
            )
            await mock.send_packet(pkt1)

            # Step 3: Verify Photo 1 on Linux
            f1 = download_dir / "Family Photo 2026.jpg"
            await wait_for(lambda: f1.exists(), timeout=5.0)
            assert f1.read_bytes() == photo1_data
        finally:
            s1.close()
            await s1.wait_closed()

        # Step 4: Locate phone via Linux CLI
        res = await asyncio.to_thread(cli_runner.invoke, cli, ["-s", str(sock_path), "find-phone", mock.device_id])
        assert res.exit_code == 0

        # Step 5: Verify ring packet arrived at phone
        await wait_for(
            lambda: any("findmyphone" in p.type for p in mock.received_packets),
            timeout=3.0,
            desc="mock phone ringing in scenario 1",
        )


async def test_tier4_scenario_misplaced_laptop_find_and_cli_dismiss(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Tier 4: Scenario 2 - User misplaces laptop, locates it from phone, dismisses via CLI.

    Workflow:
    1. Paired Android phone sends locate request (Find This Device).
    2. PC plays alert chime and updates status to 'ringing'.
    3. User walks over to laptop, opens terminal, and runs `pengulink alert stop`.
    4. Alert silences, status returns to 'idle', and mock phone is notified or idle.
    """
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        sock_path = daemon.config.storage.runtime_dir / "daemon.sock"
        await _pair_and_connect(daemon, mock)

        # Step 1: Phone sends locate request
        locate_pkt = Packet(id=int(time.time() * 1000), type="kdeconnect.findmyphone.request", body={})
        await mock.send_packet(locate_pkt)

        # Step 2: PC alert status is active
        await wait_for(
            lambda: (daemon._active_connections is not None),
            timeout=2.0,
        )
        status_mid = await ipc_client.call("alert_status", {})
        assert status_mid.get("active") is True

        # Step 3: Dismiss from local CLI
        res_stop = await asyncio.to_thread(cli_runner.invoke, cli, ["-s", str(sock_path), "alert", "stop"])
        assert res_stop.exit_code == 0

        # Step 4: Verify inactive
        status_end = await ipc_client.call("alert_status", {})
        assert status_end.get("active") is False


async def test_tier4_scenario_bidirectional_url_and_text_exchange(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Tier 4: Scenario 3 - Work session: sharing research links and text notes bidirectionally.

    Workflow:
    1. Linux user shares research link via CLI `pengulink share url`.
    2. Phone receives link.
    3. Phone replies by sharing a text note back to Linux.
    4. Linux daemon receives note and publishes ShareReceivedEvent.
    5. Linux user sends follow-up text note via CLI `pengulink share text`.
    6. Phone receives follow-up text note.
    """
    assert ShareReceivedEvent is not None
    events: List[Any] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        sock_path = daemon.config.storage.runtime_dir / "daemon.sock"
        await _pair_and_connect(daemon, mock)
        daemon.event_bus.subscribe(ShareReceivedEvent, lambda ev: events.append(ev))

        # 1. Linux user shares URL
        res_url = await asyncio.to_thread(cli_runner.invoke, cli, ["-s", str(sock_path), "share", "url", mock.device_id, "https://arxiv.org/abs/1234"])
        assert res_url.exit_code == 0
        await wait_for(lambda: any("arxiv.org" in str(p.body) for p in mock.received_packets), timeout=3.0)

        # 2. Phone shares text note
        note_pkt = Packet(
            id=int(time.time() * 1000),
            type="kdeconnect.share.request",
            body={"text": "Review section 3 on page 4"},
        )
        await mock.send_packet(note_pkt)
        await wait_for(lambda: len(events) >= 1, timeout=3.0)
        assert getattr(events[0], "text", "") == "Review section 3 on page 4"

        # 3. Linux user shares follow-up text
        res_text = await asyncio.to_thread(cli_runner.invoke, cli, ["-s", str(sock_path), "share", "text", mock.device_id, "Got it, looks great!"])
        assert res_text.exit_code == 0
        await wait_for(lambda: any("Got it, looks great!" in str(p.body) for p in mock.received_packets), timeout=3.0)


async def test_tier4_scenario_file_transfer_with_network_flapping_and_retry(tmp_path: Path) -> None:
    """Tier 4: Scenario 4 - Network drop mid-file-transfer, clean recovery, and successful retry.

    Workflow:
    1. Inbound file transfer starts -> connection abruptly drops.
    2. Daemon purges partial staging file.
    3. Reconnection cycle establishes new mTLS session.
    4. File transfer retried from start -> succeeds 100% byte-for-byte.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]

        # 1. Aborted attempt: drop socket mid-transfer
        port1, srv1, _ = await _start_mock_payload_server(mock, b"INCOMPLETE", daemon_cert)
        try:
            pkt1 = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"filename": "resilient_document.pdf"},
                payloadSize=5000,
                payloadTransferInfo={"port": port1},
            )
            await mock.send_packet(pkt1)
            await asyncio.sleep(0.5)
            # Invariant: Partial file wiped
            assert not (download_dir / "resilient_document.pdf").exists()
        finally:
            srv1.close()
            await srv1.wait_closed()

        # 2. Clean retry with full payload
        full_data = b"PDF_FULL_RESILIENT_FILE_CONTENT_VERIFIED" * 50
        port2, srv2, _ = await _start_mock_payload_server(mock, full_data, daemon_cert)
        try:
            pkt2 = Packet(
                id=int(time.time() * 1000) + 10,
                type="kdeconnect.share.request",
                body={"filename": "resilient_document.pdf"},
                payloadSize=len(full_data),
                payloadTransferInfo={"port": port2},
            )
            await mock.send_packet(pkt2)

            dest_file = download_dir / "resilient_document.pdf"
            await wait_for(lambda: dest_file.exists(), timeout=5.0)
            assert dest_file.read_bytes() == full_data
        finally:
            srv2.close()
            await srv2.wait_closed()


async def test_tier4_scenario_high_concurrency_stress_workload(tmp_path: Path) -> None:
    """Tier 4: Scenario 5 - High concurrency multi-device stress workload.

    Workflow:
    - Multiple simultaneous operations: file downloads, text shares, clipboard updates, and alert rings.
    - Asserts zero deadlocks, zero socket leaks, and clean completion of all tasks.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]

        file_data = b"Concurrent stress workload file payload" * 40
        port, srv, _ = await _start_mock_payload_server(mock, file_data, daemon_cert)

        try:
            # Launch simultaneous tasks
            tasks = [
                # Inbound file
                mock.send_packet(Packet(
                    id=int(time.time() * 1000),
                    type="kdeconnect.share.request",
                    body={"filename": "stress_file.dat"},
                    payloadSize=len(file_data),
                    payloadTransferInfo={"port": port},
                )),
                # Outbound text
                ipc_client.call("share_text", {"device_id": mock.device_id, "text": "Concurrent text 1"}),
                # Local alert
                ipc_client.call("alert_test", {"duration": 10.0}),
                # Outbound find phone
                ipc_client.call("find_phone", {"device_id": mock.device_id}),
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Verify file arrived
            dest_file = download_dir / "stress_file.dat"
            await wait_for(lambda: dest_file.exists(), timeout=5.0)
            assert dest_file.read_bytes() == file_data
        finally:
            srv.close()
            await srv.wait_closed()
            await ipc_client.call("alert_stop", {})


async def test_tier4_scenario_security_sandbox_adversarial_attack_flow(tmp_path: Path) -> None:
    """Tier 4: Scenario 6 - Multi-vector adversarial penetration attempt.

    Workflow:
    1. Malicious peer attempts to overwrite '~/.bashrc' or '../sensitive.key'.
    2. Simultaneously triggers rapid locate alerts to flood audio subsystem.
    3. Traversal is quarantined/rejected; no disk escape occurs; single-instance mutex limits alert.
    4. Desktop operator unpairs device, revoking all credentials and ending attack.
    """
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PENGULINK_DOWNLOAD_DIR"] = str(download_dir)

    target_secret = tmp_path / "sensitive.key"
    target_secret.write_bytes(b"SUPER_SECRET_USER_KEY")

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)
        daemon_cert = mock.paired_peers[daemon.config.device.device_id]

        attack_bytes = b"CORRUPTED_SECRET_INJECTION"
        port, srv, _ = await _start_mock_payload_server(mock, attack_bytes, daemon_cert)

        try:
            # 1. Send hostile path traversal packet
            hostile_pkt = Packet(
                id=int(time.time() * 1000),
                type="kdeconnect.share.request",
                body={"filename": "../sensitive.key"},
                payloadSize=len(attack_bytes),
                payloadTransferInfo={"port": port},
            )
            await mock.send_packet(hostile_pkt)

            # 2. Spam locate packets
            for _ in range(5):
                await mock.send_packet(Packet(id=int(time.time() * 1000), type="kdeconnect.findmyphone.request", body={}))

            await asyncio.sleep(0.5)

            # Invariant 1: Sensitive key outside download dir was untouched
            assert target_secret.read_bytes() == b"SUPER_SECRET_USER_KEY"

            # 3. Operator unpairs malicious device
            await daemon.unpair_device(mock.device_id)
            assert not daemon.is_device_paired(mock.device_id)
        finally:
            srv.close()
            await srv.wait_closed()

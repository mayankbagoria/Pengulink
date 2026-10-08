"""Empirical Adversarial Stress Tests for PenguLink Network Subsystem.

Covers:
1. Rapid socket connect/disconnect cycles.
2. Simultaneous bidirectional connection collisions and tie-breaking.
3. TCP port exhaustion across range 1716..1764 and fallback resilience.
4. Abrupt connection termination (TCP RST via SO_LINGER, dropped sockets).
5. Half-open sockets, stalled preambles, and read/write timeouts.
6. UDP discovery fuzzing, corrupted datagrams, reflection attacks, and port fallback.
7. File descriptor (FD) leak and dangling asyncio task verification.
8. Empirical reproduction of identified architectural defects:
   - Deadlock in TCPStreamManager.stop() with active connections.
   - CancelledError bypass of conn.close() in _process_inbound_client.
   - Unhandled ValueError on non-integer protocolVersion in UDPDiscovery.
   - Stale connection retention in _active_connections.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import struct
import time
from typing import List, Optional, Set, Tuple

import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.core.exceptions import (
    ConnectionClosedError,
    ConnectionTimeoutError,
    LimitOverrunError,
    PortBindingError,
)
from pengulink.network.packet import MAX_PACKET_SIZE, IdentityBody, Packet
from pengulink.network.tcp import (
    ConnectionRole,
    TCPConnection,
    TCPStreamManager,
)
from pengulink.network.udp import (
    DiscoveredDevice,
    UDPDiscovery,
)


def get_open_fd_count() -> int:
    """Read the number of open file descriptors for the current Linux process."""
    try:
        return len(os.listdir("/proc/self/fd"))
    except (FileNotFoundError, OSError):
        return 0


# ==============================================================================
# 1. Rapid Socket Connect/Disconnect Cycles
# ==============================================================================

async def test_adv_rapid_connect_disconnect_cycles(sample_config: PenguLinkConfig) -> None:
    """Stress test: 100 rapid connection and clean disconnect cycles on loopback."""
    sample_config.network.tcp_port_min = 1770
    sample_config.network.tcp_port_max = 1775
    server_mgr = TCPStreamManager(sample_config)
    port = await server_mgr.start()

    initial_fds = get_open_fd_count()
    cycle_count = 100

    for i in range(cycle_count):
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect(("127.0.0.1", port))
        
        # Send valid identity preamble
        client_id = f"client_stress_{i:04d}"
        preamble = Packet(
            type="kdeconnect.identity",
            body={
                "deviceId": client_id,
                "deviceName": f"Stress-{i}",
                "deviceType": "phone",
                "protocolVersion": 7,
                "tcpPort": 1716,
            },
        ).to_wire()
        client_sock.sendall(preamble)
        
        # Clean close from client side
        client_sock.shutdown(socket.SHUT_RDWR)
        client_sock.close()

    # Allow asyncio server loop to process inbound connection tasks
    await asyncio.sleep(0.1)

    # Server must remain running and pending tasks must drain
    assert server_mgr.is_running is True
    assert len(server_mgr._pending_tasks) == 0

    # Clean active connections explicitly before stop to avoid known stop() deadlock
    for conn in list(server_mgr._active_connections.values()):
        await conn.close()
    await server_mgr.stop()
    await asyncio.sleep(0.05)

    final_fds = get_open_fd_count()
    assert abs(final_fds - initial_fds) <= 4


async def test_adv_rapid_connect_immediate_close_before_preamble(
    sample_config: PenguLinkConfig,
) -> None:
    """Stress test: 80 connections that close immediately on SYN+ACK before sending any data."""
    sample_config.network.tcp_port_min = 1776
    sample_config.network.tcp_port_max = 1780
    server_mgr = TCPStreamManager(sample_config)
    port = await server_mgr.start()

    initial_fds = get_open_fd_count()

    for _ in range(80):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", port))
        s.close()

    await asyncio.sleep(0.1)

    assert server_mgr.is_running is True
    assert len(server_mgr._pending_tasks) == 0

    await server_mgr.stop()
    await asyncio.sleep(0.05)

    final_fds = get_open_fd_count()
    assert abs(final_fds - initial_fds) <= 3


async def test_adv_rapid_corrupted_preambles(sample_config: PenguLinkConfig) -> None:
    """Stress test: 50 connections sending invalid or corrupted preambles."""
    sample_config.network.tcp_port_min = 1781
    sample_config.network.tcp_port_max = 1785
    server_mgr = TCPStreamManager(sample_config)
    port = await server_mgr.start()

    corrupted_payloads = [
        b"NOT_JSON_DATA\n",
        b"{}\n",
        b'{"type": "wrong.type", "body": {}}\n',
        b'{"type": "kdeconnect.identity"}\n',
        b'{"type": "kdeconnect.identity", "body": {"deviceName": "NoId"}}\n',
        b"\x00\x01\x02\x03\n",
        b'{"type": "kdeconnect.identity", "body": {"deviceId": ""}}\n',
    ]

    for i in range(50):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", port))
        payload = corrupted_payloads[i % len(corrupted_payloads)]
        s.sendall(payload)
        s.close()

    await asyncio.sleep(0.1)

    assert server_mgr.is_running is True
    assert len(server_mgr._pending_tasks) == 0

    await server_mgr.stop()


# ==============================================================================
# 2. Simultaneous Bidirectional Connection Collisions
# ==============================================================================

async def test_adv_simultaneous_bidirectional_collision() -> None:
    """Stress test: Simultaneous bidirectional connection between two nodes on loopback.

    Node Alpha: id='aaa_device' (lexicographically lower)
    Node Beta:  id='zzz_device' (lexicographically higher)
    Both nodes concurrently dial each other while accepting incoming connections.
    Tie-breaking rule:
      - Beta (higher) drops incoming collision from Alpha, keeping its outbound.
      - Alpha (lower) drops its outbound to Beta, accepting incoming from Beta.
    """
    cfg_a = PenguLinkConfig()
    cfg_a.device.device_id = "aaa_device_1111111111111111"
    cfg_a.device.device_name = "Node-Alpha"
    cfg_a.network.tcp_port_min = 1790
    cfg_a.network.tcp_port_max = 1792

    cfg_b = PenguLinkConfig()
    cfg_b.device.device_id = "zzz_device_9999999999999999"
    cfg_b.device.device_name = "Node-Beta"
    cfg_b.network.tcp_port_min = 1795
    cfg_b.network.tcp_port_max = 1797

    established_a: List[TCPConnection] = []
    established_b: List[TCPConnection] = []

    mgr_a = TCPStreamManager(cfg_a, on_connection_established=lambda c: established_a.append(c))
    mgr_b = TCPStreamManager(cfg_b, on_connection_established=lambda c: established_b.append(c))

    port_a = await mgr_a.start()
    port_b = await mgr_b.start()

    # Concurrently initiate outbound connections from both sides
    results = await asyncio.gather(
        mgr_a.connect_to_device("127.0.0.1", port_b, cfg_b.device.device_id),
        mgr_b.connect_to_device("127.0.0.1", port_a, cfg_a.device.device_id),
        return_exceptions=True,
    )

    await asyncio.sleep(0.1)

    # Winner on Beta: outbound connection to Alpha should remain open
    conn_b_outbound = mgr_b._active_connections.get("aaa_device_1111111111111111")
    assert conn_b_outbound is not None
    assert conn_b_outbound.is_closed is False
    assert conn_b_outbound.role == ConnectionRole.OUTBOUND

    # Loser on Alpha: Alpha's outbound was closed by tie-breaker, accepted inbound from Beta
    conn_a_active = mgr_a._active_connections.get("zzz_device_9999999999999999")
    assert conn_a_active is not None
    assert conn_a_active.is_closed is False
    assert conn_a_active.role == ConnectionRole.INBOUND

    # Clean shutdown
    for c in list(mgr_a._active_connections.values()) + list(mgr_b._active_connections.values()):
        await c.close()
    await mgr_a.stop()
    await mgr_b.stop()


async def test_adv_repeated_collision_cycles() -> None:
    """Stress test: 10 repeated collision tie-break rounds without resource leaks."""
    cfg_a = PenguLinkConfig()
    cfg_a.device.device_id = "device_lower_000000000000"
    cfg_a.network.tcp_port_min = 1800
    cfg_a.network.tcp_port_max = 1802

    cfg_b = PenguLinkConfig()
    cfg_b.device.device_id = "device_upper_999999999999"
    cfg_b.network.tcp_port_min = 1805
    cfg_b.network.tcp_port_max = 1807

    mgr_a = TCPStreamManager(cfg_a)
    mgr_b = TCPStreamManager(cfg_b)

    port_a = await mgr_a.start()
    port_b = await mgr_b.start()

    initial_fds = get_open_fd_count()

    for _ in range(10):
        conn_a = await mgr_a.connect_to_device("127.0.0.1", port_b, cfg_b.device.device_id)
        conn_b = await mgr_b.connect_to_device("127.0.0.1", port_a, cfg_a.device.device_id)
        await asyncio.sleep(0.02)

        for c in list(mgr_a._active_connections.values()) + list(mgr_b._active_connections.values()):
            await c.close()
        await asyncio.sleep(0.02)

    await mgr_a.stop()
    await mgr_b.stop()
    await asyncio.sleep(0.05)

    final_fds = get_open_fd_count()
    assert abs(final_fds - initial_fds) <= 4


# ==============================================================================
# 3. Port Exhaustion in Range 1716..1764
# ==============================================================================

async def test_adv_tcp_port_exhaustion_entire_range() -> None:
    """Stress test: Occupy all 49 ports across 1716..1764 and verify PortBindingError."""
    occupied_sockets: List[socket.socket] = []
    try:
        for port in range(1716, 1765):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", port))
            s.listen(1)
            occupied_sockets.append(s)

        cfg = PenguLinkConfig()
        cfg.network.tcp_port_min = 1716
        cfg.network.tcp_port_max = 1764
        mgr = TCPStreamManager(cfg)

        with pytest.raises(PortBindingError) as exc_info:
            await mgr.start()
        assert mgr.is_running is False
        assert mgr.bound_port is None
        assert "1716-1764" in str(exc_info.value)
    finally:
        for s in occupied_sockets:
            s.close()


async def test_adv_tcp_port_fallback_resilience() -> None:
    """Verify fallback when all ports except the very last port in range are occupied."""
    port_min = 1730
    port_max = 1735
    occupied_sockets: List[socket.socket] = []

    # Occupy ports 1730..1734, leaving 1735 free
    for port in range(port_min, port_max):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        s.listen(1)
        occupied_sockets.append(s)

    cfg = PenguLinkConfig()
    cfg.network.tcp_port_min = port_min
    cfg.network.tcp_port_max = port_max
    mgr = TCPStreamManager(cfg)

    try:
        bound = await mgr.start()
        assert bound == 1735
        assert mgr.bound_port == 1735
        assert mgr.is_running is True
        await mgr.stop()
    finally:
        for s in occupied_sockets:
            s.close()


async def test_adv_tcp_inverted_port_range() -> None:
    """Verify PortBindingError when min_port > max_port."""
    cfg = PenguLinkConfig()
    cfg.network.tcp_port_min = 1764
    cfg.network.tcp_port_max = 1716
    mgr = TCPStreamManager(cfg)

    with pytest.raises(PortBindingError):
        await mgr.start()


# ==============================================================================
# 4. Abrupt Connection Termination (TCP RST via SO_LINGER)
# ==============================================================================

async def test_adv_abrupt_rst_during_preamble(sample_config: PenguLinkConfig) -> None:
    """Stress test: Client sends partial preamble and immediately resets socket via TCP RST."""
    sample_config.network.tcp_port_min = 1810
    sample_config.network.tcp_port_max = 1815
    server_mgr = TCPStreamManager(sample_config)
    port = await server_mgr.start()

    initial_fds = get_open_fd_count()

    for _ in range(30):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", port))
        s.sendall(b'{"type": "kdeconnect.identity", "body": {')
        # Configure SO_LINGER onoff=1, linger=0 to emit immediate TCP RST on close()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        s.close()

    await asyncio.sleep(0.1)

    assert server_mgr.is_running is True
    assert len(server_mgr._pending_tasks) == 0

    await server_mgr.stop()
    await asyncio.sleep(0.05)

    final_fds = get_open_fd_count()
    assert abs(final_fds - initial_fds) <= 3


async def test_adv_abrupt_rst_after_established(sample_config: PenguLinkConfig) -> None:
    """Stress test: Client fully establishes preamble, then brutally RSTs the connection."""
    sample_config.network.tcp_port_min = 1816
    sample_config.network.tcp_port_max = 1820
    established: List[TCPConnection] = []

    server_mgr = TCPStreamManager(
        sample_config, on_connection_established=lambda c: established.append(c)
    )
    port = await server_mgr.start()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", port))

    preamble = Packet(
        type="kdeconnect.identity",
        body={"deviceId": "rst_victim_device_01", "deviceName": "RSTVictim"},
    ).to_wire()
    s.sendall(preamble)

    await asyncio.sleep(0.05)
    assert len(established) == 1
    server_conn = established[0]

    # Client fires TCP RST
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    s.close()

    await asyncio.sleep(0.05)

    # Server reading on dropped connection must detect RST and raise ConnectionClosedError
    with pytest.raises(ConnectionClosedError):
        await server_conn.read_line(timeout=1.0)

    assert server_conn.is_closed is True

    # Server writing to dropped connection must raise ConnectionClosedError
    with pytest.raises(ConnectionClosedError):
        await server_conn.write_line(b'{"type":"kdeconnect.ping","body":{}}')

    # Close active connections before stop to prevent deadlock
    for conn in list(server_mgr._active_connections.values()):
        await conn.close()
    await server_mgr.stop()


# ==============================================================================
# 5. Half-Open Sockets & Timeout Handling
# ==============================================================================

async def test_adv_half_open_silent_client() -> None:
    """Verify server detects silent client and times out cleanly."""
    sock_a, sock_b = socket.socketpair()
    sock_a.setblocking(False)
    sock_b.setblocking(False)

    reader, writer = await asyncio.open_connection(sock=sock_a)
    conn = TCPConnection(
        reader=reader,
        writer=writer,
        role=ConnectionRole.INBOUND,
        peer_ip="127.0.0.1",
        peer_port=9999,
    )

    try:
        with pytest.raises(ConnectionTimeoutError) as exc_info:
            await conn.read_line(timeout=0.08)
        assert exc_info.value.context.get("timeout_seconds") == 0.08
    finally:
        await conn.close()
        sock_b.close()


async def test_adv_half_open_partial_line_timeout() -> None:
    """Verify timeout occurs when line has no newline delimiter."""
    sock_a, sock_b = socket.socketpair()
    sock_a.setblocking(False)
    sock_b.setblocking(False)

    reader, writer = await asyncio.open_connection(sock=sock_a)
    conn = TCPConnection(
        reader=reader,
        writer=writer,
        role=ConnectionRole.INBOUND,
        peer_ip="127.0.0.1",
        peer_port=9999,
    )

    try:
        sock_b.sendall(b'{"id":1,"type":"kdeconnect.ping"')
        with pytest.raises(ConnectionTimeoutError):
            await conn.read_line(timeout=0.08)
    finally:
        await conn.close()
        sock_b.close()


async def test_adv_clean_half_close_eof() -> None:
    """Verify clean EOF reception when remote shuts down write half."""
    sock_a, sock_b = socket.socketpair()
    sock_a.setblocking(False)
    sock_b.setblocking(False)

    reader, writer = await asyncio.open_connection(sock=sock_a)
    conn = TCPConnection(
        reader=reader,
        writer=writer,
        role=ConnectionRole.INBOUND,
        peer_ip="127.0.0.1",
        peer_port=9999,
    )

    try:
        sock_b.sendall(b'{"test":1}\n')
        sock_b.shutdown(socket.SHUT_WR)

        line1 = await conn.read_line(timeout=1.0)
        assert line1 == b'{"test":1}\n'

        line2 = await conn.read_line(timeout=1.0)
        assert line2 == b""  # Clean EOF
    finally:
        await conn.close()
        sock_b.close()


# ==============================================================================
# 6. UDP Discovery Loopback Adversarial Stress
# ==============================================================================

async def test_adv_udp_datagram_fuzzing(sample_config: PenguLinkConfig) -> None:
    """Stress test: Fuzz UDP discovery listener with malformed and invalid packets."""
    discovered_devices: List[DiscoveredDevice] = []
    sample_config.network.udp_port = 0

    discovery = UDPDiscovery(
        config=sample_config,
        bound_tcp_port_getter=lambda: 1716,
        on_device_discovered=lambda d: discovered_devices.append(d),
    )
    await discovery.start()

    # Truly invalid payloads that must never produce DiscoveredDevice
    invalid_fuzz_payloads = [
        b"",
        b"\n",
        b"   \r\n",
        b"NOT_A_JSON_PACKET",
        b'{"invalid_json": ',
        b"null",
        b"12345",
        b'"just_a_string"',
        b"[]",
        b'{"type": "wrong_type"}',
        b'{"type": "kdeconnect.identity"}',
        b'{"type": "kdeconnect.identity", "body": null}',
        b'{"type": "kdeconnect.identity", "body": "string_body"}',
        b'{"type": "kdeconnect.identity", "body": {"deviceId": null}}',
        b'{"type": "kdeconnect.identity", "body": {"deviceId": 123}}',
        b'{"type": "kdeconnect.identity", "body": {"deviceId": ""}}',
        json.dumps({
            "type": "kdeconnect.identity",
            "body": {"deviceId": sample_config.device.device_id, "tcpPort": 1716},
        }).encode("utf-8"),
    ]

    for payload in invalid_fuzz_payloads:
        discovery._handle_datagram(payload, ("127.0.0.1", 9999))

    assert len(discovered_devices) == 0

    # Verify discovery still functions cleanly after hostile barrage
    valid_payload = json.dumps({
        "type": "kdeconnect.identity",
        "body": {
            "deviceId": "valid_post_fuzz_node",
            "deviceName": "ValidNode",
            "deviceType": "phone",
            "protocolVersion": 7,
            "tcpPort": 1718,
        },
    }).encode("utf-8")
    discovery._handle_datagram(valid_payload, ("127.0.0.1", 1718))

    assert len(discovered_devices) == 1
    assert discovered_devices[0].device_id == "valid_post_fuzz_node"
    assert discovered_devices[0].tcp_port == 1718

    await discovery.stop()


async def test_adv_udp_rapid_lifecycle_churn(sample_config: PenguLinkConfig) -> None:
    """Stress test: Rapidly start and stop UDPDiscovery 20 times without task or socket leaks."""
    sample_config.network.udp_port = 0
    initial_fds = get_open_fd_count()

    for _ in range(20):
        discovery = UDPDiscovery(
            config=sample_config,
            bound_tcp_port_getter=lambda: 1716,
        )
        await discovery.start()
        assert discovery.is_running is True
        assert discovery._broadcast_task is not None

        await discovery.stop()
        assert discovery.is_running is False
        assert discovery._broadcast_task is None
        assert discovery._transport is None

    await asyncio.sleep(0.05)
    final_fds = get_open_fd_count()
    assert abs(final_fds - initial_fds) <= 2


async def test_adv_udp_port_busy_fallback(sample_config: PenguLinkConfig) -> None:
    """Assert UDPDiscovery falls back to ephemeral port 0 if port is occupied."""
    test_udp_port = 17160
    occupier = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    occupier.bind(("0.0.0.0", test_udp_port))

    sample_config.network.udp_port = test_udp_port
    discovery = UDPDiscovery(
        config=sample_config,
        bound_tcp_port_getter=lambda: 1716,
    )

    try:
        await discovery.start()
        assert discovery.is_running is True
        await discovery.stop()
    finally:
        occupier.close()


# ==============================================================================
# 7. Verification of Remediated Architectural Defects
# ==============================================================================

async def test_adv_bug_stop_deadlock_with_active_connection() -> None:
    """VERIFY FIX: TCPStreamManager.stop() closes cleanly without deadlock when active connections exist."""
    cfg = PenguLinkConfig()
    cfg.network.tcp_port_min = 1840
    cfg.network.tcp_port_max = 1845
    mgr = TCPStreamManager(cfg)
    port = await mgr.start()

    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    preamble = Packet(
        type="kdeconnect.identity",
        body={"deviceId": "deadlock_client_01", "deviceName": "DeadlockClient"},
    ).to_wire()
    writer.write(preamble)
    await writer.drain()
    await asyncio.sleep(0.05)

    # Calling mgr.stop() while client is connected must complete cleanly and quickly
    await asyncio.wait_for(mgr.stop(), timeout=1.0)
    assert mgr.is_running is False
    assert len(mgr._active_connections) == 0

    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass


async def test_adv_bug_cancelled_pending_task_socket_leak() -> None:
    """VERIFY FIX: Cancelled pending task closes connection without leaking socket in server._clients."""
    cfg = PenguLinkConfig()
    cfg.network.tcp_port_min = 1850
    cfg.network.tcp_port_max = 1855
    mgr = TCPStreamManager(cfg)
    port = await mgr.start()

    c_reader, c_writer = await asyncio.open_connection("127.0.0.1", port)
    await asyncio.sleep(0.05)

    assert len(mgr._pending_tasks) == 1
    assert len(mgr._server._clients) == 1

    # Cancel pending tasks as mgr.stop() does
    for task in list(mgr._pending_tasks):
        task.cancel()
    await asyncio.gather(*mgr._pending_tasks, return_exceptions=True)

    # Verify fix: transport was detached from server._clients because conn.close() was executed in finally
    assert len(mgr._server._clients) == 0

    c_writer.close()
    await c_writer.wait_closed()
    if mgr._server:
        mgr._server.close()
        await mgr._server.wait_closed()


async def test_adv_bug_udp_protocol_version_non_integer_crash(
    sample_config: PenguLinkConfig,
) -> None:
    """VERIFY FIX: Non-integer protocolVersion in UDP datagram defaults to 7 without crashing."""
    sample_config.network.udp_port = 0
    discovered_list: list[DiscoveredDevice] = []
    discovery = UDPDiscovery(
        sample_config,
        bound_tcp_port_getter=lambda: 1716,
        on_device_discovered=lambda d: discovered_list.append(d),
    )
    await discovery.start()

    malicious_datagram = json.dumps({
        "type": "kdeconnect.identity",
        "body": {
            "deviceId": "crasher_node",
            "protocolVersion": "not_an_int",
            "tcpPort": 1716,
        },
    }).encode("utf-8")

    # Invoking _handle_datagram must not raise ValueError; defaults protocol_version to 7
    discovery._handle_datagram(malicious_datagram, ("127.0.0.1", 1716))
    assert len(discovered_list) == 1
    assert discovered_list[0].protocol_version == 7
    assert discovered_list[0].device_id == "crasher_node"

    await discovery.stop()


async def test_adv_bug_active_connections_stale_retention(sample_config: PenguLinkConfig) -> None:
    """VERIFY FIX: TCPStreamManager evicts closed connections from _active_connections."""
    sample_config.network.tcp_port_min = 1860
    sample_config.network.tcp_port_max = 1865
    mgr = TCPStreamManager(sample_config)
    port = await mgr.start()

    conns: list[TCPConnection] = []
    for i in range(5):
        s_reader, s_writer = await asyncio.open_connection("127.0.0.1", port)
        dev_id = f"stale_node_{i}"
        preamble = Packet(
            type="kdeconnect.identity",
            body={"deviceId": dev_id, "deviceName": f"StaleNode-{i}"},
        ).to_wire()
        s_writer.write(preamble)
        await s_writer.drain()
        await asyncio.sleep(0.02)
        # Verify connection was registered
        assert dev_id in mgr._active_connections
        conn = mgr._active_connections[dev_id]
        conns.append(conn)

        # Close the connection
        await conn.close()
        s_writer.close()
        await s_writer.wait_closed()

    # Verify fix: all closed connections have been evicted from _active_connections
    assert len(mgr._active_connections) == 0

    await mgr.stop()

"""Unit tests for PenguLink TCP Stream Manager and Connection lifecycle."""

import asyncio
import socket
import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.core.exceptions import (
    ConnectionClosedError,
    ConnectionTimeoutError,
    LimitOverrunError,
    PortBindingError,
)
from pengulink.network.packet import MAX_PACKET_SIZE, Packet
from pengulink.network.tcp import (
    ConnectionRole,
    TCPConnection,
    TCPStreamManager,
    configure_stream_socket,
)


async def test_tcp_port_binding_and_fallback(sample_config: PenguLinkConfig) -> None:
    """Assert TCPStreamManager binds first available port or falls back to next."""
    # Temporarily bind 1716 using a raw socket
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupied.bind(("127.0.0.1", 1716))
    occupied.listen(1)

    try:
        sample_config.network.tcp_port_min = 1716
        sample_config.network.tcp_port_max = 1720
        mgr = TCPStreamManager(sample_config)

        # Manager should skip 1716 and bind on 1717
        bound_port = await mgr.start()
        assert bound_port == 1717
        assert mgr.bound_port == 1717

        await mgr.stop()
    finally:
        occupied.close()


async def test_tcp_port_exhaustion(sample_config: PenguLinkConfig) -> None:
    """Assert PortBindingError is raised when all ports in range are occupied."""
    sample_config.network.tcp_port_min = 1790
    sample_config.network.tcp_port_max = 1791

    s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s1.bind(("127.0.0.1", 1790))
    s1.listen(1)

    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s2.bind(("127.0.0.1", 1791))
    s2.listen(1)

    try:
        mgr = TCPStreamManager(sample_config)
        with pytest.raises(PortBindingError):
            await mgr.start()
    finally:
        s1.close()
        s2.close()


async def test_tcp_connection_preamble_exchange(sample_config: PenguLinkConfig) -> None:
    """Assert outbound client sends preamble and server receives it."""
    established_connections = []

    async def on_conn(conn: TCPConnection) -> None:
        established_connections.append(conn)

    sample_config.network.tcp_port_min = 1730
    sample_config.network.tcp_port_max = 1735
    server_mgr = TCPStreamManager(sample_config, on_connection_established=on_conn)
    port = await server_mgr.start()

    # Create client config
    client_config = PenguLinkConfig()
    client_config.device.device_id = "client_device_99999999999999999999"
    client_config.device.device_name = "Client-Rig"
    client_mgr = TCPStreamManager(client_config)

    # Connect client to server
    outbound_conn = await client_mgr.connect_to_device(
        host="127.0.0.1",
        port=port,
        target_device_id=sample_config.device.device_id,
    )
    assert outbound_conn.role == ConnectionRole.OUTBOUND

    # Wait briefly for server acceptance
    await asyncio.sleep(0.1)

    assert len(established_connections) == 1
    inbound_conn = established_connections[0]
    assert inbound_conn.role == ConnectionRole.INBOUND
    assert inbound_conn.remote_device_id == client_config.device.device_id

    # Clean teardown
    await outbound_conn.close()
    await inbound_conn.close()
    await client_mgr.stop()
    await server_mgr.stop()


async def test_tcp_buffer_overrun_mitigation(sample_config: PenguLinkConfig) -> None:
    """Assert sending >128 KB without newline triggers LimitOverrunError and drops socket."""
    sample_config.network.tcp_port_min = 1740
    sample_config.network.tcp_port_max = 1745
    server_mgr = TCPStreamManager(sample_config)
    port = await server_mgr.start()

    # Raw socket connection that sends oversized flood
    reader, writer = await asyncio.open_connection("127.0.0.1", port)

    # Create managed TCPConnection wrapper around server client
    # First, feed valid identity preamble to pass initial check
    preamble = Packet(type="kdeconnect.identity", body={"deviceId": "attacker_device_000000000000000000"}).to_wire()
    writer.write(preamble)
    await writer.drain()

    await asyncio.sleep(0.05)
    server_conn = list(server_mgr._active_connections.values())[0]

    # Now attacker sends 130 KB of non-newline data
    writer.write(b"Z" * (MAX_PACKET_SIZE + 4096))
    await writer.drain()

    # Server reading next line must catch overrun and raise LimitOverrunError
    with pytest.raises(LimitOverrunError):
        await server_conn.read_line(timeout=2.0)

    assert server_conn.is_closed is True

    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass
    await server_mgr.stop()


async def test_tcp_read_timeout() -> None:
    """Assert read_line raises ConnectionTimeoutError when peer is silent."""
    # Use paired socketpair for clean isolated stream test
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
        with pytest.raises(ConnectionTimeoutError):
            await conn.read_line(timeout=0.05)
    finally:
        await conn.close()
        sock_b.close()


async def test_tcp_collision_tie_breaking(sample_config: PenguLinkConfig) -> None:
    """Assert deterministic tie-breaking based on alphabetical device ID comparison."""
    from unittest.mock import AsyncMock, MagicMock

    sample_config.device.device_id = "device_zzz_9999999999999999999999"
    mgr = TCPStreamManager(sample_config)

    # 1. Local is higher than remote: should drop inbound collision
    assert mgr._should_drop_inbound_collision("device_aaa_1111111111111111111111") is False  # no existing connection yet

    # Simulate existing active connection
    mock_reader = asyncio.StreamReader()
    mock_writer = MagicMock()
    mock_writer.wait_closed = AsyncMock()
    existing_conn = TCPConnection(
        reader=mock_reader,
        writer=mock_writer,
        role=ConnectionRole.OUTBOUND,
        peer_ip="127.0.0.1",
        peer_port=1716,
    )
    mgr._active_connections["device_aaa_1111111111111111111111"] = existing_conn

    # Now check again: local ("device_zzz_...") > remote ("device_aaa_...")
    # Should drop inbound in favor of existing outbound
    should_drop = mgr._should_drop_inbound_collision("device_aaa_1111111111111111111111")
    assert should_drop is True

    # 2. Local is lower than remote:
    # Set remote to higher string ("device_zzz_extra...")
    mgr._active_connections["device_zzz_extra_99999999999999"] = existing_conn
    should_drop_higher = mgr._should_drop_inbound_collision("device_zzz_extra_99999999999999")
    # Remote wins: should accept inbound (return False) and cancel older connection
    assert should_drop_higher is False


async def test_tcp_socket_options(sample_config: PenguLinkConfig) -> None:
    """Assert configure_stream_socket applies TCP_NODELAY and SO_KEEPALIVE to AF_INET sockets."""
    sample_config.network.tcp_port_min = 1760
    sample_config.network.tcp_port_max = 1764
    server_mgr = TCPStreamManager(sample_config)
    port = await server_mgr.start()

    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    configure_stream_socket(writer, keepalive_idle=10, keepalive_interval=5, keepalive_count=3)

    raw_sock = writer.get_extra_info("socket")
    assert raw_sock is not None
    # Verify TCP_NODELAY is enabled on TCP socket
    assert raw_sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) == 1
    # Verify SO_KEEPALIVE is enabled
    assert raw_sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) == 1

    writer.close()
    await writer.wait_closed()
    await server_mgr.stop()


async def test_tcp_write_line_behavior() -> None:
    """Assert write_line automatically appends newline and raises when closed."""
    sock_a, sock_b = socket.socketpair()
    sock_a.setblocking(False)
    sock_b.setblocking(False)

    reader, writer = await asyncio.open_connection(sock=sock_a)
    conn = TCPConnection(
        reader=reader,
        writer=writer,
        role=ConnectionRole.OUTBOUND,
        peer_ip="127.0.0.1",
        peer_port=8888,
    )

    # Write data without newline
    await conn.write_line(b'{"ping":true}')
    received = sock_b.recv(1024)
    assert received == b'{"ping":true}\n'

    # Close connection and verify write raises ConnectionClosedError
    await conn.close()
    with pytest.raises(ConnectionClosedError):
        await conn.write_line(b"after close")

    sock_b.close()


async def test_tcp_clean_teardown_no_leaks(sample_config: PenguLinkConfig) -> None:
    """Assert starting and stopping TCPStreamManager cleans up server and tasks."""
    sample_config.network.tcp_port_min = 1750
    sample_config.network.tcp_port_max = 1755
    mgr = TCPStreamManager(sample_config)

    port = await mgr.start()
    assert mgr.is_running is True
    assert mgr.bound_port == port

    await mgr.stop()
    assert mgr.is_running is False
    assert mgr.bound_port is None
    assert len(mgr._active_connections) == 0
    assert len(mgr._pending_tasks) == 0


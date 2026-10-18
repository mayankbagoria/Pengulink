"""pengulink.network.tcp
~~~~~~~~~~~~~~~~~~~~~
KDE Connect TCP Stream Manager.
Manages port fallback binding (1716..1764), outbound client connections,
inbound client acceptance, socket options, keepalive, and connection lifecycles.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import ssl
import time
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Optional, Set

from pengulink.core.exceptions import (
    ConnectionClosedError,
    ConnectionTimeoutError,
    LimitOverrunError,
    PortBindingError,
    TLSUpgradeError,
    TruncatedPacketError,
)
from pengulink.network.packet import MAX_PACKET_SIZE, IdentityBody, Packet

logger = logging.getLogger("pengulink.network.tcp")


class ConnectionRole(str, Enum):
    """Direction of the underlying TCP connection."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"


def configure_stream_socket(
    writer: asyncio.StreamWriter,
    keepalive_idle: int = 10,
    keepalive_interval: int = 5,
    keepalive_count: int = 3,
) -> None:
    """Apply Linux keep-alive and low-latency options to underlying socket."""
    sock: Optional[socket.socket] = writer.get_extra_info("socket")
    if not sock:
        return

    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        # Linux-specific TCP keep-alive timing
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, keepalive_idle)
        elif hasattr(socket, "TCP_KEEPALIVE"):  # macOS fallback
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, keepalive_idle)

        if hasattr(socket, "TCP_KEEPINTVL"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, keepalive_interval)

        if hasattr(socket, "TCP_KEEPCNT"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, keepalive_count)
    except OSError as exc:
        logger.debug("Failed configuring socket options: %s", exc)


class TCPConnection:
    """Encapsulates a managed TCP stream with framing, timeouts, and TLS upgrade capabilities."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        role: ConnectionRole,
        peer_ip: str,
        peer_port: int,
        keepalive_idle: int = 10,
        keepalive_interval: int = 5,
        keepalive_count: int = 3,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.role = role
        self.peer_ip = peer_ip
        self.peer_port = peer_port

        self.remote_device_id: Optional[str] = None
        self.remote_protocol_version: int = 7
        self.is_tls_upgraded: bool = False
        self.last_activity: float = time.time()
        self._closed: bool = False
        self._close_callbacks: list[Callable[[TCPConnection], Any]] = []
        self._close_tasks: Set[asyncio.Task[Any]] = set()

        configure_stream_socket(
            self.writer,
            keepalive_idle=keepalive_idle,
            keepalive_interval=keepalive_interval,
            keepalive_count=keepalive_count,
        )

    def add_close_callback(self, cb: Callable[[TCPConnection], Any]) -> None:
        """Register a callback to be invoked when this connection closes."""
        if self._closed:
            try:
                res = cb(self)
                if asyncio.iscoroutine(res):
                    task = asyncio.create_task(res)
                    self._close_tasks.add(task)
                    task.add_done_callback(self._close_tasks.discard)
            except Exception as exc:
                logger.debug("Error in immediate close callback: %s", exc)
            return
        self._close_callbacks.append(cb)

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def read_line(self, timeout: Optional[float] = 10.0) -> bytes:
        """Read a single newline-delimited line with timeout and 128KB buffer defense.

        Returns b"" on clean EOF.
        """
        if self._closed:
            return b""

        try:
            coro = self.reader.readline()
            line = await asyncio.wait_for(coro, timeout=timeout) if timeout else await coro
            if not line:
                await self.close()
                return b""  # Clean EOF
            if not line.endswith(b"\n"):
                logger.warning(
                    "Truncated line received at stream EOF from %s:%d (%d bytes)",
                    self.peer_ip,
                    self.peer_port,
                    len(line),
                )
                await self.close()
                raise TruncatedPacketError("Stream reached EOF before newline delimiter was received")
            self.last_activity = time.time()
            return line
        except (asyncio.LimitOverrunError, ValueError) as exc:
            # Handle Python 3.14 dual exception behavior for limit overrun
            is_overrun = isinstance(exc, asyncio.LimitOverrunError) or (
                "exceed the limit" in str(exc).lower()
                or "separator is not found" in str(exc).lower()
            )
            if is_overrun:
                logger.warning(
                    "128KB buffer limit overrun from %s:%d. Terminating connection to mitigate CVE-2020-26164.",
                    self.peer_ip,
                    self.peer_port,
                )
                if hasattr(self.reader, "_buffer"):
                    self.reader._buffer.clear()
                await self.close()
                raise LimitOverrunError(
                    "Stream line exceeded buffer guard limit (128 KB) without newline delimiter",
                    consumed_bytes=MAX_PACKET_SIZE,
                ) from exc
            raise
        except (asyncio.TimeoutError, TimeoutError) as exc:
            logger.debug("Read timed out on connection from %s:%d", self.peer_ip, self.peer_port)
            raise ConnectionTimeoutError(
                "Read line timed out",
                timeout_seconds=timeout,
                host=self.peer_ip,
                port=self.peer_port,
            ) from exc
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            logger.debug(
                "Connection dropped during read from %s:%d: %s",
                self.peer_ip,
                self.peer_port,
                exc,
            )
            await self.close()
            raise ConnectionClosedError(f"Socket dropped: {exc}") from exc

    async def write_line(self, data: bytes, timeout: float = 10.0) -> None:
        """Write newline-terminated bytes and await socket drain with timeout."""
        if self._closed:
            raise ConnectionClosedError("Cannot write to closed connection")

        if not data.endswith(b"\n"):
            data = data + b"\n"

        try:
            self.writer.write(data)
            await asyncio.wait_for(self.writer.drain(), timeout=timeout)
            self.last_activity = time.time()
        except (asyncio.TimeoutError, TimeoutError) as exc:
            logger.warning("Socket write drain timed out to %s:%d", self.peer_ip, self.peer_port)
            await self.close()
            raise ConnectionTimeoutError(
                "Write drain timed out",
                timeout_seconds=timeout,
                host=self.peer_ip,
                port=self.peer_port,
            ) from exc
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            logger.debug(
                "Connection dropped during write to %s:%d: %s",
                self.peer_ip,
                self.peer_port,
                exc,
            )
            await self.close()
            raise ConnectionClosedError(f"Socket dropped: {exc}") from exc

    async def upgrade_to_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_side: bool,
        timeout: float = 10.0,
    ) -> None:
        """Upgrade the active TCP stream to TLS in-place via loop.start_tls().

        KDE Connect Inversion Rule:
          Outbound connector -> server_side=True
          Inbound listener   -> server_side=False
        """
        if self._closed or self.is_tls_upgraded:
            return

        if hasattr(self.reader, "_buffer") and len(self.reader._buffer) > 0:
            logger.error(
                "SECURITY ALERT: STARTTLS injection attempt detected (%d buffered bytes)",
                len(self.reader._buffer),
            )
            await self.close()
            raise TLSUpgradeError("STARTTLS injection: unconsumed cleartext bytes in stream buffer")

        loop = asyncio.get_running_loop()
        try:
            transport = self.writer.transport
            protocol = transport.get_protocol()
            new_transport = await asyncio.wait_for(
                loop.start_tls(
                    transport,
                    protocol,
                    ssl_context,
                    server_side=server_side,
                ),
                timeout=timeout,
            )

            if hasattr(self.reader, "_buffer") and len(self.reader._buffer) > 0:
                logger.error(
                    "SECURITY ALERT: STARTTLS injection attempt detected post-handshake (%d buffered bytes)",
                    len(self.reader._buffer),
                )
                await self.close()
                raise TLSUpgradeError("STARTTLS injection: unconsumed cleartext bytes in stream buffer")

            # Rebind underlying transport in StreamWriter
            self.writer._transport = new_transport
            self.is_tls_upgraded = True
            self.last_activity = time.time()
            logger.info(
                "TLS upgrade completed for %s:%d (server_side=%s)",
                self.peer_ip,
                self.peer_port,
                server_side,
            )
        except TLSUpgradeError:
            raise
        except Exception as exc:
            logger.error("TLS upgrade failed for %s:%d: %s", self.peer_ip, self.peer_port, exc)
            await self.close()
            raise TLSUpgradeError(f"TLS upgrade failed: {exc}") from exc

    async def close(self) -> None:
        """Cleanly close StreamWriter and underlying socket without resource leaks."""
        if self._closed:
            return
        self._closed = True

        try:
            self.writer.close()
            wait_closed = getattr(self.writer, "wait_closed", None)
            if callable(wait_closed):
                res = wait_closed()
                if asyncio.iscoroutine(res):
                    await res
        except (OSError, asyncio.CancelledError):
            pass
        finally:
            for cb in list(self._close_callbacks):
                try:
                    res = cb(self)
                    if asyncio.iscoroutine(res):
                        task = asyncio.create_task(res)
                        self._close_tasks.add(task)
                        task.add_done_callback(self._close_tasks.discard)
                except Exception as exc:
                    logger.debug("Error in TCPConnection close callback: %s", exc)
            self._close_callbacks.clear()
            logger.debug("Closed TCP connection to %s:%d", self.peer_ip, self.peer_port)


class TCPStreamManager:
    """Manages TCP servers, outbound connections, and active stream sessions."""

    def __init__(
        self,
        config: Any,
        on_connection_established: Optional[
            Callable[[TCPConnection], Union[Coroutine[Any, Any, None], None]]
        ] = None,
    ) -> None:
        self.config = config
        self.on_connection_established = on_connection_established

        # Extract port configurations
        if hasattr(config, "tcp_port_min"):
            self.port_min = int(config.tcp_port_min)
        elif hasattr(config, "network") and hasattr(config.network, "tcp_port_min"):
            self.port_min = int(config.network.tcp_port_min)
        else:
            self.port_min = 1716

        if hasattr(config, "tcp_port_max"):
            self.port_max = int(config.tcp_port_max)
        elif hasattr(config, "network") and hasattr(config.network, "tcp_port_max"):
            self.port_max = int(config.network.tcp_port_max)
        else:
            self.port_max = 1764

        if hasattr(config, "stream_line_limit"):
            self.stream_limit = int(config.stream_line_limit)
        elif hasattr(config, "network") and hasattr(config.network, "stream_line_limit"):
            self.stream_limit = int(config.network.stream_line_limit)
        else:
            self.stream_limit = MAX_PACKET_SIZE

        # Keepalive settings
        if hasattr(config, "network"):
            self.keepalive_idle = getattr(config.network, "tcp_keepalive_idle", 10)
            self.keepalive_interval = getattr(config.network, "tcp_keepalive_interval", 5)
            self.keepalive_count = getattr(config.network, "tcp_keepalive_count", 3)
        else:
            self.keepalive_idle = 10
            self.keepalive_interval = 5
            self.keepalive_count = 3

        self.bound_port: Optional[int] = None
        self._server: Optional[asyncio.Server] = None
        self._active_connections: Dict[str, TCPConnection] = {}
        self._pending_tasks: Set[asyncio.Task[None]] = set()
        self._running: bool = False

    @property
    def is_running(self) -> bool:
        return self._running

    def _get_device_id(self) -> str:
        if hasattr(self.config, "device_id"):
            return str(self.config.device_id)
        if hasattr(self.config, "device") and hasattr(self.config.device, "device_id"):
            return str(self.config.device.device_id)
        return "pengulink_local"

    def _get_device_name(self) -> str:
        if hasattr(self.config, "device_name"):
            return str(self.config.device_name)
        if hasattr(self.config, "device") and hasattr(self.config.device, "device_name"):
            return str(self.config.device.device_name)
        return "PenguLink-Linux"

    def _get_device_type(self) -> str:
        if hasattr(self.config, "device_type"):
            return str(self.config.device_type)
        if hasattr(self.config, "device") and hasattr(self.config.device, "device_type"):
            return str(self.config.device.device_type)
        return "desktop"

    def _get_protocol_version(self) -> int:
        if hasattr(self.config, "protocol_version"):
            return int(self.config.protocol_version)
        if hasattr(self.config, "device") and hasattr(self.config.device, "protocol_version"):
            return int(self.config.device.protocol_version)
        return 7

    def _get_incoming_capabilities(self) -> list[str]:
        if hasattr(self.config, "incoming_capabilities"):
            return list(self.config.incoming_capabilities)
        if hasattr(self.config, "device") and hasattr(self.config.device, "incoming_capabilities"):
            return list(self.config.device.incoming_capabilities)
        return []

    def _get_outgoing_capabilities(self) -> list[str]:
        if hasattr(self.config, "outgoing_capabilities"):
            return list(self.config.outgoing_capabilities)
        if hasattr(self.config, "device") and hasattr(self.config.device, "outgoing_capabilities"):
            return list(self.config.device.outgoing_capabilities)
        return []

    async def start(self) -> int:
        """Bind TCP server on first available port in port_min..port_max."""
        if self._running and self.bound_port is not None:
            return self.bound_port

        for port in range(self.port_min, self.port_max + 1):
            try:
                self._server = await asyncio.start_server(
                    self._on_client_connected,
                    host="0.0.0.0",
                    port=port,
                    limit=self.stream_limit,
                    reuse_address=True,
                )
                self.bound_port = port
                self._running = True
                logger.info("TCP server successfully bound on port %d", port)
                return port
            except OSError as exc:
                logger.debug("Port %d busy: %s. Trying next...", port, exc)
                continue

        raise PortBindingError(
            f"Could not bind TCP port in range {self.port_min}..{self.port_max}",
            protocol="TCP",
            port_range=f"{self.port_min}-{self.port_max}",
        )

    def _on_connection_closed(self, conn: TCPConnection) -> None:
        """Evict closed connection from _active_connections."""
        dev_id = conn.remote_device_id
        if dev_id and dev_id in self._active_connections:
            if self._active_connections[dev_id] is conn:
                self._active_connections.pop(dev_id, None)
                logger.debug("Evicted closed connection for device %s", dev_id)

    async def stop(self) -> None:
        """Cleanly stop TCP server and close all active connections without deadlock."""
        if not self._running:
            return
        self._running = False

        server = self._server
        self._server = None

        # 1. Stop accepting new incoming connections immediately
        if server:
            server.close()
            if hasattr(server, "close_clients"):
                try:
                    server.close_clients()
                except Exception:
                    pass

        # 2. Cancel and drain in-flight connection handlers
        for task in list(self._pending_tasks):
            task.cancel()
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()

        # 3. Close all active established connections
        connections = list(self._active_connections.values())
        self._active_connections.clear()
        if connections:
            await asyncio.gather(*(conn.close() for conn in connections), return_exceptions=True)

        # 4. Wait for server closure with timeout guard
        if server:
            try:
                await asyncio.wait_for(server.wait_closed(), timeout=5.0)
            except (asyncio.TimeoutError, TimeoutError):
                logger.warning("TCP server.wait_closed() timed out after 5.0s during shutdown")
            except Exception as exc:
                logger.debug("Exception awaiting server.wait_closed(): %s", exc)

        self.bound_port = None
        logger.info("TCP Stream Manager stopped.")

    async def connect_to_device(
        self,
        host: str,
        port: int,
        target_device_id: str,
        timeout: float = 10.0,
    ) -> TCPConnection:
        """Establish an outbound TCP connection to a remote peer.

        Transmits initial cleartext kdeconnect.identity preamble.
        """
        logger.info("Connecting outbound TCP to %s:%d (%s)", host, port, target_device_id)
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, limit=self.stream_limit),
                timeout=timeout,
            )
        except Exception as exc:
            logger.warning("Failed outbound TCP connect to %s:%d: %s", host, port, exc)
            raise ConnectionClosedError(f"Could not connect to {host}:{port}: {exc}") from exc

        conn = TCPConnection(
            reader=reader,
            writer=writer,
            role=ConnectionRole.OUTBOUND,
            peer_ip=host,
            peer_port=port,
            keepalive_idle=self.keepalive_idle,
            keepalive_interval=self.keepalive_interval,
            keepalive_count=self.keepalive_count,
        )
        conn.remote_device_id = target_device_id

        # Send cleartext identity preamble
        preamble = self._build_cleartext_identity(target_device_id=target_device_id)
        await conn.write_line(preamble, timeout=timeout)
        conn.add_close_callback(self._on_connection_closed)
        self._active_connections[target_device_id] = conn
        return conn

    def _build_cleartext_identity(self, target_device_id: Optional[str] = None) -> bytes:
        """Construct the initial cleartext identity JSON packet ending with newline."""
        body = IdentityBody(
            deviceId=self._get_device_id(),
            deviceName=self._get_device_name(),
            deviceType=self._get_device_type(),
            protocolVersion=self._get_protocol_version(),
            incomingCapabilities=self._get_incoming_capabilities(),
            outgoingCapabilities=self._get_outgoing_capabilities(),
            tcpPort=self.bound_port or self.port_min,
            targetDeviceId=target_device_id if self._get_protocol_version() >= 8 else None,
            targetProtocolVersion=8 if (target_device_id and self._get_protocol_version() >= 8) else None,
        )
        packet = Packet.create_identity(body)
        return packet.to_wire()

    def _on_client_connected(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Callback invoked by asyncio.Server upon accepting an incoming TCP client."""
        task = asyncio.create_task(self._process_inbound_client(reader, writer))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _process_inbound_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        peer_ip, peer_port = peer[0], peer[1] if peer else ("unknown", 0)
        logger.debug("Accepted inbound TCP connection from %s:%d", peer_ip, peer_port)

        conn = TCPConnection(
            reader=reader,
            writer=writer,
            role=ConnectionRole.INBOUND,
            peer_ip=peer_ip,
            peer_port=peer_port,
            keepalive_idle=self.keepalive_idle,
            keepalive_interval=self.keepalive_interval,
            keepalive_count=self.keepalive_count,
        )

        established = False
        remote_device_id: Optional[str] = None
        try:
            # 1. Read cleartext identity line within 10 seconds
            line = await conn.read_line(timeout=10.0)
            if not line:
                return

            packet_dict = json.loads(line.decode("utf-8").strip())
            if packet_dict.get("type") != "kdeconnect.identity":
                logger.warning(
                    "Inbound client sent invalid preamble type: %s", packet_dict.get("type")
                )
                return

            body = packet_dict.get("body", {})
            remote_device_id = body.get("deviceId")
            if not remote_device_id:
                logger.warning("Inbound client preamble missing deviceId")
                return

            conn.remote_device_id = remote_device_id
            raw_proto = body.get("protocolVersion", 7)
            try:
                conn.remote_protocol_version = int(raw_proto)
            except (TypeError, ValueError):
                conn.remote_protocol_version = 7

            # 2. Collision tie-breaking check against active connections
            if self._should_drop_inbound_collision(remote_device_id):
                logger.info(
                    "Dropping inbound collision from %s in favor of outbound session",
                    remote_device_id,
                )
                return

            # Register connection with auto-eviction callback
            conn.add_close_callback(self._on_connection_closed)
            self._active_connections[remote_device_id] = conn
            established = True

            # Hand off to session/TLS upgrade coordinator
            if self.on_connection_established:
                cb_res = self.on_connection_established(conn)
                if asyncio.iscoroutine(cb_res):
                    await cb_res

        except asyncio.CancelledError:
            if established and remote_device_id:
                self._active_connections.pop(remote_device_id, None)
                established = False
            raise
        except Exception as exc:
            logger.debug(
                "Error during inbound client handling from %s:%d: %s",
                peer_ip,
                peer_port,
                exc,
            )
        finally:
            if not established:
                try:
                    await asyncio.shield(conn.close())
                except (Exception, asyncio.CancelledError):
                    pass

    def _should_drop_inbound_collision(self, remote_device_id: str) -> bool:
        """Deterministic tie-breaking:

        If an existing connection is active, compare device IDs:
        If local_device_id > remote_device_id: drop inbound (return True).
        If local_device_id < remote_device_id: drop existing, keep inbound (return False).
        """
        existing = self._active_connections.get(remote_device_id)
        if not existing or existing.is_closed:
            return False

        # CVE-2020-26164 Defense: Never allow an unauthenticated inbound collision to drop an active TLS session
        if getattr(existing, "is_tls_upgraded", False):
            return True

        local_id = str(self._get_device_id())
        remote_id = str(remote_device_id) if remote_device_id is not None else ""
        if local_id > remote_id:
            return True  # Local wins: keep outbound, reject inbound
        else:
            # Remote wins: terminate older connection and accept this inbound one
            task = asyncio.create_task(existing.close())
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
            return False


def _should_drop_inbound_collision(local_device_id: Any, remote_device_id: Any) -> bool:
    """Deterministic tie-breaker comparison between two device IDs.

    Enforces type conversion to str to prevent TypeError on mixed/non-string types.
    Returns True if inbound collision should be dropped (str(local) < str(remote)),
    otherwise False.
    """
    return str(local_device_id) < str(remote_device_id)


# Alias for TcpStreamManager
TcpStreamManager = TCPStreamManager

__all__ = [
    "ConnectionRole",
    "configure_stream_socket",
    "TCPConnection",
    "TCPStreamManager",
    "TcpStreamManager",
    "_should_drop_inbound_collision",
]


"""tests.mock.mock_device
~~~~~~~~~~~~~~~~~~~~~~~~
Virtual KDE Connect Android device simulator for automated integration and E2E testing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from pengulink.core.exceptions import (
    ConnectionClosedError,
    ConnectionTimeoutError,
    LimitOverrunError,
    MalformedPacketError,
    TLSUpgradeError,
)
from pengulink.crypto.certificate import (
    dump_certificate_pem,
    dump_private_key_pem,
    generate_device_key,
    generate_self_signed_cert,
    get_certificate_fingerprint,
    load_certificate_pem,
)
from pengulink.crypto.pairing import (
    compute_verification_code,
    create_pair_packet,
    validate_pairing_timestamp,
)
from pengulink.crypto.tls import (
    create_paired_ssl_context,
    create_tofu_ssl_context,
    verify_peer_certificate,
)
from pengulink.network.packet import (
    MAX_PACKET_SIZE,
    IdentityBody,
    Packet,
    PacketEngine,
)
from pengulink.network.tcp import ConnectionRole, TCPConnection, configure_stream_socket
from pengulink.plugins.battery import (
    create_battery_request_packet,
    create_battery_status_packet,
)
from pengulink.plugins.notification import (
    ANDROID_ID_PREFIX,
    NOTIFICATION_ACTION_TYPE,
    NOTIFICATION_PACKET_TYPE,
    NOTIFICATION_REPLY_TYPE,
    NOTIFICATION_REQUEST_TYPE,
    sanitize_notification_id,
)
from pengulink.plugins.ping import create_ping_packet

logger = logging.getLogger("pengulink.mock_device")


@dataclass
class MockDeviceConfig:
    """Configuration profile for MockKdeConnectDevice."""

    device_id: str = field(default_factory=lambda: f"mock_{uuid.uuid4().hex[:16]}")
    device_name: str = "Mock Pixel 8"
    device_type: str = "phone"
    protocol_version: int = 7
    incoming_capabilities: list[str] = field(
        default_factory=lambda: [
            "kdeconnect.ping",
            "kdeconnect.battery",
            "kdeconnect.battery.request",
            "kdeconnect.notification",
            "kdeconnect.connectivity_report",
            "kdeconnect.connectivity_report.request",
            "kdeconnect.clipboard",
            "kdeconnect.clipboard.connect",
        ]
    )
    outgoing_capabilities: list[str] = field(
        default_factory=lambda: [
            "kdeconnect.ping",
            "kdeconnect.battery",
            "kdeconnect.battery.request",
            "kdeconnect.notification",
            "kdeconnect.notification.request",
            "kdeconnect.notification.action",
            "kdeconnect.notification.reply",
            "kdeconnect.connectivity_report",
            "kdeconnect.connectivity_report.request",
            "kdeconnect.clipboard",
            "kdeconnect.clipboard.connect",
        ]
    )
    tcp_port_min: int = 1716
    tcp_port_max: int = 1764
    udp_port: int = 1716
    auto_pair: bool = True
    auto_reject_pair: bool = False
    auto_pong: bool = False
    auto_battery_reply: bool = True
    auto_connectivity_reply: bool = True


@dataclass
class ChaosConfig:
    """Declarative configuration for fault and chaos injection."""

    drop_on_connect: bool = False
    drop_after_preamble: bool = False
    drop_mid_tls: bool = False
    drop_after_tls: bool = False
    drop_with_rst: bool = False

    corrupt_preamble: bool = False
    corrupt_next_n_packets: int = 0
    corruption_payload: Optional[bytes] = None

    send_oversized_packet: bool = False
    oversized_bytes: int = 140000

    send_truncated_packet: bool = False

    spoof_cert_cn: Optional[str] = None
    use_untrusted_cert: bool = False
    pairing_timestamp_offset: float = 0.0
    auto_reject_pairing: bool = False
    ignore_pairing: bool = False

    packet_interceptor: Optional[Callable[[Packet], Optional[Union[Packet, bytes]]]] = None


class MockKdeConnectDevice:
    """Automated KDE Connect device simulator representing an Android smartphone or tablet."""

    def __init__(
        self,
        config: Optional[MockDeviceConfig] = None,
        chaos_config: Optional[ChaosConfig] = None,
        private_key: Optional[rsa.RSAPrivateKey] = None,
        certificate: Optional[x509.Certificate] = None,
    ) -> None:
        self.config = config or MockDeviceConfig()
        self.chaos_config = chaos_config or ChaosConfig()

        # Crypto & Identity
        self.private_key = private_key or generate_device_key()
        cert_cn = self.chaos_config.spoof_cert_cn or self.config.device_id
        self.certificate = certificate or generate_self_signed_cert(self.private_key, cert_cn)
        self.key_pem = dump_private_key_pem(self.private_key)
        self.cert_pem = dump_certificate_pem(self.certificate)
        self.fingerprint = get_certificate_fingerprint(self.certificate)

        # Peer tracking: peer_device_id -> peer_cert_pem (bytes)
        self.paired_peers: dict[str, bytes] = {}

        # Sockets & Networking
        self.bound_tcp_port: Optional[int] = None
        self._server: Optional[asyncio.Server] = None
        self._udp_transport: Optional[asyncio.DatagramTransport] = None
        self.active_connection: Optional[TCPConnection] = None
        self._active_connections: list[TCPConnection] = []

        # Background task & waiter tracking (Zero Leaks)
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._packet_waiters: list[tuple[str, asyncio.Future[Packet]]] = []
        self._received_packets: list[Packet] = []
        self._running: bool = False

        # State Telemetry
        self.last_verification_code: Optional[str] = None
        self.last_ping_message: Optional[str] = None
        self.current_battery_charge: int = 85
        self.is_battery_charging: bool = False
        self.last_clipboard_content: Optional[str] = None
        self.last_clipboard_timestamp: Optional[int] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_connected(self) -> bool:
        return self.active_connection is not None and not self.active_connection.is_closed

    @property
    def device_id(self) -> str:
        return self.config.device_id

    @property
    def device_name(self) -> str:
        return self.config.device_name

    # ------------------------------------------------------------------
    # Lifecycle Management
    # ------------------------------------------------------------------
    async def start(
        self,
        listen_tcp: bool = True,
        port: int = 0,
    ) -> None:
        """Spin up mock device services and bind TCP listener if requested."""
        if self._running:
            return

        self._running = True

        if listen_tcp:
            await self._start_tcp_server(port=port)

    async def _start_tcp_server(self, port: int = 0) -> None:
        """Start TCP listener, scanning port range or binding port 0."""
        loop = asyncio.get_running_loop()

        if port == 0:
            # Bind ephemeral port
            server = await asyncio.start_server(
                self._handle_inbound_socket,
                host="127.0.0.1",
                port=0,
                limit=MAX_PACKET_SIZE,
                reuse_address=True,
            )
            self._server = server
            self.bound_tcp_port = server.sockets[0].getsockname()[1]
            logger.info("MockDevice listening on ephemeral TCP port %d", self.bound_tcp_port)
            return

        # Scan configured range
        for candidate in range(self.config.tcp_port_min, self.config.tcp_port_max + 1):
            try:
                server = await asyncio.start_server(
                    self._handle_inbound_socket,
                    host="127.0.0.1",
                    port=candidate,
                    limit=MAX_PACKET_SIZE,
                    reuse_address=True,
                )
                self._server = server
                self.bound_tcp_port = candidate
                logger.info("MockDevice bound TCP port %d", self.bound_tcp_port)
                return
            except OSError:
                continue

        # If all range ports occupied, fall back to ephemeral
        server = await asyncio.start_server(
            self._handle_inbound_socket,
            host="127.0.0.1",
            port=0,
            limit=MAX_PACKET_SIZE,
            reuse_address=True,
        )
        self._server = server
        self.bound_tcp_port = server.sockets[0].getsockname()[1]
        logger.warning(
            "MockDevice range %d-%d occupied; bound ephemeral TCP port %d",
            self.config.tcp_port_min,
            self.config.tcp_port_max,
            self.bound_tcp_port,
        )

    async def stop(self) -> None:
        """Completely teardown sockets, active connections, and background tasks without leaks."""
        if not self._running:
            return

        self._running = False

        # 1. Close TCP server
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass
            self._server = None

        # 2. Close UDP transport if any
        if self._udp_transport is not None:
            self._udp_transport.close()
            self._udp_transport = None

        # 3. Close active TCP connections
        conns_to_close = list(self._active_connections)
        self._active_connections.clear()
        self.active_connection = None

        for conn in conns_to_close:
            try:
                await asyncio.shield(conn.close())
            except Exception:
                pass

        # 4. Cancel and drain all background tasks
        tasks = list(self._background_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self._background_tasks.clear()

        # 5. Cancel any pending packet waiter futures
        for _, fut in self._packet_waiters:
            if not fut.done():
                fut.cancel()
        self._packet_waiters.clear()

        self.bound_tcp_port = None
        logger.info("MockDevice stopped cleanly with zero resource leaks.")

    async def __aenter__(self) -> MockKdeConnectDevice:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Discovery Operations (UDP)
    # ------------------------------------------------------------------
    def build_identity_packet(self, target_device_id: Optional[str] = None) -> Packet:
        """Build kdeconnect.identity packet containing current bound TCP port."""
        body = IdentityBody(
            deviceId=self.config.device_id,
            deviceName=self.config.device_name,
            deviceType=self.config.device_type,
            protocolVersion=self.config.protocol_version,
            incomingCapabilities=self.config.incoming_capabilities,
            outgoingCapabilities=self.config.outgoing_capabilities,
            tcpPort=self.bound_tcp_port or self.config.tcp_port_min,
        )
        return Packet.create_identity(body)

    async def broadcast_identity(
        self,
        target_ip: str = "127.0.0.1",
        target_port: int = 1716,
    ) -> None:
        """Transmit discovery datagram to target UDP endpoint."""
        packet = self.build_identity_packet()
        wire_bytes = PacketEngine.serialize(packet)

        loop = asyncio.get_running_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)

        try:
            await loop.sock_sendto(sock, wire_bytes, (target_ip, target_port))
        finally:
            sock.close()

    # ------------------------------------------------------------------
    # Connection Handshake & Inverted mTLS
    # ------------------------------------------------------------------
    async def connect_to_daemon(
        self,
        host: str = "127.0.0.1",
        port: int = 1716,
        target_device_id: str = "pengulink_local",
        timeout: float = 5.0,
    ) -> TCPConnection:
        """Initiate outbound TCP connection to PenguLink daemon.

        KDE Connect Inverted TLS Rule:
          Outbound connector -> server_side=True (PROTOCOL_TLS_SERVER)
        """
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, limit=MAX_PACKET_SIZE),
            timeout=timeout,
        )
        configure_stream_socket(writer)

        conn = TCPConnection(
            reader=reader,
            writer=writer,
            role=ConnectionRole.OUTBOUND,
            peer_ip=host,
            peer_port=port,
        )
        conn.remote_device_id = target_device_id
        self._register_connection(conn)

        if self.chaos_config.drop_on_connect:
            await self._apply_drop(conn)
            raise ConnectionClosedError("Chaos: dropped on connect")

        # 1. Transmit cleartext identity preamble
        if self.chaos_config.corrupt_preamble:
            await conn.write_line(b"INVALID_PREAMBLE_CORRUPT\n")
        else:
            ident_packet = self.build_identity_packet()
            await conn.write_line(PacketEngine.serialize(ident_packet))

        if self.chaos_config.drop_after_preamble:
            await self._apply_drop(conn)
            raise ConnectionClosedError("Chaos: dropped after preamble")

        # 2. Upgrade to TLS (Inverted: server_side=True)
        is_paired = target_device_id in self.paired_peers
        if is_paired:
            ssl_ctx = create_paired_ssl_context(
                local_cert_pem=self.cert_pem,
                local_key_pem=self.key_pem,
                peer_cert_pem=self.paired_peers[target_device_id],
                server_side=True,
            )
        else:
            ssl_ctx = create_tofu_ssl_context(
                local_cert_pem=self.cert_pem,
                local_key_pem=self.key_pem,
                server_side=True,
            )

        if self.chaos_config.drop_mid_tls:
            await self._apply_drop(conn)
            raise ConnectionClosedError("Chaos: dropped mid TLS")

        await conn.upgrade_to_tls(ssl_ctx, server_side=True, timeout=timeout)

        if self.chaos_config.drop_after_tls:
            await self._apply_drop(conn)
            raise ConnectionClosedError("Chaos: dropped after TLS")

        self.active_connection = conn
        self._spawn_read_loop(conn)
        return conn

    async def _handle_inbound_socket(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle incoming TCP connection from PenguLink.

        KDE Connect Inverted TLS Rule:
          Inbound listener -> server_side=False (PROTOCOL_TLS_CLIENT)
        """
        configure_stream_socket(writer)
        peer_name = writer.get_extra_info("peername") or ("127.0.0.1", 0)

        conn = TCPConnection(
            reader=reader,
            writer=writer,
            role=ConnectionRole.INBOUND,
            peer_ip=peer_name[0],
            peer_port=peer_name[1],
        )
        self._register_connection(conn)

        try:
            if self.chaos_config.drop_on_connect:
                await self._apply_drop(conn)
                return

            # 1. Read cleartext identity preamble from connecting peer
            line = await conn.read_line(timeout=5.0)
            if not line:
                await conn.close()
                return

            preamble_packet = PacketEngine.parse(line)
            remote_id = preamble_packet.body.get("deviceId")
            conn.remote_device_id = remote_id

            if self.chaos_config.drop_after_preamble:
                await self._apply_drop(conn)
                return

            # 2. STARTTLS Upgrade (Inverted: server_side=False)
            is_paired = remote_id in self.paired_peers if remote_id else False
            if is_paired and remote_id:
                ssl_ctx = create_paired_ssl_context(
                    local_cert_pem=self.cert_pem,
                    local_key_pem=self.key_pem,
                    peer_cert_pem=self.paired_peers[remote_id],
                    server_side=False,
                )
            else:
                ssl_ctx = create_tofu_ssl_context(
                    local_cert_pem=self.cert_pem,
                    local_key_pem=self.key_pem,
                    server_side=False,
                )

            if self.chaos_config.drop_mid_tls:
                await self._apply_drop(conn)
                return

            await conn.upgrade_to_tls(ssl_ctx, server_side=False, timeout=5.0)

            if self.chaos_config.drop_after_tls:
                await self._apply_drop(conn)
                return

            self.active_connection = conn
            self._spawn_read_loop(conn)

        except Exception as exc:
            logger.debug("Error in MockDevice inbound connection handling: %s", exc)
            await asyncio.shield(conn.close())

    def _register_connection(self, conn: TCPConnection) -> None:
        self._active_connections.append(conn)

        def _cleanup(c: TCPConnection) -> None:
            if c in self._active_connections:
                self._active_connections.remove(c)
            if self.active_connection is c:
                self.active_connection = None

        conn.add_close_callback(_cleanup)

    def _spawn_read_loop(self, conn: TCPConnection) -> None:
        task = asyncio.create_task(self._connection_read_loop(conn))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _connection_read_loop(self, conn: TCPConnection) -> None:
        """Stream reader loop dispatching packets and resolving waiters."""
        try:
            while self._running and not conn.is_closed:
                line = await conn.read_line(timeout=None)
                if not line:
                    break

                try:
                    packet = PacketEngine.parse(line)
                except Exception as exc:
                    logger.debug("MockDevice received invalid packet: %s", exc)
                    continue

                self._received_packets.append(packet)
                self._dispatch_packet_waiters(packet)
                await self._handle_protocol_packet(packet, conn)

        except (asyncio.CancelledError, ConnectionClosedError):
            pass
        except Exception as exc:
            logger.debug("MockDevice read loop terminated: %s", exc)
        finally:
            await conn.close()

    def _dispatch_packet_waiters(self, packet: Packet) -> None:
        remaining: list[tuple[str, asyncio.Future[Packet]]] = []
        for ptype, fut in self._packet_waiters:
            if not fut.done() and (ptype == "*" or packet.type == ptype):
                fut.set_result(packet)
            elif not fut.done():
                remaining.append((ptype, fut))
        self._packet_waiters = remaining

    async def _handle_protocol_packet(self, packet: Packet, conn: TCPConnection) -> None:
        """Handle incoming protocol packets (pair, ping, battery, notification)."""
        # 1. Pairing negotiation
        if packet.type == "kdeconnect.pair":
            is_pair_request = packet.body.get("pair") is True
            peer_id = conn.remote_device_id or "unknown_peer"

            if is_pair_request:
                ts = packet.body.get("timestamp", int(time.time() * 1000))
                # Derive verification code if peer certificate is available
                peer_cert_der = None
                try:
                    sock = conn.writer.get_extra_info("ssl_object")
                    if sock:
                        peer_cert_der = sock.getpeercert(binary_form=True)
                except Exception:
                    pass

                if peer_cert_der:
                    peer_cert = x509.load_der_x509_certificate(peer_cert_der)
                    peer_pubkey_pem = (
                        peer_cert.public_key()
                        .public_bytes(
                            encoding=serialization.Encoding.PEM,
                            format=serialization.PublicFormat.SubjectPublicKeyInfo,
                        )
                        .decode("utf-8")
                    )
                    local_pubkey_pem = (
                        self.certificate.public_key()
                        .public_bytes(
                            encoding=serialization.Encoding.PEM,
                            format=serialization.PublicFormat.SubjectPublicKeyInfo,
                        )
                        .decode("utf-8")
                    )
                    self.last_verification_code = compute_verification_code(
                        local_pubkey_pem, peer_pubkey_pem, ts
                    )

                if self.chaos_config.auto_reject_pairing or self.config.auto_reject_pair:
                    reject_pkt = create_pair_packet(pair=False)
                    await self.send_packet(reject_pkt)
                elif self.config.auto_pair and not self.chaos_config.ignore_pairing:
                    # Save peer certificate if present
                    if peer_cert_der:
                        peer_cert = x509.load_der_x509_certificate(peer_cert_der)
                        self.paired_peers[peer_id] = dump_certificate_pem(peer_cert)
                    accept_pkt = create_pair_packet(pair=True, timestamp=ts)
                    await self.send_packet(accept_pkt)

        # 2. Ping handling
        elif packet.type == "kdeconnect.ping":
            self.last_ping_message = packet.body.get("message")
            if self.config.auto_pong:
                pong = create_ping_packet(message="Pong from MockDevice")
                await self.send_packet(pong)

        # 3. Battery status query handling
        elif packet.type == "kdeconnect.battery.request":
            if self.config.auto_battery_reply:
                bat_pkt = create_battery_status_packet(
                    current_charge=self.current_battery_charge,
                    is_charging=self.is_battery_charging,
                )
                await self.send_packet(bat_pkt)

        # 4. Connectivity status query handling
        elif packet.type == "kdeconnect.connectivity_report.request":
            if self.config.auto_connectivity_reply:
                await self.send_connectivity_report()

        # 5. Clipboard packet handling
        elif packet.type in ("kdeconnect.clipboard", "kdeconnect.clipboard.connect"):
            self.last_clipboard_content = packet.body.get("content")
            self.last_clipboard_timestamp = packet.body.get("timestamp")

    # ------------------------------------------------------------------
    # Packet Transmission & Helpers
    # ------------------------------------------------------------------
    async def send_packet(self, packet: Packet) -> None:
        """Serialize and send packet over active connection with interceptor support."""
        if not self.active_connection or self.active_connection.is_closed:
            raise ConnectionClosedError("No active connection to send packet")

        # 1. Packet Interceptor Hook
        if self.chaos_config.packet_interceptor:
            res = self.chaos_config.packet_interceptor(packet)
            if res is None:
                return  # Dropped by interceptor
            if isinstance(res, bytes):
                await self.active_connection.write_line(res)
                return
            packet = res

        # 2. Corruption Chaos
        if self.chaos_config.corrupt_next_n_packets > 0:
            self.chaos_config.corrupt_next_n_packets -= 1
            payload = self.chaos_config.corruption_payload or b'{"corrupt": true\n'
            await self.active_connection.write_line(payload)
            return

        wire_bytes = PacketEngine.serialize(packet)
        await self.active_connection.write_line(wire_bytes)

    async def wait_for_packet(self, packet_type: str, timeout: float = 5.0) -> Packet:
        """Wait for an incoming packet of specified type (or '*' for any)."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Packet] = loop.create_future()
        self._packet_waiters.append((packet_type, fut))

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            if (packet_type, fut) in self._packet_waiters:
                self._packet_waiters.remove((packet_type, fut))
            raise ConnectionTimeoutError(
                f"Timed out waiting for packet '{packet_type}'",
                timeout_seconds=timeout,
            ) from exc

    @property
    def received_packets(self) -> list[Packet]:
        """Return history of all received packets in chronological order."""
        return list(self._received_packets)

    def get_received_packets(self, packet_type: Optional[str] = None) -> list[Packet]:
        """Return history of received packets, optionally filtered by type."""
        if packet_type is None:
            return list(self._received_packets)
        return [p for p in self._received_packets if p.type == packet_type]

    # ------------------------------------------------------------------
    # High-Level Android Feature Simulators
    # ------------------------------------------------------------------
    async def send_ping(self, message: Optional[str] = "Ping from Mock Android") -> Packet:
        """Transmit kdeconnect.ping packet."""
        pkt = create_ping_packet(message=message)
        await self.send_packet(pkt)
        return pkt

    async def send_battery_update(
        self,
        current_charge: int = 85,
        is_charging: bool = False,
        threshold_event: int = 0,
    ) -> Packet:
        """Transmit kdeconnect.battery packet."""
        self.current_battery_charge = current_charge
        self.is_battery_charging = is_charging
        pkt = create_battery_status_packet(
            current_charge=current_charge,
            is_charging=is_charging,
            threshold_event=threshold_event,
        )
        await self.send_packet(pkt)
        return pkt

    async def send_notification(
        self,
        notification_id: str = "1001",
        app_name: str = "WhatsApp",
        title: str = "Mom",
        text: str = "Dinner is ready! Use code 492018.",
        ticker: str = "New message",
        actions: Optional[list[str]] = None,
        request_reply_id: Optional[str] = None,
    ) -> Packet:
        """Transmit mobile notification, applying Android ID prefix if absent."""
        if not notification_id.startswith(ANDROID_ID_PREFIX):
            full_id = f"{ANDROID_ID_PREFIX}{notification_id}"
        else:
            full_id = notification_id

        body = {
            "id": full_id,
            "appName": app_name,
            "title": title,
            "text": text,
            "ticker": ticker,
            "isClearable": True,
        }
        if actions:
            body["actions"] = actions
        if request_reply_id:
            body["requestReplyId"] = request_reply_id

        pkt = Packet(type=NOTIFICATION_PACKET_TYPE, body=body)
        await self.send_packet(pkt)
        return pkt

    async def cancel_notification(self, notification_id: str) -> Packet:
        """Transmit isCancel notification packet."""
        if not notification_id.startswith(ANDROID_ID_PREFIX):
            full_id = f"{ANDROID_ID_PREFIX}{notification_id}"
        else:
            full_id = notification_id

        pkt = Packet(
            type=NOTIFICATION_PACKET_TYPE,
            body={
                "id": full_id,
                "isCancel": True,
            },
        )
        await self.send_packet(pkt)
        return pkt

    async def send_connectivity_report(
        self,
        signal_strengths: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Packet:
        """Transmit kdeconnect.connectivity_report packet."""
        if signal_strengths is None:
            signal_strengths = {
                "0": {
                    "networkType": "5G",
                    "signalStrength": 4,
                }
            }

        pkt = Packet(
            type="kdeconnect.connectivity_report",
            body={"signalStrengths": signal_strengths},
        )
        await self.send_packet(pkt)
        return pkt

    async def send_clipboard(
        self,
        content: str,
        timestamp: Optional[int] = None,
    ) -> Packet:
        """Transmit kdeconnect.clipboard packet over active TLS stream."""
        ts = timestamp if timestamp is not None else int(time.time() * 1000)
        packet = Packet(
            id=ts,
            type="kdeconnect.clipboard",
            body={"content": content, "timestamp": ts},
        )
        await self.send_packet(packet)
        return packet

    async def send_clipboard_connect(
        self,
        content: str,
        timestamp: Optional[int] = None,
    ) -> Packet:
        """Transmit kdeconnect.clipboard.connect packet for initial sync over active TLS stream."""
        ts = timestamp if timestamp is not None else int(time.time() * 1000)
        packet = Packet(
            id=ts,
            type="kdeconnect.clipboard.connect",
            body={"content": content, "timestamp": ts},
        )
        await self.send_packet(packet)
        return packet

    async def initiate_pairing(self, target_pubkey_pem: Optional[str] = None) -> str:
        """Send kdeconnect.pair request and compute/return verification code."""
        ts = int(time.time() * 1000)
        if self.chaos_config.pairing_timestamp_offset != 0.0:
            ts += int(self.chaos_config.pairing_timestamp_offset * 1000)

        if target_pubkey_pem:
            local_pubkey_pem = (
                self.certificate.public_key()
                .public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                .decode("utf-8")
            )
            self.last_verification_code = compute_verification_code(
                local_pubkey_pem, target_pubkey_pem, ts
            )

        pair_pkt = create_pair_packet(pair=True, timestamp=ts)
        await self.send_packet(pair_pkt)
        return self.last_verification_code or ""

    # ------------------------------------------------------------------
    # Chaos & Fault Injection Interface
    # ------------------------------------------------------------------
    def configure_chaos(self, **kwargs: Any) -> None:
        """Update declarative chaos options dynamically."""
        for k, v in kwargs.items():
            if hasattr(self.chaos_config, k):
                setattr(self.chaos_config, k, v)

    def reset_chaos(self) -> None:
        """Reset chaos options to default."""
        self.chaos_config = ChaosConfig()

    async def _apply_drop(self, conn: TCPConnection) -> None:
        """Terminate connection using FIN or RST as configured."""
        if self.chaos_config.drop_with_rst:
            try:
                sock = conn.writer.get_extra_info("socket")
                if sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            except Exception:
                pass
        await conn.close()

    async def drop_connection(self, reset: bool = False) -> None:
        """Imperatively terminate active TCP connection."""
        if not self.active_connection:
            return
        if reset:
            try:
                sock = self.active_connection.writer.get_extra_info("socket")
                if sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            except Exception:
                pass
        await self.active_connection.close()
        self.active_connection = None

    async def send_raw_bytes(self, data: bytes) -> None:
        """Imperatively transmit raw bytes over active stream without framing."""
        if not self.active_connection or self.active_connection.is_closed:
            raise ConnectionClosedError("No active connection to write raw bytes")
        self.active_connection.writer.write(data)
        await self.active_connection.writer.drain()

    async def send_malformed_json(
        self,
        corrupt_payload: Union[str, bytes] = b"INVALID_JSON_CORRUPT\n",
    ) -> None:
        """Imperatively transmit corrupted JSON line terminated with newline."""
        if isinstance(corrupt_payload, str):
            corrupt_payload = corrupt_payload.encode("utf-8")
        if not corrupt_payload.endswith(b"\n"):
            corrupt_payload = corrupt_payload + b"\n"
        await self.send_raw_bytes(corrupt_payload)

    async def send_oversized_packet(self, size: int = 140000) -> None:
        """Imperatively transmit packet exceeding 128 KB buffer guard."""
        garbage = b"A" * size + b"\n"
        await self.send_raw_bytes(garbage)

    async def send_truncated_packet(
        self,
        partial_payload: bytes = b'{"type":"kdeconnect.ping"',
    ) -> None:
        """Imperatively transmit partial packet without trailing newline, then drop stream."""
        await self.send_raw_bytes(partial_payload)
        await self.drop_connection(reset=False)

    async def simulate_flapping(
        self,
        target_host: str = "127.0.0.1",
        target_port: int = 1716,
        cycles: int = 10,
        interval: float = 0.02,
    ) -> None:
        """Rapidly connect and disconnect in a loop to test peer resilience."""
        for _ in range(cycles):
            try:
                reader, writer = await asyncio.open_connection(target_host, target_port)
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            if interval > 0:
                await asyncio.sleep(interval)


# Convenience Alias
MockDevice = MockKdeConnectDevice

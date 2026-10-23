"""tests.mock.mock_device
~~~~~~~~~~~~~~~~~~~~~~~~
Virtual KDE Connect Android device simulator for automated integration and E2E testing.
"""

from __future__ import annotations

import asyncio
import json
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
            "kdeconnect.findmyphone.request",
            "kdeconnect.findmyphone",
            "kdeconnect.share.request",
            "kdeconnect.share.request.update",
            "kdeconnect.share",
            "kdeconnect.contacts.request_all_uids_timestamps",
            "kdeconnect.contacts.request_vcards_by_uid",
            "kdeconnect.contacts",
            "kdeconnect.mpris",
            "kdeconnect.mpris.request",
            # RemoteCommands Capabilities
            "kdeconnect.remotecommands",
            "kdeconnect.remotecommands.request",
            "kdeconnect.runcommand",
            "kdeconnect.runcommand.request",
            "kdeconnect.runcommand.output",
            "kdeconnect.remoteforcecommand",
            # SystemVolume & RemoteSystemVolume Capabilities
            "kdeconnect.systemvolume",
            "kdeconnect.systemvolume.request",
            # Telephony & PauseMusic Capabilities
            "kdeconnect.telephony",
            "kdeconnect.telephony.request_mute",
            "kdeconnect.telephony.request",
            "kdeconnect.pausemusic",
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
            "kdeconnect.findmyphone.request",
            "kdeconnect.findmyphone",
            "kdeconnect.share.request",
            "kdeconnect.share.request.update",
            "kdeconnect.share",
            "kdeconnect.contacts.response_uids_timestamps",
            "kdeconnect.contacts.response_vcards",
            "kdeconnect.contacts",
            "kdeconnect.mpris",
            "kdeconnect.mpris.request",
            # RemoteCommands Capabilities
            "kdeconnect.remotecommands",
            "kdeconnect.remotecommands.request",
            "kdeconnect.runcommand",
            "kdeconnect.runcommand.request",
            "kdeconnect.runcommand.output",
            "kdeconnect.remoteforcecommand",
            # SystemVolume & RemoteSystemVolume Capabilities
            "kdeconnect.systemvolume",
            "kdeconnect.systemvolume.request",
            # Telephony & PauseMusic Capabilities
            "kdeconnect.telephony",
            "kdeconnect.telephony.request_mute",
            "kdeconnect.telephony.request",
            "kdeconnect.pausemusic",
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
    auto_contacts_reply: bool = True
    auto_mpris_reply: bool = True
    auto_mpris_broadcast: bool = True
    # RemoteCommands Configuration Flags
    auto_commands_reply: bool = True
    auto_command_output: bool = False
    default_command_list_format: str = "dict"
    # SystemVolume Configuration Flags
    auto_systemvolume_reply: bool = True
    auto_systemvolume_broadcast: bool = True


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

    contacts_ignore_requests: bool = False
    contacts_delay: float = 0.0
    contacts_malform_vcards: bool = False
    contacts_drop_vcard_response: bool = False
    contacts_chunk_size: Optional[int] = None
    contacts_omit_uids: bool = False
    contacts_send_oversized_vcard: bool = False

    mpris_ignore_requests: bool = False
    mpris_delay: float = 0.0
    mpris_corrupt_updates: bool = False
    mpris_drop_player_list: bool = False

    # RemoteCommands Fault & Chaos Injection
    commands_ignore_requests: bool = False
    commands_delay: float = 0.0
    commands_corrupt_command_list: bool = False
    commands_malformed_json_string: bool = False
    commands_drop_output: bool = False

    # SystemVolume Fault & Chaos Injection
    systemvolume_ignore_requests: bool = False
    systemvolume_delay: float = 0.0
    systemvolume_drop_sink_list: bool = False
    systemvolume_corrupt_sink_list: bool = False


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

        # Contacts Simulation State
        self._mock_contacts: Dict[str, Dict[str, Any]] = {}
        self._contacts_requested_uids: List[str] = []

        # MPRIS Simulation State
        self._mock_players: Dict[str, Dict[str, Any]] = {}
        self._mpris_requests: List[Packet] = []

        # RemoteCommands Simulation State
        self._mock_commands: Dict[str, Dict[str, str]] = {
            "cmd_lock": {
                "key": "cmd_lock",
                "name": "Lock Screen",
                "command": "loginctl lock-session",
            },
            "cmd_suspend": {
                "key": "cmd_suspend",
                "name": "Suspend System",
                "command": "systemctl suspend",
            },
            "cmd_ring": {
                "key": "cmd_ring",
                "name": "Find Device Ring",
                "command": "echo 'Ring'",
            },
        }
        self._executed_commands: List[str] = []
        self._command_exec_requests: List[Packet] = []
        self._command_stop_requests: List[Packet] = []
        self._received_command_outputs: List[Packet] = []

        # SystemVolume Simulation State
        self._mock_sinks: Dict[str, Dict[str, Any]] = {
            "speaker": {
                "name": "speaker",
                "description": "Internal Speaker",
                "muted": False,
                "volume": 32768,
                "maxVolume": 65536,
                "enabled": True,
            },
            "headphones": {
                "name": "headphones",
                "description": "Wired Headphones",
                "muted": False,
                "volume": 49152,
                "maxVolume": 65536,
                "enabled": False,
            },
        }
        self._systemvolume_requests: List[Packet] = []
        self._systemvolume_announcements: List[Packet] = []

        # Telephony Simulation State
        self._telephony_actions: List[Packet] = []
        self._last_telephony_call: Optional[Dict[str, Any]] = None


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
            if not fut.done() and (
                ptype == "*"
                or packet.type == ptype
                or (ptype == "kdeconnect.findmyphone.request" and packet.type == "kdeconnect.findmyphone")
            ):
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

        # 6. Contacts UID and timestamp request handling
        elif packet.type == "kdeconnect.contacts.request_all_uids_timestamps":
            if self.config.auto_contacts_reply and not self.chaos_config.contacts_ignore_requests:
                if self.chaos_config.contacts_delay > 0:
                    await asyncio.sleep(self.chaos_config.contacts_delay)

                uids = list(self._mock_contacts.keys())
                body: Dict[str, Any] = {"uids": uids}
                for uid in uids:
                    body[uid] = self._mock_contacts[uid]["timestamp"]

                resp_pkt = Packet(
                    type="kdeconnect.contacts.response_uids_timestamps",
                    body=body,
                )
                await self.send_packet(resp_pkt)

        # 7. Contacts vCard batch request handling
        elif packet.type == "kdeconnect.contacts.request_vcards_by_uid":
            requested_uids = list(packet.body.get("uids", []))
            self._contacts_requested_uids.extend(requested_uids)

            if self.config.auto_contacts_reply and not self.chaos_config.contacts_ignore_requests:
                if self.chaos_config.contacts_delay > 0:
                    await asyncio.sleep(self.chaos_config.contacts_delay)

                if not self.chaos_config.contacts_drop_vcard_response:
                    chunk_size = self.chaos_config.contacts_chunk_size or len(requested_uids)
                    chunks = [
                        requested_uids[i : i + chunk_size]
                        for i in range(0, len(requested_uids), max(1, chunk_size))
                    ] if requested_uids else [[]]

                    for chunk in chunks:
                        body: Dict[str, Any] = {
                            "uids": [] if self.chaos_config.contacts_omit_uids else list(chunk)
                        }
                        for uid in chunk:
                            if uid in self._mock_contacts:
                                vcard_str = self._mock_contacts[uid]["vcard"]
                                if self.chaos_config.contacts_malform_vcards:
                                    vcard_str = "INVALID_VCARD_BODY_CORRUPT"
                                body[uid] = vcard_str

                        resp_pkt = Packet(
                            type="kdeconnect.contacts.response_vcards",
                            body=body,
                        )
                        await self.send_packet(resp_pkt)

        # 8. MPRIS Media Control Request handling
        elif packet.type == "kdeconnect.mpris.request":
            self._mpris_requests.append(packet)

            if self.chaos_config.mpris_ignore_requests:
                return

            if self.chaos_config.mpris_delay > 0:
                await asyncio.sleep(self.chaos_config.mpris_delay)

            if not self.config.auto_mpris_reply:
                return

            body = packet.body or {}
            target_player = body.get("player")

            # 1. Handle requestPlayerList
            if body.get("requestPlayerList"):
                if not self.chaos_config.mpris_drop_player_list:
                    player_list = list(self._mock_players.keys())
                    resp_pkt = Packet(
                        type="kdeconnect.mpris",
                        body={"playerList": player_list, "supportAlbumArtPayload": True},
                    )
                    await self.send_packet(resp_pkt)

            # 2. Handle requestNowPlaying
            if body.get("requestNowPlaying") and target_player in self._mock_players:
                player_state = self._mock_players[target_player].copy()
                player_state.pop("last_pos_timestamp", None)
                resp_pkt = Packet(type="kdeconnect.mpris", body=player_state)
                await self.send_packet(resp_pkt)

            # 3. Handle requestVolume
            if body.get("requestVolume") and target_player in self._mock_players:
                vol = self._mock_players[target_player].get("volume", 50)
                resp_pkt = Packet(
                    type="kdeconnect.mpris",
                    body={"player": target_player, "volume": vol},
                )
                await self.send_packet(resp_pkt)

            # 4. Handle Transport Action (Play, Pause, PlayPause, Stop, Next, Previous)
            action = body.get("action")
            if action and target_player in self._mock_players:
                p = self._mock_players[target_player]
                now = time.time()
                if action == "Play":
                    p["isPlaying"] = True
                    p["last_pos_timestamp"] = now
                elif action == "Pause":
                    if p.get("isPlaying", False):
                        elapsed_ms = int((now - p.get("last_pos_timestamp", now)) * 1000)
                        p["pos"] = min(p.get("length", 0), p.get("pos", 0) + elapsed_ms)
                    p["isPlaying"] = False
                    p["last_pos_timestamp"] = now
                elif action == "PlayPause":
                    if p.get("isPlaying", False):
                        elapsed_ms = int((now - p.get("last_pos_timestamp", now)) * 1000)
                        p["pos"] = min(p.get("length", 0), p.get("pos", 0) + elapsed_ms)
                        p["isPlaying"] = False
                    else:
                        p["isPlaying"] = True
                        p["last_pos_timestamp"] = now
                elif action == "Stop":
                    p["isPlaying"] = False
                    p["pos"] = 0
                    p["last_pos_timestamp"] = now
                elif action == "Next":
                    p["title"] = f"{p.get('title', 'Track')} (Next)"
                    p["pos"] = 0
                    p["last_pos_timestamp"] = now
                elif action == "Previous":
                    p["pos"] = 0
                    p["last_pos_timestamp"] = now

                if self.config.auto_mpris_broadcast:
                    state_copy = p.copy()
                    state_copy.pop("last_pos_timestamp", None)
                    resp_pkt = Packet(type="kdeconnect.mpris", body=state_copy)
                    await self.send_packet(resp_pkt)

            # 5. Handle setVolume (0..100)
            if "setVolume" in body and target_player in self._mock_players:
                raw_vol = int(body["setVolume"])
                clamped_vol = max(0, min(100, raw_vol))
                self._mock_players[target_player]["volume"] = clamped_vol
                if self.config.auto_mpris_broadcast:
                    resp_pkt = Packet(
                        type="kdeconnect.mpris",
                        body={"player": target_player, "volume": clamped_vol},
                    )
                    await self.send_packet(resp_pkt)

            # 6. Handle Seek (Microseconds: us -> ms)
            if "Seek" in body and target_player in self._mock_players:
                offset_us = int(body["Seek"])
                offset_ms = offset_us // 1000
                p = self._mock_players[target_player]
                cur_pos = p.get("pos", 0)
                max_len = p.get("length", 0)
                new_pos = max(0, min(max_len, cur_pos + offset_ms)) if max_len > 0 else max(0, cur_pos + offset_ms)
                p["pos"] = new_pos
                p["last_pos_timestamp"] = time.time()
                if self.config.auto_mpris_broadcast:
                    resp_pkt = Packet(
                        type="kdeconnect.mpris",
                        body={"player": target_player, "pos": new_pos},
                    )
                    await self.send_packet(resp_pkt)

            # 7. Handle SetPosition (Milliseconds: ms)
            if "SetPosition" in body and target_player in self._mock_players:
                pos_ms = int(body["SetPosition"])
                p = self._mock_players[target_player]
                max_len = p.get("length", 0)
                new_pos = max(0, min(max_len, pos_ms)) if max_len > 0 else max(0, pos_ms)
                p["pos"] = new_pos
                p["last_pos_timestamp"] = time.time()
                if self.config.auto_mpris_broadcast:
                    resp_pkt = Packet(
                        type="kdeconnect.mpris",
                        body={"player": target_player, "pos": new_pos},
                    )
                    await self.send_packet(resp_pkt)

            # 8. Handle setLoopStatus & setShuffle
            if "setLoopStatus" in body and target_player in self._mock_players:
                loop_mode = str(body["setLoopStatus"])
                self._mock_players[target_player]["loopStatus"] = loop_mode
                if self.config.auto_mpris_broadcast:
                    await self.send_packet(Packet(type="kdeconnect.mpris", body={"player": target_player, "loopStatus": loop_mode}))

            if "setShuffle" in body and target_player in self._mock_players:
                shuf = bool(body["setShuffle"])
                self._mock_players[target_player]["shuffle"] = shuf
                if self.config.auto_mpris_broadcast:
                    await self.send_packet(Packet(type="kdeconnect.mpris", body={"player": target_player, "shuffle": shuf}))

        # ------------------------------------------------------------------
        # 9. Remote Commands Request Handling
        # ------------------------------------------------------------------
        elif packet.type in (
            "kdeconnect.runcommand.request",
            "kdeconnect.remotecommands.request",
            "kdeconnect.remoteforcecommand",
        ):
            body = packet.body or {}

            # Case A: Remote command list query
            if body.get("requestCommandList") or (not body.get("key") and packet.type != "kdeconnect.remoteforcecommand"):
                if self.config.auto_commands_reply and not self.chaos_config.commands_ignore_requests:
                    if self.chaos_config.commands_delay > 0:
                        await asyncio.sleep(self.chaos_config.commands_delay)
                    await self.send_command_list(format=self.config.default_command_list_format)

            # Case B: Remote command execution request
            if "key" in body:
                cmd_key = str(body["key"])
                self._executed_commands.append(cmd_key)
                self._command_exec_requests.append(packet)

                if self.chaos_config.commands_ignore_requests:
                    return

                if self.chaos_config.commands_delay > 0:
                    await asyncio.sleep(self.chaos_config.commands_delay)

                if self.config.auto_command_output and not self.chaos_config.commands_drop_output:
                    output_msg = (
                        f"Executed '{cmd_key}' successfully"
                        if cmd_key in self._mock_commands
                        else f"Command '{cmd_key}' not found"
                    )
                    code = 0 if cmd_key in self._mock_commands else 1
                    await self.send_command_output(key=cmd_key, output=output_msg, exit_code=code)

            # Case C: Process stop request
            if body.get("stop"):
                self._command_stop_requests.append(packet)

        # ------------------------------------------------------------------
        # 10. RunCommand Output & Announcement Handling
        # ------------------------------------------------------------------
        elif packet.type == "kdeconnect.runcommand.output":
            self._received_command_outputs.append(packet)

        # ------------------------------------------------------------------
        # 11. SystemVolume Request & Status Handling
        # ------------------------------------------------------------------
        elif packet.type == "kdeconnect.systemvolume.request":
            self._systemvolume_requests.append(packet)
            body = packet.body or {}

            # Case A: Request sink list
            if body.get("requestSinks"):
                if self.config.auto_systemvolume_reply and not self.chaos_config.systemvolume_ignore_requests:
                    if self.chaos_config.systemvolume_delay > 0:
                        await asyncio.sleep(self.chaos_config.systemvolume_delay)
                    if not self.chaos_config.systemvolume_drop_sink_list:
                        await self.send_sink_list()

            # Case B: Volume / Mute / Enabled adjustment
            if "name" in body:
                sink_name = str(body["name"])
                if sink_name in self._mock_sinks:
                    sink = self._mock_sinks[sink_name]
                    if "volume" in body:
                        raw_vol = int(body["volume"])
                        max_vol = int(sink.get("maxVolume", 65536))
                        sink["volume"] = max(0, min(max_vol, raw_vol))
                        sink["muted"] = False  # KDE Connect upstream un-mutes on volume change
                    if "muted" in body:
                        sink["muted"] = bool(body["muted"])
                    if "enabled" in body:
                        sink["enabled"] = bool(body["enabled"])

                    if self.config.auto_systemvolume_broadcast:
                        await self.send_sink_update(
                            name=sink_name,
                            volume=sink.get("volume"),
                            muted=sink.get("muted"),
                            enabled=sink.get("enabled"),
                        )

        elif packet.type == "kdeconnect.systemvolume":
            self._systemvolume_announcements.append(packet)

        # ------------------------------------------------------------------
        # 12. Telephony Action Handling
        # ------------------------------------------------------------------
        elif packet.type in (
            "kdeconnect.telephony.request_mute",
            "kdeconnect.telephony.request",
        ) or (packet.type == "kdeconnect.telephony" and packet.body and "action" in packet.body):
            self._telephony_actions.append(packet)


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
    # Find My Phone & Find This Device Simulators
    # ------------------------------------------------------------------
    async def send_find_this_device_request(self) -> Packet:
        """Simulate phone requesting Linux desktop to ring (Find This Device)."""
        pkt = Packet(
            type="kdeconnect.findmyphone.request",
            body={},
        )
        await self.send_packet(pkt)
        return pkt

    def get_find_phone_requests(self) -> list[Packet]:
        """Return history of all received findmyphone packets from daemon."""
        return [
            p for p in self._received_packets
            if p.type in ("kdeconnect.findmyphone.request", "kdeconnect.findmyphone")
        ]

    async def wait_for_find_phone_request(self, timeout: float = 5.0) -> Packet:
        """Wait for daemon to send a kdeconnect.findmyphone.request packet."""
        existing = self.get_find_phone_requests()
        if existing:
            return existing[-1]
        return await self.wait_for_packet("kdeconnect.findmyphone.request", timeout=timeout)

    def has_received_find_phone_request(self) -> bool:
        """Return True if mock phone received at least one ring request."""
        return len(self.get_find_phone_requests()) > 0

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

    async def send_share_text(self, text: str) -> Packet:
        """Transmit text share request to daemon."""
        pkt = Packet(
            type="kdeconnect.share.request",
            body={"text": text},
        )
        await self.send_packet(pkt)
        return pkt

    async def send_share_url(self, url: str) -> Packet:
        """Transmit URL share request to daemon."""
        pkt = Packet(
            type="kdeconnect.share.request",
            body={"url": url},
        )
        await self.send_packet(pkt)
        return pkt

    async def send_share_file(
        self,
        filename: str,
        payload_bytes: bytes,
        target_daemon_cert_pem: Optional[str] = None,
        last_modified: Optional[int] = None,
        creation_time: Optional[int] = None,
        open_flag: bool = False,
        number_of_files: int = 1,
        total_payload_size: Optional[int] = None,
    ) -> Tuple[Packet, int, asyncio.Server, asyncio.Event]:
        """Start ephemeral TLS server and transmit file share control packet."""
        completed_event = asyncio.Event()

        async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                writer.write(payload_bytes)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                completed_event.set()

        daemon_cert = target_daemon_cert_pem
        if not daemon_cert and self.paired_peers:
            daemon_cert = list(self.paired_peers.values())[0]

        ssl_ctx = create_paired_ssl_context(
            local_cert_pem=self.cert_pem,
            local_key_pem=self.key_pem,
            peer_cert_pem=daemon_cert if isinstance(daemon_cert, bytes) else (daemon_cert or "").encode("utf-8"),
            server_side=True,
        )
        server = await asyncio.start_server(_handle_client, host="127.0.0.1", port=0, ssl=ssl_ctx)
        port = server.sockets[0].getsockname()[1]

        now_ms = int(time.time() * 1000)
        body: Dict[str, Any] = {
            "filename": filename,
            "lastModified": last_modified if last_modified is not None else now_ms,
            "creationTime": creation_time if creation_time is not None else now_ms,
            "open": bool(open_flag),
            "numberOfFiles": number_of_files,
            "totalPayloadSize": total_payload_size if total_payload_size is not None else len(payload_bytes),
        }

        pkt = Packet(
            type="kdeconnect.share.request",
            body=body,
            payloadSize=len(payload_bytes),
            payloadTransferInfo={"port": port},
        )
        await self.send_packet(pkt)
        return pkt, port, server, completed_event

    async def receive_share_file(
        self,
        target_port: int,
        expected_size: int,
        target_daemon_cert_pem: Optional[str] = None,
        timeout: float = 5.0,
    ) -> bytes:
        """Connect outbound to daemon ephemeral payload server and stream file bytes."""
        daemon_cert = target_daemon_cert_pem
        if not daemon_cert and self.paired_peers:
            daemon_cert = list(self.paired_peers.values())[0]

        ssl_ctx = create_paired_ssl_context(
            local_cert_pem=self.cert_pem,
            local_key_pem=self.key_pem,
            peer_cert_pem=daemon_cert if isinstance(daemon_cert, bytes) else (daemon_cert or "").encode("utf-8"),
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

    def get_share_packets(self) -> List[Packet]:
        """Retrieve all received share packets."""
        return [p for p in self.received_packets if p.type in ("kdeconnect.share.request", "kdeconnect.share.request.update")]

    async def wait_for_share_packet(self, timeout: float = 5.0) -> Packet:
        """Await reception of share packet."""
        return await self.wait_for_packet("kdeconnect.share.request", timeout=timeout)

    # ------------------------------------------------------------------
    # Contacts Feature Simulators & Helpers
    # ------------------------------------------------------------------
    def set_mock_contacts(self, contacts: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
        """Populate or replace mock device address book.
        
        Accepts:
          - List of dicts: [{"uid": "1", "timestamp": 1712345678000, "vcard": "BEGIN:VCARD..."}]
          - Dict mapping: {uid: {"timestamp": ts, "vcard": vcard_str}} or {uid: vcard_str}
        """
        self._mock_contacts.clear()
        now_ms = int(time.time() * 1000)
        if isinstance(contacts, list):
            for entry in contacts:
                uid = str(entry.get("uid", uuid.uuid4().hex[:8]))
                ts = int(entry.get("timestamp", now_ms))
                vcard = str(entry.get("vcard", ""))
                self._mock_contacts[uid] = {"uid": uid, "timestamp": ts, "vcard": vcard}
        elif isinstance(contacts, dict):
            for uid, val in contacts.items():
                uid_str = str(uid)
                if isinstance(val, dict):
                    ts = int(val.get("timestamp", now_ms))
                    vcard = str(val.get("vcard", ""))
                else:
                    ts = now_ms
                    vcard = str(val)
                self._mock_contacts[uid_str] = {"uid": uid_str, "timestamp": ts, "vcard": vcard}

    def add_mock_contact(self, uid: str, vcard: str, timestamp: Optional[int] = None) -> None:
        """Add or update a single mock contact."""
        ts = timestamp if timestamp is not None else int(time.time() * 1000)
        self._mock_contacts[str(uid)] = {"uid": str(uid), "timestamp": ts, "vcard": str(vcard)}

    def remove_mock_contact(self, uid: str) -> bool:
        """Remove a mock contact from the phonebook."""
        return self._mock_contacts.pop(str(uid), None) is not None

    def clear_mock_contacts(self) -> None:
        """Wipe all mock contacts."""
        self._mock_contacts.clear()

    def get_mock_contacts_count(self) -> int:
        """Return total contacts currently staged on the mock phone."""
        return len(self._mock_contacts)

    async def send_contacts_uids_timestamps(self, uids_timestamps: Dict[str, int]) -> Packet:
        """Manually transmit kdeconnect.contacts.response_uids_timestamps."""
        body: Dict[str, Any] = {"uids": list(uids_timestamps.keys())}
        body.update(uids_timestamps)
        pkt = Packet(type="kdeconnect.contacts.response_uids_timestamps", body=body)
        await self.send_packet(pkt)
        return pkt

    async def send_contacts_vcards(self, vcards: Dict[str, str]) -> Packet:
        """Manually transmit kdeconnect.contacts.response_vcards."""
        body: Dict[str, Any] = {"uids": list(vcards.keys())}
        body.update(vcards)
        pkt = Packet(type="kdeconnect.contacts.response_vcards", body=body)
        await self.send_packet(pkt)
        return pkt

    def get_contacts_packets(self) -> List[Packet]:
        """Return all received contacts packets."""
        return [
            p for p in self.received_packets
            if p.type.startswith("kdeconnect.contacts.")
        ]

    async def wait_for_contacts_request(
        self,
        packet_type: str = "kdeconnect.contacts.request_all_uids_timestamps",
        timeout: float = 5.0,
    ) -> Packet:
        """Wait for incoming contacts request packet."""
        return await self.wait_for_packet(packet_type, timeout=timeout)

    # ------------------------------------------------------------------
    # MPRIS Simulation Helpers
    # ------------------------------------------------------------------
    def set_mock_players(self, players: Dict[str, Dict[str, Any]]) -> None:
        """Stage mock media players on the device."""
        self._mock_players = {}
        for name, data in players.items():
            entry = dict(data)
            entry["player"] = name
            entry.setdefault("isPlaying", True)
            entry.setdefault("pos", 0)
            entry.setdefault("length", 180000)
            entry.setdefault("volume", 70)
            entry.setdefault("last_pos_timestamp", time.time())
            self._mock_players[name] = entry

    def add_mock_player(
        self,
        player: str,
        title: str = "Test Track",
        artist: str = "Test Artist",
        album: str = "Test Album",
        is_playing: bool = True,
        pos: int = 0,
        length: int = 240000,
        volume: int = 80,
        album_art_url: Optional[str] = None,
        can_play: bool = True,
        can_pause: bool = True,
        can_go_next: bool = True,
        can_go_previous: bool = True,
        can_seek: bool = True,
        loop_status: str = "None",
        shuffle: bool = False,
    ) -> None:
        """Stage or update a single mock player with sensible defaults."""
        self._mock_players[player] = {
            "player": player,
            "title": title,
            "artist": artist,
            "album": album,
            "nowPlaying": f"{artist} - {title}",
            "albumArtUrl": album_art_url or "",
            "isPlaying": is_playing,
            "pos": pos,
            "length": length,
            "volume": volume,
            "canPlay": can_play,
            "canPause": can_pause,
            "canGoNext": can_go_next,
            "canGoPrevious": can_go_previous,
            "canSeek": can_seek,
            "loopStatus": loop_status,
            "shuffle": shuffle,
            "last_pos_timestamp": time.time(),
        }

    def remove_mock_player(self, player: str) -> bool:
        """Remove a mock player from the phone."""
        return self._mock_players.pop(player, None) is not None

    def clear_mock_players(self) -> None:
        """Clear all staged mock players."""
        self._mock_players.clear()

    def get_mock_players(self) -> Dict[str, Dict[str, Any]]:
        """Return dictionary copy of currently staged mock players."""
        return {k: dict(v) for k, v in self._mock_players.items()}

    async def send_mpris_player_list(self, players: Optional[List[str]] = None) -> Packet:
        """Transmit kdeconnect.mpris packet advertising available player list."""
        player_names = players if players is not None else list(self._mock_players.keys())
        pkt = Packet(
            type="kdeconnect.mpris",
            body={"playerList": player_names, "supportAlbumArtPayload": True},
        )
        await self.send_packet(pkt)
        return pkt

    async def send_mpris_update(
        self,
        player: str,
        title: Optional[str] = None,
        artist: Optional[str] = None,
        album: Optional[str] = None,
        is_playing: Optional[bool] = None,
        pos: Optional[int] = None,
        length: Optional[int] = None,
        volume: Optional[int] = None,
        can_play: Optional[bool] = None,
        can_pause: Optional[bool] = None,
        can_go_next: Optional[bool] = None,
        can_go_previous: Optional[bool] = None,
        can_seek: Optional[bool] = None,
        loop_status: Optional[str] = None,
        shuffle: Optional[bool] = None,
        **extra: Any,
    ) -> Packet:
        """Transmit incremental or full kdeconnect.mpris player update packet."""
        body: Dict[str, Any] = {"player": player}
        if title is not None:
            body["title"] = title
        if artist is not None:
            body["artist"] = artist
        if album is not None:
            body["album"] = album
        if is_playing is not None:
            body["isPlaying"] = is_playing
        if pos is not None:
            body["pos"] = pos
        if length is not None:
            body["length"] = length
        if volume is not None:
            body["volume"] = volume
        if can_play is not None:
            body["canPlay"] = can_play
        if can_pause is not None:
            body["canPause"] = can_pause
        if can_go_next is not None:
            body["canGoNext"] = can_go_next
        if can_go_previous is not None:
            body["canGoPrevious"] = can_go_previous
        if can_seek is not None:
            body["canSeek"] = can_seek
        if loop_status is not None:
            body["loopStatus"] = loop_status
        if shuffle is not None:
            body["shuffle"] = shuffle
        body.update(extra)

        pkt = Packet(type="kdeconnect.mpris", body=body)
        await self.send_packet(pkt)
        return pkt

    def get_mpris_requests(self, action: Optional[str] = None) -> List[Packet]:
        """Return history of received kdeconnect.mpris.request packets."""
        if action is None:
            return list(self._mpris_requests)
        return [p for p in self._mpris_requests if p.body.get("action") == action]

    async def wait_for_mpris_request(
        self,
        action: Optional[str] = None,
        timeout: float = 5.0,
    ) -> Packet:
        """Wait for an incoming kdeconnect.mpris.request packet, optionally filtering by action."""
        loop = asyncio.get_running_loop()
        start_time = loop.time()

        while loop.time() - start_time < timeout:
            for p in self._mpris_requests:
                if action is None or p.body.get("action") == action:
                    return p
            try:
                pkt = await self.wait_for_packet("kdeconnect.mpris.request", timeout=min(0.2, timeout))
                if action is None or pkt.body.get("action") == action:
                    return pkt
            except ConnectionTimeoutError:
                pass

        raise ConnectionTimeoutError(
            f"Timed out waiting for kdeconnect.mpris.request (action={action})",
            timeout_seconds=timeout,
        )

    # ------------------------------------------------------------------
    # RemoteCommands Simulation Helpers (Milestone 15)
    # ------------------------------------------------------------------
    def set_mock_commands(
        self,
        commands: Union[Dict[str, Dict[str, str]], List[Dict[str, str]]],
    ) -> None:
        """Stage mock commands on the device."""
        self._mock_commands.clear()
        if isinstance(commands, list):
            for cmd in commands:
                k = cmd.get("key")
                if k:
                    self._mock_commands[str(k)] = {
                        "key": str(k),
                        "name": str(cmd.get("name", k)),
                        "command": str(cmd.get("command", "")),
                    }
        elif isinstance(commands, dict):
            for k, val in commands.items():
                if isinstance(val, dict):
                    self._mock_commands[str(k)] = {
                        "key": str(val.get("key", k)),
                        "name": str(val.get("name", k)),
                        "command": str(val.get("command", "")),
                    }
                else:
                    self._mock_commands[str(k)] = {
                        "key": str(k),
                        "name": str(val),
                        "command": "",
                    }

    def add_mock_command(self, key: str, name: str, command: str = "") -> None:
        """Add or update a single staged mock command."""
        self._mock_commands[key] = {
            "key": key,
            "name": name,
            "command": command,
        }

    def remove_mock_command(self, key: str) -> bool:
        """Remove a staged mock command by key."""
        return self._mock_commands.pop(key, None) is not None

    def clear_mock_commands(self) -> None:
        """Clear all staged mock commands."""
        self._mock_commands.clear()

    def get_mock_commands(self) -> Dict[str, Dict[str, str]]:
        """Return a copy of currently staged mock commands."""
        return {k: dict(v) for k, v in self._mock_commands.items()}

    async def send_command_list(
        self,
        commands: Optional[Union[Dict[str, Dict[str, str]], List[Dict[str, str]]]] = None,
        format: str = "dict",
        packet_type: str = "kdeconnect.runcommand",
    ) -> Packet:
        """Transmit command list packet advertising available mock commands.

        Args:
            commands: Optional override commands. Defaults to self._mock_commands.
            format: "dict" for native JSON dictionary or "string" for JSON-escaped string.
            packet_type: "kdeconnect.runcommand" or "kdeconnect.remotecommands".
        """
        if commands is None:
            cmd_dict = {k: dict(v) for k, v in self._mock_commands.items()}
        elif isinstance(commands, list):
            cmd_dict = {c["key"]: dict(c) for c in commands if "key" in c}
        else:
            cmd_dict = {k: dict(v) for k, v in commands.items()}

        if self.chaos_config.commands_corrupt_command_list:
            body: Dict[str, Any] = {"commandList": 123456}
        elif self.chaos_config.commands_malformed_json_string:
            body = {"commandList": '{"unclosed_json: "broken'}
        elif format == "string":
            body = {"commandList": json.dumps(cmd_dict)}
        else:
            body = {"commandList": cmd_dict}

        pkt = Packet(type=packet_type, body=body)
        await self.send_packet(pkt)
        return pkt

    async def send_command_output(
        self,
        key: str,
        output: str = "Command executed successfully",
        exit_code: int = 0,
    ) -> Packet:
        """Transmit kdeconnect.runcommand.output packet."""
        pkt = Packet(
            type="kdeconnect.runcommand.output",
            body={
                "key": key,
                "output": output,
                "exitCode": exit_code,
            },
        )
        await self.send_packet(pkt)
        return pkt

    @property
    def executed_commands(self) -> List[str]:
        """Return chronological history of executed command keys."""
        return list(self._executed_commands)

    def get_command_exec_requests(self, key: Optional[str] = None) -> List[Packet]:
        """Return history of received execution request packets."""
        if key is None:
            return list(self._command_exec_requests)
        return [p for p in self._command_exec_requests if p.body.get("key") == key]

    async def wait_for_command_exec(
        self,
        key: Optional[str] = None,
        timeout: float = 5.0,
    ) -> Packet:
        """Wait for an incoming command execution packet, optionally filtering by key."""
        loop = asyncio.get_running_loop()
        start_time = loop.time()

        while loop.time() - start_time < timeout:
            for p in self._command_exec_requests:
                if key is None or p.body.get("key") == key:
                    return p
            try:
                pkt = await self.wait_for_packet("*", timeout=min(0.2, timeout))
                if (
                    pkt.type in (
                        "kdeconnect.remoteforcecommand",
                        "kdeconnect.runcommand.request",
                        "kdeconnect.remotecommands.request",
                    )
                    and "key" in (pkt.body or {})
                ):
                    if key is None or pkt.body.get("key") == key:
                        return pkt
            except ConnectionTimeoutError:
                pass

        raise ConnectionTimeoutError(
            f"Timed out waiting for remote command execution (key={key})",
            timeout_seconds=timeout,
        )

    async def send_runcommand_request(
        self,
        key: Optional[str] = None,
        request_command_list: bool = False,
        stop: bool = False,
    ) -> Packet:
        """Transmit kdeconnect.runcommand.request packet to trigger execution, query commands, or stop."""
        body: Dict[str, Any] = {}
        if key is not None:
            body["key"] = key
        elif request_command_list:
            body["requestCommandList"] = True
        elif stop:
            body["stop"] = True

        pkt = Packet(type="kdeconnect.runcommand.request", body=body)
        await self.send_packet(pkt)
        return pkt

    def get_command_stop_requests(self) -> List[Packet]:
        """Return history of received process stop request packets."""
        return list(self._command_stop_requests)

    def get_received_command_outputs(self, key: Optional[str] = None) -> List[Packet]:
        """Return history of received kdeconnect.runcommand.output packets."""
        if key is None:
            return list(self._received_command_outputs)
        return [p for p in self._received_command_outputs if p.body.get("key") == key]

    async def wait_for_runcommand_output(
        self,
        key: Optional[str] = None,
        phase: Optional[str] = None,
        timeout: float = 5.0,
    ) -> Packet:
        """Wait for an incoming kdeconnect.runcommand.output packet, optionally filtering by key or phase.

        Args:
            key: Optional command key filter.
            phase: Optional phase filter: 'commandStarted', 'commandOutput', or 'commandFinished'.
            timeout: Timeout in seconds.
        """
        loop = asyncio.get_running_loop()
        start_time = loop.time()

        def _matches(p: Packet) -> bool:
            if p.type != "kdeconnect.runcommand.output":
                return False
            body = p.body or {}
            if key is not None and body.get("key") != key:
                return False
            if phase is not None and not body.get(phase):
                return False
            return True

        while loop.time() - start_time < timeout:
            for p in self._received_command_outputs:
                if _matches(p):
                    return p
            try:
                pkt = await self.wait_for_packet("kdeconnect.runcommand.output", timeout=min(0.2, timeout))
                if _matches(pkt):
                    return pkt
            except ConnectionTimeoutError:
                pass

        raise ConnectionTimeoutError(
            f"Timed out waiting for kdeconnect.runcommand.output (key={key}, phase={phase})",
            timeout_seconds=timeout,
        )

    async def wait_for_runcommand_list(self, timeout: float = 5.0) -> Packet:
        """Wait for an incoming kdeconnect.runcommand packet containing commandList."""
        loop = asyncio.get_running_loop()
        start_time = loop.time()

        while loop.time() - start_time < timeout:
            for p in self._received_packets:
                if p.type == "kdeconnect.runcommand" and "commandList" in (p.body or {}):
                    return p
            try:
                pkt = await self.wait_for_packet("kdeconnect.runcommand", timeout=min(0.2, timeout))
                if "commandList" in (pkt.body or {}):
                    return pkt
            except ConnectionTimeoutError:
                pass

        raise ConnectionTimeoutError(
            "Timed out waiting for kdeconnect.runcommand with commandList",
            timeout_seconds=timeout,
        )

    # ------------------------------------------------------------------
    # SystemVolume Simulation Helpers (Audio Sinks & Mixer Controls)
    # ------------------------------------------------------------------
    def set_mock_sinks(
        self,
        sinks: Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]]],
    ) -> None:
        """Stage mock audio output sinks on the virtual device."""
        self._mock_sinks.clear()
        if isinstance(sinks, list):
            for s in sinks:
                name = s.get("name")
                if name:
                    self._mock_sinks[str(name)] = {
                        "name": str(name),
                        "description": str(s.get("description", name)),
                        "muted": bool(s.get("muted", False)),
                        "volume": int(s.get("volume", 32768)),
                        "maxVolume": int(s.get("maxVolume", 65536)),
                        "enabled": bool(s.get("enabled", False)),
                    }
        elif isinstance(sinks, dict):
            for name, val in sinks.items():
                if isinstance(val, dict):
                    self._mock_sinks[str(name)] = {
                        "name": str(val.get("name", name)),
                        "description": str(val.get("description", name)),
                        "muted": bool(val.get("muted", False)),
                        "volume": int(val.get("volume", 32768)),
                        "maxVolume": int(val.get("maxVolume", 65536)),
                        "enabled": bool(val.get("enabled", False)),
                    }

    def add_mock_sink(
        self,
        name: str,
        description: str = "",
        volume: int = 32768,
        max_volume: int = 65536,
        muted: bool = False,
        enabled: bool = False,
    ) -> None:
        """Add or update a single staged mock audio sink."""
        self._mock_sinks[name] = {
            "name": name,
            "description": description or name,
            "muted": muted,
            "volume": volume,
            "maxVolume": max_volume,
            "enabled": enabled,
        }

    def remove_mock_sink(self, name: str) -> bool:
        """Remove a staged mock sink by name."""
        return self._mock_sinks.pop(name, None) is not None

    def clear_mock_sinks(self) -> None:
        """Clear all staged mock sinks."""
        self._mock_sinks.clear()

    def get_mock_sinks(self) -> Dict[str, Dict[str, Any]]:
        """Return a copy of currently staged mock sinks."""
        return {k: dict(v) for k, v in self._mock_sinks.items()}

    async def send_sink_list(
        self,
        sinks: Optional[Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]] = None,
    ) -> Packet:
        """Transmit kdeconnect.systemvolume packet advertising available audio sinks."""
        if sinks is None:
            sink_list = [dict(v) for v in self._mock_sinks.values()]
        elif isinstance(sinks, list):
            sink_list = [dict(s) for s in sinks]
        else:
            sink_list = [dict(v) for v in sinks.values()]

        if self.chaos_config.systemvolume_corrupt_sink_list:
            body: Dict[str, Any] = {"sinkList": "corrupted_non_list_value"}
        else:
            body = {"sinkList": sink_list}

        pkt = Packet(type="kdeconnect.systemvolume", body=body)
        await self.send_packet(pkt)
        return pkt

    async def send_sink_update(
        self,
        name: str,
        volume: Optional[int] = None,
        muted: Optional[bool] = None,
        enabled: Optional[bool] = None,
    ) -> Packet:
        """Transmit incremental single sink mutation update (kdeconnect.systemvolume)."""
        body: Dict[str, Any] = {"name": name}
        if volume is not None:
            body["volume"] = volume
        if muted is not None:
            body["muted"] = muted
        if enabled is not None:
            body["enabled"] = enabled

        pkt = Packet(type="kdeconnect.systemvolume", body=body)
        await self.send_packet(pkt)
        return pkt

    async def send_volume_request(self, name: str, volume: int) -> Packet:
        """Transmit kdeconnect.systemvolume.request packet to set volume on remote sink."""
        pkt = Packet(
            type="kdeconnect.systemvolume.request",
            body={"name": name, "volume": volume},
        )
        await self.send_packet(pkt)
        return pkt

    async def send_mute_request(self, name: str, muted: bool) -> Packet:
        """Transmit kdeconnect.systemvolume.request packet to toggle mute on remote sink."""
        pkt = Packet(
            type="kdeconnect.systemvolume.request",
            body={"name": name, "muted": muted},
        )
        await self.send_packet(pkt)
        return pkt

    async def send_default_sink_request(self, name: str, enabled: bool = True) -> Packet:
        """Transmit kdeconnect.systemvolume.request packet to set active default sink."""
        pkt = Packet(
            type="kdeconnect.systemvolume.request",
            body={"name": name, "enabled": enabled},
        )
        await self.send_packet(pkt)
        return pkt

    async def send_request_sinks(self) -> Packet:
        """Transmit kdeconnect.systemvolume.request packet requesting complete sink list."""
        pkt = Packet(
            type="kdeconnect.systemvolume.request",
            body={"requestSinks": True},
        )
        await self.send_packet(pkt)
        return pkt

    def get_systemvolume_requests(self, name: Optional[str] = None) -> List[Packet]:
        """Return history of received kdeconnect.systemvolume.request packets."""
        if name is None:
            return list(self._systemvolume_requests)
        return [p for p in self._systemvolume_requests if p.body.get("name") == name]

    def get_systemvolume_announcements(self) -> List[Packet]:
        """Return history of received kdeconnect.systemvolume announcement packets."""
        return list(self._systemvolume_announcements)

    async def wait_for_systemvolume_request(
        self,
        name: Optional[str] = None,
        timeout: float = 5.0,
    ) -> Packet:
        """Wait for an incoming kdeconnect.systemvolume.request packet, optionally filtering by sink name."""
        loop = asyncio.get_running_loop()
        start_time = loop.time()

        while loop.time() - start_time < timeout:
            for p in self._systemvolume_requests:
                if name is None or p.body.get("name") == name:
                    return p
            try:
                pkt = await self.wait_for_packet("kdeconnect.systemvolume.request", timeout=min(0.2, timeout))
                if name is None or pkt.body.get("name") == name:
                    return pkt
            except ConnectionTimeoutError:
                pass

        raise ConnectionTimeoutError(
            f"Timed out waiting for kdeconnect.systemvolume.request (name={name})",
            timeout_seconds=timeout,
        )

    async def wait_for_systemvolume_announcement(self, timeout: float = 5.0) -> Packet:
        """Wait for an incoming kdeconnect.systemvolume packet from peer."""
        loop = asyncio.get_running_loop()
        start_time = loop.time()

        while loop.time() - start_time < timeout:
            for p in self._systemvolume_announcements:
                return p
            try:
                pkt = await self.wait_for_packet("kdeconnect.systemvolume", timeout=min(0.2, timeout))
                return pkt
            except ConnectionTimeoutError:
                pass

        raise ConnectionTimeoutError(
            "Timed out waiting for kdeconnect.systemvolume announcement",
            timeout_seconds=timeout,
        )

    # ------------------------------------------------------------------
    # Telephony Simulation Helpers
    # ------------------------------------------------------------------
    async def send_telephony_call(
        self,
        event: str,
        phone_number: str | None = None,
        contact_name: str | None = None,
        phone_thumbnail: str | None = None,
        is_cancel: bool | str | None = None,
    ) -> Packet:
        """Transmit kdeconnect.telephony packet to simulate call event."""
        body: dict[str, Any] = {"event": event}
        if phone_number is not None:
            body["phoneNumber"] = phone_number
        if contact_name is not None:
            body["contactName"] = contact_name
        if phone_thumbnail is not None:
            body["phoneThumbnail"] = phone_thumbnail
        if is_cancel is not None:
            body["isCancel"] = is_cancel

        self._last_telephony_call = dict(body)
        pkt = Packet(type="kdeconnect.telephony", body=body)
        await self.send_packet(pkt)
        return pkt

    async def simulate_incoming_call(
        self,
        phone_number: str = "+15551234567",
        contact_name: str = "Alice",
        phone_thumbnail: str | None = None,
    ) -> Packet:
        """Simulate an incoming ringing phone call."""
        return await self.send_telephony_call(
            event="ringing",
            phone_number=phone_number,
            contact_name=contact_name,
            phone_thumbnail=phone_thumbnail,
            is_cancel=False,
        )

    async def simulate_answer_call(
        self,
        phone_number: str = "+15551234567",
        contact_name: str = "Alice",
        phone_thumbnail: str | None = None,
    ) -> Packet:
        """Simulate user answering call (call active/talking)."""
        return await self.send_telephony_call(
            event="talking",
            phone_number=phone_number,
            contact_name=contact_name,
            phone_thumbnail=phone_thumbnail,
            is_cancel=False,
        )

    async def simulate_end_call(
        self,
        phone_number: str = "+15551234567",
        contact_name: str = "Alice",
    ) -> Packet:
        """Simulate call ending (cancelling/idle)."""
        return await self.send_telephony_call(
            event="idle",
            phone_number=phone_number,
            contact_name=contact_name,
            is_cancel=True,
        )

    async def simulate_missed_call(
        self,
        phone_number: str = "+15551234567",
        contact_name: str = "Alice",
    ) -> Packet:
        """Simulate a missed incoming call."""
        return await self.send_telephony_call(
            event="missedCall",
            phone_number=phone_number,
            contact_name=contact_name,
            is_cancel=False,
        )

    def get_telephony_actions(self, action: str | None = None) -> list[Packet]:
        """Return history of received telephony action packets."""
        if action is None:
            return list(self._telephony_actions)
        return [
            p
            for p in self._telephony_actions
            if (p.type == "kdeconnect.telephony.request_mute" and action in ("mute", "muteRinger"))
            or (p.type == "kdeconnect.telephony.request" and p.body.get("action") == action)
        ]

    async def wait_for_telephony_action(
        self,
        action: str | None = None,
        timeout: float = 5.0,
    ) -> Packet:
        """Wait for an outbound action packet from host daemon.

        Args:
            action: Optional action name filter (e.g. "mute", "muteRinger", "rejectCall").
            timeout: Maximum seconds to wait.
        """
        loop = asyncio.get_running_loop()
        start_time = loop.time()

        def _matches(p: Packet) -> bool:
            if action is None:
                return p.type in ("kdeconnect.telephony.request_mute", "kdeconnect.telephony.request")
            if action in ("mute", "muteRinger"):
                if p.type == "kdeconnect.telephony.request_mute":
                    return True
                return bool(p.type == "kdeconnect.telephony.request" and p.body.get("action") in ("mute", "muteRinger"))
            return bool(p.type == "kdeconnect.telephony.request" and p.body.get("action") == action)

        while loop.time() - start_time < timeout:
            for p in self._telephony_actions:
                if _matches(p):
                    return p
            await asyncio.sleep(0.05)

        raise ConnectionTimeoutError(
            f"Timed out waiting for telephony action (action={action})",
            timeout_seconds=timeout,
        )



# Convenience Alias
MockDevice = MockKdeConnectDevice



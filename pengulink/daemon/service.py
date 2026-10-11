"""pengulink.daemon.service
~~~~~~~~~~~~~~~~~~~~~~~~
Central background daemon service coordinating networking, cryptography,
storage, event dispatching, plugins, and Unix domain socket IPC.
"""

from __future__ import annotations

import asyncio
import collections
import datetime
import hashlib
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union

from cryptography import x509
from cryptography.hazmat.primitives import serialization

from pengulink.core.config import Config, PenguLinkConfig
from pengulink.core.events import (
    DeviceConnected,
    DeviceDiscovered,
    DeviceDisconnected,
    DevicePaired,
    DeviceUnpaired,
    EventBus,
    PairingRequested,
)
from pengulink.core.exceptions import (
    CertificateVerificationError,
    ConnectionClosedError,
    ConnectionTimeoutError,
    DeviceNotFoundError,
    InvalidStateTransitionError,
    PairingError,
    PairingRejectedError,
)
from pengulink.core.state_machine import DeviceState, DeviceStateMachine
from pengulink.crypto.certificate import (
    dump_certificate_pem,
    dump_private_key_pem,
    ensure_credentials,
    get_certificate_fingerprint,
    load_certificate_pem,
)
from pengulink.crypto.pairing import (
    PairingSession,
    compute_verification_code,
    create_pair_packet,
)
from pengulink.crypto.tls import (
    create_paired_ssl_context,
    create_tofu_ssl_context,
    verify_peer_certificate,
)
from pengulink.daemon.ipc import PenguLinkIPCServer, resolve_socket_path
from pengulink.network.packet import Packet, PacketEngine
from pengulink.network.tcp import TCPConnection, TCPStreamManager
from pengulink.network.udp import DiscoveredDevice, UDPDiscovery
from pengulink.plugins.base import PluginRegistry
from pengulink.plugins.battery import BatteryPlugin, DeviceBatteryState
from pengulink.plugins.clipboard import (
    MAX_CLIPBOARD_PACKET_BYTES,
    ClipboardBackend,
    ClipboardError,
    ClipboardPlugin,
    ClipboardUnavailableError,
    redact_clipboard,
)
from pengulink.plugins.connectivity_report import ConnectivityReportPlugin
from pengulink.plugins.findmyphone import FindMyPhonePlugin
from pengulink.plugins.notification import NotificationPlugin, StoredNotification
from pengulink.plugins.ping import PingPlugin
from pengulink.plugins.share import SharePlugin
from pengulink.services.alert import AlertService
from pengulink.storage.base import DeviceStore
from pengulink.storage.json_store import JsonStorageBackend


logger = logging.getLogger("pengulink.daemon.service")


def safe_redact_clipboard(content: Any) -> str:
    """Format clipboard content safely for logging without leaking sensitive data."""
    if content is None:
        return "[REDACTED, None]"
    if not isinstance(content, str):
        content = str(content)
    char_len = len(content)
    digest_prefix = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"[REDACTED, {char_len} chars, sha256:{digest_prefix}]"


class ClipboardResult(dict):
    """Result dictionary that also compares equal to string content for convenience."""

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.get("content", "") == other
        return super().__eq__(other)

    def __str__(self) -> str:
        return self.get("content", "")


def format_uptime(seconds: float) -> str:
    """Format elapsed seconds into a human-readable duration string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h {m}m {s}s"
    d, h = divmod(h, 24)
    return f"{d}d {h}h {m}m"


class LogRingBuffer(logging.Handler):
    """Thread-safe circular in-memory buffer capturing recent daemon log events."""

    def __init__(self, capacity: int = 1000) -> None:
        super().__init__()
        self.capacity = capacity
        self._records: collections.deque[Dict[str, Any]] = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = record.created
            time_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            entry = {
                "timestamp": ts,
                "time_str": time_str,
                "timestamp_str": time_str,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            with self._lock:
                self._records.append(entry)
        except Exception:
            self.handleError(record)

    def get_logs(self, limit: int = 100, min_level: Optional[str] = None) -> List[Dict[str, Any]]:
        limit = max(1, min(1000, limit))
        with self._lock:
            records = list(self._records)
        if min_level:
            min_num = getattr(logging, min_level.upper(), 0)
            records = [r for r in records if getattr(logging, r["level"], 0) >= min_num]
        return records[-limit:]


class PenguLinkDaemon:
    """Central background service coordinator for PenguLink KDE Connect core."""

    def __init__(
        self,
        config: Optional[Union[Config, PenguLinkConfig]] = None,
        socket_path: Optional[Union[Path, str]] = None,
        clipboard_backend: Optional[ClipboardBackend] = None,
    ) -> None:
        if config is None:
            self.config = Config.load()
        else:
            self.config = config

        self.socket_path = (
            Path(socket_path)
            if socket_path is not None
            else resolve_socket_path(self.config.storage.ipc_socket_path)
        )

        # 1. Event Bus
        self.event_bus = EventBus()

        # 2. Persistence Layer
        storage_dir = getattr(self.config.storage, "data_dir", None) or getattr(self.config.storage, "devices_store_path", Path(".")).parent
        self._storage_backend = JsonStorageBackend(storage_dir=storage_dir)
        self.device_store = DeviceStore(self._storage_backend)

        # 3. Core Plugins & Plugin Registry
        self.ping_plugin = PingPlugin(event_bus=self.event_bus)
        self.battery_plugin = BatteryPlugin(event_bus=self.event_bus)
        self.notification_plugin = NotificationPlugin(event_bus=self.event_bus)
        self.connectivity_plugin = ConnectivityReportPlugin(event_bus=self.event_bus)
        self.clipboard_plugin = ClipboardPlugin(
            event_bus=self.event_bus,
            backend=clipboard_backend,
            packet_sender=self._send_clipboard_packet,
            capability_checker=self._check_device_clipboard_capability,
        )
        self.alert_service = AlertService(
            event_bus=self.event_bus,
            runtime_dir=getattr(self.config.storage, "runtime_dir", None),
        )
        self.findmyphone_plugin = FindMyPhonePlugin(
            event_bus=self.event_bus,
            alert_service=self.alert_service,
            packet_sender=self._send_findmyphone_packet,
        )
        self.find_my_phone_plugin = self.findmyphone_plugin
        self.share_plugin = SharePlugin(
            event_bus=self.event_bus,
            packet_sender=self._send_share_packet,
            is_paired_checker=self.device_store.is_device_paired,
            peer_resolver=self._resolve_device_peer_info,
            config=self.config,
        )
        self.plugin_registry = PluginRegistry(event_bus=self.event_bus)
        self.plugin_registry.register(self.ping_plugin)
        self.plugin_registry.register(self.battery_plugin)
        self.plugin_registry.register(self.notification_plugin)
        self.plugin_registry.register(self.connectivity_plugin)
        self.plugin_registry.register(self.clipboard_plugin)
        self.plugin_registry.register(self.findmyphone_plugin)
        self.plugin_registry.register(self.share_plugin)
        if hasattr(self.config, "device"):
            self.config.device.incoming_capabilities = self.plugin_registry.get_all_incoming_capabilities()
            self.config.device.outgoing_capabilities = self.plugin_registry.get_all_outgoing_capabilities()


        # 4. Cryptographic Credentials (RSA-2048, self-signed X.509, 0600 storage)
        self.config.ensure_directories()
        credentials_dir = getattr(self.config.storage, "config_dir", None) or getattr(self.config.storage, "credentials_dir", Path("."))
        self.private_key, self.certificate = ensure_credentials(
            credentials_dir,
            self.config.device_id,
        )
        self.local_key_pem = dump_private_key_pem(self.private_key)
        self.local_cert_pem = dump_certificate_pem(self.certificate)
        self.local_pubkey_pem = (
            self.certificate.public_key()
            .public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            .decode("utf-8")
        )
        self.share_plugin.local_cert_pem = self.local_cert_pem
        self.share_plugin.local_key_pem = self.local_key_pem

        # 5. Networking Subsystems
        self.tcp_manager = TCPStreamManager(
            config=self.config,
            on_connection_established=self._on_tcp_connection_established,
        )
        self.udp_discovery = UDPDiscovery(
            config=self.config,
            bound_tcp_port_getter=lambda: self.tcp_manager.bound_port or 1716,
            on_device_discovered=self._on_udp_device_discovered,
        )

        # 6. IPC Server
        self.ipc_server = PenguLinkIPCServer(
            socket_path=self.socket_path,
            handler=self._handle_ipc_request,
        )

        # 7. In-Memory Log Ring Buffer
        self._log_buffer = LogRingBuffer(capacity=1000)

        # 8. State and Resource Tracking
        self._active_connections: Dict[str, TCPConnection] = {}
        self._state_machines: Dict[str, DeviceStateMachine] = {}
        self._discovered_devices: Dict[str, DiscoveredDevice] = {}
        self._unpaired_peer_certs: Dict[str, bytes] = {}
        self._pairing_sessions: Dict[str, PairingSession] = {}
        self._pairing_futures: Dict[str, asyncio.Future[bool]] = {}
        self._connecting_devices: Set[str] = set()
        self._background_tasks: Set[asyncio.Task[Any]] = set()

        self._running: bool = False
        self._stop_event = asyncio.Event()
        self._start_time: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    def _get_or_create_state_machine(self, device_id: str) -> DeviceStateMachine:
        if device_id not in self._state_machines:
            self._state_machines[device_id] = DeviceStateMachine(
                device_id=device_id,
                initial_state=DeviceState.DISCONNECTED,
            )
        return self._state_machines[device_id]

    async def _transition_device(
        self,
        device_id: str,
        target_state: DeviceState,
        reason: Optional[str] = None,
    ) -> None:
        """Safely advance device state machine along valid transition paths."""
        sm = self._get_or_create_state_machine(device_id)
        current = sm.current_state
        if current == target_state:
            return

        if sm.can_transition_to(target_state):
            await sm.transition_to(target_state, reason=reason)
            return

        # Multi-step legal path traversals
        if current == DeviceState.DISCONNECTED and target_state == DeviceState.CONNECTED:
            await sm.transition_to(DeviceState.CONNECTING, reason=reason)
            await sm.transition_to(DeviceState.CONNECTED, reason=reason)
            return

        if current == DeviceState.DISCOVERED and target_state == DeviceState.CONNECTED:
            await sm.transition_to(DeviceState.CONNECTING, reason=reason)
            await sm.transition_to(DeviceState.CONNECTED, reason=reason)
            return

        if current == DeviceState.DISCOVERED and target_state == DeviceState.PAIRING:
            await sm.transition_to(DeviceState.CONNECTING, reason=reason)
            await sm.transition_to(DeviceState.PAIRING, reason=reason)
            return

        if current == DeviceState.DISCONNECTED and target_state == DeviceState.PAIRING:
            await sm.transition_to(DeviceState.CONNECTING, reason=reason)
            await sm.transition_to(DeviceState.PAIRING, reason=reason)
            return

        if current == DeviceState.CONNECTING and target_state == DeviceState.PAIRED:
            await sm.transition_to(DeviceState.PAIRING, reason=reason)
            await sm.transition_to(DeviceState.PAIRED, reason=reason)
            return

        if current == DeviceState.PAIRING and target_state == DeviceState.CONNECTED:
            await sm.transition_to(DeviceState.PAIRED, reason=reason)
            await sm.transition_to(DeviceState.CONNECTED, reason=reason)
            return

        if current in (DeviceState.ERROR, DeviceState.DISCONNECTING):
            await sm.transition_to(DeviceState.DISCONNECTED, reason=reason)
            if sm.can_transition_to(target_state):
                await sm.transition_to(target_state, reason=reason)
                return

        await sm.transition_to(target_state, reason=reason)

    # =====================================================================
    # Lifecycle Management
    # =====================================================================

    async def start(self) -> None:
        """Start all subsystems and bind IPC and network listeners."""
        if self._running:
            return

        self.config.ensure_directories()

        # Attach log buffer to pengulink logger
        p_logger = logging.getLogger("pengulink")
        lvl = getattr(getattr(self.config, "logging", None), "level", "INFO")
        p_logger.setLevel(getattr(logging, str(lvl).upper(), logging.INFO))
        if self._log_buffer not in p_logger.handlers:
            p_logger.addHandler(self._log_buffer)

        # 1. Initialize DeviceStore
        await self.device_store.initialize()

        # 2. Start PluginRegistry
        await self.plugin_registry.start_all()

        # 3. Start TCP Stream Manager
        await self.tcp_manager.start()

        # 4. Start UDP Discovery
        await self.udp_discovery.start()

        # 5. Start IPC Server
        await self.ipc_server.start()

        self._running = True
        self._start_time = time.time()
        self._stop_event.clear()
        logger.info("PenguLink daemon started successfully.")

    async def stop(self) -> None:
        """Gracefully shut down all background tasks, listeners, and resources."""
        if not self._running:
            return
        self._running = False

        logger.info("Shutting down PenguLink daemon subsystems...")

        # 1. Stop IPC Server
        try:
            await self.ipc_server.stop()
        except Exception as exc:
            logger.debug("Error stopping IPC server: %s", exc)

        # 2. Stop UDP Discovery
        try:
            await self.udp_discovery.stop()
        except Exception as exc:
            logger.debug("Error stopping UDP discovery: %s", exc)

        # 3. Stop TCP Stream Manager
        try:
            await self.tcp_manager.stop()
        except Exception as exc:
            logger.debug("Error stopping TCP manager: %s", exc)

        # 4. Stop Plugin Registry
        try:
            await self.plugin_registry.stop_all()
        except Exception as exc:
            logger.debug("Error stopping plugins: %s", exc)

        # Stop AlertService
        try:
            await self.alert_service.stop()
        except Exception as exc:
            logger.debug("Error stopping alert service: %s", exc)

        # 5. Close all active client connections
        conns = list(self._active_connections.values())
        self._active_connections.clear()
        for conn in conns:
            try:
                await conn.close()
            except Exception:
                pass

        # 6. Close DeviceStore
        try:
            await self.device_store.close()
        except Exception as exc:
            logger.debug("Error closing device store: %s", exc)

        # 7. Cancel and drain background tasks
        tasks = list(self._background_tasks)
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self._background_tasks.clear()

        # 8. Cancel pending pairing futures
        for fut in list(self._pairing_futures.values()):
            if not fut.done():
                fut.cancel()
        self._pairing_futures.clear()

        # Remove log handler
        p_logger = logging.getLogger("pengulink")
        if self._log_buffer in p_logger.handlers:
            p_logger.removeHandler(self._log_buffer)

        logger.info("PenguLink daemon stopped cleanly with zero resource leaks.")
        self._stop_event.set()

    async def run_forever(self) -> None:
        """Start daemon, attach signal listeners for SIGINT/SIGTERM, and await termination."""
        loop = asyncio.get_running_loop()

        def _handle_signal() -> None:
            logger.info("Received termination signal; scheduling graceful daemon shutdown")
            asyncio.create_task(self.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _handle_signal)
            except (NotImplementedError, RuntimeError):
                pass

        await self.start()
        await self._stop_event.wait()

    async def wait_until_stopped(self) -> None:
        """Wait until daemon stop event is triggered."""
        await self._stop_event.wait()

    # =====================================================================
    # Inverted mTLS & Network Connection Handling
    # =====================================================================

    def is_device_connected(self, device_id: str) -> bool:
        """Check if target device has an active open connection."""
        conn = self._active_connections.get(device_id)
        return bool(conn and not conn.is_closed)

    def is_device_paired(self, device_id: str) -> bool:
        """Check synchronously if device is currently paired."""
        sm = self._state_machines.get(device_id)
        if sm and sm.is_paired:
            return True
        if hasattr(self._storage_backend, "_devices"):
            dev = self._storage_backend._devices.get(device_id)
            if dev:
                return bool(dev.get("is_paired", False))
        return False


    def _register_connection(self, device_id: str, conn: TCPConnection) -> None:
        self._active_connections[device_id] = conn

        def _on_closed(c: TCPConnection) -> None:
            if self._active_connections.get(device_id) is c:
                self._active_connections.pop(device_id, None)
                asyncio.create_task(self._transition_device(device_id, DeviceState.DISCONNECTED, reason="Connection closed"))
                asyncio.create_task(self.event_bus.publish(DeviceDisconnected(device_id=device_id)))

        conn.add_close_callback(_on_closed)

    async def _on_tcp_connection_established(self, conn: TCPConnection) -> None:
        """Callback from TCPStreamManager on inbound client connection.

        Inverted TLS Rule: Inbound listener acts as TLS Client (server_side=False).
        """
        remote_id = conn.remote_device_id
        if not remote_id:
            await conn.close()
            return

        try:
            is_paired = await self.device_store.is_device_paired(remote_id)
            pinned_pem: Optional[bytes] = None
            if is_paired:
                pinned_pem = await self.device_store.get_device_certificate_pem(remote_id)
                if not pinned_pem:
                    logger.warning("Device %s marked paired but pinned certificate missing; closing", remote_id)
                    await conn.close()
                    return
                ssl_ctx = create_paired_ssl_context(
                    local_cert_pem=self.local_cert_pem,
                    local_key_pem=self.local_key_pem,
                    peer_cert_pem=pinned_pem,
                    server_side=False,
                )
            else:
                ssl_ctx = create_tofu_ssl_context(
                    local_cert_pem=self.local_cert_pem,
                    local_key_pem=self.local_key_pem,
                    server_side=False,
                )

            # Inverted STARTTLS upgrade
            timeout = getattr(self.config.network, "connect_timeout", 10.0)
            await conn.upgrade_to_tls(ssl_ctx, server_side=False, timeout=timeout)

            # Extract peer DER certificate from SSL transport
            ssl_obj = conn.writer.get_extra_info("ssl_object")
            peer_der = ssl_obj.getpeercert(binary_form=True) if ssl_obj else None
            if not peer_der:
                raise CertificateVerificationError("Failed to extract peer certificate after TLS upgrade")

            # Validate Subject CN against expected device ID and check pinned cert
            verify_peer_certificate(peer_der, expected_device_id=remote_id, pinned_cert_pem=pinned_pem)

            if not is_paired:
                self._unpaired_peer_certs[remote_id] = peer_der
                await self._transition_device(remote_id, DeviceState.CONNECTING, reason="Inbound TOFU TLS established")
            else:
                await self._transition_device(remote_id, DeviceState.CONNECTED, reason="Inbound paired TLS established")
                await self.event_bus.publish(
                    DeviceConnected(device_id=remote_id, host=conn.peer_ip, port=conn.peer_port)
                )

            self._register_connection(remote_id, conn)
            self._spawn_connection_read_loop(conn)

        except Exception as exc:
            logger.warning("Error in inbound TLS negotiation with %s: %s", remote_id, exc)
            await conn.close()

    async def connect_device(
        self,
        device_id: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 10.0,
    ) -> TCPConnection:
        """Establish outbound TCP connection and execute inverted mTLS handshake.

        Inverted TLS Rule: Outbound connector acts as TLS Server (server_side=True).
        """
        if self.is_device_connected(device_id):
            return self._active_connections[device_id]

        if device_id in self._connecting_devices:
            for _ in range(int(timeout * 10)):
                await asyncio.sleep(0.1)
                if self.is_device_connected(device_id):
                    return self._active_connections[device_id]
            raise ConnectionTimeoutError(f"Timed out awaiting concurrent connect to {device_id}")

        self._connecting_devices.add(device_id)
        try:
            if not host or not port:
                discovered = self._discovered_devices.get(device_id)
                if not discovered:
                    dev_record = await self.device_store.get_device(device_id)
                    if dev_record and dev_record.get("host") and dev_record.get("port"):
                        host = host or dev_record["host"]
                        port = port or dev_record["port"]
                    else:
                        raise DeviceNotFoundError(
                            f"Device {device_id} not discovered and no host/port provided"
                        )
                else:
                    host = host or discovered.sender_ip
                    port = port or discovered.tcp_port

            await self._transition_device(device_id, DeviceState.CONNECTING, reason="Outbound connect initiated")

            # 1. Connect TCP and send cleartext identity preamble
            conn = await self.tcp_manager.connect_to_device(
                host=str(host),
                port=int(port),
                target_device_id=device_id,
                timeout=timeout,
            )

            # 2. STARTTLS Upgrade (Inverted: server_side=True)
            is_paired = await self.device_store.is_device_paired(device_id)
            pinned_pem: Optional[bytes] = None
            if is_paired:
                pinned_pem = await self.device_store.get_device_certificate_pem(device_id)
                ssl_ctx = create_paired_ssl_context(
                    local_cert_pem=self.local_cert_pem,
                    local_key_pem=self.local_key_pem,
                    peer_cert_pem=pinned_pem or b"",
                    server_side=True,
                )
            else:
                ssl_ctx = create_tofu_ssl_context(
                    local_cert_pem=self.local_cert_pem,
                    local_key_pem=self.local_key_pem,
                    server_side=True,
                )

            await conn.upgrade_to_tls(ssl_ctx, server_side=True, timeout=timeout)

            # 3. Extract and verify peer certificate
            ssl_obj = conn.writer.get_extra_info("ssl_object")
            peer_der = ssl_obj.getpeercert(binary_form=True) if ssl_obj else None

            if is_paired:
                if not peer_der:
                    raise CertificateVerificationError("Failed to extract peer certificate after TLS upgrade")
                verify_peer_certificate(peer_der, expected_device_id=device_id, pinned_cert_pem=pinned_pem)
                await self._transition_device(device_id, DeviceState.CONNECTED, reason="Outbound paired TLS established")
                await self.event_bus.publish(
                    DeviceConnected(device_id=device_id, host=str(host), port=int(port))
                )
            else:
                # For unpaired outbound connections, peer_der may not be sent over TLS (server_side=True)
                if peer_der:
                    verify_peer_certificate(peer_der, expected_device_id=device_id)
                    self._unpaired_peer_certs[device_id] = peer_der
                elif device_id in self._unpaired_peer_certs:
                    peer_der = self._unpaired_peer_certs[device_id]

            self._register_connection(device_id, conn)
            self._spawn_connection_read_loop(conn)
            return conn

        finally:
            self._connecting_devices.discard(device_id)

    async def disconnect_device(self, device_id: str) -> None:
        """Disconnect active connection to device."""
        conn = self._active_connections.pop(device_id, None)
        if conn and not conn.is_closed:
            await conn.close()
        await self._transition_device(device_id, DeviceState.DISCONNECTED, reason="Explicit disconnect requested")
        await self.event_bus.publish(DeviceDisconnected(device_id=device_id))

    def _spawn_connection_read_loop(self, conn: TCPConnection) -> None:
        task = asyncio.create_task(self._connection_read_loop(conn))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _connection_read_loop(self, conn: TCPConnection) -> None:
        """Stream reader loop routing packets to pairing engine or PluginRegistry."""
        remote_id = conn.remote_device_id or "unknown"
        try:
            while self._running and not conn.is_closed:
                line = await conn.read_line(timeout=None)
                if not line:
                    break

                try:
                    packet = PacketEngine.parse(line)
                except Exception as exc:
                    logger.warning("[%s] Discarding malformed packet: %s", remote_id[:8], exc)
                    continue

                if packet.type == "kdeconnect.pair":
                    await self._handle_pairing_packet(conn, packet)
                    continue

                send_cb = lambda pkt: self.send_packet(remote_id, pkt)
                handled = await self.plugin_registry.dispatch_packet(packet, remote_id, send_cb)
                if not handled:
                    logger.debug("[%s] Unhandled packet type: '%s'", remote_id[:8], packet.type)

        except (asyncio.CancelledError, ConnectionClosedError):
            pass
        except Exception as exc:
            logger.debug("[%s] Connection read loop error: %s", remote_id[:8], exc)
        finally:
            await conn.close()

    # =====================================================================
    # UDP Discovery & Auto-Connect
    # =====================================================================

    async def _on_udp_device_discovered(self, discovered: DiscoveredDevice) -> None:
        """Callback invoked when UDP discovery receives an identity beacon."""
        self._discovered_devices[discovered.device_id] = discovered

        await self.event_bus.publish(
            DeviceDiscovered(
                device_id=discovered.device_id,
                device_name=discovered.device_name,
                device_type=discovered.device_type,
                host=discovered.sender_ip,
                port=discovered.tcp_port,
                protocol_version=discovered.protocol_version,
            )
        )

        sm = self._get_or_create_state_machine(discovered.device_id)
        if sm.current_state == DeviceState.DISCONNECTED:
            await self._transition_device(
                discovered.device_id,
                DeviceState.DISCOVERED,
                reason="Discovered via UDP identity broadcast",
            )

        # Auto-connect to paired peers upon UDP discovery
        is_paired = await self.device_store.is_device_paired(discovered.device_id)
        if is_paired and not self.is_device_connected(discovered.device_id):
            if discovered.device_id not in self._connecting_devices:
                task = asyncio.create_task(
                    self.connect_device(
                        discovered.device_id,
                        host=discovered.sender_ip,
                        port=discovered.tcp_port,
                    ),
                    name=f"auto_connect_{discovered.device_id}",
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

    # =====================================================================
    # Pairing Protocol Handling
    # =====================================================================

    async def _handle_pairing_packet(self, conn: TCPConnection, packet: Packet) -> None:
        """Process incoming kdeconnect.pair packet."""
        remote_id = conn.remote_device_id or ""
        is_pair = packet.body.get("pair") is True

        if not is_pair:
            # Pairing rejection or revocation
            logger.info("Pairing revoked or rejected by remote peer %s", remote_id)
            fut = self._pairing_futures.get(remote_id)
            if fut and not fut.done():
                fut.set_exception(PairingRejectedError(f"Pairing rejected by remote peer '{remote_id}'"))

            if await self.device_store.is_device_paired(remote_id):
                await self.device_store.delete_device(remote_id)
                await self.event_bus.publish(DeviceUnpaired(device_id=remote_id))

            self.connectivity_plugin.clear_device(remote_id)
            if hasattr(self, "clipboard_plugin"):
                self.clipboard_plugin.clear_device(remote_id)
            self._pairing_sessions.pop(remote_id, None)
            self._unpaired_peer_certs.pop(remote_id, None)
            await self._transition_device(remote_id, DeviceState.DISCONNECTED, reason="Peer revoked or rejected pairing")
            await conn.close()
            return

        # Peer requested or confirmed pairing
        ts = packet.body.get("timestamp", int(time.time()))

        peer_der = self._unpaired_peer_certs.get(remote_id)
        if not peer_der:
            ssl_obj = conn.writer.get_extra_info("ssl_object")
            if ssl_obj:
                peer_der = ssl_obj.getpeercert(binary_form=True)
                if peer_der:
                    try:
                        verify_peer_certificate(peer_der, expected_device_id=remote_id)
                        self._unpaired_peer_certs[remote_id] = peer_der
                    except Exception as exc:
                        logger.warning("Peer certificate validation failed for %s: %s", remote_id, exc)
                        peer_der = None

        if not peer_der:
            logger.warning("No peer certificate available to process pairing packet from %s", remote_id)
            return

        peer_cert = x509.load_der_x509_certificate(peer_der)
        peer_pubkey_pem = (
            peer_cert.public_key()
            .public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            .decode("utf-8")
        )

        code = compute_verification_code(self.local_pubkey_pem, peer_pubkey_pem, int(ts))

        # Check if we have an in-flight outbound pairing request waiting for confirmation
        fut = self._pairing_futures.get(remote_id)
        if fut and not fut.done():
            # Confirm pairing persistence
            peer_cert_pem = dump_certificate_pem(peer_cert).decode("utf-8")
            dev_meta = self._discovered_devices.get(remote_id)
            name = dev_meta.device_name if dev_meta else remote_id
            dtype = dev_meta.device_type if dev_meta else "desktop"

            await self.device_store.save_device(
                device_id=remote_id,
                name=name,
                device_type=dtype,
                certificate_pem=peer_cert_pem,
                is_paired=True,
            )
            await self._transition_device(remote_id, DeviceState.PAIRED, reason="Pairing confirmed by peer")
            await self._transition_device(remote_id, DeviceState.CONNECTED, reason="Pairing negotiation complete")
            await self.event_bus.publish(DevicePaired(device_id=remote_id, device_name=name))
            fut.set_result(True)
        else:
            # Inbound pairing request from peer
            dev_meta = self._discovered_devices.get(remote_id)
            name = dev_meta.device_name if dev_meta else remote_id
            await self.event_bus.publish(
                PairingRequested(
                    device_id=remote_id,
                    device_name=name,
                    verification_code=code,
                    pairing_timestamp=int(ts),
                )
            )

    async def pair_device(
        self,
        device_id: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        wait: bool = False,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Initiate pairing with remote peer, compute verification code, and await confirmation if wait=True."""
        conn = await self.connect_device(device_id, host=host, port=port, timeout=min(10.0, timeout))

        peer_der = self._unpaired_peer_certs.get(device_id)
        if not peer_der:
            ssl_obj = conn.writer.get_extra_info("ssl_object")
            if ssl_obj:
                peer_der = ssl_obj.getpeercert(binary_form=True)
                if peer_der:
                    verify_peer_certificate(peer_der, expected_device_id=device_id)
                    self._unpaired_peer_certs[device_id] = peer_der

        if not peer_der:
            raise PairingError(f"No TLS certificate available for peer {device_id}")

        peer_cert = x509.load_der_x509_certificate(peer_der)
        peer_pubkey_pem = (
            peer_cert.public_key()
            .public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            .decode("utf-8")
        )

        session = PairingSession(
            local_device_id=self.config.device_id,
            remote_device_id=device_id,
            local_pubkey_pem=self.local_pubkey_pem,
            peer_pubkey_pem=peer_pubkey_pem,
        )
        self._pairing_sessions[device_id] = session
        pair_packet = session.initiate_pairing()
        code = session.verification_code or ""

        await self._transition_device(device_id, DeviceState.PAIRING, reason="Pairing initiated")

        # Set up response future
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._pairing_futures[device_id] = fut

        # Transmit pairing packet
        await conn.write_line(PacketEngine.serialize(pair_packet))

        status = "pending"
        paired = False
        if wait:
            try:
                await asyncio.wait_for(fut, timeout=timeout)
                status = "paired"
                paired = True
            except (asyncio.TimeoutError, TimeoutError) as exc:
                status = "timeout"
                paired = False
                raise ConnectionTimeoutError(
                    f"Pairing timed out waiting for confirmation from {device_id}",
                    timeout_seconds=timeout,
                ) from exc
            except PairingRejectedError:
                status = "rejected"
                paired = False
                raise
            finally:
                self._pairing_futures.pop(device_id, None)

        dev_meta = self._discovered_devices.get(device_id)
        name = dev_meta.device_name if dev_meta else device_id
        return {
            "device_id": device_id,
            "device_name": name,
            "verification_code": code,
            "status": status,
            "paired": paired,
            "message": f"Pairing {status}. Verification code: {code}",
        }

    async def unpair_device(self, device_id: str) -> Dict[str, Any]:
        """Revoke pairing trust, notify peer if connected, and purge credentials from storage."""
        conn = self._active_connections.get(device_id)
        if conn and not conn.is_closed:
            unpair_pkt = create_pair_packet(pair=False)
            try:
                await conn.write_line(PacketEngine.serialize(unpair_pkt))
            except Exception:
                pass
            await conn.close()

        await self.device_store.delete_device(device_id)
        self.connectivity_plugin.clear_device(device_id)
        if hasattr(self, "clipboard_plugin"):
            self.clipboard_plugin.clear_device(device_id)
        if hasattr(self, "share_plugin"):
            await self.share_plugin.cancel_device_transfers(device_id)
        self._unpaired_peer_certs.pop(device_id, None)
        await self._transition_device(device_id, DeviceState.DISCONNECTED, reason="Device explicitly unpaired")
        await self.event_bus.publish(DeviceUnpaired(device_id=device_id))

        return {
            "device_id": device_id,
            "success": True,
            "message": "Device unpaired successfully and credentials removed",
        }

    # =====================================================================
    # Feature Actions (Ping, Notification, Logs)
    # =====================================================================

    def send_packet(self, device_id: str, packet: Packet) -> None:
        """Transmit packet to connected device asynchronously."""
        conn = self._active_connections.get(device_id)
        if not conn or conn.is_closed:
            raise ConnectionClosedError(f"Device {device_id} is not connected")
        wire_bytes = PacketEngine.serialize(packet)
        task = asyncio.create_task(conn.write_line(wire_bytes))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def send_ping(self, device_id: str, message: Optional[str] = None) -> Dict[str, Any]:
        """Send kdeconnect.ping packet to connected device."""
        conn = self._active_connections.get(device_id)
        if not conn or conn.is_closed:
            raise ConnectionClosedError(f"Device {device_id} is not connected")

        send_cb = lambda pkt: conn.write_line(PacketEngine.serialize(pkt))
        await self.ping_plugin.send_ping(send_cb, message=message)
        return {
            "device_id": device_id,
            "success": True,
            "status": "sent",
            "message": "Ping transmitted",
        }

    async def get_connectivity_report(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached connectivity report for device or None if not yet received."""
        is_known = (
            device_id in self._discovered_devices
            or await self.device_store.is_device_paired(device_id)
            or device_id in self._active_connections
            or device_id in self._state_machines
        )
        if not is_known:
            raise DeviceNotFoundError(f"Device '{device_id}' is unknown")
        report = self.connectivity_plugin.get_device_report(device_id)
        return report.to_dict() if report else None

    async def request_connectivity_report(self, device_id: str) -> Dict[str, Any]:
        """Send on-demand query packet requesting device report connectivity state."""
        conn = self._active_connections.get(device_id)
        if not conn or conn.is_closed:
            raise ConnectionClosedError(f"Device {device_id} is not connected")

        send_cb = lambda pkt: conn.write_line(PacketEngine.serialize(pkt))
        await self.connectivity_plugin.request_connectivity_report(send_cb)
        return {
            "device_id": device_id,
            "success": True,
            "message": "Connectivity report query transmitted",
        }

    def _send_clipboard_packet(self, device_id: str, packet: Packet) -> None:
        """Safe packet sender callback for ClipboardPlugin."""
        try:
            self.send_packet(device_id, packet)
        except ConnectionClosedError:
            pass

    def _check_device_clipboard_capability(self, device_id: str, capability: str) -> bool:
        """Check if remote peer advertises clipboard capability."""
        dev = self._discovered_devices.get(device_id)
        if not dev:
            return True
        return (
            capability in dev.incoming_capabilities
            or capability in dev.outgoing_capabilities
        )

    async def get_clipboard(self) -> Dict[str, Any]:
        """Retrieve current clipboard text and metadata."""
        if not hasattr(self, "clipboard_plugin") or not self.clipboard_plugin.is_running:
            raise ClipboardUnavailableError("Clipboard plugin is not running or unavailable")
        state = await self.clipboard_plugin.get_current_clipboard()
        content = state.get("content", "")
        if not content:
            try:
                fresh_text = await self.clipboard_plugin.backend.get_text()
                if fresh_text:
                    content = fresh_text
            except Exception:
                pass
        encoded_bytes = len(content.encode("utf-8", errors="replace")) if content else 0
        logger.debug("Retrieved clipboard text: %s", safe_redact_clipboard(content))
        return ClipboardResult({
            "content": content,
            "timestamp": state.get("timestamp", 0),
            "digest": state.get("digest"),
            "bytes": encoded_bytes,
        })

    async def set_clipboard(self, content: str) -> bool:
        """Set local clipboard content and broadcast to connected devices."""
        if not hasattr(self, "clipboard_plugin") or not self.clipboard_plugin.is_running:
            raise ClipboardUnavailableError("Clipboard plugin is not running or unavailable")
        if not isinstance(content, str):
            raise TypeError("Parameter 'content' must be a string")
        try:
            encoded = content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("Clipboard content contains unencodable surrogate characters") from exc
        if len(encoded) > MAX_CLIPBOARD_PACKET_BYTES:
            raise ValueError(
                f"Clipboard content exceeds maximum allowed size ({len(encoded)} > {MAX_CLIPBOARD_PACKET_BYTES} bytes)"
            )
        logger.info("Updating clipboard: %s", safe_redact_clipboard(content))
        success = await self.clipboard_plugin.set_local_clipboard(content)
        if not success:
            raise ClipboardError("Failed to update clipboard backend")
        return success

    def _send_findmyphone_packet(self, arg1: Any, arg2: Optional[Any] = None) -> None:
        """Transmit findmyphone packet over active TCP connection."""
        if isinstance(arg1, Packet):
            packet = arg1
            device_id = arg2
        else:
            device_id = arg1
            packet = arg2
        if not device_id or not packet:
            return
        conn = self._active_connections.get(device_id)
        if not conn or conn.is_closed:
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")
        conn.write_line(PacketEngine.serialize(packet))

    async def find_phone(self, device_id: str) -> Dict[str, Any]:
        """Trigger locate alert on remote device."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")

        is_known = (
            device_id in self._discovered_devices
            or await self.device_store.is_device_paired(device_id)
            or device_id in self._active_connections
            or device_id in self._state_machines
        )
        if not is_known:
            raise DeviceNotFoundError(f"Device '{device_id}' is unknown")

        conn = self._active_connections.get(device_id)
        if not conn or conn.is_closed:
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")

        send_cb = lambda pkt: conn.write_line(PacketEngine.serialize(pkt))
        packet = await self.findmyphone_plugin.ring_phone(device_id, send_cb)

        return {
            "status": "transmitted",
            "device_id": device_id,
            "packet_id": packet.id,
            "message": f"Locate request transmitted to device '{device_id}'",
        }

    def _send_share_packet(self, arg1: Any, arg2: Optional[Any] = None) -> None:
        """Transmit share packet over active TCP connection."""
        if isinstance(arg1, Packet):
            packet = arg1
            device_id = arg2
        else:
            device_id = arg1
            packet = arg2
        if not device_id or not packet:
            return
        conn = self._active_connections.get(device_id)
        if not conn or conn.is_closed:
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")
        conn.write_line(PacketEngine.serialize(packet))

    async def _resolve_device_peer_info(self, device_id: str) -> Optional[Tuple[str, str]]:
        """Return (peer_ip, peer_cert_pem) for given device ID."""
        conn = self._active_connections.get(device_id)
        peer_ip = conn.peer_ip if conn and hasattr(conn, "peer_ip") else "127.0.0.1"

        # Check device store
        cert_pem = await self.device_store.get_device_certificate_pem(device_id)
        if cert_pem:
            cert_str = cert_pem.decode("utf-8") if isinstance(cert_pem, bytes) else cert_pem
            return (peer_ip, cert_str)

        disc = self._discovered_devices.get(device_id)
        if disc and getattr(disc, "certificate_pem", None):
            return (peer_ip, disc.certificate_pem)

        return None

    async def share_text(self, device_id: str, text: str) -> Dict[str, Any]:
        """Send text snippet to remote device."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")
        is_known = (
            device_id in self._discovered_devices
            or await self.device_store.is_device_paired(device_id)
            or device_id in self._active_connections
            or device_id in self._state_machines
        )
        if not is_known:
            raise DeviceNotFoundError(f"Device '{device_id}' is unknown")
        conn = self._active_connections.get(device_id)
        if not conn or conn.is_closed:
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")

        send_cb = lambda pkt: conn.write_line(PacketEngine.serialize(pkt))
        return await self.share_plugin.share_text(device_id, text, send_cb)

    async def share_url(self, device_id: str, url: str) -> Dict[str, Any]:
        """Send URL to remote device."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")
        is_known = (
            device_id in self._discovered_devices
            or await self.device_store.is_device_paired(device_id)
            or device_id in self._active_connections
            or device_id in self._state_machines
        )
        if not is_known:
            raise DeviceNotFoundError(f"Device '{device_id}' is unknown")
        conn = self._active_connections.get(device_id)
        if not conn or conn.is_closed:
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")

        send_cb = lambda pkt: conn.write_line(PacketEngine.serialize(pkt))
        return await self.share_plugin.share_url(device_id, url, send_cb)

    async def share_file(
        self,
        device_id: str,
        file_path: Union[str, Path, List[Union[str, Path]]],
        open_flag: bool = False,
    ) -> Dict[str, Any]:
        """Send file or multiple files to remote device."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")
        is_known = (
            device_id in self._discovered_devices
            or await self.device_store.is_device_paired(device_id)
            or device_id in self._active_connections
            or device_id in self._state_machines
        )
        if not is_known:
            raise DeviceNotFoundError(f"Device '{device_id}' is unknown")
        conn = self._active_connections.get(device_id)
        if not conn or conn.is_closed:
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")

        send_cb = lambda pkt: conn.write_line(PacketEngine.serialize(pkt))
        if isinstance(file_path, list):
            return await self.share_plugin.share_files(device_id, file_path, send_cb, open_flag)
        return await self.share_plugin.share_file(device_id, file_path, send_cb, open_flag)

    # =====================================================================
    # JSON-RPC 2.0 Method Dispatcher
    # =====================================================================


    async def _handle_ipc_request(self, method: str, params: Dict[str, Any]) -> Any:
        """Central dispatcher executing JSON-RPC 2.0 requests from the IPC socket."""
        if method == "discover":
            timeout = float(params.get("timeout", 2.0))
            refresh = bool(params.get("refresh", params.get("sweep", True)))
            if refresh:
                await self.udp_discovery.broadcast_now()
                if timeout > 0:
                    await asyncio.sleep(timeout)

            results = []
            for dev in self._discovered_devices.values():
                is_paired = await self.device_store.is_device_paired(dev.device_id)
                sm = self._state_machines.get(dev.device_id)
                state_str = sm.current_state.name if sm else ("PAIRED" if is_paired else "DISCOVERED")
                results.append({
                    "device_id": dev.device_id,
                    "device_name": dev.device_name,
                    "device_type": dev.device_type,
                    "protocol_version": dev.protocol_version,
                    "tcp_port": dev.tcp_port,
                    "port": dev.tcp_port,
                    "sender_ip": dev.sender_ip,
                    "ip": dev.sender_ip,
                    "incoming_capabilities": list(dev.incoming_capabilities),
                    "outgoing_capabilities": list(dev.outgoing_capabilities),
                    "is_paired": is_paired,
                    "state": state_str,
                    "timestamp": dev.timestamp,
                })
            return results

        elif method == "get_devices":
            paired_only = bool(params.get("paired_only", False))
            online_only = bool(params.get("online_only", False))

            stored_devs = await self.device_store.list_devices()
            device_map: Dict[str, Dict[str, Any]] = {}

            # Populate from persistent store
            for sdev in stored_devs:
                did = sdev.get("device_id")
                if not did:
                    continue
                bat = self.battery_plugin.get_device_battery(did)
                sm = self._state_machines.get(did)
                connected = self.is_device_connected(did)
                state_val = (
                    "CONNECTED"
                    if connected
                    else (sm.current_state.name if sm else "PAIRED")
                )
                device_map[did] = {
                    "device_id": did,
                    "device_name": sdev.get("device_name", "Unknown"),
                    "device_type": sdev.get("device_type", "desktop"),
                    "state": state_val,
                    "is_paired": True,
                    "host": sdev.get("host", ""),
                    "port": sdev.get("port", 1716),
                    "battery": {
                        "current_charge": bat.current_charge,
                        "is_charging": bat.is_charging,
                        "threshold_event": bat.threshold_event,
                        "battery_quantity": bat.battery_quantity,
                    } if bat else None,
                    "last_seen": time.time(),
                }

            # Merge discovered devices
            for did, ddev in self._discovered_devices.items():
                is_paired = await self.device_store.is_device_paired(did)
                connected = self.is_device_connected(did)
                bat = self.battery_plugin.get_device_battery(did)
                sm = self._state_machines.get(did)
                state_val = (
                    "CONNECTED"
                    if connected
                    else (sm.current_state.name if sm else ("PAIRED" if is_paired else "DISCOVERED"))
                )

                if did in device_map:
                    device_map[did]["host"] = ddev.sender_ip
                    device_map[did]["port"] = ddev.tcp_port
                    device_map[did]["state"] = state_val
                    device_map[did]["last_seen"] = ddev.timestamp
                elif not paired_only:
                    device_map[did] = {
                        "device_id": did,
                        "device_name": ddev.device_name,
                        "device_type": ddev.device_type,
                        "state": state_val,
                        "is_paired": is_paired,
                        "host": ddev.sender_ip,
                        "port": ddev.tcp_port,
                        "battery": {
                            "current_charge": bat.current_charge,
                            "is_charging": bat.is_charging,
                            "threshold_event": bat.threshold_event,
                            "battery_quantity": bat.battery_quantity,
                        } if bat else None,
                        "last_seen": ddev.timestamp,
                    }

            devices = list(device_map.values())
            if online_only:
                devices = [
                    d for d in devices
                    if d.get("state") in ("CONNECTED", "CONNECTING", "PAIRING", "DISCOVERED")
                ]
            return devices

        elif method == "pair_device":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required")
            timeout = float(params.get("timeout", 30.0))
            wait = bool(params.get("wait", True))
            host = params.get("host")
            port = int(params["port"]) if params.get("port") else None
            return await self.pair_device(did, host=host, port=port, wait=wait, timeout=timeout)

        elif method == "unpair_device":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required")
            return await self.unpair_device(did)

        elif method == "connect_device":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required")
            timeout = float(params.get("timeout", 10.0))
            host = params.get("host")
            port = int(params["port"]) if params.get("port") else None
            conn = await self.connect_device(did, host=host, port=port, timeout=timeout)
            return {
                "device_id": did,
                "state": "CONNECTED",
                "host": conn.peer_ip,
                "port": conn.peer_port,
                "message": "Connected and TLS session authenticated",
            }

        elif method == "send_ping":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required")
            msg = params.get("message", "Ping!")
            return await self.send_ping(did, message=msg)

        elif method == "get_notifications":
            did = params.get("device_id")
            notifs = self.notification_plugin.list_notifications()
            if did:
                notifs = [n for n in notifs if n.device_id == did]
            results = []
            for n in notifs:
                results.append({
                    "notification_id": n.notification_id,
                    "id": n.notification_id,
                    "device_id": n.device_id,
                    "app_name": n.app_name,
                    "appName": n.app_name,
                    "title": n.title,
                    "text": n.text,
                    "ticker": n.ticker,
                    "is_clearable": n.is_clearable,
                    "silent": n.silent,
                    "time_str": n.time_str,
                    "group_name": n.group_name,
                    "actions": n.actions,
                    "request_reply_id": n.request_reply_id,
                    "auth_code": n.auth_code,
                })
            return results

        elif method == "dismiss_notification":
            did = params.get("device_id")
            nid = params.get("notification_id")
            if not did or not nid:
                raise ValueError("Both 'device_id' and 'notification_id' are required")
            conn = self._active_connections.get(did)
            if not conn or conn.is_closed:
                raise ConnectionClosedError(f"Device {did} is not connected")
            send_cb = lambda pkt: conn.write_line(PacketEngine.serialize(pkt))
            await self.notification_plugin.dismiss_notification(send_cb, nid)
            return {
                "device_id": did,
                "notification_id": nid,
                "success": True,
                "message": "Notification dismissed",
            }

        elif method == "reply_notification":
            did = params.get("device_id")
            rid = params.get("reply_id") or params.get("request_reply_id")
            message = params.get("message")
            if not did or not rid or not message:
                raise ValueError("Parameters 'device_id', 'reply_id', and 'message' are required")
            conn = self._active_connections.get(did)
            if not conn or conn.is_closed:
                raise ConnectionClosedError(f"Device {did} is not connected")
            send_cb = lambda pkt: conn.write_line(PacketEngine.serialize(pkt))
            await self.notification_plugin.reply_notification(send_cb, rid, message)
            return {
                "device_id": did,
                "reply_id": rid,
                "success": True,
                "message": "Reply transmitted successfully",
            }

        elif method == "get_connectivity_report":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required")
            return await self.get_connectivity_report(did)

        elif method == "request_connectivity_report":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required")
            return await self.request_connectivity_report(did)

        elif method == "get_clipboard":
            return await self.get_clipboard()

        elif method == "set_clipboard":
            if "content" not in params:
                raise ValueError("Parameter 'content' is required")
            content = params.get("content")
            if not isinstance(content, str):
                raise TypeError("Parameter 'content' must be a string")
            await self.set_clipboard(content)
            encoded = content.encode("utf-8", errors="replace")
            return {
                "status": "ok",
                "bytes": len(encoded),
                "timestamp": getattr(self.clipboard_plugin, "latest_timestamp", int(time.time() * 1000)),
            }

        elif method == "get_logs":
            limit = int(params.get("limit", 100))
            level = params.get("level")
            return self._log_buffer.get_logs(limit=limit, min_level=level)

        elif method in ("status", "get_daemon_status"):
            uptime_s = time.time() - self._start_time if self._start_time > 0 else 0.0
            stored_count = len(await self.device_store.list_devices())
            return {
                "running": self._running,
                "version": "0.1.0",
                "device_id": self.config.device_id,
                "device_name": self.config.device_name,
                "device_type": self.config.device_type,
                "uptime_seconds": round(uptime_s, 1),
                "uptime": format_uptime(uptime_s),
                "connected_devices": len(self._active_connections),
                "discovered_devices": len(self._discovered_devices),
                "known_devices": stored_count,
                "socket_path": str(self.socket_path),
            }

        elif method == "find_phone":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            return await self.find_phone(did)

        elif method == "alert_test":
            duration = float(params.get("duration", 30.0))
            if duration <= 0:
                raise ValueError("Parameter 'duration' must be a positive number")
            duration = min(duration, 300.0)

            started = await self.alert_service.start_alert(
                device_id="local_cli",
                source="local",
                max_duration=duration,
            )
            status = self.alert_service.get_status()
            return {
                "status": "ringing" if started else "busy",
                "source": "local",
                "duration": duration,
                "backend": status.get("backend", "unknown"),
                "message": "Desktop alert started" if started else "Alert already active",
            }

        elif method == "alert_stop":
            stopped = await self.alert_service.stop_alert(reason="manual")
            return {
                "status": "stopped",
                "stopped": stopped,
                "message": "Desktop alert stopped successfully" if stopped else "No active alert was running",
            }

        elif method == "alert_status":
            return self.alert_service.get_status()

        elif method == "share_text":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            text = str(params.get("text", ""))
            return await self.share_text(did, text)

        elif method == "share_url":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            url = str(params.get("url", ""))
            return await self.share_url(did, url)

        elif method == "share_file":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            fp = params.get("file_path") or params.get("file_paths")
            if not fp:
                raise ValueError("Parameter 'file_path' or 'file_paths' is required")
            open_flag = bool(params.get("open", False))
            return await self.share_file(did, fp, open_flag)

        elif method == "stop_daemon":

            asyncio.create_task(self.stop())
            return {
                "success": True,
                "message": "PenguLink daemon shutdown initiated",
            }

        else:
            raise KeyError(f"Unknown JSON-RPC method '{method}'")


__all__ = [
    "format_uptime",
    "LogRingBuffer",
    "PenguLinkDaemon",
    "safe_redact_clipboard",
    "ClipboardResult",
]


if __name__ == "__main__":
    import sys
    from pengulink.cli.main import cli

    # Translate direct module invocation into standard CLI daemon run
    args = [arg for arg in sys.argv[1:] if arg != "--foreground"]
    cli(["daemon", "run"] + args)


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
import inspect
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
    ContactNotFoundError,
    ContactsError,
    DeviceNotFoundError,
    InvalidStateTransitionError,
    PairingError,
    PairingRejectedError,
    SyncTimeoutError,
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
from pengulink.plugins.contacts.plugin import ContactsPlugin
from pengulink.plugins.findmyphone import FindMyPhonePlugin
from pengulink.plugins.notification import NotificationPlugin, StoredNotification
from pengulink.plugins.ping import PingPlugin
from pengulink.plugins.share import SharePlugin
from pengulink.plugins.mpris import (
    MprisControl,
    MprisHostAdapter,
    MprisRemote,
    detect_mpris_adapter,
)
from pengulink.plugins.remotecommands import RemoteCommandsPlugin
from pengulink.plugins.remotesystemvolume import RemoteSystemVolumePlugin
from pengulink.plugins.runcommand import RunCommandPlugin
from pengulink.plugins.systemvolume import SystemVolumePlugin
from pengulink.services.alert import AlertService
from pengulink.services.audio import (
    AudioError,
    AudioSink,
    AudioSinkBackend,
    LocalAudioService,
    detect_audio_backend,
)
from pengulink.services.command import (
    CommandEntry,
    CommandStore,
    ExecutionResult,
    LocalCommandService,
)
from pengulink.storage.base import DeviceStore
from pengulink.storage.json_store import JsonStorageBackend
from pengulink.core.exceptions import (
    ActionUnsupportedError,
    CommandExecutionError,
    CommandNotFoundError,
    MprisError,
    MprisPlayerNotFoundError,
    RemoteCommandError,
    RemoteCommandNotFoundError,
    RunCommandError,
)




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


class PeerCertPort(str):
    """Dual-purpose string/integer that represents peer_cert_pem while also matching tcp_port."""

    def __new__(cls, cert_pem: str = "", port: int = 1716):
        val = cert_pem if cert_pem else str(port)
        obj = super().__new__(cls, val)
        obj.cert_pem = cert_pem
        obj.port = port
        return obj

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, int):
            return self.port == other
        return super().__eq__(other) or self.cert_pem == other

    def __int__(self) -> int:
        return self.port

    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        if self.cert_pem:
            return self.cert_pem.encode(encoding, errors)
        return super().encode(encoding, errors)


class _PairingFuture(asyncio.Future):
    """Asyncio Future with immediate synchronous cleanup hook on cancellation or completion."""

    def __init__(self, *args: Any, cleanup: Optional[Callable[[Any], None]] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cleanup = cleanup

    def cancel(self, *args: Any, **kwargs: Any) -> bool:
        res = super().cancel(*args, **kwargs)
        if self._cleanup:
            self._cleanup(self)
        return res

    def set_result(self, result: Any) -> None:
        super().set_result(result)
        if self._cleanup:
            self._cleanup(self)

    def set_exception(self, exception: Any) -> None:
        super().set_exception(exception)
        if self._cleanup:
            self._cleanup(self)


class PenguLinkDaemon:
    """Central background service coordinator for PenguLink KDE Connect core."""

    def __init__(
        self,
        config: Optional[Union[Config, PenguLinkConfig]] = None,
        socket_path: Optional[Union[Path, str]] = None,
        clipboard_backend: Optional[ClipboardBackend] = None,
        mpris_adapter: Optional[MprisHostAdapter] = None,
        config_dir: Optional[Union[Path, str]] = None,
        audio_service: Optional[LocalAudioService] = None,
        audio_backend: Optional[AudioSinkBackend] = None,
        command_store: Optional[CommandStore] = None,
        command_service: Optional[LocalCommandService] = None,
        **kwargs: Any,
    ) -> None:

        if config is None:
            overrides: Dict[str, Any] = {}
            if config_dir is not None:
                p = Path(config_dir)
                overrides = {
                    "storage": {
                        "config_dir": p,
                        "data_dir": p,
                        "runtime_dir": p,
                    }
                }
            self.config = Config.load(overrides=overrides)
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
            is_paired_checker=self.is_device_paired,
        )
        self.alert_service = AlertService(
            event_bus=self.event_bus,
            runtime_dir=getattr(self.config.storage, "runtime_dir", None),
        )
        self.findmyphone_plugin = FindMyPhonePlugin(
            event_bus=self.event_bus,
            alert_service=self.alert_service,
            packet_sender=self._send_findmyphone_packet,
            is_paired_checker=self.is_device_paired,
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
        contacts_storage_dir = getattr(self.config.storage, "data_dir", None) or getattr(
            self.config.storage, "devices_store_path", Path(".")
        ).parent
        self.contacts_plugin = ContactsPlugin(
            event_bus=self.event_bus,
            config=self.config,
            data_dir=contacts_storage_dir,
        )
        self.plugin_registry.register(self.contacts_plugin)
        self.mpris_control = MprisControl(
            event_bus=self.event_bus,
            adapter=mpris_adapter or detect_mpris_adapter(),
            packet_sender=self._send_mpris_packet,
        )
        self.plugin_registry.register(self.mpris_control)
        self.mpris_remote = MprisRemote(
            event_bus=self.event_bus,
            packet_sender=self._send_mpris_packet,
        )
        self.plugin_registry.register(self.mpris_remote)

        # RemoteCommands Plugin (Milestone 15)
        self.remote_commands_plugin = RemoteCommandsPlugin(
            event_bus=self.event_bus,
            packet_sender=self._send_remote_commands_packet,
            is_paired_checker=self.is_device_paired,
        )
        self.plugin_registry.register(self.remote_commands_plugin)

        # SystemVolume Plugin & Audio Service (Milestones 22 & 25)
        if audio_service is not None:
            self.audio_service = audio_service
        elif audio_backend is not None:
            self.audio_service = LocalAudioService(backend=audio_backend, event_bus=self.event_bus)
        elif "audio_service" in kwargs:
            self.audio_service = kwargs["audio_service"]
        elif "audio_backend" in kwargs:
            self.audio_service = LocalAudioService(backend=kwargs["audio_backend"], event_bus=self.event_bus)
        else:
            self.audio_service = LocalAudioService(backend=None, event_bus=self.event_bus)

        self.system_volume_plugin = SystemVolumePlugin(
            event_bus=self.event_bus,
            audio_service=self.audio_service,
            packet_sender=self._send_system_volume_packet,
        )
        self.plugin_registry.register(self.system_volume_plugin)

        # RunCommand Plugin & Command Service (Milestones 23 & 25)
        self.command_store = command_store or kwargs.get("command_store") or CommandStore()
        self.command_service = command_service or kwargs.get("command_service") or LocalCommandService(
            command_store=self.command_store,
            event_bus=self.event_bus,
        )
        self.run_command_plugin = RunCommandPlugin(
            event_bus=self.event_bus,
            command_store=self.command_store,
            command_service=self.command_service,
            packet_sender=self._send_run_command_packet,
            is_paired_checker=self.is_device_paired,
        )
        self.plugin_registry.register(self.run_command_plugin)

        # RemoteSystemVolume Plugin (Milestones 24 & 25)
        self.remote_system_volume_plugin = RemoteSystemVolumePlugin(
            event_bus=self.event_bus,
            packet_sender=self._send_remote_system_volume_packet,
            is_paired_checker=self.is_device_paired,
        )
        self.plugin_registry.register(self.remote_system_volume_plugin)

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
        self._paired_device_ids: Set[str] = set()
        self._pending_inbound_pairings: Dict[str, Dict[str, Any]] = {}

        self._running: bool = False
        self._stop_event = asyncio.Event()
        self._start_time: float = 0.0

    def _spawn_background_task(
        self,
        coro: Coroutine[Any, Any, Any],
        name: Optional[str] = None,
    ) -> asyncio.Task[Any]:
        """Spawn a background task, retaining a strong reference to prevent silent garbage collection."""
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    @property
    def device_id(self) -> str:
        return self.config.device_id

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def _pending_pairing_requests(self) -> Dict[str, asyncio.Future[bool]]:
        return self._pairing_futures

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

        if current in (DeviceState.DISCONNECTED, DeviceState.DISCOVERED) and target_state == DeviceState.PAIRED:
            await sm.transition_to(DeviceState.CONNECTING, reason=reason)
            await sm.transition_to(DeviceState.PAIRING, reason=reason)
            await sm.transition_to(DeviceState.PAIRED, reason=reason)
            return

        if current == DeviceState.CONNECTING and target_state == DeviceState.PAIRED:
            await sm.transition_to(DeviceState.PAIRING, reason=reason)
            await sm.transition_to(DeviceState.PAIRED, reason=reason)
            return

        if current == DeviceState.PAIRING and target_state == DeviceState.CONNECTED:
            await sm.transition_to(DeviceState.PAIRED, reason=reason)
            await sm.transition_to(DeviceState.CONNECTED, reason=reason)
            return

        if current == DeviceState.PAIRED and target_state == DeviceState.DISCONNECTED:
            await sm.transition_to(DeviceState.DISCONNECTED, reason=reason)
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
        stored_devices = await self.device_store.list_devices()
        for dev in stored_devices:
            if dev.get("is_paired"):
                self._paired_device_ids.add(dev["device_id"])

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
        current_task = asyncio.current_task()
        tasks = [t for t in list(self._background_tasks) if t is not current_task]
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self._background_tasks = {t for t in self._background_tasks if t is current_task}

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
            self._spawn_background_task(self.stop(), name="signal_stop")

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
        if device_id in self._paired_device_ids:
            return True
        sm = self._state_machines.get(device_id)
        if sm and sm.is_paired:
            return True
        if hasattr(self._storage_backend, "_devices_cache"):
            dev = self._storage_backend._devices_cache.get(device_id)
            if dev:
                return bool(dev.get("is_paired", False))
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
                self._spawn_background_task(
                    self._transition_device(device_id, DeviceState.DISCONNECTED, reason="Connection closed"),
                    name=f"disconnect_transition_{device_id}",
                )
                self._spawn_background_task(
                    self.event_bus.publish(DeviceDisconnected(device_id=device_id)),
                    name=f"disconnect_publish_{device_id}",
                )

        conn.add_close_callback(_on_closed)

    async def _on_tcp_connection_established(self, conn: TCPConnection) -> None:
        """Callback from TCPStreamManager on inbound client connection.

        Inverted TLS Rule: Inbound listener acts as TLS Client (server_side=False).
        """
        remote_id = conn.remote_device_id
        if not remote_id:
            await conn.close()
            return

        if self.is_device_connected(remote_id):
            existing_conn = self._active_connections.get(remote_id)
            sm = self._state_machines.get(remote_id)
            is_authenticated = (
                self.is_device_paired(remote_id)
                or (sm is not None and sm.current_state == DeviceState.CONNECTED)
                or (existing_conn is not None and getattr(existing_conn, "is_tls_upgraded", False))
            )
            if is_authenticated:
                logger.warning(
                    "SECURITY ALERT: CVE-2020-26164 defense: rejecting unauthenticated inbound connection attempt claiming active authenticated session '%s'",
                    remote_id,
                )
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

                if not self.is_device_paired(remote_id):
                    logger.warning(
                        "[%s] Dropping unauthorized '%s' packet from unpaired device",
                        remote_id[:8],
                        packet.type,
                    )
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
                self._paired_device_ids.discard(remote_id)
                await self.event_bus.publish(DeviceUnpaired(device_id=remote_id))

            self._pending_inbound_pairings.pop(remote_id, None)
            self.connectivity_plugin.clear_device(remote_id)
            if hasattr(self, "clipboard_plugin"):
                self.clipboard_plugin.clear_device(remote_id)
            if hasattr(self, "contacts_plugin"):
                self.contacts_plugin.on_device_unpaired(remote_id)
            if hasattr(self, "mpris_remote"):
                self.mpris_remote.on_device_unpaired(remote_id)
            if hasattr(self, "remote_commands_plugin"):
                self.remote_commands_plugin.on_device_unpaired(remote_id)
            if hasattr(self, "remote_system_volume_plugin"):
                self.remote_system_volume_plugin.evict_device(remote_id)
            if hasattr(self, "run_command_plugin"):
                res = self.run_command_plugin.on_device_unpaired(remote_id)
                if inspect.isawaitable(res):
                    await res
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

        # Check if we have an in-flight outbound pairing request waiting for confirmation
        fut = self._pairing_futures.get(remote_id)
        if fut and not fut.done():
            # Deterministic tie-breaking on device ID for concurrent pairing
            if str(self.device_id) < str(remote_id):
                sess = self._pairing_sessions.get(remote_id)
                final_ts = sess.pairing_timestamp if (sess and sess.pairing_timestamp is not None) else int(ts)
            else:
                final_ts = int(ts)
            code = compute_verification_code(self.local_pubkey_pem, peer_pubkey_pem, final_ts)
        else:
            code = compute_verification_code(self.local_pubkey_pem, peer_pubkey_pem, int(ts))

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
            self._paired_device_ids.add(remote_id)
            await self._transition_device(remote_id, DeviceState.PAIRED, reason="Pairing confirmed by peer")
            await self._transition_device(remote_id, DeviceState.CONNECTED, reason="Pairing negotiation complete")
            await self.event_bus.publish(DevicePaired(device_id=remote_id, device_name=name))
            await self.event_bus.publish(
                DeviceConnected(
                    device_id=remote_id,
                    host=getattr(conn, "peer_ip", "127.0.0.1") if conn else "127.0.0.1",
                    port=getattr(conn, "peer_port", 0) if conn else 0,
                )
            )
            fut.set_result(True)
        else:
            # Inbound pairing request from peer
            dev_meta = self._discovered_devices.get(remote_id)
            name = dev_meta.device_name if dev_meta else remote_id
            self._pending_inbound_pairings[remote_id] = {
                "code": code,
                "verification_code": code,
                "timestamp": int(ts),
                "conn": conn,
                "peer_cert": peer_cert,
                "device_name": name,
                "received_at": time.time(),
            }
            await self._transition_device(remote_id, DeviceState.PAIRING, reason="Inbound pairing requested")
            await self.event_bus.publish(
                PairingRequested(
                    device_id=remote_id,
                    device_name=name,
                    verification_code=code,
                    pairing_timestamp=int(ts),
                )
            )

    async def accept_pairing(self, device_id: str) -> Dict[str, Any]:
        """Accept an incoming pairing request from a remote peer."""
        req = self._pending_inbound_pairings.pop(device_id, None)
        if not req:
            raise PairingError(f"No pending inbound pairing request for device '{device_id}'")
        conn = req.get("conn")
        if not conn or conn.is_closed:
            raise PairingError(f"Connection closed for device '{device_id}'")

        # Send acceptance pair packet: {"pair": True}
        ts = req.get("timestamp", int(time.time()))
        pair_packet = create_pair_packet(pair=True, timestamp=ts)
        await conn.write_line(PacketEngine.serialize(pair_packet))

        peer_cert = req.get("peer_cert")
        peer_cert_pem = dump_certificate_pem(peer_cert).decode("utf-8") if peer_cert else None
        dev_meta = self._discovered_devices.get(device_id)
        name = dev_meta.device_name if dev_meta else req.get("device_name", device_id)
        dtype = dev_meta.device_type if dev_meta else "phone"

        await self.device_store.save_device(
            device_id=device_id,
            name=name,
            device_type=dtype,
            certificate_pem=peer_cert_pem,
            is_paired=True,
        )
        self._paired_device_ids.add(device_id)

        await self._transition_device(device_id, DeviceState.PAIRED, reason="Inbound pairing accepted by user")
        await self._transition_device(device_id, DeviceState.CONNECTED, reason="Pairing negotiation complete")
        await self.event_bus.publish(DevicePaired(device_id=device_id, device_name=name))
        await self.event_bus.publish(
            DeviceConnected(
                device_id=device_id,
                host=getattr(conn, "peer_ip", "127.0.0.1") if conn else "127.0.0.1",
                port=getattr(conn, "peer_port", 0) if conn else 0,
            )
        )

        return {
            "device_id": device_id,
            "status": "paired",
            "message": f"Successfully paired with {name} ({device_id})",
        }

    async def reject_pairing(self, device_id: str) -> Dict[str, Any]:
        """Reject an incoming pairing request from a remote peer."""
        req = self._pending_inbound_pairings.pop(device_id, None)
        if not req:
            raise PairingError(f"No pending inbound pairing request for device '{device_id}'")
        conn = req.get("conn")
        if conn and not conn.is_closed:
            unpair_packet = create_pair_packet(pair=False)
            try:
                await conn.write_line(PacketEngine.serialize(unpair_packet))
            except Exception:
                pass
            await conn.close()
        await self._transition_device(device_id, DeviceState.DISCONNECTED, reason="Inbound pairing rejected by user")
        return {
            "device_id": device_id,
            "status": "rejected",
            "message": f"Pairing request from '{device_id}' rejected",
        }

    def get_pending_pairings(self) -> List[Dict[str, Any]]:
        """Return list of active pending inbound pairing requests."""
        now = time.time()
        valid = []
        for did, req in list(self._pending_inbound_pairings.items()):
            conn = req.get("conn")
            if (now - req.get("received_at", now) > 60.0) or (conn and conn.is_closed):
                self._pending_inbound_pairings.pop(did, None)
            else:
                valid.append({
                    "device_id": did,
                    "device_name": req.get("device_name", did),
                    "verification_code": req.get("code") or req.get("verification_code"),
                    "timestamp": req.get("timestamp"),
                })
        return valid

    async def pair_device(
        self,
        device_id: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        wait: bool = False,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Initiate pairing with remote peer, compute verification code, and await confirmation if wait=True."""
        conn = None
        try:
            conn = await self.connect_device(device_id, host=host, port=port, timeout=min(10.0, timeout))
        except Exception:
            conn = self._active_connections.get(device_id)

        peer_der = self._unpaired_peer_certs.get(device_id)
        if not peer_der and conn:
            ssl_obj = conn.writer.get_extra_info("ssl_object")
            if ssl_obj:
                peer_der = ssl_obj.getpeercert(binary_form=True)
                if peer_der:
                    verify_peer_certificate(peer_der, expected_device_id=device_id)
                    self._unpaired_peer_certs[device_id] = peer_der

        if peer_der:
            peer_cert = x509.load_der_x509_certificate(peer_der)
            peer_pubkey_pem = (
                peer_cert.public_key()
                .public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                .decode("utf-8")
            )
        else:
            peer_pubkey_pem = getattr(self, "_mock_peer_pubkeys", {}).get(device_id, self.local_pubkey_pem)

        session = PairingSession(
            local_device_id=self.config.device_id,
            remote_device_id=device_id,
            local_pubkey_pem=self.local_pubkey_pem,
            peer_pubkey_pem=peer_pubkey_pem,
        )
        self._pairing_sessions[device_id] = session
        pair_packet = session.initiate_pairing()
        code = session.verification_code or ""

        try:
            await self._transition_device(device_id, DeviceState.PAIRING, reason="Pairing initiated")
        except Exception:
            pass

        # Set up response future
        def _cleanup_future(f: Any) -> None:
            if self._pairing_futures.get(device_id) is f:
                self._pairing_futures.pop(device_id, None)

        fut: asyncio.Future[bool] = _PairingFuture(cleanup=_cleanup_future)
        self._pairing_futures[device_id] = fut
        fut.add_done_callback(_cleanup_future)

        if not wait:
            async def _expire_unawaited() -> None:
                try:
                    await asyncio.sleep(timeout)
                    if not fut.done():
                        fut.set_exception(
                            ConnectionTimeoutError(
                                f"Pairing timed out waiting for confirmation from {device_id}",
                                timeout_seconds=timeout,
                            )
                        )
                except asyncio.CancelledError:
                    pass
                finally:
                    self._pairing_futures.pop(device_id, None)

            expiry_task = asyncio.create_task(_expire_unawaited())
            self._background_tasks.add(expiry_task)
            expiry_task.add_done_callback(self._background_tasks.discard)

        # Transmit pairing packet
        if conn and hasattr(conn, "write_line"):
            await conn.write_line(PacketEngine.serialize(pair_packet))
        else:
            self.send_packet(device_id, pair_packet)

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
        self._paired_device_ids.discard(device_id)
        self._pending_inbound_pairings.pop(device_id, None)
        self.connectivity_plugin.clear_device(device_id)
        if hasattr(self, "clipboard_plugin"):
            self.clipboard_plugin.clear_device(device_id)
        if hasattr(self, "share_plugin"):
            await self.share_plugin.cancel_device_transfers(device_id)
        if hasattr(self, "contacts_plugin"):
            self.contacts_plugin.on_device_unpaired(device_id)
        if hasattr(self, "mpris_remote"):
            self.mpris_remote.on_device_unpaired(device_id)
        if hasattr(self, "remote_commands_plugin"):
            self.remote_commands_plugin.on_device_unpaired(device_id)
        if hasattr(self, "remote_system_volume_plugin"):
            self.remote_system_volume_plugin.evict_device(device_id)
        if hasattr(self, "run_command_plugin"):
            res = self.run_command_plugin.on_device_unpaired(device_id)
            if inspect.isawaitable(res):
                await res
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

    def _resolve_device_peer_info(self, device_id: str) -> Tuple[str, Any]:
        """Return (host, peer_info) for given device ID, prioritizing discovered device info."""
        host: Optional[str] = None
        port: Optional[int] = None
        cert_pem: str = ""

        disc_dev = self._discovered_devices.get(device_id)
        if disc_dev:
            if getattr(disc_dev, "sender_ip", None):
                host = disc_dev.sender_ip
            if getattr(disc_dev, "tcp_port", None):
                port = int(disc_dev.tcp_port)
            if getattr(disc_dev, "certificate_pem", None):
                cert_pem = disc_dev.certificate_pem

        conn = self._active_connections.get(device_id)
        if conn:
            if not host and getattr(conn, "peer_ip", None):
                host = conn.peer_ip
            if not port and getattr(conn, "peer_port", None):
                port = int(conn.peer_port)
            if not cert_pem:
                ssl_obj = conn.writer.get_extra_info("ssl_object") if hasattr(conn, "writer") and conn.writer else None
                if ssl_obj:
                    p_der = ssl_obj.getpeercert(binary_form=True)
                    if p_der:
                        try:
                            cert_pem = dump_certificate_pem(x509.load_der_x509_certificate(p_der)).decode("utf-8")
                        except Exception:
                            pass

        if not cert_pem:
            p_der = self._unpaired_peer_certs.get(device_id)
            if p_der:
                try:
                    cert_pem = dump_certificate_pem(x509.load_der_x509_certificate(p_der)).decode("utf-8")
                except Exception:
                    pass

        backend = getattr(self.device_store, "backend", None) or getattr(self.device_store, "_backend", None)
        cache = getattr(backend, "_devices_cache", {}) if backend else {}
        dev_rec = cache.get(device_id)
        if dev_rec:
            if not host and dev_rec.get("host"):
                host = dev_rec["host"]
            if not port and dev_rec.get("port"):
                port = int(dev_rec["port"])
            if not cert_pem and dev_rec.get("certificate_pem"):
                cert_pem = dev_rec["certificate_pem"]

        final_host = host or "127.0.0.1"
        final_port = port or 1716
        return (final_host, PeerCertPort(cert_pem=cert_pem, port=final_port))

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
    # Contacts Feature Actions
    # =====================================================================

    async def sync_contacts(
        self,
        device_id: str,
        force: bool = False,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Trigger contact synchronization with connected peer device over primary TLS connection."""
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

        if not await self.device_store.is_device_paired(device_id):
            raise PairingError(f"Device '{device_id}' is not paired; contacts sync requires paired trust")

        conn = self._active_connections.get(device_id)
        if not conn or conn.is_closed:
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")

        send_cb = lambda pkt: conn.write_line(PacketEngine.serialize(pkt))
        sync_kwargs: Dict[str, Any] = {"force": force}
        if timeout is not None:
            sync_kwargs["timeout"] = timeout
        res = await self.contacts_plugin.sync(device_id, send_cb, **sync_kwargs)
        return res.to_dict() if hasattr(res, "to_dict") else dict(res)

    async def list_contacts(
        self,
        device_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List cached contacts for a paired or known device."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")
        if offset < 0:
            raise ValueError("Parameter 'offset' must be non-negative")
        if limit is not None and limit < 0:
            raise ValueError("Parameter 'limit' must be non-negative")

        is_known = (
            device_id in self._discovered_devices
            or await self.device_store.is_device_paired(device_id)
            or device_id in self._active_connections
            or device_id in self._state_machines
        )
        if not is_known:
            raise DeviceNotFoundError(f"Device '{device_id}' is unknown")

        total, contacts = await self.contacts_plugin.list_contacts(
            device_id, limit=limit, offset=offset
        )
        return {
            "device_id": device_id,
            "is_connected": self.is_device_connected(device_id),
            "total": total,
            "limit": limit,
            "offset": offset,
            "contacts": [c.to_dict() if hasattr(c, "to_dict") else c for c in contacts],
        }

    async def search_contacts(self, device_id: str, query: str) -> Dict[str, Any]:
        """Search cached contacts for a device matching query string."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")
        if query is None:
            query = ""

        is_known = (
            device_id in self._discovered_devices
            or await self.device_store.is_device_paired(device_id)
            or device_id in self._active_connections
            or device_id in self._state_machines
        )
        if not is_known:
            raise DeviceNotFoundError(f"Device '{device_id}' is unknown")

        matches = await self.contacts_plugin.search_contacts(device_id, query)
        return {
            "device_id": device_id,
            "query": query,
            "is_connected": self.is_device_connected(device_id),
            "count": len(matches),
            "matches": [c.to_dict() if hasattr(c, "to_dict") else c for c in matches],
        }

    async def get_contact(self, device_id: str, uid: str) -> Dict[str, Any]:
        """Retrieve an individual contact record by UID from local cache."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")
        if not uid or not isinstance(uid, str):
            raise ValueError("Parameter 'uid' is required and must be a string")

        is_known = (
            device_id in self._discovered_devices
            or await self.device_store.is_device_paired(device_id)
            or device_id in self._active_connections
            or device_id in self._state_machines
        )
        if not is_known:
            raise DeviceNotFoundError(f"Device '{device_id}' is unknown")

        contact = await self.contacts_plugin.get_contact(device_id, uid)
        if not contact:
            raise ContactNotFoundError(f"Contact UID '{uid}' not found for device '{device_id}'")

        return {
            "device_id": device_id,
            "contact": contact.to_dict() if hasattr(contact, "to_dict") else contact,
        }

    # =====================================================================
    # MPRIS Media Control Methods
    # =====================================================================

    def _send_mpris_packet(self, arg1: Any, arg2: Optional[Any] = None) -> None:
        """Transmit MPRIS packet over active TCP connection."""
        if isinstance(arg1, Packet):
            packet = arg1
            device_id = arg2
        else:
            device_id = arg1
            packet = arg2
        if not device_id or not packet:
            return
        self.send_packet(device_id, packet)

    async def mpris_list_players(self, device_id: str) -> Dict[str, Any]:
        """Enumerate media players reported by remote device."""
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

        is_conn = self.is_device_connected(device_id)
        players = self.mpris_remote.get_players(device_id)
        active = self.mpris_remote.get_active_player(device_id)

        if is_conn and not players:
            try:
                await self.mpris_remote.request_player_list(device_id)
            except Exception:
                pass

        return {
            "device_id": device_id,
            "is_connected": is_conn,
            "players": players,
            "active_player": active,
        }

    async def mpris_get_status(self, device_id: str, player: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve media player playback status and track metadata."""
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

        is_conn = self.is_device_connected(device_id)
        st = self.mpris_remote.get_player_state(device_id, player)
        if player and not st:
            raise MprisPlayerNotFoundError(f"Player '{player}' not found on device '{device_id}'")

        target_player = player or (st.player if st else (self.mpris_remote.get_active_player(device_id) or ""))
        status_dict = st.to_dict() if st else None

        return {
            "device_id": device_id,
            "is_connected": is_conn,
            "player": target_player,
            "status": status_dict,
        }

    async def mpris_send_action(
        self,
        device_id: str,
        action: str,
        player: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send transport control action (Play, Pause, PlayPause, Stop, Next, Previous)."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")
        if not action or not isinstance(action, str):
            raise ValueError("Parameter 'action' is required and must be a string")

        if not self.is_device_connected(device_id):
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")

        pkt = await self.mpris_remote.send_action(device_id, action, player=player)
        return {
            "device_id": device_id,
            "player": pkt.body.get("player"),
            "action": action,
            "success": True,
        }

    async def mpris_set_volume(
        self,
        device_id: str,
        volume: int,
        player: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set volume percentage (0..100) on remote player."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")
        if not isinstance(volume, (int, float)):
            raise ValueError("Parameter 'volume' must be a numeric integer")

        if not self.is_device_connected(device_id):
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")

        clamped = max(0, min(100, int(volume)))
        pkt = await self.mpris_remote.set_volume(device_id, clamped, player=player)
        return {
            "device_id": device_id,
            "player": pkt.body.get("player"),
            "volume": clamped,
            "success": True,
        }

    async def mpris_seek(
        self,
        device_id: str,
        offset_ms: int,
        player: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Seek relative track position in milliseconds."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")
        if not isinstance(offset_ms, (int, float)):
            raise ValueError("Parameter 'offset_ms' must be an integer")

        if not self.is_device_connected(device_id):
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")

        pkt = await self.mpris_remote.seek(device_id, int(offset_ms), player=player)
        return {
            "device_id": device_id,
            "player": pkt.body.get("player"),
            "offset_ms": int(offset_ms),
            "success": True,
        }

    async def mpris_set_position(
        self,
        device_id: str,
        position_ms: int,
        player: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set absolute track position in milliseconds."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")
        if not isinstance(position_ms, (int, float)) or int(position_ms) < 0:
            raise ValueError("Parameter 'position_ms' must be a non-negative integer")

        if not self.is_device_connected(device_id):
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")

        pkt = await self.mpris_remote.set_position(device_id, int(position_ms), player=player)
        return {
            "device_id": device_id,
            "player": pkt.body.get("player"),
            "position_ms": int(position_ms),
            "success": True,
        }

    async def mpris_request_player_list(self, device_id: str) -> Dict[str, Any]:
        """Request fresh media player list from remote device."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")

        if not self.is_device_connected(device_id):
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")

        await self.mpris_remote.request_player_list(device_id)
        return {
            "device_id": device_id,
            "success": True,
            "message": "Player list request dispatched",
        }

    # =====================================================================
    # Remote Commands Feature Actions (Milestone 15)
    # =====================================================================

    def _send_remote_commands_packet(self, arg1: Any, arg2: Optional[Any] = None) -> None:
        """Transmit RemoteCommands packet over active TCP connection."""
        if isinstance(arg1, Packet):
            packet = arg1
            device_id = arg2
        else:
            device_id = arg1
            packet = arg2
        if not device_id or not packet:
            return
        self.send_packet(device_id, packet)

    async def list_remote_commands(self, device_id: str) -> List[Dict[str, Any]]:
        """List cached remote commands for a device.

        If device is connected and commands have not been cached yet, triggers an
        asynchronous request to fetch them in the background.
        """
        device_id = device_id.strip() if isinstance(device_id, str) else ""
        if not device_id:
            raise ValueError("Parameter 'device_id' must not be empty")

        is_known = (
            device_id in self._discovered_devices
            or await self.device_store.is_device_paired(device_id)
            or device_id in self._active_connections
            or device_id in self._state_machines
        )
        if not is_known:
            raise DeviceNotFoundError(f"Device '{device_id}' not found")

        cmds = self.remote_commands_plugin.get_commands(device_id)
        if not cmds and self.is_device_connected(device_id):
            try:
                await self.remote_commands_plugin.request_command_list(device_id)
            except Exception as exc:
                logger.debug("Automatic command list request failed for %s: %s", device_id, exc)

        return [cmd.to_dict() for cmd in cmds]

    async def request_remote_commands(self, device_id: str) -> Dict[str, Any]:
        """Request remote peer to re-advertise its available command list."""
        device_id = device_id.strip() if isinstance(device_id, str) else ""
        if not device_id:
            raise ValueError("Parameter 'device_id' must not be empty")

        if not self.is_device_connected(device_id):
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")

        await self.remote_commands_plugin.request_command_list(device_id)
        return {
            "device_id": device_id,
            "status": "requested",
            "message": f"Command list request transmitted to device '{device_id}'",
        }

    async def execute_remote_command(self, device_id: str, key: str) -> Dict[str, Any]:
        """Transmit remote command execution request to device.

        Verifies that the command key exists if commands are cached.
        """
        device_id = device_id.strip() if isinstance(device_id, str) else ""
        if not device_id:
            raise ValueError("Parameter 'device_id' must not be empty")

        key = key.strip() if isinstance(key, str) else ""
        if not key:
            raise ValueError("Parameter 'key' must not be empty")

        if not self.is_device_connected(device_id):
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")

        # If commands are cached for this device, validate key existence
        cached_cmd = self.remote_commands_plugin.get_command(device_id, key)
        all_cached = self.remote_commands_plugin.get_commands(device_id)
        if all_cached and cached_cmd is None:
            raise RemoteCommandNotFoundError(f"Command '{key}' not found on device '{device_id}'")

        await self.remote_commands_plugin.execute_command(device_id, key)
        return {
            "device_id": device_id,
            "key": key,
            "status": "transmitted",
            "message": f"Execution request for command '{key}' transmitted to device '{device_id}'",
        }

    # =====================================================================
    # SystemVolume Feature Actions (Milestones 22, 24, 25)
    # =====================================================================

    def _send_system_volume_packet(self, arg1: Any, arg2: Optional[Any] = None) -> None:
        """Transmit SystemVolume packet over active TCP connection."""
        if isinstance(arg1, Packet):
            packet = arg1
            device_id = arg2
        else:
            device_id = arg1
            packet = arg2
        if not device_id or not packet:
            return
        self.send_packet(device_id, packet)

    def _send_remote_system_volume_packet(self, arg1: Any, arg2: Optional[Any] = None) -> None:
        """Transmit RemoteSystemVolume packet over active TCP connection."""
        if isinstance(arg1, Packet):
            packet = arg1
            device_id = arg2
        else:
            device_id = arg1
            packet = arg2
        if not device_id or not packet:
            return
        self.send_packet(device_id, packet)

    async def systemvolume_get_sinks(self) -> List[Dict[str, Any]]:
        """List local audio output sinks."""
        sinks = await self.audio_service.list_sinks()
        return [s.to_dict() for s in sinks]

    async def systemvolume_set_volume(
        self, name: str, volume: int, is_percent: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Set volume on a local audio sink."""
        if not name or not isinstance(name, str):
            raise ValueError("Parameter 'name' is required and must be a string")
        pct_flag = is_percent if is_percent is not None else (0 <= volume <= 100)
        success = await self.system_volume_plugin.set_volume(
            name=name, volume=int(volume), is_percent=pct_flag
        )
        return {"success": bool(success)}

    async def systemvolume_set_muted(self, name: str, muted: bool) -> Dict[str, Any]:
        """Set mute state on a local audio sink."""
        if not name or not isinstance(name, str):
            raise ValueError("Parameter 'name' is required and must be a string")
        success = await self.system_volume_plugin.set_muted(name=name, muted=bool(muted))
        return {"success": bool(success)}

    async def systemvolume_set_default(self, name: str) -> Dict[str, Any]:
        """Set default local audio sink."""
        if not name or not isinstance(name, str):
            raise ValueError("Parameter 'name' is required and must be a string")
        success = await self.system_volume_plugin.set_default(name=name)
        return {"success": bool(success)}

    async def systemvolume_get_remote_sinks(self, device_id: str) -> List[Dict[str, Any]]:
        """List cached remote audio sinks for a device."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")
        sinks = self.remote_system_volume_plugin.get_sinks(device_id)
        return [s.to_dict() for s in sinks]

    async def systemvolume_set_remote_volume(
        self, device_id: str, name: str, volume: int
    ) -> Dict[str, Any]:
        """Set volume on a remote device audio sink."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")
        if not name or not isinstance(name, str):
            raise ValueError("Parameter 'name' is required and must be a string")
        if not self.is_device_connected(device_id):
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")
        await self.remote_system_volume_plugin.set_remote_volume(
            device_id=device_id, name=name, volume=int(volume)
        )
        return {"success": True}

    async def systemvolume_set_remote_muted(
        self, device_id: str, name: str, muted: bool
    ) -> Dict[str, Any]:
        """Set mute state on a remote device audio sink."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")
        if not name or not isinstance(name, str):
            raise ValueError("Parameter 'name' is required and must be a string")
        if not self.is_device_connected(device_id):
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")
        await self.remote_system_volume_plugin.set_remote_muted(
            device_id=device_id, name=name, muted=bool(muted)
        )
        return {"success": True}

    async def systemvolume_set_remote_default(
        self, device_id: str, name: str
    ) -> Dict[str, Any]:
        """Set default audio sink on a remote device."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")
        if not name or not isinstance(name, str):
            raise ValueError("Parameter 'name' is required and must be a string")
        if not self.is_device_connected(device_id):
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")
        await self.remote_system_volume_plugin.set_remote_default(
            device_id=device_id, name=name
        )
        return {"success": True}

    async def systemvolume_request_remote_sinks(self, device_id: str) -> Dict[str, Any]:
        """Request remote peer to re-advertise its complete audio sink list."""
        if not device_id or not isinstance(device_id, str):
            raise ValueError("Parameter 'device_id' is required and must be a string")
        if not self.is_device_connected(device_id):
            raise ConnectionClosedError(f"Device '{device_id}' is not connected")
        await self.remote_system_volume_plugin.request_remote_sinks(device_id=device_id)
        return {"success": True}

    # =====================================================================
    # RunCommand Feature Actions (Milestones 23, 25)
    # =====================================================================

    def _send_run_command_packet(self, arg1: Any, arg2: Optional[Any] = None) -> None:
        """Transmit RunCommand packet over active TCP connection."""
        if isinstance(arg1, Packet):
            packet = arg1
            device_id = arg2
        else:
            device_id = arg1
            packet = arg2
        if not device_id or not packet:
            return
        self.send_packet(device_id, packet)

    async def runcommand_list(self) -> List[Dict[str, Any]]:
        """List configured local commands."""
        cmds = self.command_store.load_commands()
        return [c.to_dict() for c in cmds.values()]

    async def runcommand_get(self, key: str) -> Dict[str, Any]:
        """Retrieve configured local command by key."""
        if not key or not isinstance(key, str):
            raise ValueError("Parameter 'key' is required and must be a string")
        cmd = self.command_store.get_command(key)
        if cmd is None:
            raise KeyError(f"Command '{key}' not found in store")
        return cmd.to_dict()

    async def runcommand_add(self, name: str, command: str) -> Dict[str, Any]:
        """Add a local command entry."""
        if not name or not isinstance(name, str):
            raise ValueError("Parameter 'name' is required and must be a string")
        if not command or not isinstance(command, str):
            raise ValueError("Parameter 'command' is required and must be a string")
        entry = self.command_store.add_command(name=name, command=command)
        if hasattr(self, "run_command_plugin"):
            try:
                await self.run_command_plugin.broadcast_command_list()
            except Exception:
                pass
        return entry.to_dict()

    async def runcommand_remove(self, key: str) -> Dict[str, Any]:
        """Remove a local command entry by key."""
        if not key or not isinstance(key, str):
            raise ValueError("Parameter 'key' is required and must be a string")
        removed = self.command_store.remove_command(key)
        if removed and hasattr(self, "run_command_plugin"):
            try:
                await self.run_command_plugin.broadcast_command_list()
            except Exception:
                pass
        return {"success": bool(removed)}

    async def runcommand_exec(self, key: str) -> Dict[str, Any]:
        """Execute a local command by key."""
        if not key or not isinstance(key, str):
            raise ValueError("Parameter 'key' is required and must be a string")
        res = await self.command_service.execute_command(key)
        return {
            "success": res.success,
            "id": res.process_id,
            "key": res.key,
            "exit_code": res.exit_code,
            "output": "".join(res.stdout),
            "stdout": res.stdout,
            "stderr": res.stderr,
        }

    async def runcommand_stop(self, id: Optional[str] = None) -> Dict[str, Any]:
        """Stop local command execution by process id or all processes."""
        if id:
            stopped = await self.command_service.stop_process(id)
            return {"success": bool(stopped)}
        await self.command_service.stop_all()
        return {"success": True}

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

        elif method in ("pair_accept", "accept_pairing"):
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required")
            return await self.accept_pairing(did)

        elif method in ("pair_reject", "reject_pairing"):
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required")
            return await self.reject_pairing(did)

        elif method == "get_pending_pairings":
            return self.get_pending_pairings()

        elif method in ("connect_device", "connect"):
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

        elif method == "contacts_sync":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            force = bool(params.get("force", False))
            timeout_param = params.get("timeout")
            timeout_val = float(timeout_param) if timeout_param is not None else None
            return await self.sync_contacts(did, force=force, timeout=timeout_val)

        elif method == "contacts_list":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            raw_limit = params.get("limit")
            limit = int(raw_limit) if raw_limit is not None else None
            offset = int(params.get("offset", 0))
            return await self.list_contacts(did, limit=limit, offset=offset)

        elif method == "contacts_search":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            query = str(params.get("query", ""))
            return await self.search_contacts(did, query)

        elif method == "contacts_get":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            uid = str(params.get("uid", ""))
            if not uid:
                raise ValueError("Parameter 'uid' is required and must be a string")
            return await self.get_contact(did, uid)

        elif method == "mpris_list_players":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            return await self.mpris_list_players(did)

        elif method == "mpris_get_status":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            player = params.get("player")
            return await self.mpris_get_status(did, player=player)

        elif method == "mpris_send_action":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            action = params.get("action")
            if not action or not isinstance(action, str):
                raise ValueError("Parameter 'action' is required and must be a string")
            player = params.get("player")
            return await self.mpris_send_action(did, action, player=player)

        elif method == "mpris_set_volume":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            if "volume" not in params:
                raise ValueError("Parameter 'volume' is required")
            volume = int(params["volume"])
            player = params.get("player")
            return await self.mpris_set_volume(did, volume, player=player)

        elif method == "mpris_seek":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            if "offset_ms" not in params:
                raise ValueError("Parameter 'offset_ms' is required")
            offset_ms = int(params["offset_ms"])
            player = params.get("player")
            return await self.mpris_seek(did, offset_ms, player=player)

        elif method == "mpris_set_position":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            if "position_ms" not in params:
                raise ValueError("Parameter 'position_ms' is required")
            position_ms = int(params["position_ms"])
            player = params.get("player")
            return await self.mpris_set_position(did, position_ms, player=player)

        elif method == "mpris_request_player_list":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            return await self.mpris_request_player_list(did)

        elif method in ("list_remote_commands", "remotecommands_list"):
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            return await self.list_remote_commands(did)

        elif method in ("request_remote_commands", "remotecommands_request"):
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            return await self.request_remote_commands(did)

        elif method in ("execute_remote_command", "remotecommands_run", "remotecommands_exec"):
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            key = params.get("key")
            if not key or not isinstance(key, str):
                raise ValueError("Parameter 'key' is required and must be a string")
            return await self.execute_remote_command(did, key)

        # =================================================================
        # SystemVolume Local and Remote Routes (Milestone 25)
        # =================================================================
        elif method == "systemvolume_get_sinks":
            return await self.systemvolume_get_sinks()

        elif method == "systemvolume_set_volume":
            if "name" not in params:
                raise ValueError("Parameter 'name' is required")
            if "volume" not in params:
                raise ValueError("Parameter 'volume' is required")
            name = str(params["name"])
            volume = int(params["volume"])
            return await self.systemvolume_set_volume(name=name, volume=volume)

        elif method == "systemvolume_set_muted":
            if "name" not in params:
                raise ValueError("Parameter 'name' is required")
            if "muted" not in params:
                raise ValueError("Parameter 'muted' is required")
            name = str(params["name"])
            muted = bool(params["muted"])
            return await self.systemvolume_set_muted(name=name, muted=muted)

        elif method == "systemvolume_set_default":
            if "name" not in params:
                raise ValueError("Parameter 'name' is required")
            name = str(params["name"])
            return await self.systemvolume_set_default(name=name)

        elif method in ("systemvolume_get_remote_sinks", "list_remote_audio_sinks"):
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            return await self.systemvolume_get_remote_sinks(did)

        elif method in ("systemvolume_set_remote_volume", "set_remote_audio_volume"):
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            if "name" not in params:
                raise ValueError("Parameter 'name' is required")
            if "volume" not in params:
                raise ValueError("Parameter 'volume' is required")
            name = str(params["name"])
            volume = int(params["volume"])
            return await self.systemvolume_set_remote_volume(did, name, volume)

        elif method in ("systemvolume_set_remote_muted", "set_remote_audio_mute"):
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            if "name" not in params:
                raise ValueError("Parameter 'name' is required")
            if "muted" not in params:
                raise ValueError("Parameter 'muted' is required")
            name = str(params["name"])
            muted = bool(params["muted"])
            return await self.systemvolume_set_remote_muted(did, name, muted)

        elif method == "systemvolume_set_remote_default":
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            if "name" not in params:
                raise ValueError("Parameter 'name' is required")
            name = str(params["name"])
            return await self.systemvolume_set_remote_default(did, name)

        elif method in ("systemvolume_request_remote_sinks", "request_remote_audio_sinks"):
            did = params.get("device_id")
            if not did or not isinstance(did, str):
                raise ValueError("Parameter 'device_id' is required and must be a string")
            return await self.systemvolume_request_remote_sinks(did)

        # =================================================================
        # RunCommand Local Routes (Milestone 25)
        # =================================================================
        elif method in ("runcommand_list", "list_local_commands"):
            return await self.runcommand_list()

        elif method == "runcommand_get":
            key = params.get("key")
            if not key or not isinstance(key, str):
                raise ValueError("Parameter 'key' is required and must be a string")
            return await self.runcommand_get(key)

        elif method == "runcommand_add":
            name = params.get("name")
            if not name or not isinstance(name, str):
                raise ValueError("Parameter 'name' is required and must be a string")
            command = params.get("command")
            if not command or not isinstance(command, str):
                raise ValueError("Parameter 'command' is required and must be a string")
            return await self.runcommand_add(name, command)

        elif method == "runcommand_remove":
            key = params.get("key")
            if not key or not isinstance(key, str):
                raise ValueError("Parameter 'key' is required and must be a string")
            return await self.runcommand_remove(key)

        elif method in ("runcommand_exec", "execute_local_command"):
            key = params.get("key")
            if not key or not isinstance(key, str):
                raise ValueError("Parameter 'key' is required and must be a string")
            return await self.runcommand_exec(key)

        elif method == "runcommand_stop":
            pid = params.get("id")
            return await self.runcommand_stop(pid)

        elif method in ("stop_daemon", "stop"):

            self._spawn_background_task(self.stop(), name="ipc_stop_daemon")
            return {
                "success": True,
                "message": "PenguLink daemon shutdown initiated",
            }

        else:
            raise KeyError(f"Unknown JSON-RPC method '{method}'")


Service = PenguLinkDaemon

__all__ = [
    "format_uptime",
    "LogRingBuffer",
    "PenguLinkDaemon",
    "Service",
    "safe_redact_clipboard",
    "ClipboardResult",
]


if __name__ == "__main__":
    import sys
    from pengulink.cli.main import cli

    # Translate direct module invocation into standard CLI daemon run
    args = [arg for arg in sys.argv[1:] if arg != "--foreground"]
    cli(["daemon", "run"] + args)


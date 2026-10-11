"""pengulink.plugins.share.payload
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Out-of-band TLS payload streaming engine for KDE Connect Share plugin.
Manages ephemeral TCP servers on ports 1739-1764, outbound TLS client
connections, raw byte streaming, progress reporting, and cleanup.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import logging
import os
from pathlib import Path
import ssl
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union

from pengulink.core.exceptions import (
    ConnectionClosedError,
    ConnectionTimeoutError,
    IncompleteTransferError,
    PathTraversalError,
    PayloadPortBindingError,
    PayloadSizeExceededError,
    PayloadTimeoutError,
    PayloadTransferError,
)
from pengulink.crypto.tls import create_paired_ssl_context
from pengulink.network.packet import Packet

logger = logging.getLogger("pengulink.plugins.share.payload")

# Protocol Constants
PAYLOAD_PORT_MIN = 1739
PAYLOAD_PORT_MAX = 1764
CHUNK_SIZE = 65536  # 64 KB streaming chunks
DEFAULT_ACCEPT_TIMEOUT = 15.0
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_SOCKET_TIMEOUT = 15.0
PROGRESS_EMIT_INTERVAL = 0.1  # 100 ms throttle


def _to_bytes(cert_or_key: Union[str, bytes]) -> bytes:
    if isinstance(cert_or_key, str):
        return cert_or_key.encode("utf-8")
    return cert_or_key


@dataclass
class TransferProgress:
    """Snapshot of file transfer progress."""

    device_id: str = ""
    filename: str = ""
    direction: str = "outgoing"  # "outgoing" or "incoming"
    bytes_transferred: int = 0
    total_bytes: int = 0
    speed_bps: float = 0.0
    percentage: float = 0.0
    file_index: int = 1
    total_files: int = 1
    is_complete: bool = False


@dataclass
class BatchMetadata:
    """Metadata for sequential multi-file batch transfers."""

    number_of_files: int
    total_payload_size: int
    current_index: int = 1
    completed_bytes: int = 0


class PayloadServer:
    """Ephemeral TCP server for out-of-band TLS file uploads (Linux -> Android).

    Roles:
      TCP: Listener (binds 1739..1764)
      TLS: Server (server_side=True) with pinned peer certificate
    """

    def __init__(
        self,
        local_cert_pem: Union[str, bytes],
        local_key_pem: Union[str, bytes],
        peer_cert_pem: Union[str, bytes],
        port_min: int = PAYLOAD_PORT_MIN,
        port_max: int = PAYLOAD_PORT_MAX,
        allow_ephemeral_fallback: bool = True,
    ) -> None:
        self.local_cert_pem = _to_bytes(local_cert_pem)
        self.local_key_pem = _to_bytes(local_key_pem)
        self.peer_cert_pem = _to_bytes(peer_cert_pem)
        self.port_min = port_min
        self.port_max = port_max
        self.allow_ephemeral_fallback = allow_ephemeral_fallback

        self.server: Optional[asyncio.Server] = None
        self.bound_port: Optional[int] = None
        self._connected_event = asyncio.Event()
        self._client_connection: Optional[Tuple[asyncio.StreamReader, asyncio.StreamWriter]] = None
        self._lock = asyncio.Lock()

    async def bind_server(self, host: str = "0.0.0.0") -> int:
        """Bind TCP server on available port in 1739..1764 with TLS server context."""
        ssl_ctx = create_paired_ssl_context(
            local_cert_pem=self.local_cert_pem,
            local_key_pem=self.local_key_pem,
            peer_cert_pem=self.peer_cert_pem,
            server_side=True,
        )

        for port in range(self.port_min, self.port_max + 1):
            try:
                self.server = await asyncio.start_server(
                    self._on_client_connected,
                    host=host,
                    port=port,
                    ssl=ssl_ctx,
                    reuse_address=True,
                )
                self.bound_port = port
                logger.debug("Payload server bound on port %d", port)
                return port
            except OSError:
                continue

        if self.allow_ephemeral_fallback:
            try:
                self.server = await asyncio.start_server(
                    self._on_client_connected,
                    host=host,
                    port=0,
                    ssl=ssl_ctx,
                    reuse_address=True,
                )
                self.bound_port = self.server.sockets[0].getsockname()[1]
                logger.info("Payload server bound on fallback ephemeral port %d", self.bound_port)
                return self.bound_port
            except OSError as exc:
                raise PayloadPortBindingError(f"Failed binding fallback ephemeral port: {exc}") from exc

        raise PayloadPortBindingError(
            f"Could not bind TCP port in range {self.port_min}..{self.port_max}",
            context={"port_range": f"{self.port_min}-{self.port_max}"},
        )

    def _on_client_connected(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._client_connection is None:
            self._client_connection = (reader, writer)
            self._connected_event.set()
        else:
            # Reject secondary concurrent connections to single-use payload port
            try:
                writer.close()
            except Exception:
                pass

    async def wait_for_peer_and_stream(
        self,
        source_path: Path,
        expected_size: Optional[int] = None,
        target_device_id: str = "",
        progress_callback: Optional[Callable[..., Any]] = None,
        batch_metadata: Optional[BatchMetadata] = None,
        accept_timeout: float = DEFAULT_ACCEPT_TIMEOUT,
        socket_timeout: float = DEFAULT_SOCKET_TIMEOUT,
    ) -> int:
        """Wait for peer to connect, stream file bytes with progress tracking, and clean up."""
        try:
            # 1. Await incoming peer connection
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=accept_timeout)
            except (asyncio.TimeoutError, TimeoutError) as exc:
                raise PayloadTimeoutError(
                    f"Peer did not connect to payload port {self.bound_port} within {accept_timeout}s"
                ) from exc

            # Stop accepting new clients on single-shot server
            if self.server:
                self.server.close()

            if self._client_connection is None:
                raise ConnectionClosedError("No client connection available")

            reader, writer = self._client_connection
            file_size = expected_size if expected_size is not None else source_path.stat().st_size
            bytes_sent = 0
            start_time = time.time()
            last_progress = start_time

            with open(source_path, "rb") as f:
                while bytes_sent < file_size:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    writer.write(chunk)
                    try:
                        await asyncio.wait_for(writer.drain(), timeout=socket_timeout)
                    except (asyncio.TimeoutError, TimeoutError) as exc:
                        raise PayloadTimeoutError(f"Socket write drain timed out during upload: {exc}") from exc
                    except (ConnectionResetError, BrokenPipeError, OSError) as exc:
                        raise ConnectionClosedError(f"Socket dropped by peer during upload: {exc}") from exc

                    bytes_sent += len(chunk)
                    now = time.time()
                    if progress_callback and (now - last_progress >= PROGRESS_EMIT_INTERVAL or bytes_sent == file_size):
                        elapsed = max(now - start_time, 0.001)
                        pct = (bytes_sent / file_size * 100.0) if file_size > 0 else 100.0
                        res = progress_callback(bytes_sent, file_size, pct)
                        if inspect.isawaitable(res):
                            await res
                        last_progress = now

            if bytes_sent < file_size:
                raise IncompleteTransferError(f"Premature EOF: sent {bytes_sent} of {file_size} bytes")

            return bytes_sent
        finally:
            await self.close()

    # Alias for explorer 2 design
    send_file = wait_for_peer_and_stream

    async def close(self) -> None:
        """Cleanly close server and client socket."""
        if self._client_connection:
            _, writer = self._client_connection
            try:
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), timeout=2.0)
            except Exception:
                pass
            self._client_connection = None

        if self.server:
            try:
                self.server.close()
                await asyncio.wait_for(self.server.wait_closed(), timeout=2.0)
            except Exception:
                pass
            self.server = None


class PayloadClient:
    """Outbound TCP client for out-of-band TLS file downloads (Android -> Linux).

    Roles:
      TCP: Client (connects to peer IP:port)
      TLS: Client (server_side=False) with pinned peer certificate
    """

    def __init__(
        self,
        local_cert_pem: Union[str, bytes],
        local_key_pem: Union[str, bytes],
        peer_cert_pem: Union[str, bytes],
    ) -> None:
        self.local_cert_pem = _to_bytes(local_cert_pem)
        self.local_key_pem = _to_bytes(local_key_pem)
        self.peer_cert_pem = _to_bytes(peer_cert_pem)

    async def download_to_staging(
        self,
        target_host: str,
        target_port: int,
        expected_size: int,
        staging_writer: Any,
        target_device_id: str = "",
        filename: str = "",
        progress_callback: Optional[Callable[..., Any]] = None,
        batch_metadata: Optional[BatchMetadata] = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        socket_timeout: float = DEFAULT_SOCKET_TIMEOUT,
    ) -> Path:
        """Connect to peer payload server, stream bytes into staging writer, and finalize."""
        # 1. Zero-byte payload optimization
        if expected_size == 0:
            staging_writer.open()
            return staging_writer.commit()

        ssl_ctx = create_paired_ssl_context(
            local_cert_pem=self.local_cert_pem,
            local_key_pem=self.local_key_pem,
            peer_cert_pem=self.peer_cert_pem,
            server_side=False,
        )

        writer: Optional[asyncio.StreamWriter] = None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(target_host, target_port, ssl=ssl_ctx),
                timeout=connect_timeout,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            staging_writer.abort()
            raise PayloadTimeoutError(f"Connection to payload port {target_host}:{target_port} timed out") from exc
        except Exception as exc:
            staging_writer.abort()
            raise ConnectionClosedError(f"Could not connect to payload port {target_host}:{target_port}: {exc}") from exc

        bytes_received = 0
        start_time = time.time()
        last_progress = start_time

        try:
            staging_writer.open()
            while bytes_received < expected_size:
                to_read = min(CHUNK_SIZE, expected_size - bytes_received + 1)
                try:
                    chunk = await asyncio.wait_for(reader.read(to_read), timeout=socket_timeout)
                except (asyncio.TimeoutError, TimeoutError) as exc:
                    raise PayloadTimeoutError(f"Payload read timed out from {target_host}:{target_port}") from exc
                except (ConnectionResetError, BrokenPipeError, OSError) as exc:
                    raise ConnectionClosedError(f"Socket dropped by peer during download: {exc}") from exc

                if not chunk:
                    break

                bytes_received += len(chunk)
                if bytes_received > expected_size:
                    raise PayloadSizeExceededError(
                        f"Received {bytes_received} bytes exceeding promised payloadSize {expected_size}"
                    )

                if hasattr(staging_writer, "write_chunk") and inspect.iscoroutinefunction(staging_writer.write_chunk):
                    await staging_writer.write_chunk(chunk)
                else:
                    staging_writer.write(chunk)

                now = time.time()
                if progress_callback and (now - last_progress >= PROGRESS_EMIT_INTERVAL or bytes_received == expected_size):
                    elapsed = max(now - start_time, 0.001)
                    pct = (bytes_received / expected_size * 100.0) if expected_size > 0 else 100.0
                    res = progress_callback(bytes_received, expected_size, pct)
                    if inspect.isawaitable(res):
                        await res
                    last_progress = now

            if bytes_received < expected_size:
                raise IncompleteTransferError(
                    f"Truncated transfer: received {bytes_received} of {expected_size} bytes"
                )

            final_path = staging_writer.commit()
            return final_path
        except (Exception, asyncio.CancelledError):
            staging_writer.abort()
            raise
        finally:
            if writer is not None:
                try:
                    writer.close()
                    await asyncio.wait_for(writer.wait_closed(), timeout=2.0)
                except Exception:
                    pass


class PayloadTransferManager:
    """Central orchestrator managing active file transfers and batch sessions."""

    def __init__(self, event_bus: Optional[Any] = None) -> None:
        self.event_bus = event_bus
        self._active_tasks: Set[asyncio.Task[Any]] = set()
        self._device_tasks: Dict[str, Set[asyncio.Task[Any]]] = {}
        self._active_batches: Dict[str, BatchMetadata] = {}

    def track_task(self, device_id: str, task: asyncio.Task[Any]) -> None:
        self._active_tasks.add(task)
        if device_id not in self._device_tasks:
            self._device_tasks[device_id] = set()
        self._device_tasks[device_id].add(task)

        def _cleanup(t: asyncio.Task[Any]) -> None:
            self._active_tasks.discard(t)
            if device_id in self._device_tasks:
                self._device_tasks[device_id].discard(t)
                if not self._device_tasks[device_id]:
                    del self._device_tasks[device_id]

        task.add_done_callback(_cleanup)

    async def cancel_device_transfers(self, device_id: str) -> None:
        """Cancel all in-flight transfers for a disconnected or unpaired device."""
        tasks = list(self._device_tasks.get(device_id, []))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._device_tasks.pop(device_id, None)
        self._active_batches.pop(device_id, None)

    async def stop_all(self) -> None:
        """Cancel and drain all in-flight transfers upon daemon / plugin shutdown."""
        tasks = list(self._active_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self._active_tasks.clear()
        self._device_tasks.clear()
        self._active_batches.clear()


__all__ = [
    "PAYLOAD_PORT_MIN",
    "PAYLOAD_PORT_MAX",
    "CHUNK_SIZE",
    "DEFAULT_ACCEPT_TIMEOUT",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_SOCKET_TIMEOUT",
    "TransferProgress",
    "BatchMetadata",
    "PayloadServer",
    "PayloadClient",
    "PayloadTransferManager",
]

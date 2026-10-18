"""pengulink.plugins.share.plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect Share plugin for PenguLink daemon.

Handles:
- Inbound: text, URL, and file reception (kdeconnect.share.request, kdeconnect.share.request.update).
- Outbound: text, URL, single file, and multi-file batch transfers.
- Security coordination: destination resolution, path traversal defense, atomic staging.
- Payload coordination: out-of-band TLS streaming on ports 1739..1764.
- Event emission: ShareReceivedEvent, FileTransferProgress, FileReceivedEvent.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union

from pengulink.core.config import Config
from pengulink.core.events import (
    EventBus,
    FileReceivedEvent,
    FileTransferProgress,
    ShareReceivedEvent,
)
from pengulink.core.exceptions import (
    ConnectionClosedError,
    DeviceNotFoundError,
    IncompleteTransferError,
    PathTraversalError,
    PayloadTransferError,
    ShareError,
)
from pengulink.network.packet import Packet
from pengulink.plugins.base import Plugin, SendPacketCallable, safe_send_packet
from pengulink.plugins.share.payload import (
    PayloadClient,
    PayloadServer,
    PayloadTransferManager,
)
from pengulink.plugins.share.security import (
    AtomicStagingWriter,
    cleanup_orphaned_part_files,
    resolve_collision,
    resolve_download_dir,
    sanitize_filename,
    validate_containment,
)

logger = logging.getLogger("pengulink.plugins.share")

SHARE_REQUEST_TYPE = "kdeconnect.share.request"
SHARE_UPDATE_TYPE = "kdeconnect.share.request.update"
SHARE_LEGACY_TYPE = "kdeconnect.share"


def create_share_text_packet(text: str) -> Packet:
    """Construct a text share packet."""
    return Packet(
        type=SHARE_REQUEST_TYPE,
        body={"text": text},
    )


def create_share_url_packet(url: str) -> Packet:
    """Construct a URL share packet."""
    return Packet(
        type=SHARE_REQUEST_TYPE,
        body={"url": url},
    )


def create_share_file_packet(
    filename: str,
    payload_size: int,
    payload_port: int,
    last_modified: Optional[int] = None,
    creation_time: Optional[int] = None,
    open_flag: bool = False,
    number_of_files: int = 1,
    total_payload_size: Optional[int] = None,
) -> Packet:
    """Construct an outbound file share control packet."""
    now_ms = int(time.time() * 1000)
    body: Dict[str, Any] = {
        "filename": filename,
        "lastModified": last_modified if last_modified is not None else now_ms,
        "creationTime": creation_time if creation_time is not None else now_ms,
        "open": bool(open_flag),
        "numberOfFiles": number_of_files,
        "totalPayloadSize": total_payload_size if total_payload_size is not None else payload_size,
    }
    return Packet(
        type=SHARE_REQUEST_TYPE,
        body=body,
        payloadSize=payload_size,
        payloadTransferInfo={"port": payload_port},
    )


class SharePlugin(Plugin):
    """Core Share plugin implementing bidirectional KDE Connect sharing."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        packet_sender: Optional[SendPacketCallable] = None,
        is_paired_checker: Optional[Callable[[str], Union[bool, Coroutine[Any, Any, bool]]]] = None,
        peer_resolver: Optional[Callable[[str], Optional[Tuple[str, str]]]] = None,
        config: Optional[Config] = None,
        local_cert_pem: Optional[Union[str, bytes]] = None,
        local_key_pem: Optional[Union[str, bytes]] = None,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self.packet_sender = packet_sender
        self.is_paired_checker = is_paired_checker
        self.peer_resolver = peer_resolver
        self.config = config
        self.local_cert_pem = local_cert_pem
        self.local_key_pem = local_key_pem

        self.payload_manager = PayloadTransferManager(event_bus=event_bus)
        self._active_transfers: Dict[str, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()
        self._reserved_destinations: Set[Path] = set()
        self._collision_lock = asyncio.Lock()

    @property
    def plugin_id(self) -> str:
        return "share"

    @property
    def incoming_capabilities(self) -> List[str]:
        return [SHARE_REQUEST_TYPE, SHARE_UPDATE_TYPE, SHARE_LEGACY_TYPE]

    @property
    def outgoing_capabilities(self) -> List[str]:
        return [SHARE_REQUEST_TYPE, SHARE_UPDATE_TYPE, SHARE_LEGACY_TYPE]

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Process incoming share packets."""
        if packet.type not in self.incoming_capabilities:
            return

        if not device_id or not isinstance(device_id, str) or not device_id.strip():
            logger.warning("SharePlugin ignoring packet with empty device_id")
            return

        # Authenticate device pairing if checker provided
        if self.is_paired_checker is not None:
            check_res = self.is_paired_checker(device_id)
            if inspect.isawaitable(check_res):
                check_res = await check_res
            if not check_res:
                logger.warning("SharePlugin rejecting share packet from unauthenticated device %s", device_id)
                return

        body = packet.body if isinstance(packet.body, dict) else {}

        # Branch 1: Inbound File Transfer
        if packet.payloadTransferInfo is not None or "filename" in body or packet.payloadSize is not None:
            await self._handle_inbound_file(packet, device_id)
            return

        # Branch 2: Inbound Text Share
        if "text" in body:
            text_val = str(body.get("text", ""))
            logger.info("Received text share from device %s (%d chars)", device_id, len(text_val))
            if self.event_bus:
                await self.event_bus.publish(
                    ShareReceivedEvent(
                        device_id=device_id,
                        share_type="text",
                        text=text_val,
                    )
                )
            return

        # Branch 3: Inbound URL Share
        if "url" in body:
            url_val = str(body.get("url", ""))
            logger.info("Received URL share from device %s: %s", device_id, url_val)
            if self.event_bus:
                await self.event_bus.publish(
                    ShareReceivedEvent(
                        device_id=device_id,
                        share_type="url",
                        url=url_val,
                    )
                )
            return

        # Branch 4: Dynamic Batch Update
        if packet.type == SHARE_UPDATE_TYPE:
            num_files = body.get("numberOfFiles", 0)
            total_size = body.get("totalPayloadSize", 0)
            logger.info("Received share batch update from %s: %d files, %d bytes", device_id, num_files, total_size)
            return

    async def _handle_inbound_file(self, packet: Packet, device_id: str) -> None:
        """Process incoming file transfer and stream payload."""
        body = packet.body if isinstance(packet.body, dict) else {}
        raw_filename = body.get("filename", "unnamed_share")
        payload_size = packet.payloadSize or 0
        transfer_info = packet.payloadTransferInfo or {}
        port = transfer_info.get("port")

        dest_dir = resolve_download_dir(config=self.config)
        clean_name = sanitize_filename(raw_filename)
        async with self._collision_lock:
            final_path = resolve_collision(dest_dir, clean_name, self._reserved_destinations)
            self._reserved_destinations.add(final_path)

        try:
            validate_containment(dest_dir, final_path)

            # Handle zero-byte file
            if payload_size == 0 or port is None:
                final_path.touch(mode=0o644, exist_ok=True)
                try:
                    os.chmod(final_path, 0o644)
                except OSError:
                    pass
                if self.event_bus:
                    await self.event_bus.publish(
                        FileReceivedEvent(device_id=device_id, file_path=final_path, file_size=0, filename=final_path.name)
                    )
                    await self.event_bus.publish(
                        ShareReceivedEvent(
                            device_id=device_id,
                            share_type="file",
                            filename=final_path.name,
                            file_path=final_path,
                            file_size=0,
                        )
                    )
                return

            # Resolve peer connection info
            peer_info = self.peer_resolver(device_id) if self.peer_resolver else None
            if inspect.isawaitable(peer_info):
                peer_info = await peer_info
            if not peer_info:
                logger.error("Cannot receive file payload: peer info unresolvable for device %s", device_id)
                return
            peer_ip, peer_cert_pem = peer_info

            client = PayloadClient(
                local_cert_pem=self.local_cert_pem or b"",
                local_key_pem=self.local_key_pem or b"",
                peer_cert_pem=peer_cert_pem,
            )

            async def _progress_cb(transferred: int, total: int, pct: float) -> None:
                if self.event_bus:
                    await self.event_bus.publish(
                        FileTransferProgress(
                            device_id=device_id,
                            filename=final_path.name,
                            direction="incoming",
                            bytes_transferred=transferred,
                            total_bytes=total,
                            percentage=pct,
                        )
                    )

            staging_writer = AtomicStagingWriter(
                dest_dir=dest_dir,
                final_path=final_path,
                expected_size=payload_size,
                last_modified=body.get("lastModified"),
            )

            task = asyncio.create_task(
                client.download_to_staging(
                    target_host=peer_ip,
                    target_port=port,
                    expected_size=payload_size,
                    staging_writer=staging_writer,
                    target_device_id=device_id,
                    filename=final_path.name,
                    progress_callback=_progress_cb,
                )
            )
            self.payload_manager.track_task(device_id, task)
            self._active_transfers[final_path.name] = task
            try:
                await task
            finally:
                self._active_transfers.pop(final_path.name, None)

            if self.event_bus:
                await self.event_bus.publish(
                    FileReceivedEvent(
                        device_id=device_id,
                        file_path=final_path,
                        file_size=payload_size,
                        filename=final_path.name,
                    )
                )
                await self.event_bus.publish(
                    ShareReceivedEvent(
                        device_id=device_id,
                        share_type="file",
                        filename=final_path.name,
                        file_path=final_path,
                        file_size=payload_size,
                    )
                )
        finally:
            async with self._collision_lock:
                self._reserved_destinations.discard(final_path)

    async def share_text(
        self,
        device_id: str,
        text: str,
        send_packet: Optional[SendPacketCallable] = None,
    ) -> Dict[str, Any]:
        """Send a text share packet to remote device."""
        sender = send_packet or self.packet_sender
        if not callable(sender):
            raise ConnectionClosedError("send_packet callable is missing or disconnected")

        packet = create_share_text_packet(text)
        await safe_send_packet(sender, packet)
        return {
            "status": "transmitted",
            "device_id": device_id,
            "text": text,
            "packet_id": packet.id,
        }

    async def share_url(
        self,
        device_id: str,
        url: str,
        send_packet: Optional[SendPacketCallable] = None,
    ) -> Dict[str, Any]:
        """Send a URL share packet to remote device."""
        sender = send_packet or self.packet_sender
        if not callable(sender):
            raise ConnectionClosedError("send_packet callable is missing or disconnected")

        packet = create_share_url_packet(url)
        await safe_send_packet(sender, packet)
        return {
            "status": "transmitted",
            "device_id": device_id,
            "url": url,
            "packet_id": packet.id,
        }

    async def share_file(
        self,
        device_id: str,
        file_path: Union[str, Path],
        send_packet: Optional[SendPacketCallable] = None,
        open_flag: bool = False,
        file_index: int = 1,
        total_files: int = 1,
        total_batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send a single file over out-of-band TLS payload connection."""
        source = Path(file_path).resolve()
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Source file does not exist: {source}")

        sender = send_packet or self.packet_sender
        if not callable(sender):
            raise ConnectionClosedError("send_packet callable is missing or disconnected")

        peer_info = self.peer_resolver(device_id) if self.peer_resolver else None
        if inspect.isawaitable(peer_info):
            peer_info = await peer_info
        if not peer_info:
            raise DeviceNotFoundError(f"Peer certificate or address unresolvable for {device_id}")
        peer_ip, peer_cert_pem = peer_info

        file_size = source.stat().st_size

        server = PayloadServer(
            local_cert_pem=self.local_cert_pem or b"",
            local_key_pem=self.local_key_pem or b"",
            peer_cert_pem=peer_cert_pem,
        )
        bound_port = await server.bind_server()

        packet = create_share_file_packet(
            filename=source.name,
            payload_size=file_size,
            payload_port=bound_port,
            open_flag=open_flag,
            number_of_files=total_files,
            total_payload_size=total_batch_size if total_batch_size is not None else file_size,
        )

        async def _progress_cb(transferred: int, total: int, pct: float) -> None:
            if self.event_bus:
                await self.event_bus.publish(
                    FileTransferProgress(
                        device_id=device_id,
                        filename=source.name,
                        direction="outgoing",
                        bytes_transferred=transferred,
                        total_bytes=total,
                        percentage=pct,
                    )
                )

        try:
            # Transmit control packet over primary mTLS connection
            await safe_send_packet(sender, packet)

            # Stream payload bytes to connecting peer
            await server.wait_for_peer_and_stream(
                source_path=source,
                expected_size=file_size,
                target_device_id=device_id,
                progress_callback=_progress_cb,
            )
        finally:
            await server.close()

        return {
            "status": "ok",
            "device_id": device_id,
            "filename": source.name,
            "size": file_size,
            "packet_id": packet.id,
        }

    async def share_files(
        self,
        device_id: str,
        file_paths: List[Union[str, Path]],
        send_packet: Optional[SendPacketCallable] = None,
        open_flag: bool = False,
    ) -> Dict[str, Any]:
        """Sequentially upload multiple files as a batch."""
        total_files = len(file_paths)
        if total_files == 0:
            raise ValueError("No files provided for batch transfer")

        resolved_paths = [Path(p).resolve() for p in file_paths]
        total_batch_size = sum(p.stat().st_size for p in resolved_paths)

        results = []
        for idx, path in enumerate(resolved_paths, start=1):
            res = await self.share_file(
                device_id=device_id,
                file_path=path,
                send_packet=send_packet,
                open_flag=open_flag,
                file_index=idx,
                total_files=total_files,
                total_batch_size=total_batch_size,
            )
            results.append(res)

        return {
            "status": "ok",
            "device_id": device_id,
            "files_transferred": total_files,
            "total_size": total_batch_size,
            "files": [r["filename"] for r in results],
        }

    async def cancel_device_transfers(self, device_id: str) -> None:
        """Cancel all in-flight file transfers for a specific device."""
        await self.payload_manager.cancel_device_transfers(device_id)

    async def stop(self) -> None:
        """Cancel and drain all active transfers on shutdown."""
        await self.payload_manager.stop_all()
        tasks = list(self._active_transfers.values())
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_transfers.clear()


__all__ = [
    "SHARE_REQUEST_TYPE",
    "SHARE_UPDATE_TYPE",
    "SHARE_LEGACY_TYPE",
    "create_share_text_packet",
    "create_share_url_packet",
    "create_share_file_packet",
    "SharePlugin",
]

"""pengulink.network.udp
~~~~~~~~~~~~~~~~~~~~~
KDE Connect UDP Discovery Broadcaster and Listener on port 1716.
Handles multi-homed broadcast dispatch, local packet deduplication,
and remote device discovery extraction.
"""

from __future__ import annotations

import array
import asyncio
import collections
import fcntl
import inspect
import json
import logging
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional, Set, Tuple, Union

from pengulink.core.config import PenguLinkConfig
from pengulink.network.packet import IdentityBody, Packet

logger = logging.getLogger("pengulink.network.udp")


@dataclass(frozen=True)
class DiscoveredDevice:
    """Immutable representation of a discovered remote KDE Connect node."""

    device_id: str
    device_name: str
    device_type: str
    protocol_version: int
    tcp_port: int
    sender_ip: str
    incoming_capabilities: Tuple[str, ...] = field(default_factory=tuple)
    outgoing_capabilities: Tuple[str, ...] = field(default_factory=tuple)
    timestamp: float = field(default_factory=time.time)

    def is_compatible(self) -> bool:
        """Validate protocol version compatibility (versions 7 and 8 supported)."""
        return self.protocol_version in (7, 8)


def get_broadcast_addresses() -> Set[str]:
    """Enumerate all active non-loopback IPv4 network interfaces on Linux

    and return their directed broadcast addresses alongside 255.255.255.255.
    """
    broadcasts: Set[str] = {"255.255.255.255"}

    # Constants for Linux ioctl calls
    SIOCGIFCONF = 0x8912
    SIOCGIFFLAGS = 0x8913
    SIOCGIFBRDADDR = 0x8919
    IFF_UP = 0x1
    IFF_BROADCAST = 0x2
    IFF_LOOPBACK = 0x8

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            arr = array.array("B", b"\0" * 4096)
            ifconf = struct.pack("iP", len(arr), arr.buffer_info()[0])
            res = fcntl.ioctl(s.fileno(), SIOCGIFCONF, ifconf)
            out_len, _ = struct.unpack("iP", res)
            raw = arr.tobytes()[:out_len]

            is_64bit = struct.calcsize("P") == 8
            ifreq_size = 40 if is_64bit else 32
            for i in range(0, out_len, ifreq_size):
                item = raw[i : i + ifreq_size]
                name = item[:16].split(b"\0", 1)[0].decode("ascii", errors="ignore")
                if not name or name == "lo":
                    continue

                ifreq = struct.pack("16s16s", name.encode("ascii"), b"\0" * 16)
                try:
                    flags_res = fcntl.ioctl(s.fileno(), SIOCGIFFLAGS, ifreq)
                    flags = struct.unpack("16sH14s", flags_res)[1]
                    is_up = bool(flags & IFF_UP)
                    is_loopback = bool(flags & IFF_LOOPBACK)
                    supports_broadcast = bool(flags & IFF_BROADCAST)

                    if is_up and not is_loopback and supports_broadcast:
                        brd_res = fcntl.ioctl(s.fileno(), SIOCGIFBRDADDR, ifreq)
                        brd_ip = socket.inet_ntoa(brd_res[20:24])
                        if brd_ip and brd_ip != "0.0.0.0":
                            broadcasts.add(brd_ip)
                except OSError:
                    pass
        finally:
            s.close()
    except Exception as exc:
        logger.debug("Network interface enumeration fallback: %s", exc)

    return broadcasts


class UDPDiscoveryProtocol(asyncio.DatagramProtocol):
    """Asyncio DatagramProtocol bridging low-level socket reception with packet processing."""

    def __init__(self, on_packet_received: Callable[[bytes, Tuple[str, int]], None]) -> None:
        self.transport: Optional[asyncio.DatagramTransport] = None
        self._on_packet_received = on_packet_received

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        if isinstance(transport, asyncio.DatagramTransport):
            self.transport = transport

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        try:
            self._on_packet_received(data, addr)
        except Exception as exc:
            logger.error("Error processing incoming datagram from %s: %s", addr, exc, exc_info=True)

    def error_received(self, exc: Exception) -> None:
        logger.warning("UDP datagram socket error received: %s", exc)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        logger.info("UDP datagram socket connection closed.")


class UDPDiscovery:
    """Manages KDE Connect UDP discovery broadcasts and listener lifecycle."""

    def __init__(
        self,
        config: Any,
        bound_tcp_port_getter: Optional[Union[Callable[[], int], int]] = None,
        on_device_discovered: Optional[
            Callable[[DiscoveredDevice], Union[Coroutine[Any, Any, None], None]]
        ] = None,
        broadcast_interval: Optional[float] = None,
    ) -> None:
        self.config = config
        self._bound_tcp_port_getter = bound_tcp_port_getter
        self.on_device_discovered = on_device_discovered

        # Resolve broadcast interval
        if broadcast_interval is not None:
            self.broadcast_interval = broadcast_interval
        elif hasattr(config, "broadcast_interval"):
            self.broadcast_interval = config.broadcast_interval
        elif hasattr(config, "network") and hasattr(config.network, "broadcast_interval"):
            self.broadcast_interval = config.network.broadcast_interval
        else:
            self.broadcast_interval = 15.0

        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[UDPDiscoveryProtocol] = None
        self._broadcast_task: Optional[asyncio.Task[None]] = None
        self._sweep_task: Optional[asyncio.Task[None]] = None
        self._background_tasks: Set[asyncio.Task[Any]] = set()
        self._running = False

        # Debounce cache: device_id -> (timestamp, sender_ip, tcp_port)
        self._seen_cache_max_size = 1024
        self._seen_cache_ttl = 300.0  # 5 minutes expiration
        self._seen_cache: collections.OrderedDict[str, Tuple[float, str, int]] = collections.OrderedDict()
        self._debounce_window = 2.0
        self._discovery_semaphore = asyncio.Semaphore(16)

    @property
    def is_running(self) -> bool:
        return self._running

    def _get_device_id(self) -> str:
        if hasattr(self.config, "device_id"):
            return str(self.config.device_id)
        if hasattr(self.config, "device") and hasattr(self.config.device, "device_id"):
            return str(self.config.device.device_id)
        return "pengulink_unknown"

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

    def _get_udp_port(self) -> int:
        if hasattr(self.config, "udp_port"):
            return int(self.config.udp_port)
        if hasattr(self.config, "network") and hasattr(self.config.network, "udp_port"):
            return int(self.config.network.udp_port)
        return 1716

    def _get_tcp_port(self) -> int:
        if callable(self._bound_tcp_port_getter):
            try:
                return self._bound_tcp_port_getter()
            except Exception:
                pass
        elif isinstance(self._bound_tcp_port_getter, int):
            return self._bound_tcp_port_getter

        if hasattr(self.config, "tcp_port_min"):
            return int(self.config.tcp_port_min)
        if hasattr(self.config, "network") and hasattr(self.config.network, "tcp_port_min"):
            return int(self.config.network.tcp_port_min)
        return 1716

    async def start(self) -> None:
        """Initialize UDP socket, attach datagram endpoint, and start broadcast loop."""
        if self._running:
            return

        loop = asyncio.get_running_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)

        udp_port = self._get_udp_port()
        try:
            sock.bind(("0.0.0.0", udp_port))
            logger.info("Bound UDP discovery socket to 0.0.0.0:%d", udp_port)
        except OSError as exc:
            logger.warning(
                "Port %d busy (%s). Binding ephemeral UDP port for outbound broadcast.",
                udp_port,
                exc,
            )
            sock.bind(("0.0.0.0", 0))

        transport, protocol = await loop.create_datagram_endpoint(
            lambda: UDPDiscoveryProtocol(self._handle_datagram),
            sock=sock,
        )
        self._transport = transport
        self._protocol = protocol
        self._running = True

        # Launch startup burst and continuous broadcast loop
        self._broadcast_task = asyncio.create_task(
            self._broadcast_loop(), name="udp_broadcast_loop"
        )

    async def stop(self) -> None:
        """Cleanly cancel broadcast tasks and close the datagram transport."""
        if not self._running:
            return
        self._running = False

        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
            self._broadcast_task = None

        if self._sweep_task:
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except asyncio.CancelledError:
                pass
            self._sweep_task = None

        # Cancel and drain in-flight background callback tasks
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        if self._transport:
            self._transport.close()
            self._transport = None
        self._protocol = None
        self._seen_cache.clear()
        logger.info("UDP discovery service stopped.")

    def build_identity_packet(self) -> bytes:
        """Construct a serialized newline-terminated kdeconnect.identity UDP packet."""
        tcp_port = self._get_tcp_port()
        identity_body = IdentityBody(
            deviceId=self._get_device_id(),
            deviceName=self._get_device_name(),
            deviceType=self._get_device_type(),
            protocolVersion=self._get_protocol_version(),
            incomingCapabilities=self._get_incoming_capabilities(),
            outgoingCapabilities=self._get_outgoing_capabilities(),
            tcpPort=tcp_port,
        )
        packet = Packet.create_identity(identity_body)
        return packet.to_wire()

    async def broadcast_now(self, target_ip: Optional[str] = None) -> None:
        """Dispatch discovery datagram to all subnets or a specific target."""
        if not self._transport or self._transport.is_closing():
            return

        data = self.build_identity_packet()
        destinations = {target_ip} if target_ip else get_broadcast_addresses()
        udp_port = self._get_udp_port()

        for dest in destinations:
            try:
                self._transport.sendto(data, (dest, udp_port))
                logger.debug("Broadcasted identity to %s:%d", dest, udp_port)
            except OSError as exc:
                logger.debug("Failed sending broadcast to %s:%d: %s", dest, udp_port, exc)

    async def trigger_burst(self, count: int = 3, delay: float = 1.0) -> None:
        """Transmit rapid burst of broadcasts to overcome Wi-Fi multicast packet drops."""
        for _ in range(count):
            if not self._running:
                break
            await self.broadcast_now()
            await asyncio.sleep(delay)

    async def trigger_sweep(self, duration: float = 30.0, interval: float = 5.0) -> None:
        """Execute active sweep broadcast for interactive CLI/UI discovery."""
        if self._sweep_task and not self._sweep_task.done():
            self._sweep_task.cancel()

        async def _sweep() -> None:
            start_time = time.time()
            while self._running and (time.time() - start_time) < duration:
                await self.broadcast_now()
                await asyncio.sleep(interval)

        self._sweep_task = asyncio.create_task(_sweep(), name="udp_sweep_task")

    async def _broadcast_loop(self) -> None:
        """Continuous periodic background broadcast loop."""
        try:
            # Initial startup burst
            await self.trigger_burst(count=3, delay=1.0)
            while self._running:
                await asyncio.sleep(self.broadcast_interval)
                await self.broadcast_now()
        except asyncio.CancelledError:
            pass

    def _handle_datagram(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Process incoming raw datagram, parse identity, and dispatch discovered devices."""
        sender_ip, _ = addr
        text = data.decode("utf-8", errors="ignore").strip()
        if not text:
            return

        try:
            packet_dict = json.loads(text)
        except json.JSONDecodeError:
            logger.debug("Discarding malformed JSON UDP datagram from %s", sender_ip)
            return

        if not isinstance(packet_dict, dict):
            return

        packet_type = packet_dict.get("type")
        if packet_type != "kdeconnect.identity":
            return

        body = packet_dict.get("body")
        if not isinstance(body, dict):
            return

        remote_device_id = body.get("deviceId")
        if not remote_device_id or not isinstance(remote_device_id, str):
            return

        # 1. Deduplicate local loopback reflections
        local_id = self._get_device_id()
        if remote_device_id == local_id:
            logger.debug("Ignored local self-broadcast reflection from %s", sender_ip)
            return

        tcp_port = body.get("tcpPort", 1716)
        if not isinstance(tcp_port, int) or not (1 <= tcp_port <= 65535):
            tcp_port = 1716

        # 2. Debounce rapid repeat broadcasts and maintain bounded LRU seen cache
        now = time.time()
        last_entry = self._seen_cache.get(remote_device_id)
        if last_entry:
            last_time, last_ip, last_port = last_entry
            if (
                (now - last_time < self._debounce_window)
                and (last_ip == sender_ip)
                and (last_port == tcp_port)
            ):
                return
            self._seen_cache.pop(remote_device_id, None)

        # Prune expired entries
        while self._seen_cache:
            oldest_key, (oldest_time, _, _) = next(iter(self._seen_cache.items()))
            if now - oldest_time > self._seen_cache_ttl:
                self._seen_cache.pop(oldest_key, None)
            else:
                break

        # Cap cache capacity
        while len(self._seen_cache) >= self._seen_cache_max_size:
            self._seen_cache.popitem(last=False)

        self._seen_cache[remote_device_id] = (now, sender_ip, tcp_port)

        raw_version = body.get("protocolVersion", 7)
        try:
            protocol_version = int(raw_version)
        except (TypeError, ValueError):
            protocol_version = 7

        raw_incoming = body.get("incomingCapabilities", [])
        incoming_caps = tuple(raw_incoming) if isinstance(raw_incoming, (list, tuple)) else ()

        raw_outgoing = body.get("outgoingCapabilities", [])
        outgoing_caps = tuple(raw_outgoing) if isinstance(raw_outgoing, (list, tuple)) else ()

        discovered = DiscoveredDevice(
            device_id=remote_device_id,
            device_name=str(body.get("deviceName", "Unknown Device")),
            device_type=str(body.get("deviceType", "desktop")),
            protocol_version=protocol_version,
            tcp_port=tcp_port,
            sender_ip=sender_ip,
            incoming_capabilities=incoming_caps,
            outgoing_capabilities=outgoing_caps,
            timestamp=now,
        )

        logger.info(
            "Discovered device '%s' (%s) at %s:%d",
            discovered.device_name,
            discovered.device_id,
            sender_ip,
            tcp_port,
        )

        # Dispatch discovered device: invoke sync callbacks immediately, throttle async coroutines
        if self.on_device_discovered:
            cb_res = self.on_device_discovered(discovered)
            if inspect.isawaitable(cb_res):
                async def _throttled_dispatch(coro: Any) -> None:
                    async with self._discovery_semaphore:
                        await coro

                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(
                        _throttled_dispatch(cb_res),
                        name=f"udp_on_device_discovered_{discovered.device_id}",
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                except RuntimeError:
                    cb_res.close()


# Alias for UdpDiscovery
UdpDiscovery = UDPDiscovery

__all__ = [
    "DiscoveredDevice",
    "get_broadcast_addresses",
    "UDPDiscoveryProtocol",
    "UDPDiscovery",
    "UdpDiscovery",
]

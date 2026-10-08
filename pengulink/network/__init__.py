"""PenguLink networking subsystem.

Provides packet parsing, newline JSON framing, 128 KB buffer protection,
UDP discovery, and TCP stream management.
"""

from pengulink.network.packet import (
    MAX_PACKET_SIZE,
    IdentityBody,
    NetworkPacket,
    Packet,
    PacketEngine,
    StreamLineReader,
)
from pengulink.network.tcp import (
    ConnectionRole,
    TCPConnection,
    TCPStreamManager,
    TcpStreamManager,
    configure_stream_socket,
)
from pengulink.network.udp import (
    DiscoveredDevice,
    UDPDiscovery,
    UdpDiscovery,
    get_broadcast_addresses,
)

__all__ = [
    # Packet Engine
    "Packet",
    "NetworkPacket",
    "IdentityBody",
    "PacketEngine",
    "StreamLineReader",
    "MAX_PACKET_SIZE",
    # UDP Discovery
    "UDPDiscovery",
    "UdpDiscovery",
    "DiscoveredDevice",
    "get_broadcast_addresses",
    # TCP Streaming
    "TCPStreamManager",
    "TcpStreamManager",
    "TCPConnection",
    "ConnectionRole",
    "configure_stream_socket",
]

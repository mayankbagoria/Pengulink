"""pengulink.plugins.ping
~~~~~~~~~~~~~~~~~~~~~
KDE Connect Ping plugin implementation.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from pengulink.core.events import EventBus, PingReceived
from pengulink.network.packet import Packet
from pengulink.plugins.base import Plugin, SendPacketCallable, safe_send_packet

logger = logging.getLogger("pengulink.plugins.ping")

PING_PACKET_TYPE = "kdeconnect.ping"


def create_ping_packet(message: Optional[str] = None) -> Packet:
    """Construct a kdeconnect.ping packet."""
    body = {}
    if message:
        body["message"] = str(message)
    return Packet(type=PING_PACKET_TYPE, body=body)


class PingPlugin(Plugin):
    """Handles sending and receiving KDE Connect ping packets."""

    @property
    def plugin_id(self) -> str:
        return "ping"

    @property
    def incoming_capabilities(self) -> List[str]:
        return [PING_PACKET_TYPE]

    @property
    def outgoing_capabilities(self) -> List[str]:
        return [PING_PACKET_TYPE]

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Handle incoming kdeconnect.ping packet and emit PingReceived event."""
        if packet.type != PING_PACKET_TYPE:
            return

        msg = packet.body.get("message")
        display_message = str(msg) if msg else "Ping!"

        logger.info("Ping received from %s: '%s'", device_id, display_message)

        if self.event_bus:
            await self.event_bus.publish(
                PingReceived(
                    device_id=device_id,
                    message=display_message,
                )
            )

    async def send_ping(
        self,
        send_packet: SendPacketCallable,
        message: Optional[str] = None,
    ) -> Packet:
        """Transmit a ping packet to the remote device."""
        packet = create_ping_packet(message)
        await safe_send_packet(send_packet, packet)
        return packet

"""pengulink.plugins.battery
~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect Battery information and monitoring plugin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from pengulink.core.events import BatteryUpdated, EventBus
from pengulink.network.packet import Packet
from pengulink.plugins.base import Plugin, SendPacketCallable, safe_send_packet

logger = logging.getLogger("pengulink.plugins.battery")

BATTERY_PACKET_TYPE = "kdeconnect.battery"
BATTERY_REQUEST_TYPE = "kdeconnect.battery.request"


@dataclass
class DeviceBatteryState:
    """Cached battery status for a device."""

    current_charge: int = -1
    is_charging: bool = False
    threshold_event: int = 0
    battery_quantity: int = 1


def create_battery_status_packet(
    current_charge: int,
    is_charging: bool,
    threshold_event: int = 0,
    battery_quantity: int = 1,
) -> Packet:
    """Construct a kdeconnect.battery status packet."""
    charge = max(-1, min(100, int(current_charge)))
    return Packet(
        type=BATTERY_PACKET_TYPE,
        body={
            "currentCharge": charge,
            "isCharging": bool(is_charging),
            "thresholdEvent": int(threshold_event),
            "batteryQuantity": int(battery_quantity),
        },
    )


def create_battery_request_packet() -> Packet:
    """Construct a kdeconnect.battery.request query packet."""
    return Packet(
        type=BATTERY_REQUEST_TYPE,
        body={"request": True},
    )


class BatteryPlugin(Plugin):
    """Monitors remote device battery and responds to battery queries."""

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        super().__init__(event_bus)
        self._device_states: Dict[str, DeviceBatteryState] = {}
        # Local host battery defaults (-1 = desktop/plugged-in AC)
        self.local_charge: int = -1
        self.local_is_charging: bool = True

    @property
    def plugin_id(self) -> str:
        return "battery"

    @property
    def incoming_capabilities(self) -> List[str]:
        return [BATTERY_PACKET_TYPE, BATTERY_REQUEST_TYPE]

    @property
    def outgoing_capabilities(self) -> List[str]:
        return [BATTERY_PACKET_TYPE, BATTERY_REQUEST_TYPE]

    def get_device_battery(self, device_id: str) -> Optional[DeviceBatteryState]:
        """Get cached battery state for a device."""
        return self._device_states.get(device_id)

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Handle incoming battery status and request packets."""
        if packet.type == BATTERY_PACKET_TYPE:
            body = packet.body

            # Defensive parsing
            raw_charge = body.get("currentCharge", -1)
            try:
                current_charge = max(-1, min(100, int(raw_charge)))
            except (TypeError, ValueError):
                current_charge = -1

            is_charging = bool(body.get("isCharging", False))

            try:
                threshold_event = int(body.get("thresholdEvent", 0))
            except (TypeError, ValueError):
                threshold_event = 0

            try:
                battery_quantity = int(body.get("batteryQuantity", 1))
            except (TypeError, ValueError):
                battery_quantity = 1

            state = DeviceBatteryState(
                current_charge=current_charge,
                is_charging=is_charging,
                threshold_event=threshold_event,
                battery_quantity=battery_quantity,
            )
            self._device_states[device_id] = state

            logger.info(
                "Battery update from %s: %d%% (%scharging)",
                device_id,
                current_charge,
                "" if is_charging else "not ",
            )

            if self.event_bus:
                await self.event_bus.publish(
                    BatteryUpdated(
                        device_id=device_id,
                        current_charge=current_charge,
                        is_charging=is_charging,
                        threshold_event=threshold_event,
                        battery_quantity=battery_quantity,
                    )
                )

        elif packet.type == BATTERY_REQUEST_TYPE:
            logger.debug("Received battery status request from %s; responding...", device_id)
            resp = create_battery_status_packet(
                current_charge=self.local_charge,
                is_charging=self.local_is_charging,
            )
            await safe_send_packet(send_packet, resp)

    async def request_battery_status(self, send_packet: SendPacketCallable) -> Packet:
        """Transmit a query requesting the remote peer report battery status."""
        packet = create_battery_request_packet()
        await safe_send_packet(send_packet, packet)
        return packet

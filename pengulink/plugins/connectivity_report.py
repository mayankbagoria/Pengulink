"""pengulink.plugins.connectivity_report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect cellular connectivity and signal strength monitoring plugin.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from pengulink.core.events import ConnectivityReportReceived, DeviceUnpaired, EventBus
from pengulink.network.packet import Packet
from pengulink.plugins.base import Plugin, SendPacketCallable, safe_send_packet

logger = logging.getLogger("pengulink.plugins.connectivity_report")

CONNECTIVITY_REPORT_PACKET_TYPE = "kdeconnect.connectivity_report"
CONNECTIVITY_REPORT_REQUEST_TYPE = "kdeconnect.connectivity_report.request"

# Canonical recognized network generation types
KNOWN_NETWORK_TYPES = {
    "GSM", "CDMA", "IDEN", "UMTS", "CDMA2000",
    "EDGE", "GPRS", "HSPA", "HSPAP", "LTE", "5G", "UNKNOWN"
}

# Maximum allowed subscriptions in a single report to prevent memory exhaustion DoS
MAX_SUBSCRIPTIONS = 16
# Maximum subscription identifier string length
MAX_SUB_ID_LEN = 32
# Maximum network type string length
MAX_NETWORK_TYPE_LEN = 32


@dataclass(frozen=True)
class SignalState:
    """Cellular radio status for an individual SIM subscription."""

    network_type: str = "Unknown"  # e.g., 'LTE', '5G', 'EDGE', 'Unknown'
    signal_strength: int = 0      # Coarse signal bars clamped strictly to 0..4

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary format matching upstream schema."""
        return {
            "network_type": self.network_type,
            "signal_strength": self.signal_strength,
            "networkType": self.network_type,
            "signalStrength": self.signal_strength,
        }


@dataclass(frozen=True)
class ConnectivityReport:
    """Immutable cached cellular connectivity report for a device."""

    device_id: str
    signal_strengths: Dict[str, SignalState] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON-RPC IPC and CLI consumption."""
        return {
            "device_id": self.device_id,
            "signal_strengths": {
                sub_id: state.to_dict()
                for sub_id, state in self.signal_strengths.items()
            },
            "timestamp": self.timestamp,
        }


def parse_signal_state(raw_data: Any) -> SignalState:
    """Defensively coerce and validate an individual Signal object."""
    if not isinstance(raw_data, dict):
        return SignalState(network_type="Unknown", signal_strength=0)

    # 1. Coerce networkType
    raw_type = raw_data.get("networkType") or raw_data.get("network_type") or "Unknown"
    cleaned_type = str(raw_type)[:MAX_NETWORK_TYPE_LEN].strip()
    if not cleaned_type or cleaned_type.upper() == "UNKNOWN":
        cleaned_type = "Unknown"
    else:
        upper_type = cleaned_type.upper()
        if upper_type in KNOWN_NETWORK_TYPES or upper_type in {"4G", "3G", "2G"}:
            cleaned_type = upper_type

    # 2. Coerce signalStrength
    raw_strength = raw_data.get("signalStrength") if "signalStrength" in raw_data else raw_data.get("signal_strength", 0)
    try:
        if isinstance(raw_strength, (int, float)):
            int_val = int(round(raw_strength))
        elif isinstance(raw_strength, str):
            int_val = int(round(float(raw_strength.strip())))
        else:
            int_val = 0
        clamped_strength = max(0, min(4, int_val))
    except (ValueError, TypeError, OverflowError):
        clamped_strength = 0

    return SignalState(network_type=cleaned_type, signal_strength=clamped_strength)


def create_connectivity_request_packet() -> Packet:
    """Construct an on-demand kdeconnect.connectivity_report.request query packet."""
    return Packet(
        type=CONNECTIVITY_REPORT_REQUEST_TYPE,
        body={},
    )


class ConnectivityReportPlugin(Plugin):
    """Monitors cellular radio connectivity and signal strength updates."""

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        super().__init__(event_bus)
        self._device_reports: Dict[str, ConnectivityReport] = {}
        self._unsub_unpair: Optional[Callable[[], None]] = None
        if self.event_bus:
            self._unsub_unpair = self.event_bus.subscribe(DeviceUnpaired, self._on_device_unpaired)

    def _on_device_unpaired(self, event: DeviceUnpaired) -> None:
        """Purge cached state when a device is unpaired."""
        if event and event.device_id:
            self.clear_device(event.device_id)

    @property
    def plugin_id(self) -> str:
        return "connectivity_report"

    @property
    def incoming_capabilities(self) -> List[str]:
        return [CONNECTIVITY_REPORT_PACKET_TYPE]

    @property
    def outgoing_capabilities(self) -> List[str]:
        return [CONNECTIVITY_REPORT_REQUEST_TYPE]

    def get_device_report(self, device_id: str) -> Optional[ConnectivityReport]:
        """Get latest cached connectivity report for a device."""
        return self._device_reports.get(device_id)

    def clear_device(self, device_id: str) -> bool:
        """Purge cached connectivity report for a device.

        Returns True if a report was removed, False otherwise.
        """
        removed = self._device_reports.pop(device_id, None) is not None
        if removed:
            logger.debug("Purged connectivity report cache for device %s", device_id)
        return removed

    def clear_all(self) -> None:
        """Purge all cached device connectivity reports."""
        self._device_reports.clear()

    async def stop(self) -> None:
        """Clean up resources on plugin shutdown."""
        if self._unsub_unpair:
            try:
                self._unsub_unpair()
            except Exception:
                pass
            self._unsub_unpair = None
        self.clear_all()

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Process incoming kdeconnect.connectivity_report packets."""
        if packet.type != CONNECTIVITY_REPORT_PACKET_TYPE:
            return

        body = packet.body
        if not isinstance(body, dict):
            logger.debug("Received malformed non-dict body in connectivity report from %s", device_id)
            return

        raw_strengths = body.get("signalStrengths")
        if raw_strengths is None:
            # Tolerant parsing for empty or missing signalStrengths
            logger.debug("Missing signalStrengths in connectivity report from %s", device_id)
            raw_strengths = {}
        elif not isinstance(raw_strengths, dict):
            logger.debug("Non-dict signalStrengths in connectivity report from %s", device_id)
            raw_strengths = {}

        if len(raw_strengths) > MAX_SUBSCRIPTIONS:
            logger.warning(
                "Excessive subscriptions (%d > %d) from %s; truncating.",
                len(raw_strengths),
                MAX_SUBSCRIPTIONS,
                device_id,
            )

        parsed_strengths: Dict[str, SignalState] = {}
        for idx, (sub_id_raw, val) in enumerate(raw_strengths.items()):
            if idx >= MAX_SUBSCRIPTIONS:
                break
            sub_id = str(sub_id_raw)[:MAX_SUB_ID_LEN].strip()
            if not sub_id:
                sub_id = "0"
            parsed_strengths[sub_id] = parse_signal_state(val)

        report = ConnectivityReport(
            device_id=device_id,
            signal_strengths=parsed_strengths,
            timestamp=time.time(),
        )
        self._device_reports[device_id] = report

        logger.info(
            "Connectivity report update from %s: %d active subscription(s)",
            device_id,
            len(parsed_strengths),
        )

        if self.event_bus:
            await self.event_bus.publish(
                ConnectivityReportReceived(
                    device_id=device_id,
                    signal_strengths=report.to_dict()["signal_strengths"],
                )
            )

    async def request_connectivity_report(self, send_packet: SendPacketCallable) -> Packet:
        """Transmit an on-demand query requesting remote peer report connectivity state."""
        packet = create_connectivity_request_packet()
        await safe_send_packet(send_packet, packet)
        return packet

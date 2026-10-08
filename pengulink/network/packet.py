"""PenguLink Packet Engine & Framing.

Provides Pydantic v2 schemas for the KDE Connect protocol packet envelope,
identity packet body, wire serialization/deserialization, 128 KB framing guards,
and permanent mitigation against CVE-2020-26164 buffer exhaustion attacks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pengulink.core.exceptions import (
    InvalidPacketSchemaError,
    MalformedPacketError,
    PacketBufferOverrunError,
    PacketFramingError,
    TruncatedPacketError,
)

logger = logging.getLogger("pengulink.network.packet")

# Maximum allowable packet line length in bytes (128 KB)
MAX_PACKET_SIZE = 131072

# Regex matching valid KDE Connect dot-separated packet types
PACKET_TYPE_REGEX = re.compile(r"^kdeconnect(\.[a-zA-Z0-9_-]+)+$")

# Standard recognized device types
DeviceType = Literal["desktop", "laptop", "phone", "tablet", "tv"]


class IdentityBody(BaseModel):
    """Schema for the body of a 'kdeconnect.identity' packet."""

    model_config = ConfigDict(extra="allow")

    deviceId: str = Field(
        min_length=1, max_length=128, description="Unique permanent device identifier"
    )
    deviceName: str = Field(
        min_length=1, max_length=64, description="Human-readable device name"
    )
    deviceType: str = Field(default="desktop", description="Device form factor")
    protocolVersion: int = Field(
        default=7, ge=6, le=10, description="KDE Connect protocol revision"
    )
    incomingCapabilities: List[str] = Field(
        default_factory=list, description="Packet types device can receive"
    )
    outgoingCapabilities: List[str] = Field(
        default_factory=list, description="Packet types device can send"
    )
    tcpPort: int = Field(default=1716, ge=1024, le=65535, description="TCP listener port")
    targetDeviceId: Optional[str] = Field(
        default=None, description="Optional target device ID for directed preambles"
    )
    targetProtocolVersion: Optional[int] = Field(
        default=None, description="Optional target protocol version"
    )

    @field_validator("deviceName")
    @classmethod
    def sanitize_device_name(cls, v: str) -> str:
        # Strip leading/trailing whitespace and control characters
        cleaned = re.sub(r"[\x00-\x1f\x7f]", "", v).strip()
        if not cleaned:
            return "PenguLink Device"
        return cleaned[:64]

    @field_validator("deviceType")
    @classmethod
    def normalize_device_type(cls, v: str) -> str:
        v_lower = v.lower()
        if v_lower in {"desktop", "laptop", "phone", "tablet", "tv"}:
            return v_lower
        return "desktop"


class Packet(BaseModel):
    """Top-level KDE Connect protocol network packet envelope."""

    model_config = ConfigDict(extra="allow")

    id: int = Field(
        default_factory=lambda: int(time.time() * 1000),
        description="Millisecond epoch timestamp",
    )
    type: str = Field(..., description="KDE Connect packet type")
    body: Dict[str, Any] = Field(default_factory=dict, description="Plugin-specific payload object")
    payloadSize: Optional[int] = Field(
        default=None, description="Attached binary payload size in bytes"
    )
    payloadTransferInfo: Optional[Dict[str, Any]] = Field(
        default=None, description="Attached binary payload connection parameters"
    )

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id(cls, v: Any) -> int:
        try:
            val = int(v)
            if val < 0:
                raise ValueError("Packet ID cannot be negative")
            return val
        except (ValueError, TypeError) as exc:
            raise InvalidPacketSchemaError(
                f"Invalid packet id: {v!r}. Must be a non-negative integer timestamp."
            ) from exc

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if not isinstance(v, str) or not PACKET_TYPE_REGEX.match(v):
            raise InvalidPacketSchemaError(
                f"Invalid packet type: {v!r}. Must match regex {PACKET_TYPE_REGEX.pattern}."
            )
        return v

    @field_validator("body")
    @classmethod
    def validate_body(cls, v: Any) -> Dict[str, Any]:
        if not isinstance(v, dict):
            raise InvalidPacketSchemaError(
                f"Packet body must be a JSON object (dict), got {type(v).__name__}."
            )
        return v

    def to_wire(self) -> bytes:
        """Serialize packet to newline-delimited UTF-8 JSON bytes."""
        json_str = self.model_dump_json(exclude_none=True)
        return json_str.encode("utf-8") + b"\n"

    def to_dict(self) -> Dict[str, Any]:
        """Convert packet to standard Python dictionary."""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_wire(cls, data: bytes | str) -> Packet:
        """Parse and validate a wire-formatted JSON line into a Packet instance."""
        return PacketEngine.parse(data)

    @classmethod
    def create_identity(cls, identity: IdentityBody) -> Packet:
        """Convenience constructor to create an identity packet."""
        return cls(
            type="kdeconnect.identity",
            body=identity.model_dump(exclude_none=True),
        )

    def parse_identity(self) -> IdentityBody:
        """Extract and validate the body as IdentityBody."""
        if self.type != "kdeconnect.identity":
            raise InvalidPacketSchemaError(f"Expected 'kdeconnect.identity', got '{self.type}'.")
        return IdentityBody.model_validate(self.body)


# Backward and cross-milestone compatibility alias
NetworkPacket = Packet


class PacketEngine:
    """Stateless packet serialization, parsing, and framing utilities."""

    @staticmethod
    def serialize(packet: Packet) -> bytes:
        """Serializes a packet to newline-delimited UTF-8 bytes."""
        return packet.to_wire()

    @staticmethod
    def parse(line: bytes | str) -> Packet:
        """Parses a single newline-delimited JSON line into a validated Packet.

        Raises:
            PacketBufferOverrunError: If line exceeds MAX_PACKET_SIZE.
            MalformedPacketError: If line is not valid UTF-8 or valid JSON object.
            InvalidPacketSchemaError: If JSON does not conform to the Packet schema.
        """
        if isinstance(line, str):
            raw_bytes = line.encode("utf-8")
        else:
            raw_bytes = line

        if len(raw_bytes) > MAX_PACKET_SIZE:
            raise PacketBufferOverrunError(
                f"Line length ({len(raw_bytes)} bytes) exceeds maximum limit ({MAX_PACKET_SIZE} bytes)."
            )

        # Strip trailing carriage returns and newlines
        clean_bytes = raw_bytes.rstrip(b"\r\n")
        if not clean_bytes.strip():
            raise MalformedPacketError("Received empty or whitespace-only line.")

        try:
            text = clean_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            snippet = clean_bytes[:64].hex()
            raise MalformedPacketError(
                f"Failed to decode UTF-8 wire bytes (hex snippet: {snippet}): {exc}"
            ) from exc

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            snippet = text[:128]
            raise MalformedPacketError(
                f"JSON decode failed on line snippet '{snippet}': {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise MalformedPacketError(
                f"Root JSON entity must be a dictionary, got {type(data).__name__}."
            )

        try:
            return Packet.model_validate(data)
        except Exception as exc:
            if isinstance(exc, (InvalidPacketSchemaError, MalformedPacketError, PacketBufferOverrunError)):
                raise
            raise InvalidPacketSchemaError(f"Packet validation error: {exc}") from exc

    @staticmethod
    def parse_safe(line: bytes | str) -> Optional[Packet]:
        """Safe parser that logs warnings and returns None on malformed packets

        instead of raising exceptions. Does NOT suppress PacketBufferOverrunError.
        """
        try:
            return PacketEngine.parse(line)
        except PacketBufferOverrunError:
            raise
        except (MalformedPacketError, InvalidPacketSchemaError) as exc:
            logger.warning("Discarded invalid packet: %s", exc)
            return None


class StreamLineReader:
    """Asynchronous line reader wrapping asyncio.StreamReader with strict

    buffer overrun guards against CVE-2020-26164.
    """

    def __init__(self, reader: asyncio.StreamReader, max_consecutive_errors: int = 5) -> None:
        self._reader = reader
        self._max_consecutive_errors = max_consecutive_errors
        self._consecutive_errors = 0

    async def read_packet(self) -> Optional[Packet]:
        """Reads next complete packet from the stream.

        Returns:
            Packet instance on success.
            None on EOF (peer closed connection).

        Raises:
            PacketBufferOverrunError: On buffer overrun (>128 KB without newline).
            MalformedPacketError: If consecutive parsing error threshold is reached.
            TruncatedPacketError: If EOF is reached without terminating newline.
        """
        while True:
            try:
                line = await self._reader.readline()
            except (asyncio.LimitOverrunError, ValueError) as exc:
                # Handle Python 3.14 dual exception behavior for limit overrun
                is_overrun = isinstance(exc, asyncio.LimitOverrunError) or (
                    "exceed the limit" in str(exc).lower()
                    or "separator is not found" in str(exc).lower()
                )
                if is_overrun:
                    # Clean internal reader buffer to prevent memory exhaustion
                    if hasattr(self._reader, "_buffer"):
                        self._reader._buffer.clear()
                    logger.error(
                        "SECURITY ALERT: Buffer overrun detected (>%d bytes). "
                        "Terminating connection to mitigate CVE-2020-26164.",
                        MAX_PACKET_SIZE,
                    )
                    raise PacketBufferOverrunError(
                        f"Stream buffer overrun exceeded {MAX_PACKET_SIZE} bytes without newline."
                    ) from exc
                raise

            # Detect clean EOF
            if not line:
                return None

            # Skip empty lines (blank newlines)
            if not line.strip():
                continue

            # Check if line was truncated at EOF (missing terminating newline)
            if not line.endswith(b"\n"):
                logger.warning("Truncated line received at stream EOF (%d bytes).", len(line))
                raise TruncatedPacketError("Stream reached EOF before newline was received.")

            packet = PacketEngine.parse_safe(line)
            if packet is not None:
                self._consecutive_errors = 0
                return packet

            self._consecutive_errors += 1
            if self._consecutive_errors >= self._max_consecutive_errors:
                raise MalformedPacketError(
                    f"Exceeded maximum allowable consecutive malformed packets ({self._max_consecutive_errors})."
                )

    async def stream_packets(self) -> AsyncIterator[Packet]:
        """Asynchronous generator yielding packets until EOF or fatal error."""
        while True:
            packet = await self.read_packet()
            if packet is None:
                break
            yield packet


__all__ = [
    "MAX_PACKET_SIZE",
    "PACKET_TYPE_REGEX",
    "DeviceType",
    "IdentityBody",
    "Packet",
    "NetworkPacket",
    "PacketEngine",
    "StreamLineReader",
]

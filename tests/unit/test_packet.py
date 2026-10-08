"""Unit tests for PenguLink Packet Engine and Framing."""

import asyncio
import pytest

from pengulink.core.exceptions import (
    InvalidPacketSchemaError,
    LimitOverrunError,
    MalformedPacketError,
    PacketBufferOverrunError,
    TruncatedPacketError,
)
from pengulink.network.packet import (
    MAX_PACKET_SIZE,
    IdentityBody,
    Packet,
    PacketEngine,
    StreamLineReader,
)


def test_valid_serialization() -> None:
    """TEST-PKT-001: Assert valid serialization produces newline-terminated UTF-8 bytes."""
    pkt = Packet(type="kdeconnect.ping", body={})
    wire = pkt.to_wire()
    assert wire.endswith(b"\n")
    assert b'"type":"kdeconnect.ping"' in wire
    assert b'"body":{}' in wire


def test_valid_deserialization() -> None:
    """TEST-PKT-002: Assert valid wire bytes deserialize into Packet."""
    wire = b'{"id":1,"type":"kdeconnect.ping","body":{}}\n'
    pkt = Packet.from_wire(wire)
    assert pkt.id == 1
    assert pkt.type == "kdeconnect.ping"
    assert pkt.body == {}


def test_id_timestamp_coercion() -> None:
    """TEST-PKT-003: Assert ID is coerced to integer."""
    wire = b'{"id":"1701026880000","type":"kdeconnect.ping","body":{}}\n'
    pkt = Packet.from_wire(wire)
    assert pkt.id == 1701026880000
    assert isinstance(pkt.id, int)


def test_identity_creation_and_extraction() -> None:
    """TEST-PKT-004: Assert identity packet creation and extraction."""
    ident = IdentityBody(
        deviceId="dev_123456789012345678901234567890",
        deviceName="Phone 1",
        deviceType="phone",
        protocolVersion=7,
        tcpPort=1716,
        incomingCapabilities=["kdeconnect.ping"],
        outgoingCapabilities=["kdeconnect.ping"],
    )
    pkt = Packet.create_identity(ident)
    assert pkt.type == "kdeconnect.identity"
    assert pkt.body["deviceId"] == ident.deviceId

    extracted = pkt.parse_identity()
    assert extracted.deviceId == ident.deviceId
    assert extracted.deviceName == "Phone 1"
    assert extracted.deviceType == "phone"
    assert extracted.tcpPort == 1716


def test_device_name_sanitization() -> None:
    """TEST-PKT-005: Assert control characters are stripped from device name."""
    ident = IdentityBody(
        deviceId="dev_123456789012345678901234567890",
        deviceName="\x00Dirty\x1fDevice\x7f",
    )
    assert ident.deviceName == "DirtyDevice"

    # Fallback when completely empty after stripping
    ident_empty = IdentityBody(
        deviceId="dev_123456789012345678901234567890",
        deviceName="\x00\x1f",
    )
    assert ident_empty.deviceName == "PenguLink Device"


def test_cve_2020_26164_buffer_overrun_parse() -> None:
    """TEST-PKT-006: Assert line exceeding 128 KB raises PacketBufferOverrunError in PacketEngine.parse."""
    oversized = b"A" * (MAX_PACKET_SIZE + 10)
    with pytest.raises(PacketBufferOverrunError):
        PacketEngine.parse(oversized)


async def test_cve_2020_26164_buffer_overrun_stream() -> None:
    """TEST-PKT-006b: Assert StreamLineReader raises PacketBufferOverrunError on oversized chunk."""
    reader = asyncio.StreamReader(limit=MAX_PACKET_SIZE)
    stream_reader = StreamLineReader(reader)

    # Feed data exceeding limit without newline
    reader.feed_data(b"X" * (MAX_PACKET_SIZE + 1024))

    with pytest.raises(PacketBufferOverrunError):
        await stream_reader.read_packet()


def test_malformed_json_syntax() -> None:
    """TEST-PKT-007: Assert broken JSON syntax raises MalformedPacketError."""
    wire = b'{"id": 1, "type": "kdeconnect.ping"\n'
    with pytest.raises(MalformedPacketError):
        Packet.from_wire(wire)


def test_default_fields() -> None:
    """TEST-PKT-008: Assert missing ID and body are defaulted."""
    wire = b'{"type": "kdeconnect.ping"}\n'
    pkt = Packet.from_wire(wire)
    assert pkt.id > 0
    assert pkt.body == {}


def test_null_body_rejection() -> None:
    """TEST-PKT-009: Assert null body raises InvalidPacketSchemaError."""
    wire = b'{"id":1,"type":"kdeconnect.ping","body":null}\n'
    with pytest.raises(InvalidPacketSchemaError):
        Packet.from_wire(wire)


def test_primitive_body_rejection() -> None:
    """TEST-PKT-010: Assert non-dict body raises InvalidPacketSchemaError."""
    wire = b'{"id":1,"type":"kdeconnect.ping","body":"not-a-dict"}\n'
    with pytest.raises(InvalidPacketSchemaError):
        Packet.from_wire(wire)


def test_crlf_handling() -> None:
    """TEST-PKT-011: Assert CRLF is cleanly handled and stripped."""
    wire = b'{"id":1,"type":"kdeconnect.ping","body":{}}\r\n'
    pkt = Packet.from_wire(wire)
    assert pkt.type == "kdeconnect.ping"


def test_multiline_escaped_json() -> None:
    """TEST-PKT-012: Assert multiline strings have exactly 1 physical newline on wire."""
    pkt = Packet(type="kdeconnect.ping", body={"text": "Line 1\nLine 2\nLine 3"})
    wire = pkt.to_wire()
    # Exactly one terminating newline
    assert wire.count(b"\n") == 1
    # Wire contains escaped newline \n
    assert b"Line 1\\nLine 2" in wire

    reloaded = Packet.from_wire(wire)
    assert reloaded.body["text"] == "Line 1\nLine 2\nLine 3"


def test_unicode_emojis() -> None:
    """TEST-PKT-013: Assert Unicode emojis survive roundtrip."""
    pkt = Packet(type="kdeconnect.ping", body={"title": "Penguin 🐧❤️"})
    wire = pkt.to_wire()
    reloaded = Packet.from_wire(wire)
    assert reloaded.body["title"] == "Penguin 🐧❤️"


def test_extra_field_tolerance() -> None:
    """TEST-PKT-014: Assert extra protocol fields are preserved in model."""
    wire = b'{"id":1,"type":"kdeconnect.ping","body":{},"customField":999}\n'
    pkt = Packet.from_wire(wire)
    assert pkt.type == "kdeconnect.ping"
    assert getattr(pkt, "customField", None) == 999 or pkt.__pydantic_extra__.get("customField") == 999


async def test_consecutive_malformed_limit() -> None:
    """TEST-PKT-015: Assert StreamLineReader raises MalformedPacketError after 5 bad lines."""
    reader = asyncio.StreamReader(limit=MAX_PACKET_SIZE)
    stream_reader = StreamLineReader(reader, max_consecutive_errors=3)

    # Feed 3 malformed lines
    reader.feed_data(b"{bad json 1\n{bad json 2\n{bad json 3\n")

    with pytest.raises(MalformedPacketError):
        await stream_reader.read_packet()


async def test_stream_packets_generator() -> None:
    """Assert stream_packets yields packets until EOF."""
    reader = asyncio.StreamReader(limit=MAX_PACKET_SIZE)
    stream_reader = StreamLineReader(reader)

    p1 = Packet(type="kdeconnect.ping", body={"idx": 1}).to_wire()
    p2 = Packet(type="kdeconnect.ping", body={"idx": 2}).to_wire()
    reader.feed_data(p1 + p2)
    reader.feed_eof()

    packets = [p async for p in stream_reader.stream_packets()]
    assert len(packets) == 2
    assert packets[0].body["idx"] == 1
    assert packets[1].body["idx"] == 2


async def test_truncated_packet_at_eof() -> None:
    """Assert EOF without terminating newline raises TruncatedPacketError."""
    reader = asyncio.StreamReader(limit=MAX_PACKET_SIZE)
    stream_reader = StreamLineReader(reader)

    # Incomplete line followed by EOF
    reader.feed_data(b'{"id":1,"type":"kdeconnect.ping"')
    reader.feed_eof()

    with pytest.raises(TruncatedPacketError):
        await stream_reader.read_packet()


def test_negative_id_rejection() -> None:
    """Assert negative packet ID is rejected."""
    wire = b'{"id":-10,"type":"kdeconnect.ping","body":{}}\n'
    with pytest.raises(InvalidPacketSchemaError):
        Packet.from_wire(wire)


def test_invalid_packet_type_rejection() -> None:
    """Assert non-KDE Connect packet types are rejected."""
    # Missing kdeconnect prefix
    with pytest.raises(InvalidPacketSchemaError):
        Packet.from_wire(b'{"id":1,"type":"custom.ping","body":{}}\n')

    # Empty type
    with pytest.raises(InvalidPacketSchemaError):
        Packet.from_wire(b'{"id":1,"type":"","body":{}}\n')


def test_parse_safe_behavior() -> None:
    """Assert parse_safe returns None on malformed packet, but raises on buffer overrun."""
    # Malformed JSON returns None
    assert PacketEngine.parse_safe(b"{broken json\n") is None

    # Oversized line raises PacketBufferOverrunError (not suppressed)
    with pytest.raises(PacketBufferOverrunError):
        PacketEngine.parse_safe(b"A" * (MAX_PACKET_SIZE + 10))


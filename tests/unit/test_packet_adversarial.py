"""Empirical Adversarial Stress Tests for PenguLink Packet Engine.

Covers CVE-2020-26164 buffer exhaustion attacks, corrupted JSON, binary garbage,
null bytes, unicode edge cases, rapid multi-packet streaming, EOF truncations,
ReDoS regex verification, and memory/resource safety.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, List
import pytest
from pydantic import ValidationError

from pengulink.core.exceptions import (
    InvalidPacketSchemaError,
    MalformedPacketError,
    PacketBufferOverrunError,
    TruncatedPacketError,
)
from pengulink.network.packet import (
    MAX_PACKET_SIZE,
    PACKET_TYPE_REGEX,
    IdentityBody,
    Packet,
    PacketEngine,
    StreamLineReader,
)


# ==============================================================================
# 1. CVE-2020-26164 Buffer Exhaustion & Oversized Line Defense
# ==============================================================================

def test_adv_oversized_line_exact_boundary() -> None:
    """Verify behavior at exact MAX_PACKET_SIZE boundary (131072 bytes)."""
    # 131072 bytes total: valid JSON padded with spaces inside body
    padding_len = MAX_PACKET_SIZE - len(b'{"id":1,"type":"kdeconnect.ping","body":{"pad":""}}\n')
    valid_large = (
        b'{"id":1,"type":"kdeconnect.ping","body":{"pad":"'
        + b"a" * padding_len
        + b'"}}\n'
    )
    assert len(valid_large) == MAX_PACKET_SIZE
    pkt = PacketEngine.parse(valid_large)
    assert pkt.type == "kdeconnect.ping"
    assert len(pkt.body["pad"]) == padding_len

    # Exactly 1 byte over MAX_PACKET_SIZE (131073 bytes)
    over_by_one = valid_large[:-1] + b"a\n"
    assert len(over_by_one) == MAX_PACKET_SIZE + 1
    with pytest.raises(PacketBufferOverrunError):
        PacketEngine.parse(over_by_one)

    with pytest.raises(PacketBufferOverrunError):
        PacketEngine.parse_safe(over_by_one)


def test_adv_massive_stream_overrun_no_newline() -> None:
    """Verify PacketEngine.parse raises immediately on massive input without parsing JSON."""
    massive = b"X" * (1024 * 1024 * 5)  # 5 MB
    with pytest.raises(PacketBufferOverrunError) as exc_info:
        PacketEngine.parse(massive)
    assert f"{MAX_PACKET_SIZE}" in str(exc_info.value)


async def test_adv_stream_reader_incremental_overrun() -> None:
    """Verify StreamLineReader triggers buffer overrun when fed in small incremental chunks."""
    reader = asyncio.StreamReader(limit=MAX_PACKET_SIZE)
    stream_reader = StreamLineReader(reader)

    # Feed in chunks of 4096 bytes without newline until exceeding limit
    total_fed = 0
    chunk = b"A" * 4096
    while total_fed <= MAX_PACKET_SIZE + 8192:
        reader.feed_data(chunk)
        total_fed += len(chunk)

    with pytest.raises(PacketBufferOverrunError):
        await stream_reader.read_packet()

    # Verify internal buffer was purged to prevent memory exhaustion
    if hasattr(reader, "_buffer"):
        assert len(reader._buffer) == 0


async def test_adv_stream_reader_recovery_after_overrun_attempt() -> None:
    """Verify that buffer purge leaves reader clean."""
    reader = asyncio.StreamReader(limit=MAX_PACKET_SIZE)
    stream_reader = StreamLineReader(reader)

    # Overrun feed
    reader.feed_data(b"Z" * (MAX_PACKET_SIZE + 4096))
    with pytest.raises(PacketBufferOverrunError):
        await stream_reader.read_packet()

    # Buffer was cleared
    if hasattr(reader, "_buffer"):
        assert len(reader._buffer) == 0


# ==============================================================================
# 2. Corrupted and Malformed JSON Syntax
# ==============================================================================

@pytest.mark.parametrize(
    "corrupted_input",
    [
        b"",  # Empty
        b"\n",  # Single newline
        b"\r\n",  # CRLF only
        b"   \t  \r\n",  # Whitespace only
        b"{",  # Unclosed brace
        b"}",  # Stray close brace
        b'{"id":',  # Truncated key-value
        b'{"id": 1, "type": "kdeconnect.ping"',  # Missing closing brace
        b'{"id": 1, "type": "kdeconnect.ping", "body": {}',  # Missing closing brace
        b'{"id": 1, "type": "kdeconnect.ping", "body": {}} trailing_garbage\n',
        b'leading_garbage {"id": 1, "type": "kdeconnect.ping", "body": {}}\n',
        b'{"id": 1, "type": "kdeconnect.ping", "body": {}}{"id": 2}\n',  # Multi-JSON on one line
        b"{'id': 1, 'type': 'kdeconnect.ping', 'body': {}}\n",  # Single quotes (invalid JSON)
        b'{"id": 1, "type": "kdeconnect.ping", "body": {},,,}\n',  # Extra commas
        b'{"id": 1, "type": "kdeconnect.ping", "body": undefined}\n',  # JavaScript undefined
        b'\xef\xbb\xbf{"id": 1, "type": "kdeconnect.ping", "body": {}}\n',  # Leading UTF-8 BOM
    ],
)
def test_adv_corrupted_json_syntax(corrupted_input: bytes) -> None:
    """Assert all malformed JSON inputs raise MalformedPacketError and return None in parse_safe."""
    with pytest.raises(MalformedPacketError):
        PacketEngine.parse(corrupted_input)

    assert PacketEngine.parse_safe(corrupted_input) is None


@pytest.mark.parametrize(
    "non_dict_json",
    [
        b"[]\n",
        b"[1, 2, 3]\n",
        b'"just a string"\n',
        b"123456\n",
        b"-99.9\n",
        b"true\n",
        b"false\n",
        b"null\n",
    ],
)
def test_adv_non_dict_root_types(non_dict_json: bytes) -> None:
    """Root JSON entity must be a dictionary; non-dict primitives must raise MalformedPacketError."""
    with pytest.raises(MalformedPacketError) as exc_info:
        PacketEngine.parse(non_dict_json)
    assert "dictionary" in str(exc_info.value).lower()
    assert PacketEngine.parse_safe(non_dict_json) is None


def test_adv_deeply_nested_json() -> None:
    """Stress test parser against deeply nested JSON structures (recursion defense)."""
    # Create 200 levels of nested objects: {"a": {"a": ... {}}}
    nested = b'{"id": 1, "type": "kdeconnect.ping", "body": ' + (b'{"nested": ' * 200) + b"{}" + (b"}" * 200) + b"}\n"
    try:
        pkt = PacketEngine.parse(nested)
        assert pkt.type == "kdeconnect.ping"
    except (MalformedPacketError, InvalidPacketSchemaError, RecursionError):
        pass  # Handled safely without segmentation fault or unhandled fatal exception


# ==============================================================================
# 3. Binary Garbage & Non-UTF8 Injections
# ==============================================================================

def test_adv_random_binary_garbage() -> None:
    """Verify random binary byte sequences are safely rejected with MalformedPacketError."""
    for length in [16, 64, 256, 1024, 65535]:
        garbage = os.urandom(length)
        # Avoid accidental newline terminating it early
        garbage = garbage.replace(b"\n", b"\x01") + b"\n"
        with pytest.raises(MalformedPacketError):
            PacketEngine.parse(garbage)
        assert PacketEngine.parse_safe(garbage) is None


def test_adv_invalid_utf8_sequences() -> None:
    """Verify invalid UTF-8 multibyte sequences trigger MalformedPacketError."""
    invalid_sequences = [
        b"\xff\xfe\xfd\n",
        b"\xc0\xaf\n",  # Overlong encoding
        b"\xe0\x80\xaf\n",  # Overlong 3-byte
        b"\xf0\x80\x80\xaf\n",  # Overlong 4-byte
        b"\xed\xa0\x80\n",  # UTF-16 surrogate
        b'{"id": 1, "type": "kdeconnect.ping", "body": {"bad": "\x80\x81\x82"}}\n',
    ]
    for bad_seq in invalid_sequences:
        with pytest.raises(MalformedPacketError):
            PacketEngine.parse(bad_seq)
        assert PacketEngine.parse_safe(bad_seq) is None


# ==============================================================================
# 4. Null Bytes Injections
# ==============================================================================

def test_adv_null_bytes_in_wire_stream() -> None:
    """Verify null bytes in raw wire data are rejected."""
    wire_with_null = b'\x00{"id": 1, "type": "kdeconnect.ping", "body": {}}\n'
    with pytest.raises(MalformedPacketError):
        PacketEngine.parse(wire_with_null)


def test_adv_null_bytes_in_packet_type() -> None:
    """Null bytes in packet type must fail regex validation."""
    wire = b'{"id": 1, "type": "kdeconnect.ping\\u0000", "body": {}}\n'
    with pytest.raises(InvalidPacketSchemaError):
        PacketEngine.parse(wire)


def test_adv_null_bytes_in_identity_body() -> None:
    """Null bytes in deviceName should be stripped and sanitized by IdentityBody."""
    ident = IdentityBody(
        deviceId="device\x00123",
        deviceName="Phone\x00\x01\x1fName\x7f",
        deviceType="phone",
    )
    # Control chars stripped
    assert ident.deviceName == "PhoneName"

    # All-control-char name falls back to default
    ident_fallback = IdentityBody(
        deviceId="device123",
        deviceName="\x00\x01\x02\x03",
    )
    assert ident_fallback.deviceName == "PenguLink Device"


# ==============================================================================
# 5. Unicode Edge Cases & Escaped Content
# ==============================================================================

def test_adv_unicode_complex_scripts_and_emojis() -> None:
    """Complex multilingual, RTL, and composite emoji sequences."""
    test_strings = [
        "مرحبا بالعالم",  # Arabic (RTL)
        "שלום עולם",  # Hebrew (RTL)
        "こんにちは世界",  # Japanese Kanji/Kana
        "👩🏽‍💻 Family ZWJ 👨‍👩‍👧‍👦",  # Composite Emojis with Zero-Width-Joiners
        "Z̸a̷l̸g̸o̶ ̷T̵e̸x̵t̶",  # Combining diacritics
        "Line1\r\nLine2\rLine3\nLine4",  # Internal escaped CR/LF
        "Double quote: \" and backslash: \\ and slash: /",
    ]
    for text in test_strings:
        pkt = Packet(type="kdeconnect.ping", body={"content": text})
        wire = pkt.to_wire()
        # Must have exactly 1 physical newline
        assert wire.count(b"\n") == 1
        reloaded = Packet.from_wire(wire)
        assert reloaded.body["content"] == text


def test_adv_unicode_large_within_limit() -> None:
    """Large UTF-8 payload (100 KB) within 128 KB limit roundtrips without error."""
    large_unicode = "🎉" * 25000  # 4 bytes per emoji = 100,000 bytes
    pkt = Packet(type="kdeconnect.ping", body={"data": large_unicode})
    wire = pkt.to_wire()
    assert len(wire) < MAX_PACKET_SIZE
    reloaded = Packet.from_wire(wire)
    assert reloaded.body["data"] == large_unicode


# ==============================================================================
# 6. Rapid Multi-Packet Streams & Stream Framing
# ==============================================================================

async def test_adv_rapid_stream_1000_packets() -> None:
    """Feed 1,000 packets in a single feed_data call to test framing integrity."""
    reader = asyncio.StreamReader(limit=MAX_PACKET_SIZE)
    stream_reader = StreamLineReader(reader)

    packet_count = 1000
    wire_stream = b"".join(
        Packet(type="kdeconnect.ping", body={"seq": i}).to_wire()
        for i in range(packet_count)
    )

    reader.feed_data(wire_stream)
    reader.feed_eof()

    received: List[Packet] = []
    async for pkt in stream_reader.stream_packets():
        received.append(pkt)

    assert len(received) == packet_count
    for i, pkt in enumerate(received):
        assert pkt.body["seq"] == i


async def test_adv_consecutive_error_counter_reset() -> None:
    """Verify that interleaved bad and good packets reset consecutive error count."""
    reader = asyncio.StreamReader(limit=MAX_PACKET_SIZE)
    stream_reader = StreamLineReader(reader, max_consecutive_errors=3)

    # Feed sequence: Bad, Bad, Good, Bad, Bad, Good
    wire = (
        b"{bad1\n"
        b"{bad2\n"
        b'{"id":1,"type":"kdeconnect.ping","body":{"idx":1}}\n'
        b"{bad3\n"
        b"{bad4\n"
        b'{"id":2,"type":"kdeconnect.ping","body":{"idx":2}}\n'
    )
    reader.feed_data(wire)
    reader.feed_eof()

    p1 = await stream_reader.read_packet()
    assert p1 is not None and p1.body["idx"] == 1

    p2 = await stream_reader.read_packet()
    assert p2 is not None and p2.body["idx"] == 2

    # Verify stream EOF
    p3 = await stream_reader.read_packet()
    assert p3 is None


async def test_adv_consecutive_error_exceeded_triggers_error() -> None:
    """Verify that consecutive errors reaching threshold raises MalformedPacketError."""
    reader = asyncio.StreamReader(limit=MAX_PACKET_SIZE)
    stream_reader = StreamLineReader(reader, max_consecutive_errors=4)

    # 4 consecutive bad lines
    reader.feed_data(b"bad1\nbad2\nbad3\nbad4\n")

    with pytest.raises(MalformedPacketError) as exc_info:
        await stream_reader.read_packet()
    assert "consecutive malformed" in str(exc_info.value).lower()


async def test_adv_byte_by_byte_stream_delivery() -> None:
    """Stream a packet 1 byte at a time to test reassembly across chunk boundaries."""
    reader = asyncio.StreamReader(limit=MAX_PACKET_SIZE)
    stream_reader = StreamLineReader(reader)

    pkt = Packet(type="kdeconnect.ping", body={"msg": "reassembly_test"})
    wire = pkt.to_wire()

    # Feed each byte separately
    for b in wire:
        reader.feed_data(bytes([b]))

    parsed = await stream_reader.read_packet()
    assert parsed is not None
    assert parsed.body["msg"] == "reassembly_test"


# ==============================================================================
# 7. Truncated Packets at EOF
# ==============================================================================

async def test_adv_truncated_eof_handling() -> None:
    """Assert EOF without trailing newline raises TruncatedPacketError."""
    reader = asyncio.StreamReader(limit=MAX_PACKET_SIZE)
    stream_reader = StreamLineReader(reader)

    # Incomplete packet missing closing brace and newline
    reader.feed_data(b'{"id":1,"type":"kdeconnect.ping"')
    reader.feed_eof()

    with pytest.raises(TruncatedPacketError):
        await stream_reader.read_packet()


async def test_adv_clean_empty_eof() -> None:
    """Assert clean EOF on empty stream returns None immediately."""
    reader = asyncio.StreamReader(limit=MAX_PACKET_SIZE)
    stream_reader = StreamLineReader(reader)
    reader.feed_eof()

    result = await stream_reader.read_packet()
    assert result is None


async def test_adv_packets_followed_by_truncated_tail() -> None:
    """Valid packet followed by truncated fragment raises TruncatedPacketError after valid packet."""
    reader = asyncio.StreamReader(limit=MAX_PACKET_SIZE)
    stream_reader = StreamLineReader(reader)

    p1 = Packet(type="kdeconnect.ping", body={"id": "valid"}).to_wire()
    truncated_tail = b'{"id": 2, "type": "kdeconnect'

    reader.feed_data(p1 + truncated_tail)
    reader.feed_eof()

    # First packet passes
    pkt1 = await stream_reader.read_packet()
    assert pkt1 is not None and pkt1.body["id"] == "valid"

    # Truncated fragment raises TruncatedPacketError
    with pytest.raises(TruncatedPacketError):
        await stream_reader.read_packet()


# ==============================================================================
# 8. Schema Boundary & Type Safety
# ==============================================================================

@pytest.mark.parametrize(
    "invalid_type",
    [
        "kdeconnect",  # No dot
        "kdeconnect.",  # Trailing dot
        "kdeconnect..ping",  # Consecutive dots
        "KDECONNECT.PING",  # Uppercase prefix
        "kdeconnect/ping",  # Slash separator
        "kdeconnect ping",  # Space separator
        "org.kdeconnect.ping",  # Non-kdeconnect prefix
        "",  # Empty
        "kdeconnect.ping@device",  # Disallowed character @
        "kdeconnect.ping!alert",  # Disallowed character !
        "kdeconnect.ping#hash",  # Disallowed character #
    ],
)
def test_adv_packet_type_regex_strictness(invalid_type: str) -> None:
    """Assert packet type must strictly adhere to kdeconnect dot-notation regex."""
    with pytest.raises(InvalidPacketSchemaError):
        Packet(type=invalid_type, body={})


def test_adv_packet_type_regex_redos_defense() -> None:
    """Assert PACKET_TYPE_REGEX is immune to catastrophic backtracking (ReDoS)."""
    # Hostile payload with many valid-looking subcomponents ending in invalid character
    hostile_input = "kdeconnect." + "sub." * 5000 + "invalid!"
    t0 = time.perf_counter()
    match = PACKET_TYPE_REGEX.match(hostile_input)
    elapsed = time.perf_counter() - t0
    assert match is None
    # Must evaluate in under 50 milliseconds
    assert elapsed < 0.05, f"Potential ReDoS detected: took {elapsed:.4f}s"


@pytest.mark.parametrize(
    "invalid_id",
    [
        -1,
        -999999999,
        "not-a-number",
        "-12345",
        [],
        {},
        None,
    ],
)
def test_adv_packet_id_coercion_failures(invalid_id: Any) -> None:
    """Assert negative IDs and non-numeric representations raise InvalidPacketSchemaError."""
    with pytest.raises(InvalidPacketSchemaError):
        Packet(id=invalid_id, type="kdeconnect.ping", body={})


@pytest.mark.parametrize(
    "invalid_body_bytes",
    [
        b"null",
        b"[]",
        b"[1, 2, 3]",
        b'"string_body"',
        b"12345",
        b"true",
    ],
)
def test_adv_wire_packet_body_non_dict_rejected(invalid_body_bytes: bytes) -> None:
    """Assert wire packets with non-dictionary body types raise InvalidPacketSchemaError."""
    wire = b'{"id": 1, "type": "kdeconnect.ping", "body": ' + invalid_body_bytes + b"}\n"
    with pytest.raises(InvalidPacketSchemaError):
        Packet.from_wire(wire)


@pytest.mark.parametrize(
    "invalid_body",
    [
        None,
        [],
        [1, 2, 3],
        "string_body",
        12345,
        True,
    ],
)
def test_adv_model_packet_body_non_dict_rejected(invalid_body: Any) -> None:
    """Assert direct Packet constructor rejects non-dict body with ValidationError or InvalidPacketSchemaError."""
    with pytest.raises((InvalidPacketSchemaError, ValidationError)):
        Packet(type="kdeconnect.ping", body=invalid_body)


def test_adv_identity_body_boundaries() -> None:
    """Verify bounds checking in IdentityBody schema."""
    # Port out of bounds
    with pytest.raises(ValidationError):
        IdentityBody(deviceId="dev1", deviceName="Dev", tcpPort=80)  # < 1024
    with pytest.raises(ValidationError):
        IdentityBody(deviceId="dev1", deviceName="Dev", tcpPort=70000)  # > 65535

    # Protocol version out of bounds
    with pytest.raises(ValidationError):
        IdentityBody(deviceId="dev1", deviceName="Dev", protocolVersion=4)  # < 6
    with pytest.raises(ValidationError):
        IdentityBody(deviceId="dev1", deviceName="Dev", protocolVersion=15)  # > 10

    # Device ID too long (>128)
    with pytest.raises(ValidationError):
        IdentityBody(deviceId="a" * 129, deviceName="Dev")

    # Non-identity packet calling parse_identity()
    non_ident = Packet(type="kdeconnect.ping", body={})
    with pytest.raises(InvalidPacketSchemaError):
        non_ident.parse_identity()

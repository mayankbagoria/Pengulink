"""Unit tests for PenguLink exception hierarchy."""

import pytest

from pengulink.core.exceptions import (
    ConfigError,
    ConfigValidationError,
    ConnectionClosedError,
    ConnectionTimeoutError,
    CryptoError,
    FramingError,
    InvalidPacketSchemaError,
    InvalidStateTransitionError,
    LimitOverrunError,
    MalformedPacketError,
    NetworkError,
    PacketBufferOverrunError,
    PacketError,
    PacketFramingError,
    PacketParseError,
    PacketValidationError,
    PairingError,
    PenguLinkError,
    PortBindingError,
    StateMachineError,
    TLSHandshakeError,
    TLSUpgradeError,
    TruncatedPacketError,
)


def test_exception_hierarchy() -> None:
    """Assert all custom exceptions inherit from PenguLinkError."""
    assert issubclass(ConfigError, PenguLinkError)
    assert issubclass(PacketError, PenguLinkError)
    assert issubclass(NetworkError, PenguLinkError)
    assert issubclass(CryptoError, PenguLinkError)
    assert issubclass(StateMachineError, PenguLinkError)
    assert issubclass(PairingError, PenguLinkError)

    # Packet hierarchy
    assert issubclass(PacketParseError, PacketError)
    assert issubclass(MalformedPacketError, PacketParseError)
    assert issubclass(TruncatedPacketError, MalformedPacketError)
    assert issubclass(PacketValidationError, PacketError)
    assert issubclass(InvalidPacketSchemaError, PacketValidationError)
    assert issubclass(FramingError, PacketError)
    assert issubclass(PacketFramingError, PacketError)
    assert issubclass(LimitOverrunError, FramingError)
    assert issubclass(PacketBufferOverrunError, LimitOverrunError)

    # Network hierarchy
    assert issubclass(PortBindingError, NetworkError)
    assert issubclass(ConnectionTimeoutError, NetworkError)
    assert issubclass(ConnectionClosedError, NetworkError)

    # TLS hierarchy
    assert issubclass(TLSUpgradeError, CryptoError)
    assert issubclass(TLSHandshakeError, CryptoError)


def test_exception_context_serialization() -> None:
    """Assert to_dict() produces expected dictionary structure."""
    err = PenguLinkError("Something failed", context={"peer": "192.168.1.10", "port": 1716})
    data = err.to_dict()
    assert data["error_type"] == "PenguLinkError"
    assert data["message"] == "Something failed"
    assert data["context"] == {"peer": "192.168.1.10", "port": 1716}

    # Verify __str__ includes context
    assert "peer='192.168.1.10'" in str(err)
    assert "port=1716" in str(err)


def test_invalid_state_transition_formatting() -> None:
    """Assert InvalidStateTransitionError formats descriptive transition string."""
    err = InvalidStateTransitionError("dev_alpha", "DISCONNECTED", "CONNECTED")
    assert "dev_alpha" in str(err)
    assert "DISCONNECTED" in str(err)
    assert "CONNECTED" in str(err)
    assert err.context["device_id"] == "dev_alpha"
    assert err.context["current_state"] == "DISCONNECTED"
    assert err.context["attempted_state"] == "CONNECTED"


def test_packet_parse_error_snippet() -> None:
    """Assert PacketParseError attaches and bounds raw snippet."""
    raw = b"BAD_DATA_" * 50
    err = PacketParseError("Malformed header", raw_snippet=raw)
    assert "snippet" in err.context
    assert len(err.context["snippet"]) <= 128


def test_limit_overrun_error() -> None:
    """Assert LimitOverrunError records consumed bytes."""
    err = LimitOverrunError("Buffer exceeded", consumed_bytes=200000)
    assert err.context["consumed_bytes"] == 200000
    assert issubclass(type(err), FramingError)
    assert issubclass(type(err), PacketError)

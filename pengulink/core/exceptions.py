"""PenguLink custom exception hierarchy.

Provides structured, typed exceptions for configuration, packet parsing,
network streaming, cryptography, device state machine, pairing, storage,
and daemon IPC.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class PenguLinkError(Exception):
    """Base exception for all PenguLink errors.

    Attributes:
        message: Human-readable error description.
        context: Optional dictionary containing debugging metadata.
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} ({ctx_str})"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Serialize exception details for structured logging or IPC transmission."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
        }


# =====================================================================
# Configuration Exceptions
# =====================================================================

class ConfigError(PenguLinkError):
    """Base exception for configuration failures."""


class ConfigValidationError(ConfigError):
    """Raised when configuration values fail schema or semantic validation."""


class ConfigNotFoundError(ConfigError):
    """Raised when an explicit configuration file cannot be found."""


# =====================================================================
# Packet & Framing Exceptions
# =====================================================================

class PacketError(PenguLinkError):
    """Base exception for packet handling and framing errors."""


class PacketParseError(PacketError):
    """Raised when raw stream bytes cannot be decoded as UTF-8 or parsed as JSON."""

    def __init__(
        self,
        message: str,
        raw_snippet: Optional[bytes] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = dict(context) if context else {}
        if raw_snippet is not None:
            # Keep snippet bounded to avoid log flooding
            ctx["snippet"] = raw_snippet[:128].decode("utf-8", errors="replace")
        super().__init__(message, ctx)


class MalformedPacketError(PacketParseError):
    """Raised when a packet line is syntactically invalid or non-object root JSON."""


class TruncatedPacketError(MalformedPacketError):
    """Raised when stream ends prematurely or line is truncated at EOF."""


class PacketValidationError(PacketError):
    """Raised when JSON payload violates KDE Connect envelope invariants.

    Invariants require top-level dict, 'id' (int/float), 'type' (str), and 'body' (dict).
    """


class InvalidPacketSchemaError(PacketValidationError):
    """Raised when packet envelope schema validation fails."""


class UnknownPacketTypeError(PacketError):
    """Raised when a packet type cannot be routed to any registered plugin."""


class FramingError(PacketError):
    """Raised when stream framing or newline delimitation is violated."""


# Alias for FramingError
PacketFramingError = FramingError


class LimitOverrunError(FramingError):
    """Raised when an incoming line exceeds the 128 KB buffer guard without a newline.

    Mitigates CVE-2020-26164 buffer exhaustion attacks.
    """

    def __init__(
        self,
        message: str = "Stream line exceeded buffer guard limit (128 KB) without newline delimiter",
        consumed_bytes: int = 131072,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = dict(context) if context else {}
        ctx["consumed_bytes"] = consumed_bytes
        super().__init__(message, ctx)


# Alias for LimitOverrunError
PacketBufferOverrunError = LimitOverrunError


# =====================================================================
# Network & Socket Exceptions
# =====================================================================

class NetworkError(PenguLinkError):
    """Base exception for networking and socket I/O errors."""


class PortBindingError(NetworkError):
    """Raised when the daemon cannot bind to UDP 1716 or any TCP port in 1716..1764."""

    def __init__(
        self,
        message: str,
        protocol: str = "TCP",
        port_range: Optional[str] = "1716-1764",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = dict(context) if context else {}
        ctx["protocol"] = protocol
        ctx["port_range"] = port_range
        super().__init__(message, ctx)


class ConnectionTimeoutError(NetworkError):
    """Raised when a network socket connect, read, or write operation times out."""

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = dict(context) if context else {}
        if timeout_seconds is not None:
            ctx["timeout_seconds"] = timeout_seconds
        if host is not None:
            ctx["host"] = host
        if port is not None:
            ctx["port"] = port
        super().__init__(message, ctx)


class ConnectionClosedError(NetworkError):
    """Raised when a remote socket closes unexpectedly (EOF or TCP reset)."""


class ConnectionRejectedError(NetworkError):
    """Raised when a remote endpoint actively refuses connection or drops handshake."""


class InterfaceEnumerationError(NetworkError):
    """Raised when local network interfaces or broadcast addresses cannot be enumerated."""


class SocketConfigurationError(NetworkError):
    """Raised when setting socket options (SO_REUSEADDR, SO_BROADCAST, TCP keepalive) fails."""


# =====================================================================
# Cryptography & TLS Exceptions
# =====================================================================

class CryptoError(PenguLinkError):
    """Base exception for cryptographic operations and TLS handshakes."""


class KeyGenerationError(CryptoError):
    """Raised when 2048-bit RSA key generation fails."""


class CertificateError(CryptoError):
    """Raised when X.509 certificate generation, serialization, or parsing fails."""


class CertificateVerificationError(CryptoError):
    """Raised when peer TLS certificate does not match pinned trust or Subject CN mismatches.

    Critical security defense against identity spoofing.
    """

    def __init__(
        self,
        message: str,
        expected_device_id: Optional[str] = None,
        presented_cn: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = dict(context) if context else {}
        if expected_device_id is not None:
            ctx["expected_device_id"] = expected_device_id
        if presented_cn is not None:
            ctx["presented_cn"] = presented_cn
        super().__init__(message, ctx)


class CertificatePermissionError(CryptoError):
    """Raised when private key file permissions on disk violate the strict POSIX 0600 rule."""


class TLSUpgradeError(CryptoError):
    """Raised when STARTTLS upgrade on an active TCP connection fails."""


# Alias for TLSUpgradeError
TLSHandshakeError = TLSUpgradeError


# =====================================================================
# State Machine Exceptions
# =====================================================================

class StateMachineError(PenguLinkError):
    """Base exception for device lifecycle state machine violations."""


class InvalidStateTransitionError(StateMachineError):
    """Raised when an illegal transition between device lifecycle states is attempted."""

    def __init__(
        self,
        device_id: str,
        current_state: str,
        attempted_state: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = dict(context) if context else {}
        ctx["device_id"] = device_id
        ctx["current_state"] = current_state
        ctx["attempted_state"] = attempted_state
        msg = f"Illegal transition for device '{device_id}': '{current_state}' -> '{attempted_state}'"
        super().__init__(msg, ctx)


class DeviceNotFoundError(StateMachineError):
    """Raised when querying a device ID not tracked in the active registry or storage."""


class DeviceAlreadyConnectedError(StateMachineError):
    """Raised when an unauthenticated incoming connection attempts to preempt an active session.

    Mitigates CVE-2020-26164 session replacement attacks.
    """


# =====================================================================
# Pairing Negotiation Exceptions
# =====================================================================

class PairingError(PenguLinkError):
    """Base exception for pairing negotiation failures."""


class PairingRejectedError(PairingError):
    """Raised when the remote peer or local user explicitly rejects pairing (pair: false)."""


class PairingTimeoutError(PairingError):
    """Raised when user confirmation for pairing is not received within the timeout."""


class AuthenticationCodeMismatchError(PairingError):
    """Raised when the calculated 8-character visual pairing code does not match."""


class PairingTimestampExpiredError(PairingError):
    """Raised when pairing packet timestamp exceeds the allowable 30-minute drift window."""


# =====================================================================
# Storage & Persistence Exceptions
# =====================================================================

class StorageError(PenguLinkError):
    """Base exception for storage backend operations."""


class StorageCorruptedError(StorageError):
    """Raised when stored device JSON or SQLite database file is corrupted on disk."""


class StorageWriteError(StorageError):
    """Raised when atomic file persistence or disk flush fails."""


# =====================================================================
# Plugin System Exceptions
# =====================================================================

class PluginError(PenguLinkError):
    """Base exception for plugin registration and packet routing."""


class PluginNotFoundError(PluginError):
    """Raised when an action is dispatched to an unregistered plugin."""


class PluginInitializationError(PluginError):
    """Raised when a plugin fails its start() lifecycle initialization."""


# =====================================================================
# Daemon & IPC Exceptions
# =====================================================================

class DaemonError(PenguLinkError):
    """Base exception for daemon service operations."""


class IPCError(DaemonError):
    """Raised when Unix domain socket JSON-RPC communication fails."""


class DaemonAlreadyRunningError(DaemonError):
    """Raised when another PenguLink daemon instance holds the Unix domain socket lock."""


class DaemonNotRunningError(DaemonError):
    """Raised when the CLI attempts to communicate with a daemon that is not running."""


__all__ = [
    "PenguLinkError",
    "ConfigError",
    "ConfigValidationError",
    "ConfigNotFoundError",
    "PacketError",
    "PacketParseError",
    "MalformedPacketError",
    "TruncatedPacketError",
    "PacketValidationError",
    "InvalidPacketSchemaError",
    "UnknownPacketTypeError",
    "FramingError",
    "PacketFramingError",
    "LimitOverrunError",
    "PacketBufferOverrunError",
    "NetworkError",
    "PortBindingError",
    "ConnectionTimeoutError",
    "ConnectionClosedError",
    "ConnectionRejectedError",
    "InterfaceEnumerationError",
    "SocketConfigurationError",
    "CryptoError",
    "KeyGenerationError",
    "CertificateError",
    "CertificateVerificationError",
    "CertificatePermissionError",
    "TLSUpgradeError",
    "TLSHandshakeError",
    "StateMachineError",
    "InvalidStateTransitionError",
    "DeviceNotFoundError",
    "DeviceAlreadyConnectedError",
    "PairingError",
    "PairingRejectedError",
    "PairingTimeoutError",
    "AuthenticationCodeMismatchError",
    "PairingTimestampExpiredError",
    "StorageError",
    "StorageCorruptedError",
    "StorageWriteError",
    "PluginError",
    "PluginNotFoundError",
    "PluginInitializationError",
    "DaemonError",
    "IPCError",
    "DaemonAlreadyRunningError",
    "DaemonNotRunningError",
]

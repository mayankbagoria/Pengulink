"""PenguLink custom exception hierarchy.

Provides structured, typed exceptions for configuration, packet parsing,
network streaming, cryptography, device state machine, pairing, storage,
and daemon IPC.
"""

from __future__ import annotations

from typing import Any


class PenguLinkError(Exception):
    """Base exception for all PenguLink errors.

    Attributes:
        message: Human-readable error description.
        context: Optional dictionary containing debugging metadata.
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} ({ctx_str})"
        return self.message

    def to_dict(self) -> dict[str, Any]:
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
        raw_snippet: bytes | None = None,
        context: dict[str, Any] | None = None,
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
        context: dict[str, Any] | None = None,
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
        port_range: str | None = "1716-1764",
        context: dict[str, Any] | None = None,
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
        timeout_seconds: float | None = None,
        host: str | None = None,
        port: int | None = None,
        context: dict[str, Any] | None = None,
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
        expected_device_id: str | None = None,
        presented_cn: str | None = None,
        context: dict[str, Any] | None = None,
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
        context: dict[str, Any] | None = None,
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
# Contacts Exceptions
# =====================================================================

class ContactsError(PluginError):
    """Base exception for contacts synchronization and vCard handling."""


class ContactNotFoundError(ContactsError):
    """Raised when a requested contact UID is not found for a device."""


class SyncTimeoutError(ContactsError, ConnectionTimeoutError):
    """Raised when contact synchronization packet exchange times out."""


ContactsSyncError = ContactsError
ContactsTimeoutError = SyncTimeoutError


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


# =====================================================================
# Alert & Audio Exceptions
# =====================================================================

class AlertError(PenguLinkError):
    """Base exception for desktop alert service operations."""


class AudioBackendError(AlertError):
    """Raised when an audio backend utility fails or an invalid backend is requested."""


class AudioUnavailableError(AlertError):
    """Raised when no audio playback utility is functional on the host."""


# =====================================================================
# Share & Transfer Exceptions
# =====================================================================

class ShareError(PenguLinkError):
    """Base exception for Share feature operations."""


class PathTraversalError(PenguLinkError):
    """Raised when an incoming filename or destination path violates sandbox containment."""


class TransferError(ShareError):
    """Base exception for file transfer operations."""


class IncompleteTransferError(TransferError):
    """Raised when an incoming file transfer terminates before payloadSize bytes are received."""


class PayloadSizeExceededError(TransferError):
    """Raised when an incoming stream exceeds the declared packet payloadSize."""


class DestinationDirectoryError(StorageError):
    """Raised when the resolved download directory cannot be created or is unwritable."""


class PayloadTransferError(TransferError):
    """Base exception for out-of-band payload transfer failures."""


class PayloadPortBindingError(PayloadTransferError, PortBindingError):
    """Raised when no TCP ports in 1739..1764 are available for binding."""


class PayloadTimeoutError(PayloadTransferError, ConnectionTimeoutError):
    """Raised when peer connection or socket I/O times out during payload transfer."""


# =====================================================================
# MPRIS Exceptions
# =====================================================================

class MprisError(PluginError):
    """Base exception for media control subsystem and MPRIS operations."""


class MprisAdapterError(MprisError):
    """Raised when an MPRIS host adapter encounters an operational error."""


class MprisAdapterUnavailableError(MprisAdapterError):
    """Raised when a requested MPRIS host adapter is not functional in current environment."""


class MprisPlayerNotFoundError(MprisError):
    """Raised when targeted media player does not exist or is not registered."""


PlayerNotFoundError = MprisPlayerNotFoundError


class ActionUnsupportedError(MprisError):
    """Raised when a media transport action cannot be executed or is unsupported."""


MprisActionError = ActionUnsupportedError


class MprisPayloadError(MprisError):
    """Raised when out-of-band album art payload transfer fails."""


# =====================================================================
# Remote Commands Exceptions (Milestone 15)
# =====================================================================

class RemoteCommandError(PluginError):
    """Base exception for RemoteCommands plugin errors."""


class RemoteCommandNotFoundError(RemoteCommandError):
    """Raised when a specified remote command key is not found on the device."""


# =====================================================================
# Run Command Host Exceptions (Milestone 23 & 25)
# =====================================================================

class RunCommandError(PluginError):
    """Base exception for RunCommand plugin and local execution errors."""


class CommandNotFoundError(RunCommandError, KeyError):
    """Raised when a local command key is not found in the CommandStore."""


class CommandExecutionError(RunCommandError):
    """Raised when execution of a local command fails."""


# =====================================================================
# Telephony Exceptions (Milestone 27)
# =====================================================================

class TelephonyError(PluginError):
    """Base exception for telephony plugin and call management errors."""


class CallNotFoundError(TelephonyError, KeyError):
    """Raised when an active call cannot be found for a given device ID."""


class InvalidCallStateError(TelephonyError):
    """Raised when an invalid call event or state transition is encountered."""


# =====================================================================
# PauseMusic Exceptions (Milestone 28)
# =====================================================================

class PauseMusicError(PluginError):
    """Base exception for PauseMusic plugin and media coordination errors."""


class PauseMusicConfigError(PauseMusicError):
    """Raised when PauseMusic configuration validation fails."""


__all__ = [
    "ActionUnsupportedError",
    "AlertError",
    "AudioBackendError",
    "AudioUnavailableError",
    "AuthenticationCodeMismatchError",
    "CallNotFoundError",
    "CertificateError",
    "CertificatePermissionError",
    "CertificateVerificationError",
    "CommandExecutionError",
    "CommandNotFoundError",
    "ConfigError",
    "ConfigNotFoundError",
    "ConfigValidationError",
    "ConnectionClosedError",
    "ConnectionRejectedError",
    "ConnectionTimeoutError",
    "ContactNotFoundError",
    "ContactsError",
    "ContactsSyncError",
    "ContactsTimeoutError",
    "CryptoError",
    "DaemonAlreadyRunningError",
    "DaemonError",
    "DaemonNotRunningError",
    "DestinationDirectoryError",
    "DeviceAlreadyConnectedError",
    "DeviceNotFoundError",
    "FramingError",
    "IPCError",
    "IncompleteTransferError",
    "InterfaceEnumerationError",
    "InvalidCallStateError",
    "InvalidPacketSchemaError",
    "InvalidStateTransitionError",
    "KeyGenerationError",
    "LimitOverrunError",
    "MalformedPacketError",
    "MprisActionError",
    "MprisAdapterError",
    "MprisAdapterUnavailableError",
    "MprisError",
    "MprisPayloadError",
    "MprisPlayerNotFoundError",
    "NetworkError",
    "PacketBufferOverrunError",
    "PacketError",
    "PacketFramingError",
    "PacketParseError",
    "PacketValidationError",
    "PairingError",
    "PairingRejectedError",
    "PairingTimeoutError",
    "PairingTimestampExpiredError",
    "PathTraversalError",
    "PauseMusicConfigError",
    "PauseMusicError",
    "PayloadPortBindingError",
    "PayloadSizeExceededError",
    "PayloadTimeoutError",
    "PayloadTransferError",
    "PenguLinkError",
    "PlayerNotFoundError",
    "PluginError",
    "PluginInitializationError",
    "PluginNotFoundError",
    "PortBindingError",
    "RemoteCommandError",
    "RemoteCommandNotFoundError",
    "RunCommandError",
    "ShareError",
    "SocketConfigurationError",
    "StateMachineError",
    "StorageCorruptedError",
    "StorageError",
    "StorageWriteError",
    "SyncTimeoutError",
    "TLSHandshakeError",
    "TLSUpgradeError",
    "TelephonyError",
    "TransferError",
    "TruncatedPacketError",
    "UnknownPacketTypeError",
]

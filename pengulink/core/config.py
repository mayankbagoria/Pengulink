"""PenguLink configuration management.

Provides typed, validated configuration models using Pydantic v2, supporting
hierarchical loading (defaults -> JSON file -> environment variables),
safe atomic disk persistence, and directory initialization.
"""

from __future__ import annotations

import json
import os
import platform
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from pengulink.core.exceptions import ConfigValidationError


# Canonical regex for device IDs (32-38 chars, hex or hyphen/underscore)
DEVICE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{32,38}$")

# Characters to strip from hostnames for device name sanitization
INVALID_NAME_CHARS_REGEX = re.compile(r"[\"',;:\.!\?\(\)\[\]<>]")


def sanitize_device_name(raw_name: str) -> str:
    """Sanitize device name to conform with KDE Connect conventions.

    Strips forbidden punctuation and truncates to 32 characters.
    """
    cleaned = INVALID_NAME_CHARS_REGEX.sub("", raw_name).strip()
    if not cleaned:
        cleaned = "PenguLink-Linux"
    return cleaned[:32]


def default_device_id() -> str:
    """Generate a canonical 32-character lowercase hex device ID."""
    return uuid.uuid4().hex


def default_device_name() -> str:
    """Generate default device name from system hostname."""
    host = platform.node() or "PenguLink-Linux"
    return sanitize_device_name(host)


def default_config_dir() -> Path:
    """Return XDG-compliant configuration directory."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "pengulink"
    return Path.home() / ".config" / "pengulink"


def default_data_dir() -> Path:
    """Return XDG-compliant persistent data directory."""
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "pengulink"
    return Path.home() / ".local" / "share" / "pengulink"


def default_runtime_dir() -> Path:
    """Return XDG-compliant runtime directory for sockets and locks."""
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "pengulink"
    return Path.home() / ".cache" / "pengulink" / "run"


# =====================================================================
# Sub-Configuration Models
# =====================================================================

class DeviceConfig(BaseModel):
    """Local device identity and capabilities settings."""

    device_id: str = Field(
        default_factory=default_device_id,
        description="Unique permanent identifier of the node (matches certificate Subject CN)",
    )
    device_name: str = Field(
        default_factory=default_device_name,
        description="Human-readable device name advertised to peers",
    )
    device_type: Literal["desktop", "laptop", "phone", "tablet", "tv"] = Field(
        default="desktop",
        description="KDE Connect device form factor",
    )
    protocol_version: int = Field(
        default=7,
        ge=6,
        le=10,
        description="Advertised KDE Connect protocol version (7 baseline, 8 post-TLS identity)",
    )
    incoming_capabilities: List[str] = Field(
        default_factory=lambda: [
            "kdeconnect.ping",
            "kdeconnect.battery",
            "kdeconnect.battery.request",
            "kdeconnect.notification",
            "kdeconnect.notification.request",
            "kdeconnect.connectivity_report",
        ],
        description="Packet types this device can receive and process",
    )
    outgoing_capabilities: List[str] = Field(
        default_factory=lambda: [
            "kdeconnect.ping",
            "kdeconnect.battery",
            "kdeconnect.battery.request",
            "kdeconnect.notification",
            "kdeconnect.notification.action",
            "kdeconnect.notification.reply",
            "kdeconnect.connectivity_report.request",
        ],
        description="Packet types this device can generate and transmit",
    )

    @field_validator("device_id")
    @classmethod
    def validate_device_id(cls, v: str) -> str:
        if not DEVICE_ID_REGEX.match(v):
            raise ValueError(
                f"device_id '{v}' is invalid. Must be 32-38 alphanumeric characters, hyphens, or underscores."
            )
        return v

    @field_validator("device_name")
    @classmethod
    def validate_device_name(cls, v: str) -> str:
        sanitized = sanitize_device_name(v)
        if not sanitized:
            raise ValueError("device_name cannot be empty after sanitization")
        return sanitized


class NetworkConfig(BaseModel):
    """Network transport, ports, buffers, and keepalive settings."""

    udp_port: int = Field(
        default=1716,
        ge=1024,
        le=65535,
        description="Standard UDP discovery listening and broadcasting port",
    )
    tcp_port_min: int = Field(
        default=1716,
        ge=1024,
        le=65535,
        description="Starting port for sequential TCP server binding fallback",
    )
    tcp_port_max: int = Field(
        default=1764,
        ge=1024,
        le=65535,
        description="Ending port for sequential TCP server binding fallback",
    )
    broadcast_interval: float = Field(
        default=15.0,
        gt=0.0,
        description="Interval in seconds between periodic background UDP broadcasts",
    )
    discovery_sweep_duration: float = Field(
        default=30.0,
        gt=0.0,
        description="Active discovery sweep duration when initiated by CLI",
    )
    connect_timeout: float = Field(
        default=10.0,
        gt=0.0,
        description="TCP connection and initial handshake timeout in seconds",
    )
    stream_line_limit: int = Field(
        default=131072,  # 128 KB
        ge=1024,
        le=10485760,  # 10 MB maximum
        description="StreamReader line buffer limit preventing DoS memory exhaustion (CVE-2020-26164)",
    )
    tcp_keepalive_idle: int = Field(
        default=10,
        ge=1,
        description="Linux TCP_KEEPIDLE in seconds before starting keepalive probes",
    )
    tcp_keepalive_interval: int = Field(
        default=5,
        ge=1,
        description="Linux TCP_KEEPINTVL in seconds between keepalive probes",
    )
    tcp_keepalive_count: int = Field(
        default=3,
        ge=1,
        description="Linux TCP_KEEPCNT maximum unanswered probes before dropping socket",
    )
    ping_interval: float = Field(
        default=30.0,
        gt=0.0,
        description="Application-level kdeconnect.ping heartbeat interval during idle",
    )
    ping_timeout: float = Field(
        default=10.0,
        gt=0.0,
        description="Application-level ping response timeout before marking channel dead",
    )
    max_unauthenticated_connections: int = Field(
        default=10,
        ge=1,
        description="Maximum concurrent unauthenticated TCP connections allowed",
    )
    unauthenticated_timeout: float = Field(
        default=10.0,
        gt=0.0,
        description="Timeout for unauthenticated connections to complete TLS upgrade",
    )

    @model_validator(mode="after")
    def validate_port_range(self) -> NetworkConfig:
        if self.tcp_port_min > self.tcp_port_max:
            raise ValueError(
                f"tcp_port_min ({self.tcp_port_min}) cannot exceed tcp_port_max ({self.tcp_port_max})"
            )
        return self


class StorageConfig(BaseModel):
    """File paths and persistence directories."""

    config_dir: Path = Field(
        default_factory=default_config_dir,
        description="Base configuration directory",
    )
    data_dir: Path = Field(
        default_factory=default_data_dir,
        description="Base persistent data storage directory",
    )
    runtime_dir: Path = Field(
        default_factory=default_runtime_dir,
        description="Runtime socket and lock directory",
    )
    private_key_filename: str = Field(default="private_key.pem")
    certificate_filename: str = Field(default="certificate.pem")
    devices_filename: str = Field(default="devices.json")
    sqlite_filename: str = Field(default="pengulink.db")
    ipc_socket_filename: str = Field(default="daemon.sock")

    @property
    def private_key_path(self) -> Path:
        return self.config_dir / self.private_key_filename

    @property
    def certificate_path(self) -> Path:
        return self.config_dir / self.certificate_filename

    @property
    def devices_store_path(self) -> Path:
        return self.data_dir / self.devices_filename

    @property
    def sqlite_db_path(self) -> Path:
        return self.data_dir / self.sqlite_filename

    @property
    def ipc_socket_path(self) -> Path:
        return self.runtime_dir / self.ipc_socket_filename

    @property
    def config_file_path(self) -> Path:
        return self.config_dir / "config.json"


class LoggingConfig(BaseModel):
    """Logging subsystem configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging verbosity level",
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Optional file destination for daemon logs",
    )
    rich_console: bool = Field(
        default=True,
        description="Enable Rich formatting when stdout is an interactive terminal",
    )


# =====================================================================
# Root Configuration Model
# =====================================================================

class PenguLinkConfig(BaseModel):
    """Root configuration coordinating device, network, storage, and logging settings."""

    device: DeviceConfig = Field(default_factory=DeviceConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Convenience delegating properties
    @property
    def device_id(self) -> str:
        return self.device.device_id

    @property
    def device_name(self) -> str:
        return self.device.device_name

    @property
    def device_type(self) -> str:
        return self.device.device_type

    @property
    def protocol_version(self) -> int:
        return self.device.protocol_version

    @property
    def incoming_capabilities(self) -> List[str]:
        return self.device.incoming_capabilities

    @property
    def outgoing_capabilities(self) -> List[str]:
        return self.device.outgoing_capabilities

    @property
    def udp_port(self) -> int:
        return self.network.udp_port

    @property
    def tcp_port_min(self) -> int:
        return self.network.tcp_port_min

    @property
    def tcp_port_max(self) -> int:
        return self.network.tcp_port_max

    @property
    def stream_line_limit(self) -> int:
        return self.network.stream_line_limit

    @property
    def broadcast_interval(self) -> float:
        return self.network.broadcast_interval

    def ensure_directories(self) -> None:
        """Create configuration, data, and runtime directories with strict POSIX permissions.

        Applies mode 0700 (rwx------) to ensure user isolation in multi-user systems.
        """
        for directory in [self.storage.config_dir, self.storage.data_dir, self.storage.runtime_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(directory, 0o700)
            except OSError:
                pass

    def save(self, target_path: Optional[Path] = None) -> Path:
        """Save configuration atomically to disk in JSON format.

        Uses temporary file write followed by atomic rename to prevent corrupted files.
        """
        self.ensure_directories()
        destination = target_path or self.storage.config_file_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_destination = destination.with_suffix(".tmp")

        payload = self.model_dump(mode="json")
        serialized = json.dumps(payload, indent=2)

        try:
            tmp_destination.write_text(serialized, encoding="utf-8")
            try:
                os.chmod(tmp_destination, 0o600)
            except OSError:
                pass
            tmp_destination.replace(destination)
            try:
                os.chmod(destination, 0o600)
            except OSError:
                pass
        except OSError as exc:
            if tmp_destination.exists():
                try:
                    tmp_destination.unlink()
                except OSError:
                    pass
            raise ConfigValidationError(
                f"Failed to persist configuration to '{destination}': {exc}",
                context={"destination": str(destination)},
            ) from exc

        return destination

    @classmethod
    def load(
        cls,
        config_path: Optional[Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> PenguLinkConfig:
        """Load configuration through the hierarchical resolution chain:

        1. Default model values.
        2. Merged with JSON config file on disk (if present).
        3. Overridden by PENGULINK_* environment variables.
        4. Overridden by explicit dictionary arguments.
        """
        data: Dict[str, Any] = {}

        # 1. Determine config file location
        env_config = os.environ.get("PENGULINK_CONFIG_FILE")
        resolved_path = config_path or (Path(env_config) if env_config else None)
        if resolved_path is None:
            resolved_path = default_config_dir() / "config.json"

        # 2. Ingest JSON config file if it exists
        if resolved_path.is_file():
            try:
                content = resolved_path.read_text(encoding="utf-8")
                data = json.loads(content)
            except Exception as exc:
                raise ConfigValidationError(
                    f"Failed to read configuration file from '{resolved_path}': {exc}",
                    context={"path": str(resolved_path)},
                ) from exc

        # 3. Ingest PENGULINK_* environment variables
        env_map = cls._extract_env_overrides()
        cls._deep_merge(data, env_map)

        # 4. Ingest explicit overrides
        if overrides:
            cls._deep_merge(data, overrides)

        try:
            instance = cls.model_validate(data)
        except Exception as exc:
            raise ConfigValidationError(
                f"Configuration validation failed: {exc}",
                context={"raw_data": data},
            ) from exc

        return instance

    @classmethod
    def _extract_env_overrides(cls) -> Dict[str, Any]:
        """Extract and map supported PENGULINK_* environment variables into nested dict."""
        overrides: Dict[str, Any] = {}

        mapping = {
            "PENGULINK_DEVICE_ID": ("device", "device_id"),
            "PENGULINK_DEVICE_NAME": ("device", "device_name"),
            "PENGULINK_DEVICE_TYPE": ("device", "device_type"),
            "PENGULINK_PROTOCOL_VERSION": ("device", "protocol_version", int),
            "PENGULINK_UDP_PORT": ("network", "udp_port", int),
            "PENGULINK_TCP_PORT_MIN": ("network", "tcp_port_min", int),
            "PENGULINK_TCP_PORT_MAX": ("network", "tcp_port_max", int),
            "PENGULINK_BROADCAST_INTERVAL": ("network", "broadcast_interval", float),
            "PENGULINK_STREAM_LIMIT": ("network", "stream_line_limit", int),
            "PENGULINK_CONFIG_DIR": ("storage", "config_dir", Path),
            "PENGULINK_DATA_DIR": ("storage", "data_dir", Path),
            "PENGULINK_RUNTIME_DIR": ("storage", "runtime_dir", Path),
            "PENGULINK_LOG_LEVEL": ("logging", "level"),
            "PENGULINK_LOG_FILE": ("logging", "log_file", Path),
        }

        for env_var, target in mapping.items():
            val = os.environ.get(env_var)
            if val is not None and val != "":
                section, key = target[0], target[1]
                type_conv = target[2] if len(target) > 2 else str
                try:
                    converted = type_conv(val)
                except (ValueError, TypeError):
                    continue

                if section not in overrides:
                    overrides[section] = {}
                overrides[section][key] = converted

        return overrides

    @staticmethod
    def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively merge dictionary update into base dictionary in-place."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                PenguLinkConfig._deep_merge(base[key], value)
            else:
                base[key] = value


# Alias for PenguLinkConfig
Config = PenguLinkConfig


__all__ = [
    "DEVICE_ID_REGEX",
    "INVALID_NAME_CHARS_REGEX",
    "sanitize_device_name",
    "default_device_id",
    "default_device_name",
    "default_config_dir",
    "default_data_dir",
    "default_runtime_dir",
    "DeviceConfig",
    "NetworkConfig",
    "StorageConfig",
    "LoggingConfig",
    "PenguLinkConfig",
    "Config",
]

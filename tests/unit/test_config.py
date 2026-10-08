"""Unit tests for PenguLink configuration management."""

import os
from pathlib import Path
import pytest
from pydantic import ValidationError

from pengulink.core.config import (
    Config,
    DeviceConfig,
    NetworkConfig,
    PenguLinkConfig,
    StorageConfig,
    sanitize_device_name,
)
from pengulink.core.exceptions import ConfigValidationError


def test_default_config() -> None:
    """Assert default model values adhere to KDE Connect specifications."""
    cfg = PenguLinkConfig()
    assert len(cfg.device.device_id) == 32
    assert len(cfg.device.device_name) > 0
    assert cfg.device.device_type == "desktop"
    assert cfg.device.protocol_version == 7
    assert cfg.network.udp_port == 1716
    assert cfg.network.tcp_port_min == 1716
    assert cfg.network.tcp_port_max == 1764
    assert cfg.network.stream_line_limit == 131072
    assert cfg.logging.level == "INFO"

    # Test delegating properties
    assert cfg.device_id == cfg.device.device_id
    assert cfg.device_name == cfg.device.device_name
    assert cfg.device_type == cfg.device.device_type
    assert cfg.protocol_version == cfg.device.protocol_version
    assert cfg.udp_port == cfg.network.udp_port
    assert cfg.tcp_port_min == cfg.network.tcp_port_min
    assert cfg.tcp_port_max == cfg.network.tcp_port_max
    assert cfg.stream_line_limit == cfg.network.stream_line_limit


def test_device_name_sanitization() -> None:
    """Assert invalid punctuation is stripped and names are clamped."""
    assert sanitize_device_name('John\'s "Phone" [Linux]!') == "Johns Phone Linux"
    assert sanitize_device_name("   ") == "PenguLink-Linux"
    assert sanitize_device_name("a" * 50) == "a" * 32


def test_device_id_validation() -> None:
    """Assert device ID regex validation rules."""
    # Valid 32-char hex
    dev = DeviceConfig(device_id="a" * 32)
    assert dev.device_id == "a" * 32

    # Valid 38-char with hyphens and underscores
    dev2 = DeviceConfig(device_id="a" * 30 + "_-123456")
    assert len(dev2.device_id) == 38

    # Invalid: too short (< 32)
    with pytest.raises(ValidationError):
        DeviceConfig(device_id="short")

    # Invalid: illegal characters
    with pytest.raises(ValidationError):
        DeviceConfig(device_id="a" * 31 + "@")


def test_port_range_validation() -> None:
    """Assert tcp_port_min cannot exceed tcp_port_max."""
    # Valid
    net = NetworkConfig(tcp_port_min=1716, tcp_port_max=1764)
    assert net.tcp_port_min <= net.tcp_port_max

    # Invalid
    with pytest.raises(ValidationError):
        NetworkConfig(tcp_port_min=1770, tcp_port_max=1760)


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Assert environment variables override defaults."""
    monkeypatch.setenv("PENGULINK_DEVICE_NAME", "Rig99")
    monkeypatch.setenv("PENGULINK_UDP_PORT", "1818")
    monkeypatch.setenv("PENGULINK_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("PENGULINK_CONFIG_FILE", str(tmp_path / "nonexistent.json"))

    cfg = PenguLinkConfig.load()
    assert cfg.device.device_name == "Rig99"
    assert cfg.network.udp_port == 1818
    assert cfg.logging.level == "DEBUG"


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Assert saving configuration to disk and reloading it preserves values."""
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / "config"
    cfg.storage.data_dir = tmp_path / "data"
    cfg.storage.runtime_dir = tmp_path / "run"
    cfg.device.device_name = "Roundtrip-Device"
    cfg.network.udp_port = 1999

    save_path = tmp_path / "config" / "test_config.json"
    cfg.save(target_path=save_path)

    assert save_path.is_file()

    loaded = PenguLinkConfig.load(config_path=save_path)
    assert loaded.device.device_name == "Roundtrip-Device"
    assert loaded.network.udp_port == 1999
    assert loaded.device.device_id == cfg.device.device_id


def test_atomic_save_permissions(tmp_path: Path) -> None:
    """Assert directories are created with 0700 and config files with 0600."""
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / "secure_config"
    cfg.storage.data_dir = tmp_path / "secure_data"
    cfg.storage.runtime_dir = tmp_path / "secure_run"

    cfg_file = cfg.storage.config_dir / "config.json"
    cfg.save(target_path=cfg_file)

    # Check directory permissions (on POSIX systems)
    if os.name == "posix":
        dir_stat = os.stat(cfg.storage.config_dir).st_mode & 0o777
        assert dir_stat == 0o700

        file_stat = os.stat(cfg_file).st_mode & 0o777
        assert file_stat == 0o600


def test_load_corrupted_config(tmp_path: Path) -> None:
    """Assert loading invalid JSON raises ConfigValidationError."""
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ConfigValidationError):
        PenguLinkConfig.load(config_path=corrupt_file)

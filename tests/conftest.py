"""Pytest configuration and async test runner hook.

Provides execution support for coroutine tests when pytest-asyncio is not installed,
and common fixtures for PenguLink testing.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, Dict

import pytest

from pengulink.core.config import PenguLinkConfig


def pytest_pyfunc_call(pyfuncitem: Any) -> Any:
    """Execute coroutine test functions with asyncio.run when pytest-asyncio is absent."""
    testfunction = pyfuncitem.obj
    if inspect.iscoroutinefunction(testfunction):
        argnames = pyfuncitem._fixtureinfo.argnames
        funcargs = {arg: pyfuncitem.funcargs[arg] for arg in argnames}
        asyncio.run(testfunction(**funcargs))
        return True
    return None


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for configuration tests."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir


@pytest.fixture
def sample_config(tmp_path: Path) -> PenguLinkConfig:
    """Fixture providing a standard test configuration."""
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / "config"
    cfg.storage.data_dir = tmp_path / "data"
    cfg.storage.runtime_dir = tmp_path / "run"
    cfg.device.device_id = "0123456789abcdef0123456789abcdef"
    cfg.device.device_name = "Test-Device"
    cfg.network.udp_port = 1716
    cfg.network.tcp_port_min = 1716
    cfg.network.tcp_port_max = 1720
    return cfg

"""tests.e2e.conftest
~~~~~~~~~~~~~~~~~~~
Shared fixtures, ephemeral port helpers, and asynchronous test harnesses
for the Milestone 7 End-to-End integration test suite.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Optional, Tuple

import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.daemon.ipc import PenguLinkIPCClient
from pengulink.daemon.service import PenguLinkDaemon
from tests.mock.mock_device import ChaosConfig, MockDevice, MockDeviceConfig


def get_free_ports(count: int = 2) -> list[int]:
    """Find available TCP ports on loopback interface."""
    sockets = []
    ports = []
    for _ in range(count):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        ports.append(s.getsockname()[1])
        sockets.append(s)
    for s in sockets:
        s.close()
    return ports


def get_free_udp_port() -> int:
    """Find an available UDP port on loopback interface."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def wait_for(
    predicate: Callable[[], Any],
    timeout: float = 5.0,
    interval: float = 0.05,
    desc: str = "condition",
) -> Any:
    """Asynchronously poll a predicate function until it returns a truthy value or times out."""
    loop = asyncio.get_running_loop()
    start = loop.time()
    while loop.time() - start < timeout:
        result = predicate()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return result
        await asyncio.sleep(interval)
    raise TimeoutError(f"Timed out after {timeout:.2f}s waiting for {desc}")


@contextlib.asynccontextmanager
async def e2e_environment(
    tmp_path: Path,
    mock_config: Optional[MockDeviceConfig] = None,
    mock_chaos: Optional[ChaosConfig] = None,
    auto_pair: bool = True,
    auto_pong: bool = True,
    auto_battery_reply: bool = True,
    auto_connectivity_reply: bool = True,
    daemon_config: Optional[PenguLinkConfig] = None,
    start_daemon: bool = True,
    start_mock: bool = True,
) -> AsyncGenerator[Tuple[PenguLinkDaemon, MockDevice, PenguLinkIPCClient], None]:
    """Asynchronous context manager provisioning isolated daemon, mock device, and IPC client.

    All storage directories are sandboxed within tmp_path with POSIX 0600 permissions,
    using ephemeral ports and loopback (127.0.0.1).
    """
    tcp_ports = get_free_ports(count=2)
    tcp_base = tcp_ports[0]
    udp_port = get_free_udp_port()

    if daemon_config is not None:
        cfg = daemon_config
    else:
        cfg = PenguLinkConfig()
        cfg.storage.config_dir = tmp_path / "config"
        cfg.storage.data_dir = tmp_path / "data"
        cfg.storage.runtime_dir = tmp_path / "run"
        cfg.device.device_id = f"daemon_{uuid.uuid4().hex[:16]}"
        cfg.device.device_name = "PenguLink-Test-Daemon"
        cfg.network.tcp_port_min = tcp_base
        cfg.network.tcp_port_max = tcp_base + 5
        cfg.network.udp_port = udp_port

    cfg.ensure_directories()
    sock_path = cfg.storage.runtime_dir / "daemon.sock"

    daemon = PenguLinkDaemon(config=cfg, socket_path=sock_path)
    if start_daemon:
        await daemon.start()

    m_cfg = mock_config or MockDeviceConfig(
        device_id=f"mock_{uuid.uuid4().hex[:16]}",
        device_name="Mock Android Phone",
        auto_pair=auto_pair,
        auto_pong=auto_pong,
        auto_battery_reply=auto_battery_reply,
        auto_connectivity_reply=auto_connectivity_reply,
    )
    mock = MockDevice(config=m_cfg, chaos_config=mock_chaos)
    if start_mock:
        await mock.start(listen_tcp=True, port=0)

    ipc_client = PenguLinkIPCClient(socket_path=sock_path)

    try:
        yield daemon, mock, ipc_client
    finally:
        try:
            if mock.is_running:
                await mock.stop()
        except Exception:
            pass
        try:
            if daemon.is_running:
                await daemon.stop()
        except Exception:
            pass
        # Yield to allow close callbacks and pending event bus tasks to drain
        await asyncio.sleep(0.05)


@pytest.fixture
def free_ports() -> list[int]:
    """Synchronous pytest fixture returning free loopback ports."""
    return get_free_ports(4)

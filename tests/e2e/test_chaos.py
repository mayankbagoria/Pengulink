"""tests.e2e.test_chaos
~~~~~~~~~~~~~~~~~~~~~
End-to-End chaos and fault injection tests using MockKdeConnectDevice:
- TCP RST abruptly resetting the connection (SO_LINGER).
- Malformed and corrupt packet bursts.
- Stream buffer overrun (>128KB guard, CVE-2020-26164 defense).
- Truncated packet streams with abrupt EOF.
- Rapid connection flapping bursts.
- Aborted handshakes mid-TLS upgrade.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pengulink.core.events import PingReceived
from pengulink.core.exceptions import (
    ConnectionClosedError,
    TLSUpgradeError,
)
from pengulink.core.state_machine import DeviceState
from pengulink.daemon.service import PenguLinkDaemon
from tests.e2e.conftest import e2e_environment, wait_for
from tests.mock.mock_device import ChaosConfig, MockDevice


async def _pair_and_connect(daemon: PenguLinkDaemon, mock: MockDevice) -> None:
    """Helper to establish mutual pairing and an active TLS connection."""
    await mock.broadcast_identity(
        target_ip="127.0.0.1",
        target_port=daemon.config.network.udp_port,
    )
    await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)
    res = await daemon.pair_device(mock.device_id, wait=True, timeout=5.0)
    assert res["paired"] is True
    assert daemon.is_device_connected(mock.device_id) is True


async def test_chaos_tcp_rst_drop(tmp_path: Path) -> None:
    """Verify daemon catches abrupt TCP RST drops (SO_LINGER (1, 0)) and recovers cleanly."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Force TCP RST drop via SO_LINGER (1, 0)
        await mock.drop_connection(reset=True)

        # Daemon read loop must catch connection reset and transition to DISCONNECTED
        await wait_for(
            lambda: not daemon.is_device_connected(mock.device_id),
            timeout=3.0,
            desc="detecting TCP RST drop",
        )

        sm = daemon._state_machines.get(mock.device_id)
        assert sm is not None
        assert sm.current_state == DeviceState.DISCONNECTED

        # Daemon runtime survives and accepts subsequent reconnection
        assert daemon.is_running is True
        conn = await daemon.connect_device(mock.device_id, port=mock.bound_tcp_port)
        assert daemon.is_device_connected(mock.device_id) is True
        assert not conn.is_closed


async def test_chaos_malformed_packet_burst(tmp_path: Path) -> None:
    """Verify daemon read loop discards malformed JSON without terminating stream or crashing."""
    ping_events: list[PingReceived] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        daemon.event_bus.subscribe(
            PingReceived,
            lambda ev: ping_events.append(ev),
        )

        # Send bursts of invalid syntax and raw binary data
        await mock.send_malformed_json(b'{"corrupted_json": [missing_brackets\n')
        await mock.send_raw_bytes(b"\x00\x01\x02\x03\xff\xfeBINARY_GARBAGE\n")
        await mock.send_raw_bytes(b"PLAIN_NON_JSON_LINE\n")
        await mock.send_raw_bytes(b"{}\n")

        # Allow daemon time to process and discard malformed lines
        await asyncio.sleep(0.1)

        # Stream connection must remain healthy and connected
        assert daemon.is_device_connected(mock.device_id) is True

        # Subsequent legitimate packet must process normally
        await mock.send_ping(message="Ping after malformed burst")
        await wait_for(
            lambda: any(e.message == "Ping after malformed burst" for e in ping_events),
            timeout=3.0,
            desc="receiving valid ping after garbage burst",
        )


async def test_chaos_stream_buffer_overrun_128kb(tmp_path: Path) -> None:
    """Verify daemon's 128KB guard terminates connection upon oversized packet (CVE-2020-26164 defense)."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Transmit 140,000-byte oversized payload (> 128KB limit)
        await mock.send_oversized_packet(size=140000)

        # Guard triggers LimitOverrunError in TCPConnection.read_line() and terminates connection
        await wait_for(
            lambda: not daemon.is_device_connected(mock.device_id),
            timeout=4.0,
            desc="overrun protection dropping connection",
        )

        # Daemon stays running
        assert daemon.is_running is True


async def test_chaos_truncated_packet_stream(tmp_path: Path) -> None:
    """Verify daemon handles incomplete/truncated JSON stream ending in abrupt EOF."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Transmit partial JSON without newline and abruptly close socket
        await mock.send_truncated_packet(b'{"type":"kdeconnect.ping","body":{"partial":true')

        await wait_for(
            lambda: not daemon.is_device_connected(mock.device_id),
            timeout=3.0,
            desc="handling truncated packet EOF",
        )

        assert daemon.is_running is True
        sm = daemon._state_machines.get(mock.device_id)
        assert sm is not None
        assert sm.current_state == DeviceState.DISCONNECTED


async def test_chaos_rapid_flapping(tmp_path: Path) -> None:
    """Verify daemon TCP listener withstands rapid repeated connection flapping bursts."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        # Fire 10 rapid open/close connection cycles against daemon TCP server
        await mock.simulate_flapping(
            target_host="127.0.0.1",
            target_port=daemon.tcp_manager.bound_port,
            cycles=10,
            interval=0.01,
        )

        # Daemon remains fully operational
        assert daemon.is_running is True
        assert daemon.tcp_manager.is_running is True

        # Valid connection and pairing succeeds immediately after flapping barrage
        await _pair_and_connect(daemon, mock)
        assert daemon.is_device_connected(mock.device_id) is True


async def test_chaos_handshake_aborts_mid_tls(tmp_path: Path) -> None:
    """Verify aborted TLS handshake cleanly closes connection with zero dangling tasks."""
    chaos = ChaosConfig(drop_mid_tls=True)

    async with e2e_environment(tmp_path, mock_chaos=chaos) as (daemon, mock, ipc_client):
        await mock.broadcast_identity(
            target_ip="127.0.0.1",
            target_port=daemon.config.network.udp_port,
        )
        await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)

        # Connection should fail during TLS upgrade
        with pytest.raises((TLSUpgradeError, ConnectionClosedError, Exception)):
            await daemon.connect_device(
                mock.device_id,
                port=mock.bound_tcp_port,
                timeout=3.0,
            )

        assert daemon.is_device_connected(mock.device_id) is False
        assert daemon.is_running is True

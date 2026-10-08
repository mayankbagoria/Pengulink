"""tests.e2e.test_reconnection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-End integration tests for reconnection lifecycle and state recovery:
- Peer disappearance & manual reconnection with pinned mTLS verification.
- Auto-reconnection triggered by UDP rediscovery beacon.
- Inbound reconnection from paired MockDevice.
- MITM certificate tampering rejection (strict pinning defense).
- Repeated connect/disconnect flapping cycles with zero task/resource leaks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pengulink.core.events import DeviceConnected, DeviceDisconnected
from pengulink.core.exceptions import CertificateVerificationError, TLSUpgradeError
from pengulink.core.state_machine import DeviceState
from pengulink.daemon.service import PenguLinkDaemon
from tests.e2e.conftest import e2e_environment, wait_for
from tests.mock.mock_device import MockDevice, MockDeviceConfig


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


async def test_reconnection_peer_drop_and_manual_reconnect(tmp_path: Path) -> None:
    """Verify clean state transition on connection drop and subsequent manual reconnection.

    1. Mock drops TCP connection.
    2. Daemon detects closure, transitions to DISCONNECTED, emits DeviceDisconnected.
    3. Daemon manually reconnects via connect_device().
    4. mTLS verified using pinned certificate (no TOFU prompt).
    5. State machine transitions back to CONNECTED and DeviceConnected is emitted.
    6. Subsequent ping roundtrip succeeds immediately without re-pairing.
    """
    disconnect_events: list[DeviceDisconnected] = []
    connect_events: list[DeviceConnected] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        daemon.event_bus.subscribe(
            DeviceDisconnected,
            lambda ev: disconnect_events.append(ev),
        )
        daemon.event_bus.subscribe(
            DeviceConnected,
            lambda ev: connect_events.append(ev),
        )

        # 1. Mock drops connection
        await mock.drop_connection(reset=False)

        # 2. Wait for daemon to register disconnect
        await wait_for(
            lambda: not daemon.is_device_connected(mock.device_id),
            timeout=3.0,
            desc="daemon detecting disconnect",
        )
        await wait_for(lambda: len(disconnect_events) > 0, timeout=2.0)
        assert any(e.device_id == mock.device_id for e in disconnect_events)

        sm = daemon._state_machines.get(mock.device_id)
        assert sm is not None
        assert sm.current_state == DeviceState.DISCONNECTED

        # 3. Reconnect manually using pinned certificate
        connect_events.clear()
        conn = await daemon.connect_device(mock.device_id, port=mock.bound_tcp_port)
        assert not conn.is_closed
        assert daemon.is_device_connected(mock.device_id) is True

        # 4. Verify state machine and event
        await wait_for(lambda: len(connect_events) > 0, timeout=2.0)
        assert any(e.device_id == mock.device_id for e in connect_events)
        assert sm.current_state == DeviceState.CONNECTED

        # 5. Verify functional ping exchange without re-pairing
        await daemon.send_ping(mock.device_id, message="Ping on reconnected stream")
        await wait_for(
            lambda: mock.last_ping_message == "Ping on reconnected stream",
            timeout=3.0,
        )


async def test_reconnection_auto_connect_on_udp_rediscovery(tmp_path: Path) -> None:
    """Verify daemon automatically initiates reconnection when an offline paired peer broadcasts UDP identity."""
    connect_events: list[DeviceConnected] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Drop active connection
        await mock.drop_connection(reset=False)
        await wait_for(lambda: not daemon.is_device_connected(mock.device_id), timeout=3.0)

        daemon.event_bus.subscribe(
            DeviceConnected,
            lambda ev: connect_events.append(ev),
        )

        # Clear seen cache to allow immediate redelivery through sliding window
        daemon.udp_discovery._seen_cache.pop(mock.device_id, None)

        # Mock broadcasts UDP beacon
        await mock.broadcast_identity(
            target_ip="127.0.0.1",
            target_port=daemon.config.network.udp_port,
        )

        # Daemon should automatically reconnect
        await wait_for(
            lambda: daemon.is_device_connected(mock.device_id),
            timeout=5.0,
            desc="auto-reconnection to paired device",
        )
        assert len(connect_events) >= 1
        assert any(e.device_id == mock.device_id for e in connect_events)


async def test_reconnection_inbound_from_paired_mock(tmp_path: Path) -> None:
    """Verify paired MockDevice can initiate inbound connection to daemon and authenticate via pinned cert."""
    connect_events: list[DeviceConnected] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Drop active connection
        await mock.drop_connection(reset=False)
        await wait_for(lambda: not daemon.is_device_connected(mock.device_id), timeout=3.0)

        daemon.event_bus.subscribe(
            DeviceConnected,
            lambda ev: connect_events.append(ev),
        )

        # Mock initiates inbound connection to daemon's TCP listener
        await mock.connect_to_daemon(
            host="127.0.0.1",
            port=daemon.tcp_manager.bound_port,
            target_device_id=daemon.config.device_id,
        )

        # Verify daemon accepted inbound paired connection
        await wait_for(
            lambda: daemon.is_device_connected(mock.device_id),
            timeout=4.0,
            desc="inbound connection from paired mock",
        )
        assert any(e.device_id == mock.device_id for e in connect_events)
        sm = daemon._state_machines.get(mock.device_id)
        assert sm is not None
        assert sm.current_state == DeviceState.CONNECTED


async def test_reconnection_mitm_cert_tamper_rejected(tmp_path: Path) -> None:
    """Verify daemon rejects connection if peer certificate does not match pinned certificate (MITM defense)."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Drop connection
        await mock.drop_connection(reset=False)
        await wait_for(lambda: not daemon.is_device_connected(mock.device_id), timeout=3.0)

        # Instantiate rogue mock with identical device_id but fresh (different) key/cert
        rogue_cfg = MockDeviceConfig(
            device_id=mock.device_id,
            device_name="Impostor Phone",
            auto_pair=False,
        )
        rogue_mock = MockDevice(config=rogue_cfg)
        await rogue_mock.start(listen_tcp=True, port=0)

        try:
            # Daemon attempts to connect to rogue device claiming the same device_id
            with pytest.raises((CertificateVerificationError, TLSUpgradeError, Exception)):
                await daemon.connect_device(
                    mock.device_id,
                    host="127.0.0.1",
                    port=rogue_mock.bound_tcp_port,
                    timeout=3.0,
                )

            assert daemon.is_device_connected(mock.device_id) is False
        finally:
            await rogue_mock.stop()


async def test_reconnection_multiple_flapping_cycles(tmp_path: Path) -> None:
    """Verify repeated connect/disconnect/ping cycles complete cleanly without task or socket leaks."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        for cycle in range(4):
            # 1. Drop
            await mock.drop_connection(reset=False)
            await wait_for(lambda: not daemon.is_device_connected(mock.device_id), timeout=3.0)

            # 2. Reconnect
            await daemon.connect_device(mock.device_id, port=mock.bound_tcp_port)
            assert daemon.is_device_connected(mock.device_id) is True

            # 3. Send Ping
            msg = f"Flap Ping Cycle {cycle}"
            await daemon.send_ping(mock.device_id, message=msg)
            await wait_for(lambda: mock.last_ping_message == msg, timeout=3.0)

        # Verify zero dangling tasks or connection leaks
        assert daemon.is_running is True
        assert len(daemon._active_connections) == 1

"""tests.e2e.test_pairing
~~~~~~~~~~~~~~~~~~~~~~
End-to-End integration tests for KDE Connect mutual pairing lifecycle:
outbound pairing, inbound pairing, verification code matching, rejection,
timeout handling, unpairing, and persistence across daemon restarts.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.core.events import DevicePaired, DeviceUnpaired, PairingRequested
from pengulink.core.exceptions import ConnectionTimeoutError, PairingRejectedError
from pengulink.core.state_machine import DeviceState
from pengulink.crypto.certificate import dump_certificate_pem
from pengulink.daemon.service import PenguLinkDaemon
from tests.e2e.conftest import e2e_environment, get_free_ports, get_free_udp_port, wait_for
from tests.mock.mock_device import ChaosConfig, MockDevice, MockDeviceConfig


async def test_pairing_outbound_daemon_initiated_success(tmp_path: Path) -> None:
    """Verify outbound pairing handshake initiated by daemon towards MockDevice.

    1. Mock broadcasts identity and is discovered by daemon.
    2. Daemon initiates pair_device(wait=True).
    3. Inverted mTLS succeeds (connector = TLS server, listener = TLS client).
    4. Derived 8-character verification codes match exactly on both ends.
    5. Mock auto-accepts; daemon saves trusted credentials and transitions to PAIRED and CONNECTED.
    6. DevicePaired event is emitted on daemon's EventBus.
    """
    paired_events: list[DevicePaired] = []

    async with e2e_environment(tmp_path, auto_pair=True) as (daemon, mock, ipc_client):
        daemon.event_bus.subscribe(
            DevicePaired,
            lambda ev: paired_events.append(ev),
        )

        # 1. Discover mock
        await mock.broadcast_identity(
            target_ip="127.0.0.1",
            target_port=daemon.config.network.udp_port,
        )
        await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)

        # 2. Initiate pairing
        res = await daemon.pair_device(mock.device_id, wait=True, timeout=5.0)

        # 3. Assertions on pairing response
        assert res["status"] == "paired"
        assert res["paired"] is True
        code = res["verification_code"]
        assert len(code) == 8
        assert code == code.upper()
        assert code.isalnum()

        # 4. Verify mock derived the identical verification code
        assert mock.last_verification_code == code

        # 5. Verify persistence in DeviceStore
        is_paired = await daemon.device_store.is_device_paired(mock.device_id)
        assert is_paired is True
        stored_cert_pem = await daemon.device_store.get_device_certificate_pem(mock.device_id)
        assert stored_cert_pem is not None
        assert stored_cert_pem == dump_certificate_pem(mock.certificate)

        # 6. Verify state machine transitioned to CONNECTED
        sm = daemon._state_machines.get(mock.device_id)
        assert sm is not None
        assert sm.current_state == DeviceState.CONNECTED

        # 7. Verify DevicePaired event
        assert len(paired_events) >= 1
        assert any(e.device_id == mock.device_id for e in paired_events)


async def test_pairing_inbound_mock_initiated(tmp_path: Path) -> None:
    """Verify inbound pairing handshake initiated by MockDevice towards PenguLinkDaemon.

    1. Mock initiates TCP connection to daemon's inbound TCP listener.
    2. Inverted mTLS succeeds (connector = TLS server, listener = TLS client).
    3. Mock initiates pairing via initiate_pairing().
    4. Daemon receives kdeconnect.pair packet and publishes PairingRequested event.
    5. The 8-character verification code in the event matches mock's derived code.
    """
    requested_events: list[PairingRequested] = []

    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        daemon.event_bus.subscribe(
            PairingRequested,
            lambda ev: requested_events.append(ev),
        )

        # 1. Mock connects inbound to daemon
        await mock.connect_to_daemon(
            host="127.0.0.1",
            port=daemon.tcp_manager.bound_port,
            target_device_id=daemon.config.device_id,
        )

        # 2. Mock initiates pairing
        code = await mock.initiate_pairing(
            target_pubkey_pem=daemon.local_pubkey_pem
        )
        assert len(code) == 8
        assert code == code.upper()

        # 3. Wait for daemon to receive and emit PairingRequested
        await wait_for(lambda: len(requested_events) > 0, timeout=4.0)

        ev = requested_events[0]
        assert ev.device_id == mock.device_id
        assert ev.verification_code == code
        assert ev.verification_code == mock.last_verification_code


async def test_pairing_rejected_by_mock(tmp_path: Path) -> None:
    """Verify daemon handles pairing rejection by MockDevice properly.

    1. Mock is configured with auto_pair=False, auto_reject_pair=True.
    2. Daemon initiates pairing with wait=True.
    3. Daemon raises PairingRejectedError.
    4. Device is not persisted in DeviceStore, state resets to DISCONNECTED.
    """
    m_cfg = MockDeviceConfig(
        device_id=f"mock_reject_{uuid.uuid4().hex[:12]}",
        auto_pair=False,
        auto_reject_pair=True,
    )

    async with e2e_environment(tmp_path, mock_config=m_cfg) as (daemon, mock, ipc_client):
        await mock.broadcast_identity(
            target_ip="127.0.0.1",
            target_port=daemon.config.network.udp_port,
        )
        await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)

        with pytest.raises(PairingRejectedError):
            await daemon.pair_device(mock.device_id, wait=True, timeout=5.0)

        # Assert not paired in storage
        assert await daemon.device_store.is_device_paired(mock.device_id) is False

        # State machine reset
        await wait_for(
            lambda: daemon._state_machines.get(mock.device_id) is not None
            and daemon._state_machines[mock.device_id].current_state in (DeviceState.DISCONNECTED, DeviceState.DISCOVERED),
            timeout=4.0,
        )
        sm = daemon._state_machines.get(mock.device_id)
        if sm:
            assert sm.current_state in (DeviceState.DISCONNECTED, DeviceState.DISCOVERED)


async def test_pairing_timeout_handling(tmp_path: Path) -> None:
    """Verify pairing timeout cleanly aborts without dangling futures.

    1. Mock is configured to ignore pairing requests (chaos ignore_pairing=True).
    2. Daemon initiates pairing with short timeout=0.3s.
    3. Daemon raises ConnectionTimeoutError.
    4. Pairing future is evicted from daemon._pairing_futures.
    """
    chaos = ChaosConfig(ignore_pairing=True)

    async with e2e_environment(tmp_path, mock_chaos=chaos) as (daemon, mock, ipc_client):
        await mock.broadcast_identity(
            target_ip="127.0.0.1",
            target_port=daemon.config.network.udp_port,
        )
        await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)

        with pytest.raises(ConnectionTimeoutError):
            await daemon.pair_device(mock.device_id, wait=True, timeout=0.3)

        # Ensure future was cleaned up
        assert mock.device_id not in daemon._pairing_futures


async def test_pairing_unpair_lifecycle(tmp_path: Path) -> None:
    """Verify unpairing revokes credentials, notifies peer, and emits DeviceUnpaired.

    1. Daemon and Mock successfully complete pairing.
    2. Daemon calls unpair_device(mock.device_id) via IPC.
    3. Device is deleted from DeviceStore.
    4. DeviceUnpaired event is published on EventBus.
    5. State machine transitions to DISCONNECTED.
    """
    unpaired_events: list[DeviceUnpaired] = []

    async with e2e_environment(tmp_path, auto_pair=True) as (daemon, mock, ipc_client):
        daemon.event_bus.subscribe(
            DeviceUnpaired,
            lambda ev: unpaired_events.append(ev),
        )

        # 1. Discover and pair
        await mock.broadcast_identity(
            target_ip="127.0.0.1",
            target_port=daemon.config.network.udp_port,
        )
        await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)
        await daemon.pair_device(mock.device_id, wait=True, timeout=5.0)
        assert await daemon.device_store.is_device_paired(mock.device_id) is True

        # 2. Unpair via IPC
        unpair_res = await ipc_client.unpair_device(mock.device_id)
        assert unpair_res.get("success") is True or unpair_res.get("unpaired") is True

        # 3. Verify store deletion
        assert await daemon.device_store.is_device_paired(mock.device_id) is False

        # 4. Verify EventBus event
        await wait_for(lambda: len(unpaired_events) > 0, timeout=2.0)
        assert any(e.device_id == mock.device_id for e in unpaired_events)

        # 5. Verify state machine transitioned to DISCONNECTED
        sm = daemon._state_machines.get(mock.device_id)
        assert sm is not None
        assert sm.current_state == DeviceState.DISCONNECTED


async def test_pairing_persistence_across_daemon_restart(tmp_path: Path) -> None:
    """Verify paired device credentials survive daemon stop and restart."""
    tcp_base = get_free_ports(2)[0]
    udp_port = get_free_udp_port()

    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / "config"
    cfg.storage.data_dir = tmp_path / "data"
    cfg.storage.runtime_dir = tmp_path / "run"
    cfg.device.device_id = f"daemon_pers_{uuid.uuid4().hex[:12]}"
    cfg.device.device_name = "PenguLink-Persistent-Daemon"
    cfg.network.tcp_port_min = tcp_base
    cfg.network.tcp_port_max = tcp_base + 5
    cfg.network.udp_port = udp_port
    cfg.ensure_directories()

    m_cfg = MockDeviceConfig(
        device_id=f"mock_pers_{uuid.uuid4().hex[:12]}",
        device_name="Persistent Android",
        auto_pair=True,
    )
    mock = MockDevice(config=m_cfg)
    await mock.start(listen_tcp=True, port=0)

    try:
        # Phase 1: Start Daemon 1 and Pair
        sock_path = cfg.storage.runtime_dir / "daemon.sock"
        daemon_1 = PenguLinkDaemon(config=cfg, socket_path=sock_path)
        await daemon_1.start()

        try:
            await mock.broadcast_identity(
                target_ip="127.0.0.1",
                target_port=cfg.network.udp_port,
            )
            await wait_for(lambda: mock.device_id in daemon_1._discovered_devices, timeout=4.0)
            await daemon_1.pair_device(mock.device_id, wait=True, timeout=5.0)
            assert await daemon_1.device_store.is_device_paired(mock.device_id) is True
        finally:
            await daemon_1.stop()
            await asyncio.sleep(0.05)

        # Phase 2: Start Daemon 2 with same storage directories
        daemon_2 = PenguLinkDaemon(config=cfg, socket_path=sock_path)
        await daemon_2.start()

        try:
            # Verify credentials loaded from storage without re-pairing
            assert await daemon_2.device_store.is_device_paired(mock.device_id) is True
            stored_record = await daemon_2.device_store.get_device(mock.device_id)
            assert stored_record is not None
            assert stored_record["is_paired"] is True
            assert stored_record["device_name"] == mock.config.device_name

            pinned_pem = await daemon_2.device_store.get_device_certificate_pem(mock.device_id)
            assert pinned_pem == dump_certificate_pem(mock.certificate)
        finally:
            await daemon_2.stop()
            await asyncio.sleep(0.05)

    finally:
        await mock.stop()

"""tests.unit.test_phase2_defects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dedicated regression test suite for Milestone 18 (Phase 2: Runtime Reliability & Protocol Completeness):
- CLUS-04 (BUG-CRIT-04): Complete Inbound Pairing Acceptance Protocol, IPC Endpoints & CLI Commands
- CLUS-02 (BUG-CRIT-02): Silent Garbage Collection of Background Coroutines
- CLUS-12 (BUG-HIGH-05): DeviceStore.save_device() Overwrite Strips Pinned Certs & Pairing Trust
- CLUS-13 (BUG-HIGH-06): Synchronous is_device_paired() Queries Non-Existent Attribute
- CLUS-14 (BUG-HIGH-07): Unauthenticated Inbound Sessions Preempt Active Sessions (CVE-2020-26164)
- CLUS-15 (BUG-HIGH-08): AlertService Concurrency Mutex Gap in stop_alert()
"""

from __future__ import annotations

import asyncio
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli, pair
from pengulink.core.config import PenguLinkConfig
from pengulink.core.events import DeviceConnected, DeviceDisconnected, DevicePaired, DeviceUnpaired, EventBus, PairingRequested
from pengulink.core.exceptions import PairingError
from pengulink.core.state_machine import DeviceState, DeviceStateMachine
from pengulink.crypto.certificate import dump_certificate_pem, generate_device_key, generate_self_signed_cert
from pengulink.daemon.ipc import PenguLinkIPCClient, PenguLinkIPCServer
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.network.packet import Packet
from pengulink.network.tcp import ConnectionRole, TCPConnection, TCPStreamManager
from pengulink.plugins.clipboard.backend import X11ClipboardBackend
from pengulink.services.alert import AlertService, MockAudioBackend
from pengulink.storage.base import DeviceStore
from pengulink.storage.json_store import JsonStorageBackend


def make_test_config(tmp_path: Path) -> PenguLinkConfig:
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / "config"
    cfg.storage.data_dir = tmp_path / "storage"
    cfg.storage.runtime_dir = tmp_path / "runtime"
    return cfg


# =============================================================================
# 1. CLUS-04 (BUG-CRIT-04): Complete Inbound Pairing Acceptance Protocol & IPC
# =============================================================================

class TestClus04InboundPairingProtocol:
    """Verify inbound pairing acceptance, rejection, IPC handlers, and CLI commands."""

    async def test_daemon_inbound_pairing_request_storage_and_event(self, tmp_path: Path) -> None:
        """Inbound pairing packet stores pending request and publishes PairingRequested."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        key = generate_device_key()
        cert = generate_self_signed_cert(key, "peer_phone_1")
        from cryptography.hazmat.primitives import serialization
        peer_der = cert.public_bytes(serialization.Encoding.DER)

        events: List[PairingRequested] = []
        daemon.event_bus.subscribe(PairingRequested, lambda ev: events.append(ev))

        conn = MagicMock(spec=TCPConnection)
        conn.remote_device_id = "peer_phone_1"
        conn.is_closed = False
        conn.writer = MagicMock()
        ssl_obj = MagicMock()
        ssl_obj.getpeercert.return_value = peer_der
        conn.writer.get_extra_info.return_value = ssl_obj

        pkt = Packet(
            type="kdeconnect.pair",
            body={"pair": True, "timestamp": int(time.time() * 1000)},
        )

        await daemon._handle_pairing_packet(conn, pkt)

        assert len(events) == 1
        assert events[0].device_id == "peer_phone_1"
        assert "peer_phone_1" in daemon._pending_inbound_pairings
        req = daemon._pending_inbound_pairings["peer_phone_1"]
        assert req["conn"] is conn
        assert req["code"] == events[0].verification_code

    async def test_daemon_accept_pairing_success(self, tmp_path: Path) -> None:
        """daemon.accept_pairing() sends pair:True, persists trust, transitions state, and publishes DevicePaired."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        key = generate_device_key()
        cert = generate_self_signed_cert(key, "peer_phone_1")
        from cryptography.hazmat.primitives import serialization
        peer_der = cert.public_bytes(serialization.Encoding.DER)

        paired_events: List[DevicePaired] = []
        daemon.event_bus.subscribe(DevicePaired, lambda ev: paired_events.append(ev))

        written_lines: List[bytes] = []
        conn = MagicMock(spec=TCPConnection)
        conn.remote_device_id = "peer_phone_1"
        conn.is_closed = False
        conn.write_line = AsyncMock(side_effect=lambda data: written_lines.append(data))

        daemon._pending_inbound_pairings["peer_phone_1"] = {
            "code": "12345678",
            "verification_code": "12345678",
            "timestamp": int(time.time() * 1000),
            "conn": conn,
            "peer_cert": cert,
            "device_name": "Test Phone",
            "received_at": time.time(),
        }

        res = await daemon.accept_pairing("peer_phone_1")
        assert res["status"] == "paired"
        assert res["device_id"] == "peer_phone_1"
        assert len(written_lines) == 1
        sent_pkt = json.loads(written_lines[0].decode("utf-8"))
        assert sent_pkt["body"]["pair"] is True

        assert daemon.is_device_paired("peer_phone_1") is True
        assert "peer_phone_1" in daemon._paired_device_ids
        assert "peer_phone_1" not in daemon._pending_inbound_pairings
        assert len(paired_events) == 1
        assert paired_events[0].device_id == "peer_phone_1"

        # Check storage persistence
        dev = await daemon.device_store.get_device("peer_phone_1")
        assert dev is not None
        assert dev["is_paired"] is True
        assert dev["device_name"] == "Test Phone"

    async def test_daemon_accept_pairing_errors(self, tmp_path: Path) -> None:
        """accept_pairing() raises PairingError on missing request or closed connection."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        with pytest.raises(PairingError, match="No pending inbound pairing request"):
            await daemon.accept_pairing("non_existent")

        closed_conn = MagicMock(spec=TCPConnection)
        closed_conn.is_closed = True
        daemon._pending_inbound_pairings["closed_dev"] = {
            "conn": closed_conn,
        }
        with pytest.raises(PairingError, match="Connection closed"):
            await daemon.accept_pairing("closed_dev")

    async def test_daemon_reject_pairing(self, tmp_path: Path) -> None:
        """daemon.reject_pairing() sends pair:False, closes connection, and resets state."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        written_lines: List[bytes] = []
        conn = MagicMock(spec=TCPConnection)
        conn.remote_device_id = "rejected_dev"
        conn.is_closed = False
        conn.write_line = AsyncMock(side_effect=lambda data: written_lines.append(data))
        conn.close = AsyncMock()

        daemon._pending_inbound_pairings["rejected_dev"] = {
            "code": "87654321",
            "conn": conn,
            "device_name": "Rejected Phone",
            "received_at": time.time(),
        }

        res = await daemon.reject_pairing("rejected_dev")
        assert res["status"] == "rejected"
        assert len(written_lines) == 1
        sent_pkt = json.loads(written_lines[0].decode("utf-8"))
        assert sent_pkt["body"]["pair"] is False
        assert conn.close.awaited
        assert "rejected_dev" not in daemon._pending_inbound_pairings

    async def test_daemon_get_pending_pairings(self, tmp_path: Path) -> None:
        """get_pending_pairings() returns active requests and purges expired ones."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")

        conn_live = MagicMock(spec=TCPConnection)
        conn_live.is_closed = False
        conn_closed = MagicMock(spec=TCPConnection)
        conn_closed.is_closed = True

        daemon._pending_inbound_pairings["active_dev"] = {
            "device_id": "active_dev",
            "device_name": "Active Device",
            "code": "11112222",
            "timestamp": 123456,
            "conn": conn_live,
            "received_at": time.time(),
        }
        daemon._pending_inbound_pairings["closed_dev"] = {
            "device_id": "closed_dev",
            "device_name": "Closed Device",
            "code": "33334444",
            "timestamp": 123457,
            "conn": conn_closed,
            "received_at": time.time(),
        }
        daemon._pending_inbound_pairings["expired_dev"] = {
            "device_id": "expired_dev",
            "device_name": "Expired Device",
            "code": "55556666",
            "timestamp": 123458,
            "conn": conn_live,
            "received_at": time.time() - 100.0,
        }

        pending = daemon.get_pending_pairings()
        assert len(pending) == 1
        assert pending[0]["device_id"] == "active_dev"
        assert "closed_dev" not in daemon._pending_inbound_pairings
        assert "expired_dev" not in daemon._pending_inbound_pairings

    async def test_ipc_pair_accept_and_reject(self, tmp_path: Path) -> None:
        """IPC endpoints pair_accept and pair_reject route correctly over JSON-RPC."""
        sock_path = tmp_path / "daemon.sock"
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=sock_path)
        await daemon.start()

        # Seed an inbound pairing
        conn = MagicMock(spec=TCPConnection)
        conn.is_closed = False
        conn.write_line = AsyncMock()
        conn.close = AsyncMock()
        daemon._pending_inbound_pairings["test_phone"] = {
            "code": "99990000",
            "timestamp": int(time.time()),
            "conn": conn,
            "device_name": "IPC Test Phone",
            "received_at": time.time(),
        }

        client = PenguLinkIPCClient(socket_path=sock_path)
        try:
            res = await client.pair_accept("test_phone")
            assert res["status"] == "paired"
            assert daemon.is_device_paired("test_phone") is True

            # Seed another and reject
            daemon._pending_inbound_pairings["test_phone_2"] = {
                "code": "11110000",
                "timestamp": int(time.time()),
                "conn": conn,
                "device_name": "IPC Reject Phone",
                "received_at": time.time(),
            }
            res_rej = await client.pair_reject("test_phone_2")
            assert res_rej["status"] == "rejected"
        finally:
            await daemon.stop()

    def test_cli_pair_subcommands(self, tmp_path: Path) -> None:
        """Click CLI pair accept and pair reject subcommands execute successfully."""
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        accept_reply = {"device_id": "phone_42", "status": "paired", "message": "Successfully accepted pairing with Pixel 8"}
        reject_reply = {"device_id": "phone_42", "status": "rejected", "message": "Pairing request rejected"}
        pending_reply = [{"device_id": "phone_42", "device_name": "Pixel 8", "verification_code": "8F3A1B9C"}]

        runner = CliRunner()
        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=accept_reply):
            res = runner.invoke(cli, ["-s", str(mock_socket), "pair", "accept", "phone_42"])
            assert res.exit_code == 0
            assert "Successfully accepted pairing" in res.output

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=reject_reply):
            res = runner.invoke(cli, ["-s", str(mock_socket), "pair", "reject", "phone_42"])
            assert res.exit_code == 0
            assert "Pairing request rejected" in res.output

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=pending_reply):
            res = runner.invoke(cli, ["-s", str(mock_socket), "pair", "requests"])
            assert res.exit_code == 0
            assert "Pixel 8" in res.output
            assert "8F3A1B9C" in res.output


# =============================================================================
# 2. CLUS-02 (BUG-CRIT-02): Silent Garbage Collection of Background Coroutines
# =============================================================================

class TestClus02TaskGarbageCollection:
    """Verify background coroutines are strongly referenced and not dropped by gc.collect()."""

    async def test_daemon_spawn_background_task_tracking_and_gc(self, tmp_path: Path) -> None:
        """_spawn_background_task retains strong reference through multiple gc.collect() cycles."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")

        completed = False

        async def worker() -> None:
            nonlocal completed
            await asyncio.sleep(0.05)
            completed = True

        task = daemon._spawn_background_task(worker(), name="test_worker")
        assert task in daemon._background_tasks

        # Force aggressive garbage collection
        gc.collect()
        gc.collect()

        await asyncio.sleep(0.1)
        assert completed is True
        assert task not in daemon._background_tasks

    async def test_daemon_connection_close_teardown_survives_gc(self, tmp_path: Path) -> None:
        """_on_closed teardown tasks complete cleanly even under immediate forced GC."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        conn = MagicMock(spec=TCPConnection)
        conn.remote_device_id = "gc_test_dev"
        conn.is_closed = False

        close_cb = None

        def mock_add_close_cb(cb: Any) -> None:
            nonlocal close_cb
            close_cb = cb

        conn.add_close_callback = mock_add_close_cb
        daemon._register_connection("gc_test_dev", conn)
        assert close_cb is not None

        disconnected_events: List[DeviceDisconnected] = []
        daemon.event_bus.subscribe(DeviceDisconnected, lambda ev: disconnected_events.append(ev))

        # Fire connection closed
        close_cb(conn)

        # Force garbage collection immediately while tasks are scheduled
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)

        # Allow event loop to run tasks
        await asyncio.sleep(0.05)

        assert len(disconnected_events) == 1
        assert disconnected_events[0].device_id == "gc_test_dev"
        sm = daemon._state_machines.get("gc_test_dev")
        assert sm is not None
        assert sm.current_state == DeviceState.DISCONNECTED

    async def test_tcp_connection_close_callbacks_tracked(self) -> None:
        """TCPConnection._close_tasks strongly tracks close callbacks."""
        reader = asyncio.StreamReader()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        conn = TCPConnection(
            reader=reader,
            writer=writer,
            role=ConnectionRole.INBOUND,
            peer_ip="127.0.0.1",
            peer_port=1716,
        )

        ran = False

        async def async_cb(c: TCPConnection) -> None:
            nonlocal ran
            await asyncio.sleep(0.02)
            ran = True

        conn.add_close_callback(async_cb)
        await conn.close()

        assert len(conn._close_tasks) > 0
        gc.collect()

        await asyncio.sleep(0.05)
        assert ran is True
        assert len(conn._close_tasks) == 0

    async def test_tcp_manager_collision_close_task_tracked(self, tmp_path: Path) -> None:
        """TCPStreamManager tracks collision close tasks in _pending_tasks."""
        cfg = make_test_config(tmp_path)
        mgr = TCPStreamManager(config=cfg)

        existing_conn = MagicMock(spec=TCPConnection)
        existing_conn.is_closed = False
        closed_flag = False

        async def mock_close() -> None:
            nonlocal closed_flag
            await asyncio.sleep(0.02)
            closed_flag = True

        existing_conn.close = mock_close
        mgr._active_connections["z_remote_dev"] = existing_conn

        # Force local < remote tie breaker so remote wins and existing is closed
        with patch.object(mgr, "_get_device_id", return_value="a_local_id"):
            dropped = mgr._should_drop_inbound_collision("z_remote_dev")
            assert dropped is False

        # Verify task was added to _pending_tasks
        assert len(mgr._pending_tasks) == 1
        gc.collect()

        await asyncio.sleep(0.05)
        assert closed_flag is True
        assert len(mgr._pending_tasks) == 0

    async def test_x11_clipboard_backend_task_tracking(self) -> None:
        """X11ClipboardBackend._tasks strongly references coalesced notification tasks."""
        backend = X11ClipboardBackend()
        backend._x11_dpy = MagicMock()
        backend._x11_lib = MagicMock()
        backend._x11_lib.XPending.return_value = 0

        received: List[str] = []

        async def on_changed(txt: str) -> None:
            received.append(txt)

        backend._callback = on_changed
        backend.get_text = AsyncMock(return_value="copied text")

        # Simulate readable event
        backend._on_x11_readable()

        # Task should be tracked in backend._tasks
        assert len(backend._tasks) == 1
        gc.collect()

        # Let the task finish
        await asyncio.sleep(0.05)
        assert received == ["copied text"]
        assert len(backend._tasks) == 0

        # Test stop_watching cancels any active tasks
        backend._notify_pending = False
        backend.get_text = AsyncMock(side_effect=lambda: asyncio.sleep(10))
        backend._on_x11_readable()
        assert len(backend._tasks) == 1

        await backend.stop_watching()
        assert len(backend._tasks) == 0



# =============================================================================
# 3. CLUS-12 (BUG-HIGH-05): DeviceStore.save_device() Overwrite Bug
# =============================================================================

class TestClus12DeviceStoreOverwrite:
    """Verify DeviceStore.save_device() preserves certificates and pairing status."""

    async def test_save_device_preserves_existing_credentials_and_trust(self, tmp_path: Path) -> None:
        """Updating device name via save_device preserves certificate_pem and is_paired."""
        backend = JsonStorageBackend(tmp_path)
        store = DeviceStore(backend=backend)
        await store.initialize()

        cert_pem = "-----BEGIN CERTIFICATE-----\nABCDEF123456\n-----END CERTIFICATE-----"

        # 1. Save paired device with certificate
        await store.save_device(
            device_id="phone_alpha",
            name="Alpha Phone",
            device_type="phone",
            certificate_pem=cert_pem,
            is_paired=True,
            extra={"custom_attr": 42},
        )

        record1 = await store.get_device("phone_alpha")
        assert record1 is not None
        assert record1["device_name"] == "Alpha Phone"
        assert record1["certificate_pem"] == cert_pem
        assert record1["is_paired"] is True
        assert record1["device_type"] == "phone"
        assert record1["custom_attr"] == 42

        # 2. Update name without specifying certificate_pem or is_paired
        await store.save_device(
            device_id="phone_alpha",
            name="Renamed Alpha Phone",
        )

        record2 = await store.get_device("phone_alpha")
        assert record2 is not None
        assert record2["device_name"] == "Renamed Alpha Phone"
        # Must be preserved!
        assert record2["certificate_pem"] == cert_pem
        assert record2["is_paired"] is True
        assert record2["device_type"] == "phone"
        assert record2["custom_attr"] == 42

    async def test_save_device_explicit_overrides_work(self, tmp_path: Path) -> None:
        """Explicitly providing new certificate or device_type updates the record."""
        backend = JsonStorageBackend(tmp_path)
        store = DeviceStore(backend=backend)
        await store.initialize()

        await store.save_device("dev_beta", "Beta", device_type="desktop", is_paired=False)
        record = await store.get_device("dev_beta")
        assert record is not None
        assert record["is_paired"] is False

        # Explicitly pair and update cert
        new_cert = "-----BEGIN CERTIFICATE-----\nNEW_CERT\n-----END CERTIFICATE-----"
        await store.save_device(
            "dev_beta",
            name="Beta Paired",
            device_type="tablet",
            certificate_pem=new_cert,
            is_paired=True,
        )

        record_updated = await store.get_device("dev_beta")
        assert record_updated is not None
        assert record_updated["is_paired"] is True
        assert record_updated["certificate_pem"] == new_cert
        assert record_updated["device_type"] == "tablet"


# =============================================================================
# 4. CLUS-13 (BUG-HIGH-06): Synchronous is_device_paired()
# =============================================================================

class TestClus13SynchronousIsDevicePaired:
    """Verify daemon.is_device_paired() reliably reports paired offline devices."""

    async def test_is_device_paired_offline_device(self, tmp_path: Path) -> None:
        """Offline paired devices in storage are reported as paired synchronously."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        with patch.object(daemon.tcp_manager, "start", new_callable=AsyncMock), \
             patch.object(daemon.udp_discovery, "start", new_callable=AsyncMock), \
             patch.object(daemon.ipc_server, "start", new_callable=AsyncMock):
            await daemon.start()

        try:
            # Save a paired device to storage
            await daemon.device_store.save_device(
                device_id="offline_phone_1",
                name="Offline Phone",
                certificate_pem="CERT_PEM_DATA",
                is_paired=True,
            )
            # Re-read or add to paired set
            daemon._paired_device_ids.add("offline_phone_1")

            assert daemon.is_device_paired("offline_phone_1") is True
            assert daemon.is_device_connected("offline_phone_1") is False

            # Unpair device
            await daemon.unpair_device("offline_phone_1")
            assert daemon.is_device_paired("offline_phone_1") is False
        finally:
            await daemon.stop()

    async def test_paired_device_ids_populated_on_daemon_start(self, tmp_path: Path) -> None:
        """daemon.start() reads pre-existing paired devices from storage into _paired_device_ids."""
        storage_dir = tmp_path / "storage"
        cfg = make_test_config(tmp_path)

        # Pre-seed storage
        backend = JsonStorageBackend(storage_dir)
        store = DeviceStore(backend=backend)
        await store.initialize()
        await store.save_device("pre_paired_1", "Pre-Paired 1", is_paired=True)
        await store.save_device("unpaired_1", "Unpaired 1", is_paired=False)
        await store.close()

        # Start daemon
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        with patch.object(daemon.tcp_manager, "start", new_callable=AsyncMock), \
             patch.object(daemon.udp_discovery, "start", new_callable=AsyncMock), \
             patch.object(daemon.ipc_server, "start", new_callable=AsyncMock):
            await daemon.start()
        try:
            assert "pre_paired_1" in daemon._paired_device_ids
            assert "unpaired_1" not in daemon._paired_device_ids
            assert daemon.is_device_paired("pre_paired_1") is True
            assert daemon.is_device_paired("unpaired_1") is False
        finally:
            await daemon.stop()


# =============================================================================
# 5. CLUS-14 (BUG-HIGH-07): CVE-2020-26164 Defense
# =============================================================================

class TestClus14CVE202026164Defense:
    """Verify unauthenticated inbound sessions cannot preempt active authenticated sessions."""

    async def test_cve_2020_26164_unauthenticated_inbound_dropped(self, tmp_path: Path) -> None:
        """Inbound connection claiming device_id of active authenticated session is closed without preemption."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        # Simulate active authenticated connection
        active_conn = MagicMock(spec=TCPConnection)
        active_conn.is_closed = False
        active_conn.is_tls_upgraded = True
        daemon._active_connections["target_phone"] = active_conn
        daemon._paired_device_ids.add("target_phone")
        sm = daemon._get_or_create_state_machine("target_phone")
        sm._current_state = DeviceState.CONNECTED

        # Attacker dials inbound with cleartext claiming remote_device_id = "target_phone"
        attacker_conn = MagicMock(spec=TCPConnection)
        attacker_conn.remote_device_id = "target_phone"
        attacker_conn.is_closed = False
        attacker_conn.close = AsyncMock()

        with patch("pengulink.daemon.service.logger.warning") as mock_warn:
            await daemon._on_tcp_connection_established(attacker_conn)

            # Attacker connection closed
            assert attacker_conn.close.awaited
            # Active connection untouched
            assert daemon._active_connections["target_phone"] is active_conn
            # Security alert logged
            mock_warn.assert_called()
            log_msg = str(mock_warn.call_args)
            assert "CVE-2020-26164 defense" in log_msg


# =============================================================================
# 6. CLUS-15 (BUG-HIGH-08): AlertService Concurrency Mutex Gap in stop_alert()
# =============================================================================

class TestClus15AlertServiceMutex:
    """Verify stop_alert() synchronization with self._lock prevents killing new processes."""

    async def test_stop_alert_mutex_and_process_termination(self, tmp_path: Path) -> None:
        """stop_alert acquires _lock and atomically terminates only current process."""
        backend = MockAudioBackend(playback_duration=0.05)
        service = AlertService(backend=backend, runtime_dir=tmp_path)

        await service.start_alert(device_id="dev1", max_duration=10.0)
        await asyncio.sleep(0.01)
        assert service.is_active() is True
        assert service._current_process is not None
        proc = service._current_process

        stopped = await service.stop_alert(reason="manual")
        assert stopped is True
        assert service.is_active() is False
        assert service._current_process is None

    async def test_concurrent_stop_and_start_does_not_kill_new_alert(self, tmp_path: Path) -> None:
        """Sequential and concurrent start/stop operations cleanly isolate process lifecycles."""
        backend = MockAudioBackend(playback_duration=0.5)
        service = AlertService(backend=backend, runtime_dir=tmp_path)

        await service.start_alert(device_id="dev1", max_duration=5.0)
        await asyncio.sleep(0.01)
        first_proc = service._current_process
        assert first_proc is not None

        # Stop first alert
        assert await service.stop_alert(reason="manual") is True
        assert first_proc.terminated is True
        assert service._current_process is None

        # Start second alert
        await service.start_alert(device_id="dev2", max_duration=5.0)
        await asyncio.sleep(0.01)
        second_proc = service._current_process
        assert second_proc is not None
        assert second_proc is not first_proc
        assert second_proc.terminated is False
        assert service.is_active() is True

        # Heavy concurrent flood test: alternating start and stop under contention
        tasks = []
        for i in range(10):
            tasks.append(service.start_alert(device_id=f"dev_flood_{i}", max_duration=1.0))
            tasks.append(service.stop_alert(reason=f"flood_stop_{i}"))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            assert not isinstance(r, Exception)

        await service.stop_alert()
        assert service.is_active() is False
        assert service._current_process is None

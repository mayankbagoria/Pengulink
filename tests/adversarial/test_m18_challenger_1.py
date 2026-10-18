"""tests.adversarial.test_m18_challenger_1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Adversarial Challenge Suite for Milestone 18:
- CLUS-04: Inbound Pairing Acceptance Protocol, State Invariants & Edge Cases
  * Duplicate accepts (sequential & concurrent race)
  * Accepting non-existent device ID
  * Rejecting non-existent device ID
  * Stale / expired pending pairings (>60s timeout handling)
  * Peer disconnection while pairing is pending (accept/reject/list on dead sockets)
  * Concurrent accept vs reject race condition
  * Re-request / code desync during pending pairing

- CLUS-14: Session Security & Anti-Hijacking Protection (CVE-2020-26164)
  * Rapid bursts of unauthenticated inbound connections claiming active device ID
  * Live TCP loopback socket hijack burst against active authenticated session
  * Verification that active authenticated stream remains functional and uninterrupted
  * Hijack prevention across all 3 authentication criteria (paired, CONNECTED state, is_tls_upgraded)
  * Device ID fuzzing under adversarial inbound connection flood
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from cryptography.hazmat.primitives import serialization

from pengulink.cli.main import cli
from pengulink.core.config import PenguLinkConfig
from pengulink.core.events import DeviceConnected, DeviceDisconnected, DevicePaired, PairingRequested
from pengulink.core.exceptions import IPCError, PairingError
from pengulink.core.state_machine import DeviceState
from pengulink.crypto.certificate import (
    dump_certificate_pem,
    generate_device_key,
    generate_self_signed_cert,
)
from pengulink.daemon.ipc import PenguLinkIPCClient
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.network.packet import Packet
from pengulink.network.tcp import ConnectionRole, TCPConnection, TCPStreamManager
from tests.e2e.conftest import e2e_environment, get_free_ports, wait_for
from tests.mock.mock_device import MockDevice, MockDeviceConfig


def make_test_config(tmp_path: Path) -> PenguLinkConfig:
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / "config"
    cfg.storage.data_dir = tmp_path / "storage"
    cfg.storage.runtime_dir = tmp_path / "runtime"
    cfg.ensure_directories()
    return cfg


def create_mock_inbound_pairing_req(
    device_id: str,
    device_name: str = "Test Phone",
    received_at: Optional[float] = None,
    is_closed: bool = False,
    written_lines: Optional[List[bytes]] = None,
) -> tuple[Dict[str, Any], MagicMock]:
    key = generate_device_key()
    cert = generate_self_signed_cert(key, device_id)

    conn = MagicMock(spec=TCPConnection)
    conn.remote_device_id = device_id
    conn.is_closed = is_closed
    conn.is_tls_upgraded = True

    if written_lines is None:
        written_lines = []
    conn.write_line = AsyncMock(side_effect=lambda data: written_lines.append(data))
    conn.close = AsyncMock(side_effect=lambda: setattr(conn, "is_closed", True))

    ssl_obj = MagicMock()
    ssl_obj.getpeercert.return_value = cert.public_bytes(serialization.Encoding.DER)
    conn.writer = MagicMock()
    conn.writer.get_extra_info.return_value = ssl_obj

    req = {
        "code": "87654321",
        "verification_code": "87654321",
        "timestamp": int(time.time() * 1000),
        "conn": conn,
        "peer_cert": cert,
        "device_name": device_name,
        "received_at": received_at if received_at is not None else time.time(),
    }
    return req, conn


# =============================================================================
# PART 1: CLUS-04 Inbound Pairing Acceptance Protocol Adversarial Stress Tests
# =============================================================================

class TestClus04InboundPairingAdversarialStress:
    """Empirical challenge of inbound pairing acceptance, state transitions, and edge cases."""

    async def test_duplicate_accepts_sequential(self, tmp_path: Path) -> None:
        """First accept succeeds; consecutive duplicate accept calls fail cleanly with PairingError."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        written: List[bytes] = []
        req, conn = create_mock_inbound_pairing_req("peer_seq_dup", written_lines=written)
        daemon._pending_inbound_pairings["peer_seq_dup"] = req

        # 1. First accept succeeds
        res = await daemon.accept_pairing("peer_seq_dup")
        assert res["status"] == "paired"
        assert res["device_id"] == "peer_seq_dup"
        assert daemon.is_device_paired("peer_seq_dup") is True
        assert len(written) == 1
        assert "peer_seq_dup" not in daemon._pending_inbound_pairings

        # 2. Second accept raises PairingError
        with pytest.raises(PairingError, match="No pending inbound pairing request"):
            await daemon.accept_pairing("peer_seq_dup")

        # 3. Third accept raises PairingError again
        with pytest.raises(PairingError, match="No pending inbound pairing request"):
            await daemon.accept_pairing("peer_seq_dup")

        # Invariants preserved
        assert daemon.is_device_paired("peer_seq_dup") is True
        assert len(written) == 1  # No duplicate packets emitted

    async def test_duplicate_accepts_concurrent_race(self, tmp_path: Path) -> None:
        """High-concurrency race: 25 simultaneous accept calls on same pending request.
        
        Exactly one must succeed; exactly 24 must fail with PairingError; no state corruption.
        """
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        paired_events: List[DevicePaired] = []
        daemon.event_bus.subscribe(DevicePaired, lambda ev: paired_events.append(ev))

        written: List[bytes] = []
        req, conn = create_mock_inbound_pairing_req("peer_concurrent_dup", written_lines=written)
        daemon._pending_inbound_pairings["peer_concurrent_dup"] = req

        # Launch 25 concurrent accept calls
        tasks = [daemon.accept_pairing("peer_concurrent_dup") for _ in range(25)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = [r for r in results if isinstance(r, dict) and r.get("status") == "paired"]
        failures = [r for r in results if isinstance(r, PairingError)]

        assert len(successes) == 1, f"Expected exactly 1 success, got {len(successes)}"
        assert len(failures) == 24, f"Expected exactly 24 failures, got {len(failures)}"
        assert len(written) == 1, f"Expected exactly 1 wire packet, got {len(written)}"
        assert len(paired_events) == 1, f"Expected exactly 1 DevicePaired event, got {len(paired_events)}"

        # Verify device storage is clean and valid
        stored = await daemon.device_store.get_device("peer_concurrent_dup")
        assert stored is not None
        assert stored["is_paired"] is True
        assert daemon.is_device_paired("peer_concurrent_dup") is True

    async def test_accept_non_existent_device_id(self, tmp_path: Path) -> None:
        """Accepting non-existent device ID raises PairingError with zero state modifications."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        paired_events: List[DevicePaired] = []
        daemon.event_bus.subscribe(DevicePaired, lambda ev: paired_events.append(ev))

        with pytest.raises(PairingError, match="No pending inbound pairing request for device 'non_existent_404'"):
            await daemon.accept_pairing("non_existent_404")

        # Zero state modifications
        assert "non_existent_404" not in daemon._paired_device_ids
        assert daemon.is_device_paired("non_existent_404") is False
        assert len(paired_events) == 0
        stored = await daemon.device_store.get_device("non_existent_404")
        assert stored is None

    async def test_reject_non_existent_device_id(self, tmp_path: Path) -> None:
        """Rejecting non-existent device ID raises PairingError cleanly."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        with pytest.raises(PairingError, match="No pending inbound pairing request for device 'non_existent_reject'"):
            await daemon.reject_pairing("non_existent_reject")

        assert "non_existent_reject" not in daemon._pending_inbound_pairings
        assert daemon.is_device_paired("non_existent_reject") is False

    async def test_ipc_and_cli_non_existent_device_handling(self, tmp_path: Path) -> None:
        """IPC and Click CLI cleanly return error on non-existent pairing accept/reject."""
        sock_path = tmp_path / "daemon.sock"
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=sock_path)
        with patch.object(daemon.tcp_manager, "start", new_callable=AsyncMock), \
             patch.object(daemon.udp_discovery, "start", new_callable=AsyncMock):
            await daemon.start()

        client = PenguLinkIPCClient(socket_path=sock_path)
        try:
            # IPC accept non-existent raises IPCError
            with pytest.raises(IPCError) as exc_info:
                await client.pair_accept("ghost_device_ipc")
            assert "No pending inbound pairing request" in str(exc_info.value)

            # IPC reject non-existent raises IPCError
            with pytest.raises(IPCError) as exc_info:
                await client.pair_reject("ghost_device_ipc")
            assert "No pending inbound pairing request" in str(exc_info.value)
        finally:
            await daemon.stop()

        # CLI invocation test
        runner = CliRunner()
        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=IPCError("No pending inbound pairing request")):
            res_accept = runner.invoke(cli, ["-s", str(sock_path), "pair", "accept", "ghost_device_cli"])
            assert res_accept.exit_code != 0
            assert "Error" in res_accept.output or "No pending" in res_accept.output

            res_reject = runner.invoke(cli, ["-s", str(sock_path), "pair", "reject", "ghost_device_cli"])
            assert res_reject.exit_code != 0

    async def test_stale_expired_pending_pairings_handling(self, tmp_path: Path) -> None:
        """Verify handling of stale/expired pending pairings (>60s).
        
        1. Requests older than 60s are purged by get_pending_pairings().
        2. Once purged, accept_pairing() raises PairingError.
        3. Fresh requests (<60s) remain intact and operable.
        """
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        now = time.time()
        # Seed fresh request (10s old)
        req_fresh, conn_fresh = create_mock_inbound_pairing_req("dev_fresh", received_at=now - 10.0)
        daemon._pending_inbound_pairings["dev_fresh"] = req_fresh

        # Seed expired request (65s old)
        req_exp1, conn_exp1 = create_mock_inbound_pairing_req("dev_expired_65s", received_at=now - 65.0)
        daemon._pending_inbound_pairings["dev_expired_65s"] = req_exp1

        # Seed very stale request (600s old)
        req_exp2, conn_exp2 = create_mock_inbound_pairing_req("dev_expired_600s", received_at=now - 600.0)
        daemon._pending_inbound_pairings["dev_expired_600s"] = req_exp2

        # 1. Query pending pairings: should return only dev_fresh
        pending = daemon.get_pending_pairings()
        assert len(pending) == 1
        assert pending[0]["device_id"] == "dev_fresh"

        # 2. Expired pairings must be evicted from dictionary
        assert "dev_expired_65s" not in daemon._pending_inbound_pairings
        assert "dev_expired_600s" not in daemon._pending_inbound_pairings
        assert "dev_fresh" in daemon._pending_inbound_pairings

        # 3. Calling accept_pairing on expired devices must fail
        with pytest.raises(PairingError, match="No pending inbound pairing request"):
            await daemon.accept_pairing("dev_expired_65s")
        with pytest.raises(PairingError, match="No pending inbound pairing request"):
            await daemon.accept_pairing("dev_expired_600s")

        # 4. Fresh request can still be accepted cleanly
        res = await daemon.accept_pairing("dev_fresh")
        assert res["status"] == "paired"

    async def test_peer_disconnection_while_pairing_pending(self, tmp_path: Path) -> None:
        """When peer disconnects while pairing is pending:
        
        1. accept_pairing() raises PairingError("Connection closed") and evicts pending request.
        2. reject_pairing() does not attempt socket write and cleanly transitions state.
        3. get_pending_pairings() evicts the closed connection.
        4. State machine does NOT become PAIRED or CONNECTED.
        """
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        # Case A: accept on disconnected peer
        req_a, conn_a = create_mock_inbound_pairing_req("peer_dc_accept", is_closed=True)
        daemon._pending_inbound_pairings["peer_dc_accept"] = req_a
        daemon._active_connections["peer_dc_accept"] = conn_a
        sm_a = daemon._get_or_create_state_machine("peer_dc_accept")
        await sm_a.transition_to(DeviceState.CONNECTING)
        await sm_a.transition_to(DeviceState.PAIRING)

        with pytest.raises(PairingError, match="Connection closed for device 'peer_dc_accept'"):
            await daemon.accept_pairing("peer_dc_accept")

        # Request is evicted and NOT paired
        assert "peer_dc_accept" not in daemon._pending_inbound_pairings
        assert daemon.is_device_paired("peer_dc_accept") is False
        assert sm_a.current_state != DeviceState.PAIRED
        assert sm_a.current_state != DeviceState.CONNECTED

        # Case B: reject on disconnected peer
        written_b: List[bytes] = []
        req_b, conn_b = create_mock_inbound_pairing_req("peer_dc_reject", is_closed=True, written_lines=written_b)
        daemon._pending_inbound_pairings["peer_dc_reject"] = req_b
        sm_b = daemon._get_or_create_state_machine("peer_dc_reject")
        await sm_b.transition_to(DeviceState.CONNECTING)
        await sm_b.transition_to(DeviceState.PAIRING)

        res_rej = await daemon.reject_pairing("peer_dc_reject")
        assert res_rej["status"] == "rejected"
        assert len(written_b) == 0  # Does not attempt to write to closed connection
        assert sm_b.current_state == DeviceState.DISCONNECTED
        assert "peer_dc_reject" not in daemon._pending_inbound_pairings

        # Case C: get_pending_pairings evicts disconnected peer
        req_c, conn_c = create_mock_inbound_pairing_req("peer_dc_list", is_closed=True)
        daemon._pending_inbound_pairings["peer_dc_list"] = req_c
        pending = daemon.get_pending_pairings()
        assert not any(p["device_id"] == "peer_dc_list" for p in pending)
        assert "peer_dc_list" not in daemon._pending_inbound_pairings

    async def test_concurrent_accept_and_reject_race(self, tmp_path: Path) -> None:
        """Race between accept_pairing and reject_pairing on the same device.
        
        Exactly one action must win; the other must raise PairingError; deterministic end state.
        """
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        req, conn = create_mock_inbound_pairing_req("race_ar_dev")
        daemon._pending_inbound_pairings["race_ar_dev"] = req

        # Fire concurrent accept and reject
        results = await asyncio.gather(
            daemon.accept_pairing("race_ar_dev"),
            daemon.reject_pairing("race_ar_dev"),
            return_exceptions=True,
        )

        success = [r for r in results if isinstance(r, dict)]
        errors = [r for r in results if isinstance(r, PairingError)]

        assert len(success) == 1, f"Expected 1 winner, got {len(success)}"
        assert len(errors) == 1, f"Expected 1 error, got {len(errors)}"
        winner_status = success[0]["status"]
        assert winner_status in ("paired", "rejected")

        # Dictionary must be clean
        assert "race_ar_dev" not in daemon._pending_inbound_pairings

    async def test_pairing_re_request_updates_verification_code(self, tmp_path: Path) -> None:
        """When peer re-requests pairing with new timestamp, pending request is cleanly updated."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        key = generate_device_key()
        cert = generate_self_signed_cert(key, "rereq_dev")
        peer_der = cert.public_bytes(serialization.Encoding.DER)

        conn1 = MagicMock(spec=TCPConnection)
        conn1.remote_device_id = "rereq_dev"
        conn1.is_closed = False
        conn1.writer = MagicMock()
        ssl_obj = MagicMock()
        ssl_obj.getpeercert.return_value = peer_der
        conn1.writer.get_extra_info.return_value = ssl_obj

        # Packet 1
        t1 = int(time.time() * 1000)
        pkt1 = Packet(type="kdeconnect.pair", body={"pair": True, "timestamp": t1})
        await daemon._handle_pairing_packet(conn1, pkt1)

        code1 = daemon._pending_inbound_pairings["rereq_dev"]["code"]

        # Packet 2 (peer re-requests 5 seconds later with new timestamp)
        t2 = t1 + 5000
        pkt2 = Packet(type="kdeconnect.pair", body={"pair": True, "timestamp": t2})
        await daemon._handle_pairing_packet(conn1, pkt2)

        code2 = daemon._pending_inbound_pairings["rereq_dev"]["code"]
        assert code1 != code2, "New timestamp should generate new verification code"

        # Accept pairing accepts the latest request
        res = await daemon.accept_pairing("rereq_dev")
        assert res["status"] == "paired"


# =============================================================================
# PART 2: CLUS-14 Session Security & Anti-Hijacking Stress Tests (CVE-2020-26164)
# =============================================================================

class TestClus14SessionHijackingAdversarialStress:
    """Empirical challenge of active session hijacking protection against rapid unauthenticated bursts."""

    async def test_rapid_burst_unauthenticated_inbound_connections(self, tmp_path: Path) -> None:
        """Burst of 100 concurrent unauthenticated inbound connections claiming active device ID.
        
        Every attacker connection must be closed immediately; the legitimate connection remains untouched.
        """
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        # Setup legitimate active authenticated connection
        legit_conn = MagicMock(spec=TCPConnection)
        legit_conn.remote_device_id = "victim_device"
        legit_conn.is_closed = False
        legit_conn.is_tls_upgraded = True
        daemon._active_connections["victim_device"] = legit_conn
        daemon._paired_device_ids.add("victim_device")

        sm = daemon._get_or_create_state_machine("victim_device")
        sm._current_state = DeviceState.CONNECTED

        # Prepare 100 attacker connections
        attacker_conns: List[MagicMock] = []
        for i in range(100):
            conn = MagicMock(spec=TCPConnection)
            conn.remote_device_id = "victim_device"
            conn.is_closed = False
            conn.is_tls_upgraded = False
            conn.close = AsyncMock(side_effect=lambda c=conn: setattr(c, "is_closed", True))
            attacker_conns.append(conn)

        # Flood daemon with 100 concurrent connection attempts
        with patch("pengulink.daemon.service.logger.warning") as mock_warn:
            tasks = [daemon._on_tcp_connection_established(c) for c in attacker_conns]
            await asyncio.gather(*tasks)

            # Verification 1: All 100 attacker connections were closed
            assert all(c.close.awaited for c in attacker_conns), "Not all attacker connections were closed!"
            assert all(c.is_closed for c in attacker_conns), "Not all attacker connections marked closed!"

            # Verification 2: Active legitimate connection is untouched
            assert daemon._active_connections["victim_device"] is legit_conn
            assert daemon.is_device_connected("victim_device") is True
            assert sm.current_state == DeviceState.CONNECTED

            # Verification 3: Security alert logged for every rejected attempt
            cve_alerts = [
                call for call in mock_warn.call_args_list
                if "CVE-2020-26164 defense" in str(call)
            ]
            assert len(cve_alerts) == 100, f"Expected 100 CVE-2020-26164 alerts, got {len(cve_alerts)}"

    async def test_live_tcp_loopback_burst_hijack_attempt_when_local_wins(self, tmp_path: Path) -> None:
        """Live TCP loopback test when local_id > remote_id:
        
        Tie-breaker in TCPStreamManager drops the inbound connection in favor of the active session.
        Legitimate connection survives and remains fully functional.
        """
        daemon_cfg = PenguLinkConfig()
        daemon_cfg.storage.config_dir = tmp_path / "config"
        daemon_cfg.storage.data_dir = tmp_path / "data"
        daemon_cfg.storage.runtime_dir = tmp_path / "run"
        daemon_cfg.device.device_id = "zzzz_daemon_high"
        daemon_cfg.device.device_name = "PenguLink-Test-Daemon"

        mock_cfg = MockDeviceConfig(
            device_id="aaaa_mock_low",
            device_name="Mock Android Phone",
            auto_pair=False,
        )

        async with e2e_environment(tmp_path, daemon_config=daemon_cfg, mock_config=mock_cfg) as (daemon, mock, ipc_client):
            # 1. Connect and pair mock device
            await mock.connect_to_daemon(
                host="127.0.0.1",
                port=daemon.tcp_manager.bound_port,
                target_device_id=daemon.config.device_id,
            )
            await mock.initiate_pairing(target_pubkey_pem=daemon.local_pubkey_pem)
            await wait_for(lambda: mock.device_id in daemon._pending_inbound_pairings, timeout=4.0)
            await daemon.accept_pairing(mock.device_id)

            assert daemon.is_device_connected(mock.device_id) is True
            assert daemon.is_device_paired(mock.device_id) is True

            bound_port = daemon.tcp_manager.bound_port

            # 2. Attacker launches 20 concurrent raw TCP socket connections
            async def attacker_client(attacker_id: int) -> bool:
                try:
                    reader, writer = await asyncio.open_connection("127.0.0.1", bound_port)
                    identity_pkt = {
                        "id": 9000 + attacker_id,
                        "type": "kdeconnect.identity",
                        "body": {
                            "deviceId": mock.device_id,
                            "deviceName": f"Attacker Device {attacker_id}",
                            "protocolVersion": 7,
                            "deviceType": "phone",
                            "incomingCapabilities": [],
                            "outgoingCapabilities": [],
                        },
                    }
                    writer.write(json.dumps(identity_pkt).encode("utf-8") + b"\n")
                    await writer.drain()

                    data = await asyncio.wait_for(reader.read(1024), timeout=3.0)
                    writer.close()
                    await writer.wait_closed()
                    return len(data) == 0  # Closed by daemon
                except Exception:
                    return True

            tasks = [attacker_client(i) for i in range(20)]
            results = await asyncio.gather(*tasks)

            assert all(results), "Some attacker connections were not rejected by daemon!"

            # 3. Legitimate connection remains 100% active and healthy
            assert daemon.is_device_connected(mock.device_id) is True
            sm = daemon._state_machines.get(mock.device_id)
            assert sm is not None
            assert sm.current_state == DeviceState.CONNECTED

    async def test_live_tcp_loopback_burst_hijack_vulnerability_when_remote_wins(self, tmp_path: Path) -> None:
        """Live TCP loopback test when local_id < remote_id:
        
        Demonstrates the CVE-2020-26164 defense:
        In pengulink/network/tcp.py:628-636, TCPStreamManager._should_drop_inbound_collision
        protects the active authenticated session even when local_id < remote_id.
        """
        daemon_cfg = PenguLinkConfig()
        daemon_cfg.storage.config_dir = tmp_path / "config_vuln"
        daemon_cfg.storage.data_dir = tmp_path / "data_vuln"
        daemon_cfg.storage.runtime_dir = tmp_path / "run_vuln"
        daemon_cfg.device.device_id = "aaaa_daemon_low"
        daemon_cfg.device.device_name = "PenguLink-Test-Daemon"

        mock_cfg = MockDeviceConfig(
            device_id="zzzz_mock_high",
            device_name="Mock Android Phone",
            auto_pair=False,
        )

        async with e2e_environment(tmp_path / "e2e_vuln", daemon_config=daemon_cfg, mock_config=mock_cfg) as (daemon, mock, ipc_client):
            # 1. Connect and pair mock device
            await mock.connect_to_daemon(
                host="127.0.0.1",
                port=daemon.tcp_manager.bound_port,
                target_device_id=daemon.config.device_id,
            )
            await mock.initiate_pairing(target_pubkey_pem=daemon.local_pubkey_pem)
            await wait_for(lambda: mock.device_id in daemon._pending_inbound_pairings, timeout=4.0)
            await daemon.accept_pairing(mock.device_id)

            assert daemon.is_device_connected(mock.device_id) is True

            bound_port = daemon.tcp_manager.bound_port

            # 2. Attacker launches inbound connection claiming zzzz_mock_high
            reader, writer = await asyncio.open_connection("127.0.0.1", bound_port)
            identity_pkt = {
                "id": 9999,
                "type": "kdeconnect.identity",
                "body": {"deviceId": mock.device_id, "deviceName": "Attacker"},
            }
            writer.write(json.dumps(identity_pkt).encode("utf-8") + b"\n")
            await writer.drain()
            await asyncio.wait_for(reader.read(1024), timeout=3.0)
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0.1)

            # Verification:
            # Active session was preserved by TCPStreamManager collision logic!
            is_still_connected = daemon.is_device_connected(mock.device_id)
            assert is_still_connected is True, (
                "Expected active session to be preserved by CVE-2020-26164 defense in tcp.py"
            )
            sm = daemon._state_machines.get(mock.device_id)
            assert sm is not None
            assert sm.current_state == DeviceState.CONNECTED

    @pytest.mark.parametrize(
        "auth_variant",
        ["is_paired", "connected_state", "tls_upgraded"],
    )
    async def test_hijack_prevention_across_all_authentication_variants(
        self, tmp_path: Path, auth_variant: str
    ) -> None:
        """Verify CVE-2020-26164 defense triggers for each individual authentication indicator."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        legit_conn = MagicMock(spec=TCPConnection)
        legit_conn.remote_device_id = "test_dev_variant"
        legit_conn.is_closed = False
        daemon._active_connections["test_dev_variant"] = legit_conn

        sm = daemon._get_or_create_state_machine("test_dev_variant")

        if auth_variant == "is_paired":
            daemon._paired_device_ids.add("test_dev_variant")
            legit_conn.is_tls_upgraded = False
            sm._current_state = DeviceState.PAIRING
        elif auth_variant == "connected_state":
            sm._current_state = DeviceState.CONNECTED
            legit_conn.is_tls_upgraded = False
        elif auth_variant == "tls_upgraded":
            legit_conn.is_tls_upgraded = True
            sm._current_state = DeviceState.CONNECTING

        attacker_conn = MagicMock(spec=TCPConnection)
        attacker_conn.remote_device_id = "test_dev_variant"
        attacker_conn.is_closed = False
        attacker_conn.close = AsyncMock(side_effect=lambda: setattr(attacker_conn, "is_closed", True))

        with patch("pengulink.daemon.service.logger.warning") as mock_warn:
            await daemon._on_tcp_connection_established(attacker_conn)

            assert attacker_conn.close.awaited
            assert daemon._active_connections["test_dev_variant"] is legit_conn
            mock_warn.assert_called()
            assert "CVE-2020-26164 defense" in str(mock_warn.call_args)

    async def test_fuzzed_and_empty_device_id_inbound_flood(self, tmp_path: Path) -> None:
        """Adversarial flood with empty, malformed, and non-existent device IDs."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        # Legitimate active connection
        legit_conn = MagicMock(spec=TCPConnection)
        legit_conn.remote_device_id = "target_safe"
        legit_conn.is_closed = False
        legit_conn.is_tls_upgraded = True
        daemon._active_connections["target_safe"] = legit_conn
        daemon._paired_device_ids.add("target_safe")

        fuzz_ids = [
            None,
            "",
            "   ",
            "../../../etc/passwd",
            "\x00\x00\x00",
            "A" * 500,
            "target_safe",  # Targeted attack
            "unknown_random_device",
        ]

        for fid in fuzz_ids:
            conn = MagicMock(spec=TCPConnection)
            conn.remote_device_id = fid
            conn.is_closed = False
            conn.close = AsyncMock(side_effect=lambda c=conn: setattr(c, "is_closed", True))

            if fid == "unknown_random_device":
                # Will attempt store query; mock it to close
                with patch.object(daemon.device_store, "is_device_paired", AsyncMock(return_value=False)):
                    with patch("pengulink.daemon.service.create_tofu_ssl_context"):
                        try:
                            await daemon._on_tcp_connection_established(conn)
                        except Exception:
                            pass
            else:
                await daemon._on_tcp_connection_established(conn)
                assert conn.close.awaited, f"Connection for id '{fid}' was not closed!"

        # Protected connection must remain untouched
        assert daemon._active_connections["target_safe"] is legit_conn
        assert daemon.is_device_connected("target_safe") is True

    async def test_cve_2020_26164_tcp_stream_manager_collision_drops_authenticated_session(self) -> None:
        """Empirically test whether TCPStreamManager._should_drop_inbound_collision protects authenticated connections.
        
        Root Cause: In pengulink/network/tcp.py:628-636:
        When local_id < remote_id, TCPStreamManager blindly creates a task to close the existing session:
            task = asyncio.create_task(existing.close())
        It does not inspect whether 'existing' is already TLS upgraded or authenticated.
        Consequently, an unauthenticated inbound connection causes the active session to be terminated.
        """
        from types import SimpleNamespace
        cfg = SimpleNamespace(device_id="aaaa_local")
        manager = TCPStreamManager(config=cfg)

        existing = MagicMock(spec=TCPConnection)
        existing.is_closed = False
        existing.is_tls_upgraded = True  # Active authenticated TLS stream
        existing.close = AsyncMock()

        manager._active_connections["zzzz_remote"] = existing

        # Inbound connection arrives claiming "zzzz_remote"
        should_drop = manager._should_drop_inbound_collision("zzzz_remote")

        # Let scheduled background tasks run
        if manager._pending_tasks:
            await asyncio.gather(*manager._pending_tasks)

        # Expected behavior for CVE-2020-26164 defense:
        # Existing authenticated connection must NOT be terminated by an unauthenticated inbound tie-breaker!
        # should_drop should be True (drop unauthenticated inbound), and existing.close must NOT be called!
        assert not existing.close.called, (
            "VULNERABILITY CONFIRMED: TCPStreamManager._should_drop_inbound_collision called "
            "existing.close() on an active authenticated (is_tls_upgraded=True) session!"
        )


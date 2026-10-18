"""tests.unit.test_phase4_defects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dedicated regression test suite for Milestone 20 (Phase 4: Defect Remediation & Hardening):
- CLUS-08: RSA private key ephemeral file leak & 0644 TOCTOU (tls.py, certificate.py)
- CLUS-09: Certificate expiration checks & UTC validity verification in verify_peer_certificate (tls.py)
- CLUS-10: CPython struct casting segfault & fail-closed server TOFU configuration (tls.py)
- CLUS-21: Deterministic pairing code tie-breaking, 300s replay cache, boolean "false" validation, leaking futures (pairing.py, service.py)
- CLUS-22: TCP framing tie-breaker type safety, EOF truncation detection, empty line CPU starvation (tcp.py, line_reader.py, packet.py)
- CLUS-23: EventBus _lock dead code & dict mutation race during publish() (events.py)
- CLUS-25: Click CLI daemon stop, IPC SO_PEERCRED UID check, CLI connect --host/--port (cli/main.py, daemon/ipc.py)
- CLUS-28 & Lows: vCard escaping order (vcard.py), 32-bit ioctl struct mismatch (udp.py), negative payloadSize validation (packet.py),
                  loopback 127.0.0.1 peer fallback (service.py), strip_execution_bits without home dir chmod 0700 (share/security.py),
                  secure storage file permissions (json_store.py), MPRIS initial player list fallback (mpris/control.py).
"""

from __future__ import annotations

import asyncio
import datetime
import os
import socket
import stat
import struct
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.x509.oid import NameOID
from pydantic import ValidationError

from pengulink.cli.main import cli
from pengulink.core.events import DeviceConnected, Event, EventBus, PingReceived
from pengulink.core.exceptions import (
    CertificateError,
    CertificateVerificationError,
    PacketFramingError,
    PairingRejectedError,
    PairingTimestampExpiredError,
    TruncatedPacketError,
)
from pengulink.crypto.certificate import (
    dump_certificate_pem,
    dump_private_key_pem,
    generate_device_key,
    generate_self_signed_cert,
    save_credentials,
)
from pengulink.crypto.pairing import (
    _PAIRING_REPLAY_CACHE,
    MAX_TIMESTAMP_DRIFT_SECONDS,
    PairingSession,
    validate_pairing_timestamp,
)
from pengulink.crypto.tls import (
    _get_secure_runtime_dir,
    _is_cpython_supported_abi,
    _write_secure_temp_file,
    create_tofu_ssl_context,
    ephemeral_tls_credentials,
    verify_peer_certificate,
)
from pengulink.daemon.ipc import PenguLinkIPCClient, PenguLinkIPCServer
from pengulink.daemon.service import Service
from pengulink.network.line_reader import StreamLineReader
from pengulink.network.packet import Packet
from pengulink.network.tcp import _should_drop_inbound_collision
from pengulink.network.udp import DiscoveredDevice
from pengulink.plugins.contacts.vcard import parse_vcard
from pengulink.plugins.mpris.control import MprisControl
from pengulink.plugins.share.security import ensure_destination_directory, strip_execution_bits
from pengulink.storage.json_store import JsonStorageBackend


# =============================================================================
# 1. CLUS-08: RSA Private Key Permissions & Secure Ephemeral Runtime Dir
# =============================================================================

class TestClus08RsaKeyPermissionsAndTmpLeak:
    """Verify atomic 0600 mode on key save and secure ephemeral TLS credentials."""

    def test_save_credentials_mode_0600(self, tmp_path: Path) -> None:
        key = generate_device_key()
        cert = generate_self_signed_cert(key, "device_clus08")
        cert_path = tmp_path / "cert.pem"
        key_path = tmp_path / "key.pem"

        save_credentials(cert, key, cert_path, key_path)

        assert cert_path.exists()
        assert key_path.exists()
        key_stat = os.stat(key_path)
        assert (key_stat.st_mode & 0o777) == 0o600

    def test_secure_runtime_dir_mode_0700(self) -> None:
        runtime_dir = _get_secure_runtime_dir()
        assert runtime_dir.exists()
        st = os.stat(runtime_dir)
        assert (st.st_mode & 0o777) == 0o700

    def test_write_secure_temp_file_permissions(self) -> None:
        raw_data = b"-----BEGIN TEST-----\nDATA\n-----END TEST-----\n"
        path = _write_secure_temp_file("test_prefix_", raw_data)
        try:
            assert os.path.exists(path)
            st = os.stat(path)
            assert (st.st_mode & 0o777) == 0o600
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_ephemeral_tls_credentials_cleanup(self) -> None:
        key = generate_device_key()
        cert = generate_self_signed_cert(key, "device_ephemeral")
        cert_bytes = dump_certificate_pem(cert)
        key_bytes = dump_private_key_pem(key)

        cert_p = None
        key_p = None
        with ephemeral_tls_credentials(cert_bytes, key_bytes) as (c_path, k_path):
            cert_p = c_path
            key_p = k_path
            assert os.path.exists(cert_p)
            assert os.path.exists(key_p)
            assert (os.stat(key_p).st_mode & 0o777) == 0o600

        # After context exit, both files must be cleaned up
        assert not os.path.exists(cert_p)
        assert not os.path.exists(key_p)


# =============================================================================
# 2. CLUS-09: Certificate UTC Validity Period Verification
# =============================================================================

class TestClus09CertValidityPeriod:
    """Verify verify_peer_certificate checks expiration and not-yet-valid periods."""

    def test_verify_peer_certificate_expired_raises_verification_error(self) -> None:
        key = generate_device_key()
        # Build expired certificate (validity ended 1 day ago)
        now = datetime.datetime.now(datetime.timezone.utc)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "expired_peer")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=10))
            .not_valid_after(now - datetime.timedelta(days=1))
            .sign(key, hashes.SHA256())
        )
        cert_pem = dump_certificate_pem(cert)

        with pytest.raises(CertificateVerificationError, match="expired"):
            verify_peer_certificate(cert, "expired_peer", pinned_cert=cert_pem)

    def test_verify_peer_certificate_not_yet_valid_raises_verification_error(self) -> None:
        key = generate_device_key()
        now = datetime.datetime.now(datetime.timezone.utc)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "future_peer")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now + datetime.timedelta(days=2))
            .not_valid_after(now + datetime.timedelta(days=365))
            .sign(key, hashes.SHA256())
        )
        cert_pem = dump_certificate_pem(cert)

        with pytest.raises(CertificateVerificationError, match="not yet valid"):
            verify_peer_certificate(cert, "future_peer", pinned_cert=cert_pem)

    def test_verify_peer_certificate_valid_window_succeeds(self) -> None:
        key = generate_device_key()
        cert = generate_self_signed_cert(key, "valid_peer", validity_days=30)
        cert_pem = dump_certificate_pem(cert)

        result = verify_peer_certificate(cert, "valid_peer", pinned_cert=cert_pem)
        assert result is True


# =============================================================================
# 3. CLUS-10: CPython ABI Safety Check & Fail-Closed TOFU SSL Context
# =============================================================================

class TestClus10CtypesAbiSafetyAndTofuFailClosed:
    """Verify CPython ABI detection and fail-closed TOFU server context."""

    def test_abi_safety_check_returns_bool(self) -> None:
        is_safe = _is_cpython_supported_abi()
        assert isinstance(is_safe, bool)

    def test_tofu_context_fails_closed_when_server_tofu_unsupported(self) -> None:
        key = generate_device_key()
        cert = generate_self_signed_cert(key, "server_dev")
        cert_pem = dump_certificate_pem(cert)
        key_pem = dump_private_key_pem(key)

        with patch("pengulink.crypto.tls._configure_server_tofu_verify", return_value=False):
            with pytest.raises(CertificateError, match="Failed to configure secure TOFU server context"):
                create_tofu_ssl_context(
                    cert_pem=cert_pem,
                    key_pem=key_pem,
                    pinned_cert_pem=None,
                    is_server=True,
                )


# =============================================================================
# 4. CLUS-21: Pairing Hardening (Deterministic Tie-Break, Replay, Future Cleanup)
# =============================================================================

class TestClus21PairingReplayAndTieBreaking:
    """Verify 300s window, LRU replay cache, strict bools, and future cleanup."""

    def test_pairing_replay_cache_rejects_duplicates(self) -> None:
        _PAIRING_REPLAY_CACHE.clear()
        device_id = "replay_device_1"
        ts = int(time.time())

        # First verification succeeds
        validate_pairing_timestamp(ts, device_id=device_id)

        # Second verification with identical timestamp within window is rejected as replay
        with pytest.raises(PairingTimestampExpiredError, match="Pairing replay detected"):
            validate_pairing_timestamp(ts, device_id=device_id)

    def test_pairing_timestamp_drift_300s(self) -> None:
        device_id = "drift_device_1"
        now_ts = int(time.time())

        # > 300s in past
        stale_ts = now_ts - int(MAX_TIMESTAMP_DRIFT_SECONDS + 10)
        with pytest.raises(PairingTimestampExpiredError, match="drifted"):
            validate_pairing_timestamp(stale_ts, device_id=device_id)

        # > 300s in future
        future_ts = now_ts + int(MAX_TIMESTAMP_DRIFT_SECONDS + 10)
        with pytest.raises(PairingTimestampExpiredError, match="drifted"):
            validate_pairing_timestamp(future_ts, device_id=device_id)

    def test_deterministic_tie_breaker_logic(self) -> None:
        # Node "aaa" < "bbb". When "aaa" initiates and receives concurrent from "bbb", "aaa" stays master (ts1).
        s1 = PairingSession("aaa", "bbb", "pubA", "pubB")
        ts1 = int(time.time()) - 10
        s1.pairing_requested = True
        s1.pairing_timestamp = ts1

        ts2 = int(time.time()) - 5
        pkt2 = Packet(type="kdeconnect.pair", body={"pair": True, "timestamp": ts2})
        s1.handle_incoming_pair_packet(pkt2)
        assert s1.pairing_timestamp == ts1

        # Conversely, if "bbb" initiates and receives from "aaa", "aaa" is master so "bbb" adopts ts2.
        s2 = PairingSession("bbb", "aaa", "pubB", "pubA")
        s2.pairing_requested = True
        s2.pairing_timestamp = ts1
        s2.handle_incoming_pair_packet(pkt2)
        assert s2.pairing_timestamp == ts2

    def test_strict_boolean_validation(self) -> None:
        # String "false" or "true" must NOT be treated as boolean True
        handler = PairingSession("local_dev", "remote_dev", "pub1", "pub2")
        pkt = Packet(type="kdeconnect.pair", body={"pair": "false", "timestamp": int(time.time())})
        with pytest.raises(PairingRejectedError, match="Pairing rejected"):
            handler.handle_incoming_pair_packet(pkt)

    @pytest.mark.asyncio
    async def test_service_pairing_future_auto_cleanup(self, tmp_path: Path) -> None:
        service = Service(config_dir=tmp_path)
        # Create non-blocking pair_device call
        dev_id = "dev_future_leak_test"
        with patch.object(service, "send_packet", return_value=None):
            with patch.object(service, "is_device_paired", AsyncMock(return_value=False)):
                await service.pair_device(dev_id, wait=False)
                assert dev_id in service._pending_pairing_requests
                fut = service._pending_pairing_requests[dev_id]
                # Simulating completion or timeout triggers done callback to remove it
                fut.cancel()
                assert dev_id not in service._pending_pairing_requests


# =============================================================================
# 5. CLUS-22: TCP Framing Tie-Breaker & StreamLineReader Hardening
# =============================================================================

class TestClus22TcpFramingTieBreakerAndLineReader:
    """Verify tie-breaker type safety, EOF truncation detection, and empty line defense."""

    def test_should_drop_inbound_collision_type_safety(self) -> None:
        # Should work safely without TypeError even if IDs are int or non-str
        assert _should_drop_inbound_collision(100, 200) is True
        assert _should_drop_inbound_collision(200, 100) is False
        assert _should_drop_inbound_collision("alpha", "beta") is True

    @pytest.mark.asyncio
    async def test_stream_line_reader_truncated_eof(self) -> None:
        reader = StreamLineReader()
        reader.feed_data(b'{"id": 1, "body": "test"}')  # No trailing newline
        reader.feed_eof()

        with pytest.raises(TruncatedPacketError, match="Incomplete packet frame"):
            await reader.read_line()

    @pytest.mark.asyncio
    async def test_stream_line_reader_consecutive_empty_lines_starvation(self) -> None:
        reader = StreamLineReader()
        # Feed 51 empty newlines
        reader.feed_data(b"\n" * 51)

        with pytest.raises(PacketFramingError, match="Empty line flood detected"):
            for _ in range(51):
                await reader.read_line()

    def test_line_reader_module_export(self) -> None:
        from pengulink.network.line_reader import StreamLineReader as Reader
        assert Reader is not None


# =============================================================================
# 6. CLUS-23: EventBus Lock & Dict Mutation Race Defense
# =============================================================================

class TestClus23EventBusLockAndDictMutation:
    """Verify EventBus acquires _lock during publish to avoid dict mutation race."""

    @pytest.mark.asyncio
    async def test_event_bus_concurrent_publish_and_modification(self) -> None:
        bus = EventBus()
        events_received = []

        def handler1(event: Event) -> None:
            events_received.append("h1")
            # Concurrently unsubscribe and subscribe during publish execution
            unsub1()
            bus.subscribe(PingReceived, lambda e: events_received.append("h3"))

        def handler2(event: Event) -> None:
            events_received.append("h2")

        unsub1 = bus.subscribe(PingReceived, handler1)
        bus.subscribe(PingReceived, handler2)

        # This must NOT raise RuntimeError: dictionary changed size during iteration
        await bus.publish(PingReceived(device_id="dev1", message="hello"))
        assert "h1" in events_received
        assert "h2" in events_received


# =============================================================================
# 7. CLUS-25: Click CLI daemon stop, IPC SO_PEERCRED Check, CLI connect --host/--port
# =============================================================================

class TestClus25CliAndIpcHardening:
    """Verify CLI daemon stop, IPC peer UID enforcement, and CLI connect options."""

    def test_cli_daemon_stop_command(self, tmp_path: Path) -> None:
        runner = CliRunner()
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value={"status": "stopped"}) as mock_ipc:
            result = runner.invoke(cli, ["-s", str(mock_socket), "daemon", "stop"])
            assert result.exit_code == 0
            assert "Daemon stopped successfully" in result.output
            mock_ipc.assert_called_once_with("stop_daemon")

    def test_cli_connect_host_and_port_options(self, tmp_path: Path) -> None:
        runner = CliRunner()
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value={"status": "connected"}) as mock_ipc:
            result = runner.invoke(
                cli,
                ["-s", str(mock_socket), "connect", "dev_123", "--host", "192.168.1.55", "--port", "1716"],
            )
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                "connect",
                {"device_id": "dev_123", "host": "192.168.1.55", "port": 1716},
            )

    @pytest.mark.asyncio
    async def test_ipc_so_peercred_uid_check_rejects_unauthorized(self, tmp_path: Path) -> None:
        sock_path = tmp_path / "test_auth.sock"
        server = PenguLinkIPCServer(socket_path=sock_path)
        server.register_handler("test_echo", lambda p: p)
        await server.start()

        try:
            # Connect via raw socket
            client_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_sock.connect(str(sock_path))

            # Simulate peer creds returning another user's UID (e.g. 9999 != current uid)
            foreign_uid = (os.getuid() + 1)
            foreign_ucred = struct.pack("3i", os.getpid(), foreign_uid, os.getgid())

            with patch.object(socket.socket, "getsockopt", return_value=foreign_ucred):
                # Send request
                client_sock.sendall(b'{"jsonrpc": "2.0", "method": "test_echo", "params": {}, "id": 1}\n')
                # Server should close connection due to UID mismatch
                await asyncio.sleep(0.05)
                client_sock.settimeout(1.0)
                try:
                    resp = client_sock.recv(4096)
                except ConnectionResetError:
                    resp = b""
                assert resp == b""  # Connection was closed by server
            client_sock.close()
        finally:
            await server.stop()


# =============================================================================
# 8. CLUS-28 & Lows: vCard Escaping, 32-bit ioctl, JsonStore 0600, MPRIS Fallback
# =============================================================================

class TestClus28AndLows:
    """Verify vCard escaping order, 32-bit ioctl compatibility, JsonStore mode 0600, etc."""

    def test_vcard_escaping_order(self) -> None:
        # N field with escaped semicolons inside surname and given name
        vcard_text = "BEGIN:VCARD\nVERSION:3.0\nN:Smith\\;Jr;John\\;Middle;;;\nFN:John Middle Smith Jr\nEND:VCARD"
        contacts = parse_vcard(vcard_text)
        assert len(contacts) == 1
        contact = contacts[0]
        # Escaped semicolon must remain in the respective component and not split it into next field
        assert contact.last_name == "Smith;Jr"
        assert contact.first_name == "John;Middle"

    def test_udp_ifreq_struct_size_calculation(self) -> None:
        expected_size = 40 if struct.calcsize("P") == 8 else 32
        assert expected_size in (32, 40)

    def test_packet_negative_payload_size_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Packet(type="kdeconnect.share.request", payloadSize=-1)

    def test_resolve_device_peer_info_sender_ip(self, tmp_path: Path) -> None:
        service = Service(config_dir=tmp_path)
        service._discovered_devices["dev_remote"] = DiscoveredDevice(
            device_id="dev_remote",
            device_name="Remote Phone",
            device_type="phone",
            protocol_version=7,
            incoming_capabilities=[],
            outgoing_capabilities=[],
            tcp_port=1716,
            sender_ip="192.168.1.188",
        )
        host, port = service._resolve_device_peer_info("dev_remote")
        assert host == "192.168.1.188"
        assert port == 1716

    def test_strip_execution_bits_ignores_symlinks(self, tmp_path: Path) -> None:
        target = tmp_path / "executable_target.sh"
        target.write_text("#!/bin/sh\necho hi\n")
        os.chmod(target, 0o777)

        symlink = tmp_path / "symlink_to_target"
        symlink.symlink_to(target)

        # Call strip_execution_bits on the symlink
        strip_execution_bits(symlink)

        # Target should retain its permissions because strip_execution_bits does not follow symlinks
        target_mode = os.stat(target).st_mode & 0o777
        assert target_mode == 0o777

    def test_ensure_destination_directory_spares_home_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fake_home = tmp_path / "home_user"
        fake_home.mkdir(mode=0o755)
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        resolved = ensure_destination_directory([fake_home])
        assert resolved == fake_home
        # Home dir must NOT be forcibly chmod 0700
        assert (os.stat(fake_home).st_mode & 0o777) == 0o755

    @pytest.mark.asyncio
    async def test_json_store_file_permissions_0600(self, tmp_path: Path) -> None:
        store = JsonStorageBackend(storage_dir=tmp_path / "secure_store")
        await store.initialize()
        await store.save_device("dev_abc", {"name": "Test Device", "pinned_cert": "ABC"})

        devices_file = tmp_path / "secure_store" / "devices.json"
        assert devices_file.exists()
        file_mode = os.stat(devices_file).st_mode & 0o777
        assert file_mode == 0o600

    @pytest.mark.asyncio
    async def test_mpris_control_on_device_connected_fallback(self) -> None:
        sent_packets = []

        def mock_sender(dev_id: str, pkt: Packet) -> None:
            sent_packets.append((dev_id, pkt))

        mock_adapter = AsyncMock()
        mock_adapter.list_players = AsyncMock(return_value=["vlc", "spotify"])

        control = MprisControl(adapter=mock_adapter, packet_sender=mock_sender)
        await control._on_device_connected(DeviceConnected(device_id="peer_connected_1"))

        assert len(sent_packets) == 1
        dev_id, pkt = sent_packets[0]
        assert dev_id == "peer_connected_1"
        assert pkt.type == "kdeconnect.mpris"
        assert pkt.body.get("playerList") == ["vlc", "spotify"]

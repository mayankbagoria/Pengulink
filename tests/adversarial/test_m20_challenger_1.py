"""tests.adversarial.test_m20_challenger_1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial Stress Test Suite — Challenger 1 (Milestone 20 / Phase 4 Integration & Hardening)

Empirical verification targeting:
1. Cryptography & Certificate Verification (CLUS-08, CLUS-09):
   - Expired certificates (1s, 1d, 1y ago) strictly raising CertificateVerificationError.
   - Future certificates (valid in 5s, 1d, 1y) strictly raising CertificateVerificationError.
   - Invalid UTC validity windows, malformed DER/PEM, and truncated certificate bytes.
   - Subject Common Name (CN) spoofing, empty CN, path traversal injection in CN.
   - Certificate pinning fingerprint mismatches.
   - Ephemeral TLS credentials file mode 0600 and guaranteed deletion upon normal exit,
     unhandled exceptions, early returns, and concurrent execution.
2. CPython ABI Safety & Server TOFU (CLUS-10):
   - CPython ABI safety detection under simulated ABI mutations (PyPy, Py_TRACE_REFS, Py_GIL_DISABLED).
   - Server-side TOFU context creation fail-closed behavior when libssl/ctypes verification cannot be secured.
   - Missing certificate/key inputs fail-closed.
3. Pairing Replay Cache & State Machine Resilience (CLUS-21):
   - 300s window boundary tests (exactly 300.0s, 300.5s drift, past and future).
   - Replay attack defenses: single replay and 100-packet duplicate packet bursts rejected.
   - Replay cache eviction / capacity limits under 1100 unique entries.
   - Per-device cache isolation.
   - Deterministic tie-breaking on device IDs during concurrent pairing negotiation.
   - Strict boolean validation on pair payload (rejecting string 'false', 'true', int, None, collections).
4. Transport Framing & Starvation Defenses (CLUS-22):
   - Collision drop tie-breaker type safety across heterogeneous types.
   - StreamLineReader empty line flood (>100 empty lines) CPU starvation defense raising PacketFramingError.
   - StreamLineReader counter reset on valid non-empty packets (intermittent blank lines).
   - StreamLineReader incomplete line EOF truncation detection raising TruncatedPacketError.
   - StreamLineReader 128 KB buffer overrun protection raising PacketBufferOverrunError.
   - StreamLineReader consecutive malformed packet threshold defense raising MalformedPacketError.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import stat
import sys
import sysconfig
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from pengulink.core.exceptions import (
    CertificateError,
    CertificateVerificationError,
    MalformedPacketError,
    PacketBufferOverrunError,
    PacketFramingError,
    PairingError,
    PairingRejectedError,
    PairingTimestampExpiredError,
    TruncatedPacketError,
)
from pengulink.crypto.certificate import (
    dump_certificate_pem,
    dump_private_key_pem,
    generate_device_key,
    generate_self_signed_cert,
)
from pengulink.crypto.pairing import (
    _PAIRING_REPLAY_CACHE,
    MAX_TIMESTAMP_DRIFT_SECONDS,
    PairingSession,
    compute_verification_code,
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
from pengulink.network.line_reader import StreamLineReader
from pengulink.network.packet import MAX_PACKET_SIZE, Packet
from pengulink.network.tcp import _should_drop_inbound_collision


# ==============================================================================
# Helper Factories
# ==============================================================================

def create_cert_with_bounds(
    key: rsa.RSAPrivateKey,
    device_id: str,
    not_valid_before: datetime.datetime,
    not_valid_after: datetime.datetime,
) -> x509.Certificate:
    """Construct an X.509 certificate with explicit UTC validity bounds."""
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, device_id)])
    return (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_valid_before)
        .not_valid_after(not_valid_after)
        .sign(key, hashes.SHA256())
    )


# ==============================================================================
# 1. Cryptography & Certificate Verification (CLUS-08 & CLUS-09)
# ==============================================================================

class TestAdversarialCertificates:
    """Stress tests for verify_peer_certificate across expiration, future, and malformed certs."""

    def test_expired_certificate_immediate_past(self) -> None:
        """Certificate whose validity ended 1 second ago must strictly raise CertificateVerificationError."""
        key = generate_device_key()
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = create_cert_with_bounds(
            key,
            device_id="expired_node_1s",
            not_valid_before=now - datetime.timedelta(days=1),
            not_valid_after=now - datetime.timedelta(seconds=1),
        )
        cert_pem = dump_certificate_pem(cert)

        with pytest.raises(CertificateVerificationError, match="has expired"):
            verify_peer_certificate(cert, "expired_node_1s", pinned_cert=cert_pem)

    def test_expired_certificate_distant_past(self) -> None:
        """Certificate whose validity ended years ago must strictly raise CertificateVerificationError."""
        key = generate_device_key()
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = create_cert_with_bounds(
            key,
            device_id="expired_node_years",
            not_valid_before=now - datetime.timedelta(days=730),
            not_valid_after=now - datetime.timedelta(days=365),
        )
        cert_pem = dump_certificate_pem(cert)

        with pytest.raises(CertificateVerificationError, match="has expired"):
            verify_peer_certificate(cert, "expired_node_years", pinned_cert=cert_pem)

    def test_future_certificate_immediate(self) -> None:
        """Certificate valid starting in 5 seconds must strictly raise CertificateVerificationError."""
        key = generate_device_key()
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = create_cert_with_bounds(
            key,
            device_id="future_node_5s",
            not_valid_before=now + datetime.timedelta(seconds=5),
            not_valid_after=now + datetime.timedelta(days=365),
        )
        cert_pem = dump_certificate_pem(cert)

        with pytest.raises(CertificateVerificationError, match="not yet valid"):
            verify_peer_certificate(cert, "future_node_5s", pinned_cert=cert_pem)

    def test_future_certificate_distant(self) -> None:
        """Certificate valid starting in 30 days must strictly raise CertificateVerificationError."""
        key = generate_device_key()
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = create_cert_with_bounds(
            key,
            device_id="future_node_30d",
            not_valid_before=now + datetime.timedelta(days=30),
            not_valid_after=now + datetime.timedelta(days=365),
        )
        cert_pem = dump_certificate_pem(cert)

        with pytest.raises(CertificateVerificationError, match="not yet valid"):
            verify_peer_certificate(cert, "future_node_30d", pinned_cert=cert_pem)

    def test_malformed_and_corrupt_certificate_inputs(self) -> None:
        """Malformed, truncated, or random bytes must raise CertificateVerificationError."""
        bad_inputs = [
            b"",
            b"\x00" * 32,
            b"RANDOM_NON_CERTIFICATE_BYTES_FUZZING_DATA",
            b"-----BEGIN CERTIFICATE-----\ninvalid_base64===\n-----END CERTIFICATE-----",
            b"-----BEGIN CERTIFICATE-----\nMIIC",  # Truncated PEM
            b"\x30\x82\x01\x0a",  # Truncated ASN.1 DER sequence
        ]
        for bad_data in bad_inputs:
            with pytest.raises(CertificateVerificationError, match="Failed to parse peer certificate"):
                verify_peer_certificate(bad_data, "any_device_id")

    def test_subject_common_name_spoofing_rejected(self) -> None:
        """Mismatched Subject Common Name (CN) must strictly raise CertificateVerificationError."""
        key = generate_device_key()
        cert = generate_self_signed_cert(key, "genuine_device_id")
        cert_pem = dump_certificate_pem(cert)

        with pytest.raises(CertificateVerificationError, match="does not match expected device ID"):
            verify_peer_certificate(cert, "impostor_device_id", pinned_cert=cert_pem)

    def test_subject_common_name_injection_and_traversal(self) -> None:
        """Adversarial CN values with path traversal or strange characters are properly evaluated."""
        key = generate_device_key()
        cert = generate_self_signed_cert(key, "../../etc/shadow")
        cert_pem = dump_certificate_pem(cert)

        # Mismatch against legitimate ID
        with pytest.raises(CertificateVerificationError):
            verify_peer_certificate(cert, "valid_device_id", pinned_cert=cert_pem)

        # Exact match succeeds on identity check (subject to valid validity window)
        assert verify_peer_certificate(cert, "../../etc/shadow", pinned_cert=cert_pem) is True

    def test_pinned_certificate_fingerprint_mismatch(self) -> None:
        """Valid cert matching expected CN but different from pinned cert must raise error."""
        key1 = generate_device_key()
        key2 = generate_device_key()
        cert1 = generate_self_signed_cert(key1, "shared_node_id")
        cert2 = generate_self_signed_cert(key2, "shared_node_id")
        cert2_pem = dump_certificate_pem(cert2)

        with pytest.raises(CertificateVerificationError, match="does not match pinned fingerprint"):
            verify_peer_certificate(cert1, "shared_node_id", pinned_cert=cert2_pem)

    def test_valid_certificate_within_window_succeeds(self) -> None:
        """Standard self-signed certificate within validity window passes verification."""
        key = generate_device_key()
        cert = generate_self_signed_cert(key, "node_valid_active", validity_days=10)
        cert_pem = dump_certificate_pem(cert)

        assert verify_peer_certificate(cert, "node_valid_active", pinned_cert=cert_pem) is True


class TestAdversarialEphemeralCredentials:
    """Stress tests for ephemeral credentials: file modes, leak elimination, cleanup."""

    def test_ephemeral_credentials_mode_0600_and_parent_0700(self) -> None:
        """Credentials files must be mode 0600 and reside in a mode 0700 runtime directory."""
        key = generate_device_key()
        cert = generate_self_signed_cert(key, "device_perm_audit")
        cert_bytes = dump_certificate_pem(cert)
        key_bytes = dump_private_key_pem(key)

        with ephemeral_tls_credentials(cert_bytes, key_bytes) as (c_path, k_path):
            assert c_path.exists()
            assert k_path.exists()

            c_mode = c_path.stat().st_mode & 0o777
            k_mode = k_path.stat().st_mode & 0o777
            parent_mode = c_path.parent.stat().st_mode & 0o777

            # Must have strict 0600 permissions
            assert c_mode == 0o600
            assert k_mode == 0o600

            # No group or world permissions allowed
            assert (c_mode & 0o077) == 0
            assert (k_mode & 0o077) == 0

            # Parent directory must have mode 0700
            assert parent_mode == 0o700

    def test_ephemeral_credentials_cleanup_on_normal_exit(self) -> None:
        """Ephemeral files must not persist after normal context manager exit."""
        key = generate_device_key()
        cert = generate_self_signed_cert(key, "device_clean_normal")
        c_bytes = dump_certificate_pem(cert)
        k_bytes = dump_private_key_pem(key)

        saved_c, saved_k = None, None
        with ephemeral_tls_credentials(c_bytes, k_bytes) as (c_path, k_path):
            saved_c = c_path
            saved_k = k_path
            assert saved_c.exists()
            assert saved_k.exists()

        assert not saved_c.exists()
        assert not saved_k.exists()

    def test_ephemeral_credentials_cleanup_on_unhandled_exception(self) -> None:
        """Ephemeral files must be purged immediately when an exception occurs in context."""
        key = generate_device_key()
        cert = generate_self_signed_cert(key, "device_clean_exc")
        c_bytes = dump_certificate_pem(cert)
        k_bytes = dump_private_key_pem(key)

        saved_c, saved_k = None, None
        with pytest.raises(RuntimeError, match="Simulated crash during TLS setup"):
            with ephemeral_tls_credentials(c_bytes, k_bytes) as (c_path, k_path):
                saved_c = c_path
                saved_k = k_path
                assert saved_c.exists()
                assert saved_k.exists()
                raise RuntimeError("Simulated crash during TLS setup")

        assert not saved_c.exists()
        assert not saved_k.exists()

    def test_concurrent_ephemeral_credentials_isolation(self) -> None:
        """50 rapid invocations create unique non-colliding paths and leave zero leaked files."""
        key = generate_device_key()
        cert = generate_self_signed_cert(key, "device_concurrent")
        c_bytes = dump_certificate_pem(cert)
        k_bytes = dump_private_key_pem(key)

        paths_observed: set[Path] = set()

        for _ in range(50):
            with ephemeral_tls_credentials(c_bytes, k_bytes) as (c_path, k_path):
                assert c_path not in paths_observed
                assert k_path not in paths_observed
                paths_observed.add(c_path)
                paths_observed.add(k_path)
                assert c_path.exists()
                assert k_path.exists()

        # All observed paths must be completely cleaned up
        for p in paths_observed:
            assert not p.exists()


# ==============================================================================
# 2. CPython ABI Safety & Server TOFU (CLUS-10)
# ==============================================================================

class TestAdversarialAbiSafetyAndServerTofu:
    """Stress tests for CPython ABI safety verification and fail-closed server TOFU."""

    def test_abi_safety_returns_boolean_in_current_environment(self) -> None:
        """Environment ABI safety check must return a concrete boolean."""
        res = _is_cpython_supported_abi()
        assert isinstance(res, bool)

    def test_abi_safety_detects_non_cpython_runtime(self) -> None:
        """Non-CPython runtimes (e.g., PyPy) must be identified as unsupported."""
        with patch.object(sys, "implementation", type("Impl", (), {"name": "pypy"})()):
            assert _is_cpython_supported_abi() is False

    def test_abi_safety_detects_debug_trace_refs(self) -> None:
        """CPython debug builds with Py_TRACE_REFS must be identified as unsupported."""
        with patch.object(sys, "getobjects", create=True):
            assert _is_cpython_supported_abi() is False

        with patch("sysconfig.get_config_var", return_value="1"):
            assert _is_cpython_supported_abi() is False

    def test_abi_safety_detects_free_threaded_gil_disabled(self) -> None:
        """Free-threaded CPython (GIL disabled) must be identified as unsupported."""
        with patch("sysconfig.get_config_var", side_effect=lambda var: "1" if var == "Py_GIL_DISABLED" else None):
            assert _is_cpython_supported_abi() is False

        with patch.object(sys, "_is_gil_enabled", return_value=False, create=True):
            assert _is_cpython_supported_abi() is False

    def test_server_tofu_context_fails_closed_when_abi_unsupported(self) -> None:
        """Server-side TOFU context creation must fail closed if ABI is unsupported."""
        key = generate_device_key()
        cert = generate_self_signed_cert(key, "dev_tofu_fail_abi")
        c_pem = dump_certificate_pem(cert)
        k_pem = dump_private_key_pem(key)

        with patch("pengulink.crypto.tls._is_cpython_supported_abi", return_value=False):
            with pytest.raises(CertificateError, match="Failed to configure secure TOFU server context"):
                create_tofu_ssl_context(cert_pem=c_pem, key_pem=k_pem, is_server=True)

    def test_server_tofu_context_fails_closed_when_libssl_unavailable(self) -> None:
        """Server-side TOFU context creation must fail closed if libssl is unavailable."""
        key = generate_device_key()
        cert = generate_self_signed_cert(key, "dev_tofu_fail_libssl")
        c_pem = dump_certificate_pem(cert)
        k_pem = dump_private_key_pem(key)

        with patch("pengulink.crypto.tls._get_libssl", return_value=None):
            with pytest.raises(CertificateError, match="Failed to configure secure TOFU server context"):
                create_tofu_ssl_context(cert_pem=c_pem, key_pem=k_pem, is_server=True)

    def test_server_tofu_context_missing_credentials_fails_closed(self) -> None:
        """Missing certificate or key PEM must fail immediately without proceeding."""
        with pytest.raises(CertificateError, match="Certificate and private key PEM bytes are required"):
            create_tofu_ssl_context(cert_pem=None, key_pem=None, is_server=True)


# ==============================================================================
# 3. Pairing Replay Cache & State Machine Resilience (CLUS-21)
# ==============================================================================

class TestAdversarialPairingReplayAndState:
    """Stress tests for pairing replay prevention, 300s window boundaries, and tie-breaking."""

    def setup_method(self) -> None:
        _PAIRING_REPLAY_CACHE.clear()

    def test_pairing_timestamp_boundary_exact_drift(self) -> None:
        """Boundary verification at 300.0s (allowed) and 301.0s (expired)."""
        fixed_now = 1_000_000.0
        with patch("pengulink.crypto.pairing.time.time", return_value=fixed_now):
            # At boundary: 300.0s in past is within allowable window (drift == 300.0)
            validate_pairing_timestamp(int(fixed_now - 300.0), max_drift=300.0, device_id="boundary_pass_past")

            # Exceeding boundary: 301s in past is rejected (drift == 301.0)
            with pytest.raises(PairingTimestampExpiredError, match="drifted"):
                validate_pairing_timestamp(int(fixed_now - 301.0), max_drift=300.0, device_id="boundary_fail_past")

            # At boundary: 300.0s in future is within allowable window (drift == 300.0)
            validate_pairing_timestamp(int(fixed_now + 300.0), max_drift=300.0, device_id="boundary_pass_future")

            # Exceeding boundary: 301s in future is rejected (drift == 301.0)
            with pytest.raises(PairingTimestampExpiredError, match="drifted"):
                validate_pairing_timestamp(int(fixed_now + 301.0), max_drift=300.0, device_id="boundary_fail_future")

    def test_pairing_replay_single_duplicate_rejected(self) -> None:
        """Identical (device_id, timestamp) pair must be rejected as a replay."""
        device_id = "replay_target_peer"
        ts = int(time.time())

        # First request succeeds
        validate_pairing_timestamp(ts, device_id=device_id)

        # Immediate re-submission must fail
        with pytest.raises(PairingTimestampExpiredError, match="Pairing replay detected"):
            validate_pairing_timestamp(ts, device_id=device_id)

    def test_pairing_replay_duplicate_burst_rejected(self) -> None:
        """Burst of 100 duplicate packets: exactly 1 accepted, 99 rejected."""
        device_id = "burst_target_peer"
        ts = int(time.time())

        # First packet accepted
        validate_pairing_timestamp(ts, device_id=device_id)

        # Next 99 duplicate packets in the burst must all raise PairingTimestampExpiredError
        rejection_count = 0
        for _ in range(99):
            try:
                validate_pairing_timestamp(ts, device_id=device_id)
            except PairingTimestampExpiredError:
                rejection_count += 1

        assert rejection_count == 99

    def test_pairing_replay_device_isolation(self) -> None:
        """Identical timestamp for different device IDs must not collide."""
        ts = int(time.time())
        validate_pairing_timestamp(ts, device_id="peer_alpha")
        validate_pairing_timestamp(ts, device_id="peer_beta")
        # Both alpha and beta succeeded for the same timestamp

    def test_pairing_replay_cache_eviction_under_pressure(self) -> None:
        """Replay cache must evict oldest entries when exceeding capacity (1024 entries)."""
        now = time.time()
        for i in range(1100):
            validate_pairing_timestamp(int(now), device_id=f"node_pressure_{i:04d}")

        # Cache must not grow unbounded (capped at MAX_SIZE + 1)
        assert len(_PAIRING_REPLAY_CACHE) <= 1025

    def test_pairing_session_deterministic_tie_breaking_symmetric(self) -> None:
        """When two endpoints initiate pairing concurrently, both derive the exact same code."""
        key_a = "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA1111"
        key_b = "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA2222"
        ts_a = int(time.time()) - 10
        ts_b = int(time.time()) - 5

        # Node A ("aaa") initiates first with ts_a
        session_a = PairingSession("aaa", "bbb", key_a, key_b)
        session_a.pairing_requested = True
        session_a.pairing_timestamp = ts_a

        # Node B ("bbb") initiates concurrently with ts_b
        session_b = PairingSession("bbb", "aaa", key_b, key_a)
        session_b.pairing_requested = True
        session_b.pairing_timestamp = ts_b

        # A receives B's packet
        pkt_from_b = Packet(type="kdeconnect.pair", body={"pair": True, "timestamp": ts_b})
        session_a.handle_incoming_pair_packet(pkt_from_b)

        # B receives A's packet
        pkt_from_a = Packet(type="kdeconnect.pair", body={"pair": True, "timestamp": ts_a})
        session_b.handle_incoming_pair_packet(pkt_from_a)

        # "aaa" < "bbb", so "aaa" is master. Both sessions must converge on ts_a
        assert session_a.pairing_timestamp == ts_a
        assert session_b.pairing_timestamp == ts_a

        # Both sessions must calculate the identical visual verification code
        assert session_a.verification_code == session_b.verification_code

    def test_pairing_session_strict_boolean_validation_fuzzing(self) -> None:
        """Only literal True is accepted as valid pair request; non-booleans must be rejected."""
        session = PairingSession("local_node", "remote_node", "key1", "key2")

        invalid_bodies = [
            "false",
            "true",
            "True",
            "False",
            1,
            0,
            1.0,
            0.0,
            None,
            [],
            [True],
            {},
            {"pair": True},
        ]
        for val in invalid_bodies:
            pkt = Packet(type="kdeconnect.pair", body={"pair": val, "timestamp": int(time.time())})
            with pytest.raises(PairingRejectedError, match="Pairing rejected"):
                session.handle_incoming_pair_packet(pkt)


# ==============================================================================
# 4. Transport Framing & Starvation Defenses (CLUS-22)
# ==============================================================================

class TestAdversarialTransportFraming:
    """Stress tests for TCP framing, starvation defense, and EOF truncation."""

    def test_collision_tie_breaker_heterogeneous_types(self) -> None:
        """Comparison between heterogeneous types must not raise TypeError."""
        assert _should_drop_inbound_collision("aaa", "bbb") is True
        assert _should_drop_inbound_collision("bbb", "aaa") is False
        assert _should_drop_inbound_collision(12345, 67890) is True
        assert _should_drop_inbound_collision(99999, 11111) is False
        assert _should_drop_inbound_collision("uuid_1", 100) is False

    async def test_stream_line_reader_empty_line_flood_starvation_defense(self) -> None:
        """Burst of >100 empty lines must trigger starvation defense raising PacketFramingError."""
        reader = StreamLineReader(max_consecutive_empty_lines=100)
        # Feed 105 consecutive empty lines
        reader.feed_data(b"\n" * 105)

        with pytest.raises(PacketFramingError, match="Empty line flood detected: exceeded 100 consecutive blank lines"):
            for _ in range(105):
                await reader.read_line()

    async def test_stream_line_reader_empty_line_flood_default_threshold(self) -> None:
        """Default threshold (50) correctly stops an empty line flood."""
        reader = StreamLineReader()
        reader.feed_data(b"\r\n" * 55)

        with pytest.raises(PacketFramingError, match="Empty line flood detected"):
            for _ in range(55):
                await reader.read_line()

    async def test_stream_line_reader_intermittent_empty_lines_reset(self) -> None:
        """Valid non-empty packets interspersed with empty lines reset the starvation counter."""
        reader = StreamLineReader(max_consecutive_empty_lines=30)
        pkt1 = Packet(type="kdeconnect.ping", body={}).to_wire()
        pkt2 = Packet(type="kdeconnect.ping", body={}).to_wire()

        # Feed 20 empty lines, pkt1, 20 empty lines, pkt2
        reader.feed_data((b"\n" * 20) + pkt1 + (b"\n" * 20) + pkt2)

        res1 = await reader.read_packet()
        assert res1 is not None
        assert res1.type == "kdeconnect.ping"

        res2 = await reader.read_packet()
        assert res2 is not None
        assert res2.type == "kdeconnect.ping"

    async def test_stream_line_reader_truncated_line_eof_detection(self) -> None:
        """Stream closing before newline termination must raise TruncatedPacketError."""
        reader = StreamLineReader()
        reader.feed_data(b'{"id": 42, "type": "kdeconnect.ping"')
        reader.feed_eof()

        with pytest.raises(TruncatedPacketError, match="Incomplete packet frame: EOF reached without newline"):
            await reader.read_line()

    async def test_stream_line_reader_clean_eof(self) -> None:
        """Stream closing at clean boundary returns empty bytes on read_line and None on read_packet."""
        reader = StreamLineReader()
        reader.feed_data(b'{"id": 1, "type": "kdeconnect.ping", "body": {}}\n')
        reader.feed_eof()

        pkt = await reader.read_packet()
        assert pkt is not None
        assert pkt.id == 1

        # Subsequent read returns None on clean EOF
        next_pkt = await reader.read_packet()
        assert next_pkt is None

    async def test_stream_line_reader_buffer_overrun_128kb(self) -> None:
        """Stream sending >128 KB without newline triggers PacketBufferOverrunError (CVE-2020-26164)."""
        reader = StreamLineReader()
        # Feed 130 KB without newline
        reader.feed_data(b"A" * (MAX_PACKET_SIZE + 1024))

        with pytest.raises(PacketBufferOverrunError, match="Stream buffer overrun exceeded 131072 bytes"):
            await reader.read_line()

    async def test_stream_line_reader_consecutive_malformed_packets_limit(self) -> None:
        """Exceeding 5 consecutive malformed packets terminates stream with MalformedPacketError."""
        reader = StreamLineReader(max_consecutive_errors=5)
        bad_lines = b"NOT_JSON_1\nNOT_JSON_2\nNOT_JSON_3\nNOT_JSON_4\nNOT_JSON_5\n"
        reader.feed_data(bad_lines)

        with pytest.raises(MalformedPacketError, match="Exceeded maximum allowable consecutive malformed packets"):
            await reader.read_packet()

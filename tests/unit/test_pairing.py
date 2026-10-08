"""Unit tests for pengulink.crypto.pairing."""

from __future__ import annotations

import time
import pytest

from pengulink.core.exceptions import (
    PairingError,
    PairingRejectedError,
    PairingTimestampExpiredError,
)
from pengulink.crypto.pairing import (
    PairingSession,
    compute_verification_code,
    create_pair_packet,
    validate_pairing_timestamp,
)
from pengulink.network.packet import Packet


def test_create_pair_packet() -> None:
    p_req = create_pair_packet(pair=True, timestamp=1600000000)
    assert p_req.type == "kdeconnect.pair"
    assert p_req.body["pair"] is True
    assert p_req.body["timestamp"] == 1600000000

    p_rej = create_pair_packet(pair=False)
    assert p_rej.type == "kdeconnect.pair"
    assert p_rej.body["pair"] is False


def test_compute_verification_code_order_independent() -> None:
    key_alice = "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A...\n-----END PUBLIC KEY-----"
    key_bob = "-----BEGIN PUBLIC KEY-----\nMIICITANBgkqhkiG9w0BAQEFAAOCAQ8B...\n-----END PUBLIC KEY-----"
    ts = 1690000000

    code_from_alice = compute_verification_code(key_alice, key_bob, ts)
    code_from_bob = compute_verification_code(key_bob, key_alice, ts)

    assert code_from_alice == code_from_bob
    assert len(code_from_alice) == 8
    assert code_from_alice.isupper()


def test_validate_pairing_timestamp() -> None:
    now = int(time.time())
    # Recent timestamp should succeed
    validate_pairing_timestamp(now - 10)

    # Stale timestamp (e.g. 1 hour old) should raise
    with pytest.raises(PairingTimestampExpiredError) as exc_info:
        validate_pairing_timestamp(now - 3600, max_drift=1800.0)
    assert "drifted" in str(exc_info.value)


def test_pairing_session_full_flow() -> None:
    alice_key = "PUBKEY_ALICE"
    bob_key = "PUBKEY_BOB"

    alice_session = PairingSession(
        local_device_id="alice_phone",
        remote_device_id="bob_linux",
        local_pubkey_pem=alice_key,
        peer_pubkey_pem=bob_key,
    )
    bob_session = PairingSession(
        local_device_id="bob_linux",
        remote_device_id="alice_phone",
        local_pubkey_pem=bob_key,
        peer_pubkey_pem=alice_key,
    )

    # 1. Alice initiates pairing
    alice_req_packet = alice_session.initiate_pairing()
    assert alice_session.verification_code is not None

    # 2. Bob receives Alice's request
    accepted, bob_code = bob_session.handle_incoming_pair_packet(alice_req_packet)
    assert accepted is True

    # 3. Verification codes match on both screens!
    assert bob_code == alice_session.verification_code

    # 4. Bob accepts and sends pair packet back
    bob_accept_packet = bob_session.accept_pairing()
    assert bob_accept_packet.body["pair"] is True


def test_pairing_session_rejection() -> None:
    alice_session = PairingSession(
        local_device_id="alice_phone",
        remote_device_id="bob_linux",
        local_pubkey_pem="PUB_A",
        peer_pubkey_pem="PUB_B",
    )

    reject_packet = Packet(type="kdeconnect.pair", body={"pair": False})
    with pytest.raises(PairingRejectedError):
        alice_session.handle_incoming_pair_packet(reject_packet)

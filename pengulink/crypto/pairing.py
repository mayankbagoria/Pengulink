"""pengulink.crypto.pairing
~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect pairing packet creation, verification code derivation,
and pairing negotiation protocol handler.
"""

from __future__ import annotations

import collections
import hashlib
import logging
import time
from typing import Any, Dict, Optional, Tuple

from pengulink.core.exceptions import (
    AuthenticationCodeMismatchError,
    PairingError,
    PairingRejectedError,
    PairingTimestampExpiredError,
)
from pengulink.network.packet import Packet

logger = logging.getLogger("pengulink.crypto.pairing")

PAIR_PACKET_TYPE = "kdeconnect.pair"
MAX_TIMESTAMP_DRIFT_SECONDS = 300.0  # 5-minute drift window for replay protection
_PAIRING_REPLAY_CACHE: collections.OrderedDict[Tuple[str, int], float] = collections.OrderedDict()
_REPLAY_CACHE_MAX_SIZE = 1024


def create_pair_packet(pair: bool, timestamp: Optional[int] = None) -> Packet:
    """Create a standardized kdeconnect.pair packet.

    Args:
        pair: True to request/accept pairing; False to reject or unpair.
        timestamp: Optional UNIX timestamp in seconds. Defaults to current time.
    """
    ts = timestamp if timestamp is not None else int(time.time())
    return Packet(
        type=PAIR_PACKET_TYPE,
        body={
            "pair": bool(pair),
            "timestamp": ts,
        },
    )


def compute_verification_code(
    local_pubkey_pem: str,
    peer_pubkey_pem: str,
    timestamp: int,
) -> str:
    """Compute standard 8-character visual verification code.

    KDE Connect algorithm:
      1. Normalize and strip trailing whitespace from both PEM public keys.
      2. Lexicographically sort both keys so derivation is order-independent.
      3. Concatenate sorted[0] + sorted[1] + str(timestamp).
      4. Compute SHA-256 and return first 8 uppercase hex characters.
    """
    key_a = local_pubkey_pem.strip()
    key_b = peer_pubkey_pem.strip()

    sorted_keys = sorted([key_a, key_b])
    payload = f"{sorted_keys[0]}{sorted_keys[1]}{timestamp}".encode("utf-8")
    full_digest = hashlib.sha256(payload).hexdigest()
    return full_digest[:8].upper()


def validate_pairing_timestamp(
    timestamp: int,
    max_drift: float = MAX_TIMESTAMP_DRIFT_SECONDS,
    device_id: Optional[str] = None,
) -> None:
    """Ensure pairing packet timestamp is within allowable drift window and check for replay."""
    now = time.time()
    drift = abs(now - timestamp)
    if drift > max_drift:
        raise PairingTimestampExpiredError(
            f"Pairing timestamp {timestamp} drifted by {drift:.1f}s (max allowed: {max_drift}s)",
            context={"timestamp": timestamp, "drift": drift, "max_drift": max_drift},
        )

    # Replay protection cache (keyed on (device_id, timestamp))
    cache_key = (str(device_id or "global"), int(timestamp))
    cutoff = now - max_drift
    while _PAIRING_REPLAY_CACHE:
        oldest_k, oldest_time = next(iter(_PAIRING_REPLAY_CACHE.items()))
        if oldest_time < cutoff or len(_PAIRING_REPLAY_CACHE) > _REPLAY_CACHE_MAX_SIZE:
            _PAIRING_REPLAY_CACHE.pop(oldest_k, None)
        else:
            break

    if cache_key in _PAIRING_REPLAY_CACHE:
        raise PairingTimestampExpiredError(
            f"Pairing replay detected: timestamp {timestamp} was already processed for {cache_key[0]}",
            context={"timestamp": timestamp, "device_id": device_id},
        )
    _PAIRING_REPLAY_CACHE[cache_key] = now


class PairingSession:
    """Manages pairing negotiation state for a single connection."""

    def __init__(
        self,
        local_device_id: str,
        remote_device_id: str,
        local_pubkey_pem: str,
        peer_pubkey_pem: str,
    ) -> None:
        self.local_device_id = local_device_id
        self.remote_device_id = remote_device_id
        self.local_pubkey_pem = local_pubkey_pem
        self.peer_pubkey_pem = peer_pubkey_pem

        self.pairing_requested: bool = False
        self.pairing_accepted: bool = False
        self.pairing_timestamp: Optional[int] = None
        self.verification_code: Optional[str] = None

    def initiate_pairing(self) -> Packet:
        """Generate outbound pairing request packet and calculate verification code."""
        ts = int(time.time())
        self.pairing_requested = True
        self.pairing_timestamp = ts
        self.verification_code = compute_verification_code(
            self.local_pubkey_pem,
            self.peer_pubkey_pem,
            ts,
        )
        logger.info(
            "Initiating pairing with %s. Verification code: %s",
            self.remote_device_id,
            self.verification_code,
        )
        return create_pair_packet(pair=True, timestamp=ts)

    def handle_incoming_pair_packet(self, packet: Packet) -> Tuple[bool, Optional[str]]:
        """Process an incoming kdeconnect.pair packet.

        Returns:
            Tuple of (pair_accepted, verification_code).
        Raises:
            PairingRejectedError if pair is False or not boolean True.
            PairingTimestampExpiredError if timestamp is stale or replayed.
        """
        if packet.type != PAIR_PACKET_TYPE:
            raise PairingError(f"Expected packet type '{PAIR_PACKET_TYPE}', got '{packet.type}'")

        body = packet.body
        raw_pair = body.get("pair")
        # Strict boolean validation: string "false", False, None, etc. must evaluate to False
        is_pair = (raw_pair is True)
        if not is_pair:
            logger.info("Pairing request was rejected or revoked by %s", self.remote_device_id)
            raise PairingRejectedError(
                f"Pairing rejected by remote peer '{self.remote_device_id}'",
                context={"remote_device_id": self.remote_device_id},
            )

        ts = body.get("timestamp")
        if ts is not None:
            validate_pairing_timestamp(int(ts), max_drift=MAX_TIMESTAMP_DRIFT_SECONDS, device_id=self.remote_device_id)

        # Deterministic tie-breaking on device ID for concurrent pairing initiation
        if self.pairing_requested and self.pairing_timestamp is not None:
            # Both endpoints initiated pairing concurrently.
            # Master is deterministically the node with smaller device ID.
            if str(self.local_device_id) < str(self.remote_device_id):
                # We are master: keep our own timestamp
                chosen_ts = self.pairing_timestamp
            else:
                # Remote is master: adopt remote timestamp
                chosen_ts = int(ts) if ts is not None else self.pairing_timestamp
            self.pairing_timestamp = chosen_ts
        else:
            if ts is not None:
                self.pairing_timestamp = int(ts)
            elif self.pairing_timestamp is None:
                self.pairing_timestamp = int(time.time())

        self.verification_code = compute_verification_code(
            self.local_pubkey_pem,
            self.peer_pubkey_pem,
            self.pairing_timestamp,
        )
        self.pairing_accepted = True

        logger.info(
            "Pairing packet processed from %s. Verification code: %s",
            self.remote_device_id,
            self.verification_code,
        )
        return True, self.verification_code

    def accept_pairing(self) -> Packet:
        """Confirm pairing acceptance to the remote peer."""
        ts = self.pairing_timestamp or int(time.time())
        return create_pair_packet(pair=True, timestamp=ts)

    def reject_pairing(self) -> Packet:
        """Reject or revoke pairing."""
        return create_pair_packet(pair=False, timestamp=int(time.time()))

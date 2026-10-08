"""pengulink.crypto
~~~~~~~~~~~~~~~~
KDE Connect Cryptography, X.509 Certificates, Mutual TLS, and Pairing.
"""

from __future__ import annotations

from pengulink.crypto.certificate import (
    dump_certificate_pem,
    dump_private_key_pem,
    ensure_credentials,
    generate_device_key,
    generate_self_signed_cert,
    get_certificate_common_name,
    get_certificate_fingerprint,
    load_certificate_pem,
    load_credentials,
    load_private_key_pem,
    save_credentials,
)
from pengulink.crypto.pairing import (
    PairingSession,
    compute_verification_code,
    create_pair_packet,
    validate_pairing_timestamp,
)
from pengulink.crypto.tls import (
    create_paired_ssl_context,
    create_tofu_ssl_context,
    verify_peer_certificate,
)

__all__ = [
    # Certificate & Key
    "generate_device_key",
    "generate_self_signed_cert",
    "get_certificate_fingerprint",
    "get_certificate_common_name",
    "dump_private_key_pem",
    "load_private_key_pem",
    "dump_certificate_pem",
    "load_certificate_pem",
    "save_credentials",
    "load_credentials",
    "ensure_credentials",
    # TLS
    "create_paired_ssl_context",
    "create_tofu_ssl_context",
    "verify_peer_certificate",
    # Pairing
    "create_pair_packet",
    "compute_verification_code",
    "validate_pairing_timestamp",
    "PairingSession",
]

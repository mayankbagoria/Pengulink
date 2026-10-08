"""pengulink.crypto.tls
~~~~~~~~~~~~~~~~~~~~
KDE Connect Mutual TLS (mTLS) Context Factory and Certificate Pinning.

In KDE Connect:
- Outbound TCP connector acts as TLS Server (server_side=True).
- Inbound TCP listener acts as TLS Client (server_side=False).
- Paired sessions enforce strict certificate pinning.
- Peer Certificate CN must strictly match deviceId (CVE-2020-26164 defense).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import ssl
import tempfile
from typing import Optional, Union

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization

from pengulink.core.exceptions import (
    CertificateError,
    CertificateVerificationError,
    TLSUpgradeError,
)
from pengulink.crypto.certificate import (
    get_certificate_common_name,
    get_certificate_fingerprint,
    load_certificate_pem,
)

logger = logging.getLogger("pengulink.crypto.tls")

_VERIFY_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
_C_ALWAYS_OK = _VERIFY_CB(lambda pre, ctx: 1)
_SSL_VERIFY_PEER = 0x01


class _PySSLContextStruct(ctypes.Structure):
    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("ctx", ctypes.c_void_p),
    ]


_libssl_handle: Optional[ctypes.CDLL] = None
_libssl_searched: bool = False


def _get_libssl() -> Optional[ctypes.CDLL]:
    global _libssl_handle, _libssl_searched
    if _libssl_searched:
        return _libssl_handle
    _libssl_searched = True

    candidates = [
        ctypes.util.find_library("ssl"),
        ctypes.util.find_library("libssl"),
        "libssl.so.3",
        "libssl.so.1.1",
        "libssl.so",
        "libssl.dylib",
        "libssl-3-x64.dll",
        "libssl-1_1-x64.dll",
        "ssleay32.dll",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            lib = ctypes.CDLL(candidate)
            if hasattr(lib, "SSL_CTX_set_verify"):
                _libssl_handle = lib
                break
        except (OSError, Exception):
            continue
    return _libssl_handle


def _configure_server_tofu_verify(ctx: ssl.SSLContext) -> bool:
    """Configure server SSLContext to request peer certificate without failing on self-signed certs."""
    libssl = _get_libssl()
    if not libssl:
        logger.warning("libssl not available; cannot configure server TOFU peer certificate request")
        return False
    try:
        struct = _PySSLContextStruct.from_address(id(ctx))
        if not struct.ctx:
            logger.warning("PySSLContextStruct.ctx is NULL")
            return False
        libssl.SSL_CTX_set_verify(struct.ctx, _SSL_VERIFY_PEER, _C_ALWAYS_OK)
        return True
    except Exception as exc:
        logger.warning("Failed to configure SSL_CTX_set_verify via ctypes: %s", exc)
        return False


def create_paired_ssl_context(
    local_cert_pem: bytes,
    local_key_pem: bytes,
    peer_cert_pem: bytes,
    server_side: bool,
) -> ssl.SSLContext:
    """Construct an SSLContext enforcing strict certificate pinning for paired peers.

    Args:
        local_cert_pem: Local X.509 certificate PEM bytes.
        local_key_pem: Local RSA private key PEM bytes.
        peer_cert_pem: Stored trusted peer certificate PEM bytes.
        server_side: True if this node acts as TLS server (KDE Connect outbound connection).
    """
    proto = ssl.PROTOCOL_TLS_SERVER if server_side else ssl.PROTOCOL_TLS_CLIENT
    ctx = ssl.SSLContext(proto)

    # Minimum TLS 1.2, allow TLS 1.3
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    # Load local credentials via temporary files (Python SSLContext requires paths or buffers)
    with tempfile.NamedTemporaryFile("wb", delete=False) as f_cert, \
         tempfile.NamedTemporaryFile("wb", delete=False) as f_key:
        try:
            f_cert.write(local_cert_pem)
            f_cert.flush()
            f_key.write(local_key_pem)
            f_key.flush()
            ctx.load_cert_chain(f_cert.name, f_key.name)
        finally:
            import os
            try:
                os.remove(f_cert.name)
            except OSError:
                pass
            try:
                os.remove(f_key.name)
            except OSError:
                pass

    # Pin peer certificate
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED

    # Load peer cert as trusted CA / trust anchor
    try:
        ctx.load_verify_locations(cadata=peer_cert_pem.decode("utf-8"))
    except Exception as exc:
        raise CertificateError(f"Failed to load pinned peer certificate into SSLContext: {exc}") from exc

    return ctx


def create_tofu_ssl_context(
    local_cert_pem: bytes,
    local_key_pem: bytes,
    server_side: bool,
) -> ssl.SSLContext:
    """Construct an SSLContext for unpaired devices during initial pairing (TOFU).

    Peer certificate is captured for user inspection and subsequent pinning.
    """
    proto = ssl.PROTOCOL_TLS_SERVER if server_side else ssl.PROTOCOL_TLS_CLIENT
    ctx = ssl.SSLContext(proto)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    with tempfile.NamedTemporaryFile("wb", delete=False) as f_cert, \
         tempfile.NamedTemporaryFile("wb", delete=False) as f_key:
        try:
            f_cert.write(local_cert_pem)
            f_cert.flush()
            f_key.write(local_key_pem)
            f_key.flush()
            ctx.load_cert_chain(f_cert.name, f_key.name)
        finally:
            import os
            try:
                os.remove(f_cert.name)
            except OSError:
                pass
            try:
                os.remove(f_key.name)
            except OSError:
                pass

    ctx.check_hostname = False
    # Unpaired connections permit self-signed peer certificates for TOFU capture
    ctx.verify_mode = ssl.CERT_NONE

    if server_side:
        _configure_server_tofu_verify(ctx)

    return ctx


def verify_peer_certificate(
    peer_cert_bytes: bytes,
    expected_device_id: str,
    pinned_cert_pem: Optional[bytes] = None,
) -> x509.Certificate:
    """Validate a peer's X.509 certificate against device ID and pinned trust.

    CVE-2020-26164 Defense:
    1. Extracts Common Name (CN) from Subject attributes.
    2. CN must strictly match expected_device_id.
    3. If pinned_cert_pem is provided, peer cert must match pinned cert.
    """
    try:
        if peer_cert_bytes.startswith(b"-----BEGIN CERTIFICATE-----"):
            peer_cert = load_certificate_pem(peer_cert_bytes)
        else:
            peer_cert = x509.load_der_x509_certificate(peer_cert_bytes)
    except Exception as exc:
        raise CertificateVerificationError(
            f"Failed to parse peer certificate: {exc}",
            expected_device_id=expected_device_id,
        ) from exc

    # 1. Verify Common Name (CN) matches expected deviceId
    cn = get_certificate_common_name(peer_cert)
    if not cn or cn != expected_device_id:
        raise CertificateVerificationError(
            f"Certificate CN '{cn}' does not match expected device ID '{expected_device_id}'",
            expected_device_id=expected_device_id,
            presented_cn=cn,
        )

    # 2. Verify pinned certificate if provided
    if pinned_cert_pem:
        pinned_cert = load_certificate_pem(pinned_cert_pem)
        peer_fp = get_certificate_fingerprint(peer_cert)
        pinned_fp = get_certificate_fingerprint(pinned_cert)
        if peer_fp != pinned_fp:
            raise CertificateVerificationError(
                f"Peer certificate fingerprint {peer_fp} does not match pinned fingerprint {pinned_fp}",
                expected_device_id=expected_device_id,
                presented_cn=cn,
                context={"peer_fingerprint": peer_fp, "pinned_fingerprint": pinned_fp},
            )

    return peer_cert

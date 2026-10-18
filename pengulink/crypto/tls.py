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

import contextlib
import ctypes
import ctypes.util
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import ssl
import stat
import sys
import sysconfig
import tempfile
import uuid
from typing import Iterator, Optional, Tuple, Union

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


def _is_cpython_supported_abi() -> bool:
    """Check if the runtime CPython ABI supports ctypes pointer access to PySSLContext."""
    if sys.implementation.name != "cpython":
        return False
    # Check for Py_TRACE_REFS (debug build with object tracing modifies PyObject header)
    if hasattr(sys, "getobjects"):
        return False
    trace_refs = sysconfig.get_config_var("Py_TRACE_REFS")
    if trace_refs and str(trace_refs) not in ("0", "None", ""):
        return False
    # Check for free-threaded Python (PEP 703) which modifies object headers
    gil_disabled = sysconfig.get_config_var("Py_GIL_DISABLED")
    if gil_disabled and str(gil_disabled) not in ("0", "None", ""):
        return False
    if hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled():
        return False
    return True


def _get_secure_runtime_dir() -> Path:
    """Return a dedicated secure directory (0700) for ephemeral TLS credentials."""
    runtime_base = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_base and os.path.isdir(runtime_base):
        base_dir = Path(runtime_base)
    else:
        base_dir = Path(tempfile.gettempdir())
    secure_dir = base_dir / f"pengulink_{os.getuid() if hasattr(os, 'getuid') else 'user'}"
    secure_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(secure_dir, 0o700)
    except OSError:
        pass
    return secure_dir


def _write_secure_temp_file(
    prefix_or_dir: Union[Path, str],
    data_or_prefix: Union[str, bytes],
    data: Optional[bytes] = None,
    mode: int = 0o600,
) -> Path:
    """Write data to a unique temporary file with strict atomic permissions."""
    if data is None:
        # Signature: _write_secure_temp_file(prefix, data, mode=0o600)
        directory = _get_secure_runtime_dir()
        prefix = str(prefix_or_dir)
        raw_data = data_or_prefix if isinstance(data_or_prefix, bytes) else data_or_prefix.encode("utf-8")
    else:
        # Signature: _write_secure_temp_file(directory, prefix, data, mode=0o600)
        directory = Path(prefix_or_dir)
        prefix = str(data_or_prefix)
        raw_data = data if isinstance(data, bytes) else data.encode("utf-8")

    filename = f"{prefix}_{os.getpid()}_{uuid.uuid4().hex}.pem"
    file_path = directory / filename
    fd = os.open(file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
    try:
        with open(fd, "wb", closefd=True) as f:
            f.write(raw_data)
            f.flush()
    except Exception:
        try:
            os.unlink(file_path)
        except OSError:
            pass
        raise
    try:
        os.chmod(file_path, mode)
    except OSError:
        pass
    return file_path


@contextlib.contextmanager
def ephemeral_tls_credentials(
    cert_bytes: bytes,
    key_bytes: bytes,
) -> Iterator[Tuple[Path, Path]]:
    """Context manager creating temporary cert and key files in secure 0700 runtime dir,
    cleaned up on exit.
    """
    secure_dir = _get_secure_runtime_dir()
    cert_path = _write_secure_temp_file(secure_dir, "ephemeral_cert", cert_bytes, mode=0o600)
    key_path = _write_secure_temp_file(secure_dir, "ephemeral_key", key_bytes, mode=0o600)
    try:
        yield cert_path, key_path
    finally:
        try:
            cert_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            key_path.unlink(missing_ok=True)
        except OSError:
            pass


def _configure_server_tofu_verify(ctx: ssl.SSLContext) -> bool:
    """Configure server SSLContext to request peer certificate without failing on self-signed certs."""
    if not _is_cpython_supported_abi():
        logger.warning("Unsupported CPython ABI or debug build; cannot safely configure ctypes PySSLContextStruct")
        return False

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
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    with ephemeral_tls_credentials(local_cert_pem, local_key_pem) as (f_cert, f_key):
        ctx.load_cert_chain(str(f_cert), str(f_key))

    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED

    try:
        cadata = peer_cert_pem.decode("utf-8") if isinstance(peer_cert_pem, bytes) else peer_cert_pem
        ctx.load_verify_locations(cadata=cadata)
    except Exception as exc:
        raise CertificateError(f"Failed to load pinned peer certificate into SSLContext: {exc}") from exc

    return ctx


def create_tofu_ssl_context(
    local_cert_pem: Optional[bytes] = None,
    local_key_pem: Optional[bytes] = None,
    server_side: Optional[bool] = None,
    *,
    cert_pem: Optional[bytes] = None,
    key_pem: Optional[bytes] = None,
    pinned_cert_pem: Optional[bytes] = None,
    is_server: Optional[bool] = None,
) -> ssl.SSLContext:
    """Construct an SSLContext for unpaired devices during initial pairing (TOFU).

    Peer certificate is captured for user inspection and subsequent pinning.
    """
    cert = local_cert_pem if local_cert_pem is not None else cert_pem
    key = local_key_pem if local_key_pem is not None else key_pem
    if cert is None or key is None:
        raise CertificateError("Certificate and private key PEM bytes are required")

    is_srv = server_side if server_side is not None else (is_server if is_server is not None else True)

    proto = ssl.PROTOCOL_TLS_SERVER if is_srv else ssl.PROTOCOL_TLS_CLIENT
    ctx = ssl.SSLContext(proto)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    with ephemeral_tls_credentials(cert, key) as (f_cert, f_key):
        ctx.load_cert_chain(str(f_cert), str(f_key))

    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    if is_srv:
        configured = _configure_server_tofu_verify(ctx)
        if not configured:
            raise CertificateError(
                "Failed to configure secure TOFU server context: unable to request client certificate "
                "without strict CA verification on this platform/ABI."
            )

    return ctx


def verify_peer_certificate(
    peer_cert_or_bytes: Union[bytes, x509.Certificate],
    expected_device_id: str,
    pinned_cert_pem: Optional[Union[bytes, str]] = None,
    *,
    pinned_cert: Optional[Union[bytes, str]] = None,
) -> bool:
    """Validate a peer's X.509 certificate against device ID, validity window, and pinned trust.

    CVE-2020-26164 Defense:
    1. Extracts Common Name (CN) from Subject attributes.
    2. CN must strictly match expected_device_id.
    3. Certificate must be valid according to current UTC time.
    4. If pinned_cert_pem is provided, peer cert must match pinned cert.
    """
    effective_pinned = pinned_cert_pem if pinned_cert_pem is not None else pinned_cert
    if isinstance(effective_pinned, str):
        effective_pinned = effective_pinned.encode("utf-8")

    try:
        if isinstance(peer_cert_or_bytes, x509.Certificate):
            peer_cert = peer_cert_or_bytes
        elif peer_cert_or_bytes.startswith(b"-----BEGIN CERTIFICATE-----"):
            peer_cert = load_certificate_pem(peer_cert_or_bytes)
        else:
            peer_cert = x509.load_der_x509_certificate(peer_cert_or_bytes)
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

    # 2. Verify certificate validity window (UTC expiration check)
    now_utc = datetime.now(timezone.utc)
    try:
        not_before = peer_cert.not_valid_before_utc
        not_after = peer_cert.not_valid_after_utc
    except AttributeError:
        not_before = peer_cert.not_valid_before.replace(tzinfo=timezone.utc)
        not_after = peer_cert.not_valid_after.replace(tzinfo=timezone.utc)

    if now_utc < not_before:
        raise CertificateVerificationError(
            f"Peer certificate is not yet valid (valid from {not_before}, current UTC time is {now_utc})",
            expected_device_id=expected_device_id,
            presented_cn=cn,
            context={"not_valid_before": str(not_before), "current_time": str(now_utc)},
        )

    if now_utc > not_after:
        raise CertificateVerificationError(
            f"Peer certificate has expired (expired at {not_after}, current UTC time is {now_utc})",
            expected_device_id=expected_device_id,
            presented_cn=cn,
            context={"not_valid_after": str(not_after), "current_time": str(now_utc)},
        )

    # 3. Verify pinned certificate if provided
    if effective_pinned:
        pinned_c = load_certificate_pem(effective_pinned)
        peer_fp = get_certificate_fingerprint(peer_cert)
        pinned_fp = get_certificate_fingerprint(pinned_c)
        if peer_fp != pinned_fp:
            raise CertificateVerificationError(
                f"Peer certificate fingerprint {peer_fp} does not match pinned fingerprint {pinned_fp}",
                expected_device_id=expected_device_id,
                presented_cn=cn,
                context={"peer_fingerprint": peer_fp, "pinned_fingerprint": pinned_fp},
            )

    return True


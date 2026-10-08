"""pengulink.crypto.certificate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RSA 2048-bit key generation, X.509 self-signed certificate creation,
SHA-256 fingerprint generation, and secure POSIX 0600 persistence.
"""

from __future__ import annotations

import datetime
import logging
import os
import stat
from pathlib import Path
from typing import Optional, Tuple, Union

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from pengulink.core.exceptions import (
    CertificateError,
    CertificatePermissionError,
    KeyGenerationError,
)

logger = logging.getLogger("pengulink.crypto.certificate")


def generate_device_key(key_size: int = 2048, public_exponent: int = 65537) -> rsa.RSAPrivateKey:
    """Generate a cryptographically secure RSA private key.

    KDE Connect protocol mandates 2048-bit RSA keys with exponent 65537.
    """
    try:
        return rsa.generate_private_key(
            public_exponent=public_exponent,
            key_size=key_size,
        )
    except Exception as exc:
        raise KeyGenerationError(f"Failed to generate RSA private key: {exc}") from exc


def generate_self_signed_cert(
    private_key: rsa.RSAPrivateKey,
    device_id: str,
    validity_days: int = 3650,
) -> x509.Certificate:
    """Generate a self-signed X.509 v3 certificate for the local device.

    Subject and Issuer Common Name (CN) are both set to device_id.
    Valid for 10 years (~3650 days) with a 5-minute skew tolerance backward.
    """
    if not device_id or not isinstance(device_id, str):
        raise CertificateError("device_id must be a non-empty string for certificate generation")

    try:
        name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, device_id)
        ])
        now = datetime.datetime.now(datetime.timezone.utc)

        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=validity_days))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True,
            )
            .sign(private_key, hashes.SHA256())
        )
        return cert
    except Exception as exc:
        raise CertificateError(f"Failed to build self-signed certificate: {exc}") from exc


def get_certificate_fingerprint(cert: x509.Certificate) -> str:
    """Compute uppercase colon-separated SHA-256 fingerprint for certificate verification."""
    try:
        raw_fingerprint = cert.fingerprint(hashes.SHA256())
        return raw_fingerprint.hex(":").upper()
    except Exception as exc:
        raise CertificateError(f"Failed to compute certificate fingerprint: {exc}") from exc


def get_certificate_common_name(cert: x509.Certificate) -> Optional[str]:
    """Extract Common Name (CN) from certificate Subject attributes."""
    try:
        attrs = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        if not attrs:
            return None
        val = attrs[0].value
        return str(val) if isinstance(val, str) else val.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("Failed extracting Common Name from certificate: %s", exc)
        return None


def dump_private_key_pem(key: rsa.RSAPrivateKey) -> bytes:
    """Serialize RSA private key to PKCS#8 PEM format without passphrase."""
    try:
        return key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    except Exception as exc:
        raise CertificateError(f"Failed to dump private key to PEM: {exc}") from exc


def load_private_key_pem(data: bytes) -> rsa.RSAPrivateKey:
    """Load an RSA private key from PEM bytes."""
    try:
        key = serialization.load_pem_private_key(data, password=None)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise CertificateError("Loaded key is not an RSA private key")
        return key
    except Exception as exc:
        raise CertificateError(f"Failed to parse private key PEM: {exc}") from exc


def dump_certificate_pem(cert: x509.Certificate) -> bytes:
    """Serialize X.509 certificate to PEM format."""
    try:
        return cert.public_bytes(serialization.Encoding.PEM)
    except Exception as exc:
        raise CertificateError(f"Failed to dump certificate to PEM: {exc}") from exc


def load_certificate_pem(data: bytes) -> x509.Certificate:
    """Load an X.509 certificate from PEM bytes."""
    try:
        return x509.load_pem_x509_certificate(data)
    except Exception as exc:
        raise CertificateError(f"Failed to parse certificate PEM: {exc}") from exc


def save_credentials(
    storage_dir: Union[str, Path],
    private_key: rsa.RSAPrivateKey,
    certificate: x509.Certificate,
) -> Tuple[Path, Path]:
    """Persist private key and certificate atomically with strict POSIX 0600 permissions."""
    target_dir = Path(storage_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    key_path = target_dir / "private_key.pem"
    cert_path = target_dir / "certificate.pem"

    key_pem = dump_private_key_pem(private_key)
    cert_pem = dump_certificate_pem(certificate)

    # 1. Write private key atomically with 0600 permissions
    tmp_key = key_path.with_suffix(".tmp")
    tmp_key.write_bytes(key_pem)
    try:
        os.chmod(tmp_key, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError as exc:
        logger.warning("Could not set 0600 permissions on temp key file: %s", exc)
    tmp_key.replace(key_path)

    # Re-verify permissions on destination key
    try:
        os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass

    # 2. Write certificate atomically with 0644 permissions
    tmp_cert = cert_path.with_suffix(".tmp")
    tmp_cert.write_bytes(cert_pem)
    try:
        os.chmod(tmp_cert, 0o644)
    except OSError:
        pass
    tmp_cert.replace(cert_path)

    logger.debug("Credentials successfully saved to %s", target_dir)
    return key_path, cert_path


def load_credentials(
    storage_dir: Union[str, Path],
    enforce_permissions: bool = True,
) -> Tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """Load private key and certificate from storage directory.

    Enforces that private key does not have group or world read/write permissions.
    """
    target_dir = Path(storage_dir)
    key_path = target_dir / "private_key.pem"
    cert_path = target_dir / "certificate.pem"

    if not key_path.exists():
        raise CertificateError(f"Private key not found at {key_path}")
    if not cert_path.exists():
        raise CertificateError(f"Certificate not found at {cert_path}")

    # Verify POSIX permissions: private key MUST NOT be group or world readable/writable
    if enforce_permissions and hasattr(os, "stat"):
        try:
            st = key_path.stat()
            # Check if group (S_IRWXG) or other (S_IRWXO) bits are set
            insecure_bits = st.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
            if insecure_bits != 0:
                octal_perms = oct(st.st_mode & 0o777)
                raise CertificatePermissionError(
                    f"Insecure private key permissions {octal_perms} on {key_path}. "
                    "Private key must have strict 0600 (-rw-------) permissions."
                )
        except CertificatePermissionError:
            raise
        except OSError as exc:
            logger.warning("Could not stat key permissions: %s", exc)

    key_bytes = key_path.read_bytes()
    cert_bytes = cert_path.read_bytes()

    key = load_private_key_pem(key_bytes)
    cert = load_certificate_pem(cert_bytes)

    return key, cert


def ensure_credentials(
    storage_dir: Union[str, Path],
    device_id: str,
) -> Tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """Load existing credentials or generate and persist new ones."""
    target_dir = Path(storage_dir)
    key_path = target_dir / "private_key.pem"
    cert_path = target_dir / "certificate.pem"

    if key_path.exists() and cert_path.exists():
        try:
            return load_credentials(target_dir)
        except CertificatePermissionError:
            raise
        except Exception as exc:
            logger.warning("Existing credentials corrupted (%s), regenerating...", exc)

    logger.info("Generating new 2048-bit RSA key and self-signed certificate for %s", device_id)
    key = generate_device_key()
    cert = generate_self_signed_cert(key, device_id)
    save_credentials(target_dir, key, cert)
    return key, cert

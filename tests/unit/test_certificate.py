"""Unit tests for pengulink.crypto.certificate."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import rsa

from pengulink.core.exceptions import (
    CertificateError,
    CertificatePermissionError,
    KeyGenerationError,
)
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


def test_generate_device_key() -> None:
    key = generate_device_key()
    assert isinstance(key, rsa.RSAPrivateKey)
    assert key.key_size == 2048
    assert key.public_key().public_numbers().e == 65537


def test_generate_self_signed_cert() -> None:
    key = generate_device_key()
    device_id = "test_device_12345"
    cert = generate_self_signed_cert(key, device_id, validity_days=365)

    assert isinstance(cert, x509.Certificate)
    assert get_certificate_common_name(cert) == device_id
    assert cert.issuer == cert.subject

    # Check BasicConstraints CA=True
    bc = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
    assert bc.ca is True


def test_invalid_device_id_raises() -> None:
    key = generate_device_key()
    with pytest.raises(CertificateError):
        generate_self_signed_cert(key, "")  # empty string


def test_get_certificate_fingerprint() -> None:
    key = generate_device_key()
    cert = generate_self_signed_cert(key, "dev_fp_test")
    fp = get_certificate_fingerprint(cert)

    assert isinstance(fp, str)
    assert len(fp.split(":")) == 32  # SHA-256 is 32 bytes -> 32 hex pairs
    assert fp.isupper()


def test_pem_serialization_roundtrip() -> None:
    key = generate_device_key()
    cert = generate_self_signed_cert(key, "dev_roundtrip")

    key_pem = dump_private_key_pem(key)
    assert b"BEGIN PRIVATE KEY" in key_pem
    loaded_key = load_private_key_pem(key_pem)
    assert loaded_key.key_size == key.key_size

    cert_pem = dump_certificate_pem(cert)
    assert b"BEGIN CERTIFICATE" in cert_pem
    loaded_cert = load_certificate_pem(cert_pem)
    assert get_certificate_common_name(loaded_cert) == "dev_roundtrip"


def test_save_and_load_credentials(tmp_path: Path) -> None:
    key = generate_device_key()
    cert = generate_self_signed_cert(key, "dev_saved")

    key_path, cert_path = save_credentials(tmp_path, key, cert)
    assert key_path.exists()
    assert cert_path.exists()

    # Verify POSIX 0600 permissions
    st = key_path.stat()
    perms = st.st_mode & 0o777
    assert perms == 0o600

    loaded_key, loaded_cert = load_credentials(tmp_path)
    assert loaded_key.key_size == 2048
    assert get_certificate_common_name(loaded_cert) == "dev_saved"


def test_insecure_key_permissions_rejected(tmp_path: Path) -> None:
    key = generate_device_key()
    cert = generate_self_signed_cert(key, "dev_insecure")
    key_path, _ = save_credentials(tmp_path, key, cert)

    # Intentionally make private key world-readable (0644 or 0666)
    os.chmod(key_path, 0o644)

    with pytest.raises(CertificatePermissionError) as exc_info:
        load_credentials(tmp_path, enforce_permissions=True)

    assert "Insecure private key permissions" in str(exc_info.value)


def test_ensure_credentials_idempotent(tmp_path: Path) -> None:
    dev_id = "dev_ensure"
    key1, cert1 = ensure_credentials(tmp_path, dev_id)
    fp1 = get_certificate_fingerprint(cert1)

    # Second call should load the identical credentials
    key2, cert2 = ensure_credentials(tmp_path, dev_id)
    fp2 = get_certificate_fingerprint(cert2)

    assert fp1 == fp2
    assert key1.public_key().public_numbers().n == key2.public_key().public_numbers().n

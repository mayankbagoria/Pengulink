"""Unit tests for pengulink.crypto.tls."""

from __future__ import annotations

import ssl
import pytest

from pengulink.core.exceptions import CertificateVerificationError
from pengulink.crypto.certificate import (
    dump_certificate_pem,
    dump_private_key_pem,
    generate_device_key,
    generate_self_signed_cert,
)
from pengulink.crypto.tls import (
    create_paired_ssl_context,
    create_tofu_ssl_context,
    verify_peer_certificate,
)


@pytest.fixture
def local_identity():
    key = generate_device_key()
    cert = generate_self_signed_cert(key, "local_device_node")
    return dump_private_key_pem(key), dump_certificate_pem(cert), "local_device_node"


@pytest.fixture
def remote_identity():
    key = generate_device_key()
    cert = generate_self_signed_cert(key, "remote_peer_node")
    return dump_private_key_pem(key), dump_certificate_pem(cert), "remote_peer_node"


def test_create_paired_ssl_context_server_and_client(local_identity, remote_identity) -> None:
    local_key_pem, local_cert_pem, _ = local_identity
    _, remote_cert_pem, _ = remote_identity

    # Outbound KDE Connect connection: server_side=True
    server_ctx = create_paired_ssl_context(
        local_cert_pem,
        local_key_pem,
        remote_cert_pem,
        server_side=True,
    )
    assert isinstance(server_ctx, ssl.SSLContext)
    assert server_ctx.verify_mode == ssl.CERT_REQUIRED
    assert server_ctx.check_hostname is False

    # Inbound KDE Connect connection: server_side=False
    client_ctx = create_paired_ssl_context(
        local_cert_pem,
        local_key_pem,
        remote_cert_pem,
        server_side=False,
    )
    assert isinstance(client_ctx, ssl.SSLContext)
    assert client_ctx.verify_mode == ssl.CERT_REQUIRED
    assert client_ctx.check_hostname is False


def test_create_tofu_ssl_context(local_identity) -> None:
    local_key_pem, local_cert_pem, _ = local_identity
    ctx_server = create_tofu_ssl_context(local_cert_pem, local_key_pem, server_side=True)
    assert isinstance(ctx_server, ssl.SSLContext)
    # Server-side TOFU requests client cert via SSL_VERIFY_PEER (CERT_OPTIONAL)
    assert ctx_server.verify_mode == ssl.CERT_OPTIONAL
    assert ctx_server.check_hostname is False

    ctx_client = create_tofu_ssl_context(local_cert_pem, local_key_pem, server_side=False)
    assert isinstance(ctx_client, ssl.SSLContext)
    assert ctx_client.verify_mode == ssl.CERT_NONE
    assert ctx_client.check_hostname is False


def test_verify_peer_certificate_valid(remote_identity) -> None:
    _, remote_cert_pem, remote_id = remote_identity
    verified_cert = verify_peer_certificate(
        remote_cert_pem,
        expected_device_id=remote_id,
        pinned_cert_pem=remote_cert_pem,
    )
    assert verified_cert is not None


def test_verify_peer_certificate_cn_mismatch_raises(remote_identity) -> None:
    _, remote_cert_pem, _ = remote_identity
    with pytest.raises(CertificateVerificationError) as exc_info:
        verify_peer_certificate(
            remote_cert_pem,
            expected_device_id="spoofed_device_id",
        )
    assert "does not match expected device ID" in str(exc_info.value)


def test_verify_peer_certificate_pinning_mismatch_raises(remote_identity) -> None:
    _, remote_cert_pem, remote_id = remote_identity

    # Generate an alternate cert for the same device ID to simulate MITM / cert replacement
    alt_key = generate_device_key()
    alt_cert = generate_self_signed_cert(alt_key, remote_id)
    alt_cert_pem = dump_certificate_pem(alt_cert)

    with pytest.raises(CertificateVerificationError) as exc_info:
        verify_peer_certificate(
            remote_cert_pem,
            expected_device_id=remote_id,
            pinned_cert_pem=alt_cert_pem,
        )
    assert "does not match pinned fingerprint" in str(exc_info.value)

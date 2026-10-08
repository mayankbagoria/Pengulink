"""Unit tests for tests.mock.mock_device (Milestone 4)."""

from __future__ import annotations

import asyncio
import ssl
import time
from typing import Optional

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from pengulink.core.exceptions import ConnectionClosedError, LimitOverrunError
from pengulink.crypto.certificate import (
    dump_certificate_pem,
    dump_private_key_pem,
    generate_device_key,
    generate_self_signed_cert,
)
from pengulink.crypto.pairing import compute_verification_code, create_pair_packet
from pengulink.crypto.tls import create_tofu_ssl_context
from pengulink.network.packet import MAX_PACKET_SIZE, IdentityBody, Packet, PacketEngine
from pengulink.network.tcp import ConnectionRole, TCPConnection, configure_stream_socket
from tests.mock import ChaosConfig, MockDevice, MockDeviceConfig, MockKdeConnectDevice


async def test_mock_device_lifecycle() -> None:
    """Verify MockDevice clean startup, dynamic port binding, and leak-free teardown."""
    mock = MockDevice()
    assert not mock.is_running
    assert mock.bound_tcp_port is None

    async with mock:
        assert mock.is_running
        assert mock.bound_tcp_port is not None
        assert mock.bound_tcp_port > 0

    assert not mock.is_running
    assert mock.bound_tcp_port is None
    assert len(mock._background_tasks) == 0
    assert len(mock._active_connections) == 0


async def test_mock_device_discovery_identity() -> None:
    """Verify MockDevice identity packet construction and UDP broadcast."""
    config = MockDeviceConfig(
        device_id="android_test_123",
        device_name="Samsung S24 Ultra",
        device_type="phone",
    )
    mock = MockDevice(config=config)
    async with mock:
        ident = mock.build_identity_packet()
        assert ident.type == "kdeconnect.identity"
        assert ident.body["deviceId"] == "android_test_123"
        assert ident.body["deviceName"] == "Samsung S24 Ultra"
        assert ident.body["deviceType"] == "phone"
        assert ident.body["tcpPort"] == mock.bound_tcp_port

        # Broadcast should succeed without error
        await mock.broadcast_identity(target_ip="127.0.0.1", target_port=17169)


async def test_mock_device_inbound_inverted_tls_handshake() -> None:
    """Verify PenguLink client connecting to MockDevice listener.

    KDE Connect Inverted TLS:
      - Connector (PenguLink) = TLS Server (server_side=True)
      - Listener (MockDevice) = TLS Client (server_side=False)
    """
    mock = MockDevice()
    async with mock:
        client_key = generate_device_key()
        client_cert = generate_self_signed_cert(client_key, "pengulink_daemon")
        client_cert_pem = dump_certificate_pem(client_cert)
        client_key_pem = dump_private_key_pem(client_key)

        # Connector connects to mock
        reader, writer = await asyncio.open_connection("127.0.0.1", mock.bound_tcp_port)
        configure_stream_socket(writer)
        client_conn = TCPConnection(
            reader=reader,
            writer=writer,
            role=ConnectionRole.OUTBOUND,
            peer_ip="127.0.0.1",
            peer_port=mock.bound_tcp_port,
        )

        try:
            # 1. Client sends cleartext identity preamble
            client_ident = Packet(
                type="kdeconnect.identity",
                body={
                    "deviceId": "pengulink_daemon",
                    "deviceName": "PenguLink PC",
                    "deviceType": "desktop",
                    "protocolVersion": 7,
                    "tcpPort": 1716,
                },
            )
            await client_conn.write_line(PacketEngine.serialize(client_ident))

            # 2. Client upgrades to TLS with server_side=True
            client_ssl_ctx = create_tofu_ssl_context(
                local_cert_pem=client_cert_pem,
                local_key_pem=client_key_pem,
                server_side=True,
            )
            await client_conn.upgrade_to_tls(client_ssl_ctx, server_side=True)

            # Wait briefly for mock read loop to attach
            await asyncio.sleep(0.05)
            assert mock.is_connected

            # 3. Test bidirectional communication over TLS
            ping = Packet(type="kdeconnect.ping", body={"message": "hello_from_client"})
            await client_conn.write_line(PacketEngine.serialize(ping))

            # Mock should receive the ping
            received = await mock.wait_for_packet("kdeconnect.ping", timeout=2.0)
            assert received.body.get("message") == "hello_from_client"

        finally:
            await client_conn.close()


async def test_mock_device_outbound_inverted_tls_handshake() -> None:
    """Verify MockDevice connecting outbound to a daemon listener.

    KDE Connect Inverted TLS:
      - Connector (MockDevice) = TLS Server (server_side=True)
      - Listener (Daemon)     = TLS Client (server_side=False)
    """
    server_key = generate_device_key()
    server_cert = generate_self_signed_cert(server_key, "daemon_listener")
    server_cert_pem = dump_certificate_pem(server_cert)
    server_key_pem = dump_private_key_pem(server_key)

    server_conn_future: asyncio.Future[TCPConnection] = asyncio.get_running_loop().create_future()

    async def _on_client(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        configure_stream_socket(w)
        conn = TCPConnection(
            reader=r,
            writer=w,
            role=ConnectionRole.INBOUND,
            peer_ip="127.0.0.1",
            peer_port=0,
        )
        try:
            line = await conn.read_line(timeout=2.0)
            preamble = PacketEngine.parse(line)
            conn.remote_device_id = preamble.body.get("deviceId")

            ssl_ctx = create_tofu_ssl_context(
                local_cert_pem=server_cert_pem,
                local_key_pem=server_key_pem,
                server_side=False,
            )
            await conn.upgrade_to_tls(ssl_ctx, server_side=False, timeout=2.0)
            if not server_conn_future.done():
                server_conn_future.set_result(conn)
        except Exception as exc:
            if not server_conn_future.done():
                server_conn_future.set_exception(exc)
            await conn.close()

    server = await asyncio.start_server(
        _on_client,
        host="127.0.0.1",
        port=0,
        reuse_address=True,
    )
    server_port = server.sockets[0].getsockname()[1]

    mock = MockDevice()
    async with mock:
        try:
            # Outbound connect
            mock_conn = await mock.connect_to_daemon(
                host="127.0.0.1",
                port=server_port,
                target_device_id="daemon_listener",
            )
            server_conn = await asyncio.wait_for(server_conn_future, timeout=2.0)

            # Transmit from mock to server
            await mock.send_ping("outbound_mock_ping")
            server_line = await server_conn.read_line(timeout=2.0)
            server_pkt = PacketEngine.parse(server_line)
            assert server_pkt.type == "kdeconnect.ping"
            assert server_pkt.body.get("message") == "outbound_mock_ping"

            await server_conn.close()
        finally:
            server.close()
            await server.wait_closed()


async def test_mock_device_pairing_negotiation() -> None:
    """Verify mutual pairing negotiation and visual verification code match."""
    mock = MockDevice(config=MockDeviceConfig(auto_pair=True))
    async with mock:
        client_key = generate_device_key()
        client_cert = generate_self_signed_cert(client_key, "client_pc")
        client_cert_pem = dump_certificate_pem(client_cert)
        client_key_pem = dump_private_key_pem(client_key)
        client_pubkey_pem = (
            client_cert.public_key()
            .public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            .decode("utf-8")
        )

        r, w = await asyncio.open_connection("127.0.0.1", mock.bound_tcp_port)
        client_conn = TCPConnection(
            reader=r,
            writer=w,
            role=ConnectionRole.OUTBOUND,
            peer_ip="127.0.0.1",
            peer_port=mock.bound_tcp_port,
        )

        try:
            # Preamble & TLS
            ident = mock.build_identity_packet()
            ident.body["deviceId"] = "client_pc"
            await client_conn.write_line(PacketEngine.serialize(ident))

            client_ssl_ctx = create_tofu_ssl_context(
                local_cert_pem=client_cert_pem,
                local_key_pem=client_key_pem,
                server_side=True,
            )
            await client_conn.upgrade_to_tls(client_ssl_ctx, server_side=True)
            await asyncio.sleep(0.05)

            # Send pairing request
            ts = 1700000000
            pair_request = create_pair_packet(pair=True, timestamp=ts)
            await client_conn.write_line(PacketEngine.serialize(pair_request))

            # Client computes expected code
            mock_pubkey_pem = (
                mock.certificate.public_key()
                .public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                .decode("utf-8")
            )
            expected_code = compute_verification_code(client_pubkey_pem, mock_pubkey_pem, ts)

            # Wait for mock's auto-accepted pair response
            reply_line = await client_conn.read_line(timeout=2.0)
            reply_pkt = PacketEngine.parse(reply_line)
            assert reply_pkt.type == "kdeconnect.pair"
            assert reply_pkt.body.get("pair") is True
            assert reply_pkt.body.get("timestamp") == ts

            # Check that mock derived the exact same 8-char verification code
            assert mock.last_verification_code == expected_code
            assert len(expected_code) == 8
            assert expected_code == expected_code.upper()

        finally:
            await client_conn.close()


async def test_mock_device_pairing_auto_reject() -> None:
    """Verify mock device rejects pairing when auto_reject_pairing is configured."""
    mock = MockDevice(config=MockDeviceConfig(auto_pair=False, auto_reject_pair=True))
    async with mock:
        client_key = generate_device_key()
        client_cert = generate_self_signed_cert(client_key, "client_pc")
        client_cert_pem = dump_certificate_pem(client_cert)
        client_key_pem = dump_private_key_pem(client_key)

        r, w = await asyncio.open_connection("127.0.0.1", mock.bound_tcp_port)
        client_conn = TCPConnection(
            reader=r,
            writer=w,
            role=ConnectionRole.OUTBOUND,
            peer_ip="127.0.0.1",
            peer_port=mock.bound_tcp_port,
        )

        try:
            ident = mock.build_identity_packet()
            ident.body["deviceId"] = "client_pc"
            await client_conn.write_line(PacketEngine.serialize(ident))

            client_ssl_ctx = create_tofu_ssl_context(
                local_cert_pem=client_cert_pem,
                local_key_pem=client_key_pem,
                server_side=True,
            )
            await client_conn.upgrade_to_tls(client_ssl_ctx, server_side=True)
            await asyncio.sleep(0.05)

            # Send pairing request
            pair_request = create_pair_packet(pair=True, timestamp=1700000000)
            await client_conn.write_line(PacketEngine.serialize(pair_request))

            # Read reject response
            reply_line = await client_conn.read_line(timeout=2.0)
            reply_pkt = PacketEngine.parse(reply_line)
            assert reply_pkt.type == "kdeconnect.pair"
            assert reply_pkt.body.get("pair") is False

        finally:
            await client_conn.close()


async def test_mock_device_plugins_simulation() -> None:
    """Verify mock device ping, battery, and notification plugin simulations."""
    mock = MockDevice(config=MockDeviceConfig(auto_pong=True, auto_battery_reply=True))
    async with mock:
        client_key = generate_device_key()
        client_cert = generate_self_signed_cert(client_key, "client_pc")
        client_cert_pem = dump_certificate_pem(client_cert)
        client_key_pem = dump_private_key_pem(client_key)

        r, w = await asyncio.open_connection("127.0.0.1", mock.bound_tcp_port)
        client_conn = TCPConnection(
            reader=r,
            writer=w,
            role=ConnectionRole.OUTBOUND,
            peer_ip="127.0.0.1",
            peer_port=mock.bound_tcp_port,
        )

        try:
            ident = mock.build_identity_packet()
            ident.body["deviceId"] = "client_pc"
            await client_conn.write_line(PacketEngine.serialize(ident))

            client_ssl_ctx = create_tofu_ssl_context(
                local_cert_pem=client_cert_pem,
                local_key_pem=client_key_pem,
                server_side=True,
            )
            await client_conn.upgrade_to_tls(client_ssl_ctx, server_side=True)
            await asyncio.sleep(0.05)

            # 1. Test ping auto-pong
            client_ping = Packet(type="kdeconnect.ping", body={"message": "ping_test"})
            await client_conn.write_line(PacketEngine.serialize(client_ping))

            pong_line = await client_conn.read_line(timeout=2.0)
            pong_pkt = PacketEngine.parse(pong_line)
            assert pong_pkt.type == "kdeconnect.ping"
            assert pong_pkt.body.get("message") == "Pong from MockDevice"

            # 2. Test battery request auto-reply
            bat_req = Packet(type="kdeconnect.battery.request", body={"request": True})
            await client_conn.write_line(PacketEngine.serialize(bat_req))

            bat_line = await client_conn.read_line(timeout=2.0)
            bat_pkt = PacketEngine.parse(bat_line)
            assert bat_pkt.type == "kdeconnect.battery"
            assert bat_pkt.body.get("currentCharge") == 85
            assert bat_pkt.body.get("isCharging") is False

            # 3. Test send battery update from mock
            await mock.send_battery_update(current_charge=42, is_charging=True)
            bat_update_line = await client_conn.read_line(timeout=2.0)
            bat_update_pkt = PacketEngine.parse(bat_update_line)
            assert bat_update_pkt.type == "kdeconnect.battery"
            assert bat_update_pkt.body.get("currentCharge") == 42
            assert bat_update_pkt.body.get("isCharging") is True

            # 4. Test send notification with Android prefix
            await mock.send_notification(
                notification_id="msg_99",
                app_name="WhatsApp",
                title="Alice",
                text="Verification code is 987654",
            )
            notif_line = await client_conn.read_line(timeout=2.0)
            notif_pkt = PacketEngine.parse(notif_line)
            assert notif_pkt.type == "kdeconnect.notification"
            assert notif_pkt.body.get("id") == "org.kde.kdeconnect_tp::msg_99"
            assert notif_pkt.body.get("appName") == "WhatsApp"
            assert "987654" in notif_pkt.body.get("text", "")

            # 5. Test cancel notification
            await mock.cancel_notification("msg_99")
            cancel_line = await client_conn.read_line(timeout=2.0)
            cancel_pkt = PacketEngine.parse(cancel_line)
            assert cancel_pkt.type == "kdeconnect.notification"
            assert cancel_pkt.body.get("id") == "org.kde.kdeconnect_tp::msg_99"
            assert cancel_pkt.body.get("isCancel") is True

        finally:
            await client_conn.close()


async def test_mock_device_chaos_injection() -> None:
    """Verify mock device chaos injection: malformed JSON, raw bytes, interceptor, and reset drop."""
    mock = MockDevice()
    async with mock:
        client_key = generate_device_key()
        client_cert = generate_self_signed_cert(client_key, "client_pc")
        client_cert_pem = dump_certificate_pem(client_cert)
        client_key_pem = dump_private_key_pem(client_key)

        r, w = await asyncio.open_connection("127.0.0.1", mock.bound_tcp_port)
        client_conn = TCPConnection(
            reader=r,
            writer=w,
            role=ConnectionRole.OUTBOUND,
            peer_ip="127.0.0.1",
            peer_port=mock.bound_tcp_port,
        )

        try:
            ident = mock.build_identity_packet()
            ident.body["deviceId"] = "client_pc"
            await client_conn.write_line(PacketEngine.serialize(ident))

            client_ssl_ctx = create_tofu_ssl_context(
                local_cert_pem=client_cert_pem,
                local_key_pem=client_key_pem,
                server_side=True,
            )
            await client_conn.upgrade_to_tls(client_ssl_ctx, server_side=True)
            await asyncio.sleep(0.05)

            # 1. Inject malformed JSON
            await mock.send_malformed_json(b'{"corrupt": true\n')
            corrupt_line = await client_conn.read_line(timeout=2.0)
            assert corrupt_line == b'{"corrupt": true\n'

            # 2. Packet Interceptor Mutation
            mock.chaos_config.packet_interceptor = lambda p: Packet(
                type="kdeconnect.ping", body={"intercepted": True}
            )
            await mock.send_ping("regular_ping")
            inter_line = await client_conn.read_line(timeout=2.0)
            inter_pkt = PacketEngine.parse(inter_line)
            assert inter_pkt.body.get("intercepted") is True

            # 3. Drop connection with TCP RST (SO_LINGER)
            await mock.drop_connection(reset=True)
            assert not mock.is_connected

            # Reading from client should now yield EOF or ConnectionResetError
            line = await client_conn.read_line(timeout=1.0)
            assert line == b""

        finally:
            await client_conn.close()

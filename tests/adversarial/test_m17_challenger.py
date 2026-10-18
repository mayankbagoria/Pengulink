"""tests.adversarial.test_m17_challenger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Adversarial Challenge Suite for Milestone 17:
- CLUS-01: STARTTLS Cleartext Stream Buffer Injection Stress (tcp.py)
- CLUS-06: ContactStore Path Traversal & Host File Protection Fuzzing (contacts/store.py)
- CLUS-07 & CLUS-20: Unauthenticated Input Rejection & Output Buffer Clamping Stress
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import ssl
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.core.events import (
    EventBus,
    FindDeviceAlertStoppedEvent,
    FindDeviceAlertTriggeredEvent,
    RemoteCommandOutputEvent,
    RemoteCommandsReceivedEvent,
)
from pengulink.core.exceptions import (
    LimitOverrunError,
    PathTraversalError,
    TLSUpgradeError,
)
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.network.packet import Packet
from pengulink.network.tcp import ConnectionRole, TCPConnection
from pengulink.plugins.clipboard.backend import InMemoryClipboardBackend
from pengulink.plugins.clipboard.plugin import ClipboardPlugin
from pengulink.plugins.contacts.store import ContactStore
from pengulink.plugins.contacts.vcard import Contact
from pengulink.plugins.findmyphone import FindMyPhonePlugin
from pengulink.plugins.remotecommands import RemoteCommandsPlugin


# =============================================================================
# 1. CLUS-01: STARTTLS Cleartext Stream Buffer Injection Adversarial Stress
# =============================================================================

class TestClus01PipelinedCleartextStreamBufferInjection:
    """Empirically stress-test STARTTLS stream buffer injection detection and cleanup."""

    @pytest.mark.parametrize(
        "pipelined_payload",
        [
            b"{",  # Single opening brace
            b"X",  # Single arbitrary byte
            b'{"id": 2',  # Incomplete JSON opening
            b'{"id": 2, "type": "kdeconnect.',  # Truncated packet
            b"NO_NEWLINE_RAW_BUFFER_BYTES",  # Raw buffer without newline
            b'{"id":2,"type":"kdeconnect.ping","body":{}}',  # Complete JSON without newline
            b"\n",  # Single orphan newline
            b"\n\n\n\n\n",  # Multiple orphan newlines
            b"\r\n\r\n\r\n",  # CRLF sequences
            b"   \t  \r\n   ",  # Whitespace padding
            b'{"id":2,"type":"kdeconnect.clipboard","body":{"content":"injected"}}\n',  # Full packet
            b'{"id":2,"type":"kdeconnect.ping","body":{}}\n{"id":3,"type":"kdeconnect.findmyphone.request"}\n',  # Multiple packets
            b"A" * 65536,  # 64 KB cleartext buffer
            b"B" * 131072,  # 128 KB buffer
            b"C" * 262144,  # 256 KB buffer
        ],
    )
    async def test_pre_tls_buffer_injection_all_payload_variants(self, pipelined_payload: bytes) -> None:
        """Any unconsumed bytes in reader._buffer prior to TLS upgrade strictly aborts with TLSUpgradeError."""
        reader = asyncio.StreamReader()
        reader.feed_data(pipelined_payload)

        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        conn = TCPConnection(
            reader=reader,
            writer=writer,
            role=ConnectionRole.INBOUND,
            peer_ip="192.168.1.50",
            peer_port=1716,
        )

        mock_ssl_ctx = MagicMock(spec=ssl.SSLContext)

        with pytest.raises(TLSUpgradeError, match="STARTTLS injection: unconsumed cleartext bytes in stream buffer"):
            await conn.upgrade_to_tls(mock_ssl_ctx, server_side=False)

        assert conn.is_closed is True
        assert conn.is_tls_upgraded is False
        writer.close.assert_called_once()

    @pytest.mark.parametrize(
        "post_handshake_payload",
        [
            b"TRAILING_HANDSHAKE_DATA",
            b"\n\n\n",
            b'{"id": 99, "type": "kdeconnect.runcommand"}\n',
            b"D" * 65536,
        ],
    )
    async def test_post_tls_buffer_injection_rejection(self, post_handshake_payload: bytes) -> None:
        """If data appears in reader._buffer while loop.start_tls() completes, upgrade is rejected."""
        reader = asyncio.StreamReader()
        # Pre-handshake buffer is empty initially

        mock_transport = MagicMock()
        mock_protocol = MagicMock()
        mock_transport.get_protocol.return_value = mock_protocol

        writer = MagicMock()
        writer.transport = mock_transport
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        conn = TCPConnection(
            reader=reader,
            writer=writer,
            role=ConnectionRole.OUTBOUND,
            peer_ip="192.168.1.100",
            peer_port=1716,
        )

        mock_ssl_ctx = MagicMock(spec=ssl.SSLContext)
        new_transport = MagicMock()

        async def fake_start_tls(*args: Any, **kwargs: Any) -> Any:
            # Simulate cleartext packet injected during handshake
            reader.feed_data(post_handshake_payload)
            return new_transport

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.start_tls = AsyncMock(side_effect=fake_start_tls)
            mock_get_loop.return_value = mock_loop

            with pytest.raises(TLSUpgradeError, match="STARTTLS injection: unconsumed cleartext bytes in stream buffer"):
                await conn.upgrade_to_tls(mock_ssl_ctx, server_side=True)

            assert conn.is_closed is True
            assert conn.is_tls_upgraded is False
            writer.close.assert_called_once()

    async def test_clean_buffer_handshake_completes_and_is_idempotent(self) -> None:
        """When buffer is empty, TLS upgrade succeeds; subsequent calls are idempotent no-ops."""
        reader = asyncio.StreamReader()
        mock_transport = MagicMock()
        mock_protocol = MagicMock()
        mock_transport.get_protocol.return_value = mock_protocol

        writer = MagicMock()
        writer.transport = mock_transport
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        conn = TCPConnection(
            reader=reader,
            writer=writer,
            role=ConnectionRole.OUTBOUND,
            peer_ip="127.0.0.1",
            peer_port=1716,
        )

        mock_ssl_ctx = MagicMock(spec=ssl.SSLContext)
        new_transport = MagicMock()

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.start_tls = AsyncMock(return_value=new_transport)
            mock_get_loop.return_value = mock_loop

            # 1. Clean upgrade succeeds
            await conn.upgrade_to_tls(mock_ssl_ctx, server_side=True)
            assert conn.is_tls_upgraded is True
            assert conn.is_closed is False
            assert conn.writer._transport is new_transport

            # 2. Subsequent call is an idempotent no-op
            await conn.upgrade_to_tls(mock_ssl_ctx, server_side=True)
            assert mock_loop.start_tls.call_count == 1

        # 3. Closed connection returns immediately
        await conn.close()
        assert conn.is_closed is True
        await conn.upgrade_to_tls(mock_ssl_ctx, server_side=True)

    async def test_live_loopback_cleartext_pipelined_injection(self) -> None:
        """Empirically test live TCP loopback stream where client pipelines cleartext bytes with identity."""
        received_errors: List[Exception] = []
        server_conn_ref: List[TCPConnection] = []

        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            conn = TCPConnection(
                reader=reader,
                writer=writer,
                role=ConnectionRole.INBOUND,
                peer_ip="127.0.0.1",
                peer_port=0,
            )
            server_conn_ref.append(conn)

            try:
                # Read identity line
                identity_line = await conn.read_line(timeout=2.0)
                assert b"kdeconnect.identity" in identity_line

                # Attempt TLS upgrade with dummy SSL context
                mock_ssl = MagicMock(spec=ssl.SSLContext)
                await conn.upgrade_to_tls(mock_ssl, server_side=False)
            except Exception as exc:
                received_errors.append(exc)
            finally:
                await conn.close()

        server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
        server_port = server.sockets[0].getsockname()[1]

        try:
            client_reader, client_writer = await asyncio.open_connection("127.0.0.1", server_port)

            # Transmit identity concatenated with injected cleartext packet in a single TCP write
            pipelined_stream = (
                b'{"id":1,"type":"kdeconnect.identity","body":{"deviceId":"bad_actor"}}\n'
                b'{"id":2,"type":"kdeconnect.clipboard","body":{"content":"pwned"}}\n'
            )
            client_writer.write(pipelined_stream)
            await client_writer.drain()

            # Await server to handle and disconnect
            await asyncio.sleep(0.1)

            assert len(received_errors) == 1
            err = received_errors[0]
            assert isinstance(err, TLSUpgradeError)
            assert "STARTTLS injection: unconsumed cleartext bytes in stream buffer" in str(err)
            assert server_conn_ref[0].is_closed is True

            # Client should observe EOF/closed socket
            client_writer.close()
            await client_writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()


# =============================================================================
# 2. CLUS-06: ContactStore Path Traversal Adversarial Fuzzing
# =============================================================================

class TestClus06ContactStorePathTraversalFuzzing:
    """Fuzz ContactStore._get_device_file and methods with extensive traversal payloads."""

    TRAVERSAL_PAYLOADS = [
        # Relative traversal
        "../",
        "../../",
        "../../../",
        "../../tmp/malicious",
        "../escape.json",
        "..\\",
        "..\\..\\",
        "..\\..\\Windows\\System32\\cmd.exe",
        "subdir/nested",
        "subdir\\nested",
        "....//....//etc/passwd",
        ".../...",
        "foo/../../bar",
        # URL / Percent encodings
        "..%2f",
        "..%2F",
        "..%5c",
        "..%5C",
        "%2e%2e%2f",
        "%252e%252e%252f",
        "..%c0%af",
        "..%c1%9c",
        # Null bytes & truncation
        "\x00",
        "device\x00",
        "device\x00.json",
        "valid_id\x00/../../etc/passwd",
        "\x00/etc/passwd",
        # Absolute paths
        "/etc/passwd",
        "/etc/shadow",
        "/root/.ssh/id_rsa",
        "/dev/null",
        "/tmp/evil",
        "C:\\boot.ini",
        "D:\\Windows\\secrets.txt",
        # Non-ASCII & Unicode homoglyphs
        "device_id_üñîçødé",
        "device_🦀",
        "device_日本語",
        "\uff0e\uff0e\uff0f",  # Fullwidth dot-dot-slash
        "\u202e_right_to_left",
        "dev\u0000ice",
        # Special characters & command metacharacters
        "",
        " ",
        "   \t\n   ",
        "device id with spaces",
        "device;rm -rf /",
        "device$(id)",
        "device`id`",
        "device|cat /etc/passwd",
        "device>output",
        "device<input",
        "device&bg",
        "device*",
        "device?",
        "device[0]",
        # Boundary cases
        ".",
        "..",
        "...",
        "valid_id/",
        "/valid_id",
        "valid_id\\",
        "\\valid_id",
    ]

    @pytest.mark.parametrize("payload", TRAVERSAL_PAYLOADS)
    def test_get_device_file_strictly_raises_path_traversal_error(
        self, tmp_path: Path, payload: str
    ) -> None:
        """Every traversal and injection payload strictly raises PathTraversalError."""
        store = ContactStore(data_dir=tmp_path)
        with pytest.raises(PathTraversalError):
            store._get_device_file(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            "../../etc/passwd",
            "../escape",
            "..\\Windows\\system.ini",
            "/tmp/arbitrary_overwrite",
            "malicious\x00_device",
            "dev;touch /tmp/pwn",
        ],
    )
    async def test_store_crud_methods_prevent_host_manipulation(
        self, tmp_path: Path, payload: str
    ) -> None:
        """save_device, clear_device, load_device, and get_contacts_diff strictly reject traversal."""
        store = ContactStore(data_dir=tmp_path)
        await store.initialize()

        # 1. save_device cannot write outside contacts_dir
        with pytest.raises(PathTraversalError):
            await store.save_device(payload, {}, {})

        # 2. clear_device cannot unlink files outside contacts_dir
        with pytest.raises(PathTraversalError):
            await store.clear_device(payload)

        # 3. load_device cannot read files outside contacts_dir
        with pytest.raises(PathTraversalError):
            await store.load_device(payload)

        # 4. get_contact and list_contacts cannot read files outside contacts_dir
        with pytest.raises(PathTraversalError):
            await store.get_contact(payload, "uid1")

        with pytest.raises(PathTraversalError):
            await store.list_contacts(payload)

    @pytest.mark.parametrize(
        "valid_id",
        [
            "device123",
            "android-phone-01",
            "My_Pixel_7a",
            "NODE_42-test",
            "0123456789",
            "A",
            "a-b_c",
        ],
    )
    def test_valid_alphanumeric_device_ids_allowed(self, tmp_path: Path, valid_id: str) -> None:
        """Legitimate device IDs matching ^[a-zA-Z0-9_-]+$ resolve safely within contacts_dir."""
        store = ContactStore(data_dir=tmp_path)
        path = store._get_device_file(valid_id)
        assert path.name == f"{valid_id}.json"
        assert path.resolve().is_relative_to(store.contacts_dir.resolve())


# =============================================================================
# 3. CLUS-07 & CLUS-20: Unauthenticated Input, State Mutation Immunity & Clamping
# =============================================================================

class TestClus07And20UnauthenticatedInputAndClampingStress:
    """Empirically stress-test plugin auth enforcement, state mutation immunity, and output clamping."""

    # -------------------------------------------------------------------------
    # 3.1 ClipboardPlugin
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("use_async_checker", [False, True])
    async def test_clipboard_unauthenticated_zero_execution_and_no_mutation(
        self, use_async_checker: bool
    ) -> None:
        """Incoming clipboard packets from unpaired peers cause zero host clipboard mutations and no state pollution."""
        backend = InMemoryClipboardBackend()
        await backend.set_text("INITIAL_SAFE_CLIPBOARD")

        def sync_checker(dev: str) -> bool:
            return False

        async def async_checker(dev: str) -> bool:
            return False

        checker = async_checker if use_async_checker else sync_checker

        mock_sender = AsyncMock()
        plugin = ClipboardPlugin(
            backend=backend,
            packet_sender=mock_sender,
            is_paired_checker=checker,
        )
        plugin._running = True

        packet = Packet(
            id=101,
            type="kdeconnect.clipboard",
            body={"content": "UNAUTHORIZED_PASTEJACK_ATTACK\nrm -rf ~;\n", "timestamp": 9999999},
        )

        send_cb = AsyncMock()
        res = await plugin.handle_packet(packet, "attacker_node_666", send_cb)

        # 1. Packet is consumed/dropped cleanly
        assert res is True

        # 2. Host clipboard is completely unchanged
        current_clip = await backend.get_text()
        assert current_clip == "INITIAL_SAFE_CLIPBOARD"

        # 3. Plugin internal state is not polluted with attacker node
        assert "attacker_node_666" not in plugin._connected_devices
        assert "attacker_node_666" not in plugin._device_senders

        # 4. Zero network fanout to peers
        mock_sender.assert_not_called()
        send_cb.assert_not_called()

    # -------------------------------------------------------------------------
    # 3.2 FindMyPhonePlugin
    # -------------------------------------------------------------------------

    async def test_findmyphone_unauthenticated_zero_execution_and_cannot_toggle_active_alert(self) -> None:
        """FindMyPhonePlugin rejects packets from unpaired peers; cannot trigger or stop desktop audio alerts."""
        mock_alert = MagicMock()
        mock_alert.is_active = MagicMock(return_value=False)
        mock_alert.start_alert = AsyncMock()
        mock_alert.stop_alert = AsyncMock()

        is_paired_checker = MagicMock(return_value=False)

        plugin = FindMyPhonePlugin(
            alert_service=mock_alert,
            is_paired_checker=is_paired_checker,
        )

        packet = Packet(id=201, type="kdeconnect.findmyphone.request", body={})
        send_cb = AsyncMock()

        # 1. Unpaired device cannot start an inactive alert
        res = await plugin.handle_packet(packet, "unpaired_attacker", send_cb)
        assert res is True
        mock_alert.start_alert.assert_not_called()

        # 2. Legacy packet type is also rejected
        legacy_packet = Packet(id=202, type="kdeconnect.findmyphone", body={})
        res2 = await plugin.handle_packet(legacy_packet, "unpaired_attacker", send_cb)
        assert res2 is True
        mock_alert.start_alert.assert_not_called()

        # 3. Unpaired device cannot stop or toggle an already active alert
        mock_alert.is_active.return_value = True
        res3 = await plugin.handle_packet(packet, "unpaired_attacker", send_cb)
        assert res3 is True
        mock_alert.stop_alert.assert_not_called()

    # -------------------------------------------------------------------------
    # 3.3 RemoteCommandsPlugin & Buffer Clamping
    # -------------------------------------------------------------------------

    async def test_remotecommands_unauthenticated_zero_execution_and_no_mutation(self) -> None:
        """RemoteCommandsPlugin rejects packets from unpaired peers; zero state mutations and no events."""
        event_bus = EventBus()
        captured_events: List[Any] = []
        event_bus.subscribe(RemoteCommandsReceivedEvent, lambda ev: captured_events.append(ev))
        event_bus.subscribe(RemoteCommandOutputEvent, lambda ev: captured_events.append(ev))

        is_paired_checker = MagicMock(return_value=False)
        plugin = RemoteCommandsPlugin(
            event_bus=event_bus,
            is_paired_checker=is_paired_checker,
        )

        send_cb = AsyncMock()

        # 1. Command list injection rejected
        cmd_list_packet = Packet(
            id=301,
            type="kdeconnect.runcommand",
            body={"commandList": '{"malicious": {"name": "Evil", "command": "rm -rf /"}}'},
        )
        res1 = await plugin.handle_packet(cmd_list_packet, "attacker_dev", send_cb)
        assert res1 is True
        assert len(plugin.get_commands("attacker_dev")) == 0
        assert "attacker_dev" not in plugin._connected_senders
        assert len(captured_events) == 0

        # 2. Command output spoofing rejected
        output_packet = Packet(
            id=302,
            type="kdeconnect.runcommand.output",
            body={"key": "cmd1", "output": "Fake command output", "exitCode": 0},
        )
        res2 = await plugin.handle_packet(output_packet, "attacker_dev", send_cb)
        assert res2 is True
        assert len(captured_events) == 0

    @pytest.mark.parametrize(
        "output_size, expected_size",
        [
            (10, 10),
            (65535, 65535),  # Just below 64 KB
            (65536, 65536),  # Exactly 64 KB boundary
            (65537, 65536),  # Just above 64 KB
            (100000, 65536),  # 100 KB
            (1000000, 65536),  # 1 MB
        ],
    )
    async def test_remotecommands_output_clamping_exact_boundaries(
        self, output_size: int, expected_size: int
    ) -> None:
        """RemoteCommandsPlugin strictly clamps stdout feedback to 65536 characters (CLUS-20)."""
        event_bus = EventBus()
        captured_events: List[RemoteCommandOutputEvent] = []
        event_bus.subscribe(RemoteCommandOutputEvent, lambda ev: captured_events.append(ev))

        is_paired_checker = MagicMock(return_value=True)
        plugin = RemoteCommandsPlugin(
            event_bus=event_bus,
            is_paired_checker=is_paired_checker,
        )

        raw_payload = "Z" * output_size
        packet = Packet(
            id=303,
            type="kdeconnect.runcommand.output",
            body={"key": "benchmark_cmd", "output": raw_payload, "exitCode": 0},
        )

        send_cb = AsyncMock()
        await plugin.handle_packet(packet, "paired_node_1", send_cb)

        assert len(captured_events) == 1
        ev = captured_events[0]
        assert len(ev.output) == expected_size
        assert ev.output == "Z" * expected_size
        assert ev.exit_code == 0

    # -------------------------------------------------------------------------
    # 3.4 Central Dispatcher Defense-in-Depth
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "packet_type, body",
        [
            ("kdeconnect.clipboard", {"content": "attack"}),
            ("kdeconnect.findmyphone.request", {}),
            ("kdeconnect.runcommand", {"commandList": "{}"}),
            ("kdeconnect.runcommand.output", {"key": "x", "output": "y"}),
            ("kdeconnect.ping", {}),
            ("kdeconnect.battery", {"currentCharge": 100}),
            ("kdeconnect.notification", {"id": "1", "appName": "evil"}),
        ],
    )
    async def test_central_dispatcher_drops_unauthenticated_non_pairing_packets(
        self, sample_config: PenguLinkConfig, packet_type: str, body: Dict[str, Any]
    ) -> None:
        """Daemon._connection_read_loop unconditionally drops all non-pairing packets from unpaired devices."""
        sample_config.ensure_directories()
        daemon = PenguLinkDaemon(config=sample_config)
        daemon._running = True

        mock_conn = MagicMock()
        mock_conn.is_closed = False
        mock_conn.remote_device_id = "unpaired_hostile_node"
        mock_conn.close = AsyncMock()

        packet = Packet(id=500, type=packet_type, body=body)
        raw_bytes = packet.to_wire()
        mock_conn.read_line = AsyncMock(side_effect=[raw_bytes, b""])

        daemon.is_device_paired = MagicMock(return_value=False)
        daemon.plugin_registry.dispatch_packet = AsyncMock()

        await daemon._connection_read_loop(mock_conn)

        # Non-pairing packet must be blocked at dispatcher level
        daemon.plugin_registry.dispatch_packet.assert_not_called()
        daemon.is_device_paired.assert_called_with("unpaired_hostile_node")

    async def test_central_dispatcher_permits_pairing_packet_from_unpaired_device(
        self, sample_config: PenguLinkConfig
    ) -> None:
        """Daemon._connection_read_loop permits kdeconnect.pair packets from unpaired devices for pairing negotiation."""
        sample_config.ensure_directories()
        daemon = PenguLinkDaemon(config=sample_config)
        daemon._running = True

        mock_conn = MagicMock()
        mock_conn.is_closed = False
        mock_conn.remote_device_id = "new_device_pairing"
        mock_conn.close = AsyncMock()

        pair_packet = Packet(id=501, type="kdeconnect.pair", body={"pair": True})
        mock_conn.read_line = AsyncMock(side_effect=[pair_packet.to_wire(), b""])

        daemon.is_device_paired = MagicMock(return_value=False)
        daemon._handle_pairing_packet = AsyncMock()

        await daemon._connection_read_loop(mock_conn)

        # Pairing packet must NOT be discarded
        daemon._handle_pairing_packet.assert_awaited_once()

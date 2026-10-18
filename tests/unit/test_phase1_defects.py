"""tests.unit.test_phase1_defects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dedicated regression test suite for Milestone 17 (Phase 1: Immediate Security & Crash Mitigation):
- CLUS-01 (BUG-CRIT-01): STARTTLS Cleartext Stream Buffer Injection (tcp.py)
- CLUS-06 (BUG-CRIT-06): Path Traversal in ContactStore (contacts/store.py)
- CLUS-07 & CLUS-20 (BUG-CRIT-07, BUG-HIGH-13, BUG-016): Unauthenticated Input & Dispatcher Bypass
- CLUS-03 & CLUS-24 (BUG-CRIT-03, BUG-A2-10): State Machine Transition Matrix Matrix Dead Ends
- CLUS-05 (BUG-CRIT-05): Persistent Device Store Wipeout on Uninitialized Save (json_store.py)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.core.events import EventBus, RemoteCommandOutputEvent
from pengulink.core.exceptions import (
    PathTraversalError,
    TLSUpgradeError,
)
from pengulink.core.state_machine import DeviceState, DeviceStateMachine
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.network.packet import Packet
from pengulink.network.tcp import ConnectionRole, TCPConnection
from pengulink.plugins.clipboard.backend import InMemoryClipboardBackend
from pengulink.plugins.clipboard.plugin import ClipboardPlugin
from pengulink.plugins.contacts.store import ContactStore
from pengulink.plugins.contacts.vcard import Contact
from pengulink.plugins.findmyphone import FindMyPhonePlugin
from pengulink.plugins.remotecommands import RemoteCommandsPlugin
from pengulink.storage.json_store import JsonStorageBackend


# =============================================================================
# 1. CLUS-01 (BUG-CRIT-01): STARTTLS Cleartext Stream Buffer Injection
# =============================================================================

class TestClus01StarttlsBufferInjection:
    """Verify that unconsumed cleartext bytes in stream reader buffer are rejected."""

    async def test_starttls_pre_handshake_buffer_injection_rejected(self) -> None:
        """Injecting pipelined cleartext data before TLS upgrade triggers security alert and raises TLSUpgradeError."""
        reader = asyncio.StreamReader()
        # Feed pipelined cleartext data directly into reader's internal buffer
        reader.feed_data(b'{"id":2,"type":"kdeconnect.clipboard","body":{"content":"injected"}}\n')

        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        conn = TCPConnection(
            reader=reader,
            writer=writer,
            role=ConnectionRole.INBOUND,
            peer_ip="127.0.0.1",
            peer_port=1716,
        )

        mock_ssl_ctx = MagicMock()

        with pytest.raises(TLSUpgradeError, match="STARTTLS injection: unconsumed cleartext bytes in stream buffer"):
            await conn.upgrade_to_tls(mock_ssl_ctx, server_side=False)

        assert conn.is_closed is True
        writer.close.assert_called_once()

    async def test_starttls_clean_buffer_proceeds_to_start_tls(self) -> None:
        """When reader buffer is empty, upgrade_to_tls proceeds to loop.start_tls."""
        reader = asyncio.StreamReader()
        # reader buffer is empty

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

        mock_ssl_ctx = MagicMock()
        new_transport = MagicMock()

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.start_tls = AsyncMock(return_value=new_transport)
            mock_get_loop.return_value = mock_loop

            await conn.upgrade_to_tls(mock_ssl_ctx, server_side=True)

            mock_loop.start_tls.assert_awaited_once_with(
                mock_transport,
                mock_protocol,
                mock_ssl_ctx,
                server_side=True,
            )
            assert conn.is_tls_upgraded is True
            assert conn.writer._transport is new_transport


# =============================================================================
# 2. CLUS-06 (BUG-CRIT-06): Path Traversal in ContactStore
# =============================================================================

class TestClus06ContactStorePathTraversal:
    """Verify that ContactStore strictly validates device_id and prevents filesystem traversal."""

    def test_get_device_file_rejects_path_traversal(self, tmp_path: Path) -> None:
        """Directory traversal patterns raise PathTraversalError."""
        store = ContactStore(data_dir=tmp_path)

        traversal_payloads = [
            "../../tmp/evil",
            "../escape",
            "subdir/nested",
            "subdir\\nested",
            "device\x00null",
            "..",
            "",
            "bad dev id with spaces",
            "dev;rm -rf",
        ]

        for payload in traversal_payloads:
            with pytest.raises(PathTraversalError):
                store._get_device_file(payload)

    def test_get_device_file_accepts_valid_alphanumeric_device_ids(self, tmp_path: Path) -> None:
        """Valid device IDs containing alphanumeric, dashes, and underscores resolve cleanly."""
        store = ContactStore(data_dir=tmp_path)
        valid_ids = ["device123", "device-test_01", "MY_PHONE-42"]

        for valid_id in valid_ids:
            path = store._get_device_file(valid_id)
            assert path.name == f"{valid_id}.json"
            assert path.resolve().is_relative_to(store.contacts_dir.resolve())

    async def test_save_and_clear_device_rejects_traversal(self, tmp_path: Path) -> None:
        """Calling save_device or clear_device with malicious device_id raises PathTraversalError."""
        store = ContactStore(data_dir=tmp_path)
        await store.initialize()

        with pytest.raises(PathTraversalError):
            await store.save_device("../../etc/passwd", {}, {})

        with pytest.raises(PathTraversalError):
            await store.clear_device("../../etc/shadow")


# =============================================================================
# 3. CLUS-07 & CLUS-20 (BUG-CRIT-07, BUG-HIGH-13, BUG-016): Unauthenticated Input
# =============================================================================

class TestClus07And20UnauthenticatedInputMitigation:
    """Verify that unauthenticated input is rejected by Clipboard, FindMyPhone, and RemoteCommands."""

    async def test_clipboard_plugin_drops_unpaired_packet(self) -> None:
        """ClipboardPlugin rejects incoming clipboard packet from unpaired peer."""
        backend = InMemoryClipboardBackend()
        is_paired_checker = MagicMock(return_value=False)

        plugin = ClipboardPlugin(
            backend=backend,
            is_paired_checker=is_paired_checker,
        )
        plugin._running = True

        packet = Packet(
            id=101,
            type="kdeconnect.clipboard",
            body={"content": "malicious payload", "timestamp": 12345},
        )

        send_cb = AsyncMock()
        res = await plugin.handle_packet(packet, "unpaired_device_1", send_cb)

        # Dropped: local clipboard unchanged
        assert res is True
        current_text = await backend.get_text()
        assert current_text == ""
        is_paired_checker.assert_called_once_with("unpaired_device_1")

    async def test_clipboard_plugin_accepts_paired_packet(self) -> None:
        """ClipboardPlugin accepts incoming clipboard packet from paired peer."""
        backend = InMemoryClipboardBackend()
        is_paired_checker = MagicMock(return_value=True)

        plugin = ClipboardPlugin(
            backend=backend,
            is_paired_checker=is_paired_checker,
        )
        plugin._running = True

        packet = Packet(
            id=102,
            type="kdeconnect.clipboard",
            body={"content": "trusted content", "timestamp": 123456789},
        )

        send_cb = AsyncMock()
        await plugin.handle_packet(packet, "paired_device_1", send_cb)

        current_text = await backend.get_text()
        assert current_text == "trusted content"

    async def test_findmyphone_plugin_drops_unpaired_packet(self) -> None:
        """FindMyPhonePlugin rejects incoming locate packet from unpaired peer."""
        mock_alert = MagicMock()
        mock_alert.is_active = MagicMock(return_value=False)
        mock_alert.start_alert = AsyncMock()

        is_paired_checker = MagicMock(return_value=False)

        plugin = FindMyPhonePlugin(
            alert_service=mock_alert,
            is_paired_checker=is_paired_checker,
        )

        packet = Packet(
            id=201,
            type="kdeconnect.findmyphone.request",
            body={},
        )

        send_cb = AsyncMock()
        res = await plugin.handle_packet(packet, "unpaired_attacker", send_cb)

        assert res is True
        mock_alert.start_alert.assert_not_called()

    async def test_remotecommands_plugin_pairing_and_buffer_clamping(self) -> None:
        """RemoteCommandsPlugin enforces pairing check and clamps output to 64 KB."""
        event_bus = EventBus()
        captured_events = []
        event_bus.subscribe(RemoteCommandOutputEvent, lambda ev: captured_events.append(ev))

        is_paired_checker = MagicMock(return_value=False)

        plugin = RemoteCommandsPlugin(
            event_bus=event_bus,
            is_paired_checker=is_paired_checker,
        )

        # 1. Unpaired device packet is rejected
        packet_unpaired = Packet(
            id=301,
            type="kdeconnect.runcommand",
            body={"commandList": '{"cmd1": {"name": "ls", "command": "ls"}}'},
        )
        send_cb = AsyncMock()
        res = await plugin.handle_packet(packet_unpaired, "unpaired_node", send_cb)
        assert res is True
        assert len(plugin.get_commands("unpaired_node")) == 0

        # 2. Paired device packet with 100 KB output is clamped to 64 KB (65536 bytes)
        is_paired_checker.return_value = True
        huge_output = "A" * 100_000
        packet_output = Packet(
            id=302,
            type="kdeconnect.runcommand.output",
            body={"key": "cmd1", "output": huge_output, "exitCode": 0},
        )
        await plugin.handle_packet(packet_output, "paired_node", send_cb)

        assert len(captured_events) == 1
        output_event = captured_events[0]
        assert len(output_event.output) == 65536
        assert output_event.output == "A" * 65536

    async def test_daemon_read_loop_drops_unauthenticated_packets(self, sample_config: PenguLinkConfig) -> None:
        """Daemon._connection_read_loop drops non-pairing packets from unpaired devices."""
        sample_config.ensure_directories()
        daemon = PenguLinkDaemon(config=sample_config)
        daemon._running = True

        mock_conn = MagicMock()
        mock_conn.is_closed = False
        mock_conn.remote_device_id = "unpaired_remote_node"
        mock_conn.close = AsyncMock()

        # Feed a non-pairing packet, then None (EOF)
        packet_bytes = b'{"id": 401, "type": "kdeconnect.findmyphone.request", "body": {}}\n'
        mock_conn.read_line = AsyncMock(side_effect=[packet_bytes, b""])

        daemon.is_device_paired = MagicMock(return_value=False)
        daemon.plugin_registry.dispatch_packet = AsyncMock()

        await daemon._connection_read_loop(mock_conn)

        # Discarded: dispatch_packet was never called
        daemon.plugin_registry.dispatch_packet.assert_not_called()
        daemon.is_device_paired.assert_called_with("unpaired_remote_node")


# =============================================================================
# 4. CLUS-03 & CLUS-24 (BUG-CRIT-03, BUG-A2-10): State Machine Matrix
# =============================================================================

class TestClus03And24StateMachineTransitions:
    """Verify state machine legal transitions for PAIRED -> DISCONNECTED, PAIRED -> ERROR, and PAIRING -> ERROR."""

    async def test_paired_to_disconnected_transition(self) -> None:
        """DeviceState.PAIRED can transition cleanly to DISCONNECTED."""
        sm = DeviceStateMachine(device_id="node_1", initial_state=DeviceState.DISCONNECTED)
        await sm.transition_to(DeviceState.CONNECTING)
        await sm.transition_to(DeviceState.PAIRING)
        await sm.transition_to(DeviceState.PAIRED)

        assert sm.current_state == DeviceState.PAIRED
        assert sm.can_transition_to(DeviceState.DISCONNECTED) is True

        await sm.transition_to(DeviceState.DISCONNECTED, reason="Peer abrupt disconnect")
        assert sm.current_state == DeviceState.DISCONNECTED

    async def test_paired_to_error_and_pairing_to_error(self) -> None:
        """DeviceState.PAIRED and PAIRING can transition cleanly to ERROR."""
        sm1 = DeviceStateMachine(device_id="node_2", initial_state=DeviceState.DISCONNECTED)
        await sm1.transition_to(DeviceState.CONNECTING)
        await sm1.transition_to(DeviceState.PAIRING)

        assert sm1.can_transition_to(DeviceState.ERROR) is True
        await sm1.transition_to(DeviceState.ERROR, reason="Handshake timeout in pairing")
        assert sm1.current_state == DeviceState.ERROR

        sm2 = DeviceStateMachine(device_id="node_3", initial_state=DeviceState.DISCONNECTED)
        await sm2.transition_to(DeviceState.CONNECTING)
        await sm2.transition_to(DeviceState.PAIRING)
        await sm2.transition_to(DeviceState.PAIRED)

        assert sm2.can_transition_to(DeviceState.ERROR) is True
        await sm2.transition_to(DeviceState.ERROR, reason="Fatal socket error while paired")
        assert sm2.current_state == DeviceState.ERROR


# =============================================================================
# 5. CLUS-05 (BUG-CRIT-05): Persistent Device Store Wipeout on Uninitialized Save
# =============================================================================

class TestClus05JsonStorageBackendUninitializedProtection:
    """Verify that calling save_device on an uninitialized JsonStorageBackend preserves existing disk state."""

    async def test_uninitialized_save_device_preserves_existing_data(self, tmp_path: Path) -> None:
        """Instantiating a new backend and saving a device without awaiting initialize() loads existing devices first."""
        storage_dir = tmp_path / "pengulink_storage"
        storage_dir.mkdir(parents=True, exist_ok=True)
        devices_file = storage_dir / "devices.json"

        # Seed pre-existing devices.json with dev1
        existing_data = {
            "dev_existing": {
                "device_id": "dev_existing",
                "name": "Existing Phone",
                "certificate_pem": "-----BEGIN CERTIFICATE-----\nABC\n-----END CERTIFICATE-----",
                "is_paired": True,
            }
        }
        devices_file.write_text(json.dumps(existing_data), encoding="utf-8")

        # Instantiate fresh backend WITHOUT calling initialize()
        backend = JsonStorageBackend(storage_dir)
        assert backend._initialized is False
        assert backend._devices_cache == {}

        # Save new device dev_new
        new_device_data = {
            "device_id": "dev_new",
            "name": "New Tablet",
            "is_paired": True,
        }
        await backend.save_device("dev_new", new_device_data)

        # Both devices must exist in cache and on disk
        dev_existing = await backend.get_device("dev_existing")
        assert dev_existing is not None
        assert dev_existing["name"] == "Existing Phone"
        assert dev_existing["is_paired"] is True

        dev_new = await backend.get_device("dev_new")
        assert dev_new is not None
        assert dev_new["name"] == "New Tablet"

        # Check raw disk content
        raw_disk = json.loads(devices_file.read_text(encoding="utf-8"))
        assert "dev_existing" in raw_disk
        assert "dev_new" in raw_disk
        assert raw_disk["dev_existing"]["name"] == "Existing Phone"

    async def test_uninitialized_get_and_list_loads_cache(self, tmp_path: Path) -> None:
        """get_device, list_devices, and delete_device automatically initialize if uninitialized."""
        storage_dir = tmp_path / "pengulink_storage_2"
        storage_dir.mkdir(parents=True, exist_ok=True)
        devices_file = storage_dir / "devices.json"

        seeded = {"dev_1": {"device_id": "dev_1", "name": "Phone 1"}}
        devices_file.write_text(json.dumps(seeded), encoding="utf-8")

        backend = JsonStorageBackend(storage_dir)
        assert backend._initialized is False

        # list_devices triggers _ensure_initialized
        devices = await backend.list_devices()
        assert len(devices) == 1
        assert devices[0]["name"] == "Phone 1"
        assert backend._initialized is True

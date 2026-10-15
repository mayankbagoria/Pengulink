"""tests.unit.test_plugin_remotecommands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Milestone 15: RemoteCommands Plugin, dual-format parser,
data models, event emissions, cache eviction, and strict zero-subprocess security invariant.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from pengulink.core.events import (
    DeviceUnpaired,
    EventBus,
    RemoteCommandExecutedEvent,
    RemoteCommandOutputEvent,
    RemoteCommandsReceivedEvent,
)
from pengulink.core.exceptions import ConnectionClosedError, RemoteCommandError
from pengulink.network.packet import Packet
from pengulink.plugins.remotecommands import (
    MAX_COMMAND_LENGTH,
    MAX_COMMANDS_PER_DEVICE,
    MAX_KEY_LENGTH,
    MAX_NAME_LENGTH,
    DeviceRemoteCommands,
    RemoteCommand,
    RemoteCommandsPlugin,
    create_execute_command_packet,
    create_request_commands_packet,
    parse_command_list,
)


# =====================================================================
# 1. Plugin Identity & Strict Zero-Subprocess Security Invariant
# =====================================================================

class TestPluginBasics:
    """Validate plugin identity, capabilities, and strict security invariant."""

    def test_plugin_identity_and_capabilities(self) -> None:
        plugin = RemoteCommandsPlugin()
        assert plugin.plugin_id == "remotecommands"
        assert set(plugin.incoming_capabilities) == {
            "kdeconnect.remotecommands",
            "kdeconnect.runcommand",
            "kdeconnect.runcommand.output",
        }
        assert set(plugin.outgoing_capabilities) == {
            "kdeconnect.remotecommands.request",
            "kdeconnect.remoteforcecommand",
            "kdeconnect.runcommand.request",
        }

    def test_security_invariant_zero_local_subprocess(self) -> None:
        """Inspect pengulink/plugins/remotecommands.py module AST.

        Verifies that subprocess, os.system, os.popen, pty, shlex, or asyncio.create_subprocess_*
        are NEVER imported or invoked in the plugin implementation.
        """
        plugin_file = Path("pengulink/plugins/remotecommands.py")
        assert plugin_file.is_file(), "Plugin file must exist"

        tree = ast.parse(plugin_file.read_text(encoding="utf-8"))
        prohibited_imports = {"subprocess", "os.system", "os.popen", "pty", "shlex"}
        prohibited_calls = {"system", "popen", "spawn", "create_subprocess_exec", "create_subprocess_shell"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in prohibited_imports, f"Prohibited import '{alias.name}' detected"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert mod not in prohibited_imports, f"Prohibited from-import '{mod}' detected"
                for alias in node.names:
                    assert alias.name not in prohibited_imports, f"Prohibited import name '{alias.name}' detected"
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in prohibited_calls:
                    pytest.fail(f"Prohibited call '{func.id}' detected in AST")
                elif isinstance(func, ast.Attribute) and func.attr in prohibited_calls:
                    pytest.fail(f"Prohibited attribute call '{func.attr}' detected in AST")


# =====================================================================
# 2. Resilient Dual-Format Parser Tests
# =====================================================================

class TestWireParsing:
    """Test wire command list parsing across formats and platforms."""

    def test_parse_dict_format_canonical_runcommand(self) -> None:
        body = {
            "commandList": {
                "cmd_lock": {"name": "Lock Screen", "command": "loginctl lock-session"},
                "cmd_suspend": {"name": "Suspend", "command": "systemctl suspend"},
            }
        }
        cmds = parse_command_list(body)
        assert len(cmds) == 2
        assert "cmd_lock" in cmds
        assert cmds["cmd_lock"].name == "Lock Screen"
        assert cmds["cmd_lock"].command == "loginctl lock-session"
        assert cmds["cmd_suspend"].name == "Suspend"

    def test_parse_string_format_json_escaped_runcommand(self) -> None:
        """Test Qt/C++ wire convention where commandList is a JSON-escaped string."""
        raw_json = '{"cmd_lock":{"command":"loginctl lock-session","name":"Lock Screen"},"cmd_suspend":{"command":"systemctl suspend","name":"Suspend"}}'
        body = {"commandList": raw_json}
        cmds = parse_command_list(body)
        assert len(cmds) == 2
        assert cmds["cmd_lock"].name == "Lock Screen"
        assert cmds["cmd_lock"].command == "loginctl lock-session"
        assert cmds["cmd_suspend"].name == "Suspend"

    def test_parse_dict_format_remotecommands_packet_type(self) -> None:
        body = {
            "commandList": {
                "cmd_ring": {"name": "Ring Device", "command": "ring"},
            }
        }
        cmds = parse_command_list(body)
        assert len(cmds) == 1
        assert cmds["cmd_ring"].name == "Ring Device"

    def test_parse_alternative_commands_key(self) -> None:
        body = {
            "commands": {
                "cmd_custom": {"name": "Custom Script", "command": "/bin/custom"},
            }
        }
        cmds = parse_command_list(body)
        assert len(cmds) == 1
        assert cmds["cmd_custom"].name == "Custom Script"

    def test_parse_list_format_commands(self) -> None:
        body = {
            "commandList": [
                {"key": "c1", "name": "Command One", "command": "c1"},
                {"id": "c2", "name": "Command Two", "command": "c2"},
            ]
        }
        cmds = parse_command_list(body)
        assert len(cmds) == 2
        assert cmds["c1"].name == "Command One"
        assert cmds["c2"].name == "Command Two"

    def test_parse_direct_body_dict(self) -> None:
        body = {
            "cmd_direct": {"name": "Direct Cmd", "command": "echo direct"},
        }
        cmds = parse_command_list(body)
        assert len(cmds) == 1
        assert cmds["cmd_direct"].name == "Direct Cmd"

    def test_parse_empty_command_list(self) -> None:
        assert parse_command_list({"commandList": {}}) == {}
        assert parse_command_list({"commandList": "{}"}) == {}
        assert parse_command_list({"commandList": ""}) == {}
        assert parse_command_list({}) == {}
        assert parse_command_list(None) == {}


# =====================================================================
# 3. Malformed and Boundary Payload Defenses
# =====================================================================

class TestMalformedAndAdversarialPayloads:
    """Validate boundary defenses against malformed or malicious packets."""

    def test_malformed_json_string_graceful_fallback(self) -> None:
        body = {"commandList": "{unclosed_broken_json: true"}
        cmds = parse_command_list(body)
        assert cmds == {}

    def test_null_and_missing_command_list(self) -> None:
        assert parse_command_list({"commandList": None}) == {}
        assert parse_command_list({"commands": None}) == {}
        assert parse_command_list("not a dict") == {}

    def test_non_dict_command_entries_skipped(self) -> None:
        body = {
            "commandList": {
                "cmd_valid": {"name": "Valid", "command": "ls"},
                "cmd_num": 12345,
                "cmd_str": "Simple String Name",
            }
        }
        cmds = parse_command_list(body)
        assert "cmd_valid" in cmds
        assert cmds["cmd_valid"].name == "Valid"
        # String entries are converted to RemoteCommand with name=string
        assert "cmd_str" in cmds
        assert cmds["cmd_str"].name == "Simple String Name"
        # Numeric values become string name
        assert "cmd_num" in cmds
        assert cmds["cmd_num"].name == "12345"

    def test_missing_name_defaults_to_key(self) -> None:
        body = {
            "commandList": {
                "cmd_noname": {"command": "echo hi"},
            }
        }
        cmds = parse_command_list(body)
        assert cmds["cmd_noname"].name == "cmd_noname"
        assert cmds["cmd_noname"].command == "echo hi"

    def test_clamped_field_lengths(self) -> None:
        huge_key = "k" * 500
        huge_name = "n" * 1000
        huge_cmd = "c" * 5000
        body = {
            "commandList": {
                huge_key: {"name": huge_name, "command": huge_cmd},
            }
        }
        cmds = parse_command_list(body)
        expected_key = huge_key[:MAX_KEY_LENGTH]
        assert expected_key in cmds
        cmd = cmds[expected_key]
        assert len(cmd.key) == MAX_KEY_LENGTH
        assert len(cmd.name) == MAX_NAME_LENGTH
        assert len(cmd.command) == MAX_COMMAND_LENGTH

    def test_max_commands_per_device_limit(self) -> None:
        body = {
            "commandList": {
                f"cmd_{i}": {"name": f"Cmd {i}", "command": "echo"}
                for i in range(MAX_COMMANDS_PER_DEVICE + 50)
            }
        }
        cmds = parse_command_list(body)
        assert len(cmds) == MAX_COMMANDS_PER_DEVICE

    async def test_unrecognized_packet_type_safe(self) -> None:
        plugin = RemoteCommandsPlugin()
        send_mock = AsyncMock()
        pkt = Packet(type="kdeconnect.unknown", body={"some": "data"})
        # Should not raise exception or store state
        await plugin.handle_packet(pkt, "dev1", send_mock)
        assert plugin.get_commands("dev1") == []

    async def test_ignore_incoming_execution_request(self) -> None:
        """Verify strict client-only invariant: incoming execution requests are discarded."""
        plugin = RemoteCommandsPlugin()
        send_mock = AsyncMock()
        pkt = Packet(type="kdeconnect.runcommand.request", body={"key": "malicious_cmd"})
        await plugin.handle_packet(pkt, "dev1", send_mock)
        send_mock.assert_not_called()
        assert plugin.get_commands("dev1") == []


# =====================================================================
# 4. EventBus Emissions
# =====================================================================

class TestEventBusEmissions:
    """Verify strongly typed EventBus publications."""

    async def test_emits_remote_commands_received_event(self) -> None:
        bus = EventBus()
        events: List[RemoteCommandsReceivedEvent] = []

        async def handler(evt: RemoteCommandsReceivedEvent) -> None:
            events.append(evt)

        bus.subscribe(RemoteCommandsReceivedEvent, handler)
        plugin = RemoteCommandsPlugin(event_bus=bus)

        pkt = Packet(
            type="kdeconnect.runcommand",
            body={
                "commandList": {
                    "cmd1": {"name": "Cmd 1", "command": "c1"},
                    "cmd2": {"name": "Cmd 2", "command": "c2"},
                }
            },
        )
        await plugin.handle_packet(pkt, "phone_1", AsyncMock())

        assert len(events) == 1
        evt = events[0]
        assert evt.device_id == "phone_1"
        assert evt.count == 2
        assert "cmd1" in evt.commands
        assert "cmd2" in evt.commands

    async def test_emits_remote_command_executed_event(self) -> None:
        bus = EventBus()
        events: List[RemoteCommandExecutedEvent] = []

        async def handler(evt: RemoteCommandExecutedEvent) -> None:
            events.append(evt)

        bus.subscribe(RemoteCommandExecutedEvent, handler)
        plugin = RemoteCommandsPlugin(event_bus=bus)

        send_mock = AsyncMock()
        await plugin.execute_command("phone_1", "cmd_lock", send_packet=send_mock)

        assert len(events) == 1
        assert events[0].device_id == "phone_1"
        assert events[0].key == "cmd_lock"

    async def test_emits_remote_command_output_event(self) -> None:
        bus = EventBus()
        events: List[RemoteCommandOutputEvent] = []

        async def handler(evt: RemoteCommandOutputEvent) -> None:
            events.append(evt)

        bus.subscribe(RemoteCommandOutputEvent, handler)
        plugin = RemoteCommandsPlugin(event_bus=bus)

        pkt = Packet(
            type="kdeconnect.runcommand.output",
            body={
                "key": "cmd_lock",
                "output": "Session locked",
                "exitCode": 0,
            },
        )
        await plugin.handle_packet(pkt, "phone_1", AsyncMock())

        assert len(events) == 1
        assert events[0].device_id == "phone_1"
        assert events[0].key == "cmd_lock"
        assert events[0].output == "Session locked"
        assert events[0].exit_code == 0


# =====================================================================
# 5. Cache Lifecycle and Unpair Eviction
# =====================================================================

class TestCacheLifecycleAndUnpair:
    """Verify cache isolation per device and unpairing state eviction."""

    async def test_device_scoped_cache_isolation(self) -> None:
        plugin = RemoteCommandsPlugin()
        p1 = Packet(type="kdeconnect.runcommand", body={"commandList": {"c1": {"name": "Phone 1 Cmd"}}})
        p2 = Packet(type="kdeconnect.runcommand", body={"commandList": {"c2": {"name": "Phone 2 Cmd"}}})

        await plugin.handle_packet(p1, "phone_1", AsyncMock())
        await plugin.handle_packet(p2, "phone_2", AsyncMock())

        p1_cmds = plugin.get_commands("phone_1")
        p2_cmds = plugin.get_commands("phone_2")

        assert len(p1_cmds) == 1 and p1_cmds[0].key == "c1"
        assert len(p2_cmds) == 1 and p2_cmds[0].key == "c2"

    async def test_unpair_purges_cache(self) -> None:
        bus = EventBus()
        plugin = RemoteCommandsPlugin(event_bus=bus)
        await plugin.start()

        pkt = Packet(type="kdeconnect.runcommand", body={"commandList": {"c1": {"name": "Test"}}})
        await plugin.handle_packet(pkt, "phone_1", AsyncMock())
        assert len(plugin.get_commands("phone_1")) == 1

        # Publish DeviceUnpaired event
        await bus.publish(DeviceUnpaired(device_id="phone_1"))
        assert len(plugin.get_commands("phone_1")) == 0

        await plugin.stop()

    async def test_start_and_stop_lifecycle(self) -> None:
        bus = EventBus()
        plugin = RemoteCommandsPlugin(event_bus=bus)
        await plugin.start()

        pkt = Packet(type="kdeconnect.runcommand", body={"commandList": {"c1": {"name": "Test"}}})
        await plugin.handle_packet(pkt, "phone_1", AsyncMock())
        assert len(plugin.get_commands("phone_1")) == 1

        await plugin.stop()
        assert len(plugin.get_commands("phone_1")) == 0


# =====================================================================
# 6. Outbound Packet Generation
# =====================================================================

class TestOutboundPacketGeneration:
    """Test packet construction and transmission."""

    def test_build_request_command_list_packet(self) -> None:
        pkt = create_request_commands_packet(legacy=False)
        assert pkt.type == "kdeconnect.runcommand.request"
        assert pkt.body == {"requestCommandList": True}

        legacy_pkt = create_request_commands_packet(legacy=True)
        assert legacy_pkt.type == "kdeconnect.remotecommands.request"
        assert legacy_pkt.body == {"requestCommandList": True}

    def test_build_execute_command_packet(self) -> None:
        pkt = create_execute_command_packet("cmd_lock")
        assert pkt.type == "kdeconnect.remoteforcecommand"
        assert pkt.body == {"key": "cmd_lock"}

        run_pkt = create_execute_command_packet("cmd_lock", packet_type="kdeconnect.runcommand.request")
        assert run_pkt.type == "kdeconnect.runcommand.request"
        assert run_pkt.body == {"key": "cmd_lock"}

    async def test_execute_command_empty_key_raises(self) -> None:
        plugin = RemoteCommandsPlugin()
        with pytest.raises(RemoteCommandError, match="must be a non-empty string"):
            await plugin.execute_command("phone_1", "")

        with pytest.raises(RemoteCommandError, match="must be a non-empty string"):
            await plugin.execute_command("phone_1", "   ")

    async def test_execute_command_send_failure_raises(self) -> None:
        plugin = RemoteCommandsPlugin()
        # No sender registered
        with pytest.raises(ConnectionClosedError, match="No active connection"):
            await plugin.execute_command("phone_1", "cmd_lock")

    async def test_request_command_list_send(self) -> None:
        plugin = RemoteCommandsPlugin()
        send_mock = AsyncMock()
        pkt = await plugin.request_command_list("phone_1", send_packet=send_mock)
        assert pkt.type == "kdeconnect.runcommand.request"
        send_mock.assert_called_once()

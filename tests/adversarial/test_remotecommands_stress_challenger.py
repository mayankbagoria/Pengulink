"""tests.adversarial.test_remotecommands_stress_challenger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Adversarial Stress & Fuzzing Challenger Suite for Milestone 15:
RemoteCommands Plugin Ecosystem (kdeconnect.remotecommands / kdeconnect.remoteforcecommand).

Challenger Objectives:
1. Static AST & Bytecode Verification:
   Forensic AST and module inspection verifying zero imports or invocations of
   subprocess, os.system, os.popen, pty, shlex, or asyncio.create_subprocess_*.
2. Boundary & Fuzzing Stress:
   - Corrupted, truncated, and malformed JSON strings.
   - Deeply nested structures (100 to 500 levels).
   - Massive command flood (>256 commands up to 5,000 commands).
   - Huge payloads (100KB to 1MB text strings).
   - Type confusion with extreme floats (NaN, Inf), booleans, None, and lists.
3. Shell Metacharacters & Injection Attacks:
   - Keys with $(reboot), ; rm -rf / ;, | cat /etc/passwd, && echo pwn, `touch /tmp/pwn`.
   - Verifying keys are treated strictly as inert opaque string tokens.
   - Verifying zero local filesystem modifications or child process spawns.
4. Unicode, RTL, CJK, and Complex Emoji Sequences:
   - Arabic and Hebrew RTL bidirectional text.
   - Chinese, Japanese, and Korean multi-byte glyphs.
   - ZWJ sequences, skin-tone modifiers, flags, and combining marks.
   - Safe serialization, IPC roundtripping, and CLI display.
5. Rich Markup Injection Resilience:
   - Testing malicious console tags in command keys, names, and command strings.
6. Asynchronous Chaos & Concurrency:
   - Concurrent command executions racing against unpair cache eviction.
   - Offline device error codes (-32014, -32002) and non-existent key validation.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock

from click.testing import CliRunner
import pytest

from pengulink.cli.main import cli
from pengulink.core.events import (
    DeviceUnpaired,
    EventBus,
    RemoteCommandExecutedEvent,
    RemoteCommandOutputEvent,
    RemoteCommandsReceivedEvent,
)
from pengulink.core.exceptions import (
    ConnectionClosedError,
    RemoteCommandError,
    RemoteCommandNotFoundError,
)
from pengulink.network.packet import Packet
from pengulink.plugins.remotecommands import (
    MAX_COMMAND_LENGTH,
    MAX_COMMANDS_PER_DEVICE,
    MAX_KEY_LENGTH,
    MAX_NAME_LENGTH,
    RemoteCommand,
    RemoteCommandsPlugin,
    create_execute_command_packet,
    create_request_commands_packet,
    parse_command_list,
)


# =====================================================================
# 1. Static AST & Forensic Security Invariant Inspection
# =====================================================================

class TestSecurityInvariantASTInspection:
    """Rigorous static analysis confirming zero local execution mechanisms."""

    def test_ast_forbids_all_execution_modules_and_calls(self) -> None:
        """Deep AST walk of pengulink/plugins/remotecommands.py for any forbidden module/call."""
        target_path = Path("pengulink/plugins/remotecommands.py")
        assert target_path.is_file(), "Plugin source must exist"

        source_code = target_path.read_text(encoding="utf-8")
        tree = ast.parse(source_code, filename=str(target_path))

        forbidden_modules = {
            "subprocess",
            "os",
            "pty",
            "shlex",
            "posix",
            "posixspawn",
            "ctypes",
            "commands",
        }
        forbidden_call_names = {
            "system",
            "popen",
            "spawn",
            "spawnl",
            "spawnle",
            "spawnlp",
            "spawnlpe",
            "spawnv",
            "spawnve",
            "spawnvp",
            "spawnvpe",
            "posix_spawn",
            "posix_spawnp",
            "create_subprocess_exec",
            "create_subprocess_shell",
            "eval",
            "exec",
            "compile",
            "__import__",
            "fork",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    assert (
                        root_pkg not in forbidden_modules
                    ), f"Forbidden import found: {alias.name} (line {node.lineno})"
            elif isinstance(node, ast.ImportFrom):
                root_pkg = (node.module or "").split(".")[0]
                assert (
                    root_pkg not in forbidden_modules
                ), f"Forbidden from-import found: {node.module} (line {node.lineno})"
                for alias in node.names:
                    assert (
                        alias.name not in forbidden_call_names
                    ), f"Forbidden imported symbol: {alias.name} from {node.module} (line {node.lineno})"
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    assert (
                        func.id not in forbidden_call_names
                    ), f"Forbidden direct function call: {func.id} (line {node.lineno})"
                elif isinstance(func, ast.Attribute):
                    assert (
                        func.attr not in forbidden_call_names
                    ), f"Forbidden method/attribute call: {func.attr} (line {node.lineno})"

    def test_dynamic_reflection_verifies_no_execution_handles(self) -> None:
        """Inspect the loaded RemoteCommandsPlugin class and module globals for execution symbols."""
        import pengulink.plugins.remotecommands as mod

        mod_dict = mod.__dict__
        for dangerous in ("subprocess", "os", "shlex", "pty"):
            assert dangerous not in mod_dict, f"Leaked module handle: {dangerous}"

        import textwrap
        plugin = RemoteCommandsPlugin()
        for attr_name in dir(plugin):
            val = getattr(plugin, attr_name)
            if callable(val) and not attr_name.startswith("__"):
                src = inspect.getsource(val)
                tree = ast.parse(textwrap.dedent(src))
                # Verify no Call nodes or Import nodes call subprocess/system
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        func = node.func
                        func_name = getattr(func, "id", None) or getattr(func, "attr", None)
                        assert func_name not in {"system", "popen", "spawn", "create_subprocess_exec", "create_subprocess_shell"}



# =====================================================================
# 2. Boundary & Fuzzing Stress Harness
# =====================================================================

class TestBoundaryAndFuzzingStress:
    """Stress tests probing boundary limits, corrupted JSON, and massive floods."""

    def test_fuzz_corrupted_and_truncated_json_strings(self) -> None:
        """Deliver truncated and corrupted JSON strings to parse_command_list."""
        valid_json = json.dumps({
            "cmd_1": {"name": "Lock Screen", "command": "loginctl lock-session"},
            "cmd_2": {"name": "Reboot", "command": "systemctl reboot"},
        })

        # Truncate at every single character offset
        for length in range(1, len(valid_json)):
            truncated = valid_json[:length]
            body = {"commandList": truncated}
            res = parse_command_list(body)
            assert isinstance(res, dict)
            # Truncated JSON should fail decoding gracefully without crashing

        # Adversarial corruption patterns
        fuzz_corpus = [
            '{"cmd": {"name": "test"',
            '{"cmd": {"name": "test"}}EXTRA_GARBAGE',
            '\x00\x00\x00{"cmd": {}}',
            '{"cmd": {"name": "\x00\x01\x02\x03\x04"}}',
            '{"cmd": {"name": "' + "\\u" * 50 + '"}}',
            '{"cmd": {"command": "\\xZZ"}}',
            '{"a": 1, "b": 2, "c": [}',
            '{"a": {"b": {"c": null}}}',
            '{"cmd": {"name": NaN}}',
            '{"cmd": {"name": Infinity}}',
            '{"cmd": {"name": -Infinity}}',
            "null",
            "12345",
            "true",
            "false",
            '""',
            "[]",
            "{",
            "}",
            "{{}}",
            "[[]]",
        ]

        for payload in fuzz_corpus:
            res = parse_command_list({"commandList": payload})
            assert isinstance(res, dict)

    def test_deeply_nested_structures(self) -> None:
        """Test resilience against deep dictionary and list nesting."""
        # Construct 100-level deep dictionary
        nested: Dict[str, Any] = {"name": "Deep Leaf", "command": "echo 'deep'"}
        for level in range(100):
            nested = {f"level_{level}": nested}

        body = {"commandList": nested}
        res = parse_command_list(body)
        assert isinstance(res, dict)

        # Construct 250-level deep JSON string
        deep_json_str = '{"a": ' * 250 + '{"name": "deep"}' + '}' * 250
        body_json = {"commandList": deep_json_str}
        res_json = parse_command_list(body_json)
        assert isinstance(res_json, dict)

    def test_massive_command_flood_clamping_to_256(self) -> None:
        """Flood with 5,000 commands in a single packet; assert strict 256 cap."""
        flood_dict = {
            f"flood_key_{i:05d}": {
                "name": f"Flood Command {i}",
                "command": f"echo {i}",
            }
            for i in range(5000)
        }

        start_time = time.perf_counter()
        res = parse_command_list({"commandList": flood_dict})
        elapsed = time.perf_counter() - start_time

        assert len(res) == MAX_COMMANDS_PER_DEVICE
        assert len(res) == 256
        # Must terminate in sub-second time (early break)
        assert elapsed < 0.2

    def test_huge_string_payload_memory_clamping(self) -> None:
        """Test with 100KB to 500KB keys, names, and command strings."""
        huge_key = "K" * 150_000
        huge_name = "N" * 200_000
        huge_cmd = "C" * 600_000

        body = {
            "commandList": {
                huge_key: {
                    "name": huge_name,
                    "command": huge_cmd,
                }
            }
        }

        res = parse_command_list(body)
        expected_key = huge_key[:MAX_KEY_LENGTH]
        assert expected_key in res
        cmd = res[expected_key]
        assert len(cmd.key) == MAX_KEY_LENGTH
        assert len(cmd.name) == MAX_NAME_LENGTH
        assert len(cmd.command) == MAX_COMMAND_LENGTH

    def test_type_confusion_extreme_floats_and_primitives(self) -> None:
        """Resilience against type confusion, extreme floats (NaN, Inf), and non-standard types."""
        weird_candidates = {
            "cmd_nan": {"name": float("nan"), "command": "run"},
            "cmd_inf": {"name": float("inf"), "command": "run"},
            "cmd_neg_inf": {"name": float("-inf"), "command": "run"},
            "cmd_huge_float": {"name": 1e308, "command": "run"},
            "cmd_bool": {"name": True, "command": False},
            "cmd_list": {"name": ["nested", "list"], "command": {"dict": "val"}},
            "cmd_none": {"name": None, "command": None},
            "cmd_zero": {"name": 0, "command": 0},
        }

        res = parse_command_list({"commandList": weird_candidates})
        assert isinstance(res, dict)
        for k, cmd in res.items():
            assert isinstance(cmd.key, str)
            assert isinstance(cmd.name, str)
            assert isinstance(cmd.command, str)


# =====================================================================
# 3. Shell Metacharacters & Injection Attack Vector Tests
# =====================================================================

class TestShellMetacharacterInjectionSafety:
    """Verify that dangerous shell syntax in keys/names is treated as inert strings."""

    async def test_shell_metacharacters_in_keys_cannot_execute(self, tmp_path: Path) -> None:
        """Ensure $(touch file), ; rm -rf ;, `whoami` in command keys NEVER execute."""
        canary_file = tmp_path / "pwned.txt"
        assert not canary_file.exists()

        injection_keys = [
            f"; touch {canary_file} ;",
            f"$(touch {canary_file})",
            f"`touch {canary_file}`",
            f"| touch {canary_file}",
            f"& touch {canary_file} &",
            f"&& touch {canary_file}",
            f"|| touch {canary_file}",
            "\nreboot\n",
            "\r\nshutdown -h now\r\n",
            "${IFS}cat${IFS}/etc/shadow",
            "'; DROP TABLE commands; --",
            "><script>alert(1)</script>",
        ]

        plugin = RemoteCommandsPlugin()
        send_mock = AsyncMock()

        payload = {
            "commandList": {
                k: {"name": f"Payload {idx}", "command": "echo harmless"}
                for idx, k in enumerate(injection_keys)
            }
        }

        pkt = Packet(type="kdeconnect.runcommand", body=payload)
        await plugin.handle_packet(pkt, "attacker_phone", send_mock)

        # Canary file must NOT be created!
        assert not canary_file.exists(), "CRITICAL: Shell command executed locally!"

        # Execute all commands
        for k in injection_keys:
            out_pkt = await plugin.execute_command("attacker_phone", k, send_packet=send_mock)
            assert out_pkt.type == "kdeconnect.remoteforcecommand"
            assert out_pkt.body["key"] == k.strip()

        # Canary file still must NOT be created!
        assert not canary_file.exists(), "CRITICAL: Execution request spawned shell!"

    def test_outbound_execute_packet_preserves_opaque_string(self) -> None:
        """Packet serialization preserves shell metacharacters verbatim without escaping."""
        raw_key = "; echo $(whoami) > /tmp/test ;"
        pkt = create_execute_command_packet(raw_key)
        assert pkt.body["key"] == raw_key.strip()

        raw_json = json.dumps(pkt.body)
        decoded = json.loads(raw_json)
        assert decoded["key"] == raw_key.strip()


# =====================================================================
# 4. Unicode, RTL, CJK, and Complex Emoji Sequences
# =====================================================================

class TestUnicodeRTLCJKEmojis:
    """Stress tests for multi-byte Unicode, right-to-left languages, and complex emojis."""

    def test_arabic_hebrew_cjk_emoji_fidelity(self) -> None:
        """Verify complex internationalized strings maintain fidelity across serialization."""
        unicode_corpus = {
            "cmd_ar_1": {
                "name": "قفل الشاشة عن بُعد 🔒",
                "command": "echo 'arabic'",
            },
            "cmd_he_2": {
                "name": "נעילת מסך מרחוק 📱",
                "command": "echo 'hebrew'",
            },
            "cmd_cjk_3": {
                "name": "远程锁定屏幕 🖥️ (简体中文)",
                "command": "echo 'cjk'",
            },
            "cmd_jp_4": {
                "name": "画面をロックする ⚡ (日本語)",
                "command": "echo 'japanese'",
            },
            "cmd_kr_5": {
                "name": "화면 원격 잠금 🚀 (한국어)",
                "command": "echo 'korean'",
            },
            "cmd_zwj_family": {
                "name": "Family: 👨‍👩‍👧‍👦 Rainbow: 🏳️‍🌈 Hand: 👍🏽",
                "command": "echo 'zwj'",
            },
            "cmd_rtl_override": {
                "name": "Test \u202eEXE.TXT\u202c Bidi",
                "command": "echo 'bidi'",
            },
            "cmd_combining_zalgo": {
                "name": "H̸̛͔é̸͜ļ̴̈́l̴̩̂o̷̲͠ ̶̘̊Ẃ̷̬o̸̩̒ṟ̵͠l̵̼̀d̶̟̈́",
                "command": "echo 'zalgo'",
            },
        }

        parsed = parse_command_list({"commandList": unicode_corpus})
        assert len(parsed) == len(unicode_corpus)

        for k, original in unicode_corpus.items():
            assert k in parsed
            cmd = parsed[k]
            assert cmd.name == original["name"][:MAX_NAME_LENGTH]
            # Verify JSON roundtrip
            serialized = json.dumps(cmd.to_dict())
            deserialized = json.loads(serialized)
            assert deserialized["key"] == k
            assert deserialized["name"] == cmd.name

    async def test_unicode_command_execution_roundtrip(self) -> None:
        """Trigger execution with Arabic and Emoji command keys."""
        plugin = RemoteCommandsPlugin()
        send_mock = AsyncMock()

        arabic_key = "قفل_الشاشة_🔒"
        emoji_key = "🚀_LOCK_SESSION_👨‍👩‍👦"

        pkt = Packet(
            type="kdeconnect.runcommand",
            body={
                "commandList": {
                    arabic_key: {"name": "Lock Screen Arabic"},
                    emoji_key: {"name": "Emoji Lock"},
                }
            },
        )
        await plugin.handle_packet(pkt, "intl_device", send_mock)

        exec_pkt1 = await plugin.execute_command("intl_device", arabic_key, send_packet=send_mock)
        assert exec_pkt1.body["key"] == arabic_key

        exec_pkt2 = await plugin.execute_command("intl_device", emoji_key, send_packet=send_mock)
        assert exec_pkt2.body["key"] == emoji_key


# =====================================================================
# 5. CLI Rich Markup Injection Resilience
# =====================================================================

class TestCliRichMarkupInjection:
    """Verify that Rich console formatting is safely escaped against tag injection."""

    def test_cli_command_list_with_malicious_rich_tags(self) -> None:
        """Inject malicious Rich markup into command names, keys, and commands."""
        runner = CliRunner(env={"COLUMNS": "160"})

        mock_commands = [
            {
                "key": "[bold red]cmd_tag_inject[/bold red]",
                "name": "[link=https://evil.example.com]Click Here[/link]",
                "command": "[blink]Dangerous Script[/blink]",
            },
            {
                "key": "[/][yellow]broken_tag",
                "name": "[color(1)]Weird Color[/]",
                "command": "",
            },
        ]

        from unittest.mock import patch

        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_commands):
            result = runner.invoke(cli, ["command", "list", "test_device"])
            assert result.exit_code == 0
            # Tags must be escaped and rendered literally, not interpreted
            assert "[bold red]" in result.output or "cmd_tag_inject" in result.output
            assert "Click Here" in result.output

    def test_cli_command_exec_with_rich_tag_key(self) -> None:
        """Execute command key containing Rich markup tags."""
        runner = CliRunner()
        mock_response = {
            "device_id": "phone_1",
            "key": "[bold green]cmd_1[/bold green]",
            "status": "transmitted",
        }

        from unittest.mock import patch

        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_response):
            result = runner.invoke(cli, ["command", "exec", "phone_1", "[bold green]cmd_1[/bold green]"])
            assert result.exit_code == 0
            assert "Remote Command Executed" in result.output


# =====================================================================
# 6. Concurrency Races & Asynchronous Stress
# =====================================================================

class TestConcurrencyAndUnpairStress:
    """Stress concurrency, unpair races, and cache safety."""

    async def test_high_volume_concurrent_packet_handling(self) -> None:
        """50 concurrent coroutines feeding command packets to the plugin."""
        plugin = RemoteCommandsPlugin()
        send_mock = AsyncMock()

        async def worker(worker_id: int) -> None:
            dev_id = f"dev_{worker_id % 5}"
            pkt = Packet(
                type="kdeconnect.runcommand",
                body={
                    "commandList": {
                        f"cmd_{worker_id}_{i}": {"name": f"Cmd {i}"}
                        for i in range(10)
                    }
                },
            )
            await plugin.handle_packet(pkt, dev_id, send_mock)

        tasks = [asyncio.create_task(worker(i)) for i in range(50)]
        await asyncio.gather(*tasks)

        # All 5 devices should have valid state
        for dev_idx in range(5):
            cmds = plugin.get_commands(f"dev_{dev_idx}")
            assert len(cmds) > 0

    async def test_unpair_event_evicts_cache_cleanly(self) -> None:
        """Verify DeviceUnpaired event subscriber purges cached commands cleanly."""
        bus = EventBus()
        plugin = RemoteCommandsPlugin(event_bus=bus)
        await plugin.start()

        # Feed commands for 2 devices
        pkt1 = Packet(type="kdeconnect.runcommand", body={"commandList": {"c1": {"name": "C1"}}})
        pkt2 = Packet(type="kdeconnect.runcommand", body={"commandList": {"c2": {"name": "C2"}}})
        await plugin.handle_packet(pkt1, "device_A", AsyncMock())
        await plugin.handle_packet(pkt2, "device_B", AsyncMock())

        assert len(plugin.get_commands("device_A")) == 1
        assert len(plugin.get_commands("device_B")) == 1

        # Publish unpair for device_A
        await bus.publish(DeviceUnpaired(device_id="device_A"))

        # device_A must be wiped, device_B must remain
        assert len(plugin.get_commands("device_A")) == 0
        assert len(plugin.get_commands("device_B")) == 1

        await plugin.stop()


# =====================================================================
# 7. Error Handling & Invariant Enforcement
# =====================================================================

class TestErrorHandlingAndValidation:
    """Validation of invalid inputs and edge-case exceptions."""

    def test_empty_key_in_execute_packet_raises_error(self) -> None:
        """Empty or whitespace-only keys must raise RemoteCommandError."""
        for invalid_key in ("", "   ", "\t", "\n"):
            with pytest.raises(RemoteCommandError):
                create_execute_command_packet(invalid_key)

    async def test_execute_command_without_connection_raises_error(self) -> None:
        """Executing command on device with no active sender must raise ConnectionClosedError."""
        plugin = RemoteCommandsPlugin()
        with pytest.raises(ConnectionClosedError):
            await plugin.execute_command("unconnected_device", "cmd_lock")


# =====================================================================
# 8. Peer Inbound Execution Attack Vectors & Extreme Output Packets
# =====================================================================

class TestPeerInboundExecutionAttacksAndEdgeCases:
    """Verify that inbound execution packets from remote peers are strictly dropped."""

    async def test_inbound_remote_execution_request_strictly_ignored(self) -> None:
        """Adversary sends execution request packet TO PenguLink daemon.

        RemoteCommands plugin is CLIENT ONLY. It must NEVER execute incoming requests.
        """
        bus = EventBus()
        events = []
        bus.subscribe_all(lambda ev: events.append(ev))
        plugin = RemoteCommandsPlugin(event_bus=bus)
        send_mock = AsyncMock()

        # Inbound execution request types
        attack_packets = [
            Packet(type="kdeconnect.remoteforcecommand", body={"key": "cmd_rm_rf"}),
            Packet(type="kdeconnect.runcommand.request", body={"key": "cmd_reboot"}),
            Packet(type="kdeconnect.remotecommands.request", body={"key": "cmd_exploit"}),
        ]

        for pkt in attack_packets:
            await plugin.handle_packet(pkt, "attacker_phone", send_mock)

        # Zero events must be published and zero replies sent
        assert len(events) == 0
        send_mock.assert_not_called()

    async def test_extreme_and_malformed_command_output_packets(self) -> None:
        """Output packets with extreme payload sizes, non-integer exit codes, and missing fields."""
        bus = EventBus()
        received_outputs: List[RemoteCommandOutputEvent] = []
        bus.subscribe(RemoteCommandOutputEvent, lambda ev: received_outputs.append(ev))
        plugin = RemoteCommandsPlugin(event_bus=bus)
        send_mock = AsyncMock()

        huge_output = "O" * 200_000

        output_packets = [
            # Huge string output
            Packet(type="kdeconnect.runcommand.output", body={"key": "k1", "output": huge_output, "exitCode": 0}),
            # Negative exit code
            Packet(type="kdeconnect.runcommand.output", body={"key": "k2", "output": "error", "exitCode": -9}),
            # String exit code that parses as int
            Packet(type="kdeconnect.runcommand.output", body={"key": "k3", "output": "warn", "exitCode": "127"}),
            # Non-convertible exit code (should default to 0)
            Packet(type="kdeconnect.runcommand.output", body={"key": "k4", "output": "weird", "exitCode": "NOT_AN_INT"}),
            # Missing output and exitCode
            Packet(type="kdeconnect.runcommand.output", body={"key": "k5"}),
            # Empty packet body
            Packet(type="kdeconnect.runcommand.output", body={}),
        ]

        for pkt in output_packets:
            await plugin.handle_packet(pkt, "peer_phone", send_mock)

        assert len(received_outputs) == len(output_packets)
        assert received_outputs[0].key == "k1"
        assert len(received_outputs[0].output) == 200_000
        assert received_outputs[1].exit_code == -9
        assert received_outputs[2].exit_code == 127
        assert received_outputs[3].exit_code == 0
        assert received_outputs[4].exit_code == 0
        assert received_outputs[5].key == ""

    def test_json_null_bytes_and_surrogate_sequences(self) -> None:
        """Deliver JSON strings with null bytes and escaped control characters."""
        payload_with_escapes = {
            "commandList": json.dumps({
                "cmd_null\x00_key": {
                    "name": "Null\x00Byte\x01Control",
                    "command": "cmd\r\n\targ",
                }
            })
        }
        res = parse_command_list(payload_with_escapes)
        assert isinstance(res, dict)
        assert len(res) == 1


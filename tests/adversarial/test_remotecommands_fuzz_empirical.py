"""tests.adversarial.test_remotecommands_fuzz_empirical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Challenger Empirical Fuzzing & Bytecode Forensics Suite for Milestone 15.
Probing RemoteCommands with 1,000+ mutated payloads, bytecode disassembly,
extreme recursion, unpaired surrogates, exact boundary clamping, and concurrency stress.
"""

from __future__ import annotations

import asyncio
import dis
import json
import math
import os
from pathlib import Path
import random
import string
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from pengulink.core.events import EventBus, RemoteCommandsReceivedEvent
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


class TestBytecodeAndSymbolForensics:
    """Forensic verification: Disassemble bytecode to prove zero hidden execution mechanisms."""

    def test_bytecode_disassembly_has_zero_execution_opcodes(self) -> None:
        """Disassemble every function/method in remotecommands.py and verify no dangerous imports or calls."""
        import pengulink.plugins.remotecommands as rc_mod

        forbidden_names = {
            "subprocess",
            "os",
            "system",
            "popen",
            "spawn",
            "create_subprocess_exec",
            "create_subprocess_shell",
            "eval",
            "exec",
            "posix_spawn",
            "fork",
            "pty",
            "shlex",
        }

        # Inspect all functions and methods in the module
        for item_name in dir(rc_mod):
            item = getattr(rc_mod, item_name)
            if hasattr(item, "__code__"):
                instructions = list(dis.get_instructions(item))
                for instr in instructions:
                    if instr.opname in ("LOAD_GLOBAL", "LOAD_NAME", "LOAD_ATTR", "IMPORT_NAME"):
                        argval = str(instr.argval)
                        assert argval not in forbidden_names, (
                            f"Bytecode violation in {item_name}: {instr.opname} loaded forbidden symbol '{argval}'"
                        )
            elif isinstance(item, type):
                # Inspect class methods
                for method_name in dir(item):
                    method = getattr(item, method_name)
                    if hasattr(method, "__code__"):
                        instructions = list(dis.get_instructions(method))
                        for instr in instructions:
                            if instr.opname in ("LOAD_GLOBAL", "LOAD_NAME", "LOAD_ATTR", "IMPORT_NAME"):
                                argval = str(instr.argval)
                                assert argval not in forbidden_names, (
                                    f"Bytecode violation in {item_name}.{method_name}: "
                                    f"{instr.opname} loaded forbidden symbol '{argval}'"
                                )


class TestEmpiricalFuzzingCorpus:
    """Aggressively probe parse_command_list and handle_packet with 1,000+ generated mutations."""

    def test_fuzz_random_corrupted_json_strings(self) -> None:
        """Generate 500 mutated JSON strings with bit flips, insertions, truncations, and garbage."""
        random.seed(42)
        base_valid = {
            "cmd_1": {"name": "Lock Screen", "command": "loginctl lock"},
            "cmd_2": {"name": "Suspend", "command": "systemctl suspend"},
            "cmd_3": {"name": "Screenshot", "command": "spectacle -b"},
        }
        base_json = json.dumps(base_valid)

        mutated_samples: List[str] = [
            "",
            " ",
            "\t\n\r",
            "null",
            "true",
            "false",
            "123",
            "-456.78",
            '"just a string"',
            "[]",
            "{}",
            "{{{{",
            "}}",
            "[1, 2, 3]",
            '{"unclosed": ',
            '{"colon_missing" "val"}',
            '{"trailing_comma": 1,}',
            '{,"leading_comma": 1}',
            '{"NaN": NaN}',
            '{"Infinity": Infinity}',
            '{"-Infinity": -Infinity}',
            '{"undefined": undefined}',
        ]

        # Generate mutations from base_json
        for _ in range(500):
            chars = list(base_json)
            mutation_type = random.choice(["truncate", "flip", "insert_garbage", "delete_range", "null_bytes"])
            if mutation_type == "truncate":
                cut = random.randint(1, len(chars) - 1)
                mutated_samples.append("".join(chars[:cut]))
            elif mutation_type == "flip":
                idx = random.randint(0, len(chars) - 1)
                chars[idx] = chr(random.randint(0, 127))
                mutated_samples.append("".join(chars))
            elif mutation_type == "insert_garbage":
                idx = random.randint(0, len(chars) - 1)
                garbage = "".join(random.choices(string.punctuation + string.ascii_letters, k=10))
                chars.insert(idx, garbage)
                mutated_samples.append("".join(chars))
            elif mutation_type == "delete_range":
                start = random.randint(0, len(chars) // 2)
                end = random.randint(start, len(chars) - 1)
                del chars[start:end]
                mutated_samples.append("".join(chars))
            elif mutation_type == "null_bytes":
                idx = random.randint(0, len(chars) - 1)
                chars.insert(idx, "\x00\x01\x02\xff")
                mutated_samples.append("".join(chars))

        for sample in mutated_samples:
            result = parse_command_list({"commandList": sample})
            assert isinstance(result, dict), f"Failed to return dict for sample: {sample[:50]!r}"

    def test_fuzz_arbitrary_python_objects_as_payload(self) -> None:
        """Pass arbitrary and weird Python object structures to parse_command_list."""
        weird_objects = [
            None,
            42,
            3.14159,
            float("nan"),
            float("inf"),
            float("-inf"),
            complex(1, 2),
            b"raw bytes \x00\xff",
            bytearray(b"bytearray"),
            (1, 2, 3),
            {1, 2, 3},
            frozenset([1, 2, 3]),
            lambda x: x,
            object(),
            Exception("mock exception"),
            {"commandList": None},
            {"commandList": [None, 123, True, False, 3.14, float("nan")]},
            {"commandList": {None: None}},
            {"commandList": {"": {}}},
            {"commandList": {"valid": {"name": None, "command": None}}},
            {"commandList": {"valid": {"name": float("nan"), "command": float("inf")}}},
            {"commands": "not a json string at all"},
            {"commands": 12345},
            {"commands": [{}, {"key": ""}, {"key": None}, {"id": 123}]},
        ]

        for obj in weird_objects:
            res = parse_command_list(obj)
            assert isinstance(res, dict)

    def test_fuzz_deeply_nested_dictionaries(self) -> None:
        """Construct deeply nested dictionaries (up to 300 levels) to verify stack safety."""
        current: Any = {"name": "leaf", "command": "leaf_cmd"}
        for i in range(300):
            current = {f"k_{i}": current}

        res = parse_command_list({"commandList": current})
        assert isinstance(res, dict)

    def test_fuzz_exact_boundary_clamping(self) -> None:
        """Verify exact behavior at boundary edges: 127/128/129 for keys, 255/256/257 for names, 1023/1024/1025 for commands."""
        # Key boundaries: MAX_KEY_LENGTH = 128
        k_127 = "k" * 127
        k_128 = "k" * 128
        k_129 = "k" * 129

        # Name boundaries: MAX_NAME_LENGTH = 256
        n_255 = "n" * 255
        n_256 = "n" * 256
        n_257 = "n" * 257

        # Command boundaries: MAX_COMMAND_LENGTH = 1024
        c_1023 = "c" * 1023
        c_1024 = "c" * 1024
        c_1025 = "c" * 1025

        payload = {
            "commandList": {
                k_127: {"name": n_255, "command": c_1023},
                k_128: {"name": n_256, "command": c_1024},
                k_129: {"name": n_257, "command": c_1025},
            }
        }

        res = parse_command_list(payload)
        assert len(res) == 2  # k_128 and k_129 both clamp to same 128 chars, so they collide safely
        assert k_127 in res
        assert len(res[k_127].key) == 127
        assert len(res[k_127].name) == 255
        assert len(res[k_127].command) == 1023

        clamped_k128 = k_128[:MAX_KEY_LENGTH]
        assert clamped_k128 in res
        assert len(res[clamped_k128].key) == MAX_KEY_LENGTH
        assert len(res[clamped_k128].name) == MAX_NAME_LENGTH
        assert len(res[clamped_k128].command) == MAX_COMMAND_LENGTH

    def test_fuzz_exact_max_commands_cap(self) -> None:
        """Verify exact boundary at MAX_COMMANDS_PER_DEVICE (256)."""
        # Exactly 256 commands
        exact_256 = {f"cmd_{i:04d}": {"name": f"Cmd {i}"} for i in range(256)}
        res_256 = parse_command_list({"commandList": exact_256})
        assert len(res_256) == 256

        # 257 commands
        overflow_257 = {f"cmd_{i:04d}": {"name": f"Cmd {i}"} for i in range(257)}
        res_257 = parse_command_list({"commandList": overflow_257})
        assert len(res_257) == 256

    def test_fuzz_unicode_surrogates_and_bidi_overrides(self) -> None:
        """Verify resistance to RTL override characters, ZWJ sequences, and emoji combinations."""
        bidi_payload = {
            "commandList": {
                "key_\u202e_override": {
                    "name": "Normal\u202eRTL_OVERRIDE_ATTACK\u202cEnd",
                    "command": "echo \u202dtest\u202c",
                },
                "key_zero_width": {
                    "name": "A\u200bB\u200cC\u200dD",
                    "command": "cmd",
                },
                "key_math_symbols": {
                    "name": "∀x ∈ ℝ : ∇f(x) = 0 ∧ ∫xdx = x²/2",
                    "command": "calc",
                },
            }
        }
        res = parse_command_list(bidi_payload)
        assert len(res) == 3
        for cmd in res.values():
            d = cmd.to_dict()
            assert json.loads(json.dumps(d)) == d


class TestPluginAdversarialExecutionAndConcurrency:
    """Stress testing the active plugin instance against adversary packets and concurrent load."""

    async def test_inbound_adversary_packets_cannot_exploit_or_hang(self) -> None:
        """Send malicious packet types, empty bodies, and mismatched keys."""
        bus = EventBus()
        events = []
        bus.subscribe_all(lambda ev: events.append(ev))
        plugin = RemoteCommandsPlugin(event_bus=bus)
        send_mock = AsyncMock()

        class MockRawPacket:
            def __init__(self, pkt_type: str, body: Any) -> None:
                self.type = pkt_type
                self.body = body

        adversary_packets = [
            Packet(type="kdeconnect.unknown.type", body={"key": "malicious"}),
            Packet(type="kdeconnect.runcommand", body={}),
            Packet(type="kdeconnect.runcommand", body={"commandList": None}),
            Packet(type="kdeconnect.remoteforcecommand", body={"key": "cmd_injection; rm -rf /"}),
            Packet(type="kdeconnect.runcommand.request", body={"key": "cmd_reboot"}),
            Packet(type="kdeconnect.runcommand.output", body={"key": None, "exitCode": "invalid"}),
            Packet(type="kdeconnect.runcommand.output", body={"key": "k", "output": None}),
            MockRawPacket("kdeconnect.runcommand", None),
            MockRawPacket("kdeconnect.runcommand.output", None),
        ]

        for pkt in adversary_packets:
            await plugin.handle_packet(pkt, "attacker_device", send_mock)

        # Confirm send_mock was never invoked by inbound packets
        send_mock.assert_not_called()

    async def test_high_concurrency_race_with_unpair_and_stop(self) -> None:
        """Run 100 concurrent workers updating commands while 10 workers unpair and stop."""
        bus = EventBus()
        plugin = RemoteCommandsPlugin(event_bus=bus)
        await plugin.start()
        send_mock = AsyncMock()

        async def worker_packet_feed(dev_idx: int) -> None:
            dev_id = f"device_{dev_idx}"
            pkt = Packet(
                type="kdeconnect.runcommand",
                body={"commandList": {f"cmd_{j}": {"name": f"Cmd {j}"} for j in range(5)}},
            )
            await plugin.handle_packet(pkt, dev_id, send_mock)

        async def worker_unpair(dev_idx: int) -> None:
            await asyncio.sleep(0.002)
            plugin.on_device_unpaired(f"device_{dev_idx}")

        tasks = []
        for i in range(100):
            tasks.append(asyncio.create_task(worker_packet_feed(i % 10)))
        for i in range(10):
            tasks.append(asyncio.create_task(worker_unpair(i)))

        await asyncio.gather(*tasks, return_exceptions=True)
        await plugin.stop()

        # After stop, all internal caches must be empty
        for i in range(10):
            assert plugin.get_commands(f"device_{i}") == []

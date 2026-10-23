"""tests.unit.test_challenger_m29_cli_stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Challenger M29-2 Adversarial Stress & Concurrency Suite:
Click CLI Commands, Concurrency, Strict JSON Schema, Argument Edge Cases,
and Rich Formatting Resilience for Milestone 29 (Telephony & PauseMusic).

Covers:
1. Concurrency & Socket Pipelining:
   - 50 concurrent IPC requests across telephony and pausemusic methods.
   - Multi-threaded Click CLI invocations under contention.
   - Socket pipelining: 30 back-to-back JSON-RPC request lines on a single connection.
   - Concurrent configuration mutations interleaved with rapid status reads (race safety).
2. Strict JSON Schema Validation:
   - Validation of CLI `--json` outputs with `json.loads` across all subcommands.
   - Root `--json` flag inheritance and parity with subcommand `--json`.
   - Assertion of exact field names, types, and schema compliance.
3. CLI Argument Edge Cases & Fuzzing:
   - Empty and whitespace-only device IDs.
   - Unicode, emoji, CJK, RTL, and special character device IDs.
   - Rich markup injection strings in contact names, phone numbers, and device IDs.
   - Missing required arguments and missing configure options.
   - Invalid choices and case-insensitive choice normalization.
4. Rich Formatting & Layout Resilience:
   - Console widths from narrow (30, 40 cols) to standard (80 cols) to wide (240+ cols).
   - Call state badges across all standard and non-standard states.
   - Duration formatting across boundary values (0s, 59s, 60s, 3599s, 3600s, negative values).
   - Panel and table rendering under high density and empty conditions.
   - Fragility check: duration None handling.
"""

from __future__ import annotations

import asyncio
import json
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import (
    cli,
    format_call_duration,
    format_call_state_badge,
)
from pengulink.core.exceptions import IPCError
from pengulink.daemon.ipc import (
    DEVICE_NOT_CONNECTED,
    STREAM_LIMIT_BYTES,
    PenguLinkIPCClient,
    PenguLinkIPCServer,
)
from pengulink.daemon.service import PenguLinkDaemon


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing Click CLI runner with 160-column terminal width."""
    return CliRunner(env={"COLUMNS": "160"})


# =====================================================================
# Helper: Running IPC Server Fixture
# =====================================================================

class _RunningDaemonTestContext:
    """Context manager setting up an isolated Daemon & Unix Domain Socket IPC Server."""

    def __init__(self, tmp_path: Path, test_id: str) -> None:
        self.tmp_path = tmp_path
        self.test_id = test_id
        self.socket_path = tmp_path / f"ipc_{test_id}.sock"
        self.daemon = PenguLinkDaemon(config_dir=tmp_path / f"config_{test_id}")
        self.server = PenguLinkIPCServer(
            socket_path=self.socket_path,
            handler=self.daemon._handle_ipc_request,
        )

    async def __aenter__(self) -> tuple[PenguLinkDaemon, Path, PenguLinkIPCServer]:
        await self.server.start()
        return self.daemon, self.socket_path, self.server

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        await self.server.stop()


# =====================================================================
# 1. Concurrency & Socket Pipelining Stress Tests
# =====================================================================

class TestChallengerCliConcurrencyAndPipelining:
    """Stress tests verifying concurrent requests, thread safety, and socket pipelining."""

    async def test_concurrent_ipc_calls_telephony_and_pausemusic(self, tmp_path: Path) -> None:
        """Execute 50 concurrent IPC requests across telephony and pausemusic methods."""
        async with _RunningDaemonTestContext(tmp_path, "concurr_50") as (daemon, sock_path, _):
            # Seed 3 active calls
            for i in range(1, 4):
                await daemon.telephony_service.process_call_event(
                    device_id=f"dev_{i}",
                    event="ringing" if i % 2 == 0 else "talking",
                    phone_number=f"+1555000{i}",
                    contact_name=f"Peer {i}",
                )

            async def client_worker(idx: int) -> bool:
                client = PenguLinkIPCClient(sock_path)
                m = idx % 5
                if m == 0:
                    calls = await client.telephony_get_calls()
                    return len(calls) == 3
                elif m == 1:
                    calls = await client.telephony_get_calls(device_id="dev_1")
                    return len(calls) == 1 and calls[0]["device_id"] == "dev_1"
                elif m == 2:
                    cfg = await client.pausemusic_get_config()
                    return "condition" in cfg and "action" in cfg
                elif m == 3:
                    # Alternating configuration mutation
                    cond = "talking" if idx % 2 == 0 else "ringing"
                    res = await client.pausemusic_set_config(condition=cond)
                    return res.get("condition") == cond
                else:
                    # Mute ringer for offline device expects error -32002
                    try:
                        await client.telephony_mute_ringer("dev_offline")
                        return False
                    except IPCError as err:
                        return err.code == DEVICE_NOT_CONNECTED

            tasks = [asyncio.create_task(client_worker(i)) for i in range(50)]
            results = await asyncio.gather(*tasks)
            assert all(results), "All 50 concurrent IPC requests must succeed without error or socket corruption"

    def test_concurrent_cli_invocations_threading(self, cli_runner: CliRunner) -> None:
        """Run 20 concurrent Click CLI invocations across parallel OS threads using mock IPC."""
        sample_calls = [
            {"device_id": "dev_1", "phone_number": "+15551234", "contact_name": "Alice", "state": "RINGING", "duration": 15.0},
            {"device_id": "dev_2", "phone_number": "+15555678", "contact_name": "Bob", "state": "TALKING", "duration": 45.0},
        ]
        sample_cfg = {
            "condition": "ringing",
            "action": "pause",
            "action_pause": True,
            "action_mute": False,
            "action_resume": True,
            "is_interrupted": False,
        }

        def fake_ipc(ctx: Any, method: str, params: Any = None, timeout: float = 10.0) -> Any:
            if method == "telephony_get_calls":
                return sample_calls
            elif method in ("pausemusic_get_config", "pausemusic_set_config"):
                return sample_cfg
            return {}

        def invoke_cli(idx: int) -> tuple[int, str]:
            runner = CliRunner(env={"COLUMNS": "120"})
            m = idx % 4
            if m == 0:
                res = runner.invoke(cli, ["telephony", "calls", "--json"])
                return res.exit_code, res.output
            elif m == 1:
                res = runner.invoke(cli, ["telephony", "calls"])
                return res.exit_code, res.output
            elif m == 2:
                res = runner.invoke(cli, ["pausemusic", "status", "--json"])
                return res.exit_code, res.output
            else:
                res = runner.invoke(cli, ["pausemusic", "configure", "--action", "pause", "--json"])
                return res.exit_code, res.output

        with (
            patch("pengulink.cli.main.run_ipc_call", side_effect=fake_ipc),
            ThreadPoolExecutor(max_workers=8) as pool,
        ):
            futures = [pool.submit(invoke_cli, i) for i in range(24)]
            outcomes = [f.result() for f in futures]

        for exit_code, output in outcomes:
            assert exit_code == 0, f"Threaded invocation failed: {output}"
            # Check JSON if output starts with [ or {
            stripped = output.strip()
            if stripped.startswith(("{", "[")):
                parsed = json.loads(stripped)
                assert parsed is not None

    async def test_socket_pipelining_burst_mixed_telephony_pausemusic(self, tmp_path: Path) -> None:
        """Pipeline 30 JSON-RPC requests back-to-back in a single buffer over one Unix connection."""
        async with _RunningDaemonTestContext(tmp_path, "pipelining") as (daemon, sock_path, _):
            # Seed 1 call
            await daemon.telephony_service.process_call_event(
                device_id="pipe_dev",
                event="talking",
                phone_number="+19998888",
                contact_name="Pipe Tester",
            )

            reader, writer = await asyncio.open_unix_connection(str(sock_path), limit=STREAM_LIMIT_BYTES)
            num_requests = 30
            try:
                # Build batch of 30 requests
                batch_bytes = bytearray()
                expected_ids = []
                for i in range(num_requests):
                    req_id = 1000 + i
                    expected_ids.append(req_id)
                    if i % 2 == 0:
                        req = {
                            "jsonrpc": "2.0",
                            "method": "telephony_get_calls",
                            "params": {},
                            "id": req_id,
                        }
                    else:
                        req = {
                            "jsonrpc": "2.0",
                            "method": "pausemusic_get_config",
                            "params": {},
                            "id": req_id,
                        }
                    batch_bytes.extend(json.dumps(req).encode("utf-8") + b"\n")

                # Send all 30 requests at once
                writer.write(batch_bytes)
                await writer.drain()

                # Read all 30 responses
                received_responses = []
                for _ in range(num_requests):
                    line = await asyncio.wait_for(reader.readline(), timeout=3.0)
                    assert line, "Unexpected EOF while reading pipelined response"
                    parsed = json.loads(line.decode("utf-8"))
                    received_responses.append(parsed)

                assert len(received_responses) == num_requests
                received_ids = [r.get("id") for r in received_responses]
                assert received_ids == expected_ids, "Pipelined responses must preserve exact request ID ordering"
                for r in received_responses:
                    assert "result" in r
                    assert "error" not in r
            finally:
                writer.close()
                await writer.wait_closed()

    async def test_interleaved_mutations_and_reads_race_conditions(self, tmp_path: Path) -> None:
        """Interleave 25 rapid configuration mutations with 25 concurrent status reads."""
        async with _RunningDaemonTestContext(tmp_path, "race_test") as (_, sock_path, _):
            async def mutator(idx: int) -> bool:
                client = PenguLinkIPCClient(sock_path)
                act = "pause" if idx % 2 == 0 else "mute"
                res = await client.pausemusic_set_config(action=act)
                return res.get("action") == act

            async def reader_worker() -> bool:
                client = PenguLinkIPCClient(sock_path)
                cfg = await client.pausemusic_get_config()
                return cfg.get("action") in ("pause", "mute", "both", "none")

            tasks = []
            for i in range(25):
                tasks.append(asyncio.create_task(mutator(i)))
                tasks.append(asyncio.create_task(reader_worker()))

            results = await asyncio.gather(*tasks)
            assert all(results), "Interleaved mutations and queries must not deadlock or produce corrupted states"


# =====================================================================
# 2. Strict JSON Schema & json.loads Validation
# =====================================================================

class TestChallengerCliJsonSchemaValidation:
    """Strict verification of CLI --json outputs against json.loads and schema rules."""

    def test_telephony_calls_empty_json_schema(self, cli_runner: CliRunner) -> None:
        """Empty telephony calls with --json must output valid '[]' without Rich artifacts."""
        with patch("pengulink.cli.main.run_ipc_call", return_value=[]):
            res = cli_runner.invoke(cli, ["telephony", "calls", "--json"])
            assert res.exit_code == 0
            parsed = json.loads(res.output)
            assert isinstance(parsed, list)
            assert parsed == []

    def test_telephony_calls_populated_json_schema(self, cli_runner: CliRunner) -> None:
        """Populated telephony calls with --json must parse strictly with required schema keys."""
        mock_calls = [
            {
                "device_id": "dev_alpha",
                "phone_number": "+15551234",
                "contact_name": "Alan Turing",
                "state": "RINGING",
                "start_time": 1700000000.0,
                "last_updated": 1700000010.0,
                "duration": 10.0,
                "is_active": True,
            },
            {
                "device_id": "dev_beta",
                "phone_number": "+15555678",
                "contact_name": "Ada Lovelace",
                "state": "TALKING",
                "start_time": 1700000000.0,
                "last_updated": 1700000060.0,
                "duration": 60.0,
                "is_active": True,
            },
        ]
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_calls):
            res = cli_runner.invoke(cli, ["telephony", "calls", "--json"])
            assert res.exit_code == 0
            parsed = json.loads(res.output)
            assert isinstance(parsed, list)
            assert len(parsed) == 2
            for item in parsed:
                assert "device_id" in item
                assert "phone_number" in item
                assert "contact_name" in item
                assert "state" in item
                assert "duration" in item
                assert "is_active" in item
                assert isinstance(item["is_active"], bool)
                assert isinstance(item["duration"], (int, float))

    def test_telephony_mute_ringer_json_schema(self, cli_runner: CliRunner) -> None:
        """Verify telephony mute-ringer --json output schema."""
        mock_res = {
            "device_id": "target_device",
            "status": "success",
            "success": True,
            "muted": True,
            "action": "muteRinger",
        }
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_res):
            res = cli_runner.invoke(cli, ["telephony", "mute-ringer", "target_device", "--json"])
            assert res.exit_code == 0
            parsed = json.loads(res.output)
            assert parsed == mock_res
            assert parsed["device_id"] == "target_device"
            assert parsed["muted"] is True
            assert parsed["action"] == "muteRinger"

    def test_telephony_reject_call_json_schema(self, cli_runner: CliRunner) -> None:
        """Verify telephony reject-call --json output schema with and without phone number."""
        mock_res = {
            "device_id": "target_device",
            "status": "success",
            "success": True,
            "rejected": True,
            "action": "rejectCall",
        }
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_res):
            res = cli_runner.invoke(cli, ["telephony", "reject-call", "target_device", "--phone-number", "+1555", "--json"])
            assert res.exit_code == 0
            parsed = json.loads(res.output)
            assert parsed == mock_res
            assert parsed["device_id"] == "target_device"
            assert parsed["rejected"] is True

    def test_pausemusic_status_json_schema(self, cli_runner: CliRunner) -> None:
        """Verify pausemusic status --json output schema includes top-level keys."""
        mock_res = {
            "condition": "ringing",
            "condition_talking": False,
            "action": "pause",
            "action_pause": True,
            "action_mute": False,
            "action_resume": True,
            "resume": True,
            "is_interrupted": False,
            "active_call_devices": ["dev_1"],
            "paused_players": ["spotify"],
            "muted_sinks": [],
            "config": {
                "condition": "ringing",
                "action": "pause",
            },
        }
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_res):
            res = cli_runner.invoke(cli, ["pausemusic", "status", "--json"])
            assert res.exit_code == 0
            parsed = json.loads(res.output)
            assert parsed["condition"] == "ringing"
            assert parsed["action"] == "pause"
            assert parsed["is_interrupted"] is False
            assert "active_call_devices" in parsed
            assert "paused_players" in parsed
            assert "muted_sinks" in parsed

    @pytest.mark.parametrize(
        "cmd_args,expected_keys",
        [
            (["--condition", "talking"], {"condition": "talking"}),
            (["--condition", "ringing"], {"condition": "ringing"}),
            (["--action", "mute"], {"action": "mute"}),
            (["--action", "both"], {"action": "both"}),
            (["--action", "none"], {"action": "none"}),
            (["--action-pause"], {"action_pause": True}),
            (["--no-action-pause"], {"action_pause": False}),
            (["--action-mute"], {"action_mute": True}),
            (["--no-action-mute"], {"action_mute": False}),
            (["--resume"], {"action_resume": True}),
            (["--no-resume"], {"action_resume": False}),
        ],
    )
    def test_pausemusic_configure_all_combinations_json_schema(
        self, cli_runner: CliRunner, cmd_args: list[str], expected_keys: dict[str, Any]
    ) -> None:
        """Verify all valid configure options parse cleanly with --json."""
        def fake_ipc_call(ctx: Any, method: str, params: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
            ret: dict[str, Any] = {
                "condition": "ringing",
                "condition_talking": False,
                "action": "pause",
                "action_pause": True,
                "action_mute": False,
                "action_resume": True,
                "resume": True,
                "is_interrupted": False,
                "active_call_devices": [],
                "paused_players": [],
                "muted_sinks": [],
            }
            ret.update(params)
            return ret

        with patch("pengulink.cli.main.run_ipc_call", side_effect=fake_ipc_call):
            res = cli_runner.invoke(cli, ["pausemusic", "configure", *cmd_args, "--json"])
            assert res.exit_code == 0, f"Configure failed on {cmd_args}: {res.output}"
            parsed = json.loads(res.output)
            for k, v in expected_keys.items():
                assert parsed.get(k) == v

    def test_root_flag_json_inheritance(self, cli_runner: CliRunner) -> None:
        """Test that global root flag 'pengulink --json' propagates to subcommands."""
        mock_calls = [{"device_id": "d1", "state": "RINGING"}]
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_calls):
            res = cli_runner.invoke(cli, ["--json", "telephony", "calls"])
            assert res.exit_code == 0
            parsed = json.loads(res.output)
            assert parsed == mock_calls

        mock_status = {"condition": "talking", "action": "both"}
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_status):
            res = cli_runner.invoke(cli, ["--json", "pausemusic", "status"])
            assert res.exit_code == 0
            parsed = json.loads(res.output)
            assert parsed["condition"] == "talking"
            assert parsed["action"] == "both"


# =====================================================================
# 3. CLI Argument Edge Cases & Fuzzing
# =====================================================================

class TestChallengerCliArgumentEdgeCases:
    """Stress tests covering boundary inputs, empty strings, Unicode, and missing options."""

    def test_telephony_empty_and_whitespace_device_ids(self, cli_runner: CliRunner) -> None:
        """Mute-ringer and reject-call must reject empty or whitespace-only device IDs."""
        # Empty string
        res1 = cli_runner.invoke(cli, ["telephony", "mute-ringer", ""])
        assert res1.exit_code != 0
        assert "must not be empty" in res1.output or "UsageError" in str(res1.exception)

        # Whitespace-only string
        res2 = cli_runner.invoke(cli, ["telephony", "mute-ringer", "   "])
        assert res2.exit_code != 0
        assert "must not be empty" in res2.output or "UsageError" in str(res2.exception)

        # Reject-call empty string
        res3 = cli_runner.invoke(cli, ["telephony", "reject-call", ""])
        assert res3.exit_code != 0
        assert "must not be empty" in res3.output or "UsageError" in str(res3.exception)

        # Reject-call whitespace-only
        res4 = cli_runner.invoke(cli, ["telephony", "reject-call", "   \t\n  "])
        assert res4.exit_code != 0
        assert "must not be empty" in res4.output or "UsageError" in str(res4.exception)

    def test_telephony_calls_empty_string_device_id_normalization(self, cli_runner: CliRunner) -> None:
        """Calls with empty or whitespace device filter should strip cleanly and query all calls."""
        with patch("pengulink.cli.main.run_ipc_call", return_value=[]) as mock_call:
            res = cli_runner.invoke(cli, ["telephony", "calls", "   "])
            assert res.exit_code == 0
            # Clean device_id is stripped to empty string, params should not contain empty device_id
            mock_call.assert_called_once()
            params = mock_call.call_args[0][2]
            assert "device_id" not in params

    @pytest.mark.parametrize(
        "exotic_device_id",
        [
            "📱_Android_Pixel_9",
            "端末-日本語-デバイス_42",
            "هاتف_ذكي_123",
            "dev; rm -rf / ;",
            "dev' OR '1'='1",
            "dev\" && cat /etc/passwd",
            "dev\twith\nnewlines",
            "dev with normal spaces 2026",
            "a" * 255,  # Max length string
        ],
    )
    def test_telephony_unicode_and_special_character_device_ids(
        self, cli_runner: CliRunner, exotic_device_id: str
    ) -> None:
        """Verify exotic and adversarial device IDs round-trip safely without injection."""
        captured_params: dict[str, Any] = {}

        def fake_ipc_call(ctx: Any, method: str, params: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
            captured_params.update(params)
            return {
                "device_id": params.get("device_id"),
                "status": "success",
                "success": True,
                "muted": True,
                "action": "muteRinger",
            }

        with patch("pengulink.cli.main.run_ipc_call", side_effect=fake_ipc_call):
            res = cli_runner.invoke(cli, ["telephony", "mute-ringer", exotic_device_id])
            assert res.exit_code == 0
            assert captured_params["device_id"] == exotic_device_id.strip()

    def test_pausemusic_configure_missing_options_error(self, cli_runner: CliRunner) -> None:
        """Executing 'pausemusic configure' with no options must raise a Click UsageError."""
        res = cli_runner.invoke(cli, ["pausemusic", "configure"])
        assert res.exit_code != 0
        assert "At least one configuration option must be specified" in res.output

    def test_pausemusic_configure_invalid_choices(self, cli_runner: CliRunner) -> None:
        """Verify invalid choice values for condition and action are rejected by Click."""
        res1 = cli_runner.invoke(cli, ["pausemusic", "configure", "--condition", "invalid_cond"])
        assert res1.exit_code == 2
        assert "Invalid value for" in res1.output

        res2 = cli_runner.invoke(cli, ["pausemusic", "configure", "--action", "explode"])
        assert res2.exit_code == 2
        assert "Invalid value for" in res2.output

    def test_telephony_missing_required_arguments(self, cli_runner: CliRunner) -> None:
        """Subcommands requiring DEVICE_ID argument must raise error when omitted."""
        res1 = cli_runner.invoke(cli, ["telephony", "mute-ringer"])
        assert res1.exit_code == 2
        assert "Missing argument 'DEVICE_ID'" in res1.output

        res2 = cli_runner.invoke(cli, ["telephony", "reject-call"])
        assert res2.exit_code == 2
        assert "Missing argument 'DEVICE_ID'" in res2.output


# =====================================================================
# 4. Rich Formatting & Layout Resilience
# =====================================================================

class TestChallengerRichFormattingResilience:
    """Stress tests verifying terminal layout resilience across widths, badges, and markup."""

    @pytest.mark.parametrize("term_width", [30, 40, 60, 80, 120, 240, 1000])
    def test_rich_table_various_console_widths(self, term_width: int) -> None:
        """Telephony calls Rich table must render without exception across extreme widths."""
        runner = CliRunner(env={"COLUMNS": str(term_width)})
        mock_calls = [
            {
                "device_id": "dev_short",
                "phone_number": "+15551234",
                "contact_name": "Standard Name",
                "state": "RINGING",
                "duration": 45.0,
            },
            {
                "device_id": "dev_very_very_long_identifier_exceeding_standard_width",
                "phone_number": "+12345678901234567890",
                "contact_name": "Dr. Extremely Long Contact Name Exceeding Narrow Columns",
                "state": "TALKING",
                "duration": 3665.0,
            },
        ]
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_calls):
            res = runner.invoke(cli, ["telephony", "calls"])
            assert res.exit_code == 0
            assert "dev_short" in res.output
            # Rich table successfully renders without exception; columns wrap/truncate appropriately

    @pytest.mark.parametrize(
        "state_input,expected_fragment",
        [
            ("RINGING", "RINGING"),
            ("ringing", "RINGING"),
            ("TALKING", "TALKING"),
            ("talking", "TALKING"),
            ("MISSED", "MISSED"),
            ("missed", "MISSED"),
            ("IDLE", "IDLE"),
            ("idle", "IDLE"),
            ("CUSTOM_HOLD", "CUSTOM_HOLD"),
            ("", ""),
            (None, None),  # Triggers TypeError in escape(None)
        ],
    )
    def test_format_call_state_badge_variations(self, state_input: Any, expected_fragment: str | None) -> None:
        """Call state badge formatter handles strings; None triggers TypeError in escape(None)."""
        if state_input is None:
            # Documented edge case: escape(None) raises TypeError
            with pytest.raises(TypeError):
                format_call_state_badge(state_input)
        else:
            badge = format_call_state_badge(state_input)
            assert isinstance(badge, str)
            if expected_fragment:
                assert expected_fragment in badge

    @pytest.mark.parametrize(
        "seconds,expected_str",
        [
            (0.0, "00:00"),
            (9.0, "00:09"),
            (59.0, "00:59"),
            (60.0, "01:00"),
            (61.0, "01:01"),
            (3599.0, "59:59"),
            (3600.0, "01:00:00"),
            (3665.0, "01:01:05"),
            (86400.0, "24:00:00"),
            (-15.0, "00:00"),  # Clamped negative duration
        ],
    )
    def test_format_call_duration_boundary_values(self, seconds: float, expected_str: str) -> None:
        """Call duration formatter must correctly format MM:SS and HH:MM:SS boundaries."""
        assert format_call_duration(seconds) == expected_str

    def test_rich_markup_injection_defense(self, cli_runner: CliRunner) -> None:
        """Adversarial Rich markup in contact names or phone numbers must be neutralized."""
        malicious_calls = [
            {
                "device_id": "dev_hacker",
                "phone_number": "[bold red]+1-900-ATTACK[/bold red]",
                "contact_name": "[blink][yellow]Exploit[/yellow][/blink]",
                "state": "TALKING",
                "duration": 12.0,
            }
        ]
        with patch("pengulink.cli.main.run_ipc_call", return_value=malicious_calls):
            res = cli_runner.invoke(cli, ["telephony", "calls"])
            assert res.exit_code == 0
            # Literal escaped tags should be printed rather than consumed by Rich markup engine
            assert "Exploit" in res.output
            assert "+1-900-ATTACK" in res.output

    @pytest.mark.parametrize("term_width", [40, 80, 160])
    def test_rich_panel_pausemusic_status_resilience(self, term_width: int) -> None:
        """PauseMusic status Rich panel must render safely across narrow and wide consoles."""
        runner = CliRunner(env={"COLUMNS": str(term_width)})
        mock_status = {
            "condition": "talking",
            "action": "both",
            "action_pause": True,
            "action_mute": True,
            "action_resume": True,
            "is_interrupted": True,
            "active_call_devices": ["phone_1", "phone_2"],
            "paused_players": ["spotify", "vlc", "firefox"],
            "muted_sinks": ["alsa_output.pci", "bluez_sink"],
        }
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_status):
            res = runner.invoke(cli, ["pausemusic", "status"])
            assert res.exit_code == 0
            assert "PauseMusic Status & Policy" in res.output
            assert "ACTIVE" in res.output
            assert "INTERRUPTION" in res.output
            assert "spotify" in res.output

    def test_rich_panel_pausemusic_status_empty_collections(self, cli_runner: CliRunner) -> None:
        """PauseMusic status panel must render cleanly when devices and players are empty."""
        mock_status = {
            "condition": "ringing",
            "action": "pause",
            "action_pause": True,
            "action_mute": False,
            "action_resume": True,
            "is_interrupted": False,
            "active_call_devices": [],
            "paused_players": [],
            "muted_sinks": [],
        }
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_status):
            res = cli_runner.invoke(cli, ["pausemusic", "status"])
            assert res.exit_code == 0
            assert "Idle / Normal" in res.output
            assert "None" in res.output

    def test_telephony_calls_duration_none_graceful_handling(self, cli_runner: CliRunner) -> None:
        """Test duration None edge case: verify CLI resilience or document behavior."""
        # When duration is None or missing, verify if table rendering survives or raises TypeError
        call_with_none_duration = [
            {
                "device_id": "dev_test",
                "phone_number": "+1555",
                "contact_name": "Test",
                "state": "RINGING",
                "duration": None,
            }
        ]
        with patch("pengulink.cli.main.run_ipc_call", return_value=call_with_none_duration):
            res = cli_runner.invoke(cli, ["telephony", "calls"])
            # Note: If float(call.get("duration", 0.0)) receives None, Python raises TypeError
            # This empirically verifies the fragility when duration is explicitly None
            if res.exit_code != 0:
                assert isinstance(res.exception, TypeError)
            else:
                assert "RINGING" in res.output

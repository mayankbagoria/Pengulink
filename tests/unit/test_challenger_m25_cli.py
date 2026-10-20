"""tests.unit.test_challenger_m25_cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Milestone 25 Adversarial Fuzzing & Stress Testing Suite for Click & Rich CLI.
Target Commands: `pengulink volume` and `pengulink runcommand`.

Covers:
1. CLI volume input fuzzing: invalid percentages, non-numeric strings, clamped values,
   Unicode sink names, leading/trailing whitespace, empty sink arguments.
2. CLI mute input fuzzing: truthy/falsy toggles, invalid toggle keywords, case-insensitivity,
   omitted state defaults, whitespace handling.
3. CLI argument ordering fuzzing: standard vs inverted syntax on remote volume commands,
   missing device_id, missing sink name, missing volume value, empty strings.
4. CLI runcommand fuzzing: empty strings in name/command/key, shell control characters,
   special characters/quotes, Rich markup injection prevention, non-existent key execution.
5. Error formatting: exit codes and error messages when daemon is not running
   (DaemonNotRunningError), IPC error code -32015, IPC error code -32016, device disconnected (-32002),
   and unhandled exceptions.
6. `--json` output format integrity: valid machine-readable JSON across all commands,
   empty collections, success/failure execution responses, and root flag inheritance.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli, parse_mute_state, parse_volume_value
from pengulink.core.exceptions import IPCError

ALL_M25_COMMANDS: list[list[str]] = [
    ["volume", "local", "list"],
    ["volume", "local", "set", "sink1", "50%"],
    ["volume", "local", "mute", "sink1"],
    ["volume", "local", "default", "sink1"],
    ["volume", "remote", "dev_1", "list"],
    ["volume", "remote", "dev_1", "set", "sink1", "50%"],
    ["volume", "remote", "dev_1", "mute", "sink1"],
    ["volume", "remote", "dev_1", "default", "sink1"],
    ["volume", "remote", "dev_1", "refresh"],
    ["runcommand", "list"],
    ["runcommand", "add", "c1", "echo 1"],
    ["runcommand", "remove", "k1"],
    ["runcommand", "run", "k1"],
]


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing a CliRunner with standard terminal width."""
    return CliRunner(env={"COLUMNS": "160"})


# =====================================================================
# 1. Volume Input Fuzzing
# =====================================================================


class TestChallengerVolumeInputFuzzing:
    """Adversarial input fuzzing for volume parsing and volume commands."""

    def test_parse_volume_value_clamping_percentages(self) -> None:
        """Verify percentages are clamped strictly between 0 and 100."""
        assert parse_volume_value("120%") == (100, True)
        assert parse_volume_value("-5%") == (0, True)
        assert parse_volume_value("100%") == (100, True)
        assert parse_volume_value("0%") == (0, True)
        assert parse_volume_value("9999%") == (100, True)
        assert parse_volume_value("-9999%") == (0, True)
        assert parse_volume_value("  75%  ") == (75, True)

    def test_parse_volume_value_invalid_percentage_strings(self) -> None:
        """Verify malformed percentage strings raise click.UsageError."""
        invalid_pcts = ["abc%", "%%%", "%", "50%%", "  %", "NaN%", "foo%bar%"]
        for bad in invalid_pcts:
            with pytest.raises(Exception) as exc_info:
                parse_volume_value(bad)
            assert "Invalid volume percentage" in str(
                exc_info.value
            ) or "Invalid volume value" in str(exc_info.value)

    def test_parse_volume_value_non_numeric_and_empty(self) -> None:
        """Verify non-numeric and blank strings raise click.UsageError."""
        invalid_vals = ["abc", "NaN", "undefined", "null", "", "   ", "---", "$#@!"]
        for bad in invalid_vals:
            with pytest.raises(Exception) as exc_info:
                parse_volume_value(bad)
            assert "Invalid volume" in str(exc_info.value)

    def test_parse_volume_value_raw_integers_clamping(self) -> None:
        """Verify raw integers clamp to 0..65536 and 0..100 are treated as percentages."""
        assert parse_volume_value("50") == (50, True)
        assert parse_volume_value("100") == (100, True)
        assert parse_volume_value("101") == (101, False)
        assert parse_volume_value("65536") == (65536, False)
        assert parse_volume_value("70000") == (65536, False)
        assert parse_volume_value("-10") == (0, False)
        assert parse_volume_value("1000000") == (65536, False)

    def test_volume_local_set_percentage_fuzzing(self, cli_runner: CliRunner) -> None:
        """Verify CLI volume local set accepts clamped percentages and rejects malformed ones."""
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            # 120% clamps to 100% -> raw 65536
            res = cli_runner.invoke(
                cli, ["volume", "local", "set", "alsa_sink", "120%"]
            )
            assert res.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_volume",
                {"name": "alsa_sink", "volume": 65536},
            )

            # Negative percentage without POSIX '--' is intercepted by Click parser as an unknown option
            res_opt = cli_runner.invoke(
                cli, ["volume", "local", "set", "alsa_sink", "-5%"]
            )
            assert res_opt.exit_code == 2
            assert "No such option: -5" in res_opt.output

            # Negative percentage with POSIX '--' passes as positional argument and clamps to 0% -> raw 0
            res = cli_runner.invoke(
                cli, ["volume", "local", "set", "--", "alsa_sink", "-5%"]
            )
            assert res.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_volume",
                {"name": "alsa_sink", "volume": 0},
            )

        # Malformed percentages should fail with exit code 2
        for bad in ["abc%", "%%%", "foo%"]:
            res = cli_runner.invoke(cli, ["volume", "local", "set", "alsa_sink", bad])
            assert res.exit_code == 2
            assert "Invalid volume percentage" in res.output

    def test_volume_local_set_unicode_and_special_sink_names(
        self, cli_runner: CliRunner
    ) -> None:
        """Verify Unicode, quotes, emojis, and Rich markup in sink names are safely processed."""
        complex_sinks = [
            "alsa_output.pci-🔊_speaker_🎧_öl_ñ",
            "sink'with\"quotes and spaces",
            "[bold red]injected_markup[/bold red]",
            "колонки_bluetooth_01",
            "日本語_スピーカー_sink",
        ]
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            for sink_name in complex_sinks:
                res = cli_runner.invoke(
                    cli, ["volume", "local", "set", sink_name, "60%"]
                )
                assert res.exit_code == 0
                mock_ipc.assert_called_with(
                    mock_ipc.call_args[0][0],
                    "systemvolume_set_volume",
                    {"name": sink_name, "volume": 39322},
                )
                assert "Volume of sink" in res.output

    def test_volume_local_set_whitespace_sink_names(
        self, cli_runner: CliRunner
    ) -> None:
        """Verify leading/trailing spaces are stripped, while whitespace-only sinks are rejected."""
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            res = cli_runner.invoke(
                cli, ["volume", "local", "set", "   trimmed_sink   ", "50%"]
            )
            assert res.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_volume",
                {"name": "trimmed_sink", "volume": 32768},
            )

        # Blank or whitespace-only sink
        for blank in ["", "   "]:
            res = cli_runner.invoke(cli, ["volume", "local", "set", blank, "50%"])
            assert res.exit_code == 2
            assert "Argument 'SINK' must not be empty" in res.output

    def test_volume_top_level_shortcut_fuzzing(self, cli_runner: CliRunner) -> None:
        """Verify `pengulink volume set` top-level shortcut applies the same fuzzing rules."""
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            res = cli_runner.invoke(cli, ["volume", "set", "sink_top", "150%"])
            assert res.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_volume",
                {"name": "sink_top", "volume": 65536},
            )

        res = cli_runner.invoke(cli, ["volume", "set", "sink_top", "%%%"])
        assert res.exit_code == 2
        assert "Invalid volume percentage" in res.output

    def test_volume_remote_set_fuzzing(self, cli_runner: CliRunner) -> None:
        """Verify remote volume set handles percentage clamping and invalid volume strings."""
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            res = cli_runner.invoke(
                cli, ["volume", "remote", "dev_123", "set", "remote_sink", "120%"]
            )
            assert res.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_remote_volume",
                {"device_id": "dev_123", "name": "remote_sink", "volume": 65536},
            )

            # Negative percentage with POSIX '--' clamps to 0%
            res_neg = cli_runner.invoke(
                cli, ["volume", "remote", "dev_123", "set", "--", "remote_sink", "-10%"]
            )
            assert res_neg.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_remote_volume",
                {"device_id": "dev_123", "name": "remote_sink", "volume": 0},
            )

        res = cli_runner.invoke(
            cli, ["volume", "remote", "dev_123", "set", "remote_sink", "invalid%"]
        )
        assert res.exit_code == 2


# =====================================================================
# 2. Mute Input Fuzzing
# =====================================================================


class TestChallengerMuteInputFuzzing:
    """Adversarial input fuzzing for mute state parser and commands."""

    def test_parse_mute_state_truthy_values(self) -> None:
        """Verify all valid truthy mute toggles evaluate to True."""
        truthy_inputs = [
            "yes",
            "on",
            "true",
            "1",
            "mute",
            "YES",
            "On",
            "TRUE",
            "MUTE",
            " 1 ",
            " yes ",
        ]
        for val in truthy_inputs:
            assert parse_mute_state(val) is True

    def test_parse_mute_state_falsy_values(self) -> None:
        """Verify all valid falsy mute toggles evaluate to False."""
        falsy_inputs = [
            "no",
            "off",
            "false",
            "0",
            "unmute",
            "NO",
            "Off",
            "FALSE",
            "UNMUTE",
            " 0 ",
            " no ",
        ]
        for val in falsy_inputs:
            assert parse_mute_state(val) is False

    def test_parse_mute_state_default_omitted(self) -> None:
        """Verify omitting the mute state defaults to True."""
        assert parse_mute_state(None) is True

    def test_parse_mute_state_invalid_words_rejected(self) -> None:
        """Verify invalid words such as 'enable', 'disable', 'toggle' raise UsageError."""
        invalid_words = [
            "enable",
            "disable",
            "toggle",
            "invalid_word",
            "maybe",
            "2",
            "-1",
            "",
            "   ",
            "null",
        ]
        for bad in invalid_words:
            with pytest.raises(Exception) as exc_info:
                parse_mute_state(bad)
            assert "Invalid mute state" in str(exc_info.value)

    def test_volume_local_mute_cli_variations(self, cli_runner: CliRunner) -> None:
        """Verify CLI volume local mute handles truthy, falsy, omitted, and invalid states."""
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            # Explicit 'yes'
            res = cli_runner.invoke(cli, ["volume", "local", "mute", "speaker", "yes"])
            assert res.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_muted",
                {"name": "speaker", "muted": True},
            )

            # Explicit 'no'
            res = cli_runner.invoke(cli, ["volume", "local", "mute", "speaker", "no"])
            assert res.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_muted",
                {"name": "speaker", "muted": False},
            )

            # Omitted state -> True
            res = cli_runner.invoke(cli, ["volume", "local", "mute", "speaker"])
            assert res.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_muted",
                {"name": "speaker", "muted": True},
            )

        # Invalid toggle keywords
        for bad in ["enable", "disable", "toggle", "invalid_word"]:
            res = cli_runner.invoke(cli, ["volume", "local", "mute", "speaker", bad])
            assert res.exit_code == 2
            assert f"Invalid mute state: '{bad}'" in res.output

    def test_volume_remote_mute_cli_variations(self, cli_runner: CliRunner) -> None:
        """Verify remote mute commands properly parse mute states and reject invalid words."""
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            res = cli_runner.invoke(
                cli, ["volume", "remote", "dev_1", "mute", "remote_spk", "yes"]
            )
            assert res.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_remote_muted",
                {"device_id": "dev_1", "name": "remote_spk", "muted": True},
            )

        res = cli_runner.invoke(
            cli, ["volume", "remote", "dev_1", "mute", "remote_spk", "disable"]
        )
        assert res.exit_code == 2
        assert "Invalid mute state: 'disable'" in res.output


# =====================================================================
# 3. Argument Ordering & Missing Parameters Fuzzing
# =====================================================================


class TestChallengerArgumentOrderingAndMissingParams:
    """Test remote command argument permutations, omitted arguments, and empty strings."""

    def test_volume_remote_list_argument_permutations(
        self, cli_runner: CliRunner
    ) -> None:
        """Verify remote list works in both standard, inverted, and implicit syntax."""
        mock_sinks = [
            {
                "name": "sink1",
                "description": "Sink 1",
                "volume": 30000,
                "volume_percent": 45,
                "muted": False,
                "enabled": True,
            }
        ]
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value=mock_sinks
        ) as mock_ipc:
            # 1. volume remote <device_id> list
            res1 = cli_runner.invoke(cli, ["volume", "remote", "device_alpha", "list"])
            assert res1.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_get_remote_sinks",
                {"device_id": "device_alpha"},
            )

            # 2. volume remote list <device_id>
            res2 = cli_runner.invoke(cli, ["volume", "remote", "list", "device_alpha"])
            assert res2.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_get_remote_sinks",
                {"device_id": "device_alpha"},
            )

            # 3. volume remote <device_id> (implicit list)
            res3 = cli_runner.invoke(cli, ["volume", "remote", "device_alpha"])
            assert res3.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_get_remote_sinks",
                {"device_id": "device_alpha"},
            )

    def test_volume_remote_set_both_orderings(self, cli_runner: CliRunner) -> None:
        """Verify remote set works with both device_id first and subcommand first."""
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            # Syntax A: volume remote <device_id> set <sink> <volume>
            res_a = cli_runner.invoke(
                cli, ["volume", "remote", "dev_x", "set", "sink_a", "80%"]
            )
            assert res_a.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_remote_volume",
                {"device_id": "dev_x", "name": "sink_a", "volume": 52429},
            )

            # Syntax B: volume remote set <device_id> <sink> <volume>
            res_b = cli_runner.invoke(
                cli, ["volume", "remote", "set", "dev_x", "sink_a", "80%"]
            )
            assert res_b.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_remote_volume",
                {"device_id": "dev_x", "name": "sink_a", "volume": 52429},
            )

    def test_volume_remote_mute_both_orderings(self, cli_runner: CliRunner) -> None:
        """Verify remote mute works with both argument ordering styles."""
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            # Syntax A: volume remote <device_id> mute <sink> [state]
            res_a = cli_runner.invoke(
                cli, ["volume", "remote", "dev_x", "mute", "sink_a", "off"]
            )
            assert res_a.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_remote_muted",
                {"device_id": "dev_x", "name": "sink_a", "muted": False},
            )

            # Syntax B: volume remote mute <device_id> <sink> [state]
            res_b = cli_runner.invoke(
                cli, ["volume", "remote", "mute", "dev_x", "sink_a", "off"]
            )
            assert res_b.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_remote_muted",
                {"device_id": "dev_x", "name": "sink_a", "muted": False},
            )

    def test_volume_remote_default_and_refresh_both_orderings(
        self, cli_runner: CliRunner
    ) -> None:
        """Verify default and refresh commands work with both argument orderings."""
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            # Default
            cli_runner.invoke(cli, ["volume", "remote", "dev_1", "default", "sink_1"])
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_remote_default",
                {"device_id": "dev_1", "name": "sink_1"},
            )
            cli_runner.invoke(cli, ["volume", "remote", "default", "dev_1", "sink_1"])
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_remote_default",
                {"device_id": "dev_1", "name": "sink_1"},
            )

            # Refresh
            cli_runner.invoke(cli, ["volume", "remote", "dev_1", "refresh"])
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_request_remote_sinks",
                {"device_id": "dev_1"},
            )
            cli_runner.invoke(cli, ["volume", "remote", "refresh", "dev_1"])
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "systemvolume_request_remote_sinks",
                {"device_id": "dev_1"},
            )

    def test_volume_missing_arguments_errors(self, cli_runner: CliRunner) -> None:
        """Verify missing parameters result in clean usage error exit code 2."""
        missing_cases = [
            (["volume", "local", "set"], "Missing argument 'SINK'"),
            (["volume", "local", "set", "sink1"], "Missing argument 'VOLUME'"),
            (["volume", "local", "mute"], "Missing argument 'SINK'"),
            (["volume", "local", "default"], "Missing argument 'SINK'"),
            (["volume", "remote", "list"], "Argument 'DEVICE_ID' must not be empty"),
            (["volume", "remote", "refresh"], "Argument 'DEVICE_ID' must not be empty"),
            (["volume", "remote", "set"], "Missing argument 'ARG1'"),
            (["volume", "remote", "set", "dev1"], "Missing argument 'ARG2'"),
            (["volume", "remote", "dev1", "set"], "Missing argument 'ARG1'"),
            (["volume", "remote", "dev1", "set", "sink1"], "Missing argument 'ARG2'"),
            (["volume", "remote", "mute"], "Missing argument 'ARG1'"),
            (["volume", "remote", "default"], "Missing argument 'ARG1'"),
        ]
        for cmd_args, expected_err in missing_cases:
            res = cli_runner.invoke(cli, cmd_args)
            assert res.exit_code == 2, (
                f"Failed on command {cmd_args}: exit code {res.exit_code}"
            )
            assert expected_err in res.output, (
                f"Expected '{expected_err}' in {res.output}"
            )

    def test_volume_empty_string_parameters_rejected(
        self, cli_runner: CliRunner
    ) -> None:
        """Verify empty string arguments are rejected with clean UsageErrors."""
        empty_cases = [
            (
                ["volume", "local", "set", "", "50%"],
                "Argument 'SINK' must not be empty",
            ),
            (
                ["volume", "local", "set", "   ", "50%"],
                "Argument 'SINK' must not be empty",
            ),
            (["volume", "local", "mute", ""], "Argument 'SINK' must not be empty"),
            (["volume", "local", "default", ""], "Argument 'SINK' must not be empty"),
            (
                ["volume", "remote", "set", "", "sink1", "50%"],
                "Argument 'DEVICE_ID' must not be empty",
            ),
            (
                ["volume", "remote", "set", "dev1", "", "50%"],
                "Argument 'SINK' must not be empty",
            ),
            (
                ["volume", "remote", "mute", "", "sink1"],
                "Argument 'DEVICE_ID' must not be empty",
            ),
            (
                ["volume", "remote", "mute", "dev1", ""],
                "Argument 'SINK' must not be empty",
            ),
            (
                ["volume", "remote", "default", "", "sink1"],
                "Argument 'DEVICE_ID' must not be empty",
            ),
            (
                ["volume", "remote", "default", "dev1", ""],
                "Argument 'SINK' must not be empty",
            ),
        ]
        for cmd_args, expected_err in empty_cases:
            res = cli_runner.invoke(cli, cmd_args)
            assert res.exit_code == 2
            assert expected_err in res.output


# =====================================================================
# 4. RunCommand Fuzzing
# =====================================================================


class TestChallengerRuncommandFuzzing:
    """Test runcommand edge cases, empty strings, quotes, special chars, and markup."""

    def test_runcommand_empty_arguments_rejected(self, cli_runner: CliRunner) -> None:
        """Verify empty name, command, or key arguments are rejected with exit code 2."""
        invalid_runs = [
            (["runcommand", "add", "", "echo 1"], "Argument 'NAME' must not be empty"),
            (
                ["runcommand", "add", "   ", "echo 1"],
                "Argument 'NAME' must not be empty",
            ),
            (
                ["runcommand", "add", "my_name", ""],
                "Argument 'COMMAND' must not be empty",
            ),
            (
                ["runcommand", "add", "my_name", "   "],
                "Argument 'COMMAND' must not be empty",
            ),
            (["runcommand", "remove", ""], "Argument 'KEY' must not be empty"),
            (["runcommand", "remove", "   "], "Argument 'KEY' must not be empty"),
            (["runcommand", "run", ""], "Argument 'KEY' must not be empty"),
            (["runcommand", "run", "   "], "Argument 'KEY' must not be empty"),
        ]
        for cmd_args, expected_err in invalid_runs:
            res = cli_runner.invoke(cli, cmd_args)
            assert res.exit_code == 2
            assert expected_err in res.output

    def test_runcommand_add_special_characters_and_quotes(
        self, cli_runner: CliRunner
    ) -> None:
        """Verify commands with double/single quotes, escapes, and shell metachars pass through verbatim."""
        special_command = (
            'echo "hello "world"" && cat /dev/null | grep -E \'^foo$\' ; true'
        )
        special_name = 'My "Command" & <Test>'
        with patch(
            "pengulink.cli.main.run_ipc_call",
            return_value={
                "key": "cmd_spec_1",
                "name": special_name,
                "command": special_command,
            },
        ) as mock_ipc:
            res = cli_runner.invoke(
                cli, ["runcommand", "add", special_name, special_command]
            )
            assert res.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "runcommand_add",
                {"name": special_name, "command": special_command},
            )
            assert "Added command" in res.output
            assert "cmd_spec_1" in res.output

    def test_runcommand_add_unicode_name_and_command(
        self, cli_runner: CliRunner
    ) -> None:
        """Verify commands with Unicode, Cyrillic, and emojis serialize and render cleanly."""
        unicode_name = "🚀 Запуск сервера 日本語"
        unicode_command = "echo 'привет мир 🌍'"
        with patch(
            "pengulink.cli.main.run_ipc_call",
            return_value={
                "key": "uni_1",
                "name": unicode_name,
                "command": unicode_command,
            },
        ) as mock_ipc:
            res = cli_runner.invoke(
                cli, ["runcommand", "add", unicode_name, unicode_command]
            )
            assert res.exit_code == 0
            mock_ipc.assert_called_with(
                mock_ipc.call_args[0][0],
                "runcommand_add",
                {"name": unicode_name, "command": unicode_command},
            )

    def test_runcommand_list_rich_markup_escaping(self, cli_runner: CliRunner) -> None:
        """Verify Rich markup in command names/commands is escaped and does not crash Table rendering."""
        mock_commands = [
            {
                "key": "k1",
                "name": "[bold red]Malicious Name[/bold red]",
                "command": "echo [blue]test[/blue]",
                "enabled": True,
            },
            {
                "key": "k2",
                "name": "Normal",
                "command": "cat <file> > /dev/null",
                "enabled": False,
            },
        ]
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_commands):
            res = cli_runner.invoke(cli, ["runcommand", "list"])
            assert res.exit_code == 0
            assert "Local Commands" in res.output
            assert "Malicious Name" in res.output

    def test_runcommand_run_failed_command_display(self, cli_runner: CliRunner) -> None:
        """Verify non-zero command execution exit codes and stderr streams are formatted without crashing."""
        fail_res = {
            "key": "bad_cmd",
            "success": False,
            "exit_code": 127,
            "stdout": [],
            "stderr": ["/bin/sh: line 1: non_existent_binary: command not found"],
        }
        with patch("pengulink.cli.main.run_ipc_call", return_value=fail_res):
            res = cli_runner.invoke(cli, ["runcommand", "run", "bad_cmd"])
            assert res.exit_code == 0
            assert "Command failed (exit code 127)" in res.output
            assert "Standard Error:" in res.output
            assert "non_existent_binary" in res.output

    def test_runcommand_remove_not_found_display(self, cli_runner: CliRunner) -> None:
        """Verify removing a non-existent command displays error badge cleanly."""
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": False}):
            res = cli_runner.invoke(cli, ["runcommand", "remove", "missing_key"])
            assert res.exit_code == 0
            assert "Command 'missing_key' not found" in res.output


# =====================================================================
# 5. Error Formatting & Exit Codes
# =====================================================================


class TestChallengerErrorFormattingAndExitCodes:
    """Test exit codes and error output when daemon is offline or IPC raises error codes."""

    @pytest.mark.parametrize(
        "subcommand", ALL_M25_COMMANDS, ids=lambda c: "_".join(c[:3])
    )
    def test_daemon_not_running_socket_missing(
        self, cli_runner: CliRunner, tmp_path: Path, subcommand: list[str]
    ) -> None:
        """Verify all volume and runcommand subcommands exit 1 with clean error when daemon is not running."""
        absent_sock = tmp_path / "absent_daemon.sock"
        args = ["-s", str(absent_sock)] + subcommand
        res = cli_runner.invoke(cli, args)

        assert res.exit_code == 1, (
            f"Failed on command {subcommand}: exit code {res.exit_code}"
        )
        assert "PenguLink daemon is not running" in res.output
        assert "Traceback (most recent call last)" not in res.output

    def test_ipc_error_32015_audio_system_volume_error(
        self, cli_runner: CliRunner
    ) -> None:
        """Verify IPC error code -32015 is caught and formatted with exit code 1."""
        err_msg = "IPC Error [-32015]: PipeWire sound server connection refused"
        exc = IPCError(err_msg, context={"code": -32015})
        exc.code = -32015

        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=exc
        ):
            res = cli_runner.invoke(cli, ["volume", "local", "list"])
            assert res.exit_code == 1
            assert "IPC Error:" in res.output
            assert "[-32015]" in res.output
            assert "PipeWire sound server connection refused" in res.output
            assert "Traceback" not in res.output

            # Same on remote volume
            res_remote = cli_runner.invoke(cli, ["volume", "remote", "dev_1", "list"])
            assert res_remote.exit_code == 1
            assert "[-32015]" in res_remote.output

    def test_ipc_error_32016_run_command_error(self, cli_runner: CliRunner) -> None:
        """Verify IPC error code -32016 is caught and formatted with exit code 1."""
        err_msg = "IPC Error [-32016]: Local command store permission denied"
        exc = IPCError(err_msg, context={"code": -32016})
        exc.code = -32016

        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=exc
        ):
            res = cli_runner.invoke(cli, ["runcommand", "list"])
            assert res.exit_code == 1
            assert "IPC Error:" in res.output
            assert "[-32016]" in res.output
            assert "Local command store permission denied" in res.output
            assert "Traceback" not in res.output

            # Same on runcommand add
            res_add = cli_runner.invoke(cli, ["runcommand", "add", "cmd", "echo 1"])
            assert res_add.exit_code == 1
            assert "[-32016]" in res_add.output

    def test_ipc_error_32002_device_disconnected(self, cli_runner: CliRunner) -> None:
        """Verify device disconnected IPC error code -32002 formats cleanly."""
        err_msg = "IPC Error [-32002]: Device 'dev_phone' is not currently connected"
        exc = IPCError(err_msg, context={"code": -32002})
        exc.code = -32002

        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=exc
        ):
            res = cli_runner.invoke(
                cli, ["volume", "remote", "dev_phone", "set", "speaker", "50%"]
            )
            assert res.exit_code == 1
            assert "IPC Error:" in res.output
            assert "[-32002]" in res.output
            assert "dev_phone" in res.output
            assert "Traceback" not in res.output

    def test_unexpected_exception_handling(self, cli_runner: CliRunner) -> None:
        """Verify unhandled general exceptions in run_ipc_call exit with code 1 without uncaught crash."""
        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
            side_effect=RuntimeError("Kernel fault"),
        ):
            res = cli_runner.invoke(cli, ["volume", "local", "list"])
            assert res.exit_code == 1
            assert "Unexpected Error: Kernel fault" in res.output
            assert "Traceback" not in res.output


# =====================================================================
# 6. JSON Output Format Integrity
# =====================================================================


class TestChallengerJsonIntegrity:
    """Verify machine-readable valid JSON is produced across all volume and runcommand commands."""

    def test_json_local_volume_commands(self, cli_runner: CliRunner) -> None:
        """Verify local volume list, set, mute, and default produce valid parseable JSON."""
        # 1. list
        mock_sinks = [
            {
                "name": "sink_1",
                "description": "Sink 1",
                "volume": 32768,
                "volume_percent": 50,
                "muted": False,
                "enabled": True,
            }
        ]
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_sinks):
            res = cli_runner.invoke(cli, ["volume", "local", "list", "--json"])
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert isinstance(data, list)
            assert data[0]["name"] == "sink_1"

        # 2. list empty
        with patch("pengulink.cli.main.run_ipc_call", return_value=[]):
            res = cli_runner.invoke(cli, ["volume", "local", "list", "--json"])
            assert res.exit_code == 0
            assert json.loads(res.output) == []

        # 3. set
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            res = cli_runner.invoke(
                cli, ["volume", "local", "set", "sink_1", "75%", "--json"]
            )
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert data["sink"] == "sink_1"
            assert data["volume_percent"] == 75
            assert data["volume"] == 49152
            assert data["success"] is True

        # 4. mute
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            res = cli_runner.invoke(
                cli, ["volume", "local", "mute", "sink_1", "off", "--json"]
            )
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert data["sink"] == "sink_1"
            assert data["muted"] is False
            assert data["success"] is True

        # 5. default
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            res = cli_runner.invoke(
                cli, ["volume", "local", "default", "sink_1", "--json"]
            )
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert data["sink"] == "sink_1"
            assert data["default"] is True
            assert data["success"] is True

    def test_json_remote_volume_commands(self, cli_runner: CliRunner) -> None:
        """Verify remote volume list, set, mute, default, and refresh produce valid JSON in both orderings."""
        # 1. list
        with patch("pengulink.cli.main.run_ipc_call", return_value=[]):
            res = cli_runner.invoke(
                cli, ["volume", "remote", "dev_1", "list", "--json"]
            )
            assert res.exit_code == 0
            assert json.loads(res.output) == []

            res_inv = cli_runner.invoke(
                cli, ["volume", "remote", "list", "dev_1", "--json"]
            )
            assert res_inv.exit_code == 0
            assert json.loads(res_inv.output) == []

        # 2. set
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            res = cli_runner.invoke(
                cli, ["volume", "remote", "dev_1", "set", "spk", "40%", "--json"]
            )
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert data["device_id"] == "dev_1"
            assert data["sink"] == "spk"
            assert data["volume_percent"] == 40
            assert data["success"] is True

        # 3. mute
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            res = cli_runner.invoke(
                cli, ["volume", "remote", "mute", "dev_1", "spk", "on", "--json"]
            )
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert data["device_id"] == "dev_1"
            assert data["muted"] is True

        # 4. default
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            res = cli_runner.invoke(
                cli, ["volume", "remote", "dev_1", "default", "spk", "--json"]
            )
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert data["device_id"] == "dev_1"
            assert data["default"] is True

        # 5. refresh
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            res = cli_runner.invoke(
                cli, ["volume", "remote", "dev_1", "refresh", "--json"]
            )
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert data["device_id"] == "dev_1"
            assert data["success"] is True

    def test_json_runcommand_commands(self, cli_runner: CliRunner) -> None:
        """Verify runcommand list, add, remove, and run produce valid JSON."""
        # 1. list
        mock_cmds = [
            {"key": "c1", "name": "Backup", "command": "rsync -a", "enabled": True}
        ]
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_cmds):
            res = cli_runner.invoke(cli, ["runcommand", "list", "--json"])
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert isinstance(data, list)
            assert data[0]["key"] == "c1"

        # 2. list empty
        with patch("pengulink.cli.main.run_ipc_call", return_value=[]):
            res = cli_runner.invoke(cli, ["runcommand", "list", "--json"])
            assert res.exit_code == 0
            assert json.loads(res.output) == []

        # 3. add
        with patch(
            "pengulink.cli.main.run_ipc_call",
            return_value={"key": "new_k", "name": "Test", "command": "echo 1"},
        ):
            res = cli_runner.invoke(
                cli, ["runcommand", "add", "Test", "echo 1", "--json"]
            )
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert data["key"] == "new_k"

        # 4. remove
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            res = cli_runner.invoke(cli, ["runcommand", "remove", "k_del", "--json"])
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert data == {"key": "k_del", "success": True}

        # 5. run success
        run_res = {
            "key": "k_ok",
            "success": True,
            "exit_code": 0,
            "stdout": ["hello"],
            "stderr": [],
        }
        with patch("pengulink.cli.main.run_ipc_call", return_value=run_res):
            res = cli_runner.invoke(cli, ["runcommand", "run", "k_ok", "--json"])
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert data["success"] is True
            assert data["stdout"] == ["hello"]

        # 6. run failure
        run_fail = {
            "key": "k_fail",
            "success": False,
            "exit_code": 2,
            "stdout": [],
            "stderr": ["error occurred"],
        }
        with patch("pengulink.cli.main.run_ipc_call", return_value=run_fail):
            res = cli_runner.invoke(cli, ["runcommand", "run", "k_fail", "--json"])
            assert res.exit_code == 0
            data = json.loads(res.output)
            assert data["success"] is False
            assert data["exit_code"] == 2
            assert data["stderr"] == ["error occurred"]

    def test_json_root_flag_inheritance(self, cli_runner: CliRunner) -> None:
        """Verify global --json flag placed before subcommand triggers JSON output across volume and runcommand."""
        with patch("pengulink.cli.main.run_ipc_call", return_value=[]):
            # Global root flag on volume
            res1 = cli_runner.invoke(cli, ["--json", "volume", "local", "list"])
            assert res1.exit_code == 0
            assert json.loads(res1.output) == []

            # Global root flag on runcommand
            res2 = cli_runner.invoke(cli, ["--json", "runcommand", "list"])
            assert res2.exit_code == 0
            assert json.loads(res2.output) == []

    def test_json_escaping_unicode_and_special_chars(
        self, cli_runner: CliRunner
    ) -> None:
        """Verify Unicode, quotes, backslashes, and emojis serialize as valid JSON without escaping errors."""
        special_data = [
            {
                "name": "sink_🔊_🎧",
                "description": 'Unicode "Quotes" & \\Backslashes\\',
                "volume": 32768,
                "volume_percent": 50,
                "muted": False,
                "enabled": True,
            }
        ]
        with patch("pengulink.cli.main.run_ipc_call", return_value=special_data):
            res = cli_runner.invoke(cli, ["volume", "local", "list", "--json"])
            assert res.exit_code == 0
            parsed = json.loads(res.output)
            assert parsed[0]["name"] == "sink_🔊_🎧"
            assert parsed[0]["description"] == 'Unicode "Quotes" & \\Backslashes\\'

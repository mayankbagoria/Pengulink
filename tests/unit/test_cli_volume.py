"""tests.unit.test_cli_volume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Milestone 25: Click/Rich CLI volume command groups
(pengulink volume local [list|set|mute|default] and
 pengulink volume remote [list|set|mute|default|refresh], plus shortcuts).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.exceptions import IPCError


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner(env={"COLUMNS": "160"})


# =====================================================================
# 1. Local Volume Commands
# =====================================================================


class TestVolumeLocalList:
    """Test pengulink volume local list and top-level volume list shortcut."""

    def test_volume_local_list_table_display(self, cli_runner: CliRunner) -> None:
        mock_sinks = [
            {
                "name": "alsa_output.pci-0000_00_1f.3.speaker",
                "description": "Internal Speakers",
                "volume": 32768,
                "volume_percent": 50,
                "muted": False,
                "enabled": True,
            },
            {
                "name": "bluez_output.00_11_22_33_44_55.a2dp",
                "description": "Bluetooth Headphones",
                "volume": 49152,
                "volume_percent": 75,
                "muted": True,
                "enabled": False,
            },
        ]
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value=mock_sinks
        ) as mock_ipc:
            result = cli_runner.invoke(cli, ["volume", "local", "list"])
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0], "systemvolume_get_sinks"
            )
            assert "Local Audio Sinks" in result.output
            assert "Internal Speakers" in result.output
            assert "Bluetooth Headphones" in result.output
            assert "50%" in result.output
            assert "75%" in result.output
            assert "Default" in result.output

    def test_volume_local_list_empty(self, cli_runner: CliRunner) -> None:
        with patch("pengulink.cli.main.run_ipc_call", return_value=[]):
            result = cli_runner.invoke(cli, ["volume", "local", "list"])
            assert result.exit_code == 0
            assert "No local audio sinks found" in result.output

    def test_volume_local_list_json(self, cli_runner: CliRunner) -> None:
        mock_sinks = [
            {
                "name": "alsa_output.speaker",
                "description": "Speakers",
                "volume": 32768,
                "volume_percent": 50,
                "muted": False,
                "enabled": True,
            }
        ]
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_sinks):
            result = cli_runner.invoke(cli, ["volume", "local", "list", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["name"] == "alsa_output.speaker"

    def test_volume_top_level_list_shortcut(self, cli_runner: CliRunner) -> None:
        mock_sinks = [
            {
                "name": "alsa_output.speaker",
                "description": "Speakers",
                "volume": 32768,
                "volume_percent": 50,
                "muted": False,
                "enabled": True,
            }
        ]
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_sinks):
            result = cli_runner.invoke(cli, ["volume", "list"])
            assert result.exit_code == 0
            assert "Local Audio Sinks" in result.output
            assert "Speakers" in result.output

    def test_volume_top_level_list_shortcut_json(self, cli_runner: CliRunner) -> None:
        mock_sinks = [{"name": "alsa_output.speaker", "volume": 32768}]
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_sinks):
            result = cli_runner.invoke(cli, ["volume", "list", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data[0]["name"] == "alsa_output.speaker"

    def test_volume_local_list_ipc_error(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", side_effect=IPCError("Daemon error")
        ):
            result = cli_runner.invoke(cli, ["volume", "local", "list"])
            assert result.exit_code == 1


class TestVolumeLocalSet:
    """Test pengulink volume local set and top-level volume set shortcut."""

    def test_volume_local_set_percentage(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            result = cli_runner.invoke(cli, ["volume", "local", "set", "sink1", "50%"])
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_volume",
                {"name": "sink1", "volume": 32768},
            )
            assert "50%" in result.output

    def test_volume_local_set_raw_integer(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            result = cli_runner.invoke(
                cli, ["volume", "local", "set", "sink1", "45000"]
            )
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_volume",
                {"name": "sink1", "volume": 45000},
            )
            assert "45000" in result.output

    def test_volume_local_set_json(self, cli_runner: CliRunner) -> None:
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            result = cli_runner.invoke(
                cli, ["volume", "local", "set", "sink1", "60%", "--json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["sink"] == "sink1"
            assert data["volume_percent"] == 60
            assert data["success"] is True

    def test_volume_top_set_shortcut(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            result = cli_runner.invoke(cli, ["volume", "set", "sink1", "80%"])
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_volume",
                {"name": "sink1", "volume": 52429},
            )

    def test_volume_local_set_invalid_volume(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(
            cli, ["volume", "local", "set", "sink1", "not_a_number"]
        )
        assert result.exit_code != 0
        assert "Invalid volume" in result.output or "Error" in result.output

    def test_volume_local_set_empty_sink(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(cli, ["volume", "local", "set", "   ", "50%"])
        assert result.exit_code != 0


class TestVolumeLocalMute:
    """Test pengulink volume local mute and top-level volume mute shortcut."""

    def test_volume_local_mute_explicit_on(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            result = cli_runner.invoke(cli, ["volume", "local", "mute", "sink1", "on"])
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_muted",
                {"name": "sink1", "muted": True},
            )
            assert "muted" in result.output

    def test_volume_local_mute_explicit_off(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            result = cli_runner.invoke(cli, ["volume", "local", "mute", "sink1", "off"])
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_muted",
                {"name": "sink1", "muted": False},
            )
            assert "unmuted" in result.output

    def test_volume_local_mute_default_toggle(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            result = cli_runner.invoke(cli, ["volume", "local", "mute", "sink1"])
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_muted",
                {"name": "sink1", "muted": True},
            )

    def test_volume_local_mute_json(self, cli_runner: CliRunner) -> None:
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            result = cli_runner.invoke(
                cli, ["volume", "local", "mute", "sink1", "false", "--json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["sink"] == "sink1"
            assert data["muted"] is False
            assert data["success"] is True

    def test_volume_top_mute_shortcut(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            result = cli_runner.invoke(cli, ["volume", "mute", "sink1", "1"])
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_muted",
                {"name": "sink1", "muted": True},
            )


class TestVolumeLocalDefault:
    """Test pengulink volume local default command."""

    def test_volume_local_default_success(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            result = cli_runner.invoke(
                cli, ["volume", "local", "default", "sink_headset"]
            )
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_default",
                {"name": "sink_headset"},
            )
            assert "Default local sink set to 'sink_headset'" in result.output

    def test_volume_local_default_json(self, cli_runner: CliRunner) -> None:
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            result = cli_runner.invoke(
                cli, ["volume", "local", "default", "sink_headset", "--json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["sink"] == "sink_headset"
            assert data["default"] is True


# =====================================================================
# 2. Remote Volume Commands
# =====================================================================


class TestVolumeRemoteList:
    """Test pengulink volume remote <device_id> list and flexible syntax."""

    def test_volume_remote_list_standard_syntax(self, cli_runner: CliRunner) -> None:
        mock_sinks = [
            {
                "name": "remote_speaker",
                "description": "Phone Speaker",
                "volume": 15,
                "volume_percent": 60,
                "muted": False,
                "enabled": True,
            }
        ]
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value=mock_sinks
        ) as mock_ipc:
            result = cli_runner.invoke(cli, ["volume", "remote", "phone_1", "list"])
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_get_remote_sinks",
                {"device_id": "phone_1"},
            )
            assert "Remote Audio Sinks — phone_1" in result.output
            assert "Phone Speaker" in result.output
            assert "60%" in result.output

    def test_volume_remote_list_inverted_syntax(self, cli_runner: CliRunner) -> None:
        mock_sinks = [
            {
                "name": "remote_speaker",
                "description": "Phone Speaker",
                "volume": 15,
                "volume_percent": 60,
                "muted": False,
                "enabled": True,
            }
        ]
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value=mock_sinks
        ) as mock_ipc:
            result = cli_runner.invoke(cli, ["volume", "remote", "list", "phone_1"])
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_get_remote_sinks",
                {"device_id": "phone_1"},
            )
            assert "Remote Audio Sinks — phone_1" in result.output

    def test_volume_remote_list_implicit_subcommand(
        self, cli_runner: CliRunner
    ) -> None:
        mock_sinks = [
            {
                "name": "remote_speaker",
                "description": "Phone Speaker",
                "volume": 15,
                "volume_percent": 60,
            }
        ]
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_sinks):
            result = cli_runner.invoke(cli, ["volume", "remote", "phone_1"])
            assert result.exit_code == 0
            assert "Remote Audio Sinks — phone_1" in result.output

    def test_volume_remote_list_empty(self, cli_runner: CliRunner) -> None:
        with patch("pengulink.cli.main.run_ipc_call", return_value=[]):
            result = cli_runner.invoke(cli, ["volume", "remote", "phone_1", "list"])
            assert result.exit_code == 0
            assert "No remote audio sinks cached for device 'phone_1'" in result.output

    def test_volume_remote_list_json(self, cli_runner: CliRunner) -> None:
        mock_sinks = [{"name": "speaker", "volume": 10}]
        with patch("pengulink.cli.main.run_ipc_call", return_value=mock_sinks):
            result = cli_runner.invoke(
                cli, ["volume", "remote", "phone_1", "list", "--json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert len(data) == 1
            assert data[0]["name"] == "speaker"


class TestVolumeRemoteSet:
    """Test pengulink volume remote set."""

    def test_volume_remote_set_standard_syntax(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            result = cli_runner.invoke(
                cli, ["volume", "remote", "phone_1", "set", "Speaker", "75%"]
            )
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_remote_volume",
                {"device_id": "phone_1", "name": "Speaker", "volume": 49152},
            )
            assert (
                "Remote sink 'Speaker' volume set to 75% on device 'phone_1'"
                in result.output
            )

    def test_volume_remote_set_inverted_syntax(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            result = cli_runner.invoke(
                cli, ["volume", "remote", "set", "phone_1", "Speaker", "40000"]
            )
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_remote_volume",
                {"device_id": "phone_1", "name": "Speaker", "volume": 40000},
            )

    def test_volume_remote_set_json(self, cli_runner: CliRunner) -> None:
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            result = cli_runner.invoke(
                cli, ["volume", "remote", "phone_1", "set", "Speaker", "50%", "--json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["device_id"] == "phone_1"
            assert data["sink"] == "Speaker"
            assert data["volume_percent"] == 50
            assert data["success"] is True


class TestVolumeRemoteMute:
    """Test pengulink volume remote mute."""

    def test_volume_remote_mute_standard_syntax(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            result = cli_runner.invoke(
                cli, ["volume", "remote", "phone_1", "mute", "Speaker", "true"]
            )
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_remote_muted",
                {"device_id": "phone_1", "name": "Speaker", "muted": True},
            )
            assert (
                "Remote sink 'Speaker' is now muted on device 'phone_1'"
                in result.output
            )

    def test_volume_remote_mute_inverted_syntax(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            result = cli_runner.invoke(
                cli, ["volume", "remote", "mute", "phone_1", "Speaker", "off"]
            )
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_remote_muted",
                {"device_id": "phone_1", "name": "Speaker", "muted": False},
            )
            assert "unmuted" in result.output

    def test_volume_remote_mute_json(self, cli_runner: CliRunner) -> None:
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            result = cli_runner.invoke(
                cli, ["volume", "remote", "phone_1", "mute", "Speaker", "1", "--json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["device_id"] == "phone_1"
            assert data["muted"] is True


class TestVolumeRemoteDefaultAndRefresh:
    """Test pengulink volume remote default and refresh commands."""

    def test_volume_remote_default_standard(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            result = cli_runner.invoke(
                cli, ["volume", "remote", "phone_1", "default", "Speaker"]
            )
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_set_remote_default",
                {"device_id": "phone_1", "name": "Speaker"},
            )
            assert (
                "Default remote sink set to 'Speaker' on device 'phone_1'"
                in result.output
            )

    def test_volume_remote_default_json(self, cli_runner: CliRunner) -> None:
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            result = cli_runner.invoke(
                cli, ["volume", "remote", "phone_1", "default", "Speaker", "--json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["device_id"] == "phone_1"
            assert data["sink"] == "Speaker"
            assert data["default"] is True

    def test_volume_remote_refresh(self, cli_runner: CliRunner) -> None:
        with patch(
            "pengulink.cli.main.run_ipc_call", return_value={"success": True}
        ) as mock_ipc:
            result = cli_runner.invoke(cli, ["volume", "remote", "phone_1", "refresh"])
            assert result.exit_code == 0
            mock_ipc.assert_called_once_with(
                mock_ipc.call_args[0][0],
                "systemvolume_request_remote_sinks",
                {"device_id": "phone_1"},
            )
            assert (
                "Requested audio sink list update from remote device 'phone_1'"
                in result.output
            )

    def test_volume_remote_refresh_json(self, cli_runner: CliRunner) -> None:
        with patch("pengulink.cli.main.run_ipc_call", return_value={"success": True}):
            result = cli_runner.invoke(
                cli, ["volume", "remote", "phone_1", "refresh", "--json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["device_id"] == "phone_1"
            assert data["success"] is True

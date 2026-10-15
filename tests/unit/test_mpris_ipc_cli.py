"""Unit tests for MPRIS JSON-RPC IPC server/client and Click/Rich CLI."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.exceptions import (
    ActionUnsupportedError,
    DeviceNotFoundError,
    IPCError,
    MprisAdapterError,
    MprisPlayerNotFoundError,
)
from pengulink.daemon.ipc import (
    DEVICE_NOT_FOUND,
    MPRIS_ERROR,
    PenguLinkIPCClient,
    PenguLinkIPCServer,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner(env={"COLUMNS": "160"})


# =====================================================================
# IPC Server and Client Tests
# =====================================================================

class TestMprisIPCServerAndClient:
    async def test_ipc_mpris_list_players(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_mpris_list.sock"

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            if method == "mpris_list_players":
                assert params["device_id"] == "phone_1"
                return {"device_id": "phone_1", "players": ["Spotify", "VLC"], "active_player": "Spotify"}
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            res = await client.mpris_list_players("phone_1")
            assert res["device_id"] == "phone_1"
            assert res["players"] == ["Spotify", "VLC"]
            assert res["active_player"] == "Spotify"
        finally:
            await server.stop()

    async def test_ipc_mpris_get_status(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_mpris_status.sock"

        sample_status = {
            "player": "Spotify",
            "title": "Bohemian Rhapsody",
            "artist": "Queen",
            "album": "A Night at the Opera",
            "isPlaying": True,
            "pos": 45000,
            "length": 354000,
            "volume": 75,
            "canPlay": True,
            "canPause": True,
            "canGoNext": True,
            "canGoPrevious": True,
            "canSeek": True,
            "loopStatus": "None",
            "shuffle": False,
        }

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            if method == "mpris_get_status":
                assert params["device_id"] == "phone_1"
                assert params["player"] == "Spotify"
                return {
                    "device_id": "phone_1",
                    "is_connected": True,
                    "player": "Spotify",
                    "status": sample_status,
                }
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            res = await client.mpris_get_status("phone_1", "Spotify")
            assert res["player"] == "Spotify"
            st = res["status"]
            assert st["title"] == "Bohemian Rhapsody"
            assert st["isPlaying"] is True
            assert st["pos"] == 45000
        finally:
            await server.stop()

    async def test_ipc_mpris_transport_and_controls(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_mpris_ctrl.sock"
        calls: List[str] = []

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            calls.append(method)
            if method == "mpris_send_action":
                assert params["device_id"] == "phone_1"
                assert params["player"] == "Spotify"
                assert params["action"] == "Play"
                return {"status": "sent", "player": "Spotify"}
            elif method == "mpris_set_volume":
                assert params["volume"] == 80
                return {"status": "sent", "player": "Spotify"}
            elif method == "mpris_seek":
                assert params["offset_ms"] == 5000
                return {"status": "sent", "player": "Spotify"}
            elif method == "mpris_set_position":
                assert params["position_ms"] == 30000
                return {"status": "sent", "player": "Spotify"}
            elif method == "mpris_request_player_list":
                assert params["device_id"] == "phone_1"
                return {"status": "requested"}
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            assert (await client.mpris_send_action("phone_1", "Play", "Spotify"))["status"] == "sent"
            assert (await client.mpris_set_volume("phone_1", 80, "Spotify"))["status"] == "sent"
            assert (await client.mpris_seek("phone_1", 5000, "Spotify"))["status"] == "sent"
            assert (await client.mpris_set_position("phone_1", 30000, "Spotify"))["status"] == "sent"
            assert (await client.mpris_request_player_list("phone_1"))["status"] == "requested"
            assert len(calls) == 5
        finally:
            await server.stop()

    async def test_ipc_mpris_error_handling(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_mpris_err.sock"

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            if method == "mpris_get_status":
                if params.get("player") == "Ghost":
                    raise MprisPlayerNotFoundError("No player named 'Ghost' found")
                raise MprisAdapterError("Adapter communication failed")
            elif method == "mpris_send_action":
                raise ActionUnsupportedError("Action not supported")
            elif method == "mpris_list_players":
                raise DeviceNotFoundError("Device phone_404 not found")
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)

            with pytest.raises(IPCError) as exc_info:
                await client.mpris_get_status("phone_1", "Ghost")
            assert str(MPRIS_ERROR) in str(exc_info.value)
            assert "No player named 'Ghost' found" in str(exc_info.value)

            with pytest.raises(IPCError) as exc_info:
                await client.mpris_get_status("phone_1", "Other")
            assert str(MPRIS_ERROR) in str(exc_info.value)
            assert "Adapter communication failed" in str(exc_info.value)

            with pytest.raises(IPCError) as exc_info:
                await client.mpris_send_action("phone_1", "InvalidAction", "p1")
            assert str(MPRIS_ERROR) in str(exc_info.value)

            with pytest.raises(IPCError) as exc_info:
                await client.mpris_list_players("phone_404")
            assert str(DEVICE_NOT_FOUND) in str(exc_info.value)
        finally:
            await server.stop()


# =====================================================================
# Click CLI Tests
# =====================================================================

class TestMprisCLI:
    def test_cli_mpris_players_table(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_players = {"device_id": "phone_123", "players": ["Spotify", "VLC"], "active_player": "Spotify"}
        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_players):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "mpris", "players", "phone_123"])
            assert res.exit_code == 0
            assert "Media Players on phone_123" in res.output
            assert "Spotify" in res.output
            assert "VLC" in res.output

    def test_cli_mpris_players_empty(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_players = {"device_id": "phone_123", "players": []}
        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_players):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "mpris", "players", "phone_123"])
            assert res.exit_code == 0
            assert "No active media players found for device 'phone_123'" in res.output

    def test_cli_mpris_players_json(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_players = {"device_id": "phone_123", "players": ["Spotify"]}
        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_players):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "mpris", "players", "phone_123", "--json"])
            assert res.exit_code == 0
            parsed = json.loads(res.output)
            assert parsed == ["Spotify"]

    def test_cli_mpris_status_display(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_status = {
            "player": "Spotify",
            "title": "Starman",
            "artist": "David Bowie",
            "album": "The Rise and Fall of Ziggy Stardust",
            "isPlaying": True,
            "pos": 75000,
            "length": 250000,
            "volume": 60,
            "canPlay": True,
            "canPause": True,
            "canGoNext": True,
            "canGoPrevious": True,
            "canSeek": True,
            "loopStatus": "Playlist",
            "shuffle": True,
        }

        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
            return_value={"device_id": "phone_123", "status": sample_status},
        ):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "mpris", "status", "phone_123", "--player", "Spotify"])
            assert res.exit_code == 0
            assert "Starman" in res.output
            assert "David Bowie" in res.output
            assert "Playing" in res.output
            assert "60%" in res.output
            assert "Playlist" in res.output

    def test_cli_mpris_status_json(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_status = {
            "player": "Spotify",
            "title": "Starman",
            "artist": "David Bowie",
            "isPlaying": True,
        }

        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
            return_value={"device_id": "phone_123", "status": sample_status},
        ):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "mpris", "status", "phone_123", "--json"])
            assert res.exit_code == 0
            parsed = json.loads(res.output)
            assert parsed["title"] == "Starman"
            assert parsed["artist"] == "David Bowie"

    def test_cli_mpris_status_empty(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value={"device_id": "phone_123", "status": None}):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "mpris", "status", "phone_123"])
            assert res.exit_code == 0
            assert "No media player status available for device 'phone_123'" in res.output

    def test_cli_mpris_transport_commands(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        commands_and_actions = [
            ("play", "Play"),
            ("pause", "Pause"),
            ("play-pause", "PlayPause"),
            ("next", "Next"),
            ("previous", "Previous"),
            ("stop", "Stop"),
        ]

        for cmd, expected_action in commands_and_actions:
            with patch(
                "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
                return_value={"status": "sent", "player": "VLC"},
            ) as mock_call:
                res = cli_runner.invoke(cli, ["-s", str(mock_socket), "mpris", cmd, "phone_123", "--player", "VLC"])
                assert res.exit_code == 0
                assert f"Action '{expected_action}' sent to VLC on 'phone_123'" in res.output
                mock_call.assert_called_once_with(
                    method="mpris_send_action",
                    params={"device_id": "phone_123", "action": expected_action, "player": "VLC"},
                    timeout=10.0,
                )

    def test_cli_mpris_volume(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        # Valid volume
        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
            return_value={"status": "sent", "player": "player"},
        ) as mock_call:
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "mpris", "volume", "phone_123", "85"])
            assert res.exit_code == 0
            assert "Volume set to 85% on player (phone_123)" in res.output
            mock_call.assert_called_once_with(
                method="mpris_set_volume",
                params={"device_id": "phone_123", "volume": 85},
                timeout=10.0,
            )

        # Invalid volume < 0
        res_neg = cli_runner.invoke(cli, ["-s", str(mock_socket), "mpris", "volume", "--", "phone_123", "-10"])
        assert res_neg.exit_code != 0
        assert "Volume must be between 0 and 100" in res_neg.output

        # Invalid volume > 100
        res_high = cli_runner.invoke(cli, ["-s", str(mock_socket), "mpris", "volume", "phone_123", "150"])
        assert res_high.exit_code != 0
        assert "Volume must be between 0 and 100" in res_high.output

    def test_cli_mpris_seek(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
            return_value={"status": "sent", "player": "MockPlayer"},
        ) as mock_call:
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "mpris", "seek", "phone_123", "5000", "--player", "MockPlayer"])
            assert res.exit_code == 0
            assert "+5000ms" in res.output
            mock_call.assert_called_once_with(
                method="mpris_seek",
                params={"device_id": "phone_123", "offset_ms": 5000, "player": "MockPlayer"},
                timeout=10.0,
            )

    def test_cli_mpris_position(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
            return_value={"status": "sent", "player": "MockPlayer"},
        ) as mock_call:
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "mpris", "position", "phone_123", "45000"])
            assert res.exit_code == 0
            assert "Position set to 45000ms" in res.output
            mock_call.assert_called_once_with(
                method="mpris_set_position",
                params={"device_id": "phone_123", "position_ms": 45000},
                timeout=10.0,
            )

    def test_cli_mpris_rich_markup_injection_safety(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Verify that malicious remote player metadata containing Rich markup does NOT exploit or crash."""
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        malicious_status = {
            "player": "[bold red]PwnedPlayer[/bold red]",
            "title": "[blink green][link=http://malicious.site]Injected Title[/link][/blink]",
            "artist": "[italic red]<script>alert(1)</script>[/italic red]",
            "album": "[strike blue]Hacked Album[/strike blue]",
            "isPlaying": True,
            "pos": 10000,
            "length": 60000,
            "volume": 50,
        }

        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
            return_value={"device_id": "phone_123", "status": malicious_status},
        ):
            # This should render successfully without throwing MarkupError or formatting raw tags
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "mpris", "status", "phone_123"])
            assert res.exit_code == 0
            assert "PwnedPlayer" in res.output
            assert "Injected Title" in res.output
            assert "alert(1)" in res.output
            assert "Hacked Album" in res.output

    def test_cli_mpris_error_handling(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
            side_effect=IPCError("Target player not active"),
        ):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "mpris", "status", "phone_123"])
            assert res.exit_code == 1
            assert "Target player not active" in res.output

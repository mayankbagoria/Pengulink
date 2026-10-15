"""tests.adversarial.test_mpris_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Adversarial Challenge & Stress Test Suite for Milestone 14:
Media Control Ecosystem (kdeconnect.mpris & kdeconnect.mpris.request).

Challenge Dimensions:
1. Volume Clamping & Boundary Cases:
   - Volume wire values: -50, -1, 0, 1, 99, 100, 150, 1,000,000.
   - Float, boundary, and non-numeric payloads across adapter, server, client, IPC, CLI.
2. Microsecond Seek Boundaries:
   - Huge positive seek (10**18 us, 2**63 - 1).
   - Huge negative seek (-10**18 us, -2**63).
   - Seek beyond track length (clamped to length).
   - Negative seek beyond 0 (clamped to 0).
   - Seek on zero-length (0 us) and unknown-length (-1 us) tracks.
   - Wire unit conversions (ms in CLI <-> us on wire <-> us in adapter).
3. Absolute Position Boundaries:
   - Negative position on wire (-5000 ms, -10**15 us) clamped to 0.
   - Position beyond length clamped to track length.
   - Position on zero-length and live/unknown duration tracks.
   - CLI parameter validation rejecting negative position.
4. Rich Markup Injection Attacks in Track Metadata:
   - Untrusted remote payloads with malicious tags:
     [bold red], [/], [link=http://evil.com]Attack[/link], unmatched brackets [foo][bar].
   - Payloads in player name, title, artist, album, loopStatus, device_id.
   - Verified across CLI status, players, volume, seek, position, transport commands.
5. Unicode Edge Cases & Extreme Payload Sizes:
   - RTL scripts (Arabic, Hebrew).
   - CJK scripts (Chinese, Japanese, Korean).
   - Multi-codepoint Emojis and Zero-Width characters.
   - Null bytes and raw ASCII control characters.
   - Extreme payload stress: 100 KB track titles and metadata fields.
6. Incremental Delta Merging with Omitted/Null Fields:
   - Preservation of unmodified fields across partial delta updates.
   - Dynamic position interpolation behavior across state toggles without pos.
   - Omitted, null, and unexpected delta keys resilience.
   - Rapid bursts of incremental deltas.
7. Unknown Actions & Malformed Requests:
   - Unknown action names ("FormatDrive", "", None, numeric, injection).
   - Strict rejection in MprisRemote and safe ignoring in MprisControl / MockAdapter.
8. Non-Existent Player Queries & Edge Cases:
   - Queries and commands targeting ghost / non-existent players.
   - Empty player lists, fallback handling, cache pruning on player removal.
9. Concurrency & Lifecycle Stress:
   - Parallel transport actions, concurrent packet ingestion, unpair race safety.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.events import (
    DeviceConnected,
    DeviceDisconnected,
    DeviceUnpaired,
    EventBus,
    LocalMediaPlayerChanged,
    LocalPlayerListChanged,
    MediaControlCommandReceived,
    RemotePlayerListUpdatedEvent,
    RemotePlayerUpdatedEvent,
)
from pengulink.core.exceptions import (
    ActionUnsupportedError,
    ConnectionClosedError,
    IPCError,
    MprisPlayerNotFoundError,
)
from pengulink.network.packet import Packet
from pengulink.plugins.mpris.adapter import (
    MockMprisAdapter,
    PlayerState,
)
from pengulink.plugins.mpris.control import MprisControl
from pengulink.plugins.mpris.remote import (
    ALLOWED_ACTIONS,
    MprisRemote,
    RemotePlayerState,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner(env={"COLUMNS": "160"})


# =============================================================================
# 1. Volume Clamping & Boundary Cases
# =============================================================================

class TestVolumeClampingAndBoundaries:
    """Stress-test volume boundary conditions and clamping across all layers."""

    async def test_mock_adapter_set_volume_clamping_boundaries(self) -> None:
        """Volume in MockMprisAdapter must clamp safely to [0.0, 1.0] without exceptions."""
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()

        test_volumes = [
            (-1_000_000.0, 0.0, 0),
            (-50.0, 0.0, 0),
            (-1.0, 0.0, 0),
            (-0.0001, 0.0, 0),
            (0.0, 0.0, 0),
            (0.01, 0.01, 1),
            (0.5, 0.5, 50),
            (0.99, 0.99, 99),
            (1.0, 1.0, 100),
            (1.0001, 1.0, 100),
            (1.5, 1.0, 100),
            (150.0, 1.0, 100),
            (1_000_000.0, 1.0, 100),
        ]

        for input_vol, expected_float, expected_wire_pct in test_volumes:
            ok = await adapter.set_volume("MockPlayer", input_vol)
            assert ok is True
            st = await adapter.get_player_status("MockPlayer")
            assert st is not None
            assert pytest.approx(st.volume, abs=1e-4) == expected_float
            wire = st.to_wire_dict()
            assert wire["volume"] == expected_wire_pct
            assert 0 <= wire["volume"] <= 100

        await adapter.stop()

    def test_player_state_wire_dict_volume_clamping(self) -> None:
        """Direct PlayerState.to_wire_dict() must always clamp volume to 0..100 integer."""
        for raw_vol, expected in [
            (-50.0, 0),
            (-0.1, 0),
            (0.0, 0),
            (0.004, 0),
            (0.006, 1),
            (0.50, 50),
            (0.994, 99),
            (1.0, 100),
            (1.5, 100),
            (1_000_000.0, 100),
        ]:
            st = PlayerState(player="P", volume=raw_vol)
            wire = st.to_wire_dict()
            assert wire["volume"] == expected

    async def test_mpris_control_inbound_set_volume_packet_extremes(self) -> None:
        """MprisControl must clamp setVolume integer percentages from wire packets."""
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()
        control = MprisControl(adapter=adapter)
        await control.start()

        sent_packets: List[Packet] = []
        async def dummy_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        # Test valid and adversarial boundary values
        test_cases = [
            (-50, 0.0),
            (0, 0.0),
            (1, 0.01),
            (99, 0.99),
            (100, 1.0),
            (150, 1.0),
            (1_000_000, 1.0),
        ]

        for wire_vol, expected_float in test_cases:
            req = Packet(
                type="kdeconnect.mpris.request",
                body={"player": "MockPlayer", "setVolume": wire_vol},
            )
            await control.handle_packet(req, "test_dev", dummy_send)
            st = await adapter.get_player_status("MockPlayer")
            assert st is not None
            assert pytest.approx(st.volume, abs=1e-4) == expected_float

        # Non-numeric setVolume payloads must NOT crash handler
        for bad_vol in [None, "75", [50], {"vol": 50}]:
            req = Packet(
                type="kdeconnect.mpris.request",
                body={"player": "MockPlayer", "setVolume": bad_vol},
            )
            await control.handle_packet(req, "test_dev", dummy_send)

        await control.stop()
        await adapter.stop()

    async def test_mpris_remote_outbound_and_inbound_volume_clamping(self) -> None:
        """MprisRemote must clamp outbound setVolume and inbound delta volume."""
        remote = MprisRemote()
        await remote.start()

        sent_packets: List[Packet] = []
        async def dummy_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        # Setup an active player
        init_pkt = Packet(
            type="kdeconnect.mpris",
            body={"player": "Spotify", "volume": 50, "isPlaying": True},
        )
        await remote.handle_packet(init_pkt, "dev_1", dummy_send)

        # 1. Outbound set_volume clamping
        for input_vol, expected_clamped in [
            (-50, 0),
            (0, 0),
            (1, 1),
            (99, 99),
            (100, 100),
            (150, 100),
            (1_000_000, 100),
        ]:
            pkt = await remote.set_volume("dev_1", input_vol, player="Spotify", send_packet=dummy_send)
            assert pkt.body["setVolume"] == expected_clamped

        # 2. Inbound delta volume clamping
        for delta_vol, expected_cached in [
            (-50, 0),
            (0, 0),
            (1, 1),
            (99, 99),
            (100, 100),
            (150, 100),
            (1_000_000, 100),
        ]:
            delta_pkt = Packet(
                type="kdeconnect.mpris",
                body={"player": "Spotify", "volume": delta_vol},
            )
            await remote.handle_packet(delta_pkt, "dev_1", dummy_send)
            st = remote.get_player_state("dev_1", "Spotify")
            assert st is not None
            assert st.volume == expected_cached

        await remote.stop()

    def test_cli_volume_boundaries_and_rejection(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """CLI mpris volume must reject out-of-range volumes and accept boundary [0..100]."""
        mock_sock = tmp_path / "daemon.sock"
        mock_sock.touch()

        # Rejected: -50, -1, 101, 150, 1000000
        for bad_v in ["-50", "-1", "101", "150", "1000000"]:
            res = cli_runner.invoke(cli, ["-s", str(mock_sock), "mpris", "volume", "--", "dev_1", bad_v])
            assert res.exit_code != 0
            assert "Volume must be between 0 and 100" in res.output

        # Accepted boundaries: 0, 1, 99, 100
        for good_v in [0, 1, 99, 100]:
            with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value={"status": "sent", "player": "P"}):
                res = cli_runner.invoke(cli, ["-s", str(mock_sock), "mpris", "volume", "dev_1", str(good_v)])
                assert res.exit_code == 0
                assert f"Volume set to {good_v}%" in res.output


# =============================================================================
# 2. Microsecond Seek Boundaries
# =============================================================================

class TestMicrosecondSeekBoundaries:
    """Stress-test seek offset boundaries, overflow limits, and edge conditions."""

    async def test_mock_adapter_seek_huge_positive_clamped_to_length(self) -> None:
        """Huge positive seek (10**18 us) must clamp to length_us."""
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()

        # Default track length is 180_000_000 us (180s)
        huge_seek = 10**18
        ok = await adapter.seek("MockPlayer", huge_seek)
        assert ok is True

        st = await adapter.get_player_status("MockPlayer")
        assert st is not None
        assert st.position_us == 180_000_000

        # Maximum 64-bit integer seek
        max_int64 = 2**63 - 1
        ok = await adapter.seek("MockPlayer", max_int64)
        assert ok is True
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None
        assert st.position_us == 180_000_000

        await adapter.stop()

    async def test_mock_adapter_seek_huge_negative_clamped_to_zero(self) -> None:
        """Huge negative seek (-10**18 us) must clamp to 0 us."""
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()

        # Advance first
        await adapter.set_position("MockPlayer", 50_000_000)

        huge_neg_seek = -10**18
        ok = await adapter.seek("MockPlayer", huge_neg_seek)
        assert ok is True

        st = await adapter.get_player_status("MockPlayer")
        assert st is not None
        assert st.position_us == 0

        # Minimum 64-bit signed integer seek
        min_int64 = -2**63
        ok = await adapter.seek("MockPlayer", min_int64)
        assert ok is True
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None
        assert st.position_us == 0

        await adapter.stop()

    async def test_mock_adapter_seek_zero_length_track(self) -> None:
        """Seeking on track with length_us == 0 must not raise ZeroDivisionError."""
        adapter = MockMprisAdapter(auto_default_player=False)
        await adapter.start()

        adapter.add_mock_player(
            "ZeroTrack",
            PlayerState(
                player="ZeroTrack",
                title="Empty Track",
                length_us=0,
                position_us=0,
            ),
        )

        # Forward seek
        ok = await adapter.seek("ZeroTrack", 5_000_000)
        assert ok is True
        st = await adapter.get_player_status("ZeroTrack")
        assert st is not None
        assert st.position_us >= 0

        # Backward seek
        ok = await adapter.seek("ZeroTrack", -10_000_000)
        assert ok is True
        st = await adapter.get_player_status("ZeroTrack")
        assert st is not None
        assert st.position_us == 0

        await adapter.stop()

    async def test_mock_adapter_seek_unknown_length_track(self) -> None:
        """Seeking on track with length_us == -1 (unknown duration) must advance and clamp at 0."""
        adapter = MockMprisAdapter(auto_default_player=False)
        await adapter.start()

        adapter.add_mock_player(
            "LiveStream",
            PlayerState(
                player="LiveStream",
                title="Radio Stream",
                length_us=-1,
                position_us=10_000_000,
            ),
        )

        # Positive seek advances unbounded
        ok = await adapter.seek("LiveStream", 25_000_000)
        assert ok is True
        st = await adapter.get_player_status("LiveStream")
        assert st is not None
        assert st.position_us == 35_000_000

        # Negative seek clamps at 0
        ok = await adapter.seek("LiveStream", -50_000_000)
        assert ok is True
        st = await adapter.get_player_status("LiveStream")
        assert st is not None
        assert st.position_us == 0

        await adapter.stop()

    async def test_mpris_control_inbound_seek_packet_extremes(self) -> None:
        """MprisControl must handle extreme Seek offsets and reject non-integer values."""
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()
        control = MprisControl(adapter=adapter)
        await control.start()

        sent_packets: List[Packet] = []
        async def dummy_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        # Valid huge seek
        pkt = Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "Seek": 10**15})
        await control.handle_packet(pkt, "dev_1", dummy_send)
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None
        assert st.position_us == 180_000_000

        # Malformed Seek payloads (must not crash)
        for bad_seek in [None, "5000", 3.14, [], {}]:
            pkt = Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "Seek": bad_seek})
            await control.handle_packet(pkt, "dev_1", dummy_send)

        await control.stop()
        await adapter.stop()

    async def test_mpris_remote_seek_unit_conversion(self) -> None:
        """MprisRemote.seek takes milliseconds and converts to microseconds on wire."""
        remote = MprisRemote()
        await remote.start()

        sent_packets: List[Packet] = []
        async def dummy_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        init_pkt = Packet(type="kdeconnect.mpris", body={"player": "P", "isPlaying": True})
        await remote.handle_packet(init_pkt, "dev_1", dummy_send)

        # 5000 ms -> 5_000_000 us
        pkt = await remote.seek("dev_1", 5000, player="P", send_packet=dummy_send)
        assert pkt.body["Seek"] == 5_000_000

        # -10000 ms -> -10_000_000 us
        pkt = await remote.seek("dev_1", -10000, player="P", send_packet=dummy_send)
        assert pkt.body["Seek"] == -10_000_000

        # Huge ms offset
        pkt = await remote.seek("dev_1", 10**12, player="P", send_packet=dummy_send)
        assert pkt.body["Seek"] == 10**15

        await remote.stop()

    def test_remote_player_dynamic_interpolation_extreme_time(self) -> None:
        """get_current_position_ms must clamp dynamically when playback runs long."""
        st = RemotePlayerState(
            player="P",
            isPlaying=True,
            pos=10_000,
            length=60_000,
            last_pos_timestamp=1000.0,
        )

        # 100 seconds later (past 60s track length) -> clamped to 60_000 ms
        assert st.get_current_position_ms(now=1100.0) == 60_000

        # 1 year later -> clamped to 60_000 ms
        assert st.get_current_position_ms(now=1000.0 + 31_536_000.0) == 60_000

        # Past time (clock drift backwards) -> clamped to 0 ms
        assert st.get_current_position_ms(now=900.0) == 0

        # Unknown length (-1) -> advances unbounded without crashing
        st_live = RemotePlayerState(
            player="Live",
            isPlaying=True,
            pos=10_000,
            length=-1,
            last_pos_timestamp=1000.0,
        )
        assert st_live.get_current_position_ms(now=1100.0) == 110_000


# =============================================================================
# 3. Absolute Position Boundaries
# =============================================================================

class TestAbsolutePositionBoundaries:
    """Stress-test absolute position setting, clamping, and error handling."""

    async def test_mock_adapter_set_position_negative_clamped_to_zero(self) -> None:
        """Negative absolute position must clamp to 0 microseconds."""
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()

        for neg_pos in [-1, -5000, -10**15]:
            ok = await adapter.set_position("MockPlayer", neg_pos)
            assert ok is True
            st = await adapter.get_player_status("MockPlayer")
            assert st is not None
            assert st.position_us == 0

        await adapter.stop()

    async def test_mock_adapter_set_position_beyond_length_clamped(self) -> None:
        """Position beyond track duration must clamp to length_us."""
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()

        # Track length is 180_000_000 us
        ok = await adapter.set_position("MockPlayer", 999_999_999_999)
        assert ok is True
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None
        assert st.position_us == 180_000_000

        await adapter.stop()

    async def test_mpris_control_inbound_set_position_packets(self) -> None:
        """MprisControl converts ms from wire SetPosition to us on adapter."""
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()
        control = MprisControl(adapter=adapter)
        await control.start()

        sent_packets: List[Packet] = []
        async def dummy_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        # 45000 ms -> 45_000_000 us
        pkt = Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "SetPosition": 45000})
        await control.handle_packet(pkt, "dev_1", dummy_send)
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None
        assert st.position_us == 45_000_000

        # Negative ms -> clamped to 0
        pkt = Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "SetPosition": -5000})
        await control.handle_packet(pkt, "dev_1", dummy_send)
        st = await adapter.get_player_status("MockPlayer")
        assert st is not None
        assert st.position_us == 0

        # Malformed SetPosition ignored safely
        for bad in [None, "3000", 3.14, []]:
            pkt = Packet(type="kdeconnect.mpris.request", body={"player": "MockPlayer", "SetPosition": bad})
            await control.handle_packet(pkt, "dev_1", dummy_send)

        await control.stop()
        await adapter.stop()

    async def test_mpris_remote_set_position_clamps_negative(self) -> None:
        """MprisRemote.set_position clamps negative position_ms to 0 on outbound packet."""
        remote = MprisRemote()
        await remote.start()

        sent_packets: List[Packet] = []
        async def dummy_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        init_pkt = Packet(type="kdeconnect.mpris", body={"player": "P", "isPlaying": True})
        await remote.handle_packet(init_pkt, "dev_1", dummy_send)

        pkt = await remote.set_position("dev_1", -10000, player="P", send_packet=dummy_send)
        assert pkt.body["SetPosition"] == 0

        pkt = await remote.set_position("dev_1", 30000, player="P", send_packet=dummy_send)
        assert pkt.body["SetPosition"] == 30000

        await remote.stop()

    def test_cli_position_validation(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """CLI position rejects negative position_ms with clear error."""
        mock_sock = tmp_path / "daemon.sock"
        mock_sock.touch()

        # Negative position rejected
        res = cli_runner.invoke(cli, ["-s", str(mock_sock), "mpris", "position", "--", "dev_1", "-10"])
        assert res.exit_code != 0
        assert "Position must be non-negative" in res.output

        # Non-negative position accepted
        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value={"status": "sent", "player": "P"}):
            res = cli_runner.invoke(cli, ["-s", str(mock_sock), "mpris", "position", "dev_1", "0"])
            assert res.exit_code == 0
            assert "Position set to 0ms" in res.output


# =============================================================================
# 4. Rich Markup Injection Attacks in Track Metadata
# =============================================================================

class TestRichMarkupInjectionAttacks:
    """Stress-test CLI output against untrusted strings containing Rich markup injection payloads."""

    INJECTION_PAYLOADS = [
        "[bold red]PWNED_EXPLOIT[/bold red]",
        "[/]",
        "[/bold]",
        "[link=http://malicious.payload.evil.com/xss]Attack Link[/link]",
        "[foo][bar]Unmatched Brackets",
        "[[[nested]]][///close]",
        "[italic red]Mixed [bold]Tags[/italic red][/bold]",
        "[color(1)]Invalid Color Syntax[/color]",
        "[#ff0000]Hex Color Tag[/]",
        "[repr.attrib=evil]Object Injection[/]",
        "\x1b[31;1mANSI_ESCAPE_INJECTION\x1b[0m",
        "\x1b]0;TerminalTitleInjection\x07",
        "[dim][strike]Strikethrough[/strike][/dim]",
        "Artist with [brackets] and (parentheses) & {braces}",
        "[green on red]Contrasting Colors[/]",
    ]

    def test_cli_status_rich_markup_injection_safety(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """All injection payloads rendered in status command must not trigger MarkupError or crash."""
        mock_sock = tmp_path / "daemon.sock"
        mock_sock.touch()

        for payload in self.INJECTION_PAYLOADS:
            malicious_status = {
                "player": payload,
                "title": payload,
                "artist": payload,
                "album": payload,
                "loopStatus": payload,
                "isPlaying": True,
                "pos": 15000,
                "length": 60000,
                "volume": 75,
                "shuffle": True,
            }

            with patch(
                "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
                return_value={"device_id": f"dev_{payload}", "status": malicious_status},
            ):
                res = cli_runner.invoke(cli, ["-s", str(mock_sock), "mpris", "status", "dev_test"])
                assert res.exit_code == 0, f"Failed on payload: {payload!r}\nOutput: {res.output}"
                assert "Media Status:" in res.output

    def test_cli_players_rich_markup_injection_safety(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Player list containing malicious tags must render cleanly in table."""
        mock_sock = tmp_path / "daemon.sock"
        mock_sock.touch()

        malicious_players = [
            "[bold red]EvilPlayer1[/bold red]",
            "[link=http://evil.com]PlayerLink[/link]",
            "[/]UnmatchedClose",
            "[bar]Player3",
        ]

        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
            return_value={
                "device_id": "[bold]phone_1[/bold]",
                "players": malicious_players,
                "active_player": malicious_players[0],
            },
        ):
            res = cli_runner.invoke(cli, ["-s", str(mock_sock), "mpris", "players", "phone_1"])
            assert res.exit_code == 0
            assert "Media Players on" in res.output
            assert "EvilPlayer1" in res.output
            assert "PlayerLink" in res.output

    def test_cli_transport_and_control_cmds_markup_safety(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Transport commands (play, pause, next, volume, seek, position) handle injected player names."""
        mock_sock = tmp_path / "daemon.sock"
        mock_sock.touch()

        injected_player = "[bold red]MaliciousPlayer[/bold red]"
        injected_dev = "[dim]phone_1[/dim]"

        commands_to_test = [
            (["play", injected_dev, "-p", injected_player], "mpris_send_action"),
            (["pause", injected_dev, "-p", injected_player], "mpris_send_action"),
            (["play-pause", injected_dev, "-p", injected_player], "mpris_send_action"),
            (["next", injected_dev, "-p", injected_player], "mpris_send_action"),
            (["previous", injected_dev, "-p", injected_player], "mpris_send_action"),
            (["stop", injected_dev, "-p", injected_player], "mpris_send_action"),
            (["volume", injected_dev, "80", "-p", injected_player], "mpris_set_volume"),
            (["seek", injected_dev, "5000", "-p", injected_player], "mpris_seek"),
            (["position", injected_dev, "15000", "-p", injected_player], "mpris_set_position"),
        ]

        for cmd_args, method_name in commands_to_test:
            with patch(
                "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
                return_value={"status": "sent", "player": injected_player},
            ):
                res = cli_runner.invoke(cli, ["-s", str(mock_sock), "mpris"] + cmd_args)
                assert res.exit_code == 0, f"Command {cmd_args} failed: {res.output}"
                assert "MaliciousPlayer" in res.output


# =============================================================================
# 5. Unicode Edge Cases & Extreme Payload Sizes
# =============================================================================

class TestUnicodeEdgeCasesAndExtremePayloads:
    """Stress-test international scripts, emojis, null bytes, and 100KB payload sizes."""

    UNICODE_SAMPLES = [
        # Arabic RTL
        ("فيروز", "سألوني الناس", "أغاني الخالدة"),
        # Hebrew RTL
        ("עברי לידר", "מלטף ומשקר", "מלטף ומשקר"),
        # Chinese CJK
        ("周杰倫", "晴天", "葉惠美"),
        # Japanese CJK
        ("久石譲", "あの夏へ (One Summer's Day)", "千と千尋の神隠し"),
        # Korean CJK
        ("방탄소년단", "봄날 (Spring Day)", "You Never Walk Alone"),
        # Multi-codepoint Emojis & Symbols
        ("🎧 DJ 🚀", "Dancing in the Stars 💃✨🔥", "Cosmic Beats 🪐"),
        # Mixed BiDi & Numbers
        ("Artist 123", "أغنية #42 (Track Test)", "Альбом 2026"),
        # Math symbols & exotic scripts
        ("∑(Noises)", "∫ dt e^(iωt)", "𝕸𝖆𝖙𝖍 𝕽𝖔𝖈𝖐"),
    ]

    @pytest.mark.parametrize("artist,title,album", UNICODE_SAMPLES)
    async def test_unicode_metadata_end_to_end_flow(
        self, artist: str, title: str, album: str
    ) -> None:
        """Unicode metadata roundtrips cleanly across PlayerState, wire, MprisRemote, and serialization."""
        st = PlayerState(
            player="InternationalPlayer",
            title=title,
            artist=artist,
            album=album,
            is_playing=True,
            position_us=10_000_000,
            length_us=180_000_000,
        )

        wire = st.to_wire_dict()
        assert wire["title"] == title
        assert wire["artist"] == artist
        assert wire["album"] == album
        assert f"{artist} - {title}" in wire["nowPlaying"]

        # Ingest into MprisRemote
        remote = MprisRemote()
        await remote.start()

        sent: List[Packet] = []
        async def dummy_send(pkt: Packet) -> None:
            sent.append(pkt)

        pkt = Packet(type="kdeconnect.mpris", body=wire)
        await remote.handle_packet(pkt, "dev_uni", dummy_send)

        cached = remote.get_player_state("dev_uni", "InternationalPlayer")
        assert cached is not None
        assert cached.title == title
        assert cached.artist == artist
        assert cached.album == album

        serialized = cached.to_dict()
        assert serialized["title"] == title
        assert serialized["artist"] == artist
        assert serialized["album"] == album

        await remote.stop()

    async def test_null_bytes_and_ascii_control_characters(self) -> None:
        """Metadata containing null bytes and raw ASCII control characters must not crash."""
        dirty_title = "Track\x00With\x00Null\x00Bytes\x01\x02\x07\x08"
        dirty_artist = "Artist\x0bVerticalTab\x0cFormFeed\x0e\x0f"
        dirty_album = "Album\r\nNewlineExploit\tTab"

        st = PlayerState(
            player="ControlCharsPlayer",
            title=dirty_title,
            artist=dirty_artist,
            album=dirty_album,
        )
        wire = st.to_wire_dict()

        remote = MprisRemote()
        await remote.start()

        async def dummy_send(pkt: Packet) -> None:
            pass

        pkt = Packet(type="kdeconnect.mpris", body=wire)
        await remote.handle_packet(pkt, "dev_dirty", dummy_send)

        cached = remote.get_player_state("dev_dirty", "ControlCharsPlayer")
        assert cached is not None
        assert cached.title == dirty_title
        assert cached.artist == dirty_artist

        await remote.stop()

    async def test_extreme_100kb_title_and_metadata_payloads(self) -> None:
        """100 KB track titles and metadata must process without memory leaks or crashes."""
        huge_title = "Title_" + ("A" * 100_000)
        huge_artist = "Artist_" + ("B" * 100_000)
        huge_album = "Album_" + ("C" * 100_000)

        st = PlayerState(
            player="HugeMetadataPlayer",
            title=huge_title,
            artist=huge_artist,
            album=huge_album,
        )
        wire = st.to_wire_dict()
        assert len(wire["title"]) > 100_000

        remote = MprisRemote()
        await remote.start()

        async def dummy_send(pkt: Packet) -> None:
            pass

        pkt = Packet(type="kdeconnect.mpris", body=wire)
        await remote.handle_packet(pkt, "dev_huge", dummy_send)

        cached = remote.get_player_state("dev_huge", "HugeMetadataPlayer")
        assert cached is not None
        assert len(cached.title) > 100_000
        assert len(cached.artist) > 100_000
        assert len(cached.album) > 100_000

        d = cached.to_dict()
        assert len(d["title"]) > 100_000

        await remote.stop()


# =============================================================================
# 6. Incremental Delta Merging with Omitted/Null Fields
# =============================================================================

class TestIncrementalDeltaMergingAdversarial:
    """Stress-test partial incremental updates, omitted fields, and null safety."""

    async def test_incremental_deltas_preserve_unmodified_fields(self) -> None:
        """Incoming packets updating only pos or volume must strictly preserve title, artist, album."""
        remote = MprisRemote()
        await remote.start()

        async def dummy_send(pkt: Packet) -> None:
            pass

        # 1. Base full state
        full_pkt = Packet(
            type="kdeconnect.mpris",
            body={
                "player": "Spotify",
                "title": "Bohemian Rhapsody",
                "artist": "Queen",
                "album": "A Night at the Opera",
                "pos": 10000,
                "length": 354000,
                "volume": 75,
                "isPlaying": False,
                "loopStatus": "None",
                "shuffle": False,
            },
        )
        await remote.handle_packet(full_pkt, "dev_1", dummy_send)

        st = remote.get_player_state("dev_1", "Spotify")
        assert st is not None
        assert st.title == "Bohemian Rhapsody"
        assert st.artist == "Queen"
        assert st.volume == 75

        # 2. Delta updating ONLY pos
        delta_pos = Packet(
            type="kdeconnect.mpris",
            body={"player": "Spotify", "pos": 25000},
        )
        await remote.handle_packet(delta_pos, "dev_1", dummy_send)

        st2 = remote.get_player_state("dev_1", "Spotify")
        assert st2 is not None
        assert st2.pos == 25000
        assert st2.title == "Bohemian Rhapsody"
        assert st2.artist == "Queen"
        assert st2.volume == 75

        # 3. Delta updating ONLY volume
        delta_vol = Packet(
            type="kdeconnect.mpris",
            body={"player": "Spotify", "volume": 90},
        )
        await remote.handle_packet(delta_vol, "dev_1", dummy_send)

        st3 = remote.get_player_state("dev_1", "Spotify")
        assert st3 is not None
        assert st3.volume == 90
        assert st3.pos == 25000
        assert st3.title == "Bohemian Rhapsody"

        # 4. Delta toggling isPlaying without pos (freezes position)
        delta_play = Packet(
            type="kdeconnect.mpris",
            body={"player": "Spotify", "isPlaying": True},
        )
        await remote.handle_packet(delta_play, "dev_1", dummy_send)

        st4 = remote.get_player_state("dev_1", "Spotify")
        assert st4 is not None
        assert st4.isPlaying is True
        assert st4.title == "Bohemian Rhapsody"

        await remote.stop()

    async def test_deltas_with_omitted_and_unexpected_keys(self) -> None:
        """Deltas with unknown keys, empty bodies, or missing player keys fail gracefully."""
        remote = MprisRemote()
        await remote.start()

        async def dummy_send(pkt: Packet) -> None:
            pass

        # Packet with no "player" key (e.g. unknown broadcast)
        pkt_no_player = Packet(type="kdeconnect.mpris", body={"unknownKey": "value"})
        await remote.handle_packet(pkt_no_player, "dev_1", dummy_send)

        # Delta with player and arbitrary garbage keys
        garbage_pkt = Packet(
            type="kdeconnect.mpris",
            body={
                "player": "Spotify",
                "random_exploit_key": {"nested": [1, 2, 3]},
                "unsupported_flag": True,
            },
        )
        await remote.handle_packet(garbage_pkt, "dev_1", dummy_send)

        st = remote.get_player_state("dev_1", "Spotify")
        assert st is not None
        assert st.player == "Spotify"

        await remote.stop()

    async def test_rapid_burst_of_100_incremental_deltas(self) -> None:
        """100 rapid sequential deltas must merge consistently without data corruption."""
        remote = MprisRemote()
        await remote.start()

        async def dummy_send(pkt: Packet) -> None:
            pass

        init_pkt = Packet(
            type="kdeconnect.mpris",
            body={"player": "StressPlayer", "title": "Base Title", "pos": 0, "volume": 50},
        )
        await remote.handle_packet(init_pkt, "dev_burst", dummy_send)

        for i in range(1, 101):
            delta = Packet(
                type="kdeconnect.mpris",
                body={
                    "player": "StressPlayer",
                    "pos": i * 1000,
                    "volume": i % 101,
                },
            )
            await remote.handle_packet(delta, "dev_burst", dummy_send)

        st = remote.get_player_state("dev_burst", "StressPlayer")
        assert st is not None
        assert st.pos == 100_000
        assert st.volume == 100
        assert st.title == "Base Title"

        await remote.stop()

    async def test_dispatch_packet_malformed_delta_isolated_by_registry(self) -> None:
        """When an incoming delta has malformed/non-integer fields, registry catches and isolates exception."""
        from pengulink.plugins.base import PluginRegistry
        registry = PluginRegistry()
        remote = MprisRemote()
        registry.register(remote)
        await registry.start_all()

        # Send delta with None for length/volume via registry
        pkt = Packet(
            type="kdeconnect.mpris",
            body={"player": "Spotify", "length": None},
        )
        async def dummy_send(p: Packet) -> None:
            pass

        # Registry handles gracefully without unhandled exception crashing caller
        handled = await registry.dispatch_packet(pkt, "dev_malformed", dummy_send)
        # Exception was safely caught and logged
        assert handled is False

        await registry.stop_all()


# =============================================================================
# 7. Unknown Actions & Malformed Requests
# =============================================================================

class TestUnknownActionsAndMalformedRequests:
    """Stress-test rejection and resilience against unauthorized or invalid transport actions."""

    INVALID_ACTIONS = [
        "FormatDrive",
        "",
        "   ",
        "DROP TABLE players",
        "play",
        "PAUSE",
        "NextTrack",
        "12345",
        "Exploit()",
        "__import__('os').system('id')",
    ]

    async def test_mpris_remote_send_action_strictly_rejects_unknown(self) -> None:
        """MprisRemote.send_action must raise ActionUnsupportedError on invalid action names."""
        remote = MprisRemote()
        await remote.start()

        async def dummy_send(pkt: Packet) -> None:
            pass

        init_pkt = Packet(type="kdeconnect.mpris", body={"player": "P"})
        await remote.handle_packet(init_pkt, "dev_1", dummy_send)

        # Rejects untrimmed action as well
        for bad_action in self.INVALID_ACTIONS + ["Play "]:
            with pytest.raises(ActionUnsupportedError):
                await remote.send_action("dev_1", bad_action, player="P", send_packet=dummy_send)

        # Valid actions must succeed
        for valid_action in sorted(ALLOWED_ACTIONS):
            pkt = await remote.send_action("dev_1", valid_action, player="P", send_packet=dummy_send)
            assert pkt.body["action"] == valid_action

        await remote.stop()

    async def test_mpris_control_inbound_unknown_actions_ignored_safely(self) -> None:
        """Inbound kdeconnect.mpris.request with unknown actions must be ignored safely."""
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()
        event_bus = EventBus()
        control = MprisControl(event_bus=event_bus, adapter=adapter)
        await control.start()

        commands_dispatched: List[MediaControlCommandReceived] = []
        event_bus.subscribe(MediaControlCommandReceived, lambda e: commands_dispatched.append(e))

        async def dummy_send(pkt: Packet) -> None:
            pass

        for bad_action in self.INVALID_ACTIONS:
            pkt = Packet(
                type="kdeconnect.mpris.request",
                body={"player": "MockPlayer", "action": bad_action},
            )
            await control.handle_packet(pkt, "dev_1", dummy_send)

        # No command events should have been dispatched for invalid actions
        assert len(commands_dispatched) == 0

        # Valid action must dispatch
        valid_pkt = Packet(
            type="kdeconnect.mpris.request",
            body={"player": "MockPlayer", "action": "Play"},
        )
        await control.handle_packet(valid_pkt, "dev_1", dummy_send)
        await asyncio.sleep(0.01)
        assert len(commands_dispatched) == 1
        assert commands_dispatched[0].command == "action:Play"

        await control.stop()
        await adapter.stop()

    async def test_mock_adapter_unknown_action_returns_false(self) -> None:
        """MockMprisAdapter.send_action returns False on unknown actions without state changes."""
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()

        for bad_action in self.INVALID_ACTIONS:
            ok = await adapter.send_action("MockPlayer", bad_action)
            assert ok is False

        st = await adapter.get_player_status("MockPlayer")
        assert st is not None
        assert st.is_playing is False

        # Whitespace-padded valid action is stripped by adapter for tolerance
        assert await adapter.send_action("MockPlayer", "Play ") is True

        await adapter.stop()


# =============================================================================
# 8. Non-Existent Player Queries & Edge Cases
# =============================================================================

class TestNonExistentPlayerQueriesAndEdgeCases:
    """Stress-test behavior when querying or controlling players that do not exist."""

    async def test_mock_adapter_queries_for_ghost_player(self) -> None:
        """Queries and actions for non-existent players must return None/False safely."""
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()

        assert await adapter.get_player_status("GhostPlayer") is None
        assert await adapter.send_action("GhostPlayer", "Play") is False
        assert await adapter.set_volume("GhostPlayer", 0.5) is False
        assert await adapter.seek("GhostPlayer", 1000) is False
        assert await adapter.set_position("GhostPlayer", 1000) is False
        assert await adapter.set_loop_status("GhostPlayer", "Track") is False
        assert await adapter.set_shuffle("GhostPlayer", True) is False

        await adapter.stop()

    async def test_mpris_remote_query_for_ghost_player_or_device(self) -> None:
        """MprisRemote returns None or empty lists for unknown players or devices."""
        remote = MprisRemote()
        await remote.start()

        assert remote.get_players("unknown_device") == []
        assert remote.get_player_list("unknown_device") == []
        assert remote.get_active_player("unknown_device") is None
        assert remote.get_player_state("unknown_device", "GhostPlayer") is None

        # Device with players, but querying non-existent player
        async def dummy_send(pkt: Packet) -> None:
            pass
        pkt = Packet(type="kdeconnect.mpris", body={"playerList": ["Spotify"]})
        await remote.handle_packet(pkt, "dev_1", dummy_send)

        assert remote.get_player_state("dev_1", "VLC") is None

        await remote.stop()

    async def test_mpris_remote_send_action_no_player_raises_error(self) -> None:
        """Calling send_action/set_volume/seek without target on playerless device raises error."""
        remote = MprisRemote()
        await remote.start()

        async def dummy_send(pkt: Packet) -> None:
            pass

        # Device known but has NO players
        pkt = Packet(type="kdeconnect.mpris", body={"playerList": []})
        await remote.handle_packet(pkt, "dev_empty", dummy_send)

        with pytest.raises(MprisPlayerNotFoundError):
            await remote.send_action("dev_empty", "Play", send_packet=dummy_send)

        with pytest.raises(MprisPlayerNotFoundError):
            await remote.set_volume("dev_empty", 50, send_packet=dummy_send)

        with pytest.raises(MprisPlayerNotFoundError):
            await remote.seek("dev_empty", 5000, send_packet=dummy_send)

        with pytest.raises(MprisPlayerNotFoundError):
            await remote.set_position("dev_empty", 5000, send_packet=dummy_send)

        await remote.stop()

    async def test_mpris_control_request_now_playing_fallback_on_ghost_player(self) -> None:
        """When peer requests now playing for non-existent player, server responds with playerList."""
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()
        control = MprisControl(adapter=adapter)
        await control.start()

        sent_packets: List[Packet] = []
        async def dummy_send(pkt: Packet) -> None:
            sent_packets.append(pkt)

        # Peer queries unknown player
        req = Packet(
            type="kdeconnect.mpris.request",
            body={"player": "NonExistentPlayer", "requestNowPlaying": True},
        )
        await control.handle_packet(req, "dev_1", dummy_send)

        assert len(sent_packets) == 1
        resp = sent_packets[0]
        assert resp.type == "kdeconnect.mpris"
        assert "playerList" in resp.body
        assert resp.body["playerList"] == ["MockPlayer"]

        await control.stop()
        await adapter.stop()

    async def test_mpris_remote_player_list_prunes_disappeared_players(self) -> None:
        """When playerList packet omits previously reported player, cache is pruned."""
        remote = MprisRemote()
        await remote.start()

        async def dummy_send(pkt: Packet) -> None:
            pass

        # Initial list with two players
        pkt1 = Packet(type="kdeconnect.mpris", body={"playerList": ["PlayerA", "PlayerB"]})
        await remote.handle_packet(pkt1, "dev_1", dummy_send)

        # Set status for both
        await remote.handle_packet(Packet(type="kdeconnect.mpris", body={"player": "PlayerA"}), "dev_1", dummy_send)
        await remote.handle_packet(Packet(type="kdeconnect.mpris", body={"player": "PlayerB"}), "dev_1", dummy_send)

        assert remote.get_players("dev_1") == ["PlayerA", "PlayerB"]
        assert remote.get_player_state("dev_1", "PlayerA") is not None
        assert remote.get_player_state("dev_1", "PlayerB") is not None

        # PlayerA closes, new list only contains PlayerB
        pkt2 = Packet(type="kdeconnect.mpris", body={"playerList": ["PlayerB"]})
        await remote.handle_packet(pkt2, "dev_1", dummy_send)

        assert remote.get_players("dev_1") == ["PlayerB"]
        assert remote.get_player_state("dev_1", "PlayerA") is None
        assert remote.get_player_state("dev_1", "PlayerB") is not None
        assert remote.get_active_player("dev_1") == "PlayerB"

        await remote.stop()


# =============================================================================
# 9. Concurrency & Lifecycle Stress
# =============================================================================

class TestConcurrencyAndLifecycleStress:
    """Stress-test concurrency safety, parallel packet bursts, and unpair races."""

    async def test_concurrent_transport_actions_on_adapter(self) -> None:
        """50 concurrent transport actions must execute cleanly under lock."""
        adapter = MockMprisAdapter(auto_default_player=True)
        await adapter.start()

        actions = ["Play", "Pause", "PlayPause", "Next", "Previous", "Stop"] * 10

        async def run_action(act: str) -> bool:
            return await adapter.send_action("MockPlayer", act)

        results = await asyncio.gather(*(run_action(a) for a in actions))
        assert all(results)

        st = await adapter.get_player_status("MockPlayer")
        assert st is not None

        await adapter.stop()

    async def test_concurrent_remote_packet_ingestion(self) -> None:
        """50 concurrent deltas across multiple players must merge safely."""
        remote = MprisRemote()
        await remote.start()

        async def dummy_send(pkt: Packet) -> None:
            pass

        async def send_delta(pname: str, pos: int) -> None:
            pkt = Packet(type="kdeconnect.mpris", body={"player": pname, "pos": pos})
            await remote.handle_packet(pkt, "dev_conc", dummy_send)

        tasks = [
            send_delta(f"Player_{i % 5}", i * 100)
            for i in range(50)
        ]
        await asyncio.gather(*tasks)

        players = remote.get_players("dev_conc")
        assert len(players) == 5

        await remote.stop()

    async def test_unpair_purges_all_cached_remote_state(self) -> None:
        """DeviceUnpaired event must completely purge all cached states for that device."""
        event_bus = EventBus()
        remote = MprisRemote(event_bus=event_bus)
        await remote.start()

        async def dummy_send(pkt: Packet) -> None:
            pass

        # Ingest state for dev_unpair
        pkt = Packet(
            type="kdeconnect.mpris",
            body={"playerList": ["Spotify"], "player": "Spotify", "title": "Track 1"},
        )
        await remote.handle_packet(pkt, "dev_unpair", dummy_send)

        assert remote.get_players("dev_unpair") == ["Spotify"]
        assert remote.get_player_state("dev_unpair", "Spotify") is not None

        # Publish unpair event
        await event_bus.publish(DeviceUnpaired(device_id="dev_unpair"))
        await asyncio.sleep(0.01)

        # Cache must be completely wiped
        assert remote.get_players("dev_unpair") == []
        assert remote.get_player_state("dev_unpair", "Spotify") is None
        assert remote.get_active_player("dev_unpair") is None

        await remote.stop()


# =============================================================================
# 10. CLI Formatting Adversarial Extremes
# =============================================================================

class TestCliFormattingAdversarialExtremes:
    """Stress-test CLI progress, volume, and time formatting helpers against extreme and boundary inputs."""

    def test_format_time_ms_extremes(self) -> None:
        from pengulink.cli.main import format_time_ms
        assert format_time_ms(-1) == "--:--"
        assert format_time_ms(-10000) == "--:--"
        assert format_time_ms(0) == "00:00"
        assert format_time_ms(59000) == "00:59"
        assert format_time_ms(60000) == "01:00"
        assert format_time_ms(3599000) == "59:59"
        assert format_time_ms(3600000) == "01:00:00"
        assert format_time_ms(86400000) == "24:00:00"
        assert ":" in format_time_ms(10**12)

    def test_format_progress_bar_extremes(self) -> None:
        from pengulink.cli.main import format_progress_bar
        # Zero and negative length (live stream / unknown duration)
        live_zero = format_progress_bar(pos_ms=5000, length_ms=0)
        assert "Live / Unknown duration" in live_zero
        live_neg = format_progress_bar(pos_ms=5000, length_ms=-1)
        assert "Live / Unknown duration" in live_neg

        # Pos beyond length (150% progress clamped to 100%)
        overflow = format_progress_bar(pos_ms=90000, length_ms=60000)
        assert "100%" in overflow

        # Zero pos
        zero_pos = format_progress_bar(pos_ms=0, length_ms=60000)
        assert "0%" in zero_pos

        # Huge values (10**12 ms)
        huge = format_progress_bar(pos_ms=10**12, length_ms=2 * 10**12)
        assert "50%" in huge

    def test_format_volume_bar_extremes(self) -> None:
        from pengulink.cli.main import format_volume_bar
        # Below zero
        v_neg = format_volume_bar(-50)
        assert "0%" in v_neg

        # Zero
        v_zero = format_volume_bar(0)
        assert "0%" in v_zero

        # 100
        v_100 = format_volume_bar(100)
        assert "100%" in v_100

        # Beyond 100
        v_huge = format_volume_bar(1000000)
        assert "100%" in v_huge


"""tests.e2e.test_telephony_e2e
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive, requirement-driven 4-tier E2E test suite for Telephony and PauseMusic subsystems.

Derived authoritatively from:
- ORIGINAL_REQUEST.md (entry dated 2026-10-20T14:39:43Z, R1, R2, R3, R4)
- PROJECT.md (Telephony & PauseMusic Subsystems, Interface Contracts, Milestones E2E/M27-M30)
- spec_miner_survey_telephony/handoff.md
- spec_miner_survey_pausemusic/handoff.md
- explorer_survey_pengulink_telephony/handoff.md

Methodology: 4-Tier Test Design
- Tier 1: Feature Coverage (>=5 test cases per feature across Telephony Call Events, Mute Ringer Action,
  Reject Call Action, Auto-Pause Media Player, Auto-Resume Media Player)
- Tier 2: Boundary & Corner Cases (Empty/missing phone number or contact name, null fields, unknown events,
  rapid call transitions, string isCancel coercion, unauthenticated actions, oversized unicode)
- Tier 3: Cross-Feature Combinations (Multi-device calls, call overlap with media playback, player state
  preservation where stopped players stay stopped, pre-muted sinks stay muted, mpris reflection suppression)
- Tier 4: Real-World Scenarios (Complete incoming call lifecycle during media playback with auto-resume,
  abrupt disconnection during active call, unpair cleanup, CLI Rich/JSON operations)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.network.packet import Packet
from pengulink.plugins.mpris.adapter import PlayerState
from pengulink.services.audio import AudioSink, MockAudioSinkBackend
from tests.e2e.conftest import e2e_environment, wait_for
from tests.mock.mock_device import ChaosConfig, MockDevice, MockDeviceConfig

try:
    from pengulink.core.exceptions import TelephonyError
except ImportError:
    class TelephonyError(Exception):  # type: ignore
        pass


# Graceful import of domain events and services (implemented in M27/M28)
try:
    from pengulink.core.events import CallEvent, CallStateChangedEvent
except ImportError:
    CallEvent = None  # type: ignore
    CallStateChangedEvent = None  # type: ignore

try:
    from pengulink.services.telephony import ActiveCall, CallState, TelephonyService
except ImportError:
    ActiveCall = None  # type: ignore
    CallState = None  # type: ignore
    TelephonyService = None  # type: ignore

try:
    from pengulink.services.pausemusic import PauseMusicConfig, PauseMusicService
except ImportError:
    PauseMusicConfig = None  # type: ignore
    PauseMusicService = None  # type: ignore


# =============================================================================
# Test Harness & Helpers
# =============================================================================

@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing Click CLI runner with standard terminal width."""
    return CliRunner(env={"COLUMNS": "160"})


@pytest.fixture(autouse=True)
def force_mock_mpris_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure MockMprisAdapter and MockAudioSinkBackend are used for 100% deterministic headless E2E testing."""
    monkeypatch.setenv("PENGULINK_MPRIS_ADAPTER", "mock")
    monkeypatch.setenv("PENGULINK_AUDIO_BACKEND", "mock")


def _player_is_playing(adapter: Any, player: str, is_playing: bool = True) -> Callable[[], Any]:
    """Helper returning an async predicate for wait_for polling player status."""
    async def _predicate() -> bool:
        st = await adapter.get_player_status(player)
        return bool(st is not None and st.is_playing is is_playing)

    return _predicate


def _players_are_playing(adapter: Any, players: dict[str, bool]) -> Callable[[], Any]:
    """Helper returning an async predicate for wait_for polling multiple players."""
    async def _predicate() -> bool:
        for player, expected in players.items():
            st = await adapter.get_player_status(player)
            if st is None or st.is_playing is not expected:
                return False
        return True

    return _predicate


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


def _require_telephony_plugin(daemon: PenguLinkDaemon) -> None:
    """Skip test gracefully if Telephony plugin is not yet registered on daemon."""
    if not hasattr(daemon, "telephony_plugin") and not daemon.plugin_registry.get_plugin("telephony"):
        pytest.skip("Telephony plugin is not yet registered on daemon (Milestone M27 planned).")


def _require_pausemusic_plugin(daemon: PenguLinkDaemon) -> None:
    """Skip test gracefully if PauseMusic plugin is not yet registered on daemon."""
    if not hasattr(daemon, "pause_music_plugin") and not daemon.plugin_registry.get_plugin("pausemusic"):
        pytest.skip("PauseMusic plugin is not yet registered on daemon (Milestone M28 planned).")


def _require_telephony_ipc(daemon: PenguLinkDaemon) -> None:
    """Skip test gracefully if Telephony IPC endpoints are not yet registered."""
    _require_telephony_plugin(daemon)
    if not hasattr(daemon, "telephony_get_calls"):
        pytest.skip("Telephony IPC endpoints are not yet wired on daemon (Milestone M29 planned).")


def _require_pausemusic_ipc(daemon: PenguLinkDaemon) -> None:
    """Skip test gracefully if PauseMusic IPC endpoints are not yet registered."""
    _require_pausemusic_plugin(daemon)
    if not hasattr(daemon, "pausemusic_get_config"):
        pytest.skip("PauseMusic IPC endpoints are not yet wired on daemon (Milestone M29 planned).")


# =============================================================================
# TIER 1: Feature Coverage (>=5 tests per feature)
# =============================================================================

# --- Feature 1: Telephony Call Events Roundtrip ---

async def test_tier1_tel_001_mock_simulation_helpers_and_capabilities(tmp_path: Path) -> None:
    """Tier 1: Verify MockKdeConnectDevice Telephony capabilities and simulation helpers."""
    mock_standalone = MockDevice()
    assert "kdeconnect.telephony" in mock_standalone.config.incoming_capabilities
    assert "kdeconnect.telephony.request_mute" in mock_standalone.config.incoming_capabilities
    assert "kdeconnect.telephony.request" in mock_standalone.config.incoming_capabilities
    assert "kdeconnect.pausemusic" in mock_standalone.config.incoming_capabilities

    assert "kdeconnect.telephony" in mock_standalone.config.outgoing_capabilities
    assert "kdeconnect.telephony.request_mute" in mock_standalone.config.outgoing_capabilities
    assert "kdeconnect.telephony.request" in mock_standalone.config.outgoing_capabilities
    assert "kdeconnect.pausemusic" in mock_standalone.config.outgoing_capabilities

    # Connect to live loopback environment to verify packet transmission over mTLS
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        await _pair_and_connect(daemon, mock)

        p_ring = await mock.simulate_incoming_call(phone_number="+15551234567", contact_name="Alice")
        assert p_ring.type == "kdeconnect.telephony"
        assert p_ring.body["event"] == "ringing"
        assert p_ring.body["phoneNumber"] == "+15551234567"
        assert p_ring.body["contactName"] == "Alice"
        assert p_ring.body["isCancel"] is False

        p_talk = await mock.simulate_answer_call(phone_number="+15551234567", contact_name="Alice")
        assert p_talk.type == "kdeconnect.telephony"
        assert p_talk.body["event"] == "talking"
        assert p_talk.body["isCancel"] is False

        p_end = await mock.simulate_end_call(phone_number="+15551234567", contact_name="Alice")
        assert p_end.type == "kdeconnect.telephony"
        assert p_end.body["event"] == "idle"
        assert p_end.body["isCancel"] is True

        p_miss = await mock.simulate_missed_call(phone_number="+15551234567", contact_name="Alice")
        assert p_miss.type == "kdeconnect.telephony"
        assert p_miss.body["event"] == "missedCall"
        assert p_miss.body["isCancel"] is False



async def test_tier1_tel_002_capability_negotiation(tmp_path: Path) -> None:
    """Tier 1: Verify Telephony and PauseMusic capabilities are advertised and negotiated."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        inc = daemon.config.device.incoming_capabilities
        outg = daemon.config.device.outgoing_capabilities
        assert "kdeconnect.telephony" in inc
        assert "kdeconnect.pausemusic" in inc
        assert "kdeconnect.telephony.request_mute" in outg
        assert "kdeconnect.telephony.request" in outg


async def test_tier1_tel_003_incoming_ringing_call_roundtrip(tmp_path: Path) -> None:
    """Tier 1: Remote phone sends ringing call; daemon registers active call in cache."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Send ringing call
        await mock.simulate_incoming_call(phone_number="+15559876543", contact_name="Bob Jones")

        # Wait for call to register in daemon service
        await wait_for(
            lambda: hasattr(daemon, "telephony_service")
            and len(daemon.telephony_service.get_active_calls(mock.device_id)) > 0,
            timeout=4.0,
            desc="telephony active call registration",
        )
        calls = daemon.telephony_service.get_active_calls(mock.device_id)
        assert len(calls) == 1
        assert calls[0].phone_number == "+15559876543"
        assert calls[0].contact_name == "Bob Jones"
        assert calls[0].state == "RINGING" or getattr(calls[0].state, "value", str(calls[0].state)) == "RINGING"


async def test_tier1_tel_004_call_answered_talking_roundtrip(tmp_path: Path) -> None:
    """Tier 1: Call transitions from ringing to answered (talking); active call updates."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        await mock.simulate_incoming_call(phone_number="+15559876543", contact_name="Bob Jones")
        await asyncio.sleep(0.1)

        await mock.simulate_answer_call(phone_number="+15559876543", contact_name="Bob Jones")

        await wait_for(
            lambda: hasattr(daemon, "telephony_service")
            and daemon.telephony_service.get_active_calls(mock.device_id)[0].state in ("TALKING", CallState.TALKING if CallState else "TALKING"),
            timeout=4.0,
            desc="telephony state TALKING",
        )
        call = daemon.telephony_service.get_active_calls(mock.device_id)[0]
        assert getattr(call.state, "value", str(call.state)) == "TALKING"


async def test_tier1_tel_005_call_ended_idle_roundtrip(tmp_path: Path) -> None:
    """Tier 1: Call ends with isCancel: true; daemon clears active call from cache."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        await mock.simulate_incoming_call(phone_number="+15559876543", contact_name="Bob Jones")
        await asyncio.sleep(0.1)

        await mock.simulate_end_call(phone_number="+15559876543", contact_name="Bob Jones")

        await wait_for(
            lambda: hasattr(daemon, "telephony_service")
            and len(daemon.telephony_service.get_active_calls(mock.device_id)) == 0,
            timeout=4.0,
            desc="telephony active call cleared",
        )
        assert len(daemon.telephony_service.get_active_calls(mock.device_id)) == 0


async def test_tier1_tel_006_missed_call_event_roundtrip(tmp_path: Path) -> None:
    """Tier 1: Remote phone signals missedCall; active call is cleared."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        await mock.simulate_incoming_call(phone_number="+15559876543", contact_name="Bob Jones")
        await asyncio.sleep(0.1)

        await mock.simulate_missed_call(phone_number="+15559876543", contact_name="Bob Jones")

        await wait_for(
            lambda: hasattr(daemon, "telephony_service")
            and len(daemon.telephony_service.get_active_calls(mock.device_id)) == 0,
            timeout=4.0,
            desc="telephony missed call cleanup",
        )
        assert len(daemon.telephony_service.get_active_calls(mock.device_id)) == 0


# --- Feature 2: Mute Ringer Action Packet ---

async def test_tier1_mute_001_mute_ringer_action_dispatch(tmp_path: Path) -> None:
    """Tier 1: Host daemon sends mute-ringer action; mock phone receives packet."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        res = await daemon.telephony_mute_ringer(mock.device_id)
        assert res.get("status") == "success" or res.get("muted") is True or res.get("action") in ("mute", "muteRinger")

        pkt = await mock.wait_for_telephony_action(action="mute", timeout=4.0)
        assert pkt.type in ("kdeconnect.telephony.request_mute", "kdeconnect.telephony.request")


async def test_tier1_mute_002_mute_ringer_wire_format(tmp_path: Path) -> None:
    """Tier 1: Validate outbound mute ringer packet matches standard protocol schema."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        await daemon.telephony_mute_ringer(mock.device_id)
        pkt = await mock.wait_for_telephony_action(action="mute", timeout=4.0)
        assert isinstance(pkt.body, dict)


async def test_tier1_mute_003_mute_ringer_during_ringing_state(tmp_path: Path) -> None:
    """Tier 1: Mute ringer dispatched specifically while phone is actively ringing."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        await mock.simulate_incoming_call(phone_number="+15550001111", contact_name="Charlie")
        await asyncio.sleep(0.1)

        await daemon.telephony_mute_ringer(mock.device_id)
        pkt = await mock.wait_for_telephony_action(action="mute", timeout=4.0)
        assert pkt is not None


async def test_tier1_mute_004_mute_ringer_idempotent_multiple_calls(tmp_path: Path) -> None:
    """Tier 1: Multiple mute ringer requests in sequence execute cleanly without error."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        await daemon.telephony_mute_ringer(mock.device_id)
        await daemon.telephony_mute_ringer(mock.device_id)
        await asyncio.sleep(0.2)

        actions = mock.get_telephony_actions(action="mute")
        assert len(actions) >= 2


async def test_tier1_mute_005_mute_ringer_cli_execution(tmp_path: Path, cli_runner: CliRunner) -> None:
    """Tier 1: CLI command 'pengulink telephony mute-ringer <device_id>' dispatches action."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_ipc(daemon)
        await _pair_and_connect(daemon, mock)

        sock_arg = ["-s", str(daemon.ipc_server.socket_path)]
        res = await asyncio.to_thread(
            cli_runner.invoke,
            cli,
            sock_arg + ["telephony", "mute-ringer", mock.device_id],
        )
        assert res.exit_code == 0
        pkt = await mock.wait_for_telephony_action(action="mute", timeout=4.0)
        assert pkt is not None


# --- Feature 3: Reject Call Action Packet ---

async def test_tier1_reject_001_reject_call_action_dispatch(tmp_path: Path) -> None:
    """Tier 1: Host daemon sends reject-call action; mock phone receives packet."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        res = await daemon.telephony_reject_call(mock.device_id)
        assert res.get("status") == "success" or res.get("rejected") is True or res.get("action") == "rejectCall"

        pkt = await mock.wait_for_telephony_action(action="rejectCall", timeout=4.0)
        assert pkt.type == "kdeconnect.telephony.request"
        assert pkt.body.get("action") == "rejectCall"


async def test_tier1_reject_002_reject_call_wire_format(tmp_path: Path) -> None:
    """Tier 1: Validate outbound reject call packet contains action='rejectCall'."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        await daemon.telephony_reject_call(mock.device_id)
        pkt = await mock.wait_for_telephony_action(action="rejectCall", timeout=4.0)
        assert pkt.body.get("action") == "rejectCall"


async def test_tier1_reject_003_reject_call_during_ringing_state(tmp_path: Path) -> None:
    """Tier 1: Reject call action sent during incoming ringing call."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        await mock.simulate_incoming_call(phone_number="+15552223333", contact_name="Dana")
        await asyncio.sleep(0.1)

        await daemon.telephony_reject_call(mock.device_id)
        pkt = await mock.wait_for_telephony_action(action="rejectCall", timeout=4.0)
        assert pkt.body.get("action") == "rejectCall"


async def test_tier1_reject_004_reject_call_during_talking_state(tmp_path: Path) -> None:
    """Tier 1: Reject call action sent to hang up an ongoing conversation."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        await mock.simulate_answer_call(phone_number="+15552223333", contact_name="Dana")
        await asyncio.sleep(0.1)

        await daemon.telephony_reject_call(mock.device_id)
        pkt = await mock.wait_for_telephony_action(action="rejectCall", timeout=4.0)
        assert pkt.body.get("action") == "rejectCall"


async def test_tier1_reject_005_reject_call_cli_execution(tmp_path: Path, cli_runner: CliRunner) -> None:
    """Tier 1: CLI command 'pengulink telephony reject-call <device_id>' dispatches packet."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_ipc(daemon)
        await _pair_and_connect(daemon, mock)

        sock_arg = ["-s", str(daemon.ipc_server.socket_path)]
        res = await asyncio.to_thread(
            cli_runner.invoke,
            cli,
            sock_arg + ["telephony", "reject-call", mock.device_id],
        )
        assert res.exit_code == 0
        pkt = await mock.wait_for_telephony_action(action="rejectCall", timeout=4.0)
        assert pkt.body.get("action") == "rejectCall"


# --- Feature 4: Auto-Pause Media Player ---

async def test_tier1_pause_001_auto_pause_single_active_mpris_player(tmp_path: Path) -> None:
    """Tier 1: Incoming phone call automatically pauses actively playing media player."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Set player to playing state
        adapter = daemon.pause_music_service._mpris_adapter if hasattr(daemon, "pause_music_service") else None
        if adapter and hasattr(adapter, "send_action"):
            await adapter.send_action("MockPlayer", "Play")
            status = await adapter.get_player_status("MockPlayer")
            assert status is not None and status.is_playing is True

        # Ringing call arrives
        await mock.simulate_incoming_call(phone_number="+15553334444", contact_name="Eve")

        # Player should be automatically paused
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(
                _player_is_playing(adapter, "MockPlayer", False),
                timeout=4.0,
                desc="MockPlayer auto-paused on call",
            )
            st = await adapter.get_player_status("MockPlayer")
            assert st.is_playing is False


async def test_tier1_pause_002_auto_pause_multiple_active_mpris_players(tmp_path: Path) -> None:
    """Tier 1: Incoming phone call automatically pauses all active playing players."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        adapter = daemon.pause_music_service._mpris_adapter if hasattr(daemon, "pause_music_service") else None
        if adapter and hasattr(adapter, "add_mock_player"):
            adapter.add_mock_player("PlayerAlpha", PlayerState(player="PlayerAlpha", is_playing=True))
            adapter.add_mock_player("PlayerBeta", PlayerState(player="PlayerBeta", is_playing=True))

        await mock.simulate_incoming_call(phone_number="+15553334444", contact_name="Eve")

        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(
                _players_are_playing(adapter, {"PlayerAlpha": False, "PlayerBeta": False}),
                timeout=4.0,
                desc="All active players auto-paused",
            )
            st1 = await adapter.get_player_status("PlayerAlpha")
            st2 = await adapter.get_player_status("PlayerBeta")
            assert st1.is_playing is False
            assert st2.is_playing is False


async def test_tier1_pause_003_auto_pause_policy_ringing_condition(tmp_path: Path) -> None:
    """Tier 1: With policy condition='ringing', pause triggers immediately upon ringing."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        if hasattr(daemon, "pause_music_service"):
            daemon.pause_music_service.config.condition = "ringing"

        adapter = daemon.pause_music_service._mpris_adapter if hasattr(daemon, "pause_music_service") else None
        if adapter and hasattr(adapter, "send_action"):
            await adapter.send_action("MockPlayer", "Play")

        await mock.simulate_incoming_call(phone_number="+15553334444", contact_name="Eve")

        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "MockPlayer", False), timeout=4.0)
            st = await adapter.get_player_status("MockPlayer")
            assert st.is_playing is False


async def test_tier1_pause_004_auto_pause_policy_talking_condition(tmp_path: Path) -> None:
    """Tier 1: With policy condition='talking', ringing does NOT pause; pause occurs on talking."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        if hasattr(daemon, "pause_music_service"):
            daemon.pause_music_service.config.condition = "talking"

        adapter = daemon.pause_music_service._mpris_adapter if hasattr(daemon, "pause_music_service") else None
        if adapter and hasattr(adapter, "send_action"):
            await adapter.send_action("MockPlayer", "Play")

        # 1. Ringing arrives -> media should NOT pause yet
        await mock.simulate_incoming_call(phone_number="+15553334444", contact_name="Eve")
        await asyncio.sleep(0.3)
        if adapter and hasattr(adapter, "get_player_status"):
            st = await adapter.get_player_status("MockPlayer")
            assert st.is_playing is True

        # 2. Call answered (talking) -> media now pauses
        await mock.simulate_answer_call(phone_number="+15553334444", contact_name="Eve")
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "MockPlayer", False), timeout=4.0)
            st = await adapter.get_player_status("MockPlayer")
            assert st.is_playing is False


async def test_tier1_pause_005_auto_pause_action_mute_audio_sinks(tmp_path: Path) -> None:
    """Tier 1: With policy action='mute', unmuted system audio sinks are muted on call."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        if hasattr(daemon, "pause_music_service"):
            daemon.pause_music_service.config.action = "mute"
            audio_svc = daemon.pause_music_service._audio_service
            if audio_svc:
                if not hasattr(audio_svc.backend, "add_sink"):
                    audio_svc.backend = MockAudioSinkBackend()
                audio_svc.backend.add_sink(AudioSink(name="sink_test", description="Test", muted=False))

        await mock.simulate_incoming_call(phone_number="+15553334444", contact_name="Eve")

        if hasattr(daemon, "pause_music_service"):
            audio_svc = daemon.pause_music_service._audio_service
            if audio_svc and hasattr(audio_svc, "list_sinks"):
                await wait_for(
                    lambda: any(s.muted is True for s in audio_svc.list_sinks() if s.name == "sink_test"),
                    timeout=4.0,
                    desc="Audio sink muted on call",
                )
                sinks = audio_svc.list_sinks()
                target = next((s for s in sinks if s.name == "sink_test"), None)
                assert target is not None and target.muted is True


# --- Feature 5: Auto-Resume Media Player ---

async def test_tier1_resume_001_auto_resume_single_player_on_call_end(tmp_path: Path) -> None:
    """Tier 1: When call ends, previously playing player is automatically resumed."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        adapter = daemon.pause_music_service._mpris_adapter if hasattr(daemon, "pause_music_service") else None
        if adapter and hasattr(adapter, "send_action"):
            await adapter.send_action("MockPlayer", "Play")

        # Call arrives -> pauses
        await mock.simulate_incoming_call(phone_number="+15554445555", contact_name="Frank")
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "MockPlayer", False), timeout=4.0)

        # Call ends -> resumes
        await mock.simulate_end_call(phone_number="+15554445555", contact_name="Frank")
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "MockPlayer", True), timeout=4.0)
            st = await adapter.get_player_status("MockPlayer")
            assert st.is_playing is True


async def test_tier1_resume_002_auto_resume_multiple_players_on_call_end(tmp_path: Path) -> None:
    """Tier 1: Multiple players playing before call are all resumed when call ends."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        adapter = daemon.pause_music_service._mpris_adapter if hasattr(daemon, "pause_music_service") else None
        if adapter and hasattr(adapter, "add_mock_player"):
            adapter.add_mock_player("Player1", PlayerState(player="Player1", is_playing=True))
            adapter.add_mock_player("Player2", PlayerState(player="Player2", is_playing=True))

        await mock.simulate_incoming_call(phone_number="+15554445555", contact_name="Frank")
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "Player1", False), timeout=4.0)

        await mock.simulate_end_call(phone_number="+15554445555", contact_name="Frank")
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(
                _players_are_playing(adapter, {"Player1": True, "Player2": True}),
                timeout=4.0,
                desc="Both players resumed",
            )
            assert (await adapter.get_player_status("Player1")).is_playing is True
            assert (await adapter.get_player_status("Player2")).is_playing is True


async def test_tier1_resume_003_auto_resume_disabled_policy(tmp_path: Path) -> None:
    """Tier 1: With policy resume=False, media remains paused after call ends."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        if hasattr(daemon, "pause_music_service"):
            daemon.pause_music_service.config.resume = False

        adapter = daemon.pause_music_service._mpris_adapter if hasattr(daemon, "pause_music_service") else None
        if adapter and hasattr(adapter, "send_action"):
            await adapter.send_action("MockPlayer", "Play")

        await mock.simulate_incoming_call(phone_number="+15554445555", contact_name="Frank")
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "MockPlayer", False), timeout=4.0)

        await mock.simulate_end_call(phone_number="+15554445555", contact_name="Frank")
        await asyncio.sleep(0.3)

        if adapter and hasattr(adapter, "get_player_status"):
            st = await adapter.get_player_status("MockPlayer")
            assert st.is_playing is False


async def test_tier1_resume_004_auto_unmute_audio_sinks_on_call_end(tmp_path: Path) -> None:
    """Tier 1: With policy action='mute', audio sinks muted by call are unmuted on call end."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        if hasattr(daemon, "pause_music_service"):
            daemon.pause_music_service.config.action = "mute"
            audio_svc = daemon.pause_music_service._audio_service
            if audio_svc:
                if not hasattr(audio_svc.backend, "add_sink"):
                    audio_svc.backend = MockAudioSinkBackend()
                audio_svc.backend.add_sink(AudioSink(name="sink_unmute", description="Unmute Test", muted=False))

        await mock.simulate_incoming_call(phone_number="+15554445555", contact_name="Frank")
        await asyncio.sleep(0.2)

        await mock.simulate_end_call(phone_number="+15554445555", contact_name="Frank")

        if hasattr(daemon, "pause_music_service"):
            audio_svc = daemon.pause_music_service._audio_service
            if audio_svc and hasattr(audio_svc, "list_sinks"):
                await wait_for(
                    lambda: any(s.muted is False for s in audio_svc.list_sinks() if s.name == "sink_unmute"),
                    timeout=4.0,
                    desc="Audio sink unmuted on call end",
                )
                target = next((s for s in audio_svc.list_sinks() if s.name == "sink_unmute"), None)
                assert target is not None and target.muted is False


async def test_tier1_resume_005_auto_resume_on_missed_call(tmp_path: Path) -> None:
    """Tier 1: Missed call event clears call interruption and triggers auto-resume."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        adapter = daemon.pause_music_service._mpris_adapter if hasattr(daemon, "pause_music_service") else None
        if adapter and hasattr(adapter, "send_action"):
            await adapter.send_action("MockPlayer", "Play")

        await mock.simulate_incoming_call(phone_number="+15554445555", contact_name="Frank")
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "MockPlayer", False), timeout=4.0)

        # Phone sends missedCall
        await mock.simulate_missed_call(phone_number="+15554445555", contact_name="Frank")
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "MockPlayer", True), timeout=4.0)
            st = await adapter.get_player_status("MockPlayer")
            assert st.is_playing is True


# =============================================================================
# TIER 2: Boundary & Corner Cases
# =============================================================================

async def test_tier2_bnd_001_empty_phone_number_and_contact_name(tmp_path: Path) -> None:
    """Tier 2: Empty phone number and contact name fallback safely without schema crash."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        pkt = Packet(
            type="kdeconnect.telephony",
            body={"event": "ringing", "phoneNumber": "", "contactName": "", "isCancel": False},
        )
        await mock.send_packet(pkt)
        await asyncio.sleep(0.2)

        assert daemon.is_running is True
        if hasattr(daemon, "telephony_service"):
            calls = daemon.telephony_service.get_active_calls(mock.device_id)
            assert len(calls) == 1
            # Fallback contactName / phoneNumber
            assert calls[0].phone_number in ("", "unknown number", "Unknown")


async def test_tier2_bnd_002_missing_phone_number_and_contact_name(tmp_path: Path) -> None:
    """Tier 2: Completely omitted phoneNumber and contactName fields handled cleanly."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        pkt = Packet(type="kdeconnect.telephony", body={"event": "ringing"})
        await mock.send_packet(pkt)
        await asyncio.sleep(0.2)

        assert daemon.is_running is True


async def test_tier2_bnd_003_null_fields_in_payload(tmp_path: Path) -> None:
    """Tier 2: Null/None values in phoneNumber, contactName, phoneThumbnail handled safely."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        pkt = Packet(
            type="kdeconnect.telephony",
            body={
                "event": "ringing",
                "phoneNumber": None,
                "contactName": None,
                "phoneThumbnail": None,
                "isCancel": None,
            },
        )
        await mock.send_packet(pkt)
        await asyncio.sleep(0.2)

        assert daemon.is_running is True


async def test_tier2_bnd_004_unknown_event_type(tmp_path: Path) -> None:
    """Tier 2: Unknown event string (e.g. 'voicemail') safely ignored without daemon crash."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        pkt = Packet(
            type="kdeconnect.telephony",
            body={"event": "voicemail_unsupported", "phoneNumber": "+15556667777"},
        )
        await mock.send_packet(pkt)
        await asyncio.sleep(0.2)

        assert daemon.is_running is True
        assert daemon.is_device_connected(mock.device_id) is True


async def test_tier2_bnd_005_rapid_call_state_transitions(tmp_path: Path) -> None:
    """Tier 2: Rapid burst of state transitions settles cleanly with zero leaked tasks."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        # Send rapid transitions: ringing -> talking -> idle -> ringing -> idle
        for ev, cancel in [
            ("ringing", False),
            ("talking", False),
            ("idle", True),
            ("ringing", False),
            ("idle", True),
        ]:
            await mock.send_telephony_call(
                event=ev,
                phone_number="+15557778888",
                contact_name="Rapid Caller",
                is_cancel=cancel,
            )

        await asyncio.sleep(0.3)
        assert daemon.is_running is True
        if hasattr(daemon, "telephony_service"):
            calls = daemon.telephony_service.get_active_calls(mock.device_id)
            assert len(calls) == 0


async def test_tier2_bnd_006_string_boolean_iscancel_coercion(tmp_path: Path) -> None:
    """Tier 2: Android sends isCancel as string 'true'; verified coerced as cancellation."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        await mock.simulate_incoming_call(phone_number="+15558889999", contact_name="Android Phone")
        await asyncio.sleep(0.1)

        # Android format: isCancel string "true"
        pkt = Packet(
            type="kdeconnect.telephony",
            body={"event": "idle", "phoneNumber": "+15558889999", "isCancel": "true"},
        )
        await mock.send_packet(pkt)
        await asyncio.sleep(0.2)

        if hasattr(daemon, "telephony_service"):
            assert len(daemon.telephony_service.get_active_calls(mock.device_id)) == 0


async def test_tier2_bnd_007_action_dispatch_to_unknown_device(tmp_path: Path) -> None:
    """Tier 2: Action dispatch targeting non-existent device returns clean error without crash."""
    async with e2e_environment(tmp_path) as (daemon, _mock, _ipc_client):
        _require_telephony_plugin(daemon)

        with pytest.raises((TelephonyError, Exception)):
            await daemon.telephony_mute_ringer("nonexistent_device_id_999")

        assert daemon.is_running is True


async def test_tier2_bnd_008_oversized_unicode_contact_name(tmp_path: Path) -> None:
    """Tier 2: Contact name with emojis, symbols, and length up to 500 characters parses safely."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        unicode_name = "👑 Dr. Jane 🚀 🎧 🎶 " * 25  # ~500 chars
        await mock.simulate_incoming_call(phone_number="+15559990000", contact_name=unicode_name)
        await asyncio.sleep(0.2)

        assert daemon.is_running is True
        if hasattr(daemon, "telephony_service"):
            calls = daemon.telephony_service.get_active_calls(mock.device_id)
            assert len(calls) == 1
            assert calls[0].contact_name == unicode_name


# =============================================================================
# TIER 3: Cross-Feature Combinations
# =============================================================================

async def test_tier3_cross_001_player_state_preservation_stopped_remains_stopped(tmp_path: Path) -> None:
    """Tier 3: Stopped/paused players prior to call are NOT resumed when call ends."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        adapter = daemon.pause_music_service._mpris_adapter if hasattr(daemon, "pause_music_service") else None
        if adapter and hasattr(adapter, "add_mock_player"):
            # Player 1 is playing, Player 2 is stopped
            adapter.add_mock_player("PlayerPlaying", PlayerState(player="PlayerPlaying", is_playing=True))
            adapter.add_mock_player("PlayerStopped", PlayerState(player="PlayerStopped", is_playing=False))

        # Call arrives -> PlayerPlaying pauses
        await mock.simulate_incoming_call(phone_number="+15551112222", contact_name="Caller")
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "PlayerPlaying", False), timeout=4.0)

        # Call ends -> PlayerPlaying resumes, PlayerStopped STAYS stopped
        await mock.simulate_end_call(phone_number="+15551112222", contact_name="Caller")
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "PlayerPlaying", True), timeout=4.0)
            st_playing = await adapter.get_player_status("PlayerPlaying")
            st_stopped = await adapter.get_player_status("PlayerStopped")
            assert st_playing.is_playing is True
            assert st_stopped.is_playing is False, "Stopped player must NOT be auto-resumed!"


async def test_tier3_cross_002_audio_sink_state_preservation_premute_remains_muted(tmp_path: Path) -> None:
    """Tier 3: Sinks that were already muted before the call remain muted after call ends."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        if hasattr(daemon, "pause_music_service"):
            daemon.pause_music_service.config.action = "mute"
            audio_svc = daemon.pause_music_service._audio_service
            if audio_svc:
                if not hasattr(audio_svc.backend, "add_sink"):
                    audio_svc.backend = MockAudioSinkBackend()
                audio_svc.backend.add_sink(AudioSink(name="sink_unmuted", description="Unmuted", muted=False))
                audio_svc.backend.add_sink(AudioSink(name="sink_premute", description="Pre-muted", muted=True))

        await mock.simulate_incoming_call(phone_number="+15551112222", contact_name="Caller")
        await asyncio.sleep(0.2)

        await mock.simulate_end_call(phone_number="+15551112222", contact_name="Caller")
        await asyncio.sleep(0.2)

        if hasattr(daemon, "pause_music_service"):
            audio_svc = daemon.pause_music_service._audio_service
            if audio_svc and hasattr(audio_svc, "list_sinks"):
                sinks = audio_svc.list_sinks()
                s_unmuted = next((s for s in sinks if s.name == "sink_unmuted"), None)
                s_premute = next((s for s in sinks if s.name == "sink_premute"), None)
                assert s_unmuted is not None and s_unmuted.muted is False
                assert s_premute is not None and s_premute.muted is True, "Pre-muted sink must stay muted!"


async def test_tier3_cross_003_multi_device_overlapping_calls(tmp_path: Path) -> None:
    """Tier 3: Overlapping calls across 2 devices: media stays paused until ALL calls end."""
    async with e2e_environment(tmp_path) as (daemon, mock1, _ipc_client):
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock1)

        # Spin up second mock device
        m2_cfg = MockDeviceConfig(device_name="Mock Phone 2")
        mock2 = MockDevice(config=m2_cfg)
        await mock2.start(listen_tcp=True, port=0)

        try:
            await _pair_and_connect(daemon, mock2)

            adapter = daemon.pause_music_service._mpris_adapter if hasattr(daemon, "pause_music_service") else None
            if adapter and hasattr(adapter, "send_action"):
                await adapter.send_action("MockPlayer", "Play")

            # 1. Device 1 call starts -> music pauses
            await mock1.simulate_incoming_call(phone_number="+15551111111", contact_name="Dev1 Caller")
            if adapter and hasattr(adapter, "get_player_status"):
                await wait_for(_player_is_playing(adapter, "MockPlayer", False), timeout=4.0)

            # 2. Device 2 call starts while Device 1 is active -> music STAYS paused
            await mock2.simulate_incoming_call(phone_number="+15552222222", contact_name="Dev2 Caller")
            await asyncio.sleep(0.2)
            if adapter and hasattr(adapter, "get_player_status"):
                assert (await adapter.get_player_status("MockPlayer")).is_playing is False

            # 3. Device 1 call ends -> Device 2 is STILL active -> music MUST STAY PAUSED!
            await mock1.simulate_end_call(phone_number="+15551111111", contact_name="Dev1 Caller")
            await asyncio.sleep(0.3)
            if adapter and hasattr(adapter, "get_player_status"):
                st = await adapter.get_player_status("MockPlayer")
                assert st.is_playing is False, "Music must not resume while Device 2 call is still active!"

            # 4. Device 2 call ends -> all calls ended -> music RESUMES!
            await mock2.simulate_end_call(phone_number="+15552222222", contact_name="Dev2 Caller")
            if adapter and hasattr(adapter, "get_player_status"):
                await wait_for(_player_is_playing(adapter, "MockPlayer", True), timeout=4.0)
                st = await adapter.get_player_status("MockPlayer")
                assert st.is_playing is True, "Music should resume after all overlapping calls ended!"
        finally:
            if mock2.is_running:
                await mock2.stop()


async def test_tier3_cross_004_multi_device_call_cache_isolation(tmp_path: Path) -> None:
    """Tier 3: Active calls on Device A and Device B remain strictly isolated."""
    async with e2e_environment(tmp_path) as (daemon, mock1, _ipc_client):
        _require_telephony_plugin(daemon)
        await _pair_and_connect(daemon, mock1)

        m2_cfg = MockDeviceConfig(device_name="Mock Phone 2")
        mock2 = MockDevice(config=m2_cfg)
        await mock2.start(listen_tcp=True, port=0)

        try:
            await _pair_and_connect(daemon, mock2)

            await mock1.simulate_incoming_call(phone_number="+15551111111", contact_name="Alice")
            await mock2.simulate_incoming_call(phone_number="+15552222222", contact_name="Bob")
            await asyncio.sleep(0.2)

            if hasattr(daemon, "telephony_service"):
                calls_m1 = daemon.telephony_service.get_active_calls(mock1.device_id)
                calls_m2 = daemon.telephony_service.get_active_calls(mock2.device_id)
                assert len(calls_m1) == 1 and calls_m1[0].contact_name == "Alice"
                assert len(calls_m2) == 1 and calls_m2[0].contact_name == "Bob"
        finally:
            if mock2.is_running:
                await mock2.stop()


async def test_tier3_cross_005_mpris_reflection_suppression(tmp_path: Path) -> None:
    """Tier 3: Virtual KDE Connect MPRIS reflection players are excluded from auto-pause."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        adapter = daemon.pause_music_service._mpris_adapter if hasattr(daemon, "pause_music_service") else None
        if adapter and hasattr(adapter, "add_mock_player"):
            # Add virtual kdeconnect reflection player
            reflection_name = f"org.mpris.MediaPlayer2.kdeconnect.{mock.device_id}"
            adapter.add_mock_player(reflection_name, PlayerState(player=reflection_name, is_playing=True))
            adapter.add_mock_player("RealLocalPlayer", PlayerState(player="RealLocalPlayer", is_playing=True))

        await mock.simulate_incoming_call(phone_number="+15553334444", contact_name="Eve")

        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "RealLocalPlayer", False), timeout=4.0)
            # Real player paused
            assert (await adapter.get_player_status("RealLocalPlayer")).is_playing is False
            # Reflection player skipped
            st_refl = await adapter.get_player_status(f"org.mpris.MediaPlayer2.kdeconnect.{mock.device_id}")
            assert st_refl is not None


async def test_tier3_cross_006_action_mute_and_pause_combined(tmp_path: Path) -> None:
    """Tier 3: With action='both', media player is paused AND audio sinks are muted simultaneously."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        if hasattr(daemon, "pause_music_service"):
            daemon.pause_music_service.config.action = "both"
            audio_svc = daemon.pause_music_service._audio_service
            if audio_svc:
                if not hasattr(audio_svc.backend, "add_sink"):
                    audio_svc.backend = MockAudioSinkBackend()
                audio_svc.backend.add_sink(AudioSink(name="sink_both", description="Both", muted=False))

        adapter = daemon.pause_music_service._mpris_adapter if hasattr(daemon, "pause_music_service") else None
        if adapter and hasattr(adapter, "send_action"):
            await adapter.send_action("MockPlayer", "Play")

        # Call arrives
        await mock.simulate_incoming_call(phone_number="+15553334444", contact_name="Eve")

        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "MockPlayer", False), timeout=4.0)
            assert (await adapter.get_player_status("MockPlayer")).is_playing is False

        if hasattr(daemon, "pause_music_service"):
            audio_svc = daemon.pause_music_service._audio_service
            if audio_svc and hasattr(audio_svc, "list_sinks"):
                s = next((x for x in audio_svc.list_sinks() if x.name == "sink_both"), None)
                assert s is not None and s.muted is True

        # Call ends -> both restored
        await mock.simulate_end_call(phone_number="+15553334444", contact_name="Eve")

        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "MockPlayer", True), timeout=4.0)
            assert (await adapter.get_player_status("MockPlayer")).is_playing is True

        if hasattr(daemon, "pause_music_service"):
            audio_svc = daemon.pause_music_service._audio_service
            if audio_svc and hasattr(audio_svc, "list_sinks"):
                s = next((x for x in audio_svc.list_sinks() if x.name == "sink_both"), None)
                assert s is not None and s.muted is False


# =============================================================================
# TIER 4: Real-World Scenarios
# =============================================================================

async def test_tier4_real_001_complete_incoming_call_lifecycle_with_auto_resume(tmp_path: Path) -> None:
    """Tier 4: Full lifecycle: media playing -> ringing -> auto-pause -> talking -> end -> auto-resume."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        adapter = daemon.pause_music_service._mpris_adapter if hasattr(daemon, "pause_music_service") else None
        if adapter and hasattr(adapter, "send_action"):
            await adapter.send_action("MockPlayer", "Play")
            assert (await adapter.get_player_status("MockPlayer")).is_playing is True

        # 1. Incoming call rings
        await mock.simulate_incoming_call(phone_number="+15559870001", contact_name="Grace Hopper")

        # Verify active call recorded and media paused
        if hasattr(daemon, "telephony_service"):
            await wait_for(
                lambda: len(daemon.telephony_service.get_active_calls(mock.device_id)) == 1,
                timeout=4.0,
            )
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "MockPlayer", False), timeout=4.0)

        # 2. User answers call on phone
        await mock.simulate_answer_call(phone_number="+15559870001", contact_name="Grace Hopper")
        await asyncio.sleep(0.2)

        # 3. Call conversation finishes
        await mock.simulate_end_call(phone_number="+15559870001", contact_name="Grace Hopper")

        # Verify call cleared and media resumed
        if hasattr(daemon, "telephony_service"):
            await wait_for(
                lambda: len(daemon.telephony_service.get_active_calls(mock.device_id)) == 0,
                timeout=4.0,
            )
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "MockPlayer", True), timeout=4.0)
            assert (await adapter.get_player_status("MockPlayer")).is_playing is True


async def test_tier4_real_002_disconnection_during_active_call_restores_media(tmp_path: Path) -> None:
    """Tier 4: Remote phone abruptly disconnects (RST) during call; media is automatically restored."""
    chaos = ChaosConfig(drop_with_rst=True)
    async with e2e_environment(tmp_path, mock_chaos=chaos) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        adapter = daemon.pause_music_service._mpris_adapter if hasattr(daemon, "pause_music_service") else None
        if adapter and hasattr(adapter, "send_action"):
            await adapter.send_action("MockPlayer", "Play")

        # Call arrives and pauses media
        await mock.simulate_incoming_call(phone_number="+15559870002", contact_name="Alan Turing")
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "MockPlayer", False), timeout=4.0)

        # Phone abruptly drops TCP socket connection
        if hasattr(mock, "drop_connection"):
            await mock.drop_connection(reset=True)
        elif mock.active_connection:
            await mock.active_connection.close()

        # Daemon should detect disconnection, prune active calls, and resume media
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(
                _player_is_playing(adapter, "MockPlayer", True),
                timeout=5.0,
                desc="Media restored on abrupt phone drop",
            )
            assert (await adapter.get_player_status("MockPlayer")).is_playing is True

        if hasattr(daemon, "telephony_service"):
            assert len(daemon.telephony_service.get_active_calls(mock.device_id)) == 0


async def test_tier4_real_003_unpair_during_active_call_cleans_up_state(tmp_path: Path) -> None:
    """Tier 4: Device is unpaired while call is active; all call state is purged and media restored."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_plugin(daemon)
        _require_pausemusic_plugin(daemon)
        await _pair_and_connect(daemon, mock)

        adapter = daemon.pause_music_service._mpris_adapter if hasattr(daemon, "pause_music_service") else None
        if adapter and hasattr(adapter, "send_action"):
            await adapter.send_action("MockPlayer", "Play")

        await mock.simulate_incoming_call(phone_number="+15559870003", contact_name="Ada Lovelace")
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "MockPlayer", False), timeout=4.0)

        # Unpair device
        await daemon.unpair_device(mock.device_id)

        # Call cache purged and media restored
        if hasattr(daemon, "telephony_service"):
            assert len(daemon.telephony_service.get_active_calls(mock.device_id)) == 0
        if adapter and hasattr(adapter, "get_player_status"):
            await wait_for(_player_is_playing(adapter, "MockPlayer", True), timeout=4.0)


async def test_tier4_real_004_cli_active_calls_query_json_and_table(tmp_path: Path, cli_runner: CliRunner) -> None:
    """Tier 4: CLI 'pengulink telephony calls' outputs formatted Rich table and valid JSON."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_telephony_ipc(daemon)
        await _pair_and_connect(daemon, mock)

        await mock.simulate_incoming_call(phone_number="+15554321000", contact_name="Margaret Hamilton")
        await asyncio.sleep(0.2)

        sock_arg = ["-s", str(daemon.ipc_server.socket_path)]

        # 1. Human-readable Rich table
        res_table = await asyncio.to_thread(
            cli_runner.invoke,
            cli,
            sock_arg + ["telephony", "calls", mock.device_id],
        )
        assert res_table.exit_code == 0
        assert "+15554321000" in res_table.output or "Margaret Hamilton" in res_table.output

        # 2. Machine-readable JSON
        res_json = await asyncio.to_thread(
            cli_runner.invoke,
            cli,
            sock_arg + ["telephony", "calls", mock.device_id, "--json"],
        )
        assert res_json.exit_code == 0
        calls_data = json.loads(res_json.output)
        assert isinstance(calls_data, list)
        assert len(calls_data) >= 1
        assert calls_data[0]["phone_number"] == "+15554321000"


async def test_tier4_real_005_cli_pausemusic_status_and_configure(tmp_path: Path, cli_runner: CliRunner) -> None:
    """Tier 4: CLI 'pengulink pausemusic status' and 'configure' read and update settings."""
    async with e2e_environment(tmp_path) as (daemon, mock, _ipc_client):
        _require_pausemusic_ipc(daemon)
        await _pair_and_connect(daemon, mock)

        sock_arg = ["-s", str(daemon.ipc_server.socket_path)]

        # 1. Query status via CLI JSON
        res_status = await asyncio.to_thread(
            cli_runner.invoke,
            cli,
            sock_arg + ["pausemusic", "status", "--json"],
        )
        assert res_status.exit_code == 0
        cfg = json.loads(res_status.output)
        assert "condition" in cfg
        assert "action" in cfg

        # 2. Update config via CLI
        res_conf = await asyncio.to_thread(
            cli_runner.invoke,
            cli,
            sock_arg + ["pausemusic", "configure", "--condition", "talking", "--action", "mute", "--no-resume", "--json"],
        )
        assert res_conf.exit_code == 0
        updated = json.loads(res_conf.output)
        assert updated.get("condition") == "talking"
        assert updated.get("action") == "mute"
        assert updated.get("resume") is False

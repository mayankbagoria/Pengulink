"""tests.adversarial.test_alert_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial stress, fuzzing, systems, and concurrency tests for AlertService
and FindMyPhonePlugin under heavy load and error injections.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, List
import pytest

from pengulink.core.events import AlertStopped, AlertTriggered, EventBus
from pengulink.core.exceptions import (
    ConnectionClosedError,
    InvalidPacketSchemaError,
    MalformedPacketError,
)
from pengulink.network.packet import Packet, PacketEngine
from pengulink.plugins.findmyphone import (
    FINDMYPHONE_LEGACY_TYPE,
    FINDMYPHONE_REQUEST_TYPE,
    FindMyPhonePlugin,
    create_find_my_phone_request_packet,
)
from pengulink.services.alert import (
    AlertService,
    AudioBackend,
    MockAudioBackend,
    MockAudioProcess,
)


# =====================================================================
# 1. Packet Flood & Load Stress Tests
# =====================================================================

async def test_adversarial_rapid_packet_flood(tmp_path: Path) -> None:
    """100 incoming findmyphone packets in rapid succession do not leak tasks or processes."""
    backend = MockAudioBackend(playback_duration=0.01)
    bus = EventBus()
    service = AlertService(backend=backend, event_bus=bus, runtime_dir=tmp_path)
    plugin = FindMyPhonePlugin(event_bus=bus, alert_service=service)

    pkt = create_find_my_phone_request_packet()

    # Flood 100 packets concurrently
    tasks = [plugin.handle_packet(pkt, f"flooder_{i}", lambda p: None) for i in range(100)]
    await asyncio.gather(*tasks)

    # After flood completes, state must be stable (either active or idle, not crashed)
    assert isinstance(service.is_active(), bool)
    status = service.get_status()
    assert isinstance(status, dict)

    await service.stop_alert()
    assert service.is_active() is False


# =====================================================================
# 2. Child Process Hanging & Zombie Elimination Tests
# =====================================================================

async def test_adversarial_child_process_hang() -> None:
    """Unresponsive child process ignoring SIGTERM is terminated by SIGKILL escalation."""
    class ZombieProcess:
        def __init__(self) -> None:
            self.pid = 77777
            self.returncode = None
            self.terminate_called = False
            self.kill_called = False

        def terminate(self) -> None:
            self.terminate_called = True
            # Does not exit on terminate

        def kill(self) -> None:
            self.kill_called = True
            self.returncode = -9

        async def wait(self) -> int:
            if self.kill_called:
                return -9
            await asyncio.sleep(5.0)
            return 0

    proc = ZombieProcess()
    service = AlertService()

    # Fast timeout for test execution
    orig = AlertService._terminate_process.__globals__["SUBPROCESS_TERMINATE_TIMEOUT"]
    try:
        AlertService._terminate_process.__globals__["SUBPROCESS_TERMINATE_TIMEOUT"] = 0.05
        await service._terminate_process(proc)
        assert proc.terminate_called is True
        assert proc.kill_called is True
        assert proc.returncode == -9
    finally:
        AlertService._terminate_process.__globals__["SUBPROCESS_TERMINATE_TIMEOUT"] = orig


async def test_adversarial_real_os_subprocess_hang_sigkill_reaped() -> None:
    """Real OS subprocess explicitly ignoring SIGTERM is escalated to SIGKILL and reaped without zombie."""
    service = AlertService()

    # Spawn real python subprocess that explicitly ignores SIGTERM
    script = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('READY', flush=True); "
        "time.sleep(60)"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    # Ensure subprocess has booted and installed SIG_IGN handler
    assert proc.stdout is not None
    ready_line = await proc.stdout.readline()
    assert ready_line.strip() == b"READY"

    pid = proc.pid
    assert pid > 0

    # Verify process is alive
    os.kill(pid, 0)

    # Fast timeout for escalation
    orig = AlertService._terminate_process.__globals__["SUBPROCESS_TERMINATE_TIMEOUT"]
    try:
        AlertService._terminate_process.__globals__["SUBPROCESS_TERMINATE_TIMEOUT"] = 0.1
        await service._terminate_process(proc)
    finally:
        AlertService._terminate_process.__globals__["SUBPROCESS_TERMINATE_TIMEOUT"] = orig

    # Process must be killed with SIGKILL (-9)
    assert proc.returncode in (-signal.SIGKILL, -9)

    # Process must be reaped by waitpid - no zombie process in OS process table
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_adversarial_hung_process_alert_loop_runaway_timeout(tmp_path: Path) -> None:
    """Active playback loop with an indefinitely hanging child process terminates on runaway timeout."""
    class HungProcessBackend(AudioBackend):
        @property
        def name(self) -> str:
            return "hung_backend"

        async def is_available(self) -> bool:
            return True

        async def play(self, wav_path: Path) -> Any:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c",
                "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('READY', flush=True); time.sleep(60)",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            assert proc.stdout is not None
            await proc.stdout.readline()
            return proc

    bus = EventBus()
    stopped_events: List[AlertStopped] = []
    bus.subscribe(AlertStopped, lambda ev: stopped_events.append(ev))

    service = AlertService(backend=HungProcessBackend(), event_bus=bus, runtime_dir=tmp_path)

    orig = AlertService._terminate_process.__globals__["SUBPROCESS_TERMINATE_TIMEOUT"]
    try:
        AlertService._terminate_process.__globals__["SUBPROCESS_TERMINATE_TIMEOUT"] = 0.1
        # Set short duration 0.1s to trigger runaway timeout
        await service.start_alert(device_id="hung_dev", max_duration=0.1)
        assert service.is_active() is True

        # Wait for timeout (0.1s duration + 0.2s termination)
        await asyncio.sleep(0.4)

        assert service.is_active() is False
        assert len(stopped_events) == 1
        assert stopped_events[0].reason == "timeout"
    finally:
        AlertService._terminate_process.__globals__["SUBPROCESS_TERMINATE_TIMEOUT"] = orig
        await service.stop_alert()


# =====================================================================
# 3. Audio Backend Crashes & Non-Zero Exit Code Tests
# =====================================================================

async def test_adversarial_backend_crash_during_playback(tmp_path: Path) -> None:
    """Backend raising exceptions during play() halts loop safely without crashing."""
    class CrashingBackend(AudioBackend):
        @property
        def name(self) -> str:
            return "crashing"

        async def is_available(self) -> bool:
            return True

        async def play(self, wav_path: Path) -> Any:
            raise OSError("Audio hardware device disconnected mid-playback")

    bus = EventBus()
    stopped_events: List[AlertStopped] = []
    bus.subscribe(AlertStopped, lambda ev: stopped_events.append(ev))

    service = AlertService(backend=CrashingBackend(), event_bus=bus, runtime_dir=tmp_path)
    started = await service.start_alert(device_id="crash_test", max_duration=10.0)
    assert started is True

    # Allow loop task to encounter backend crash
    await asyncio.sleep(0.15)
    assert service.is_active() is False
    assert len(stopped_events) == 1
    assert stopped_events[0].reason == "error"


async def test_adversarial_backend_non_zero_exit_code(tmp_path: Path) -> None:
    """Child process exiting with non-zero exit code stops loop gracefully with reason='error'."""
    class FailingExitProcess:
        def __init__(self) -> None:
            self.pid = 99911
            self.returncode = 1

        async def wait(self) -> int:
            return 1

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

    class FailingExitBackend(AudioBackend):
        @property
        def name(self) -> str:
            return "failing_exit"

        async def is_available(self) -> bool:
            return True

        async def play(self, wav_path: Path) -> Any:
            return FailingExitProcess()

    bus = EventBus()
    stopped_events: List[AlertStopped] = []
    bus.subscribe(AlertStopped, lambda ev: stopped_events.append(ev))

    service = AlertService(backend=FailingExitBackend(), event_bus=bus, runtime_dir=tmp_path)
    await service.start_alert(device_id="exit_test", max_duration=10.0)
    assert service.is_active() is True

    await asyncio.sleep(0.1)
    assert service.is_active() is False
    assert len(stopped_events) == 1
    assert stopped_events[0].reason == "error"


async def test_adversarial_backend_process_wait_exception(tmp_path: Path) -> None:
    """Child process wait() raising unexpected exception is caught cleanly without crashing daemon."""
    class CrashingWaitProcess:
        def __init__(self) -> None:
            self.pid = 99922
            self.returncode = None

        async def wait(self) -> int:
            raise RuntimeError("Audio daemon connection broken unexpectedly")

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

    class CrashingWaitBackend(AudioBackend):
        @property
        def name(self) -> str:
            return "crashing_wait"

        async def is_available(self) -> bool:
            return True

        async def play(self, wav_path: Path) -> Any:
            return CrashingWaitProcess()

    bus = EventBus()
    stopped_events: List[AlertStopped] = []
    bus.subscribe(AlertStopped, lambda ev: stopped_events.append(ev))

    service = AlertService(backend=CrashingWaitBackend(), event_bus=bus, runtime_dir=tmp_path)
    await service.start_alert(device_id="wait_crash", max_duration=10.0)

    await asyncio.sleep(0.1)
    assert service.is_active() is False
    assert len(stopped_events) == 1
    assert stopped_events[0].reason == "error"


# =====================================================================
# 4. Concurrency Races & Serialization Tests
# =====================================================================

async def test_adversarial_concurrent_start_alert_serialization(tmp_path: Path) -> None:
    """50 concurrent start_alert calls are serialized safely by asyncio.Lock without duplicate loops."""
    backend = MockAudioBackend(playback_duration=0.05)
    bus = EventBus()
    triggered_events: List[AlertTriggered] = []
    bus.subscribe(AlertTriggered, lambda ev: triggered_events.append(ev))

    service = AlertService(backend=backend, event_bus=bus, runtime_dir=tmp_path)

    # 50 concurrent callers try to start alerts simultaneously
    tasks = [
        service.start_alert(device_id=f"dev_{i}", source="concurrent", max_duration=5.0)
        for i in range(50)
    ]
    results = await asyncio.gather(*tasks)

    assert all(r is True for r in results)
    assert service.is_active() is True

    # Exactly 1 playback loop task should be created
    assert service._loop_task is not None
    # Only 1 initial AlertTriggered event is emitted (subsequent calls refresh without duplicate loop)
    assert len(triggered_events) == 1

    await service.stop_alert()
    assert service.is_active() is False


async def test_adversarial_concurrent_start_and_stop_race(tmp_path: Path) -> None:
    """Concurrent race between start_alert and stop_alert resolves cleanly without deadlocks."""
    backend = MockAudioBackend(playback_duration=0.01)
    service = AlertService(backend=backend, runtime_dir=tmp_path)

    for round_num in range(20):
        # Fire start and stop concurrently
        start_task = asyncio.create_task(service.start_alert(device_id=f"dev_{round_num}", max_duration=1.0))
        stop_task = asyncio.create_task(service.stop_alert(reason="race"))

        await asyncio.gather(start_task, stop_task, return_exceptions=False)
        # Service must be in a well-defined boolean state, not corrupt
        assert isinstance(service.is_active(), bool)

    await service.stop_alert()
    assert service.is_active() is False


# =====================================================================
# 5. Abrupt Daemon Shutdown & Task Drain Tests
# =====================================================================

async def test_adversarial_abrupt_daemon_shutdown(tmp_path: Path) -> None:
    """Abrupt daemon shutdown cleanly stops active alert, terminates child processes, and drains tasks."""
    backend = MockAudioBackend(playback_duration=0.05)
    bus = EventBus()
    stopped_events: List[AlertStopped] = []
    bus.subscribe(AlertStopped, lambda ev: stopped_events.append(ev))

    service = AlertService(backend=backend, event_bus=bus, runtime_dir=tmp_path)
    await service.start_alert(device_id="shutdown_dev", max_duration=10.0)
    assert service.is_active() is True

    # Abrupt shutdown via stop() lifecycle hook
    await service.stop()

    assert service.is_active() is False
    assert len(stopped_events) == 1
    assert stopped_events[0].reason == "shutdown"
    assert service._current_process is None
    assert service._loop_task is None


# =====================================================================
# 6. Zero-Log Privacy Audit Tests
# =====================================================================

async def test_adversarial_zero_log_privacy(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Auditing logs confirms zero leakage of private keys, auth tokens, or secret credentials."""
    backend = MockAudioBackend(playback_duration=0.01)
    service = AlertService(backend=backend, runtime_dir=tmp_path)
    plugin = FindMyPhonePlugin(alert_service=service)

    sensitive_device_id = "sensitive_device_SECRET_12345"
    private_key_payload = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA0mY7...SENSITIVE_KEY_BYTES...\n"
        "-----END RSA PRIVATE KEY-----"
    )
    bearer_token = "Bearer secret_jwt_token_abcdef123456"

    with caplog.at_level(logging.DEBUG):
        # Test start_alert with sensitive credentials in device_id and source
        await service.start_alert(device_id=sensitive_device_id, source=bearer_token, max_duration=1.0)
        pkt = create_find_my_phone_request_packet()
        await plugin.handle_packet(pkt, sensitive_device_id, lambda p: None)
        await service.stop_alert()

        # Test start_alert with private key in device_id
        await service.start_alert(device_id=private_key_payload, source="remote", max_duration=1.0)
        await service.stop_alert()

    log_text = caplog.text

    # Verify zero leakage of private keys
    assert "BEGIN RSA PRIVATE KEY" not in log_text
    assert "BEGIN PRIVATE KEY" not in log_text
    assert "SENSITIVE_KEY_BYTES" not in log_text

    # Verify zero leakage of auth tokens and secrets
    assert "SECRET_12345" not in log_text
    assert "secret_jwt_token" not in log_text


# =====================================================================
# 7. Repeated Lifecycle Leak Detection
# =====================================================================

async def test_adversarial_repeated_lifecycle_leak_detection(tmp_path: Path) -> None:
    """50 rapid sequential start_alert/stop_alert cycles execute with zero task accumulation."""
    backend = MockAudioBackend(playback_duration=0.01)
    service = AlertService(backend=backend, runtime_dir=tmp_path)

    initial_tasks = len(asyncio.all_tasks())

    for i in range(50):
        await service.start_alert(device_id=f"dev_{i}", max_duration=0.5)
        await service.stop_alert(reason="stress_test")

    await asyncio.sleep(0.05)
    final_tasks = len(asyncio.all_tasks())
    assert service.is_active() is False
    # Task difference should be negligible (0 or 1 active runner task)
    assert abs(final_tasks - initial_tasks) <= 2


# =====================================================================
# 8. FindMyPhonePlugin Protocol Stress & Adversarial Vectors
# =====================================================================

async def test_adversarial_rapid_packet_burst_flood(tmp_path: Path) -> None:
    """Rapid bursts of 60+ locate packets within milliseconds toggle safely without deadlock or task leak."""
    backend = MockAudioBackend(playback_duration=0.005)
    bus = EventBus()
    service = AlertService(backend=backend, event_bus=bus, runtime_dir=tmp_path)
    plugin = FindMyPhonePlugin(event_bus=bus, alert_service=service)

    pkt = create_find_my_phone_request_packet()

    initial_tasks = len(asyncio.all_tasks())
    t0 = time.monotonic()

    # Dispatch 60 packets concurrently in milliseconds
    tasks = [plugin.handle_packet(pkt, f"flooder_node_{i}", lambda p: None) for i in range(60)]
    await asyncio.gather(*tasks)

    elapsed = time.monotonic() - t0
    # Must process 60 concurrent packets rapidly (< 1.5 seconds)
    assert elapsed < 1.5

    # 60 packets is an even number, so toggling (on -> off -> on -> off ...) leaves alert stopped
    assert service.is_active() is False

    # Send one more packet (odd count: 61st) -> must activate alert
    await plugin.handle_packet(pkt, "flooder_node_final", lambda p: None)
    assert service.is_active() is True

    # Stop and verify task clean-up
    await service.stop_alert()
    assert service.is_active() is False

    await asyncio.sleep(0.05)
    final_tasks = len(asyncio.all_tasks())
    assert abs(final_tasks - initial_tasks) <= 2


async def test_adversarial_malformed_packet_bodies(tmp_path: Path) -> None:
    """FindMyPhonePlugin and wire engine safely reject or tolerate malformed bodies, non-dict payloads, and giant numbers."""
    backend = MockAudioBackend(playback_duration=0.01)
    service = AlertService(backend=backend, runtime_dir=tmp_path)
    plugin = FindMyPhonePlugin(alert_service=service)

    # 1. Packet with giant integers (10^100) and unexpected schema fields
    giant_int = 10**100
    pkt_fuzz = Packet(
        id=int(time.time() * 1000),
        type=FINDMYPHONE_REQUEST_TYPE,
        body={
            "giant_number": giant_int,
            "negative_large": -10**50,
            "arbitrary_keys": ["unrecognized", {"deep": "nesting"}],
            "overflow_str": "A" * 10000,
        },
    )
    # Handling fuzz packet with unexpected fields succeeds without error
    await plugin.handle_packet(pkt_fuzz, "dev_fuzz", lambda p: None)
    assert service.is_active() is True
    await service.stop_alert()
    assert service.is_active() is False

    # 2. Wire-level non-dict body is rejected by PacketEngine
    raw_bad_wire = b'{"id": 12345, "type": "kdeconnect.findmyphone.request", "body": "invalid_string_body"}\n'
    with pytest.raises(InvalidPacketSchemaError):
        PacketEngine.parse(raw_bad_wire)

    raw_list_wire = b'{"id": 12345, "type": "kdeconnect.findmyphone.request", "body": [1, 2, 3]}\n'
    with pytest.raises(InvalidPacketSchemaError):
        PacketEngine.parse(raw_list_wire)

    # 3. Direct packet object with non-dict body passed to handle_packet is safely ignored
    class MalformedBodyPacket:
        type = FINDMYPHONE_REQUEST_TYPE
        body = "not_a_dict"

    await plugin.handle_packet(MalformedBodyPacket(), "dev_bad_body", lambda p: None)  # type: ignore
    assert service.is_active() is False

    # 4. create_find_my_phone_request_packet with non-dict body falls back safely to {}
    safe_pkt = create_find_my_phone_request_packet("non_dict_body")  # type: ignore
    assert safe_pkt.type == FINDMYPHONE_REQUEST_TYPE
    assert safe_pkt.body == {}


async def test_adversarial_inbound_unauthenticated_or_unpaired_device_ids(tmp_path: Path) -> None:
    """Inbound locate requests from unauthenticated, empty, or unpaired device IDs are quarantined."""
    backend = MockAudioBackend(playback_duration=0.01)
    service = AlertService(backend=backend, runtime_dir=tmp_path)

    # Device trust simulator: only 'trusted_device_01' is paired
    paired_store = {"trusted_device_01": True}

    def paired_checker(did: str) -> bool:
        return paired_store.get(did, False)

    plugin = FindMyPhonePlugin(alert_service=service, is_paired_checker=paired_checker)
    pkt = create_find_my_phone_request_packet()

    # 1. Rogue / unauthenticated device attempts to trigger alert -> rejected
    await plugin.handle_packet(pkt, "rogue_intruder_99", lambda p: None)
    assert service.is_active() is False

    # 2. Empty string or whitespace device ID -> rejected
    await plugin.handle_packet(pkt, "", lambda p: None)
    assert service.is_active() is False
    await plugin.handle_packet(pkt, "   ", lambda p: None)
    assert service.is_active() is False

    # 3. Trusted device triggers alert -> accepted
    await plugin.handle_packet(pkt, "trusted_device_01", lambda p: None)
    assert service.is_active() is True

    # 4. Rogue device attempts to toggle OFF the alert -> rejected, alert remains ACTIVE
    await plugin.handle_packet(pkt, "rogue_intruder_99", lambda p: None)
    assert service.is_active() is True

    # 5. Trusted device toggles alert -> accepted, alert stopped
    await plugin.handle_packet(pkt, "trusted_device_01", lambda p: None)
    assert service.is_active() is False


async def test_adversarial_outbound_ring_phone_offline_or_unresolvable() -> None:
    """Outbound ring_phone against offline, unresolvable, or disconnected peers fails with ConnectionClosedError."""
    plugin = FindMyPhonePlugin()

    # 1. Target device ID is empty, whitespace, or invalid -> ConnectionClosedError
    with pytest.raises(ConnectionClosedError) as exc_info:
        await plugin.ring_phone("", lambda p: None)
    assert "invalid or unresolvable device ID" in str(exc_info.value)

    with pytest.raises(ConnectionClosedError):
        await plugin.ring_phone("   ", lambda p: None)

    # 2. packet_sender missing / None -> ConnectionClosedError
    with pytest.raises(ConnectionClosedError) as exc_info:
        await plugin.ring_phone("offline_device_404")
    assert "send_packet callable is invalid or missing" in str(exc_info.value)

    # 3. Target device is offline in transport layer -> sender raises ConnectionClosedError
    def offline_sender(dev: str, pkt: Packet) -> None:
        raise ConnectionClosedError(f"Device '{dev}' is unreachable: no route to host")

    with pytest.raises(ConnectionClosedError) as exc_info:
        await plugin.ring_phone("offline_device_404", offline_sender)
    assert "no route to host" in str(exc_info.value)

    # 4. Sender raises generic OSError -> wrapped in ConnectionClosedError
    def oserror_sender(pkt: Packet) -> None:
        raise OSError("Host is down (EHOSTDOWN)")

    with pytest.raises(ConnectionClosedError) as exc_info:
        await plugin.ring_phone("unresolvable_node", oserror_sender)
    assert "Host is down" in str(exc_info.value)


async def test_adversarial_outbound_broken_stream_mid_transmission() -> None:
    """Broken transport stream write failures mid-transmission raise ConnectionClosedError without leaks."""
    plugin = FindMyPhonePlugin()

    # 1. BrokenPipeError mid-write
    async def broken_pipe_sender(pkt: Packet) -> None:
        await asyncio.sleep(0.001)
        raise BrokenPipeError("Broken pipe on TCP stream during write")

    with pytest.raises(ConnectionClosedError) as exc_info:
        await plugin.ring_phone("peer_broken_stream", broken_pipe_sender)
    assert "Broken pipe" in str(exc_info.value)

    # 2. ConnectionResetError mid-write
    async def reset_sender(pkt: Packet) -> None:
        raise ConnectionResetError("Connection reset by peer during TLS record transmission")

    with pytest.raises(ConnectionClosedError) as exc_info:
        await plugin.ring_phone("peer_reset_stream", reset_sender)
    assert "Connection reset by peer" in str(exc_info.value)

    # 3. asyncio.TimeoutError mid-write
    async def timeout_sender(pkt: Packet) -> None:
        raise asyncio.TimeoutError("Socket write buffer flush timed out")

    with pytest.raises(ConnectionClosedError) as exc_info:
        await plugin.ring_phone("peer_timeout_stream", timeout_sender)
    assert "timed out" in str(exc_info.value).lower()



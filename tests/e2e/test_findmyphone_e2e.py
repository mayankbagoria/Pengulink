"""tests.e2e.test_findmyphone_e2e
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-End integration test suite for Find My Phone & Find This Device
features running against MockKdeConnectDevice with live mTLS and IPC.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any, List
import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.events import (
    DeviceConnected,
    FindDeviceAlertStoppedEvent,
    FindDeviceAlertTriggeredEvent,
)
from pengulink.network.packet import Packet
from tests.e2e.conftest import e2e_environment, wait_for
from tests.mock.mock_device import MockDevice


async def _pair_and_connect(daemon: Any, mock: MockDevice) -> None:
    """Helper to establish mutual pairing and an active TLS connection."""
    await mock.broadcast_identity(
        target_ip="127.0.0.1",
        target_port=daemon.config.network.udp_port,
    )
    await wait_for(lambda: mock.device_id in daemon._discovered_devices, timeout=4.0)
    res = await daemon.pair_device(mock.device_id, wait=True, timeout=5.0)
    assert res["paired"] is True
    assert daemon.is_device_connected(mock.device_id) is True

    await daemon.event_bus.publish(
        DeviceConnected(
            device_id=mock.device_id,
            host="127.0.0.1",
            port=mock.bound_tcp_port or 0,
        )
    )
    await asyncio.sleep(0.05)


async def test_e2e_find_phone_outbound(tmp_path: Path) -> None:
    """Verify triggering find_phone via IPC transmits packet to mock device."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        res = await ipc_client.find_phone(mock.device_id)
        assert res.get("status") in ("transmitted", "ok")
        assert res.get("device_id") == mock.device_id

        # Wait for packet arrival at mock device
        pkt = await mock.wait_for_find_phone_request(timeout=3.0)
        assert pkt.type in ("kdeconnect.findmyphone.request", "kdeconnect.findmyphone")
        assert pkt.body == {}


async def test_e2e_find_this_device_inbound(tmp_path: Path) -> None:
    """Verify phone sending findmyphone request triggers local AlertService."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        triggered_events: List[Any] = []
        daemon.event_bus.subscribe(
            FindDeviceAlertTriggeredEvent,
            lambda ev: triggered_events.append(ev),
        )

        # Phone triggers desktop alert
        await mock.send_find_this_device_request()

        await wait_for(lambda: len(triggered_events) >= 1, timeout=4.0)
        assert triggered_events[0].device_id == mock.device_id

        status = await ipc_client.alert_status()
        assert status.get("active") is True
        assert status.get("source") == "remote"

        # Silence alert
        await ipc_client.alert_stop()


async def test_e2e_find_this_device_toggle_cancellation(tmp_path: Path) -> None:
    """Verify second locate request toggles off active alert."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        stopped_events: List[Any] = []
        daemon.event_bus.subscribe(
            FindDeviceAlertStoppedEvent,
            lambda ev: stopped_events.append(ev),
        )

        # First request rings
        await mock.send_find_this_device_request()
        await wait_for(lambda: daemon.alert_service.is_active(), timeout=3.0)

        status_1 = await ipc_client.alert_status()
        assert status_1.get("active") is True

        # Second request silences
        await mock.send_find_this_device_request()
        await wait_for(lambda: not daemon.alert_service.is_active(), timeout=3.0)

        status_2 = await ipc_client.alert_status()
        assert status_2.get("active") is False
        assert len(stopped_events) >= 1


async def test_e2e_alert_cli_lifecycle(tmp_path: Path) -> None:
    """Verify Click CLI commands trigger, introspect, and stop desktop alerts."""
    runner = CliRunner()
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        sock = str(daemon.config.storage.runtime_dir / "daemon.sock")

        # 1. Start alert via CLI
        res_test = await asyncio.to_thread(runner.invoke, cli, ["-s", sock, "alert", "test", "--duration", "20"])
        assert res_test.exit_code == 0
        assert "alert" in res_test.output.lower() or "ringing" in res_test.output.lower()

        # 2. Status via CLI
        res_stat = await asyncio.to_thread(runner.invoke, cli, ["-s", sock, "alert", "status"])
        assert res_stat.exit_code == 0
        assert "ringing" in res_stat.output.lower() or "active" in res_stat.output.lower()

        # 3. Stop via CLI
        res_stop = await asyncio.to_thread(runner.invoke, cli, ["-s", sock, "alert", "stop"])
        assert res_stop.exit_code == 0
        assert "stopped" in res_stop.output.lower() or "silenced" in res_stop.output.lower()

        # 4. Status is now idle
        status = await ipc_client.alert_status()
        assert status.get("active") is False


async def test_e2e_disconnection_resilience(tmp_path: Path) -> None:
    """Phone disconnecting during active alert does not crash alert service."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        # Trigger alert from phone
        await mock.send_find_this_device_request()
        await wait_for(lambda: daemon.alert_service.is_active(), timeout=3.0)

        # Drop phone connection abruptly
        await mock.drop_connection()
        await wait_for(lambda: not daemon.is_device_connected(mock.device_id), timeout=3.0)

        # Alert continues or halts cleanly without daemon crash
        status = await ipc_client.alert_status()
        assert isinstance(status, dict)

        await ipc_client.alert_stop()


async def test_e2e_daemon_shutdown_clean_drain(tmp_path: Path) -> None:
    """Stopping daemon during active alert drains all audio processes cleanly."""
    async with e2e_environment(tmp_path) as (daemon, mock, ipc_client):
        await _pair_and_connect(daemon, mock)

        await ipc_client.alert_test(duration=30.0)
        assert daemon.alert_service.is_active() is True

        # Stop daemon
        await daemon.stop()
        assert daemon.alert_service.is_active() is False

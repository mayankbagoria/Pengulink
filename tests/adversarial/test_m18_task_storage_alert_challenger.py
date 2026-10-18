"""tests.adversarial.test_m18_task_storage_alert_challenger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial Empirical Challenge Suite for Milestone 18:
- CLUS-02 (BUG-CRIT-02): Task lifecycle retention under heavy garbage collection pressure.
  100+ connection/disconnection cycles with aggressive multi-generation gc.collect() passes;
  confirms zero dropped teardown tasks and zero stuck state machines.
- CLUS-12 (BUG-HIGH-05): DeviceStore.save_device() concurrent renaming stress.
  Verifies pinned certificates, is_paired trust flags, and device metadata are never lost or reset.
- CLUS-15 (BUG-HIGH-08): AlertService concurrency mutex race conditions.
  75+ rapid concurrent start_alert / stop_alert iterations; confirms stop_alert never terminates
  newly started audio processes and zero deadlocks occur under contention.
"""

from __future__ import annotations

import asyncio
import gc
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.core.events import DeviceConnected, DeviceDisconnected
from pengulink.core.state_machine import DeviceState
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.network.tcp import ConnectionRole, TCPConnection
from pengulink.services.alert import AlertService, MockAudioBackend, MockAudioProcess
from pengulink.storage.base import DeviceStore
from pengulink.storage.json_store import JsonStorageBackend


def make_test_config(tmp_path: Path) -> PenguLinkConfig:
    cfg = PenguLinkConfig()
    cfg.storage.config_dir = tmp_path / "config"
    cfg.storage.data_dir = tmp_path / "storage"
    cfg.storage.runtime_dir = tmp_path / "runtime"
    return cfg


# =============================================================================
# 1. CLUS-02: Task Retention Under Heavy Garbage Collection Pressure (100+ Cycles)
# =============================================================================

class TestClus02LifecycleGCStress:
    """Empirical challenge suite for coroutine lifecycle and GC resilience."""

    async def test_100_plus_connection_disconnection_cycles_under_gc_pressure(
        self, tmp_path: Path
    ) -> None:
        """Execute 120 connection/disconnection cycles under multi-generation GC pressure.

        Confirms:
        1. Zero dropped teardown tasks (len(_background_tasks) settles to 0).
        2. Zero stuck state machines (all 120 machines reach DeviceState.DISCONNECTED).
        3. All 120 DeviceDisconnected events are dispatched and received.
        4. Zero connection leaks in daemon._active_connections.
        """
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        disconnected_events: List[DeviceDisconnected] = []
        daemon.event_bus.subscribe(DeviceDisconnected, lambda ev: disconnected_events.append(ev))

        total_cycles = 120
        stuck_machines: List[str] = []

        for i in range(total_cycles):
            dev_id = f"gc_stress_dev_{i}"

            # Create mock connection handles
            reader = asyncio.StreamReader()
            writer = MagicMock()
            writer.close = MagicMock()
            writer.wait_closed = AsyncMock()

            conn = TCPConnection(
                reader=reader,
                writer=writer,
                role=ConnectionRole.INBOUND,
                peer_ip="127.0.0.1",
                peer_port=1716 + (i % 64),
            )
            conn.remote_device_id = dev_id

            # Advance state machine to CONNECTED
            sm = daemon._get_or_create_state_machine(dev_id)
            await sm.transition_to(DeviceState.CONNECTING)
            await sm.transition_to(DeviceState.CONNECTED)
            assert sm.current_state == DeviceState.CONNECTED

            # Register connection into daemon
            daemon._register_connection(dev_id, conn)
            assert daemon.is_device_connected(dev_id) is True

            # Trigger asynchronous connection close
            close_task = asyncio.create_task(conn.close())

            # Aggressive GC sweep across all generations during task scheduling
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)

            await close_task

            # Immediate secondary GC collection while background tasks run
            gc.collect()

            # Yield control to event loop to allow teardown coroutines to progress
            await asyncio.sleep(0.003)

            # Check if state machine was stuck
            if sm.current_state != DeviceState.DISCONNECTED:
                stuck_machines.append(dev_id)

            assert daemon.is_device_connected(dev_id) is False

        # Quiescence check: allow event loop to settle remaining tasks
        await asyncio.sleep(0.05)
        gc.collect()

        assert len(stuck_machines) == 0, f"Stuck state machines: {stuck_machines}"
        assert len(disconnected_events) == total_cycles, (
            f"Expected {total_cycles} disconnect events, received {len(disconnected_events)}"
        )
        assert len(daemon._background_tasks) == 0, (
            f"Leaked {len(daemon._background_tasks)} background tasks in daemon"
        )
        assert len(daemon._active_connections) == 0, (
            f"Leaked {len(daemon._active_connections)} active connections in daemon"
        )

    async def test_concurrent_multi_device_flapping_under_gc_pressure(
        self, tmp_path: Path
    ) -> None:
        """Concurrent connection flapping across 10 devices simultaneously under GC pressure."""
        cfg = make_test_config(tmp_path)
        daemon = PenguLinkDaemon(config=cfg, socket_path=tmp_path / "test.sock")
        await daemon.device_store.initialize()

        disconnected_events: List[DeviceDisconnected] = []
        daemon.event_bus.subscribe(DeviceDisconnected, lambda ev: disconnected_events.append(ev))

        num_devices = 10
        flaps_per_device = 12
        total_flaps = num_devices * flaps_per_device

        async def device_flap_worker(d_idx: int) -> None:
            dev_id = f"flap_dev_{d_idx}"
            for cycle in range(flaps_per_device):
                reader = asyncio.StreamReader()
                writer = MagicMock()
                writer.close = MagicMock()
                writer.wait_closed = AsyncMock()

                conn = TCPConnection(
                    reader=reader,
                    writer=writer,
                    role=ConnectionRole.INBOUND,
                    peer_ip="127.0.0.1",
                    peer_port=1800 + d_idx,
                )
                conn.remote_device_id = dev_id

                sm = daemon._get_or_create_state_machine(dev_id)
                if sm.current_state == DeviceState.DISCONNECTED:
                    await sm.transition_to(DeviceState.CONNECTING)
                    await sm.transition_to(DeviceState.CONNECTED)

                daemon._register_connection(dev_id, conn)

                # Small jitter before closing
                await asyncio.sleep(random.uniform(0.001, 0.005))
                await conn.close()
                gc.collect()

        workers = [asyncio.create_task(device_flap_worker(i)) for i in range(num_devices)]
        await asyncio.gather(*workers)

        # Allow full draining
        await asyncio.sleep(0.08)
        gc.collect()

        assert len(disconnected_events) == total_flaps
        for d_idx in range(num_devices):
            sm = daemon._state_machines.get(f"flap_dev_{d_idx}")
            assert sm is not None
            assert sm.current_state == DeviceState.DISCONNECTED
        assert len(daemon._background_tasks) == 0


# =============================================================================
# 2. CLUS-12: Storage Preservation & Concurrent Renaming Stress
# =============================================================================

class TestClus12StoragePreservationStress:
    """Empirical challenge suite for DeviceStore.save_device() credential preservation."""

    async def test_100_concurrent_renaming_operations_preserve_cert_and_paired_state(
        self, tmp_path: Path
    ) -> None:
        """100 concurrent save_device() calls with only name parameter.

        Verifies that:
        1. certificate_pem is strictly preserved across all concurrent writes.
        2. is_paired remains True throughout the contention.
        3. device_type remains unchanged.
        4. Extra attributes persist.
        5. Re-initializing from disk confirms on-disk JSON integrity.
        """
        backend = JsonStorageBackend(tmp_path / "storage")
        store = DeviceStore(backend)
        await store.initialize()

        test_cert = (
            "-----BEGIN CERTIFICATE-----\n"
            "MIIBjjCCATSgAwIBAgIUQ1b3L...CLUS12...STRESS...TEST...==\n"
            "-----END CERTIFICATE-----\n"
        )
        dev_id = "clus12_target_phone"

        # Initial pair and storage
        await store.save_device(
            device_id=dev_id,
            name="Original Pixel 8",
            device_type="phone",
            certificate_pem=test_cert,
            is_paired=True,
            extra={"hardware_rev": "rev3", "vendor": "Google"},
        )

        # Verify initial record
        initial = await store.get_device(dev_id)
        assert initial is not None
        assert initial["certificate_pem"] == test_cert
        assert initial["is_paired"] is True
        assert initial["device_type"] == "phone"

        # Launch 100 concurrent renaming operations
        num_renames = 100

        async def rename_worker(i: int) -> None:
            await asyncio.sleep(random.uniform(0.0001, 0.003))
            # Calling save_device with only name, leaving cert=None and is_paired=False (defaults)
            await store.save_device(
                device_id=dev_id,
                name=f"Renamed Phone #{i}",
            )

        tasks = [asyncio.create_task(rename_worker(i)) for i in range(num_renames)]
        await asyncio.gather(*tasks)

        # Verify in-memory state
        in_mem = await store.get_device(dev_id)
        assert in_mem is not None
        assert in_mem["certificate_pem"] == test_cert, "Pinned certificate was lost during concurrent renames!"
        assert in_mem["is_paired"] is True, "is_paired trust flag was reset to False!"
        assert in_mem["device_type"] == "phone"
        assert in_mem["hardware_rev"] == "rev3"
        assert in_mem["vendor"] == "Google"
        assert in_mem["device_name"].startswith("Renamed Phone #")

        # Verify on-disk persistence with clean backend reload
        await backend.close()

        fresh_backend = JsonStorageBackend(tmp_path / "storage")
        fresh_store = DeviceStore(fresh_backend)
        await fresh_store.initialize()

        on_disk = await fresh_store.get_device(dev_id)
        assert on_disk is not None
        assert on_disk["certificate_pem"] == test_cert, "Pinned certificate lost in on-disk JSON!"
        assert on_disk["is_paired"] is True, "is_paired lost in on-disk JSON!"
        assert on_disk["device_type"] == "phone"
        assert on_disk["hardware_rev"] == "rev3"

    async def test_multi_device_concurrent_renaming_storm(self, tmp_path: Path) -> None:
        """200 concurrent rename operations across 10 distinct paired devices."""
        backend = JsonStorageBackend(tmp_path / "storage_storm")
        store = DeviceStore(backend)
        await store.initialize()

        num_devices = 10
        ops_per_dev = 20

        # Seed 10 paired devices with unique certificates
        for d in range(num_devices):
            await store.save_device(
                device_id=f"storm_dev_{d}",
                name=f"Device {d} Seed",
                device_type="tablet" if d % 2 == 0 else "phone",
                certificate_pem=f"-----CERT-{d}-----",
                is_paired=True,
                extra={"slot": d},
            )

        # Launch concurrent barrage of renames
        storm_tasks = []
        for d in range(num_devices):
            for op in range(ops_per_dev):
                storm_tasks.append(
                    store.save_device(
                        device_id=f"storm_dev_{d}",
                        name=f"Storm Dev {d} Op {op}",
                    )
                )

        await asyncio.gather(*storm_tasks)

        # Verify all 10 devices retained certificates and trust
        for d in range(num_devices):
            rec = await store.get_device(f"storm_dev_{d}")
            assert rec is not None
            assert rec["certificate_pem"] == f"-----CERT-{d}-----"
            assert rec["is_paired"] is True
            assert rec["slot"] == d

        # Verify list_devices() returns all 10 intact
        all_devs = await store.list_devices()
        assert len(all_devs) == num_devices
        for dev_entry in all_devs:
            assert dev_entry["is_paired"] is True
            assert dev_entry["certificate_pem"].startswith("-----CERT-")


# =============================================================================
# 3. CLUS-15: AlertService Concurrency Mutex & Race Conditions (50+ Iterations)
# =============================================================================

class TestClus15AlertServiceRaceStress:
    """Empirical challenge suite for AlertService stop_alert mutex and process lifecycle."""

    async def test_rapid_concurrent_start_stop_race_50_plus_iterations(
        self, tmp_path: Path
    ) -> None:
        """Trigger rapid concurrent start_alert / stop_alert race conditions across 75 iterations.

        Confirms:
        1. Zero deadlocks under concurrent mutex contention (timeout = 3.0s per round).
        2. stop_alert never terminates newly started audio processes.
        3. Alert state correctly recovers without orphaned active flags.
        """
        backend = MockAudioBackend(playback_duration=0.05)
        service = AlertService(backend=backend, runtime_dir=tmp_path)

        total_iterations = 75
        deadlocks = 0
        new_procs_terminated = 0

        for i in range(total_iterations):
            # 1. Establish an active alert
            await service.start_alert(device_id=f"dev_init_{i}", max_duration=2.0)
            await asyncio.sleep(random.uniform(0.001, 0.008))
            init_proc = service._current_process
            assert init_proc is not None

            # 2. Race condition: launch stop_alert and start_alert concurrently
            stop_task = asyncio.create_task(service.stop_alert(reason=f"race_stop_{i}"))
            start_task = asyncio.create_task(
                service.start_alert(device_id=f"dev_new_{i}", max_duration=2.0)
            )

            try:
                await asyncio.wait_for(asyncio.gather(stop_task, start_task), timeout=3.0)
            except asyncio.TimeoutError:
                deadlocks += 1
                pytest.fail(f"DEADLOCK encountered on iteration {i} during concurrent start/stop!")

            await asyncio.sleep(0.005)
            new_proc = service._current_process

            # 3. Verification: if a new process was started, confirm stop_alert did not terminate it
            if new_proc is not None and new_proc is not init_proc:
                if getattr(new_proc, "terminated", False):
                    new_procs_terminated += 1

            # 4. Clean shutdown for next iteration
            await service.stop_alert()
            await asyncio.sleep(0.002)

        assert deadlocks == 0, f"Encountered {deadlocks} deadlocks!"
        assert new_procs_terminated == 0, (
            f"stop_alert terminated {new_procs_terminated} newly started audio processes!"
        )

    async def test_concurrent_alert_hammer_burts(self, tmp_path: Path) -> None:
        """Heavy concurrent burst: 30 simultaneous start and stop requests in random order."""
        backend = MockAudioBackend(playback_duration=0.03)
        service = AlertService(backend=backend, runtime_dir=tmp_path)

        burst_size = 30
        tasks = []
        for i in range(burst_size):
            if i % 2 == 0:
                tasks.append(service.start_alert(device_id=f"burst_dev_{i}", max_duration=0.5))
            else:
                tasks.append(service.stop_alert(reason=f"burst_stop_{i}"))

        # Shuffle execution order
        random.shuffle(tasks)

        # Confirm all complete within 5 seconds with zero deadlocks
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
        for r in results:
            assert not isinstance(r, Exception), f"Unexpected exception during burst: {r}"

        # Clean final stop
        await service.stop_alert()
        await asyncio.sleep(0.05)
        assert service.is_active() is False
        assert service._current_process is None

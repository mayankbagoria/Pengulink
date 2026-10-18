"""tests.adversarial.test_m17_state_storage_stress_challenger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial and Empirical Stress Test Harness for Milestone 17 Fixes:
- CLUS-03 & CLUS-24: DeviceStateMachine transition matrix (PAIRED -> DISCONNECTED,
  PAIRED -> ERROR, PAIRING -> ERROR), concurrent transition storm, listener isolation,
  and daemon transition routing.
- CLUS-05: JsonStorageBackend _ensure_initialized() concurrency stress, race condition
  proofing against data loss / wipeout, concurrent uninitialized stampedes, and corrupt
  file recovery under load.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.core.events import EventBus
from pengulink.core.exceptions import InvalidStateTransitionError, StorageCorruptedError
from pengulink.core.state_machine import DeviceState, DeviceStateMachine
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.storage.json_store import JsonStorageBackend


# =============================================================================
# 1. CLUS-03 & CLUS-24: DeviceStateMachine Empirical Stress Tests
# =============================================================================

class TestStateMachineStress:
    """Empirical challenge suite for DeviceStateMachine transitions and concurrency."""

    async def test_paired_to_disconnected_transition_invariants(self) -> None:
        """Verify PAIRED -> DISCONNECTED transition, audit record, and state properties."""
        sm = DeviceStateMachine(device_id="test_paired_disc", initial_state=DeviceState.DISCONNECTED)
        await sm.transition_to(DeviceState.CONNECTING)
        await sm.transition_to(DeviceState.PAIRING)
        await sm.transition_to(DeviceState.PAIRED)

        assert sm.current_state == DeviceState.PAIRED
        assert sm.is_paired is True
        assert sm.can_transition_to(DeviceState.DISCONNECTED) is True

        # Perform transition
        reason = "Abrupt TCP socket teardown"
        await sm.transition_to(DeviceState.DISCONNECTED, reason=reason)

        assert sm.current_state == DeviceState.DISCONNECTED
        assert sm.is_paired is False
        last_record = sm.history[-1]
        assert last_record.from_state == DeviceState.PAIRED
        assert last_record.to_state == DeviceState.DISCONNECTED
        assert last_record.reason == reason

    async def test_paired_and_pairing_to_error_transitions(self) -> None:
        """Verify PAIRING -> ERROR and PAIRED -> ERROR transitions with recovery paths."""
        # 1. PAIRING -> ERROR
        sm1 = DeviceStateMachine(device_id="test_pairing_err", initial_state=DeviceState.DISCONNECTED)
        await sm1.transition_to(DeviceState.CONNECTING)
        await sm1.transition_to(DeviceState.PAIRING)

        assert sm1.can_transition_to(DeviceState.ERROR) is True
        await sm1.transition_to(DeviceState.ERROR, reason="Handshake protocol error")
        assert sm1.current_state == DeviceState.ERROR
        assert sm1.is_paired is False

        # Recovery from ERROR: can transition to CONNECTING or DISCONNECTED
        assert sm1.can_transition_to(DeviceState.CONNECTING) is True
        assert sm1.can_transition_to(DeviceState.DISCONNECTED) is True
        await sm1.transition_to(DeviceState.DISCONNECTED, reason="Error recovery reset")
        assert sm1.current_state == DeviceState.DISCONNECTED

        # 2. PAIRED -> ERROR
        sm2 = DeviceStateMachine(device_id="test_paired_err", initial_state=DeviceState.DISCONNECTED)
        await sm2.transition_to(DeviceState.CONNECTING)
        await sm2.transition_to(DeviceState.PAIRING)
        await sm2.transition_to(DeviceState.PAIRED)

        assert sm2.can_transition_to(DeviceState.ERROR) is True
        await sm2.transition_to(DeviceState.ERROR, reason="Fatal TLS read error")
        assert sm2.current_state == DeviceState.ERROR
        assert sm2.is_paired is False

        # Recovery from ERROR directly to CONNECTING
        await sm2.transition_to(DeviceState.CONNECTING, reason="Auto-reconnect after error")
        assert sm2.current_state == DeviceState.CONNECTING

    async def test_illegal_transitions_still_rejected(self) -> None:
        """Verify that expanding valid transitions did not weaken illegal transition enforcement."""
        sm = DeviceStateMachine(device_id="test_invariants", initial_state=DeviceState.PAIRED)

        # Forbidden transitions from PAIRED: DISCOVERED, PAIRING
        assert sm.can_transition_to(DeviceState.DISCOVERED) is False
        with pytest.raises(InvalidStateTransitionError):
            await sm.transition_to(DeviceState.DISCOVERED)

        assert sm.can_transition_to(DeviceState.PAIRING) is False
        with pytest.raises(InvalidStateTransitionError):
            await sm.transition_to(DeviceState.PAIRING)

        # Forbidden transitions from PAIRING: DISCOVERED, CONNECTING
        sm_pairing = DeviceStateMachine(device_id="test_pairing_inv", initial_state=DeviceState.PAIRING)
        assert sm_pairing.can_transition_to(DeviceState.DISCOVERED) is False
        with pytest.raises(InvalidStateTransitionError):
            await sm_pairing.transition_to(DeviceState.DISCOVERED)

    async def test_daemon_transition_device_routing(self, sample_config: PenguLinkConfig) -> None:
        """Verify PenguLinkDaemon._transition_device properly routes PAIRED->DISCONNECTED and ERROR."""
        sample_config.ensure_directories()
        daemon = PenguLinkDaemon(config=sample_config)

        # 1. PAIRED -> DISCONNECTED via daemon._transition_device
        sm = daemon._get_or_create_state_machine("dev_d1")
        await sm.transition_to(DeviceState.CONNECTING)
        await sm.transition_to(DeviceState.PAIRING)
        await sm.transition_to(DeviceState.PAIRED)
        assert sm.current_state == DeviceState.PAIRED

        await daemon._transition_device("dev_d1", DeviceState.DISCONNECTED, reason="Connection dropped")
        assert sm.current_state == DeviceState.DISCONNECTED

        # 2. PAIRED -> ERROR via daemon._transition_device
        sm2 = daemon._get_or_create_state_machine("dev_d2")
        await sm2.transition_to(DeviceState.CONNECTING)
        await sm2.transition_to(DeviceState.PAIRING)
        await sm2.transition_to(DeviceState.PAIRED)

        await daemon._transition_device("dev_d2", DeviceState.ERROR, reason="Cryptographic failure")
        assert sm2.current_state == DeviceState.ERROR

        # 3. PAIRING -> ERROR via daemon._transition_device
        sm3 = daemon._get_or_create_state_machine("dev_d3")
        await sm3.transition_to(DeviceState.CONNECTING)
        await sm3.transition_to(DeviceState.PAIRING)

        await daemon._transition_device("dev_d3", DeviceState.ERROR, reason="Handshake timeout")
        assert sm3.current_state == DeviceState.ERROR

    async def test_concurrent_state_machine_transition_storm(self) -> None:
        """Stress test 100 concurrent state machines rapidly executing randomized valid transition sequences."""
        num_devices = 100
        iterations_per_device = 30

        async def run_lifecycle(dev_idx: int) -> None:
            dev_id = f"storm_dev_{dev_idx}"
            sm = DeviceStateMachine(device_id=dev_id, initial_state=DeviceState.DISCONNECTED)

            for step in range(iterations_per_device):
                # Permutation A: Normal disconnect cycle
                await sm.transition_to(DeviceState.CONNECTING, reason=f"step {step} connect")
                await sm.transition_to(DeviceState.PAIRING, reason=f"step {step} pairing")
                await sm.transition_to(DeviceState.PAIRED, reason=f"step {step} paired")
                assert sm.is_paired is True

                if step % 3 == 0:
                    # PAIRED -> DISCONNECTED (CLUS-03)
                    await sm.transition_to(DeviceState.DISCONNECTED, reason=f"step {step} disconnect")
                    assert sm.is_paired is False
                elif step % 3 == 1:
                    # PAIRED -> ERROR (CLUS-24)
                    await sm.transition_to(DeviceState.ERROR, reason=f"step {step} paired error")
                    assert sm.is_paired is False
                    await sm.transition_to(DeviceState.DISCONNECTED, reason=f"step {step} error reset")
                else:
                    # PAIRED -> CONNECTED -> DISCONNECTING -> DISCONNECTED
                    await sm.transition_to(DeviceState.CONNECTED, reason=f"step {step} connected")
                    await sm.transition_to(DeviceState.DISCONNECTING, reason=f"step {step} disconnecting")
                    await sm.transition_to(DeviceState.DISCONNECTED, reason=f"step {step} disconnected")

                # Permutation B: PAIRING -> ERROR (CLUS-24)
                await sm.transition_to(DeviceState.CONNECTING, reason=f"step {step} connect2")
                await sm.transition_to(DeviceState.PAIRING, reason=f"step {step} pairing2")
                await sm.transition_to(DeviceState.ERROR, reason=f"step {step} pairing error")
                assert sm.current_state == DeviceState.ERROR
                await sm.transition_to(DeviceState.CONNECTING, reason=f"step {step} reconnect from error")
                await sm.transition_to(DeviceState.DISCONNECTED, reason=f"step {step} abort to disconnected")

            assert sm.current_state == DeviceState.DISCONNECTED

        tasks = [run_lifecycle(i) for i in range(num_devices)]
        await asyncio.gather(*tasks)

    async def test_listener_exception_isolation_under_transitions(self) -> None:
        """Verify that buggy or slow listeners do not corrupt transitions or current_state."""
        sm = DeviceStateMachine(device_id="test_listeners", initial_state=DeviceState.DISCONNECTED)

        async def failing_async_listener(from_s: DeviceState, to_s: DeviceState, r: Any) -> None:
            await asyncio.sleep(0.001)
            raise RuntimeError("Listener exploded!")

        def failing_sync_listener(from_s: DeviceState, to_s: DeviceState, r: Any) -> None:
            raise ValueError("Sync listener blew up!")

        sm.add_listener(failing_async_listener)
        sm.add_listener(failing_sync_listener)

        # Transition PAIRED -> DISCONNECTED must complete despite listener exceptions
        await sm.transition_to(DeviceState.CONNECTING)
        await sm.transition_to(DeviceState.PAIRING)
        await sm.transition_to(DeviceState.PAIRED)
        assert sm.current_state == DeviceState.PAIRED

        await sm.transition_to(DeviceState.DISCONNECTED, reason="Check listener isolation")
        assert sm.current_state == DeviceState.DISCONNECTED

        # Transition PAIRED -> ERROR must also complete
        await sm.transition_to(DeviceState.CONNECTING)
        await sm.transition_to(DeviceState.PAIRING)
        await sm.transition_to(DeviceState.PAIRED)
        await sm.transition_to(DeviceState.ERROR, reason="Check listener isolation in error")
        assert sm.current_state == DeviceState.ERROR


# =============================================================================
# 2. CLUS-05: JsonStorageBackend Empirical Stress Tests
# =============================================================================

class TestJsonStorageBackendStress:
    """Empirical challenge suite for JsonStorageBackend uninitialized concurrency."""

    async def test_uninitialized_save_preserves_large_preexisting_dataset(self, tmp_path: Path) -> None:
        """Verify CLUS-05: Uninitialized save_device never wipes existing 50 pinned devices."""
        storage_dir = tmp_path / "storage_dataset"
        storage_dir.mkdir(parents=True, exist_ok=True)
        devices_file = storage_dir / "devices.json"

        # Seed 50 pre-existing devices with certificate fingerprints
        seeded_devices: Dict[str, Dict[str, Any]] = {}
        for i in range(50):
            dev_id = f"device_{i:03d}"
            seeded_devices[dev_id] = {
                "device_id": dev_id,
                "name": f"Phone Model {i}",
                "certificate_fingerprint": f"SHA256:FINGERPRINT_{i:04d}",
                "is_paired": True,
                "paired_at": 1729000000.0 + i,
            }
        devices_file.write_text(json.dumps(seeded_devices, indent=2), encoding="utf-8")

        # Instantiate uninitialized backend
        backend = JsonStorageBackend(storage_dir)
        assert backend._initialized is False
        assert backend._devices_cache == {}

        # Save brand new device WITHOUT prior initialize()
        new_device = {
            "device_id": "device_brand_new",
            "name": "Super Phone 2026",
            "certificate_fingerprint": "SHA256:NEW_DEVICE_FP",
            "is_paired": True,
        }
        await backend.save_device("device_brand_new", new_device)

        # Inspect disk file directly
        disk_data = json.loads(devices_file.read_text(encoding="utf-8"))
        assert len(disk_data) == 51, f"Expected 51 devices on disk, found {len(disk_data)}"
        for i in range(50):
            dev_id = f"device_{i:03d}"
            assert dev_id in disk_data, f"Seeded device {dev_id} was lost!"
            assert disk_data[dev_id]["name"] == f"Phone Model {i}"
        assert "device_brand_new" in disk_data

    async def test_massive_uninitialized_concurrent_stampede(self, tmp_path: Path) -> None:
        """Stress test: 150 concurrent uninitialized tasks calling save, get, list, and delete."""
        storage_dir = tmp_path / "storage_stampede"
        storage_dir.mkdir(parents=True, exist_ok=True)
        devices_file = storage_dir / "devices.json"
        config_file = storage_dir / "config.json"

        # Seed 30 existing devices and initial config
        seeded: Dict[str, Dict[str, Any]] = {
            f"seed_{i}": {
                "device_id": f"seed_{i}",
                "name": f"Seed Device {i}",
                "is_paired": True,
            }
            for i in range(30)
        }
        devices_file.write_text(json.dumps(seeded), encoding="utf-8")
        config_file.write_text(json.dumps({"initial_key": "initial_value"}), encoding="utf-8")

        # Instantiate totally uninitialized backend
        backend = JsonStorageBackend(storage_dir)
        assert backend._initialized is False

        # Build 150 concurrent operations
        tasks = []

        # 50 tasks saving new devices
        for i in range(50):
            tasks.append(
                backend.save_device(
                    f"new_node_{i}",
                    {"device_id": f"new_node_{i}", "name": f"New Node {i}", "is_paired": True},
                )
            )

        # 30 tasks reading devices
        for i in range(30):
            tasks.append(backend.get_device(f"seed_{i}"))

        # 20 tasks listing devices
        for _ in range(20):
            tasks.append(backend.list_devices())

        # 20 tasks updating config
        for i in range(20):
            tasks.append(backend.set_config(f"cfg_key_{i}", f"value_{i}"))

        # 10 tasks reading config
        for _ in range(10):
            tasks.append(backend.get_config("initial_key"))

        # 10 tasks deleting specific seeded devices
        for i in range(10):
            tasks.append(backend.delete_device(f"seed_{i}"))

        # Shuffle execution order to maximize race window
        random.shuffle(tasks)

        # Execute concurrent stampede
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Assert no exceptions occurred
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                pytest.fail(f"Task {idx} failed with unexpected exception: {res}")

        # Verify disk state consistency
        disk_devices = json.loads(devices_file.read_text(encoding="utf-8"))
        # 30 seeded - 10 deleted + 50 new = 70 total devices
        assert len(disk_devices) == 70, f"Expected 70 devices on disk, got {len(disk_devices)}"

        # Deleted devices must NOT exist
        for i in range(10):
            assert f"seed_{i}" not in disk_devices

        # Remaining seeded devices must exist
        for i in range(10, 30):
            assert f"seed_{i}" in disk_devices
            assert disk_devices[f"seed_{i}"]["name"] == f"Seed Device {i}"

        # All 50 new devices must exist
        for i in range(50):
            assert f"new_node_{i}" in disk_devices
            assert disk_devices[f"new_node_{i}"]["name"] == f"New Node {i}"

        # Verify config file on disk
        disk_config = json.loads(config_file.read_text(encoding="utf-8"))
        assert disk_config["initial_key"] == "initial_value"
        for i in range(20):
            assert disk_config[f"cfg_key_{i}"] == f"value_{i}"

    async def test_concurrent_read_only_operations_on_uninitialized_backend(self, tmp_path: Path) -> None:
        """Verify concurrent read-only operations (list_devices, get_device) safely initialize."""
        storage_dir = tmp_path / "storage_reads"
        storage_dir.mkdir(parents=True, exist_ok=True)
        devices_file = storage_dir / "devices.json"

        seeded = {
            f"ro_{i}": {"device_id": f"ro_{i}", "name": f"RO Device {i}", "is_paired": True}
            for i in range(20)
        }
        devices_file.write_text(json.dumps(seeded), encoding="utf-8")

        backend = JsonStorageBackend(storage_dir)

        # 40 concurrent readers
        tasks = []
        for i in range(20):
            tasks.append(backend.get_device(f"ro_{i}"))
        for _ in range(20):
            tasks.append(backend.list_devices())

        results = await asyncio.gather(*tasks)

        # First 20 are get_device results
        for i in range(20):
            dev = results[i]
            assert dev is not None
            assert dev["name"] == f"RO Device {i}"

        # Next 20 are list_devices results
        for i in range(20, 40):
            dev_list = results[i]
            assert len(dev_list) == 20

    async def test_corrupt_devices_file_handling_under_uninitialized_save(self, tmp_path: Path) -> None:
        """Verify that uninitialized save_device handles corrupt devices.json by backing it up."""
        storage_dir = tmp_path / "storage_corrupt"
        storage_dir.mkdir(parents=True, exist_ok=True)
        devices_file = storage_dir / "devices.json"

        # Write corrupted partial JSON
        devices_file.write_text('{"device_1": {"device_id": "device_1", "corrupt_trailing": ', encoding="utf-8")

        backend = JsonStorageBackend(storage_dir)
        assert backend._initialized is False

        # save_device should trigger _ensure_initialized, detect corruption, backup, and save new device cleanly
        await backend.save_device("clean_dev", {"device_id": "clean_dev", "name": "Clean Recovery"})

        assert backend._initialized is True
        # Check corrupt backup was created
        backup_files = list(storage_dir.glob("devices.corrupt.*.bak"))
        assert len(backup_files) >= 1

        # Check devices.json now contains clean_dev
        disk_content = json.loads(devices_file.read_text(encoding="utf-8"))
        assert "clean_dev" in disk_content
        assert disk_content["clean_dev"]["name"] == "Clean Recovery"

    async def test_empty_devices_file_uninitialized_initialization(self, tmp_path: Path) -> None:
        """Verify empty string or whitespace devices.json safely initializes to empty dict."""
        storage_dir = tmp_path / "storage_empty"
        storage_dir.mkdir(parents=True, exist_ok=True)
        devices_file = storage_dir / "devices.json"
        devices_file.write_text("   \n\t  ", encoding="utf-8")

        backend = JsonStorageBackend(storage_dir)
        devices = await backend.list_devices()
        assert devices == []

        await backend.save_device("first_dev", {"device_id": "first_dev", "name": "First"})
        disk_data = json.loads(devices_file.read_text(encoding="utf-8"))
        assert "first_dev" in disk_data

    async def test_delayed_io_initialization_prevents_double_load(self, tmp_path: Path) -> None:
        """Verify that under artificial disk delay, 50 concurrent tasks invoke _load_json_file only once."""
        storage_dir = tmp_path / "storage_delayed"
        storage_dir.mkdir(parents=True, exist_ok=True)
        devices_file = storage_dir / "devices.json"
        devices_file.write_text(json.dumps({"dev_init": {"device_id": "dev_init", "name": "Initial"}}), encoding="utf-8")

        backend = JsonStorageBackend(storage_dir)

        real_load = backend._load_json_file
        load_count = 0

        def slow_load(path: Path) -> Dict[str, Any]:
            nonlocal load_count
            load_count += 1
            time.sleep(0.01)  # Artificial disk delay in thread
            return real_load(path)

        backend._load_json_file = slow_load  # type: ignore[assignment]

        # 50 concurrent tasks hitting uninitialized backend
        tasks = []
        for i in range(25):
            tasks.append(backend.get_device("dev_init"))
            tasks.append(backend.save_device(f"delay_dev_{i}", {"device_id": f"delay_dev_{i}"}))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                pytest.fail(f"Unexpected error in delayed load stress: {r}")

        # _load_json_file should be called exactly twice: once for devices.json, once for config.json
        assert load_count == 2, f"Expected exactly 2 file loads (devices + config), got {load_count}"
        assert backend._initialized is True

        # All 25 saved devices + initial device must be present
        all_devs = await backend.list_devices()
        assert len(all_devs) == 26


class TestExhaustiveFiveStateRandomWalk:
    """Intensive fuzzer exercising transitions among PAIRED, DISCONNECTED, ERROR, CONNECTING, PAIRING."""

    async def test_random_walk_fuzzer_five_states(self) -> None:
        """Execute 2,000 randomized state transitions across the 5 core states without any failure."""
        target_subgraph = {
            DeviceState.PAIRED,
            DeviceState.DISCONNECTED,
            DeviceState.ERROR,
            DeviceState.CONNECTING,
            DeviceState.PAIRING,
        }

        sm = DeviceStateMachine(device_id="fuzz_node", initial_state=DeviceState.DISCONNECTED)

        counts = {
            ("PAIRED", "DISCONNECTED"): 0,
            ("PAIRED", "ERROR"): 0,
            ("PAIRING", "ERROR"): 0,
            ("CONNECTING", "PAIRING"): 0,
            ("PAIRING", "PAIRED"): 0,
            ("ERROR", "DISCONNECTED"): 0,
            ("ERROR", "CONNECTING"): 0,
        }

        steps = 5000
        for step in range(steps):
            current = sm.current_state
            # Allowed next states within our 5 target states
            allowed = [s for s in sm.VALID_TRANSITIONS[current] if s in target_subgraph]
            assert len(allowed) > 0, f"No legal transitions from {current} within target subgraph"

            next_state = random.choice(allowed)
            transition_key = (current.value, next_state.value)

            # Assert transition is legal
            assert sm.can_transition_to(next_state) is True

            # Execute transition
            await sm.transition_to(next_state, reason=f"fuzz step {step}")
            assert sm.current_state == next_state

            if transition_key in counts:
                counts[transition_key] += 1

        # Confirm critical milestone 17 transitions were heavily exercised
        assert counts[("PAIRED", "DISCONNECTED")] > 30, f"PAIRED -> DISCONNECTED count: {counts[('PAIRED', 'DISCONNECTED')]}"
        assert counts[("PAIRED", "ERROR")] > 30, f"PAIRED -> ERROR count: {counts[('PAIRED', 'ERROR')]}"
        assert counts[("PAIRING", "ERROR")] > 30, f"PAIRING -> ERROR count: {counts[('PAIRING', 'ERROR')]}"
        assert len(sm.history) == steps + 1


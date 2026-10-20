"""tests.unit.test_daemon_volume_runcommand_ipc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Milestone 25: JSON-RPC IPC server/client integration, serialization,
error code mapping (-32015, -32016), parameter validation, and client methods for
SystemVolume and RunCommand features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pengulink.core.config import PenguLinkConfig
from pengulink.core.exceptions import (
    AudioBackendError,
    CommandExecutionError,
    ConnectionClosedError,
    IPCError,
    StorageWriteError,
)
from pengulink.daemon.ipc import (
    AUDIO_SYSTEM_VOLUME_ERROR,
    DEVICE_NOT_CONNECTED,
    RUN_COMMAND_ERROR,
    PenguLinkIPCClient,
    PenguLinkIPCServer,
)
from pengulink.daemon.service import PenguLinkDaemon
from pengulink.services.audio import AudioError, AudioSink, MockAudioSinkBackend

# =====================================================================
# 1. IPC Server Error Code Mapping & Serialization Unit Tests
# =====================================================================


class TestIPCServerVolumeAndRunCommandDispatch:
    """Tests for PenguLinkIPCServer handling systemvolume and runcommand requests."""

    async def test_audio_error_code_mapping_32015(self, tmp_path: Path) -> None:
        """Test audio errors map strictly to AUDIO_SYSTEM_VOLUME_ERROR (-32015)."""
        socket_path = tmp_path / "test_audio_err.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            if method == "systemvolume_set_volume":
                raise AudioError("Sound server daemon failed")
            elif method == "systemvolume_get_sinks":
                raise AudioBackendError("Backend pactl exited with error")
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)

            with pytest.raises(IPCError) as exc_info:
                await client.systemvolume_set_volume("sink1", 50)
            assert exc_info.value.code == AUDIO_SYSTEM_VOLUME_ERROR
            assert "Sound server daemon failed" in str(exc_info.value)

            with pytest.raises(IPCError) as exc_info:
                await client.systemvolume_get_sinks()
            assert exc_info.value.code == AUDIO_SYSTEM_VOLUME_ERROR
            assert "Backend pactl exited with error" in str(exc_info.value)
        finally:
            await server.stop()

    async def test_run_command_error_code_mapping_32016(self, tmp_path: Path) -> None:
        """Test command errors and storage errors map strictly to RUN_COMMAND_ERROR (-32016)."""
        socket_path = tmp_path / "test_cmd_err.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            if method == "runcommand_exec":
                raise CommandExecutionError("Process terminated by signal")
            elif method == "runcommand_get":
                raise KeyError(f"Command '{params.get('key')}' not found")
            elif method == "runcommand_add":
                raise StorageWriteError("Failed writing commands storage file")
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)

            with pytest.raises(IPCError) as exc_info:
                await client.runcommand_exec("cmd_lock")
            assert exc_info.value.code == RUN_COMMAND_ERROR
            assert "Process terminated by signal" in str(exc_info.value)

            with pytest.raises(IPCError) as exc_info:
                await client.runcommand_get("cmd_missing")
            assert exc_info.value.code == RUN_COMMAND_ERROR
            assert "not found" in str(exc_info.value).lower()

            with pytest.raises(IPCError) as exc_info:
                await client.runcommand_add("New Cmd", "ls -la")
            assert exc_info.value.code == RUN_COMMAND_ERROR
            assert "Failed writing" in str(exc_info.value)
        finally:
            await server.stop()

    async def test_disconnected_device_error_mapping(self, tmp_path: Path) -> None:
        """Test remote volume commands map ConnectionClosedError to DEVICE_NOT_CONNECTED."""
        socket_path = tmp_path / "test_remote_vol_err.sock"

        async def handler(method: str, params: dict[str, Any]) -> Any:
            if method == "systemvolume_set_remote_volume":
                raise ConnectionClosedError("Device 'peer1' is not connected")
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc_info:
                await client.systemvolume_set_remote_volume("peer1", "Speaker", 45000)
            assert exc_info.value.code == DEVICE_NOT_CONNECTED
            assert "not connected" in str(exc_info.value).lower()
        finally:
            await server.stop()


# =====================================================================
# 2. Client Wrapper Methods & Parameter Validation Tests
# =====================================================================


class TestIPCClientVolumeAndRunCommandMethods:
    """Test all PenguLinkIPCClient wrapper methods for correct JSON-RPC call payloads."""

    async def test_systemvolume_client_methods(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_client_vol.sock"
        received_calls: list[tuple[str, dict[str, Any]]] = []

        async def handler(method: str, params: dict[str, Any]) -> Any:
            received_calls.append((method, params))
            if method == "systemvolume_get_sinks":
                return [
                    {
                        "name": "sink_1",
                        "description": "Built-in Audio",
                        "volume": 32768,
                        "volume_percent": 50,
                        "muted": False,
                        "enabled": True,
                    }
                ]
            elif (
                method == "systemvolume_set_volume"
                or method == "systemvolume_set_muted"
                or method == "systemvolume_set_default"
            ):
                return {"success": True}
            elif method == "systemvolume_get_remote_sinks":
                return [
                    {
                        "name": "remote_sink_1",
                        "description": "Phone Speaker",
                        "volume": 15,
                        "volume_percent": 60,
                        "muted": False,
                        "enabled": True,
                    }
                ]
            elif (
                method == "systemvolume_set_remote_volume"
                or method == "systemvolume_set_remote_muted"
                or method == "systemvolume_set_remote_default"
                or method == "systemvolume_request_remote_sinks"
            ):
                return {"success": True}
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)

            # Local volume get sinks
            sinks = await client.systemvolume_get_sinks()
            assert len(sinks) == 1
            assert sinks[0]["name"] == "sink_1"

            # Local set volume
            res = await client.systemvolume_set_volume("sink_1", 45000)
            assert res.get("success") is True

            # Local set muted
            res = await client.systemvolume_set_muted("sink_1", True)
            assert res.get("success") is True

            # Local set default
            res = await client.systemvolume_set_default("sink_1")
            assert res.get("success") is True

            # Remote get sinks
            rem_sinks = await client.systemvolume_get_remote_sinks("phone_id")
            assert len(rem_sinks) == 1
            assert rem_sinks[0]["name"] == "remote_sink_1"

            # Remote set volume
            res = await client.systemvolume_set_remote_volume(
                "phone_id", "remote_sink_1", 20
            )
            assert res.get("success") is True

            # Remote set muted
            res = await client.systemvolume_set_remote_muted(
                "phone_id", "remote_sink_1", True
            )
            assert res.get("success") is True

            # Remote set default
            res = await client.systemvolume_set_remote_default(
                "phone_id", "remote_sink_1"
            )
            assert res.get("success") is True

            # Remote request sinks
            res = await client.systemvolume_request_remote_sinks("phone_id")
            assert res.get("success") is True

            # Test aliases
            res_alias_sinks = await client.list_remote_audio_sinks("phone_id")
            assert len(res_alias_sinks) == 1

            res_alias_vol = await client.set_remote_audio_volume(
                "phone_id", "remote_sink_1", 18
            )
            assert res_alias_vol.get("success") is True

            res_alias_mute = await client.set_remote_audio_mute(
                "phone_id", "remote_sink_1", False
            )
            assert res_alias_mute.get("success") is True

            res_alias_req = await client.request_remote_audio_sinks("phone_id")
            assert res_alias_req.get("success") is True

            # Verify received call method names and parameters
            method_names = [c[0] for c in received_calls]
            assert "systemvolume_get_sinks" in method_names
            assert "systemvolume_set_volume" in method_names
            assert "systemvolume_set_muted" in method_names
            assert "systemvolume_set_default" in method_names
            assert "systemvolume_get_remote_sinks" in method_names
        finally:
            await server.stop()

    async def test_runcommand_client_methods(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_client_cmd.sock"
        received_calls: list[tuple[str, dict[str, Any]]] = []

        async def handler(method: str, params: dict[str, Any]) -> Any:
            received_calls.append((method, params))
            if method == "runcommand_list":
                return [
                    {
                        "key": "k1",
                        "name": "Lock",
                        "command": "loginctl lock-session",
                        "enabled": True,
                    }
                ]
            elif method == "runcommand_get":
                return {
                    "key": params["key"],
                    "name": "Lock",
                    "command": "loginctl lock-session",
                }
            elif method == "runcommand_add":
                return {
                    "key": "k2",
                    "name": params["name"],
                    "command": params["command"],
                }
            elif method == "runcommand_remove":
                return {"success": True}
            elif method == "runcommand_exec":
                return {
                    "success": True,
                    "id": "1",
                    "key": params["key"],
                    "exit_code": 0,
                    "output": "ok",
                }
            elif method == "runcommand_stop":
                return {"success": True}
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)

            # List
            cmds = await client.runcommand_list()
            assert len(cmds) == 1
            assert cmds[0]["key"] == "k1"

            # Get
            cmd = await client.runcommand_get("k1")
            assert cmd["name"] == "Lock"

            # Add
            added = await client.runcommand_add("Sleep", "systemctl suspend")
            assert added["key"] == "k2"

            # Remove
            res_rm = await client.runcommand_remove("k2")
            assert res_rm.get("success") is True

            # Exec
            res_exec = await client.runcommand_exec("k1")
            assert res_exec.get("success") is True
            assert res_exec.get("output") == "ok"

            # Stop
            res_stop = await client.runcommand_stop("1")
            assert res_stop.get("success") is True

            # Aliases
            alias_list = await client.list_local_commands()
            assert len(alias_list) == 1
            alias_exec = await client.execute_local_command("k1")
            assert alias_exec.get("success") is True
        finally:
            await server.stop()


# =====================================================================
# 3. End-to-End Daemon Service & IPC Integration Tests
# =====================================================================


class TestDaemonVolumeAndRunCommandIPCIntegration:
    """End-to-end integration tests between PenguLinkDaemon and PenguLinkIPCClient."""

    @pytest.fixture
    def test_daemon_config(self, tmp_path: Path) -> PenguLinkConfig:
        cfg = PenguLinkConfig()
        cfg.storage.config_dir = tmp_path / "config"
        cfg.storage.data_dir = tmp_path / "data"
        cfg.storage.runtime_dir = tmp_path / "run"
        cfg.device.device_id = "testdaemon1234567890abcdef123456"
        cfg.device.device_name = "PenguLink-Daemon-Test"
        cfg.network.udp_port = 17169
        cfg.network.tcp_port_min = 17210
        cfg.network.tcp_port_max = 17215
        cfg.ensure_directories()
        return cfg

    @pytest.fixture
    def mock_backend(self) -> MockAudioSinkBackend:
        return MockAudioSinkBackend(
            initial_sinks=[
                AudioSink(
                    name="alsa_output.pci-0000_00_1f.3.analog-stereo",
                    description="Built-in Analog Stereo",
                    volume=45875,
                    max_volume=65536,
                    muted=False,
                    enabled=True,
                ),
                AudioSink(
                    name="bluez_output.00_11_22_33_44_55.a2dp-sink",
                    description="Bluetooth Headphones",
                    volume=32768,
                    max_volume=65536,
                    muted=True,
                    enabled=False,
                ),
            ]
        )

    async def test_daemon_systemvolume_ipc_roundtrip(
        self,
        test_daemon_config: PenguLinkConfig,
        tmp_path: Path,
        mock_backend: MockAudioSinkBackend,
    ) -> None:
        """Verify real daemon routes systemvolume IPC requests cleanly."""
        socket_path = tmp_path / "daemon_vol.sock"

        daemon = PenguLinkDaemon(
            config=test_daemon_config,
            socket_path=socket_path,
            audio_backend=mock_backend,
        )

        await daemon.start()

        try:
            client = PenguLinkIPCClient(socket_path)

            # 1. Get local sinks
            sinks = await client.systemvolume_get_sinks()
            assert len(sinks) == 2
            assert sinks[0]["name"] == "alsa_output.pci-0000_00_1f.3.analog-stereo"
            assert sinks[0]["volume"] == 45875
            assert sinks[0]["enabled"] is True

            # 2. Set volume
            res = await client.systemvolume_set_volume(
                "alsa_output.pci-0000_00_1f.3.analog-stereo", 52428
            )
            assert res.get("success") is True

            # Check update reflected in backend
            sink0 = await daemon.audio_service.get_sink(
                "alsa_output.pci-0000_00_1f.3.analog-stereo"
            )
            assert sink0 is not None
            assert sink0.volume == 52428

            # 3. Set muted
            res_mute = await client.systemvolume_set_muted(
                "alsa_output.pci-0000_00_1f.3.analog-stereo", True
            )
            assert res_mute.get("success") is True
            sink0 = await daemon.audio_service.get_sink(
                "alsa_output.pci-0000_00_1f.3.analog-stereo"
            )
            assert sink0 is not None
            assert sink0.muted is True

            # 4. Set default sink
            res_def = await client.systemvolume_set_default(
                "bluez_output.00_11_22_33_44_55.a2dp-sink"
            )
            assert res_def.get("success") is True
            sink1 = await daemon.audio_service.get_sink(
                "bluez_output.00_11_22_33_44_55.a2dp-sink"
            )
            assert sink1 is not None
            assert sink1.enabled is True

            # 5. Remote system volume on unknown device returns empty list
            rem_sinks = await client.systemvolume_get_remote_sinks("unknown_phone_999")
            assert rem_sinks == []

        finally:
            await daemon.stop()

    async def test_daemon_runcommand_ipc_roundtrip(
        self, test_daemon_config: PenguLinkConfig, tmp_path: Path
    ) -> None:
        """Verify real daemon routes runcommand IPC requests cleanly."""
        socket_path = tmp_path / "daemon_cmd.sock"

        daemon = PenguLinkDaemon(
            config=test_daemon_config,
            socket_path=socket_path,
        )

        await daemon.start()

        try:
            client = PenguLinkIPCClient(socket_path)

            # 1. Add command
            added = await client.runcommand_add("Echo Hello", "echo 'Hello PenguLink'")
            assert "key" in added
            cmd_key = added["key"]
            assert added["name"] == "Echo Hello"

            # 2. Get command
            fetched = await client.runcommand_get(cmd_key)
            assert fetched["key"] == cmd_key
            assert fetched["name"] == "Echo Hello"

            # 3. List commands
            cmd_list = await client.runcommand_list()
            assert any(c["key"] == cmd_key for c in cmd_list)

            # 4. Execute command
            res_exec = await client.runcommand_exec(cmd_key)
            assert res_exec.get("success") is True
            assert res_exec.get("exit_code") == 0
            assert "Hello PenguLink" in res_exec.get("output", "")

            # 5. Remove command
            res_rm = await client.runcommand_remove(cmd_key)
            assert res_rm.get("success") is True

            # 6. Verify non-existent key raises error mapped to -32016
            with pytest.raises(IPCError) as exc_info:
                await client.runcommand_get(cmd_key)
            assert exc_info.value.code == RUN_COMMAND_ERROR or str(
                RUN_COMMAND_ERROR
            ) in str(exc_info.value)

        finally:
            await daemon.stop()

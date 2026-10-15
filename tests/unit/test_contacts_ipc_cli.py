"""Unit tests for Contacts JSON-RPC IPC server/client and Click/Rich CLI."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from pengulink.cli.main import cli
from pengulink.core.exceptions import (
    ContactNotFoundError,
    DeviceNotFoundError,
    IPCError,
    SyncTimeoutError,
)
from pengulink.daemon.ipc import (
    CONTACTS_ERROR,
    DEVICE_NOT_FOUND,
    PenguLinkIPCClient,
    PenguLinkIPCServer,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner(env={"COLUMNS": "160"})


class TestContactsIPCServerAndClient:
    async def test_ipc_contacts_list_and_search(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_contacts.sock"

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            if method == "contacts_list":
                assert params["device_id"] == "phone_1"
                assert params["offset"] == 0
                return {
                    "device_id": "phone_1",
                    "total": 1,
                    "contacts": [
                        {
                            "uid": "c1",
                            "name": "Alice Wonderland",
                            "phones": [{"number": "+15550100", "normalized": "+15550100", "type": "CELL"}],
                            "emails": [{"email": "alice@wonder.land", "type": "HOME"}],
                            "organization": "Fantasy",
                        }
                    ],
                }
            elif method == "contacts_search":
                assert params["query"] == "Alice"
                return {
                    "device_id": "phone_1",
                    "query": "Alice",
                    "count": 1,
                    "matches": [{"uid": "c1", "name": "Alice Wonderland"}],
                }
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            # List
            list_res = await client.contacts_list("phone_1")
            assert list_res["total"] == 1
            assert list_res["contacts"][0]["name"] == "Alice Wonderland"

            # Search
            search_res = await client.contacts_search("phone_1", "Alice")
            assert search_res["count"] == 1
            assert search_res["matches"][0]["uid"] == "c1"
        finally:
            await server.stop()

    async def test_ipc_contacts_sync_and_get(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_sync.sock"

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            if method == "contacts_sync":
                assert params["device_id"] == "phone_1"
                assert params["force"] is True
                return {
                    "device_id": "phone_1",
                    "status": "synchronized",
                    "total": 10,
                    "added": 2,
                    "updated": 1,
                    "deleted": 0,
                    "duration": 0.45,
                }
            elif method == "contacts_get":
                if params["uid"] == "missing":
                    raise ContactNotFoundError("UID not found")
                return {
                    "device_id": "phone_1",
                    "contact": {"uid": "c1", "name": "Alice", "raw_vcard": "VCARD..."},
                }
            raise KeyError(method)

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            # Sync
            sync_res = await client.contacts_sync("phone_1", force=True)
            assert sync_res["status"] == "synchronized"
            assert sync_res["added"] == 2

            # Get
            get_res = await client.contacts_get("phone_1", "c1")
            assert get_res["contact"]["name"] == "Alice"

            # Get not found error mapping
            with pytest.raises(IPCError) as exc_info:
                await client.contacts_get("phone_1", "missing")
            assert str(CONTACTS_ERROR) in str(exc_info.value)
        finally:
            await server.stop()

    async def test_ipc_sync_timeout_error_code(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "test_timeout.sock"

        async def handler(method: str, params: Dict[str, Any]) -> Any:
            raise SyncTimeoutError("Timed out waiting for peer vCards")

        server = PenguLinkIPCServer(socket_path, handler=handler)
        await server.start()

        try:
            client = PenguLinkIPCClient(socket_path)
            with pytest.raises(IPCError) as exc_info:
                await client.contacts_sync("phone_1")
            assert str(CONTACTS_ERROR) in str(exc_info.value)
            assert "Timed out" in str(exc_info.value)
        finally:
            await server.stop()


class TestContactsCLI:
    def test_cli_contacts_list_rich_table(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_list = {
            "device_id": "phone_123",
            "total": 1,
            "contacts": [
                {
                    "uid": "101",
                    "name": "Sherlock Holmes",
                    "phones": [{"number": "+44 20 7946 0958", "normalized": "+442079460958", "type": "CELL"}],
                    "emails": [{"email": "sherlock@bakerst.co.uk", "type": "WORK"}],
                    "organization": "Baker Street Consulting",
                }
            ],
        }

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_list):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "contacts", "list", "phone_123"])
            assert res.exit_code == 0
            assert "Contacts for phone_123" in res.output
            assert "Sherlock Holmes" in res.output
            assert "+44 20 7946 0958" in res.output
            assert "Baker Street Consulting" in res.output

    def test_cli_contacts_list_pagination(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        captured_call: Dict[str, Any] = {}

        def mock_call_sync(method: str, params: Dict[str, Any], timeout: float = 10.0) -> Any:
            captured_call["method"] = method
            captured_call["params"] = params
            return {"device_id": "phone_123", "total": 0, "contacts": []}

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=mock_call_sync):
            res = cli_runner.invoke(
                cli, ["-s", str(mock_socket), "contacts", "list", "phone_123", "--limit", "5", "--offset", "10"]
            )
            assert res.exit_code == 0
            assert captured_call["params"]["limit"] == 5
            assert captured_call["params"]["offset"] == 10

    def test_cli_contacts_list_empty(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
            return_value={"device_id": "phone_123", "total": 0, "contacts": []},
        ):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "contacts", "list", "phone_123"])
            assert res.exit_code == 0
            assert "No contacts found for device 'phone_123'" in res.output

    def test_cli_contacts_list_json(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_list = {"device_id": "phone_123", "total": 0, "contacts": []}
        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_list):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "contacts", "list", "phone_123", "--json"])
            assert res.exit_code == 0
            parsed = json.loads(res.output)
            assert parsed["device_id"] == "phone_123"

    def test_cli_contacts_search_name_match(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_search = {
            "device_id": "phone_123",
            "query": "John",
            "count": 1,
            "matches": [
                {
                    "uid": "102",
                    "name": "Dr. John Watson",
                    "phones": [{"number": "555-0102", "normalized": "5550102", "type": "WORK"}],
                    "emails": [],
                    "organization": "Medical Corps",
                }
            ],
        }

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_search):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "contacts", "search", "phone_123", "John"])
            assert res.exit_code == 0
            assert "Search Results for 'John'" in res.output
            assert "Dr. John Watson" in res.output

    def test_cli_contacts_search_no_match(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_search = {"device_id": "phone_123", "query": "Unknown", "count": 0, "matches": []}
        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_search):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "contacts", "search", "phone_123", "Unknown"])
            assert res.exit_code == 0
            assert "No contacts matching 'Unknown' found" in res.output

    def test_cli_contacts_search_json(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_search = {"device_id": "phone_123", "query": "Watson", "count": 0, "matches": []}
        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_search):
            res = cli_runner.invoke(
                cli, ["-s", str(mock_socket), "contacts", "search", "phone_123", "Watson", "--json"]
            )
            assert res.exit_code == 0
            parsed = json.loads(res.output)
            assert parsed["query"] == "Watson"

    def test_cli_contacts_view_rich_card(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_get = {
            "device_id": "phone_123",
            "contact": {
                "uid": "101",
                "name": "Irene Adler",
                "organization": "Opera",
                "title": "Prima Donna",
                "phones": [{"number": "+1-555-0199", "normalized": "+15550199", "type": "CELL"}],
                "emails": [{"email": "irene@adler.org", "type": "HOME"}],
                "note": "A woman of intellect",
                "timestamp": 1712345678000,
                "cached_at": 1712345680.0,
            },
        }

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_get):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "contacts", "view", "phone_123", "101"])
            assert res.exit_code == 0
            assert "Irene Adler" in res.output
            assert "Prima Donna at Opera" in res.output
            assert "+1-555-0199" in res.output
            assert "irene@adler.org" in res.output
            assert "A woman of intellect" in res.output

    def test_cli_contacts_view_raw_vcard(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_get = {
            "device_id": "phone_123",
            "contact": {
                "uid": "101",
                "name": "Irene Adler",
                "raw_vcard": "BEGIN:VCARD\r\nFN:Irene Adler\r\nEND:VCARD\r\n",
            },
        }

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_get):
            res = cli_runner.invoke(
                cli, ["-s", str(mock_socket), "contacts", "view", "phone_123", "101", "--raw"]
            )
            assert res.exit_code == 0
            assert "BEGIN:VCARD" in res.output
            assert "FN:Irene Adler" in res.output

    def test_cli_contacts_view_not_found(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
            side_effect=IPCError("Contact not found"),
        ):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "contacts", "view", "phone_123", "missing"])
            assert res.exit_code == 1
            assert "Contact not found" in res.output

    def test_cli_contacts_sync_success_summary(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        sample_sync = {
            "device_id": "phone_123",
            "status": "synchronized",
            "total": 42,
            "added": 5,
            "updated": 2,
            "deleted": 1,
            "duration": 0.88,
        }

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", return_value=sample_sync):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "contacts", "sync", "phone_123"])
            assert res.exit_code == 0
            assert "Contacts Synchronization" in res.output
            assert "42" in res.output
            assert "+5 added" in res.output
            assert "~2 updated" in res.output
            assert "-1 deleted" in res.output

    def test_cli_contacts_sync_force_flag(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        captured_call: Dict[str, Any] = {}

        def mock_call_sync(method: str, params: Dict[str, Any], timeout: float = 60.0) -> Any:
            captured_call["method"] = method
            captured_call["params"] = params
            return {"device_id": "phone_123", "total": 1, "added": 1, "updated": 0, "deleted": 0, "duration": 0.1}

        with patch("pengulink.daemon.ipc.PenguLinkIPCClient.call_sync", side_effect=mock_call_sync):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "contacts", "sync", "phone_123", "--force"])
            assert res.exit_code == 0
            assert captured_call["params"]["force"] is True

    def test_cli_contacts_sync_offline_error(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        with patch(
            "pengulink.daemon.ipc.PenguLinkIPCClient.call_sync",
            side_effect=IPCError("Device is not connected"),
        ):
            res = cli_runner.invoke(cli, ["-s", str(mock_socket), "contacts", "sync", "offline_device"])
            assert res.exit_code == 1
            assert "Device is not connected" in res.output

    def test_cli_contacts_validation_errors(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        mock_socket = tmp_path / "daemon.sock"
        mock_socket.touch()

        # Empty device ID
        res = cli_runner.invoke(cli, ["-s", str(mock_socket), "contacts", "list", "   "])
        assert res.exit_code != 0

        # Negative offset
        res = cli_runner.invoke(cli, ["-s", str(mock_socket), "contacts", "list", "dev1", "--offset", "-5"])
        assert res.exit_code != 0

        # Negative limit
        res = cli_runner.invoke(cli, ["-s", str(mock_socket), "contacts", "list", "dev1", "--limit", "-1"])
        assert res.exit_code != 0

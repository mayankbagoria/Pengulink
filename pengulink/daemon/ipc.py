"""pengulink.daemon.ipc
~~~~~~~~~~~~~~~~~~~~
Unix Domain Socket JSON-RPC 2.0 server and client for PenguLink daemon.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import logging
import os
from pathlib import Path
import socket
import struct
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

from pengulink.core.exceptions import (
    ActionUnsupportedError,
    AlertError,
    AudioBackendError,
    AudioUnavailableError,
    CommandExecutionError,
    CommandNotFoundError,
    ConnectionClosedError,
    ConnectionTimeoutError,
    ContactNotFoundError,
    ContactsError,
    DaemonAlreadyRunningError,
    DaemonNotRunningError,
    DeviceNotFoundError,
    InvalidStateTransitionError,
    IPCError,
    MprisAdapterError,
    MprisError,
    MprisPlayerNotFoundError,
    PairingError,
    PairingRejectedError,
    PairingTimestampExpiredError,
    PathTraversalError,
    PayloadTransferError,
    PenguLinkError,
    RemoteCommandError,
    RemoteCommandNotFoundError,
    RunCommandError,
    ShareError,
    StorageCorruptedError,
    StorageError,
    StorageWriteError,
    SyncTimeoutError,
    TransferError,
)
from pengulink.plugins.clipboard.backend import ClipboardError, ClipboardUnavailableError
from pengulink.services.audio import AudioError

logger = logging.getLogger("pengulink.daemon.ipc")

JSONRPC_VERSION = "2.0"
STREAM_LIMIT_BYTES = 1_048_576  # 1 MB line guard limit

# Standard JSON-RPC 2.0 Error Codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# PenguLink Domain Error Codes
APPLICATION_ERROR = -32000
DEVICE_NOT_FOUND = -32001
DEVICE_NOT_CONNECTED = -32002
PAIRING_ERROR = -32003
INVALID_STATE_TRANSITION = -32004
NOTIFICATION_NOT_FOUND = -32005
DAEMON_NOT_READY = -32006
OPERATION_TIMEOUT = -32007
DAEMON_ALREADY_RUNNING = -32008
CLIPBOARD_ERROR = -32009
ALERT_ERROR = -32010
SHARE_ERROR = -32011
CONTACTS_ERROR = -32012
MPRIS_ERROR = -32013
REMOTE_COMMANDS_ERROR = -32014
AUDIO_SYSTEM_VOLUME_ERROR = -32015
RUN_COMMAND_ERROR = -32016





def resolve_socket_path(socket_path: Optional[Union[Path, str]] = None) -> Path:
    """Resolve Unix domain socket path using explicit arg, env var, or XDG runtime/config hierarchy."""
    if socket_path is not None:
        return Path(socket_path)

    env_path = os.environ.get("PENGULINK_SOCKET_PATH") or os.environ.get("PENGULINK_IPC_SOCKET")
    if env_path:
        return Path(env_path)

    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime:
        return Path(xdg_runtime) / "pengulink" / "daemon.sock"

    return Path.home() / ".config" / "pengulink" / "daemon.sock"


class PenguLinkIPCServer:
    """Unix Domain Socket server implementing JSON-RPC 2.0 with POSIX user isolation."""

    def __init__(
        self,
        socket_path: Union[Path, str],
        handler: Optional[Callable[[str, Dict[str, Any]], Coroutine[Any, Any, Any]]] = None,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.handler = handler
        self._server: Optional[asyncio.Server] = None
        self._running: bool = False
        self._client_tasks: set[asyncio.Task[Any]] = set()
        self._client_writers: set[asyncio.StreamWriter] = set()
        self._handlers: Dict[str, Callable[[Any], Any]] = {}

    def register_handler(self, method: str, handler: Callable[[Any], Any]) -> None:
        """Register a handler callback for a specific JSON-RPC method."""
        self._handlers[method] = handler

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Create socket directory, probe for active daemon collisions, unlink stale socket, and bind server."""
        if self._running:
            return

        # 1. Enforce 0700 on parent directory
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.socket_path.parent, 0o700)
        except OSError:
            pass

        # 2. Collision detection & stale socket recovery: probe existing socket before unlinking
        if self.socket_path.exists():
            is_active = False
            try:
                probe_reader, probe_writer = await asyncio.wait_for(
                    asyncio.open_unix_connection(str(self.socket_path)),
                    timeout=0.3,
                )
                is_active = True
                try:
                    probe_writer.close()
                    await probe_writer.wait_closed()
                except Exception:
                    pass
            except (ConnectionRefusedError, FileNotFoundError):
                is_active = False
            except (asyncio.TimeoutError, TimeoutError):
                is_active = False
            except OSError:
                is_active = False

            if is_active:
                raise DaemonAlreadyRunningError(
                    f"Another PenguLink daemon instance is already active on {self.socket_path}"
                )

            # Unlink stale socket file
            try:
                self.socket_path.unlink()
            except OSError as exc:
                logger.warning("Could not unlink stale socket %s: %s", self.socket_path, exc)

        # 3. Bind Unix Domain Socket server with 1 MB buffer limit
        try:
            self._server = await asyncio.start_unix_server(
                self._handle_client,
                path=str(self.socket_path),
                limit=STREAM_LIMIT_BYTES,
            )
            # Enforce 0600 on socket file
            try:
                os.chmod(self.socket_path, 0o600)
            except OSError:
                pass
            self._running = True
            logger.info("PenguLink IPC server listening on %s", self.socket_path)
        except DaemonAlreadyRunningError:
            raise
        except Exception as exc:
            raise IPCError(f"Failed to bind IPC socket at {self.socket_path}: {exc}") from exc

    async def stop(self) -> None:
        """Gracefully terminate client sessions, close server, and clean up socket file."""
        if not self._running:
            return
        self._running = False

        # Allow brief tick for in-flight socket accept handlers to register
        try:
            await asyncio.sleep(0.05)
        except Exception:
            pass

        # 1. Stop accepting new connections
        if self._server:
            self._server.close()
            if hasattr(self._server, "close_clients"):
                try:
                    self._server.close_clients()
                except Exception:
                    pass

        # 2. Close all client writers
        writers = list(self._client_writers)
        for w in writers:
            try:
                w.close()
            except Exception:
                pass

        # 3. Cancel active client tasks, await gather, clear tasks and writers
        tasks = list(self._client_tasks)
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self._client_tasks.clear()
        self._client_writers.clear()

        # 4. Await self._server.wait_closed() with timeout
        if self._server:
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
            except (asyncio.TimeoutError, TimeoutError, Exception):
                pass
            # 5. Set self._server = None
            self._server = None

        # Clean up socket file from filesystem
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass

        logger.info("PenguLink IPC server stopped.")

    def _is_authorized_peer(self, sock: Optional[socket.socket]) -> bool:
        """Validate peer credentials via SO_PEERCRED on Unix domain sockets."""
        if sock and hasattr(socket, "SO_PEERCRED") and hasattr(os, "getuid"):
            try:
                cred_bytes = sock.getsockopt(
                    socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
                )
                _, peer_uid, _ = struct.unpack("3i", cred_bytes)
                if peer_uid != os.getuid():
                    logger.warning(
                        "SECURITY ALERT: Rejected IPC connection from unauthorized UID %d (expected UID %d)",
                        peer_uid,
                        os.getuid(),
                    )
                    return False
            except OSError as exc:
                logger.debug("SO_PEERCRED query failed or unsupported on socket: %s", exc)
        return True

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle individual client connection processing newline-delimited JSON-RPC requests."""
        task = asyncio.current_task()
        if task:
            self._client_tasks.add(task)
            task.add_done_callback(self._client_tasks.discard)
        self._client_writers.add(writer)

        # SO_PEERCRED UID validation on Unix domain sockets
        sock: Optional[socket.socket] = writer.get_extra_info("socket")
        if not self._is_authorized_peer(sock):
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return

        try:
            while self._running:
                try:
                    line = await reader.readline()
                except asyncio.LimitOverrunError:
                    logger.warning("Client exceeded 1 MB line guard limit on IPC stream; closing connection")
                    break

                if not line:
                    break

                if not self._is_authorized_peer(sock):
                    break

                response_bytes = await self._process_request_line(line)
                if response_bytes:
                    writer.write(response_bytes)
                    await writer.drain()
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as exc:
            logger.debug("IPC client session error: %s", exc)
        finally:
            self._client_writers.discard(writer)
            try:
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
            except (Exception, asyncio.CancelledError):
                pass

    async def _process_request_line(self, line: bytes) -> bytes:
        """Parse and execute a single JSON-RPC 2.0 request."""
        req_id: Optional[Union[str, int]] = None
        has_id: bool = False
        try:
            text = line.decode("utf-8").strip()
            if not text:
                return b""
            data = json.loads(text)
        except Exception as exc:
            return self._format_error(None, PARSE_ERROR, f"Parse error: {exc}")

        if not isinstance(data, dict) or data.get("jsonrpc") != JSONRPC_VERSION:
            return self._format_error(None, INVALID_REQUEST, "Invalid JSON-RPC 2.0 Request")

        has_id = "id" in data
        req_id = data.get("id")
        method = data.get("method")
        params = data.get("params", {})

        if not isinstance(method, str) or not method:
            return self._format_error(req_id, INVALID_REQUEST, "Method must be a non-empty string")

        if not isinstance(params, dict):
            return self._format_error(req_id, INVALID_PARAMS, "Params must be an object")

        if method in self._handlers:
            handler_fn = self._handlers[method]
            try:
                res = handler_fn(params)
                if inspect.isawaitable(res):
                    res = await res
                result = res
                if not has_id:
                    return b""
                response = {
                    "jsonrpc": JSONRPC_VERSION,
                    "result": result,
                    "id": req_id,
                }
                return (json.dumps(response) + "\n").encode("utf-8")
            except Exception as exc:
                return self._format_error(req_id if has_id else None, INTERNAL_ERROR, str(exc)) if has_id else b""

        if not self.handler:
            return self._format_error(req_id, INTERNAL_ERROR, "No request handler registered")

        try:
            result = await self.handler(method, params)
            if not has_id:
                # JSON-RPC Notification: no response sent
                return b""

            response = {
                "jsonrpc": JSONRPC_VERSION,
                "result": result,
                "id": req_id,
            }
            return (json.dumps(response) + "\n").encode("utf-8")

        except DeviceNotFoundError as exc:
            return self._format_error(req_id if has_id else None, DEVICE_NOT_FOUND, str(exc)) if has_id else b""
        except (PairingRejectedError, PairingTimestampExpiredError, PairingError) as exc:
            return self._format_error(req_id if has_id else None, PAIRING_ERROR, str(exc)) if has_id else b""
        except (ContactsError, ContactNotFoundError, SyncTimeoutError) as exc:
            return self._format_error(req_id if has_id else None, CONTACTS_ERROR, str(exc)) if has_id else b""
        except (MprisError, MprisPlayerNotFoundError, ActionUnsupportedError, MprisAdapterError) as exc:
            return self._format_error(req_id if has_id else None, MPRIS_ERROR, str(exc)) if has_id else b""
        except (RemoteCommandError, RemoteCommandNotFoundError) as exc:
            return self._format_error(req_id if has_id else None, REMOTE_COMMANDS_ERROR, str(exc)) if has_id else b""
        except (RunCommandError, CommandNotFoundError, CommandExecutionError) as exc:
            return self._format_error(req_id if has_id else None, RUN_COMMAND_ERROR, str(exc)) if has_id else b""
        except (StorageError, StorageCorruptedError, StorageWriteError) as exc:
            if method.startswith("runcommand_"):
                return self._format_error(req_id if has_id else None, RUN_COMMAND_ERROR, str(exc)) if has_id else b""
            return self._format_error(req_id if has_id else None, APPLICATION_ERROR, str(exc)) if has_id else b""
        except AudioError as exc:
            return self._format_error(req_id if has_id else None, AUDIO_SYSTEM_VOLUME_ERROR, str(exc)) if has_id else b""

        except InvalidStateTransitionError as exc:
            return self._format_error(req_id if has_id else None, INVALID_STATE_TRANSITION, str(exc)) if has_id else b""
        except (ConnectionTimeoutError, asyncio.TimeoutError, TimeoutError) as exc:
            return self._format_error(req_id if has_id else None, OPERATION_TIMEOUT, str(exc)) if has_id else b""
        except ConnectionClosedError as exc:
            return self._format_error(req_id if has_id else None, DEVICE_NOT_CONNECTED, str(exc)) if has_id else b""
        except DaemonAlreadyRunningError as exc:
            return self._format_error(req_id if has_id else None, DAEMON_ALREADY_RUNNING, str(exc)) if has_id else b""
        except KeyError as exc:
            msg = str(exc).lower()
            if "method" in msg or "unknown" in msg:
                return self._format_error(req_id if has_id else None, METHOD_NOT_FOUND, f"Method not found: {exc}") if has_id else b""
            if method.startswith("runcommand_"):
                return self._format_error(req_id if has_id else None, RUN_COMMAND_ERROR, str(exc)) if has_id else b""
            return self._format_error(req_id if has_id else None, INVALID_PARAMS, f"Invalid parameters: {exc}") if has_id else b""
        except (ValueError, TypeError) as exc:
            if method.startswith("runcommand_") and ("disabled" in str(exc).lower() or "not found" in str(exc).lower()):
                return self._format_error(req_id if has_id else None, RUN_COMMAND_ERROR, str(exc)) if has_id else b""
            return self._format_error(req_id if has_id else None, INVALID_PARAMS, f"Invalid parameters: {exc}") if has_id else b""
        except ClipboardUnavailableError as exc:
            return self._format_error(req_id if has_id else None, INTERNAL_ERROR, str(exc)) if has_id else b""
        except ClipboardError as exc:
            return self._format_error(req_id if has_id else None, CLIPBOARD_ERROR, str(exc)) if has_id else b""
        except (AlertError, AudioBackendError, AudioUnavailableError) as exc:
            if method.startswith("systemvolume_"):
                return self._format_error(req_id if has_id else None, AUDIO_SYSTEM_VOLUME_ERROR, str(exc)) if has_id else b""
            return self._format_error(req_id if has_id else None, ALERT_ERROR, str(exc)) if has_id else b""
        except (ShareError, PathTraversalError, TransferError, PayloadTransferError) as exc:
            return self._format_error(req_id if has_id else None, SHARE_ERROR, str(exc)) if has_id else b""
        except PenguLinkError as exc:
            msg = str(exc)
            if "not connected" in msg.lower():
                code = DEVICE_NOT_CONNECTED
            elif "not found" in msg.lower():
                if method.startswith("runcommand_"):
                    code = RUN_COMMAND_ERROR
                elif method.startswith("systemvolume_"):
                    code = AUDIO_SYSTEM_VOLUME_ERROR
                else:
                    code = DEVICE_NOT_FOUND
            elif "clipboard" in msg.lower() or "backend" in msg.lower():
                code = CLIPBOARD_ERROR
            elif method.startswith("systemvolume_"):
                code = AUDIO_SYSTEM_VOLUME_ERROR
            elif method.startswith("runcommand_"):
                code = RUN_COMMAND_ERROR
            else:
                code = APPLICATION_ERROR
            return self._format_error(req_id if has_id else None, code, msg) if has_id else b""
        except Exception as exc:
            if method.startswith("systemvolume_"):
                return self._format_error(req_id if has_id else None, AUDIO_SYSTEM_VOLUME_ERROR, str(exc)) if has_id else b""
            if method.startswith("runcommand_"):
                return self._format_error(req_id if has_id else None, RUN_COMMAND_ERROR, str(exc)) if has_id else b""
            if method == "set_clipboard":
                logger.warning("Error executing IPC method '%s': %s", method, type(exc).__name__)
            else:
                logger.warning("Error executing IPC method '%s': %s", method, exc)
            return self._format_error(req_id if has_id else None, APPLICATION_ERROR, str(exc)) if has_id else b""

    def _format_error(
        self,
        req_id: Optional[Union[str, int]],
        code: int,
        message: str,
        data: Optional[Any] = None,
    ) -> bytes:
        err: Dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        payload = {
            "jsonrpc": JSONRPC_VERSION,
            "error": err,
            "id": req_id,
        }
        return (json.dumps(payload) + "\n").encode("utf-8")


class PenguLinkIPCClient:
    """Unix Domain Socket client implementing JSON-RPC 2.0 communication."""

    def __init__(
        self,
        socket_path: Optional[Union[Path, str]] = None,
        default_timeout: float = 10.0,
    ) -> None:
        self.socket_path = resolve_socket_path(socket_path)
        self.default_timeout = default_timeout

    async def call(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Execute async JSON-RPC 2.0 call over Unix socket."""
        call_timeout = timeout if timeout is not None else self.default_timeout

        if not self.socket_path.exists():
            raise DaemonNotRunningError(
                f"PenguLink daemon is not running (socket not found at {self.socket_path}). "
                "Start with 'pengulink daemon run' or start the background service."
            )

        req_id = 1
        request = {
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
            "params": params if params is not None else {},
            "id": req_id,
        }
        request_bytes = (json.dumps(request) + "\n").encode("utf-8")

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(
                    path=str(self.socket_path),
                    limit=STREAM_LIMIT_BYTES,
                ),
                timeout=call_timeout,
            )
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            raise DaemonNotRunningError(
                f"PenguLink daemon is not running or not responding at {self.socket_path}. "
                "Start with 'pengulink daemon run' or start the background service."
            ) from exc
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise IPCError(f"Connection to IPC socket {self.socket_path} timed out") from exc

        try:
            writer.write(request_bytes)
            await asyncio.wait_for(writer.drain(), timeout=call_timeout)

            line = await asyncio.wait_for(reader.readline(), timeout=call_timeout)
            if not line:
                raise IPCError("Connection closed by daemon before response was received")

            response = json.loads(line.decode("utf-8"))
            if "error" in response and response["error"]:
                err = response["error"]
                err_code = err.get("code")
                err_msg = err.get("message", "Unknown error")
                exc = IPCError(f"IPC Error [{err_code}]: {err_msg}", context={"code": err_code})
                setattr(exc, "code", err_code)
                raise exc

            return response.get("result")
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise IPCError(f"IPC call '{method}' timed out after {call_timeout}s") from exc
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    def call_sync(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Synchronous wrapper for Click CLI commands."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self.call(method, params, timeout=timeout))
                return future.result()
        else:
            return asyncio.run(self.call(method, params, timeout=timeout))

    async def is_running(self) -> bool:
        """Check if daemon is currently responsive."""
        try:
            res = await self.call("status", timeout=1.0)
            return bool(isinstance(res, dict) and res.get("running"))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Typed Method Helpers
    # ------------------------------------------------------------------
    async def discover(self, timeout: float = 2.0, refresh: bool = True) -> List[Dict[str, Any]]:
        """Trigger UDP discovery sweep or return discovered devices."""
        res = await self.call("discover", {"timeout": timeout, "refresh": refresh}, timeout=timeout + 5.0)
        return list(res) if isinstance(res, list) else []

    async def get_devices(self, paired_only: bool = False, online_only: bool = False) -> List[Dict[str, Any]]:
        """Retrieve aggregated device list with connection state and battery."""
        res = await self.call("get_devices", {"paired_only": paired_only, "online_only": online_only})
        return list(res) if isinstance(res, list) else []

    async def pair_device(self, device_id: str, timeout: float = 30.0) -> Dict[str, Any]:
        """Initiate pairing and return visual verification code."""
        res = await self.call("pair_device", {"device_id": device_id, "timeout": timeout}, timeout=timeout + 5.0)
        return dict(res) if isinstance(res, dict) else {}

    async def pair_accept(self, device_id: str) -> Dict[str, Any]:
        """Accept an incoming pairing request from a remote device."""
        res = await self.call("pair_accept", {"device_id": device_id})
        return dict(res) if isinstance(res, dict) else {}

    async def pair_reject(self, device_id: str) -> Dict[str, Any]:
        """Reject an incoming pairing request from a remote device."""
        res = await self.call("pair_reject", {"device_id": device_id})
        return dict(res) if isinstance(res, dict) else {}

    async def get_pending_pairings(self) -> List[Dict[str, Any]]:
        """List active pending inbound pairing requests."""
        res = await self.call("get_pending_pairings")
        return list(res) if isinstance(res, list) else []

    async def unpair_device(self, device_id: str) -> Dict[str, Any]:
        """Revoke pairing and purge stored credentials."""
        res = await self.call("unpair_device", {"device_id": device_id})
        return dict(res) if isinstance(res, dict) else {}

    async def connect_device(
        self,
        device_id: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Manually trigger connection and mTLS upgrade."""
        params: Dict[str, Any] = {"device_id": device_id, "timeout": timeout}
        if host is not None:
            params["host"] = host
        if port is not None:
            params["port"] = port
        res = await self.call("connect_device", params, timeout=timeout + 5.0)
        return dict(res) if isinstance(res, dict) else {}

    async def send_ping(self, device_id: str, message: Optional[str] = None) -> Dict[str, Any]:
        """Send ping packet to connected peer."""
        params: Dict[str, Any] = {"device_id": device_id}
        if message is not None:
            params["message"] = message
        res = await self.call("send_ping", params)
        return dict(res) if isinstance(res, dict) else {}

    async def get_notifications(self, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch active mobile notifications with extracted 2FA/OTP codes."""
        params: Dict[str, Any] = {}
        if device_id is not None:
            params["device_id"] = device_id
        res = await self.call("get_notifications", params)
        return list(res) if isinstance(res, list) else []

    async def dismiss_notification(self, device_id: str, notification_id: str) -> Dict[str, Any]:
        """Send remote notification cancellation packet."""
        res = await self.call("dismiss_notification", {"device_id": device_id, "notification_id": notification_id})
        return dict(res) if isinstance(res, dict) else {}

    async def reply_notification(self, device_id: str, reply_id: str, message: str) -> Dict[str, Any]:
        """Send quick reply to a messaging notification."""
        res = await self.call(
            "reply_notification",
            {"device_id": device_id, "reply_id": reply_id, "message": message},
        )
        return dict(res) if isinstance(res, dict) else {}

    async def get_connectivity_report(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve latest cellular connectivity report for a device."""
        res = await self.call("get_connectivity_report", {"device_id": device_id})
        return dict(res) if isinstance(res, dict) else None

    async def request_connectivity_report(self, device_id: str) -> Dict[str, Any]:
        """Request remote device transmit fresh connectivity report."""
        res = await self.call("request_connectivity_report", {"device_id": device_id})
        return dict(res) if isinstance(res, dict) else {}

    async def get_clipboard(self) -> Dict[str, Any]:
        """Fetch current clipboard content and metadata from the daemon."""
        res = await self.call("get_clipboard")
        return dict(res) if isinstance(res, dict) else {}

    async def set_clipboard(self, content: str) -> Dict[str, Any]:
        """Update local clipboard and broadcast to connected peers."""
        res = await self.call("set_clipboard", {"content": content})
        return dict(res) if isinstance(res, dict) else {}

    async def get_logs(self, limit: int = 100, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch recent records from in-memory LogRingBuffer."""
        params: Dict[str, Any] = {"limit": limit}
        if level is not None:
            params["level"] = level
        res = await self.call("get_logs", params)
        return list(res) if isinstance(res, list) else []

    async def find_phone(self, device_id: str) -> Dict[str, Any]:
        """Trigger remote ring on a connected, paired device."""
        res = await self.call("find_phone", {"device_id": device_id})
        return dict(res) if isinstance(res, dict) else {}

    async def alert_test(self, duration: float = 30.0) -> Dict[str, Any]:
        """Trigger local desktop alert test."""
        res = await self.call("alert_test", {"duration": duration})
        return dict(res) if isinstance(res, dict) else {}

    async def alert_stop(self) -> Dict[str, Any]:
        """Stop any active local desktop alert."""
        res = await self.call("alert_stop")
        return dict(res) if isinstance(res, dict) else {}

    async def alert_status(self) -> Dict[str, Any]:
        """Query current status of the desktop alert service."""
        res = await self.call("alert_status")
        return dict(res) if isinstance(res, dict) else {}

    async def status(self) -> Dict[str, Any]:
        """Fetch daemon health, uptime, and peer counts."""
        res = await self.call("status")
        return dict(res) if isinstance(res, dict) else {}

    async def stop_daemon(self) -> Dict[str, Any]:
        """Request graceful shutdown of the background daemon."""
        res = await self.call("stop_daemon")
        return dict(res) if isinstance(res, dict) else {}

    async def share_text(self, device_id: str, text: str, timeout: float = 10.0) -> Dict[str, Any]:
        """Send text to paired device."""
        res = await self.call("share_text", {"device_id": device_id, "text": text}, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def share_url(self, device_id: str, url: str, timeout: float = 10.0) -> Dict[str, Any]:
        """Send URL to paired device."""
        res = await self.call("share_url", {"device_id": device_id, "url": url}, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def share_file(
        self,
        device_id: str,
        file_path: Union[str, Path, List[Union[str, Path]]],
        open_flag: bool = False,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Send file(s) to paired device."""
        payload: Dict[str, Any] = {"device_id": device_id, "open": open_flag}
        if isinstance(file_path, list):
            payload["file_paths"] = [str(p) for p in file_path]
        else:
            payload["file_path"] = str(file_path)
        res = await self.call("share_file", payload, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def contacts_sync(
        self,
        device_id: str,
        force: bool = False,
        sync_timeout: Optional[float] = None,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Trigger contact synchronization with connected peer device."""
        params: Dict[str, Any] = {"device_id": device_id, "force": force}
        if sync_timeout is not None:
            params["timeout"] = sync_timeout
        res = await self.call("contacts_sync", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def contacts_list(
        self,
        device_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Fetch list of cached contacts for a device."""
        params: Dict[str, Any] = {"device_id": device_id, "offset": offset}
        if limit is not None:
            params["limit"] = limit
        res = await self.call("contacts_list", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def contacts_search(
        self,
        device_id: str,
        query: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Search contacts by query string across name, phone, email, organization."""
        params: Dict[str, Any] = {"device_id": device_id, "query": query}
        res = await self.call("contacts_search", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def contacts_get(
        self,
        device_id: str,
        uid: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Fetch single contact by UID."""
        params: Dict[str, Any] = {"device_id": device_id, "uid": uid}
        res = await self.call("contacts_get", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def mpris_list_players(
        self,
        device_id: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Fetch media player list for a device."""
        params: Dict[str, Any] = {"device_id": device_id}
        res = await self.call("mpris_list_players", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def mpris_get_status(
        self,
        device_id: str,
        player: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Fetch media player status and metadata for a device."""
        params: Dict[str, Any] = {"device_id": device_id}
        if player is not None:
            params["player"] = player
        res = await self.call("mpris_get_status", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def mpris_send_action(
        self,
        device_id: str,
        action: str,
        player: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Send media action (Play, Pause, PlayPause, Stop, Next, Previous) to device."""
        params: Dict[str, Any] = {"device_id": device_id, "action": action}
        if player is not None:
            params["player"] = player
        res = await self.call("mpris_send_action", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def mpris_set_volume(
        self,
        device_id: str,
        volume: int,
        player: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Set volume percentage (0..100) on remote media player."""
        params: Dict[str, Any] = {"device_id": device_id, "volume": volume}
        if player is not None:
            params["player"] = player
        res = await self.call("mpris_set_volume", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def mpris_seek(
        self,
        device_id: str,
        offset_ms: int,
        player: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Seek relative position in milliseconds on remote media player."""
        params: Dict[str, Any] = {"device_id": device_id, "offset_ms": offset_ms}
        if player is not None:
            params["player"] = player
        res = await self.call("mpris_seek", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def mpris_set_position(
        self,
        device_id: str,
        position_ms: int,
        player: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Set absolute track position in milliseconds on remote media player."""
        params: Dict[str, Any] = {"device_id": device_id, "position_ms": position_ms}
        if player is not None:
            params["player"] = player
        res = await self.call("mpris_set_position", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def mpris_request_player_list(
        self,
        device_id: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Request updated player list from remote device."""
        params: Dict[str, Any] = {"device_id": device_id}
        res = await self.call("mpris_request_player_list", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    # =====================================================================
    # Remote Commands Methods (Milestone 15)
    # =====================================================================

    async def list_remote_commands(
        self,
        device_id: str,
        timeout: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """List configured remote commands for device."""
        params: Dict[str, Any] = {"device_id": device_id}
        res = await self.call("list_remote_commands", params, timeout=timeout)
        return list(res) if isinstance(res, list) else []

    async def request_remote_commands(
        self,
        device_id: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Request remote peer to re-advertise its available command list."""
        params: Dict[str, Any] = {"device_id": device_id}
        res = await self.call("request_remote_commands", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def execute_remote_command(
        self,
        device_id: str,
        key: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Transmit remote command execution request to device."""
        params: Dict[str, Any] = {"device_id": device_id, "key": key}
        res = await self.call("execute_remote_command", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    # Aliases
    remotecommands_list = list_remote_commands
    remotecommands_run = execute_remote_command

    # =====================================================================
    # SystemVolume Local and Remote Methods (Milestones 22, 24, 25)
    # =====================================================================

    async def systemvolume_get_sinks(
        self,
        timeout: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """List local audio output sinks."""
        res = await self.call("systemvolume_get_sinks", {}, timeout=timeout)
        return list(res) if isinstance(res, list) else []

    async def systemvolume_set_volume(
        self,
        name: str,
        volume: int,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Set volume on a local audio sink."""
        params: Dict[str, Any] = {"name": name, "volume": volume}
        res = await self.call("systemvolume_set_volume", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def systemvolume_set_muted(
        self,
        name: str,
        muted: bool,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Set mute state on a local audio sink."""
        params: Dict[str, Any] = {"name": name, "muted": muted}
        res = await self.call("systemvolume_set_muted", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def systemvolume_set_default(
        self,
        name: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Set default local audio sink."""
        params: Dict[str, Any] = {"name": name}
        res = await self.call("systemvolume_set_default", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def systemvolume_get_remote_sinks(
        self,
        device_id: str,
        timeout: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """List cached audio sinks for a remote device."""
        params: Dict[str, Any] = {"device_id": device_id}
        res = await self.call("systemvolume_get_remote_sinks", params, timeout=timeout)
        return list(res) if isinstance(res, list) else []

    async def systemvolume_set_remote_volume(
        self,
        device_id: str,
        name: str,
        volume: int,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Set volume on a remote device audio sink."""
        params: Dict[str, Any] = {"device_id": device_id, "name": name, "volume": volume}
        res = await self.call("systemvolume_set_remote_volume", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def systemvolume_set_remote_muted(
        self,
        device_id: str,
        name: str,
        muted: bool,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Set mute state on a remote device audio sink."""
        params: Dict[str, Any] = {"device_id": device_id, "name": name, "muted": muted}
        res = await self.call("systemvolume_set_remote_muted", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def systemvolume_set_remote_default(
        self,
        device_id: str,
        name: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Set default audio sink on a remote device."""
        params: Dict[str, Any] = {"device_id": device_id, "name": name}
        res = await self.call("systemvolume_set_remote_default", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def systemvolume_request_remote_sinks(
        self,
        device_id: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Request remote peer to re-advertise its complete audio sink list."""
        params: Dict[str, Any] = {"device_id": device_id}
        res = await self.call("systemvolume_request_remote_sinks", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    # SystemVolume Client Aliases
    list_remote_audio_sinks = systemvolume_get_remote_sinks
    set_remote_audio_volume = systemvolume_set_remote_volume
    set_remote_audio_mute = systemvolume_set_remote_muted
    request_remote_audio_sinks = systemvolume_request_remote_sinks

    # =====================================================================
    # RunCommand Local Methods (Milestones 23, 25)
    # =====================================================================

    async def runcommand_list(
        self,
        timeout: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """List all configured local commands."""
        res = await self.call("runcommand_list", {}, timeout=timeout)
        return list(res) if isinstance(res, list) else []

    async def runcommand_get(
        self,
        key: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Retrieve single configured local command by key."""
        params: Dict[str, Any] = {"key": key}
        res = await self.call("runcommand_get", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def runcommand_add(
        self,
        name: str,
        command: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Add a locally configured command entry."""
        params: Dict[str, Any] = {"name": name, "command": command}
        res = await self.call("runcommand_add", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def runcommand_remove(
        self,
        key: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Remove a locally configured command entry by key."""
        params: Dict[str, Any] = {"key": key}
        res = await self.call("runcommand_remove", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def runcommand_exec(
        self,
        key: str,
        timeout: float = 65.0,
    ) -> Dict[str, Any]:
        """Execute a locally configured command by key."""
        params: Dict[str, Any] = {"key": key}
        res = await self.call("runcommand_exec", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    async def runcommand_stop(
        self,
        id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Stop running local command process by process id or all processes."""
        params: Dict[str, Any] = {}
        if id is not None:
            params["id"] = id
        res = await self.call("runcommand_stop", params, timeout=timeout)
        return dict(res) if isinstance(res, dict) else {}

    # RunCommand Client Aliases
    list_local_commands = runcommand_list
    execute_local_command = runcommand_exec


# Convenience Alias
IPCSocketClient = PenguLinkIPCClient

__all__ = [
    "JSONRPC_VERSION",
    "STREAM_LIMIT_BYTES",
    "PARSE_ERROR",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "INVALID_PARAMS",
    "INTERNAL_ERROR",
    "APPLICATION_ERROR",
    "DEVICE_NOT_FOUND",
    "DEVICE_NOT_CONNECTED",
    "PAIRING_ERROR",
    "INVALID_STATE_TRANSITION",
    "NOTIFICATION_NOT_FOUND",
    "DAEMON_NOT_READY",
    "OPERATION_TIMEOUT",
    "DAEMON_ALREADY_RUNNING",
    "CLIPBOARD_ERROR",
    "ALERT_ERROR",
    "SHARE_ERROR",
    "CONTACTS_ERROR",
    "MPRIS_ERROR",
    "REMOTE_COMMANDS_ERROR",
    "AUDIO_SYSTEM_VOLUME_ERROR",
    "RUN_COMMAND_ERROR",
    "resolve_socket_path",
    "PenguLinkIPCServer",
    "PenguLinkIPCClient",
    "IPCSocketClient",
]

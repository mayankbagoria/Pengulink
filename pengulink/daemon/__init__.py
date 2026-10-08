"""pengulink.daemon
~~~~~~~~~~~~~~~~
Background daemon service, JSON-RPC Unix Domain Socket IPC, and lifecycle management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pengulink.daemon.ipc import PenguLinkIPCClient, PenguLinkIPCServer
    from pengulink.daemon.service import PenguLinkDaemon

__all__ = [
    "PenguLinkDaemon",
    "PenguLinkIPCServer",
    "PenguLinkIPCClient",
]


def __getattr__(name: str):
    if name in ("PenguLinkIPCClient", "PenguLinkIPCServer"):
        from pengulink.daemon.ipc import PenguLinkIPCClient, PenguLinkIPCServer

        return locals()[name]
    if name == "PenguLinkDaemon":
        from pengulink.daemon.service import PenguLinkDaemon

        return PenguLinkDaemon
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


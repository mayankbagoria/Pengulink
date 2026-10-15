"""pengulink.plugins.mpris
~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect Media Player Control and Remote Subsystems (MPRIS).

Provides:
- MprisControl: Server plugin exposing local Linux media players over MPRIS to peers.
- MprisRemote: Client plugin controlling and querying remote media players.
- MprisHostAdapter: Abstract interface for host player adapters.
- MockMprisAdapter: Headless in-memory adapter for deterministic testing.
- DBusMprisAdapter: Session D-Bus adapter with echo suppression.
- detect_mpris_adapter: Adapter factory function.
- PlayerState: Host media player state representation.
- RemotePlayerState: Remote device media player state representation.
"""

from pengulink.plugins.mpris.adapter import (
    DBusMprisAdapter,
    MockMprisAdapter,
    MprisHostAdapter,
    PlayerState,
    detect_mpris_adapter,
)
from pengulink.plugins.mpris.control import MprisControl
from pengulink.plugins.mpris.remote import RemotePlayerState, MprisRemote

__all__ = [
    "PlayerState",
    "RemotePlayerState",
    "MprisHostAdapter",
    "MockMprisAdapter",
    "DBusMprisAdapter",
    "detect_mpris_adapter",
    "MprisControl",
    "MprisRemote",
]

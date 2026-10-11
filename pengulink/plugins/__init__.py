"""pengulink.plugins
~~~~~~~~~~~~~~~~~
KDE Connect core plugin framework and plugin implementations.
"""

from __future__ import annotations

from pengulink.plugins.base import Plugin, PluginRegistry
from pengulink.plugins.battery import BatteryPlugin
from pengulink.plugins.connectivity_report import (
    ConnectivityReport,
    ConnectivityReportPlugin,
    SignalState,
    create_connectivity_request_packet,
)
from pengulink.plugins.findmyphone import (
    FindMyPhonePlugin,
    create_find_my_phone_request_packet,
)
from pengulink.plugins.notification import NotificationPlugin
from pengulink.plugins.ping import PingPlugin
from pengulink.plugins.share import SharePlugin

__all__ = [
    "Plugin",
    "PluginRegistry",
    "PingPlugin",
    "BatteryPlugin",
    "NotificationPlugin",
    "ConnectivityReportPlugin",
    "SignalState",
    "ConnectivityReport",
    "create_connectivity_request_packet",
    "FindMyPhonePlugin",
    "create_find_my_phone_request_packet",
    "SharePlugin",
]


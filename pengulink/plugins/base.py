"""pengulink.plugins.base
~~~~~~~~~~~~~~~~~~~~~~
Abstract Plugin base class and dynamic PluginRegistry.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Union

from pengulink.core.events import EventBus
from pengulink.network.packet import Packet

logger = logging.getLogger("pengulink.plugins.base")

SendPacketCallable = Callable[[Packet], Union[Coroutine[Any, Any, None], None]]


async def safe_send_packet(send_packet: SendPacketCallable, packet: Packet) -> None:
    """Helper to dispatch packet via send_packet callback whether sync or async."""
    res = send_packet(packet)
    if inspect.isawaitable(res):
        await res


class Plugin(ABC):
    """Abstract base class for all KDE Connect plugins."""

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.event_bus = event_bus

    @property
    @abstractmethod
    def plugin_id(self) -> str:
        """Unique plugin identifier (e.g. 'ping', 'battery', 'notification')."""
        ...

    @property
    @abstractmethod
    def incoming_capabilities(self) -> List[str]:
        """Packet types this plugin can receive."""
        ...

    @property
    @abstractmethod
    def outgoing_capabilities(self) -> List[str]:
        """Packet types this plugin can transmit."""
        ...

    @abstractmethod
    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Process an incoming packet routed to this plugin."""
        ...

    async def start(self) -> None:
        """Lifecycle hook: initialize resources when daemon starts."""
        pass

    async def stop(self) -> None:
        """Lifecycle hook: clean up resources when daemon stops."""
        pass


class PluginRegistry:
    """Registry coordinating dynamic plugin loading and packet routing."""

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.event_bus = event_bus
        self._plugins: Dict[str, Plugin] = {}
        self._packet_routing_table: Dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        """Register a plugin instance and map its incoming packet types."""
        self._plugins[plugin.plugin_id] = plugin
        for cap in plugin.incoming_capabilities:
            self._packet_routing_table[cap] = plugin
        logger.debug(
            "Registered plugin '%s' with capabilities %s",
            plugin.plugin_id,
            plugin.incoming_capabilities,
        )

    def unregister(self, plugin_id: str) -> Optional[Plugin]:
        """Unregister a plugin."""
        plugin = self._plugins.pop(plugin_id, None)
        if plugin:
            for cap in plugin.incoming_capabilities:
                if self._packet_routing_table.get(cap) is plugin:
                    del self._packet_routing_table[cap]
        return plugin

    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get plugin by ID."""
        return self._plugins.get(plugin_id)

    def get_plugin_for_packet_type(self, packet_type: str) -> Optional[Plugin]:
        """Find the plugin registered to handle the given packet type."""
        return self._packet_routing_table.get(packet_type)

    def get_all_incoming_capabilities(self) -> List[str]:
        """List all unique incoming capabilities across registered plugins."""
        caps: Set[str] = set()
        for p in self._plugins.values():
            caps.update(p.incoming_capabilities)
        return sorted(list(caps))

    def get_all_outgoing_capabilities(self) -> List[str]:
        """List all unique outgoing capabilities across registered plugins."""
        caps: Set[str] = set()
        for p in self._plugins.values():
            caps.update(p.outgoing_capabilities)
        return sorted(list(caps))

    async def dispatch_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> bool:
        """Route packet to the appropriate plugin.

        Returns True if a plugin handled it; False if packet type is unknown.
        """
        plugin = self.get_plugin_for_packet_type(packet.type)
        if not plugin:
            logger.debug(
                "No registered plugin handles packet type '%s' from device %s",
                packet.type,
                device_id,
            )
            return False

        try:
            await plugin.handle_packet(packet, device_id, send_packet)
            return True
        except Exception as exc:
            logger.error(
                "Error executing plugin '%s' for packet type '%s' from device %s: %s",
                plugin.plugin_id,
                packet.type,
                device_id,
                exc,
                exc_info=True,
            )
            return False

    async def start_all(self) -> None:
        """Start all registered plugins."""
        for plugin in self._plugins.values():
            try:
                await plugin.start()
            except Exception as exc:
                logger.error("Failed to start plugin '%s': %s", plugin.plugin_id, exc)

    async def stop_all(self) -> None:
        """Stop all registered plugins."""
        for plugin in self._plugins.values():
            try:
                await plugin.stop()
            except Exception as exc:
                logger.error("Failed to stop plugin '%s': %s", plugin.plugin_id, exc)

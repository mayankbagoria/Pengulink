"""tests.unit.test_plugin_telephony
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit test suite for KDE Connect Telephony plugin and active call state management:
- Plugin metadata, capabilities advertisement, and lifecycle subscriptions
- Inbound packet parsing for all event types (ringing, talking, missedCall, idle, callEnded)
- isCancel boolean, string, and integer parsing
- Domain event emission on EventBus (CallEvent, CallStateChangedEvent)
- Outbound telephony actions (mute_ringer, reject_call)
- Malformed payloads and defensive resilience (non-dict body, missing fields, extreme string lengths)
- Device disconnect and unpair lifecycle cleanup
- Multi-device call state isolation
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from pengulink.core.events import (
    CallEvent,
    CallStateChangedEvent,
    DeviceDisconnected,
    DeviceUnpaired,
    DeviceUnpairedEvent,
    Event,
    EventBus,
)
from pengulink.core.exceptions import (
    CallNotFoundError,
    ConnectionClosedError,
    InvalidCallStateError,
    PluginError,
    TelephonyError,
)
from pengulink.network.packet import Packet
from pengulink.plugins.telephony import (
    KNOWN_CALL_EVENTS,
    TELEPHONY_PACKET_TYPE,
    TELEPHONY_REQUEST_MUTE_TYPE,
    TELEPHONY_REQUEST_TYPE,
    TelephonyPlugin,
    coerce_boolean,
    create_telephony_mute_packet,
    create_telephony_reject_packet,
)
from pengulink.services.telephony import (
    ActiveCall,
    CallState,
    TelephonyService,
    coerce_is_cancel,
)


class MockPacketSender:
    """Mock sender callback capturing dispatched wire packets."""

    def __init__(self, is_async: bool = True) -> None:
        self.sent_packets: list[Packet] = []
        self.is_async = is_async

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        packet = None
        for arg in args:
            if isinstance(arg, Packet):
                packet = arg
                break
        if packet is None:
            packet = kwargs.get("packet")
        if packet is not None:
            self.sent_packets.append(packet)

        if self.is_async:
            async def _coro() -> None:
                pass
            return _coro()
        return None


# =====================================================================
# Test Class 1: Events, Exceptions and Data Models
# =====================================================================


class TestEventsAndExceptions:
    def test_call_event_subclass_and_defaults(self) -> None:
        assert issubclass(CallEvent, Event)
        ev = CallEvent()
        assert ev.device_id == ""
        assert ev.event_type == ""
        assert ev.event == ""  # alias property
        assert ev.phone_number == ""
        assert ev.contact_name == ""
        assert ev.is_cancel is False
        assert ev.phone_thumbnail is None
        assert isinstance(ev.timestamp, float)

    def test_call_state_changed_event_subclass_and_defaults(self) -> None:
        assert issubclass(CallStateChangedEvent, Event)
        ev = CallStateChangedEvent()
        assert ev.device_id == ""
        assert ev.old_state == "IDLE"
        assert ev.new_state == "IDLE"
        assert ev.phone_number == ""
        assert ev.contact_name == ""
        assert isinstance(ev.timestamp, float)

    def test_telephony_exceptions_hierarchy(self) -> None:
        assert issubclass(TelephonyError, PluginError)
        assert issubclass(CallNotFoundError, TelephonyError)
        assert issubclass(CallNotFoundError, KeyError)
        assert issubclass(InvalidCallStateError, TelephonyError)

        err = CallNotFoundError("call 1 not found")
        assert isinstance(err, KeyError)
        assert isinstance(err, TelephonyError)

    def test_call_state_enum_values(self) -> None:
        assert CallState.IDLE == "IDLE"
        assert CallState.RINGING == "RINGING"
        assert CallState.TALKING == "TALKING"
        assert CallState.MISSED == "MISSED"
        assert isinstance(CallState.RINGING, str)
        assert "RINGING" in KNOWN_CALL_EVENTS or "ringing" in KNOWN_CALL_EVENTS

    def test_active_call_dataclass_methods_and_properties(self) -> None:
        call = ActiveCall(
            device_id="dev_1",
            phone_number="+15551234567",
            contact_name="Alice",
            state=CallState.RINGING,
        )
        assert call.is_active is True
        assert call.duration >= 0.0

        d = call.to_dict()
        assert d["device_id"] == "dev_1"
        assert d["phone_number"] == "+15551234567"
        assert d["contact_name"] == "Alice"
        assert d["state"] == "RINGING"
        assert d["is_active"] is True
        assert "duration" in d

        call.state = CallState.TALKING
        assert call.is_active is True
        assert call.to_dict()["state"] == "TALKING"

        call.state = CallState.IDLE
        assert call.is_active is False

        call.state = CallState.MISSED
        assert call.is_active is False


# =====================================================================
# Test Class 2: Plugin Metadata and Capabilities
# =====================================================================


class TestTelephonyPluginMetadataAndCapabilities:
    async def test_plugin_id(self) -> None:
        plugin = TelephonyPlugin()
        assert plugin.plugin_id == "telephony"

    async def test_incoming_capabilities(self) -> None:
        plugin = TelephonyPlugin()
        assert plugin.incoming_capabilities == [TELEPHONY_PACKET_TYPE]

    async def test_outgoing_capabilities(self) -> None:
        plugin = TelephonyPlugin()
        assert plugin.outgoing_capabilities == [
            TELEPHONY_REQUEST_MUTE_TYPE,
            TELEPHONY_REQUEST_TYPE,
        ]

    async def test_default_telephony_service_instantiation(self) -> None:
        event_bus = EventBus()
        plugin = TelephonyPlugin(event_bus=event_bus)
        assert isinstance(plugin.telephony_service, TelephonyService)
        assert plugin.telephony_service.event_bus is event_bus

    async def test_lifecycle_subscriptions_registered_on_start_and_cleared_on_stop(self) -> None:
        event_bus = EventBus()
        plugin = TelephonyPlugin(event_bus=event_bus)
        assert len(plugin._unsub_callbacks) == 0

        await plugin.start()
        assert len(plugin._unsub_callbacks) >= 2

        plugin._connected_peers["dev1"] = MockPacketSender()
        assert len(plugin._connected_peers) == 1

        await plugin.stop()
        assert len(plugin._unsub_callbacks) == 0
        assert len(plugin._connected_peers) == 0


# =====================================================================
# Test Class 3: Packet Parsing and State Transitions
# =====================================================================


class TestTelephonyPacketParsingAndTransitions:
    @pytest.fixture
    def plugin(self) -> TelephonyPlugin:
        event_bus = EventBus()
        return TelephonyPlugin(event_bus=event_bus)

    async def test_incoming_call_ringing_transition(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        packet = Packet(
            type="kdeconnect.telephony",
            body={
                "event": "ringing",
                "phoneNumber": "+15551234567",
                "contactName": "Alice Smith",
            },
        )
        await plugin.handle_packet(packet, "phone_alpha", sender)

        call = await plugin.get_call("phone_alpha")
        assert call is not None
        assert call["state"] == "RINGING"
        assert call["phone_number"] == "+15551234567"
        assert call["contact_name"] == "Alice Smith"
        assert call["is_active"] is True

    async def test_call_answered_talking_transition(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        # 1. Ringing
        await plugin.handle_packet(
            Packet(
                type="kdeconnect.telephony",
                body={"event": "ringing", "phoneNumber": "+15551234567", "contactName": "Alice"},
            ),
            "phone_alpha",
            sender,
        )
        # 2. Answered / talking
        await plugin.handle_packet(
            Packet(
                type="kdeconnect.telephony",
                body={"event": "talking", "phoneNumber": "+15551234567", "contactName": "Alice"},
            ),
            "phone_alpha",
            sender,
        )

        call = await plugin.get_call("phone_alpha")
        assert call is not None
        assert call["state"] == "TALKING"
        assert call["is_active"] is True

    async def test_call_answered_directly_without_prior_ringing(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        await plugin.handle_packet(
            Packet(
                type="kdeconnect.telephony",
                body={"event": "talking", "phoneNumber": "+15559876543", "contactName": "Bob"},
            ),
            "phone_beta",
            sender,
        )

        call = await plugin.get_call("phone_beta")
        assert call is not None
        assert call["state"] == "TALKING"
        assert call["contact_name"] == "Bob"

    async def test_missed_call_transition_from_ringing(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        # 1. Ringing
        await plugin.handle_packet(
            Packet(
                type="kdeconnect.telephony",
                body={"event": "ringing", "phoneNumber": "+15551234567", "contactName": "Alice"},
            ),
            "phone_alpha",
            sender,
        )
        assert await plugin.get_call("phone_alpha") is not None

        # 2. Missed call
        await plugin.handle_packet(
            Packet(
                type="kdeconnect.telephony",
                body={"event": "missedCall", "phoneNumber": "+15551234567", "contactName": "Alice"},
            ),
            "phone_alpha",
            sender,
        )
        # Missed call evicts from active calls
        assert await plugin.get_call("phone_alpha") is None

    async def test_direct_missed_call_on_idle_device(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        await plugin.handle_packet(
            Packet(
                type="kdeconnect.telephony",
                body={"event": "missedCall", "phoneNumber": "+15559876543"},
            ),
            "phone_alpha",
            sender,
        )
        assert await plugin.get_call("phone_alpha") is None

    async def test_call_ended_via_idle_event(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        await plugin.handle_packet(
            Packet(
                type="kdeconnect.telephony",
                body={"event": "talking", "phoneNumber": "+15551234567"},
            ),
            "phone_alpha",
            sender,
        )
        assert await plugin.get_call("phone_alpha") is not None

        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "idle"}),
            "phone_alpha",
            sender,
        )
        assert await plugin.get_call("phone_alpha") is None

    async def test_call_ended_via_callEnded_event(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        await plugin.handle_packet(
            Packet(
                type="kdeconnect.telephony",
                body={"event": "ringing", "phoneNumber": "+15551234567"},
            ),
            "phone_alpha",
            sender,
        )
        assert await plugin.get_call("phone_alpha") is not None

        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "callEnded"}),
            "phone_alpha",
            sender,
        )
        assert await plugin.get_call("phone_alpha") is None

    async def test_legacy_sms_event_ignored(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        await plugin.handle_packet(
            Packet(
                type="kdeconnect.telephony",
                body={"event": "sms", "messageBody": "Hello world"},
            ),
            "phone_alpha",
            sender,
        )
        assert await plugin.get_call("phone_alpha") is None

    async def test_unknown_event_type_handled_gracefully(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        await plugin.handle_packet(
            Packet(
                type="kdeconnect.telephony",
                body={"event": "unrecognized_custom_event"},
            ),
            "phone_alpha",
            sender,
        )
        assert await plugin.get_call("phone_alpha") is None


# =====================================================================
# Test Class 4: isCancel Parsing Variations
# =====================================================================


class TestTelephonyIsCancelParsing:
    @pytest.fixture
    def plugin(self) -> TelephonyPlugin:
        return TelephonyPlugin(event_bus=EventBus())

    async def test_is_cancel_boolean_true(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123"}),
            "p1",
            sender,
        )
        assert await plugin.get_call("p1") is not None

        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123", "isCancel": True}),
            "p1",
            sender,
        )
        assert await plugin.get_call("p1") is None

    async def test_is_cancel_boolean_false(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123", "isCancel": False}),
            "p1",
            sender,
        )
        assert await plugin.get_call("p1") is not None

    async def test_is_cancel_string_lowercase_true(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "talking", "phoneNumber": "123"}),
            "p1",
            sender,
        )
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "talking", "phoneNumber": "123", "isCancel": "true"}),
            "p1",
            sender,
        )
        assert await plugin.get_call("p1") is None

    async def test_is_cancel_string_titlecase_true(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123"}),
            "p1",
            sender,
        )
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123", "isCancel": "True"}),
            "p1",
            sender,
        )
        assert await plugin.get_call("p1") is None

    async def test_is_cancel_string_one(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123"}),
            "p1",
            sender,
        )
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123", "isCancel": "1"}),
            "p1",
            sender,
        )
        assert await plugin.get_call("p1") is None

    async def test_is_cancel_integer_one(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123"}),
            "p1",
            sender,
        )
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123", "isCancel": 1}),
            "p1",
            sender,
        )
        assert await plugin.get_call("p1") is None

    async def test_is_cancel_falsy_values(self, plugin: TelephonyPlugin) -> None:
        for val in ("false", "0", 0, ""):
            sender = MockPacketSender()
            await plugin.handle_packet(
                Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123", "isCancel": val}),
                "p1",
                sender,
            )
            assert await plugin.get_call("p1") is not None
            await plugin.clear_device("p1")

    def test_coerce_helper_functions(self) -> None:
        assert coerce_boolean(True) is True
        assert coerce_boolean(False) is False
        assert coerce_boolean("true") is True
        assert coerce_boolean("True") is True
        assert coerce_boolean("1") is True
        assert coerce_boolean(1) is True
        assert coerce_boolean("false") is False
        assert coerce_boolean(None) is False
        assert coerce_is_cancel(True) is True
        assert coerce_is_cancel("yes") is True
        assert coerce_is_cancel("no") is False


# =====================================================================
# Test Class 5: Domain Events Emission
# =====================================================================


class TestTelephonyDomainEvents:
    @pytest.fixture
    def setup_bus(self) -> tuple[EventBus, TelephonyPlugin]:
        bus = EventBus()
        plugin = TelephonyPlugin(event_bus=bus)
        return bus, plugin

    async def test_call_event_emitted_for_inbound_packets(
        self, setup_bus: tuple[EventBus, TelephonyPlugin]
    ) -> None:
        bus, plugin = setup_bus
        events_received: list[CallEvent] = []
        bus.subscribe(CallEvent, lambda ev: events_received.append(ev))

        packet = Packet(
            type="kdeconnect.telephony",
            body={
                "event": "ringing",
                "phoneNumber": "+15551234567",
                "contactName": "Alice",
                "isCancel": False,
                "phoneThumbnail": "base64thumb",
            },
        )
        await plugin.handle_packet(packet, "phone_alpha", MockPacketSender())

        assert len(events_received) == 1
        ev = events_received[0]
        assert ev.device_id == "phone_alpha"
        assert ev.event_type == "ringing"
        assert ev.event == "ringing"
        assert ev.phone_number == "+15551234567"
        assert ev.contact_name == "Alice"
        assert ev.is_cancel is False
        assert ev.phone_thumbnail == "base64thumb"

    async def test_call_state_changed_event_on_new_ringing_call(
        self, setup_bus: tuple[EventBus, TelephonyPlugin]
    ) -> None:
        bus, plugin = setup_bus
        state_events: list[CallStateChangedEvent] = []
        bus.subscribe(CallStateChangedEvent, lambda ev: state_events.append(ev))

        packet = Packet(
            type="kdeconnect.telephony",
            body={"event": "ringing", "phoneNumber": "+15551234567", "contactName": "Alice"},
        )
        await plugin.handle_packet(packet, "phone_alpha", MockPacketSender())

        assert len(state_events) == 1
        ev = state_events[0]
        assert ev.device_id == "phone_alpha"
        assert ev.old_state == "IDLE"
        assert ev.new_state == "RINGING"
        assert ev.phone_number == "+15551234567"
        assert ev.contact_name == "Alice"

    async def test_call_state_changed_event_on_talking_transition(
        self, setup_bus: tuple[EventBus, TelephonyPlugin]
    ) -> None:
        bus, plugin = setup_bus
        state_events: list[CallStateChangedEvent] = []
        bus.subscribe(CallStateChangedEvent, lambda ev: state_events.append(ev))

        # Ringing
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123", "contactName": "Alice"}),
            "dev1",
            MockPacketSender(),
        )
        # Talking
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "talking", "phoneNumber": "123", "contactName": "Alice"}),
            "dev1",
            MockPacketSender(),
        )

        assert len(state_events) == 2
        assert state_events[0].old_state == "IDLE"
        assert state_events[0].new_state == "RINGING"
        assert state_events[1].old_state == "RINGING"
        assert state_events[1].new_state == "TALKING"

    async def test_call_state_changed_event_on_call_termination(
        self, setup_bus: tuple[EventBus, TelephonyPlugin]
    ) -> None:
        bus, plugin = setup_bus
        state_events: list[CallStateChangedEvent] = []
        bus.subscribe(CallStateChangedEvent, lambda ev: state_events.append(ev))

        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "talking", "phoneNumber": "123", "contactName": "Alice"}),
            "dev1",
            MockPacketSender(),
        )
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "idle"}),
            "dev1",
            MockPacketSender(),
        )

        assert len(state_events) == 2
        assert state_events[1].old_state == "TALKING"
        assert state_events[1].new_state == "IDLE"

    async def test_duplicate_state_does_not_reemit_state_changed(
        self, setup_bus: tuple[EventBus, TelephonyPlugin]
    ) -> None:
        bus, plugin = setup_bus
        call_events: list[CallEvent] = []
        state_events: list[CallStateChangedEvent] = []
        bus.subscribe(CallEvent, lambda ev: call_events.append(ev))
        bus.subscribe(CallStateChangedEvent, lambda ev: state_events.append(ev))

        pkt = Packet(
            type="kdeconnect.telephony",
            body={"event": "ringing", "phoneNumber": "+1234", "contactName": "Bob"},
        )
        # Send two successive ringing packets for the same call
        await plugin.handle_packet(pkt, "dev1", MockPacketSender())
        await plugin.handle_packet(pkt, "dev1", MockPacketSender())

        # CallEvent emitted twice
        assert len(call_events) == 2
        # CallStateChangedEvent emitted only once!
        assert len(state_events) == 1
        assert state_events[0].new_state == "RINGING"

    async def test_eventbus_subscriber_exception_isolation(
        self, setup_bus: tuple[EventBus, TelephonyPlugin]
    ) -> None:
        bus, plugin = setup_bus

        def bad_handler(ev: Any) -> None:
            raise RuntimeError("Handler crash simulated")

        bus.subscribe(CallEvent, bad_handler)
        bus.subscribe(CallStateChangedEvent, bad_handler)

        pkt = Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123"})
        # Should not raise exception
        await plugin.handle_packet(pkt, "dev1", MockPacketSender())
        call = await plugin.get_call("dev1")
        assert call is not None
        assert call["state"] == "RINGING"


# =====================================================================
# Test Class 6: Outbound Telephony Actions
# =====================================================================


class TestTelephonyOutboundActions:
    @pytest.fixture
    def plugin(self) -> TelephonyPlugin:
        return TelephonyPlugin()

    async def test_mute_ringer_packet_dispatch(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        result = await plugin.mute_ringer("dev1", send_packet=sender)

        assert result is True
        assert len(sender.sent_packets) == 1
        pkt = sender.sent_packets[0]
        assert pkt.type == TELEPHONY_REQUEST_MUTE_TYPE
        assert pkt.body == {}

    async def test_reject_call_packet_dispatch(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        result = await plugin.reject_call("dev1", send_packet=sender)

        assert result is True
        assert len(sender.sent_packets) == 1
        pkt = sender.sent_packets[0]
        assert pkt.type == TELEPHONY_REQUEST_TYPE
        assert pkt.body == {"action": "rejectCall"}

    async def test_reject_call_with_phone_number(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        result = await plugin.reject_call(
            "dev1", send_packet=sender, phone_number="+15551234567"
        )

        assert result is True
        assert len(sender.sent_packets) == 1
        pkt = sender.sent_packets[0]
        assert pkt.type == TELEPHONY_REQUEST_TYPE
        assert pkt.body == {"action": "rejectCall", "phoneNumber": "+15551234567"}

    async def test_outbound_actions_support_sync_and_async_sender(self, plugin: TelephonyPlugin) -> None:
        sync_sender = MockPacketSender(is_async=False)
        async_sender = MockPacketSender(is_async=True)

        res_sync = await plugin.mute_ringer("dev1", send_packet=sync_sender)
        assert res_sync is True
        assert len(sync_sender.sent_packets) == 1

        res_async = await plugin.reject_call("dev1", send_packet=async_sender)
        assert res_async is True
        assert len(async_sender.sent_packets) == 1

    async def test_outbound_action_missing_sender_raises_error(self, plugin: TelephonyPlugin) -> None:
        with pytest.raises(ConnectionClosedError):
            await plugin.mute_ringer("dev_unknown")

        with pytest.raises(ConnectionClosedError):
            await plugin.reject_call("dev_unknown")

    async def test_outbound_actions_parameter_reversal_guard(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        # Test passing (send_packet, device_id)
        res = await plugin.mute_ringer(sender, "dev_inverted")  # type: ignore
        assert res is True
        assert len(sender.sent_packets) == 1

        res2 = await plugin.reject_call(sender, "dev_inverted")  # type: ignore
        assert res2 is True
        assert len(sender.sent_packets) == 2

    def test_packet_creation_helpers(self) -> None:
        mute_pkt = create_telephony_mute_packet()
        assert mute_pkt.type == TELEPHONY_REQUEST_MUTE_TYPE
        assert mute_pkt.body == {}

        reject_pkt = create_telephony_reject_packet("+1234")
        assert reject_pkt.type == TELEPHONY_REQUEST_TYPE
        assert reject_pkt.body["action"] == "rejectCall"
        assert reject_pkt.body["phoneNumber"] == "+1234"


# =====================================================================
# Test Class 7: Defensive Resilience
# =====================================================================


class TestTelephonyDefensiveResilience:
    @pytest.fixture
    def plugin(self) -> TelephonyPlugin:
        return TelephonyPlugin()

    async def test_non_dict_body_dropped(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        for invalid_body in ("not a dict", None, [1, 2, 3], 12345):
            bad_pkt = MagicMock()
            bad_pkt.type = "kdeconnect.telephony"
            bad_pkt.body = invalid_body
            await plugin.handle_packet(bad_pkt, "dev1", sender)
            assert await plugin.get_call("dev1") is None

    async def test_missing_event_dropped_cleanly(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        pkt = Packet(type="kdeconnect.telephony", body={"phoneNumber": "123"})
        await plugin.handle_packet(pkt, "dev1", sender)
        assert await plugin.get_call("dev1") is None

    async def test_missing_event_with_is_cancel_treated_as_idle(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        # Set up a ringing call
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123"}),
            "dev1",
            sender,
        )
        assert await plugin.get_call("dev1") is not None

        # Packet with missing event but isCancel=True
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"isCancel": True}),
            "dev1",
            sender,
        )
        assert await plugin.get_call("dev1") is None

    async def test_phone_number_and_contact_name_fallbacks(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()

        # Case 1: Phone number given, contact name omitted
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "+1555"}),
            "dev1",
            sender,
        )
        call1 = await plugin.get_call("dev1")
        assert call1 is not None
        assert call1["phone_number"] == "+1555"
        assert call1["contact_name"] == "+1555"

        # Case 2: Neither given
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing"}),
            "dev2",
            sender,
        )
        call2 = await plugin.get_call("dev2")
        assert call2 is not None
        assert call2["phone_number"] == "unknown number"
        assert call2["contact_name"] == "unknown number"

    async def test_extreme_string_lengths_clamped(self, plugin: TelephonyPlugin) -> None:
        sender = MockPacketSender()
        long_str = "A" * 2000
        await plugin.handle_packet(
            Packet(
                type="kdeconnect.telephony",
                body={"event": "ringing", "phoneNumber": long_str, "contactName": long_str},
            ),
            "dev1",
            sender,
        )
        call = await plugin.get_call("dev1")
        assert call is not None
        assert len(call["phone_number"]) <= 128
        assert len(call["contact_name"]) <= 1024

    async def test_unauthenticated_device_packet_dropped(self) -> None:
        def auth_check(dev_id: str) -> bool:
            return dev_id == "trusted_dev"

        plugin = TelephonyPlugin(is_paired_checker=auth_check)
        sender = MockPacketSender()

        # Untrusted device packet dropped
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123"}),
            "untrusted_dev",
            sender,
        )
        assert await plugin.get_call("untrusted_dev") is None

        # Trusted device packet accepted
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123"}),
            "trusted_dev",
            sender,
        )
        assert await plugin.get_call("trusted_dev") is not None

    async def test_empty_device_id_raises_invalid_call_state(self) -> None:
        service = TelephonyService()
        with pytest.raises(InvalidCallStateError):
            await service.process_call_event(device_id="", event="ringing")
        with pytest.raises(InvalidCallStateError):
            await service.process_call_event(device_id="   ", event="ringing")


# =====================================================================
# Test Class 8: Lifecycle and Multi-Device Isolation
# =====================================================================


class TestTelephonyLifecycleAndMultiDeviceIsolation:
    @pytest.fixture
    def setup_system(self) -> tuple[EventBus, TelephonyPlugin]:
        bus = EventBus()
        plugin = TelephonyPlugin(event_bus=bus)
        return bus, plugin

    async def test_device_disconnected_clears_active_call_and_emits_idle(
        self, setup_system: tuple[EventBus, TelephonyPlugin]
    ) -> None:
        bus, plugin = setup_system
        await plugin.start()

        state_events: list[CallStateChangedEvent] = []
        bus.subscribe(CallStateChangedEvent, lambda ev: state_events.append(ev))

        # Setup active ringing call
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123"}),
            "dev_disc",
            MockPacketSender(),
        )
        assert await plugin.get_call("dev_disc") is not None
        assert len(state_events) == 1

        # Publish DeviceDisconnected
        await bus.publish(DeviceDisconnected(device_id="dev_disc"))

        assert await plugin.get_call("dev_disc") is None
        assert len(state_events) == 2
        assert state_events[1].device_id == "dev_disc"
        assert state_events[1].old_state == "RINGING"
        assert state_events[1].new_state == "IDLE"

        await plugin.stop()

    async def test_device_unpaired_clears_active_call_and_emits_idle(
        self, setup_system: tuple[EventBus, TelephonyPlugin]
    ) -> None:
        bus, plugin = setup_system
        await plugin.start()

        state_events: list[CallStateChangedEvent] = []
        bus.subscribe(CallStateChangedEvent, lambda ev: state_events.append(ev))

        # Setup active talking call
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "talking", "phoneNumber": "123"}),
            "dev_unp",
            MockPacketSender(),
        )
        assert await plugin.get_call("dev_unp") is not None

        # Publish DeviceUnpaired
        await bus.publish(DeviceUnpaired(device_id="dev_unp"))

        assert await plugin.get_call("dev_unp") is None
        assert len(state_events) == 2
        assert state_events[1].device_id == "dev_unp"
        assert state_events[1].old_state == "TALKING"
        assert state_events[1].new_state == "IDLE"

        await plugin.stop()

    async def test_device_unpaired_alias_event(
        self, setup_system: tuple[EventBus, TelephonyPlugin]
    ) -> None:
        bus, plugin = setup_system
        await plugin.start()

        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "123"}),
            "dev_alias",
            MockPacketSender(),
        )
        assert await plugin.get_call("dev_alias") is not None

        await bus.publish(DeviceUnpairedEvent(device_id="dev_alias"))
        assert await plugin.get_call("dev_alias") is None

        await plugin.stop()

    async def test_disconnect_of_idle_device_is_noop(
        self, setup_system: tuple[EventBus, TelephonyPlugin]
    ) -> None:
        bus, plugin = setup_system
        await plugin.start()

        state_events: list[CallStateChangedEvent] = []
        bus.subscribe(CallStateChangedEvent, lambda ev: state_events.append(ev))

        # Device has no active call; publishing disconnect should be safe no-op
        await bus.publish(DeviceDisconnected(device_id="idle_dev"))

        assert await plugin.get_call("idle_dev") is None
        assert len(state_events) == 0

        await plugin.stop()

    async def test_multi_device_independent_calls(self, setup_system: tuple[EventBus, TelephonyPlugin]) -> None:
        _, plugin = setup_system

        # phone_a receives call from Alice
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "111", "contactName": "Alice"}),
            "phone_a",
            MockPacketSender(),
        )
        # phone_b receives call from Bob
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "222", "contactName": "Bob"}),
            "phone_b",
            MockPacketSender(),
        )

        call_a = await plugin.get_call("phone_a")
        call_b = await plugin.get_call("phone_b")
        assert call_a is not None and call_a["contact_name"] == "Alice"
        assert call_b is not None and call_b["contact_name"] == "Bob"

        # phone_a transitions to talking
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "talking", "phoneNumber": "111"}),
            "phone_a",
            MockPacketSender(),
        )
        # phone_b remains ringing
        call_a = await plugin.get_call("phone_a")
        call_b = await plugin.get_call("phone_b")
        assert call_a["state"] == "TALKING"
        assert call_b["state"] == "RINGING"

        # phone_b ends call
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "idle"}),
            "phone_b",
            MockPacketSender(),
        )
        assert await plugin.get_call("phone_b") is None
        # phone_a is still talking!
        call_a = await plugin.get_call("phone_a")
        assert call_a is not None and call_a["state"] == "TALKING"

    async def test_multi_device_isolated_disconnect(self, setup_system: tuple[EventBus, TelephonyPlugin]) -> None:
        bus, plugin = setup_system
        await plugin.start()

        # Both devices have active calls
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "talking", "phoneNumber": "111"}),
            "phone_a",
            MockPacketSender(),
        )
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "222"}),
            "phone_b",
            MockPacketSender(),
        )

        # Disconnect phone_a
        await bus.publish(DeviceDisconnected(device_id="phone_a"))

        # phone_a is evicted
        assert await plugin.get_call("phone_a") is None
        # phone_b is still active!
        call_b = await plugin.get_call("phone_b")
        assert call_b is not None and call_b["state"] == "RINGING"

        await plugin.stop()

    async def test_get_active_calls_and_clear_all(self, setup_system: tuple[EventBus, TelephonyPlugin]) -> None:
        _, plugin = setup_system

        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "ringing", "phoneNumber": "111"}),
            "phone_a",
            MockPacketSender(),
        )
        await plugin.handle_packet(
            Packet(type="kdeconnect.telephony", body={"event": "talking", "phoneNumber": "222"}),
            "phone_b",
            MockPacketSender(),
        )

        # All calls query
        all_calls = await plugin.get_active_calls()
        assert len(all_calls) == 2

        # Filtered query
        filtered = await plugin.get_active_calls("phone_a")
        assert len(filtered) == 1
        assert filtered[0]["device_id"] == "phone_a"

        # Clear device
        cleared = await plugin.clear_device("phone_a")
        assert cleared is True
        assert len(await plugin.get_active_calls()) == 1

        # Clear all
        await plugin.telephony_service.clear_all()
        assert len(await plugin.get_active_calls()) == 0

    async def test_telephony_service_get_call_object_and_sync(self) -> None:
        service = TelephonyService()
        await service.process_call_event("d1", "ringing", "+123", "Alice")

        obj = await service.get_call_object("d1")
        assert isinstance(obj, ActiveCall)
        assert obj.state == CallState.RINGING

        sync_call = service.get_call_sync("d1")
        assert sync_call is not None
        assert sync_call["state"] == "RINGING"

        sync_calls = service.get_active_calls_sync()
        assert len(sync_calls) == 1

        with pytest.raises(CallNotFoundError):
            service.get_call_sync("nonexistent", raise_if_not_found=True)

        with pytest.raises(CallNotFoundError):
            await service.get_call_object("nonexistent", raise_if_not_found=True)


# =====================================================================
# Test Class 9: EventBus Subscriber Re-entrancy
# =====================================================================


class TestTelephonyEventBusReentrancy:
    """Regression tests for EventBus subscriber re-entrancy into TelephonyService.

    Verifies that EventBus subscribers querying TelephonyService methods
    (e.g. get_active_calls, get_call) during event dispatch never deadlock
    on asyncio.Lock and complete well under the 0.5s timeout threshold.
    """

    async def test_event_bus_subscriber_query_during_call_state_changed(self) -> None:
        """An async subscriber subscribes to CallStateChangedEvent on EventBus.

        When fired, it awaits telephony_service.get_active_calls() and
        telephony_service.get_call("dev1"). Asserts that the call resolves
        in < 0.5s without timing out, and observes the updated call state.
        """
        bus = EventBus()
        telephony_service = TelephonyService(event_bus=bus)

        observed_calls: list[dict[str, Any]] = []
        observed_call: dict[str, Any] | None = None
        subscriber_invoked = asyncio.Event()
        query_duration: float = 0.0

        async def subscriber(ev: CallStateChangedEvent) -> None:
            nonlocal query_duration, observed_call
            t0 = time.monotonic()
            calls = await telephony_service.get_active_calls()
            call = await telephony_service.get_call("dev1")
            query_duration = time.monotonic() - t0
            observed_calls.extend(calls)
            observed_call = call
            subscriber_invoked.set()

        bus.subscribe(CallStateChangedEvent, subscriber)

        start_time = time.monotonic()
        await asyncio.wait_for(
            telephony_service.process_call_event(
                device_id="dev1",
                event="ringing",
                phone_number="+15551234567",
                contact_name="Alice",
            ),
            timeout=0.5,
        )
        await asyncio.wait_for(subscriber_invoked.wait(), timeout=0.5)
        total_duration = time.monotonic() - start_time

        assert total_duration < 0.5
        assert query_duration < 0.5
        assert subscriber_invoked.is_set()

        # Observes updated call state
        assert len(observed_calls) == 1
        assert observed_calls[0]["device_id"] == "dev1"
        assert observed_calls[0]["state"] == CallState.RINGING.value
        assert observed_calls[0]["phone_number"] == "+15551234567"
        assert observed_calls[0]["contact_name"] == "Alice"

        assert observed_call is not None
        assert observed_call["device_id"] == "dev1"
        assert observed_call["state"] == CallState.RINGING.value
        assert observed_call["phone_number"] == "+15551234567"
        assert observed_call["contact_name"] == "Alice"

        telephony_service.close()

    async def test_event_bus_subscriber_query_during_disconnect_cleanup(self) -> None:
        """An async subscriber subscribes to CallStateChangedEvent on EventBus.

        When DeviceDisconnected or clear_device fires, the subscriber awaits
        telephony_service.get_active_calls() and confirms empty list without deadlock.
        """
        bus = EventBus()
        telephony_service = TelephonyService(event_bus=bus)

        # 1. Establish an active call first
        await telephony_service.process_call_event(
            device_id="dev1",
            event="ringing",
            phone_number="+15551234567",
            contact_name="Alice",
        )
        active_before = await telephony_service.get_active_calls()
        assert len(active_before) == 1

        observed_calls_disconnect: list[dict[str, Any]] = []
        observed_call_dev1: dict[str, Any] | None = {"marker": "exists"}
        disconnect_invoked = asyncio.Event()
        disconnect_query_duration: float = 0.0

        async def subscriber_on_disconnect(ev: CallStateChangedEvent) -> None:
            nonlocal disconnect_query_duration, observed_call_dev1
            if ev.new_state == CallState.IDLE.value:
                t0 = time.monotonic()
                calls = await telephony_service.get_active_calls()
                call = await telephony_service.get_call("dev1")
                disconnect_query_duration = time.monotonic() - t0
                observed_calls_disconnect.extend(calls)
                observed_call_dev1 = call
                disconnect_invoked.set()

        bus.subscribe(CallStateChangedEvent, subscriber_on_disconnect)

        # 2. Trigger DeviceDisconnected event
        start_time = time.monotonic()
        await asyncio.wait_for(
            bus.publish(DeviceDisconnected(device_id="dev1")),
            timeout=0.5,
        )
        await asyncio.wait_for(disconnect_invoked.wait(), timeout=0.5)
        total_duration = time.monotonic() - start_time

        assert total_duration < 0.5
        assert disconnect_query_duration < 0.5
        assert disconnect_invoked.is_set()

        # Confirms empty list and None for dev1 without deadlock
        assert len(observed_calls_disconnect) == 0
        assert observed_call_dev1 is None
        assert len(await telephony_service.get_active_calls()) == 0

        # 3. Also verify clear_device directly on another device
        await telephony_service.process_call_event(
            device_id="dev2",
            event="talking",
            phone_number="+19876543210",
            contact_name="Bob",
        )
        assert len(await telephony_service.get_active_calls()) == 1

        clear_invoked = asyncio.Event()
        observed_calls_clear: list[dict[str, Any]] = []

        async def subscriber_on_clear(ev: CallStateChangedEvent) -> None:
            if ev.device_id == "dev2" and ev.new_state == CallState.IDLE.value:
                calls = await telephony_service.get_active_calls()
                observed_calls_clear.extend(calls)
                clear_invoked.set()

        bus.subscribe(CallStateChangedEvent, subscriber_on_clear)

        start_clear = time.monotonic()
        await asyncio.wait_for(
            telephony_service.clear_device("dev2"),
            timeout=0.5,
        )
        await asyncio.wait_for(clear_invoked.wait(), timeout=0.5)
        assert (time.monotonic() - start_clear) < 0.5
        assert len(observed_calls_clear) == 0
        assert await telephony_service.get_call("dev2") is None

        telephony_service.close()

    async def test_event_bus_call_event_subscriber_query_observes_updated_state(
        self,
    ) -> None:
        """CallEvent subscriber query re-entrancy and state consistency.

        Verifies that subscribers to CallEvent can safely query TelephonyService
        without deadlock and observe the freshly added call in _active_calls.
        """
        bus = EventBus()
        telephony_service = TelephonyService(event_bus=bus)

        observed_calls: list[dict[str, Any]] = []
        subscriber_invoked = asyncio.Event()

        async def call_event_subscriber(ev: CallEvent) -> None:
            calls = await telephony_service.get_active_calls()
            observed_calls.extend(calls)
            subscriber_invoked.set()

        bus.subscribe(CallEvent, call_event_subscriber)

        await asyncio.wait_for(
            telephony_service.process_call_event(
                device_id="dev_call_ev",
                event="ringing",
                phone_number="+1112223333",
                contact_name="Charlie",
            ),
            timeout=0.5,
        )
        await asyncio.wait_for(subscriber_invoked.wait(), timeout=0.5)

        assert subscriber_invoked.is_set()
        assert len(observed_calls) == 1
        assert observed_calls[0]["device_id"] == "dev_call_ev"
        assert observed_calls[0]["state"] == CallState.RINGING.value

        telephony_service.close()

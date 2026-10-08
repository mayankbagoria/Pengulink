"""pengulink.plugins.notification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect mobile application notification synchronization plugin.

Handles incoming notifications, cancellation, 2FA/OTP code extraction,
remote dismissal, quick reply, and custom notification action execution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pengulink.core.events import EventBus, NotificationCleared, NotificationReceived
from pengulink.network.packet import Packet
from pengulink.plugins.base import Plugin, SendPacketCallable, safe_send_packet

logger = logging.getLogger("pengulink.plugins.notification")

NOTIFICATION_PACKET_TYPE = "kdeconnect.notification"
NOTIFICATION_REQUEST_TYPE = "kdeconnect.notification.request"
NOTIFICATION_ACTION_TYPE = "kdeconnect.notification.action"
NOTIFICATION_REPLY_TYPE = "kdeconnect.notification.reply"

# Regex to strip invisible formatting and soft hyphens
INVISIBLE_CHARS_REGEX = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
# Regex to match 4 to 8 digit verification / OTP codes
OTP_CODE_REGEX = re.compile(r"\b(\d{4,8})\b")
# Android KDE Connect ID prefix
ANDROID_ID_PREFIX = "org.kde.kdeconnect_tp::"


@dataclass
class StoredNotification:
    """In-memory representation of an active mobile notification."""

    notification_id: str
    device_id: str
    app_name: str
    title: str
    text: str
    ticker: str
    is_clearable: bool = True
    silent: bool = False
    time_str: str = ""
    group_name: Optional[str] = None
    actions: List[str] = field(default_factory=list)
    request_reply_id: Optional[str] = None
    auth_code: Optional[str] = None


def sanitize_notification_id(raw_id: str) -> str:
    """Strip Android package prefix from notification identifier."""
    if not raw_id:
        return ""
    cleaned = str(raw_id)
    if cleaned.startswith(ANDROID_ID_PREFIX):
        cleaned = cleaned[len(ANDROID_ID_PREFIX):]
    return cleaned


def extract_otp_code(text: str, ticker: str = "") -> Optional[str]:
    """Extract SMS/2FA verification code from notification text."""
    combined = f"{text} {ticker}"
    sanitized = INVISIBLE_CHARS_REGEX.sub("", combined)

    # Keywords commonly surrounding OTPs
    keywords = ["code", "otp", "pin", "verify", "verification", "password", "login", "auth"]
    has_keyword = any(kw in sanitized.lower() for kw in keywords)

    matches = OTP_CODE_REGEX.findall(sanitized)
    if matches and has_keyword:
        return matches[0]
    return None


class NotificationPlugin(Plugin):
    """Processes notifications from mobile devices and transmits user responses."""

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        super().__init__(event_bus)
        # Mapping: notification_id -> StoredNotification
        self._active_notifications: Dict[str, StoredNotification] = {}

    @property
    def plugin_id(self) -> str:
        return "notification"

    @property
    def incoming_capabilities(self) -> List[str]:
        return [NOTIFICATION_PACKET_TYPE]

    @property
    def outgoing_capabilities(self) -> List[str]:
        return [
            NOTIFICATION_REQUEST_TYPE,
            NOTIFICATION_ACTION_TYPE,
            NOTIFICATION_REPLY_TYPE,
        ]

    def get_notification(self, notification_id: str) -> Optional[StoredNotification]:
        return self._active_notifications.get(notification_id)

    def list_notifications(self) -> List[StoredNotification]:
        return list(self._active_notifications.values())

    async def handle_packet(
        self,
        packet: Packet,
        device_id: str,
        send_packet: SendPacketCallable,
    ) -> None:
        """Process incoming kdeconnect.notification packet."""
        if packet.type != NOTIFICATION_PACKET_TYPE:
            return

        body = packet.body
        raw_id = body.get("id")
        if not raw_id:
            logger.debug("Received notification without ID from %s; ignoring", device_id)
            return

        notification_id = sanitize_notification_id(str(raw_id))

        # Check for remote cancellation
        if body.get("isCancel") is True:
            logger.debug("Remote notification %s cleared on device %s", notification_id, device_id)
            self._active_notifications.pop(notification_id, None)
            if self.event_bus:
                await self.event_bus.publish(
                    NotificationCleared(
                        device_id=device_id,
                        notification_id=notification_id,
                    )
                )
            return

        # New or updated notification
        app_name = str(body.get("appName", "Unknown App"))
        title = str(body.get("title", ""))
        text = str(body.get("text", ""))
        ticker = str(body.get("ticker", ""))
        is_clearable = bool(body.get("isClearable", True))
        silent = bool(body.get("silent", False))
        time_str = str(body.get("time", ""))
        group_name = body.get("groupName")
        raw_actions = body.get("actions", [])
        actions = list(raw_actions) if isinstance(raw_actions, (list, tuple)) else []
        request_reply_id = body.get("requestReplyId")

        auth_code = extract_otp_code(text, ticker)

        stored = StoredNotification(
            notification_id=notification_id,
            device_id=device_id,
            app_name=app_name,
            title=title,
            text=text,
            ticker=ticker,
            is_clearable=is_clearable,
            silent=silent,
            time_str=time_str,
            group_name=str(group_name) if group_name else None,
            actions=[str(a) for a in actions],
            request_reply_id=str(request_reply_id) if request_reply_id else None,
            auth_code=auth_code,
        )
        self._active_notifications[notification_id] = stored

        logger.info(
            "Notification from %s [%s]: '%s' - '%s'%s",
            device_id,
            app_name,
            title,
            text[:64],
            f" (2FA OTP: {auth_code})" if auth_code else "",
        )

        if self.event_bus:
            await self.event_bus.publish(
                NotificationReceived(
                    device_id=device_id,
                    notification_id=notification_id,
                    app_name=app_name,
                    title=title,
                    text=text,
                    ticker=ticker,
                    is_clearable=is_clearable,
                    silent=silent,
                    time_str=time_str,
                    group_name=stored.group_name,
                    actions=stored.actions,
                    request_reply_id=stored.request_reply_id,
                    auth_code=stored.auth_code,
                )
            )

    async def dismiss_notification(
        self,
        send_packet: SendPacketCallable,
        notification_id: str,
    ) -> Packet:
        """Send a dismissal request to the remote device and clear from local cache."""
        clean_id = sanitize_notification_id(notification_id)
        packet = Packet(
            type=NOTIFICATION_REQUEST_TYPE,
            body={"cancel": clean_id},
        )
        await safe_send_packet(send_packet, packet)

        # Optimistically clear from local cache
        stored = self._active_notifications.pop(clean_id, None)
        if stored and self.event_bus:
            await self.event_bus.publish(
                NotificationCleared(
                    device_id=stored.device_id,
                    notification_id=clean_id,
                )
            )
        return packet

    async def reply_notification(
        self,
        send_packet: SendPacketCallable,
        request_reply_id: str,
        message: str,
    ) -> Packet:
        """Send a quick reply to a messaging notification."""
        packet = Packet(
            type=NOTIFICATION_REPLY_TYPE,
            body={
                "requestReplyId": str(request_reply_id),
                "message": str(message),
            },
        )
        await safe_send_packet(send_packet, packet)
        return packet

    async def trigger_action(
        self,
        send_packet: SendPacketCallable,
        notification_id: str,
        action: str,
    ) -> Packet:
        """Trigger an action button on a remote notification."""
        clean_id = sanitize_notification_id(notification_id)
        packet = Packet(
            type=NOTIFICATION_ACTION_TYPE,
            body={
                "key": clean_id,
                "action": str(action),
            },
        )
        await safe_send_packet(send_packet, packet)
        return packet

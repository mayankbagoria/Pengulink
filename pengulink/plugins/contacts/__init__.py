"""pengulink.plugins.contacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect Contacts synchronization plugin ecosystem for PenguLink.
"""

from pengulink.plugins.contacts.plugin import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SYNC_TIMEOUT,
    LEGACY_CONTACTS,
    MAX_BATCH_SIZE,
    REQUEST_ALL_UIDS_TIMESTAMPS,
    REQUEST_VCARDS_BY_UID,
    RESPONSE_UIDS_TIMESTAMPS,
    RESPONSE_VCARDS,
    ContactsPlugin,
    ContactsSyncResult,
)
from pengulink.plugins.contacts.store import (
    ContactStore,
    SyncDiff,
    chunk_list,
)
from pengulink.plugins.contacts.vcard import (
    Contact,
    EmailAddress,
    PhoneNumber,
    normalize_phone_number,
    parse_vcard,
    parse_vcard_line,
    resolve_contact_name,
    unfold_vcard_lines,
)

__all__ = [
    "ContactsPlugin",
    "ContactsSyncResult",
    "ContactStore",
    "SyncDiff",
    "chunk_list",
    "Contact",
    "PhoneNumber",
    "EmailAddress",
    "parse_vcard",
    "normalize_phone_number",
    "unfold_vcard_lines",
    "parse_vcard_line",
    "resolve_contact_name",
    "REQUEST_ALL_UIDS_TIMESTAMPS",
    "RESPONSE_UIDS_TIMESTAMPS",
    "REQUEST_VCARDS_BY_UID",
    "RESPONSE_VCARDS",
    "LEGACY_CONTACTS",
    "DEFAULT_BATCH_SIZE",
    "MAX_BATCH_SIZE",
    "DEFAULT_SYNC_TIMEOUT",
]

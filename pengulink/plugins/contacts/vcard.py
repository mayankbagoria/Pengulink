"""pengulink.plugins.contacts.vcard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RFC 2426 / 2.1 vCard parser and contact models using standard library.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import quopri

logger = logging.getLogger("pengulink.plugins.contacts.vcard")


@dataclass(frozen=True)
class PhoneNumber:
    """Represents an individual contact phone number with normalized digits."""

    number: str
    normalized: str
    type: str = "CELL"  # CELL, WORK, HOME, OTHER, VOICE, FAX, PAGER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "normalized": self.normalized,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PhoneNumber:
        return cls(
            number=str(data.get("number", "")),
            normalized=str(data.get("normalized", "")),
            type=str(data.get("type", "CELL")),
        )


@dataclass(frozen=True)
class EmailAddress:
    """Represents an individual contact email address."""

    email: str
    type: str = "HOME"  # HOME, WORK, OTHER, INTERNET

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email": self.email,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmailAddress:
        return cls(
            email=str(data.get("email", "")),
            type=str(data.get("type", "HOME")),
        )


@dataclass
class Contact:
    """Normalized, strongly-typed contact record parsed from a vCard."""

    uid: str
    device_id: str = ""
    name: str = ""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phones: List[PhoneNumber] = field(default_factory=list)
    emails: List[EmailAddress] = field(default_factory=list)
    organization: Optional[str] = None
    title: Optional[str] = None
    note: Optional[str] = None
    timestamp: int = 0  # Remote epoch ms
    raw_vcard: str = ""
    cached_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "device_id": self.device_id,
            "name": self.name,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "phones": [p.to_dict() for p in self.phones],
            "emails": [e.to_dict() for e in self.emails],
            "organization": self.organization,
            "title": self.title,
            "note": self.note,
            "timestamp": self.timestamp,
            "raw_vcard": self.raw_vcard,
            "cached_at": self.cached_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Contact:
        return cls(
            uid=str(data.get("uid", "")),
            device_id=str(data.get("device_id", "")),
            name=str(data.get("name", "Unknown Contact")),
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            phones=[
                PhoneNumber.from_dict(p)
                for p in data.get("phones", [])
                if isinstance(p, dict)
            ],
            emails=[
                EmailAddress.from_dict(e)
                for e in data.get("emails", [])
                if isinstance(e, dict)
            ],
            organization=data.get("organization"),
            title=data.get("title"),
            note=data.get("note"),
            timestamp=int(data.get("timestamp", 0)),
            raw_vcard=str(data.get("raw_vcard", "")),
            cached_at=float(data.get("cached_at", time.time())),
        )


def normalize_phone_number(raw_number: str) -> str:
    """Normalize phone number to international or standard digit form.
    
    Preserves leading '+' if present and strips all whitespace and punctuation.
    If no digits are present (e.g. carrier shortcodes or alphanumeric codes),
    returns the trimmed original string.
    """
    cleaned = raw_number.strip()
    if not cleaned:
        return ""
    if re.search(r"[*#]", cleaned):
        return cleaned
    has_plus = cleaned.startswith("+")
    digits = re.sub(r"[^\d]", "", cleaned)
    if digits:
        return f"+{digits}" if has_plus else digits
    return cleaned


def unfold_vcard_lines(vcard_text: str) -> List[str]:
    """Unfold RFC 2426 / 2.1 continuation lines and split into distinct logical lines.
    
    1. Unfolds quoted-printable soft breaks ('=\\r?\\n').
    2. Unfolds RFC 2425/2426 space or tab continuation ('\\r?\\n[ \\t]').
    3. Normalizes CRLF to LF and splits into lines.
    """
    if not vcard_text:
        return []

    # Replace Quoted-Printable soft line breaks
    text = re.sub(r"=\r?\n", "", vcard_text)

    # Replace RFC 2425/2426 line folding (newline followed by space or tab)
    text = re.sub(r"\r?\n[ \t]", "", text)

    # Normalize CRLF and CR to LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return lines


def _unescape_vcard_value(val: str) -> str:
    """Unescape vCard 3.0 escape sequences: \\,, \\;, \\:, \\\\, \\n, \\N."""
    val = val.replace(r"\n", "\n").replace(r"\N", "\n")
    val = val.replace(r"\,", ",").replace(r"\;", ";").replace(r"\:", ":")
    val = val.replace(r"\\", "\\")
    return val


def parse_vcard_line(line: str, unescape: bool = True) -> Tuple[str, Dict[str, str], str]:
    """Parse a single logical vCard line into (property_name, parameters_dict, value).
    
    Handles:
    - Group prefixes (e.g. 'item1.TEL' -> 'TEL').
    - Parameters (e.g. 'TEL;TYPE=CELL,VOICE:...', 'TEL;WORK:...', 'ENCODING=QUOTED-PRINTABLE;CHARSET=UTF-8:...').
    - Quoted-Printable decoding.
    - Value unescaping (delayed for structured fields like N and ORG to prevent splitting corruption).
    """
    if ":" not in line:
        return "", {}, ""

    head, _, value = line.partition(":")
    parts = head.split(";")
    prop_raw = parts[0].strip()

    # Strip group prefix like item1.TEL
    if "." in prop_raw:
        prop_name = prop_raw.split(".", 1)[1].upper()
    else:
        prop_name = prop_raw.upper()

    params: Dict[str, str] = {}
    known_types = {"CELL", "WORK", "HOME", "VOICE", "INTERNET", "PREF", "FAX", "PAGER", "MAIN", "OTHER"}

    for p in parts[1:]:
        p = p.strip()
        if not p:
            continue
        if "=" in p:
            k, v = p.split("=", 1)
            params[k.strip().upper()] = v.strip()
        else:
            # vCard 2.1 bare parameters like 'CELL', 'WORK', 'QUOTED-PRINTABLE'
            p_upper = p.upper()
            if p_upper == "QUOTED-PRINTABLE":
                params["ENCODING"] = "QUOTED-PRINTABLE"
            elif p_upper in ("UTF-8", "ASCII", "ISO-8859-1"):
                params["CHARSET"] = p_upper
            elif p_upper in known_types:
                params["TYPE"] = p_upper
            else:
                params[p_upper] = p

    # Check for Quoted-Printable encoding
    encoding = params.get("ENCODING", "").upper()
    charset = params.get("CHARSET", "UTF-8")

    if encoding in ("QUOTED-PRINTABLE", "QP"):
        try:
            raw_bytes = quopri.decodestring(value.encode("latin-1"))
            value = raw_bytes.decode(charset, errors="replace")
        except Exception as exc:
            logger.debug("Failed Quoted-Printable decoding for property %s: %s", prop_name, exc)

    # Delay unescaping for structured fields to prevent premature delimiter splitting
    if unescape and prop_name not in ("N", "ORG", "ADR"):
        value = _unescape_vcard_value(value.strip())
    else:
        value = value.strip()
    return prop_name, params, value


def resolve_contact_name(
    fn: str,
    n_components: List[str],
    organization: Optional[str],
    phones: List[PhoneNumber],
) -> Tuple[str, Optional[str], Optional[str]]:
    """Determine best contact name, first name, and last name using precedence rules:
    
    1. FN property if present and non-empty.
    2. Assembled from N (Given + Family).
    3. ORG organization name.
    4. First phone number.
    5. Fallback 'Unknown Contact'.
    """
    first_name: Optional[str] = None
    last_name: Optional[str] = None

    if n_components:
        # N: Family;Given;Additional;Prefix;Suffix
        fam = n_components[0].strip() if len(n_components) > 0 else ""
        giv = n_components[1].strip() if len(n_components) > 1 else ""
        first_name = giv if giv else None
        last_name = fam if fam else None

    # Priority 1: FN
    fn_clean = fn.strip()
    if fn_clean:
        return fn_clean, first_name, last_name

    # Priority 2: Structured N
    if first_name and last_name:
        return f"{first_name} {last_name}", first_name, last_name
    if first_name:
        return first_name, first_name, last_name
    if last_name:
        return last_name, first_name, last_name

    # Priority 3: ORG
    if organization and organization.strip():
        return organization.strip(), first_name, last_name

    # Priority 4: First phone number
    if phones:
        return phones[0].number, first_name, last_name

    # Priority 5: Fallback
    return "Unknown Contact", first_name, last_name


def parse_vcard(
    uid_or_raw: str,
    raw_vcard: Optional[str] = None,
    device_id: str = "",
    timestamp: int = 0,
) -> Union[Contact, List[Contact]]:
    """Parse raw RFC 2426 / 2.1 vCard text into a Contact dataclass or list of Contacts."""
    is_single_arg = (raw_vcard is None)
    if is_single_arg:
        raw_vcard_str = str(uid_or_raw or "")
        uid = f"contact_{int(time.time()*1000)}"
    else:
        uid = str(uid_or_raw).strip()
        raw_vcard_str = str(raw_vcard or "")
    lines = unfold_vcard_lines(raw_vcard_str)

    fn = ""
    n_components: List[str] = []
    phones: List[PhoneNumber] = []
    emails: List[EmailAddress] = []
    organization: Optional[str] = None
    title: Optional[str] = None
    note: Optional[str] = None

    for line in lines:
        if ":" not in line:
            if note is not None:
                note += "\n" + line
            continue

        try:
            prop_name, params, val = parse_vcard_line(line)
        except Exception as exc:
            logger.debug("Skipping invalid vCard line '%s': %s", line, exc)
            continue

        if not prop_name or not val:
            continue

        if prop_name == "FN":
            if not fn:
                fn = val

        elif prop_name == "N":
            if not n_components:
                # Family;Given;Additional;Prefix;Suffix
                # Split on unescaped semicolons first, then unescape individual components
                raw_parts = re.split(r"(?<!\\);", val)
                n_components = [_unescape_vcard_value(p) for p in raw_parts]

        elif prop_name == "TEL":
            ptype_raw = params.get("TYPE", "CELL").upper()
            # If multiple types like CELL,VOICE, pick primary
            types = [t.strip() for t in ptype_raw.split(",") if t.strip()]
            ptype = "CELL"
            for t in types:
                if t in ("CELL", "MOBILE"):
                    ptype = "CELL"
                    break
                elif t in ("WORK", "HOME", "VOICE", "FAX", "PAGER", "MAIN", "OTHER"):
                    ptype = t
            normalized = normalize_phone_number(val)
            phones.append(PhoneNumber(number=val, normalized=normalized, type=ptype))

        elif prop_name == "EMAIL":
            etype_raw = params.get("TYPE", "HOME").upper()
            etypes = [t.strip() for t in etype_raw.split(",") if t.strip()]
            etype = "HOME"
            for t in etypes:
                if t in ("HOME", "WORK", "INTERNET", "OTHER"):
                    etype = t
                    break
            emails.append(EmailAddress(email=val, type=etype))

        elif prop_name == "ORG":
            if not organization:
                # If structured like Org;Department, split on unescaped semicolons first
                raw_parts = re.split(r"(?<!\\);", val)
                org_parts = [_unescape_vcard_value(p.strip()) for p in raw_parts if p.strip()]
                organization = "; ".join(org_parts) if org_parts else _unescape_vcard_value(val)

        elif prop_name == "TITLE":
            if not title:
                title = val

        elif prop_name == "NOTE":
            if note:
                note += "\n" + val
            else:
                note = val

    name, first_name, last_name = resolve_contact_name(
        fn=fn,
        n_components=n_components,
        organization=organization,
        phones=phones,
    )

    contact = Contact(
        uid=uid,
        device_id=device_id,
        name=name,
        first_name=first_name,
        last_name=last_name,
        phones=phones,
        emails=emails,
        organization=organization,
        title=title,
        note=note,
        timestamp=int(timestamp),
        raw_vcard=raw_vcard_str,
        cached_at=time.time(),
    )
    if is_single_arg:
        return [contact]
    return contact


__all__ = [
    "PhoneNumber",
    "EmailAddress",
    "Contact",
    "normalize_phone_number",
    "unfold_vcard_lines",
    "parse_vcard_line",
    "resolve_contact_name",
    "parse_vcard",
]

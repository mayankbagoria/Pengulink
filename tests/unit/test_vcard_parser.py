"""Unit tests for RFC 2426 / 2.1 vCard parsing engine."""

import pytest

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


class TestVCardUnfolding:
    def test_unfold_crlf_space(self) -> None:
        raw = "BEGIN:VCARD\r\nFN:John \r\n Doe\r\nEND:VCARD\r\n"
        lines = unfold_vcard_lines(raw)
        assert any(line == "FN:John Doe" for line in lines)

    def test_unfold_crlf_tab(self) -> None:
        raw = "NOTE:Line one\r\n\tContinued note\r\n"
        lines = unfold_vcard_lines(raw)
        assert lines == ["NOTE:Line oneContinued note"]

    def test_unfold_lf_space(self) -> None:
        raw = "FN:Jane \n Smith\n"
        lines = unfold_vcard_lines(raw)
        assert lines == ["FN:Jane Smith"]

    def test_unfold_multiple_consecutive(self) -> None:
        raw = "NOTE:Part1\r\n Part2\r\n\tPart3\r\n Part4\r\n"
        lines = unfold_vcard_lines(raw)
        assert lines == ["NOTE:Part1Part2Part3Part4"]

    def test_unfold_empty_string(self) -> None:
        assert unfold_vcard_lines("") == []
        assert unfold_vcard_lines("\r\n\r\n") == []


class TestNameExtraction:
    def test_parse_fn_priority(self) -> None:
        vcard = (
            "BEGIN:VCARD\r\n"
            "VERSION:3.0\r\n"
            "N:Doe;John;;;\r\n"
            "FN:Johnny D.\r\n"
            "ORG:Acme Corp\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("1", vcard)
        assert c.name == "Johnny D."
        assert c.first_name == "John"
        assert c.last_name == "Doe"

    def test_parse_n_structured(self) -> None:
        vcard = (
            "BEGIN:VCARD\r\n"
            "N:Smith;Jane;Robert;Dr.;Jr.\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("2", vcard)
        assert c.first_name == "Jane"
        assert c.last_name == "Smith"
        assert c.name == "Jane Smith"

    def test_fallback_fn_to_n(self) -> None:
        vcard = (
            "BEGIN:VCARD\r\n"
            "N:Wayne;Bruce;;;\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("3", vcard)
        assert c.name == "Bruce Wayne"
        assert c.first_name == "Bruce"
        assert c.last_name == "Wayne"

    def test_fallback_n_to_org(self) -> None:
        vcard = (
            "BEGIN:VCARD\r\n"
            "ORG:Cyberdyne Systems\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("4", vcard)
        assert c.name == "Cyberdyne Systems"
        assert c.organization == "Cyberdyne Systems"

    def test_fallback_org_to_phone(self) -> None:
        vcard = (
            "BEGIN:VCARD\r\n"
            "TEL;TYPE=CELL:+15550199\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("5", vcard)
        assert c.name == "+15550199"

    def test_fallback_to_unknown(self) -> None:
        vcard = "BEGIN:VCARD\r\nVERSION:3.0\r\nEND:VCARD\r\n"
        c = parse_vcard("6", vcard)
        assert c.name == "Unknown Contact"


class TestPhoneNormalization:
    def test_phone_cell_type_and_normalization(self) -> None:
        vcard = (
            "BEGIN:VCARD\r\n"
            "FN:Alice\r\n"
            "TEL;TYPE=CELL,VOICE:+1 (555) 019-9283\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("10", vcard)
        assert len(c.phones) == 1
        p = c.phones[0]
        assert p.number == "+1 (555) 019-9283"
        assert p.normalized == "+15550199283"
        assert p.type == "CELL"

    def test_phone_work_home_types(self) -> None:
        vcard = (
            "BEGIN:VCARD\r\n"
            "FN:Bob\r\n"
            "TEL;TYPE=WORK:555-1234\r\n"
            "TEL;TYPE=HOME:+44 20 7946 0912\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("11", vcard)
        assert len(c.phones) == 2
        assert c.phones[0].type == "WORK"
        assert c.phones[0].normalized == "5551234"
        assert c.phones[1].type == "HOME"
        assert c.phones[1].normalized == "+442079460912"

    def test_phone_shortcodes(self) -> None:
        assert normalize_phone_number("12345") == "12345"
        assert normalize_phone_number("911") == "911"
        assert normalize_phone_number("*123#") == "*123#"

    def test_phone_preserves_international_prefix(self) -> None:
        assert normalize_phone_number("+49 30 123456") == "+4930123456"
        assert normalize_phone_number("  +1-800-555-0199  ") == "+18005550199"
        assert normalize_phone_number("0049 30 123") == "004930123"

    def test_multiple_phone_numbers(self) -> None:
        vcard = (
            "BEGIN:VCARD\r\n"
            "FN:Multi\r\n"
            "TEL;TYPE=CELL:111\r\n"
            "TEL;TYPE=WORK:222\r\n"
            "TEL;TYPE=HOME:333\r\n"
            "TEL;TYPE=FAX:444\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("12", vcard)
        assert len(c.phones) == 4
        assert [p.type for p in c.phones] == ["CELL", "WORK", "HOME", "FAX"]


class TestEmailAndDetails:
    def test_email_types_and_addresses(self) -> None:
        vcard = (
            "BEGIN:VCARD\r\n"
            "FN:Charlie\r\n"
            "EMAIL;TYPE=INTERNET,HOME:charlie@home.org\r\n"
            "EMAIL;TYPE=WORK:charlie@corp.com\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("20", vcard)
        assert len(c.emails) == 2
        assert c.emails[0].email == "charlie@home.org"
        assert c.emails[0].type in ("HOME", "INTERNET")
        assert c.emails[1].email == "charlie@corp.com"
        assert c.emails[1].type == "WORK"

    def test_organization_and_title(self) -> None:
        vcard = (
            "BEGIN:VCARD\r\n"
            "FN:Dan\r\n"
            "ORG:Google LLC;Android Platform\r\n"
            "TITLE:Staff Software Engineer\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("21", vcard)
        assert "Google LLC" in (c.organization or "")
        assert c.title == "Staff Software Engineer"

    def test_note_multiline(self) -> None:
        vcard = (
            "BEGIN:VCARD\r\n"
            "FN:Eva\r\n"
            "NOTE:Met at Open Source Summit 2026.\nFollow up about KDE Connect.\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("22", vcard)
        assert c.note is not None
        assert "Open Source Summit" in c.note
        assert "KDE Connect" in c.note

    def test_note_multiple_lines_joined(self) -> None:
        vcard = (
            "BEGIN:VCARD\r\n"
            "FN:Frank\r\n"
            "NOTE:First line\r\n"
            "NOTE:Second line\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("23", vcard)
        assert c.note == "First line\nSecond line"


class TestEncodingAndUnicode:
    def test_quoted_printable_decoding(self) -> None:
        # =E2=9C=A8 is ✨ (Sparkles in UTF-8)
        vcard = (
            "BEGIN:VCARD\r\n"
            "FN;ENCODING=QUOTED-PRINTABLE;CHARSET=UTF-8:=E2=9C=A8 Star\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("30", vcard)
        assert "✨" in c.name
        assert "Star" in c.name

    def test_utf8_cjk_names(self) -> None:
        vcard = (
            "BEGIN:VCARD\r\n"
            "FN:山田 太郎\r\n"
            "ORG:トヨタ自動車株式会社\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("31", vcard)
        assert c.name == "山田 太郎"
        assert c.organization == "トヨタ自動車株式会社"

    def test_utf8_arabic_and_cyrillic(self) -> None:
        vcard_ar = "BEGIN:VCARD\r\nFN:محمد بن سلمان\r\nEND:VCARD\r\n"
        c_ar = parse_vcard("32", vcard_ar)
        assert c_ar.name == "محمد بن سلمان"

        vcard_ru = "BEGIN:VCARD\r\nFN:Алексей Смирнов\r\nEND:VCARD\r\n"
        c_ru = parse_vcard("33", vcard_ru)
        assert c_ru.name == "Алексей Смирнов"

    def test_emojis_in_fields(self) -> None:
        vcard = (
            "BEGIN:VCARD\r\n"
            "FN:Developer 🚀\r\n"
            "NOTE:Works remotely 🌍🐧\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("34", vcard)
        assert "🚀" in c.name
        assert "🌍🐧" in (c.note or "")


class TestMalformedResilience:
    def test_missing_begin_end_tags(self) -> None:
        vcard = "FN:No Envelopes\r\nTEL;TYPE=CELL:5550000\r\n"
        c = parse_vcard("40", vcard)
        assert c.name == "No Envelopes"
        assert len(c.phones) == 1

    def test_empty_vcard_string(self) -> None:
        c = parse_vcard("41", "")
        assert c.uid == "41"
        assert c.name == "Unknown Contact"
        assert c.phones == []

    def test_binary_garbage_payload(self) -> None:
        garbage = "BEGIN:VCARD\r\n\x00\x01\x02\xff\xfeRANDOM_BIN\r\nFN:Survived\r\nEND:VCARD\r\n"
        c = parse_vcard("42", garbage)
        assert c.name == "Survived"

    def test_invalid_property_syntax(self) -> None:
        vcard = (
            "BEGIN:VCARD\r\n"
            "INVALID_NO_COLON_LINE\r\n"
            ":EMPTY_PROP\r\n"
            "FN:Valid Name\r\n"
            "END:VCARD\r\n"
        )
        c = parse_vcard("43", vcard)
        assert c.name == "Valid Name"


class TestDataclassSerialization:
    def test_contact_to_dict_and_from_dict(self) -> None:
        original = Contact(
            uid="50",
            device_id="dev_123",
            name="John Doe",
            first_name="John",
            last_name="Doe",
            phones=[PhoneNumber(number="+1-555-0199", normalized="+15550199", type="CELL")],
            emails=[EmailAddress(email="john@example.com", type="HOME")],
            organization="Acme",
            title="Engineer",
            note="Test note",
            timestamp=1712345678000,
            raw_vcard="BEGIN:VCARD...",
            cached_at=1234567.89,
        )
        d = original.to_dict()
        assert d["uid"] == "50"
        assert d["device_id"] == "dev_123"
        assert d["phones"][0]["normalized"] == "+15550199"

        restored = Contact.from_dict(d)
        assert restored.uid == original.uid
        assert restored.name == original.name
        assert len(restored.phones) == 1
        assert restored.phones[0].normalized == "+15550199"
        assert restored.emails[0].email == "john@example.com"
        assert restored.timestamp == 1712345678000

    def test_phone_and_email_from_dict(self) -> None:
        p = PhoneNumber.from_dict({"number": "123", "normalized": "123", "type": "WORK"})
        assert p.number == "123"
        assert p.type == "WORK"

        e = EmailAddress.from_dict({"email": "test@test.org", "type": "WORK"})
        assert e.email == "test@test.org"
        assert e.type == "WORK"

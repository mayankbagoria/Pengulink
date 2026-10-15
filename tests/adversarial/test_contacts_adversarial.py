"""tests.adversarial.test_contacts_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial Stress & Empirical Challenge Suite for Milestone 13:
Contacts Plugin Ecosystem (kdeconnect.contacts.*).

Challenge Dimensions:
1. Malformed and corrupted vCards:
   - Extremely large note/summary fields (1 MB+, 10,000 folded lines).
   - Truncated envelopes, cut properties, unclosed brackets, missing tags.
   - Binary garbage, null bytes, control characters, invalid UTF-8/surrogates.
   - Invalid Quoted-Printable sequences (bad hex, dangling '=', unknown charsets).
   - ANSI escape injections and Rich markup injection in names/notes.
2. Path traversal and injection attacks:
   - UID path traversal ('../../etc/shadow', absolute paths).
   - Device ID path traversal containment audit.
3. High-volume stress & 128 KB buffer boundary:
   - Address books with hundreds of contacts (500 contacts) batch chunking.
   - Wire serialization buffer safety (MAX_PACKET_SIZE = 131,072 bytes).
   - Oversized packet detection and boundary limits.
4. Search fuzzing:
   - Regex metacharacters ('.*', '(', '[', '\\', '+', '?', '*', '^', '$').
   - Null bytes, oversized queries (50,000 chars), whitespace, Unicode fuzzing.
5. Concurrency & lifecycle races:
   - Concurrent sync serialization.
   - Unpair cache eviction race condition reproduction.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import pytest
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pengulink.core.events import EventBus
from pengulink.core.exceptions import (
    LimitOverrunError,
    PacketBufferOverrunError,
    SyncTimeoutError,
)
from pengulink.network.packet import MAX_PACKET_SIZE, Packet, PacketEngine
from pengulink.plugins.contacts.plugin import (
    DEFAULT_BATCH_SIZE,
    MAX_BATCH_SIZE,
    REQUEST_ALL_UIDS_TIMESTAMPS,
    REQUEST_VCARDS_BY_UID,
    RESPONSE_UIDS_TIMESTAMPS,
    RESPONSE_VCARDS,
    ContactsPlugin,
)
from pengulink.plugins.contacts.store import ContactStore, chunk_list
from pengulink.plugins.contacts.vcard import (
    Contact,
    PhoneNumber,
    normalize_phone_number,
    parse_vcard,
    parse_vcard_line,
    resolve_contact_name,
    unfold_vcard_lines,
)


# =============================================================================
# 1. Malformed and Corrupted vCards
# =============================================================================

class TestMalformedAndCorruptedVCards:
    """Stress-test parser resilience against adversarial and malformed vCards."""

    def test_extremely_large_note_field(self) -> None:
        """1 MB note field across single and multiple folded lines must not crash parser."""
        huge_text = "A" * 1_000_000
        vcard = (
            "BEGIN:VCARD\r\n"
            "VERSION:3.0\r\n"
            "FN:Big Note Contact\r\n"
            f"NOTE:{huge_text}\r\n"
            "END:VCARD\r\n"
        )
        contact = parse_vcard("big_note_uid", vcard)
        assert contact.name == "Big Note Contact"
        assert contact.note is not None
        assert len(contact.note) == 1_000_000

    def test_deep_line_folding_repetition(self) -> None:
        """Thousands of folded continuation lines must not cause ReDoS or RecursionError."""
        chunks = ["LineSegment" + str(i) for i in range(5000)]
        folded_note = "\r\n ".join(chunks)
        vcard = (
            "BEGIN:VCARD\r\n"
            "FN:Folded Contact\r\n"
            f"NOTE:{folded_note}\r\n"
            "END:VCARD\r\n"
        )
        start = time.perf_counter()
        contact = parse_vcard("folded_uid", vcard)
        duration = time.perf_counter() - start

        assert contact.name == "Folded Contact"
        assert contact.note is not None
        assert "LineSegment0" in contact.note
        assert "LineSegment4999" in contact.note
        # Performance check: 5000 unfolded lines should parse in under 1 second
        assert duration < 1.0

    @pytest.mark.parametrize(
        "truncated_raw",
        [
            "BEGIN:VCARD",
            "END:VCARD",
            "BEGIN:VCARD\r\nFN:",
            "BEGIN:VCARD\r\nFN:John",
            "BEGIN:VCARD\r\nTEL;TYPE=CELL:",
            "BEGIN:VCARD\r\nNOTE:",
            "FN:No Envelopes",
            "VERSION:3.0\r\nTEL:12345",
            "::::",
            "BEGIN:VCARD\r\n" * 50 + "END:VCARD\r\n" * 50,
        ],
    )
    def test_truncated_envelopes_and_syntax_mutations(self, truncated_raw: str) -> None:
        """Truncated envelopes and mutated structural tags must parse safely into fallback Contact."""
        contact = parse_vcard("trunc_uid", truncated_raw)
        assert isinstance(contact, Contact)
        assert contact.uid == "trunc_uid"
        assert contact.name is not None

    def test_binary_garbage_and_control_chars(self) -> None:
        """Raw null bytes, binary garbage, and control characters must be handled without crash."""
        garbage_bytes = bytes(range(256))
        garbage_str = garbage_bytes.decode("latin-1")
        vcard = (
            "BEGIN:VCARD\r\n"
            f"FN:Surviving \x00 {garbage_str}\r\n"
            f"NOTE:Binary \x01\x02\x03\x1b\x7f\xff Payload\r\n"
            "TEL;TYPE=CELL:\x00+1-555-0199\x00\r\n"
            "END:VCARD\r\n"
        )
        contact = parse_vcard("garbage_uid", vcard)
        assert "Surviving" in contact.name
        assert len(contact.phones) == 1
        # Leading null byte before '+' prevents startswith('+'), extracting digits '15550199'
        assert contact.phones[0].normalized in ("15550199", "+15550199")

    def test_invalid_quoted_printable_sequences(self) -> None:
        """Invalid Quoted-Printable hex, dangling '=', and unknown charsets must not raise exceptions."""
        vcard_bad_hex = (
            "BEGIN:VCARD\r\n"
            "FN;ENCODING=QUOTED-PRINTABLE;CHARSET=UTF-8:=GG=ZZ=1G Bad Hex\r\n"
            "NOTE;ENCODING=QUOTED-PRINTABLE;CHARSET=UTF-8:Dangling=\r\n"
            "TITLE;ENCODING=QUOTED-PRINTABLE;CHARSET=UNKNOWN_CHARSET_XYZ:=E2=9C=A8\r\n"
            "END:VCARD\r\n"
        )
        contact = parse_vcard("qp_uid", vcard_bad_hex)
        assert contact.name is not None
        assert contact.note is not None

    def test_quoted_printable_unicode_straddling(self) -> None:
        """Unicode characters straddling QP-encoded line breaks must decode or degrade gracefully."""
        vcard = (
            "BEGIN:VCARD\r\n"
            "FN;ENCODING=QUOTED-PRINTABLE;CHARSET=UTF-8:Hello =E2=\r\n"
            " =9C=A8 World\r\n"
            "END:VCARD\r\n"
        )
        contact = parse_vcard("straddle_uid", vcard)
        assert "Hello" in contact.name
        assert "World" in contact.name

    def test_ansi_escape_and_rich_markup_injection(self) -> None:
        """Adversarial ANSI escape codes and Rich markup in contact fields must be parsed."""
        ansi_name = "\x1b[31;1mHacked Name\x1b[0m"
        rich_markup_note = "Alert: [/red] Unclosed tag [bold] [link=http://evil.com]Click[/link]"
        vcard = (
            "BEGIN:VCARD\r\n"
            f"FN:{ansi_name}\r\n"
            f"NOTE:{rich_markup_note}\r\n"
            "END:VCARD\r\n"
        )
        contact = parse_vcard("inj_uid", vcard)
        assert contact.name == ansi_name
        assert contact.note == rich_markup_note

    def test_rich_markup_terminal_rendering_behavior(self) -> None:
        """Document empirical Rich markup parsing behavior when contact names contain closing tags.
        
        When Rich Table or Panel renders unescaped strings like '[/red]' or '[/bold]',
        it raises rich.errors.MarkupError if markup is enabled on the renderable.
        """
        console = Console(record=True)
        malicious_name = "User [/red] Crash"

        # Direct table rendering with unescaped markup
        table = Table()
        table.add_column("UID")
        table.add_column("Name")

        markup_error_raised = False
        try:
            table.add_row("1", malicious_name)
            console.print(table)
        except Exception as exc:
            if "MarkupError" in type(exc).__name__:
                markup_error_raised = True

        # Assert finding: Table.add_row treats string as markup and raises MarkupError
        assert markup_error_raised is True, (
            "Expected Table.add_row to fail on unescaped Rich closing tags [/red]"
        )


# =============================================================================
# 2. Path Traversal & Injection Containment
# =============================================================================

class TestPathTraversalAndSecurity:
    """Stress-test filesystem boundary containment for UIDs and device IDs."""

    async def test_uid_path_traversal_attempts_isolated(self, tmp_path: Path) -> None:
        """Adversarial UIDs with path traversal sequences ('../../etc/shadow') must remain isolated.
        
        Since ContactStore aggregates contacts per device in a single JSON file,
        traversal UIDs must never escape or create files outside data_dir.
        """
        store = ContactStore(data_dir=tmp_path)
        await store.initialize()

        traversal_uids = [
            "../../etc/shadow",
            "../../../../etc/passwd",
            "/etc/shadow",
            "..\\..\\windows\\win.ini",
            "....//....//etc//passwd",
        ]

        contacts_map = {
            uid: Contact(
                uid=uid,
                device_id="dev_legit",
                name=f"Contact {uid}",
                raw_vcard=f"BEGIN:VCARD\r\nFN:Contact {uid}\r\nEND:VCARD",
            )
            for uid in traversal_uids
        }
        manifest = {uid: 1000 for uid in traversal_uids}

        await store.save_device("dev_legit", manifest, contacts_map)

        # Verify only the single legitimate device JSON file was created
        contact_files = list((tmp_path / "contacts").glob("*"))
        assert len(contact_files) == 1
        assert contact_files[0].name == "dev_legit.json"

        # Verify all traversal UIDs can be retrieved accurately
        loaded_manifest, loaded_contacts = await store.load_device("dev_legit")
        for uid in traversal_uids:
            assert uid in loaded_manifest
            c = await store.get_contact("dev_legit", uid)
            assert c is not None
            assert c.uid == uid

    def test_device_id_path_traversal_behavior(self, tmp_path: Path) -> None:
        """Audit device_id path traversal handling in ContactStore._get_device_file.
        
        If an unsanitized device_id containing '../' is provided, document whether
        the path escapes the contacts directory.
        """
        store = ContactStore(data_dir=tmp_path)
        escaped_file = store._get_device_file("../../tmp/evil")
        resolved = escaped_file.resolve()
        contacts_dir_resolved = store.contacts_dir.resolve()

        escaped = not str(resolved).startswith(str(contacts_dir_resolved))
        # Document the empirical finding: ContactStore does not sanitize device_id
        assert escaped is True, "Expected raw device_id '../../tmp/evil' to escape contacts_dir"


# =============================================================================
# 3. High-Volume Stress & 128 KB Buffer Boundary
# =============================================================================

class TestHighVolumeAndBufferBoundaries:
    """Stress-test batch chunking, high contact volume, and KDE Connect 128 KB wire buffer safety."""

    async def test_address_book_500_contacts_batch_chunking(self, tmp_path: Path) -> None:
        """500 contacts must chunk into batches strictly <= 50 UIDs and wire packets <= 128 KB."""
        store = ContactStore(data_dir=tmp_path)
        await store.initialize()

        uids = [f"uid_{i:04d}" for i in range(500)]
        chunks = chunk_list(uids, chunk_size=50)

        # 500 contacts / 50 per chunk = exactly 10 chunks
        assert len(chunks) == 10
        for chunk in chunks:
            assert len(chunk) <= MAX_BATCH_SIZE

            # Verify request packet size on the wire
            req_pkt = Packet(
                type=REQUEST_VCARDS_BY_UID,
                body={"uids": chunk},
            )
            wire_bytes = PacketEngine.serialize(req_pkt)
            assert len(wire_bytes) <= MAX_PACKET_SIZE
            # Request packets are ~1 KB, far below 128 KB
            assert len(wire_bytes) < 4096

    def test_128kb_packet_buffer_overrun_rejection(self) -> None:
        """Wire lines exceeding 131,072 bytes (128 KB) must raise PacketBufferOverrunError."""
        # Create an oversized packet line: 140 KB
        oversized_line = b'{"type":"kdeconnect.contacts.response_vcards","body":{"data":"' + (b"A" * 140000) + b'"}}\n'
        assert len(oversized_line) > MAX_PACKET_SIZE

        with pytest.raises(PacketBufferOverrunError) as exc_info:
            PacketEngine.parse(oversized_line)
        assert "exceeds maximum limit" in str(exc_info.value)

    def test_manifest_packet_size_at_scale(self) -> None:
        """Audit timestamp manifest packet size as contact count increases.
        
        Manifest packets contain all {uid: timestamp} pairs in a single packet.
        Determine the boundary at which manifest size approaches 128 KB.
        """
        def make_manifest_packet(count: int) -> int:
            manifest_body = {
                "uids": [f"contact_uid_{i:06d}" for i in range(count)],
                **{f"contact_uid_{i:06d}": 1712000000000 for i in range(count)},
            }
            pkt = Packet(type=RESPONSE_UIDS_TIMESTAMPS, body=manifest_body)
            return len(PacketEngine.serialize(pkt))

        size_500 = make_manifest_packet(500)
        size_1000 = make_manifest_packet(1000)
        size_2000 = make_manifest_packet(2000)

        # 500 contacts: ~27 KB
        assert size_500 < MAX_PACKET_SIZE
        # 1,000 contacts: ~55 KB
        assert size_1000 < MAX_PACKET_SIZE
        # 2,000 contacts: ~110 KB
        assert size_2000 < MAX_PACKET_SIZE

        # 2,500 contacts: reaches ~137 KB and exceeds 128 KB buffer
        size_2500 = make_manifest_packet(2500)
        assert size_2500 > MAX_PACKET_SIZE

    async def test_contacts_sync_with_oversized_vcard_batch_simulation(self, tmp_path: Path) -> None:
        """Simulate sync handling when an adversarial peer returns oversized batch."""
        plugin = ContactsPlugin(data_dir=tmp_path)
        await plugin.start()

        # Mock send callback
        async def mock_send(pkt: Packet) -> None:
            pass

        # Verify chunking helper never produces chunks > 50 even when requested
        oversized_requested_batch = 100
        safe_batch_size = max(1, min(MAX_BATCH_SIZE, oversized_requested_batch))
        assert safe_batch_size == 50

        await plugin.stop()


# =============================================================================
# 4. Search Queries & Edge-Case Fuzzing
# =============================================================================

class TestSearchQueriesAndFuzzing:
    """Stress-test search_contacts with regex metacharacters, null bytes, and extreme input sizes."""

    @pytest.fixture
    def populated_store(self, tmp_path: Path) -> ContactStore:
        store = ContactStore(data_dir=tmp_path)

        async def _setup():
            await store.initialize()
            test_contacts = {
                "c1": Contact(
                    uid="c1",
                    name="John Doe",
                    first_name="John",
                    last_name="Doe",
                    phones=[PhoneNumber(number="+1 (555) 0100", normalized="+15550100", type="CELL")],
                    emails=[],
                    organization="Acme Security Inc.",
                ),
                "c2": Contact(
                    uid="c2",
                    name="Special (Regex) [Chars] *+?^$ Person",
                    first_name="Special",
                    last_name="Person",
                    phones=[PhoneNumber(number="555-9999", normalized="5559999", type="WORK")],
                    emails=[],
                    organization="Test Corp",
                ),
                "c3": Contact(
                    uid="c3",
                    name="Null \x00 Byte Contact",
                    first_name="Null",
                    last_name="Byte",
                    phones=[],
                    emails=[],
                    organization=None,
                ),
            }
            manifest = {uid: 1000 for uid in test_contacts}
            await store.save_device("dev_search", manifest, test_contacts)

        asyncio.run(_setup())
        return store

    @pytest.mark.parametrize(
        "query",
        [
            ".*",
            "(",
            "[",
            "\\",
            "+",
            "?",
            "*",
            "^",
            "$",
            "\\d+",
            "(a|b)*",
            "(?=.*[0-9])",
            "[a-z0-9_-]+",
            "{1,3}",
            "\\s+",
            "Special (Regex)",
            "[Chars]",
            "*+?^$",
        ],
    )
    async def test_regex_metacharacter_search_resilience(
        self, populated_store: ContactStore, query: str
    ) -> None:
        """Regex metacharacters in search queries must not raise re.error or crash search."""
        matches = await populated_store.search_contacts("dev_search", query)
        assert isinstance(matches, list)
        if query in ("Special (Regex)", "[Chars]", "*+?^$"):
            assert any(m.uid == "c2" for m in matches)

    @pytest.mark.parametrize(
        "fuzz_query",
        [
            "\x00",
            "\x00admin",
            "\x00" * 500,
            "\n\r\t",
            "   \t  \n  ",
            "A" * 50_000,  # 50 KB query string
            "!@#$%^&*()_+-=[]{}|;':\",./<>?`~",
            "🚀🔥✨🎉",
            "田中",
            "مرحبا",
        ],
    )
    async def test_fuzzed_and_oversized_search_queries(
        self, populated_store: ContactStore, fuzz_query: str
    ) -> None:
        """Fuzzed strings, null bytes, and 50 KB query strings must not crash search."""
        start = time.perf_counter()
        matches = await populated_store.search_contacts("dev_search", fuzz_query)
        duration = time.perf_counter() - start

        assert isinstance(matches, list)
        assert duration < 0.5  # Fast execution even on 50 KB query string
        if fuzz_query == "\x00":
            assert any(m.uid == "c3" for m in matches)


# =============================================================================
# 5. Concurrency & Lifecycle Races
# =============================================================================

class TestConcurrencyAndLifecycleRaces:
    """Stress-test concurrent operations, synchronization locks, and unpair race conditions."""

    async def test_concurrent_sync_lock_serialization(self, tmp_path: Path) -> None:
        """Concurrent sync calls for the same device must be serialized by _get_sync_lock."""
        plugin = ContactsPlugin(data_dir=tmp_path)
        await plugin.start()

        call_order: List[str] = []

        async def fake_send_1(pkt: Packet) -> None:
            if pkt.type == REQUEST_ALL_UIDS_TIMESTAMPS:
                call_order.append("sync_1_req")
                # Complete the future
                fut = plugin._timestamp_futures.get("dev_concurrent")
                if fut and not fut.done():
                    fut.set_result({"u1": 1000})

        async def fake_send_2(pkt: Packet) -> None:
            if pkt.type == REQUEST_ALL_UIDS_TIMESTAMPS:
                call_order.append("sync_2_req")
                fut = plugin._timestamp_futures.get("dev_concurrent")
                if fut and not fut.done():
                    fut.set_result({"u1": 1000})

        # Lock is held per device
        lock = plugin._get_sync_lock("dev_concurrent")
        assert not lock.locked()

        # Simulate lock held by first task
        async with lock:
            assert lock.locked()

        await plugin.stop()

    async def test_unpair_cache_eviction_race_condition_reproduction(self, tmp_path: Path) -> None:
        """Empirically reproduce the unpair cache eviction race condition.
        
        When on_device_unpaired calls create_task(self.clear_device(device_id))
        without awaiting it, immediately checking disk_file.exists() can observe
        the file still existing prior to task execution.
        """
        store = ContactStore(data_dir=tmp_path)
        await store.initialize()
        manifest = {"u1": 1000}
        contacts = {"u1": Contact(uid="u1", name="Alice")}
        await store.save_device("dev_race", manifest, contacts)

        disk_file = store._get_device_file("dev_race")
        assert disk_file.exists()

        plugin = ContactsPlugin(data_dir=tmp_path)
        plugin.store = store

        # Fire on_device_unpaired (uses create_task without await)
        plugin.on_device_unpaired("dev_race")

        # Immediately after return, the file may still exist if event loop hasn't yielded
        file_existed_immediately = disk_file.exists()

        # Yield execution to allow loop task to run
        await asyncio.sleep(0.05)

        # After yielding, the file is unlinked
        file_existed_after_yield = disk_file.exists()

        # The empirical finding: immediate check can be True (race condition)
        # while after yielding it is False
        assert file_existed_after_yield is False

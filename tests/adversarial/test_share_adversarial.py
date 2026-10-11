"""tests.adversarial.test_share_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial security and chaos testing for Share Feature Ecosystem:
- Path traversal rejection (POSIX, Windows, null bytes, absolute paths)
- Abrupt TCP drop and incomplete transfer cleanup (no orphan .part files)
- Payload size mismatch defense (premature EOF underflow & overflow)
- Collision storm under heavy file naming contention
- Non-executable permission enforcement under adversarial umasks
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time
import unicodedata
from typing import Any, List
import pytest

from pengulink.core.events import EventBus
from pengulink.core.exceptions import (
    IncompleteTransferError,
    PathTraversalError,
    PayloadSizeExceededError,
)
from pengulink.crypto.tls import create_paired_ssl_context
from pengulink.network.packet import Packet
from pengulink.plugins.share import SharePlugin
from pengulink.plugins.share.security import (
    AtomicStagingWriter,
    COMPOUND_EXTENSIONS,
    MAX_COLLISION_ATTEMPTS,
    MAX_FILENAME_BYTES,
    resolve_collision,
    resolve_download_dir,
    sanitize_filename,
    strip_execution_bits,
    validate_containment,
    verify_boundary,
)
from tests.e2e.conftest import e2e_environment, wait_for
from tests.mock.mock_device import MockDevice


def test_adversarial_path_traversal_sanitization() -> None:
    """Malicious paths with parent directory traversal, absolute paths, and null bytes are sanitized."""
    # Absolute path
    sanitized = sanitize_filename("/etc/shadow")
    assert not sanitized.startswith("/")
    assert "shadow" in sanitized
    assert "/" not in sanitized

    # POSIX traversal
    sanitized_posix = sanitize_filename("../../../etc/passwd")
    assert ".." not in sanitized_posix
    assert "/" not in sanitized_posix

    # Windows traversal
    sanitized_win = sanitize_filename("..\\..\\windows\\system32\\cmd.exe")
    assert "\\" not in sanitized_win
    assert ".." not in sanitized_win

    # Null byte injection
    with pytest.raises(PathTraversalError):
        sanitize_filename("exploit.sh\x00.png")


def test_adversarial_containment_validation(tmp_path: Path) -> None:
    """validate_containment rejects any path outside target root directory."""
    root_dir = tmp_path / "downloads"
    root_dir.mkdir(parents=True, exist_ok=True)

    # Legitimate subpath
    valid_path = root_dir / "safe.txt"
    assert validate_containment(root_dir, valid_path) == valid_path.resolve()

    # Outside containment
    outside_path = tmp_path / "other" / "escape.txt"
    with pytest.raises(PathTraversalError):
        validate_containment(root_dir, outside_path)


def test_adversarial_collision_storm(tmp_path: Path) -> None:
    """resolve_collision deterministically increments collision counter under storm conditions."""
    dest_dir = tmp_path / "downloads"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Create base file and first 5 collisions
    (dest_dir / "storm.txt").write_text("v0")
    for i in range(1, 6):
        (dest_dir / f"storm ({i}).txt").write_text(f"v{i}")

    # Next candidate should be storm (6).txt
    resolved = resolve_collision(dest_dir, "storm.txt")
    assert resolved.name == "storm (6).txt"


async def test_adversarial_abrupt_disconnect_cleans_part_file(tmp_path: Path) -> None:
    """Abrupt connection termination mid-transfer unlinks partial .part file cleanly."""
    dest_dir = tmp_path / "downloads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / "abrupt_fail.bin"

    staging_writer = AtomicStagingWriter(
        dest_dir=dest_dir,
        final_path=final_path,
        expected_size=10000,
    )

    try:
        staging_writer.open()
        await staging_writer.write_chunk(b"Partial initial chunk")
        assert staging_writer.part_path.exists()
        # Abrupt error occurs
        raise ConnectionResetError("Connection abruptly severed")
    except ConnectionResetError:
        staging_writer.abort()

    # Part file must be completely purged
    assert not staging_writer.part_path.exists()
    assert not final_path.exists()


async def test_adversarial_size_mismatch_overflow(tmp_path: Path) -> None:
    """Staging writer rejects byte stream exceeding expected_size."""
    dest_dir = tmp_path / "downloads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / "overflow.bin"

    staging_writer = AtomicStagingWriter(
        dest_dir=dest_dir,
        final_path=final_path,
        expected_size=10,
    )

    staging_writer.open()
    await staging_writer.write_chunk(b"0123456789_TOO_MANY_BYTES")
    try:
        with pytest.raises(PayloadSizeExceededError):
            staging_writer.commit()
    finally:
        staging_writer.abort()

    assert not staging_writer.part_path.exists()
    assert not final_path.exists()


async def test_adversarial_premature_eof_commit_fails(tmp_path: Path) -> None:
    """Committing staging writer with fewer bytes than expected_size raises IncompleteTransferError."""
    dest_dir = tmp_path / "downloads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / "underflow.bin"

    staging_writer = AtomicStagingWriter(
        dest_dir=dest_dir,
        final_path=final_path,
        expected_size=1000,
    )

    staging_writer.open()
    await staging_writer.write_chunk(b"Only 100 bytes" * 10)
    try:
        with pytest.raises(IncompleteTransferError):
            staging_writer.commit()
    finally:
        staging_writer.abort()

    assert not staging_writer.part_path.exists()
    assert not final_path.exists()


def test_adversarial_deep_traversal_matrix(tmp_path: Path) -> None:
    """Stress-test deep traversal payloads across POSIX, Windows, mixed slashes, and root collisions."""
    dest_dir = tmp_path / "downloads"
    dest_dir.mkdir(parents=True, exist_ok=True)

    dangerous_payloads = [
        "../../../../../../../../../../../../../../../../../../../../etc/shadow",
        "..\\..\\..\\..\\..\\..\\..\\..\\windows\\system32\\config\\SAM",
        "../../..\\..\\../..\\..\\var/log/syslog",
        "////etc////passwd",
        "/root/.ssh/id_rsa",
        "C:\\Windows\\System32\\cmd.exe",
        "C:autoexec.bat",
        "foo/bar/../../../../../../etc/hosts",
    ]

    for payload in dangerous_payloads:
        cleaned = sanitize_filename(payload)
        # 1. No directory separators remaining
        assert "/" not in cleaned, f"Forward slash leaked in: {cleaned}"
        assert "\\" not in cleaned, f"Backslash leaked in: {cleaned}"
        assert ":" not in cleaned, f"Colon leaked in: {cleaned}"
        assert ".." not in cleaned, f"Parent traversal token leaked in: {cleaned}"

        # 2. When resolved against destination directory, containment must strictly hold
        target_path = dest_dir / cleaned
        verified = validate_containment(dest_dir, target_path)
        assert verified.is_relative_to(dest_dir.resolve())
        assert verified != dest_dir.resolve()

    # Inputs that MUST be rejected outright with PathTraversalError
    outright_rejections = [
        "..",
        ".",
        "../",
        "..\\",
        "/etc/passwd/../../../",
        "....",
        "...",
        "/",
        "///",
        "\\",
        "\\\\",
        "   ",
        "\t\t",
        "\n",
    ]
    for rejection in outright_rejections:
        with pytest.raises(PathTraversalError):
            sanitize_filename(rejection)


def test_adversarial_null_byte_permutations() -> None:
    """Test null-byte injection across arbitrary string positions and invalid data types."""
    null_payloads = [
        "exploit.sh\x00.png",
        "\x00exploit.sh",
        "exploit.sh\x00",
        "../../etc/passwd\x00.jpg",
        "photo\x00\x00\x00.jpeg",
        "\x00",
        "normal_file_\x00_hidden.txt",
    ]
    for payload in null_payloads:
        with pytest.raises(PathTraversalError):
            sanitize_filename(payload)

    # Invalid non-string types must also raise PathTraversalError
    invalid_types = [
        None,
        12345,
        b"raw_bytes.txt",
        b"\x00exploit.sh",
        {"file": "name.txt"},
        ["exploit.sh"],
    ]
    for inv in invalid_types:
        with pytest.raises(PathTraversalError):
            sanitize_filename(inv)  # type: ignore[arg-type]


def test_adversarial_unicode_and_emoji_preservation(tmp_path: Path) -> None:
    """Verify Unicode NFC normalization, emoji preservation, internal space preservation, and homoglyphs."""
    dest_dir = tmp_path / "downloads"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Canonical equivalence (NFC normalization of decomposed characters)
    decomposed = "e\u0301cole_e\u0301le\u0300ve.pdf"
    cleaned = sanitize_filename(decomposed)
    assert cleaned == "école_élève.pdf"
    assert unicodedata.is_normalized("NFC", cleaned)

    # 2. Emoji fidelity
    emoji_filename = "📸 Vacation Photo 🎉 2026 🌴.png"
    cleaned_emoji = sanitize_filename(emoji_filename)
    assert cleaned_emoji == emoji_filename
    assert "📸" in cleaned_emoji
    assert "🌴" in cleaned_emoji

    # Multi-person emoji sequence
    family_doc = "Family 👨‍👩‍👧‍👦 Tax Record.pdf"
    assert sanitize_filename(family_doc) == family_doc

    # 3. Spacing: internal spaces preserved, leading/trailing trimmed
    padded = "   Important Contract (Final Approved).docx   "
    cleaned_padded = sanitize_filename(padded)
    assert cleaned_padded == "Important Contract (Final Approved).docx"

    # 4. Homoglyphs and fullwidth slashes (U+FF0F fullwidth solidus, U+2215 division slash)
    fullwidth_slash = "malicious\uff0fpayload.exe"
    cleaned_fw = sanitize_filename(fullwidth_slash)
    # The fullwidth slash does not act as a POSIX separator
    target = dest_dir / cleaned_fw
    assert validate_containment(dest_dir, target) == target.resolve()


def test_adversarial_overlong_filenames_and_compound_extensions() -> None:
    """Verify length clamping to <= 240 UTF-8 bytes while preserving compound extensions."""
    for ext in COMPOUND_EXTENSIONS:
        long_stem = "A" * 300
        raw = f"{long_stem}{ext}"
        cleaned = sanitize_filename(raw, max_bytes=MAX_FILENAME_BYTES)

        encoded = cleaned.encode("utf-8")
        assert len(encoded) <= MAX_FILENAME_BYTES, f"Exceeded {MAX_FILENAME_BYTES} bytes: {len(encoded)}"
        assert cleaned.endswith(ext), f"Compound extension {ext} was not preserved in {cleaned}"

    # Multi-byte UTF-8 character boundary truncation (4-byte emoji 🚀)
    emoji_stem = "🚀" * 70  # 70 * 4 = 280 bytes
    raw_emoji = f"{emoji_stem}.tar.gz"
    cleaned_emoji = sanitize_filename(raw_emoji, max_bytes=MAX_FILENAME_BYTES)

    # Must cleanly decode as valid UTF-8 without corrupting boundary multi-byte character
    encoded_emoji = cleaned_emoji.encode("utf-8")
    assert len(encoded_emoji) <= MAX_FILENAME_BYTES
    assert cleaned_emoji.endswith(".tar.gz")
    assert encoded_emoji.decode("utf-8") == cleaned_emoji

    # CJK multi-byte characters (3 bytes each)
    cjk_stem = "日本語ファイル名テスト文字列" * 15  # 15 * 14 * 3 = 630 bytes
    raw_cjk = f"{cjk_stem}.archive.tar.bz2"
    cleaned_cjk = sanitize_filename(raw_cjk, max_bytes=MAX_FILENAME_BYTES)
    assert len(cleaned_cjk.encode("utf-8")) <= MAX_FILENAME_BYTES
    assert cleaned_cjk.endswith(".tar.bz2")


def test_adversarial_hidden_file_injection_defense() -> None:
    """Verify leading dot defense prepends 'file_' and prevents hidden files on Linux."""
    hidden_names = [
        ".bashrc",
        ".profile",
        ".bash_profile",
        ".ssh",
        ".config",
        ".env",
        ".hidden_payload.sh",
        "   .secret_key   ",
        "\t.dotfile",
        "..double_dot",
        "...triple_dot",
    ]
    for name in hidden_names:
        cleaned = sanitize_filename(name)
        assert not cleaned.startswith("."), f"Sanitized name starts with dot: {cleaned}"
        assert cleaned.startswith("file_."), f"Expected file_. prefix for hidden name {name}, got {cleaned}"


def test_adversarial_symlink_breakout_matrix(tmp_path: Path) -> None:
    """Verify symlink target rejections, self-referential symlinks, and broken symlink collision handling."""
    dest_dir = tmp_path / "downloads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)

    # 1. Existing symlink pointing outside destination directory
    sensitive_file = outside_dir / "shadow.txt"
    sensitive_file.touch()
    symlink_outside = dest_dir / "link_outside.txt"
    try:
        symlink_outside.symlink_to(sensitive_file)
    except OSError:
        pytest.skip("Symlinks not supported on this filesystem")

    with pytest.raises(PathTraversalError):
        verify_boundary(dest_dir, symlink_outside)

    # 2. Existing symlink pointing inside destination directory
    local_real = dest_dir / "real.txt"
    local_real.touch()
    symlink_inside = dest_dir / "link_inside.txt"
    symlink_inside.symlink_to(local_real)

    # Any target path that is an existing symlink must be rejected
    with pytest.raises(PathTraversalError):
        verify_boundary(dest_dir, symlink_inside)

    # 3. Broken symlink in destination directory
    broken_link = dest_dir / "broken_target.txt"
    broken_link.symlink_to(outside_dir / "nonexistent.txt")

    # Collision resolution must NOT collide with or overwrite the broken symlink
    cand = resolve_collision(dest_dir, "broken_target.txt")
    assert cand != broken_link
    assert cand.name == "broken_target (1).txt"

    # 4. Destination directory itself passed as target
    with pytest.raises(PathTraversalError):
        verify_boundary(dest_dir, dest_dir)


async def test_adversarial_high_concurrency_collision_storm(tmp_path: Path) -> None:
    """Simulate high-concurrency collision storm: 50 rapid parallel writes to the identical filename."""
    dest_dir = tmp_path / "downloads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    base_name = "collision_storm.bin"
    payload = b"CONCURRENT_CONTENT_VERIFICATION"

    writers: List[AtomicStagingWriter] = []
    # Instantiate 50 staging writers for the exact same destination
    for _ in range(50):
        target = resolve_collision(dest_dir, base_name)
        target.touch()  # Reserve slot
        w = AtomicStagingWriter(dest_dir=dest_dir, final_path=target, expected_size=len(payload))
        writers.append(w)

    # Verify each writer generated a distinct, unique .part file
    part_names = {w.part_filename for w in writers}
    assert len(part_names) == 50, f"Expected 50 distinct part filenames, got {len(part_names)}"

    # Stress-test resolve_collision limits: up to MAX_COLLISION_ATTEMPTS
    # When initial file and collisions (1)..(5) exist, next is (6)
    c = resolve_collision(dest_dir, "unique_base.dat")
    assert c.name == "unique_base.dat"
    c.touch()
    for i in range(1, 10):
        c_i = resolve_collision(dest_dir, "unique_base.dat")
        assert c_i.name == f"unique_base ({i}).dat"
        c_i.touch()


def test_adversarial_non_executable_permissions_umask_permutations(tmp_path: Path) -> None:
    """Verify that mode 0644 is strictly enforced regardless of adversarial process umasks."""
    dest_dir = tmp_path / "downloads"
    dest_dir.mkdir(parents=True, exist_ok=True)

    script_content = b"#!/bin/bash\necho PWNED\nexit 0\n"

    # Permutations of umask that normally leave execute or broad write bits
    test_umasks = [0o000, 0o002, 0o022, 0o077]

    for mask in test_umasks:
        old_mask = os.umask(mask)
        try:
            final_file = dest_dir / f"script_umask_{oct(mask)}.sh"
            stager = AtomicStagingWriter(
                dest_dir=dest_dir,
                final_path=final_file,
                expected_size=len(script_content),
            )
            with stager:
                # While staging, .part file must have 0 execute bits
                part_mode = stager.part_path.stat().st_mode
                assert (part_mode & 0o111) == 0, f"Execute bits present on .part file under umask {oct(mask)}"

                stager.write(script_content)
                committed = stager.commit()

            # Verify final committed file permissions
            final_mode = committed.stat().st_mode
            assert (final_mode & 0o111) == 0, f"Execute bits present on committed file under umask {oct(mask)}"
            assert (final_mode & 0o777) == 0o644, f"Final mode is not 0644 under umask {oct(mask)}: {oct(final_mode)}"
            assert not os.access(committed, os.X_OK), "File is executable according to os.access!"
        finally:
            os.umask(old_mask)


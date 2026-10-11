"""tests.unit.test_share_security
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for PenguLink Share filesystem security, 7-tier resolution,
path traversal sanitization, boundary verification, collision deduplication,
and atomic staging.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import time
import unicodedata
from unittest.mock import patch
import pytest

from pengulink.core.config import Config, StorageConfig
from pengulink.core.exceptions import (
    DestinationDirectoryError,
    IncompleteTransferError,
    PathTraversalError,
    PayloadSizeExceededError,
)
from pengulink.plugins.share.security import (
    AtomicFileStager,
    AtomicStagingWriter,
    cleanup_orphaned_part_files,
    parse_xdg_user_dirs_file,
    resolve_collision,
    resolve_download_dir,
    sanitize_filename,
    split_stem_and_suffix,
    strip_execution_bits,
    verify_boundary,
    validate_containment,
)


def test_resolve_download_dir_cli_override(tmp_path: Path) -> None:
    dest = tmp_path / "custom_downloads"
    resolved = resolve_download_dir(dest_override=dest)
    assert resolved == dest.resolve()
    assert resolved.exists()
    assert (resolved.stat().st_mode & 0o777) == 0o700


def test_resolve_download_dir_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_dest = tmp_path / "env_downloads"
    monkeypatch.setenv("PENGULINK_DOWNLOAD_DIR", str(env_dest))
    resolved = resolve_download_dir()
    assert resolved == env_dest.resolve()
    assert resolved.exists()


def test_resolve_download_dir_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PENGULINK_DOWNLOAD_DIR", raising=False)
    cfg_dest = tmp_path / "cfg_downloads"
    storage_cfg = StorageConfig(download_dir=cfg_dest)
    config = Config(storage=storage_cfg)

    resolved = resolve_download_dir(config=config)
    assert resolved == cfg_dest.resolve()
    assert resolved.exists()


def test_resolve_download_dir_xdg_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PENGULINK_DOWNLOAD_DIR", raising=False)
    xdg_dest = tmp_path / "xdg_downloads"
    monkeypatch.setenv("XDG_DOWNLOAD_DIR", str(xdg_dest))

    resolved = resolve_download_dir()
    assert resolved == xdg_dest.resolve()
    assert resolved.exists()


def test_resolve_download_dir_xdg_user_dirs_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PENGULINK_DOWNLOAD_DIR", raising=False)
    monkeypatch.delenv("XDG_DOWNLOAD_DIR", raising=False)
    mock_dirs = tmp_path / "user-dirs.dirs"
    target_dl = tmp_path / "home_downloads"
    mock_dirs.write_text(f'XDG_DOWNLOAD_DIR="{target_dl}"\n', encoding="utf-8")

    with patch("pengulink.plugins.share.security.parse_xdg_user_dirs_file", return_value=target_dl):
        resolved = resolve_download_dir()
        assert resolved == target_dl.resolve()


def test_resolve_download_dir_posix_permissions(tmp_path: Path) -> None:
    dest = tmp_path / "secure_dir"
    resolved = resolve_download_dir(dest_override=dest)
    mode = resolved.stat().st_mode & 0o777
    assert mode == 0o700


def test_resolve_download_dir_unwritable_error(tmp_path: Path) -> None:
    read_only_dir = tmp_path / "readonly"
    read_only_dir.mkdir(mode=0o500)
    try:
        os.chmod(read_only_dir, 0o500)
    except OSError:
        pass

    with patch("os.access", return_value=False):
        with pytest.raises(DestinationDirectoryError):
            resolve_download_dir(dest_override=read_only_dir)


def test_sanitize_posix_traversal() -> None:
    raw = "../../../../tmp/evil.sh"
    cleaned = sanitize_filename(raw)
    assert cleaned == "evil.sh"
    assert "/" not in cleaned
    assert ".." not in cleaned


def test_sanitize_windows_backslash_traversal() -> None:
    raw = r"..\..\windows\system32\cmd.exe"
    cleaned = sanitize_filename(raw)
    assert cleaned == "cmd.exe"
    assert "\\" not in cleaned


def test_sanitize_null_byte_rejection() -> None:
    raw = "doc.pdf\x00.sh"
    with pytest.raises(PathTraversalError):
        sanitize_filename(raw)


def test_sanitize_control_chars() -> None:
    raw = "file\x01\x1b[31m\x7fname.txt"
    cleaned = sanitize_filename(raw)
    assert cleaned == "file[31mname.txt"


def test_sanitize_hazardous_chars() -> None:
    raw = r'bad:file*name?with<hazard>chars|test.txt'
    cleaned = sanitize_filename(raw)
    assert cleaned == "bad_file_name_with_hazard_chars_test.txt"


def test_sanitize_unicode_nfc() -> None:
    decomposed = "e\u0301cole.pdf"
    cleaned = sanitize_filename(decomposed)
    assert cleaned == "école.pdf"
    assert unicodedata.is_normalized("NFC", cleaned)


def test_sanitize_leading_dot_defense() -> None:
    assert sanitize_filename(".hidden") == "file_.hidden"
    assert sanitize_filename(".bashrc") == "file_.bashrc"


def test_sanitize_internal_spaces_preserved() -> None:
    name = "Meeting Notes (2026).pdf"
    assert sanitize_filename(name) == "Meeting Notes (2026).pdf"


def test_sanitize_length_clamping() -> None:
    long_stem = "a" * 300
    cleaned = sanitize_filename(f"{long_stem}.tar.gz")
    assert len(cleaned.encode("utf-8")) <= 240
    assert cleaned.endswith(".tar.gz")


def test_sanitize_empty_or_dots_only() -> None:
    with pytest.raises(PathTraversalError):
        sanitize_filename("..")
    with pytest.raises(PathTraversalError):
        sanitize_filename(".")
    with pytest.raises(PathTraversalError):
        sanitize_filename("   ")


def test_verify_boundary_valid(tmp_path: Path) -> None:
    dest = tmp_path / "downloads"
    dest.mkdir()
    target = dest / "file.txt"
    target.touch()
    verified = verify_boundary(dest, target)
    assert verified == target.resolve()
    assert validate_containment(dest, target) == target.resolve()


def test_verify_boundary_escape_raises(tmp_path: Path) -> None:
    dest = tmp_path / "downloads"
    dest.mkdir()
    target = dest / "../outside.txt"
    with pytest.raises(PathTraversalError):
        verify_boundary(dest, target)


def test_verify_boundary_root_collision_raises(tmp_path: Path) -> None:
    dest = tmp_path / "downloads"
    dest.mkdir()
    with pytest.raises(PathTraversalError):
        verify_boundary(dest, dest)


def test_verify_boundary_symlink_rejection(tmp_path: Path) -> None:
    dest = tmp_path / "downloads"
    dest.mkdir()
    real_file = tmp_path / "real.txt"
    real_file.touch()
    link = dest / "link.txt"
    try:
        link.symlink_to(real_file)
    except OSError:
        pytest.skip("Symlinks not supported on filesystem")

    with pytest.raises(PathTraversalError):
        verify_boundary(dest, link)


def test_resolve_collision_no_collision(tmp_path: Path) -> None:
    candidate = resolve_collision(tmp_path, "report.txt")
    assert candidate == tmp_path.resolve() / "report.txt"


def test_resolve_collision_sequential(tmp_path: Path) -> None:
    (tmp_path / "report.txt").touch()
    cand1 = resolve_collision(tmp_path, "report.txt")
    assert cand1 == tmp_path.resolve() / "report (1).txt"

    cand1.touch()
    cand2 = resolve_collision(tmp_path, "report.txt")
    assert cand2 == tmp_path.resolve() / "report (2).txt"


def test_resolve_collision_compound_extension(tmp_path: Path) -> None:
    (tmp_path / "archive.tar.gz").touch()
    cand = resolve_collision(tmp_path, "archive.tar.gz")
    assert cand == tmp_path.resolve() / "archive (1).tar.gz"


def test_atomic_stager_success_lifecycle(tmp_path: Path) -> None:
    with AtomicFileStager(tmp_path, "success.txt", expected_size=11) as stager:
        assert stager.part_path.exists()
        assert stager.part_path.name.startswith(".success.txt.")
        stager.write(b"hello world")
        final = stager.commit()

    assert final == tmp_path.resolve() / "success.txt"
    assert final.exists()
    assert final.read_bytes() == b"hello world"
    assert not stager.part_path.exists()
    mode = final.stat().st_mode & 0o777
    assert mode == 0o644


def test_atomic_stager_incomplete_cleanup(tmp_path: Path) -> None:
    stager = AtomicFileStager(tmp_path, "incomplete.txt", expected_size=100)
    with pytest.raises(IncompleteTransferError):
        with stager:
            stager.write(b"short payload")
            stager.commit()

    assert not (tmp_path / "incomplete.txt").exists()
    assert not stager.part_path.exists()


def test_atomic_stager_non_executable_mode(tmp_path: Path) -> None:
    with AtomicFileStager(tmp_path, "install.sh") as stager:
        stager.write(b"#!/bin/bash\necho hello\n")
        final = stager.commit()

    st_mode = final.stat().st_mode
    assert (st_mode & 0o111) == 0  # No execute bits
    assert (st_mode & 0o777) == 0o644


def test_cleanup_orphaned_part_files(tmp_path: Path) -> None:
    stale_part = tmp_path / ".old_file.abcd1234.part"
    stale_part.touch()
    # Age it 2 hours
    old_time = time.time() - 7200
    os.utime(stale_part, (old_time, old_time))

    fresh_part = tmp_path / ".new_file.1234abcd.part"
    fresh_part.touch()

    cleaned = cleanup_orphaned_part_files(tmp_path, max_age_seconds=3600.0)
    assert cleaned == 1
    assert not stale_part.exists()
    assert fresh_part.exists()

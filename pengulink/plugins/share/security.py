"""pengulink.plugins.share.security
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Filesystem security, destination resolution, path traversal defense,
deterministic collision resolution, and atomic staging for PenguLink Share.
"""

from __future__ import annotations

import io
import logging
import os
import posixpath
import re
import stat
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, IO, Optional, Tuple, Union

from pengulink.core.exceptions import (
    DestinationDirectoryError,
    IncompleteTransferError,
    PathTraversalError,
    PayloadSizeExceededError,
)

logger = logging.getLogger("pengulink.plugins.share.security")

# Constants
MAX_FILENAME_BYTES = 240
MAX_COLLISION_ATTEMPTS = 10000
POSIX_FILE_MODE = 0o644
POSIX_DIR_MODE = 0o700

# Known compound compression extensions
COMPOUND_EXTENSIONS = (
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".tar.zst",
    ".tar.lz",
    ".tar.lzma",
)

# Regex patterns
HAZARDOUS_CHARS_REGEX = re.compile(r'[/\\:*?"<>|]')
CONTROL_CHARS_REGEX = re.compile(r"[\x00-\x1f\x7f-\x9f]")
XDG_USER_DIRS_REGEX = re.compile(r'^XDG_DOWNLOAD_DIR=["\']?([^"\']+)["\']?', re.MULTILINE)


def parse_xdg_user_dirs_file(dirs_path: Optional[Path] = None) -> Optional[Path]:
    """Parse ~/.config/user-dirs.dirs for XDG_DOWNLOAD_DIR."""
    path = dirs_path or (Path.home() / ".config" / "user-dirs.dirs")
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8")
        match = XDG_USER_DIRS_REGEX.search(content)
        if match:
            raw_val = match.group(1).strip()
            # Expand $HOME or ${HOME}
            expanded = raw_val.replace("$HOME", str(Path.home())).replace("${HOME}", str(Path.home()))
            return Path(expanded)
    except Exception as exc:
        logger.debug("Failed to parse user-dirs.dirs from '%s': %s", path, exc)
    return None


def resolve_download_dir(
    dest_override: Optional[Union[str, Path]] = None,
    config: Optional[Any] = None,
) -> Path:
    """Resolve download destination directory using the strict 7-tier resolution hierarchy:

    1. Explicit call/CLI override (dest_override)
    2. Environment variable PENGULINK_DOWNLOAD_DIR
    3. Configuration storage.download_dir
    4. XDG_DOWNLOAD_DIR (env var, then ~/.config/user-dirs.dirs)
    5. Standard ~/Downloads
    6. App fallback ~/.local/share/pengulink/downloads
    7. Home directory ~/ fallback

    Enforces POSIX 0700 creation mode and verifies write permissions.
    """
    candidate: Optional[Path] = None

    # Tier 1: Explicit override
    if dest_override:
        candidate = Path(dest_override).expanduser()

    # Tier 2: Environment variable PENGULINK_DOWNLOAD_DIR
    elif os.environ.get("PENGULINK_DOWNLOAD_DIR"):
        candidate = Path(os.environ["PENGULINK_DOWNLOAD_DIR"]).expanduser()

    # Tier 3: Config storage.download_dir
    elif config is not None:
        storage_cfg = getattr(config, "storage", config)
        cfg_dir = getattr(storage_cfg, "download_dir", None)
        if cfg_dir:
            candidate = Path(cfg_dir).expanduser()

    # Tier 4: XDG_DOWNLOAD_DIR
    if candidate is None:
        xdg_env = os.environ.get("XDG_DOWNLOAD_DIR")
        if xdg_env:
            candidate = Path(xdg_env).expanduser()
        else:
            xdg_parsed = parse_xdg_user_dirs_file()
            if xdg_parsed:
                candidate = xdg_parsed

    # Tier 5: ~/Downloads
    if candidate is None:
        downloads_candidate = Path.home() / "Downloads"
        candidate = downloads_candidate

    # Ensure candidate or fallbacks can be created and are writable
    candidates_to_try = [candidate]

    # Tier 6 fallback
    if config is not None and hasattr(getattr(config, "storage", config), "data_dir"):
        data_dir = getattr(getattr(config, "storage", config), "data_dir")
        candidates_to_try.append(Path(data_dir) / "downloads")
    else:
        candidates_to_try.append(Path.home() / ".local" / "share" / "pengulink" / "downloads")

    # Tier 7 fallback
    candidates_to_try.append(Path.home())

    last_error: Optional[Exception] = None
    for cand in candidates_to_try:
        if cand is None:
            continue
        try:
            resolved = cand.resolve()
            resolved.mkdir(parents=True, exist_ok=True, mode=POSIX_DIR_MODE)
            try:
                os.chmod(resolved, POSIX_DIR_MODE)
            except OSError:
                pass
            if os.access(resolved, os.W_OK):
                return resolved
        except Exception as exc:
            last_error = exc
            logger.debug("Candidate directory '%s' cannot be accessed: %s", cand, exc)

    raise DestinationDirectoryError(
        f"Failed to resolve or create a writable download directory: {last_error}",
        context={"candidates": [str(c) for c in candidates_to_try if c is not None]},
    )


def split_stem_and_suffix(filename: str) -> Tuple[str, str]:
    """Split a filename into stem and suffix, preserving known compound extensions."""
    lower_name = filename.lower()
    for compound in COMPOUND_EXTENSIONS:
        if lower_name.endswith(compound):
            stem = filename[: -len(compound)]
            suffix = filename[-len(compound) :]
            return stem, suffix

    p = Path(filename)
    return p.stem, p.suffix


def sanitize_filename(raw_filename: str, max_bytes: int = MAX_FILENAME_BYTES) -> str:
    """Sanitize raw incoming filename against path traversal and hazardous input.

    Rules:
    - Rejects null bytes with PathTraversalError.
    - Normalizes '\\' to '/' and extracts pure basename.
    - Rejects empty or traversal components with PathTraversalError.
    - Strips control characters.
    - Applies NFC Unicode normalization.
    - Replaces hazardous characters with '_'.
    - Strips leading/trailing whitespace and dots.
    - Defends against hidden files (prepends 'file_' if name starts with '.').
    - Clamps UTF-8 byte length to <= max_bytes while preserving extension.
    """
    if not isinstance(raw_filename, str) or "\x00" in raw_filename:
        raise PathTraversalError(
            "Null byte or invalid type detected in filename",
            context={"raw_filename": raw_filename},
        )

    # 1. Normalize backslashes to forward slashes (POSIX separator defense)
    normalized = raw_filename.replace("\\", "/")
    # Strip trailing slashes before extracting basename
    normalized = normalized.rstrip("/")

    # 2. Extract basename
    basename = posixpath.basename(normalized)

    # 3. Traversal check on basename
    if basename in ("", ".", ".."):
        raise PathTraversalError(
            f"Invalid traversal filename: {raw_filename!r}",
            context={"raw_filename": raw_filename},
        )

    # 4. Strip control characters
    cleaned = CONTROL_CHARS_REGEX.sub("", basename)

    # 5. Unicode normalization (NFC)
    cleaned = unicodedata.normalize("NFC", cleaned)

    # 6. Replace hazardous characters with underscore
    cleaned = HAZARDOUS_CHARS_REGEX.sub("_", cleaned)

    # 7. Strip whitespace and trailing dots
    cleaned = cleaned.strip()
    cleaned = cleaned.rstrip(".")

    # 8. Leading dot defense
    if cleaned.startswith("."):
        cleaned = f"file_{cleaned}"

    if not cleaned:
        raise PathTraversalError(
            f"Filename became empty after sanitization: {raw_filename!r}",
            context={"raw_filename": raw_filename},
        )

    # 9. Clamping byte length to <= max_bytes UTF-8 while preserving suffix
    stem, suffix = split_stem_and_suffix(cleaned)
    suffix_bytes = suffix.encode("utf-8")
    if len(suffix_bytes) > max_bytes - 1:
        # Suffix itself is abnormally long; clamp whole string directly
        cleaned_bytes = cleaned.encode("utf-8")[:max_bytes]
        cleaned = cleaned_bytes.decode("utf-8", errors="ignore")
    else:
        max_stem_bytes = max_bytes - len(suffix_bytes)
        stem_bytes = stem.encode("utf-8")
        if len(stem_bytes) > max_stem_bytes:
            truncated_stem = stem_bytes[:max_stem_bytes].decode("utf-8", errors="ignore")
            cleaned = f"{truncated_stem}{suffix}"

    return cleaned


def verify_boundary(dest_dir: Path, target_path: Path) -> Path:
    """Verify that target_path resolves strictly within dest_dir.

    Raises PathTraversalError if the path escapes or attempts symlink breakout.
    """
    resolved_dest = dest_dir.resolve()
    resolved_target = target_path.resolve()

    if target_path.is_symlink():
        raise PathTraversalError(
            f"Target path is an existing symlink: {target_path}",
            context={"target_path": str(target_path)},
        )

    if resolved_target == resolved_dest:
        raise PathTraversalError(
            f"Target path cannot be the destination directory itself: {target_path}",
            context={"target_path": str(target_path)},
        )

    if not resolved_target.is_relative_to(resolved_dest):
        raise PathTraversalError(
            f"Target path '{target_path}' escapes destination directory '{dest_dir}'",
            context={"dest_dir": str(dest_dir), "target_path": str(target_path)},
        )

    return resolved_target


# Canonical alias for explorer 3 interface
validate_containment = verify_boundary


def resolve_collision(dest_dir: Path, filename: str) -> Path:
    """Deterministically resolve filename collision using 'stem (counter).suffix' up to 10,000."""
    dest_dir = dest_dir.resolve()
    stem, suffix = split_stem_and_suffix(filename)

    initial_candidate = dest_dir / f"{stem}{suffix}"
    if not initial_candidate.exists() and not initial_candidate.is_symlink():
        return initial_candidate

    for counter in range(1, MAX_COLLISION_ATTEMPTS + 1):
        candidate = dest_dir / f"{stem} ({counter}){suffix}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate

    # Fallback if 10,000 collisions exist: append 8-char hex UUID
    random_suffix = uuid.uuid4().hex[:8]
    return dest_dir / f"{stem}_{random_suffix}{suffix}"


def strip_execution_bits(path: Path) -> None:
    """Enforce POSIX 0644 non-executable permissions."""
    try:
        current_mode = os.stat(path).st_mode
        # Clear execute bits and non-owner write bits: exact 0o644 mode
        new_mode = (current_mode & ~0o777) | POSIX_FILE_MODE
        os.chmod(path, new_mode)
    except OSError:
        pass


def cleanup_orphaned_part_files(dest_dir: Path, max_age_seconds: float = 3600.0) -> int:
    """Scan dest_dir for lingering .*.part files older than max_age_seconds and unlink them."""
    count = 0
    now = time.time()
    try:
        resolved_dest = dest_dir.resolve()
        for item in resolved_dest.iterdir():
            if item.name.startswith(".") and item.name.endswith(".part"):
                try:
                    stat_info = item.stat()
                    if now - stat_info.st_mtime > max_age_seconds:
                        item.unlink(missing_ok=True)
                        count += 1
                except OSError:
                    pass
    except OSError:
        pass
    return count


class AtomicFileStager:
    """In-directory atomic staging context manager.

    Creates temporary .filename.<uuid>.part inside destination directory,
    enforces POSIX 0644 non-executable permissions, provides atomic os.replace
    on completion, and guarantees unlinking on error or cancellation.
    """

    def __init__(
        self,
        dest_dir: Path,
        final_filename: Optional[str] = None,
        final_path: Optional[Path] = None,
        expected_size: Optional[int] = None,
        last_modified: Optional[int] = None,
    ) -> None:
        self.dest_dir = dest_dir.resolve()
        if final_path is not None:
            self.final_path = final_path.resolve()
            self.final_filename = self.final_path.name
        elif final_filename is not None:
            self.final_filename = final_filename
            self.final_path = (self.dest_dir / self.final_filename).resolve()
        else:
            raise ValueError("Either final_filename or final_path must be provided")

        self.expected_size = expected_size
        self.last_modified = last_modified
        token = uuid.uuid4().hex[:8]
        self.part_filename = f".{self.final_filename}.{token}.part"
        self.part_path = self.dest_dir / self.part_filename

        self._committed = False
        self._file_obj: Optional[IO[bytes]] = None
        self.bytes_written: int = 0

    def __enter__(self) -> AtomicFileStager:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
        if not self._committed:
            self.unlink_part()

    def open(self) -> None:
        verify_boundary(self.dest_dir, self.part_path)
        self._file_obj = open(self.part_path, "wb")
        strip_execution_bits(self.part_path)

    def write(self, data: bytes) -> int:
        if self._file_obj is None:
            raise RuntimeError("AtomicFileStager is not open")
        n = self._file_obj.write(data)
        self.bytes_written += n
        return n

    # Async-compatible methods for stream writers
    async def write_chunk(self, chunk: bytes) -> int:
        return self.write(chunk)

    def close(self) -> None:
        if self._file_obj is not None:
            try:
                self._file_obj.flush()
                self._file_obj.close()
            except OSError:
                pass
            self._file_obj = None

    def unlink_part(self) -> None:
        if self.part_path.exists():
            try:
                self.part_path.unlink()
            except OSError:
                pass

    def abort(self) -> None:
        self.close()
        self.unlink_part()

    def commit(self) -> Path:
        """Atomically rename .part file to final destination and ensure non-executable."""
        self.close()
        verify_boundary(self.dest_dir, self.final_path)

        if self.expected_size is not None and self.bytes_written != self.expected_size:
            self.unlink_part()
            if self.bytes_written < self.expected_size:
                raise IncompleteTransferError(
                    f"Incomplete transfer: wrote {self.bytes_written} bytes, expected {self.expected_size}"
                )
            else:
                raise PayloadSizeExceededError(
                    f"Payload size exceeded: wrote {self.bytes_written} bytes, expected {self.expected_size}"
                )

        os.replace(self.part_path, self.final_path)
        strip_execution_bits(self.final_path)

        if self.last_modified is not None:
            try:
                # last_modified in ms or s
                mtime_sec = self.last_modified / 1000.0 if self.last_modified > 1e11 else float(self.last_modified)
                os.utime(self.final_path, (mtime_sec, mtime_sec))
            except OSError:
                pass

        self._committed = True
        return self.final_path

    # Aliases
    finalize = commit


# Canonical alias for explorer 3 & 2
AtomicStagingWriter = AtomicFileStager
StagingFileWriter = AtomicFileStager


__all__ = [
    "MAX_FILENAME_BYTES",
    "MAX_COLLISION_ATTEMPTS",
    "POSIX_FILE_MODE",
    "POSIX_DIR_MODE",
    "COMPOUND_EXTENSIONS",
    "parse_xdg_user_dirs_file",
    "resolve_download_dir",
    "split_stem_and_suffix",
    "sanitize_filename",
    "verify_boundary",
    "validate_containment",
    "resolve_collision",
    "strip_execution_bits",
    "cleanup_orphaned_part_files",
    "AtomicFileStager",
    "AtomicStagingWriter",
    "StagingFileWriter",
]

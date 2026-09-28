"""Atomic single-file publication with explicit durability acknowledgements."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class StorageError(RuntimeError):
    """Sanitized public storage error; original details remain in __cause__."""


class CorruptStoreError(StorageError):
    def __init__(self):
        super().__init__("Persisted data is invalid; explicit recovery is required")


class PersistenceError(StorageError):
    def __init__(self, *, outcome_uncertain: bool = False):
        self.outcome_uncertain = outcome_uncertain
        super().__init__(
            "Filesystem change completed but durability is uncertain"
            if outcome_uncertain
            else "Persistence failed before the filesystem change"
        )


def _sync_directory(path: Path) -> None:
    # POSIX directory fsync makes the rename durable. Unsupported/failing fsync
    # is an uncertain acknowledgement, never silently treated as success.
    if os.name != "posix":
        raise NotImplementedError("Directory durability sync requires POSIX")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Replace a file, fsync its contents and parent, or raise PersistenceError.

    Pre-replacement failures preserve existing bytes. After replacement a sync
    failure means the new bytes are visible but crash durability is uncertain.
    This alone does not serialize read/modify/write; callers must hold a lock.
    """
    tmp_fd = None
    tmp_path = None
    replaced = False
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            text=True,
        )
        stream = os.fdopen(tmp_fd, "w", encoding=encoding)
        tmp_fd = None  # ownership transferred only after successful fdopen
        with stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
        replaced = True
        _sync_directory(path.parent)
    except Exception as exc:
        raise PersistenceError(outcome_uncertain=replaced) from exc
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path is not None and not replaced:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
    encoding: str = "utf-8",
) -> None:
    try:
        text = json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent)
    except Exception as exc:
        raise PersistenceError() from exc
    atomic_write_text(path, text, encoding=encoding)

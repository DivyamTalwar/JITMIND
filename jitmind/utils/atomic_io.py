# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Optional
from pathlib import Path
import json
import os
import tempfile


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """
    Atomically write text to `path` by writing to a temp file and renaming.

    This prevents partially-written/corrupted JSON if the process crashes or
    multiple writers interleave writes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd: Optional[int] = None
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            text=True,
        )
        with os.fdopen(tmp_fd, "w", encoding=encoding) as f:
            tmp_fd = None  # fd is owned by the file object now
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except Exception:
                pass
        if tmp_path is not None:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
    encoding: str = "utf-8",
) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent),
        encoding=encoding,
    )


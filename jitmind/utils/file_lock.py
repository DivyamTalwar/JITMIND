# -*- coding: utf-8 -*-
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import time


@contextmanager
def file_lock(
    lock_path: Path,
    *,
    timeout_s: float = 10.0,
    poll_s: float = 0.05,
    stale_s: float = 300.0,
):
    """
    Cross-process lock using an atomic lockfile create.

    This is intentionally simple:
    - Acquire: create `lock_path` with O_EXCL
    - Release: delete `lock_path`
    - Stale lock recovery: if lock is older than `stale_s`, remove it

    This is not a perfect distributed lock, but it's a pragmatic guardrail for
    file-backed stores in local/dev setups.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout_s

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"pid={os.getpid()} ts={time.time()}\n".encode("utf-8"))
            finally:
                os.close(fd)
            break
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_s:
                    try:
                        lock_path.unlink()
                    except Exception:
                        pass
            except Exception:
                pass

            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(poll_s)

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

"""Cooperative local-filesystem locking; lock files must never be removed."""

from __future__ import annotations

import errno
import math
import os
import threading
import time
import weakref
from contextlib import contextmanager
from pathlib import Path

from .atomic_io import StorageError


class _LocalLock:
    def __init__(self):
        self.mutex = threading.RLock()
        self.depth = 0
        self.fd = None


_registry = weakref.WeakValueDictionary()
_registry_guard = threading.Lock()
# Separate from per-path mutexes: fork preparation never waits for a lock body
# or kernel contention. Only descriptor open/register and close/remove are
# indivisible with respect to fork. No other locks are acquired under this guard.
_descriptor_guard = threading.Lock()
_descriptors = {}


def _before_fork():
    _descriptor_guard.acquire()


def _after_fork_parent():
    _descriptor_guard.release()


def _after_fork():
    # Close inherited copies without LOCK_UN (which would unlock the parent).
    global _registry, _registry_guard, _descriptor_guard, _descriptors
    for fd, entry in _descriptors.items():
        os.close(fd)
        entry.fd = None
    _descriptors = {}
    _registry = weakref.WeakValueDictionary()
    _registry_guard = threading.Lock()
    _descriptor_guard = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_fork_parent,
        after_in_child=_after_fork,
    )


@contextmanager
def file_lock(
    lock_path: Path,
    *,
    timeout_s: float = 10.0,
    poll_s: float = 0.05,
    stale_s: float = 300.0,
):
    """Hold a POSIX flock plus a per-path reentrant thread lock.

    Uses a persistent inode, never mtime-based stealing or unlink at release.
    Same-thread nesting on the canonical path is supported. Other threads and
    independently opened process handles contend with a finite monotonic wait.
    ``stale_s`` is retained, ignored, and diagnostic-only for compatibility.
    Windows is explicitly unsupported pending validated msvcrt implementation.
    All participants must cooperate; network filesystems are not qualified.
    """
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s < 0:
        raise ValueError("timeout_s must be finite and non-negative")
    if isinstance(poll_s, bool) or not math.isfinite(poll_s) or poll_s <= 0:
        raise ValueError("poll_s must be finite and positive")
    if os.name != "posix":
        raise NotImplementedError("Advisory file locking requires POSIX")
    import fcntl

    deadline = time.monotonic() + timeout_s
    try:
        path = Path(lock_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError) as exc:
        raise StorageError("Unable to prepare persistence lock") from exc
    with _registry_guard:
        entry = _registry.get(path)
        if entry is None:
            entry = _LocalLock()
            _registry[path] = entry
    # Poll rather than RLock.acquire(timeout=...) to support any finite value
    # without overflowing the platform's native timeout representation.
    while not entry.mutex.acquire(blocking=False):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for persistence lock")
        time.sleep(min(poll_s, remaining, 1.0))
    try:
        if entry.depth:
            entry.depth += 1
            try:
                yield
            finally:
                entry.depth -= 1
            return
        try:
            with _descriptor_guard:
                entry.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                _descriptors[entry.fd] = entry
            while True:
                try:
                    fcntl.flock(entry.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN):
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "Timed out waiting for persistence lock"
                        ) from None
                    time.sleep(min(poll_s, remaining, 1.0))
        except TimeoutError:
            raise
        except OSError as exc:
            raise StorageError("Unable to acquire persistence lock") from exc
        entry.depth = 1
        try:
            yield
        finally:
            entry.depth = 0
            # Closing our last descriptor releases the kernel lock, including
            # when the body raises. Never unlink this inode.
    finally:
        try:
            if not entry.depth and entry.fd is not None:
                with _descriptor_guard:
                    fd = entry.fd
                    try:
                        os.close(fd)
                    finally:
                        # Do not retry close after an error: POSIX may already
                        # have released this number for reuse by another thread.
                        entry.fd = None
                        _descriptors.pop(fd, None)
        finally:
            entry.mutex.release()

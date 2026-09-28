"""Bounded descriptor-relative capture. No repository hooks or target code run."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from threading import Event, RLock
from uuid import UUID

from jitmind.scope import ScopeDenied

from .models import PARSER, safe_path, under
from .process import Unavailable, checkpoint

MAX_FILES = 128
MAX_FILE_BYTES = 131072
MAX_TOTAL_BYTES = 2097152
MAX_ENTRIES = 2048
MAX_DEPTH = 32
EXCLUDE = frozenset(
    {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv"}
)
EXTENSIONS = (".py", ".pyi")
COVERAGE = {
    "extensions": EXTENSIONS,
    "exclude_directories": sorted(EXCLUDE),
    "max_files": MAX_FILES,
    "max_file_bytes": MAX_FILE_BYTES,
    "max_total_bytes": MAX_TOTAL_BYTES,
    "max_entries": MAX_ENTRIES,
    "max_depth": MAX_DEPTH,
    "encoding": "strict-utf8",
    "symlinks": "never",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(data) -> bytes:
    return json.dumps(
        data, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode()


COVERAGE_DIGEST = digest(canonical({"parser": PARSER, **COVERAGE}))


@dataclass(frozen=True)
class Repository:
    repo_id: str
    namespace_id: str
    root: str
    identity: tuple[int, int]


class RepoRegistry:
    """Trusted-host administrative registry. UUID identities never derive from names."""

    def __init__(self):
        self._repos: dict[str, Repository] = {}
        self._lock = RLock()

    def register(self, repo_id: str, root: str, namespace_id: str) -> None:
        if str(UUID(repo_id)) != repo_id or not namespace_id or "\x00" in namespace_id:
            raise ValueError("Invalid registry identifier")
        path = Path(root)
        if not path.is_absolute() or path.is_symlink():
            raise ValueError("Root must be an absolute real directory")
        real = path.resolve(strict=True)
        if str(real) != str(path):
            raise ValueError("Root must be canonical; symlinks are not allowed")
        fd = os.open(real, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            st = os.fstat(fd)
            entry = Repository(repo_id, namespace_id, str(real), (st.st_dev, st.st_ino))
        finally:
            os.close(fd)
        with self._lock:
            if repo_id in self._repos:
                raise ValueError("Repository identity already registered")
            self._repos[repo_id] = entry

    def get(self, repo_id: str, namespace_id: str) -> Repository:
        with self._lock:
            repo = self._repos.get(repo_id)
        if repo is None or repo.namespace_id != namespace_id:
            raise ScopeDenied()
        return repo


@dataclass(frozen=True)
class SourceFile:
    path: str
    digest: str | None
    source: str | None
    status: str


@dataclass(frozen=True)
class Snapshot:
    files: tuple[SourceFile, ...]
    manifest_digest: str
    snapshot_id: str
    scan_complete: bool
    reasons: tuple[str, ...]

    def scoped(self, prefix):
        return tuple(f for f in self.files if under(f.path, prefix))


def stamp(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _open_root(repo):
    # Resolve components with O_NOFOLLOW as well as the final component. A renamed
    # ancestor replaced by a symlink must not redirect even a metadata read.
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in Path(repo.root).parts[1:]:
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
        st = os.fstat(fd)
        if (st.st_dev, st.st_ino) != repo.identity:
            raise Unavailable("source_changed")
        return fd
    except BaseException:
        os.close(fd)
        raise


def capture(repo: Repository, deadline: float, cancel: Event | None = None) -> Snapshot:
    checkpoint(deadline, cancel)
    files = []
    reasons = set()
    total = 0
    entries = 0
    complete = True
    try:
        root_fd = _open_root(repo)
    except OSError:
        raise Unavailable("source_unknown") from None
    root_before = os.fstat(root_fd)

    def walk(fd, prefix, depth):
        nonlocal total, entries, complete
        checkpoint(deadline, cancel)
        before = os.fstat(fd)
        if entries >= MAX_ENTRIES:
            complete = False
            reasons.add("entry_budget")
            return
        if depth > MAX_DEPTH:
            complete = False
            reasons.add("directory_budget")
            return
        try:
            # os.scandir(fd) never follows a path to enumerate the directory.
            with os.scandir(fd) as iterator:
                names = []
                for item in iterator:
                    entries += 1
                    checkpoint(deadline, cancel)
                    if entries > MAX_ENTRIES:
                        complete = False
                        reasons.add("entry_budget")
                        break
                    names.append(item.name)
            for name in sorted(names):
                checkpoint(deadline, cancel)
                path = f"{prefix}/{name}" if prefix else name
                try:
                    safe_path(path)
                    st = os.stat(name, dir_fd=fd, follow_symlinks=False)
                except (OSError, ValueError, UnicodeError):
                    complete = False
                    reasons.add("source_unknown")
                    continue
                if stat.S_ISDIR(st.st_mode):
                    if name in EXCLUDE:
                        continue
                    try:
                        child = os.open(
                            name,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=fd,
                        )
                    except OSError:
                        complete = False
                        reasons.add("source_unknown")
                        continue
                    try:
                        if stamp(st) != stamp(os.fstat(child)):
                            raise Unavailable("source_changed")
                        walk(child, path, depth + 1)
                        after = os.stat(name, dir_fd=fd, follow_symlinks=False)
                        if stamp(after) != stamp(os.fstat(child)):
                            raise Unavailable("source_changed")
                    finally:
                        os.close(child)
                    continue
                if len(files) >= MAX_FILES:
                    complete = False
                    reasons.add("file_budget")
                    return
                if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                    files.append(SourceFile(path, None, None, "excluded_special"))
                    continue
                if not path.endswith(EXTENSIONS):
                    files.append(SourceFile(path, None, None, "unsupported"))
                    continue
                if st.st_size > MAX_FILE_BYTES or total + st.st_size > MAX_TOTAL_BYTES:
                    files.append(SourceFile(path, None, None, "input_budget"))
                    continue
                try:
                    source_fd = os.open(
                        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
                    )
                    try:
                        opened = os.fstat(source_fd)
                        if not stat.S_ISREG(opened.st_mode) or stamp(st) != stamp(
                            opened
                        ):
                            raise Unavailable("source_changed")
                        data = bytearray()
                        while len(data) <= MAX_FILE_BYTES:
                            checkpoint(deadline, cancel)
                            chunk = os.read(
                                source_fd, min(16384, MAX_FILE_BYTES + 1 - len(data))
                            )
                            if not chunk:
                                break
                            data.extend(chunk)
                        if stamp(opened) != stamp(os.fstat(source_fd)) or stamp(
                            opened
                        ) != stamp(os.stat(name, dir_fd=fd, follow_symlinks=False)):
                            raise Unavailable("source_changed")
                    finally:
                        os.close(source_fd)
                    if (
                        len(data) > MAX_FILE_BYTES
                        or total + len(data) > MAX_TOTAL_BYTES
                    ):
                        files.append(SourceFile(path, None, None, "input_budget"))
                        continue
                    total += len(data)
                    hashed = digest(data)
                    try:
                        source = bytes(data).decode("utf8")
                        if "\x00" in source:
                            raise UnicodeError()
                    except UnicodeError:
                        files.append(SourceFile(path, hashed, None, "encoding_unknown"))
                        continue
                    files.append(SourceFile(path, hashed, source, "ok"))
                except OSError:
                    files.append(SourceFile(path, None, None, "unreadable"))
            if stamp(before) != stamp(os.fstat(fd)):
                raise Unavailable("source_changed")
        except OSError:
            complete = False
            reasons.add("source_unknown")

    try:
        walk(root_fd, "", 0)
        try:
            verify_fd = _open_root(repo)
        except OSError:
            raise Unavailable("source_changed") from None
        try:
            if stamp(root_before) != stamp(os.fstat(verify_fd)):
                raise Unavailable("source_changed")
        finally:
            os.close(verify_fd)
    finally:
        os.close(root_fd)
    manifest = digest(
        canonical(
            {
                "files": [(f.path, f.digest, f.status) for f in files],
                "scan_complete": complete,
                "reasons": sorted(reasons),
            }
        )
    )
    snapshot_id = digest(
        canonical([repo.repo_id, repo.namespace_id, manifest, COVERAGE_DIGEST])
    )
    return Snapshot(
        tuple(files), manifest, snapshot_id, complete, tuple(sorted(reasons))
    )

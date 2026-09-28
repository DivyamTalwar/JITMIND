"""Local work/lesson metadata only; primary facts remain in the durable fact store.

Native adaptation informed by koragraph practice modules, revision c9ce746.
See docs/open-work-preflight.md for source attribution and retained BSL notice.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class WorkError(Exception):
    def __init__(self, code: str = "invalid_request") -> None:
        self.code = code
        super().__init__(code)


class Conflict(WorkError):
    def __init__(self) -> None:
        super().__init__("revision_or_idempotency_conflict")


class Deferred(WorkError):
    def __init__(self) -> None:
        super().__init__("deadline_or_contention")


def identifier(value: str) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or len(value) > 512
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise WorkError()
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise WorkError() from None
    return value


def revision(value: int, *, zero: bool = False) -> int:
    if type(value) is not int or not (0 if zero else 1) <= value < 2**63 - 1:
        raise WorkError()
    return value


def finite(value: float, maximum: float) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 < value <= maximum
    ):
        raise WorkError()
    return value


def encoded(value) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )


def digest(value) -> str:
    return hashlib.sha256(encoded(value).encode()).hexdigest()


class Budget:
    def __init__(self, seconds: float = 0.15) -> None:
        self.end = time.monotonic() + finite(seconds, 30)

    def remaining(self) -> float:
        left = self.end - time.monotonic()
        if left <= 0:
            raise Deferred()
        return left


SCHEMA = (
    "CREATE TABLE work (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, namespace TEXT NOT NULL, repo TEXT NOT NULL, payload TEXT NOT NULL)",
    "CREATE INDEX work_scope ON work(namespace,repo,seq)",
    "CREATE TABLE history (work_id TEXT NOT NULL REFERENCES work(id), revision INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(work_id,revision))",
    "CREATE TRIGGER history_no_update BEFORE UPDATE ON history BEGIN SELECT RAISE(ABORT,'immutable_history'); END",
    "CREATE TRIGGER history_no_delete BEFORE DELETE ON history BEGIN SELECT RAISE(ABORT,'immutable_history'); END",
    "CREATE TABLE work_receipts (namespace TEXT NOT NULL, repo TEXT NOT NULL, actor TEXT NOT NULL, key TEXT NOT NULL, digest TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(namespace,repo,actor,key))",
    "CREATE TABLE lessons (namespace TEXT NOT NULL, repo TEXT NOT NULL, id TEXT NOT NULL, fact_id TEXT NOT NULL, version INTEGER NOT NULL, file TEXT NOT NULL, symbol TEXT NOT NULL, action TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(namespace,repo,id))",
    "CREATE INDEX lesson_target ON lessons(namespace,repo,file,symbol,action,id)",
    "CREATE INDEX lesson_fact ON lessons(namespace,fact_id)",
    "CREATE TABLE tombstones (namespace TEXT NOT NULL, fact_id TEXT NOT NULL, revision INTEGER NOT NULL, PRIMARY KEY(namespace,fact_id))",
    "CREATE TABLE sessions (id TEXT PRIMARY KEY, expires REAL NOT NULL)",
    "CREATE INDEX session_expiry ON sessions(expires)",
    "CREATE TABLE deliveries (session TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, code TEXT NOT NULL, lesson TEXT NOT NULL, version INTEGER NOT NULL, policy INTEGER NOT NULL, PRIMARY KEY(session,code,lesson,version,policy))",
    "CREATE TABLE delivery_receipts (session TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, key TEXT NOT NULL, digest TEXT NOT NULL, refs TEXT NOT NULL, PRIMARY KEY(session,key))",
)
APPLICATION_ID = 0x4A36574B


class WorkDatabase:
    """Host initializes explicitly, outside request hotpaths. DELETE/FULL only.

    No auto-adoption of other databases or future schemas. Connections never wait
    for SQLite locks: callers receive Deferred immediately and may retry later.
    This also keeps aggregate SQLite busy waits within any residual deadline.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).absolute()
        with self.connect(Budget(2), initialize=True) as db:
            objects = db.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            version = db.execute("PRAGMA user_version").fetchone()[0]
            app = db.execute("PRAGMA application_id").fetchone()[0]
            if not objects and version == app == 0:
                db.execute("PRAGMA journal_mode=DELETE")
                with self.transaction(db):
                    for statement in SCHEMA:
                        db.execute(statement)
                    db.execute(f"PRAGMA application_id={APPLICATION_ID}")
                    db.execute("PRAGMA user_version=1")
            elif (
                version != 1
                or app != APPLICATION_ID
                or {r[0] for r in objects} != set(SCHEMA)
            ):
                raise WorkError("schema_mismatch")
            if db.execute("PRAGMA journal_mode").fetchone()[0].lower() != "delete":
                raise WorkError("schema_mismatch")
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise WorkError("storage_unavailable")

    @contextmanager
    def connect(self, budget: Budget, *, initialize: bool = False):
        budget.remaining()
        db = None
        try:
            db = sqlite3.connect(
                self.path.as_uri() + ("?mode=rwc" if initialize else "?mode=rw"),
                uri=True,
                timeout=0,
                isolation_level=None,
            )
            db.row_factory = sqlite3.Row
            db.set_progress_handler(lambda: int(time.monotonic() >= budget.end), 100)
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA synchronous=FULL")
            if not initialize and (
                db.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
                or db.execute("PRAGMA user_version").fetchone()[0] != 1
                or db.execute("PRAGMA journal_mode").fetchone()[0].lower() != "delete"
            ):
                raise WorkError("schema_mismatch")
            budget.remaining()
            yield db
        except sqlite3.Error as exc:
            if (getattr(exc, "sqlite_errorcode", 0) & 255) in (5, 6, 9):
                raise Deferred() from None
            raise WorkError("storage_unavailable") from None
        finally:
            if db is not None:
                db.close()

    @staticmethod
    @contextmanager
    def transaction(db):
        db.execute("BEGIN IMMEDIATE")
        try:
            yield
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                # Cancellation must not interrupt rollback.
                db.set_progress_handler(None, 0)
                db.execute("ROLLBACK")
            raise

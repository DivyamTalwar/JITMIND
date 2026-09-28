"""Optional SQLite event ledger. No model calls or automatic source deletion."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any

from jitmind.scope import ScopeAuthority, ScopeContext

MAX_SEQUENCE = 2**63 - 2
MAX_RANGE = 100_000
MAX_ACQUIRED_BYTES = 4_000_000
MAX_PREFIX_EVENTS = 100_000
MAX_HISTORY_PAGE = 100
_DEADLINE: ContextVar[float | None] = ContextVar("checkpoint_deadline", default=None)
SOURCE_SCHEMA = "jitmind.checkpoints.events.v1"
TERMINALS = frozenset({"tool_result", "tool_cancel", "tool_timeout"})
KINDS = TERMINALS | {"message", "tool_start"}


def _row_bytes(columns: str) -> str:
    # SQLite computes lengths before Python ever receives the text/payload.
    return "+".join(
        f"COALESCE(length(CAST({column} AS BLOB)),0)"
        for column in columns.split(",")
    )


_EVENT_BYTES = _row_bytes(
    "namespace,stream,sequence,event_id,payload_json,payload_digest,observed_time,kind,tool_call_id"
)
_PREFIX_BYTES = _row_bytes("sequence,kind,tool_call_id")
_HISTORY_BYTES = _row_bytes(
    "namespace,stream,revision,idempotency_key,draft_json,draft_digest,created_at"
)


class CheckpointError(Exception):
    """Base class for sanitized checkpoint failures."""


class InvalidInput(CheckpointError):
    pass


class CapacityExceeded(InvalidInput):
    """A complete operation exceeds the documented acquisition/work budget."""


def _check_time() -> None:
    deadline = _DEADLINE.get()
    if deadline is not None and monotonic() >= deadline:
        raise CapacityExceeded("Checkpoint operation deadline exceeded")


class EventConflict(CheckpointError):
    pass


class UnsafeRange(CheckpointError):
    pass


class SourceChanged(CheckpointError):
    pass


class RevisionConflict(CheckpointError):
    pass


class IdempotencyConflict(CheckpointError):
    pass


class StorageError(CheckpointError):
    pass


class SchemaMismatch(StorageError):
    pass


class CommitUncertain(StorageError):
    """Commit acknowledgement failed; reopen and retry the exact draft/key."""


def _id(value: object) -> None:
    if (
        type(value) is not str
        or not 0 < len(value) <= 1024
        or not value.strip()
        or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value)
    ):
        raise InvalidInput("Invalid identifier")


def _integer(value: object) -> None:
    if type(value) is not int or not 0 <= value <= MAX_SEQUENCE:
        raise InvalidInput("Invalid bounded integer")


def _range(start: int, end: int) -> None:
    _integer(start)
    _integer(end)
    if start > end or end - start + 1 > MAX_RANGE:
        raise InvalidInput("Invalid event range")


def _json(value: Any) -> str:
    # Reject coercions such as integer dict keys, tuples, NaN, or arbitrary objects.
    remaining = MAX_ACQUIRED_BYTES

    def valid(item: Any, depth: int = 0) -> bool:
        nonlocal remaining
        _check_time()
        remaining -= 1
        if type(item) is str:
            remaining -= len(item)
        if remaining < 0:
            raise CapacityExceeded("JSON acquisition budget exceeded")
        if depth > 64:
            return False
        if type(item) is int and item.bit_length() > 4096:
            return False
        if item is None or type(item) in (bool, int, float, str):
            return True
        if type(item) is list:
            return all(valid(v, depth + 1) for v in item)
        if type(item) is dict:
            return all(
                type(k) is str and valid(k, depth + 1) and valid(v, depth + 1)
                for k, v in item.items()
            )
        return False

    try:
        if not valid(value):
            raise ValueError()
        encoder = json.JSONEncoder(
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        chunks = []
        size = 0
        for chunk in encoder.iterencode(value):
            _check_time()
            size += len(chunk)
            if size > MAX_ACQUIRED_BYTES:
                raise CapacityExceeded("JSON acquisition budget exceeded")
            chunks.append(chunk)
        return "".join(chunks)
    except (ValueError, TypeError, RecursionError):
        raise InvalidInput("Invalid JSON data") from None


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Event:
    namespace: str
    stream: str
    sequence: int
    event_id: str
    payload_json: str
    payload_digest: str
    observed_time: str
    kind: str
    tool_call_id: str | None

    @property
    def payload(self) -> Any:
        return json.loads(self.payload_json)


@dataclass(frozen=True)
class Snapshot:
    namespace: str
    stream: str
    start: int
    end: int
    source_schema: str
    source_digest: str
    expected_revision: int
    events: tuple[Event, ...]

    def draft(
        self,
        *,
        summary: str,
        idempotency_key: str,
        model_metadata: dict[str, Any] | None = None,
    ) -> Draft:
        """Call after generating summary outside all ledger transactions."""
        return Draft(
            self.namespace,
            self.stream,
            self.start,
            self.end,
            self.source_schema,
            self.source_digest,
            self.expected_revision,
            idempotency_key,
            summary,
            _json(model_metadata or {}),
        )


@dataclass(frozen=True)
class Draft:
    namespace: str
    stream: str
    start: int
    end: int
    source_schema: str
    source_digest: str
    expected_revision: int
    idempotency_key: str
    summary: str
    model_metadata_json: str = "{}"

    def canonical(self) -> str:
        _id(self.namespace)
        _id(self.stream)
        _id(self.idempotency_key)
        _range(self.start, self.end)
        _integer(self.expected_revision)
        if self.source_schema != SOURCE_SCHEMA:
            raise InvalidInput("Unsupported source schema")
        if (
            type(self.source_digest) is not str
            or len(self.source_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.source_digest)
        ):
            raise InvalidInput("Invalid source digest")
        if (
            type(self.summary) is not str
            or not self.summary.strip()
            or len(self.summary) > 1_000_000
        ):
            raise InvalidInput("Invalid summary")
        try:
            if (
                type(self.model_metadata_json) is not str
                or len(self.model_metadata_json) > MAX_ACQUIRED_BYTES
            ):
                raise ValueError()
            meta = json.loads(self.model_metadata_json)
        except (TypeError, ValueError, RecursionError):
            raise InvalidInput("Invalid model metadata") from None
        if type(meta) is not dict or _json(meta) != self.model_metadata_json:
            raise InvalidInput("Model metadata must be a canonical JSON object")
        return _json(asdict(self))


@dataclass(frozen=True)
class Checkpoint:
    revision: int
    draft: Draft
    draft_digest: str
    created_at: str
    validation: str = "validated_range"


@dataclass(frozen=True)
class HistoryPage:
    checkpoints: tuple[Checkpoint, ...]
    complete: bool
    next_revision: int | None


class HistoryIncomplete(CheckpointError):
    """Use history_page and its cursor to acquire the remaining revisions."""

    def __init__(self, page: HistoryPage):
        super().__init__("Checkpoint history exceeds one page; use history_page")
        self.page = page


_SCHEMA = [
    "CREATE TABLE IF NOT EXISTS jitmind_cp_meta (singleton INTEGER PRIMARY KEY CHECK(singleton=1), version INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS jitmind_cp_streams (namespace TEXT NOT NULL, stream TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0), PRIMARY KEY(namespace,stream))",
    """CREATE TABLE IF NOT EXISTS jitmind_cp_events (
        namespace TEXT NOT NULL, stream TEXT NOT NULL, sequence INTEGER NOT NULL CHECK(sequence>=0),
        event_id TEXT NOT NULL, payload_json TEXT NOT NULL, payload_digest TEXT NOT NULL,
        observed_time TEXT NOT NULL, kind TEXT NOT NULL, tool_call_id TEXT,
        PRIMARY KEY(namespace,stream,sequence), UNIQUE(namespace,stream,event_id),
        FOREIGN KEY(namespace,stream) REFERENCES jitmind_cp_streams(namespace,stream))""",
    """CREATE TABLE IF NOT EXISTS jitmind_cp_history (
        namespace TEXT NOT NULL, stream TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision>0),
        idempotency_key TEXT NOT NULL, draft_json TEXT NOT NULL, draft_digest TEXT NOT NULL,
        created_at TEXT NOT NULL, PRIMARY KEY(namespace,stream,revision),
        UNIQUE(namespace,stream,idempotency_key),
        FOREIGN KEY(namespace,stream) REFERENCES jitmind_cp_streams(namespace,stream))""",
]
for _table in ("jitmind_cp_events", "jitmind_cp_history"):
    for _action in ("UPDATE", "DELETE"):
        _SCHEMA.append(
            f"CREATE TRIGGER IF NOT EXISTS {_table}_{_action.lower()} BEFORE {_action} ON {_table} BEGIN SELECT RAISE(ABORT, 'immutable checkpoint record'); END"
        )


class SQLiteCheckpointLedger:
    """One trusted host-selected local DB; namespace comes only from authority.

    New events must increase sequence (gaps are permitted but cannot be repaired).
    Exact replay is allowed. Raw events and checkpoint history are retained forever.
    Each operation owns a connection; no model/effect executes under writer lock.
    """

    def __init__(
        self,
        path: str | Path,
        authority: ScopeAuthority,
        *,
        busy_timeout_ms: int = 2000,
        operation_timeout_ms: int = 5000,
    ) -> None:
        if type(busy_timeout_ms) is not int or not 1 <= busy_timeout_ms <= 30_000:
            raise InvalidInput("Invalid busy timeout")
        if type(operation_timeout_ms) is not int or not 1 <= operation_timeout_ms <= 30_000:
            raise InvalidInput("Invalid operation timeout")
        if str(path) == ":memory:" or str(path).startswith("file:"):
            raise InvalidInput("A local filesystem database is required")
        self.path = Path(path)
        self.authority = authority
        self.busy_timeout_ms = busy_timeout_ms
        self.operation_timeout_ms = operation_timeout_ms
        # Initialization is a host administration operation, never a request API.
        with self._transaction(write=True, initialize=True) as db:
            db.execute(_SCHEMA[0])
            version = db.execute(
                "SELECT version FROM jitmind_cp_meta WHERE singleton=1"
            ).fetchone()
            if version is not None and version[0] != 1:
                raise SchemaMismatch("Unsupported checkpoint schema")
            for statement in _SCHEMA[1:]:
                db.execute(statement)
            db.execute("INSERT OR IGNORE INTO jitmind_cp_meta VALUES (1,1)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(
            str(self.path),
            timeout=min(self.busy_timeout_ms, self.operation_timeout_ms) / 1000,
            isolation_level=None,
        )

    @contextmanager
    def _transaction(
        self, *, write: bool = False, initialize: bool = False
    ) -> Iterator[sqlite3.Connection]:
        db = None
        deadline = monotonic() + self.operation_timeout_ms / 1000
        token = _DEADLINE.set(deadline)
        try:
            db = self._connect()
            db.set_progress_handler(lambda: int(monotonic() >= deadline), 1000)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            wait_ms = min(self.busy_timeout_ms, self.operation_timeout_ms)
            db.execute(f"PRAGMA busy_timeout={wait_ms}")
            mode = db.execute("PRAGMA journal_mode").fetchone()[0]
            if mode.lower() != "delete":
                # Do not change a shared DB's journal policy implicitly.
                raise SchemaMismatch("Checkpoint database requires DELETE journal mode")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            if not initialize:
                row = db.execute(
                    "SELECT version FROM jitmind_cp_meta WHERE singleton=1"
                ).fetchone()
                if row is None or row[0] != 1:
                    raise SchemaMismatch("Unsupported checkpoint schema")
            _check_time()
            yield db
            _check_time()
            try:
                db.commit()
            except sqlite3.Error:
                raise CommitUncertain(
                    "Commit acknowledgement failed; retry exact operation after reopening"
                ) from None
        except sqlite3.Error:
            _check_time()
            raise StorageError("Checkpoint storage operation failed") from None
        finally:
            _DEADLINE.reset(token)
            if db is not None:
                db.set_progress_handler(None, 0)
                try:
                    if db.in_transaction:
                        db.rollback()
                except sqlite3.Error:
                    # Preserve the primary failure (including uncertain commit).
                    pass
                finally:
                    try:
                        db.close()
                    except sqlite3.Error:
                        pass

    def append(
        self,
        scope: ScopeContext,
        stream: str,
        sequence: int,
        *,
        event_id: str,
        payload: Any,
        observed_time: str,
        kind: str = "message",
        tool_call_id: str | None = None,
    ) -> Event:
        self.authority.require(scope)
        _id(stream)
        _id(event_id)
        _integer(sequence)
        if type(kind) is not str or kind not in KINDS:
            raise InvalidInput("Unknown event kind")
        if kind == "message":
            if tool_call_id is not None:
                raise InvalidInput("Message cannot identify a tool call")
        else:
            _id(tool_call_id)
        try:
            if type(observed_time) is not str or len(observed_time) > 128:
                raise ValueError()
            # Reject repeated/misplaced UTC designators before Python 3.10's
            # permissive parser can accept a leftover character in the time.
            if "Z" in observed_time[:-1] or (
                observed_time.endswith("Z")
                and any(sign in observed_time[10:-1] for sign in "+-")
            ):
                raise ValueError()
            # Python 3.10 does not accept the terminal UTC designator.
            parsed = datetime.fromisoformat(
                observed_time[:-1] + "+00:00" if observed_time.endswith("Z") else observed_time
            )
            if parsed.tzinfo is None:
                raise ValueError()
        except (TypeError, ValueError):
            raise InvalidInput("Observed time must include a timezone") from None
        encoded = _json(payload)
        event = Event(
            scope.namespace_id,
            stream,
            sequence,
            event_id,
            encoded,
            _digest(encoded),
            observed_time,
            kind,
            tool_call_id,
        )
        with self._transaction(write=True) as db:
            self.authority.require(scope)
            size = db.execute(
                f"SELECT COALESCE(SUM(n),0) FROM (SELECT {_EVENT_BYTES} AS n FROM jitmind_cp_events WHERE namespace=? AND stream=? AND (sequence=? OR event_id=?) LIMIT 3)",
                (scope.namespace_id, stream, sequence, event_id),
            ).fetchone()[0]
            if size > MAX_ACQUIRED_BYTES:
                raise CapacityExceeded("Event replay acquisition budget exceeded")
            existing = db.execute(
                "SELECT * FROM jitmind_cp_events WHERE namespace=? AND stream=? AND (sequence=? OR event_id=?)",
                (scope.namespace_id, stream, sequence, event_id),
            ).fetchall()
            if existing:
                if len(existing) != 1 or Event(**dict(existing[0])) != event:
                    raise EventConflict("Event identity conflicts with retained source")
            else:
                last = db.execute(
                    "SELECT MAX(sequence) FROM jitmind_cp_events WHERE namespace=? AND stream=?",
                    (scope.namespace_id, stream),
                ).fetchone()[0]
                if last is not None and sequence <= last:
                    raise EventConflict("New events must increase sequence")
                if kind != "message":
                    calls = db.execute(
                        "SELECT kind FROM jitmind_cp_events WHERE namespace=? AND stream=? AND tool_call_id=? ORDER BY sequence LIMIT 3",
                        (scope.namespace_id, stream, tool_call_id),
                    ).fetchall()
                    if (kind == "tool_start" and calls) or (
                        kind in TERMINALS and [r[0] for r in calls] != ["tool_start"]
                    ):
                        raise UnsafeRange(
                            "Unknown or already completed tool interaction"
                        )
                db.execute(
                    "INSERT OR IGNORE INTO jitmind_cp_streams(namespace,stream) VALUES (?,?)",
                    (scope.namespace_id, stream),
                )
                db.execute(
                    "INSERT INTO jitmind_cp_events VALUES (?,?,?,?,?,?,?,?,?)",
                    tuple(asdict(event).values()),
                )
            self.authority.require(scope)
        self.authority.require(scope)
        return event

    def _snapshot(
        self, db: sqlite3.Connection, namespace: str, stream: str, start: int, end: int
    ) -> Snapshot:
        # WHERE is inclusive and scope-exact; PK makes every sequence unique.
        count, size = db.execute(
            f"SELECT COUNT(*),COALESCE(SUM(n),0) FROM (SELECT {_EVENT_BYTES} AS n FROM jitmind_cp_events WHERE namespace=? AND stream=? AND sequence BETWEEN ? AND ? LIMIT ?)",
            (namespace, stream, start, end, MAX_RANGE + 1),
        ).fetchone()
        if count > MAX_RANGE or size > MAX_ACQUIRED_BYTES:
            raise CapacityExceeded("Source acquisition budget exceeded")
        if count != end - start + 1:
            raise UnsafeRange("Source range is not contiguous")
        # Bound the *entire* prefix, including message rows, before any payload
        # acquisition. The limited subquery also bounds aggregate SQL work.
        prefix_count, origin, prefix_size = db.execute(
            f"SELECT COUNT(*),MIN(sequence),COALESCE(SUM(n),0) FROM (SELECT sequence,{_PREFIX_BYTES} AS n FROM jitmind_cp_events WHERE namespace=? AND stream=? AND sequence<=? ORDER BY sequence LIMIT ?)",
            (namespace, stream, end, MAX_PREFIX_EVENTS + 1),
        ).fetchone()
        if prefix_count > MAX_PREFIX_EVENTS or prefix_size > MAX_ACQUIRED_BYTES:
            raise CapacityExceeded("Tool prefix acquisition budget exceeded")
        if origin is None or prefix_count != end - origin + 1:
            raise UnsafeRange("Tool boundary history is incomplete")
        _check_time()
        rows = db.execute(
            "SELECT * FROM jitmind_cp_events WHERE namespace=? AND stream=? AND sequence BETWEEN ? AND ? ORDER BY sequence",
            (namespace, stream, start, end),
        ).fetchall()
        if len(rows) != end - start + 1:
            raise UnsafeRange("Source range is not contiguous")
        events = tuple(Event(**dict(row)) for row in rows)
        for event in events:
            _check_time()
            if _digest(event.payload_json) != event.payload_digest:
                raise StorageError("Stored event digest mismatch")
        # Replay only through end. Later results never close an earlier boundary.
        calls: set[str] = set()
        prefix = db.execute(
            "SELECT sequence,kind,tool_call_id FROM jitmind_cp_events WHERE namespace=? AND stream=? AND sequence<=? AND kind!='message' ORDER BY sequence",
            (namespace, stream, end),
        )
        checked_start = False
        for row in prefix:
            _check_time()
            if row[0] >= start and not checked_start:
                if calls:
                    raise UnsafeRange("Tool interaction crosses start boundary")
                checked_start = True
            if row[1] == "tool_start":
                if row[2] is None or row[2] in calls:
                    raise UnsafeRange("Unknown tool interaction")
                calls.add(row[2])
            elif row[1] in TERMINALS and row[2] in calls:
                calls.remove(row[2])
            else:
                raise UnsafeRange("Unknown tool interaction")
        if calls:
            raise UnsafeRange("Tool interaction crosses end boundary")
        head = db.execute(
            "SELECT revision FROM jitmind_cp_streams WHERE namespace=? AND stream=?",
            (namespace, stream),
        ).fetchone()
        source = _json(
            {
                "schema": SOURCE_SCHEMA,
                "namespace": namespace,
                "stream": stream,
                "start": start,
                "end": end,
                "events": [asdict(e) for e in events],
            }
        )
        return Snapshot(
            namespace,
            stream,
            start,
            end,
            SOURCE_SCHEMA,
            _digest(source),
            head[0],
            events,
        )

    def snapshot(
        self, scope: ScopeContext, stream: str, start: int, end: int
    ) -> Snapshot:
        self.authority.require(scope)
        _id(stream)
        _range(start, end)
        with self._transaction() as db:
            result = self._snapshot(db, scope.namespace_id, stream, start, end)
            self.authority.require(scope)
        self.authority.require(scope)
        return result

    @staticmethod
    def _checkpoint(row: sqlite3.Row) -> Checkpoint:
        try:
            draft = Draft(**json.loads(row["draft_json"]))
            if _digest(draft.canonical()) != row["draft_digest"]:
                raise ValueError()
            if (
                draft.namespace != row["namespace"]
                or draft.stream != row["stream"]
                or draft.idempotency_key != row["idempotency_key"]
                or draft.expected_revision + 1 != row["revision"]
            ):
                raise ValueError()
            return Checkpoint(
                row["revision"], draft, row["draft_digest"], row["created_at"]
            )
        except CapacityExceeded:
            raise
        except (TypeError, ValueError, InvalidInput):
            raise StorageError("Invalid stored checkpoint") from None

    def publish(self, scope: ScopeContext, draft: Draft) -> Checkpoint:
        self.authority.require(scope)
        if type(draft) is not Draft:
            raise InvalidInput("Invalid checkpoint draft")
        if draft.namespace != scope.namespace_id:
            raise InvalidInput("Draft scope mismatch")
        encoded = draft.canonical()
        digest = _digest(encoded)
        with self._transaction(write=True) as db:
            self.authority.require(scope)
            prior_size = db.execute(
                f"SELECT {_HISTORY_BYTES} FROM jitmind_cp_history WHERE namespace=? AND stream=? AND idempotency_key=?",
                (scope.namespace_id, draft.stream, draft.idempotency_key),
            ).fetchone()
            if prior_size is not None and prior_size[0] > MAX_ACQUIRED_BYTES:
                raise CapacityExceeded("History acquisition budget exceeded")
            prior = db.execute(
                "SELECT * FROM jitmind_cp_history WHERE namespace=? AND stream=? AND idempotency_key=?",
                (scope.namespace_id, draft.stream, draft.idempotency_key),
            ).fetchone()
            if prior is not None:
                result = self._checkpoint(prior)
                if result.draft_digest != digest or prior["draft_json"] != encoded:
                    raise IdempotencyConflict(
                        "Idempotency key identifies a different draft"
                    )
            else:
                source = self._snapshot(
                    db, scope.namespace_id, draft.stream, draft.start, draft.end
                )
                if source.source_digest != draft.source_digest:
                    raise SourceChanged("Source digest changed")
                if source.expected_revision != draft.expected_revision:
                    raise RevisionConflict("Checkpoint revision changed")
                changed = db.execute(
                    "UPDATE jitmind_cp_streams SET revision=revision+1 WHERE namespace=? AND stream=? AND revision=?",
                    (scope.namespace_id, draft.stream, draft.expected_revision),
                ).rowcount
                if changed != 1:
                    raise RevisionConflict("Checkpoint revision changed")
                created = datetime.now(timezone.utc).isoformat()
                revision = draft.expected_revision + 1
                values = (
                    scope.namespace_id, draft.stream, revision,
                    draft.idempotency_key, encoded, digest, created,
                )
                if sum(len(str(value).encode("utf-8")) for value in values) > MAX_ACQUIRED_BYTES:
                    raise CapacityExceeded("Checkpoint exceeds history acquisition budget")
                db.execute(
                    "INSERT INTO jitmind_cp_history VALUES (?,?,?,?,?,?,?)",
                    values,
                )
                result = Checkpoint(revision, draft, digest, created)
            self.authority.require(scope)
        self.authority.require(scope)
        return result

    def history(self, scope: ScopeContext, stream: str) -> tuple[Checkpoint, ...]:
        """Compatible complete tuple, or explicit HistoryIncomplete with first page."""
        page = self.history_page(scope, stream)
        if not page.complete:
            raise HistoryIncomplete(page)
        return page.checkpoints

    def history_page(
        self, scope: ScopeContext, stream: str, *, after_revision: int = 0,
        limit: int = MAX_HISTORY_PAGE,
    ) -> HistoryPage:
        self.authority.require(scope)
        _id(stream)
        _integer(after_revision)
        if type(limit) is not int or not 1 <= limit <= MAX_HISTORY_PAGE:
            raise InvalidInput("Invalid history page limit")
        with self._transaction() as db:
            sizes = db.execute(
                f"SELECT revision,{_HISTORY_BYTES} FROM jitmind_cp_history WHERE namespace=? AND stream=? AND revision>? ORDER BY revision LIMIT ?",
                (scope.namespace_id, stream, after_revision, limit + 1),
            ).fetchall()
            total = 0
            revisions = []
            for revision, size in sizes[:limit]:
                _check_time()
                if total + size > MAX_ACQUIRED_BYTES:
                    break
                total += size
                revisions.append(revision)
            if sizes and not revisions:
                raise CapacityExceeded("One checkpoint exceeds history acquisition budget")
            complete = len(revisions) == len(sizes)
            rows = db.execute(
                "SELECT * FROM jitmind_cp_history WHERE namespace=? AND stream=? AND revision>? AND revision<=? ORDER BY revision LIMIT ?",
                (scope.namespace_id, stream, after_revision,
                 revisions[-1] if revisions else after_revision, limit),
            )
            result = tuple(self._checkpoint(row) for row in rows)
            _check_time()
            self.authority.require(scope)
        self.authority.require(scope)
        return HistoryPage(result, complete, None if complete else revisions[-1])

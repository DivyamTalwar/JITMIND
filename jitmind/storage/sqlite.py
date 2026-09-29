"""Single-host transactional page/fact authority; no provider calls in transactions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from jitmind.schemas import MemoryEntry, MemoryState, MemoryUpdate, Page

from .history import HISTORY_SCHEMA, HistoryMixin, processing_bound
from .history import timestamp as history_timestamp
from .models import (
    AcknowledgementUncertain,
    CapacityExceeded,
    DurableError,
    DurableReceipt,
    IdempotencyConflict,
    IngestRequest,
    InvalidRequest,
    MigrationError,
    NamespaceSnapshot,
    ProjectionEvent,
    Proposal,
    ProposalFailure,
    ReceiptContent,
    SchemaMismatch,
    StaleRevision,
    StorageBusy,
    StorageFailure,
    TargetNotFound,
    UnsafeJournal,
    canonical_json,
    validate_identifier,
)

SCHEMA_VERSION = 2
APPLICATION_ID = 0x4A49544D
MAX_SQLITE_INTEGER = 2**63 - 1
# Positive authority is required even for pages with no fact relationship.
# NOOP/DELETE pages remain immutable administrative history, never ordinary text.
VISIBLE_PAGE = "EXISTS (SELECT 1 FROM facts f WHERE f.namespace_id=p.namespace_id AND f.page_id=p.page_id AND f.status='active') AND NOT EXISTS (SELECT 1 FROM facts f WHERE f.namespace_id=p.namespace_id AND f.page_id=p.page_id AND f.status!='active')"
SCHEMA = [
    "CREATE TABLE namespaces (namespace_id TEXT PRIMARY KEY, revision INTEGER NOT NULL)",
    "CREATE TABLE pages (namespace_id TEXT NOT NULL, page_id TEXT NOT NULL, payload TEXT NOT NULL, revision INTEGER NOT NULL, PRIMARY KEY(namespace_id,page_id), FOREIGN KEY(namespace_id) REFERENCES namespaces(namespace_id))",
    "CREATE TABLE facts (namespace_id TEXT NOT NULL, memory_id TEXT NOT NULL, page_id TEXT NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(namespace_id,memory_id), FOREIGN KEY(namespace_id,page_id) REFERENCES pages(namespace_id,page_id))",
    "CREATE TABLE operations (namespace_id TEXT NOT NULL, idempotency_key TEXT NOT NULL, digest TEXT NOT NULL, receipt TEXT NOT NULL, PRIMARY KEY(namespace_id,idempotency_key), FOREIGN KEY(namespace_id) REFERENCES namespaces(namespace_id))",
    "CREATE TABLE outbox (namespace_id TEXT NOT NULL, event_id TEXT NOT NULL, revision INTEGER NOT NULL, memory_ids TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending', delivered_at TEXT, PRIMARY KEY(namespace_id,event_id), UNIQUE(namespace_id,revision), FOREIGN KEY(namespace_id) REFERENCES namespaces(namespace_id))",
    "CREATE TABLE aliases (namespace_id TEXT NOT NULL, legacy_page_id TEXT NOT NULL, page_id TEXT NOT NULL, PRIMARY KEY(namespace_id,legacy_page_id), FOREIGN KEY(namespace_id,page_id) REFERENCES pages(namespace_id,page_id))",
    "CREATE TABLE imports (namespace_id TEXT PRIMARY KEY, source_digest TEXT NOT NULL, page_count INTEGER NOT NULL, fact_count INTEGER NOT NULL, FOREIGN KEY(namespace_id) REFERENCES namespaces(namespace_id))",
    "CREATE TABLE projection (namespace_id TEXT NOT NULL, memory_id TEXT NOT NULL, revision INTEGER NOT NULL, status TEXT NOT NULL, PRIMARY KEY(namespace_id,memory_id), FOREIGN KEY(namespace_id,memory_id) REFERENCES facts(namespace_id,memory_id))",
    "CREATE INDEX facts_revision ON facts(namespace_id,revision,memory_id)",
    "CREATE INDEX outbox_pending ON outbox(namespace_id,state,revision)",
]


SCHEMAS = {1: SCHEMA, 2: SCHEMA + HISTORY_SCHEMA}


def _db_error(exc: sqlite3.Error) -> DurableError:
    # SQLite primary codes are stable; Python 3.10 does not expose the aliases
    # or sqlite_errorcode metadata. Prefer numeric engine evidence when present.
    code = getattr(exc, "sqlite_errorcode", None)
    if type(code) is int and code >= 0:
        return StorageBusy() if code & 255 in (5, 6) else StorageFailure()
    if code is None and isinstance(exc, sqlite3.OperationalError):
        message = str(exc)
        if message in {
            "database is locked",
            "database table is locked",
            "database schema is locked",
        } or message.startswith(
            ("database table is locked: ", "database schema is locked: ")
        ):
            return StorageBusy()
    return StorageFailure()


def _bound(limit: int) -> int:
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise InvalidRequest()
    return limit


def _revision_bound(value: int) -> None:
    if type(value) is not int or not 0 <= value <= MAX_SQLITE_INTEGER:
        raise InvalidRequest()


def _cursor(value: str) -> None:
    if type(value) is not str:
        raise InvalidRequest()
    if value:
        validate_identifier(value)


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError
    return parsed


def _record(payload: str, fields: set[str], max_bytes: int) -> dict:
    """Stored records must be complete; never synthesize IDs/defaults on corruption."""
    try:
        if not isinstance(payload, str) or len(payload) > max_bytes:
            raise StorageFailure()
        data = json.loads(payload)
        if type(data) is not dict or set(data) != fields:
            raise StorageFailure()
        canonical_json(data, max_bytes=max_bytes)
        return data
    except (InvalidRequest, ValueError, TypeError, RecursionError):
        raise StorageFailure() from None


def _entry(payload: str) -> MemoryEntry:
    entry = MemoryEntry.model_validate(
        _record(payload, set(MemoryEntry.model_fields), 262144), strict=True
    )
    try:
        times = {
            field: _timestamp(value)
            for field in (
                "t_created",
                "t_observed",
                "t_valid",
                "t_invalid",
                "t_expired",
                "last_accessed",
            )
            if (value := getattr(entry, field)) is not None
        }
        if (
            "t_valid" in times
            and "t_invalid" in times
            and times["t_valid"] > times["t_invalid"]
        ) or ("t_expired" in times and times["t_expired"] < times["t_created"]):
            raise ValueError
    except (ValueError, TypeError):
        raise StorageFailure() from None
    return entry


def _page(payload: str) -> Page:
    return Page.model_validate(
        _record(payload, set(Page.model_fields), 1048576), strict=True
    )


class SQLiteDurableStore(HistoryMixin):
    """Connections are operation-local, making instances safe across threads.

    Defaults: DELETE/FULL, 250ms SQLite busy timeout, two COMMIT retries.
    ``fault_hook(stage)`` and ``clock()`` are trusted application/test injections.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        journal_mode: str = "DELETE",
        busy_timeout_ms: int = 250,
        commit_retries: int = 2,
        clock: Callable[[], str] | None = None,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.path = Path(path).absolute()
        if journal_mode not in ("DELETE", "WAL"):
            raise UnsafeJournal()
        if (
            type(busy_timeout_ms) is not int
            or not 0 <= busy_timeout_ms <= 5000
            or type(commit_retries) is not int
            or not 0 <= commit_retries <= 5
        ):
            raise InvalidRequest()
        self.journal_mode = journal_mode
        self.busy_timeout_ms = busy_timeout_ms
        self.commit_retries = commit_retries
        self.clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self.fault_hook = fault_hook
        try:
            # Inspect identity/integrity before changing any persistent PRAGMA.
            with self._connection(initialize=True) as conn:
                if conn.execute("PRAGMA journal_mode").fetchone()[0].upper() == "WAL":
                    self._check_wal(conn)
                self._check_schema(conn, allow_empty=True)
                if self.journal_mode == "WAL":
                    self._check_wal(conn)
                actual = conn.execute(f"PRAGMA journal_mode={journal_mode}").fetchone()[
                    0
                ]
                if actual.upper() != journal_mode:
                    raise UnsafeJournal()
                with self._transaction(conn):
                    if conn.execute("PRAGMA user_version").fetchone()[0] == 0:
                        for statement in SCHEMAS[SCHEMA_VERSION]:
                            conn.execute(statement)
                        conn.execute(f"PRAGMA application_id={APPLICATION_ID}")
                        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                    self._check_schema(conn)
                    self.schema_version = conn.execute(
                        "PRAGMA user_version"
                    ).fetchone()[0]
        except OSError:
            raise StorageFailure() from None

    @staticmethod
    def _check_wal(conn: sqlite3.Connection) -> None:
        version = tuple(
            int(part)
            for part in conn.execute("SELECT sqlite_version()").fetchone()[0].split(".")
        )
        if version < (3, 51, 3):
            raise UnsafeJournal()

    @staticmethod
    def _check_schema(
        conn: sqlite3.Connection, *, allow_empty: bool = False, integrity: bool = True
    ) -> None:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        app_id = conn.execute("PRAGMA application_id").fetchone()[0]
        statements = {
            r[0]
            for r in conn.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL")
        }
        if allow_empty and version == 0 and app_id == 0 and not statements:
            return
        if (
            version not in SCHEMAS
            or app_id != APPLICATION_ID
            or statements != set(SCHEMAS.get(version, ()))
        ):
            raise SchemaMismatch()
        if not integrity:
            return
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise StorageFailure()
        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise StorageFailure()

    @staticmethod
    def _configure(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=FULL")
        if (
            conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1
            or conn.execute("PRAGMA synchronous").fetchone()[0] != 2
        ):
            raise StorageFailure()

    @contextmanager
    def _connection(
        self, *, initialize: bool = False, read_only: bool = False
    ) -> Iterator[sqlite3.Connection]:
        conn = None
        try:
            uri = self.path.as_uri() + (
                "?mode=rwc" if initialize else "?mode=ro" if read_only else "?mode=rw"
            )
            conn = sqlite3.connect(
                uri, uri=True, isolation_level=None, timeout=self.busy_timeout_ms / 1000
            )
            conn.row_factory = sqlite3.Row
            self._configure(conn)
            if not initialize:
                if (
                    conn.execute("PRAGMA user_version").fetchone()[0]
                    != self.schema_version
                    or conn.execute("PRAGMA application_id").fetchone()[0]
                    != APPLICATION_ID
                ):
                    raise SchemaMismatch()
                mode = conn.execute("PRAGMA journal_mode").fetchone()[0].upper()
                if mode != self.journal_mode:
                    raise UnsafeJournal()
                if mode == "WAL":
                    self._check_wal(conn)
            yield conn
        except sqlite3.Error as exc:
            raise _db_error(exc) from None
        except (ValidationError, json.JSONDecodeError):
            raise StorageFailure() from None
        finally:
            if conn is not None:
                conn.close()

    @contextmanager
    def _transaction(self, conn: sqlite3.Connection) -> Iterator[None]:
        conn.execute("BEGIN IMMEDIATE")
        try:
            if (
                hasattr(self, "schema_version")
                and conn.execute("PRAGMA user_version").fetchone()[0]
                != self.schema_version
            ):
                raise SchemaMismatch()
            yield
            for attempt in range(self.commit_retries + 1):
                try:
                    conn.execute("COMMIT")
                    break
                except sqlite3.Error as exc:
                    if (
                        isinstance(_db_error(exc), StorageBusy)
                        and attempt < self.commit_retries
                    ):
                        continue
                    raise
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def _fault(self, stage: str) -> None:
        if self.fault_hook:
            try:
                self.fault_hook(stage)
            except Exception:  # noqa: BLE001 - trusted hook may raise any private exception
                raise StorageFailure() from None

    def _now(self) -> str:
        try:
            now = self.clock()
            parsed = datetime.fromisoformat(now.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError
            return now
        except Exception:  # noqa: BLE001 - trusted clock may raise any private exception
            raise InvalidRequest() from None

    @staticmethod
    def _revision(conn: sqlite3.Connection, namespace_id: str) -> int:
        row = conn.execute(
            "SELECT revision FROM namespaces WHERE namespace_id=?", (namespace_id,)
        ).fetchone()
        value = row[0] if row else 0
        try:
            _revision_bound(value)
        except InvalidRequest:
            raise StorageFailure() from None
        return value

    @staticmethod
    def _receipt(
        conn: sqlite3.Connection, request: IngestRequest
    ) -> DurableReceipt | None:
        row = conn.execute(
            "SELECT digest,receipt FROM operations WHERE namespace_id=? AND idempotency_key=?",
            (request.namespace_id, request.idempotency_key),
        ).fetchone()
        if row is None:
            return None
        if row["digest"] != request.digest:
            raise IdempotencyConflict()
        return DurableReceipt.model_validate_json(row["receipt"])

    def lookup_receipt(self, request: IngestRequest) -> DurableReceipt | None:
        request.validated()
        with self._connection() as conn:
            return self._receipt(conn, request)

    def snapshot(self, namespace_id: str, *, limit: int = 100) -> NamespaceSnapshot:
        validate_identifier(namespace_id)
        _bound(limit)
        with self._connection() as conn:
            conn.execute("BEGIN")
            revision = self._revision(conn, namespace_id)
            rows = conn.execute(
                "SELECT payload FROM facts WHERE namespace_id=? AND status='active' ORDER BY revision DESC,memory_id LIMIT ?",
                (namespace_id, limit + 1),
            ).fetchall()
            return NamespaceSnapshot(
                revision,
                tuple(_entry(r[0]) for r in rows[:limit]),
                len(rows) > limit,
            )

    def ingest(
        self,
        request: IngestRequest,
        propose: Callable[[NamespaceSnapshot], Proposal],
        *,
        max_replans: int = 2,
        context_limit: int = 100,
    ) -> DurableReceipt:
        """Replay first; generate outside write lock; CAS namespace revision."""
        request.validated()
        _bound(context_limit)
        if type(max_replans) is not int or not 0 <= max_replans <= 5:
            raise InvalidRequest()
        for attempt in range(max_replans + 1):
            receipt = self.lookup_receipt(request)
            if receipt is not None:
                return receipt
            snapshot = self.snapshot(request.namespace_id, limit=context_limit)
            try:
                proposal = propose(snapshot)
            except DurableError:
                raise
            except Exception:  # noqa: BLE001 - provider/callback details are private
                raise ProposalFailure() from None
            try:
                return self.commit_proposal(request, snapshot.revision, proposal)
            except StaleRevision:
                if attempt == max_replans:
                    raise
        raise StaleRevision()  # unreachable; keeps the return contract explicit

    def commit_proposal(
        self, request: IngestRequest, expected_revision: int, proposal: Proposal
    ) -> DurableReceipt:
        return self._commit_proposal(request, expected_revision, proposal)

    def _commit_proposal(
        self,
        request: IngestRequest,
        expected_revision: int,
        proposal: Proposal,
        *,
        corrections=(),
    ) -> DurableReceipt:
        request.validated()
        _revision_bound(expected_revision)
        try:
            proposal = Proposal.model_validate(proposal.model_dump())
            canonical_json(proposal.model_dump(), max_bytes=524288)
            decision = proposal.decision
            for value in (decision.t_observed, decision.t_valid, decision.t_invalid):
                if (
                    value is not None
                    and datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo
                    is None
                ):
                    raise ValueError
            if (
                decision.t_valid
                and decision.t_invalid
                and datetime.fromisoformat(decision.t_valid.replace("Z", "+00:00"))
                > datetime.fromisoformat(decision.t_invalid.replace("Z", "+00:00"))
            ):
                raise ValueError
            if decision.operation in ("update", "delete") and not decision.target_id:
                raise ValueError
        except (AttributeError, ValidationError, ValueError, TypeError):
            raise InvalidRequest() from None
        now = self._now()
        page_id, event_id = str(uuid.uuid4()), str(uuid.uuid4())
        namespace = request.namespace_id
        self._fault("before_transaction")
        with self._connection() as conn, self._transaction(conn):
            existing = self._receipt(conn, request)
            if existing is not None:
                return existing
            if self._revision(conn, namespace) != expected_revision:
                raise StaleRevision()
            revision = expected_revision + 1
            _revision_bound(revision)
            target = None
            if decision.operation in ("update", "delete"):
                target = conn.execute(
                    "SELECT payload FROM facts WHERE namespace_id=? AND memory_id=? AND status='active'",
                    (namespace, decision.target_id),
                ).fetchone()
                if target is None:
                    raise TargetNotFound()
                target = _entry(target[0])
                # Validate the resulting parent BEFORE any writes. Corrupt stored
                # intervals fail in _entry; they must never be repaired implicitly.
                if _timestamp(now) < _timestamp(target.t_created):
                    raise InvalidRequest()
                if decision.operation == "update":
                    end = decision.t_valid or now
                    if target.t_valid and _timestamp(end) < _timestamp(target.t_valid):
                        raise InvalidRequest()
                elif target.t_valid and _timestamp(target.t_valid) > _timestamp(now):
                    # Explicit future-fact retraction: an empty validity interval.
                    end = target.t_valid
                else:
                    end = target.t_invalid or now
                target.t_invalid = end
                target.t_expired = now
                target.status = (
                    "deleted" if decision.operation == "delete" else "superseded"
                )
            conn.execute(
                "INSERT INTO namespaces VALUES (?,?) ON CONFLICT(namespace_id) DO UPDATE SET revision=excluded.revision",
                (namespace, revision),
            )
            memory_id = (
                str(uuid.uuid4()) if decision.operation in ("add", "update") else None
            )
            fact_meta = json.loads(request.metadata_json)
            if target and decision.operation == "update" and self.schema_version == 2:
                key, repo, snapshot, single = self._history_identity(
                    conn, namespace, target.id
                )
                inherited = {
                    "fact_key": key,
                    "repo_id": repo,
                    "snapshot_id": snapshot,
                    "single_valued": single,
                }
                for field, value in inherited.items():
                    if field in fact_meta and fact_meta[field] != value:
                        raise InvalidRequest()
                    if value is not None:
                        fact_meta[field] = value
            meta = dict(fact_meta)
            meta.update(
                page_id=page_id,
                memory_id=memory_id,
                decorated=proposal.decorated,
                t_observed=decision.t_observed,
                t_valid=decision.t_valid,
                t_invalid=decision.t_invalid,
                _durable={"decision": decision.model_dump(), "revision": revision},
            )
            if request.user_id is not None:
                meta["user_id"] = request.user_id
            page = Page(header=proposal.header, content=request.message, meta=meta)
            conn.execute(
                "INSERT INTO pages VALUES (?,?,?,?)",
                (
                    namespace,
                    page_id,
                    canonical_json(page.model_dump(), max_bytes=1048576),
                    revision,
                ),
            )
            if self.schema_version == 2:
                conn.execute(
                    "INSERT INTO history_pages SELECT * FROM pages WHERE namespace_id=? AND page_id=?",
                    (namespace, page_id),
                )
            self._fault("after_page_insert")
            changed = []
            if target:
                old = target
                conn.execute(
                    "UPDATE facts SET status=?,revision=?,payload=? WHERE namespace_id=? AND memory_id=?",
                    (
                        old.status,
                        revision,
                        canonical_json(old.model_dump()),
                        namespace,
                        old.id,
                    ),
                )
                changed.append(old.id)
            if memory_id:
                entry = MemoryEntry(
                    id=memory_id,
                    content=decision.updated_content or proposal.abstract,
                    tier=decision.importance,
                    t_created=now,
                    t_observed=decision.t_observed or now,
                    t_valid=decision.t_valid,
                    t_invalid=decision.t_invalid,
                    source_page_id=page_id,
                    version_of=decision.target_id
                    if decision.operation == "update"
                    else None,
                    meta=fact_meta,
                )
                conn.execute(
                    "INSERT INTO facts VALUES (?,?,?,?,?,?)",
                    (
                        namespace,
                        entry.id,
                        page_id,
                        entry.status,
                        revision,
                        canonical_json(entry.model_dump()),
                    ),
                )
                changed.append(entry.id)
            self._history_correct(
                conn, namespace, revision, corrections, changed, fact_meta
            )
            self._fault("after_fact_insert")
            receipt = DurableReceipt(
                namespace_id=namespace,
                idempotency_key=request.idempotency_key,
                request_digest=request.digest,
                operation=decision.operation,
                page_id=page_id,
                memory_id=memory_id,
                revision=revision,
                event_id=event_id,
                committed_at=now,
            )
            conn.execute(
                "INSERT INTO outbox(namespace_id,event_id,revision,memory_ids) VALUES (?,?,?,?)",
                (namespace, event_id, revision, canonical_json(changed)),
            )
            conn.execute(
                "INSERT INTO operations VALUES (?,?,?,?)",
                (
                    namespace,
                    request.idempotency_key,
                    request.digest,
                    receipt.model_dump_json(),
                ),
            )
            self._history_record(conn, namespace, revision, now, event_id, changed)
            self._fault("after_outbox_receipt")
            self._fault("before_commit")
        try:
            self._fault("after_commit")
        except DurableError:
            raise AcknowledgementUncertain() from None
        return receipt

    def visible_entries(
        self, namespace_id: str, memory_ids: list[str]
    ) -> list[MemoryEntry]:
        """Filter external/stale hits against authority in a single read snapshot.

        Returns active entries in input order, removing duplicates and unknown IDs.
        Payloads come from SQLite, never from the external projection.
        """
        validate_identifier(namespace_id)
        if type(memory_ids) is not list or len(memory_ids) > 1000:
            raise InvalidRequest()
        ids = list(dict.fromkeys(validate_identifier(value) for value in memory_ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT memory_id,payload FROM facts WHERE namespace_id=? AND status='active' AND memory_id IN ({placeholders})",
                (namespace_id, *ids),
            ).fetchall()
            found = {row["memory_id"]: _entry(row["payload"]) for row in rows}
        return [found[identifier] for identifier in ids if identifier in found]

    def get_entry(
        self, namespace_id: str, memory_id: str, *, include_inactive: bool = False
    ) -> MemoryEntry | None:
        validate_identifier(namespace_id)
        validate_identifier(memory_id)
        if type(include_inactive) is not bool:
            raise InvalidRequest()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT payload,status FROM facts WHERE namespace_id=? AND memory_id=?",
                (namespace_id, memory_id),
            ).fetchone()
            if row is None or (row["status"] != "active" and not include_inactive):
                return None
            return _entry(row["payload"])

    @staticmethod
    def _visible_page(
        conn: sqlite3.Connection, namespace: str, page_id: str
    ) -> Page | None:
        row = conn.execute(
            f"SELECT payload FROM pages p WHERE namespace_id=? AND page_id=? AND {VISIBLE_PAGE}",
            (namespace, page_id),
        ).fetchone()
        return _page(row[0]) if row else None

    def get_page(self, namespace_id: str, page_id: str) -> Page | None:
        validate_identifier(namespace_id)
        validate_identifier(page_id)
        with self._connection() as conn:
            return self._visible_page(conn, namespace_id, page_id)

    def resolve_page_alias(self, namespace_id: str, legacy_page_id: str) -> str | None:
        validate_identifier(namespace_id)
        validate_identifier(legacy_page_id)
        with self._connection() as conn:
            row = conn.execute(
                "SELECT page_id FROM aliases WHERE namespace_id=? AND legacy_page_id=?",
                (namespace_id, legacy_page_id),
            ).fetchone()
            return row[0] if row else None

    @staticmethod
    def _validate_receipt(receipt: DurableReceipt) -> None:
        if not isinstance(receipt, DurableReceipt):
            raise InvalidRequest()
        validate_identifier(receipt.namespace_id)
        validate_identifier(receipt.idempotency_key)
        _revision_bound(receipt.revision)

    def _receipt_content(
        self, conn: sqlite3.Connection, receipt: DurableReceipt
    ) -> ReceiptContent:
        row = conn.execute(
            "SELECT receipt FROM operations WHERE namespace_id=? AND idempotency_key=?",
            (receipt.namespace_id, receipt.idempotency_key),
        ).fetchone()
        if row is None or DurableReceipt.model_validate_json(row[0]) != receipt:
            raise InvalidRequest()
        page = self._visible_page(conn, receipt.namespace_id, receipt.page_id)
        if page is not None:
            return ReceiptContent(receipt, "available", page)
        retired = conn.execute(
            "SELECT 1 FROM facts WHERE namespace_id=? AND page_id=? AND status!='active'",
            (receipt.namespace_id, receipt.page_id),
        ).fetchone()
        return ReceiptContent(receipt, "retired" if retired else "unavailable", None)

    def receipt_content(self, receipt: DurableReceipt) -> ReceiptContent:
        """Reconcile identity and read current visibility in one snapshot.

        NOOP/DELETE pages have no active fact and return unavailable. A retired
        ADD/UPDATE returns retired. Neither result carries historical payloads.
        """
        self._validate_receipt(receipt)
        with self._connection() as conn:
            conn.execute("BEGIN")
            return self._receipt_content(conn, receipt)

    def memory_update(
        self, receipt: DurableReceipt, *, limit: int = 1000
    ) -> MemoryUpdate:
        """Compatibility view of current authority; receipts never cache private payloads."""
        _bound(limit)
        self._validate_receipt(receipt)
        with self._connection() as conn:
            conn.execute("BEGIN")
            content = self._receipt_content(conn, receipt)
            rows = conn.execute(
                "SELECT payload FROM facts WHERE namespace_id=? AND status='active' ORDER BY revision,memory_id LIMIT ?",
                (receipt.namespace_id, limit + 1),
            ).fetchall()
            page = content.page
            if page is None:
                page = Page(
                    header="",
                    content="",
                    meta={
                        "page_id": receipt.page_id,
                        "redacted": True,
                        "content_status": content.status,
                    },
                )
            return MemoryUpdate(
                new_state=MemoryState(
                    abstracts=[_entry(r[0]).content for r in rows[:limit]]
                ),
                new_page=page,
                debug={
                    "durable_receipt": receipt.model_dump(),
                    "content_status": content.status,
                    "state_truncated": len(rows) > limit,
                    "decorated_page": page.meta.get("decorated", ""),
                },
            )

    def pending_events(
        self, namespace_id: str, *, limit: int = 100, after_revision: int = 0
    ) -> list[dict]:
        validate_identifier(namespace_id)
        _bound(limit)
        _revision_bound(after_revision)
        with self._connection() as conn:
            return [
                dict(row) | {"memory_ids": json.loads(row["memory_ids"])}
                for row in conn.execute(
                    "SELECT * FROM outbox WHERE namespace_id=? AND state='pending' AND revision>? ORDER BY revision LIMIT ?",
                    (namespace_id, after_revision, limit),
                )
            ]

    def outbox_events(
        self, namespace_id: str, *, after_revision: int = 0, limit: int = 100
    ) -> tuple[ProjectionEvent, ...]:
        """All committed events, independent of reference-consumer delivery state."""
        validate_identifier(namespace_id)
        _revision_bound(after_revision)
        _bound(limit)
        with self._connection(read_only=True) as conn:
            conn.execute("BEGIN")
            size = conn.execute(
                "SELECT COALESCE(SUM(n),0) FROM (SELECT length(CAST(memory_ids AS BLOB)) n FROM outbox WHERE namespace_id=? AND revision>? ORDER BY revision LIMIT ?)",
                (namespace_id, after_revision, limit),
            ).fetchone()[0]
            if size > 8 * 1024 * 1024:
                raise CapacityExceeded()
            return tuple(
                self._projection_event(row)
                for row in conn.execute(
                    "SELECT namespace_id,event_id,revision,memory_ids FROM outbox WHERE namespace_id=? AND revision>? ORDER BY revision LIMIT ?",
                    (namespace_id, after_revision, limit),
                )
            )

    def get_event(self, namespace_id: str, event_id: str) -> ProjectionEvent | None:
        validate_identifier(namespace_id)
        validate_identifier(event_id)
        with self._connection(read_only=True) as conn, processing_bound(conn):
            conn.execute("BEGIN")
            self._check_schema(conn, integrity=False)
            size = conn.execute(
                "SELECT length(CAST(memory_ids AS BLOB)) FROM outbox WHERE namespace_id=? AND event_id=?",
                (namespace_id, event_id),
            ).fetchone()
            if size is None:
                return None
            if type(size[0]) is not int or not 0 < size[0] <= 262144:
                raise StorageFailure()
            row = conn.execute(
                "SELECT namespace_id,event_id,revision,memory_ids FROM outbox WHERE namespace_id=? AND event_id=?",
                (namespace_id, event_id),
            ).fetchone()
            return self._projection_event(row) if row is not None else None

    @staticmethod
    def _projection_event(row) -> ProjectionEvent:
        if len(row["memory_ids"].encode("utf-8")) > 262144:
            raise StorageFailure()
        try:
            _revision_bound(row["revision"])
            validate_identifier(row["namespace_id"])
            validate_identifier(row["event_id"])
        except InvalidRequest:
            raise StorageFailure() from None
        ids = json.loads(row["memory_ids"])
        if type(ids) is not list or len(ids) > 1000:
            raise StorageFailure()
        try:
            ids = tuple(validate_identifier(value) for value in ids)
        except InvalidRequest:
            raise StorageFailure() from None
        return ProjectionEvent(
            row["namespace_id"], row["event_id"], row["revision"], ids
        )

    def deliver_event(self, namespace_id: str, event_id: str) -> None:
        """Idempotent reference projection, atomically delivered with its receipt.

        Re-reads current authority, so delayed events cannot resurrect facts.
        External projections must likewise filter hits through get_entry/get_page.
        """
        validate_identifier(namespace_id)
        validate_identifier(event_id)
        now = self._now()
        with self._connection() as conn, self._transaction(conn):
            event = conn.execute(
                "SELECT * FROM outbox WHERE namespace_id=? AND event_id=?",
                (namespace_id, event_id),
            ).fetchone()
            if event is None:
                raise InvalidRequest()
            if event["state"] == "delivered":
                return
            for memory_id in json.loads(event["memory_ids"]):
                fact = conn.execute(
                    "SELECT status,revision FROM facts WHERE namespace_id=? AND memory_id=?",
                    (namespace_id, memory_id),
                ).fetchone()
                conn.execute(
                    "INSERT INTO projection VALUES (?,?,?,?) ON CONFLICT(namespace_id,memory_id) DO UPDATE SET revision=excluded.revision,status=excluded.status WHERE excluded.revision>projection.revision",
                    (namespace_id, memory_id, fact["revision"], fact["status"]),
                )
            conn.execute(
                "UPDATE outbox SET state='delivered',delivered_at=? WHERE namespace_id=? AND event_id=?",
                (now, namespace_id, event_id),
            )
            self._fault("before_projection_commit")
        self._fault("after_projection")

    def drain_outbox(self, namespace_id: str, *, limit: int = 100) -> int:
        events = self.pending_events(namespace_id, limit=limit)
        for event in events:
            self.deliver_event(namespace_id, event["event_id"])
        return len(events)

    def projected_entries(
        self, namespace_id: str, *, limit: int = 100, after_id: str = ""
    ) -> list[MemoryEntry]:
        validate_identifier(namespace_id)
        _bound(limit)
        _cursor(after_id)
        with self._connection() as conn:
            return [
                _entry(r[0])
                for r in conn.execute(
                    "SELECT f.payload FROM projection p JOIN facts f USING(namespace_id,memory_id) WHERE p.namespace_id=? AND p.memory_id>? AND p.status='active' AND f.status='active' ORDER BY p.memory_id LIMIT ?",
                    (namespace_id, after_id, limit),
                )
            ]

    def status(self, namespace_id: str) -> dict:
        validate_identifier(namespace_id)
        with self._connection() as conn:
            conn.execute("BEGIN")
            revision = self._revision(conn, namespace_id)
            pending, first = conn.execute(
                "SELECT COUNT(*),MIN(revision) FROM outbox WHERE namespace_id=? AND state='pending'",
                (namespace_id,),
            ).fetchone()
            counts = {
                table: conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE namespace_id=?",
                    (namespace_id,),
                ).fetchone()[0]
                for table in ("pages", "facts", "operations")
            }
            return dict(
                revision=revision,
                pending_events=pending,
                delivery_watermark=first - 1 if first else revision,
                **counts,
            )

    def diagnostics(self) -> dict:
        with self._connection() as conn:
            return {
                "sqlite_version": conn.execute("SELECT sqlite_version()").fetchone()[0],
                "sqlite_source_id": conn.execute(
                    "SELECT sqlite_source_id()"
                ).fetchone()[0],
                "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
                "synchronous": conn.execute("PRAGMA synchronous").fetchone()[0],
                "foreign_keys": conn.execute("PRAGMA foreign_keys").fetchone()[0],
                "schema_version": conn.execute("PRAGMA user_version").fetchone()[0],
            }

    def import_staged(self, staged, namespace_id: str) -> dict:
        """Import validated legacy data into an empty namespace in one transaction."""
        from .migration import (
            MAX_RECORDS,
            StagedLegacy,
            prepare_records,
            source_envelope,
        )

        validate_identifier(namespace_id)
        if not isinstance(staged, StagedLegacy) or any(
            type(count) is not int or not 0 <= count <= MAX_RECORDS
            for count in (staged.page_count, staged.fact_count)
        ):
            raise MigrationError()
        if (
            hashlib.sha256(source_envelope(staged.source_json)).hexdigest()
            != staged.source_digest
        ):
            raise MigrationError()
        pages, entries, aliases = prepare_records(staged.source_json)
        if len(pages) != staged.page_count or len(entries) != staged.fact_count:
            raise MigrationError()
        report = {
            "source_digest": staged.source_digest,
            "page_count": len(pages),
            "fact_count": len(entries),
        }
        self._fault("before_import")
        with self._connection() as conn, self._transaction(conn):
            prior = conn.execute(
                "SELECT source_digest FROM imports WHERE namespace_id=?",
                (namespace_id,),
            ).fetchone()
            if prior and prior[0] == staged.source_digest:
                return report
            if self._revision(conn, namespace_id) != 0:
                raise MigrationError()
            conn.execute("INSERT INTO namespaces VALUES (?,1)", (namespace_id,))
            for page in pages:
                conn.execute(
                    "INSERT INTO pages VALUES (?,?,?,1)",
                    (
                        namespace_id,
                        page.meta["page_id"],
                        canonical_json(page.model_dump(), max_bytes=1048576),
                    ),
                )
            self._fault("after_import_pages")
            for entry in entries:
                conn.execute(
                    "INSERT INTO facts VALUES (?,?,?,?,1,?)",
                    (
                        namespace_id,
                        entry.id,
                        entry.source_page_id,
                        entry.status,
                        canonical_json(entry.model_dump()),
                    ),
                )
            for alias, page_id in aliases.items():
                conn.execute(
                    "INSERT INTO aliases VALUES (?,?,?)", (namespace_id, alias, page_id)
                )
            conn.execute(
                "INSERT INTO imports VALUES (?,?,?,?)",
                (namespace_id, staged.source_digest, len(pages), len(entries)),
            )
            conn.execute(
                "INSERT INTO outbox(namespace_id,event_id,revision,memory_ids) VALUES (?,?,1,?)",
                (
                    namespace_id,
                    str(uuid.uuid4()),
                    canonical_json([e.id for e in entries]),
                ),
            )
            if self.schema_version == 2:
                recorded = self._now()
                self._history_baseline(
                    conn, namespace_id, 1, history_timestamp(recorded)
                )
                conn.execute(
                    "INSERT INTO history_pages SELECT * FROM pages WHERE namespace_id=?",
                    (namespace_id,),
                )
                for entry in entries:
                    self._history_version(conn, namespace_id, 1, entry, strict=False)
                with processing_bound(conn):
                    self._history_validate_baseline(conn, namespace_id, 1)
            self._fault("before_import_commit")
        return report

    def backup(self, destination: str | Path, *, timeout_seconds: float = 10) -> Path:
        """Consistent SQLite backup to a NEW file. Never replace a candidate/live DB."""
        dest = Path(destination).absolute()
        if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 60:
            raise InvalidRequest()
        created = False
        try:
            with dest.open("xb"):
                created = True
            deadline = time.monotonic() + timeout_seconds

            def progress(status: int, remaining: int, total: int) -> None:
                if time.monotonic() > deadline:
                    raise StorageBusy()

            with self._connection(read_only=True) as source:
                target = sqlite3.connect(dest, isolation_level=None)
                try:
                    self._configure(target)
                    source.backup(target, pages=128, progress=progress, sleep=0.01)
                    self._check_schema(target)
                finally:
                    target.close()
            return dest
        except (OSError, sqlite3.Error):
            if created:
                dest.unlink(missing_ok=True)
            raise StorageFailure() from None
        except BaseException:
            if created:
                dest.unlink(missing_ok=True)
            raise

    @classmethod
    def restore(
        cls, backup_path: str | Path, destination: str | Path, **kwargs
    ) -> SQLiteDurableStore:
        """Validate/read backup and restore to a NEW destination; never roll back live writes."""
        if "journal_mode" in kwargs:
            raise InvalidRequest()
        source_path = Path(backup_path).absolute()
        try:
            source = sqlite3.connect(
                source_path.as_uri() + "?mode=ro", uri=True, isolation_level=None
            )
            try:
                cls._configure(source)
                journal = source.execute("PRAGMA journal_mode").fetchone()[0].upper()
                if journal == "WAL":
                    cls._check_wal(source)
                cls._check_schema(source)
                source_version = source.execute("PRAGMA user_version").fetchone()[0]
            finally:
                source.close()
        except sqlite3.Error:
            raise StorageFailure() from None
        # Use the same bounded backup API without opening the source for mutation.
        holder = object.__new__(cls)
        holder.path = source_path
        holder.schema_version = source_version
        holder.journal_mode = journal
        holder.busy_timeout_ms = 250
        holder.backup(destination)
        return cls(destination, journal_mode=journal, **kwargs)


class DurableMemoryAdapter:
    """Bounded read adapter. Writes MUST go through SQLiteDurableStore.ingest."""

    def __init__(self, store: SQLiteDurableStore, namespace_id: str):
        self.store, self.namespace_id = store, validate_identifier(namespace_id)

    def load(self) -> MemoryState:
        snapshot = self.store.snapshot(self.namespace_id, limit=1000)
        if snapshot.truncated:
            raise CapacityExceeded()
        return MemoryState(abstracts=[e.content for e in reversed(snapshot.entries)])

    def get_entries(
        self, include_inactive: bool = False, *, limit: int = 100, after_id: str = ""
    ) -> list[MemoryEntry]:
        _bound(limit)
        _cursor(after_id)
        if type(include_inactive) is not bool:
            raise InvalidRequest()
        with self.store._connection() as conn:
            return [
                _entry(r[0])
                for r in conn.execute(
                    "SELECT payload FROM facts WHERE namespace_id=? AND memory_id>? AND (? OR status='active') ORDER BY memory_id LIMIT ?",
                    (self.namespace_id, after_id, include_inactive, limit),
                )
            ]

    def get_entry_by_id(self, entry_id: str) -> MemoryEntry | None:
        return self.store.get_entry(self.namespace_id, entry_id)

    def add(self, abstract: str) -> None:
        raise InvalidRequest()

    def save(self, state: MemoryState) -> None:
        raise InvalidRequest()


class DurablePageAdapter:
    def __init__(self, store: SQLiteDurableStore, namespace_id: str):
        self.store, self.namespace_id = store, validate_identifier(namespace_id)

    def get(self, page_id: str) -> Page | None:
        return self.store.get_page(self.namespace_id, page_id)

    def list_pages(self, *, limit: int = 100, after_id: str = "") -> list[Page]:
        _bound(limit)
        _cursor(after_id)
        with self.store._connection() as conn:
            return [
                _page(row[0])
                for row in conn.execute(
                    f"SELECT payload FROM pages p WHERE namespace_id=? AND page_id>? AND {VISIBLE_PAGE} ORDER BY page_id LIMIT ?",
                    (self.namespace_id, after_id, limit),
                )
            ]

    def load(self) -> list[Page]:
        with self.store._connection() as conn:
            rows = conn.execute(
                f"SELECT payload FROM pages p WHERE namespace_id=? AND {VISIBLE_PAGE} ORDER BY page_id LIMIT 1001",
                (self.namespace_id,),
            ).fetchall()
            if len(rows) > 1000:
                raise CapacityExceeded()
            return [_page(row[0]) for row in rows]

    def add(self, page: Page) -> None:
        raise InvalidRequest()

    def save(self, pages: list[Page]) -> None:
        raise InvalidRequest()

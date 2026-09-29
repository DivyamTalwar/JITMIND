"""Append-only bitemporal versions in the durable authority's SQLite transaction.

Raw methods are trusted-host APIs. Scope capabilities belong to the calling facade.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import (
    CapacityExceeded,
    DurableReceipt,
    HistoricalFact,
    HistoricalSnapshot,
    IngestRequest,
    InvalidRequest,
    Proposal,
    SchemaMismatch,
    StorageFailure,
    TargetNotFound,
    ValidityCorrection,
    canonical_json,
    validate_identifier,
)

EFFECT_COLUMNS = (
    "namespace_id",
    "memory_id",
    "revision",
    "page_id",
    "fact_key",
    "repo_id",
    "snapshot_id",
    "single_valued",
    "scope_valid",
    "valid_from",
    "valid_to",
    "status",
)

HISTORY_SCHEMA = [
    "CREATE TABLE history_pages (namespace_id TEXT NOT NULL, page_id TEXT NOT NULL, payload TEXT NOT NULL, revision INTEGER NOT NULL, PRIMARY KEY(namespace_id,page_id), FOREIGN KEY(namespace_id) REFERENCES namespaces(namespace_id))",
    "CREATE TABLE history_coverage (namespace_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, recorded_at TEXT NOT NULL, FOREIGN KEY(namespace_id) REFERENCES namespaces(namespace_id))",
    "CREATE TABLE history_revisions (namespace_id TEXT NOT NULL, revision INTEGER NOT NULL, recorded_at TEXT NOT NULL, event_id TEXT, PRIMARY KEY(namespace_id,revision), FOREIGN KEY(namespace_id) REFERENCES history_coverage(namespace_id), FOREIGN KEY(namespace_id,event_id) REFERENCES outbox(namespace_id,event_id) DEFERRABLE INITIALLY DEFERRED)",
    "CREATE TABLE history_versions (namespace_id TEXT NOT NULL, memory_id TEXT NOT NULL, revision INTEGER NOT NULL, page_id TEXT NOT NULL, fact_key TEXT NOT NULL, repo_id TEXT, snapshot_id TEXT, single_valued INTEGER NOT NULL, scope_valid INTEGER NOT NULL, valid_from TEXT, valid_to TEXT, status TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(namespace_id,memory_id,revision), FOREIGN KEY(namespace_id,revision) REFERENCES history_revisions(namespace_id,revision), FOREIGN KEY(namespace_id,page_id) REFERENCES history_pages(namespace_id,page_id))",
    "CREATE TABLE history_effects (namespace_id TEXT NOT NULL, memory_id TEXT NOT NULL, revision INTEGER NOT NULL, page_id TEXT NOT NULL, fact_key TEXT NOT NULL, repo_id TEXT, snapshot_id TEXT, single_valued INTEGER NOT NULL, scope_valid INTEGER NOT NULL, valid_from TEXT, valid_to TEXT, status TEXT NOT NULL, payload_digest TEXT NOT NULL, page_digest TEXT NOT NULL, PRIMARY KEY(namespace_id,memory_id,revision), FOREIGN KEY(namespace_id,revision) REFERENCES history_revisions(namespace_id,revision))",
    "CREATE INDEX history_time ON history_revisions(namespace_id,recorded_at,revision)",
    "CREATE INDEX history_scope ON history_versions(namespace_id,repo_id,snapshot_id,memory_id,revision)",
    "CREATE INDEX history_keys ON history_versions(namespace_id,fact_key,repo_id,snapshot_id,revision)",
]
for _table in (
    "history_coverage",
    "history_revisions",
    "history_versions",
    "history_pages",
    "history_effects",
    "operations",
):
    for _action in ("UPDATE", "DELETE"):
        HISTORY_SCHEMA.append(
            f"CREATE TRIGGER {_table}_{_action.lower()} BEFORE {_action} ON {_table} BEGIN SELECT RAISE(ABORT, 'immutable history'); END"
        )

# Payload materialization is bounded separately from SQL processing. A progress
# budget prevents an adversarial archive from making even a LIMIT query unbounded.
MAX_HISTORY_BYTES = 8 * 1024 * 1024
MAX_HISTORY_STEPS = 2_000_000


@contextmanager
def processing_bound(conn):
    remaining = MAX_HISTORY_STEPS // 1000

    def progress():
        nonlocal remaining
        remaining -= 1
        return int(remaining <= 0)

    conn.set_progress_handler(progress, 1000)
    try:
        yield
    except sqlite3.OperationalError:
        if remaining <= 0:
            raise CapacityExceeded() from None
        raise
    finally:
        conn.set_progress_handler(None, 0)


def timestamp(value: str | datetime) -> str:
    try:
        if type(value) is str:
            if len(value) > 64:
                raise ValueError
            offset = re.search(r"([+-])(\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?$", value)
            if not value.endswith("Z") and (
                offset is None
                or int(offset[2]) > 23
                or int(offset[3]) > 59
                or (offset[4] and int(offset[4]) > 59)
            ):
                raise ValueError
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except (ValueError, TypeError, OverflowError):
        raise InvalidRequest() from None


def identity(entry):
    key = validate_identifier(entry.meta.get("fact_key", entry.id))
    repo, snapshot = entry.meta.get("repo_id"), entry.meta.get("snapshot_id")
    if (repo is None) != (snapshot is None):
        raise InvalidRequest()
    if repo is not None:
        validate_identifier(repo)
        validate_identifier(snapshot)
    single = entry.meta.get("single_valued", True)
    if type(single) is not bool:
        raise InvalidRequest()
    return key, repo, snapshot, single


def source_identity(entry, page, namespace):
    archived = entry.model_copy(deep=True)
    for meta in (archived.meta, page.meta):
        if ("namespace_id" in meta and meta["namespace_id"] != namespace) or (
            "namespace" in meta and meta["namespace"] not in (namespace, [namespace])
        ):
            raise InvalidRequest()
    if page.meta.get("page_id") != entry.source_page_id or (
        page.meta.get("memory_id") is not None and page.meta["memory_id"] != entry.id
    ):
        raise InvalidRequest()
    for field in ("repo_id", "snapshot_id"):
        fact_value, page_value = archived.meta.get(field), page.meta.get(field)
        if (
            fact_value is not None
            and page_value is not None
            and fact_value != page_value
        ):
            raise InvalidRequest()
        if fact_value is None and page_value is not None:
            archived.meta[field] = page_value
    return archived, identity(archived)


def digest(payload):
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def page_payload(conn, namespace, page_id):
    # Preflight in the caller's transaction, before returning any payload bytes.
    size = conn.execute(
        "SELECT length(CAST(payload AS BLOB)) FROM history_pages WHERE namespace_id=? AND page_id=?",
        (namespace, page_id),
    ).fetchone()
    if size is None:
        return None
    if type(size[0]) is not int or not 0 < size[0] <= 1048576:
        raise CapacityExceeded()
    return conn.execute(
        "SELECT payload FROM history_pages WHERE namespace_id=? AND page_id=?",
        (namespace, page_id),
    ).fetchone()[0]


def _interval(start, end):
    start = timestamp(start) if start is not None else None
    end = timestamp(end) if end is not None else None
    if start is not None and end is not None and start > end:
        raise InvalidRequest()
    return start, end


class HistoryMixin:
    def enable_history(
        self, *, quiesced: bool = False, recorded_at: str | datetime
    ) -> None:
        """Explicit, transactional v1 migration; stop all old writers first."""
        from .sqlite import _entry

        if quiesced is not True:
            raise InvalidRequest()
        recorded = timestamp(recorded_at)
        if self.schema_version == 2:
            return
        with (
            self._connection() as conn,
            self._transaction(conn),
            processing_bound(conn),
        ):
            self._check_schema(conn)
            for statement in HISTORY_SCHEMA:
                conn.execute(statement)
            conn.execute("INSERT INTO history_pages SELECT * FROM pages")
            for namespace, revision in conn.execute(
                "SELECT namespace_id,revision FROM namespaces"
            ):
                self._history_baseline(conn, namespace, revision, recorded)
                for row in conn.execute(
                    "SELECT payload FROM facts WHERE namespace_id=?", (namespace,)
                ):
                    self._history_version(
                        conn, namespace, revision, _entry(row[0]), strict=False
                    )
                self._history_validate_baseline(conn, namespace, revision)
            self._fault("before_history_migration_commit")
            conn.execute("PRAGMA user_version=2")
            self._check_schema(conn)
        self.schema_version = 2

    @staticmethod
    def _history_baseline(conn, namespace, revision, recorded):
        conn.execute(
            "INSERT INTO history_coverage VALUES (?,?,?)",
            (namespace, revision, recorded),
        )
        conn.execute(
            "INSERT INTO history_revisions VALUES (?,?,?,NULL)",
            (namespace, revision, recorded),
        )

    @staticmethod
    def _history_version(conn, namespace, revision, entry, *, strict=True):
        from .sqlite import _page

        archived = entry.model_copy(deep=True)
        source = page_payload(conn, namespace, entry.source_page_id)
        if source is None:
            raise StorageFailure()
        page = _page(source)
        scope_valid = True
        try:
            archived, (key, repo, snapshot, single) = source_identity(
                entry, page, namespace
            )
        except InvalidRequest:
            if strict:
                raise
            scope_valid = False
            key, repo, snapshot, single = entry.id, None, None, True
        start, end = _interval(entry.t_valid, entry.t_invalid)
        conn.execute(
            "INSERT INTO history_versions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                namespace,
                entry.id,
                revision,
                entry.source_page_id,
                key,
                repo,
                snapshot,
                single,
                scope_valid,
                start,
                end,
                entry.status,
                canonical_json(archived.model_dump()),
            ),
        )
        conn.execute(
            "INSERT INTO history_effects SELECT "
            + ",".join(EFFECT_COLUMNS)
            + ",?,? FROM history_versions WHERE namespace_id=? AND memory_id=? AND revision=?",
            (
                digest(canonical_json(archived.model_dump())),
                digest(source),
                namespace,
                entry.id,
                revision,
            ),
        )
        return key, repo, snapshot, single

    def _history_validate_baseline(self, conn, namespace, revision):
        for key, repo, snapshot in conn.execute(
            "SELECT DISTINCT fact_key,repo_id,snapshot_id FROM history_versions WHERE namespace_id=? AND scope_valid=1",
            (namespace,),
        ):
            self._history_check_key(conn, namespace, revision, key, repo, snapshot)

    @staticmethod
    def _history_integrity(conn, namespace, revision):
        # Metadata only: do not hydrate private payloads to diagnose corruption.
        equality = " AND ".join(f"e.{c} IS v.{c}" for c in EFFECT_COLUMNS)
        join = "e.namespace_id=v.namespace_id AND e.memory_id=v.memory_id AND e.revision=v.revision"
        for query in (
            "SELECT 1 FROM history_effects e LEFT JOIN history_versions v ON "
            + join
            + " WHERE e.namespace_id=? AND e.revision<=? AND (v.memory_id IS NULL OR NOT ("
            + equality
            + ")) LIMIT 1",
            "SELECT 1 FROM history_versions v LEFT JOIN history_effects e ON "
            + join
            + " WHERE v.namespace_id=? AND v.revision<=? AND e.memory_id IS NULL LIMIT 1",
        ):
            if conn.execute(query, (namespace, revision)).fetchone() is not None:
                raise StorageFailure()

    def _history_record(self, conn, namespace, revision, now, event_id, changed):
        if self.schema_version != 2:
            return
        from .sqlite import _entry

        recorded = timestamp(now)
        head = conn.execute(
            "SELECT revision,recorded_at FROM history_revisions WHERE namespace_id=? ORDER BY revision DESC LIMIT 1",
            (namespace,),
        ).fetchone()
        if head is None:
            # No earlier observation exists, including for newly created namespaces.
            conn.execute(
                "INSERT INTO history_coverage VALUES (?,?,?)",
                (namespace, revision, recorded),
            )
        elif recorded < head[1] or revision <= head[0]:
            raise InvalidRequest()
        conn.execute(
            "INSERT INTO history_revisions VALUES (?,?,?,?)",
            (namespace, revision, recorded, event_id),
        )
        touched = set()
        for memory_id in changed:
            entry = _entry(
                conn.execute(
                    "SELECT payload FROM facts WHERE namespace_id=? AND memory_id=?",
                    (namespace, memory_id),
                ).fetchone()[0]
            )
            version_identity = self._history_version(
                conn, namespace, revision, entry, strict=entry.status != "deleted"
            )
            touched.add(version_identity[:3])
        with processing_bound(conn):
            for key, repo, snapshot in touched:
                self._history_check_key(conn, namespace, revision, key, repo, snapshot)
        self._fault("after_history_insert")

    @staticmethod
    def _history_check_key(conn, namespace, revision, key, repo, snapshot):
        rows = conn.execute(
            "SELECT v.valid_from,v.valid_to,v.single_valued FROM history_versions v "
            "WHERE namespace_id=? AND fact_key=? AND repo_id IS ? AND snapshot_id IS ? AND revision<=? "
            "AND status!='deleted' AND scope_valid=1 AND NOT EXISTS (SELECT 1 FROM history_versions n WHERE "
            "n.namespace_id=v.namespace_id AND n.memory_id=v.memory_id AND n.revision>v.revision AND n.revision<=?) "
            "ORDER BY valid_from LIMIT 1002",
            (namespace, key, repo, snapshot, revision, revision),
        ).fetchall()
        if len(rows) > 1000:
            raise CapacityExceeded()
        if len({r[2] for r in rows}) > 1:
            raise InvalidRequest()
        if not rows or not rows[0][2]:
            return
        previous_end = None
        previous = False
        for start, end, _ in rows:
            if start is None:
                # Unknown starts cannot prove an overlap or non-overlap. Queries
                # expose incomplete validity instead of assigning a guessed date.
                continue
            if start == end:
                continue
            if previous and (previous_end is None or start < previous_end):
                raise InvalidRequest()
            previous, previous_end = True, end

    def commit_temporal(
        self,
        request: IngestRequest,
        expected_revision: int,
        proposal: Proposal,
        *,
        corrections: tuple[ValidityCorrection, ...] = (),
    ) -> DurableReceipt:
        """Bind exact structured changes to the normal durable operation receipt."""
        request.validated()
        if self.schema_version != 2:
            raise SchemaMismatch()
        if type(corrections) is not tuple or len(corrections) > 100:
            raise InvalidRequest()
        normalized = []
        for correction in corrections:
            if type(correction) is not ValidityCorrection:
                raise InvalidRequest()
            validate_identifier(correction.memory_id)
            start, end = _interval(correction.valid_from, correction.valid_to)
            normalized.append(ValidityCorrection(correction.memory_id, start, end))
        if len({c.memory_id for c in normalized}) != len(normalized):
            raise InvalidRequest()
        try:
            body = canonical_json(
                {
                    "proposal": proposal.model_dump(),
                    "corrections": [asdict(c) for c in normalized],
                },
                max_bytes=524288,
            )
        except AttributeError:
            raise InvalidRequest() from None
        meta = json.loads(request.metadata_json)
        if "_temporal_command" in meta:
            raise InvalidRequest()
        meta["_temporal_command"] = hashlib.sha256(body.encode()).hexdigest()
        bound = IngestRequest.create(
            request.namespace_id,
            request.idempotency_key,
            request.message,
            meta,
            request.user_id,
        )
        return self._commit_proposal(
            bound, expected_revision, proposal, corrections=tuple(normalized)
        )

    @staticmethod
    def _history_identity(conn, namespace, memory_id):
        from .sqlite import _entry

        row = conn.execute(
            "SELECT payload,scope_valid FROM history_versions WHERE namespace_id=? AND memory_id=? ORDER BY revision DESC LIMIT 1",
            (namespace, memory_id),
        ).fetchone()
        if row is None or not row["scope_valid"]:
            raise InvalidRequest()
        return identity(_entry(row["payload"]))

    @staticmethod
    def _history_correct(conn, namespace, revision, corrections, changed, meta):
        from .sqlite import _entry

        for correction in corrections:
            if correction.memory_id in changed:
                raise InvalidRequest()
            row = conn.execute(
                "SELECT payload FROM facts WHERE namespace_id=? AND memory_id=? AND status!='deleted'",
                (namespace, correction.memory_id),
            ).fetchone()
            if row is None:
                raise TargetNotFound()
            entry = _entry(row[0])
            if HistoryMixin._history_identity(conn, namespace, entry.id)[1:3] != (
                meta.get("repo_id"),
                meta.get("snapshot_id"),
            ):
                raise InvalidRequest()
            entry.t_valid, entry.t_invalid = correction.valid_from, correction.valid_to
            conn.execute(
                "UPDATE facts SET payload=?,revision=? WHERE namespace_id=? AND memory_id=?",
                (canonical_json(entry.model_dump()), revision, namespace, entry.id),
            )
            changed.append(entry.id)

    def historical_page(self, namespace_id: str, page_id: str):
        """Trusted-host lookup of an immutable retained source, never ordinary visibility."""
        from .sqlite import _page

        validate_identifier(namespace_id)
        validate_identifier(page_id)
        with self._connection(read_only=True) as conn:
            if self.schema_version != 2:
                return None
            conn.execute("BEGIN")
            with processing_bound(conn):
                self._check_schema(conn, integrity=False)
                payload = page_payload(conn, namespace_id, page_id)
                if payload is None:
                    return None
                page = _page(payload)
                if (
                    page.meta.get("page_id") != page_id
                    or any(
                        key in page.meta and page.meta[key] != namespace_id
                        for key in ("namespace_id",)
                    )
                    or (
                        "namespace" in page.meta
                        and page.meta["namespace"] not in (namespace_id, [namespace_id])
                    )
                ):
                    raise StorageFailure()
                if (
                    conn.execute(
                        "SELECT 1 FROM history_effects WHERE namespace_id=? AND page_id=? AND page_digest!=? LIMIT 1",
                        (namespace_id, page_id, digest(payload)),
                    ).fetchone()
                    is not None
                ):
                    raise StorageFailure()
                return page

    def compact_history(
        self,
        destination: str | Path,
        namespace_id: str,
        *,
        before_revision: int,
        quiesced: bool = False,
        timeout_seconds: float = 10,
    ) -> Path:
        """Trusted offline maintenance: prune old versions in a NEW complete image.

        The source, pages, receipts, outbox and other namespaces are preserved.
        This is history-version retirement, not erasure of source text or backups.
        Caller stops writers through explicit cutover to avoid losing later writes.
        """
        from .sqlite import _revision_bound

        validate_identifier(namespace_id)
        _revision_bound(before_revision)
        if (
            quiesced is not True
            or type(timeout_seconds) not in (int, float)
            or not 0 < timeout_seconds <= 60
        ):
            raise InvalidRequest()
        if self.schema_version != 2:
            raise SchemaMismatch()
        dest = Path(destination).absolute()
        deadline = time.monotonic() + timeout_seconds
        created = False

        def check_deadline():
            if time.monotonic() >= deadline:
                raise CapacityExceeded()

        try:
            with self._connection(read_only=True) as source, processing_bound(source):
                source.execute("BEGIN")
                self._check_schema(source)
                baseline = source.execute(
                    "SELECT revision FROM history_coverage WHERE namespace_id=?",
                    (namespace_id,),
                ).fetchone()
                cutoff = source.execute(
                    "SELECT recorded_at FROM history_revisions WHERE namespace_id=? AND revision=?",
                    (namespace_id, before_revision),
                ).fetchone()
                if baseline is None or cutoff is None or before_revision < baseline[0]:
                    raise InvalidRequest()
                for ns, head in source.execute(
                    "SELECT namespace_id,revision FROM namespaces"
                ):
                    self._history_integrity(source, ns, head)
                with dest.open("xb"):
                    created = True
                target_store = type(self)(dest)
                with (
                    target_store._connection() as target,
                    target_store._transaction(target),
                    processing_bound(target),
                ):
                    for table in (
                        "namespaces",
                        "pages",
                        "facts",
                        "operations",
                        "outbox",
                        "aliases",
                        "imports",
                        "projection",
                        "history_pages",
                        "history_coverage",
                        "history_revisions",
                        "history_versions",
                        "history_effects",
                    ):
                        query, args = f"SELECT * FROM {table}", ()
                        if table in ("history_versions", "history_effects"):
                            query += (
                                " v WHERE namespace_id!=? OR revision>? OR revision=("
                                "SELECT MAX(b.revision) FROM history_versions b WHERE "
                                "b.namespace_id=v.namespace_id AND b.memory_id=v.memory_id AND b.revision<=?)"
                            )
                            args = (namespace_id, before_revision, before_revision)
                        for row in source.execute(query, args):
                            check_deadline()
                            values = tuple(row)
                            if (
                                table == "history_coverage"
                                and row["namespace_id"] == namespace_id
                            ):
                                values = (namespace_id, before_revision, cutoff[0])
                            target.execute(
                                f"INSERT INTO {table} VALUES ({','.join('?' for _ in values)})",
                                values,
                            )
                    check_deadline()
                    target_store._check_schema(target)
                    self._fault("before_history_compact_commit")
                return dest
        except (OSError, sqlite3.Error):
            raise StorageFailure() from None
        finally:
            import sys

            if created and sys.exc_info()[0] is not None:
                try:
                    dest.unlink()
                except OSError:
                    pass

    def snapshot_at(
        self,
        namespace_id,
        *,
        transaction_at=None,
        revision=None,
        valid_at=None,
        limit=100,
        repo_id=None,
        snapshot_id=None,
        eligible_at=None,
    ):
        """One consistent read, bounded acquisition, explicit missing coverage."""
        from .sqlite import _bound, _entry, _revision_bound

        validate_identifier(namespace_id)
        _bound(limit)
        if transaction_at is not None and revision is not None:
            raise InvalidRequest()
        if revision is not None:
            _revision_bound(revision)
        transaction = timestamp(transaction_at) if transaction_at is not None else None
        valid = timestamp(valid_at) if valid_at is not None else None
        eligible = timestamp(eligible_at) if eligible_at is not None else None
        if (repo_id is None) != (snapshot_id is None):
            raise InvalidRequest()
        if repo_id is not None:
            validate_identifier(repo_id)
            validate_identifier(snapshot_id)
        with self._connection(read_only=True) as conn:
            conn.execute("BEGIN")
            if self.schema_version != 2:
                return HistoricalSnapshot(
                    namespace_id,
                    None,
                    None,
                    None,
                    None,
                    "unavailable",
                    False,
                    False,
                    False,
                    (),
                )
            with processing_bound(conn):
                self._check_schema(conn, integrity=False)
                return self._snapshot_history(
                    conn,
                    namespace_id,
                    transaction,
                    revision,
                    valid,
                    eligible,
                    limit,
                    repo_id,
                    snapshot_id,
                    _entry,
                )

    def _snapshot_history(
        self,
        conn,
        namespace,
        transaction,
        revision,
        valid,
        eligible,
        limit,
        repo,
        snapshot,
        decode,
    ):
        baseline = conn.execute(
            "SELECT revision,recorded_at FROM history_coverage WHERE namespace_id=?",
            (namespace,),
        ).fetchone()
        base_rev, base_time = baseline if baseline else (None, None)
        head = self._revision(conn, namespace)
        if revision is not None and revision > head:
            raise InvalidRequest()
        selected = conn.execute(
            "SELECT revision,recorded_at FROM history_revisions WHERE namespace_id=? AND revision<=? "
            "AND (? IS NULL OR recorded_at<=?) ORDER BY revision DESC LIMIT 1",
            (
                namespace,
                head if revision is None else revision,
                transaction,
                transaction,
            ),
        ).fetchone()
        if (
            selected is None
            or base_rev is None
            or selected[0] < base_rev
            or (transaction is not None and transaction < base_time)
            or (revision is not None and selected[0] != revision)
        ):
            return HistoricalSnapshot(
                namespace,
                None,
                None,
                base_rev,
                base_time,
                "unavailable",
                False,
                False,
                False,
                (),
            )
        at, recorded = selected
        self._history_integrity(conn, namespace, at)
        where = (
            "v.namespace_id=? AND v.repo_id IS ? AND v.snapshot_id IS ? AND v.revision<=? AND "
            "NOT EXISTS (SELECT 1 FROM history_versions n WHERE n.namespace_id=v.namespace_id "
            "AND n.memory_id=v.memory_id AND n.revision>v.revision AND n.revision<=?) AND v.status!='deleted' AND v.scope_valid=1"
        )
        args = (namespace, repo, snapshot, at, at)
        unknown_scope = (
            conn.execute(
                "SELECT 1 FROM history_versions v WHERE namespace_id=? AND revision<=? AND scope_valid=0 AND status!='deleted' "
                "AND NOT EXISTS (SELECT 1 FROM history_versions n WHERE n.namespace_id=v.namespace_id AND n.memory_id=v.memory_id AND n.revision>v.revision AND n.revision<=?) LIMIT 1",
                (namespace, at, at),
            ).fetchone()
            is not None
        )
        unknown_eligibility = False
        # Unknown starts are counted within exact scope before validity filtering.
        unknown = (
            valid is not None
            and conn.execute(
                f"SELECT 1 FROM history_versions v WHERE {where} AND valid_from IS NULL LIMIT 1",
                args,
            ).fetchone()
            is not None
        )
        if valid is not None:
            where += " AND valid_from IS NOT NULL AND valid_from<=? AND (valid_to IS NULL OR ?<valid_to)"
            args += (valid, valid)
        query = f"SELECT v.* FROM history_versions v WHERE {where} ORDER BY v.memory_id LIMIT ?"
        # Size-only acquisition first in this same read transaction.
        sizes = conn.execute(
            f"SELECT length(CAST(q.payload AS BLOB)), COALESCE((SELECT length(CAST(p.payload AS BLOB)) "
            f"FROM history_pages p WHERE p.namespace_id=q.namespace_id AND p.page_id=q.page_id),0) FROM ({query}) q",
            (*args, limit + 1),
        ).fetchall()
        if (
            any(r[0] > 262144 or r[1] > 1048576 for r in sizes)
            or sum(sum(r) for r in sizes) > MAX_HISTORY_BYTES
        ):
            raise CapacityExceeded()
        facts = []
        rows = conn.execute(query, (*args, limit + 1)).fetchall()
        checked = set()
        for row in rows[:limit]:
            effect = conn.execute(
                "SELECT payload_digest,page_digest FROM history_effects WHERE namespace_id=? AND memory_id=? AND revision=?",
                (namespace, row["memory_id"], row["revision"]),
            ).fetchone()
            if digest(row["payload"]) != effect[0]:
                raise StorageFailure()
            source = page_payload(conn, namespace, row["page_id"])
            if source is None or digest(source) != effect[1]:
                raise StorageFailure()
            from .sqlite import _page

            page = _page(source)
            key_scope = (row["fact_key"], row["repo_id"], row["snapshot_id"])
            if key_scope not in checked:
                self._history_check_key(conn, namespace, at, *key_scope)
                checked.add(key_scope)
            entry = decode(row["payload"])
            try:
                _, resolved_identity = source_identity(entry, page, namespace)
            except InvalidRequest:
                raise StorageFailure() from None
            if (
                entry.status != row["status"]
                or row["single_valued"] not in (0, 1)
                or entry.id != row["memory_id"]
                or entry.source_page_id != row["page_id"]
                or resolved_identity
                != (
                    row["fact_key"],
                    row["repo_id"],
                    row["snapshot_id"],
                    bool(row["single_valued"]),
                )
                or _interval(entry.t_valid, entry.t_invalid)
                != (row["valid_from"], row["valid_to"])
            ):
                raise StorageFailure()
            # t_expired on retired facts is retirement time, NOT historical TTL.
            # Explicit eligibility metadata applies independently of retention.
            expires = entry.meta.get("expires_at")
            if expires is None and entry.status == "expired":
                expires = entry.t_expired
            if eligible is not None:
                try:
                    ends = [timestamp(expires)] if expires is not None else []
                    if "expires_at" in page.meta:
                        ends.append(timestamp(page.meta["expires_at"]))
                    for metadata in (entry.meta, page.meta):
                        if "ttl_seconds" not in metadata:
                            continue
                        ttl = metadata["ttl_seconds"]
                        if (
                            type(ttl) not in (int, float)
                            or not math.isfinite(ttl)
                            or ttl < 0
                        ):
                            raise InvalidRequest()
                        origin = datetime.fromisoformat(timestamp(entry.t_created))
                        ends.append(timestamp(origin + timedelta(seconds=ttl)))
                except (InvalidRequest, OverflowError):
                    unknown_eligibility = True
                    continue
                if any(eligible >= end for end in ends):
                    continue
            facts.append(
                HistoricalFact(
                    entry,
                    row["revision"],
                    row["fact_key"],
                    row["repo_id"],
                    row["snapshot_id"],
                    bool(row["single_valued"]),
                    row["page_id"],
                )
            )
        truncated = len(rows) > limit
        return HistoricalSnapshot(
            namespace,
            at,
            recorded,
            base_rev,
            base_time,
            "available",
            not truncated
            and not unknown
            and not unknown_scope
            and not unknown_eligibility,
            truncated,
            unknown,
            tuple(facts),
            unknown_scope,
            unknown_eligibility,
        )

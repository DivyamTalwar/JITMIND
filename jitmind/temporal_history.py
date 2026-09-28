"""Append-only, local SQLite storage for complete strict temporal perspectives.

This module owns a dedicated database, never mutable legacy fact rows. Generation
and provider calls belong outside storage. See docs/durable-temporal-history.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

from jitmind.scope import ScopeAuthority, ScopeContext, ScopeDenied, _identifier
from jitmind.scoped_temporal import (
    StrictTemporalOracle,
    TemporalConflict,
    TemporalFact,
    TemporalPerspective,
    utc,
)

MAX_PAYLOAD_BYTES = 1_048_576
SCHEMA_VERSION = 1
# Deliberately do not use database-global user_version/application_id: ownership
# is explicit in the prefixed metadata. v1 still requires a dedicated database.
SCHEMA = (
    "CREATE TABLE jth_meta (singleton INTEGER PRIMARY KEY CHECK(singleton=1), version INTEGER NOT NULL CHECK(version=1))",
    "CREATE TABLE jth_perspectives (namespace_id TEXT NOT NULL CHECK(typeof(namespace_id)='text'), revision INTEGER NOT NULL CHECK(typeof(revision)='integer' AND revision>0), recorded_at TEXT NOT NULL CHECK(typeof(recorded_at)='text' AND length(recorded_at)=27 AND substr(recorded_at,27,1)='Z'), payload TEXT NOT NULL CHECK(typeof(payload)='text' AND length(CAST(payload AS BLOB))<=1048576), digest TEXT NOT NULL CHECK(length(digest)=64), PRIMARY KEY(namespace_id,revision), UNIQUE(namespace_id,recorded_at))",
    "CREATE TABLE jth_operations (namespace_id TEXT NOT NULL, idempotency_key TEXT NOT NULL CHECK(typeof(idempotency_key)='text'), operation_id TEXT NOT NULL UNIQUE CHECK(length(operation_id)=36), digest TEXT NOT NULL CHECK(length(digest)=64), revision INTEGER NOT NULL, PRIMARY KEY(namespace_id,idempotency_key), UNIQUE(namespace_id,revision), FOREIGN KEY(namespace_id,revision) REFERENCES jth_perspectives(namespace_id,revision))",
    "CREATE TRIGGER jth_append BEFORE INSERT ON jth_perspectives BEGIN SELECT CASE WHEN NEW.revision != COALESCE((SELECT MAX(revision) FROM jth_perspectives WHERE namespace_id=NEW.namespace_id),0)+1 OR NEW.recorded_at <= COALESCE((SELECT MAX(recorded_at) FROM jth_perspectives WHERE namespace_id=NEW.namespace_id),'') THEN RAISE(ABORT,'temporal conflict') END; END",
    "CREATE TRIGGER jth_operation_digest BEFORE INSERT ON jth_operations BEGIN SELECT CASE WHEN NEW.digest != (SELECT digest FROM jth_perspectives WHERE namespace_id=NEW.namespace_id AND revision=NEW.revision) THEN RAISE(ABORT,'temporal conflict') END; END",
) + tuple(
    f"CREATE TRIGGER {table}_{action.lower()} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'immutable temporal history'); END"
    for table in ("jth_meta", "jth_perspectives", "jth_operations")
    for action in ("UPDATE", "DELETE")
)


class TemporalStorageError(RuntimeError):
    def __init__(self):
        super().__init__("Temporal storage unavailable")


class TemporalSchemaError(TemporalStorageError):
    def __init__(self):
        RuntimeError.__init__(self, "Unsupported or corrupt temporal database")


class TemporalCapacityError(TemporalStorageError):
    def __init__(self):
        RuntimeError.__init__(self, "Temporal acquisition limit exceeded")


class TemporalAcknowledgementUncertain(TemporalStorageError):
    def __init__(self):
        RuntimeError.__init__(
            self, "Temporal acknowledgement uncertain; reconcile the same key and body"
        )


class TemporalIdempotencyConflict(TemporalConflict):
    pass


@dataclass(frozen=True)
class PublicationReceipt:
    operation_id: str
    payload_digest: str
    revision: int


@dataclass(frozen=True)
class PerspectivePage:
    perspectives: tuple[TemporalPerspective, ...]
    next_revision: int
    head_revision: int
    has_more: bool


def _integer(value, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise TemporalConflict()
    return value


def _timestamp(value):
    if type(value) not in (str, datetime):
        raise TemporalConflict()
    if type(value) is str and len(value) > 128:
        raise TemporalConflict()
    return utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _digest(payload):
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical(perspective, max_bytes, max_facts):
    """Reconstruct even frozen dataclasses: object.__setattr__ can bypass them."""
    try:
        if (
            type(perspective) is not TemporalPerspective
            or type(perspective.facts) is not tuple
        ):
            raise TemporalConflict()
        if len(perspective.facts) > max_facts:
            raise TemporalCapacityError()
        recorded = _timestamp(perspective.recorded_at)
        facts = []
        encoded = []
        size = len('{"facts":[],"recorded_at":' + json.dumps(recorded) + "}")
        if size > max_bytes:
            raise TemporalCapacityError()
        for fact in perspective.facts:
            if type(fact) is not TemporalFact:
                raise TemporalConflict()
            values = {
                field.name: getattr(fact, field.name) for field in fields(TemporalFact)
            }
            if type(values["content"]) is not str:
                raise TemporalConflict()
            if len(values["content"]) > max_bytes:
                raise TemporalCapacityError()
            for name in ("valid_from", "valid_to", "expires_at"):
                if values[name] is not None:
                    values[name] = _timestamp(values[name])
            rebuilt = TemporalFact(**values)
            part = json.dumps(
                values,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            size += len(part.encode("utf-8")) + bool(encoded)
            if size > max_bytes:
                raise TemporalCapacityError()
            facts.append(rebuilt)
            encoded.append(part)
        validated = TemporalPerspective(recorded, tuple(facts))
        payload = (
            '{"facts":['
            + ",".join(encoded)
            + '],"recorded_at":'
            + json.dumps(recorded)
            + "}"
        )
        if len(payload.encode("utf-8")) > max_bytes:
            raise TemporalCapacityError()
        return validated, payload, _digest(payload)
    except (
        ValueError,
        TypeError,
        AttributeError,
        UnicodeError,
        OverflowError,
        RecursionError,
        ScopeDenied,
    ):
        raise TemporalConflict() from None


def _no_number(_):
    raise ValueError


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _decode(payload, recorded_at, digest, max_bytes, max_facts):
    try:
        if type(payload) is not str or len(payload.encode("utf-8")) > max_bytes:
            raise TemporalCapacityError()
        data = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_int=_no_number,
            parse_float=_no_number,
            parse_constant=_no_number,
        )
        if (
            type(data) is not dict
            or set(data) != {"recorded_at", "facts"}
            or type(data["facts"]) is not list
        ):
            raise ValueError
        if len(data["facts"]) > max_facts:
            raise TemporalCapacityError()
        if any(
            type(f) is not dict or set(f) != {v.name for v in fields(TemporalFact)}
            for f in data["facts"]
        ):
            raise ValueError
        item = TemporalPerspective(
            data["recorded_at"], tuple(TemporalFact(**f) for f in data["facts"])
        )
        validated, canonical, actual_digest = _canonical(item, max_bytes, max_facts)
        if (
            payload != canonical
            or digest != actual_digest
            or recorded_at != _timestamp(validated.recorded_at)
        ):
            raise ValueError
        return validated
    except TemporalCapacityError:
        raise
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError, ScopeDenied):
        raise TemporalStorageError() from None


class SQLiteTemporalHistory:
    """Operation-local connections; DELETE/FULL only, no journal mode changes.

    Constructor creates only an absent file; existing empty/foreign/future/corrupt
    databases are refused. Bounds are acquisition limits, not retention policies.
    fault_hook is a trusted test-only injection, never a provider/generator.
    """

    def __init__(
        self,
        path,
        *,
        busy_timeout_ms=250,
        max_payload_bytes=MAX_PAYLOAD_BYTES,
        max_facts=10_000,
        max_read_count=1000,
        max_read_bytes=16_777_216,
        journal_mode="DELETE",
        fault_hook=None,
    ):
        _integer(busy_timeout_ms, 0, 5000)
        _integer(max_payload_bytes, 1, MAX_PAYLOAD_BYTES)
        _integer(max_facts, 1, 100_000)
        _integer(max_read_count, 1, 100_000)
        _integer(max_read_bytes, 1, 67_108_864)
        if journal_mode != "DELETE":
            raise TemporalSchemaError()
        self.path = Path(path).absolute()
        self.busy_timeout_ms = busy_timeout_ms
        self.max_payload_bytes = max_payload_bytes
        self.max_facts = max_facts
        self.max_read_count = max_read_count
        self.max_read_bytes = max_read_bytes
        self._fault_hook = fault_hook
        created = False
        try:
            try:
                with self.path.open("xb"):
                    created = True
            except FileExistsError:
                pass
            with self._connection(check=False) as conn:
                if created:
                    with self._transaction(conn, write=True):
                        for statement in SCHEMA:
                            conn.execute(statement)
                        conn.execute("INSERT INTO jth_meta VALUES (1,1)")
                with self._transaction(conn):
                    self._audit(conn, time.monotonic() + 10)
        except OSError:
            raise TemporalStorageError() from None

    @staticmethod
    def _schema(conn, *, integrity=False):
        statements = {
            row[0]
            for row in conn.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"
            )
        }
        if statements != set(SCHEMA):
            raise TemporalSchemaError()
        if conn.execute("SELECT singleton,version FROM jth_meta").fetchall() != [
            (1, SCHEMA_VERSION)
        ]:
            raise TemporalSchemaError()
        if (
            conn.execute("PRAGMA user_version").fetchone()[0] != 0
            or conn.execute("PRAGMA application_id").fetchone()[0] != 0
        ):
            raise TemporalSchemaError()
        if integrity and (
            conn.execute("PRAGMA quick_check").fetchone()[0] != "ok"
            or conn.execute("PRAGMA foreign_key_check").fetchone() is not None
        ):
            raise TemporalSchemaError()

    @classmethod
    def _audit(cls, conn, deadline):
        """Streaming integrity audit, never materialize an unbounded history."""
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        try:
            cls._schema(conn, integrity=True)
            previous_namespace, previous_revision, previous_time = None, 0, None
            for namespace, revision, recorded, size in conn.execute(
                "SELECT namespace_id,revision,recorded_at,length(CAST(payload AS BLOB)) FROM jth_perspectives ORDER BY namespace_id,revision"
            ):
                if time.monotonic() >= deadline:
                    raise TemporalStorageError()
                try:
                    _identifier(namespace)
                except ScopeDenied:
                    raise TemporalStorageError() from None
                if namespace != previous_namespace:
                    previous_revision, previous_time = 0, None
                if revision != previous_revision + 1 or (
                    previous_time is not None and recorded <= previous_time
                ):
                    raise TemporalStorageError()
                if size > MAX_PAYLOAD_BYTES:
                    raise TemporalCapacityError()
                payload, digest = conn.execute(
                    "SELECT payload,digest FROM jth_perspectives WHERE namespace_id=? AND revision=?",
                    (namespace, revision),
                ).fetchone()
                _decode(payload, recorded, digest, MAX_PAYLOAD_BYTES, 100_000)
                previous_namespace, previous_revision, previous_time = (
                    namespace,
                    revision,
                    recorded,
                )
            for namespace, key, operation_id, digest, revision, parent in conn.execute(
                "SELECT o.namespace_id,o.idempotency_key,o.operation_id,o.digest,o.revision,p.digest FROM jth_operations o JOIN jth_perspectives p USING(namespace_id,revision)"
            ):
                if time.monotonic() >= deadline:
                    raise TemporalStorageError()
                try:
                    _identifier(key)
                    if str(uuid.UUID(operation_id)) != operation_id or digest != parent:
                        raise ValueError
                except (ValueError, AttributeError, ScopeDenied):
                    raise TemporalStorageError() from None
        finally:
            conn.set_progress_handler(None, 0)

    @contextmanager
    def _connection(self, *, check=True, read_only=False):
        conn = None
        try:
            conn = sqlite3.connect(
                self.path.as_uri() + ("?mode=ro" if read_only else "?mode=rw"),
                uri=True,
                isolation_level=None,
                timeout=self.busy_timeout_ms / 1000,
            )
            # Read mode before any persistent changes. WAL is unsupported even on
            # patched runtimes, avoiding implicit conversion of an existing DB.
            if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "delete":
                raise TemporalSchemaError()
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=FULL")
            if (
                conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1
                or conn.execute("PRAGMA synchronous").fetchone()[0] != 2
            ):
                raise TemporalStorageError()
            if check:
                self._schema(conn)
            yield conn
        except ScopeDenied:
            raise
        except (sqlite3.Error, OSError):
            raise TemporalStorageError() from None
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    @contextmanager
    def _transaction(conn, *, write=False):
        conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def _fault(self, stage):
        if self._fault_hook is not None:
            try:
                self._fault_hook(stage)
            except Exception:  # noqa: BLE001 - sanitize trusted fault injection
                raise TemporalStorageError() from None

    @staticmethod
    def _status(conn, namespace_id):
        count, size, first, head, latest = conn.execute(
            "SELECT COUNT(*),COALESCE(SUM(length(CAST(payload AS BLOB))),0),MIN(revision),COALESCE(MAX(revision),0),MAX(recorded_at) FROM jth_perspectives WHERE namespace_id=?",
            (namespace_id,),
        ).fetchone()
        if count and (first != 1 or head != count):
            raise TemporalStorageError()
        return {
            "revision": head,
            "perspective_count": count,
            "payload_bytes": size,
            "latest_recorded_at": latest,
        }

    def status(self, namespace_id: str) -> dict:
        _identifier(namespace_id)
        with self._connection(read_only=True) as conn, self._transaction(conn):
            return self._status(conn, namespace_id)

    def diagnostics(self) -> dict:
        with self._connection(read_only=True) as conn:
            return {
                "sqlite_version": conn.execute("SELECT sqlite_version()").fetchone()[0],
                "sqlite_source_id": conn.execute(
                    "SELECT sqlite_source_id()"
                ).fetchone()[0],
                "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
                "synchronous": conn.execute("PRAGMA synchronous").fetchone()[0],
                "foreign_keys": conn.execute("PRAGMA foreign_keys").fetchone()[0],
                "busy_timeout_ms": conn.execute("PRAGMA busy_timeout").fetchone()[0],
                "schema_version": SCHEMA_VERSION,
            }

    def _read(self, conn, namespace, after, limit, head):
        # Aggregate before materializing payloads, in the SAME read transaction.
        count, size, largest = conn.execute(
            "SELECT COUNT(*),COALESCE(SUM(n),0),COALESCE(MAX(n),0) FROM (SELECT length(CAST(payload AS BLOB)) n FROM jth_perspectives WHERE namespace_id=? AND revision>? AND revision<=? ORDER BY revision LIMIT ?)",
            (namespace, after, head, limit),
        ).fetchone()
        if (
            count > self.max_read_count
            or size > self.max_read_bytes
            or largest > self.max_payload_bytes
        ):
            raise TemporalCapacityError()
        rows = conn.execute(
            "SELECT revision,payload,recorded_at,digest FROM jth_perspectives WHERE namespace_id=? AND revision>? AND revision<=? ORDER BY revision LIMIT ?",
            (namespace, after, head, limit),
        ).fetchall()
        result = []
        previous_time = None
        if after:
            row = conn.execute(
                "SELECT recorded_at FROM jth_perspectives WHERE namespace_id=? AND revision=?",
                (namespace, after),
            ).fetchone()
            if row is None:
                raise TemporalConflict()
            previous_time = row[0]
        for expected, (revision, payload, recorded, digest) in enumerate(
            rows, after + 1
        ):
            if revision != expected or (
                previous_time is not None and recorded <= previous_time
            ):
                raise TemporalStorageError()
            result.append(
                _decode(
                    payload, recorded, digest, self.max_payload_bytes, self.max_facts
                )
            )
            previous_time = recorded
        return tuple(result)

    def read_perspectives(self, namespace_id: str) -> tuple[TemporalPerspective, ...]:
        _identifier(namespace_id)
        with self._connection(read_only=True) as conn, self._transaction(conn):
            status = self._status(conn, namespace_id)
            if (
                status["perspective_count"] > self.max_read_count
                or status["payload_bytes"] > self.max_read_bytes
            ):
                raise TemporalCapacityError()
            return self._read(
                conn, namespace_id, 0, self.max_read_count, status["revision"]
            )

    def read_page(
        self,
        namespace_id: str,
        *,
        after_revision: int = 0,
        limit: int = 100,
        through_revision: int | None = None,
    ) -> PerspectivePage:
        _identifier(namespace_id)
        _integer(after_revision)
        _integer(limit, 1, self.max_read_count)
        if through_revision is not None:
            _integer(through_revision)
        with self._connection(read_only=True) as conn, self._transaction(conn):
            head = self._status(conn, namespace_id)["revision"]
            through = head if through_revision is None else through_revision
            if through > head or after_revision > through:
                raise TemporalConflict()
            items = self._read(conn, namespace_id, after_revision, limit, through)
            next_revision = after_revision + len(items)
            return PerspectivePage(
                items, next_revision, through, next_revision < through
            )

    def _receipt(self, conn, namespace, key, digest):
        row = conn.execute(
            "SELECT o.operation_id,o.digest,o.revision,p.digest FROM jth_operations o JOIN jth_perspectives p USING(namespace_id,revision) WHERE o.namespace_id=? AND o.idempotency_key=?",
            (namespace, key),
        ).fetchone()
        if row is None:
            return None
        operation_id, stored, revision, parent = row
        try:
            if str(uuid.UUID(operation_id)) != operation_id or stored != parent:
                raise ValueError
            _integer(revision, 1)
        except (ValueError, TypeError, AttributeError):
            raise TemporalStorageError() from None
        if stored != digest:
            raise TemporalIdempotencyConflict()
        size = conn.execute(
            "SELECT length(CAST(payload AS BLOB)) FROM jth_perspectives WHERE namespace_id=? AND revision=?",
            (namespace, revision),
        ).fetchone()[0]
        if size > self.max_payload_bytes or size > self.max_read_bytes:
            raise TemporalCapacityError()
        payload, recorded = conn.execute(
            "SELECT payload,recorded_at FROM jth_perspectives WHERE namespace_id=? AND revision=?",
            (namespace, revision),
        ).fetchone()
        _decode(payload, recorded, stored, self.max_payload_bytes, self.max_facts)
        return PublicationReceipt(operation_id, stored, revision)

    def append_perspective(
        self,
        namespace_id: str,
        expected_revision: int,
        perspective: TemporalPerspective,
    ) -> None:
        _identifier(namespace_id)
        _integer(expected_revision)
        item, payload, digest = _canonical(
            perspective, self.max_payload_bytes, self.max_facts
        )
        self._append(namespace_id, expected_revision, item, payload, digest)

    def _append(
        self, namespace, expected, item, payload, digest, *, key=None, authorize=None
    ):
        # Validation/canonicalization and UUID generation are outside the writer lock.
        operation_id = str(uuid.uuid4()) if key is not None else None
        if authorize:
            authorize()
        commit_started = False
        try:
            with self._connection() as conn, self._transaction(conn, write=True):
                if authorize:
                    authorize()
                receipt = (
                    self._receipt(conn, namespace, key, digest)
                    if key is not None
                    else None
                )
                if receipt is not None:
                    if authorize:
                        authorize()
                    return receipt
                # Indexed head lookup keeps writer work independent of archive size.
                head = conn.execute(
                    "SELECT revision,recorded_at FROM jth_perspectives WHERE namespace_id=? ORDER BY revision DESC LIMIT 1",
                    (namespace,),
                ).fetchone()
                revision, latest = head if head is not None else (0, None)
                recorded = _timestamp(item.recorded_at)
                if (expected is not None and expected != revision) or (
                    latest is not None and recorded <= latest
                ):
                    raise TemporalConflict()
                _integer(revision + 1, 1)
                conn.execute(
                    "INSERT INTO jth_perspectives VALUES (?,?,?,?,?)",
                    (namespace, revision + 1, recorded, payload, digest),
                )
                receipt = None
                if key is not None:
                    receipt = PublicationReceipt(operation_id, digest, revision + 1)
                    conn.execute(
                        "INSERT INTO jth_operations VALUES (?,?,?,?,?)",
                        (namespace, key, operation_id, digest, revision + 1),
                    )
                self._fault("before_commit")
                # Last authorization / dispatch boundary. Revocation after this
                # point cannot retroactively undo an accepted COMMIT.
                if authorize:
                    authorize()
                commit_started = True
            self._fault("after_commit")
        except TemporalStorageError:
            if not commit_started or key is None:
                raise
            # Never repeat effects after an uncertain commit. Reconcile using a
            # fresh read connection; absence means uncertainty, not permission to
            # auto-retry. The caller may retry the SAME key/body later.
            if authorize:
                authorize()
            try:
                with self._connection(read_only=True) as conn, self._transaction(conn):
                    receipt = self._receipt(conn, namespace, key, digest)
            except TemporalStorageError:
                raise TemporalAcknowledgementUncertain() from None
            if receipt is None:
                raise TemporalAcknowledgementUncertain() from None
        if authorize:
            authorize()
        return receipt

    def backup(self, destination: str | Path, *, timeout_seconds: float = 10) -> Path:
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 60
        ):
            raise TemporalConflict()
        dest = Path(destination).absolute()
        created = False
        deadline = time.monotonic() + timeout_seconds

        def progress(status, remaining, total):
            if time.monotonic() >= deadline:
                raise TemporalStorageError()

        try:
            with dest.open("xb"):
                created = True
            with self._connection(read_only=True) as source:
                target = sqlite3.connect(dest, isolation_level=None)
                try:
                    target.execute("PRAGMA synchronous=FULL")
                    source.backup(target, pages=128, progress=progress, sleep=0.01)
                    self._audit(target, deadline)
                finally:
                    target.close()
            return dest
        except (OSError, sqlite3.Error):
            raise TemporalStorageError() from None
        finally:
            # Success keeps the exclusive destination; errors remove only our file.
            # sys.exc_info avoids masking an original storage error with cleanup.
            import sys

            if created and sys.exc_info()[0] is not None:
                try:
                    dest.unlink()
                except OSError:
                    pass

    @classmethod
    def restore(
        cls,
        backup_path: str | Path,
        destination: str | Path,
        *,
        timeout_seconds: float = 10,
        **kwargs,
    ) -> SQLiteTemporalHistory:
        """Read-only source, SQLite backup API, exclusive NEW destination."""
        # Bypass the constructor only for the read-only source; it must not create
        # a missing backup or mutate a future/corrupt source database.
        source = object.__new__(cls)
        source.path = Path(backup_path).absolute()
        source.busy_timeout_ms = 250
        source.backup(destination, timeout_seconds=timeout_seconds)
        return cls(destination, **kwargs)


class DurableTemporalOracle:
    """Recommended acknowledgement API; strict queries retain the J03 semantics."""

    def __init__(self, authority: ScopeAuthority, backend: SQLiteTemporalHistory):
        if (
            type(authority) is not ScopeAuthority
            or type(backend) is not SQLiteTemporalHistory
        ):
            raise TemporalConflict()
        self.authority = authority
        self.backend = backend

    def query_as_of(
        self,
        scope: ScopeContext,
        valid_at,
        *,
        transaction_at=None,
        eligible_at=None,
        snapshots=(),
    ) -> tuple[TemporalFact, ...]:
        return StrictTemporalOracle(self.authority, self.backend).query_as_of(
            scope,
            valid_at,
            transaction_at=transaction_at,
            eligible_at=eligible_at,
            snapshots=snapshots,
        )

    def publish(
        self,
        scope: ScopeContext,
        perspective: TemporalPerspective,
        idempotency_key: str,
        expected_revision: int | None = None,
    ) -> PublicationReceipt:
        self.authority.require(scope)
        _identifier(idempotency_key)
        if expected_revision is not None:
            _integer(expected_revision)
        item, payload, digest = _canonical(
            perspective, self.backend.max_payload_bytes, self.backend.max_facts
        )
        repos = frozenset(fact.repo_id for fact in item.facts)

        def authorize():
            self.authority.require(scope)
            for repo in repos:
                self.authority.require(scope, repo)
            self.authority.require(scope)

        authorize()
        receipt = self.backend._append(
            scope.namespace_id,
            expected_revision,
            item,
            payload,
            digest,
            key=idempotency_key,
            authorize=authorize,
        )
        authorize()
        return receipt

"""Independent consumer ledger and explicit host administration."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from jitmind.scope import ScopeDenied
from jitmind.storage.models import canonical_json, validate_identifier

from .models import ProjectionError, ProjectionEvent, PurgePlan, integer, selection
from .source import digest

COMMON_SCHEMA = (
    "CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE states (namespace TEXT PRIMARY KEY, selection TEXT NOT NULL, baseline INTEGER NOT NULL, watermark INTEGER NOT NULL, observed INTEGER NOT NULL, anchor TEXT)",
    "CREATE TABLE receipts (namespace TEXT NOT NULL, revision INTEGER NOT NULL, event TEXT NOT NULL, state TEXT NOT NULL, attempts INTEGER NOT NULL, error TEXT, PRIMARY KEY(namespace,revision))",
    "CREATE TABLE facts (namespace TEXT NOT NULL, fact TEXT NOT NULL, revision INTEGER NOT NULL, status TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(namespace,fact))",
    "CREATE TABLE purges (namespace TEXT NOT NULL, fact TEXT NOT NULL, revision INTEGER NOT NULL, PRIMARY KEY(namespace,fact))",
)


def event_json(event):
    return canonical_json(asdict(event) | {"memory_ids": list(event.memory_ids)})


class SQLiteProjection:
    kind = "base"
    app_id = 0
    extra_schema = ()

    def __init__(self, path, *, source_id, identity, fault_hook=None):
        validate_identifier(source_id)
        self.path = Path(path).absolute()
        self.source_id = source_id
        self.identity = canonical_json(identity)
        self.fault_hook = fault_hook
        self.schema = COMMON_SCHEMA + self.extra_schema
        with self._connect(initialize=True) as conn:
            empty = not conn.execute("SELECT 1 FROM sqlite_master").fetchone()
            if empty:
                if (
                    conn.execute("PRAGMA application_id").fetchone()[0]
                    or conn.execute("PRAGMA user_version").fetchone()[0]
                ):
                    raise ProjectionError("schema_mismatch")
                conn.execute("PRAGMA journal_mode=DELETE")
                with conn:
                    for statement in self.schema:
                        conn.execute(statement)
                    conn.execute(f"PRAGMA application_id={self.app_id}")
                    conn.execute("PRAGMA user_version=1")
                    conn.executemany(
                        "INSERT INTO config VALUES (?,?)",
                        (
                            ("identity", self.identity),
                            ("source_id", source_id),
                            ("enabled", "true"),
                            ("generation", "0"),
                        ),
                    )
            self._validate(conn)

    def _validate(self, conn):
        statements = {
            r[0]
            for r in conn.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL")
        }
        if (
            conn.execute("PRAGMA application_id").fetchone()[0] != self.app_id
            or conn.execute("PRAGMA user_version").fetchone()[0] != 1
            or statements != set(self.schema)
        ):
            raise ProjectionError("schema_mismatch")
        config = dict(conn.execute("SELECT key,value FROM config"))
        if (
            config.get("identity") != self.identity
            or config.get("source_id") != self.source_id
        ):
            raise ProjectionError("identity_mismatch")
        if (
            conn.execute("PRAGMA quick_check").fetchone()[0] != "ok"
            or conn.execute("PRAGMA journal_mode").fetchone()[0] != "delete"
        ):
            raise ProjectionError("storage_unavailable")

    @contextmanager
    def _connect(self, *, initialize=False, write=False):
        conn = None
        try:
            mode = "rwc" if initialize else "rw" if write else "ro"
            conn = sqlite3.connect(
                self.path.as_uri() + "?mode=" + mode, uri=True, timeout=0.25
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA synchronous=FULL")
            if not (write or initialize):
                conn.execute("PRAGMA query_only=ON")
            deadline = time.monotonic() + 3
            ticks = 0

            def budget():
                nonlocal ticks
                ticks += 1
                return ticks > 10000 or time.monotonic() > deadline

            conn.set_progress_handler(budget, 1000)
            if not initialize:
                self._validate(conn)
            yield conn
        except (ProjectionError, ScopeDenied):
            raise
        except sqlite3.Error:
            raise ProjectionError("storage_unavailable") from None
        finally:
            if conn is not None:
                conn.close()

    def _fault(self, stage):
        if self.fault_hook:
            try:
                self.fault_hook(stage)
            except Exception:  # noqa: BLE001 - trusted callback error sanitization
                raise ProjectionError("callback_failed") from None

    @staticmethod
    def _generation(conn):
        value = conn.execute(
            "SELECT value FROM config WHERE key='generation'"
        ).fetchone()
        try:
            parsed = int(value[0])
            if str(parsed) != value[0]:
                raise ValueError
            return integer(parsed)
        except (ValueError, TypeError):
            raise ProjectionError("invalid_generation") from None

    def _bump(self, conn):
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        next_value = integer(self._generation(conn) + 1)
        conn.execute(
            "UPDATE config SET value=? WHERE key='generation'", (str(next_value),)
        )

    def _state(self, conn, namespace):
        row = conn.execute(
            "SELECT * FROM states WHERE namespace=?", (namespace,)
        ).fetchone()
        if row:
            for key in ("baseline", "watermark", "observed"):
                integer(row[key])
            if row["baseline"] > row["watermark"] or row["watermark"] > row["observed"]:
                raise ProjectionError("invalid_cursor")
        return row

    def _check(self, authority, scope, source, snapshots, snapshot, *, worker=False):
        selection(authority, scope, snapshots)
        if source.source_id != self.source_id:
            raise ProjectionError("source_identity_mismatch")
        with self._connect() as conn:
            if (
                conn.execute("SELECT value FROM config WHERE key='enabled'").fetchone()[
                    0
                ]
                != "true"
            ):
                raise ProjectionError("consumer_disabled")
            state = self._state(conn, scope.namespace_id)
        if state:
            previous = tuple(tuple(x) for x in json.loads(state["selection"]))
            if (worker and previous != snapshots) or (
                not worker and not set(snapshots).issubset(previous)
            ):
                raise ProjectionError("selection_requires_rebuild")
            if snapshot.revision < state["observed"]:
                raise ProjectionError("authority_rollback")
            if state["anchor"]:
                raw = json.loads(state["anchor"])
                old = ProjectionEvent(
                    raw["namespace_id"],
                    raw["event_id"],
                    raw["revision"],
                    tuple(raw["memory_ids"]),
                ).validate()
                if source.event(authority, scope, old.event_id) != old:
                    raise ProjectionError("authority_changed_rebuild_required")
        return state

    @staticmethod
    def _advance(conn, namespace):
        state = conn.execute(
            "SELECT baseline,watermark FROM states WHERE namespace=?", (namespace,)
        ).fetchone()
        cursor = integer(state["watermark"])
        for _ in range(1000):
            row = conn.execute(
                "SELECT state FROM receipts WHERE namespace=? AND revision=?",
                (namespace, cursor + 1),
            ).fetchone()
            if row is None or row[0] != "done":
                break
            cursor += 1
        conn.execute(
            "UPDATE states SET watermark=? WHERE namespace=?", (cursor, namespace)
        )

    def _fence(self, conn, namespace):
        enabled = conn.execute(
            "SELECT value FROM config WHERE key='enabled'"
        ).fetchone()[0]
        if enabled not in ("true", "false"):
            raise ProjectionError("invalid_enabled")
        state = self._state(conn, namespace)
        return self._generation(conn), enabled, state["selection"] if state else None

    def _capture_fence(self, namespace, *, maintenance=False):
        with self._connect() as conn:
            fence = self._fence(conn, namespace)
        if not maintenance and fence[1] != "true":
            raise ProjectionError("consumer_disabled")
        return fence

    def _write_fence(self, conn, namespace, fence, *, maintenance=False):
        conn.execute("BEGIN IMMEDIATE")
        current = self._fence(conn, namespace)
        if not maintenance and current[1] != "true":
            raise ProjectionError("consumer_disabled")
        if current != fence:
            raise ProjectionError("consumer_changed")

    def _record_failure(self, event, code, fence):
        with self._connect(write=True) as conn, conn:
            self._write_fence(conn, event.namespace_id, fence)
            self._bump(conn)
            old = conn.execute(
                "SELECT event FROM receipts WHERE namespace=? AND revision=?",
                (event.namespace_id, event.revision),
            ).fetchone()
            if old and old[0] != event_json(event):
                raise ProjectionError("receipt_conflict")
            conn.execute(
                "INSERT INTO receipts VALUES (?,?,?,'retry',1,?) ON CONFLICT(namespace,revision) "
                "DO UPDATE SET state='retry',attempts=attempts+1,error=excluded.error",
                (event.namespace_id, event.revision, event_json(event), code),
            )
            conn.execute(
                "UPDATE states SET watermark=min(watermark,?) WHERE namespace=? AND baseline<?",
                (event.revision - 1, event.namespace_id, event.revision),
            )

    def _write_fact(self, conn, namespace, fact, prepared):
        old = conn.execute(
            "SELECT revision,status FROM facts WHERE namespace=? AND fact=?",
            (namespace, fact.fact_id),
        ).fetchone()
        purged = conn.execute(
            "SELECT 1 FROM purges WHERE namespace=? AND fact IN (?, '')",
            (namespace, fact.fact_id),
        ).fetchone()
        if purged or (
            old
            and (
                old["revision"] > fact.revision
                or old["status"] in ("deleted", "superseded", "purged")
            )
        ):
            return
        self._remove(conn, namespace, fact.fact_id)
        conn.execute(
            "INSERT INTO facts VALUES (?,?,?,?,?) ON CONFLICT(namespace,fact) DO UPDATE SET "
            "revision=excluded.revision,status=excluded.status,digest=excluded.digest",
            (
                namespace,
                fact.fact_id,
                fact.revision,
                fact.status,
                digest(asdict(fact) | {"visible": False}),
            ),
        )
        if fact.status == "active":
            self._insert(conn, namespace, fact, prepared)

    def deliver(self, authority, scope, source, event, *, snapshots=()):
        snapshots = selection(authority, scope, snapshots)
        if type(event) is not ProjectionEvent:
            raise ProjectionError("invalid_event")
        event.validate()
        if event.namespace_id != scope.namespace_id:
            raise ScopeDenied()
        fence = self._capture_fence(scope.namespace_id)
        if source.event(authority, scope, event.event_id) != event:
            raise ProjectionError("event_identity_mismatch")
        snapshot = source.snapshot(authority, scope, snapshots)
        self._check(authority, scope, source, snapshots, snapshot, worker=True)
        try:
            if snapshot.truncated:
                raise ProjectionError("snapshot_budget")
            chosen = [
                fact for fact in snapshot.facts if fact.fact_id in event.memory_ids
            ]
            prepared = {
                f.fact_id: self._prepare(f) for f in chosen if f.status == "active"
            }
            self._fault("after_compute")
            if (
                source.event(authority, scope, event.event_id) != event
                or source.snapshot(authority, scope, snapshots).fingerprint
                != snapshot.fingerprint
            ):
                raise ProjectionError("authority_changed")
            authority.require(scope)
            with self._connect(write=True) as conn, conn:
                self._write_fence(conn, scope.namespace_id, fence)
                self._bump(conn)
                old = conn.execute(
                    "SELECT event FROM receipts WHERE namespace=? AND revision=?",
                    (scope.namespace_id, event.revision),
                ).fetchone()
                if old and old[0] != event_json(event):
                    raise ProjectionError("receipt_conflict")
                if (
                    conn.execute(
                        "SELECT value FROM config WHERE key='enabled'"
                    ).fetchone()[0]
                    != "true"
                ):
                    raise ProjectionError("consumer_disabled")
                state = self._state(conn, scope.namespace_id)
                if state and state["selection"] != canonical_json(
                    snapshots_as_lists(snapshots)
                ):
                    raise ProjectionError("selection_requires_rebuild")
                if state and snapshot.revision < state["observed"]:
                    raise ProjectionError("authority_changed")
                conn.execute(
                    "INSERT INTO states VALUES (?,?,0,0,?,?) ON CONFLICT(namespace) DO UPDATE SET "
                    "observed=excluded.observed,anchor=excluded.anchor",
                    (
                        scope.namespace_id,
                        canonical_json(snapshots_as_lists(snapshots)),
                        snapshot.revision,
                        event_json(snapshot.anchor) if snapshot.anchor else None,
                    ),
                )
                for fact in chosen:
                    self._write_fact(
                        conn, scope.namespace_id, fact, prepared.get(fact.fact_id)
                    )
                conn.execute(
                    "INSERT INTO receipts VALUES (?,?,?,'done',1,NULL) ON CONFLICT(namespace,revision) "
                    "DO UPDATE SET state='done',attempts=attempts+1,error=NULL",
                    (scope.namespace_id, event.revision, event_json(event)),
                )
                self._advance(conn, scope.namespace_id)
                self._fault("before_commit")
                committed_fence = self._fence(conn, scope.namespace_id)
            fence = committed_fence
            self._fault("after_commit")
            if (
                source.snapshot(authority, scope, snapshots).fingerprint
                != snapshot.fingerprint
            ):
                raise ProjectionError("authority_changed")
            authority.require(scope)
        except ProjectionError as exc:
            try:
                self._record_failure(event, exc.code, fence)
            except ProjectionError:
                # A changed generation owns its ledger. Never overwrite its
                # receipt/baseline with an obsolete operation's retry outcome.
                pass
            raise

    def rebuild(self, authority, scope, source, *, snapshots=(), limit=1000):
        """Host-only bounded reconciliation; preserve deletion/retention guards."""
        snapshots = selection(authority, scope, snapshots)
        if source.source_id != self.source_id:
            raise ProjectionError("source_identity_mismatch")
        fence = self._capture_fence(scope.namespace_id, maintenance=True)
        snapshot = source.snapshot(authority, scope, snapshots, limit=limit)
        if snapshot.truncated:
            raise ProjectionError("snapshot_budget")
        prepared = {
            f.fact_id: self._prepare(f) for f in snapshot.facts if f.status == "active"
        }
        if (
            source.snapshot(authority, scope, snapshots, limit=limit).fingerprint
            != snapshot.fingerprint
        ):
            raise ProjectionError("authority_changed")
        authority.require(scope)
        with self._connect(write=True) as conn, conn:
            self._write_fence(conn, scope.namespace_id, fence, maintenance=True)
            self._bump(conn)
            # Rebuild never lowers observations, even when explicitly requested.
            state = self._state(conn, scope.namespace_id)
            if state and state["observed"] > snapshot.revision:
                raise ProjectionError("authority_rollback")
            for row in conn.execute(
                "SELECT fact FROM facts WHERE namespace=? AND status='active'",
                (scope.namespace_id,),
            ).fetchall():
                self._remove(conn, scope.namespace_id, row[0])
            for fact in snapshot.facts:
                self._write_fact(
                    conn, scope.namespace_id, fact, prepared.get(fact.fact_id)
                )
            conn.execute(
                "INSERT INTO states VALUES (?,?,?,?,?,?) ON CONFLICT(namespace) DO UPDATE SET "
                "selection=excluded.selection,baseline=excluded.baseline,watermark=excluded.watermark,"
                "observed=excluded.observed,anchor=excluded.anchor",
                (
                    scope.namespace_id,
                    canonical_json(snapshots_as_lists(snapshots)),
                    snapshot.revision,
                    snapshot.revision,
                    snapshot.revision,
                    event_json(snapshot.anchor) if snapshot.anchor else None,
                ),
            )
            self._fault("before_commit")
        if (
            source.snapshot(authority, scope, snapshots, limit=limit).fingerprint
            != snapshot.fingerprint
        ):
            # Baseline cannot be advertised as current while authority advances.
            raise ProjectionError("authority_changed")
        authority.require(scope)

    def status(self, authority, scope, source, *, snapshots=()):
        snapshots = selection(authority, scope, snapshots)
        snap = source.snapshot(authority, scope, snapshots)
        return self._status_snapshot(authority, scope, source, snapshots, snap)

    def _status_snapshot(self, authority, scope, source, snapshots, snap):
        state = self._check(authority, scope, source, snapshots, snap)
        watermark = state["watermark"] if state else 0
        reasons = list(snap.reasons)
        if snap.truncated:
            reasons.append("snapshot_budget")
        if not state:
            reasons.append("not_initialized")
        if watermark < snap.revision:
            reasons.append("consumer_lag")
            events = source.events(authority, scope, after_revision=watermark, limit=1)
            if not events or events[0].revision != watermark + 1:
                reasons.append("event_gap")
        authority.require(scope)
        return {
            "consumer": self.kind,
            "status": "partial" if reasons else "complete",
            "reasons": tuple(reasons),
            "watermark": watermark,
            "authority_revision": snap.revision,
            "lag": snap.revision - watermark,
        }

    def _query_start(self, authority, scope, source, snapshots, candidate_limit):
        snapshots = selection(authority, scope, snapshots)
        generation = self._capture_fence(scope.namespace_id)[0]
        snap = source.snapshot(authority, scope, snapshots, limit=candidate_limit)
        state = self._status_snapshot(authority, scope, source, snapshots, snap)
        reasons = list(state["reasons"])
        if snap.truncated and "snapshot_budget" not in reasons:
            reasons.append("snapshot_budget")
        with self._connect() as conn:
            if self._generation(conn) != generation:
                raise ProjectionError("consumer_changed")
            if (
                conn.execute("SELECT value FROM config WHERE key='enabled'").fetchone()[
                    0
                ]
                != "true"
            ):
                raise ProjectionError("consumer_disabled")
        return snapshots, snap, reasons, generation

    def _query_finish(
        self, authority, scope, source, snapshots, snap, candidate_limit, generation
    ):
        if (
            source.snapshot(
                authority, scope, snapshots, limit=candidate_limit
            ).fingerprint
            != snap.fingerprint
        ):
            raise ProjectionError("authority_changed")
        with self._connect() as conn:
            if self._generation(conn) != generation:
                raise ProjectionError("consumer_changed")
        authority.require(scope)

    def _live(self, conn, namespace, snap):
        result = {}
        for fact in snap.facts:
            if fact.status != "active" or not fact.visible:
                continue
            row = conn.execute(
                "SELECT revision,status,digest FROM facts WHERE namespace=? AND fact=?",
                (namespace, fact.fact_id),
            ).fetchone()
            if row:
                integer(row["revision"], 1)
            if (
                row
                and row["revision"] == fact.revision
                and row["status"] == "active"
                and row["digest"] == digest(asdict(fact) | {"visible": False})
            ):
                result[fact.fact_id] = fact
        return result

    def admin_disable(self, disabled=True):
        if type(disabled) is not bool:
            raise ProjectionError("invalid_request")
        with self._connect(write=True) as conn, conn:
            self._bump(conn)
            conn.execute(
                "UPDATE config SET value=? WHERE key='enabled'",
                ("false" if disabled else "true",),
            )

    def admin_plan_purge(self, namespace_id, fact_id=None):
        validate_identifier(namespace_id)
        if fact_id is not None:
            validate_identifier(fact_id)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT fact,revision FROM facts WHERE namespace=? AND (? IS NULL OR fact=?) ORDER BY fact LIMIT 1001",
                (namespace_id, fact_id, fact_id),
            ).fetchall()
            if len(rows) > 1000:
                raise ProjectionError("purge_budget")
            subjects = tuple((r[0], integer(r[1], 1)) for r in rows)
        return PurgePlan(
            namespace_id,
            fact_id,
            subjects,
            digest(
                [
                    self.source_id,
                    self.kind,
                    namespace_id,
                    fact_id,
                    list(map(list, subjects)),
                ]
            ),
        )

    def admin_purge(self, plan):
        if (
            type(plan) is not PurgePlan
            or self.admin_plan_purge(plan.namespace_id, plan.fact_id) != plan
        ):
            raise ProjectionError("purge_plan_changed")
        with self._connect(write=True) as conn, conn:
            # Revalidate exact subjects inside the write transaction.
            conn.execute("BEGIN IMMEDIATE")
            self._bump(conn)
            rows = conn.execute(
                "SELECT fact,revision FROM facts WHERE namespace=? AND (? IS NULL OR fact=?) ORDER BY fact LIMIT 1001",
                (plan.namespace_id, plan.fact_id, plan.fact_id),
            ).fetchall()
            if tuple((r[0], r[1]) for r in rows) != plan.subjects:
                raise ProjectionError("purge_plan_changed")
            revision = max((r[1] for r in plan.subjects), default=0)
            conn.execute(
                "INSERT INTO purges VALUES (?,?,?) ON CONFLICT(namespace,fact) DO UPDATE SET revision=max(revision,excluded.revision)",
                (plan.namespace_id, plan.fact_id or "", revision),
            )
            for fact, _ in plan.subjects:
                self._remove(conn, plan.namespace_id, fact)
                conn.execute(
                    "UPDATE facts SET status='purged',digest='' WHERE namespace=? AND fact=?",
                    (plan.namespace_id, fact),
                )

    def diagnostics(self):
        with self._connect() as conn:
            return {
                "kind": self.kind,
                "schema_version": 1,
                "source_id": self.source_id,
                "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
                "synchronous": conn.execute("PRAGMA synchronous").fetchone()[0],
                "enabled": conn.execute(
                    "SELECT value FROM config WHERE key='enabled'"
                ).fetchone()[0]
                == "true",
                "identity": json.loads(self.identity),
            }

    def export(self, namespace_id, *, limit=100):
        """Host-only bounded metadata export; never includes authority bodies."""
        validate_identifier(namespace_id)
        integer(limit, 1, 1000)
        with self._connect() as conn:
            result = {}
            for table in ("states", "receipts", "facts", "purges"):
                rows = conn.execute(
                    f"SELECT * FROM {table} WHERE namespace=? LIMIT ?",
                    (namespace_id, limit + 1),
                ).fetchall()
                result[table] = [dict(row) for row in rows[:limit]]
                result[table + "_truncated"] = len(rows) > limit
            return result

    def backup(self, destination):
        destination = Path(destination).absolute()
        # Exclusive creation prevents accidental replacement of a database/backup.
        with destination.open("xb"):
            pass
        with self._connect() as source:
            target = sqlite3.connect(str(destination))
            deadline = time.monotonic() + 10
            try:

                def progress(status, remaining, total):
                    if time.monotonic() > deadline:
                        raise ProjectionError("backup_budget")

                source.backup(target, pages=128, progress=progress, sleep=0.01)
            finally:
                target.close()
        return destination

    @classmethod
    def restore(cls, backup, destination, **kwargs):
        """Host-only: verify backup without modifying it, copy to new destination."""
        # Constructors validate existing identity but do not write existing files.
        if not Path(backup).is_file():
            raise ProjectionError("backup_unavailable")
        source = cls(backup, **kwargs)
        source.backup(destination)
        return cls(destination, **kwargs)


def snapshots_as_lists(snapshots):
    return [list(pair) for pair in snapshots]

"""Read-only adapter over the installed, strictly validated durable authority.

No delivered-bit dependency, schema migration, source writes or network access.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from jitmind.scope import ScopeDenied
from jitmind.storage import SQLiteDurableStore
from jitmind.storage.models import canonical_json, validate_identifier
from jitmind.storage.sqlite import _entry

from .models import (
    FactRecord,
    ProjectionError,
    ProjectionEvent,
    SourceSnapshot,
    integer,
    selection,
)


def digest(value):
    return hashlib.sha256(
        canonical_json(
            json.loads(json.dumps(value, allow_nan=False)),
            max_bytes=16_777_216,
            max_nodes=100_000,
        ).encode()
    ).hexdigest()


class SQLiteProjectionSource:
    def __init__(self, store, *, source_id: str):
        if type(store) is not SQLiteDurableStore:
            raise ProjectionError("invalid_source")
        validate_identifier(source_id)
        self.store = store
        self.source_id = source_id

    @contextmanager
    def _read(self):
        conn = None
        try:
            conn = sqlite3.connect(
                self.store.path.as_uri() + "?mode=ro",
                uri=True,
                timeout=0.25,
                isolation_level=None,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            deadline = time.monotonic() + 3
            ticks = 0

            def budget():
                nonlocal ticks
                ticks += 1
                return ticks > 10000 or time.monotonic() > deadline

            conn.set_progress_handler(budget, 1000)
            # Use the *actual installed* validator, including C03 v2 when composed.
            self.store._check_schema(conn)
            if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "delete":
                raise ProjectionError("source_journal_unavailable")
            conn.execute("BEGIN")
            yield conn
        except (ProjectionError, ScopeDenied):
            raise
        except Exception:  # noqa: BLE001 - sanitize corrupt authority/schema data
            raise ProjectionError("source_unavailable") from None
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def _event(row):
        if row is None:
            return None
        raw = row["memory_ids"]
        if type(raw) is not str or len(raw.encode()) > 262144:
            raise ProjectionError("invalid_event")
        ids = json.loads(raw)
        if type(ids) is not list:
            raise ProjectionError("invalid_event")
        return ProjectionEvent(
            row["namespace_id"], row["event_id"], row["revision"], tuple(ids)
        ).validate()

    def events(self, authority, scope, *, after_revision=0, limit=100):
        authority.require(scope)
        integer(after_revision)
        integer(limit, 1, 1000)
        with self._read() as conn:
            rows = conn.execute(
                "SELECT namespace_id,event_id,revision,memory_ids FROM outbox "
                "WHERE namespace_id=? AND revision>? ORDER BY revision LIMIT ?",
                (scope.namespace_id, after_revision, limit),
            )
            result = []
            size = 0
            for row in rows:
                size += len(row["memory_ids"].encode())
                if size > 8_388_608:
                    raise ProjectionError("event_budget")
                result.append(self._event(row))
            result = tuple(result)
            if hasattr(self.store, "outbox_events"):
                public = self.store.outbox_events(
                    scope.namespace_id, after_revision=after_revision, limit=limit
                )
                normalized = tuple(
                    ProjectionEvent(
                        e.namespace_id, e.event_id, e.revision, e.memory_ids
                    ).validate()
                    for e in public
                )
                if normalized != result:
                    raise ProjectionError("authority_changed")
        authority.require(scope)
        return result

    def event(self, authority, scope, event_id):
        authority.require(scope)
        validate_identifier(event_id)
        with self._read() as conn:
            result = self._event(
                conn.execute(
                    "SELECT namespace_id,event_id,revision,memory_ids FROM outbox "
                    "WHERE namespace_id=? AND event_id=?",
                    (scope.namespace_id, event_id),
                ).fetchone()
            )
            if hasattr(self.store, "get_event"):
                public = self.store.get_event(scope.namespace_id, event_id)
                normalized = (
                    ProjectionEvent(
                        public.namespace_id,
                        public.event_id,
                        public.revision,
                        public.memory_ids,
                    ).validate()
                    if public
                    else None
                )
                if normalized != result:
                    raise ProjectionError("authority_changed")
        authority.require(scope)
        return result

    @staticmethod
    def _filter(namespace, snapshots):
        # Metadata restrictions are evaluated by SQL BEFORE payload selection.
        clauses = ["f.namespace_id=?"]
        args = [namespace]
        for alias in ("f", "p"):
            for key in ("namespace_id", "namespace"):
                field = f"json_extract({alias}.payload,'$.meta.{key}')"
                missing = f"json_type({alias}.payload,'$.meta.{key}') IS NULL"
                if key == "namespace":
                    clauses.append(f"({missing} OR {field}=? OR {field}=?)")
                    args.extend(
                        (namespace, json.dumps([namespace], separators=(",", ":")))
                    )
                else:
                    clauses.append(f"({missing} OR {field}=?)")
                    args.append(namespace)
        f_repo = "json_extract(f.payload,'$.meta.repo_id')"
        p_repo = "json_extract(p.payload,'$.meta.repo_id')"
        f_snap = "json_extract(f.payload,'$.meta.snapshot_id')"
        p_snap = "json_extract(p.payload,'$.meta.snapshot_id')"
        clauses.extend(
            (
                f"({f_repo} IS NULL OR {p_repo} IS NULL OR {f_repo}={p_repo})",
                f"({f_snap} IS NULL OR {p_snap} IS NULL OR {f_snap}={p_snap})",
            )
        )
        repo = f"coalesce({f_repo},{p_repo})"
        snap = f"coalesce({f_snap},{p_snap})"
        pairs = [f"({repo} IS NULL AND {snap} IS NULL)"]
        for r, s in snapshots:
            pairs.append(f"({repo}=? AND {snap}=?)")
            args.extend((r, s))
        clauses.append("(" + " OR ".join(pairs) + ")")
        return " AND ".join(clauses), args, repo, snap

    def snapshot(self, authority, scope, snapshots=(), *, limit=1000):
        snapshots = selection(authority, scope, snapshots)
        integer(limit, 1, 1000)
        where, args, repo, snap = self._filter(scope.namespace_id, snapshots)
        with self._read() as conn:
            row = conn.execute(
                "SELECT revision FROM namespaces WHERE namespace_id=?",
                (scope.namespace_id,),
            ).fetchone()
            revision = integer(row[0] if row else 0)
            anchor = self._event(
                conn.execute(
                    "SELECT namespace_id,event_id,revision,memory_ids FROM outbox "
                    "WHERE namespace_id=? AND revision=?",
                    (scope.namespace_id, revision),
                ).fetchone()
            )
            population = (
                "FROM facts f JOIN pages p ON f.namespace_id=p.namespace_id AND f.page_id=p.page_id "
                "WHERE " + where + " ORDER BY f.memory_id"
            )
            truncated = (
                conn.execute(
                    "SELECT 1 " + population + " LIMIT 1 OFFSET ?", (*args, limit)
                ).fetchone()
                is not None
            )
            sizes = conn.execute(
                "SELECT CASE WHEN f.status='active' THEN "
                "length(CAST(f.payload AS BLOB))+length(CAST(json_extract(p.payload,'$.meta') AS BLOB)) "
                "ELSE 0 END " + population + " LIMIT ?",
                (*args, limit),
            ).fetchall()
            if sum(row[0] for row in sizes) > 8_388_608:
                raise ProjectionError("payload_budget")
            rows = conn.execute(
                "SELECT f.memory_id,f.page_id,f.revision,f.status,"
                + repo
                + " AS repo,"
                + snap
                + " AS snap,"
                "CASE WHEN f.status='active' THEN f.payload ELSE NULL END AS payload,"
                "CASE WHEN f.status='active' THEN json_extract(p.payload,'$.meta') ELSE NULL END AS page_meta "
                "FROM facts f JOIN pages p ON f.namespace_id=p.namespace_id AND f.page_id=p.page_id "
                "WHERE " + where + " ORDER BY f.memory_id LIMIT ?",
                (*args, limit),
            )
            facts = []
            size = 0
            now = datetime.now(timezone.utc)
            reasons = set()
            for row in rows:
                authority.require(scope, row["repo"])
                validate_identifier(row["memory_id"])
                integer(row["revision"], 1, revision)
                if row["status"] not in ("active", "deleted", "superseded"):
                    raise ProjectionError("invalid_fact")
                content, meta, visible = "", {}, False
                if row["status"] == "active":
                    size += len(row["payload"].encode()) + len(
                        row["page_meta"].encode()
                    )
                    if size > 8_388_608:
                        raise ProjectionError("payload_budget")
                    entry = _entry(row["payload"])
                    if (
                        entry.id != row["memory_id"]
                        or entry.status != row["status"]
                        or entry.source_page_id != row["page_id"]
                    ):
                        raise ProjectionError("invalid_fact")
                    content = entry.content
                    canonical_json(content, max_bytes=131072)
                    page_meta = json.loads(row["page_meta"])
                    meta = dict(entry.meta)
                    meta.update(
                        namespace_id=scope.namespace_id,
                        source_page_id=entry.source_page_id,
                        t_valid=entry.t_valid,
                        t_invalid=entry.t_invalid,
                    )
                    if row["repo"] is not None:
                        meta.update(repo_id=row["repo"], snapshot_id=row["snap"])
                    # Page validity is the original observation, not a second
                    # current interval. C03 corrections only update the fact.
                    visible = True
                    for key in ("t_valid", "t_invalid"):
                        raw = meta.get(key)
                        if raw is not None:
                            instant = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                            if instant.tzinfo is None:
                                raise ProjectionError("invalid_time")
                            if (key == "t_valid" and now < instant) or (
                                key == "t_invalid" and now >= instant
                            ):
                                visible = False
                    for restrictions in (meta, page_meta):
                        try:
                            ends = []
                            raw = restrictions.get("expires_at")
                            if raw is not None:
                                instant = datetime.fromisoformat(
                                    raw.replace("Z", "+00:00")
                                )
                                if instant.tzinfo is None:
                                    raise ValueError
                                ends.append(instant)
                            if "ttl_seconds" in restrictions:
                                ttl = restrictions["ttl_seconds"]
                                if (
                                    type(ttl) not in (int, float)
                                    or not math.isfinite(ttl)
                                    or ttl < 0
                                ):
                                    raise ValueError
                                origin = datetime.fromisoformat(
                                    entry.t_created.replace("Z", "+00:00")
                                )
                                ends.append(origin + timedelta(seconds=ttl))
                            if any(now >= end for end in ends):
                                visible = False
                        except (ValueError, TypeError, AttributeError, OverflowError):
                            visible = False
                            reasons.add("unknown_eligibility")
                facts.append(
                    FactRecord(
                        row["memory_id"],
                        row["revision"],
                        row["status"],
                        content,
                        canonical_json(meta),
                        row["repo"],
                        row["snap"],
                        visible,
                    )
                )
            fingerprint = digest(
                [
                    revision,
                    [asdict(f) for f in facts],
                    truncated,
                    sorted(reasons),
                    asdict(anchor) if anchor else None,
                ]
            )
        authority.require(scope)
        return SourceSnapshot(
            scope.namespace_id,
            revision,
            tuple(facts),
            truncated,
            fingerprint,
            anchor,
            tuple(sorted(reasons)),
        )

"""Lazy SQLite metadata storage. No code, fact content, or provider work in writes."""

import sqlite3
from contextlib import contextmanager

from .binding_models import (
    BindingConflict,
    BindingStaleSource,
    BindingStorageError,
    CodeBinding,
)

SCHEMA = 2


class BindingStore:
    def __init__(self, path):
        self.path = str(path)

    @contextmanager
    def connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.path, timeout=0.25, isolation_level=None)
            conn.execute("PRAGMA busy_timeout=250")
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            if mode.lower() not in ("delete",):
                raise BindingStorageError()
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, SCHEMA):
                raise BindingStorageError()
            if version < SCHEMA:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    if conn.execute("PRAGMA user_version").fetchone()[0] == 0:
                        if conn.execute(
                            "SELECT 1 FROM sqlite_master WHERE type='table'"
                        ).fetchone():
                            raise BindingStorageError()
                        conn.execute(
                            "CREATE TABLE heads(ns TEXT, logical TEXT, revision INTEGER, PRIMARY KEY(ns,logical))"
                        )
                        conn.execute(
                            "CREATE TABLE history(ns TEXT, logical TEXT, revision INTEGER, payload TEXT NOT NULL, PRIMARY KEY(ns,logical,revision), FOREIGN KEY(ns,logical) REFERENCES heads(ns,logical))"
                        )
                        conn.execute(
                            "CREATE TABLE requests(ns TEXT, logical TEXT, request TEXT, digest TEXT, revision INTEGER, PRIMARY KEY(ns,logical,request), FOREIGN KEY(ns,logical,revision) REFERENCES history(ns,logical,revision))"
                        )
                        conn.execute(
                            "CREATE TABLE sources(ns TEXT, repo TEXT, revision INTEGER, snapshot TEXT, generation INTEGER, PRIMARY KEY(ns,repo))"
                        )
                        conn.execute(
                            "CREATE TABLE facts(ns TEXT, fact TEXT, revision INTEGER, terminal INTEGER, PRIMARY KEY(ns,fact))"
                        )
                        conn.execute("PRAGMA user_version=1")
                    if conn.execute("PRAGMA user_version").fetchone()[0] == 1:
                        # Local schema migration extracts indexing metadata in SQL;
                        # it never hydrates binding payloads in the application.
                        conn.execute("ALTER TABLE history ADD COLUMN repo TEXT")
                        conn.execute("ALTER TABLE history ADD COLUMN snapshot TEXT")
                        conn.execute(
                            "UPDATE history SET repo=json_extract(payload,'$.repo_id'), snapshot=json_extract(payload,'$.snapshot_id')"
                        )
                        conn.execute(
                            "CREATE INDEX history_scope ON history(ns,logical,repo,snapshot,revision)"
                        )
                        # Older releases could record 1 -> 2 -> 1. Retain their
                        # immutable history, but recover the highest observation.
                        conn.execute(
                            "UPDATE sources SET (snapshot,generation)=(SELECT snapshot,json_extract(payload,'$.source_generation') FROM history WHERE history.ns=sources.ns AND history.repo=sources.repo ORDER BY json_extract(payload,'$.source_generation') DESC,rowid DESC LIMIT 1) WHERE EXISTS (SELECT 1 FROM history WHERE history.ns=sources.ns AND history.repo=sources.repo)"
                        )
                        conn.execute("PRAGMA user_version=2")
                    conn.commit()
                except BaseException:
                    conn.rollback()
                    raise
            yield conn
        except sqlite3.Error:
            raise BindingStorageError() from None
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def _repos(repo_ids):
        # No unscoped payload-read API, including the empty-grant case.
        return ",".join("?" for _ in repo_ids) or "NULL"

    def current(self, ns, logical, *, repo_ids):
        with self.connection() as conn:
            row = conn.execute(
                "SELECT h.payload FROM history h JOIN heads d ON h.ns=d.ns AND h.logical=d.logical AND h.revision=d.revision WHERE h.ns=? AND h.logical=? "
                f"AND h.repo IN ({self._repos(repo_ids)})",
                (ns, logical, *repo_ids),
            ).fetchone()
        return CodeBinding.model_validate_json(row[0]) if row else None

    def history(self, ns, logical, after, limit, *, repo_ids, snapshot_id=None):
        snapshot_filter = " AND snapshot=?" if snapshot_id is not None else ""
        params = (ns, logical, *repo_ids, after)
        if snapshot_id is not None:
            params += (snapshot_id,)
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT payload FROM history WHERE ns=? AND logical=? "
                f"AND repo IN ({self._repos(repo_ids)}) AND revision>?"
                + snapshot_filter
                + " ORDER BY revision LIMIT ?",
                (*params, limit),
            ).fetchall()
        return tuple(CodeBinding.model_validate_json(row[0]) for row in rows)

    def replay(self, ns, logical, request, digest, *, repo_ids, snapshot_id):
        with self.connection() as conn:
            row = conn.execute(
                "SELECT r.digest,h.payload FROM requests r JOIN history h ON h.ns=r.ns AND h.logical=r.logical AND h.revision=r.revision WHERE r.ns=? AND r.logical=? AND r.request=? "
                f"AND h.repo IN ({self._repos(repo_ids)}) AND h.snapshot=?",
                (ns, logical, request, *repo_ids, snapshot_id),
            ).fetchone()
            if (
                row is None
                and conn.execute(
                    "SELECT 1 FROM requests r JOIN history h ON h.ns=r.ns AND h.logical=r.logical AND h.revision=r.revision WHERE r.ns=? AND r.logical=? AND r.request=? "
                    f"AND h.repo IN ({self._repos(repo_ids)})",
                    (ns, logical, request, *repo_ids),
                ).fetchone()
            ):
                # Conflicting reuse at another snapshot needs no payload read.
                raise BindingConflict()
        if row and row[0] != digest:
            raise BindingConflict()
        return CodeBinding.model_validate_json(row[1]) if row else None

    def source_revision(self, ns, repo):
        with self.connection() as conn:
            row = conn.execute(
                "SELECT revision FROM sources WHERE ns=? AND repo=?", (ns, repo)
            ).fetchone()
        return row[0] if row else 0

    def is_terminal(self, ns, fact):
        with self.connection() as conn:
            row = conn.execute(
                "SELECT terminal FROM facts WHERE ns=? AND fact=?", (ns, fact)
            ).fetchone()
        return bool(row and row[0])

    def project(self, ns, fact, revision, terminal):
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT revision,terminal FROM facts WHERE ns=? AND fact=?",
                    (ns, fact),
                ).fetchone()
                if row and (row[1] or (revision <= row[0] and not terminal)):
                    conn.rollback()
                    return False
                if row:
                    revision = max(revision, row[0])
                conn.execute(
                    "INSERT INTO facts VALUES (?,?,?,?) ON CONFLICT(ns,fact) DO UPDATE SET revision=excluded.revision,terminal=excluded.terminal",
                    (ns, fact, revision, int(terminal)),
                )
                conn.commit()
                return True
            except BaseException:
                conn.rollback()
                raise

    def append(
        self, binding, expected_revision, source_revision, request, request_digest
    ):
        ns, logical = binding.namespace_id, binding.logical_binding_id
        payload = binding.model_dump_json()
        if len(payload.encode()) > 65536:
            raise BindingStorageError()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                head = conn.execute(
                    "SELECT revision FROM heads WHERE ns=? AND logical=?", (ns, logical)
                ).fetchone()
                source = conn.execute(
                    "SELECT revision,generation,snapshot FROM sources WHERE ns=? AND repo=?",
                    (ns, binding.repo_id),
                ).fetchone()
                fact = conn.execute(
                    "SELECT terminal FROM facts WHERE ns=? AND fact=?",
                    (ns, binding.fact_version_id),
                ).fetchone()
                if (
                    (head[0] if head else 0) != expected_revision
                    or (source[0] if source else 0) != source_revision
                    or (fact and fact[0])
                ):
                    raise BindingConflict()
                if source and (
                    binding.source_generation < source[1]
                    or (
                        binding.source_generation == source[1]
                        and binding.snapshot_id != source[2]
                    )
                ):
                    # Unknown observations are subject to the same watermark.
                    # Reject before any head/history/request/source mutation.
                    raise BindingStaleSource()
                conn.execute(
                    "INSERT INTO heads VALUES (?,?,?) ON CONFLICT(ns,logical) DO UPDATE SET revision=excluded.revision",
                    (ns, logical, binding.revision),
                )
                conn.execute(
                    "INSERT INTO history(ns,logical,revision,payload,repo,snapshot) VALUES (?,?,?,?,?,?)",
                    (
                        ns,
                        logical,
                        binding.revision,
                        payload,
                        binding.repo_id,
                        binding.snapshot_id,
                    ),
                )
                conn.execute(
                    "INSERT INTO requests VALUES (?,?,?,?,?)",
                    (ns, logical, request, request_digest, binding.revision),
                )
                conn.execute(
                    "INSERT INTO sources VALUES (?,?,?,?,?) ON CONFLICT(ns,repo) DO UPDATE SET revision=excluded.revision,snapshot=excluded.snapshot,generation=excluded.generation",
                    (
                        ns,
                        binding.repo_id,
                        source_revision + 1,
                        binding.snapshot_id,
                        binding.source_generation,
                    ),
                )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return binding

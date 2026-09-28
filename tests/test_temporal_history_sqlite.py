"""Real SQLite/restart/fault tests for J09, no provider calls."""

import hashlib
import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from threading import Barrier, Event

import pytest

from jitmind.scope import ScopeAuthority, ScopeContext, ScopeDenied
from jitmind.scoped_temporal import (
    StrictTemporalOracle,
    TemporalConflict,
    TemporalFact,
    TemporalPerspective,
)
from jitmind.temporal_history import (
    SCHEMA,
    DurableTemporalOracle,
    SQLiteTemporalHistory,
    TemporalAcknowledgementUncertain,
    TemporalCapacityError,
    TemporalIdempotencyConflict,
    TemporalSchemaError,
    TemporalStorageError,
)


def ts(month, day=1):
    return datetime(2026, month, day, tzinfo=timezone.utc)


def perspective(month=1, content="value", **kwargs):
    return TemporalPerspective(
        ts(month), (TemporalFact("id", "key", content, ts(1), **kwargs),)
    )


def setup(path, **kwargs):
    authority = ScopeAuthority()
    authority.grant("p", "n", ["repo"])
    scope = authority.context("p", "n", "r")
    backend = SQLiteTemporalHistory(path, **kwargs)
    return authority, scope, backend, DurableTemporalOracle(authority, backend)


def oracle_fixture():
    a = TemporalFact("a", "retry_limit", "3", ts(1))
    b = TemporalFact("b", "retry_limit", "3", ts(1), ts(2))
    c = TemporalFact("c", "retry_limit", "5", ts(2))
    d = TemporalFact("d", "retry_limit", "3", ts(1), ts(1, 15))
    e = TemporalFact("e", "retry_limit", "4", ts(1, 15), ts(2))
    return (
        TemporalPerspective(ts(1), (a,)),
        TemporalPerspective(ts(2), (b, c)),
        TemporalPerspective(ts(3), (d, e, c)),
    )


def query_fixture(path):
    _, scope, backend, oracle = setup(path)
    cases = [
        (ts(1, 20), ts(1, 20)),
        (ts(1, 20), ts(2, 15)),
        (ts(1, 20), ts(3)),
        (ts(1, 10), ts(3, 2)),
        (ts(1, 15), ts(3, 2)),
        (ts(2), ts(3, 2)),
        (ts(2, 10), ts(1, 20)),
    ]
    answers = [
        [
            f.content
            for f in oracle.query_as_of(scope, valid, transaction_at=transaction)
        ]
        for valid, transaction in cases
    ]
    answers.append([f.content for f in oracle.query_as_of(scope, ts(1, 20))])
    assert oracle.query_as_of(scope, "2026-01-20T05:30:00+05:30") == oracle.query_as_of(
        scope, ts(1, 20)
    )
    assert backend.read_perspectives("n")[0].facts[0].valid_to is None
    return answers


def child_query(path, queue):
    try:
        queue.put(query_fixture(path))
    except Exception as exc:  # noqa: BLE001 - report child failure to parent
        queue.put(type(exc).__name__)


def test_exact_seven_queries_latest_and_spawned_process_restart(tmp_path):
    path = tmp_path / "history.db"
    _, scope, backend, oracle = setup(path)
    for index, item in enumerate(oracle_fixture()):
        oracle.publish(scope, item, str(index))
    # Every operation has already closed its connection. Drop all Python holders.
    del oracle, backend
    expected = [["3"], ["3"], ["4"], ["3"], ["4"], ["5"], ["3"], ["4"]]
    assert query_fixture(path) == expected
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=child_query, args=(str(path), queue))
    process.start()
    try:
        assert queue.get(timeout=20) == expected
        process.join(timeout=20)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        queue.close()


def test_utc_canonical_idempotency_and_receipt_after_later_revisions(tmp_path):
    path = tmp_path / "history.db"
    _, scope, _backend, oracle = setup(path)
    item = perspective()
    receipt = oracle.publish(scope, item, "request", expected_revision=0)
    equivalent = TemporalPerspective(
        "2026-01-01T05:30:00+05:30",
        (TemporalFact("id", "key", "value", "2026-01-01T01:00:00+01:00"),),
    )
    assert oracle.publish(scope, equivalent, "request", expected_revision=0) == receipt
    oracle.publish(scope, perspective(2), "next", expected_revision=1)
    _, scope2, reopened, oracle2 = setup(path)
    assert (
        oracle2.publish(scope2, equivalent, "request", expected_revision=999) == receipt
    )
    with pytest.raises(TemporalIdempotencyConflict):
        oracle2.publish(scope2, perspective(content="changed"), "request")
    with pytest.raises(FrozenInstanceError):
        receipt.revision = 5
    with sqlite3.connect(path) as conn:
        recorded, payload, digest = conn.execute(
            "SELECT recorded_at,payload,digest FROM jth_perspectives WHERE revision=1"
        ).fetchone()
    assert recorded == "2026-01-01T00:00:00.000000Z"
    assert (
        hashlib.sha256(payload.encode()).hexdigest() == receipt.payload_digest == digest
    )
    assert reopened.status("n")["revision"] == 2


def test_concurrent_same_key_and_stale_cas(tmp_path):
    path = tmp_path / "history.db"
    _, scope, backend, oracle = setup(path, busy_timeout_ms=2000)
    barrier = Barrier(6)

    def duplicate(_):
        barrier.wait()
        return oracle.publish(scope, perspective(), "same", expected_revision=0)

    with ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(pool.map(duplicate, range(6)))
    assert len(set(receipts)) == 1
    assert backend.status("n")["revision"] == 1
    barrier = Barrier(2)

    def competing(index):
        barrier.wait()
        try:
            return oracle.publish(
                scope, perspective(2, str(index)), f"key{index}", expected_revision=1
            )
        except TemporalConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(competing, range(2)))
    assert sum(r is not None for r in results) == 1
    assert backend.status("n")["revision"] == 2
    with pytest.raises(TemporalConflict):
        backend.append_perspective("n", 0, perspective(3))
    with pytest.raises(TemporalConflict):
        backend.append_perspective("n", True, perspective(3))
    with pytest.raises(TemporalConflict):
        backend.append_perspective("n", 2, perspective(1))
    assert backend.read_perspectives("n")[0] == perspective()


def test_base_protocol_and_exact_namespace_identity(tmp_path):
    authority, scope, backend, _ = setup(tmp_path / "history.db")
    StrictTemporalOracle(authority, backend).publish(scope, perspective())
    for name in ("tenant/a", "tenant_a", "tenant%2Fa", "n2"):
        backend.append_perspective(name, 0, perspective(content=name))
        assert backend.read_perspectives(name)[0].facts[0].content == name
    assert backend.read_perspectives("missing") == ()
    assert backend.read_perspectives("n")[0].facts[0].content == "value"


def test_validity_ttl_and_snapshot_filters_are_not_retention(tmp_path):
    _, scope, backend, oracle = setup(tmp_path / "history.db")
    item = perspective(
        valid_to=ts(2), expires_at=ts(1, 15), repo_id="repo", snapshot_id="main"
    )
    oracle.publish(scope, item, "a")
    args = {"snapshots": (("repo", "main"),)}
    assert oracle.query_as_of(scope, ts(1), **args) == item.facts
    assert oracle.query_as_of(scope, ts(2), **args) == ()
    assert oracle.query_as_of(scope, ts(1), eligible_at=ts(1, 15), **args) == ()
    assert oracle.query_as_of(scope, ts(1), eligible_at=ts(1, 14), **args) == item.facts
    assert oracle.query_as_of(scope, ts(1)) == ()
    assert len(backend.read_perspectives("n")) == 1


def test_bounded_full_history_preflight_and_pinned_pages(tmp_path, monkeypatch):
    path = tmp_path / "history.db"
    _, scope, backend, oracle = setup(path)
    for month in range(1, 5):
        oracle.publish(scope, perspective(month, "x" * 300), str(month))
    total = backend.status("n")["payload_bytes"]
    reader = SQLiteTemporalHistory(path, max_read_bytes=total - 1)
    import jitmind.temporal_history as module

    original = module._decode
    calls = []

    def decode(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(module, "_decode", decode)
    with pytest.raises(TemporalCapacityError):
        reader.read_perspectives("n")
    assert calls == []  # aggregate refusal precedes payload decoding
    count_reader = SQLiteTemporalHistory(path, max_read_count=3)
    with pytest.raises(TemporalCapacityError):
        count_reader.read_perspectives("n")
    first = reader.read_page("n", limit=2)
    oracle.publish(scope, perspective(5), "5")
    second = reader.read_page(
        "n",
        after_revision=first.next_revision,
        limit=2,
        through_revision=first.head_revision,
    )
    assert first.has_more and not second.has_more
    assert first.head_revision == second.head_revision == 4
    assert first.next_revision == 2 and second.next_revision == 4
    assert len(first.perspectives + second.perspectives) == 4
    with pytest.raises(TemporalCapacityError):
        SQLiteTemporalHistory(path, max_read_bytes=1).read_page("n", limit=1)
    with pytest.raises(TemporalConflict):
        reader.read_page("n", after_revision=True)


@pytest.mark.parametrize("mutation", ["overlap", "bool", "date", "content", "tuple"])
def test_revalidates_manually_constructed_models(tmp_path, mutation):
    _, scope, backend, oracle = setup(tmp_path / "history.db")
    item = perspective()
    if mutation == "overlap":
        object.__setattr__(
            item,
            "facts",
            item.facts + (TemporalFact("second", "key", "conflict", ts(1)),),
        )
    elif mutation == "bool":
        object.__setattr__(item.facts[0], "single_valued", 1)
    elif mutation == "date":
        object.__setattr__(item, "recorded_at", ts(1).replace(tzinfo=None))
    elif mutation == "content":
        object.__setattr__(item.facts[0], "content", float("nan"))
    else:
        object.__setattr__(item, "facts", list(item.facts))
    with pytest.raises(TemporalConflict):
        oracle.publish(scope, item, "x")
    assert backend.status("n")["revision"] == 0


def test_payload_and_fact_count_limits(tmp_path):
    _, scope, backend, oracle = setup(
        tmp_path / "history.db", max_payload_bytes=400, max_facts=1
    )
    with pytest.raises(TemporalCapacityError):
        oracle.publish(scope, perspective(content="雪" * 200), "x")
    with pytest.raises(TemporalCapacityError):
        oracle.publish(
            scope,
            TemporalPerspective(
                ts(1),
                (
                    TemporalFact("a", "a", "1", ts(1)),
                    TemporalFact("b", "b", "2", ts(1)),
                ),
            ),
            "y",
        )
    assert backend.status("n")["revision"] == 0


def tamper_payload(path, transform):
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TRIGGER jth_perspectives_update")
        payload = conn.execute("SELECT payload FROM jth_perspectives").fetchone()[0]
        changed = transform(payload)
        conn.execute(
            "UPDATE jth_perspectives SET payload=?,digest=?",
            (changed, hashlib.sha256(changed.encode()).hexdigest()),
        )
        conn.execute(
            next(
                s
                for s in SCHEMA
                if s.startswith("CREATE TRIGGER jth_perspectives_update ")
            )
        )


@pytest.mark.parametrize(
    "transform",
    [
        lambda p: p.replace(
            '"content":"value"', '"content":"value","content":"duplicate"'
        ),
        lambda p: p.replace('"content":"value"', '"content":NaN'),
        lambda p: p.replace('"content":"value"', '"content":Infinity'),
        lambda p: p.replace('"content":"value"', '"content":' + "9" * 5000),
        lambda p: p.replace('"single_valued":true', '"single_valued":1'),
        lambda p: p.replace(
            '"valid_to":null', '"valid_to":"2025-01-01T00:00:00.000000Z"'
        ),
    ],
)
def test_reloaded_json_is_strict_even_with_matching_digest(tmp_path, transform):
    path = tmp_path / "history.db"
    _, scope, backend, oracle = setup(path)
    oracle.publish(scope, perspective(), "x")
    tamper_payload(path, transform)
    original = path.read_bytes()
    with pytest.raises(TemporalStorageError):
        backend.read_perspectives("n")
    assert path.read_bytes() == original
    with pytest.raises(TemporalStorageError):
        SQLiteTemporalHistory(path)
    with pytest.raises(TemporalStorageError):
        backend.backup(tmp_path / "corrupt-backup.db")
    assert not (tmp_path / "corrupt-backup.db").exists()
    with pytest.raises(TemporalStorageError):
        SQLiteTemporalHistory.restore(path, tmp_path / "corrupt-restore.db")
    assert not (tmp_path / "corrupt-restore.db").exists()
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "corruption", ["future", "missing_trigger", "foreign", "bytes", "empty"]
)
def test_schema_and_source_fail_closed_without_changes(tmp_path, corruption):
    path = tmp_path / "history.db"
    SQLiteTemporalHistory(path)
    if corruption == "bytes":
        path.write_bytes(b"not a sqlite database")
    elif corruption == "empty":
        path.write_bytes(b"")
    else:
        with sqlite3.connect(path) as conn:
            if corruption == "future":
                conn.execute("PRAGMA user_version=99")
            elif corruption == "missing_trigger":
                conn.execute("DROP TRIGGER jth_append")
            else:
                conn.execute("CREATE TABLE unexpected(x)")
    before = path.read_bytes()
    with pytest.raises(TemporalStorageError):
        SQLiteTemporalHistory(path)
    with pytest.raises(TemporalStorageError):
        SQLiteTemporalHistory.restore(path, tmp_path / "restore.db")
    assert not (tmp_path / "restore.db").exists()
    assert path.read_bytes() == before


def test_immutable_db_constraints_and_diagnostics(tmp_path):
    path = tmp_path / "history.db"
    _, scope, backend, oracle = setup(path)
    oracle.publish(scope, perspective(), "x")
    with sqlite3.connect(path) as conn:
        for sql in (
            "UPDATE jth_perspectives SET payload='{}'",
            "DELETE FROM jth_perspectives",
            "UPDATE jth_operations SET revision=2",
            "DELETE FROM jth_operations",
            "UPDATE jth_meta SET version=2",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(sql)
        payload, digest = conn.execute(
            "SELECT payload,digest FROM jth_perspectives"
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO jth_perspectives VALUES ('n',3,?,?,?)",
                ("2026-02-01T00:00:00.000000Z", payload, digest),
            )
        source_id = conn.execute("SELECT sqlite_source_id()").fetchone()[0]
    diagnostic = backend.diagnostics()
    assert diagnostic["journal_mode"] == "delete"
    assert diagnostic["synchronous"] == 2 and diagnostic["foreign_keys"] == 1
    assert diagnostic["busy_timeout_ms"] == 250
    assert diagnostic["sqlite_source_id"] == source_id
    assert str(tmp_path) not in str(diagnostic)
    with pytest.raises(TemporalSchemaError):
        SQLiteTemporalHistory(path, journal_mode="WAL")


def test_existing_wal_refused_without_journal_conversion(tmp_path):
    path = tmp_path / "wal.db"
    # Empty WAL database: no WAL writes on this potentially affected runtime.
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    before = path.read_bytes()
    with pytest.raises(TemporalSchemaError):
        SQLiteTemporalHistory(path)
    assert path.read_bytes() == before
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_before_commit_rolls_back_and_after_commit_reconciles_same_key(tmp_path):
    path = tmp_path / "history.db"
    stage = ["before_commit"]

    def fault(where):
        if where == stage[0]:
            raise RuntimeError("private content /host/path")

    _, scope, backend, oracle = setup(path, fault_hook=fault)
    with pytest.raises(TemporalStorageError, match="^Temporal storage unavailable$"):
        oracle.publish(scope, perspective(), "same")
    assert backend.status("n")["revision"] == 0
    stage[0] = "after_commit"
    receipt = oracle.publish(scope, perspective(), "same")
    assert receipt.revision == 1
    _, scope2, reopened, oracle2 = setup(path)
    assert oracle2.publish(scope2, perspective(), "same") == receipt
    assert reopened.status("n")["revision"] == 1


def test_uncertain_ack_does_not_repeat_effects(tmp_path, monkeypatch):
    _, scope, backend, oracle = setup(tmp_path / "history.db")
    original_receipt = backend._receipt
    reads = [0]

    def unavailable(conn, namespace, key, digest):
        reads[0] += 1
        if reads[0] > 1:
            raise TemporalStorageError()
        return original_receipt(conn, namespace, key, digest)

    def fault(stage):
        if stage == "after_commit":
            raise RuntimeError()

    backend._fault_hook = fault
    monkeypatch.setattr(backend, "_receipt", unavailable)
    with pytest.raises(TemporalAcknowledgementUncertain):
        oracle.publish(scope, perspective(), "key")
    assert backend.status("n")["revision"] == 1
    monkeypatch.setattr(backend, "_receipt", original_receipt)
    assert oracle.publish(scope, perspective(), "key").revision == 1


def test_scope_issuance_repo_checks_and_revocation_boundaries(tmp_path, monkeypatch):
    authority, scope, backend, oracle = setup(tmp_path / "history.db")
    forged = ScopeContext("p", "n", ("repo",), scope.authorization_version, "r")
    original_connection = backend._connection

    def no_acquisition(**kwargs):
        pytest.fail("unauthorized storage acquisition")

    monkeypatch.setattr(backend, "_connection", no_acquisition)
    with pytest.raises(ScopeDenied):
        oracle.publish(forged, perspective(), "x")
    with pytest.raises(ScopeDenied):
        oracle.publish(scope, perspective(repo_id="denied", snapshot_id="main"), "x")
    monkeypatch.setattr(backend, "_connection", original_connection)
    backend._fault_hook = lambda stage: (
        authority.revoke("p", "n") if stage == "before_commit" else None
    )
    with pytest.raises(ScopeDenied):
        oracle.publish(scope, perspective(), "x")
    assert backend.status("n")["revision"] == 0
    authority.grant("p", "n", ["repo"])
    scope = authority.context("p", "n", "new")
    backend._fault_hook = lambda stage: (
        authority.revoke("p", "n") if stage == "after_commit" else None
    )
    with pytest.raises(ScopeDenied):
        oracle.publish(scope, perspective(), "x")
    assert backend.status("n")["revision"] == 1  # accepted commit is not undone
    authority.grant("p", "n", ["repo"])
    scope = authority.context("p", "n", "retry")
    assert oracle.publish(scope, perspective(), "x").revision == 1


def test_backup_restore_preserves_candidate_era_and_never_overwrites(tmp_path):
    path = tmp_path / "candidate.db"
    _, scope, backend, oracle = setup(path)
    first = oracle.publish(scope, perspective(), "first")
    backup1 = backend.backup(tmp_path / "before.db")
    candidate = oracle.publish(scope, perspective(2), "candidate")
    backup2 = backend.backup(tmp_path / "after.db")
    restored = SQLiteTemporalHistory.restore(backup2, tmp_path / "restored.db")
    assert restored.read_perspectives("n") == backend.read_perspectives("n")
    restored_oracle = DurableTemporalOracle(oracle.authority, restored)
    assert restored_oracle.publish(scope, perspective(), "first") == first
    assert restored_oracle.publish(scope, perspective(2), "candidate") == candidate
    before = path.read_bytes()
    with pytest.raises(TemporalStorageError):
        SQLiteTemporalHistory.restore(backup1, path)
    assert path.read_bytes() == before
    with pytest.raises(TemporalStorageError):
        backend.backup(backup2)
    oracle.publish(scope, perspective(3), "later")
    assert backend.status("n")["revision"] == 3
    assert restored.status("n")["revision"] == 2
    with pytest.raises(TemporalStorageError):
        SQLiteTemporalHistory.restore(tmp_path / "missing", tmp_path / "new")
    assert not (tmp_path / "missing").exists()


def test_backup_deadline_removes_partial_destination(tmp_path):
    _, scope, backend, oracle = setup(tmp_path / "history.db")
    oracle.publish(scope, perspective(), "x")
    destination = tmp_path / "partial.db"
    with pytest.raises(TemporalStorageError):
        backend.backup(destination, timeout_seconds=1e-12)
    assert not destination.exists()


def test_backup_during_publication_is_a_consistent_snapshot(tmp_path, monkeypatch):
    _, scope, backend, oracle = setup(tmp_path / "history.db", busy_timeout_ms=2000)
    for month in range(1, 5):
        oracle.publish(scope, perspective(month, "x" * 200_000), str(month))
    started, published = Event(), Event()
    real_connect = sqlite3.connect

    class InterleavedBackup(sqlite3.Connection):
        def backup(self, target, **kwargs):
            original_progress = kwargs["progress"]
            triggered = False

            def interleave(status, remaining, total):
                nonlocal triggered
                original_progress(status, remaining, total)
                if not triggered and remaining:
                    triggered = True
                    started.set()
                    assert published.wait(5)

            kwargs["progress"] = interleave
            return super().backup(target, **kwargs)

    def connect(*args, **kwargs):
        return real_connect(*args, **kwargs, factory=InterleavedBackup)

    def writer():
        assert started.wait(5)
        receipt = oracle.publish(scope, perspective(5, "new"), "during-backup")
        published.set()
        return receipt

    monkeypatch.setattr(sqlite3, "connect", connect)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(writer)
        backup = backend.backup(tmp_path / "backup.db")
        assert pending.result(timeout=5).revision == 5
    restored = SQLiteTemporalHistory.restore(backup, tmp_path / "restored.db")
    assert restored.read_perspectives("n") == backend.read_perspectives("n")


def test_j02_database_coexistence_requires_separate_files(tmp_path):
    SQLiteDurableStore = pytest.importorskip("jitmind.storage").SQLiteDurableStore

    j02_path = tmp_path / "memory.db"
    store = SQLiteDurableStore(j02_path)
    before = j02_path.read_bytes()
    with pytest.raises(TemporalSchemaError):
        SQLiteTemporalHistory(j02_path)
    assert j02_path.read_bytes() == before
    history = SQLiteTemporalHistory(tmp_path / "temporal.db")
    history.append_perspective("n", 0, perspective())
    assert store.status("n")["revision"] == 0
    assert history.status("n")["revision"] == 1
    assert SQLiteDurableStore(j02_path).diagnostics()["journal_mode"] == "delete"


def test_busy_is_bounded_and_error_does_not_disclose_paths(tmp_path):
    _, scope, backend, oracle = setup(tmp_path / "private-name.db", busy_timeout_ms=1)
    with sqlite3.connect(backend.path, isolation_level=None) as locked:
        locked.execute("BEGIN IMMEDIATE")
        with pytest.raises(
            TemporalStorageError, match="^Temporal storage unavailable$"
        ):
            oracle.publish(scope, perspective(), "secret-key")
        locked.execute("ROLLBACK")
    assert backend.status("n")["revision"] == 0
    assert oracle.publish(scope, perspective(), "secret-key").revision == 1


@pytest.mark.parametrize("committed", [False, True])
def test_commit_error_reconciles_without_repeating_insert(
    tmp_path, monkeypatch, committed
):
    _, scope, backend, oracle = setup(tmp_path / "history.db")
    real_connect = sqlite3.connect
    attempted = []
    injected = [False]

    class CommitError(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql.startswith("INSERT INTO jth_perspectives"):
                attempted.append(1)
            if sql == "COMMIT" and not injected[0]:
                injected[0] = True
                if committed:
                    super().execute(sql, parameters)
                raise sqlite3.OperationalError("private database /host/path")
            return super().execute(sql, parameters)

    def connect(*args, **kwargs):
        return real_connect(*args, **kwargs, factory=CommitError)

    monkeypatch.setattr(sqlite3, "connect", connect)
    if committed:
        receipt = oracle.publish(scope, perspective(), "same")
        assert receipt.revision == 1
    else:
        with pytest.raises(TemporalAcknowledgementUncertain):
            oracle.publish(scope, perspective(), "same")
    assert attempted == [1]
    assert backend.status("n")["revision"] == int(committed)
    assert oracle.publish(scope, perspective(), "same").revision == 1
    assert len(attempted) == (1 if committed else 2)


def test_response_lost_after_commit_reconciles_on_restart(tmp_path):
    path = tmp_path / "history.db"

    def lost(stage):
        if stage == "after_commit":
            raise KeyboardInterrupt  # delivery never reaches the caller

    _, scope, backend, oracle = setup(path, fault_hook=lost)
    with pytest.raises(KeyboardInterrupt):
        oracle.publish(scope, perspective(), "lost-response")
    assert backend.status("n")["revision"] == 1
    _, new_scope, reopened, restarted = setup(path)
    receipt = restarted.publish(new_scope, perspective(), "lost-response")
    assert receipt.revision == 1
    assert restarted.publish(new_scope, perspective(), "lost-response") == receipt
    assert reopened.status("n")["revision"] == 1


def test_missing_database_is_not_recreated_by_reads_or_writes(tmp_path):
    path = tmp_path / "history.db"
    _, scope, backend, oracle = setup(path)
    path.unlink()
    with pytest.raises(TemporalStorageError):
        backend.read_perspectives("n")
    with pytest.raises(TemporalStorageError):
        oracle.publish(scope, perspective(), "x")
    assert not path.exists()

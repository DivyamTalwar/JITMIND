"""Synthetic fixtures only; no provider or real transcript evidence."""

import multiprocessing as mp
import os
import sqlite3
import sys
from dataclasses import replace

import pytest

from jitmind.checkpoints import (
    CommitUncertain,
    EventConflict,
    IdempotencyConflict,
    InvalidInput,
    RevisionConflict,
    SchemaMismatch,
    SourceChanged,
    SQLiteCheckpointLedger,
    StorageError,
    UnsafeRange,
)
from jitmind.scope import ScopeAuthority, ScopeDenied

STAMP = "2026-09-28T10:00:00+00:00"


def setup_ledger(path, namespace="space", **kwargs):
    authority = ScopeAuthority()
    authority.grant("alice", namespace, [])
    return SQLiteCheckpointLedger(path, authority, **kwargs), authority.context(
        "alice", namespace, "r"
    )


def append(
    ledger, scope, seq, *, stream="main", kind="message", call=None, payload=None
):
    return ledger.append(
        scope,
        stream,
        seq,
        event_id=f"e{seq}",
        payload={"seq": seq} if payload is None else payload,
        observed_time=STAMP,
        kind=kind,
        tool_call_id=call,
    )


def prepared(path):
    ledger, scope = setup_ledger(path)
    append(ledger, scope, 0)
    append(ledger, scope, 1, kind="tool_start", call="tool")
    append(ledger, scope, 2, kind="tool_result", call="tool")
    snapshot = ledger.snapshot(scope, "main", 0, 2)
    return (
        ledger,
        scope,
        snapshot.draft(
            summary="Synthetic summary",
            idempotency_key="k",
            model_metadata={"model": "fixture"},
        ),
    )


def test_publish_replay_restart_history_and_raw_retention(tmp_path):
    path = tmp_path / "ledger.db"
    ledger, scope, draft = prepared(path)
    checkpoint = ledger.publish(scope, draft)
    assert checkpoint.revision == 1
    assert checkpoint.validation == "validated_range"
    reopened, again = setup_ledger(path)
    assert reopened.publish(again, draft) == checkpoint
    assert reopened.history(again, "main") == (checkpoint,)
    assert len(reopened.snapshot(again, "main", 0, 2).events) == 3
    with pytest.raises(IdempotencyConflict):
        reopened.publish(again, replace(draft, summary="Different"))
    with pytest.raises(RevisionConflict):
        reopened.publish(again, replace(draft, idempotency_key="other"))
    snapshot = reopened.snapshot(again, "main", 0, 2)
    second = reopened.publish(
        again, snapshot.draft(summary="Next", idempotency_key="next")
    )
    assert second.revision == 2
    assert (
        reopened.publish(again, draft) == checkpoint
    )  # historical replay after newer revisions
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert db.execute("SELECT COUNT(*) FROM jitmind_cp_events").fetchone()[0] == 3
        for table in ("jitmind_cp_events", "jitmind_cp_history"):
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(f"DELETE FROM {table}")
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(f"UPDATE {table} SET namespace='other'")


def test_missing_event_and_identity_conflicts(tmp_path):
    ledger, scope = setup_ledger(tmp_path / "ledger.db")
    first = append(ledger, scope, 0)
    assert append(ledger, scope, 0) == first
    with pytest.raises(EventConflict):
        append(ledger, scope, 0, payload={"changed": True})
    with pytest.raises(EventConflict):
        ledger.append(scope, "main", 1, event_id="e0", payload={}, observed_time=STAMP)
    append(ledger, scope, 2)
    with pytest.raises(UnsafeRange):
        ledger.snapshot(scope, "main", 0, 2)
    with pytest.raises(UnsafeRange, match="boundary history"):
        ledger.snapshot(scope, "main", 2, 2)
    with pytest.raises(EventConflict):
        append(ledger, scope, 1)  # no backfill that can rewrite a historical boundary
    with pytest.raises(UnsafeRange):
        ledger.snapshot(scope, "missing-stream", 0, 0)


@pytest.mark.parametrize("terminal", ["tool_result", "tool_cancel", "tool_timeout"])
def test_tool_boundaries_and_explicit_terminals(tmp_path, terminal):
    ledger, scope = setup_ledger(tmp_path / "ledger.db")
    append(ledger, scope, 0, kind="tool_start", call="tool")
    append(ledger, scope, 1)
    with pytest.raises(UnsafeRange):
        ledger.snapshot(scope, "main", 0, 1)
    append(ledger, scope, 2, kind=terminal, call="tool")
    # A late result never retroactively closes 0..1, or permits 1..2.
    for start, end in [(0, 1), (1, 2), (2, 2), (1, 1)]:
        with pytest.raises(UnsafeRange):
            ledger.snapshot(scope, "main", start, end)
    snap = ledger.snapshot(scope, "main", 0, 2)
    assert (
        ledger.publish(
            scope, snap.draft(summary="closed", idempotency_key="k")
        ).revision
        == 1
    )
    append(ledger, scope, 3)
    assert ledger.snapshot(scope, "main", 3, 3).events[0].sequence == 3
    with pytest.raises(UnsafeRange):
        append(ledger, scope, 4, kind="tool_result", call="tool")
    with pytest.raises(UnsafeRange):
        append(ledger, scope, 4, kind="tool_start", call="tool")
    with pytest.raises(UnsafeRange):
        append(ledger, scope, 4, kind="tool_result", call="unknown")


def test_overlapping_calls_and_midrange_closed_call(tmp_path):
    ledger, scope = setup_ledger(tmp_path / "ledger.db")
    for seq, kind, call in [
        (0, "message", None),
        (1, "tool_start", "a"),
        (2, "tool_start", "b"),
        (3, "tool_result", "a"),
        (4, "tool_cancel", "b"),
        (5, "message", None),
    ]:
        append(ledger, scope, seq, kind=kind, call=call)
    assert len(ledger.snapshot(scope, "main", 0, 5).events) == 6
    for start, end in [(1, 3), (2, 4), (3, 5)]:
        with pytest.raises(UnsafeRange):
            ledger.snapshot(scope, "main", start, end)


@pytest.mark.parametrize(
    "start,end",
    [(True, 1), (0, False), (-1, 0), (1, 0), (0, 2**63), (0, 100000), (0.0, 1)],
)
def test_range_validation(tmp_path, start, end):
    ledger, scope = setup_ledger(tmp_path / "ledger.db")
    with pytest.raises(InvalidInput):
        ledger.snapshot(scope, "main", start, end)


def test_source_digest_schema_metadata_revision_validation(tmp_path):
    ledger, scope, draft = prepared(tmp_path / "ledger.db")
    with pytest.raises(SourceChanged):
        ledger.publish(scope, replace(draft, source_digest="0" * 64))
    for mutation in [
        {"source_schema": "other"},
        {"expected_revision": True},
        {"start": True},
        {"model_metadata_json": '{"not": NaN}'},
        {"model_metadata_json": "[]"},
        {"summary": ""},
        {"source_digest": "z" * 64},
    ]:
        with pytest.raises(InvalidInput):
            ledger.publish(scope, replace(draft, **mutation))
    assert ledger.history(scope, "main") == ()


def test_namespace_stream_isolation_and_auth_before_disk(tmp_path):
    path = tmp_path / "ledger.db"
    ledger, scope, draft = prepared(path)
    ledger.authority.grant("alice", "other", [])
    other = ledger.authority.context("alice", "other", "request")
    append(ledger, other, 0)
    own = ledger.snapshot(other, "main", 0, 0).draft(
        summary="Other", idempotency_key="k"
    )
    assert (
        ledger.publish(other, own).revision
        == ledger.publish(scope, draft).revision
        == 1
    )
    with pytest.raises(InvalidInput):
        ledger.publish(other, draft)
    append(ledger, scope, 0, stream="second")
    with pytest.raises(SourceChanged):
        ledger.publish(
            scope, replace(draft, stream="second", end=0, idempotency_key="new")
        )
    ledger.authority.revoke("alice", "space")
    ledger._connect = lambda: pytest.fail("Revoked scope accessed disk")
    for operation in [
        lambda: ledger.snapshot(scope, "main", 0, 2),
        lambda: ledger.publish(scope, draft),
        lambda: ledger.history(scope, "main"),
        lambda: append(ledger, scope, 3),
    ]:
        with pytest.raises(ScopeDenied):
            operation()


def test_payload_copy_and_input_rejection(tmp_path):
    ledger, scope = setup_ledger(tmp_path / "ledger.db")
    payload = {"nested": [1]}
    event = append(ledger, scope, 0, payload=payload)
    payload["nested"].append(2)
    event.payload["nested"].append(3)
    assert event.payload == {"nested": [1]}
    for value in [float("nan"), {1: "a"}, (1, 2), {"bad": object()}]:
        with pytest.raises(InvalidInput):
            append(ledger, scope, 1, payload=value)
    for changes in [
        {"sequence": True},
        {"kind": "unknown"},
        {"observed_time": "yesterday"},
        {"observed_time": "2026-09-28"},
        {"tool_call_id": "stray"},
    ]:
        kwargs = {
            "sequence": 1,
            "event_id": "new",
            "payload": {},
            "observed_time": STAMP,
        }
        kwargs.update(changes)
        with pytest.raises(InvalidInput):
            ledger.append(scope, "main", **kwargs)


def publisher_process(path, draft, barrier, output):
    try:
        ledger, scope = setup_ledger(path)
        barrier.wait(timeout=15)
        output.put(("ok", ledger.publish(scope, draft).revision))
    except RevisionConflict:
        output.put(("conflict", None))
    except Exception as error:  # noqa: BLE001 - transport child failures to the asserting parent
        output.put(("error", type(error).__name__))


def test_two_process_publishers_compare_and_swap(tmp_path):
    path = tmp_path / "ledger.db"
    ledger, scope, draft = prepared(path)
    ctx = mp.get_context("spawn")
    barrier, output = ctx.Barrier(2), ctx.Queue()
    processes = [
        ctx.Process(
            target=publisher_process,
            args=(path, replace(draft, idempotency_key=str(i)), barrier, output),
        )
        for i in range(2)
    ]
    for process in processes:
        process.start()
    try:
        results = [output.get(timeout=25) for _ in processes]
        for process in processes:
            process.join(25)
            assert process.exitcode == 0
        assert sorted(status for status, _ in results) == ["conflict", "ok"]
        assert len(ledger.history(scope, "main")) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(5)


def crash_process(path, draft, after):
    ledger, scope = setup_ledger(path)

    class CrashConnection(sqlite3.Connection):
        def commit(self):
            if after:
                super().commit()
            os._exit(42)

    ledger._connect = lambda: sqlite3.connect(
        str(path), isolation_level=None, factory=CrashConnection
    )
    ledger.publish(scope, draft)


@pytest.mark.parametrize("after", [False, True])
def test_real_process_crash_and_exact_retry(tmp_path, after):
    path = tmp_path / "ledger.db"
    _, _, draft = prepared(path)
    process = mp.get_context("spawn").Process(
        target=crash_process, args=(path, draft, after)
    )
    process.start()
    try:
        process.join(25)
        assert process.exitcode == 42
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
    reopened, scope = setup_ledger(path)
    assert len(reopened.history(scope, "main")) == int(after)
    assert reopened.publish(scope, draft).revision == 1
    assert len(reopened.history(scope, "main")) == 1


def test_busy_is_bounded_and_errors_sanitized(tmp_path):
    path = tmp_path / "sensitive.db"
    ledger, scope, draft = prepared(path)
    ledger.busy_timeout_ms = 10
    with sqlite3.connect(path) as db:
        db.execute("BEGIN IMMEDIATE")
        with pytest.raises(StorageError) as error:
            ledger.publish(scope, draft)
        assert str(path) not in str(error.value)
    assert ledger.publish(scope, draft).revision == 1


def test_rollback_and_commit_ack_failure(tmp_path):
    path = tmp_path / "ledger.db"
    ledger, scope, draft = prepared(path)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TRIGGER reject_history BEFORE INSERT ON jitmind_cp_history BEGIN SELECT RAISE(ABORT, 'sensitive details'); END"
        )
    with pytest.raises(StorageError, match="^Checkpoint storage operation failed$"):
        ledger.publish(scope, draft)
    assert ledger.snapshot(scope, "main", 0, 2).expected_revision == 0
    with sqlite3.connect(path) as db:
        db.execute("DROP TRIGGER reject_history")

    class UncertainConnection(sqlite3.Connection):
        def commit(self):
            super().commit()
            raise sqlite3.OperationalError("secret path")

    ledger._connect = lambda: sqlite3.connect(
        str(path), isolation_level=None, factory=UncertainConnection
    )
    with pytest.raises(CommitUncertain):
        ledger.publish(scope, draft)
    reopened, scope = setup_ledger(path)
    assert reopened.publish(scope, draft).revision == 1


def test_schema_policy_and_foreign_keys(tmp_path):
    path = tmp_path / "ledger.db"
    ledger, _ = setup_ledger(path)
    with ledger._transaction() as db:
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert db.execute("PRAGMA synchronous").fetchone()[0] == 2
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE durable_memory_fixture (id INTEGER)")
        db.execute("PRAGMA user_version=97")
        db.execute("UPDATE jitmind_cp_meta SET version=2")
    with pytest.raises(SchemaMismatch):
        setup_ledger(path)
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 97


def test_optional_module_not_imported_by_core():
    # A separate interpreter avoids pytest importing the optional module above.
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import jitmind; assert 'jitmind.checkpoints' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_missing_or_modified_raw_source_cannot_be_replaced_by_summary(tmp_path):
    path = tmp_path / "ledger.db"
    ledger, scope, draft = prepared(path)
    # Deliberately simulate an out-of-band damaged database, bypassing immutability.
    with sqlite3.connect(path) as db:
        db.execute("DROP TRIGGER jitmind_cp_events_delete")
        db.execute("DELETE FROM jitmind_cp_events WHERE sequence=1")
    with pytest.raises(UnsafeRange):
        ledger.publish(scope, draft)
    assert ledger.history(scope, "main") == ()


def test_changed_source_and_corrupted_payload_are_distinct(tmp_path):
    from hashlib import sha256

    path = tmp_path / "ledger.db"
    ledger, scope, draft = prepared(path)
    with sqlite3.connect(path) as db:
        db.execute("DROP TRIGGER jitmind_cp_events_update")
        db.execute(
            "UPDATE jitmind_cp_events SET payload_json='{}',payload_digest=? WHERE sequence=0",
            (sha256(b"{}").hexdigest(),),
        )
    with pytest.raises(SourceChanged):
        ledger.publish(scope, draft)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE jitmind_cp_events SET payload_digest='bad' WHERE sequence=0")
    with pytest.raises(StorageError, match="digest mismatch"):
        ledger.snapshot(scope, "main", 0, 2)


def test_forged_open_draft_cannot_publish_after_late_result(tmp_path):
    from jitmind.checkpoints import Draft
    from jitmind.checkpoints.ledger import SOURCE_SCHEMA

    ledger, scope = setup_ledger(tmp_path / "ledger.db")
    append(ledger, scope, 0, kind="tool_start", call="call")
    forged = Draft(
        scope.namespace_id, "main", 0, 0, SOURCE_SCHEMA, "0" * 64, 0, "key", "summary"
    )
    with pytest.raises(UnsafeRange):
        ledger.publish(scope, forged)
    append(ledger, scope, 1, kind="tool_result", call="call")
    with pytest.raises(UnsafeRange):
        ledger.publish(scope, forged)
    assert ledger.history(scope, "main") == ()


def test_authorization_rechecked_and_writer_rolled_back(tmp_path):
    ledger, scope, draft = prepared(tmp_path / "ledger.db")
    original = ledger.authority.require
    calls = 0

    def revoke_on_last_check(context, repo_id=None):
        nonlocal calls
        calls += 1
        if calls == 3:  # before commit, after insertion
            ledger.authority.revoke("alice", "space")
        original(context, repo_id)

    ledger.authority.require = revoke_on_last_check
    with pytest.raises(ScopeDenied):
        ledger.publish(scope, draft)
    ledger.authority.require = original
    ledger.authority.grant("alice", "space", [])
    current = ledger.authority.context("alice", "space", "new")
    assert ledger.history(current, "main") == ()
    assert ledger.snapshot(current, "main", 0, 2).expected_revision == 0


def trace_reads(ledger):
    statements = []
    original = ledger._connect

    def connect():
        db = original()
        db.set_trace_callback(statements.append)
        return db

    ledger._connect = connect
    return statements


def payload_reads(statements, table):
    return [sql for sql in statements if sql.startswith(f"SELECT * FROM {table}")]


@pytest.mark.parametrize("operation", ["snapshot", "publish"])
def test_aggregate_rejected_before_payload_query(tmp_path, operation):
    from jitmind.checkpoints import CapacityExceeded, Draft
    from jitmind.checkpoints.ledger import SOURCE_SCHEMA

    ledger, scope = setup_ledger(tmp_path / "ledger.db")
    for seq in range(16):
        append(ledger, scope, seq, payload={"text": "x" * 500_000})
    # A smaller range remains usable with the same data.
    assert len(ledger.snapshot(scope, "main", 0, 2).events) == 3
    statements = trace_reads(ledger)
    with pytest.raises(CapacityExceeded, match="Source acquisition"):
        if operation == "snapshot":
            ledger.snapshot(scope, "main", 0, 15)
        else:
            ledger.publish(scope, Draft(scope.namespace_id, "main", 0, 15,
                           SOURCE_SCHEMA, "0" * 64, 0, "k", "summary"))
    assert not payload_reads(statements, "jitmind_cp_events")
    assert ledger.history(scope, "main") == ()
    assert ledger.snapshot(scope, "main", 0, 0).expected_revision == 0


@pytest.mark.parametrize("budget", ["count", "bytes"])
@pytest.mark.parametrize("operation", ["snapshot", "publish"])
def test_prefix_bounded_before_payload_query(tmp_path, monkeypatch, budget, operation):
    import jitmind.checkpoints.ledger as module
    from jitmind.checkpoints import CapacityExceeded

    ledger, scope = setup_ledger(tmp_path / "ledger.db")
    for seq in range(4):
        append(ledger, scope, seq,
               kind="tool_start" if seq % 2 == 0 else "tool_result",
               call=str(seq // 2) * 800)
    append(ledger, scope, 4)
    draft = ledger.snapshot(scope, "main", 4, 4).draft(summary="closed", idempotency_key="k")
    statements = trace_reads(ledger)
    if budget == "count":
        monkeypatch.setattr(module, "MAX_PREFIX_EVENTS", 4)
    else:
        monkeypatch.setattr(module, "MAX_ACQUIRED_BYTES", 2000)
    with pytest.raises(CapacityExceeded, match="Tool prefix"):
        if operation == "snapshot":
            ledger.snapshot(scope, "main", 4, 4)
        else:
            ledger.publish(scope, draft)
    assert not payload_reads(statements, "jitmind_cp_events")
    assert ledger.history(scope, "main") == ()


def test_prefix_count_exact_boundary_is_usable(tmp_path, monkeypatch):
    import jitmind.checkpoints.ledger as module

    ledger, scope = setup_ledger(tmp_path / "ledger.db")
    append(ledger, scope, 0, kind="tool_start", call="a")
    append(ledger, scope, 1, kind="tool_result", call="a")
    append(ledger, scope, 2)
    monkeypatch.setattr(module, "MAX_PREFIX_EVENTS", 3)
    assert ledger.snapshot(scope, "main", 2, 2).events[0].sequence == 2


def populate_history(ledger, scope, count, summary="summary"):
    append(ledger, scope, 0)
    results = []
    for index in range(count):
        draft = ledger.snapshot(scope, "main", 0, 0).draft(
            summary=summary, idempotency_key=f"key-{index}")
        results.append(ledger.publish(scope, draft))
    return tuple(results)


def test_history_byte_budget_pages_and_explicit_incomplete_status(tmp_path):
    from jitmind.checkpoints import HistoryIncomplete

    ledger, scope = setup_ledger(tmp_path / "ledger.db")
    expected = populate_history(ledger, scope, 16, "x" * 500_000)
    with pytest.raises(HistoryIncomplete) as error:
        ledger.history(scope, "main")
    first = error.value.page
    assert not first.complete
    assert 0 < len(first.checkpoints) < len(expected)
    assert first.next_revision == first.checkpoints[-1].revision
    results = list(first.checkpoints)
    cursor = first.next_revision
    while cursor is not None:
        page = ledger.history_page(scope, "main", after_revision=cursor)
        assert page.complete == (page.next_revision is None)
        results.extend(page.checkpoints)
        cursor = page.next_revision
    assert tuple(results) == expected
    empty = ledger.history_page(scope, "main", after_revision=16)
    assert empty.complete and empty.next_revision is None and empty.checkpoints == ()


def test_history_count_limit_and_small_pages(tmp_path):
    from jitmind.checkpoints import HistoryIncomplete

    ledger, scope = setup_ledger(tmp_path / "ledger.db")
    expected = populate_history(ledger, scope, 101)
    with pytest.raises(HistoryIncomplete) as error:
        ledger.history(scope, "main")
    assert len(error.value.page.checkpoints) == 100
    assert error.value.page.next_revision == 100
    assert ledger.history_page(scope, "main", after_revision=100).checkpoints == expected[100:]
    page = ledger.history_page(scope, "main", limit=2)
    assert page.checkpoints == expected[:2] and page.next_revision == 2
    for kwargs in ({"limit": True}, {"limit": 0}, {"limit": 101}, {"after_revision": -1}):
        with pytest.raises(InvalidInput):
            ledger.history_page(scope, "main", **kwargs)


@pytest.mark.parametrize("operation", ["history", "replay"])
def test_oversized_stored_history_not_fetched(tmp_path, operation):
    from jitmind.checkpoints import CapacityExceeded

    path = tmp_path / "ledger.db"
    ledger, scope, draft = prepared(path)
    ledger.publish(scope, draft)
    with sqlite3.connect(path) as db:
        db.execute("DROP TRIGGER jitmind_cp_history_update")
        db.execute("UPDATE jitmind_cp_history SET draft_json=?", ("x" * 4_000_001,))
    statements = trace_reads(ledger)
    with pytest.raises(CapacityExceeded):
        if operation == "history":
            ledger.history_page(scope, "main")
        else:
            ledger.publish(scope, draft)
    assert not payload_reads(statements, "jitmind_cp_history")


@pytest.mark.parametrize("operation", ["snapshot", "publish"])
def test_sql_deadline_interrupts_actual_scan_and_rolls_back(tmp_path, monkeypatch, operation):
    import jitmind.checkpoints.ledger as module
    from jitmind.checkpoints import CapacityExceeded

    path = tmp_path / "ledger.db"
    ledger, scope, draft = prepared(path)
    now = [0.0]
    callbacks = []
    monkeypatch.setattr(module, "monotonic", lambda: now[0])

    class ExpiringConnection(sqlite3.Connection):
        def set_progress_handler(self, callback, steps):
            if callback is None:
                return super().set_progress_handler(None, steps)

            def progress():
                result = callback()
                callbacks.append(result)
                return result

            return super().set_progress_handler(progress, steps)

        def execute(self, sql, parameters=()):
            if sql.startswith("SELECT COUNT(*)"):
                now[0] = 10.0
                # Real VM execution invokes the installed progress callback.
                super().execute("WITH RECURSIVE n(x) AS (VALUES(0) UNION ALL SELECT x+1 FROM n WHERE x<10000) SELECT sum(x) FROM n").fetchone()
            return super().execute(sql, parameters)

    original = ledger._connect
    ledger._connect = lambda: sqlite3.connect(path, isolation_level=None, factory=ExpiringConnection)
    with pytest.raises(CapacityExceeded, match="deadline"):
        if operation == "snapshot":
            ledger.snapshot(scope, "main", 0, 2)
        else:
            ledger.publish(scope, draft)
    assert 1 in callbacks
    ledger._connect = original
    now[0] = 0.0
    assert ledger.history(scope, "main") == ()
    assert ledger.snapshot(scope, "main", 0, 2).expected_revision == 0


def test_python_deadline_inside_publication_rolls_back(tmp_path, monkeypatch):
    import jitmind.checkpoints.ledger as module
    from jitmind.checkpoints import CapacityExceeded

    ledger, scope, draft = prepared(tmp_path / "ledger.db")
    now = [0.0]
    monkeypatch.setattr(module, "monotonic", lambda: now[0])
    original = module._digest

    def expire_on_payload(text):
        if text == '{"seq":0}':
            now[0] = 10.0
        return original(text)

    monkeypatch.setattr(module, "_digest", expire_on_payload)
    with pytest.raises(CapacityExceeded, match="deadline"):
        ledger.publish(scope, draft)
    monkeypatch.setattr(module, "_digest", original)
    now[0] = 0.0
    assert ledger.history(scope, "main") == ()
    assert ledger.snapshot(scope, "main", 0, 2).expected_revision == 0


def test_json_budget_rejects_before_serializer(tmp_path, monkeypatch):
    import jitmind.checkpoints.ledger as module
    from jitmind.checkpoints import CapacityExceeded

    ledger, scope = setup_ledger(tmp_path / "ledger.db")
    monkeypatch.setattr(module.json.JSONEncoder, "iterencode", lambda *args: pytest.fail("oversized input reached encoder"))
    with pytest.raises(CapacityExceeded):
        append(ledger, scope, 0, payload={"text": "x" * 4_000_001})


@pytest.mark.parametrize("stamp", ["2026-09-28T00:00:00Z", "2026-09-28T00:00:00+00:00"])
def test_utc_spelling_validates_on_310_style_parser_and_preserves_identity(tmp_path, monkeypatch, stamp):
    import jitmind.checkpoints.ledger as module
    from datetime import datetime

    class Python310Parser:
        @staticmethod
        def fromisoformat(text):
            assert not text.endswith("Z")
            return datetime.fromisoformat(text)

    ledger, scope = setup_ledger(tmp_path / "ledger.db")
    monkeypatch.setattr(module, "datetime", Python310Parser)
    event = ledger.append(scope, "main", 0, event_id="e0", payload={}, observed_time=stamp)
    assert event.observed_time == stamp
    assert ledger.snapshot(scope, "main", 0, 0).events[0].observed_time == stamp
    assert ledger.append(scope, "main", 0, event_id="e0", payload={}, observed_time=stamp) == event
    alternate = "2026-09-28T00:00:00+00:00" if stamp.endswith("Z") else "2026-09-28T00:00:00Z"
    with pytest.raises(EventConflict):
        ledger.append(scope, "main", 0, event_id="e0", payload={}, observed_time=alternate)


@pytest.mark.parametrize("stamp", ["2026-09-28T00:00:00", "2026-09-28Z", "2026-09-28T00:00:00ZZ", "2026-09-28T00:00:00+00:00Z", "2026-09-28T00:00:00z", "2026-02-30T00:00:00Z"])
def test_utc_normalization_still_rejects_invalid_times(tmp_path, stamp):
    ledger, scope = setup_ledger(tmp_path / "ledger.db")
    with pytest.raises(InvalidInput):
        ledger.append(scope, "main", 0, event_id="e0", payload={}, observed_time=stamp)


@pytest.mark.parametrize("timeout", [True, 0, -1, 30_001, 1.5])
def test_operation_deadline_configuration_validation(tmp_path, timeout):
    with pytest.raises(InvalidInput):
        setup_ledger(tmp_path / "ledger.db", operation_timeout_ms=timeout)


def test_publication_refuses_unpageable_row_and_rolls_back_revision(tmp_path, monkeypatch):
    import jitmind.checkpoints.ledger as module
    from jitmind.checkpoints import CapacityExceeded

    ledger, scope = setup_ledger(tmp_path / "ledger.db")
    append(ledger, scope, 0)
    draft = ledger.snapshot(scope, "main", 0, 0).draft(
        summary="x" * 2000, idempotency_key="k")
    monkeypatch.setattr(module, "MAX_ACQUIRED_BYTES", len(draft.canonical()) + 1)
    with pytest.raises(CapacityExceeded, match="history acquisition"):
        ledger.publish(scope, draft)
    assert ledger.history(scope, "main") == ()
    assert ledger.snapshot(scope, "main", 0, 0).expected_revision == 0


def test_oversized_stored_event_replay_not_fetched(tmp_path):
    from jitmind.checkpoints import CapacityExceeded

    path = tmp_path / "ledger.db"
    ledger, scope = setup_ledger(path)
    append(ledger, scope, 0)
    with sqlite3.connect(path) as db:
        db.execute("DROP TRIGGER jitmind_cp_events_update")
        db.execute("UPDATE jitmind_cp_events SET payload_json=?", ("x" * 4_000_001,))
    statements = trace_reads(ledger)
    with pytest.raises(CapacityExceeded, match="Event replay acquisition"):
        append(ledger, scope, 0)
    assert not payload_reads(statements, "jitmind_cp_events")

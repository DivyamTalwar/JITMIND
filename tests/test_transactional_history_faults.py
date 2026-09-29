"""Disposable process/storage failures against the actual library."""

import multiprocessing
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_transactional_history import FEB1, JAN1, MAR1, legacy_store, proposal

from jitmind.storage import (
    AcknowledgementUncertain,
    CapacityExceeded,
    IngestRequest,
    InvalidRequest,
    SQLiteDurableStore,
    StorageFailure,
)

STAGES = [
    "before_transaction",
    "after_page_insert",
    "after_fact_insert",
    "after_history_insert",
    "after_outbox_receipt",
    "before_commit",
    "after_commit",
]


def write(store, key="key"):
    return store.ingest(
        IngestRequest.create("team", key, "3"),
        lambda _: proposal("3", start=JAN1, end=None),
        max_replans=5,
    )


def counts(path):
    with sqlite3.connect(path) as conn:
        return {
            table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in (
                "facts",
                "pages",
                "operations",
                "outbox",
                "history_versions",
                "history_effects",
                "history_revisions",
                "history_pages",
            )
        }


@pytest.mark.parametrize("stage", STAGES)
def test_fault_atomicity_and_exact_retry(tmp_path, stage):
    def fault(current):
        if current == stage:
            raise RuntimeError("private fault data")

    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JAN1, fault_hook=fault)
    with pytest.raises(
        AcknowledgementUncertain if stage == "after_commit" else StorageFailure
    ):
        write(store)
    assert set(counts(store.path).values()) == {int(stage == "after_commit")}
    reopened = SQLiteDurableStore(store.path, clock=lambda: JAN1)
    receipt = write(reopened)
    assert write(reopened) == receipt
    assert set(counts(store.path).values()) == {1}


def die_at(path, stage):
    def fault(current):
        if current == stage:
            os._exit(71)

    write(SQLiteDurableStore(path, clock=lambda: JAN1, fault_hook=fault))


@pytest.mark.parametrize("stage", STAGES)
def test_actual_process_death_at_each_write_stage(tmp_path, stage):
    path = tmp_path / "db"
    SQLiteDurableStore(path)
    child = multiprocessing.get_context("spawn").Process(
        target=die_at, args=(path, stage)
    )
    child.start()
    child.join(20)
    assert not child.is_alive()
    assert child.exitcode == 71
    reopened = SQLiteDurableStore(path, clock=lambda: JAN1)
    assert set(counts(path).values()) == {int(stage == "after_commit")}
    receipt = write(reopened)
    assert write(SQLiteDurableStore(path)) == receipt
    assert set(counts(path).values()) == {1}
    assert reopened.snapshot_at("team", revision=1).entries[0].content == "3"


def test_forced_history_storage_failure_never_commits_fact(tmp_path, monkeypatch):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JAN1)

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(store, "_history_version", fail)
    with pytest.raises(StorageFailure):
        write(store)
    assert set(counts(store.path).values()) == {0}
    assert store.snapshot_at("team").coverage == "unavailable"


def test_concurrent_same_key_and_distinct_writers(tmp_path):
    path = tmp_path / "db"
    SQLiteDurableStore(path)

    def run(key):
        return write(
            SQLiteDurableStore(path, busy_timeout_ms=5000, clock=lambda: JAN1), key
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(run, ["same"] * 4))
        distinct = list(pool.map(run, ["a", "b", "c", "d"]))
    assert len({r.event_id for r in receipts}) == 1
    assert sorted(r.revision for r in distinct) == [2, 3, 4, 5]
    assert set(counts(path).values()) == {5}
    store = SQLiteDurableStore(path)
    assert store.snapshot_at("team", transaction_at=JAN1).revision == 5
    assert len(store.snapshot_at("team", revision=1).entries) == 1


def test_outbox_all_events_are_namespace_qualified_and_delivery_independent(tmp_path):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JAN1)
    first, second = write(store, "a"), write(store, "b")
    store.deliver_event("team", first.event_id)
    store.deliver_event("team", second.event_id)
    assert store.pending_events("team") == []
    events = store.outbox_events("team")
    assert tuple(e.revision for e in events) == (1, 2)
    assert events[0].memory_ids == (first.memory_id,)
    assert store.outbox_events("team", limit=1) == events[:1]
    assert store.outbox_events("team", after_revision=1) == events[1:]
    assert store.get_event("team", first.event_id) == events[0]
    assert store.get_event("other", first.event_id) is None
    assert store.outbox_events("other") == ()
    assert store.outbox_events("team", after_revision=2**63 - 1) == ()
    for kwargs in (
        {"limit": True},
        {"limit": 1001},
        {"after_revision": -1},
        {"after_revision": 2**63},
    ):
        with pytest.raises(InvalidRequest):
            store.outbox_events("team", **kwargs)


def test_backup_restore_retains_post_migration_writes_and_receipts(tmp_path):
    store = legacy_store(tmp_path / "db")
    first = write(store, "first")
    store.enable_history(quiesced=True, recorded_at=FEB1)
    store.clock = lambda: MAR1
    second = write(store, "second")
    store.deliver_event("team", first.event_id)
    before = store.snapshot_at("team")
    store.backup(tmp_path / "backup")
    restored = SQLiteDurableStore.restore(tmp_path / "backup", tmp_path / "restored")
    assert restored.snapshot_at("team") == before
    assert restored.outbox_events("team") == store.outbox_events("team")
    assert write(restored, "first") == first
    assert write(restored, "second") == second
    assert restored.snapshot_at("team", transaction_at=JAN1).coverage == "unavailable"
    assert restored.diagnostics()["journal_mode"] == "delete"
    assert restored.diagnostics()["synchronous"] == 2


def test_byte_and_processing_bounds_raise_explicit_capacity(tmp_path, monkeypatch):
    from jitmind.storage import history

    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JAN1)
    write(store)
    monkeypatch.setattr(history, "MAX_HISTORY_BYTES", 1)
    with pytest.raises(CapacityExceeded):
        store.snapshot_at("team")
    monkeypatch.setattr(history, "MAX_HISTORY_BYTES", 8 * 1024 * 1024)
    for i in range(40):
        write(store, str(i))
    monkeypatch.setattr(history, "MAX_HISTORY_STEPS", 1)
    with pytest.raises(CapacityExceeded):
        store.snapshot_at("team")


def test_backwards_recording_clock_rejected_without_partial_write(tmp_path):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: MAR1)
    write(store)
    before = counts(store.path)
    store.clock = lambda: JAN1
    with pytest.raises(InvalidRequest):
        write(store, "backwards")
    assert counts(store.path) == before

"""Disposable corruption fixtures exercise the actual bounded SQLite authority."""

import json
import sqlite3
from contextlib import contextmanager

import pytest
from test_transactional_history import FEB1, JAN1, JAN15, JAN20, legacy_store, proposal

from jitmind.storage import (
    CapacityExceeded,
    IngestRequest,
    InvalidRequest,
    SQLiteDurableStore,
    StorageFailure,
    ValidityCorrection,
    stage_legacy,
)


@contextmanager
def corrupt(path, table, action="update"):
    # Preserve exact DDL: reopening must see the real candidate schema.
    with sqlite3.connect(path) as conn:
        name = f"{table}_{action}"
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (name,)
        ).fetchone()[0]
        conn.execute(f"DROP TRIGGER {name}")
        yield conn
        conn.execute(ddl)


def seeded(tmp_path, *, meta=None):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JAN1)
    request = IngestRequest.create("n", "one", "PRIVATE_SENTINEL", meta)
    receipt = store.commit_proposal(request, 0, proposal(start=JAN1, end=None))
    return store, request, receipt


@pytest.mark.parametrize(
    "assignment",
    [
        "status='deleted'",
        "repo_id='other'",
        "snapshot_id='other'",
        "scope_valid=0",
        "single_valued=2",
        "valid_from='2099-01-01'",
        "valid_to='2000-01-01'",
        "fact_key='other'",
        "page_id='missing'",
    ],
)
def test_index_corruption_fails_before_private_payload_acquisition(
    tmp_path, monkeypatch, assignment
):
    from jitmind.storage import sqlite as storage

    store, _, _ = seeded(tmp_path, meta={"repo_id": "repo", "snapshot_id": "main"})
    with corrupt(store.path, "history_versions") as conn:
        conn.execute(f"UPDATE history_versions SET {assignment}")

    def forbidden(_):
        pytest.fail("private payload decoded to diagnose index corruption")

    monkeypatch.setattr(storage, "_entry", forbidden)
    with pytest.raises(StorageFailure) as error:
        store.snapshot_at("n", valid_at=JAN20)
    assert "PRIVATE_SENTINEL" not in str(error.value)
    assert store.snapshot_at("other").coverage == "unavailable"


@pytest.mark.parametrize("table", ["history_versions", "history_effects"])
def test_missing_effect_cannot_certify_empty_truth_after_restart(tmp_path, table):
    store, request, receipt = seeded(tmp_path)
    with corrupt(store.path, table, "delete") as conn:
        conn.execute(f"DELETE FROM {table}")
    reopened = SQLiteDurableStore(store.path)
    assert reopened.lookup_receipt(request) == receipt
    assert reopened.get_event("n", receipt.event_id).memory_ids == (receipt.memory_id,)
    with pytest.raises(StorageFailure):
        reopened.snapshot_at("n", valid_at=JAN20)


@pytest.mark.parametrize(
    "table,field,value",
    [
        ("history_versions", "status", "deleted"),
        ("history_versions", "source_page_id", "other"),
        ("history_versions", "meta", {"namespace_id": "other"}),
        ("history_pages", "meta", {"repo_id": "private", "snapshot_id": "main"}),
    ],
)
def test_immutable_payload_mismatch_never_returns_complete_truth(
    tmp_path, table, field, value
):
    store, _, receipt = seeded(tmp_path)
    with corrupt(store.path, table) as conn:
        data = json.loads(conn.execute(f"SELECT payload FROM {table}").fetchone()[0])
        data[field] = value
        conn.execute(f"UPDATE {table} SET payload=?", (json.dumps(data),))
    reopened = SQLiteDurableStore(store.path)
    with pytest.raises(StorageFailure):
        reopened.snapshot_at("n", valid_at=JAN20)
    if table == "history_pages":
        with pytest.raises(StorageFailure):
            reopened.historical_page("n", receipt.page_id)


@pytest.mark.parametrize("api", ["event", "page"])
def test_size_preflight_precedes_payload_select(tmp_path, monkeypatch, api):
    from jitmind.storage import sqlite as storage

    store, _, receipt = seeded(tmp_path)
    statements = []
    original = store._connection

    @contextmanager
    def traced(**kwargs):
        with original(**kwargs) as conn:
            conn.set_trace_callback(statements.append)
            yield conn

    monkeypatch.setattr(store, "_connection", traced)
    if api == "page":
        with corrupt(store.path, "history_pages") as conn:
            conn.execute("UPDATE history_pages SET payload=?", ("x" * 1048577,))
        monkeypatch.setattr(
            storage, "_page", lambda _: pytest.fail("oversized page decoded")
        )
        with pytest.raises(CapacityExceeded):
            store.historical_page("n", receipt.page_id)
        assert not any(
            s.startswith("SELECT payload FROM history_pages") for s in statements
        )
    else:
        with sqlite3.connect(store.path) as conn:
            conn.execute(
                "UPDATE outbox SET memory_ids=?", ('["' + "é" * 131073 + '"]',)
            )
        monkeypatch.setattr(
            store, "_projection_event", lambda _: pytest.fail("oversized event decoded")
        )
        with pytest.raises(StorageFailure):
            store.get_event("n", receipt.event_id)
        assert not any(
            s.startswith("SELECT namespace_id,event_id,revision,memory_ids")
            for s in statements
        )


@pytest.mark.parametrize("value", [-1, 1.5, 2**63 - 1])
def test_event_revision_numeric_checks(tmp_path, value):
    store, _, receipt = seeded(tmp_path)
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE outbox SET revision=?", (value,))
    if value == 2**63 - 1:
        assert store.get_event("n", receipt.event_id).revision == value
    else:
        with pytest.raises(StorageFailure):
            store.get_event("n", receipt.event_id)


def test_migration_overlap_rolls_back_prior_namespace_and_receipts(tmp_path):
    store = legacy_store(tmp_path / "db")
    requests = [
        IngestRequest.create(ns, key, key, {"fact_key": "same"})
        for ns, key in (("a", "good"), ("b", "one"), ("b", "two"))
    ]
    receipts = [
        store.ingest(r, lambda _: proposal(start=JAN1, end=None)) for r in requests
    ]
    before = store.path.read_bytes()
    with pytest.raises(InvalidRequest):
        store.enable_history(quiesced=True, recorded_at=FEB1)
    assert store.path.read_bytes() == before
    reopened = SQLiteDurableStore(store.path)
    assert reopened.schema_version == 1
    assert [reopened.lookup_receipt(r) for r in requests] == receipts
    assert not reopened.snapshot_at("b").complete


def test_ambiguous_migration_is_explicit_unknown_not_widened(tmp_path):
    store = legacy_store(tmp_path / "db")
    request = IngestRequest.create("n", "one", "private", {"repo_id": "r"})
    receipt = store.ingest(request, lambda _: proposal(start=JAN1, end=None))
    store.enable_history(quiesced=True, recorded_at=FEB1)
    assert store.lookup_receipt(request) == receipt
    for selectors in ({}, {"repo_id": "r", "snapshot_id": "main"}):
        result = store.snapshot_at("n", valid_at=JAN20, **selectors)
        assert not result.complete and result.unknown_scope and not result.entries


@pytest.mark.parametrize(
    "page_meta,unknown,visible",
    [
        ({}, False, True),
        ({"expires_at": JAN15}, False, False),
        ({"ttl_seconds": 0}, False, False),
        ({"ttl_seconds": -1}, True, False),
        ({"ttl_seconds": True}, True, False),
        ({"ttl_seconds": "bad"}, True, False),
    ],
)
def test_corrected_fact_validity_and_independent_original_page_expiry(
    tmp_path, page_meta, unknown, visible
):
    from jitmind.schemas import MemoryEntry, Page

    source = tmp_path / "legacy"
    source.mkdir()
    entry = MemoryEntry(
        id="fact",
        content="value",
        t_created=JAN1,
        t_observed=JAN1,
        t_valid=JAN1,
        t_invalid=JAN15,
        source_page_id="page",
    )
    page = Page(
        header="h",
        content="source",
        meta={
            "page_id": "page",
            "memory_id": "fact",
            "t_valid": JAN1,
            "t_invalid": JAN15,
            **page_meta,
        },
    )
    (source / "advanced_memory_state.json").write_text(
        json.dumps({"entries": [entry.model_dump()]})
    )
    (source / "pages.json").write_text(json.dumps([page.model_dump()]))
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: FEB1)
    store.import_staged(stage_legacy(source, quiesced=True), "n")
    store.commit_temporal(
        IngestRequest.create("n", "correct", "correct"),
        1,
        proposal(operation="noop"),
        corrections=(ValidityCorrection("fact", JAN1, None),),
    )
    result = store.snapshot_at("n", valid_at=JAN20, eligible_at=JAN20)
    assert bool(result.entries) == visible
    assert result.unknown_eligibility == unknown
    assert result.complete == (not unknown)
    assert (
        store.historical_page("n", store.get_entry("n", "fact").source_page_id).meta[
            "t_invalid"
        ]
        == JAN15
    )
    assert not store.snapshot_at("n", revision=1, valid_at=JAN20).entries


@pytest.mark.parametrize("overlap", [True, False])
def test_import_baseline_rejects_overlap_and_accepts_adjacent_intervals(
    tmp_path, overlap
):
    from jitmind.schemas import MemoryEntry, Page

    source = tmp_path / "legacy"
    source.mkdir()
    entries = [
        MemoryEntry(
            id=str(i),
            content=str(i),
            t_created=JAN1,
            t_observed=JAN1,
            t_valid=JAN1 if i == 0 or overlap else JAN15,
            t_invalid=JAN15 if i == 0 else FEB1,
            source_page_id=f"p{i}",
            meta={"fact_key": "same"},
        )
        for i in range(2)
    ]
    pages = [
        Page(header="h", content=str(i), meta={"page_id": f"p{i}", "memory_id": str(i)})
        for i in range(2)
    ]
    (source / "advanced_memory_state.json").write_text(
        json.dumps({"entries": [e.model_dump() for e in entries]})
    )
    (source / "pages.json").write_text(json.dumps([p.model_dump() for p in pages]))
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: FEB1)
    staged = stage_legacy(source, quiesced=True)
    if overlap:
        before = store.path.read_bytes()
        with pytest.raises(InvalidRequest):
            store.import_staged(staged, "n")
        assert store.path.read_bytes() == before
        assert store.outbox_events("n") == ()
        assert not store.snapshot_at("n").complete
    else:
        store.import_staged(staged, "n")
        result = store.snapshot_at("n", valid_at=JAN20)
        assert result.complete and [e.content for e in result.entries] == ["1"]


@pytest.mark.parametrize("api", ["page", "event", "snapshot"])
def test_payload_apis_revalidate_exact_schema_after_open(tmp_path, api):
    from jitmind.storage import SchemaMismatch

    store, _, receipt = seeded(tmp_path)
    with sqlite3.connect(store.path) as conn:
        conn.execute("CREATE TABLE unexpected (value TEXT)")
    with pytest.raises(SchemaMismatch):
        if api == "page":
            store.historical_page("n", receipt.page_id)
        elif api == "event":
            store.get_event("n", receipt.event_id)
        else:
            store.snapshot_at("n")


def test_compaction_preserves_exact_retained_effect_inventory(tmp_path):
    store, _, receipt = seeded(tmp_path)
    store.clock = lambda: FEB1
    store.commit_temporal(
        IngestRequest.create("n", "correct", "correct"),
        1,
        proposal(operation="noop"),
        corrections=(ValidityCorrection(receipt.memory_id, JAN1, JAN15),),
    )
    compacted = SQLiteDurableStore(
        store.compact_history(
            tmp_path / "compact", "n", before_revision=2, quiesced=True
        )
    )
    result = compacted.snapshot_at("n")
    assert result.complete and result.facts == store.snapshot_at("n").facts
    assert result.baseline_revision == 2
    assert not compacted.snapshot_at("n", revision=1).complete
    with sqlite3.connect(compacted.path) as conn:
        assert conn.execute(
            "SELECT memory_id,revision FROM history_effects"
        ).fetchall() == [(receipt.memory_id, 2)]
    with corrupt(compacted.path, "history_versions", "delete") as conn:
        conn.execute("DELETE FROM history_versions")
    with pytest.raises(StorageFailure):
        SQLiteDurableStore(compacted.path).snapshot_at("n")

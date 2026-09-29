"""Actual v2 authority and MemoryAgent integration; no providers or fixture hooks."""

import ast
import json
import sqlite3
from pathlib import Path

import pytest

from jitmind.agents.memory_agent import MemoryAgent
from jitmind.storage import (
    HistoricalSnapshot,
    IdempotencyConflict,
    IngestRequest,
    InvalidRequest,
    Proposal,
    SchemaMismatch,
    SQLiteDurableStore,
    StorageFailure,
    TargetNotFound,
    ValidityCorrection,
)
from jitmind.storage.sqlite import APPLICATION_ID, SCHEMA

JAN1 = "2026-01-01T00:00:00Z"
JAN10 = "2026-01-10T00:00:00Z"
JAN15 = "2026-01-15T00:00:00Z"
JAN20 = "2026-01-20T00:00:00Z"
FEB1 = "2026-02-01T00:00:00Z"
FEB10 = "2026-02-10T00:00:00Z"
FEB15 = "2026-02-15T00:00:00Z"
MAR1 = "2026-03-01T00:00:00Z"
MAR2 = "2026-03-02T00:00:00Z"


class RecordingGenerator:
    def __init__(self):
        self.prompts = []
        self.decision = {"operation": "add", "t_valid": JAN1}
        self.content = "3"

    def generate_single(self, prompt, schema=None):
        self.prompts.append((prompt, schema))
        return {"json": self.decision} if schema else {"text": self.content}


def proposal(value="4", operation="add", start=JAN15, end=FEB1, target=None):
    return Proposal(
        abstract=value,
        header="header",
        decorated=value,
        decision={
            "operation": operation,
            "t_valid": start,
            "t_invalid": end,
            "target_id": target,
        },
    )


def make_oracle(tmp_path):
    now = [JAN1]
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: now[0])
    gen = RecordingGenerator()
    agent = MemoryAgent(generator=gen, durable_store=store, namespace_id="team")
    first = agent.memorize_durable(
        "retry is 3", idempotency_key="first", meta={"fact_key": "retry_limit"}
    )
    now[0] = FEB1
    gen.content = "5"
    gen.decision = {
        "operation": "update",
        "target_id": first.memory_id,
        "t_valid": FEB1,
    }
    second = agent.memorize_durable("retry is 5", idempotency_key="second")
    now[0] = MAR1
    request = IngestRequest.create(
        "team", "correction", "retry was 4 since Jan15", {"fact_key": "retry_limit"}
    )
    correction = (ValidityCorrection(first.memory_id, JAN1, JAN15),)
    third = store.commit_temporal(request, 2, proposal(), corrections=correction)
    return store, agent, gen, now, first, second, third, request, correction


@pytest.mark.parametrize(
    "valid,transaction,value",
    [
        (JAN20, JAN20, "3"),
        (JAN20, FEB15, "3"),
        (JAN20, MAR1, "4"),
        (JAN10, MAR2, "3"),
        (JAN15, MAR2, "4"),
        (FEB1, MAR2, "5"),
        (FEB10, JAN20, "3"),
    ],
)
def test_seven_query_oracle_normal_agent_and_structured_correction(
    tmp_path, valid, transaction, value
):
    store, *_rest = make_oracle(tmp_path)
    result = store.snapshot_at("team", valid_at=valid, transaction_at=transaction)
    assert isinstance(result, HistoricalSnapshot)
    assert result.coverage == "available" and result.complete
    assert [e.content for e in result.entries] == [value]
    assert result.facts[0].fact_key == "retry_limit"


def test_delete_noop_retry_restart_and_unchanged_observations(tmp_path):
    store, agent, gen, now, first, second, third, request, corrections = make_oracle(
        tmp_path
    )
    before = [store.snapshot_at("team", revision=r) for r in (1, 2, 3)]
    assert (
        store.commit_temporal(request, 0, proposal(), corrections=corrections) == third
    )
    with pytest.raises(IdempotencyConflict):
        store.commit_temporal(request, 3, proposal("6"), corrections=corrections)
    with pytest.raises(IdempotencyConflict):
        store.commit_temporal(
            request,
            3,
            proposal(),
            corrections=(ValidityCorrection(first.memory_id, JAN1, JAN10),),
        )
    now[0] = MAR2
    gen.decision = {"operation": "delete", "target_id": second.memory_id}
    deleted = agent.memorize_durable("delete 5", idempotency_key="delete")
    gen.decision = {"operation": "noop"}
    noop = agent.memorize_durable("nothing", idempotency_key="noop")
    calls = len(gen.prompts)
    assert agent.memorize_durable("nothing", idempotency_key="noop") == noop
    assert len(gen.prompts) == calls
    reopened = SQLiteDurableStore(store.path)
    assert [reopened.snapshot_at("team", revision=r) for r in (1, 2, 3)] == before
    assert reopened.snapshot_at("team", valid_at=FEB10).entries == ()
    assert reopened.snapshot_at("team", valid_at=JAN20).entries[0].content == "4"
    assert (
        reopened.snapshot_at(
            "team", valid_at=JAN20, transaction_at="2026-03-01T05:30:00+05:30"
        )
        .entries[0]
        .content
        == "4"
    )
    assert reopened.get_entry("team", second.memory_id) is None
    assert reopened.get_page("team", second.page_id) is None
    assert reopened.receipt_content(second).status == "retired"
    assert reopened.snapshot_at("team", transaction_at=MAR2).revision == noop.revision
    assert reopened.snapshot_at("team", revision=deleted.revision).revision == 4
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM history_revisions").fetchone()[0] == 5
        assert conn.execute("SELECT count(*) FROM history_versions").fetchone()[0] == 6
        assert conn.execute("SELECT count(*) FROM pages").fetchone()[0] == 5
        for table in ("history_versions", "history_revisions", "history_coverage"):
            with pytest.raises(sqlite3.IntegrityError, match="immutable history"):
                conn.execute(f"DELETE FROM {table}")
            with pytest.raises(sqlite3.IntegrityError, match="immutable history"):
                conn.execute(f"UPDATE {table} SET revision=revision")


def legacy_store(path):
    with sqlite3.connect(path) as conn:
        for statement in SCHEMA:
            conn.execute(statement)
        conn.execute(f"PRAGMA application_id={APPLICATION_ID}")
        conn.execute("PRAGMA user_version=1")
    return SQLiteDurableStore(path, clock=lambda: JAN1)


def test_explicit_v1_migration_preserves_receipts_and_fences_stale_instances(tmp_path):
    old = legacy_store(tmp_path / "v1")
    stale = SQLiteDurableStore(old.path)
    request = IngestRequest.create("team", "first", "3", {"fact_key": "retry_limit"})
    receipt = old.commit_proposal(request, 0, proposal("3", start=JAN1, end=None))
    assert old.diagnostics()["schema_version"] == 1
    assert old.snapshot_at("team").coverage == "unavailable"
    old.backup(tmp_path / "v1-backup")
    with pytest.raises(InvalidRequest):
        old.enable_history(recorded_at=FEB1)
    old.enable_history(quiesced=True, recorded_at=FEB1)
    assert old.lookup_receipt(request) == receipt
    with pytest.raises(SchemaMismatch):
        stale.ingest(IngestRequest.create("team", "bad", "bad"), lambda _: proposal())
    assert old.snapshot_at("team", transaction_at=JAN20).coverage == "unavailable"
    baseline = old.snapshot_at("team", transaction_at=FEB1, valid_at=JAN20)
    assert baseline.complete and baseline.entries[0].content == "3"
    assert baseline.baseline_revision == receipt.revision
    old.clock = lambda: MAR1
    later = old.commit_proposal(
        IngestRequest.create("team", "later", "5"),
        1,
        proposal("5", "update", FEB1, None, receipt.memory_id),
    )
    assert later.revision == 2
    assert old.snapshot_at("team", transaction_at=FEB1).entries[0].t_invalid is None
    old.backup(tmp_path / "v2-backup")
    restored = SQLiteDurableStore.restore(tmp_path / "v2-backup", tmp_path / "restored")
    assert restored.snapshot_at("team") == old.snapshot_at("team")
    old_restored = SQLiteDurableStore.restore(
        tmp_path / "v1-backup", tmp_path / "old-restored"
    )
    assert old_restored.lookup_receipt(request) == receipt
    assert old_restored.snapshot_at("team").coverage == "unavailable"


def test_migration_fault_is_atomic(tmp_path):
    store = legacy_store(tmp_path / "v1")

    def fault(stage):
        if stage == "before_history_migration_commit":
            raise RuntimeError("fail")

    store.fault_hook = fault
    with pytest.raises(StorageFailure):
        store.enable_history(quiesced=True, recorded_at=JAN1)
    assert SQLiteDurableStore(store.path).diagnostics()["schema_version"] == 1
    with sqlite3.connect(store.path) as conn:
        assert not conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'history_%'"
        ).fetchall()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"revision": True},
        {"revision": -1},
        {"revision": 2**63},
        {"revision": 1.0},
        {"limit": True},
        {"limit": 0},
        {"limit": 1001},
        {"limit": 1.0},
        {"transaction_at": "2026-01-01"},
        {"valid_at": "2026-01-01T00:00:00"},
        {"eligible_at": "bad"},
        {"revision": 1, "transaction_at": JAN1},
        {"repo_id": "r"},
        {"snapshot_id": "s"},
        {"revision": 999},
    ],
)
def test_strict_historical_inputs(tmp_path, kwargs):
    store = SQLiteDurableStore(tmp_path / "db")
    with pytest.raises(InvalidRequest):
        store.snapshot_at("team", **kwargs)


def test_unknown_validity_truncation_scope_and_ttl(tmp_path):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JAN1)

    def add(ns, key, meta, start=JAN1):
        return store.ingest(
            IngestRequest.create(ns, key, key, meta),
            lambda _: proposal(key, start=start, end=None),
        )

    add("team", "a", {"expires_at": FEB1})
    add("team", "b", {}, start=None)
    add("other", "a", {})
    add("team", "repo", {"repo_id": "r", "snapshot_id": "s"})
    snap = store.snapshot_at("team", limit=1)
    assert snap.truncated and not snap.complete
    unknown = store.snapshot_at("team", valid_at=JAN20)
    assert unknown.unknown_validity and not unknown.complete
    assert [e.content for e in unknown.entries] == ["a"]
    assert store.snapshot_at("team", valid_at=JAN20, eligible_at=FEB1).entries == ()
    assert store.snapshot_at("team", valid_at=JAN20).entries[0].content == "a"
    assert [
        e.content
        for e in store.snapshot_at("team", repo_id="r", snapshot_id="s").entries
    ] == ["repo"]
    assert store.snapshot_at("team", repo_id="other", snapshot_id="s").entries == ()
    assert [e.content for e in store.snapshot_at("other").entries] == ["a"]
    assert store.snapshot_at("missing").coverage == "unavailable"


def test_conflicting_intervals_and_cross_namespace_correction_rollback(tmp_path):
    store, _, _, _, first, _, _, _, _ = make_oracle(tmp_path)
    status = store.status("team")
    with pytest.raises(InvalidRequest):
        store.commit_proposal(
            IngestRequest.create("team", "conflict", "6", {"fact_key": "retry_limit"}),
            3,
            proposal("6"),
        )
    assert store.status("team") == status
    with pytest.raises(TargetNotFound):
        store.commit_temporal(
            IngestRequest.create("other", "bad", "4"),
            0,
            proposal(),
            corrections=(ValidityCorrection(first.memory_id, JAN1, JAN15),),
        )
    assert store.status("other")["revision"] == 0


def test_python310_syntax_and_exception_metadata_independence():
    for path in Path("jitmind/storage").glob("*.py"):
        ast.parse(path.read_text(), feature_version=(3, 10))
    from jitmind.storage.sqlite import _db_error

    assert isinstance(
        _db_error(sqlite3.OperationalError("interrupted")), StorageFailure
    )


def test_legacy_import_history_and_aliases(tmp_path):
    from jitmind.schemas import MemoryEntry, Page
    from jitmind.storage import stage_legacy

    source = tmp_path / "source"
    source.mkdir()
    entry = MemoryEntry(
        id="same-id",
        content="3",
        t_created=JAN1,
        t_observed=JAN1,
        t_valid=JAN1,
        source_page_id="old-page",
    )
    page = Page(
        header="h",
        content="source",
        meta={"page_id": "old-page", "memory_id": "same-id"},
    )
    (source / "pages.json").write_text(json.dumps([page.model_dump()]))
    (source / "advanced_memory_state.json").write_text(
        json.dumps({"entries": [entry.model_dump()]})
    )
    staged = stage_legacy(source, quiesced=True)
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: FEB1)
    store.import_staged(staged, "one")
    store.import_staged(staged, "two")
    assert store.snapshot_at("one", transaction_at=JAN20).coverage == "unavailable"
    one, two = store.snapshot_at("one"), store.snapshot_at("two")
    assert one.entries[0].id == two.entries[0].id == "same-id"
    assert one.facts[0].page_id != two.facts[0].page_id
    assert store.resolve_page_alias("one", "old-page") == one.facts[0].page_id
    store.import_staged(staged, "one")
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM history_versions").fetchone()[0] == 2


def test_compact_copy_prunes_versions_preserves_identity_and_source(tmp_path):
    store, _, _, _, _first, _second, third, request, corrections = make_oracle(tmp_path)
    other = store.commit_proposal(
        IngestRequest.create("other", "first", "3"),
        0,
        proposal("3", start=JAN1, end=None),
    )
    old = store.snapshot_at("team", revision=1)
    latest = store.snapshot_at("team")
    with pytest.raises(InvalidRequest):
        store.compact_history(tmp_path / "denied", "team", before_revision=3)
    assert not (tmp_path / "denied").exists()
    path = store.compact_history(
        tmp_path / "compact", "team", before_revision=3, quiesced=True
    )
    compact = SQLiteDurableStore(path)
    assert compact.snapshot_at("team", revision=1).coverage == "unavailable"
    assert compact.snapshot_at("team", transaction_at=FEB15).coverage == "unavailable"
    assert compact.snapshot_at("team").facts == latest.facts
    assert compact.snapshot_at("team").baseline_revision == 3
    assert compact.snapshot_at("other", revision=1).entries[0].id == other.memory_id
    assert (
        compact.commit_temporal(request, 0, proposal(), corrections=corrections)
        == third
    )
    assert compact.outbox_events("team") == store.outbox_events("team")
    assert store.snapshot_at("team", revision=1) == old
    with sqlite3.connect(path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM history_versions WHERE namespace_id='team'"
            ).fetchone()[0]
            == 3
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM pages WHERE namespace_id='team'"
            ).fetchone()[0]
            == 3
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable history"):
            conn.execute("DELETE FROM history_versions")
    compact.clock = lambda: MAR2
    receipt = compact.commit_proposal(
        IngestRequest.create("team", "later", "noop"), 3, proposal(operation="noop")
    )
    assert receipt.revision == 4
    with pytest.raises(StorageFailure):
        store.compact_history(path, "team", before_revision=3, quiesced=True)
    assert SQLiteDurableStore(path).status("team")["revision"] == 4


def test_compact_copy_failure_preserves_source_and_removes_only_its_destination(
    tmp_path,
):
    store, *_ = make_oracle(tmp_path)
    before = store.snapshot_at("team")

    def fault(stage):
        if stage == "before_history_compact_commit":
            raise RuntimeError("disk full")

    store.fault_hook = fault
    with pytest.raises(StorageFailure):
        store.compact_history(
            tmp_path / "failed", "team", before_revision=3, quiesced=True
        )
    assert not (tmp_path / "failed").exists()
    assert store.snapshot_at("team") == before


def test_repository_lineage_cannot_escape_through_update_or_correction(tmp_path):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JAN1)
    first = store.commit_proposal(
        IngestRequest.create(
            "team",
            "first",
            "3",
            {
                "repo_id": "r",
                "snapshot_id": "s",
                "fact_key": "retry_limit",
            },
        ),
        0,
        proposal("3", start=JAN1, end=None),
    )
    with pytest.raises(InvalidRequest):
        store.commit_temporal(
            IngestRequest.create("team", "wrong", "fix"),
            1,
            proposal(operation="noop"),
            corrections=(ValidityCorrection(first.memory_id, JAN1, JAN15),),
        )
    with pytest.raises(InvalidRequest):
        store.commit_proposal(
            IngestRequest.create(
                "team", "move", "5", {"repo_id": "other", "snapshot_id": "s"}
            ),
            1,
            proposal("5", "update", FEB1, None, first.memory_id),
        )
    store.clock = lambda: FEB1
    second = store.commit_proposal(
        IngestRequest.create("team", "second", "5"),
        1,
        proposal("5", "update", FEB1, None, first.memory_id),
    )
    assert store.get_entry("team", second.memory_id).meta["repo_id"] == "r"
    assert store.snapshot_at("team").entries == ()
    scoped = store.snapshot_at("team", repo_id="r", snapshot_id="s", valid_at=FEB15)
    assert scoped.entries[0].id == second.memory_id


@pytest.mark.parametrize(
    "meta",
    [
        {"fact_key": ""},
        {"fact_key": 1},
        {"fact_key": "bad\x00key"},
        {"single_valued": 1},
        {"repo_id": "r"},
    ],
)
def test_invalid_temporal_identity_rolls_back(tmp_path, meta):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JAN1)
    with pytest.raises(InvalidRequest):
        store.commit_proposal(
            IngestRequest.create("team", "bad", "bad", meta), 0, proposal()
        )
    assert store.status("team")["revision"] == 0


def test_unknown_interval_does_not_become_false_empty_and_multivalue_is_explicit(
    tmp_path,
):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JAN1)
    for key in ("one", "two"):
        store.ingest(
            IngestRequest.create(
                "team", key, key, {"fact_key": "tags", "single_valued": False}
            ),
            lambda _, value=key: proposal(value, start=JAN1, end=None),
        )
    assert len(store.snapshot_at("team", valid_at=JAN15).entries) == 2
    with pytest.raises(InvalidRequest):
        store.commit_proposal(
            IngestRequest.create("team", "single", "x", {"fact_key": "tags"}),
            2,
            proposal("x", start=FEB1, end=None),
        )


def test_immutable_page_copy_survives_current_metadata_change(tmp_path):
    store, _, _, _, first, *_ = make_oracle(tmp_path)
    source = store.historical_page("team", first.page_id)
    with sqlite3.connect(store.path) as conn:
        payload = json.loads(
            conn.execute(
                "SELECT payload FROM pages WHERE page_id=?", (first.page_id,)
            ).fetchone()[0]
        )
        payload["meta"]["expires_at"] = "2001-01-01T00:00:00Z"
        conn.execute(
            "UPDATE pages SET payload=? WHERE page_id=?",
            (json.dumps(payload), first.page_id),
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable history"):
            conn.execute("UPDATE history_pages SET payload='{}'")
    assert store.historical_page("team", first.page_id) == source
    assert store.historical_page("other", first.page_id) is None
    assert store.snapshot_at("team", revision=1).entries[0].t_invalid is None


@pytest.mark.parametrize(
    "meta",
    [
        {"expires_at": "bad"},
        {"expires_at": "2026-01-01T00:00:00+01:99"},
        {"ttl_seconds": None},
        {"ttl_seconds": True},
        {"ttl_seconds": -1},
    ],
)
def test_malformed_legacy_eligibility_is_retained_but_unknown(tmp_path, meta):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JAN1)
    store.commit_proposal(
        IngestRequest.create("team", "k", "3", meta),
        0,
        proposal("3", start=JAN1, end=None),
    )
    assert store.snapshot_at("team", valid_at=JAN15).entries[0].content == "3"
    result = store.snapshot_at("team", valid_at=JAN15, eligible_at=JAN15)
    assert result.entries == () and not result.complete and result.unknown_eligibility


def test_ttl_eligibility_does_not_prune_history(tmp_path):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JAN1)
    store.commit_proposal(
        IngestRequest.create("team", "k", "3", {"ttl_seconds": 60}),
        0,
        proposal("3", start=JAN1, end=None),
    )
    assert store.snapshot_at(
        "team", valid_at=JAN1, eligible_at="2026-01-01T00:00:59Z"
    ).entries
    assert not store.snapshot_at(
        "team", valid_at=JAN1, eligible_at="2026-01-01T00:01:00Z"
    ).entries
    assert store.snapshot_at("team", valid_at=JAN1).entries


def test_imported_split_provenance_and_ambiguous_scope_are_not_widened(tmp_path):
    from test_scoped_durable import migrated_pair

    good = tmp_path / "good"
    bad = tmp_path / "bad"
    good.mkdir()
    bad.mkdir()
    store, _, _ = migrated_pair(good, {"repo_id": "repo"}, {"snapshot_id": "main"})
    assert store.snapshot_at("n").entries == ()
    result = store.snapshot_at("n", repo_id="repo", snapshot_id="main")
    assert result.entries[0].id == "m" and result.complete
    store.commit_proposal(
        IngestRequest.create("n", "update", "new"),
        1,
        proposal("new", "update", None, None, "m"),
    )
    assert store.snapshot_at("n", repo_id="repo", snapshot_id="main").entries
    ambiguous, _, _ = migrated_pair(
        bad, {"repo_id": "repo", "snapshot_id": "main"}, {"repo_id": "other"}
    )
    for kwargs in ({}, {"repo_id": "repo", "snapshot_id": "main"}):
        result = ambiguous.snapshot_at("n", **kwargs)
        assert result.entries == () and result.unknown_scope and not result.complete


def test_bad_timestamp_offset_rejected(tmp_path):
    store = SQLiteDurableStore(tmp_path / "db")
    with pytest.raises(InvalidRequest):
        store.snapshot_at("team", valid_at="2026-01-01T00:00:00+01:99")


def test_active_correction_then_agent_update_and_delete_preserves_each_observation(
    tmp_path,
):
    now = [JAN1]
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: now[0])
    generator = RecordingGenerator()
    agent = MemoryAgent(generator=generator, durable_store=store, namespace_id="team")
    first = agent.memorize_durable("3", idempotency_key="first")
    now[0] = JAN15
    store.commit_temporal(
        IngestRequest.create("team", "correct", "end at Feb1"),
        1,
        proposal(operation="noop"),
        corrections=(ValidityCorrection(first.memory_id, JAN1, FEB1),),
    )
    corrected = store.snapshot_at("team", revision=2)
    now[0] = FEB1
    generator.decision = {
        "operation": "update",
        "target_id": first.memory_id,
        "t_valid": FEB1,
    }
    generator.content = "5"
    second = agent.memorize_durable("5", idempotency_key="second")
    now[0] = MAR1
    generator.decision = {"operation": "delete", "target_id": second.memory_id}
    agent.memorize_durable("delete", idempotency_key="delete")
    assert store.snapshot_at("team", revision=1).entries[0].t_invalid is None
    assert store.snapshot_at("team", revision=2) == corrected
    assert (
        store.snapshot_at("team", revision=3, valid_at=FEB15).entries[0].content == "5"
    )
    assert not store.snapshot_at("team", valid_at=FEB15).entries


def test_migrated_conflicting_key_never_uses_insertion_order(tmp_path):
    store = legacy_store(tmp_path / "v1")
    for key in ("one", "two"):
        store.ingest(
            IngestRequest.create("team", key, key, {"fact_key": "same"}),
            lambda _, value=key: proposal(value, start=JAN1, end=None),
        )
    before = store.path.read_bytes()
    # Reject at the transaction boundary, preserving the original failure contract
    # without first committing an overlapping, unqueryable history baseline.
    with pytest.raises(InvalidRequest):
        store.enable_history(quiesced=True, recorded_at=JAN15)
    assert store.path.read_bytes() == before
    reopened = SQLiteDurableStore(store.path)
    assert reopened.schema_version == 1
    result = reopened.snapshot_at("team", valid_at=JAN20, limit=1)
    assert not result.complete and result.coverage == "unavailable"


def test_delete_ambiguous_import_still_hides_current_content(tmp_path):
    from test_scoped_durable import migrated_pair

    store, _, _ = migrated_pair(tmp_path, {"repo_id": "repo"}, {"repo_id": "other"})
    before = store.snapshot_at("n", revision=1)
    assert before.unknown_scope
    deleted = store.commit_proposal(
        IngestRequest.create("n", "delete", "forget"),
        1,
        proposal(operation="delete", target="m"),
    )
    assert deleted.operation == "delete"
    assert store.get_entry("n", "m") is None
    assert store.snapshot_at("n", revision=1) == before
    current = store.snapshot_at("n")
    assert current.entries == () and current.complete and not current.unknown_scope

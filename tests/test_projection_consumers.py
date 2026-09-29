"""Actual candidate authority + persistent SQLite algorithms, no service mocks."""

import json
import sqlite3
from dataclasses import replace

import pytest

from jitmind.projections import (
    EmbeddingIdentity,
    FeatureHashingEmbedder,
    ProjectionCoordinator,
    ProjectionError,
    SQLiteGraphProjection,
    SQLiteProjectionSource,
    SQLiteVectorProjection,
)
from jitmind.scope import ScopeAuthority, ScopeContext, ScopeDenied
from jitmind.storage import IngestRequest, Proposal, SQLiteDurableStore


@pytest.fixture
def env(tmp_path):
    authority = ScopeAuthority()
    authority.grant("host", "ns", ["repo"])
    scope = authority.context("host", "ns", "request")
    store = SQLiteDurableStore(tmp_path / "source.sqlite")
    source = SQLiteProjectionSource(store, source_id="fixture-source")
    vector = SQLiteVectorProjection(
        tmp_path / "vector.sqlite", source_id=source.source_id
    )
    graph = SQLiteGraphProjection(
        tmp_path / "graph.sqlite", source_id=source.source_id, trusted_relations=True
    )
    return authority, scope, store, source, vector, graph


def write(
    store,
    key,
    content="tea lemon",
    *,
    namespace="ns",
    operation="add",
    target=None,
    meta=None,
    **decision,
):
    return store.ingest(
        IngestRequest.create(namespace, key, content, meta),
        lambda _: Proposal(
            abstract=content,
            header="header",
            decorated=content,
            decision={
                "operation": operation,
                "target_id": target,
                "updated_content": content,
                **decision,
            },
        ),
    )


def relation(a, b):
    return {"subject": a, "predicate": "depends_on", "object": b}


def drain(env, **kwargs):
    a, s, _, source, vector, graph = env
    return ProjectionCoordinator(a, source, (vector, graph)).drain(s, **kwargs)


def test_real_cosine_multihop_and_independent_ledgers(env):
    a, s, db, source, v, g = env
    tea = write(
        db, "tea", "tea lemon tea", meta={"relations": [relation("app", "cache")]}
    )
    write(
        db,
        "coffee",
        "coffee espresso",
        meta={"relations": [relation("cache", "database")]},
    )
    db.drain_outbox("ns")
    assert db.pending_events("ns") == []
    events = source.events(a, s)
    v.deliver(a, s, source, events[1])
    assert v.status(a, s, source)["watermark"] == 0
    assert g.status(a, s, source)["lag"] == 2
    v.deliver(a, s, source, events[0])
    assert v.status(a, s, source)["watermark"] == 2
    assert all(x["status"] == "complete" for x in drain(env).values())
    result = v.search(a, s, source, "tea lemon")
    assert result.status == "complete"
    assert result.hits[0].fact_id == tea.memory_id
    assert result.hits[0].score > result.hits[1].score
    graph = g.traverse(a, s, source, "app", depth=2)
    assert graph.status == "complete"
    assert graph.nodes == ("app", "cache", "database")
    assert len(graph.edges) == 2
    with sqlite3.connect(v.path) as conn:
        assert (
            len(json.loads(conn.execute("SELECT vector FROM vectors").fetchone()[0]))
            == 256
        )
    with sqlite3.connect(g.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 2


def test_update_delete_duplicate_reorder_shared_support(env):
    a, s, db, source, v, g = env
    first = write(db, "one", meta={"relations": [relation("a", "b")]})
    second = write(db, "two", meta={"relations": [relation("a", "b")]})
    drain(env)
    old_event = source.events(a, s)[0]
    updated = write(db, "update", "new tea", operation="update", target=first.memory_id)
    drain(env)
    assert {h.fact_id for h in v.search(a, s, source, "tea").hits} == {
        second.memory_id,
        updated.memory_id,
    }
    assert len(g.traverse(a, s, source, "a").edges[0].supports) == 1
    write(db, "delete", operation="delete", target=second.memory_id)
    # Lagging graph must not expose cached supports before delivery.
    assert not g.traverse(a, s, source, "a").edges
    drain(env)
    for consumer in (v, g):
        consumer.deliver(a, s, source, old_event)
        consumer.deliver(a, s, source, old_event)
        assert consumer.status(a, s, source)["watermark"] == 4
    assert not g.traverse(a, s, source, "a").edges
    with sqlite3.connect(g.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 0


@pytest.mark.parametrize("consumer_index", [4, 5])
def test_stale_return_deletion_and_scope_revocation(env, consumer_index):
    a, s, db, source, v, _ = env
    first = write(db, "one", meta={"relations": [relation("a", "b")]})
    drain(env)
    consumer = env[consumer_index]

    def fault(stage):
        if stage == "before_return":
            write(db, "delete", operation="delete", target=first.memory_id)

    consumer.fault_hook = fault
    result = (
        consumer.search(a, s, source, "tea")
        if consumer is v
        else consumer.traverse(a, s, source, "a")
    )
    assert result.status == "unavailable" and result.reasons == ("authority_changed",)
    consumer.fault_hook = lambda stage: (
        a.revoke("host", "ns") if stage == "before_return" else None
    )
    with pytest.raises(ScopeDenied):
        consumer.search(a, s, source, "tea") if consumer is v else consumer.traverse(
            a, s, source, "a"
        )


def test_repo_filter_before_payload_decode_and_embedding(env, monkeypatch):
    a, s, db, source, v, _ = env
    allowed = write(db, "allowed", "visible")
    hidden = write(
        db, "hidden", "secret", meta={"repo_id": "other", "snapshot_id": "s"}
    )
    import jitmind.projections.source as source_module

    original = source_module._entry
    seen = []

    def decode(payload):
        entry = original(payload)
        seen.append(entry.id)
        assert entry.id != hidden.memory_id
        return entry

    monkeypatch.setattr(source_module, "_entry", decode)
    drain(env)
    assert [h.fact_id for h in v.search(a, s, source, "visible").hits] == [
        allowed.memory_id
    ]
    assert hidden.memory_id not in seen
    with pytest.raises(ScopeDenied):
        v.search(a, s, source, "secret", snapshots=(("other", "s"),))
    fake = ScopeContext("host", "ns", ("repo",), 1, "fake")
    with pytest.raises(ScopeDenied):
        source.snapshot(a, fake)


def test_snapshot_selection_and_page_restriction(env):
    a, s, db, source, v, _ = env
    write(db, "repo", meta={"repo_id": "repo", "snapshot_id": "s1"})
    assert not source.snapshot(a, s).facts
    assert not source.snapshot(a, s, (("repo", "s2"),)).facts
    drain(env, snapshots=(("repo", "s1"),))
    assert len(v.search(a, s, source, "tea", snapshots=(("repo", "s1"),)).hits) == 1
    assert not v.search(a, s, source, "tea").hits
    assert (
        v.search(a, s, source, "tea", snapshots=(("repo", "s2"),)).status
        == "unavailable"
    )
    assert source.snapshot(a, s, (("repo", "s1"),)).facts[0].repo_id == "repo"


@pytest.mark.parametrize(
    "raw", [[True] * 4, [float("nan")] * 4, [float("inf")] * 4, [0.1] * 3]
)
def test_invalid_embeddings_persist_retry(env, tmp_path, raw):
    a, s, db, source, _, _ = env
    write(db, "one")

    class Embed:
        identity = EmbeddingIdentity("bad", "v1", 4)

        def embed(self, text):
            return raw

    v = SQLiteVectorProjection(
        tmp_path / "bad.sqlite", source_id=source.source_id, embedder=Embed()
    )
    event = source.events(a, s)[0]
    with pytest.raises(ProjectionError):
        v.deliver(a, s, source, event)
    assert v.export("ns")["receipts"][0]["state"] == "retry"
    assert v.status(a, s, source)["watermark"] == 0


@pytest.mark.parametrize(
    "stage", ["before_compute", "after_compute", "before_commit", "after_commit"]
)
def test_fault_retry_restart_without_resurrection(env, stage):
    a, s, db, source, v, _ = env
    first = write(db, "one")
    event = source.events(a, s)[0]

    def fail(current):
        if current == stage:
            raise OSError("fixture ENOSPC")

    v.fault_hook = fail
    with pytest.raises(ProjectionError):
        v.deliver(a, s, source, event)
    restarted = SQLiteVectorProjection(v.path, source_id=source.source_id)
    assert restarted.export("ns")["receipts"][0]["state"] == "retry"
    assert restarted.status(a, s, source)["watermark"] == 0
    restarted.deliver(a, s, source, event)
    assert restarted.search(a, s, source, "tea").hits[0].fact_id == first.memory_id
    assert restarted.export("ns")["receipts"][0]["state"] == "done"


def test_callback_runs_outside_transaction_and_deletion_during_compute(env):
    a, s, db, source, v, _ = env
    first = write(db, "one")

    class Embed:
        identity = v.embedding_identity

        def embed(self, text):
            # Acquire independent writer lock: callback must not be under lock.
            with sqlite3.connect(v.path, timeout=0.01) as conn:
                conn.execute("BEGIN IMMEDIATE")
            write(db, "delete", operation="delete", target=first.memory_id)
            return [0.0] * 256

    v.embedder = Embed()
    with pytest.raises(ProjectionError, match="authority_changed"):
        v.deliver(a, s, source, source.events(a, s)[0])
    assert not v.search(a, s, source, "tea").hits


def test_model_change_and_schema_rejection(env):
    _, _, _, source, v, g = env
    with pytest.raises(ProjectionError, match="identity_mismatch"):
        SQLiteVectorProjection(
            v.path, source_id=source.source_id, embedder=FeatureHashingEmbedder(128)
        )
    with pytest.raises(ProjectionError, match="schema_mismatch"):
        SQLiteGraphProjection(v.path, source_id=source.source_id)
    with pytest.raises(ProjectionError, match="identity_mismatch"):
        SQLiteGraphProjection(g.path, source_id="other", trusted_relations=True)


@pytest.mark.parametrize("field", ["revision", "after_revision", "limit"])
@pytest.mark.parametrize("value", [True, -1, 1.5, "1", 2**63])
def test_numeric_validation(env, field, value):
    a, s, db, source, v, _ = env
    write(db, "one")
    with pytest.raises(ProjectionError):
        if field == "revision":
            v.deliver(a, s, source, replace(source.events(a, s)[0], revision=value))
        else:
            source.events(a, s, **{field: value})


def test_fabricated_event_and_other_namespace(env):
    a, s, db, source, v, _ = env
    write(db, "one")
    event = source.events(a, s)[0]
    with pytest.raises(ProjectionError, match="event_identity_mismatch"):
        v.deliver(a, s, source, replace(event, memory_ids=("fake",)))
    with pytest.raises(ScopeDenied):
        v.deliver(a, s, source, replace(event, namespace_id="other"))
    assert v.export("ns")["receipts"] == []


def test_future_expired_and_namespace_isolation(env):
    a, s, db, source, v, _ = env
    write(db, "future", t_valid="2999-01-01T00:00:00+00:00")
    write(db, "expired", t_invalid="2000-01-01T00:00:00+00:00")
    live = write(db, "live")
    write(db, "other", namespace="other")
    drain(env)
    assert [h.fact_id for h in v.search(a, s, source, "tea").hits] == [live.memory_id]
    assert len(source.events(a, s)) == 3


@pytest.mark.parametrize(
    "budget,reason",
    [
        ("depth", "depth_budget"),
        ("edge_limit", "edge_budget"),
        ("visited_limit", "visited_budget"),
    ],
)
def test_bounded_graph_traversals(env, budget, reason):
    a, s, db, source, _, g = env
    write(
        db,
        "chain",
        meta={
            "relations": [relation("a", "b"), relation("b", "c"), relation("c", "d")]
        },
    )
    drain(env)
    result = g.traverse(a, s, source, "a", **{budget: 1})
    assert result.status == "partial" and reason in result.reasons


def test_backup_restore_retention_disable_and_rebuild(env, tmp_path):
    a, s, db, source, v, g = env
    first = write(db, "one", meta={"relations": [relation("a", "b")]})
    drain(env)
    for consumer in (v, g):
        consumer.admin_purge(consumer.admin_plan_purge("ns", first.memory_id))
        consumer.rebuild(a, s, source)
        backup = consumer.backup(tmp_path / (consumer.kind + ".backup"))
        kwargs = {"trusted_relations": True} if consumer is g else {}
        restored = type(consumer).restore(
            backup,
            tmp_path / (consumer.kind + ".restored"),
            source_id=source.source_id,
            **kwargs,
        )
        assert restored.export("ns")["purges"][0]["fact"] == first.memory_id
        assert restored.export("ns")["facts"][0]["status"] == "purged"
        restored.admin_disable()
        result = (
            restored.search(a, s, source, "tea")
            if consumer is v
            else restored.traverse(a, s, source, "a")
        )
        assert result.reasons == ("consumer_disabled",)
        with pytest.raises(FileExistsError):
            restored.backup(backup)


def test_gap_and_bounded_rebuild(env):
    a, s, db, source, v, g = env
    # v2 history references outbox records and correctly rejects destructive
    # pruning. Exercise event compaction on the supported validated v1 layout.
    import jitmind.storage.sqlite as storage_module

    if storage_module.SCHEMA_VERSION != 1:
        path = db.path.parent / "gap-v1"
        with sqlite3.connect(path) as conn:
            for statement in storage_module.SCHEMAS[1]:
                conn.execute(statement)
            conn.execute(f"PRAGMA application_id={storage_module.APPLICATION_ID}")
            conn.execute("PRAGMA user_version=1")
        db = SQLiteDurableStore(path)
        source = SQLiteProjectionSource(db, source_id=source.source_id)
        env = (a, s, db, source, v, g)
    write(db, "one")
    write(db, "two")
    # Disposable authority retention fixture; no runtime/source adapter writes.
    with sqlite3.connect(db.path) as conn:
        conn.execute("DELETE FROM outbox WHERE revision=1")
    drain(env)
    assert "event_gap" in v.status(a, s, source)["reasons"]
    with pytest.raises(ProjectionError, match="snapshot_budget"):
        v.rebuild(a, s, source, limit=1)
    v.rebuild(a, s, source)
    assert v.status(a, s, source)["watermark"] == 2
    assert len(v.search(a, s, source, "tea").hits) == 2


def test_authority_restore_rollback_is_unavailable(env, tmp_path):
    a, s, db, source, v, _ = env
    first = write(db, "one")
    backup = db.backup(tmp_path / "old-source")
    drain(env)
    write(db, "delete", operation="delete", target=first.memory_id)
    drain(env)
    restored_db = SQLiteDurableStore.restore(backup, tmp_path / "restored-source")
    restored = SQLiteProjectionSource(restored_db, source_id=source.source_id)
    assert v.search(a, s, restored, "tea").reasons == ("authority_rollback",)
    with pytest.raises(ProjectionError, match="authority_rollback"):
        v.rebuild(a, s, restored)


def test_queries_do_not_mutate_projection_files(env):
    a, s, db, source, v, g = env
    write(db, "one", meta={"relations": [relation("a", "b")]})
    drain(env)
    before = (v.path.read_bytes(), g.path.read_bytes(), db.path.read_bytes())
    v.search(a, s, source, "tea")
    g.traverse(a, s, source, "a")
    assert before == (v.path.read_bytes(), g.path.read_bytes(), db.path.read_bytes())


@pytest.mark.parametrize("consumer_index", [4, 5])
@pytest.mark.parametrize("action", ["purge", "disable"])
def test_query_rechecks_consumer_before_return(env, consumer_index, action):
    a, s, db, source, v, _ = env
    first = write(db, "one", meta={"relations": [relation("a", "b")]})
    drain(env)
    consumer = env[consumer_index]

    def hook(stage):
        if stage == "before_return":
            if action == "purge":
                consumer.admin_purge(consumer.admin_plan_purge("ns", first.memory_id))
            else:
                consumer.admin_disable()

    consumer.fault_hook = hook
    result = (
        consumer.search(a, s, source, "tea")
        if consumer is v
        else consumer.traverse(a, s, source, "a")
    )
    assert result.status == "unavailable" and result.reasons == ("consumer_changed",)


def test_consumer_failure_does_not_ack_other_consumer(env):
    a, s, db, source, v, g = env
    write(db, "one", meta={"relations": [relation("a", "b")]})

    def fail(stage):
        if stage == "before_compute":
            raise RuntimeError("private callback detail")

    v.fault_hook = fail
    outcomes = drain(env)
    assert outcomes[str(v.path)]["status"] == "unavailable"
    assert g.status(a, s, source)["watermark"] == 1
    assert v.status(a, s, source)["watermark"] == 0
    assert "private" not in str(v.export("ns"))
    v.fault_hook = None
    drain(env)
    assert v.status(a, s, source)["watermark"] == 1


def test_same_fact_id_other_namespace_is_separate(env):
    a, s, db, source, v, g = env
    first = write(db, "one", "tea")
    other = write(db, "other", "coffee", namespace="other")
    with sqlite3.connect(db.path) as conn:
        payload = json.loads(
            conn.execute(
                "SELECT payload FROM facts WHERE namespace_id='other'"
            ).fetchone()[0]
        )
        payload["id"] = first.memory_id
        conn.execute(
            "UPDATE facts SET memory_id=?,payload=? WHERE namespace_id='other'",
            (first.memory_id, json.dumps(payload)),
        )
        conn.execute(
            "UPDATE outbox SET memory_ids=? WHERE namespace_id='other'",
            (json.dumps([first.memory_id]),),
        )
    a.grant("host", "other", [])
    other_scope = a.context("host", "other", "other-request")
    ProjectionCoordinator(a, source, (v, g)).drain(other_scope)
    drain(env)
    assert v.search(a, s, source, "tea").hits[0].content == "tea"
    other_hits = v.search(a, other_scope, source, "coffee").hits
    assert (
        other_hits[0].fact_id == first.memory_id and other_hits[0].content == "coffee"
    )
    assert other.memory_id != first.memory_id


@pytest.mark.parametrize(
    "relations", [[{"subject": "a"}], [relation("a", "b")] * 101, [relation(True, "b")]]
)
def test_bad_graph_relations_fail_with_retry(env, relations):
    a, s, db, source, _, g = env
    write(db, "one", meta={"relations": relations})
    with pytest.raises(ProjectionError):
        g.deliver(a, s, source, source.events(a, s)[0])
    assert g.export("ns")["receipts"][0]["state"] == "retry"


def test_untrusted_graph_relation_opt_in(env, tmp_path):
    a, s, db, source, _, _ = env
    write(db, "one", meta={"relations": [relation("a", "b")]})
    graph = SQLiteGraphProjection(tmp_path / "untrusted", source_id=source.source_id)
    with pytest.raises(ProjectionError, match="untrusted_relations"):
        graph.deliver(a, s, source, source.events(a, s)[0])


def test_cursor_corruption_fails_closed(env):
    a, s, db, source, v, _ = env
    write(db, "one")
    drain(env)
    with sqlite3.connect(v.path) as conn:
        conn.execute("UPDATE states SET watermark=-1")
    assert v.search(a, s, source, "tea").status == "unavailable"


def test_namespace_purge_survives_new_events_and_rebuild(env):
    a, s, db, source, v, _ = env
    write(db, "one")
    drain(env)
    plan = v.admin_plan_purge("ns")
    write(db, "two")
    drain(env)
    with pytest.raises(ProjectionError, match="purge_plan_changed"):
        v.admin_purge(plan)
    v.admin_purge(v.admin_plan_purge("ns"))
    write(db, "three")
    drain(env)
    v.rebuild(a, s, source)
    assert not v.search(a, s, source, "tea").hits
    assert v.export("ns")["purges"][0]["fact"] == ""


def test_mutated_event_receipt_rejected(env):
    a, s, db, source, v, _ = env
    write(db, "one")
    drain(env)
    event = source.events(a, s)[0]
    with sqlite3.connect(v.path) as conn:
        conn.execute("UPDATE receipts SET event='{}'")
    with pytest.raises(ProjectionError, match="receipt_conflict"):
        v.deliver(a, s, source, event)


def test_rebuild_preserves_monotonic_tombstone_with_same_revision_restore(
    env, tmp_path
):
    a, s, db, source, v, _ = env
    first = write(db, "one")
    backup = db.backup(tmp_path / "fork")
    write(db, "delete", operation="delete", target=first.memory_id)
    drain(env)
    branch = SQLiteDurableStore.restore(backup, tmp_path / "branch")
    write(branch, "unrelated")
    branch_source = SQLiteProjectionSource(branch, source_id=source.source_id)
    assert v.search(a, s, branch_source, "tea").status == "unavailable"
    v.rebuild(a, s, branch_source)
    assert first.memory_id not in {
        h.fact_id for h in v.search(a, s, branch_source, "tea").hits
    }


def test_budgeted_source_and_query_fail_closed(env):
    a, s, db, source, v, g = env
    for index in range(3):
        write(db, str(index))
    drain(env)
    assert source.snapshot(a, s, limit=1).truncated
    assert "snapshot_budget" in v.search(a, s, source, "tea", candidate_limit=1).reasons
    assert "snapshot_budget" in g.traverse(a, s, source, "a", candidate_limit=1).reasons
    with pytest.raises(Exception, match="invalid_request"):
        v.search(a, s, source, "x" * 131073)


def test_embedder_exception_and_identity_drift(env):
    a, s, db, source, v, _ = env
    write(db, "one")

    class Broken:
        identity = v.embedding_identity

        def embed(self, text):
            raise RuntimeError("private text")

    v.embedder = Broken()
    with pytest.raises(ProjectionError, match="embedding_failed"):
        v.deliver(a, s, source, source.events(a, s)[0])
    v.embedder.identity = EmbeddingIdentity("different", "v2", 256)
    with pytest.raises(ProjectionError, match="embedding_identity_changed"):
        v.deliver(a, s, source, source.events(a, s)[0])


def test_public_event_api_if_present_is_checked(env, monkeypatch):
    a, s, db, source, _, _ = env
    write(db, "one")
    if not hasattr(db, "outbox_events"):
        # Base candidate intentionally has no public all-event method.
        assert source.events(a, s)[0].revision == 1
        return
    original = db.outbox_events
    called = []

    def observed(*args, **kwargs):
        called.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "outbox_events", observed)
    assert source.events(a, s)[0].revision == 1
    assert called == [("ns",)]


def test_literal_star_purge_is_not_namespace_purge(env):
    a, s, db, source, v, _ = env
    write(db, "one")
    drain(env)
    v.admin_purge(v.admin_plan_purge("ns", "*"))
    v.rebuild(a, s, source)
    assert len(v.search(a, s, source, "tea").hits) == 1


def test_corrupt_vector_rejected_without_cached_content(env):
    a, s, db, source, v, _ = env
    write(db, "one")
    drain(env)
    with sqlite3.connect(v.path) as conn:
        conn.execute("UPDATE vectors SET vector='not-json'")
    result = v.search(a, s, source, "tea")
    assert result.status == "unavailable" and not result.hits


def test_explicit_bad_namespace_metadata_is_excluded(env):
    a, s, db, source, v, _ = env
    # Current v2 rejects ambiguous metadata at ingestion. Keep that contract,
    # then exercise exclusion against an explicitly labelled real v1 fixture.
    import jitmind.storage.sqlite as storage_module
    from jitmind.storage import InvalidRequest

    if storage_module.SCHEMA_VERSION != 1:
        for metadata in ({"namespace_id": None}, {"namespace_id": ["ns"]}):
            with pytest.raises(InvalidRequest):
                write(db, "rejected", meta=metadata)
        path = db.path.parent / "legacy-malformed-provenance-v1"
        with sqlite3.connect(path) as conn:
            for statement in storage_module.SCHEMAS[1]:
                conn.execute(statement)
            conn.execute(f"PRAGMA application_id={storage_module.APPLICATION_ID}")
            conn.execute("PRAGMA user_version=1")
        db = SQLiteDurableStore(path)
        source = SQLiteProjectionSource(db, source_id=source.source_id)
        env = (a, s, db, source, v, env[5])
    write(db, "bad", meta={"namespace_id": None})
    write(db, "bad-array", meta={"namespace_id": ["ns"]})
    good = write(db, "good", meta={"namespace": ["ns"]})
    drain(env)
    assert [h.fact_id for h in v.search(a, s, source, "tea").hits] == [good.memory_id]


def test_rebuild_retains_immutable_receipt_history(env):
    a, s, db, source, v, _ = env
    write(db, "one")
    drain(env)
    receipts = v.export("ns")["receipts"]
    v.rebuild(a, s, source)
    assert v.export("ns")["receipts"] == receipts


def test_disable_during_embedding_prevents_delivery(env):
    a, s, db, source, v, _ = env
    write(db, "one")

    def disable(stage):
        if stage == "after_compute":
            v.admin_disable()

    v.fault_hook = disable
    with pytest.raises(ProjectionError, match="consumer_disabled"):
        v.deliver(a, s, source, source.events(a, s)[0])
    assert v.export("ns")["facts"] == []

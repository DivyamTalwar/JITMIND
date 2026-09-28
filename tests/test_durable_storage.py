import multiprocessing
import os
import sqlite3
import uuid

import pytest

from jitmind.agents.memory_agent import MemoryAgent
from jitmind.schemas import InMemoryMemoryStore, InMemoryPageStore, MemoryUpdate
from jitmind.storage import (
    AcknowledgementUncertain,
    DurableMemoryAdapter,
    DurablePageAdapter,
    IdempotencyConflict,
    IngestRequest,
    InvalidRequest,
    Proposal,
    SchemaMismatch,
    SQLiteDurableStore,
    StaleRevision,
    StorageBusy,
    StorageFailure,
    TargetNotFound,
    UnsafeJournal,
)

NOW = "2026-01-01T00:00:00+00:00"


def proposal(operation="add", target_id=None, content="fact"):
    return Proposal(
        abstract=content,
        header="header",
        decorated="decorated",
        decision={
            "operation": operation,
            "target_id": target_id,
            "t_observed": NOW,
            "updated_content": content,
        },
    )


def write(store, key="key", namespace="a", operation="add", target=None):
    return store.ingest(
        IngestRequest.create(namespace, key, "message", {"custom": [1, {"x": True}]}),
        lambda _: proposal(operation, target),
    )


def test_real_transaction_receipt_lifecycle_and_namespace(tmp_path):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: NOW)
    first = write(store)
    assert str(uuid.UUID(first.page_id)) == first.page_id
    assert str(uuid.UUID(first.memory_id)) == first.memory_id
    assert store.get_page("a", first.page_id).meta["custom"] == [1, {"x": True}]
    assert store.get_entry("a", first.memory_id).meta == {"custom": [1, {"x": True}]}
    assert store.get_entry("b", first.memory_id) is None
    assert store.get_page("b", first.page_id) is None
    updated = write(store, "update", operation="update", target=first.memory_id)
    assert updated.revision == 2
    old = store.get_entry("a", first.memory_id, include_inactive=True)
    assert old.status == "superseded" and old.t_expired == NOW
    assert store.get_entry("a", updated.memory_id).version_of == first.memory_id
    assert store.get_entry("a", first.memory_id) is None
    assert store.get_page("a", first.page_id) is None
    assert store.memory_update(first).new_page.meta["redacted"] is True
    deleted = write(store, "delete", operation="delete", target=updated.memory_id)
    assert deleted.memory_id is None and deleted.revision == 3
    assert store.get_entry("a", updated.memory_id) is None
    assert store.snapshot("a").entries == ()
    noop = write(store, "noop", operation="noop")
    assert noop.revision == 4 and noop.memory_id is None
    assert store.status("a") == {
        "revision": 4,
        "pending_events": 4,
        "delivery_watermark": 0,
        "pages": 4,
        "facts": 2,
        "operations": 4,
    }
    assert store.status("b")["revision"] == 0
    with pytest.raises(TargetNotFound):
        write(store, "wrong", "b", "delete", first.memory_id)
    assert store.status("b")["pages"] == 0


@pytest.mark.parametrize(
    "stage",
    [
        "before_transaction",
        "after_page_insert",
        "after_fact_insert",
        "after_outbox_receipt",
        "before_commit",
    ],
)
def test_fault_rolls_back_entire_pair(tmp_path, stage):
    path = tmp_path / "db"

    def fault(current):
        if current == stage:
            raise RuntimeError("private payload /secret/path credential")

    store = SQLiteDurableStore(path, fault_hook=fault)
    with pytest.raises(StorageFailure, match="^storage_failure$"):
        write(store)
    reopened = SQLiteDurableStore(path)
    assert reopened.status("a") == {
        "revision": 0,
        "pending_events": 0,
        "delivery_watermark": 0,
        "pages": 0,
        "facts": 0,
        "operations": 0,
    }
    assert write(reopened).revision == 1


def test_true_durable_ack_after_commit_reopens_without_generation(tmp_path):
    path = tmp_path / "db"

    def fault(stage):
        if stage == "after_commit":
            raise RuntimeError("ack lost")

    store = SQLiteDurableStore(path, fault_hook=fault)
    with pytest.raises(AcknowledgementUncertain):
        write(store)
    reopened = SQLiteDurableStore(path)
    request = IngestRequest.create("a", "key", "message", {"custom": [1, {"x": True}]})
    receipt = reopened.ingest(request, lambda _: pytest.fail("replay generated"))
    assert reopened.status("a")["operations"] == 1
    assert reopened.get_entry("a", receipt.memory_id).source_page_id == receipt.page_id
    assert reopened.lookup_receipt(request) == receipt
    with pytest.raises(IdempotencyConflict):
        reopened.ingest(
            IngestRequest.create("a", "key", "different"),
            lambda _: pytest.fail("conflict generated"),
        )


class FakeGenerator:
    def __init__(self):
        self.calls = 0
        self.target = None
        self.operation = "add"

    def generate_single(self, prompt, schema=None):
        self.calls += 1
        if schema:
            return {"json": {"operation": self.operation, "target_id": self.target}}
        return {"text": "generated fact"}


class ForbiddenSideEffect:
    def __getattr__(self, name):
        raise AssertionError("unexpected side effect: " + name)


def test_public_memory_agent_and_legacy_compatibility(tmp_path):
    generator = FakeGenerator()
    store = SQLiteDurableStore(tmp_path / "db")
    agent = MemoryAgent(
        generator=generator,
        durable_store=store,
        namespace_id="tenant",
        graph_store=ForbiddenSideEffect(),
        profile_agent=ForbiddenSideEffect(),
    )
    receipt = agent.memorize_durable(
        "hello", meta={"custom": 1}, user_id="user", idempotency_key="stable"
    )
    assert generator.calls == 2
    assert (
        agent.memorize_durable(
            "hello", meta={"custom": 1}, user_id="user", idempotency_key="stable"
        )
        == receipt
    )
    assert generator.calls == 2
    update = agent.memorize("hello", {"custom": 1}, "user", idempotency_key="stable")
    assert isinstance(update, MemoryUpdate) and generator.calls == 2
    assert update.new_state.abstracts == ["generated fact"]
    assert update.new_page.meta["user_id"] == "user"
    assert DurableMemoryAdapter(store, "tenant").load() == update.new_state
    assert DurablePageAdapter(store, "tenant").get(receipt.page_id) == update.new_page
    assert not (tmp_path / "advanced_memory_state.json").exists()
    generator.operation, generator.target = "delete", receipt.memory_id
    agent.memorize_durable("forget", idempotency_key="delete")
    replay = agent.memorize("hello", {"custom": 1}, "user", idempotency_key="stable")
    assert replay.new_state.abstracts == [] and replay.new_page.content == ""
    other = MemoryAgent(
        generator=FakeGenerator(), durable_store=store, namespace_id="other"
    )
    assert other.memory_store.load().abstracts == []
    legacy = MemoryAgent(
        generator=FakeGenerator(),
        memory_store=InMemoryMemoryStore(),
        page_store=InMemoryPageStore(),
        graph_store=ForbiddenSideEffect(),
    )
    assert isinstance(legacy.memorize("hello"), MemoryUpdate)
    assert legacy.page_store.load()[0].meta["page_id"] == 0


def test_generation_is_outside_lock_and_replans_captured_revision(tmp_path):
    store = SQLiteDurableStore(tmp_path / "db", busy_timeout_ms=0)
    seen = []

    def generate(snapshot):
        seen.append(snapshot.revision)
        if len(seen) == 1:
            write(store, "interloper")  # would fail if generation held a write lock
        return proposal()

    result = store.ingest(IngestRequest.create("a", "outer", "outer"), generate)
    assert seen == [0, 1] and result.revision == 2
    with pytest.raises(StaleRevision):
        store.commit_proposal(IngestRequest.create("a", "stale", "x"), 0, proposal())
    assert store.status("a")["operations"] == 2

    def always_stale(snapshot):
        write(store, str(uuid.uuid4()))
        return proposal()

    with pytest.raises(StaleRevision):
        store.ingest(
            IngestRequest.create("a", "bounded", "x"), always_stale, max_replans=1
        )
    assert store.status("a")["operations"] == 4


def test_busy_commit_explicitly_rolls_back_without_repeating_generation(tmp_path):
    path = tmp_path / "db"
    readers = []
    calls = []

    def fault(stage):
        if stage == "before_commit":
            reader = sqlite3.connect(path, isolation_level=None)
            reader.execute("BEGIN")
            reader.execute("SELECT * FROM namespaces").fetchall()
            readers.append(reader)

    store = SQLiteDurableStore(
        path, busy_timeout_ms=0, commit_retries=1, fault_hook=fault
    )
    try:
        with pytest.raises(StorageBusy):
            store.ingest(
                IngestRequest.create("a", "key", "x"),
                lambda s: calls.append(s.revision) or proposal(),
            )
        assert calls == [0]
    finally:
        for reader in readers:
            reader.close()
    reopened = SQLiteDurableStore(path)
    assert reopened.status("a")["pages"] == 0
    assert write(reopened).revision == 1


def test_projection_duplicate_out_of_order_and_forget_filter(tmp_path):
    store = SQLiteDurableStore(tmp_path / "db")
    first = write(store)
    store.deliver_event("a", first.event_id)
    assert len(store.projected_entries("a")) == 1
    deleted = write(store, "delete", operation="delete", target=first.memory_id)
    # Stale projection is immediately filtered before delivery of the delete.
    assert store.projected_entries("a") == []
    store.deliver_event("a", deleted.event_id)
    store.deliver_event("a", first.event_id)
    assert store.projected_entries("a") == []
    late = write(store, "late")
    removal = write(store, "remove", operation="delete", target=late.memory_id)
    store.deliver_event("a", removal.event_id)
    assert store.status("a")["delivery_watermark"] == 2
    store.deliver_event("a", late.event_id)
    assert store.status("a")["delivery_watermark"] == 4
    assert store.projected_entries("a") == []
    other = write(store, namespace="b")
    with pytest.raises(InvalidRequest):
        store.deliver_event("a", other.event_id)
    assert store.projected_entries("b") == []
    assert store.drain_outbox("b", limit=1) == 1
    assert len(store.projected_entries("b")) == 1


def test_projection_failure_does_not_rollback_authority(tmp_path):
    path = tmp_path / "db"

    def fault(stage):
        if stage == "after_projection":
            raise RuntimeError("lost ack")

    store = SQLiteDurableStore(path, fault_hook=fault)
    receipt = write(store)
    with pytest.raises(StorageFailure):
        store.deliver_event("a", receipt.event_id)
    reopened = SQLiteDurableStore(path)
    assert reopened.status("a")["operations"] == 1
    assert reopened.status("a")["pending_events"] == 0
    reopened.deliver_event("a", receipt.event_id)
    assert len(reopened.projected_entries("a")) == 1


@pytest.mark.parametrize(
    "meta",
    [
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": object()},
        {1: "x"},
        {"__proto__": {}},
        {"x": "a" * 65537},
        {"x": [0] * 2049},
        {"page_id": "spoof"},
    ],
)
def test_metadata_rejects_unsafe_or_unbounded(meta):
    with pytest.raises(InvalidRequest):
        IngestRequest.create("a", "k", "x", meta)


def test_request_canonical_digest_immutable_metadata_and_validation(tmp_path):
    meta = {"b": [1], "a": 2}
    req = IngestRequest.create("a", "key", "hello", meta)
    meta["b"].append(2)
    assert req == IngestRequest.create("a", "key", "hello", {"a": 2, "b": [1]})
    assert (
        req.digest
        != IngestRequest.create("a", "key", " hello", {"a": 2, "b": [1]}).digest
    )
    for args in [
        ("", "k", "x"),
        ("a", "", "x"),
        ("a", "k", " "),
        ("a", "k", "x" * 131073),
    ]:
        with pytest.raises(InvalidRequest):
            IngestRequest.create(*args)
    store = SQLiteDurableStore(tmp_path / "db")
    for limit in [0, -1, 1001, True]:
        with pytest.raises(InvalidRequest):
            store.pending_events("a", limit=limit)
    with pytest.raises(InvalidRequest):
        store.commit_proposal(req, 0, proposal("update"))
    assert store.status("a")["pages"] == 0


def test_actual_pragmas_version_and_refuse_unknown_or_corrupt(tmp_path):
    store = SQLiteDurableStore(tmp_path / "db")
    diagnostic = store.diagnostics()
    assert diagnostic["sqlite_version"] == sqlite3.sqlite_version
    assert diagnostic["sqlite_source_id"]
    assert diagnostic["synchronous"] == 2 and diagnostic["foreign_keys"] == 1
    assert diagnostic["journal_mode"] == "delete"
    if sqlite3.sqlite_version_info < (3, 51, 3):
        with pytest.raises(UnsafeJournal):
            SQLiteDurableStore(tmp_path / "wal", journal_mode="WAL")
    corrupt = tmp_path / "corrupt"
    corrupt.write_bytes(b"not sqlite private secret")
    with pytest.raises(StorageFailure, match="^storage_failure$"):
        SQLiteDurableStore(corrupt)
    assert corrupt.read_bytes() == b"not sqlite private secret"
    conn = sqlite3.connect(tmp_path / "db")
    conn.execute("PRAGMA user_version=99")
    conn.close()
    before = (tmp_path / "db").read_bytes()
    with pytest.raises(SchemaMismatch):
        SQLiteDurableStore(tmp_path / "db")
    assert (tmp_path / "db").read_bytes() == before


def _crash_worker(path, stage):
    def fault(current):
        if current == stage:
            os._exit(37)

    store = SQLiteDurableStore(path, fault_hook=fault)
    write(store)


def _duplicate_worker(path, namespace, gate, output):
    try:
        store = SQLiteDurableStore(path, busy_timeout_ms=1000)
        gate.wait(timeout=15)
        output.put(("ok", write(store, namespace=namespace).model_dump()))
    except Exception as exc:  # noqa: BLE001 - report child failure to its parent
        output.put(("error", type(exc).__name__))


def _cleanup(children):
    for child in children:
        child.join(timeout=20)
    for child in children:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        if child.is_alive():
            child.kill()
            child.join(timeout=5)


@pytest.mark.parametrize(
    "stage,committed",
    [
        ("before_transaction", False),
        ("after_page_insert", False),
        ("after_fact_insert", False),
        ("after_outbox_receipt", False),
        ("after_commit", True),
    ],
)
def test_process_crash_fresh_reader_sees_zero_or_whole_pair(tmp_path, stage, committed):
    path = tmp_path / "db"
    SQLiteDurableStore(path)
    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(target=_crash_worker, args=(str(path), stage))
    child.start()
    try:
        child.join(timeout=20)
        assert child.exitcode == 37
        store = SQLiteDurableStore(path)
        status = store.status("a")
        assert (
            status["pages"] == status["facts"] == status["operations"] == int(committed)
        )
        receipt = write(store)
        assert receipt.revision == 1
        assert store.status("a")["pages"] == 1
    finally:
        _cleanup([child])


def test_multiprocess_duplicate_keys_and_namespace_isolation(tmp_path):
    path = tmp_path / "db"
    SQLiteDurableStore(path)
    ctx = multiprocessing.get_context("spawn")
    gate, output = ctx.Barrier(4), ctx.Queue()
    children = [
        ctx.Process(target=_duplicate_worker, args=(str(path), namespace, gate, output))
        for namespace in ["a", "a", "b", "b"]
    ]
    try:
        for child in children:
            child.start()
        results = [output.get(timeout=30) for _ in children]
        assert all(result[0] == "ok" for result in results), results
        for namespace in ("a", "b"):
            receipts = [r[1] for r in results if r[1]["namespace_id"] == namespace]
            assert receipts[0] == receipts[1]
            assert SQLiteDurableStore(path).status(namespace)["operations"] == 1
        assert results[0][1]["revision"] == 1
    finally:
        _cleanup(children)
        output.close()
        output.join_thread()
    assert all(child.exitcode == 0 for child in children)


def test_public_facade_lost_ack_restarts_without_provider_or_side_effects(tmp_path):
    def fault(stage):
        if stage == "after_commit":
            raise RuntimeError("lost acknowledgement")

    generator = FakeGenerator()
    first = MemoryAgent(
        generator=generator,
        durable_store=SQLiteDurableStore(tmp_path / "db", fault_hook=fault),
        namespace_id="a",
    )
    with pytest.raises(AcknowledgementUncertain):
        first.memorize_durable("hello", idempotency_key="persistent")
    assert generator.calls == 2
    new_generator = FakeGenerator()
    fresh = MemoryAgent(
        generator=new_generator,
        durable_store=SQLiteDurableStore(tmp_path / "db"),
        namespace_id="a",
        graph_store=ForbiddenSideEffect(),
        profile_agent=ForbiddenSideEffect(),
    )
    receipt = fresh.memorize_durable("hello", idempotency_key="persistent")
    assert receipt.revision == 1 and new_generator.calls == 0
    assert fresh.page_store.get(receipt.page_id).content == "hello"


def test_visible_hit_filter_and_detached_payloads(tmp_path):
    store = SQLiteDurableStore(tmp_path / "db")
    first = write(store)
    second = write(store, "second")
    foreign = write(store, namespace="foreign")
    assert [
        e.id
        for e in store.visible_entries(
            "a",
            [second.memory_id, first.memory_id, second.memory_id, foreign.memory_id],
        )
    ] == [second.memory_id, first.memory_id]
    detached = store.get_entry("a", first.memory_id)
    detached.meta["custom"].append("mutated")
    assert store.get_entry("a", first.memory_id).meta["custom"] == [1, {"x": True}]
    write(store, "delete", operation="delete", target=first.memory_id)
    assert [
        e.id for e in store.visible_entries("a", [first.memory_id, second.memory_id])
    ] == [second.memory_id]
    assert store.visible_entries("a", []) == []
    with pytest.raises(InvalidRequest):
        store.visible_entries("a", [second.memory_id] * 1001)
    snapshot = store.snapshot("a", limit=1)
    assert not snapshot.truncated
    write(store, "third")
    assert store.snapshot("a", limit=1).truncated


def test_provider_errors_are_safe_and_proposals_are_strict(tmp_path):
    from pydantic import ValidationError

    from jitmind.storage import ProposalFailure

    class BadProvider:
        def generate_single(self, **kwargs):
            raise RuntimeError("private payload credential /private/path")

    agent = MemoryAgent(
        generator=BadProvider(), durable_store=SQLiteDurableStore(tmp_path / "db")
    )
    with pytest.raises(ProposalFailure, match="^proposal_failure$"):
        agent.memorize_durable("hello", idempotency_key="k")
    assert agent.durable_store.status("default")["pages"] == 0
    with pytest.raises(ValidationError):
        Proposal(
            abstract="a",
            header="h",
            decorated="d",
            decision={"operation": "add", "fault_hook": "unsafe"},
        )
    with pytest.raises(InvalidRequest):
        IngestRequest.create("\ud800", "key", "hello")


def _conflict_worker(path, message, gate, output):
    try:
        store = SQLiteDurableStore(path, busy_timeout_ms=1000)
        gate.wait(timeout=15)
        receipt = store.ingest(
            IngestRequest.create("a", "shared", message), lambda _: proposal()
        )
        output.put(("ok", receipt.request_digest))
    except IdempotencyConflict:
        output.put(("conflict", ""))


def test_multiprocess_changed_payload_under_same_key_conflicts(tmp_path):
    path = tmp_path / "db"
    SQLiteDurableStore(path)
    ctx = multiprocessing.get_context("spawn")
    gate, output = ctx.Barrier(2), ctx.Queue()
    children = [
        ctx.Process(target=_conflict_worker, args=(str(path), message, gate, output))
        for message in ("one", "two")
    ]
    try:
        for child in children:
            child.start()
        results = [output.get(timeout=30) for _ in children]
        assert sorted(result[0] for result in results) == ["conflict", "ok"]
        assert SQLiteDurableStore(path).status("a")["operations"] == 1
    finally:
        _cleanup(children)
        output.close()
        output.join_thread()
    assert all(child.exitcode == 0 for child in children)


def _projection_crash_worker(path):
    def fault(stage):
        if stage == "after_projection":
            os._exit(38)

    store = SQLiteDurableStore(path, fault_hook=fault)
    store.drain_outbox("a")


def test_process_crash_after_projection_is_idempotent(tmp_path):
    path = tmp_path / "db"
    receipt = write(SQLiteDurableStore(path))
    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(target=_projection_crash_worker, args=(str(path),))
    child.start()
    try:
        child.join(timeout=20)
        assert child.exitcode == 38
        store = SQLiteDurableStore(path)
        assert store.status("a")["delivery_watermark"] == 1
        store.deliver_event("a", receipt.event_id)
        assert len(store.projected_entries("a")) == 1
    finally:
        _cleanup([child])


def test_unknown_schema_and_invalid_payload_preserve_database(tmp_path):
    path = tmp_path / "unknown"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE unrelated(value TEXT)")
    conn.commit()
    conn.close()
    original = path.read_bytes()
    with pytest.raises(SchemaMismatch):
        SQLiteDurableStore(path)
    assert path.read_bytes() == original
    store = SQLiteDurableStore(tmp_path / "db")
    receipt = write(store)
    conn = sqlite3.connect(tmp_path / "db")
    conn.execute("UPDATE facts SET payload=?", ('{"content":"private-payload"}',))
    conn.commit()
    conn.close()
    with pytest.raises(StorageFailure, match="^storage_failure$"):
        store.visible_entries("a", [receipt.memory_id])
    with pytest.raises(StorageFailure, match="^storage_failure$"):
        store.get_entry("a", receipt.memory_id)
    assert store.status("a")["operations"] == 1

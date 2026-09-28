"""Acceptance supplements using real local storage and the provisioned parser.

Only ENOSPC is injected. SQLite capacity and child exit are real, bounded faults.
No provider, source execution, package installation, or external fixture is used.
"""

import errno
import hashlib
import json
import multiprocessing
import os
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest

from jitmind.code_context import CodeContext, CodeQuery, NodeParser, RepoRegistry
from jitmind.code_memory.anchors import CodeMemoryService
from jitmind.code_memory.lesson_models import (
    DurableBindingAuthority,
    DurableFactAuthority,
    LessonProjection,
    ProposedAction,
)
from jitmind.code_memory.open_work import OpenWorkService
from jitmind.code_memory.preflight import PreflightService
from jitmind.code_memory.work_storage import Budget, WorkDatabase
from jitmind.schemas import MemoryEntry, Page
from jitmind.schemas.advanced_memory import AdvancedMemoryStore
from jitmind.schemas.memory import InMemoryMemoryStore
from jitmind.schemas.page import InMemoryPageStore
from jitmind.schemas.ttl_memory import TTLMemoryStore
from jitmind.schemas.ttl_page import TTLPageStore
from jitmind.scope import ScopeAuthority
from jitmind.storage import (
    DurableMemoryAdapter,
    DurablePageAdapter,
    IngestRequest,
    Proposal,
    SQLiteDurableStore,
    StorageFailure,
    stage_legacy,
)
from jitmind.utils import atomic_io


@pytest.mark.parametrize(
    "store_class,filename",
    [
        (AdvancedMemoryStore, "advanced_memory_state.json"),
        (InMemoryMemoryStore, "memory_state.json"),
        (TTLMemoryStore, "ttl_memory_state.json"),
        (InMemoryPageStore, "pages.json"),
        (TTLPageStore, "ttl_pages.json"),
    ],
    ids=["advanced", "memory", "ttl_memory", "pages", "ttl_pages"],
)
def test_enospc_at_legacy_publication_preserves_ack_and_bytes(
    tmp_path, monkeypatch, store_class, filename
):
    def reader():
        kwargs = (
            {"enable_auto_cleanup": False}
            if store_class in (AdvancedMemoryStore, TTLMemoryStore, TTLPageStore)
            else {}
        )
        return store_class(str(tmp_path), **kwargs)

    def append(store, body):
        store.add(
            Page(header="fixture", content=body)
            if store_class in (InMemoryPageStore, TTLPageStore)
            else body
        )

    def contents(store):
        state = store.load()
        return (
            state.abstracts
            if hasattr(state, "abstracts")
            else [p.content for p in state]
        )

    store = reader()
    append(store, "previously committed")
    path = tmp_path / filename
    previous = path.read_bytes()
    calls = []
    cause = OSError(errno.ENOSPC, "sanitized_fixture")

    def full(source, destination):
        # Exercise the actual post-fsync, pre-publication filesystem boundary.
        assert Path(destination) == path
        assert Path(source).is_file()
        assert b"unacknowledged memory" in Path(source).read_bytes()
        calls.append((Path(source).name, Path(destination).name))
        raise cause

    with monkeypatch.context() as fault:
        fault.setattr(atomic_io.os, "replace", full)
        with pytest.raises(atomic_io.PersistenceError) as raised:
            append(store, "unacknowledged memory")
    assert len(calls) == 1
    assert raised.value.__cause__ is cause
    assert cause.errno == errno.ENOSPC
    assert not raised.value.outcome_uncertain
    assert "sanitized_fixture" not in str(raised.value)
    assert path.read_bytes() == previous
    assert contents(store) == contents(reader()) == ["previously committed"]
    assert not list(tmp_path.glob("*.tmp"))
    append(store, "positive after fault removed")
    assert contents(reader()) == [
        "previously committed",
        "positive after fault removed",
    ]


def _proposal(body, operation="add", target=None):
    return Proposal(
        abstract=body,
        header="fixture",
        decorated=body,
        decision={"operation": operation, "target_id": target},
    )


def test_real_sqlite_full_rolls_back_every_effect_then_accepts_same_request(
    tmp_path, monkeypatch
):
    store = SQLiteDurableStore(tmp_path / "capacity.sqlite")
    original = store._connection
    native_errors = []
    capacities = []

    @contextmanager
    def capped(**kwargs):
        with original(**kwargs) as connection:
            pages = connection.execute("PRAGMA page_count").fetchone()[0]
            cap = connection.execute(f"PRAGMA max_page_count={pages}").fetchone()[0]
            capacities.append((pages, cap))
            try:
                yield connection
            except sqlite3.Error as exc:
                native_errors.append(exc.sqlite_errorcode)
                raise

    # Below the public 131072-character input bound; comfortably larger than
    # this empty DB's free space. This never attempts to fill the host disk.
    request = IngestRequest.create("tenant", "capacity-write", "x" * 100000)
    with monkeypatch.context() as fault:
        fault.setattr(store, "_connection", capped)
        with pytest.raises(StorageFailure, match="^storage_failure$"):
            store.ingest(request, lambda _: _proposal("bounded capacity fact"))
    assert native_errors == [sqlite3.SQLITE_FULL]
    assert capacities and all(count == cap for count, cap in capacities)
    fresh = SQLiteDurableStore(store.path)
    assert fresh.status("tenant") == {
        "revision": 0,
        "pending_events": 0,
        "delivery_watermark": 0,
        "pages": 0,
        "facts": 0,
        "operations": 0,
    }
    assert fresh.lookup_receipt(request) is None
    assert fresh.snapshot("tenant").entries == ()
    assert fresh.pending_events("tenant") == []
    assert fresh.projected_entries("tenant") == []
    with sqlite3.connect(store.path) as connection:
        for table in (
            "namespaces",
            "facts",
            "pages",
            "operations",
            "outbox",
            "projection",
        ):
            assert (
                connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
            )
        connection.execute("PRAGMA max_page_count=10000")
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    receipt = fresh.ingest(request, lambda _: _proposal("bounded capacity fact"))
    assert receipt.revision == 1
    assert (
        fresh.get_entry("tenant", receipt.memory_id).source_page_id == receipt.page_id
    )
    assert fresh.get_page("tenant", receipt.page_id).content == request.message
    assert fresh.lookup_receipt(request) == receipt
    assert len(fresh.pending_events("tenant")) == 1
    assert fresh.ingest(request, lambda _: pytest.fail("replay generated")) == receipt
    fresh.deliver_event("tenant", receipt.event_id)
    assert [entry.id for entry in fresh.projected_entries("tenant")] == [
        receipt.memory_id
    ]


def _interrupt_import(source, destination, stage):
    staged = stage_legacy(source, quiesced=True)

    def interrupt(current):
        if current == stage:
            os._exit(73)

    SQLiteDurableStore(destination, fault_hook=interrupt).import_staged(
        staged, "tenant"
    )


@pytest.mark.parametrize("stage", ["after_import_pages", "before_import_commit"])
def test_real_migration_exit_preserves_source_and_allows_retry(tmp_path, stage):
    source = tmp_path / "legacy"
    source.mkdir()
    fact = MemoryEntry(
        id="legacy-fact",
        content="migration fixture fact",
        source_page_id="0",
        t_created="2026-01-01T00:00:00+00:00",
        t_observed="2026-01-01T00:00:00+00:00",
    )
    page = Page(
        header="fixture",
        content="migration provenance",
        meta={"page_id": 0, "memory_id": fact.id},
    )
    (source / "advanced_memory_state.json").write_text(
        json.dumps({"entries": [fact.model_dump()]})
    )
    (source / "pages.json").write_text(json.dumps([page.model_dump()]))
    originals = {p.name: p.read_bytes() for p in source.iterdir()}
    path = tmp_path / "candidate.sqlite"
    SQLiteDurableStore(path)
    child = multiprocessing.get_context("spawn").Process(
        target=_interrupt_import, args=(str(source), str(path), stage)
    )
    try:
        child.start()
        child.join(20)
        assert child.exitcode == 73, "child must exit at the actual import boundary"
    finally:
        if child.is_alive():
            child.terminate()
            child.join(5)
        if child.is_alive():
            child.kill()
            child.join(5)
        assert not child.is_alive()
        child.close()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == originals
    fresh = SQLiteDurableStore(path)
    with sqlite3.connect(path) as connection:
        for table in (
            "namespaces",
            "pages",
            "facts",
            "operations",
            "outbox",
            "aliases",
            "imports",
        ):
            assert (
                connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
            )
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    staged = stage_legacy(source, quiesced=True)
    report = fresh.import_staged(staged, "tenant")
    assert report["page_count"] == report["fact_count"] == 1
    restored = fresh.get_entry("tenant", fact.id)
    assert restored.content == fact.content
    assert fresh.get_page("tenant", restored.source_page_id).content == page.content
    assert fresh.resolve_page_alias("tenant", "0") == restored.source_page_id
    assert fresh.import_staged(staged, "tenant") == report
    assert fresh.status("tenant")["revision"] == 1
    assert {p.name: p.read_bytes() for p in source.iterdir()} == originals


@pytest.mark.parametrize("delay_add", [False, True], ids=["projected", "add_pending"])
def test_real_parser_binding_projection_redelivery_and_primary_forget(
    tmp_path, delay_add
):
    sentinel = "MEMORY_ONLY_SENTINEL_83f91c"
    source = "def original(x):\n    return x + 123\n"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    path = checkout / "source.py"
    path.write_text(source)
    assert sentinel not in source
    repo = str(uuid4())
    authority = ScopeAuthority()
    authority.grant("alice", "tenant", [repo])

    def scope():
        return authority.context("alice", "tenant", str(uuid4()))

    registry = RepoRegistry()
    registry.register(repo, str(checkout.resolve()), "tenant")
    adapter = Path(
        os.environ.get(
            "JITMIND_TEST_GRAFT_ADAPTER",
            Path(__file__).resolve().parents[1] / "adapters/graft",
        )
    )
    node = os.environ.get("JITMIND_TEST_NODE") or shutil.which("node")
    assert node and adapter.is_dir(), (
        "Provision the actual J04 Node/tree-sitter adapter"
    )
    context = CodeContext(
        authority,
        registry,
        NodeParser(str(Path(node).resolve()), str(adapter.resolve())),
    )
    facts = SQLiteDurableStore(tmp_path / "facts.sqlite")

    def ingest(key, operation="add", target=None):
        return facts.ingest(
            IngestRequest.create("tenant", key, sentinel),
            lambda _: _proposal(sentinel, operation, target),
        )

    fact = ingest("add")
    duplicate = ingest("noop", "noop", fact.memory_id)
    bindings = CodeMemoryService(
        authority, facts, context, tmp_path / "bindings.sqlite"
    )

    def build():
        result = context.build(scope(), repo, deadline_ms=10000)
        assert result.status == "ok", result
        return result

    first = build()
    binding = bindings.create_binding(
        scope(),
        logical_binding_id="logical",
        fact_version_id=fact.memory_id,
        repo_id=repo,
        snapshot_id=first.snapshot_id,
        path="source.py",
        qualified_name="original",
    )
    assert binding.validation_state == "verified_current"
    database = WorkDatabase(tmp_path / "work.sqlite")
    work = OpenWorkService(database, authority)
    item = work.create(
        scope(),
        repo,
        summary="Complete the outstanding review",
        expected_revision=0,
        idempotency_key="obligation",
        binding_refs=("logical",),
        decision_refs=(fact.memory_id,),
    )
    assert (
        work.binding_locations(scope(), repo, item.id, bindings)[0]["path"]
        == "source.py"
    )
    projection = LessonProjection(
        database,
        authority,
        DurableFactAuthority(facts, authority),
        binding_authority=DurableBindingAuthority(bindings, authority),
    )

    def project(version):
        return projection.project_binding_observation(
            scope(),
            repo,
            bindings,
            logical_binding_id="logical",
            lesson_id="lesson",
            version=version,
            source_revision=fact.revision,
            body=sentinel,
        )

    assert project(1)
    service = PreflightService(projection)
    delivery_scope = scope()
    session = service.start_session(delivery_scope, repo)
    original_target = ProposedAction(repo, "source.py", "original")
    before = service.deliver(
        delivery_scope, session, original_target, delivery_key="original"
    )
    assert before.state == "delivered" and sentinel in before.payload
    assert facts.status("tenant")["pending_events"] == 2
    assert facts.status("tenant")["delivery_watermark"] == 0
    assert facts.get_entry("tenant", fact.memory_id).content == sentinel

    # Source generation and fact outbox revision are independent watermarks.
    path.rename(checkout / "moved.py")
    stale = context.query(
        scope(),
        CodeQuery(
            operation="find_code",
            repo_id=repo,
            snapshot_id=first.snapshot_id,
            query="original",
        ),
    )
    assert stale.freshness["state"] != "fresh"
    assert bindings.get_current(scope(), "logical").validation_state == "unknown"
    current = build()
    assert (
        current.generation > first.generation
        and current.snapshot_id != first.snapshot_id
    )
    relocated = bindings.revalidate(
        scope(),
        "logical",
        snapshot_id=current.snapshot_id,
        expected_revision=binding.revision,
    )
    assert (
        relocated.reason_code == "file_move"
        and relocated.validation_state == "verified_current"
    )
    assert relocated.source_generation == current.generation
    assert (
        work.binding_locations(scope(), repo, item.id, bindings)[0]["path"]
        == "moved.py"
    )
    assert project(2)
    assert project(2)  # Same material version is idempotent.
    assert not project(1)  # Out-of-order lesson cannot replace the moved version.
    target = ProposedAction(repo, "moved.py", "original")
    lessons = projection.candidates(scope(), target, Budget(2))
    assert len(lessons) == 1 and lessons[0].version == 2
    delivered = service.deliver(delivery_scope, session, target, delivery_key="moved")
    assert delivered.state == "delivered" and sentinel in delivered.payload
    assert bindings.project_fact(
        scope(), fact_version_id=fact.memory_id, event_revision=fact.revision
    )
    assert not bindings.project_fact(
        scope(), fact_version_id=fact.memory_id, event_revision=fact.revision
    )
    facts.deliver_event("tenant", duplicate.event_id)
    facts.deliver_event("tenant", duplicate.event_id)
    assert facts.status("tenant")["delivery_watermark"] == 0
    if delay_add:
        assert facts.status("tenant")["pending_events"] == 1
        assert facts.projected_entries("tenant") == []
    else:
        facts.deliver_event("tenant", fact.event_id)
        assert facts.status("tenant")["pending_events"] == 0
        assert facts.status("tenant")["delivery_watermark"] == duplicate.revision
        assert [entry.id for entry in facts.projected_entries("tenant")] == [
            fact.memory_id
        ]
    assert (
        bindings.get_current(scope(), "logical").source_generation == current.generation
    )

    deleted = ingest("forget", "delete", fact.memory_id)
    # Delay all deletion projections. Current output must already be hidden.
    assert facts.status("tenant")["pending_events"] == 1 + int(delay_add)
    assert facts.get_entry("tenant", fact.memory_id) is None
    assert facts.snapshot("tenant").entries == ()
    assert facts.projected_entries("tenant") == []
    assert DurableMemoryAdapter(facts, "tenant").load().abstracts == []
    assert DurablePageAdapter(facts, "tenant").load() == []
    for receipt in (fact, duplicate, deleted):
        assert facts.get_page("tenant", receipt.page_id) is None
        assert sentinel not in repr(facts.memory_update(receipt))
        assert sentinel not in repr(facts.receipt_content(receipt))
        assert sentinel not in receipt.model_dump_json()
    assert projection.candidates(scope(), target, Budget(2)) == []
    for action, key in ((original_target, "original"), (target, "moved")):
        result = service.deliver(delivery_scope, session, action, delivery_key=key)
        assert result.state == "nothing_relevant" and result.payload == ""
    assert bindings.get_current(scope(), "logical").reason_code == "fact_unavailable"
    assert not project(3)
    # New deletion metadata first, then delayed old events: no resurrection.
    assert bindings.project_fact(
        scope(), fact_version_id=fact.memory_id, event_revision=deleted.revision
    )
    assert not bindings.project_fact(
        scope(), fact_version_id=fact.memory_id, event_revision=fact.revision
    )
    report = projection.forget(
        scope(), fact.memory_id, tombstone_revision=deleted.revision
    )
    assert report["purge"] == "projection_purged"
    assert not project(1)
    facts.deliver_event("tenant", deleted.event_id)
    if delay_add:
        assert facts.status("tenant")["delivery_watermark"] == 0
        assert facts.pending_events("tenant")[0]["event_id"] == fact.event_id
    for receipt in (duplicate, fact, deleted):
        facts.deliver_event("tenant", receipt.event_id)
    assert facts.status("tenant")["pending_events"] == 0
    assert facts.status("tenant")["delivery_watermark"] == deleted.revision
    assert facts.projected_entries("tenant") == []
    assert projection.candidates(scope(), target, Budget(2)) == []
    assert work.get(scope(), repo, item.id).status == "open"
    assert work.get(scope(), repo, item.id).revision == 1
    assert work.binding_locations(scope(), repo, item.id, bindings) == (
        {"binding_id": "logical", "repo_id": repo, "grain": "repo"},
    )
    history = bindings.history(scope(), "logical")
    assert len(history) == 2 and sentinel not in repr(history)
    assert sentinel not in repr(work.history(scope(), repo, item.id))
    with database.connect(Budget(2)) as connection:
        assert connection.execute("SELECT count(*) FROM lessons").fetchone()[0] == 0
        for table in (
            "history",
            "work_receipts",
            "deliveries",
            "delivery_receipts",
            "tombstones",
        ):
            assert sentinel not in repr(
                [tuple(row) for row in connection.execute(f"SELECT * FROM {table}")]
            )
    with bindings.store.connection() as connection:
        for table in ("history", "requests", "sources", "facts"):
            assert sentinel not in repr(
                connection.execute(f"SELECT * FROM {table}").fetchall()
            )
    # J04 indexes authorized source, not fact bodies. Forgetting a memory does
    # not delete source files; raw code remains readable with correct identity.
    code = context.query(
        scope(),
        CodeQuery(
            operation="find_code",
            repo_id=repo,
            snapshot_id=current.snapshot_id,
            query="original",
        ),
    )
    assert code.status == "ok" and code.freshness["state"] == "fresh"
    assert len(code.results) == 1 and code.results[0].source == source
    assert code.generation == current.generation
    assert code.results[0].digest == hashlib.sha256(source.encode()).hexdigest()
    assert sentinel not in code.model_dump_json()
    assert (checkout / "moved.py").read_text() == source

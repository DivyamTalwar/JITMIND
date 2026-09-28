"""Actual J02 + J04 + J05 integration, using the provisioned local parser only."""

import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from jitmind.code_context import CodeContext, NodeParser, RepoRegistry
from jitmind.code_memory.anchors import CodeMemoryService
from jitmind.code_memory.lesson_models import (
    DurableBindingAuthority,
    DurableFactAuthority,
    LessonProjection,
    ProposedAction,
)
from jitmind.code_memory.open_work import OpenWorkService
from jitmind.code_memory.preflight import PreflightPolicy, PreflightService
from jitmind.code_memory.work_storage import WorkDatabase
from jitmind.scope import ScopeAuthority
from jitmind.storage import IngestRequest, Proposal, SQLiteDurableStore


@pytest.mark.parametrize(
    "source,state",
    [("# code removed\n", "orphaned_confirmed"), ("invalid syntax ???\n", "unknown")],
)
def test_j16_j31_real_binding_move_disappearance_forget_keep_obligation(
    tmp_path, source, state
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    path = checkout / "source.py"
    path.write_text("def original(x):\n    return x + 123\n")
    repo = str(uuid4())
    authority = ScopeAuthority()
    authority.grant("alice", "namespace", [repo])

    def scope():
        return authority.context("alice", "namespace", str(uuid4()))

    registry = RepoRegistry()
    registry.register(repo, str(checkout.resolve()), "namespace")
    adapter = Path(
        os.environ.get(
            "JITMIND_TEST_GRAFT_ADAPTER",
            Path(__file__).resolve().parents[1] / "adapters/graft",
        )
    )
    node = os.environ.get("JITMIND_TEST_NODE") or shutil.which("node")
    assert node and adapter.is_dir(), (
        "J04 controlled adapter must be provisioned for this integration test"
    )
    context = CodeContext(
        authority,
        registry,
        NodeParser(str(Path(node).resolve()), str(adapter.resolve())),
    )
    facts = SQLiteDurableStore(tmp_path / "facts.sqlite")

    def ingest(operation, target=None):
        return facts.ingest(
            IngestRequest.create("namespace", str(uuid4()), "private lesson body"),
            lambda _: Proposal(
                abstract="private lesson body",
                header="header",
                decorated="private lesson body",
                decision={"operation": operation, "target_id": target},
            ),
        )

    fact = ingest("add")
    bindings = CodeMemoryService(
        authority, facts, context, tmp_path / "bindings.sqlite"
    )

    def build():
        result = context.build(scope(), repo, deadline_ms=10000)
        assert result.status == "ok", result
        return result.snapshot_id

    binding = bindings.create_binding(
        scope(),
        logical_binding_id="logical",
        fact_version_id=fact.memory_id,
        repo_id=repo,
        snapshot_id=build(),
        path="source.py",
        qualified_name="original",
    )
    database = WorkDatabase(tmp_path / "work.sqlite")
    work = OpenWorkService(database, authority)
    item = work.create(
        scope(),
        repo,
        summary="Complete review after relocation",
        expected_revision=0,
        idempotency_key="create",
        binding_refs=("logical",),
        decision_refs=(fact.memory_id,),
    )
    assert (
        work.binding_locations(scope(), repo, item.id, bindings)[0]["path"]
        == "source.py"
    )
    moved = checkout / "moved.py"
    path.rename(moved)
    relocated = bindings.revalidate(
        scope(), "logical", snapshot_id=build(), expected_revision=binding.revision
    )
    assert relocated.reason_code == "file_move"
    assert (
        work.binding_locations(scope(), repo, item.id, bindings)[0]["path"]
        == "moved.py"
    )
    moved.write_text(source)
    orphan = bindings.revalidate(
        scope(), "logical", snapshot_id=build(), expected_revision=relocated.revision
    )
    assert orphan.validation_state == state
    assert work.binding_locations(scope(), repo, item.id, bindings) == (
        {"binding_id": "logical", "repo_id": repo, "grain": "repo"},
    )
    assert work.get(scope(), repo, item.id).status == "open"
    projection = LessonProjection(
        database,
        authority,
        DurableFactAuthority(facts, authority),
        binding_authority=DurableBindingAuthority(bindings, authority),
    )
    target = ProposedAction(repo, "moved.py", "original")
    projection.project_binding_observation(
        scope(),
        repo,
        bindings,
        logical_binding_id="logical",
        lesson_id="bound-lesson",
        version=1,
        source_revision=fact.revision,
        body="private lesson body",
    )
    preflight = PreflightService(
        projection, policy=PreflightPolicy(deadline_seconds=3.0)
    )
    request_scope = scope()
    session = preflight.start_session(request_scope, repo)
    assert preflight.deliver(request_scope, session, target).state == "nothing_relevant"
    # An independently validated file-level observation is still eligible.
    projection.import_observation(
        scope(),
        ProposedAction(repo, "moved.py"),
        lesson_id="file-lesson",
        fact_id=fact.memory_id,
        version=1,
        source_revision=fact.revision,
        body="private lesson body",
        source_verified=True,
        confidence=0.9,
    )
    assert preflight.deliver(request_scope, session, target).state == "delivered"
    deletion = ingest("delete", fact.memory_id)
    assert preflight.deliver(request_scope, session, target).payload == ""
    bindings.project_fact(
        scope(), fact_version_id=fact.memory_id, event_revision=deletion.revision
    )
    report = projection.forget(
        scope(), fact.memory_id, tombstone_revision=deletion.revision
    )
    assert report["visibility"] == "hidden_by_primary_authority"
    historical = bindings.history(scope(), "logical")
    assert len(historical) == 3 and "private lesson body" not in repr(historical)
    assert work.get(scope(), repo, item.id).status == "open"
    assert work.get(scope(), repo, item.id).revision == 1
    assert (
        work.binding_locations(scope(), repo, item.id, bindings)[0]["grain"] == "repo"
    )


@pytest.fixture
def bound_host(tmp_path):
    """All four actual owner services; projection deliberately independently timed."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    path = checkout / "source.py"
    path.write_text("def original(x):\n    return x + 123\n")
    repo = str(uuid4())
    authority = ScopeAuthority()
    authority.grant("alice", "namespace", [repo])
    scope = authority.context("alice", "namespace", "request")
    registry = RepoRegistry()
    registry.register(repo, str(checkout.resolve()), "namespace")
    adapter = Path(
        os.environ.get(
            "JITMIND_TEST_GRAFT_ADAPTER",
            Path(__file__).resolve().parents[1] / "adapters/graft",
        )
    )
    node = os.environ.get("JITMIND_TEST_NODE") or shutil.which("node")
    assert node and adapter.is_dir(), "Controlled J04 adapter must be provisioned"
    context = CodeContext(
        authority,
        registry,
        NodeParser(str(Path(node).resolve()), str(adapter.resolve())),
    )
    facts = SQLiteDurableStore(tmp_path / "facts.sqlite")
    fact = facts.ingest(
        IngestRequest.create("namespace", "fact", "private bound lesson"),
        lambda _: Proposal(
            abstract="private bound lesson",
            header="header",
            decorated="private bound lesson",
            decision={"operation": "add"},
        ),
    )
    bindings = CodeMemoryService(
        authority, facts, context, tmp_path / "bindings.sqlite"
    )

    def build():
        result = context.build(scope, repo, deadline_ms=10000)
        assert result.status == "ok", result
        return result.snapshot_id

    binding = bindings.create_binding(
        scope,
        logical_binding_id="logical",
        fact_version_id=fact.memory_id,
        repo_id=repo,
        snapshot_id=build(),
        path="source.py",
        qualified_name="original",
    )
    database = WorkDatabase(tmp_path / "work.sqlite")
    projection = LessonProjection(
        database,
        authority,
        DurableFactAuthority(facts, authority),
        can_author=lambda *args: True,
        binding_authority=DurableBindingAuthority(bindings, authority),
    )
    projection.project_binding_observation(
        scope,
        repo,
        bindings,
        logical_binding_id="logical",
        lesson_id="observation",
        version=1,
        source_revision=fact.revision,
        body="private bound observation",
    )
    target = ProposedAction(repo, "source.py", "original")
    assert projection.author_instruction(
        scope,
        target,
        lesson_id="instruction",
        fact_id=fact.memory_id,
        version=1,
        source_revision=fact.revision,
        body="private bound trusted instruction",
        binding_id="logical",
        binding_revision=binding.revision,
    )
    service = PreflightService(projection, policy=PreflightPolicy(deadline_seconds=3.0))
    session = service.start_session(scope, repo)
    work = OpenWorkService(database, authority)
    item = work.create(
        scope,
        repo,
        summary="Review pending source change",
        expected_revision=0,
        idempotency_key="open",
        binding_refs=("logical",),
    )
    return locals()


@pytest.mark.parametrize(
    "source,state",
    [
        ("def original(x):\n    return x + 456\n", "changed"),
        ("invalid syntax ???\n", "unknown"),
        ("# removed\n", "orphaned_confirmed"),
        ("@cached\ndef original(x):\n    return x + 123\n", "unknown"),
    ],
)
def test_known_j05_change_blocks_delayed_projection_and_receipt(
    bound_host, source, state
):
    h = bound_host
    service, scope, session, target = (
        h[k] for k in ("service", "scope", "session", "target")
    )
    # This fixture verifies source invalidation, not shared-runner fsync speed.
    from dataclasses import replace

    service.policy = replace(service.policy, deadline_seconds=3.0)
    first = service.deliver(scope, session, target, delivery_key="saved")
    assert (
        first.state == "delivered" and "Trusted authored instruction" in first.payload
    )
    h["path"].write_text(source)
    current = h["bindings"].revalidate(
        h["authority"].context("alice", "namespace", str(uuid4())),
        "logical",
        snapshot_id=h["build"](),
        expected_revision=h["binding"].revision,
    )
    assert current.validation_state == state
    # No J06 projection operation follows the J05 revalidation.
    replay = service.deliver(scope, session, target, delivery_key="saved")
    assert replay.payload == "" and replay.state == "nothing_relevant"
    fresh = service.deliver(scope, service.start_session(scope, h["repo"]), target)
    assert fresh.payload == "" and fresh.state == "nothing_relevant"
    assert h["work"].get(scope, h["repo"], h["item"].id).status == "open"


def test_bound_records_require_available_authority_and_recover(bound_host):
    h = bound_host
    adapter = h["projection"].binding_authority
    h["projection"].binding_authority = None
    result = h["service"].deliver(h["scope"], h["session"], h["target"])
    assert (
        result.state == "unavailable"
        and result.reason == "binding_authority_unavailable"
    )
    assert result.payload == ""
    h["projection"].binding_authority = adapter
    assert (
        h["service"].deliver(h["scope"], h["session"], h["target"]).state == "delivered"
    )


def test_j05_known_change_after_delivery_commit_blocks_response(
    bound_host, monkeypatch
):
    from contextlib import contextmanager

    h = bound_host
    h["path"].write_text("def original(x):\n    return x + 456\n")
    snapshot = h["build"]()
    transaction = h["database"].transaction

    @contextmanager
    def revalidate_after_commit(db):
        with transaction(db):
            yield
        changed = h["bindings"].revalidate(
            h["authority"].context("alice", "namespace", str(uuid4())),
            "logical",
            snapshot_id=snapshot,
            expected_revision=h["binding"].revision,
        )
        assert changed.validation_state == "changed"

    monkeypatch.setattr(h["database"], "transaction", revalidate_after_commit)
    result = h["service"].deliver(
        h["scope"], h["session"], h["target"], delivery_key="saved"
    )
    assert result.payload == "" and result.reason == "authority_changed"
    monkeypatch.setattr(h["database"], "transaction", transaction)
    assert (
        h["service"]
        .deliver(h["scope"], h["session"], h["target"], delivery_key="saved")
        .payload
        == ""
    )
    assert h["work"].get(h["scope"], h["repo"], h["item"].id).status == "open"


def test_unobserved_live_source_change_is_not_claimed_atomic(bound_host):
    h = bound_host
    h["path"].write_text("def original(x):\n    return x + 456\n")
    # Without an authoritative observed change, this adapter checks the recorded
    # J05 revision only. This explicit limit is not live filesystem atomicity.
    result = h["service"].deliver(h["scope"], h["session"], h["target"])
    assert result.state == "delivered"


def test_binding_primary_retraction_suppresses_delayed_projection(bound_host):
    h = bound_host
    assert (
        h["service"]
        .deliver(h["scope"], h["session"], h["target"], delivery_key="saved")
        .state
        == "delivered"
    )
    deletion = h["facts"].ingest(
        IngestRequest.create("namespace", "delete", "forget"),
        lambda _: Proposal(
            abstract="forget",
            header="header",
            decorated="forget",
            decision={"operation": "delete", "target_id": h["fact"].memory_id},
        ),
    )
    h["bindings"].project_fact(
        h["scope"],
        fact_version_id=h["fact"].memory_id,
        event_revision=deletion.revision,
    )
    assert (
        h["service"]
        .deliver(h["scope"], h["session"], h["target"], delivery_key="saved")
        .payload
        == ""
    )
    assert h["work"].get(h["scope"], h["repo"], h["item"].id).status == "open"


@pytest.mark.parametrize("mode", ["IMMEDIATE", "EXCLUSIVE"])
def test_binding_authority_independent_process_contention(bound_host, mode):
    import multiprocessing
    import time

    from test_preflight import _hold_lock

    h = bound_host
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    process = ctx.Process(
        target=_hold_lock, args=(str(h["bindings"].store.path), mode, ready, release)
    )
    process.start()
    try:
        assert ready.wait(5)
        # EXCLUSIVE tests the real deadline; IMMEDIATE is a read-compatibility
        # control and still must satisfy the unchanged 300 ms ceiling below.
        h["service"].policy = (
            PreflightPolicy()
            if mode == "EXCLUSIVE"
            else PreflightPolicy(deadline_seconds=3.0)
        )
        started = time.monotonic()
        result = h["service"].deliver(h["scope"], h["session"], h["target"])
        elapsed = time.monotonic() - started
        assert elapsed < 0.3
        assert result.state == ("deferred" if mode == "EXCLUSIVE" else "delivered")
        print(f"binding lock={mode} seconds={elapsed:.6f} state={result.state}")
    finally:
        release.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0
    if result.state == "deferred":
        h["service"].policy = PreflightPolicy(deadline_seconds=3.0)
        assert (
            h["service"].deliver(h["scope"], h["session"], h["target"]).state
            == "delivered"
        )


def test_indeterminate_binding_callback_is_explicit_unavailability(bound_host):
    from types import SimpleNamespace

    h = bound_host
    h["projection"].binding_authority = SimpleNamespace(is_current=lambda *args: None)
    result = h["service"].deliver(h["scope"], h["session"], h["target"])
    assert result.state == "unavailable"
    assert result.reason == "binding_authority_unavailable" and not result.payload

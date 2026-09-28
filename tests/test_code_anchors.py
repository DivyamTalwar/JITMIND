"""Real scoped source snapshots + SQLite authority/bindings; no provider calls."""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest
from pydantic import ValidationError

from jitmind.code_context import CodeContext, NodeParser, RepoRegistry
from jitmind.code_memory import (
    BindingConflict,
    BindingStorageError,
    BindingUnavailable,
    CodeBinding,
    CodeMemoryService,
)
from jitmind.scope import ScopeAuthority, ScopeContext, ScopeDenied
from jitmind.storage import IngestRequest, Proposal, SQLiteDurableStore


class Host:
    def __init__(self, tmp):
        self.root = tmp / "checkout"
        self.root.mkdir()
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.repo = str(uuid4())
        self.auth = ScopeAuthority()
        self.auth.grant("alice", "opaque/team", (self.repo,))
        self.counter = 0
        self.registry = RepoRegistry()
        self.registry.register(self.repo, str(self.root.resolve()), "opaque/team")
        adapter = Path(
            os.environ.get(
                "JITMIND_TEST_GRAFT_ADAPTER",
                Path(__file__).resolve().parents[1] / "adapters/graft",
            )
        )
        node = os.environ.get("JITMIND_TEST_NODE") or shutil.which("node")
        assert node and adapter.is_dir(), (
            "Provision J04 Node adapter for integration tests"
        )
        self.context = CodeContext(
            self.auth,
            self.registry,
            NodeParser(str(Path(node).resolve()), str(adapter)),
        )
        self.facts = SQLiteDurableStore(tmp / "facts.db")
        self.path = tmp / "bindings.db"
        self.service = CodeMemoryService(self.auth, self.facts, self.context, self.path)
        self.receipt = self.ingest("add")
        self.fact = self.receipt.memory_id
        self.write("main.py", "def original(x):\n    return x + 123\n")

    def scope(self, request=None):
        self.counter += 1
        return self.auth.context("alice", "opaque/team", request or f"r-{self.counter}")

    def write(self, path, source):
        (self.root / path).write_text(source)

    def ingest(self, operation, target=None):
        request = IngestRequest.create(
            "opaque/team", str(uuid4()), "private authoritative fact"
        )
        proposal = Proposal(
            abstract="private authoritative fact",
            header="fact",
            decorated="fact",
            decision={"operation": operation, "target_id": target},
        )
        return self.facts.commit_proposal(
            request, self.facts.snapshot("opaque/team").revision, proposal
        )

    def build(self):
        # J04 snapshots are scoped to principal and authorization version, not request ID.
        page = self.context.build(self.scope(), self.repo, deadline_ms=10000)
        assert page.status == "ok", page
        return page.snapshot_id

    def create(self, name="original", path="main.py", logical="binding"):
        snapshot = self.build()
        return self.service.create_binding(
            self.scope(),
            logical_binding_id=logical,
            fact_version_id=self.fact,
            repo_id=self.repo,
            snapshot_id=snapshot,
            path=path,
            qualified_name=name,
        )

    def revalidate(self, old, snapshot=None, scope=None):
        return self.service.revalidate(
            scope or self.scope(),
            old.logical_binding_id,
            snapshot_id=snapshot or self.build(),
            expected_revision=old.revision,
        )


@pytest.fixture
def host(tmp_path):
    return Host(tmp_path)


def test_j11_samefile_rename_persists_fact_and_revision(host):
    old = host.create()
    host.write("main.py", "def renamed(x):\n    return x + 123\n")
    new = host.revalidate(old)
    assert (new.validation_state, new.reason_code) == (
        "verified_current",
        "same_file_rename",
    )
    assert new.fact_version_id == old.fact_version_id
    assert new.raw_source_digest != old.raw_source_digest
    assert new.body_digest == old.body_digest
    assert new.supersedes_binding_id == old.binding_id
    assert new.revision == 2
    restarted = CodeMemoryService(host.auth, host.facts, host.context, host.path)
    assert restarted.get_current(host.scope(), "binding") == new
    assert restarted.history(host.scope(), "binding") == (old, new)
    assert host.facts.get_entry("opaque/team", host.fact).status == "active"


@pytest.mark.parametrize("rename", [False, True])
def test_j12_move_and_rename_move(host, rename):
    old = host.create()
    (host.root / "main.py").rename(host.root / "moved.py")
    if rename:
        host.write("moved.py", "def renamed(x):\n    return x + 123\n")
    new = host.revalidate(old)
    assert new.reason_code == ("rename_and_move" if rename else "file_move")
    assert new.path == "moved.py" and new.fact_version_id == old.fact_version_id
    assert len(host.service.history(host.scope(), "binding")) == 2


def test_j13_all_identical_getters_are_ambiguous_and_evidence_bounded(host):
    host.write("main.py", "def get():\n    return 1\n")
    old = host.create("get")
    (host.root / "main.py").unlink()
    for n in range(12):
        host.write(f"get{n}.py", "def get():\n    return 1\n")
    new = host.revalidate(old)
    assert new.validation_state == "unknown"
    assert new.reason_code == "ambiguous_candidates"
    assert new.candidate_count == 12 and len(new.evidence_refs) == 8
    assert "return 1" not in new.model_dump_json()


def test_j14_stale_snapshot_and_shifted_lines_never_read_current_disk(host):
    old = host.create()
    host.write("main.py", "\n" * 20 + 'def other():\n    return "unrelated"\n')
    new = host.revalidate(old, snapshot=old.snapshot_id)
    assert new.validation_state == "unknown"
    assert new.raw_source_digest == old.raw_source_digest
    assert new.path == old.path
    assert (
        host.service.history(host.scope(), "binding", snapshot_id=old.snapshot_id)[0]
        == old
    )


def test_j15_missing_checkout_preserves_facts_and_history(host):
    old = host.create()
    shutil.rmtree(host.root)
    new = host.revalidate(old, snapshot=old.snapshot_id)
    assert new.validation_state == "unknown"
    assert len(host.facts.snapshot("opaque/team").entries) == 1
    assert len(host.service.history(host.scope(), "binding")) == 2


def test_changed_body_stays_review_required_without_rebasing_baseline(host):
    old = host.create()
    host.write("main.py", "def original(x):\n    return x + 124\n")
    new = host.revalidate(old)
    assert new.validation_state == "changed" and new.review_required
    assert new.body_digest == old.body_digest
    again = host.revalidate(new)
    assert again.validation_state == "changed"
    assert host.facts.get_entry("opaque/team", host.fact).status == "active"


def test_wrong_branch_never_inherits_current_verification(host):
    old = host.create()
    subprocess.run(
        ["git", "-C", str(host.root), "symbolic-ref", "HEAD", "refs/heads/other"],
        check=True,
    )
    host.write("main.py", "def branch_only():\n    return False\n")
    current = host.service.get_current(host.scope(), "binding")
    assert current.validation_state == "unknown"
    assert current.reason_code == "snapshot_mismatch"
    new = host.revalidate(old)
    assert new.validation_state != "verified_current"


@pytest.mark.parametrize(
    "source", ["broken syntax !!!\n", "def f(x):\n    return x\n" * 51]
)
def test_partial_parse_or_budgets_are_unknown(host, source):
    old = host.create()
    host.write("main.py", source)
    assert host.revalidate(old).validation_state == "unknown"


def test_unsupported_coverage_is_unknown(host):
    old = host.create()
    (host.root / "main.py").unlink()
    host.write("unindexed.js", "function original(x) { return x + 123; }")
    assert host.revalidate(old).validation_state == "unknown"


def test_positive_absence_orphans_only_binding(host):
    old = host.create()
    host.write("main.py", "# declaration removed\n")
    new = host.revalidate(old)
    assert new.validation_state == "orphaned_confirmed"
    assert host.facts.get_entry("opaque/team", host.fact).status == "active"


def test_near_identical_short_function_remains_unknown(host):
    old = host.create()
    host.write("main.py", "def other(x):\n    return x + 124\n")
    new = host.revalidate(old)
    assert (
        new.validation_state == "unknown" and new.reason_code == "plausible_candidates"
    )


def test_qualified_methods_and_signature_constraints(host):
    host.write("main.py", "class A:\n    def get(self, x):\n        return x + 123\n")
    old = host.create("A.get")
    host.write(
        "main.py", "class A:\n    def renamed(self, x, y):\n        return x + 123\n"
    )
    new = host.revalidate(old)
    assert new.validation_state != "verified_current"


def test_duplicate_same_identity_is_unknown(host):
    old = host.create()
    host.write("main.py", "def original(x):\n    return x + 123\n" * 2)
    assert host.revalidate(old).reason_code == "ambiguous_candidates"


def test_scope_checked_before_database_or_source_and_again_on_return(host, monkeypatch):
    sentinel = "PRIVATE_SENTINEL"
    denied = ScopeContext("alice", "opaque/team", (host.repo,), 1, sentinel)
    with pytest.raises(ScopeDenied, match="^Scope denied$"):
        host.service.get_current(denied, sentinel)
    assert not host.path.exists()
    old = host.create()
    original = host.service.sources.freshness

    def revoke(*args):
        page = original(*args)
        host.auth.revoke("alice", "opaque/team")
        return page

    monkeypatch.setattr(host.service.sources, "freshness", revoke)
    with pytest.raises(ScopeDenied, match="^Scope denied$"):
        host.service.get_current(host.scope(), old.logical_binding_id)


def test_namespace_isolation(host):
    old = host.create()
    host.auth.grant("bob", "other/team", (host.repo,))
    scope = host.auth.context("bob", "other/team", "request")
    assert host.service.get_current(scope, old.logical_binding_id) is None
    assert host.service.history(scope, old.logical_binding_id) == ()


def test_two_revalidations_only_one_cas_wins_without_sleeps(host, monkeypatch):
    old = host.create()
    snapshot = host.build()
    barrier = Barrier(2)
    original = host.service.sources.load

    def synchronize(*args):
        result = original(*args)
        barrier.wait(timeout=10)
        return result

    monkeypatch.setattr(host.service.sources, "load", synchronize)
    scopes = [host.scope(), host.scope()]

    def run(scope):
        try:
            return host.revalidate(old, snapshot, scope)
        except BindingConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, scopes))
    assert sum(isinstance(r, CodeBinding) for r in results) == 1
    assert results.count("conflict") == 1
    assert len(host.service.history(host.scope(), "binding")) == 2


def test_rebuild_during_validation_conflicts(host, monkeypatch):
    old = host.create()
    original = host.service.sources.load

    def rebuild(*args):
        view = original(*args)
        host.build()
        return view

    monkeypatch.setattr(host.service.sources, "load", rebuild)
    with pytest.raises(BindingConflict):
        host.revalidate(old, snapshot=old.snapshot_id)
    assert len(host.service.history(host.scope(), "binding")) == 1


def test_request_idempotency_and_conflicting_reuse(host):
    old = host.create()
    snapshot = host.build()
    scope = host.scope("idempotent")
    first = host.revalidate(old, snapshot, scope)
    assert host.revalidate(old, snapshot, scope) == first
    assert len(host.service.history(host.scope(), "binding")) == 2
    with pytest.raises(BindingConflict):
        host.revalidate(first, snapshot, scope)


def test_real_j02_projection_redelivery_and_terminal_tombstone(host):
    old = host.create()
    initial = host.facts.pending_events("opaque/team")[0]
    assert host.service.project_fact(
        host.scope(), fact_version_id=host.fact, event_revision=initial["revision"]
    )
    assert not host.service.project_fact(
        host.scope(), fact_version_id=host.fact, event_revision=initial["revision"]
    )
    deleted = host.ingest("delete", host.fact)
    events = host.facts.pending_events("opaque/team")
    assert len(events) == 2
    assert host.service.project_fact(
        host.scope(), fact_version_id=host.fact, event_revision=deleted.revision
    )
    assert not host.service.project_fact(
        host.scope(), fact_version_id=host.fact, event_revision=initial["revision"]
    )
    # Even a later event cannot revive a terminal fact version.
    assert not host.service.project_fact(
        host.scope(), fact_version_id=host.fact, event_revision=deleted.revision + 1
    )
    host.facts.deliver_event("opaque/team", initial["event_id"])
    host.facts.deliver_event("opaque/team", deleted.event_id)
    assert host.facts.projected_entries("opaque/team") == []
    result = host.service.get_current(host.scope(), "binding")
    assert result.reason_code == "fact_unavailable"
    assert host.service.history(host.scope(), "binding") == (old,)
    assert "private authoritative fact" not in host.path.read_bytes().decode(
        "utf8", errors="ignore"
    )
    with pytest.raises(BindingUnavailable):
        host.revalidate(old, snapshot=old.snapshot_id)


def test_deleted_authority_with_no_projection_cannot_return_verified(host):
    old = host.create()
    host.ingest("delete", host.fact)
    assert (
        host.service.get_current(host.scope(), "binding").reason_code
        == "fact_unavailable"
    )
    assert host.service.history(
        host.scope(), "binding", snapshot_id=old.snapshot_id
    ) == (old,)


def test_immutable_models_and_bad_lookup_inputs(host):
    old = host.create()
    with pytest.raises(ValidationError):
        old.path = "new.py"
    with pytest.raises(ValueError):
        host.service.revalidate(
            host.scope(), "binding", snapshot_id=old.snapshot_id, expected_revision=True
        )
    with pytest.raises(ValueError):
        host.service.get_current(host.scope(), "\x00bad")
    with pytest.raises(ValueError):
        host.service.history(host.scope(), "binding", limit=1000)
    for field, value in [
        ("schema_version", True),
        ("source_generation", True),
        ("evidence_refs", old.evidence_refs * 9),
        ("logical_binding_id", "x" * 201),
        ("logical_binding_id", "\ud800"),
    ]:
        data = old.model_dump()
        data[field] = value
        with pytest.raises(ValueError):
            CodeBinding(**data)


def test_sqlite_schema_and_durability(host):
    host.create()
    with host.service.store.connection() as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert db.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
    with sqlite3.connect(host.path) as db:
        db.execute("PRAGMA user_version=99")
    with pytest.raises(BindingStorageError, match="^BindingStorageError$"):
        host.service.get_current(host.scope(), "binding")


def test_byte_identical_branch_switch_is_unknown_even_with_same_snapshot(host):
    old = host.create()
    subprocess.run(
        ["git", "-C", str(host.root), "symbolic-ref", "HEAD", "refs/heads/identical"],
        check=True,
    )
    assert (
        host.service.get_current(host.scope(), "binding").validation_state == "unknown"
    )
    assert host.build() == old.snapshot_id
    new = host.revalidate(old)
    assert new.reason_code == "snapshot_mismatch"
    assert new.validation_state == "unknown"


def test_unsupported_git_pointer_layout_is_unknown(host):
    old = host.create()
    shutil.rmtree(host.root / ".git")
    (host.root / ".git").write_text("gitdir: /outside/PRIVATE_SENTINEL")
    new = host.revalidate(old, snapshot=old.snapshot_id)
    assert new.validation_state == "unknown"
    assert "PRIVATE_SENTINEL" not in new.model_dump_json()


def test_unversioned_registered_root_supported(host):
    shutil.rmtree(host.root / ".git")
    old = host.create()
    assert (
        host.service.get_current(host.scope(), "binding").validation_state
        == "verified_current"
    )
    assert host.revalidate(old).validation_state == "verified_current"


def test_create_request_replay_after_new_binding_cannot_claim_current(host):
    snapshot = host.build()
    scope = host.scope("create-idempotent")
    kwargs = {
        "logical_binding_id": "binding",
        "fact_version_id": host.fact,
        "repo_id": host.repo,
        "snapshot_id": snapshot,
        "path": "main.py",
        "qualified_name": "original",
    }
    old = host.service.create_binding(scope, **kwargs)
    assert host.service.create_binding(scope, **kwargs) == old
    host.revalidate(old, snapshot=snapshot)
    replay = host.service.create_binding(scope, **kwargs)
    assert replay.binding_id == old.binding_id and replay.validation_state == "unknown"


def test_authoritative_delete_establishes_tombstone_even_with_old_delivery_revision(
    host,
):
    host.create()
    assert host.service.project_fact(
        host.scope(), fact_version_id=host.fact, event_revision=100
    )
    host.ingest("delete", host.fact)
    assert host.service.project_fact(
        host.scope(), fact_version_id=host.fact, event_revision=1
    )
    assert host.service.store.is_terminal("opaque/team", host.fact)
    assert not host.service.store.project("opaque/team", host.fact, 101, False)


def test_delete_between_computation_and_publish_is_withheld(host, monkeypatch):
    old = host.create()
    original = host.service.sources.load

    def forget(*args):
        view = original(*args)
        host.ingest("delete", host.fact)
        return view

    monkeypatch.setattr(host.service.sources, "load", forget)
    with pytest.raises(BindingUnavailable):
        host.revalidate(old, snapshot=old.snapshot_id)
    assert len(host.service.history(host.scope(), "binding")) == 1


def test_renamed_symbol_disappearing_into_changed_signature_is_plausible(host):
    old = host.create()
    (host.root / "main.py").unlink()
    host.write("moved.py", "def original(x, y):\n    return x + y\n")
    new = host.revalidate(old)
    assert new.validation_state == "unknown"
    assert new.reason_code == "plausible_candidates"


def test_no_permission_never_acquires_fact_authority(host, monkeypatch):
    old = host.create()
    scope = host.scope()
    host.auth.revoke("alice", "opaque/team")

    def must_not_call(*args, **kwargs):
        raise AssertionError("fact authority acquired")

    monkeypatch.setattr(host.facts, "get_entry", must_not_call)
    with pytest.raises(ScopeDenied, match="^Scope denied$"):
        host.service.revalidate(
            scope, "binding", snapshot_id=old.snapshot_id, expected_revision=1
        )


def test_historical_superseded_fact_metadata_does_not_become_current(host):
    old = host.create()
    host.ingest("update", host.fact)
    assert (
        host.service.get_current(host.scope(), "binding").reason_code
        == "fact_unavailable"
    )
    assert host.service.history(
        host.scope(), "binding", snapshot_id=old.snapshot_id
    ) == (old,)


def test_new_decorator_is_not_silently_verified_from_incomplete_symbol_span(host):
    old = host.create()
    host.write("main.py", "@cached\ndef original(x):\n    return x + 123\n")
    assert host.revalidate(old).validation_state == "unknown"


# These hooks observe rows actually selected by SQLite, not only values returned
# by the service. All fixtures still use J02 authority and the real J04 parser.
def observe_binding_sql(host, monkeypatch):
    import json
    from contextlib import contextmanager

    statements, payloads = [], []
    original = host.service.store.connection

    def observe_row(cursor, row):
        for column, value in zip(cursor.description, row):
            if column[0] == "payload":
                payloads.append(json.loads(value))
        return row

    @contextmanager
    def connection():
        with original() as db:
            db.set_trace_callback(statements.append)
            db.row_factory = observe_row
            yield db

    monkeypatch.setattr(host.service.store, "connection", connection)
    return statements, payloads


@pytest.mark.parametrize("grant", ["other-repo", "empty"])
def test_unauthorized_exact_id_and_absent_share_contract_before_payload_selection(
    host, monkeypatch, grant
):
    old = host.create()
    repos = (str(uuid4()),) if grant == "other-repo" else ()
    host.auth.grant("bob", "opaque/team", repos)
    scope = host.auth.context("bob", "opaque/team", "read")
    statements, payloads = observe_binding_sql(host, monkeypatch)
    for logical in (old.logical_binding_id, "absent"):
        assert host.service.get_current(scope, logical) is None
        assert host.service.history(scope, logical) == ()
        assert host.service.history(scope, logical, snapshot_id=old.snapshot_id) == ()
        with pytest.raises(BindingUnavailable, match="^BindingUnavailable$"):
            host.service.revalidate(
                scope, logical, snapshot_id=old.snapshot_id, expected_revision=1
            )
    assert payloads == []
    selects = [s for s in statements if s.startswith("SELECT") and "payload" in s]
    assert selects and all("repo IN (" in s for s in selects)
    # Positive control: exactly the same logical ID and DB yields a real row.
    assert host.service.get_current(host.scope(), old.logical_binding_id) == old
    assert payloads and {p["repo_id"] for p in payloads} == {host.repo}


def test_history_snapshot_predicate_precedes_limit_and_pagination_is_complete(
    host, monkeypatch
):
    a = "def original(x):\n    return x + 123\n"
    b = "def renamed(x):\n    return x + 123\n"
    first = host.create()
    records = [first]
    # Rebuild the alternating content snapshots explicitly. Their generations
    # increase even when the content-addressed snapshot ID repeats.
    for source in (b, a, b, a, b):
        host.write("main.py", source)
        records.append(host.revalidate(records[-1]))
    statements, payloads = observe_binding_sql(host, monkeypatch)
    target = records[1].snapshot_id
    assert host.service.history(
        host.scope(), "binding", snapshot_id=target, limit=1
    ) == (records[1],)
    for limit in (1, 2, 100):
        after, found = 0, []
        while True:
            page = host.service.history(
                host.scope(),
                "binding",
                snapshot_id=target,
                after_revision=after,
                limit=limit,
            )
            if not page:
                break
            found.extend(page)
            after = page[-1].revision
        assert found == records[1::2]
    assert payloads and {p["snapshot_id"] for p in payloads} == {target}
    selects = [s for s in statements if s.startswith("SELECT") and "payload" in s]
    assert all(s.index("snapshot=") < s.index("LIMIT") for s in selects)
    assert host.service.history(host.scope(), "binding") == tuple(records)


def test_multi_repository_snapshot_reads_only_select_authorized_payloads(
    host, tmp_path, monkeypatch
):
    repo_b = str(uuid4())
    root_b = tmp_path / "checkout-b"
    root_b.mkdir()
    (root_b / "other.py").write_text("def other(x):\n    return x * 99\n")
    host.registry.register(repo_b, str(root_b.resolve()), "opaque/team")
    host.auth.grant("alice", "opaque/team", (host.repo, repo_b))
    first = host.create()
    host.write("main.py", "def renamed(x):\n    return x + 123\n")
    second = host.revalidate(first)
    page_b = host.context.build(host.scope(), repo_b, deadline_ms=10000)
    other = host.service.create_binding(
        host.scope(),
        logical_binding_id="other-binding",
        fact_version_id=host.fact,
        repo_id=repo_b,
        snapshot_id=page_b.snapshot_id,
        path="other.py",
        qualified_name="other",
    )
    host.auth.grant("bob", "opaque/team", (host.repo,))
    bob = host.auth.context("bob", "opaque/team", "history-b")
    statements, payloads = observe_binding_sql(host, monkeypatch)
    for logical in ("other-binding", "absent"):
        assert host.service.get_current(bob, logical) is None
        for snapshot in (first.snapshot_id, second.snapshot_id, other.snapshot_id):
            assert host.service.history(bob, logical, snapshot_id=snapshot) == ()
    assert payloads == []
    assert host.service.history(
        bob, "binding", snapshot_id=second.snapshot_id, limit=1
    ) == (second,)
    assert {p["repo_id"] for p in payloads} == {host.repo}
    assert {p["snapshot_id"] for p in payloads} == {second.snapshot_id}
    # A principal with both grants can inspect either repository's history.
    assert host.service.history(host.scope(), "other-binding") == (other,)
    assert any(p["repo_id"] == repo_b for p in payloads)
    assert all(
        "repo IN (" in s
        for s in statements
        if s.startswith("SELECT") and "payload" in s
    )

    # Exercise stored history with different repository identities as can occur
    # in host-imported metadata. Both source records above came from real J04
    # snapshots; only this trusted storage fixture retargets the logical chain.
    data = other.model_dump()
    data.update(
        logical_binding_id="binding",
        revision=3,
        supersedes_binding_id=second.binding_id,
    )
    imported = host.service._record(data)
    host.service.store.append(
        imported,
        2,
        host.service.store.source_revision("opaque/team", repo_b),
        "host-import",
        "import-digest",
    )
    payloads.clear()
    assert host.service.get_current(bob, "binding") is None
    assert host.service.history(bob, "binding") == (first, second)
    assert host.service.history(bob, "binding", snapshot_id=other.snapshot_id) == ()
    assert {p["repo_id"] for p in payloads} == {host.repo}
    host.auth.grant("carol", "opaque/team", (repo_b,))
    carol = host.auth.context("carol", "opaque/team", "read-b")
    payloads.clear()
    assert host.service.history(carol, "binding", snapshot_id=first.snapshot_id) == ()
    assert host.service.history(carol, "binding", limit=1) == (imported,)
    assert {p["repo_id"] for p in payloads} == {repo_b}
    assert host.service.history(host.scope(), "binding") == (first, second, imported)


def test_old_snapshot_publication_rejected_without_moving_current_or_observation(host):
    from jitmind.code_memory import BindingStaleSource

    first = host.create()
    host.write("main.py", "def renamed(x):\n    return x + 123\n")
    second = host.revalidate(first)
    assert (first.source_generation, second.source_generation) == (1, 2)
    with host.service.store.connection() as db:
        before = db.execute("SELECT * FROM sources").fetchall()
        requests = db.execute("SELECT COUNT(*) FROM requests").fetchone()
    with pytest.raises(BindingStaleSource, match="^BindingStaleSource$"):
        host.revalidate(second, snapshot=first.snapshot_id)
    with host.service.store.connection() as db:
        assert db.execute("SELECT * FROM sources").fetchall() == before
        assert db.execute("SELECT COUNT(*) FROM requests").fetchone() == requests
    assert host.service.get_current(host.scope(), "binding") == second
    assert host.service.history(host.scope(), "binding") == (first, second)
    assert host.service.history(
        host.scope(), "binding", snapshot_id=first.snapshot_id
    ) == (first,)
    # The old at-check verified record survives only as history. Moving disk back
    # without rebuilding cannot resurrect that verification in current retrieval.
    host.write("main.py", "def original(x):\n    return x + 123\n")
    assert (
        host.service.get_current(host.scope(), "binding").validation_state == "unknown"
    )
    # Even if old snapshot A becomes readable/fresh again, its generation 1
    # observation cannot replace the newer projection with verified_current.
    with pytest.raises(BindingStaleSource):
        host.revalidate(second, snapshot=first.snapshot_id)
    # A newly built observation of old content has a NEW generation and is valid.
    third = host.revalidate(second)
    assert third.snapshot_id == first.snapshot_id and third.source_generation == 3
    assert third.validation_state == "verified_current"
    with host.service.store.connection() as db:
        assert db.execute("SELECT revision,generation FROM sources").fetchone() == (
            3,
            3,
        )
    host.ingest("delete", host.fact)
    assert host.service.project_fact(
        host.scope(), fact_version_id=host.fact, event_revision=10
    )
    assert not host.service.store.project("opaque/team", host.fact, 11, False)
    with pytest.raises(BindingUnavailable):
        host.revalidate(third, snapshot=first.snapshot_id)
    assert (
        host.service.get_current(host.scope(), "binding").reason_code
        == "fact_unavailable"
    )


def test_restart_and_other_principal_cannot_publish_below_durable_source_watermark(
    host,
):
    from jitmind.code_memory import BindingStaleSource

    first = host.create()
    second = host.revalidate(first)  # same content, increasing generation
    assert second.source_generation == 2
    host.auth.grant("bob", "opaque/team", (host.repo,))
    bob = host.auth.context("bob", "opaque/team", "missing-source")
    with pytest.raises(BindingStaleSource):
        host.revalidate(second, snapshot=first.snapshot_id, scope=bob)
    # Simulate a fresh J04 process counter, while retaining the real durable DB.
    context = CodeContext(host.auth, host.registry, host.context.parser)
    restarted = CodeMemoryService(host.auth, host.facts, context, host.path)
    snapshot = context.build(host.scope(), host.repo, deadline_ms=10000)
    assert snapshot.generation == 1
    with pytest.raises(BindingStaleSource):
        restarted.revalidate(
            host.scope(),
            "binding",
            snapshot_id=snapshot.snapshot_id,
            expected_revision=2,
        )
    assert restarted.history(host.scope(), "binding") == (first, second)


def test_replay_at_another_snapshot_conflicts_without_selecting_old_payload(
    host, monkeypatch
):
    first = host.create()
    replay_scope = host.scope("replay")
    second = host.revalidate(first, snapshot=first.snapshot_id, scope=replay_scope)
    host.write("main.py", "def renamed(x):\n    return x + 123\n")
    snapshot = host.build()
    statements, payloads = observe_binding_sql(host, monkeypatch)
    with pytest.raises(BindingConflict):
        host.revalidate(second, snapshot=snapshot, scope=replay_scope)
    # Only the authorized current-head read occurs; the replay never hydrates a
    # historical payload outside the requested snapshot.
    replay_queries = [
        s for s in statements if s.startswith("SELECT r.digest,h.payload")
    ]
    assert len(replay_queries) == 1 and f"h.snapshot='{snapshot}'" in replay_queries[0]
    assert [p["binding_id"] for p in payloads] == [second.binding_id]


def test_revocation_and_foreign_issuer_checked_before_or_after_scoped_sql(
    host, monkeypatch
):
    from contextlib import contextmanager

    old = host.create()
    other_auth = ScopeAuthority()
    other_auth.grant("alice", "opaque/team", (host.repo,))
    foreign = other_auth.context("alice", "opaque/team", "foreign")
    statements, payloads = observe_binding_sql(host, monkeypatch)
    with pytest.raises(ScopeDenied):
        host.service.history(foreign, "binding")
    assert statements == [] and payloads == []
    scope = host.scope()
    original = host.service.store.connection

    @contextmanager
    def revoke_after_read():
        with original() as db:
            yield db
        host.auth.revoke("alice", "opaque/team")

    monkeypatch.setattr(host.service.store, "connection", revoke_after_read)
    with pytest.raises(ScopeDenied):
        host.service.history(scope, "binding", snapshot_id=old.snapshot_id)


def test_v1_migration_preserves_history_receipts_tombstones_and_recovers_watermark(
    host,
):
    from jitmind.code_memory import BindingStaleSource

    first = host.create()
    host.write("main.py", "def renamed(x):\n    return x + 123\n")
    second = host.revalidate(first)
    # Build the exact v1 layout, including the formerly possible 1 -> 2 -> 1
    # unknown observation, without depending on stale J06 implementation files.
    data = second.model_dump()
    data.update(
        revision=3,
        snapshot_id=first.snapshot_id,
        source_generation=1,
        supersedes_binding_id=second.binding_id,
        validation_state="unknown",
        reason_code="source_unknown",
    )
    legacy = host.service._record(data)
    host.service.store.project("opaque/team", "terminal-fact", 8, True)
    with sqlite3.connect(host.path) as db:
        db.execute("DROP INDEX history_scope")
        db.execute("ALTER TABLE history DROP COLUMN repo")
        db.execute("ALTER TABLE history DROP COLUMN snapshot")
        db.execute(
            "INSERT INTO history VALUES (?,?,?,?)",
            (
                "opaque/team",
                "binding",
                3,
                legacy.model_dump_json(),
            ),
        )
        db.execute("UPDATE heads SET revision=3")
        db.execute(
            "UPDATE sources SET revision=3,snapshot=?,generation=1",
            (first.snapshot_id,),
        )
        db.execute("PRAGMA user_version=1")
        payloads_before = db.execute(
            "SELECT payload FROM history ORDER BY revision"
        ).fetchall()
        receipts_before = db.execute("SELECT * FROM requests").fetchall()
    restarted = CodeMemoryService(host.auth, host.facts, host.context, host.path)
    assert restarted.history(host.scope(), "binding") == (first, second, legacy)
    with restarted.store.connection() as db:
        assert db.execute("PRAGMA user_version").fetchone() == (2,)
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert (
            db.execute("SELECT payload FROM history ORDER BY revision").fetchall()
            == payloads_before
        )
        assert db.execute("SELECT * FROM requests").fetchall() == receipts_before
        assert db.execute(
            "SELECT revision,generation,snapshot FROM sources"
        ).fetchone() == (3, 2, second.snapshot_id)
    with pytest.raises(BindingStaleSource):
        restarted.revalidate(
            host.scope(), "binding", snapshot_id=first.snapshot_id, expected_revision=3
        )
    assert restarted.store.is_terminal("opaque/team", "terminal-fact")
    assert not restarted.store.project("opaque/team", "terminal-fact", 99, False)


def test_repository_watermark_cannot_be_bypassed_by_another_binding(host):
    from jitmind.code_memory import BindingStaleSource

    first = host.create()
    # Another binding advances the shared repository watermark, even though the
    # first logical binding's current revision remains at generation 1.
    second = host.create(logical="another-binding")
    assert second.source_generation == 2
    # An unavailable snapshot is represented by the adapter's conservative
    # unknown generation 1; it must not lower the recorded observation.
    with pytest.raises(BindingStaleSource):
        host.revalidate(first, snapshot="0" * 64)
    assert host.service.history(host.scope(), "binding") == (first,)
    with host.service.store.connection() as db:
        assert db.execute("SELECT revision,generation FROM sources").fetchone() == (
            2,
            2,
        )


def test_equal_generation_cannot_replace_another_snapshot_with_unknown(host):
    from jitmind.code_memory import BindingStaleSource

    first = host.create()
    with pytest.raises(BindingStaleSource):
        host.revalidate(first, snapshot="0" * 64)
    assert host.service.get_current(host.scope(), "binding") == first
    assert host.service.history(host.scope(), "binding") == (first,)

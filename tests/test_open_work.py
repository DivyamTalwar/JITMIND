from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import UUID

import pytest

from jitmind.code_memory.open_work import OpenWorkService
from jitmind.code_memory.work_storage import (
    Budget,
    Conflict,
    Deferred,
    WorkDatabase,
    WorkError,
)
from jitmind.scope import ScopeAuthority, ScopeContext, ScopeDenied


@pytest.fixture
def work(tmp_path):
    authority = ScopeAuthority()
    for actor in ("owner", "peer", "admin"):
        authority.grant(actor, "tenant", ["host/a/repo", "host/b/repo"])
    scope = authority.context("owner", "tenant", "request-not-identity")
    database = WorkDatabase(tmp_path / "work.sqlite")
    service = OpenWorkService(
        database,
        authority,
        is_admin=lambda s, repo: s.principal_id == "admin",
        verify_evidence=lambda s, repo, refs, reason: (
            refs == ("test-run:verified",) and reason != "done"
        ),
    )
    return service, authority, scope


def create(service, scope, key="create", **kwargs):
    return service.create(
        scope,
        "host/a/repo",
        summary="Review the deletion migration",
        expected_revision=0,
        idempotency_key=key,
        **kwargs,
    )


def resolve(service, scope, record, key="resolve", **kwargs):
    return service.transition(
        scope,
        record.repo,
        record.id,
        status="resolved",
        expected_revision=record.revision,
        idempotency_key=key,
        reason="Confirmed by configured test record",
        evidence_refs=("test-run:verified",),
        **kwargs,
    )


def test_durable_cas_replay_owner_history_and_explicit_reopen(work):
    service, authority, scope = work
    first = create(service, scope, due_at="2000-01-01T00:00:00Z")
    assert (
        UUID(first.id)
        and first.creator == first.owner == "owner"
        and first.revision == 1
    )
    assert create(service, scope, due_at="2000-01-01T00:00:00Z") == first
    with pytest.raises(Conflict):
        create(service, scope, owner="owner", binding_refs=("different",))
    peer = authority.context("peer", "tenant", scope.request_id)
    with pytest.raises(ScopeDenied):
        resolve(service, peer, first)
    with pytest.raises(ScopeDenied):
        create(service, peer, key="assign", owner="owner")
    done = resolve(service, scope, first)
    restarted = OpenWorkService(WorkDatabase(service.database.path), authority)
    assert (
        resolve(restarted, scope, first) == done
    )  # no verifier called on identical retry
    with pytest.raises(Conflict):
        resolve(service, scope, first, key="competing")
    with pytest.raises(WorkError, match="explicit_reopen_required"):
        service.transition(
            scope,
            first.repo,
            first.id,
            status="open",
            expected_revision=2,
            idempotency_key="reopen",
            reason="More work",
        )
    reopened = service.transition(
        scope,
        first.repo,
        first.id,
        status="open",
        expected_revision=2,
        idempotency_key="reopen",
        reason="New evidence needs review",
        reopen=True,
    )
    assert reopened.revision == 3 and reopened.status == "open"
    admin = authority.context("admin", "tenant", "admin-request")
    cancelled = service.transition(
        admin,
        first.repo,
        first.id,
        status="cancelled",
        expected_revision=3,
        idempotency_key="cancel",
        reason="Owner requested cancellation",
    )
    assert cancelled.revision == 4
    with pytest.raises(WorkError, match="invalid_transition"):
        service.transition(
            scope,
            first.repo,
            first.id,
            status="open",
            expected_revision=4,
            idempotency_key="uncancel",
            reason="Undo",
            reopen=True,
        )
    events = service.history(scope, first.repo, first.id)
    assert [e["revision"] for e in events] == [1, 2, 3, 4]
    assert events[-1]["actor"] == "admin"
    with pytest.raises(WorkError), service.database.connect(Budget(1)) as db:
        db.execute("DELETE FROM history")


def test_resolution_requires_configured_evidence_not_done_strings(work, tmp_path):
    service, _, scope = work
    item = create(service, scope)
    for reason, refs in [
        ("done", ()),
        ("done", ("test-run:verified",)),
        ("$(touch /tmp/J06-SHOULD-NOT-EXIST)", ("API_KEY=negative-sentinel",)),
    ]:
        with pytest.raises(WorkError, match="evidence_required"):
            service.transition(
                scope,
                item.repo,
                item.id,
                status="resolved",
                expected_revision=1,
                idempotency_key="attempt",
                reason=reason,
                evidence_refs=refs,
            )
    assert service.get(scope, item.repo, item.id).status == "open"
    assert not (tmp_path / "J06-SHOULD-NOT-EXIST").exists()


def test_competing_resolution_accepts_one(work):
    service, _, scope = work
    item = create(service, scope)
    barrier = Barrier(2)

    def compete(key):
        barrier.wait()
        try:
            return resolve(service, scope, item, key).status
        except (Conflict, Deferred):
            return "rejected"

    with ThreadPoolExecutor(2) as executor:
        outcomes = list(executor.map(compete, ["one", "two"]))
    assert sorted(outcomes) == ["rejected", "resolved"]
    assert len(service.history(scope, item.repo, item.id)) == 2


def test_keyset_pagination_does_not_skip_after_transition(work):
    service, _, scope = work
    items = [create(service, scope, key=str(i)) for i in range(4)]
    page = service.list(scope, items[0].repo, limit=2)
    resolve(service, scope, items[0])
    next_page = service.list(scope, items[0].repo, limit=2, after=page.next_cursor)
    assert [x.id for x in page.items + next_page.items] == [x.id for x in items]
    assert next_page.next_cursor is None
    assert service.list(scope, "host/b/repo").items == ()


def test_forged_scope_and_revocation_before_disk(work, monkeypatch):
    service, authority, scope = work
    forged = ScopeContext(
        scope.principal_id,
        scope.namespace_id,
        scope.authorized_repo_ids,
        scope.authorization_version,
        scope.request_id,
    )
    monkeypatch.setattr(
        service.database, "connect", lambda *a, **kw: pytest.fail("disk access")
    )
    with pytest.raises(ScopeDenied):
        create(service, forged)
    authority.revoke("owner", "tenant")
    with pytest.raises(ScopeDenied):
        create(service, scope)


@pytest.mark.parametrize("bad", [True, -1, 2**63, 1.0])
def test_revision_validation(work, bad):
    service, _, scope = work
    with pytest.raises(WorkError):
        service.create(
            scope,
            "host/a/repo",
            summary="x",
            expected_revision=bad,
            idempotency_key="k",
        )

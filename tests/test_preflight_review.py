"""J56 regressions using real J02 authority; synthetic local data only."""

import json
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime

import pytest
from test_preflight import REPO, add, write_fact
from test_preflight import setup as setup_fixture

from jitmind.code_memory.lesson_models import ProposedAction
from jitmind.code_memory.open_work import OpenWorkService
from jitmind.code_memory.work_storage import Budget, Conflict, WorkError
from jitmind.scope import ScopeDenied
from jitmind.storage import IngestRequest, Proposal

setup = setup_fixture


def stamp(value):
    return datetime.fromisoformat(value).timestamp()


def temporal_fact(facts, key, *, start=None, end=None, meta=None):
    return facts.ingest(
        IngestRequest.create("tenant", key, "backing evidence", meta=meta),
        lambda _: Proposal(
            abstract="backing evidence",
            header="header",
            decorated="backing evidence",
            decision={"operation": "add", "t_valid": start, "t_invalid": end},
        ),
    )


def author(projection, scope, target, fact, key="policy"):
    return projection.author_instruction(
        scope,
        target,
        lesson_id=key,
        fact_id=fact.memory_id,
        version=1,
        source_revision=fact.revision,
        body="Preserve this repository rule.",
    )


@pytest.mark.parametrize(
    "start,end,active",
    [
        ("2000-01-01T00:00:00Z", "2999-01-01T00:00:00Z", True),
        ("2999-01-01T00:00:00Z", None, False),
        ("2000-01-01T00:00:00Z", "2001-01-01T00:00:00Z", False),
        ("2026-01-01T00:00:00Z", None, True),
        (None, "2026-01-01T00:00:00Z", False),
        ("2026-01-01T05:30:00+05:30", "2026-01-01T00:00:01Z", True),
        (None, "2025-12-31T19:00:00-05:00", False),
        ("2026-01-01T05:30:01+05:30", None, False),
        ("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", False),
    ],
)
def test_primary_temporal_interval_applies_to_authored_policy(
    setup, start, end, active
):
    _, scope, facts, _, projection, service, session, target = setup
    projection.facts.clock = lambda: stamp("2026-01-01T00:00:00+00:00")
    fact = temporal_fact(facts, "time", start=start, end=end)
    assert author(projection, scope, target, fact) is active
    result = service.deliver(scope, session, target)
    assert result.state == ("delivered" if active else "nothing_relevant")
    assert bool(result.payload) is active
    # Administrative existence alone is intentionally still true in J02.
    assert facts.get_entry("tenant", fact.memory_id) is not None


@pytest.mark.parametrize(
    "meta,active",
    [
        ({"ttl_seconds": 10}, True),
        ({"ttl_seconds": 0}, False),
        ({"expires_at": "2026-01-01T00:00:00Z"}, False),
        ({"expires_at": "2026-01-01T05:30:01+05:30"}, True),
        ({"t_expired": "2026-01-01T00:00:00Z"}, False),
    ],
)
def test_backing_ttl_exact_boundaries(setup, meta, active):
    _, scope, facts, _, projection, service, session, target = setup
    # This oracle checks temporal eligibility, not shared-runner fsync speed.
    # Dedicated contention/deadline tests retain the 150 ms production budget.
    service.policy = replace(service.policy, deadline_seconds=3.0)
    facts.clock = lambda: "2026-01-01T00:00:00+00:00"
    projection.facts.clock = lambda: stamp(facts.clock())
    fact = temporal_fact(facts, "ttl", meta=meta)
    assert author(projection, scope, target, fact) is active
    assert bool(service.deliver(scope, session, target).payload) is active


@pytest.mark.parametrize(
    "meta",
    [
        {"ttl_seconds": True},
        {"ttl_seconds": -1},
        {"ttl_seconds": "10"},
        {"ttl_seconds": None},
        {"expires_at": "2001-01-01"},
        {"expires_at": "2026-01-01T00:00:00"},
        {"expires_at": "2026-01-01T00:00:00+01:99"},
        {"expires_at": "garbage"},
    ],
)
def test_malformed_primary_temporal_metadata_is_explicitly_ineligible(setup, meta):
    _, scope, facts, _, projection, _, _, target = setup
    fact = temporal_fact(facts, "malformed", meta=meta)
    with pytest.raises(WorkError, match="invalid_temporal_metadata"):
        author(projection, scope, target, fact)


def mutate_page(facts, page_id, values):
    # Simulate independently changed primary metadata, never change J06's copy.
    import sqlite3

    with sqlite3.connect(facts.path) as db:
        raw = db.execute(
            "SELECT payload FROM pages WHERE page_id=?", (page_id,)
        ).fetchone()[0]
        data = json.loads(raw)
        data["meta"].update(values)
        db.execute(
            "UPDATE pages SET payload=? WHERE page_id=?", (json.dumps(data), page_id)
        )


@pytest.mark.parametrize(
    "metadata,reason",
    [
        ({"expires_at": "2001-01-01T00:00:00Z"}, "receipt_no_longer_visible"),
        ({"t_valid": "2999-01-01T00:00:00Z"}, "receipt_no_longer_visible"),
        ({"expires_at": "not-a-time"}, "invalid_temporal_metadata"),
        ({"ttl_seconds": True}, "invalid_temporal_metadata"),
    ],
)
def test_page_authority_rechecked_before_candidate_and_saved_receipt(
    setup, metadata, reason
):
    _, scope, facts, _, projection, service, session, target = setup
    fact = temporal_fact(facts, "page")
    assert author(projection, scope, target, fact)
    assert (
        service.deliver(scope, session, target, delivery_key="saved").state
        == "delivered"
    )
    mutate_page(facts, fact.page_id, metadata)
    result = service.deliver(scope, session, target, delivery_key="saved")
    assert result.payload == "" and result.reason == reason
    fresh = service.start_session(scope, REPO)
    assert service.deliver(scope, fresh, target).payload == ""


def test_expiry_between_commit_and_response_and_replay(setup, monkeypatch):
    _, scope, facts, database, projection, service, session, target = setup
    now = [stamp("2026-01-01T00:00:00+00:00")]
    projection.facts.clock = lambda: now[0]
    fact = temporal_fact(facts, "edge", end="2026-01-01T00:00:01Z")
    assert author(projection, scope, target, fact)
    transaction = database.transaction

    @contextmanager
    def expire(db):
        with transaction(db):
            yield
        now[0] += 1

    monkeypatch.setattr(database, "transaction", expire)
    result = service.deliver(scope, session, target, delivery_key="saved")
    assert result.payload == "" and result.reason == "authority_changed"
    assert service.deliver(scope, session, target, delivery_key="saved").payload == ""
    # Expiry is not an accepted primary forget.
    with pytest.raises(WorkError, match="primary_forget_required"):
        projection.forget(scope, fact.memory_id, tombstone_revision=2)


@pytest.mark.parametrize("prefix", ["low", "expired", "unrelated", "lexical"])
def test_twenty_plus_rows_cannot_starve_repository_instruction(setup, prefix):
    _, scope, facts, _, projection, service, session, target = setup
    fact = write_fact(facts, "backing")
    for n in range(25):
        candidate_target = target
        if prefix == "unrelated":
            candidate_target = replace(target, symbol="unrelated")
        if prefix == "lexical":
            candidate_target = replace(target, symbol="pro")
        projection.import_observation(
            scope,
            candidate_target,
            lesson_id=f"a-{n:02}",
            fact_id=fact.memory_id,
            version=1,
            source_revision=fact.revision,
            body="observation",
            source_verified=True,
            confidence=0.1 if prefix == "low" else 0.95,
            valid_until=time.time() - 1 if prefix == "expired" else None,
        )
    assert author(projection, scope, ProposedAction(REPO, "*"), fact, "z-policy")
    from jitmind.code_memory.preflight import PreflightPolicy

    service.policy = PreflightPolicy()  # preserve the candidate-selection timing oracle
    started = time.monotonic()
    result = service.deliver(scope, session, target)
    elapsed = time.monotonic() - started
    assert result.state == "delivered" and "z-policy" in result.lesson_ids
    assert result.candidate_count == 1 and result.selection_complete
    assert elapsed < 0.3  # local regression ceiling, not a production guarantee
    print(
        f"candidate prefix={prefix} rows=26 examined={result.candidate_count} seconds={elapsed:.6f}"
    )


def test_instruction_priority_and_lexical_parent_scope(setup):
    _, scope, facts, _, projection, service, session, target = setup
    fact = write_fact(facts, "backing")
    for n in range(21):
        projection.import_observation(
            scope,
            target,
            lesson_id=f"a-{n}",
            fact_id=fact.memory_id,
            version=1,
            source_revision=1,
            body="specific observation",
            source_verified=True,
            confidence=1,
        )
    assert author(projection, scope, ProposedAction(REPO, "*"), fact, "z-policy")
    result = service.deliver(scope, session, target)
    assert result.lesson_ids[0] == "z-policy"
    assert result.candidate_count == 20 and not result.selection_complete
    other_session = service.start_session(scope, REPO)
    assert author(projection, scope, replace(target, symbol="Outer"), fact, "parent")
    nested = service.deliver(
        scope, other_session, replace(target, symbol="Outer.inner")
    )
    assert "parent" in nested.lesson_ids
    unrelated = service.deliver(
        scope, service.start_session(scope, REPO), replace(target, symbol="Outerish")
    )
    assert "parent" not in unrelated.lesson_ids


@pytest.mark.parametrize("blocked", ["forgotten", "delivered"])
def test_unexamined_primary_candidates_disclose_incomplete_selection(setup, blocked):
    _, scope, facts, _, _, service, session, target = setup
    for n in range(21):
        fact = add(setup, f"a-{n:02}")
        if blocked == "forgotten" and n < 20:
            write_fact(facts, f"delete-{n}", operation="delete", target=fact.memory_id)
    if blocked == "delivered":
        for _ in range(10):
            assert service.deliver(scope, session, target).state == "delivered"
    result = service.deliver(scope, session, target)
    assert result.state == "deferred" and result.reason == "selection_incomplete"
    assert result.payload == "" and result.candidate_count == 20
    assert result.selection_complete is False


def test_empty_rank_never_restores_candidates_when_window_incomplete(
    setup, monkeypatch
):
    _, scope, _, _, _, service, session, target = setup
    for n in range(21):
        add(setup, f"a-{n}")
    monkeypatch.setattr(service, "_rank", lambda *args: [])
    result = service.deliver(scope, session, target)
    assert result.payload == "" and result.state == "deferred"
    assert result.reason == "selection_incomplete"


def test_durable_policy_author_departure_and_explicit_repo_owner_retirement(setup):
    authority, scope, facts, database, projection, service, _, target = setup
    fact = write_fact(facts, "policy")
    assert author(projection, scope, target, fact)
    work = OpenWorkService(database, authority)
    obligation = work.create(
        scope,
        REPO,
        summary="Review remains open",
        expected_revision=0,
        idempotency_key="open",
    )
    projection.can_author = lambda *args: False
    with pytest.raises(ScopeDenied):
        author(projection, scope, target, fact, "future")
    authority.revoke("alice", "tenant")
    bob = authority.context("bob", "tenant", "read")
    session = service.start_session(bob, REPO)
    assert (
        service.deliver(bob, session, target, delivery_key="saved").state == "delivered"
    )
    assert work.get(bob, REPO, obligation.id).status == "open"
    with pytest.raises(ScopeDenied):
        projection.retire_instruction(bob, REPO, "policy", expected_version=1)
    projection.can_retire = lambda s, repo: s.principal_id == "bob" and repo == REPO
    projection.retire_instruction(bob, REPO, "policy", expected_version=1)
    assert service.deliver(bob, session, target, delivery_key="saved").payload == ""
    assert service.deliver(bob, service.start_session(bob, REPO), target).payload == ""
    assert work.get(bob, REPO, obligation.id).status == "open"
    projection.can_author = lambda *args: True
    with pytest.raises(Conflict):
        author(projection, bob, target, fact)


def test_primary_instruction_forget_suppresses_saved_receipt(setup):
    _, scope, facts, _, projection, service, session, target = setup
    fact = write_fact(facts, "policy")
    assert author(projection, scope, target, fact)
    assert (
        service.deliver(scope, session, target, delivery_key="saved").state
        == "delivered"
    )
    write_fact(facts, "forget-policy", operation="delete", target=fact.memory_id)
    assert service.deliver(scope, session, target, delivery_key="saved").payload == ""
    assert not author(projection, scope, target, fact, "old-event")


def test_ttl_expiry_suppresses_previously_delivered_policy(setup):
    _, scope, facts, _, projection, service, session, target = setup
    facts.clock = lambda: "2026-01-01T00:00:00+00:00"
    now = [stamp(facts.clock())]
    projection.facts.clock = lambda: now[0]
    fact = temporal_fact(facts, "ttl-replay", meta={"ttl_seconds": 1})
    assert author(projection, scope, target, fact)
    assert (
        service.deliver(scope, session, target, delivery_key="saved").state
        == "delivered"
    )
    now[0] += 1
    assert service.deliver(scope, session, target, delivery_key="saved").payload == ""
    assert projection.candidates(scope, target, Budget(1)) == []


def test_retirement_marker_survives_primary_forget_cleanup(setup):
    _, scope, facts, _, projection, service, session, target = setup
    fact = write_fact(facts, "policy")
    assert author(projection, scope, target, fact)
    projection.can_retire = lambda *args: True
    projection.retire_instruction(scope, REPO, "policy", expected_version=1)
    deleted = write_fact(facts, "forget", operation="delete", target=fact.memory_id)
    projection.forget(scope, fact.memory_id, tombstone_revision=deleted.revision)
    new_fact = write_fact(facts, "new-evidence")
    with pytest.raises(Conflict):
        author(projection, scope, target, new_fact)
    assert service.deliver(scope, session, target).payload == ""


def test_legacy_unbound_payload_remains_readable_and_idempotent(setup):
    import sqlite3

    _, scope, facts, database, projection, service, session, target = setup
    fact = write_fact(facts, "old")
    assert author(projection, scope, target, fact)
    with sqlite3.connect(database.path) as db:
        raw = json.loads(db.execute("SELECT payload FROM lessons").fetchone()[0])
        for key in ("binding_id", "binding_revision", "retired"):
            raw.pop(key)
        db.execute("UPDATE lessons SET payload=?", (json.dumps(raw),))
    assert author(projection, scope, target, fact)
    assert service.deliver(scope, session, target).state == "delivered"

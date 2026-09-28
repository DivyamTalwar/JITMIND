from datetime import datetime, timezone

import pytest

from jitmind.schemas import AdvancedMemoryStore, MemoryEntry
from jitmind.scope import ScopeAuthority, ScopeDenied
from jitmind.scoped_temporal import (
    InMemoryTemporalHistory,
    StrictTemporalOracle,
    TemporalConflict,
    TemporalFact,
    TemporalPerspective,
    utc,
)


def ts(month, day=1):
    return datetime(2026, month, day, tzinfo=timezone.utc)


def setup():
    authority = ScopeAuthority()
    authority.grant("p", "n", ["repo"])
    scope = authority.context("p", "n", "r")
    backend = InMemoryTemporalHistory()
    return authority, scope, backend, StrictTemporalOracle(authority, backend)


def test_seven_query_reviewed_retry_limit_oracle():
    _, scope, backend, oracle = setup()
    a = TemporalFact("a", "retry_limit", "3", ts(1))
    b = TemporalFact("b", "retry_limit", "3", ts(1), ts(2))
    c = TemporalFact("c", "retry_limit", "5", ts(2))
    d = TemporalFact("d", "retry_limit", "3", ts(1), ts(1, 15))
    e = TemporalFact("e", "retry_limit", "4", ts(1, 15), ts(2))
    oracle.publish(scope, TemporalPerspective(ts(1), (a,)))
    oracle.publish(scope, TemporalPerspective(ts(2), (b, c)))
    oracle.publish(scope, TemporalPerspective(ts(3), (d, e, c)))
    cases = [
        (ts(1, 20), ts(1, 20), "3"),
        (ts(1, 20), ts(2, 15), "3"),
        (ts(1, 20), ts(3), "4"),
        (ts(1, 10), ts(3, 2), "3"),
        (ts(1, 15), ts(3, 2), "4"),
        (ts(2), ts(3, 2), "5"),
        (ts(2, 10), ts(1, 20), "3"),
    ]
    for valid, transaction, expected in cases:
        assert [
            fact.content
            for fact in oracle.query_as_of(scope, valid, transaction_at=transaction)
        ] == [expected]
    assert [fact.content for fact in oracle.query_as_of(scope, ts(1, 20))] == ["4"]
    assert oracle.query_as_of(scope, "2026-01-20T05:30:00+05:30") == (e,)
    assert backend.read_perspectives("n")[0].facts[0].valid_to is None


def test_ttl_eligibility_is_distinct_from_history_retention():
    _, scope, backend, oracle = setup()
    fact = TemporalFact("old", "office", "Delhi", ts(1), expires_at=ts(2))
    oracle.publish(scope, TemporalPerspective(ts(1), (fact,)))
    assert oracle.query_as_of(scope, ts(3)) == (fact,)
    assert oracle.query_as_of(scope, ts(3), eligible_at=ts(2)) == ()
    assert oracle.query_as_of(scope, ts(3), eligible_at=ts(1, 31)) == (fact,)
    assert len(backend.read_perspectives("n")) == 1


@pytest.mark.parametrize(
    "value",
    [
        "bad",
        "2026-01-01",
        ts(1).replace(tzinfo=None),
        True,
        None,
        1,
        "2026-13-01T00:00:00Z",
    ],
)
def test_strict_timestamps_reject_invalid_or_naive(value):
    with pytest.raises(TemporalConflict):
        utc(value)


def test_utc_offsets_and_exact_half_open_boundaries():
    _, scope, _, oracle = setup()
    fact = TemporalFact(
        "x", "key", "value", "2026-01-01T01:00:00+01:00", "2026-02-01T00:00:00Z"
    )
    oracle.publish(scope, TemporalPerspective(ts(1), (fact,)))
    assert fact.valid_from == ts(1)
    assert oracle.query_as_of(scope, ts(1)) == (fact,)
    assert oracle.query_as_of(scope, ts(2)) == ()
    for end in (ts(1), ts(1).replace(year=2025)):
        with pytest.raises(TemporalConflict):
            TemporalFact("x", "key", "bad", ts(1), end)


def test_declared_single_valued_overlap_fails_and_multiple_values_opt_in():
    left = TemporalFact("a", "key", "one", ts(1), ts(3))
    right = TemporalFact("b", "key", "two", ts(2))
    with pytest.raises(TemporalConflict):
        TemporalPerspective(ts(1), (left, right))
    with pytest.raises(TemporalConflict):
        TemporalPerspective(
            ts(1), (left, TemporalFact("b", "key", "two", ts(2), single_valued=False))
        )
    TemporalPerspective(
        ts(1),
        (
            TemporalFact("a", "key", "one", ts(1), ts(2)),
            right,
        ),
    )
    TemporalPerspective(
        ts(1),
        (
            TemporalFact("a", "key", "one", ts(1), single_valued=False),
            TemporalFact("b", "key", "two", ts(1), single_valued=False),
        ),
    )


def test_transactions_must_append_and_revision_is_not_bool():
    _, scope, backend, oracle = setup()
    perspective = TemporalPerspective(ts(1), ())
    oracle.publish(scope, perspective)
    with pytest.raises(TemporalConflict):
        oracle.publish(scope, perspective)
    with pytest.raises(TemporalConflict):
        backend.append_perspective("n", True, TemporalPerspective(ts(2), ()))
    with pytest.raises(TemporalConflict):
        backend.append_perspective("n", 0, TemporalPerspective(ts(2), ()))


def test_authorization_and_explicit_snapshot_selection():
    authority, scope, _, oracle = setup()
    main = TemporalFact("a", "key", "main", ts(1), repo_id="repo", snapshot_id="main")
    other = TemporalFact(
        "b", "key", "other", ts(1), repo_id="repo", snapshot_id="other"
    )
    oracle.publish(scope, TemporalPerspective(ts(1), (main, other)))
    assert not oracle.query_as_of(scope, ts(2))
    assert oracle.query_as_of(scope, ts(2), snapshots=(("repo", "main"),)) == (main,)
    assert oracle.query_as_of(scope, ts(2), snapshots=(("repo", "other"),)) == (other,)
    authority.revoke("p", "n")
    with pytest.raises(ScopeDenied):
        oracle.query_as_of(scope, ts(2))


def test_query_revocation_during_backend_read_drops_response():
    authority, scope, backend, oracle = setup()

    class Revoking:
        def read_perspectives(self, namespace):
            authority.revoke("p", "n")
            return backend.read_perspectives(namespace)

    oracle.backend = Revoking()
    with pytest.raises(ScopeDenied):
        oracle.query_as_of(scope, ts(1))


def test_legacy_naive_query_input_compatibility_is_unchanged():
    store = AdvancedMemoryStore(enable_auto_cleanup=False)
    store.add_entry(
        MemoryEntry(
            id="legacy",
            content="old",
            t_valid=ts(1).isoformat(),
            t_created=ts(1).isoformat(),
        )
    )
    assert [entry.id for entry in store.query_as_of(ts(2).replace(tzinfo=None))] == [
        "legacy"
    ]
    assert [entry.id for entry in store.query_as_of("2026-02-01T00:00:00")] == [
        "legacy"
    ]

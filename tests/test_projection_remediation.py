"""Regression evidence for current-authority eligibility and administrative races."""

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from test_projection_consumers import drain, relation, write
from test_projection_consumers import env as projection_env

from jitmind.projections import ProjectionError
from jitmind.storage import IngestRequest, InvalidRequest, Proposal


@pytest.fixture
def env(tmp_path):
    return projection_env.__wrapped__(tmp_path)


def query(consumer, a, s, source, **kwargs):
    if consumer.kind == "vector":
        return consumer.search(a, s, source, "tea", **kwargs)
    return consumer.traverse(a, s, source, "a", **kwargs)


def ids(result):
    return {hit.fact_id for hit in result.hits} | {
        hit.fact_id for edge in result.edges for hit in edge.supports
    }


@pytest.mark.parametrize(
    "ttl", ["absent", 86400, 0, -1, True, False, None, "60", 1e308]
)
def test_ttl_matches_current_c03_and_preserves_positive_control(env, ttl):
    a, s, db, source, v, g = env
    meta = {"relations": [relation("a", "b")]}
    if ttl != "absent":
        meta["ttl_seconds"] = ttl
    subject = write(db, "subject", meta=meta, t_valid="2000-01-01T00:00:00Z")
    control = write(
        db,
        "control",
        meta={"relations": [relation("a", "c")]},
        t_valid="2000-01-01T00:00:00Z",
    )
    drain(env)
    valid = ttl == "absent" or (type(ttl) is int and ttl == 86400)
    unknown = not valid and not (type(ttl) is int and ttl == 0)
    expected = {control.memory_id} | ({subject.memory_id} if valid else set())
    for consumer in (v, g):
        result = query(consumer, a, s, source)
        assert ids(result) == expected
        assert result.status == ("partial" if unknown else "complete")
        assert ("unknown_eligibility" in result.reasons) == unknown
    if hasattr(db, "snapshot_at"):
        now = datetime.now(timezone.utc)
        actual = db.snapshot_at("ns", valid_at=now, eligible_at=now)
        assert {entry.id for entry in actual.entries} == expected
        assert actual.unknown_eligibility == unknown


@pytest.mark.parametrize("ttl", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_ttl_rejected_on_ingest_and_corrupt_legacy_read(env, ttl):
    a, s, db, source, v, g = env
    with pytest.raises(InvalidRequest):
        write(db, "invalid", meta={"ttl_seconds": ttl})
    subject = write(db, "subject", meta={"relations": [relation("a", "b")]})
    drain(env)
    # Explicit disposable corruption fixture: current source decoder must reject
    # non-JSON numeric values even if a legacy external writer persisted them.
    with sqlite3.connect(db.path) as conn:
        payload = json.loads(
            conn.execute(
                "SELECT payload FROM facts WHERE memory_id=?", (subject.memory_id,)
            ).fetchone()[0]
        )
        payload["meta"]["ttl_seconds"] = ttl
        conn.execute(
            "UPDATE facts SET payload=? WHERE memory_id=?",
            (json.dumps(payload), subject.memory_id),
        )
    for consumer in (v, g):
        result = query(consumer, a, s, source)
        assert result.status == "unavailable" and not ids(result)


@pytest.mark.parametrize("expiry", [None, "2001-01-01T00:00:00Z"])
def test_c03_corrected_validity_retains_historical_page_and_independent_expiry(
    env, expiry
):
    a, s, db, source, v, g = env
    if not hasattr(db, "commit_temporal"):
        pytest.skip("Requires actual C03 transactional-history dependency")
    from jitmind.storage import ValidityCorrection

    meta = {"relations": [relation("a", "b")]}
    if expiry:
        meta["expires_at"] = expiry
    subject = write(
        db,
        "old",
        meta=meta,
        t_valid="2000-01-01T00:00:00Z",
        t_invalid="2001-01-01T00:00:00Z",
    )
    drain(env)
    page = db.historical_page("ns", subject.page_id)
    assert page.meta["t_invalid"] == "2001-01-01T00:00:00Z"
    for consumer in (v, g):
        assert not ids(query(consumer, a, s, source))
    db.commit_temporal(
        IngestRequest.create("ns", "reopen", "correction"),
        1,
        Proposal(
            abstract="correction",
            header="correction",
            decorated="correction",
            decision={"operation": "noop"},
        ),
        corrections=(
            ValidityCorrection(subject.memory_id, "2000-01-01T00:00:00Z", None),
        ),
    )
    drain(env)
    for consumer in (v, g):
        result = query(consumer, a, s, source)
        assert result.status == "complete"
        assert ids(result) == (set() if expiry else {subject.memory_id})
    now = datetime.now(timezone.utc)
    assert bool(db.snapshot_at("ns", valid_at=now, eligible_at=now).entries) == (
        expiry is None
    )
    assert db.historical_page("ns", subject.page_id) == page
    assert not db.snapshot_at("ns", revision=1, valid_at=now).entries
    db.commit_temporal(
        IngestRequest.create("ns", "narrow", "correction"),
        2,
        Proposal(
            abstract="correction",
            header="correction",
            decorated="correction",
            decision={"operation": "noop"},
        ),
        corrections=(
            ValidityCorrection(
                subject.memory_id, "2000-01-01T00:00:00Z", "2002-01-01T00:00:00Z"
            ),
        ),
    )
    drain(env)
    for consumer in (v, g):
        assert not ids(query(consumer, a, s, source))
    assert db.historical_page("ns", subject.page_id) == page


@pytest.mark.parametrize("kind", [4, 5])
def test_candidate_limit_bounds_payload_select_and_decode(env, monkeypatch, kind):
    a, s, db, source, _, _ = env
    for i in range(4):
        write(db, str(i), meta={"relations": [relation("a", "b")]})
    drain(env)
    import jitmind.projections.source as module

    original = module._entry
    decoded, queries = [], []

    def decode(payload):
        value = original(payload)
        decoded.append(value.id)
        return value

    read = source._read

    @contextmanager
    def tracked():
        with read() as conn:
            conn.set_trace_callback(queries.append)
            yield conn

    monkeypatch.setattr(module, "_entry", decode)
    monkeypatch.setattr(source, "_read", tracked)
    result = query(env[kind], a, s, source, candidate_limit=1)
    assert result.status == "partial" and "snapshot_budget" in result.reasons
    assert len(decoded) == 2 and len(set(decoded)) == 1
    payload_queries = [sql for sql in queries if " AS payload," in sql]
    assert len(payload_queries) == 2
    assert all(sql.endswith("LIMIT 1") for sql in payload_queries)


@pytest.mark.parametrize("kind", [4, 5])
@pytest.mark.parametrize("operation", ["deliver", "rebuild"])
@pytest.mark.parametrize(
    "change", ["disable", "toggle", "rebuild", "selection", "delivery"]
)
def test_preparation_fences_admin_generation_and_selection(
    env, monkeypatch, kind, operation, change
):
    a, s, db, source, _, _ = env
    consumer = env[kind]
    old = write(
        db,
        "old",
        meta={
            "repo_id": "repo",
            "snapshot_id": "old",
            "relations": [relation("a", "b")],
        },
    )
    new = write(
        db,
        "new",
        meta={
            "repo_id": "repo",
            "snapshot_id": "new",
            "relations": [relation("a", "c")],
        },
    )
    event = source.events(a, s)[0]
    prepare = consumer._prepare
    captured = []

    def interleaved(fact):
        result = prepare(fact)
        monkeypatch.setattr(consumer, "_prepare", prepare)
        if change == "disable":
            consumer.admin_disable()
        elif change == "toggle":
            consumer.admin_disable()
            consumer.admin_disable(False)
        elif change == "delivery":
            consumer.deliver(a, s, source, event, snapshots=(("repo", "old"),))
        else:
            consumer.rebuild(
                a,
                s,
                source,
                snapshots=(("repo", "new" if change == "selection" else "old"),),
            )
        captured.append(consumer.export("ns"))
        return result

    monkeypatch.setattr(consumer, "_prepare", interleaved)
    with pytest.raises(ProjectionError, match="consumer_changed|consumer_disabled"):
        if operation == "deliver":
            consumer.deliver(a, s, source, event, snapshots=(("repo", "old"),))
        else:
            consumer.rebuild(a, s, source, snapshots=(("repo", "old"),))
    assert consumer.export("ns") == captured[0]
    if change == "selection":
        result = query(consumer, a, s, source, snapshots=(("repo", "new"),))
        assert ids(result) == {new.memory_id}
        assert old.memory_id not in ids(result)


@pytest.mark.parametrize("kind", [4, 5])
@pytest.mark.parametrize("operation", ["deliver", "rebuild"])
def test_source_acquisition_also_fenced(env, monkeypatch, kind, operation):
    a, s, db, source, _, _ = env
    consumer = env[kind]
    write(db, "one")
    event = source.events(a, s)[0]
    snapshot = source.snapshot

    def changed(*args, **kwargs):
        result = snapshot(*args, **kwargs)
        monkeypatch.setattr(source, "snapshot", snapshot)
        consumer.admin_disable()
        consumer.admin_disable(False)
        return result

    monkeypatch.setattr(source, "snapshot", changed)
    with pytest.raises(ProjectionError, match="consumer_changed"):
        if operation == "deliver":
            consumer.deliver(a, s, source, event)
        else:
            consumer.rebuild(a, s, source)
    assert consumer.export("ns")["facts"] == []
    assert consumer.export("ns")["receipts"] == []


@pytest.mark.parametrize("kind", [4, 5])
def test_same_done_event_while_disabled_and_reenabled_restart(env, kind):
    a, s, db, source, _, _ = env
    consumer = env[kind]
    write(db, "one", meta={"relations": [relation("a", "b")]})
    drain(env)
    event = source.events(a, s)[0]
    before = consumer.export("ns")
    consumer.admin_disable()
    with pytest.raises(ProjectionError, match="consumer_disabled"):
        consumer.deliver(a, s, source, event)
    assert consumer.export("ns") == before
    # Explicit maintenance while disabled remains supported.
    consumer.rebuild(a, s, source)
    assert query(consumer, a, s, source).status == "unavailable"
    consumer.admin_disable(False)
    kwargs = {"trusted_relations": True} if consumer.kind == "graph" else {}
    restarted = type(consumer)(consumer.path, source_id=source.source_id, **kwargs)
    restarted.deliver(a, s, source, event)
    assert query(restarted, a, s, source).status == "complete"


@pytest.mark.parametrize("kind", [4, 5])
def test_final_revalidation_rejects_new_expiry(env, kind):
    a, s, db, source, _, _ = env
    subject = write(db, "one", meta={"relations": [relation("a", "b")]})
    drain(env)
    consumer = env[kind]

    def expire(stage):
        if stage == "before_return":
            # Disposable current-row mutation with no revision bump: fingerprint
            # revalidation must still reject previously ranked/expanded results.
            with sqlite3.connect(db.path) as conn:
                payload = json.loads(
                    conn.execute(
                        "SELECT payload FROM facts WHERE memory_id=?",
                        (subject.memory_id,),
                    ).fetchone()[0]
                )
                payload["meta"]["ttl_seconds"] = 0
                conn.execute(
                    "UPDATE facts SET payload=? WHERE memory_id=?",
                    (json.dumps(payload), subject.memory_id),
                )

    consumer.fault_hook = expire
    result = query(consumer, a, s, source)
    assert result.status == "unavailable" and not ids(result)


@pytest.mark.parametrize("raw", [[1e308, -1e308], [5e-324, 5e-324], [0.0, 0.0]])
def test_extreme_normalization_remains_finite(raw):
    from jitmind.projections.vector import normalized

    actual = normalized(raw, 2)
    assert all(math.isfinite(value) for value in actual)
    assert sum(value * value for value in actual) == pytest.approx(
        0 if raw == [0.0, 0.0] else 1
    )


@pytest.mark.parametrize(
    "restriction", [{"expires_at": "2001-01-01T00:00:00Z"}, {"ttl_seconds": 0}]
)
def test_independent_page_expiry_is_not_widened(env, restriction):
    a, s, db, source, v, g = env
    subject = write(db, "one", meta={"relations": [relation("a", "b")], **restriction})
    # Explicit legacy split-metadata fixture: retained page owns an independent
    # restriction even where the current fact has no duplicate expiry metadata.
    with sqlite3.connect(db.path) as conn:
        page_before = conn.execute(
            "SELECT payload FROM pages WHERE page_id=?", (subject.page_id,)
        ).fetchone()[0]
        payload = json.loads(
            conn.execute(
                "SELECT payload FROM facts WHERE memory_id=?", (subject.memory_id,)
            ).fetchone()[0]
        )
        for key in restriction:
            del payload["meta"][key]
        conn.execute(
            "UPDATE facts SET payload=? WHERE memory_id=?",
            (json.dumps(payload), subject.memory_id),
        )
    drain(env)
    for consumer in (v, g):
        result = query(consumer, a, s, source)
        assert result.status == "complete" and not ids(result)
    with sqlite3.connect(db.path) as conn:
        assert (
            conn.execute(
                "SELECT payload FROM pages WHERE page_id=?", (subject.page_id,)
            ).fetchone()[0]
            == page_before
        )


@pytest.mark.parametrize("kind", [4, 5])
def test_ttl_expiring_during_read_rejected_without_source_mutation(
    env, monkeypatch, kind
):
    from datetime import timedelta

    import jitmind.projections.source as source_module

    a, s, db, source, _, _ = env
    subject = write(
        db, "one", meta={"ttl_seconds": 60, "relations": [relation("a", "b")]}
    )
    with sqlite3.connect(db.path) as conn:
        payload = json.loads(
            conn.execute(
                "SELECT payload FROM facts WHERE memory_id=?", (subject.memory_id,)
            ).fetchone()[0]
        )
    created = datetime.fromisoformat(payload["t_created"].replace("Z", "+00:00"))

    class Clock(datetime):
        current = created + timedelta(seconds=59)

        @classmethod
        def now(cls, tz=None):
            return cls.current

    monkeypatch.setattr(source_module, "datetime", Clock)
    drain(env)
    consumer = env[kind]
    assert ids(query(consumer, a, s, source)) == {subject.memory_id}

    def advance(stage):
        if stage == "before_return":
            Clock.current = created + timedelta(seconds=60)

    consumer.fault_hook = advance
    result = query(consumer, a, s, source)
    assert result.status == "unavailable" and result.reasons == ("authority_changed",)
    assert not ids(result)
    consumer.fault_hook = None
    assert not ids(query(consumer, a, s, source))

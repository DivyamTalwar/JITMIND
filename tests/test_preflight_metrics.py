"""Disposable real-storage measurement tests; no provider or parser substitutes."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from threading import Barrier
from types import SimpleNamespace

import pytest
from test_preflight import REPO, add
from test_preflight import setup as setup_fixture

from jitmind.code_memory.preflight import (
    PreflightPolicy,
    PreflightResult,
    PreflightService,
)
from jitmind.code_memory.telemetry import STAGES, BoundedTraceSink, _Recorder, _recorder
from jitmind.code_memory.work_storage import Budget, Conflict, WorkError
from jitmind.scope import ScopeDenied

setup = setup_fixture


def stages(metrics):
    return metrics.to_dict()["stages"]


def test_exclusive_repeated_nested_spans_exact_clock():
    clock = [10.0]
    recorder = _Recorder(clock=lambda: clock[0])
    clock[0] = 11
    with recorder.span("lookup"):
        clock[0] = 13
        with recorder.span("eligibility"):
            clock[0] = 16
            with recorder.span("eligibility"):
                clock[0] = 17
            clock[0] = 18
        clock[0] = 20
    with recorder.span("dedup"):
        clock[0] = 22
        with recorder.span("rendering"):
            clock[0] = 25
        clock[0] = 26
    clock[0] = 27
    result = recorder.finish("delivered")
    measured = stages(result)
    assert result.elapsed_seconds == 17
    assert measured["lookup"]["elapsed_seconds"] == 4
    assert measured["eligibility"]["elapsed_seconds"] == 5
    assert measured["eligibility"]["calls"] == 2
    assert measured["dedup"]["elapsed_seconds"] == 3
    assert measured["rendering"]["elapsed_seconds"] == 3
    assert measured["ranking"]["elapsed_seconds"] is None
    assert measured["ranking"]["status"] == "not_entered"
    assert sum(s["elapsed_seconds"] or 0 for s in measured.values()) == 15
    assert json.loads(result.to_json()) == result.to_dict()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, "secret-clock"])
def test_clock_anomalies_are_null_and_finite_json(bad):
    clock = [10.0]
    recorder = _Recorder(clock=lambda: clock[0])
    with recorder.span("lookup"):
        clock[0] = bad
    clock[0] = 20.0
    result = recorder.finish("unavailable")
    assert result.elapsed_seconds is None and result.clock_anomalies == 1
    assert stages(result)["lookup"]["status"] == "clock_invalid"
    assert stages(result)["lookup"]["elapsed_seconds"] is None
    assert "secret-clock" not in result.to_json()


def test_clock_exception_and_real_zero_are_distinguished():
    def broken():
        raise RuntimeError("secret-clock-error")

    recorder = _Recorder(clock=broken)
    with recorder.span("rendering"):
        pass
    assert recorder.finish("unavailable").elapsed_seconds is None
    recorder = _Recorder(clock=lambda: 1.0)
    with recorder.span("lookup"):
        pass
    result = recorder.finish("nothing_relevant")
    assert stages(result)["lookup"]["elapsed_seconds"] == 0
    assert stages(result)["rendering"]["elapsed_seconds"] is None


def test_real_fact_projection_binding_callbacks_and_receipt_replay(setup):
    _, scope, _, _, projection, service, session, target = setup
    calls = []

    class BindingAuthority:
        def is_current(self, callback_scope, repo, lesson, budget):
            assert callback_scope is scope and repo == target.repo_id
            budget.remaining()
            calls.append(lesson.binding_id)
            return True

    projection.binding_authority = BindingAuthority()
    add(setup, binding_id="private-binding", binding_revision=1)
    sink = BoundedTraceSink(2)
    service.metrics_sink = sink
    first = service.deliver(scope, session, target, delivery_key="private-receipt")
    assert first.state == "delivered"
    assert len(calls) == 4  # acquisition, selection, precommit, postcommit
    result = first.metrics
    assert result.outcome == "delivered" and result.export_status == "published"
    assert result.elapsed_seconds >= sum(
        s["elapsed_seconds"] for s in stages(result).values()
    )
    assert all(s["status"] == "measured" for s in stages(result).values())
    replay = service.deliver(scope, session, target, delivery_key="private-receipt")
    assert replay.replayed and replay.payload == first.payload
    assert len(calls) == 7  # replay keeps original three authority checks
    assert (
        stages(replay.metrics)["eligibility"]["calls"]
        < stages(result)["eligibility"]["calls"]
    )
    dedup = service.deliver(scope, session, target)
    assert dedup.state == "nothing_relevant"
    assert stages(dedup.metrics)["rendering"]["status"] == "not_entered"
    assert len(sink.snapshot()) == 2
    assert sink.snapshot() == (replay.metrics, dedup.metrics)
    encoded = result.to_json()
    for secret in (
        target.file_path,
        target.repo_id,
        scope.principal_id,
        "private-binding",
        "private-receipt",
        first.payload,
        "lesson",
    ):
        assert secret not in encoded
    with pytest.raises(FrozenInstanceError):
        result.outcome = "forged"


def test_no_lessons_and_backward_default_and_disabled(setup):
    _, scope, _, _, projection, service, session, target = setup
    result = service.deliver(scope, session, target)
    assert result.state == "nothing_relevant"
    assert stages(result.metrics)["lookup"]["calls"] == 1
    assert stages(result.metrics)["dedup"]["calls"] == 1
    assert stages(result.metrics)["rendering"]["elapsed_seconds"] is None
    assert PreflightResult("deferred", "deadline_or_contention").metrics is None
    disabled = PreflightService(projection, measure=False)
    session = disabled.start_session(scope, REPO)
    assert disabled.deliver(scope, session, target).metrics is None
    assert _recorder.get() is None


def test_denied_before_acquisition_and_invalid_and_conflict(setup):
    authority, scope, _, _, _, service, session, target = setup
    sink = service.metrics_sink = BoundedTraceSink()
    with pytest.raises(WorkError):
        service.deliver(scope, session, replace(target, file_path="../secret"))
    assert sink.snapshot()[-1].outcome == "invalid"
    service.deliver(scope, session, target, delivery_key="empty")
    add(setup)
    service.deliver(scope, session, target, delivery_key="first")
    with pytest.raises(Conflict):
        service.deliver(
            scope, session, replace(target, symbol="other"), delivery_key="first"
        )
    assert sink.snapshot()[-1].outcome == "conflict"
    authority.grant("alice", "tenant", [])
    with pytest.raises(ScopeDenied):
        service.deliver(scope, session, target)
    denied = sink.snapshot()[-1]
    assert denied.outcome == "denied"
    assert stages(denied)["eligibility"]["failed"] == 1
    for name in ("lookup", "ranking", "rendering", "dedup"):
        assert stages(denied)[name]["elapsed_seconds"] is None
    assert _recorder.get() is None


@pytest.mark.parametrize(
    "exception", [RuntimeError("secret-error"), KeyboardInterrupt("secret-cancel")]
)
def test_failure_and_cancellation_retain_spans_and_reset(setup, monkeypatch, exception):
    _, scope, _, _, projection, service, session, target = setup
    add(setup)
    sink = service.metrics_sink = BoundedTraceSink()

    def fail(*args):
        raise exception

    monkeypatch.setattr(projection.facts, "is_active", fail)
    if isinstance(exception, Exception):
        result = service.deliver(scope, session, target)
        assert result.state == "unavailable" and not result.payload
    else:
        with pytest.raises(KeyboardInterrupt):
            service.deliver(scope, session, target)
    metrics = sink.snapshot()[-1]
    assert metrics.outcome == (
        "unavailable" if isinstance(exception, Exception) else "cancelled"
    )
    assert stages(metrics)["lookup"]["failed"] == 1
    assert stages(metrics)["eligibility"]["failed"] == 1
    assert stages(metrics)["rendering"]["elapsed_seconds"] is None
    assert "secret" not in metrics.to_json()
    assert _recorder.get() is None


def test_default_deadline_preserved_with_known_measurements(setup, monkeypatch):
    import jitmind.code_memory.work_storage as storage

    _, scope, _, database, _, service, session, target = setup
    add(setup)
    service.policy = PreflightPolicy()
    assert service.policy.deadline_seconds == 0.15
    clock = [10.0]
    monkeypatch.setattr(storage, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    service.approved_repo = lambda *args: clock.__setitem__(0, 11.0) or True
    result = service.deliver(scope, session, target)
    assert result.state == "deferred" and not result.payload
    assert stages(result.metrics)["eligibility"]["elapsed_seconds"] is not None
    assert stages(result.metrics)["lookup"]["elapsed_seconds"] is None
    with database.connect(Budget(1)) as db:
        assert db.execute("SELECT count(*) FROM delivery_receipts").fetchone()[0] == 0


def test_storage_contention_then_restart_retains_receipt(setup):
    _, scope, _, database, projection, service, session, target = setup
    add(setup)
    with sqlite3.connect(database.path, isolation_level=None, timeout=0) as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            result = service.deliver(scope, session, target)
            assert result.state == "deferred" and not result.payload
            assert stages(result.metrics)["lookup"]["completed"] == 1
            assert stages(result.metrics)["dedup"]["failed"] == 1
            assert stages(result.metrics)["rendering"]["elapsed_seconds"] is None
        finally:
            db.execute("ROLLBACK")
    assert service.deliver(scope, session, target).state == "delivered"
    restarted = PreflightService(projection, metrics_sink=BoundedTraceSink())
    with pytest.raises(ScopeDenied):
        restarted.deliver(scope, session, target)
    assert restarted.metrics_sink.snapshot()[-1].outcome == "denied"
    new_session = restarted.start_session(scope, REPO)
    assert restarted.deliver(scope, new_session, target).state == "delivered"
    with database.connect(Budget(1)) as db:
        assert db.execute("SELECT count(*) FROM delivery_receipts").fetchone()[0] == 2


@pytest.mark.parametrize("failure", ["contended", "failed"])
def test_export_failure_cannot_change_delivery(setup, failure):
    _, scope, _, _, _, service, session, target = setup
    sink = service.metrics_sink = BoundedTraceSink()
    if failure == "failed":

        class BrokenBuffer:
            def append(self, value):
                raise RuntimeError("export-secret")

        sink._traces = BrokenBuffer()
    else:
        sink._lock.acquire()
    try:
        result = service.deliver(scope, session, target)
        assert result.state == "nothing_relevant" and not result.payload
        assert result.metrics.export_status == failure
        add(setup)
        result = service.deliver(scope, session, target)
        assert result.state == "delivered" and result.metrics.export_status == failure
        assert "export-secret" not in result.metrics.to_json()
    finally:
        if failure == "contended":
            sink._lock.release()


def test_sink_and_metadata_cannot_supply_authority(setup):
    _, scope, _, _, projection, service, session, target = setup
    with pytest.raises(WorkError):
        PreflightService(projection, metrics_sink=lambda trace: True)
    with pytest.raises(TypeError):
        service.deliver(scope, session, target, recorder={"eligible": True})
    with pytest.raises(TypeError):
        service.deliver(scope, session, target, metadata={"trusted": True})
    for capacity in (0, -1, 1025, True):
        with pytest.raises(ValueError):
            BoundedTraceSink(capacity)


def test_concurrent_users_isolate_request_local_traces(setup):
    authority, scope, _, _, _, service, session, target = setup
    add(setup)
    bob = authority.context("bob", "tenant", "bob-request-secret")
    bob_session = service.start_session(bob, REPO)
    barrier = Barrier(2)
    service.approved_repo = lambda *args: barrier.wait(timeout=5) is not None
    rank_barrier = Barrier(2)
    original_rank = service._rank

    def synchronized_rank(*args):
        ranked = original_rank(*args)
        # Both real acquisitions finish before either writer can contend.
        rank_barrier.wait(timeout=5)
        return ranked

    service._rank = synchronized_rank
    service.metrics_sink = BoundedTraceSink(4)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(service.deliver, scope, session, target)
        second = pool.submit(
            service.deliver, bob, bob_session, replace(target, file_path="other.py")
        )
        alice_result, bob_result = first.result(), second.result()
    # Shared SQLite writers may legitimately contend; each trace must still
    # contain exactly its own lookup/rank and zero/one rendering invocation.
    assert alice_result.state in ("delivered", "deferred")
    assert bob_result.state in ("nothing_relevant", "deferred")
    assert alice_result.metrics is not bob_result.metrics
    for result in (alice_result, bob_result):
        trace = stages(result.metrics)
        assert trace["lookup"]["calls"] == 1
        assert trace["ranking"]["calls"] == 1
        assert trace["dedup"]["calls"] == 1
        assert trace["rendering"]["calls"] <= 1
        assert result.metrics.outcome == result.state
        assert set(trace) == set(STAGES)
    assert stages(bob_result.metrics)["rendering"]["calls"] == 0
    assert len(service.metrics_sink.snapshot()) == 2
    assert _recorder.get() is None


def test_revocation_during_render_rolls_back_and_records_denial(setup, monkeypatch):
    authority, scope, _, database, _, service, session, target = setup
    add(setup)
    sink = service.metrics_sink = BoundedTraceSink()
    original = service._render

    def revoke(lessons):
        payload = original(lessons)
        authority.grant("alice", "tenant", [])
        return payload

    monkeypatch.setattr(service, "_render", revoke)
    with pytest.raises(ScopeDenied):
        service.deliver(scope, session, target)
    trace = sink.snapshot()[-1]
    assert trace.outcome == "denied"
    assert stages(trace)["rendering"]["completed"] == 1
    assert stages(trace)["dedup"]["failed"] == 1
    with database.connect(Budget(1)) as db:
        assert db.execute("SELECT count(*) FROM delivery_receipts").fetchone()[0] == 0


def test_failed_receipt_write_preserves_render_measurement(setup):
    _, scope, _, database, _, service, session, target = setup
    add(setup)
    with database.connect(Budget(1)) as db:
        db.execute(
            "CREATE TRIGGER fail_receipt BEFORE INSERT ON delivery_receipts "
            "BEGIN SELECT RAISE(ABORT, 'private-storage-error'); END"
        )
    result = service.deliver(scope, session, target)
    assert result.state == "unavailable" and not result.payload
    assert stages(result.metrics)["rendering"]["completed"] == 1
    assert stages(result.metrics)["dedup"]["failed"] == 1
    assert "private-storage-error" not in result.metrics.to_json()
    with database.connect(Budget(1)) as db:
        assert db.execute("SELECT count(*) FROM deliveries").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM delivery_receipts").fetchone()[0] == 0
        db.execute("DROP TRIGGER fail_receipt")
    assert service.deliver(scope, session, target).state == "delivered"


def test_nested_request_restores_outer_context_and_disabled_masks_it(setup):
    _, scope, _, _, projection, service, session, target = setup
    child = PreflightService(projection, measure=False)
    child_session = child.start_session(scope, REPO)

    def registry(*args):
        outer = _recorder.get()
        assert outer is not None
        assert child.deliver(scope, child_session, target).metrics is None
        assert _recorder.get() is outer
        return True

    service.approved_repo = registry
    result = service.deliver(scope, session, target)
    assert stages(result.metrics)["lookup"]["calls"] == 1
    assert stages(result.metrics)["dedup"]["calls"] == 1
    assert _recorder.get() is None


def test_recursive_span_storage_is_bounded():
    recorder = _Recorder(clock=lambda: 1.0)

    def recurse(depth):
        with recorder.span("lookup"):
            assert len(recorder.stack) <= 32
            if depth:
                recurse(depth - 1)

    recurse(50)
    result = recorder.finish("unavailable")
    assert result.span_overflows == 19
    assert result.clock_anomalies == 0
    assert result.elapsed_seconds is None
    assert stages(result)["lookup"]["calls"] == 51
    assert stages(result)["lookup"]["completed"] == 51
    assert stages(result)["lookup"]["elapsed_seconds"] is None


def test_observational_metrics_preserve_result_equality(setup):
    _, scope, _, _, _, service, session, target = setup
    first = service.deliver(scope, session, target)
    second = service.deliver(scope, session, target)
    assert first.state == second.state == "nothing_relevant"
    assert first.metrics is not None and second.metrics is not None
    changed_metrics = replace(first.metrics, elapsed_seconds=123.0, export_status="contended")
    observed = replace(first, metrics=changed_metrics)
    assert first == second == observed == replace(first, metrics=None)
    assert replace(first, reason="different_semantic_reason") != first


def test_observational_metrics_preserve_result_hash_identity(setup):
    _, scope, _, _, _, service, session, target = setup
    first = service.deliver(scope, session, target)
    second = service.deliver(scope, session, target)
    assert len({first, second, replace(first, metrics=None)}) == 1
    values = {first: "original"}
    assert values[second] == "original"
    assert len({first, replace(first, state="unavailable")}) == 2

"""Content-free, bounded, process-local preflight measurements (schema v1).

No exporter callbacks run on the delivery path. See docs/preflight-measurement.md.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from functools import wraps
from threading import Lock

STAGES = ("lookup", "eligibility", "ranking", "rendering", "dedup")
_MAX_COUNT = 2**31 - 1


@dataclass(frozen=True)
class StageMetrics:
    name: str
    calls: int
    completed: int
    failed: int
    accepted: int
    rejected: int
    elapsed_seconds: float | None
    status: str


@dataclass(frozen=True)
class PreflightMetrics:
    outcome: str
    elapsed_seconds: float | None
    clock_anomalies: int
    stages: tuple[StageMetrics, ...]
    export_status: str = "not_configured"
    schema_version: int = 1
    span_overflows: int = 0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["stages"] = {
            stage.name: {k: v for k, v in asdict(stage).items() if k != "name"}
            for stage in self.stages
        }
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)


class BoundedTraceSink:
    """Host-owned ring buffer. Export snapshots outside the delivery call.

    No user code, I/O or blocking lock acquisition occurs during publication.
    Full buffers evict their oldest trace. Snapshot readers use the same lock;
    a contending publisher declines publication instead of waiting.
    """

    def __init__(self, capacity: int = 64):
        if type(capacity) is not int or not 1 <= capacity <= 1024:
            raise ValueError("invalid_metrics_capacity")
        self._traces: deque[PreflightMetrics] = deque(maxlen=capacity)
        self._lock = Lock()

    def snapshot(self) -> tuple[PreflightMetrics, ...]:
        with self._lock:
            return tuple(self._traces)

    def _publish(self, metrics: PreflightMetrics) -> PreflightMetrics:
        if not self._lock.acquire(blocking=False):
            return replace(metrics, export_status="contended")
        try:
            published = replace(metrics, export_status="published")
            self._traces.append(published)
            return published
        finally:
            self._lock.release()


class _Recorder:
    """Exclusive segment accounting; never retains arguments or exceptions."""

    def __init__(self, clock=None):
        self.clock = clock or time.monotonic
        self.values = {
            name: {
                "calls": 0,
                "completed": 0,
                "failed": 0,
                "accepted": 0,
                "rejected": 0,
            }
            for name in STAGES
        }
        self.seconds = dict.fromkeys(STAGES, 0.0)
        self.invalid: set[str] = set()
        self.stack: list[str] = []
        self.anomalies = 0
        self.overflows = 0
        self.last = None
        self.total = 0.0
        self._tick()

    def _count(self, name, key):
        self.values[name][key] = min(_MAX_COUNT, self.values[name][key] + 1)

    def _tick(self):
        try:
            now = self.clock()
            valid = type(now) in (int, float) and math.isfinite(now)
            delta = now - self.last if valid and self.last is not None else 0.0
            valid = valid and math.isfinite(delta) and delta >= 0
            valid = valid and math.isfinite(self.total + delta)
        except Exception:  # noqa: BLE001 -- clocks cannot expose diagnostic text
            now, delta, valid = None, 0.0, False
        if not valid:
            self.anomalies = min(_MAX_COUNT, self.anomalies + 1)
            self.invalid.update(self.stack)
            self.last = None
            return
        # After an invalid read, no elapsed interval can be reconstructed.
        if self.last is None and self.anomalies:
            self.invalid.update(self.stack)
        if self.stack:
            self.seconds[self.stack[-1]] += delta
        self.total += delta
        self.last = now

    @contextmanager
    def span(self, name):
        previous_anomalies = self.anomalies
        self._tick()
        if self.last is None or self.anomalies != previous_anomalies:
            self.invalid.add(name)
        self._count(name, "calls")
        # Product nesting is bounded; also bound unexpected host recursion.
        if len(self.stack) >= 32:
            self.invalid.update(STAGES)
            self.overflows = min(_MAX_COUNT, self.overflows + 1)
            try:
                yield
            except BaseException:
                self._count(name, "failed")
                raise
            else:
                self._count(name, "completed")
            return
        self.stack.append(name)
        try:
            yield
        except BaseException:
            self._count(name, "failed")
            raise
        else:
            self._count(name, "completed")
        finally:
            self._tick()
            self.stack.pop()

    def finish(self, outcome):
        self._tick()
        return PreflightMetrics(
            outcome=outcome,
            elapsed_seconds=None if self.anomalies or self.overflows else self.total,
            span_overflows=self.overflows,
            clock_anomalies=self.anomalies,
            stages=tuple(
                StageMetrics(
                    name=name,
                    **self.values[name],
                    elapsed_seconds=(
                        self.seconds[name]
                        if self.values[name]["calls"] and name not in self.invalid
                        else None
                    ),
                    status=(
                        "not_entered"
                        if not self.values[name]["calls"]
                        else "measurement_invalid"
                        if self.overflows
                        else "clock_invalid"
                        if name in self.invalid
                        else "measured"
                    ),
                )
                for name in STAGES
            ),
        )


_recorder: ContextVar[_Recorder | None] = ContextVar("preflight_recorder", default=None)


@contextmanager
def _stage(name):
    recorder = _recorder.get()
    if recorder is None:
        yield
    else:
        with recorder.span(name):
            yield


def _measured(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            recorder = _recorder.get()
            if recorder is None:
                return function(*args, **kwargs)
            with recorder.span(name):
                result = function(*args, **kwargs)
                if type(result) is bool:
                    recorder._count(name, "accepted" if result else "rejected")
                return result

        return wrapped

    return decorate


def _finish(recorder, outcome, sink):
    metrics = recorder.finish(outcome)
    if sink is not None:
        try:
            # Deliberately bypass overrides: delivery accepts no exporter callback.
            metrics = BoundedTraceSink._publish(sink, metrics)
        except Exception:  # noqa: BLE001 -- optional export cannot change delivery
            metrics = replace(metrics, export_status="failed")
    return metrics

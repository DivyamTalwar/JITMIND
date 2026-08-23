"""Dependency-free, privacy-safe retrieval traces and sinks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterator, List, Optional, Protocol
import uuid

from jitmind.utils.file_lock import file_lock


@dataclass
class TraceStage:
    name: str
    started_at: str
    duration_ms: float = 0.0
    input_count: Optional[int] = None
    output_count: Optional[int] = None
    status: str = "ok"
    error_type: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    trace_id: str
    started_at: str
    query_sha256: str
    query_chars: int
    query: Optional[str] = None
    duration_ms: float = 0.0
    status: str = "running"
    stages: List[TraceStage] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TraceSink(Protocol):
    def emit(self, trace: RetrievalTrace) -> None: ...


class InMemoryTraceSink:
    def __init__(self) -> None:
        self.traces: List[RetrievalTrace] = []

    def emit(self, trace: RetrievalTrace) -> None:
        self.traces.append(trace)


class JsonlTraceSink:
    """Append traces under a process-safe lock for local inspection/export."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def emit(self, trace: RetrievalTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = Path(f"{self.path}.lock")
        payload = json.dumps(trace.to_dict(), ensure_ascii=False, sort_keys=True)
        with file_lock(lock_path):
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(payload)
                stream.write("\n")


class TraceRecorder:
    def __init__(self, query: str, *, capture_content: bool = False) -> None:
        self._started = perf_counter()
        self.trace = RetrievalTrace(
            trace_id=str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc).isoformat(),
            query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            query_chars=len(query),
            query=query if capture_content else None,
        )

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        input_count: Optional[int] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[TraceStage]:
        stage = TraceStage(
            name=name,
            started_at=datetime.now(timezone.utc).isoformat(),
            input_count=input_count,
            attributes=attributes or {},
        )
        started = perf_counter()
        try:
            yield stage
        except Exception as exc:
            stage.status = "error"
            stage.error_type = type(exc).__name__
            raise
        finally:
            stage.duration_ms = round((perf_counter() - started) * 1_000, 3)
            self.trace.stages.append(stage)

    def record(
        self,
        name: str,
        *,
        input_count: Optional[int] = None,
        output_count: Optional[int] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> TraceStage:
        stage = TraceStage(
            name=name,
            started_at=datetime.now(timezone.utc).isoformat(),
            input_count=input_count,
            output_count=output_count,
            attributes=attributes or {},
        )
        self.trace.stages.append(stage)
        return stage

    def finish(self, status: str = "ok") -> RetrievalTrace:
        self.trace.status = status
        self.trace.duration_ms = round((perf_counter() - self._started) * 1_000, 3)
        return self.trace


__all__ = [
    "InMemoryTraceSink",
    "JsonlTraceSink",
    "RetrievalTrace",
    "TraceRecorder",
    "TraceSink",
    "TraceStage",
]

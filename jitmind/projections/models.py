"""Bounded immutable contracts for independently delivered projections."""

from __future__ import annotations

from dataclasses import dataclass

from jitmind.scope import ScopeAuthority, ScopeContext
from jitmind.storage.models import validate_identifier


class ProjectionError(RuntimeError):
    def __init__(self, code: str = "projection_unavailable"):
        self.code = code
        super().__init__(code)


def integer(value, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ProjectionError("invalid_integer")
    return value


def selection(authority: ScopeAuthority, scope: ScopeContext, snapshots):
    authority.require(scope)
    if type(snapshots) is not tuple or len(snapshots) > 100:
        raise ProjectionError("invalid_selection")
    for pair in snapshots:
        if type(pair) is not tuple or len(pair) != 2:
            raise ProjectionError("invalid_selection")
        for item in pair:
            validate_identifier(item)
        authority.require(scope, pair[0])
    if len({pair[0] for pair in snapshots}) != len(snapshots):
        raise ProjectionError("invalid_selection")
    return tuple(sorted(snapshots))


@dataclass(frozen=True)
class ProjectionEvent:
    namespace_id: str
    event_id: str
    revision: int
    memory_ids: tuple[str, ...]

    def validate(self):
        validate_identifier(self.namespace_id)
        validate_identifier(self.event_id)
        integer(self.revision, 1)
        if type(self.memory_ids) is not tuple or len(self.memory_ids) > 1000:
            raise ProjectionError("invalid_event")
        for item in self.memory_ids:
            validate_identifier(item)
        if len(set(self.memory_ids)) != len(self.memory_ids):
            raise ProjectionError("invalid_event")
        return self


@dataclass(frozen=True)
class FactRecord:
    fact_id: str
    revision: int
    status: str
    content: str
    metadata_json: str
    repo_id: str | None
    snapshot_id: str | None
    visible: bool


@dataclass(frozen=True)
class SourceSnapshot:
    namespace_id: str
    revision: int
    facts: tuple[FactRecord, ...]
    truncated: bool
    fingerprint: str
    anchor: ProjectionEvent | None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectionHit:
    fact_id: str
    score: float
    content: str
    metadata_json: str
    revision: int
    repo_id: str | None
    snapshot_id: str | None


@dataclass(frozen=True)
class GraphEdge:
    subject: str
    predicate: str
    object: str
    supports: tuple[ProjectionHit, ...]


@dataclass(frozen=True)
class QueryResult:
    status: str
    reasons: tuple[str, ...]
    hits: tuple[ProjectionHit, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    nodes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PurgePlan:
    namespace_id: str
    fact_id: str | None
    subjects: tuple[tuple[str, int], ...]
    fingerprint: str

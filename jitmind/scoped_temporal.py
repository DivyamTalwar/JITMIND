"""Strict bitemporal snapshots without changing legacy query_as_of semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from threading import RLock
from typing import Protocol

from jitmind.scope import ScopeAuthority, ScopeContext, _identifier
from jitmind.scoped_research import _snapshots


class TemporalConflict(ValueError):
    def __init__(self):
        super().__init__("Invalid or ambiguous temporal history")


def utc(value: str | datetime) -> datetime:
    """Strict new API: timezone required; normalize offsets to UTC."""
    try:
        parsed = (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            if type(value) is str
            else value
        )
        if (
            not isinstance(parsed, datetime)
            or parsed.tzinfo is None
            or parsed.utcoffset() is None
        ):
            raise TemporalConflict()
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        raise TemporalConflict() from None


@dataclass(frozen=True)
class TemporalFact:
    fact_id: str
    fact_key: str
    content: str
    valid_from: datetime | str
    valid_to: datetime | str | None = None
    expires_at: datetime | str | None = None
    single_valued: bool = True
    repo_id: str | None = None
    snapshot_id: str | None = None

    def __post_init__(self):
        _identifier(self.fact_id)
        _identifier(self.fact_key)
        if type(self.content) is not str or type(self.single_valued) is not bool:
            raise TemporalConflict()
        for name in ("valid_from", "valid_to", "expires_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, utc(value))
        if self.valid_from is None or (
            self.valid_to is not None and self.valid_to <= self.valid_from
        ):
            raise TemporalConflict()
        if (self.repo_id is None) != (self.snapshot_id is None):
            raise TemporalConflict()
        if self.repo_id is not None:
            _identifier(self.repo_id)
            _identifier(self.snapshot_id)


@dataclass(frozen=True)
class TemporalPerspective:
    recorded_at: datetime | str
    facts: tuple[TemporalFact, ...]

    def __post_init__(self):
        object.__setattr__(self, "recorded_at", utc(self.recorded_at))
        if type(self.facts) is not tuple or len(self.facts) > 100_000:
            raise TemporalConflict()
        if any(type(fact) is not TemporalFact for fact in self.facts):
            raise TemporalConflict()
        if len({fact.fact_id for fact in self.facts}) != len(self.facts):
            raise TemporalConflict()
        grouped = {}
        for fact in self.facts:
            grouped.setdefault(
                (fact.fact_key, fact.repo_id, fact.snapshot_id), []
            ).append(fact)
        for facts in grouped.values():
            declarations = {fact.single_valued for fact in facts}
            if len(declarations) != 1:
                raise TemporalConflict()
            if not facts[0].single_valued:
                continue
            ordered = sorted(facts, key=lambda fact: fact.valid_from)
            for left, right in pairwise(ordered):
                if left.valid_to is None or left.valid_to > right.valid_from:
                    raise TemporalConflict()


class TemporalHistoryBackend(Protocol):
    """Durable adapter contract: exact namespace reads and atomic append CAS.

    Persist complete perspectives, including original effective intervals.
    No cleanup/promotion operation may rewrite previous perspectives.
    """

    def read_perspectives(
        self, namespace_id: str
    ) -> tuple[TemporalPerspective, ...]: ...

    def append_perspective(
        self,
        namespace_id: str,
        expected_revision: int,
        perspective: TemporalPerspective,
    ) -> None: ...


class InMemoryTemporalHistory:
    def __init__(self):
        self._lock = RLock()
        self._history = {}

    def read_perspectives(self, namespace_id):
        _identifier(namespace_id)
        with self._lock:
            return self._history.get(namespace_id, ())

    def append_perspective(self, namespace_id, expected_revision, perspective):
        _identifier(namespace_id)
        if type(expected_revision) is not int or expected_revision < 0:
            raise TemporalConflict()
        if type(perspective) is not TemporalPerspective:
            raise TemporalConflict()
        with self._lock:
            old = self._history.get(namespace_id, ())
            if len(old) != expected_revision or (
                old and perspective.recorded_at <= old[-1].recorded_at
            ):
                raise TemporalConflict()
            self._history[namespace_id] = old + (perspective,)


class StrictTemporalOracle:
    """Trusted host supplies the backend; callers supply issued scope capabilities.

    publish replaces the current *complete* knowledge perspective, not a delta.
    Historical queries select the latest perspective with recorded_at <= time.
    """

    def __init__(self, authority: ScopeAuthority, backend: TemporalHistoryBackend):
        self.authority = authority
        self.backend = backend

    def publish(self, scope: ScopeContext, perspective: TemporalPerspective) -> None:
        self.authority.require(scope)
        if type(perspective) is not TemporalPerspective:
            raise TemporalConflict()
        for fact in perspective.facts:
            self.authority.require(scope, fact.repo_id)
        self.authority.require(scope)
        previous = self.backend.read_perspectives(scope.namespace_id)
        self.authority.require(scope)
        self.backend.append_perspective(scope.namespace_id, len(previous), perspective)
        self.authority.require(scope)

    def query_as_of(
        self,
        scope: ScopeContext,
        valid_at,
        *,
        transaction_at=None,
        eligible_at=None,
        snapshots=(),
    ) -> tuple[TemporalFact, ...]:
        self.authority.require(scope)
        valid = utc(valid_at)
        transaction = utc(transaction_at) if transaction_at is not None else None
        eligible = utc(eligible_at) if eligible_at is not None else None
        selected = dict(_snapshots(snapshots))
        for repo in selected:
            self.authority.require(scope, repo)
        self.authority.require(scope)
        history = self.backend.read_perspectives(scope.namespace_id)
        self.authority.require(scope)
        previous_time = None
        perspective = None
        for item in history:
            if type(item) is not TemporalPerspective or (
                previous_time is not None and item.recorded_at <= previous_time
            ):
                raise TemporalConflict()
            previous_time = item.recorded_at
            if transaction is None or item.recorded_at <= transaction:
                perspective = item
        result = []
        for fact in perspective.facts if perspective else ():
            if fact.repo_id is not None:
                if (
                    fact.repo_id not in scope.authorized_repo_ids
                    or selected.get(fact.repo_id) != fact.snapshot_id
                ):
                    continue
                self.authority.require(scope, fact.repo_id)
            if (
                fact.valid_from <= valid
                and (fact.valid_to is None or valid < fact.valid_to)
                and (
                    eligible is None
                    or fact.expires_at is None
                    or eligible < fact.expires_at
                )
            ):
                result.append(fact)
        self.authority.require(scope)
        return tuple(result)

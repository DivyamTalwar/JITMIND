"""Durable obligations, independent of observation and anchor validity.

Informed by koragraph open-loops.js / loop-anchors.js (c9ce746); attribution
and retained license in docs/open-work-preflight.md. No automatic resolution.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from uuid import UUID, uuid4

from jitmind.scope import ScopeAuthority, ScopeContext, ScopeDenied

from .work_storage import (
    Budget,
    Conflict,
    WorkDatabase,
    WorkError,
    digest,
    encoded,
    identifier,
    revision,
)

STATUSES = frozenset({"open", "in_progress", "blocked", "resolved", "cancelled"})


def _refs(values) -> tuple[str, ...]:
    if type(values) not in (tuple, list) or len(values) > 32:
        raise WorkError()
    return tuple(identifier(v) for v in values)


def _text(value: str, limit: int = 2000) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or len(value) > limit
        or "\x00" in value
    ):
        raise WorkError()
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise WorkError() from None
    return value


def _timestamp(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(identifier(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError()
        return parsed.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError):
        raise WorkError() from None


@dataclass(frozen=True)
class Obligation:
    id: str
    namespace: str
    repo: str
    creator: str
    owner: str
    revision: int
    status: str
    summary: str
    binding_refs: tuple[str, ...]
    decision_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    created_at: str
    updated_at: str
    due_at: str | None

    @classmethod
    def from_json(cls, raw: str) -> Obligation:
        data = json.loads(raw)
        for key in ("binding_refs", "decision_refs", "evidence_refs"):
            data[key] = tuple(data[key])
        return cls(**data)


@dataclass(frozen=True)
class WorkPage:
    items: tuple[Obligation, ...]
    next_cursor: int | None


class OpenWorkService:
    """Any repo-authorized caller may create self-owned work and read history.

    Only owner or a host-configured admin may transition or assign another owner.
    Resolution additionally requires a host evidence verifier, nonempty reason,
    evidence IDs, and the actor from an issued ScopeContext. Request IDs are never
    identities. Callbacks are trusted host configuration, not request fields.
    Summaries are independently authored work text; fact bodies are never copied
    or hydrated from decision/evidence/binding references.
    """

    def __init__(
        self,
        database: WorkDatabase,
        authority: ScopeAuthority,
        *,
        is_admin: Callable[[ScopeContext, str], bool] | None = None,
        verify_evidence: Callable[[ScopeContext, str, tuple[str, ...], str], bool]
        | None = None,
    ):
        self.database = database
        self.authority = authority
        self.is_admin = is_admin or (lambda scope, repo: False)
        self.verify_evidence = verify_evidence

    def _owner(self, scope, repo, owner):
        if owner != scope.principal_id and self.is_admin(scope, repo) is not True:
            raise ScopeDenied()

    def _receipt(self, db, scope, repo, key, request_digest):
        row = db.execute(
            "SELECT digest,payload FROM work_receipts WHERE namespace=? AND repo=? AND actor=? AND key=?",
            (scope.namespace_id, repo, scope.principal_id, key),
        ).fetchone()
        if row:
            if row["digest"] != request_digest:
                raise Conflict()
            return Obligation.from_json(row["payload"])
        return None

    def _save(self, db, scope, record, key, request_digest, reason, previous):
        raw = encoded(asdict(record))
        if previous is None:
            db.execute(
                "INSERT INTO work(id,namespace,repo,payload) VALUES (?,?,?,?)",
                (record.id, record.namespace, record.repo, raw),
            )
        else:
            db.execute("UPDATE work SET payload=? WHERE id=?", (raw, record.id))
        # History contains reference IDs and transition metadata, never lesson bodies.
        history = {
            "revision": record.revision,
            "actor": scope.principal_id,
            "from_status": previous,
            "to_status": record.status,
            "reason": reason,
            "evidence_refs": record.evidence_refs,
            "at": record.updated_at,
        }
        db.execute(
            "INSERT INTO history VALUES (?,?,?)",
            (record.id, record.revision, encoded(history)),
        )
        db.execute(
            "INSERT INTO work_receipts VALUES (?,?,?,?,?,?)",
            (
                record.namespace,
                record.repo,
                scope.principal_id,
                key,
                request_digest,
                raw,
            ),
        )

    def create(
        self,
        scope: ScopeContext,
        repo: str,
        *,
        summary: str,
        expected_revision: int,
        idempotency_key: str,
        owner: str | None = None,
        binding_refs=(),
        decision_refs=(),
        due_at: str | None = None,
    ) -> Obligation:
        self.authority.require(scope, repo)
        if revision(expected_revision, zero=True) != 0:
            raise Conflict()
        owner = identifier(owner if owner is not None else scope.principal_id)
        self._owner(scope, repo, owner)
        key = identifier(idempotency_key)
        summary = _text(summary)
        bindings, decisions = _refs(binding_refs), _refs(decision_refs)
        due = _timestamp(due_at)
        request_digest = digest(
            ["create", summary, expected_revision, owner, bindings, decisions, due]
        )
        with self.database.connect(Budget(2)) as db, self.database.transaction(db):
            record = self._receipt(db, scope, repo, key, request_digest)
            if record is None:
                now = datetime.now(timezone.utc).isoformat()
                record = Obligation(
                    str(uuid4()),
                    scope.namespace_id,
                    repo,
                    scope.principal_id,
                    owner,
                    1,
                    "open",
                    summary,
                    bindings,
                    decisions,
                    (),
                    now,
                    now,
                    due,
                )
                self._save(db, scope, record, key, request_digest, "created", None)
            self.authority.require(scope, repo)
        self.authority.require(scope, repo)
        return record

    def _get(self, db, scope, repo, work_id):
        try:
            UUID(identifier(work_id))
        except ValueError:
            raise WorkError() from None
        row = db.execute(
            "SELECT payload FROM work WHERE namespace=? AND repo=? AND id=?",
            (scope.namespace_id, repo, work_id),
        ).fetchone()
        if row is None:
            raise WorkError("work_not_found")
        return Obligation.from_json(row[0])

    def transition(
        self,
        scope: ScopeContext,
        repo: str,
        work_id: str,
        *,
        status: str,
        expected_revision: int,
        idempotency_key: str,
        reason: str,
        evidence_refs=(),
        reopen: bool = False,
    ) -> Obligation:
        self.authority.require(scope, repo)
        revision(expected_revision)
        key, reason = identifier(idempotency_key), _text(reason)
        refs = _refs(evidence_refs)
        if status not in STATUSES or type(reopen) is not bool:
            raise WorkError()
        request_digest = digest(
            ["transition", work_id, status, expected_revision, reason, refs, reopen]
        )
        with self.database.connect(Budget(2)) as db, self.database.transaction(db):
            current = self._get(db, scope, repo, work_id)
            self._owner(scope, repo, current.owner)
            record = self._receipt(db, scope, repo, key, request_digest)
            if record is None:
                if current.revision != expected_revision:
                    raise Conflict()
                if current.status == "cancelled" or status == current.status:
                    raise WorkError("invalid_transition")
                if current.status == "resolved":
                    if not reopen or status != "open":
                        raise WorkError("explicit_reopen_required")
                elif reopen or status == "open":
                    raise WorkError("invalid_transition")
                if status == "resolved" and (
                    not refs
                    or self.verify_evidence is None
                    or self.verify_evidence(scope, repo, refs, reason) is not True
                ):
                    raise WorkError("evidence_required")
                revision(current.revision + 1)
                record = replace(
                    current,
                    status=status,
                    revision=current.revision + 1,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    evidence_refs=refs,
                )
                self._save(
                    db, scope, record, key, request_digest, reason, current.status
                )
            self.authority.require(scope, repo)
        self.authority.require(scope, repo)
        return record

    def get(self, scope: ScopeContext, repo: str, work_id: str) -> Obligation:
        self.authority.require(scope, repo)
        with self.database.connect(Budget(2)) as db:
            result = self._get(db, scope, repo, work_id)
        self.authority.require(scope, repo)
        return result

    def list(
        self, scope: ScopeContext, repo: str, *, limit: int = 50, after: int = 0
    ) -> WorkPage:
        self.authority.require(scope, repo)
        revision(after, zero=True)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise WorkError()
        with self.database.connect(Budget(2)) as db:
            rows = db.execute(
                "SELECT seq,payload FROM work WHERE namespace=? AND repo=? AND seq>? ORDER BY seq LIMIT ?",
                (scope.namespace_id, repo, after, limit + 1),
            ).fetchall()
        self.authority.require(scope, repo)
        return WorkPage(
            tuple(Obligation.from_json(r["payload"]) for r in rows[:limit]),
            rows[limit - 1]["seq"] if len(rows) > limit else None,
        )

    def history(
        self,
        scope: ScopeContext,
        repo: str,
        work_id: str,
        *,
        after_revision: int = 0,
        limit: int = 50,
    ) -> tuple[dict, ...]:
        self.authority.require(scope, repo)
        revision(after_revision, zero=True)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise WorkError()
        with self.database.connect(Budget(2)) as db:
            self._get(db, scope, repo, work_id)
            rows = db.execute(
                "SELECT payload FROM history WHERE work_id=? AND revision>? ORDER BY revision LIMIT ?",
                (work_id, after_revision, limit),
            ).fetchall()
        self.authority.require(scope, repo)
        return tuple(json.loads(r[0]) for r in rows)

    def binding_locations(
        self, scope: ScopeContext, repo: str, work_id: str, bindings
    ) -> tuple[dict, ...]:
        """Read J05 BindingService locations; never mutate obligation state.

        Unknown/missing bindings use only this exact authorized repository. J05
        remains the authority for locating, moving and revalidating code.
        """
        record = self.get(scope, repo, work_id)
        result = []
        for ref in record.binding_refs:
            self.authority.require(scope, repo)
            location = bindings.get_current(scope, ref)
            if location is not None and location.repo_id != repo:
                raise ScopeDenied()
            if location is None or location.validation_state in (
                "unknown",
                "orphaned_confirmed",
            ):
                result.append({"binding_id": ref, "repo_id": repo, "grain": "repo"})
            else:
                result.append(
                    {
                        "binding_id": ref,
                        "repo_id": repo,
                        "grain": "symbol",
                        "path": location.path,
                        "symbol": location.qualified_name,
                    }
                )
        self.authority.require(scope, repo)
        return tuple(result)

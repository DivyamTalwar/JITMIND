"""Synchronous bounded lesson delivery for any authorized library client.

No code parser, graph, model, filesystem source read or provider call. Behavioral
adaptation of koragraph preflight/recall c9ce746; retained BSL notice and design
differences are documented in docs/open-work-preflight.md.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from threading import RLock
from uuid import uuid4

from jitmind.scope import ScopeContext, ScopeDenied

from .lesson_models import Lesson, LessonProjection, ProposedAction, safe_quote
from .work_storage import (
    Budget,
    Conflict,
    Deferred,
    WorkError,
    digest,
    encoded,
    finite,
    identifier,
    revision,
)


@dataclass(frozen=True)
class Session:
    id: str
    principal: str
    namespace: str
    repo: str
    authorization_version: int
    expires: float


@dataclass(frozen=True)
class PreflightPolicy:
    revision: int = 1
    token_budget: int = 600
    byte_cap: int = 2400
    deadline_seconds: float = 0.15
    minimum_confidence: float = 0.8
    allow_unvalidated: bool = False
    allow_hypotheses: bool = False

    def __post_init__(self):
        revision(self.revision)
        finite(self.deadline_seconds, 30)
        for value, maximum in ((self.token_budget, 10000), (self.byte_cap, 40000)):
            if type(value) is not int or not 64 <= value <= maximum:
                raise WorkError()
        if (
            type(self.minimum_confidence) not in (int, float)
            or not 0 <= self.minimum_confidence <= 1
            or type(self.allow_unvalidated) is not bool
            or type(self.allow_hypotheses) is not bool
        ):
            raise WorkError()


@dataclass(frozen=True)
class PreflightResult:
    state: str
    reason: str
    payload: str = ""
    lesson_ids: tuple[str, ...] = ()
    receipt_id: str | None = None
    replayed: bool = False
    payload_bytes: int = 0
    token_count: int = 0
    token_count_method: str = "utf8_byte_upper_bound"
    candidate_count: int = 0
    selection_complete: bool = True


class PreflightService:
    """Host-owned service. Pass issued scopes and sessions, never request claims.

    Session capabilities are process-local, expire within 24h, and are bounded to
    1024 live sessions. Durable dedup/receipt rows cascade at expiry/end; each
    session permits at most 256 receipts and 512 lesson deliveries. Sessions must
    be newly issued after restart. Work idempotency receipts survive indefinitely.

    Optional approved_repo(scope, repo, relative_path, budget) is a host registry
    adapter (e.g. J04); it must not wait beyond budget. It is not supplied by the
    request. Policy changes require a new immutable policy with a new revision.
    """

    def __init__(
        self,
        projection: LessonProjection,
        *,
        policy: PreflightPolicy | None = None,
        approved_repo=None,
    ):
        self.projection = projection
        self.database = projection.database
        self.authority = projection.authority
        self.policy = policy or PreflightPolicy()
        self.approved_repo = approved_repo
        self._sessions: dict[str, tuple[Session, tuple]] = {}
        self._lock = RLock()

    @staticmethod
    def _claims(session):
        return (
            session.id,
            session.principal,
            session.namespace,
            session.repo,
            session.authorization_version,
            session.expires,
        )

    def _require_session(self, scope, session, repo):
        self.authority.require(scope, repo)
        if not self._lock.acquire(blocking=False):
            raise Deferred()
        try:
            if type(session) is not Session:
                raise ScopeDenied()
            issued = self._sessions.get(session.id)
            if (
                issued is None
                or issued[0] is not session
                or issued[1] != self._claims(session)
                or session.principal != scope.principal_id
                or session.namespace != scope.namespace_id
                or session.repo != repo
                or session.authorization_version != scope.authorization_version
                or session.expires <= time.time()
            ):
                raise ScopeDenied()
        finally:
            self._lock.release()

    def start_session(
        self, scope: ScopeContext, repo: str, *, ttl_seconds: float = 3600
    ) -> Session:
        self.authority.require(scope, repo)
        finite(ttl_seconds, 86400)
        self.cleanup(scope)
        with self._lock:
            if len(self._sessions) >= 1024:
                raise WorkError("session_quota")
            session = Session(
                str(uuid4()),
                scope.principal_id,
                scope.namespace_id,
                repo,
                scope.authorization_version,
                time.time() + ttl_seconds,
            )
            with self.database.connect(Budget(2)) as db, self.database.transaction(db):
                if db.execute("SELECT count(*) FROM sessions").fetchone()[0] >= 1024:
                    raise WorkError("session_quota")
                db.execute(
                    "INSERT INTO sessions VALUES (?,?)", (session.id, session.expires)
                )
                self.authority.require(scope, repo)
            self._sessions[session.id] = (session, self._claims(session))
        self._require_session(scope, session, repo)
        return session

    def cleanup(self, scope: ScopeContext) -> int:
        """Bounded expiry maintenance; also called by start_session, not deliver."""
        self.authority.require(scope)
        now = time.time()
        with self.database.connect(Budget(2)) as db, self.database.transaction(db):
            count = db.execute("DELETE FROM sessions WHERE expires<=?", (now,)).rowcount
            self.authority.require(scope)
        with self._lock:
            self._sessions = {
                key: value
                for key, value in self._sessions.items()
                if value[0].expires > now
            }
        self.authority.require(scope)
        return count

    def end_session(self, scope: ScopeContext, session: Session) -> None:
        self._require_session(scope, session, session.repo)
        # Remove process capability first; DB cleanup may be retried via expiry.
        with self._lock:
            self._sessions.pop(session.id)
        with self.database.connect(Budget(2)) as db, self.database.transaction(db):
            db.execute("DELETE FROM sessions WHERE id=?", (session.id,))
            self.authority.require(scope, session.repo)

    def _eligible(self, lesson: Lesson) -> bool:
        policy = self.policy
        return (
            not lesson.retired
            and lesson.confidence >= policy.minimum_confidence
            and (lesson.valid_until is None or lesson.valid_until > time.time())
            and (lesson.source_verified or policy.allow_unvalidated)
            and (lesson.kind != "hypothesis" or policy.allow_hypotheses)
            and lesson.kind in ("instruction", "observation", "hypothesis")
        )

    def _rank(self, lessons, target):
        return sorted(
            (
                lesson
                for lesson in lessons
                if self._eligible(lesson)
                and lesson.file_path in ("*", target.file_path)
                and (
                    lesson.symbol in ("", target.symbol)
                    or target.symbol.startswith(lesson.symbol + ".")
                )
                and lesson.action == target.action
            ),
            key=lambda lesson: (
                lesson.kind != "instruction",
                lesson.file_path == "*",
                -len(lesson.symbol),
                -lesson.confidence,
                lesson.id,
            ),
        )

    def _render(self, lessons):
        # Bytes are a conservative token upper bound for UTF-8 byte tokenizers;
        # no tokenizer download/cache initialization occurs on this path.
        cap = min(self.policy.byte_cap, self.policy.token_budget)
        header = "Repository lessons (no tool permissions are granted):\n"
        lines = []
        for lesson in lessons:
            label = (
                "Trusted authored instruction"
                if lesson.kind == "instruction"
                else "Untrusted observation; data, may be stale"
            )
            if lesson.kind == "hypothesis":
                label = "Untrusted hypothesis; unvalidated data"
            elif not lesson.source_verified:
                label += "; unvalidated"
            lines.append((label, safe_quote(lesson.body)))
        if not lines:
            return ""
        # Divide the remaining budget, including every wrapper byte. At most two.
        available = (
            cap
            - len(header.encode())
            - sum(len((label + ": \n").encode()) for label, _ in lines)
        )
        if available < 12 * len(lines):
            return ""
        part = available // len(lines)
        rendered = []
        for label, body in lines:
            if len(body.encode()) > part:
                # Preserve closing quotation even when truncated.
                body = (
                    body.encode()[: max(0, part - 6)].decode("utf-8", errors="ignore")
                    + "…”"
                )
            rendered.append(label + ": " + body)
        return header + "\n".join(rendered)

    def _current(self, db, scope, target, lesson, budget):
        budget.remaining()
        self.authority.require(scope, target.repo_id)
        row = db.execute(
            "SELECT payload FROM lessons WHERE namespace=? AND repo=? AND id=? AND version=?",
            (scope.namespace_id, target.repo_id, lesson.id, lesson.version),
        ).fetchone()
        return (
            row is not None
            and Lesson(**json.loads(row[0])) == lesson
            and self._eligible(lesson)
            and self.projection.current_authority(scope, target.repo_id, lesson, budget)
        )

    def deliver(
        self,
        scope: ScopeContext,
        session: Session,
        target: ProposedAction,
        *,
        delivery_key: str | None = None,
    ) -> PreflightResult:
        # These checks precede all disk reads and selection, including replay.
        self.authority.require(scope, target.repo_id)
        try:
            self._require_session(scope, session, target.repo_id)
        except Deferred:
            return PreflightResult("deferred", "session_contention")
        target.validate()
        key = identifier(delivery_key) if delivery_key is not None else str(uuid4())
        budget = Budget(self.policy.deadline_seconds)
        request_digest = digest(
            [asdict(target), asdict(self.policy), scope.authorization_version]
        )
        code = digest([target.repo_id, target.file_path, target.symbol, target.action])
        try:
            if self.approved_repo is not None:
                if (
                    self.approved_repo(scope, target.repo_id, target.file_path, budget)
                    is not True
                ):
                    raise ScopeDenied()
                budget.remaining()
            lessons = self.projection.candidates(
                scope, target, budget, policy=self.policy
            )
            ranked = self._rank(lessons, target)
            # An empty ranked list is authoritative: no alternate/broader query.
            with self.database.connect(budget) as db, self.database.transaction(db):
                receipt = db.execute(
                    "SELECT digest,refs FROM delivery_receipts WHERE session=? AND key=?",
                    (session.id, key),
                ).fetchone()
                replayed = receipt is not None
                if receipt:
                    if receipt["digest"] != request_digest:
                        raise Conflict()
                    refs = json.loads(receipt["refs"])
                    by_ref = {(lesson.id, lesson.version): lesson for lesson in ranked}
                    chosen = [
                        by_ref[tuple(ref)] for ref in refs if tuple(ref) in by_ref
                    ]
                    if len(chosen) != len(refs):
                        return self._empty(
                            scope,
                            session,
                            target,
                            "deferred" if lessons.incomplete else "nothing_relevant",
                            "selection_incomplete"
                            if lessons.incomplete
                            else "receipt_no_longer_visible",
                            lessons,
                        )
                else:
                    chosen = []
                    for lesson in ranked:
                        if len(chosen) == 2:
                            break
                        seen = db.execute(
                            "SELECT 1 FROM deliveries WHERE session=? AND code=? AND lesson=? AND version=? AND policy=?",
                            (
                                session.id,
                                code,
                                lesson.id,
                                lesson.version,
                                self.policy.revision,
                            ),
                        ).fetchone()
                        if not seen and self._current(
                            db, scope, target, lesson, budget
                        ):
                            chosen.append(lesson)
                if not chosen:
                    return self._empty(
                        scope,
                        session,
                        target,
                        "deferred" if lessons.incomplete else "nothing_relevant",
                        "selection_incomplete"
                        if lessons.incomplete
                        else "no_eligible_undelivered_lessons",
                        lessons,
                    )
                payload = self._render(chosen)
                if not payload:
                    return self._empty(
                        scope,
                        session,
                        target,
                        "nothing_relevant",
                        "render_budget",
                        lessons,
                    )
                # Recheck every primary payload after selection and rendering.
                if not all(
                    self._current(db, scope, target, lesson, budget)
                    for lesson in chosen
                ):
                    return self._empty(
                        scope,
                        session,
                        target,
                        "nothing_relevant",
                        "authority_changed",
                        lessons,
                    )
                if not replayed:
                    if (
                        db.execute(
                            "SELECT count(*) FROM delivery_receipts WHERE session=?",
                            (session.id,),
                        ).fetchone()[0]
                        >= 256
                    ):
                        return self._empty(
                            scope, session, target, "deferred", "session_quota", lessons
                        )
                    for lesson in chosen:
                        db.execute(
                            "INSERT INTO deliveries VALUES (?,?,?,?,?)",
                            (
                                session.id,
                                code,
                                lesson.id,
                                lesson.version,
                                self.policy.revision,
                            ),
                        )
                    db.execute(
                        "INSERT INTO delivery_receipts VALUES (?,?,?,?)",
                        (
                            session.id,
                            key,
                            request_digest,
                            encoded([(lesson.id, lesson.version) for lesson in chosen]),
                        ),
                    )
                self._require_session(scope, session, target.repo_id)
                budget.remaining()
            # No cached body survives a forget between commit and final response.
            with self.database.connect(budget) as db:
                if not all(
                    self._current(db, scope, target, lesson, budget)
                    for lesson in chosen
                ):
                    return self._empty(
                        scope,
                        session,
                        target,
                        "nothing_relevant",
                        "authority_changed",
                        lessons,
                    )
            self._require_session(scope, session, target.repo_id)
            budget.remaining()
            count = len(payload.encode())
            return PreflightResult(
                "delivered",
                "receipt_replay" if replayed else "new_delivery",
                payload,
                tuple(lesson.id for lesson in chosen),
                key,
                replayed,
                count,
                count,
                candidate_count=lessons.examined,
                selection_complete=not lessons.incomplete,
            )
        except (ScopeDenied, Conflict):
            raise
        except Deferred:
            return self._empty(
                scope, session, target, "deferred", "deadline_or_contention"
            )
        except WorkError as exc:
            return self._empty(
                scope,
                session,
                target,
                "unavailable",
                exc.code
                if exc.code
                in ("invalid_temporal_metadata", "binding_authority_unavailable")
                else "authority_or_storage_unavailable",
            )
        except Exception as exc:  # noqa: BLE001 -- sanitize failures from host-supplied authority adapters
            # Do not log exception text: providers/SQL callbacks may embed secrets.
            from jitmind.storage.models import StorageBusy

            if isinstance(exc, StorageBusy):
                return self._empty(
                    scope, session, target, "deferred", "primary_contention"
                )
            return self._empty(
                scope,
                session,
                target,
                "unavailable",
                "authority_or_storage_unavailable",
            )

    def _empty(self, scope, session, target, state, reason, window=None):
        try:
            self._require_session(scope, session, target.repo_id)
        except Deferred:
            return PreflightResult("deferred", "session_contention")
        return PreflightResult(
            state,
            reason,
            candidate_count=window.examined if window is not None else 0,
            selection_complete=not window.incomplete
            if window is not None
            else state == "nothing_relevant",
        )

"""Bounded lesson projection and explicit host authority interfaces.

Native Python adaptation informed by koragraph practice, c9ce746. See
docs/open-work-preflight.md for attribution and retained BSL notice.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from jitmind.scope import ScopeAuthority, ScopeContext, ScopeDenied

from .work_storage import (
    Budget,
    Conflict,
    Deferred,
    WorkDatabase,
    WorkError,
    encoded,
    identifier,
    is_transient_sqlite_error,
    revision,
)


def relative_path(value: str) -> str:
    identifier(value)
    # No decoding or normalization: reject escape spellings, URI/drive paths,
    # and dot components before any registry or filesystem access.
    if (
        value.startswith("/")
        or "\\" in value
        or ":" in value
        or "%" in value
        or any(p in ("", ".", "..") for p in value.split("/"))
    ):
        raise WorkError("invalid_path")
    return value


@dataclass(frozen=True)
class ProposedAction:
    repo_id: str
    file_path: str
    symbol: str = ""
    action: str = "edit"

    def validate(self):
        identifier(self.repo_id)
        relative_path(self.file_path)
        if self.symbol:
            identifier(self.symbol)
            if any(c in self.symbol for c in ("/", "\\", "%")) or ".." in self.symbol:
                raise WorkError("invalid_symbol")
        if type(self.symbol) is not str or self.action not in (
            "edit",
            "write",
            "read",
            "test",
        ):
            raise WorkError()
        return self


class FactAuthority(Protocol):
    """Host-configured active-fact check, invoked for every output body.

    Implementations must consult current primary authority, never a projection,
    and respect budget (nonwaiting I/O); false means absent/forgotten or outside
    backing fact/page validity and TTL. Malformed temporal metadata must fail.
    An optional is_present check distinguishes administrative absence for forget.
    Callback
    provenance is established by host configuration, not this type annotation.
    """

    def is_active(self, scope: ScopeContext, fact_id: str, budget: Budget) -> bool: ...


class DurableFactAuthority:
    """Calls actual J02 SQLiteDurableStore.get_entry, using an isolated view.

    The view shares its authoritative DB path but no connection or mutable timeout
    with ingest. No constructor/schema migration occurs in the hotpath.
    """

    def __init__(self, store, authority: ScopeAuthority, *, clock=time.time):
        from jitmind.storage import SQLiteDurableStore

        if type(store) is not SQLiteDurableStore:
            raise WorkError("invalid_fact_authority")
        self.store = copy.copy(store)
        self.store.busy_timeout_ms = 0
        self.store.commit_retries = 0
        self.authority = authority
        self.clock = clock

    def is_present(self, scope, fact_id, budget):
        """Administrative presence only, for accepted-forget cleanup."""
        self.authority.require(scope)
        budget.remaining()
        result = self.store.get_entry(scope.namespace_id, fact_id)
        budget.remaining()
        self.authority.require(scope)
        return result is not None

    def is_active(self, scope, fact_id, budget):
        self.authority.require(scope)
        budget.remaining()
        entry = self.store.get_entry(scope.namespace_id, fact_id)
        if entry is None:
            return False
        page = self.store.get_page(scope.namespace_id, entry.source_page_id)
        budget.remaining()
        self.authority.require(scope)
        if page is None or entry.status != "active":
            return False
        now = self.clock()
        # Validate all metadata before deciding applicability; malformed optional
        # values must not disappear behind an earlier false interval predicate.
        checks = [
            temporal_applicable(entry.model_dump(), now),
            temporal_applicable(entry.meta, now, created=entry.t_created),
            temporal_applicable(page.meta, now, created=entry.t_created),
        ]
        return all(checks)


def temporal_applicable(meta, now, *, created=None):
    """Half-open validity/TTL intervals. No naive timestamps or coercion."""

    def timestamp(value):
        if type(value) is not str or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})",
            value,
        ):
            raise WorkError("invalid_temporal_metadata")
        try:
            # datetime accepts offsets such as +01:99 by normalizing them.
            if value[-1] != "Z" and (int(value[-5:-3]) > 23 or int(value[-2:]) > 59):
                raise ValueError
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except (ValueError, OverflowError):
            raise WorkError("invalid_temporal_metadata") from None

    times = {
        key: timestamp(meta[key])
        for key in (
            "t_created",
            "created_at",
            "t_observed",
            "t_valid",
            "t_invalid",
            "t_expired",
            "expires_at",
            "last_accessed",
        )
        if meta.get(key) is not None
    }
    start, end = times.get("t_valid"), times.get("t_invalid")
    if start is not None and end is not None and start > end:
        raise WorkError("invalid_temporal_metadata")
    ends = [times[k] for k in ("t_invalid", "t_expired", "expires_at") if k in times]
    if "ttl_seconds" in meta:
        ttl = meta["ttl_seconds"]
        if type(ttl) not in (int, float) or not math.isfinite(ttl) or ttl < 0:
            raise WorkError("invalid_temporal_metadata")
        origin = times.get("t_created", times.get("created_at"))
        if origin is None:
            origin = timestamp(created)
        ends.append(origin + ttl)
    return (start is None or start <= now) and all(now < end for end in ends)


class DurableBindingAuthority:
    """Read only J05 schema-v2 heads, without source access or revalidation.

    This adapter deliberately checks recorded authority, not live filesystem
    freshness. A future J05 schema requires an adapter update, never fallback.
    """

    def __init__(self, bindings, authority):
        self.path = Path(bindings.store.path).absolute()
        self.authority = authority

    def is_current(self, scope, repo, lesson, budget):
        self.authority.require(scope, repo)
        budget.remaining()
        db = None
        try:
            db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=0)
            db.set_progress_handler(lambda: int(time.monotonic() >= budget.end), 100)
            if db.execute("PRAGMA user_version").fetchone()[0] != 2:
                raise WorkError("binding_authority_unavailable")
            row = db.execute(
                """SELECT h.payload, f.terminal, s.snapshot, s.generation
                FROM heads d JOIN history h USING(ns,logical,revision)
                LEFT JOIN facts f ON f.ns=h.ns AND f.fact=json_extract(h.payload,'$.fact_version_id')
                LEFT JOIN sources s ON s.ns=h.ns AND s.repo=h.repo
                WHERE d.ns=? AND d.logical=? AND h.repo=?""",
                (scope.namespace_id, lesson.binding_id, repo),
            ).fetchone()
            budget.remaining()
            self.authority.require(scope, repo)
            if row is None:
                return False
            from .binding_models import CodeBinding

            binding = CodeBinding.model_validate_json(row[0])
            return (
                not row[1]
                and binding.namespace_id == scope.namespace_id
                and binding.repo_id == repo
                and binding.revision == lesson.binding_revision
                and binding.fact_version_id == lesson.fact_id
                and binding.validation_state == "verified_current"
                and not binding.review_required
                and binding.snapshot_id == row[2]
                and binding.source_generation == row[3]
            )
        except sqlite3.Error as exc:
            if is_transient_sqlite_error(exc):
                raise Deferred() from None
            raise WorkError("binding_authority_unavailable") from None
        finally:
            if db is not None:
                db.close()


@dataclass(frozen=True)
class Lesson:
    id: str
    fact_id: str
    version: int
    source_revision: int
    file_path: str
    symbol: str
    action: str
    body: str
    kind: str
    source_verified: bool
    confidence: float
    valid_until: float | None
    author: str | None = None
    binding_id: str | None = None
    binding_revision: int | None = None
    retired: bool = False


class CandidateWindow(list):
    """List-compatible bounded selection with explicit completeness evidence."""

    def __init__(self, lessons, examined, incomplete):
        super().__init__(lessons)
        self.examined = examined
        self.incomplete = incomplete


class LessonProjection:
    """Host ingestion API; never expose author_instruction as generic ingest.

    import_observation cannot create/overwrite an instruction, even with tier law.
    Verification/confidence fields must be assigned by a trusted validation job,
    not accepted directly from model or imported metadata. Preflight is read-only
    with respect to lessons. Body size is bounded at ingestion.
    """

    def __init__(
        self,
        database: WorkDatabase,
        authority: ScopeAuthority,
        facts: FactAuthority,
        *,
        can_author=None,
        can_retire=None,
        binding_authority=None,
    ):
        self.database = database
        self.authority = authority
        self.facts = facts
        self.can_author = can_author
        self.can_retire = can_retire
        self.binding_authority = binding_authority

    def import_observation(
        self,
        scope: ScopeContext,
        target: ProposedAction,
        *,
        lesson_id: str,
        fact_id: str,
        version: int,
        source_revision: int,
        body: str,
        tier: str = "observation",
        source_verified: bool = False,
        confidence: float = 0,
        valid_until: float | None = None,
        binding_id: str | None = None,
        binding_revision: int | None = None,
    ) -> bool:
        self.authority.require(scope, target.repo_id)
        if tier not in ("observation", "law", "hypothesis"):
            raise WorkError()
        return self._upsert(
            scope,
            target,
            Lesson(
                lesson_id,
                fact_id,
                version,
                source_revision,
                target.file_path,
                target.symbol,
                target.action,
                body,
                "hypothesis" if tier == "hypothesis" else "observation",
                source_verified,
                confidence,
                valid_until,
                binding_id=binding_id,
                binding_revision=binding_revision,
            ),
        )

    def author_instruction(
        self,
        scope: ScopeContext,
        target: ProposedAction,
        *,
        lesson_id: str,
        fact_id: str,
        version: int,
        source_revision: int,
        body: str,
        binding_id: str | None = None,
        binding_revision: int | None = None,
    ) -> bool:
        self.authority.require(scope, target.repo_id)
        if (
            self.can_author is None
            or self.can_author(scope, target.repo_id) is not True
        ):
            raise ScopeDenied()
        return self._upsert(
            scope,
            target,
            Lesson(
                lesson_id,
                fact_id,
                version,
                source_revision,
                target.file_path,
                target.symbol,
                target.action,
                body,
                "instruction",
                True,
                1,
                None,
                scope.principal_id,
                binding_id,
                binding_revision,
            ),
        )

    def project_binding_observation(
        self,
        scope: ScopeContext,
        repo: str,
        bindings,
        *,
        logical_binding_id: str,
        lesson_id: str,
        version: int,
        source_revision: int,
        body: str,
        confidence: float = 0.9,
    ) -> bool:
        """Host background projection from actual J05 CodeMemoryService.

        Source freshness/revalidation belongs to J05 and runs outside preflight.
        Unknown, changed and orphaned bindings cannot become verified lessons.
        The host republishes with a higher material version when J05 changes.
        """
        self.authority.require(scope, repo)
        identifier(logical_binding_id)
        binding = bindings.get_current(scope, logical_binding_id)
        if binding is None:
            raise WorkError("binding_unavailable")
        if binding.repo_id != repo or binding.namespace_id != scope.namespace_id:
            raise ScopeDenied()
        self.authority.require(scope, repo)
        return self.import_observation(
            scope,
            ProposedAction(repo, binding.path, binding.qualified_name),
            lesson_id=lesson_id,
            fact_id=binding.fact_version_id,
            version=version,
            source_revision=source_revision,
            body=body,
            confidence=confidence,
            source_verified=binding.validation_state == "verified_current"
            and not binding.review_required,
            binding_id=logical_binding_id,
            binding_revision=binding.revision,
        )

    def _upsert(self, scope, target, lesson):
        target.validate()
        identifier(lesson.id)
        identifier(lesson.fact_id)
        revision(lesson.version)
        revision(lesson.source_revision)
        if lesson.binding_id is not None:
            identifier(lesson.binding_id)
            revision(lesson.binding_revision)
        elif lesson.binding_revision is not None:
            raise WorkError("invalid_binding_metadata")
        if target.file_path == "*" and target.symbol:
            raise WorkError("invalid_repo_scope")
        if (
            type(lesson.body) is not str
            or not lesson.body.strip()
            or len(lesson.body) > 4096
            or type(lesson.source_verified) is not bool
            or type(lesson.confidence) not in (int, float)
            or not math.isfinite(lesson.confidence)
            or not 0 <= lesson.confidence <= 1
            or (
                lesson.valid_until is not None
                and (
                    type(lesson.valid_until) not in (int, float)
                    or not math.isfinite(lesson.valid_until)
                )
            )
        ):
            raise WorkError()
        try:
            if len(lesson.body.encode()) > 8192:
                raise WorkError()
        except UnicodeError:
            raise WorkError() from None
        budget = Budget(2)
        if self.facts.is_active(scope, lesson.fact_id, budget) is not True:
            self.authority.require(scope, target.repo_id)
            return False
        raw = encoded(asdict(lesson))
        with self.database.connect(budget) as db, self.database.transaction(db):
            tomb = db.execute(
                "SELECT revision FROM tombstones WHERE namespace=? AND fact_id=?",
                (scope.namespace_id, lesson.fact_id),
            ).fetchone()
            if tomb and tomb[0] >= lesson.source_revision:
                self.authority.require(scope, target.repo_id)
                return False
            old = db.execute(
                "SELECT version,kind,payload FROM lessons WHERE namespace=? AND repo=? AND id=?",
                (scope.namespace_id, target.repo_id, lesson.id),
            ).fetchone()
            if old:
                if json.loads(old["payload"]).get("retired", False):
                    raise Conflict()
                if old["kind"] != lesson.kind:
                    raise Conflict()
                if old["version"] > lesson.version:
                    self.authority.require(scope, target.repo_id)
                    return False
                if old["version"] == lesson.version:
                    if Lesson(**json.loads(old["payload"])) != lesson:
                        raise Conflict()
                    self.authority.require(scope, target.repo_id)
                    return True
            db.execute(
                "INSERT INTO lessons VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(namespace,repo,id) DO UPDATE SET fact_id=excluded.fact_id,version=excluded.version,file=excluded.file,symbol=excluded.symbol,action=excluded.action,kind=excluded.kind,payload=excluded.payload",
                (
                    scope.namespace_id,
                    target.repo_id,
                    lesson.id,
                    lesson.fact_id,
                    lesson.version,
                    lesson.file_path,
                    lesson.symbol,
                    lesson.action,
                    lesson.kind,
                    raw,
                ),
            )
            self.authority.require(scope, target.repo_id)
        self.authority.require(scope, target.repo_id)
        return True

    def forget(
        self, scope: ScopeContext, fact_id: str, *, tombstone_revision: int
    ) -> dict:
        """Project an already accepted primary forget. Not a primary forget API.

        Body visibility does not wait for this cleanup: every delivery checks the
        current fact authority. Tombstone IDs/revisions prevent old event replay.
        """
        self.authority.require(scope)
        identifier(fact_id)
        revision(tombstone_revision)
        budget = Budget(2)
        if getattr(self.facts, "is_present", self.facts.is_active)(
            scope, fact_id, budget
        ):
            raise WorkError("primary_forget_required")
        from .work_storage import Deferred

        try:
            with self.database.connect(budget) as db, self.database.transaction(db):
                db.execute(
                    "INSERT INTO tombstones VALUES (?,?,?) ON CONFLICT(namespace,fact_id) DO UPDATE SET revision=max(revision,excluded.revision)",
                    (scope.namespace_id, fact_id, tombstone_revision),
                )
                db.execute(
                    "DELETE FROM lessons WHERE namespace=? AND fact_id=? AND coalesce(json_extract(payload,'$.retired'),0)=0",
                    (scope.namespace_id, fact_id),
                )
                self.authority.require(scope)
            state = "projection_purged"
        except Deferred:
            state = "projection_purge_deferred"
        self.authority.require(scope)
        return {
            "visibility": "hidden_by_primary_authority",
            "purge": state,
            "retention": "historical_ids_and_backups_may_remain",
        }

    def retire_instruction(self, scope, repo, lesson_id, *, expected_version):
        """Explicit host repo-owner retirement; author departure is not completion."""
        self.authority.require(scope, repo)
        identifier(lesson_id)
        revision(expected_version)
        if self.can_retire is None or self.can_retire(scope, repo) is not True:
            raise ScopeDenied()
        with self.database.connect(Budget(2)) as db, self.database.transaction(db):
            row = db.execute(
                "SELECT payload FROM lessons WHERE namespace=? AND repo=? AND id=?",
                (scope.namespace_id, repo, lesson_id),
            ).fetchone()
            if row is None:
                raise Conflict()
            data = json.loads(row[0])
            if data["kind"] != "instruction" or data["version"] != expected_version:
                raise Conflict()
            data.update(retired=True, body="", version=revision(expected_version + 1))
            db.execute(
                "UPDATE lessons SET version=?,payload=? WHERE namespace=? AND repo=? AND id=?",
                (data["version"], encoded(data), scope.namespace_id, repo, lesson_id),
            )
            self.authority.require(scope, repo)
        self.authority.require(scope, repo)

    def current_authority(self, scope, repo, lesson, budget):
        if self.facts.is_active(scope, lesson.fact_id, budget) is not True:
            return False
        if lesson.binding_id is not None:
            if self.binding_authority is None:
                raise WorkError("binding_authority_unavailable")
            result = self.binding_authority.is_current(scope, repo, lesson, budget)
            budget.remaining()
            if type(result) is not bool:
                raise WorkError("binding_authority_unavailable")
            return result
        return True

    def candidates(self, scope, target, budget, *, policy=None) -> list[Lesson]:
        from .preflight import PreflightPolicy

        policy = policy or PreflightPolicy()
        self.authority.require(scope, target.repo_id)
        target.validate()
        with self.database.connect(budget) as db:
            # Fetch only IDs for the extra sentinel; hydrate/check <=20 bodies.
            rows = db.execute(
                """SELECT id FROM lessons WHERE namespace=? AND repo=?
                AND file IN (?, '*') AND action=?
                AND (symbol='' OR symbol=? OR substr(?,1,length(symbol)+1)=symbol||'.')
                AND kind IN ('instruction','observation','hypothesis')
                AND (kind!='hypothesis' OR ?)
                AND json_extract(payload,'$.confidence')>=?
                AND (json_extract(payload,'$.source_verified')=1 OR ?)
                AND coalesce(json_extract(payload,'$.retired'),0)=0
                AND (json_extract(payload,'$.valid_until') IS NULL OR json_extract(payload,'$.valid_until')>?)
                ORDER BY (kind='instruction') DESC,(file!='*') DESC,length(symbol) DESC,
                json_extract(payload,'$.confidence') DESC,id LIMIT 21""",
                (
                    scope.namespace_id,
                    target.repo_id,
                    target.file_path,
                    target.action,
                    target.symbol,
                    target.symbol,
                    policy.allow_hypotheses,
                    policy.minimum_confidence,
                    policy.allow_unvalidated,
                    time.time(),
                ),
            ).fetchall()
            payloads = [
                db.execute(
                    "SELECT payload FROM lessons WHERE namespace=? AND repo=? AND id=?",
                    (scope.namespace_id, target.repo_id, r[0]),
                ).fetchone()[0]
                for r in rows[:20]
            ]
        lessons = [Lesson(**json.loads(raw)) for raw in payloads]
        lessons = [
            lesson
            for lesson in lessons
            if self.current_authority(scope, target.repo_id, lesson, budget)
        ]
        self.authority.require(scope, target.repo_id)
        return CandidateWindow(lessons, len(payloads), len(rows) > 20)


def safe_quote(text: str) -> str:
    """Limited redaction and structural quoting, never a permission mechanism.

    Fixed bounded patterns operate only on the bounded projection body. This is
    not a universal secret detector; opaque evidence is deliberately not rendered.
    """
    text = text[:4096]
    text = re.sub(
        r"(?i)\b(?:sk-[\w-]{12,}|gh[pousr]_[\w]{12,}|github_pat_[\w]{12,}|AKIA[A-Z0-9]{16}|xox[baprs]-[\w-]{10,})",
        "[redacted]",
        text,
    )
    text = re.sub(
        r"(?i)\b[\w.-]*(?:secret|password|passwd|api[_-]?key|token|credential|private[_-]?key)[\w.-]*[\"']?\s*[:=]\s*[\"']?[^\s,;\"']+",
        "[redacted]",
        text,
    )
    text = re.sub(r"(?i)\b(?:bearer|basic)\s+\S+", "[redacted]", text)
    text = re.sub(
        r"(?i)([a-z][a-z0-9+.-]*://)[^\s/@]+:[^\s/@]+@", r"\1[redacted]@", text
    )
    text = " ".join(
        "".join(
            c if 32 <= ord(c) < 127 or (ord(c) >= 160 and c.isprintable()) else " "
            for c in text
        ).split()
    )
    text = text.replace("<", "‹").replace(">", "›").replace("`", "'")
    return "“" + text.replace("“", '"').replace("”", '"') + "”"

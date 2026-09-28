"""Persistent code bindings composed with JITMIND scope, durable facts and snapshots."""

from __future__ import annotations

import json
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from jitmind.code_context.models import CodeQuery, safe_path
from jitmind.scope import ScopeAuthority, ScopeContext

from .binding_models import (
    BindingConflict,
    BindingUnavailable,
    CodeBinding,
    identifier,
    revision,
)
from .binding_storage import BindingStore
from .revalidation import candidate, decide, digest


@dataclass(frozen=True)
class SourceView:
    snapshot_id: str
    generation: int
    complete: bool
    candidates: tuple
    lineage: str | None = None


class CodeContextSnapshotAdapter:
    """Use only public CodeContext queries; never read checkout spans independently.

    Public result/context budgets bound enumeration. Exceeding them is unknown,
    never a claim that unreturned declarations are absent.
    """

    def __init__(self, context):
        self.context = context

    def lineage(self, scope, repo):
        """Fixed, bounded HEAD metadata read; no config, refs, commands or code crawl.

        J04 content snapshots do not include branch identity yet. Non-Git roots
        have a distinct lineage. Git worktree pointer files/symlinks fail closed.
        """
        self.context.authority.require(scope, repo)
        registered = self.context.registry.get(repo, scope.namespace_id)
        descriptors = []
        try:
            fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
            descriptors.append(fd)
            for component in registered.root.split("/")[1:]:
                fd = os.open(
                    component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
                descriptors.append(fd)
            root_stat = os.fstat(fd)
            if (root_stat.st_dev, root_stat.st_ino) != registered.identity:
                return None
            try:
                git = os.open(
                    ".git", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
            except FileNotFoundError:
                return digest("unversioned-root")
            descriptors.append(git)
            head = os.open(
                "HEAD", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=git
            )
            descriptors.append(head)
            info = os.fstat(head)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024:
                return None
            raw = os.read(head, 1025)
            if len(raw) > 1024:
                return None
            text = raw.decode("ascii").strip()
            if not re.fullmatch(
                r"(?:ref: refs/heads/[^\s\x00-\x1f]{1,200}|[a-f0-9]{40}|[a-f0-9]{64})",
                text,
            ):
                return None
            return digest(text)
        except (OSError, UnicodeError):
            return None
        finally:
            for fd in reversed(descriptors):
                os.close(fd)
            self.context.authority.require(scope, repo)

    def freshness(self, scope, repo, snapshot):
        return self.context.query(
            scope,
            CodeQuery(operation="check_freshness", repo_id=repo, snapshot_id=snapshot),
        )

    @staticmethod
    def adequate(page):
        return (
            page.status == "ok"
            and page.freshness.get("state") == "fresh"
            and page.coverage.get("state") == "complete_subset"
            and not page.truncated
        )

    def load(self, scope, repo, snapshot):
        lineage = self.lineage(scope, repo)
        page = self.context.query(
            scope,
            CodeQuery(
                operation="repo_map",
                repo_id=repo,
                snapshot_id=snapshot,
                limit=50,
                max_context_tokens=16000,
            ),
        )
        generation = page.generation or 1
        if lineage is None or not self.adequate(page):
            return SourceView(snapshot, generation, False, ())
        # J04 currently starts declaration spans AFTER decorators. Until it exposes
        # complete decorated spans, any @ syntax makes this subset insufficient.
        decorators = self.context.query(
            scope,
            CodeQuery(
                operation="find_all",
                repo_id=repo,
                snapshot_id=snapshot,
                query="@",
                limit=1,
                max_context_tokens=16000,
            ),
        )
        if (
            not self.adequate(decorators)
            or decorators.generation != generation
            or decorators.results
        ):
            return SourceView(snapshot, generation, False, ())
        candidates = []
        for file in page.results:
            symbols = self.context.query(
                scope,
                CodeQuery(
                    operation="file_api",
                    repo_id=repo,
                    snapshot_id=snapshot,
                    query=file.path,
                    limit=50,
                    max_context_tokens=16000,
                ),
            )
            if not self.adequate(symbols) or symbols.generation != generation:
                return SourceView(snapshot, generation, False, ())
            for symbol in symbols.results:
                if (
                    symbol.namespace_id != scope.namespace_id
                    or symbol.repo_id != repo
                    or symbol.snapshot_id != snapshot
                ):
                    return SourceView(snapshot, generation, False, ())
                # Public spans retain every enclosing declaration in an adequate page.
                parents = sorted(
                    (
                        p
                        for p in symbols.results
                        if p.start_line < symbol.start_line
                        and p.end_line >= symbol.end_line
                    ),
                    key=lambda p: p.start_line,
                )
                qualified = ".".join([p.name for p in parents] + [symbol.name])
                try:
                    candidates.append(candidate(symbol, qualified))
                except (ValueError, SyntaxError, RecursionError):
                    return SourceView(snapshot, generation, False, ())
        check = self.freshness(scope, repo, snapshot)
        complete = (
            self.adequate(check)
            and check.generation == generation
            and self.lineage(scope, repo) == lineage
        )
        return SourceView(
            snapshot,
            generation,
            complete,
            tuple(
                sorted(
                    candidates, key=lambda c: (c.evidence.path, c.evidence.start_line)
                )
            ),
            lineage,
        )

    @contextmanager
    def publication(self, scope, repo):
        # J04 has no public generation lease yet. Keep this one synchronization
        # dependency inside the adapter: builds publish under this same RLock.
        # Freshness/fact work runs before the SQLite transaction, while the
        # generation cannot change between its final check and the CAS write.
        self.context.authority.require(scope, repo)
        with self.context._lock:
            yield
        self.context.authority.require(scope, repo)

    def still_current(self, scope, repo, view):
        check = self.freshness(scope, repo, view.snapshot_id)
        return (
            self.adequate(check)
            and check.generation == view.generation
            and self.lineage(scope, repo) == view.lineage
        )


class CodeMemoryService:
    """Trusted host supplies the SAME authority and namespace-keyed durable store.

    The database is lazy: construction performs no IO. No method returns fact or
    raw source text, including history. Historical metadata can survive forgetting.
    """

    def __init__(self, authority: ScopeAuthority, facts, code_context, binding_path):
        self.authority = authority
        self.facts = facts
        self.sources = CodeContextSnapshotAdapter(code_context)
        self.store = BindingStore(binding_path)

    def _fact(self, scope, fact, *, historical=False):
        self.authority.require(scope)
        entry = self.facts.get_entry(
            scope.namespace_id, fact, include_inactive=historical
        )
        if entry is None or entry.id != fact:
            return None
        if entry.status != "active" and not (
            historical and entry.status == "superseded"
        ):
            return None
        if self.store.is_terminal(scope.namespace_id, fact):
            return None
        self.authority.require(scope)
        return entry

    def _return(self, scope, repo, result):
        self.authority.require(scope, repo)
        return result

    @staticmethod
    def _request(operation, **kwargs):
        return digest(
            json.dumps([operation, kwargs], sort_keys=True, separators=(",", ":"))
        )

    def _read(self, scope, logical):
        self.authority.require(scope)
        identifier(logical)
        binding = self.store.current(
            scope.namespace_id, logical, repo_ids=scope.authorized_repo_ids
        )
        self.authority.require(scope)
        if binding is not None:
            self.authority.require(scope, binding.repo_id)
        return binding

    def create_binding(
        self,
        scope: ScopeContext,
        *,
        logical_binding_id: str,
        fact_version_id: str,
        repo_id: str,
        snapshot_id: str,
        path: str,
        qualified_name: str,
    ) -> CodeBinding:
        self.authority.require(scope, repo_id)
        for key in (logical_binding_id, fact_version_id):
            identifier(key)
        safe_path(path)
        if (
            not isinstance(qualified_name, str)
            or not 1 <= len(qualified_name) <= 4096
            or "\x00" in qualified_name
        ):
            raise ValueError("Invalid qualified name")
        CodeQuery(operation="check_freshness", repo_id=repo_id, snapshot_id=snapshot_id)
        if self._fact(scope, fact_version_id) is None:
            raise BindingUnavailable()
        request = self._request(
            "create",
            fact=fact_version_id,
            repo=repo_id,
            snapshot=snapshot_id,
            path=path,
            name=qualified_name,
        )
        replay = self.store.replay(
            scope.namespace_id,
            logical_binding_id,
            scope.request_id,
            request,
            repo_ids=(repo_id,),
            snapshot_id=snapshot_id,
        )
        if replay:
            return self._return(scope, repo_id, self._current_view(scope, replay))
        source_revision = self.store.source_revision(scope.namespace_id, repo_id)
        view = self.sources.load(scope, repo_id, snapshot_id)
        hits = [
            c
            for c in view.candidates
            if c.evidence.path == path and c.evidence.qualified_name == qualified_name
        ]
        if not view.complete or len(hits) != 1:
            raise BindingUnavailable()
        hit = hits[0]
        data = {
            "logical_binding_id": logical_binding_id,
            "namespace_id": scope.namespace_id,
            "fact_version_id": fact_version_id,
            "repo_id": repo_id,
            "snapshot_id": snapshot_id,
            "source_generation": view.generation,
            "source_lineage": view.lineage,
            "revision": 1,
            "path": path,
            "qualified_name": qualified_name,
            "kind": hit.kind,
            "signature_digest": hit.signature_digest,
            "raw_source_digest": hit.raw_source_digest,
            "body_digest": hit.body_digest,
            "start_line": hit.evidence.start_line,
            "end_line": hit.evidence.end_line,
            "validation_state": "verified_current",
            "reason_code": "created",
            "evidence_refs": (hit.evidence,),
            "candidate_count": 1,
        }
        binding = self._record(data)
        with self.sources.publication(scope, repo_id):
            self._publish_guard(scope, binding, view)
            self.store.append(binding, 0, source_revision, scope.request_id, request)
        return self._return(scope, repo_id, self._current_view(scope, binding))

    @staticmethod
    def _record(data):
        data = dict(data)
        data.pop("binding_id", None)
        data["validated_at"] = datetime.now(timezone.utc).isoformat()
        identity = [
            data["namespace_id"],
            data["logical_binding_id"],
            data["revision"],
            data["snapshot_id"],
            data["source_generation"],
        ]
        data["binding_id"] = digest(json.dumps(identity, separators=(",", ":")))
        return CodeBinding(**data)

    def _publish_guard(self, scope, binding, view):
        if self._fact(scope, binding.fact_version_id) is None:
            raise BindingUnavailable()
        if view.complete and not self.sources.still_current(
            scope, binding.repo_id, view
        ):
            raise BindingConflict()
        self.authority.require(scope, binding.repo_id)

    def revalidate(
        self,
        scope: ScopeContext,
        logical_binding_id: str,
        *,
        snapshot_id: str,
        expected_revision: int,
    ) -> CodeBinding:
        self.authority.require(scope)
        revision(expected_revision)
        old = self._read(scope, logical_binding_id)
        if old is None:
            raise BindingUnavailable()
        CodeQuery(
            operation="check_freshness", repo_id=old.repo_id, snapshot_id=snapshot_id
        )
        if self._fact(scope, old.fact_version_id) is None:
            raise BindingUnavailable()
        request = self._request(
            "revalidate", snapshot=snapshot_id, expected=expected_revision
        )
        replay = self.store.replay(
            scope.namespace_id,
            logical_binding_id,
            scope.request_id,
            request,
            repo_ids=(old.repo_id,),
            snapshot_id=snapshot_id,
        )
        if replay:
            return self._return(scope, old.repo_id, self._current_view(scope, replay))
        if old.revision != expected_revision:
            raise BindingConflict()
        source_revision = self.store.source_revision(scope.namespace_id, old.repo_id)
        view = self.sources.load(scope, old.repo_id, snapshot_id)
        state, reason, hit, evidence = decide(old, view.candidates, view.complete)
        if view.complete and view.lineage != old.source_lineage:
            state, reason, hit, evidence = "unknown", "snapshot_mismatch", None, ()
        data = old.model_dump()
        data.update(
            snapshot_id=snapshot_id,
            source_generation=view.generation,
            revision=old.revision + 1,
            supersedes_binding_id=old.binding_id,
            validation_state=state,
            reason_code=reason,
            review_required=old.review_required or state == "changed",
            evidence_refs=tuple(c.evidence for c in evidence[:8]),
            candidate_count=len(evidence),
        )
        if hit:
            data.update(
                path=hit.evidence.path,
                qualified_name=hit.evidence.qualified_name,
                start_line=hit.evidence.start_line,
                end_line=hit.evidence.end_line,
                raw_source_digest=hit.raw_source_digest,
                body_digest=hit.body_digest,
                signature_digest=hit.signature_digest,
            )
        binding = self._record(data)
        with self.sources.publication(scope, old.repo_id):
            self._publish_guard(scope, binding, view)
            self.store.append(
                binding, expected_revision, source_revision, scope.request_id, request
            )
        return self._return(scope, old.repo_id, self._current_view(scope, binding))

    def _current_view(self, scope, binding):
        head = self._read(scope, binding.logical_binding_id)
        if head is None or head.binding_id != binding.binding_id:
            return binding.model_copy(
                update={
                    "validation_state": "unknown",
                    "reason_code": "snapshot_mismatch",
                }
            )
        if self._fact(scope, binding.fact_version_id) is None:
            return binding.model_copy(
                update={
                    "validation_state": "unknown",
                    "reason_code": "fact_unavailable",
                }
            )
        if binding.validation_state == "verified_current":
            check = self.sources.freshness(scope, binding.repo_id, binding.snapshot_id)
            if (
                not self.sources.adequate(check)
                or check.generation != binding.source_generation
                or self.sources.lineage(scope, binding.repo_id)
                != binding.source_lineage
            ):
                return binding.model_copy(
                    update={
                        "validation_state": "unknown",
                        "reason_code": "snapshot_mismatch",
                    }
                )
        return binding

    def get_current(
        self, scope: ScopeContext, logical_binding_id: str
    ) -> CodeBinding | None:
        binding = self._read(scope, logical_binding_id)
        if binding is None:
            return self._return(scope, None, None)
        return self._return(scope, binding.repo_id, self._current_view(scope, binding))

    def history(
        self,
        scope: ScopeContext,
        logical_binding_id: str,
        *,
        snapshot_id: str | None = None,
        after_revision: int = 0,
        limit: int = 100,
    ) -> tuple[CodeBinding, ...]:
        self.authority.require(scope)
        identifier(logical_binding_id)
        revision(after_revision, zero=True)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Invalid limit")
        if snapshot_id is not None and (
            not isinstance(snapshot_id, str)
            or not re.fullmatch(r"[a-f0-9]{64}", snapshot_id)
        ):
            raise ValueError("Invalid snapshot")
        rows = self.store.history(
            scope.namespace_id,
            logical_binding_id,
            after_revision,
            limit,
            repo_ids=scope.authorized_repo_ids,
            snapshot_id=snapshot_id,
        )
        # Each historical revision must be scoped independently of today's head.
        # Consult fact authority even here; only content-free audit records survive.
        for row in rows:
            self.authority.require(scope, row.repo_id)
            self._fact(scope, row.fact_version_id, historical=True)
        return self._return(scope, None, rows)

    def project_fact(
        self, scope: ScopeContext, *, fact_version_id: str, event_revision: int
    ) -> bool:
        """Redelivery-safe metadata projection; status always comes from authority.

        The host passes a J02 event revision, never a cached payload/status.
        Forgotten version IDs are terminal and can never be revived by an upsert.
        """
        self.authority.require(scope)
        identifier(fact_version_id)
        revision(event_revision)
        entry = self.facts.get_entry(
            scope.namespace_id, fact_version_id, include_inactive=True
        )
        terminal = (
            entry is None or entry.id != fact_version_id or entry.status != "active"
        )
        self.authority.require(scope)
        result = self.store.project(
            scope.namespace_id, fact_version_id, event_revision, terminal
        )
        return self._return(scope, None, result)

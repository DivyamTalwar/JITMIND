"""Opt-in retrieval with authorization before candidate/index construction.

Extension protocols here are trusted-host boundaries, not a sandbox for arbitrary
Python plugins. Legacy ResearchAgent and its unscoped retrievers are unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Protocol, runtime_checkable

from jitmind.schemas import (
    AdvancedMemoryStore,
    Hit,
    InMemoryMemoryStore,
    InMemoryPageStore,
    ResearchOutput,
)
from jitmind.scope import ScopeAuthority, ScopeContext, ScopeDenied, _identifier


class ScopedBackendError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Scoped backend unavailable")


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _snapshots(value) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple or len(value) > 1024:
        raise ScopeDenied()
    for pair in value:
        if type(pair) is not tuple or len(pair) != 2:
            raise ScopeDenied()
        for item in pair:
            _identifier(item)
    if len({p[0] for p in value}) != len(value):
        raise ScopeDenied()
    return tuple(sorted(value))


@dataclass(frozen=True)
class ScopedDocument:
    document_id: str
    namespace_id: str
    content: str
    source: str = "memory"
    repo_id: str | None = None
    snapshot_id: str | None = None
    metadata_json: str = "{}"

    def __post_init__(self):
        _identifier(self.document_id)
        _identifier(self.namespace_id)
        _identifier(self.source)
        if type(self.content) is not str or type(self.metadata_json) is not str:
            raise ScopeDenied()
        if (self.repo_id is None) != (self.snapshot_id is None):
            raise ScopeDenied()
        if self.repo_id is not None:
            _identifier(self.repo_id)
            _identifier(self.snapshot_id)
        if type(json.loads(self.metadata_json)) is not dict:
            raise ScopeDenied()


@dataclass(frozen=True)
class ScopedView:
    namespace_id: str
    snapshot_id: str
    documents: tuple[ScopedDocument, ...]
    repo_snapshots: tuple[tuple[str, str], ...] = ()

    def require(self, authority: ScopeAuthority, scope: ScopeContext) -> None:
        authority.require(scope)
        _identifier(self.snapshot_id)
        if self.namespace_id != scope.namespace_id or type(self.documents) is not tuple:
            raise ScopeDenied()
        selected = dict(_snapshots(self.repo_snapshots))
        for repo_id in selected:
            authority.require(scope, repo_id)
        seen = set()
        for doc in self.documents:
            if (
                type(doc) is not ScopedDocument
                or doc.namespace_id != scope.namespace_id
            ):
                raise ScopeDenied()
            if doc.document_id in seen:
                raise ScopeDenied()
            seen.add(doc.document_id)
            if doc.repo_id is not None:
                authority.require(scope, doc.repo_id)
                if selected.get(doc.repo_id) != doc.snapshot_id:
                    raise ScopeDenied()
        authority.require(scope)


@runtime_checkable
class ScopedBackend(Protocol):
    """Read an isolated snapshot BEFORE any scoring, expansion or provider call.

    validate_view must reject changed backing state/snapshot selection. Both
    methods must require authority before disk access and before returning.
    """

    scope_protocol: str

    def open_view(
        self,
        authority: ScopeAuthority,
        scope: ScopeContext,
        snapshots: tuple[tuple[str, str], ...],
    ) -> ScopedView: ...

    def validate_view(
        self, authority: ScopeAuthority, scope: ScopeContext, view: ScopedView
    ) -> None: ...


@dataclass(frozen=True)
class LegacyStoreBinding:
    """Host assigns an entire existing store pair to one namespace.

    Untagged old-format rows belong to this host assignment. Tagged rows must
    match exactly. Never bind a shared untagged tenant database this way.
    """

    namespace_id: str
    page_store: InMemoryPageStore
    memory_store: InMemoryMemoryStore | AdvancedMemoryStore | None = None


class LegacyScopedBackend:
    scope_protocol = "jitmind.scoped.v1"

    def __init__(self, bindings: tuple[LegacyStoreBinding, ...]):
        self._bindings = {}
        identities = set()
        for binding in bindings:
            _identifier(binding.namespace_id)
            if binding.namespace_id in self._bindings:
                raise ScopedBackendError()
            for store, allowed in (
                (binding.page_store, (InMemoryPageStore,)),
                (binding.memory_store, (InMemoryMemoryStore, AdvancedMemoryStore)),
            ):
                if store is None:
                    continue
                # Custom subclasses may read global state before filtering.
                if type(store) not in allowed:
                    raise ScopedBackendError()
                identity = (
                    (type(store), str(store._dir_path.resolve()))
                    if store._dir_path
                    else id(store)
                )
                if identity in identities:
                    raise ScopedBackendError()
                identities.add(identity)
            self._bindings[binding.namespace_id] = binding

    @staticmethod
    def _belongs(meta: dict, namespace: str) -> bool:
        if "namespace_id" in meta and meta["namespace_id"] != namespace:
            return False
        # PR #27 has component namespaces. Only a singleton exact mapping is
        # implicit; multi-component mappings require an explicit host adapter.
        return "namespace" not in meta or meta["namespace"] in (
            namespace,
            [namespace],
            (namespace,),
        )

    @staticmethod
    def _eligible(meta: dict) -> bool:
        now = datetime.now(timezone.utc)
        for key in ("t_valid", "t_invalid", "expires_at"):
            raw = meta.get(key)
            if raw is None:
                continue
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            # Legacy naive data is intentionally interpreted as UTC here.
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if key == "t_valid" and now < parsed:
                return False
            if key != "t_valid" and now >= parsed:
                return False
        return True

    @classmethod
    def _provenance(cls, metadata, namespace):
        """Intersect linked restrictions; only missing values may inherit.

        None denotes an excluded namespace component. Contradictory explicit
        repository/snapshot values are corrupt provenance, never overrides.
        """
        if any(not cls._belongs(meta, namespace) for meta in metadata):
            return None
        provenance = {}
        for key in ("repo_id", "snapshot_id"):
            values = [meta[key] for meta in metadata if meta.get(key) is not None]
            for value in values:
                _identifier(value)
            if len(set(values)) > 1:
                raise ScopeDenied()
            if values:
                provenance[key] = values[0]
        return provenance

    @classmethod
    def _linked_metadata(cls, entries, pages, namespace):
        # Follow both forward source references and reverse memory_id links.
        # Connected components prevent a shared page from laundering a second
        # owner's restriction through an untagged abstract.
        metadata = {("page", str(i)): dict(p.meta) for i, p in enumerate(pages)}
        for entry in entries:
            meta = dict(entry.meta)
            if hasattr(entry, "namespace"):
                meta["namespace"] = entry.namespace
            metadata[("memory", entry.id)] = meta
        edges = {key: set() for key in metadata}
        owners = {str(i): set() for i in range(len(pages))}

        def link(memory_id, page_id):
            left, right = ("memory", memory_id), ("page", page_id)
            if left not in edges or right not in edges:
                raise ScopeDenied()
            edges[left].add(right)
            edges[right].add(left)
            owners[page_id].add(memory_id)

        for entry in entries:
            if entry.source_page_id is not None:
                link(entry.id, entry.source_page_id)
        for i, page in enumerate(pages):
            if page.meta.get("memory_id") is not None:
                link(page.meta["memory_id"], str(i))
        visited = set()
        for key in metadata:
            if key in visited:
                continue
            component, pending = set(), [key]
            while pending:
                node = pending.pop()
                if node in component:
                    continue
                component.add(node)
                pending.extend(edges[node] - component)
            visited.update(component)
            provenance = cls._provenance(
                [metadata[node] for node in component], namespace
            )
            for node in component:
                metadata[node] = (
                    None if provenance is None else {**metadata[node], **provenance}
                )
        return metadata, owners

    def open_view(self, authority, scope, snapshots=()) -> ScopedView:
        authority.require(scope)
        snapshots = _snapshots(snapshots)
        for repo_id, _ in snapshots:
            authority.require(scope, repo_id)
        binding = self._bindings.get(scope.namespace_id)
        if binding is None:
            raise ScopeDenied()
        documents = []
        try:
            memory = binding.memory_store
            memory_documents = []
            eligible_entries = {}
            page_owners = {}
            authority.require(scope)
            pages = binding.page_store.load()
            linked = None
            if type(memory) is AdvancedMemoryStore:
                authority.require(scope)
                entries = memory.get_entries(include_inactive=True)
                linked, page_owners = self._linked_metadata(
                    entries, pages, scope.namespace_id
                )
                for entry in entries:
                    meta = linked[("memory", entry.id)]
                    if entry.status != "active" or meta is None:
                        continue
                    meta.update(t_valid=entry.t_valid, t_invalid=entry.t_invalid)
                    if not self._eligible(meta) or memory._apply_ttl(entry):
                        continue
                    doc = self._document(
                        f"memory:{entry.id}",
                        scope,
                        entry.content,
                        "memory",
                        meta,
                        snapshots,
                    )
                    if doc is not None:
                        eligible_entries[entry.id] = doc
                        memory_documents.append(doc)
            elif memory is not None:
                authority.require(scope)
                for index, content in enumerate(memory.load().abstracts):
                    memory_documents.append(
                        ScopedDocument(f"memory:{index}", scope.namespace_id, content)
                    )
            authority.require(scope)
            for index, page in enumerate(pages):
                meta = (
                    linked[("page", str(index))]
                    if linked is not None
                    else dict(page.meta)
                )
                if (
                    meta is None
                    or not self._belongs(meta, scope.namespace_id)
                    or not self._eligible(meta)
                ):
                    continue
                if type(memory) is AdvancedMemoryStore:
                    owners = set(page_owners.get(str(index), ()))
                    if not owners.issubset(eligible_entries):
                        continue
                documents.append(
                    self._document(
                        f"page:{index}", scope, page.content, "page", meta, snapshots
                    )
                )
            documents.extend(memory_documents)
        except ScopeDenied:
            raise
        except Exception:  # noqa: BLE001 - sanitize plugin/storage errors at the trust boundary
            raise ScopedBackendError() from None
        documents = tuple(doc for doc in documents if doc is not None)
        revision = _digest(_json([doc.__dict__ for doc in documents]))
        view = ScopedView(scope.namespace_id, revision, documents, snapshots)
        view.require(authority, scope)
        return view

    @staticmethod
    def _document(identifier, scope, content, source, meta, snapshots):
        repo = meta.get("repo_id")
        snapshot = meta.get("snapshot_id")
        if repo is not None or snapshot is not None:
            if (
                repo not in scope.authorized_repo_ids
                or dict(snapshots).get(repo) != snapshot
            ):
                return None
            if repo is None or snapshot is None:
                return None
        return ScopedDocument(
            identifier, scope.namespace_id, content, source, repo, snapshot, _json(meta)
        )

    def validate_view(self, authority, scope, view):
        view.require(authority, scope)
        current = self.open_view(authority, scope, view.repo_snapshots)
        if current.snapshot_id != view.snapshot_id:
            raise ScopeDenied()
        authority.require(scope)


@runtime_checkable
class ScopedIndexFactory(Protocol):
    """Trusted factory: build a fresh isolated index only from this view.

    Persistent index paths and all graph nodes/edges must be scoped to the view
    key; no ambient corpus, reused index, or fallback to unscoped retrieval.
    """

    scope_protocol: str

    def build_scoped(
        self, view: ScopedView, authority: ScopeAuthority, scope: ScopeContext
    ): ...


class LexicalScopedIndex:
    scope_protocol = "jitmind.scoped.v1"

    def build_scoped(self, view, authority, scope):
        view.require(authority, scope)
        # Request-local immutable postings; no shared current-user field.
        postings = tuple(
            (doc.document_id, frozenset(re.findall(r"\w+", doc.content.lower())))
            for doc in view.documents
        )

        class Index:
            def search(self, query, top_k):
                view.require(authority, scope)
                terms = set(re.findall(r"\w+", query.lower()))
                scores = [
                    (doc_id, float(len(terms & words))) for doc_id, words in postings
                ]
                answer = sorted(
                    (row for row in scores if row[1]), key=lambda row: (-row[1], row[0])
                )[:top_k]
                view.require(authority, scope)
                return answer

        return Index()


def _capability(value, protocol):
    if not isinstance(value, protocol) or value.scope_protocol != "jitmind.scoped.v1":
        raise ScopedBackendError()


@dataclass(frozen=True)
class ScopedResponse:
    """Call serialize at the delivery boundary to recheck revocation/state."""

    _authority: ScopeAuthority
    _scope: ScopeContext
    _views: tuple[tuple[ScopedBackend, ScopedView], ...]
    _payload: str

    def serialize(self) -> dict:
        self._authority.require(self._scope)
        try:
            for backend, view in self._views:
                view.require(self._authority, self._scope)
                backend.validate_view(self._authority, self._scope, view)
                view.require(self._authority, self._scope)
            result = json.loads(self._payload)
            self._authority.require(self._scope)
            return result
        except ScopeDenied:
            raise
        except Exception:  # noqa: BLE001 - do not disclose storage/plugin errors
            self._authority.require(self._scope)
            raise ScopedBackendError() from None


class ScopedResearchFacade:
    """One-pass retrieval/integration. Does not invoke legacy agent side effects.

    No ambient graph, tools, profiles, replay, checkpoint, or reflection writes.
    Optional backends fail to an already-authorized primary memory channel only.
    """

    def __init__(
        self,
        authority: ScopeAuthority,
        backend: ScopedBackend,
        *,
        index_factory: ScopedIndexFactory | None = None,
        generator=None,
        reranker=None,
        optional_backends: tuple[ScopedBackend, ...] = (),
        cache_size: int = 128,
        max_document_bytes: int = 16_384,
        max_evidence_bytes: int = 65_536,
    ):
        _capability(backend, ScopedBackend)
        for extra in optional_backends:
            _capability(extra, ScopedBackend)
        factory = index_factory or LexicalScopedIndex()
        _capability(factory, ScopedIndexFactory)
        if type(cache_size) is not int or not 0 <= cache_size <= 4096:
            raise ValueError("Invalid cache size")
        for budget in (max_document_bytes, max_evidence_bytes):
            if type(budget) is not int or not 1 <= budget <= 16_777_216:
                raise ValueError("Invalid evidence budget")
        self.authority = authority
        self.backend = backend
        self.optional_backends = tuple(optional_backends)
        self.index_factory = factory
        self.generator = generator
        self.reranker = reranker
        self._cache_size = cache_size
        self._max_document_bytes = max_document_bytes
        self._max_evidence_bytes = max_evidence_bytes
        self._cache = OrderedDict()
        self._lock = RLock()

    def _check(self, scope, views):
        self.authority.require(scope)
        for backend, view in views:
            view.require(self.authority, scope)
            backend.validate_view(self.authority, scope, view)
        self.authority.require(scope)

    def _require_evidence_budget(self, hits):
        # Count strict UTF-8 bytes, including source labels and separators.
        # Reject the entire selection deterministically; never silently omit or
        # truncate evidence and never call either model hook on overflow.
        total = 0
        for index, hit in enumerate(hits):
            if len(hit.snippet) > self._max_document_bytes:
                raise ScopedBackendError()
            size = len(hit.snippet.encode("utf-8"))
            if size > self._max_document_bytes:
                raise ScopedBackendError()
            total += size + len(f"[{hit.page_id}] ".encode()) + (index > 0)
            if total > self._max_evidence_bytes:
                raise ScopedBackendError()

    def research(
        self, scope: ScopeContext, request: str, *, snapshots=(), top_k=10
    ) -> ScopedResponse:
        self.authority.require(scope)
        if (
            type(request) is not str
            or not request.strip()
            or len(request) > 100_000
            or "\0" in request
        ):
            raise ValueError("Invalid request")
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError("Invalid result limit")
        snapshots = _snapshots(snapshots)
        for repo_id, _ in snapshots:
            self.authority.require(scope, repo_id)
        try:
            views = []
            channels = []
            for index, backend in enumerate((self.backend,) + self.optional_backends):
                self.authority.require(scope)
                try:
                    view = backend.open_view(self.authority, scope, snapshots)
                    view.require(self.authority, scope)
                    if view.repo_snapshots != snapshots:
                        raise ScopeDenied()
                    views.append((backend, view))
                    channels.append(index)
                except ScopeDenied:
                    raise
                except Exception:
                    self.authority.require(scope)
                    if index == 0:
                        raise
            views = tuple(views)
            self._check(scope, views)
            hits = []
            for channel, (_, view) in zip(channels, views):
                key = (
                    scope.namespace_id,
                    scope.principal_id,
                    scope.authorization_version,
                    view.snapshot_id,
                    snapshots,
                    channel,
                    request,
                    top_k,
                )
                self.authority.require(scope)
                with self._lock:
                    # Regrant cannot reuse old versions, even if content is equal.
                    for old in list(self._cache):
                        if old[:2] == key[:2] and old[2] != key[2]:
                            del self._cache[old]
                    rows = self._cache.get(key)
                if rows is None:
                    view.require(self.authority, scope)
                    isolated = self.index_factory.build_scoped(
                        view, self.authority, scope
                    )
                    view.require(self.authority, scope)
                    rows = isolated.search(request, top_k)
                    if type(rows) not in (tuple, list) or len(rows) > top_k:
                        raise ScopeDenied()
                    rows = tuple(rows)
                self._check(scope, views)
                documents = {doc.document_id: doc for doc in view.documents}
                if len(rows) > top_k:
                    raise ScopeDenied()
                seen = set()
                for rank, (doc_id, score) in enumerate(rows):
                    if (
                        doc_id not in documents
                        or doc_id in seen
                        or type(score) not in (float, int)
                        or not math.isfinite(score)
                    ):
                        raise ScopeDenied()
                    seen.add(doc_id)
                    doc = documents[doc_id]
                    self.authority.require(scope, doc.repo_id)
                    meta = json.loads(doc.metadata_json)
                    meta.update(
                        namespace_id=doc.namespace_id,
                        repo_id=doc.repo_id,
                        snapshot_id=doc.snapshot_id,
                        view_snapshot=view.snapshot_id,
                        source_digest=meta.get("source_digest", _digest(doc.content)),
                        score=score,
                        rrf_score=1.0 / (61 + rank),
                    )
                    # Resolve canonical text from the authorized view, never a
                    # cached/retriever-supplied snippet or global page lookup.
                    hits.append(
                        Hit(
                            page_id=f"{channel}:{doc_id}",
                            snippet=doc.content,
                            source=doc.source,
                            meta=meta,
                        )
                    )
                with self._lock:
                    if self._cache_size:
                        self._cache[key] = rows
                        self._cache.move_to_end(key)
                        while len(self._cache) > self._cache_size:
                            self._cache.popitem(last=False)
            hits.sort(key=lambda hit: -hit.meta["rrf_score"])
            hits = hits[:top_k]
            self._check(scope, views)
            self._require_evidence_budget(hits)
            if self.reranker and hits:
                ranked = self.reranker.rerank(
                    request, [hit.snippet for hit in hits], top_k=len(hits)
                )
                self._check(scope, views)
                scores = {}
                for index, score in ranked:
                    if (
                        type(index) is not int
                        or not 0 <= index < len(hits)
                        or type(score) not in (float, int)
                        or not math.isfinite(score)
                    ):
                        raise ScopeDenied()
                    scores[index] = score
                hits = [
                    hit
                    for index, hit in sorted(
                        enumerate(hits), key=lambda pair: -scores.get(pair[0], 0)
                    )
                ]
            self._check(scope, views)
            evidence = "\n".join(f"[{hit.page_id}] {hit.snippet}" for hit in hits)
            answer = evidence
            if self.generator and hits:
                from jitmind.schemas import INTEGRATE_SCHEMA

                response = self.generator.generate_single(
                    prompt=f"Question: {request}\nEvidence:\n{evidence}",
                    schema=INTEGRATE_SCHEMA,
                )
                self._check(scope, views)
                data = response.get("json") or json.loads(response["text"])
                answer = data["content"]
                if type(answer) is not str:
                    raise ScopedBackendError()
            output = ResearchOutput(
                integrated_memory=answer,
                raw_memory={
                    "hits": [hit.model_dump() for hit in hits],
                    "sources": [hit.page_id for hit in hits],
                    "request_id": scope.request_id,
                },
            )
            self._check(scope, views)
            return ScopedResponse(
                self.authority, scope, views, output.model_dump_json()
            )
        except ScopeDenied:
            with self._lock:
                self._cache.clear()
            raise
        except Exception:  # noqa: BLE001 - sanitize plugin/storage errors at the trust boundary
            # Neither plugin errors nor foreign content are echoed to logs/errors.
            self.authority.require(scope)
            raise ScopedBackendError() from None

    async def research_async(self, scope, request, **kwargs):
        return await asyncio.to_thread(self.research, scope, request, **kwargs)


class DurableScopedBackend:
    """J02 SQLiteDurableStore adapter using namespace-qualified public reads.

    A bounded, complete namespace snapshot is required; truncation fails closed.
    Pages are expanded only by IDs from that snapshot and a revision check
    rejects concurrent mutation. Does not depend on eventual projections.
    """

    scope_protocol = "jitmind.scoped.v1"

    def __init__(self, store, *, limit: int = 1000):
        try:
            from jitmind.storage import SQLiteDurableStore
        except ImportError:
            raise ScopedBackendError() from None
        if (
            type(store) is not SQLiteDurableStore
            or type(limit) is not int
            or not 1 <= limit <= 1000
        ):
            raise ScopedBackendError()
        self._store = store
        self._limit = limit

    def open_view(self, authority, scope, snapshots=()):
        authority.require(scope)
        snapshots = _snapshots(snapshots)
        for repo_id, _ in snapshots:
            authority.require(scope, repo_id)
        try:
            authority.require(scope)
            snapshot = self._store.snapshot(scope.namespace_id, limit=self._limit)
            if (
                snapshot.truncated
                or type(snapshot.revision) is not int
                or snapshot.revision < 0
            ):
                raise ScopedBackendError()
            documents = []
            for entry in snapshot.entries:
                authority.require(scope)
                meta = dict(entry.meta)
                if not LegacyScopedBackend._belongs(meta, scope.namespace_id):
                    continue
                meta.update(t_valid=entry.t_valid, t_invalid=entry.t_invalid)
                if not LegacyScopedBackend._eligible(meta):
                    continue
                # Honor known fact restrictions before reading its source. A
                # page may add missing restrictions, which are reconciled below.
                repo = meta.get("repo_id")
                if repo is not None:
                    if repo not in scope.authorized_repo_ids:
                        continue
                    authority.require(scope, repo)
                    if (
                        meta.get("snapshot_id") is not None
                        and dict(snapshots).get(repo) != meta["snapshot_id"]
                    ):
                        continue
                page = None
                page_meta = None
                if entry.source_page_id is not None:
                    authority.require(scope)
                    page = self._store.get_page(
                        scope.namespace_id, entry.source_page_id
                    )
                    authority.require(scope)
                    if page is None:
                        raise ScopeDenied()
                    page_meta = dict(page.meta)
                    provenance = LegacyScopedBackend._provenance(
                        [meta, page_meta], scope.namespace_id
                    )
                    if provenance is None:
                        raise ScopeDenied()
                    meta.update(provenance)
                    page_meta.update(provenance)
                doc = LegacyScopedBackend._document(
                    f"memory:{entry.id}",
                    scope,
                    entry.content,
                    "memory",
                    meta,
                    snapshots,
                )
                if doc is None:
                    continue
                authority.require(scope, doc.repo_id)
                documents.append(doc)
                if page is None or not LegacyScopedBackend._eligible(page_meta):
                    continue
                documents.append(
                    ScopedDocument(
                        f"page:{entry.source_page_id}",
                        scope.namespace_id,
                        page.content,
                        "page",
                        doc.repo_id,
                        doc.snapshot_id,
                        _json(page_meta),
                    )
                )
            authority.require(scope)
            if self._store.status(scope.namespace_id)["revision"] != snapshot.revision:
                raise ScopeDenied()
            revision = f"{snapshot.revision}:" + _digest(
                _json([doc.__dict__ for doc in documents])
            )
            view = ScopedView(scope.namespace_id, revision, tuple(documents), snapshots)
            view.require(authority, scope)
            return view
        except (ScopeDenied, ScopedBackendError):
            raise
        except Exception:  # noqa: BLE001 - sanitize plugin/storage errors at the trust boundary
            raise ScopedBackendError() from None

    def validate_view(self, authority, scope, view):
        view.require(authority, scope)
        current = self.open_view(authority, scope, view.repo_snapshots)
        if current.snapshot_id != view.snapshot_id:
            raise ScopeDenied()
        authority.require(scope)

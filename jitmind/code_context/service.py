"""Explicit snapshot build and scoped, bounded operations over local AST evidence."""

from __future__ import annotations

import re
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from threading import Event, RLock
from typing import Literal

from pydantic import Field, ValidationError, field_validator

from jitmind.scope import ScopeAuthority, ScopeContext

from .models import (
    PROTOCOL,
    CodeEvidence,
    CodeEvidencePage,
    CodeQuery,
    StrictModel,
    finish,
    safe_path,
    source_lines,
    strict_json,
    under,
)
from .process import NodeParser, Unavailable, checkpoint
from .snapshot import (
    COVERAGE_DIGEST,
    RepoRegistry,
    Snapshot,
    canonical,
    capture,
    digest,
)


class _Symbol(StrictModel):
    name: str = Field(min_length=1, max_length=1024)
    qualified_name: str = Field(min_length=1, max_length=4096)
    kind: Literal["function", "method", "class"]
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    signature: str = Field(max_length=131072)
    calls: list[str] = Field(max_length=1024)

    @field_validator("calls")
    @classmethod
    def calls_bounded(cls, value):
        if any(not s or len(s) > 1024 for s in value):
            raise ValueError("call name budget")
        return value


class _File(StrictModel):
    path: str
    digest: str = Field(pattern="^[a-f0-9]{64}$")
    status: Literal["ok", "parse_error", "unsupported"]
    symbols: list[_Symbol] = Field(max_length=1024)

    @field_validator("path")
    @classmethod
    def path_valid(cls, value):
        return safe_path(value)


class _Reply(StrictModel):
    schema_version: Literal["jitmind.graft-parser/v1"] = Field(alias="schema")
    snapshot_id: str = Field(pattern="^[a-f0-9]{64}$")
    parser: Literal["graft-python-subset/1;tree-sitter/0.21.1;python/0.21.0"]
    files: list[_File] = Field(max_length=128)


@dataclass(frozen=True)
class _Node:
    path: str
    ordinal: int
    name: str
    qualified_name: str
    kind: str
    start_line: int
    end_line: int
    signature: str
    calls: tuple[str, ...]

    @property
    def key(self):
        return self.path, self.ordinal


@dataclass(frozen=True)
class _Index:
    snapshot: Snapshot
    nodes: tuple[_Node, ...]
    statuses: tuple[tuple[str, str], ...]
    generation: int


def _validate_reply(raw, snapshot):
    try:
        reply = _Reply.model_validate(strict_json(raw))
        expected = {f.path: f for f in snapshot.files if f.status == "ok"}
        if reply.snapshot_id != snapshot.snapshot_id or len(reply.files) != len(
            expected
        ):
            raise ValueError("snapshot mismatch")
        seen = set()
        nodes = []
        statuses = {f.path: f.status for f in snapshot.files}
        for file in reply.files:
            source = expected.get(file.path)
            if source is None or file.path in seen or file.digest != source.digest:
                raise ValueError("source mismatch")
            seen.add(file.path)
            statuses[file.path] = file.status
            if file.status != "ok" and file.symbols:
                raise ValueError("invalid partial parse")
            for ordinal, symbol in enumerate(file.symbols):
                if (
                    not symbol.start_line
                    <= symbol.end_line
                    <= len(source_lines(source.source))
                ):
                    raise ValueError("invalid span")
                nodes.append(
                    _Node(
                        file.path,
                        ordinal,
                        symbol.name,
                        symbol.qualified_name,
                        symbol.kind,
                        symbol.start_line,
                        symbol.end_line,
                        symbol.signature,
                        tuple(symbol.calls),
                    )
                )
                if len(nodes) > 2048:
                    raise ValueError("symbol budget")
        return tuple(nodes), tuple(sorted(statuses.items()))
    except (ValueError, TypeError, RecursionError, ValidationError):
        raise Unavailable("invalid_adapter_response") from None


class CodeContext:
    """Optional service: register trusted roots, build explicitly, then query snapshots.

    Generation retention is process-local, at most eight 2-MiB source snapshots.
    Authorization checks guard every read and every return, including cache hits.
    """

    def __init__(
        self, authority: ScopeAuthority, registry: RepoRegistry, parser: NodeParser
    ):
        self.authority = authority
        self.registry = registry
        self.parser = parser
        self._indices = OrderedDict()
        self._lock = RLock()
        self._generation = 0

    @staticmethod
    def _key(scope, repo_id, snapshot_id):
        return (
            scope.namespace_id,
            scope.principal_id,
            scope.authorization_version,
            repo_id,
            snapshot_id,
        )

    def _return(self, scope, repo_id, page):
        self.authority.require(scope, repo_id)
        return finish(page)

    def _unavailable(self, scope, repo_id, reason):
        return self._return(
            scope,
            repo_id,
            CodeEvidencePage(
                status="unavailable", repo_id=repo_id, reason_codes=(reason,)
            ),
        )

    def build(
        self,
        scope: ScopeContext,
        repo_id: str,
        *,
        deadline_ms: int = 1500,
        cancel: Event | None = None,
    ) -> CodeEvidencePage:
        self.authority.require(scope, repo_id)
        if type(deadline_ms) is not int or not 1 <= deadline_ms <= 30000:
            raise ValueError("Invalid deadline")
        repo = self.registry.get(repo_id, scope.namespace_id)
        deadline = time.monotonic() + deadline_ms / 1000
        try:
            snapshot = capture(repo, deadline, cancel)
            payload = canonical(
                {
                    "schema": PROTOCOL,
                    "snapshot_id": snapshot.snapshot_id,
                    "files": [
                        {"path": f.path, "digest": f.digest, "source": f.source}
                        for f in snapshot.files
                        if f.status == "ok"
                    ],
                }
            )
            raw = self.parser.parse(payload, deadline, cancel)
            nodes, statuses = _validate_reply(raw, snapshot)
            # A second bounded byte capture detects edits during extraction, including
            # same-size, same-HEAD edits. Never silently publish a mixed generation.
            verify = capture(repo, deadline, cancel)
            if (
                verify.snapshot_id != snapshot.snapshot_id
                or verify.scan_complete != snapshot.scan_complete
            ):
                raise Unavailable("source_changed")
            checkpoint(deadline, cancel)
            self.authority.require(scope, repo_id)
            with self._lock:
                self._generation += 1
                index = _Index(snapshot, nodes, statuses, self._generation)
                key = self._key(scope, repo_id, snapshot.snapshot_id)
                self._indices[key] = index
                self._indices.move_to_end(key)
                while len(self._indices) > 8:
                    self._indices.popitem(last=False)
            page = self._page(index, repo_id, "", freshness={"state": "fresh"})
            return self._return(scope, repo_id, page)
        except Unavailable as error:
            return self._unavailable(scope, repo_id, error.reason)

    def _page(self, index, repo_id, prefix, **kwargs):
        statuses = Counter(
            status for path, status in index.statuses if under(path, prefix)
        )
        incomplete = not index.snapshot.scan_complete or any(
            k != "ok" for k in statuses
        )
        return CodeEvidencePage(
            status="ok",
            repo_id=repo_id,
            snapshot_id=index.snapshot.snapshot_id,
            manifest_digest=index.snapshot.manifest_digest,
            coverage_digest=COVERAGE_DIGEST,
            generation=index.generation,
            coverage={
                "state": "incomplete" if incomplete else "complete_subset",
                "files": dict(statuses),
                "languages": ["python"],
                "call_resolution": "same_file_bare_name_candidates",
                "runtime_complete": False,
            },
            **kwargs,
        )

    def _freshness(self, old, now, prefix):
        before = {f.path: f for f in old.scoped(prefix)}
        after = {f.path: f for f in now.scoped(prefix)}
        changed, added, removed, unknown = [], [], [], []
        for path in sorted(before.keys() | after.keys()):
            a, b = before.get(path), after.get(path)
            if b is None:
                (removed if now.scan_complete else unknown).append(path)
            elif b.status != "ok" or (a is not None and a.status != "ok"):
                unknown.append(path)
            elif a is None:
                added.append(path)
            elif a.digest != b.digest:
                changed.append(path)
        state = (
            "unknown"
            if unknown or not now.scan_complete
            else "stale"
            if changed or added or removed
            else "fresh"
        )
        return {
            "state": state,
            "changed": changed,
            "added": added,
            "removed": removed,
            "unknown": unknown,
        }

    def query(
        self,
        scope: ScopeContext,
        request: CodeQuery | dict,
        *,
        cancel: Event | None = None,
    ) -> CodeEvidencePage:
        return self._query(scope, request, cancel=cancel)

    def _query(self, scope, request, *, cancel=None, deadline=None):
        """Use the workspace's absolute ceiling without rounding or renewing it."""
        # Check authority before even validating potentially hostile request payloads.
        self.authority.require(scope)
        query = CodeQuery.model_validate(
            request.model_dump(by_alias=True)
            if isinstance(request, CodeQuery)
            else request
        )
        self.authority.require(scope, query.repo_id)
        repo = self.registry.get(query.repo_id, scope.namespace_id)
        request_deadline = time.monotonic() + query.deadline_ms / 1000
        deadline = (
            request_deadline if deadline is None else min(deadline, request_deadline)
        )
        try:
            checkpoint(deadline, cancel)
            with self._lock:
                index = self._indices.get(
                    self._key(scope, query.repo_id, query.snapshot_id)
                )
            if index is None:
                raise Unavailable("snapshot_unavailable")
            now = capture(repo, deadline, cancel)
            freshness = self._freshness(index.snapshot, now, query.path_prefix)
            if query.operation == "check_freshness":
                return self._return(
                    scope,
                    query.repo_id,
                    self._page(
                        index, query.repo_id, query.path_prefix, freshness=freshness
                    ),
                )
            candidates, reasons, walked_complete = self._select(
                index, query, deadline, cancel
            )
            results = []
            total_bytes = 0
            files = {f.path: f for f in index.snapshot.files}
            truncated = not walked_complete
            # One UTF-8 byte per token is a deliberately conservative upper bound,
            # not a fabricated tokenizer measurement. Exact bytes are also returned.
            context_bytes = 0
            for node, score, depth in candidates:
                checkpoint(deadline, cancel)
                if len(results) >= query.limit:
                    truncated = True
                    reasons.add("result_limit")
                    break
                source = files[node.path]
                lines = source_lines(source.source)
                selected = []
                signature = node.signature[:2048]
                if len(signature) < len(node.signature):
                    truncated = True
                    reasons.add("signature_truncated")
                metadata_bytes = len((node.name + signature).encode("utf8"))
                remaining = query.max_context_tokens - context_bytes - metadata_bytes
                for line in lines[node.start_line - 1 : node.end_line]:
                    size = len(line.encode("utf8"))
                    if size > remaining:
                        truncated = True
                        reasons.add("context_budget")
                        break
                    selected.append(line)
                    remaining -= size
                if not selected:
                    continue
                text = "".join(selected)
                total_bytes += len(text.encode("utf8"))
                context_bytes += len(text.encode("utf8")) + metadata_bytes
                source_id = digest(
                    canonical(
                        [
                            scope.namespace_id,
                            scope.principal_id,
                            scope.authorization_version,
                            query.repo_id,
                            query.snapshot_id,
                            node.path,
                            node.ordinal,
                            node.start_line,
                            node.start_line + len(selected) - 1,
                        ]
                    )
                )
                results.append(
                    CodeEvidence(
                        source_id=source_id,
                        repo_id=query.repo_id,
                        namespace_id=scope.namespace_id,
                        snapshot_id=query.snapshot_id,
                        path=node.path,
                        digest=source.digest,
                        start_line=node.start_line,
                        end_line=node.start_line + len(selected) - 1,
                        name=node.name,
                        kind=node.kind,
                        source=text,
                        signature=signature,
                        score=float(score),
                        depth=depth,
                    )
                )
            checkpoint(deadline, cancel)
            base = self._page(index, query.repo_id, query.path_prefix)
            if query.operation == "repo_map":
                # Empty files have identity/digest metadata, but no source span.
                # Keep CodeEvidence's positive, exact source-span contract intact.
                empty_files = [
                    {"path": f.path, "digest": f.digest}
                    for f in index.snapshot.scoped(query.path_prefix)
                    if f.status == "ok" and not f.source
                ]
                base = base.model_copy(
                    update={"coverage": {**base.coverage, "empty_files": empty_files}}
                )
                if empty_files:
                    reasons.add("empty_files_without_source")
            if base.coverage["state"] == "incomplete":
                reasons.add("coverage_incomplete")
            reasons.update(index.snapshot.reasons)
            if freshness["state"] != "fresh":
                reasons.add("source_" + freshness["state"])
            count = (
                len(candidates)
                if walked_complete and base.coverage["state"] != "incomplete"
                else None
            )
            page = base.model_copy(
                update={
                    "results": tuple(results),
                    "freshness": freshness,
                    "truncated": truncated,
                    "reason_codes": tuple(sorted(reasons)),
                    "count": count,
                    "source_bytes": total_bytes,
                    "context_tokens_upper_bound": context_bytes,
                }
            )
            return self._return(scope, query.repo_id, page)
        except Unavailable as error:
            return self._unavailable(scope, query.repo_id, error.reason)

    def _select(self, index, query, deadline, cancel):
        nodes = [n for n in index.nodes if under(n.path, query.path_prefix)]
        files = [
            f
            for f in index.snapshot.files
            if f.status == "ok" and under(f.path, query.path_prefix)
        ]
        reasons = set()
        if query.operation == "file_api":
            safe_path(query.query)
            return [(n, 1, None) for n in nodes if n.path == query.query], reasons, True
        if query.operation == "repo_map":
            counts = Counter(n.path for n in nodes)
            return (
                [
                    (
                        _Node(f.path, -1, f.path, f.path, "file", 1, 1, "", ()),
                        counts[f.path],
                        None,
                    )
                    for f in sorted(files, key=lambda f: (-counts[f.path], f.path))
                    if f.source
                ],
                reasons,
                True,
            )
        if not query.query:
            raise ValueError("This operation requires a query")
        if query.operation == "find_all":
            matches = []
            for f in files:
                for line_number, line in enumerate(source_lines(f.source), 1):
                    checkpoint(deadline, cancel)
                    if query.query in line:
                        matches.append(
                            (
                                _Node(
                                    f.path,
                                    -line_number,
                                    query.query,
                                    query.query,
                                    "match",
                                    line_number,
                                    line_number,
                                    "",
                                    (),
                                ),
                                1,
                                None,
                            )
                        )
                        if len(matches) >= 2048:
                            reasons.add("match_budget")
                            return matches, reasons, False
            return matches, reasons, True
        if query.operation == "find_code":
            terms = set(re.findall(r"\w+", query.query.casefold()))
            source_map = {f.path: source_lines(f.source) for f in files}
            ranked = []
            for node in nodes:
                checkpoint(deadline, cancel)
                body = "".join(
                    source_map[node.path][node.start_line - 1 : node.end_line]
                ).casefold()
                score = sum(
                    4 * (term in node.qualified_name.casefold()) + (term in body)
                    for term in terms
                )
                if score:
                    ranked.append((node, score, None))
            ranked.sort(key=lambda row: (-row[1], row[0].path, row[0].start_line))
            return ranked, reasons, True
        # Resolve within the filtered graph BEFORE expansion. Edges are candidate
        # syntactic calls, never proof of runtime reachability or dead code.
        reasons.add("structural_candidates_only")
        by_name = {}
        for node in nodes:
            if node.kind == "function":
                by_name.setdefault((node.path, node.name), []).append(node)
        adjacency = {n.key: set() for n in nodes}
        candidate_count = 0
        complete = True
        for node in nodes:
            for name in node.calls:
                if candidate_count >= 4096:
                    reasons.add("edge_budget")
                    complete = False
                    break
                checkpoint(deadline, cancel)
                # Charge every candidate, including unresolved/ambiguous calls
                # and duplicates, before attempting name resolution.
                candidate_count += 1
                targets = by_name.get((node.path, name), [])
                if len(targets) == 1:
                    target = targets[0]
                    src, dst = (
                        (node.key, target.key)
                        if query.direction == "out"
                        else (target.key, node.key)
                    )
                    adjacency[src].add(dst)
            if not complete:
                break
        seeds = [n for n in nodes if query.query in (n.name, n.qualified_name, n.path)]
        if len(seeds) > 1024:
            seeds = seeds[:1024]
            reasons.add("node_budget")
            complete = False
        visited = {n.key for n in seeds}
        queue = deque((n.key, 0) for n in seeds)
        by_key = {n.key: n for n in nodes}
        found = []
        max_depth = 1024 if query.depth == "all" else query.depth
        while queue:
            checkpoint(deadline, cancel)
            key, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for nxt in sorted(adjacency[key]):
                if nxt in visited:
                    continue
                if len(visited) >= 1024:
                    reasons.add("node_budget")
                    return found, reasons, False
                visited.add(nxt)
                queue.append((nxt, depth + 1))
                found.append((by_key[nxt], 1, depth + 1))
        return found, reasons, complete

    def workspace(
        self,
        scope: ScopeContext,
        requests: list[CodeQuery],
        *,
        cancel: Event | None = None,
    ) -> tuple[CodeEvidencePage, ...]:
        """Per-repository pages retain EVERY option; no unscoped fallback or merge."""
        self.authority.require(scope)
        if type(requests) is not list or not 1 <= len(requests) <= 16:
            raise ValueError("Workspace request budget")
        for request in requests:
            if not isinstance(request, CodeQuery):
                raise TypeError("Expected CodeQuery")
            self.authority.require(scope, request.repo_id)
        requests = [
            CodeQuery.model_validate(r.model_dump(by_alias=True)) for r in requests
        ]
        deadline = time.monotonic() + max(r.deadline_ms for r in requests) / 1000
        pages = []
        total = 2
        try:
            for request in requests:
                checkpoint(deadline, cancel)
                page = self._query(scope, request, cancel=cancel, deadline=deadline)
                checkpoint(deadline, cancel)
                total += page.response_bytes + 1
                if total > 262144:
                    raise Unavailable("workspace_response_budget")
                pages.append(page)
            self.authority.require(scope)
            checkpoint(deadline, cancel)
            return tuple(pages)
        except Unavailable as error:
            return tuple(
                self._unavailable(scope, r.repo_id, error.reason) for r in requests
            )

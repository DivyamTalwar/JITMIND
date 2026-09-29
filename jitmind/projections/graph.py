"""Persistent directed graph from explicit trusted relation assertions."""

from __future__ import annotations

import json
from collections import deque

from jitmind.storage.models import validate_identifier

from .base import SQLiteProjection
from .models import GraphEdge, ProjectionError, ProjectionHit, QueryResult, integer


class SQLiteGraphProjection(SQLiteProjection):
    kind = "graph"
    app_id = 0x4A475031
    extra_schema = (
        "CREATE TABLE nodes (namespace TEXT NOT NULL, entity TEXT NOT NULL, PRIMARY KEY(namespace,entity))",
        "CREATE TABLE edges (namespace TEXT NOT NULL, subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL, PRIMARY KEY(namespace,subject,predicate,object))",
        "CREATE TABLE supports (namespace TEXT NOT NULL, subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL, fact TEXT NOT NULL, revision INTEGER NOT NULL, PRIMARY KEY(namespace,subject,predicate,object,fact))",
        "CREATE INDEX supports_fact ON supports(namespace,fact)",
    )

    def __init__(self, path, *, source_id, trusted_relations=False, fault_hook=None):
        if type(trusted_relations) is not bool:
            raise ProjectionError("invalid_request")
        self.trusted_relations = trusted_relations
        super().__init__(
            path,
            source_id=source_id,
            identity={"relations": "host-asserted-v1", "trusted": trusted_relations},
            fault_hook=fault_hook,
        )

    def _prepare(self, fact):
        relations = json.loads(fact.metadata_json).get("relations", [])
        if relations and not self.trusted_relations:
            raise ProjectionError("untrusted_relations")
        if type(relations) is not list or len(relations) > 100:
            raise ProjectionError("relation_budget")
        result = set()
        for relation in relations:
            if type(relation) is not dict or set(relation) != {
                "subject",
                "predicate",
                "object",
            }:
                raise ProjectionError("invalid_relation")
            triple = tuple(relation[key] for key in ("subject", "predicate", "object"))
            try:
                for value in triple:
                    validate_identifier(value)
            except Exception:  # noqa: BLE001 - sanitize trusted callback/validation failures
                raise ProjectionError("invalid_relation") from None
            result.add(triple)
        return tuple(sorted(result))

    def _remove(self, conn, namespace, fact):
        conn.execute(
            "DELETE FROM supports WHERE namespace=? AND fact=?", (namespace, fact)
        )
        conn.execute(
            "DELETE FROM edges WHERE namespace=? AND NOT EXISTS (SELECT 1 FROM supports s WHERE "
            "s.namespace=edges.namespace AND s.subject=edges.subject AND s.predicate=edges.predicate AND s.object=edges.object)",
            (namespace,),
        )
        conn.execute(
            "DELETE FROM nodes WHERE namespace=? AND NOT EXISTS (SELECT 1 FROM edges e WHERE "
            "e.namespace=nodes.namespace AND (e.subject=nodes.entity OR e.object=nodes.entity))",
            (namespace,),
        )

    def _insert(self, conn, namespace, fact, prepared):
        for subject, predicate, obj in prepared:
            conn.executemany(
                "INSERT OR IGNORE INTO nodes VALUES (?,?)",
                ((namespace, subject), (namespace, obj)),
            )
            conn.execute(
                "INSERT OR IGNORE INTO edges VALUES (?,?,?,?)",
                (namespace, subject, predicate, obj),
            )
            conn.execute(
                "INSERT INTO supports VALUES (?,?,?,?,?,?)",
                (namespace, subject, predicate, obj, fact.fact_id, fact.revision),
            )

    def traverse(
        self,
        authority,
        scope,
        source,
        start,
        *,
        snapshots=(),
        depth=2,
        edge_limit=100,
        visited_limit=100,
        candidate_limit=1000,
    ):
        authority.require(scope)
        validate_identifier(start)
        integer(depth, 0, 10)
        integer(edge_limit, 1, 1000)
        integer(visited_limit, 1, 1000)
        integer(candidate_limit, 1, 1000)
        try:
            snapshots, snap, reasons, generation = self._query_start(
                authority, scope, source, snapshots, candidate_limit
            )
            with self._connect() as conn:
                live = self._live(conn, scope.namespace_id, snap)
                if any(
                    f.visible and f.status == "active" and f.fact_id not in live
                    for f in snap.facts
                ):
                    reasons.append("coverage_gap")
                # Supporting IDs are authorized BEFORE expansion or relation reads.
                if not live:
                    edges, visited = {}, set()
                else:
                    placeholders = ",".join("?" for _ in live)
                    edges, visited = {}, {start}
                    queue = deque([(start, 0)])
                    stopped = False
                    support_count = 0
                    output_bytes = 0
                    while queue and not stopped:
                        entity, distance = queue.popleft()
                        authority.require(scope)
                        rows = conn.execute(
                            "SELECT subject,predicate,object,fact,revision FROM supports WHERE namespace=? "
                            f"AND subject=? AND fact IN ({placeholders}) ORDER BY predicate,object,fact LIMIT ?",
                            (scope.namespace_id, entity, *live, 1001 - support_count),
                        ).fetchall()
                        if distance >= depth:
                            if rows:
                                reasons.append("depth_budget")
                            continue
                        for row in rows:
                            support_count += 1
                            if support_count > 1000:
                                reasons.append("support_budget")
                                stopped = True
                                break
                            integer(row["revision"], 1)
                            fact = live[row["fact"]]
                            authority.require(scope, fact.repo_id)
                            if row["revision"] != fact.revision:
                                reasons.append("coverage_gap")
                                continue
                            output_bytes += len(fact.content.encode()) + len(
                                fact.metadata_json.encode()
                            )
                            if output_bytes > 8_388_608:
                                reasons.append("output_budget")
                                stopped = True
                                break
                            key = (row["subject"], row["predicate"], row["object"])
                            if key not in edges and len(edges) >= edge_limit:
                                reasons.append("edge_budget")
                                stopped = True
                                break
                            if row["object"] not in visited:
                                if len(visited) >= visited_limit:
                                    reasons.append("visited_budget")
                                    stopped = True
                                    break
                                visited.add(row["object"])
                                queue.append((row["object"], distance + 1))
                            edges.setdefault(key, []).append(
                                ProjectionHit(
                                    fact.fact_id,
                                    1.0,
                                    fact.content,
                                    fact.metadata_json,
                                    fact.revision,
                                    fact.repo_id,
                                    fact.snapshot_id,
                                )
                            )
                    if not edges:
                        visited = set()
            self._fault("before_return")
            self._query_finish(
                authority, scope, source, snapshots, snap, candidate_limit, generation
            )
            reasons = tuple(dict.fromkeys(reasons))
            return QueryResult(
                "partial" if reasons else "complete",
                reasons,
                edges=tuple(
                    GraphEdge(*key, tuple(supports))
                    for key, supports in sorted(edges.items())
                ),
                nodes=tuple(sorted(visited)),
            )
        except ProjectionError as exc:
            authority.require(scope)
            return QueryResult("unavailable", (exc.code,))

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional
from contextlib import contextmanager
import hashlib

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None  # type: ignore

from jitmind.graph.ontology import GraphOntology


class GraphMemoryStore:
    """
    Neo4j-backed graph memory store with proper session management.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        ontology: Optional[GraphOntology] = None,
    ) -> None:
        if GraphDatabase is None:
            raise ImportError("neo4j package is required for GraphMemoryStore. Install with: pip install neo4j")
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database
        self.ontology = ontology
        self._ensure_constraints()

    def close(self) -> None:
        self._driver.close()

    @contextmanager
    def _get_session(self):
        """Context manager for session reuse with proper cleanup."""
        session = self._driver.session(database=self._database)
        try:
            yield session
        finally:
            session.close()

    def _ensure_constraints(self) -> None:
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Semantic) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE",
        ]
        try:
            with self._get_session() as session:
                for q in queries:
                    try:
                        session.run(q)
                    except Exception as e:
                        print(f"[WARN] Failed to create constraint: {e}")
        except Exception as e:
            print(f"[WARN] Failed to connect to Neo4j for constraints: {e}")

    def _now_iso(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def run_cypher(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run an arbitrary Cypher query and return records as dicts."""
        params = params or {}
        try:
            with self._get_session() as session:
                result = session.run(query, **params)
                return [dict(r) for r in result]
        except Exception as e:
            print(f"[WARN] Cypher query failed: {e}")
            return []

    def upsert_memory(self, memory_id: str, props: Dict[str, Any]) -> None:
        query = """
        MERGE (m:Memory {id: $id})
        SET m += $props
        """
        try:
            with self._get_session() as session:
                session.run(query, id=memory_id, props=props)
        except Exception as e:
            print(f"[WARN] Failed to upsert memory {memory_id}: {e}")

    def mark_memory_status(self, memory_id: str, status: str) -> None:
        query = "MATCH (m:Memory {id: $id}) SET m.status = $status"
        try:
            with self._get_session() as session:
                session.run(query, id=memory_id, status=status)
        except Exception as e:
            print(f"[WARN] Failed to mark memory status {memory_id}: {e}")

    def mark_memory_latest(self, memory_id: str, is_latest: bool) -> None:
        query = "MATCH (m:Memory {id: $id}) SET m.is_latest = $is_latest"
        try:
            with self._get_session() as session:
                session.run(query, id=memory_id, is_latest=is_latest)
        except Exception as e:
            print(f"[WARN] Failed to mark memory latest {memory_id}: {e}")

    def link_memory_relation(self, src_id: str, dst_id: str, rel_type: str) -> None:
        if not src_id or not dst_id:
            return
        rtype = rel_type.replace(" ", "_").upper()
        query = f"""
        MATCH (a:Memory {{id: $src}})
        MATCH (b:Memory {{id: $dst}})
        MERGE (a)-[r:{rtype}]->(b)
        SET r.created_at = $ts
        """
        try:
            with self._get_session() as session:
                session.run(query, src=src_id, dst=dst_id, ts=self._now_iso())
        except Exception as e:
            print(f"[WARN] Failed to link memory relation {src_id}->{dst_id}: {e}")

    def add_entities_relations(
        self,
        memory_id: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        t_observed: Optional[str] = None,
        t_valid: Optional[str] = None,
        t_invalid: Optional[str] = None,
    ) -> None:
        if not entities and not relations:
            return
        entity_type_map: Dict[str, str] = {}
        try:
            with self._get_session() as session:
                for e in entities:
                    name = e.get("name")
                    etype = e.get("type") or "Entity"
                    if self.ontology:
                        etype = self.ontology.normalize_entity_type(etype)
                    if not name or not isinstance(name, str) or len(name.strip()) == 0:
                        continue  # Skip invalid entity names
                    name = name.strip()
                    key = f"{etype}:{name}"
                    entity_type_map[name] = etype
                    try:
                        session.run(
                            """
                            MERGE (ent:Entity {key: $key})
                            SET ent.name = $name, ent.type = $type
                            WITH ent
                            MATCH (m:Memory {id: $mid})
                            MERGE (m)-[:MENTIONS]->(ent)
                            """,
                            key=key,
                            name=name,
                            type=etype,
                            mid=memory_id,
                        )
                    except Exception as e:
                        print(f"[WARN] Failed to add entity {name}: {e}")

                for r in relations:
                    head = r.get("head")
                    tail = r.get("tail")
                    rel_type = r.get("relation") or "RELATED_TO"
                    if self.ontology:
                        rel_type = self.ontology.normalize_relation_type(rel_type)
                    if not head or not tail:
                        continue
                    if not isinstance(head, str) or not isinstance(tail, str):
                        continue  # Skip invalid relations
                    head = head.strip()
                    tail = tail.strip()
                    if not head or not tail:
                        continue
                    # Sanitize relation type (Neo4j doesn't allow spaces in rel types)
                    rel_type = rel_type.replace(" ", "_").upper()
                    h_type = entity_type_map.get(head, "Entity")
                    t_type = entity_type_map.get(tail, "Entity")
                    h_key = f"{h_type}:{head}"
                    t_key = f"{t_type}:{tail}"
                    try:
                        session.run(
                            """
                            MERGE (h:Entity {key: $h_key})
                            SET h.name = $head, h.type = $h_type
                            MERGE (t:Entity {key: $t_key})
                            SET t.name = $tail, t.type = $t_type
                            MERGE (h)-[r:RELATION {type: $rel_type}]->(t)
                            SET r.source_memory_id = $mid,
                                r.t_observed = $t_observed,
                                r.t_valid = $t_valid,
                                r.t_invalid = $t_invalid
                            """,
                            h_key=h_key,
                            t_key=t_key,
                            head=head,
                            tail=tail,
                            rel_type=rel_type,
                            mid=memory_id,
                            h_type=h_type,
                            t_type=t_type,
                            t_observed=t_observed or r.get("t_observed"),
                            t_valid=t_valid or r.get("t_valid"),
                            t_invalid=t_invalid or r.get("t_invalid"),
                        )
                    except Exception as e:
                        print(f"[WARN] Failed to add relation {head}->{tail}: {e}")
        except Exception as e:
            print(f"[WARN] Failed to add entities/relations for memory {memory_id}: {e}")

    # ---- 3-tier graph architecture ----
    def add_episode(
        self,
        memory_id: str,
        content: str,
        timestamp: Optional[str] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Add a timestamped Episode node linked to a Memory entry."""
        episode_id = f"episode:{memory_id}"
        try:
            with self._get_session() as session:
                session.run(
                    """
                    MERGE (e:Episode {id: $eid})
                    SET e.content = $content,
                        e.timestamp = $ts
                    WITH e
                    MATCH (m:Memory {id: $mid})
                    MERGE (m)-[:HAS_EPISODE]->(e)
                    """,
                    eid=episode_id,
                    mid=memory_id,
                    content=content,
                    ts=timestamp,
                )
                # Link Episode -> Entity (optional)
                for ent in entities or []:
                    name = ent.get("name")
                    etype = ent.get("type") or "Entity"
                    if not name or not isinstance(name, str):
                        continue
                    name = name.strip()
                    if not name:
                        continue
                    key = f"{etype}:{name}"
                    session.run(
                        """
                        MERGE (en:Entity {key: $key})
                        SET en.name = $name, en.type = $type
                        WITH en
                        MATCH (e:Episode {id: $eid})
                        MERGE (e)-[:MENTIONS]->(en)
                        """,
                        key=key,
                        name=name,
                        type=etype,
                        eid=episode_id,
                    )
        except Exception as e:
            print(f"[WARN] Failed to add episode for memory {memory_id}: {e}")
            return None
        return episode_id

    def add_semantic_fact(
        self,
        head: str,
        relation: str,
        tail: str,
        memory_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        t_valid: Optional[str] = None,
        t_invalid: Optional[str] = None,
        head_type: Optional[str] = None,
        tail_type: Optional[str] = None,
    ) -> Optional[str]:
        """Add a Semantic node representing a (head, relation, tail) fact."""
        if not head or not tail:
            return None
        head = head.strip()
        tail = tail.strip()
        if not head or not tail:
            return None
        rel = (relation or "RELATED_TO").strip()
        if self.ontology:
            rel = self.ontology.normalize_relation_type(rel)
        fact = f"{head} {rel} {tail}"
        sid = self._semantic_id(fact)
        h_type = self.ontology.normalize_entity_type(head_type) if self.ontology else (head_type or "Entity")
        t_type = self.ontology.normalize_entity_type(tail_type) if self.ontology else (tail_type or "Entity")
        h_key = f"{h_type}:{head}"
        t_key = f"{t_type}:{tail}"
        try:
            with self._get_session() as session:
                session.run(
                    """
                    MERGE (s:Semantic {id: $sid})
                    SET s.fact = $fact,
                        s.relation = $rel,
                        s.t_observed = $ts,
                        s.t_valid = $t_valid,
                        s.t_invalid = $t_invalid
                    """,
                    sid=sid,
                    fact=fact,
                    rel=rel,
                    ts=timestamp,
                    t_valid=t_valid,
                    t_invalid=t_invalid,
                )
                # Ensure entity nodes exist and link
                session.run(
                    """
                    MERGE (h:Entity {key: $h_key})
                    SET h.name = $head, h.type = $h_type
                    MERGE (t:Entity {key: $t_key})
                    SET t.name = $tail, t.type = $t_type
                    WITH h, t
                    MATCH (s:Semantic {id: $sid})
                    MERGE (s)-[:ABOUT]->(h)
                    MERGE (s)-[:ABOUT]->(t)
                    """,
                    h_key=h_key,
                    t_key=t_key,
                    head=head,
                    tail=tail,
                    h_type=h_type,
                    t_type=t_type,
                    sid=sid,
                )
                if memory_id:
                    session.run(
                        """
                        MATCH (m:Memory {id: $mid})
                        MATCH (s:Semantic {id: $sid})
                        MERGE (m)-[:HAS_SEMANTIC]->(s)
                        """,
                        mid=memory_id,
                        sid=sid,
                    )
        except Exception as e:
            print(f"[WARN] Failed to add semantic fact {fact}: {e}")
            return None
        return sid

    def add_semantic_statement(
        self,
        content: str,
        entities: List[Dict[str, Any]],
        memory_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Optional[str]:
        """Add a Semantic node for a free-form statement linked to entities."""
        if not content:
            return None
        sid = self._semantic_id(content)
        try:
            with self._get_session() as session:
                session.run(
                    """
                    MERGE (s:Semantic {id: $sid})
                    SET s.fact = $fact,
                        s.relation = 'STATEMENT',
                        s.t_observed = $ts
                    """,
                    sid=sid,
                    fact=content,
                    ts=timestamp,
                )
                if memory_id:
                    session.run(
                        """
                        MATCH (m:Memory {id: $mid})
                        MATCH (s:Semantic {id: $sid})
                        MERGE (m)-[:HAS_SEMANTIC]->(s)
                        """,
                        mid=memory_id,
                        sid=sid,
                    )
                for ent in entities:
                    name = ent.get("name")
                    etype = ent.get("type") or "Entity"
                    if not name or not isinstance(name, str):
                        continue
                    name = name.strip()
                    if not name:
                        continue
                    key = f"{etype}:{name}"
                    session.run(
                        """
                        MERGE (e:Entity {key: $key})
                        SET e.name = $name, e.type = $type
                        WITH e
                        MATCH (s:Semantic {id: $sid})
                        MERGE (s)-[:ABOUT]->(e)
                        """,
                        key=key,
                        name=name,
                        type=etype,
                        sid=sid,
                    )
        except Exception as e:
            print(f"[WARN] Failed to add semantic statement: {e}")
            return None
        return sid

    def add_community_summary(
        self,
        community_id: str,
        summary: str,
        member_ids: Optional[List[str]] = None,
        level: Optional[int] = None,
    ) -> None:
        """Add a Community node and link to member Memory nodes."""
        if not community_id:
            return
        try:
            with self._get_session() as session:
                session.run(
                    """
                    MERGE (c:Community {id: $cid})
                    SET c.summary = $summary,
                        c.level = $level
                    """,
                    cid=community_id,
                    summary=summary,
                    level=level,
                )
                # Link summary Memory node (if it exists) to its Community node
                session.run(
                    """
                    MATCH (c:Community {id: $cid})
                    OPTIONAL MATCH (m:Memory {id: $cid})
                    FOREACH (_ IN CASE WHEN m IS NULL THEN [] ELSE [1] END |
                      MERGE (m)-[:HAS_COMMUNITY]->(c)
                    )
                    """,
                    cid=community_id,
                )
                for mid in member_ids or []:
                    if not mid:
                        continue
                    session.run(
                        """
                        MATCH (c:Community {id: $cid})
                        MATCH (m:Memory {id: $mid})
                        MERGE (c)-[:HAS_MEMBER]->(m)
                        """,
                        cid=community_id,
                        mid=mid,
                    )
                    # Also encode explicit semantics on the Memory graph:
                    # the summary/consolidated Memory (id == community_id) derives from member memories.
                    session.run(
                        """
                        MATCH (parent:Memory {id: $cid})
                        MATCH (child:Memory {id: $mid})
                        MERGE (parent)-[:DERIVES]->(child)
                        """,
                        cid=community_id,
                        mid=mid,
                    )
        except Exception as e:
            print(f"[WARN] Failed to add community summary {community_id}: {e}")

    # ---- Graph CRUD API ----
    def upsert_entity(self, name: str, etype: str = "Entity", props: Optional[Dict[str, Any]] = None) -> None:
        if not name:
            return
        name = name.strip()
        if not name:
            return
        etype = self.ontology.normalize_entity_type(etype) if self.ontology else etype
        key = f"{etype}:{name}"
        props = props or {}
        props.update({"name": name, "type": etype})
        try:
            with self._get_session() as session:
                session.run(
                    """
                    MERGE (e:Entity {key: $key})
                    SET e += $props
                    """,
                    key=key,
                    props=props,
                )
        except Exception as e:
            print(f"[WARN] Failed to upsert entity {name}: {e}")

    def delete_entity(self, name: str, etype: str = "Entity") -> None:
        if not name:
            return
        name = name.strip()
        if not name:
            return
        etype = self.ontology.normalize_entity_type(etype) if self.ontology else etype
        key = f"{etype}:{name}"
        try:
            with self._get_session() as session:
                session.run("MATCH (e:Entity {key: $key}) DETACH DELETE e", key=key)
        except Exception as e:
            print(f"[WARN] Failed to delete entity {name}: {e}")

    def upsert_relation(
        self,
        head: str,
        relation: str,
        tail: str,
        props: Optional[Dict[str, Any]] = None,
        head_type: str = "Entity",
        tail_type: str = "Entity",
    ) -> None:
        if not head or not tail:
            return
        head = head.strip()
        tail = tail.strip()
        if not head or not tail:
            return
        # Keep entity-entity edges uniform: relationship type is always `RELATION`,
        # with the semantic label stored in the `type` property. This simplifies
        # traversal and avoids schema explosion in Neo4j.
        rel_type = self.ontology.normalize_relation_type(relation) if self.ontology else relation
        rel_type = (rel_type or "RELATED_TO").replace(" ", "_").upper()
        props = props or {}
        h_type = self.ontology.normalize_entity_type(head_type) if self.ontology else (head_type or "Entity")
        t_type = self.ontology.normalize_entity_type(tail_type) if self.ontology else (tail_type or "Entity")
        h_key = f"{h_type}:{head}"
        t_key = f"{t_type}:{tail}"

        try:
            with self._get_session() as session:
                session.run(
                    """
                    MERGE (h:Entity {key: $h_key})
                    SET h.name = $head, h.type = $h_type
                    MERGE (t:Entity {key: $t_key})
                    SET t.name = $tail, t.type = $t_type
                    MERGE (h)-[r:RELATION {type: $rel_type}]->(t)
                    SET r += $props
                    """,
                    h_key=h_key,
                    t_key=t_key,
                    head=head,
                    tail=tail,
                    h_type=h_type,
                    t_type=t_type,
                    rel_type=rel_type,
                    props=props,
                )
        except Exception as e:
            print(f"[WARN] Failed to upsert relation {head}-{rel_type}-{tail}: {e}")

    def delete_relation(
        self,
        head: str,
        relation: str,
        tail: str,
        head_type: str = "Entity",
        tail_type: str = "Entity",
    ) -> None:
        if not head or not tail:
            return
        head = head.strip()
        tail = tail.strip()
        if not head or not tail:
            return
        rel_type = self.ontology.normalize_relation_type(relation) if self.ontology else relation
        rel_type = (rel_type or "RELATED_TO").replace(" ", "_").upper()
        h_type = self.ontology.normalize_entity_type(head_type) if self.ontology else (head_type or "Entity")
        t_type = self.ontology.normalize_entity_type(tail_type) if self.ontology else (tail_type or "Entity")
        try:
            with self._get_session() as session:
                session.run(
                    """
                    MATCH (h:Entity {key: $h_key})-[r:RELATION {type: $rel_type}]->(t:Entity {key: $t_key})
                    DELETE r
                    """,
                    h_key=f"{h_type}:{head}",
                    t_key=f"{t_type}:{tail}",
                    rel_type=rel_type,
                )
        except Exception as e:
            print(f"[WARN] Failed to delete relation {head}-{rel_type}-{tail}: {e}")

    def delete_memory(self, memory_id: str) -> None:
        if not memory_id:
            return
        try:
            with self._get_session() as session:
                session.run("MATCH (m:Memory {id: $id}) DETACH DELETE m", id=memory_id)
        except Exception as e:
            print(f"[WARN] Failed to delete memory {memory_id}: {e}")

    def delete_episode(self, episode_id: str) -> None:
        if not episode_id:
            return
        try:
            with self._get_session() as session:
                session.run("MATCH (e:Episode {id: $id}) DETACH DELETE e", id=episode_id)
        except Exception as e:
            print(f"[WARN] Failed to delete episode {episode_id}: {e}")

    def delete_semantic(self, semantic_id: str) -> None:
        if not semantic_id:
            return
        try:
            with self._get_session() as session:
                session.run("MATCH (s:Semantic {id: $id}) DETACH DELETE s", id=semantic_id)
        except Exception as e:
            print(f"[WARN] Failed to delete semantic {semantic_id}: {e}")

    def delete_community(self, community_id: str) -> None:
        if not community_id:
            return
        try:
            with self._get_session() as session:
                session.run("MATCH (c:Community {id: $id}) DETACH DELETE c", id=community_id)
        except Exception as e:
            print(f"[WARN] Failed to delete community {community_id}: {e}")

    def query_memories(self, entity_names: List[str], depth: int = 1, limit: int = 20) -> List[Dict[str, Any]]:
        if not entity_names:
            return []
        # Filter out invalid entity names
        valid_names = [n.strip() for n in entity_names if isinstance(n, str) and n.strip()]
        if not valid_names:
            return []

        results: Dict[str, Dict[str, Any]] = {}
        try:
            with self._get_session() as session:
                direct = session.run(
                    """
                    MATCH (m:Memory)-[:MENTIONS]->(e:Entity)
                    WHERE e.name IN $names AND (m.status IS NULL OR m.status = 'active')
                    RETURN m.id AS id, m.content AS content, m.source_page_id AS page_id
                    LIMIT $limit
                    """,
                    names=valid_names,
                    limit=limit,
                )
                for r in direct:
                    if r["id"] is None:
                        continue
                    results[r["id"]] = {
                        "id": r["id"],
                        "content": r["content"],
                        "page_id": r.get("page_id"),
                        "score": 1.0,
                    }

                # Use parameterized depth with a safe maximum
                safe_depth = min(depth, 3)  # Limit depth to prevent expensive queries
                related = session.run(
                    f"""
                    MATCH (e:Entity)
                    WHERE e.name IN $names
                    MATCH (e)-[:RELATION*1..{safe_depth}]-(e2:Entity)<-[:MENTIONS]-(m:Memory)
                    WHERE m.status IS NULL OR m.status = 'active'
                    RETURN DISTINCT m.id AS id, m.content AS content, m.source_page_id AS page_id
                    LIMIT $limit
                    """,
                    names=valid_names,
                    limit=limit,
                )
                for r in related:
                    mid = r["id"]
                    if mid is None or mid in results:
                        continue
                    results[mid] = {
                        "id": mid,
                        "content": r["content"],
                        "page_id": r.get("page_id"),
                        "score": 0.5,
                    }
        except Exception as e:
            print(f"[WARN] Failed to query graph memories: {e}")
        return list(results.values())

    # ---- HippoRAG: Personalized PageRank ----
    def personalized_pagerank(
        self,
        entity_names: List[str],
        damping: float = 0.85,
        depth: int = 2,
        max_iter: int = 20,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        if not entity_names:
            return []
        valid_names = [n.strip() for n in entity_names if isinstance(n, str) and n.strip()]
        if not valid_names:
            return []

        # Build subgraph edges
        edges: List[tuple[str, str]] = []
        node_types: Dict[str, str] = {}
        try:
            with self._get_session() as session:
                safe_depth = min(depth, 3)
                records = session.run(
                    f"""
                    MATCH (seed:Entity)
                    WHERE seed.name IN $names
                    // Traverse any relationship types but restrict traversal to our known memory-graph labels.
                    // This avoids Neo4j warnings for relationship types that aren't present yet (e.g. HAS_MEMBER).
                    MATCH p=(seed)-[*1..{safe_depth}]-(n)
                    WHERE ALL(x IN nodes(p) WHERE x:Entity OR x:Memory OR x:Episode OR x:Semantic OR x:Community)
                    WITH relationships(p) AS rels
                    UNWIND rels AS r
                    WITH startNode(r) AS s, endNode(r) AS t
                    RETURN DISTINCT s AS s, t AS t
                    LIMIT 1000
                    """,
                    names=valid_names,
                )
                for rec in records:
                    s = rec["s"]
                    t = rec["t"]
                    s_key, s_type = self._node_key(s)
                    t_key, t_type = self._node_key(t)
                    if not s_key or not t_key:
                        continue
                    node_types[s_key] = s_type
                    node_types[t_key] = t_type
                    edges.append((s_key, t_key))
        except Exception as e:
            print(f"[WARN] Failed to build subgraph for PPR: {e}")
            return []

        if not edges:
            return []

        # Build adjacency
        adj: Dict[str, set[str]] = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # Seed nodes (Entity labels)
        seeds = [k for k, t in node_types.items() if t == "Entity" and any(n in k for n in valid_names)]
        if not seeds:
            return []

        pr: Dict[str, float] = {n: 0.0 for n in adj.keys()}
        seed_weight = 1.0 / len(seeds)
        for s in seeds:
            pr[s] = seed_weight

        for _ in range(max_iter):
            new_pr = {n: 0.0 for n in adj.keys()}
            for n in pr:
                new_pr[n] += (1.0 - damping) * (seed_weight if n in seeds else 0.0)
            for n, neighbors in adj.items():
                if not neighbors:
                    continue
                share = pr[n] * damping / len(neighbors)
                for nb in neighbors:
                    new_pr[nb] += share
            pr = new_pr

        # Collect memory nodes
        memory_scores = [(n, score) for n, score in pr.items() if node_types.get(n) == "Memory"]
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        top_ids = [mid for mid, _ in memory_scores[:limit]]
        if not top_ids:
            return []

        # Fetch memory content
        results: List[Dict[str, Any]] = []
        try:
            with self._get_session() as session:
                rows = session.run(
                    """
                    MATCH (m:Memory)
                    WHERE m.id IN $ids AND (m.status IS NULL OR m.status = 'active')
                    RETURN m.id AS id, m.content AS content, m.source_page_id AS page_id
                    """,
                    ids=top_ids,
                )
                mem_map = {r["id"]: r for r in rows}
            for mid, score in memory_scores[:limit]:
                row = mem_map.get(mid)
                if not row:
                    continue
                results.append({
                    "id": mid,
                    "content": row.get("content"),
                    "page_id": row.get("page_id"),
                    "score": score,
                })
        except Exception as e:
            print(f"[WARN] Failed to fetch PPR memories: {e}")
        return results

    def _node_key(self, node) -> tuple[Optional[str], str]:
        try:
            labels = list(node.labels)
            if "Memory" in labels:
                return node.get("id"), "Memory"
            if "Episode" in labels:
                return node.get("id"), "Episode"
            if "Semantic" in labels:
                return node.get("id"), "Semantic"
            if "Community" in labels:
                return node.get("id"), "Community"
            if "Entity" in labels:
                name = node.get("name")
                key = node.get("key") or (f"Entity:{name}" if name else None)
                return key, "Entity"
            return node.get("id") or node.get("name"), "Other"
        except Exception:
            return None, "Other"

    def _semantic_id(self, fact: str) -> str:
        return hashlib.sha256(fact.lower().encode("utf-8")).hexdigest()

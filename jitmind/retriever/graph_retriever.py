# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Any, List, Optional

from jitmind.retriever.base import AbsRetriever
from jitmind.schemas import Hit

try:
    from jitmind.graph import GraphMemoryStore
except ImportError:
    GraphMemoryStore = None  # type: ignore


class GraphRetriever(AbsRetriever):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if GraphMemoryStore is None:
            raise ImportError("neo4j package is required for GraphRetriever. Install with: pip install neo4j")
        self.graph_store: Optional[GraphMemoryStore] = config.get("graph_store")
        self.depth = config.get("depth", 1)
        self.top_k = config.get("top_k", 10)
        self.use_ppr = config.get("use_ppr", True)

    def build(self, page_store):
        # Graph store is updated by MemoryAgent; no build needed.
        return None

    def load(self):
        return None

    def update(self, page_store):
        return None

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        results_all: List[List[Hit]] = []
        for q in query_list:
            if not self.graph_store:
                results_all.append([])
                continue
            entity_names = [e.strip() for e in q.split(",") if e.strip()]
            if self.use_ppr:
                rows = self.graph_store.personalized_pagerank(entity_names, depth=self.depth, limit=top_k)
            else:
                rows = self.graph_store.query_memories(entity_names, depth=self.depth, limit=top_k)
            hits: List[Hit] = []
            for rank, row in enumerate(rows):
                hits.append(
                    Hit(
                        page_id=row.get("page_id"),
                        snippet=row.get("content", ""),
                        source="graph",
                        meta={"rank": rank, "score": row.get("score", 0.0)},
                    )
                )
            results_all.append(hits)
        return results_all

    def search_temporal(
        self,
        query_list: List[str],
        *,
        valid_at: str | None = None,
        observed_at: str | None = None,
        relation_types: List[str] | None = None,
        top_k: int = 10,
    ) -> List[List[Hit]]:
        """Search versioned graph facts and preserve their temporal evidence."""

        if not self.graph_store:
            return [[] for _ in query_list]
        results_all: List[List[Hit]] = []
        for query in query_list:
            names = [name.strip() for name in query.split(",") if name.strip()]
            facts = self.graph_store.query_facts(
                names,
                relation_types=relation_types,
                valid_at=valid_at,
                observed_at=observed_at,
                limit=top_k,
            )
            results_all.append(
                [
                    Hit(
                        page_id=fact.source_page_id,
                        snippet=fact.statement,
                        source="graph_fact",
                        meta={
                            "fact_id": fact.id,
                            "memory_id": fact.source_memory_id,
                            "relation": fact.relation,
                            "t_observed": fact.t_observed,
                            "t_valid": fact.t_valid,
                            "t_invalid": fact.t_invalid,
                        },
                    )
                    for fact in facts
                ]
            )
        return results_all

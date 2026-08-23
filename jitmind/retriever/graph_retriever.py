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
        self.namespace = config.get("namespace")

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
                rows = self.graph_store.personalized_pagerank(
                    entity_names,
                    depth=self.depth,
                    limit=top_k,
                    namespace=self.namespace,
                )
            else:
                rows = self.graph_store.query_memories(
                    entity_names,
                    depth=self.depth,
                    limit=top_k,
                    namespace=self.namespace,
                )
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

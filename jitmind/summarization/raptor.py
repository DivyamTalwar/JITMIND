# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Dict, Any, Optional
import os
import numpy as np

try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    GaussianMixture = None  # type: ignore

try:
    import cohere  # type: ignore
except ImportError:  # pragma: no cover
    cohere = None  # type: ignore

from jitmind.schemas import MemoryEntry, AdvancedMemoryStore
from jitmind.generator import AbsGenerator
from jitmind.prompts import HIERARCHICAL_SUMMARY_PROMPT
from jitmind.utils.retry import RetryConfig, retry_call


class HierarchicalSummarizer:
    """
    RAPTOR-style hierarchical summarization over memory entries.
    Level 0: raw entries
    Level 1: cluster summaries (GMM)
    Level 2: summaries of summaries
    Level 3: global summary
    """

    def __init__(
        self,
        generator: AbsGenerator,
        memory_store: AdvancedMemoryStore,
        api_key: Optional[str] = None,
        base_url: str = "https://api.cohere.com",
        embed_model: str = "embed-v4.0",
        input_type: str = "search_document",
        max_level1_clusters: int = 10,
        max_level2_clusters: int = 5,
    ) -> None:
        if GaussianMixture is None:
            raise ImportError("scikit-learn is required for HierarchicalSummarizer. Install scikit-learn.")
        if cohere is None:
            raise ImportError("cohere is required for HierarchicalSummarizer. Install with: pip install cohere")
        self.generator = generator
        self.memory_store = memory_store
        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError("Cohere API key required for summarization embeddings")
        self.client = cohere.Client(key, base_url=base_url)
        self.embed_model = embed_model
        self.input_type = input_type
        self.max_level1_clusters = max_level1_clusters
        self.max_level2_clusters = max_level2_clusters

    def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        retry_cfg = RetryConfig(max_attempts=4, base_delay_s=1.0, max_delay_s=10.0, jitter_s=0.3)

        def _call():
            return self.client.embed(
                texts=texts,
                model=self.embed_model,
                input_type=self.input_type,
            )

        resp = retry_call(_call, config=retry_cfg)
        embeddings = getattr(resp, "embeddings", None) or resp["embeddings"]
        emb = np.array(embeddings, dtype=np.float32)
        # Defensive: providers can occasionally return NaN/inf; sanitize and normalize.
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return emb / norms

    def _cluster(self, embeddings: np.ndarray, max_clusters: int) -> List[List[int]]:
        n = embeddings.shape[0]
        if n <= 2:
            return [[i] for i in range(n)]
        k = min(max_clusters, max(2, int(np.sqrt(n))))
        try:
            # `diag` + small `reg_covar` is much more stable for high-dim embeddings than `full`.
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                reg_covar=1e-6,
                random_state=42,
                n_init=3,
                max_iter=200,
            )
            labels = gmm.fit_predict(np.asarray(embeddings, dtype=np.float64))
        except Exception:
            # Fall back to singleton clusters instead of failing maintenance.
            return [[i] for i in range(n)]
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(idx)
        return list(clusters.values())

    def _summarize_cluster(self, texts: List[str]) -> str:
        joined = "\n".join(texts)
        prompt = HIERARCHICAL_SUMMARY_PROMPT.format(cluster_content=joined)
        response = self.generator.generate_single(prompt=prompt)
        return (response.get("text") or "").strip()

    def build_summaries(self, limit: int = 200, include_existing_summaries: bool = False) -> List[MemoryEntry]:
        entries = [
            e for e in self.memory_store.get_entries()
            if e.status == "active" and (include_existing_summaries or not e.meta.get("summary_level"))
        ]
        if not entries:
            return []
        entries = entries[:limit]

        level0_texts = [e.content for e in entries]
        level0_ids = [e.id for e in entries]
        emb0 = self._embed(level0_texts)
        clusters_lvl1 = self._cluster(emb0, self.max_level1_clusters)

        level1_entries: List[MemoryEntry] = []
        for cidx, cluster in enumerate(clusters_lvl1):
            texts = [level0_texts[i] for i in cluster]
            summary = self._summarize_cluster(texts)
            if not summary:
                continue
            level1_entries.append(
                MemoryEntry(
                    content=summary,
                    tier="mid",
                    meta={"summary_level": 1, "children": [level0_ids[i] for i in cluster]},
                )
            )

        # Level 2
        level2_entries: List[MemoryEntry] = []
        if level1_entries:
            lvl1_texts = [e.content for e in level1_entries]
            lvl1_ids = [e.id for e in level1_entries]
            emb1 = self._embed(lvl1_texts)
            clusters_lvl2 = self._cluster(emb1, self.max_level2_clusters)
            for cluster in clusters_lvl2:
                texts = [lvl1_texts[i] for i in cluster]
                summary = self._summarize_cluster(texts)
                if not summary:
                    continue
                level2_entries.append(
                    MemoryEntry(
                        content=summary,
                        tier="long",
                        meta={"summary_level": 2, "children": [lvl1_ids[i] for i in cluster]},
                    )
                )

        # Level 3 (global)
        level3_entries: List[MemoryEntry] = []
        if level2_entries:
            texts = [e.content for e in level2_entries]
            summary = self._summarize_cluster(texts)
            if summary:
                level3_entries.append(
                    MemoryEntry(
                        content=summary,
                        tier="long",
                        meta={"summary_level": 3, "children": [e.id for e in level2_entries]},
                    )
                )

        return level1_entries + level2_entries + level3_entries

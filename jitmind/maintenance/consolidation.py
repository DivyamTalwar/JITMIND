# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Optional, Dict, Any
import os
import numpy as np
try:
    import cohere  # type: ignore
except ImportError:  # pragma: no cover
    cohere = None  # type: ignore

from jitmind.schemas import AdvancedMemoryStore, MemoryEntry
from jitmind.prompts import CONSOLIDATION_PROMPT, ConflictCheck_PROMPT
from jitmind.generator import AbsGenerator
from jitmind.utils.retry import RetryConfig, retry_call


class MemoryConsolidator:
    """
    Sleep-time consolidation:
    - cluster similar short-term memories
    - merge/deduplicate into consolidated entries
    - resolve contradictions (newer wins)
    """

    def __init__(
        self,
        generator: AbsGenerator,
        memory_store: AdvancedMemoryStore,
        api_key: Optional[str] = None,
        base_url: str = "https://api.cohere.com",
        embed_model: str = "embed-v4.0",
        similarity_threshold: float = 0.85,
        max_cluster_size: int = 10,
    ) -> None:
        if cohere is None:
            raise ImportError("cohere is required for MemoryConsolidator. Install with: pip install cohere")
        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError("Cohere API key required for consolidation embeddings")
        self.client = cohere.Client(key, base_url=base_url)
        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold
        self.max_cluster_size = max_cluster_size
        self.memory_store = memory_store
        self.generator = generator

    def _embed(self, texts: List[str]) -> np.ndarray:
        retry_cfg = RetryConfig(max_attempts=4, base_delay_s=1.0, max_delay_s=10.0, jitter_s=0.3)

        def _call():
            return self.client.embed(
                texts=texts,
                model=self.embed_model,
                input_type="search_document",
            )

        resp = retry_call(_call, config=retry_cfg)
        embeddings = getattr(resp, "embeddings", None) or resp["embeddings"]
        emb = np.array(embeddings, dtype=np.float32)
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return emb / norms

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def _is_contradictory(self, existing_fact: str, new_fact: str) -> bool:
        prompt = ConflictCheck_PROMPT.format(existing_fact=existing_fact, new_fact=new_fact)
        try:
            response = self.generator.generate_single(prompt=prompt)
            text = response.get("text", "")
            import json
            data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            return bool(data.get("contradict", False))
        except Exception:
            return False

    def consolidate(self, limit: int = 200) -> List[MemoryEntry]:
        entries = [e for e in self.memory_store.get_entries() if e.status == "active" and e.tier == "short"]
        if not entries:
            return []
        entries = entries[:limit]

        texts = [e.content for e in entries]
        emb = self._embed(texts)

        used = set()
        new_entries: List[MemoryEntry] = []

        for i, entry in enumerate(entries):
            if i in used:
                continue
            cluster = [i]
            used.add(i)
            for j in range(i + 1, len(entries)):
                if j in used:
                    continue
                if self._cosine_sim(emb[i], emb[j]) >= self.similarity_threshold:
                    cluster.append(j)
                    used.add(j)
                if len(cluster) >= self.max_cluster_size:
                    break

            if len(cluster) == 1:
                continue

            cluster_texts = [texts[idx] for idx in cluster]
            prompt = CONSOLIDATION_PROMPT.format(cluster_content="\n".join(cluster_texts))
            response = self.generator.generate_single(prompt=prompt)
            merged = (response.get("text") or "").strip()
            if not merged:
                continue

            new_entry = MemoryEntry(
                content=merged,
                tier="mid",
                meta={"consolidated_from": [entries[idx].id for idx in cluster]},
            )
            self.memory_store.add_entry(new_entry)
            new_entries.append(new_entry)

            # Supersede old entries; resolve contradictions by keeping newest
            for idx in cluster:
                old_entry = entries[idx]
                if self._is_contradictory(old_entry.content, merged):
                    self.memory_store.supersede_entry(old_entry.id, t_invalid=new_entry.t_observed)
                else:
                    self.memory_store.supersede_entry(old_entry.id, t_invalid=new_entry.t_observed)

        return new_entries

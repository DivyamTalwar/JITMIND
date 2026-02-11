# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple, Optional
import cohere

from jitmind.utils.retry import RetryConfig, retry_call

class CohereReranker:
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get("model_name") or os.getenv("COHERE_RERANK_MODEL", "rerank-v4.0-pro")
        self.top_k = config.get("top_k", 20)
        api_key = config.get("api_key") or os.getenv("COHERE_API_KEY")
        base_url = config.get("base_url") or os.getenv("COHERE_BASE_URL", "https://api.cohere.com")
        if not api_key:
            raise ValueError("Cohere API key is required (config.api_key or COHERE_API_KEY)")
        self.client = cohere.Client(api_key, base_url=base_url)

    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        if not documents:
            return []
        k = top_k or self.top_k
        retry_cfg = RetryConfig(max_attempts=4, base_delay_s=1.0, max_delay_s=10.0, jitter_s=0.3)

        def _call():
            return self.client.rerank(
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=min(k, len(documents))
            )

        resp = retry_call(_call, config=retry_cfg)
        results = getattr(resp, "results", None) or resp["results"]
        ranked: List[Tuple[int, float]] = []
        for r in results:
            idx = getattr(r, "index", None) if hasattr(r, "index") else r.get("index")
            score = getattr(r, "relevance_score", None) if hasattr(r, "relevance_score") else r.get("relevance_score")
            if idx is None or score is None:
                continue
            ranked.append((int(idx), float(score)))
        return ranked

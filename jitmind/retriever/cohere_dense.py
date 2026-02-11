# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import numpy as np
from typing import Dict, Any, List
import faiss
import cohere

from jitmind.retriever.base import AbsRetriever
from jitmind.schemas import InMemoryPageStore, Hit, Page
from jitmind.utils.retry import RetryConfig, retry_call


def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    embeddings_normalized = embeddings.copy()
    faiss.normalize_L2(embeddings_normalized)
    index.add(embeddings_normalized)
    return index


class CohereDenseRetriever(AbsRetriever):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.index = None
        self.pages: List[Page] = []
        self.doc_emb = None
        self.model_name = config.get("model_name") or os.getenv("COHERE_EMBED_MODEL", "embed-v4.0")
        self.input_type_doc = config.get("input_type_doc") or os.getenv("COHERE_EMBED_INPUT_TYPE_DOC", "search_document")
        self.input_type_query = config.get("input_type_query") or os.getenv("COHERE_EMBED_INPUT_TYPE_QUERY", "search_query")
        self.batch_size = config.get("batch_size", 32)
        self.max_length = config.get("max_length", 512)
        self.index_dir = config.get("index_dir", "./index/cohere_dense")

        api_key = config.get("api_key") or os.getenv("COHERE_API_KEY")
        base_url = config.get("base_url") or os.getenv("COHERE_BASE_URL", "https://api.cohere.com")
        if not api_key:
            raise ValueError("Cohere API key is required (config.api_key or COHERE_API_KEY)")
        self.client = cohere.Client(api_key, base_url=base_url)

    def _pages_dir(self) -> str:
        return os.path.join(self.index_dir, "pages")

    def _emb_path(self) -> str:
        return os.path.join(self.index_dir, "doc_emb.npy")

    def _encode(self, texts: List[str], input_type: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        all_embs = []
        retry_cfg = RetryConfig(max_attempts=4, base_delay_s=1.0, max_delay_s=10.0, jitter_s=0.3)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            def _call():
                return self.client.embed(
                    texts=batch,
                    model=self.model_name,
                    input_type=input_type,
                )
            resp = retry_call(_call, config=retry_cfg)
            embeddings = getattr(resp, "embeddings", None) or resp["embeddings"]
            emb = np.array(embeddings, dtype=np.float32)
            emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            all_embs.append(emb)
        out = np.vstack(all_embs)
        return out

    def load(self) -> None:
        try:
            self.doc_emb = np.load(self._emb_path())
            self.index = _build_faiss_index(self.doc_emb)
            self.pages = InMemoryPageStore.load(self._pages_dir()).load()
        except Exception as e:
            print("CohereDenseRetriever.load() failed, will need build():", e)

    def build(self, page_store: InMemoryPageStore):
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self._pages_dir(), exist_ok=True)
        self.pages = page_store.load()
        texts = [p.content for p in self.pages]
        self.doc_emb = self._encode(texts, input_type=self.input_type_doc)
        self.index = _build_faiss_index(self.doc_emb)
        np.save(self._emb_path(), self.doc_emb)
        temp_store = InMemoryPageStore(dir_path=self._pages_dir())
        temp_store.save(self.pages)

    def update(self, page_store: InMemoryPageStore):
        self.build(page_store)

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        if self.index is None:
            self.load()
        if self.index is None:
            return []
        query_emb = self._encode(query_list, input_type=self.input_type_query)
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, top_k)

        results_all: List[List[Hit]] = []
        for q_scores, q_indices in zip(scores, indices):
            hits: List[Hit] = []
            for rank, (score, idx) in enumerate(zip(q_scores, q_indices)):
                if idx < 0 or idx >= len(self.pages):
                    continue
                page = self.pages[int(idx)]
                hits.append(
                    Hit(
                        page_id=str(idx),
                        snippet=page.content,
                        source="vector",
                        meta={"rank": rank, "score": float(score)}
                    )
                )
            results_all.append(hits)
        return results_all

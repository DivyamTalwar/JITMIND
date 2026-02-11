import os
import json
import numpy as np
import requests
from typing import Dict, Any, List, Optional
from FlagEmbedding import FlagAutoModel
import faiss

from jitmind.retriever.base import AbsRetriever
from jitmind.schemas import InMemoryPageStore, Hit, Page


def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index.
    embeddings: numpy array (n, dim)
    """
    dimension = embeddings.shape[1]
    # Use inner product index (cosine similarity after normalization)
    index = faiss.IndexFlatIP(dimension)
    # L2 normalize to support cosine similarity (copy to avoid modifying input)
    embeddings_normalized = embeddings.copy()
    faiss.normalize_L2(embeddings_normalized)
    index.add(embeddings_normalized)
    return index


def _search_faiss_index(index: faiss.Index, query_embeddings: np.ndarray, top_k: int):
    """
    Search in a FAISS index.
    index: FAISS index
    query_embeddings: query vectors (n_queries, dim)
    top_k: number of results to return
    Returns: (scores_list, indices_list) where each element is a (top_k,) array
    """
    # L2 normalize query vectors (copy to avoid modifying input)
    query_embeddings_normalized = query_embeddings.copy()
    faiss.normalize_L2(query_embeddings_normalized)
    
    # Search
    scores, indices = index.search(query_embeddings_normalized, top_k)
    
    scores_list = [scores[i] for i in range(len(query_embeddings))]
    indices_list = [indices[i] for i in range(len(query_embeddings))]
    
    return scores_list, indices_list


class DenseRetriever(AbsRetriever):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pages = None
        self.index = None
        self.doc_emb = None
        
        # Check if API mode is enabled
        self.api_url = config.get("api_url")  # e.g. "http://localhost:8001"
        self.use_api = self.api_url is not None
        
        if self.use_api:
            # API mode: do not load local model
            print(f"[DenseRetriever] Using API mode: {self.api_url}")
            self.model = None
            # Test connection
            try:
                response = requests.get(f"{self.api_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"[DenseRetriever] API service healthy: {response.json()}")
                else:
                    print(f"[DenseRetriever] Warning: API service returned {response.status_code}")
            except Exception as e:
                print(f"[DenseRetriever] Warning: failed to connect to API service: {e}")
        else:
            # Local mode: load model
            model_name = config.get("model_name")
            print(f"[DenseRetriever] Using local mode, loading model: {model_name}")
            try:
                # Detect available GPU
                import torch
                has_cuda = torch.cuda.is_available()
                default_device = "cuda:0" if has_cuda else "cpu"
                devices = config.get("devices", default_device)
                
                self.model = FlagAutoModel.from_finetuned(
                    model_name,
                    model_class=config.get("model_class", None),
                    normalize_embeddings=config.get("normalize_embeddings", True),
                    pooling_method=config.get("pooling_method", "cls"),
                    trust_remote_code=config.get("trust_remote_code", True),
                    query_instruction_for_retrieval=config.get("query_instruction_for_retrieval"),
                    use_fp16=config.get("use_fp16", False),
                    devices=devices
                )
                if self.model is None:
                    raise RuntimeError("Model load failed: FlagAutoModel.from_finetuned() returned None")
                print(f"[DenseRetriever] Model loaded successfully on devices: {devices}")
            except Exception as e:
                error_msg = (
                    f"[DenseRetriever] Model load failed: {e}\n"
                    f"  Model name: {model_name}\n"
                    f"  Check: 1) model name 2) network 3) disk space"
                )
                print(error_msg)
                raise RuntimeError(error_msg) from e


    # ---------- Internal helpers ----------
    def _index_dir(self) -> str:
        return self.config["index_dir"]

    def _pages_dir(self) -> str:
        return os.path.join(self._index_dir(), "pages")

    def _emb_path(self) -> str:
        return os.path.join(self._index_dir(), "doc_emb.npy")

    def _encode_via_api(self, texts: List[str], encode_type: str = "corpus") -> np.ndarray:
        """
        Encode text via API.
        
        Args:
            texts: list of texts
            encode_type: "corpus" or "query"
        
        Returns:
            embeddings as numpy array
        """
        # Validate input
        if not texts:
            raise ValueError("[DenseRetriever] Text list is empty; cannot encode")
        
        # Filter empty strings
        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            raise ValueError("[DenseRetriever] All texts are empty; cannot encode")
        
        if len(non_empty_texts) != len(texts):
            print(f"[DenseRetriever] Warning: filtered {len(texts) - len(non_empty_texts)} empty texts")
        
        try:
            request_data = {
                "texts": non_empty_texts,
                "type": encode_type,
                "batch_size": self.config.get("batch_size", 32),
                "max_length": self.config.get("max_length", 512),
            }
            
            response = requests.post(
                f"{self.api_url}/encode",
                json=request_data,
                timeout=300  # 5-minute timeout for large batches
            )
            
            # If request fails, print detailed error info
            if response.status_code != 200:
                error_detail = ""
                try:
                    error_response = response.json()
                    error_detail = f" Server error: {error_response}"
                except:
                    error_detail = f" Response body: {response.text[:500]}"
                
                error_msg = (
                    f"[DenseRetriever] API encode failed: {response.status_code} {response.reason}\n"
                    f"  URL: {self.api_url}/encode\n"
                    f"  Params: texts={len(non_empty_texts)}, type={encode_type}, "
                    f"batch_size={request_data['batch_size']}, max_length={request_data['max_length']}\n"
                    f"{error_detail}"
                )
                print(error_msg)
                response.raise_for_status()
            
            result = response.json()
            embeddings = np.array(result["embeddings"], dtype=np.float32)
            
            # If empty texts were filtered, pad zeros to keep index alignment
            if len(non_empty_texts) != len(texts):
                # Locate empty indices
                empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]
                # Build full embeddings array
                full_embeddings = np.zeros((len(texts), embeddings.shape[1]), dtype=np.float32)
                non_empty_idx = 0
                for i in range(len(texts)):
                    if i not in empty_indices:
                        full_embeddings[i] = embeddings[non_empty_idx]
                        non_empty_idx += 1
                embeddings = full_embeddings
            
            return embeddings
        except requests.exceptions.RequestException as e:
            error_msg = (
                f"[DenseRetriever] API encode failed (network): {e}\n"
                f"  URL: {self.api_url}/encode\n"
                f"  Check: 1) service running 2) URL correct 3) network OK"
            )
            print(error_msg)
            raise
        except Exception as e:
            print(f"[DenseRetriever] API encode failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _encode_pages(self, pages: List[Page]) -> np.ndarray:
        # Keep encoding consistent with build() / update()
        # Handle possible None values
        texts = []
        for p in pages:
            header = p.header if p.header is not None else ""
            content = p.content if p.content is not None else ""
            text = (header + " " + content).strip()
            text = p.content
            texts.append(text)
        
        if self.use_api:
            # API mode
            return self._encode_via_api(texts, encode_type="corpus")
        else:
            # Local mode
            if self.model is None:
                raise RuntimeError("DenseRetriever model not initialized. Check model loading.")
            return self.model.encode_corpus(
                texts,
                batch_size=self.config.get("batch_size", 32),
                max_length=self.config.get("max_length", 512),
            )

    # ---------- Public API ----------
    def load(self) -> None:
        """
        Restore from disk:
        - pages snapshot
        - doc_emb.npy
        - FAISS index
        """
        # If load fails, don't crash; ResearchAgent can rebuild.
        try:
            # Load embeddings
            self.doc_emb = np.load(self._emb_path())
            # Rebuild index
            self.index = _build_faiss_index(self.doc_emb)
            # Load pages
            self.pages = InMemoryPageStore.load(self._pages_dir()).load()
        except Exception as e:
            print("DenseRetriever.load() failed, will need build():", e)

    def build(self, page_store: InMemoryPageStore) -> None:
        """
        Full rebuild of the vector index.
        """
        os.makedirs(self._pages_dir(), exist_ok=True)

        # 1. Load pages
        self.pages = page_store.load()

        # 2. Encode all pages
        self.doc_emb = self._encode_pages(self.pages)

        # 3. Build FAISS index
        self.index = _build_faiss_index(self.doc_emb)

        # 4. Persist to disk
        # Create temporary PageStore to save
        temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
        temp_page_store.save(self.pages)
        np.save(self._emb_path(), self.doc_emb)

    def update(self, page_store: InMemoryPageStore) -> None:
        """
        Incremental update: if only new pages were added or tail changed,
        re-encode from the first diff position instead of full rebuild.
        """
        # If never built, do a full build
        if not self.pages or self.doc_emb is None or self.index is None:
            self.build(page_store)
            return

        new_pages = page_store.load()
        old_pages = self.pages

        # 1. Find first difference index
        max_shared = min(len(new_pages), len(old_pages))
        diff_idx = max_shared  # Assume identical initially
        for i in range(max_shared):
            if Page.equal(new_pages[i], old_pages[i]):
                continue
            diff_idx = i
            break

        # 2. Check if anything changed
        changed = (diff_idx < max_shared) or (len(new_pages) != len(old_pages))
        if not changed:
            # No changes, return
            return

        # 3. Keep old vectors before diff_idx; re-encode the tail
        keep_emb = self.doc_emb[:diff_idx]

        tail_pages = new_pages[diff_idx:]
        tail_emb = self._encode_pages(tail_pages)

        new_doc_emb = np.concatenate([keep_emb, tail_emb], axis=0)

        # 4. Rebuild FAISS index
        self.index = _build_faiss_index(new_doc_emb)

        # 5. Persist + refresh in-memory state
        self.pages = new_pages
        self.doc_emb = new_doc_emb
        
        # Create temporary PageStore to save
        temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
        temp_page_store.save(self.pages)
        np.save(self._emb_path(), self.doc_emb)

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        """
        Input: multiple queries
        Output: retrieval results for each query (Hit lists)
        For multiple queries, we search each independently and then aggregate scores by page_id,
        returning top_k results.
        """
        if self.index is None:
            # If no index (e.g., build/load not called), try load
            self.load()
            # If still no index, return empty
            if self.index is None:
                return [[] for _ in query_list]

        # Encode all queries
        if self.use_api:
            # API mode
            queries_emb = self._encode_via_api(query_list, encode_type="query")
        else:
            # Local mode
            queries_emb = self.model.encode_queries(
                query_list,
                batch_size=self.config.get("batch_size", 32),
                max_length=self.config.get("max_length", 512),
            )

        # Use helper search
        scores_list, indices_list = _search_faiss_index(self.index, queries_emb, top_k)

        # Aggregate by page_id: if a page appears for multiple queries, sum scores
        page_scores: Dict[str, float] = {}  # page_id -> total score
        page_hits: Dict[str, Hit] = {}      # page_id -> representative Hit

        for scores, indices in zip(scores_list, indices_list):
            for rank, (idx, sc) in enumerate(zip(indices, scores)):
                idx_int = int(idx)
                if idx_int < 0 or idx_int >= len(self.pages):
                    continue
                page = self.pages[idx_int]
                snippet = page.content
                page_id = str(idx_int)
                score = float(sc)

                if page_id in page_scores:
                    # Accumulate score
                    page_scores[page_id] += score
                else:
                    # First time this page is seen
                    page_scores[page_id] = score
                    page_hits[page_id] = Hit(
                        page_id=page_id,
                        snippet=snippet,
                        source="vector",
                        meta={"rank": rank, "score": score},
                    )

        # Sort by total score, take top_k
        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_pages = sorted_pages[:top_k]

        # Build final hits list using aggregated scores
        final_hits: List[Hit] = []
        for rank, (page_id, total_score) in enumerate(top_k_pages):
            hit = page_hits[page_id]
            # Update meta score with aggregated total
            updated_meta = hit.meta.copy() if hit.meta else {}
            updated_meta["rank"] = rank
            updated_meta["score"] = total_score
            final_hits.append(
                Hit(
                    page_id=hit.page_id,
                    snippet=hit.snippet,
                    source=hit.source,
                    meta=updated_meta
                )
            )

        # Return List[List[Hit]] (single list for aggregated results)
        return [final_hits]

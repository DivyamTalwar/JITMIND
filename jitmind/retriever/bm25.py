from __future__ import annotations
import os, json, subprocess, shutil, time
from typing import Dict, Any, List, Optional

try:
    from pyserini.search.lucene import LuceneSearcher
except ImportError:
    LuceneSearcher = None  # type: ignore

from jitmind.retriever.base import AbsRetriever
from jitmind.schemas import InMemoryPageStore, Hit, Page


def _safe_rmtree(path: str, max_retries: int = 3, delay: float = 0.5) -> None:
    """
    Safely remove a directory tree with retries.
    
    Args:
        path: target directory path
        max_retries: max retry attempts
        delay: retry delay (seconds)
    """
    if not os.path.exists(path):
        return
    
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            # Ensure directory is removed
            if not os.path.exists(path):
                return
            time.sleep(delay)
        except OSError as e:
            if attempt == max_retries - 1:
                # Final attempt failed, try force delete
                try:
                    # Aggressive removal
                    import subprocess
                    subprocess.run(['rm', '-rf', path], check=False, capture_output=True)
                    if not os.path.exists(path):
                        return
                except Exception:
                    pass
                raise OSError(f"Failed to remove directory {path}: {e}")
            time.sleep(delay)


class BM25Retriever(AbsRetriever):
    """
        Keyword retriever (BM25 / Lucene)
        config expects:
        {
            "index_dir": "xxx",   # stores index/ and pages/
            "threads": 4
        }
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if LuceneSearcher is None:
            raise ImportError("BM25Retriever requires pyserini to be installed")
        self.index_dir = self.config["index_dir"]
        self.searcher: Optional[LuceneSearcher] = None
        self.pages: List[Page] = []

    def _pages_dir(self):
        return os.path.join(self.index_dir, "pages")

    def _lucene_dir(self):
        return os.path.join(self.index_dir, "index")

    def _docs_dir(self):
        return os.path.join(self.index_dir, "documents")

    def load(self) -> None:
        # Try to restore from disk
        if not os.path.exists(self._lucene_dir()):
            raise RuntimeError("BM25 index not found, need build() first.")
        self.pages = InMemoryPageStore.load(self._pages_dir()).load()
        self.searcher = LuceneSearcher(self._lucene_dir())  # type: ignore

    def build(self, page_store: InMemoryPageStore) -> None:
        # 0. Clean old directories to ensure a fresh build
        # Use safe delete with retries
        _safe_rmtree(self._lucene_dir())
        _safe_rmtree(self._docs_dir())
        
        # 1. Create required directories
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self._docs_dir(), exist_ok=True)

        # 2. Dump pages -> documents.jsonl (pyserini needs id + contents)
        pages = page_store.load()
        docs_path = os.path.join(self._docs_dir(), "documents.jsonl")
        with open(docs_path, "w", encoding="utf-8") as f:
            for i, p in enumerate(pages):
                text = (p.header + " " + p.content).strip()
                text = '\n'.join(p.content.split('\n')[1:])
                text = p.content
                json.dump({"id": str(i), "contents": text}, f, ensure_ascii=False)
                f.write("\n")

        # 3. Ensure Lucene index directory is clean
        os.makedirs(self._lucene_dir(), exist_ok=True)

        # 4. Run pyserini to build Lucene index
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", self._docs_dir(),
            "--index", self._lucene_dir(),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(self.config.get("threads", 1)),
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]
        
        # Retry to handle occasional build failures
        max_build_retries = 2
        for attempt in range(max_build_retries):
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                break
            except subprocess.CalledProcessError as e:
                if attempt == max_build_retries - 1:
                    print("[ERROR] Pyserini index build failed:")
                    print(f"  stdout: {e.stdout}")
                    print(f"  stderr: {e.stderr}")
                    raise
                print(f"[WARN] Pyserini index build failed, retry {attempt + 1}/{max_build_retries}...")
                # Clean failed index
                _safe_rmtree(self._lucene_dir())
                os.makedirs(self._lucene_dir(), exist_ok=True)
                time.sleep(1)

        # 5. Persist pages to disk for load() / search() lookup
        # Create temporary PageStore instance to save
        temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
        temp_page_store.save(pages)
        
        # 6. Refresh in-memory state
        self.pages = pages
        self.searcher = LuceneSearcher(self._lucene_dir())  # type: ignore

    def update(self, page_store: InMemoryPageStore) -> None:
        # Lucene does not provide a simple incremental update API here.
        # For this prototype, do a full rebuild for reliability.
        self.build(page_store)

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        if self.searcher is None:
            # Safety: if load/build was missed
            self.load()

        results_all: List[List[Hit]] = []
        for q in query_list:
            q = q.strip()
            if not q:
                results_all.append([])
                continue

            hits_for_q = []
            py_hits = self.searcher.search(q, k=top_k)
            for rank, h in enumerate(py_hits):
                # h.docid is a string id
                idx = int(h.docid)
                if idx < 0 or idx >= len(self.pages):
                    continue
                page = self.pages[idx]
                snippet = page.content
                hits_for_q.append(
                    Hit(
                        page_id=str(idx),
                        snippet=snippet,
                        source="keyword",
                        meta={"rank": rank, "score": float(h.score)}
                    )
                )
            results_all.append(hits_for_q)
        return results_all

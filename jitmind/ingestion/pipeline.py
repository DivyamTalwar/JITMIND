from __future__ import annotations

from typing import List, Optional, Dict, Any
import hashlib

from .documents import Document, Chunk
from .chunking import BaseChunker, SimpleChunker
from .loaders import BaseLoader


class IngestionPipeline:
    def __init__(
        self,
        chunker: Optional[BaseChunker] = None,
        deduplicate: bool = True,
    ) -> None:
        self.chunker = chunker or SimpleChunker()
        self.deduplicate = deduplicate

    def ingest_loaders(
        self,
        loaders: List[BaseLoader],
        memory_agent,
        user_id: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        documents: List[Document] = []
        for loader in loaders:
            documents.extend(loader.load())
        return self.ingest_documents(documents, memory_agent, user_id=user_id, extra_meta=extra_meta)

    def ingest_documents(
        self,
        documents: List[Document],
        memory_agent,
        user_id: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        updates = []
        seen_hashes = self._existing_hashes(memory_agent) if self.deduplicate else set()
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            updates.extend(self._ingest_chunks(chunks, memory_agent, user_id, extra_meta, seen_hashes))
        return updates

    def _ingest_chunks(
        self,
        chunks: List[Chunk],
        memory_agent,
        user_id: Optional[str],
        extra_meta: Optional[Dict[str, Any]],
        seen_hashes: set[str],
    ) -> List[Any]:
        updates = []
        for ch in chunks:
            content_hash = self._hash(ch.text)
            if self.deduplicate and content_hash in seen_hashes:
                continue
            meta = {
                "source": ch.source,
                "doc_id": ch.doc_id,
                "chunk_id": ch.chunk_id,
                "chunk_index": ch.index,
                "content_hash": content_hash,
            }
            meta.update(ch.metadata)
            if extra_meta:
                meta.update(extra_meta)
            updates.append(memory_agent.memorize(ch.text, meta=meta, user_id=user_id))
            seen_hashes.add(content_hash)
        return updates

    def _existing_hashes(self, memory_agent) -> set[str]:
        hashes: set[str] = set()
        page_store = getattr(memory_agent, "page_store", None)
        if not page_store:
            return hashes
        try:
            pages = page_store.load()
        except Exception:
            return hashes
        for p in pages:
            meta = getattr(p, "meta", None) or {}
            h = meta.get("content_hash")
            if h:
                hashes.add(str(h))
        return hashes

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

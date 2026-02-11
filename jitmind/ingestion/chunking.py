from __future__ import annotations

from dataclasses import dataclass
from typing import List
import hashlib

from .documents import Document, Chunk


@dataclass
class ChunkingConfig:
    max_chars: int = 2000
    overlap: int = 200
    respect_paragraphs: bool = True


class BaseChunker:
    def chunk(self, doc: Document) -> List[Chunk]:
        raise NotImplementedError


class SimpleChunker(BaseChunker):
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, doc: Document) -> List[Chunk]:
        text = (doc.content or "").strip()
        if not text:
            return []

        max_chars = self.config.max_chars
        overlap = max(0, min(self.config.overlap, max_chars // 2))
        chunks: List[str] = []

        if self.config.respect_paragraphs:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            current = ""
            for p in paragraphs:
                if len(current) + len(p) + 2 <= max_chars:
                    current = f"{current}\n\n{p}".strip()
                else:
                    if current:
                        chunks.append(current)
                    current = p
            if current:
                chunks.append(current)
        else:
            chunks = [text]

        # Sliding window over each chunk if it exceeds max_chars
        final_chunks: List[str] = []
        for c in chunks:
            if len(c) <= max_chars:
                final_chunks.append(c)
                continue
            start = 0
            while start < len(c):
                end = min(len(c), start + max_chars)
                final_chunks.append(c[start:end])
                if end == len(c):
                    break
                start = max(0, end - overlap)

        out: List[Chunk] = []
        for i, c in enumerate(final_chunks):
            cid = self._chunk_id(doc, i, c)
            meta = dict(doc.metadata)
            meta.update({"chunk_index": i, "chunk_id": cid})
            out.append(Chunk(
                text=c,
                source=doc.source,
                doc_id=doc.doc_id,
                chunk_id=cid,
                index=i,
                metadata=meta,
            ))
        return out

    def _chunk_id(self, doc: Document, idx: int, text: str) -> str:
        base = f"{doc.doc_id or doc.source}:{idx}:{text[:200]}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]

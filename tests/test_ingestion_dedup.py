from __future__ import annotations

from typing import Any, Dict, Optional

from jitmind import IngestionPipeline, Document
from jitmind.schemas import Page, InMemoryPageStore


class _StubMemoryAgent:
    def __init__(self, page_store: InMemoryPageStore) -> None:
        self.page_store = page_store
        self.calls = 0

    def memorize(self, message: str, meta: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None):
        self.calls += 1
        page = Page(header="stub", content=message, meta=dict(meta or {}))
        self.page_store.add(page)
        return {"ok": True}


def test_ingestion_pipeline_deduplicates(tmp_path):
    page_store = InMemoryPageStore(dir_path=str(tmp_path))
    agent = _StubMemoryAgent(page_store)

    pipeline = IngestionPipeline(deduplicate=True)
    docs = [
        Document(content="hello world", source="t", doc_id="1"),
        Document(content="another doc", source="t", doc_id="2"),
    ]

    updates1 = pipeline.ingest_documents(docs, memory_agent=agent, user_id="u1")
    pages1 = page_store.load()
    assert len(updates1) == len(pages1) == 2

    updates2 = pipeline.ingest_documents(docs, memory_agent=agent, user_id="u1")
    pages2 = page_store.load()

    assert updates2 == []
    assert len(pages2) == len(pages1)


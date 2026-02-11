from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Document:
    """Ingestion document container."""
    content: str
    source: str = ""
    doc_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Chunked unit of a document for ingestion."""
    text: str
    source: str
    doc_id: Optional[str]
    chunk_id: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

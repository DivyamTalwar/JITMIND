from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol
from pydantic import BaseModel, Field

class SearchPlan(BaseModel):
    """Search planning structure"""
    info_needs: List[str] = Field(default_factory=list, description="List of information needs")
    tools: List[str] = Field(default_factory=list, description="Tools to use for searching")
    keyword_collection: List[str] = Field(default_factory=list, description="Keywords to search for")
    vector_queries: List[str] = Field(default_factory=list, description="Semantic search queries")
    page_index: List[int] = Field(default_factory=list, description="Specific page indices to retrieve")
    graph_queries: List[str] = Field(default_factory=list, description="Entity queries for graph retrieval")

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        schema = super().model_json_schema()
        props = list(schema.get("properties", {}).keys())
        schema["required"] = props
        schema["additionalProperties"] = False
        return schema

class Hit(BaseModel):
    """Search result hit"""
    page_id: Optional[str] = Field(None, description="Page ID in store")
    snippet: str = Field(..., description="Text snippet from the source")
    source: str = Field(..., description="Source type (keyword/vector/page_index/tool)")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class Retriever(Protocol):
    """Unified interface for keyword / vector / page-id retrievers."""
    name: str
    def build(self, page_store) -> None: ...
    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]: ...

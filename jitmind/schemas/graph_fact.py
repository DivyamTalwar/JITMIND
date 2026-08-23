"""Typed evidence returned by temporal knowledge-graph queries."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class GraphFact(BaseModel):
    id: str = Field(..., description="Stable observation id")
    head: str
    relation: str
    tail: str
    source_memory_id: Optional[str] = None
    source_page_id: Optional[str] = None
    t_observed: Optional[str] = None
    t_valid: Optional[str] = None
    t_invalid: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @property
    def statement(self) -> str:
        return f"{self.head} {self.relation} {self.tail}"


__all__ = ["GraphFact"]

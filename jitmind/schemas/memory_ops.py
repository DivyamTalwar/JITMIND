# -*- coding: utf-8 -*-
"""
Schemas for memory operations, temporal tagging, and graph extraction.
"""

from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    name: str = Field(..., description="Entity surface form")
    type: str = Field(..., description="Entity type (person, org, location, concept, etc.)")


class ExtractedRelation(BaseModel):
    head: str = Field(..., description="Head entity name")
    relation: str = Field(..., description="Relation type/label")
    tail: str = Field(..., description="Tail entity name")
    t_valid: Optional[str] = Field(default=None, description="When relation became true (ISO-8601)")
    t_invalid: Optional[str] = Field(default=None, description="When relation stopped being true (ISO-8601)")


class MemoryOperationDecision(BaseModel):
    operation: Literal["add", "update", "delete", "noop"] = Field(..., description="Memory operation")
    target_id: Optional[str] = Field(default=None, description="Target memory id for update/delete")
    extends_id: Optional[str] = Field(default=None, description="Target memory id to extend without superseding")
    updated_content: Optional[str] = Field(default=None, description="New content for update")
    importance: Literal["short", "mid", "long"] = Field(default="short", description="Memory tier")
    t_observed: Optional[str] = Field(default=None, description="When the fact was observed (ISO-8601)")
    t_valid: Optional[str] = Field(default=None, description="When the fact became true (ISO-8601)")
    t_invalid: Optional[str] = Field(default=None, description="When the fact stopped being true (ISO-8601)")
    entities: List[ExtractedEntity] = Field(default_factory=list, description="Entities to store in graph")
    relations: List[ExtractedRelation] = Field(default_factory=list, description="Relations to store in graph")

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        schema = super().model_json_schema()
        props = list(schema.get("properties", {}).keys())
        schema["required"] = props
        schema["additionalProperties"] = False
        return schema

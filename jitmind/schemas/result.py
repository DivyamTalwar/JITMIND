from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Result(BaseModel):
    """Search and integration result"""
    content: str = Field("", description="Integrated content about the question")
    sources: List[Optional[str]] = Field(default_factory=list, description="List of page IDs of sources used")

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        schema = super().model_json_schema()
        props = list(schema.get("properties", {}).keys())  # ["content", "sources"]
        schema["required"] = props
        schema["additionalProperties"] = False
        return schema

class EnoughDecision(BaseModel):
    """Decision on whether information is sufficient"""
    enough: bool = Field(..., description="Whether information is sufficient")

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        schema = super().model_json_schema()
        schema["required"] = ["enough"]
        schema["additionalProperties"] = False
        return schema

class ReflectionDecision(BaseModel):
    """Complete reflection decision with new request if information is insufficient"""
    enough: bool = Field(..., description="Whether information is sufficient")
    new_request: Optional[str] = Field(None, description="New search request if information is insufficient")

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        schema = super().model_json_schema()
        schema["required"] = ["enough"]
        schema["additionalProperties"] = False
        return schema

class ResearchOutput(BaseModel):
    """Research output"""
    integrated_memory: str = Field(..., description="Integrated memory content")
    raw_memory: Dict[str, Any] = Field(..., description="Raw memory data")

class GenerateRequests(BaseModel):
    """Generate new requests"""
    new_requests: List[str] = Field(..., description="List of new search requests")

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        schema = super().model_json_schema()
        schema["required"] = ["new_requests"]
        schema["additionalProperties"] = False
        return schema

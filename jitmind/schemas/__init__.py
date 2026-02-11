"""
Schemas Module

This module exposes all core data models and protocol definitions for the JITMind (JITMind) framework.
It organizes memory, page, search, tool, and result schemas for unified import and type safety across the system.
"""
from .memory import MemoryState, MemoryUpdate, MemoryStore, InMemoryMemoryStore
from .advanced_memory import AdvancedMemoryStore, AdvancedMemoryState, MemoryEntry
from .page import Page, PageStore, InMemoryPageStore
from .ttl_memory import TTLMemoryStore, TTLMemoryState, TTLMemoryEntry
from .ttl_page import TTLPageStore
from .search import SearchPlan, Retriever, Hit
from .tools import ToolResult, Tool, ToolRegistry
from .result import Result, EnoughDecision, ReflectionDecision, ResearchOutput, GenerateRequests
from .memory_ops import MemoryOperationDecision, ExtractedEntity, ExtractedRelation
from .self_rag import SelfRAGDecision

# =============================
# Model rebuilding for forward references
# =============================
# Explicitly rebuild models so forward references (e.g., 'Page') resolve correctly
# in concurrent / multi-threaded environments.
MemoryUpdate.model_rebuild()
ResearchOutput.model_rebuild()

# JSON Schema constants for LLM and system validation
PLANNING_SCHEMA = SearchPlan.model_json_schema()
INTEGRATE_SCHEMA = Result.model_json_schema()
INFO_CHECK_SCHEMA = EnoughDecision.model_json_schema()
GENERATE_REQUESTS_SCHEMA = GenerateRequests.model_json_schema()
MEMORY_OP_SCHEMA = MemoryOperationDecision.model_json_schema()
SELF_RAG_SCHEMA = SelfRAGDecision.model_json_schema()

__all__ = [
    "MemoryState", "MemoryUpdate", "MemoryStore", "InMemoryMemoryStore",
    "AdvancedMemoryStore", "AdvancedMemoryState", "MemoryEntry",
    "Page", "PageStore", "InMemoryPageStore",
    "TTLMemoryStore", "TTLMemoryState", "TTLMemoryEntry",
    "TTLPageStore",
    "SearchPlan", "Retriever", "Hit",
    "ToolResult", "Tool", "ToolRegistry",
    "Result", "EnoughDecision", "ReflectionDecision", "ResearchOutput", "GenerateRequests",
    "MemoryOperationDecision", "ExtractedEntity", "ExtractedRelation", "SelfRAGDecision",
    "PLANNING_SCHEMA", "INTEGRATE_SCHEMA", "INFO_CHECK_SCHEMA", "GENERATE_REQUESTS_SCHEMA", "MEMORY_OP_SCHEMA", "SELF_RAG_SCHEMA",
]

# -*- coding: utf-8 -*-
"""
JITMind Framework

A dual-agent architecture for building long-term memory with deep research capabilities.

Key Components:
- MemoryAgent: Builds structured memory from raw messages
- ResearchAgent: Performs multi-iteration research with reflection
"""

from __future__ import annotations

# Core agents
from jitmind.agents import MemoryAgent, ResearchAgent

# Generators
from jitmind.generator import AbsGenerator, OpenAIGenerator, VLLMGenerator

# Retrievers
from jitmind.retriever import AbsRetriever, IndexRetriever, GraphRetriever

# Try to import optional retrievers
try:
    from jitmind.retriever import BM25Retriever
except ImportError:
    BM25Retriever = None  # type: ignore

try:
    from jitmind.retriever import DenseRetriever
except ImportError:
    DenseRetriever = None  # type: ignore

try:
    from jitmind.retriever import CohereDenseRetriever, CohereReranker
except ImportError:
    CohereDenseRetriever = None  # type: ignore
    CohereReranker = None  # type: ignore

# Configurations
from jitmind.config import (
    OpenAIGeneratorConfig,
    VLLMGeneratorConfig,
    DenseRetrieverConfig,
    BM25RetrieverConfig,
    IndexRetrieverConfig,
    CohereEmbedRetrieverConfig,
    CohereRerankerConfig,
)

# Schemas
from jitmind.schemas import (
    MemoryState,
    AdvancedMemoryStore,
    AdvancedMemoryState,
    MemoryEntry,
    Page,
    MemoryUpdate,
    SearchPlan,
    Hit,
    Result,
    EnoughDecision,
    ReflectionDecision,
    ResearchOutput,
    InMemoryMemoryStore,
    InMemoryPageStore,
    TTLMemoryStore,
    TTLMemoryState,
    TTLMemoryEntry,
    TTLPageStore,
)
try:
    from jitmind.graph import GraphMemoryStore, GraphOntology
except ImportError:
    GraphMemoryStore = None  # type: ignore
    GraphOntology = None  # type: ignore
from jitmind.ingestion import (
    Document,
    Chunk,
    BaseChunker,
    SimpleChunker,
    ChunkingConfig,
    BaseLoader,
    TextFileLoader,
    DirectoryLoader,
    URLLoader,
    JSONLLoader,
    S3Loader,
    NotionLoader,
    GDriveLoader,
    IngestionPipeline,
)
from jitmind.profile import UserProfile, UserProfileStore, UserProfileAgent
try:
    from jitmind.summarization import HierarchicalSummarizer
except ImportError:
    HierarchicalSummarizer = None  # type: ignore
try:
    from jitmind.maintenance import MemoryConsolidator
except ImportError:
    MemoryConsolidator = None  # type: ignore
from jitmind.utils import CheckpointManager
from jitmind.learning import ExperienceReplayBuffer
from jitmind.scoping import DEFAULT_NAMESPACE, Namespace, normalize_namespace
try:
    from jitmind.evaluation import RAGASEvaluator
except Exception:
    RAGASEvaluator = None  # type: ignore

__version__ = "0.1.0"
__all__ = [
    # Core agents
    "MemoryAgent",
    "ResearchAgent",
    
    # Generators
    "AbsGenerator",
    "OpenAIGenerator",
    "VLLMGenerator",
    
    # Retrievers
    "AbsRetriever",
    "IndexRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "CohereDenseRetriever",
    "CohereReranker",
    "GraphRetriever",
    
    # Configurations
    "OpenAIGeneratorConfig",
    "VLLMGeneratorConfig",
    "DenseRetrieverConfig",
    "BM25RetrieverConfig",
    "IndexRetrieverConfig",
    "CohereEmbedRetrieverConfig",
    "CohereRerankerConfig",
    
    # Schemas
    "MemoryState",
    "AdvancedMemoryStore",
    "AdvancedMemoryState",
    "MemoryEntry",
    "Page",
    "MemoryUpdate",
    "SearchPlan",
    "Hit",
    "Result",
    "EnoughDecision",
    "ReflectionDecision",
    "ResearchOutput",
    "InMemoryMemoryStore",
    "InMemoryPageStore",
    "TTLMemoryStore",
    "TTLMemoryState",
    "TTLMemoryEntry",
    "TTLPageStore",
    "GraphMemoryStore",
    "GraphOntology",
    "Document",
    "Chunk",
    "BaseChunker",
    "SimpleChunker",
    "ChunkingConfig",
    "BaseLoader",
    "TextFileLoader",
    "DirectoryLoader",
    "URLLoader",
    "JSONLLoader",
    "S3Loader",
    "NotionLoader",
    "GDriveLoader",
    "IngestionPipeline",
    "UserProfile",
    "UserProfileStore",
    "UserProfileAgent",
    "HierarchicalSummarizer",
    "MemoryConsolidator",
    "CheckpointManager",
    "ExperienceReplayBuffer",
    "DEFAULT_NAMESPACE",
    "Namespace",
    "normalize_namespace",
    "RAGASEvaluator",
]

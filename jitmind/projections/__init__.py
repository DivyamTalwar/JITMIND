"""Opt-in persistent projections. Importing this package invokes no providers."""

from .coordinator import ProjectionCoordinator
from .graph import SQLiteGraphProjection
from .models import (
    GraphEdge,
    ProjectionError,
    ProjectionEvent,
    ProjectionHit,
    PurgePlan,
    QueryResult,
)
from .source import SQLiteProjectionSource
from .vector import (
    Embedder,
    EmbeddingIdentity,
    FeatureHashingEmbedder,
    SQLiteVectorProjection,
)

__all__ = [
    "Embedder",
    "EmbeddingIdentity",
    "FeatureHashingEmbedder",
    "GraphEdge",
    "ProjectionCoordinator",
    "ProjectionError",
    "ProjectionEvent",
    "ProjectionHit",
    "PurgePlan",
    "QueryResult",
    "SQLiteGraphProjection",
    "SQLiteProjectionSource",
    "SQLiteVectorProjection",
]

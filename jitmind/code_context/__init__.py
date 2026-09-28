"""Opt-in local code context. Importing this package does not start Node."""

from .models import CodeEvidence, CodeEvidencePage, CodeQuery
from .process import CapabilityUnavailable, NodeParser
from .retriever import CodeContextRetriever
from .service import CodeContext
from .snapshot import RepoRegistry

__all__ = [
    "CapabilityUnavailable",
    "CodeContext",
    "CodeContextRetriever",
    "CodeEvidence",
    "CodeEvidencePage",
    "CodeQuery",
    "NodeParser",
    "RepoRegistry",
]

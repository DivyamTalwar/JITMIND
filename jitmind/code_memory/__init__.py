"""Optional persistent source bindings. Legacy memory remains independent."""

from .anchors import CodeContextSnapshotAdapter, CodeMemoryService
from .binding_models import (
    BindingConflict,
    BindingError,
    BindingStaleSource,
    BindingStorageError,
    BindingUnavailable,
    CodeBinding,
    EvidenceRef,
)

__all__ = [
    "BindingConflict",
    "BindingError",
    "BindingStaleSource",
    "BindingStorageError",
    "BindingUnavailable",
    "CodeBinding",
    "CodeContextSnapshotAdapter",
    "CodeMemoryService",
    "EvidenceRef",
]

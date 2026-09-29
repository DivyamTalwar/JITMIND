"""Opt-in SQLite authority. Legacy JSON stores remain the default."""

from .models import (
    AcknowledgementUncertain,
    CapacityExceeded,
    DurableError,
    DurableReceipt,
    HistoricalFact,
    HistoricalSnapshot,
    IdempotencyConflict,
    IngestRequest,
    InvalidRequest,
    MigrationError,
    NamespaceSnapshot,
    ProjectionEvent,
    Proposal,
    ProposalFailure,
    ReceiptContent,
    SchemaMismatch,
    StaleRevision,
    StorageBusy,
    StorageFailure,
    TargetNotFound,
    UnsafeJournal,
    ValidityCorrection,
)
from .sqlite import DurableMemoryAdapter, DurablePageAdapter, SQLiteDurableStore

__all__ = [
    "AcknowledgementUncertain",
    "CapacityExceeded",
    "DurableError",
    "DurableMemoryAdapter",
    "DurablePageAdapter",
    "DurableReceipt",
    "IdempotencyConflict",
    "IngestRequest",
    "InvalidRequest",
    "MigrationError",
    "NamespaceSnapshot",
    "Proposal",
    "ProposalFailure",
    "ReceiptContent",
    "SQLiteDurableStore",
    "SchemaMismatch",
    "StaleRevision",
    "StorageBusy",
    "StorageFailure",
    "TargetNotFound",
    "UnsafeJournal",
]

from .migration import StagedLegacy, stage_legacy

__all__ += ["StagedLegacy", "stage_legacy"]

__all__ += [
    "HistoricalFact",
    "HistoricalSnapshot",
    "ProjectionEvent",
    "ValidityCorrection",
]

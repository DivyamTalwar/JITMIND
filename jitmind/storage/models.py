"""Public durable domain types. Error messages deliberately contain no input data."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jitmind.schemas import MemoryEntry, MemoryOperationDecision, Page


class DurableError(Exception):
    code = "durable_error"

    def __init__(self) -> None:
        super().__init__(self.code)


class InvalidRequest(DurableError):
    code = "invalid_request"


class ProposalFailure(DurableError):
    code = "proposal_failure"


class IdempotencyConflict(DurableError):
    code = "idempotency_conflict"


class StaleRevision(DurableError):
    code = "stale_revision"


class StorageBusy(DurableError):
    code = "storage_busy"


class StorageFailure(DurableError):
    code = "storage_failure"


class SchemaMismatch(DurableError):
    code = "schema_mismatch"


class UnsafeJournal(DurableError):
    code = "unsafe_journal"


class AcknowledgementUncertain(DurableError):
    code = "acknowledgement_uncertain"


class MigrationError(DurableError):
    code = "invalid_legacy_source"


class CapacityExceeded(DurableError):
    code = "bounded_read_exceeded"


class TargetNotFound(DurableError):
    code = "target_not_active"


def validate_identifier(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 200:
        raise InvalidRequest()
    try:
        if len(value.encode("utf-8")) > 200:
            raise InvalidRequest()
    except UnicodeError:
        raise InvalidRequest() from None
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise InvalidRequest()
    return value


def canonical_json(
    value: Any, *, max_bytes: int = 262144, max_nodes: int = 8192
) -> str:
    """Strict JSON only; no coercion, non-finite floats, cycles, or unsafe keys."""
    count = 0

    def check(item: Any, depth: int) -> None:
        nonlocal count
        count += 1
        if count > max_nodes or depth > 16:
            raise InvalidRequest()
        if item is None or type(item) in (bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is str:
            if len(item) > max_bytes:
                raise InvalidRequest()
            try:
                item.encode("utf-8")
            except UnicodeError:
                raise InvalidRequest() from None
            return
        if type(item) is list:
            for child in item:
                check(child, depth + 1)
            return
        if type(item) is dict:
            for key, child in item.items():
                if (
                    type(key) is not str
                    or len(key) > 256
                    or key in {"__proto__", "prototype", "constructor"}
                    or any(ord(c) < 32 for c in key)
                ):
                    raise InvalidRequest()
                check(key, depth + 1)
                check(child, depth + 1)
            return
        raise InvalidRequest()

    check(value, 0)
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if len(encoded.encode("utf-8")) > max_bytes:
            raise InvalidRequest()
        return encoded
    except (ValueError, TypeError, UnicodeError):
        raise InvalidRequest() from None


@dataclass(frozen=True)
class IngestRequest:
    namespace_id: str
    idempotency_key: str
    message: str
    metadata_json: str
    user_id: str | None
    digest: str

    @classmethod
    def create(
        cls,
        namespace_id: str,
        idempotency_key: str,
        message: str,
        meta: dict | None = None,
        user_id: str | None = None,
    ) -> IngestRequest:
        validate_identifier(namespace_id)
        validate_identifier(idempotency_key)
        if user_id is not None:
            validate_identifier(user_id)
        if not isinstance(message, str) or not message.strip():
            raise InvalidRequest()
        canonical_json(message, max_bytes=131072)
        if meta is not None and type(meta) is not dict:
            raise InvalidRequest()
        metadata_json = canonical_json(
            meta if meta is not None else {}, max_bytes=65536, max_nodes=2048
        )
        reserved = {
            "page_id",
            "memory_id",
            "decorated",
            "t_observed",
            "t_valid",
            "t_invalid",
            "user_id",
            "_durable",
        }
        if reserved.intersection(json.loads(metadata_json)):
            raise InvalidRequest()
        payload = canonical_json(
            {"message": message, "meta": json.loads(metadata_json), "user_id": user_id}
        )
        return cls(
            namespace_id,
            idempotency_key,
            message,
            metadata_json,
            user_id,
            hashlib.sha256(payload.encode()).hexdigest(),
        )

    def validated(self) -> IngestRequest:
        """Revalidate even manually constructed dataclasses at the trust boundary."""
        try:
            actual = self.create(
                self.namespace_id,
                self.idempotency_key,
                self.message,
                json.loads(self.metadata_json),
                self.user_id,
            )
        except (ValueError, TypeError, RecursionError):
            raise InvalidRequest() from None
        if actual != self:
            raise InvalidRequest()
        return actual


class DurableDecision(MemoryOperationDecision):
    model_config = ConfigDict(extra="forbid", strict=True)


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    abstract: str = Field(min_length=1, max_length=131072)
    header: str = Field(max_length=131072)
    decorated: str = Field(max_length=262144)
    decision: DurableDecision


class DurableReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    namespace_id: str
    idempotency_key: str
    request_digest: str
    operation: Literal["add", "update", "delete", "noop"]
    page_id: str
    memory_id: str | None
    revision: int
    event_id: str
    committed_at: str


@dataclass(frozen=True)
class ReceiptContent:
    """Current content availability, independent of committed operation identity."""

    receipt: DurableReceipt
    status: Literal["available", "retired", "unavailable"]
    page: Page | None


@dataclass(frozen=True)
class NamespaceSnapshot:
    revision: int
    entries: tuple[MemoryEntry, ...]
    truncated: bool


@dataclass(frozen=True)
class ProjectionEvent:
    namespace_id: str
    event_id: str
    revision: int
    memory_ids: tuple[str, ...]


@dataclass(frozen=True)
class ValidityCorrection:
    memory_id: str
    valid_from: str | None
    valid_to: str | None


@dataclass(frozen=True)
class HistoricalFact:
    entry: MemoryEntry
    revision: int
    fact_key: str
    repo_id: str | None
    snapshot_id: str | None
    single_valued: bool
    page_id: str


@dataclass(frozen=True)
class HistoricalSnapshot:
    namespace_id: str
    revision: int | None
    recorded_at: str | None
    baseline_revision: int | None
    baseline_recorded_at: str | None
    coverage: Literal["available", "unavailable"]
    complete: bool
    truncated: bool
    unknown_validity: bool
    facts: tuple[HistoricalFact, ...]
    unknown_scope: bool = False
    unknown_eligibility: bool = False

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(fact.entry for fact in self.facts)

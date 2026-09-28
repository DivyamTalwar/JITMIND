"""Immutable, bounded binding records; these records never contain fact/source text."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Identifier = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=200)
]
Digest = Annotated[str, StringConstraints(strict=True, pattern=r"^[a-f0-9]{64}$")]
Revision = Annotated[int, Field(strict=True, ge=1, lt=2**63)]


class BindingError(Exception):
    def __init__(self):
        super().__init__(self.__class__.__name__)


class BindingConflict(BindingError):
    """The binding or source generation changed; obtain fresh state and retry."""


class BindingStaleSource(BindingConflict):
    """Publication would move behind the repository's recorded source generation."""


class BindingUnavailable(BindingError):
    """No authorized, adequately covered source/fact was available."""


class BindingStorageError(BindingError):
    pass


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @field_validator("*")
    @classmethod
    def safe_strings(cls, value):
        if isinstance(value, str):
            if not value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
                raise ValueError("Invalid identifier")
            try:
                value.encode("utf8")
            except UnicodeError:
                raise ValueError("Invalid identifier") from None
        return value


class EvidenceRef(Record):
    source_id: Digest
    path: str = Field(min_length=1, max_length=1024)
    qualified_name: str = Field(min_length=1, max_length=4096)
    start_line: Revision
    end_line: Revision


class CodeBinding(Record):
    schema_version: Literal[1] = 1
    binding_id: Digest
    logical_binding_id: Identifier
    namespace_id: Identifier
    fact_version_id: Identifier
    repo_id: Identifier
    snapshot_id: Digest
    source_generation: Revision
    source_lineage: Digest
    revision: Revision
    path: str = Field(min_length=1, max_length=1024)
    qualified_name: str = Field(min_length=1, max_length=4096)
    kind: Literal["function", "method", "class"]
    signature_digest: Digest
    raw_source_digest: Digest
    body_digest: Digest
    fingerprint_policy: Literal["python-ast-name-only/v1"] = "python-ast-name-only/v1"
    start_line: Revision
    end_line: Revision
    validation_state: Literal[
        "verified_current", "changed", "unknown", "orphaned_confirmed"
    ]
    reason_code: Literal[
        "created",
        "same_identity",
        "same_file_rename",
        "file_move",
        "rename_and_move",
        "body_changed",
        "identity_changed",
        "ambiguous_candidates",
        "plausible_candidates",
        "source_unknown",
        "symbol_absent",
        "fact_unavailable",
        "snapshot_mismatch",
    ]
    validated_at: str = Field(max_length=40)
    supersedes_binding_id: Digest | None = None
    evidence_refs: tuple[EvidenceRef, ...] = Field(default=(), max_length=8)
    candidate_count: int = Field(default=0, strict=True, ge=0, le=2500)
    review_required: bool = False

    @field_validator("schema_version", mode="before")
    @classmethod
    def schema_strict(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Invalid schema version")
        return value

    @field_validator("repo_id")
    @classmethod
    def repo_uuid(cls, value):
        if str(UUID(value)) != value:
            raise ValueError("Invalid repository identifier")
        return value

    @field_validator("validated_at")
    @classmethod
    def timestamp(cls, value):
        if datetime.fromisoformat(value).tzinfo is None:
            raise ValueError("Expected timezone")
        return value

    @field_validator("path")
    @classmethod
    def path_valid(cls, value):
        from jitmind.code_context.models import safe_path

        return safe_path(value)

    @model_validator(mode="after")
    def line_range(self):
        if self.end_line < self.start_line:
            raise ValueError("Invalid range")
        return self


def identifier(value: str) -> str:
    # Use the same validation for lookup keys as immutable records.
    class Key(Record):
        value: Identifier

    return Key(value=value).value


def revision(value: int, *, zero: bool = False) -> int:
    if type(value) is not int or not (0 if zero else 1) <= value < 2**63:
        raise ValueError("Invalid revision")
    return value

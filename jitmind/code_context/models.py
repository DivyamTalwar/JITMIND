"""Versioned, strict wire contracts for optional local source evidence."""

from __future__ import annotations

import json
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

MAX_RESPONSE_BYTES = 262144
PARSER = "graft-python-subset/1;tree-sitter/0.21.1;python/0.21.0"
PROTOCOL = "jitmind.graft-parser/v1"


def safe_path(value: str, *, empty: bool = False) -> str:
    if type(value) is not str or len(value.encode("utf8")) > 1024:
        raise ValueError("Invalid relative path")
    if empty and value == "":
        return value
    if (
        not value
        or any(ord(c) < 32 for c in value)
        or "\\" in value
        or ":" in value
        or value.startswith("/")
        or any(p in ("", ".", "..") for p in value.split("/"))
    ):
        raise ValueError("Invalid relative path")
    return value


def source_lines(source: str) -> list[str]:
    """Tree-sitter rows count LF, not Unicode paragraph/form-feed separators."""
    parts = source.split("\n")
    return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def under(path: str, prefix: str) -> bool:
    return not prefix or path == prefix or path.startswith(prefix + "/")


def strict_json(raw: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("Nonfinite JSON number")

    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("JSON budget")
    return json.loads(
        raw.decode("utf8"), object_pairs_hook=pairs, parse_constant=invalid
    )


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )


class CodeQuery(StrictModel):
    schema_version: Literal["jitmind.code-query/v1"] = Field(
        default="jitmind.code-query/v1", alias="schema"
    )
    operation: Literal[
        "find_code",
        "file_api",
        "trace_calls",
        "find_all",
        "repo_map",
        "check_freshness",
    ]
    repo_id: str
    snapshot_id: str = Field(pattern="^[a-f0-9]{64}$")
    query: str = Field(default="", max_length=1024)
    path_prefix: str = ""
    limit: StrictInt = Field(default=10, ge=1, le=50)
    max_context_tokens: StrictInt = Field(default=2000, ge=1, le=16000)
    deadline_ms: StrictInt = Field(default=1500, ge=1, le=30000)
    direction: Literal["in", "out"] = "in"
    depth: StrictInt | Literal["all"] = Field(default=1)
    fixed: StrictBool = True

    @field_validator("repo_id")
    @classmethod
    def uuid(cls, v):
        if str(UUID(v)) != v:
            raise ValueError("Expected canonical internal repo UUID")
        return v

    @field_validator("query")
    @classmethod
    def query_text(cls, v):
        if "\x00" in v:
            raise ValueError("Invalid query")
        return v

    @field_validator("depth")
    @classmethod
    def depth_bound(cls, v):
        if v != "all" and not 1 <= v <= 1024:
            raise ValueError("Invalid depth")
        return v

    @field_validator("fixed")
    @classmethod
    def literal_only(cls, v):
        if not v:
            raise ValueError("Only fixed-string search is supported")
        return v

    @field_validator("path_prefix")
    @classmethod
    def prefix(cls, v):
        return safe_path(v, empty=True)

    @model_validator(mode="after")
    def operation_arguments(self):
        if self.operation not in ("repo_map", "check_freshness") and not self.query:
            raise ValueError("This operation requires a query")
        if self.operation == "file_api":
            safe_path(self.query)
        return self

    @classmethod
    def from_json(cls, raw: bytes):
        return cls.model_validate(strict_json(raw))


class CodeEvidence(StrictModel):
    source_id: str = Field(pattern="^[a-f0-9]{64}$")
    repo_id: str
    namespace_id: str
    snapshot_id: str = Field(pattern="^[a-f0-9]{64}$")
    path: str
    digest: str = Field(pattern="^[a-f0-9]{64}$")
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    name: str
    kind: str
    source: str
    signature: str = ""
    score: float = 0.0
    depth: int | None = None


class CodeEvidencePage(StrictModel):
    schema_version: Literal["jitmind.code-evidence/v1"] = Field(
        default="jitmind.code-evidence/v1", alias="schema"
    )
    status: Literal["ok", "unavailable"]
    repo_id: str
    snapshot_id: str | None = None
    parser: str = PARSER
    generation: int | None = None
    commit_sha: str | None = None
    commit_status: str = "unavailable"
    manifest_digest: str | None = None
    coverage_digest: str | None = None
    results: tuple[CodeEvidence, ...] = ()
    coverage: dict = Field(default_factory=dict)
    freshness: dict = Field(default_factory=lambda: {"state": "unknown"})
    truncated: bool = False
    reason_codes: tuple[str, ...] = ()
    count: int | None = None
    source_bytes: int = 0
    context_tokens_upper_bound: int = 0
    response_bytes: int = 0

    def wire_bytes(self) -> bytes:
        return self.model_dump_json(by_alias=True).encode("utf8")


def finish(page: CodeEvidencePage) -> CodeEvidencePage:
    # Byte count includes its own digits. Diagnostics are fixed codes, never stderr.
    for _ in range(4):
        size = len(page.wire_bytes())
        if page.response_bytes == size:
            break
        page = page.model_copy(update={"response_bytes": size})
    if len(page.wire_bytes()) > MAX_RESPONSE_BYTES:
        return finish(
            CodeEvidencePage(
                status="unavailable",
                repo_id=page.repo_id,
                reason_codes=("response_budget",),
            )
        )
    return page

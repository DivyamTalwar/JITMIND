"""Explicit, bounded, read-only acquisition of quiesced legacy JSON stores."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from jitmind.schemas import MemoryEntry, Page

from .models import DurableError, MigrationError, canonical_json, validate_identifier

MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_RECORDS = 1000


@dataclass(frozen=True)
class StagedLegacy:
    source_digest: str
    source_json: str
    page_count: int
    fact_count: int


def source_envelope(source_json: str) -> bytes:
    """Bound untrusted staging objects before encoding, hashing, or JSON parsing."""
    if type(source_json) is not str or len(source_json) > MAX_SOURCE_BYTES:
        raise MigrationError()
    try:
        raw = source_json.encode("utf-8")
    except UnicodeError:
        raise MigrationError() from None
    if len(raw) > MAX_SOURCE_BYTES:
        raise MigrationError()
    return raw


def _strict_json(raw: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise MigrationError()
            result[key] = value
        return result

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(MigrationError()),
    )


def prepare_records(
    source_json: str,
) -> tuple[list[Page], list[MemoryEntry], dict[str, str]]:
    """Revalidate staged data at import, returning detached models and explicit aliases."""
    try:
        data = _strict_json(source_envelope(source_json))
        canonical_json(data, max_bytes=MAX_SOURCE_BYTES, max_nodes=200000)
        if type(data) is not dict or set(data) != {"memory", "pages"}:
            raise MigrationError()
        memory, page_data = data["memory"], data["pages"]
        if type(memory) is not dict or set(memory) != {"entries"}:
            raise MigrationError()
        if type(page_data) is dict:
            if set(page_data) != {"pages"}:
                raise MigrationError()
            page_data = page_data["pages"]
        if type(page_data) is not list or type(memory["entries"]) is not list:
            raise MigrationError()
        if len(page_data) > MAX_RECORDS or len(memory["entries"]) > MAX_RECORDS:
            raise MigrationError()
        aliases, pages, entries = {}, [], []
        original_pages = {}
        for raw in page_data:
            if (
                type(raw) is not dict
                or set(raw) - set(Page.model_fields)
                or not {"header", "content", "meta"} <= set(raw)
            ):
                raise MigrationError()
            page = Page.model_validate(raw, strict=True)
            canonical_json(page.model_dump(), max_bytes=1048576)
            legacy = page.meta.get("page_id")
            if type(legacy) not in (int, str) or (type(legacy) is int and legacy < 0):
                raise MigrationError()
            alias = validate_identifier(str(legacy))
            if alias in aliases:
                raise MigrationError()
            page_id = str(uuid.uuid4())
            aliases[alias] = page_id
            original_pages[alias] = page
            pages.append(page)
        by_id = {}
        referenced = set()
        for raw in memory["entries"]:
            required = {
                "id",
                "content",
                "status",
                "tier",
                "t_created",
                "t_observed",
                "source_page_id",
            }
            if (
                type(raw) is not dict
                or set(raw) - set(MemoryEntry.model_fields)
                or not required <= set(raw)
            ):
                raise MigrationError()
            entry = MemoryEntry.model_validate(raw, strict=True)
            canonical_json(entry.model_dump())
            validate_identifier(entry.id)
            if entry.id in by_id or entry.source_page_id not in aliases:
                raise MigrationError()
            # A page containing a conflicting ID is ambiguous provenance, not a hint.
            linked = original_pages[entry.source_page_id].meta.get("memory_id")
            if linked is not None and linked != entry.id:
                raise MigrationError()
            if entry.source_page_id in referenced:
                raise MigrationError()
            referenced.add(entry.source_page_id)
            for field in (
                "t_created",
                "t_observed",
                "t_valid",
                "t_invalid",
                "t_expired",
                "last_accessed",
            ):
                value = getattr(entry, field)
                if (
                    value is not None
                    and datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo
                    is None
                ):
                    raise MigrationError()
            if (
                entry.t_valid
                and entry.t_invalid
                and datetime.fromisoformat(entry.t_valid.replace("Z", "+00:00"))
                > datetime.fromisoformat(entry.t_invalid.replace("Z", "+00:00"))
            ):
                raise MigrationError()
            if entry.t_expired and datetime.fromisoformat(
                entry.t_expired.replace("Z", "+00:00")
            ) < datetime.fromisoformat(entry.t_created.replace("Z", "+00:00")):
                raise MigrationError()
            by_id[entry.id] = entry
            entries.append(entry)
        children = set()
        for entry in entries:
            if entry.version_of:
                if entry.version_of not in by_id or entry.version_of in children:
                    raise MigrationError()
                children.add(entry.version_of)
                parent = by_id[entry.version_of]
                if parent.status == "active" or datetime.fromisoformat(
                    parent.t_created.replace("Z", "+00:00")
                ) > datetime.fromisoformat(entry.t_created.replace("Z", "+00:00")):
                    raise MigrationError()
                # Transaction time and validity time are separate axes. Compare
                # only explicit values; legacy missing endpoints stay missing.
                if parent.t_expired and datetime.fromisoformat(
                    parent.t_expired.replace("Z", "+00:00")
                ) > datetime.fromisoformat(entry.t_created.replace("Z", "+00:00")):
                    raise MigrationError()
                if (
                    parent.t_invalid
                    and entry.t_valid
                    and datetime.fromisoformat(parent.t_invalid.replace("Z", "+00:00"))
                    > datetime.fromisoformat(entry.t_valid.replace("Z", "+00:00"))
                ):
                    raise MigrationError()
            seen = set()
            current = entry
            while current.version_of:
                if current.id in seen:
                    raise MigrationError()
                seen.add(current.id)
                current = by_id[current.version_of]
        for alias, page in original_pages.items():
            linked = page.meta.get("memory_id")
            if linked is not None and (
                linked not in by_id or by_id[linked].source_page_id != alias
            ):
                raise MigrationError()
        for entry in entries:
            entry.source_page_id = aliases[entry.source_page_id]
        for alias, page in original_pages.items():
            page.meta["page_id"] = aliases[alias]
        return pages, entries, aliases
    except (
        DurableError,
        ValidationError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        UnicodeError,
        RecursionError,
    ):
        raise MigrationError() from None


def stage_legacy(
    source_dir: str | Path,
    *,
    quiesced: bool = False,
    max_bytes: int = MAX_SOURCE_BYTES,
    max_records: int = MAX_RECORDS,
) -> StagedLegacy:
    """Caller must stop legacy writers throughout acquisition and explicit cutover.

    A second bounded read detects changes during acquisition; it is not a lock
    against future writers. Original files are never changed or instantiated as stores.
    """
    if (
        quiesced is not True
        or type(max_bytes) is not int
        or not 1 <= max_bytes <= MAX_SOURCE_BYTES
        or type(max_records) is not int
        or not 1 <= max_records <= MAX_RECORDS
    ):
        raise MigrationError()
    directory = Path(source_dir)

    def acquire() -> tuple[bytes, bytes]:
        result = []
        remaining = max_bytes
        for name in ("advanced_memory_state.json", "pages.json"):
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise MigrationError()
            with path.open("rb") as stream:
                raw = stream.read(remaining + 1)
            if len(raw) > remaining:
                raise MigrationError()
            result.append(raw)
            remaining -= len(raw)
        return tuple(result)

    try:
        before = acquire()
        source_json = canonical_json(
            {"memory": _strict_json(before[0]), "pages": _strict_json(before[1])},
            max_bytes=MAX_SOURCE_BYTES,
            max_nodes=200000,
        )
        pages, entries, _ = prepare_records(source_json)
        if (
            len(pages) > max_records
            or len(entries) > max_records
            or before != acquire()
        ):
            raise MigrationError()
        digest = hashlib.sha256(source_json.encode()).hexdigest()
        return StagedLegacy(digest, source_json, len(pages), len(entries))
    except (OSError, ValueError, TypeError, DurableError, RecursionError):
        raise MigrationError() from None

# -*- coding: utf-8 -*-
"""
Advanced memory schema and store with:
- Bi-temporal fields
- Self-editing operations (add/update/delete/versioning)
- Hierarchical tiers
- Strength-based decay
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal, Sequence
from pydantic import BaseModel, Field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math
import uuid
import threading
from contextlib import nullcontext

from jitmind.utils.atomic_io import atomic_write_json
from jitmind.utils.file_lock import file_lock
from jitmind.scoping import DEFAULT_NAMESPACE, namespace_matches, normalize_namespace


MemoryStatus = Literal["active", "deleted", "superseded", "expired"]
MemoryTier = Literal["short", "mid", "long"]


class MemoryEntry(BaseModel):
    """Rich memory entry supporting temporal and lifecycle semantics."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Stable memory id")
    content: str = Field(..., description="Abstract content")
    status: MemoryStatus = Field(default="active", description="Lifecycle status")
    tier: MemoryTier = Field(default="short", description="Memory tier")

    # Bi-temporal fields
    t_created: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    t_observed: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    t_valid: Optional[str] = Field(default=None)
    t_invalid: Optional[str] = Field(default=None)
    t_expired: Optional[str] = Field(default=None)

    # Strength/decay
    last_accessed: Optional[str] = Field(default=None)
    strength: float = Field(default=1.0)

    # Provenance and versioning
    source_page_id: Optional[str] = Field(default=None)
    version_of: Optional[str] = Field(default=None)
    meta: Dict[str, Any] = Field(default_factory=dict)
    namespace: tuple[str, ...] = Field(
        default=DEFAULT_NAMESPACE,
        description="Component-safe isolation boundary for this memory",
    )


class AdvancedMemoryState(BaseModel):
    entries: List[MemoryEntry] = Field(default_factory=list)

    def to_abstracts(self, include_inactive: bool = False) -> List[str]:
        if include_inactive:
            return [e.content for e in self.entries]
        return [e.content for e in self.entries if e.status == "active"]


class AdvancedMemoryStore:
    """
    Advanced memory store with bi-temporal fields, decay, and tiers.
    Provides add/update/delete operations while remaining compatible with MemoryState.
    """

    def __init__(
        self,
        dir_path: Optional[str] = None,
        enable_auto_cleanup: bool = True,
        ttl_seconds: Optional[int] = None,
        retention_threshold: float = 0.2,
        short_to_mid_days: int = 7,
        mid_to_long_days: int = 30,
        short_to_mid_strength: float = 3.0,
        mid_to_long_strength: float = 6.0,
        demotion_enabled: bool = True,
        demotion_grace_days: int = 14,
        long_to_mid_retention: float = 0.35,
        mid_to_short_retention: float = 0.25,
        retain_history: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._dir_path = Path(dir_path) if dir_path else None
        self._enable_auto_cleanup = enable_auto_cleanup
        self._ttl_seconds = ttl_seconds
        self._retention_threshold = retention_threshold

        self._short_to_mid_days = short_to_mid_days
        self._mid_to_long_days = mid_to_long_days
        self._short_to_mid_strength = short_to_mid_strength
        self._mid_to_long_strength = mid_to_long_strength
        self._demotion_enabled = demotion_enabled
        self._demotion_grace_days = demotion_grace_days
        self._long_to_mid_retention = long_to_mid_retention
        self._mid_to_short_retention = mid_to_short_retention
        self._retain_history = retain_history

        self._state = AdvancedMemoryState()

        if self._dir_path:
            self._memory_file = self._dir_path / "advanced_memory_state.json"
            if self._memory_file.exists():
                self._state = self._load_from_disk()
                if self._enable_auto_cleanup:
                    self.cleanup_expired()
            else:
                # Backward compatibility: migrate legacy memory_state.json if present
                legacy_file = self._dir_path / "memory_state.json"
                if legacy_file.exists():
                    try:
                        with open(legacy_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, dict) and "abstracts" in data:
                            self._state.entries = [MemoryEntry(content=a) for a in data["abstracts"]]
                            self._save_to_disk()
                    except Exception as e:
                        print(f"Warning: Failed to migrate legacy memory_state.json: {e}")

    def _load_from_disk(self) -> AdvancedMemoryState:
        try:
            with open(self._memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "entries" in data:
                return AdvancedMemoryState(**data)
            return AdvancedMemoryState()
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Failed to load advanced memory state from {self._memory_file}: {e}")
            return AdvancedMemoryState()

    def _save_to_disk(self) -> None:
        if self._dir_path:
            self._dir_path.mkdir(parents=True, exist_ok=True)
            try:
                atomic_write_json(self._memory_file, self._state.model_dump(), ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning: Failed to save advanced memory state to {self._memory_file}: {e}")

    def _persistence_lock(self):
        if not self._dir_path:
            return nullcontext()
        return file_lock(Path(str(self._memory_file) + ".lock"))

    def _refresh_from_disk(self) -> None:
        if not self._dir_path:
            return
        if getattr(self, "_memory_file", None) is None:
            return
        if self._memory_file.exists():
            self._state = self._load_from_disk()
        else:
            self._state = AdvancedMemoryState()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _parse_ts(self, ts: Optional[str]) -> Optional[datetime]:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None

    def _retention(self, entry: MemoryEntry) -> float:
        # R = e^(-t/S)
        now = datetime.now(timezone.utc)
        last_ts = self._parse_ts(entry.last_accessed) or self._parse_ts(entry.t_observed) or self._parse_ts(entry.t_created)
        if last_ts is None:
            return 1.0
        # Use days as the base unit so defaults don't immediately expire memory.
        t_days = (now - last_ts).total_seconds() / 86400.0
        S_days = max(entry.strength, 1.0)
        return math.exp(-t_days / S_days)

    def _apply_ttl(self, entry: MemoryEntry) -> bool:
        if self._ttl_seconds is None:
            return False
        created = self._parse_ts(entry.t_created)
        if created is None:
            return False
        if datetime.now(timezone.utc) - created > timedelta(seconds=self._ttl_seconds):
            return True
        return False

    # ---- public ----
    def load(
        self,
        namespace: str | Sequence[str] | None = None,
        *,
        include_descendants: bool = False,
    ):
        with self._lock:
            self._refresh_from_disk()
            if self._enable_auto_cleanup:
                self.cleanup_expired()
            # return MemoryState-compatible object
            from .memory import MemoryState
            entries = [e for e in self._state.entries if e.status == "active"]
            if namespace is not None:
                entries = [
                    e
                    for e in entries
                    if namespace_matches(
                        e.namespace,
                        namespace,
                        include_descendants=include_descendants,
                    )
                ]
            return MemoryState(abstracts=[e.content for e in entries])

    def get_entries(
        self,
        include_inactive: bool = False,
        *,
        namespace: str | Sequence[str] | None = None,
        include_descendants: bool = False,
    ) -> List[MemoryEntry]:
        with self._lock:
            self._refresh_from_disk()
            entries = (
                list(self._state.entries)
                if include_inactive
                else [e for e in self._state.entries if e.status == "active"]
            )
            if namespace is None:
                return entries
            return [
                e
                for e in entries
                if namespace_matches(
                    e.namespace,
                    namespace,
                    include_descendants=include_descendants,
                )
            ]

    def save(self, state: Any) -> None:
        with self._lock:
            with self._persistence_lock():
                self._refresh_from_disk()
                # Accept MemoryState-like and map into entries if needed
                if hasattr(state, "abstracts"):
                    self._state.entries = [
                        MemoryEntry(content=abstract) for abstract in state.abstracts
                    ]
                elif isinstance(state, AdvancedMemoryState):
                    self._state = state
                if self._dir_path:
                    self._save_to_disk()

    def add_entry(self, entry: MemoryEntry) -> None:
        with self._lock:
            with self._persistence_lock():
                self._refresh_from_disk()
                entry.namespace = normalize_namespace(entry.namespace)
                self._state.entries.append(entry)
                self.promote_demote()
                if self._dir_path:
                    self._save_to_disk()

    def update_entry(
        self,
        entry_id: str,
        new_entry: MemoryEntry,
        *,
        namespace: str | Sequence[str] | None = None,
    ) -> None:
        with self._lock:
            with self._persistence_lock():
                self._refresh_from_disk()
                expected = normalize_namespace(namespace) if namespace is not None else None
                if expected is not None and tuple(new_entry.namespace) != expected:
                    raise ValueError("replacement memory must remain in the target namespace")
                matched = False
                # mark old as superseded, add new version
                for e in self._state.entries:
                    if (
                        e.id == entry_id
                        and e.status == "active"
                        and (expected is None or tuple(e.namespace) == expected)
                    ):
                        matched = True
                        e.status = "superseded"
                        e.t_invalid = new_entry.t_valid or self._now_iso()
                        e.t_expired = self._now_iso()
                if expected is not None and not matched:
                    raise LookupError("target memory does not exist in this namespace")
                self._state.entries.append(new_entry)
                self.promote_demote()
                if self._dir_path:
                    self._save_to_disk()

    def delete_entry(
        self,
        entry_id: str,
        *,
        namespace: str | Sequence[str] | None = None,
    ) -> None:
        with self._lock:
            with self._persistence_lock():
                self._refresh_from_disk()
                expected = normalize_namespace(namespace) if namespace is not None else None
                for e in self._state.entries:
                    if (
                        e.id == entry_id
                        and e.status == "active"
                        and (expected is None or tuple(e.namespace) == expected)
                    ):
                        e.status = "deleted"
                        e.t_invalid = e.t_invalid or self._now_iso()
                        e.t_expired = self._now_iso()
                if self._dir_path:
                    self._save_to_disk()

    def supersede_entry(
        self,
        entry_id: str,
        t_invalid: Optional[str] = None,
        *,
        namespace: str | Sequence[str] | None = None,
    ) -> None:
        with self._lock:
            with self._persistence_lock():
                self._refresh_from_disk()
                expected = normalize_namespace(namespace) if namespace is not None else None
                for e in self._state.entries:
                    if (
                        e.id == entry_id
                        and e.status == "active"
                        and (expected is None or tuple(e.namespace) == expected)
                    ):
                        e.status = "superseded"
                        e.t_invalid = t_invalid or self._now_iso()
                        e.t_expired = self._now_iso()
                if self._dir_path:
                    self._save_to_disk()

    # Backward-compatible add
    def add(self, abstract: str) -> None:
        if not abstract:
            return
        entry = MemoryEntry(content=abstract)
        self.add_entry(entry)

    def touch(self, entry_ids: List[str]) -> None:
        with self._lock:
            with self._persistence_lock():
                self._refresh_from_disk()
                now = self._now_iso()
                for e in self._state.entries:
                    if e.id in entry_ids and e.status == "active":
                        e.last_accessed = now
                        e.strength = e.strength + 1.0
                # Re-evaluate tiers after strength updates
                self.promote_demote()
                if self._dir_path:
                    self._save_to_disk()

    def cleanup_expired(self) -> int:
        """Mark expired entries and optionally purge them from memory."""
        with self._lock:
            with self._persistence_lock():
                self._refresh_from_disk()
            removed = 0
            now = datetime.now(timezone.utc)
            for e in self._state.entries:
                if e.status != "active":
                    continue
                t_invalid = self._parse_ts(e.t_invalid)
                if t_invalid and t_invalid <= now:
                    e.status = "expired"
                    e.t_expired = self._now_iso()
                    removed += 1
                    continue
                if self._apply_ttl(e):
                    e.status = "expired"
                    e.t_expired = self._now_iso()
                    removed += 1
                    continue
                if self._retention(e) < self._retention_threshold:
                    e.status = "expired"
                    e.t_expired = self._now_iso()
                    removed += 1

            if not self._retain_history:
                self._state.entries = [
                    e for e in self._state.entries if e.status == "active"
                ]

            if removed and self._dir_path:
                self._save_to_disk()
            return removed

    def _coerce_datetime(self, value: str | datetime, *, field_name: str) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = self._parse_ts(value)
            if parsed is None:
                raise ValueError(f"{field_name} must be an ISO-8601 timestamp")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def query_as_of(
        self,
        valid_at: str | datetime,
        *,
        transaction_at: str | datetime | None = None,
        namespace: str | Sequence[str] | None = None,
        include_descendants: bool = False,
    ) -> List[MemoryEntry]:
        """Return facts valid at a time, optionally as known at a transaction time.

        With no ``transaction_at`` the complete retained ledger is consulted,
        which answers "what is now known to have been valid then?". Supplying a
        transaction time answers "what did the store believe at that moment?".
        """
        valid = self._coerce_datetime(valid_at, field_name="valid_at")
        transaction = (
            self._coerce_datetime(transaction_at, field_name="transaction_at")
            if transaction_at is not None
            else None
        )
        with self._lock:
            self._refresh_from_disk()
            matches: List[MemoryEntry] = []
            for entry in self._state.entries:
                if namespace is not None and not namespace_matches(
                    entry.namespace,
                    namespace,
                    include_descendants=include_descendants,
                ):
                    continue
                valid_start = (
                    self._parse_ts(entry.t_valid)
                    or self._parse_ts(entry.t_observed)
                    or self._parse_ts(entry.t_created)
                )
                valid_end = self._parse_ts(entry.t_invalid)
                if valid_start is None or valid < valid_start:
                    continue
                if valid_end is not None and valid >= valid_end:
                    continue
                if transaction is not None:
                    created = self._parse_ts(entry.t_created)
                    retired = self._parse_ts(entry.t_expired)
                    if created is not None and transaction < created:
                        continue
                    if retired is not None and transaction >= retired:
                        continue
                matches.append(entry.model_copy(deep=True))
            matches.sort(key=lambda entry: (entry.t_valid or entry.t_observed, entry.id))
            return matches

    def get_version_history(self, entry_id: str) -> List[MemoryEntry]:
        """Return the complete oldest-to-newest version chain for an entry."""
        with self._lock:
            self._refresh_from_disk()
            by_id = {entry.id: entry for entry in self._state.entries}
            current = by_id.get(entry_id)
            if current is None:
                return []
            seen: set[str] = set()
            while current.version_of and current.version_of in by_id:
                if current.id in seen:
                    break
                seen.add(current.id)
                current = by_id[current.version_of]
            root_id = current.id

            def root_of(entry: MemoryEntry) -> str:
                cursor = entry
                visited: set[str] = set()
                while cursor.version_of and cursor.version_of in by_id:
                    if cursor.id in visited:
                        break
                    visited.add(cursor.id)
                    cursor = by_id[cursor.version_of]
                return cursor.id

            history = [
                entry.model_copy(deep=True)
                for entry in self._state.entries
                if root_of(entry) == root_id
            ]
            history.sort(key=lambda entry: (entry.t_created, entry.id))
            return history

    def diff_as_of(
        self,
        before: str | datetime,
        after: str | datetime,
        *,
        transaction_at: str | datetime | None = None,
    ) -> Dict[str, List[MemoryEntry]]:
        """Return added, removed, and changed facts between two valid times."""
        before_entries = self.query_as_of(before, transaction_at=transaction_at)
        after_entries = self.query_as_of(after, transaction_at=transaction_at)

        def roots(entries: List[MemoryEntry]) -> Dict[str, MemoryEntry]:
            mapped: Dict[str, MemoryEntry] = {}
            for entry in entries:
                history = self.get_version_history(entry.id)
                root = history[0].id if history else entry.id
                mapped[root] = entry
            return mapped

        old = roots(before_entries)
        new = roots(after_entries)
        return {
            "added": [new[key] for key in sorted(new.keys() - old.keys())],
            "removed": [old[key] for key in sorted(old.keys() - new.keys())],
            "changed": [
                new[key]
                for key in sorted(old.keys() & new.keys())
                if old[key].id != new[key].id or old[key].content != new[key].content
            ],
        }

    def promote_demote(self) -> None:
        with self._lock:
            now = datetime.now(timezone.utc)
            for e in self._state.entries:
                if e.status != "active":
                    continue
                observed = self._parse_ts(e.t_observed) or now
                age_days = (now - observed).days
                promoted = False
                # Promotion: require time + strength, or very high strength
                if e.tier == "short":
                    if (age_days >= self._short_to_mid_days and e.strength >= self._short_to_mid_strength) or \
                       (e.strength >= self._short_to_mid_strength * 2):
                        e.tier = "mid"
                        promoted = True
                elif e.tier == "mid":
                    if (age_days >= self._mid_to_long_days and e.strength >= self._mid_to_long_strength) or \
                       (e.strength >= self._mid_to_long_strength * 2):
                        e.tier = "long"
                        promoted = True

                # Demotion: only if not just promoted
                if self._demotion_enabled and not promoted:
                    last_ts = self._parse_ts(e.last_accessed) or observed
                    inactive_days = (now - last_ts).days
                    retention = self._retention(e)
                    if e.tier == "long":
                        if inactive_days >= self._demotion_grace_days and retention < self._long_to_mid_retention:
                            e.tier = "mid"
                    elif e.tier == "mid":
                        if inactive_days >= self._demotion_grace_days and retention < self._mid_to_short_retention:
                            e.tier = "short"

    # ---- Tier-weighted retrieval ----
    TIER_WEIGHTS: Dict[str, float] = {"long": 1.5, "mid": 1.2, "short": 1.0}

    def get_tier_score(self, entry_id: str) -> float:
        """Get tier-based score multiplier for an entry."""
        with self._lock:
            self._refresh_from_disk()
            for e in self._state.entries:
                if e.id == entry_id and e.status == "active":
                    return self.TIER_WEIGHTS.get(e.tier, 1.0) * self._retention(e)
            return 1.0

    def get_entry_by_page_id(self, page_id: str) -> Optional[MemoryEntry]:
        """Find entry by source page ID."""
        with self._lock:
            self._refresh_from_disk()
            for e in self._state.entries:
                if e.source_page_id == page_id and e.status == "active":
                    return e
            return None

    def get_entry_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get entry by ID."""
        with self._lock:
            self._refresh_from_disk()
            for e in self._state.entries:
                if e.id == entry_id:
                    return e
            return None

    def get_ranked_entries(
        self,
        limit: int = 50,
        *,
        namespace: str | Sequence[str] | None = None,
    ) -> List[MemoryEntry]:
        with self._lock:
            self._refresh_from_disk()
            active = [e for e in self._state.entries if e.status == "active"]
            if namespace is not None:
                active = [e for e in active if namespace_matches(e.namespace, namespace)]
            self.promote_demote()
            scored = [
                (e, self.TIER_WEIGHTS.get(e.tier, 1.0) * self._retention(e))
                for e in active
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [e for e, _ in scored[:limit]]

    def get_ranked_abstracts(self, limit: int = 50) -> List[str]:
        """Get abstracts ranked by tier and retention score."""
        with self._lock:
            self._refresh_from_disk()
            active = [e for e in self._state.entries if e.status == "active"]
            # Trigger promotion check on retrieval
            self.promote_demote()
            # Sort by tier weight * retention
            scored = [
                (e, self.TIER_WEIGHTS.get(e.tier, 1.0) * self._retention(e))
                for e in active
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [e.content for e, _ in scored[:limit]]

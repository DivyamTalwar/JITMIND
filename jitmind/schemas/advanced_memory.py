# -*- coding: utf-8 -*-
"""
Advanced memory schema and store with:
- Bi-temporal fields
- Self-editing operations (add/update/delete/versioning)
- Hierarchical tiers
- Strength-based decay
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
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
    def load(self):
        with self._lock:
            self._refresh_from_disk()
            if self._enable_auto_cleanup:
                self.cleanup_expired()
            # return MemoryState-compatible object
            from .memory import MemoryState
            return MemoryState(abstracts=self._state.to_abstracts(include_inactive=False))

    def get_entries(self, include_inactive: bool = False) -> List[MemoryEntry]:
        with self._lock:
            self._refresh_from_disk()
            if include_inactive:
                return list(self._state.entries)
            return [e for e in self._state.entries if e.status == "active"]

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
                self._state.entries.append(entry)
                self.promote_demote()
                if self._dir_path:
                    self._save_to_disk()

    def update_entry(self, entry_id: str, new_entry: MemoryEntry) -> None:
        with self._lock:
            with self._persistence_lock():
                self._refresh_from_disk()
                # mark old as superseded, add new version
                for e in self._state.entries:
                    if e.id == entry_id and e.status == "active":
                        e.status = "superseded"
                        e.t_invalid = new_entry.t_valid or self._now_iso()
                        e.t_expired = self._now_iso()
                self._state.entries.append(new_entry)
                self.promote_demote()
                if self._dir_path:
                    self._save_to_disk()

    def delete_entry(self, entry_id: str) -> None:
        with self._lock:
            with self._persistence_lock():
                self._refresh_from_disk()
                for e in self._state.entries:
                    if e.id == entry_id and e.status == "active":
                        e.status = "deleted"
                        e.t_invalid = e.t_invalid or self._now_iso()
                        e.t_expired = self._now_iso()
                if self._dir_path:
                    self._save_to_disk()

    def supersede_entry(self, entry_id: str, t_invalid: Optional[str] = None) -> None:
        with self._lock:
            with self._persistence_lock():
                self._refresh_from_disk()
                for e in self._state.entries:
                    if e.id == entry_id and e.status == "active":
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
            to_remove_ids: List[str] = []

            for e in self._state.entries:
                if e.status != "active":
                    # Already expired/deleted - mark for removal from list
                    to_remove_ids.append(e.id)
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

            # Purge all non-active entries to prevent memory leak
            self._state.entries = [e for e in self._state.entries if e.status == "active"]

            if (removed or to_remove_ids) and self._dir_path:
                self._save_to_disk()
            return removed

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

    def get_ranked_entries(self, limit: int = 50) -> List[MemoryEntry]:
        with self._lock:
            self._refresh_from_disk()
            active = [e for e in self._state.entries if e.status == "active"]
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

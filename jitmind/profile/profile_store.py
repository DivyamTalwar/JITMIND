from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime, timezone

from jitmind.utils.atomic_io import atomic_write_json, CorruptStoreError
from jitmind.utils.file_lock import file_lock

@dataclass
class UserProfile:
    user_id: str
    static: Dict[str, Any] = field(default_factory=dict)  # stable preferences / facts
    dynamic: Dict[str, Any] = field(default_factory=dict)  # recent context, ephemeral signals
    traits: Dict[str, Any] = field(default_factory=dict)  # long-term personality cues
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class UserProfileStore:
    def __init__(self, dir_path: str = "./profiles") -> None:
        self._dir = Path(dir_path)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: str) -> Path:
        safe_id = "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))
        return self._dir / f"{safe_id}.json"

    def load(self, user_id: str) -> UserProfile:
        path = self._path(user_id)
        if not path.exists():
            return UserProfile(user_id=user_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not any(key in data for key in ("static", "dynamic", "traits")):
                raise ValueError("Invalid profile envelope")
            if any(not isinstance(data.get(key, {}), dict) for key in ("static", "dynamic", "traits")):
                raise ValueError("Invalid profile fields")
            return UserProfile(
                user_id=user_id,
                static=data.get("static", {}),
                dynamic=data.get("dynamic", {}),
                traits=data.get("traits", {}),
                updated_at=data.get("updated_at") or UserProfile(user_id).updated_at,
            )
        except Exception as exc:
            raise CorruptStoreError() from exc

    def save(self, profile: UserProfile) -> None:
        path = self._path(profile.user_id)
        with file_lock(Path(str(path) + ".lock")):
            self.load(profile.user_id)
            updated_at = datetime.now(timezone.utc).isoformat()
            payload = {
                "user_id": profile.user_id,
                "static": profile.static,
                "dynamic": profile.dynamic,
                "traits": profile.traits,
                "updated_at": updated_at,
            }
            atomic_write_json(path, payload, ensure_ascii=False, indent=2)
            profile.updated_at = updated_at

    def update(
        self,
        user_id: str,
        static_updates: Optional[Dict[str, Any]] = None,
        dynamic_updates: Optional[Dict[str, Any]] = None,
        traits_updates: Optional[Dict[str, Any]] = None,
    ) -> UserProfile:
        path = self._path(user_id)
        lock_path = Path(str(path) + ".lock")
        with file_lock(lock_path):
            profile = self.load(user_id)
            if static_updates:
                profile.static.update(static_updates)
            if dynamic_updates:
                profile.dynamic.update(dynamic_updates)
            if traits_updates:
                profile.traits.update(traits_updates)
            self.save(profile)
            return profile

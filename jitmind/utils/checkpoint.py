# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timezone
import json

from jitmind.utils.atomic_io import atomic_write_json
from jitmind.utils.file_lock import file_lock

class CheckpointManager:
    def __init__(self, dir_path: str = "./checkpoints") -> None:
        self._dir = Path(dir_path)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, thread_id: str) -> Path:
        safe_id = "".join(c for c in thread_id if c.isalnum() or c in ("-", "_"))
        return self._dir / f"{safe_id}.json"

    def save_checkpoint(self, thread_id: str, state: Dict[str, Any]) -> None:
        payload = {
            "thread_id": thread_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": state,
        }
        path = self._path(thread_id)
        lock_path = Path(str(path) + ".lock")
        with file_lock(lock_path):
            atomic_write_json(path, payload, ensure_ascii=False, indent=2)

    def load_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(thread_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("state")
        except Exception:
            return None

    def list_checkpoints(self) -> List[str]:
        return [p.stem for p in self._dir.glob("*.json")]

    def delete_checkpoint(self, thread_id: str) -> bool:
        path = self._path(thread_id)
        if not path.exists():
            return False
        path.unlink()
        return True

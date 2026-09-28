from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING
from pydantic import BaseModel, Field
import json
from pathlib import Path
import threading
from contextlib import nullcontext

from jitmind.utils.atomic_io import atomic_write_json, CorruptStoreError, PersistenceError
from jitmind.utils.file_lock import file_lock

if TYPE_CHECKING:
    from .page import Page


class MemoryState(BaseModel):
    """Long-term memory: only abstracts list."""
    abstracts: List[str] = Field(default_factory=list, description="List of memory abstracts")

class MemoryUpdate(BaseModel):
    """Memory update result"""
    new_state: MemoryState = Field(..., description="Updated memory state")
    new_page: 'Page' = Field(..., description="New page added")
    debug: Dict[str, Any] = Field(default_factory=dict, description="Debug information")

class MemoryStore(Protocol):
    def load(self) -> MemoryState: ...
    def save(self, state: MemoryState) -> None: ...
    def add(self, abstract: str) -> None: ...

class InMemoryMemoryStore:
    def __init__(self, dir_path: Optional[str] = None, init_state: Optional[MemoryState] = None) -> None:
        self._dir_path = Path(dir_path) if dir_path else None
        self._state = init_state or MemoryState()
        self._lock = threading.RLock()
        if self._dir_path:
            self._memory_file = self._dir_path / "memory_state.json"
            if self._memory_file.exists():
                self._state = self.load()

    def load(self) -> MemoryState:
        with self._lock:
            if self._dir_path and self._memory_file.exists():
                try:
                    with open(self._memory_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if not isinstance(data, dict) or "abstracts" not in data:
                            raise ValueError("Invalid memory envelope")
                        self._state = MemoryState(**data)
                except Exception as exc:
                    raise CorruptStoreError() from exc
            return self._state.model_copy(deep=True)

    def save(self, state: MemoryState) -> None:
        with self._lock:
            lock = file_lock(Path(str(self._memory_file) + ".lock")) if self._dir_path else nullcontext()
            with lock:
                if self._dir_path:
                    # Refuse to overwrite corrupt data even for explicit save.
                    self.load()
                    try:
                        atomic_write_json(self._memory_file, state.model_dump(), ensure_ascii=False, indent=2)
                    except Exception as exc:
                        self._state = MemoryState()
                        try:
                            self.load()
                        except Exception:
                            pass
                        if isinstance(exc, PersistenceError):
                            raise
                        raise PersistenceError() from exc
                self._state = state.model_copy(deep=True)

    def add(self, abstract: str) -> None:
        with self._lock:
            if not abstract:
                return
            if self._dir_path:
                lock_path = Path(str(self._memory_file) + ".lock")
                with file_lock(lock_path):
                    state = self.load()
                    if abstract not in state.abstracts:
                        state.abstracts.append(abstract)
                        self.save(state)
            else:
                if abstract not in self._state.abstracts:
                    self._state.abstracts.append(abstract)

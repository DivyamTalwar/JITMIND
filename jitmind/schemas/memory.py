from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol
from pydantic import BaseModel, Field
import json
from pathlib import Path
import threading

from jitmind.utils.atomic_io import atomic_write_json
from jitmind.utils.file_lock import file_lock

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
                    with open(self._memory_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        return MemoryState(**data)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"Warning: Failed to load memory state from {self._memory_file}: {e}")
                    return MemoryState()
            return self._state

    def save(self, state: MemoryState) -> None:
        with self._lock:
            self._state = state
            if self._dir_path:
                self._dir_path.mkdir(parents=True, exist_ok=True)
                try:
                    atomic_write_json(self._memory_file, state.model_dump(), ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Warning: Failed to save memory state to {self._memory_file}: {e}")

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

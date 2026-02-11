from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol
from pydantic import BaseModel, Field
import json
from pathlib import Path
import threading

from jitmind.utils.atomic_io import atomic_write_json
from jitmind.utils.file_lock import file_lock

class Page(BaseModel):
    """Page data structure"""
    header: str = Field(..., description="Page header")
    content: str = Field(..., description="Page content")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata")

    @staticmethod
    def equal(page1: 'Page', page2: 'Page') -> bool:
        return page1 == page2

class PageStore(Protocol):
    def add(self, page: Page) -> None: ...
    def load(self) -> List[Page]: ...
    def save(self, pages: List[Page]) -> None: ...

class InMemoryPageStore:
    """
    Simple append-only list store for Page.
    Uses file system persistence.
    """
    def __init__(self, dir_path: Optional[str] = None) -> None:
        self._dir_path = Path(dir_path) if dir_path else None
        self._pages: List[Page] = []
        self._lock = threading.RLock()
        if self._dir_path:
            self._pages_file = self._dir_path / "pages.json"
            if self._pages_file.exists():
                self._pages = self.load()

    def load(self) -> List[Page]:
        with self._lock:
            if self._dir_path and self._pages_file.exists():
                try:
                    with open(self._pages_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            return [Page(**page_data) for page_data in data]
                        else:
                            return [Page(**page_data) for page_data in data.get('pages', [])]
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"Warning: Failed to load pages from {self._pages_file}: {e}")
                    return []
            return self._pages

    def save(self, pages: List[Page]) -> None:
        with self._lock:
            self._pages = pages
            if self._dir_path:
                self._dir_path.mkdir(parents=True, exist_ok=True)
                try:
                    pages_data = [page.model_dump() for page in pages]
                    atomic_write_json(self._pages_file, pages_data, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Warning: Failed to save pages to {self._pages_file}: {e}")

    def add(self, page: Page) -> None:
        with self._lock:
            if self._dir_path:
                lock_path = Path(str(self._pages_file) + ".lock")
                with file_lock(lock_path):
                    # Refresh from disk to avoid lost updates across processes.
                    pages = self.load()
                    pages.append(page)
                    self.save(pages)
            else:
                self._pages.append(page)

    def get(self, index: int) -> Optional[Page]:
        if 0 <= index < len(self._pages):
            return self._pages[index]
        return None

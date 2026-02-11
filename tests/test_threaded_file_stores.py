from __future__ import annotations

import json
import threading

from jitmind.schemas import InMemoryPageStore, Page
from jitmind.schemas.advanced_memory import AdvancedMemoryStore, MemoryEntry


def test_page_store_threaded_writes_are_valid_json(tmp_path):
    store = InMemoryPageStore(dir_path=str(tmp_path))

    def worker(tid: int) -> None:
        for i in range(50):
            store.add(Page(header=f"h{tid}-{i}", content=f"c{tid}-{i}", meta={"tid": tid, "i": i}))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # File should be valid JSON and contain all pages.
    pages_path = tmp_path / "pages.json"
    data = json.loads(pages_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 500

    pages = store.load()
    assert len(pages) == 500


def test_advanced_memory_store_threaded_add_is_consistent(tmp_path):
    store = AdvancedMemoryStore(dir_path=str(tmp_path))

    def worker(tid: int) -> None:
        for i in range(50):
            store.add_entry(MemoryEntry(content=f"m{tid}-{i}"))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # Reload from disk to validate persisted state.
    store2 = AdvancedMemoryStore(dir_path=str(tmp_path))
    entries = store2.get_entries(include_inactive=True)
    assert len(entries) == 500


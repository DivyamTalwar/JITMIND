import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from test_durable_storage import NOW, write

from jitmind.schemas import MemoryEntry, Page
from jitmind.storage import (
    MigrationError,
    SQLiteDurableStore,
    StorageFailure,
    stage_legacy,
)


def legacy_source(directory):
    first = MemoryEntry(
        id="stable-old",
        content="old fact",
        status="superseded",
        tier="long",
        t_created=NOW,
        t_observed=NOW,
        t_valid=NOW,
        t_invalid="2026-02-01T00:00:00+00:00",
        t_expired="2026-02-01T00:00:00+00:00",
        source_page_id="0",
        strength=2.5,
        meta={"custom": {"tags": ["a", "b"]}},
    )
    second = MemoryEntry(
        id="stable-new",
        content="new fact",
        version_of=first.id,
        t_created="2026-02-01T00:00:00+00:00",
        t_observed=NOW,
        source_page_id="legacy-string",
        meta={"origin": "keep"},
    )
    pages = [
        Page(
            header="one",
            content="old source",
            meta={"page_id": 0, "memory_id": first.id, "custom": [1, 2]},
        ),
        Page(
            header="two",
            content="new source",
            meta={
                "page_id": "legacy-string",
                "memory_id": second.id,
                "custom": "value",
            },
        ),
    ]
    (directory / "advanced_memory_state.json").write_text(
        json.dumps({"entries": [e.model_dump() for e in (first, second)]})
    )
    (directory / "pages.json").write_text(json.dumps([p.model_dump() for p in pages]))
    return first, second


def test_migration_preserves_ids_metadata_temporal_history_aliases(tmp_path):
    first, second = legacy_source(tmp_path)
    originals = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    staged = stage_legacy(tmp_path, quiesced=True)
    store = SQLiteDurableStore(tmp_path / "candidate")
    report = store.import_staged(staged, "a")
    assert report == {
        "source_digest": staged.source_digest,
        "page_count": 2,
        "fact_count": 2,
    }
    for original in (first, second):
        imported = store.get_entry("a", original.id, include_inactive=True)
        expected = original.model_dump()
        expected["source_page_id"] = store.resolve_page_alias(
            "a", original.source_page_id
        )
        assert imported.model_dump() == expected
    page_id = store.resolve_page_alias("a", "legacy-string")
    assert store.get_page("a", page_id).meta["custom"] == "value"
    assert store.get_page("a", page_id).meta["page_id"] == page_id
    assert store.resolve_page_alias("b", "legacy-string") is None
    assert store.get_page("a", store.resolve_page_alias("a", "0")) is None
    assert store.import_staged(staged, "a") == report
    assert store.status("a")["revision"] == 1
    assert store.drain_outbox("a") == 1
    assert [e.id for e in store.projected_entries("a")] == [second.id]
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.json")} == originals


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate_id",
        "dangling_page",
        "missing_page_id",
        "duplicate_page_alias",
        "dangling_version",
        "cycle",
        "branch",
        "bad_time",
        "reversed_time",
        "bad_meta",
        "unknown_field",
        "wrong_page_memory",
        "nan",
        "truncated",
        "duplicate_json_key",
    ],
)
def test_invalid_migration_refuses_without_changes(tmp_path, damage):
    legacy_source(tmp_path)
    memory_path = tmp_path / "advanced_memory_state.json"
    pages_path = tmp_path / "pages.json"
    data = json.loads(memory_path.read_text())
    pages = json.loads(pages_path.read_text())
    old, new = data["entries"]
    if damage == "duplicate_id":
        new["id"] = old["id"]
    elif damage == "dangling_page":
        new["source_page_id"] = "missing"
    elif damage == "missing_page_id":
        del pages[0]["meta"]["page_id"]
    elif damage == "duplicate_page_alias":
        pages[1]["meta"]["page_id"] = "0"
    elif damage == "dangling_version":
        new["version_of"] = "missing"
    elif damage == "cycle":
        old["version_of"] = new["id"]
    elif damage == "branch":
        third = dict(new, id="third", source_page_id="third")
        data["entries"].append(third)
        pages.append(
            {
                "header": "third",
                "content": "third",
                "meta": {"page_id": "third", "memory_id": "third"},
            }
        )
    elif damage == "bad_time":
        new["t_observed"] = "yesterday"
    elif damage == "reversed_time":
        new["t_valid"], new["t_invalid"] = "2026-02-01T00:00:00+00:00", NOW
    elif damage == "bad_meta":
        new["meta"] = {"__proto__": {}}
    elif damage == "unknown_field":
        new["unknown"] = "must not drop"
    elif damage == "wrong_page_memory":
        pages[1]["meta"]["memory_id"] = "other"
    elif damage == "nan":
        new["strength"] = float("nan")
    memory_path.write_text(json.dumps(data))
    pages_path.write_text(json.dumps(pages))
    if damage == "truncated":
        memory_path.write_text('{"entries":[')
    if damage == "duplicate_json_key":
        memory_path.write_text('{"entries":[],"entries":[]}')
    before = {p.name: p.read_bytes() for p in (memory_path, pages_path)}
    with pytest.raises(MigrationError, match="^invalid_legacy_source$"):
        stage_legacy(tmp_path, quiesced=True)
    assert before == {p.name: p.read_bytes() for p in (memory_path, pages_path)}
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "advanced_memory_state.json",
        "pages.json",
    ]


def test_staging_requires_quiescence_and_detects_changed_source(tmp_path):
    legacy_source(tmp_path)
    with pytest.raises(MigrationError):
        stage_legacy(tmp_path)
    with pytest.raises(MigrationError):
        stage_legacy(tmp_path, quiesced=True, max_bytes=1)
    with pytest.raises(MigrationError):
        stage_legacy(tmp_path, quiesced=True, max_records=1)
    original = Path.open
    reads = 0

    def changed_open(path, *args, **kwargs):
        nonlocal reads
        if path.name == "advanced_memory_state.json" and args and args[0] == "rb":
            reads += 1
            if reads == 2:
                with original(path, "ab") as stream:
                    stream.write(b" ")
        return original(path, *args, **kwargs)

    with patch.object(Path, "open", changed_open), pytest.raises(MigrationError):
        stage_legacy(tmp_path, quiesced=True)


@pytest.mark.parametrize(
    "stage", ["before_import", "after_import_pages", "before_import_commit"]
)
def test_staged_failure_leaves_legacy_and_candidate_unchanged(tmp_path, stage):
    legacy_source(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    staged = stage_legacy(tmp_path, quiesced=True)

    def fault(current):
        if current == stage:
            raise RuntimeError("fail")

    store = SQLiteDurableStore(tmp_path / "candidate", fault_hook=fault)
    with pytest.raises(StorageFailure):
        store.import_staged(staged, "a")
    fresh = SQLiteDurableStore(tmp_path / "candidate")
    assert fresh.status("a")["pages"] == fresh.status("a")["facts"] == 0
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    fresh.import_staged(staged, "a")
    assert fresh.status("a")["pages"] == 2


def test_backup_reopen_restore_preserves_candidate_era_writes(tmp_path):
    legacy_source(tmp_path)
    staged = stage_legacy(tmp_path, quiesced=True)
    store = SQLiteDurableStore(tmp_path / "candidate")
    store.import_staged(staged, "a")
    older = store.backup(tmp_path / "older")
    receipt = write(store, "new-after-import")
    current = store.backup(tmp_path / "current")
    assert SQLiteDurableStore(current).status("a")["revision"] == 2
    assert SQLiteDurableStore(older).status("a")["revision"] == 1
    restored = SQLiteDurableStore.restore(current, tmp_path / "restored")
    assert restored.get_entry("a", receipt.memory_id).source_page_id == receipt.page_id
    assert write(restored, "new-after-import") == receipt
    assert restored.resolve_page_alias("a", "0") == store.resolve_page_alias("a", "0")
    # An old backup cannot overwrite the candidate and lose new writes.
    with pytest.raises(StorageFailure):
        SQLiteDurableStore.restore(older, tmp_path / "candidate")
    assert store.get_entry("a", receipt.memory_id) is not None
    with pytest.raises(StorageFailure):
        store.backup(current)
    assert SQLiteDurableStore(current).status("a")["revision"] == 2
    assert restored.status("a") == store.status("a")
    assert not sqlite3.connect(current).execute("PRAGMA foreign_key_check").fetchall()


def test_bounded_adapters_reject_unbounded_loads(tmp_path):
    from jitmind.storage import (
        CapacityExceeded,
        DurableMemoryAdapter,
        DurablePageAdapter,
    )

    entries = [
        MemoryEntry(
            id=f"fact-{i}",
            content=f"fact {i}",
            t_created=NOW,
            t_observed=NOW,
            source_page_id=str(i),
        )
        for i in range(1000)
    ]
    pages = [
        Page(
            header="h", content=f"page {i}", meta={"page_id": i, "memory_id": entry.id}
        )
        for i, entry in enumerate(entries)
    ]
    (tmp_path / "advanced_memory_state.json").write_text(
        json.dumps({"entries": [e.model_dump() for e in entries]})
    )
    (tmp_path / "pages.json").write_text(
        json.dumps({"pages": [p.model_dump() for p in pages]})
    )
    store = SQLiteDurableStore(tmp_path / "candidate")
    store.import_staged(stage_legacy(tmp_path, quiesced=True), "a")
    memory, page = DurableMemoryAdapter(store, "a"), DurablePageAdapter(store, "a")
    assert len(memory.load().abstracts) == len(page.load()) == 1000
    receipt = write(store, "after-import")
    with pytest.raises(CapacityExceeded):
        memory.load()
    with pytest.raises(CapacityExceeded):
        page.load()
    assert store.memory_update(receipt, limit=10).debug["state_truncated"]
    assert len(memory.get_entries(limit=10)) == len(page.list_pages(limit=10)) == 10


def test_deep_json_is_a_safe_migration_error(tmp_path):
    (tmp_path / "advanced_memory_state.json").write_text("[" * 2000 + "0" + "]" * 2000)
    (tmp_path / "pages.json").write_text("[]")
    with pytest.raises(MigrationError, match="^invalid_legacy_source$"):
        stage_legacy(tmp_path, quiesced=True)

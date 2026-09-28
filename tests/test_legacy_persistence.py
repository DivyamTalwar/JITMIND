import json
import multiprocessing as mp
import os
from datetime import datetime, timedelta, timezone

import pytest
from test_file_lock_regressions import finish, observe_contention

from jitmind.schemas.advanced_memory import AdvancedMemoryStore, MemoryEntry
from jitmind.schemas.memory import InMemoryMemoryStore, MemoryState
from jitmind.schemas.page import InMemoryPageStore, Page
from jitmind.schemas.ttl_memory import TTLMemoryStore
from jitmind.schemas.ttl_page import TTLPageStore
from jitmind.utils.atomic_io import CorruptStoreError, PersistenceError


def test_cleanup_persists_previously_inactive_rows(tmp_path):
    store = AdvancedMemoryStore(str(tmp_path), enable_auto_cleanup=False)
    store.add_entry(MemoryEntry(content="old", status="deleted"))
    purge = AdvancedMemoryStore(
        str(tmp_path), enable_auto_cleanup=False, retain_history=False
    )
    assert purge.cleanup_expired() == 0
    assert (
        AdvancedMemoryStore(str(tmp_path), enable_auto_cleanup=False).get_entries(True)
        == []
    )


def test_advanced_save_failure_reaches_caller(tmp_path, monkeypatch):
    import jitmind.utils.atomic_io as io

    store = AdvancedMemoryStore(str(tmp_path), enable_auto_cleanup=False)
    store.add("old")

    def fail(*args):
        raise OSError("private path")

    monkeypatch.setattr(io.os, "replace", fail)
    with pytest.raises(PersistenceError):
        store.add("unpublished")
    assert store.load().abstracts == ["old"]


CTX = mp.get_context("spawn")


def schedule_worker(directory, operation, paused, release, contended, pause):
    observe_contention(contended)
    store = AdvancedMemoryStore(directory, enable_auto_cleanup=False, ttl_seconds=60)
    method = "_apply_ttl" if operation == "cleanup" else "_save_to_disk"
    original = getattr(store, method)
    if pause:

        def blocked(*args, **kwargs):
            paused.set()
            assert release.wait(12)
            return original(*args, **kwargs)

        setattr(store, method, blocked)
    if operation == "cleanup":
        store.cleanup_expired()
    else:
        store.add("B")


@pytest.mark.skipif(os.name != "posix", reason="POSIX locking")
@pytest.mark.parametrize("first", ["cleanup", "append"])
def test_cleanup_append_schedule(tmp_path, first):
    store = AdvancedMemoryStore(str(tmp_path), enable_auto_cleanup=False)
    old = datetime.now(timezone.utc) - timedelta(days=3)
    store.add_entry(MemoryEntry(id="expired", content="old", t_created=old.isoformat()))
    paused, release, contended = CTX.Event(), CTX.Event(), CTX.Event()
    spare_events = [CTX.Event() for _ in range(3)]
    processes = [
        CTX.Process(
            target=schedule_worker,
            args=(str(tmp_path), first, paused, release, spare_events[0], True),
        ),
        CTX.Process(
            target=schedule_worker,
            args=(
                str(tmp_path),
                "append" if first == "cleanup" else "cleanup",
                spare_events[1],
                spare_events[2],
                contended,
                False,
            ),
        ),
    ]
    try:
        processes[0].start()
        assert paused.wait(8)
        processes[1].start()
        assert contended.wait(8), "second transaction never observed lock contention"
        # Release A BEFORE joining/waiting for B's commit.
        release.set()
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
        fresh = AdvancedMemoryStore(str(tmp_path), enable_auto_cleanup=False)
        entries = fresh.get_entries(True)
        assert [e.content for e in entries].count("B") == 1
        expired = next(e for e in entries if e.id == "expired")
        assert expired.status == "expired" and expired.t_expired is not None
        assert fresh.load().abstracts == ["B"]
    finally:
        release.set()
        finish(processes)


STORE_CASES = [
    (AdvancedMemoryStore, "advanced_memory_state.json"),
    (InMemoryMemoryStore, "memory_state.json"),
    (TTLMemoryStore, "ttl_memory_state.json"),
    (InMemoryPageStore, "pages.json"),
    (TTLPageStore, "ttl_pages.json"),
]


def create(cls, path):
    if cls in (AdvancedMemoryStore, TTLMemoryStore, TTLPageStore):
        return cls(str(path), enable_auto_cleanup=False)
    return cls(str(path))


def append(store, content):
    store.add(
        Page(header=content, content=content)
        if isinstance(store, (InMemoryPageStore, TTLPageStore))
        else content
    )


def contents(store):
    state = store.load()
    return (
        state.abstracts if hasattr(state, "abstracts") else [p.content for p in state]
    )


@pytest.mark.parametrize("cls,filename", STORE_CASES)
@pytest.mark.parametrize("late", [False, True])
def test_all_stores_acknowledge_and_reload_failures(
    tmp_path, monkeypatch, cls, filename, late
):
    import jitmind.utils.atomic_io as io

    store = create(cls, tmp_path)
    append(store, "old")
    old_bytes = (tmp_path / filename).read_bytes()
    cause = OSError("/private/secret-do-not-expose")

    def fail(*args):
        raise cause

    monkeypatch.setattr(io, "_sync_directory", fail) if late else monkeypatch.setattr(
        io.os, "replace", fail
    )
    with pytest.raises(PersistenceError) as raised:
        append(store, "new")
    assert raised.value.outcome_uncertain is late
    assert raised.value.__cause__ is cause
    assert "secret" not in str(raised.value)
    expected = ["old", "new"] if late else ["old"]
    # Check the actual cache before a read has the opportunity to repair it.
    if hasattr(store, "_pages"):
        assert [p.content for p in store._pages] == expected
        assert store.get(len(expected) - 1).content == expected[-1]
    elif isinstance(store, InMemoryMemoryStore):
        assert store._state.abstracts == expected
    else:
        assert [e.content for e in store._state.entries] == expected
    assert contents(store) == expected
    assert contents(create(cls, tmp_path)) == expected
    assert not list(tmp_path.glob("*.tmp"))
    if not late:
        assert (tmp_path / filename).read_bytes() == old_bytes


@pytest.mark.parametrize("cls,filename", STORE_CASES)
@pytest.mark.parametrize(
    "payload",
    [
        b"not JSON",
        b"{}",
        b'{"entries": null, "pages": null, "abstracts": null}',
        b'{"entries": [{}], "pages": [{}], "abstracts": [null]}',
    ],
)
def test_corruption_refuses_constructor_and_mutations(tmp_path, cls, filename, payload):
    store = create(cls, tmp_path)
    path = tmp_path / filename
    path.write_bytes(payload)
    with pytest.raises(CorruptStoreError):
        create(cls, tmp_path)
    with pytest.raises(CorruptStoreError):
        append(store, "new")
    replacement = (
        [Page(header="new", content="new")]
        if "pages" in filename
        else MemoryState(abstracts=["new"])
    )
    with pytest.raises(CorruptStoreError):
        store.save(replacement)
    assert path.read_bytes() == payload


def test_single_cleanup_clock_and_history(tmp_path, monkeypatch):
    import jitmind.schemas.advanced_memory as module

    now = datetime.now(timezone.utc)
    store = AdvancedMemoryStore(str(tmp_path), enable_auto_cleanup=False, ttl_seconds=1)
    for i in range(2):
        store.add_entry(
            MemoryEntry(
                content=str(i), t_created=(now - timedelta(seconds=10)).isoformat()
            )
        )

    class Clock(datetime):
        calls = 0

        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            return now

    monkeypatch.setattr(module, "datetime", Clock)
    assert store.cleanup_expired() == 2
    assert Clock.calls == 1
    assert {e.t_expired for e in store.get_entries(True)} == {now.isoformat()}


def test_migration_refuses_corrupt_legacy(tmp_path):
    path = tmp_path / "memory_state.json"
    path.write_text("{}")
    original = path.read_bytes()
    with pytest.raises(CorruptStoreError):
        AdvancedMemoryStore(str(tmp_path))
    assert path.read_bytes() == original
    assert not (tmp_path / "advanced_memory_state.json").exists()


@pytest.mark.parametrize(
    "operation",
    [
        "save",
        "update",
        "delete",
        "supersede",
        "touch",
        "cleanup",
        "promote",
        "ranked_entries",
        "ranked_abstracts",
    ],
)
def test_advanced_mutators_restore_precommit_state(tmp_path, monkeypatch, operation):
    import jitmind.utils.atomic_io as io

    store = AdvancedMemoryStore(str(tmp_path), enable_auto_cleanup=False, ttl_seconds=1)
    old = datetime.now(timezone.utc) - timedelta(days=3)
    store.add_entry(
        MemoryEntry(id="old", content="old", t_created=old.isoformat(), strength=10)
    )
    original = store._state.model_dump()

    def fail(*args):
        raise OSError("secret")

    monkeypatch.setattr(io.os, "replace", fail)
    operations = {
        "save": lambda: store.save(MemoryState(abstracts=["new"])),
        "update": lambda: store.update_entry(
            "old", MemoryEntry(content="new", version_of="old")
        ),
        "delete": lambda: store.delete_entry("old"),
        "supersede": lambda: store.supersede_entry("old"),
        "touch": lambda: store.touch(["old"]),
        "cleanup": store.cleanup_expired,
        "promote": store.promote_demote,
        "ranked_entries": store.get_ranked_entries,
        "ranked_abstracts": store.get_ranked_abstracts,
    }
    with pytest.raises(PersistenceError):
        operations[operation]()
    assert store._state.model_dump() == original
    assert create(AdvancedMemoryStore, tmp_path)._state.model_dump() == original


@pytest.mark.parametrize("cls,filename", STORE_CASES)
def test_first_write_failure_does_not_leave_cached_mutation(
    tmp_path, monkeypatch, cls, filename
):
    import jitmind.utils.atomic_io as io

    store = create(cls, tmp_path)

    def fail(*args):
        raise OSError("secret")

    monkeypatch.setattr(io.os, "replace", fail)
    with pytest.raises(PersistenceError):
        append(store, "unpublished")
    assert contents(store) == []
    assert not (tmp_path / filename).exists()


@pytest.mark.parametrize("cls,filename", STORE_CASES)
def test_explicit_save_failure_restores_state(tmp_path, monkeypatch, cls, filename):
    import jitmind.utils.atomic_io as io

    store = create(cls, tmp_path)
    append(store, "old")
    state = (
        [Page(header="new", content="new")]
        if "pages" in filename
        else MemoryState(abstracts=["new"])
    )

    def fail(*args):
        raise OSError("secret")

    monkeypatch.setattr(io.os, "replace", fail)
    with pytest.raises(PersistenceError):
        store.save(state)
    assert contents(store) == ["old"]


@pytest.mark.parametrize(
    "cls,filename,payload",
    [
        (InMemoryMemoryStore, "memory_state.json", {"abstracts": ["old"]}),
        (TTLMemoryStore, "ttl_memory_state.json", {"abstracts": ["old"]}),
        (TTLMemoryStore, "ttl_memory_state.json", ["old"]),
        (TTLMemoryStore, "ttl_memory_state.json", [{"content": "old"}]),
        (InMemoryPageStore, "pages.json", [{"header": "h", "content": "old"}]),
        (
            InMemoryPageStore,
            "pages.json",
            {"pages": [{"header": "h", "content": "old"}]},
        ),
        (TTLPageStore, "ttl_pages.json", [{"header": "h", "content": "old"}]),
        (
            TTLPageStore,
            "ttl_pages.json",
            {"pages": [{"header": "h", "content": "old"}]},
        ),
        (AdvancedMemoryStore, "memory_state.json", {"abstracts": ["old"]}),
    ],
)
def test_old_formats_remain_readable_and_writable(tmp_path, cls, filename, payload):
    (tmp_path / filename).write_text(json.dumps(payload))
    store = create(cls, tmp_path)
    assert contents(store) == ["old"]
    append(store, "new")
    assert contents(create(cls, tmp_path)) == ["old", "new"]


@pytest.mark.parametrize("cls", [TTLMemoryStore, TTLPageStore])
@pytest.mark.parametrize("late", [False, True])
def test_ttl_cleanup_failure_restores_visible_state(tmp_path, monkeypatch, cls, late):
    import jitmind.utils.atomic_io as io

    store = create(cls, tmp_path)
    append(store, "old")
    store._ttl_seconds = -1

    def fail(*args):
        raise OSError("secret")

    monkeypatch.setattr(io, "_sync_directory", fail) if late else monkeypatch.setattr(
        io.os, "replace", fail
    )
    with pytest.raises(PersistenceError) as raised:
        store.cleanup_expired()
    assert raised.value.outcome_uncertain is late
    assert contents(store) == ([] if late else ["old"])
    assert contents(create(cls, tmp_path)) == ([] if late else ["old"])

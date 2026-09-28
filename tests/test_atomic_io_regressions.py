import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from jitmind.profile.profile_store import UserProfile, UserProfileStore
from jitmind.utils import atomic_io as io
from jitmind.utils.checkpoint import CheckpointManager


@pytest.mark.parametrize(
    "stage",
    [
        "mkdir",
        "mkstemp",
        "fdopen",
        "write",
        "flush",
        "fsync",
        "replace",
        "directory_open",
        "directory_fsync",
    ],
)
def test_atomic_failure_stages_clean_temp_and_descriptors(tmp_path, monkeypatch, stage):
    path = tmp_path / "private-state.json"
    path.write_text("old")
    cause = OSError("secret /private/path")
    descriptors = []
    original_mkstemp = io.tempfile.mkstemp
    original_fdopen = io.os.fdopen
    original_open = io.os.open
    original_fsync = io.os.fsync

    def fail(*args, **kwargs):
        raise cause

    def mkstemp(*args, **kwargs):
        fd, name = original_mkstemp(*args, **kwargs)
        descriptors.append(fd)
        return fd, name

    monkeypatch.setattr(io.tempfile, "mkstemp", mkstemp)
    if stage == "mkdir":
        monkeypatch.setattr(type(path), "mkdir", fail)
    elif stage == "mkstemp":
        monkeypatch.setattr(io.tempfile, "mkstemp", fail)
    elif stage == "fdopen":
        monkeypatch.setattr(io.os, "fdopen", fail)
    elif stage in ("write", "flush"):

        class Stream:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def write(self, text):
                return fail() if stage == "write" else self.stream.write(text)

            def flush(self):
                return fail() if stage == "flush" else self.stream.flush()

            def fileno(self):
                return self.stream.fileno()

        monkeypatch.setattr(
            io.os,
            "fdopen",
            lambda *args, **kwargs: Stream(original_fdopen(*args, **kwargs)),
        )
    elif stage == "fsync":
        monkeypatch.setattr(io.os, "fsync", fail)
    elif stage == "replace":
        monkeypatch.setattr(io.os, "replace", fail)
    elif stage == "directory_open":

        def open_file(name, *args, **kwargs):
            if name == path.parent:
                fail()
            return original_open(name, *args, **kwargs)

        monkeypatch.setattr(io.os, "open", open_file)
    else:
        calls = []

        def fsync(fd):
            calls.append(fd)
            if len(calls) == 2:
                descriptors.append(fd)
                fail()
            return original_fsync(fd)

        monkeypatch.setattr(io.os, "fsync", fsync)
    with pytest.raises(io.PersistenceError) as raised:
        io.atomic_write_text(path, "new")
    late = stage.startswith("directory_")
    assert raised.value.outcome_uncertain is late
    assert raised.value.__cause__ is cause
    assert "secret" not in str(raised.value) and str(path) not in str(raised.value)
    assert path.read_text() == ("new" if late else "old")
    assert list(tmp_path.iterdir()) == [path]
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_serialization_failure_is_precommit(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("old")
    with pytest.raises(io.PersistenceError) as raised:
        io.atomic_write_json(path, {"bad": object()})
    assert not raised.value.outcome_uncertain
    assert isinstance(raised.value.__cause__, TypeError)
    assert path.read_text() == "old"


@pytest.mark.parametrize("kind", ["profile", "checkpoint"])
@pytest.mark.parametrize("late", [False, True])
def test_other_legacy_stores_errors_and_corruption(tmp_path, monkeypatch, kind, late):
    store = (
        UserProfileStore(str(tmp_path))
        if kind == "profile"
        else CheckpointManager(str(tmp_path))
    )

    def save(value):
        if kind == "profile":
            store.save(UserProfile("user", static={"value": value}))
        else:
            store.save_checkpoint("user", {"value": value})

    def load():
        return (
            store.load("user").static
            if kind == "profile"
            else store.load_checkpoint("user")
        )

    save("old")
    path = store._path("user")
    cause = OSError("private secret")

    def fail(*args):
        raise cause

    with monkeypatch.context() as patch:
        patch.setattr(io, "_sync_directory", fail) if late else patch.setattr(
            io.os, "replace", fail
        )
        with pytest.raises(io.PersistenceError) as raised:
            save("new")
        assert raised.value.outcome_uncertain is late
        assert raised.value.__cause__ is cause
    assert load() == {"value": "new" if late else "old"}
    path.write_text("{broken")
    for operation in (load, lambda: save("overwrite")):
        with pytest.raises(io.CorruptStoreError):
            operation()
    assert path.read_text() == "{broken"


def test_checkpoint_public_delete_between_exists_and_open(tmp_path, monkeypatch):
    import jitmind.utils.checkpoint as module

    store = CheckpointManager(str(tmp_path))
    assert store.load_checkpoint("user") is None
    assert store.delete_checkpoint("user") is False
    assert store.save_checkpoint("user", {"old": True}) is None
    assert store.load_checkpoint("user") == {"old": True}
    path = store._path("user")
    opening, resume, waiting = threading.Event(), threading.Event(), threading.Event()
    reads, deletes = [], []
    original_open, original_lock = Path.open, module.file_lock

    def open_file(name, *args, **kwargs):
        if name == path and threading.current_thread() is reader:
            opening.set()
            assert resume.wait(5)
        return original_open(name, *args, **kwargs)

    @contextmanager
    def observed_lock(lock_path):
        assert lock_path == tmp_path / ".checkpoints.lock"
        if threading.current_thread() is deleter:
            # A real zero-timeout attempt proves contention without a sleep oracle.
            with pytest.raises(TimeoutError):
                with original_lock(lock_path, timeout_s=0):
                    pytest.fail("Deletion acquired the paused reader's lock")
            waiting.set()
        with original_lock(lock_path, timeout_s=5):
            yield

    def load():
        try:
            reads.append(store.load_checkpoint("user"))
        except BaseException as exc:
            reads.append(exc)

    def delete():
        try:
            deletes.append(store.delete_checkpoint("user"))
        except BaseException as exc:
            deletes.append(exc)

    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(module, "file_lock", observed_lock)
    reader = threading.Thread(target=load)
    deleter = threading.Thread(target=delete)
    try:
        reader.start()
        assert opening.wait(5)
        deleter.start()
        assert waiting.wait(5)
        assert deleter.is_alive()
        assert deletes == []
        assert path.exists()
    finally:
        # The deleter needs this lock: release the reader before joining either.
        resume.set()
        reader.join(5)
        if deleter.ident is not None:
            deleter.join(5)
    assert not reader.is_alive() and not deleter.is_alive()
    assert reads == [{"old": True}]
    assert deletes == [True]
    assert store.load_checkpoint("user") is None
    assert store.delete_checkpoint("user") is False


@pytest.fixture
def genuine_v1_checkpoint(tmp_path):
    # Deliberately use the historical filename/envelope, never the v2 writer.
    path = tmp_path / "user.json"
    path.write_text(json.dumps({
        "thread_id": "user",
        "timestamp": "2026-09-28T00:00:00Z",
        "state": {"legacy": True},
    }))
    return path


def test_checkpoint_genuine_v1_compatibility(tmp_path, genuine_v1_checkpoint):
    store = CheckpointManager(str(tmp_path))
    original = genuine_v1_checkpoint.read_bytes()
    assert store.load_checkpoint("user") == {"legacy": True}
    assert store.save_checkpoint("user", {"v2": True}) is None
    assert store._path("user") != genuine_v1_checkpoint
    assert genuine_v1_checkpoint.read_bytes() == original
    assert store.load_checkpoint("user") == {"v2": True}
    assert store.delete_checkpoint("user") is True
    assert not genuine_v1_checkpoint.exists()
    assert not store._path("user").exists()


@pytest.mark.parametrize("layout", ["v1", "v2"])
@pytest.mark.parametrize("invalid", [
    b"{broken", b'{"state": []}', b"\xff",
    b'{"thread_id":"","timestamp":"x","state":{}}',
    b'{"thread_id":"user","timestamp":"x","state":{},"namespace":[]}',
])
def test_checkpoint_public_corruption_preserves_bytes(
    tmp_path, genuine_v1_checkpoint, layout, invalid
):
    from jitmind.utils.checkpoint import CheckpointCorrupt, CheckpointIOError

    store = CheckpointManager(str(tmp_path))
    path = genuine_v1_checkpoint
    if layout == "v2":
        store.save_checkpoint("user", {"old": True})
        path = store._path("user")
    path.write_bytes(invalid)
    for operation in (
        lambda: store.load_checkpoint("user"),
        lambda: store.save_checkpoint("user", {"overwrite": True}),
        lambda: store.delete_checkpoint("user"),
        store.list_checkpoints,
    ):
        with pytest.raises(io.CorruptStoreError) as raised:
            operation()
        assert isinstance(raised.value, CheckpointCorrupt)
        assert isinstance(raised.value, io.StorageError)
        assert not isinstance(raised.value, CheckpointIOError)
        assert path.read_bytes() == invalid
    if layout == "v1":
        assert not store._path("user").exists()


@pytest.mark.parametrize("operation", ["load", "save", "delete", "list"])
def test_checkpoint_io_keeps_generic_error_catch_and_cause(
    tmp_path, monkeypatch, operation
):
    from jitmind.utils.checkpoint import CheckpointIOError

    store = CheckpointManager(str(tmp_path))
    store.save_checkpoint("user", {"old": True})
    path = store._path("user")
    original = path.read_bytes()
    cause = PermissionError("private read failure")
    original_open = Path.open

    def denied(candidate, *args, **kwargs):
        if candidate == path:
            raise cause
        return original_open(candidate, *args, **kwargs)

    actions = {
        "load": lambda: store.load_checkpoint("user"),
        "save": lambda: store.save_checkpoint("user", {"new": True}),
        "delete": lambda: store.delete_checkpoint("user"),
        "list": store.list_checkpoints,
    }
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", denied)
        with pytest.raises(io.StorageError) as raised:
            actions[operation]()
    assert isinstance(raised.value, CheckpointIOError)
    assert raised.value.__cause__ is cause
    assert "private" not in str(raised.value)
    assert path.read_bytes() == original


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_noncooperating_disappearance_before_open(
    tmp_path, monkeypatch, legacy
):
    store = CheckpointManager(str(tmp_path))
    store.save_checkpoint("user", {"old": True})
    path = store._path("user")
    old = tmp_path / "user.json"
    if legacy:
        old.write_text(json.dumps({
            "thread_id": "user", "timestamp": "2026-09-28T00:00:00Z",
            "state": {"legacy": True},
        }))
    original_open = Path.open
    opened = []

    def disappear(candidate, *args, **kwargs):
        opened.append(candidate)
        if candidate == path or (legacy and candidate == old):
            candidate.unlink()
        # Exercise the actual open's FileNotFoundError, not a synthetic return.
        return original_open(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", disappear)
    assert store.load_checkpoint("user") is None
    assert opened == [path, old]  # Default namespace permits legacy fallback.
    assert not path.exists() and not old.exists()


@pytest.mark.parametrize("stage", ["unlink", "directory_sync", "corrupt"])
def test_checkpoint_delete_failure_preserves_acknowledgement(
    tmp_path, monkeypatch, stage
):
    store = CheckpointManager(str(tmp_path))
    store.save_checkpoint("user", {"old": True})
    path = store._path("user")
    cause = OSError("private deletion failure")

    def fail(*args):
        raise cause

    if stage == "corrupt":
        path.write_text("{broken")
    original = path.read_bytes()
    if stage == "unlink":
        monkeypatch.setattr(type(path), "unlink", fail)
    elif stage == "directory_sync":
        import jitmind.utils.checkpoint as module

        monkeypatch.setattr(module, "_sync_directory", fail)
    if stage == "corrupt":
        with pytest.raises(io.CorruptStoreError):
            store.delete_checkpoint("user")
    else:
        with pytest.raises(io.PersistenceError) as raised:
            store.delete_checkpoint("user")
        assert raised.value.__cause__ is cause
        assert raised.value.outcome_uncertain is (stage == "directory_sync")
    if stage == "directory_sync":
        assert not path.exists()
        assert store.load_checkpoint("user") is None
    else:
        assert path.read_bytes() == original
    assert (tmp_path / ".checkpoints.lock").exists()

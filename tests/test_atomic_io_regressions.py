import os
import threading

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
    path = tmp_path / "user.json"
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
    import builtins
    import jitmind.utils.checkpoint as module

    store = CheckpointManager(str(tmp_path))
    assert store.load_checkpoint("user") is None
    assert store.delete_checkpoint("user") is False
    assert store.save_checkpoint("user", {"old": True}) is None
    assert store.load_checkpoint("user") == {"old": True}
    path = tmp_path / "user.json"
    opening, resume = threading.Event(), threading.Event()
    results = []

    def open_file(name, *args, **kwargs):
        if name == path and threading.current_thread() is reader:
            opening.set()
            assert resume.wait(5)
        return builtins.open(name, *args, **kwargs)

    def load():
        try:
            results.append(store.load_checkpoint("user"))
        except BaseException as exc:
            results.append(exc)

    monkeypatch.setattr(module, "open", open_file, raising=False)
    reader = threading.Thread(target=load)
    try:
        reader.start()
        assert opening.wait(5)
        assert store.delete_checkpoint("user") is True
    finally:
        resume.set()
        reader.join(5)
    assert not reader.is_alive()
    assert results == [None]
    assert store.load_checkpoint("user") is None
    assert store.delete_checkpoint("user") is False
    # Absence and corrupt data retain distinct public results after the race.
    for invalid in ("{broken", '{"state": []}'):
        path.write_text(invalid)
        with pytest.raises(io.CorruptStoreError):
            store.load_checkpoint("user")
        assert path.read_text() == invalid


@pytest.mark.parametrize("stage", ["unlink", "directory_sync", "corrupt"])
def test_checkpoint_delete_failure_preserves_acknowledgement(
    tmp_path, monkeypatch, stage
):
    store = CheckpointManager(str(tmp_path))
    store.save_checkpoint("user", {"old": True})
    path = tmp_path / "user.json"
    cause = OSError("private deletion failure")

    def fail(*args):
        raise cause

    if stage == "corrupt":
        path.write_text("{broken")
    original = path.read_bytes()
    if stage == "unlink":
        monkeypatch.setattr(type(path), "unlink", fail)
    elif stage == "directory_sync":
        monkeypatch.setattr(io, "_sync_directory", fail)
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
    assert (tmp_path / "user.json.lock").exists()

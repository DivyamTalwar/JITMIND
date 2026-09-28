import json
import multiprocessing as mp
import threading
from contextlib import contextmanager

import pytest

from jitmind.utils.checkpoint import (
    CheckpointCorrupt,
    CheckpointIdentityError,
    CheckpointIOError,
    CheckpointManager,
    CheckpointSchemaError,
)


def legacy(directory, thread="a/b", state=None):
    path = directory / "ab.json"
    path.write_text(
        json.dumps(
            {
                "thread_id": thread,
                "timestamp": "2026-09-28T00:00:00Z",
                "state": state or {"step": 2},
            }
        )
    )
    return path


def test_legacy_identity_collision_and_migration(tmp_path):
    path = legacy(tmp_path)
    original = path.read_bytes()
    manager = CheckpointManager(str(tmp_path))
    assert manager.load_checkpoint("a/b") == {"step": 2}
    assert manager.load_checkpoint_record("a/b")["validation"] == "legacy_unvalidated"
    assert manager.list_checkpoints() == ["a/b"]
    with pytest.raises(CheckpointIdentityError):
        manager.load_checkpoint("ab")
    with pytest.raises(CheckpointIdentityError):
        manager.delete_checkpoint("ab")
    assert path.read_bytes() == original
    manager.save_checkpoint("ab", {"step": 7})
    assert path.read_bytes() == original
    assert manager.load_checkpoint("a/b") == {"step": 2}
    assert manager.load_checkpoint("ab") == {"step": 7}
    assert manager.delete_checkpoint("ab")
    assert path.read_bytes() == original
    manager.save_checkpoint("a/b", {"step": 3})
    assert path.read_bytes() == original
    assert manager.load_checkpoint("a/b") == {"step": 3}
    assert manager.delete_checkpoint("a/b")
    assert not path.exists()
    assert manager.load_checkpoint("a/b") is None


def test_full_identity_filenames_and_namespace(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    ids = ["a/b", "ab", "a.b", "a?b", "../ab", "é", "e\u0301", "x" * 1024, "///"]
    for index, thread in enumerate(ids):
        manager.save_checkpoint(thread, {"index": index})
    for index, thread in enumerate(ids):
        assert manager.load_checkpoint(thread) == {"index": index}
    assert set(manager.list_checkpoints()) == set(ids)
    manager.save_checkpoint("a/b", {"other": 1}, namespace="a/b")
    manager.save_checkpoint("a/b", {"other": 2}, namespace="ab")
    assert manager.load_checkpoint("a/b", namespace="a/b") == {"other": 1}
    assert manager.load_checkpoint("a/b", namespace="ab") == {"other": 2}
    assert manager.load_checkpoint("a/b", namespace="missing") is None
    for thread in ids:
        assert manager.delete_checkpoint(thread)
    assert manager.list_checkpoints() == []
    assert manager.list_checkpoints(namespace="ab") == ["a/b"]


def test_requested_identity_verified_even_in_v2_path(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    manager.save_checkpoint("target", {"x": 1})
    path = manager._path("target")
    data = json.loads(path.read_text())
    data["thread_id"] = "other"
    path.write_text(json.dumps(data))
    original = path.read_bytes()
    for operation in [
        lambda: manager.load_checkpoint("target"),
        lambda: manager.delete_checkpoint("target"),
        lambda: manager.save_checkpoint("target", {"new": True}),
    ]:
        with pytest.raises(CheckpointIdentityError):
            operation()
        assert path.read_bytes() == original
    data["thread_id"] = "target"
    data["namespace"] = "other"
    path.write_text(json.dumps(data))
    with pytest.raises(CheckpointIdentityError):
        manager.load_checkpoint("target")


@pytest.mark.parametrize(
    "content,error",
    [
        ("{invalid", CheckpointCorrupt),
        ('{"thread_id":"ab","thread_id":"other"}', CheckpointCorrupt),
        ('{"x":NaN}', CheckpointCorrupt),
        ("[]", CheckpointSchemaError),
        ('{"thread_id":"ab","timestamp":"x","state":null}', CheckpointSchemaError),
        (
            '{"thread_id":"ab","timestamp":"x","state":{},"schema_version":true}',
            CheckpointSchemaError,
        ),
        (
            '{"thread_id":"ab","timestamp":"x","state":{},"schema_version":99}',
            CheckpointSchemaError,
        ),
    ],
)
def test_corruption_is_not_absence_and_originals_survive(tmp_path, content, error):
    manager = CheckpointManager(str(tmp_path))
    path = tmp_path / "ab.json"
    path.write_text(content)
    for operation in [
        lambda: manager.load_checkpoint("ab"),
        lambda: manager.delete_checkpoint("ab"),
        lambda: manager.list_checkpoints(),
        lambda: manager.save_checkpoint("ab", {"new": 1}),
    ]:
        with pytest.raises(error):
            operation()
        assert path.read_text() == content
    assert manager.load_checkpoint("missing") is None
    assert manager.delete_checkpoint("missing") is False


def test_modern_corruption_prevents_overwrite(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    manager.save_checkpoint("id", {})
    path = manager._path("id")
    path.write_bytes(b"\xff")
    with pytest.raises(CheckpointCorrupt):
        manager.save_checkpoint("id", {})
    assert path.read_bytes() == b"\xff"


def test_symlink_is_not_followed(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    target = tmp_path / "untouched.txt"
    target.write_text("sensitive")
    manager._path("id").symlink_to(target)
    for operation in [
        lambda: manager.load_checkpoint("id"),
        lambda: manager.save_checkpoint("id", {}),
        lambda: manager.delete_checkpoint("id"),
    ]:
        with pytest.raises(CheckpointIOError):
            operation()
    assert target.read_text() == "sensitive"


def test_threaded_collisions_and_deletes(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    barrier = threading.Barrier(4)
    errors = []

    def run(identity):
        try:
            barrier.wait(timeout=5)
            for i in range(15):
                manager.save_checkpoint(identity, {"id": identity, "i": i})
                assert manager.load_checkpoint(identity)["id"] == identity
                assert manager.delete_checkpoint(identity)
            manager.save_checkpoint(identity, {"id": identity})
        except Exception as error:  # noqa: BLE001 - transport child failures to the asserting parent
            errors.append(error)

    identities = ["ab", "a/b", "a.b", "a?b"]
    threads = [
        threading.Thread(target=run, args=(identity,)) for identity in identities
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
        assert not thread.is_alive()
    assert not errors
    assert set(manager.list_checkpoints()) == set(identities)


def held_save_process(directory, held, release, output):
    import jitmind.utils.checkpoint as module

    original = module.atomic_write_json

    def pause(path, payload, **kwargs):
        held.set()  # save owns the actual file lock
        if not release.wait(15):
            raise RuntimeError("test coordination timed out")
        original(path, payload, **kwargs)

    module.atomic_write_json = pause
    try:
        module.CheckpointManager(directory).save_checkpoint("id", {"saved": True})
        output.put("saved")
    except Exception as error:  # noqa: BLE001 - transport child failures to the asserting parent
        output.put(type(error).__name__)


def bounded_delete_process(directory, output):
    import jitmind.utils.checkpoint as module

    original = module.file_lock

    @contextmanager
    def bounded(path):
        with original(path, timeout_s=0.02, poll_s=0.005):
            yield

    module.file_lock = bounded
    try:
        module.CheckpointManager(directory).delete_checkpoint("id")
        output.put("deleted")
    except CheckpointIOError:
        output.put("locked")


def test_process_delete_cannot_bypass_save_lock(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    manager.save_checkpoint("id", {"old": True})
    ctx = mp.get_context("spawn")
    held, release, output = ctx.Event(), ctx.Event(), ctx.Queue()
    saver = ctx.Process(
        target=held_save_process, args=(str(tmp_path), held, release, output)
    )
    deleter = ctx.Process(target=bounded_delete_process, args=(str(tmp_path), output))
    saver.start()
    try:
        assert held.wait(15)
        deleter.start()
        assert output.get(timeout=15) == "locked"
        assert json.loads(manager._path("id").read_text())["state"] == {"old": True}
        release.set()
        assert output.get(timeout=15) == "saved"
        for process in (saver, deleter):
            process.join(15)
            assert process.exitcode == 0
    finally:
        release.set()
        for process in (saver, deleter):
            if process.pid is not None and process.is_alive():
                process.kill()
                process.join(5)
    assert manager.load_checkpoint("id") == {"saved": True}
    assert manager.delete_checkpoint("id")


def test_research_agent_state_contract(tmp_path):
    # Exercise actual integration methods without constructing providers or stores.
    from jitmind.agents.research_agent import ResearchAgent, Result

    agent = ResearchAgent.__new__(ResearchAgent)
    agent.checkpoint_manager = CheckpointManager(str(tmp_path))
    agent._save_checkpoint_state(
        "a/b", [{"step": 0}], Result(content="fixture"), "next", 1
    )
    restored = agent._load_checkpoint_state("a/b")
    assert restored["step"] == 1
    assert restored["temp"].content == "fixture"
    assert restored["next_request"] == "next"


@pytest.mark.parametrize("state", [
    {"nested": {1: "numeric", "1": "string"}},
    {"nested": [{None: "null"}]},
    {"nested": {True: "boolean"}},
    {"nested": {1.5: "float"}},
    {"nested": {1: "coercion without collision"}},
    {"nested": float("nan")},
    {"nested": float("inf")},
])
def test_writer_rejects_coercing_json_without_replacing_bytes(tmp_path, state):
    manager = CheckpointManager(str(tmp_path))
    manager.save_checkpoint("id", {"old": True})
    before = manager._path("id").read_bytes()
    with pytest.raises(CheckpointSchemaError):
        manager.save_checkpoint("id", state)
    assert manager._path("id").read_bytes() == before
    assert manager.load_checkpoint("id") == {"old": True}


def test_writer_checks_duplicate_serialized_names_and_valid_nested_state(tmp_path):
    class DuplicateNames(dict):
        def items(self):
            return [("same", 1), ("same", 2)]

    manager = CheckpointManager(str(tmp_path))
    good = {"nested": [{"1": "numeric name", "null": None, "unicode": "é"}]}
    manager.save_checkpoint("id", good)
    before = manager._path("id").read_bytes()
    with pytest.raises(CheckpointSchemaError):
        manager.save_checkpoint("id", {"nested": DuplicateNames(x=1)})
    assert manager._path("id").read_bytes() == before
    assert manager.load_checkpoint("id") == good


@pytest.mark.parametrize("operation", ["save", "load", "list", "delete"])
def test_actual_j01_lock_failure_has_checkpoint_io_boundary(tmp_path, operation):
    manager = CheckpointManager(str(tmp_path))
    (tmp_path / ".checkpoints.lock").mkdir()
    actions = {
        "save": lambda: manager.save_checkpoint("id", {}),
        "load": lambda: manager.load_checkpoint("id"),
        "list": manager.list_checkpoints,
        "delete": lambda: manager.delete_checkpoint("id"),
    }
    with pytest.raises(CheckpointIOError):
        actions[operation]()


@pytest.mark.parametrize("legacy_mode", ["none", "matching", "collision", "corrupt"])
def test_disappearing_v2_uses_only_valid_exact_legacy_fallback(tmp_path, monkeypatch, legacy_mode):
    from pathlib import Path

    manager = CheckpointManager(str(tmp_path))
    manager.save_checkpoint("a/b", {"new": True})
    selected = manager._path("a/b")
    if legacy_mode != "none":
        old = legacy(tmp_path, thread="other" if legacy_mode == "collision" else "a/b")
        if legacy_mode == "corrupt":
            old.write_text("{")
    original = Path.open

    def disappear(path, *args, **kwargs):
        if path == selected:
            path.unlink(missing_ok=True)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", disappear)
    if legacy_mode == "collision":
        with pytest.raises(CheckpointIdentityError):
            manager.load_checkpoint("a/b")
    elif legacy_mode == "corrupt":
        with pytest.raises(CheckpointCorrupt):
            manager.load_checkpoint("a/b")
    else:
        expected = {"step": 2} if legacy_mode == "matching" else None
        assert manager.load_checkpoint("a/b") == expected


def test_disappearing_legacy_and_namespaced_v2_are_absence(tmp_path, monkeypatch):
    from pathlib import Path

    manager = CheckpointManager(str(tmp_path))
    old = legacy(tmp_path)
    manager.save_checkpoint("a/b", {}, namespace="separate")
    selected = manager._path("a/b", namespace="separate")
    original = Path.open

    def disappear(path, *args, **kwargs):
        if path in (old, selected):
            path.unlink(missing_ok=True)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", disappear)
    assert manager.load_checkpoint("a/b", namespace="separate") is None
    assert old.exists()  # no legacy lookup outside the default namespace
    assert manager.load_checkpoint("a/b") is None


def test_permission_failure_never_falls_back(tmp_path, monkeypatch):
    from pathlib import Path

    manager = CheckpointManager(str(tmp_path))
    manager.save_checkpoint("a/b", {})
    legacy(tmp_path)
    selected = manager._path("a/b")
    original = Path.open

    def denied(path, *args, **kwargs):
        if path == selected:
            raise PermissionError("private fixture detail")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(CheckpointIOError, match="^Checkpoint read failed$"):
        manager.load_checkpoint("a/b")


@pytest.mark.parametrize("phase", ["replace", "save_sync", "unlink", "delete_sync", "partial_delete"])
def test_j01_mutation_failure_and_uncertainty_survive_boundary(tmp_path, monkeypatch, phase):
    from pathlib import Path
    import jitmind.utils.atomic_io as atomic
    import jitmind.utils.checkpoint as checkpoint

    manager = CheckpointManager(str(tmp_path))
    manager.save_checkpoint("a/b", {"old": True})
    path = manager._path("a/b")
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("fixture")

    if phase == "replace":
        monkeypatch.setattr(atomic.os, "replace", fail)
    elif phase == "save_sync":
        monkeypatch.setattr(atomic, "_sync_directory", fail)
    elif phase == "delete_sync":
        monkeypatch.setattr(checkpoint, "_sync_directory", fail)
    else:
        original = Path.unlink
        if phase == "partial_delete":
            old = legacy(tmp_path)

        def unlink(candidate, *args, **kwargs):
            if candidate == (old if phase == "partial_delete" else path):
                fail()
            return original(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(atomic.PersistenceError) as error:
        if phase in ("replace", "save_sync"):
            manager.save_checkpoint("a/b", {"new": True})
        else:
            manager.delete_checkpoint("a/b")
    assert error.value.outcome_uncertain == (phase in ("save_sync", "delete_sync", "partial_delete"))
    if phase in ("replace", "unlink"):
        assert path.read_bytes() == before
    elif phase == "save_sync":
        assert manager.load_checkpoint("a/b") == {"new": True}
    else:
        assert not path.exists()
        if phase == "partial_delete":
            assert old.exists()

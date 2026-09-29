"""Reproduced review defects; controlled tool evidence, never acceptance credit."""

import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from scripts import qualify_jitmind as q
import test_qualification as helpers
from test_qualification import refresh, verdict


@pytest.fixture
def bundle(tmp_path):
    return helpers.bundle.__wrapped__(tmp_path)


def rejected(bundle):
    try:
        assert not verdict(bundle)["checks_only_complete"]
    except q.EvidenceError:
        pass


@pytest.mark.parametrize(
    "corruption", ["digest", "bytes", "missing", "duplicate", "alias"]
)
def test_snapshot_roster_is_bound(bundle, corruption):
    identity = bundle[2]["candidate"]
    name = "tests/test_product.py"
    roster = [[name, q.digest(bundle[0] / "sources" / name)]]
    if corruption == "bytes":
        roster[0][1] = "e" * 64
    elif corruption == "missing":
        roster.append(["absent.py", "e" * 64])
    elif corruption == "duplicate":
        roster *= 2
    elif corruption == "alias":
        roster[0][0] = "tests/./test_product.py"
    identity["snapshot_files"] = roster
    identity["snapshot_sha256"] = hashlib.sha256(
        json.dumps(roster, separators=(",", ":")).encode()
    ).hexdigest()
    if corruption == "digest":
        identity["snapshot_sha256"] = "e" * 64
    bundle[3]["candidate"] = copy.deepcopy(identity)
    refresh(bundle)
    rejected(bundle)


@pytest.mark.parametrize("mode", ["filter", "skip", "importorskip", "error"])
def test_real_collection_omissions_are_recorded(tmp_path, mode):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_good.py").write_text(
        "def test_expected(): pass\ndef test_other(): pass\n"
    )
    if mode == "filter":
        (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = -k test_expected\n")
    else:
        text = {
            "skip": 'import pytest\npytest.skip("collection", allow_module_level=True)',
            "importorskip": 'import pytest\npytest.importorskip("jitmind_nonexistent_fixture_module")',
            "error": 'raise RuntimeError("collection failure")',
        }[mode]
        (tmp_path / "tests/test_bad.py").write_text(text)
    output = tmp_path / "pytest-results.json"
    q.write_json(
        tmp_path / "context.json",
        {"repo": str(tmp_path), "output": str(output), "candidate": {}, "kinds": {}},
    )
    env = q.clean_env(
        tmp_path, Path(sys.executable), Path(__file__).resolve().parents[1]
    )
    env["JITMIND_QUALIFICATION_CONTEXT"] = str(tmp_path / "context.json")
    q.run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "scripts.qualify_jitmind",
            "tests",
        ],
        cwd=tmp_path,
        env=env,
        log=tmp_path / "pytest.log",
        deadline=time.monotonic() + 10,
    )
    data = q.strict_json(output)
    expected = q.strict_json(tmp_path / "expected-collection.json")
    assert len(expected["nodes"]) == 2
    if mode == "filter":
        assert expected["deselected"] == ["tests/test_good.py::test_other"]
        assert expected["filtered"] is True
    else:
        assert expected["collection_reports"][-1]["outcome"] == (
            "failed" if mode == "error" else "skipped"
        )
    assert data["expected_collection_sha256"] == q.digest(
        tmp_path / "expected-collection.json"
    )


def test_oversized_copy_rejected_before_destination(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "MAX_ARTIFACT", 4096)
    src, dst = tmp_path / "large.whl", tmp_path / "copy.whl"
    src.write_bytes(b"x" * 5000)
    with pytest.raises(q.EvidenceError, match="artifact_too_large"):
        q.bounded_copy(src, dst)
    assert not dst.exists()


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    try:
        if not stat.exists():
            return True  # No procfs on non-Linux hosts; kill(pid, 0) succeeded.
        record = stat.read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False  # The inspected process exited during the procfs read.
    _, closing, fields = record.rpartition(")")
    if not closing or not fields.split():
        raise ValueError("Malformed process status fixture")
    return fields.split()[0] != "Z"


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signal_wrapper_cleans_owned_descendants(tmp_path, sig):
    repo = Path(__file__).resolve().parents[1]
    grand = "import time; time.sleep(60)"
    code = f'import os,pathlib,subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c",{grand!r}]); pathlib.Path("pids").write_text(str(os.getpid())+" "+str(p.pid)); time.sleep(60)'
    wrapper = f'from scripts import qualify_jitmind as q; from pathlib import Path; import sys,time,json; p=Path.cwd(); r=q.run_command([sys.executable,"-c",{code!r}],cwd=p,env=q.clean_env(p,Path(sys.executable)),log=p/"child.log",deadline=time.monotonic()+10); print(json.dumps(r))'
    pids = []
    with (tmp_path / "wrapper.log").open("wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-c", wrapper],
            cwd=tmp_path,
            env=q.clean_env(tmp_path, Path(sys.executable), repo),
            stdout=log,
            stderr=log,
        )
        try:
            end = time.monotonic() + 5
            while not (tmp_path / "pids").exists() and time.monotonic() < end:
                time.sleep(0.01)
            pids = [int(x) for x in (tmp_path / "pids").read_text().split()]
            proc.send_signal(sig)
            assert proc.wait(timeout=3) == 0
            assert (
                json.loads((tmp_path / "wrapper.log").read_text())["status"]
                == "cancelled"
            )
            end = time.monotonic() + 2
            while any(alive(pid) for pid in pids) and time.monotonic() < end:
                time.sleep(0.01)
            assert not any(alive(pid) for pid in pids)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=3)
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_deleting_unmapped_failure_does_not_shrink_expected_collection(bundle):
    root, _, _, tests = bundle
    extra = "tests/test_product.py::test_unmapped_failure"
    expected = q.strict_json(root / "expected-collection.json")
    expected["nodes"].append(extra)
    expected["selected"].append(extra)
    (root / "expected-collection.json").write_text(json.dumps(expected))
    tests["collected"].append(extra)
    tests["collection_count"] += 1
    tests["records"].append(
        dict(
            node_id=extra,
            kind="unmapped",
            source_sha256=tests["records"][0]["source_sha256"],
            status="failed",
            phases=dict(setup="passed", call="failed", teardown="passed"),
        )
    )
    tests["exit_code"] = 1
    refresh(bundle)
    assert not verdict(bundle)["checks_only_complete"]
    tests["records"].pop()
    tests["collected"].pop()
    tests["collection_count"] -= 1
    tests["exit_code"] = 0
    refresh(bundle)
    assert not verdict(bundle)["checks_only_complete"]


def test_missing_unmapped_record_retains_unknown_denominator(bundle):
    root, _, _, tests = bundle
    extra = "tests/test_product.py::test_missing_record"
    expected = q.strict_json(root / "expected-collection.json")
    expected["nodes"].append(extra)
    expected["selected"].append(extra)
    (root / "expected-collection.json").write_text(json.dumps(expected))
    tests["collected"].append(extra)
    tests["collection_count"] += 1
    refresh(bundle)
    result = verdict(bundle)
    assert not result["checks_only_complete"]
    assert result["test_counts"]["unknown"] == 1


@pytest.mark.parametrize("field", ["nodes", "selected", "deselected"])
def test_duplicate_collection_nodes_refused(bundle, field):
    path = bundle[0] / "expected-collection.json"
    expected = q.strict_json(path)
    expected[field] = expected["nodes"] * 2
    path.write_text(json.dumps(expected))
    refresh(bundle)
    with pytest.raises(q.EvidenceError, match="duplicate_or_invalid_collection"):
        verdict(bundle)


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_testscollected", 2),
        ("session_testsfailed", 1),
        ("session_testsfailed", False),
        ("phase_counts", {"passed": 4, "failed": 0, "skipped": 0}),
    ],
)
def test_inconsistent_plugin_counters_refused(bundle, field, value):
    bundle[3][field] = value
    (bundle[0] / "pytest-results.json").write_text(json.dumps(bundle[3]))
    refresh(bundle, tests=False)
    with pytest.raises(q.EvidenceError, match="inconsistent_plugin_counters"):
        verdict(bundle)


@pytest.mark.parametrize("field", ["filtered", "deselected", "collection_reports"])
def test_collection_incompleteness_blocks_evaluator(bundle, field):
    path = bundle[0] / "expected-collection.json"
    expected = q.strict_json(path)
    if field == "filtered":
        expected[field] = True
    elif field == "deselected":
        expected["nodes"].append("tests/test_product.py::test_omitted")
        expected[field] = ["tests/test_product.py::test_omitted"]
    else:
        expected[field] = [{"node_id": "tests/test_skip.py", "outcome": "skipped"}]
    path.write_text(json.dumps(expected))
    refresh(bundle)
    assert not verdict(bundle)["checks_only_complete"]


def test_runtime_record_source_must_match_snapshot(bundle):
    bundle[3]["records"][0]["source_sha256"] = "e" * 64
    refresh(bundle)
    with pytest.raises(q.EvidenceError, match="record_source_mismatch"):
        verdict(bundle)


def test_expected_collection_identity_and_digest_are_independent(bundle):
    bundle[3]["expected_collection_sha256"] = "e" * 64
    (bundle[0] / "pytest-results.json").write_text(json.dumps(bundle[3]))
    refresh(bundle, tests=False)
    with pytest.raises(q.EvidenceError, match="collection_digest_mismatch"):
        verdict(bundle)


@pytest.mark.parametrize("change", ["grow", "replace", "same_size"])
def test_copy_rejects_source_change_during_stream(tmp_path, monkeypatch, change):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.write_bytes(b"a" * 32)
    monkeypatch.setattr(q, "MAX_ARTIFACT", 64)
    real_fdopen = os.fdopen
    requests = []

    class ChangingReader:
        def __init__(self, fd, mode, **kwargs):
            self.stream = real_fdopen(fd, mode, **kwargs)
            self.changed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            requests.append(size)
            if not self.changed:
                self.changed = True
                if change == "grow":
                    with source.open("ab") as out:
                        out.write(b"b" * 64)
                elif change == "replace":
                    other = tmp_path / "replacement"
                    other.write_bytes(b"a" * 32)
                    other.replace(source)
                else:
                    source.write_bytes(b"b" * 32)
            return self.stream.read(size)

    monkeypatch.setattr(q.os, "fdopen", ChangingReader)
    with pytest.raises(
        q.EvidenceError, match="artifact_too_large|source_changed_during_copy"
    ):
        q.bounded_copy(source, dest)
    assert max(requests) <= 65
    assert not dest.exists()


def test_copy_exclusive_hash_deadline_and_exact_limit(tmp_path, monkeypatch):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.write_bytes(b"a" * 64)
    monkeypatch.setattr(q, "MAX_ARTIFACT", 64)
    assert q.bounded_copy(source, dest, expected_hash=q.digest(source)) == q.digest(
        dest
    )
    with pytest.raises(FileExistsError):
        q.bounded_copy(source, dest)
    assert dest.read_bytes() == b"a" * 64
    dest.unlink()
    with pytest.raises(q.EvidenceError, match="source_hash_mismatch"):
        q.bounded_copy(source, dest, expected_hash="e" * 64)
    assert not dest.exists()
    with pytest.raises(q.EvidenceError, match="overall_deadline"):
        q.bounded_copy(source, dest, deadline=time.monotonic() - 1)
    assert not dest.exists()


def test_copy_symlink_refused(tmp_path):
    source = tmp_path / "source"
    source.symlink_to(__file__)
    with pytest.raises(q.EvidenceError, match="artifact_missing_or_symlink"):
        q.bounded_copy(source, tmp_path / "dest")


def test_signals_restored_and_worker_threads_do_not_mutate_signals(
    tmp_path, monkeypatch
):
    import threading

    previous = signal.getsignal(signal.SIGTERM)
    seen = []
    real_signal = signal.signal

    def observed(*args):
        seen.append(threading.current_thread())
        return real_signal(*args)

    monkeypatch.setattr(q.signal, "signal", observed)

    def run(name):
        return q.run_command(
            [sys.executable, "-c", 'print("done")'],
            cwd=tmp_path,
            env=q.clean_env(tmp_path, Path(sys.executable)),
            log=tmp_path / name,
            deadline=time.monotonic() + 3,
        )

    assert run("main.log")["status"] == "passed"
    assert signal.getsignal(signal.SIGTERM) is previous
    assert len(seen) == 2
    results = []
    thread = threading.Thread(target=lambda: results.append(run("thread.log")))
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert results[0]["status"] == "passed"
    assert len(seen) == 2


@pytest.mark.parametrize("mode", ["normal", "timeout", "loglimit"])
def test_owned_group_cleanup_does_not_kill_unrelated_child(tmp_path, mode):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    pids = []
    grand = "import time; time.sleep(30)"
    ending = {
        "normal": "sys.exit(0)",
        "timeout": "time.sleep(30)",
        "loglimit": '\nwhile True: os.write(1,b"x"*8192)',
    }[mode]
    script = f'import os,pathlib,subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c",{grand!r}],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); pathlib.Path("pids").write_text(str(os.getpid())+" "+str(p.pid)); {ending}'
    try:
        result = q.run_command(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=q.clean_env(tmp_path, Path(sys.executable)),
            log=tmp_path / "child.log",
            deadline=time.monotonic() + 1,
            max_bytes=100,
        )
        pids = list(map(int, (tmp_path / "pids").read_text().split()))
        assert (
            result["status"]
            == {"normal": "passed", "timeout": "timeout", "loglimit": "output_limit"}[
                mode
            ]
        )
        end = time.monotonic() + 2
        while any(alive(pid) for pid in pids) and time.monotonic() < end:
            time.sleep(0.01)
        assert not any(alive(pid) for pid in pids)
        assert other.poll() is None
    finally:
        other.kill()
        other.wait(timeout=3)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_real_plugin_result_evaluates_with_attested_source(bundle, tmp_path):
    root, _, report, tests = bundle
    execution = root / "execution"
    execution.mkdir()
    q.write_json(
        execution / "context.json",
        {
            "repo": str(root / "sources"),
            "output": str(execution / "pytest-results.json"),
            "candidate": report["candidate"],
            "kinds": {tests["collected"][0]: "real_storage"},
        },
    )
    env = q.clean_env(
        execution, Path(sys.executable), Path(__file__).resolve().parents[1]
    )
    env["JITMIND_QUALIFICATION_CONTEXT"] = str(execution / "context.json")
    result = q.run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "scripts.qualify_jitmind",
            "tests",
        ],
        cwd=root / "sources",
        env=env,
        log=execution / "pytest.log",
        deadline=time.monotonic() + 10,
    )
    assert result["status"] == "passed"
    for name in ("pytest-results.json", "expected-collection.json"):
        (root / name).write_bytes((execution / name).read_bytes())
    refresh(bundle, tests=False)
    assert verdict(bundle)["checks_only_complete"]


def test_normal_completion_closes_both_pipes(tmp_path, monkeypatch):
    real_popen = q.subprocess.Popen
    children = []

    def capture(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(q.subprocess, "Popen", capture)
    previous = signal.getsignal(signal.SIGTERM)
    handler = lambda _signum, _frame: None
    signal.signal(signal.SIGTERM, handler)
    try:
        result = q.run_command(
            [sys.executable, "-c", 'print("ok")'],
            cwd=tmp_path,
            env=q.clean_env(tmp_path, Path(sys.executable)),
            log=tmp_path / "child.log",
            deadline=time.monotonic() + 3,
        )
        assert result["status"] == "passed"
        assert signal.getsignal(signal.SIGTERM) is handler
        assert children[0].returncode == 0
        assert children[0].stdout.closed and children[0].stderr.closed
    finally:
        signal.signal(signal.SIGTERM, previous)


@pytest.mark.parametrize("error", [FileNotFoundError, ProcessLookupError])
def test_alive_handles_process_disappearance_during_status_read(monkeypatch, error):
    class Status:
        def exists(self):
            return True

        def read_text(self):
            raise error("fixture process exited")

    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    monkeypatch.setitem(alive.__globals__, "Path", lambda path: Status())
    assert alive(12345) is False


@pytest.mark.parametrize(
    "record, expected",
    [("123 (worker) S 1 2", True), ("123 (worker) Z 1 2", False),
     ("123 (worker name) Z 1 2", False)],
)
def test_alive_preserves_live_and_zombie_controls(monkeypatch, record, expected):
    class Status:
        def exists(self):
            return True

        def read_text(self):
            return record

    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    monkeypatch.setitem(alive.__globals__, "Path", lambda path: Status())
    assert alive(12345) is expected


def test_alive_does_not_hide_status_permission_failure(monkeypatch):
    class Status:
        def exists(self):
            return True

        def read_text(self):
            raise PermissionError("fixture permission")

    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    monkeypatch.setitem(alive.__globals__, "Path", lambda path: Status())
    with pytest.raises(PermissionError):
        alive(12345)


def test_alive_does_not_call_malformed_status_dead(monkeypatch):
    class Status:
        def exists(self):
            return True

        def read_text(self):
            return "malformed"

    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    monkeypatch.setitem(alive.__globals__, "Path", lambda path: Status())
    with pytest.raises(ValueError, match="Malformed"):
        alive(12345)

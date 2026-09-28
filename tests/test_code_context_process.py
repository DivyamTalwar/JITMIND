"""Boundary probes using real hostile child processes, separate from parser tests."""

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from jitmind.code_context.process import Unavailable, run_bounded


def run(script, *, payload=b"{}", seconds=2, cancel=None):
    return run_bounded(
        (sys.executable, "-I", "-c", script),
        payload,
        time.monotonic() + seconds,
        cancel,
    )


def test_streaming_pipe_pressure_is_bounded():
    # Both pipes fill while input remains pending. communicate-then-slice would
    # allocate unlimited memory; sequential reading would deadlock here.
    script = 'import os\nwhile True:\n os.write(2,b"e"*8192)\n os.write(1,b"o"*8192)'
    start = time.monotonic()
    with pytest.raises(Unavailable, match="adapter_output_budget"):
        run(script, payload=b"x" * 1000000)
    assert time.monotonic() - start < 2


def test_stuck_process_deadline_and_cancellation():
    with pytest.raises(Unavailable, match="deadline"):
        run("import time; time.sleep(60)", seconds=0.15)
    cancel = threading.Event()
    timer = threading.Timer(0.1, cancel.set)
    timer.start()
    try:
        with pytest.raises(Unavailable, match="cancelled"):
            run("import time; time.sleep(60)", cancel=cancel)
    finally:
        timer.join()


def test_scrubbed_environment_and_controlled_cwd(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sentinel")
    monkeypatch.setenv("NODE_OPTIONS", "--require /evil")
    monkeypatch.setenv("PYTHONPATH", "/evil")
    monkeypatch.setenv("HOME", "/developer/config")
    raw = run(
        'import json,os; print(json.dumps({"env":dict(os.environ),"cwd":os.getcwd()}))'
    )
    result = json.loads(raw)
    assert not any(
        k in result["env"] for k in ("OPENAI_API_KEY", "NODE_OPTIONS", "PYTHONPATH")
    )
    assert Path(result["env"]["HOME"]).resolve() == Path(result["cwd"])
    assert not Path(result["cwd"]).exists()


def test_timeout_kills_grandchildren_holding_pipes(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    child = "import time; time.sleep(60)"
    script = (
        f'import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,"-I","-c",{child!r}]); '
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    with pytest.raises(Unavailable, match="deadline"):
        run(script, seconds=0.3)
    pid = int(pidfile.read_text())
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        # Linux can briefly retain an orphan zombie until PID1 reaps it.
        state = Path(f"/proc/{pid}/stat")
        assert state.exists() and state.read_text().split()[2] == "Z"


def test_exited_leader_with_inherited_pipes_still_deadlines(tmp_path):
    script = 'import subprocess,sys; subprocess.Popen([sys.executable,"-I","-c","import time; time.sleep(60)"])'
    with pytest.raises(Unavailable, match="deadline"):
        run(script, seconds=0.2)


def test_nonzero_exit_is_explicit_unavailable():
    with pytest.raises(Unavailable, match="adapter_failed"):
        run('import sys; print("private diagnostic",file=sys.stderr); sys.exit(4)')


@pytest.mark.parametrize(
    "missing,reason",
    [
        ("executable", "node_executable_unavailable"),
        ("directory", "adapter_directory_unavailable"),
        ("runner", "adapter_runner_unavailable"),
        ("nonexecutable", "node_executable_unavailable"),
        ("runner_symlink", "adapter_runner_unavailable"),
    ],
)
def test_constructor_missing_capability_without_node(
    tmp_path, monkeypatch, missing, reason
):
    from jitmind.code_context import CapabilityUnavailable, NodeParser

    def forbidden(*args, **kwargs):
        pytest.fail("constructor started a runtime")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    directory = tmp_path / "adapter"
    directory.mkdir()
    runner = directory / "runner.mjs"
    runner.write_text("// intentionally never executed")
    node = Path(sys.executable)
    if missing == "executable":
        node = tmp_path / "absent-node"
    elif missing == "directory":
        directory = tmp_path / "absent-adapter"
    elif missing == "runner":
        runner.unlink()
    elif missing == "nonexecutable":
        node = tmp_path / "nonexecutable"
        node.write_text("not an executable")
        node.chmod(0o600)
    else:
        runner.unlink()
        runner.symlink_to(tmp_path / "absent-runner")
    with pytest.raises(CapabilityUnavailable) as error:
        NodeParser(str(node), str(directory))
    assert isinstance(error.value, Unavailable)
    assert error.value.reason == reason
    assert str(error.value) == reason


def test_constructor_valid_paths_does_not_start_runtime(tmp_path, monkeypatch):
    from jitmind.code_context import NodeParser

    def forbidden(*args, **kwargs):
        pytest.fail("constructor started a runtime")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    (tmp_path / "runner.mjs").write_text("// intentionally never executed")
    parser = NodeParser(sys.executable, str(tmp_path))
    assert parser.argv == (
        str(Path(sys.executable).resolve()),
        "--max-old-space-size=128",
        str(tmp_path.resolve() / "runner.mjs"),
    )
    for node, directory in [("node", str(tmp_path)), (sys.executable, "adapter")]:
        with pytest.raises(ValueError, match="Trusted absolute runtime paths required"):
            NodeParser(node, directory)


def test_fresh_optional_import_without_node_or_provider():
    import jitmind.scope

    # Fresh isolated Python, no PATH in run_bounded's environment. J03 may be
    # supplied through the test-only peer package path; never copy its source.
    root = str(Path(__file__).resolve().parents[1])
    scope_package = str(Path(jitmind.scope.__file__).resolve().parent)
    script = f"""
import sys
def deny(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'socket.connect'):
        raise AssertionError('optional import attempted runtime/provider access')
sys.addaudithook(deny)
sys.path.insert(0, {root!r})
import jitmind
jitmind.__path__.append({scope_package!r})
from jitmind.code_context import CapabilityUnavailable, CodeContext, NodeParser
try:
    NodeParser('/definitely-absent-jitmind-node', '/definitely-absent-jitmind-adapter')
except CapabilityUnavailable as error:
    assert error.reason == 'node_executable_unavailable'
else:
    raise AssertionError('missing capability accepted')
print('optional-import-ok')
"""
    assert run(script, seconds=10).decode().strip().endswith("optional-import-ok")

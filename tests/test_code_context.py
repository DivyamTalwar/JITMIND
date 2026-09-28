"""Actual filesystem + extracted Graft/tree-sitter integration (no provider calls)."""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from jitmind.code_context import (
    CodeContext,
    CodeContextRetriever,
    CodeQuery,
    NodeParser,
    RepoRegistry,
)
from jitmind.code_context.models import MAX_RESPONSE_BYTES
from jitmind.code_context.snapshot import capture
from jitmind.scope import ScopeAuthority, ScopeContext, ScopeDenied

ADAPTER = Path(__file__).resolve().parents[1] / "adapters/graft"


@pytest.fixture
def host(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    repo_id = str(uuid4())
    authority = ScopeAuthority()
    authority.grant("alice", "team", (repo_id,))
    scope = authority.context("alice", "team", "request")
    registry = RepoRegistry()
    registry.register(repo_id, str(root.resolve()), "team")
    node = os.environ.get("JITMIND_TEST_NODE") or shutil.which("node")
    assert node, "Provision Node for actual adapter tests"
    parser = NodeParser(str(Path(node).resolve()), str(ADAPTER))
    return root, repo_id, authority, scope, registry, parser


def built(
    host, source="def alpha():\n    return beta()\ndef beta():\n    return alpha()\n"
):
    root, repo, auth, scope, registry, parser = host
    (root / "main.py").write_text(source)
    service = CodeContext(auth, registry, parser)
    page = service.build(scope, repo, deadline_ms=10000)
    assert page.status == "ok", page
    return service, page


def query(host, page, operation, **kwargs):
    return CodeQuery(
        operation=operation,
        repo_id=host[1],
        snapshot_id=page.snapshot_id,
        deadline_ms=10000,
        **kwargs,
    )


def test_real_parser_cycle_scope_depth_workspace(host):
    root = host[0]
    (root / "src/api").mkdir(parents=True)
    (root / "src/apix").mkdir()
    (root / "src/api/a.py").write_text(
        "def a():\n    b()\ndef b():\n    c()\ndef c():\n    a()\n"
    )
    (root / "src/apix/a.py").write_text("def b():\n    pass\n")
    service, page = built(host)
    request = query(
        host,
        page,
        "trace_calls",
        query="a",
        path_prefix="src/api",
        direction="out",
        depth="all",
    )
    result = service.query(host[3], request)
    assert [e.name for e in result.results] == ["b", "c"]
    assert [e.depth for e in result.results] == [1, 2]
    assert all(e.path == "src/api/a.py" for e in result.results)
    assert service.workspace(host[3], [request])[0].results == result.results
    direct = service.query(host[3], request.model_copy(update={"depth": 1}))
    assert [e.name for e in direct.results] == ["b"]
    incoming = service.query(
        host[3], request.model_copy(update={"direction": "in", "depth": 1})
    )
    assert [e.name for e in incoming.results] == ["c"]
    limited = service.query(host[3], request.model_copy(update={"limit": 1}))
    assert limited.truncated and "result_limit" in limited.reason_codes
    assert result.coverage["runtime_complete"] is False


def test_real_parser_unicode_crlf_and_operations(host):
    source = '# λ comment\r\nclass Café:\r\n    def run(self):\r\n        return "你好"\r\n\r\ndef target():\r\n    return 2\r\n'
    service, page = built(host, source)
    result = service.query(host[3], query(host, page, "file_api", query="main.py"))
    assert {r.name for r in result.results} == {"Café", "run", "target"}
    for e in result.results:
        assert e.source == "".join(
            source.splitlines(keepends=True)[e.start_line - 1 : e.end_line]
        )
        assert e.digest == hashlib.sha256(source.encode()).hexdigest()
    matched = service.query(host[3], query(host, page, "find_all", query="你好"))
    assert len(matched.results) == 1 and matched.results[0].start_line == 4
    ranked = service.query(host[3], query(host, page, "find_code", query="target"))
    assert ranked.results[0].name == "target"
    mapped = service.query(host[3], query(host, page, "repo_map"))
    assert mapped.results[0].path == "main.py"
    assert mapped.response_bytes == len(mapped.wire_bytes()) <= MAX_RESPONSE_BYTES
    assert mapped.source_bytes == sum(len(e.source.encode()) for e in mapped.results)


def test_same_size_edit_new_removed_and_old_generation(host):
    service, page = built(host, "def old():\n    return 1\n")
    path = host[0] / "main.py"
    old_time = path.stat().st_mtime_ns
    path.write_text("def new():\n    return 2\n")
    os.utime(path, ns=(old_time, old_time))
    fresh = service.query(host[3], query(host, page, "check_freshness"))
    assert (
        fresh.freshness["changed"] == ["main.py"]
        and fresh.freshness["state"] == "stale"
    )
    assert fresh.generation == page.generation
    assert (
        service.query(host[3], query(host, page, "file_api", query="main.py"))
        .results[0]
        .name
        == "old"
    )
    newer = service.build(host[3], host[1], deadline_ms=10000)
    assert newer.snapshot_id != page.snapshot_id
    path.unlink()
    (host[0] / "added.py").write_text("def added(): pass\n")
    changed = service.query(host[3], query(host, newer, "check_freshness"))
    assert changed.freshness["removed"] == ["main.py"]
    assert changed.freshness["added"] == ["added.py"]


def test_unsupported_encoding_parse_and_unreadable_unknown(host, monkeypatch):
    service, page = built(host)
    (host[0] / "bad.py").write_bytes(b"\xff\xfeX\x00")
    (host[0] / "other.ts").write_text("function t() {}")
    (host[0] / "syntax.py").write_text("def broken(:\n")
    newer = service.build(host[3], host[1], deadline_ms=10000)
    assert newer.coverage["files"]["encoding_unknown"] == 1
    assert newer.coverage["files"]["parse_error"] == 1
    assert newer.coverage["files"]["unsupported"] == 1
    real_open = os.open

    def deny(path, *args, **kwargs):
        if path == "main.py":
            raise PermissionError("fixture unreadable")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny)
    result = service.query(host[3], query(host, page, "check_freshness"))
    assert "main.py" in result.freshness["unknown"]
    assert result.freshness["removed"] == []
    assert result.freshness["state"] == "unknown"


def test_symlink_sentinel_never_opened_and_basename_identity(
    host, tmp_path, monkeypatch
):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("def SECRET_SENTINEL(): pass\n")
    (host[0] / "escape.py").symlink_to(outside / "secret.py")
    (host[0] / "directory").symlink_to(outside, target_is_directory=True)
    real_open = os.open
    opened = []

    def track(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", track)
    service, page = built(host)
    result = service.query(
        host[3], query(host, page, "find_all", query="SECRET_SENTINEL")
    )
    assert not result.results
    assert not any("secret.py" in p or p in ("escape.py", "directory") for p in opened)
    second = tmp_path / "other/repo"
    second.mkdir(parents=True)
    (second / "main.py").write_text("def different(): pass\n")
    other_id = str(uuid4())
    host[4].register(other_id, str(second.resolve()), "team")
    host[2].grant("alice", "team", (host[1], other_id))
    scope = host[2].context("alice", "team", "two")
    other = service.build(scope, other_id, deadline_ms=10000)
    assert other.snapshot_id != page.snapshot_id
    with pytest.raises(ScopeDenied):
        service.query(host[3], query(host, page, "repo_map"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("limit", True),
        ("limit", 0),
        ("limit", 51),
        ("deadline_ms", float("nan")),
        ("max_context_tokens", float("inf")),
        ("depth", True),
        ("depth", 0),
        ("depth", "full"),
        ("fixed", False),
        ("path_prefix", "src/../x"),
        ("path_prefix", "/etc"),
        ("path_prefix", "C:\\x"),
        ("path_prefix", "\\\\host\\x"),
        ("path_prefix", "x\x00y"),
        ("path_prefix", "src//x"),
    ],
)
def test_strict_requests(host, field, value):
    with pytest.raises((ValueError, ValidationError)):
        CodeQuery(
            operation="repo_map",
            repo_id=host[1],
            snapshot_id="a" * 64,
            **{field: value},
        )


def test_duplicate_json_and_nonfinite(host):
    for raw in (b'{"operation":"repo_map","operation":"find_all"}', b'{"limit":NaN}'):
        with pytest.raises(ValueError):
            CodeQuery.from_json(raw)


def test_denied_before_read_and_revoked_before_return(host, monkeypatch):
    service, page = built(host)
    forged = ScopeContext(
        "alice", "team", (host[1],), host[3].authorization_version, "fake"
    )

    def forbidden(*args, **kwargs):
        pytest.fail("disk accessed before authorization")

    with monkeypatch.context() as m:
        m.setattr("jitmind.code_context.service.capture", forbidden)
        with pytest.raises(ScopeDenied):
            service.query(forged, query(host, page, "repo_map"))
    real_capture = capture

    def revoke(*args, **kwargs):
        value = real_capture(*args, **kwargs)
        host[2].revoke("alice", "team")
        return value

    monkeypatch.setattr("jitmind.code_context.service.capture", revoke)
    with pytest.raises(ScopeDenied):
        service.query(host[3], query(host, page, "repo_map"))


def test_cancelled_build_and_changed_during_parser(host):
    service, old = built(host)
    event = threading.Event()
    event.set()
    result = service.build(host[3], host[1], cancel=event)
    assert result.status == "unavailable" and result.reason_codes == ("cancelled",)
    real_parse = host[5].parse

    def edit(*args):
        result = real_parse(*args)
        (host[0] / "main.py").write_text("def changed(): pass\n")
        return result

    host[5].parse = edit
    changed = service.build(host[3], host[1], deadline_ms=10000)
    assert changed.status == "unavailable" and changed.reason_codes == (
        "source_changed",
    )
    assert (
        service.query(host[3], query(host, old, "file_api", query="main.py")).generation
        == old.generation
    )


def test_response_forgery_rejected_and_no_partial_generation(host):
    import json

    service, old = built(host)
    real_parse = host[5].parse
    for mutation in ("snapshot", "digest", "path", "span", "duplicate", "nonfinite"):

        def forged(*args, mutation=mutation):
            raw = real_parse(*args)
            response = json.loads(raw)
            if mutation == "snapshot":
                response["snapshot_id"] = "f" * 64
            elif mutation == "digest":
                response["files"][0]["digest"] = "f" * 64
            elif mutation == "path":
                response["files"][0]["path"] = "../secret.py"
            elif mutation == "span":
                response["files"][0]["symbols"][0]["start_line"] = 0
            elif mutation == "duplicate":
                return b'{"schema":1,"schema":2}'
            elif mutation == "nonfinite":
                return b'{"value":NaN}'
            return json.dumps(response).encode()

        host[5].parse = forged
        result = service.build(host[3], host[1], deadline_ms=10000)
        assert result.status == "unavailable" and result.reason_codes == (
            "invalid_adapter_response",
        )
    assert (
        service.query(host[3], query(host, old, "repo_map")).generation
        == old.generation
    )


def test_optional_retriever_keeps_evidence_no_pagestore_ids(host):
    service, page = built(host)
    retriever = CodeContextRetriever(service, host[3], host[1], page.snapshot_id)
    hit = retriever.search(["alpha"])[0][0]
    assert hit.page_id is None
    assert hit.meta["code_evidence"]["snapshot_id"] == page.snapshot_id
    assert hit.meta["code_evidence"]["path"] == "main.py"
    service.parser.argv = ("/nonexistent/node",)
    result = service.build(host[3], host[1])
    assert result.status == "unavailable" and result.reason_codes == (
        "runtime_unavailable",
    )
    assert retriever.search(["alpha"])[0]  # existing authorized captured generation


def test_context_budget_and_retention(host):
    service, page = built(host)
    result = service.query(
        host[3], query(host, page, "find_code", query="alpha", max_context_tokens=10)
    )
    assert result.context_tokens_upper_bound <= 10 and result.truncated
    for number in range(9):
        (host[0] / "main.py").write_text(f"def v{number}(): pass\n")
        assert service.build(host[3], host[1], deadline_ms=10000).status == "ok"
    assert len(service._indices) == 8
    assert service.query(host[3], query(host, page, "repo_map")).reason_codes == (
        "snapshot_unavailable",
    )


def test_final_file_swap_to_symlink_never_opens_target(host, tmp_path, monkeypatch):
    service, _ = built(host)
    sentinel = tmp_path / "outside.py"
    sentinel.write_text("def SECRET_SENTINEL(): pass\n")
    target = host[0] / "main.py"
    real_open = os.open
    swapped = False

    def race(path, *args, **kwargs):
        nonlocal swapped
        if path == "main.py" and not swapped:
            swapped = True
            target.unlink()
            target.symlink_to(sentinel)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", race)
    result = service.build(host[3], host[1], deadline_ms=10000)
    assert result.status == "unavailable"
    assert result.reason_codes == ("source_changed",)
    assert "SECRET_SENTINEL" not in result.model_dump_json()


def test_directory_swap_rejected(host, tmp_path, monkeypatch):
    nested = host[0] / "src"
    nested.mkdir()
    (nested / "module.py").write_text("def old(): pass\n")
    service, _ = built(host)
    moved = tmp_path / "moved"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "module.py").write_text("def SECRET_SENTINEL(): pass\n")
    real_open = os.open
    swapped = False

    def race(path, *args, **kwargs):
        nonlocal swapped
        if path == "src" and not swapped:
            swapped = True
            nested.rename(moved)
            nested.symlink_to(outside, target_is_directory=True)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", race)
    result = service.build(host[3], host[1], deadline_ms=10000)
    assert result.status == "unavailable"
    assert "SECRET_SENTINEL" not in result.model_dump_json()


def test_coverage_limits_and_namespace_preflight(host, monkeypatch):
    service, old = built(host)
    (host[0] / "large.py").write_bytes(b"x" * 131073)
    page = service.build(host[3], host[1], deadline_ms=10000)
    assert page.coverage["files"]["input_budget"] == 1
    host[2].grant("bob", "other-namespace", (host[1],))
    scope = host[2].context("bob", "other-namespace", "other")

    def forbidden(*args, **kwargs):
        pytest.fail("read before namespace registry check")

    monkeypatch.setattr("jitmind.code_context.service.capture", forbidden)
    with pytest.raises(ScopeDenied):
        service.query(scope, query(host, old, "repo_map"))
    # Re-grant same principal and namespace changes the cache authorization key.
    host[2].grant("alice", "team", (host[1],))
    new_scope = host[2].context("alice", "team", "new-version")
    assert service.query(new_scope, query(host, old, "repo_map")).reason_codes == (
        "snapshot_unavailable",
    )


def test_copied_model_cannot_bypass_boundaries(host):
    service, page = built(host)
    request = query(host, page, "trace_calls", query="alpha").model_copy(
        update={"limit": True}
    )
    with pytest.raises(ValidationError):
        service.query(host[3], request)


def test_large_real_parser_callback_input(host):
    # Exercises donor's chunked callback (>32 KiB) rather than parse(string).
    source = "# comment padding\n" * 2200 + "def callback_large():\n    return 42\n"
    service, page = built(host, source)
    result = service.query(host[3], query(host, page, "file_api", query="main.py"))
    assert result.results[0].start_line == 2201
    assert result.results[0].name == "callback_large"


def test_real_parser_recorded_fixture_golden(host):
    import json
    import time

    from jitmind.code_context.models import PROTOCOL
    from jitmind.code_context.snapshot import canonical, digest

    source = (ADAPTER / "fixtures/cycle.py").read_text()
    expected = json.loads((ADAPTER / "fixtures/cycle.expected.json").read_text())
    payload = canonical(
        {
            "schema": PROTOCOL,
            "snapshot_id": "a" * 64,
            "files": [
                {
                    "path": "cycle.py",
                    "source": source,
                    "digest": digest(source.encode()),
                }
            ],
        }
    )
    raw = host[5].parse(payload, time.monotonic() + 10)
    file = json.loads(raw)["files"][0]
    assert {"status": file["status"], "symbols": file["symbols"]} == expected


def test_same_size_mutation_during_open_read_is_unknown(host, monkeypatch):
    service, old = built(host, "def old(): pass\n")
    real_read = os.read
    touched = False

    def race(fd, count):
        nonlocal touched
        data = real_read(fd, count)
        if data == b"def old(): pass\n" and not touched:
            touched = True
            (host[0] / "main.py").write_text("def new(): pass\n")
        return data

    monkeypatch.setattr(os, "read", race)
    result = service.query(host[3], query(host, old, "check_freshness"))
    assert result.status == "unavailable" and result.reason_codes == ("source_changed",)


def test_unicode_separators_do_not_change_ast_line_spans(host):
    source = 'marker = "x\u2028y"\ndef actual():\n    return marker\n'
    service, page = built(host, source)
    result = service.query(host[3], query(host, page, "file_api", query="main.py"))
    assert result.results[0].start_line == 2
    assert result.results[0].source == "def actual():\n    return marker\n"


@pytest.mark.parametrize(
    "path", ["/etc/passwd", "../secret.py", "C:\\secret.py", "\\\\host\\secret.py"]
)
def test_file_api_path_rejected_at_schema(host, path):
    with pytest.raises(ValidationError):
        CodeQuery(
            operation="file_api", repo_id=host[1], snapshot_id="a" * 64, query=path
        )


@pytest.mark.parametrize("call_count", [4000, 4096, 4097, 6000])
def test_real_parser_unresolved_candidate_budget(host, call_count):
    # All calls are unresolved; per-symbol limits remain valid in every fixture.
    source = "".join(
        f"def f{offset}():\n" + "    missing()\n" * min(1024, call_count - offset)
        for offset in range(0, call_count, 1024)
    )
    service, page = built(host, source)
    index = next(iter(service._indices.values()))
    assert sum(len(n.calls) for n in index.nodes) == call_count
    result = service.query(
        host[3],
        query(host, page, "trace_calls", query="f0", direction="out", depth="all"),
    )
    assert result.status == "ok" and not result.results
    assert result.truncated is (call_count > 4096)
    assert ("edge_budget" in result.reason_codes) is (call_count > 4096)
    assert result.count == (None if call_count > 4096 else 0)


@pytest.mark.parametrize("length", [2048, 2500])
def test_real_parser_long_unicode_signature_discloses_cut(host, length):
    signature = 'def alpha(x="' + "界" * (length - len('def alpha(x=""):')) + '"):'
    source = signature + "\n    return x\n"
    service, page = built(host, source)
    result = service.query(
        host[3],
        query(host, page, "file_api", query="main.py", max_context_tokens=16000),
    )
    assert result.status == "ok" and result.count == 1
    evidence = result.results[0]
    assert evidence.signature == signature[:2048]
    assert evidence.source == source
    assert (evidence.start_line, evidence.end_line) == (1, 2)
    assert evidence.digest == hashlib.sha256(source.encode()).hexdigest()
    assert result.truncated is (length > 2048)
    assert ("signature_truncated" in result.reason_codes) is (length > 2048)
    assert result.source_bytes == len(source.encode())
    assert result.context_tokens_upper_bound == len(
        (source + evidence.name + evidence.signature).encode()
    )


@pytest.mark.parametrize("source", ["", "def alpha(): pass\n"])
def test_real_parser_empty_file_map_metadata_and_count(host, source):
    (host[0] / "empty.py").write_bytes(b"")
    service, page = built(host, source)
    result = service.query(host[3], query(host, page, "repo_map"))
    empty_paths = ["empty.py"] + ([] if source else ["main.py"])
    assert result.coverage["empty_files"] == [
        {"path": path, "digest": hashlib.sha256(b"").hexdigest()}
        for path in empty_paths
    ]
    assert result.coverage["files"] == {"ok": 2}
    assert result.coverage["state"] == "complete_subset"
    assert result.count == len(result.results) == bool(source)
    assert [e.path for e in result.results] == (["main.py"] if source else [])
    assert not result.truncated
    assert "empty_files_without_source" in result.reason_codes
    assert result.source_bytes == sum(len(e.source.encode()) for e in result.results)
    assert result.response_bytes == len(result.wire_bytes())
    scoped = service.query(
        host[3], query(host, page, "repo_map", path_prefix="empty.py")
    )
    assert scoped.count == 0 and not scoped.results
    assert scoped.coverage["empty_files"] == result.coverage["empty_files"][:1]
    # Empty versus a single newline is a content change, not deletion.
    (host[0] / "empty.py").write_bytes(b"\n")
    freshness = service.query(host[3], query(host, page, "check_freshness"))
    assert freshness.freshness["changed"] == ["empty.py"]
    newer = service.build(host[3], host[1], deadline_ms=10000)
    assert newer.snapshot_id != page.snapshot_id
    mapped = service.query(
        host[3], query(host, newer, "repo_map", path_prefix="empty.py")
    )
    assert mapped.coverage["empty_files"] == []
    assert mapped.count == 1 and mapped.results[0].source == "\n"
    assert mapped.results[0].start_line == mapped.results[0].end_line == 1


def controlled_clock(monkeypatch, start=100.0):
    clock = SimpleNamespace(now=start)
    timer = SimpleNamespace(monotonic=lambda: clock.now)
    monkeypatch.setattr("jitmind.code_context.service.time", timer)
    monkeypatch.setattr("jitmind.code_context.process.time", timer)
    return clock


def test_workspace_checks_expiry_before_first_dispatch(host, monkeypatch):
    service, page = built(host)
    calls = iter([100.0, 100.001])
    timer = SimpleNamespace(monotonic=lambda: next(calls, 100.001))
    monkeypatch.setattr("jitmind.code_context.service.time", timer)
    monkeypatch.setattr("jitmind.code_context.process.time", timer)

    def forbidden(*args, **kwargs):
        pytest.fail("workspace dispatched after its deadline")

    monkeypatch.setattr("jitmind.code_context.service.capture", forbidden)
    request = query(host, page, "repo_map").model_copy(update={"deadline_ms": 1})
    pages = service.workspace(host[3], [request] * 3)
    assert len(pages) == 3
    assert all(
        p.status == "unavailable" and p.reason_codes == ("deadline",) for p in pages
    )


@pytest.mark.parametrize("elapsed", [(0.001,), (0.00075, 0.00030)])
def test_workspace_shares_absolute_deadline_and_stops_dispatch(
    host, monkeypatch, elapsed
):
    service, page = built(host)
    clock = controlled_clock(monkeypatch)
    durations = iter(elapsed)
    deadlines = []
    starts = []

    def timed_capture(repo, deadline, cancel):
        starts.append(clock.now)
        deadlines.append(deadline)
        result = capture(repo, deadline, cancel)
        clock.now += next(durations, 0)
        return result

    monkeypatch.setattr("jitmind.code_context.service.capture", timed_capture)
    request = query(host, page, "repo_map").model_copy(update={"deadline_ms": 1})
    pages = service.workspace(host[3], [request] * 3)
    assert len(starts) == len(elapsed)
    assert all(start < 100.001 for start in starts)
    assert deadlines == [100.001] * len(elapsed)
    assert len(pages) == 3
    assert all(
        p.status == "unavailable" and p.reason_codes == ("deadline",) for p in pages
    )


def test_workspace_valid_budget_preserves_request_deadlines(host, monkeypatch):
    service, page = built(host)
    clock = controlled_clock(monkeypatch)
    deadlines = []

    def timed_capture(repo, deadline, cancel):
        deadlines.append(deadline)
        result = capture(repo, deadline, cancel)
        clock.now += 0.00025
        return result

    monkeypatch.setattr("jitmind.code_context.service.capture", timed_capture)
    request = query(host, page, "repo_map")
    pages = service.workspace(
        host[3],
        [
            request.model_copy(update={"deadline_ms": 1}),
            request.model_copy(update={"deadline_ms": 10}),
        ],
    )
    assert all(p.status == "ok" and not p.truncated for p in pages)
    assert deadlines == [100.001, 100.01]

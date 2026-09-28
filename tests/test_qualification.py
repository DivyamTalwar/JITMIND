"""Gate/tool tests use controlled evidence, never product-acceptance credit."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import signal
import sys
import threading
import time

import pytest

from scripts import qualify_jitmind as q


def save(path, data):
    path.write_text(json.dumps(data, allow_nan=False))


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "sources/tests/test_product.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_contract():\n    assert True\n")
    ref = {"path": "tests/test_product.py", "sha256": q.digest(source)}
    mapping = {
        "schema": "jitmind.acceptance-map/v1",
        "source_handoff_sha256": "a" * 64,
        "cases": [
            {
                "case_id": "J01",
                "ticket": "J-01",
                "status": "not_run",
                "required_in_tickets": ["J-01"],
                "stages": [
                    {
                        "stage_id": "J-01",
                        "status": "not_run",
                        "nodes": [
                            {
                                "node_id": "tests/test_product.py::test_contract",
                                "kind": "real_storage",
                                "source": ref,
                                "fixtures": [ref],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    roster = [[ref["path"], ref["sha256"]]]
    candidate = {
        "git_sha": "b" * 40,
        "snapshot_files": roster,
        "snapshot_sha256": hashlib.sha256(
            json.dumps(roster, separators=(",", ":")).encode()
        ).hexdigest(),
        "dirty": False,
    }
    node = mapping["cases"][0]["stages"][0]["nodes"][0]["node_id"]
    tests = {
        "schema": "jitmind.pytest/v2",
        "candidate": copy.deepcopy(candidate),
        "collected": [node],
        "collection_count": 1,
        "exit_code": 0,
        "records": [
            {
                "node_id": node,
                "kind": "real_storage",
                "source_sha256": ref["sha256"],
                "status": "passed",
                "phases": {"setup": "passed", "call": "passed", "teardown": "passed"},
            }
        ],
    }
    expected = {
        "schema": "jitmind.collection/v1",
        "candidate": copy.deepcopy(candidate),
        "nodes": [node],
        "selected": [node],
        "deselected": [],
        "collection_reports": [],
        "filtered": False,
    }
    save(tmp_path / "expected-collection.json", expected)
    tests.update(
        expected_collection_sha256=q.digest(tmp_path / "expected-collection.json"),
        phase_counts={"passed": 3, "failed": 0, "skipped": 0},
        session_testscollected=1,
        session_testsfailed=0,
    )
    save(tmp_path / "pytest-results.json", tests)
    save(tmp_path / "acceptance-map.json", mapping)
    (tmp_path / "candidate.whl").write_bytes(
        b"controlled-test-fixture-not-an-installable-wheel"
    )
    smoke = {
        "schema": "jitmind.installed-smoke/v1",
        "node_id": q.SMOKE_NODE,
        "checks": dict.fromkeys(q.SMOKE_CHECKS, "passed"),
        "status": "passed",
        "provider_calls": 0,
        "source_sha256": "d" * 64,
    }
    save(tmp_path / "installed-results.json", smoke)
    commands = []
    for name in sorted(q.COMMANDS):
        (tmp_path / (name + ".log")).write_text(
            "controlled fixture, not actual execution\n"
        )
        commands.append(
            {
                "id": name,
                "status": "passed",
                "exit_code": 0,
                "log_truncated": False,
                "log": name + ".log",
            }
        )
    report = {
        "schema": "jitmind.qualification/v2",
        "candidate": candidate,
        "mode": "release",
        "commands": commands,
        "findings": [],
        "test_results": "pytest-results.json",
        "expected_collection": "expected-collection.json",
        "package_sha256": q.digest(tmp_path / "candidate.whl"),
        "acceptance_map": q.artifact(tmp_path, "acceptance-map.json"),
        "artifacts": [
            q.artifact(tmp_path, str(p.relative_to(tmp_path)))
            for p in tmp_path.rglob("*")
            if p.is_file() and p.name != "acceptance-map.json"
        ],
    }
    return tmp_path, mapping, report, tests


def refresh(bundle, *, tests=True, mapping=True):
    root, expected, report, records = bundle
    if tests:
        collection = q.strict_json(root / "expected-collection.json")
        collection["candidate"] = copy.deepcopy(report["candidate"])
        save(root / "expected-collection.json", collection)
        records["expected_collection_sha256"] = q.digest(
            root / "expected-collection.json"
        )
        if records.get("available", True):
            records["phase_counts"] = {
                outcome: sum(
                    p == outcome
                    for r in records["records"]
                    for p in r["phases"].values()
                )
                for outcome in ("passed", "failed", "skipped")
            }
            records["session_testscollected"] = records["collection_count"]
            records["session_testsfailed"] = records["phase_counts"]["failed"] + sum(
                r["outcome"] == "failed" for r in collection["collection_reports"]
            )
        else:
            records.update(
                phase_counts=None, session_testscollected=None, session_testsfailed=None
            )
        save(root / "pytest-results.json", records)
    if mapping:
        save(root / "acceptance-map.json", expected)
        report["acceptance_map"] = q.artifact(root, "acceptance-map.json")
    report["artifacts"] = [q.artifact(root, ref["path"]) for ref in report["artifacts"]]


def verdict(bundle):
    root, mapping, report, _ = bundle
    return q.evaluate(mapping, report, root)


def test_complete_controlled_fixture_is_consistent_but_never_signoff(bundle):
    result = verdict(bundle)
    assert result["checks_only_complete"] is True
    assert result["release_qualified"] is False
    assert result["case_count"] == result["stage_count"] == result["passed_cases"] == 1
    assert result["benchmark_measurement"] is None
    assert "independent_signoff_not_recorded" in result["release_blockers"]


@pytest.mark.parametrize(
    "raw", ['{"a":1,"a":2}', '{"n":NaN}', '{"n":Infinity}', '{"n":1e999}', "{"]
)
def test_untrusted_json_rejects_duplicates_nonfinite_truncation(tmp_path, raw):
    path = tmp_path / "input.json"
    path.write_text(raw)
    with pytest.raises(q.EvidenceError):
        q.strict_json(path)


def test_oversized_json_is_bounded_before_decode(tmp_path):
    path = tmp_path / "input.json"
    path.write_bytes(b" " * 101)
    with pytest.raises(q.EvidenceError, match="input_too_large"):
        q.strict_json(path, limit=100)


def test_duplicate_case_rejected(bundle):
    bundle[1]["cases"].append(copy.deepcopy(bundle[1]["cases"][0]))
    refresh(bundle)
    with pytest.raises(q.EvidenceError, match="duplicate_case"):
        verdict(bundle)


def test_missing_repeat_stage_rejected(bundle):
    bundle[1]["cases"][0]["required_in_tickets"].append("J-04")
    refresh(bundle)
    with pytest.raises(q.EvidenceError, match="missing_stage"):
        verdict(bundle)


def test_unmapped_stage_keeps_denominator(bundle):
    case = bundle[1]["cases"][0]
    case["required_in_tickets"].append("J-04")
    case["stages"].append({"stage_id": "J-04", "nodes": [], "status": "not_run"})
    refresh(bundle)
    result = verdict(bundle)
    assert result["case_count"] == 1 and result["stage_count"] == 2
    assert result["passed_cases"] == 0 and not result["checks_only_complete"]


def test_partial_mapping_cannot_complete_stage(bundle):
    bundle[1]["cases"][0]["stages"][0]["additional_mapping_required"] = True
    refresh(bundle)
    assert not verdict(bundle)["checks_only_complete"]


def test_no_checks_false_pass_rejected(bundle):
    bundle[2]["commands"] = []
    assert not verdict(bundle)["checks_only_complete"]


@pytest.mark.parametrize("field", ["collection_count", "exit_code"])
def test_boolean_is_not_integer(bundle, field):
    bundle[3][field] = True if field == "collection_count" else False
    refresh(bundle)
    with pytest.raises(q.EvidenceError):
        verdict(bundle)


def test_modified_artifact_rejected(bundle):
    (bundle[0] / "pytest.log").write_text("tampered")
    with pytest.raises(q.EvidenceError, match="artifact_modified"):
        verdict(bundle)


@pytest.mark.parametrize("value", ["main", "a" * 39, "z" * 40, True])
def test_invalid_candidate_sha(bundle, value):
    bundle[2]["candidate"]["git_sha"] = value
    with pytest.raises(q.EvidenceError, match="invalid_candidate_sha"):
        verdict(bundle)


def test_source_identity_mismatch(bundle):
    bundle[3]["candidate"]["snapshot_sha256"] = "e" * 64
    refresh(bundle)
    with pytest.raises(q.EvidenceError, match="test_source_identity_mismatch"):
        verdict(bundle)


def test_absent_fixture_never_passes(bundle):
    bundle[1]["cases"][0]["stages"][0]["nodes"][0]["fixtures"] = [
        {"path": "fixtures/missing.json", "sha256": "e" * 64}
    ]
    refresh(bundle)
    assert not verdict(bundle)["checks_only_complete"]


def test_source_digest_not_current_cannot_pass(bundle):
    bundle[1]["cases"][0]["stages"][0]["nodes"][0]["source"]["sha256"] = "e" * 64
    refresh(bundle)
    assert not verdict(bundle)["checks_only_complete"]


def test_unit_stub_cannot_be_product_acceptance(bundle):
    bundle[3]["records"][0]["kind"] = "unit_stub"
    refresh(bundle)
    assert not verdict(bundle)["checks_only_complete"]


def test_synthetic_measurement_cannot_claim_unrun_real_comparison(bundle):
    bundle[2]["benchmark_measurement"] = {
        "workload": "real_donor",
        "synthetic": True,
        "latency": 0,
    }
    with pytest.raises(q.EvidenceError, match="unreviewed_benchmark_claim"):
        verdict(bundle)


def test_empty_pytest_collection_never_passes(bundle):
    bundle[3].update(collected=[], records=[], collection_count=0)
    refresh(bundle)
    assert not verdict(bundle)["checks_only_complete"]


def test_missing_expected_node_never_passes(bundle):
    bundle[1]["cases"][0]["stages"][0]["nodes"][0]["node_id"] += "_missing"
    refresh(bundle)
    assert not verdict(bundle)["checks_only_complete"]


@pytest.mark.parametrize(
    "phase,outcome", [("call", "failed"), ("setup", "skipped"), ("teardown", "failed")]
)
def test_nonpass_retains_case_and_count(bundle, phase, outcome):
    record = bundle[3]["records"][0]
    record["phases"][phase] = outcome
    record["status"] = outcome
    refresh(bundle)
    result = verdict(bundle)
    assert not result["checks_only_complete"] and result["case_count"] == 1
    assert result["test_counts"][outcome] == 1


def test_missing_teardown_is_unknown(bundle):
    record = bundle[3]["records"][0]
    del record["phases"]["teardown"]
    record["status"] = "unknown"
    refresh(bundle)
    assert verdict(bundle)["test_counts"]["unknown"] == 1


def test_claimed_pass_with_failed_phase_rejected(bundle):
    bundle[3]["records"][0]["phases"]["call"] = "failed"
    refresh(bundle)
    with pytest.raises(q.EvidenceError, match="false_test_pass"):
        verdict(bundle)


def test_duplicate_test_record_rejected(bundle):
    bundle[3]["records"].append(copy.deepcopy(bundle[3]["records"][0]))
    refresh(bundle)
    with pytest.raises(q.EvidenceError, match="duplicate_or_uncollected_test"):
        verdict(bundle)


def test_truncated_logs_never_pass(bundle):
    bundle[2]["commands"][0]["log_truncated"] = True
    assert not verdict(bundle)["checks_only_complete"]


def test_missing_checker_retains_existing_finding(bundle):
    bundle[2]["findings"] = ["known_prior_failure"]
    bundle[2]["commands"].pop()
    result = verdict(bundle)
    assert result["findings"] and not result["checks_only_complete"]


def test_dirty_release_refused_worktree_labeled(bundle):
    bundle[2]["candidate"]["dirty"] = True
    bundle[3]["candidate"]["dirty"] = True
    refresh(bundle)
    with pytest.raises(q.EvidenceError, match="dirty_release"):
        verdict(bundle)
    bundle[2]["mode"] = "worktree"
    assert "worktree_candidate" in verdict(bundle)["release_blockers"]


def test_artifact_escape_and_symlink_refused(tmp_path):
    (tmp_path / "target").write_text("x")
    (tmp_path / "link").symlink_to(tmp_path / "target")
    for name in ["../outside", "/etc/passwd", "link", "a\\b"]:
        with pytest.raises(q.EvidenceError):
            q.artifact(tmp_path, name)


def test_exclusive_report_creation(tmp_path):
    path = tmp_path / "report.json"
    q.write_json(path, {"original": True})
    with pytest.raises(FileExistsError):
        q.write_json(path, {"changed": True})
    assert q.strict_json(path) == {"original": True}


def child(tmp_path, code, *, seconds=2, max_bytes=4096, cancel=None):
    return q.run_command(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env=q.clean_env(tmp_path, Path(sys.executable)),
        log=tmp_path / "child.log",
        deadline=time.monotonic() + seconds,
        timeout=seconds,
        max_bytes=max_bytes,
        cancel=cancel,
    )


def test_process_both_pipes_drained_and_output_bounded(tmp_path):
    result = child(
        tmp_path,
        'import os\nwhile True:\n os.write(1,b"a"*8192)\n os.write(2,b"b"*8192)',
    )
    assert result["status"] == "output_limit"
    assert result["log_truncated"] and (tmp_path / "child.log").stat().st_size == 4096


def test_process_nonzero_has_no_automatic_retry(tmp_path):
    result = child(tmp_path, 'import sys; print("once"); sys.exit(7)')
    assert result["status"] == "failed" and result["exit_code"] == 7
    assert (tmp_path / "child.log").read_text() == "once\n"


def test_process_timeout_reaps_leader(tmp_path):
    result = child(tmp_path, "import time; time.sleep(60)", seconds=0.2)
    assert result["status"] == "timeout" and result["exit_code"] == -signal.SIGKILL
    assert result["elapsed_seconds"] < 2


def test_process_cancellation(tmp_path):
    cancel = threading.Event()
    timer = threading.Timer(0.1, cancel.set)
    timer.start()
    try:
        result = child(tmp_path, "import time; time.sleep(60)", cancel=cancel)
    finally:
        timer.join()
    assert result["status"] == "cancelled"


def test_exited_parent_with_grandchild_pipes_times_out(tmp_path):
    result = child(
        tmp_path,
        'import subprocess,sys; subprocess.Popen([sys.executable,"-I","-c","import time; time.sleep(60)"])',
        seconds=0.2,
    )
    assert result["status"] == "timeout" and result["elapsed_seconds"] < 2


def test_deadline_exhausted_does_not_launch(tmp_path):
    result = q.run_command(
        [sys.executable, "-c", "raise Exception()"],
        cwd=tmp_path,
        env={},
        log=tmp_path / "child.log",
        deadline=time.monotonic() - 1,
    )
    assert result["status"] == "timeout" and result["exit_code"] is None


def test_environment_excludes_ambient_keys_and_pythonpath(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sentinel")
    monkeypatch.setenv("PYTHONPATH", "/bad")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--bad")
    monkeypatch.setenv("PIP_INDEX_URL", "https://secret")
    result = child(tmp_path, "import os,json; print(json.dumps(dict(os.environ)))")
    assert result["status"] == "passed"
    env = json.loads((tmp_path / "child.log").read_text())
    assert (
        not {"OPENAI_API_KEY", "PYTHONPATH", "PYTEST_ADDOPTS", "PIP_INDEX_URL"}
        & env.keys()
    )


def test_verification_import_never_executes_argv(bundle, monkeypatch):
    bundle[2]["commands"][0]["argv"] = ["sh", "-c", "arbitrary effects"]
    monkeypatch.setattr(
        q.subprocess, "Popen", lambda *a, **k: pytest.fail("executed evidence")
    )
    verdict(bundle)


def test_installed_smoke_missing_checks_is_not_a_pass(bundle):
    root = bundle[0]
    smoke = q.strict_json(root / "installed-results.json")
    smoke["checks"].pop("missing_adapter_capability")
    save(root / "installed-results.json", smoke)
    refresh(bundle)
    with pytest.raises(q.EvidenceError, match="invalid_smoke_checks"):
        verdict(bundle)


def test_installed_smoke_failure_blocks_generic_pytest_success(bundle):
    root = bundle[0]
    smoke = q.strict_json(root / "installed-results.json")
    smoke["checks"]["missing_adapter_capability"] = "not_run"
    save(root / "installed-results.json", smoke)
    refresh(bundle)
    assert not verdict(bundle)["checks_only_complete"]


def test_real_pytest_plugin_records_exact_nodes_and_teardown(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    (tmp_path / "test_sample.py").write_text("def test_one():\n    assert 2 + 2 == 4\n")
    output = tmp_path / "pytest-results.json"
    context = tmp_path / "context.json"
    q.write_json(
        context,
        {"repo": str(tmp_path), "output": str(output), "candidate": {}, "kinds": {}},
    )
    env = q.clean_env(tmp_path, Path(sys.executable), repo)
    env["JITMIND_QUALIFICATION_CONTEXT"] = str(context)
    result = q.run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "scripts.qualify_jitmind",
            "test_sample.py",
        ],
        cwd=tmp_path,
        env=env,
        log=tmp_path / "pytest.log",
        deadline=time.monotonic() + 10,
    )
    assert result["status"] == "passed", (tmp_path / "pytest.log").read_text()
    record = q.strict_json(output)
    assert record["collected"] == ["test_sample.py::test_one"]
    assert record["collection_count"] == 1
    assert record["records"][0]["phases"] == dict.fromkeys(
        ["setup", "call", "teardown"], "passed"
    )
    assert record["records"][0]["kind"] == "unmapped"


def test_shipped_map_preserves_32_cases_41_stages_and_not_run():
    path = Path(__file__).resolve().parents[1] / "docs/engineering/acceptance-map.json"
    mapping = q.strict_json(path)
    cases = q.validate_map(mapping)
    assert len(cases) == 32 and sum(len(c["stages"]) for c in cases) == 41
    assert all(c["status"] == "not_run" for c in cases)
    assert [x[-1] for x in mapping["temporal_oracle"]["answers"]] == [
        3,
        3,
        4,
        3,
        4,
        5,
        3,
    ]


def test_missing_test_artifact_measurements_are_null(bundle):
    bundle[3].update(
        available=False, collected=[], collection_count=None, records=[], exit_code=99
    )
    refresh(bundle)
    result = verdict(bundle)
    assert result["collection_count"] is None
    assert result["test_counts"] == dict.fromkeys(
        ["passed", "failed", "skipped", "unknown"]
    )
    assert not result["checks_only_complete"]


def test_tool_tests_cannot_be_mapped_as_product_acceptance(bundle):
    bundle[1]["cases"][0]["stages"][0]["nodes"][0]["node_id"] = (
        "tests/test_qualification.py::test_fake"
    )
    refresh(bundle)
    with pytest.raises(q.EvidenceError, match="tool_test_is_not_acceptance"):
        verdict(bundle)


def test_unknown_external_finding_is_retained_without_echoing_payload(bundle):
    bundle[2]["findings"] = ["private-path API_KEY=negative-sentinel"]
    result = verdict(bundle)
    assert result["findings"] == ["retained_external_finding"]
    assert "negative-sentinel" not in json.dumps(result)


def test_timeout_kills_owned_grandchild(tmp_path):
    import os

    pid_path = tmp_path / "grandchild.pid"
    script = (
        "import pathlib,subprocess,sys,time; "
        'p=subprocess.Popen([sys.executable,"-I","-c","import time; time.sleep(60)"]); '
        f"pathlib.Path({str(pid_path)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    result = child(tmp_path, script, seconds=1)
    assert result["status"] == "timeout"
    pid = int(pid_path.read_text())
    until = time.monotonic() + 2
    while time.monotonic() < until:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        linux_state = Path(f"/proc/{pid}/stat")
        if linux_state.exists() and linux_state.read_text().split()[2] == "Z":
            return  # direct child is reaped; orphan zombie belongs to OS init
        time.sleep(0.01)
    pytest.fail("owned grandchild remained live after timeout")


@pytest.mark.parametrize("target", ["command_exit", "artifact_bytes"])
def test_evidence_boolean_numeric_fields_refused(bundle, target):
    if target == "command_exit":
        bundle[2]["commands"][0]["exit_code"] = False
    else:
        bundle[2]["artifacts"][0]["bytes"] = True
    with pytest.raises(q.EvidenceError):
        verdict(bundle)


def test_json_symlinks_and_special_files_refused(tmp_path):
    import os

    path = tmp_path / "fifo"
    os.mkfifo(path)
    with pytest.raises(q.EvidenceError):
        q.strict_json(path)
    target = tmp_path / "real.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(q.EvidenceError):
        q.strict_json(link)


def test_hashing_observes_exhausted_deadline(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"source bytes")
    with pytest.raises(q.EvidenceError, match="overall_deadline"):
        q.digest(path, deadline=time.monotonic() - 1)

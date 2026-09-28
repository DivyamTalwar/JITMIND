#!/usr/bin/env python3
"""Offline, fail-closed candidate checks. Imported evidence never executes commands.

This module also supplies a pytest plugin recording exact node IDs and all phases.
Only the CLI's fixed command constructors execute tools; map data is never argv.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import selectors
import signal
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

MAX_JSON = 8 * 1024 * 1024
MAX_LOG = 2 * 1024 * 1024
MAX_ARTIFACT = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_FILES = 10000
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
KINDS = {
    "real_storage",
    "injected_storage_fault",
    "real_parser",
    "public_api",
    "migration",
    "projection",
    "installed_package",
    "temporal_oracle",
}
COMMANDS = {"pytest", "compile", "wheel", "venv", "install", "smoke"}
SMOKE_NODE = "tests/test_standalone_install.py::installed_smoke"
SMOKE_CHECKS = {
    "isolated_origin",
    "optional_dependencies_absent",
    "legacy_memory_research",
    "optional_adapter_import",
    "missing_adapter_capability",
}
RELEASE_BLOCKERS = [
    "independent_signoff_not_recorded",
    "platform_matrix_not_run",
    "provider_checks_not_run",
    "power_loss_drill_not_run",
    "benchmark_comparisons_not_run",
]


class EvidenceError(ValueError):
    """Messages are fixed reason codes, never input payloads or paths."""


def require(condition: Any, code: str) -> None:
    if not condition:
        raise EvidenceError(code)


def strict_json(path: Path, limit: int = MAX_JSON) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def constant(_value):
        raise EvidenceError("nonfinite_json")

    try:
        require(path.is_file() and not path.is_symlink(), "invalid_json_artifact")
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        require(len(raw) <= limit, "input_too_large")
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)

        # 1e999 is parsed as infinity without invoking parse_constant.
        def finite(item):
            if isinstance(item, float):
                require(math.isfinite(item), "nonfinite_json")
            elif isinstance(item, dict):
                for val in item.values():
                    finite(val)
            elif isinstance(item, list):
                for val in item:
                    finite(val)

        finite(value)
        return value
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise EvidenceError("invalid_json_artifact") from None


def digest(path: Path, *, deadline: float | None = None) -> str:
    require(path.is_file() and not path.is_symlink(), "artifact_missing_or_symlink")
    require(path.stat().st_size <= MAX_ARTIFACT, "artifact_too_large")
    result = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            if deadline is not None:
                require(time.monotonic() < deadline, "overall_deadline")
            total += len(block)
            require(total <= MAX_ARTIFACT, "artifact_too_large")
            result.update(block)
    return result.hexdigest()


def bounded_copy(
    source: Path,
    destination: Path,
    *,
    expected_hash: str | None = None,
    deadline: float | None = None,
) -> str:
    """Exclusive, bounded copy of a stable regular file; remove partial output.

    Pin the opened inode and reject replacement or metadata changes during copying.
    Hash checks bind bytes when copying a previously inventoried source.
    """

    def identity(info):
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    before = source.lstat()
    require(stat.S_ISREG(before.st_mode), "artifact_missing_or_symlink")
    require(before.st_size <= MAX_ARTIFACT, "artifact_too_large")
    created = False
    try:
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb", buffering=0) as src:
            opened = os.fstat(src.fileno())
            require(identity(opened) == identity(before), "source_changed_during_copy")
            total = 0
            hasher = hashlib.sha256()
            with destination.open("xb") as dst:
                created = True
                while True:
                    if deadline is not None:
                        require(time.monotonic() < deadline, "overall_deadline")
                    block = src.read(min(65536, MAX_ARTIFACT - total + 1))
                    if not block:
                        break
                    total += len(block)
                    require(total <= MAX_ARTIFACT, "artifact_too_large")
                    hasher.update(block)
                    dst.write(block)
                require(
                    identity(os.fstat(src.fileno())) == identity(opened)
                    and identity(source.lstat()) == identity(opened),
                    "source_changed_during_copy",
                )
                require(total == opened.st_size, "source_changed_during_copy")
                result = hasher.hexdigest()
                require(
                    expected_hash is None or result == expected_hash,
                    "source_hash_mismatch",
                )
        return result
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise


@contextmanager
def command_signals(cancel: threading.Event):
    """Temporarily translate SIGTERM on the main thread; never change worker signals."""
    installed = threading.current_thread() is threading.main_thread()
    if installed:
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda _signum, _frame: cancel.set())
    try:
        yield
    finally:
        if installed:
            signal.signal(signal.SIGTERM, previous)


def relative_file(root: Path, name: str) -> Path:
    require(
        isinstance(name, str) and bool(name) and "\\" not in name,
        "invalid_artifact_path",
    )
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "invalid_artifact_path")
    current = root.resolve()
    for part in path.parts:
        current = current / part
        require(not current.is_symlink(), "artifact_symlink")
    require(current.is_relative_to(root.resolve()), "artifact_escape")
    return current


def artifact(root: Path, name: str) -> dict:
    path = relative_file(root, name)
    return {"path": name, "sha256": digest(path), "bytes": path.stat().st_size}


def verify_artifact(root: Path, ref: dict) -> Path:
    require(isinstance(ref, dict), "invalid_artifact")
    require(
        isinstance(ref.get("sha256"), str) and SHA256.fullmatch(ref["sha256"]),
        "invalid_artifact_digest",
    )
    require(
        type(ref.get("bytes")) is int and 0 <= ref["bytes"] <= MAX_ARTIFACT,
        "invalid_artifact_size",
    )
    path = relative_file(root, ref.get("path"))
    require(
        digest(path) == ref["sha256"] and path.stat().st_size == ref["bytes"],
        "artifact_modified",
    )
    return path


def write_json(path: Path, value: Any) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    require(len(data.encode()) <= MAX_JSON, "report_too_large")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(data)


def clean_env(home: Path, python: Path, checkout: Path | None = None) -> dict:
    """Allowlist: do not inherit credentials, proxy config, plugins, or PYTHONPATH."""
    env = {
        "HOME": str(home),
        "TMPDIR": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": str(python.parent) + os.pathsep + "/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    if checkout is not None:
        env["PYTHONPATH"] = str(checkout)
    return env


def run_command(
    argv: list[str],
    *,
    cwd: Path,
    env: dict,
    log: Path,
    deadline: float,
    timeout: float = 120,
    max_bytes: int = MAX_LOG,
    cancel: threading.Event | None = None,
) -> dict:
    """Trusted internal argv only. Drain both pipes; bound storage before capture.

    POSIX owned process groups include children inheriting pipes after leader exit.
    Kill the group even on successful leader exit; never retry a command.
    """
    require(os.name == "posix", "process_groups_unsupported_platform")
    require(
        isinstance(argv, list)
        and bool(argv)
        and all(isinstance(arg, str) for arg in argv),
        "invalid_argv",
    )
    require(type(max_bytes) is int and max_bytes > 0, "invalid_log_budget")
    require(
        type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0,
        "invalid_timeout",
    )
    require(
        type(deadline) in (int, float) and math.isfinite(deadline), "invalid_deadline"
    )
    started = time.monotonic()
    end = min(deadline, started + timeout)
    outcome = "not_run"
    emitted = 0
    proc = None
    cancel = cancel if cancel is not None else threading.Event()
    with command_signals(cancel), log.open("xb") as output:
        try:
            if cancel is not None and cancel.is_set():
                outcome = "cancelled"
            elif started >= end:
                outcome = "timeout"
            else:
                proc = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=env,
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                outcome = "passed"
                with selectors.DefaultSelector() as selector:
                    for pipe in (proc.stdout, proc.stderr):
                        os.set_blocking(pipe.fileno(), False)
                        selector.register(pipe, selectors.EVENT_READ)
                    while selector.get_map() or proc.poll() is None:
                        if cancel is not None and cancel.is_set():
                            outcome = "cancelled"
                            break
                        if time.monotonic() >= end:
                            outcome = "timeout"
                            break
                        for key, _ in selector.select(
                            min(0.02, max(0, end - time.monotonic()))
                        ):
                            chunk = os.read(key.fileobj.fileno(), 16384)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            room = max_bytes - emitted
                            output.write(chunk[:room])
                            emitted += min(room, len(chunk))
                            if len(chunk) > room:
                                outcome = "output_limit"
                                break
                        if outcome != "passed":
                            break
                if outcome == "passed" and proc.poll() != 0:
                    outcome = "failed"
        except KeyboardInterrupt:
            outcome = "cancelled"
        except OSError:
            outcome = "launch_error"
        finally:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                proc.stdout.close()
                proc.stderr.close()
    return {
        "status": outcome,
        "exit_code": None if proc is None else proc.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "log_bytes": emitted,
        "log_truncated": outcome == "output_limit",
    }


def candidate_identity(
    repo: Path, scratch: Path, python: Path, deadline: float
) -> dict:
    def git(name, args):
        log = scratch / (name + ".log")
        result = run_command(
            ["/usr/bin/git", "-c", "core.fsmonitor=false", *args],
            cwd=repo,
            env=clean_env(scratch, python),
            log=log,
            deadline=deadline,
            timeout=20,
            max_bytes=MAX_JSON,
        )
        require(result["status"] == "passed", "git_identity_unavailable")
        return log.read_bytes()

    sha = git("head", ["rev-parse", "HEAD"]).decode().strip()
    require(GIT_SHA.fullmatch(sha), "invalid_candidate_sha")
    # --porcelain suppresses warning output only on standard installations; fail closed otherwise.
    status = git("status", ["status", "--porcelain=v1", "-z", "--untracked-files=all"])
    paths = git(
        "files", ["ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    )
    records = []
    source_names = sorted(set(paths.split(b"\0")) - {b""})
    require(len(source_names) <= MAX_FILES, "too_many_source_files")
    source_bytes = 0
    for raw in source_names:
        require(time.monotonic() < deadline, "overall_deadline")
        name = raw.decode("utf-8")
        path = relative_file(repo, name)
        if path.exists():
            source_bytes += path.stat().st_size
            require(source_bytes <= MAX_TOTAL_BYTES, "source_snapshot_too_large")
        records.append(
            [name, digest(path, deadline=deadline) if path.exists() else None]
        )
    snapshot = hashlib.sha256(
        json.dumps(records, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "git_sha": sha,
        "dirty": bool(status),
        "snapshot_sha256": snapshot,
        "snapshot_files": records,
    }


def snapshot_roster(identity: dict) -> dict:
    records = identity.get("snapshot_files")
    require(
        isinstance(records, list) and bool(records) and len(records) <= MAX_FILES,
        "invalid_snapshot_roster",
    )
    names = []
    for record in records:
        require(isinstance(record, list) and len(record) == 2, "invalid_snapshot_entry")
        name, sha = record
        require(
            isinstance(name, str)
            and bool(name)
            and "\\" not in name
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and Path(name).as_posix() == name
            and name != ".",
            "invalid_snapshot_path",
        )
        require(
            sha is None or isinstance(sha, str) and SHA256.fullmatch(sha),
            "invalid_snapshot_source_digest",
        )
        names.append(name)
    require(names == sorted(set(names)), "snapshot_roster_not_normalized")
    expected = hashlib.sha256(
        json.dumps(records, separators=(",", ":")).encode()
    ).hexdigest()
    require(identity["snapshot_sha256"] == expected, "snapshot_digest_mismatch")
    return dict(records)


def validate_map(mapping: dict) -> list[dict]:
    require(
        isinstance(mapping, dict)
        and mapping.get("schema") == "jitmind.acceptance-map/v1",
        "invalid_map_schema",
    )
    require(
        isinstance(mapping.get("source_handoff_sha256"), str)
        and SHA256.fullmatch(mapping["source_handoff_sha256"]),
        "invalid_handoff_digest",
    )
    cases = mapping.get("cases")
    require(isinstance(cases, list) and bool(cases), "empty_cases")
    ids = set()
    for case in cases:
        require(
            isinstance(case, dict)
            and isinstance(case.get("case_id"), str)
            and re.fullmatch(r"J[0-9]{2}", case["case_id"]),
            "invalid_case",
        )
        require(case["case_id"] not in ids, "duplicate_case")
        ids.add(case["case_id"])
        required = case.get("required_in_tickets")
        require(
            isinstance(required, list)
            and required
            and all(
                isinstance(s, str) and re.fullmatch(r"J-[0-9]{2}", s) for s in required
            )
            and len(set(required)) == len(required),
            "invalid_required_stages",
        )
        require(
            case.get("ticket") in required and case.get("status") == "not_run",
            "invalid_case_owner_or_status",
        )
        stages = case.get("stages")
        require(
            isinstance(stages, list) and len(stages) == len(required), "missing_stage"
        )
        require({s.get("stage_id") for s in stages} == set(required), "missing_stage")
        for stage in stages:
            require(stage.get("status") == "not_run", "map_is_not_execution_evidence")
            require(
                type(stage.get("additional_mapping_required", False)) is bool,
                "invalid_mapping_gap",
            )
            nodes = stage.get("nodes")
            require(isinstance(nodes, list), "invalid_nodes")
            seen = set()
            for node in nodes:
                node_id = node.get("node_id")
                require(
                    isinstance(node_id, str) and node_id and node_id not in seen,
                    "duplicate_or_invalid_node",
                )
                require(
                    node_id.startswith("tests/")
                    and ".." not in node_id.split("::")[0].split("/"),
                    "invalid_test_path",
                )
                require(
                    not node_id.startswith("tests/test_qualification")
                    and (
                        not node_id.startswith("tests/test_standalone_install.py::")
                        or node_id == SMOKE_NODE
                    ),
                    "tool_test_is_not_acceptance",
                )
                seen.add(node_id)
                require(node.get("kind") in KINDS, "unsupported_evidence_kind")
                require(
                    isinstance(node.get("source"), dict) and node.get("fixtures"),
                    "missing_source_or_fixture",
                )
                for ref in [node["source"], *node["fixtures"]]:
                    require(
                        isinstance(ref, dict)
                        and isinstance(ref.get("path"), str)
                        and isinstance(ref.get("sha256"), str)
                        and SHA256.fullmatch(ref["sha256"]),
                        "invalid_source_ref",
                    )
    return cases


def evaluate(mapping: dict, report: dict, root: Path) -> dict:
    """Validate evidence bytes and recompute verdict. No command execution or network.

    A digest verifies consistency, not authorship. External input is never signoff.
    Structural errors raise; incomplete/failed checks remain in the denominator.
    """
    cases = validate_map(mapping)
    require(report.get("schema") == "jitmind.qualification/v2", "invalid_report_schema")
    identity = report.get("candidate", {})
    require(
        isinstance(identity.get("git_sha"), str)
        and GIT_SHA.fullmatch(identity["git_sha"]),
        "invalid_candidate_sha",
    )
    require(
        isinstance(identity.get("snapshot_sha256"), str)
        and SHA256.fullmatch(identity["snapshot_sha256"]),
        "invalid_snapshot_digest",
    )
    roster = snapshot_roster(identity)
    require(type(identity.get("dirty")) is bool, "invalid_dirty_flag")
    require(report.get("mode") in {"worktree", "release"}, "invalid_run_mode")
    require(not identity["dirty"] or report["mode"] == "worktree", "dirty_release")
    map_path = verify_artifact(root, report.get("acceptance_map"))
    require(strict_json(map_path) == mapping, "map_identity_mismatch")
    artifacts = report.get("artifacts")
    require(isinstance(artifacts, list), "invalid_artifacts")
    require(len(artifacts) <= MAX_FILES, "too_many_artifacts")
    refs = {}
    artifact_bytes = 0
    for ref in artifacts:
        require(
            isinstance(ref, dict)
            and type(ref.get("bytes")) is int
            and ref["bytes"] >= 0,
            "invalid_artifact_size",
        )
        artifact_bytes += ref["bytes"]
        require(artifact_bytes <= MAX_TOTAL_BYTES, "artifacts_too_large")
        verify_artifact(root, ref)
        require(ref["path"] not in refs, "duplicate_artifact")
        refs[ref["path"]] = ref
    for name, sha in roster.items():
        ref = refs.get("sources/" + name)
        require(
            (ref is None if sha is None else ref is not None and ref["sha256"] == sha),
            "snapshot_source_mismatch",
        )
    require(
        all(not name.startswith("sources/") or name[8:] in roster for name in refs),
        "source_outside_snapshot",
    )
    commands = report.get("commands")
    require(isinstance(commands, list), "missing_commands")
    command_ids = set()
    command_ok = bool(commands)
    findings = report.get("findings", [])
    require(
        isinstance(findings, list) and all(isinstance(f, str) for f in findings),
        "invalid_findings",
    )
    # Preserve the existence of imported findings without echoing arbitrary payloads.
    safe_findings = {
        "candidate_wheel_missing",
        "candidate_changed_during_checks",
        "candidate_final_identity_unavailable",
    }
    findings = [
        f if f in safe_findings else "retained_external_finding" for f in findings
    ]
    for cmd in commands:
        require(
            cmd.get("id") in COMMANDS and cmd["id"] not in command_ids,
            "invalid_command_id",
        )
        command_ids.add(cmd["id"])
        require(
            type(cmd.get("exit_code")) is int or cmd.get("exit_code") is None,
            "invalid_exit_code",
        )
        require(type(cmd.get("log_truncated")) is bool, "invalid_truncation_flag")
        require(cmd.get("log") in refs, "missing_command_log")
        command_ok &= (
            cmd.get("status") == "passed"
            and cmd.get("exit_code") == 0
            and not cmd["log_truncated"]
        )
    command_ok &= command_ids == COMMANDS
    require(report.get("test_results") in refs, "missing_test_results")
    test_data = strict_json(root / report["test_results"])
    require(test_data.get("schema") == "jitmind.pytest/v2", "invalid_test_schema")
    require(test_data.get("candidate") == identity, "test_source_identity_mismatch")
    expected_name = report.get("expected_collection")
    require(expected_name in refs, "missing_expected_collection")
    expected = strict_json(root / expected_name)
    require(
        expected.get("schema") == "jitmind.collection/v1"
        and expected.get("candidate") == identity,
        "collection_identity_mismatch",
    )
    require(
        test_data.get("expected_collection_sha256") == refs[expected_name]["sha256"],
        "collection_digest_mismatch",
    )
    for field in ("nodes", "selected", "deselected"):
        values = expected.get(field)
        require(
            isinstance(values, list)
            and all(isinstance(n, str) for n in values)
            and len(values) == len(set(values)),
            "duplicate_or_invalid_collection",
        )
    require(type(expected.get("filtered")) is bool, "invalid_collection_filter")
    collection_reports = expected.get("collection_reports")
    require(
        isinstance(collection_reports, list)
        and all(
            isinstance(r, dict)
            and isinstance(r.get("node_id"), str)
            and r.get("outcome") in {"failed", "skipped"}
            for r in collection_reports
        ),
        "invalid_collection_reports",
    )
    require(
        set(expected["selected"]).isdisjoint(expected["deselected"])
        and set(expected["nodes"])
        == set(expected["selected"]) | set(expected["deselected"]),
        "collection_inventory_mismatch",
    )
    collected = test_data.get("collected")
    require(
        isinstance(collected, list)
        and all(isinstance(n, str) for n in collected)
        and len(collected) == len(set(collected)),
        "duplicate_or_invalid_collection",
    )
    available = test_data.get("available", True)
    require(type(available) is bool, "invalid_test_availability")
    require(
        (
            available
            and type(test_data.get("collection_count")) is int
            and test_data["collection_count"] == len(collected)
        )
        or (
            not available
            and test_data.get("collection_count") is None
            and not collected
        ),
        "invalid_collection_count",
    )
    records = test_data.get("records")
    require(isinstance(records, list), "missing_test_records")
    by_node = {}
    counts = {"passed": 0, "failed": 0, "skipped": 0, "unknown": 0}
    for record in records:
        node_id = record.get("node_id")
        require(
            isinstance(node_id, str)
            and node_id in collected
            and node_id not in by_node,
            "duplicate_or_uncollected_test",
        )
        source_name = node_id.split("::")[0]
        require(
            source_name in roster
            and roster[source_name] is not None
            and record.get("source_sha256") == roster[source_name],
            "record_source_mismatch",
        )
        phases = record.get("phases")
        require(
            isinstance(phases, dict) and set(phases) <= {"setup", "call", "teardown"},
            "invalid_phases",
        )
        require(
            all(p in {"passed", "failed", "skipped"} for p in phases.values()),
            "invalid_phase_outcome",
        )
        status = (
            "failed"
            if "failed" in phases.values()
            else "skipped"
            if "skipped" in phases.values()
            else "passed"
            if phases == {"setup": "passed", "call": "passed", "teardown": "passed"}
            else "unknown"
        )
        require(record.get("status") == status, "false_test_pass")
        counts[status] += 1
        by_node[node_id] = record
    counts["unknown"] += len((set(collected) | set(expected["nodes"])) - set(by_node))
    phase_counts = {
        outcome: sum(p == outcome for r in records for p in r["phases"].values())
        for outcome in ("passed", "failed", "skipped")
    }
    if available:
        counters = test_data.get("phase_counts")
        require(
            isinstance(counters, dict)
            and set(counters) == set(phase_counts)
            and all(type(v) is int for v in counters.values())
            and counters == phase_counts,
            "inconsistent_plugin_counters",
        )
        require(
            type(test_data.get("session_testscollected")) is int
            and test_data["session_testscollected"] == len(collected)
            and type(test_data.get("session_testsfailed")) is int
            and test_data["session_testsfailed"]
            == phase_counts["failed"]
            + sum(r["outcome"] == "failed" for r in collection_reports),
            "inconsistent_plugin_counters",
        )
    else:
        require(
            all(
                test_data.get(k) is None
                for k in (
                    "phase_counts",
                    "session_testscollected",
                    "session_testsfailed",
                )
            ),
            "inconsistent_plugin_counters",
        )
    tests_ok = (
        available
        and collected == expected["selected"]
        and set(collected) == set(expected["nodes"])
        and not expected["filtered"]
        and not expected["deselected"]
        and not collection_reports
        and bool(collected)
        and counts["passed"] == len(collected)
        and test_data.get("exit_code") == 0
    )
    require(type(test_data.get("exit_code")) is int, "invalid_pytest_exit")
    if not tests_ok:
        findings.append("pytest_incomplete_or_failed")
    if not command_ok:
        findings.append("fixed_suite_incomplete_or_failed")
    package = report.get("package_sha256")
    package_ok = (
        isinstance(package, str)
        and SHA256.fullmatch(package)
        and any(
            name.endswith(".whl") and ref["sha256"] == package
            for name, ref in refs.items()
        )
    )
    if not package_ok:
        findings.append("candidate_wheel_unverified")
    installed = None
    if "installed-results.json" in refs:
        installed = strict_json(root / "installed-results.json")
        require(
            installed.get("schema") == "jitmind.installed-smoke/v1",
            "invalid_smoke_schema",
        )
        checks = installed.get("checks")
        require(
            isinstance(checks, dict) and set(checks) == SMOKE_CHECKS,
            "invalid_smoke_checks",
        )
        require(installed.get("node_id") == SMOKE_NODE, "invalid_smoke_identity")
        require(
            type(installed.get("provider_calls")) is int
            and installed["provider_calls"] == 0,
            "invalid_smoke_provider_count",
        )
        smoke_ok = (
            package_ok
            and installed.get("status") == "passed"
            and all(s == "passed" for s in checks.values())
        )
        by_node[SMOKE_NODE] = {
            "kind": "installed_package",
            "source_sha256": installed.get("source_sha256"),
            "status": "passed" if smoke_ok else "failed",
        }
    if installed is None or by_node[SMOKE_NODE]["status"] != "passed":
        findings.append("installed_smoke_incomplete")
    require(report.get("benchmark_measurement") is None, "unreviewed_benchmark_claim")
    outcomes = []
    for case in cases:
        stages = []
        for stage in case["stages"]:
            statuses = []
            for node in stage["nodes"]:
                source_ok = node["node_id"].split("::")[0] == node["source"]["path"]
                for ref in [node["source"], *node["fixtures"]]:
                    name = "sources/" + ref["path"]
                    source_ok &= name in refs and refs[name]["sha256"] == ref["sha256"]
                record = by_node.get(node["node_id"])
                if not source_ok:
                    status = "source_unverified"
                elif not record:
                    status = "not_run"
                elif record.get("kind") != node["kind"]:
                    status = "evidence_kind_mismatch"
                elif record.get("source_sha256") != node["source"]["sha256"]:
                    status = "source_unverified"
                else:
                    status = record["status"]
                statuses.append(status)
            stages.append(
                {
                    "stage_id": stage["stage_id"],
                    "node_results": statuses,
                    "status": "passed"
                    if statuses
                    and not stage.get("additional_mapping_required", False)
                    and all(s == "passed" for s in statuses)
                    else "inconclusive",
                }
            )
        outcomes.append(
            {
                "case_id": case["case_id"],
                "stages": stages,
                "status": "passed"
                if all(s["status"] == "passed" for s in stages)
                else "inconclusive",
            }
        )
    complete = (
        tests_ok
        and command_ok
        and not findings
        and all(c["status"] == "passed" for c in outcomes)
    )
    return {
        "checks_only_complete": bool(complete),
        "release_qualified": False,
        "evidence_authorship": "digest_consistency_only_not_signer_identity",
        "case_count": len(cases),
        "stage_count": sum(len(c["stages"]) for c in cases),
        "passed_cases": sum(c["status"] == "passed" for c in outcomes),
        "test_counts": counts if available else dict.fromkeys(counts),
        "collection_count": len(expected["nodes"]) if available else None,
        "selected_count": len(collected) if available else None,
        "cases": outcomes,
        "findings": findings,
        "release_blockers": RELEASE_BLOCKERS
        + (["worktree_candidate"] if report["mode"] == "worktree" else []),
        "benchmark_measurement": None,
    }


# Fixed pytest plugin. These hooks only activate in an authorized runner session.
_SESSION: dict = {}


def pytest_sessionstart(session):
    context = os.environ.get("JITMIND_QUALIFICATION_CONTEXT")
    if context:
        _SESSION.clear()
        _SESSION.update(strict_json(Path(context)))
        _SESSION["records"] = {}
        _SESSION["collected"] = []
        _SESSION["nodes"] = []
        _SESSION["deselected"] = []
        _SESSION["collection_reports"] = []
        _SESSION["expected_path"] = str(
            Path(_SESSION["output"]).with_name("expected-collection.json")
        )


def pytest_itemcollected(item):
    if _SESSION:
        _SESSION["nodes"].append(item.nodeid)


def pytest_deselected(items):
    if _SESSION:
        _SESSION["deselected"].extend(item.nodeid for item in items)


def pytest_collectreport(report):
    if _SESSION and report.outcome != "passed":
        _SESSION["collection_reports"].append(
            {"node_id": report.nodeid, "outcome": report.outcome}
        )


def collection_record(session):
    options = session.config.option
    filtered = any(
        getattr(options, name, None)
        for name in (
            "keyword",
            "markexpr",
            "deselect",
            "ignore",
            "ignore_glob",
            "lf",
            "ff",
            "collectonly",
        )
    )
    # Fixed runner invokes the full tests directory. Keep arbitrary diagnostic runs
    # observable but never credit a file/node-targeted invocation as the fixed suite.
    filtered |= session.config.args != ["tests"]
    return {
        "schema": "jitmind.collection/v1",
        "candidate": _SESSION["candidate"],
        "nodes": _SESSION["nodes"],
        "selected": _SESSION["collected"],
        "deselected": _SESSION["deselected"],
        "collection_reports": _SESSION["collection_reports"],
        "filtered": bool(filtered),
    }


def pytest_collection_finish(session):
    if _SESSION:
        _SESSION["collected"] = [item.nodeid for item in session.items]
        # Exclusive artifact captured before execution, independent of runtime records.
        write_json(Path(_SESSION["expected_path"]), collection_record(session))


def pytest_runtest_logreport(report):
    if _SESSION:
        phases = _SESSION["records"].setdefault(report.nodeid, {})
        require(report.when not in phases, "duplicate_test_phase")
        phases[report.when] = (
            "skipped" if hasattr(report, "wasxfail") else report.outcome
        )


def pytest_sessionfinish(session, exitstatus):
    if not _SESSION:
        return
    expected_path = Path(_SESSION["expected_path"])
    if not expected_path.exists():
        write_json(expected_path, collection_record(session))
    records = []
    for node, phases in _SESSION["records"].items():
        status = (
            "failed"
            if "failed" in phases.values()
            else "skipped"
            if "skipped" in phases.values()
            else "passed"
            if phases == {"setup": "passed", "call": "passed", "teardown": "passed"}
            else "unknown"
        )
        source = relative_file(Path(_SESSION["repo"]), node.split("::")[0])
        records.append(
            {
                "node_id": node,
                "phases": phases,
                "status": status,
                "source_sha256": digest(source),
                "kind": _SESSION["kinds"].get(node, "unmapped"),
            }
        )
    write_json(
        Path(_SESSION["output"]),
        {
            "schema": "jitmind.pytest/v2",
            "candidate": _SESSION["candidate"],
            "collected": _SESSION["collected"],
            "available": True,
            "collection_count": len(_SESSION["collected"]),
            "expected_collection_sha256": digest(expected_path),
            "session_testscollected": session.testscollected,
            "session_testsfailed": session.testsfailed,
            "phase_counts": {
                outcome: sum(
                    p == outcome for r in records for p in r["phases"].values()
                )
                for outcome in ("passed", "failed", "skipped")
            },
            "records": records,
            "exit_code": int(exitstatus),
        },
    )


def host_facts(home: Path) -> dict:
    with sqlite3.connect(home / "qualification-probe.sqlite") as conn:
        pragmas = {
            key: conn.execute("PRAGMA " + key).fetchone()[0]
            for key in ("journal_mode", "synchronous", "foreign_keys", "busy_timeout")
        }
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "sqlite": sqlite3.sqlite_version,
        "probe_connection_pragmas": pragmas,
        "pragma_scope": "fresh stdlib connection; not product storage settings",
        "dependencies": sorted(
            [
                [d.metadata["Name"], d.version]
                for d in importlib.metadata.distributions()
            ]
        ),
    }


def run_suite(
    repo: Path,
    output: Path,
    wheelhouse: Path | None,
    worktree: bool,
    overall: float,
    per_command: float,
    node_binary: Path | None = None,
) -> dict:
    require(os.name == "posix", "process_groups_unsupported_platform")
    require(
        math.isfinite(overall)
        and overall > 0
        and math.isfinite(per_command)
        and per_command > 0,
        "invalid_deadline",
    )
    repo = repo.resolve()
    output = output.absolute()
    require(
        not output.resolve().is_relative_to(repo), "output_must_be_outside_checkout"
    )
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    python = Path(sys.executable).absolute()
    deadline = time.monotonic() + overall
    with tempfile.TemporaryDirectory(prefix="jitmind-qualification-") as temporary:
        scratch = Path(temporary)
        identity = candidate_identity(repo, scratch, python, deadline)
        require(worktree or not identity["dirty"], "dirty_release")
        mapping_path = repo / "docs/engineering/acceptance-map.json"
        mapping = strict_json(mapping_path)
        cases = validate_map(mapping)
        write_json(output / "acceptance-map.json", mapping)
        report = {
            "schema": "jitmind.qualification/v2",
            "mode": "worktree" if worktree else "release",
            "candidate": identity,
            "host": host_facts(scratch),
            "findings": [],
            "commands": [],
            "artifacts": [],
            "test_results": "pytest-results.json",
            "expected_collection": "expected-collection.json",
            "acceptance_map": artifact(output, "acceptance-map.json"),
            "package_sha256": None,
            "suite_source": "trusted_fixed_argv_v1",
            "checkout_tests_pythonpath": "explicit_checkout_only",
        }
        build_source = scratch / "build-source"
        build_source.mkdir()
        for name, expected_hash in identity["snapshot_files"]:
            require(time.monotonic() < deadline, "overall_deadline")
            if expected_hash is None:
                continue
            source = relative_file(repo, name)
            require(
                digest(source, deadline=deadline) == expected_hash,
                "candidate_changed_during_snapshot",
            )
            destination = build_source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            bounded_copy(
                source, destination, expected_hash=expected_hash, deadline=deadline
            )
            require(
                digest(destination, deadline=deadline) == expected_hash,
                "candidate_changed_during_snapshot",
            )
        report["build_source"] = (
            "fresh_copy_of_pinned_snapshot_without_ignored_build_outputs"
        )
        report["node_binary_sha256"] = digest(node_binary) if node_binary else None
        kinds = {}
        for case in cases:
            for stage in case["stages"]:
                for node in stage["nodes"]:
                    previous = kinds.setdefault(node["node_id"], node["kind"])
                    require(previous == node["kind"], "conflicting_node_kind")
        # Every live snapshot entry is retained, including non-code fixtures/config.
        copied_bytes = 0
        for name, expected_hash in identity["snapshot_files"]:
            if expected_hash is None:
                continue
            src = relative_file(build_source, name)
            copied_bytes += src.stat().st_size
            require(copied_bytes <= MAX_TOTAL_BYTES, "source_snapshot_too_large")
            dest = output / "sources" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            bounded_copy(src, dest, expected_hash=expected_hash, deadline=deadline)
            report["artifacts"].append(artifact(output, "sources/" + name))
        context = scratch / "context.json"
        write_json(
            context,
            {
                "repo": str(repo),
                "output": str(output / "pytest-results.json"),
                "candidate": identity,
                "kinds": kinds,
            },
        )
        env = clean_env(scratch, python, repo)
        env["JITMIND_QUALIFICATION_CONTEXT"] = str(context)
        if node_binary is not None:
            require(
                node_binary.is_absolute() and node_binary.is_file(),
                "invalid_node_runtime",
            )
            env["JITMIND_TEST_NODE"] = str(node_binary.resolve())

        cancelled = threading.Event()

        def execute(name, argv, cwd, environment):
            record = run_command(
                argv,
                cwd=cwd,
                env=environment,
                log=output / (name + ".log"),
                deadline=deadline,
                timeout=per_command,
                cancel=cancelled,
            )
            if record["status"] == "cancelled":
                cancelled.set()
            # Exact argv may contain private local paths; report portable placeholders.
            record.update(
                {
                    "id": name,
                    "argv": [
                        arg.replace(str(repo), "<checkout>")
                        .replace(str(scratch), "<temporary>")
                        .replace(str(output), "<report>")
                        .replace(str(python), "<python>")
                        .replace(str(wheelhouse), "<wheelhouse>")
                        if wheelhouse
                        else arg.replace(str(repo), "<checkout>")
                        .replace(str(scratch), "<temporary>")
                        .replace(str(output), "<report>")
                        .replace(str(python), "<python>")
                        for arg in argv
                    ],
                    "log": name + ".log",
                }
            )
            report["commands"].append(record)
            report["artifacts"].append(artifact(output, name + ".log"))
            return record["status"] == "passed"

        execute(
            "pytest",
            [
                str(python),
                "-m",
                "pytest",
                "-q",
                "tests",
                "-p",
                "scripts.qualify_jitmind",
            ],
            repo,
            env,
        )
        if not (output / "expected-collection.json").exists():
            write_json(
                output / "expected-collection.json",
                {
                    "schema": "jitmind.collection/v1",
                    "candidate": identity,
                    "nodes": [],
                    "selected": [],
                    "deselected": [],
                    "collection_reports": [],
                    "filtered": True,
                },
            )
        report["artifacts"].append(artifact(output, "expected-collection.json"))
        if not (output / "pytest-results.json").exists():
            write_json(
                output / "pytest-results.json",
                {
                    "schema": "jitmind.pytest/v2",
                    "candidate": identity,
                    "collected": [],
                    "collection_count": None,
                    "expected_collection_sha256": digest(
                        output / "expected-collection.json"
                    ),
                    "session_testscollected": None,
                    "session_testsfailed": None,
                    "phase_counts": None,
                    "available": False,
                    "records": [],
                    "exit_code": 99,
                },
            )
        report["artifacts"].append(artifact(output, "pytest-results.json"))
        execute(
            "compile",
            [
                str(python),
                "-m",
                "compileall",
                "-q",
                "jitmind",
                "scripts/qualify_jitmind.py",
            ],
            repo,
            env,
        )
        wheels = scratch / "wheels"
        wheels.mkdir()
        build_env = clean_env(scratch, python)
        built = execute(
            "wheel",
            [
                str(python),
                "-m",
                "pip",
                "wheel",
                "--no-index",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheels),
                str(build_source),
            ],
            scratch,
            build_env,
        )
        available = list(wheels.glob("jitmind-*.whl"))
        if built and len(available) == 1:
            wheel = available[0]
            dest = output / wheel.name
            wheel_hash = bounded_copy(wheel, dest, deadline=deadline)
            report["artifacts"].append(artifact(output, wheel.name))
            report["package_sha256"] = wheel_hash
            venv = scratch / "installed"
            created = execute(
                "venv", [str(python), "-m", "venv", str(venv)], scratch, build_env
            )
            vpython = venv / "bin/python"
            install_env = clean_env(scratch, vpython)
            install_argv = [
                str(vpython),
                "-I",
                "-m",
                "pip",
                "install",
                "--no-index",
                "--only-binary=:all:",
            ]
            if wheelhouse is not None:
                install_argv += ["--find-links", str(wheelhouse.resolve())]
            if created and execute(
                "install", install_argv + [str(dest)], scratch, install_env
            ):
                install_env["PATH"] = str(vpython.parent)
                execute(
                    "smoke",
                    [
                        str(vpython),
                        "-I",
                        str(repo / "tests/test_standalone_install.py"),
                        "--installed-smoke",
                        str(venv),
                        str(output / "installed-results.json"),
                    ],
                    scratch,
                    install_env,
                )
                if (output / "installed-results.json").exists():
                    report["artifacts"].append(
                        artifact(output, "installed-results.json")
                    )
                    smoke = strict_json(output / "installed-results.json")
                    report["installed"] = smoke
        else:
            report["findings"].append("candidate_wheel_missing")
        # Preserve every suite slot even when an upstream prerequisite failed.
        for name in sorted(COMMANDS - {c["id"] for c in report["commands"]}):
            with (output / (name + ".log")).open("xb"):
                pass
            report["commands"].append(
                {
                    "id": name,
                    "status": "not_run",
                    "exit_code": None,
                    "elapsed_seconds": None,
                    "log_bytes": 0,
                    "log_truncated": False,
                    "log": name + ".log",
                    "argv": None,
                    "reason": "prerequisite_not_passed",
                }
            )
            report["artifacts"].append(artifact(output, name + ".log"))
        # Detect mutations during tests/build; build output is gitignored, source is not.
        final_scratch = scratch / "final"
        final_scratch.mkdir()
        try:
            after = candidate_identity(repo, final_scratch, python, deadline)
            if after != identity:
                report["findings"].append("candidate_changed_during_checks")
        except EvidenceError:
            report["findings"].append("candidate_final_identity_unavailable")
        write_json(output / "report.json", report)
        verdict = evaluate(mapping, report, output)
        write_json(output / "verdict.json", verdict)
        return verdict


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument(
        "--output", type=Path, required=True, help="new directory outside checkout"
    )
    run.add_argument(
        "--wheelhouse", type=Path, help="reviewed offline core dependency wheels"
    )
    run.add_argument(
        "--node-binary",
        type=Path,
        help="trusted provisioned Node executable for parser tests only",
    )
    run.add_argument(
        "--worktree", action="store_true", help="label dirty developer checks"
    )
    run.add_argument("--overall-seconds", type=float, default=900)
    run.add_argument("--command-seconds", type=float, default=180)
    verify = sub.add_parser(
        "verify", help="read-only validation; never execute evidence commands"
    )
    verify.add_argument("report_dir", type=Path)
    verify.add_argument(
        "--candidate-sha", required=True, help="expected reviewed Git SHA"
    )
    verify.add_argument(
        "--snapshot-sha",
        required=True,
        help="expected reviewed source snapshot SHA-256",
    )
    args = parser.parse_args(argv)
    try:
        if args.action == "run":
            verdict = run_suite(
                Path(__file__).resolve().parents[1],
                args.output,
                args.wheelhouse,
                args.worktree,
                args.overall_seconds,
                args.command_seconds,
                args.node_binary,
            )
        else:
            report = strict_json(args.report_dir / "report.json")
            require(
                report.get("candidate", {}).get("git_sha") == args.candidate_sha,
                "unexpected_candidate_sha",
            )
            require(
                report.get("candidate", {}).get("snapshot_sha256") == args.snapshot_sha,
                "unexpected_snapshot_sha",
            )
            # Expected mapping is trusted checkout configuration, not imported claims.
            mapping = strict_json(
                Path(__file__).resolve().parents[1]
                / "docs/engineering/acceptance-map.json"
            )
            verdict = evaluate(mapping, report, args.report_dir)
        print(json.dumps(verdict, sort_keys=True, allow_nan=False))
        return 0 if verdict["checks_only_complete"] else 1
    except (EvidenceError, OSError, KeyError, TypeError, AttributeError, ValueError):
        print(
            json.dumps(
                {
                    "checks_only_complete": False,
                    "release_qualified": False,
                    "error": "qualification_invalid_or_unavailable",
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

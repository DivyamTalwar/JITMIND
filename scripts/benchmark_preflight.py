"""Offline preflight measurements against disposable actual SQLite stores.

Run with PYTHONPATH=$PWD python scripts/benchmark_preflight.py --iterations 20.
JSON stdout contains raw samples. No pass/fail latency threshold is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from jitmind.code_memory.lesson_models import (
    DurableFactAuthority,
    LessonProjection,
    ProposedAction,
)
from jitmind.code_memory.preflight import PreflightPolicy, PreflightService
from jitmind.code_memory.work_storage import WorkDatabase
from jitmind.scope import ScopeAuthority
from jitmind.storage import IngestRequest, Proposal, SQLiteDurableStore


def percentile(samples, fraction):
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "jitmind").rglob("*.py")) + [Path(__file__).resolve()]
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": hashes}


def benchmark(iterations: int = 20) -> dict:
    if type(iterations) is not int or not 1 <= iterations <= 1000:
        raise ValueError("iterations must be 1..1000")
    samples = []
    with tempfile.TemporaryDirectory(
        prefix="jitmind-preflight-benchmark-"
    ) as temporary:
        root = Path(temporary)
        authority = ScopeAuthority()
        authority.grant("fixture-user", "fixture-ns", ["fixture-repo"])
        scope = authority.context("fixture-user", "fixture-ns", "fixture-request")
        facts = SQLiteDurableStore(root / "facts.sqlite")
        work = WorkDatabase(root / "work.sqlite")
        binding_checks = [0]

        class BindingAuthority:
            def is_current(self, scope, repo, lesson, budget):
                budget.remaining()
                binding_checks[0] += 1
                return True

        projection = LessonProjection(
            work,
            authority,
            DurableFactAuthority(facts, authority),
            binding_authority=BindingAuthority(),
        )
        target = ProposedAction("fixture-repo", "src/fixture.py", "fixture")
        # Fixed synthetic corpus; real J02 ingestion and J06 persistence.
        for index in range(4):
            body = f"Fixture rule {index}: check the input before editing."
            receipt = facts.ingest(
                IngestRequest.create("fixture-ns", f"fixture-{index}", body),
                lambda _, body=body: Proposal(
                    abstract=body,
                    header="Fixture",
                    decorated=body,
                    decision={"operation": "add", "t_observed": "2026-01-01T00:00:00Z"},
                ),
            )
            projection.import_observation(
                scope,
                target,
                lesson_id=f"fixture-{index}",
                fact_id=receipt.memory_id,
                version=1,
                source_revision=receipt.revision,
                body=body,
                source_verified=True,
                confidence=0.95,
                binding_id=f"fixture-binding-{index}",
                binding_revision=1,
            )
        services = {
            enabled: PreflightService(
                projection, measure=enabled, policy=PreflightPolicy(deadline_seconds=3)
            )
            for enabled in (False, True)
        }
        for iteration in range(iterations):
            # Alternate order to reduce systematic warm-cache/order bias.
            for enabled in (False, True) if iteration % 2 == 0 else (True, False):
                service = services[enabled]
                session = service.start_session(scope, target.repo_id)
                tasks = (
                    ("first", target, "first"),
                    ("replay", target, "first"),
                    ("remaining", target, "remaining"),
                    ("dedup", target, "dedup"),
                    ("empty", replace(target, file_path="src/absent.py"), "empty"),
                )
                for task_id, action, key in tasks:
                    before_checks = binding_checks[0]
                    start = time.perf_counter()
                    result = service.deliver(scope, session, action, delivery_key=key)
                    elapsed = time.perf_counter() - start
                    samples.append(
                        {
                            "iteration": iteration,
                            "task_id": task_id,
                            "measurement_enabled": enabled,
                            "elapsed_seconds": elapsed,
                            "outcome": result.state,
                            "reason": result.reason,
                            "candidate_count": result.candidate_count,
                            "binding_checks": binding_checks[0] - before_checks,
                            "metrics": result.metrics.to_dict()
                            if result.metrics
                            else None,
                        }
                    )
                service.end_session(scope, session)
    summary = {}
    for enabled in (False, True):
        group = [s for s in samples if s["measurement_enabled"] is enabled]
        durations = [s["elapsed_seconds"] for s in group]
        summary["enabled" if enabled else "disabled"] = {
            "sample_count": len(group),
            "p50_seconds": statistics.median(durations),
            "p95_seconds": percentile(durations, 0.95),
            "outcomes": dict(Counter(s["outcome"] for s in group)),
            "binding_checks": sum(s["binding_checks"] for s in group),
            "stage_calls": dict(
                Counter(
                    {
                        name: sum(s["metrics"]["stages"][name]["calls"] for s in group)
                        for name in group[0]["metrics"]["stages"]
                    }
                )
            )
            if enabled
            else None,
        }
    paired = {}
    for sample in samples:
        key = (sample["iteration"], sample["task_id"])
        paired.setdefault(key, {})[sample["measurement_enabled"]] = sample[
            "elapsed_seconds"
        ]
    differences = [pair[True] - pair[False] for pair in paired.values()]
    return {
        "schema_version": 1,
        "description": "Actual local measurements on a synthetic fixed corpus; no production latency claim",
        "environment": {
            "python": sys.version,
            "sqlite": sqlite3.sqlite_version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "clock": "perf_counter elapsed; preflight stages use monotonic",
        },
        "source_fingerprint": source_fingerprint(),
        "iterations": iterations,
        "corpus_lessons": 4,
        "deadline_seconds": 3,
        "summary": summary,
        "paired_overhead_seconds": {
            "p50": statistics.median(differences),
            "p95": percentile(differences, 0.95),
            "raw": differences,
        },
        "samples": samples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    arguments = parser.parse_args()
    print(
        json.dumps(
            benchmark(arguments.iterations), sort_keys=True, indent=2, allow_nan=False
        )
    )

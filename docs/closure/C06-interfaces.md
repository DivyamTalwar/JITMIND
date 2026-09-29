# C06 preflight measurements interface

Implementation contract for C08 and composed consumers (schema version 1).

- `PreflightService(projection, *, policy=None, approved_repo=None, measure=True,
  metrics_sink: BoundedTraceSink | None = None)` retains `deliver(...)` arguments.
- `PreflightResult.metrics: PreflightMetrics | None = None` is additive. Disabled
  measurement returns `None`; policy and the 0.15 second default are unchanged.
- `PreflightMetrics.to_dict()` / `.to_json()` export versioned content-free JSON:
  `schema_version`, `outcome`, `elapsed_seconds`, `clock_anomalies`,
  `span_overflows`, `export_status`, `stages`. Each fixed stage (`lookup`,
  `eligibility`, `ranking`, `rendering`, `dedup`) has `calls`, `completed`, `failed`,
  `accepted`, `rejected`, `elapsed_seconds`, `status` (`not_entered`, `measured`,
  `clock_invalid`, `measurement_invalid`). Durations are exclusive:
  parent spans pause during children, including repeated same-stage spans.
- Repeated checks aggregate; no individual spans, lesson/request/user identifiers,
  content, paths, exception text, author data or wall-clock timestamps are stored.
- `BoundedTraceSink(capacity=64).snapshot()` returns immutable completed traces;
  capacity is 1..1024, oldest traces evict, publication never waits for its lock.
  Only this concrete in-memory sink is accepted, no exporter callback is invoked
  in delivery. Hosts export snapshots outside the authorization/response path.
- Denied/conflicting/invalid/cancelled calls retain their exception semantics and
  can be observed through the optional sink. Returned traces report sink failure
  or contention safely; no telemetry result participates in eligibility.

Storage/schema migration: none. Existing receipt digests, session capabilities,
source identities, SQLite schemas and authority callbacks are unchanged. The
ContextVar recorder is private and request-local, with reset on every exit.

Residual gates: see [measurement evidence](../preflight-measurement.md) for executed
verification; C08 must benchmark the composed candidate independently. No production latency, donor comparison, parser
or platform qualification claim follows from instrumentation. J05/C05 anchor code
is outside this ticket.

Benchmark consumer: `scripts.benchmark_preflight.benchmark(iterations: int = 20)
-> dict`, accepting 1..1000 iterations, or CLI `--iterations N`. Output schema v1
has `environment`, `source_fingerprint` (aggregate SHA-256 and per-source hashes),
`iterations`, `corpus_lessons`, `deadline_seconds`, `summary` by enabled/disabled
mode, `paired_overhead_seconds` (p50/p95/raw), and `samples` (iteration, repeated
synthetic task_id, measurement_enabled, elapsed_seconds, outcome, reason,
candidate_count, binding_checks, metrics). Measurements run on disposable actual
J02/J06 stores. Final evidence and exact commands are in the measurement document.

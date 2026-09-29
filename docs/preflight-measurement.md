# Preflight stage measurements

C06 closes the separate measurement gap in the actual synchronous J06 preflight
path. It does not establish a production latency target, donor comparison, or ZIP
completion. [The interface note](closure/C06-interfaces.md) describes the versioned
consumer contract. Existing authorization, receipt identity, failure states,
selection order, session expiry and the **0.15 second default deadline** remain.
No storage migration, dependency or provider call is introduced.

## API and safe export

```python
from jitmind.code_memory.preflight import PreflightService
from jitmind.code_memory.telemetry import BoundedTraceSink

sink = BoundedTraceSink(capacity=64)
service = PreflightService(projection, metrics_sink=sink)
session = service.start_session(issued_scope, registered_repo)
result = service.deliver(issued_scope, session, proposed_action)
if result.metrics is not None:
    serialized_metrics = result.metrics.to_json()
# Host diagnostics can export sink.snapshot() outside the delivery path.
```

`measure=True` is the default. `measure=False` disables recording and returns the
backward-compatible `PreflightResult.metrics=None`. It also masks any outer
request's recorder during a nested call. Other result fields and the `deliver`
arguments are unchanged. Old positional construction of `PreflightResult` works.
Callers should serialize metrics through `to_dict()` / `to_json()` for the stable
schema rather than depend on the internal dataclass representation.

The sink is optional and accepts only the concrete `BoundedTraceSink`, not an
exporter callback. It retains immutable traces for raised exceptions as well as
ordinary results. Exceptions still propagate as before. There are no request,
lesson, fact, receipt or binding IDs in traces; request correlation is through
the returned result object. Concurrent exception traces in a shared sink are
intentionally anonymous, not mapped back to a principal. There is no `last_result`
or shared mutable active trace. A private `ContextVar` is set/reset in a `finally`
block for every call. Recorder/metadata arguments are not accepted from callers
and no measured value participates in eligibility or authorization.

Only fixed stage names, numeric aggregates and fixed status labels are exported.
No lesson/query text, paths, tokens, authors, exception messages, identities,
wall timestamps or callback arguments are captured. Durations/counts themselves
may disclose workload characteristics; hosts control diagnostic access. The sink
makes no network calls, invokes no host callbacks and performs no I/O during
publication. Therefore an exporter cannot mutate authority between the final
check and returned content. Hosts doing their own later export own its access and
retention policy; telemetry does not remove database logs or backups.

The sink retains 1..1024 traces (default 64), evicting oldest entries at capacity.
Publication uses a nonblocking lock. `export_status` is `not_configured`,
`published`, `contended`, or `failed`. Returned results retain measurements when
publication fails. A raised-exception trace can be lost when publication fails or
contends; delivery never waits or changes that exception to force observability.
Snapshots are immutable tuples; callers should release references when finished.
There is no durable telemetry recovery or hidden fallback exporter.

## Accounting and schema version 1

All durations use `time.monotonic()` elapsed seconds. This is elapsed process
observation time, including scheduling, SQLite and callback waits. It is **not
CPU time**, and is not a civil/wall-clock timestamp. Session expiry and temporal
eligibility continue using the original wall clock. Deadline accounting uses the
original separate `Budget`: initial session checks still precede budget creation,
150 ms remains the default, and instrumentation inside that budget consumes real
time. Recording does not grant extra time, replace deadlines or promise hard
preemption. A host callback must still obey its existing budget contract.

| Stage | Existing work measured |
| --- | --- |
| `lookup` | `LessonProjection.candidates`: DB acquisition, SQL population filtering/order/limit, hydration, decoding and projection scope checks, excluding nested current-authority work |
| `eligibility` | initial scope/session checks, optional approved-repo callback, policy eligibility, `_current` checks and `current_authority` fact/binding callbacks, including final connection/checks |
| `ranking` | in-memory candidate filtering/sorting, excluding nested policy eligibility |
| `rendering` | actual quoting, labels, UTF-8 truncation and payload construction |
| `dedup` | receipt connection/transaction, replay selection, delivery lookup/writes, commit/rollback and connection close, excluding nested rendering/eligibility |

Durations are **exclusive**. Entering a child span pauses the parent's accounting;
leaving resumes it. Repeated same-stage spans aggregate exclusive segments once,
including receipt replay and the final authority check. SQL predicates/order
inside acquisition remain charged to lookup; they are not rerun as artificial
eligibility/ranking passes. Repeated operation counts count invocations, not unique
lessons. No per-lesson or per-span log grows with the candidate population.

`elapsed_seconds` at the top level is inclusive time from recorder creation until
measurement finalization. It includes interstage validation/hashing/error handling,
so it can exceed the sum of stage durations. It excludes immutable result/trace
construction, sink publication and the final wrapper return. Benchmark outer
elapsed time includes those costs. Do not sum top-level and stage durations.

Every trace has these fields:

- `schema_version`: integer `1`.
- `outcome`: the returned state (`delivered`, `nothing_relevant`, `deferred`,
  `unavailable`) or raised `denied`, `conflict`, `invalid`, `error`, `cancelled`.
- `elapsed_seconds`: finite nonnegative duration or `null` when unknown.
- `clock_anomalies`, `span_overflows`: bounded nonnegative diagnostic counters.
- `export_status`: sink status described above.
- `stages`: exactly five entries. Each has `calls`, `completed`, `failed`,
  `accepted`, `rejected`, `elapsed_seconds`, `status`.

`completed` means an invocation returned normally; it does not mean a lesson was
eligible. Boolean-returning checks also increment `accepted` or `rejected`.
`failed` includes propagated exceptions/cancellation at that stage. Thus a failure
can increment both lookup and its nested eligibility counter, although their
elapsed durations remain exclusive. Counters saturate at 2^31-1.

Unexecuted stages have zero calls, `status="not_entered"` and a **null duration**.
Executed stages have `status="measured"`; zero duration is possible only from an
actual equal clock reading. A backward/nonfinite/throwing clock invalidates the
active affected stages (`clock_invalid`, null elapsed), and makes total elapsed
unknown. Valid measurements from unaffected stages are retained. Diagnostics do
not expose clock exception text. Unexpected nesting beyond 32 retained frames
increments `span_overflows`, marks measured stages `measurement_invalid`, and
makes their durations/total unknown; no unbounded list of spans is retained.
JSON encoding rejects NaN/infinity.

## Reproducible benchmark

```sh
PY=/Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python
PYTHONPATH=$PWD "$PY" scripts/benchmark_preflight.py --iterations 20 > /private/tmp/jitmind-C06-preflight-benchmark.json
```

`benchmark(iterations: int = 20) -> dict` in that script is also callable. Iterations
are limited to 1..1000. It ingests four fixed **synthetic** fixture lessons into real
J02/J06 SQLite databases, with a host binding callback, then alternates enabled
and disabled measurement order. Each mode repeats task IDs `first`, `replay`,
`remaining`, `dedup`, `empty` with fresh sessions. Both modes use the same fixture
corpus and a three-second functional policy to avoid conflating delivery
compatibility with a loaded-host deadline benchmark. No product logic is replaced.
This benchmark does not invoke Graft; parser integration is verified separately.

Versioned output contains Python/SQLite/platform/clock information, SHA-256 of
all candidate Python sources and the benchmark script (plus an aggregate hash),
raw outer elapsed/metrics/candidate/binding counts and outcomes, p50/p95 summaries,
and paired enabled-minus-disabled overhead samples. Median uses the ordinary
median; p95 is the nearest sample at rounded `(n-1)*0.95`. Overhead may be negative
because scheduling/filesystem noise is not controlled. There is no performance
pass threshold, donor superiority claim or simulated latency evidence. Results
are local observations on synthetic inputs, not synthetic/invented timings.

## Executed verification (2026-09-29)

Runtime: isolated Python 3.12.12, SQLite 3.51.2, macOS arm64. All data came from
disposable fixtures and the actual candidate library. No live provider, credential,
production data, installation or manifest change was used. Existing donor notices
remain intact; the new telemetry is native code, with no copied donor module.

Commands from the assigned C06 worktree (`PY` as above):

```sh
PYTHONPATH=$PWD "$PY" -m pytest -q tests/test_preflight_metrics.py tests/test_preflight.py tests/test_preflight_review.py tests/test_preflight_python310.py

JITMIND_TEST_GRAFT_ADAPTER=/Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/delivery/J08/adapters/graft JITMIND_TEST_NODE=/opt/homebrew/bin/node PYTHONPATH=$PWD "$PY" -m pytest -q tests > /private/tmp/jitmind-C06-full-suite.txt 2>&1

PYTHONPATH=$PWD "$PY" -m ruff check jitmind/code_memory/preflight.py jitmind/code_memory/lesson_models.py jitmind/code_memory/telemetry.py tests/test_preflight_metrics.py scripts/benchmark_preflight.py
PYTHONPATH=$PWD "$PY" -m compileall -q jitmind/code_memory/preflight.py jitmind/code_memory/lesson_models.py jitmind/code_memory/telemetry.py tests/test_preflight_metrics.py scripts/benchmark_preflight.py
git diff --check
```

- Existing preflight regressions before adding the new tests: **84 passed**, three
  optional-dependency warnings, 7.83 s.
- Combined targeted suite: **105 passed**, three warnings, 17.61 s. The 21 new
  metrics tests separately passed in 2.29 s after adding a barrier to make the
  concurrency oracle independent of legitimate SQLite read/write contention.
- Initial full run: **869 passed, 30 failed**, three warnings, 94.59 s. All failures
  were `tests/test_code_context.py` real-parser calls returning `adapter_failed`:
  that file hardcodes the local adapter path rather than reading the supplied
  environment override, and this worktree had no provisioned `node_modules`.
- Recovery: verified `adapters/graft/package-lock.json` with `cmp` against the named
  J08 runtime, then temporarily symlinked only `adapters/graft/node_modules` to
  its provisioned J08 counterpart. No dependency code was copied or installed.
  The actual candidate adapter source remained in use. **The symlink was removed
  after testing and is excluded from the final diff.** Reproduction requires the
  same reviewed local runtime provisioning for the hardcoded-path parser tests.
- Full suite with that runtime available: **903 passed, zero failed, zero skipped**,
  three optional-dependency warnings, **204.06 s**. This includes actual parser,
  binding, SQLite and independent-process contention tests. Exact output is
  retained at `/private/tmp/jitmind-C06-full-suite.txt`.
- Ruff, compilation and whitespace validation passed. All five changed/new Python
  files also parsed with `ast.parse(..., feature_version=(3, 10))`. That is a
  grammar check, **not a Python 3.10 runtime run or typecheck**. Mypy was not
  installed and no typecheck is claimed. Python 3.10/3.11 and other platforms remain
  CI/supervisor gates; optional provider integrations were not run.

New cases cover exact exclusive fake-clock accounting; same-stage nesting; real
zero versus unentered/null; backward/nonfinite/throwing clocks; bounded recursion;
real J02 facts plus J06 projection/preflight and binding callbacks; no lessons;
receipt replay/dedup; default deadline expiry; revocation before acquisition and
during rendering; conflict/invalid arguments; storage contention and recovery;
service restart rejecting old sessions while retaining durable receipts; failed
receipt writes and rollback; adapter exceptions/cancellation; sink contention and
failure; metadata/recorder injection rejection; concurrent users; and nested
measurement-disabled calls restoring the outer ContextVar.

A delivered bound lesson still makes exactly four binding-authority callbacks
(acquisition, selection, before commit, after commit); replay makes three. The
positive composed measurement test does not replace/intercept product methods.
Fault tests inject failures only to verify rollback, sanitized outcomes and retained
measurements. Existing failure, payload and elapsed-time assertions were retained.

## Retained pre-review benchmark observation

Final command: `PYTHONPATH=$PWD "$PY" scripts/benchmark_preflight.py --iterations 20
> /private/tmp/jitmind-C06-preflight-benchmark-final.json` (run after the full suite).
The raw JSON is retained separately at that path, with all 200 actual samples,
per-source SHA-256 hashes, environment, task IDs and paired differences. The earlier
exploratory run is retained at `/private/tmp/jitmind-C06-preflight-benchmark.json`;
it ran concurrently with tests and is not the final-source observation.

Pre-review source aggregate SHA-256: `ee3ab396e9ba1ef9aef8dac28e0c31101dec05cc06763c6199f57f3bd21dddbf`. The manifest was
recomputed against the final working Python sources and verified to match.
The output file itself has SHA-256 `54e3109aecd5dffad0051ddcd117bbf559015c5cbc448e8734032b891ee33459`.

| Mode | Samples | p50 elapsed (ms) | p95 elapsed (ms) | Delivered | Nothing relevant | Binding checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| disabled | 100 | 12.162313 | 20.942500 | 60 | 40 | 640 |
| enabled | 100 | 10.442583 | 22.847708 | 60 | 40 | 640 |

Paired enabled-minus-disabled overhead: p50 **0.047062 ms**,
p95 **5.644875 ms**. These are observations, not acceptance limits.
Enabled stage invocation totals: `{"dedup": 100, "eligibility": 2020, "lookup": 100, "ranking": 100, "rendering": 60}`.
All 100 enabled traces had zero clock anomalies and span overflows. All stage
sums were no greater than their inclusive total. Both modes made the same number
of binding calls and returned the same outcome counts. This compares recording
on/off in the final instrumented library; it is not a baseline run of old code.

Remaining gates: C08 composition/comparisons, supervisor review, Python-version
and platform matrix, independent release signoff, and live-provider qualification
if selected by the owner. No provider or production benchmark was attempted.

## Independent review correction: semantic result identity

The independent review reproduced a compatibility issue: the new timing field
participated in dataclass equality and hashing. Metrics now use
`field(default=None, compare=False)`, so observational clock/export differences
do not change the identity of an otherwise identical legacy result. All original
semantic fields still participate. Two real empty-delivery regressions cover
equality, sets/dictionaries, disabled metrics and changed-semantic-field controls.
The benchmark above is retained for its exact pre-review source identity, not
relabelled as a measurement of the corrected candidate.

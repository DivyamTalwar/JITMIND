# C02 — persistent projection integration contract

Implemented in `jitmind.projections`. Owned files: its seven Python modules,
`tests/test_projection_consumers.py`, `tests/test_projection_process.py`,
`tests/test_projection_remediation.py`,
`docs/projections.md`, and this note. No storage, MemoryAgent, ScopeAuthority,
research or code-context changes. No dependencies or donor code added.

## Public signatures

```python
SQLiteProjectionSource(store: SQLiteDurableStore, *, source_id: str)
source.events(authority, scope, *, after_revision=0, limit=100) -> tuple[ProjectionEvent, ...]
source.event(authority, scope, event_id) -> ProjectionEvent | None
source.snapshot(authority, scope, snapshots=(), *, limit=1000) -> SourceSnapshot

SQLiteVectorProjection(path, *, source_id, embedder=None, fault_hook=None)
SQLiteGraphProjection(path, *, source_id, trusted_relations=False, fault_hook=None)
ProjectionCoordinator(authority, source, consumers)
coordinator.deliver(scope, event, *, snapshots=()) -> dict
coordinator.drain(scope, *, snapshots=(), limit=100) -> dict
coordinator.rebuild(scope, *, snapshots=(), limit=1000) -> dict

vector.search(authority, scope, source, query, *, snapshots=(), limit=10,
              candidate_limit=1000) -> QueryResult
graph.traverse(authority, scope, source, start, *, snapshots=(), depth=2,
               edge_limit=100, visited_limit=100, candidate_limit=1000) -> QueryResult
consumer.status(authority, scope, source, *, snapshots=()) -> dict
```

Consumers also expose individual `deliver(authority, scope, source, event, ...)`
and `rebuild(authority, scope, source, ...)` worker methods. Coordinator results
are keyed by consumer path, not kind; consumers acknowledge independently.
`ProjectionEvent(namespace_id, event_id, revision, memory_ids: tuple[str,...])`
is frozen. Source adapter normalizes C03 values into this package's event type.

`QueryResult(status, reasons, hits=(), edges=(), nodes=())` is frozen.
`ProjectionHit(fact_id, score, content, metadata_json, revision, repo_id,
snapshot_id)` always carries current authoritative content/provenance.
`GraphEdge(subject, predicate, object, supports)` identifies each live supporting
fact and source revision. C01 should preserve `partial`/`unavailable` reasons
when mapping these hits into fusion. No automatic scoped-research backend added.

## State, authorization and migration

- Host-issued ScopeAuthority is unchanged. All APIs validate token/namespace;
  repo/snapshot filtering precedes payload selection and candidate expansion.
  Final authority/token/consumer-generation checks reject stale positives.
- Actual installed v1/v2 authority schema is validated on read-only connections.
  Base v1 fallback reads all committed events, including reference-delivered
  ones. With C03 installed, public all-event/get-event identities are additionally
  cross-checked. No speculative schema migration or source writes.
- New independent DELETE/FULL v1 consumer databases only: distinct graph/vector
  application IDs, exact schema/config/model identities, immutable source_id,
  event receipts, retry counts/codes, per-fact monotonic revision/tombstone,
  contiguous watermark and rebuild baseline. Rebuild preserves receipt history.
- A namespace binds one worker repo/snapshot selection; change via explicit
  bounded rebuild. Reader selections may be narrower. Event gaps stay partial.
  Rebuild requires an untruncated authoritative snapshot and never lowers the
  observed source revision or removes deletion/retention guards.
- Vector identity: `EmbeddingIdentity(model, version, dimensions, normalization)`;
  trusted `Embedder.embed(text)` callback. Default is deterministic **lexical
  feature hashing**, not learned semantic quality. Invalid dimensions/components,
  model drift and exceptions fail closed; compute is outside write transactions.
- Graph metadata: `relations: [{subject, predicate, object}]`, each a bounded
  identifier string, max 100 relations/fact. Host must establish metadata trust
  before opting into `trusted_relations=True`. No inferred factual relations.
- Missing/changed source anchor or restored revision rollback is unavailable.
  Restore always creates a new destination; never overwrite an authority/index.

## C07 / C09 maintenance

Administrative methods are host-only, never request RPCs:

```python
consumer.admin_plan_purge(namespace_id, fact_id=None) -> PurgePlan
consumer.admin_purge(plan)                  # validates exact subjects/revisions
consumer.admin_disable(disabled=True)
consumer.diagnostics() -> dict
consumer.export(namespace_id, *, limit=100) -> dict
consumer.backup(destination) -> Path
SQLiteVectorProjection.restore(backup, destination, *, source_id, embedder=None)
SQLiteGraphProjection.restore(backup, destination, *, source_id, trusted_relations=False)
```

Purge removes only selected payload vectors/supports; metadata-only guards
survive rebuild/restart and subsequent backup/restore. Namespace purge also
suppresses future namespace facts. A literal `"*"` is an exact fact ID, never a
wildcard. No automatic runtime purge, backup deletion or log deletion. Restoring
a backup predating retention requires host replay of the current C07 retention
plan before serving. `export` is bounded metadata; backup includes full index.

## Initial implementation evidence (historical; superseded by remediation below)

Interpreter used throughout:
`/Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python`
with `PYTHONPATH=$PWD`; Python 3.12.12 / SQLite 3.51.2 / macOS arm64.

C03 note read during implementation. Actual C03 storage was copied read-only to
`/private/tmp/C02-C03-compose-dalhsxof`, alongside C02 candidate library and tests.
These temporary dependency copies are excluded from the final diff. Exact copied
storage SHA-256 values:

| File in jitmind/storage | SHA-256 |
| --- | --- |
| `__init__.py` | `0c5823ee8e2ff29f4ba20ea7b9c7105aad68f9bd29873a4e4e2e108914c7a6d1` |
| `history.py` | `ef802ca68775d951a930a80e5b93b77608430da64093b48dce109b32db10b03f` |
| `migration.py` | `26cf90e0911fe6bfae875ef31c50f0b8cbf94b05c2d5a694cd5408622533a15a` |
| `models.py` | `8730e560ebc02d8e02992fdcb0a5d8a7307381b255463303702bd8a0d9dc9f7b` |
| `sqlite.py` | `0a872fe67d3ddde7d1807f11ec620294f5c2645345e588a4aa4dd6333ccf01ba` |

From that isolated directory:
`PYTHONPATH=$PWD $PY -m pytest -q tests/test_projection_*.py tests/test_transactional_history*.py`
→ **124 passed, 0 skipped, 3 optional-dependency warnings, 22.67 seconds**.
This includes 65 C02 projection cases and 59 actual C03 transactional-history
cases, with real SQLite files and process interruption, not parser mocks.

First composition run: 118 passed, 1 failed. Its outbox-pruning fixture broke
v2 history foreign keys; strict validation correctly rejected it. The missing
event/bounded rebuild test now creates a real supported v1 authority, retaining
all gap assertions. Production validation was not weakened.

`$PY -m ruff check jitmind/projections tests/test_projection_*.py` passes;
`$PY -m ruff format --check jitmind/projections tests/test_projection_*.py` passes
(all nine Python files). Python 3.10 AST syntax parsing passes for all seven
modules. The offline example in `docs/projections.md` executed successfully.
No mypy/typecheck, live provider, learned-model quality, other-platform or
physical power-loss testing was performed.

Final base command: `PYTHONPATH=$PWD $PY -m pytest -q` from the C02 worktree
→ **947 passed, 0 skipped, 3 optional-dependency warnings, 169.63 seconds**.
This includes all 65 C02 projection cases. The earlier base run passed 942 tests
with no skips in 127.66 seconds; five additional regression cases were added
afterward. Warnings name the absent optional DenseRetriever,
CohereDenseRetriever and CohereReranker integrations; no provider was invoked.
The host-provisioned untracked
`adapters/graft/node_modules` symlink is not a C02 source change and was left
untouched. Existing real Graft parser tests run as part of the full suite;
no skipped parser result is counted as integration success.

Remaining gates: supervisor final-stack composition with C01/C07/C09; enforce
host relation trust and source identity policy; replay retention when restoring
older backups. Trusted in-process embedders are not preemptible or sandboxed.
No production-ready, learned semantic superiority or complete-ZIP claim.

Implementation checks also corrected concurrent purge/disable return races,
literal-star versus namespace purge ambiguity, and preservation of immutable
receipt history during rebuild. Regression cases protect each behavior. Existing
failure assertions outside C02 files were not edited. All C02 files are left
uncommitted for supervisor review.

## Remediation contract

No consumer schema change: v1 already has `config(key TEXT PRIMARY KEY, value
TEXT NOT NULL)` with canonical decimal `generation`, boolean-text `enabled`,
immutable `source_id` and configuration `identity`. Delivery/rebuild capture
that generation, enabled state and exact namespace selection before source or
embedding work, then compare all three inside `BEGIN IMMEDIATE` before writes.
Retry receipt writes use the same fence so obsolete work cannot alter a newer
rebuild/delivery ledger. Rebuild remains allowed while disabled if administration
has not changed during preparation. No migration or peer storage edits required.
Quiesce older application writers when deploying this behavioral fix; schema v1
alone cannot distinguish an older implementation with weaker fencing.

Current corrected fact validity is authoritative; immutable page `t_valid` and
`t_invalid` remain historical observations. Independent page/fact `expires_at`
and `ttl_seconds` still restrict eligibility; TTL starts at fact `t_created`, as
in actual C03. Malformed eligibility excludes positives and reports an explicit
unknown outcome. Query status reuses the bounded source observation, and final
validation preserves `candidate_limit` without acquiring an extra payload row.
`SourceSnapshot` adds a trailing defaulted `reasons: tuple[str, ...] = ()`;
existing six-argument construction remains valid. Malformed JSON-valid TTL
produces `unknown_eligibility`, excludes the affected fact and yields partial
status with safe positives; nonfinite stored JSON fails unavailable. Public
consumer signatures and all persisted table schemas remain unchanged. The
`config.generation` increments transactionally for every administrative or
consumer write. Disabling rejects even an already-done event without touching
its receipt. Rebuild while already disabled remains explicit maintenance.

The frozen review had weaker write checks than current C02 at task start.
Current C02 already rejected disable and changed selection during delivery, but
its failure handler wrote a retry afterward and same-selection rebuild or a
re-enable still allowed obsolete work. All these cases are now fenced. Existing
receipt/deletion/race assertions remain; the incompatible metadata test now
asserts actual v2 ingestion rejection then runs its original exclusion/positive
assertions on a real explicitly labelled v1 legacy schema fixture.

The payload acquisition simplification removes the redundant full snapshot for
query status and the extra `limit + 1` payload. A bounded SQL existence probe
reports truncation; a size-only SQL preflight rejects excessive bytes before
payload SELECT/decode. Initial and final query observations each select no more
than `candidate_limit` payloads. SQL metadata inspection still occurs within
processing bounds; this is not a claim of constant-time database work.

## Remediation verification, 2026-09-29

All commands use the isolated interpreter above, `PYTHONDONTWRITEBYTECODE=1`,
`PYTHONPATH=$PWD`, real local SQLite fixtures and no live providers or installs.
Temporary composition: `/private/tmp/C02-remediation-_3xh6pk3`; this is an
identified dependency/test copy, excluded from the final diff. It contains the
current C02 package/tests, actual C03 `jitmind/storage` and its two history test
modules, plus existing candidate scripts/docs/adapters for full-suite fixtures.
The original review probes are copied byte-for-byte as
`tests/test_independent_review.py` only in that temporary directory; the original
oracle was not edited. Probe SHA-256:
`478782de9a62762337d8c3d5c716d0d4ce097aa86463134d68ec8288a057a02f`.

C03 storage dependency SHA-256 manifest (`dependency-manifest.json`):

| File | SHA-256 |
| --- | --- |
| `models.py` | `189c4179ec36f2ecefd7587004ade27f2c87da88142981e721b25c51be38635b` |
| `__init__.py` | `0c5823ee8e2ff29f4ba20ea7b9c7105aad68f9bd29873a4e4e2e108914c7a6d1` |
| `sqlite.py` | `aa6ef08de29e547b562165412e7a9b965214726e954b0e0024172e0cfeff8ec4` |
| `migration.py` | `26cf90e0911fe6bfae875ef31c50f0b8cbf94b05c2d5a694cd5408622533a15a` |
| `history.py` | `2930421f4afc12c8f668099e7b810903e61987b41fd0e5ec4f7bd63276ee6b48` |

Exact commands below use `$PY` for
`/Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python`;
prefix each pytest command with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD`.
Log paths are relative to the temporary composition unless stated otherwise.

- Pre-fix actual current C02+C03:
  `$PY -m pytest -q -p no:cacheprovider tests/test_independent_review.py tests/test_projection_consumers.py tests/test_projection_process.py tests/test_transactional_history.py tests/test_transactional_history_faults.py`
  → **146 passed, 7 failed, 0 skipped**, 22.29s (`baseline-tests.log`).
  Failures were TTL (3), corrected validity (1), candidate acquisition (2), and
  incompatible metadata ingestion fixture (1). The two reviewed delivery probes
  already passed against task-start C02, unlike the frozen review snapshot.
- New regressions against saved pre-fix code:
  `$PY -m pytest -q -p no:cacheprovider tests/test_projection_remediation.py`
  → **13 passed, 34 failed, 0 skipped**, 4.11s (`regression-before.log`).
  This precedes the four final page-only/time-elapse controls. These failures
  also exposed obsolete retry writes and rebuild/generation interleavings.
- First fixed combined targeted run, before those four final controls:
  `$PY -m pytest -q -p no:cacheprovider tests/test_projection_remediation.py tests/test_independent_review.py tests/test_projection_consumers.py tests/test_projection_process.py tests/test_transactional_history.py tests/test_transactional_history_faults.py`
  → **200 passed, 0 skipped**, 22.16s (`targeted-after.log`).
- Final same combined targeted command, including all 51 remediation cases:
  → **204 passed, 0 skipped**, 30.18s (`targeted-final.log`). Covers real vector
  normalization/ranking, graph support/BFS, independent receipts/watermarks,
  immutable histories, current deletion/revocation/purge/restore guards,
  process exits, rollback and simulated ENOSPC without real-disk exhaustion.
- `$PY -m ruff check --no-cache jitmind/projections tests/test_projection_*.py`
  → passed. `$PY -m ruff format --check jitmind/projections tests/test_projection_*.py`
  → all 10 files formatted. Python `ast.parse(..., feature_version=(3,10))`
  → all seven modules and three test files passed grammar parsing. Mypy remains
  unavailable; neither a typecheck nor Python 3.10 runtime execution is claimed.

Two harness failures are retained honestly: the initial absolute probe path
caused pytest to inspect a restricted sibling under `/tmp` (one collection
error, `baseline.log`), fixed by copying the unchanged probe into the fixture
root. The first composed full-suite attempt lacked the candidate `scripts`
module (two collection errors, `full-composed.log`); existing scripts/docs were
then copied into the fixture. Neither attempt ran tests or counts as success.

No new donor code was copied. Existing Graft pin
`80692e5ad0bc8e8f7e1edea648247d90b9f76820` / `@nanonets/graft` 0.20.0 and
copyright/notices remain unchanged. The existing host-provisioned parser runtime
symlink is untouched. Optional integrations still emit the three existing
DenseRetriever/CohereDenseRetriever/CohereReranker warnings; no provider invoked.

Full owner suite from the original C02 worktree:
`$PY -m pytest -q -p no:cacheprovider`
→ **996 passed, 2 skipped**, 140.50s (`full-owner.log`). The two skips are the
new C03-only validity-correction cases; base C02 lacks `commit_temporal`.
Both pass in real composition. There are no parser skips: the real Graft parser
and existing process tests executed. This is not an all-platform/provider claim.

The first runnable broader composed suite used the initial manifest above and
reported **1084 passed, 2 failed, 0 skipped**, 140.86s
(`full-composed-final.log`). Both failures were omitted fixture files (`setup.py`
and `eval/README.md`), not product failures. Existing candidate setup/eval files
were then copied as test dependencies; no build hook or eval provider invoked.

C03 changed during verification. The temporary dependency was refreshed only
after the initial full composed process exited; original dependency bytes remain
in `initial-c03-storage`. Updated manifest (`dependency-manifest-current.json`)
changes only these storage hashes; the other three remain as listed above:

| File | Refreshed SHA-256 |
| --- | --- |
| `sqlite.py` | `ee29b1fa1d49f106c4c95e4b41e9a166e9ad433e6ac41c9e4b62c543e2f6424c` |
| `history.py` | `8ad10d6ca381b2fd199fdd3aedd0e8368d9569fdf16b682fd1381044615796d1` |

The exact final combined targeted command above was rerun against refreshed C03:
**204 passed, 0 skipped**, 20.13s (`targeted-current-c03.log`). All original
independent review oracles pass unchanged; none required deletion or reversal.

Remediation changed files: `jitmind/projections/source.py`, `base.py`,
`models.py`; `tests/test_projection_consumers.py`,
`tests/test_projection_remediation.py`; `docs/projections.md` and this interface
note. The real vector/graph algorithms and process tests remain covered. Public
consumer methods retain their signatures; `SourceSnapshot.reasons` is the sole
additive value-contract field. All source changes remain uncommitted.

Remaining external gates: supervisor final-stack integration with C01/C07/C09,
quiescing older projection writers at deployment, and host enforcement of
relation trust/source identity and replay of retention plans for older backups.
No live providers, learned model quality, other Python/OS runtime matrix, typecheck,
physical power-loss or production-data qualification was performed.

Final broader composition command, from the temporary root with refreshed C03:
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD $PY -m pytest -q -p no:cacheprovider`
→ **1086 passed, 0 skipped**, 122.28s (`full-current-c03.log`), with only the
three existing optional-dependency warnings. This includes unchanged independent
review probes, all C02 owner tests, actual C03 transactional history tests, actual
Graft parser integration, and process/storage tests. All five remediation items
are resolved in these measured fixtures. The earlier harness errors are retained
above and in logs; no failure assertion or review oracle was removed.

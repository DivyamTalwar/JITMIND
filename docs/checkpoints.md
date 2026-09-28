# Optional conversation checkpoints (storage v1, file format v2)

This extension records synthetic or application-supplied conversation events and
publishes summaries of verified, closed source ranges. It does **not** restore a
process, browser, filesystem, provider session, tool execution, or external effect.
Summaries are application content, not evidence that missing events ever existed.
No provider/model is imported or called by the extension.

## Integration and dependencies

Compose J03's `jitmind.scope` before enabling the ledger. Use its single
`ScopeAuthority`, `ScopeContext`, and `ScopeDenied` implementation; this ticket
intentionally does not ship a second authorization implementation. The trusted
host authenticates principals, calls `grant`/`context`, owns the authority, and
chooses the local DB path. Never expose grant/revoke, DB path, or file-checkpoint
root as user request parameters. A request namespace string is not authorization.
Each ledger data operation checks the supplied scope before storage access,
before commit/return, and after its transaction. Authority is process-local;
coordinating revocation across host processes remains the host's responsibility.
Revocation and SQLite commit are not a single distributed transaction. A denial
after commit can withhold the response even though data committed; retry using a
fresh authorized context and the identical draft/key to determine the outcome.

Import explicitly:

```python
from jitmind.checkpoints import SQLiteCheckpointLedger
from jitmind.scope import ScopeAuthority
```

Core `import jitmind` and the existing `CheckpointManager` do not import this
extension or open a DB. There are no new dependencies. Compose J01's revised
`file_lock` and `atomic_io` before deploying the file helper; this ticket uses
their existing call contracts and does not supply a second lock implementation.
The baseline `.gitignore` ignores directories named `checkpoints`. Preserve the
accompanying source-only allowlist when composing this ticket so
`jitmind/checkpoints/__init__.py` and `ledger.py` are included. If that allowlist is
not composed, these two source files require explicit staging with `git add -f`.

## Runnable example

From the composed repository, run `PYTHONPATH=$PWD python` with this program.
All values below are fixtures, not agent transcript evidence. Temporary storage
is destroyed only by the example's trusted host cleanup.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from jitmind.checkpoints import SQLiteCheckpointLedger
from jitmind.scope import ScopeAuthority

with TemporaryDirectory() as directory:
    authority = ScopeAuthority()
    authority.grant("demo-principal", "demo/namespace", [])
    scope = authority.context("demo-principal", "demo/namespace", "request-1")
    ledger = SQLiteCheckpointLedger(Path(directory) / "conversation.sqlite", authority)
    time = "2026-09-28T10:00:00+00:00"
    for seq, kind, call, payload in [
        (0, "message", None, {"text": "Synthetic question"}),
        (1, "tool_start", "call-1", {"tool": "fixture"}),
        (2, "tool_result", "call-1", {"text": "Synthetic result"}),
    ]:
        ledger.append(scope, "conversation-1", seq, event_id=f"event-{seq}",
                      payload=payload, observed_time=time, kind=kind, tool_call_id=call)
    snapshot = ledger.snapshot(scope, "conversation-1", 0, 2)
    # Generate your summary HERE, after snapshot has closed its read transaction.
    # No callbacks or model execution are accepted by publish().
    summary = "Fixture question and completed fixture tool call."
    draft = snapshot.draft(summary=summary, idempotency_key="summary-request-1",
                           model_metadata={"model": "fixture", "schema": "demo-v1"})
    first = ledger.publish(scope, draft)
    reopened = SQLiteCheckpointLedger(Path(directory) / "conversation.sqlite", authority)
    assert reopened.publish(scope, draft) == first
    assert first.revision == 1
    assert len(reopened.snapshot(scope, "conversation-1", 0, 2).events) == 3
    assert reopened.history(scope, "conversation-1") == (first,)
```

The integration methods are:

- `append(scope, stream, sequence, *, event_id, payload, observed_time,
  kind="message", tool_call_id=None) -> Event`.
- `snapshot(scope, stream, start, end) -> Snapshot`.
- `snapshot.draft(*, summary, idempotency_key, model_metadata=None) -> Draft`.
- `publish(scope, draft) -> Checkpoint`.
- `history(scope, stream) -> tuple[Checkpoint, ...]`.
- `history_page(scope, stream, *, after_revision=0, limit=100) -> HistoryPage`.

`history` retains its tuple return for complete histories fitting in one bounded
page. Larger histories raise `HistoryIncomplete`; its `page` contains the first
page, never an implicitly truncated successful tuple. `HistoryPage.checkpoints`
is a tuple, `complete` explicitly indicates exhaustion, and `next_revision` is
the exclusive cursor for the next request (or `None` when complete). A page is
limited to 100 checkpoints and 4,000,000 aggregate stored field bytes, whichever
is reached first. `limit` accepts integers 1–100. Every page independently checks
authorization and uses a read transaction. Concurrent publications may appear on
later pages; completion describes that page's transaction, not a frozen view of
all future revisions. For example:

```python
cursor = 0
while True:
    page = ledger.history_page(scope, "conversation-1", after_revision=cursor)
    for checkpoint in page.checkpoints:
        print(checkpoint.revision)
    if page.complete:
        break
    cursor = page.next_revision
```

`Event.payload` decodes a detached copy; event/snapshot/draft/checkpoint records
are frozen. Persist a draft for retry with `json.dumps(dataclasses.asdict(draft))`
and restore using `Draft(**json.loads(text))`. Use trusted storage for drafts;
publication verifies the retained source independently of the draft. Exact replay
includes the summary, metadata, schema, digest, range, expected revision, and key.
A different key with a stale revision fails; an exact historical key replays even
after newer revisions. The API allows overlapping valid ranges; the application
chooses whether a new summary replaces or extends earlier summaries.

## Range and tool invariants

Namespace is taken from an issued, current scope. Every SQL source query also
filters the exact stream. SQLite's primary key is `(namespace, stream, sequence)`;
`(namespace, stream, event_id)` is separately unique. `BETWEEN start AND end` is
inclusive. Therefore count equals `end-start+1` only for a unique contiguous range
in this implementation. Duplicate replay requires the complete canonical event to
match, including payload, digest, timestamp, kind, call ID, and identity. Canonical
JSON for each event payload, whole snapshot digest input, or draft is limited to
4,000,000 ASCII characters. Validation bounds traversal and input string lengths
before serialization, then caps incremental encoded output; it rejects nesting
beyond 64 and integers larger than 4,096 bits. Conflicts
raise `EventConflict`. Sequences/revisions must be integers (never booleans), from
0 through `2**63-2`; ranges contain at most 100,000 events. New events must increase
the stream sequence. Gaps are retained as gaps, cannot be backfilled, and reject
any snapshot spanning them or following them: a missing prefix event could conceal
a tool start. The first appended sequence establishes the stream origin; all
source from that origin through the snapshot end must be present. Begin a new
stream for a corrected import. This rule
prevents later insertion of a start/result into previously examined history.

Snapshot and publication first run bounded count/byte queries **before selecting
payload rows**, in the same transaction as their subsequent reads. Requested
source acquisition is capped at 4,000,000 aggregate stored field bytes (UTF-8 text
bytes and decimal integer representations). The entire prefix through the end,
including messages, is capped at 100,000 rows and 4,000,000 bytes of sequence,
kind, and call-ID metadata. Tool replay iterates a payload-free cursor; it does
not materialize the prefix. Both aggregates use limited subqueries. A narrow
range late in a stream can therefore exceed the prefix budget. Use a new stream
at a verified closed boundary; reducing the requested range does not erase prior
tool dependencies. No pruning or automatic boundary reset is performed.

History first reads at most `limit + 1` revision/size pairs, then selects only the
payloads that fit the aggregate budget. An individually oversized stored history
row raises `CapacityExceeded`; no empty page with a nonadvancing cursor is
returned. New publication checks its stored row size before commit. Exact
idempotency and event replays also size-check retained rows before acquiring
their payloads. These byte limits bound acquisition, not total Python RSS:
decoded objects, canonical encoding, and copies have additional bounded overhead.

Timezone-aware timestamps are required. A terminal uppercase `Z` is normalized
to `+00:00` only for validation, including on Python 3.10; the original spelling
is retained in event identity and digests. Replaying with the other spelling is
an identity conflict. Missing timezones and malformed timestamps remain invalid.

Kinds are `message`, `tool_start`, `tool_result`, `tool_cancel`, `tool_timeout`.
A start has a nonempty call ID unique for that stream's lifetime; precisely one
explicit terminal follows it. Unknown terminals, reused starts, repeated results,
and unrecognized kinds are refused. Cancellation/timeout close the recorded
interaction only; this library does not cancel an actual running tool.

Snapshot and publish both replay tool correlations **only through the range end**.
An open call before the start, including one enclosing the entire range, or an
open call at the end rejects the range. A result inside the range whose start
precedes it is rejected. Concurrent/nested calls entirely inside the range work.
A terminal appended after the end cannot retroactively close that range. The
source digest includes schema, exact namespace/stream/range and every event field;
publish rechecks it inside the writer transaction. A summary never fills holes.

## Storage, retention, errors, and rollback

The supported repository is a trusted local SQLite filesystem database, not a
URI or in-memory database. Use an existing parent directory. All tables/triggers
are prefixed `jitmind_cp_`; `jitmind_cp_meta` stores extension schema version 1.
The extension does not use or modify global `PRAGMA user_version`, so J02 durable
memory can use separate tables in the same host-selected database. Existing
non-DELETE journal modes are rejected without changing them. New databases use
SQLite's default rollback DELETE journal; each connection uses `synchronous=FULL`,
foreign keys, and a bounded busy timeout (default 2,000 ms, configurable 1–30,000).
`operation_timeout_ms` defaults to 5,000 and accepts integers 1–30,000. Each
transaction has a monotonic deadline: SQLite's progress handler checks every
1,000 VM instructions; Python validation, encoding and scan loops also check it,
with a final check before commit. Busy waiting is capped at the lesser of the two
timeouts. These are cooperative execution bounds, not hard real-time deadlines:
an individual native operation, OS I/O or commit sync cannot be preempted by a
Python check. Deadline failures before commit roll back; a commit acknowledgement
failure retains `CommitUncertain` semantics. Publication uses the same count,
byte and deadline checks inside its `BEGIN IMMEDIATE` transaction.
The code never enables WAL, including on the environment's SQLite 3.51.2. Shared
DB users must agree on DELETE journal mode and compatible connection policies.

`BEGIN IMMEDIATE` serializes publication; revision compare-and-swap, checkpoint
history, and idempotency key insertion commit together. History and raw event
UPDATE/DELETE are blocked by triggers. The explicit retention policy is **retain
all raw events and all checkpoint revisions indefinitely**. There is no pruning
API and no deletion after summarization. Out-of-band DB editing can defeat these
protections and is unsupported. Large histories still cost disk; scans over the
documented budgets fail explicitly. This is not a performance or
production-readiness claim.

Errors have sanitized messages: `InvalidInput`, `EventConflict`, `UnsafeRange`,
`SourceChanged`, `RevisionConflict`, `IdempotencyConflict`, `SchemaMismatch`,
`StorageError`, `CommitUncertain`, `CapacityExceeded` (an `InvalidInput` subtype),
and `HistoryIncomplete`. All inherit `CheckpointError`; authorization
uses J03 `ScopeDenied`. Only an actual absent JSON file returns `None`; invalid
source data never becomes an empty successful checkpoint. A commit acknowledgement
failure raises `CommitUncertain`, never success. Reopen storage and replay exactly
the original draft/key to resolve before retrying with a different key. Busy and
other database errors fail with bounded waiting, with transactions rolled back on
failure. Model work always happens between snapshot and publish, outside locks.

For rollback, disable the opt-in ledger at the host; existing ResearchAgent JSON
integration is independent. Preserve a consistent DB backup using SQLite backup
or a stopped host. Do not rewrite metadata to downgrade a schema or remove raw
events; restore a known compatible backup if schema migration is needed. Current
code rejects unknown schema versions. Raw events are not encrypted and local OS
access controls remain the host's responsibility.

## Legacy file checkpoints and explicit v2 migration

`CheckpointManager(dir_path)` keeps its existing constructor and positional
`thread_id`/`state` APIs. Each method also accepts keyword-only
`namespace="default"`. The JSON writer uses `v2-<sha256>.json`, hashing canonical
JSON of the **full exact** `[namespace, thread_id]`; it never deletes identity
characters. Payload identity is verified before load/overwrite/delete. The helper
uses the shared file lock for save/load/list/delete, with atomic JSON publication.
It is local trusted-host storage, not a scope-authorized request service. Wrap it
with host authorization or use the scoped ledger for request-facing operations.

Save validates and detaches the serialized JSON before taking the writer lock.
Every dictionary key, including nested keys, must be a string; numeric, Boolean
and null keys are rejected rather than coerced. Duplicate serialized names,
nonfinite numbers and excessive nesting are rejected before replacing old bytes.

A genuine v1 `{thread_id, timestamp, state}` file remains readable at its original
sanitized filename in the default namespace only, provided its stored identity
matches the requested identity. `load_checkpoint_record` exposes metadata with
`validation="legacy_unvalidated"`. **All file state, including new v2 JSON, remains
legacy_unvalidated**: there is no event-range proof to infer from ResearchAgent's
iteration list. Existing `load_checkpoint` still returns the state dictionary.

There is one explicit versioned listing migration: v1 `list_checkpoints()` returned
filename stems (the actual implementation did not return full paths); v2 returns
sorted **exact stored thread IDs**, so each entry can be loaded/deleted safely.
Consumers that saved old stems or wrapped them in full paths must migrate those
handles from the trusted original JSON's `thread_id`, not guess or remove more
characters. For example, old `ab.json` may hold `a/b`; asking for `ab` now raises
`CheckpointIdentityError`. Automatically accepting that alias would reintroduce
the collision vulnerability. No arbitrary full-path loading API is added.

Migration procedure:

1. Stop all v1 writers/readers. Back up the directory. Mixed v1/v2 writers use
   different locks/filenames and are unsupported.
2. List v2 exact IDs, or read trusted original files to convert externally stored
   filename/path handles to their stored `thread_id`.
3. Load with the exact ID and save normally to create v2 state. The matching v1
   original remains intact for review/rollback; v2 takes precedence thereafter.
4. Do not convert old iteration counts into ledger events. Only separately retained
   real source events may be explicitly imported using `append` in sequence order.

Loading treats a candidate that disappears before open as absent. Only in the
default namespace may an absent v2 candidate fall back to the exact-ID-validated
legacy candidate. If that also disappears, load returns `None`. Corruption,
permission failures, symlinks and identity mismatches never trigger fallback.

Deletion validates candidates before unlinking; it removes matching v2 and v1
copies to avoid resurrecting old state. A colliding alias for another identity is
never removed or overwritten. Corrupt candidates cause failure before deletion.
File deletion across two copies is not a crash-atomic multi-file transaction; an
OS unlink failure can leave a copy. Deletion syncs the parent directory before
acknowledging success. Unlink failure before any removal raises
`PersistenceError(outcome_uncertain=False)`; partial deletion or a directory-sync
failure after removal raises it with `outcome_uncertain=True`. Disappeared
candidates are skipped; if none is removed, deletion returns `False`. Deletion
durability is not a restore promise. Rolling back file code requires stopped writers and a backup/explicit
export to a collision-free legacy namespace; older code cannot read hashed v2
files. Never roll back by stripping exact identity characters again.

JSON corruption/duplicate keys/nonfinite constants raise `CheckpointCorrupt`;
unsupported or malformed schema (including invalid stored identity fields) raises
`CheckpointSchemaError`, a `CheckpointCorrupt` subtype. Both remain catchable as
J01 `CorruptStoreError`. A valid but mismatched identity raises
`CheckpointIdentityError`; local read/lock failures raise `CheckpointIOError`,
including actual J01 `StorageError` lock failures. All file checkpoint errors
inherit the generic J01 `StorageError`. Specific checkpoint failures pass through
before the broad storage-error handler, so corruption never becomes generic I/O.
J01 atomic publication and checkpoint deletion preserve sanitized
`PersistenceError`, its exact original `__cause__`, and its `outcome_uncertain`
flag. Read/parsing errors retain their original cause as well. Originals are retained
on validation errors. Callers that previously relied on errors becoming `None`
must now explicitly handle these failures.

## Architectural reference and verification limits

Native Python/SQLite implementation, with no copied Go or SQL code and no Go,
PostgreSQL, runtime-lock, provider, or other product dependency. Architectural
reference: Omnara commit `5fa80ea4f3be999aa63fbc44513de3020ee940af`, specifically
`internal/storage/executionstore/context_checkpoints_boundary.go`,
`context_checkpoints_store.go`, and the locally supplied
`internal/storage/queries/context_checkpoints.sql`.

The query was available and its project/agent predicates and inclusive sequence
range were inspected. The donor schema/uniqueness constraints were **not** included
in the pinned local bundle, so donor count-to-continuity is not independently
proven. This implementation establishes its own primary-key uniqueness. The donor
open-tool query checks starts within the range; this implementation additionally
rejects interactions crossing the start edge. This is selected-invariant coverage,
not full source or runtime coverage.

Tests use actual SQLite databases, bounded spawned processes, explicit barriers
and events, and injected commit failures. They cover a publisher race, crashes
before/after commit, restart replay, missing/changed raw source, tool boundaries,
scoped isolation/revocation, legacy identity collisions, invalid JSON, and concurrent
save/delete exclusion. No sleeps are used as correctness oracles. No test claims
real transcript provenance, power-loss simulation, external-effect restoration,
network-filesystem safety, or provider behavior.

## J07 review remediation verification

All five findings in `reports/J07-review-agent.md` were remediated in the assigned
J07 worktree. The final run adds 53 targeted regressions to the original 60
checkpoint tests. Regression evidence includes rejection before payload SELECTs,
byte/count pagination, actual SQLite progress-handler interruption, rollback
inside publication, valid and invalid UTC spellings, serialized-key rejection
with original-byte preservation, actual J01 lock failure translation, selected
file disappearance and restricted legacy fallback. The current J01 deletion
invariants are covered by failed unlink, late sync and partial-copy deletion
regressions. Existing collision, corruption, scope, restart and spawned process
coverage remains active.

Commands actually run from the J07 worktree with Python 3.12.12 / SQLite 3.51.2
on Darwin:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python -m pytest -p no:cacheprovider -q -ra tests/test_checkpoint_manager.py tests/test_checkpoint_ledger.py tests/test_checkpoint_scope.py --basetemp=/tmp/j07-fix-target-final
# 113 passed, 0 skipped, 3 expected optional-provider warnings, 3.01s.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python -m pytest -p no:cacheprovider -q -ra tests --basetemp=/tmp/j07-fix-full
# 151 passed, 0 skipped, 3 expected optional-provider warnings, 20.44s.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python -m ruff check --no-cache --isolated --select E9,F jitmind/checkpoints jitmind/utils/checkpoint.py tests/test_checkpoint_manager.py tests/test_checkpoint_ledger.py tests/test_checkpoint_scope.py
# All selected checks passed (6 Python files).

PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python -m compileall -q jitmind tests
# Exit 0.

git diff --check
# Exit 0.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python -m mypy --version
# Exit 1: No module named mypy. No type-check claim; no package installed.
```

The final J01 `atomic_io.py` and `file_lock.py` were copied to the actual
`jitmind/utils/` paths **before** running tests. They are dependency-only copies,
not J07-owned changes, and spawned children import those same files. No process
test was excluded or replaced with in-memory module composition. `cmp` against
the corresponding J01 worktree paths exited 0 for both files. Copied SHA-256:

- `atomic_io.py`: `aa58bbae5ca9f686ad020f7ebe74a27c082400964aebad3614fe46cd4466202d`
- `file_lock.py`: `4498fbe42d1a9b8f7224f17d684a8a35fb15dbb815b49de83d1246be5ed6e398`

The already-composed J03 `scope.py`, source-only `.gitignore` exception, scope
tests and pre-existing `.omx/` files were not edited during remediation. No
stale-card or unrelated implementation was changed. All work is uncommitted.

No reproduced finding remains blocked. Qualification followups: run the suite on
an actual Python 3.10 interpreter (none was available on PATH), and run the
project's type checker in an environment that supplies it. The UTC regression
uses a parser double rejecting terminal `Z`, proving that normalization happens
before parsing while preserving stored spelling; it is not a Python 3.10 runtime
test. This run does not qualify Windows, network filesystems, hardware power loss,
production readiness or benchmark improvements. Retained history consumes disk;
large prefixes deliberately require a new stream at a verified closed boundary.


## Composed J01/J03/J07 verification and v2 test adaptation

This subsequent verification supersedes the earlier 151-test full-suite result
for integration purposes: that earlier run did not include all three new J01
regression modules and all J01 store implementations. The actual composed baseline
uses J01 commit `b3fe1d51a4cb98ea2ddce7e94fd3f732010b623d`, the existing J03 scope,
and the J07 checkpoint implementation. All commands below ran in the assigned
J07 worktree, with actual on-disk dependency modules also imported by spawned
children. No test selections, exclusions, skips, provider calls, installs, or
external writes were used in either full-suite run.

The unadapted baseline reproduced the supervisor's **6 failed, 252 passed**.
All six failed cases are now fixed:

- Both `test_other_legacy_stores_errors_and_corruption` checkpoint parameters
  target `store._path("user")` for new writes and still catch `CorruptStoreError`.
- `test_checkpoint_public_delete_between_exists_and_open` patches `Path.open`.
  Bounded reader/deleter threads use the common `.checkpoints.lock`. A real
  zero-timeout acquisition proves deletion is blocked while the reader is paused;
  the reader is released before either join. The reader observes the old state,
  deletion succeeds, and subsequent reads observe absence. Thread exceptions are
  captured and asserted as failures, never swallowed.
- All three `test_checkpoint_delete_failure_preserves_acknowledgement` parameters
  use the v2 path and common lock. Late sync is injected at the executed
  `jitmind.utils.checkpoint._sync_directory` binding. Failed unlink preserves old
  bytes and has `outcome_uncertain=False`; successful unlink plus failed directory
  sync leaves the file absent and has `outcome_uncertain=True`. Both preserve the
  exact injected exception in `__cause__`. Corrupt bytes prevent deletion.

These are white-box adaptations to intentional v2 filenames, lock placement and
I/O bindings; they do not relax the legacy invariants. A separate genuine-v1
fixture writes the old filename and envelope directly, proves load/migration and
joint deletion, and exercises malformed JSON, schema, identity and UTF-8 bytes.
Both v1 and v2 corruption tests require load/save/delete/list to reject the data,
retain original bytes, and raise a specific corruption error catchable through
`CorruptStoreError` and `StorageError`. Invalid persisted identities are schema
corruption, not a valid alias collision that save may bypass. Valid colliding
legacy identities retain the documented migration behavior.

Separate disappearance cases unlink immediately before calling the original
`Path.open`; the resulting actual `FileNotFoundError` permits default-namespace
legacy fallback and ultimately returns `None` when both candidates are absent.
Existing tests still cover matching fallback, corrupt or colliding legacy data,
namespace restrictions, permissions, save/delete process exclusion and partial
multi-copy deletion. Generic I/O catch regressions also verify exact causes and
unchanged bytes for load/save/delete/list failures.

Only these files were edited as J07-owned changes during this follow-up:

- `jitmind/utils/checkpoint.py`
- `tests/test_atomic_io_regressions.py` (the sole modified J01 integration test)
- `docs/checkpoints.md`

The following dependency-only copies are byte-for-byte identical to both the
J01 worktree and commit above; they must not be attributed to the J07 patch:

- `jitmind/profile/profile_store.py`
- `jitmind/schemas/advanced_memory.py`
- `jitmind/schemas/memory.py`
- `jitmind/schemas/page.py`
- `jitmind/schemas/ttl_memory.py`
- `jitmind/schemas/ttl_page.py`
- `jitmind/utils/__init__.py`
- `jitmind/utils/atomic_io.py`
- `jitmind/utils/file_lock.py`
- `tests/test_file_lock_regressions.py`
- `tests/test_legacy_persistence.py`

All three J01 regression modules were copied before verification; only
`test_atomic_io_regressions.py` was subsequently adapted. Existing J07 ledger,
manager/scope tests, J03 scope, `.gitignore` and `.omx/` content were unchanged by
this follow-up. All owned changes remain uncommitted. J01 peer, delivery and
verification trees were not modified.

Exact test commands and observed results:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python -m pytest -p no:cacheprovider -q -ra tests --basetemp=/tmp/j07-composed-before > /tmp/j07-composed-before.log 2>&1
# 6 failed, 252 passed, 3 warnings in 30.48s.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python -m pytest -p no:cacheprovider -q -ra tests/test_atomic_io_regressions.py --basetemp=/tmp/j07-integration-red > /tmp/j07-integration-red.log 2>&1
# Adapted regression probes before implementation fix: 11 failed, 16 passed,
# 3 warnings in 0.48s; failures exposed corruption catches and deletion causes.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python -m pytest -p no:cacheprovider -q -ra tests/test_atomic_io_regressions.py tests/test_checkpoint_manager.py --basetemp=/tmp/j07-integration-green
# Intermediate focused verification: 65 passed, 3 warnings in 1.34s.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python -m pytest -p no:cacheprovider -q -ra tests --basetemp=/tmp/j07-composed-final > /tmp/j07-composed-final.log 2>&1
# Final composed suite: 275 passed, 0 failed, 0 skipped, 3 warnings in 25.45s.
# All original 258 cases retained; 17 regression cases added.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python -m ruff check --no-cache --isolated --select E9,F jitmind/checkpoints jitmind/utils/checkpoint.py tests/test_atomic_io_regressions.py tests/test_checkpoint_manager.py tests/test_checkpoint_ledger.py tests/test_checkpoint_scope.py
# All checks passed.

git diff --check
# Exit 0.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python -m mypy --version
# Exit 1: No module named mypy. No install attempted or type-check claim made.
```

A Python verification script compiled every `jitmind/**/*.py` and `tests/**/*.py`
source with `compile(..., "exec")` without emitting bytecode, and asserted byte
identity of all 11 untouched copies against both J01 disk contents and
`git show b3fe1d51a4cb98ea2ddce7e94fd3f732010b623d:<path>`; all checks passed.
The three warnings remain the expected optional-provider import warnings.
Remaining qualification limits are unchanged: Python 3.12/Darwin execution only,
no mypy, no hardware power-loss or unsupported-filesystem qualification, and no
production-readiness claim.

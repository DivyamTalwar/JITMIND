# Transactional ordinary-memory history

Schema v2 makes each ordinary durable MemoryAgent ADD, UPDATE, DELETE and NOOP
record its knowledge effect in the **same SQLite transaction** as facts, immutable
pages, receipts and outbox. An UPDATE appends the retired parent and new child;
a DELETE appends a tombstone; a NOOP appends only its operation revision. Earlier
versions are never overwritten. A replay returns the original receipt without
appending versions or invoking generation. No second database or callback publishes
history. Legacy JSON APIs and J09's separate structured oracle are unchanged.

This document supersedes the schema-v1-only description in durable-storage.md for
new databases. Historical access is an explicit trusted-host API. It can expose
retained facts hidden from current reads. Do not expose the store directly to an
untrusted caller. C01 must validate issued namespace/repository authority before
calling it and recheck before delivering results. Current `get_entry`, `get_page`,
`memory_update`, receipt content and projection visibility still suppress retired
or deleted text. A historical observation does not grant current disclosure rights.

## Runnable local example

Run the following Python from the repository root with `PYTHONPATH=$PWD`. It uses
real SQLite and a deterministic recording generator; there is no network access.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from jitmind.agents.memory_agent import MemoryAgent
from jitmind.storage import (
    IngestRequest, Proposal, SQLiteDurableStore, ValidityCorrection,
)

class RecordingGenerator:
    def __init__(self):
        self.calls = []
        self.content = "3"
        self.decision = {"operation": "add", "t_valid": "2026-01-01T00:00:00Z"}

    def generate_single(self, prompt, schema=None):
        self.calls.append(prompt)
        return {"json": self.decision} if schema else {"text": self.content}

with TemporaryDirectory() as directory:
    now = ["2026-01-01T00:00:00Z"]
    store = SQLiteDurableStore(Path(directory) / "memory.sqlite", clock=lambda: now[0])
    generator = RecordingGenerator()
    agent = MemoryAgent(generator=generator, durable_store=store, namespace_id="demo")
    first = agent.memorize_durable(
        "retry_limit is 3", idempotency_key="add", meta={"fact_key": "retry_limit"},
    )
    now[0] = "2026-02-01T00:00:00Z"
    generator.content = "5"
    generator.decision = {
        "operation": "update", "target_id": first.memory_id,
        "t_valid": "2026-02-01T00:00:00Z",
    }
    second = agent.memorize_durable("retry_limit is 5", idempotency_key="update")
    now[0] = "2026-03-01T00:00:00Z"
    correction = Proposal(abstract="4", header="Correction", decorated="4", decision={
        "operation": "add", "t_valid": "2026-01-15T00:00:00Z",
        "t_invalid": "2026-02-01T00:00:00Z",
    })
    request = IngestRequest.create(
        "demo", "correct", "retry_limit was 4 since Jan15", {"fact_key": "retry_limit"},
    )
    changes = (ValidityCorrection(
        first.memory_id, "2026-01-01T00:00:00Z", "2026-01-15T00:00:00Z",
    ),)
    third = store.commit_temporal(request, 2, correction, corrections=changes)
    assert store.commit_temporal(request, 0, correction, corrections=changes) == third
    old = store.snapshot_at("demo", valid_at="2026-01-20T00:00:00Z",
                            transaction_at="2026-02-15T00:00:00Z")
    latest = store.snapshot_at("demo", valid_at="2026-01-20T00:00:00Z")
    assert old.complete and old.entries[0].content == "3"
    assert latest.complete and latest.entries[0].content == "4"
    generator.decision = {"operation": "delete", "target_id": second.memory_id}
    agent.memorize_durable("forget current value", idempotency_key="delete")
    assert store.get_entry("demo", second.memory_id) is None
    assert store.snapshot_at("demo", revision=2).entries
    backup = store.backup(Path(directory) / "backup.sqlite")
    restored = SQLiteDurableStore.restore(backup, Path(directory) / "restored.sqlite")
    assert restored.snapshot_at("demo", revision=2) == store.snapshot_at("demo", revision=2)
```

## Historical query and correction contracts

`store.snapshot_at(namespace_id, *, transaction_at=None, revision=None,
valid_at=None, limit=100, repo_id=None, snapshot_id=None, eligible_at=None)` returns
`HistoricalSnapshot`. Transaction timestamp and revision are mutually exclusive.
A supplied revision must be an exact integer in `0..2**63-1`, never bool/float;
a revision beyond the head is rejected. Missing/pruned/pre-migration revisions
report `coverage="unavailable", complete=False`, with no inferred facts. Before a
namespace's first committed observation, history is also unavailable.

`transaction_at` chooses the last revision recorded at or before an aware instant.
Times normalize to fixed UTC microseconds; same timestamp operations are ordered
by revision and a timestamp query selects the last of them. Use revision to select
an earlier same-time operation. New writes reject clocks moving backwards relative
to the last history observation, including a migration baseline. The clock runs
outside the write lock; a stale clock from a racing writer is rejected atomically,
and the caller may retry the same key after the clock advances. Receipt timestamps
retain their legacy string representation. Historical recorded time is the trusted
clock sampled for the transaction, not an assertion about an external wall-clock
commit instant.

`valid_at` applies half-open `[t_valid, t_invalid)` intervals. Missing valid start
means unknown, not creation time. A validity query excludes those facts and sets
`unknown_validity=True, complete=False`. Missing end explicitly means open-ended;
equal endpoints mean empty. Without valid_at, the result includes retained effective
states (active, superseded and expired), with deleted versions excluded. At an old
transaction revision, a subsequently deleted fact retains its then-visible state.
UPDATE's latest parent interval reflects retirement; earlier transactions retain
the original interval. A structured correction appends another version of an active
or superseded fact; it cannot resurrect a tombstone. The normal proposal and all
corrections succeed or roll back together.

`HistoricalSnapshot` includes namespace, selected revision/time, baseline
revision/time, coverage, completeness, truncation, unknown-validity flag and detached
`HistoricalFact` values. Each fact includes the MemoryEntry, immutable version
revision, fact key, repo/snapshot identifiers, single-value declaration and page ID.
`entries` is a convenience tuple. Pages remain in the authority under a qualified
foreign key to `history_pages`; current page APIs intentionally do not expose retired historical pages.
Immutable `history_pages` stores a source copy once per page; its UPDATE/DELETE
triggers protect historical source identity while current page metadata remains
available to existing administrative consumers. `historical_page(namespace_id,
page_id)` returns that retained copy (or None), for trusted hosts only.
All returned models are detached; mutating one never changes stored history.

Optional metadata declares `fact_key` (default the original fact ID),
`single_valued` (default True), and paired `repo_id`/`snapshot_id`. Identity uses
exact case-sensitive UTF-8 identifiers, bounded to 200 bytes with no control
characters. No lossy Unicode, case, path or numeric normalization is performed.
Temporal identity is inherited across ordinary UPDATE and cannot change its scope
or key. Conflicting overlapping single-valued intervals are rejected, including
same-content duplicates; multi-valued intervals require explicit False consistently
for the key/scope. Unknown starts remain unknown instead of inventing non-overlap.
Metadata using these names must meet this documented v2 contract; other metadata
and v1 behavior remain unchanged.

Omitting repository selectors acquires only namespace-level facts. Providing both
selectors acquires exactly that repo/snapshot; unauthorized rows are not loaded and
then filtered in Python. Each namespace remains independently revisioned. Structured
corrections require the same repo/snapshot scope as the request metadata (or inherited
UPDATE scope), and look up every target in the exact namespace.

`commit_temporal(request, expected_revision, proposal, *, corrections=())` accepts
up to 100 `ValidityCorrection(memory_id, valid_from, valid_to)` records, with distinct
targets that are not also changed by the proposal. All fields are revalidated.
A canonical SHA-256 of the proposal and normalized corrections is placed in reserved
structured metadata `_temporal_command`, which participates in the existing request
digest. Same-key altered corrections/proposals raise `IdempotencyConflict`.
Reconcile structured requests by repeating this method with the same body; ordinary
`lookup_receipt(original_request)` lacks that structured digest and will conflict.
Same-key replay precedes the expected-revision precondition. No LLM interprets the
correction list. Ordinary writes continue to record history without this method.

TTL eligibility does not remove observations. `eligible_at` applies explicit aware
metadata `expires_at` and nonnegative finite `ttl_seconds` from fact creation; imported
expired facts also use their t_expired when no
explicit expires_at exists. Supersession/deletion t_expired is a retirement timestamp,
not historical TTL. Without eligible_at, valid historical observations remain
queryable. Malformed legacy eligibility metadata is retained on ingestion; eligibility queries
exclude it and report `unknown_eligibility=True, complete=False`. No automatic TTL
cleanup or physical deletion is introduced.

Limits are 1..1000 facts, 8 MiB aggregate serialized fact bytes before materialization,
1000 effective versions per fact key/scope, and 2,000,000 SQLite VM steps per history
query/key-validation pass. Excess work/bytes raise `CapacityExceeded`. Count truncation
is explicit; never treat incomplete snapshots as complete empty truth. Eligibility
filtering follows bounded acquisition and may yield fewer results when truncated.
History insertion is proportional to changed facts, not the whole corpus. SQLite
acquisition, size checks and result reads share a read transaction. Schema opening
and explicit backup still use the pre-existing full integrity checks.

## Version 1 migration and writer fencing

Fresh databases use exact schema v2. Opening an exact v1 database keeps it v1:
ordinary writes remain supported, and `snapshot_at` reports history unavailable.
Unknown versions/extra or altered schema definitions still fail closed on opening.
Make and retain a backup, stop **all** old writers, then call:

```python
store.enable_history(quiesced=True, recorded_at="2026-09-29T00:00:00Z")
```

The host supplies a truthful baseline recording time. This streams each current
fact once into version history, at the namespace's existing revision. It does not
infer older perspectives from overwritten rows or source creation timestamps.
Migration is atomic, bounded by the processing ceiling, and changes user_version
only after all rows exist. Failure leaves v1 intact. Stop writers through cutover;
quiesced is an explicit host assertion, not a distributed lock. Instances opened on
v1 reject later v2 connections; the writer rechecks the pinned version after acquiring
BEGIN IMMEDIATE, fencing the check/lock race. Old candidate code checks version 1
and rejects v2. Do not run two versions' writers together.

Imports on v2 atomically record an observed baseline at import time, preserving
fact IDs, aliases, outbox and import digest. Earlier history is unavailable.
Missing fact repo/snapshot fields may inherit consistent source page values in the
archived copy. Conflicting/malformed legacy scope remains stored with scope_valid=0,
is excluded before payload acquisition, and yields `unknown_scope=True, complete=False`.
Original current facts/metadata are not rewritten. Reimporting the same source still
does nothing. Migration and import do not invent
historical ingest receipts. New operation receipts remain unchanged in shape.

## Independent outbox consumers

`outbox_events(namespace_id, *, after_revision=0, limit=100)` returns frozen
`ProjectionEvent(namespace_id, event_id, revision, memory_ids)` tuples, including
all delivered reference-consumer events. `get_event(namespace_id, event_id)` is a
qualified lookup. These are read-only; event memory IDs are immutable tuples.
Pagination uses namespace revision, with exact bounded numeric arguments and an
8 MiB batch memory-ID payload ceiling. `pending_events` retains its prior return
shape and pending-only behavior. Every external consumer owns its own receipt
and watermark; the reference consumer's delivered flag is never global delivery.

## Maintained offline history retirement

`compact_history(destination, namespace_id, *, before_revision,
quiesced=False, timeout_seconds=10) -> Path` is a narrowly administrative operation.
The host must authorize maintenance and stop all writers through cutover. It creates
an exclusive NEW v2 database, reads one consistent source snapshot and streams all
retained records into the target in one transaction with original schemas/triggers.
It never drops a trigger or deletes a source row. Timeout is >0 and <=60 seconds;
capacity and deadline failures remove only the newly created failed destination.
An existing destination is always refused.

For the selected namespace it keeps the last version of each fact at the cutoff
plus every later version. Version revision/page IDs are preserved. The cutoff must
be an existing history revision at or beyond the old baseline. Coverage advances
to that revision/time, making earlier queries unavailable. Revision metadata stays
for foreign keys and audit identity; all receipts, outbox, aliases, current facts,
current and immutable historical source pages and other namespaces are preserved. Validate the resulting image and
explicitly switch application traffic while still quiesced. New writes continue
from the original namespace head, not the cutoff.

This primitive retires **version records**, not all source text. Current inactive
facts, source pages, receipts, backups and logs are intentionally retained. C07 must
apply its authorized broader retention policy to those resources and held copies;
this method is not secure erasure. Never silently erase or ship backups/logs.

Layout for integrations: history_coverage has one baseline per namespace;
history_revisions has `(namespace_id,revision,recorded_at,event_id)`; history_versions
has `(namespace_id,memory_id,revision,page_id,fact_key,repo_id,snapshot_id,
single_valued,scope_valid,valid_from,valid_to,status,payload)`. Namespace/version and namespace/page
foreign keys prevent dangling versions. `history_pages` has the same columns as
`pages` but owns immutable source copies. History tables and operations have immutable
UPDATE/DELETE triggers. Ordinary facts remain a current materialized view.

## Backup, rollback and remaining integration gates

The existing SQLite backup API now validates exact v1 and v2 schemas. One backup
contains pages, facts, receipts, outbox and history from the same consistent image.
Restore still requires a new destination and preserves all candidate-era writes in
that image. A v1 binary cannot read v2; forward rollback needs compatible application
code retaining the v2 authority, or a separately reviewed reconciliation/export.
Do not replace it with a pre-migration backup and lose new writes. J09 remains a
separate supported API; no two-file atomicity or automatic J09 publication is claimed.

C01 authorization/composed historical answering, C02 independent projection consumers,
C07 broader retention and C09 application rollback qualification remain their owned
integration gates. No donor code was copied. No provider, production-data, power-loss
or cross-platform qualification is claimed by the local tests.

## Local verification evidence (2026-09-29)

Environment: Python 3.12.12, SQLite 3.51.2, macOS arm64, DELETE/FULL. All input
repositories/databases were disposable fixtures. No installs or live providers.
Commands run from the C03 worktree; `PY` below is exactly
`/Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python`.

* `PYTHONPATH=$PWD $PY -m pytest tests/test_transactional_history*.py tests/test_durable*.py tests/test_scoped_durable.py tests/test_preflight_review.py -q`
  — **259 passed**, three optional-provider import warnings, 29.62 seconds.
* `JITMIND_TEST_NODE=/opt/homebrew/bin/node PYTHONPATH=$PWD $PY -m pytest -q --junitxml=/private/tmp/C03-final-tests.xml`
  — **952 passed, 0 failed, 0 skipped**, three optional-provider import warnings,
  169.66 seconds. This run includes the actual Graft parser and process tests;
  it does not count skipped parser checks as integration. The runtime-only
  untracked `adapters/graft/node_modules` symlink is environment provisioning,
  excluded from C03 source changes and must not be committed.
* `PYTHONPATH=$PWD $PY -m pytest tests/test_transactional_history*.py -q --junitxml=/private/tmp/C03-history-tests.xml`
  — **70 passed, 0 failed, 0 skipped**, three warnings, 13.01 seconds. Includes
  active correction before UPDATE/DELETE, migrated key-conflict rejection, ambiguous
  imported provenance DELETE, and the corrected storage-fault injection helper.
* `$PY -m ruff check jitmind/storage tests/test_transactional_history*.py` — passed.
* `$PY -m ruff format --check jitmind/storage tests/test_transactional_history*.py`
  — all seven files formatted. `git diff --check` — passed.
* Executed the first Python code block from this document against the actual library
  using `exec(compile(code, 'transactional-history example', 'exec'))` — passed.
* Parsed every `jitmind/storage/*.py` with `ast.parse(..., feature_version=(3, 10))`
  — passed. Existing `test_durable_python310.py` metadata-free SQLite errors passed
  in the suite. **No native Python 3.10 runtime or mypy typecheck was run**; the
  minimum-version CI matrix is still required.

Development failures retained for review: the first durable regression run had
1 failure/137 passes because a historical consecutive-revision check broke the
existing signed-64-bit boundary fixture. Increasing revision checks preserve that
contract; unrecorded revision queries report unavailable. An intermediate full run
had 10 failures/921 passes: 2 parser adapter failures, 3 TTL metadata compatibility
regressions, 4 page-metadata mutation compatibility regressions and 1 split-provenance
import regression. The parser acceptance rerun passed all 10 acceptance tests; the
final full run also passed them. TTL is now retained with explicit unknown eligibility,
immutable history_pages preserve original sources independently of current page
metadata, and legacy provenance inherits only consistent missing fields. A later full run had 951 passes/1 failure because the newly added fault-injection
helper did not accept the internal strict keyword; the helper was corrected and
all 70 history tests then passed. An earlier checkpoint full run passed 949 tests
before those last regression additions. No baseline expectations were removed or
edited.

New coverage includes the seven-query frozen oracle, real MemoryAgent add/update/
delete/noop and structured correction, immutable earlier observations, strict offset
and numeric input rejection, namespace/repository isolation, unknown coverage, TTL,
explicit v1 migration/restart and writer fencing, compact-copy and backup/restore,
concurrent writers/receipts, forced history storage failure, and real spawned-process
exit at all seven normal write stages before/after COMMIT. Process exits are not
physical power-loss qualification. The maintained compact-copy path prunes only
history versions, not source text, and requires C07's wider retention policy.

## Final-data-integrity remediation

The later remediation evidence and fingerprints are in
[closure/C03-remediation.md](closure/C03-remediation.md). Its measured results
supersede the earlier checkpoint evidence above.

The exact unpublished v2 layout now includes immutable `history_effects` rows:
`(namespace_id,memory_id,revision,page_id,fact_key,repo_id,snapshot_id,
single_valued,scope_valid,valid_from,valid_to,status,payload_digest,page_digest)`.
The key is `(namespace_id,memory_id,revision)` and the namespace/revision references
`history_revisions`. Effects deliberately do not reference version rows: a bounded
metadata-only bidirectional comparison detects missing effects/versions or changed
indexes before any history scope selection. Selected fact and original-page bytes
must match the SHA-256 recorded in the same write transaction, and their decoded
identity, scope, status and intervals must agree with the indexes before return.
Failures expose generic storage error codes, never private payload text. This is
inconsistent-data detection, not cryptographic authenticity against a database
rewriter. Unselected payloads are not exhaustively hydrated or authenticated.
The VM ceiling applies to namespace inventory checks; a large archive can return
`CapacityExceeded` instead of a completeness claim. No new full-payload startup
scan is introduced; the existing SQLite structural checks remain.

The compact-copy path checks source inventory consistency and copies effect rows
for exactly the retained version rows, including retained baseline states whose
original revision precedes the new coverage cutoff. All source identities and
receipts remain unchanged. C07 must include `history_effects` in schema-aware
maintenance. Earlier experimental v2 schemas lacking this inventory are rejected
by exact-schema validation; this is not a silent upgrade or certification of an
old image. Preserve those images/backups for supervisor-reviewed reconciliation.
Published v1 opens unchanged and uses the explicit, atomic history migration.

Migration/import now validates all known-scope baseline fact keys before commit.
Overlapping single-valued intervals (or inconsistent single-valued declarations)
raise `InvalidRequest` and roll back the entire migration/import. This intentionally
moves the old query-time contradiction failure to publication time. Adjacent
half-open intervals remain valid. Ambiguous legacy provenance remains explicit
unknown scope, with no fact returned under an unqualified or guessed selector.

Original source-page `t_valid`/`t_invalid` remain historical evidence. Corrected
fact intervals govern history selection. Independent original-page AND fact
`expires_at`/`ttl_seconds` constrain `eligible_at`, with TTL measured from fact
`t_created`. Malformed expiry metadata excludes the result and marks unknown
eligibility. Current mutable page metadata does not rewrite original evidence.

`get_event` and `historical_page` validate exact schema in the same bounded read
transaction as a SQL byte-size preflight, before fetching payloads. Limits are
262144 serialized UTF-8 bytes per event and 1048576 bytes per historical page.
Events also validate revision range, identifiers and the 1000-ID ceiling. History
snapshot acquisition now includes original-page bytes in the aggregate 8 MiB
ceiling, plus per-fact/page limits. Public method signatures and receipt shapes
are unchanged.

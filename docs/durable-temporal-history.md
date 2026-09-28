# Durable strict temporal history

`jitmind.temporal_history.SQLiteTemporalHistory` persists the complete knowledge
perspectives defined by J03's `TemporalPerspective` and `TemporalFact`. It is an
opt-in, local SQLite backend for `TemporalHistoryBackend`. Existing legacy query
APIs and old-format files are unchanged. There is no provider call, generation
step, or other product dependency in this module.

A **structured temporal writer is required**. Each publication supplies the entire
knowledge perspective at its `recorded_at`, including all historical effective
intervals known at that time. Generic text updates do not express those semantics.
There is no automatic MemoryAgent bridge, legacy-history inference, or conversion
of J02 mutable fact rows into historical perspectives. J02's current rows and
ID-only events cannot recover intervals that a later correction replaced.

## Example

```python
from jitmind.scope import ScopeAuthority
from jitmind.scoped_temporal import TemporalFact, TemporalPerspective
from jitmind.temporal_history import DurableTemporalOracle, SQLiteTemporalHistory

authority = ScopeAuthority()  # trusted host authenticates the caller first
# Administrative grant/context methods must not be exposed as caller RPCs.
authority.grant("alice", "team/a", ["repo-a"])
scope = authority.context("alice", "team/a", "request-1")
history = SQLiteTemporalHistory("temporal-history.sqlite")
oracle = DurableTemporalOracle(authority, history)

perspective = TemporalPerspective(
    "2026-01-01T00:00:00Z",
    (TemporalFact("retry-1", "retry_limit", "3", "2026-01-01T00:00:00Z"),),
)
receipt = oracle.publish(scope, perspective, "operation-1", expected_revision=0)
assert receipt.revision == 1
assert oracle.publish(scope, perspective, "operation-1") == receipt
assert oracle.query_as_of(scope, "2026-01-20T00:00:00Z")[0].content == "3"
```

`publish(scope, perspective, idempotency_key, expected_revision=None)` returns a
frozen `PublicationReceipt(operation_id, payload_digest, revision)`. Its operation
ID is a UUID and its digest is SHA-256 of the canonical perspective body. Revisions
are one-based; an empty namespace has revision zero. Supplying `expected_revision`
enables optimistic CAS. Omitting it appends against the current head under the
SQLite writer lock; it does not merge stale perspectives. For callers that derive
a new perspective from a prior read, supply that read's revision.

Idempotency keys are exact, namespace-local identifiers. The same key and canonical
body return the original receipt after a restart, after response loss, and after
later publications. A changed body raises `TemporalIdempotencyConflict`. On replay,
the saved result takes precedence over a stale `expected_revision`; that revision
is a first-execution precondition and is not part of the payload digest. Fact tuple
order is preserved and participates in the digest. UTC-equivalent timestamps have
the same canonical representation; text and identifiers are not normalized or
converted into file paths. Distinct namespaces can reuse every fact ID and key.

If COMMIT or post-commit acknowledgement fails, the wrapper performs a fresh,
read-only same-key reconciliation. It never retries the insert implicitly. If it
cannot prove the result, it raises `TemporalAcknowledgementUncertain`; retry the
**same key and body** after storage is available. A failure before COMMIT dispatch
rolls back. Exceptions contain no content, identifiers, or host paths. The trusted
`fault_hook` constructor argument exists for tests only.

The original protocol remains available:

```python
history.append_perspective("team/a", expected_revision=1, perspective=next_perspective)
all_perspectives = history.read_perspectives("team/a")
```

It retains the `None` return and strict CAS behavior. It is non-idempotent and has
no acknowledgement receipt; the wrapper is recommended for publication. The raw
backend, status, pagination, diagnostics, and backup APIs are trusted-host APIs,
not authorization boundaries for untrusted callers.

## Authorization and the acceptance boundary

`DurableTemporalOracle` requires an actual `ScopeAuthority` and its process-local,
issued `ScopeContext`. A lookalike dataclass, serialized context, context issued by
a different authority, revoked capability, or unauthorized fact repository cannot
publish. Namespace and repository grants are checked before connection acquisition,
after acquiring the writer lock, immediately before COMMIT dispatch, and before
returning a receipt. Replays and uncertain-commit reconciliation recheck current
authority. Queries delegate to J03's `StrictTemporalOracle`, including repository
snapshot selection and checks surrounding history acquisition and delivery.

The successful last authority check immediately before COMMIT dispatch is the
publication acceptance boundary. Revocation after that check can race with COMMIT;
it cannot undo an accepted committed write. The final authority check suppresses
a receipt if revocation is observed before return. Hosts that need stronger
revocation/dispatch serialization must coordinate it above these APIs. There is
no claim of distributed authorization, or undo after response delivery.

## Persistence, validation, and bounds

The dedicated schema uses `jth_meta`, `jth_perspectives`, and `jth_operations`.
Each namespace/revision has exactly one immutable full payload. Triggers disallow
updates/deletes, enforce contiguous appends and increasing recorded timestamps,
and bind operation digests to the corresponding revision. Unique constraints and
foreign keys protect receipt/revision relationships. SQL triggers are not a defense
against an attacker who can rewrite the database and schema themselves.

All timestamps are timezone-aware and stored as fixed-width
`YYYY-MM-DDTHH:MM:SS.ffffffZ`, so SQL ordering agrees with UTC ordering. Recorded times
must strictly increase per namespace. Equal instants with different offsets still
conflict when published under different keys. Queries first select the last
perspective recorded at or before transaction time, then apply half-open valid
intervals (`valid_from <= valid_at < valid_to`). Without transaction time, they use
the latest complete perspective. Publication never changes earlier valid intervals.

Writes reconstruct and validate the frozen dataclasses, including ones modified
through `object.__setattr__`. Reads reject duplicate JSON keys, extra/missing fields,
non-finite numbers, numeric/huge-integer substitutions, invalid intervals,
conflicting single-valued overlaps, duplicate fact IDs, noncanonical timestamps,
and digest mismatches. Startup performs a streaming schema/data audit with a
10-second cooperative deadline. Large archives that cannot be audited within this
budget fail explicitly; this release has no incremental audit index. Validation
is per record and does not hold the whole archive in memory.

| Setting | Default | Allowed maximum |
| --- | ---: | ---: |
| `max_payload_bytes` | 1,048,576 UTF-8 bytes | 1,048,576 |
| `max_facts` per perspective | 10,000 | 100,000 |
| `max_read_count` per acquisition | 1,000 perspectives | 100,000 |
| `max_read_bytes` per acquisition | 16,777,216 UTF-8 bytes | 67,108,864 |
| `busy_timeout_ms` per SQLite lock wait | 250 ms | 5,000 ms |

Payload construction checks cumulative size while serializing facts. Full-history
reads precheck count, aggregate UTF-8 bytes, and largest payload before fetching
payload rows, in the same consistent read transaction. Exceeding a limit raises
`TemporalCapacityError`; **no oldest evidence is silently dropped**. J03's strict
query protocol therefore also fails when the full history exceeds its configured
read limits. The wrapper can continue appending without reading the full archive.
Capacity limits bound acquisition, not lifetime disk consumption.

`status(namespace_id)` returns `revision`, `perspective_count`, `payload_bytes`, and
`latest_recorded_at`. `read_page(namespace_id, after_revision=0, limit=100,
through_revision=None)` returns a frozen `PerspectivePage` with `perspectives`,
`next_revision`, `head_revision`, and `has_more`. Use the first page's
`head_revision` as `through_revision` for subsequent pages to pin the upper bound
while new perspectives arrive. Each page independently applies aggregate byte and
count limits. Pagination is for explicit host workflows; it is not a truncated
replacement for the full-history protocol.

## SQLite runtime and J02 coexistence

Connections are operation-local and use `foreign_keys=ON`, `synchronous=FULL`, and
the bounded busy timeout. Journal mode must already be DELETE; a newly created
SQLite file uses DELETE. This implementation **does not change journal mode**.
WAL is unsupported, including on patched runtimes. This avoids relying on affected
Mac SQLite 3.51.2 WAL behavior; no runtime version is claimed WAL-qualified here.
`diagnostics()` returns actual connection journal mode, synchronous level, foreign
key setting, busy timeout, SQLite version/source ID, and the module schema version.

J02 currently validates its entire schema by exact equality. Adding J09 tables to
that file would break J02 reopening. Consequently J09 v1 requires its **own file**,
rejects unrelated tables, and leaves an existing J02 database unchanged. The
prefixed schema reserves a clear boundary for a future coordinated integration.
Independent commits across J02 and J09 databases are not atomic. Co-locating files
on a host does not create a transaction across them.

Existing empty, foreign, malformed, or future-version files fail closed. Only an
absent path is initialized. Read/write operations use SQLite `mode=ro`/`mode=rw` and
do not recreate a database that disappears after initialization. Schema metadata,
DDL, integrity, stored payloads, and receipts are audited at startup; operations
recheck schema and validate the payloads they acquire. No repair or migration is
performed implicitly. Use a trusted local directory; file replacement by a hostile
process and cryptographically authenticated storage are outside this boundary.

## Backup, restore, and retention

```python
backup_path = history.backup("new-backup.sqlite", timeout_seconds=10)
restored = SQLiteTemporalHistory.restore(
    backup_path, "new-restored.sqlite", timeout_seconds=10
)
```

`SQLiteTemporalHistory.restore` is the only classmethod intended as public API.
Backup/restore use the SQLite backup API with 128-page progress steps, a cooperative
deadline (default 10 seconds, maximum 60), exclusive destination creation, and a
streaming integrity/data audit. Existing destinations are never overwritten.
Failure removes only the newly created partial destination when the filesystem
permits cleanup. Source databases are opened read-only. No active WAL main file is
copied. Restoring an older backup never replaces or rolls back the candidate/live
file; writes after that backup remain in the original database. A new backup taken
after candidate-era publication includes those perspectives and receipts. The
backup API also produces a consistent snapshot when publications occur during
copying. The deadline is cooperative, not a hard real-time bound on OS I/O; the
returned restored object's startup audit has its own 10-second budget.

Default retention is indefinite. TTL/`expires_at` controls query eligibility only;
old perspectives and historical facts are not automatically deleted when facts
expire, become superseded, or disappear from the latest perspective. This retains
potentially sensitive historical content. A physical-forgetting policy requires
an explicit host-level archival/deletion design covering live storage, backups,
and other copies. This release offers no per-fact purge, and makes no claim that
all backup copies have been erased.

## Verification and integration provenance

J09 owns only `jitmind/temporal_history.py`,
`tests/test_temporal_history_sqlite.py`, and this document. It requires J03's
finished `scope.py`, `scoped_research.py`, and `scoped_temporal.py` interfaces. Local
qualification copied those files and J03's four `test_scoped_*.py` files unchanged.
J02's `storage/{__init__,models,sqlite,migration}.py` were copied unchanged only to
run J03's durable-adapter tests and J09's separate-file coexistence test. They are
not J09 implementation changes or runtime dependencies. The coexistence test skips
when the optional J02 package is absent; temporal storage itself does not import it.

The new tests use real local SQLite files, the exact seven-query synthetic oracle
plus latest-perspective query, reopening and a spawned child process, offset and
half-open boundaries, namespace isolation, CAS contention, canonical idempotency,
manually bypassed models, malformed JSON/schema/source data, aggregate read bounds,
immutable SQL constraints, before/after-commit faults, uncertain-COMMIT recovery,
revocation/forgery, concurrent backup publication, and candidate-era restore.
No live providers, production secrets, dependency installation, or legacy-data
conversion are involved. Final integration qualification remains with the owner
when J03 and J09 are stacked.

Local qualification used
`/Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python`
with `PYTHONPATH=$PWD` on Python 3.12. The three separate pytest runs passed
38 original tests, 64 copied J03 tests, and 37 new J09 tests (139 distinct tests;
reruns are not added to that total). Each run emitted the three expected optional
retriever warnings. Ruff lint, format, compilation, and Python 3.10 grammar checks
passed for the owned Python files. Mypy was unavailable (`No module named mypy`);
no type-check success is claimed and no package was installed.

SQLite reported version `3.51.2` and source ID
`2026-01-09 17:27:48 b270f8339eb13b504d0b2ba154ebca966b7dde08e40c3ed7d559749818cb2075`.
All 11 copied dependency files were verified byte-identical to their sibling
worktrees. Interface/backend source SHA-256 values used for qualification:

| Source | SHA-256 |
| --- | --- |
| J03 `jitmind/scope.py` | `2762e46016b2f4e1a6ea6d95208954724f1cfef8e4e59812a19fc06e97bdf34a` |
| J03 `jitmind/scoped_temporal.py` | `cb463c776e743af32c55512a7cdb921f1134a9909e616af235c851d44db76326` |
| J03 `jitmind/scoped_research.py` | `d0143f36b446d4384c5052f726b0ed5f4d1145738d97dc3ae5f3ec409a155a7f` |
| J02 `jitmind/storage/sqlite.py` | `33dc5b83d28e6133d905308404893d7a0a9ab5bde112f2412d4d4132d238a190` |

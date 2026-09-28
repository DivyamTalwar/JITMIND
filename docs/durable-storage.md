# Opt-in durable memory and page storage

`SQLiteDurableStore` is a single local SQLite authority for memory facts, pages,
operation receipts, and a projection outbox. Existing `MemoryAgent` construction
continues to use its existing JSON stores. There is no automatic migration,
dual write, global SQLite change, or added dependency.

This backend is for cooperative processes on one host with a local filesystem.
Network filesystems, multi-host access, provider integrations, and physical
power-loss qualification are outside its tested scope. A successful receipt
means SQLite COMMIT completed using the verified settings below; it is not an
acknowledgement from a graph, vector index, or profile service.

## Run a complete example without a provider

From the repository root, run this with your installed Python environment:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from jitmind.agents.memory_agent import MemoryAgent
from jitmind.storage import SQLiteDurableStore

class ExampleGenerator:
    def generate_single(self, prompt, schema=None):
        if schema is not None:
            return {"json": {"operation": "add", "importance": "long"}}
        return {"text": "The user prefers tea."}

with TemporaryDirectory() as directory:
    db = SQLiteDurableStore(Path(directory) / "memory.sqlite3")
    agent = MemoryAgent(
        generator=ExampleGenerator(), durable_store=db, namespace_id="demo"
    )
    receipt = agent.memorize_durable(
        "I prefer tea.", idempotency_key="request-001", meta={"source": "chat"}
    )
    assert receipt == agent.memorize_durable(
        "I prefer tea.", idempotency_key="request-001", meta={"source": "chat"}
    )
    update = agent.memorize(
        "I prefer tea.", idempotency_key="request-001", meta={"source": "chat"}
    )
    assert update.new_page.meta["page_id"] == receipt.page_id
    assert db.drain_outbox("demo", limit=10) == 1
    assert db.projected_entries("demo")[0].id == receipt.memory_id
    backup = db.backup(Path(directory) / "backup.sqlite3")
    restored = SQLiteDurableStore.restore(backup, Path(directory) / "restored.sqlite3")
    assert restored.get_page("demo", receipt.page_id).content == "I prefer tea."
    print(db.status("demo"))
    print(db.diagnostics())
```

For this worktree's isolated interpreter, use
`PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python`.

## Public integration contract

All public types below are exported by `jitmind.storage`.

- `SQLiteDurableStore(path, *, journal_mode="DELETE", busy_timeout_ms=250,
  commit_retries=2, clock=None, fault_hook=None)` owns one database. The parent
  directory must exist. Connections are short lived; no close method is needed.
- `MemoryAgent(..., durable_store=db, namespace_id="tenant")` explicitly chooses
  this authority. Combining it with `memory_store`, `page_store`, or `dir_path`
  is rejected. Bind a separate agent to each namespace. The namespace is an
  application authorization boundary, not a substitute for caller authentication.
- `agent.memorize_durable(message, *, idempotency_key, meta=None, user_id=None,
  max_replans=2) -> DurableReceipt` is the retryable acknowledgement API.
- `agent.memorize(message, meta=None, user_id=None, *, idempotency_key=None)`
  retains the `MemoryUpdate` result. Omitting the key generates a random key;
  supply your own key if you need to reconcile an interrupted request. The
  committed receipt is also in `update.debug["durable_receipt"]`.
- `IngestRequest.create(namespace_id, idempotency_key, message, meta=None,
  user_id=None)` validates and detaches request data. `db.ingest(request,
  propose, *, max_replans=2, context_limit=100)` calls
  `propose(NamespaceSnapshot) -> Proposal` outside SQLite transactions.
- `NamespaceSnapshot` contains `revision`, a tuple of detached active `MemoryEntry`
  values, and `truncated`. Context is bounded to recent entries. A `Proposal`
  has `abstract`, `header`, `decorated`, and a `decision` matching the existing
  `MemoryOperationDecision` fields, with unknown top-level decision fields rejected.
- `db.commit_proposal(request, expected_revision, proposal)` is the explicit CAS
  API for integrations that own planning. `db.lookup_receipt(request)` reconciles
  a key and checks its digest without invoking any generator.

`DurableReceipt` has exactly `namespace_id`, `idempotency_key`, `request_digest`,
`operation` (`add/update/delete/noop`), `page_id`, `memory_id` (null for delete/noop),
`revision`, `event_id`, and `committed_at`. It contains no remembered payload.
New page IDs and new fact IDs are random UUID strings. UPDATE supersedes the
previous fact and creates a new fact UUID with `version_of` pointing to its
predecessor. DELETE retains its fact ID, tombstone status, and deletion revision.
NOOP still commits the input page, receipt, revision, and an empty outbox event.
Every operation, including NOOP, advances its namespace revision once.

Receipts replay exactly after reopening. `memory_update(receipt, limit=1000)`
constructs a compatibility view from **current** authority. Its state is bounded
and reports `debug["state_truncated"]`. The additive
`receipt_content(receipt) -> ReceiptContent` API checks the exact committed receipt
and reads visibility in one SQLite snapshot. Its frozen result has `receipt`,
`status: Literal["available", "retired", "unavailable"]`, and `page: Page | None`:

| Receipt | Content status | Ordinary page |
| --- | --- | --- |
| ADD/UPDATE whose source fact is active | `available` | Detached current page |
| Prior ADD/UPDATE after fact retirement | `retired` | `None` |
| Successful NOOP (including a targeted duplicate) | `unavailable` | `None` |
| Successful DELETE | `unavailable` | `None` |

NOOP/DELETE input pages are immutable administrative history. They never acquire
ordinary visibility merely by being unlinked, even while a NOOP target is active.
`memory_update()` and durable `MemoryAgent.memorize()` acknowledge all these
successful operations with `MemoryUpdate`. For unavailable/retired content,
`new_page` has empty header/content and exactly `page_id`, `redacted=True`, and
`content_status` in its metadata. No original metadata or decoration is copied.
`debug["content_status"]` reports the same status; `debug["durable_receipt"]`
retains the full payload-free identity. This is not a failed DELETE acknowledgement.
The state contains only currently active facts, never a historical snapshot.
`memorize_durable()` and same-key retries always return the original receipt,
without regenerating or resurrecting text. Fabricated/mismatched receipts raise
`InvalidRequest`. Availability describes the read snapshot; applications must
recheck authority before later using cached payloads.

Existing `IngestionPipeline` callers continue to receive `MemoryUpdate`, including
for NOOP/DELETE. Its page-based duplicate cache sees only visible pages. J03's
`DurableScopedBackend` expands source pages from active snapshot entries using
`get_page()` and verifies revision afterward; those public contracts are unchanged.
Legacy-only `MemoryAgent` behavior is unchanged.

## Temporal mutation rules

Validity intervals use `[t_valid, t_invalid)` when both endpoints are known;
equal endpoints are an explicit empty interval. Transaction timestamps
(`t_created`, `t_expired`) are a separate axis. Comparisons use aware instants,
including timezone offsets; original strings are retained.

UPDATE retires the parent at the proposed successor's `t_valid`, or at the
transaction clock when that value is absent. If that end precedes the parent's
known `t_valid`, the entire operation raises `InvalidRequest` **before any row
writes**, including the namespace revision. No retroactive repair is inferred.
Retirement before the parent's `t_created` also raises `InvalidRequest`.

DELETE of a future-valid fact is legal: when `t_valid` is later than the transaction
clock, retirement sets `t_invalid = t_valid`, explicitly retracting the entire
future interval. This applies even if it previously had a later scheduled end.
The fact is `deleted`, with `t_expired` recording the actual deletion clock.
For other facts, DELETE preserves a supplied end or uses the deletion clock.
Already inverted or malformed stored intervals raise `StorageFailure`; neither
UPDATE nor DELETE repairs them. Ordinary visibility remains lifecycle-based
(`active`); this backend does not automatically evaluate validity or TTL windows.

## Generation and retries

The backend checks for a matching receipt, captures a namespace revision and
bounded state in a read transaction, generates outside the transaction, then
uses `BEGIN IMMEDIATE` to check both the operation key/digest and revision.
Page, lifecycle changes, receipt, and outbox are committed together. Stale
proposals replan at most twice by default (configurable 0–5). Concurrent initial
requests may generate speculative proposals; only one same-key proposal commits.
A committed replay never calls the generator. No graph/profile side effects run
in this path, including when graph/profile objects are supplied to MemoryAgent.

The key is scoped by `(namespace_id, idempotency_key)`. SHA-256 covers canonical
JSON of the **exact** message, metadata, and user ID. Metadata dictionary order
is irrelevant; whitespace changes in the message are distinct requests.
A changed payload under an existing key raises `IdempotencyConflict`.

`AcknowledgementUncertain` after COMMIT means retry/reconcile the **same key and
payload**. The same reconciliation rule is safe after an interrupted process or
storage failure. Never allocate a fresh key merely because acknowledgement was
lost. SQLite busy waits are bounded (default 250 ms per statement). Busy COMMIT
is retried at most twice, within the same transaction, without rerunning the
model. An exhausted COMMIT attempt explicitly rolls back an active transaction.
There is no automatic generator retry on `StorageBusy`.

`clock()` returns an aware ISO-8601 string. `fault_hook(stage)` supports trusted
application tests only: `before_transaction`, `after_page_insert`,
`after_fact_insert`, `after_outbox_receipt`, `before_commit`, `after_commit`,
`before_projection_commit`, `after_projection`, `before_import`,
`after_import_pages`, and `before_import_commit`. Neither hook comes from a
model decision or stored metadata.

## Reads, visibility, and projection delivery

- `snapshot(namespace_id, limit=100)` returns bounded recent active facts.
- `get_entry(namespace_id, memory_id)` and `get_page(namespace_id, page_id)`
  return detached current payloads or `None`. Pages require an active authoritative
  fact linked by namespace and page ID, with no retired fact sharing the page.
  Retired and unlinked pages (including imported orphan/admin pages) are hidden.
  This predicate is shared by receipt reconstruction and all page adapter reads.
- `visible_entries(namespace_id, memory_ids)` validates up to 1,000 IDs, filters
  inactive/unknown/cross-namespace hits, removes duplicates, and returns current
  SQLite payloads in input order. External retrieval consumers must call this
  immediately before using or returning hit payloads; do not trust cached text.
- `get_entry(..., include_inactive=True)` is an explicit administrative history
  API. It exposes retained historical payloads. This is logical deletion, not
  secure erasure of history, files, backups, or application-held copies.
- `pending_events(namespace_id, limit=100, after_revision=0)` returns bounded
  pending events. `status(namespace_id)` is read-only and reports revision,
  page/fact/operation counts, pending count, and a contiguous delivery watermark.
- `deliver_event(namespace_id, event_id)` applies the built-in reference projection
  and marks that event delivered in one short transaction. `drain_outbox(namespace_id,
  limit=100)` processes one bounded batch. `projected_entries(namespace_id,
  limit=100, after_id="")` queries the projection with current-authority filtering.

The reference projection is real SQLite state; it records `(namespace, memory ID,
revision, status)` and joins authority for payloads. Delivery reads current facts,
so delivering an old upsert after a newer delete cannot resurrect it. Duplicates
are idempotent. The watermark advances only past contiguous delivered revisions,
not simply to the largest observed revision. Primary commits do not depend on
projection availability. Outbox records contain IDs rather than copied fact text.
The delivered state is specifically for this reference consumer. An external
consumer requires its **own** delivery ledger/watermark and idempotent apply API;
the existing delivered flag does not claim delivery to arbitrary services.

`DurableMemoryAdapter(db, namespace)` supports bounded `load()`,
`get_entries(include_inactive=False, limit=100, after_id="")`, and
`get_entry_by_id(id)`. `DurablePageAdapter` supports `get(string_id)`, bounded
`load()`, and `list_pages(limit=100, after_id="")`. Adapter `load()` raises
`CapacityExceeded` above 1,000 records rather than silently returning an unbounded
list. `add()`/`save()` on these adapters are deliberately rejected: separate store
writes would break the transaction contract. Arbitrary MemoryStore/PageStore
pairs are not atomic. Legacy integer/list-index retrievers and ResearchAgent need
explicit scoped string-ID integration; these adapters do not reinterpret indices.

## Database format and settings

Schema version 1 uses `PRAGMA application_id = 0x4A49544D` and
`PRAGMA user_version = 1`. Unknown versions, identities, or SQL schema definitions
are refused. Existing corruption is never replaced with an empty database.
SQLite integrity and foreign-key checks run on opening an existing authority.

| Table | Key | Stored values |
| --- | --- | --- |
| `namespaces` | `namespace_id` | Monotonic revision |
| `pages` | `(namespace_id, page_id)` | Page JSON, creation revision |
| `facts` | `(namespace_id, memory_id)` | Page FK, lifecycle status, latest revision, complete MemoryEntry JSON |
| `operations` | `(namespace_id, idempotency_key)` | Request digest, committed receipt JSON |
| `outbox` | `(namespace_id, event_id)`; unique namespace/revision | Changed memory IDs JSON, `pending/delivered`, delivery timestamp |
| `aliases` | `(namespace_id, legacy_page_id)` | Canonical page FK |
| `projection` | `(namespace_id, memory_id)` | Latest applied authoritative revision and status |
| `imports` | `namespace_id` | Source digest and imported page/fact counts |

Each connection enables and verifies `foreign_keys=ON` and `synchronous=FULL`.
`diagnostics()` queries the application connection for actual journal mode,
synchronous value, foreign-key value, schema version, SQLite version, and SQLite
source ID. This environment reports SQLite 3.51.2; tests use DELETE rollback
journaling. WAL is opt-in and rejected below 3.51.3, including otherwise fixed
older backport families. There is no vendor-backport override in this implementation.
This deliberately narrow WAL policy follows [SQLite's WAL-reset bug documentation,
section 11](https://www.sqlite.org/wal.html#walresetbug); no WAL qualification is
claimed for the local 3.51.2 library.

## Validation and safe errors

Messages allow at most 131,072 canonical JSON bytes. Metadata allows at most
65,536 canonical JSON bytes and 2,048 JSON nodes. Identifiers allow at most 200
UTF-8 bytes with no control characters. JSON depth is at most 16. Proposals are
bounded to 524,288 JSON bytes. Read/event limits must be integers from 1 to 1,000.
These are hard limits, not truncation policies.

Revision cursors and expected revisions must be exact Python integers in
`0..2**63-1`; booleans/floats and out-of-range values raise `InvalidRequest` before
SQL binding. New writes at revision exhaustion also raise `InvalidRequest`;
already committed same-key operations remain reconcilable. Counts retain their
smaller documented bounds, including on replay. String ID cursors accept `""`
or a valid identifier, never integers. Administrative `include_inactive` flags
must be booleans. Manually constructed staged counts must be integers in `0..1000`.

NaN/infinity, arbitrary Python objects, non-string dictionary keys, cycles,
unsafe prototype keys, control characters in keys, and invalid Unicode are
rejected. Metadata is copied through canonical JSON; mutable defaults are not
shared. Reserved page metadata keys are `page_id`, `memory_id`, `decorated`,
`t_observed`, `t_valid`, `t_invalid`, `user_id`, and `_durable`. Custom metadata is
preserved on both page and new fact. `_durable` stores the validated decision and
revision for explicit future integrations. Generated decisions cannot select
runtime hooks, the namespace, the idempotency key, or SQL operations.

All backend domain errors inherit `DurableError` and expose a stable `.code`.
Their messages contain only the safe code, never the database path or raw payload:
`InvalidRequest` (`invalid_request`), `ProposalFailure` (`proposal_failure`),
`IdempotencyConflict`, `StaleRevision`, `StorageBusy`, `StorageFailure`,
`SchemaMismatch`, `UnsafeJournal`, `AcknowledgementUncertain`, `MigrationError`
(`invalid_legacy_source`), `CapacityExceeded` (`bounded_read_exceeded`), and
`TargetNotFound` (`target_not_active`). Other listed codes are the snake-case class
name. Pydantic model construction itself can raise validation errors; applications
should not expose raw Pydantic errors to untrusted clients.

## Explicit legacy migration and cutover

Stop **all** legacy writers before acquisition and keep them stopped through
cutover. The `quiesced=True` flag is a caller assertion, not an interprocess lock.
Staging performs a second byte-for-byte bounded read to detect source changes
around acquisition; it cannot prevent a writer from changing files later.

```python
from pathlib import Path
from jitmind.storage import SQLiteDurableStore, stage_legacy

# These paths are operator choices; stop legacy writers before running this.
source = Path("legacy-data")
staged = stage_legacy(source, quiesced=True)
candidate = SQLiteDurableStore("candidate.sqlite3")
report = candidate.import_staged(staged, namespace_id="tenant")
assert report["page_count"] == staged.page_count
assert report["fact_count"] == staged.fact_count
print(report, candidate.status("tenant"))
# Explicit cutover: construct MemoryAgent(durable_store=candidate, namespace_id="tenant", ...)
# Keep legacy writers stopped. There is no dual-write mode.
```

`stage_legacy(source_dir, quiesced=False, max_bytes=16777216, max_records=1000)`
reads exactly `advanced_memory_state.json` and `pages.json`, supporting the actual
legacy page list and `{ "pages": [...] }` formats. `StagedLegacy` contains immutable
`source_digest`, `source_json`, `page_count`, and `fact_count`. Import revalidates
staged contents. The source byte cap is aggregate; the record cap applies to each
list. Larger stores require a separately reviewed migration path.
Manually constructed `StagedLegacy.source_json` has the same 16 MiB UTF-8 envelope
cap, including whitespace, enforced before hashing or `json.loads`. Character
length is checked before encoding; UTF-8 byte length is checked before parsing.

Each page must have an explicit string or nonnegative integer `meta.page_id`.
Aliases are not guessed from list position. Numeric `0` and string `"0"` are the
same alias, so duplicates are rejected. Every fact requires a unique explicit ID,
source page, original status/tier/creation/observation fields, and valid temporal
values. Missing/dangling/ambiguous page provenance, duplicate IDs or JSON keys,
unknown fields, cycles, branching version chains, invalid timestamps, and reversed
time intervals are rejected with `MigrationError`. Legacy files containing
naive timestamps or incomplete provenance need explicit repair outside this tool.
No partial identity guesses or silently dropped fields are accepted.

Version chains require a non-active parent, parent creation no later than child
creation, and, when supplied, parent retirement (`t_expired`) no later than child
creation. Each supplied retirement must be at or after its own creation. When
parent `t_invalid` and child `t_valid` are both supplied, parent validity must end
no later than child validity begins. Gaps and equal boundaries are accepted;
missing optional endpoints remain absent and are not inferred. Transaction time
is never compared to validity time. These checks run both at staging and again
at import, before any writes. Legacy records produced by separate clocks can
contain retirement after successor creation; even small contradictions are
rejected, without an undocumented tolerance or timestamp repair. Operators must
resolve such inconsistent sources explicitly before import.

Migration preserves fact IDs, custom metadata, and all temporal/version fields.
It allocates canonical page UUIDs and maps original IDs through `aliases`;
`resolve_page_alias(namespace, legacy_id_as_string)` returns that mapping.
Page metadata's `page_id` and fact `source_page_id` change to the canonical ID.
Import requires an empty namespace and commits the whole import, alias map,
manifest, and one revision-1 projection event atomically. It creates no invented
ingest receipts for historical requests. Reimporting the same digest is a no-op,
including after subsequent writes; a different import into a populated namespace
is refused. Legacy source files remain untouched.

## Backup, export, and recovery

`db.backup(new_path, timeout_seconds=10)` exports a consistent database using the
SQLite backup API, including receipts, aliases, history, pending events, and
writes committed after migration. It never copies just the live main DB file.
The timeout is bounded (maximum 60 seconds). `SQLiteDurableStore.restore(backup,
new_destination)` validates the backup and uses the backup API to create a new
store. Existing destinations are always refused, so restoration cannot silently
roll a live candidate back to an older snapshot. There is no pickle deserialization.

Before an intentional rollback/cutover, quiesce candidate writers and export its
latest state. Keep that export and reconcile any candidate-era writes before
switching application traffic. Restoring an older backup into a **new** file does
not merge those writes automatically. The original candidate stays available.

## Verification and boundaries

Tests use real SQLite connections and spawned processes with bounded barriers,
joins, and child cleanup. They cover faults/process exits before a transaction,
after page/fact/outbox insertion, after COMMIT before acknowledgement, and after
projection; concurrent same-key duplicates and payload conflicts; namespace
isolation; stale revision replanning; busy COMMIT rollback; public MemoryAgent
replay; strict migration; and backup/reopen/restore with candidate-era writes.

No live provider, graph, vector, or production secret is used. Retention decay,
automatic TTL cleanup, graph conflict resolution, and profile updates remain
legacy behaviors and are not run automatically by this durable backend.
Authorization, encryption, administrative historical access controls, external
consumer delivery ledgers, scoped ResearchAgent integration, and operational
power-loss/filesystem testing remain integration responsibilities.


## Python 3.10 error-classification compatibility

The initial Ubuntu/Python 3.10 CI exposed missing `sqlite3.SQLITE_BUSY` and
`SQLITE_LOCKED` aliases in exception translation. The backend now prefers
numeric SQLite primary result codes when available and uses only exact native
lock-message forms for metadata-free OperationalError instances. Disk-full,
corrupt-database and unrelated failures remain sanitized StorageFailure results.
The existing busy-commit rollback/no-regeneration contract is unchanged.
`tests/test_durable_python310.py` simulates missing metadata on newer Python;
the existing CI matrix separately exercises the actual minimum runtime.

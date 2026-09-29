# Persistent graph and vector projections

`jitmind.projections` supplies two opt-in, local SQLite consumers of the durable
fact authority. The vector consumer stores fixed-dimensional float vectors and
runs exact cosine ranking. The graph consumer persists entity nodes, directed
relation edges, and independent supporting fact/revision rows, then runs bounded
breadth-first traversal. Neither uses the reference projection's delivered bit.
No dependencies, provider calls, automatic workers or authority mutations are
introduced. Existing MemoryAgent and scoped research APIs are unchanged.

## Offline example

The host authenticates the caller, issues a ScopeContext and supplies the trusted
source and consumers. Keep these objects in host code, never accept them from a
request payload. This example needs only the existing core installation:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from jitmind.scope import ScopeAuthority
from jitmind.storage import IngestRequest, Proposal, SQLiteDurableStore
from jitmind.projections import (
    ProjectionCoordinator, SQLiteProjectionSource,
    SQLiteVectorProjection, SQLiteGraphProjection,
)

with TemporaryDirectory() as directory:
    root = Path(directory)
    authority = ScopeAuthority()
    authority.grant("demo-host", "demo", [])
    scope = authority.context("demo-host", "demo", "request-1")
    store = SQLiteDurableStore(root / "authority.db")
    source = SQLiteProjectionSource(store, source_id="demo-authority")
    vector = SQLiteVectorProjection(root / "vector.db", source_id=source.source_id)
    graph = SQLiteGraphProjection(root / "graph.db", source_id=source.source_id,
                                  trusted_relations=True)
    for key, content, subject, obj in (
        ("tea", "tea lemon tea", "app", "cache"),
        ("coffee", "coffee espresso", "cache", "database"),
    ):
        request = IngestRequest.create("demo", key, content, {"relations": [
            {"subject": subject, "predicate": "depends_on", "object": obj}
        ]})
        store.ingest(request, lambda _, text=content: Proposal(
            abstract=text, header=text, decorated=text,
            decision={"operation": "add"},
        ))
    coordinator = ProjectionCoordinator(authority, source, (vector, graph))
    statuses = coordinator.drain(scope)
    assert all(status["status"] == "complete" for status in statuses.values())
    ranked = vector.search(authority, scope, source, "tea lemon")
    assert ranked.hits[0].content == "tea lemon tea"
    reached = graph.traverse(authority, scope, source, "app", depth=2)
    assert reached.nodes == ("app", "cache", "database")
```

`FeatureHashingEmbedder(dimensions=256)` is **lexical feature hashing**, not a
learned semantic model. It hashes casefolded Unicode word tokens with SHA-256,
counts features, and normalizes them to L2 length. Hash collisions are possible.
A zero-token input produces a zero vector and zero cosine scores. Scores tie by
fact ID. This implementation deliberately scans a bounded authorized candidate
population; it is not an approximate nearest-neighbor service or scale benchmark.

## Request APIs and authorization

- `vector.search(authority, scope, source, query, *, snapshots=(), limit=10,
  candidate_limit=1000) -> QueryResult`
- `graph.traverse(authority, scope, source, start, *, snapshots=(), depth=2,
  edge_limit=100, visited_limit=100, candidate_limit=1000) -> QueryResult`
- `consumer.status(authority, scope, source, *, snapshots=()) -> dict`

`QueryResult` is immutable: `status`, `reasons`, `hits`, `edges`, `nodes`.
`ProjectionHit` contains `fact_id`, `score`, authoritative `content`, canonical
`metadata_json`, source `revision`, `repo_id` and `snapshot_id`. A `GraphEdge`
has `subject`, `predicate`, `object`, and a tuple of supporting `ProjectionHit`s.
Current source page ID, validity fields and fact metadata are included in
metadata_json. No projection stores or serves cached fact bodies. C01 can map
these IDs/scores and metadata into its fusion protocol without a competing
research facade. Partial results must retain their reasons in the host response.

Every call requires the exact live, issued ScopeContext. Forged or revoked
contexts raise ScopeDenied. ScopeAuthority is unchanged. The host selects
`snapshots=((repo_id, snapshot_id), ...)`; all selected repositories must be
in the token. Facts without repository restrictions remain namespace scoped.
Fact/page metadata restrictions must agree. Repo facts require both repo and
snapshot to match the explicit selection; omitted, conflicting or unselected
provenance is excluded. SQL applies those conditions before payloads are returned
to Python. Metadata predicates necessarily inspect persisted restriction fields.
Inactive facts return metadata-only state; future and expired facts do not rank
or support expansion. Current fact validity takes precedence over the original
`t_valid`/`t_invalid` observation retained in its immutable page, including C03
validity corrections. Independent fact and page `expires_at` restrictions still
apply. When present, `ttl_seconds` must be a finite nonnegative number (booleans
are invalid); its origin is fact `t_created`, not `t_valid`. Missing TTL is
unrestricted, zero is immediately expired. Malformed eligibility excludes that
fact and reports `unknown_eligibility` as partial; invalid stored JSON/records
fail unavailable. Historical observations and source pages are never rewritten. Future active facts may already have materialized vectors
and edges, so becoming time-eligible does not require a new event.

Readers acquire bounded current authorized facts before embedding, ranking,
or graph expansion. They select index entries/supports only by those authorized
IDs and source revisions/content fingerprints. They reread source authority,
consumer generation, and token validity immediately before returning. A changed
source, concurrent purge or disable returns no positive results. These are
validated read observations, not locks covering an application's later use.
Readers never deliver events, repair, rebuild, or acknowledge anything.

## Delivery, identities and recovery

Host worker construction:

```python
ProjectionCoordinator(authority, source, consumers)
coordinator.deliver(scope, event, snapshots=())
coordinator.drain(scope, snapshots=(), limit=100)
coordinator.rebuild(scope, snapshots=(), limit=1000)
```

Each returns a mapping keyed by consumer database path, so two consumers of the
same kind remain distinguishable. Failures do not acknowledge another consumer.
The equivalent `consumer.deliver(authority, scope, source, event, ...)` and
`consumer.rebuild(...)` methods raise sanitized ProjectionError codes.

`SQLiteProjectionSource(store, *, source_id)` accepts the actual
SQLiteDurableStore. It validates the actual installed authority schema through
its strict validator on read-only connections. The base v1 adapter reads all
outbox events regardless of reference state. With C03 installed, it additionally
cross-checks the public `outbox_events` and `get_event` results against the same
validated read-only authority. It never calls `pending_events`, changes schema,
or falls back to a network service. C03's public event value is normalized to the
projection package's frozen ProjectionEvent with identical fields.

The host-assigned `source_id` must remain stable for one authority lineage; do
not reuse it for an unrelated database. Exact event ID/revision/fact-ID identity
is revalidated before processing, not accepted on the strength of source_id.
Each consumer has a distinct SQLite application ID and exact v1 schema plus a
persisted model/graph configuration identity. Existing different schemas,
models, dimensions, versions or normalization identities are rejected.

Each namespace state records worker selection, rebuild baseline, contiguous
watermark, highest observed authority revision, and immutable authority event
anchor. Each event receipt records the complete event identity, revision,
`done`/`retry`, attempt count and sanitized failure code. Failure before a fact
transaction persists retry when storage is available. Failure after commit can
be reconciled using the same event. Physical process exit before commit leaves
no acknowledgement; after commit the complete receipt survives. If the failure
also prevents recording retry, the gap remains unacknowledged and the worker
must retry after storage recovery.

Embedding and relation preparation happen outside write transactions. Before
source acquisition, delivery/rebuild capture the existing consumer generation,
enabled state and exact namespace selection. Each fact/receipt/rebuild write
checks that fence inside `BEGIN IMMEDIATE`; disable/re-enable, intervening
rebuild (including the same selection), purge or another writer forces retry of
the whole operation. Obsolete failure handlers cannot overwrite a newer done
receipt or rebuild baseline. The source is revalidated afterward and again after
commit. Administrative generation increments also hold the write transaction.
The schema remains v1; deploy with old writers quiesced because schema validation
cannot distinguish their weaker implementation. Explicit rebuild while already
disabled remains supported; changing enabled state during it rejects the work. Per-fact revisions are
monotonic. Deleted, superseded and purged fact IDs cannot become active again,
including when an old event is delivered late. A source revision rollback or
missing/changed anchor is unavailable. Explicit rebuild can reconcile an
advanced/divergent source while retaining tombstones; it cannot lower observed
revision. Never overwrite an authority database with an older backup.

Out-of-order done receipts do not cross missing or retry revisions. An event gap
reports `event_gap`, with `consumer_lag`; it never claims delivery completion.
A complete bounded authoritative rebuild establishes a new baseline even when
older events have been compacted. Truncated snapshots fail rebuild. Rebuild
requires an explicitly selected scope and preserves retention/deletion markers
and prior event receipt history.
One consumer namespace binds one worker repo/snapshot selection. Changing that
selection requires rebuild; reader selections may be narrower. Use separate
consumer databases for independent worker selections. Delivery status describes
that configured selection, not global consumer or namespace coverage.

## Trusted plugins and graph assertions

An optional trusted embedder has `identity: EmbeddingIdentity(model, version,
dimensions, normalization="l2")` and `embed(text) -> list[float]`. Callbacks must
be synchronous, bounded and side-effect controlled by the host. They are not
sandboxed or preempted if they hang. In-process arbitrary Python plugins cannot
provide an enforceable callback deadline; isolate untrusted implementations
outside this API. This package makes no remote calls itself. Learned/remote
embedders are explicitly opt-in and were not invoked in verification.
Dimension drift, booleans, nonfinite components, changed identity and callback
exceptions are rejected. Stored components are normalized finite floats.

Graph ingestion requires `trusted_relations=True` for nonempty `relations` fact
metadata. The host must establish that metadata's trust before ingestion; the
flag is not a content authenticator. Each relation is exactly
`{subject: str, predicate: str, object: str}`. No relationship is inferred from
prose. Shared edges have separate fact/revision supports: removing one fact
removes only its supports, then orphan edges/nodes. Traversal is directed BFS.

## Budgets and outcomes

| Boundary | Limit |
| --- | --- |
| Source snapshot / event batch | 1..1000 rows; 8 MiB aggregate payload/event bytes |
| Query or individual embedded body | 128 KiB UTF-8 JSON envelope |
| Embedding dimensions | 1..4096 |
| Vector result count | 1..100 |
| Relations per fact | 100 |
| Repo/snapshot selections | 100 |
| Graph depth / edges / visited | 0..10 / 1..1000 / 1..1000 |
| Graph support expansion/output | 1000 support rows / 8 MiB content+metadata |
| SQLite connection work | 10 million VM instructions or 3 seconds |
| Backup | 10 seconds, 128 pages per step |

SQLite VM/time budgets do not preempt Python callbacks or individual system
calls. Source payload byte sizes are checked in SQL before acquisition. Each query
acquires at most `candidate_limit` payloads for its initial observation and the
same bounded population for final validation. Status reuses the initial snapshot;
truncation uses a payload-free existence probe, never an extra payload row.
Metadata predicates and byte-size checks still examine persisted restriction
fields under the SQLite processing budget. Local
DELETE journal and FULL synchronous settings are required; busy timeout is
250 ms. Failure rolls back the consumer transaction. No WAL qualification,
remote filesystem, multiple-host or physical power-loss guarantee is made.

Complete means no known omission within the selected bounded read/delivery
contract. `partial` retains safe results with reasons such as `consumer_lag`,
`event_gap`, `not_initialized`, `coverage_gap`, `snapshot_budget`, `depth_budget`,
`edge_budget`, `visited_budget`, `support_budget`, `unknown_eligibility`, or `output_budget`.
`unavailable` contains no hits/edges after errors such as `authority_changed`,
`authority_rollback`, `authority_changed_rebuild_required`, `consumer_changed`,
`consumer_disabled`, `selection_requires_rebuild`, `embedding_failed`, or
`storage_unavailable`. Invalid identifiers/arguments are rejected; scope errors
remain ScopeDenied and are not treated as empty success.

## Host administration, retention and operations

Do not expose these as caller-controlled request RPCs:

- `admin_disable(disabled=True)` durably disables delivery/query use; explicit
  re-enable uses `False`. Rebuild remains an explicit maintenance operation.
- `admin_plan_purge(namespace_id, fact_id=None) -> PurgePlan` returns exact
  namespace/subject revisions and a fingerprint; max 1000 subjects.
- `admin_purge(plan)` revalidates exact subjects inside a short transaction,
  removes vectors/graph supports, and retains metadata-only permanent markers.
  `fact_id=None` suppresses the entire namespace, including later facts. The
  internal namespace marker uses the empty string, which cannot be a fact ID.
  A literal `"*"` is an ordinary exact fact subject. Plans are host maintenance
  values, not bearer capabilities; administrative possession supplies authority.
- `diagnostics()` reports kind/schema/source/model and SQLite settings.
- `export(namespace_id, limit=100)` exports bounded ledger/fact/tombstone metadata
  with explicit per-table truncation. It does not export source bodies. Use the
  SQLite backup for complete vectors/adjacency/receipts.
- `backup(destination)` exclusively creates a new destination.
  `SQLiteVectorProjection.restore(backup, destination, source_id=..., embedder=...)`
  and `SQLiteGraphProjection.restore(..., source_id=..., trusted_relations=True)`
  verify configuration and restore only into a new destination. A failed copy
  is retained for inspection; it is not silently erased or reused.

C07 can invoke the exact host maintenance protocol after approving its retention
plan. Nothing auto-purges runtime data, logs, source history or backups. Backups
made after purge retain the markers. An older backup cannot know later purge
requests: before serving such a restore the host must replay the current C07
retention plan. C02 cannot attest that an external retention record is complete.
Authority revalidation still protects current source deletions independently.
C09 can disable, inspect metadata, back up, restore to a new location and rebuild.

## Verification boundaries

Tests use actual candidate storage files, actual cosine/BFS implementations,
trusted deterministic embedding, process exits and simulated ENOSPC callbacks.
The C03 composition is an identified temporary read-only code copy outside the
final diff. No donor code was copied; the existing Graft donor audit/notices
remain unchanged. This consumer does not require a code parser. Full-suite parser
results, optional-provider skips, exact commands/counts and final composition
manifest are recorded in `docs/closure/C02-interfaces.md`. No learned model
quality, production readiness, complete ZIP scope, or all-platform claim follows
from these tests. Mypy is unavailable; no typecheck is claimed.

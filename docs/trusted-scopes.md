# Trusted scopes and strict temporal queries

This opt-in API adds a host-issued authorization boundary to retrieval. It does
not change `ResearchAgent`, `MemoryAgent`, legacy store formats, or legacy
`query_as_of`. Plain legacy public APIs remain single-user APIs; passing a
`user_id` string to them is not a multi-tenant authentication guarantee.

Namespace isolation PR **#27 is open, not merged** into this baseline. Its
component namespaces and mutation isolation are separate work. This change does
not reproduce that patch or modify its branch. `jitmind.scope` uses an **opaque,
exact string** namespace. A singleton PR #27 component can map to that string;
multiple components need an explicit, injective host mapping. Do not join them
with delimiters or use a basename as an identity.

## Runnable local example

Run from the repository with `PYTHONPATH=$PWD`. No provider, network, or additional
package is needed for this example.

```python
from jitmind.scope import ScopeAuthority, ScopeDenied
from jitmind.schemas import InMemoryPageStore, Page
from jitmind.scoped_research import (
    LegacyScopedBackend, LegacyStoreBinding, ScopedResearchFacade,
)

authority = ScopeAuthority()
# Trusted host administration, after authenticating the principal:
authority.grant("alice", "workspace-A", ["repo-stable-id"])
scope = authority.context("alice", "workspace-A", "request-123")

pages = InMemoryPageStore()
pages.add(Page(header="Preference", content="Alice prefers concise updates"))
backend = LegacyScopedBackend((LegacyStoreBinding("workspace-A", pages),))
research = ScopedResearchFacade(authority, backend)
response = research.research(scope, "updates")
print(response.serialize()["integrated_memory"])

authority.revoke("alice", "workspace-A")
try:
    response.serialize()
except ScopeDenied:
    print("Response denied after revocation")
```

`grant` and `revoke` are trusted administrative operations. `context` is only
issued for an existing grant; the host must authenticate the caller before
choosing principal and namespace. Never expose an unrestricted context issuance
endpoint or authority instance to an untrusted plugin. The implementation has no
global remote identity service. IDs alone do not authenticate anyone.

`ScopeContext` is a frozen dataclass with exactly these fields:

```text
principal_id: str
namespace_id: str
authorized_repo_ids: tuple[str, ...]
authorization_version: int
request_id: str
```

The authority checks the identity and original claims of its issued context,
current authoritative grant, version, and optional repository membership.
Constructed copies and deserialized lookalikes are rejected. Contexts are
process-local capabilities. Restarting the authority requires issuing new
contexts. Grants/revocations advance versions; regranting identical repository
IDs does not revive old contexts. Identifiers reject empty/whitespace-padded,
control/NUL, malformed Unicode, non-string, and overlong values. Repository
lists are bounded to 1,024 entries, reject duplicates, and do not consume
arbitrary iterators. Revisions reject booleans and non-positive integers.

## Retrieval boundary

`ScopedResearchFacade` performs one pass of isolated retrieval, reciprocal-rank
ordering across channels, optional reranking, and optional generation. It returns
`ScopedResponse`; call `serialize()` at the delivery boundary. No model runs by
default. A supplied generator implements `generate_single(prompt=..., schema=...)`
and a supplied reranker implements `rerank(query, documents, top_k=...)`.
Only authorized canonical source text reaches those calls. These injected Python
objects are trusted host components and must not log provider error payloads.

Authorization precedes storage reads, index construction, cache access, search,
source expansion, reranking, prompts, and response delivery. It is checked again
after potentially long operations. A midflight revocation drops the response;
it never invokes an unscoped fallback. Already-delivered data cannot be recalled.
Checking permission and returning Python data is not a distributed atomic
revocation protocol; callers must use `serialize()` immediately before delivery.

Each request has its own frozen view and fresh index. There is no shared
`current_user` or `current_namespace`. The bounded cache stores canonical source
IDs and finite scores, not snippets or generated answers. Keys include namespace,
principal, authorization version, view snapshot, explicit repository snapshots,
channel, query, and limit. Authorization is checked before every reuse. Old
versions are inaccessible immediately and evicted on regrant use; an in-flight
denial clears the facade cache. This is invalidation, not a secure-memory-erasure
guarantee. Every cached/derived ID must resolve inside the current view before
fusion or prompt construction.

Channel numbers are the positions in the configured backend tuple (primary is
0, optional channels start at 1). An optional outage leaves a gap; it never
renumbers surviving channels or reuses the failed channel's cache identity.
Treat backend configuration as fixed for a facade instance.

Evidence limits apply to the selected canonical hits **before reranking or
generation**, including cache hits and generator-only requests. Constructor
options `max_document_bytes=16384` and `max_evidence_bytes=65536` bound each
snippet and the complete evidence block respectively, measured in strict UTF-8
bytes. The aggregate includes `[channel:document_id] ` labels and newline
separators. Exact limits are accepted; one byte over either limit raises generic
`ScopedBackendError` before either model hook runs. The entire request fails:
no truncation, dropped hits, or partial response. Limits accept positive integers
up to 16,777,216 (booleans are rejected). Multibyte characters count all their
bytes; malformed Unicode fails closed and byte strings are not valid document
content. The existing 100,000-character question limit is separate (at most
400,000 UTF-8 bytes for valid Unicode). These are model-evidence limits, not a
bound on storage acquisition, index size, metadata, or generated output.

The facade deliberately does not reuse legacy checkpoints: they are keyed by
sanitized caller IDs and do not carry this authority/version/snapshot contract.
It also does not enable ambient graph providers, profiles, tools, replay, or
reflection writes. Do not pass legacy retrievers as scoped index factories.
There is no blanket claim that arbitrary legacy retrieval plugins are safe.

### Supported store adapters

`LegacyScopedBackend` accepts exact `InMemoryPageStore`, `InMemoryMemoryStore`,
and `AdvancedMemoryStore` types. Its trusted binding assigns an entire store
pair to one namespace. **Untagged old-format data belongs to that host assignment**;
this cannot discover ownership inside an old mixed-user untagged file. Do not
bind such a file. Reusing the same store object or canonical backing directory
across namespace bindings is rejected. Tagged foreign records are excluded
before index construction. Source pages attached to inactive, expired, foreign,
or wrong-snapshot memories are excluded too. Independent unlinked pages still
need their own validity/expiry metadata. TTL eligibility does not purge history.

For `AdvancedMemoryStore`, both `source_page_id` and page `memory_id` links are
resolved before either abstracts or pages enter the index. All linked records
must agree on every explicit repository/snapshot value; missing or null values
may inherit. This includes shared pages and multiple reverse links. A foreign
namespace anywhere in a linked component excludes that component. Conflicting
repository/snapshot values or dangling explicit links raise `ScopeDenied`.
This protects the original `MemoryAgent.memorize()` format, which puts caller
repository metadata only on the page. Untagged linked data remains available in
its host-assigned namespace. No stored metadata is rewritten. Old abstract-only
`InMemoryMemoryStore` records have no source-link contract; their ownership
continues to come from the whole-store host binding.

A view contains detached immutable strings, not mutable store objects. Its
content digest changes on same-count edits. Validation rereads authorized state
and denies stale views. These correctness checks trade throughput for simplicity;
no performance claim is made. Mutable legacy stores cannot provide a transactional
cross-file snapshot, so concurrent detected edits fail closed.

`DurableScopedBackend(store, limit=1000)` integrates with J02's
`SQLiteDurableStore` through its existing public APIs:

| API | Use |
| --- | --- |
| `snapshot(namespace_id, limit=...)` | Namespace-qualified active facts and authoritative revision |
| `get_page(namespace_id, page_id)` | Source expansion only for facts in that snapshot |
| `status(namespace_id)["revision"]` | Detect mutation during reads and before delivery |

The adapter reads authoritative facts, not eventual projections. Truncation
fails closed instead of silently returning a misleading partial index. The
maximum is currently 1,000 facts. It is optional: importing the scoped facade
does not require J02 storage. Constructing the adapter without that package
raises generic `ScopedBackendError`. The actual J02 SQLite implementation was
exercised from its dependency worktree without modifying that worktree.

Durable fact and source-page provenance is reconciled before either document
is indexed. Explicit conflicts are denied, never overwritten with fact tags.
Missing values inherit in either direction, so a page-only repository restriction
also protects its untagged abstract. Page `expires_at`, `t_valid`, and `t_invalid`
are checked independently of fact eligibility; an expired or not-yet-valid page
is excluded while an otherwise eligible fact may remain. Historical rows are
retained, and validation rereads eligibility at delivery as well as revision.

### J04/J05 extension protocol

Trusted backends must implement `ScopedBackend`, declare
`scope_protocol = "jitmind.scoped.v1"`, and provide:

```text
open_view(authority, scope, snapshots: tuple[(repo_id, snapshot_id), ...]) -> ScopedView
validate_view(authority, scope, view) -> None
```

Both operations must call `authority.require(scope, repo_id)` before any code
or disk access and again before returning. `open_view` must select the namespace
and exact repositories/snapshots **before** candidates, graph traversal, scores,
embeddings, or provider input are constructed. A marker is an explicit capability
contract, not a sandbox for malicious Python. Unsupported unscoped backends and
custom legacy-store subclasses are rejected without calling their load/search.

Return `ScopedView(namespace_id, snapshot_id, documents, repo_snapshots)` with a
tuple of frozen `ScopedDocument(document_id, namespace_id, content, source,
repo_id, snapshot_id, metadata_json)` records. Document IDs must be unique within
a channel; response IDs are channel-qualified. Metadata is a JSON object. Source
digest, validation metadata and repository snapshot survive fusion in hit metadata.
The facade verifies namespace, repository grant and exact snapshot membership.
It does not trust an arbitrary hit's snippet or look it up in a global store.

Pass repository selection explicitly:

```python
response = research.research(
    scope, "retry policy", snapshots=(("repo-stable-id", "commit-or-snapshot-id"),)
)
```

Omitting selection excludes repository-backed records. It never resolves an
ambient default branch. Changing a branch/revision changes cache identity.
`validate_view` must recheck root registration and snapshot validity, including
cached/derived references. Stable opaque repository IDs must distinguish roots
with identical basenames. This module never opens repository paths. J04 owns
canonical root containment, hostile path, symlink, and sibling-prefix checks;
those filesystem checks require J04's real adapter tests, not these in-memory ID
tests.

`ScopedIndexFactory.build_scoped(view, authority, scope)` must create an isolated
request index and return an object with `search(query, top_k)` returning a bounded
list/tuple of `(document_id, finite_score)` pairs. It must use only the supplied
view. Persistent indexes need namespace/principal/version/snapshot keys. A graph
extension must isolate all traversed nodes/edges before expansion. No legacy
graph retriever is automatically enabled.

Optional backends can be supplied through `optional_backends=(code_backend,)`.
An unavailable optional adapter leaves only already-authorized primary memory
views. Authorization errors terminate the entire request. Other plugin failures
are exposed as generic errors without including their exception text. The
facade itself does not log content; trusted plugins remain responsible for their
own logging behavior.

## Strict temporal oracle

Legacy `AdvancedMemoryStore.query_as_of` still accepts naive query datetimes as
UTC and retains its existing history behavior. Its update path changes old
`t_invalid`, so it cannot recover every earlier knowledge perspective after a
retroactive correction. This change does not silently reinterpret that API.

`StrictTemporalOracle` is a separate API with timezone-aware UTC-normalized
inputs. It stores complete immutable `TemporalPerspective` snapshots. Publishing
a correction appends a new perspective; it does not mutate effective intervals
in earlier perspectives. Valid time is `[valid_from, valid_to)` and transaction
time is `[recorded_at, next_recorded_at)`. No transaction time means the latest
perspective. Single-valued fact keys reject overlapping intervals at publication,
including inconsistent single/multi-value declarations. Multi-valued facts
require explicit `single_valued=False`.

```python
from jitmind.scope import ScopeAuthority
from jitmind.scoped_temporal import (
    InMemoryTemporalHistory, StrictTemporalOracle, TemporalFact, TemporalPerspective,
)

authority = ScopeAuthority()
authority.grant("alice", "settings", [])
scope = authority.context("alice", "settings", "temporal-request")
history = InMemoryTemporalHistory()
oracle = StrictTemporalOracle(authority, history)
oracle.publish(scope, TemporalPerspective("2026-01-01T00:00:00Z", (
    TemporalFact("a", "retry_limit", "3", "2026-01-01T00:00:00Z"),
)))
assert oracle.query_as_of(scope, "2026-02-10T00:00:00Z")[0].content == "3"
```

`publish` accepts the **complete** new namespace perspective, not a delta. Empty
perspectives represent deletion of all current facts while retaining history.
`query_as_of(..., eligible_at=...)` optionally applies TTL; normal historical
queries ignore expiry eligibility. `expires_at` never deletes a perspective.
Repository facts require explicit `snapshots` selection, just as retrieval does.

The reviewed golden fixture is encoded in `test_scoped_temporal.py`:

| Valid time | Transaction time | retry_limit |
| --- | --- | --- |
| Jan 20 | Jan 20 | 3 |
| Jan 20 | Feb 15 | 3 |
| Jan 20 | Mar 1 | 4 |
| Jan 10 | Mar 2 | 3 |
| Jan 15 | Mar 2 | 4 |
| Feb 1 | Mar 2 | 5 |
| Feb 10 | Jan 20 | 3 |

All dates are 2026 UTC. Latest-perspective Jan 20 returns 4, including an
equivalent `+05:30` instant. Boundaries and ambiguous overlaps have separate tests.

### Durable temporal followup

`TemporalHistoryBackend` specifies namespace-qualified
`read_perspectives(namespace_id) -> tuple[TemporalPerspective, ...]` and atomic
`append_perspective(namespace_id, expected_revision, perspective)`. Expected
revision is the count of retained perspectives. Append must compare that count,
reject non-increasing transaction times, and retain prior intervals unchanged.
The host supplies this trusted backend; the oracle checks authority before and
after backend access. `InMemoryTemporalHistory` implements this contract today.

J02's currently available durable APIs do not implement this history contract.
Its mutable fact payloads and ID-only outbox cannot reconstruct earlier effective
intervals after correction. Durable strict temporal persistence therefore needs
an append-only perspective/event API in J02, atomically coupled to its commit.
No SQL schema or private durable internals are modified here, and durable temporal
integration is not claimed. The SQLite retrieval adapter above is implemented
and independently tested.

The optional J09 peer now exposes `SQLiteTemporalHistory.read_perspectives` and
`append_perspective`, plus a receipt-returning `DurableTemporalOracle.publish`.
Those APIs were inspected read-only during remediation. They are not copied,
required, or tested here. Supervising integration should verify that bridge
separately; this patch leaves `ScopeAuthority` and the strict temporal API intact.

## Verification scope

Tests inspect index build inputs, reranker documents and generator prompts,
using deterministic local components and real legacy stores. They cover forged
contexts, revocation before access/during build/during generation/before delivery,
regrant cache separation, concurrent namespaces, stale source IDs, metadata
preservation, explicit branch snapshots, TTL/source-page eligibility, strict
intervals and unchanged legacy naive-query behavior. SQLite integration tests
run when J02 storage is importable; standalone runs explicitly skip that module.
No live providers, model-quality evaluation, production-readiness assertion, or
benchmark claim is part of this verification.

### Review remediation verification (2026-09-29)

The added regressions exercise all five findings using the actual legacy writer,
real J02 staged migration and SQLite commits, and recording local model hooks.
Before production edits, the expanded retrieval/durable target run produced
**19 failed, 27 passed**. After remediation that run produced **46 passed**;
additional boundary cases were then added. Existing test oracles and both
review probe scripts were left unchanged.

Commands below run from J03 with the prescribed interpreter. J02 modules are
loaded read-only via `jitmind.__path__`; **no dependency files were copied**.

```sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
PY=/Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python

"$PY" -m pytest -q -p no:cacheprovider
# 127 passed, 1 skipped (optional durable module), 3 expected provider warnings.

"$PY" -c 'import jitmind; jitmind.__path__.append("/Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/worktrees/J02/jitmind"); import pytest; raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider"]))'
# Full actual J03 suite with real J02 backend: 141 passed, no skips.
# 3 optional-provider import warnings; 1 pytest already-imported-plugin warning.

"$PY" -c 'import jitmind; jitmind.__path__.append("/Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/worktrees/J02/jitmind"); import pytest; raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "tests/test_scoped_research.py", "tests/test_scoped_durable.py", "--tb=short"]))'
# Initial regression run: 19 failed, 27 passed; after fixes: 46 passed.
# Final full-suite counts above include the subsequent 17 boundary cases.

"$PY" /private/tmp/j03_review_probes.py
# Exit 1 at line 22: its assertion that REPO_SECRET leaks is now false.
"$PY" /private/tmp/j03_review_integration_probes.py
# Exit 1 at line 24: its assertion that RESTRICTED_ABSTRACT leaks is now false.
# Both scripts stop at their first old bug assertion; later script statements
# do not run. New pytest regressions cover their remaining findings and controls,
# including conflicting snapshots, page expiry, channel cache reuse, the 700k
# document, NOOP/DELETE visibility, and mutation between snapshot and page read.

"$PY" -m ruff check --no-cache jitmind/scope.py jitmind/scoped_research.py jitmind/scoped_temporal.py tests/test_scoped_authority.py tests/test_scoped_durable.py tests/test_scoped_research.py tests/test_scoped_temporal.py
# All checks passed.
"$PY" -m ruff format --check --no-cache jitmind/scope.py jitmind/scoped_research.py jitmind/scoped_temporal.py tests/test_scoped_authority.py tests/test_scoped_durable.py tests/test_scoped_research.py tests/test_scoped_temporal.py
# 7 files already formatted.
"$PY" -m compileall -q jitmind/scope.py jitmind/scoped_research.py jitmind/scoped_temporal.py tests/test_scoped_authority.py tests/test_scoped_durable.py tests/test_scoped_research.py tests/test_scoped_temporal.py
# Passed.
"$PY" -m mypy jitmind/scope.py jitmind/scoped_research.py jitmind/scoped_temporal.py
# Exit 1: No module named mypy. No packages installed.
```

Remediation changes only `jitmind/scoped_research.py`,
`tests/test_scoped_research.py`, `tests/test_scoped_durable.py`, and this document.
The shared authority and temporal contract are unchanged. Provenance handling
now uses one shared reconciliation rule instead of overwriting page tags.
No live provider calls, credentials, installs, commits, or peer-worktree writes
were used. The evidence limits do not address whole-corpus acquisition/index
resource bounds or generated-output size. Mutable legacy stores still lack
transactional cross-file snapshots. J09 integration and J04 filesystem-boundary
verification remain separate supervising followups.
